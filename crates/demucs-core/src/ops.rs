//! Tensor kernels shared by every model forward.
//!
//! These are the operations the eventual wgpu backend must provide, so they are
//! kept in one place with the PyTorch semantics they mirror spelled out. The CPU
//! implementations here double as the numerical reference for the GPU kernels.

use ndarray::{Array1, Array2, Array3, ArrayView1, Axis};
use ndarray::{ArrayBase, Data, DataMut, Ix2, Ix3};
use rayon::prelude::*;

use crate::error::{Error, Result};

/// A fully-connected layer stored pre-transposed, so the hot path is a plain
/// `(m, k) x (k, n)` matrix multiply with both operands contiguous.
#[derive(Debug, Clone)]
pub struct Linear {
    /// `(in_features, out_features)`
    pub weight_t: Array2<f32>,
    pub bias: Option<Array1<f32>>,
}

impl Linear {
    /// `w` is the PyTorch layout, `(out_features, in_features)`.
    pub fn new(w: ndarray::ArrayView2<f32>, bias: Option<ArrayView1<f32>>) -> Self {
        Self {
            weight_t: w.t().to_owned(),
            bias: bias.map(|b| b.to_owned()),
        }
    }

    pub fn in_features(&self) -> usize {
        self.weight_t.dim().0
    }

    pub fn out_features(&self) -> usize {
        self.weight_t.dim().1
    }

    /// `x` is `(rows, in_features)`; returns `(rows, out_features)`.
    ///
    /// Delegates to the `demucs` GEMM so both share one backend (and one A/B
    /// switch). The bias is folded into the output write-back pass: a second
    /// full sweep over the same memory would be the attention's whole linear
    /// path paying twice for bandwidth.
    pub fn forward<S: Data<Elem = f32> + Sync>(&self, x: &ArrayBase<S, Ix2>) -> Array2<f32> {
        let (_rows, k) = x.dim();
        debug_assert_eq!(k, self.in_features());
        crate::demucs::ops::linear_forward(x, &self.weight_t, self.bias.as_ref())
    }

    /// Same as [`Linear::forward`] for a `(batch, rows, in_features)` input.
    pub fn forward_3d<S: Data<Elem = f32>>(&self, x: &ArrayBase<S, Ix3>) -> Array3<f32> {
        let (b, rows, k) = x.dim();
        let flat = x
            .view()
            .into_shape_with_order((b * rows, k))
            .expect("contiguous view");
        let out = self.forward(&flat);
        out.to_shape((b, rows, self.out_features()))
            .expect("reshape back")
            .to_owned()
    }
}

/// `F.normalize(x, dim=-1) * sqrt(dim) * gamma` — the RMSNorm variant this model
/// uses. Note it normalises by the L2 norm of the feature vector, not by its
/// root-mean-square, and it has no bias term.
pub fn rms_norm<S: Data<Elem = f32>>(
    x: &ArrayBase<S, Ix2>,
    gamma: &ArrayView1<f32>,
) -> Result<Array2<f32>> {
    let (rows, dim) = x.dim();
    if gamma.len() != dim {
        return Err(Error::Shape(format!(
            "rms_norm gamma has {} entries but the feature dim is {dim}",
            gamma.len()
        )));
    }
    let scale = (dim as f32).sqrt();
    let mut out = Array2::<f32>::zeros((rows, dim));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(x.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut out_row, x_row)| {
            let norm = x_row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            let factor = scale / norm;
            for i in 0..dim {
                out_row[i] = x_row[i] * factor * gamma[i];
            }
        });
    Ok(out)
}

/// Exact (erf-based) GELU, as `nn.GELU()` uses by default.
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26 is not accurate enough for 1e-4 alignment at
    // large |x|, so use the complementary form with the standard rational
    // approximation of erfc on the tail.
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let tau = t
        * (-z * z - 1.265_512_2
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    let value = 1.0 - tau;
    if x >= 0.0 {
        value
    } else {
        -value
    }
}

/// Numerically stable softmax over the last axis.
pub fn softmax_in_place(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// Precomputed rotary cos/sin tables for one sequence length.
///
/// `rotary_embedding_torch` computes `angle = position * freqs[k]` in float32 and
/// caches it per sequence length, which is what this mirrors. `rotate_half` pairs
/// *adjacent* (even, odd) elements, so the rotation is the interleaved variant.
pub struct RopeTable {
    cos: Vec<f32>,
    sin: Vec<f32>,
    seq: usize,
    dim: usize,
}

impl RopeTable {
    pub fn new(seq: usize, freqs: &[f32]) -> Self {
        let dim = freqs.len() * 2;
        let mut cos = vec![0.0f32; seq * dim];
        let mut sin = vec![0.0f32; seq * dim];
        for p in 0..seq {
            for k in 0..freqs.len() {
                let angle = p as f32 * freqs[k];
                let c = angle.cos();
                let s = angle.sin();
                cos[p * dim + 2 * k] = c;
                cos[p * dim + 2 * k + 1] = c;
                sin[p * dim + 2 * k] = s;
                sin[p * dim + 2 * k + 1] = s;
            }
        }
        Self { cos, sin, seq, dim }
    }

    pub fn seq_len(&self) -> usize {
        self.seq
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// `x` is `(seq, dim)` and each row is rotated by its own position index.
    pub fn apply<S: DataMut<Elem = f32>>(&self, x: &mut ArrayBase<S, Ix2>) {
        let (seq, dim) = x.dim();
        debug_assert_eq!(dim, self.dim);
        debug_assert_eq!(seq, self.seq);
        for p in 0..seq {
            let mut row = x.row_mut(p);
            let slice = row.as_slice_mut().expect("contiguous row");
            let offset = p * dim;
            // rotate_half pairs (even, odd), so the rotated vector is
            // (-x[2k+1], x[2k]) before the angle is applied.
            let mut rotated = vec![0.0f32; dim];
            for k in 0..dim / 2 {
                let a = slice[2 * k];
                let b = slice[2 * k + 1];
                rotated[2 * k] = -b;
                rotated[2 * k + 1] = a;
            }
            for i in 0..dim {
                slice[i] = slice[i] * self.cos[offset + i] + rotated[i] * self.sin[offset + i];
            }
        }
    }
}

/// `nn.GLU(dim=-1)` on a `(rows, 2 * features)` tensor.
pub fn glu<S: Data<Elem = f32>>(x: &ArrayBase<S, Ix2>) -> Result<Array2<f32>> {
    let (rows, width) = x.dim();
    if width % 2 != 0 {
        return Err(Error::Shape(format!(
            "GLU needs an even feature width, found {width}"
        )));
    }
    let half = width / 2;
    let mut out = Array2::<f32>::zeros((rows, half));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(x.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut out_row, x_row)| {
            for i in 0..half {
                let value = x_row[i];
                let gate = x_row[half + i];
                out_row[i] = value * (1.0 / (1.0 + (-gate).exp()));
            }
        });
    Ok(out)
}

/// `(rows, in) -> (in, rows)`, used where einops patterns reorder axes.
pub fn transpose_owned<S: Data<Elem = f32>>(x: &ArrayBase<S, Ix2>) -> Array2<f32> {
    x.t().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn linear_adds_bias_and_matches_manual_matmul() {
        let w = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![0.5f32, -0.5];
        let layer = Linear::new(w.view(), Some(b.view()));
        let x = array![[1.0f32, 1.0], [2.0, 0.0]];
        let out = layer.forward(&x.view());
        assert!(close(out[[0, 0]], 1.0 * 1.0 + 2.0 * 1.0 + 0.5, 1e-6));
        assert!(close(out[[1, 1]], 2.0 * 3.0 + 0.0 * 4.0 - 0.5, 1e-6));
    }

    #[test]
    fn rms_norm_scales_by_sqrt_of_the_feature_dim() {
        let x = array![[3.0f32, 4.0]]; // L2 norm 5
        let gamma = array![1.0f32, 1.0];
        let out = rms_norm(&x.view(), &gamma.view()).unwrap();
        let scale = (2.0f32).sqrt() / 5.0;
        assert!(close(out[[0, 0]], 3.0 * scale, 1e-6));
        assert!(close(out[[0, 1]], 4.0 * scale, 1e-6));
    }

    #[test]
    fn rms_norm_rejects_a_mismatched_gamma() {
        let x = array![[1.0f32, 2.0, 3.0]];
        let gamma = array![1.0f32, 1.0];
        assert!(rms_norm(&x.view(), &gamma.view()).is_err());
    }

    #[test]
    fn gelu_matches_torch_exact_gelu() {
        // Values from torch.nn.GELU() (the erf form, not the tanh approximation).
        assert!((gelu_erf(0.0) - 0.0).abs() < 1e-7);
        assert!((gelu_erf(1.0) - 0.841_344_71).abs() < 1e-5);
        assert!((gelu_erf(-1.0) - -0.158_655_29).abs() < 1e-5);
        assert!((gelu_erf(2.0) - 1.954_499_96).abs() < 1e-5);
        assert!((gelu_erf(-3.0) - -0.004_049_87).abs() < 1e-6);
        // Large magnitudes must saturate rather than blow up.
        assert!(gelu_erf(20.0) > 19.9);
        assert!(gelu_erf(-20.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one_and_handles_large_offsets() {
        let mut row = vec![1000.0f32, 1000.0, 1000.0];
        softmax_in_place(&mut row);
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((row[0] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn rope_at_position_zero_is_the_identity() {
        let freqs = array![0.5f32, 0.25];
        let table = RopeTable::new(1, freqs.as_slice().unwrap());
        let mut x = array![[1.0f32, 2.0, 3.0, 4.0]];
        let original = x.clone();
        table.apply(&mut x.view_mut());
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_rotates_adjacent_pairs_by_their_position() {
        let freqs = array![std::f32::consts::FRAC_PI_2];
        let table = RopeTable::new(2, freqs.as_slice().unwrap());
        let mut x = array![[1.0f32, 0.0], [1.0f32, 0.0]];
        table.apply(&mut x.view_mut());
        // Position 0 leaves the pair alone; position 1 rotates it by 90 degrees.
        assert!((x[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((x[[0, 1]] - 0.0).abs() < 1e-6);
        assert!((x[[1, 0]] - 0.0).abs() < 1e-6);
        assert!((x[[1, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rope_preserves_the_pair_norm() {
        let freqs = array![0.7f32, 0.13];
        let table = RopeTable::new(5, freqs.as_slice().unwrap());
        let mut x = array![
            [0.3f32, -0.9, 1.7, 0.2],
            [0.1, 0.4, -0.6, 0.8],
            [1.0, 1.0, 1.0, 1.0],
            [-0.2, 0.5, 0.3, -0.1],
            [0.0, 0.0, 0.0, 0.0]
        ];
        let before: f32 = x.iter().map(|v| v * v).sum();
        table.apply(&mut x.view_mut());
        let after: f32 = x.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-5);
    }

    #[test]
    fn glu_gates_the_first_half_with_a_sigmoid_of_the_second() {
        let x = array![[2.0f32, 0.0, 0.0, 0.0]];
        let out = glu(&x.view()).unwrap();
        assert_eq!(out.dim(), (1, 2));
        assert!(close(out[[0, 0]], 2.0 * 0.5, 1e-6));
        assert!(close(out[[0, 1]], 0.0, 1e-6));
    }

    #[test]
    fn glu_rejects_odd_widths() {
        let x = array![[1.0f32, 2.0, 3.0]];
        assert!(glu(&x.view()).is_err());
    }
}
