//! Convolution kernels shared by the conv-based architectures (DTTNet, SCNet,
//! MDX23C, HTDemucs).
//!
//! Both convolutions run as implicit GEMM: `nn.Conv2d` becomes a patch matrix
//! times the weight (`(out_channels, k) @ (k, positions)`), and
//! `nn.ConvTranspose2d` becomes one GEMM per kernel tap followed by a strided
//! accumulate. The patch matrix is built directly in `(k, positions)` order so
//! that the GEMM's output lands channel-major, which is exactly the layout
//! `(b, c, h, w)` wants — no transpose pass.
//!
//! The CPU implementations here are the numerical reference for the wgpu
//! kernels, the same way `ops.rs` is for the elementwise ones.

use ndarray::{Array4, ArrayView2, ArrayViewMut2, Axis, s};
use rayon::prelude::*;

use crate::ckpt::Checkpoint;
use crate::error::{Error, Result};

/// Positions (output pixels) per GEMM call. Large enough for the matmul to
/// amortise, small enough that the patch matrix stays a few megabytes.
const POSITIONS_PER_TILE: usize = 8192;

/// `nn.Conv2d` over the two spatial axes of a `(b, c, h, w)` tensor.
#[derive(Debug, Clone)]
pub struct Conv2d {
    /// PyTorch layout `(out_channels, in_channels, kh, kw)`, contiguous.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    out_channels: usize,
    in_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
}

impl Conv2d {
    pub fn load(
        ckpt: &Checkpoint,
        prefix: &str,
        stride: (usize, usize),
        pad: (usize, usize),
        with_bias: bool,
    ) -> Result<Self> {
        let weight = ckpt.get_required(&format!("{prefix}.weight"))?;
        let shape = weight.shape();
        if shape.len() != 4 {
            return Err(Error::WeightShape {
                name: format!("{prefix}.weight"),
                found: shape.to_vec(),
                expected: vec![0, 0, 0, 0],
            });
        }
        let (out_channels, in_channels, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
        let bias = if with_bias {
            Some(
                ckpt.get_shaped(&format!("{prefix}.bias"), &[out_channels])?
                    .iter()
                    .copied()
                    .collect(),
            )
        } else {
            None
        };
        Ok(Self {
            weight: weight.iter().copied().collect(),
            bias,
            out_channels,
            in_channels,
            kernel: (kh, kw),
            stride,
            pad,
        })
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    pub fn kernel(&self) -> (usize, usize) {
        self.kernel
    }

    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    pub fn pad(&self) -> (usize, usize) {
        self.pad
    }

    pub fn forward(&self, x: &Array4<f32>) -> Array4<f32> {
        let (b, in_channels, h, w) = x.dim();
        debug_assert_eq!(in_channels, self.in_channels);
        let (kh, kw) = self.kernel;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.pad;
        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;
        let k = in_channels * kh * kw;
        let oc = self.out_channels;
        let tile_rows = (POSITIONS_PER_TILE / out_w).max(1);

        let mut out = Array4::<f32>::zeros((b, oc, out_h, out_w));
        let weight = ArrayView2::from_shape((oc, k), &self.weight).expect("weight is contiguous");

        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(bi, mut out_sample)| {
                let input = x.slice(s![bi, .., .., ..]);
                let mut row = 0usize;
                while row < out_h {
                    let rows_here = tile_rows.min(out_h - row);
                    let positions = rows_here * out_w;
                    let mut patches = vec![0.0f32; k * positions];

                    // Patches in `(ic, ky, kx)` order, one row per tap.
                    let mut krow = 0usize;
                    for ic in 0..in_channels {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let dst = &mut patches[krow * positions..(krow + 1) * positions];
                                let mut p = 0usize;
                                for oy in row..row + rows_here {
                                    let iy = oy * sh + ky;
                                    if iy < ph || iy >= ph + h {
                                        p += out_w;
                                        continue;
                                    }
                                    let src = input.slice(s![ic, iy - ph, ..]);
                                    for ox in 0..out_w {
                                        let ix = ox * sw + kx;
                                        dst[p] = if ix < pw || ix >= pw + w {
                                            0.0
                                        } else {
                                            src[ix - pw]
                                        };
                                        p += 1;
                                    }
                                }
                                krow += 1;
                            }
                        }
                    }

                    let patch_view =
                        ArrayView2::from_shape((k, positions), &patches).expect("exact size");
                    let mut product = vec![0.0f32; oc * positions];
                    {
                        let mut out_view = ArrayViewMut2::from_shape((oc, positions), &mut product)
                            .expect("exact size");
                        ndarray::linalg::general_mat_mul(1.0, &weight, &patch_view, 0.0, &mut out_view);
                    }

                    for (channel, row_values) in product.chunks_exact(positions).enumerate() {
                        let bias = self.bias.as_ref().map(|b| b[channel]).unwrap_or(0.0);
                        let mut target = out_sample.slice_mut(s![channel, row..row + rows_here, ..]);
                        for (v, a) in target.iter_mut().zip(row_values.iter()) {
                            *v = *a + bias;
                        }
                    }
                    row += rows_here;
                }
            });
        out
    }
}

/// `nn.ConvTranspose2d` with `kernel == stride` (no padding, no output padding),
/// the only form the band-splitting models use. Each input pixel scatters into a
/// `kh x kw` block, so the op is one GEMM per tap plus a strided accumulate.
#[derive(Debug, Clone)]
pub struct ConvTranspose2d {
    /// One `(out_channels, in_channels)` matrix per `(ky, kx)` tap.
    taps: Vec<Vec<f32>>,
    bias: Option<Vec<f32>>,
    in_channels: usize,
    out_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
}

impl ConvTranspose2d {
    pub fn load(
        ckpt: &Checkpoint,
        prefix: &str,
        stride: (usize, usize),
        with_bias: bool,
    ) -> Result<Self> {
        let weight = ckpt.get_required(&format!("{prefix}.weight"))?;
        let shape = weight.shape();
        if shape.len() != 4 {
            return Err(Error::WeightShape {
                name: format!("{prefix}.weight"),
                found: shape.to_vec(),
                expected: vec![0, 0, 0, 0],
            });
        }
        let (in_channels, out_channels, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
        let flat: Vec<f32> = weight.iter().copied().collect();
        // (ic, oc, kh, kw) -> one contiguous (oc, ic) matrix per tap.
        let mut taps = Vec::with_capacity(kh * kw);
        for ky in 0..kh {
            for kx in 0..kw {
                let mut tap = vec![0.0f32; out_channels * in_channels];
                for ic in 0..in_channels {
                    for oc in 0..out_channels {
                        tap[oc * in_channels + ic] =
                            flat[((ic * out_channels + oc) * kh + ky) * kw + kx];
                    }
                }
                taps.push(tap);
            }
        }
        let bias = if with_bias {
            Some(
                ckpt.get_shaped(&format!("{prefix}.bias"), &[out_channels])?
                    .iter()
                    .copied()
                    .collect(),
            )
        } else {
            None
        };
        Ok(Self {
            taps,
            bias,
            in_channels,
            out_channels,
            kernel: (kh, kw),
            stride,
        })
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// One `(out_channels, in_channels)` matrix per `(ky, kx)` tap, in tap order.
    ///
    /// The device path wants the other packing — rows `(oc, ky, kx)` and columns
    /// `in_channels` — but that is a rearrangement of this one, so it is built by
    /// the loader rather than stored twice here.
    pub fn taps(&self) -> &[Vec<f32>] {
        &self.taps
    }

    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    pub fn kernel(&self) -> (usize, usize) {
        self.kernel
    }

    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    pub fn forward(&self, x: &Array4<f32>) -> Array4<f32> {
        let (b, in_channels, h, w) = x.dim();
        debug_assert_eq!(in_channels, self.in_channels);
        let (kh, kw) = self.kernel;
        let (sh, sw) = self.stride;
        let out_h = (h - 1) * sh + kh;
        let out_w = (w - 1) * sw + kw;
        let oc = self.out_channels;
        let positions = h * w;

        let mut out = Array4::<f32>::zeros((b, oc, out_h, out_w));
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(bi, mut out_sample)| {
                if let Some(bias) = &self.bias {
                    for (channel, value) in bias.iter().enumerate() {
                        out_sample.slice_mut(s![channel, .., ..]).fill(*value);
                    }
                }
                let input = x.slice(s![bi, .., .., ..]);
                let input_2d = input
                    .to_shape((in_channels, positions))
                    .expect("standard layout");
                let mut product = vec![0.0f32; oc * positions];
                for (tap, weights) in self.taps.iter().enumerate() {
                    let (ky, kx) = (tap / kw, tap % kw);
                    let weight = ArrayView2::from_shape((oc, in_channels), weights)
                        .expect("tap is contiguous");
                    {
                        let mut out_view = ArrayViewMut2::from_shape((oc, positions), &mut product)
                            .expect("exact size");
                        ndarray::linalg::general_mat_mul(
                            1.0,
                            &weight,
                            &input_2d,
                            0.0,
                            &mut out_view,
                        );
                    }
                    for channel in 0..oc {
                        let mut target = out_sample.slice_mut(s![channel, .., ..]);
                        let values = &product[channel * positions..(channel + 1) * positions];
                        for iy in 0..h {
                            for ix in 0..w {
                                target[[iy * sh + ky, ix * sw + kx]] += values[iy * w + ix];
                            }
                        }
                    }
                }
            });
        out
    }
}
