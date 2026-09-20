//! The batched GEMM, which is what turns the axial attention into two matmuls.
//!
//! Attention in this model never needs the full `s x s` score matrix in one
//! buffer: the time axis is 801 and the band axis 60, so a single batched
//! dispatch per stage covers it. What it does need is per-batch addressing, and
//! the operands are indexed by `(band, head)` — two levels that are not a single
//! linear stride apart inside the fused QKV tensor. This test pins both the
//! arithmetic and the stride convention.

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{pad_matrix_for, GemmJob, Kernels};
use demucs_core::gpu::shaders::{BK, BM, BN};
use demucs_core::gpu::Gpu;
use ndarray::{Array2, ShapeBuilder};

fn gpu_or_skip() -> Option<Gpu> {
    match Gpu::new() {
        Ok(gpu) => {
            println!("adapter: {}", gpu.info.describe());
            Some(gpu)
        }
        Err(e) => {
            eprintln!("skipping: no usable wgpu adapter ({e})");
            None
        }
    }
}

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.8
        })
        .collect()
}

fn read(gpu: &Gpu, tensor: &DevTensor) -> Vec<f32> {
    let bytes = gpu
        .readback(&tensor.buffer, (tensor.len() * 4) as u64)
        .unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

/// Reference for a batched multiply where both operands and the output are
/// described by explicit strides, so the test does not need to know how the
/// kernel lays its tiles out.
///
/// The row strides matter: the kernel reads A with `lda` between rows and B with
/// `ldb`, and in the model those are *not* the logical widths (a band-major
/// tensor has rows 801 bands apart, not `k` apart).
#[allow(clippy::too_many_arguments)]
fn reference(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    batches: usize,
    inner: usize,
    a_outer: usize,
    a_inner: usize,
    b_outer: usize,
    b_inner: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batches * m * n];
    for batch in 0..batches {
        let outer = batch / inner;
        let inb = batch % inner;
        let abase = outer * a_outer + inb * a_inner;
        let bbase = outer * b_outer + inb * b_inner;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for q in 0..k {
                    acc += a[abase + i * lda + q] * b[bbase + q * ldb + j];
                }
                out[batch * m * n + i * n + j] = acc;
            }
        }
    }
    out
}

#[test]
fn batched_gemm_matches_the_reference_for_a_plain_stride() {
    // The frequency transformer's shape: 801 time steps, each a 60x384 slice of a
    // band-major tensor, multiplied by one shared weight.
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 512 << 20);

    let bands = 60usize;
    let frames = 801usize;
    let dim = 384usize;
    let out_dim = 1536usize;

    // A is the band-major (f, t, d) tensor: row f of batch t starts at
    // t*dim + f*(frames*dim).
    let a_len = bands * frames * dim;
    let a_host = fill(a_len, 3);
    let w_host = fill(out_dim * dim, 5);

    let a = arena.upload(&gpu, &[a_len], &a_host, "a").unwrap();
    let w = arena.upload(&gpu, &[dim, out_dim], &w_host, "w").unwrap();
    let c = arena.tensor(&gpu, &[frames * bands * out_dim], "c").unwrap();

    let job = GemmJob {
        m: bands,
        n: out_dim,
        k: dim,
        lda: frames * dim,
        ldb: out_dim,
        ldc: out_dim,
        batches: frames,
        inner_count: 1,
        a_outer: dim,
        a_inner: 0,
        b_outer: 0,
        b_inner: 0,
        c_outer: bands * out_dim,
        c_inner: 0,
        transb: false,
    };
    let mut recorder = Recorder::new(&gpu);
    kernels
        .gemm_into(&gpu, &mut arena, &mut recorder, &a, &w, &c, job)
        .unwrap();
    recorder.submit(&gpu).unwrap();

    let expected = reference(
        &a_host,
        &w_host,
        bands,
        out_dim,
        dim,
        job.lda,
        job.ldb,
        frames,
        1,
        dim,
        0,
        0,
        0,
    );
    let actual = read(&gpu, &c);
    let comparison = compare(&expected, &actual).unwrap();
    println!(
        "freq transformer qkv (batched stride): rms_relative={:.3e} max_abs={:.3e}",
        comparison.rms_relative_error(),
        comparison.max_abs
    );
    assert!(comparison.rms_relative_error() < 1e-5);
}

#[test]
fn batched_gemm_matches_the_reference_for_two_level_batches() {
    // The attention shapes: (band, head) batches over a fused QKV tensor whose
    // head stride is 64 and whose band stride is frames * 3 * heads * dim_head.
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 512 << 20);

    let bands = 4usize;
    let frames = 65usize;
    let heads = 8usize;
    let dim_head = 64usize;
    let inner = heads * dim_head; // q, k, v each take `inner` columns
    let row = 3 * inner;

    let qkv_len = bands * frames * row;
    let qkv_host = fill(qkv_len, 7);

    // Q as a left operand and K as a right one, both read out of the same fused
    // tensor through offset views — the constant `inner` offset belongs to the
    // view, not to a per-head stride.
    let q = arena.upload(&gpu, &[qkv_len], &qkv_host, "qkv").unwrap();
    let k = q.slice(inner, vec![qkv_len - inner]).unwrap();
    // Scores: one (frames, frames) block per (band, head).
    let c = arena
        .tensor(&gpu, &[bands * heads * frames * frames], "scores")
        .unwrap();

    let job = GemmJob {
        m: frames,
        n: frames,
        k: dim_head,
        lda: row,
        ldb: row,
        ldc: frames,
        batches: bands * heads,
        inner_count: heads,
        a_outer: frames * row,
        a_inner: dim_head,
        b_outer: frames * row,
        b_inner: dim_head,
        c_outer: heads * frames * frames,
        c_inner: frames * frames,
        transb: true,
    };
    let mut recorder = Recorder::new(&gpu);
    kernels
        .gemm_into(&gpu, &mut arena, &mut recorder, &q, &k, &c, job)
        .unwrap();
    recorder.submit(&gpu).unwrap();

    // Reference: Q(band, head) @ K(band, head)^T, with both taken from the fused
    // layout by explicit indexing rather than by strides.
    let mut expected = vec![0.0f32; bands * heads * frames * frames];
    for band in 0..bands {
        for head in 0..heads {
            for i in 0..frames {
                for j in 0..frames {
                    let mut acc = 0.0f32;
                    for q_i in 0..dim_head {
                        let qi = band * frames * row + i * row + head * dim_head + q_i;
                        let ki = band * frames * row + j * row + inner + head * dim_head + q_i;
                        acc += qkv_host[qi] * qkv_host[ki];
                    }
                    let out = (band * heads + head) * frames * frames + i * frames + j;
                    expected[out] = acc;
                }
            }
        }
    }
    let actual = read(&gpu, &c);
    let comparison = compare(&expected, &actual).unwrap();
    println!(
        "attention scores (band, head) two-level batch: rms_relative={:.3e} max_abs={:.3e}",
        comparison.rms_relative_error(),
        comparison.max_abs
    );
    assert!(comparison.rms_relative_error() < 1e-5);
}

#[test]
fn unbatched_gemm_is_unaffected_by_the_batch_machinery() {
    // Regression guard: the plain path must not have picked up the batched
    // shader's base arithmetic, which measured 16% slower on the linear shapes.
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 256 << 20);

    let m = 3204usize;
    let n = 1152usize;
    let k = 384usize;
    let a_host = fill(m * k, 11);
    let b_host = fill(k * n, 13);

    // Both operands padded to the tile multiples the kernel assumes.
    let (a_padded, a_rows, a_pitch) = pad_matrix_for(&a_host, m, k, BM, BK).unwrap();
    let (b_padded, b_rows, b_pitch) = pad_matrix_for(&b_host, k, n, BK, BN).unwrap();
    let a = arena
        .upload(&gpu, &[a_rows, a_pitch], &a_padded, "a")
        .unwrap();
    let b = arena
        .upload(&gpu, &[b_rows, b_pitch], &b_padded, "b")
        .unwrap();
    let c = arena.tensor(&gpu, &[m, n], "c").unwrap();

    let mut recorder = Recorder::new(&gpu);
    kernels
        .linear_into(&gpu, &mut arena, &mut recorder, &a, &b, &c, m, n, k)
        .unwrap();
    recorder.submit(&gpu).unwrap();

    let a2 = Array2::from_shape_vec((m, k), a_host).unwrap();
    let b2 = Array2::from_shape_vec((k, n), b_host).unwrap();
    let mut expected = Array2::<f32>::zeros((m, n));
    ndarray::linalg::general_mat_mul(1.0, &a2, &b2, 0.0, &mut expected);
    let comparison = compare(expected.as_slice().unwrap(), &read(&gpu, &c)).unwrap();
    println!(
        "unbatched linear: rms_relative={:.3e}",
        comparison.rms_relative_error()
    );
    assert!(comparison.rms_relative_error() < 1e-5);
}
