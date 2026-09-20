//! Validation of the wgpu GEMM kernel against the CPU reference.
//!
//! The kernel is gated the same way every other stage in this port is: it is
//! diffed against the already-verified CPU implementation before anything is
//! built on top of it. Tests skip when no adapter is available so the suite stays
//! runnable on machines without one.
//!
//! This goes through [`Kernels::linear`] rather than a private test path, so what
//! is validated is the same op the model will call.

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{pad_matrix_for, Kernels};
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

/// Deterministic pseudo-random fill, so a failure is reproducible.
fn fill(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.8
        })
        .collect()
}

/// `C = A @ B` on the CPU, with the same row-major convention as the kernel.
fn reference(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let a = Array2::from_shape_vec((m, k), a.to_vec()).unwrap();
    let b = Array2::from_shape_vec((k, n), b.to_vec()).unwrap();
    let mut c = Array2::<f32>::zeros((m, n).f());
    ndarray::linalg::general_mat_mul(1.0, &a, &b, 0.0, &mut c);
    c.iter().copied().collect()
}

fn read(gpu: &Gpu, tensor: &DevTensor) -> Vec<f32> {
    let bytes = gpu
        .readback(&tensor.buffer, (tensor.len() * 4) as u64)
        .unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

fn check(gpu: &Gpu, arena: &mut Arena, kernels: &Kernels, m: usize, n: usize, k: usize, label: &str) {
    let a = fill(m, k, 1);
    let b = fill(k, n, 2);
    let expected = reference(&a, &b, m, n, k);

    // The kernel's inner loop has no bounds checks, so both operands are padded
    // to the tile multiples with zeros before they reach it: A to
    // (ceil(m/BM), ceil(k/BK)) and B to (ceil(k/BK), ceil(n/BN)).
    let (a_padded, a_rows, a_pitch) = pad_matrix_for(&a, m, k, BM, BK).unwrap();
    let (b_padded, b_rows, b_pitch) = pad_matrix_for(&b, k, n, BK, BN).unwrap();
    let a_dev = arena
        .upload(gpu, &[a_rows, a_pitch], &a_padded, "a")
        .unwrap();
    let b_dev = arena
        .upload(gpu, &[b_rows, b_pitch], &b_padded, "b")
        .unwrap();

    let mut recorder = Recorder::new(gpu);
    let c = kernels
        .linear(gpu, arena, &mut recorder, &a_dev, &b_dev, m, n, k)
        .unwrap();
    recorder.submit(gpu).unwrap();

    let actual = read(gpu, &c);
    assert_eq!(actual.len(), m * n, "{label}: wrong output length");
    let comparison = compare(&expected, &actual).expect("comparable");
    println!(
        "{label} ({m}x{k})@({k}x{n}): rms_relative={:.3e} max_abs={:.3e}",
        comparison.rms_relative_error(),
        comparison.max_abs
    );
    // Both sides accumulate in f32 but in different orders, so the bound is
    // round-off, not exactness.
    assert!(
        comparison.rms_relative_error() < 1e-5,
        "{label}: rms-relative error {:.3e} against a reference of scale {:.3e}",
        comparison.rms_relative_error(),
        comparison.reference_max_abs
    );
}

#[test]
fn gemm_matches_the_cpu_reference_across_tile_boundaries() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("kernel compiles and validates");
    let mut arena = Arena::new(&gpu, 256 << 20);

    // Exactly one tile, then shapes that straddle every boundary: rows not a
    // multiple of the M tile, columns not a multiple of the N tile, and K not a
    // multiple of the reduction step.
    check(&gpu, &mut arena, &kernels, 1, 1, 1, "scalar");
    check(&gpu, &mut arena, &kernels, 128, 128, 16, "exact tile");
    check(&gpu, &mut arena, &kernels, 129, 129, 17, "tile + 1");
    check(&gpu, &mut arena, &kernels, 255, 137, 33, "ragged");
    check(&gpu, &mut arena, &kernels, 384, 384, 384, "square");
}

#[test]
fn gemm_matches_the_cpu_reference_at_model_shapes() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("kernel compiles");
    let mut arena = Arena::new(&gpu, 512 << 20);

    // The band-split projection: (frames, bands, dim) flattened to rows.
    check(&gpu, &mut arena, &kernels, 3204, 1152, 384, "to_qkv");
    // The mask estimator's hidden layer.
    check(&gpu, &mut arena, &kernels, 3204, 1536, 1536, "mask hidden");
    // The time transformer at a batch of four chunks: 4 x 801 frames x 60 bands.
    check(&gpu, &mut arena, &kernels, 19224, 384, 384, "to_out, batch 4");
}
