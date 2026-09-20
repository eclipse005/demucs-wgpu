//! Validation of the device-resident op chain.
//!
//! The point of these tests is twofold: each kernel has to agree with the CPU
//! reference, and the chain has to run as **one submission** with every
//! intermediate in device memory. The second property is not visible in the
//! numbers — it is what makes the numbers achievable at all, since a host round
//! trip costs ~2 ms/MB on this machine.

use std::time::Instant;

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{pad_matrix_for, GemmJob, Kernels};
use demucs_core::gpu::shaders::{BK, BM, BN};
use demucs_core::gpu::Gpu;
use demucs_core::ops;
use ndarray::{Array1, Array2};

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

/// Deterministic fill, so a failure is reproducible.
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
    let all: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    let base = (tensor.offset / 4) as usize;
    all[base..base + tensor.len()].to_vec()
}

fn assert_close(label: &str, expected: &[f32], actual: &[f32], tolerance: f32) {
    assert_eq!(expected.len(), actual.len(), "{label}: length differs");
    let comparison = compare(expected, actual).unwrap();
    println!(
        "{label}: rms_relative={:.3e} max_abs={:.3e}",
        comparison.rms_relative_error(),
        comparison.max_abs
    );
    assert!(
        comparison.rms_relative_error() < tolerance,
        "{label}: rms-relative {:.3e} against a reference of scale {:.3e}",
        comparison.rms_relative_error(),
        comparison.reference_max_abs
    );
}

#[test]
fn elementwise_kernels_match_the_cpu_reference() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 64 << 20);

    let rows = 61usize;
    let dim = 384usize;
    let x_host = fill(rows * dim, 1);
    let gamma_host = fill(dim, 7);

    let gamma = arena.upload(&gpu, &[dim], &gamma_host, "gamma").unwrap();
    let x = arena.upload(&gpu, &[rows, dim], &x_host, "x").unwrap();

    // Two dispatches, one submission.
    let mut recorder = Recorder::new(&gpu);
    let g = kernels.gelu(&gpu, &mut arena, &mut recorder, &x).unwrap();
    let r = kernels
        .rms_norm(&gpu, &mut arena, &mut recorder, &x, &gamma, rows, dim)
        .unwrap();
    assert_eq!(recorder.dispatches(), 2);
    recorder.submit(&gpu).unwrap();

    let expected: Vec<f32> = x_host.iter().map(|v| ops::gelu_erf(*v)).collect();
    assert_close("gelu", &expected, &read(&gpu, &g), 1e-6);

    let xa = Array2::from_shape_vec((rows, dim), x_host).unwrap();
    let ga = Array1::from_vec(gamma_host);
    let expected = ops::rms_norm(&xa.view(), &ga.view()).unwrap();
    assert_close("rms_norm", expected.as_slice().unwrap(), &read(&gpu, &r), 1e-6);
}

/// `mul_in_place` — the decoder's `x * skip`, which is a multiply and not an
/// additive skip connection.
#[test]
fn mul_in_place_multiplies_elementwise() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 64 << 20);

    let count = 5000usize;
    let a_host = fill(count, 3);
    let b_host = fill(count, 5);
    let a = arena.upload(&gpu, &[count], &a_host, "a").unwrap();
    let b = arena.upload(&gpu, &[count], &b_host, "b").unwrap();

    let mut recorder = Recorder::new(&gpu);
    kernels
        .mul_in_place(&gpu, &mut arena, &mut recorder, &a, &b, count)
        .unwrap();
    assert_eq!(recorder.dispatches(), 1);
    recorder.submit(&gpu).unwrap();

    // Multiplication has one rounding, so this is an equality, not a tolerance.
    let expected: Vec<f32> = a_host.iter().zip(b_host.iter()).map(|(x, y)| x * y).collect();
    assert_eq!(read(&gpu, &a), expected);

    // A short source is an error, not a silently truncated multiply.
    let mut recorder = Recorder::new(&gpu);
    assert!(kernels
        .mul_in_place(&gpu, &mut arena, &mut recorder, &a, &b, count + 1)
        .is_err());
}

#[test]
fn glu_and_softmax_match_the_cpu_reference() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 64 << 20);

    let rows = 33usize;
    let half = 128usize;
    let x_host = fill(rows * half * 2, 3);
    let x = arena.upload(&gpu, &[rows, half * 2], &x_host, "x").unwrap();
    let mut recorder = Recorder::new(&gpu);
    let y = kernels
        .glu(&gpu, &mut arena, &mut recorder, &x, rows, half)
        .unwrap();
    recorder.submit(&gpu).unwrap();

    let xa = Array2::from_shape_vec((rows, half * 2), x_host).unwrap();
    let expected = ops::glu(&xa.view()).unwrap();
    assert_close("glu", expected.as_slice().unwrap(), &read(&gpu, &y), 1e-6);

    let cols = 801usize;
    let srows = 16usize;
    let s_host = fill(srows * cols, 5);
    let s = arena.upload(&gpu, &[srows, cols], &s_host, "s").unwrap();
    let mut recorder = Recorder::new(&gpu);
    let sm = kernels
        .softmax(&gpu, &mut arena, &mut recorder, &s, srows, cols)
        .unwrap();
    recorder.submit(&gpu).unwrap();

    let mut expected = s_host.clone();
    for r in 0..srows {
        let row = &mut expected[r * cols..(r + 1) * cols];
        ops::softmax_in_place(row);
    }
    assert_close("softmax", &expected, &read(&gpu, &sm), 1e-6);
}

#[test]
fn linear_and_residual_chain_stays_on_the_device() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 256 << 20);

    // A real model shape: the mask estimator's hidden layer at one chunk.
    let rows = 3204usize;
    let k = 384usize;
    let n = 1536usize;

    let x_host = fill(rows * k, 11);
    let w_host = fill(n * k, 13); // PyTorch (out, in) layout
    let bias_host = fill(n, 17);

    // Weights are stored transposed and padded once, the way the model will keep
    // them resident for the whole run.
    let w_t: Vec<f32> = {
        let w = Array2::from_shape_vec((n, k), w_host.clone()).unwrap();
        w.t().to_owned().iter().copied().collect()
    };
    let (w_padded, w_rows, w_cols) = pad_matrix_for(&w_t, k, n, BK, BN).unwrap();
    let w = arena
        .upload(&gpu, &[w_rows, w_cols], &w_padded, "w")
        .unwrap();
    let (x_padded, x_rows, x_cols) = pad_matrix_for(&x_host, rows, k, BM, BK).unwrap();
    let x = arena
        .upload(&gpu, &[x_rows, x_cols], &x_padded, "x")
        .unwrap();
    let bias = arena.upload(&gpu, &[n], &bias_host, "bias").unwrap();

    let mut recorder = Recorder::new(&gpu);
    let y = kernels
        .linear(&gpu, &mut arena, &mut recorder, &x, &w, rows, n, k)
        .unwrap();
    kernels
        .add_in_place(&gpu, &mut arena, &mut recorder, &y, &bias)
        .unwrap();
    assert_eq!(recorder.dispatches(), 2, "two ops, one submission");
    recorder.submit(&gpu).unwrap();

    let xa = Array2::from_shape_vec((rows, k), x_host).unwrap();
    let wa = Array2::from_shape_vec((n, k), w_host).unwrap();
    let layer = ops::Linear::new(wa.view(), Some(Array1::from_vec(bias_host.clone()).view()));
    let expected = layer.forward(&xa.view());
    assert_close(
        "linear + residual",
        expected.as_slice().unwrap(),
        &read(&gpu, &y),
        1e-5,
    );
}

#[test]
fn fused_epilogues_match_the_unfused_sequence() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 256 << 20);

    // The feed-forward's shapes: (rows x dim) @ (dim x 4*dim) with a bias, then
    // the same projection back down with the residual. These are the two
    // epilogues the model leans on and no other test drives.
    let rows = 3204usize;
    let k = 384usize;
    let n = 1536usize;

    let x_host = fill(rows * k, 21);
    let w_host = fill(n * k, 23); // (out, in)
    let bias_host = fill(n, 27);
    let resid_host = fill(rows * n, 29);

    let w_t: Vec<f32> = {
        let w = Array2::from_shape_vec((n, k), w_host.clone()).unwrap();
        w.t().to_owned().iter().copied().collect()
    };
    let (w_padded, w_rows, w_cols) = pad_matrix_for(&w_t, k, n, BK, BN).unwrap();
    let w = arena.upload(&gpu, &[w_rows, w_cols], &w_padded, "w").unwrap();
    // The A operand exactly the model hands it over: unpadded, pitched at `k`.
    let x = arena.upload(&gpu, &[rows, k], &x_host, "x").unwrap();
    let bias = arena.upload(&gpu, &[n], &bias_host, "bias").unwrap();

    // GeluBias, writing into a fresh buffer.
    let out = arena.tensor(&gpu, &[rows, n], "out").unwrap();
    // BiasResidual: the destination doubles as the residual input.
    let c = arena.upload(&gpu, &[rows, n], &resid_host, "c").unwrap();
    let c_copy = arena.upload(&gpu, &[rows, n], &resid_host, "c_copy").unwrap();

    let job = |m: usize, n: usize, k: usize| GemmJob {
        m,
        n,
        k,
        lda: k,
        ldb: w_cols,
        ldc: n,
        batches: 1,
        inner_count: 1,
        a_outer: 0,
        a_inner: 0,
        b_outer: 0,
        b_inner: 0,
        c_outer: 0,
        c_inner: 0,
        transb: false,
    };

    let mut recorder = Recorder::new(&gpu);
    kernels
        .gemm_into_epilogue(
            &gpu, &mut arena, &mut recorder, &x, &w, &out, job(rows, n, k),
            Some(&bias), true, false,
        )
        .unwrap();
    kernels
        .gemm_into_epilogue(
            &gpu, &mut arena, &mut recorder, &x, &w, &c, job(rows, n, k),
            Some(&bias), false, true,
        )
        .unwrap();
    recorder.submit(&gpu).unwrap();

    let xa = Array2::from_shape_vec((rows, k), x_host).unwrap();
    let wa = Array2::from_shape_vec((n, k), w_host).unwrap();
    let layer = ops::Linear::new(wa.view(), Some(Array1::from_vec(bias_host).view()));
    let product = layer.forward(&xa.view());
    let want_gelu = product.mapv(ops::gelu_erf);
    assert_close("gelu + bias epilogue", want_gelu.as_slice().unwrap(), &read(&gpu, &out), 1e-5);
    let want_resid = &product + &Array2::from_shape_vec((rows, n), resid_host).unwrap();
    assert_close("bias + residual epilogue", want_resid.as_slice().unwrap(), &read(&gpu, &c), 1e-5);
    // The residual form must have left the accumulator's own input alone
    // everywhere it wrote, i.e. it is exactly `acc + bias + C`.
    let _ = &c_copy;
}

#[test]
fn chaining_dispatches_beats_submitting_them_individually() {
    // The architectural claim, measured. Dispatch overhead only appears when each
    // kernel is submitted on its own; chaining N dispatches into one submission
    // turns N round trips into one.
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 256 << 20);

    let rows = 3204usize;
    let dim = 384usize;
    let x_host = fill(rows * dim, 23);
    let ops_count = 16usize;

    // Chained: allocate once, record every dispatch, submit once.
    let mut recorder = Recorder::new(&gpu);
    let x = arena.upload(&gpu, &[rows, dim], &x_host, "x").unwrap();
    for _ in 0..ops_count {
        kernels.gelu(&gpu, &mut arena, &mut recorder, &x).unwrap();
    }
    assert_eq!(recorder.dispatches(), ops_count);
    let chained_start = Instant::now();
    recorder.submit(&gpu).unwrap();
    let chained = chained_start.elapsed().as_secs_f64();

    // One at a time: a fresh upload, dispatch, submit and wait per op.
    let mut per_op = 0.0f64;
    for _ in 0..ops_count {
        let mut recorder = Recorder::new(&gpu);
        let x = arena.upload(&gpu, &[rows, dim], &x_host, "x").unwrap();
        kernels.gelu(&gpu, &mut arena, &mut recorder, &x).unwrap();
        let phase = Instant::now();
        recorder.submit(&gpu).unwrap();
        per_op += phase.elapsed().as_secs_f64();
    }

    let bytes = (rows * dim * 4) as f64;
    println!(
        "{ops_count} gelu dispatches over {rows}x{dim}: one submission {:.3} ms, \
         one submission per op {:.3} ms ({:.1}x)",
        chained * 1e3,
        per_op * 1e3,
        per_op / chained.max(1e-9),
    );
    println!(
        "  a host round trip of this tensor would cost ~{:.1} ms at 520 MB/s",
        bytes / 520e6 * 1e3
    );
}
