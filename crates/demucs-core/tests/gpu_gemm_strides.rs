//! Sweep `(k, lda)` on the GEMM to find where it stops agreeing with the host.
//!
//! Everything else in the band split is verified — the packed input, the norm's
//! output including its padding columns, and the transposed weight's layout. The
//! one thing that path does differently from every other GEMM in the model is
//! `lda > k`, so this isolates that variable on synthetic operands.

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, Recorder};
use demucs_core::gpu::kernels::{pad_ceil, GemmJob, Kernels};
use demucs_core::gpu::shaders::BK;
use demucs_core::gpu::Gpu;

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
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

#[test]
fn gemm_agrees_with_the_host_across_strides() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 1 << 30);

    let m = 61usize;
    let n = 384usize;

    // `lda` is the A row stride; `natural` is what the GEMM derives when left to
    // itself. The band split is the only caller that passes something larger.
    for (k, lda) in [
        (64usize, pad_ceil(64, BK)),
        (28, pad_ceil(28, BK)),
        (28, 528),
        (32, 528),
        (16, 64),
        (48, 64),
        (520, 528),
    ] {
        let mut host_a = vec![0.0f32; m * lda];
        let values = fill(m * k, 11);
        for row in 0..m {
            host_a[row * lda..row * lda + k].copy_from_slice(&values[row * k..(row + 1) * k]);
        }
        // The weight is `(in, out)` with `out` padded.
        let ldb = pad_ceil(n, 128);
        let mut host_b = vec![0.0f32; k * ldb];
        let w = fill(k * n, 23);
        for i in 0..k {
            host_b[i * ldb..i * ldb + n].copy_from_slice(&w[i * n..(i + 1) * n]);
        }

        let a = arena.upload(&gpu, &[m, lda], &host_a, "a").unwrap();
        let b = arena.upload(&gpu, &[k, ldb], &host_b, "b").unwrap();
        let c = arena.tensor(&gpu, &[m, n], "c").unwrap();

        let mut recorder = Recorder::new(&gpu);
        kernels
            .gemm_into(
                &gpu,
                &mut arena,
                &mut recorder,
                &a,
                &b,
                &c,
                GemmJob {
                    m,
                    n,
                    k,
                    lda,
                    ldb,
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
                },
            )
            .unwrap();
        recorder.submit(&gpu).unwrap();

        let bytes = gpu.readback(&c.buffer, (c.len() * 4) as u64).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);

        // Host reference, straight from the same padded operands.
        let mut want = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f64;
                for i in 0..k {
                    acc += (host_a[row * lda + i] as f64) * (host_b[i * ldb + col] as f64);
                }
                want[row * n + col] = acc as f32;
            }
        }
        let comparison = compare(&want, got).unwrap();
        println!(
            "k={k:>4} lda={lda:>4} (natural {:>4}): rms_relative={:.3e}",
            pad_ceil(k, BK),
            comparison.rms_relative_error()
        );
        assert!(
            comparison.rms_relative_error() < 1e-5,
            "k={k} lda={lda}: GEMM diverges"
        );

        // The model never uses a bare GEMM: every linear is followed by a bias
        // broadcast over the rows. That is the one step this sweep had left out.
        let host_bias = fill(n, 71);
        let bias = arena.upload(&gpu, &[n], &host_bias, "bias").unwrap();
        let mut recorder = Recorder::new(&gpu);
        kernels
            .add_in_place(&gpu, &mut arena, &mut recorder, &c, &bias)
            .unwrap();
        recorder.submit(&gpu).unwrap();
        let bytes = gpu.readback(&c.buffer, (c.len() * 4) as u64).unwrap();
        let got: &[f32] = bytemuck::cast_slice(&bytes);
        for row in 0..m {
            for col in 0..n {
                want[row * n + col] += host_bias[col];
            }
        }
        let biased = compare(&want, got).unwrap();
        println!("    with broadcast bias:            rms_relative={:.3e}", biased.rms_relative_error());
        assert!(
            biased.rms_relative_error() < 1e-5,
            "k={k} lda={lda}: bias broadcast diverges"
        );
        arena.reset();
    }
}
