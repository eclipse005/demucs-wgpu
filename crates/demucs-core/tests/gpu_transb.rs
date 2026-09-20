//! Isolation of the transposed batched GEMM, which is the scores step of
//! attention.
//!
//! `Q @ K^T` reads K out of its `[frame][dim_head]` storage with a transposed
//! stride and batches it by `(band, head)` with two different strides. This test
//! checks that step on its own, so a failure there is not confused with the
//! softmax or the AV product downstream.

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{GemmJob, Kernels};
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

#[test]
fn transposed_batched_gemm_matches_the_reference() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 512 << 20);

    // Small enough to reason about, shaped like the real thing.
    let bands = 3usize;
    let frames = 130usize; // crosses the 128 tile boundary
    let heads = 2usize;
    let dim_head = 64usize;
    let inner = heads * dim_head;
    let row = 3 * inner;

    let qkv_host = fill(bands * frames * row, 3);
    let qkv = arena
        .upload(&gpu, &[bands * frames * row], &qkv_host, "qkv")
        .unwrap();
    let q = qkv.slice(0, vec![qkv.len()]).unwrap();
    let k = qkv.slice(inner, vec![qkv.len() - inner]).unwrap();
    let scores = arena
        .tensor(&gpu, &[bands * heads * frames * frames], "scores")
        .unwrap();

    // The score tensor is batched contiguously by (band, head).
    let (c_outer, c_inner) = GemmJob::contiguous_batch(frames * frames, heads);
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
        c_outer,
        c_inner,
        transb: true,
    };
    let mut recorder = Recorder::new(&gpu);
    kernels
        .gemm_into(&gpu, &mut arena, &mut recorder, &q, &k, &scores, job)
        .unwrap();
    recorder.submit(&gpu).unwrap();

    // Q[i][d] = qkv[band][i][head*d + d], K[j][d] = qkv[band][j][inner + head*d + d].
    let mut expected = vec![0.0f32; bands * heads * frames * frames];
    for band in 0..bands {
        for head in 0..heads {
            for i in 0..frames {
                for j in 0..frames {
                    let mut acc = 0.0f32;
                    for d in 0..dim_head {
                        let qi = band * frames * row + i * row + head * dim_head + d;
                        let ki = band * frames * row + j * row + inner + head * dim_head + d;
                        acc += qkv_host[qi] * qkv_host[ki];
                    }
                    expected[(band * heads + head) * frames * frames + i * frames + j] = acc;
                }
            }
        }
    }

    let actual = read(&gpu, &scores);
    let comparison = compare(&expected, &actual).unwrap();
    println!(
        "QK^T transposed+batched: rms_relative={:.3e} max_abs={:.3e}",
        comparison.rms_relative_error(),
        comparison.max_abs
    );
    for batch in 0..(bands * heads) {
        let base = batch * frames * frames;
        println!(
            "  batch {batch} (band {}, head {}): expected {:?} got {:?}",
            batch / heads,
            batch % heads,
            &expected[base..base + 3],
            &actual[base..base + 3]
        );
    }
    assert!(comparison.rms_relative_error() < 1e-5);
}

/// Linear projections are `transb` with `batches == 1`. They used to share the
/// batched shader (dead `abase`/`bbase`/`cbase`); this pins the unbatched
/// variant against the same reference.
#[test]
fn single_batch_transb_matches_the_reference() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 64 << 20);

    // Crosses the 128 tile on both axes; k = 64 is a Linear / QK^T head width.
    let m = 192usize;
    let n = 160usize;
    let k = 64usize;
    let a_host = fill(m * k, 4);
    let b_host = fill(n * k, 5);
    let a = arena.upload(&gpu, &[m, k], &a_host, "a").unwrap();
    let b = arena.upload(&gpu, &[n, k], &b_host, "b").unwrap();
    let c = arena.tensor(&gpu, &[m, n], "c").unwrap();

    let job = GemmJob {
        m,
        n,
        k,
        lda: k,
        ldb: k,
        ldc: n,
        batches: 1,
        inner_count: 1,
        a_outer: 0,
        a_inner: 0,
        b_outer: 0,
        b_inner: 0,
        c_outer: 0,
        c_inner: 0,
        transb: true,
    };
    let mut recorder = Recorder::new(&gpu);
    kernels
        .gemm_into(&gpu, &mut arena, &mut recorder, &a, &b, &c, job)
        .unwrap();
    recorder.submit(&gpu).unwrap();

    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a_host[i * k + p] * b_host[j * k + p];
            }
            expected[i * n + j] = acc;
        }
    }
    let actual = read(&gpu, &c);
    let comparison = compare(&expected, &actual).unwrap();
    println!(
        "single-batch transb {m}x{n}x{k}: rms_relative={:.3e} max_abs={:.3e}",
        comparison.rms_relative_error(),
        comparison.max_abs
    );
    assert!(comparison.rms_relative_error() < 1e-5);
}
