//! Validation of the DTTNet band-sequence permutation.
//!
//! This is the op where a wrong answer looks most like a right one: the
//! permutation reorders which (time, frequency) pair each value belongs to, so
//! getting it wrong still produces a tensor of plausible activations that
//! separates audio — just worse. Two properties pin it down: the two modes have
//! to be exact inverses (a round trip must be bit-exact, since both are pure
//! moves), and mode 0 has to agree with a longhand index computation written out
//! in this file.

use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::Kernels;
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
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_201_223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        })
        .collect()
}

fn read(gpu: &Gpu, tensor: &DevTensor) -> Vec<f32> {
    let bytes = gpu
        .readback(&tensor.buffer, (tensor.len() * 4) as u64)
        .unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

/// `(b, c, t, f) -> (b * heads, f, t, per_head)`, as the reference's
/// `view(b * n_heads, c // n_heads, t, f).permute(0, 3, 2, 1)` describes it.
#[allow(clippy::too_many_arguments)]
fn reference_scatter(
    x: &[f32],
    batches: usize,
    heads: usize,
    per_head: usize,
    t: usize,
    f: usize,
) -> Vec<f32> {
    let channels = heads * per_head;
    let mut out = vec![0.0f32; x.len()];
    for b in 0..batches {
        for head in 0..heads {
            for fi in 0..f {
                for ti in 0..t {
                    for j in 0..per_head {
                        let source =
                            ((b * channels + head * per_head + j) * t + ti) * f + fi;
                        let dest = ((b * heads + head) * f + fi) * (t * per_head)
                            + ti * per_head
                            + j;
                        out[dest] = x[source];
                    }
                }
            }
        }
    }
    out
}

fn check(gpu: &Gpu, kernels: &Kernels, arena: &mut Arena, batches: usize, heads: usize, per_head: usize, t: usize, f: usize, label: &str) {
    let count = batches * heads * per_head * t * f;
    let host = fill(count, 17);
    let x = arena.upload(gpu, &[count], &host, "x").unwrap();
    let permuted = arena.tensor(gpu, &[count], "permuted").unwrap();
    let restored = arena.tensor(gpu, &[count], "restored").unwrap();

    let mut recorder = Recorder::new(gpu);
    kernels
        .heads_permute_into(gpu, arena, &mut recorder, &x, &permuted, batches, heads, per_head, t, f, false)
        .unwrap();
    kernels
        .heads_permute_into(gpu, arena, &mut recorder, &permuted, &restored, batches, heads, per_head, t, f, true)
        .unwrap();
    assert_eq!(recorder.dispatches(), 2);
    recorder.submit(gpu).unwrap();

    // The scatter direction must agree with the longhand map, element for element
    // (a permutation has no rounding, so this is an equality test).
    let expected = reference_scatter(&host, batches, heads, per_head, t, f);
    let actual = read(gpu, &permuted);
    let mismatches = expected
        .iter()
        .zip(actual.iter())
        .filter(|(want, got)| *want != *got)
        .count();
    assert_eq!(mismatches, 0, "{label}: scatter disagrees with the longhand map");

    // ... and the gather must be its exact inverse.
    let back = read(gpu, &restored);
    let mismatches = host
        .iter()
        .zip(back.iter())
        .filter(|(want, got)| *want != *got)
        .count();
    assert_eq!(mismatches, 0, "{label}: the round trip is not exact");
    println!("{label}: exact in both directions (count={count})");

    // A permutation must also preserve the multiset: nothing may be dropped or
    // duplicated if a grid bound is off by one.
    let mut sorted_expected = expected.clone();
    let mut sorted_actual = actual.clone();
    sorted_expected.sort_by(f32::total_cmp);
    sorted_actual.sort_by(f32::total_cmp);
    assert_eq!(sorted_expected, sorted_actual, "{label}: values were lost");
    arena.reset();
}

#[test]
fn heads_permute_matches_the_reference_view() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("permute pipeline compiles");
    let mut arena = Arena::new(&gpu, 256 << 20);

    // Ragged: nothing here divides anything else.
    check(&gpu, &kernels, &mut arena, 2, 3, 5, 4, 7, "ragged");
    // DTTNet's bottleneck at one chunk: `(1, 96, 16, 512)` in, `(2, 512, 16, 48)`
    // out, which is the shape the band sequence actually runs.
    check(&gpu, &kernels, &mut arena, 1, 2, 48, 16, 512, "DTTNet bottleneck");
    // Batch > 1, to catch a batch stride that uses the un-permuted head count.
    check(&gpu, &kernels, &mut arena, 4, 2, 48, 3, 5, "batch 4");
}
