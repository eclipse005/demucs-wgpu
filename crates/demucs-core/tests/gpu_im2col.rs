//! Validation of the im2col gather against a host reference.
//!
//! This kernel is the front half of every convolution in the conv family: its
//! output is the GEMM's B operand, so an error here is a wrong convolution that
//! still looks like a convolution. The reference is written out longhand in the
//! test rather than borrowed from `conv.rs`, so the two implementations cannot
//! share a mistake.

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{Im2ColShape, Kernels};
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

/// Patches laid out `(batch, k, positions)` with `k = (ic, ky, kx)` and
/// `positions = (oy, ox)`, zero padded — the same description the kernel gets.
fn reference(input: &[f32], shape: Im2ColShape) -> Vec<f32> {
    let (in_channels, h, w) = (shape.in_channels, shape.h, shape.w);
    let (kh, kw) = shape.kernel;
    let (sh, sw) = shape.stride;
    let (ph, pw) = shape.pad;
    let (out_h, out_w) = shape.out_hw();
    let k = shape.k();
    let positions = out_h * out_w;

    let mut patches = vec![0.0f32; shape.output_len()];
    for batch in 0..shape.batch {
        for ic in 0..in_channels {
            for ky in 0..kh {
                for kx in 0..kw {
                    let krow = (ic * kh + ky) * kw + kx;
                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let iy = oy * sh + ky;
                            let ix = ox * sw + kx;
                            if iy < ph || iy >= ph + h || ix < pw || ix >= pw + w {
                                continue;
                            }
                            let value = input[((batch * in_channels + ic) * h + (iy - ph)) * w
                                + (ix - pw)];
                            patches[(batch * k + krow) * positions + oy * out_w + ox] = value;
                        }
                    }
                }
            }
        }
    }
    patches
}

fn check(gpu: &Gpu, kernels: &Kernels, arena: &mut Arena, shape: Im2ColShape, label: &str) {
    // A tight buffer: a row per reduction element, one batch's worth per batch.
    check_at_pitch(gpu, kernels, arena, shape, shape.positions(), shape.k(), label);
}

/// `pitch == positions` and `rows == k` is the tight layout; wider values are
/// what the GEMM needs, and its margin columns must stay the caller's zeros.
fn check_at_pitch(
    gpu: &Gpu,
    kernels: &Kernels,
    arena: &mut Arena,
    shape: Im2ColShape,
    pitch: usize,
    rows: usize,
    label: &str,
) {
    let input_host = fill(shape.batch * shape.in_channels * shape.h * shape.w, 7);
    let input = arena
        .upload(gpu, &[shape.batch, shape.in_channels, shape.h, shape.w], &input_host, "x")
        .unwrap();
    let out = arena
        .tensor(gpu, &[shape.batch * rows, pitch], "patches")
        .unwrap();
    // The kernel never writes the margin, so it has to start at zero.
    arena.clear(gpu, &out);

    let mut recorder = Recorder::new(gpu);
    kernels
        .im2col_into(gpu, arena, &mut recorder, &input, &out, shape, pitch, rows)
        .unwrap();
    assert_eq!(recorder.dispatches(), 1);
    recorder.submit(gpu).unwrap();

    let expected = reference(&input_host, shape);
    let actual = read(gpu, &out);
    // Gather the live columns back out of the pitched buffer. `rows` is the
    // padded row count, so a batch's live rows are its first `k`.
    let positions = shape.positions();
    let k = shape.k();
    let mut live = Vec::with_capacity(expected.len());
    let mut margin_nonzero = 0usize;
    for batch in 0..shape.batch {
        for row in 0..rows {
            let base = (batch * rows + row) * pitch;
            if row < k {
                live.extend_from_slice(&actual[base..base + positions]);
            }
            margin_nonzero += actual[base + positions..base + pitch]
                .iter()
                .filter(|v| **v != 0.0)
                .count();
        }
    }
    let comparison = compare(&expected, &live).unwrap();
    println!(
        "{label}: rms_relative={:.3e} max_abs={:.3e} (k={} positions={} pitch={pitch} rows={rows}, margin non-zero {margin_nonzero})",
        comparison.rms_relative_error(),
        comparison.max_abs,
        k,
        positions,
    );
    assert_eq!(margin_nonzero, 0, "{label}: the kernel wrote into the tile margin");
    assert!(
        comparison.rms_relative_error() < 1e-7,
        "{label}: im2col diverges (rms-relative {:.3e})",
        comparison.rms_relative_error()
    );
    arena.reset();
}

/// The gather is exact arithmetic (copies), so every shape must be bit-exact.
#[test]
fn im2col_matches_the_host_gather() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("im2col pipeline compiles");
    let mut arena = Arena::new(&gpu, 256 << 20);

    // DTTNet's first_conv: 1x1, no padding.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Im2ColShape {
            batch: 1,
            in_channels: 4,
            h: 256,
            w: 2048,
            kernel: (1, 1),
            stride: (1, 1),
            pad: (0, 0),
        },
        "1x1 (DTTNet first_conv)",
    );
    // The TFC blocks: 3x3, padding 1 — the case with the most edge work.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Im2ColShape {
            batch: 2,
            in_channels: 32,
            h: 33,
            w: 37,
            kernel: (3, 3),
            stride: (1, 1),
            pad: (1, 1),
        },
        "3x3 pad 1, ragged, batch 2",
    );
    // DTTNet's downsampling: 2x2, stride 2, no padding.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Im2ColShape {
            batch: 1,
            in_channels: 32,
            h: 256,
            w: 2048,
            kernel: (2, 2),
            stride: (2, 2),
            pad: (0, 0),
        },
        "2x2 stride 2 (DTTNet ds)",
    );
    // SCNet's SD layer: a column kernel down the frequency axis, no padding,
    // and the frequency extent is not a multiple of the stride.
    check(
        &gpu,
&kernels,
        &mut arena,
        Im2ColShape {
            batch: 1,
            in_channels: 16,
            h: 359,
            w: 128,
            kernel: (3, 1),
            stride: (1, 1),
            pad: (0, 0),
        },
        "3x1 stride 1 (SCNet SD low band)",
    );
    check(
        &gpu,
        &kernels,
        &mut arena,
        Im2ColShape {
            batch: 1,
            in_channels: 16,
            h: 359,
            w: 128,
            kernel: (16, 1),
            stride: (16, 1),
            pad: (0, 0),
        },
        "16x1 stride 16 (SCNet SD high band)",
    );
    // The waveform branch at a real chunk: `w` is the sample count, past the
    // 65535 a packed parameter pair could hold.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Im2ColShape {
            batch: 1,
            in_channels: 2,
            h: 1,
            w: 70_000,
            kernel: (1, 8),
            stride: (1, 4),
            pad: (0, 2),
        },
        "1x8 stride 4, w past 16 bits (htdemucs tencoder.0 at a real chunk)",
    );
    // The form the conv composition actually uses: rows padded to the GEMM's
    // 128-wide N tile, so the reduction is one dispatch with no repacking.
    let gemm_tile = 128;
    let shape = Im2ColShape {
        batch: 1,
        in_channels: 32,
        h: 64,
        w: 130,
        kernel: (3, 3),
        stride: (1, 1),
        pad: (1, 1),
    };
    let pitch = shape.positions().div_ceil(gemm_tile) * gemm_tile;
    let rows = shape.k().div_ceil(16) * 16;
    check_at_pitch(
        &gpu,
        &kernels,
        &mut arena,
        shape,
        pitch,
        rows,
        "3x3 pad 1, N- and K-padded for the GEMM",
    );
}
