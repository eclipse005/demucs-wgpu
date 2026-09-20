//! Validation of the device convolution (im2col + GEMM) against a host reference.
//!
//! The reference is a longhand convolution written out in this file, so it
//! shares no code with either the gather kernel or the GEMM. The weights are
//! padded on the host exactly the way the model's loader will pad them, because
//! that padding is part of the op's contract.

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{pad_ceil, Im2ColShape, Kernels};
use demucs_core::gpu::shaders::{BK, BM, BN};
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

/// The weight as the model uploads it: `(out_channels, k)` zero-padded to the
/// GEMM's A tiles.
fn padded_weight(weight: &[f32], out_channels: usize, k: usize) -> Vec<f32> {
    let (rows, cols) = (pad_ceil(out_channels, BM), pad_ceil(k, BK));
    let mut padded = vec![0.0f32; rows * cols];
    for row in 0..out_channels {
        padded[row * cols..row * cols + k].copy_from_slice(&weight[row * k..(row + 1) * k]);
    }
    padded
}

/// Longhand convolution with a per-output-channel bias.
fn reference(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    shape: Im2ColShape,
    out_channels: usize,
) -> Vec<f32> {
    let (in_channels, h, w) = (shape.in_channels, shape.h, shape.w);
    let (kh, kw) = shape.kernel;
    let (sh, sw) = shape.stride;
    let (ph, pw) = shape.pad;
    let (out_h, out_w) = shape.out_hw();
    let positions = out_h * out_w;
    let mut out = vec![0.0f32; shape.batch * out_channels * positions];
    for batch in 0..shape.batch {
        for oc in 0..out_channels {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = 0.0f64;
                    for ic in 0..in_channels {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy * sh + ky;
                                let ix = ox * sw + kx;
                                if iy < ph || iy >= ph + h || ix < pw || ix >= pw + w {
                                    continue;
                                }
                                let value = input
                                    [((batch * in_channels + ic) * h + (iy - ph)) * w + (ix - pw)];
                                let weight_value = weight
                                    [oc * (in_channels * kh * kw) + (ic * kh + ky) * kw + kx];
                                acc += (value * weight_value) as f64;
                            }
                        }
                    }
                    out[(batch * out_channels + oc) * positions + oy * out_w + ox] =
                        (acc as f32) + bias[oc];
                }
            }
        }
    }
    out
}

fn check(
    gpu: &Gpu,
    kernels: &Kernels,
    arena: &mut Arena,
    shape: Im2ColShape,
    out_channels: usize,
    label: &str,
) {
    let k = shape.k();
    let positions = shape.positions();
    let input_host = fill(shape.batch * shape.in_channels * shape.h * shape.w, 11);
    let weight_host = fill(out_channels * k, 23);
    let bias_host = fill(out_channels, 31);

    let input = arena
        .upload(gpu, &[shape.batch, shape.in_channels, shape.h, shape.w], &input_host, "x")
        .unwrap();
    let weight = arena
        .upload(
            gpu,
            &[pad_ceil(out_channels, BM), pad_ceil(k, BK)],
            &padded_weight(&weight_host, out_channels, k),
            "w",
        )
        .unwrap();
    let bias = arena.upload(gpu, &[out_channels], &bias_host, "b").unwrap();
    let pitch = pad_ceil(positions, BN);
    let k_pad = pad_ceil(k, BK);
    let patches = arena
        .tensor(gpu, &[shape.batch * k_pad, pitch], "patches")
        .unwrap();
    let out = arena
        .tensor(gpu, &[shape.batch, out_channels, positions], "conv.out")
        .unwrap();

    let mut recorder = Recorder::new(gpu);
    kernels
        .conv2d_into(
            gpu,
            arena,
            &mut recorder,
            &input,
            &weight,
            Some(&bias),
            &patches,
            &out,
            shape,
            out_channels,
        )
        .unwrap();
    assert_eq!(
        recorder.dispatches(),
        3,
        "one gather, then one batched GEMM and one bias pass over every batch"
    );
    recorder.submit(gpu).unwrap();

    let expected = reference(&input_host, &weight_host, &bias_host, shape, out_channels);
    let actual = read(gpu, &out);
    let comparison = compare(&expected, &actual).unwrap();
    println!(
        "{label}: rms_relative={:.3e} max_abs={:.3e} (oc={out_channels} k={k} positions={positions})",
        comparison.rms_relative_error(),
        comparison.max_abs,
    );
    assert!(
        comparison.rms_relative_error() < 1e-5,
        "{label}: conv2d diverges (rms-relative {:.3e})",
        comparison.rms_relative_error()
    );
    arena.reset();
}

#[test]
fn conv2d_matches_the_host_convolution() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("conv pipelines compile");
    let mut arena = Arena::new(&gpu, 512 << 20);

    // DTTNet's first_conv: 1x1, no padding, one reduction step.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Im2ColShape {
            batch: 1,
            in_channels: 4,
            h: 64,
            w: 96,
            kernel: (1, 1),
            stride: (1, 1),
            pad: (0, 0),
        },
        32,
        "1x1 (DTTNet first_conv)",
    );
    // A TFC block's 3x3: the case with edge padding on both axes.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Im2ColShape {
            batch: 1,
            in_channels: 32,
            h: 31,
            w: 29,
            kernel: (3, 3),
            stride: (1, 1),
            pad: (1, 1),
        },
        32,
        "3x3 pad 1, ragged",
    );
    // DTTNet's downsampler, with a batch of two: the gather is shared and each
    // batch gets its own GEMM against the same weights.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Im2ColShape {
            batch: 2,
            in_channels: 32,
            h: 64,
            w: 128,
            kernel: (2, 2),
            stride: (2, 2),
            pad: (0, 0),
        },
        64,
        "2x2 stride 2, batch 2 (DTTNet ds)",
    );
    // DTTNet's bottleneck: 96 channels, a 3x3 kernel (k = 864), and a 64 x 512
    // grid. The reduction is 54 whole BK steps and the output is 96 rows, so the
    // weight's padded rows (128) are wider than the output.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Im2ColShape {
            batch: 1,
            in_channels: 96,
            h: 64,
            w: 512,
            kernel: (3, 3),
            stride: (1, 1),
            pad: (1, 1),
        },
        96,
        "3x3 pad 1 at the bottleneck (96 channels, 64x512)",
    );
    // SCNet's SD layer: a frequency-axis-only kernel with no padding, where the
    // largest reduction (16 * 1 = 16) is narrower than a single BK step.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Im2ColShape {
            batch: 1,
            in_channels: 16,
            h: 357,
            w: 96,
            kernel: (16, 1),
            stride: (16, 1),
            pad: (0, 0),
        },
        64,
        "16x1 stride 16 (SCNet SD high band)",
    );
}

/// The patch scratch is not cleared before the gather, so the margins hold
/// whatever the previous layer left there. This is the test that says so.
///
/// Two mechanisms make that safe, and this checks both at once: the gather's
/// never-written column margin (`positions` is not a multiple of `BN` here) only
/// feeds accumulators the GEMM's epilogue discards, and the row margin
/// (`k = 4 * 9 = 36`, padded to 48) only multiplies the weight matrix's own k
/// padding, which is zero. A sentinel of 1e30 in both margins would flood the
/// output at ~1e30 if either argument were wrong.
#[test]
fn poisoned_patch_margins_cannot_reach_the_output() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("conv pipelines compile");
    let mut arena = Arena::new(&gpu, 512 << 20);

    // 4 channels, 3x3 kernel: k = 36 -> 48 rows. h * w = 13 * 11 = 143 positions
    // -> a 256-wide pitch.
    let shape = Im2ColShape {
        batch: 2,
        in_channels: 4,
        h: 13,
        w: 11,
        kernel: (3, 3),
        stride: (1, 1),
        pad: (1, 1),
    };
    let out_channels = 32;
    let k = shape.k();
    let positions = shape.positions();
    let input_host = fill(shape.batch * shape.in_channels * shape.h * shape.w, 11);
    let weight_host = fill(out_channels * k, 23);
    let bias_host = fill(out_channels, 31);

    let input = arena
        .upload(&gpu, &[shape.batch, shape.in_channels, shape.h, shape.w], &input_host, "x")
        .unwrap();
    let weight = arena
        .upload(
            &gpu,
            &[pad_ceil(out_channels, BM), pad_ceil(k, BK)],
            &padded_weight(&weight_host, out_channels, k),
            "w",
        )
        .unwrap();
    let bias = arena.upload(&gpu, &[out_channels], &bias_host, "b").unwrap();
    let pitch = pad_ceil(positions, BN);
    let patches = arena
        .tensor(&gpu, &[shape.batch * pad_ceil(k, BK), pitch], "patches")
        .unwrap();
    let out = arena
        .tensor(&gpu, &[shape.batch, out_channels, positions], "conv.out")
        .unwrap();

    // Every element of the scratch, not just the margins: the k *padding* rows
    // have to be poison too, or the test would pass by relying on them being zero.
    let sentinel = vec![1.0e30f32; patches.len()];
    gpu.upload(&patches.buffer, bytemuck::cast_slice(&sentinel));

    let mut recorder = Recorder::new(&gpu);
    kernels
        .conv2d_into(
            &gpu,
            &mut arena,
            &mut recorder,
            &input,
            &weight,
            Some(&bias),
            &patches,
            &out,
            shape,
            out_channels,
        )
        .unwrap();
    recorder.submit(&gpu).unwrap();

    let expected = reference(&input_host, &weight_host, &bias_host, shape, out_channels);
    let actual = read(&gpu, &out);
    let comparison = compare(&expected, &actual).unwrap();
    println!(
        "poisoned scratch: rms_relative={:.3e} max_abs={:.3e}",
        comparison.rms_relative_error(),
        comparison.max_abs
    );
    assert!(
        comparison.rms_relative_error() < 1e-5,
        "a poisoned patch margin reached the output (rms-relative {:.3e})",
        comparison.rms_relative_error()
    );
    assert!(
        actual.iter().all(|v| v.abs() < 1.0e6),
        "the output contains the sentinel"
    );
}
