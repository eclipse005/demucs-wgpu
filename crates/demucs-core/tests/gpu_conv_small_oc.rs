//! The fused small-`oc` convolution (`shaders::conv_small_oc`): the im2col
//! gather and the GEMM replaced by one kernel that carries
//! `out_channels * positions` accumulators per thread.
//!
//! The claim under test is stronger than "close": the fused kernel accumulates
//! in the materialised path's own order — `fma(w, v, acc)` with `k` ascending
//! over `(in_channel, tap)`, which is the order im2col lays the patch rows out
//! in — so the two forms must agree *bit for bit*. A tolerance here would hide
//! a reordering, and a reordering is exactly the thing that would make this
//! kernel's output differ from the model's reference.
//!
//! It lives in its own binary because the fused form is selected from an
//! environment variable read at call time, and a sibling test asserting the
//! materialised form's dispatch count would see the flag flip under it.

use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{pad_ceil, Im2ColShape, Kernels};
use demucs_core::gpu::shaders::{BK, BM};
use demucs_core::gpu::Gpu;

fn gpu_or_skip() -> Option<Gpu> {
    match Gpu::new() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("skipping: no usable wgpu adapter ({e})");
            None
        }
    }
}

fn fill(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32 / (u32::MAX as f32) - 0.5
        })
        .collect()
}

fn read(gpu: &Gpu, tensor: &DevTensor) -> Vec<f32> {
    let bytes = gpu
        .readback(&tensor.buffer, (tensor.len() * 4) as u64)
        .unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

/// Longhand convolution, the reference both GPU forms are checked against.
#[allow(clippy::too_many_arguments)]
fn reference(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    in_channels: usize,
    w: usize,
    kernel_w: usize,
    pad: usize,
    out_channels: usize,
    out_w: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_channels * out_w];
    for b in 0..batch {
        for oc in 0..out_channels {
            for ox in 0..out_w {
                let mut acc = bias[oc];
                for ic in 0..in_channels {
                    for kx in 0..kernel_w {
                        let ix = ox + kx;
                        if ix < pad || ix >= pad + w {
                            continue;
                        }
                        let pixel = input[(b * in_channels + ic) * w + (ix - pad)];
                        let tap = weight[oc * (in_channels * kernel_w) + ic * kernel_w + kx];
                        acc += pixel * tap;
                    }
                }
                out[(b * out_channels + oc) * out_w + ox] = acc;
            }
        }
    }
    out
}

/// The three shapes the model actually asks for — `hidden` of 6, 12 and 24,
/// with the dilated 3- and 5-tap kernels — against both the materialised form
/// and the longhand reference.
///
/// `positions` is deliberately not a multiple of the per-thread block (8 and 4):
/// the last block's stores are the ones `out_w`'s guard covers, and a wrong
/// guard would scribble past the row or drop its tail.
#[test]
fn the_fused_small_oc_form_is_bit_identical_to_the_materialised_one() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut arena = Arena::new(&gpu, 256 << 20);
    let kernels = Kernels::new(&gpu).unwrap();

    for (out_channels, kernel_w, batch) in [(6usize, 3usize, 3usize), (12, 5, 2), (24, 5, 2)] {
        let in_channels = 2 * out_channels;
        let h = 1usize;
        // 37 positions with a 5-tap kernel and 2 of padding: a partial last
        // block, and both edges of the window.
        let w = 37usize;
        let pad = (0usize, kernel_w / 2);
        let shape = Im2ColShape {
            batch,
            in_channels,
            h,
            w,
            kernel: (1, kernel_w),
            stride: (1, 1),
            pad,
        };
        let (_, out_w) = shape.out_hw();
        let k = shape.k();
        let k_pad = pad_ceil(k, BK);

        let input_host = fill(batch * in_channels * w, 11);
        let weight_host = fill(out_channels * k, 23);
        let bias_host = fill(out_channels, 31);
        // The weight as the model uploads it: `(out_channels, k)` zero-padded to
        // the GEMM's A tiles, row by row.
        let mut padded = vec![0.0f32; pad_ceil(out_channels, BM) * k_pad];
        for row in 0..out_channels {
            padded[row * k_pad..row * k_pad + k]
                .copy_from_slice(&weight_host[row * k..(row + 1) * k]);
        }

        let input = arena
            .upload(&gpu, &[batch, in_channels, h, w], &input_host, "x")
            .unwrap();
        let weight = arena
            .upload(
                &gpu,
                &[pad_ceil(out_channels, BM), k_pad],
                &padded,
                "w",
            )
            .unwrap();
        let bias = arena.upload(&gpu, &[out_channels], &bias_host, "b").unwrap();
        let patches = arena
            .tensor(&gpu, &[batch * k_pad * pad_ceil(out_w, 128)], "patches")
            .unwrap();
        let materialised_out = arena
            .tensor(&gpu, &[batch, out_channels, out_w], "conv.materialised")
            .unwrap();
        let fused_out = arena
            .tensor(&gpu, &[batch, out_channels, out_w], "conv.fused")
            .unwrap();

        let mut run = |fused: bool| -> Vec<f32> {
            let mut recorder = Recorder::new(&gpu);
            let stored = if fused { &fused_out } else { &materialised_out };
            if fused {
                kernels
                    .conv2d_small_oc_into(
                        &gpu,
                        &mut arena,
                        &mut recorder,
                        &input,
                        &weight,
                        Some(&bias),
                        stored,
                        shape,
                        out_channels,
                    )
                    .unwrap();
            } else {
                // The materialised pair, with the fused form switched off: this
                // binary's own selection reads the same flag.
                std::env::set_var("DEMUCS_SMALL_OC", "0");
                kernels
                    .conv2d_into(
                        &gpu,
                        &mut arena,
                        &mut recorder,
                        &input,
                        &weight,
                        Some(&bias),
                        &patches,
                        stored,
                        shape,
                        out_channels,
                    )
                    .unwrap();
                std::env::remove_var("DEMUCS_SMALL_OC");
            }
            recorder.submit(&gpu).unwrap();
            read(&gpu, stored)
        };
        let materialised = run(false);
        let fused = run(true);

        let expected = reference(
            &input_host,
            &weight_host,
            &bias_host,
            batch,
            in_channels,
            w,
            kernel_w,
            pad.1,
            out_channels,
            out_w,
        );
        let worst = expected
            .iter()
            .zip(materialised.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-5,
            "oc {out_channels}: the materialised form is off by {worst:.3e}"
        );
        assert_eq!(
            materialised, fused,
            "oc {out_channels}: the fused form differs from the materialised one"
        );
    }
}

/// The form is only valid for the shapes it was generated for, and the ones it
/// refuses have to be refused loudly: a silently wrong convolution is worse
/// than a slow one.
#[test]
fn the_fused_small_oc_form_refuses_shapes_it_cannot_take() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let mut arena = Arena::new(&gpu, 64 << 20);
    let kernels = Kernels::new(&gpu).unwrap();
    let input = arena.upload(&gpu, &[1, 4, 1, 16], &fill(64, 3), "x").unwrap();
    let k_pad = pad_ceil(4 * 3, BK);
    let weight = arena
        .upload(&gpu, &[BM, k_pad], &fill(BM * k_pad, 5), "w")
        .unwrap();
    let out = arena.tensor(&gpu, &[1, 6, 16], "y").unwrap();

    let base = Im2ColShape {
        batch: 1,
        in_channels: 4,
        h: 1,
        w: 16,
        kernel: (1, 3),
        stride: (1, 1),
        pad: (0, 1),
    };
    let mut attempts = Vec::new();
    // A 2-D kernel: the fused form's window is one row of taps.
    attempts.push(Im2ColShape {
        kernel: (3, 3),
        pad: (1, 1),
        ..base
    });
    // A rank-4 input.
    attempts.push(Im2ColShape { h: 8, ..base });
    // Strided.
    attempts.push(Im2ColShape {
        stride: (1, 2),
        ..base
    });
    for shape in attempts {
        let mut recorder = Recorder::new(&gpu);
        let err = kernels
            .conv2d_small_oc_into(
                &gpu,
                &mut arena,
                &mut recorder,
                &input,
                &weight,
                None,
                &out,
                shape,
                6,
            )
            .expect_err("the fused form must refuse this shape");
        assert!(
            format!("{err}").contains("fused small-oc"),
            "unexpected error for {:?}: {err}",
            shape.kernel
        );
    }

    // Too many output channels for the register block it is generated with.
    let mut recorder = Recorder::new(&gpu);
    let err = kernels
        .conv2d_small_oc_into(
            &gpu,
            &mut arena,
            &mut recorder,
            &input,
            &weight,
            None,
            &out,
            base,
            demucs_core::gpu::shaders::SMALL_OC_MAX + 1,
        )
        .expect_err("the fused form must refuse too many channels");
    assert!(format!("{err}").contains("fused small-oc"), "{err}");
}
