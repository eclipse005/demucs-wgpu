//! The implicit-GEMM conv form (`DEMUCS_CONV_DIRECT`): the gather folded into
//! the GEMM's `B` staging, so no patch matrix is materialised.
//!
//! It lives in its own binary because the form is selected from an environment
//! variable read at call time: a sibling test in `gpu_conv2d.rs` asserts the
//! materialised form's dispatch count, and a parallel run would see the flag
//! flip under it.

use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{pad_ceil, Kernels};
use demucs_core::gpu::shaders::{BM, BK, BN};
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
        .map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32 / (u32::MAX as f32) - 0.5)
        .collect()
}

fn read(gpu: &Gpu, tensor: &DevTensor) -> Vec<f32> {
    let bytes = gpu.readback(&tensor.buffer, (tensor.len() * 4) as u64).unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

/// Longhand convolution with a per-output-channel bias.
fn reference(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    in_channels: usize,
    h: usize,
    w: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    out_channels: usize,
) -> Vec<f32> {
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = pad;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let positions = out_h * out_w;
    let mut out = vec![0.0f32; batch * out_channels * positions];
    for b in 0..batch {
        for oc in 0..out_channels {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = bias[oc];
                    for ic in 0..in_channels {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy * sh + ky;
                                let ix = ox * sw + kx;
                                if iy < ph || ix < pw || iy >= ph + h || ix >= pw + w {
                                    continue;
                                }
                                let pixel = input[((b * in_channels + ic) * h + (iy - ph)) * w + (ix - pw)];
                                let tap = weight[oc * (in_channels * kh * kw) + (ic * kh + ky) * kw + kx];
                                acc += pixel * tap;
                            }
                        }
                    }
                    out[(b * out_channels + oc) * positions + oy * out_w + ox] = acc;
                }
            }
        }
    }
    out
}

/// A multi-batch conv must gather from each batch's own block of the input.
///
/// The direct form re-derives the gather inside the GEMM, and the unbatched
/// variant only ever read batch 0's input — with a batch of two that silently
/// computes batch 0 twice. The batched variant offsets the gather by each
/// batch's block, which is what this checks: both forms against the host
/// reference, and against each other bit for bit.
///
/// The width is 258 with no padding so `out_w = 256` is a whole number of `BN`
/// tiles: the form requires that a tile's positions sit in one output row, and
/// an `out_w` that straddles (200, say) reads the wrong row's input past the
/// boundary. `conv2d_into` refuses those widths rather than guessing.
#[test]
fn the_batched_conv_direct_matches_the_materialised_form() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let (batch, in_channels, h, w) = (2usize, 4usize, 8usize, 258usize);
    let kernel = (3usize, 3usize);
    let stride = (1usize, 1usize);
    let pad = (0usize, 0usize);
    let out_channels = 6usize;
    let shape = demucs_core::gpu::kernels::Im2ColShape {
        batch,
        in_channels,
        h,
        w,
        kernel,
        stride,
        pad,
    };
    let k = shape.k();
    let positions = shape.positions();
    let input_host = fill(batch * in_channels * h * w, 11);
    let weight_host = fill(out_channels * k, 23);
    // The weight as the model uploads it: `(out_channels, k)` zero-padded to the
    // GEMM's A tiles, row by row — padding the whole vector instead would spill
    // row 1's weights into row 0's tail.
    let k_pad = pad_ceil(k, BK);
    let mut padded = vec![0.0f32; pad_ceil(out_channels, BM) * k_pad];
    for row in 0..out_channels {
        padded[row * k_pad..row * k_pad + k].copy_from_slice(&weight_host[row * k..(row + 1) * k]);
    }
    let bias_host = fill(out_channels, 31);

    let mut arena = Arena::new(&gpu, 256 << 20);
    let kernels = Kernels::new(&gpu).unwrap();
    let input = arena
        .upload(&gpu, &[batch, in_channels, h, w], &input_host, "x")
        .unwrap();
    let weight = arena
        .upload(&gpu, &[pad_ceil(out_channels, BM), k_pad], &padded, "w")
        .unwrap();
    let bias = arena.upload(&gpu, &[out_channels], &bias_host, "b").unwrap();
    let pitch = pad_ceil(positions, BN);
    let patches = arena
        .tensor(&gpu, &[batch * k_pad, pitch], "patches")
        .unwrap();
    let out = arena
        .tensor(&gpu, &[batch, out_channels, positions], "conv.out")
        .unwrap();

    let mut run = |direct: bool| -> Vec<f32> {
        // The form is read from the environment at call time.
        std::env::set_var("DEMUCS_CONV_DIRECT", if direct { "1" } else { "0" });
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
        read(&gpu, &out)
    };
    let materialised = run(false);
    let direct = run(true);
    std::env::remove_var("DEMUCS_CONV_DIRECT");

    let expected = reference(
        &input_host,
        &weight_host,
        &bias_host,
        batch,
        in_channels,
        h,
        w,
        kernel,
        stride,
        pad,
        out_channels,
    );
    let worst = expected
        .iter()
        .zip(materialised.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "the materialised form is off by {worst:.3e}");
    let drift = materialised
        .iter()
        .zip(direct.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        drift, 0.0,
        "the batched direct form differs from the materialised one by {drift:.3e}"
    );
}
