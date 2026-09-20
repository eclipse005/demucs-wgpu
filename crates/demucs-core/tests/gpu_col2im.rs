//! Validation of the col2im gather — the second half of `nn.ConvTranspose2d`.
//!
//! The indexing here is the subtle part: an output pixel receives a tap only
//! when the tap's offset is congruent to it modulo the stride, so an off-by-one
//! or a missing parity check shows up as taps landing one pixel over — a
//! plausible-looking but wrong upsampling. The reference below is a longhand
//! scatter (the form the kernel deliberately avoids), so the two disagree if
//! either the parity or the bounds logic is wrong.

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{pad_ceil, Col2ImShape, Kernels};
use demucs_core::gpu::shaders::{BM, BN};
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

/// Longhand scatter: every input pixel drops its `kh x kw` taps into the output,
/// which is what the kernel must reproduce exactly.
fn reference(taps: &[f32], shape: Col2ImShape, pitch: usize) -> Vec<f32> {
    let (out_h, out_w) = shape.out_hw();
    let (sh, sw) = shape.stride;
    let (kh, kw) = shape.kernel;
    let m = shape.m();
    let mut out = vec![0.0f32; shape.batch * shape.out_channels * out_h * out_w];
    for batch in 0..shape.batch {
        for channel in 0..shape.out_channels {
            for iy in 0..shape.in_h {
                for ix in 0..shape.in_w {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let row = channel * kh * kw + ky * kw + kx;
                            let value = taps[(batch * pad_ceil(m, BM) + row) * pitch
                                + iy * shape.in_w
                                + ix];
                            let oh = iy * sh + ky;
                            let ow = ix * sw + kx;
                            out[((batch * shape.out_channels + channel) * out_h + oh) * out_w + ow] +=
                                value;
                        }
                    }
                }
            }
        }
    }
    out
}

fn check(gpu: &Gpu, kernels: &Kernels, arena: &mut Arena, shape: Col2ImShape, label: &str) {
    let m = shape.m();
    let positions_in = shape.positions_in();
    let pitch = pad_ceil(positions_in, BN);
    let (out_h, out_w) = shape.out_hw();

    let taps_host = fill(shape.batch * pad_ceil(m, BM) * pitch, 29);
    let taps = arena
        .upload(gpu, &[shape.batch, pad_ceil(m, BM), pitch], &taps_host, "taps")
        .unwrap();
    let out = arena
        .tensor(
            gpu,
            &[shape.batch, shape.out_channels, out_h, out_w],
            "col2im.out",
        )
        .unwrap();

    let mut recorder = Recorder::new(gpu);
    kernels
        .col2im_into(gpu, arena, &mut recorder, &taps, &out, shape, pitch)
        .unwrap();
    assert_eq!(recorder.dispatches(), 1);
    recorder.submit(gpu).unwrap();

    let expected = reference(&taps_host, shape, pitch);
    let actual = read(gpu, &out);
    let comparison = compare(&expected, &actual).unwrap();
    println!(
        "{label}: rms_relative={:.3e} max_abs={:.3e} (m={m} positions_in={positions_in} out={out_h}x{out_w})",
        comparison.rms_relative_error(),
        comparison.max_abs
    );
    assert!(
        comparison.rms_relative_error() < 1e-5,
        "{label}: col2im diverges (rms-relative {:.3e})",
        comparison.rms_relative_error()
    );
    arena.reset();
}

#[test]
fn col2im_matches_the_host_scatter() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("col2im pipeline compiles");
    let mut arena = Arena::new(&gpu, 512 << 20);

    // DTTNet's `us.*`: a 2x2 kernel at stride 2, the case with no tap overlap.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Col2ImShape {
            batch: 1,
            in_channels: 96,
            out_channels: 64,
            in_h: 64,
            in_w: 512,
            kernel: (2, 2),
            stride: (2, 2),
        },
        "2x2 stride 2 (DTTNet us)",
    );
    // Stride 1 with a 3x3 kernel: every interior pixel sums nine taps, the case
    // where the parity check must *not* filter anything out.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Col2ImShape {
            batch: 1,
            in_channels: 32,
            out_channels: 32,
            in_h: 17,
            in_w: 23,
            kernel: (3, 3),
            stride: (1, 1),
        },
        "3x3 stride 1 (overlapping taps)",
    );
    // Ragged on both axes with a batch of two: nothing divides evenly.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Col2ImShape {
            batch: 2,
            in_channels: 8,
            out_channels: 6,
            in_h: 13,
            in_w: 11,
            kernel: (4, 4),
            stride: (3, 3),
        },
        "4x4 stride 3, ragged, batch 2",
    );
    // The last tdecoder's conv_tr at a real chunk: `out_w` is the sample count,
    // past what two parameters packed into one `u32` could carry.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Col2ImShape {
            batch: 1,
            in_channels: 16,
            out_channels: 2,
            in_h: 1,
            in_w: 86_000,
            kernel: (1, 4),
            stride: (1, 4),
        },
        "1x4 stride 4, out_w past 16 bits (tdecoder.3 at a real chunk)",
    );
}
