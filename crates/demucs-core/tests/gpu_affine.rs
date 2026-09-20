//! Validation of the fused per-channel affine + activation kernel.
//!
//! Every convolution block ends in `norm -> act`, and a BatchNorm in eval mode
//! is just an affine per channel, so this op is on the hot path of the conv
//! family. The reference is computed here in longhand, activation included.

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{Activation, Kernels};
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
            ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 4.0
        })
        .collect()
}

fn read(gpu: &Gpu, tensor: &DevTensor) -> Vec<f32> {
    let bytes = gpu
        .readback(&tensor.buffer, (tensor.len() * 4).max(4) as u64)
        .unwrap();
    bytemuck::cast_slice(&bytes).to_vec()
}

/// Exact-GELU reference; the kernel uses Abramowitz & Stegun's erf, so this is
/// the arbiter for the other three modes and reaches the same value to ~1e-7.
fn activate(value: f32, activation: Activation) -> f32 {
    match activation {
        Activation::Identity => value,
        Activation::Relu => value.max(0.0),
        Activation::Gelu => demucs_core::ops::gelu_erf(value),
        Activation::Silu => value / (1.0 + (-value).exp()),
    }
}

fn check(gpu: &Gpu, kernels: &Kernels, arena: &mut Arena, channels: usize, plane: usize, activation: Activation, label: &str) {
    let batch = 2usize;
    let values = fill(batch * channels * plane, 13);
    let scale_host = fill(channels, 17);
    let shift_host = fill(channels, 19);

    let x = arena
        .upload(gpu, &[batch, channels, plane], &values, "x")
        .unwrap();
    let scale = arena.upload(gpu, &[channels], &scale_host, "scale").unwrap();
    let shift = arena.upload(gpu, &[channels], &shift_host, "shift").unwrap();

    let mut recorder = Recorder::new(gpu);
    kernels
        .channel_affine_act_in_place(
            gpu, arena, &mut recorder, &x, &scale, &shift, batch, channels, plane, activation,
        )
        .unwrap();
    assert_eq!(recorder.dispatches(), 1);
    recorder.submit(gpu).unwrap();

    let expected: Vec<f32> = (0..batch * channels * plane)
        .map(|i| {
            let channel = (i / plane) % channels;
            activate(values[i] * scale_host[channel] + shift_host[channel], activation)
        })
        .collect();
    let actual = read(gpu, &x);
    let comparison = compare(&expected, &actual).unwrap();
    println!(
        "{label}: rms_relative={:.3e} max_abs={:.3e} (channels={channels} plane={plane})",
        comparison.rms_relative_error(),
        comparison.max_abs
    );
    // ReLU, SiLU and the identity are exact; GELU carries the kernel's erf
    // approximation, held to the alignment tolerance the models are.
    let tolerance = if activation == Activation::Gelu { 1e-5 } else { 1e-6 };
    assert!(
        comparison.rms_relative_error() < tolerance,
        "{label}: diverges (rms-relative {:.3e})",
        comparison.rms_relative_error()
    );
    arena.reset();
}

#[test]
fn channel_affine_with_activation_matches_the_host() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("affine pipeline compiles");
    let mut arena = Arena::new(&gpu, 128 << 20);

    // A DTTNet block's shape: 32 channels over a 64 x 96 plane.
    for (activation, label) in [
        (Activation::Relu, "relu (DTTNet blocks)"),
        (Activation::Gelu, "gelu"),
        (Activation::Silu, "silu"),
        (Activation::Identity, "identity (norm only)"),
    ] {
        check(&gpu, &kernels, &mut arena, 32, 64 * 96, activation, label);
    }
    // A ragged plane, where the channel boundary does not fall on a thread
    // boundary: the case an off-by-one in the index arithmetic would show up in.
    check(&gpu, &kernels, &mut arena, 5, 7, Activation::Relu, "ragged plane");
}
