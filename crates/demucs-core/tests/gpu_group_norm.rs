//! Validation of the device group normalisation.
//!
//! The reference is a longhand `nn.GroupNorm` written out in this file, in f64 so
//! the comparison measures the kernel rather than the reference's own rounding.
//! The statistics use the host's formula — `mean = sum/n` and
//! `var = max(sum_sq/n - mean^2, 0)`, one pass over the slice — because that is
//! the form the port is aligned against PyTorch with; a reference that used
//! Welford's algorithm would be *more* accurate and would report a difference
//! that the host path has too.
//!
//! The layouts matter as much as the arithmetic. The DTTNet band sequence
//! normalises a slice whose sequence axis is not contiguous — the channels are
//! the innermost run — so a test that only covered `(rows, channels, len)`
//! contiguous would not exercise the stride path the model actually takes.

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{GroupNormShape, Kernels};
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

/// Longhand `nn.GroupNorm` over the described slice, in f64.
fn reference(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    shape: GroupNormShape,
) -> Vec<f32> {
    let GroupNormShape {
        rows,
        groups,
        per_group,
        len,
        row_stride,
        len_stride,
        channel_stride,
        eps,
    } = shape;
    let index = |row: usize, channel: usize, position: usize| {
        row * row_stride + channel * channel_stride + position * len_stride
    };
    let mut out = vec![0.0f32; shape.needed()];
    for row in 0..rows {
        for group in 0..groups {
            let group_size = (per_group * len) as f64;
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            for local in 0..per_group * len {
                let channel = group * per_group + local / len;
                let position = local % len;
                let value = x[index(row, channel, position)] as f64;
                sum += value;
                sum_sq += value * value;
            }
            let mean = sum / group_size;
            let variance = (sum_sq / group_size - mean * mean).max(0.0);
            let scale = 1.0f64 / (variance + eps as f64).sqrt();
            for local in 0..per_group * len {
                let local_channel = local / len;
                let position = local % len;
                let channel = group * per_group + local_channel;
                let offset = index(row, channel, position);
                out[offset] = ((x[offset] as f64 - mean) * scale * gamma[channel] as f64
                    + beta[channel] as f64) as f32;
            }
        }
    }
    out
}

fn check(
    gpu: &Gpu,
    kernels: &Kernels,
    arena: &mut Arena,
    shape: GroupNormShape,
    label: &str,
) {
    let needed = shape.needed();
    let channels = shape.channels();
    let x_host = fill(needed, 13);
    let gamma_host = fill(channels, 29);
    let beta_host = fill(channels, 37);

    let x = arena.upload(gpu, &[needed], &x_host, "x").unwrap();
    let gamma = arena.upload(gpu, &[channels], &gamma_host, "gamma").unwrap();
    let beta = arena.upload(gpu, &[channels], &beta_host, "beta").unwrap();
    let out = arena.tensor(gpu, &[needed], "group_norm.out").unwrap();
    let sentinel = vec![9.0f32; needed];
    gpu.upload(&out.buffer, bytemuck::cast_slice(&sentinel));

    let mut recorder = Recorder::new(gpu);
    kernels
        .group_norm_into(gpu, arena, &mut recorder, &x, &gamma, &beta, &out, shape)
        .unwrap();
    // One dispatch for a slice one workgroup can walk; three (per-segment
    // reduce, per-pair stats, apply) once the slice is cut into segments.
    // `DEMUCS_GN_COMBINE=0` drops the stats pass and the apply re-sums.
    let split = shape.per_group * shape.len > 4096;
    let combine = demucs_core::gpu::shaders::group_norm_combine_stats();
    let want = if !split {
        1
    } else if combine {
        3
    } else {
        2
    };
    assert_eq!(
        recorder.dispatches(),
        want,
        "a slice of {} elements splits into segments: {} dispatches",
        shape.per_group * shape.len,
        want,
    );
    recorder.submit(gpu).unwrap();

    let expected = reference(&x_host, &gamma_host, &beta_host, shape);
    let actual = read(gpu, &out);
    // Only the elements the layout touches are compared; the rest is the
    // padding between rows, which the op must leave alone.
    let touched: Vec<usize> = (0..shape.rows)
        .flat_map(|row| {
            (0..channels).flat_map(move |channel| {
                (0..shape.len).map(move |position| {
                    row * shape.row_stride + channel * shape.channel_stride + position * shape.len_stride
                })
            })
        })
        .collect();
    let expected_values: Vec<f32> = touched.iter().map(|i| expected[*i]).collect();
    let actual_values: Vec<f32> = touched.iter().map(|i| actual[*i]).collect();
    let comparison = compare(&expected_values, &actual_values).unwrap();
    println!(
        "{label}: rms_relative={:.3e} max_abs={:.3e} (rows={} groups={} per_group={} len={} \
         len_stride={})",
        comparison.rms_relative_error(),
        comparison.max_abs,
        shape.rows,
        shape.groups,
        shape.per_group,
        shape.len,
        shape.len_stride,
    );
    assert!(
        comparison.rms_relative_error() < 1e-6,
        "{label}: group_norm diverges (rms-relative {:.3e})",
        comparison.rms_relative_error()
    );
    // The layout's padding — every element outside `touched` — must be left
    // alone, still carrying the sentinel. A sorted merge-walk rather than
    // `touched.contains`: the waveform DConv case touches 8.2 million
    // elements, and asking a Vec that size per index is quadratic (it hangs).
    let mut sorted = touched.clone();
    sorted.sort_unstable();
    sorted.dedup();
    let mut cursor = 0usize;
    for index in 0..needed {
        while cursor < sorted.len() && sorted[cursor] < index {
            cursor += 1;
        }
        if cursor < sorted.len() && sorted[cursor] == index {
            continue;
        }
        assert_eq!(
            actual[index], 9.0,
            "element {index} is outside the layout but was written"
        );
    }
    arena.reset();
}

#[test]
fn group_norm_matches_the_host_normalisation() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("group-norm pipeline compiles");
    let mut arena = Arena::new(&gpu, 128 << 20);

    // A plain contiguous `(rows, channels, len)`: the `len_stride = 1` case, and
    // ragged on every axis.
    check(
        &gpu,
        &kernels,
        &mut arena,
        GroupNormShape {
            rows: 7,
            groups: 3,
            per_group: 4,
            len: 13,
            row_stride: 12 * 13,
            len_stride: 1,
            channel_stride: 13,
            eps: 1e-5,
        },
        "contiguous, ragged",
    );
    // The same, with rows wider than the data they hold: the padding between rows
    // must survive untouched, which is what the sentinel check below is for.
    check(
        &gpu,
        &kernels,
        &mut arena,
        GroupNormShape {
            rows: 7,
            groups: 3,
            per_group: 4,
            len: 13,
            row_stride: 200,
            len_stride: 1,
            channel_stride: 13,
            eps: 1e-5,
        },
        "contiguous, padded row pitch",
    );
    // DTTNet's `across_t` module: rows are `(batch * heads, freq)`, the channels
    // are the innermost run of `per_head = 48`, and the sequence is the time axis
    // with stride `per_head`.
    check(
        &gpu,
        &kernels,
        &mut arena,
        GroupNormShape {
            rows: 32,
            groups: 3,
            per_group: 16,
            len: 16,
            row_stride: 16 * 48,
            len_stride: 48,
            channel_stride: 1,
            eps: 1e-5,
        },
        "across_t (channels innermost, len = time)",
    );
    // DTTNet's `across_k` module: same channels, but the sequence is the
    // frequency axis — 512 long at the bottleneck, and a group's slice is 8192
    // elements, which is the case that walks its slice more than once per thread.
    check(
        &gpu,
        &kernels,
        &mut arena,
        GroupNormShape {
            rows: 5,
            groups: 3,
            per_group: 16,
            len: 512,
            row_stride: 512 * 48,
            len_stride: 48,
            channel_stride: 1,
            eps: 1e-5,
        },
        "across_k (len = frequency, multi-walk)",
    );
    // The waveform branch's DConv at a real segment: one row, one group, 8.2
    // million elements in the slice — the shape whose single-workgroup form
    // measured 23.7 ms, and the reason the split path exists.
    check(
        &gpu,
        &kernels,
        &mut arena,
        GroupNormShape {
            rows: 1,
            groups: 1,
            per_group: 96,
            len: 86_000,
            row_stride: 96 * 86_000,
            len_stride: 1,
            channel_stride: 86_000,
            eps: 1e-5,
        },
        "waveform DConv at a real segment (8.2M-element slice)",
    );
}
