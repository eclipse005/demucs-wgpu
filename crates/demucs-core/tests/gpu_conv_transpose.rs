//! Validation of the device `ConvTranspose2d` (padded copy + GEMM + col2im).
//!
//! The reference here is a longhand transposed convolution written out in this
//! file — the gather form, `out[oh, ow] += W[ic, oc, ky, kx] * x[iy, ix]` for
//! every `(iy, ix)` and tap that lands on `(oh, ow)` — so it shares no code with
//! either the tap GEMM or the col2im kernel. The host model's `conv.rs` uses the
//! scatter form and the device uses a gather; both have to agree with this.
//!
//! Two shapes are worth more than they look: `in_channels` not a multiple of
//! `BK` exercises the padded copy's *row* margin, and `positions_in` not a
//! multiple of `BN` exercises its *column* margin. Both margins are left
//! uninitialised by design (see `Kernels::conv_transpose2d_into`), so a case that
//! covers them is what proves the "0 * garbage = 0" argument in that comment.

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{pad_ceil, Col2ImShape, Kernels};
use demucs_core::gpu::shaders::{self, BK, BM, BN};
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

/// PyTorch's `(in_channels, out_channels, kh, kw)` weight rearranged into the
/// `(oc * kh * kw, ic)` matrix the GEMM consumes and zero-padded to its tiles.
fn padded_tap_matrix(
    weight: &[f32],
    in_channels: usize,
    out_channels: usize,
    kernel: (usize, usize),
) -> Vec<f32> {
    let (kh, kw) = kernel;
    let m = out_channels * kh * kw;
    let (rows, cols) = (pad_ceil(m, BM), pad_ceil(in_channels, BK));
    let mut padded = vec![0.0f32; rows * cols];
    for oc in 0..out_channels {
        for ky in 0..kh {
            for kx in 0..kw {
                let row = (oc * kh + ky) * kw + kx;
                for ic in 0..in_channels {
                    padded[row * cols + ic] =
                        weight[((ic * out_channels + oc) * kh + ky) * kw + kx];
                }
            }
        }
    }
    padded
}

/// Longhand transposed convolution: every input pixel scatters its `kh x kw`
/// taps, summed in f64 so the reference is not the thing being measured.
fn reference(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    shape: Col2ImShape,
) -> Vec<f32> {
    let (out_h, out_w) = shape.out_hw();
    let (kh, kw) = shape.kernel;
    let (sh, sw) = shape.stride;
    let positions_out = out_h * out_w;
    let mut acc = vec![0.0f64; shape.batch * shape.out_channels * positions_out];
    for batch in 0..shape.batch {
        for ic in 0..shape.in_channels {
            for iy in 0..shape.in_h {
                for ix in 0..shape.in_w {
                    let value = input
                        [((batch * shape.in_channels + ic) * shape.in_h + iy) * shape.in_w + ix]
                        as f64;
                    for oc in 0..shape.out_channels {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let tap =
                                    weight[((ic * shape.out_channels + oc) * kh + ky) * kw + kx]
                                        as f64;
                                let slot = ((batch * shape.out_channels + oc) * out_h
                                    + (iy * sh + ky))
                                    * out_w
                                    + (ix * sw + kx);
                                acc[slot] += value * tap;
                            }
                        }
                    }
                }
            }
        }
    }
    acc.into_iter()
        .enumerate()
        .map(|(index, value)| {
            let mut value = value as f32;
            if let Some(bias) = bias {
                value += bias[(index / positions_out) % shape.out_channels];
            }
            value
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn check(
    gpu: &Gpu,
    kernels: &Kernels,
    arena: &mut Arena,
    shape: Col2ImShape,
    with_bias: bool,
    label: &str,
) {
    let (in_channels, out_channels) = (shape.in_channels, shape.out_channels);
    let (kh, kw) = shape.kernel;
    let positions_in = shape.positions_in();
    let m = shape.m();
    let input_host = fill(shape.batch * in_channels * positions_in, 11);
    let weight_host = fill(in_channels * out_channels * kh * kw, 23);
    let bias_host = fill(out_channels, 31);

    let input = arena
        .upload(
            gpu,
            &[shape.batch, in_channels, shape.in_h, shape.in_w],
            &input_host,
            "x",
        )
        .unwrap();
    let weight = arena
        .upload(
            gpu,
            &[pad_ceil(m, BM), pad_ceil(in_channels, BK)],
            &padded_tap_matrix(&weight_host, in_channels, out_channels, shape.kernel),
            "w",
        )
        .unwrap();
    let bias = with_bias.then(|| {
        arena
            .upload(gpu, &[out_channels], &bias_host, "b")
            .unwrap()
    });
    let k_pad = pad_ceil(in_channels, BK);
    let pitch = pad_ceil(positions_in, BN);
    let padded_input = arena
        .tensor(gpu, &[shape.batch * k_pad, pitch], "input.padded")
        .unwrap();
    let taps = arena
        .tensor(gpu, &[shape.batch * pad_ceil(m, BM), pitch], "taps")
        .unwrap();
    let (out_h, out_w) = shape.out_hw();
    let out = arena
        .tensor(gpu, &[shape.batch, out_channels, out_h, out_w], "convt.out")
        .unwrap();

    // Poison every scratch with a sentinel: the copy's margins are documented as
    // never read into a stored element, so this is what would catch it if a
    // margin ever leaked into the arithmetic. `NaN` would poison the *products*
    // (the row margins do feed accumulators, harmless only because they multiply
    // zeros), so the sentinel is an ordinary large value instead.
    let sentinel = vec![1.0e30f32; padded_input.len()];
    gpu.upload(&padded_input.buffer, bytemuck::cast_slice(&sentinel));
    let sentinel = vec![-1.0e30f32; taps.len()];
    gpu.upload(&taps.buffer, bytemuck::cast_slice(&sentinel));

    let mut recorder = Recorder::new(gpu);
    kernels
        .conv_transpose2d_into(
            gpu,
            arena,
            &mut recorder,
            &input,
            &weight,
            bias.as_ref(),
            &padded_input,
            &taps,
            &out,
            shape,
        )
        .unwrap();
    // The bias rides the gather's own store when the fold is on, so the
    // per-batch row-bias pass is gone in that case.
    let bias_pass = usize::from(with_bias) * usize::from(!shaders::conv_bias_fuse());
    let expected_dispatches = 1 + shape.batch * (1 + 1 + bias_pass);
    assert_eq!(
        recorder.dispatches(),
        expected_dispatches,
        "one padded copy, then a GEMM and a gather per batch (the bias rides the gather)"
    );
    recorder.submit(gpu).unwrap();

    let expected = reference(&input_host, &weight_host, bias.as_ref().map(|_| &bias_host[..]), shape);
    let actual = read(gpu, &out);
    let comparison = compare(&expected, &actual).unwrap();
    println!(
        "{label}: rms_relative={:.3e} max_abs={:.3e} (ic={in_channels} oc={out_channels} \
         k={}x{} positions={} -> {}x{})",
        comparison.rms_relative_error(),
        comparison.max_abs,
        kh,
        kw,
        positions_in,
        out_h,
        out_w
    );
    assert!(
        comparison.rms_relative_error() < 1e-5,
        "{label}: conv_transpose2d diverges (rms-relative {:.3e})",
        comparison.rms_relative_error()
    );
    arena.reset();
}

#[test]
fn conv_transpose2d_matches_the_host_convolution() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("conv-transpose pipelines compile");
    let mut arena = Arena::new(&gpu, 512 << 20);

    // DTTNet's `us.*`: a 2x2 kernel at stride 2, where every output pixel
    // receives exactly one tap and the gather's parity test never rejects.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Col2ImShape {
            batch: 1,
            in_channels: 24,
            out_channels: 16,
            in_h: 20,
            in_w: 34,
            kernel: (2, 2),
            stride: (2, 2),
        },
        true,
        "2x2 stride 2 (DTTNet us)",
    );
    // `in_channels = 5` pads to 16 rows and `in_h * in_w = 143` pads to 256
    // columns: both margins of the padded copy are live here, and the kernel is
    // 3x3 at stride 1, so interior pixels sum nine taps.
    check(
        &gpu,
        &kernels,
        &mut arena,
        Col2ImShape {
            batch: 1,
            in_channels: 5,
            out_channels: 7,
            in_h: 13,
            in_w: 11,
            kernel: (3, 3),
            stride: (1, 1),
        },
        true,
        "3x3 stride 1, ragged, in_channels < BK",
    );
    // No bias, ragged on both axes, and a batch of two so the per-batch scratch
    // slicing is exercised: nothing about this divides evenly.
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
        false,
        "4x4 stride 3, ragged, batch 2, no bias",
    );
}

/// `copy_pitched` on its own, since the conv-transpose above only shows that its
/// margins cannot corrupt *that* op's output.
#[test]
fn copy_pitched_moves_rows_without_touching_the_margins() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("pipelines compile");
    let mut arena = Arena::new(&gpu, 64 << 20);

    let (batch, rows, cols, src_pitch, dst_rows, dst_pitch) = (3, 5, 7, 9, 16, 11);
    let source_host = fill(batch * rows * src_pitch, 7);
    let source = arena
        .upload(&gpu, &[batch, rows, src_pitch], &source_host, "src")
        .unwrap();
    let dest = arena
        .tensor(&gpu, &[batch * dst_rows, dst_pitch], "dst")
        .unwrap();
    let sentinel = vec![-7.0f32; dest.len()];
    gpu.upload(&dest.buffer, bytemuck::cast_slice(&sentinel));

    let mut recorder = Recorder::new(&gpu);
    kernels
        .copy_pitched_into(
            &gpu,
            &mut arena,
            &mut recorder,
            &source,
            &dest,
            batch,
            rows,
            cols,
            src_pitch,
            dst_pitch,
            dst_rows,
        )
        .unwrap();
    assert_eq!(recorder.dispatches(), 1);
    recorder.submit(&gpu).unwrap();

    let actual = read(&gpu, &dest);
    let mut mismatches = 0;
    for b in 0..batch {
        for r in 0..rows {
            for c in 0..dst_pitch {
                let got = actual[(b * dst_rows + r) * dst_pitch + c];
                let want = if c < cols {
                    source_host[(b * rows + r) * src_pitch + c]
                } else {
                    -7.0
                };
                if got != want {
                    mismatches += 1;
                }
            }
        }
        // The row margin inside each batch's block stays as it was.
        for r in rows..dst_rows {
            for c in 0..dst_pitch {
                if actual[(b * dst_rows + r) * dst_pitch + c] != -7.0 {
                    mismatches += 1;
                }
            }
        }
    }
    assert_eq!(mismatches, 0, "copy_pitched must be exact and leave margins alone");
}

/// `fill_zero`, which is the device-side clear every large scratch needs.
#[test]
fn fill_zero_clears_exactly_the_requested_range() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("pipelines compile");
    let mut arena = Arena::new(&gpu, 64 << 20);

    let count = 1000usize;
    let tensor = arena.tensor(&gpu, &[2048], "zeroed").unwrap();
    let sentinel = vec![3.5f32; tensor.len()];
    gpu.upload(&tensor.buffer, bytemuck::cast_slice(&sentinel));

    let mut recorder = Recorder::new(&gpu);
    kernels
        .fill_zero_into(&gpu, &mut arena, &mut recorder, &tensor, count)
        .unwrap();
    assert_eq!(recorder.dispatches(), 1);
    recorder.submit(&gpu).unwrap();

    let actual = read(&gpu, &tensor);
    assert!(actual[..count].iter().all(|v| *v == 0.0), "the range is zeroed");
    assert!(
        actual[count..].iter().all(|v| *v == 3.5),
        "and nothing past it is"
    );

    // An out-of-range request is an error rather than a silent short clear.
    let mut recorder = Recorder::new(&gpu);
    assert!(kernels
        .fill_zero_into(&gpu, &mut arena, &mut recorder, &tensor, 4096)
        .is_err());
}
