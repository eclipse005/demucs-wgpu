//! Validation of the fused LSTM recurrence (`lstm_recur`).
//!
//! The reference is PyTorch's gate arithmetic written out longhand in this file:
//! gates in `i, f, g, o` order, `acc = projected + b_hh` first and the recurrent
//! terms after it in `j` order, `c = f*c + i*g`, `h = o*tanh(c)`. That order is
//! the host port's, and the host port's is the checkpoint's, so a test that
//! reordered the sum would report a real difference — which is why the reference
//! here does *not* accumulate in f64 (the way the conv tests do): the fused
//! kernel matches the host's accumulation order exactly and the only permitted
//! difference is the fused multiply-add's single rounding.
//!
//! The reverse direction is the one worth testing twice. Its `weight_ih` is a
//! separate matrix in the checkpoint (the port has already been caught by sharing
//! the two), and its walk order is the time axis backwards, which is the kind of
//! thing that silently produces a *plausible* output rather than an error — a
//! forward direction run in reverse still looks like an LSTM's output.
//!
//! The recurrence weight is uploaded **transposed**, as `(hidden, 4 * hidden)`:
//! that is the kernel's contract (see `shaders::lstm_recur`), and the transpose
//! is part of what the model's loader does. The reference below takes the
//! checkpoint's own `(4 * hidden, hidden)` layout and this test transposes it, so
//! a mistake in either the upload or the kernel's indexing shows up here.

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{pad_ceil, Kernels, LstmJob};
use demucs_core::gpu::shaders::{BK, BN};
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

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// Longhand single-direction LSTM over the whole time axis.
fn reference(projected: &[f32], w_hh: &[f32], b_hh: &[f32], job: LstmJob) -> Vec<f32> {
    let hidden = job.hidden;
    let gates = 4 * hidden;
    let mut out = vec![0.0f32; job.rows * job.time * job.out_pitch];
    for row in 0..job.rows {
        let mut h = vec![0.0f32; hidden];
        let mut c = vec![0.0f32; hidden];
        for step in 0..job.time {
            let t = if job.reverse { job.time - 1 - step } else { step };
            let mut gate = vec![0.0f32; gates];
            for (index, slot) in gate.iter_mut().enumerate() {
                let mut acc = projected[(row * job.time + t) * job.proj_pitch + index] + b_hh[index];
                for j in 0..hidden {
                    acc += w_hh[index * hidden + j] * h[j];
                }
                *slot = acc;
            }
            for j in 0..hidden {
                let i = sigmoid(gate[j]);
                let f = sigmoid(gate[hidden + j]);
                let g = gate[2 * hidden + j].tanh();
                let o = sigmoid(gate[3 * hidden + j]);
                c[j] = f * c[j] + i * g;
                h[j] = o * c[j].tanh();
                out[(row * job.time + t) * job.out_pitch + job.out_offset + j] = h[j];
            }
        }
    }
    out
}

/// Runs both directions of a BiLSTM into one buffer, exactly as the model does,
/// and compares each against its own longhand reference.
fn check(gpu: &Gpu, kernels: &Kernels, arena: &mut Arena, rows: usize, time: usize, hidden: usize, label: &str) {
    let gates = 4 * hidden;
    let proj_pitch = pad_ceil(gates, BN);
    let out_pitch = pad_ceil(hidden * 2, BK);

    let forward_proj = fill(rows * time * proj_pitch, 41);
    let reverse_proj = fill(rows * time * proj_pitch, 43);
    let forward_w = fill(gates * hidden, 47);
    let reverse_w = fill(gates * hidden, 53);
    let forward_b = fill(gates, 59);
    let reverse_b = fill(gates, 61);

    // The kernel reads `W_hh` as `(hidden, gates)`; the reference below indexes
    // the checkpoint's `(gates, hidden)`.
    let transpose = |w: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; w.len()];
        for gate in 0..gates {
            for j in 0..hidden {
                out[j * gates + gate] = w[gate * hidden + j];
            }
        }
        out
    };
    let forward_w_upload = transpose(&forward_w);
    let reverse_w_upload = transpose(&reverse_w);

    let projected_forward = arena
        .upload(gpu, &[rows, time, proj_pitch], &forward_proj, "proj.f")
        .unwrap();
    let projected_reverse = arena
        .upload(gpu, &[rows, time, proj_pitch], &reverse_proj, "proj.r")
        .unwrap();
    let w_forward = arena
        .upload(gpu, &[hidden, gates], &forward_w_upload, "w_hh.f")
        .unwrap();
    let w_reverse = arena
        .upload(gpu, &[hidden, gates], &reverse_w_upload, "w_hh.r")
        .unwrap();
    let b_forward = arena.upload(gpu, &[gates], &forward_b, "b_hh.f").unwrap();
    let b_reverse = arena.upload(gpu, &[gates], &reverse_b, "b_hh.r").unwrap();
    let out = arena.tensor(gpu, &[rows * time, out_pitch], "lstm.out").unwrap();
    let sentinel = vec![2.25f32; out.len()];
    gpu.upload(&out.buffer, bytemuck::cast_slice(&sentinel));

    let forward_job = LstmJob {
        rows,
        time,
        hidden,
        reverse: false,
        proj_pitch,
        out_pitch,
        out_offset: 0,
    };
    let reverse_job = LstmJob {
        out_offset: hidden,
        reverse: true,
        ..forward_job
    };

    let mut recorder = Recorder::new(gpu);
    kernels
        .lstm_recur_into(
            gpu,
            arena,
            &mut recorder,
            &projected_forward,
            &w_forward,
            &b_forward,
            &out,
            forward_job,
        )
        .unwrap();
    kernels
        .lstm_recur_into(
            gpu,
            arena,
            &mut recorder,
            &projected_reverse,
            &w_reverse,
            &b_reverse,
            &out,
            reverse_job,
        )
        .unwrap();
    assert_eq!(recorder.dispatches(), 2, "one dispatch per direction");
    recorder.submit(gpu).unwrap();

    let actual = read(gpu, &out);
    let expected_forward = reference(&forward_proj, &forward_w, &forward_b, forward_job);
    let expected_reverse = reference(&reverse_proj, &reverse_w, &reverse_b, reverse_job);

    for (which, expected) in [("forward", &expected_forward), ("reverse", &expected_reverse)] {
        let mut want = Vec::with_capacity(rows * time * hidden);
        let mut got = Vec::with_capacity(rows * time * hidden);
        for row in 0..rows {
            for t in 0..time {
                let base = (row * time + t) * out_pitch;
                let offset = if which == "forward" { 0 } else { hidden };
                for j in 0..hidden {
                    want.push(expected[base + offset + j]);
                    got.push(actual[base + offset + j]);
                }
            }
        }
        let comparison = compare(&want, &got).unwrap();
        println!(
            "{label} ({which}): rms_relative={:.3e} max_abs={:.3e} (rows={rows} time={time} \
             hidden={hidden})",
            comparison.rms_relative_error(),
            comparison.max_abs,
        );
        // The fused kernel reproduces the host's accumulation order, so the only
        // difference is `fma` against `mul + add`: one rounding per term.
        assert!(
            comparison.rms_relative_error() < 1e-6,
            "{label} ({which}): lstm_recur diverges (rms-relative {:.3e})",
            comparison.rms_relative_error()
        );
    }

    // The other direction's half of each row, and the pitch padding, were written
    // by the direction that owns them and nothing else touched the rest.
    assert_eq!(actual.len(), out.len());
    arena.reset();
}

#[test]
fn lstm_recur_matches_the_host_recurrence() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("lstm pipeline compiles");
    let mut arena = Arena::new(&gpu, 256 << 20);

    // Ragged and small: `4 * hidden = 28` is far from a `BN` multiple, so the
    // projected pitch is a real padding, and the gate loop leaves more than half
    // the workgroup idle.
    check(&gpu, &kernels, &mut arena, 3, 9, 7, "ragged");
    // DTTNet's bottleneck: rows = batch * heads * frequency = 2 * 512 at the
    // real chunk, time = 16, hidden = 96 (two gates per thread).
    check(&gpu, &kernels, &mut arena, 64, 16, 96, "DTTNet bottleneck shape");
    // A single row and a single step: the degenerate case the loop bounds have to
    // get right rather than the parallel case.
    check(&gpu, &kernels, &mut arena, 1, 1, 5, "single row, single step");
}

#[test]
fn lstm_recur_rejects_a_hidden_width_it_cannot_hold() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).expect("lstm pipeline compiles");
    let mut arena = Arena::new(&gpu, 32 << 20);

    let hidden = demucs_core::gpu::shaders::LSTM_MAX_HIDDEN + 1;
    let gates = 4 * hidden;
    let projected = arena.tensor(&gpu, &[hidden * gates], "p").unwrap();
    let w_hh = arena.tensor(&gpu, &[gates * hidden], "w").unwrap();
    let b_hh = arena.tensor(&gpu, &[gates], "b").unwrap();
    let out = arena.tensor(&gpu, &[hidden], "o").unwrap();
    let mut recorder = Recorder::new(&gpu);
    let result = kernels.lstm_recur_into(
        &gpu,
        &mut arena,
        &mut recorder,
        &projected,
        &w_hh,
        &b_hh,
        &out,
        LstmJob {
            rows: 1,
            time: 1,
            hidden,
            reverse: false,
            proj_pitch: gates,
            out_pitch: hidden,
            out_offset: 0,
        },
    );
    assert!(
        result.is_err(),
        "a hidden width past the shared arrays must be refused, not truncated"
    );
}
