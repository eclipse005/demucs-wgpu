//! Validation of the axial attention op against the CPU reference.
//!
//! Attention is where the layouts get subtle: Q, K and V are three offset views
//! of one fused tensor, the score matrix is `Q @ K^T` with K read transposed out
//! of `[frame][dim_head]` storage, and the batches are indexed by `(band, head)`
//! with two different strides. If any of that is off by one the result is
//! plausible-looking and wrong, so it is checked against a straightforward host
//! implementation of the same maths.

use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, DevTensor, Recorder};
use demucs_core::gpu::kernels::{AttentionShape, Kernels};
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
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
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

/// The reference's probabilities, i.e. the post-softmax score tensor, laid out
/// `(band, head, frame, key)`.
fn reference_probs(qkv: &[f32], shape: AttentionShape) -> Vec<f32> {
    let mut probs = reference_scores(qkv, shape);
    let frames = shape.frames;
    for row in probs.chunks_mut(frames) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0f32;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            total += *value;
        }
        for value in row.iter_mut() {
            *value /= total;
        }
    }
    probs
}

/// The reference's raw scores, `scale * Q K^T`, laid out `(band, head, frame,
/// key)` — kept separate so the test can check the `Q K^T` dispatch on its own.
fn reference_scores(qkv: &[f32], shape: AttentionShape) -> Vec<f32> {
    let AttentionShape {
        bands,
        frames,
        heads,
        dim_head,
    } = shape;
    let inner = shape.inner();
    let row = shape.row();
    let scale = 1.0 / (dim_head as f32).sqrt();
    let mut probs = vec![0.0f32; bands * heads * frames * frames];

    for band in 0..bands {
        for head in 0..heads {
            for i in 0..frames {
                let base = (band * heads + head) * frames * frames + i * frames;
                for j in 0..frames {
                    let mut acc = 0.0f32;
                    for d in 0..dim_head {
                        let qi = band * frames * row + i * row + head * dim_head + d;
                        let ki = band * frames * row + j * row + inner + head * dim_head + d;
                        acc += qkv[qi] * qkv[ki];
                    }
                    probs[base + j] = acc * scale;
                }
            }
        }
    }
    probs
}

/// The reference's attention output, derived from a probability tensor the caller
/// supplies — which lets the test feed the GPU's own probabilities back in and so
/// separate a scores/softmax bug from an AV bug.
fn reference_output_from_probs(qkv: &[f32], probs: &[f32], shape: AttentionShape) -> Vec<f32> {
    let AttentionShape {
        bands,
        frames,
        heads,
        dim_head,
    } = shape;
    let inner = shape.inner();
    let row = shape.row();
    let mut out = vec![0.0f32; bands * frames * inner];
    for band in 0..bands {
        for head in 0..heads {
            for i in 0..frames {
                let pbase = (band * heads + head) * frames * frames + i * frames;
                for d in 0..dim_head {
                    let mut acc = 0.0f32;
                    for j in 0..frames {
                        let vi = band * frames * row + j * row + 2 * inner + head * dim_head + d;
                        acc += probs[pbase + j] * qkv[vi];
                    }
                    out[band * frames * inner + i * inner + head * dim_head + d] = acc;
                }
            }
        }
    }
    out
}

/// The reference's own attention: `softmax(Q K^T / sqrt(d)) V` over the fused
/// layout, written out as `(bands, frames, heads * dim_head)`.
fn reference(qkv: &[f32], shape: AttentionShape) -> Vec<f32> {
    let probs = reference_probs(qkv, shape);
    reference_output_from_probs(qkv, &probs, shape)
}

fn run_case(bands: usize, frames: usize, heads: usize, dim_head: usize, tolerance: f32) {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 1 << 30);

    let shape = AttentionShape {
        bands,
        frames,
        heads,
        dim_head,
    };
    let qkv_host = fill(bands * frames * shape.row(), 3);
    let qkv = arena
        .upload(&gpu, &[bands * frames * shape.row()], &qkv_host, "qkv")
        .unwrap();
    let scores = arena.tensor(&gpu, &[shape.scores_len()], "scores").unwrap();
    let out = arena.tensor(&gpu, &[shape.output_len()], "attn.out").unwrap();

    let mut recorder = Recorder::new(&gpu);
    kernels
        .attention(&gpu, &mut arena, &mut recorder, &qkv, &scores, &out, shape)
        .unwrap();
    assert_eq!(recorder.dispatches(), 3, "scores, softmax, AV");
    recorder.submit(&gpu).unwrap();

    let expected = reference(&qkv_host, shape);
    let actual = read(&gpu, &out);
    let comparison = compare(&expected, &actual).unwrap();
    println!(
        "attention (bands={bands} frames={frames} heads={heads} d={dim_head}): \
         rms_relative={:.3e} max_abs={:.3e}",
        comparison.rms_relative_error(),
        comparison.max_abs
    );
    if comparison.rms_relative_error() >= tolerance {
        // Which half is wrong: the probabilities, or the weighted sum of V?
        // The scratch is pitched to the GEMM's reduction step, so its rows are
        // `pitch` wide and have to be gathered back to the logical `frames`.
        let pitch = shape.scores_pitch();
        let padded = read(&gpu, &scores);
        let mut gpu_probs = Vec::with_capacity(bands * heads * frames * frames);
        for block in 0..bands * heads {
            let base = block * frames * pitch;
            for row in 0..frames {
                let start = base + row * pitch;
                gpu_probs.extend_from_slice(&padded[start..start + frames]);
            }
        }
        let expected_probs = reference_probs(&qkv_host, shape);
        let prob_comparison = compare(&expected_probs, &gpu_probs).unwrap();
        // Where the probabilities disagree, by (key column) and (row) — a bad
        // band of columns points at the pitch padding, a bad row at the tiling.
        let mut worst = (0.0f32, 0usize);
        let mut over = 0usize;
        let mut bad_rows = std::collections::BTreeMap::new();
        let mut bad_keys = std::collections::BTreeMap::new();
        for (index, (want, got)) in expected_probs.iter().zip(gpu_probs.iter()).enumerate() {
            let diff = (want - got).abs();
            if diff > worst.0 {
                worst = (diff, index);
            }
            if diff > 1e-5 {
                over += 1;
                *bad_rows.entry((index / frames) % frames).or_insert(0usize) += 1;
                *bad_keys.entry(index % frames).or_insert(0usize) += 1;
            }
        }
        let key = worst.1 % frames;
        let row = (worst.1 / frames) % frames;
        let block = worst.1 / (frames * frames);
        println!(
            "  worst probability: |diff|={:.3e} at block {block} row {row} key {key} \
             (want {:.6e} got {:.6e})",
            worst.0, expected_probs[worst.1], gpu_probs[worst.1]
        );
        println!(
            "  entries over 1e-5: {over} of {} ({} rows, {} keys)",
            expected_probs.len(),
            bad_rows.len(),
            bad_keys.len()
        );
        let show = |map: &std::collections::BTreeMap<usize, usize>, what: &str| {
            let mut list: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
            list.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
            let head: Vec<String> = list.iter().take(8).map(|(k, v)| format!("{k}x{v}")).collect();
            println!("  worst {what}: {}", head.join(" "));
        };
        show(&bad_rows, "rows");
        show(&bad_keys, "keys");
        println!(
            "  post-softmax scores: rms_relative={:.3e} expected[0..3]={:?} got[0..3]={:?}",
            prob_comparison.rms_relative_error(),
            &expected_probs[0..3],
            &gpu_probs[0..3]
        );
        // Feed the GPU's own probabilities through the reference AV: if this
        // matches the GPU output, the AV step is right and the scores are not.
        let from_gpu_probs = reference_output_from_probs(&qkv_host, &gpu_probs, shape);
        let av_comparison = compare(&from_gpu_probs, &actual).unwrap();
        println!(
            "  AV given the GPU's probabilities: rms_relative={:.3e}",
            av_comparison.rms_relative_error()
        );

        let per_band = frames * shape.inner();
        for band in 0..bands.min(4) {
            let range = band * per_band..(band + 1) * per_band;
            let band_comparison = compare(&expected[range.clone()], &actual[range]).unwrap();
            println!(
                "  band {band}: rms_relative={:.3e} expected[0..3]={:?} got[0..3]={:?}",
                band_comparison.rms_relative_error(),
                &expected[band * per_band..band * per_band + 3],
                &actual[band * per_band..band * per_band + 3]
            );
        }
    }
    assert!(
        comparison.rms_relative_error() < tolerance,
        "attention diverges: rms-relative {:.3e}",
        comparison.rms_relative_error()
    );
}

#[test]
fn rope_matches_the_cpu_reference_on_the_fused_qkv_layout() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 64 << 20);

    let bands = 3usize;
    let frames = 200usize;
    let heads = 2usize;
    let dim_head = 8usize;
    let inner = heads * dim_head;
    let row = 3 * inner;

    let qkv_host = fill(bands * frames * row, 9);
    let freqs_host = fill(dim_head / 2, 11);
    let qkv = arena
        .upload(&gpu, &[bands * frames * row], &qkv_host, "qkv")
        .unwrap();
    let freqs = arena.upload(&gpu, &[dim_head / 2], &freqs_host, "freqs").unwrap();

    let mut recorder = Recorder::new(&gpu);
    // Q is heads 0..heads, K is heads heads..2*heads.
    kernels
        .rope_heads(&gpu, &mut arena, &mut recorder, &qkv, &freqs, bands, frames, heads, dim_head, 0, heads)
        .unwrap();
    kernels
        .rope_heads(&gpu, &mut arena, &mut recorder, &qkv, &freqs, bands, frames, heads, dim_head, heads, heads)
        .unwrap();
    assert_eq!(recorder.dispatches(), 2);
    recorder.submit(&gpu).unwrap();

    // The reference rotates Q and K in place with the same interleaved form.
    let mut expected = qkv_host.clone();
    for band in 0..bands {
        for t in 0..frames {
            for head in 0..(2 * heads) {
                let base = band * frames * row + t * row + head * dim_head;
                for k in 0..dim_head / 2 {
                    let angle = t as f32 * freqs_host[k];
                    let (c, s) = (angle.cos(), angle.sin());
                    let a = expected[base + 2 * k];
                    let b = expected[base + 2 * k + 1];
                    expected[base + 2 * k] = a * c - b * s;
                    expected[base + 2 * k + 1] = a * s + b * c;
                }
            }
        }
    }
    let comparison = compare(&expected, &read(&gpu, &qkv)).unwrap();
    println!(
        "rope on fused qkv: rms_relative={:.3e}",
        comparison.rms_relative_error()
    );
    assert!(comparison.rms_relative_error() < 1e-5);
}

#[test]
fn transpose_matches_a_host_transpose() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 64 << 20);

    // Deliberately ragged: neither side is a multiple of the 32-wide tile.
    for (rows, cols) in [(60usize, 801usize), (801, 60), (33, 33), (1, 100)] {
        let host = fill(rows * cols, 17);
        let x = arena.upload(&gpu, &[rows * cols], &host, "x").unwrap();
        let out = arena.tensor(&gpu, &[rows * cols], "transposed").unwrap();
        let mut recorder = Recorder::new(&gpu);
        kernels
            .transpose(&gpu, &mut arena, &mut recorder, &x, &out, 1, rows, cols, 1)
            .unwrap();
        recorder.submit(&gpu).unwrap();

        let mut expected = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                expected[c * rows + r] = host[r * cols + c];
            }
        }
        let comparison = compare(&expected, &read(&gpu, &out)).unwrap();
        println!(
            "transpose {rows}x{cols}: rms_relative={:.3e}",
            comparison.rms_relative_error()
        );
        assert!(comparison.rms_relative_error() < 1e-6);
    }
}

#[test]
fn attention_matches_the_cpu_reference_for_the_frequency_axis() {
    // The frequency transformer: 60 bands is the sequence, 801 frames is the batch.
    run_case(801, 60, 8, 64, 1e-5);
}

#[test]
fn attention_matches_the_cpu_reference_for_the_time_axis() {
    // The time transformer, at a reduced band count so the test stays quick: the
    // full shape is 60 bands x 801 frames, and the score tensor for that is the
    // multi-gigabyte one the model allocates once.
    run_case(4, 801, 8, 64, 1e-5);
}

/// The grouped attention path: `groups` bands walked per dispatch out of one
/// shared score scratch. htdemucs runs attention with `groups = 1`, so nothing
/// in this model reaches it — it is here because the batched kernel still
/// carries the per-workgroup offsets, and the single-group cases above cannot
/// see a bug that only shows up once the scratch is reused across groups or the
/// QKV/out slices walk.
#[test]
fn attention_in_groups_matches_the_cpu_reference() {
    const GROUP: usize = 8;

    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 1 << 30);
    let (bands, frames, heads, dim_head) = (60usize, 61usize, 8usize, 64usize);
    let shape = AttentionShape { bands, frames, heads, dim_head };
    let inner = shape.inner();
    let row = shape.row();
    let pitch = shape.scores_pitch();
    let group = GROUP.min(bands);

    let qkv_host = fill(bands * frames * row, 3);
    let qkv = arena
        .upload(&gpu, &[bands * frames * row], &qkv_host, "qkv")
        .unwrap();
    let scratch = arena
        .tensor(&gpu, &[group * heads * frames * pitch], "scores")
        .unwrap();
    let out = arena.tensor(&gpu, &[bands * frames * inner], "attn.out").unwrap();

    let mut recorder = Recorder::new(&gpu);
    for start in (0..bands).step_by(group) {
        let take = group.min(bands - start);
        let qkv_part = qkv.slice(start * frames * row, vec![take * frames * row]).unwrap();
        let out_part = out.slice(start * frames * inner, vec![take * frames * inner]).unwrap();
        let out_part = DevTensor {
            buffer: out_part.buffer.clone(),
            offset: out_part.offset,
            shape: vec![take, frames, inner],
        };
        kernels
            .attention(
                &gpu,
                &mut arena,
                &mut recorder,
                &qkv_part,
                &scratch,
                &out_part,
                AttentionShape { bands: take, frames, heads, dim_head },
            )
            .unwrap();
    }
    recorder.submit(&gpu).unwrap();

    let expected = reference(&qkv_host, shape);
    let actual = read(&gpu, &out);
    let comparison = compare(&expected, &actual).unwrap();
    println!(
        "attention in groups of {group} (bands={bands} frames={frames}): \
         rms_relative={:.3e} max_abs={:.3e}",
        comparison.rms_relative_error(),
        comparison.max_abs
    );
    assert!(comparison.rms_relative_error() < 1e-5);
}

/// The `Q K^T` dispatch on its own, against the same host reference.
///
/// The full attention test can only say that the fused path is wrong; this says
/// whether the scores themselves are, which is the difference between a GEMM
/// addressing bug and a softmax one.
#[test]
fn qk_matches_the_cpu_reference() {
    use demucs_core::gpu::kernels::GemmJob;

    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 1 << 30);
    let env = |key: &str, fallback: usize| {
        std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(fallback)
    };
    let (bands, frames, heads, dim_head) = (
        env("DEMUCS_QK_BANDS", 4),
        env("DEMUCS_QK_FRAMES", 801),
        env("DEMUCS_QK_HEADS", 8),
        env("DEMUCS_QK_DIM", 64),
    );
    let shape = AttentionShape { bands, frames, heads, dim_head };
    let inner = shape.inner();
    let row = shape.row();
    let pitch = shape.scores_pitch();

    let qkv_host = fill(bands * frames * row, 3);
    let qkv = arena
        .upload(&gpu, &[bands * frames * row], &qkv_host, "qkv")
        .unwrap();
    let scores = arena.tensor(&gpu, &[shape.scores_len()], "scores").unwrap();
    arena.clear(&gpu, &scores);

    let q = qkv.slice(0, vec![qkv.len()]).unwrap();
    let k = qkv.slice(inner, vec![qkv.len() - inner]).unwrap();
    let batches = bands * heads;
    let (scores_outer, scores_inner) = GemmJob::contiguous_batch(frames * pitch, heads);
    let mut recorder = Recorder::new(&gpu);
    kernels
        .gemm_into(
            &gpu,
            &mut arena,
            &mut recorder,
            &q,
            &k,
            &scores,
            GemmJob {
                m: frames,
                n: frames,
                k: dim_head,
                lda: row,
                ldb: row,
                ldc: pitch,
                batches,
                inner_count: heads,
                a_outer: frames * row,
                a_inner: dim_head,
                b_outer: frames * row,
                b_inner: dim_head,
                c_outer: scores_outer,
                c_inner: scores_inner,
                transb: true,
            },
        )
        .unwrap();
    recorder.submit(&gpu).unwrap();

    let padded = read(&gpu, &scores);
    let mut got = Vec::with_capacity(batches * frames * frames);
    for block in 0..batches {
        let base = block * frames * pitch;
        for r in 0..frames {
            let start = base + r * pitch;
            got.extend_from_slice(&padded[start..start + frames]);
        }
    }
    // The dispatch leaves the scores unscaled: the attention scale rides in the
    // softmax's uniform, so the reference has to drop it here too.
    let scale = 1.0 / (dim_head as f32).sqrt();
    let expected: Vec<f32> = reference_scores(&qkv_host, shape)
        .into_iter()
        .map(|v| v / scale)
        .collect();
    let comparison = compare(&expected, &got).unwrap();
    println!(
        "QK^T alone (bands={bands} frames={frames}): rms_relative={:.3e} max_abs={:.3e}",
        comparison.rms_relative_error(),
        comparison.max_abs
    );
    assert!(comparison.rms_relative_error() < 1e-6);

    // Now the softmax, on top of the scores the dispatch just produced: giving it
    // the GPU's own input separates a softmax bug from a scores bug.
    let mut recorder = Recorder::new(&gpu);
    kernels
        .softmax_in_place(&gpu, &mut arena, &mut recorder, &scores, batches * frames, frames, pitch, scale)
        .unwrap();
    recorder.submit(&gpu).unwrap();
    let padded = read(&gpu, &scores);
    let mut gpu_probs = Vec::with_capacity(batches * frames * frames);
    for block in 0..batches {
        let base = block * frames * pitch;
        for r in 0..frames {
            let start = base + r * pitch;
            gpu_probs.extend_from_slice(&padded[start..start + frames]);
        }
    }
    // Reference softmax over the *GPU's* scores, so the comparison cannot be
    // contaminated by a scores difference. The scale rides in the softmax's
    // uniform, and softmax is shift- but not scale-invariant, so it has to be
    // applied here too.
    let mut want_probs = got.clone();
    for row in want_probs.chunks_mut(frames) {
        for value in row.iter_mut() {
            *value *= scale;
        }
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0f32;
        for value in row.iter_mut() {
            *value = (*value - max).exp();
            total += *value;
        }
        for value in row.iter_mut() {
            *value /= total;
        }
    }
    let softmax_cmp = compare(&want_probs, &gpu_probs).unwrap();
    println!(
        "softmax over the dispatch's own scores: rms_relative={:.3e} max_abs={:.3e}",
        softmax_cmp.rms_relative_error(),
        softmax_cmp.max_abs
    );
    let mut worst = (0.0f32, 0usize);
    for (index, (want, got)) in want_probs.iter().zip(gpu_probs.iter()).enumerate() {
        let diff = (want - got).abs();
        if diff > worst.0 {
            worst = (diff, index);
        }
    }
    println!(
        "  worst probability: |diff|={:.3e} at block {} row {} key {} (want {:.6e} got {:.6e})",
        worst.0,
        worst.1 / (frames * frames),
        (worst.1 / frames) % frames,
        worst.1 % frames,
        want_probs[worst.1],
        gpu_probs[worst.1]
    );
    // Is the whole row off by a constant factor (a normalisation bug) or are
    // individual entries wrong (an indexing one)?
    let bad_row = (worst.1 / frames) % frames;
    let block = worst.1 / (frames * frames);
    let start = (block * frames + bad_row) * frames;
    let mut ratios = Vec::new();
    for key in 0..frames {
        let want = want_probs[start + key];
        if want > 1e-6 {
            ratios.push(gpu_probs[start + key] / want);
        }
    }
    let min = ratios.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = ratios.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "  bad row {bad_row} of block {block}: ratio over {} entries in [{min:.6}, {max:.6}]",
        ratios.len()
    );
    // The row max the GPU saw, recovered from the ratio: `p = exp(s - m) / sum`.
    for key in [0usize, 100, 400, 700] {
        println!(
            "    key {key}: want {:.6e} got {:.6e} ratio {:.4}",
            want_probs[start + key],
            gpu_probs[start + key],
            gpu_probs[start + key] / want_probs[start + key]
        );
    }
    // Now the same two dispatches in ONE submission, which is how the op runs
    // them: if this differs from the two separately submitted steps, the bug is
    // in how the passes interact rather than in either one.
    let qkv2 = arena
        .upload(&gpu, &[bands * frames * row], &qkv_host, "qkv2")
        .unwrap();
    let scores2 = arena.tensor(&gpu, &[shape.scores_len()], "scores2").unwrap();
    let q2 = qkv2.slice(0, vec![qkv2.len()]).unwrap();
    let k2 = qkv2.slice(inner, vec![qkv2.len() - inner]).unwrap();
    let mut recorder = Recorder::new(&gpu);
    kernels
        .gemm_into(
            &gpu,
            &mut arena,
            &mut recorder,
            &q2,
            &k2,
            &scores2,
            GemmJob {
                m: frames,
                n: frames,
                k: dim_head,
                lda: row,
                ldb: row,
                ldc: pitch,
                batches,
                inner_count: heads,
                a_outer: frames * row,
                a_inner: dim_head,
                b_outer: frames * row,
                b_inner: dim_head,
                c_outer: scores_outer,
                c_inner: scores_inner,
                transb: true,
            },
        )
        .unwrap();
    kernels
        .softmax_in_place(&gpu, &mut arena, &mut recorder, &scores2, batches * frames, frames, pitch, scale)
        .unwrap();
    recorder.submit(&gpu).unwrap();
    let padded = read(&gpu, &scores2);
    let mut chained = Vec::with_capacity(batches * frames * frames);
    for block in 0..batches {
        let base = block * frames * pitch;
        for r in 0..frames {
            let start = base + r * pitch;
            chained.extend_from_slice(&padded[start..start + frames]);
        }
    }
    let chained_cmp = compare(&want_probs, &chained).unwrap();
    println!(
        "softmax chained behind QK^T in one submission: rms_relative={:.3e} max_abs={:.3e}",
        chained_cmp.rms_relative_error(),
        chained_cmp.max_abs
    );

    assert!(softmax_cmp.rms_relative_error() < 1e-6);
}

#[test]
fn attention_handles_a_single_band() {
    // Degenerate batching, where the two-level stride collapses.
    run_case(1, 128, 8, 64, 1e-5);
}
