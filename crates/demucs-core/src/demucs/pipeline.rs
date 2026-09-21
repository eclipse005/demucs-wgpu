//! The `apply_model` wrapper: chunking, overlap-add with a triangular window,
//! the shift trick, and the global normalisation `Separator.separate_tensor`
//! performs.
//!
//! A transcription of `demucs/apply.py` for the `HTDemucs` family only: `split`
//! is always on, `num_workers`/`pool`/`progress` do not exist, and the models are
//! never moved between devices.

use ndarray::{Array3, Array4, s};

use crate::demucs::host::{Htdemucs, TraceSink};
use crate::error::{Error, Result};

/// Inference settings, mirroring the CLI defaults of `demucs.separate`.
#[derive(Debug, Clone, Copy)]
pub struct SeparateOptions {
    /// Number of random time shifts; 1 is the reference default, 0 makes the
    /// result deterministic (and is what the alignment tests use).
    pub shifts: usize,
    /// Fraction of the segment each chunk overlaps its neighbour by.
    pub overlap: f64,
    /// Exponent applied to the triangular window (`transition_power`).
    pub transition_power: f64,
    /// Overrides `model.segment` (seconds) when set.
    pub segment: Option<f64>,
}

impl Default for SeparateOptions {
    fn default() -> Self {
        Self {
            shifts: 1,
            overlap: 0.25,
            transition_power: 1.0,
            segment: None,
        }
    }
}

/// The global normalisation `Separator.separate_tensor` applies before calling
/// `apply_model`, and the values needed to undo it.
#[derive(Debug, Clone, Copy)]
pub struct GlobalNorm {
    pub mean: f32,
    pub std: f32,
}

/// `(wav - mean) / (std + 1e-8)` over the channel-averaged signal.
pub fn normalize(wav: &Array3<f32>) -> (Array3<f32>, GlobalNorm) {
    let (batch, channels, length) = wav.dim();
    // `ref = wav.mean(0)` then `mean = ref.mean()`, `std = ref.std()`.
    let mut reference = vec![0.0f64; length];
    for b in 0..batch {
        for t in 0..length {
            let mut sum = 0.0f64;
            for c in 0..channels {
                sum += wav[[b, c, t]] as f64;
            }
            reference[t] += sum / channels as f64;
        }
    }
    let count = (batch * length) as f64;
    let mean = reference.iter().sum::<f64>() / count;
    let variance = reference
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        / (count - 1.0);
    let std = variance.sqrt() as f32 + 1e-8;

    let mut out = wav.clone();
    for value in out.iter_mut() {
        *value = (*value - mean as f32) / std;
    }
    (
        out,
        GlobalNorm {
            mean: mean as f32,
            std,
        },
    )
}

/// Inverse of [`normalize`], applied to the `(batch, sources, channels, time)`
/// output.
pub fn denormalize(sources: &mut Array4<f32>, norm: GlobalNorm) {
    for value in sources.iter_mut() {
        *value = *value * norm.std + norm.mean;
    }
}

/// `TensorChunk.padded`: centre the chunk inside `target_length` zeros, taking
/// whatever context the parent signal has on each side.
fn padded_chunk(
    mix: &Array3<f32>,
    offset: usize,
    length: usize,
    target_length: usize,
) -> Result<Array3<f32>> {
    let (batch, channels, total) = mix.dim();
    if offset >= total {
        return Err(Error::Shape(format!(
            "chunk offset {offset} is past the {total}-sample signal"
        )));
    }
    let delta = target_length
        .checked_sub(length)
        .ok_or_else(|| Error::Shape(format!("cannot pad {length} samples down to {target_length}")))?;
    let start = offset as isize - (delta / 2) as isize;
    let end = start + target_length as isize;
    let correct_start = start.max(0) as usize;
    let correct_end = (end.min(total as isize)).max(0) as usize;
    let pad_left = correct_start as isize - start;
    let pad_right = end - correct_end as isize;

    let mut out = Array3::<f32>::zeros((batch, channels, target_length));
    if correct_end > correct_start {
        out.slice_mut(s![
            ..,
            ..,
            pad_left as usize..(pad_left as usize + correct_end - correct_start)
        ])
        .assign(&mix.slice(s![.., .., correct_start..correct_end]));
    }
    let _ = (pad_right, batch);
    Ok(out)
}

/// `center_trim(tensor, length)`.
pub fn center_trim(tensor: &Array4<f32>, reference: usize) -> Array4<f32> {
    let length = tensor.dim().3;
    if length <= reference {
        return tensor.clone();
    }
    let delta = length - reference;
    let start = delta / 2;
    tensor.slice(s![.., .., .., start..start + reference]).to_owned()
}

/// The triangular overlap-add window `apply.py` builds, raised to
/// `transition_power`.
pub fn transition_weight(segment_length: usize, transition_power: f64) -> Vec<f32> {
    let half = segment_length / 2;
    let mut weight = Vec::with_capacity(segment_length);
    for i in 1..=half {
        weight.push(i as f32);
    }
    for i in (1..=segment_length - half).rev() {
        weight.push(i as f32);
    }
    let max = weight.iter().cloned().fold(0.0f32, f32::max).max(1.0);
    for value in weight.iter_mut() {
        *value = (*value / max).powf(transition_power as f32);
    }
    weight
}

/// A small deterministic RNG for the shift offsets. The reference uses the
/// unseeded global `random` module, so its offsets are not reproducible either;
/// this one at least makes our own runs repeatable.
struct ShiftRng(u64);

impl ShiftRng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }

    /// `random.randint(0, max)` inclusive.
    fn randint(&mut self, max: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % (max + 1)
    }
}

/// `apply_model(model, mix, shifts, split=True, overlap, transition_power)`.
///
/// `mix` is `(batch, channels, samples)` and must already be normalised; the
/// result is `(batch, sources, channels, samples)` in the same units.
pub fn separate(
    model: &Htdemucs,
    mix: &Array3<f32>,
    options: SeparateOptions,
    trace: &mut dyn TraceSink,
) -> Result<Array4<f32>> {
    separate_with(model, &|padded, trace| model.forward(padded, trace), mix, options, trace)
}

/// One group of segments' forward, split into "queue it" and "wait for it".
///
/// The device path needs both halves: a chunk can stay in flight while the loop
/// prepares and submits the next one, which is what keeps the readback off the
/// critical path. The host path implements the pair as compute-then-hand-back,
/// so its arithmetic and the order it runs in are unchanged.
///
/// [`SegmentForward::start`] hands back the queued group's handle and the caller
/// — which is the only thing that knows when it wants the result — passes it to
/// [`SegmentForward::collect`]. Keeping the handle in the loop's state rather
/// than inside the forward is what lets a group be in flight across the next
/// group's submit.
pub trait SegmentForward {
    /// What one queued group hands to its `collect`: the device path's submitted
    /// chunk, the host path's finished tensor.
    type Pending;

    /// Queues one forward of `padded`, a `(group * mix_batch, channels,
    /// segment_length)` tensor, and returns with the work in flight.
    fn start(&self, padded: &Array3<f32>, trace: &mut dyn TraceSink) -> Result<Self::Pending>;

    /// Waits for the group queued by the matching [`SegmentForward::start`] and
    /// returns its output.
    fn collect(&self, pending: Self::Pending, trace: &mut dyn TraceSink) -> Result<Array4<f32>>;
}

/// A [`SegmentForward`] over a plain closure: the whole forward happens inside
/// `start`, the queued handle is its result.
struct Immediate<'a, F: ?Sized> {
    forward: &'a F,
}

impl<'a, F: ?Sized> Immediate<'a, F> {
    fn new(forward: &'a F) -> Self {
        Self { forward }
    }
}

impl<F: ?Sized> SegmentForward for Immediate<'_, F>
where
    F: Fn(&Array3<f32>, &mut dyn TraceSink) -> Result<Array4<f32>>,
{
    type Pending = Array4<f32>;

    fn start(&self, padded: &Array3<f32>, trace: &mut dyn TraceSink) -> Result<Array4<f32>> {
        (self.forward)(padded, trace)
    }

    fn collect(&self, pending: Array4<f32>, _trace: &mut dyn TraceSink) -> Result<Array4<f32>> {
        Ok(pending)
    }
}

/// [separate] with an injected per-chunk forward — the host path and the tests
/// pass a closure; the device path passes its own [`SegmentForward`].
pub fn separate_with(
    model: &Htdemucs,
    forward: &dyn Fn(&Array3<f32>, &mut dyn TraceSink) -> Result<Array4<f32>>,
    mix: &Array3<f32>,
    options: SeparateOptions,
    trace: &mut dyn TraceSink,
) -> Result<Array4<f32>> {
    separate_with_progress(model, forward, mix, options, &mut |_, _| {}, trace)
}

/// The wgpu path through the same chunking/overlap machinery.
pub fn separate_gpu(
    model: &Htdemucs,
    runner: &crate::gpu::htdemucs::GpuHtdemucsRunner,
    mix: &Array3<f32>,
    options: SeparateOptions,
    trace: &mut dyn TraceSink,
) -> Result<Array4<f32>> {
    separate_gpu_progress(model, runner, mix, options, &mut |_, _| {}, trace)
}

/// [`separate_gpu`] with a per-chunk progress callback (`done`, `total`).
pub fn separate_gpu_progress(
    model: &Htdemucs,
    runner: &crate::gpu::htdemucs::GpuHtdemucsRunner,
    mix: &Array3<f32>,
    options: SeparateOptions,
    on_progress: &mut dyn FnMut(usize, usize),
    trace: &mut dyn TraceSink,
) -> Result<Array4<f32>> {
    let forward = crate::gpu::htdemucs::DeviceForward::new(model, runner);
    separate_forwarding(model, &forward, mix, options, on_progress, trace)
}

/// [`separate_with`] with a per-chunk progress callback (`done`, `total`).
///
/// The callback runs on the calling thread, once per completed segment; the
/// final call has `done == total`. Shifts multiply the totals: `total` counts
/// the segments every shift pass processes.
pub fn separate_with_progress(
    model: &Htdemucs,
    forward: &dyn Fn(&Array3<f32>, &mut dyn TraceSink) -> Result<Array4<f32>>,
    mix: &Array3<f32>,
    options: SeparateOptions,
    on_progress: &mut dyn FnMut(usize, usize),
    trace: &mut dyn TraceSink,
) -> Result<Array4<f32>> {
    separate_forwarding(
        model,
        &Immediate::new(forward),
        mix,
        options,
        on_progress,
        trace,
    )
}

/// The shift loop, then the chunking/overlap pass, over either forward
/// implementation.
fn separate_forwarding<P>(
    model: &Htdemucs,
    forward: &dyn SegmentForward<Pending = P>,
    mix: &Array3<f32>,
    options: SeparateOptions,
    on_progress: &mut dyn FnMut(usize, usize),
    trace: &mut dyn TraceSink,
) -> Result<Array4<f32>> {
    let (batch, channels, length) = mix.dim();
    let batch_chunks = forward_batch();
    if options.shifts > 0 {
        let samplerate = model.config.samplerate;
        let max_shift = (0.5 * samplerate as f64) as usize;
        let padded = padded_chunk(mix, 0, length, length + 2 * max_shift)?;
        let mut out: Option<Array4<f32>> = None;
        let mut rng = ShiftRng::new(0x5eed_1234);
        // One work unit is a segment, and the total spans every shift pass. Each
        // pass runs over its own shifted slice, so its segment count is only
        // known once the first callback fires; `base` walks the done counter
        // forward by each completed pass's actual count so the whole run is
        // monotonic and ends at `done == total`.
        let mut base = 0usize;
        let mut pass_total = 0usize;
        for pass in 0..options.shifts {
            let offset = rng.randint(max_shift);
            let shifted_length = length + max_shift - offset;
            let shifted = padded
                .slice(s![.., .., offset..offset + shifted_length])
                .to_owned();
            pass_total = 0;
            let result = split_pass(
                model,
                forward,
                &shifted,
                options,
                batch_chunks,
                &mut |d, t| {
                    if pass_total == 0 {
                        pass_total = t;
                    }
                    let remaining = options.shifts - pass - 1;
                    on_progress(base + d, base + t * (remaining + 1));
                },
                trace,
            )?;
            base += pass_total;
            let trimmed = result
                .slice(s![.., .., .., max_shift - offset..])
                .to_owned();
            out = Some(match out {
                None => trimmed,
                Some(previous) => previous + trimmed,
            });
        }
        let mut out = out.expect("at least one shift");
        let scale = 1.0 / options.shifts as f32;
        out.mapv_inplace(|v| v * scale);
        return Ok(out);
    }
    let _ = (batch, channels);
    split_pass(model, forward, mix, options, batch_chunks, on_progress, trace)
}

/// How many segments one forward pass carries, from `DEMUCS_BATCH` (default 1).
/// The device path is what makes a group larger than 1 worth anything: it takes
/// the batched `(batch, channels, segment_length)` tensor directly.
fn forward_batch() -> usize {
    std::env::var("DEMUCS_BATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

/// The `split=True` branch: one model call per segment, overlap-added with the
/// triangular window.
///
/// Segments are grouped `batch_chunks` at a time and forwarded together (the
/// device path's `separate_gpu` takes a batched `(batch, channels,
/// segment_length)` tensor and keeps every segment independent, so one pass
/// over B segments costs about the host-side overhead of one). The mix's own
/// batch axis multiplies in. The host path passes 1: its forward is equally
/// per-segment, so a group would only deepen the overlap bookkeeping for no
/// gain.
///
/// Groups are queued one ahead: iteration n queues group n and then resolves
/// group n-1, so a `SegmentForward` that submits asynchronously — the device
/// path does — pays for the readback while the group queued a moment ago is
/// still running. Groups are still spliced in order, and the window's weights
/// are summed in the same order, so the overlap-add result is unchanged down to
/// the rounding: with 1 segment per group the sequence of adds is literally the
/// same as it was.
fn split_pass<P>(
    model: &Htdemucs,
    forward: &dyn SegmentForward<Pending = P>,
    mix: &Array3<f32>,
    options: SeparateOptions,
    batch_chunks: usize,
    on_progress: &mut dyn FnMut(usize, usize),
    trace: &mut dyn TraceSink,
) -> Result<Array4<f32>> {
    let (batch, channels, length) = mix.dim();
    let sources = model.config.sources.len();
    let segment_seconds = options.segment.unwrap_or(model.config.segment);
    let segment_length = (model.config.samplerate as f64 * segment_seconds) as usize;
    let stride = ((1.0 - options.overlap) * segment_length as f64) as usize;
    if stride == 0 {
        return Err(Error::Shape("overlap leaves a zero stride".into()));
    }
    let weight = transition_weight(segment_length, options.transition_power);
    // Count the segments up front so `total` is known before the first callback.
    let total = num_chunks_for(length, segment_length, stride);

    let mut out = Array4::<f32>::zeros((batch, sources, channels, length));
    let mut sum_weight = vec![0.0f32; length];
    let mut chunks = 0usize;
    // The group queued last — its spans and its handle — whose results are
    // still coming.
    let mut queued: Option<(Vec<(usize, usize)>, P)> = None;
    let mut offset = 0usize;
    // One extra turn round the loop: the last iteration has nothing left to
    // queue and only resolves the tail group.
    while offset < length || queued.is_some() {
        let resolved = if offset < length {
            // One group: up to `batch_chunks` segments, all padded to
            // `segment_length`, forwarded together. The mix's own batch axis
            // rides along inside each segment's tensor, so the forward's batch
            // is `group * mix_batch`.
            let mut spans = Vec::with_capacity(batch_chunks);
            let mut cursor = offset;
            while spans.len() < batch_chunks && cursor < length {
                let chunk_length = (length - cursor).min(segment_length);
                spans.push((cursor, chunk_length));
                cursor += stride;
            }
            let group = spans.len();
            let mut batched = Array3::<f32>::zeros((group * batch, channels, segment_length));
            for (slot, (start, chunk_length)) in spans.iter().enumerate() {
                let padded = padded_chunk(mix, *start, *chunk_length, segment_length)?;
                batched
                    .slice_mut(s![slot * batch..(slot + 1) * batch, .., ..])
                    .assign(&padded);
            }
            // Queued *before* the previous group is collected: on the device
            // path that is what keeps the device earning while the host waits
            // for the readback.
            let pending = forward.start(&batched, trace)?;
            offset = cursor;
            queued.replace((spans, pending))
        } else {
            queued.take()
        };
        let Some((spans, pending)) = resolved else {
            continue;
        };
        let chunk_out = forward.collect(pending, trace)?;
        splice_group(
            &mut out,
            &mut sum_weight,
            &chunk_out,
            &spans,
            &weight,
            batch,
            sources,
            channels,
        )?;
        // One callback per segment, still in segment order. A group larger than
        // one segment reports its whole group at once; at the default group of
        // one that is the same call, in the same place, as before.
        for _ in 0..spans.len() {
            chunks += 1;
            on_progress(chunks, total);
        }
    }
    let _ = chunks;
    if sum_weight.iter().cloned().fold(f32::INFINITY, f32::min) <= 0.0 {
        return Err(Error::Shape("overlap-add left samples with no weight".into()));
    }
    for b in 0..batch {
        for src in 0..sources {
            for c in 0..channels {
                for t in 0..length {
                    out[[b, src, c, t]] /= sum_weight[t];
                }
            }
        }
    }
    Ok(out)
}

/// Overlap-adds one group's output into `out`, weighted, counting each segment
/// into `sum_weight`.
fn splice_group(
    out: &mut Array4<f32>,
    sum_weight: &mut [f32],
    chunk_out: &Array4<f32>,
    spans: &[(usize, usize)],
    weight: &[f32],
    batch: usize,
    sources: usize,
    channels: usize,
) -> Result<()> {
    for (slot, (start, chunk_length)) in spans.iter().enumerate() {
        let trimmed = center_trim(
            &chunk_out
                .slice(s![slot * batch..(slot + 1) * batch, .., .., ..])
                .to_owned(),
            *chunk_length,
        );
        if trimmed.dim().3 != *chunk_length {
            return Err(Error::Shape(format!(
                "the model returned {} samples for a {chunk_length}-sample chunk",
                trimmed.dim().3
            )));
        }
        let w_offset = *start;
        for b in 0..batch {
            for src in 0..sources {
                for c in 0..channels {
                    for t in 0..*chunk_length {
                        let w = weight[t];
                        out[[b, src, c, w_offset + t]] += w * trimmed[[b, src, c, t]];
                    }
                }
            }
        }
        for t in 0..*chunk_length {
            sum_weight[w_offset + t] += weight[t];
        }
    }
    Ok(())
}

/// How many segments the split path draws for a mix of `length` samples.
fn num_chunks_for(length: usize, segment_length: usize, stride: usize) -> usize {
    if length <= segment_length || stride == 0 {
        return 1;
    }
    (length - segment_length).div_ceil(stride) + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn the_transition_window_rises_then_falls() {
        let weight = transition_weight(8, 1.0);
        assert_eq!(weight.len(), 8);
        assert!((weight[3] - 1.0).abs() < 1e-6, "{weight:?}");
        assert!(weight[0] < weight[1] && weight[1] < weight[2]);
        assert!(weight[7] < weight[6] && weight[6] < weight[5]);
    }

    #[test]
    fn a_centred_chunk_takes_context_from_both_sides() {
        let mix = Array3::from_shape_vec((1, 1, 10), (0..10).map(|v| v as f32).collect()).unwrap();
        // A 4-sample chunk at offset 4 padded to 8 takes 2 samples on each side.
        let padded = padded_chunk(&mix, 4, 4, 8).unwrap();
        assert_eq!(
            padded.slice(s![0, 0, ..]).to_vec(),
            vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn a_chunk_at_the_start_is_zero_padded_on_the_left() {
        let mix = Array3::from_shape_vec((1, 1, 6), (0..6).map(|v| v as f32).collect()).unwrap();
        let padded = padded_chunk(&mix, 0, 3, 7).unwrap();
        assert_eq!(
            padded.slice(s![0, 0, ..]).to_vec(),
            vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn center_trim_removes_the_larger_half_first() {
        let tensor = Array4::from_shape_vec((1, 1, 1, 5), vec![0.0f32, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let trimmed = center_trim(&tensor, 3);
        assert_eq!(trimmed.dim(), (1, 1, 1, 3));
        assert_eq!(trimmed.slice(s![0, 0, 0, ..]).to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn normalisation_round_trips() {
        let wav = array![[[0.5f32, -0.25, 1.0, 0.0]], [[0.5, -0.25, 1.0, 0.0]]];
        let (normalized, norm) = normalize(&wav);
        let mut back = normalized.clone();
        for value in back.iter_mut() {
            *value = *value * norm.std + norm.mean;
        }
        for (a, b) in back.iter().zip(wav.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn the_shift_rng_stays_in_range() {
        let mut rng = ShiftRng::new(7);
        for _ in 0..1000 {
            let value = rng.randint(22050);
            assert!(value <= 22050);
        }
    }

    /// Grouping the segments' chunks into one batched forward is a scheduling
    /// change, not a maths change: the overlap-add weights and the trims are
    /// per chunk, so a group of 3 must land on exactly the same numbers as one
    /// chunk at a time — including the short tail chunk, whose trim differs.
    #[test]
    fn a_batched_group_of_chunks_matches_them_one_at_a_time() {
        let checkpoint = crate::paths::htdemucs_checkpoint();
        if !checkpoint.exists() {
            eprintln!("skipping: {} is not present", checkpoint.display());
            return;
        }
        let loaded = crate::demucs::load_weights(&checkpoint).expect("weights");
        let mut config = loaded.config.clone();
        // A short segment keeps the host forward quick while still forcing
        // several chunks per mix.
        config.segment = 0.25;
        let weights = crate::demucs::HtdemucsWeights::load(&loaded.checkpoint, &config)
            .expect("weights");
        let model = Htdemucs::new(config, weights).expect("model");

        let segment = model.training_length();
        // 2.5 segments: the last chunk is a short one that gets padded.
        let mix_len = segment * 5 / 2;
        let mut mix = Array3::<f32>::zeros((1, 2, mix_len));
        for (i, value) in mix.iter_mut().enumerate() {
            *value = ((i as i32 % 97) as f32 - 48.0) / 48.0;
        }
        let options = SeparateOptions {
            shifts: 0,
            overlap: 0.25,
            transition_power: 1.0,
            segment: Some(0.25),
        };
        let mut trace = crate::demucs::host::NoTrace;
        // The host forward, wrapped: `split_pass` drives every forward through
        // the two-phase trait so the device path can leave a chunk in flight.
        let forward = |padded: &Array3<f32>, trace: &mut dyn TraceSink| model.forward(padded, trace);
        let alone = split_pass(
            &model,
            &Immediate::new(&forward),
            &mix,
            options,
            1,
            &mut |_, _| {},
            &mut trace,
        )
        .expect("split pass, one chunk at a time");
        let grouped = split_pass(
            &model,
            &Immediate::new(&forward),
            &mix,
            options,
            3,
            &mut |_, _| {},
            &mut trace,
        )
        .expect("split pass, chunks in threes");
        assert_eq!(alone.dim(), grouped.dim());
        let worst = alone
            .iter()
            .zip(grouped.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(worst, 0.0, "batched grouping moved the output by {worst:.3e}");
    }
}
