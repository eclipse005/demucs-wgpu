//! Streaming overlap-add for unbounded duration.
//!
//! Semantics follow upstream `demucs/apply.py` (`apply_model`, `split=True`) and
//! `Separator.separate_tensor` global normalisation (`demucs/api.py`): triangular
//! `transition_weight`, stride `(1 - overlap) * segment_length`, shift pad of
//! `max_shift = int(0.5 * samplerate)`, and unbiased channel-mean std.
//!
//! Storage is `O(segment)` rings; flush after segment start `S` closes
//! `[flush_pos, min(S + stride, length))` (tail flushes through `length`).

use ndarray::{s, Array3, Array4};

use crate::demucs::host::{Htdemucs, TraceSink};
use crate::demucs::mix::{
    padded_chunk_mix, padded_chunk_shifted_mix, MixSamples,
};
use crate::demucs::pipeline::{
    center_trim, transition_weight, GlobalNorm, SegmentForward, SeparateOptions,
};
use crate::error::{Error, Result};

/// Shift view into the virtually padded mix (`apply.py` shift trick).
#[derive(Debug, Clone, Copy)]
pub struct ShiftView {
    pub offset: usize,
    pub max_shift: usize,
}

/// Segment / source metadata the streaming OLA needs from the model config.
#[derive(Debug, Clone, Copy)]
pub struct SplitMeta {
    pub sources: usize,
    pub samplerate: usize,
    /// Model default segment length in seconds (`model.segment`).
    pub segment: f64,
}

impl SplitMeta {
    pub fn from_model(model: &Htdemucs) -> Self {
        Self {
            sources: model.config.sources.len(),
            samplerate: model.config.samplerate,
            segment: model.config.segment,
        }
    }
}

/// Scan mean/std as `Separator.separate_tensor` without cloning the waveform.
///
/// `ref = wav.mean(0)` (mean over channels), then `mean = ref.mean()`,
/// `std = ref.std(unbiased=True) + 1e-8` (PyTorch default).
pub fn scan_norm(wav: &Array3<f32>) -> GlobalNorm {
    let (batch, channels, length) = wav.dim();
    let count = (batch * length) as f64;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for b in 0..batch {
        for t in 0..length {
            let mut acc = 0.0f64;
            for c in 0..channels {
                acc += wav[[b, c, t]] as f64;
            }
            let r = acc / channels as f64;
            sum += r;
            sum_sq += r * r;
        }
    }
    let mean = sum / count;
    let variance = if count > 1.0 {
        (sum_sq - sum * sum / count) / (count - 1.0)
    } else {
        0.0
    };
    GlobalNorm {
        mean: mean as f32,
        std: (variance.max(0.0).sqrt() as f32) + 1e-8,
    }
}

/// Pass-B in-place `(x - mean) / std`.
pub fn apply_norm_inplace(wav: &mut Array3<f32>, norm: GlobalNorm) {
    for value in wav.iter_mut() {
        *value = (*value - norm.mean) / norm.std;
    }
}

/// `len(range(0, length, stride))` — upstream `apply.py` offset count.
pub fn num_chunks_for(length: usize, _segment_length: usize, stride: usize) -> usize {
    if length == 0 || stride == 0 {
        0
    } else {
        (length + stride - 1) / stride
    }
}

struct OlaRing {
    batch: usize,
    sources: usize,
    channels: usize,
    length: usize,
    stride: usize,
    weight: Vec<f32>,
    ring_cap: usize,
    out_ring: Vec<f32>,
    sum_weight: Vec<f32>,
    flush_pos: usize,
}

impl OlaRing {
    fn new(
        batch: usize,
        sources: usize,
        channels: usize,
        length: usize,
        segment_length: usize,
        stride: usize,
        weight: Vec<f32>,
    ) -> Self {
        let ring_cap = segment_length.max(1);
        Self {
            batch,
            sources,
            channels,
            length,
            stride,
            weight,
            ring_cap,
            out_ring: vec![0.0f32; batch * sources * channels * ring_cap],
            sum_weight: vec![0.0f32; ring_cap],
            flush_pos: 0,
        }
    }

    #[inline]
    fn ring_index(&self, abs_t: usize) -> usize {
        abs_t % self.ring_cap
    }

    #[inline]
    fn out_index(&self, b: usize, src: usize, c: usize, abs_t: usize) -> usize {
        let r = self.ring_index(abs_t);
        (((b * self.sources + src) * self.channels + c) * self.ring_cap) + r
    }

    #[allow(dead_code)]
    fn splice_group(&mut self, chunk_out: &Array4<f32>, spans: &[(usize, usize)]) -> Result<()> {
        for (slot, (start, chunk_length)) in spans.iter().enumerate() {
            let trimmed = center_trim(
                &chunk_out
                    .slice(s![slot * self.batch..(slot + 1) * self.batch, .., .., ..])
                    .to_owned(),
                *chunk_length,
            );
            if trimmed.dim().3 != *chunk_length {
                return Err(Error::Shape(format!(
                    "the model returned {} samples for a {chunk_length}-sample chunk",
                    trimmed.dim().3
                )));
            }
            for b in 0..self.batch {
                for src in 0..self.sources {
                    for c in 0..self.channels {
                        for t in 0..*chunk_length {
                            let w = self.weight[t];
                            let idx = self.out_index(b, src, c, start + t);
                            self.out_ring[idx] += w * trimmed[[b, src, c, t]];
                        }
                    }
                }
            }
            for t in 0..*chunk_length {
                let idx = self.ring_index(start + t);
                self.sum_weight[idx] += self.weight[t];
            }
        }
        Ok(())
    }

    /// Splice one already-collected batch slot into the ring.
    fn splice_slot(
        &mut self,
        chunk_out: &Array4<f32>,
        slot: usize,
        start: usize,
        chunk_length: usize,
    ) -> Result<()> {
        let trimmed = center_trim(
            &chunk_out
                .slice(s![slot * self.batch..(slot + 1) * self.batch, .., .., ..])
                .to_owned(),
            chunk_length,
        );
        if trimmed.dim().3 != chunk_length {
            return Err(Error::Shape(format!(
                "the model returned {} samples for a {chunk_length}-sample chunk",
                trimmed.dim().3
            )));
        }
        for b in 0..self.batch {
            for src in 0..self.sources {
                for c in 0..self.channels {
                    for t in 0..chunk_length {
                        let w = self.weight[t];
                        let idx = self.out_index(b, src, c, start + t);
                        self.out_ring[idx] += w * trimmed[[b, src, c, t]];
                    }
                }
            }
        }
        for t in 0..chunk_length {
            let idx = self.ring_index(start + t);
            self.sum_weight[idx] += self.weight[t];
        }
        Ok(())
    }

    fn flush_to<F>(&mut self, flush_end: usize, mut emit: F) -> Result<()>
    where
        F: FnMut(usize, &[f32]) -> Result<()>,
    {
        let flush_end = flush_end.min(self.length);
        if flush_end <= self.flush_pos {
            return Ok(());
        }
        let mut frame = vec![0.0f32; self.batch * self.sources * self.channels];
        for t in self.flush_pos..flush_end {
            let r = self.ring_index(t);
            let w = self.sum_weight[r];
            if w <= 0.0 {
                return Err(Error::Shape(format!(
                    "overlap-add left sample {t} with no weight"
                )));
            }
            for b in 0..self.batch {
                for src in 0..self.sources {
                    for c in 0..self.channels {
                        let idx = self.out_index(b, src, c, t);
                        frame[(b * self.sources + src) * self.channels + c] = self.out_ring[idx] / w;
                        self.out_ring[idx] = 0.0;
                    }
                }
            }
            self.sum_weight[r] = 0.0;
            emit(t, &frame)?;
        }
        self.flush_pos = flush_end;
        Ok(())
    }

    fn flush_after_segment<F>(&mut self, segment_start: usize, last: bool, emit: F) -> Result<()>
    where
        F: FnMut(usize, &[f32]) -> Result<()>,
    {
        let flush_end = if last {
            self.length
        } else {
            (segment_start + self.stride).min(self.length)
        };
        self.flush_to(flush_end, emit)
    }
}

/// Streaming `split=True` pass. Emits final `/ sum_weight` frames via `on_frame`.
///
/// `frame` layout: `[b][src][c]` flat. With [`ShiftView`], indexes the virtually
/// padded shifted mix from `apply.py` (no full-length pad buffer).
pub fn stream_split_pass<P, F, M>(
    model: &Htdemucs,
    forward: &dyn SegmentForward<Pending = P>,
    mix: &M,
    options: SeparateOptions,
    batch_chunks: usize,
    shift: Option<ShiftView>,
    on_progress: &mut dyn FnMut(usize, usize),
    on_frame: &mut F,
    trace: &mut dyn TraceSink,
) -> Result<()>
where
    F: FnMut(usize, &[f32]) -> Result<()>,
    M: MixSamples + ?Sized,
{
    stream_split_pass_meta(
        &SplitMeta::from_model(model),
        forward,
        mix,
        options,
        batch_chunks,
        shift,
        on_progress,
        on_frame,
        trace,
    )
}

/// Like [`stream_split_pass`] but with explicit [`SplitMeta`] (for tests / mocks).
pub fn stream_split_pass_meta<P, F, M>(
    meta: &SplitMeta,
    forward: &dyn SegmentForward<Pending = P>,
    mix: &M,
    options: SeparateOptions,
    batch_chunks: usize,
    shift: Option<ShiftView>,
    on_progress: &mut dyn FnMut(usize, usize),
    on_frame: &mut F,
    trace: &mut dyn TraceSink,
) -> Result<()>
where
    F: FnMut(usize, &[f32]) -> Result<()>,
    M: MixSamples + ?Sized,
{
    let batch = mix.batch();
    let channels = mix.channels();
    let mix_length = mix.length();
    let sources = meta.sources;
    let segment_seconds = options.segment.unwrap_or(meta.segment);
    let segment_length = (meta.samplerate as f64 * segment_seconds) as usize;
    let stride = ((1.0 - options.overlap) * segment_length as f64) as usize;
    if stride == 0 {
        return Err(Error::Shape("overlap leaves a zero stride".into()));
    }
    let weight = transition_weight(segment_length, options.transition_power);

    let (view_length, shift_offset, max_shift) = match shift {
        Some(s) => (mix_length + s.max_shift - s.offset, s.offset, s.max_shift),
        None => (mix_length, 0usize, 0usize),
    };
    if view_length == 0 {
        return Ok(());
    }

    let total = num_chunks_for(view_length, segment_length, stride);
    let mut ring = OlaRing::new(
        batch,
        sources,
        channels,
        view_length,
        segment_length,
        stride,
        weight,
    );

    let mut chunks = 0usize;
    let mut queued: Option<(Vec<(usize, usize)>, P)> = None;
    let mut offset = 0usize;

    while offset < view_length || queued.is_some() {
        let resolved = if offset < view_length {
            let mut spans = Vec::with_capacity(batch_chunks);
            let mut cursor = offset;
            while spans.len() < batch_chunks && cursor < view_length {
                let chunk_length = (view_length - cursor).min(segment_length);
                spans.push((cursor, chunk_length));
                cursor += stride;
            }
            let group = spans.len();
            let mut batched = Array3::<f32>::zeros((group * batch, channels, segment_length));
            for (slot, (start, chunk_length)) in spans.iter().enumerate() {
                let padded = if max_shift == 0 {
                    padded_chunk_mix(mix, *start, *chunk_length, segment_length)?
                } else {
                    padded_chunk_shifted_mix(
                        mix,
                        shift_offset,
                        max_shift,
                        mix_length,
                        *start,
                        *chunk_length,
                        segment_length,
                    )?
                };
                batched
                    .slice_mut(s![slot * batch..(slot + 1) * batch, .., ..])
                    .assign(&padded);
            }
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
        // Splice+flush one segment at a time so the ring stays O(segment).
        // Forward still runs as a batch; only the OLA bookkeeping is serialised.
        for (slot, (seg_start, chunk_length)) in spans.iter().enumerate() {
            ring.splice_slot(&chunk_out, slot, *seg_start, *chunk_length)?;
            let is_last = *seg_start + stride >= view_length;
            ring.flush_after_segment(*seg_start, is_last, &mut *on_frame)?;
            chunks += 1;
            on_progress(chunks, total);
        }
    }
    if ring.flush_pos < ring.length {
        ring.flush_to(ring.length, on_frame)?;
    }
    let _ = chunks;
    Ok(())
}

/// Collect streaming split into a dense `Array4` (tests / in-memory callers).
pub fn stream_split_pass_collect<P, M>(
    model: &Htdemucs,
    forward: &dyn SegmentForward<Pending = P>,
    mix: &M,
    options: SeparateOptions,
    batch_chunks: usize,
    shift: Option<ShiftView>,
    on_progress: &mut dyn FnMut(usize, usize),
    trace: &mut dyn TraceSink,
) -> Result<Array4<f32>>
where
    M: MixSamples + ?Sized,
{
    stream_split_pass_collect_meta(
        &SplitMeta::from_model(model),
        forward,
        mix,
        options,
        batch_chunks,
        shift,
        on_progress,
        trace,
    )
}

/// [`stream_split_pass_collect`] with explicit [`SplitMeta`].
pub fn stream_split_pass_collect_meta<P, M>(
    meta: &SplitMeta,
    forward: &dyn SegmentForward<Pending = P>,
    mix: &M,
    options: SeparateOptions,
    batch_chunks: usize,
    shift: Option<ShiftView>,
    on_progress: &mut dyn FnMut(usize, usize),
    trace: &mut dyn TraceSink,
) -> Result<Array4<f32>>
where
    M: MixSamples + ?Sized,
{
    let batch = mix.batch();
    let channels = mix.channels();
    let mix_length = mix.length();
    let sources = meta.sources;
    let view_length = match shift {
        Some(s) => mix_length + s.max_shift - s.offset,
        None => mix_length,
    };
    let mut out = Array4::<f32>::zeros((batch, sources, channels, view_length));
    stream_split_pass_meta(
        meta,
        forward,
        mix,
        options,
        batch_chunks,
        shift,
        on_progress,
        &mut |t, frame| {
            for b in 0..batch {
                for src in 0..sources {
                    for c in 0..channels {
                        out[[b, src, c, t]] = frame[(b * sources + src) * channels + c];
                    }
                }
            }
            Ok(())
        },
        trace,
    )?;
    Ok(out)
}

/// Run the full `apply_model` streaming path (optional shifts) over an already
/// normalised mix, emitting denormalised frames via `on_frame`.
///
/// `on_frame(t, frame)` receives original-timeline sample index `t` and a flat
/// `[b][src][c]` frame (batch is almost always 1). Matches upstream
/// `apply_model` + caller's `out * std + mean` when `norm` is the Pass-A scan.
pub fn separate_streaming_normalized<P, F, M>(
    model: &Htdemucs,
    forward: &dyn SegmentForward<Pending = P>,
    mix: &M,
    options: SeparateOptions,
    norm: GlobalNorm,
    on_progress: &mut dyn FnMut(usize, usize),
    on_frame: &mut F,
    trace: &mut dyn TraceSink,
) -> Result<()>
where
    F: FnMut(usize, &[f32]) -> Result<()>,
    M: MixSamples + ?Sized,
{
    let batch = mix.batch();
    let length = mix.length();
    let sources = model.config.sources.len();
    let channels = mix.channels();
    let batch_chunks = std::env::var("DEMUCS_BATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let scale = if options.shifts > 0 {
        1.0 / options.shifts as f32
    } else {
        1.0
    };

    if options.shifts == 0 {
        stream_split_pass(
            model,
            forward,
            mix,
            options,
            batch_chunks,
            None,
            on_progress,
            &mut |t, frame| {
                let mut denorm = frame.to_vec();
                for v in denorm.iter_mut() {
                    *v = *v * norm.std + norm.mean;
                }
                on_frame(t, &denorm)
            },
            trace,
        )?;
        return Ok(());
    }

    // shifts >= 1: lockstep accumulate into a small pending map is hard for
    // unbounded; for the default shifts=1 we stream one pass and scale.
    // For shifts>1 we accumulate per-output-sample across passes using an
    // O(segment)-bounded approach only when shifts==1... For shifts>1 fall
    // back to allocating the output (still streaming OLA per pass).
    let samplerate = model.config.samplerate;
    let max_shift = (0.5 * samplerate as f64) as usize;
    let mut rng_state = 0x5eed_1234u64;
    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut next_rand = || {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rng_state >> 33) as usize) % (max_shift + 1)
    };

    if options.shifts == 1 {
        let offset = next_rand();
        let trim = max_shift - offset;
        let mut base_total = 0usize;
        stream_split_pass(
            model,
            forward,
            mix,
            SeparateOptions {
                shifts: 0,
                ..options
            },
            batch_chunks,
            Some(ShiftView { offset, max_shift }),
            &mut |d, t| {
                if base_total == 0 {
                    base_total = t;
                }
                on_progress(d, t);
            },
            &mut |t_view, frame| {
                if t_view < trim {
                    return Ok(());
                }
                let t = t_view - trim;
                if t >= length {
                    return Ok(());
                }
                let mut denorm = frame.to_vec();
                for v in denorm.iter_mut() {
                    *v = *v * scale * norm.std + norm.mean;
                }
                on_frame(t, &denorm)
            },
            trace,
        )?;
        return Ok(());
    }

    // shifts > 1: accumulate then emit (output still O(T); OLA rings are O(segment)).
    let mut out = Array4::<f32>::zeros((batch, sources, channels, length));
    let mut base = 0usize;
    let mut pass_total = 0usize;
    for pass in 0..options.shifts {
        let offset = next_rand();
        let trim = max_shift - offset;
        pass_total = 0;
        stream_split_pass(
            model,
            forward,
            mix,
            SeparateOptions {
                shifts: 0,
                ..options
            },
            batch_chunks,
            Some(ShiftView { offset, max_shift }),
            &mut |d, t| {
                if pass_total == 0 {
                    pass_total = t;
                }
                let remaining = options.shifts - pass - 1;
                on_progress(base + d, base + t * (remaining + 1));
            },
            &mut |t_view, frame| {
                if t_view < trim {
                    return Ok(());
                }
                let t = t_view - trim;
                if t >= length {
                    return Ok(());
                }
                for b in 0..batch {
                    for src in 0..sources {
                        for c in 0..channels {
                            out[[b, src, c, t]] +=
                                frame[(b * sources + src) * channels + c];
                        }
                    }
                }
                Ok(())
            },
            trace,
        )?;
        base += pass_total;
    }
    for t in 0..length {
        let mut frame = vec![0.0f32; batch * sources * channels];
        for b in 0..batch {
            for src in 0..sources {
                for c in 0..channels {
                    let v = out[[b, src, c, t]] * scale;
                    frame[(b * sources + src) * channels + c] = v * norm.std + norm.mean;
                }
            }
        }
        on_frame(t, &frame)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::demucs::host::NoTrace;
    use crate::demucs::mix::{padded_chunk_mix, padded_chunk_shifted_mix};
    use crate::demucs::pipeline::{self, padded_chunk, SeparateOptions};
    use ndarray::Array3;

    /// Identity forward: echo the mix into every source channel.
    struct IdForward {
        sources: usize,
    }

    impl SegmentForward for IdForward {
        type Pending = Array4<f32>;

        fn start(&self, padded: &Array3<f32>, _trace: &mut dyn TraceSink) -> Result<Array4<f32>> {
            let (batch, channels, time) = padded.dim();
            let mut out = Array4::<f32>::zeros((batch, self.sources, channels, time));
            for src in 0..self.sources {
                out.slice_mut(s![.., src, .., ..]).assign(padded);
            }
            Ok(out)
        }

        fn collect(&self, pending: Array4<f32>, _trace: &mut dyn TraceSink) -> Result<Array4<f32>> {
            Ok(pending)
        }
    }

    fn meta(sources: usize, segment: f64) -> SplitMeta {
        SplitMeta {
            sources,
            samplerate: 44100,
            segment,
        }
    }

    fn synthetic_mix(length: usize) -> Array3<f32> {
        let mut mix = Array3::<f32>::zeros((1, 2, length));
        for (i, v) in mix.iter_mut().enumerate() {
            *v = ((i as i32 % 97) as f32 - 48.0) / 48.0;
        }
        mix
    }

    /// Full-buffer OLA matching upstream apply.py / pipeline::split_pass, for
    /// bit-exact comparison against the ring path (no model weights needed).
    fn full_buffer_ola(
        meta: &SplitMeta,
        forward: &IdForward,
        mix: &Array3<f32>,
        options: SeparateOptions,
    ) -> Array4<f32> {
        let (batch, channels, length) = mix.dim();
        let sources = meta.sources;
        let segment_seconds = options.segment.unwrap_or(meta.segment);
        let segment_length = (meta.samplerate as f64 * segment_seconds) as usize;
        let stride = ((1.0 - options.overlap) * segment_length as f64) as usize;
        let weight = transition_weight(segment_length, options.transition_power);
        let mut out = Array4::<f32>::zeros((batch, sources, channels, length));
        let mut sum_weight = vec![0.0f32; length];
        let mut offset = 0usize;
        while offset < length {
            let chunk_length = (length - offset).min(segment_length);
            let padded = padded_chunk(mix, offset, chunk_length, segment_length).unwrap();
            let chunk_out = forward.start(&padded, &mut NoTrace).unwrap();
            let trimmed = center_trim(&chunk_out, chunk_length);
            for b in 0..batch {
                for src in 0..sources {
                    for c in 0..channels {
                        for t in 0..chunk_length {
                            out[[b, src, c, offset + t]] += weight[t] * trimmed[[b, src, c, t]];
                        }
                    }
                }
            }
            for t in 0..chunk_length {
                sum_weight[offset + t] += weight[t];
            }
            offset += stride;
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
        out
    }

    #[test]
    fn streaming_ola_matches_full_buffer_bit_exact() {
        let meta = meta(2, 0.25);
        let forward = IdForward { sources: 2 };
        // 2.5 segments with a short tail — exercises end-chunk padding.
        let segment_length = (44100.0 * 0.25) as usize;
        let mix_len = segment_length * 5 / 2;
        let mix = synthetic_mix(mix_len);
        let options = SeparateOptions {
            shifts: 0,
            overlap: 0.25,
            transition_power: 1.0,
            segment: Some(0.25),
        };
        let full = full_buffer_ola(&meta, &forward, &mix, options);
        let streamed = stream_split_pass_collect_meta(
            &meta,
            &forward,
            &mix,
            options,
            1,
            None,
            &mut |_, _| {},
            &mut NoTrace,
        )
        .unwrap();
        assert_eq!(full.dim(), streamed.dim());
        let worst = full
            .iter()
            .zip(streamed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(worst, 0.0, "streaming OLA drifted by {worst:.3e}");
    }

    #[test]
    fn streaming_ola_matches_full_buffer_with_batched_groups() {
        let meta = meta(2, 0.25);
        let forward = IdForward { sources: 2 };
        let segment_length = (44100.0 * 0.25) as usize;
        let mix = synthetic_mix(segment_length * 4);
        let options = SeparateOptions {
            shifts: 0,
            overlap: 0.25,
            transition_power: 1.0,
            segment: Some(0.25),
        };
        let full = full_buffer_ola(&meta, &forward, &mix, options);
        let streamed = stream_split_pass_collect_meta(
            &meta,
            &forward,
            &mix,
            options,
            3,
            None,
            &mut |_, _| {},
            &mut NoTrace,
        )
        .unwrap();
        let worst = full
            .iter()
            .zip(streamed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(worst, 0.0, "batched streaming OLA drifted by {worst:.3e}");
    }

    #[test]
    fn scan_norm_matches_normalize() {
        let mix = synthetic_mix(4096);
        let (normalized, norm) = pipeline::normalize(&mix);
        let scanned = scan_norm(&mix);
        assert_eq!(scanned.mean, norm.mean);
        assert_eq!(scanned.std, norm.std);
        let mut inplace = mix.clone();
        apply_norm_inplace(&mut inplace, scanned);
        let worst = normalized
            .iter()
            .zip(inplace.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(worst, 0.0);
    }

    #[test]
    fn shifted_padded_chunk_matches_full_pad() {
        let mix = synthetic_mix(1000);
        let max_shift = 100;
        let offset = 37;
        let length = mix.dim().2;
        let padded = padded_chunk(&mix, 0, length, length + 2 * max_shift).unwrap();
        let view_start = 50;
        let chunk_length = 128;
        let target = 200;
        let from_full = padded_chunk(
            &padded.slice(s![.., .., offset..offset + length + max_shift - offset]).to_owned(),
            view_start,
            chunk_length,
            target,
        )
        .unwrap();
        let from_virt = padded_chunk_shifted_mix(
            &mix, offset, max_shift, length, view_start, chunk_length, target,
        )
        .unwrap();
        let worst = from_full
            .iter()
            .zip(from_virt.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(worst, 0.0, "virtual shift pad drifted by {worst:.3e}");
    }

    #[test]
    fn num_chunks_matches_upstream_range() {
        // len(range(0, length, stride))
        assert_eq!(num_chunks_for(100, 200, 150), 1);
        assert_eq!(num_chunks_for(200, 200, 150), 2); // [0, 150]
        assert_eq!(num_chunks_for(343980 * 2, 343980, 257985), 3);
    }

    /// Virtual shift view + streaming OLA must match full pad → split → trim
    /// (upstream apply.py shift trick), bit-exact, with a fixed offset.
    #[test]
    fn shift_view_streaming_matches_full_pad_trim() {
        let meta = meta(2, 0.25);
        let forward = IdForward { sources: 2 };
        let segment_length = (44100.0 * 0.25) as usize;
        let mix_len = segment_length * 3;
        let mix = synthetic_mix(mix_len);
        let options = SeparateOptions {
            shifts: 0, // inner pass; we apply one shift manually
            overlap: 0.25,
            transition_power: 1.0,
            segment: Some(0.25),
        };
        let max_shift = 1000usize;
        let offset = 123usize;
        let trim = max_shift - offset;
        // Full-pad reference (apply.py).
        let padded = padded_chunk(&mix, 0, mix_len, mix_len + 2 * max_shift).unwrap();
        let shifted_len = mix_len + max_shift - offset;
        let shifted = padded
            .slice(s![.., .., offset..offset + shifted_len])
            .to_owned();
        let full_view = full_buffer_ola(&meta, &forward, &shifted, options);
        let full_trimmed = full_view.slice(s![.., .., .., trim..trim + mix_len]).to_owned();

        let mut streamed = Array4::<f32>::zeros((1, 2, 2, mix_len));
        stream_split_pass_meta(
            &meta,
            &forward,
            &mix,
            options,
            1,
            Some(ShiftView { offset, max_shift }),
            &mut |_, _| {},
            &mut |t_view, frame| {
                if t_view < trim {
                    return Ok(());
                }
                let t = t_view - trim;
                if t >= mix_len {
                    return Ok(());
                }
                for b in 0..1 {
                    for src in 0..2 {
                        for c in 0..2 {
                            streamed[[b, src, c, t]] =
                                frame[(b * 2 + src) * 2 + c];
                        }
                    }
                }
                Ok(())
            },
            &mut NoTrace,
        )
        .unwrap();
        let worst = full_trimmed
            .iter()
            .zip(streamed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(worst, 0.0, "shift-view streaming drifted by {worst:.3e}");
    }
}
