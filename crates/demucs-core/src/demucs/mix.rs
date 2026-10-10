//! Windowed mix access: Pass A O(1) norm scan, Pass B O(segment) sample window.
//!
//! Eliminates a resident full-track `Array3` on the file / CLI path so peak RAM
//! is O(segment) + model working set for common streaming emit paths.

use std::cell::RefCell;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use ndarray::Array3;

use crate::audio::{self, AudioInfo};
use crate::demucs::pipeline::GlobalNorm;
use crate::error::{Error, IoContext, Result};

/// Absolute-index sample access for OLA padding (batch is almost always 1).
pub trait MixSamples {
    fn batch(&self) -> usize;
    fn channels(&self) -> usize;
    fn length(&self) -> usize;

    /// Sample at `(batch, channel, t)`. `t` must be in `0..length()`.
    fn at(&self, batch: usize, channel: usize, t: usize) -> f32;
}

impl MixSamples for Array3<f32> {
    fn batch(&self) -> usize {
        self.dim().0
    }
    fn channels(&self) -> usize {
        self.dim().1
    }
    fn length(&self) -> usize {
        self.dim().2
    }
    fn at(&self, batch: usize, channel: usize, t: usize) -> f32 {
        self[[batch, channel, t]]
    }
}

impl<T: MixSamples + ?Sized> MixSamples for &T {
    fn batch(&self) -> usize {
        (**self).batch()
    }
    fn channels(&self) -> usize {
        (**self).channels()
    }
    fn length(&self) -> usize {
        (**self).length()
    }
    fn at(&self, batch: usize, channel: usize, t: usize) -> f32 {
        (**self).at(batch, channel, t)
    }
}

/// Scan mean/std as `Separator.separate_tensor` while streaming the file once.
///
/// Peak RAM is O(window) scratch, not O(T). Mono inputs destined for stereo are
/// treated as duplicated channels (channel-mean equals the mono sample).
pub fn scan_norm_file(
    path: impl AsRef<Path>,
    wanted_channels: usize,
) -> Result<(GlobalNorm, AudioInfo)> {
    let path = path.as_ref();
    let info = audio::probe_audio(path)?;
    if info.channels == 0 {
        return Err(Error::Audio("zero channels".into()));
    }
    if info.channels != wanted_channels && !(info.channels == 1 && wanted_channels == 2) {
        return Err(Error::Audio(format!(
            "cannot convert {} channels to {wanted_channels}",
            info.channels
        )));
    }
    let file_channels = info.channels;
    let logical_channels = wanted_channels;

    let mut file = File::open(path).with_path(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).with_path(path)?;

    let (sum, sum_sq, count) = if magic == *b"fLaC" {
        scan_norm_flac(path, file_channels, logical_channels)?
    } else {
        file.seek(SeekFrom::Start(0)).with_path(path)?;
        scan_norm_wav(&mut file, file_channels, logical_channels)?
    };

    let frames = info.frames;
    if count == 0.0 {
        return Ok((
            GlobalNorm {
                mean: 0.0,
                std: 1e-8,
            },
            AudioInfo {
                sample_rate: info.sample_rate,
                channels: logical_channels,
                frames,
            },
        ));
    }
    let mean = sum / count;
    let variance = if count > 1.0 {
        (sum_sq - sum * sum / count) / (count - 1.0)
    } else {
        0.0
    };
    Ok((
        GlobalNorm {
            mean: mean as f32,
            std: (variance.max(0.0).sqrt() as f32) + 1e-8,
        },
        AudioInfo {
            sample_rate: info.sample_rate,
            channels: logical_channels,
            frames,
        },
    ))
}

fn scan_norm_flac(
    path: &Path,
    file_channels: usize,
    logical_channels: usize,
) -> Result<(f64, f64, f64)> {
    let mut reader = claxon::FlacReader::open(path)
        .map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
    let info = reader.streaminfo();
    let scale = 1.0f32 / (1i64 << (info.bits_per_sample - 1)) as f32;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0.0f64;
    let mut frame = vec![0.0f32; file_channels];
    let mut idx = 0usize;
    for sample in reader.samples() {
        let value = sample.map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
        frame[idx % file_channels] = value as f32 * scale;
        idx += 1;
        if idx % file_channels == 0 {
            let r = channel_mean(&frame, file_channels, logical_channels);
            sum += r;
            sum_sq += r * r;
            count += 1.0;
        }
    }
    if idx % file_channels != 0 {
        return Err(Error::Audio(format!(
            "{}: truncated FLAC frame",
            path.display()
        )));
    }
    Ok((sum, sum_sq, count))
}

fn scan_norm_wav(
    reader: &mut (impl Read + Seek),
    file_channels: usize,
    logical_channels: usize,
) -> Result<(f64, f64, f64)> {
    let (format_tag, channels, _rate, bits, data_len, _pos) = audio::scan_wav_header(reader)?;
    if channels != file_channels {
        return Err(Error::Audio("channel count mismatch".into()));
    }
    match (format_tag, bits) {
        (3, 32) | (3, 64) | (1, 16) | (1, 24) | (1, 32) | (0xFFFE, 32) => {}
        (tag, bits) => {
            return Err(Error::Audio(format!(
                "unsupported WAV format {tag} with {bits} bits per sample"
            )))
        }
    }
    let bpf = audio::wav_bytes_per_frame(file_channels, bits)?;
    if data_len % bpf != 0 {
        return Err(Error::Audio("invalid WAV data size".into()));
    }
    let frames = data_len / bpf;
    const WINDOW: usize = 4096;
    let mut scratch = vec![0u8; WINDOW * bpf];
    let mut frame_vals = vec![0.0f32; file_channels];
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0.0f64;
    let mut done = 0usize;
    while done < frames {
        let n = (frames - done).min(WINDOW);
        reader.read_exact(&mut scratch[..n * bpf])?;
        for i in 0..n {
            audio::decode_frame(
                format_tag,
                bits,
                &scratch[i * bpf..(i + 1) * bpf],
                file_channels,
                &mut frame_vals,
            );
            let r = channel_mean(&frame_vals, file_channels, logical_channels);
            sum += r;
            sum_sq += r * r;
            count += 1.0;
        }
        done += n;
    }
    Ok((sum, sum_sq, count))
}

#[inline]
fn channel_mean(frame: &[f32], file_channels: usize, logical_channels: usize) -> f64 {
    if file_channels == logical_channels {
        let mut acc = 0.0f64;
        for c in 0..file_channels {
            acc += frame[c] as f64;
        }
        acc / file_channels as f64
    } else {
        debug_assert_eq!(file_channels, 1);
        debug_assert_eq!(logical_channels, 2);
        frame[0] as f64
    }
}

/// Sliding-window mix: Pass B reads O(segment) frames, optionally applying the
/// Pass-A global norm on the way out (`(x - mean) / std`).
pub struct WindowedMix {
    path: PathBuf,
    frames: usize,
    file_channels: usize,
    wanted_channels: usize,
    norm: Option<GlobalNorm>,
    cap: usize,
    inner: RefCell<WindowInner>,
}

struct WindowInner {
    /// Absolute frame index of slot 0 in `data`.
    start: usize,
    /// Valid frames currently buffered.
    len: usize,
    /// Channel-major file channels: `data[c * cap + i]` at absolute `start + i`.
    data: Vec<f32>,
    backend: DecodeBackend,
}

enum DecodeBackend {
    Wav {
        file: File,
        format_tag: u16,
        bits: u16,
        bpf: usize,
        data_pos: u64,
        /// Next absolute frame the sequential cursor will decode.
        cursor: usize,
    },
    Flac {
        cursor: usize,
        scale: f32,
        /// Lazy; re-opened on backward seeks.
        open: bool,
    },
}

impl WindowedMix {
    /// Open `path` for Pass-B windowed reads. `cap` is the ring capacity in
    /// frames (typically `segment + 2 * max_shift`).
    pub fn open(
        path: impl AsRef<Path>,
        wanted_channels: usize,
        cap: usize,
        norm: Option<GlobalNorm>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let info = audio::probe_audio(path)?;
        if info.channels == 0 {
            return Err(Error::Audio("zero channels".into()));
        }
        if info.channels != wanted_channels && !(info.channels == 1 && wanted_channels == 2) {
            return Err(Error::Audio(format!(
                "cannot convert {} channels to {wanted_channels}",
                info.channels
            )));
        }
        let file_channels = info.channels;
        let cap = cap.max(1);
        let mut file = File::open(path).with_path(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).with_path(path)?;
        let backend = if magic == *b"fLaC" {
            let reader = claxon::FlacReader::open(path)
                .map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
            let si = reader.streaminfo();
            let scale = 1.0f32 / (1i64 << (si.bits_per_sample - 1)) as f32;
            drop(reader);
            DecodeBackend::Flac {
                cursor: 0,
                scale,
                open: false,
            }
        } else {
            file.seek(SeekFrom::Start(0)).with_path(path)?;
            let (format_tag, channels, _rate, bits, data_len, data_pos) =
                audio::scan_wav_header(&mut file)?;
            let bpf = audio::wav_bytes_per_frame(channels, bits)?;
            if data_len % bpf != 0 {
                return Err(Error::Audio("invalid WAV data size".into()));
            }
            DecodeBackend::Wav {
                file,
                format_tag,
                bits,
                bpf,
                data_pos,
                cursor: 0,
            }
        };
        Ok(Self {
            path: path.to_path_buf(),
            frames: info.frames,
            file_channels,
            wanted_channels,
            norm,
            cap,
            inner: RefCell::new(WindowInner {
                start: 0,
                len: 0,
                data: vec![0.0f32; file_channels * cap],
                backend,
            }),
        })
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn wanted_channels(&self) -> usize {
        self.wanted_channels
    }

    /// Raw (pre-norm) sample at absolute frame `t`, logical channel `c`.
    pub fn raw_at(&self, channel: usize, t: usize) -> f32 {
        self.ensure(t);
        let inner = self.inner.borrow();
        debug_assert!(t >= inner.start && t < inner.start + inner.len);
        let i = t - inner.start;
        if self.file_channels == self.wanted_channels {
            inner.data[channel * self.cap + i]
        } else {
            // mono → stereo
            inner.data[i]
        }
    }

    fn ensure(&self, t: usize) {
        if t >= self.frames {
            return;
        }
        let mut inner = self.inner.borrow_mut();
        if t >= inner.start && t < inner.start + inner.len {
            return;
        }
        // Choose a window that covers `t` and prefers forward fills.
        let half = self.cap / 4;
        let mut new_start = t.saturating_sub(half);
        if new_start + self.cap > self.frames {
            new_start = self.frames.saturating_sub(self.cap);
        }
        let new_end = (new_start + self.cap).min(self.frames);
        let need = new_end - new_start;

        // Overlap reuse when sliding forward.
        if new_start >= inner.start && new_start < inner.start + inner.len {
            let overlap_from = new_start - inner.start;
            let overlap_len = (inner.start + inner.len - new_start).min(need);
            if overlap_from > 0 && overlap_len > 0 {
                for c in 0..self.file_channels {
                    let base = c * self.cap;
                    inner.data.copy_within(base + overlap_from..base + overlap_from + overlap_len, base);
                }
                inner.start = new_start;
                inner.len = overlap_len;
            } else if overlap_from == 0 {
                inner.start = new_start;
                // len already correct prefix
                inner.len = overlap_len;
            } else {
                inner.start = new_start;
                inner.len = 0;
            }
        } else {
            inner.start = new_start;
            inner.len = 0;
        }

        let fill_from = inner.start + inner.len;
        let fill_to = new_end;
        if fill_from < fill_to {
            Self::decode_range(
                &self.path,
                self.file_channels,
                self.cap,
                &mut inner,
                fill_from,
                fill_to,
            )
            .expect("windowed decode");
            inner.len = fill_to - inner.start;
        }
    }

    fn decode_range(
        path: &Path,
        file_channels: usize,
        cap: usize,
        inner: &mut WindowInner,
        from: usize,
        to: usize,
    ) -> Result<()> {
        match inner.backend {
            DecodeBackend::Wav { .. } => Self::decode_range_wav(path, file_channels, cap, inner, from, to),
            DecodeBackend::Flac { .. } => Self::decode_range_flac(path, file_channels, cap, inner, from, to),
        }
    }

    fn decode_range_wav(
        path: &Path,
        file_channels: usize,
        cap: usize,
        inner: &mut WindowInner,
        from: usize,
        to: usize,
    ) -> Result<()> {
        let (format_tag, bits, bpf, data_pos) = match &inner.backend {
            DecodeBackend::Wav {
                format_tag,
                bits,
                bpf,
                data_pos,
                ..
            } => (*format_tag, *bits, *bpf, *data_pos),
            _ => unreachable!(),
        };
        {
            let DecodeBackend::Wav { file, cursor, .. } = &mut inner.backend else {
                unreachable!()
            };
            if *cursor != from {
                let byte = data_pos.saturating_add((from as u64).saturating_mul(bpf as u64));
                file.seek(SeekFrom::Start(byte)).with_path(path)?;
                *cursor = from;
            }
        }
        let n = to - from;
        // Fixed stack scratch: up to 8 bytes/frame (float64 stereo) × 4096.
        const CHUNK: usize = 4096;
        let mut stack = [0u8; CHUNK * 8];
        let mut frame_vals = [0.0f32; 8];
        assert!(bpf <= 8, "WAV bytes/frame exceeds stack scratch");
        assert!(file_channels <= 8);
        let mut done = 0usize;
        while done < n {
            let m = (n - done).min(CHUNK);
            let bytes = m * bpf;
            {
                let DecodeBackend::Wav { file, cursor, .. } = &mut inner.backend else {
                    unreachable!()
                };
                file.read_exact(&mut stack[..bytes]).with_path(path)?;
                *cursor += m;
            }
            for i in 0..m {
                let raw = &stack[i * bpf..(i + 1) * bpf];
                audio::decode_frame(
                    format_tag,
                    bits,
                    raw,
                    file_channels,
                    &mut frame_vals[..file_channels],
                );
                let abs = from + done + i;
                let slot = abs - inner.start;
                for c in 0..file_channels {
                    inner.data[c * cap + slot] = frame_vals[c];
                }
            }
            done += m;
        }
        Ok(())
    }

    fn decode_range_flac(
        path: &Path,
        file_channels: usize,
        cap: usize,
        inner: &mut WindowInner,
        from: usize,
        to: usize,
    ) -> Result<()> {
        let scale = match &inner.backend {
            DecodeBackend::Flac { scale, .. } => *scale,
            _ => unreachable!(),
        };
        let mut reader = claxon::FlacReader::open(path)
            .map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
        let mut samples = reader.samples();
        let mut frame = vec![0.0f32; file_channels];
        let mut abs = 0usize;
        while abs < to {
            for c in 0..file_channels {
                let value = samples
                    .next()
                    .ok_or_else(|| Error::Audio("FLAC ended early".into()))?
                    .map_err(|e| Error::Audio(format!("{}: {e}", path.display())))?;
                frame[c] = value as f32 * scale;
            }
            if abs >= from {
                let slot = abs - inner.start;
                for c in 0..file_channels {
                    inner.data[c * cap + slot] = frame[c];
                }
            }
            abs += 1;
        }
        if let DecodeBackend::Flac { cursor, open, .. } = &mut inner.backend {
            *cursor = to;
            *open = false;
        }
        Ok(())
    }
}

impl MixSamples for WindowedMix {
    fn batch(&self) -> usize {
        1
    }
    fn channels(&self) -> usize {
        self.wanted_channels
    }
    fn length(&self) -> usize {
        self.frames
    }
    fn at(&self, batch: usize, channel: usize, t: usize) -> f32 {
        debug_assert_eq!(batch, 0);
        let raw = self.raw_at(channel, t);
        match self.norm {
            Some(n) => (raw - n.mean) / n.std,
            None => raw,
        }
    }
}

/// Centre-pad a chunk from any [`MixSamples`] (same indexing as `TensorChunk.padded`).
pub fn padded_chunk_mix<M: MixSamples + ?Sized>(
    mix: &M,
    offset: usize,
    length: usize,
    target_length: usize,
) -> Result<Array3<f32>> {
    let batch = mix.batch();
    let channels = mix.channels();
    let total = mix.length();
    if offset >= total {
        return Err(Error::Shape(format!(
            "chunk offset {offset} is past the {total}-sample signal"
        )));
    }
    let delta = target_length.checked_sub(length).ok_or_else(|| {
        Error::Shape(format!(
            "cannot pad {length} samples down to {target_length}"
        ))
    })?;
    let start = offset as isize - (delta / 2) as isize;
    let mut out = Array3::<f32>::zeros((batch, channels, target_length));
    for b in 0..batch {
        for c in 0..channels {
            for i in 0..target_length {
                let idx = start + i as isize;
                if idx >= 0 && (idx as usize) < total {
                    out[[b, c, i]] = mix.at(b, c, idx as usize);
                }
            }
        }
    }
    Ok(out)
}

/// Shift-view pad without a full `length + 2 * max_shift` allocation.
pub fn padded_chunk_shifted_mix<M: MixSamples + ?Sized>(
    mix: &M,
    shift_offset: usize,
    max_shift: usize,
    mix_length: usize,
    view_start: usize,
    chunk_length: usize,
    target_length: usize,
) -> Result<Array3<f32>> {
    let batch = mix.batch();
    let channels = mix.channels();
    let view_length = mix_length + max_shift - shift_offset;
    if view_start >= view_length {
        return Err(Error::Shape(format!(
            "chunk offset {view_start} is past the {view_length}-sample shifted signal"
        )));
    }
    let delta = target_length.checked_sub(chunk_length).ok_or_else(|| {
        Error::Shape(format!(
            "cannot pad {chunk_length} samples down to {target_length}"
        ))
    })?;
    let start = view_start as isize - (delta / 2) as isize;
    let mut out = Array3::<f32>::zeros((batch, channels, target_length));
    for b in 0..batch {
        for c in 0..channels {
            for i in 0..target_length {
                let view_index = start + i as isize;
                let sample = if view_index < 0 || view_index >= view_length as isize {
                    0.0
                } else {
                    let padded_index = shift_offset as isize + view_index;
                    let mix_index = padded_index - max_shift as isize;
                    if mix_index < 0 || mix_index >= mix_length as isize {
                        0.0
                    } else {
                        mix.at(b, c, mix_index as usize)
                    }
                };
                out[[b, c, i]] = sample;
            }
        }
    }
    Ok(out)
}

/// Suggested Pass-B ring capacity for a given segment / max_shift.
pub fn window_cap_for(segment_length: usize, max_shift: usize) -> usize {
    // Segment + centre-pad context + shift pad on both sides + one stride of slack.
    (segment_length * 2)
        .saturating_add(max_shift.saturating_mul(2))
        .saturating_add(1)
        .max(segment_length.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{write_wav_f32, Audio};
    use crate::demucs::streaming::{apply_norm_inplace, scan_norm};

    #[test]
    fn scan_norm_file_matches_array_scan() {
        let dir = std::env::temp_dir().join("demucs-mix-scan-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scan.wav");
        // channel-major stereo
        let mut data = Vec::new();
        let frames = 8000usize;
        for c in 0..2 {
            for t in 0..frames {
                data.push(((t as i32 + c as i32 * 3) % 97 - 48) as f32 / 48.0);
            }
        }
        write_wav_f32(&path, 44100, 2, &data).unwrap();
        let audio = Audio {
            sample_rate: 44100,
            samples: frames,
            data: data.clone(),
        };
        let arr = Array3::from_shape_vec((1, 2, frames), audio.data).unwrap();
        let expect = scan_norm(&arr);
        let (got, info) = scan_norm_file(&path, 2).unwrap();
        assert_eq!(info.frames, frames);
        assert_eq!(got.mean, expect.mean);
        assert_eq!(got.std, expect.std);
    }

    #[test]
    fn windowed_mix_matches_array_with_norm() {
        let dir = std::env::temp_dir().join("demucs-mix-win-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("win.wav");
        let frames = 20_000usize;
        let mut data = Vec::with_capacity(frames * 2);
        for c in 0..2 {
            for t in 0..frames {
                data.push(((t as i32 * (c as i32 + 1)) % 91 - 45) as f32 / 45.0);
            }
        }
        write_wav_f32(&path, 44100, 2, &data).unwrap();
        let mut arr = Array3::from_shape_vec((1, 2, frames), data).unwrap();
        let norm = scan_norm(&arr);
        apply_norm_inplace(&mut arr, norm);
        let win = WindowedMix::open(&path, 2, 4096, Some(norm)).unwrap();
        for t in [0, 1, 100, 4095, 4096, 10_000, frames - 1] {
            for c in 0..2 {
                let a = arr[[0, c, t]];
                let b = win.at(0, c, t);
                assert_eq!(a, b, "mismatch at c={c} t={t}: {a} vs {b}");
            }
        }
        // Exercise sliding across the file.
        for t in (0..frames).step_by(1024) {
            let _ = win.at(0, 0, t);
        }
    }

    #[test]
    fn windowed_padded_chunk_matches_array() {
        let dir = std::env::temp_dir().join("demucs-mix-pad-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pad.wav");
        let frames = 50_000usize;
        let mut data = Vec::with_capacity(frames * 2);
        for c in 0..2 {
            for t in 0..frames {
                data.push(((t as i32 * 3 + c as i32) % 89 - 44) as f32 / 44.0);
            }
        }
        write_wav_f32(&path, 44100, 2, &data).unwrap();
        let arr = Array3::from_shape_vec((1, 2, frames), data).unwrap();
        let norm = scan_norm(&arr);
        let mut normed = arr.clone();
        apply_norm_inplace(&mut normed, norm);
        let win = WindowedMix::open(&path, 2, 16_384, Some(norm)).unwrap();
        for (start, len, target) in [(0usize, 8000, 10000), (1000, 8000, 10000), (40000, 5000, 8000)] {
            let a = padded_chunk_mix(&normed, start, len, target).unwrap();
            let b = padded_chunk_mix(&win, start, len, target).unwrap();
            let worst = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(worst, 0.0, "pad mismatch start={start} worst={worst}");
        }
    }
}
