//! `torch.stft` / `torch.istft` compatible transforms.
//!
//! Every model config carries its own `n_fft` / `hop_length` / `win_length`, so
//! nothing here may be hard-coded. The semantics mirrored are PyTorch's:
//!
//! * `center=True` reflect-pads the signal by `n_fft / 2` on both sides,
//! * frames are strided by `hop_length` and windowed by a periodic Hann window
//!   zero-padded from `win_length` up to `n_fft`,
//! * forward uses an unnormalised real FFT (`norm="backward"`), giving
//!   `n_fft / 2 + 1` bins,
//! * inverse overlap-adds `irfft(spec) * window` and divides by the summed
//!   squared window envelope, then trims the centre padding. With
//!   `length = None` the result is exactly `hop_length * (frames - 1)` samples.

use std::sync::Arc;

use ndarray::{Array2, Array3, ArrayView3};
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Reflect,
    Constant,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StftParams {
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub center: bool,
    pub pad_mode: PadMode,
    pub normalized: bool,
}

impl StftParams {
    pub fn new(n_fft: usize, hop_length: usize, win_length: usize) -> Self {
        Self {
            n_fft,
            hop_length,
            win_length,
            center: true,
            pad_mode: PadMode::Reflect,
            normalized: false,
        }
    }
}

/// Periodic Hann window, matching `torch.hann_window(n)` with `periodic=True`.
pub fn hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / length as f64;
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect()
}

pub struct Stft {
    params: StftParams,
    /// Window of `win_length`, zero-padded to `n_fft` as PyTorch does.
    window: Vec<f32>,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
}

impl Stft {
    pub fn new(params: StftParams) -> Result<Self> {
        let window = hann_window(params.win_length);
        Self::with_window(params, window)
    }

    pub fn with_window(params: StftParams, window: Vec<f32>) -> Result<Self> {
        if window.len() != params.win_length {
            return Err(Error::Shape(format!(
                "window has {} samples but win_length is {}",
                window.len(),
                params.win_length
            )));
        }
        if params.win_length > params.n_fft {
            return Err(Error::Shape(
                "win_length must not exceed n_fft".to_string(),
            ));
        }
        let mut planner = FftPlanner::new();
        Ok(Self {
            forward: planner.plan_fft_forward(params.n_fft),
            inverse: planner.plan_fft_inverse(params.n_fft),
            window: pad_window(&window, params.n_fft),
            params,
        })
    }

    pub fn params(&self) -> &StftParams {
        &self.params
    }

    pub fn window(&self) -> &[f32] {
        &self.window
    }

    pub fn bins(&self) -> usize {
        self.params.n_fft / 2 + 1
    }

    pub fn frames_for(&self, samples: usize) -> usize {
        let padded = if self.params.center {
            samples + self.params.n_fft
        } else {
            samples
        };
        if padded < self.params.n_fft {
            0
        } else {
            1 + (padded - self.params.n_fft) / self.params.hop_length
        }
    }

    /// `(batch, bins, frames)`, matching `torch.stft(..., return_complex=True)`
    /// applied to a `(batch, samples)` input.
    pub fn forward(&self, input: &Array2<f32>) -> Result<Array3<Complex32>> {
        let (batch, samples) = input.dim();
        let frames = self.frames_for(samples);
        let bins = self.bins();
        let mut out = Array3::<Complex32>::zeros((batch, bins, frames));

        let pad = if self.params.center {
            self.params.n_fft / 2
        } else {
            0
        };
        if pad > 0 && self.params.pad_mode == PadMode::Reflect && pad >= samples {
            return Err(Error::Shape(format!(
                "reflect padding of {pad} samples needs an input longer than {pad}, got {samples}"
            )));
        }

        let norm_scale = if self.params.normalized {
            1.0 / (self.params.n_fft as f32).sqrt()
        } else {
            1.0
        };

        // Frames within a row are independent, so parallelise over (batch, frame).
        let jobs: Vec<(usize, usize)> = (0..batch)
            .flat_map(|b| (0..frames).map(move |f| (b, f)))
            .collect();

        let rows: Vec<Vec<f32>> = input
            .axis_iter(ndarray::Axis(0))
            .map(|row| {
                let slice = row.as_slice().expect("input rows are contiguous");
                if pad > 0 {
                    pad_signal(slice, pad, self.params.pad_mode)
                } else {
                    slice.to_vec()
                }
            })
            .collect();

        let frame_len = self.params.n_fft;
        let hop = self.params.hop_length;

        use rayon::prelude::*;
        // Each job produces a column of `out`; ndarray cannot prove the writes
        // are disjoint, so results are collected and placed afterwards.
        let results: Vec<((usize, usize), Vec<Complex32>)> = jobs
            .par_iter()
            .map(|&(b, f)| {
                let start = f * hop;
                let row = &rows[b];
                let mut scratch = vec![Complex32::default(); self.forward.get_inplace_scratch_len()];
                let mut buffer = vec![Complex32::default(); frame_len];
                for i in 0..frame_len {
                    buffer[i] = Complex32::new(row[start + i] * self.window[i], 0.0);
                }
                self.forward.process_with_scratch(&mut buffer, &mut scratch);
                buffer.truncate(bins);
                ((b, f), buffer)
            })
            .collect();

        for ((b, f), values) in results {
            for (bin, value) in values.into_iter().enumerate() {
                out[[b, bin, f]] = value * norm_scale;
            }
        }

        Ok(out)
    }

    /// Inverse of [`Stft::forward`]; `(batch, bins, frames)` to `(batch, samples)`.
    pub fn inverse(&self, spec: &ArrayView3<Complex32>, length: Option<usize>) -> Result<Array2<f32>> {
        let (batch, bins, frames) = spec.dim();
        if bins != self.bins() {
            return Err(Error::Shape(format!(
                "spectrogram has {bins} bins but n_fft={} needs {}",
                self.params.n_fft,
                self.bins()
            )));
        }
        if frames == 0 {
            return Err(Error::Shape("cannot invert an empty spectrogram".into()));
        }

        let n_fft = self.params.n_fft;
        let hop = self.params.hop_length;
        let expected = n_fft + hop * (frames - 1);

        // Centre trimming mirrors torch.istft: with `length = None` the result is
        // exactly `hop * (frames - 1)` samples.
        let start = if self.params.center { n_fft / 2 } else { 0 };
        let end = match length {
            Some(len) => start + len,
            None if self.params.center => expected - n_fft / 2,
            None => expected,
        };
        let end = end.min(expected);
        let out_len = end.saturating_sub(start);

        // `normalized=True` scales the forward transform *down* by `sqrt(n_fft)`,
        // so the inverse has to scale it back up: `torch.istft` multiplies by
        // `sqrt(n_fft)` where `torch.stft` divided by it.
        let norm_scale = if self.params.normalized {
            (n_fft as f32).sqrt()
        } else {
            1.0
        };

        let mut scratch = vec![Complex32::default(); self.inverse.get_inplace_scratch_len()];
        let mut buffer = vec![Complex32::default(); n_fft];

        // Frame contributions overlap, so accumulate serially per batch row.
        let mut result = Array2::<f32>::zeros((batch, out_len));
        for b in 0..batch {
            let mut y = vec![0.0f32; expected];
            let mut envelope = vec![0.0f32; expected];
            for f in 0..frames {
                // Rebuild the full Hermitian spectrum from the one-sided half.
                buffer[0] = spec[[b, 0, f]];
                for k in 1..bins {
                    let value = spec[[b, k, f]];
                    buffer[k] = value;
                    buffer[n_fft - k] = value.conj();
                }

                self.inverse.process_with_scratch(&mut buffer, &mut scratch);

                let offset = f * hop;
                let inv_n = norm_scale / n_fft as f32;
                for i in 0..n_fft {
                    let w = self.window[i];
                    y[offset + i] += buffer[i].re * inv_n * w;
                    envelope[offset + i] += w * w;
                }
            }

            for i in start..end {
                let denom = envelope[i];
                result[[b, i - start]] = if denom != 0.0 { y[i] / denom } else { 0.0 };
            }
        }
        Ok(result)
    }
}

fn pad_window(window: &[f32], n_fft: usize) -> Vec<f32> {
    if window.len() == n_fft {
        return window.to_vec();
    }
    // PyTorch centres a short window inside the FFT size.
    let left = (n_fft - window.len()) / 2;
    let mut padded = vec![0.0f32; n_fft];
    padded[left..left + window.len()].copy_from_slice(window);
    padded
}

/// `F.pad(mode='reflect')`: mirrors about the edge *without* duplicating it.
pub fn pad_signal(signal: &[f32], pad: usize, mode: PadMode) -> Vec<f32> {
    let len = signal.len();
    let mut out = vec![0.0f32; len + 2 * pad];
    out[pad..pad + len].copy_from_slice(signal);
    match mode {
        PadMode::Constant => {}
        PadMode::Reflect => {
            for i in 0..pad {
                // Left: padded[pad - 1 - i] = signal[i + 1]
                out[pad - 1 - i] = signal[i + 1];
                // Right: padded[pad + len + i] = signal[len - 2 - i]
                out[pad + len + i] = signal[len - 2 - i];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn hann_window_matches_the_periodic_definition() {
        let window = hann_window(4);
        assert!((window[0] - 0.0).abs() < 1e-7);
        assert!((window[1] - 0.5).abs() < 1e-7);
        assert!((window[2] - 1.0).abs() < 1e-7);
        assert!((window[3] - 0.5).abs() < 1e-7);
    }

    #[test]
    fn reflect_padding_mirrors_without_repeating_the_edge() {
        let padded = pad_signal(&[1.0, 2.0, 3.0, 4.0], 3, PadMode::Reflect);
        assert_eq!(padded, vec![4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn frame_count_follows_center_padding() {
        let stft = Stft::new(StftParams::new(2048, 441, 2048)).unwrap();
        // centre-padding adds n_fft samples, so frames = 1 + samples / hop.
        assert_eq!(stft.frames_for(26460), 61);
        assert_eq!(stft.frames_for(441 * 60), 61);
    }

    #[test]
    fn a_single_bin_sinusoid_lands_in_the_expected_bin() {
        let stft = Stft::new(StftParams::new(64, 16, 64)).unwrap();
        let period = 8.0; // bin = 64 / 8 = 8
        let signal: Vec<f32> = (0..256)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / period).sin())
            .collect();
        let input = Array2::from_shape_vec((1, 256), signal).unwrap();
        let spec = stft.forward(&input).unwrap();
        let peak = (0..stft.bins())
            .max_by(|a, b| {
                spec[[0, *a, 4]].norm().partial_cmp(&spec[[0, *b, 4]].norm()).unwrap()
            })
            .unwrap();
        assert_eq!(peak, 8);
    }

    #[test]
    fn forward_then_inverse_reconstructs_the_signal_interior() {
        // hop = n_fft / 4 keeps the Hann envelope well conditioned.
        let stft = Stft::new(StftParams::new(256, 64, 256)).unwrap();
        let signal: Vec<f32> = (0..2048)
            .map(|i| {
                let t = i as f32;
                (t * 0.013).sin() * 0.7 + (t * 0.11).cos() * 0.3
            })
            .collect();
        let input = Array2::from_shape_vec((1, signal.len()), signal.clone()).unwrap();
        let spec = stft.forward(&input).unwrap();
        let rebuilt = stft.inverse(&spec.view(), None).unwrap();
        // centre padding is added then removed, so the length is preserved
        assert_eq!(rebuilt.dim(), (1, signal.len()));

        // Skip the first/last window length where the envelope ramps.
        let guard = 300;
        let mut worst = 0.0f32;
        for i in guard..signal.len() - guard {
            worst = worst.max((rebuilt[[0, i]] - signal[i]).abs());
        }
        assert!(worst < 1e-4, "worst reconstruction error {worst}");
    }

    #[test]
    fn normalized_forward_and_inverse_round_trip() {
        // `torch.stft(..., normalized=True)` divides by `sqrt(n_fft)` and
        // `torch.istft(..., normalized=True)` multiplies it back; getting either
        // sign wrong scales the whole model output.
        let mut params = StftParams::new(256, 64, 256);
        params.normalized = true;
        let stft = Stft::new(params).unwrap();
        let signal: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.037).sin()).collect();
        let input = Array2::from_shape_vec((1, signal.len()), signal.clone()).unwrap();
        let spec = stft.forward(&input).unwrap();
        // The forward divides by sqrt(n_fft) relative to the unnormalised form.
        let unnormalized = Stft::new(StftParams::new(256, 64, 256)).unwrap();
        let raw = unnormalized.forward(&input).unwrap();
        let ratio = raw[[0, 8, 4]].norm() / spec[[0, 8, 4]].norm();
        assert!((ratio - 16.0).abs() < 1e-2, "sqrt(256) = 16, got {ratio}");

        let rebuilt = stft.inverse(&spec.view(), Some(signal.len())).unwrap();
        let mut worst = 0.0f32;
        for i in 200..signal.len() - 200 {
            worst = worst.max((rebuilt[[0, i]] - signal[i]).abs());
        }
        assert!(worst < 1e-4, "worst reconstruction error {worst}");
    }

    #[test]
    fn inverse_length_matches_hop_times_frames_when_length_is_none() {
        let stft = Stft::new(StftParams::new(2048, 441, 2048)).unwrap();
        let spec = Array3::<Complex32>::zeros((2, stft.bins(), 5));
        let out = stft.inverse(&spec.view(), None).unwrap();
        assert_eq!(out.dim(), (2, 441 * 4));
    }

    #[test]
    fn explicit_length_is_honoured() {
        let stft = Stft::new(StftParams::new(2048, 441, 2048)).unwrap();
        let spec = Array3::<Complex32>::zeros((1, stft.bins(), 5));
        let out = stft.inverse(&spec.view(), Some(1000)).unwrap();
        assert_eq!(out.dim(), (1, 1000));
    }

    #[test]
    fn reflect_padding_rejects_inputs_shorter_than_the_pad() {
        let stft = Stft::new(StftParams::new(2048, 441, 2048)).unwrap();
        let input = array![[0.0f32; 100]];
        assert!(stft.forward(&input).is_err());
    }
}
