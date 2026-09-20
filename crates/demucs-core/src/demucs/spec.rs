//! `HTDemucs._spec` / `_ispec`: the STFT wrappers that keep the frame count
//! aligned with the stride-4 encoder.
//!
//! Both mirror `demucs/htdemucs.py` exactly, including the reflect padding that
//! is applied *before* `torch.stft` (so the output length is an exact multiple of
//! the hop) and the `[2 : 2 + le]` frame slice that drops the frames the padding
//! added.

use ndarray::{Array3, Array4, Array5, s};
use rustfft::num_complex::Complex32;

use crate::demucs::config::HtdemucsConfig;
use crate::dsp::stft::{PadMode, Stft, StftParams};
use crate::error::{Error, Result};

/// The forward and inverse transforms for one `scale` level.
pub struct DemucsSpec {
    forward: Stft,
    inverse: Stft,
    hop: usize,
    nfft: usize,
}

impl DemucsSpec {
    pub fn new(config: &HtdemucsConfig) -> Result<Self> {
        let params = StftParams {
            n_fft: config.nfft,
            hop_length: config.hop_length(),
            win_length: config.nfft,
            center: true,
            pad_mode: PadMode::Reflect,
            normalized: true,
        };
        Ok(Self {
            forward: Stft::new(params.clone())?,
            inverse: Stft::new(params)?,
            hop: config.hop_length(),
            nfft: config.nfft,
        })
    }

    pub fn hop(&self) -> usize {
        self.hop
    }

    pub fn nfft(&self) -> usize {
        self.nfft
    }

    /// Frame count `_spec` produces for a signal of `length` samples.
    pub fn frames_for(&self, length: usize) -> usize {
        length.div_ceil(self.hop)
    }

    /// The reflect pad `_spec` applies before `torch.stft`, so the device STFT
    /// can consume the same samples.
    pub fn pad_mix(&self, mix: &Array3<f32>) -> Result<Array3<f32>> {
        pad_mix_for_spec(mix, self.hop)
    }

    /// `HTDemucs._spec`: `(b, c, length)` samples to `(b, c, nfft/2, le)` complex.
    pub fn spec(&self, mix: &Array3<f32>) -> Result<Array4<Complex32>> {
        let (b, c, length) = mix.dim();
        let hl = self.hop;
        let le = length.div_ceil(hl);
        let pad = hl / 2 * 3;
        let right = pad + le * hl - length;
        let padded_length = length + pad + right;

        let mut padded = Array3::<f32>::zeros((b, c, padded_length));
        let source = mix
            .as_slice()
            .ok_or_else(|| Error::Shape("_spec needs a standard-layout input".into()))?;
        for bi in 0..b {
            for ci in 0..c {
                let row = &source[(bi * c + ci) * length..(bi * c + ci + 1) * length];
                let values = pad1d_reflect(row, pad, right)?;
                padded
                    .slice_mut(s![bi, ci, ..])
                    .assign(&ndarray::ArrayView1::from(&values));
            }
        }

        let flat = padded
            .view()
            .into_shape_with_order((b * c, padded_length))
            .expect("contiguous view")
            .to_owned();
        let spectra = self.forward.forward(&flat)?;
        let (_, bins, frames) = spectra.dim();
        if bins != self.nfft / 2 + 1 {
            return Err(Error::Shape(format!("unexpected bin count {bins}")));
        }
        if frames != le + 4 {
            return Err(Error::Shape(format!(
                "spectro produced {frames} frames, expected {} (le={le}, length={length})",
                le + 4
            )));
        }

        // Drop the Nyquist row, then the two padding frames on each side.
        let mut out = Array4::<Complex32>::zeros((b, c, bins - 1, le));
        for bi in 0..b {
            for ci in 0..c {
                for bin in 0..bins - 1 {
                    for frame in 0..le {
                        out[[bi, ci, bin, frame]] = spectra[[bi * c + ci, bin, frame + 2]];
                    }
                }
            }
        }
        Ok(out)
    }

    /// `HTDemucs._ispec` for the `cac` path: `(b, s, c, nfft/2, frames)` complex
    /// to `(b, s, c, length)` samples, after the caller has applied both `F.pad`s.
    pub fn ispec(&self, z: &Array5<Complex32>, length: usize, scale: usize) -> Result<Array4<f32>> {
        if scale != 0 {
            return Err(Error::UnsupportedModel(
                "_ispec with scale != 0 (the HDecLayer decoder path uses scale 0)".into(),
            ));
        }
        let (b, s, c, bins, frames) = z.dim();
        let hop = self.hop;
        let pad = hop / 2 * 3;
        let le = hop * length.div_ceil(hop) + 2 * pad;

        // `ispectro` re-derives n_fft from the bin count, so the Nyquist row has
        // to be present by now.
        let nfft = 2 * bins - 2;
        if nfft != self.nfft {
            return Err(Error::Shape(format!(
                "_ispec got {bins} bins, implying nfft={nfft}, but the transform is {}",
                self.nfft
            )));
        }

        let rows = b * s * c;
        let mut flat = vec![Complex32::default(); rows * bins * frames];
        for index in 0..rows {
            for bin in 0..bins {
                for frame in 0..frames {
                    flat[(index * bins + bin) * frames + frame] = z[[
                        index / (s * c),
                        (index / c) % s,
                        index % c,
                        bin,
                        frame,
                    ]];
                }
            }
        }
        let flat = Array3::from_shape_vec((rows, bins, frames), flat)
            .map_err(|e| Error::Shape(format!("_ispec reshape: {e}")))?;
        let rebuilt = self.inverse.inverse(&flat.view(), Some(le))?;
        if rebuilt.dim().1 < pad + length {
            return Err(Error::Shape(format!(
                "istft produced {} samples, need at least {}",
                rebuilt.dim().1,
                pad + length
            )));
        }

        let mut out = Array4::<f32>::zeros((b, s, c, length));
        let values = rebuilt
            .as_slice()
            .ok_or_else(|| Error::Shape("istft output is not contiguous".into()))?;
        for index in 0..rows {
            let (bi, si, ci) = (index / (s * c), (index / c) % s, index % c);
            let row = &values[index * rebuilt.dim().1 + pad..index * rebuilt.dim().1 + pad + length];
            out.slice_mut(s![bi, si, ci, ..])
                .assign(&ndarray::ArrayView1::from(row));
        }
        Ok(out)
    }
}

/// Reflect-pad a mix the way `_spec` does before `torch.stft`.
pub fn pad_mix_for_spec(mix: &Array3<f32>, hop: usize) -> Result<Array3<f32>> {
    let (b, c, length) = mix.dim();
    let le = length.div_ceil(hop);
    let pad = hop / 2 * 3;
    let right = pad + le * hop - length;
    let padded_length = length + pad + right;
    let mut padded = Array3::<f32>::zeros((b, c, padded_length));
    let source = mix
        .as_slice()
        .ok_or_else(|| Error::Shape("_spec needs a standard-layout input".into()))?;
    for bi in 0..b {
        for ci in 0..c {
            let row = &source[(bi * c + ci) * length..(bi * c + ci + 1) * length];
            let values = pad1d_reflect(row, pad, right)?;
            padded
                .slice_mut(s![bi, ci, ..])
                .assign(&ndarray::ArrayView1::from(&values));
        }
    }
    Ok(padded)
}

/// `pad1d(x, (left, right), mode='reflect')` from `hdemucs.py`, including its
/// fallback for inputs shorter than the padding (extra zeros are inserted before
/// the reflection so `F.pad` does not fail).
pub fn pad1d_reflect(x: &[f32], left: usize, right: usize) -> Result<Vec<f32>> {
    let length = x.len();
    let max_pad = left.max(right);
    let (extra_left, extra_right) = if length <= max_pad {
        let extra = max_pad - length + 1;
        let extra_right = right.min(extra);
        (extra - extra_right, extra_right)
    } else {
        (0, 0)
    };

    let (pad_left, pad_right) = (left - extra_left, right - extra_right);
    let mut base = Vec::with_capacity(length + extra_left + extra_right);
    base.extend(std::iter::repeat(0.0f32).take(extra_left));
    base.extend_from_slice(x);
    base.extend(std::iter::repeat(0.0f32).take(extra_right));
    if base.len() <= pad_left || base.len() <= pad_right {
        return Err(Error::Shape(format!(
            "reflect padding ({pad_left}, {pad_right}) needs more than {} samples",
            base.len()
        )));
    }

    let mut out = vec![0.0f32; base.len() + pad_left + pad_right];
    out[pad_left..pad_left + base.len()].copy_from_slice(&base);
    for i in 0..pad_left {
        out[pad_left - 1 - i] = base[i + 1];
    }
    for i in 0..pad_right {
        out[pad_left + base.len() + i] = base[base.len() - 2 - i];
    }
    Ok(out)
}

/// `torch.view_as_real(z).permute(0, 1, 4, 2, 3).reshape(b, 2c, fr, t)`: the
/// `cac=True` packing, real and imaginary parts interleaved along channels.
pub fn pack_complex_as_channels(z: &Array4<Complex32>) -> Array4<f32> {
    let (b, c, fr, t) = z.dim();
    let mut out = Array4::<f32>::zeros((b, 2 * c, fr, t));
    for bi in 0..b {
        for ci in 0..c {
            for f in 0..fr {
                for frame in 0..t {
                    out[[bi, 2 * ci, f, frame]] = z[[bi, ci, f, frame]].re;
                    out[[bi, 2 * ci + 1, f, frame]] = z[[bi, ci, f, frame]].im;
                }
            }
        }
    }
    out
}

/// `_mask`'s inverse packing: `(b, 4s, fr, t)` real to `(b, s, 2, fr, t)` complex
/// via `view(b, s, -1, 2, fr, t).permute(0, 1, 2, 4, 5, 3)`.
pub fn unpack_channels_as_complex(x: &Array4<f32>, sources: usize) -> Result<Array5<Complex32>> {
    let (b, channels, fr, t) = x.dim();
    if channels != sources * 4 {
        return Err(Error::Shape(format!(
            "expected {} channels for {sources} sources, found {channels}",
            sources * 4
        )));
    }
    let mut out = Array5::<Complex32>::zeros((b, sources, 2, fr, t));
    for bi in 0..b {
        for si in 0..sources {
            for ci in 0..2 {
                for f in 0..fr {
                    for frame in 0..t {
                        out[[bi, si, ci, f, frame]] = Complex32::new(
                            x[[bi, si * 4 + ci * 2, f, frame]],
                            x[[bi, si * 4 + ci * 2 + 1, f, frame]],
                        );
                    }
                }
            }
        }
    }
    Ok(out)
}

/// The two `F.pad` calls `_ispec` performs, as one helper:
/// `(b, s, c, fr, t) -> (b, s, c, fr + 1, t + 4)`.
pub fn pad_for_ispec(z: &Array5<Complex32>) -> Array5<Complex32> {
    let (b, s, c, fr, t) = z.dim();
    let mut out = Array5::<Complex32>::zeros((b, s, c, fr + 1, t + 4));
    out.slice_mut(s![.., .., .., ..fr, 2..2 + t]).assign(z);
    out
}
