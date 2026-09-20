//! Loader and comparators for the reference dumps written by
//! `tools/dump_reference.py`.
//!
//! The dumps are a directory of `.npy` files plus a `manifest.json` describing
//! each tensor's name, shape and provenance. Keeping them as separate files
//! means validation can stream one tensor at a time instead of materialising a
//! multi-gigabyte archive.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::npy::{read_npy, NpyArray};

#[derive(Debug, Clone, Deserialize)]
pub struct TensorEntry {
    pub index: usize,
    pub name: String,
    pub kind: String,
    pub file: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub meta: serde_json::Value,
    pub tensors: Vec<TensorEntry>,
}

#[derive(Debug, Clone)]
pub struct ReferenceDump {
    dir: PathBuf,
    pub manifest: Manifest,
}

impl ReferenceDump {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let text = std::fs::read_to_string(dir.join("manifest.json")).map_err(|e| {
            Error::Shape(format!("no reference dump at {}: {e}", dir.display()))
        })?;
        let manifest: Manifest = serde_json::from_str(&text)
            .map_err(|e| Error::Shape(format!("bad manifest in {}: {e}", dir.display())))?;
        Ok(Self { dir, manifest })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn meta_usize(&self, key: &str) -> Option<usize> {
        self.manifest.meta.get(key)?.as_u64().map(|v| v as usize)
    }

    pub fn meta_f64(&self, key: &str) -> Option<f64> {
        self.manifest.meta.get(key)?.as_f64()
    }

    pub fn entry(&self, name: &str) -> Option<&TensorEntry> {
        self.manifest.tensors.iter().find(|t| t.name == name)
    }

    /// All tensors whose name starts with `prefix`, in dump order.
    pub fn entries_starting_with(&self, prefix: &str) -> Vec<&TensorEntry> {
        self.manifest
            .tensors
            .iter()
            .filter(|t| t.name.starts_with(prefix))
            .collect()
    }

    pub fn load(&self, name: &str) -> Result<NpyArray> {
        let entry = self.entry(name).ok_or_else(|| {
            Error::Shape(format!("reference dump has no tensor named `{name}`"))
        })?;
        self.load_entry(entry)
    }

    pub fn load_entry(&self, entry: &TensorEntry) -> Result<NpyArray> {
        let array = read_npy(self.dir.join(&entry.file))?;
        if array.shape != entry.shape {
            return Err(Error::Shape(format!(
                "{}: manifest says {:?} but the .npy holds {:?}",
                entry.name, entry.shape, array.shape
            )));
        }
        Ok(array)
    }

    pub fn load_if_present(&self, name: &str) -> Result<Option<NpyArray>> {
        match self.entry(name) {
            Some(entry) => Ok(Some(self.load_entry(entry)?)),
            None => Ok(None),
        }
    }
}

/// Difference between a reference tensor and the value under test.
#[derive(Debug, Clone, Copy, Default)]
pub struct Comparison {
    pub elements: usize,
    pub max_abs: f32,
    pub max_abs_index: usize,
    pub max_rel: f32,
    pub mean_abs: f32,
    pub reference_max_abs: f32,
    pub reference_rms: f32,
}

impl Comparison {
    /// Error relative to the scale of the reference signal, which is the metric
    /// that stays meaningful as activations shrink layer by layer.
    pub fn normalized_error(&self) -> f32 {
        if self.reference_max_abs > 0.0 {
            self.max_abs / self.reference_max_abs
        } else {
            self.max_abs
        }
    }

    /// Typical error relative to the reference, the RMS-error counterpart of
    /// [`Comparison::normalized_error`].
    ///
    /// This is the metric to gate on. The max-abs form is dominated by the worst
    /// single element, and Mel-Band Roformer has genuinely ill-conditioned spots:
    /// `band_split` normalises each mel band by that band's own L2 norm, and the
    /// top bands (above ~15 kHz) carry norms around 1.7e-2 against ~4e1 in the
    /// low bands. Perturbing such a band's input by 1e-7 changes its normalised
    /// output by ~50%, so any two correct implementations of the same maths
    /// disagree by ~1e-3 there on float32 round-off alone.
    pub fn rms_relative_error(&self) -> f32 {
        if self.reference_rms > 0.0 {
            self.mean_abs / self.reference_rms
        } else {
            self.mean_abs
        }
    }

    /// Signal-to-noise ratio in dB between reference and test.
    pub fn snr_db(&self) -> f32 {
        let relative = self.rms_relative_error();
        if relative <= 0.0 {
            f32::INFINITY
        } else {
            -20.0 * relative.log10()
        }
    }
}

/// Compare two flat buffers element by element.
pub fn compare(expected: &[f32], actual: &[f32]) -> Result<Comparison> {
    if expected.len() != actual.len() {
        return Err(Error::Shape(format!(
            "cannot compare {} elements against {}",
            expected.len(),
            actual.len()
        )));
    }
    let mut result = Comparison {
        elements: expected.len(),
        ..Default::default()
    };
    let mut sum_abs = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut sum_sq_actual = 0.0f64;
    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        if diff > result.max_abs {
            result.max_abs = diff;
            result.max_abs_index = i;
        }
        let scale = e.abs().max(1e-6);
        let rel = diff / scale;
        if rel > result.max_rel {
            result.max_rel = rel;
        }
        sum_abs += diff as f64;
        sum_sq_ref += (e as f64) * (e as f64);
        sum_sq_actual += (a as f64) * (a as f64);
        result.reference_max_abs = result.reference_max_abs.max(e.abs());
    }
    if !expected.is_empty() {
        result.mean_abs = (sum_abs / expected.len() as f64) as f32;
        result.reference_rms = (sum_sq_ref / expected.len() as f64).sqrt() as f32;
    }
    let _ = sum_sq_actual;
    Ok(result)
}

/// Expand an interleaved complex buffer `(..., 2)` into `Complex32` values.
pub fn deinterleave_complex(flat: &[f32]) -> Vec<rustfft::num_complex::Complex32> {
    flat.chunks_exact(2)
        .map(|c| rustfft::num_complex::Complex32::new(c[0], c[1]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_reports_scale_relative_error() {
        let reference = [1.0f32, 2.0, 3.0];
        let actual = [1.0f32, 2.0, 3.0003];
        let result = compare(&reference, &actual).unwrap();
        assert_eq!(result.elements, 3);
        assert!((result.max_abs - 0.0003).abs() < 1e-7);
        assert!(result.normalized_error() < 1e-4);
    }

    #[test]
    fn comparison_rejects_length_mismatches() {
        assert!(compare(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn identical_buffers_have_zero_error() {
        let values = [0.5f32, -0.25];
        let result = compare(&values, &values).unwrap();
        assert_eq!(result.max_abs, 0.0);
        assert_eq!(result.max_rel, 0.0);
    }

    #[test]
    fn complex_buffers_split_into_real_and_imaginary() {
        let values = [1.0f32, -1.0, 2.0, -2.0];
        let complex = deinterleave_complex(&values);
        assert_eq!(complex.len(), 2);
        assert_eq!(complex[0].re, 1.0);
        assert_eq!(complex[0].im, -1.0);
    }
}
