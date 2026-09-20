//! Reader for the `safetensors` format.
//!
//! Layout: a little-endian `u64` header length, a JSON header, then the raw
//! tensor data. Each header entry (except the reserved `__metadata__`) is
//! `{"dtype": "F16", "shape": [...], "data_offsets": [begin, end]}` with the
//! offsets relative to the start of the data section.
//!
//! The HuggingFace checkpoints demucs ships store the weights as `F16` and the
//! model configuration inside `__metadata__` (`klass`, `kwargs`, `args`), so the
//! returned metadata is what the config parser consumes.

use std::collections::BTreeMap;
use std::path::Path;

use ndarray::ArrayD;

use crate::error::{Error, Result};

/// Tensors plus the `__metadata__` string map, both as found in the file.
#[derive(Debug, Clone, Default)]
pub struct SafeTensors {
    pub tensors: BTreeMap<String, ArrayD<f32>>,
    pub metadata: BTreeMap<String, String>,
}

impl SafeTensors {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Checkpoint(format!("{}: {e}", path.display())))?;
        Self::parse(&bytes)
    }

    pub fn get(&self, name: &str) -> Option<&ArrayD<f32>> {
        self.tensors.get(name)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::Checkpoint("safetensors file is shorter than its header length".into()));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let start = 8usize;
        let end = start
            .checked_add(header_len)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| {
                Error::Checkpoint(format!(
                    "safetensors header claims {header_len} bytes but the file holds {}",
                    bytes.len()
                ))
            })?;
        let header: serde_json::Value = serde_json::from_slice(&bytes[start..end])
            .map_err(|e| Error::Checkpoint(format!("safetensors header is not JSON: {e}")))?;
        let data = &bytes[end..];

        let mut out = SafeTensors::default();
        let entries = header
            .as_object()
            .ok_or_else(|| Error::Checkpoint("safetensors header is not an object".into()))?;

        for (name, entry) in entries {
            if name == "__metadata__" {
                if let Some(map) = entry.as_object() {
                    for (key, value) in map {
                        if let Some(text) = value.as_str() {
                            out.metadata.insert(key.clone(), text.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = entry
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Checkpoint(format!("{name}: no dtype")))?;
            let shape: Vec<usize> = entry
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Checkpoint(format!("{name}: no shape")))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let offsets = entry
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| Error::Checkpoint(format!("{name}: no data_offsets")))?;
            if offsets.len() != 2 {
                return Err(Error::Checkpoint(format!("{name}: malformed data_offsets")));
            }
            let begin = offsets[0].as_u64().unwrap_or(0) as usize;
            let stop = offsets[1].as_u64().unwrap_or(0) as usize;
            if stop < begin || stop > data.len() {
                return Err(Error::Checkpoint(format!(
                    "{name}: data range [{begin}, {stop}) is outside the {} data bytes",
                    data.len()
                )));
            }
            let expected: usize = shape.iter().product();
            let values = decode(&data[begin..stop], dtype, expected, name)?;
            let array = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), values)
                .map_err(|e| Error::Checkpoint(format!("{name}: {e}")))?;
            out.tensors.insert(name.clone(), array);
        }
        Ok(out)
    }
}

/// Decodes one tensor blob; `expected` is the element count implied by the shape.
pub fn decode(raw: &[u8], dtype: &str, expected: usize, name: &str) -> Result<Vec<f32>> {
    let (width, decode_one): (usize, fn(&[u8]) -> f32) = match dtype {
        "F32" => (4, |b| f32::from_le_bytes(b.try_into().unwrap())),
        "F64" => (8, |b| f64::from_le_bytes(b.try_into().unwrap()) as f32),
        "F16" => (2, |b| half_to_f32(u16::from_le_bytes(b.try_into().unwrap()))),
        "BF16" => (2, |b| {
            f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16)
        }),
        "I64" => (8, |b| i64::from_le_bytes(b.try_into().unwrap()) as f32),
        "I32" => (4, |b| i32::from_le_bytes(b.try_into().unwrap()) as f32),
        "I16" => (2, |b| i16::from_le_bytes(b.try_into().unwrap()) as f32),
        "U8" | "BOOL" => (1, |b| b[0] as f32),
        "I8" => (1, |b| b[0] as i8 as f32),
        other => {
            return Err(Error::Checkpoint(format!(
                "{name}: unsupported safetensors dtype {other}"
            )))
        }
    };
    if raw.len() != expected * width {
        return Err(Error::Checkpoint(format!(
            "{name}: {} data bytes do not match {expected} elements of {dtype}",
            raw.len()
        )));
    }
    Ok(raw.chunks_exact(width).map(decode_one).collect())
}

/// IEEE 754 binary16 to binary32, subnormals and infinities included.
pub fn half_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let value = match exponent {
        0 => {
            if mantissa == 0 {
                0.0
            } else {
                (mantissa as f32) * 2f32.powi(-24)
            }
        }
        0x1f => {
            if mantissa == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => f32::from_bits(((exponent + 112) << 23) | (mantissa << 13)),
    };
    if sign == 1 {
        -value
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal one-tensor file so the parser is exercised end to end.
    fn build(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn a_single_f32_tensor_round_trips() {
        let header = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let parsed = SafeTensors::parse(&build(header, &data)).unwrap();
        let w = parsed.get("w").unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(w[[0, 1]], 2.0);
        assert_eq!(w[[1, 1]], 4.0);
    }

    #[test]
    fn f16_tensors_are_upcast() {
        // 1.0, -2.0, 0.5, 0.0 in binary16.
        let values: [u16; 4] = [0x3c00, 0xc000, 0x3800, 0x0000];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = r#"{"w":{"dtype":"F16","shape":[4],"data_offsets":[0,8]}}"#;
        let parsed = SafeTensors::parse(&build(header, &data)).unwrap();
        assert_eq!(parsed.get("w").unwrap().iter().copied().collect::<Vec<_>>(),
                   vec![1.0, -2.0, 0.5, 0.0]);
    }

    #[test]
    fn metadata_is_exposed_and_not_treated_as_a_tensor() {
        let header = r#"{"__metadata__":{"klass":"demucs.htdemucs.HTDemucs"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let parsed = SafeTensors::parse(&build(header, &0.5f32.to_le_bytes())).unwrap();
        assert_eq!(parsed.tensors.len(), 1);
        assert_eq!(
            parsed.metadata.get("klass").map(String::as_str),
            Some("demucs.htdemucs.HTDemucs")
        );
    }

    #[test]
    fn truncated_data_is_rejected() {
        let header = r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        assert!(SafeTensors::parse(&build(header, &[0u8; 8])).is_err());
    }

    #[test]
    fn a_header_longer_than_the_file_is_rejected() {
        let bytes = u64::to_le_bytes(4096).to_vec();
        assert!(SafeTensors::parse(&bytes).is_err());
    }

    #[test]
    fn half_decoding_handles_subnormals_and_infinities() {
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
        assert!(half_to_f32(0x7c00).is_infinite());
        assert!(half_to_f32(0xfc00).is_sign_negative());
        assert!(half_to_f32(0x7e00).is_nan());
    }
}
