//! Reader for PyTorch `.ckpt` / `.pth` archives.
//!
//! A `torch.save` file is a zip holding:
//!   * `<root>/data.pkl`   – a pickle describing the object graph,
//!   * `<root>/data/<key>` – one raw little-endian blob per storage,
//!   * `<root>/byteorder`  – storage endianness.
//!
//! We parse the pickle ourselves (`super::pickle`) and materialise every tensor
//! as a contiguous row-major `f32` array, so no Python or libtorch is needed at
//! run time.

pub mod pickle;

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use ndarray::ArrayD;

use crate::error::{Error, IoContext, Result};
use pickle::{Dtype, TensorRef, Unpickler, Value};

/// Endianness of the storage blobs, from the archive's `byteorder` record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

/// A checkpoint's tensors, keyed by their state-dict name.
#[derive(Debug, Default)]
pub struct Checkpoint {
    tensors: BTreeMap<String, ArrayD<f32>>,
    source: PathBuf,
}

impl Checkpoint {
    /// Loads every tensor in the archive.
    ///
    /// Checkpoints saved by MSST training runs wrap the weights in a dict with
    /// `model_state_dict` / `state_dict` / `state`; plain inference checkpoints
    /// are the bare state dict. Both are handled.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let bytes = std::fs::read(&path).with_path(&path)?;
        Self::from_zip_bytes(&bytes, path)
    }

    /// Loads a torch-zip checkpoint straight from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_zip_bytes(bytes, PathBuf::from("<bytes>"))
    }

    fn from_zip_bytes(bytes: &[u8], path: PathBuf) -> Result<Self> {
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes.to_vec()))
            .map_err(|e| Error::Checkpoint(format!("{}: {e}", path.display())))?;

        let root = find_root(&mut archive)?;
        let byteorder = read_optional(&mut archive, &format!("{root}/byteorder"))?;
        let endian = match byteorder.as_deref().map(|s| s.trim()) {
            Some("little") => Endian::Little,
            Some("big") => Endian::Big,
            _ => Endian::Little,
        };

        let pickle_bytes = read_entry(&mut archive, &format!("{root}/data.pkl"))?;
        let root_value = Unpickler::new(&pickle_bytes).run()?;

        let state = extract_state_dict(root_value)?;

        // Materialise storage by storage so only one blob is resident at a time;
        // several tensors may be views onto the same storage.
        let mut tensors: BTreeMap<String, ArrayD<f32>> = BTreeMap::new();
        let mut by_key: BTreeMap<String, Vec<(String, TensorRef)>> = BTreeMap::new();
        for (name, value) in state {
            match value {
                Value::Tensor(tref) => {
                    if tref.size.is_none() {
                        return Err(Error::Checkpoint(format!(
                            "tensor `{name}` is sparse, which is not supported"
                        )));
                    }
                    by_key
                        .entry(tref.storage.key.clone())
                        .or_default()
                        .push((name, tref));
                }
                Value::None | Value::Opaque(_) => continue,
                other => {
                    return Err(Error::Checkpoint(format!(
                        "state dict entry `{name}` is not a tensor: {other:?}"
                    )))
                }
            }
        }

        for (key, refs) in by_key {
            let entry = format!("{root}/data/{key}");
            let raw = read_entry(&mut archive, &entry)?;
            let dtype = refs[0].1.storage.dtype;
            let elements = decode_storage(&raw, dtype, endian)?;
            for (name, tref) in refs {
                let numel = tref.storage.numel;
                if elements.len() < numel {
                    return Err(Error::Checkpoint(format!(
                        "storage {key} holds {} elements but is declared with {numel}",
                        elements.len()
                    )));
                }
                let shape = tref.size.clone().unwrap_or_default();
                let expected: usize = shape.iter().product();
                let data = gather(&elements, &tref, expected)?;
                let array = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), data)
                    .map_err(|e| Error::Checkpoint(format!("tensor `{name}`: {e}")))?;
                tensors.insert(name, array);
            }
        }

        Ok(Self {
            tensors,
            source: path,
        })
    }

    pub fn source(&self) -> &Path {
        &self.source
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&ArrayD<f32>> {
        self.tensors.get(name)
    }

    /// The backing map, for models that want to index weights themselves.
    pub fn tensors(&self) -> &BTreeMap<String, ArrayD<f32>> {
        &self.tensors
    }

    /// Builds a checkpoint from an already-decoded tensor map, so a reader for a
    /// different container (safetensors) can feed the same consumers.
    pub fn from_tensors(tensors: BTreeMap<String, ArrayD<f32>>, source: impl Into<PathBuf>) -> Self {
        Self {
            tensors,
            source: source.into(),
        }
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn get_required(&self, name: &str) -> Result<&ArrayD<f32>> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::MissingWeight(name.to_string()))
    }

    /// Fetch a weight and assert the exact shape the model expects.
    pub fn get_shaped(&self, name: &str, expected: &[usize]) -> Result<&ArrayD<f32>> {
        let tensor = self.get_required(name)?;
        let found = tensor.shape().to_vec();
        if found != expected {
            return Err(Error::WeightShape {
                name: name.to_string(),
                found,
                expected: expected.to_vec(),
            });
        }
        Ok(tensor)
    }
}

/// The archive prefix is derived from the file name at save time, so locate it
/// via the `data.pkl` entry rather than assuming a fixed value.
fn find_root<R: Read + std::io::Seek>(archive: &mut zip::ZipArchive<R>) -> Result<String> {
    for i in 0..archive.len() {
        let entry = archive
            .by_index(i)
            .map_err(|e| Error::Checkpoint(format!("bad zip entry {i}: {e}")))?;
        let name = entry.name().to_string();
        if let Some(root) = name.strip_suffix("/data.pkl") {
            return Ok(root.to_string());
        }
    }
    Err(Error::Checkpoint(
        "archive contains no `<root>/data.pkl`; is this a torch.save file?".into(),
    ))
}

fn read_entry<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
) -> Result<Vec<u8>> {
    let mut entry = archive
        .by_name(name)
        .map_err(|e| Error::Checkpoint(format!("missing archive entry {name}: {e}")))?;
    let mut buf = Vec::with_capacity(entry.size() as usize);
    entry
        .read_to_end(&mut buf)
        .map_err(|e| Error::Checkpoint(format!("reading {name}: {e}")))?;
    Ok(buf)
}

fn read_optional<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
) -> Result<Option<String>> {
    match archive.by_name(name) {
        Ok(mut entry) => {
            let mut buf = String::new();
            entry
                .read_to_string(&mut buf)
                .map_err(|e| Error::Checkpoint(format!("reading {name}: {e}")))?;
            Ok(Some(buf))
        }
        Err(_) => Ok(None),
    }
}

/// Unwraps the usual checkpoint containers to reach the `name -> tensor` mapping.
fn extract_state_dict(root: Value) -> Result<Vec<(String, Value)>> {
    let Value::Dict(entries) = root else {
        return Err(Error::Checkpoint(format!(
            "checkpoint top level is not a dict: {root:?}"
        )));
    };

    for key in ["model_state_dict", "state_dict", "state"] {
        if let Some((_, inner)) = entries.iter().find(|(k, _)| k.as_str().ok() == Some(key)) {
            if let Value::Dict(inner_entries) = inner {
                // Guard against metadata that merely happens to share the name.
                if looks_like_state_dict(inner_entries) {
                    return named_entries(inner_entries);
                }
            }
        }
    }

    if looks_like_state_dict(&entries) {
        return named_entries(&entries);
    }

    Err(Error::Checkpoint(
        "checkpoint has no tensor state dict (looked for model_state_dict/state_dict/state)".into(),
    ))
}

fn named_entries(entries: &[(Value, Value)]) -> Result<Vec<(String, Value)>> {
    entries
        .iter()
        .map(|(k, v)| Ok((k.as_str()?.to_string(), v.clone())))
        .collect()
}

fn looks_like_state_dict(entries: &[(Value, Value)]) -> bool {
    entries
        .iter()
        .any(|(_, v)| matches!(v, Value::Tensor(t) if t.size.is_some()))
}

/// Decodes a raw storage blob into `f32` values.
fn decode_storage(raw: &[u8], dtype: Dtype, endian: Endian) -> Result<Vec<f32>> {
    let size = dtype.size();
    if raw.len() % size != 0 {
        return Err(Error::Checkpoint(format!(
            "storage length {} is not a multiple of the {size}-byte element size",
            raw.len()
        )));
    }
    let count = raw.len() / size;
    let mut out = Vec::with_capacity(count);

    macro_rules! read_ints {
        ($ty:ty, $width:expr) => {{
            for chunk in raw.chunks_exact($width) {
                let bytes: [u8; $width] = chunk.try_into().unwrap();
                let value = match endian {
                    Endian::Little => <$ty>::from_le_bytes(bytes),
                    Endian::Big => <$ty>::from_be_bytes(bytes),
                };
                out.push(value as f32);
            }
        }};
    }

    match dtype {
        Dtype::F32 => {
            for chunk in raw.chunks_exact(4) {
                let bytes: [u8; 4] = chunk.try_into().unwrap();
                out.push(match endian {
                    Endian::Little => f32::from_le_bytes(bytes),
                    Endian::Big => f32::from_be_bytes(bytes),
                });
            }
        }
        Dtype::F64 => {
            for chunk in raw.chunks_exact(8) {
                let bytes: [u8; 8] = chunk.try_into().unwrap();
                out.push(match endian {
                    Endian::Little => f64::from_le_bytes(bytes),
                    Endian::Big => f64::from_be_bytes(bytes),
                } as f32);
            }
        }
        Dtype::F16 => {
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                out.push(half_to_f32(bits));
            }
        }
        Dtype::Bf16 => {
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                out.push(f32::from_bits((bits as u32) << 16));
            }
        }
        Dtype::U8 => read_ints!(u8, 1),
        Dtype::I8 => read_ints!(i8, 1),
        Dtype::I16 => read_ints!(i16, 2),
        Dtype::I32 => read_ints!(i32, 4),
        Dtype::I64 => read_ints!(i64, 8),
        Dtype::Bool => read_ints!(u8, 1),
    }
    Ok(out)
}

fn half_to_f32(bits: u16) -> f32 {
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

/// Copies a (possibly strided, possibly offset) view out of a storage.
fn gather(elements: &[f32], tref: &TensorRef, expected: usize) -> Result<Vec<f32>> {
    let shape = tref
        .size
        .as_ref()
        .ok_or_else(|| Error::Checkpoint("sparse tensor has no dense shape".into()))?;

    let contiguous = match &tref.stride {
        None => true,
        Some(stride) => {
            let mut acc = 1i64;
            let mut ok = true;
            for (dim, &st) in shape.iter().zip(stride.iter()).rev() {
                if st != acc {
                    ok = false;
                    break;
                }
                acc *= *dim as i64;
            }
            ok
        }
    };

    if contiguous {
        let end = tref.offset + expected;
        if end > elements.len() {
            return Err(Error::Checkpoint(format!(
                "tensor view [{}, {end}) runs past its storage of {} elements",
                tref.offset,
                elements.len()
            )));
        }
        return Ok(elements[tref.offset..end].to_vec());
    }

    let stride = tref.stride.as_ref().unwrap();
    let mut out = Vec::with_capacity(expected);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..expected {
        let mut src = tref.offset as i64;
        for (dim, &offset) in index.iter().enumerate() {
            src += offset as i64 * stride[dim];
        }
        if src < 0 || src as usize >= elements.len() {
            return Err(Error::Checkpoint(format!(
                "strided view reads element {src} outside a storage of {} elements",
                elements.len()
            )));
        }
        out.push(elements[src as usize]);

        // Row-major odometer over the view shape.
        for dim in (0..shape.len()).rev() {
            index[dim] += 1;
            if index[dim] < shape[dim] {
                break;
            }
            index[dim] = 0;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_views_are_copied_directly() {
        let elements: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tref = TensorRef {
            storage: pickle::StorageRef {
                dtype: Dtype::F32,
                key: "0".into(),
                location: "cpu".into(),
                numel: 16,
            },
            offset: 2,
            size: Some(vec![2, 3]),
            stride: Some(vec![3, 1]),
        };
        assert_eq!(gather(&elements, &tref, 6).unwrap(), vec![2., 3., 4., 5., 6., 7.]);
    }

    #[test]
    fn transposed_views_follow_their_stride() {
        // A 2x3 view of a 3x2 buffer: index (i, j) reads storage[i + 2 * j].
        let elements: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let tref = TensorRef {
            storage: pickle::StorageRef {
                dtype: Dtype::F32,
                key: "0".into(),
                location: "cpu".into(),
                numel: 8,
            },
            offset: 0,
            size: Some(vec![3, 2]),
            stride: Some(vec![1, 3]),
        };
        assert_eq!(
            gather(&elements, &tref, 6).unwrap(),
            vec![0., 3., 1., 4., 2., 5.]
        );
    }

    #[test]
    fn out_of_range_views_are_rejected() {
        let elements = vec![0.0f32; 4];
        let tref = TensorRef {
            storage: pickle::StorageRef {
                dtype: Dtype::F32,
                key: "0".into(),
                location: "cpu".into(),
                numel: 4,
            },
            offset: 3,
            size: Some(vec![4]),
            stride: Some(vec![1]),
        };
        assert!(gather(&elements, &tref, 4).is_err());
    }

    #[test]
    fn half_precision_decoding_matches_known_values() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0xbc00), -1.0);
        assert!((half_to_f32(0x3555) - 0.333_251_95).abs() < 1e-6);
        assert!(half_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn little_endian_f32_storage_round_trips() {
        let raw: Vec<u8> = [1.5f32, -2.25f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(decode_storage(&raw, Dtype::F32, Endian::Little).unwrap(), vec![1.5, -2.25]);
    }

    #[test]
    fn misaligned_storage_lengths_are_rejected() {
        assert!(decode_storage(&[0, 1, 2], Dtype::F32, Endian::Little).is_err());
    }
}
