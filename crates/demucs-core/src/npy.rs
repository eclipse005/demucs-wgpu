//! Minimal reader for NumPy `.npy` files (v1.0/2.0/3.0), enough to consume the
//! reference dumps produced by `tools/dump_reference.py`.

use std::path::Path;

use ndarray::{ArrayD, IxDyn};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDtype {
    F32,
    F64,
    I64,
    I32,
    U8,
    I8,
    Bool,
}

impl NpyDtype {
    fn parse(descr: &str) -> Result<Self> {
        // descr looks like "<f4", "|u1", "<i8", ...
        let body = descr.trim_start_matches(['<', '>', '=', '|']);
        Ok(match body {
            "f4" => NpyDtype::F32,
            "f8" => NpyDtype::F64,
            "i8" => NpyDtype::I64,
            "i4" => NpyDtype::I32,
            "u1" => NpyDtype::U8,
            "i1" => NpyDtype::I8,
            "b1" => NpyDtype::Bool,
            other => {
                return Err(Error::Shape(format!(
                    "unsupported .npy dtype descriptor `{other}`"
                )))
            }
        })
    }

    fn size(self) -> usize {
        match self {
            NpyDtype::F32 | NpyDtype::I32 => 4,
            NpyDtype::F64 | NpyDtype::I64 => 8,
            NpyDtype::U8 | NpyDtype::I8 | NpyDtype::Bool => 1,
        }
    }
}

/// A parsed `.npy` array, always presented as float32 for comparison purposes.
#[derive(Debug, Clone)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl NpyArray {
    pub fn into_array(self) -> ArrayD<f32> {
        ArrayD::from_shape_vec(IxDyn(&self.shape), self.data)
            .expect("shape and data length agree by construction")
    }

    pub fn to_array(&self) -> ArrayD<f32> {
        self.clone().into_array()
    }
}

pub fn read_npy(path: impl AsRef<Path>) -> Result<NpyArray> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    parse_npy(&bytes).map_err(|e| Error::Shape(format!("{}: {e}", path.display())))
}

fn parse_npy(bytes: &[u8]) -> Result<NpyArray> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(Error::Shape("not a .npy file (bad magic)".into()));
    }
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        other => {
            return Err(Error::Shape(format!(
                "unsupported .npy version {other}.{0}",
                bytes[7]
            )))
        }
    };
    let header_end = header_start + header_len;
    if header_end > bytes.len() {
        return Err(Error::Shape("truncated .npy header".into()));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|e| Error::Shape(format!("invalid .npy header: {e}")))?;

    let descr = find_quoted(header, "'descr'")
        .or_else(|| find_quoted(header, "\"descr\""))
        .ok_or_else(|| Error::Shape("npy header has no descr".into()))?;
    let fortran = header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True");
    let shape_text = extract_shape(header)?;
    let mut shape = Vec::new();
    if !shape_text.trim().is_empty() {
        for part in shape_text.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            shape.push(
                part.parse::<usize>()
                    .map_err(|e| Error::Shape(format!("bad npy shape `{shape_text}`: {e}")))?,
            );
        }
    }

    let dtype = NpyDtype::parse(&descr)?;
    let count: usize = shape.iter().product();
    let payload = &bytes[header_end..];
    let needed = count * dtype.size();
    if payload.len() < needed {
        return Err(Error::Shape(format!(
            "npy payload holds {} bytes, need {needed}",
            payload.len()
        )));
    }

    let mut data = Vec::with_capacity(count);
    match dtype {
        NpyDtype::F32 => {
            for chunk in payload[..needed].chunks_exact(4) {
                data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
        }
        NpyDtype::F64 => {
            for chunk in payload[..needed].chunks_exact(8) {
                data.push(f64::from_le_bytes(chunk.try_into().unwrap()) as f32);
            }
        }
        NpyDtype::I64 => {
            for chunk in payload[..needed].chunks_exact(8) {
                data.push(i64::from_le_bytes(chunk.try_into().unwrap()) as f32);
            }
        }
        NpyDtype::I32 => {
            for chunk in payload[..needed].chunks_exact(4) {
                data.push(i32::from_le_bytes(chunk.try_into().unwrap()) as f32);
            }
        }
        NpyDtype::U8 | NpyDtype::Bool => {
            data.extend(payload[..needed].iter().map(|b| *b as f32));
        }
        NpyDtype::I8 => {
            data.extend(payload[..needed].iter().map(|b| *b as i8 as f32));
        }
    }

    // A Fortran-ordered file stores the axes in the opposite order, which
    // NumPy picks automatically for the non-contiguous tensors PyTorch hooks
    // hand out (a transpose in the model). Present everything as C order.
    if fortran && shape.len() > 1 {
        data = fortran_to_c(data, &shape);
    }

    Ok(NpyArray { shape, data })
}

fn find_quoted(header: &str, key: &str) -> Option<String> {
    let start = header.find(key)? + key.len();
    let rest = &header[start..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    let quote = rest.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let rest = &rest[1..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

fn extract_shape(header: &str) -> Result<String> {
    let start = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| Error::Shape("npy header has no shape".into()))?;
    let rest = &header[start..];
    let open = rest
        .find('(')
        .ok_or_else(|| Error::Shape("npy shape is not a tuple".into()))?;
    let close = rest
        .find(')')
        .ok_or_else(|| Error::Shape("npy shape is not a tuple".into()))?;
    Ok(rest[open + 1..close].to_string())
}

/// Reorders an array stored with the first axis varying fastest into the
/// row-major order the rest of the crate assumes.
fn fortran_to_c(data: Vec<f32>, shape: &[usize]) -> Vec<f32> {
    let rank = shape.len();
    let count: usize = shape.iter().product();
    if count == 0 {
        return data;
    }
    let mut c_stride = vec![1usize; rank];
    for dim in (0..rank - 1).rev() {
        c_stride[dim] = c_stride[dim + 1] * shape[dim + 1];
    }
    let mut f_stride = vec![1usize; rank];
    for dim in 1..rank {
        f_stride[dim] = f_stride[dim - 1] * shape[dim - 1];
    }

    let mut out = vec![0.0f32; count];
    let mut index = vec![0usize; rank];
    for slot in out.iter_mut() {
        let source: usize = (0..rank).map(|dim| index[dim] * f_stride[dim]).sum();
        *slot = data[source];
        for dim in (0..rank).rev() {
            index[dim] += 1;
            if index[dim] < shape[dim] {
                break;
            }
            index[dim] = 0;
        }
    }
    let _ = c_stride;
    out
}

/// Writes a C-ordered float32 array in `.npy` v1.0 format.
///
/// The alignment rule (header padded so the payload starts on a 64-byte
/// boundary) is what NumPy itself uses.
pub fn write_npy(path: impl AsRef<Path>, shape: &[usize], data: &[f32]) -> Result<()> {
    let count: usize = shape.iter().product();
    if count != data.len() {
        return Err(Error::Shape(format!(
            "npy writer got {count} slots from the shape but {} values",
            data.len()
        )));
    }
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}",
        shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0x93]);
    bytes.extend_from_slice(b"NUMPY");
    bytes.push(1); // major version
    bytes.push(0); // minor version
    let prefix = bytes.len() + 2; // the u16 header length that follows
    let padding = (64 - (prefix + header.len() + 1) % 64) % 64;
    let header_len = header.len() + padding + 1;
    bytes.extend_from_slice(&(header_len as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend(std::iter::repeat(b' ').take(padding));
    bytes.push(10u8); // newline
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path.as_ref(), bytes)
        .map_err(|e| Error::Shape(format!("{}: {e}", path.as_ref().display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_written_array_reads_back_identically() {
        let dir = std::env::temp_dir().join("demucs_npy_writer_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("probe.npy");
        let values: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
        write_npy(&path, &[2, 3, 4], &values).unwrap();
        let back = read_npy(&path).unwrap();
        assert_eq!(back.shape, vec![2, 3, 4]);
        assert_eq!(back.data, values);
        // The payload must start on a 64-byte boundary, like NumPy's own writer.
        let bytes = std::fs::read(&path).unwrap();
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
    }

    fn build(descr: &str, shape: &str, payload: &[u8]) -> Vec<u8> {
        let header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape}), }}");
        let mut padded = header.clone();
        while (10 + padded.len() + 1) % 64 != 0 {
            padded.push(' ');
        }
        padded.push('\n');
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&(padded.len() as u16).to_le_bytes());
        out.extend_from_slice(padded.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn reads_a_1d_float32_array() {
        let payload: Vec<u8> = [1.0f32, -2.5, 3.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = build("<f4", "3,", &payload);
        let array = parse_npy(&bytes).unwrap();
        assert_eq!(array.shape, vec![3]);
        assert_eq!(array.data, vec![1.0, -2.5, 3.25]);
    }

    #[test]
    fn reads_a_2d_int64_array() {
        let payload: Vec<u8> = [0i64, 1, 2, 3]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = build("<i8", "2, 2", &payload);
        let array = parse_npy(&bytes).unwrap();
        assert_eq!(array.shape, vec![2, 2]);
        assert_eq!(array.data, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn rejects_foreign_magic() {
        assert!(parse_npy(b"not an npy file at all").is_err());
    }

    #[test]
    fn rejects_short_payloads() {
        let bytes = build("<f4", "4,", &[0u8; 8]);
        assert!(parse_npy(&bytes).is_err());
    }
}
