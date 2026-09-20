//! Minimal Python-pickle interpreter, sufficient for PyTorch `torch.save` files.
//!
//! PyTorch checkpoints are a zip archive holding a protocol-2 (or later) pickle
//! stream plus one raw binary blob per storage. The pickle only ever uses a
//! handful of reduction functions, so instead of pulling in a full pickle
//! library we interpret the opcodes we actually meet and model the reductions
//! Torch emits.

use std::collections::HashMap;

use crate::error::{Error, Result};

/// Storage element types Torch can serialise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    F64,
    U8,
    I8,
    I16,
    I32,
    I64,
    Bool,
}

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::Bf16 | Dtype::I16 => 2,
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::U8 | Dtype::I8 | Dtype::Bool => 1,
        }
    }

    /// Maps a Torch storage class or dtype spelling onto our enum.
    pub fn from_torch_name(name: &str) -> Option<Dtype> {
        let lower = name.to_ascii_lowercase();
        let lower = lower.strip_prefix("torch.").unwrap_or(&lower);
        Some(match lower {
            "floatstorage" | "float" | "float32" | "f32" => Dtype::F32,
            "halfstorage" | "half" | "float16" | "f16" => Dtype::F16,
            "bfloat16storage" | "bfloat16" | "bf16" => Dtype::Bf16,
            "doublestorage" | "double" | "float64" | "f64" => Dtype::F64,
            "bytestorage" | "uint8" | "u8" => Dtype::U8,
            "charstorage" | "int8" | "i8" => Dtype::I8,
            "shortstorage" | "int16" | "i16" => Dtype::I16,
            "intstorage" | "int32" | "i32" => Dtype::I32,
            "longstorage" | "int64" | "i64" => Dtype::I64,
            "boolstorage" | "bool" => Dtype::Bool,
            _ => return None,
        })
    }
}

/// A reference to one serialised storage blob (`<archive>/data/<key>`).
#[derive(Debug, Clone, PartialEq)]
pub struct StorageRef {
    pub dtype: Dtype,
    pub key: String,
    pub location: String,
    pub numel: usize,
}

/// A tensor as described by `torch._utils._rebuild_tensor_v2`.
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub storage: StorageRef,
    pub offset: usize,
    /// `None` marks a sparse tensor, which this port does not support.
    pub size: Option<Vec<usize>>,
    pub stride: Option<Vec<i64>>,
}

/// Values the interpreter manipulates. Only the variants Torch actually emits.
#[derive(Debug, Clone)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    /// `module.name` from a GLOBAL/STACK_GLOBAL opcode.
    Global(String, String),
    Storage(StorageRef),
    Tensor(TensorRef),
    /// A reduction we do not model; kept so unrelated metadata cannot abort a load.
    Opaque(String),
}

impl Value {
    pub fn as_str(&self) -> Result<&str> {
        match self {
            Value::Str(s) => Ok(s),
            Value::Bytes(b) => std::str::from_utf8(b)
                .map_err(|e| Error::Pickle(format!("invalid utf-8 in pickle string: {e}"))),
            other => Err(Error::Pickle(format!("expected string, found {other:?}"))),
        }
    }

    pub fn as_int(&self) -> Result<i64> {
        match self {
            Value::Int(i) => Ok(*i),
            Value::Bool(b) => Ok(*b as i64),
            other => Err(Error::Pickle(format!("expected int, found {other:?}"))),
        }
    }

    pub fn as_tuple(&self) -> Result<&[Value]> {
        match self {
            Value::Tuple(items) | Value::List(items) => Ok(items),
            other => Err(Error::Pickle(format!("expected tuple, found {other:?}"))),
        }
    }
}

pub struct Unpickler<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: HashMap<u64, Value>,
}

impl<'a> Unpickler<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: HashMap::new(),
        }
    }

    pub fn run(mut self) -> Result<Value> {
        loop {
            let op = self.read_u8()?;
            match op {
                b'\x80' => {
                    self.read_u8()?;
                } // PROTO
                b'.' => break, // STOP
                b'\x95' => {
                    // FRAME: 8-byte length, the enclosed bytes are read normally.
                    self.read_exact(8)?;
                }
                b'(' => self.marks.push(self.stack.len()), // MARK
                b')' => self.stack.push(Value::Tuple(Vec::new())), // EMPTY_TUPLE
                b']' => self.stack.push(Value::List(Vec::new())), // EMPTY_LIST
                b'}' => self.stack.push(Value::Dict(Vec::new())), // EMPTY_DICT
                b'N' => self.stack.push(Value::None), // NONE
                b'\x88' => self.stack.push(Value::Bool(true)), // NEWTRUE
                b'\x89' => self.stack.push(Value::Bool(false)), // NEWFALSE
                b'K' => {
                    let v = self.read_u8()? as i64;
                    self.stack.push(Value::Int(v));
                } // BININT1
                b'M' => {
                    let v = self.read_u16()? as i64;
                    self.stack.push(Value::Int(v));
                } // BININT2
                b'J' => {
                    let v = self.read_i32()? as i64;
                    self.stack.push(Value::Int(v));
                } // BININT
                b'I' => {
                    // INT: decimal text terminated by '\n', with an optional
                    // trailing 'L' on ancient pickles.
                    let line = self.read_line()?;
                    let text = line.trim_end_matches('L');
                    let text = text.trim();
                    if text == "01" {
                        self.stack.push(Value::Bool(true));
                    } else if text == "00" {
                        self.stack.push(Value::Bool(false));
                    } else {
                        let v: i64 = text
                            .parse()
                            .map_err(|e| Error::Pickle(format!("bad INT `{text}`: {e}")))?;
                        self.stack.push(Value::Int(v));
                    }
                }
                b'\x8a' => {
                    // LONG1: little-endian two's complement of `n` bytes.
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_exact(n)?;
                    self.stack.push(Value::Int(decode_long(bytes)?));
                }
                b'\x8b' => {
                    // LONG4
                    let n = self.read_u32()? as usize;
                    let bytes = self.read_exact(n)?;
                    self.stack.push(Value::Int(decode_long(bytes)?));
                }
                b'F' => {
                    let line = self.read_line()?;
                    let v: f64 = line
                        .trim()
                        .parse()
                        .map_err(|e| Error::Pickle(format!("bad FLOAT: {e}")))?;
                    self.stack.push(Value::Float(v));
                }
                b'G' => {
                    let v = self.read_f64()?;
                    self.stack.push(Value::Float(v));
                } // BINFLOAT
                b'X' => {
                    let n = self.read_u32()? as usize;
                    let bytes = self.read_exact(n)?;
                    self.stack.push(Value::Str(decode_utf8(bytes)?));
                } // BINUNICODE
                b'\x8c' => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_exact(n)?;
                    self.stack.push(Value::Str(decode_utf8(bytes)?));
                } // SHORT_BINUNICODE
                b'\x8d' => {
                    let n = self.read_u64()? as usize;
                    let bytes = self.read_exact(n)?;
                    self.stack.push(Value::Str(decode_utf8(bytes)?));
                } // BINUNICODE8
                b'C' => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_exact(n)?.to_vec();
                    self.stack.push(Value::Bytes(bytes));
                } // SHORT_BINBYTES
                b'B' => {
                    let n = self.read_u32()? as usize;
                    let bytes = self.read_exact(n)?.to_vec();
                    self.stack.push(Value::Bytes(bytes));
                } // BINBYTES
                b'\x8e' => {
                    let n = self.read_u64()? as usize;
                    let bytes = self.read_exact(n)?.to_vec();
                    self.stack.push(Value::Bytes(bytes));
                } // BINBYTES8
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items));
                } // TUPLE
                b'\x85' => {
                    // TUPLE1
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a]));
                }
                b'\x86' => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b]));
                } // TUPLE2
                b'\x87' => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b, c]));
                } // TUPLE3
                b'l' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::List(items));
                } // LIST
                b'a' => {
                    // APPEND
                    let item = self.pop()?;
                    match self.stack.last_mut() {
                        Some(Value::List(items)) => items.push(item),
                        other => {
                            return Err(Error::Pickle(format!("APPEND to non-list {other:?}")))
                        }
                    }
                }
                b'e' => {
                    // APPENDS
                    let items = self.pop_mark()?;
                    match self.stack.last_mut() {
                        Some(Value::List(target)) => target.extend(items),
                        other => {
                            return Err(Error::Pickle(format!("APPENDS to non-list {other:?}")))
                        }
                    }
                }
                b'd' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Dict(pairs(items)?));
                } // DICT
                b's' => {
                    // SETITEM
                    let value = self.pop()?;
                    let key = self.pop()?;
                    match self.stack.last_mut() {
                        Some(Value::Dict(map)) => map.push((key, value)),
                        other => {
                            return Err(Error::Pickle(format!("SETITEM on non-dict {other:?}")))
                        }
                    }
                }
                b'u' => {
                    // SETITEMS
                    let items = self.pop_mark()?;
                    let additions = pairs(items)?;
                    match self.stack.last_mut() {
                        Some(Value::Dict(map)) => map.extend(additions),
                        other => {
                            return Err(Error::Pickle(format!("SETITEMS on non-dict {other:?}")))
                        }
                    }
                }
                b'c' => {
                    // GLOBAL: module, then name, each '\n' terminated.
                    let module = self.read_line()?;
                    let name = self.read_line()?;
                    self.stack.push(Value::Global(module.trim().to_string(), name.trim().to_string()));
                }
                b'\x93' => {
                    // STACK_GLOBAL
                    let name = self.pop()?;
                    let module = self.pop()?;
                    self.stack.push(Value::Global(
                        module.as_str()?.to_string(),
                        name.as_str()?.to_string(),
                    ));
                }
                b'q' => {
                    let idx = self.read_u8()? as u64;
                    self.memoize(idx)?;
                } // BINPUT
                b'r' => {
                    let idx = self.read_u32()? as u64;
                    self.memoize(idx)?;
                } // LONG_BINPUT
                b'\x94' => {
                    let idx = self.memo.len() as u64;
                    self.memoize(idx)?;
                } // MEMOIZE
                b'p' => {
                    let line = self.read_line()?;
                    let idx: u64 = line
                        .trim()
                        .parse()
                        .map_err(|e| Error::Pickle(format!("bad PUT index: {e}")))?;
                    self.memoize(idx)?;
                }
                b'h' => {
                    let idx = self.read_u8()? as u64;
                    self.stack.push(self.recall(idx)?);
                } // BINGET
                b'j' => {
                    let idx = self.read_u32()? as u64;
                    self.stack.push(self.recall(idx)?);
                } // LONG_BINGET
                b'g' => {
                    let line = self.read_line()?;
                    let idx: u64 = line
                        .trim()
                        .parse()
                        .map_err(|e| Error::Pickle(format!("bad GET index: {e}")))?;
                    self.stack.push(self.recall(idx)?);
                }
                b'Q' => {
                    // BINPERSID
                    let pid = self.pop()?;
                    self.stack.push(self.persistent_load(&pid)?);
                }
                b'P' => {
                    let line = self.read_line()?;
                    self.stack.push(Value::Opaque(format!("PERSID({})", line.trim())));
                }
                b'R' => {
                    // REDUCE
                    let args = self.pop()?;
                    let func = self.pop()?;
                    self.stack.push(self.reduce(func, args)?);
                }
                b'\x81' => {
                    // NEWOBJ: cls(*args)
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    self.stack.push(self.reduce(cls, args)?);
                }
                b'\x92' => {
                    // NEWOBJ_EX: cls(*args, **kwargs)
                    let _kwargs = self.pop()?;
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    self.stack.push(self.reduce(cls, args)?);
                }
                b'b' => {
                    // BUILD: apply a __setstate__/state pair. Torch only uses it for
                    // empty OrderedDict states, which are a no-op for us.
                    let _state = self.pop()?;
                }
                b'0' => {
                    // POP
                    self.pop()?;
                }
                other => {
                    return Err(Error::Pickle(format!(
                        "unsupported pickle opcode 0x{other:02x} at offset {}",
                        self.pos - 1
                    )))
                }
            }
        }

        self.stack
            .pop()
            .ok_or_else(|| Error::Pickle("pickle stream produced no value".into()))
    }

    fn reduce(&self, func: Value, args: Value) -> Result<Value> {
        let Value::Global(module, name) = &func else {
            return Err(Error::Pickle(format!(
                "REDUCE with non-global callable {func:?}"
            )));
        };
        let items = args.as_tuple()?;

        match (module.as_str(), name.as_str()) {
            ("collections", "OrderedDict") => {
                // OrderedDict() with no args, or OrderedDict(iterable).
                if items.is_empty() {
                    return Ok(Value::Dict(Vec::new()));
                }
                match &items[0] {
                    Value::Dict(map) => Ok(Value::Dict(map.clone())),
                    Value::List(entries) => Ok(Value::Dict(pairs(entries.clone())?)),
                    Value::Tuple(entries) => Ok(Value::Dict(pairs(entries.to_vec())?)),
                    other => Err(Error::Pickle(format!(
                        "OrderedDict from unsupported {other:?}"
                    ))),
                }
            }
            ("torch._utils", "_rebuild_tensor_v2")
            | ("torch._utils", "_rebuild_tensor")
            | ("torch._utils", "_rebuild_tensor_v3") => rebuild_tensor(items),
            ("torch._utils", "_rebuild_parameter") | ("torch._utils", "_rebuild_parameter_with_state") => {
                // The parameter wrapper only carries autograd metadata; the value
                // lives in the first argument.
                items
                    .first()
                    .cloned()
                    .ok_or_else(|| Error::Pickle("_rebuild_parameter without args".into()))
            }
            _ => Ok(Value::Opaque(format!("{module}.{name}"))),
        }
    }

    fn persistent_load(&self, pid: &Value) -> Result<Value> {
        let items = pid.as_tuple()?;
        if items.len() < 5 || items[0].as_str().ok() != Some("storage") {
            return Err(Error::Pickle(format!(
                "unsupported persistent id {pid:?}"
            )));
        }
        // Classic form: ('storage', <storage class>, key, location, numel).
        // Newer form:  ('storage', <dtype name>,    key, location, numel).
        let dtype = match &items[1] {
            Value::Global(_, class_name) => Dtype::from_torch_name(class_name),
            other => other.as_str().ok().and_then(Dtype::from_torch_name),
        }
        .ok_or_else(|| Error::Pickle(format!("unknown storage class in {pid:?}")))?;

        let numel = usize::try_from(items[4].as_int()?)
            .map_err(|_| Error::Pickle(format!("negative storage size in {pid:?}")))?;

        Ok(Value::Storage(StorageRef {
            dtype,
            key: items[2].as_str()?.to_string(),
            location: items[3].as_str().unwrap_or("cpu").to_string(),
            numel,
        }))
    }

    fn memoize(&mut self, index: u64) -> Result<()> {
        let value = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| Error::Pickle("memoize on empty stack".into()))?;
        self.memo.insert(index, value);
        Ok(())
    }

    fn recall(&self, index: u64) -> Result<Value> {
        self.memo
            .get(&index)
            .cloned()
            .ok_or_else(|| Error::Pickle(format!("pickle memo has no entry {index}")))
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack
            .pop()
            .ok_or_else(|| Error::Pickle("pickle stack underflow".into()))
    }

    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let start = self
            .marks
            .pop()
            .ok_or_else(|| Error::Pickle("MARK stack underflow".into()))?;
        Ok(self.stack.split_off(start))
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_exact(2)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_be_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_exact(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos + n;
        if end > self.data.len() {
            return Err(Error::Pickle(format!(
                "truncated pickle stream: need {n} bytes at offset {}",
                self.pos
            )));
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_line(&mut self) -> Result<String> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(Error::Pickle("unterminated line in pickle stream".into()));
        }
        let line = decode_utf8(&self.data[start..self.pos])?;
        self.pos += 1;
        Ok(line)
    }
}

fn rebuild_tensor(items: &[Value]) -> Result<Value> {
    if items.len() < 4 {
        return Err(Error::Pickle("_rebuild_tensor with too few arguments".into()));
    }
    let storage = match &items[0] {
        Value::Storage(s) => s.clone(),
        other => return Err(Error::Pickle(format!("expected storage, found {other:?}"))),
    };
    let offset = usize::try_from(items[1].as_int()?)
        .map_err(|_| Error::Pickle("negative storage offset".into()))?;

    // Sparse tensors carry (indices, values, size, ...) instead of size/stride.
    let (size, stride) = match (&items[2], &items[3]) {
        (Value::Tuple(size), Value::Tuple(stride)) => (
            Some(
                size.iter()
                    .map(|v| v.as_int().map(|i| i as usize))
                    .collect::<Result<Vec<_>>>()?,
            ),
            Some(
                stride.iter()
                    .map(|v| v.as_int())
                    .collect::<Result<Vec<_>>>()?,
            ),
        ),
        _ => (None, None),
    };

    Ok(Value::Tensor(TensorRef {
        storage,
        offset,
        size,
        stride,
    }))
}

fn pairs(items: Vec<Value>) -> Result<Vec<(Value, Value)>> {
    if items.len() % 2 != 0 {
        return Err(Error::Pickle("odd number of items in a dict".into()));
    }
    let mut out = Vec::with_capacity(items.len() / 2);
    let mut iter = items.into_iter();
    while let (Some(k), Some(v)) = (iter.next(), iter.next()) {
        out.push((k, v));
    }
    Ok(out)
}

fn decode_utf8(bytes: &[u8]) -> Result<String> {
    String::from_utf8(bytes.to_vec()).map_err(|e| Error::Pickle(format!("invalid utf-8: {e}")))
}

/// LONG opcodes store an arbitrary-precision integer as little-endian
/// two's-complement bytes, so short literals need sign extension.
fn decode_long(bytes: &[u8]) -> Result<i64> {
    if bytes.is_empty() {
        return Ok(0);
    }
    if bytes.len() > 8 {
        return Err(Error::Pickle(format!(
            "integer literal of {} bytes exceeds i64",
            bytes.len()
        )));
    }
    let negative = bytes[bytes.len() - 1] & 0x80 != 0;
    let mut buf = [if negative { 0xffu8 } else { 0x00u8 }; 8];
    buf[..bytes.len()].copy_from_slice(bytes);
    Ok(i64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_long_literals() {
        assert_eq!(decode_long(&[0xff]).unwrap(), -1);
        assert_eq!(decode_long(&[0x80, 0x01]).unwrap(), 384);
        assert_eq!(decode_long(&[]).unwrap(), 0);
        assert_eq!(decode_long(&[0x7f]).unwrap(), 127);
        assert_eq!(decode_long(&[0x00, 0x80]).unwrap(), -32_768);
        assert!(decode_long(&[0; 9]).is_err());
    }

    #[test]
    fn maps_torch_storage_classes() {
        assert_eq!(Dtype::from_torch_name("FloatStorage"), Some(Dtype::F32));
        assert_eq!(Dtype::from_torch_name("HalfStorage"), Some(Dtype::F16));
        assert_eq!(Dtype::from_torch_name("torch.float32"), Some(Dtype::F32));
        assert_eq!(Dtype::from_torch_name("nonsense"), None);
    }

    #[test]
    fn interprets_an_ordered_dict_of_ints() {
        // {"a": 1, "b": 2} built by OrderedDict(), then SETITEMS.
        let mut data = vec![0x80, 2];
        data.push(b'c');
        data.extend_from_slice(b"collections\nOrderedDict\n");
        data.push(b')');
        data.push(b'R');
        data.push(b'}');
        data.push(b'(');
        data.push(b'X');
        data.extend_from_slice(&1u32.to_le_bytes());
        data.push(b'a');
        data.push(b'K');
        data.push(1);
        data.push(b'X');
        data.extend_from_slice(&1u32.to_le_bytes());
        data.push(b'b');
        data.push(b'K');
        data.push(2);
        data.push(b'u');
        data.push(b'.');

        let value = Unpickler::new(&data).run().unwrap();
        let Value::Dict(map) = value else {
            panic!("expected dict");
        };
        assert_eq!(map.len(), 2);
        assert_eq!(map[0].1.as_int().unwrap(), 1);
        assert_eq!(map[1].1.as_int().unwrap(), 2);
    }

    #[test]
    fn unknown_reductions_become_opaque_instead_of_failing() {
        let mut data = vec![0x80, 2];
        data.push(b'c');
        data.extend_from_slice(b"numpy.core\nscalar\n");
        data.push(b')');
        data.push(b'R');
        data.push(b'.');
        let value = Unpickler::new(&data).run().unwrap();
        assert!(matches!(value, Value::Opaque(_)));
    }
}
