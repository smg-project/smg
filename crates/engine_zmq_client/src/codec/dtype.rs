// Adapted from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/dtype.rs and protocol/logprobs/array.rs.

use std::io::Cursor;

use byteorder::{BigEndian, LittleEndian, NativeEndian, ReadBytesExt};
use half::{bf16, f16};

use crate::error::{Error, Result};

/// Effective model dtype reported by the engine after config resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModelDtype {
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "bfloat16")]
    BFloat16,
    #[serde(rename = "float32")]
    Float32,
}

impl ModelDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
        }
    }
}

/// The scalar element type of a wire ndarray/tensor, parsed from its dtype
/// string (numpy `dtype.str` for ndarrays, `torch.<dtype>` names for tensors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl ScalarType {
    /// Size in bytes of one element.
    pub fn element_size(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
}

/// Byte order encoded by the numpy dtype prefix (`<`, `>`, `=`, `|`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    Little,
    Big,
    Native,
}

/// Parse a numpy dtype string (e.g. `"<i4"`, `"float32"`) into a scalar type
/// and byte order.
pub fn parse_dtype(dtype: &str, field: &str) -> Result<(ScalarType, Endianness)> {
    let (endianness, body) = match dtype.as_bytes().first().copied() {
        Some(b'<') => (Endianness::Little, &dtype[1..]),
        Some(b'>') => (Endianness::Big, &dtype[1..]),
        Some(b'=') | Some(b'|') => (Endianness::Native, &dtype[1..]),
        _ => (Endianness::Native, dtype),
    };

    let scalar = match body {
        "i4" | "int32" => ScalarType::I32,
        "i8" | "int64" => ScalarType::I64,
        "f2" | "float16" => ScalarType::F16,
        // numpy has no bfloat16; only torch tensors carry this name.
        "bfloat16" => ScalarType::BF16,
        "f4" | "float32" => ScalarType::F32,
        "f8" | "float64" => ScalarType::F64,
        _ => {
            return Err(decode_error(
                field,
                &format!("unsupported dtype string {dtype:?}"),
            ))
        }
    };
    Ok((scalar, endianness))
}

/// Decode a little/big/native-endian buffer into a `Vec<i32>`.
pub fn decode_i32_vec(bytes: &[u8], endianness: Endianness, field: &str) -> Result<Vec<i32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(decode_error(
            field,
            &format!("byte length {} is not divisible by 4", bytes.len()),
        ));
    }
    let mut cursor = Cursor::new(bytes);
    let mut values = Vec::with_capacity(bytes.len() / 4);
    while (cursor.position() as usize) < bytes.len() {
        let value = match endianness {
            Endianness::Little => cursor.read_i32::<LittleEndian>(),
            Endianness::Big => cursor.read_i32::<BigEndian>(),
            Endianness::Native => cursor.read_i32::<NativeEndian>(),
        }
        .map_err(|error| decode_error(field, &format!("failed to read i32 payload: {error}")))?;
        values.push(value);
    }
    Ok(values)
}

/// Decode a little/big/native-endian buffer into a `Vec<i64>`.
pub fn decode_i64_vec(bytes: &[u8], endianness: Endianness, field: &str) -> Result<Vec<i64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(decode_error(
            field,
            &format!("byte length {} is not divisible by 8", bytes.len()),
        ));
    }
    let mut cursor = Cursor::new(bytes);
    let mut values = Vec::with_capacity(bytes.len() / 8);
    while (cursor.position() as usize) < bytes.len() {
        let value = match endianness {
            Endianness::Little => cursor.read_i64::<LittleEndian>(),
            Endianness::Big => cursor.read_i64::<BigEndian>(),
            Endianness::Native => cursor.read_i64::<NativeEndian>(),
        }
        .map_err(|error| decode_error(field, &format!("failed to read i64 payload: {error}")))?;
        values.push(value);
    }
    Ok(values)
}

/// Decode a little/big/native-endian buffer into a `Vec<f32>`.
pub fn decode_f32_vec(bytes: &[u8], endianness: Endianness, field: &str) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(decode_error(
            field,
            &format!("byte length {} is not divisible by 4", bytes.len()),
        ));
    }
    let mut cursor = Cursor::new(bytes);
    let mut values = Vec::with_capacity(bytes.len() / 4);
    while (cursor.position() as usize) < bytes.len() {
        let value = match endianness {
            Endianness::Little => cursor.read_f32::<LittleEndian>(),
            Endianness::Big => cursor.read_f32::<BigEndian>(),
            Endianness::Native => cursor.read_f32::<NativeEndian>(),
        }
        .map_err(|error| decode_error(field, &format!("failed to read f32 payload: {error}")))?;
        values.push(value);
    }
    Ok(values)
}

/// Decode a buffer of floating-point `scalar` elements into `f32` values:
/// half precisions widen exactly, float64 narrows. Integer types are an error.
pub fn decode_float_vec(
    bytes: &[u8],
    scalar: ScalarType,
    endianness: Endianness,
    field: &str,
) -> Result<Vec<f32>> {
    let convert: fn(u64) -> f32 = match scalar {
        ScalarType::F16 => |bits| f16::from_bits(bits as u16).to_f32(),
        ScalarType::BF16 => |bits| bf16::from_bits(bits as u16).to_f32(),
        ScalarType::F32 => |bits| f32::from_bits(bits as u32),
        ScalarType::F64 => |bits| f64::from_bits(bits) as f32,
        ScalarType::I32 | ScalarType::I64 => {
            return Err(decode_error(
                field,
                &format!("expected a floating-point dtype, got {scalar:?}"),
            ))
        }
    };
    let size = scalar.element_size();
    if !bytes.len().is_multiple_of(size) {
        return Err(decode_error(
            field,
            &format!("byte length {} is not divisible by {size}", bytes.len()),
        ));
    }
    let mut cursor = Cursor::new(bytes);
    let mut values = Vec::with_capacity(bytes.len() / size);
    while (cursor.position() as usize) < bytes.len() {
        let bits = match endianness {
            Endianness::Little => cursor.read_uint::<LittleEndian>(size),
            Endianness::Big => cursor.read_uint::<BigEndian>(size),
            Endianness::Native => cursor.read_uint::<NativeEndian>(size),
        }
        .map_err(|error| {
            decode_error(
                field,
                &format!("failed to read {scalar:?} payload: {error}"),
            )
        })?;
        values.push(convert(bits));
    }
    Ok(values)
}

/// Convert a signed token id / rank into `u32`, rejecting negatives and overflow.
pub fn convert_to_u32<I>(value: I, field: &str) -> Result<u32>
where
    I: TryInto<u32> + std::fmt::Display + Copy,
{
    value.try_into().map_err(|_| {
        decode_error(
            field,
            &format!("expected non-negative value that fits in u32, got {value}"),
        )
    })
}

/// Build an [`Error::ExtValueDecode`] scoped to a field name.
pub fn decode_error(field: &str, reason: &str) -> Error {
    Error::ExtValueDecode {
        message: format!("{field}: {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_uses_protocol_dtype_strings() {
        assert_eq!(
            serde_json::to_value(ModelDtype::Float16).unwrap(),
            serde_json::json!("float16")
        );
        assert_eq!(
            serde_json::from_value::<ModelDtype>(serde_json::json!("bfloat16")).unwrap(),
            ModelDtype::BFloat16
        );
        assert_eq!(ModelDtype::Float32.as_str(), "float32");
    }

    #[test]
    fn parse_dtype_handles_endianness_and_aliases() {
        assert_eq!(
            parse_dtype("<i4", "f").unwrap(),
            (ScalarType::I32, Endianness::Little)
        );
        assert_eq!(
            parse_dtype(">i8", "f").unwrap(),
            (ScalarType::I64, Endianness::Big)
        );
        assert_eq!(
            parse_dtype("=f4", "f").unwrap(),
            (ScalarType::F32, Endianness::Native)
        );
        assert_eq!(
            parse_dtype("float32", "f").unwrap(),
            (ScalarType::F32, Endianness::Native)
        );
        assert!(parse_dtype("<c8", "f").is_err());
        // torch dtype names, as `MsgpackEncoder._encode_tensor` writes them.
        assert_eq!(
            parse_dtype("bfloat16", "f").unwrap(),
            (ScalarType::BF16, Endianness::Native)
        );
        assert_eq!(
            parse_dtype("float16", "f").unwrap(),
            (ScalarType::F16, Endianness::Native)
        );
        assert_eq!(
            parse_dtype("<f8", "f").unwrap(),
            (ScalarType::F64, Endianness::Little)
        );
    }

    #[test]
    fn decode_float_vec_widens_half_precisions_and_narrows_doubles() {
        let f16_bytes: Vec<u8> = [0.25_f32, -1.5, 3.0]
            .into_iter()
            .flat_map(|v| f16::from_f32(v).to_le_bytes())
            .collect();
        assert_eq!(
            decode_float_vec(&f16_bytes, ScalarType::F16, Endianness::Little, "f").unwrap(),
            vec![0.25, -1.5, 3.0]
        );
        let bf16_bytes: Vec<u8> = [0.25_f32, -1.5, 3.0]
            .into_iter()
            .flat_map(|v| bf16::from_f32(v).to_be_bytes())
            .collect();
        assert_eq!(
            decode_float_vec(&bf16_bytes, ScalarType::BF16, Endianness::Big, "f").unwrap(),
            vec![0.25, -1.5, 3.0]
        );
        let f64_bytes: Vec<u8> = [0.25_f64, -1.5]
            .into_iter()
            .flat_map(f64::to_ne_bytes)
            .collect();
        assert_eq!(
            decode_float_vec(&f64_bytes, ScalarType::F64, Endianness::Native, "f").unwrap(),
            vec![0.25, -1.5]
        );
        assert_eq!(
            decode_float_vec(
                &2.5_f32.to_le_bytes(),
                ScalarType::F32,
                Endianness::Little,
                "f"
            )
            .unwrap(),
            vec![2.5]
        );
        // Wrong element count and integer dtypes are errors, not panics.
        assert!(decode_float_vec(&[0, 1, 2], ScalarType::F16, Endianness::Little, "f").is_err());
        assert!(decode_float_vec(&[0, 1, 2, 3], ScalarType::I32, Endianness::Little, "f").is_err());
    }

    #[test]
    fn decoders_respect_endianness() {
        let le = 1_i32.to_le_bytes();
        assert_eq!(
            decode_i32_vec(&le, Endianness::Little, "f").unwrap(),
            vec![1]
        );
        let be = 1_i32.to_be_bytes();
        assert_eq!(decode_i32_vec(&be, Endianness::Big, "f").unwrap(), vec![1]);
        assert!(decode_i32_vec(&[0, 1, 2], Endianness::Little, "f").is_err());

        let f = 2.5_f32.to_le_bytes();
        assert_eq!(
            decode_f32_vec(&f, Endianness::Little, "f").unwrap(),
            vec![2.5]
        );
        let big = 7_i64.to_be_bytes();
        assert_eq!(decode_i64_vec(&big, Endianness::Big, "f").unwrap(), vec![7]);
    }

    #[test]
    fn convert_to_u32_rejects_negative() {
        assert_eq!(convert_to_u32(5_i32, "f").unwrap(), 5);
        assert!(convert_to_u32(-1_i32, "f").is_err());
    }
}
