// Adapted from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/tensor.rs and protocol/logprobs/array.rs.
//
// Mirrors Python `vllm/v1/serial_utils.py` `encode_ndarray`/`encode_tensor`.

use bytemuck::{cast_slice, Pod};
use bytes::Bytes;
use half::{bf16, f16};
use rmpv::Value;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use super::dtype::{
    convert_to_u32, decode_error, decode_f32_vec, decode_i32_vec, decode_i64_vec, parse_dtype,
    Endianness, ScalarType,
};
use crate::error::Result;

/// Tensors and ndarrays are encoded with this msgpack extension type in Python.
/// See `vllm/v1/serial_utils.py` `CUSTOM_TYPE_RAW_VIEW`.
const CUSTOM_TYPE_RAW_VIEW: i8 = 3;

/// Total number of elements implied by `shape`, or `None` on overflow.
pub fn checked_numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
}

#[derive(Serialize)]
#[serde(rename = "_ExtStruct")]
struct MsgpackExtRef<'a>((i8, ByteSlice<'a>));

struct ByteSlice<'a>(&'a [u8]);

struct PodVec<T: Pod>(Vec<T>);

impl<T: Pod> AsRef<[u8]> for PodVec<T> {
    fn as_ref(&self) -> &[u8] {
        cast_slice(&self.0)
    }
}

fn bytes_from_pod_vec<T>(data: Vec<T>) -> Bytes
where
    T: Pod + Send + 'static,
{
    Bytes::from_owner(PodVec(data))
}

impl Serialize for ByteSlice<'_> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(self.0)
    }
}

/// Python ndarray/tensor wire tuple encoded as `(dtype, shape, data)`.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct WireNdArray {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: WireArrayData,
}

impl WireNdArray {
    /// float32 tensor/ndarray backed by native-endian raw-view bytes (no copy).
    pub fn from_f32(shape: Vec<usize>, data: Vec<f32>) -> std::result::Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self::from_raw_bytes(
            "float32",
            shape,
            bytes_from_pod_vec(data),
        ))
    }

    /// float16 tensor/ndarray backed by native-endian raw-view bytes (no copy).
    pub fn from_f16(shape: Vec<usize>, data: Vec<f16>) -> std::result::Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self::from_raw_bytes(
            "float16",
            shape,
            bytes_from_pod_vec(data),
        ))
    }

    /// bfloat16 tensor/ndarray backed by native-endian raw-view bytes (no copy).
    pub fn from_bf16(shape: Vec<usize>, data: Vec<bf16>) -> std::result::Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self::from_raw_bytes(
            "bfloat16",
            shape,
            bytes_from_pod_vec(data),
        ))
    }

    /// int64 tensor/ndarray backed by native-endian raw-view bytes (no copy).
    pub fn from_i64(shape: Vec<usize>, data: Vec<i64>) -> std::result::Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self::from_raw_bytes(
            "int64",
            shape,
            bytes_from_pod_vec(data),
        ))
    }

    /// uint32 tensor/ndarray backed by native-endian raw-view bytes (no copy).
    pub fn from_u32(shape: Vec<usize>, data: Vec<u32>) -> std::result::Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self::from_raw_bytes(
            "uint32",
            shape,
            bytes_from_pod_vec(data),
        ))
    }

    /// bool tensor/ndarray: one byte per element (`torch.bool` storage), not a
    /// packed bitmap. `false -> 0`, `true -> 1`.
    pub fn from_bool(shape: Vec<usize>, data: Vec<bool>) -> std::result::Result<Self, String> {
        validate_element_count(&shape, data.len())?;
        Ok(Self {
            dtype: "bool".to_string(),
            shape,
            data: WireArrayData::RawView(Bytes::from(
                data.into_iter().map(u8::from).collect::<Vec<_>>(),
            )),
        })
    }

    /// Build from already-encoded raw-view bytes matching `dtype`/`shape`.
    pub fn from_raw(dtype: impl Into<String>, shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self::from_raw_bytes(dtype, shape, Bytes::from(data))
    }

    /// Build from little-endian `float32` bytes, casting each element to
    /// `dtype`. Mirrors the model-dtype cast the engine's own frontend applies
    /// to floating multimodal tensors before they reach the model.
    pub fn from_f32_bytes_cast(
        dtype: super::dtype::ModelDtype,
        shape: Vec<usize>,
        data: &[u8],
    ) -> std::result::Result<Self, String> {
        use super::dtype::ModelDtype;
        if !data.len().is_multiple_of(4) {
            return Err(format!(
                "float32 buffer length {} is not a multiple of 4",
                data.len()
            ));
        }
        validate_element_count(&shape, data.len() / 4)?;
        let floats = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c));
        Ok(match dtype {
            ModelDtype::Float32 => Self::from_raw("float32", shape, data.to_vec()),
            ModelDtype::Float16 => Self::from_raw_bytes(
                "float16",
                shape,
                bytes_from_pod_vec(floats.map(f16::from_f32).collect::<Vec<_>>()),
            ),
            ModelDtype::BFloat16 => Self::from_raw_bytes(
                "bfloat16",
                shape,
                bytes_from_pod_vec(floats.map(bf16::from_f32).collect::<Vec<_>>()),
            ),
        })
    }

    /// Build from an owned immutable raw-view buffer.
    pub fn from_raw_bytes(dtype: impl Into<String>, shape: Vec<usize>, data: Bytes) -> Self {
        Self {
            dtype: dtype.into(),
            shape,
            data: WireArrayData::RawView(data),
        }
    }

    // NOTE: send-side aux-frame extraction (moving large inline tensors into
    // ordered aux frames) is added with the typed multimodal module — text
    // requests carry no tensors. The receive-side resolution below is used now
    // (logprobs arrays may arrive as aux frames).
}

/// Validate that the shape product matches the data length.
fn validate_element_count(shape: &[usize], len: usize) -> std::result::Result<(), String> {
    let expected = checked_numel(shape)
        .ok_or_else(|| format!("tensor shape product overflows usize: {shape:?}"))?;
    if expected == len {
        Ok(())
    } else {
        Err(format!(
            "tensor data length {len} does not match shape {shape:?} product {expected}"
        ))
    }
}

/// Same wire shape as [`WireNdArray`]; multimodal payloads use it for tensors.
pub type WireTensor = WireNdArray;

/// Array/tensor payload inside [`WireNdArray`]: either an inline raw-view ext or
/// a one-based index into the multipart aux-frame list.
#[derive(Debug, Clone, PartialEq)]
pub enum WireArrayData {
    /// Index of the aux frame holding this array's raw bytes (one-based).
    AuxIndex(usize),
    /// Inline raw bytes of this array/tensor.
    RawView(Bytes),
}

impl WireArrayData {
    /// Consume into the inline raw view, if present.
    pub fn into_raw_view(self) -> Option<Bytes> {
        match self {
            Self::RawView(bytes) => Some(bytes),
            Self::AuxIndex(_) => None,
        }
    }

    /// Borrow the inline raw view, if present.
    pub fn as_raw_view(&self) -> Option<&Bytes> {
        match self {
            Self::RawView(bytes) => Some(bytes),
            Self::AuxIndex(_) => None,
        }
    }
}

impl<'de> Deserialize<'de> for WireArrayData {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::Ext(tag, bytes) if tag == CUSTOM_TYPE_RAW_VIEW => {
                Ok(Self::RawView(Bytes::from(bytes)))
            }
            Value::Ext(tag, _) => Err(serde::de::Error::custom(format!(
                "unsupported extension type code {tag}"
            ))),
            Value::Integer(index) => index
                .as_u64()
                .map(|index| Self::AuxIndex(index as usize))
                .ok_or_else(|| {
                    serde::de::Error::custom("aux frame index must be a non-negative integer")
                }),
            other => Err(serde::de::Error::custom(format!(
                "expected raw-view ext or aux frame index, got {other:?}"
            ))),
        }
    }
}

impl Serialize for WireArrayData {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::AuxIndex(index) => serializer.serialize_u64(*index as u64),
            Self::RawView(bytes) => {
                MsgpackExtRef((CUSTOM_TYPE_RAW_VIEW, ByteSlice(bytes))).serialize(serializer)
            }
        }
    }
}

/// A decoded rank-2 array with its dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedArray2<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

/// Resolve an array payload to raw bytes, following an aux-frame index into
/// `frames` (frame 0 is the primary msgpack, so indices are one-based).
pub fn resolve_array_bytes(value: WireArrayData, field: &str, frames: &[Bytes]) -> Result<Bytes> {
    match value {
        WireArrayData::RawView(bytes) => Ok(bytes),
        WireArrayData::AuxIndex(index) => {
            let frame = frames.get(index).ok_or_else(|| {
                decode_error(
                    field,
                    &format!(
                        "aux frame index {index} out of range for {} frames",
                        frames.len()
                    ),
                )
            })?;
            // Aux frames are refcounted buffers off the wire: share, do not copy.
            Ok(frame.clone())
        }
    }
}

/// Validate that `byte_len` matches the shape product times the scalar size.
pub fn validate_byte_length(
    shape: &[usize],
    byte_len: usize,
    field: &str,
    scalar: ScalarType,
) -> Result<()> {
    let element_count = checked_numel(shape)
        .ok_or_else(|| decode_error(field, "shape element count overflowed usize"))?;
    let expected = element_count
        .checked_mul(scalar.element_size())
        .ok_or_else(|| decode_error(field, "byte length overflowed usize"))?;
    if expected != byte_len {
        return Err(decode_error(
            field,
            &format!("byte length mismatch: expected {expected}, got {byte_len}"),
        ));
    }
    Ok(())
}

fn decode_array_metadata(
    value: WireNdArray,
    field: &str,
    frames: &[Bytes],
    expected_scalars: &[ScalarType],
) -> Result<(Vec<usize>, Bytes, ScalarType, Endianness)> {
    let WireNdArray { dtype, shape, data } = value;
    let (scalar, endianness) = parse_dtype(&dtype, field)?;
    if !expected_scalars.contains(&scalar) {
        return Err(decode_error(
            field,
            &format!("expected dtype in {expected_scalars:?}, got {dtype}"),
        ));
    }
    let bytes = resolve_array_bytes(data, field, frames)?;
    validate_byte_length(shape.as_slice(), bytes.len(), field, scalar)?;
    Ok((shape, bytes, scalar, endianness))
}

/// Decode a rank-2 integer array (i32/i64) to `u32` rows.
pub fn decode_array2_u32(
    value: WireNdArray,
    field: &str,
    frames: &[Bytes],
) -> Result<DecodedArray2<u32>> {
    let (shape, bytes, scalar, endianness) =
        decode_array_metadata(value, field, frames, &[ScalarType::I32, ScalarType::I64])?;
    if shape.len() != 2 {
        return Err(decode_error(
            field,
            &format!("expected rank-2 array, got rank {}", shape.len()),
        ));
    }
    let data = decode_int_as_u32(&bytes, scalar, endianness, field)?;
    Ok(DecodedArray2 {
        rows: shape[0],
        cols: shape[1],
        data,
    })
}

/// Decode a rank-1 integer array (i32/i64) to a `Vec<u32>`.
pub fn decode_array1_u32(value: WireNdArray, field: &str, frames: &[Bytes]) -> Result<Vec<u32>> {
    let (shape, bytes, scalar, endianness) =
        decode_array_metadata(value, field, frames, &[ScalarType::I32, ScalarType::I64])?;
    if shape.len() != 1 {
        return Err(decode_error(
            field,
            &format!("expected rank-1 array, got rank {}", shape.len()),
        ));
    }
    decode_int_as_u32(&bytes, scalar, endianness, field)
}

/// Decode a rank-2 float32 array.
pub fn decode_array2_f32(
    value: WireNdArray,
    field: &str,
    frames: &[Bytes],
) -> Result<DecodedArray2<f32>> {
    let (shape, bytes, _, endianness) =
        decode_array_metadata(value, field, frames, &[ScalarType::F32])?;
    if shape.len() != 2 {
        return Err(decode_error(
            field,
            &format!("expected rank-2 array, got rank {}", shape.len()),
        ));
    }
    let data = decode_f32_vec(&bytes, endianness, field)?;
    Ok(DecodedArray2 {
        rows: shape[0],
        cols: shape[1],
        data,
    })
}

fn decode_int_as_u32(
    bytes: &[u8],
    scalar: ScalarType,
    endianness: Endianness,
    field: &str,
) -> Result<Vec<u32>> {
    match scalar {
        ScalarType::I32 => decode_i32_vec(bytes, endianness, field)?
            .into_iter()
            .map(|value| convert_to_u32(value, field))
            .collect(),
        ScalarType::I64 => decode_i64_vec(bytes, endianness, field)?
            .into_iter()
            .map(|value| convert_to_u32(value, field))
            .collect(),
        ScalarType::F32 => Err(decode_error(field, "expected integer dtype, got f32")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_view_serializes_as_msgpack_ext() {
        let bytes = vec![1, 2, 3, 4];
        let encoded =
            rmp_serde::to_vec_named(&WireArrayData::RawView(bytes.clone().into())).expect("encode");
        let expected = rmp_serde::to_vec_named(&Value::Ext(CUSTOM_TYPE_RAW_VIEW, bytes.clone()))
            .expect("encode expected");
        assert_eq!(encoded, expected);
        assert_eq!(
            rmpv::decode::read_value(&mut std::io::Cursor::new(encoded)).expect("decode"),
            Value::Ext(CUSTOM_TYPE_RAW_VIEW, bytes)
        );
    }

    #[test]
    fn constructors_build_raw_view_tensors_without_copy() {
        let f32_data = vec![1.0, 2.5];
        let f32_data_ptr = f32_data.as_ptr().cast::<u8>();
        let f32_tensor = WireNdArray::from_f32(vec![2], f32_data).unwrap();
        assert_eq!(f32_tensor.dtype, "float32");
        assert_eq!(f32_tensor.shape, vec![2]);
        let f32_raw_view = f32_tensor.data.into_raw_view().expect("raw view");
        // Zero-copy: the raw view aliases the original allocation.
        assert_eq!(f32_raw_view.as_ptr(), f32_data_ptr);
        assert_eq!(
            f32_raw_view,
            [1.0_f32, 2.5]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect::<Vec<_>>()
        );

        let i64_tensor = WireNdArray::from_i64(vec![1], vec![-7]).unwrap();
        assert_eq!(i64_tensor.dtype, "int64");
        assert_eq!(
            i64_tensor.data.into_raw_view().expect("raw view").as_ref(),
            (-7_i64).to_ne_bytes().as_ref()
        );

        let bool_tensor = WireNdArray::from_bool(vec![2], vec![false, true]).unwrap();
        assert_eq!(
            bool_tensor.data.into_raw_view().expect("raw view"),
            vec![0, 1]
        );
    }

    #[test]
    fn constructors_validate_shape_product() {
        let err = WireNdArray::from_f32(vec![2, 2], vec![1.0, 2.0]).unwrap_err();
        assert!(err.contains("does not match shape"));
    }

    #[test]
    fn decode_array_roundtrips_through_aux_frame() {
        // Encode a rank-1 i32 array as an aux frame, then decode it back.
        let arr = WireNdArray::from_raw_bytes(
            "<i4",
            vec![3],
            Bytes::from(
                [10_i32, 20, 30]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>(),
            ),
        );
        // frame 0 = primary (unused here), frame 1 = the array bytes.
        let raw = arr.data.as_raw_view().expect("raw").clone();
        let framed = vec![Bytes::new(), raw];
        let via_aux = WireNdArray {
            dtype: "<i4".into(),
            shape: vec![3],
            data: WireArrayData::AuxIndex(1),
        };
        assert_eq!(
            decode_array1_u32(via_aux, "ids", &framed).unwrap(),
            vec![10, 20, 30]
        );
    }

    #[test]
    fn resolve_array_bytes_shares_the_aux_frame() {
        let frame = Bytes::from(vec![1_u8, 2, 3, 4]);
        let frames = vec![Bytes::new(), frame.clone()];
        let resolved = resolve_array_bytes(WireArrayData::AuxIndex(1), "ids", &frames).unwrap();
        // Zero-copy: the resolved payload aliases the received frame.
        assert_eq!(resolved.as_ptr(), frame.as_ptr());
    }

    #[test]
    fn decode_array2_f32_checks_rank_and_length() {
        let bytes = Bytes::from(
            [1.0_f32, 2.0, 3.0, 4.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        let arr = WireNdArray {
            dtype: "<f4".into(),
            shape: vec![2, 2],
            data: WireArrayData::RawView(bytes),
        };
        let decoded = decode_array2_f32(arr, "lp", &[Bytes::new()]).unwrap();
        assert_eq!((decoded.rows, decoded.cols), (2, 2));
        assert_eq!(decoded.data, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
