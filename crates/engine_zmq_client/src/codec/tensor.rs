// Adapted from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/tensor.rs and protocol/logprobs/array.rs.
//
// Mirrors Python `vllm/v1/serial_utils.py` `encode_ndarray`/`encode_tensor`.

use std::{collections::BTreeMap, fmt};

use bytemuck::{cast_slice, Pod};
use bytes::Bytes;
use half::{bf16, f16};
use rmpv::Value;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use super::{
    dtype::{
        convert_to_u32, decode_error, decode_f32_vec, decode_float_vec, decode_i32_vec,
        decode_i64_vec, parse_dtype, Endianness, ScalarType,
    },
    encode_msgpack,
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

/// Write `payload` as a msgpack extension value of type `code` (the
/// `_ExtStruct` newtype is how `rmp_serde` spells one).
pub(crate) fn serialize_ext<S>(
    code: i8,
    payload: &[u8],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    MsgpackExtRef((code, ByteSlice(payload))).serialize(serializer)
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

    // The send side moves large tensors into aux frames through
    // `encode_with_aux_frames`; the receive side resolves them with
    // `resolve_array_bytes` (logprobs arrays may arrive as aux frames too).
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
            Self::RawView(bytes) => serialize_ext(CUSTOM_TYPE_RAW_VIEW, bytes, serializer),
        }
    }
}

/// Tensors of at least this many bytes leave the primary msgpack payload and
/// travel as auxiliary frames of the same message, sharing their [`Bytes`]
/// storage ([`encode_with_aux_frames`]); smaller ones stay inline, where the
/// copy into the payload costs less than the clone of the payload's metadata
/// the extraction takes.
pub const TENSOR_AUX_MIN_BYTES: usize = 32 * 1024;

/// A wire value that carries tensors. A request payload implements it over
/// its own shape (nested lists, maps, masks); the send side moves large
/// tensors into auxiliary frames through it without knowing that shape.
pub trait TensorFields {
    /// Visit every tensor payload, in wire order.
    fn for_each_tensor(&self, visit: &mut dyn FnMut(&WireArrayData));
    /// Visit every tensor payload mutably, in the same order.
    fn for_each_tensor_mut(&mut self, visit: &mut dyn FnMut(&mut WireArrayData));
}

impl TensorFields for WireNdArray {
    fn for_each_tensor(&self, visit: &mut dyn FnMut(&WireArrayData)) {
        visit(&self.data);
    }

    fn for_each_tensor_mut(&mut self, visit: &mut dyn FnMut(&mut WireArrayData)) {
        visit(&mut self.data);
    }
}

impl<T: TensorFields> TensorFields for Option<T> {
    fn for_each_tensor(&self, visit: &mut dyn FnMut(&WireArrayData)) {
        if let Some(value) = self {
            value.for_each_tensor(visit);
        }
    }

    fn for_each_tensor_mut(&mut self, visit: &mut dyn FnMut(&mut WireArrayData)) {
        if let Some(value) = self {
            value.for_each_tensor_mut(visit);
        }
    }
}

impl<T: TensorFields> TensorFields for Vec<T> {
    fn for_each_tensor(&self, visit: &mut dyn FnMut(&WireArrayData)) {
        for value in self {
            value.for_each_tensor(visit);
        }
    }

    fn for_each_tensor_mut(&mut self, visit: &mut dyn FnMut(&mut WireArrayData)) {
        for value in self {
            value.for_each_tensor_mut(visit);
        }
    }
}

impl<K, T: TensorFields> TensorFields for BTreeMap<K, T> {
    fn for_each_tensor(&self, visit: &mut dyn FnMut(&WireArrayData)) {
        for value in self.values() {
            value.for_each_tensor(visit);
        }
    }

    fn for_each_tensor_mut(&mut self, visit: &mut dyn FnMut(&mut WireArrayData)) {
        for value in self.values_mut() {
            value.for_each_tensor_mut(visit);
        }
    }
}

/// Where a payload's tensors go on the wire, decided by one scan before
/// anything is cloned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorPlacement {
    /// Every tensor stays inline: none reaches [`TENSOR_AUX_MIN_BYTES`], or
    /// the payload already addresses auxiliary frames (a caller's own
    /// indices, which a second extraction would rebase).
    Inline,
    /// At least one tensor leaves the payload as an auxiliary frame.
    AuxFrames,
}

/// The placement of `payload`'s tensors.
pub fn tensor_placement<T: TensorFields + ?Sized>(payload: &T) -> TensorPlacement {
    let mut large = false;
    let mut indexed = false;
    payload.for_each_tensor(&mut |data| match data {
        WireArrayData::AuxIndex(_) => indexed = true,
        WireArrayData::RawView(bytes) => large |= bytes.len() >= TENSOR_AUX_MIN_BYTES,
    });
    if large && !indexed {
        TensorPlacement::AuxFrames
    } else {
        TensorPlacement::Inline
    }
}

/// Encode `payload` as msgpack with its large tensors as auxiliary frames, in
/// visiting order and addressed one-based (frame 0 is the payload itself). A
/// payload whose tensors stay inline ([`tensor_placement`]) is encoded as it
/// is, without a clone; `payload` is never changed either way.
pub fn encode_with_aux_frames<T>(payload: &T) -> Result<(Vec<u8>, Vec<Bytes>)>
where
    T: Serialize + Clone + TensorFields + fmt::Debug,
{
    if tensor_placement(payload) == TensorPlacement::Inline {
        return Ok((encode_msgpack(payload)?, Vec::new()));
    }
    // Tensor storage is `Bytes`: the clone shares every buffer, and the
    // frames below share them again.
    let mut wire = payload.clone();
    let mut frames = Vec::new();
    wire.for_each_tensor_mut(&mut |data| {
        if let WireArrayData::RawView(bytes) = data {
            if bytes.len() >= TENSOR_AUX_MIN_BYTES {
                frames.push(bytes.clone());
                *data = WireArrayData::AuxIndex(frames.len());
            }
        }
    });
    Ok((encode_msgpack(&wire)?, frames))
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

/// A floating-point tensor decoded to `f32`, flattened in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedFloatTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Decode a floating-point tensor of any rank (float32, float16, bfloat16 or
/// float64 elements) to `f32` values.
pub fn decode_tensor_f32(
    value: WireNdArray,
    field: &str,
    frames: &[Bytes],
) -> Result<DecodedFloatTensor> {
    let (shape, bytes, scalar, endianness) = decode_array_metadata(
        value,
        field,
        frames,
        &[
            ScalarType::F32,
            ScalarType::F16,
            ScalarType::BF16,
            ScalarType::F64,
        ],
    )?;
    let data = decode_float_vec(&bytes, scalar, endianness, field)?;
    Ok(DecodedFloatTensor { shape, data })
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
        ScalarType::F16 | ScalarType::BF16 | ScalarType::F32 | ScalarType::F64 => Err(
            decode_error(field, &format!("expected integer dtype, got {scalar:?}")),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::unhex;

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

    /// Golden bytes from vLLM's `MsgpackEncoder` (vLLM 0.30.1rc1) for
    /// `torch.tensor([0.25, -1.5, 3.0])` in float32, bfloat16 and float16:
    /// `(dtype, shape, raw-view ext)` with the dtype as the torch name.
    #[test]
    fn decode_tensor_f32_reads_vllm_encoded_torch_tensors() {
        for (hex, dtype) in [
            (
                "93a7666c6f617433329103c70c030000803e0000c0bf00004040",
                "float32",
            ),
            ("93a862666c6f617431369103c70603803ec0bf4040", "bfloat16"),
            ("93a7666c6f617431369103c70603003400be0042", "float16"),
        ] {
            let bytes = unhex(hex);
            let wire: WireNdArray = rmp_serde::from_slice(&bytes).expect("wire tensor");
            assert_eq!(wire.dtype, dtype);
            assert_eq!(wire.shape, vec![3]);
            let decoded = decode_tensor_f32(wire, "t", &[]).unwrap();
            assert_eq!(decoded.shape, vec![3]);
            assert_eq!(decoded.data, vec![0.25, -1.5, 3.0], "{dtype}");
        }
    }

    /// A tensor over the inline threshold rides as an aux frame: the wire
    /// tuple carries the frame index (`93 a7 float32 91 60 01`).
    #[test]
    fn decode_tensor_f32_follows_an_aux_frame_index() {
        let wire: WireNdArray =
            rmp_serde::from_slice(&unhex("93a7666c6f61743332916001")).expect("wire tensor");
        assert_eq!(wire.data, WireArrayData::AuxIndex(1));
        let values: Vec<f32> = (0..96).map(|i| i as f32 / 8.0).collect();
        let frame = Bytes::from(
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let decoded = decode_tensor_f32(wire, "t", &[Bytes::new(), frame]).unwrap();
        assert_eq!(decoded.shape, vec![96]);
        assert_eq!(decoded.data, values);
    }

    #[test]
    fn decode_tensor_f32_rejects_integer_dtypes() {
        let wire = WireNdArray::from_i64(vec![1], vec![7]).unwrap();
        let error = decode_tensor_f32(wire, "t", &[]).unwrap_err();
        assert!(error.to_string().contains("expected dtype in"), "{error}");
    }

    /// A tensor leaves the payload at the auxiliary size and not below it;
    /// a payload that already addresses frames keeps every tensor where it
    /// is; the frames come in visiting order and share the tensors' storage,
    /// and the payload the caller holds is untouched.
    #[test]
    fn large_tensors_leave_the_payload_as_frames_in_order_without_a_copy() {
        let tensor = |n: usize| WireNdArray::from_raw("uint8", vec![n], vec![n as u8; n]);
        let small = vec![tensor(0), tensor(255), tensor(TENSOR_AUX_MIN_BYTES - 1)];
        assert_eq!(tensor_placement(&small), TensorPlacement::Inline);
        let (payload, frames) = encode_with_aux_frames(&small).unwrap();
        assert!(frames.is_empty());
        assert_eq!(payload, encode_msgpack(&small).unwrap());

        let mixed = vec![
            tensor(TENSOR_AUX_MIN_BYTES),
            tensor(16),
            tensor(2 * TENSOR_AUX_MIN_BYTES),
        ];
        assert_eq!(tensor_placement(&mixed), TensorPlacement::AuxFrames);
        let inline = encode_msgpack(&mixed).unwrap();
        let (payload, frames) = encode_with_aux_frames(&mixed).unwrap();
        assert_eq!(
            frames.iter().map(Bytes::len).collect::<Vec<_>>(),
            vec![TENSOR_AUX_MIN_BYTES, 2 * TENSOR_AUX_MIN_BYTES]
        );
        let storage = |array: &WireNdArray| array.data.as_raw_view().unwrap().as_ptr();
        assert_eq!(frames[0].as_ptr(), storage(&mixed[0]));
        assert_eq!(frames[1].as_ptr(), storage(&mixed[2]));
        assert!(payload.len() < 64, "{} bytes", payload.len());
        assert_eq!(
            encode_msgpack(&mixed).unwrap(),
            inline,
            "the caller's value"
        );
        let decoded: Vec<WireNdArray> = rmp_serde::from_slice(&payload).unwrap();
        assert_eq!(decoded[0].data, WireArrayData::AuxIndex(1));
        assert_eq!(decoded[1].data, mixed[1].data);
        assert_eq!(decoded[2].data, WireArrayData::AuxIndex(2));
        let mut all = vec![Bytes::from(payload)];
        all.extend(frames);
        assert_eq!(
            resolve_array_bytes(decoded[2].data.clone(), "t", &all).unwrap(),
            *mixed[2].data.as_raw_view().unwrap()
        );

        let mut indexed = mixed.clone();
        indexed[1].data = WireArrayData::AuxIndex(1);
        assert_eq!(tensor_placement(&indexed), TensorPlacement::Inline);
        let (payload, frames) = encode_with_aux_frames(&indexed).unwrap();
        assert!(frames.is_empty());
        assert_eq!(payload, encode_msgpack(&indexed).unwrap());
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
