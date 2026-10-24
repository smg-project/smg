//! SGLang multimodal wire structs: the scheduler's own `MultimodalProcessorOutput`
//! and `MultimodalDataItem` (`managers/schedule_batch.py`), untagged
//! `msgspec.Struct(array_like=True)` positional arrays nested in the request's
//! `mm_inputs` slot, with tensors as SGLang's own msgpack extension (its
//! `utils/msgpack_utils.py`, extension type 2). The SMG plugin inside the
//! scheduler decodes the slot with SGLang's own hooks, so what this encoder
//! writes is what SGLang's encoder writes for the same object: every slot, the
//! defaults included, in declaration order. **Field order is the wire
//! contract**; the plugin declares both structs' field order at the handshake
//! and `SglangProtocol::check_layouts` compares these lists against it.
//!
//! The sender does none of the scheduler's work: the item hashes, the pad
//! values, the padded prompt and the M-RoPE positions are the scheduler's
//! (`MultimodalInputs.from_processor_output`, the model's pad pattern,
//! `_maybe_compute_mrope_positions`), as they are behind the Python servicer,
//! which hands the scheduler the same object. The tensors ride inline: the
//! plugin's decoder resolves no auxiliary frames on this wire.

use std::collections::BTreeMap;

use bytes::Bytes;
use rmpv::Value;
use serde::{
    de::{Error as _, IgnoredAny, SeqAccess, Visitor},
    ser::{Error as _, SerializeTuple},
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{
    codec::{encode_msgpack, tensor::serialize_ext},
    protocol::positional::{drain_trailing, next_field},
};

/// The struct names the plugin declares the field order of.
pub const MULTIMODAL_PROCESSOR_OUTPUT_NAME: &str = "MultimodalProcessorOutput";
pub const MULTIMODAL_DATA_ITEM_NAME: &str = "MultimodalDataItem";

/// The prefix of the scheduler's `MultimodalProcessorOutput` fields this side
/// sends, through `im_token_id`: the scheduler fills the slots after it with
/// its defaults (an `array_like` struct takes a shorter array), as it does for
/// the request struct, so a field a later release adds past the prefix costs
/// nothing here; the recorded release's whole list is
/// `recorded::PROCESSOR_OUTPUT_FIELDS_0_5_22`.
pub const PROCESSOR_OUTPUT_WIRE_FIELDS: [&str; 4] =
    ["mm_items", "input_ids", "padded_input_ids", "im_token_id"];

/// The same for `MultimodalDataItem`.
pub const DATA_ITEM_WIRE_FIELDS: [&str; 9] = [
    "modality",
    "hash",
    "pad_value",
    "offsets",
    "format",
    "feature",
    "precomputed_embeddings",
    "keep_device_embedding",
    "model_specific_data",
];

/// SGLang's msgpack extension type of a torch tensor
/// (`msgpack_utils._MSGPACK_EXT_TORCH_TENSOR`).
const EXT_TORCH_TENSOR: i8 = 2;
/// `MultimodalInputFormat.NORMAL`: the processor's raw features, not
/// precomputed embeddings.
const FORMAT_NORMAL: u8 = 1;
/// The device SGLang's encoder names for a host tensor; its decoder restores
/// the tensor there.
const DEVICE_CPU: &str = "cpu";

/// The dtype names this wire carries, as torch spells them, with their element
/// sizes: the names the Python servicer's reader accepts.
const DTYPES: [(&str, usize); 6] = [
    ("float32", 4),
    ("bfloat16", 2),
    ("float16", 2),
    ("uint8", 1),
    ("int64", 8),
    ("uint32", 4),
];

/// The element size of a dtype name this wire carries; `None` for any other.
pub fn dtype_element_size(dtype: &str) -> Option<usize> {
    DTYPES
        .iter()
        .find(|(name, _)| *name == dtype)
        .map(|(_, size)| *size)
}

/// The scheduler's `Modality` enum, encoded as its values (`IMAGE = 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SglangWireModality {
    Image = 1,
    Video = 2,
    Audio = 3,
}

impl SglangWireModality {
    fn from_value(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Image),
            2 => Some(Self::Video),
            3 => Some(Self::Audio),
            _ => None,
        }
    }
}

impl Serialize for SglangWireModality {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u8(*self as u8)
    }
}

/// A tensor as SGLang's own encoder writes one: msgpack extension 2 whose
/// payload is the length of the metadata (a big-endian u32), the metadata
/// (`[shape, dtype, device]` as msgpack), then the raw little-endian bytes.
/// The decoder in the scheduler rebuilds the tensor from exactly that.
#[derive(Debug, Clone, PartialEq)]
pub struct SglangWireTensor {
    /// The torch dtype name (`float32`, `bfloat16`, `float16`, `uint8`,
    /// `int64` or `uint32`).
    pub dtype: String,
    pub shape: Vec<u64>,
    /// The raw little-endian bytes: the shape's element count at the dtype's
    /// width.
    pub data: Bytes,
}

impl SglangWireTensor {
    /// A tensor from its wire parts, checked as the Python servicer's reader
    /// checks them: a dtype name this wire carries, and a byte length that is
    /// the shape's element count at that width (a sender and a receiver that
    /// disagree about the width would otherwise produce a plausible tensor of
    /// wrong numbers).
    pub fn new(dtype: impl Into<String>, shape: Vec<u64>, data: Bytes) -> Result<Self, String> {
        let dtype = dtype.into();
        let Some(width) = dtype_element_size(&dtype) else {
            return Err(format!(
                "unsupported tensor dtype {dtype:?}; this wire carries float32, bfloat16, \
                 float16, uint8, int64 and uint32"
            ));
        };
        let expected = shape
            .iter()
            .try_fold(1u64, |count, &dim| count.checked_mul(dim))
            .and_then(|count| count.checked_mul(width as u64))
            .ok_or_else(|| format!("tensor shape {shape:?} overflows"))?;
        if expected != data.len() as u64 {
            return Err(format!(
                "TensorData byte length mismatch for dtype={dtype:?}, shape={shape:?}: \
                 expected {expected}, got {}",
                data.len()
            ));
        }
        Ok(Self { dtype, shape, data })
    }

    /// The extension's payload: the metadata's length, the metadata, the
    /// bytes.
    fn ext_payload(&self) -> Result<Vec<u8>, String> {
        let metadata = encode_msgpack(&(&self.shape, &self.dtype, DEVICE_CPU))
            .map_err(|error| error.to_string())?;
        let length = u32::try_from(metadata.len())
            .map_err(|_| "tensor metadata longer than the wire's length field".to_string())?;
        let mut payload = Vec::with_capacity(4 + metadata.len() + self.data.len());
        payload.extend_from_slice(&length.to_be_bytes());
        payload.extend_from_slice(&metadata);
        payload.extend_from_slice(&self.data);
        Ok(payload)
    }

    /// A tensor from the extension's payload.
    fn from_ext_payload(payload: &[u8]) -> Result<Self, String> {
        let Some(length) = payload.get(..4) else {
            return Err("tensor extension is missing its metadata length".to_string());
        };
        let length = u32::from_be_bytes([length[0], length[1], length[2], length[3]]) as usize;
        let Some(metadata) = payload.get(4..4 + length) else {
            return Err("tensor extension has an invalid metadata length".to_string());
        };
        let (shape, dtype, _device): (Vec<u64>, String, String) =
            rmp_serde::from_slice(metadata).map_err(|error| error.to_string())?;
        Self::new(dtype, shape, Bytes::copy_from_slice(&payload[4 + length..]))
    }
}

impl Serialize for SglangWireTensor {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let payload = self.ext_payload().map_err(S::Error::custom)?;
        serialize_ext(EXT_TORCH_TENSOR, &payload, serializer)
    }
}

impl<'de> Deserialize<'de> for SglangWireTensor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::Ext(EXT_TORCH_TENSOR, payload) => {
                Self::from_ext_payload(&payload).map_err(D::Error::custom)
            }
            Value::Ext(code, _) => Err(D::Error::custom(format!(
                "expected a torch tensor extension (type {EXT_TORCH_TENSOR}), got type {code}"
            ))),
            other => Err(D::Error::custom(format!(
                "expected a torch tensor extension, got {other:?}"
            ))),
        }
    }
}

/// One multimodal item, the scheduler's `MultimodalDataItem`: a 9-slot
/// positional array `[modality, hash, pad_value, offsets, format, feature,
/// precomputed_embeddings, keep_device_embedding, model_specific_data]`. The
/// hash and the pad value are left to the scheduler, which derives them from
/// the feature (the Python servicer leaves them too).
#[derive(Debug, Clone, PartialEq)]
pub struct SglangWireMmItem {
    pub modality: SglangWireModality,
    /// Inclusive `[start, end]` token ranges of the item's placeholder tokens
    /// in the prompt; `None` when the sender found none (the Python servicer's
    /// rule, which warns and leaves the slot unset).
    pub offsets: Option<Vec<(u64, u64)>>,
    /// The processor's feature tensor (`pixel_values`).
    pub feature: Option<SglangWireTensor>,
    /// The model-specific tensors beside it (`image_grid_thw` and the like),
    /// in key order.
    pub model_specific_data: BTreeMap<String, SglangWireTensor>,
}

impl Serialize for SglangWireMmItem {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        const NIL: Option<()> = None;
        let mut tuple = serializer.serialize_tuple(DATA_ITEM_WIRE_FIELDS.len())?;
        tuple.serialize_element(&self.modality)?;
        tuple.serialize_element(&NIL)?; // hash: the scheduler hashes the feature
        tuple.serialize_element(&NIL)?; // pad_value: derived from the hash there
        tuple.serialize_element(&self.offsets)?;
        tuple.serialize_element(&FORMAT_NORMAL)?;
        tuple.serialize_element(&self.feature)?;
        tuple.serialize_element(&NIL)?; // precomputed_embeddings
        tuple.serialize_element(&false)?; // keep_device_embedding
        tuple.serialize_element(&self.model_specific_data)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for SglangWireMmItem {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ItemVisitor;

        impl<'de> Visitor<'de> for ItemVisitor {
            type Value = SglangWireMmItem;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "an SGLang MultimodalDataItem positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let modality: u8 = next_field(&mut seq, "modality")?;
                let modality = SglangWireModality::from_value(modality).ok_or_else(|| {
                    A::Error::custom(format!("unknown multimodal modality value {modality}"))
                })?;
                let _hash: IgnoredAny = next_field(&mut seq, "hash")?;
                let _pad_value: IgnoredAny = next_field(&mut seq, "pad_value")?;
                let offsets = next_field(&mut seq, "offsets")?;
                let _format: IgnoredAny = next_field(&mut seq, "format")?;
                let feature = next_field(&mut seq, "feature")?;
                let _precomputed: IgnoredAny = next_field(&mut seq, "precomputed_embeddings")?;
                let _keep_device: IgnoredAny = next_field(&mut seq, "keep_device_embedding")?;
                let model_specific_data = next_field(&mut seq, "model_specific_data")?;
                drain_trailing(&mut seq)?;
                Ok(SglangWireMmItem {
                    modality,
                    offsets,
                    feature,
                    model_specific_data,
                })
            }
        }

        deserializer.deserialize_seq(ItemVisitor)
    }
}

/// The request's multimodal payload, the scheduler's
/// `MultimodalProcessorOutput`: a positional array of which this side sends the
/// prefix through `im_token_id` (the token the model's pad pattern replaces),
/// with `mm_items` filled and the prompt-side slots (`input_ids`,
/// `padded_input_ids`) at their defaults; the other token ids and the position
/// tensors are the scheduler's defaults past the prefix, which computes what it
/// needs of them itself.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SglangWireMmInputs {
    pub mm_items: Vec<SglangWireMmItem>,
    pub im_token_id: Option<u32>,
}

impl Serialize for SglangWireMmInputs {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        const NIL: Option<()> = None;
        let mut tuple = serializer.serialize_tuple(PROCESSOR_OUTPUT_WIRE_FIELDS.len())?;
        tuple.serialize_element(&self.mm_items)?;
        tuple.serialize_element(&NIL)?; // input_ids
        tuple.serialize_element(&NIL)?; // padded_input_ids
        tuple.serialize_element(&self.im_token_id)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for SglangWireMmInputs {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct InputsVisitor;

        impl<'de> Visitor<'de> for InputsVisitor {
            type Value = SglangWireMmInputs;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "an SGLang MultimodalProcessorOutput positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mm_items = next_field(&mut seq, "mm_items")?;
                let _input_ids: IgnoredAny = next_field(&mut seq, "input_ids")?;
                let _padded_input_ids: IgnoredAny = next_field(&mut seq, "padded_input_ids")?;
                let im_token_id = next_field(&mut seq, "im_token_id")?;
                drain_trailing(&mut seq)?;
                Ok(SglangWireMmInputs {
                    mm_items,
                    im_token_id,
                })
            }
        }

        deserializer.deserialize_seq(InputsVisitor)
    }
}

/// The field order of the scheduler's structs in the release the wire was
/// recorded against (`__struct_fields__`, read from the wheel; 0.5.21 declares
/// the same), spelled out as the oracle the wire lists are checked against,
/// here and in the handshake test: the processor output as a prefix, the data
/// item whole.
#[cfg(test)]
pub(crate) mod recorded {
    /// `MultimodalProcessorOutput.__struct_fields__`, SGLang 0.5.22.
    pub(crate) const PROCESSOR_OUTPUT_FIELDS_0_5_22: [&str; 18] = [
        "mm_items",
        "input_ids",
        "padded_input_ids",
        "im_token_id",
        "im_start_id",
        "im_end_id",
        "slice_start_id",
        "slice_end_id",
        "video_token_id",
        "audio_token_id",
        "audio_start_id",
        "audio_end_id",
        "mrope_positions",
        "mrope_position_delta",
        "vision_position_ids",
        "media_nums_per_sample",
        "visible_frame_counts",
        "token_type_ids",
    ];
    /// `MultimodalDataItem.__struct_fields__`, SGLang 0.5.22.
    pub(crate) const DATA_ITEM_FIELDS_0_5_22: [&str; 9] = [
        "modality",
        "hash",
        "pad_value",
        "offsets",
        "format",
        "feature",
        "precomputed_embeddings",
        "keep_device_embedding",
        "model_specific_data",
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_msgpack, decode_value, unhex};

    /// `msgspec.msgpack.Encoder(enc_hook=sglang.srt.utils.msgpack_utils.enc_hook)`
    /// of `torch.arange(8, dtype=torch.float32).reshape(2, 4) * 0.5`.
    const PYTHON_PIXELS: &str = "c734020000001093920204a7666c6f61743332a3637075000000000000003f0000803f0000c03f00000040000020400000404000006040";
    /// The same for `torch.tensor([[1, 2, 2]], dtype=torch.int64)`.
    const PYTHON_GRID: &str =
        "c72a020000000e93920103a5696e743634a3637075010000000000000002000000000000000200000000000000";
    /// The same encoder's `MultimodalProcessorOutput(mm_items=[MultimodalDataItem(
    /// modality=Modality.IMAGE, feature=<pixels>, model_specific_data={"image_grid_thw":
    /// <grid>}, offsets=[(1, 3)])], im_token_id=7)`, SGLang 0.5.22., cut to its
    /// first four elements (the prefix this side sends; the engine writes all 18,
    /// the fourteen past the prefix nil): the four-element array it decodes alike.
    const PYTHON_OUTPUT: &str = "94919901c0c09192010301c734020000001093920204a7666c6f61743332a3637075000000000000003f0000803f0000c03f00000040000020400000404000006040c0c281ae696d6167655f677269645f746877c72a020000000e93920103a5696e743634a3637075010000000000000002000000000000000200000000000000c0c007";
    /// The same with `offsets` left unset and no model-specific tensors.
    const PYTHON_OUTPUT_NO_OFFSETS: &str = "94919901c0c0c001c734020000001093920204a7666c6f61743332a3637075000000000000003f0000803f0000c03f00000040000020400000404000006040c0c280c0c0c0";

    fn pixels() -> SglangWireTensor {
        let data: Vec<u8> = (0..8)
            .flat_map(|i| (i as f32 * 0.5).to_le_bytes())
            .collect();
        SglangWireTensor::new("float32", vec![2, 4], Bytes::from(data)).unwrap()
    }

    fn grid() -> SglangWireTensor {
        let data: Vec<u8> = [1i64, 2, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        SglangWireTensor::new("int64", vec![1, 3], Bytes::from(data)).unwrap()
    }

    fn inputs() -> SglangWireMmInputs {
        SglangWireMmInputs {
            mm_items: vec![SglangWireMmItem {
                modality: SglangWireModality::Image,
                offsets: Some(vec![(1, 3)]),
                feature: Some(pixels()),
                model_specific_data: BTreeMap::from([("image_grid_thw".to_string(), grid())]),
            }],
            im_token_id: Some(7),
        }
    }

    /// A tensor is SGLang's extension 2: the metadata length, the metadata
    /// `[shape, dtype, "cpu"]`, the raw bytes; byte for byte what SGLang's
    /// own encoder writes.
    #[test]
    fn tensors_encode_as_sglangs_own_extension() {
        assert_eq!(encode_msgpack(&pixels()).unwrap(), unhex(PYTHON_PIXELS));
        assert_eq!(encode_msgpack(&grid()).unwrap(), unhex(PYTHON_GRID));
        let empty = SglangWireTensor::new("bfloat16", vec![0, 4], Bytes::new()).unwrap();
        assert_eq!(
            encode_msgpack(&empty).unwrap(),
            unhex("c715020000001193920004a862666c6f61743136a3637075")
        );
        for (hex, tensor) in [(PYTHON_PIXELS, pixels()), (PYTHON_GRID, grid())] {
            let back: SglangWireTensor = decode_msgpack(&unhex(hex)).unwrap();
            assert_eq!(back, tensor);
        }
    }

    /// The wire's dtype names and the byte length are checked as the Python
    /// servicer's reader checks them.
    #[test]
    fn tensors_are_checked_by_dtype_and_length() {
        let error = SglangWireTensor::new("float64", vec![1], Bytes::from(vec![0; 8])).unwrap_err();
        assert!(
            error.contains("unsupported tensor dtype \"float64\""),
            "{error}"
        );
        let error = SglangWireTensor::new("int64", vec![4], Bytes::from(vec![0; 16])).unwrap_err();
        assert!(
            error.contains(
                "byte length mismatch for dtype=\"int64\", shape=[4]: expected 32, got 16"
            ),
            "{error}"
        );
        for (dtype, width) in DTYPES {
            SglangWireTensor::new(dtype, vec![2], Bytes::from(vec![0; 2 * width]))
                .unwrap_or_else(|error| panic!("{dtype}: {error}"));
        }
        assert_eq!(dtype_element_size("bfloat16"), Some(2));
        assert_eq!(dtype_element_size("bool"), None);
    }

    /// The whole slot is what SGLang's own encoder writes for the object the
    /// Python servicer builds: every default in its place, the modality and
    /// the format as enum values, the offsets as `[start, end]` pairs.
    #[test]
    fn the_processor_output_matches_sglangs_own_encoding() {
        assert_eq!(encode_msgpack(&inputs()).unwrap(), unhex(PYTHON_OUTPUT));
        let no_offsets = SglangWireMmInputs {
            mm_items: vec![SglangWireMmItem {
                modality: SglangWireModality::Image,
                offsets: None,
                feature: Some(pixels()),
                model_specific_data: BTreeMap::new(),
            }],
            im_token_id: None,
        };
        assert_eq!(
            encode_msgpack(&no_offsets).unwrap(),
            unhex(PYTHON_OUTPUT_NO_OFFSETS)
        );
        let Value::Array(slots) = decode_value(&unhex(PYTHON_OUTPUT)).unwrap() else {
            panic!("expected a positional array");
        };
        assert_eq!(slots.len(), PROCESSOR_OUTPUT_WIRE_FIELDS.len());
        let Value::Array(items) = &slots[0] else {
            panic!("expected mm_items");
        };
        let Value::Array(item) = &items[0] else {
            panic!("expected an item array");
        };
        assert_eq!(item.len(), DATA_ITEM_WIRE_FIELDS.len());
        assert_eq!(item[0], Value::from(1u8), "Modality.IMAGE");
        assert_eq!(
            (&item[1], &item[2]),
            (&Value::Nil, &Value::Nil),
            "hash, pad_value"
        );
        assert_eq!(item[4], Value::from(1u8), "MultimodalInputFormat.NORMAL");
        assert_eq!(item[7], Value::Boolean(false), "keep_device_embedding");
        assert_eq!(slots[3], Value::from(7u8), "im_token_id");
        // The engine's own encoding of the same output writes all 18 slots, the
        // fourteen past the prefix nil: it decodes to the same value.
        let all_slots = format!("dc0012{}{}", &PYTHON_OUTPUT[2..], "c0".repeat(14));
        let back: SglangWireMmInputs = decode_msgpack(&unhex(&all_slots)).unwrap();
        assert_eq!(back, inputs());
    }

    #[test]
    fn the_processor_output_round_trips() {
        let encoded = encode_msgpack(&inputs()).unwrap();
        let back: SglangWireMmInputs = decode_msgpack(&encoded).unwrap();
        assert_eq!(back, inputs());
        let back: Option<SglangWireMmInputs> =
            decode_msgpack(&encode_msgpack(&None::<()>).unwrap()).unwrap();
        assert_eq!(back, None);
    }

    /// The wire lists against the recorded release's declarations: the
    /// processor output's prefix through `im_token_id` (the first field past it
    /// is the scheduler's), the data item whole.
    #[test]
    fn the_wire_field_lists_match_the_recorded_release() {
        assert!(recorded::PROCESSOR_OUTPUT_FIELDS_0_5_22.starts_with(&PROCESSOR_OUTPUT_WIRE_FIELDS));
        assert_eq!(
            recorded::PROCESSOR_OUTPUT_FIELDS_0_5_22[PROCESSOR_OUTPUT_WIRE_FIELDS.len()],
            "im_start_id"
        );
        assert_eq!(DATA_ITEM_WIRE_FIELDS, recorded::DATA_ITEM_FIELDS_0_5_22);
        assert_eq!(PROCESSOR_OUTPUT_WIRE_FIELDS[3], "im_token_id");
        assert_eq!(DATA_ITEM_WIRE_FIELDS[5], "feature");
    }
}
