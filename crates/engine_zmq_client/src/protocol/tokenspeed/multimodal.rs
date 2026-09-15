// TokenSpeed multimodal wire structs — the native `MultimodalInputs` /
// `MultimodalDataItem` from `runtime/multimodal/inputs.py`. Unlike the
// request/output messages these are UNtagged `msgspec.Struct(array_like=True)`
// positional arrays. **Field order is the wire contract** — append only,
// never reorder.
//
// The msgpack transport bypasses the engine's `InputProcessor`, so everything
// it would derive must arrive precomputed: `pad_value` (via [`mm_pad_value`]),
// pad-substituted `input_ids` (the placeholder ranges overwritten with the pad
// value), and the original ids in `input_ids_unpadded`. MRoPE position tensors
// are not computed by anything on this wire yet: the translate rejects the
// known MRoPE families (items carrying `image_grid_thw`/`video_grid_thw`)
// rather than let them silently degrade to 1-D positions, and warns once for
// the rest.

use std::collections::BTreeMap;

use serde::{ser::SerializeTuple, Serialize, Serializer};

use crate::codec::tensor::WireTensor;

/// Engine-side modality discriminants (`inputs.py::Modality`).
///
/// NOTE: these differ from `smg.grpc.common.Modality`, which has AUDIO=2 and
/// VIDEO=3 — translate explicitly, never pass the proto int through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TokenSpeedWireModality {
    Image = 1,
    Video = 2,
    Audio = 3,
}

impl TokenSpeedWireModality {
    /// Pad-substitute modality tag (`inputs.py::_modality_pad_tag`):
    /// IMAGE=0, VIDEO=1, AUDIO=2 — offset from the enum discriminants.
    fn pad_tag(self) -> u64 {
        match self {
            Self::Image => 0,
            Self::Video => 1,
            Self::Audio => 2,
        }
    }
}

impl Serialize for TokenSpeedWireModality {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u8(*self as u8)
    }
}

/// First multimodal pad-substitute token id (`inputs.py::_MM_PAD_BASE`).
const MM_PAD_BASE: u64 = 1_000_000;

/// Per-modality pad-substitute hash slots (`inputs.py::_MM_PAD_HASH_SLOTS`,
/// `((2^31 - 1) - 1_000_000 + 1) / 3`).
const MM_PAD_HASH_SLOTS: u64 = 715_494_549;

/// The engine's `MultimodalDataItem.set_pad_value()` formula.
///
/// Prefix-cache content identity and speculative-decode pad substitution both
/// key off this exact value, and nothing computes it on the msgpack path, so
/// senders must reproduce it bit-for-bit. The result is always within
/// `[1_000_000, 2^31)`.
pub fn mm_pad_value(modality: TokenSpeedWireModality, hash: u64) -> u32 {
    let value = MM_PAD_BASE + modality.pad_tag() * MM_PAD_HASH_SLOTS + hash % MM_PAD_HASH_SLOTS;
    // Bounded by 1_000_000 + 2*715_494_549 + 715_494_548 = 2^31 - 3, so the
    // fallback is unreachable by construction.
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// One multimodal item — `MultimodalDataItem`, a 10-slot positional array:
/// `[modality, hash, pad_value, offsets, feature, feature_shm,
/// model_specific_data, encoded, encoded_deepstack, encode_handshake]`.
///
/// `encoded`/`encoded_deepstack` are scheduler-local ("always None on the
/// wire"), `feature_shm` is a same-host optimization, and `encode_handshake`
/// is EPD-only — this sender emits inline `feature` tensors and nil for all
/// four.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenSpeedWireMmItem {
    pub modality: TokenSpeedWireModality,
    /// Content hash: u64 little-endian fold of the gateway content hash. The
    /// engine uses it for within-batch dedup and as the pad-value seed.
    pub hash: u64,
    /// Precomputed [`mm_pad_value`]`(modality, hash)`.
    pub pad_value: u32,
    /// Inclusive `[start, end]` token ranges in the pad-substituted
    /// `input_ids`, one per placeholder span.
    pub offsets: Vec<(u64, u64)>,
    /// Primary encoder tensor. Kwarg naming (`pixel_values` vs `patches`) is
    /// the engine model integration's concern; the slot is anonymous.
    pub feature: WireTensor,
    /// Side tensors (grids, types, per-item sizes) with wire dtype preserved.
    pub model_specific_data: BTreeMap<String, WireTensor>,
}

impl Serialize for TokenSpeedWireMmItem {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        const NIL: Option<()> = None;
        let mut tuple = serializer.serialize_tuple(10)?;
        tuple.serialize_element(&self.modality)?;
        tuple.serialize_element(&self.hash)?;
        tuple.serialize_element(&self.pad_value)?;
        tuple.serialize_element(&self.offsets)?;
        tuple.serialize_element(&self.feature)?;
        tuple.serialize_element(&NIL)?; // feature_shm
        tuple.serialize_element(&self.model_specific_data)?;
        tuple.serialize_element(&NIL)?; // encoded (scheduler-local)
        tuple.serialize_element(&NIL)?; // encoded_deepstack (scheduler-local)
        tuple.serialize_element(&NIL)?; // encode_handshake (EPD only)
        tuple.end()
    }
}

/// Request-level multimodal payload — `MultimodalInputs`, a 7-slot positional
/// array: `[mm_items, im_token_id, video_token_id, mrope_positions,
/// mrope_position_delta, mrope_position_delta_scalar, <reserved>]`.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenSpeedWireMmInputs {
    pub mm_items: Vec<TokenSpeedWireMmItem>,
    pub im_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
}

impl Serialize for TokenSpeedWireMmInputs {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        const NIL: Option<()> = None;
        let mut tuple = serializer.serialize_tuple(7)?;
        tuple.serialize_element(&self.mm_items)?;
        tuple.serialize_element(&self.im_token_id)?;
        tuple.serialize_element(&self.video_token_id)?;
        tuple.serialize_element(&NIL)?; // mrope_positions (not derivable here)
        tuple.serialize_element(&NIL)?; // mrope_position_delta
        tuple.serialize_element(&NIL)?; // mrope_position_delta_scalar
        tuple.serialize_element(&NIL)?; // reserved trailing slot
        tuple.end()
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::codec::{decode_value, encode_msgpack};

    fn image_item() -> TokenSpeedWireMmItem {
        let mut model_specific_data = BTreeMap::new();
        model_specific_data.insert(
            "vit_grid".to_string(),
            WireTensor::from_raw(
                "uint32",
                vec![1, 3],
                vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
            ),
        );
        TokenSpeedWireMmItem {
            modality: TokenSpeedWireModality::Image,
            hash: 0xDEAD_BEEF,
            pad_value: mm_pad_value(TokenSpeedWireModality::Image, 0xDEAD_BEEF),
            offsets: vec![(2, 4)],
            feature: WireTensor::from_raw("bfloat16", vec![3, 2], vec![0u8; 12]),
            model_specific_data,
        }
    }

    /// Pinned against `inputs.py::set_pad_value`: `1_000_000 +
    /// tag * 715_494_549 + hash % 715_494_549`.
    #[test]
    fn pad_value_matches_engine_formula() {
        assert_eq!(
            mm_pad_value(TokenSpeedWireModality::Image, 0xDEAD_BEEF),
            1_000_000 + (0xDEAD_BEEFu64 % 715_494_549) as u32
        );
        assert_eq!(mm_pad_value(TokenSpeedWireModality::Image, 0), 1_000_000);
        assert_eq!(
            mm_pad_value(TokenSpeedWireModality::Video, 0),
            1_000_000 + 715_494_549
        );
        assert_eq!(
            mm_pad_value(TokenSpeedWireModality::Audio, 0),
            1_000_000 + 2 * 715_494_549
        );
        // Maximum stays within int32, mirroring the engine's guarantee.
        assert!(u64::from(mm_pad_value(TokenSpeedWireModality::Audio, 715_494_548)) < (1u64 << 31));
    }

    /// The item is an untagged 10-slot array with nils in the scheduler-local
    /// slots and the modality as the engine's enum int (IMAGE=1).
    #[test]
    fn item_encodes_as_untagged_ten_slot_array() {
        let encoded = encode_msgpack(&image_item()).unwrap();
        let Value::Array(slots) = decode_value(&encoded).unwrap() else {
            panic!("expected positional array");
        };
        assert_eq!(slots.len(), 10);
        assert_eq!(slots[0], Value::from(1u8)); // Modality.IMAGE
        assert_eq!(slots[1], Value::from(0xDEAD_BEEFu64));
        assert_eq!(
            slots[3],
            Value::Array(vec![Value::Array(vec![
                Value::from(2u64),
                Value::from(4u64),
            ])])
        );
        // feature rides as the (dtype, shape, ext-3) tensor tuple.
        let Value::Array(tensor) = &slots[4] else {
            panic!("expected tensor tuple, got {:?}", slots[4]);
        };
        assert_eq!(tensor[0].as_str(), Some("bfloat16"));
        assert!(matches!(&tensor[2], Value::Ext(3, _)));
        assert_eq!(slots[5], Value::Nil); // feature_shm
        let Value::Map(msd) = &slots[6] else {
            panic!("expected model_specific_data map");
        };
        assert_eq!(msd[0].0.as_str(), Some("vit_grid"));
        assert_eq!(slots[7], Value::Nil);
        assert_eq!(slots[8], Value::Nil);
        assert_eq!(slots[9], Value::Nil);
    }

    /// The request-level payload is an untagged 7-slot array with the three
    /// MRoPE slots and the reserved trailing slot nil.
    #[test]
    fn inputs_encode_as_untagged_seven_slot_array() {
        let inputs = TokenSpeedWireMmInputs {
            mm_items: vec![image_item()],
            im_token_id: Some(9),
            video_token_id: None,
        };
        let encoded = encode_msgpack(&inputs).unwrap();
        let Value::Array(slots) = decode_value(&encoded).unwrap() else {
            panic!("expected positional array");
        };
        assert_eq!(slots.len(), 7);
        assert!(matches!(&slots[0], Value::Array(items) if items.len() == 1));
        assert_eq!(slots[1], Value::from(9u32));
        for slot in &slots[2..7] {
            assert_eq!(*slot, Value::Nil);
        }
    }
}
