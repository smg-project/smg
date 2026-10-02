// Ported from the Apache-2.0 reference `vllm-engine-core-client`
// (vllm-project/vllm): protocol/request.rs.

use std::collections::{BTreeMap, HashMap};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_default::DefaultFromSerde;
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::{
    codec::OpaqueValue,
    protocol::vllm::{
        lora, multimodal::MmFeatures, pooling::PoolingParams, sampling::EngineCoreSamplingParams,
    },
    Error, Result,
};

/// Request types are single-byte protocol constants sent as a raw ZMQ frame
/// (no encoding step). Mirrors Python `EngineCoreRequestType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EngineCoreRequestType {
    Add = 0,
    Abort = 1,
    StartDpWave = 2,
    Utility = 3,
}

impl EngineCoreRequestType {
    /// Decode the single-byte request-type frame. `None` for unrecognized values.
    pub fn from_frame(frame: &[u8]) -> Option<Self> {
        let [value] = frame else {
            return None;
        };
        match value {
            0 => Some(Self::Add),
            1 => Some(Self::Abort),
            2 => Some(Self::StartDpWave),
            3 => Some(Self::Utility),
            _ => None,
        }
    }

    /// Encode as the single-byte frame used on the engine input socket.
    pub fn to_frame(self) -> Bytes {
        Bytes::from_static(match self {
            Self::Add => b"\x00",
            Self::Abort => b"\x01",
            Self::StartDpWave => b"\x02",
            Self::Utility => b"\x03",
        })
    }
}

/// Extra kwargs consumed by engine-side reasoning parsers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningParserKwargs {
    /// Effective kwargs visible to the chat template for this request.
    pub chat_template_kwargs: HashMap<String, serde_json::Value>,
}

/// `mm_features` as this client sends it: built here from Router-preprocessed
/// tensors (`Typed`), or produced by vLLM's own input processor and relayed
/// as the msgpack its encoder wrote (`Raw`; tensors over vLLM's zero-copy
/// threshold then ride the request's aux frames, see
/// [`EngineCoreClient::submit_with_aux`](crate::connector::EngineCoreClient::submit_with_aux)).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MmFeaturesPayload {
    Typed(MmFeatures),
    Raw(OpaqueValue),
}

impl From<MmFeatures> for MmFeaturesPayload {
    fn from(features: MmFeatures) -> Self {
        Self::Typed(features)
    }
}

impl MmFeaturesPayload {
    /// The features when this client built them.
    pub fn typed(&self) -> Option<&MmFeatures> {
        match self {
            Self::Typed(features) => Some(features),
            Self::Raw(_) => None,
        }
    }
}

/// Engine-core add-request payload sent from frontend to engine.
///
/// This is a msgspec `array_like=True` struct: it serializes as a positional
/// msgpack array of exactly 21 elements. **Field order is the wire contract** —
/// do not reorder. Mirrors Python `EngineCoreRequest`.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple, DefaultFromSerde)]
pub struct EngineCoreRequest {
    pub request_id: String,
    pub prompt_token_ids: Option<Vec<u32>>,
    /// Multimodal features, one per input item, sorted by placeholder offset.
    pub mm_features: Option<MmFeaturesPayload>,
    pub sampling_params: Option<EngineCoreSamplingParams>,
    /// Pooling (embedding) parameters: set, with `sampling_params` unset, on
    /// a pooling request.
    pub pooling_params: Option<PoolingParams>,
    pub arrival_time: f64,
    #[serde(default)]
    pub lora_request: Option<lora::LoraRequest>,
    #[serde(default)]
    pub cache_salt: Option<String>,
    #[serde(default)]
    pub data_parallel_rank: Option<u32>,
    /// Unsupported in this client: Python uses a custom tensor/aux-frame path.
    #[serde(default)]
    pub prompt_embeds: Option<OpaqueValue>,
    /// Per-position mask for mixed-mode inputs. `Some(true)` = real token id;
    /// `Some(false)` = position uses a pre-computed `prompt_embeds` entry;
    /// `None` for pure-tokens and pure-embeds requests.
    #[serde(default)]
    pub prompt_is_token_ids: Option<Vec<bool>>,
    /// Client index, so outputs return to the same client when the frontend
    /// scales out.
    #[serde(default)]
    pub client_index: u32,
    /// In DP mode, the wave this request is expected to belong to.
    #[serde(default)]
    pub current_wave: u32,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub trace_headers: Option<BTreeMap<String, String>>,
    #[serde(default)]
    pub resumable: bool,
    /// Original user-provided request ID, used for output reporting and aborts.
    #[serde(default)]
    pub external_req_id: Option<String>,
    #[serde(default)]
    pub reasoning_ended: Option<bool>,
    /// Reasoning-parser kwargs forwarded to the structured-output backend.
    #[serde(default)]
    pub reasoning_parser_kwargs: Option<ReasoningParserKwargs>,
    /// If `true`, add to the waiting queue and immediately abort, so
    /// connector-side cleanup runs via the standard `request_finished` hook.
    #[serde(default)]
    pub abort_immediately: bool,
    /// Stable session identity shared by related requests.
    #[serde(default)]
    pub session_id: Option<String>,
}

impl EngineCoreRequest {
    /// Validate fields intentionally not supported in this client.
    pub fn validate(&self) -> Result<()> {
        if self.prompt_embeds.is_some() {
            return Err(Error::UnsupportedField {
                context: "EngineCoreRequest",
                field: "prompt_embeds",
            });
        }
        Ok(())
    }

    // NOTE: multimodal tensors are sent as inline ext-3 raw views in the
    // request frame — valid at any size (the engine's aux-frame split is an
    // encoder-side optimization only). Send-side aux extraction is a perf
    // follow-up.
}

/// A utility RPC to one engine (`EngineCoreRequestType::Utility`): the
/// positional tuple vLLM's client sends, `(client_index, call_id, method,
/// args)`. The engine runs `getattr(EngineCore, method)(*args)` and answers
/// with a [`UtilityOutput`](super::output::UtilityOutput) carrying the same
/// `call_id`.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct UtilityCall {
    /// Which of the frontend's output sockets gets the reply; this client has
    /// one.
    pub client_index: u32,
    /// Correlates the reply. vLLM reserves negative ids for notices an engine
    /// sends unprompted, so callers issue positive ones.
    pub call_id: i64,
    pub method: String,
    pub args: Vec<OpaqueValue>,
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::codec::{decode_msgpack, decode_value, encode_msgpack, hex};

    /// Golden bytes from vLLM's own encoder (`vllm.v1.serial_utils.MsgpackEncoder`,
    /// vLLM 0.30.1rc1): `encode((0, 0x0123456789ABCDEF, "reset_prefix_cache",
    /// (False, False)))`, the tuple `AsyncMPClient._call_utility_async` sends.
    #[test]
    fn utility_call_encodes_as_vllm_client_does() {
        let call = UtilityCall {
            client_index: 0,
            call_id: 0x0123_4567_89AB_CDEF,
            method: "reset_prefix_cache".to_string(),
            args: vec![Value::from(false), Value::from(false)],
        };
        let golden = hex("9400cf0123456789abcdefb272657365745f7072656669785f636163686592c2c2");
        assert_eq!(encode_msgpack(&call).unwrap(), golden);
        assert_eq!(decode_msgpack::<UtilityCall>(&golden).unwrap(), call);

        // A small id takes msgpack's positive fixint, as msgspec encodes it.
        let first = UtilityCall { call_id: 1, ..call };
        assert_eq!(
            encode_msgpack(&first).unwrap(),
            hex("940001b272657365745f7072656669785f636163686592c2c2")
        );
    }

    #[test]
    fn request_type_frames_roundtrip() {
        for ty in [
            EngineCoreRequestType::Add,
            EngineCoreRequestType::Abort,
            EngineCoreRequestType::StartDpWave,
            EngineCoreRequestType::Utility,
        ] {
            assert_eq!(EngineCoreRequestType::from_frame(&ty.to_frame()), Some(ty));
        }
        assert_eq!(EngineCoreRequestType::Add.to_frame().as_ref(), b"\x00");
        assert_eq!(EngineCoreRequestType::from_frame(b"\x09"), None);
        assert_eq!(EngineCoreRequestType::from_frame(b""), None);
    }

    #[test]
    fn engine_core_request_serializes_as_full_array() {
        let request = EngineCoreRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 8,
                ..EngineCoreSamplingParams::for_test()
            }),
            arrival_time: 1234.5,
            client_index: 7,
            ..EngineCoreRequest::default()
        };

        let encoded = encode_msgpack(&request).unwrap();
        let value = decode_value(&encoded).unwrap();
        let array = match value {
            Value::Array(array) => array,
            other => panic!("expected array, got {other:?}"),
        };

        // 21-element positional tuple; spot-check the wire positions.
        assert_eq!(array.len(), 21);
        assert_eq!(array[0], Value::from("req-1"));
        assert_eq!(array[2], Value::Nil); // mm_features
        assert_eq!(array[4], Value::Nil); // pooling_params
        assert_eq!(array[10], Value::Nil); // prompt_is_token_ids
        assert_eq!(array[11], Value::from(7)); // client_index
    }

    /// A pooling request carries typed `pooling_params` at position 4 and no
    /// sampling params; the nested array is byte-identical to vLLM's own
    /// `msgspec` encoding of the verified `PoolingParams(task="embed")`.
    #[test]
    fn pooling_request_nests_vllm_encoded_pooling_params() {
        let request = EngineCoreRequest {
            request_id: "emb-1".to_string(),
            prompt_token_ids: Some(vec![101, 2088, 102]),
            pooling_params: Some(PoolingParams::embed()),
            arrival_time: 1.0,
            ..EngineCoreRequest::default()
        };
        let encoded = encode_msgpack(&request).unwrap();
        let Value::Array(array) = decode_value(&encoded).unwrap() else {
            panic!("expected array");
        };
        assert_eq!(array[3], Value::Nil); // sampling_params
        let mut nested = Vec::new();
        rmpv::encode::write_value(&mut nested, &array[4]).unwrap();
        assert_eq!(
            nested,
            b"\x9a\xc3\xc0\xc0\xc0\xa5embed\xc2\xc2\xc0\xc0\x02".to_vec()
        );
        let decoded: EngineCoreRequest = decode_msgpack(&encoded).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn validate_rejects_prompt_embeds() {
        let request = EngineCoreRequest {
            prompt_embeds: Some(Value::from(1)),
            ..EngineCoreRequest::default()
        };
        assert!(request.validate().is_err());
        assert!(EngineCoreRequest::default().validate().is_ok());
    }
}
