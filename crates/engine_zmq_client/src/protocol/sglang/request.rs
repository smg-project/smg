// SGLang's tokenized generate request — the scheduler's own
// `TokenizedGenerateReqInput` from `io_struct.py`, a tagged
// `msgspec.Struct(array_like=True)`: on the wire a positional msgpack array
// with the class-name tag string as element 0. **Field order is the wire
// contract** — do not reorder.

use bytes::Bytes;
use serde::{
    de::{IgnoredAny, SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::protocol::{
    positional::{drain_trailing, expect_tag, next_field},
    sglang::sampling::SamplingParams,
};

/// The msgspec tag for [`TokenizedGenerateReqInput`] (element 0 on the wire).
pub const TOKENIZED_GENERATE_REQ_INPUT_TAG: &str = "TokenizedGenerateReqInput";

/// Request types: a single raw byte sent as its own ZMQ frame ahead of the
/// msgpack payload, so the plugin dispatches without decoding first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SglangRequestType {
    Add = 0,
    Abort = 1,
}

impl SglangRequestType {
    /// Decode the single-byte request-type frame. `None` for unrecognized values.
    pub fn from_frame(frame: &[u8]) -> Option<Self> {
        match frame {
            [0] => Some(Self::Add),
            [1] => Some(Self::Abort),
            _ => None,
        }
    }

    /// Encode as the single-byte frame used on the engine input socket.
    pub fn to_frame(self) -> Bytes {
        Bytes::from_static(match self {
            Self::Add => b"\x00",
            Self::Abort => b"\x01",
        })
    }
}

/// The request SMG sends an SGLang scheduler.
///
/// Models the Python class's prefix through `require_reasoning`: the base
/// `rid` / `http_worker_ipc`, the required `input_text`, `input_ids`,
/// `input_embeds`, `mm_inputs`, `token_type_ids`, `sampling_params`,
/// `return_logprob`, `logprob_start_len`, `top_logprobs_num`,
/// `token_ids_logprob`, `stream`, then the 19 defaulted fields up to
/// `require_reasoning` (session, LoRA, PD and routing slots SMG leaves at the
/// scheduler's defaults) so that flag lands in its slot. The encoder emits
/// the shortest valid prefix: 14 elements (tag + 13 fields) through `stream`,
/// or 34 when `require_reasoning` is set; every later field keeps its
/// `msgspec` default. Token ids go as a plain integer array; the plugin
/// widens the scheduler's `array('q')` decode hook to accept it.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizedGenerateReqInput {
    /// Request id (the routing/registry key).
    pub rid: String,
    /// Original prompt text. `None`: SMG sends token ids only.
    pub input_text: Option<String>,
    /// Pre-tokenized prompt token ids (SMG tokenizes upstream).
    pub input_ids: Vec<u32>,
    /// Sampling parameters (nested untagged positional array).
    pub sampling_params: SamplingParams,
    /// Whether to return the sampled token's logprob per output token.
    pub return_logprob: bool,
    /// Prompt-logprob start offset; `-1` is the API default (none).
    pub logprob_start_len: i32,
    /// Output top-k logprob count; `0` means only the sampled token's.
    pub top_logprobs_num: u32,
    /// Token ids to report logprobs for; `None` when not requested.
    pub token_ids_logprob: Option<Vec<u32>>,
    /// Whether to stream outputs incrementally.
    pub stream: bool,
    /// Hybrid-reasoning request: the scheduler tracks the reasoning phase and
    /// accounts its tokens separately.
    pub require_reasoning: bool,
}

impl Default for TokenizedGenerateReqInput {
    fn default() -> Self {
        Self {
            rid: String::new(),
            input_text: None,
            input_ids: Vec::new(),
            sampling_params: SamplingParams::default(),
            return_logprob: false,
            logprob_start_len: -1,
            top_logprobs_num: 0,
            token_ids_logprob: None,
            stream: false,
            require_reasoning: false,
        }
    }
}

impl Serialize for TokenizedGenerateReqInput {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        const NIL: Option<()> = None;
        // The shortest valid prefix: through `stream` unless the reasoning
        // flag has to reach its slot past the 19 defaulted fields.
        let len = if self.require_reasoning { 34 } else { 14 };
        let mut tuple = serializer.serialize_tuple(len)?;
        tuple.serialize_element(TOKENIZED_GENERATE_REQ_INPUT_TAG)?;
        tuple.serialize_element(&self.rid)?;
        tuple.serialize_element(&NIL)?; // http_worker_ipc
        tuple.serialize_element(&self.input_text)?;
        tuple.serialize_element(&self.input_ids)?;
        tuple.serialize_element(&NIL)?; // input_embeds
        tuple.serialize_element(&NIL)?; // mm_inputs
        tuple.serialize_element(&NIL)?; // token_type_ids
        tuple.serialize_element(&self.sampling_params)?;
        tuple.serialize_element(&self.return_logprob)?;
        tuple.serialize_element(&self.logprob_start_len)?;
        tuple.serialize_element(&self.top_logprobs_num)?;
        tuple.serialize_element(&self.token_ids_logprob)?;
        tuple.serialize_element(&self.stream)?;
        if self.require_reasoning {
            // The defaulted fields between `stream` and `require_reasoning`,
            // at the scheduler's own defaults: return_sampling_mask,
            // return_flat_raw_top_logprobs, return_hidden_states,
            // return_routed_experts, routed_experts_start_len,
            // return_indexer_topk, then session_id .. routing_key (13
            // optionals).
            for _ in 0..4 {
                tuple.serialize_element(&false)?;
            }
            tuple.serialize_element(&0u32)?;
            tuple.serialize_element(&false)?;
            for _ in 0..13 {
                tuple.serialize_element(&NIL)?;
            }
            tuple.serialize_element(&self.require_reasoning)?;
        }
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for TokenizedGenerateReqInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ReqVisitor;

        impl<'de> Visitor<'de> for ReqVisitor {
            type Value = TokenizedGenerateReqInput;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "a tagged SGLang TokenizedGenerateReqInput positional array"
                )
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                expect_tag(&mut seq, TOKENIZED_GENERATE_REQ_INPUT_TAG)?;
                let rid = next_field(&mut seq, "rid")?;
                let _http_worker_ipc: IgnoredAny = next_field(&mut seq, "http_worker_ipc")?;
                let input_text = next_field(&mut seq, "input_text")?;
                let input_ids = next_field(&mut seq, "input_ids")?;
                let _input_embeds: IgnoredAny = next_field(&mut seq, "input_embeds")?;
                let _mm_inputs: IgnoredAny = next_field(&mut seq, "mm_inputs")?;
                let _token_type_ids: IgnoredAny = next_field(&mut seq, "token_type_ids")?;
                let sampling_params = next_field(&mut seq, "sampling_params")?;
                let return_logprob = next_field(&mut seq, "return_logprob")?;
                let logprob_start_len = next_field(&mut seq, "logprob_start_len")?;
                let top_logprobs_num = next_field(&mut seq, "top_logprobs_num")?;
                let token_ids_logprob = next_field(&mut seq, "token_ids_logprob")?;
                let stream = next_field(&mut seq, "stream")?;
                // Skip the 19 defaulted fields up to `require_reasoning`; a
                // shorter array (a frontend that stopped at `stream`) means
                // the flag is off.
                let mut skipped = 0;
                while skipped < 19 && seq.next_element::<IgnoredAny>()?.is_some() {
                    skipped += 1;
                }
                let require_reasoning = if skipped == 19 {
                    seq.next_element::<Option<bool>>()?
                        .flatten()
                        .unwrap_or(false)
                } else {
                    false
                };
                drain_trailing(&mut seq)?;
                Ok(TokenizedGenerateReqInput {
                    rid,
                    input_text,
                    input_ids,
                    sampling_params,
                    return_logprob,
                    logprob_start_len,
                    top_logprobs_num,
                    token_ids_logprob,
                    stream,
                    require_reasoning,
                })
            }
        }

        deserializer.deserialize_seq(ReqVisitor)
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::codec::{decode_msgpack, encode_msgpack};

    /// `msgspec.msgpack.encode` of the positional request the SMG plugin's
    /// tests build: rid `r1`, ids `[1, 2, 3]`, `SamplingParams(max_new_tokens=8,
    /// temperature=0.0, top_k=-1, n=1)` (the engine's `__post_init__` already
    /// collapsed greedy to temperature 1.0 / top_k 1), `stream=True`,
    /// `require_reasoning=True`, the 19 defaulted fields between them at
    /// their defaults.
    const PYTHON_REQUEST: &str = "dc0022b9546f6b656e697a656447656e6572617465526571496e707574a27231c0c093010203c0c0c0dc002008c0c0c0cb3ff0000000000000cb3ff000000000000001cb0000000000000000cb0000000000000000cb0000000000000000cb3ff00000000000000001c0c0c0c0c0c2c3c3c2c0c0c0c0c0c00000c2c2c2ff00c0c3c2c2c2c200c2c0c0c0c0c0c0c0c0c0c0c0c0c0c3";

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn encodes_the_prefix_through_require_reasoning_with_the_tag_first() {
        let request = TokenizedGenerateReqInput {
            rid: "r1".into(),
            input_ids: vec![1, 2, 3],
            sampling_params: SamplingParams {
                max_new_tokens: Some(8),
                ..SamplingParams::default()
            },
            stream: true,
            require_reasoning: true,
            ..TokenizedGenerateReqInput::default()
        };
        let value: Value = rmp_serde::from_slice(&encode_msgpack(&request).unwrap()).unwrap();
        let arr = value.as_array().unwrap();
        assert_eq!(arr.len(), 34);
        assert_eq!(arr[0].as_str(), Some(TOKENIZED_GENERATE_REQ_INPUT_TAG));
        assert_eq!(arr[1].as_str(), Some("r1"));
        assert!(arr[2].is_nil(), "http_worker_ipc at 2");
        assert!(arr[3].is_nil(), "input_text at 3");
        assert_eq!(arr[4].as_array().unwrap().len(), 3, "input_ids at 4");
        assert!(arr[5].is_nil() && arr[6].is_nil() && arr[7].is_nil());
        assert_eq!(arr[8].as_array().unwrap().len(), 32, "sampling_params at 8");
        assert_eq!(arr[9].as_bool(), Some(false), "return_logprob at 9");
        assert_eq!(arr[10].as_i64(), Some(-1), "logprob_start_len at 10");
        assert_eq!(arr[11].as_u64(), Some(0), "top_logprobs_num at 11");
        assert!(arr[12].is_nil(), "token_ids_logprob at 12");
        assert_eq!(arr[13].as_bool(), Some(true), "stream at 13");
        for (i, slot) in arr[14..18].iter().enumerate() {
            assert_eq!(slot.as_bool(), Some(false), "return_* flag at {}", 14 + i);
        }
        assert_eq!(arr[18].as_u64(), Some(0), "routed_experts_start_len at 18");
        assert_eq!(arr[19].as_bool(), Some(false), "return_indexer_topk at 19");
        assert!(
            arr[20..33].iter().all(Value::is_nil),
            "session..routing_key nil"
        );
        assert_eq!(arr[33].as_bool(), Some(true), "require_reasoning at 33");
    }

    #[test]
    fn a_request_without_reasoning_stops_at_stream() {
        let request = TokenizedGenerateReqInput {
            rid: "r1".into(),
            input_ids: vec![1],
            stream: true,
            ..TokenizedGenerateReqInput::default()
        };
        let value: Value = rmp_serde::from_slice(&encode_msgpack(&request).unwrap()).unwrap();
        assert_eq!(value.as_array().unwrap().len(), 14);
        let decoded: TokenizedGenerateReqInput =
            decode_msgpack(&encode_msgpack(&request).unwrap()).unwrap();
        assert_eq!(decoded, request);
    }

    #[test]
    fn a_prefix_that_stops_at_stream_decodes_with_require_reasoning_off() {
        let mut arr: Vec<Value> = rmp_serde::from_slice(&hex(PYTHON_REQUEST)).unwrap();
        arr.truncate(14);
        let bytes = rmp_serde::to_vec(&Value::Array(arr)).unwrap();
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&bytes).unwrap();
        assert!(decoded.stream && !decoded.require_reasoning);
    }

    #[test]
    fn python_pinned_request_round_trips() {
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&hex(PYTHON_REQUEST)).unwrap();
        assert_eq!(decoded.rid, "r1");
        assert_eq!(decoded.input_ids, vec![1, 2, 3]);
        assert_eq!(decoded.sampling_params.max_new_tokens, Some(8));
        assert_eq!(decoded.sampling_params.top_k, 1);
        assert!(decoded.stream && decoded.require_reasoning);
        // Byte-for-byte: what SMG encodes is what the Python struct encodes.
        assert_eq!(encode_msgpack(&decoded).unwrap(), hex(PYTHON_REQUEST));
    }

    #[test]
    fn decoder_tolerates_the_full_length_python_array() {
        let mut arr: Vec<Value> = rmp_serde::from_slice(&hex(PYTHON_REQUEST)).unwrap();
        for _ in 0..35 {
            arr.push(Value::Nil); // the defaulted tail the scheduler declares
        }
        let bytes = rmp_serde::to_vec(&Value::Array(arr)).unwrap();
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&bytes).unwrap();
        assert_eq!(decoded.rid, "r1");
    }

    #[test]
    fn request_type_frames() {
        assert_eq!(
            SglangRequestType::from_frame(b"\x00"),
            Some(SglangRequestType::Add)
        );
        assert_eq!(
            SglangRequestType::from_frame(b"\x01"),
            Some(SglangRequestType::Abort)
        );
        assert_eq!(SglangRequestType::from_frame(b"\x02"), None);
    }
}
