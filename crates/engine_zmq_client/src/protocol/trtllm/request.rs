// The TensorRT-LLM dialect's request: a tagged positional array (the shape of
// a `msgspec.Struct(tag=True, array_like=True)` on the engine side) that the
// engine-side shim decodes into its own struct and turns into one LLM API
// `SamplingParams`. **Field order is the wire contract**: append only, never
// reorder. The shim declares both layouts at the handshake
// (`positional_layouts`) and the protocol compares the prefixes encoded here
// against them once per connection.

use std::collections::BTreeMap;

use bytes::Bytes;
use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::protocol::positional::{drain_trailing, expect_tag, next_field};

/// The tag of [`TrtllmGenerateReqInput`] (element 0 on the wire).
pub const GENERATE_REQ_TAG: &str = "TrtllmGenerateReqInput";
/// The tag of [`TrtllmSamplingParams`] (element 0 of the nested array).
pub const SAMPLING_PARAMS_TAG: &str = "TrtllmSamplingParams";

/// The field order of [`TrtllmGenerateReqInput`] this encoder fills (the tag,
/// element 0, is not a field), compared with the shim's declaration at the
/// handshake.
pub const GENERATE_WIRE_FIELDS: [&str; 10] = [
    "rid",
    "input_ids",
    "sampling",
    "streaming",
    "lora_task_id",
    "disaggregated",
    "image_frames",
    "cache_salt",
    "trace_headers",
    "arrival_time",
];

/// The field order of [`TrtllmSamplingParams`]: the LLM API's own keyword
/// names, so the shim builds its `SamplingParams(**fields)` without renaming.
pub const SAMPLING_WIRE_FIELDS: [&str; 30] = [
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "top_p_min",
    "top_p_reset_ids",
    "top_p_decay",
    "seed",
    "min_tokens",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "length_penalty",
    "early_stopping",
    "no_repeat_ngram_size",
    "beam_search_diversity_rate",
    "prompt_ignore_length",
    "stop",
    "stop_token_ids",
    "include_stop_str_in_output",
    "bad",
    "bad_token_ids",
    "ignore_eos",
    "embedding_bias",
    "logprobs",
    "prompt_logprobs",
    "exclude_input_from_output",
    "return_perf_metrics",
    "guided_decoding",
];

/// Request types: a single raw byte sent as its own ZMQ frame ahead of the
/// msgpack payload, so the shim dispatches without decoding first. The same
/// bytes as the other msgpack dialects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TrtllmRequestType {
    Add = 0,
    Abort = 1,
    /// Reserved: the shared control triple `(call_id, method, args)`. The
    /// engine's proto exposes no RPC that would send one, so nothing on this
    /// side frames it yet.
    Control = 2,
}

impl TrtllmRequestType {
    /// Decode the single-byte request-type frame. `None` for unrecognized values.
    pub fn from_frame(frame: &[u8]) -> Option<Self> {
        match frame {
            [0] => Some(Self::Add),
            [1] => Some(Self::Abort),
            [2] => Some(Self::Control),
            _ => None,
        }
    }

    /// Encode as the single-byte frame used on the engine input socket.
    pub fn to_frame(self) -> Bytes {
        Bytes::from_static(match self {
            Self::Add => b"\x00",
            Self::Abort => b"\x01",
            Self::Control => b"\x02",
        })
    }
}

/// The sampling block of a request, field for field the LLM API's
/// `SamplingParams` keywords (plus `max_tokens`, which the proto carries at
/// the request level). `None` leaves the engine's own default; the shim
/// passes only the fields that are set.
#[derive(Debug, Clone, PartialEq)]
pub struct TrtllmSamplingParams {
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub top_p_min: Option<f32>,
    pub top_p_reset_ids: Option<i32>,
    pub top_p_decay: Option<f32>,
    pub seed: Option<u64>,
    pub min_tokens: Option<u32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub length_penalty: Option<f32>,
    pub early_stopping: Option<i32>,
    pub no_repeat_ngram_size: Option<i32>,
    pub beam_search_diversity_rate: Option<f32>,
    pub prompt_ignore_length: Option<i32>,
    /// Stop strings: the engine tokenizes and matches them itself and reports
    /// the matched one back (`stop_reason`).
    pub stop: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub include_stop_str_in_output: bool,
    pub bad: Vec<String>,
    pub bad_token_ids: Vec<u32>,
    pub ignore_eos: bool,
    /// One bias per vocabulary entry when set.
    pub embedding_bias: Option<Vec<f32>>,
    /// `None`: no logprobs; `Some(0)`: the sampled token's; `Some(k)`: the
    /// `k` ranked candidates beside it.
    pub logprobs: Option<i32>,
    /// The same for the prompt's positions.
    pub prompt_logprobs: Option<i32>,
    pub exclude_input_from_output: bool,
    pub return_perf_metrics: bool,
    /// `(kind, guide)`: the kind is one of the LLM API's `GuidedDecodingParams`
    /// field names (`json_object`, `json_schema`, `regex`, `grammar`,
    /// `structural_tag`); the guide is the schema, pattern, grammar or tag
    /// text (empty for `json_object`).
    pub guided_decoding: Option<(String, String)>,
}

impl Default for TrtllmSamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 0,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            top_p_min: None,
            top_p_reset_ids: None,
            top_p_decay: None,
            seed: None,
            min_tokens: None,
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
            length_penalty: None,
            early_stopping: None,
            no_repeat_ngram_size: None,
            beam_search_diversity_rate: None,
            prompt_ignore_length: None,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            include_stop_str_in_output: false,
            bad: Vec::new(),
            bad_token_ids: Vec::new(),
            ignore_eos: false,
            embedding_bias: None,
            logprobs: None,
            prompt_logprobs: None,
            // The LLM API's own default; the proto's flag sets it explicitly.
            exclude_input_from_output: true,
            return_perf_metrics: false,
            guided_decoding: None,
        }
    }
}

impl Serialize for TrtllmSamplingParams {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(1 + SAMPLING_WIRE_FIELDS.len())?;
        tuple.serialize_element(SAMPLING_PARAMS_TAG)?;
        tuple.serialize_element(&self.max_tokens)?;
        tuple.serialize_element(&self.temperature)?;
        tuple.serialize_element(&self.top_p)?;
        tuple.serialize_element(&self.top_k)?;
        tuple.serialize_element(&self.min_p)?;
        tuple.serialize_element(&self.top_p_min)?;
        tuple.serialize_element(&self.top_p_reset_ids)?;
        tuple.serialize_element(&self.top_p_decay)?;
        tuple.serialize_element(&self.seed)?;
        tuple.serialize_element(&self.min_tokens)?;
        tuple.serialize_element(&self.repetition_penalty)?;
        tuple.serialize_element(&self.presence_penalty)?;
        tuple.serialize_element(&self.frequency_penalty)?;
        tuple.serialize_element(&self.length_penalty)?;
        tuple.serialize_element(&self.early_stopping)?;
        tuple.serialize_element(&self.no_repeat_ngram_size)?;
        tuple.serialize_element(&self.beam_search_diversity_rate)?;
        tuple.serialize_element(&self.prompt_ignore_length)?;
        tuple.serialize_element(&self.stop)?;
        tuple.serialize_element(&self.stop_token_ids)?;
        tuple.serialize_element(&self.include_stop_str_in_output)?;
        tuple.serialize_element(&self.bad)?;
        tuple.serialize_element(&self.bad_token_ids)?;
        tuple.serialize_element(&self.ignore_eos)?;
        tuple.serialize_element(&self.embedding_bias)?;
        tuple.serialize_element(&self.logprobs)?;
        tuple.serialize_element(&self.prompt_logprobs)?;
        tuple.serialize_element(&self.exclude_input_from_output)?;
        tuple.serialize_element(&self.return_perf_metrics)?;
        tuple.serialize_element(&self.guided_decoding)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for TrtllmSamplingParams {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ParamsVisitor;

        impl<'de> Visitor<'de> for ParamsVisitor {
            type Value = TrtllmSamplingParams;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged TrtllmSamplingParams positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                expect_tag(&mut seq, SAMPLING_PARAMS_TAG)?;
                let params = TrtllmSamplingParams {
                    max_tokens: next_field(&mut seq, "max_tokens")?,
                    temperature: next_field(&mut seq, "temperature")?,
                    top_p: next_field(&mut seq, "top_p")?,
                    top_k: next_field(&mut seq, "top_k")?,
                    min_p: next_field(&mut seq, "min_p")?,
                    top_p_min: next_field(&mut seq, "top_p_min")?,
                    top_p_reset_ids: next_field(&mut seq, "top_p_reset_ids")?,
                    top_p_decay: next_field(&mut seq, "top_p_decay")?,
                    seed: next_field(&mut seq, "seed")?,
                    min_tokens: next_field(&mut seq, "min_tokens")?,
                    repetition_penalty: next_field(&mut seq, "repetition_penalty")?,
                    presence_penalty: next_field(&mut seq, "presence_penalty")?,
                    frequency_penalty: next_field(&mut seq, "frequency_penalty")?,
                    length_penalty: next_field(&mut seq, "length_penalty")?,
                    early_stopping: next_field(&mut seq, "early_stopping")?,
                    no_repeat_ngram_size: next_field(&mut seq, "no_repeat_ngram_size")?,
                    beam_search_diversity_rate: next_field(&mut seq, "beam_search_diversity_rate")?,
                    prompt_ignore_length: next_field(&mut seq, "prompt_ignore_length")?,
                    stop: next_field(&mut seq, "stop")?,
                    stop_token_ids: next_field(&mut seq, "stop_token_ids")?,
                    include_stop_str_in_output: next_field(&mut seq, "include_stop_str_in_output")?,
                    bad: next_field(&mut seq, "bad")?,
                    bad_token_ids: next_field(&mut seq, "bad_token_ids")?,
                    ignore_eos: next_field(&mut seq, "ignore_eos")?,
                    embedding_bias: next_field(&mut seq, "embedding_bias")?,
                    logprobs: next_field(&mut seq, "logprobs")?,
                    prompt_logprobs: next_field(&mut seq, "prompt_logprobs")?,
                    exclude_input_from_output: next_field(&mut seq, "exclude_input_from_output")?,
                    return_perf_metrics: next_field(&mut seq, "return_perf_metrics")?,
                    guided_decoding: next_field(&mut seq, "guided_decoding")?,
                };
                drain_trailing(&mut seq)?;
                Ok(params)
            }
        }

        deserializer.deserialize_seq(ParamsVisitor)
    }
}

/// The disaggregated-serving parameters the proto carries, passed through
/// to the LLM API's `DisaggregatedParams` as the engine's Python servicer
/// passes them: the request type by name (`context_and_generation`,
/// `context_only`, `generation_only`), the context request id, the first
/// generated token id of the context phase. On the wire: a plain 3-element
/// array, or nil.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrtllmDisaggregated {
    pub request_type: String,
    pub ctx_request_id: Option<String>,
    pub first_gen_token_id: Option<u32>,
}

type DisaggregatedTuple = (String, Option<String>, Option<u32>);

/// The request this protocol sends the shim. The array carries `image_frames`,
/// the number of auxiliary frames that follow the payload (one raw image
/// each, in order: frame 1 is the first image); `images` holds those frames
/// and is not part of the array.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrtllmGenerateReqInput {
    /// Request id (the routing/registry key).
    pub rid: String,
    /// Pre-tokenized prompt token ids.
    pub input_ids: Vec<u32>,
    pub sampling: TrtllmSamplingParams,
    /// Whether the shim streams each iteration's output or the finish alone.
    pub streaming: bool,
    /// A LoRA adapter's task id, when the request carries one (the shim
    /// builds `LoRARequest(lora_name=f"lora_{task_id}", lora_int_id=task_id)`).
    pub lora_task_id: Option<i64>,
    pub disaggregated: Option<TrtllmDisaggregated>,
    /// How many auxiliary image frames follow the payload.
    pub image_frames: u32,
    pub cache_salt: Option<String>,
    /// The caller's W3C trace context, for the engine's own spans.
    pub trace_headers: Option<BTreeMap<String, String>>,
    /// The request's arrival time (Unix seconds), for the engine's metrics.
    pub arrival_time: Option<f64>,
    /// The raw images, one auxiliary frame each after the payload. Not part
    /// of the positional array; `image_frames` says how many there are.
    pub images: Vec<Bytes>,
}

impl Serialize for TrtllmGenerateReqInput {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(1 + GENERATE_WIRE_FIELDS.len())?;
        tuple.serialize_element(GENERATE_REQ_TAG)?;
        tuple.serialize_element(&self.rid)?;
        tuple.serialize_element(&self.input_ids)?;
        tuple.serialize_element(&self.sampling)?;
        tuple.serialize_element(&self.streaming)?;
        tuple.serialize_element(&self.lora_task_id)?;
        let disaggregated: Option<(&str, &Option<String>, Option<u32>)> =
            self.disaggregated.as_ref().map(|params| {
                (
                    params.request_type.as_str(),
                    &params.ctx_request_id,
                    params.first_gen_token_id,
                )
            });
        tuple.serialize_element(&disaggregated)?;
        tuple.serialize_element(&self.image_frames)?;
        tuple.serialize_element(&self.cache_salt)?;
        tuple.serialize_element(&self.trace_headers)?;
        tuple.serialize_element(&self.arrival_time)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for TrtllmGenerateReqInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ReqVisitor;

        impl<'de> Visitor<'de> for ReqVisitor {
            type Value = TrtllmGenerateReqInput;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a tagged TrtllmGenerateReqInput positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                expect_tag(&mut seq, GENERATE_REQ_TAG)?;
                let rid = next_field(&mut seq, "rid")?;
                let input_ids = next_field(&mut seq, "input_ids")?;
                let sampling = next_field(&mut seq, "sampling")?;
                let streaming = next_field(&mut seq, "streaming")?;
                let lora_task_id = next_field(&mut seq, "lora_task_id")?;
                let disaggregated: Option<DisaggregatedTuple> =
                    next_field(&mut seq, "disaggregated")?;
                let image_frames = next_field(&mut seq, "image_frames")?;
                let cache_salt = next_field(&mut seq, "cache_salt")?;
                let trace_headers = next_field(&mut seq, "trace_headers")?;
                let arrival_time = next_field(&mut seq, "arrival_time")?;
                drain_trailing(&mut seq)?;
                Ok(TrtllmGenerateReqInput {
                    rid,
                    input_ids,
                    sampling,
                    streaming,
                    lora_task_id,
                    disaggregated: disaggregated.map(
                        |(request_type, ctx_request_id, first_gen_token_id)| TrtllmDisaggregated {
                            request_type,
                            ctx_request_id,
                            first_gen_token_id,
                        },
                    ),
                    image_frames,
                    cache_salt,
                    trace_headers,
                    arrival_time,
                    // The frames themselves travel beside the payload; a
                    // decoder of the array alone sees their count.
                    images: Vec::new(),
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
    use crate::codec::{decode_msgpack, decode_value, encode_msgpack};

    /// A request with every field set, as the translation builds one (the
    /// floats exactly representable, so the shim's float64 and this side's
    /// float32 name the same value).
    pub(crate) fn full_request() -> TrtllmGenerateReqInput {
        TrtllmGenerateReqInput {
            rid: "vec-1".to_string(),
            input_ids: vec![1, 2, 3],
            sampling: TrtllmSamplingParams {
                max_tokens: 8,
                temperature: Some(0.5),
                top_p: Some(0.75),
                top_k: Some(40),
                seed: Some(42),
                repetition_penalty: Some(1.125),
                stop: vec!["\n\n".to_string()],
                stop_token_ids: vec![2],
                logprobs: Some(0),
                guided_decoding: Some(("json_schema".to_string(), "{}".to_string())),
                ..TrtllmSamplingParams::default()
            },
            streaming: true,
            lora_task_id: Some(7),
            disaggregated: Some(TrtllmDisaggregated {
                request_type: "context_only".to_string(),
                ctx_request_id: Some("ctx-1".to_string()),
                first_gen_token_id: None,
            }),
            image_frames: 1,
            cache_salt: Some("salt".to_string()),
            trace_headers: Some(BTreeMap::from([(
                "traceparent".to_string(),
                "00-abc-def-01".to_string(),
            )])),
            arrival_time: Some(1.5),
            images: vec![Bytes::from_static(b"\xff\xd8jpeg")],
        }
    }

    /// `msgspec.msgpack.encode` of the shim's struct for [`full_request`]
    /// (the shim's wire module builds the same request in its own test;
    /// msgspec writes every float as float64).
    const PYTHON_REQUEST_VECTOR: &str =
        "9bb65472746c6c6d47656e6572617465526571496e707574a57665632d3193010203dc001fb4\
         5472746c6c6d53616d706c696e67506172616d7308cb3fe0000000000000cb3fe80000000000\
         0028c0c0c0c02ac0cb3ff2000000000000c0c0c0c0c0c0c091a20a0a9102c29090c2c000c0c3\
         c292ab6a736f6e5f736368656d61a27b7dc30793ac636f6e746578745f6f6e6c79a5637478\
         2d31c001a473616c7481ab7472616365706172656e74ad30302d6162632d6465662d3031cb3f\
         f8000000000000";

    /// What this side's encoder writes for [`full_request`]: the bytes the
    /// shim's test decodes (this side writes its `f32` fields as float32).
    const RUST_REQUEST_VECTOR: &str =
        "9bb65472746c6c6d47656e6572617465526571496e707574a57665632d3193010203dc001fb4\
         5472746c6c6d53616d706c696e67506172616d7308ca3f000000ca3f40000028c0c0c0c02ac0\
         ca3f900000c0c0c0c0c0c0c091a20a0a9102c29090c2c000c0c3c292ab6a736f6e5f73636865\
         6d61a27b7dc30793ac636f6e746578745f6f6e6c79a56374782d31c001a473616c7481ab7472\
         616365706172656e74ad30302d6162632d6465662d3031cb3ff8000000000000";

    fn hex(s: &str) -> Vec<u8> {
        let hex: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    fn to_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// The array is the declared layout: the tag, then the ten fields; the
    /// nested sampling block the tag and its thirty.
    #[test]
    fn the_request_serializes_as_the_declared_positional_arrays() {
        let encoded = encode_msgpack(&full_request()).unwrap();
        let Value::Array(array) = decode_value(&encoded).unwrap() else {
            panic!("expected a positional array");
        };
        assert_eq!(array.len(), 1 + GENERATE_WIRE_FIELDS.len());
        assert_eq!(array[0], Value::from(GENERATE_REQ_TAG));
        assert_eq!(array[1], Value::from("vec-1"));
        let Value::Array(sampling) = &array[3] else {
            panic!("expected the sampling array at 3, got {:?}", array[3]);
        };
        assert_eq!(sampling.len(), 1 + SAMPLING_WIRE_FIELDS.len());
        assert_eq!(sampling[0], Value::from(SAMPLING_PARAMS_TAG));
        assert_eq!(sampling[1], Value::from(8u32)); // max_tokens
        assert_eq!(array[7], Value::from(1u32)); // image_frames
        assert_eq!(
            array[6],
            Value::Array(vec![
                Value::from("context_only"),
                Value::from("ctx-1"),
                Value::Nil
            ])
        );
    }

    /// The bytes the shim's msgspec encoder produces for the same request
    /// decode here to the same request (the images ride apart), and this
    /// side's own bytes are pinned for the shim's test.
    #[test]
    fn the_shims_bytes_decode_and_this_sides_bytes_are_pinned() {
        let request = full_request();
        let decoded: TrtllmGenerateReqInput = decode_msgpack(&hex(PYTHON_REQUEST_VECTOR)).unwrap();
        assert_eq!(
            decoded,
            TrtllmGenerateReqInput {
                images: Vec::new(),
                ..request.clone()
            }
        );
        let encoded = encode_msgpack(&request).unwrap();
        assert_eq!(
            to_hex(&encoded),
            RUST_REQUEST_VECTOR.replace(char::is_whitespace, "")
        );
        let round_trip: TrtllmGenerateReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(
            round_trip,
            TrtllmGenerateReqInput {
                images: Vec::new(),
                ..request
            }
        );
    }

    /// A defaulted request is the shortest honest form: nils for what is
    /// unset, empty lists, the LLM API's own defaults for the flags.
    #[test]
    fn a_default_request_round_trips_with_nils() {
        let request = TrtllmGenerateReqInput {
            rid: "r".to_string(),
            input_ids: vec![5],
            sampling: TrtllmSamplingParams {
                max_tokens: 1,
                ..TrtllmSamplingParams::default()
            },
            ..TrtllmGenerateReqInput::default()
        };
        let encoded = encode_msgpack(&request).unwrap();
        let decoded: TrtllmGenerateReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(decoded, request);
        assert!(decoded.sampling.exclude_input_from_output);
        assert!(decoded.sampling.guided_decoding.is_none());
    }

    /// A newer shim-side struct with appended fields still decodes here.
    #[test]
    fn appended_fields_are_skipped_on_decode() {
        let mut with_extra = match decode_value(&encode_msgpack(&full_request()).unwrap()).unwrap()
        {
            Value::Array(array) => array,
            other => panic!("expected array, got {other:?}"),
        };
        with_extra.push(Value::from("future"));
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &Value::Array(with_extra)).unwrap();
        let decoded: TrtllmGenerateReqInput = decode_msgpack(&bytes).unwrap();
        assert_eq!(decoded.rid, "vec-1");
    }

    #[test]
    fn decode_rejects_a_wrong_tag() {
        let mut bytes = encode_msgpack(&full_request()).unwrap();
        bytes[2] = b'X';
        let error = decode_msgpack::<TrtllmGenerateReqInput>(&bytes).unwrap_err();
        assert!(error.to_string().contains("wrong msgspec tag"), "{error}");
    }

    #[test]
    fn request_type_frames_match_the_wire_contract() {
        assert_eq!(TrtllmRequestType::Add.to_frame().as_ref(), b"\x00");
        assert_eq!(TrtllmRequestType::Abort.to_frame().as_ref(), b"\x01");
        assert_eq!(TrtllmRequestType::Control.to_frame().as_ref(), b"\x02");
        assert_eq!(
            TrtllmRequestType::from_frame(b"\x01"),
            Some(TrtllmRequestType::Abort)
        );
        assert_eq!(TrtllmRequestType::from_frame(b"\x09"), None);
    }
}
