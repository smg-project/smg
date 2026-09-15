// TokenSpeed tokenized generate request — the native `TokenizedGenerateReqInput`
// from `io_struct.py`, a tagged `msgspec.Struct(array_like=True)`: on the wire
// it is a positional msgpack array with the class-name tag string as element 0.
// **Field order is the wire contract** — do not reorder.

use bytes::Bytes;
use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::protocol::tokenspeed::{
    drain_trailing, expect_tag, multimodal::TokenSpeedWireMmInputs, next_field,
    sampling::SamplingParams,
};

/// The msgspec tag for [`TokenizedGenerateReqInput`] (element 0 on the wire).
pub const TOKENIZED_GENERATE_REQ_INPUT_TAG: &str = "TokenizedGenerateReqInput";

/// Request types are single-byte protocol constants sent as a raw ZMQ frame
/// ahead of the msgpack payload (TokenSpeed's `REQ_TYPE_ADD`/`REQ_TYPE_ABORT`),
/// so the receiver can dispatch without decoding first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TokenSpeedRequestType {
    Add = 0,
    Abort = 1,
}

impl TokenSpeedRequestType {
    /// Decode the single-byte request-type frame. `None` for unrecognized values.
    pub fn from_frame(frame: &[u8]) -> Option<Self> {
        let [value] = frame else {
            return None;
        };
        match value {
            0 => Some(Self::Add),
            1 => Some(Self::Abort),
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

/// TokenSpeed tokenized generate request sent from frontend to engine.
///
/// Models the leading prefix of the Python class, through `stream` — the
/// fields SMG sets, plus the neutral logprob carry-over values between them.
/// The encoder emits exactly this 11-element array (tag + 10 fields); the
/// engine's decoder fills every later field (`input_embeds`, `session_params`,
/// multimodal payloads, ...) from its defaults, since msgspec tolerates
/// missing trailing fields. The decoder here accepts full-length arrays and
/// skips the unmodeled trailing fields.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizedGenerateReqInput {
    /// Request id (the routing/registry key).
    pub rid: String,
    /// In-process HTTP-worker return address; unused on this transport.
    pub http_worker_ipc: Option<String>,
    /// Original prompt text. `None` on the token-id path (SMG detokenizes
    /// downstream of the engine, so only ids are sent).
    pub input_text: Option<String>,
    /// Pre-tokenized prompt token ids (SMG tokenizes upstream).
    pub input_ids: Vec<u32>,
    /// Sampling parameters (nested untagged positional array).
    pub sampling_params: SamplingParams,
    /// Whether to return the sampled token's logprob for this request.
    pub return_logprob: bool,
    /// Prompt-logprob start offset. Neutral `-1`: prompt logprobs are not
    /// supported on this wire.
    pub logprob_start_len: i32,
    /// Output top-k logprob count. Neutral `0`: only the sampled token's
    /// logprob is materialized.
    pub top_logprobs_num: u32,
    /// Token ids to report logprobs for. Neutral `None`: not supported.
    pub token_ids_logprob: Option<Vec<u32>>,
    /// Whether to stream outputs incrementally.
    pub stream: bool,
    /// Original tokenizer-valid prompt ids, pre pad-substitution (wire index
    /// 21). The engine detokenizes from these; the scheduler uses `input_ids`.
    /// Required whenever `multimodal_inputs` is set.
    pub input_ids_unpadded: Option<Vec<u32>>,
    /// Multimodal payload (wire index 22). When set, `input_ids` must already
    /// carry the pad-substituted placeholder ranges — the msgpack transport
    /// bypasses the engine's `InputProcessor`, which does that rewriting on
    /// other paths.
    pub multimodal_inputs: Option<TokenSpeedWireMmInputs>,
}

impl Default for TokenizedGenerateReqInput {
    fn default() -> Self {
        Self {
            rid: String::new(),
            http_worker_ipc: None,
            input_text: None,
            input_ids: Vec::new(),
            sampling_params: SamplingParams::default(),
            return_logprob: false,
            // Neutral logprob carry-over values, mirroring the engine's own
            // input processor.
            logprob_start_len: -1,
            top_logprobs_num: 0,
            token_ids_logprob: None,
            stream: false,
            input_ids_unpadded: None,
            multimodal_inputs: None,
        }
    }
}

impl Serialize for TokenizedGenerateReqInput {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        const NIL: Option<()> = None;
        // Text requests keep the historical 11-element prefix. Multimodal
        // requests must reach wire index 22 (`multimodal_inputs`), emitting
        // the engine's declared defaults for the unmodeled fields between
        // `stream` (10) and `input_ids_unpadded` (21); msgspec fills fields
        // after the last emitted element from defaults either way.
        let multimodal = self.multimodal_inputs.is_some() || self.input_ids_unpadded.is_some();
        let mut tuple = serializer.serialize_tuple(if multimodal { 23 } else { 11 })?;
        tuple.serialize_element(TOKENIZED_GENERATE_REQ_INPUT_TAG)?;
        tuple.serialize_element(&self.rid)?;
        tuple.serialize_element(&self.http_worker_ipc)?;
        tuple.serialize_element(&self.input_text)?;
        tuple.serialize_element(&self.input_ids)?;
        tuple.serialize_element(&self.sampling_params)?;
        tuple.serialize_element(&self.return_logprob)?;
        tuple.serialize_element(&self.logprob_start_len)?;
        tuple.serialize_element(&self.top_logprobs_num)?;
        tuple.serialize_element(&self.token_ids_logprob)?;
        tuple.serialize_element(&self.stream)?;
        if multimodal {
            tuple.serialize_element(&NIL)?; // input_embeds
            tuple.serialize_element(&NIL)?; // session_params
            tuple.serialize_element(&NIL)?; // custom_logit_processor
            tuple.serialize_element(&false)?; // return_hidden_states
            tuple.serialize_element(&0.0f64)?; // created_time
            tuple.serialize_element(&NIL)?; // bootstrap_host
            tuple.serialize_element(&NIL)?; // bootstrap_port
            tuple.serialize_element(&NIL)?; // bootstrap_room
            tuple.serialize_element(&NIL)?; // input_multi_ids
            tuple.serialize_element(&NIL)?; // input_extra_infos
            tuple.serialize_element(&self.input_ids_unpadded)?;
            tuple.serialize_element(&self.multimodal_inputs)?;
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
                write!(f, "a tagged TokenizedGenerateReqInput positional array")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                expect_tag(&mut seq, TOKENIZED_GENERATE_REQ_INPUT_TAG)?;
                let request = TokenizedGenerateReqInput {
                    rid: next_field(&mut seq, "rid")?,
                    http_worker_ipc: next_field(&mut seq, "http_worker_ipc")?,
                    input_text: next_field(&mut seq, "input_text")?,
                    input_ids: next_field(&mut seq, "input_ids")?,
                    sampling_params: next_field(&mut seq, "sampling_params")?,
                    return_logprob: next_field(&mut seq, "return_logprob")?,
                    logprob_start_len: next_field(&mut seq, "logprob_start_len")?,
                    top_logprobs_num: next_field(&mut seq, "top_logprobs_num")?,
                    token_ids_logprob: next_field(&mut seq, "token_ids_logprob")?,
                    stream: next_field(&mut seq, "stream")?,
                    // Send-only fields: the decoder models the text prefix and
                    // drains everything after `stream`.
                    input_ids_unpadded: None,
                    multimodal_inputs: None,
                };
                drain_trailing(&mut seq)?;
                Ok(request)
            }
        }

        deserializer.deserialize_seq(ReqVisitor)
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::{
        codec::{decode_msgpack, decode_value, encode_msgpack},
        protocol::tokenspeed::sampling::TOP_K_DISABLED,
    };

    /// A full-length (24-element) request array captured from the Python
    /// encoder: rid "vec-1", input_ids [1, 2, 3], normalized SamplingParams
    /// (temperature 0.5, top_p 0.9, top_k disabled, max_new_tokens 8,
    /// stop_token_ids {2}, seed 42), return_logprob, stream.
    const PYTHON_REQUEST_VECTOR: &str =
        "dc0018b9546f6b656e697a656447656e6572617465526571496e707574a57665632d31c0c0\
         93010203dc001d08c09102cb3fe0000000000000cb3feccccccccccccdce40000000cb0000\
         000000000000cb0000000000000000cb0000000000000000cb3ff000000000000000c0c0c0\
         c0c2c3c3c2c0c0c0c02ac0019000c3c3ff00c0c3c0c0c0c2cb0000000000000000c0c0c0c0\
         c0c0c0c0";

    fn python_request_bytes() -> Vec<u8> {
        let hex: String = PYTHON_REQUEST_VECTOR
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }

    /// The logical request behind [`PYTHON_REQUEST_VECTOR`], built through the
    /// same normalization path SMG uses.
    fn vector_request() -> TokenizedGenerateReqInput {
        let mut sampling_params = SamplingParams {
            max_new_tokens: Some(8),
            stop_token_ids: Some(vec![2]),
            temperature: 0.5,
            top_p: 0.9,
            seed: Some(42),
            ..SamplingParams::default()
        };
        sampling_params.normalize();
        TokenizedGenerateReqInput {
            rid: "vec-1".to_string(),
            input_ids: vec![1, 2, 3],
            sampling_params,
            return_logprob: true,
            stream: true,
            ..TokenizedGenerateReqInput::default()
        }
    }

    #[test]
    fn request_type_frames_roundtrip() {
        for ty in [TokenSpeedRequestType::Add, TokenSpeedRequestType::Abort] {
            assert_eq!(TokenSpeedRequestType::from_frame(&ty.to_frame()), Some(ty));
        }
        assert_eq!(TokenSpeedRequestType::Add.to_frame().as_ref(), b"\x00");
        assert_eq!(TokenSpeedRequestType::Abort.to_frame().as_ref(), b"\x01");
        assert_eq!(TokenSpeedRequestType::from_frame(b"\x09"), None);
        assert_eq!(TokenSpeedRequestType::from_frame(b""), None);
    }

    /// The pinned cross-language vector decodes field-for-field: the Python
    /// side emits all 24 elements, and the modeled 11-element prefix plus
    /// trailing-field skip must recover every field SMG cares about.
    #[test]
    fn python_request_vector_decodes() {
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&python_request_bytes()).unwrap();
        assert_eq!(decoded.rid, "vec-1");
        assert_eq!(decoded.http_worker_ipc, None);
        assert_eq!(decoded.input_text, None);
        assert_eq!(decoded.input_ids, vec![1, 2, 3]);
        assert!(decoded.return_logprob);
        assert_eq!(decoded.logprob_start_len, -1);
        assert_eq!(decoded.top_logprobs_num, 0);
        assert_eq!(decoded.token_ids_logprob, None);
        assert!(decoded.stream);

        let sp = &decoded.sampling_params;
        assert_eq!(sp.max_new_tokens, Some(8));
        assert_eq!(sp.stop_token_ids, Some(vec![2]));
        assert_eq!(sp.temperature, 0.5);
        assert_eq!(sp.top_p, 0.9);
        assert_eq!(sp.top_k, TOP_K_DISABLED);
        assert_eq!(sp.seed, Some(42));
        assert_eq!(sp.n, 1);
        assert_eq!(sp.stop_strs, Vec::<String>::new());
        assert!(sp.is_normalized);

        // Full struct equality against the same logical request built in Rust.
        assert_eq!(decoded, vector_request());
    }

    /// The encoder emits the shortest valid prefix: an 11-element array
    /// (tag + fields through `stream`) that decodes back to the same request
    /// the full-length Python vector decodes to. The nested sampling params
    /// are always sent in full (29 elements), since `is_normalized` — the
    /// last field — is always set.
    #[test]
    fn encoder_emits_tagged_prefix_through_stream() {
        let request = vector_request();
        let encoded = encode_msgpack(&request).unwrap();

        let Value::Array(array) = decode_value(&encoded).unwrap() else {
            panic!("expected positional array");
        };
        assert_eq!(array.len(), 11);
        assert_eq!(array[0], Value::from(TOKENIZED_GENERATE_REQ_INPUT_TAG));
        assert_eq!(array[1], Value::from("vec-1"));
        let Value::Array(sampling) = &array[5] else {
            panic!("expected nested sampling params array, got {:?}", array[5]);
        };
        assert_eq!(sampling.len(), 29);
        assert_eq!(array[10], Value::from(true)); // stream

        // Round-trip: the prefix encoding is semantically identical to the
        // full-length Python encoding.
        let roundtripped: TokenizedGenerateReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(roundtripped, request);
        let from_vector: TokenizedGenerateReqInput =
            decode_msgpack(&python_request_bytes()).unwrap();
        assert_eq!(roundtripped, from_vector);
    }

    /// A multimodal request must reach wire index 22 (`multimodal_inputs`),
    /// emitting the engine's declared defaults for the unmodeled fields in
    /// between; a text request keeps the 11-element prefix (previous test).
    #[test]
    fn mm_request_emits_wire_arity_through_multimodal_inputs() {
        use std::collections::BTreeMap;

        use crate::{
            codec::tensor::WireTensor,
            protocol::tokenspeed::multimodal::{
                mm_pad_value, TokenSpeedWireMmInputs, TokenSpeedWireMmItem, TokenSpeedWireModality,
            },
        };

        let pad = mm_pad_value(TokenSpeedWireModality::Image, 0xDEAD_BEEF);
        let mut model_specific_data = BTreeMap::new();
        model_specific_data.insert(
            "vit_grid".to_string(),
            WireTensor::from_raw(
                "uint32",
                vec![1, 3],
                vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
            ),
        );
        let mut request = vector_request();
        request.rid = "mm-1".to_string();
        request.input_ids = vec![10, pad, pad, pad, 50];
        request.input_ids_unpadded = Some(vec![10, 20, 30, 40, 50]);
        request.multimodal_inputs = Some(TokenSpeedWireMmInputs {
            mm_items: vec![TokenSpeedWireMmItem {
                modality: TokenSpeedWireModality::Image,
                hash: 0xDEAD_BEEF,
                pad_value: pad,
                offsets: vec![(1, 3)],
                feature: WireTensor::from_raw("bfloat16", vec![3, 2], (0u8..12).collect()),
                model_specific_data,
            }],
            im_token_id: Some(9),
            video_token_id: None,
        });

        let encoded = encode_msgpack(&request).unwrap();

        // Pinned cross-language vector: these exact bytes were decoded
        // field-for-field by TokenSpeed's own `MsgpackDecoder` (msgspec
        // 0.21.1, tokenspeed eaf66b5b) — including tensor dtypes/values and
        // `set_pad_value()` independently reproducing `pad_value`.
        const TS_MM_VECTOR: &str =
            "dc0017b9546f6b656e697a656447656e6572617465526571496e707574a46d6d2d31c0c0950\
             ace09811a46ce09811a46ce09811a4632dc001d08c09102cb3fe0000000000000cb3feccccc\
             cccccccdce40000000cb0000000000000000cb0000000000000000cb0000000000000000cb3\
             ff000000000000000c0c0c0c0c2c3c3c2c0c0c0c02ac0019000c3c3ff00c0c3c0c0c0c2cb00\
             00000000000000c0c0c0c0c0950a141e283297919a01cedeadbeefce09811a469192010393a\
             862666c6f61743136920302c70c03000102030405060708090a0bc081a87669745f67726964\
             93a675696e743332920103c70c03010000000200000003000000c0c0c009c0c0c0c0c0";
        let hex: String = encoded.iter().map(|b| format!("{b:02x}")).collect();
        let pinned: String = TS_MM_VECTOR
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        assert_eq!(hex, pinned);

        let Value::Array(array) = decode_value(&encoded).unwrap() else {
            panic!("expected positional array");
        };
        assert_eq!(array.len(), 23);
        assert_eq!(array[0], Value::from(TOKENIZED_GENERATE_REQ_INPUT_TAG));
        // Unmodeled fields between `stream` (10) and `input_ids_unpadded` (21)
        // carry the engine defaults.
        for nil_index in [11usize, 12, 13, 16, 17, 18, 19, 20] {
            assert_eq!(array[nil_index], Value::Nil, "index {nil_index}");
        }
        assert_eq!(array[14], Value::from(false)); // return_hidden_states
        assert_eq!(array[15], Value::from(0.0f64)); // created_time
        assert!(matches!(&array[21], Value::Array(ids) if ids.len() == 5));
        assert!(matches!(&array[22], Value::Array(mm) if mm.len() == 7));

        // SMG never receives requests: the decoder models the text prefix and
        // drains the multimodal tail.
        let decoded: TokenizedGenerateReqInput = decode_msgpack(&encoded).unwrap();
        assert_eq!(decoded.input_ids, request.input_ids);
        assert_eq!(decoded.input_ids_unpadded, None);
        assert_eq!(decoded.multimodal_inputs, None);
    }

    #[test]
    fn decode_rejects_wrong_tag() {
        let mut request_bytes = python_request_bytes();
        // Corrupt one tag byte: "TokenizedGenerateReqInput" -> "TOkenized...".
        request_bytes[5] = b'O';
        let error = decode_msgpack::<TokenizedGenerateReqInput>(&request_bytes).unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("wrong msgspec tag"), "{rendered}");
        assert!(rendered.contains("TokenizedGenerateReqInput"), "{rendered}");
    }

    #[test]
    fn decode_rejects_truncated_prefix() {
        // Missing modeled fields (here: everything after input_ids) must fail
        // loudly, not silently default.
        let truncated = Value::Array(vec![
            Value::from(TOKENIZED_GENERATE_REQ_INPUT_TAG),
            Value::from("r1"),
            Value::Nil,
            Value::Nil,
            Value::Array(vec![Value::from(1)]),
        ]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &truncated).unwrap();
        let error = decode_msgpack::<TokenizedGenerateReqInput>(&bytes).unwrap_err();
        assert!(
            error.to_string().contains("missing positional field"),
            "{error}"
        );
    }
}
