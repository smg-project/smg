// SGLang `SamplingParams` — a Python `msgspec.Struct(kw_only=True,
// array_like=True)` (`srt/sampling/sampling_params.py`), so it rides the wire
// as an untagged positional msgpack array nested inside the tokenized
// request. **Field order is the wire contract.** This side sends the API
// parameters, the fields a caller sets, through `custom_params`: the
// scheduler's own fields after them (`stop_strs`, `stop_str_max_len`,
// `is_normalized`, ...) come out of its `normalize()`, and `msgspec` fills
// every trailing field an array leaves out from its default, so one encoding
// serves a release that inserts a field after `custom_params` (0.5.22 put
// `watermark` there) as well as the one before it. A release that moves a
// field inside the prefix is named at the handshake, see
// [`SglangProtocol::check_layouts`](super::SglangProtocol).

use std::collections::HashMap;

use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeTuple,
    Deserialize, Deserializer, Serialize, Serializer,
};

use crate::{
    codec::OpaqueValue,
    protocol::positional::{drain_trailing, next_field},
};

/// The API convention for "no top-k cutoff"; the scheduler resolves it.
pub const TOP_K_ALL: i32 = -1;

/// The Python class's name: the key its layout is declared under at the
/// handshake.
pub const SAMPLING_PARAMS_NAME: &str = "SamplingParams";

/// The fields of [`SamplingParams`] in wire order: the Python struct's API
/// parameters, through `custom_params`. What the encoder fills, and what the
/// handshake check compares with the field order the engine declares.
pub const WIRE_FIELDS: [&str; 26] = [
    "max_new_tokens",
    "stop",
    "stop_token_ids",
    "stop_regex",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "min_new_tokens",
    "n",
    "beam_width",
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
    "ignore_eos",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "no_stop_trim",
    "stream_interval",
    "logit_bias",
    "sampling_seed",
    "custom_params",
];

/// Engine-facing sampling parameters for SGLang text generation.
///
/// Mirrors the Python struct's API parameters in declaration order, the
/// 26-element prefix of its positional array ([`WIRE_FIELDS`]). SMG sends
/// the API-input form: the scheduler side (the SMG plugin) runs the struct's
/// `normalize()` and `verify()` on receipt, as the tokenizer manager would,
/// which derives the fields this side leaves out.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// Maximum number of tokens to generate. `None` lets the scheduler cap at
    /// the model's remaining context.
    pub max_new_tokens: Option<u32>,
    /// String stop sequences; the scheduler keeps its tokenizer and resolves
    /// them.
    pub stop: Option<Vec<String>>,
    /// Token ids that stop generation (a Python set; encoded as an array).
    pub stop_token_ids: Option<Vec<u32>>,
    /// Regex stop patterns. Not set by SMG.
    pub stop_regex: Option<Vec<String>>,
    pub temperature: f64,
    pub top_p: f64,
    /// `-1` means all tokens (the API convention the scheduler resolves).
    pub top_k: i32,
    pub min_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub repetition_penalty: f64,
    pub min_new_tokens: u32,
    /// Always `1` on the wire: SMG fans out `n > 1` itself and the plugin
    /// refuses anything else.
    pub n: u32,
    pub beam_width: Option<u32>,
    /// Structured output: at most one of the next four is set, from the
    /// request's `constraint`.
    pub json_schema: Option<String>,
    pub regex: Option<String>,
    pub ebnf: Option<String>,
    pub structural_tag: Option<String>,
    pub ignore_eos: bool,
    pub skip_special_tokens: bool,
    pub spaces_between_special_tokens: bool,
    pub no_stop_trim: bool,
    pub stream_interval: Option<u32>,
    /// Per-token logit bias keyed by stringified token id.
    pub logit_bias: Option<HashMap<String, f64>>,
    /// Random seed; `None` lets the scheduler pick.
    pub sampling_seed: Option<i64>,
    /// Free-form engine extension parameters. Not set by SMG.
    pub custom_params: Option<OpaqueValue>,
}

impl Default for SamplingParams {
    /// The Python class defaults (the API-input form).
    fn default() -> Self {
        Self {
            max_new_tokens: Some(128),
            stop: None,
            stop_token_ids: None,
            stop_regex: None,
            temperature: 1.0,
            top_p: 1.0,
            top_k: TOP_K_ALL,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            min_new_tokens: 0,
            n: 1,
            beam_width: None,
            json_schema: None,
            regex: None,
            ebnf: None,
            structural_tag: None,
            ignore_eos: false,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            no_stop_trim: false,
            stream_interval: None,
            logit_bias: None,
            sampling_seed: None,
            custom_params: None,
        }
    }
}

impl Serialize for SamplingParams {
    /// The [`WIRE_FIELDS`] prefix, in order: what `msgspec` itself encodes
    /// for the first 26 elements of the Python struct's array.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(WIRE_FIELDS.len())?;
        tuple.serialize_element(&self.max_new_tokens)?;
        tuple.serialize_element(&self.stop)?;
        tuple.serialize_element(&self.stop_token_ids)?;
        tuple.serialize_element(&self.stop_regex)?;
        tuple.serialize_element(&self.temperature)?;
        tuple.serialize_element(&self.top_p)?;
        tuple.serialize_element(&self.top_k)?;
        tuple.serialize_element(&self.min_p)?;
        tuple.serialize_element(&self.frequency_penalty)?;
        tuple.serialize_element(&self.presence_penalty)?;
        tuple.serialize_element(&self.repetition_penalty)?;
        tuple.serialize_element(&self.min_new_tokens)?;
        tuple.serialize_element(&self.n)?;
        tuple.serialize_element(&self.beam_width)?;
        tuple.serialize_element(&self.json_schema)?;
        tuple.serialize_element(&self.regex)?;
        tuple.serialize_element(&self.ebnf)?;
        tuple.serialize_element(&self.structural_tag)?;
        tuple.serialize_element(&self.ignore_eos)?;
        tuple.serialize_element(&self.skip_special_tokens)?;
        tuple.serialize_element(&self.spaces_between_special_tokens)?;
        tuple.serialize_element(&self.no_stop_trim)?;
        tuple.serialize_element(&self.stream_interval)?;
        tuple.serialize_element(&self.logit_bias)?;
        tuple.serialize_element(&self.sampling_seed)?;
        tuple.serialize_element(&self.custom_params)?;
        tuple.end()
    }
}

impl<'de> Deserialize<'de> for SamplingParams {
    /// The [`WIRE_FIELDS`] prefix, every element required; the scheduler's
    /// own fields after it, which a Python encoder's full array carries, are
    /// skipped.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ParamsVisitor;

        impl<'de> Visitor<'de> for ParamsVisitor {
            type Value = SamplingParams;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a positional SGLang SamplingParams array")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let params = SamplingParams {
                    max_new_tokens: next_field(&mut seq, "max_new_tokens")?,
                    stop: next_field(&mut seq, "stop")?,
                    stop_token_ids: next_field(&mut seq, "stop_token_ids")?,
                    stop_regex: next_field(&mut seq, "stop_regex")?,
                    temperature: next_field(&mut seq, "temperature")?,
                    top_p: next_field(&mut seq, "top_p")?,
                    top_k: next_field(&mut seq, "top_k")?,
                    min_p: next_field(&mut seq, "min_p")?,
                    frequency_penalty: next_field(&mut seq, "frequency_penalty")?,
                    presence_penalty: next_field(&mut seq, "presence_penalty")?,
                    repetition_penalty: next_field(&mut seq, "repetition_penalty")?,
                    min_new_tokens: next_field(&mut seq, "min_new_tokens")?,
                    n: next_field(&mut seq, "n")?,
                    beam_width: next_field(&mut seq, "beam_width")?,
                    json_schema: next_field(&mut seq, "json_schema")?,
                    regex: next_field(&mut seq, "regex")?,
                    ebnf: next_field(&mut seq, "ebnf")?,
                    structural_tag: next_field(&mut seq, "structural_tag")?,
                    ignore_eos: next_field(&mut seq, "ignore_eos")?,
                    skip_special_tokens: next_field(&mut seq, "skip_special_tokens")?,
                    spaces_between_special_tokens: next_field(
                        &mut seq,
                        "spaces_between_special_tokens",
                    )?,
                    no_stop_trim: next_field(&mut seq, "no_stop_trim")?,
                    stream_interval: next_field(&mut seq, "stream_interval")?,
                    logit_bias: next_field(&mut seq, "logit_bias")?,
                    sampling_seed: next_field(&mut seq, "sampling_seed")?,
                    custom_params: next_field(&mut seq, "custom_params")?,
                };
                drain_trailing(&mut seq)?;
                Ok(params)
            }
        }

        deserializer.deserialize_seq(ParamsVisitor)
    }
}

/// The field order of the Python struct in the releases the wire was recorded
/// against (`SamplingParams.__struct_fields__`), for the tests of this module
/// and of the handshake check.
#[cfg(test)]
pub(crate) mod recorded {
    /// SGLang 0.5.21: the API parameters, then the scheduler's own fields.
    pub(crate) const SAMPLING_FIELDS_0_5_21: [&str; 32] = [
        "max_new_tokens",
        "stop",
        "stop_token_ids",
        "stop_regex",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "min_new_tokens",
        "n",
        "beam_width",
        "json_schema",
        "regex",
        "ebnf",
        "structural_tag",
        "ignore_eos",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "no_stop_trim",
        "stream_interval",
        "logit_bias",
        "sampling_seed",
        "custom_params",
        "stop_strs",
        "stop_regex_strs",
        "stop_str_max_len",
        "stop_regex_max_len",
        "is_normalized",
        "ebnf_full_assistant",
    ];

    /// SGLang 0.5.22: `watermark` inserted after `custom_params`, which
    /// moved the scheduler's own fields by one.
    pub(crate) const SAMPLING_FIELDS_0_5_22: [&str; 33] = [
        "max_new_tokens",
        "stop",
        "stop_token_ids",
        "stop_regex",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "min_new_tokens",
        "n",
        "beam_width",
        "json_schema",
        "regex",
        "ebnf",
        "structural_tag",
        "ignore_eos",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "no_stop_trim",
        "stream_interval",
        "logit_bias",
        "sampling_seed",
        "custom_params",
        "watermark",
        "stop_strs",
        "stop_regex_strs",
        "stop_str_max_len",
        "stop_regex_max_len",
        "is_normalized",
        "ebnf_full_assistant",
    ];

    /// `msgspec.msgpack.encode(SamplingParams(max_new_tokens=8,
    /// stop_token_ids={7}, temperature=0.5, top_k=40, json_schema="{}",
    /// sampling_seed=3))` on each release: 32 and 33 elements.
    pub(crate) const PYTHON_RICH_0_5_21: &str = "dc002008c09107c0cb3fe0000000000000cb3ff000000000000028cb0000000000000000cb0000000000000000cb0000000000000000cb3ff00000000000000001c0a27b7dc0c0c0c2c3c3c2c0c003c0c0c00000c2c2";
    pub(crate) const PYTHON_RICH_0_5_22: &str = "dc002108c09107c0cb3fe0000000000000cb3ff000000000000028cb0000000000000000cb0000000000000000cb0000000000000000cb3ff00000000000000001c0a27b7dc0c0c0c2c3c3c2c0c003c0c0c0c00000c2c2";
    /// `msgspec.msgpack.encode` of the first 26 elements of either array
    /// above (the same bytes on both releases): what this side sends, and
    /// what each release's decoder turns back into the struct above.
    pub(crate) const PYTHON_RICH_PREFIX: &str = "dc001a08c09107c0cb3fe0000000000000cb3ff000000000000028cb0000000000000000cb0000000000000000cb0000000000000000cb3ff00000000000000001c0a27b7dc0c0c0c2c3c3c2c0c003c0";

    /// `msgspec.msgpack.encode(SamplingParams(max_new_tokens=8,
    /// temperature=0.0, top_k=-1, n=1))` on each release (its `__post_init__`
    /// collapsed the greedy request to temperature 1.0 / top_k 1 before
    /// encoding).
    pub(crate) const PYTHON_PINNED_0_5_21: &str = "dc002008c0c0c0cb3ff0000000000000cb3ff000000000000001cb0000000000000000cb0000000000000000cb0000000000000000cb3ff00000000000000001c0c0c0c0c0c2c3c3c2c0c0c0c0c0c00000c2c2";
    pub(crate) const PYTHON_PINNED_0_5_22: &str = "dc002108c0c0c0cb3ff0000000000000cb3ff000000000000001cb0000000000000000cb0000000000000000cb0000000000000000cb3ff00000000000000001c0c0c0c0c0c2c3c3c2c0c0c0c0c0c0c00000c2c2";
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::{recorded::*, *};
    use crate::codec::{decode_msgpack, encode_msgpack, hex};

    /// The struct the recorded "rich" arrays encode.
    fn rich() -> SamplingParams {
        SamplingParams {
            max_new_tokens: Some(8),
            stop_token_ids: Some(vec![7]),
            temperature: 0.5,
            top_k: 40,
            json_schema: Some("{}".into()),
            sampling_seed: Some(3),
            ..SamplingParams::default()
        }
    }

    #[test]
    fn the_wire_prefix_heads_the_field_order_of_both_releases() {
        assert!(SAMPLING_FIELDS_0_5_21.starts_with(&WIRE_FIELDS));
        assert!(SAMPLING_FIELDS_0_5_22.starts_with(&WIRE_FIELDS));
        assert_eq!(WIRE_FIELDS[WIRE_FIELDS.len() - 1], "custom_params");
        // The two releases part right after the prefix.
        assert_eq!(SAMPLING_FIELDS_0_5_21[WIRE_FIELDS.len()], "stop_strs");
        assert_eq!(SAMPLING_FIELDS_0_5_22[WIRE_FIELDS.len()], "watermark");
    }

    #[test]
    fn encodes_the_api_prefix_in_declaration_order() {
        let encoded = encode_msgpack(&rich()).unwrap();
        let value: Value = rmp_serde::from_slice(&encoded).unwrap();
        let arr = value.as_array().unwrap();
        assert_eq!(arr.len(), WIRE_FIELDS.len());
        assert_eq!(arr[0].as_u64(), Some(8), "max_new_tokens at 0");
        assert!(arr[1].is_nil(), "stop at 1");
        assert_eq!(
            arr[2].as_array().unwrap()[0].as_u64(),
            Some(7),
            "stop_token_ids at 2"
        );
        assert!(arr[3].is_nil(), "stop_regex at 3");
        assert_eq!(arr[4].as_f64(), Some(0.5), "temperature at 4");
        assert_eq!(arr[6].as_i64(), Some(40), "top_k at 6");
        assert_eq!(arr[12].as_u64(), Some(1), "n at 12");
        assert_eq!(arr[14].as_str(), Some("{}"), "json_schema at 14");
        assert_eq!(arr[19].as_bool(), Some(true), "skip_special_tokens at 19");
        assert_eq!(arr[24].as_i64(), Some(3), "sampling_seed at 24");
        assert!(arr[25].is_nil(), "custom_params at 25");
        // Byte for byte what msgspec encodes for the same 26 elements.
        assert_eq!(encoded, hex(PYTHON_RICH_PREFIX));
    }

    /// The prefix is the head of each release's own array: past the array
    /// header (three bytes for an array16), the bytes this side sends are
    /// the first bytes of what either release encodes.
    #[test]
    fn the_sent_prefix_is_the_head_of_either_releases_array() {
        let prefix = hex(PYTHON_RICH_PREFIX);
        for (release, full) in [
            ("0.5.21", PYTHON_RICH_0_5_21),
            ("0.5.22", PYTHON_RICH_0_5_22),
        ] {
            let full = hex(full);
            assert_eq!(&full[3..prefix.len()], &prefix[3..], "{release}");
        }
    }

    #[test]
    fn decodes_the_full_array_of_either_release_to_the_same_params() {
        for (release, full) in [
            ("0.5.21", PYTHON_RICH_0_5_21),
            ("0.5.22", PYTHON_RICH_0_5_22),
        ] {
            let decoded: SamplingParams = decode_msgpack(&hex(full)).unwrap();
            assert_eq!(decoded, rich(), "{release}");
            assert_eq!(
                encode_msgpack(&decoded).unwrap(),
                hex(PYTHON_RICH_PREFIX),
                "{release}"
            );
        }
    }

    #[test]
    fn python_pinned_params_round_trip() {
        for (release, pinned) in [
            ("0.5.21", PYTHON_PINNED_0_5_21),
            ("0.5.22", PYTHON_PINNED_0_5_22),
        ] {
            let decoded: SamplingParams = decode_msgpack(&hex(pinned)).unwrap();
            assert_eq!(decoded.max_new_tokens, Some(8), "{release}");
            assert_eq!(decoded.temperature, 1.0, "{release}");
            assert_eq!(decoded.top_k, 1, "{release}");
            assert_eq!(decoded.n, 1, "{release}");
            let again: SamplingParams = decode_msgpack(&encode_msgpack(&decoded).unwrap()).unwrap();
            assert_eq!(again, decoded, "{release}");
        }
    }

    #[test]
    fn a_prefix_short_of_the_api_parameters_is_refused() {
        let mut arr: Vec<Value> = rmp_serde::from_slice(&hex(PYTHON_RICH_PREFIX)).unwrap();
        arr.pop();
        let bytes = rmp_serde::to_vec(&Value::Array(arr)).unwrap();
        let error = decode_msgpack::<SamplingParams>(&bytes)
            .unwrap_err()
            .to_string();
        assert!(error.contains("custom_params"), "{error}");
    }
}
