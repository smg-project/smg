// SGLang `SamplingParams` — a Python `msgspec.Struct(kw_only=True,
// array_like=True)` (`srt/sampling/sampling_params.py`), so it rides the wire
// as an untagged positional msgpack array nested inside the tokenized
// request. **Field order is the wire contract** — append only, never reorder.

use std::collections::HashMap;

use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::codec::OpaqueValue;

/// The API convention for "no top-k cutoff"; the scheduler resolves it.
pub const TOP_K_ALL: i32 = -1;

/// Engine-facing sampling parameters for SGLang text generation.
///
/// Mirrors the Python struct's 32 fields in declaration order. SMG sends the
/// API-input form: the scheduler side (the SMG plugin) runs the struct's
/// `normalize()` and `verify()` on receipt, as the tokenizer manager would, so
/// the derived fields (`stop_strs`, `stop_str_max_len`, `is_normalized`) are
/// sent at their declared defaults.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct SamplingParams {
    /// Maximum number of tokens to generate. `None` lets the scheduler cap at
    /// the model's remaining context.
    pub max_new_tokens: Option<u32>,
    /// String stop sequences. Never sent: the scheduler runs without a
    /// tokenizer, so SMG resolves them to token ids or trims them downstream.
    pub stop: Option<Vec<String>>,
    /// Token ids that stop generation (a Python set; encoded as an array).
    pub stop_token_ids: Option<Vec<u32>>,
    /// Regex stop patterns. Never sent (same reason as `stop`).
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
    /// Derived by the scheduler's `normalize()`; sent at its default.
    pub stop_strs: Option<Vec<String>>,
    pub stop_regex_strs: Option<Vec<String>>,
    pub stop_str_max_len: u32,
    pub stop_regex_max_len: u32,
    pub is_normalized: bool,
    pub ebnf_full_assistant: bool,
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
            stop_strs: None,
            stop_regex_strs: None,
            stop_str_max_len: 0,
            stop_regex_max_len: 0,
            is_normalized: false,
            ebnf_full_assistant: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::codec::{decode_msgpack, encode_msgpack};

    /// `msgspec.msgpack.encode(SamplingParams(max_new_tokens=8, temperature=0.0,
    /// top_k=-1, n=1))` from the pinned SGLang (its `__post_init__` collapsed
    /// the greedy request to temperature 1.0 / top_k 1 before encoding).
    const PYTHON_SAMPLING: &str = "dc002008c0c0c0cb3ff0000000000000cb3ff000000000000001cb0000000000000000cb0000000000000000cb0000000000000000cb3ff00000000000000001c0c0c0c0c0c2c3c3c2c0c0c0c0c0c00000c2c2";

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn encodes_thirty_two_positional_fields_in_declaration_order() {
        let params = SamplingParams {
            max_new_tokens: Some(8),
            stop_token_ids: Some(vec![7]),
            temperature: 0.5,
            top_k: 40,
            json_schema: Some("{}".into()),
            sampling_seed: Some(3),
            ..SamplingParams::default()
        };
        let value: Value = rmp_serde::from_slice(&encode_msgpack(&params).unwrap()).unwrap();
        let arr = value.as_array().unwrap();
        assert_eq!(arr.len(), 32);
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
        assert_eq!(arr[30].as_bool(), Some(false), "is_normalized at 30");
        assert_eq!(arr[31].as_bool(), Some(false), "ebnf_full_assistant at 31");
    }

    #[test]
    fn python_pinned_params_round_trip() {
        let decoded: SamplingParams = decode_msgpack(&hex(PYTHON_SAMPLING)).unwrap();
        assert_eq!(decoded.max_new_tokens, Some(8));
        assert_eq!(decoded.temperature, 1.0);
        assert_eq!(decoded.top_k, 1);
        assert_eq!(decoded.n, 1);
        assert!(!decoded.is_normalized);
        assert_eq!(encode_msgpack(&decoded).unwrap(), hex(PYTHON_SAMPLING));
    }
}
