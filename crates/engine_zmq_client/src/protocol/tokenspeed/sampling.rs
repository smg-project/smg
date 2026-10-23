// TokenSpeed `SamplingParams` — a Python `msgspec.Struct(kw_only=True,
// array_like=True)` (runtime/sampling/sampling_params.py), so it rides the
// wire as an untagged positional msgpack array nested inside the tokenized
// request. **Field order is the wire contract** — append only, never reorder.

use std::collections::HashMap;

use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::codec::OpaqueValue;

/// TokenSpeed's "top_k is disabled" sentinel (sample from the whole vocab).
/// The engine rewrites the API convention `top_k = -1` to this value so
/// downstream kernels always see a positive cutoff.
pub const TOP_K_DISABLED: i32 = 1 << 30;

/// TokenSpeed's greedy-sampling threshold: a temperature below this collapses
/// to greedy (`temperature = 1.0`, `top_k = 1`) during normalization.
const SAMPLING_EPS: f64 = 1e-6;

/// Engine-facing sampling parameters for TokenSpeed text generation.
///
/// Mirrors the Python `SamplingParams` msgspec struct: all 29 fields, in
/// declaration order, encoded as one positional msgpack array. The engine
/// runs its `__post_init__` re-derivation only when `is_normalized` is false,
/// so senders that emit the normalized form (see [`Self::normalize`]) must
/// pre-resolve the derived fields — most notably the `top_k` sentinel.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct SamplingParams {
    /// Maximum number of tokens to generate. `None` means engine default.
    pub max_new_tokens: Option<u32>,
    /// API-input stop-string alias. SMG rejects stop strings upstream (the
    /// gateway does not enforce them on this wire), so it is always `None`.
    pub stop: Option<String>,
    /// Token IDs that stop generation (a Python set; encoded as an array).
    /// The normalized form uses `None` rather than an empty collection.
    pub stop_token_ids: Option<Vec<u32>>,
    /// Controls randomness. Below `SAMPLING_EPS` normalizes to greedy.
    pub temperature: f64,
    /// Cumulative probability threshold for nucleus sampling.
    pub top_p: f64,
    /// Maximum number of top tokens to consider. API convention `-1` means
    /// all tokens; the normalized form carries [`TOP_K_DISABLED`] instead.
    pub top_k: i32,
    /// Minimum probability threshold for token sampling.
    pub min_p: f64,
    /// Frequency penalty applied by the sampler.
    pub frequency_penalty: f64,
    /// Presence penalty applied by the sampler.
    pub presence_penalty: f64,
    /// Repetition penalty applied by the sampler.
    pub repetition_penalty: f64,
    /// Minimum number of tokens to generate before EOS / stop handling.
    pub min_new_tokens: u32,
    /// Structured-output JSON schema. Set from the request's `constraint`;
    /// at most one of these four structured-output fields is populated.
    pub json_schema: Option<String>,
    /// Structured-output regex. Set from the request's `constraint`.
    pub regex: Option<String>,
    /// Structured-output EBNF grammar. Set from the request's `constraint`.
    pub ebnf: Option<String>,
    /// Structured-output structural tag. Set from the request's `constraint`.
    pub structural_tag: Option<String>,
    /// Ignore the EOS token and keep generating until another stop condition.
    pub ignore_eos: bool,
    /// Whether detokenization skips special tokens (engine-side default true).
    pub skip_special_tokens: bool,
    /// Whether detokenization inserts spaces between special tokens.
    pub spaces_between_special_tokens: bool,
    /// Whether stop sequences are kept in the output text.
    pub no_stop_trim: bool,
    /// Token budget for thinking phases. Not set by SMG.
    pub thinking_budget: Option<u64>,
    /// Free-form engine extension parameters. Not set by SMG.
    pub custom_params: Option<OpaqueValue>,
    /// Streaming flush interval override. Not set by SMG.
    pub stream_interval: Option<u32>,
    /// Per-token logit bias, keyed by stringified token id. SMG rejects
    /// logit_bias upstream (no support on this backend).
    pub logit_bias: Option<HashMap<String, f64>>,
    /// Random seed. `None` lets the engine derive one from the rid so all
    /// TP/DP ranks agree.
    pub seed: Option<u64>,
    /// Output-logprob mode: `None` = off, `0` = the sampled token's logprob.
    /// The transport drives logprobs off the request's `return_logprob` flag,
    /// so SMG leaves this `None`.
    pub logprobs: Option<i32>,
    /// OpenAI-compat fanout count, stored but never acted on by the engine
    /// (the transport validates `n == 1`; SMG fans out n > 1 itself).
    pub n: u32,
    /// Normalized stop strings (always a list once normalized; `[]` here since
    /// SMG never sends stop strings).
    pub stop_strs: Vec<String>,
    /// Longest stop string in tokens, resolved by `normalize()`.
    pub stop_str_max_len: u32,
    /// True once the derived fields are resolved; the engine's decode-time
    /// `__post_init__` skips re-derivation when set.
    pub is_normalized: bool,
}

impl Default for SamplingParams {
    /// The Python class defaults (the API-input form, pre-normalization).
    fn default() -> Self {
        Self {
            max_new_tokens: None,
            stop: None,
            stop_token_ids: None,
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            min_new_tokens: 0,
            json_schema: None,
            regex: None,
            ebnf: None,
            structural_tag: None,
            ignore_eos: false,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            no_stop_trim: false,
            thinking_budget: None,
            custom_params: None,
            stream_interval: None,
            logit_bias: None,
            seed: None,
            logprobs: None,
            n: 1,
            stop_strs: Vec::new(),
            stop_str_max_len: 0,
            is_normalized: false,
        }
    }
}

impl SamplingParams {
    /// Resolve the derived fields the engine's `__post_init__` / `normalize()`
    /// would otherwise compute, and mark the struct normalized so the engine
    /// skips re-derivation on decode. Mirrors the Python semantics exactly:
    /// empty `stop_token_ids` collapses to `None`, a near-zero temperature
    /// collapses to greedy, and `top_k = -1` becomes [`TOP_K_DISABLED`].
    pub fn normalize(&mut self) {
        if self.is_normalized {
            return;
        }
        if self.stop_token_ids.as_ref().is_some_and(Vec::is_empty) {
            self.stop_token_ids = None;
        }
        if self.temperature < SAMPLING_EPS {
            self.temperature = 1.0;
            self.top_k = 1;
        }
        if self.top_k == -1 {
            self.top_k = TOP_K_DISABLED;
        }
        // SMG never sends stop strings, so the normalized stop_strs is always
        // the empty list with a zero max length.
        self.stop_strs = Vec::new();
        self.stop_str_max_len = 0;
        self.is_normalized = true;
    }
}

#[cfg(test)]
mod tests {
    use rmpv::Value;

    use super::*;
    use crate::codec::{decode_msgpack, decode_value, encode_msgpack};

    #[test]
    fn sampling_params_roundtrip() {
        let mut params = SamplingParams {
            max_new_tokens: Some(128),
            stop_token_ids: Some(vec![2, 3]),
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            min_p: 0.05,
            frequency_penalty: 0.1,
            presence_penalty: 0.2,
            repetition_penalty: 1.1,
            min_new_tokens: 1,
            ignore_eos: true,
            seed: Some(42),
            ..SamplingParams::default()
        };
        params.normalize();
        let encoded = encode_msgpack(&params).unwrap();
        assert_eq!(decode_msgpack::<SamplingParams>(&encoded).unwrap(), params);
    }

    #[test]
    fn sampling_params_serializes_as_29_element_array() {
        let mut params = SamplingParams {
            max_new_tokens: Some(16),
            stop_token_ids: Some(vec![7]),
            temperature: 0.5,
            ..SamplingParams::default()
        };
        params.normalize();
        let encoded = encode_msgpack(&params).unwrap();
        let value = decode_value(&encoded).unwrap();
        let array = match value {
            Value::Array(array) => array,
            other => panic!("expected array (msgspec array_like), got {other:?}"),
        };

        // 29-element positional array; spot-check the wire positions.
        assert_eq!(array.len(), 29);
        assert_eq!(array[0], Value::from(16)); // max_new_tokens
        assert_eq!(array[2], Value::Array(vec![Value::from(7)])); // stop_token_ids
        assert_eq!(array[3], Value::F64(0.5)); // temperature
        assert_eq!(array[5], Value::from(TOP_K_DISABLED)); // top_k, normalized
        assert_eq!(array[23], Value::Nil); // seed
        assert_eq!(array[26], Value::Array(vec![])); // stop_strs
        assert_eq!(array[28], Value::from(true)); // is_normalized
    }

    #[test]
    fn normalize_resolves_derived_fields() {
        // top_k = -1 (API "all tokens") becomes the engine sentinel.
        let mut params = SamplingParams::default();
        params.normalize();
        assert_eq!(params.top_k, TOP_K_DISABLED);
        assert!(params.is_normalized);

        // Near-zero temperature collapses to greedy.
        let mut params = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        params.normalize();
        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_k, 1);

        // Empty stop_token_ids collapses to None.
        let mut params = SamplingParams {
            stop_token_ids: Some(Vec::new()),
            ..SamplingParams::default()
        };
        params.normalize();
        assert_eq!(params.stop_token_ids, None);

        // Normalizing twice is a no-op (mirrors the engine's guard).
        let mut params = SamplingParams {
            top_k: 40,
            ..SamplingParams::default()
        };
        params.normalize();
        params.normalize();
        assert_eq!(params.top_k, 40);
    }
}
