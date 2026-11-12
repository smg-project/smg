use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::Validate;

use super::{
    common::{
        default_true, deserialize_null_as_false, invalid, invalid_at, is_false, join_ids,
        CachePartition, GenerationRequest, InputIds,
    },
    sampling_params::SamplingParams,
};
use crate::validated::Normalizable;

// ============================================================================
// The native Generate API (one engine's own format)
// ============================================================================

/// A request to the native generate route: the input as text, token ids or
/// embeddings with its media, the model, the sampling parameters, and the
/// log-probability, hidden-state, session and adapter options.
#[derive(Clone, Debug, Serialize, Deserialize, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_generate_request"))]
pub struct GenerateRequest {
    /// Text input - the native format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(
        default = "super::common::default_unknown_model",
        skip_serializing_if = "super::common::is_unknown_model"
    )]
    pub model: String,

    /// Input IDs for tokenized input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_ids: Option<InputIds>,

    /// Input embeddings for direct embedding input
    /// Can be a 2D array (single request) or 3D array (batch of requests)
    /// Placeholder for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_embeds: Option<Value>,

    /// Image input data
    /// Can be an image instance, file name, URL, or base64 encoded string
    /// Supports single images, lists of images, or nested lists for batch processing
    /// Placeholder for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_data: Option<Value>,

    /// Video input data
    /// Can be a file name, URL, or base64 encoded string
    /// Supports single videos, lists of videos, or nested lists for batch processing
    /// Placeholder for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_data: Option<Value>,

    /// Audio input data
    /// Can be a file name, URL, or base64 encoded string
    /// Supports single audio files, lists of audio, or nested lists for batch processing
    /// Placeholder for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_data: Option<Value>,

    /// Sampling parameters (the native API's style)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(nested)]
    pub sampling_params: Option<SamplingParams>,

    /// Whether to return logprobs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_logprob: Option<bool>,

    /// If return logprobs, the start location in the prompt for returning logprobs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob_start_len: Option<i32>,

    /// If return logprobs, the number of top logprobs to return at each position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs_num: Option<i32>,

    /// If return logprobs, the token ids to return logprob for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids_logprob: Option<Vec<u32>>,

    /// Whether to detokenize tokens in text in the returned logprobs.
    #[serde(default)]
    pub return_text_in_logprobs: bool,

    /// Whether to stream the response
    #[serde(default, deserialize_with = "deserialize_null_as_false")]
    pub stream: bool,

    /// Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    #[serde(default = "default_true")]
    pub log_metrics: bool,

    /// Return model hidden states
    #[serde(default, skip_serializing_if = "is_false")]
    pub return_hidden_states: bool,

    /// The modalities of the image data [image, multi-images, video]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,

    /// Session parameters for continual prompting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_params: Option<HashMap<String, Value>>,

    /// Path to LoRA adapter(s) for model customization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<String>,

    /// LoRA adapter ID (if pre-loaded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_id: Option<String>,

    /// Custom logit processor for advanced sampling control. Must be a serialized instance
    /// of the engine's `CustomLogitProcessor` (srt/sampling/custom_logit_processor.py).
    /// Use the processor's `to_str()` method to generate the serialized string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_logit_processor: Option<String>,

    /// For disaggregated inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_host: Option<String>,

    /// For disaggregated inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<i32>,

    /// For disaggregated inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_room: Option<i32>,

    /// For disaggregated inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_pair_key: Option<String>,

    /// Data parallel rank routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_parallel_rank: Option<i32>,

    /// Background response
    #[serde(default)]
    pub background: bool,

    /// Conversation ID for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,

    /// Priority for the request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,

    /// Extra key for classifying the request (e.g. cache_salt)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_key: Option<String>,

    /// Whether to disallow logging for this request (e.g. due to ZDR)
    #[serde(default)]
    pub no_logs: bool,

    /// Custom metric labels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_labels: Option<HashMap<String, String>>,

    /// Whether to return bytes for image generation
    #[serde(default)]
    pub return_bytes: bool,

    /// Whether to return entropy
    #[serde(default)]
    pub return_entropy: bool,

    /// Request ID for tracking (inherited from BaseReq in Python)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rid: Option<String>,

    /// Additional fields not explicitly defined above (e.g. engine-specific parameters)
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Normalizable for GenerateRequest {
    // Use default no-op implementation - no normalization needed for GenerateRequest
}

/// Validation function for GenerateRequest: a body with no prompt at all is
/// refused, an empty `input_ids` is refused as the chat route refuses an
/// empty `messages`, and a negative id in `input_ids` is refused.
fn validate_generate_request(req: &GenerateRequest) -> Result<(), validator::ValidationError> {
    // A body with no prompt at all is refused. `input_embeds` is a prompt on
    // the paths that forward the body as-is, and a body with both `text` and
    // `input_ids` stays with the routers, which read `text` first today.
    if req.text.is_none() && req.input_ids.is_none() && req.input_embeds.is_none() {
        return Err(invalid(
            "missing_required_parameter",
            "One of 'text', 'input_ids' or 'input_embeds' is required.",
        ));
    }

    // An empty list of ids is no prompt, as an empty `messages` is no
    // conversation on the chat route: the same refusal.
    let no_ids = match &req.input_ids {
        Some(InputIds::Single(ids)) => ids.is_empty(),
        Some(InputIds::Batch(sequences)) => {
            sequences.is_empty() || sequences.iter().any(Vec::is_empty)
        }
        None => false,
    };
    if no_ids {
        return Err(invalid_at(
            "input_ids",
            "empty_array",
            "Invalid 'input_ids': empty array. Expected an array with minimum length 1, but got an empty array instead.",
        ));
    }

    // A negative id is no token. Refused here, both wires answer it the same
    // way at the boundary: the HTTP routers key the routing tokens with a
    // cast that would wrap it, and an engine refuses it in its own words one
    // hop later; the gRPC stage's own check becomes a second line.
    let negative = match &req.input_ids {
        Some(InputIds::Single(ids)) => ids.iter().copied().find(|&id| id < 0),
        Some(InputIds::Batch(sequences)) => sequences.iter().flatten().copied().find(|&id| id < 0),
        None => None,
    };
    if let Some(id) = negative {
        return Err(invalid_at(
            "input_ids",
            "integer_below_min_value",
            format!("Invalid 'input_ids': a token id must be non-negative, but got {id} instead."),
        ));
    }

    Ok(())
}

impl GenerateRequest {
    /// The choices the request asks for: the sampling `n`, one when absent or zero.
    pub fn choices(&self) -> u32 {
        self.sampling_params
            .as_ref()
            .and_then(|params| params.n)
            .filter(|&n| n > 0)
            .unwrap_or(1)
    }
}

impl GenerationRequest for GenerateRequest {
    fn rid(&self) -> Option<&str> {
        self.rid.as_deref()
    }

    fn is_stream(&self) -> bool {
        self.stream
    }

    fn cache_partition(&self) -> CachePartition<'_> {
        CachePartition {
            // Another engine's `cache_salt` is a passthrough extension on this
            // protocol; `extra_key` is the native API's typed classification key.
            cache_salt: self.other.get("cache_salt").and_then(Value::as_str),
            extra_key: self.extra_key.as_deref(),
            // Either name identifies the adapter the engine namespaces by.
            lora_path: self.lora_path.as_deref().or(self.lora_id.as_deref()),
        }
    }

    fn get_model(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    fn extract_text_for_routing(&self) -> String {
        // Check fields in priority order: text, input_ids
        if let Some(ref text) = self.text {
            return text.clone();
        }

        if let Some(ref input_ids) = self.input_ids {
            return match input_ids {
                InputIds::Single(ids) => join_ids(ids),
                InputIds::Batch(batches) => join_ids(batches.iter().flatten()),
            };
        }

        // No text input found
        String::new()
    }

    fn routing_tokens(&self) -> Option<&[i32]> {
        // Token ids win over text: they key routing on what the backend KV
        // cache keys on. Empty ids fall back to text.
        match &self.input_ids {
            Some(InputIds::Single(ids)) if !ids.is_empty() => Some(ids),
            // A batch is dispatched to a single worker; the first sequence is
            // the best available affinity signal.
            Some(InputIds::Batch(seqs)) => seqs
                .first()
                .map(Vec::as_slice)
                .filter(|ids| !ids.is_empty()),
            _ => None,
        }
    }
}

// ============================================================================
// Native Generate Response Types
// ============================================================================

/// Native generate response (single completion or array for n>1)
///
/// Format for n=1:
/// ```json
/// {
///   "text": "...",
///   "output_ids": [...],
///   "meta_info": { ... }
/// }
/// ```
///
/// Format for n>1:
/// ```json
/// [
///   {"text": "...", "output_ids": [...], "meta_info": {...}},
///   {"text": "...", "output_ids": [...], "meta_info": {...}}
/// ]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GenerateResponse {
    pub text: String,
    pub output_ids: Vec<u32>,
    pub meta_info: GenerateMetaInfo,
}

/// Metadata for a single generate completion
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GenerateMetaInfo {
    pub id: String,
    pub finish_reason: GenerateFinishReason,
    pub prompt_tokens: u32,
    pub weight_version: String,
    pub input_token_logprobs: Option<Vec<Vec<Option<f64>>>>,
    pub output_token_logprobs: Option<Vec<Vec<Option<f64>>>>,
    pub completion_tokens: u32,
    pub cached_tokens: u32,
    pub reasoning_tokens: Option<u32>,
    pub e2e_latency: f64,
    pub matched_stop: Option<Value>,
}

/// Finish reason for generate endpoint
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum GenerateFinishReason {
    Length {
        #[serde(rename = "type")]
        finish_type: GenerateFinishType,
        length: u32,
    },
    Stop {
        #[serde(rename = "type")]
        finish_type: GenerateFinishType,
    },
    Other(Value),
}

/// The kind of a generate finish reason as the wire names it: `length` (the
/// token cap reached) or `stop` (a stop condition met).
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum GenerateFinishType {
    Length,
    Stop,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> GenerateRequest {
        serde_json::from_value(serde_json::json!({"model": "m"})).expect("minimal request")
    }

    #[test]
    fn return_hidden_states_false_is_omitted_and_absent_reads_false() {
        let r = req();
        let v = serde_json::to_value(&r).expect("serialize");
        assert!(v.get("return_hidden_states").is_none());

        let back: GenerateRequest = serde_json::from_value(v).expect("roundtrip");
        assert!(!back.return_hidden_states);
    }

    #[test]
    fn return_hidden_states_true_round_trips() {
        let mut r = req();
        r.return_hidden_states = true;
        let v = serde_json::to_value(&r).expect("serialize");
        assert_eq!(v["return_hidden_states"], true);

        let back: GenerateRequest = serde_json::from_value(v).expect("roundtrip");
        assert!(back.return_hidden_states);
    }

    #[test]
    fn routing_tokens_from_single_input_ids() {
        let mut r = req();
        r.input_ids = Some(InputIds::Single(vec![1, 2, 3]));
        assert_eq!(r.routing_tokens(), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn a_negative_input_id_is_refused_single_or_batched() {
        let mut r = req();
        r.input_ids = Some(InputIds::Single(vec![1, -2, 3]));
        let err = validate_generate_request(&r).expect_err("a negative id is no token");
        assert_eq!(err.code, "integer_below_min_value");
        assert_eq!(
            err.params.get("field").map(ToString::to_string).as_deref(),
            Some("\"input_ids\"")
        );
        assert!(
            err.message.as_deref().unwrap_or("").contains("-2"),
            "the message names the offending id: {err:?}"
        );

        r.input_ids = Some(InputIds::Batch(vec![vec![1], vec![2, -3]]));
        assert!(validate_generate_request(&r).is_err());

        r.input_ids = Some(InputIds::Single(vec![1, 2, 3]));
        assert!(validate_generate_request(&r).is_ok());
    }

    #[test]
    fn routing_tokens_prefer_input_ids_over_text() {
        let mut r = req();
        r.text = Some("hello".to_string());
        r.input_ids = Some(InputIds::Single(vec![1, 2, 3]));
        assert_eq!(r.routing_tokens(), Some(&[1, 2, 3][..]));

        r.input_ids = Some(InputIds::Batch(vec![vec![4, 5], vec![6]]));
        assert_eq!(r.routing_tokens(), Some(&[4, 5][..]));
    }

    #[test]
    fn routing_tokens_none_for_empty_input_ids_with_text() {
        let mut r = req();
        r.text = Some("hello".to_string());
        r.input_ids = Some(InputIds::Single(vec![]));
        assert_eq!(r.routing_tokens(), None);
        assert_eq!(r.extract_text_for_routing(), "hello");
    }

    #[test]
    fn routing_tokens_from_batch_first_sequence() {
        let mut r = req();
        r.input_ids = Some(InputIds::Batch(vec![vec![1, 2], vec![3, 4]]));
        assert_eq!(r.routing_tokens(), Some(&[1, 2][..]));
    }

    #[test]
    fn routing_tokens_none_for_empty_inputs() {
        let mut r = req();
        r.input_ids = Some(InputIds::Batch(vec![]));
        assert_eq!(r.routing_tokens(), None);

        let mut r = req();
        r.input_ids = Some(InputIds::Batch(vec![vec![], vec![1]]));
        assert_eq!(r.routing_tokens(), None);

        let r = req();
        assert_eq!(r.routing_tokens(), None);
    }

    #[test]
    fn absent_model_reads_as_the_wildcard_and_is_not_serialized_back() {
        let request: GenerateRequest =
            serde_json::from_value(serde_json::json!({ "text": "hi" })).unwrap();
        assert_eq!(request.model, crate::UNKNOWN_MODEL_ID);
        let v = serde_json::to_value(&request).unwrap();
        assert!(
            v.get("model").is_none(),
            "wildcard placeholder serialized: {v}"
        );

        let explicit: GenerateRequest =
            serde_json::from_value(serde_json::json!({ "text": "hi", "model": "m" })).unwrap();
        let v = serde_json::to_value(&explicit).unwrap();
        assert_eq!(v["model"], "m");
    }

    #[test]
    fn cache_partition_uses_the_typed_extra_key_and_falls_back_to_lora_id() {
        let request: GenerateRequest = serde_json::from_value(serde_json::json!({
            "text": "hi",
            "extra_key": "k",
            "lora_id": "adapter-id",
            "cache_salt": "tenant-a"
        }))
        .unwrap();
        let partition = request.cache_partition();
        assert_eq!(partition.cache_salt, Some("tenant-a"));
        assert_eq!(partition.extra_key, Some("k"));
        assert_eq!(partition.lora_path, Some("adapter-id"));

        let typed_path: GenerateRequest = serde_json::from_value(serde_json::json!({
            "text": "hi",
            "lora_path": "adapter-path",
            "lora_id": "adapter-id"
        }))
        .unwrap();
        assert_eq!(typed_path.cache_partition().lora_path, Some("adapter-path"));
        assert!(typed_path.cache_partition().cache_salt.is_none());
    }

    /// The sampling `n` is the number of choices; absent and zero both mean one.
    #[test]
    fn a_generate_request_asks_for_at_least_one_choice() {
        assert_eq!(req().choices(), 1);
        let with_n = |n: u32| -> GenerateRequest {
            serde_json::from_value(serde_json::json!({"model": "m", "sampling_params": {"n": n}}))
                .expect("a generate request")
        };
        assert_eq!(with_n(0).choices(), 1);
        assert_eq!(with_n(4).choices(), 4);
    }

    /// The rules of `sampling_params` run on the request as the chat route's
    /// run on its fields, with the chat route's lower bound on `n`, and an
    /// empty `input_ids` is refused as the chat route refuses an empty
    /// `messages`; a request within the rules passes.
    #[test]
    fn a_generate_request_runs_the_rules_the_chat_route_runs() {
        use validator::ValidationErrorsKind;

        let request = |value: Value| -> GenerateRequest {
            serde_json::from_value(value).expect("a generate request")
        };
        let within = request(serde_json::json!({
            "model": "m",
            "text": "hi",
            "sampling_params": {"temperature": 0.5, "top_p": 0.9, "n": 1, "max_new_tokens": 4}
        }));
        assert!(
            within.validate().is_ok(),
            "a request within the rules passes"
        );

        for (field, value, code) in [
            ("temperature", serde_json::json!(-1), "range"),
            ("top_p", serde_json::json!(2), "decimal_above_max_value"),
            ("n", serde_json::json!(0), "range"),
        ] {
            let refused = request(serde_json::json!({
                "model": "m",
                "text": "hi",
                "sampling_params": {field: value, "max_new_tokens": 4}
            }));
            let errors = refused.validate().expect_err(field);
            let Some(ValidationErrorsKind::Struct(nested)) = errors.errors().get("sampling_params")
            else {
                panic!("{field}: the refusal is under sampling_params: {errors:?}");
            };
            let Some(ValidationErrorsKind::Field(found)) = nested.errors().get(field) else {
                panic!("{field}: the refusal names the field: {nested:?}");
            };
            assert_eq!(found[0].code, code, "{field}");
        }

        let empty = request(serde_json::json!({"model": "m", "input_ids": []}));
        let errors = empty.validate().expect_err("an empty input_ids is refused");
        let Some(ValidationErrorsKind::Field(found)) = errors.errors().get("__all__") else {
            panic!("the refusal is a rule over the request: {errors:?}");
        };
        assert_eq!(found[0].code, "empty_array");
        assert_eq!(found[0].params["field"], "input_ids");
    }

    /// The shapes the routers serve today keep passing the request rule: a
    /// body with both `text` and `input_ids`, and an `input_embeds`-only body.
    /// Only a body with no prompt at all is refused, and a cross-field
    /// sampling rule names its field.
    #[test]
    fn a_generate_request_is_refused_only_without_a_prompt() {
        use validator::ValidationErrorsKind;

        let request = |value: Value| -> GenerateRequest {
            serde_json::from_value(value).expect("a generate request")
        };
        for served in [
            serde_json::json!({"model": "m", "text": "hi", "input_ids": [1, 2]}),
            serde_json::json!({"model": "m", "input_embeds": [[0.1, 0.2]]}),
        ] {
            assert!(request(served.clone()).validate().is_ok(), "{served}");
        }

        let none =
            request(serde_json::json!({"model": "m", "sampling_params": {"max_new_tokens": 4}}));
        let errors = none.validate().expect_err("no prompt at all is refused");
        let Some(ValidationErrorsKind::Field(found)) = errors.errors().get("__all__") else {
            panic!("the refusal is a rule over the request: {errors:?}");
        };
        assert_eq!(found[0].code, "missing_required_parameter");
        assert_eq!(
            found[0].message.as_deref(),
            Some("One of 'text', 'input_ids' or 'input_embeds' is required.")
        );

        let crossed = request(serde_json::json!({
            "model": "m",
            "text": "hi",
            "sampling_params": {"min_new_tokens": 8, "max_new_tokens": 4}
        }));
        let errors = crossed.validate().expect_err("min above max is refused");
        let Some(ValidationErrorsKind::Struct(nested)) = errors.errors().get("sampling_params")
        else {
            panic!("the refusal is under sampling_params: {errors:?}");
        };
        let Some(ValidationErrorsKind::Field(found)) = nested.errors().get("__all__") else {
            panic!("the refusal is the cross-field rule: {nested:?}");
        };
        assert_eq!(found[0].code, "invalid_value");
        assert_eq!(found[0].params["field"], "min_new_tokens");
    }
}
