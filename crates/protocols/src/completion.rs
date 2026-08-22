use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::Validate;

use super::{
    common::*,
    sampling_params::{validate_top_k_value, validate_top_p_value},
};
use crate::validated::Normalizable;

// ============================================================================
// Completions API (v1/completions) - DEPRECATED but still supported
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_completion_cross_parameters"))]
pub struct CompletionRequest {
    /// ID of the model to use (required for OpenAI, optional for some implementations, such as SGLang)
    pub model: String,

    /// The prompt(s) to generate completions for
    #[validate(custom(function = "validate_completion_prompt"))]
    pub prompt: StringOrArray,

    /// Generates `best_of` completions server-side and returns the "best"
    #[validate(range(min = 0, max = 20))]
    pub best_of: Option<u32>,

    /// Echo back the prompt in addition to the completion
    #[serde(default)]
    pub echo: bool,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,

    /// Modify the likelihood of specified tokens appearing in the completion
    pub logit_bias: Option<HashMap<String, f32>>,

    /// Include the log probabilities on the `logprobs` most likely tokens
    #[validate(range(min = 0, max = 5))]
    pub logprobs: Option<u32>,

    /// The maximum number of tokens to generate
    #[validate(range(min = 0))]
    pub max_tokens: Option<u32>,

    /// How many completions to generate for each prompt
    #[validate(range(min = 1, max = 128))]
    pub n: Option<u32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,

    /// If specified, our system will make a best effort to sample deterministically
    pub seed: Option<i64>,

    /// Up to 4 sequences where the API will stop generating further tokens
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,

    /// Whether to stream back partial progress
    #[serde(default, deserialize_with = "deserialize_null_as_false")]
    pub stream: bool,

    /// Options for streaming response
    pub stream_options: Option<StreamOptions>,

    /// The suffix that comes after a completion of inserted text
    pub suffix: Option<String>,

    /// What sampling temperature to use, between 0 and 2
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,

    /// An alternative to sampling with temperature (nucleus sampling)
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,

    /// A unique identifier representing your end-user
    pub user: Option<String>,

    // =============================================================================
    // Engine-Specific Sampling Parameters
    // =============================================================================
    /// Top-k sampling parameter (-1 to disable)
    #[validate(custom(function = "validate_top_k_value"))]
    pub top_k: Option<i32>,

    /// Min-p nucleus sampling parameter
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: Option<f32>,

    /// Minimum number of tokens to generate
    #[validate(range(min = 0))]
    pub min_tokens: Option<u32>,

    /// Repetition penalty for reducing repetitive text
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f32>,

    /// Regex constraint for output generation
    pub regex: Option<String>,

    /// EBNF grammar constraint for structured output
    pub ebnf: Option<String>,

    /// JSON schema constraint for structured output
    pub json_schema: Option<String>,

    /// Specific token IDs to use as stop conditions
    pub stop_token_ids: Option<Vec<u32>>,

    /// Skip trimming stop tokens from output
    #[serde(default, skip_serializing_if = "is_false")]
    pub no_stop_trim: bool,

    /// Ignore end-of-sequence tokens during generation
    #[serde(default, skip_serializing_if = "is_false")]
    pub ignore_eos: bool,

    /// Skip special tokens during detokenization
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    /// Path to LoRA adapter(s) for model customization
    pub lora_path: Option<String>,

    /// Session parameters for continual prompting
    pub session_params: Option<HashMap<String, Value>>,

    /// Return model hidden states
    #[serde(default, skip_serializing_if = "is_false")]
    pub return_hidden_states: bool,

    /// Sampling seed for deterministic outputs
    pub sampling_seed: Option<u64>,

    /// Request ID forwarded to the backend for log correlation (SGLang extension)
    pub rid: Option<String>,

    /// Additional fields including bootstrap info for PD routing
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Normalizable for CompletionRequest {}

fn validate_completion_prompt(prompt: &StringOrArray) -> Result<(), validator::ValidationError> {
    match prompt {
        StringOrArray::String(_) => {}
        StringOrArray::Array(arr) => {
            if arr.is_empty() {
                let mut error = validator::ValidationError::new("prompt_empty");
                error.message = Some("prompt array cannot be empty".into());
                return Err(error);
            }
        }
    }

    Ok(())
}

fn validate_completion_cross_parameters(
    req: &CompletionRequest,
) -> Result<(), validator::ValidationError> {
    if req.stream_options.is_some() && !req.stream {
        let mut error = validator::ValidationError::new("stream_options_requires_stream");
        error.message =
            Some("The 'stream_options' parameter is only allowed when 'stream' is enabled".into());
        return Err(error);
    }

    if let (Some(min), Some(max)) = (req.min_tokens, req.max_tokens) {
        if min > max {
            let mut error = validator::ValidationError::new("min_tokens_exceeds_max");
            error.message = Some("min_tokens cannot exceed max_tokens".into());
            return Err(error);
        }
    }

    let constraint_count =
        req.regex.is_some() as u8 + req.ebnf.is_some() as u8 + req.json_schema.is_some() as u8;
    if constraint_count > 1 {
        let mut error = validator::ValidationError::new("multiple_constraints");
        error.message = Some(
            "only one structured output constraint (regex, ebnf, or json_schema) can be active at a time"
                .into(),
        );
        return Err(error);
    }

    if let (Some(best_of), Some(n)) = (req.best_of, req.n) {
        if best_of <= n {
            let mut error = validator::ValidationError::new("best_of_less_than_n");
            error.message = Some("best_of must be greater than n".into());
            return Err(error);
        }
    }

    if req.stream && req.best_of.is_some() {
        let mut error = validator::ValidationError::new("best_of_not_supported_with_stream");
        error.message = Some("best_of is not supported when stream is enabled".into());
        return Err(error);
    }

    Ok(())
}

impl GenerationRequest for CompletionRequest {
    fn rid(&self) -> Option<&str> {
        self.rid.as_deref()
    }

    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        match &self.prompt {
            StringOrArray::String(s) => s.clone(),
            StringOrArray::Array(v) => v.join(" "),
        }
    }
}

// ============================================================================
// Response Types
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String, // "text_completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Option<Usage>,
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>, // "stop", "length", "content_filter", etc.
    /// Information about which stop condition was matched
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<Value>, // Can be string or integer
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct CompletionStreamResponse {
    pub id: String,
    pub object: String, // "text_completion"
    pub created: u64,
    pub choices: Vec<CompletionStreamChoice>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize, Serialize, schemars::JsonSchema)]
pub struct CompletionStreamChoice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(mut extra: Value) -> CompletionRequest {
        let mut value = serde_json::json!({"model": "m", "prompt": "hello"});
        value
            .as_object_mut()
            .expect("object")
            .append(extra.as_object_mut().expect("object"));
        serde_json::from_value(value).expect("request must deserialize")
    }

    #[test]
    fn default_sglang_flags_are_omitted_and_absent_reads_defaults() {
        let value = serde_json::to_value(req(serde_json::json!({}))).expect("serialize");
        for field in ["no_stop_trim", "ignore_eos", "return_hidden_states"] {
            assert!(value.get(field).is_none(), "{field} serialized at default");
        }

        let back: CompletionRequest = serde_json::from_value(value).expect("roundtrip");
        assert!(!back.no_stop_trim);
        assert!(!back.ignore_eos);
        assert!(!back.return_hidden_states);
    }

    #[test]
    fn non_default_sglang_flags_round_trip() {
        let value = serde_json::to_value(req(serde_json::json!({
            "no_stop_trim": true,
            "ignore_eos": true,
            "return_hidden_states": true
        })))
        .expect("serialize");
        assert_eq!(value["no_stop_trim"], true);
        assert_eq!(value["ignore_eos"], true);
        assert_eq!(value["return_hidden_states"], true);

        let back: CompletionRequest = serde_json::from_value(value).expect("roundtrip");
        assert!(back.no_stop_trim);
        assert!(back.ignore_eos);
        assert!(back.return_hidden_states);
    }
}
