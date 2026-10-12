use std::collections::HashMap;

use serde::{
    de::{self, value::SeqAccessDeserializer, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize,
};
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
    /// ID of the model to use (required for OpenAI, optional for some serving engines)
    pub model: String,

    /// The prompt(s) to generate completions for: a string, an array of
    /// strings, an array of token ids, or an array of token id arrays
    #[validate(custom(function = "validate_completion_prompt"))]
    pub prompt: CompletionPrompt,

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
    /// (0 to 20, as chat's `top_logprobs`)
    #[validate(range(min = 0, max = 20))]
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

    /// Skip special tokens during detokenization. Serialized only when
    /// `false`: `true` is what the engine applies without the field.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
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

    /// Request ID forwarded to the backend for log correlation (an engine extension)
    pub rid: Option<String>,

    /// Additional fields including bootstrap info for PD routing
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Normalizable for CompletionRequest {}

impl CompletionRequest {
    /// Whether the response carries the prompt's logprobs: `echo` with
    /// `logprobs` returns every prompt token with its logprob ahead of the
    /// generated ones.
    pub fn wants_prompt_logprobs(&self) -> bool {
        self.echo && self.logprobs.is_some()
    }
}

fn validate_completion_prompt(prompt: &CompletionPrompt) -> Result<(), validator::ValidationError> {
    let message = match prompt {
        CompletionPrompt::Text(_) => None,
        CompletionPrompt::Texts(texts) => {
            texts.is_empty().then_some("prompt array cannot be empty")
        }
        CompletionPrompt::TokenIds(ids) => ids.is_empty().then_some("prompt array cannot be empty"),
        CompletionPrompt::TokenIdBatches(batches) => {
            if batches.is_empty() {
                Some("prompt array cannot be empty")
            } else if batches.iter().any(Vec::is_empty) {
                Some("a token id prompt cannot be empty")
            } else {
                None
            }
        }
    };
    if let Some(message) = message {
        let mut error = validator::ValidationError::new("prompt_empty");
        error.message = Some(message.into());
        return Err(error);
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

    fn cache_partition(&self) -> CachePartition<'_> {
        CachePartition {
            // Engine extensions carried in the passthrough map, not typed
            // fields: the engines' `cache_salt`, one engine's `extra_key`.
            cache_salt: self.other.get("cache_salt").and_then(Value::as_str),
            extra_key: self.other.get("extra_key").and_then(Value::as_str),
            lora_path: self.lora_path.as_deref(),
        }
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        // Token ids route on their decimal rendering, as `/generate`'s do.
        let ids = |ids: &[u32]| ids.iter().map(u32::to_string).collect::<Vec<_>>();
        match &self.prompt {
            CompletionPrompt::Text(s) => s.clone(),
            CompletionPrompt::Texts(v) => v.join(" "),
            CompletionPrompt::TokenIds(v) => ids(v).join(" "),
            CompletionPrompt::TokenIdBatches(v) => v
                .iter()
                .flat_map(|batch| ids(batch))
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

// ============================================================================
// Prompt
// ============================================================================

/// The `prompt` of a completion request: one text, a batch of texts, one
/// prompt of token ids, or a batch of token-id prompts.
#[derive(Debug, Clone, PartialEq, Serialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Text(String),
    Texts(Vec<String>),
    TokenIds(Vec<u32>),
    TokenIdBatches(Vec<Vec<u32>>),
}

impl CompletionPrompt {
    /// The number of prompts: a batch counts its members, the others one.
    pub fn len(&self) -> usize {
        match self {
            Self::Text(_) | Self::TokenIds(_) => 1,
            Self::Texts(texts) => texts.len(),
            Self::TokenIdBatches(batches) => batches.len(),
        }
    }

    /// Whether the prompt is an empty batch.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Shape probe for the first element of an array `prompt`: it alone picks
/// the variant, so a mixed array is refused with the element that breaks it.
enum FirstPromptItem {
    Text(String),
    Id(u32),
    Ids(Vec<u32>),
}

impl<'de> Deserialize<'de> for FirstPromptItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct FirstPromptItemVisitor;

        impl<'de> Visitor<'de> for FirstPromptItemVisitor {
            type Value = FirstPromptItem;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a string, a token id or an array of token ids")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(FirstPromptItem::Text(v.to_owned()))
            }

            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(FirstPromptItem::Text(v))
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
                u32::try_from(v)
                    .map(FirstPromptItem::Id)
                    .map_err(|_| E::invalid_value(de::Unexpected::Unsigned(v), &"a token id"))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
                u32::try_from(v)
                    .map(FirstPromptItem::Id)
                    .map_err(|_| E::invalid_value(de::Unexpected::Signed(v), &"a token id"))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
                Deserialize::deserialize(SeqAccessDeserializer::new(seq)).map(FirstPromptItem::Ids)
            }
        }

        deserializer.deserialize_any(FirstPromptItemVisitor)
    }
}

impl<'de> Deserialize<'de> for CompletionPrompt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct CompletionPromptVisitor;

        impl<'de> Visitor<'de> for CompletionPromptVisitor {
            type Value = CompletionPrompt;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str(
                    "a string, an array of strings, an array of token ids or an array of token id arrays",
                )
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(CompletionPrompt::Text(v.to_owned()))
            }

            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(CompletionPrompt::Text(v))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                // An empty array parses; validation refuses it by name.
                let Some(first) = seq.next_element::<FirstPromptItem>()? else {
                    return Ok(CompletionPrompt::Texts(Vec::new()));
                };
                let capacity = seq.size_hint().unwrap_or(0).saturating_add(1);
                match first {
                    FirstPromptItem::Text(text) => {
                        let mut texts = Vec::with_capacity(capacity);
                        texts.push(text);
                        while let Some(text) = seq.next_element()? {
                            texts.push(text);
                        }
                        Ok(CompletionPrompt::Texts(texts))
                    }
                    FirstPromptItem::Id(id) => {
                        let mut ids = Vec::with_capacity(capacity);
                        ids.push(id);
                        while let Some(id) = seq.next_element()? {
                            ids.push(id);
                        }
                        Ok(CompletionPrompt::TokenIds(ids))
                    }
                    FirstPromptItem::Ids(head) => {
                        let mut batches = Vec::with_capacity(capacity);
                        batches.push(head);
                        while let Some(ids) = seq.next_element()? {
                            batches.push(ids);
                        }
                        Ok(CompletionPrompt::TokenIdBatches(batches))
                    }
                }
            }
        }

        deserializer.deserialize_any(CompletionPromptVisitor)
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

    fn logprobs_body(logprobs: u32) -> Value {
        serde_json::json!({
            "model": "m",
            "prompt": "hello",
            "echo": true,
            "logprobs": logprobs,
            "max_tokens": 1
        })
    }

    /// `logprobs` takes up to 20 alternatives, the bound of chat's
    /// `top_logprobs`; 21 crosses it.
    #[test]
    fn logprobs_takes_up_to_20_alternatives() {
        for logprobs in [0, 5, 20] {
            let request: CompletionRequest =
                serde_json::from_value(logprobs_body(logprobs)).expect("request");
            assert!(request.validate().is_ok(), "logprobs {logprobs}");
        }
        let request: CompletionRequest =
            serde_json::from_value(logprobs_body(21)).expect("request");
        let errors = request.validate().expect_err("logprobs 21 must be refused");
        let error = &errors.field_errors()["logprobs"][0];
        assert_eq!(error.code, "range");
        assert_eq!(error.params["max"], 20);
        assert_eq!(error.params["value"], 21);
    }

    /// Above 20 the body gets the public API's error for a crossed bound.
    #[cfg(feature = "axum")]
    #[tokio::test]
    async fn logprobs_above_20_is_refused_with_the_public_error() {
        use axum::{
            body::Body,
            extract::{FromRequest, Request},
            http::StatusCode,
        };

        use crate::validated::ValidatedJson;

        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(logprobs_body(21).to_string()))
            .expect("request");
        let Err(response) = ValidatedJson::<CompletionRequest>::from_request(request, &()).await
        else {
            panic!("logprobs 21 must be refused");
        };
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let body: Value = serde_json::from_slice(&bytes).expect("JSON body");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "integer_above_max_value");
        assert_eq!(body["error"]["param"], "logprobs");
    }

    #[test]
    fn default_sglang_flags_are_omitted_and_absent_reads_defaults() {
        let value = serde_json::to_value(req(serde_json::json!({}))).expect("serialize");
        for field in [
            "no_stop_trim",
            "ignore_eos",
            "return_hidden_states",
            "skip_special_tokens",
        ] {
            assert!(value.get(field).is_none(), "{field} serialized at default");
        }

        let back: CompletionRequest = serde_json::from_value(value).expect("roundtrip");
        assert!(!back.no_stop_trim);
        assert!(!back.ignore_eos);
        assert!(!back.return_hidden_states);
        assert!(back.skip_special_tokens);
    }

    #[test]
    fn non_default_sglang_flags_round_trip() {
        let value = serde_json::to_value(req(serde_json::json!({
            "no_stop_trim": true,
            "ignore_eos": true,
            "return_hidden_states": true,
            "skip_special_tokens": false
        })))
        .expect("serialize");
        assert_eq!(value["no_stop_trim"], true);
        assert_eq!(value["ignore_eos"], true);
        assert_eq!(value["return_hidden_states"], true);
        assert_eq!(value["skip_special_tokens"], false);

        let back: CompletionRequest = serde_json::from_value(value).expect("roundtrip");
        assert!(back.no_stop_trim);
        assert!(back.ignore_eos);
        assert!(back.return_hidden_states);
        assert!(!back.skip_special_tokens);
    }

    fn prompt(value: Value) -> Result<CompletionPrompt, String> {
        serde_json::from_value(value).map_err(|e| e.to_string())
    }

    #[test]
    fn every_prompt_shape_parses_and_round_trips() {
        for (value, want) in [
            (serde_json::json!("hi"), CompletionPrompt::Text("hi".into())),
            (
                serde_json::json!(["a", "b"]),
                CompletionPrompt::Texts(vec!["a".into(), "b".into()]),
            ),
            (
                serde_json::json!([1, 2, 3]),
                CompletionPrompt::TokenIds(vec![1, 2, 3]),
            ),
            (
                serde_json::json!([[1, 2], [3]]),
                CompletionPrompt::TokenIdBatches(vec![vec![1, 2], vec![3]]),
            ),
        ] {
            let parsed = prompt(value.clone()).expect("prompt must parse");
            assert_eq!(parsed, want);
            assert_eq!(serde_json::to_value(&parsed).expect("serialize"), value);
        }
        assert_eq!(
            prompt(serde_json::json!([[1], [2], [3]])).map(|p| p.len()),
            Ok(3)
        );
        assert_eq!(prompt(serde_json::json!([7, 8])).map(|p| p.len()), Ok(1));
    }

    #[test]
    fn a_prompt_of_mixed_or_invalid_shape_is_refused_by_its_element() {
        for (value, want) in [
            (serde_json::json!(["a", 1]), "expected a string"),
            (serde_json::json!([1, "a"]), "expected u32"),
            (serde_json::json!([[1], "a"]), "expected a sequence"),
            (serde_json::json!([[1], 2]), "expected a sequence"),
            (serde_json::json!([-1]), "expected a token id"),
            (serde_json::json!([1, -1]), "expected u32"),
            (serde_json::json!([4294967296u64]), "expected a token id"),
            (serde_json::json!([1.5]), "expected a string, a token id"),
            (serde_json::json!(7), "expected a string, an array"),
        ] {
            let error = prompt(value.clone()).expect_err("prompt must be refused");
            assert!(error.contains(want), "{value}: {error}");
        }
    }

    #[test]
    fn an_empty_prompt_array_is_refused_by_validation() {
        for (value, message) in [
            (serde_json::json!([]), "prompt array cannot be empty"),
            (
                serde_json::json!([[1], []]),
                "a token id prompt cannot be empty",
            ),
            (serde_json::json!([[]]), "a token id prompt cannot be empty"),
        ] {
            let request = req(serde_json::json!({"prompt": value}));
            let error = request
                .validate()
                .expect_err("empty prompt must be refused");
            assert!(error.to_string().contains(message), "{value}: {error}");
        }
        assert!(req(serde_json::json!({"prompt": [1, 2]}))
            .validate()
            .is_ok());
    }

    #[test]
    fn a_token_id_prompt_with_echo_and_logprobs_parses() {
        // A client that scores a fixed token sequence sends it as token ids.
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "prompt": [11, 22, 33, 44],
            "echo": true,
            "logprobs": 5,
            "max_tokens": 1,
            "temperature": 0
        }))
        .expect("request must deserialize");
        assert_eq!(
            request.prompt,
            CompletionPrompt::TokenIds(vec![11, 22, 33, 44])
        );
        assert!(request.echo);
        assert_eq!(request.logprobs, Some(5));
        assert!(request.validate().is_ok());
        assert_eq!(request.extract_text_for_routing(), "11 22 33 44");
    }
}
