use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::Validate;

use super::{
    common::{default_true, deserialize_null_as_false, is_false, GenerationRequest, InputIds},
    sampling_params::SamplingParams,
};
use crate::validated::Normalizable;

// ============================================================================
// SGLang Generate API (native format)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_generate_request"))]
pub struct GenerateRequest {
    /// Text input - SGLang native format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(default = "super::common::default_unknown_model")]
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

    /// Sampling parameters (sglang style)
    #[serde(skip_serializing_if = "Option::is_none")]
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
    /// of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
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

/// Validation function for GenerateRequest - ensure exactly one input type is provided
fn validate_generate_request(req: &GenerateRequest) -> Result<(), validator::ValidationError> {
    // Exactly one of text or input_ids must be provided
    // Note: input_embeds not yet supported in Rust implementation
    let has_text = req.text.is_some();
    let has_input_ids = req.input_ids.is_some();

    let count = [has_text, has_input_ids].iter().filter(|&&x| x).count();

    if count == 0 {
        return Err(validator::ValidationError::new(
            "Either text or input_ids should be provided.",
        ));
    }

    if count > 1 {
        return Err(validator::ValidationError::new(
            "Either text or input_ids should be provided.",
        ));
    }

    Ok(())
}

impl GenerationRequest for GenerateRequest {
    fn rid(&self) -> Option<&str> {
        self.rid.as_deref()
    }

    fn is_stream(&self) -> bool {
        self.stream
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
                InputIds::Single(ids) => ids
                    .iter()
                    .map(|&id| id.to_string())
                    .collect::<Vec<String>>()
                    .join(" "),
                InputIds::Batch(batches) => batches
                    .iter()
                    .flat_map(|batch| batch.iter().map(|&id| id.to_string()))
                    .collect::<Vec<String>>()
                    .join(" "),
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
// SGLang Generate Response Types
// ============================================================================

/// SGLang generate response (single completion or array for n>1)
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
}
