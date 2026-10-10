//! Builder for ResponsesResponse
//!
//! Provides an ergonomic fluent API for constructing ResponsesResponse instances.

use std::collections::HashMap;

use serde_json::Value;

use crate::{common::PromptCacheRetention, responses::*};

/// Builder for ResponsesResponse
///
/// Provides a fluent interface for constructing responses with sensible defaults.
#[must_use = "Builder does nothing until .build() is called"]
#[derive(Clone, Debug)]
pub struct ResponsesResponseBuilder {
    id: String,
    object: String,
    created_at: i64,
    completed_at: Option<i64>,
    background: Option<bool>,
    conversation: Option<String>,
    status: ResponseStatus,
    error: Option<Value>,
    incomplete_details: Option<IncompleteDetails>,
    instructions: Option<String>,
    max_output_tokens: Option<u32>,
    max_tool_calls: Option<u32>,
    model: String,
    output: Vec<ResponseOutputItem>,
    parallel_tool_calls: bool,
    previous_response_id: Option<String>,
    prompt_cache_key: Option<String>,
    prompt_cache_retention: Option<PromptCacheRetention>,
    reasoning: Option<ReasoningInfo>,
    service_tier: Option<ServiceTier>,
    store: bool,
    temperature: Option<f32>,
    text: Option<TextConfig>,
    tool_choice: ResponsesToolChoice,
    tools: Vec<ResponseTool>,
    top_logprobs: Option<u32>,
    top_p: Option<f32>,
    truncation: Option<String>,
    usage: Option<ResponsesUsage>,
    user: Option<String>,
    safety_identifier: Option<String>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    metadata: HashMap<String, Value>,
}

impl ResponsesResponseBuilder {
    /// Create a new builder with required fields
    ///
    /// # Arguments
    /// - `id`: Response ID (e.g., "resp_abc123")
    /// - `model`: Model name used for generation
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "response".to_string(),
            created_at: chrono::Utc::now().timestamp(),
            completed_at: None,
            background: None,
            conversation: None,
            status: ResponseStatus::InProgress,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            model: model.into(),
            output: Vec::new(),
            parallel_tool_calls: true,
            previous_response_id: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning: None,
            service_tier: None,
            store: true,
            temperature: None,
            text: None,
            tool_choice: ResponsesToolChoice::default(),
            tools: Vec::new(),
            top_logprobs: None,
            top_p: None,
            truncation: None,
            usage: None,
            user: None,
            safety_identifier: None,
            frequency_penalty: None,
            presence_penalty: None,
            metadata: HashMap::new(),
        }
    }

    /// Copy common fields from a ResponsesRequest
    ///
    /// This populates fields like instructions, max_output_tokens, temperature, etc.
    /// from the original request, making it easy to construct a response that mirrors
    /// the request parameters: the public API echoes every request parameter on
    /// the Response object (`null` when unset), so the echoes are copied here and
    /// the spec defaults are filled in by [`Self::build`].
    pub fn copy_from_request(mut self, request: &ResponsesRequest) -> Self {
        self.instructions.clone_from(&request.instructions);
        self.max_output_tokens = request.max_output_tokens;
        self.max_tool_calls = request.max_tool_calls;
        self.parallel_tool_calls = request.parallel_tool_calls.unwrap_or(true);
        self.previous_response_id
            .clone_from(&request.previous_response_id);
        self.prompt_cache_key.clone_from(&request.prompt_cache_key);
        self.prompt_cache_retention = request.prompt_cache_retention;
        self.reasoning = request.reasoning.as_ref().map(ReasoningInfo::from);
        self.service_tier.clone_from(&request.service_tier);
        self.text.clone_from(&request.text);
        self.top_logprobs = request.top_logprobs;
        self.truncation = request.truncation.map(|t| t.as_str().to_string());
        self.safety_identifier
            .clone_from(&request.safety_identifier);
        self.frequency_penalty = request.frequency_penalty;
        self.presence_penalty = request.presence_penalty;
        self.store = request.store.unwrap_or(true);
        // ResponsesResponse stores `conversation` as a plain `Option<String>`
        // (response side per spec is `optional { id }` only); flatten the
        // request's union-typed reference down to its underlying id string.
        self.conversation = request.conversation.as_ref().map(|c| c.as_id().to_string());
        self.temperature = request.temperature;
        // Echoed as sent: the object form stays an object, a bare string stays
        // a string; `"auto"` when the request had none.
        self.tool_choice = request.tool_choice.clone().unwrap_or_default();
        self.tools = request.tools.clone().unwrap_or_default();
        self.top_p = request.top_p;
        self.user.clone_from(&request.user);
        self.metadata = request.metadata.clone().unwrap_or_default();
        self
    }

    /// Set the object type (default: "response")
    pub fn object(mut self, object: impl Into<String>) -> Self {
        self.object = object.into();
        self
    }

    /// Set the creation timestamp (default: current time)
    pub fn created_at(mut self, timestamp: i64) -> Self {
        self.created_at = timestamp;
        self
    }

    /// Set the completion timestamp. Populate when the response reaches a
    /// terminal status (`completed`, `incomplete`, `failed`, `cancelled`).
    pub fn completed_at(mut self, timestamp: i64) -> Self {
        self.completed_at = Some(timestamp);
        self
    }

    /// Set the linked conversation ID.
    pub fn conversation(mut self, conversation: impl Into<String>) -> Self {
        self.conversation = Some(conversation.into());
        self
    }

    /// Set the response status
    pub fn status(mut self, status: ResponseStatus) -> Self {
        self.status = status;
        self
    }

    /// Set error information (if status is failed)
    pub fn error(mut self, error: Value) -> Self {
        self.error = Some(error);
        self
    }

    /// Set incomplete details (if the response reached `incomplete` status)
    pub fn incomplete_details(mut self, details: IncompleteDetails) -> Self {
        self.incomplete_details = Some(details);
        self
    }

    /// Set system instructions
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set max output tokens
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Set output items
    pub fn output(mut self, output: Vec<ResponseOutputItem>) -> Self {
        self.output = output;
        self
    }

    /// Add a single output item
    pub fn add_output(mut self, item: ResponseOutputItem) -> Self {
        self.output.push(item);
        self
    }

    /// Set whether parallel tool calls are enabled
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = enabled;
        self
    }

    /// Set previous response ID (if continuation)
    pub fn previous_response_id(mut self, id: impl Into<String>) -> Self {
        self.previous_response_id = Some(id.into());
        self
    }

    /// Set reasoning information
    pub fn reasoning(mut self, reasoning: ReasoningInfo) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Set whether the response is stored
    pub fn store(mut self, store: bool) -> Self {
        self.store = store;
        self
    }

    /// Set temperature setting
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set text format settings if provided (handles Option)
    pub fn maybe_text(mut self, text: Option<TextConfig>) -> Self {
        if let Some(t) = text {
            self.text = Some(t);
        }
        self
    }

    /// Set tool choice setting
    pub fn tool_choice(mut self, tool_choice: ResponsesToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }

    /// Set available tools
    pub fn tools(mut self, tools: Vec<ResponseTool>) -> Self {
        self.tools = tools;
        self
    }

    /// Set top-p setting
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set truncation strategy
    pub fn truncation(mut self, truncation: impl Into<String>) -> Self {
        self.truncation = Some(truncation.into());
        self
    }

    /// Set usage information
    pub fn usage(mut self, usage: ResponsesUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set usage if provided (handles Option)
    pub fn maybe_usage(mut self, usage: Option<ResponsesUsage>) -> Self {
        if let Some(u) = usage {
            self.usage = Some(u);
        }
        self
    }

    /// Copy from request if provided (handles Option)
    pub fn maybe_copy_from_request(mut self, request: Option<&ResponsesRequest>) -> Self {
        if let Some(req) = request {
            self = self.copy_from_request(req);
        }
        self
    }

    /// Set user identifier
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set safety identifier
    pub fn safety_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.safety_identifier = Some(identifier.into());
        self
    }

    /// Set metadata
    pub fn metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add a single metadata entry
    pub fn add_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set whether the response runs in background mode
    pub fn background(mut self, background: bool) -> Self {
        self.background = Some(background);
        self
    }

    /// Build the ResponsesResponse.
    ///
    /// Fields the public API always reports take their spec defaults when
    /// nothing set them: `temperature`/`top_p` 1.0, `truncation` "disabled",
    /// `service_tier` "default", `top_logprobs` 0, the penalties 0.0,
    /// `background` false, `reasoning` with both keys null, `text` with the
    /// plain text format and medium verbosity, and `completed_at` stamped the
    /// moment a terminal status is built.
    pub fn build(self) -> ResponsesResponse {
        let terminal = matches!(
            self.status,
            ResponseStatus::Completed
                | ResponseStatus::Incomplete
                | ResponseStatus::Failed
                | ResponseStatus::Cancelled
        );
        let completed_at = self
            .completed_at
            .or_else(|| terminal.then(|| chrono::Utc::now().timestamp()));
        ResponsesResponse {
            id: self.id,
            object: self.object,
            created_at: self.created_at,
            completed_at,
            background: Some(self.background.unwrap_or(false)),
            conversation: self.conversation,
            status: self.status,
            error: self.error,
            incomplete_details: self.incomplete_details,
            instructions: self.instructions,
            max_output_tokens: self.max_output_tokens,
            max_tool_calls: self.max_tool_calls,
            model: self.model,
            output: self.output,
            parallel_tool_calls: self.parallel_tool_calls,
            previous_response_id: self.previous_response_id,
            prompt_cache_key: self.prompt_cache_key,
            prompt_cache_retention: self.prompt_cache_retention,
            reasoning: Some(self.reasoning.unwrap_or_default()),
            service_tier: Some(self.service_tier.unwrap_or(ServiceTier::Default)),
            store: self.store,
            temperature: Some(self.temperature.unwrap_or(1.0)),
            text: Some(self.text.unwrap_or_default().with_response_defaults()),
            tool_choice: self.tool_choice,
            tools: self.tools,
            top_logprobs: Some(self.top_logprobs.unwrap_or(0)),
            top_p: Some(self.top_p.unwrap_or(1.0)),
            truncation: Some(
                self.truncation
                    .unwrap_or_else(|| Truncation::Disabled.as_str().to_string()),
            ),
            usage: self.usage,
            user: self.user,
            safety_identifier: self.safety_identifier,
            frequency_penalty: Some(self.frequency_penalty.unwrap_or(0.0)),
            presence_penalty: Some(self.presence_penalty.unwrap_or(0.0)),
            metadata: self.metadata,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_minimal() {
        let response = ResponsesResponse::builder("resp_123", "gpt-4").build();

        assert_eq!(response.id, "resp_123");
        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.object, "response");
        assert_eq!(response.status, ResponseStatus::InProgress);
        assert!(response.output.is_empty());
        assert!(response.parallel_tool_calls);
        assert!(response.store);
    }

    #[test]
    fn test_build_complete() {
        let response = ResponsesResponse::builder("resp_123", "gpt-4")
            .status(ResponseStatus::Completed)
            .instructions("You are a helpful assistant")
            .max_output_tokens(1000)
            .temperature(0.7)
            .top_p(0.9)
            .parallel_tool_calls(false)
            .store(false)
            .build();

        assert_eq!(response.status, ResponseStatus::Completed);
        assert_eq!(
            response.instructions.as_ref().unwrap(),
            "You are a helpful assistant"
        );
        assert_eq!(response.max_output_tokens, Some(1000));
        assert_eq!(response.temperature, Some(0.7));
        assert_eq!(response.top_p, Some(0.9));
        assert!(!response.parallel_tool_calls);
        assert!(!response.store);
    }

    #[test]
    fn test_copy_from_request() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("test".to_string()),
            instructions: Some("Be helpful".to_string()),
            max_output_tokens: Some(500),
            temperature: Some(0.8),
            top_p: Some(0.95),
            parallel_tool_calls: Some(false),
            store: Some(false),
            user: Some("user_123".to_string()),
            metadata: Some(HashMap::from([(
                "key".to_string(),
                serde_json::json!("value"),
            )])),
            ..Default::default()
        };

        let response = ResponsesResponse::builder("resp_456", "gpt-4")
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .build();

        assert_eq!(response.instructions.as_ref().unwrap(), "Be helpful");
        assert_eq!(response.max_output_tokens, Some(500));
        assert_eq!(response.temperature, Some(0.8));
        assert_eq!(response.top_p, Some(0.95));
        assert!(!response.parallel_tool_calls);
        assert!(!response.store);
        assert_eq!(response.user.as_ref().unwrap(), "user_123");
        assert_eq!(
            response.metadata.get("key").unwrap(),
            &serde_json::json!("value")
        );
    }

    #[test]
    fn test_add_output_items() {
        let response = ResponsesResponse::builder("resp_789", "gpt-4")
            .add_output(ResponseOutputItem::Message {
                id: "msg_1".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                status: "completed".to_string(),
                phase: None,
            })
            .add_output(ResponseOutputItem::Message {
                id: "msg_2".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                status: "completed".to_string(),
                phase: None,
            })
            .build();

        assert_eq!(response.output.len(), 2);
    }

    #[test]
    fn test_add_metadata() {
        let response = ResponsesResponse::builder("resp_101", "gpt-4")
            .add_metadata("key1", serde_json::json!("value1"))
            .add_metadata("key2", serde_json::json!(42))
            .build();

        assert_eq!(response.metadata.len(), 2);
        assert_eq!(response.metadata.get("key1").unwrap(), "value1");
        assert_eq!(response.metadata.get("key2").unwrap(), 42);
    }

    /// Every field the public API returns on every Response object is on the
    /// wire even when nothing set it: the spec-required ones as `null`, the
    /// rest with the spec defaults.
    #[test]
    fn minimal_response_carries_every_always_present_field() {
        let response = ResponsesResponse::builder("resp_min", "m")
            .status(ResponseStatus::Completed)
            .build();
        let wire = serde_json::to_value(&response).unwrap();
        let body = wire.as_object().unwrap();

        for key in [
            "error",
            "instructions",
            "incomplete_details",
            "previous_response_id",
            "user",
            "safety_identifier",
            "prompt_cache_key",
            "prompt_cache_retention",
            "max_output_tokens",
            "max_tool_calls",
            "usage",
        ] {
            assert_eq!(body.get(key), Some(&Value::Null), "{key} must be null");
        }
        assert_eq!(body["top_p"], serde_json::json!(1.0));
        assert_eq!(body["temperature"], serde_json::json!(1.0));
        assert_eq!(body["truncation"], "disabled");
        assert_eq!(body["service_tier"], "default");
        assert_eq!(body["top_logprobs"], 0);
        assert_eq!(body["frequency_penalty"], serde_json::json!(0.0));
        assert_eq!(body["presence_penalty"], serde_json::json!(0.0));
        assert_eq!(body["background"], false);
        assert_eq!(
            body["reasoning"],
            serde_json::json!({"effort": null, "summary": null})
        );
        assert_eq!(
            body["text"],
            serde_json::json!({"format": {"type": "text"}, "verbosity": "medium"})
        );
        assert!(
            body["completed_at"].is_i64(),
            "a completed response is stamped: {}",
            body["completed_at"]
        );
        assert!(
            !body.contains_key("conversation"),
            "an unlinked response omits conversation"
        );
    }

    #[test]
    fn in_progress_response_has_no_completed_at() {
        let wire =
            serde_json::to_value(ResponsesResponse::builder("resp_run", "m").build()).unwrap();
        assert_eq!(wire["status"], "in_progress");
        assert_eq!(wire["completed_at"], Value::Null);
    }

    #[test]
    fn copy_from_request_echoes_the_request_parameters() {
        let request = ResponsesRequest {
            model: "m".to_string(),
            input: ResponseInput::Text("test".to_string()),
            reasoning: Some(ResponseReasoningParam {
                effort: Some(ReasoningEffort::High),
                summary: Some(ReasoningSummary::Concise),
            }),
            service_tier: Some(ServiceTier::Flex),
            truncation: Some(Truncation::Auto),
            top_logprobs: Some(3),
            max_tool_calls: Some(4),
            prompt_cache_key: Some("cache-1".to_string()),
            prompt_cache_retention: Some(PromptCacheRetention::Duration24h),
            safety_identifier: Some("safe-1".to_string()),
            frequency_penalty: Some(0.5),
            presence_penalty: Some(-0.5),
            text: Some(TextConfig {
                format: None,
                verbosity: Some(Verbosity::Low),
            }),
            ..Default::default()
        };

        let wire = serde_json::to_value(
            ResponsesResponse::builder("resp_echo", "m")
                .copy_from_request(&request)
                .build(),
        )
        .unwrap();

        assert_eq!(
            wire["reasoning"],
            serde_json::json!({"effort": "high", "summary": "concise"})
        );
        assert_eq!(wire["service_tier"], "flex");
        assert_eq!(wire["truncation"], "auto");
        assert_eq!(wire["top_logprobs"], 3);
        assert_eq!(wire["max_tool_calls"], 4);
        assert_eq!(wire["prompt_cache_key"], "cache-1");
        assert_eq!(wire["prompt_cache_retention"], "24h");
        assert_eq!(wire["safety_identifier"], "safe-1");
        assert_eq!(wire["frequency_penalty"], serde_json::json!(0.5));
        assert_eq!(wire["presence_penalty"], serde_json::json!(-0.5));
        assert_eq!(
            wire["text"],
            serde_json::json!({"format": {"type": "text"}, "verbosity": "low"})
        );
    }
}
