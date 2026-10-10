//! Builder for ChatCompletionResponse
//!
//! Provides an ergonomic fluent API for constructing chat completion responses.

use crate::{chat::*, common::Usage};

/// Builder for ChatCompletionResponse
///
/// Provides a fluent interface for constructing chat completion responses with sensible defaults.
#[must_use = "Builder does nothing until .build() is called"]
#[derive(Clone, Debug)]
pub struct ChatCompletionResponseBuilder {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Option<Usage>,
    system_fingerprint: Option<String>,
    service_tier: String,
}

impl ChatCompletionResponseBuilder {
    /// Create a new builder with required fields
    ///
    /// # Arguments
    /// - `id`: Completion ID (e.g., "chatcmpl_abc123")
    /// - `model`: Model name used for generation
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: model.into(),
            choices: Vec::new(),
            usage: None,
            system_fingerprint: None,
            service_tier: DEFAULT_SERVICE_TIER.to_string(),
        }
    }

    /// Copy common fields from a ChatCompletionRequest
    ///
    /// This populates the model field from the request.
    pub fn copy_from_request(mut self, request: &ChatCompletionRequest) -> Self {
        self.model.clone_from(&request.model);
        self
    }

    /// Set the object type (default: "chat.completion")
    pub fn object(mut self, object: impl Into<String>) -> Self {
        self.object = object.into();
        self
    }

    /// Set the creation timestamp (default: current time)
    pub fn created(mut self, timestamp: u64) -> Self {
        self.created = timestamp;
        self
    }

    /// Set the choices
    pub fn choices(mut self, choices: Vec<ChatChoice>) -> Self {
        self.choices = choices;
        self
    }

    /// Add a single choice
    pub fn add_choice(mut self, choice: ChatChoice) -> Self {
        self.choices.push(choice);
        self
    }

    /// Set usage information
    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set usage if provided (handles Option)
    pub fn maybe_usage(mut self, usage: Option<Usage>) -> Self {
        if let Some(u) = usage {
            self.usage = Some(u);
        }
        self
    }

    /// Set system fingerprint if provided (handles Option)
    pub fn maybe_system_fingerprint(mut self, fingerprint: Option<impl Into<String>>) -> Self {
        if let Some(fp) = fingerprint {
            self.system_fingerprint = Some(fp.into());
        }
        self
    }

    /// Set the service tier (default: `default`)
    pub fn service_tier(mut self, service_tier: impl Into<String>) -> Self {
        self.service_tier = service_tier.into();
        self
    }

    /// Build the ChatCompletionResponse
    pub fn build(self) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: self.id,
            object: self.object,
            created: self.created,
            model: self.model,
            choices: self.choices,
            usage: self.usage,
            system_fingerprint: self.system_fingerprint,
            service_tier: self.service_tier,
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
        let response = ChatCompletionResponse::builder("chatcmpl_123", "gpt-4").build();

        assert_eq!(response.id, "chatcmpl_123");
        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.object, "chat.completion");
        assert!(response.choices.is_empty());
        assert!(response.usage.is_none());
        assert!(response.system_fingerprint.is_none());
    }

    #[test]
    fn test_build_complete() {
        let choice = ChatChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: Some("Hello!".to_string()),
                refusal: None,
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: Some("stop".to_string()),
            matched_stop: None,
            hidden_states: None,
        };

        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };

        let response = ChatCompletionResponse::builder("chatcmpl_456", "gpt-4")
            .choices(vec![choice.clone()])
            .maybe_usage(Some(usage))
            .maybe_system_fingerprint(Some("fp_123abc"))
            .build();

        assert_eq!(response.id, "chatcmpl_456");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].index, 0);
        assert!(response.usage.is_some());
        assert_eq!(response.system_fingerprint.as_ref().unwrap(), "fp_123abc");
    }

    #[test]
    fn test_add_multiple_choices() {
        let choice1 = ChatChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: Some("Option 1".to_string()),
                refusal: None,
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: Some("stop".to_string()),
            matched_stop: None,
            hidden_states: None,
        };

        let choice2 = ChatChoice {
            index: 1,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: Some("Option 2".to_string()),
                refusal: None,
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: Some("stop".to_string()),
            matched_stop: None,
            hidden_states: None,
        };

        let response = ChatCompletionResponse::builder("chatcmpl_789", "gpt-4")
            .add_choice(choice1)
            .add_choice(choice2)
            .build();

        assert_eq!(response.choices.len(), 2);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[1].index, 1);
    }

    #[test]
    fn test_copy_from_request() {
        let request = ChatCompletionRequest {
            messages: vec![],
            model: "gpt-3.5-turbo".to_string(),
            ..Default::default()
        };

        let response = ChatCompletionResponse::builder("chatcmpl_101", "gpt-4")
            .copy_from_request(&request)
            .build();

        assert_eq!(response.model, "gpt-3.5-turbo"); // Copied from request
    }

    /// The completion object carries every field the OpenAI API sends on
    /// every completion, present (and `null` or zero) when there is nothing
    /// to report: `message.refusal`, `message.content`, `choices[].logprobs`,
    /// `service_tier` and the two usage detail objects with all their
    /// counters. Typed clients read them without a presence check.
    #[test]
    fn completion_object_carries_the_always_present_fields() {
        let response = ChatCompletionResponse::builder("chatcmpl_1", "m")
            .add_choice(ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: None,
                    refusal: None,
                    tool_calls: None,
                    reasoning_content: None,
                },
                logprobs: None,
                finish_reason: Some("stop".to_string()),
                matched_stop: None,
                hidden_states: None,
            })
            .usage(Usage::from_counts(3, 2).with_complete_details())
            .build();
        let value = serde_json::to_value(&response).expect("serialize");

        assert_eq!(value["service_tier"], "default");
        let choice = &value["choices"][0];
        assert!(choice["logprobs"].is_null(), "{choice}");
        assert!(
            choice.get("logprobs").is_some(),
            "logprobs key present: {choice}"
        );
        let message = &choice["message"];
        for key in ["content", "refusal"] {
            assert!(
                message.get(key).is_some_and(serde_json::Value::is_null),
                "{key}: {message}"
            );
        }
        let usage = &value["usage"];
        assert_eq!(
            usage["prompt_tokens_details"],
            serde_json::json!({"cached_tokens": 0, "audio_tokens": 0})
        );
        assert_eq!(
            usage["completion_tokens_details"],
            serde_json::json!({
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
            })
        );
    }

    /// Counters a backend did report survive the fill.
    #[test]
    fn complete_details_keep_reported_counters() {
        let usage = Usage::from_counts(10, 4)
            .with_cached_tokens(6)
            .with_reasoning_tokens(3)
            .with_speculative_tokens(2, 5)
            .with_complete_details();
        let value = serde_json::to_value(&usage).expect("serialize");
        assert_eq!(
            value["prompt_tokens_details"],
            serde_json::json!({"cached_tokens": 6, "audio_tokens": 0})
        );
        assert_eq!(
            value["completion_tokens_details"],
            serde_json::json!({
                "reasoning_tokens": 3,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 2,
                "rejected_prediction_tokens": 3
            })
        );
    }
}
