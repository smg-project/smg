//! Per-endpoint response specs: the pipeline contract between request
//! building and response processing.
//!
//! Request building is the last reader of the parsed request; the spec it
//! hands on (the Messages spec is begun by preparation, which renders its
//! tools, and completed by request building) is the only request-derived
//! input available to response processing and streaming tasks. The Harmony
//! variant is the one deliberate exception: its tool loop re-reads the
//! request across iterations, so its spec explicitly owns a handle to it.

use std::sync::Arc;

use llm_multimodal::registry::transcription::TranscriptionFamily;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{StreamOptions, StringOrArray, Tool, ToolChoice},
    completion::CompletionRequest,
    generate::GenerateRequest,
    messages::{self, CreateMessageRequest},
    profile::ProviderProfile,
    responses::ResponsesRequest,
};

use crate::routers::grpc::regular::{
    stages::decisions::scoring::DecisionScoring, symphony::Assignment,
};

/// Response-phase contract for one request, handed on by request building.
#[derive(Clone)]
pub(crate) enum ResponseSpec {
    Chat(Box<ChatResponseSpec>),
    Generate(GenerateResponseSpec),
    Completion(CompletionResponseSpec),
    Messages(Box<MessagesResponseSpec>),
    /// Embedding/classify response processing needs only dispatch metadata.
    Embedding,
    Classify,
    Decisions(DecisionsResponseSpec),
    Harmony(HarmonyResponseSpec),
    Transcription(TranscriptionResponseSpec),
}

#[derive(Clone)]
pub(crate) struct DecisionsResponseSpec {
    pub scoring: DecisionScoring,
    pub questions: Vec<DecisionQuestionSpec>,
    pub max_input_tokens: usize,
}

/// The candidate tokens and exact prompt size expected for one scoring RPC.
#[derive(Clone)]
pub(crate) struct DecisionQuestionSpec {
    pub label_ids: Vec<u32>,
    pub input_tokens: usize,
}

/// Wire format of a `/v1/audio/transcriptions` response body.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TranscriptionResponseFormat {
    Json,
    Text,
}

/// Response-phase contract for a transcription request: the decoded text is
/// post-processed by the resolved family and rendered in the chosen format.
#[derive(Clone)]
pub(crate) struct TranscriptionResponseSpec {
    pub format: TranscriptionResponseFormat,
    /// The resolved family, for output post-processing. `'static` — families
    /// are registry constants.
    pub family: &'static dyn TranscriptionFamily,
}

#[derive(Clone)]
pub(crate) struct ChatResponseSpec {
    /// Provider dialect of the model the client asked for, as request
    /// validation selects it; picks provider-specific response behaviour.
    pub provider: ProviderProfile,
    pub separate_reasoning: bool,
    pub tool_choice: Option<ToolChoice>,
    pub tools: Option<Vec<Tool>>,
    pub history_tool_calls_count: usize,
    pub stream_options: Option<StreamOptions>,
    /// `parallel_tool_calls: false`: only the first tool call is returned.
    pub first_tool_call_only: bool,
    /// `n`, normalized.
    pub expected_choices: u32,
    pub logprobs: bool,
    /// Alternatives per token the client asked for; none when absent.
    pub top_logprobs: Option<u32>,
    pub stop: Option<StringOrArray>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub no_stop_trim: bool,
    pub ignore_eos: bool,
    /// Fallback when preparation derived no override.
    pub skip_special_tokens: bool,
    /// Rendered prompt tokens the provider does not bill; set by request building.
    pub unbilled_prompt_tokens: u32,
    /// Whether the prompt ends inside the model's thinking block, decided by
    /// request building from the rendered prompt.
    pub reasoning_in_prefill: bool,
    /// Set by request building when Symphony reads the response (`regular::symphony`):
    /// the table, and the part of the rendered prompt the engine replays.
    pub symphony: Option<Assignment>,
}

impl From<&ChatCompletionRequest> for ChatResponseSpec {
    fn from(request: &ChatCompletionRequest) -> Self {
        let provider = ProviderProfile::for_model(&request.model);
        Self {
            provider,
            separate_reasoning: request.separate_reasoning,
            tool_choice: request.tool_choice.clone(),
            // Every tool the model may call, dynamic tools declared on messages
            // included. `None` when the response is not scanned for tool calls:
            // the request declares no tools anywhere
            // (`ChatCompletionRequest::declares_tools`, the one answer every
            // gate reads) and the provider does not parse tool calls without
            // them.
            tools: (request.declares_tools() || provider.parses_tool_calls_without_tools())
                .then(|| request.effective_tools().cloned().collect()),
            history_tool_calls_count: request.history_tool_calls_count(),
            stream_options: request.stream_options.clone(),
            first_tool_call_only: request.disables_parallel_tool_calls(),
            expected_choices: request.choices(),
            logprobs: request.logprobs,
            top_logprobs: request.top_logprobs,
            stop: request.stop.clone(),
            stop_token_ids: request.stop_token_ids.clone(),
            no_stop_trim: request.no_stop_trim,
            ignore_eos: request.ignore_eos,
            skip_special_tokens: request.skip_special_tokens,
            unbilled_prompt_tokens: 0,
            reasoning_in_prefill: false,
            symphony: None,
        }
    }
}

#[derive(Clone)]
pub(crate) struct GenerateResponseSpec {
    pub return_logprob: bool,
    /// `sampling_params.n`, normalized.
    pub expected_choices: u32,
}

impl From<&GenerateRequest> for GenerateResponseSpec {
    fn from(request: &GenerateRequest) -> Self {
        Self {
            return_logprob: request.return_logprob.unwrap_or(false),
            expected_choices: request.choices(),
        }
    }
}

/// The response-phase contract of a Messages request. Preparation builds it
/// (`PreparationOutput::Messages`), since the template renders its chat
/// tools; request building fills in what the rendered prompt and the worker
/// decide.
#[derive(Clone)]
pub(crate) struct MessagesResponseSpec {
    /// The `thinking` block asks for thinking (`enabled` or `adaptive`).
    pub thinking_requested: bool,
    /// Whether the prompt ends inside the model's thinking block, decided by
    /// request building from the rendered prompt.
    pub reasoning_in_prefill: bool,
    pub tool_choice: Option<messages::ToolChoice>,
    pub has_tools: bool,
    pub history_tool_calls_count: usize,
    /// The custom tools as chat tools, converted once: the template renders
    /// them (narrowed by `tool_choice`), the constraints and the parsers
    /// read them.
    pub chat_tools: Vec<Tool>,
    /// The stop sequences as the decoders match them.
    pub stop_sequences: Option<StringOrArray>,
    /// Symphony's assignment when a table reads the message (see `regular::symphony`).
    pub symphony: Option<Assignment>,
    /// `disable_parallel_tool_use: true`: only the first tool call is
    /// returned.
    pub first_tool_call_only: bool,
}

impl From<&CreateMessageRequest> for MessagesResponseSpec {
    fn from(request: &CreateMessageRequest) -> Self {
        Self {
            thinking_requested: request.thinking_toggle().unwrap_or(false),
            reasoning_in_prefill: false,
            tool_choice: request.tool_choice.clone(),
            has_tools: request.tools.is_some(),
            history_tool_calls_count: request.history_tool_calls_count(),
            chat_tools: request.chat_tools(),
            stop_sequences: request.stop_sequences.clone().map(StringOrArray::Array),
            symphony: None,
            first_tool_call_only: request.disables_parallel_tool_use(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct CompletionResponseSpec {
    /// `n`, normalized.
    pub choices_per_prompt: u32,
    pub echo: bool,
    pub suffix: Option<String>,
    pub logprobs: bool,
    pub include_usage: bool,
    /// Populated only when `echo` (choices prepend their prompt text).
    pub prompt_texts: Vec<String>,
    pub stop: Option<StringOrArray>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub skip_special_tokens: bool,
    pub no_stop_trim: bool,
    pub ignore_eos: bool,
}

impl From<&CompletionRequest> for CompletionResponseSpec {
    fn from(request: &CompletionRequest) -> Self {
        let prompt_texts = if request.echo {
            match &request.prompt {
                StringOrArray::String(text) => vec![text.clone()],
                StringOrArray::Array(texts) => texts.clone(),
            }
        } else {
            Vec::new()
        };
        Self {
            choices_per_prompt: request.choices(),
            echo: request.echo,
            suffix: request.suffix.clone(),
            logprobs: request.logprobs.is_some(),
            include_usage: request
                .stream_options
                .as_ref()
                .and_then(|opts| opts.include_usage)
                .unwrap_or(false),
            prompt_texts,
            stop: request.stop.clone(),
            stop_token_ids: request.stop_token_ids.clone(),
            skip_special_tokens: request.skip_special_tokens,
            no_stop_trim: request.no_stop_trim,
            ignore_eos: request.ignore_eos,
        }
    }
}

/// Harmony's spec owns the request: the tool loop and channel parsers
/// legitimately re-read it after dispatch. Post-build code reaches the
/// request only through this explicit handle.
#[derive(Clone)]
pub(crate) enum HarmonyResponseSpec {
    Chat(Arc<ChatCompletionRequest>),
    Responses(Arc<ResponsesRequest>),
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;

    fn chat_request(value: Value) -> ChatCompletionRequest {
        serde_json::from_value(value).expect("request deserializes")
    }

    fn tool(name: &str) -> Value {
        json!({
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object", "properties": {}}}
        })
    }

    fn tool_names(spec: &ChatResponseSpec) -> Vec<String> {
        spec.tools
            .iter()
            .flatten()
            .map(|tool| tool.function.name.clone())
            .collect()
    }

    #[test]
    fn chat_spec_tools_include_dynamic_tools() {
        let request = chat_request(json!({
            "model": "kimi-k3",
            "messages": [
                {"role": "system", "content": "", "tools": [tool("get_weather")]},
                {"role": "user", "content": "what is the weather in beijing?"}
            ],
            "tool_choice": "required"
        }));

        let spec = ChatResponseSpec::from(&request);

        assert_eq!(tool_names(&spec), ["get_weather"]);
    }

    #[test]
    fn chat_spec_tools_keep_request_tools_first() {
        let request = chat_request(json!({
            "model": "kimi-k3",
            "messages": [
                {"role": "system", "content": "", "tools": [tool("dynamic_a")]},
                {"role": "user", "content": "hi"},
                {"role": "developer", "content": "", "tools": [tool("dynamic_b")]}
            ],
            "tools": [tool("global")]
        }));

        let spec = ChatResponseSpec::from(&request);

        assert_eq!(tool_names(&spec), ["global", "dynamic_a", "dynamic_b"]);
    }

    #[test]
    fn chat_spec_tools_follow_an_empty_request_list_beside_a_message_declaration() {
        // The empty request-level list is a declaration (the normalizer answers
        // it with `tool_choice: none`, which switches the calls off); the
        // message's tool is the set.
        let request = chat_request(json!({
            "model": "kimi-k3",
            "messages": [
                {"role": "system", "content": "", "tools": [tool("get_weather")]},
                {"role": "user", "content": "hi"}
            ],
            "tools": []
        }));

        assert_eq!(
            tool_names(&ChatResponseSpec::from(&request)),
            ["get_weather"]
        );
    }

    #[test]
    fn chat_spec_tools_none_for_a_message_declaration_the_profile_does_not_read() {
        let request = chat_request(json!({
            "model": "self-hosted-model",
            "messages": [
                {"role": "system", "content": "", "tools": [tool("get_weather")]},
                {"role": "user", "content": "hi"}
            ]
        }));

        assert!(ChatResponseSpec::from(&request).tools.is_none());
    }

    #[test]
    fn chat_spec_tools_none_without_any_declaration() {
        let request = chat_request(json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hi"}]
        }));

        assert!(ChatResponseSpec::from(&request).tools.is_none());
    }

    #[test]
    fn chat_spec_provider_follows_the_requested_model() {
        let minimax = chat_request(json!({
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let openai = chat_request(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));

        assert_eq!(
            ChatResponseSpec::from(&minimax).provider,
            ProviderProfile::Minimax
        );
        assert_eq!(
            ChatResponseSpec::from(&openai).provider,
            ProviderProfile::OpenAi
        );
    }

    #[test]
    fn chat_spec_scans_minimax_responses_for_tool_calls_without_tools() {
        let request = chat_request(json!({
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "try again"}]
        }));

        assert_eq!(ChatResponseSpec::from(&request).tools, Some(Vec::new()));
    }
}
