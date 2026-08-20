//! Response-phase views of the parsed request.
//!
//! Response processing and stream tasks consume these instead of the parsed
//! request, so the request (and its multimodal payloads) is never pinned for
//! the lifetime of the response.

use std::collections::HashMap;

use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{StreamOptions, StringOrArray, Tool, ToolChoice},
    completion::CompletionRequest,
    generate::GenerateRequest,
    messages::{self, CreateMessageRequest},
};
use serde_json::Value;

use crate::routers::grpc::utils;

/// Response-phase view of one request, set by request building (the last
/// request reader before dispatch) and taken by response processing.
pub(crate) enum RequestView {
    Chat(ChatRequestView),
    Generate(GenerateRequestView),
    Completion(CompletionRequestView),
    Messages(MessagesRequestView),
}

pub(crate) struct ChatRequestView {
    pub separate_reasoning: bool,
    pub tool_choice: Option<ToolChoice>,
    pub tools: Option<Vec<Tool>>,
    pub history_tool_calls_count: usize,
    pub stream_options: Option<StreamOptions>,
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    pub reasoning_effort: Option<String>,
    /// `sampling_params.n`, normalized.
    pub expected_choices: u32,
    pub logprobs: bool,
    pub stop: Option<StringOrArray>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub no_stop_trim: bool,
    pub ignore_eos: bool,
    /// Fallback when preparation derived no override.
    pub skip_special_tokens: bool,
}

impl From<&ChatCompletionRequest> for ChatRequestView {
    fn from(request: &ChatCompletionRequest) -> Self {
        Self {
            separate_reasoning: request.separate_reasoning,
            tool_choice: request.tool_choice.clone(),
            tools: request.tools.clone(),
            history_tool_calls_count: utils::get_history_tool_calls_count(request),
            stream_options: request.stream_options.clone(),
            chat_template_kwargs: request.chat_template_kwargs.clone(),
            reasoning_effort: request.reasoning_effort.clone(),
            expected_choices: request.n.unwrap_or(1).max(1),
            logprobs: request.logprobs,
            stop: request.stop.clone(),
            stop_token_ids: request.stop_token_ids.clone(),
            no_stop_trim: request.no_stop_trim,
            ignore_eos: request.ignore_eos,
            skip_special_tokens: request.skip_special_tokens,
        }
    }
}

pub(crate) struct GenerateRequestView {
    pub return_logprob: bool,
    /// `sampling_params.n`, normalized.
    pub expected_choices: u32,
}

impl From<&GenerateRequest> for GenerateRequestView {
    fn from(request: &GenerateRequest) -> Self {
        Self {
            return_logprob: request.return_logprob.unwrap_or(false),
            expected_choices: request
                .sampling_params
                .as_ref()
                .and_then(|p| p.n)
                .unwrap_or(1)
                .max(1),
        }
    }
}

pub(crate) struct MessagesRequestView {
    pub thinking: Option<messages::ThinkingConfig>,
    pub tool_choice: Option<messages::ToolChoice>,
    pub has_tools: bool,
    pub history_tool_calls_count: usize,
    /// Messages tools pre-converted to Chat tools for parser reuse.
    pub chat_tools: Vec<Tool>,
    pub stop_sequences: Option<Vec<String>>,
}

impl From<&CreateMessageRequest> for MessagesRequestView {
    fn from(request: &CreateMessageRequest) -> Self {
        Self {
            thinking: request.thinking.clone(),
            tool_choice: request.tool_choice.clone(),
            has_tools: request.tools.is_some(),
            history_tool_calls_count: utils::message_utils::get_history_tool_calls_count_messages(
                request,
            ),
            chat_tools: request
                .tools
                .as_deref()
                .map(utils::message_utils::extract_chat_tools)
                .unwrap_or_default(),
            stop_sequences: request.stop_sequences.clone(),
        }
    }
}

pub(crate) struct CompletionRequestView {
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

impl From<&CompletionRequest> for CompletionRequestView {
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
            choices_per_prompt: request.n.unwrap_or(1).max(1),
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
