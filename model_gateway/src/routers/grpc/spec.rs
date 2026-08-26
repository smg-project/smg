//! Per-endpoint response specs: the pipeline contract between request
//! building and response processing.
//!
//! Request building is the last reader of the parsed request; the spec it
//! produces is the only request-derived input available to response
//! processing and streaming tasks. The Harmony variant is the one deliberate
//! exception: its tool loop re-reads the request across iterations, so its
//! spec explicitly owns a handle to it.

use std::{collections::HashMap, sync::Arc};

use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{StreamOptions, StringOrArray, Tool, ToolChoice},
    completion::CompletionRequest,
    generate::GenerateRequest,
    messages::{self, CreateMessageRequest},
    responses::ResponsesRequest,
};
use serde_json::Value;

use crate::routers::grpc::utils;

/// Response-phase contract for one request, produced by request building.
#[derive(Clone)]
pub(crate) enum ResponseSpec {
    Chat(Box<ChatResponseSpec>),
    Generate(GenerateResponseSpec),
    Completion(CompletionResponseSpec),
    Messages(MessagesResponseSpec),
    /// Embedding/classify response processing needs only dispatch metadata.
    Embedding,
    Classify,
    Harmony(HarmonyResponseSpec),
}

#[derive(Clone)]
pub(crate) struct ChatResponseSpec {
    pub separate_reasoning: bool,
    pub tool_choice: Option<ToolChoice>,
    pub tools: Option<Vec<Tool>>,
    pub history_tool_calls_count: usize,
    pub stream_options: Option<StreamOptions>,
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    pub reasoning_effort: Option<String>,
    /// `n`, normalized.
    pub expected_choices: u32,
    pub logprobs: bool,
    pub stop: Option<StringOrArray>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub no_stop_trim: bool,
    pub ignore_eos: bool,
    /// Fallback when preparation derived no override.
    pub skip_special_tokens: bool,
}

impl From<&ChatCompletionRequest> for ChatResponseSpec {
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
            expected_choices: request
                .sampling_params
                .as_ref()
                .and_then(|p| p.n)
                .unwrap_or(1)
                .max(1),
        }
    }
}

#[derive(Clone)]
pub(crate) struct MessagesResponseSpec {
    pub thinking: Option<messages::ThinkingConfig>,
    pub tool_choice: Option<messages::ToolChoice>,
    pub has_tools: bool,
    pub history_tool_calls_count: usize,
    /// Messages tools pre-converted to Chat tools for parser reuse.
    pub chat_tools: Vec<Tool>,
    pub stop_sequences: Option<Vec<String>>,
}

impl From<&CreateMessageRequest> for MessagesResponseSpec {
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

/// Harmony's spec owns the request: the tool loop and channel parsers
/// legitimately re-read it after dispatch. Post-build code reaches the
/// request only through this explicit handle.
#[derive(Clone)]
pub(crate) enum HarmonyResponseSpec {
    Chat(Arc<ChatCompletionRequest>),
    Responses(Arc<ResponsesRequest>),
}
