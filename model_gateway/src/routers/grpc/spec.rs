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

use crate::routers::grpc::{
    regular::{stages::decisions::scoring::DecisionScoring, symphony::Assignment},
    utils::Names,
    wire::Wire,
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
    /// The wire the answer leaves in, selected by the profile of the model
    /// the client asked for ([`Wire::of`]): the stream's shape follows it.
    pub wire: Wire,
    pub separate_reasoning: bool,
    pub tool_choice: Option<ToolChoice>,
    /// The tools the response is read against, shared with the readers that
    /// type its calls (one per choice) rather than copied into each.
    pub tools: Option<Arc<[Tool]>>,
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
    /// The parser names the request's preparation resolved once from the model's
    /// cards (`ProcessingState::parser_names`), handed on by request building for
    /// the response's two readers (the reasoning parser's and the tool-call
    /// parser's names); `None` when no resolution was made (a spec built from the
    /// request alone), and the readers resolve them as before.
    pub parser_names: Option<Names>,
}

/// Whether the response to a chat `request` is scanned for tool calls: the
/// request declares tools (an empty list included), a message declares one,
/// or the model's provider reads tool calls without any
/// ([`ProviderProfile::parses_tool_calls_without_tools`]). A `tool_choice`
/// of `none` is the drivers' to check. [`ChatResponseSpec::tools`] is `Some`
/// exactly then, and Symphony keeps a table's calls exactly then
/// (`regular::symphony::assign_for`), so a call the conversation established
/// without a tool inventory comes back as a call under either parser. `provider`
/// is the model's profile as the request's preparation derived it once.
pub(crate) fn scans_for_tool_calls_for(
    request: &ChatCompletionRequest,
    provider: ProviderProfile,
) -> bool {
    request.declares_tools_for(provider) || provider.parses_tool_calls_without_tools()
}

impl From<&ChatCompletionRequest> for ChatResponseSpec {
    fn from(request: &ChatCompletionRequest) -> Self {
        Self::for_request(request, ProviderProfile::for_model(&request.model))
    }
}

impl ChatResponseSpec {
    /// The spec of `request` under `provider`, its model's profile as the
    /// request's preparation derived it once; `From` derives the profile for
    /// a request that carries none.
    pub(crate) fn for_request(request: &ChatCompletionRequest, provider: ProviderProfile) -> Self {
        Self {
            wire: Wire::of(provider),
            separate_reasoning: request.separate_reasoning,
            tool_choice: request.tool_choice.clone(),
            // Every tool the model may call, dynamic tools declared on messages
            // included: the request's own list shared, not copied (a new list
            // only when a message declares a tool). `None` when the response is
            // not scanned for tool calls: the request declares no tools anywhere
            // (`ChatCompletionRequest::declares_tools`, the one answer every
            // gate reads) and the provider does not parse tool calls without
            // them (`scans_for_tool_calls_for`).
            tools: scans_for_tool_calls_for(request, provider)
                .then(|| request.shared_tool_list_for(provider)),
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
            parser_names: None,
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
    /// The thinking blocks carry their text: a `thinking.display` other than
    /// `omitted`, or none (see `ThinkingDisplay::returns_text`).
    pub thinking_text_returned: bool,
    /// Whether the prompt ends inside the model's thinking block, decided by
    /// request building from the rendered prompt.
    pub reasoning_in_prefill: bool,
    /// The rendered prompt's whole token count, set by request building:
    /// what the stream's `message_start` reports as `input_tokens` before
    /// the engine has said anything. The final `message_delta` carries the
    /// engine's count, split on a prefix-cache hit into `input_tokens`
    /// outside the cache and `cache_read_input_tokens`, which this count
    /// includes.
    pub prompt_tokens: u32,
    /// The request's `max_tokens`: the budget an engine `length` finish is
    /// read against (see [`Self::length_stop_reason`]).
    pub max_tokens: u32,
    /// The context window the selected worker advertises for the model, set
    /// by request building and again when a retry selects another worker;
    /// `None` when the worker advertises none.
    pub context_window: Option<u32>,
    pub tool_choice: Option<messages::ToolChoice>,
    pub has_tools: bool,
    pub history_tool_calls_count: usize,
    /// The custom tools as chat tools, converted once: the template renders
    /// them (narrowed by `tool_choice`), the constraints and the parsers
    /// read them; shared with the reader that types the calls.
    pub chat_tools: Arc<[Tool]>,
    /// The stop sequences as the decoders match them.
    pub stop_sequences: Option<StringOrArray>,
    /// Symphony's assignment when a table reads the message (see `regular::symphony`).
    pub symphony: Option<Assignment>,
    /// `disable_parallel_tool_use: true`: only the first tool call is
    /// returned.
    pub first_tool_call_only: bool,
    /// The parser names the request's preparation resolved once from the model's
    /// cards (`ProcessingState::parser_names`), handed on by request building for
    /// the response's two readers (the reasoning parser's and the tool-call
    /// parser's names); `None` when no resolution was made (a spec built from the
    /// request alone), and the readers resolve them as before.
    pub parser_names: Option<Names>,
}

impl From<&CreateMessageRequest> for MessagesResponseSpec {
    fn from(request: &CreateMessageRequest) -> Self {
        Self {
            thinking_requested: request.thinking_toggle().unwrap_or(false),
            thinking_text_returned: request
                .thinking_display()
                .is_none_or(messages::ThinkingDisplay::returns_text),
            reasoning_in_prefill: false,
            prompt_tokens: 0,
            max_tokens: request.max_tokens,
            context_window: None,
            tool_choice: request.tool_choice.clone(),
            has_tools: request.tools.is_some(),
            history_tool_calls_count: request.history_tool_calls_count(),
            chat_tools: request.chat_tools().into(),
            stop_sequences: request.stop_sequences.clone().map(StringOrArray::Array),
            symphony: None,
            first_tool_call_only: request.disables_parallel_tool_use(),
            parser_names: None,
        }
    }
}

impl MessagesResponseSpec {
    /// The stop reason of an engine `length` finish, which the engines report
    /// alike for the request's budget and for the model's window:
    /// `max_tokens` when the generation used the whole budget, else
    /// `model_context_window_exceeded`. A generation that stopped short of
    /// the budget ran out of window (an engine that clamps the budget to the
    /// window finishes this way); so did one whose prompt and output together
    /// reached the worker's advertised window. `prompt_tokens` is the
    /// engine's count of the prompt it held, as the finish reports it: media
    /// expanded on the worker makes it larger than the rendered count this
    /// spec carries.
    pub fn length_stop_reason(
        &self,
        prompt_tokens: u32,
        output_tokens: u32,
    ) -> messages::StopReason {
        let short_of_budget = output_tokens < self.max_tokens;
        let window_filled = self
            .context_window
            .is_some_and(|window| prompt_tokens.saturating_add(output_tokens) >= window);
        if short_of_budget || window_filled {
            messages::StopReason::ModelContextWindowExceeded
        } else {
            messages::StopReason::MaxTokens
        }
    }
}

#[derive(Clone)]
pub(crate) struct CompletionResponseSpec {
    /// `n`, normalized.
    pub choices_per_prompt: u32,
    pub echo: bool,
    pub suffix: Option<String>,
    /// `logprobs`: how many alternatives per generated token the request
    /// asked for; `None` when it asked for no logprobs.
    pub logprobs: Option<u32>,
    pub include_usage: bool,
    /// Populated only when `echo` (choices prepend their prompt text).
    pub prompt_texts: Vec<String>,
    pub stop: Option<StringOrArray>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub skip_special_tokens: bool,
    pub no_stop_trim: bool,
    pub ignore_eos: bool,
}

impl CompletionResponseSpec {
    /// `prompt_texts` are the prepared prompts' texts, one per prompt (a text
    /// prompt as sent, a token prompt decoded); only an `echo` reads them, so
    /// a builder passes none when the request asks for no echo.
    pub(crate) fn new(request: &CompletionRequest, prompt_texts: Vec<String>) -> Self {
        Self {
            choices_per_prompt: request.choices(),
            echo: request.echo,
            suffix: request.suffix.clone(),
            logprobs: request.logprobs,
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
    use crate::routers::grpc::testing::tool;

    fn chat_request(value: Value) -> ChatCompletionRequest {
        serde_json::from_value(value).expect("request deserializes")
    }

    fn tool_names(spec: &ChatResponseSpec) -> Vec<String> {
        spec.tools
            .iter()
            .flat_map(|tools| tools.iter())
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

    /// The point of sharing: with request tools and no message-declared tool,
    /// the spec holds the request's own list, not a copy of it.
    #[test]
    fn chat_spec_shares_the_request_s_tool_list() {
        let request = chat_request(json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [tool("get_weather")]
        }));

        let spec = ChatResponseSpec::from(&request);

        assert!(Arc::ptr_eq(
            request.tools.as_ref().expect("the request's tools"),
            spec.tools.as_ref().expect("the spec's tools")
        ));
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

    /// The wire follows the profile of the requested model, which the
    /// stream's shape then reads off the spec.
    #[test]
    fn chat_spec_wire_follows_the_requested_model() {
        for (model, wire) in [
            ("MiniMax-M3", Wire::Minimax),
            ("deepseek-flash", Wire::DeepSeek),
            ("gpt-4o", Wire::OpenAi),
            ("self-hosted-model", Wire::OpenAi),
        ] {
            let request = chat_request(json!({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}]
            }));

            assert_eq!(ChatResponseSpec::from(&request).wire, wire, "{model}");
        }
    }

    #[test]
    fn chat_spec_scans_minimax_responses_for_tool_calls_without_tools() {
        let request = chat_request(json!({
            "model": "MiniMax-M3",
            "messages": [{"role": "user", "content": "try again"}]
        }));

        assert_eq!(
            ChatResponseSpec::from(&request).tools.as_deref(),
            Some(&[][..])
        );
    }

    /// The Messages spec carries the request's budget; the window waits for
    /// request building.
    #[test]
    fn messages_spec_takes_the_budget_from_the_request() {
        let request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .expect("request deserializes");

        let spec = MessagesResponseSpec::from(&request);

        assert_eq!(spec.max_tokens, 64);
        assert_eq!(spec.context_window, None);
    }

    /// An engine `length` finish is `max_tokens` when the budget was used
    /// whole and `model_context_window_exceeded` when the generation stopped
    /// short of it or the prompt and the output together filled the window.
    #[test]
    fn a_length_finish_reads_the_budget_against_the_window() {
        use messages::StopReason::{MaxTokens, ModelContextWindowExceeded};

        let mut spec = MessagesResponseSpec::from(
            &serde_json::from_value::<CreateMessageRequest>(json!({
                "model": "m", "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .expect("request deserializes"),
        );
        spec.prompt_tokens = 100;

        // No window known: the budget alone decides.
        assert_eq!(spec.length_stop_reason(100, 64), MaxTokens);
        assert_eq!(spec.length_stop_reason(100, 3), ModelContextWindowExceeded);
        assert_eq!(spec.length_stop_reason(100, 0), ModelContextWindowExceeded);

        // A window the whole budget stays inside: still the budget.
        spec.context_window = Some(165);
        assert_eq!(spec.length_stop_reason(100, 64), MaxTokens);
        // The prompt and the output together reach the window.
        spec.context_window = Some(164);
        assert_eq!(spec.length_stop_reason(100, 64), ModelContextWindowExceeded);
        // Short of the budget, whatever the window.
        assert_eq!(spec.length_stop_reason(100, 3), ModelContextWindowExceeded);
        // The engine's count decides, not the rendered one the spec carries:
        // media expanded on the worker fills the window the spec would miss.
        spec.context_window = Some(170);
        assert_eq!(spec.length_stop_reason(100, 64), MaxTokens);
        assert_eq!(spec.length_stop_reason(110, 64), ModelContextWindowExceeded);
        // A synthetic count never overflows the sum.
        assert_eq!(
            spec.length_stop_reason(u32::MAX, 64),
            ModelContextWindowExceeded
        );
    }
}
