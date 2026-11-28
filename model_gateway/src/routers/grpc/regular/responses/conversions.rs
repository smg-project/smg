//! Conversion utilities for translating between /v1/responses and /v1/chat/completions formats
//!
//! This module implements the conversion approach where:
//! 1. ResponsesRequest → ChatCompletionRequest (for backend processing)
//! 2. ChatCompletionResponse → ResponsesResponse (for client response)
//!
//! This allows the gRPC router to reuse the existing chat pipeline infrastructure
//! without requiring Python backend changes.

use std::sync::Arc;

use openai_protocol::{
    chat::{
        AssistantReasoning, ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
        MessageContent,
    },
    common::{
        ContentPart, FunctionCallResponse, ImageUrl, JsonSchemaFormat, ResponseFormat, Tool,
        ToolCall, ToolChoice, ToolChoiceValue, Usage,
    },
    profile::ProviderProfile,
    responses::{
        CustomToolCallOutputContent, CustomToolInputContentPart, IncludeField, ResponseContentPart,
        ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
        ResponseReasoningContent::ReasoningText, ResponseTool, ResponseUsage, ResponsesRequest,
        ResponsesResponse, ResponsesUsage, StringOrContentParts, TextConfig, TextFormat,
    },
    UNKNOWN_MODEL_ID,
};
use tracing::warn;

use crate::routers::{
    common::responses_compat::{format_agent_message_text, lower_call_id_less_tool_output_parts},
    grpc::common::responses::{
        history::hosted_call_result,
        streaming::OutputItemKind,
        utils::{
            custom_tool_input, custom_tool_names, cut_call, decode_reasoning_content,
            encode_reasoning_content, extract_tools_from_response_tools, function_call_status,
            generation_failure_error, resolve_function_identity, response_status,
        },
    },
};

/// Why a Responses request does not convert to a chat request, and the code
/// its refusal is answered under: a content part the chat pipeline has no
/// form for, in a message or in a function output, is the chat route's own
/// `unsupported_content_part`; every other refusal keeps
/// `convert_request_failed`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ConversionError {
    /// An input item kind this path does not carry.
    #[error("Unsupported input item type")]
    UnsupportedItem,
    /// A content part of a message item the chat pipeline has no form for
    /// (see `response_parts_to_message_content`).
    #[error("Unsupported content part type {part:?} for gRPC backends: {why}")]
    UnsupportedContentPart {
        part: &'static str,
        why: &'static str,
    },
    /// A function output with a part the text-only chat path cannot carry
    /// (`StringOrContentParts::to_text_only`).
    #[error("{0}")]
    UnsupportedToolOutput(String),
    #[error("Request must contain at least one message")]
    NoMessages,
    /// A tool result no call before it issued (see
    /// `ProviderProfile::unanswerable_tool_result`).
    #[error("No tool call found for function call output with call_id '{0}'")]
    UnansweredToolResult(String),
}

impl ConversionError {
    /// The error envelope's code for this refusal.
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedContentPart { .. } | Self::UnsupportedToolOutput(_) => {
                "unsupported_content_part"
            }
            Self::UnsupportedItem | Self::NoMessages | Self::UnansweredToolResult(_) => {
                "convert_request_failed"
            }
        }
    }
}

/// Convert a ResponsesRequest to ChatCompletionRequest for processing through the chat pipeline
///
/// # Conversion Logic
/// - `input` (text/items) → `messages` (chat messages)
/// - `instructions` → system message (prepended)
/// - `max_output_tokens` → `max_completion_tokens`
/// - `stop` → `stop` (the engine's stop list and the gateway's own trim of a
///   matched sequence both read it from the chat request)
/// - `tools` → function tools extracted from ResponseTools
/// - `tool_choice` → passed through from request
/// - the sampling knobs (`temperature`, `top_p`, `frequency_penalty`, `presence_penalty`,
///   `top_k`, `min_p`, `repetition_penalty`) → their chat twins; the three the Responses
///   request keeps as plain values cross only when set
/// - Response-specific fields (previous_response_id, conversation) are handled by router
pub(crate) fn responses_to_chat(
    req: &ResponsesRequest,
) -> Result<ChatCompletionRequest, ConversionError> {
    responses_to_chat_with_tools(req, None)
}

/// [`responses_to_chat`] for the MCP tool loop: the MCP tools the loop offers
/// join the request's function tools while that list is being built, so the
/// request holds one list, wrapped once, and no iteration copies the schemas
/// again. The list is present even when both are empty, as the loop has
/// always sent it.
pub(super) fn responses_to_chat_with_mcp_tools(
    req: &ResponsesRequest,
    mcp_chat_tools: &[Tool],
) -> Result<ChatCompletionRequest, ConversionError> {
    responses_to_chat_with_tools(req, Some(mcp_chat_tools))
}

fn responses_to_chat_with_tools(
    req: &ResponsesRequest,
    mcp_chat_tools: Option<&[Tool]>,
) -> Result<ChatCompletionRequest, ConversionError> {
    let mut messages = Vec::new();

    // 1. Add system message if instructions provided
    if let Some(instructions) = &req.instructions {
        messages.push(ChatMessage::System {
            content: MessageContent::Text(instructions.clone()),
            name: None,
            ext: Default::default(),
        });
    }

    // 2. Convert input to chat messages
    match &req.input {
        ResponseInput::Text(text) => {
            // Simple text input → user message
            messages.push(ChatMessage::User {
                ext: Default::default(),
                content: MessageContent::Text(text.clone()),
                name: None,
            });
        }
        ResponseInput::Items(items) => {
            // Structured items → convert each to appropriate chat message.
            // Assistant-side items are merged into one assistant turn by
            // `push_chat_message`; see its doc comment.
            for item in items {
                push_item(&mut messages, item)?;
            }
        }
    }

    // Ensure we have at least one message
    if messages.is_empty() {
        return Err(ConversionError::NoMessages);
    }

    // 3. Extract function tools from ResponseTools; the tool loop's MCP tools
    // join them here, while the list is still being built (see
    // `responses_to_chat_with_mcp_tools` and `chat_tools`).
    let (tools, tool_choice) = chat_tools(req, mcp_chat_tools);

    // 4. Build ChatCompletionRequest
    let model = if req.model.is_empty() {
        UNKNOWN_MODEL_ID.to_string()
    } else {
        req.model.clone()
    };

    // A tool result no call of the assistant turn right before it issued,
    // under an unknown id or with no call before it at all, is a malformed
    // history under the contracts that say so (the public API's words for
    // it), answered as a bad request, not sent to the engine as a turn the
    // model never asked for. The history loader keeps a stored turn with its
    // calls and drops a stored result whose call did not survive the reload
    // (`history::load_history`), so a result with no call before it here is
    // the client's.
    if let Some(call_id) = ProviderProfile::for_model(&model).unanswerable_tool_result(&messages) {
        return Err(ConversionError::UnansweredToolResult(call_id.to_string()));
    }

    Ok(chat_request(req, model, messages, tools, tool_choice))
}

/// Fold one Responses input item into the chat messages: a message for its
/// role, a tool call of the history (and its result, once it ran), a
/// reasoning item as the assistant's thought, a tool output as a tool
/// message; the gateway's own records of a hosted tool session add nothing,
/// and the item kinds this path does not carry are refused.
fn push_item(
    messages: &mut Vec<ChatMessage>,
    item: &ResponseInputOutputItem,
) -> Result<(), ConversionError> {
    match item {
        ResponseInputOutputItem::AdditionalTools { .. }
        | ResponseInputOutputItem::ToolSearchCall { .. }
        | ResponseInputOutputItem::ToolSearchOutput { .. }
        | ResponseInputOutputItem::WebSearchCall { .. } => {}
        ResponseInputOutputItem::AgentMessage {
            author,
            recipient,
            content,
            ..
        } => {
            let text = format_agent_message_text(author, recipient, content).map_err(|part| {
                ConversionError::UnsupportedContentPart {
                    part,
                    why: "only plaintext \"input_text\" parts are carried in an agent_message",
                }
            })?;
            push_role_message(messages, "user", MessageContent::Text(text))?;
        }
        ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
            let content = match content {
                StringOrContentParts::String(s) => MessageContent::Text(s.clone()),
                StringOrContentParts::Array(parts) => response_parts_to_message_content(parts)?,
            };
            push_role_message(messages, role, content)?;
        }
        ResponseInputOutputItem::Message { role, content, .. } => {
            push_role_message(messages, role, response_parts_to_message_content(content)?)?;
        }
        ResponseInputOutputItem::FunctionToolCall {
            call_id,
            name,
            namespace,
            arguments,
            output,
            ..
        } => {
            // Tool call from history: the assistant's decision,
            // then the tool result if the output is present.
            push_chat_message(
                messages,
                assistant_tool_call(call_id, name, namespace.as_deref(), arguments.clone()),
            );
            if let Some(output_text) = output {
                messages.push(tool_result(call_id, output_text.clone()));
            }
        }
        ResponseInputOutputItem::Reasoning {
            content,
            encrypted_content,
            ..
        } => {
            let reasoning_text = reasoning_text(content, encrypted_content.as_deref());
            if !reasoning_text.is_empty() {
                let assistant = ChatMessage::Assistant {
                    content: None,
                    name: None,
                    refusal: None,
                    audio: None,
                    function_call: None,
                    tool_calls: None,
                    reasoning: AssistantReasoning {
                        text: Some(reasoning_text),
                    },
                    ext: Default::default(),
                };
                push_chat_message(messages, assistant);
            }
        }
        ResponseInputOutputItem::FunctionCallOutput {
            call_id,
            name,
            namespace,
            output,
            ..
        } => {
            if let Some(call_id) = call_id.as_deref().filter(|id| !id.is_empty()) {
                // Function call output - add as tool message
                // Note: The function name is looked up from prev_outputs in Harmony path
                // For Chat path, we just use the call_id
                messages.push(tool_result(
                    call_id,
                    output
                        .to_text_only()
                        .map_err(ConversionError::UnsupportedToolOutput)?,
                ));
            } else {
                let parts = lower_call_id_less_tool_output_parts(
                    name.as_deref(),
                    namespace.as_deref(),
                    output.clone(),
                );
                push_role_message(messages, "user", response_parts_to_message_content(&parts)?)?;
            }
        }
        // A hosted tool call of a stored turn replays as the
        // assistant's call and, once it ran, its result: the
        // model sees what it asked for and what it got, as it
        // does for a function call.
        ResponseInputOutputItem::McpCall {
            id,
            name,
            arguments,
            output,
            error,
            status,
            ..
        } => {
            push_chat_message(
                messages,
                assistant_tool_call(id, name, None, arguments.clone()),
            );
            if let Some(result) =
                hosted_call_result(output.as_deref(), error.as_ref(), status.as_deref())
            {
                messages.push(tool_result(id, result));
            }
        }
        // The gateway's own record of a hosted tool session (the
        // tools it listed, the approval it asked for): nothing
        // the model said or was told.
        ResponseInputOutputItem::McpListTools { .. }
        | ResponseInputOutputItem::McpApprovalRequest { .. } => {}
        ResponseInputOutputItem::McpApprovalResponse { .. }
        | ResponseInputOutputItem::ComputerCall { .. }
        | ResponseInputOutputItem::ComputerCallOutput { .. } => {
            return unsupported(Some("Approval item reached chat conversion"));
        }
        ResponseInputOutputItem::ImageGenerationCall { .. } => {
            return unsupported(Some(
                "image_generation_call input item reached chat conversion",
            ));
        }
        ResponseInputOutputItem::Compaction { .. }
        | ResponseInputOutputItem::ItemReference { .. } => {
            return unsupported(None);
        }
        // Custom tool history replays as the function tool_call the
        // request-side downgrade produces ({"input": "..."}) plus a
        // plain tool message for the client's output.
        ResponseInputOutputItem::CustomToolCall {
            call_id,
            input,
            name,
            namespace,
            ..
        } => {
            push_chat_message(
                messages,
                assistant_tool_call(
                    call_id,
                    name,
                    namespace.as_deref(),
                    serde_json::json!({ "input": input }).to_string(),
                ),
            );
        }
        ResponseInputOutputItem::CustomToolCallOutput {
            call_id, output, ..
        } => {
            messages.push(tool_result(call_id, custom_tool_output_text(output)));
        }
        ResponseInputOutputItem::ShellCall { .. }
        | ResponseInputOutputItem::ShellCallOutput { .. } => {
            return unsupported(Some("Shell tool item reached chat conversion"));
        }
        ResponseInputOutputItem::ApplyPatchCall { .. }
        | ResponseInputOutputItem::ApplyPatchCallOutput { .. } => {
            return unsupported(Some("apply_patch item reached chat conversion"));
        }
        // T5 schema-only: forced-cascade arm, no behavior.
        ResponseInputOutputItem::LocalShellCall { .. }
        | ResponseInputOutputItem::LocalShellCallOutput { .. } => {
            return unsupported(None);
        }
        ResponseInputOutputItem::Program { .. } | ResponseInputOutputItem::ProgramOutput { .. } => {
            return unsupported(Some(
                "Programmatic tool calling item reached chat conversion",
            ));
        }
    }
    Ok(())
}

/// The refusal of an item kind this path does not carry, warned about where
/// the kind says why.
fn unsupported(warning: Option<&'static str>) -> Result<(), ConversionError> {
    if let Some(warning) = warning {
        warn!(function = "responses_to_chat", "{warning}");
    }
    Err(ConversionError::UnsupportedItem)
}

/// A tool's result for the call it answers, as the chat history carries it.
fn tool_result(call_id: &str, content: String) -> ChatMessage {
    ChatMessage::Tool {
        content: MessageContent::Text(content),
        tool_call_id: call_id.to_string(),
    }
}

/// A reasoning item's text: the plain text, else what a `store=false`
/// client handed back as the opaque blob this gateway produced.
fn reasoning_text(
    content: &[openai_protocol::responses::ResponseReasoningContent],
    encrypted_content: Option<&str>,
) -> String {
    let reasoning_text = content
        .iter()
        .map(|c| match c {
            ReasoningText { text } => text.as_str(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    if reasoning_text.is_empty() {
        return encrypted_content
            .and_then(decode_reasoning_content)
            .unwrap_or_default();
    }
    reasoning_text
}

/// A custom tool's output as text: its string, or its text parts joined.
fn custom_tool_output_text(output: &CustomToolCallOutputContent) -> String {
    match output {
        CustomToolCallOutputContent::Text(s) => s.clone(),
        CustomToolCallOutputContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                CustomToolInputContentPart::InputText { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// The chat tools and tool choice of a Responses request: its function tools,
/// with the tool loop's MCP tools joined while the list is being built, so the
/// request holds one list and no iteration copies the schemas again.
/// `tool_choice: none` forbids calls; offering the schemas anyway makes the
/// model re-issue calls on replayed histories, and with parsing disabled the
/// raw call markup leaks into the message text, so the request's own tools are
/// withheld then while the MCP tools stay offered. The chat and Messages paths
/// render the tools under `none` as the engine's own server does for every
/// API (`utils::request_as_rendered`), so the same request renders a different
/// prompt here: a known divergence, kept until this API's rule is decided on
/// its own. Under the loop the list is present even when both are empty, as
/// the loop has always sent one; the plain conversion carries none when there
/// are no tools.
fn chat_tools(
    req: &ResponsesRequest,
    mcp_chat_tools: Option<&[Tool]>,
) -> (Option<Arc<[Tool]>>, Option<ToolChoice>) {
    let tool_choice = req.tool_choice.as_ref().map(|tc| tc.to_chat_tool_choice());
    // Under `none` the request's own tools are withheld, so their schemas are
    // not copied out of the request at all.
    let function_tools = if matches!(tool_choice, Some(ToolChoice::Value(ToolChoiceValue::None))) {
        Vec::new()
    } else {
        extract_tools_from_response_tools(req.tools.as_deref())
    };
    let tools = match mcp_chat_tools {
        Some(mcp_chat_tools) => {
            let mut all_tools = function_tools;
            all_tools.extend(mcp_chat_tools.iter().cloned());
            Some(all_tools.into())
        }
        None if function_tools.is_empty() => None,
        None => Some(function_tools.into()),
    };
    (tools, tool_choice)
}

/// The chat request of the converted messages and tools, with the sampling,
/// streaming and reasoning fields the Responses request sets.
fn chat_request(
    req: &ResponsesRequest,
    model: String,
    messages: Vec<ChatMessage>,
    tools: Option<Arc<[Tool]>>,
    tool_choice: Option<ToolChoice>,
) -> ChatCompletionRequest {
    let is_streaming = req.stream.unwrap_or(false);
    ChatCompletionRequest {
        messages,
        model,
        temperature: req.temperature,
        // The chat twins of these five knobs are `Option`s whose `None` lets
        // the model's own defaults fill the knob; three of them are plain
        // values with serde defaults on the Responses request and cross only
        // when set, or every request would carry the sentinels.
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        top_k: req.top_k_if_set(),
        min_p: req.min_p_if_set(),
        repetition_penalty: req.repetition_penalty_if_set(),
        max_completion_tokens: req.max_output_tokens,
        stream: is_streaming,
        // Preserve caller-provided stream_options (e.g. `include_obfuscation: false`
        // on the Responses API) and only default `include_usage` when the caller
        // did not set it. Non-streaming requests intentionally drop stream_options.
        stream_options: if is_streaming {
            let mut opts = req.stream_options.clone().unwrap_or_default();
            if opts.include_usage.is_none() {
                opts.include_usage = Some(true);
            }
            Some(opts)
        } else {
            None
        },
        parallel_tool_calls: req.parallel_tool_calls,
        top_logprobs: req.top_logprobs,
        top_p: req.top_p,
        // The engine's stop list and the gateway's own trim of a matched
        // sequence both read the chat request's `stop`, on every turn.
        stop: req.stop.clone(),
        skip_special_tokens: true,
        tools,
        tool_choice,
        response_format: map_text_to_response_format(req.text.as_ref()),
        reasoning_effort: req
            .reasoning
            .as_ref()
            .and_then(|r| r.effort)
            .map(|effort| effort.as_str().to_string()),
        // `Default` leaves these false while the HTTP layer defaults them to
        // true; without them a reasoning model's <think> block comes back as
        // message text with any constrained tool-call JSON embedded in it.
        separate_reasoning: true,
        stream_reasoning: is_streaming,
        ..Default::default()
    }
}

/// Convert Responses content parts to chat message content. Text, output
/// text and refusals become text parts and an `input_image` with a URL
/// becomes an `image_url` part. The parts the chat pipeline has no form for
/// are refused, never dropped: an `input_file` (chat messages carry text,
/// image, audio and video parts and no file part, and the gateway keeps no
/// file store to resolve one from) and an `input_image` that names only a
/// `file_id`. A request answered as if such a part had not been sent is a
/// wrong answer the client cannot detect.
fn response_parts_to_message_content(
    content: &[ResponseContentPart],
) -> Result<MessageContent, ConversionError> {
    let mut parts = Vec::new();
    for part in content {
        match part {
            ResponseContentPart::InputText { text }
            | ResponseContentPart::OutputText { text, .. } => {
                parts.push(ContentPart::Text { text: text.clone() });
            }
            ResponseContentPart::Refusal { refusal } => {
                parts.push(ContentPart::Text {
                    text: refusal.clone(),
                });
            }
            ResponseContentPart::InputImage {
                image_url: Some(url),
                ..
            } => {
                parts.push(ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: url.clone().into(),
                        detail: None,
                        max_long_side_pixel: None,
                    },
                });
            }
            ResponseContentPart::InputImage {
                image_url: None, ..
            } => {
                return Err(ConversionError::UnsupportedContentPart {
                    part: "input_image",
                    why: "it names no image_url, and a file_id is not resolvable here",
                });
            }
            ResponseContentPart::InputFile { .. } => {
                return Err(ConversionError::UnsupportedContentPart {
                    part: "input_file",
                    why: "the chat pipeline carries no file part",
                });
            }
        }
    }
    // Plain text stays a string, as it did before parts were carried at all;
    // the one text part moves out of the list instead of being copied again.
    Ok(match parts.pop() {
        None => MessageContent::Text(String::new()),
        Some(ContentPart::Text { text }) if parts.is_empty() => MessageContent::Text(text),
        Some(last) => {
            parts.push(last);
            MessageContent::Parts(parts)
        }
    })
}

/// Append a role-tagged input message. A `developer` turn folds into the
/// leading system message: OpenAI ranks it with `system`, above `user`, and
/// appended unlabelled it reads as one more conversational turn that the
/// latest user message is free to override. The fold carries text only, so a
/// media part in a developer item is refused here instead of being dropped on
/// the way: media parts are taken in user messages.
fn push_role_message(
    messages: &mut Vec<ChatMessage>,
    role: &str,
    content: MessageContent,
) -> Result<(), ConversionError> {
    if role != "developer" {
        push_chat_message(messages, role_to_chat_message(role, content));
        return Ok(());
    }
    if let MessageContent::Parts(parts) = &content {
        if let Some(part) = parts.iter().find_map(unfoldable_part_name) {
            return Err(ConversionError::UnsupportedContentPart {
                part,
                why: "a developer message folds into the system text, and media parts are taken in user messages",
            });
        }
    }
    let text = content.to_simple_string();
    match messages.first_mut() {
        Some(ChatMessage::System {
            content: MessageContent::Text(existing),
            ..
        }) => {
            existing.push_str("\n\nDeveloper instructions:\n");
            existing.push_str(&text);
        }
        _ => messages.insert(
            0,
            ChatMessage::System {
                content: MessageContent::Text(format!("Developer instructions:\n{text}")),
                name: None,
                ext: Default::default(),
            },
        ),
    }
    Ok(())
}

/// The name of a part the developer fold would drop (`ContentPart::text` has
/// no text for it). The Responses conversion produces `image_url` parts from
/// `input_image` and no other media part, so that is the name a client sees;
/// the chat kinds this conversion never produces keep their own tag.
fn unfoldable_part_name(part: &ContentPart) -> Option<&'static str> {
    match part {
        ContentPart::Text { .. } | ContentPart::Refusal { .. } => None,
        ContentPart::ImageUrl { .. } => Some("input_image"),
        ContentPart::AudioUrl { .. } => Some("audio_url"),
        ContentPart::InputAudio { .. } => Some("input_audio"),
        ContentPart::VideoUrl { .. } => Some("video_url"),
        ContentPart::Unknown(_) => Some("unknown"),
    }
}

/// Replay one historical tool call as the assistant turn that issued it.
fn assistant_tool_call(
    call_id: &str,
    name: &str,
    namespace: Option<&str>,
    arguments: String,
) -> ChatMessage {
    ChatMessage::Assistant {
        content: None,
        name: None,
        refusal: None,
        audio: None,
        function_call: None,
        tool_calls: Some(vec![ToolCall {
            id: call_id.to_string(),
            tool_type: "function".to_string(),
            function: FunctionCallResponse {
                name: match namespace {
                    Some(namespace) => format!("{namespace}.{name}"),
                    None => name.to_string(),
                },
                arguments: Some(arguments),
            },
        }]),
        reasoning: Default::default(),
        ext: Default::default(),
    }
}

/// Append a chat message, merging consecutive assistant-side items into one
/// assistant turn. One Responses assistant turn arrives as several items
/// (reasoning, message, function_call, ...); replayed as separate assistant
/// messages they render back-to-back assistant blocks the template never
/// produces, and tool results pair with the wrong (reasoning-only) message.
fn push_chat_message(messages: &mut Vec<ChatMessage>, message: ChatMessage) {
    match (messages.last_mut(), message) {
        (
            Some(ChatMessage::Assistant {
                content,
                tool_calls,
                reasoning,
                ..
            }),
            ChatMessage::Assistant {
                content: new_content,
                tool_calls: new_tool_calls,
                reasoning: new_reasoning,
                ..
            },
        ) => {
            if let Some(new_reasoning) = new_reasoning.text {
                let merged = reasoning.text.get_or_insert_with(String::new);
                if !merged.is_empty() {
                    merged.push('\n');
                }
                merged.push_str(&new_reasoning);
            }
            if let Some(new_content) = new_content {
                *content = Some(match content.take() {
                    Some(existing) => merge_message_content(existing, new_content),
                    None => new_content,
                });
            }
            if let Some(new_tool_calls) = new_tool_calls {
                tool_calls
                    .get_or_insert_with(Vec::new)
                    .extend(new_tool_calls);
            }
        }
        (_, message) => messages.push(message),
    }
}

fn merge_message_content(a: MessageContent, b: MessageContent) -> MessageContent {
    match (a, b) {
        (MessageContent::Text(mut a), MessageContent::Text(b)) => {
            a.push('\n');
            a.push_str(&b);
            MessageContent::Text(a)
        }
        (a, b) => MessageContent::Parts(
            [a, b]
                .into_iter()
                .flat_map(|content| match content {
                    MessageContent::Text(text) => vec![ContentPart::Text { text }],
                    MessageContent::Parts(parts) => parts,
                })
                .collect(),
        ),
    }
}

/// Convert role and text to ChatMessage
fn role_to_chat_message(role: &str, content: MessageContent) -> ChatMessage {
    match role {
        "user" => ChatMessage::User {
            content,
            name: None,
            ext: Default::default(),
        },
        "assistant" => ChatMessage::Assistant {
            content: Some(content),
            name: None,
            refusal: None,
            audio: None,
            function_call: None,
            tool_calls: None,
            reasoning: Default::default(),
            ext: Default::default(),
        },
        "system" => ChatMessage::System {
            content,
            name: None,
            ext: Default::default(),
        },
        _ => {
            // Unknown role, treat as user message
            ChatMessage::User {
                content,
                name: None,
                ext: Default::default(),
            }
        }
    }
}

/// Map TextConfig from Responses API to ResponseFormat for Chat API
///
/// Converts the structured output configuration from the Responses API format
/// to the Chat API format for non-Harmony models.
fn map_text_to_response_format(text: Option<&TextConfig>) -> Option<ResponseFormat> {
    let text_config = text?;
    let format = text_config.format.as_ref()?;

    match format {
        TextFormat::Text => Some(ResponseFormat::Text),
        TextFormat::JsonObject => Some(ResponseFormat::JsonObject),
        TextFormat::JsonSchema {
            name,
            schema,
            description,
            strict,
        } => Some(ResponseFormat::JsonSchema {
            json_schema: Box::new(JsonSchemaFormat {
                description: description.clone(),
                name: name.clone(),
                schema: schema.clone(),
                strict: *strict,
            }),
        }),
    }
}

/// Convert a ChatCompletionResponse to ResponsesResponse
///
/// # Conversion Logic
/// - `id` → `response_id_override` if provided, otherwise `chat_resp.id`
/// - `model` → `model` (pass through)
/// - `choices[0].message` → `output` array (convert to ResponseOutputItem::Message)
/// - `choices[0].finish_reason` → determines `status` (stop/length → Completed)
/// - `created` timestamp → `created_at`
pub(crate) fn chat_to_responses(
    chat_resp: &ChatCompletionResponse,
    original_req: &ResponsesRequest,
    effective_tools: Option<&[ResponseTool]>,
    response_id_override: Option<String>,
) -> Result<ResponsesResponse, String> {
    // Extract the first choice (responses API doesn't support n>1)
    let choice = chat_resp
        .choices
        .first()
        .ok_or_else(|| "Chat response contains no choices".to_string())?;

    // The assistant message as output items. Reasoning comes first (OpenAI
    // order), so items replayed in emitted order rebuild the same turn.
    let mut output: Vec<ResponseOutputItem> = Vec::new();
    output.extend(reasoning_output_item(choice, original_req));
    output.extend(message_output_item(choice));
    // The cut-call read comes before the calls become items: a custom tool's
    // item carries no status.
    let cut = cut_call(
        choice.finish_reason.as_deref(),
        choice
            .message
            .tool_calls
            .iter()
            .flatten()
            .map(|call| call.function.arguments.as_deref().unwrap_or_default()),
    );
    push_tool_call_output_items(&mut output, choice, effective_tools);

    let (status, incomplete_details) = response_status(choice.finish_reason.as_deref(), cut);
    let usage = chat_resp.usage.as_ref().map(responses_usage);

    // Generate response
    let response_id = response_id_override.unwrap_or_else(|| chat_resp.id.clone());
    let mut builder = ResponsesResponse::builder(&response_id, &chat_resp.model)
        .copy_from_request(original_req)
        .created_at(chat_resp.created as i64)
        .status(status)
        .output(output)
        .maybe_text(original_req.text.clone())
        .maybe_usage(usage);
    if let Some(details) = incomplete_details {
        builder = builder.incomplete_details(details);
    }
    if let Some(error) = generation_failure_error(choice.finish_reason.as_deref()) {
        builder = builder.error(error);
    }
    Ok(builder.build())
}

/// The assistant's reasoning as the turn's first output item, when it
/// reasoned: with the opaque encrypted copy when the request asked for it.
fn reasoning_output_item(
    choice: &ChatChoice,
    original_req: &ResponsesRequest,
) -> Option<ResponseOutputItem> {
    let reasoning = choice.message.reasoning_content.as_ref()?;
    if reasoning.is_empty() {
        return None;
    }
    let id = OutputItemKind::Reasoning.new_id();
    let content = vec![ReasoningText {
        text: reasoning.clone(),
    }];
    let status = Some("completed".to_string());
    let include_encrypted = original_req
        .include
        .as_deref()
        .is_some_and(|f| f.contains(&IncludeField::ReasoningEncryptedContent));
    Some(if include_encrypted {
        ResponseOutputItem::new_reasoning_encrypted(
            id,
            vec![],
            content,
            encode_reasoning_content(reasoning),
            status,
        )
    } else {
        ResponseOutputItem::new_reasoning(id, vec![], content, status)
    })
}

/// The assistant's text as a message item, when it said any; a generation
/// that failed leaves the message in progress.
fn message_output_item(choice: &ChatChoice) -> Option<ResponseOutputItem> {
    let content = choice.message.content.as_ref()?;
    if content.is_empty() {
        return None;
    }
    Some(ResponseOutputItem::Message {
        id: OutputItemKind::Message.new_id(),
        role: "assistant".to_string(),
        content: vec![ResponseContentPart::OutputText {
            text: content.clone(),
            annotations: vec![],
            logprobs: choice.logprobs.clone(),
        }],
        status: if matches!(choice.finish_reason.as_deref(), Some("failed" | "error")) {
            "in_progress"
        } else {
            "completed"
        }
        .to_string(),
        phase: None,
    })
}

/// The assistant's tool calls as output items, in their order; a call to a
/// downgraded custom tool maps back to a `custom_tool_call` item.
fn push_tool_call_output_items(
    output: &mut Vec<ResponseOutputItem>,
    choice: &ChatChoice,
    effective_tools: Option<&[ResponseTool]>,
) {
    let Some(tool_calls) = &choice.message.tool_calls else {
        return;
    };
    let custom_names = custom_tool_names(effective_tools);
    for tool_call in tool_calls {
        let (name, namespace) =
            resolve_function_identity(effective_tools, &tool_call.function.name);
        if custom_names.contains(&tool_call.function.name) {
            output.push(ResponseOutputItem::CustomToolCall {
                call_id: tool_call.id.clone(),
                input: custom_tool_input(
                    tool_call.function.arguments.as_deref().unwrap_or_default(),
                ),
                name,
                id: Some(OutputItemKind::CustomToolCall.new_id()),
                namespace,
            });
            continue;
        }
        output.push(ResponseOutputItem::FunctionToolCall {
            id: Some(OutputItemKind::FunctionCall.new_id()),
            call_id: tool_call.id.clone(),
            name,
            namespace,
            arguments: tool_call.function.arguments.clone().unwrap_or_default(),
            output: None, // Tool execution belongs to the next turn.
            status: function_call_status(
                choice.finish_reason.as_deref(),
                tool_call.function.arguments.as_deref().unwrap_or_default(),
            )
            .to_string(),
        });
    }
}

/// The chat usage as the Responses API reports it, through the protocol's
/// one conversion: the streamed object reads it here as well.
pub(super) fn responses_usage(usage: &Usage) -> ResponsesUsage {
    ResponsesUsage::Modern(ResponseUsage::from(usage))
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/regular/responses/conversions/tests.rs"]
mod tests;
