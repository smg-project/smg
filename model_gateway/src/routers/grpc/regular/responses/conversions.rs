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
    chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent},
    common::{
        ContentPart, FunctionCallResponse, ImageUrl, JsonSchemaFormat, ResponseFormat, Tool,
        ToolCall, ToolChoice, ToolChoiceValue, UsageInfo,
    },
    profile::ProviderProfile,
    responses::{
        CustomToolCallOutputContent, CustomToolInputContentPart, IncludeField, IncompleteDetails,
        IncompleteReason, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
        ResponseOutputItem, ResponseReasoningContent::ReasoningText, ResponseStatus,
        ResponsesRequest, ResponsesResponse, ResponsesUsage, StringOrContentParts, TextConfig,
        TextFormat,
    },
    UNKNOWN_MODEL_ID,
};
use tracing::warn;

use crate::routers::grpc::common::responses::{
    history::hosted_call_result,
    utils::{
        custom_tool_input, custom_tool_names, decode_reasoning_content, encode_reasoning_content,
        extract_tools_from_response_tools, function_call_status, generation_failure_error,
        resolve_function_identity,
    },
};

/// Convert a ResponsesRequest to ChatCompletionRequest for processing through the chat pipeline
///
/// # Conversion Logic
/// - `input` (text/items) → `messages` (chat messages)
/// - `instructions` → system message (prepended)
/// - `max_output_tokens` → `max_completion_tokens`
/// - `tools` → function tools extracted from ResponseTools
/// - `tool_choice` → passed through from request
/// - Response-specific fields (previous_response_id, conversation) are handled by router
pub(crate) fn responses_to_chat(req: &ResponsesRequest) -> Result<ChatCompletionRequest, String> {
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
) -> Result<ChatCompletionRequest, String> {
    responses_to_chat_with_tools(req, Some(mcp_chat_tools))
}

fn responses_to_chat_with_tools(
    req: &ResponsesRequest,
    mcp_chat_tools: Option<&[Tool]>,
) -> Result<ChatCompletionRequest, String> {
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
        return Err("Request must contain at least one message".to_string());
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
        return Err(format!(
            "No tool call found for function call output with call_id '{call_id}'"
        ));
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
) -> Result<(), String> {
    match item {
        ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
            let content = match content {
                StringOrContentParts::String(s) => MessageContent::Text(s.clone()),
                StringOrContentParts::Array(parts) => response_parts_to_message_content(parts),
            };
            push_role_message(messages, role, content);
        }
        ResponseInputOutputItem::Message { role, content, .. } => {
            push_role_message(messages, role, response_parts_to_message_content(content));
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
                    tool_calls: None,
                    reasoning_content: Some(reasoning_text),
                    ext: Default::default(),
                };
                push_chat_message(messages, assistant);
            }
        }
        ResponseInputOutputItem::FunctionCallOutput {
            call_id, output, ..
        } => {
            // Function call output - add as tool message
            // Note: The function name is looked up from prev_outputs in Harmony path
            // For Chat path, we just use the call_id
            messages.push(tool_result(call_id, output.to_text_only()?));
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
    }
    Ok(())
}

/// The refusal of an item kind this path does not carry, warned about where
/// the kind says why.
fn unsupported(warning: Option<&'static str>) -> Result<(), String> {
    if let Some(warning) = warning {
        warn!(function = "responses_to_chat", "{warning}");
    }
    Err("Unsupported input item type".to_string())
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
    let function_tools = extract_tools_from_response_tools(req.tools.as_deref());
    let function_tools = if matches!(tool_choice, Some(ToolChoice::Value(ToolChoiceValue::None))) {
        Vec::new()
    } else {
        function_tools
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
/// text and refusals become text parts, `input_image` becomes an `image_url`
/// part, and `input_file` is dropped (no chat equivalent on this path).
fn response_parts_to_message_content(content: &[ResponseContentPart]) -> MessageContent {
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
                        url: url.clone(),
                        detail: None,
                        max_long_side_pixel: None,
                    },
                });
            }
            ResponseContentPart::InputImage { .. } | ResponseContentPart::InputFile { .. } => {}
        }
    }
    // Plain text stays a string, as it did before parts were carried at all.
    match parts.as_slice() {
        [] => MessageContent::Text(String::new()),
        [ContentPart::Text { text }] => MessageContent::Text(text.clone()),
        _ => MessageContent::Parts(parts),
    }
}

/// Append a role-tagged input message. A `developer` turn folds into the
/// leading system message: OpenAI ranks it with `system`, above `user`, and
/// appended unlabelled it reads as one more conversational turn that the
/// latest user message is free to override.
fn push_role_message(messages: &mut Vec<ChatMessage>, role: &str, content: MessageContent) {
    if role != "developer" {
        push_chat_message(messages, role_to_chat_message(role, content));
        return;
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
        reasoning_content: None,
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
                reasoning_content,
                ..
            }),
            ChatMessage::Assistant {
                content: new_content,
                tool_calls: new_tool_calls,
                reasoning_content: new_reasoning,
                ..
            },
        ) => {
            if let Some(new_reasoning) = new_reasoning {
                let merged = reasoning_content.get_or_insert_with(String::new);
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
            tool_calls: None,
            reasoning_content: None,
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
            description: _,
            strict,
        } => Some(ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name: name.clone(),
                schema: schema.clone(),
                strict: *strict,
            },
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
    response_id_override: Option<String>,
) -> Result<ResponsesResponse, String> {
    // Extract the first choice (responses API doesn't support n>1)
    let choice = chat_resp
        .choices
        .first()
        .ok_or_else(|| "Chat response contains no choices".to_string())?;

    // Convert assistant message to output items. Reasoning comes first (OpenAI
    // order), so items replayed in emitted order rebuild the same turn.
    let mut output: Vec<ResponseOutputItem> = Vec::new();

    if let Some(reasoning) = &choice.message.reasoning_content {
        if !reasoning.is_empty() {
            let id = format!("reasoning_{}", chat_resp.id);
            let content = vec![ReasoningText {
                text: reasoning.clone(),
            }];
            let status = Some("completed".to_string());
            let include_encrypted = original_req
                .include
                .as_deref()
                .is_some_and(|f| f.contains(&IncludeField::ReasoningEncryptedContent));
            output.push(if include_encrypted {
                ResponseOutputItem::new_reasoning_encrypted(
                    id,
                    vec![],
                    content,
                    encode_reasoning_content(reasoning),
                    status,
                )
            } else {
                ResponseOutputItem::new_reasoning(id, vec![], content, status)
            });
        }
    }

    // Convert message content to output item
    if let Some(content) = &choice.message.content {
        if !content.is_empty() {
            output.push(ResponseOutputItem::Message {
                id: format!("msg_{}", chat_resp.id),
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
            });
        }
    }

    // Convert tool calls if present. Calls to downgraded custom tools map back
    // to `custom_tool_call` items.
    let custom_names = custom_tool_names(original_req.tools.as_deref());
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            let (name, namespace) =
                resolve_function_identity(original_req.tools.as_deref(), &tool_call.function.name);
            if custom_names.contains(&tool_call.function.name) {
                output.push(ResponseOutputItem::CustomToolCall {
                    call_id: tool_call.id.clone(),
                    input: custom_tool_input(
                        tool_call.function.arguments.as_deref().unwrap_or_default(),
                    ),
                    name,
                    id: Some(tool_call.id.clone()),
                    namespace,
                });
                continue;
            }
            output.push(ResponseOutputItem::FunctionToolCall {
                id: Some(tool_call.id.clone()),
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

    // Determine response status based on finish_reason. "length" is a
    // max_output_tokens truncation, which the Responses contract reports as
    // status=incomplete with incomplete_details.
    let (status, incomplete_details) = match choice.finish_reason.as_deref() {
        Some("stop") => (ResponseStatus::Completed, None),
        Some("length") => (
            ResponseStatus::Incomplete,
            Some(IncompleteDetails {
                reason: IncompleteReason::MaxOutputTokens,
            }),
        ),
        Some("tool_calls") => (ResponseStatus::Completed, None),
        Some("failed") | Some("error") => (ResponseStatus::Failed, None),
        _ => (ResponseStatus::Completed, None), // Default to completed
    };

    // Convert usage from Usage to UsageInfo, then wrap in ResponsesUsage
    let usage = chat_resp.usage.as_ref().map(|u| {
        let usage_info = UsageInfo {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            reasoning_tokens: u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            prompt_tokens_details: u.prompt_tokens_details.clone(),
        };
        ResponsesUsage::Modern(usage_info.to_response_usage())
    });

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

#[cfg(test)]
mod tests {
    use openai_protocol::{
        chat::{ChatChoice, ChatCompletionMessage},
        common::{StreamOptions, Usage},
        responses::ReasoningEffort,
    };

    use super::*;
    use crate::routers::grpc::common::responses::utils::{
        assert_responses_api_usage, namespace_test_request, responses_api_test_usage,
    };

    #[test]
    fn chat_to_responses_serializes_responses_api_usage() {
        let chat_response = ChatCompletionResponse::builder("chatcmpl_test", "test-model")
            .choices(vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: Some("done".to_string()),
                    refusal: None,
                    annotations: Vec::new(),
                    tool_calls: None,
                    reasoning_content: None,
                },
                logprobs: None,
                finish_reason: Some("stop".to_string()),
                matched_stop: None,
                hidden_states: None,
            }])
            .usage(responses_api_test_usage())
            .build();

        let response = chat_to_responses(
            &chat_response,
            &ResponsesRequest::default(),
            Some("resp_test".to_string()),
        )
        .expect("chat response should convert");
        let wire = serde_json::to_value(response).expect("response should serialize");
        assert_responses_api_usage(&wire);
    }

    /// The public API returns these fields on every Response object (the
    /// schema marks `error`, `instructions`, `top_p` and both usage detail
    /// blocks required); a plain chat completion without reasoning or cache
    /// hits must still produce all of them.
    #[test]
    fn chat_to_responses_carries_the_always_present_fields() {
        let chat_response = ChatCompletionResponse::builder("chatcmpl_plain", "test-model")
            .choices(vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: Some("done".to_string()),
                    refusal: None,
                    annotations: Vec::new(),
                    tool_calls: None,
                    reasoning_content: None,
                },
                logprobs: None,
                finish_reason: Some("stop".to_string()),
                matched_stop: None,
                hidden_states: None,
            }])
            .usage(Usage::from_counts(5, 3))
            .build();

        let response = chat_to_responses(
            &chat_response,
            &ResponsesRequest::default(),
            Some("resp_plain".to_string()),
        )
        .expect("chat response should convert");
        let wire = serde_json::to_value(response).expect("response should serialize");
        let body = wire.as_object().expect("object");

        for key in [
            "error",
            "instructions",
            "incomplete_details",
            "previous_response_id",
            "user",
            "safety_identifier",
            "prompt_cache_key",
            "prompt_cache_retention",
            "max_tool_calls",
        ] {
            assert_eq!(
                body.get(key),
                Some(&serde_json::Value::Null),
                "{key} must be present and null"
            );
        }
        assert_eq!(body["top_p"], serde_json::json!(1.0));
        assert_eq!(body["truncation"], "disabled");
        assert_eq!(body["service_tier"], "default");
        assert_eq!(body["top_logprobs"], 0);
        assert_eq!(body["background"], false);
        assert_eq!(body["frequency_penalty"], serde_json::json!(0.0));
        assert_eq!(body["presence_penalty"], serde_json::json!(0.0));
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
            "completed responses are stamped"
        );
        assert_eq!(
            body["usage"]["input_tokens_details"],
            serde_json::json!({"cached_tokens": 0})
        );
        assert_eq!(
            body["usage"]["output_tokens_details"],
            serde_json::json!({"reasoning_tokens": 0})
        );
    }

    /// A Responses request for `model`, from its input items.
    fn items_request(model: &str, input: serde_json::Value) -> ResponsesRequest {
        serde_json::from_value(serde_json::json!({"model": model, "input": input}))
            .expect("a responses request")
    }

    /// The conversion's refusal of `input` for a model under the DeepSeek
    /// contract.
    fn refused_under_deepseek(input: serde_json::Value) -> String {
        responses_to_chat(&items_request("deepseek-flash", input))
            .expect_err("a tool result without its call is a malformed history")
    }

    /// The call ids of the assistant turn at `index` of a converted history.
    fn issued_call_ids(chat: &ChatCompletionRequest, index: usize) -> Vec<&str> {
        match &chat.messages[index] {
            ChatMessage::Assistant {
                tool_calls: Some(calls),
                ..
            } => calls.iter().map(|call| call.id.as_str()).collect(),
            other => panic!("expected an assistant turn with calls, got {other:?}"),
        }
    }

    /// The call id the tool message at `index` of a converted history answers.
    fn answered_call_id(chat: &ChatCompletionRequest, index: usize) -> &str {
        match &chat.messages[index] {
            ChatMessage::Tool { tool_call_id, .. } => tool_call_id,
            other => panic!("expected a tool message, got {other:?}"),
        }
    }

    /// A tool result under a call id the previous turn did not issue is a
    /// malformed history under the DeepSeek contract (as under OpenAI's): the
    /// conversion refuses it in the public API's words, naming the id; the
    /// right id converts, and a model under no such contract converts either
    /// way.
    #[test]
    fn a_tool_result_under_an_unknown_call_id_does_not_convert_under_the_deepseek_contract() {
        let with_call = |call_id: &str| {
            serde_json::json!([
                {"type": "message", "role": "user", "content": "Add 17 and 25."},
                {"type": "function_call", "call_id": "call_1", "name": "add_integers",
                 "arguments": "{\"a\":17,\"b\":25}"},
                {"type": "function_call_output", "call_id": call_id, "output": "42"}
            ])
        };
        assert_eq!(
            refused_under_deepseek(with_call("call_doesnotexist")),
            "No tool call found for function call output with call_id 'call_doesnotexist'"
        );
        assert!(responses_to_chat(&items_request("deepseek-flash", with_call("call_1"))).is_ok());
        assert!(responses_to_chat(&items_request(
            "local-model",
            with_call("call_doesnotexist")
        ))
        .is_ok());
    }

    /// A result with no call before it answers nothing: refused in the same
    /// words after the user's message and as the only item. The stored turns
    /// a request continues arrive with their calls, and the loader drops a
    /// stored result whose call did not survive the reload, so such a result
    /// is the client's, not a reload that lost the call; a model under no
    /// such contract converts it either way.
    #[test]
    fn an_orphan_tool_result_does_not_convert_under_the_deepseek_contract() {
        let after_user = serde_json::json!([
            {"type": "message", "role": "user", "content": "Add 17 and 25."},
            {"type": "function_call_output", "call_id": "call_a", "output": "42"}
        ]);
        let alone = serde_json::json!([
            {"type": "function_call_output", "call_id": "call_a", "output": "42"}
        ]);
        for input in [after_user, alone] {
            assert_eq!(
                refused_under_deepseek(input.clone()),
                "No tool call found for function call output with call_id 'call_a'"
            );
            assert!(responses_to_chat(&items_request("local-model", input)).is_ok());
        }
    }

    /// A result ahead of the call it names, or ahead of a later call and its
    /// result, has no call before it either.
    #[test]
    fn a_tool_result_ahead_of_its_call_does_not_convert_under_the_deepseek_contract() {
        let before_its_call = serde_json::json!([
            {"type": "function_call_output", "call_id": "call_a", "output": "42"},
            {"type": "function_call", "call_id": "call_a", "name": "add_integers",
             "arguments": "{\"a\":17,\"b\":25}"}
        ]);
        let before_a_later_call = serde_json::json!([
            {"type": "function_call_output", "call_id": "call_a", "output": "42"},
            {"type": "function_call", "call_id": "call_m", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_m", "output": "ok"}
        ]);
        for input in [before_its_call, before_a_later_call] {
            assert_eq!(
                refused_under_deepseek(input),
                "No tool call found for function call output with call_id 'call_a'"
            );
        }
    }

    /// The call and the result that answers it convert to the assistant's
    /// call and the tool message under its id.
    #[test]
    fn a_matching_call_and_result_convert_under_the_deepseek_contract() {
        let chat = responses_to_chat(&items_request(
            "deepseek-flash",
            serde_json::json!([
                {"type": "message", "role": "user", "content": "Add 17 and 25."},
                {"type": "function_call", "call_id": "call_a", "name": "add_integers",
                 "arguments": "{\"a\":17,\"b\":25}"},
                {"type": "function_call_output", "call_id": "call_a", "output": "42"}
            ]),
        ))
        .expect("a paired result converts");
        assert_eq!(chat.messages.len(), 3);
        assert_eq!(issued_call_ids(&chat, 1), ["call_a"]);
        assert_eq!(answered_call_id(&chat, 2), "call_a");
    }

    /// Two calls in one turn and a result for one of them: the result answers
    /// a call the turn issued, so it converts, both calls in the one assistant
    /// turn (the call still without a result is the model's to answer, not
    /// this rule's subject); a result for neither call is refused.
    #[test]
    fn a_result_for_one_of_two_calls_converts_under_the_deepseek_contract() {
        let two_calls = |call_id: &str| {
            serde_json::json!([
                {"type": "message", "role": "user", "content": "Add 17 and 25, then 3 and 4."},
                {"type": "function_call", "call_id": "call_a", "name": "add_integers",
                 "arguments": "{\"a\":17,\"b\":25}"},
                {"type": "function_call", "call_id": "call_b", "name": "add_integers",
                 "arguments": "{\"a\":3,\"b\":4}"},
                {"type": "function_call_output", "call_id": call_id, "output": "7"}
            ])
        };
        let chat = responses_to_chat(&items_request("deepseek-flash", two_calls("call_b")))
            .expect("a result for one of the turn's calls converts");
        assert_eq!(chat.messages.len(), 3);
        assert_eq!(issued_call_ids(&chat, 1), ["call_a", "call_b"]);
        assert_eq!(answered_call_id(&chat, 2), "call_b");
        assert_eq!(
            refused_under_deepseek(two_calls("call_c")),
            "No tool call found for function call output with call_id 'call_c'"
        );
    }

    #[test]
    fn test_text_input_conversion() {
        let req = ResponsesRequest {
            input: ResponseInput::Text("Hello, world!".to_string()),
            instructions: Some("You are a helpful assistant.".to_string()),
            model: "gpt-4".to_string(),
            temperature: Some(0.7),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2); // system + user
        assert_eq!(chat_req.model, "gpt-4");
        assert_eq!(chat_req.temperature, Some(0.7));
    }

    #[test]
    fn test_reasoning_effort_flows_through() {
        use openai_protocol::responses::ResponseReasoningParam;

        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            reasoning: Some(ResponseReasoningParam {
                effort: Some(ReasoningEffort::High),
                summary: None,
            }),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.reasoning_effort.as_deref(), Some("high"));
    }

    /// The outer OpenAI tiers reach the Chat pipeline verbatim, where `none`
    /// already means thinking off.
    #[test]
    fn test_reasoning_effort_outer_tiers_flow_through() {
        use openai_protocol::responses::ResponseReasoningParam;

        for effort in [
            ReasoningEffort::None,
            ReasoningEffort::Xhigh,
            ReasoningEffort::Max,
        ] {
            let req = ResponsesRequest {
                input: ResponseInput::Text("hi".to_string()),
                reasoning: Some(ResponseReasoningParam {
                    effort: Some(effort),
                    summary: None,
                }),
                ..Default::default()
            };

            let chat_req = responses_to_chat(&req).unwrap();
            assert_eq!(chat_req.reasoning_effort.as_deref(), Some(effort.as_str()));
        }
    }

    #[test]
    fn test_reasoning_effort_absent_when_reasoning_none() {
        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.reasoning_effort, None);
    }

    #[test]
    fn test_items_input_conversion() {
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![
                ResponseInputOutputItem::Message {
                    id: "msg_1".to_string(),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText {
                        text: "Hello!".to_string(),
                    }],
                    status: None,
                    phase: None,
                },
                ResponseInputOutputItem::Message {
                    id: "msg_2".to_string(),
                    role: "assistant".to_string(),
                    content: vec![ResponseContentPart::OutputText {
                        text: "Hi there!".to_string(),
                        annotations: vec![],
                        logprobs: None,
                    }],
                    status: None,
                    phase: None,
                },
            ]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2); // user + assistant
    }

    #[test]
    fn test_function_call_history_uses_call_id_for_chat_tool_messages() {
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionToolCall {
                id: Some("fc_item_id".to_string()),
                call_id: "call_tool_id".to_string(),
                name: "lookup".to_string(),
                namespace: Some("weather".to_string()),
                arguments: "{\"q\":\"rust\"}".to_string(),
                output: Some("done".to_string()),
                status: Some("completed".to_string()),
            }]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2);

        match &chat_req.messages[0] {
            ChatMessage::Assistant {
                tool_calls: Some(tool_calls),
                ..
            } => {
                assert_eq!(tool_calls[0].id, "call_tool_id");
                assert_eq!(tool_calls[0].function.name, "weather.lookup");
            }
            other => panic!("expected assistant tool call, got {other:?}"),
        }

        match &chat_req.messages[1] {
            ChatMessage::Tool { tool_call_id, .. } => {
                assert_eq!(tool_call_id, "call_tool_id");
            }
            other => panic!("expected tool message, got {other:?}"),
        }
    }

    /// A hosted tool call of a stored turn replays as the call and its result;
    /// the tools listing stored next to it says nothing to the model.
    #[test]
    fn stored_hosted_tool_calls_replay_as_a_call_and_its_result() {
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "input": [
                {"role": "user", "content": "search it"},
                {"type": "mcp_list_tools", "id": "mcpl_1", "server_label": "wiki", "tools": []},
                {"type": "mcp_call", "id": "mcp_1", "name": "search", "server_label": "wiki",
                 "arguments": "{\"q\":\"rust\"}", "output": "found", "status": "completed"},
                {"type": "message", "id": "msg_a", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Found it.", "annotations": []}]}
            ]
        }))
        .unwrap();

        let chat_req = responses_to_chat(&req).unwrap();

        assert_eq!(chat_req.messages.len(), 4, "{:?}", chat_req.messages);
        match &chat_req.messages[1] {
            ChatMessage::Assistant {
                tool_calls: Some(tool_calls),
                ..
            } => {
                assert_eq!(tool_calls[0].id, "mcp_1");
                assert_eq!(tool_calls[0].function.name, "search");
                assert_eq!(
                    tool_calls[0].function.arguments.as_deref(),
                    Some("{\"q\":\"rust\"}")
                );
            }
            other => panic!("expected the assistant's call, got {other:?}"),
        }
        match &chat_req.messages[2] {
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => {
                assert_eq!(tool_call_id, "mcp_1");
                assert!(matches!(content, MessageContent::Text(text) if text == "found"));
            }
            other => panic!("expected the call's result, got {other:?}"),
        }
    }

    /// A hosted tool call that failed stores an empty output and the failure
    /// under `error`: it replays as the call and a result that says the call
    /// failed, not as a successful empty result.
    #[test]
    fn a_failed_stored_hosted_tool_call_replays_its_error_as_the_result() {
        for (error, expected) in [
            (
                serde_json::json!({"type": "mcp_tool_execution_error", "content": "wiki is down"}),
                "The tool call failed: wiki is down",
            ),
            (
                serde_json::json!({"type": "http_error", "code": 503, "message": "unavailable"}),
                "The tool call failed (HTTP 503): unavailable",
            ),
            (
                serde_json::json!("legacy text"),
                "The tool call failed: legacy text",
            ),
        ] {
            let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
                "model": "m",
                "input": [
                    {"role": "user", "content": "search it"},
                    {"type": "mcp_call", "id": "mcp_1", "name": "search", "server_label": "wiki",
                     "arguments": "{}", "output": "", "status": "failed", "error": error}
                ]
            }))
            .unwrap();
            let chat_req = responses_to_chat(&req).unwrap();
            assert_eq!(chat_req.messages.len(), 3, "{:?}", chat_req.messages);
            match &chat_req.messages[2] {
                ChatMessage::Tool {
                    tool_call_id,
                    content,
                } => {
                    assert_eq!(tool_call_id, "mcp_1");
                    assert!(
                        matches!(content, MessageContent::Text(text) if text == expected),
                        "{content:?}"
                    );
                }
                other => panic!("expected the failure as the result, got {other:?}"),
            }
        }
        // A failed call with no error text still says it failed.
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "input": [
                {"role": "user", "content": "search it"},
                {"type": "mcp_call", "id": "mcp_1", "name": "search", "server_label": "wiki",
                 "arguments": "{}", "status": "failed"}
            ]
        }))
        .unwrap();
        let chat_req = responses_to_chat(&req).unwrap();
        assert!(
            matches!(&chat_req.messages[2], ChatMessage::Tool { content: MessageContent::Text(text), .. } if text == "The tool call failed."),
            "{:?}",
            chat_req.messages
        );
    }

    #[test]
    fn test_empty_input_error() {
        let req = ResponsesRequest {
            input: ResponseInput::Text(String::new()),
            ..Default::default()
        };

        // Empty text should still create a user message, so this should succeed
        let result = responses_to_chat(&req);
        assert!(result.is_ok());
    }

    #[test]
    fn test_stream_options_include_obfuscation_roundtrip() {
        // Regression: ensure caller-provided stream_options (e.g. `include_obfuscation`)
        // are preserved through the Responses → Chat conversion when streaming.
        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: None,
                include_obfuscation: Some(false),
                ..StreamOptions::default()
            }),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert!(chat_req.stream);
        let opts = chat_req
            .stream_options
            .expect("stream_options populated when streaming");
        // Caller-provided value is preserved verbatim.
        assert_eq!(opts.include_obfuscation, Some(false));
        // include_usage defaults to true when absent so downstream consumers
        // still emit the usage block at end-of-stream.
        assert_eq!(opts.include_usage, Some(true));
    }

    #[test]
    fn test_stream_options_caller_include_usage_preserved() {
        // Caller-set `include_usage` must not be clobbered by the conversion layer.
        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: Some(false),
                include_obfuscation: Some(true),
                ..StreamOptions::default()
            }),
            ..Default::default()
        };

        let opts = responses_to_chat(&req).unwrap().stream_options.unwrap();
        assert_eq!(opts.include_usage, Some(false));
        assert_eq!(opts.include_obfuscation, Some(true));
    }

    #[test]
    fn test_stream_options_unknown_fields_survive_grpc_chat_builder() {
        // Regression: the gRPC path rebuilds a ChatCompletionRequest field by
        // field, so engine-specific streaming options carried in the catch-all
        // map must survive the conversion and reach the SSE assembler, which is
        // what renders usage chunks in gRPC mode.
        let mut extra = serde_json::Map::new();
        extra.insert(
            "step_usage_chunks".to_string(),
            serde_json::Value::String("all".to_string()),
        );

        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
                other: extra,
                ..StreamOptions::default()
            }),
            ..Default::default()
        };

        let opts = responses_to_chat(&req).unwrap().stream_options.unwrap();
        assert_eq!(
            opts.other.get("step_usage_chunks").and_then(|v| v.as_str()),
            Some("all")
        );
    }

    #[test]
    fn test_stream_options_non_streaming_dropped() {
        // stream=false must produce None stream_options even if caller set it.
        let req = ResponsesRequest {
            input: ResponseInput::Text("hi".to_string()),
            stream: Some(false),
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
                include_obfuscation: Some(false),
                ..StreamOptions::default()
            }),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert!(!chat_req.stream);
        assert!(chat_req.stream_options.is_none());
    }

    #[test]
    fn test_image_generation_call_input_rejected() {
        // Regression: `image_generation_call` items are server-produced
        // output (populated via the shared MCP transformer) and must not
        // be round-tripped back into the chat conversion as input.
        // The regular gRPC path — used by non-Harmony text LLMs that only do
        // function calling — rejects this variant with the same contract as
        // sibling hosted-tool items (Computer/Shell/Custom/ApplyPatch).
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::ImageGenerationCall {
                id: "ig_test".to_string(),
                action: None,
                background: None,
                output_format: None,
                quality: None,
                result: Some("base64data".to_string()),
                revised_prompt: Some("a cat".to_string()),
                size: None,
                status: None,
            }]),
            ..Default::default()
        };

        let result = responses_to_chat(&req);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Unsupported input item type");
    }
    #[test]
    fn namespace_chat_response_roundtrips_identity() {
        let request: ResponsesRequest = namespace_test_request();
        let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
            "choices":[{"index":0,"message":{"role":"assistant","tool_calls":[
                {"id":"call_weather","type":"function","function":{"name":"weather.lookup","arguments":"{}"}}
            ]},"finish_reason":"tool_calls"}]
        })).unwrap();
        let response = chat_to_responses(&chat, &request, None).unwrap();
        let wire = serde_json::to_value(&response.output[0]).unwrap();
        assert_eq!(wire["name"], "lookup");
        assert_eq!(wire["namespace"], "weather");
        let mut replay = request;
        replay.input = ResponseInput::Items(vec![serde_json::from_value(wire).unwrap()]);
        let converted = responses_to_chat(&replay).unwrap();
        let wire = serde_json::to_value(converted).unwrap();
        assert_eq!(
            wire["messages"][0]["tool_calls"][0]["function"]["name"],
            "weather.lookup"
        );
        assert_eq!(wire["tools"][0]["function"]["name"], "weather.lookup");
    }
    #[test]
    fn generated_tool_calls_complete_the_response_without_executing_tools() {
        let request = ResponsesRequest::default();
        let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
            "choices":[{"index":0,"message":{"role":"assistant","tool_calls":[
                {"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{}"}},
                {"id":"call_time","type":"function","function":{"name":"time","arguments":"{}"}}
            ]},"finish_reason":"tool_calls"}]
        })).unwrap();
        let wire = serde_json::to_value(chat_to_responses(&chat, &request, None).unwrap()).unwrap();
        assert_eq!(wire["status"], "completed");
        assert_eq!(wire["output"].as_array().unwrap().len(), 2);
        for (item, call_id) in wire["output"]
            .as_array()
            .unwrap()
            .iter()
            .zip(["call_weather", "call_time"])
        {
            assert_eq!(item["status"], "completed");
            assert_eq!(item["call_id"], call_id);
            assert_eq!(item["arguments"], "{}");
        }
    }

    #[test]
    fn failed_generation_does_not_complete_partial_messages_or_tool_calls() {
        for finish_reason in ["failed", "error"] {
            let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
                "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
                "choices":[{"index":0,"message":{"role":"assistant","content":"Partial reply","tool_calls":[
                    {"id":"call_weather","type":"function","function":{
                        "name":"weather","arguments":"{\"city\":"
                    }}
                ]},"finish_reason":finish_reason}]
            }))
            .unwrap();
            let response = chat_to_responses(&chat, &ResponsesRequest::default(), None).unwrap();
            let wire = serde_json::to_value(response).unwrap();
            assert_eq!(wire["status"], "failed", "{finish_reason}");
            assert_eq!(wire["output"].as_array().unwrap().len(), 2);
            assert_eq!(wire["output"][0]["type"], "message");
            assert_eq!(
                wire["output"][0]["status"], "in_progress",
                "{finish_reason}"
            );
            assert_eq!(wire["output"][0]["content"][0]["text"], "Partial reply");
            let item = &wire["output"][1];
            assert_eq!(item["status"], "in_progress", "{finish_reason}");
            assert_eq!(item["call_id"], "call_weather");
            assert_eq!(item["arguments"], "{\"city\":");
        }
    }

    #[test]
    fn length_truncated_tool_call_preserves_earlier_completed_calls() {
        let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
            "choices":[{"index":0,"message":{"role":"assistant","tool_calls":[
                {"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{}"}},
                {"id":"call_time","type":"function","function":{"name":"time","arguments":"{\"tz\":"}}
            ]},"finish_reason":"length"}]
        })).unwrap();
        let response = chat_to_responses(&chat, &ResponsesRequest::default(), None).unwrap();
        let wire = serde_json::to_value(response).unwrap();
        assert_eq!(wire["status"], "incomplete");
        assert_eq!(wire["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(wire["output"].as_array().unwrap().len(), 2);
        assert_eq!(wire["output"][0]["status"], "completed");
        assert_eq!(wire["output"][0]["arguments"], "{}");
        assert_eq!(wire["output"][1]["status"], "incomplete");
        assert_eq!(wire["output"][1]["call_id"], "call_time");
        assert_eq!(wire["output"][1]["arguments"], "{\"tz\":");
    }

    #[test]
    fn length_finish_preserves_a_fully_parsed_tool_call() {
        let chat: ChatCompletionResponse = serde_json::from_value(serde_json::json!({
            "id":"chat_test","object":"chat.completion","created":0,"model":"test-model",
            "choices":[{"index":0,"message":{"role":"assistant","tool_calls":[
                {"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{}"}}
            ]},"finish_reason":"length"}]
        })).unwrap();
        let wire = serde_json::to_value(
            chat_to_responses(&chat, &ResponsesRequest::default(), None).unwrap(),
        )
        .unwrap();
        assert_eq!(wire["status"], "incomplete");
        assert_eq!(wire["output"][0]["status"], "completed");
        assert_eq!(wire["output"][0]["arguments"], "{}");
    }

    /// The MCP tool loop's conversion: the request's function tools and the
    /// MCP tools are one list (nothing is wrapped twice), `tool_choice: none`
    /// withholds the request's own tools and keeps the MCP tools offered, and a
    /// request without either still carries an (empty) list, as the loop has
    /// always sent one; the plain conversion carries none.
    #[test]
    fn the_tool_loop_offers_the_mcp_tools_in_one_list_with_the_request_s_function_tools() {
        use openai_protocol::responses::{ResponsesToolChoice, ToolChoiceOptions};
        let mcp: Tool = serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {"name": "mcp_echo", "parameters": {"type": "object", "properties": {}}}
        }))
        .unwrap();
        let names = |chat: &ChatCompletionRequest| {
            chat.tools.as_deref().map(|tools| {
                tools
                    .iter()
                    .map(|tool| tool.function.name.clone())
                    .collect::<Vec<_>>()
            })
        };

        let request = namespace_test_request();
        let chat = responses_to_chat_with_mcp_tools(&request, std::slice::from_ref(&mcp)).unwrap();
        assert_eq!(
            names(&chat).unwrap(),
            ["weather.lookup", "travel.lookup", "mcp_echo"]
        );

        let mut forbidden = namespace_test_request();
        forbidden.tool_choice = Some(ResponsesToolChoice::Options(ToolChoiceOptions::None));
        let chat =
            responses_to_chat_with_mcp_tools(&forbidden, std::slice::from_ref(&mcp)).unwrap();
        assert_eq!(names(&chat).unwrap(), ["mcp_echo"]);

        let bare: ResponsesRequest =
            serde_json::from_value(serde_json::json!({"model": "test-model", "input": "hi"}))
                .unwrap();
        assert_eq!(
            names(&responses_to_chat_with_mcp_tools(&bare, &[]).unwrap()),
            Some(Vec::new())
        );
        assert_eq!(names(&responses_to_chat(&bare).unwrap()), None);
    }
}
