//! Conversion utilities for translating between /v1/responses and /v1/chat/completions formats
//!
//! This module implements the conversion approach where:
//! 1. ResponsesRequest → ChatCompletionRequest (for backend processing)
//! 2. ChatCompletionResponse → ResponsesResponse (for client response)
//!
//! This allows the gRPC router to reuse the existing chat pipeline infrastructure
//! without requiring Python backend changes.

use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent},
    common::{
        ContentPart, FunctionCallResponse, ImageUrl, JsonSchemaFormat, ResponseFormat, ToolCall,
        ToolChoice, ToolChoiceValue, UsageInfo,
    },
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

use crate::routers::grpc::common::responses::utils::{
    custom_tool_input, custom_tool_names, decode_reasoning_content, encode_reasoning_content,
    extract_tools_from_response_tools, resolve_function_identity,
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
                content: MessageContent::Text(text.clone()),
                name: None,
            });
        }
        ResponseInput::Items(items) => {
            // Structured items → convert each to appropriate chat message.
            // Assistant-side items are merged into one assistant turn by
            // `push_chat_message`; see its doc comment.
            for item in items {
                match item {
                    ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
                        let content = match content {
                            StringOrContentParts::String(s) => MessageContent::Text(s.clone()),
                            StringOrContentParts::Array(parts) => {
                                response_parts_to_message_content(parts)
                            }
                        };
                        push_role_message(&mut messages, role, content);
                    }
                    ResponseInputOutputItem::Message { role, content, .. } => {
                        push_role_message(
                            &mut messages,
                            role,
                            response_parts_to_message_content(content),
                        );
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
                            &mut messages,
                            assistant_tool_call(
                                call_id,
                                name,
                                namespace.as_deref(),
                                arguments.clone(),
                            ),
                        );
                        if let Some(output_text) = output {
                            messages.push(ChatMessage::Tool {
                                content: MessageContent::Text(output_text.clone()),
                                tool_call_id: call_id.clone(),
                            });
                        }
                    }
                    ResponseInputOutputItem::Reasoning {
                        content,
                        encrypted_content,
                        ..
                    } => {
                        // Prefer the plain text; a `store=false` client may hand
                        // back only the opaque blob this gateway produced.
                        let mut reasoning_text = content
                            .iter()
                            .map(|c| match c {
                                ReasoningText { text } => text.as_str(),
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        if reasoning_text.is_empty() {
                            reasoning_text = encrypted_content
                                .as_deref()
                                .and_then(decode_reasoning_content)
                                .unwrap_or_default();
                        }
                        if !reasoning_text.is_empty() {
                            let assistant = ChatMessage::Assistant {
                                content: None,
                                name: None,
                                tool_calls: None,
                                reasoning_content: Some(reasoning_text),
                            };
                            push_chat_message(&mut messages, assistant);
                        }
                    }
                    ResponseInputOutputItem::FunctionCallOutput {
                        call_id, output, ..
                    } => {
                        // Function call output - add as tool message
                        // Note: The function name is looked up from prev_outputs in Harmony path
                        // For Chat path, we just use the call_id
                        messages.push(ChatMessage::Tool {
                            content: MessageContent::Text(output.to_text_only()?),
                            tool_call_id: call_id.clone(),
                        });
                    }
                    ResponseInputOutputItem::McpApprovalResponse { .. }
                    | ResponseInputOutputItem::McpApprovalRequest { .. }
                    | ResponseInputOutputItem::ComputerCall { .. }
                    | ResponseInputOutputItem::ComputerCallOutput { .. }
                    | ResponseInputOutputItem::McpCall { .. }
                    | ResponseInputOutputItem::McpListTools { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "Approval item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    ResponseInputOutputItem::ImageGenerationCall { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "image_generation_call input item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    ResponseInputOutputItem::Compaction { .. }
                    | ResponseInputOutputItem::ItemReference { .. } => {
                        return Err("Unsupported input item type".to_string());
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
                            &mut messages,
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
                        let output_text = match output {
                            CustomToolCallOutputContent::Text(s) => s.clone(),
                            CustomToolCallOutputContent::Parts(parts) => parts
                                .iter()
                                .filter_map(|p| match p {
                                    CustomToolInputContentPart::InputText { text } => {
                                        Some(text.as_str())
                                    }
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n"),
                        };
                        messages.push(ChatMessage::Tool {
                            content: MessageContent::Text(output_text),
                            tool_call_id: call_id.clone(),
                        });
                    }
                    ResponseInputOutputItem::ShellCall { .. }
                    | ResponseInputOutputItem::ShellCallOutput { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "Shell tool item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    ResponseInputOutputItem::ApplyPatchCall { .. }
                    | ResponseInputOutputItem::ApplyPatchCallOutput { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "apply_patch item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
                    }
                    // T5 schema-only: forced-cascade arm, no behavior.
                    ResponseInputOutputItem::LocalShellCall { .. }
                    | ResponseInputOutputItem::LocalShellCallOutput { .. } => {
                        return Err("Unsupported input item type".to_string());
                    }
                }
            }
        }
    }

    // Ensure we have at least one message
    if messages.is_empty() {
        return Err("Request must contain at least one message".to_string());
    }

    // 3. Extract function tools from ResponseTools.
    // MCP tools are merged later by the tool loop (see tool_loop.rs:prepare_chat_tools_and_choice).
    // `tool_choice: none` forbids calls; offering the schemas anyway makes the
    // model re-issue calls on replayed histories, and with parsing disabled the
    // raw call markup leaks into the message text.
    let tool_choice = req.tool_choice.as_ref().map(|tc| tc.to_chat_tool_choice());
    let function_tools = extract_tools_from_response_tools(req.tools.as_deref());
    let tools = if function_tools.is_empty()
        || matches!(tool_choice, Some(ToolChoice::Value(ToolChoiceValue::None)))
    {
        None
    } else {
        Some(function_tools)
    };

    // 4. Build ChatCompletionRequest
    let is_streaming = req.stream.unwrap_or(false);

    Ok(ChatCompletionRequest {
        messages,
        model: if req.model.is_empty() {
            UNKNOWN_MODEL_ID.to_string()
        } else {
            req.model.clone()
        },
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
    })
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
        },
        "assistant" => ChatMessage::Assistant {
            content: Some(content),
            name: None,
            tool_calls: None,
            reasoning_content: None,
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
                status: "completed".to_string(),
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
                output: None, // Tool hasn't been executed yet
                status: "in_progress".to_string(),
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
        Some("tool_calls") => (ResponseStatus::InProgress, None), // Waiting for tool execution
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
    use crate::routers::grpc::common::responses::utils::namespace_test_request;

    #[test]
    fn chat_to_responses_serializes_responses_api_usage() {
        let chat_response = ChatCompletionResponse::builder("chatcmpl_test", "test-model")
            .choices(vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: Some("done".to_string()),
                    tool_calls: None,
                    reasoning_content: None,
                },
                logprobs: None,
                finish_reason: Some("stop".to_string()),
                matched_stop: None,
                hidden_states: None,
            }])
            .usage(
                Usage::from_counts(12, 7)
                    .with_cached_tokens(3)
                    .with_reasoning_tokens(2),
            )
            .build();

        let response = chat_to_responses(
            &chat_response,
            &ResponsesRequest::default(),
            Some("resp_test".to_string()),
        )
        .expect("chat response should convert");
        let wire = serde_json::to_value(response).expect("response should serialize");
        let usage = wire.get("usage").expect("usage should be present");

        assert_eq!(usage.get("input_tokens"), Some(&serde_json::json!(12)));
        assert_eq!(usage.get("output_tokens"), Some(&serde_json::json!(7)));
        assert_eq!(usage.get("total_tokens"), Some(&serde_json::json!(19)));
        assert_eq!(
            usage.pointer("/input_tokens_details/cached_tokens"),
            Some(&serde_json::json!(3))
        );
        assert_eq!(
            usage.pointer("/output_tokens_details/reasoning_tokens"),
            Some(&serde_json::json!(2))
        );
        assert!(usage.get("prompt_tokens").is_none());
        assert!(usage.get("completion_tokens").is_none());
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
}
