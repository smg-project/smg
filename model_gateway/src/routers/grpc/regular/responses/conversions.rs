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
    common::{FunctionCallResponse, JsonSchemaFormat, ResponseFormat, ToolCall, UsageInfo},
    responses::{
        ReasoningEffort, ResponseContentPart, ResponseInput, ResponseInputOutputItem,
        ResponseOutputItem, ResponseReasoningContent::ReasoningText, ResponseStatus,
        ResponsesRequest, ResponsesResponse, ResponsesUsage, StringOrContentParts, TextConfig,
        TextFormat,
    },
    UNKNOWN_MODEL_ID,
};
use tracing::warn;

use crate::routers::grpc::common::responses::utils::extract_tools_from_response_tools;

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
            // Structured items → convert each to appropriate chat message
            for item in items {
                match item {
                    ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
                        // Convert SimpleInputMessage to chat message
                        let text = match content {
                            StringOrContentParts::String(s) => s.clone(),
                            StringOrContentParts::Array(parts) => {
                                // Extract text from content parts (only InputText supported)
                                parts
                                    .iter()
                                    .filter_map(|part| match part {
                                        ResponseContentPart::InputText { text } => {
                                            Some(text.as_str())
                                        }
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            }
                        };

                        messages.push(role_to_chat_message(role.as_str(), text));
                    }
                    ResponseInputOutputItem::Message { role, content, .. } => {
                        // Extract text from content parts
                        let text = extract_text_from_content(content);

                        messages.push(role_to_chat_message(role.as_str(), text));
                    }
                    ResponseInputOutputItem::FunctionToolCall {
                        call_id,
                        name,
                        arguments,
                        output,
                        ..
                    } => {
                        // Tool call from history - add as assistant message with tool call
                        // followed by tool response if output exists
                        let tool_call_id = call_id.clone();

                        // Add assistant message with tool_calls (the LLM's decision)
                        messages.push(ChatMessage::Assistant {
                            content: None,
                            name: None,
                            tool_calls: Some(vec![ToolCall {
                                id: tool_call_id.clone(),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name: name.clone(),
                                    arguments: Some(arguments.clone()),
                                },
                            }]),
                            reasoning_content: None,
                        });

                        // Add tool result message if output exists
                        if let Some(output_text) = output {
                            messages.push(ChatMessage::Tool {
                                content: MessageContent::Text(output_text.clone()),
                                tool_call_id,
                            });
                        }
                    }
                    ResponseInputOutputItem::Reasoning { content, .. } => {
                        // Reasoning content - add as assistant message with reasoning_content
                        let reasoning_text = content
                            .iter()
                            .map(|c| match c {
                                ReasoningText { text } => text.as_str(),
                            })
                            .collect::<Vec<_>>()
                            .join("\n");

                        messages.push(ChatMessage::Assistant {
                            content: None,
                            name: None,
                            tool_calls: None,
                            reasoning_content: Some(reasoning_text),
                        });
                    }
                    ResponseInputOutputItem::FunctionCallOutput {
                        call_id, output, ..
                    } => {
                        // Function call output - add as tool message
                        // Note: The function name is looked up from prev_outputs in Harmony path
                        // For Chat path, we just use the call_id
                        messages.push(ChatMessage::Tool {
                            content: MessageContent::Text(output.clone()),
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
                    ResponseInputOutputItem::CustomToolCall { .. }
                    | ResponseInputOutputItem::CustomToolCallOutput { .. } => {
                        warn!(
                            function = "responses_to_chat",
                            "Custom tool item reached chat conversion"
                        );
                        return Err("Unsupported input item type".to_string());
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
    let function_tools = extract_tools_from_response_tools(req.tools.as_deref());
    let tools = if function_tools.is_empty() {
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
        tool_choice: req.tool_choice.as_ref().map(|tc| tc.to_chat_tool_choice()),
        response_format: map_text_to_response_format(req.text.as_ref()),
        reasoning_effort: req
            .reasoning
            .as_ref()
            .and_then(|r| r.effort.as_ref())
            .map(reasoning_effort_to_str)
            .map(str::to_string),
        ..Default::default()
    })
}

/// Map the Responses `reasoning.effort` enum to the Chat `reasoning_effort`
/// string (verbatim snake_case, as the Chat pipeline expects).
fn reasoning_effort_to_str(effort: &ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
    }
}

/// Extract text content from ResponseContentPart array. `Refusal` is
/// losslessly representable as text and is preserved verbatim. Image / file
/// parts are currently dropped; the gRPC regular path is text-only and
/// relies on the multimodal pipeline for media handling (R1/R2/R3 will
/// implement full media handling).
fn extract_text_from_content(content: &[ResponseContentPart]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            ResponseContentPart::InputText { text } => Some(text.as_str()),
            ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
            ResponseContentPart::Refusal { refusal } => Some(refusal.as_str()),
            // R1/R2/R3 will implement full media handling
            ResponseContentPart::InputImage { .. } | ResponseContentPart::InputFile { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Convert role and text to ChatMessage
fn role_to_chat_message(role: &str, text: String) -> ChatMessage {
    match role {
        "user" => ChatMessage::User {
            content: MessageContent::Text(text),
            name: None,
        },
        "assistant" => ChatMessage::Assistant {
            content: Some(MessageContent::Text(text)),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        },
        "system" => ChatMessage::System {
            content: MessageContent::Text(text),
            name: None,
        },
        _ => {
            // Unknown role, treat as user message
            ChatMessage::User {
                content: MessageContent::Text(text),
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

    // Convert assistant message to output items
    let mut output: Vec<ResponseOutputItem> = Vec::new();

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

    // Convert reasoning content if present (O1-style models)
    if let Some(reasoning) = &choice.message.reasoning_content {
        if !reasoning.is_empty() {
            output.push(ResponseOutputItem::new_reasoning(
                format!("reasoning_{}", chat_resp.id),
                vec![],
                vec![ReasoningText {
                    text: reasoning.clone(),
                }],
                Some("completed".to_string()),
            ));
        }
    }

    // Convert tool calls if present
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            output.push(ResponseOutputItem::FunctionToolCall {
                id: Some(tool_call.id.clone()),
                call_id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: tool_call.function.arguments.clone().unwrap_or_default(),
                output: None, // Tool hasn't been executed yet
                status: "in_progress".to_string(),
            });
        }
    }

    // Determine response status based on finish_reason
    let status = match choice.finish_reason.as_deref() {
        Some("stop") | Some("length") => ResponseStatus::Completed,
        Some("tool_calls") => ResponseStatus::InProgress, // Waiting for tool execution
        Some("failed") | Some("error") => ResponseStatus::Failed,
        _ => ResponseStatus::Completed, // Default to completed
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
    Ok(ResponsesResponse::builder(&response_id, &chat_resp.model)
        .copy_from_request(original_req)
        .created_at(chat_resp.created as i64)
        .status(status)
        .output(output)
        .maybe_text(original_req.text.clone())
        .maybe_usage(usage)
        .build())
}

#[cfg(test)]
mod tests {
    use openai_protocol::{
        chat::{ChatChoice, ChatCompletionMessage},
        common::{StreamOptions, Usage},
    };

    use super::*;

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
            } => assert_eq!(tool_calls[0].id, "call_tool_id"),
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
}
