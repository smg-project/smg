//! Streaming execution for Regular Responses API
//!
//! This module handles streaming request execution:
//! - `execute_tool_loop_streaming` - MCP tool loop with streaming
//! - `convert_chat_stream_to_responses_stream` - Non-MCP streaming conversion
//! - Streaming accumulators for response building

use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures_util::StreamExt;
use openai_protocol::{
    chat::{
        ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
        ChatCompletionStreamResponse,
    },
    common::{FunctionCallResponse, ToolCall, Usage, UsageInfo},
    responses::{
        ResponseContentPart, ResponseOutputItem, ResponseReasoningContent, ResponseStatus,
        ResponsesRequest, ResponsesResponse, ResponsesUsage,
    },
};
use serde_json::{json, Value};
use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseStorage,
};
use smg_mcp::{McpServerBinding, McpToolSession, ToolExecutionInput};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, trace, warn};
use uuid::Uuid;

use super::{
    common::{
        build_next_request, convert_mcp_tools_to_chat_tools, extract_all_tool_calls_from_chat,
        prepare_chat_tools_and_choice, ExtractedToolCall, ResponsesCallContext, ToolLoopState,
    },
    conversions,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::{
            mcp_utils::{prepare_hosted_dispatch_args, DEFAULT_MAX_ITERATIONS},
            openai_bridge::{self, ResponseFormat},
        },
        grpc::{
            common::responses::{
                build_sse_response, persist_response_if_needed,
                streaming::{attach_mcp_server_label, OutputItemKind, ResponseStreamEventEmitter},
                ResponsesContext,
            },
            utils,
        },
    },
};

// ============================================================================
// Non-MCP Streaming Path
// ============================================================================

/// Convert chat streaming response to responses streaming format
///
/// This function:
/// 1. Gets chat SSE stream from pipeline
/// 2. Intercepts and parses each SSE event
/// 3. Converts ChatCompletionStreamResponse → ResponsesResponse delta
/// 4. Accumulates response state for final persistence
/// 5. Emits transformed SSE events in responses format
pub(super) async fn convert_chat_stream_to_responses_stream(
    ctx: &ResponsesContext,
    chat_request: Arc<ChatCompletionRequest>,
    params: ResponsesCallContext,
    original_request: &ResponsesRequest,
) -> Response {
    debug!("Converting chat SSE stream to responses SSE format");

    // Get chat streaming response
    let chat_response = ctx
        .pipeline
        .execute_chat(
            chat_request,
            params.headers,
            params.model_id,
            ctx.components.clone(),
            Some(params.tenant_request_meta),
            // Responses endpoint hasn't opted into tenant rate limiting yet
            // (out of scope for this phase; see RequestPipeline::build's
            // stage-insertion comment).
            None,
        )
        .await;

    // Extract body from chat response
    let (_parts, body) = chat_response.into_parts();

    // Create channel for transformed SSE events
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    // Spawn background task to transform stream
    let original_request_clone = original_request.clone();
    let response_storage = ctx.response_storage.clone();
    let conversation_storage = ctx.conversation_storage.clone();
    let conversation_item_storage = ctx.conversation_item_storage.clone();
    let request_context = ctx.request_context.clone();

    #[expect(
        clippy::disallowed_methods,
        reason = "streaming task is fire-and-forget; client disconnect terminates it"
    )]
    tokio::spawn(async move {
        if let Err(e) = process_and_transform_sse_stream(
            body,
            original_request_clone,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            request_context,
            tx.clone(),
        )
        .await
        {
            warn!("Error transforming SSE stream: {}", e);
            utils::send_error_sse(&tx, &e, "stream_error");
        }

        // Send final [DONE] event
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

    // Build SSE response with transformed stream
    build_sse_response(rx)
}

/// Process chat SSE stream and transform to responses format
async fn process_and_transform_sse_stream(
    body: Body,
    original_request: ResponsesRequest,
    response_storage: Arc<dyn ResponseStorage>,
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    request_context: Option<StorageRequestContext>,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
    // Create accumulator for final response
    let mut accumulator = StreamingResponseAccumulator::new(&original_request);

    // Create event emitter for OpenAI-compatible streaming
    let response_id = format!("resp_{}", Uuid::now_v7());
    let model = original_request.model.clone();
    let created_at = chrono::Utc::now().timestamp() as u64;
    let mut event_emitter = ResponseStreamEventEmitter::new(response_id, model, created_at);
    event_emitter.set_original_request(original_request.clone());

    // Emit initial response.created and response.in_progress events
    let event = event_emitter.emit_created();
    event_emitter
        .send_event(&event, &tx)
        .map_err(|_| "Failed to send response.created event".to_string())?;

    let event = event_emitter.emit_in_progress();
    event_emitter
        .send_event(&event, &tx)
        .map_err(|_| "Failed to send response.in_progress event".to_string())?;

    // Convert body to data stream
    let mut stream = body.into_data_stream();

    // Process stream chunks (each chunk is a complete SSE event)
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {e}"))?;

        // Convert chunk to string
        let event_str = String::from_utf8_lossy(&chunk);
        let event = event_str.trim();

        // Check for end of stream
        if event == "data: [DONE]" {
            break;
        }

        // Parse SSE event (format: "data: {...}\n\n" or "data: {...}")
        if let Some(json_str) = event.strip_prefix("data: ") {
            let json_str = json_str.trim();

            // Try to parse as ChatCompletionStreamResponse
            match serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
                Ok(chat_chunk) => {
                    // Update accumulator
                    accumulator.process_chunk(&chat_chunk);

                    // Process chunk through event emitter (emits proper OpenAI events)
                    event_emitter.process_chunk(&chat_chunk, &tx)?;
                }
                Err(_) => {
                    // Not a valid chat chunk - might be error event, pass through
                    debug!("Non-chunk SSE event, passing through: {}", event);
                    if tx.send(Ok(Bytes::from(format!("{event}\n\n")))).is_err() {
                        return Err("Client disconnected".to_string());
                    }
                }
            }
        }
    }

    // Emit final response.completed event with accumulated usage
    let usage_json = accumulator.usage.as_ref().map(|u| {
        let mut usage_obj = json!({
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens
        });

        // Include reasoning_tokens if present
        if let Some(details) = &u.completion_tokens_details {
            if let Some(reasoning_tokens) = details.reasoning_tokens {
                usage_obj["output_tokens_details"] =
                    json!({ "reasoning_tokens": reasoning_tokens });
            }
        }

        usage_obj
    });

    let completed_event = event_emitter.emit_completed(usage_json.as_ref());
    event_emitter.send_event(&completed_event, &tx)?;

    // Finalize and persist accumulated response
    let final_response = accumulator.finalize();
    persist_response_if_needed(
        conversation_storage,
        conversation_item_storage,
        response_storage,
        &final_response,
        &original_request,
        request_context,
    )
    .await;

    Ok(())
}

/// Response accumulator for streaming responses (non-MCP path)
struct StreamingResponseAccumulator {
    // Response metadata
    response_id: String,
    model: String,
    created_at: i64,

    // Accumulated content
    content_buffer: String,
    reasoning_buffer: String,
    tool_calls: Vec<ResponseOutputItem>,

    // Completion state
    finish_reason: Option<String>,
    usage: Option<Usage>,

    // Original request for final response construction
    original_request: ResponsesRequest,
}

impl StreamingResponseAccumulator {
    fn new(original_request: &ResponsesRequest) -> Self {
        Self {
            response_id: String::new(),
            model: String::new(),
            created_at: 0,
            content_buffer: String::new(),
            reasoning_buffer: String::new(),
            tool_calls: Vec::new(),
            finish_reason: None,
            usage: None,
            original_request: original_request.clone(),
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
        // Initialize metadata on first chunk
        if self.response_id.is_empty() {
            self.response_id.clone_from(&chunk.id);
            self.model.clone_from(&chunk.model);
            self.created_at = chunk.created as i64;
        }

        // Process first choice (responses API doesn't support n>1)
        if let Some(choice) = chunk.choices.first() {
            // Accumulate content
            if let Some(content) = &choice.delta.content {
                self.content_buffer.push_str(content);
            }

            // Accumulate reasoning
            if let Some(reasoning) = &choice.delta.reasoning_content {
                self.reasoning_buffer.push_str(reasoning);
            }

            // Process tool call deltas
            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    // Use index directly (it's a u32, not Option<u32>)
                    let index = delta.index as usize;

                    // Ensure we have enough tool calls
                    while self.tool_calls.len() <= index {
                        self.tool_calls.push(ResponseOutputItem::FunctionToolCall {
                            id: None,
                            call_id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                            output: None,
                            status: "in_progress".to_string(),
                        });
                    }

                    // Update the tool call at this index
                    if let ResponseOutputItem::FunctionToolCall {
                        id,
                        call_id,
                        name,
                        arguments,
                        ..
                    } = &mut self.tool_calls[index]
                    {
                        if let Some(delta_id) = &delta.id {
                            id.get_or_insert_with(String::new).push_str(delta_id);
                            call_id.push_str(delta_id);
                        }
                        if let Some(function) = &delta.function {
                            if let Some(delta_name) = &function.name {
                                name.push_str(delta_name);
                            }
                            if let Some(delta_args) = &function.arguments {
                                arguments.push_str(delta_args);
                            }
                        }
                    }
                }
            }

            // Update finish reason
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }

        // Update usage
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> ResponsesResponse {
        let mut output: Vec<ResponseOutputItem> = Vec::new();

        // Add message content if present
        if !self.content_buffer.is_empty() {
            output.push(ResponseOutputItem::Message {
                id: format!("msg_{}", self.response_id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: self.content_buffer,
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
                phase: None,
            });
        }

        // Add reasoning if present
        if !self.reasoning_buffer.is_empty() {
            output.push(ResponseOutputItem::new_reasoning(
                format!("reasoning_{}", self.response_id),
                vec![],
                vec![ResponseReasoningContent::ReasoningText {
                    text: self.reasoning_buffer,
                }],
                Some("completed".to_string()),
            ));
        }

        // Add tool calls
        output.extend(self.tool_calls);

        // Determine final status
        let status = match self.finish_reason.as_deref() {
            Some("stop") | Some("length") => ResponseStatus::Completed,
            Some("tool_calls") => ResponseStatus::InProgress,
            Some("failed") | Some("error") => ResponseStatus::Failed,
            _ => ResponseStatus::Completed,
        };

        // Convert usage
        let usage = self.usage.as_ref().map(|u| {
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

        ResponsesResponse::builder(&self.response_id, &self.model)
            .copy_from_request(&self.original_request)
            .created_at(self.created_at)
            .status(status)
            .output(output)
            .maybe_usage(usage)
            .build()
    }
}

// ============================================================================
// MCP Streaming Path
// ============================================================================

/// Execute MCP tool loop with streaming support
///
/// This streams each iteration's response to the client while accumulating
/// to check for tool calls. If tool calls are found, executes them and
/// continues with the next streaming iteration.
pub(super) fn execute_tool_loop_streaming(
    ctx: &ResponsesContext,
    current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    params: ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
) -> Response {
    // Create SSE channel for client
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    // Clone data for background task
    let ctx_clone = ctx.clone();
    let original_request_clone = original_request.clone();

    // Spawn background task for tool loop
    #[expect(
        clippy::disallowed_methods,
        reason = "streaming task is fire-and-forget; client disconnect terminates it"
    )]
    tokio::spawn(async move {
        let result = execute_tool_loop_streaming_internal(
            &ctx_clone,
            current_request,
            &original_request_clone,
            params,
            mcp_servers,
            tx.clone(),
        )
        .await;

        if let Err(e) = result {
            warn!("Streaming tool loop error: {}", e);
            utils::send_error_sse(&tx, &e, "tool_loop_error");
        }

        // Send [DONE]
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

    // Build SSE response
    let stream = UnboundedReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    #[expect(
        clippy::expect_used,
        reason = "Response::builder with valid status and no invalid headers is infallible"
    )]
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(body)
        .expect("infallible: valid status code, no invalid headers");

    response.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache"),
    );
    response.headers_mut().insert(
        header::CONNECTION,
        header::HeaderValue::from_static("keep-alive"),
    );

    response
}

/// Internal streaming tool loop implementation
async fn execute_tool_loop_streaming_internal(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    params: ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
    tx: mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<(), String> {
    let mut state = ToolLoopState::new(original_request.input.clone());
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    // Generate response ID first so we can use it for both emitter and session
    let response_id = format!("resp_{}", Uuid::now_v7());

    // Create session once — bundles orchestrator, request_ctx, server_keys, mcp_tools
    let session = McpToolSession::new(&ctx.mcp_orchestrator, mcp_servers, &response_id);

    // Create response event emitter
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut emitter =
        ResponseStreamEventEmitter::new(response_id, current_request.model.clone(), created_at);
    emitter.set_original_request(original_request.clone());

    // Emit initial response.created and response.in_progress events
    let event = emitter.emit_created();
    emitter.send_event(&event, &tx)?;
    let event = emitter.emit_in_progress();
    emitter.send_event(&event, &tx)?;

    // Get MCP tools and convert to chat format (do this once before loop)
    let mcp_chat_tools = convert_mcp_tools_to_chat_tools(&session);
    trace!(
        "Streaming: Converted {} MCP tools to chat format",
        mcp_chat_tools.len()
    );

    // Flag to track if mcp_list_tools has been emitted
    let mut mcp_list_tools_emitted = false;

    loop {
        state.iteration += 1;

        // Record tool loop iteration metric
        Metrics::record_mcp_tool_iteration(&current_request.model);

        if state.iteration > DEFAULT_MAX_ITERATIONS {
            return Err(format!(
                "Tool loop exceeded maximum iterations ({DEFAULT_MAX_ITERATIONS})"
            ));
        }

        trace!("Streaming MCP tool loop iteration {}", state.iteration);

        // Emit mcp_list_tools as first output item (only once, on first iteration)
        if !mcp_list_tools_emitted {
            for binding in session.mcp_servers() {
                let tools_for_server = session.list_tools_for_server(&binding.server_key);

                emitter.emit_mcp_list_tools_sequence(&binding.label, &tools_for_server, &tx)?;
            }
            mcp_list_tools_emitted = true;
        }

        // Convert to chat request
        let mut chat_request = conversions::responses_to_chat(&current_request)
            .map_err(|e| format!("Failed to convert request: {e}"))?;

        // Prepare tools and tool_choice for this iteration (same logic as non-streaming)
        prepare_chat_tools_and_choice(&mut chat_request, &mcp_chat_tools, state.iteration);

        // Execute chat streaming
        let response = ctx
            .pipeline
            .execute_chat(
                Arc::new(chat_request),
                params.headers.clone(),
                params.model_id.clone(),
                ctx.components.clone(),
                Some(params.tenant_request_meta.clone()),
                // Responses endpoint hasn't opted into tenant rate limiting yet.
                None,
            )
            .await;

        // Convert chat stream to Responses API events while accumulating for tool call detection
        // Stream text naturally - it only appears on final iteration (tool iterations have empty content)
        let accumulated_response =
            convert_and_accumulate_stream(response.into_body(), &mut emitter, &tx).await?;

        // Check for tool calls (extract all of them for parallel execution)
        let tool_calls = extract_all_tool_calls_from_chat(&accumulated_response);

        if !tool_calls.is_empty() {
            trace!(
                "Tool loop iteration {}: found {} tool call(s)",
                state.iteration,
                tool_calls.len()
            );

            // Separate MCP and function tool calls using session-exposed names.
            let (mcp_tool_calls, function_tool_calls): (Vec<ExtractedToolCall>, Vec<_>) =
                tool_calls
                    .into_iter()
                    .partition(|tc| session.has_exposed_tool(tc.name.as_str()));

            trace!(
                "Separated tool calls: {} MCP, {} function",
                mcp_tool_calls.len(),
                function_tool_calls.len()
            );

            // Check combined limit (only count MCP tools since function tools will be returned)
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                None => DEFAULT_MAX_ITERATIONS,
            };

            if state.total_calls + mcp_tool_calls.len() > effective_limit {
                warn!(
                    "Reached tool call limit: {} + {} > {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls,
                    mcp_tool_calls.len(),
                    effective_limit,
                    max_tool_calls,
                    DEFAULT_MAX_ITERATIONS
                );
                break;
            }

            // Process each MCP tool call
            for tool_call in mcp_tool_calls {
                state.total_calls += 1;

                trace!(
                    "Executing tool call {}/{}: {} (call_id: {})",
                    state.total_calls,
                    state.total_calls,
                    tool_call.name,
                    tool_call.call_id
                );

                let response_format = openai_bridge::lookup_tool_format(
                    &session,
                    &ctx.mcp_format_registry,
                    &tool_call.name,
                );

                // Use emitter helpers to determine correct type and allocate index
                let item_type =
                    ResponseStreamEventEmitter::type_str_for_format(Some(&response_format));
                let resolved_label = session.resolve_tool_server_label(&tool_call.name);

                // Allocate output_index with the format's id-prefix discriminator
                // (e.g. `ws_…` for web_search_call); see FormatDescriptor.
                let (output_index, item_id) =
                    emitter.allocate_output_index_for_format(Some(response_format));

                // Build initial tool call item
                let mut item = json!({
                    "id": item_id,
                    "type": item_type,
                    "name": tool_call.name,
                    "status": "in_progress",
                    "arguments": ""
                });
                attach_mcp_server_label(
                    &mut item,
                    Some(resolved_label.as_str()),
                    Some(&response_format),
                );

                // Emit output_item.added
                let event = emitter.emit_output_item_added(output_index, &item);
                emitter.send_event(&event, &tx)?;

                // Emit tool_call.in_progress
                let event =
                    emitter.emit_tool_call_in_progress(output_index, &item_id, response_format);
                emitter.send_event(&event, &tx)?;

                // Emit arguments events for mcp_call only (skip for builtin tools)
                if matches!(response_format, ResponseFormat::Passthrough) {
                    // Emit mcp_call_arguments.delta (simulate streaming by sending full arguments)
                    let event = emitter.emit_mcp_call_arguments_delta(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, &tx)?;

                    // Emit mcp_call_arguments.done
                    let event = emitter.emit_mcp_call_arguments_done(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, &tx)?;
                }

                // Emit searching/interpreting event for builtin tools
                if let Some(event) =
                    emitter.emit_tool_call_searching(output_index, &item_id, response_format)
                {
                    emitter.send_event(&event, &tx)?;
                }

                // Execute the MCP tool
                trace!(
                    "Calling MCP tool '{}' with args: {}",
                    tool_call.name,
                    tool_call.arguments
                );
                // Parse arguments to Value, coercing scalar/array/null payloads
                // to an empty object so hosted-tool override merge can actually
                // apply. `apply_hosted_tool_overrides` is a no-op on non-objects;
                // silently dropping caller-declared config would be surprising.
                let mut arguments = match serde_json::from_str::<Value>(&tool_call.arguments) {
                    Ok(Value::Object(map)) => Value::Object(map),
                    _ => json!({}),
                };
                prepare_hosted_dispatch_args(
                    &mut arguments,
                    response_format,
                    original_request.tools.as_deref().unwrap_or(&[]),
                    original_request.user.as_deref(),
                );

                // Execute the single tool via the normalized MCP execution API.
                // This avoids custom serialization and manual re-transformation in streaming paths.
                let tool_output = session
                    .execute_tool(ToolExecutionInput {
                        call_id: tool_call.call_id.clone(),
                        tool_name: tool_call.name.clone(),
                        arguments,
                    })
                    .await;

                let success = !tool_output.is_error;
                let output_str = tool_output.output.to_string();

                let output_item =
                    openai_bridge::transform_tool_output(&tool_output, response_format);
                let mut item_done = serde_json::to_value(&output_item).unwrap_or_else(|e| {
                    warn!(
                        tool = %tool_output.tool_name,
                        error = %e,
                        "Failed to serialize transformed output item; falling back to a minimal stub",
                    );
                    json!({
                        "id": item_id,
                        "type": item_type,
                        "status": if success { "completed" } else { "failed" },
                    })
                });
                // Override the typed item's id so output_item.done matches the
                // streaming-allocated id used by the earlier output_item.added.
                if let Some(obj) = item_done.as_object_mut() {
                    obj.insert("id".to_string(), json!(&item_id));
                }
                attach_mcp_server_label(
                    &mut item_done,
                    Some(tool_output.server_label.as_str()),
                    Some(&response_format),
                );

                if success {
                    let event =
                        emitter.emit_tool_call_completed(output_index, &item_id, response_format);
                    emitter.send_event(&event, &tx)?;
                } else {
                    let err_text = tool_output
                        .error_message
                        .clone()
                        .unwrap_or_else(|| output_str.clone());
                    warn!("Tool execution returned error: {}", err_text);

                    // `response.mcp_call.failed` is the only `*.failed` event
                    // in the Responses API. Hosted-builtin families close via
                    // `*.completed` to mirror OpenAI cloud's wire shape;
                    // the failure context (when present) lives in the item
                    // content.
                    if matches!(response_format, ResponseFormat::Passthrough) {
                        let event = emitter.emit_mcp_call_failed(output_index, &item_id, &err_text);
                        emitter.send_event(&event, &tx)?;
                    } else {
                        let event = emitter.emit_tool_call_completed(
                            output_index,
                            &item_id,
                            response_format,
                        );
                        emitter.send_event(&event, &tx)?;
                    }
                }

                let event = emitter.emit_output_item_done(output_index, &item_done);
                emitter.send_event(&event, &tx)?;
                emitter.complete_output_item(output_index);

                Metrics::record_mcp_tool_duration(
                    &current_request.model,
                    &tool_output.tool_name,
                    tool_output.duration,
                );
                Metrics::record_mcp_tool_call(
                    &current_request.model,
                    &tool_output.tool_name,
                    if success {
                        metrics_labels::RESULT_SUCCESS
                    } else {
                        metrics_labels::RESULT_ERROR
                    },
                );

                state.record_call(
                    tool_output.call_id,
                    tool_output.tool_name,
                    tool_output.arguments_str,
                    output_str,
                    output_item,
                    success,
                );
            }

            // If there are function tool calls, emit events and exit MCP loop
            if !function_tool_calls.is_empty() {
                trace!(
                    "Found {} function tool call(s) - emitting events and exiting MCP loop",
                    function_tool_calls.len()
                );

                // Emit function_tool_call events for each function tool
                for tool_call in function_tool_calls {
                    // Allocate output_index for this function_tool_call item
                    let (output_index, item_id) =
                        emitter.allocate_output_index(OutputItemKind::FunctionCall);

                    // Build initial function_call item
                    let item = json!({
                        "id": item_id,
                        "type": "function_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "status": "in_progress",
                        "arguments": ""
                    });

                    // Emit output_item.added
                    let event = emitter.emit_output_item_added(output_index, &item);
                    emitter.send_event(&event, &tx)?;

                    // Emit function_call_arguments.delta
                    let event = emitter.emit_function_call_arguments_delta(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, &tx)?;

                    // Emit function_call_arguments.done
                    let event = emitter.emit_function_call_arguments_done(
                        output_index,
                        &item_id,
                        &tool_call.arguments,
                    );
                    emitter.send_event(&event, &tx)?;

                    // Build complete item
                    let item_complete = json!({
                        "id": item_id,
                        "type": "function_call",
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                        "status": "completed",
                        "arguments": tool_call.arguments
                    });

                    // Emit output_item.done
                    let event = emitter.emit_output_item_done(output_index, &item_complete);
                    emitter.send_event(&event, &tx)?;

                    emitter.complete_output_item(output_index);
                }

                // Break loop to return response to caller
                break;
            }

            // Build next request with conversation history
            current_request = build_next_request(&state, current_request);

            continue;
        }

        // No tool calls, this is the final response
        trace!("No tool calls found, ending streaming MCP loop");

        // Check for reasoning content
        let reasoning_content = accumulated_response
            .choices
            .first()
            .and_then(|c| c.message.reasoning_content.clone());

        // Emit reasoning item if present
        if let Some(reasoning) = reasoning_content {
            if !reasoning.is_empty() {
                emitter.emit_reasoning_item(&tx, Some(reasoning))?;
            }
        }

        // Text message events already emitted naturally by process_chunk during stream processing
        // (OpenAI router approach - text only appears on final iteration when no tool calls)

        // Emit final response.completed event
        let usage_json = accumulated_response.usage.as_ref().map(|u| {
            json!({
                "input_tokens": u.prompt_tokens,
                "output_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens
            })
        });
        let event = emitter.emit_completed(usage_json.as_ref());
        emitter.send_event(&event, &tx)?;

        break;
    }

    Ok(())
}

/// Convert chat stream to Responses API events while accumulating for tool call detection
async fn convert_and_accumulate_stream(
    body: Body,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) -> Result<ChatCompletionResponse, String> {
    let mut accumulator = ChatResponseAccumulator::new();
    let mut stream = body.into_data_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {e}"))?;

        // Parse chunk
        let event_str = String::from_utf8_lossy(&chunk);
        let event = event_str.trim();

        if event == "data: [DONE]" {
            break;
        }

        if let Some(json_str) = event.strip_prefix("data: ") {
            let json_str = json_str.trim();
            if let Ok(chat_chunk) = serde_json::from_str::<ChatCompletionStreamResponse>(json_str) {
                // Convert chat chunk to Responses API events and emit
                emitter.process_chunk(&chat_chunk, tx)?;

                // Accumulate for tool call detection
                accumulator.process_chunk(&chat_chunk);
            }
        }
    }

    Ok(accumulator.finalize())
}

/// Accumulates chat streaming chunks into complete ChatCompletionResponse
struct ChatResponseAccumulator {
    id: String,
    model: String,
    content: String,
    reasoning_content: Option<String>,
    tool_calls: HashMap<usize, ToolCall>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
}

impl ChatResponseAccumulator {
    fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            content: String::new(),
            reasoning_content: None,
            tool_calls: HashMap::new(),
            finish_reason: None,
            usage: None,
        }
    }

    fn process_chunk(&mut self, chunk: &ChatCompletionStreamResponse) {
        if !chunk.id.is_empty() {
            self.id.clone_from(&chunk.id);
        }
        if !chunk.model.is_empty() {
            self.model.clone_from(&chunk.model);
        }

        if let Some(choice) = chunk.choices.first() {
            // Accumulate content
            if let Some(content) = &choice.delta.content {
                self.content.push_str(content);
            }

            // Accumulate reasoning content
            if let Some(reasoning) = &choice.delta.reasoning_content {
                self.reasoning_content
                    .get_or_insert_with(String::new)
                    .push_str(reasoning);
            }

            // Accumulate tool calls
            if let Some(tool_call_deltas) = &choice.delta.tool_calls {
                for delta in tool_call_deltas {
                    let index = delta.index as usize;
                    let entry = self.tool_calls.entry(index).or_insert_with(|| ToolCall {
                        id: String::new(),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: String::new(),
                            arguments: Some(String::new()),
                        },
                    });

                    if let Some(id) = &delta.id {
                        entry.id.clone_from(id);
                    }
                    if let Some(function) = &delta.function {
                        if let Some(name) = &function.name {
                            entry.function.name.clone_from(name);
                        }
                        if let Some(args) = &function.arguments {
                            if let Some(ref mut existing_args) = entry.function.arguments {
                                existing_args.push_str(args);
                            }
                        }
                    }
                }
            }

            // Capture finish reason
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }

        // Update usage
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> ChatCompletionResponse {
        let mut tool_calls_vec: Vec<_> = self.tool_calls.into_iter().collect();
        tool_calls_vec.sort_by_key(|(index, _)| *index);
        let tool_calls: Vec<_> = tool_calls_vec.into_iter().map(|(_, call)| call).collect();

        ChatCompletionResponse::builder(&self.id, &self.model)
            .choices(vec![ChatChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: if self.content.is_empty() {
                        None
                    } else {
                        Some(self.content)
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    reasoning_content: self.reasoning_content,
                },
                finish_reason: self.finish_reason,
                logprobs: None,
                matched_stop: None,
                hidden_states: None,
            }])
            .maybe_usage(self.usage)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_accumulator_serializes_responses_api_usage() {
        let request = ResponsesRequest::default();
        let mut accumulator = StreamingResponseAccumulator::new(&request);
        accumulator.usage = Some(
            Usage::from_counts(12, 7)
                .with_cached_tokens(3)
                .with_reasoning_tokens(2),
        );

        let wire = serde_json::to_value(accumulator.finalize())
            .expect("streaming response should serialize");
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
    fn streaming_accumulator_populates_call_id_from_tool_delta_id() {
        let request = ResponsesRequest::default();
        let mut accumulator = StreamingResponseAccumulator::new(&request);
        let chunk = ChatCompletionStreamResponse::builder("chatcmpl_test", "test-model")
            .add_choice_tool_name(0, "call_streamed", "lookup")
            .build();

        accumulator.process_chunk(&chunk);

        assert_eq!(accumulator.tool_calls.len(), 1);
        match &accumulator.tool_calls[0] {
            ResponseOutputItem::FunctionToolCall {
                id, call_id, name, ..
            } => {
                assert_eq!(id.as_deref(), Some("call_streamed"));
                assert_eq!(call_id, "call_streamed");
                assert_eq!(name, "lookup");
            }
            other => panic!("expected function tool call, got {other:?}"),
        }
    }
}
