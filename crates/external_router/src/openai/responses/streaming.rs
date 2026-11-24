//! Streaming response handling for OpenAI-compatible responses
//!
//! This module handles all streaming-related functionality including:
//! - SSE (Server-Sent Events) parsing and forwarding
//! - Streaming response accumulation for persistence
//! - Tool call detection and interception during streaming
//! - MCP tool execution loops within streaming responses
//! - Event transformation and output index remapping

use std::{borrow::Cow, io, sync::Arc};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use openai_protocol::{
    event_types::{
        is_function_call_type, is_response_event, FunctionCallEvent, ItemType, McpEvent,
        OutputItemEvent, ResponseEvent,
    },
    responses::{ResponseTool, ResponsesRequest},
};
use serde::Serialize;
use serde_json::{json, Value};
use smg_mcp::{McpOrchestrator, McpServerBinding, McpToolSession};
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

use super::{
    accumulator::StreamingResponseAccumulator,
    common::{extract_output_index, get_event_type, parse_sse_block, ChunkProcessor},
    utils::{
        patch_response_with_request_metadata, response_tool_to_value, restore_original_tools,
        rewrite_streaming_block,
    },
};
use crate::{
    openai_bridge,
    openai_bridge::{mcp_response_item_id, ResponseFormat},
};
const SSE_DONE: &str = "data: [DONE]\n\n";

use crate::{
    error,
    header_utils::{
        extract_forwardable_request_headers, insert_routed_worker_id, preserve_response_headers,
        ApiProvider,
    },
    mcp_utils::DEFAULT_MAX_ITERATIONS,
    metrics::Metrics,
    openai::{
        context::{RequestContext, StreamingEventContext, StreamingRequest},
        mcp::{
            execute_streaming_tool_calls, inject_mcp_metadata_last, prepare_mcp_tools_as_functions,
            resume_payload_in_place, retained_output_items, send_mcp_list_tools_events,
            skip_streaming_tool_calls, StreamAction, StreamingToolHandler, ToolLoopState,
            TOOL_CALL_LIMIT_REASON,
        },
    },
    persistence_utils::persist_conversation_items_owned,
    sse::{sse_channel, SseEncoder, SseSender},
    upstream_body::{
        error_body_text, read_response_body_limited, ReadBodyResult, ERROR_BODY_TOO_LARGE,
        MAX_ERROR_RESPONSE_SIZE,
    },
};

/// Apply all transformations to event data in-place (rewrite + transform)
/// Optimized to parse JSON only once instead of multiple times
/// Returns true if any changes were made
pub(super) fn apply_event_transformations_inplace(
    parsed_data: &mut Value,
    ctx: &StreamingEventContext<'_>,
) -> bool {
    let mut changed = false;

    // 1. Apply rewrite_streaming_block logic (store, previous_response_id, tools masking)
    let event_type = parsed_data
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let should_patch = is_response_event(event_type);
    // Owned copy needed for the match below since we mutate parsed_data
    let event_type = event_type.to_string();

    if should_patch {
        if let Some(response_obj) = parsed_data
            .get_mut("response")
            .and_then(|v| v.as_object_mut())
        {
            let desired_store = Value::Bool(ctx.original_request.store.unwrap_or(true));
            if response_obj.get("store") != Some(&desired_store) {
                response_obj.insert("store".to_string(), desired_store);
                changed = true;
            }

            if let Some(prev_id) = ctx.previous_response_id {
                let needs_previous = response_obj
                    .get("previous_response_id")
                    .map(|v| v.is_null() || v.as_str().map(|s| s.is_empty()).unwrap_or(false))
                    .unwrap_or(true);

                if needs_previous {
                    response_obj.insert(
                        "previous_response_id".to_string(),
                        Value::String(prev_id.to_string()),
                    );
                    changed = true;
                }
            }

            // Mask tools from function to MCP format (optimized without cloning)
            if response_obj.get("tools").is_some() {
                let requested_mcp = ctx
                    .original_request
                    .tools
                    .as_ref()
                    .map(|tools| tools.iter().any(|t| matches!(t, ResponseTool::Mcp(_))))
                    .unwrap_or(false);

                if requested_mcp {
                    if let Some(mcp_tools) = build_mcp_tools_value(ctx.original_request) {
                        response_obj.insert("tools".to_string(), mcp_tools);
                        response_obj
                            .entry("tool_choice".to_string())
                            .or_insert(Value::String("auto".to_string()));
                        changed = true;
                    }
                }
            }
        }
    }

    // 2. Apply transform_streaming_event logic (function_call → mcp_call/web_search_call)
    match event_type.as_str() {
        OutputItemEvent::ADDED | OutputItemEvent::DONE => {
            if let Some(item) = parsed_data.get_mut("item") {
                if let Some(item_type) = item.get("type").and_then(|v| v.as_str()) {
                    if is_function_call_type(item_type) {
                        // Look up response_format for the tool
                        let tool_name = item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        // Only transform if this is an MCP tool; keep function_call unchanged
                        if let Some(session) =
                            ctx.session.filter(|s| s.has_exposed_tool(&tool_name))
                        {
                            let response_format = ctx
                                .mcp_format_registry
                                .map(|reg| {
                                    openai_bridge::lookup_tool_format(session, reg, &tool_name)
                                })
                                .unwrap_or(ResponseFormat::Passthrough);

                            let d = openai_bridge::descriptor(response_format);
                            let (new_type, id_prefix) = (d.type_str, d.id_prefix);

                            item["type"] = json!(new_type);
                            if new_type == ItemType::MCP_CALL {
                                let label = session.resolve_tool_server_label(&tool_name);
                                item["server_label"] = json!(label);
                            }

                            // Transform ID from fc_* to appropriate prefix
                            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                                if new_type == ItemType::MCP_CALL {
                                    item["id"] = json!(mcp_response_item_id(id));
                                } else if let Some(stripped) = id.strip_prefix("fc_") {
                                    let new_id = format!("{id_prefix}_{stripped}");
                                    item["id"] = json!(new_id);
                                }
                            }

                            changed = true;
                        }
                    }
                }
            }
        }
        FunctionCallEvent::ARGUMENTS_DONE => {
            parsed_data["type"] = json!(McpEvent::CALL_ARGUMENTS_DONE);

            // Transform item_id from fc_* to mcp_*
            if let Some(item_id) = parsed_data.get("item_id").and_then(|v| v.as_str()) {
                parsed_data["item_id"] = json!(mcp_response_item_id(item_id));
            }

            changed = true;
        }
        _ => {}
    }

    changed
}

/// Helper to build MCP tools value
fn build_mcp_tools_value(original_body: &ResponsesRequest) -> Option<Value> {
    let tools = original_body.tools.as_ref()?;
    let mcp_tools: Vec<Value> = tools
        .iter()
        .filter(|t| matches!(t, ResponseTool::Mcp(_)))
        .filter_map(response_tool_to_value)
        .collect();

    if mcp_tools.is_empty() {
        None
    } else {
        Some(Value::Array(mcp_tools))
    }
}

/// Send an SSE event to the client channel
/// Returns false if the client disconnected
#[inline]
async fn send_sse_event(
    tx: &SseSender,
    enc: &mut SseEncoder,
    event_name: &str,
    data: &Value,
) -> bool {
    match enc.encode_event(event_name, data) {
        Ok(bytes) => tx.send(Ok(bytes)).await.is_ok(),
        Err(e) => {
            // Unreachable in practice (static event name + Value); fall back to
            // manual framing like send_final_response_event rather than treating
            // an encode error as a client disconnect.
            warn!("failed to encode SSE event '{event_name}', falling back: {e}");
            let block = format!("event: {event_name}\ndata: {data}\n\n");
            tx.send(Ok(Bytes::from(block))).await.is_ok()
        }
    }
}

/// The error code of a stream whose upstream request could not be sent: the
/// connection refused or reset, the name not resolved, the request timed out.
const UPSTREAM_REQUEST_FAILED: &str = "upstream_request_failed";
/// The error code of a stream whose upstream body ended with an error before
/// its natural end, as the gateway's own routers code the same failure.
const STREAM_ERROR: &str = "stream_error";
/// The error code of a failure in the gateway's own processing.
const INTERNAL_ERROR: &str = "internal_error";

/// End the relayed stream with the `error` event every router writes for a
/// failure of its own ([`SseEncoder::encode_responses_error`]), at the
/// stream's next sequence number; nothing follows it. Best-effort: a client
/// that already left is not told.
async fn send_error_event(
    tx: &SseSender,
    enc: &mut SseEncoder,
    sequence_number: &mut u64,
    code: &str,
    message: &str,
) {
    let frame = enc.encode_responses_error(code, message, *sequence_number);
    *sequence_number += 1;
    let _ = tx.send(Ok(frame)).await;
}

/// The error event of an upstream status answered after the stream began:
/// the upstream's own `code` (else its `type`) and `message` when its
/// (sanitized) body carries the API's error envelope, else the status's
/// error type with the status and the body as the message.
fn upstream_status_error(status: StatusCode, body: &str) -> (String, String) {
    let envelope: Option<Value> = serde_json::from_str(body).ok();
    let upstream = envelope.as_ref().and_then(|v| v.get("error"));
    let field = |name: &str| upstream.and_then(|e| e.get(name).and_then(Value::as_str));
    let code = field("code")
        .or_else(|| field("type"))
        .unwrap_or_else(|| error::error_type_for_status(status))
        .to_string();
    let message = field("message")
        .map(str::to_string)
        .unwrap_or_else(|| format!("Upstream error {status}: {body}"));
    (code, message)
}

/// Map function_call event names to mcp_call event names
#[inline]
fn map_event_name(event_name: &str) -> &str {
    match event_name {
        FunctionCallEvent::ARGUMENTS_DELTA => McpEvent::CALL_ARGUMENTS_DELTA,
        FunctionCallEvent::ARGUMENTS_DONE => McpEvent::CALL_ARGUMENTS_DONE,
        other => other,
    }
}

/// Send buffered function call arguments as a synthetic delta event.
/// Returns false if client disconnected.
async fn send_buffered_arguments(
    parsed_data: &mut Value,
    handler: &StreamingToolHandler,
    tx: &SseSender,
    enc: &mut SseEncoder,
    sequence_number: &mut u64,
    mapped_output_index: &mut Option<usize>,
) -> bool {
    let Some(output_index) = extract_output_index(parsed_data) else {
        return true;
    };

    let assigned_index = handler
        .mapped_output_index(output_index)
        .unwrap_or(output_index);
    *mapped_output_index = Some(assigned_index);

    let Some(call) = handler
        .pending_calls
        .iter()
        .find(|c| c.output_index == output_index)
    else {
        return true;
    };

    let arguments_value = if call.arguments_buffer.is_empty() {
        "{}".to_string()
    } else {
        call.arguments_buffer.clone()
    };

    // Update the done event with full arguments
    parsed_data["arguments"] = Value::String(arguments_value.clone());

    // Transform item_id
    let item_id = parsed_data
        .get("item_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let mcp_item_id = mcp_response_item_id(item_id);

    // Build synthetic delta event
    let mut delta_event = json!({
        "type": McpEvent::CALL_ARGUMENTS_DELTA,
        "sequence_number": *sequence_number,
        "output_index": assigned_index,
        "item_id": mcp_item_id,
        "delta": arguments_value,
    });

    // Add obfuscation if present
    let obfuscation = call
        .last_obfuscation
        .as_ref()
        .map(|s| Value::String(s.clone()))
        .or_else(|| parsed_data.get("obfuscation").cloned());

    if let Some(obf) = obfuscation {
        if let Some(obj) = delta_event.as_object_mut() {
            obj.insert("obfuscation".to_string(), obf);
        }
    }

    if !send_sse_event(tx, enc, McpEvent::CALL_ARGUMENTS_DELTA, &delta_event).await {
        return false;
    }

    *sequence_number += 1;
    true
}

/// An SSE event to be forwarded to the client, with optional pre-parsed JSON.
pub(super) struct SseEventData<'a> {
    pub raw_block: &'a str,
    pub event_name: Option<&'a str>,
    pub data: &'a str,
    /// Pre-parsed JSON value. When `Some`, avoids re-parsing `data`.
    pub pre_parsed: Option<Value>,
}

/// Forward and transform a streaming event to the client.
/// Returns false if client disconnected.
pub(super) async fn forward_streaming_event(
    event: SseEventData<'_>,
    handler: &mut StreamingToolHandler,
    tx: &SseSender,
    enc: &mut SseEncoder,
    ctx: &StreamingEventContext<'_>,
    sequence_number: &mut u64,
) -> bool {
    let SseEventData {
        raw_block,
        event_name,
        data,
        pre_parsed,
    } = event;

    if event_name == Some(FunctionCallEvent::ARGUMENTS_DELTA) {
        return true;
    }

    // Use pre-parsed value or parse JSON data
    let mut parsed_data: Value = match pre_parsed {
        Some(v) => v,
        None => match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => {
                let chunk = format!("{raw_block}\n\n");
                return tx.send(Ok(Bytes::from(chunk))).await.is_ok();
            }
        },
    };

    let event_type = get_event_type(event_name, &parsed_data);
    if event_type == ResponseEvent::COMPLETED {
        return true;
    }

    // Handle function_call_arguments.done - send buffered args first
    let mut mapped_output_index: Option<usize> = None;
    if event_name == Some(FunctionCallEvent::ARGUMENTS_DONE)
        && !send_buffered_arguments(
            &mut parsed_data,
            handler,
            tx,
            enc,
            sequence_number,
            &mut mapped_output_index,
        )
        .await
    {
        return false;
    }

    if mapped_output_index.is_none() {
        if let Some(idx) = extract_output_index(&parsed_data) {
            mapped_output_index = handler.mapped_output_index(idx);
        }
    }
    if let Some(mapped) = mapped_output_index {
        parsed_data["output_index"] = json!(mapped);
    }

    apply_event_transformations_inplace(&mut parsed_data, ctx);

    if let Some(response_obj) = parsed_data
        .get_mut("response")
        .and_then(|v| v.as_object_mut())
    {
        if let Some(original_id) = handler.original_response_id() {
            response_obj.insert("id".to_string(), Value::String(original_id.to_string()));
        }
    }

    if parsed_data.get("sequence_number").is_some() {
        parsed_data["sequence_number"] = json!(*sequence_number);
        *sequence_number += 1;
    }

    // Serialize directly into the reusable encoder buffer (no intermediate
    // `String`). On the (practically impossible) serialization error, fall
    // back to forwarding the original raw block unchanged.
    let final_bytes = match event_name {
        Some(evt) => enc.encode_event(map_event_name(evt), &parsed_data),
        None => enc.encode_data(&parsed_data),
    };
    let final_bytes = match final_bytes {
        Ok(bytes) => bytes,
        Err(_) => Bytes::from(format!("{raw_block}\n\n")),
    };

    if tx.send(Ok(final_bytes)).await.is_err() {
        return false;
    }

    if event_name == Some(OutputItemEvent::ADDED)
        && !maybe_inject_tool_in_progress(&parsed_data, tx, enc, sequence_number).await
    {
        return false;
    }

    true
}

/// Inject in_progress event after a tool call item is added.
/// Handles mcp_call, web_search_call, code_interpreter_call, file_search_call,
/// and image_generation_call items.
/// Returns false if client disconnected.
async fn maybe_inject_tool_in_progress(
    parsed_data: &Value,
    tx: &SseSender,
    enc: &mut SseEncoder,
    sequence_number: &mut u64,
) -> bool {
    let Some(item) = parsed_data.get("item") else {
        return true;
    };

    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

    let Some(format) = openai_bridge::format_from_type_str(item_type) else {
        return true; // Not a tool call item, nothing to inject
    };
    let event_type = openai_bridge::descriptor(format).in_progress_event;

    let Some(item_id) = item.get("id").and_then(|v| v.as_str()) else {
        return true;
    };
    let Some(output_index) = parsed_data.get("output_index").and_then(|v| v.as_u64()) else {
        return true;
    };

    let event = json!({
        "type": event_type,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item_id": item_id
    });
    *sequence_number += 1;

    send_sse_event(tx, enc, event_type, &event).await
}

/// How the tool loop ended: with the model's answer, or at the request's
/// tool-call allowance before one. Both are a normal end of the run
/// (`response.completed`, the response `completed`, the record stored); the
/// second carries the note the non-streamed loop leaves on the same stop,
/// `incomplete_details.reason` `max_tool_calls`, so a streamed and a
/// non-streamed request stopped the same way leave the same record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ToolLoopEnd {
    Answered,
    ToolCallLimit,
}

/// What the non-streamed loop does to a response it ended at the request's
/// `max_tool_calls`: the model's asks for server calls leave the items (each
/// is rendered by the item that says it was not executed), and
/// `incomplete_details` names the reason, the status left `completed` (the
/// calls that ran are answered; the ones that did not are not a failure of
/// the run).
fn end_at_tool_call_limit(response: &mut Value, original_request: &ResponsesRequest) {
    let Some(obj) = response.as_object_mut() else {
        return;
    };
    if let Some(output) = obj.get_mut("output").and_then(Value::as_array_mut) {
        let retained = retained_output_items(output, original_request);
        output.clear();
        output.extend(retained);
    }
    obj.insert(
        "incomplete_details".to_string(),
        json!({ "reason": TOOL_CALL_LIMIT_REASON }),
    );
}

/// The loop's final response shaped once, for the wire and the record alike:
/// the preserved response id written back, the loop's items in place (the
/// kept `mcp_list_tools` items move in; nothing reads them afterwards), the
/// client's view of the tools, the request's metadata, and the note of a
/// loop stopped at its tool-call allowance.
pub(super) fn shape_final_response(
    final_response: &mut Value,
    preserved_response_id: Option<&str>,
    state: &mut ToolLoopState,
    ctx: &StreamingEventContext<'_>,
    end: ToolLoopEnd,
) {
    if let Some(original_id) = preserved_response_id {
        if let Some(obj) = final_response.as_object_mut() {
            obj.insert("id".to_string(), Value::String(original_id.to_string()));
        }
    }

    if let Some(session) = ctx.session {
        inject_mcp_metadata_last(final_response, state, session);
    }

    // With the session, the wire hides what the record hides: the internal
    // servers' items and the calls to their tools. The asks for the session's
    // exposed tools that a limit stop leaves beside their not-executed items
    // are dropped by `end_at_tool_call_limit` below.
    restore_original_tools(final_response, ctx.original_request, ctx.session);
    patch_response_with_request_metadata(
        final_response,
        ctx.original_request,
        ctx.previous_response_id,
    );

    if end == ToolLoopEnd::ToolCallLimit {
        end_at_tool_call_limit(final_response, ctx.original_request);
    }
}

/// Send the `response.completed` event that ends the tool loop's stream with
/// the shaped final response. The event's response reads `completed` for
/// either end of the loop; the value keeps the status it carried afterwards,
/// so the record stores what the upstream said. Returns false if the client
/// disconnected.
pub(super) async fn send_final_response_event(
    final_response: &mut Value,
    tx: &SseSender,
    enc: &mut SseEncoder,
    sequence_number: &mut u64,
) -> bool {
    let upstream_status = final_response
        .as_object_mut()
        .map(|obj| obj.insert("status".to_string(), Value::String("completed".to_string())));

    let sequence = *sequence_number;
    *sequence_number += 1;
    let completed_payload = CompletedEventPayload {
        event_type: ResponseEvent::COMPLETED,
        sequence_number: sequence,
        response: final_response,
    };

    let sent = match enc.encode_event(ResponseEvent::COMPLETED, &completed_payload) {
        Ok(bytes) => tx.send(Ok(bytes)).await.is_ok(),
        Err(e) => {
            warn!("failed to encode response.completed via SseEncoder, falling back: {e}");
            let completed_event = format!(
                "event: {}\ndata: {}\n\n",
                ResponseEvent::COMPLETED,
                json!({
                    "type": ResponseEvent::COMPLETED,
                    "sequence_number": sequence,
                    "response": final_response
                })
            );
            tx.send(Ok(Bytes::from(completed_event))).await.is_ok()
        }
    };

    if let (Some(previous), Some(obj)) = (upstream_status, final_response.as_object_mut()) {
        match previous {
            Some(status) => {
                obj.insert("status".to_string(), status);
            }
            None => {
                obj.remove("status");
            }
        }
    }
    sent
}

/// The `response.completed` event's payload, serialised onto the wire from
/// the final response where it stands; `json!` would copy the whole response,
/// its `mcp_list_tools` schemas included, into the event first.
#[derive(Serialize)]
struct CompletedEventPayload<'a> {
    #[serde(rename = "type")]
    event_type: &'static str,
    sequence_number: u64,
    response: &'a Value,
}

/// Simple pass-through streaming without MCP interception
pub(super) async fn handle_simple_streaming_passthrough(
    client: &reqwest::Client,
    worker: &Arc<dyn crate::worker::ExternalWorker>,
    headers: Option<&HeaderMap>,
    req: StreamingRequest,
) -> Response {
    let mut request_builder = client.post(&req.url).json(&req.payload);
    let provider = ApiProvider::from_url(&req.url);
    let auth_header = provider.extract_auth_header(headers, worker.api_key());
    request_builder = provider.apply_headers(request_builder, auth_header.as_ref());

    request_builder = request_builder.header("Accept", "text/event-stream");

    let response = match request_builder.send().await {
        Ok(resp) => resp,
        Err(err) => {
            worker.record_outcome(502);
            return (
                StatusCode::BAD_GATEWAY,
                format!("Failed to forward request to OpenAI: {}", err.without_url()),
            )
                .into_response();
        }
    };

    let status = response.status();

    worker.record_outcome(status.as_u16());

    if !status.is_success() {
        let error_body = match read_response_body_limited(response, MAX_ERROR_RESPONSE_SIZE).await {
            ReadBodyResult::Ok(body) => body,
            ReadBodyResult::TooLarge => {
                return error::create_error(status, "upstream_error", ERROR_BODY_TOO_LARGE);
            }
            // The reader's text of a failed read is the body stream's own, which
            // carries no request URL.
            ReadBodyResult::Error(err) => format!("Failed to read upstream error body: {err}"),
        };
        let error_body = error::sanitize_error_body(&error_body);
        return (status, error_body).into_response();
    }

    let mut preserved_headers = preserve_response_headers(response.headers());
    insert_routed_worker_id(&mut preserved_headers, worker.redacted_url());
    let mut upstream_stream = response.bytes_stream();

    let (tx, rx) = sse_channel();

    let should_store = req.original_body.store.unwrap_or(true);
    let original_request = req.original_body;
    let previous_response_id = req.previous_response_id;
    let storage = req.storage;

    #[expect(
        clippy::disallowed_methods,
        reason = "fire-and-forget stream processing; gateway shutdown need not wait for individual response streams"
    )]
    tokio::spawn(async move {
        let mut accumulator = StreamingResponseAccumulator::new();
        let mut upstream_failed = false;
        let mut receiver_connected = true;
        let mut chunk_processor = ChunkProcessor::new();

        while let Some(chunk_result) = upstream_stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    chunk_processor.push_chunk(&chunk);

                    while let Some(raw_block) = chunk_processor.next_block() {
                        let block_cow = match rewrite_streaming_block(
                            raw_block,
                            &original_request,
                            previous_response_id.as_deref(),
                        ) {
                            Some(modified) => Cow::Owned(modified),
                            None => Cow::Borrowed(raw_block),
                        };

                        if should_store {
                            accumulator.ingest_block(&block_cow);
                        }

                        if receiver_connected {
                            let chunk_to_send = format!("{block_cow}\n\n");
                            if tx.send(Ok(Bytes::from(chunk_to_send))).await.is_err() {
                                receiver_connected = false;
                            }
                        }

                        if !receiver_connected && !should_store {
                            break;
                        }
                    }

                    if !receiver_connected && !should_store {
                        break;
                    }
                }
                Err(err) => {
                    upstream_failed = true;
                    let io_err = io::Error::other(err);
                    let _ = tx.send(Err(io_err)).await;
                    break;
                }
            }
        }

        if should_store && !upstream_failed {
            if chunk_processor.has_remaining() {
                accumulator.ingest_block(&chunk_processor.take_remaining());
            }
            let encountered_error = accumulator.encountered_error().cloned();
            if let Some(mut response_json) = accumulator.into_final_response() {
                patch_response_with_request_metadata(
                    &mut response_json,
                    &original_request,
                    previous_response_id.as_deref(),
                );

                // Always persist conversation items and response (even without conversation)
                if let Err(err) = persist_conversation_items_owned(
                    storage.conversation.clone(),
                    storage.conversation_item.clone(),
                    storage.response.clone(),
                    response_json,
                    &original_request,
                    storage.tenant.as_deref(),
                    storage.request_context.clone(),
                )
                .await
                {
                    warn!("Failed to persist conversation items (stream): {}", err);
                }
            } else if let Some(error_payload) = encountered_error {
                warn!("Upstream streaming error payload: {}", error_payload);
            } else {
                warn!("Streaming completed without a final response payload");
            }
        }
    });

    let body_stream = ReceiverStream::new(rx);
    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = status;

    let headers_mut = response.headers_mut();
    // One pair per value: a repeated upstream header keeps every value here too.
    for (name, value) in &preserved_headers {
        headers_mut.append(name, value.clone());
    }

    if !headers_mut.contains_key(CONTENT_TYPE) {
        headers_mut.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    }

    response
}

/// What one relayed upstream stream ended with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RelayEnd {
    /// The model finished without asking for a tool.
    Answered,
    /// The handler holds complete tool calls; the loop executes them.
    ToolCalls,
    /// The client is gone or the stream broke (its error event sent):
    /// nothing follows.
    Stopped,
}

/// The tool loop's flags the relay reads and sets across its iterations.
struct RelayFlags {
    /// Only the first upstream stream forwards `response.created` and
    /// `response.in_progress`; a resumed stream's copies are skipped.
    first_iteration: bool,
    /// The MCP list-tools items go out once, at the first `in_progress`.
    mcp_list_tools_sent: bool,
}

/// Relays one upstream event stream to the client: every block but the
/// upstream's end sentinel (the loop ends the client's stream with its own)
/// goes through the handler and is forwarded, buffered or dropped as it says;
/// the MCP list-tools items (the loop's, built once with its session; none
/// without one) go out at the first `in_progress` of the first stream; a
/// tool-call trigger ends the relay with the calls pending in the handler.
#[expect(
    clippy::too_many_arguments,
    reason = "The relay reads the loop's client side as the loop's own text did (the sender, the encoder, the sequence number) beside the handler and its two flags; a struct for one call would be a type the loop's other phases do not share."
)]
async fn relay_upstream_stream(
    upstream_stream: &mut (impl Stream<Item = reqwest::Result<Bytes>> + Unpin),
    handler: &mut StreamingToolHandler,
    flags: &mut RelayFlags,
    list_tools_items: &[Value],
    tx: &SseSender,
    sse_encoder: &mut SseEncoder,
    streaming_ctx: &StreamingEventContext<'_>,
    sequence_number: &mut u64,
) -> RelayEnd {
    let mut chunk_processor = ChunkProcessor::new();
    let mut seen_in_progress = false;
    while let Some(chunk_result) = upstream_stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                chunk_processor.push_chunk(&chunk);

                while let Some(raw_block) = chunk_processor.next_block() {
                    // Parse event
                    let (event_name, data) = parse_sse_block(raw_block);

                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }

                    // Parse the payload once: the handler's gate and record, the skip and
                    // in_progress checks below and the forward all read this value.
                    let parsed = serde_json::from_str::<Value>(data.as_ref()).ok();
                    let action = match parsed.as_ref() {
                        Some(value) => handler.process_parsed(event_name, value),
                        // An unparsable payload: the handler's own path logs and forwards it.
                        None => handler.process_event(event_name, data.as_ref()),
                    };

                    match action {
                        StreamAction::Forward => {
                            // Skip response.created and response.in_progress on subsequent iterations
                            let should_skip = if flags.first_iteration {
                                false
                            } else {
                                parsed.as_ref().is_some_and(|v| {
                                    matches!(
                                        v.get("type").and_then(|t| t.as_str()),
                                        Some(ResponseEvent::CREATED)
                                            | Some(ResponseEvent::IN_PROGRESS)
                                    )
                                })
                            };

                            // Check in_progress before moving parsed into SseEventData
                            let is_in_progress = !seen_in_progress
                                && parsed.as_ref().is_some_and(|v| {
                                    v.get("type").and_then(|t| t.as_str())
                                        == Some(ResponseEvent::IN_PROGRESS)
                                });

                            if !should_skip {
                                // Forward the event with pre-parsed value (moved, not cloned)
                                if !forward_streaming_event(
                                    SseEventData {
                                        raw_block,
                                        event_name,
                                        data: data.as_ref(),
                                        pre_parsed: parsed,
                                    },
                                    handler,
                                    tx,
                                    sse_encoder,
                                    streaming_ctx,
                                    sequence_number,
                                )
                                .await
                                {
                                    return RelayEnd::Stopped;
                                }
                            }

                            if is_in_progress {
                                seen_in_progress = true;
                                if !flags.mcp_list_tools_sent {
                                    for item in list_tools_items {
                                        let list_tools_index =
                                            handler.allocate_synthetic_output_index();
                                        if !send_mcp_list_tools_events(
                                            tx,
                                            sse_encoder,
                                            item,
                                            list_tools_index,
                                            sequence_number,
                                        )
                                        .await
                                        {
                                            // Client disconnected
                                            return RelayEnd::Stopped;
                                        }
                                    }
                                    flags.mcp_list_tools_sent = true;
                                }
                            }
                        }
                        StreamAction::Buffer => {
                            // Don't forward, just buffer
                        }
                        StreamAction::Drop => {
                            // R6.7c native passthrough: upstream
                            // emitted an `output_item.done` for a
                            // hosted tool-call item that we did
                            // NOT wrap as a function_call (and so
                            // did not track in `pending_calls`).
                            // The upstream envelope arrives
                            // mis-ordered BEFORE
                            // `response.<type>.completed`, so we
                            // drop it here to preserve the wire
                            // invariant that `output_item.done`
                            // is the LAST event for a given item.
                            // Unlike `ExecuteTools`, this does NOT
                            // kick the tool loop — there is no
                            // dispatched call to finish.
                        }
                        StreamAction::ExecuteTools {
                            forward_triggering_event,
                        } => {
                            // When the upstream signals tool completion via
                            // `output_item.done` (instead of a preceding
                            // `function_call_arguments.done`), forwarding the
                            // event here would emit an umbrella
                            // `response.output_item.done` BEFORE
                            // `response.<tool>.completed`, violating spec
                            // sub-event ordering. The tool loop emits its own
                            // `output_item.done` at the correct position after
                            // the `.completed` sub-event; suppress here.
                            if forward_triggering_event
                                && !forward_streaming_event(
                                    SseEventData {
                                        raw_block,
                                        event_name,
                                        data: data.as_ref(),
                                        pre_parsed: parsed,
                                    },
                                    handler,
                                    tx,
                                    sse_encoder,
                                    streaming_ctx,
                                    sequence_number,
                                )
                                .await
                            {
                                return RelayEnd::Stopped;
                            }
                            return RelayEnd::ToolCalls;
                        }
                    }
                }
            }
            Err(e) => {
                let message = format!("Stream error: {}", e.without_url());
                send_error_event(tx, sse_encoder, sequence_number, STREAM_ERROR, &message).await;
                return RelayEnd::Stopped;
            }
        }
    }
    RelayEnd::Answered
}

/// Handle streaming WITH MCP tool call interception and execution
pub(super) fn handle_streaming_with_tool_interception(
    client: &reqwest::Client,
    worker_api_key: Option<String>,
    headers: Option<&HeaderMap>,
    req: StreamingRequest,
    orchestrator: &Arc<McpOrchestrator>,
    format_registry: openai_bridge::FormatRegistry,
    mcp_servers: Vec<McpServerBinding>,
) -> Response {
    let payload = req.payload;

    let (tx, rx) = sse_channel();
    let should_store = req.original_body.store.unwrap_or(true);
    let original_request = req.original_body;
    let previous_response_id = req.previous_response_id;
    let existing_mcp_list_tools_labels = req.existing_mcp_list_tools_labels;
    let url = req.url;
    let storage = req.storage;

    let client_clone = client.clone();
    let url_clone = url.clone();
    let headers_opt = headers.cloned();
    let orchestrator_clone = Arc::clone(orchestrator);
    let forwarded_headers = extract_forwardable_request_headers(headers);

    #[expect(
        clippy::disallowed_methods,
        reason = "fire-and-forget MCP tool loop; gateway shutdown need not wait for individual tool loops"
    )]
    tokio::spawn(async move {
        let mut state = ToolLoopState::new(existing_mcp_list_tools_labels);
        let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

        // Create session inside spawned task (borrows from orchestrator_clone which lives in closure)
        let session_request_id = format!("resp_{}", uuid::Uuid::now_v7());
        let mut session = McpToolSession::new_with_headers(
            &orchestrator_clone,
            mcp_servers.clone(),
            &session_request_id,
            forwarded_headers.clone(),
        );
        if let Some(tools) = original_request.tools.as_deref() {
            openai_bridge::configure_response_tools_approval(&mut session, tools);
        }
        let mut current_payload = payload;
        prepare_mcp_tools_as_functions(&mut current_payload, &session);
        let mut flags = RelayFlags {
            first_iteration: true,
            mcp_list_tools_sent: false,
        };
        let mut sequence_number: u64 = 0;
        let mut sse_encoder = SseEncoder::new();
        let mut next_output_index: usize = 0;
        let mut preserved_response_id: Option<String> = None;

        let streaming_ctx = StreamingEventContext {
            original_request: &original_request,
            previous_response_id: previous_response_id.as_deref(),
            session: Some(&session),
            mcp_format_registry: Some(&format_registry),
        };
        let provider = ApiProvider::from_url(&url_clone);
        let auth_header =
            provider.extract_auth_header(headers_opt.as_ref(), worker_api_key.as_ref());

        let (handler, end) = loop {
            // Make streaming request
            let mut request_builder = client_clone.post(&url_clone).json(&current_payload);
            request_builder = provider.apply_headers(request_builder, auth_header.as_ref());
            request_builder = request_builder.header("Accept", "text/event-stream");

            let response = match request_builder.send().await {
                Ok(r) => r,
                Err(e) => {
                    let message = format!("Upstream request failed: {}", e.without_url());
                    send_error_event(
                        &tx,
                        &mut sse_encoder,
                        &mut sequence_number,
                        UPSTREAM_REQUEST_FAILED,
                        &message,
                    )
                    .await;
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = error_body_text(response).await;
                let (code, message) = upstream_status_error(status, &body);
                send_error_event(&tx, &mut sse_encoder, &mut sequence_number, &code, &message)
                    .await;
                return;
            }

            let mut upstream_stream = response.bytes_stream();
            let mut handler = StreamingToolHandler::with_starting_index(next_output_index);
            if let Some(ref id) = preserved_response_id {
                handler.original_response_id = Some(id.clone());
            }
            let end = relay_upstream_stream(
                &mut upstream_stream,
                &mut handler,
                &mut flags,
                state.mcp_list_tools_items(&session),
                &tx,
                &mut sse_encoder,
                &streaming_ctx,
                &mut sequence_number,
            )
            .await;
            if end == RelayEnd::Stopped {
                return;
            }

            next_output_index = handler.next_output_index();
            if let Some(id) = handler.original_response_id().map(|s| s.to_string()) {
                preserved_response_id = Some(id);
            }

            // The model answered without asking for a tool: the loop's end.
            if end == RelayEnd::Answered {
                break (handler, ToolLoopEnd::Answered);
            }

            // Execute tools
            let mut pending_calls = handler.take_pending_calls();

            // Check iteration limit
            state.iteration += 1;
            state.total_calls += pending_calls.len();

            // Record tool loop iteration metric
            Metrics::record_mcp_tool_iteration(&original_request.model);

            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                None => DEFAULT_MAX_ITERATIONS,
            };

            // The calls that fit the request's remaining allowance run, as
            // the non-streamed loop runs them; the rest of the batch is the
            // stop, the loop's end before an answer.
            let over_limit = state.total_calls.saturating_sub(effective_limit);
            let skipped = pending_calls.split_off(pending_calls.len().saturating_sub(over_limit));

            // Execute all pending tool calls. Pass the caller-declared tools so
            // hosted-tool overrides (e.g. image_generation size/quality) are
            // merged into dispatch args before MCP execution. The request-level
            // `user` is also forwarded into hosted-tool dispatch args.
            if !execute_streaming_tool_calls(
                pending_calls,
                &session,
                &format_registry,
                &tx,
                &mut state,
                &mut sequence_number,
                &original_request.model,
                original_request.tools.as_deref().unwrap_or(&[]),
                original_request.user.as_deref(),
            )
            .await
            {
                return;
            }

            if !skipped.is_empty() {
                warn!(
                    effective_limit,
                    skipped = skipped.len(),
                    "Reached the tool call limit during streaming: the remaining calls are not executed"
                );
                if !skip_streaming_tool_calls(
                    skipped,
                    &session,
                    &format_registry,
                    &tx,
                    &mut state,
                    &mut sequence_number,
                )
                .await
                {
                    return;
                }
                break (handler, ToolLoopEnd::ToolCallLimit);
            }

            // The next turn's payload: this one, the calls just executed
            // appended
            match resume_payload_in_place(&mut current_payload, state.history_to_append(), true) {
                Ok(()) => {
                    flags.first_iteration = false;
                }
                Err(e) => {
                    let message = format!("Failed to build resume payload: {e}");
                    send_error_event(
                        &tx,
                        &mut sse_encoder,
                        &mut sequence_number,
                        INTERNAL_ERROR,
                        &message,
                    )
                    .await;
                    return;
                }
            }
        };

        // The loop's end, answered or stopped at the tool-call allowance: the
        // final response once, shaped once, the terminal event from it, the
        // same value the record, the sentinel.
        let mut final_response = match handler.accumulator.into_final_response() {
            Some(resp) => resp,
            None => {
                warn!("Final response unavailable; skipping synthetic completion event");
                let _ = tx.send(Ok(Bytes::from_static(SSE_DONE.as_bytes()))).await;
                return;
            }
        };
        shape_final_response(
            &mut final_response,
            preserved_response_id.as_deref(),
            &mut state,
            &streaming_ctx,
            end,
        );
        if !send_final_response_event(
            &mut final_response,
            &tx,
            &mut sse_encoder,
            &mut sequence_number,
        )
        .await
        {
            return;
        }

        if should_store {
            // Always persist conversation items and response (even without conversation)
            if let Err(err) = persist_conversation_items_owned(
                storage.conversation.clone(),
                storage.conversation_item.clone(),
                storage.response.clone(),
                final_response,
                &original_request,
                storage.tenant.as_deref(),
                storage.request_context.clone(),
            )
            .await
            {
                warn!(
                    "Failed to persist conversation items (stream + MCP): {}",
                    err
                );
            }
        }

        let _ = tx.send(Ok(Bytes::from_static(SSE_DONE.as_bytes()))).await;
    });

    let body_stream = ReceiverStream::new(rx);
    let mut response = Response::new(Body::from_stream(body_stream));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
}

/// Main entry point for streaming responses
pub async fn handle_streaming_response(ctx: RequestContext) -> Response {
    use crate::mcp_utils::{ensure_request_mcp_routing, request_uses_mcp_routing};

    let worker = match ctx.worker() {
        Some(w) => w.clone(),
        None => {
            return error::internal_error("internal_error", "Worker not selected");
        }
    };
    let headers = ctx.headers().cloned();
    let original_body = match ctx.responses_request() {
        Some(r) => r,
        None => {
            return error::internal_error("internal_error", "Expected responses request");
        }
    };
    // Only MCP-laden requests need the orchestrator and format registry;
    // plain streaming requests must still pass through deployments without
    // MCP wiring.
    let mcp_routing = match original_body.tools.as_deref() {
        Some(tools) if request_uses_mcp_routing(tools) => {
            let Some(mcp_orchestrator) = ctx.components.mcp_orchestrator().cloned() else {
                return error::internal_error(
                    "internal_error",
                    "MCP orchestrator required for requests carrying MCP/builtin tools",
                );
            };
            let Some(registry) = ctx.components.mcp_format_registry() else {
                return error::internal_error(
                    "internal_error",
                    "MCP format registry required for requests carrying MCP/builtin tools",
                );
            };
            let registry = registry.clone();
            match ensure_request_mcp_routing(&mcp_orchestrator, &registry, tools).await {
                Ok(servers) => servers.map(|servers| (servers, mcp_orchestrator, registry)),
                Err(refusal) => return refusal,
            }
        }
        _ => None,
    };

    let client = ctx.components.client().clone();
    let req = match ctx.into_streaming_context() {
        Ok(r) => r,
        Err(msg) => {
            return error::internal_error("internal_error", msg);
        }
    };

    let Some((mcp_servers, mcp_orchestrator, mcp_format_registry)) = mcp_routing else {
        return handle_simple_streaming_passthrough(&client, &worker, headers.as_ref(), req).await;
    };

    handle_streaming_with_tool_interception(
        &client,
        worker.api_key().cloned(),
        headers.as_ref(),
        req,
        &mcp_orchestrator,
        mcp_format_registry,
        mcp_servers,
    )
}

#[cfg(test)]
mod tests {
    use openai_protocol::responses::ResponseInput;

    use super::*;

    /// The terminal event says `completed` on the wire whatever the shaped
    /// value carries, and the value is left as it was once the event is out:
    /// an upstream `incomplete` stays `incomplete` for the record, a value
    /// without a status gets none.
    #[tokio::test]
    async fn the_terminal_event_writes_completed_on_the_wire_and_leaves_the_value_its_own_status() {
        for (own_status, expected_after) in [
            (Some("incomplete"), Some(json!("incomplete"))),
            (None, None),
        ] {
            let mut final_response = json!({"id": "resp_1", "object": "response", "output": []});
            if let Some(status) = own_status {
                final_response["status"] = json!(status);
            }
            let (tx, mut rx) = sse_channel();
            let mut enc = SseEncoder::new();
            let mut sequence_number = 3;
            assert!(
                send_final_response_event(&mut final_response, &tx, &mut enc, &mut sequence_number)
                    .await
            );
            drop(tx);
            let frame = rx.recv().await.expect("the terminal").expect("a frame");
            let frame = String::from_utf8(frame.to_vec()).expect("utf-8");
            let data = frame
                .strip_prefix("event: response.completed\ndata: ")
                .expect("the completed event");
            let event: Value = serde_json::from_str(data.trim_end()).expect("json");
            assert_eq!(
                event["response"]["status"],
                json!("completed"),
                "{own_status:?}"
            );
            assert_eq!(sequence_number, 4, "{own_status:?}");
            assert_eq!(
                final_response.get("status").cloned(),
                expected_after,
                "the value keeps its own status once the event is out"
            );
        }
    }

    /// The terminal of a loop stopped at the request's tool-call allowance
    /// is the terminal of an answered loop, `response.completed` with the
    /// response `completed` at the stream's own sequence number, plus the
    /// note the non-streamed loop leaves on the same stop,
    /// `incomplete_details.reason` `max_tool_calls`; nothing else differs
    /// between the two frames.
    #[tokio::test]
    async fn the_tool_call_limit_terminal_is_the_completed_terminal_with_the_non_streamed_note() {
        let request = ResponsesRequest {
            model: "m".to_string(),
            input: ResponseInput::Text("hi".to_string()),
            ..Default::default()
        };
        let ctx = StreamingEventContext {
            original_request: &request,
            previous_response_id: None,
            session: None,
            mcp_format_registry: None,
        };
        let mut state = ToolLoopState::new(Vec::new());
        let created =
            "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\
             \"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\
             \"model\":\"m\",\"status\":\"in_progress\",\"output\":[]}}";

        let mut frames = Vec::new();
        for end in [ToolLoopEnd::Answered, ToolLoopEnd::ToolCallLimit] {
            let (tx, mut rx) = sse_channel();
            let mut enc = SseEncoder::new();
            let mut sequence_number = 7;
            let mut handler = StreamingToolHandler::with_starting_index(0);
            handler.accumulator.ingest_block(created);
            let preserved_id = handler.original_response_id().map(str::to_string);
            let mut final_response = handler
                .accumulator
                .into_final_response()
                .expect("the final response");
            shape_final_response(
                &mut final_response,
                preserved_id.as_deref(),
                &mut state,
                &ctx,
                end,
            );
            assert!(
                send_final_response_event(
                    &mut final_response,
                    &tx,
                    &mut enc,
                    &mut sequence_number,
                )
                .await
            );
            drop(tx);
            assert_eq!(sequence_number, 8, "{end:?}");
            let frame = rx.recv().await.expect("the terminal").expect("a frame");
            assert!(rx.recv().await.is_none(), "one frame, {end:?}");
            frames.push(String::from_utf8(frame.to_vec()).expect("utf-8"));
        }

        let data = |frame: &str| -> Value {
            let data = frame
                .strip_prefix("event: response.completed\ndata: ")
                .expect("the completed event");
            serde_json::from_str(data.trim_end()).expect("json")
        };
        let answered = data(&frames[0]);
        let mut limited = data(&frames[1]);
        assert_eq!(limited["type"], "response.completed");
        assert_eq!(limited["sequence_number"], 7);
        assert_eq!(limited["response"]["status"], "completed");
        assert_eq!(
            limited["response"]["incomplete_details"],
            json!({ "reason": "max_tool_calls" })
        );
        assert!(answered["response"].get("incomplete_details").is_none());
        limited["response"]
            .as_object_mut()
            .expect("an object")
            .remove("incomplete_details");
        assert_eq!(answered, limited, "the note is the only difference");
    }

    /// The error event that ends a failed stream is the frame every router
    /// writes for one, at the stream's own sequence number, which it
    /// advances; nothing follows it.
    #[tokio::test]
    async fn the_error_event_is_the_shared_frame_at_the_streams_sequence_number() {
        let (tx, mut rx) = sse_channel();
        let mut enc = SseEncoder::new();
        let mut sequence_number = 4;
        send_error_event(
            &tx,
            &mut enc,
            &mut sequence_number,
            STREAM_ERROR,
            "Stream error: connection reset",
        )
        .await;
        drop(tx);
        assert_eq!(sequence_number, 5);
        let frame = rx.recv().await.expect("the error frame").expect("a frame");
        assert_eq!(
            frame,
            SseEncoder::new().encode_responses_error(
                "stream_error",
                "Stream error: connection reset",
                4
            )
        );
        assert!(rx.recv().await.is_none(), "nothing follows the error event");
    }

    /// An upstream status answered after the stream began: the body's own
    /// code (else its type) and message when it carries the error envelope;
    /// the status's error type, with the status and the body, when it does
    /// not.
    #[test]
    fn an_upstream_status_keeps_the_bodys_code_and_message_when_it_has_them() {
        let cases = [
            (
                StatusCode::SERVICE_UNAVAILABLE,
                r#"{"error":{"code":"overloaded","type":"server_error","message":"busy"}}"#,
                "overloaded",
                "busy",
            ),
            (
                StatusCode::TOO_MANY_REQUESTS,
                r#"{"error":{"type":"rate_limit_error","message":"slow down"}}"#,
                "rate_limit_error",
                "slow down",
            ),
            (
                StatusCode::BAD_REQUEST,
                r#"{"error":{"message":"bad input"}}"#,
                "invalid_request_error",
                "bad input",
            ),
            (
                StatusCode::BAD_GATEWAY,
                "bad gateway",
                "server_error",
                "Upstream error 502 Bad Gateway: bad gateway",
            ),
        ];
        for (status, body, code, message) in cases {
            assert_eq!(
                upstream_status_error(status, body),
                (code.to_string(), message.to_string()),
                "{status} {body}"
            );
        }
    }

    /// The routed-worker header of a streaming answer names the worker
    /// masked: the header reaches every client of the stream and the access
    /// log copies it, so a credential in a worker URL never reaches either.
    #[tokio::test]
    async fn the_routed_worker_header_names_the_worker_masked() {
        use std::sync::Arc;

        use axum::{
            http::{header::CONTENT_TYPE, StatusCode},
            routing::any,
            Router,
        };
        use openai_protocol::responses::ResponsesRequest;
        use serde_json::json;
        use smg_data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        };

        use crate::{
            openai::context::StorageHandles,
            test_support::{http_client, StubWorker},
        };

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a listener");
        let port = listener.local_addr().expect("an address").port();
        #[expect(
            clippy::disallowed_methods,
            reason = "test stub upstream answering one event stream; aborted at the test's end"
        )]
        let upstream = tokio::spawn(async move {
            let app = Router::new().fallback(any(|| async {
                (
                    [(CONTENT_TYPE, "text/event-stream")],
                    "event: response.completed\ndata: {}\n\n",
                )
            }));
            axum::serve(listener, app).await.expect("the stub serves");
        });
        let worker_url = format!("http://user:secret@127.0.0.1:{port}");
        let request: ResponsesRequest =
            serde_json::from_value(json!({"model": "m", "input": "hi", "stream": true}))
                .expect("a request");
        let req = StreamingRequest {
            url: format!("{worker_url}/v1/responses"),
            payload: json!({}),
            original_body: request,
            previous_response_id: None,
            existing_mcp_list_tools_labels: Vec::new(),
            storage: StorageHandles {
                response: Arc::new(MemoryResponseStorage::new()),
                conversation: Arc::new(MemoryConversationStorage::new()),
                conversation_item: Arc::new(MemoryConversationItemStorage::new()),
                request_context: None,
                tenant: None,
            },
        };

        let response = handle_simple_streaming_passthrough(
            &http_client(),
            &StubWorker::at(&worker_url),
            None,
            req,
        )
        .await;
        upstream.abort();

        assert_eq!(response.status(), StatusCode::OK);
        let header = response
            .headers()
            .get("x-smg-routed-worker-id")
            .and_then(|value| value.to_str().ok())
            .expect("the routed-worker header");
        assert!(!header.contains("secret"), "{header}");
        assert_eq!(header, format!("http://***@127.0.0.1:{port}"));
    }

    /// A streamed forward nothing answers: the caller's 502 says why without
    /// the worker's address. The HTTP client's own error text ends with the
    /// request URL.
    #[tokio::test]
    async fn a_failed_streamed_forward_answers_without_the_workers_address() {
        use axum::http::StatusCode;

        use crate::test_support::{text_of, RefusedAddress};

        let refused = RefusedAddress::reserve();

        let response = streamed_answer_from(&format!("http://{}", refused.addr())).await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = text_of(response).await;
        assert!(!body.contains(&refused.addr().to_string()), "{body}");
    }

    /// A refusal whose body is lost mid-answer: the caller's answer says so
    /// without the worker's address. The HTTP client's text of a failed body
    /// read ends with the request URL.
    #[tokio::test]
    async fn an_unread_refusal_is_answered_without_the_workers_address() {
        use axum::http::StatusCode;

        use crate::test_support::{answering, text_of};

        let addr = answering(500, "short", true).await;

        let response = streamed_answer_from(&format!("http://{addr}")).await;

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = text_of(response).await;
        assert_eq!(
            body,
            "Failed to read upstream error body: error decoding response body"
        );
    }

    /// The plain streaming path's answer to a streamed request for the worker
    /// at `worker_url`, over empty stores.
    async fn streamed_answer_from(worker_url: &str) -> Response {
        use std::sync::Arc;

        use openai_protocol::responses::ResponsesRequest;
        use serde_json::json;
        use smg_data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        };

        use crate::{
            openai::context::StorageHandles,
            test_support::{http_client, StubWorker},
        };

        let request: ResponsesRequest =
            serde_json::from_value(json!({"model": "m", "input": "hi", "stream": true}))
                .expect("a request");
        let req = StreamingRequest {
            url: format!("{worker_url}/v1/responses"),
            payload: json!({}),
            original_body: request,
            previous_response_id: None,
            existing_mcp_list_tools_labels: Vec::new(),
            storage: StorageHandles {
                response: Arc::new(MemoryResponseStorage::new()),
                conversation: Arc::new(MemoryConversationStorage::new()),
                conversation_item: Arc::new(MemoryConversationItemStorage::new()),
                request_context: None,
                tenant: None,
            },
        };

        handle_simple_streaming_passthrough(&http_client(), &StubWorker::at(worker_url), None, req)
            .await
    }

    /// A header the upstream sent more than once reaches the client of a
    /// streamed answer with every value, in order: the relay's copy of the
    /// preserved headers appends a pair per value instead of keeping the last.
    #[tokio::test]
    async fn the_streamed_answer_keeps_every_value_of_a_repeated_upstream_header() {
        use std::sync::Arc;

        use axum::{
            http::{
                header::{CONTENT_TYPE, SET_COOKIE},
                StatusCode,
            },
            response::AppendHeaders,
            routing::any,
            Router,
        };
        use openai_protocol::responses::ResponsesRequest;
        use serde_json::json;
        use smg_data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        };

        use crate::{
            openai::context::StorageHandles,
            test_support::{http_client, StubWorker},
        };

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a listener");
        let port = listener.local_addr().expect("an address").port();
        #[expect(
            clippy::disallowed_methods,
            reason = "test stub upstream answering one event stream; aborted at the test's end"
        )]
        let upstream = tokio::spawn(async move {
            // `AppendHeaders`: the plain header array would insert and keep one cookie.
            let app = Router::new().fallback(any(|| async {
                (
                    AppendHeaders([
                        (CONTENT_TYPE, "text/event-stream"),
                        (SET_COOKIE, "a=1"),
                        (SET_COOKIE, "b=2"),
                    ]),
                    "event: response.completed\ndata: {}\n\n",
                )
            }));
            axum::serve(listener, app).await.expect("the stub serves");
        });
        let worker_url = format!("http://127.0.0.1:{port}");
        let request: ResponsesRequest =
            serde_json::from_value(json!({"model": "m", "input": "hi", "stream": true}))
                .expect("a request");
        let req = StreamingRequest {
            url: format!("{worker_url}/v1/responses"),
            payload: json!({}),
            original_body: request,
            previous_response_id: None,
            existing_mcp_list_tools_labels: Vec::new(),
            storage: StorageHandles {
                response: Arc::new(MemoryResponseStorage::new()),
                conversation: Arc::new(MemoryConversationStorage::new()),
                conversation_item: Arc::new(MemoryConversationItemStorage::new()),
                request_context: None,
                tenant: None,
            },
        };

        let response = handle_simple_streaming_passthrough(
            &http_client(),
            &StubWorker::at(&worker_url),
            None,
            req,
        )
        .await;
        upstream.abort();

        assert_eq!(response.status(), StatusCode::OK);
        let cookies: Vec<_> = response
            .headers()
            .get_all(SET_COOKIE)
            .iter()
            .map(|value| value.to_str().expect("an ascii cookie"))
            .collect();
        assert_eq!(cookies, ["a=1", "b=2"]);
    }

    /// The relay's inputs: a stream of chunks, one SSE block per chunk.
    fn chunks(blocks: &[&str]) -> impl Stream<Item = reqwest::Result<Bytes>> + Unpin {
        let frames: Vec<reqwest::Result<Bytes>> = blocks
            .iter()
            .map(|block| Ok(Bytes::from(format!("{block}\n\n"))))
            .collect();
        futures_util::stream::iter(frames)
    }

    const CREATED: &str = "event: response.created\ndata: {\"type\":\"response.created\",\
                           \"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\
                           \"response\",\"created_at\":1,\"model\":\"m\",\"status\":\
                           \"in_progress\",\"output\":[]}}";
    const IN_PROGRESS: &str = "event: response.in_progress\ndata: {\"type\":\
                               \"response.in_progress\",\"sequence_number\":1,\"response\":{\"id\":\
                               \"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"m\",\
                               \"status\":\"in_progress\",\"output\":[]}}";
    const DELTA: &str = "event: response.output_text.delta\ndata: {\"type\":\
                         \"response.output_text.delta\",\"sequence_number\":2,\"item_id\":\"msg_1\",\
                         \"output_index\":0,\"content_index\":0,\"delta\":\"hi\"}";
    const COMPLETED: &str = "event: response.completed\ndata: {\"type\":\"response.completed\",\
                             \"sequence_number\":3,\"response\":{\"id\":\"resp_1\",\"object\":\
                             \"response\",\"created_at\":1,\"model\":\"m\",\"status\":\
                             \"completed\",\"output\":[]}}";
    const CALL_ADDED: &str = "event: response.output_item.added\ndata: {\"type\":\
                              \"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\
                              \"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\
                              \"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}";
    const CALL_DONE: &str = "event: response.function_call_arguments.done\ndata: {\"type\":\
                             \"response.function_call_arguments.done\",\"sequence_number\":2,\
                             \"item_id\":\"fc_1\",\"output_index\":0,\"arguments\":\"{}\"}";

    /// Relays `blocks` through a fresh context with no session; the client
    /// reads the frames as they come, or is gone before the first one. Returns
    /// the relay's end, the frames, the sequence number and the count of
    /// blocks the relay left unread.
    async fn relayed(
        blocks: &[&str],
        flags: &mut RelayFlags,
        handler: &mut StreamingToolHandler,
        connected: bool,
    ) -> (RelayEnd, Vec<String>, u64, usize) {
        let request = ResponsesRequest {
            model: "m".to_string(),
            input: ResponseInput::Text("hi".to_string()),
            ..Default::default()
        };
        let ctx = StreamingEventContext {
            original_request: &request,
            previous_response_id: None,
            session: None,
            mcp_format_registry: None,
        };
        let (tx, rx) = sse_channel();
        let rx = connected.then_some(rx);
        let mut encoder = SseEncoder::new();
        let mut sequence_number = 0;
        let mut upstream = chunks(blocks);
        let relay = async {
            let end = relay_upstream_stream(
                &mut upstream,
                handler,
                flags,
                &[],
                &tx,
                &mut encoder,
                &ctx,
                &mut sequence_number,
            )
            .await;
            drop(tx);
            end
        };
        let read = async {
            let mut frames = Vec::new();
            if let Some(mut rx) = rx {
                while let Some(frame) = rx.recv().await {
                    let frame = frame.expect("a frame");
                    frames.push(String::from_utf8(frame.to_vec()).expect("utf-8"));
                }
            }
            frames
        };
        let (end, frames) = tokio::join!(relay, read);
        let unread = upstream.count().await;
        (end, frames, sequence_number, unread)
    }

    /// The first stream's events reach the client in order, re-stamped from
    /// the stream's own sequence number; the upstream's `response.completed`
    /// is not among them (the loop's end sends the terminal); the relay ends
    /// answered with the response id kept and the list-tools step taken.
    #[tokio::test]
    async fn the_first_stream_is_forwarded_and_ends_answered() {
        let mut flags = RelayFlags {
            first_iteration: true,
            mcp_list_tools_sent: false,
        };
        let mut handler = StreamingToolHandler::with_starting_index(0);
        let (end, frames, sequence_number, unread) = relayed(
            &[CREATED, IN_PROGRESS, DELTA, COMPLETED],
            &mut flags,
            &mut handler,
            true,
        )
        .await;
        assert_eq!(end, RelayEnd::Answered);
        assert_eq!(unread, 0, "the stream is read to its end");
        let events: Vec<&str> = frames
            .iter()
            .map(|frame| frame.split('\n').next().expect("an event line"))
            .collect();
        assert_eq!(
            events,
            [
                "event: response.created",
                "event: response.in_progress",
                "event: response.output_text.delta"
            ]
        );
        assert_eq!(sequence_number, 3);
        assert_eq!(handler.original_response_id(), Some("resp_1"));
        assert!(flags.mcp_list_tools_sent);
    }

    /// A resumed stream (after a tool round) skips its own `response.created`
    /// and `response.in_progress`: the client saw the first stream's.
    #[tokio::test]
    async fn a_resumed_stream_skips_created_and_in_progress() {
        let mut flags = RelayFlags {
            first_iteration: false,
            mcp_list_tools_sent: true,
        };
        let mut handler = StreamingToolHandler::with_starting_index(1);
        let (end, frames, sequence_number, unread) = relayed(
            &[CREATED, IN_PROGRESS, DELTA, COMPLETED],
            &mut flags,
            &mut handler,
            true,
        )
        .await;
        assert_eq!(end, RelayEnd::Answered);
        assert_eq!(unread, 0, "the stream is read to its end");
        assert_eq!(frames.len(), 1, "{frames:?}");
        assert!(frames[0].starts_with("event: response.output_text.delta"));
        assert_eq!(sequence_number, 1);
    }

    /// A function call the upstream completes ends the relay with the call
    /// pending in the handler: the loop's tool phase runs it, and the blocks
    /// after the trigger are not read by this relay.
    #[tokio::test]
    async fn a_completed_tool_call_ends_the_relay_with_the_call_pending() {
        let mut flags = RelayFlags {
            first_iteration: true,
            mcp_list_tools_sent: false,
        };
        let mut handler = StreamingToolHandler::with_starting_index(0);
        let (end, _frames, _, unread) = relayed(
            &[CREATED, CALL_ADDED, CALL_DONE, COMPLETED],
            &mut flags,
            &mut handler,
            true,
        )
        .await;
        assert_eq!(end, RelayEnd::ToolCalls);
        assert_eq!(unread, 1, "the completed after the trigger is not read");
        assert_eq!(handler.take_pending_calls().len(), 1);
    }

    /// A client that is gone before the first frame stops the relay at that
    /// frame: the frame is numbered, its send fails, nothing is read further
    /// and nothing reaches the client.
    #[tokio::test]
    async fn a_gone_client_stops_the_relay() {
        let mut flags = RelayFlags {
            first_iteration: true,
            mcp_list_tools_sent: false,
        };
        let mut handler = StreamingToolHandler::with_starting_index(0);
        let (end, frames, sequence_number, unread) = relayed(
            &[CREATED, DELTA, COMPLETED],
            &mut flags,
            &mut handler,
            false,
        )
        .await;
        assert_eq!(end, RelayEnd::Stopped);
        assert_eq!(unread, 2, "the relay stops at the first refused send");
        assert!(frames.is_empty());
        assert_eq!(sequence_number, 1);
    }

    /// The whole interception path against a stub upstream that answers one
    /// four-event stream and its own end sentinel: the client reads the
    /// stream's events, the loop's terminal and one sentinel, the loop's.
    #[tokio::test]
    async fn a_tool_interception_stream_ends_with_the_terminal_and_the_sentinel() {
        use std::sync::Arc;

        use axum::{
            http::{header::CONTENT_TYPE, StatusCode},
            routing::any,
            Router,
        };
        use smg_data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        };
        use smg_mcp::McpConfig;

        use crate::{
            openai::context::StorageHandles, openai_bridge::FormatRegistry,
            test_support::http_client,
        };

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a listener");
        let port = listener.local_addr().expect("an address").port();
        let body = [CREATED, IN_PROGRESS, DELTA, COMPLETED, "data: [DONE]"].join("\n\n") + "\n\n";
        #[expect(
            clippy::disallowed_methods,
            reason = "test stub upstream answering one event stream per request; aborted at the test's end"
        )]
        let upstream = tokio::spawn(async move {
            let app = Router::new().fallback(any(move || {
                let body = body.clone();
                async move { ([(CONTENT_TYPE, "text/event-stream")], body) }
            }));
            axum::serve(listener, app).await.expect("the stub serves");
        });
        let orchestrator = Arc::new(
            McpOrchestrator::new(McpConfig::default())
                .await
                .expect("an orchestrator with no servers"),
        );
        let client = http_client();
        let storage = StorageHandles {
            response: Arc::new(MemoryResponseStorage::new()),
            conversation: Arc::new(MemoryConversationStorage::new()),
            conversation_item: Arc::new(MemoryConversationItemStorage::new()),
            request_context: None,
            tenant: None,
        };
        let request: ResponsesRequest =
            serde_json::from_value(json!({"model": "m", "input": "hi", "stream": true}))
                .expect("a request");
        let req = StreamingRequest {
            url: format!("http://127.0.0.1:{port}/v1/responses"),
            payload: json!({"model": "m", "input": "hi", "stream": true}),
            original_body: request,
            previous_response_id: None,
            existing_mcp_list_tools_labels: Vec::new(),
            storage: StorageHandles {
                response: Arc::clone(&storage.response),
                conversation: Arc::clone(&storage.conversation),
                conversation_item: Arc::clone(&storage.conversation_item),
                request_context: None,
                tenant: None,
            },
        };
        let response = handle_streaming_with_tool_interception(
            &client,
            None,
            None,
            req,
            &orchestrator,
            FormatRegistry::new(),
            Vec::new(),
        );
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the stream to its end");
        let text = String::from_utf8(bytes.to_vec()).expect("utf-8");
        assert!(text.contains("event: response.created\n"), "{text}");
        assert!(
            text.contains("event: response.output_text.delta\n"),
            "{text}"
        );
        assert!(text.contains("event: response.completed\n"), "{text}");
        assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
        // A reader that stops at the first sentinel has read the terminal.
        assert_eq!(text.matches("data: [DONE]").count(), 1, "{text}");
        upstream.abort();
    }

    /// The streaming MCP relay hands its request payload to the loop it
    /// spawns instead of copying it: what the call itself allocates before
    /// the loop runs a single turn (the task, the channel, the loop's own
    /// state) does not grow with the request body, where a copy of the whole
    /// JSON tree would grow with it byte for byte. The stub upstream's
    /// received body is the one serialization the turn needs, as before. The
    /// counts are printed for the Numbers table.
    #[tokio::test]
    #[expect(
        clippy::print_stderr,
        reason = "the exchange's allocation counts, read from the test's output by the measurement of the relay"
    )]
    async fn a_streaming_mcp_request_hands_its_payload_to_the_loop_without_a_copy() {
        use std::sync::{Arc, Mutex};

        use axum::{
            extract::Request,
            http::{header::CONTENT_TYPE, StatusCode},
            routing::any,
            Router,
        };
        use smg_data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        };
        use smg_mcp::McpConfig;

        use crate::{
            openai::context::StorageHandles,
            openai_bridge::FormatRegistry,
            test_support::{counting_alloc, http_client},
        };

        let orchestrator = Arc::new(
            McpOrchestrator::new(McpConfig::default())
                .await
                .expect("an orchestrator with no servers"),
        );
        let client = http_client();
        let answer = [CREATED, IN_PROGRESS, COMPLETED].join("\n\n") + "\n\n";

        // (input length, body length, bytes the call allocated) per request.
        let mut by_the_call = Vec::new();
        for input_len in [2usize, 64 * 1024] {
            let input = "x".repeat(input_len);
            let payload = json!({"model": "m", "input": input, "stream": true});
            let body_len = serde_json::to_vec(&payload).expect("the body").len();
            let request: ResponsesRequest =
                serde_json::from_value(payload.clone()).expect("a request");

            // The stub keeps the body of the one request it answers.
            let received: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("a listener");
            let port = listener.local_addr().expect("an address").port();
            let (bodies, answer) = (Arc::clone(&received), answer.clone());
            #[expect(
                clippy::disallowed_methods,
                reason = "test stub upstream answering one event stream per request; aborted at the test's end"
            )]
            let upstream = tokio::spawn(async move {
                let app = Router::new().fallback(any(move |req: Request| {
                    let (bodies, answer) = (Arc::clone(&bodies), answer.clone());
                    async move {
                        let body = axum::body::to_bytes(req.into_body(), usize::MAX)
                            .await
                            .expect("the request body");
                        bodies.lock().expect("the bodies").push(body.to_vec());
                        ([(CONTENT_TYPE, "text/event-stream")], answer)
                    }
                }));
                axum::serve(listener, app).await.expect("the stub serves");
            });
            let req = StreamingRequest {
                url: format!("http://127.0.0.1:{port}/v1/responses"),
                payload,
                original_body: request,
                previous_response_id: None,
                existing_mcp_list_tools_labels: Vec::new(),
                storage: StorageHandles {
                    response: Arc::new(MemoryResponseStorage::new()),
                    conversation: Arc::new(MemoryConversationStorage::new()),
                    conversation_item: Arc::new(MemoryConversationItemStorage::new()),
                    request_context: None,
                    tenant: None,
                },
            };

            // The call spawns the loop and returns; on this current-thread
            // runtime the loop has not run yet, so the counters move only by
            // what the call itself allocated.
            let (calls_before, bytes_before) = (counting_alloc::calls(), counting_alloc::bytes());
            let response = handle_streaming_with_tool_interception(
                &client,
                None,
                None,
                req,
                &orchestrator,
                FormatRegistry::new(),
                Vec::new(),
            );
            let bytes_by_the_call = counting_alloc::bytes() - bytes_before;
            assert_eq!(response.status(), StatusCode::OK);
            let stream = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("the stream to its end");
            let (exchange_calls, exchange_bytes) = (
                counting_alloc::calls() - calls_before,
                counting_alloc::bytes() - bytes_before,
            );
            upstream.abort();

            eprintln!(
                "input {input_len} bytes (body {body_len} bytes): the call allocated {bytes_by_the_call} bytes before the loop ran; the exchange {exchange_calls} allocation calls, {exchange_bytes} bytes"
            );
            let stream = String::from_utf8(stream.to_vec()).expect("utf-8");
            assert!(stream.ends_with("data: [DONE]\n\n"), "{stream}");
            let bodies = received.lock().expect("the bodies");
            assert_eq!(bodies.len(), 1, "one turn, one upstream request");
            let on_the_wire: Value = serde_json::from_slice(&bodies[0]).expect("the body is JSON");
            assert_eq!(on_the_wire["input"].as_str().map(str::len), Some(input_len));
            assert_eq!(on_the_wire["model"], json!("m"));
            by_the_call.push((input_len, body_len, bytes_by_the_call));
        }

        let [(_, _, small), (_, large_body, large)] = by_the_call[..] else {
            panic!("two requests measured, {by_the_call:?}");
        };
        let growth = large.saturating_sub(small);
        assert!(
            growth * 4 < large_body,
            "the call's allocation grew by {growth} bytes for a {large_body}-byte body (small request {small} bytes, large {large}): the payload was copied, not moved"
        );
    }

    /// A tool loop whose upstream answers with a large final response (64
    /// message items of 2 KiB) ends with the same items on the wire, in the
    /// `response.completed` event, and in the stored record, the record
    /// carrying the upstream's status. The allocation count of the whole
    /// exchange is printed for the measurement of the loop's end.
    #[tokio::test]
    async fn a_large_final_response_is_streamed_and_stored_with_the_same_items() {
        use std::sync::Arc;

        use axum::{
            http::{header::CONTENT_TYPE, StatusCode},
            routing::any,
            Router,
        };
        use smg_data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
            ResponseId,
        };
        use smg_mcp::McpConfig;

        use crate::{
            openai::context::StorageHandles,
            openai_bridge::FormatRegistry,
            test_support::{counting_alloc, http_client},
        };

        let text = "x".repeat(2048);
        let items: Vec<Value> = (0..64)
            .map(|i| {
                json!({
                    "type": "message", "id": format!("msg_{i}"), "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]
                })
            })
            .collect();
        let completed = format!(
            "event: response.completed\ndata: {}",
            json!({
                "type": "response.completed", "sequence_number": 3,
                "response": {
                    "id": "resp_1", "object": "response", "created_at": 1, "model": "m",
                    "status": "incomplete", "output": items
                }
            })
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a listener");
        let port = listener.local_addr().expect("an address").port();
        let body = [CREATED, IN_PROGRESS, completed.as_str()].join("\n\n") + "\n\n";
        #[expect(
            clippy::disallowed_methods,
            reason = "test stub upstream answering one event stream per request; aborted at the test's end"
        )]
        let upstream = tokio::spawn(async move {
            let app = Router::new().fallback(any(move || {
                let body = body.clone();
                async move { ([(CONTENT_TYPE, "text/event-stream")], body) }
            }));
            axum::serve(listener, app).await.expect("the stub serves");
        });
        let orchestrator = Arc::new(
            McpOrchestrator::new(McpConfig::default())
                .await
                .expect("an orchestrator with no servers"),
        );
        let client = http_client();
        let storage = StorageHandles {
            response: Arc::new(MemoryResponseStorage::new()),
            conversation: Arc::new(MemoryConversationStorage::new()),
            conversation_item: Arc::new(MemoryConversationItemStorage::new()),
            request_context: None,
            tenant: None,
        };
        let request: ResponsesRequest =
            serde_json::from_value(json!({"model": "m", "input": "hi", "stream": true}))
                .expect("a request");
        let req = StreamingRequest {
            url: format!("http://127.0.0.1:{port}/v1/responses"),
            payload: json!({"model": "m", "input": "hi", "stream": true}),
            original_body: request,
            previous_response_id: None,
            existing_mcp_list_tools_labels: Vec::new(),
            storage: StorageHandles {
                response: Arc::clone(&storage.response),
                conversation: Arc::clone(&storage.conversation),
                conversation_item: Arc::clone(&storage.conversation_item),
                request_context: None,
                tenant: None,
            },
        };

        let allocations_before = counting_alloc::calls();
        let response = handle_streaming_with_tool_interception(
            &client,
            None,
            None,
            req,
            &orchestrator,
            FormatRegistry::new(),
            Vec::new(),
        );
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the stream to its end");
        let stored = storage
            .response
            .get_response(&ResponseId::from("resp_1"))
            .await
            .expect("the storage answers")
            .expect("the record");
        let allocations = counting_alloc::calls() - allocations_before;
        #[expect(
            clippy::print_stderr,
            reason = "the exchange's allocation count, read from the test's output by the measurement of the loop's end"
        )]
        {
            eprintln!("allocations during the streamed loop with 64 items: {allocations}");
        }
        upstream.abort();

        let stream = String::from_utf8(bytes.to_vec()).expect("utf-8");
        let completed_frame = stream
            .split("\n\n")
            .find(|frame| frame.starts_with("event: response.completed\n"))
            .expect("the terminal event");
        let data = completed_frame
            .strip_prefix("event: response.completed\ndata: ")
            .expect("the terminal's data");
        let event: Value = serde_json::from_str(data).expect("the terminal's JSON");
        let on_the_wire = &event["response"];
        assert_eq!(on_the_wire["id"], json!("resp_1"));
        assert_eq!(on_the_wire["status"], json!("completed"));
        assert_eq!(on_the_wire["output"].as_array().map(Vec::len), Some(64));
        assert_eq!(
            stored.raw_response["output"], on_the_wire["output"],
            "the record and the wire carry the same items"
        );
        assert_eq!(
            stored.raw_response["status"],
            json!("incomplete"),
            "the record keeps the status the upstream reported; the wire said completed"
        );
        assert!(stream.ends_with("data: [DONE]\n\n"), "{stream}");
    }
}
