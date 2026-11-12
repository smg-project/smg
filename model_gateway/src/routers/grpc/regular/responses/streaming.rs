//! Streaming execution for Regular Responses API
//!
//! This module handles streaming request execution:
//! - `execute_tool_loop_streaming` - MCP tool loop with streaming
//! - `convert_chat_stream_to_responses_stream` - Non-MCP streaming conversion
//!
//! A stream's final object (the terminal event's and the stored one) is the
//! event emitter's: the items its events announced, under the ids they carry.

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt,
    sync::{Arc, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::{Body, BodyDataStream},
    response::Response,
};
use bytes::Bytes;
use futures_util::StreamExt;
use memchr::memmem;
use openai_protocol::{
    chat::{
        ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
        ChatCompletionStreamResponse,
    },
    common::{FunctionCallResponse, ToolCall, ToolCallDelta, Usage},
    responses::{ResponseStatus, ResponsesRequest, ResponsesUsage},
};
use serde_json::{json, Value};
use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseStorage,
};
use smg_mcp::{McpServerBinding, McpToolSession, ToolExecutionInput};
use tracing::{debug, trace, warn};
use uuid::Uuid;

use super::{
    common::{
        build_next_request, convert_mcp_tools_to_chat_tools, dispatch_arguments,
        extract_all_tool_calls_from_chat, has_unfinished_mcp_call, partition_tool_calls,
        prepare_tool_choice, record_mcp_call, ExtractedToolCall, McpToolCallLimit,
        ResponsesCallContext, ToolLoopState,
    },
    conversions,
};
use crate::{
    observability::metrics::Metrics,
    routers::{
        common::{
            mcp_utils::DEFAULT_MAX_ITERATIONS,
            openai_bridge::{self, FormatRegistry, ResponseFormat},
            sse::{sse_channel, SseSender},
        },
        error,
        grpc::{
            common::{
                responses::{
                    await_stream_startup, build_sse_response, naming_parser,
                    persist_response_if_needed, signal_stream_startup, stored_foreground,
                    stream_startup_channel,
                    streaming::{
                        attach_mcp_server_label, ResponseStreamEventEmitter, StreamStartupSender,
                    },
                    BackgroundStream, ResponsesContext,
                },
                stages::{iteration_charge, RateLimitCell},
            },
            regular::symphony,
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
    background: Option<BackgroundStream>,
) -> Response {
    debug!("Converting chat SSE stream to responses SSE format");
    let response_id = params.response_id;
    let tenant = crate::tenant::storage_tenant(&params.tenant_request_meta);

    // Get chat streaming response, under the request's tenant rate-limit
    // reservation, taken under a charge id of the iteration's own: the chat
    // stream settles it with the engine's counts as it ends (the handoff to
    // the body), a refused or failed start closes it with its debit kept.
    let reservation = Arc::new(RateLimitCell::new());
    let chat_response = ctx
        .pipeline
        .execute_chat(
            chat_request,
            params.headers,
            params.model_id,
            ctx.components.clone(),
            Some(iteration_charge(&params.tenant_request_meta)),
            Some(reservation.clone()),
            // Tool-loop iterations are not retried.
            None,
            // No startup barrier: the status is checked below before the
            // stream is forwarded.
            None,
        )
        .await;

    if !chat_response.status().is_success() {
        reservation.close_unsettled().await;
        return chat_response;
    }

    // The chat answer's body, and the parser that read it, which this
    // answer names as the chat route does.
    let (parts, body) = chat_response.into_parts();
    let parser = parts.headers.get(symphony::HEADER).cloned();

    // Create channel for transformed SSE events
    let (tx, rx) = sse_channel();

    // Spawn background task to transform stream
    let original_request_clone = original_request.clone();
    let response_storage = ctx.response_storage.clone();
    let conversation_storage = ctx.conversation_storage.clone();
    let conversation_item_storage = ctx.conversation_item_storage.clone();
    let request_context = ctx.request_context.clone();
    // A background stream's events go through a relay, so the work and its
    // terminal record outlive a client that leaves.
    let tx = match &background {
        Some(_) => BackgroundStream::detach(tx),
        None => tx,
    };

    // The transform answers the error it ended with (after the error frame
    // went to the client), so the runner of a background stream can store it.
    let transform = async move {
        match process_and_transform_sse_stream(
            body,
            original_request_clone,
            response_id,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            tenant,
            request_context,
            tx.clone(),
        )
        .await
        {
            Ok(()) => Ok(()),
            Err(e) => {
                warn!("Error transforming SSE stream: {}", e);
                utils::send_error_sse(&tx, &e, "stream_error").await;
                Err(json!({ "code": "stream_error", "message": e }))
            }
        }
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "streaming task is fire-and-forget; client disconnect terminates it unless it runs a background response, whose slot bounds it"
    )]
    match background {
        Some(run) => {
            let storage = ctx.response_storage.clone();
            let request_context = ctx.request_context.clone();
            tokio::spawn(run.run(storage, request_context, transform));
        }
        None => {
            tokio::spawn(transform);
        }
    }

    // Build SSE response with transformed stream
    naming_parser(build_sse_response(rx), parser)
}

/// The text of one frame of the chat stream's body: the gateway's own
/// output, valid UTF-8, borrowed after the standard validator's one pass
/// (its ASCII fast path reads a word at a time; the lossy conversion's
/// chunk walk reads a byte at a time and held 5.5 % of a Responses request's
/// user instructions). Bytes that are not UTF-8 are replaced as before.
fn frame_text(chunk: &[u8]) -> Cow<'_, str> {
    match std::str::from_utf8(chunk) {
        Ok(text) => Cow::Borrowed(text),
        Err(_) => String::from_utf8_lossy(chunk),
    }
}

/// The SSE events of one frame of the chat stream's body.
///
/// The chat pipeline writes whole events to the body, but not one per frame:
/// a frame carries every chunk the engine had ready when the stream loop ran,
/// and a re-sliced stream every slice cut from one chunk. Read as one event,
/// such a frame parses as nothing and everything after its first event is lost.
fn sse_events(frame: &str) -> impl Iterator<Item = &str> {
    // The blank line found by `memchr`'s substring searcher (a dependency of
    // the gateway already), built once for the two bytes: a remainder at
    // least a vector and a byte long takes its vector search, a shorter one
    // (the usual tail: a keep-alive, the terminator, nothing) its Rabin-Karp
    // scan. `str::split("\n\n")` built the two-way substring searcher per
    // call, which held 4.6 % of a Responses request's user instructions, and
    // the one-shot `memmem::find` builds a searcher per event as well. A split
    // at an ASCII pair is at a character boundary, so the slices are text.
    static BLANK_LINE: OnceLock<memmem::Finder<'static>> = OnceLock::new();
    let blank_line = BLANK_LINE.get_or_init(|| memmem::Finder::new(b"\n\n"));
    let mut rest = Some(frame);
    std::iter::from_fn(move || {
        let text = rest?;
        Some(match blank_line.find(text.as_bytes()) {
            Some(end) => {
                rest = Some(&text[end + 2..]);
                &text[..end]
            }
            None => {
                rest = None;
                text
            }
        })
    })
    .map(str::trim)
    .filter(|event| !event.is_empty())
}

/// One `data:` event of the chat stream's body, as the Responses conversions
/// read it: the chat chunk it carries, the chat stream's error event as the
/// error object a Responses terminal carries, the `[DONE]` terminator, or any
/// other data event, left to the reader. Lines that are not data events (a
/// comment, a keep-alive) are nobody's and are not yielded.
enum ChatSseEvent<'a> {
    Chunk(ChatCompletionStreamResponse),
    UpstreamError(Value),
    Done,
    Other(&'a str),
}

/// The data events of one frame of the chat stream's body, in order; the
/// reader stops at [`ChatSseEvent::Done`].
fn chat_sse_events(frame: &str) -> impl Iterator<Item = ChatSseEvent<'_>> {
    sse_events(frame).filter_map(|event| {
        if event == "data: [DONE]" {
            return Some(ChatSseEvent::Done);
        }
        let data = event.strip_prefix("data: ")?.trim();
        if let Ok(chunk) = serde_json::from_str::<ChatCompletionStreamResponse>(data) {
            return Some(ChatSseEvent::Chunk(chunk));
        }
        let error = serde_json::from_str::<Value>(data)
            .ok()
            .and_then(|value| value.get("error").map(upstream_error_object));
        Some(match error {
            Some(error) => ChatSseEvent::UpstreamError(error),
            None => ChatSseEvent::Other(event),
        })
    })
}

/// The error object of a Responses terminal for the chat stream's error
/// event: its `code`, else its `type`, else `server_error`; its `message`,
/// else the generic wording.
fn upstream_error_object(error: &Value) -> Value {
    json!({
        "code": error.get("code").and_then(Value::as_str)
            .or_else(|| error.get("type").and_then(Value::as_str)).unwrap_or("server_error"),
        "message": error.get("message").and_then(Value::as_str).unwrap_or("Upstream generation failed"),
    })
}

/// The error object of a Responses terminal for a chat body the gateway
/// could not read.
fn stream_read_error(error: &impl fmt::Display) -> Value {
    json!({"code": "stream_error", "message": format!("Stream read error: {error}")})
}

/// One read of the chat body for the client at the receiving end of the
/// Responses stream's channel (see [`next_frame`]).
enum BodyRead {
    /// A frame of the body, or the error the read met.
    Frame(Result<Bytes, axum::Error>),
    /// The body ended.
    End,
    /// The client left before the body yielded.
    ClientLeft,
}

/// The next frame of the chat body `stream`, read for the client at the
/// receiving end of `tx`'s channel. The relay that turns the chat stream
/// into the Responses stream holds the chat layer's receiver through that
/// body: parked on a body that stalled mid-stream, it would learn that its
/// client left only when the body next yields, and the engine request behind
/// the body would run on as long. Here every read races the body against the
/// client's leaving, looked at first so a busy body cannot outrun it; on
/// [`BodyRead::ClientLeft`] the caller ends its read and drops the body,
/// and the chat layer ends the engine request the way it does for its own
/// clients. A background response's `tx` is the sender of its relay, which
/// closes with the work and not with the client: its reads are the body's
/// own, and the work outlives the client as it is meant to.
async fn next_frame(stream: &mut BodyDataStream, tx: &SseSender) -> BodyRead {
    tokio::select! {
        biased;
        () = tx.closed() => BodyRead::ClientLeft,
        frame = stream.next() => frame.map_or(BodyRead::End, BodyRead::Frame),
    }
}

/// The record of a stored foreground stream whose client left mid-stream, as
/// the `response.created` it announced promised: the items streamed so far
/// (the open ones closed, in progress), status `cancelled` when no finish had
/// been read (a finish the stream had read stands, with its own statuses), the
/// status the background cancel path writes for a response stopped before its end; the
/// GET and a `previous_response_id` chain answer from it. The emitter is
/// abandoned first: its item events are built into the object and sent to
/// nobody. An unstored stream records nothing, as before; a background one
/// never meets the departure (its relay outlives the client). The chat body
/// has gone with the frame loop, which ended the generation on the engine.
/// The answer is the departure, as before.
#[expect(
    clippy::too_many_arguments,
    reason = "the record takes the storage handles of the transform it ends"
)]
async fn record_departure(
    mut event_emitter: ResponseStreamEventEmitter,
    usage: Option<Usage>,
    original_request: &ResponsesRequest,
    response_storage: Arc<dyn ResponseStorage>,
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    tenant: Option<&str>,
    request_context: Option<StorageRequestContext>,
    tx: &SseSender,
) -> Result<(), String> {
    if stored_foreground(original_request) {
        event_emitter.abandon();
        // A finish the stream had already read stands (a `length` finish
        // keeps its `incomplete_details`); only a generation cut before its
        // finish is recorded as cancelled.
        if event_emitter.finish_reason().is_none() {
            event_emitter.set_finish_reason(Some("cancelled"));
        }
        event_emitter.close_items(tx).await?;
        let final_response = event_emitter.finalize(usage);
        if let Err(e) = persist_response_if_needed(
            conversation_storage,
            conversation_item_storage,
            response_storage,
            &final_response,
            original_request,
            tenant,
            request_context,
        )
        .await
        {
            debug!(
                response_id = %final_response.id,
                error = %e.message,
                "The record of a stream its client left could not be written"
            );
        }
    }
    Err("Client disconnected".to_string())
}

/// Process chat SSE stream and transform to responses format
#[expect(
    clippy::too_many_arguments,
    reason = "the stream transform takes the storage handles and the pre-stored id of a background response"
)]
async fn process_and_transform_sse_stream(
    body: Body,
    original_request: ResponsesRequest,
    response_id: Option<String>,
    response_storage: Arc<dyn ResponseStorage>,
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    tenant: Option<String>,
    request_context: Option<StorageRequestContext>,
    tx: SseSender,
) -> Result<(), String> {
    // The usage of the last chunk that carried one, for the final object.
    let mut usage: Option<Usage> = None;

    // Create event emitter for OpenAI-compatible streaming; a background
    // response streams under the id its queued record was stored under.
    let response_id = response_id.unwrap_or_else(|| format!("resp_{}", Uuid::now_v7()));
    let model = original_request.model.clone();
    let created_at = chrono::Utc::now().timestamp() as u64;
    let mut event_emitter = ResponseStreamEventEmitter::new(response_id, model, created_at);
    event_emitter.set_original_request(original_request.clone());

    // Emit initial response.created and response.in_progress events (a
    // background stream reports response.queued between the two)
    let event = event_emitter.emit_created();
    event_emitter
        .send_event(&event, &tx)
        .await
        .map_err(|_| "Failed to send response.created event".to_string())?;

    if original_request.background.unwrap_or(false) {
        let event = event_emitter.emit_queued();
        event_emitter
            .send_event(&event, &tx)
            .await
            .map_err(|_| "Failed to send response.queued event".to_string())?;
    }

    let event = event_emitter.emit_in_progress();
    event_emitter
        .send_event(&event, &tx)
        .await
        .map_err(|_| "Failed to send response.in_progress event".to_string())?;

    // Convert body to data stream
    let mut stream = body.into_data_stream();

    let mut terminal_error = None;
    // The client left: met by the body read, by an event it no longer takes,
    // or by the emitter's send. The loop ends there.
    let mut client_left = false;
    // Process the stream frame by frame; a frame holds one or more SSE events.
    'frames: loop {
        let chunk = match next_frame(&mut stream, &tx).await {
            BodyRead::Frame(Ok(chunk)) => chunk,
            BodyRead::Frame(Err(error)) => {
                terminal_error = Some(stream_read_error(&error));
                break;
            }
            BodyRead::End => break,
            BodyRead::ClientLeft => {
                client_left = true;
                break;
            }
        };

        let frame = frame_text(&chunk);
        for event in chat_sse_events(&frame) {
            match event {
                ChatSseEvent::Done => break 'frames,
                ChatSseEvent::Chunk(chat_chunk) => {
                    if let Some(chunk_usage) = &chat_chunk.usage {
                        usage = Some(chunk_usage.clone());
                    }
                    if let Err(error) = event_emitter.process_chunk(&chat_chunk, &tx).await {
                        if tx.is_closed() {
                            client_left = true;
                            break 'frames;
                        }
                        return Err(error);
                    }
                }
                ChatSseEvent::UpstreamError(error) => {
                    terminal_error = Some(error);
                    break 'frames;
                }
                // Pass through unrecognized non-error events.
                ChatSseEvent::Other(event) => {
                    debug!("Non-chunk SSE event, passing through: {}", event);
                    if tx
                        .send(Ok(Bytes::from(format!("{event}\n\n"))))
                        .await
                        .is_err()
                    {
                        client_left = true;
                        break 'frames;
                    }
                }
            }
        }
    }

    // The chat body goes here, which ends the generation on the engine (on
    // the client's leaving as the early return did; on every other end the
    // body is spent). A stored foreground stream its client left is still
    // recorded, under the id it announced; the answer is the departure.
    drop(stream);
    if client_left {
        return record_departure(
            event_emitter,
            usage,
            &original_request,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            tenant.as_deref(),
            request_context,
            &tx,
        )
        .await;
    }

    // The final object is the emitter's own: the items its events announced,
    // under the ids they carry, so the stored object and the stream agree. A
    // stream cut short before its finish closes its open items first (a
    // generation that failed leaves them in progress), the record follows,
    // then the terminal event.
    if terminal_error.is_some() {
        event_emitter.set_finish_reason(Some("failed"));
    }
    event_emitter.close_items(&tx).await?;
    let final_response = match terminal_error.as_ref() {
        Some(error) => event_emitter.finalize_failed(usage, error.clone()),
        None => event_emitter.finalize(usage),
    };
    let usage_json = final_response.usage.as_ref().map(|usage| json!(usage));
    let persist = persist_response_if_needed(
        conversation_storage,
        conversation_item_storage,
        response_storage,
        &final_response,
        &original_request,
        tenant.as_deref(),
        request_context,
    );
    if stored_foreground(&original_request) {
        // A foreground response the client expects stored is written before
        // its terminal event: a record that could not be written ends the
        // stream with response.failed, not with a completed response that
        // can be neither retrieved nor continued.
        let not_stored = persist.await.err().map(|e| e.error_object());
        let error = terminal_error.as_ref().or(not_stored.as_ref());
        event_emitter
            .emit_terminal(usage_json.as_ref(), error, &tx)
            .await?;
    } else {
        // A background response's terminal write is retried behind the event
        // (its client polls the record); an unstored response stores
        // nothing. Neither fails the stream.
        event_emitter
            .emit_terminal(usage_json.as_ref(), terminal_error.as_ref(), &tx)
            .await?;
        if let Err(e) = persist.await {
            debug!(response_id = %final_response.id, error = %e.message, "Response not stored");
        }
    }

    Ok(())
}

// ============================================================================
// MCP Streaming Path
// ============================================================================

/// Execute MCP tool loop with streaming support
///
/// This streams each iteration's response to the client while accumulating
/// to check for tool calls. If tool calls are found, executes them and
/// continues with the next streaming iteration.
pub(super) async fn execute_tool_loop_streaming(
    ctx: &ResponsesContext,
    current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    params: ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
    background: Option<BackgroundStream>,
) -> Response {
    // Create SSE channel for client
    let (tx, rx) = sse_channel();
    let (startup_tx, startup_rx) = stream_startup_channel();

    // Clone data for background task
    let ctx_clone = ctx.clone();
    let original_request_clone = original_request.clone();
    // A background stream's events go through a relay, so the work and its
    // terminal record outlive a client that leaves.
    let tx = match &background {
        Some(_) => BackgroundStream::detach(tx),
        None => tx,
    };

    // The loop answers the error it ended with (after the error frame went
    // to the client), so the runner of a background stream can store it.
    let tool_loop = async move {
        match execute_tool_loop_streaming_internal(
            &ctx_clone,
            current_request,
            &original_request_clone,
            params,
            mcp_servers,
            tx.clone(),
            Some(startup_tx),
        )
        .await
        {
            Ok(()) => Ok(()),
            Err(e) => {
                warn!("Streaming tool loop error: {}", e);
                utils::send_error_sse(&tx, &e, "tool_loop_error").await;
                Err(json!({ "code": "tool_loop_error", "message": e }))
            }
        }
    };
    // Spawn background task for tool loop
    #[expect(
        clippy::disallowed_methods,
        reason = "streaming task is fire-and-forget; client disconnect terminates it unless it runs a background response, whose slot bounds it"
    )]
    match background {
        Some(run) => {
            let storage = ctx.response_storage.clone();
            let request_context = ctx.request_context.clone();
            tokio::spawn(run.run(storage, request_context, tool_loop));
        }
        None => {
            tokio::spawn(tool_loop);
        }
    }

    await_stream_startup(startup_rx, rx).await
}

/// What every MCP call of one turn shares.
struct CallSite<'a> {
    session: &'a McpToolSession<'a>,
    format_registry: &'a FormatRegistry,
    original_request: &'a ResponsesRequest,
    model: &'a str,
}

/// One MCP call's whole life on the stream: announced with its progress
/// events, run, closed with its completion or failure, done, and recorded.
async fn stream_mcp_call(
    site: &CallSite<'_>,
    state: &mut ToolLoopState,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &SseSender,
    tool_call: ExtractedToolCall,
) -> Result<(), String> {
    state.total_calls += 1;

    trace!(
        "Executing tool call {}/{}: {} (call_id: {})",
        state.total_calls,
        state.total_calls,
        tool_call.name,
        tool_call.call_id
    );

    let response_format =
        openai_bridge::lookup_tool_format(site.session, site.format_registry, &tool_call.name);

    // Use emitter helpers to determine correct type and allocate index
    let item_type = ResponseStreamEventEmitter::type_str_for_format(Some(&response_format));
    let resolved_label = site.session.resolve_tool_server_label(&tool_call.name);

    // Allocate output_index with the format's id-prefix discriminator
    // (e.g. `ws_…` for web_search_call); see FormatDescriptor.
    let (output_index, item_id) = emitter.allocate_output_index_for_format(Some(response_format));

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
    emitter.send_event(&event, tx).await?;

    // Emit tool_call.in_progress
    let event = emitter.emit_tool_call_in_progress(output_index, &item_id, response_format);
    emitter.send_event(&event, tx).await?;

    // Emit arguments events for mcp_call only (skip for builtin tools)
    if matches!(response_format, ResponseFormat::Passthrough) {
        // Emit mcp_call_arguments.delta (simulate streaming by sending full arguments)
        let event =
            emitter.emit_mcp_call_arguments_delta(output_index, &item_id, &tool_call.arguments);
        emitter.send_event(&event, tx).await?;

        // Emit mcp_call_arguments.done
        let event =
            emitter.emit_mcp_call_arguments_done(output_index, &item_id, &tool_call.arguments);
        emitter.send_event(&event, tx).await?;
    }

    // Emit searching/interpreting event for builtin tools
    if let Some(event) = emitter.emit_tool_call_searching(output_index, &item_id, response_format) {
        emitter.send_event(&event, tx).await?;
    }

    // Execute the MCP tool
    trace!(
        "Calling MCP tool '{}' with args: {}",
        tool_call.name,
        tool_call.arguments
    );
    let arguments =
        dispatch_arguments(response_format, site.original_request, &tool_call.arguments);

    // Execute the single tool via the normalized MCP execution API.
    // This avoids custom serialization and manual re-transformation in streaming paths.
    let tool_output = site
        .session
        .execute_tool(ToolExecutionInput {
            call_id: tool_call.call_id,
            tool_name: tool_call.name,
            arguments,
        })
        .await;

    let success = !tool_output.is_error;
    let output_str = tool_output.output.to_string();

    let output_item = openai_bridge::transform_tool_output(&tool_output, response_format);
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
        let event = emitter.emit_tool_call_completed(output_index, &item_id, response_format);
        emitter.send_event(&event, tx).await?;
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
            emitter.send_event(&event, tx).await?;
        } else {
            let event = emitter.emit_tool_call_completed(output_index, &item_id, response_format);
            emitter.send_event(&event, tx).await?;
        }
    }

    let event = emitter.emit_output_item_done(output_index, &item_done);
    emitter.send_event(&event, tx).await?;
    emitter.complete_output_item(output_index);

    record_mcp_call(state, site.model, tool_output, output_str, output_item);
    Ok(())
}

/// Internal streaming tool loop implementation
async fn execute_tool_loop_streaming_internal(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    params: ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
    tx: SseSender,
    mut startup: Option<StreamStartupSender>,
) -> Result<(), String> {
    // The loop is rebuilt on the loaded request: a stored turn's tool calls
    // and outputs stay in front of the calls this loop executes.
    let mut state = ToolLoopState::new(&current_request);
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    // Generate response ID first so we can use it for both emitter and session
    // (a background response streams under the id its queued record holds)
    let response_id = params
        .response_id
        .clone()
        .unwrap_or_else(|| format!("resp_{}", Uuid::now_v7()));

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

    // Emit initial response.created and response.in_progress events (a
    // background stream reports response.queued between the two)
    let event = emitter.emit_created();
    emitter.send_event(&event, &tx).await?;
    if original_request.background.unwrap_or(false) {
        let event = emitter.emit_queued();
        emitter.send_event(&event, &tx).await?;
    }
    let event = emitter.emit_in_progress();
    emitter.send_event(&event, &tx).await?;

    // Get MCP tools and convert to chat format (do this once before loop)
    let mcp_chat_tools = convert_mcp_tools_to_chat_tools(&session);
    trace!(
        "Streaming: Converted {} MCP tools to chat format",
        mcp_chat_tools.len()
    );

    // Flag to track if mcp_list_tools has been emitted
    let mut mcp_list_tools_emitted = false;

    let terminal_error = loop {
        trace!("Streaming MCP tool loop iteration {}", state.iteration);

        // Emit mcp_list_tools as first output item (only once, on first iteration)
        if !mcp_list_tools_emitted {
            for binding in session.mcp_servers() {
                let tools_for_server = session.list_tools_for_server(&binding.server_key);

                emitter
                    .emit_mcp_list_tools_sequence(&binding.label, &tools_for_server, &tx)
                    .await?;
            }
            mcp_list_tools_emitted = true;
        }

        // Convert to chat request, the MCP tools in its tool list
        let mut chat_request = match conversions::responses_to_chat_with_mcp_tools(
            &current_request,
            &mcp_chat_tools,
        ) {
            Ok(request) => request,
            Err(e) => {
                let message = format!("Failed to convert request: {e}");
                if startup.is_some() {
                    let response = error::bad_request(e.code(), message.clone());
                    let _ = signal_stream_startup(&mut startup, Err(response));
                    return Ok(());
                }
                return Err(message);
            }
        };

        // Prepare tool_choice for this iteration (same logic as non-streaming)
        prepare_tool_choice(&mut chat_request, state.iteration);
        state.iteration += 1;
        Metrics::record_mcp_tool_iteration(&current_request.model);

        // Execute chat streaming, under the iteration's own tenant rate-limit
        // reservation (every iteration is a generation, under a charge id of
        // its own): the chat stream settles it with the engine's counts as
        // it ends, a refused or failed start closes it with its debit kept.
        let startup_pending = startup.is_some();
        let reservation = Arc::new(RateLimitCell::new());
        let response = ctx
            .pipeline
            .execute_chat(
                Arc::new(chat_request),
                params.headers.clone(),
                params.model_id.clone(),
                ctx.components.clone(),
                Some(iteration_charge(&params.tenant_request_meta)),
                Some(reservation.clone()),
                // Tool-loop iterations are not retried.
                None,
                startup.take(),
            )
            .await;

        if !response.status().is_success() {
            reservation.close_unsettled().await;
            if startup_pending {
                return Ok(());
            }
            return Err(format!(
                "Pipeline execution failed with status {}",
                response.status()
            ));
        }

        // Convert chat stream to Responses API events while accumulating for tool call detection
        // Stream text naturally - it only appears on final iteration (tool iterations have empty content)
        let accumulated_response =
            match convert_and_accumulate_stream(response.into_body(), &mut emitter, &tx, |name| {
                session.has_exposed_tool(name)
            })
            .await
            {
                Ok(response) => response,
                Err(error) => break Some(error),
            };
        state.record_usage(accumulated_response.usage.as_ref());
        // Truncated or failed generation cannot request MCP execution.
        if accumulated_response
            .choices
            .first()
            .and_then(|choice| choice.finish_reason.as_deref())
            .is_some_and(|reason| matches!(reason, "length" | "failed" | "error"))
        {
            break None;
        }

        // Check for tool calls (extract all of them for parallel execution)
        let tool_calls = extract_all_tool_calls_from_chat(&accumulated_response);
        // Neither can a server call the generation did not finish.
        if has_unfinished_mcp_call(&tool_calls, &session) {
            break None;
        }

        if !tool_calls.is_empty() {
            let (mcp_tool_calls, function_tool_calls, tool_call_limit) =
                partition_tool_calls(tool_calls, &session, &state, max_tool_calls, true);

            // Process each MCP tool call
            let site = CallSite {
                session: &session,
                format_registry: &ctx.mcp_format_registry,
                original_request,
                model: &current_request.model,
            };
            for tool_call in mcp_tool_calls {
                stream_mcp_call(&site, &mut state, &mut emitter, &tx, tool_call).await?;
            }

            if let Some(limit) = tool_call_limit {
                break match limit {
                    McpToolCallLimit::User => None,
                    McpToolCallLimit::Safety => Some(json!({
                        "code": "max_tool_calls_exceeded",
                        "message": format!("Internal tool call safety limit ({DEFAULT_MAX_ITERATIONS}) reached"),
                    })),
                };
            }

            // process_chunk already emitted these function-call items.
            // Return them to the client without duplicating their events.
            if !function_tool_calls.is_empty() {
                break None;
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
                emitter.emit_reasoning_item(&tx, Some(reasoning)).await?;
            }
        }

        // Text message events already emitted naturally by process_chunk during stream processing
        // (OpenAI router approach - text only appears on final iteration when no tool calls)

        break None;
    };

    let terminal_usage = state.usage.as_ref().map(|usage| json!(usage));
    // The final object, read before the terminal event drains the emitter's
    // items; a background response needs its terminal record under the id
    // the client polls.
    let mut final_response = emitter.finalize(None);
    final_response.usage = state.usage.clone().map(ResponsesUsage::Modern);
    if let Some(error) = terminal_error.clone() {
        // The emitter learns of the failure from the terminal event, which
        // now follows the final object: the stored record says failed itself.
        final_response.status = ResponseStatus::Failed;
        final_response.incomplete_details = None;
        final_response.error = Some(error);
    }
    let tenant = crate::tenant::storage_tenant(&params.tenant_request_meta);
    let persist = persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &final_response,
        original_request,
        tenant.as_deref(),
        ctx.request_context.clone(),
    );
    if stored_foreground(original_request) {
        // Written before its terminal event: a record that could not be
        // written ends the stream with response.failed.
        let not_stored = persist.await.err().map(|e| e.error_object());
        let error = terminal_error.as_ref().or(not_stored.as_ref());
        emitter
            .emit_terminal(terminal_usage.as_ref(), error, &tx)
            .await?;
    } else {
        emitter
            .emit_terminal(terminal_usage.as_ref(), terminal_error.as_ref(), &tx)
            .await?;
        if let Err(e) = persist.await {
            debug!(response_id = %final_response.id, error = %e.message, "Response not stored");
        }
    }
    Ok(())
}

/// Convert chat stream to Responses API events while accumulating for tool call detection
async fn convert_and_accumulate_stream(
    body: Body,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &SseSender,
    is_mcp_tool: impl Fn(&str) -> bool,
) -> Result<ChatCompletionResponse, Value> {
    let mut accumulator = ChatResponseAccumulator::new();
    let mut stream = body.into_data_stream();
    // The parser supplies a complete name before argument deltas. Retain any
    // preceding id-only deltas until that name tells us who executes the call.
    let mut mcp_indices: HashMap<u32, bool> = HashMap::new();
    let mut pending: HashMap<u32, Vec<ToolCallDelta>> = HashMap::new();

    // A frame holds one or more SSE events; every one of them counts.
    'frames: loop {
        let chunk = match next_frame(&mut stream, tx).await {
            BodyRead::Frame(frame) => frame.map_err(|error| stream_read_error(&error))?,
            BodyRead::End => break,
            // The chat body goes with this return, which ends the generation
            // on the engine; the loop ends with the error it meets when the
            // client's leaving is found at the next frame.
            BodyRead::ClientLeft => {
                return Err(json!({"code": "stream_error", "message": "Client disconnected"}))
            }
        };
        let frame = frame_text(&chunk);
        for event in chat_sse_events(&frame) {
            let mut chat_chunk = match event {
                ChatSseEvent::Done => break 'frames,
                ChatSseEvent::Chunk(chunk) => chunk,
                ChatSseEvent::UpstreamError(error) => return Err(error),
                ChatSseEvent::Other(_) => continue,
            };
            // Keep all calls for server execution; only client-owned calls
            // may be exposed as Responses function_call items and events.
            accumulator.process_chunk(&chat_chunk);
            if let Some(choice) = chat_chunk.choices.first_mut() {
                if let Some(calls) = choice.delta.tool_calls.take() {
                    let mut visible = Vec::new();
                    for call in calls {
                        let index = call.index;
                        if let Some(name) = call
                            .function
                            .as_ref()
                            .and_then(|f| f.name.as_deref())
                            .filter(|name| !name.is_empty())
                        {
                            mcp_indices
                                .entry(index)
                                .or_insert_with(|| is_mcp_tool(name));
                        }
                        pending.entry(index).or_default().push(call);
                        if let Some(is_mcp) = mcp_indices.get(&index) {
                            if let Some(deltas) = pending.remove(&index) {
                                if !is_mcp {
                                    visible.extend(deltas);
                                }
                            }
                        }
                    }
                    choice.delta.tool_calls = (!visible.is_empty()).then_some(visible);
                }
            }
            emitter
                .process_chunk(&chat_chunk, tx)
                .await
                .map_err(|message| {
                    json!({
                        "code": "stream_error", "message": message,
                    })
                })?;
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
                    refusal: None,
                    annotations: Vec::new(),
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
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    use super::*;
    use crate::routers::grpc::common::responses::utils::{
        assert_responses_api_usage, namespace_test_request, responses_api_test_usage,
    };

    /// Runs chat SSE frames through the Responses transform as an unstored
    /// request over empty memory storages (no tenant, no request context) and
    /// hands back the client's receiver once the transform has returned.
    async fn transformed_chat_stream(
        frames: Vec<String>,
    ) -> mpsc::Receiver<Result<Bytes, std::io::Error>> {
        let body = Body::from_stream(futures_util::stream::iter(
            frames.into_iter().map(Ok::<_, std::convert::Infallible>),
        ));
        let (tx, rx) = mpsc::channel(64);
        process_and_transform_sse_stream(
            body,
            ResponsesRequest {
                store: Some(false),
                ..Default::default()
            },
            None,
            Arc::new(smg_data_connector::MemoryResponseStorage::new()),
            Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
            None,
            None,
            tx,
        )
        .await
        .unwrap();
        rx
    }

    #[tokio::test]
    async fn terminal_stream_preserves_cache_and_reasoning_usage() {
        for cached in [Some(0), Some(6), None] {
            let mut usage = Usage::from_counts(12, 7).with_reasoning_tokens(2);
            if let Some(cached) = cached {
                usage = usage.with_cached_tokens(cached);
            }
            let chunk = ChatCompletionStreamResponse::builder("chat_test", "test-model")
                .usage(usage)
                .build();
            let frames = vec![
                format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()),
                "data: [DONE]\n\n".into(),
            ];
            let mut rx = transformed_chat_stream(frames).await;
            let mut terminal = None;
            while let Some(Ok(bytes)) = rx.recv().await {
                let text = std::str::from_utf8(&bytes).unwrap();
                let data = text
                    .lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .unwrap();
                let event: Value = serde_json::from_str(data).unwrap();
                if event["type"] == "response.completed" {
                    terminal = Some(event);
                }
            }
            let terminal = terminal.expect("completed event must be emitted");
            let usage = &terminal["response"]["usage"];
            assert_eq!(usage["input_tokens"], 12);
            assert_eq!(usage["output_tokens"], 7);
            assert_eq!(usage["total_tokens"], 19);
            assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 2);
            assert_eq!(
                usage["input_tokens_details"]["cached_tokens"],
                cached.unwrap_or(0)
            );
        }
    }

    #[tokio::test]
    async fn upstream_error_closes_reasoning_and_emits_failed_response() {
        let reasoning = json!({"id":"chat_test","object":"chat.completion.chunk","created":0,"model":"test-model",
            "choices":[{"index":0,"delta":{"reasoning_content":"Partial reasoning"},"finish_reason":null}]});
        let frames = vec![format!("data: {reasoning}\n\n"),
            "data: {\"error\":{\"message\":\"engine unavailable\",\"type\":\"internal_error\"}}\n\n".into(),
            "data: [DONE]\n\n".into()];
        let mut rx = transformed_chat_stream(frames).await;
        let mut events: Vec<Value> = Vec::new();
        while let Some(Ok(bytes)) = rx.recv().await {
            let text = std::str::from_utf8(&bytes).unwrap();
            let data = text
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap();
            events.push(serde_json::from_str(data).unwrap());
        }
        let added = events
            .iter()
            .find(|e| e["type"] == "response.content_part.added")
            .unwrap();
        let done = events
            .iter()
            .find(|e| e["type"] == "response.content_part.done")
            .expect("open reasoning must close even on error");
        assert_eq!(added["item_id"], done["item_id"]);
        assert_eq!(done["part"]["text"], "Partial reasoning");
        let terminal = events.last().unwrap();
        assert_eq!(terminal["type"], "response.failed");
        assert_eq!(terminal["response"]["status"], "failed");
        assert_eq!(
            terminal["response"]["error"]["message"],
            "engine unavailable"
        );
        assert!(!events.iter().any(|e| e["type"] == "response.completed"));
    }

    /// The `data:` payloads the client read, in order, once the transform has
    /// dropped its sender.
    async fn drained_events(rx: &mut mpsc::Receiver<Result<Bytes, std::io::Error>>) -> Vec<Value> {
        let mut events = Vec::new();
        while let Some(Ok(bytes)) = rx.recv().await {
            let text = std::str::from_utf8(&bytes).unwrap();
            events.extend(
                text.lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .map(|data| serde_json::from_str::<Value>(data).unwrap()),
            );
        }
        events
    }

    /// Runs chat SSE frames through the Responses transform as a STORED
    /// request over memory storages (no tenant, no request context) and hands
    /// back the events the client read and the object the store holds under
    /// the id `response.created` announced.
    async fn transformed_and_stored(
        frames: Vec<String>,
        request: ResponsesRequest,
    ) -> (Vec<Value>, Value) {
        let body = Body::from_stream(futures_util::stream::iter(
            frames.into_iter().map(Ok::<_, std::convert::Infallible>),
        ));
        let storage = Arc::new(smg_data_connector::MemoryResponseStorage::new());
        let (tx, mut rx) = mpsc::channel(64);
        process_and_transform_sse_stream(
            body,
            ResponsesRequest {
                store: Some(true),
                ..request
            },
            None,
            storage.clone(),
            Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
            None,
            None,
            tx,
        )
        .await
        .unwrap();
        let events = drained_events(&mut rx).await;
        let id = events[0]["response"]["id"]
            .as_str()
            .expect("response.created names the id");
        let stored = storage
            .get_response(&smg_data_connector::ResponseId::from(id))
            .await
            .unwrap()
            .expect("a stored request's record")
            .raw_response;
        (events, stored)
    }

    /// The item ids of a response object's output, in order.
    fn output_ids(response: &Value) -> Vec<String> {
        response["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["id"].as_str().unwrap().to_string())
            .collect()
    }

    #[tokio::test]
    async fn the_stored_streamed_object_serializes_responses_api_usage() {
        let chunk = ChatCompletionStreamResponse::builder("chat_test", "test-model")
            .usage(responses_api_test_usage())
            .build();
        let frames = vec![
            format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()),
            "data: [DONE]\n\n".into(),
        ];
        let (events, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
        assert_responses_api_usage(&stored);
        // A turn without items: the terminal event and the stored object both
        // read `output: []`; no message is invented for it.
        let terminal = events.last().expect("a terminal event");
        assert_eq!(terminal["response"]["output"], json!([]), "{terminal}");
        assert_eq!(terminal["response"]["output"], stored["output"]);
    }

    /// The stored object of a stream is the one its events announced: the
    /// items in their streamed order, each under the id its
    /// `response.output_item.added` carried (the reasoning item `rs_`, the
    /// message `msg_`, the function call `fc_`), a function call's own id
    /// beside the `call_id` the client echoes, and the terminal event names
    /// the same items.
    #[tokio::test]
    async fn the_stored_streamed_object_carries_the_items_the_events_announced() {
        let frames = vec![
            chat_event(json!({"reasoning_content":"think"}), None),
            chat_event(json!({"content":"Hello"}), None),
            chat_event(
                json!({"tool_calls":[{"index":0,"id":"call_streamed","type":"function",
                    "function":{"name":"lookup","arguments":"{}"}}]}),
                None,
            ),
            chat_event(json!({}), Some("tool_calls")),
            "data: [DONE]\n\n".into(),
        ];
        let (events, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
        let output = stored["output"].as_array().unwrap();
        let kinds: Vec<&str> = output
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, ["reasoning", "message", "function_call"], "{stored}");
        for (item, prefix) in output.iter().zip(["rs_", "msg_", "fc_"]) {
            let id = item["id"].as_str().unwrap();
            assert!(id.starts_with(prefix), "{item}");
            for event_type in ["response.output_item.added", "response.output_item.done"] {
                assert_eq!(
                    events
                        .iter()
                        .filter(|e| e["type"] == event_type && e["item"]["id"] == id)
                        .count(),
                    1,
                    "{event_type} of {id}"
                );
            }
        }
        let call = &output[2];
        assert_eq!(call["call_id"], "call_streamed");
        assert_ne!(call["id"], call["call_id"], "{call}");
        let terminal = events.last().unwrap();
        assert_eq!(terminal["type"], "response.completed");
        assert_eq!(output_ids(&terminal["response"]), output_ids(&stored));
    }

    #[tokio::test]
    async fn namespace_stored_streamed_call_keeps_its_structured_identity() {
        let first = ChatCompletionStreamResponse::builder("chat_test", "test-model")
            .add_choice_tool_name(0, "call_test", "weather.lookup")
            .build();
        let frames = vec![
            format!("data: {}\n\n", serde_json::to_string(&first).unwrap()),
            chat_event(
                json!({"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}),
                Some("tool_calls"),
            ),
            "data: [DONE]\n\n".into(),
        ];
        let (_, stored) = transformed_and_stored(frames, namespace_test_request()).await;
        assert_eq!(stored["output"][0]["name"], "lookup");
        assert_eq!(stored["output"][0]["namespace"], "weather");
        assert_eq!(stored["output"][0]["arguments"], "{}");
        assert_eq!(stored["output"][0]["call_id"], "call_test");
    }

    #[tokio::test]
    async fn stored_streamed_tool_calls_have_terminal_status() {
        for (finish_reason, expected_status) in
            [("tool_calls", "completed"), ("length", "incomplete")]
        {
            let frames = vec![
                chat_event(
                    json!({"tool_calls":[{"index":0,"id":"call_weather","type":"function",
                        "function":{"name":"weather","arguments":"{}"}}]}),
                    Some(finish_reason),
                ),
                "data: [DONE]\n\n".into(),
            ];
            let (_, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
            assert_eq!(stored["status"], expected_status);
            assert_eq!(stored["output"][0]["status"], "completed");
            assert_eq!(stored["output"][0]["call_id"], "call_weather");
            assert_eq!(stored["output"][0]["arguments"], "{}");
            if finish_reason == "length" {
                assert_eq!(stored["incomplete_details"]["reason"], "max_output_tokens");
            }
        }
    }

    #[tokio::test]
    async fn failed_generation_does_not_complete_stored_partial_messages_or_tool_calls() {
        for finish_reason in ["failed", "error"] {
            let frames = vec![
                chat_event(
                    json!({"content":"Partial reply","tool_calls":[{"index":0,"id":"call_weather",
                        "type":"function","function":{"name":"weather","arguments":"{\"city\":"}}]}),
                    None,
                ),
                chat_event(json!({}), Some(finish_reason)),
                "data: [DONE]\n\n".into(),
            ];
            let (_, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
            assert_eq!(stored["status"], "failed", "{finish_reason}");
            assert_eq!(stored["output"].as_array().unwrap().len(), 2);
            assert_eq!(stored["output"][0]["type"], "message");
            assert_eq!(
                stored["output"][0]["status"], "in_progress",
                "{finish_reason}"
            );
            assert_eq!(stored["output"][0]["content"][0]["text"], "Partial reply");
            let item = &stored["output"][1];
            assert_eq!(item["status"], "in_progress", "{finish_reason}");
            assert_eq!(item["call_id"], "call_weather");
            assert_eq!(item["arguments"], "{\"city\":");
        }
    }

    #[tokio::test]
    async fn length_truncated_stored_tool_call_preserves_earlier_completed_calls() {
        let frames = vec![
            chat_event(
                json!({"tool_calls":[
                    {"index":0,"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{}"}},
                    {"index":1,"id":"call_time","type":"function","function":{"name":"time","arguments":"{\"tz\":"}}
                ]}),
                None,
            ),
            chat_event(json!({}), Some("length")),
            "data: [DONE]\n\n".into(),
        ];
        let (_, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
        assert_eq!(stored["status"], "incomplete");
        assert_eq!(stored["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(stored["output"].as_array().unwrap().len(), 2);
        assert_eq!(stored["output"][0]["status"], "completed");
        assert_eq!(stored["output"][0]["arguments"], "{}");
        assert_eq!(stored["output"][1]["status"], "incomplete");
        assert_eq!(stored["output"][1]["call_id"], "call_time");
        assert_eq!(stored["output"][1]["arguments"], "{\"tz\":");
    }

    /// A call cut before its arguments were whole reads `incomplete` in the
    /// stored object whatever ended the stream: an abort, a stop, or a body
    /// that ended with no finish at all (the item is closed when the stream
    /// ends, so the record keeps it).
    #[tokio::test]
    async fn a_stored_streamed_turn_whose_call_was_cut_reads_incomplete() {
        for finish_reason in [Some("abort"), Some("stop"), None] {
            let frames = vec![
                chat_event(
                    json!({"tool_calls":[{"index":0,"id":"call_time","type":"function",
                        "function":{"name":"time","arguments":"{\"tz\":"}}]}),
                    None,
                ),
                chat_event(json!({}), finish_reason),
                "data: [DONE]\n\n".into(),
            ];
            let (_, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
            assert_eq!(stored["status"], "incomplete", "{finish_reason:?}");
            assert!(stored["incomplete_details"].is_null(), "{finish_reason:?}");
            assert_eq!(
                stored["output"][0]["status"], "incomplete",
                "{finish_reason:?}"
            );
            assert_eq!(
                stored["output"][0]["arguments"], "{\"tz\":",
                "{finish_reason:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_stored_streamed_turn_whose_custom_call_was_cut_reads_incomplete_and_keeps_its_item()
    {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "test-model", "input": "hi",
            "tools": [{"type": "custom", "name": "emit_command"}]
        }))
        .unwrap();
        for finish_reason in [Some("abort"), Some("stop"), None] {
            let frames = vec![
                chat_event(
                    json!({"tool_calls":[{"index":0,"id":"call_c","type":"function",
                        "function":{"name":"emit_command","arguments":"{\"input\": \"pw"}}]}),
                    None,
                ),
                chat_event(json!({}), finish_reason),
                "data: [DONE]\n\n".into(),
            ];
            let (_, stored) = transformed_and_stored(frames, request.clone()).await;
            assert_eq!(stored["status"], "incomplete", "{finish_reason:?}");
            assert!(stored["incomplete_details"].is_null(), "{finish_reason:?}");
            assert_eq!(
                stored["output"][0]["type"], "custom_tool_call",
                "{finish_reason:?}"
            );
            assert_eq!(
                stored["output"][0]["call_id"], "call_c",
                "{finish_reason:?}"
            );
        }
    }

    /// A stream that fails after a message and a call were partly streamed
    /// closes both (in progress) before `response.failed`, and the stored
    /// object carries the same items under the same ids with the same error,
    /// whether the failure is the generation's finish or an upstream error
    /// event before any finish.
    #[tokio::test]
    async fn failed_terminal_retains_and_closes_partial_message_and_function() {
        for finish in [None, Some("failed"), Some("error")] {
            let mut frames = vec![chat_event(
                json!({"content":"Partial answer","tool_calls":[{"index":0,"id":"call_test",
                    "type":"function","function":{"name":"lookup","arguments":"{\"city\":"}}]}),
                None,
            )];
            match finish {
                Some(finish) => frames.push(chat_event(json!({}), Some(finish))),
                None => frames.push(
                    "data: {\"error\":{\"code\":\"internal_error\",\"message\":\"engine stopped\"}}\n\n"
                        .into(),
                ),
            }
            frames.push("data: [DONE]\n\n".into());
            let (events, stored) =
                transformed_and_stored(frames, ResponsesRequest::default()).await;
            let terminal = events.last().unwrap();
            assert_eq!(terminal["type"], "response.failed", "{finish:?}");
            assert_eq!(terminal["response"]["output"].as_array().unwrap().len(), 2);
            assert_eq!(stored["status"], "failed", "{finish:?}");
            assert_eq!(stored["error"], terminal["response"]["error"], "{finish:?}");
            if finish.is_some() {
                assert_eq!(stored["error"]["code"], "server_error");
                assert!(stored["error"]["message"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty()));
            } else {
                assert_eq!(stored["error"]["code"], "internal_error");
                assert_eq!(stored["error"]["message"], "engine stopped");
            }
            for kind in ["message", "function_call"] {
                let output = terminal["response"]["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|item| item["type"] == kind)
                    .unwrap();
                let saved = stored["output"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|item| item["type"] == kind)
                    .unwrap();
                assert_eq!(output["status"], "in_progress", "{output}");
                assert_eq!(saved["id"], output["id"]);
                assert_eq!(saved["status"], output["status"]);
                assert_eq!(saved["arguments"], output["arguments"]);
                assert_eq!(saved["content"][0]["text"], output["content"][0]["text"]);
                for event_type in ["response.output_item.added", "response.output_item.done"] {
                    assert_eq!(
                        events
                            .iter()
                            .filter(|e| e["type"] == event_type && e["item"]["id"] == output["id"])
                            .count(),
                        1
                    );
                }
            }
            for kind in [
                "response.output_text.done",
                "response.content_part.done",
                "response.function_call_arguments.done",
            ] {
                assert_eq!(
                    events.iter().filter(|e| e["type"] == kind).count(),
                    1,
                    "{events:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn mcp_stream_preserves_structured_upstream_errors() {
        for (error, expected_code) in [
            (
                json!({"code":"overloaded","type":"server_error","message":"busy"}),
                "overloaded",
            ),
            (
                json!({"type":"internal_error","message":"busy"}),
                "internal_error",
            ),
            (
                json!({"code":null,"type":"internal_error","message":"busy"}),
                "internal_error",
            ),
            (json!({"message":"busy"}), "server_error"),
        ] {
            let frame = format!("data: {}\n\n", json!({"error":error}));
            let body = Body::from(frame);
            let mut emitter =
                ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
            // The client stays (its receiver lives), so the body is read.
            let (tx, _rx) = mpsc::channel(64);
            let error = convert_and_accumulate_stream(body, &mut emitter, &tx, |_| false)
                .await
                .unwrap_err();
            let wire = serde_json::to_value(error).unwrap();
            assert_eq!(wire["code"], expected_code, "{wire}");
            assert_eq!(wire["message"], "busy");
        }
    }

    #[tokio::test]
    async fn mcp_converter_hides_server_calls_but_streams_interleaved_client_calls() {
        let chunks: Vec<ChatCompletionStreamResponse> = [
            (json!({"tool_calls":[{"index":0,"id":"call_server","type":"function"},
                {"index":1,"id":"call_client","type":"function"}]}), None),
            (json!({"tool_calls":[{"index":0,"function":{"name":"server_tool","arguments":"{\"q\":"}},
                {"index":1,"function":{"name":"client_tool","arguments":"{\"q\":"}}]}), None),
            (json!({"tool_calls":[{"index":1,"function":{"arguments":"1}"}},
                {"index":0,"function":{"arguments":"2}"}}]}), None),
            (json!({}), Some("tool_calls")),
        ].into_iter().map(|(delta, finish)| serde_json::from_value(json!({
            "id":"chat_test","object":"chat.completion.chunk","created":0,"model":"test-model",
            "choices":[{"index":0,"delta":delta,"finish_reason":finish}],
        })).unwrap()).collect();
        let frames = chunks.into_iter().map(|chunk| {
            Ok::<_, std::convert::Infallible>(format!(
                "data: {}\n\n",
                serde_json::to_string(&chunk).unwrap()
            ))
        });
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
        let (tx, mut rx) = mpsc::channel(64);
        let response = convert_and_accumulate_stream(
            Body::from_stream(futures_util::stream::iter(frames)),
            &mut emitter,
            &tx,
            |name| name == "server_tool",
        )
        .await
        .unwrap();
        assert_eq!(
            response.choices[0]
                .message
                .tool_calls
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        emitter.emit_terminal(None, None, &tx).await.unwrap();
        drop(tx);
        let mut events: Vec<Value> = Vec::new();
        while let Some(Ok(bytes)) = rx.recv().await {
            let text = std::str::from_utf8(&bytes).unwrap();
            events.push(
                serde_json::from_str(
                    text.lines()
                        .find_map(|line| line.strip_prefix("data: "))
                        .unwrap(),
                )
                .unwrap(),
            );
        }
        let output = events.last().unwrap()["response"]["output"]
            .as_array()
            .unwrap();
        assert_eq!(output.len(), 1, "{events:?}");
        assert_eq!(output[0]["call_id"], "call_client");
        assert_eq!(output[0]["name"], "client_tool");
        assert_eq!(output[0]["arguments"], "{\"q\":1}");
        for kind in [
            "response.output_item.added",
            "response.output_item.done",
            "response.function_call_arguments.done",
        ] {
            assert_eq!(events.iter().filter(|e| e["type"] == kind).count(), 1);
        }
        assert_eq!(
            events
                .iter()
                .filter(|e| e["type"] == "response.function_call_arguments.delta")
                .map(|e| e["delta"].as_str().unwrap())
                .collect::<String>(),
            "{\"q\":1}"
        );
        assert!(!events.iter().any(|e| e["item"]["name"] == "server_tool"));
    }

    #[tokio::test]
    async fn mcp_transport_error_keeps_stream_error_code() {
        let body = Body::from_stream(futures_util::stream::iter([Err::<Bytes, _>(
            std::io::Error::other("connection reset"),
        )]));
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
        // The client stays (its receiver lives), so the body is read.
        let (tx, _rx) = mpsc::channel(64);
        let error = convert_and_accumulate_stream(body, &mut emitter, &tx, |_| false)
            .await
            .unwrap_err();
        assert_eq!(error["code"], "stream_error");
        assert!(error["message"]
            .as_str()
            .unwrap()
            .contains("connection reset"));
    }

    /// A stored foreground stream whose client leaves mid-stream keeps the
    /// promise `response.created` made: its record is written with the items
    /// streamed so far (closed in progress) and status `cancelled`, while the
    /// chat body is dropped with the read, as before; an unstored one records
    /// nothing. The transform answers the departure either way.
    #[tokio::test]
    async fn a_stored_stream_whose_client_leaves_is_recorded_as_cancelled() {
        for store in [true, false] {
            let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
            let body = Body::from_stream(ReceiverStream::new(body_rx));
            let storage = Arc::new(smg_data_connector::MemoryResponseStorage::new());
            let (tx, mut rx) = mpsc::channel(64);
            let transform = process_and_transform_sse_stream(
                body,
                ResponsesRequest {
                    store: Some(store),
                    ..Default::default()
                },
                None,
                storage.clone(),
                Arc::new(smg_data_connector::MemoryConversationStorage::new()),
                Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
                None,
                None,
                tx,
            );
            let client = async move {
                body_tx
                    .send(Ok(Bytes::from(chat_event(
                        json!({"content":"Hello"}),
                        None,
                    ))))
                    .await
                    .unwrap();
                // The client reads up to the text it was streamed, then leaves.
                let mut id = None;
                'read: while let Some(Ok(bytes)) = rx.recv().await {
                    let text = std::str::from_utf8(&bytes).unwrap();
                    for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
                        let event: Value = serde_json::from_str(data).unwrap();
                        if event["type"] == "response.created" {
                            id = event["response"]["id"].as_str().map(str::to_string);
                        }
                        if event["type"] == "response.output_text.delta" {
                            break 'read;
                        }
                    }
                }
                drop(rx);
                (id.expect("response.created names the id"), body_tx)
            };
            // The joined futures are large; boxed, they stay off the test's stack.
            let (outcome, (id, body_tx)) = Box::pin(tokio::time::timeout(
                std::time::Duration::from_secs(5),
                async { tokio::join!(transform, client) },
            ))
            .await
            .expect("the body never ends; the client's leaving ends the read");
            assert_eq!(outcome.unwrap_err(), "Client disconnected");
            assert!(
                body_tx.is_closed(),
                "the chat body is dropped with the read"
            );
            let record = storage
                .get_response(&smg_data_connector::ResponseId::from(id.as_str()))
                .await
                .unwrap();
            if !store {
                assert!(record.is_none(), "an unstored response records nothing");
                continue;
            }
            let stored = record.expect("the stored stream's record").raw_response;
            assert_eq!(stored["status"], "cancelled", "{stored}");
            assert!(stored["error"].is_null(), "{stored}");
            assert!(stored["incomplete_details"].is_null(), "{stored}");
            assert_eq!(stored["output"][0]["type"], "message", "{stored}");
            assert_eq!(stored["output"][0]["status"], "in_progress", "{stored}");
            assert_eq!(stored["output"][0]["content"][0]["text"], "Hello");
            assert!(
                stored["output"][0]["id"]
                    .as_str()
                    .is_some_and(|id| id.starts_with("msg_")),
                "{stored}"
            );
        }
    }

    /// Reads the client's SSE frames up to the first event of `event_type`
    /// and returns the id `response.created` announced.
    async fn read_until<E>(rx: &mut mpsc::Receiver<Result<Bytes, E>>, event_type: &str) -> String {
        let mut id = None;
        'read: while let Some(Ok(bytes)) = rx.recv().await {
            let text = std::str::from_utf8(&bytes).unwrap();
            for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
                let event: Value = serde_json::from_str(data).unwrap();
                if event["type"] == "response.created" {
                    id = event["response"]["id"].as_str().map(str::to_string);
                }
                if event["type"] == event_type {
                    break 'read;
                }
            }
        }
        id.expect("response.created names the id")
    }

    /// The stored object of the response `id`.
    async fn stored_record(storage: &smg_data_connector::MemoryResponseStorage, id: &str) -> Value {
        storage
            .get_response(&smg_data_connector::ResponseId::from(id))
            .await
            .unwrap()
            .expect("the stored stream's record")
            .raw_response
    }

    /// The client leaves while the message's closing events are being sent
    /// (an SSE channel of one slot: the client reads up to the text delta and
    /// leaves, so the first closing event's send meets the dead client). The
    /// item was stored before that send, and the finish had been read: the
    /// record keeps the message, completed, under the stream's own finish.
    #[tokio::test]
    async fn a_client_that_leaves_during_the_closing_events_keeps_the_streamed_message() {
        let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        let body = Body::from_stream(ReceiverStream::new(body_rx));
        let storage = Arc::new(smg_data_connector::MemoryResponseStorage::new());
        let (tx, mut rx) = mpsc::channel(1);
        let transform = process_and_transform_sse_stream(
            body,
            ResponsesRequest {
                store: Some(true),
                ..Default::default()
            },
            None,
            storage.clone(),
            Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
            None,
            None,
            tx,
        );
        let client = async move {
            for frame in [
                chat_event(json!({"content":"Hello"}), None),
                chat_event(json!({}), Some("stop")),
            ] {
                body_tx.send(Ok(Bytes::from(frame))).await.unwrap();
            }
            let id = read_until(&mut rx, "response.output_text.delta").await;
            drop(rx);
            (id, body_tx)
        };
        let (outcome, (id, _body_tx)) = Box::pin(tokio::time::timeout(
            std::time::Duration::from_secs(5),
            async { tokio::join!(transform, client) },
        ))
        .await
        .expect("the client's leaving ends the read");
        assert_eq!(outcome.unwrap_err(), "Client disconnected");
        let stored = stored_record(&storage, &id).await;
        assert_eq!(stored["status"], "completed", "{stored}");
        assert_eq!(stored["output"][0]["type"], "message", "{stored}");
        assert_eq!(stored["output"][0]["status"], "completed", "{stored}");
        assert_eq!(
            stored["output"][0]["content"][0]["text"], "Hello",
            "{stored}"
        );
    }

    /// A finish the stream had already read stands when the client leaves
    /// before the body ends: a `length` finish is recorded `incomplete` with
    /// its `incomplete_details`, not `cancelled`, and the message it closed
    /// keeps its text.
    #[tokio::test]
    async fn a_client_that_leaves_after_the_finish_keeps_the_finish() {
        let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        let body = Body::from_stream(ReceiverStream::new(body_rx));
        let storage = Arc::new(smg_data_connector::MemoryResponseStorage::new());
        let (tx, mut rx) = mpsc::channel(64);
        let transform = process_and_transform_sse_stream(
            body,
            ResponsesRequest {
                store: Some(true),
                ..Default::default()
            },
            None,
            storage.clone(),
            Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
            None,
            None,
            tx,
        );
        let client = async move {
            for frame in [
                chat_event(json!({"content":"Hello"}), None),
                chat_event(json!({}), Some("length")),
            ] {
                body_tx.send(Ok(Bytes::from(frame))).await.unwrap();
            }
            // The message closed under the finish; the client leaves before
            // the body ends (no usage chunk, no `[DONE]` yet).
            let id = read_until(&mut rx, "response.output_item.done").await;
            drop(rx);
            (id, body_tx)
        };
        let (outcome, (id, _body_tx)) = Box::pin(tokio::time::timeout(
            std::time::Duration::from_secs(5),
            async { tokio::join!(transform, client) },
        ))
        .await
        .expect("the client's leaving ends the read");
        assert_eq!(outcome.unwrap_err(), "Client disconnected");
        let stored = stored_record(&storage, &id).await;
        assert_eq!(stored["status"], "incomplete", "{stored}");
        assert_eq!(
            stored["incomplete_details"]["reason"], "max_output_tokens",
            "{stored}"
        );
        assert_eq!(
            stored["output"][0]["content"][0]["text"], "Hello",
            "{stored}"
        );
        assert_ne!(stored["output"][0]["status"], "in_progress", "{stored}");
    }

    /// Under `cancelled` every item still open when the client left reads
    /// `in_progress`: a reasoning item, a function call with partial
    /// arguments, as the message does.
    #[tokio::test]
    async fn a_cancelled_record_closes_the_open_reasoning_and_call_items_in_progress() {
        for (delta, leave_at, item_type) in [
            (
                json!({"reasoning_content":"think"}),
                "response.reasoning_text.delta",
                "reasoning",
            ),
            (
                json!({"tool_calls":[{"index":0,"id":"call_open","type":"function",
                    "function":{"name":"lookup","arguments":"{\"q\":"}}]}),
                "response.function_call_arguments.delta",
                "function_call",
            ),
        ] {
            let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
            let body = Body::from_stream(ReceiverStream::new(body_rx));
            let storage = Arc::new(smg_data_connector::MemoryResponseStorage::new());
            let (tx, mut rx) = mpsc::channel(64);
            let transform = process_and_transform_sse_stream(
                body,
                ResponsesRequest {
                    store: Some(true),
                    ..Default::default()
                },
                None,
                storage.clone(),
                Arc::new(smg_data_connector::MemoryConversationStorage::new()),
                Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
                None,
                None,
                tx,
            );
            let client = async move {
                body_tx
                    .send(Ok(Bytes::from(chat_event(delta, None))))
                    .await
                    .unwrap();
                let id = read_until(&mut rx, leave_at).await;
                drop(rx);
                (id, body_tx)
            };
            let (outcome, (id, _body_tx)) = Box::pin(tokio::time::timeout(
                std::time::Duration::from_secs(5),
                async { tokio::join!(transform, client) },
            ))
            .await
            .expect("the client's leaving ends the read");
            assert_eq!(outcome.unwrap_err(), "Client disconnected");
            let stored = stored_record(&storage, &id).await;
            assert_eq!(stored["status"], "cancelled", "{item_type}: {stored}");
            assert_eq!(stored["output"][0]["type"], item_type, "{stored}");
            assert_eq!(stored["output"][0]["status"], "in_progress", "{stored}");
        }
    }

    /// The reader parked on a chat body that stalled (nothing more on it)
    /// ends the moment its client leaves, with the error it ends with when
    /// it meets the dead client at the next frame, and the chat body goes
    /// with it: its sender sees the drop, as the chat layer does.
    #[tokio::test]
    async fn a_client_that_leaves_while_the_chat_body_is_stalled_ends_the_read_at_once() {
        let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        let body = Body::from_stream(ReceiverStream::new(body_rx));
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
        let (tx, rx) = mpsc::channel(64);
        let read = convert_and_accumulate_stream(body, &mut emitter, &tx, |_| false);
        let leave = async move {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            drop(rx);
        };
        let (outcome, ()) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            tokio::join!(read, leave)
        })
        .await
        .expect("the body never yields; the client's leaving ends the read");
        let error = outcome.unwrap_err();
        assert_eq!(error["code"], "stream_error");
        assert_eq!(error["message"], "Client disconnected");
        assert!(
            body_tx.is_closed(),
            "the chat body is dropped with the read"
        );
    }

    /// One reader for the chat stream's events: chunks, the error event as
    /// the terminal's error object (`code`, else `type`, else `server_error`;
    /// `message`, else the generic wording), the terminator, and every other
    /// data event left to the caller; lines that are not data events are
    /// nobody's.
    #[test]
    fn the_chat_sse_reader_sorts_a_frames_events() {
        let frame = concat!(
            ": keep-alive\n\n",
            "data: {\"id\":\"chat_test\",\"object\":\"chat.completion.chunk\",\"created\":0,\
             \"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\
             \"finish_reason\":null}]}\n\n",
            "data: {\"unexpected\":true}\n\n",
            "data: {\"error\":{\"type\":\"internal_error\",\"message\":\"engine unavailable\"}}\n\n",
            "data: {\"error\":{\"code\":\"rate_limited\"}}\n\n",
            "data: [DONE]\n\n",
            "data: {\"after\":\"done\"}\n\n",
        );
        let read: Vec<String> = chat_sse_events(frame)
            .map(|event| match event {
                ChatSseEvent::Chunk(chunk) => format!(
                    "chunk:{}",
                    chunk.choices[0].delta.content.as_deref().unwrap_or("")
                ),
                ChatSseEvent::UpstreamError(error) => format!("error:{error}"),
                ChatSseEvent::Done => "done".to_string(),
                ChatSseEvent::Other(event) => format!("other:{event}"),
            })
            .collect();
        assert_eq!(
            read,
            [
                "chunk:hi",
                "other:data: {\"unexpected\":true}",
                "error:{\"code\":\"internal_error\",\"message\":\"engine unavailable\"}",
                "error:{\"code\":\"rate_limited\",\"message\":\"Upstream generation failed\"}",
                "done",
                "other:data: {\"after\":\"done\"}",
            ]
        );
    }

    /// One chat chunk as the SSE event the chat stream writes for it.
    fn chat_event(delta: Value, finish_reason: Option<&str>) -> String {
        format!(
            "data: {}\n\n",
            json!({
                "id":"chat_test","object":"chat.completion.chunk","created":0,"model":"test-model",
                "choices":[{"index":0,"delta":delta,"finish_reason":finish_reason}]
            })
        )
    }

    /// The chat stream hands the body every chunk the engine had ready as one
    /// frame. Each event of such a frame is converted, and a `[DONE]` that
    /// shares its frame with the finish chunk still ends the stream.
    #[tokio::test]
    async fn drained_frame_feeds_every_chat_chunk_to_the_responses_stream() {
        let frames = vec![
            [
                chat_event(json!({"role":"assistant","content":""}), None),
                chat_event(json!({"content":"Hel"}), None),
                chat_event(json!({"content":"lo"}), None),
            ]
            .concat(),
            [
                chat_event(json!({}), Some("stop")),
                "data: [DONE]\n\n".to_string(),
            ]
            .concat(),
        ];
        let mut rx = transformed_chat_stream(frames).await;
        let mut events: Vec<Value> = Vec::new();
        while let Some(Ok(bytes)) = rx.recv().await {
            let text = std::str::from_utf8(&bytes).unwrap();
            for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
                // Only the chat stream's `[DONE]` is not JSON; keep it visible.
                events.push(serde_json::from_str(data).unwrap_or(Value::String(data.to_string())));
            }
        }
        assert!(
            events.iter().all(|e| e["sequence_number"].is_u64()),
            "a chat frame reached the client unconverted: {events:?}"
        );
        let text: String = events
            .iter()
            .filter(|e| e["type"] == "response.output_text.delta")
            .map(|e| e["delta"].as_str().unwrap())
            .collect();
        assert_eq!(text, "Hello");
        let terminal = events.last().unwrap();
        assert_eq!(terminal["type"], "response.completed");
        assert_eq!(
            terminal["response"]["output"][0]["content"][0]["text"],
            "Hello"
        );
    }

    /// The MCP converter reads every event of a frame as well: a server call
    /// whose name, arguments and finish arrive in one frame is accumulated
    /// for execution.
    #[tokio::test]
    async fn drained_frame_feeds_every_chat_chunk_to_the_mcp_converter() {
        let frame = [
            chat_event(
                json!({"tool_calls":[{"index":0,"id":"call_server","type":"function",
                    "function":{"name":"server_tool","arguments":""}}]}),
                None,
            ),
            chat_event(
                json!({"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":1}"}}]}),
                None,
            ),
            chat_event(json!({}), Some("tool_calls")),
            "data: [DONE]\n\n".to_string(),
        ]
        .concat();
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
        let (tx, _rx) = mpsc::channel(64);
        let response =
            convert_and_accumulate_stream(Body::from(frame), &mut emitter, &tx, |name| {
                name == "server_tool"
            })
            .await
            .unwrap();
        let choice = &response.choices[0];
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        let calls = choice
            .message
            .tool_calls
            .as_ref()
            .expect("the server call of the frame is accumulated");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_server");
        assert_eq!(calls[0].function.name, "server_tool");
        assert_eq!(calls[0].function.arguments.as_deref(), Some("{\"q\":1}"));
    }

    /// An upstream error that shares a frame with the chunks before it fails
    /// the response instead of completing it.
    #[tokio::test]
    async fn drained_frame_error_fails_the_mcp_stream() {
        let frame = [
            chat_event(json!({"role":"assistant","content":"Part"}), None),
            "data: {\"error\":{\"message\":\"engine unavailable\",\"type\":\"internal_error\"}}\n\n"
                .to_string(),
        ]
        .concat();
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
        let (tx, _rx) = mpsc::channel(64);
        let error = convert_and_accumulate_stream(Body::from(frame), &mut emitter, &tx, |_| false)
            .await
            .unwrap_err();
        assert_eq!(error["code"], "internal_error");
        assert_eq!(error["message"], "engine unavailable");
    }

    /// A chat body whose read fails mid-stream ends the response as failed,
    /// not completed: the text before the cut reaches the client, the
    /// terminal event is `response.failed` with the read error, and the
    /// stored record reads failed and carries it.
    #[tokio::test]
    async fn a_body_read_error_fails_the_responses_stream_and_its_record() {
        let frames: Vec<Result<String, std::io::Error>> = vec![
            Ok(chat_event(
                json!({"role":"assistant","content":"Part"}),
                None,
            )),
            Err(std::io::Error::other("the engine reset the connection")),
        ];
        let body = Body::from_stream(futures_util::stream::iter(frames));
        let response_storage = Arc::new(smg_data_connector::MemoryResponseStorage::new());
        let (tx, mut rx) = mpsc::channel(64);
        process_and_transform_sse_stream(
            body,
            ResponsesRequest {
                store: Some(true),
                ..Default::default()
            },
            None,
            response_storage.clone(),
            Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
            None,
            None,
            tx,
        )
        .await
        .unwrap();
        let mut events: Vec<Value> = Vec::new();
        while let Some(Ok(bytes)) = rx.recv().await {
            let text = std::str::from_utf8(&bytes).unwrap();
            for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
                events.push(serde_json::from_str(data).unwrap());
            }
        }
        let text: String = events
            .iter()
            .filter(|e| e["type"] == "response.output_text.delta")
            .map(|e| e["delta"].as_str().unwrap())
            .collect();
        assert_eq!(text, "Part", "the text before the cut reaches the client");
        let terminal = events.last().expect("a terminal event");
        assert_eq!(terminal["type"], "response.failed", "{terminal}");
        assert_eq!(terminal["response"]["status"], "failed");
        assert_eq!(terminal["response"]["error"]["code"], "stream_error");
        let message = terminal["response"]["error"]["message"]
            .as_str()
            .expect("the error's message");
        assert!(
            message.starts_with("Stream read error: ")
                && message.contains("the engine reset the connection"),
            "{message}"
        );
        assert!(!events.iter().any(|e| e["type"] == "response.completed"));
        let id = terminal["response"]["id"]
            .as_str()
            .expect("the response id");
        let stored = response_storage
            .get_response(&smg_data_connector::ResponseId::from(id))
            .await
            .unwrap()
            .expect("the failed response is stored");
        assert_eq!(stored.raw_response["status"], "failed");
        assert_eq!(
            stored.raw_response["error"], terminal["response"]["error"],
            "the record carries the error the client saw"
        );
    }

    #[test]
    fn the_events_of_a_frame_are_split_at_the_blank_line_as_before() {
        let frame = "data: {\"a\":1}\n\ndata: {\"b\":2}\n\n\n\n: keep-alive\n\ndata: [DONE]\n\n";
        let events: Vec<&str> = sse_events(frame).collect();
        let split: Vec<&str> = frame
            .split("\n\n")
            .map(str::trim)
            .filter(|event| !event.is_empty())
            .collect();
        assert_eq!(events, split);
        assert_eq!(
            events,
            [
                "data: {\"a\":1}",
                "data: {\"b\":2}",
                ": keep-alive",
                "data: [DONE]"
            ]
        );
        assert_eq!(sse_events("").count(), 0);
        assert_eq!(sse_events("\n\n").count(), 0);
        assert_eq!(sse_events("data: x").collect::<Vec<_>>(), ["data: x"]);
    }

    #[test]
    fn a_frame_of_valid_bytes_is_borrowed_and_an_invalid_one_is_replaced() {
        let valid = b"data: {\"id\":\"chatcmpl-1\"}\n\n";
        let frame = frame_text(valid);
        assert!(matches!(frame, Cow::Borrowed(_)));
        assert_eq!(frame.as_ref(), std::str::from_utf8(valid).unwrap());

        let invalid = b"data: \xff\n\n";
        let frame = frame_text(invalid);
        assert!(matches!(frame, Cow::Owned(_)));
        assert_eq!(frame.as_ref(), "data: \u{FFFD}\n\n");
    }
}
