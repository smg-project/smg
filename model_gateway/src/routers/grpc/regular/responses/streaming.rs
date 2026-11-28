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
    marker::PhantomData,
    ops::Deref,
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
    event_types::McpEvent,
    responses::{ResponseTool, ResponseUsage, ResponsesRequest},
};
use serde::{
    de::{Deserializer, IgnoredAny, SeqAccess, Visitor},
    Deserialize,
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
    middleware::RequestStarted,
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
                        attach_mcp_server_label, ChatChunkParts, ResponseStreamEventEmitter,
                        StreamStartupSender, ToolCallParts,
                    },
                    BackgroundStream, ResponsesContext,
                },
                stages::{iteration_charge, RateLimitCell},
            },
            proto_wrapper::ClientGone,
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
    effective_tools: Option<Vec<ResponseTool>>,
    params: ResponsesCallContext,
    original_request: &Arc<ResponsesRequest>,
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

    // Spawn background task to transform stream; it takes a share of the
    // request (a count), not a copy of it.
    let original_request_clone = Arc::clone(original_request);
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
            effective_tools,
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
/// read it: the chat chunk it carries (read as `C`: the owned chunk where the
/// reader keeps it, [`ChatChunkView`] where it only reads it), the chat
/// stream's error event as the error object a Responses terminal carries, the
/// `[DONE]` terminator, or any other data event, left to the reader. Lines
/// that are not data events (a comment, a keep-alive) are nobody's and are
/// not yielded.
enum ChatSseEvent<'a, C> {
    Chunk(C),
    UpstreamError(Value),
    Done,
    Other(&'a str),
}

/// The data events of one frame of the chat stream's body, in order; the
/// reader stops at [`ChatSseEvent::Done`].
fn chat_sse_events<'a, C: Deserialize<'a>>(
    frame: &'a str,
) -> impl Iterator<Item = ChatSseEvent<'a, C>> {
    sse_events(frame).filter_map(|event| {
        if event == "data: [DONE]" {
            return Some(ChatSseEvent::Done);
        }
        let data = event.strip_prefix("data: ")?.trim();
        if let Ok(chunk) = serde_json::from_str::<C>(data) {
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

/// A JSON string of a [`ChatChunkView`]: borrowed from the event's text where
/// it holds no escape, copied where it holds one.
#[derive(Deserialize)]
struct JsonText<'a>(#[serde(borrow)] Cow<'a, str>);

impl Deref for JsonText<'_> {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

/// The first element of a JSON array, the others read past: a chat chunk's
/// choices, of which the Responses events read the first.
struct FirstElement<T>(Option<T>);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for FirstElement<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FirstVisitor<T>(PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for FirstVisitor<T> {
            type Value = FirstElement<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an array")
            }

            fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<FirstElement<T>, S::Error> {
                let first = seq.next_element()?;
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(FirstElement(first))
            }
        }

        deserializer.deserialize_seq(FirstVisitor(PhantomData))
    }
}

/// One chat chunk as the stream transform reads it, borrowed from its data
/// event: the first choice's delta texts, tool-call deltas and finish
/// reason, and the usage. The keys the owned chunk requires are required
/// here with their types, so the chat stream's error event, or any other
/// data event, fails this read as it fails the owned one and takes its own
/// branch; every other key is read past.
#[derive(Deserialize)]
struct ChatChunkView<'a> {
    #[serde(borrow, rename = "id")]
    _id: JsonText<'a>,
    #[serde(borrow, rename = "object")]
    _object: JsonText<'a>,
    #[serde(rename = "created")]
    _created: u64,
    #[serde(borrow, rename = "model")]
    _model: JsonText<'a>,
    #[serde(borrow)]
    choices: FirstElement<ChoiceView<'a>>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct ChoiceView<'a> {
    #[serde(rename = "index")]
    _index: u32,
    #[serde(borrow)]
    delta: DeltaView<'a>,
    #[serde(borrow)]
    finish_reason: Option<JsonText<'a>>,
}

#[derive(Deserialize)]
struct DeltaView<'a> {
    #[serde(borrow)]
    content: Option<JsonText<'a>>,
    #[serde(borrow)]
    reasoning_content: Option<JsonText<'a>>,
    #[serde(borrow)]
    tool_calls: Option<Vec<ToolCallView<'a>>>,
}

#[derive(Deserialize)]
struct ToolCallView<'a> {
    index: u32,
    #[serde(borrow)]
    id: Option<JsonText<'a>>,
    #[serde(borrow)]
    function: Option<FunctionView<'a>>,
}

#[derive(Deserialize)]
struct FunctionView<'a> {
    #[serde(borrow)]
    name: Option<JsonText<'a>>,
    #[serde(borrow)]
    arguments: Option<JsonText<'a>>,
}

impl ChatChunkParts for ChatChunkView<'_> {
    fn reasoning(&self) -> Option<&str> {
        self.choices.0.as_ref()?.delta.reasoning_content.as_deref()
    }

    fn content(&self) -> Option<&str> {
        self.choices.0.as_ref()?.delta.content.as_deref()
    }

    fn tool_call(&self, at: usize) -> Option<ToolCallParts<'_>> {
        let call = self
            .choices
            .0
            .as_ref()?
            .delta
            .tool_calls
            .as_ref()?
            .get(at)?;
        let function = call.function.as_ref();
        Some(ToolCallParts {
            index: call.index,
            id: call.id.as_deref(),
            name: function.and_then(|f| f.name.as_deref()),
            arguments: function.and_then(|f| f.arguments.as_deref()),
        })
    }

    fn finish_reason(&self) -> Option<&str> {
        self.choices.0.as_ref()?.finish_reason.as_deref()
    }
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

/// The next frame of the chat body `stream`, read for the client
/// `client_gone` waits for. The relay that turns the chat stream
/// into the Responses stream holds the chat layer's receiver through that
/// body: parked on a body that stalled mid-stream, it would learn that its
/// client left only when the body next yields, and the engine request behind
/// the body would run on as long. Here every read races the body against the
/// client's leaving; on [`BodyRead::ClientLeft`] the caller ends its read and
/// drops the body, and the chat layer ends the engine request the way it does
/// for its own clients. A background response's wait is on the sender of its
/// relay, which closes with the work and not with the client: its reads are
/// the body's own, and the work outlives the client as it is meant to.
/// `client_gone` is the wait the relay keeps for one chat body; the order of a
/// read is [`ClientGone::poll_read`]'s.
async fn next_frame(stream: &mut BodyDataStream, client_gone: &mut ClientGone) -> BodyRead {
    std::future::poll_fn(|cx| client_gone.poll_read(cx, |cx| stream.poll_next_unpin(cx)))
        .await
        .map_or(BodyRead::ClientLeft, |frame| {
            frame.map_or(BodyRead::End, BodyRead::Frame)
        })
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
    usage: Option<ResponseUsage>,
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
    original_request: Arc<ResponsesRequest>,
    effective_tools: Option<Vec<ResponseTool>>,
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
    event_emitter.set_original_request(ResponsesRequest::clone(&original_request));
    event_emitter.set_effective_tools(effective_tools);

    // Emit initial response.created and response.in_progress events (a
    // background stream reports response.queued between the two)
    event_emitter
        .send_created(&tx)
        .await
        .map_err(|_| "Failed to send response.created event".to_string())?;

    if original_request.background.unwrap_or(false) {
        let event = event_emitter.emit_queued();
        event_emitter
            .send_event(&event, &tx)
            .await
            .map_err(|_| "Failed to send response.queued event".to_string())?;
    }

    event_emitter
        .send_in_progress(&tx)
        .await
        .map_err(|_| "Failed to send response.in_progress event".to_string())?;

    // Convert body to data stream
    let mut stream = body.into_data_stream();
    let mut client_gone = ClientGone::of(&tx);

    let mut terminal_error = None;
    // The client left: met by the body read, by an event it no longer takes,
    // or by the emitter's send. The loop ends there.
    let mut client_left = false;
    // Process the stream frame by frame; a frame holds one or more SSE events.
    'frames: loop {
        let chunk = match next_frame(&mut stream, &mut client_gone).await {
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
        for event in chat_sse_events::<ChatChunkView<'_>>(&frame) {
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
            usage.as_ref().map(ResponseUsage::from),
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
    let usage = usage.as_ref().map(ResponseUsage::from);
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
        event_emitter.emit_terminal(usage_json, error, &tx).await?;
    } else {
        // A background response's terminal write is retried behind the event
        // (its client polls the record); an unstored response stores
        // nothing. Neither fails the stream.
        event_emitter
            .emit_terminal(usage_json, terminal_error.as_ref(), &tx)
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
    original_request: &Arc<ResponsesRequest>,
    params: ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
    background: Option<BackgroundStream>,
) -> Response {
    // The request's stamp (its entry and its body's upload wait), read
    // here in the request's task: the loop runs in a task of its own, where
    // the stamp's scope is gone, and dispatches its first generation under
    // this stamp again.
    let entered = RequestStarted::carried();

    // Create SSE channel for client
    let (tx, rx) = sse_channel();
    let (startup_tx, startup_rx) = stream_startup_channel();

    // Clone data for background task; the request is shared by count, not copied.
    let ctx_clone = ctx.clone();
    let original_request_clone = Arc::clone(original_request);
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
            entered,
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
    /// The loop's current request: its model labels the call's metrics, its
    /// tools name the call's namespace.
    request: &'a ResponsesRequest,
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
    emitter
        .send_output_item_added(output_index, &item, tx)
        .await?;

    // Emit tool_call.in_progress
    let event = emitter.emit_tool_call_in_progress(output_index, &item_id, response_format);
    emitter.send_event(&event, tx).await?;

    // Emit arguments events for mcp_call only (skip for builtin tools)
    if matches!(response_format, ResponseFormat::Passthrough) {
        // Emit mcp_call_arguments.delta (simulate streaming by sending full arguments)
        emitter
            .send_arguments_delta(
                McpEvent::CALL_ARGUMENTS_DELTA,
                &tool_call.arguments,
                output_index,
                &item_id,
                tx,
            )
            .await?;

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
    let arguments = dispatch_arguments(response_format, site.request, &tool_call.arguments);

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
    // The call executed: its item is stored and marked complete, and its
    // metrics and place in the loop's history recorded, before the sends
    // that follow, so a client who left during the call changes neither the
    // record nor the counts; `output_item.done` is still sent after the
    // completion event, in order.
    emitter.keep_output_item(output_index, item_done);
    emitter.complete_output_item(output_index);
    let failure = (!success).then(|| {
        tool_output
            .error_message
            .clone()
            .unwrap_or_else(|| output_str.clone())
    });
    record_mcp_call(state, site.request, tool_output, output_str, output_item);

    if let Some(err_text) = failure {
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
    } else {
        let event = emitter.emit_tool_call_completed(output_index, &item_id, response_format);
        emitter.send_event(&event, tx).await?;
    }

    emitter.send_kept_output_item_done(output_index, tx).await?;
    Ok(())
}

/// How the tool loop's turns ended: done (the terminal event follows), failed
/// with the terminal error object, or left by the client (the departure is
/// recorded, no event follows).
enum Turns {
    Done,
    Failed(Value),
    /// The client left: at the loop's own sends between generations (the
    /// tool list, a call's events, the reasoning item), or inside one.
    ClientLeft {
        between: bool,
    },
}

impl Turns {
    /// A failure met at the loop's own sends while the client is gone is the
    /// client's leaving; any other one answers as before (the loop's error,
    /// nothing recorded).
    fn left_or(tx: &SseSender, message: String) -> Result<Self, String> {
        if tx.is_closed() {
            Ok(Self::ClientLeft { between: true })
        } else {
            Err(message)
        }
    }
}

/// Internal streaming tool loop implementation
///
/// `entered` is the request's entry stamp, when it came in under one. Every
/// turn's generation is dispatched under a stamp, so its dispatch wait is
/// recorded (see `context::dispatch_wait`): the first turn's under
/// `entered`, a later turn's under one taken where that turn starts.
#[expect(
    clippy::too_many_arguments,
    reason = "the streamed loop needs the request state, the channel, the startup barrier and the entry stamp"
)]
async fn execute_tool_loop_streaming_internal(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    params: ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
    tx: SseSender,
    mut startup: Option<StreamStartupSender>,
    entered: Option<RequestStarted>,
) -> Result<(), String> {
    // The loop is rebuilt on the loaded request: a stored turn's tool calls
    // and outputs stay in front of the calls this loop executes.
    let mut state = ToolLoopState::default();
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    // Generate response ID first so we can use it for both emitter and session
    // (a background response streams under the id its queued record holds)
    let response_id = params
        .response_id
        .clone()
        .unwrap_or_else(|| format!("resp_{}", Uuid::now_v7()));

    // Create session once — bundles orchestrator, request_ctx, server_keys, mcp_tools
    let mut session = McpToolSession::new(&ctx.mcp_orchestrator, mcp_servers, &response_id);
    // The loaded request's tools, the list the servers were connected from: it
    // also holds an `mcp` entry that an input item declared.
    if let Some(tools) = current_request.tools.as_deref() {
        openai_bridge::configure_response_tools_approval(&mut session, tools);
    }

    // Create response event emitter
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut emitter =
        ResponseStreamEventEmitter::new(response_id, current_request.model.clone(), created_at);
    emitter.set_original_request(original_request.clone());
    emitter.set_effective_tools(current_request.tools.clone());

    // Emit initial response.created and response.in_progress events (a
    // background stream reports response.queued between the two)
    emitter.send_created(&tx).await?;
    if original_request.background.unwrap_or(false) {
        let event = emitter.emit_queued();
        emitter.send_event(&event, &tx).await?;
    }
    emitter.send_in_progress(&tx).await?;

    // Get MCP tools and convert to chat format (do this once before loop)
    let mcp_chat_tools = convert_mcp_tools_to_chat_tools(&session);
    trace!(
        "Streaming: Converted {} MCP tools to chat format",
        mcp_chat_tools.len()
    );

    // Flag to track if mcp_list_tools has been emitted
    let mut mcp_list_tools_emitted = false;

    // The stamp the turn under way dispatches its generation under: the
    // request's own for the first turn. A request that came in outside a
    // stamp has none on any turn.
    let mut turn_entered = entered;

    let turns = 'turns: loop {
        trace!("Streaming MCP tool loop iteration {}", state.iteration);

        // Emit mcp_list_tools as first output item (only once, on first iteration)
        if !mcp_list_tools_emitted {
            for binding in session.mcp_servers() {
                let tools_for_server = session.list_tools_for_binding(binding);
                let list_error = binding.list_error.as_deref();

                if let Err(message) = emitter
                    .emit_mcp_list_tools_sequence(
                        &binding.label,
                        &tools_for_server,
                        list_error,
                        &tx,
                    )
                    .await
                {
                    break 'turns Turns::left_or(&tx, message)?;
                }
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
                let message = e.to_string();
                if startup.is_some() {
                    let response = error::bad_request(e.code(), message.clone());
                    let _ = signal_stream_startup(&mut startup, Err(response));
                    return Ok(());
                }
                return Err(message);
            }
        };

        // Prepare tool_choice for this iteration (same logic as non-streaming)
        prepare_tool_choice(&mut chat_request, &state, |name| {
            session.has_exposed_tool(name)
        });
        state.iteration += 1;
        Metrics::record_mcp_tool_iteration(&current_request.model);

        // Execute chat streaming, under the iteration's own tenant rate-limit
        // reservation (every iteration is a generation, under a charge id of
        // its own): the chat stream settles it with the engine's counts as
        // it ends, a refused or failed start closes it with its debit kept.
        let startup_pending = startup.is_some();
        let reservation = Arc::new(RateLimitCell::new());
        let generation = ctx.pipeline.execute_chat(
            Arc::new(chat_request),
            params.headers.clone(),
            params.model_id.clone(),
            ctx.components.clone(),
            Some(iteration_charge(&params.tenant_request_meta)),
            Some(reservation.clone()),
            // Tool-loop iterations are not retried.
            None,
            startup.take(),
        );
        // Dispatched under the turn's stamp: the chat stream reads its
        // dispatch wait from it where it is built.
        let response = match turn_entered {
            Some(entered) => entered.scope(generation).await,
            None => generation.await,
        };

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

        // A fresh generation: no finish until its terminal chunk names one, so
        // the finish of an earlier turn of the request never reads for this
        // one (as the Harmony loop does).
        emitter.set_finish_reason(None);
        // Convert chat stream to Responses API events while accumulating for tool call detection
        // Stream text naturally - it only appears on final iteration (tool iterations have empty content)
        let accumulated_response =
            match convert_and_accumulate_stream(response.into_body(), &mut emitter, &tx, |name| {
                session.has_exposed_tool(name)
            })
            .await
            {
                Ok(response) => response,
                // The client's leaving is read at the next frame (or at
                // the send the emitter could not make): a departure, not a
                // failure of the stream; the error it came with stays in the
                // log for a reader of the cancelled record.
                Err(error) if tx.is_closed() => {
                    debug!(%error, "The chat stream's error met after the client left");
                    break Turns::ClientLeft { between: false };
                }
                Err(error) => break Turns::Failed(error),
            };
        state.record_usage(accumulated_response.usage.as_ref());
        // Truncated or failed generation cannot request MCP execution.
        if accumulated_response
            .choices
            .first()
            .and_then(|choice| choice.finish_reason.as_deref())
            .is_some_and(|reason| matches!(reason, "length" | "failed" | "error"))
        {
            break Turns::Done;
        }

        // Check for tool calls (extract all of them for parallel execution)
        let tool_calls = extract_all_tool_calls_from_chat(&accumulated_response);
        // Neither can a server call the generation did not finish.
        if has_unfinished_mcp_call(&tool_calls, &session) {
            break Turns::Done;
        }

        // The final turn's answer ends the stream whatever it holds: a server
        // call made anyway was hidden by the converter and does not run.
        if !tool_calls.is_empty() && !state.final_turn {
            let (mcp_tool_calls, function_tool_calls, tool_call_limit) =
                partition_tool_calls(tool_calls, &session, &state, max_tool_calls, true);

            // Process each MCP tool call
            let site = CallSite {
                session: &session,
                format_registry: &ctx.mcp_format_registry,
                request: &current_request,
            };
            for tool_call in mcp_tool_calls {
                if let Err(message) =
                    stream_mcp_call(&site, &mut state, &mut emitter, &tx, tool_call).await
                {
                    break 'turns Turns::left_or(&tx, message)?;
                }
            }

            if tool_call_limit == Some(McpToolCallLimit::Safety) {
                break Turns::Failed(json!({
                    "code": "max_tool_calls_exceeded",
                    "message": format!("Internal tool call safety limit ({DEFAULT_MAX_ITERATIONS}) reached"),
                }));
            }

            // process_chunk already emitted these function-call items.
            // Return them to the client without duplicating their events.
            if !function_tool_calls.is_empty() {
                break Turns::Done;
            }

            state.final_turn = tool_call_limit == Some(McpToolCallLimit::User);

            // The next turn starts here, and its stamp is taken here: its
            // dispatch wait is the gateway's time from this instant to its
            // dispatch. Counted from the request's entry it would hold every
            // generation and tool call before it.
            turn_entered = turn_entered.map(|_| RequestStarted::now());

            // Build next request with conversation history
            current_request = build_next_request(&mut state, current_request);

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
                if let Err(message) = emitter.emit_reasoning_item(&tx, Some(reasoning)).await {
                    break 'turns Turns::left_or(&tx, message)?;
                }
            }
        }

        // Text message events already emitted naturally by process_chunk during stream processing
        // (OpenAI router approach - text only appears on final iteration when no tool calls)

        break Turns::Done;
    };

    let terminal_error = match turns {
        Turns::Done => None,
        Turns::Failed(error) => Some(error),
        // A stored foreground stream its client left is recorded as the
        // regular stream records it: cancelled, the open items in progress,
        // under the id it announced; the answer is the departure.
        Turns::ClientLeft { between } => {
            // Only a finish that ended the loop's last generation stands. A
            // departure between generations, or in a generation that ended
            // with tool calls, cut the response short: cancelled.
            if between || emitter.finish_reason() == Some("tool_calls") {
                emitter.set_finish_reason(None);
            }
            let tenant = crate::tenant::storage_tenant(&params.tenant_request_meta);
            return record_departure(
                emitter,
                state.usage.clone(),
                original_request,
                ctx.response_storage.clone(),
                ctx.conversation_storage.clone(),
                ctx.conversation_item_storage.clone(),
                tenant.as_deref(),
                ctx.request_context.clone(),
                &tx,
            )
            .await;
        }
    };
    let terminal_usage = state.usage.as_ref().map(|usage| json!(usage));
    // The final object, read before the terminal event drains the emitter's
    // items; a background response needs its terminal record under the id
    // the client polls.
    let final_response = match terminal_error.as_ref() {
        Some(error) => emitter.finalize_failed(state.usage.clone(), error.clone()),
        None => emitter.finalize(state.usage.clone()),
    };
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
        emitter.emit_terminal(terminal_usage, error, &tx).await?;
    } else {
        emitter
            .emit_terminal(terminal_usage, terminal_error.as_ref(), &tx)
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
    let mut client_gone = ClientGone::of(tx);
    // The parser supplies a complete name before argument deltas. Retain any
    // preceding id-only deltas until that name tells us who executes the call.
    let mut mcp_indices: HashMap<u32, bool> = HashMap::new();
    let mut pending: HashMap<u32, Vec<ToolCallDelta>> = HashMap::new();

    // A frame holds one or more SSE events; every one of them counts.
    'frames: loop {
        let chunk = match next_frame(&mut stream, &mut client_gone).await {
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
        for event in chat_sse_events::<ChatCompletionStreamResponse>(&frame) {
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
#[path = "../../../../../tests/unit/routers/grpc/regular/responses/streaming/tests.rs"]
mod tests;
