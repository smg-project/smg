//! Streaming infrastructure for /v1/responses endpoint

use std::{borrow::Cow, collections::HashMap, sync::Arc};

use axum::{body::Body, http::StatusCode, response::Response};
use bytes::{BufMut, Bytes, BytesMut};
use futures::Stream;
use http::header::{HeaderValue, CONTENT_TYPE};
use openai_protocol::{
    chat::ChatCompletionStreamResponse,
    event_types::{
        ContentPartEvent, FunctionCallEvent, McpEvent, OutputItemEvent, OutputTextEvent,
        ResponseEvent,
    },
    responses::{
        IncludeField, IncompleteDetails, ResponseOutputItem, ResponseReasoningParam,
        ResponseStatus, ResponseTool, ResponseUsage, ResponsesRequest, ResponsesResponse,
        ResponsesToolChoice, ResponsesUsage, Truncation,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use smg_external_router::retry::REQUEST_TIMEOUT_CODE;
use smg_mcp::{self as mcp};
use tokio::sync::oneshot;
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;
use uuid::Uuid;

use super::utils::generation_failure_error;
#[cfg(test)]
use crate::routers::common::sse::sse_channel;
use crate::{
    observability::metrics::Metrics,
    rate_limit::{ReservationAttachment, SharedReservationHandle},
    routers::{
        common::{
            openai_bridge::{self, descriptor, ResponseFormat},
            sse::{SseEncodeError, SseEncoder, SseReceiver, SseSender},
        },
        error,
        grpc::{context::LoadGuards, harmony::responses::ToolResult, regular::symphony},
    },
};

/// Item-id-prefix discriminator for non-format kinds. Format-driven items
/// derive their prefix from `descriptor(format).id_prefix` instead — keeping
/// the wire-shape mapping in one place.
pub(crate) enum OutputItemKind {
    Message,
    McpListTools,
    FunctionCall,
    CustomToolCall,
    Reasoning,
}

impl OutputItemKind {
    fn id_prefix(&self) -> &'static str {
        match self {
            Self::Message => "msg",
            Self::McpListTools => "mcpl",
            Self::FunctionCall => "fc",
            Self::CustomToolCall => "ctc",
            Self::Reasoning => "rs",
        }
    }

    /// A fresh id for an output item of this kind: the kind's prefix and a
    /// unique suffix, the one shape an item id has on every path, streamed or
    /// not. An item's id is its own; a function call's `call_id` is another
    /// string, the one the client echoes in its `function_call_output`.
    pub(crate) fn new_id(&self) -> String {
        ResponseStreamEventEmitter::generate_item_id(self.id_prefix())
    }
}

/// Status of an output item
#[derive(Debug, Clone, PartialEq)]
enum ItemStatus {
    InProgress,
    Completed,
}

/// State tracking for a single output item
#[derive(Debug, Clone)]
struct OutputItemState {
    output_index: usize,
    status: ItemStatus,
    item_data: Option<serde_json::Value>,
}

/// Streaming state for one in-flight chat tool-call being translated into a
/// Responses `function_call` output item.
struct ToolCallStreamItem {
    /// `delta.tool_calls[].index` from the chat stream.
    chat_index: u32,
    output_index: usize,
    item_id: String,
    call_id: String,
    name: String,
    /// Whether `name` is one of the request's custom tools; decided when the
    /// name arrives, so no later delta builds the request's custom-tool set.
    custom: bool,
    arguments: String,
    added_emitted: bool,
}

/// The bounded `reason` label for the stream-failure counter.
///
/// The gateway emits three error codes on this path: `stream_error` when the
/// backend stream could not be read, the request timeout's code when the
/// timeout cut it, counted as the stream error it is, and `server_error`
/// when the engine reported a failed generation (also the case when the
/// finish reason alone says so and no error object was built). Anything else
/// counts as `other` so a client-visible code can never widen the label set.
fn failure_reason(error: Option<&serde_json::Value>) -> &'static str {
    match error.and_then(|e| e.get("code")).and_then(|c| c.as_str()) {
        Some("stream_error" | REQUEST_TIMEOUT_CODE) => "stream_error",
        Some("server_error") | None => "server_error",
        Some("storage_unavailable") => "storage_unavailable",
        Some(_) => "other",
    }
}

/// The text delta of one token, `response.output_text.delta` or
/// `response.reasoning_text.delta` by `head`, written straight into its
/// frame: the fields in the event's order (the bytes the `json!` map wrote
/// under `preserve_order`), the item id and the delta borrowed rather than
/// copied into a map per token.
struct TextDelta<'a> {
    /// The frame up to its sequence number: [`OUTPUT_TEXT_DELTA_HEAD`] or
    /// [`REASONING_TEXT_DELTA_HEAD`].
    head: &'static [u8],
    sequence_number: u64,
    output_index: usize,
    item_id: &'a str,
    content_index: usize,
    delta: &'a str,
}

const OUTPUT_TEXT_DELTA_HEAD: &[u8] = b"event: response.output_text.delta\n\
    data: {\"type\":\"response.output_text.delta\",\"sequence_number\":";
const REASONING_TEXT_DELTA_HEAD: &[u8] = b"event: response.reasoning_text.delta\n\
    data: {\"type\":\"response.reasoning_text.delta\",\"sequence_number\":";

/// The room of a text delta frame beyond its two strings, and the size of the
/// block the emitter cuts those frames from.
const TEXT_DELTA_ROOM: usize = 224;
const TEXT_DELTA_BLOCK: usize = 4096;

impl TextDelta<'_> {
    /// The event's frame, written in one pass onto the end of `frame`: the
    /// constant text as literals and serde for the three numbers and the two
    /// strings, so a token costs the escape of its strings and not a walk of
    /// six keys and a scan of the event's type. The bytes are those
    /// [`SseEncoder::encode_event`] writes for the event, which
    /// `tests::text_delta_frames_write_the_bytes_of_the_serialized_event` pins.
    fn write(&self, frame: &mut BytesMut) -> Result<(), serde_json::Error> {
        frame.reserve(TEXT_DELTA_ROOM + self.item_id.len() + self.delta.len());
        frame.extend_from_slice(self.head);
        serde_json::to_writer((&mut *frame).writer(), &self.sequence_number)?;
        frame.extend_from_slice(b",\"output_index\":");
        serde_json::to_writer((&mut *frame).writer(), &self.output_index)?;
        frame.extend_from_slice(b",\"item_id\":");
        serde_json::to_writer((&mut *frame).writer(), self.item_id)?;
        frame.extend_from_slice(b",\"content_index\":");
        serde_json::to_writer((&mut *frame).writer(), &self.content_index)?;
        frame.extend_from_slice(b",\"delta\":");
        serde_json::to_writer((&mut *frame).writer(), self.delta)?;
        frame.extend_from_slice(b"}\n\n");
        Ok(())
    }
}

/// One fragment of a call's arguments, `response.function_call_arguments.delta`
/// or `response.mcp_call_arguments.delta` by `kind`, serialized straight into
/// its frame: the fields in the event's order, the item id and the fragment
/// borrowed.
#[derive(Serialize)]
struct ArgumentsDelta<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    sequence_number: u64,
    output_index: usize,
    item_id: &'a str,
    delta: &'a str,
}

const RETAIN_OUTPUT_ITEM: bool = true;

/// The `response.output_item.added` and `response.output_item.done` events:
/// the item an output index opens or closes with, borrowed into the frame
/// rather than copied into an event map first; the fields in the events'
/// order.
#[derive(Serialize)]
struct OutputItemFrame<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    sequence_number: u64,
    output_index: usize,
    item: &'a serde_json::Value,
}

/// The `response.created` event: the response object as the stream opens,
/// the fields in the event's order, the strings borrowed from the emitter.
#[derive(Serialize)]
struct ResponseCreated<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    sequence_number: u64,
    response: CreatedResponse<'a>,
}

#[derive(Serialize)]
struct CreatedResponse<'a> {
    id: &'a str,
    object: &'static str,
    created_at: u64,
    status: &'static str,
    background: bool,
    model: &'a str,
    /// No output items yet: `[]`.
    output: [(); 0],
}

/// The `response.in_progress` event.
#[derive(Serialize)]
struct ResponseInProgress<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    sequence_number: u64,
    response: InProgressResponse<'a>,
}

#[derive(Serialize)]
struct InProgressResponse<'a> {
    id: &'a str,
    object: &'static str,
    status: &'static str,
}

/// The event that ends the stream, `response.completed`,
/// `response.incomplete` or `response.failed` by `kind`: the final response
/// object, the fields in the event's order, the request's fields borrowed.
#[derive(Serialize)]
struct ResponseTerminal<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    sequence_number: u64,
    response: TerminalResponse<'a>,
}

#[derive(Serialize)]
struct TerminalResponse<'a> {
    id: &'a str,
    object: &'static str,
    created_at: u64,
    status: ResponseStatus,
    background: bool,
    model: &'a str,
    output: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Cow<'a, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    incomplete_details: Option<IncompleteDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<serde_json::Value>,
    /// Absent when the stream was never given its request.
    #[serde(flatten)]
    request: Option<RequestEcho<'a>>,
}

/// The request's fields a final response object echoes; the last five read
/// their defaults when the request left them out.
#[derive(Serialize)]
struct RequestEcho<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tool_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<&'a ResponseReasoningParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncation: Option<Truncation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
    parallel_tool_calls: bool,
    store: bool,
    tools: &'a [ResponseTool],
    metadata: Cow<'a, HashMap<String, serde_json::Value>>,
    tool_choice: Cow<'a, ResponsesToolChoice>,
}

/// A content part's `added` and `done` events: the part's text is empty when
/// the part is added and the whole accumulated text when it is done, borrowed
/// rather than copied into a map.
#[derive(Serialize)]
struct ContentPartFrame<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    sequence_number: u64,
    output_index: usize,
    item_id: &'a str,
    content_index: usize,
    part: OutputTextPart<'a>,
}

#[derive(Serialize)]
struct OutputTextPart<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
}

/// The `response.output_text.done` event: the whole text of the part, borrowed.
#[derive(Serialize)]
struct OutputTextDone<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    sequence_number: u64,
    output_index: usize,
    item_id: &'a str,
    content_index: usize,
    text: &'a str,
}

/// OpenAI-compatible event emitter for /v1/responses streaming
///
/// Manages state and sequence numbers to emit proper event types:
/// - response.created
/// - response.in_progress
/// - response.output_item.added
/// - response.output_item.done
/// - response.completed
/// - response.incomplete
/// - response.content_part.added
/// - response.content_part.done
/// - response.output_text.delta
/// - response.output_text.done
/// - response.mcp_list_tools.in_progress
/// - response.mcp_list_tools.completed
/// - response.mcp_call.in_progress
/// - response.mcp_call_arguments.delta
/// - response.mcp_call_arguments.done
/// - response.mcp_call.completed
/// - response.mcp_call.failed
/// - response.web_search_call.in_progress
/// - response.web_search_call.searching
/// - response.web_search_call.completed
/// - response.function_call_arguments.delta
/// - response.function_call_arguments.done
pub(crate) struct ResponseStreamEventEmitter {
    sequence_number: u64,
    pub response_id: String,
    model: String,
    created_at: u64,
    accumulated_text: String,
    has_emitted_output_item_added: bool,
    has_emitted_content_part_added: bool,
    // Output item tracking
    output_items: Vec<OutputItemState>,
    next_output_index: usize,
    current_message_output_index: Option<usize>,
    current_item_id: Option<String>,
    original_request: Option<ResponsesRequest>,
    // Local tool declarations after inline declarations and history are resolved.
    effective_tools: Option<Vec<ResponseTool>>,
    tool_call_items: Vec<ToolCallStreamItem>,
    /// In-flight reasoning item: opened on the first reasoning delta, closed
    /// before the first message/tool item or on finish.
    reasoning_item: Option<ReasoningStreamItem>,
    /// Chat `finish_reason` of the final chunk, e.g. `"length"` for a
    /// `max_output_tokens` truncation. Drives the terminal event's status.
    finish_reason: Option<String>,
    /// Whether a closed call was cut: its function-shaped arguments read
    /// `incomplete` under the finish reason, before a custom tool's call became
    /// its `custom_tool_call` item, which carries no status. Recorded by the
    /// one read of a call's status (`function_call_status`), whichever stream
    /// closes the call; the terminal event and the stored object read it.
    cut_call: bool,
    /// The response runs in the background: `response.created` reports it
    /// `queued`, `response.queued` follows, and the objects echo the flag.
    background: bool,
    /// The error of the last `error` event sent, for the runner of a
    /// background stream that ends without a terminal object.
    last_error: Option<serde_json::Value>,
    /// Frames every event of the stream but the text deltas (see
    /// `delta_frames`) through one reusable buffer.
    encoder: SseEncoder,
    /// The block the text delta frames, one a token, are cut from.
    delta_frames: BytesMut,
    /// The client is gone: nothing more is sent, and nothing counts as
    /// delivered; the items still open close into the final object without
    /// their events, so the record of a stored response keeps them.
    abandoned: bool,
}

/// Streaming state for the reasoning output item of the current turn.
struct ReasoningStreamItem {
    output_index: usize,
    item_id: String,
    text: String,
}

/// The kind of a serde error the stored object's typed read met, for its log
/// line: serde's invalid-type and unknown-variant texts quote the offending
/// value, which may be model output, so the text itself is never logged.
fn serde_error_kind(error: &serde_json::Error) -> &'static str {
    let text = error.to_string();
    if text.starts_with("missing field") {
        "missing field"
    } else if text.starts_with("invalid type")
        || text.starts_with("invalid value")
        || text.starts_with("invalid length")
    {
        "type mismatch"
    } else if text.starts_with("unknown variant") || text.starts_with("unknown field") {
        "unknown name"
    } else if text.starts_with("duplicate field") {
        "duplicate field"
    } else {
        "other"
    }
}

/// One tool-call delta of a chat chunk, as the Responses events read it.
pub struct ToolCallParts<'a> {
    pub index: u32,
    pub id: Option<&'a str>,
    pub name: Option<&'a str>,
    pub arguments: Option<&'a str>,
}

/// What the Responses events read of one chat chunk: its first choice's
/// delta texts, its tool-call deltas in order and its finish reason. The
/// owned chunk answers from its fields; a reader that borrows the chunk from
/// its frame answers from the frame's text, with no copy of it.
pub trait ChatChunkParts {
    fn reasoning(&self) -> Option<&str>;
    fn content(&self) -> Option<&str>;
    /// The tool-call delta at position `at` of the delta's list.
    fn tool_call(&self, at: usize) -> Option<ToolCallParts<'_>>;
    fn finish_reason(&self) -> Option<&str>;
}

impl ChatChunkParts for ChatCompletionStreamResponse {
    fn reasoning(&self) -> Option<&str> {
        self.choices.first()?.delta.reasoning_content.as_deref()
    }

    fn content(&self) -> Option<&str> {
        self.choices.first()?.delta.content.as_deref()
    }

    fn tool_call(&self, at: usize) -> Option<ToolCallParts<'_>> {
        let call = self.choices.first()?.delta.tool_calls.as_ref()?.get(at)?;
        let function = call.function.as_ref();
        Some(ToolCallParts {
            index: call.index,
            id: call.id.as_deref(),
            name: function.and_then(|f| f.name.as_deref()),
            arguments: function.and_then(|f| f.arguments.as_deref()),
        })
    }

    fn finish_reason(&self) -> Option<&str> {
        self.choices.first()?.finish_reason.as_deref()
    }
}

impl ResponseStreamEventEmitter {
    /// The model label the response's events carry.
    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    pub fn new(response_id: String, model: String, created_at: u64) -> Self {
        Self {
            sequence_number: 0,
            response_id,
            model,
            created_at,
            accumulated_text: String::new(),
            has_emitted_output_item_added: false,
            has_emitted_content_part_added: false,
            output_items: Vec::new(),
            next_output_index: 0,
            current_message_output_index: None,
            current_item_id: None,
            original_request: None,
            effective_tools: None,
            tool_call_items: Vec::new(),
            reasoning_item: None,
            finish_reason: None,
            cut_call: false,
            background: false,
            last_error: None,
            encoder: SseEncoder::new(),
            delta_frames: BytesMut::with_capacity(TEXT_DELTA_BLOCK),
            abandoned: false,
        }
    }

    /// Set the original request for including all fields in response.completed
    pub fn set_original_request(&mut self, request: ResponsesRequest) {
        self.background = request.background.unwrap_or(false);
        self.original_request = Some(request);
    }

    pub(crate) fn set_effective_tools(&mut self, tools: Option<Vec<ResponseTool>>) {
        self.effective_tools = tools;
    }

    pub(crate) fn effective_tools(&self) -> Option<&[ResponseTool]> {
        self.effective_tools
            .as_deref()
            .or_else(|| self.original_request.as_ref()?.tools.as_deref())
    }

    /// The finish reason of the iteration a stream that drives the emitter
    /// through its own events (never through chat chunks) has read: the
    /// terminal event, the stored object and the items the stream closes read
    /// their status from it, as they read the final chunk's. Cleared when each
    /// iteration starts and set when its terminal frame comes, so the finish
    /// of an earlier turn never reads for a later one.
    pub(crate) fn set_finish_reason(&mut self, finish_reason: Option<&str>) {
        self.finish_reason = finish_reason.map(str::to_string);
    }

    /// The finish reason the stream has read so far, if any.
    pub(crate) fn finish_reason(&self) -> Option<&str> {
        self.finish_reason.as_deref()
    }

    /// The client left: from here on nothing is sent and nothing counts as
    /// delivered, so the items still open can be closed into the final
    /// object (`close_items`, then `finalize`) for the record of a stored
    /// response the client will come back for.
    pub(crate) fn abandon(&mut self) {
        self.abandoned = true;
    }

    /// Update tool call output items with tool execution results.
    ///
    /// Replaces each matched item's stored payload with the authoritative
    /// `ResponseOutputItem` produced by `ResponseTransformer::transform`
    /// (carried on `ToolResult::output_item`). That transformer is the
    /// single source of truth for per-tool shape — e.g. `mcp_call`
    /// receives `output: string`, `web_search_call` receives
    /// `{status, action, ...}`, and `image_generation_call` receives
    /// `result: base64`. The streaming emitter's initial
    /// `output_item.added` carries a partial stub (tool-call id + arguments);
    /// this call overwrites that stub once the MCP result is in hand, so
    /// `response.completed.response.output` sees the same item a
    /// non-streaming response would emit.
    ///
    /// Matching is by the original `call_id` the stub stored, since that
    /// is the only identifier common to every tool-call item type.
    pub(crate) fn update_mcp_call_outputs(&mut self, tool_results: &[ToolResult]) {
        for tool_result in tool_results {
            let Ok(item_value) = serde_json::to_value(&tool_result.output_item) else {
                warn!(
                    call_id = %tool_result.call_id,
                    "Failed to serialize transformed tool output item; keeping streaming stub"
                );
                continue;
            };
            for item_state in &mut self.output_items {
                if let Some(ref mut item_data) = item_state.item_data {
                    if item_data.get("call_id").and_then(|c| c.as_str())
                        == Some(&tool_result.call_id)
                    {
                        *item_data = item_value;
                        break;
                    }
                }
            }
        }
    }

    fn next_sequence(&mut self) -> u64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }

    /// `response.created`, the first frame of the stream, sent as
    /// [`ResponseCreated`].
    pub async fn send_created(&mut self, tx: &SseSender) -> Result<(), String> {
        let sequence_number = self.next_sequence();
        let event = ResponseCreated {
            kind: ResponseEvent::CREATED,
            sequence_number,
            response: CreatedResponse {
                id: &self.response_id,
                object: "response",
                created_at: self.created_at,
                status: if self.background {
                    "queued"
                } else {
                    "in_progress"
                },
                background: self.background,
                model: &self.model,
                output: [],
            },
        };
        Self::send_frame(
            &mut self.encoder,
            self.abandoned,
            ResponseEvent::CREATED,
            &event,
            tx,
        )
        .await
    }

    /// The event after `response.created` on a background stream: the
    /// response is accepted and waits for its turn.
    pub fn emit_queued(&mut self) -> serde_json::Value {
        json!({
            "type": ResponseEvent::QUEUED,
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "status": "queued",
                "background": true
            }
        })
    }

    /// `response.in_progress`, sent as [`ResponseInProgress`].
    pub async fn send_in_progress(&mut self, tx: &SseSender) -> Result<(), String> {
        let sequence_number = self.next_sequence();
        let event = ResponseInProgress {
            kind: ResponseEvent::IN_PROGRESS,
            sequence_number,
            response: InProgressResponse {
                id: &self.response_id,
                object: "response",
                status: "in_progress",
            },
        };
        Self::send_frame(
            &mut self.encoder,
            self.abandoned,
            ResponseEvent::IN_PROGRESS,
            &event,
            tx,
        )
        .await
    }

    /// `content_part.added` for the text part, sent as [`ContentPartFrame`]
    /// with an empty text.
    pub async fn send_content_part_added(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
        tx: &SseSender,
    ) -> Result<(), String> {
        self.has_emitted_content_part_added = true;
        let event = ContentPartFrame {
            kind: ContentPartEvent::ADDED,
            sequence_number: self.next_sequence(),
            output_index,
            item_id,
            content_index,
            part: OutputTextPart {
                kind: "output_text",
                text: "",
            },
        };
        Self::send_frame(
            &mut self.encoder,
            self.abandoned,
            ContentPartEvent::ADDED,
            &event,
            tx,
        )
        .await
    }

    /// `output_text.done` with the whole accumulated text, sent as
    /// [`OutputTextDone`]; the text is borrowed, not copied.
    pub async fn send_text_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
        tx: &SseSender,
    ) -> Result<(), String> {
        let sequence_number = self.next_sequence();
        let event = OutputTextDone {
            kind: OutputTextEvent::DONE,
            sequence_number,
            output_index,
            item_id,
            content_index,
            text: &self.accumulated_text,
        };
        Self::send_frame(
            &mut self.encoder,
            self.abandoned,
            OutputTextEvent::DONE,
            &event,
            tx,
        )
        .await
    }

    /// `content_part.done` with the whole accumulated text, sent as
    /// [`ContentPartFrame`]; the text is borrowed, not copied.
    pub async fn send_content_part_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
        tx: &SseSender,
    ) -> Result<(), String> {
        let sequence_number = self.next_sequence();
        let event = ContentPartFrame {
            kind: ContentPartEvent::DONE,
            sequence_number,
            output_index,
            item_id,
            content_index,
            part: OutputTextPart {
                kind: "output_text",
                text: &self.accumulated_text,
            },
        };
        Self::send_frame(
            &mut self.encoder,
            self.abandoned,
            ContentPartEvent::DONE,
            &event,
            tx,
        )
        .await
    }

    /// The outcome of a send for the streams that log a failed send instead
    /// of ending on it: `true` when the event went out.
    pub fn sent(result: Result<(), String>) -> bool {
        match result {
            Ok(()) => true,
            Err(e) => {
                tracing::debug!("Failed to send event (likely client disconnect): {}", e);
                false
            }
        }
    }

    /// The stream's terminal event, the one the finish reason gives, and the
    /// encoder that writes it; `error` is that of a response its caller failed.
    // INVARIANT: this method is terminal — it drains internal state via `take()`
    // and must only be called once per emitter lifetime.
    fn terminal<'a>(
        &'a mut self,
        usage: Option<serde_json::Value>,
        error: Option<&'a serde_json::Value>,
    ) -> (ResponseTerminal<'a>, &'a mut SseEncoder) {
        debug_assert!(
            error.is_none() || self.finish_reason.as_deref() == Some("failed"),
            "a terminal with an error must be read under the failed finish reason"
        );
        // Build output array from tracked items
        let output: Vec<serde_json::Value> = self
            .output_items
            .iter_mut()
            .filter_map(|item| {
                if item.status == ItemStatus::Completed {
                    item.item_data.take()
                } else {
                    None
                }
            })
            .collect();

        // A turn without items reads `output: []` here as it does in the
        // stored object: no message is invented that nothing announced.

        // The event's status is read where the stored object and the
        // non-streaming conversion read theirs: a `length` finish is incomplete
        // with `max_output_tokens`, a cut call is incomplete without details, a
        // failed generation is failed; the stream ends with the event of that
        // status.
        let (status, incomplete_details) =
            super::utils::response_status(self.finish_reason.as_deref(), self.cut_call);
        let sequence_number = self.next_sequence();
        let event = ResponseTerminal {
            kind: match status {
                ResponseStatus::Failed => ResponseEvent::FAILED,
                ResponseStatus::Incomplete => ResponseEvent::INCOMPLETE,
                _ => ResponseEvent::COMPLETED,
            },
            sequence_number,
            response: TerminalResponse {
                id: &self.response_id,
                object: "response",
                created_at: self.created_at,
                status,
                background: self.background,
                model: &self.model,
                output,
                error: error.map(Cow::Borrowed).or_else(|| {
                    generation_failure_error(self.finish_reason.as_deref()).map(Cow::Owned)
                }),
                incomplete_details,
                usage,
                request: self.original_request.as_ref().map(|req| RequestEcho {
                    instructions: req.instructions.as_deref(),
                    max_output_tokens: req.max_output_tokens,
                    max_tool_calls: req.max_tool_calls,
                    previous_response_id: req.previous_response_id.as_deref(),
                    reasoning: req.reasoning.as_ref(),
                    // The spelling of the stored record (`1e-6`): a 32-bit
                    // field would write `0.000001`.
                    temperature: req.temperature.map(super::utils::exact_f32),
                    top_p: req.top_p.map(super::utils::exact_f32),
                    truncation: req.truncation,
                    user: req.user.as_deref(),
                    parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
                    store: req.store.unwrap_or(true),
                    tools: req.tools.as_deref().unwrap_or_default(),
                    metadata: req
                        .metadata
                        .as_ref()
                        .map_or_else(Cow::default, Cow::Borrowed),
                    tool_choice: req
                        .tool_choice
                        .as_ref()
                        .map_or_else(Cow::default, Cow::Borrowed),
                }),
            },
        };
        (event, &mut self.encoder)
    }

    #[cfg(test)]
    pub fn emit_completed(&mut self, usage: Option<serde_json::Value>) -> serde_json::Value {
        serde_json::to_value(self.terminal(usage, None).0).unwrap()
    }

    /// Close the item event sequences still open (the reasoning item, the
    /// message, the tool calls), retaining unfinished wire statuses, so the
    /// final object reads every streamed item under the id its events carry.
    /// A stream that ended on a finish chunk has closed them already; one cut
    /// short before its finish (an upstream error, a body that ended) closes
    /// them here, before its final object is read and stored.
    pub(crate) async fn close_items(&mut self, tx: &SseSender) -> Result<(), String> {
        self.close_reasoning_item(tx).await?;
        self.close_message_item(tx).await?;
        self.close_tool_call_items(tx).await
    }

    /// Close remaining item event sequences, retaining unfinished wire statuses,
    /// and emit exactly one terminal event.
    pub async fn emit_terminal(
        &mut self,
        usage: Option<serde_json::Value>,
        error: Option<&serde_json::Value>,
        tx: &SseSender,
    ) -> Result<(), String> {
        if error.is_some() && self.finish_reason.as_deref() != Some("failed") {
            self.finish_reason = Some("failed".into());
        }
        self.close_items(tx).await?;
        let abandoned = self.abandoned;
        let (event, encoder) = self.terminal(usage, error);
        let failed = event.kind == ResponseEvent::FAILED;
        Self::send_frame(encoder, abandoned, event.kind, &event, tx).await?;
        if failed {
            // Counted only once the terminal reached the client: a stream that
            // dies before its terminal is not a delivered failure, and the
            // completed/incomplete terminals never count here.
            Metrics::record_responses_stream_failure(&self.model, failure_reason(error));
        }
        Ok(())
    }

    /// The terminal of a run whose items the stream processor has already
    /// closed: `response.completed` with the usage, or `response.failed` when
    /// the record could not be stored. Sent best-effort, since nothing follows;
    /// the failure is counted only once the terminal reached the client, as
    /// [`emit_terminal`](Self::emit_terminal) counts. Unlike `emit_terminal`
    /// it closes no item, so an item left open stays as the processor left it.
    pub async fn finish_stream(
        &mut self,
        usage: Option<&ResponseUsage>,
        not_stored: Option<&serde_json::Value>,
        tx: &SseSender,
    ) {
        let usage_json = usage.map(|usage| json!(usage));
        if not_stored.is_some() {
            self.finish_reason = Some("failed".into());
        }
        let abandoned = self.abandoned;
        let (event, encoder) = self.terminal(usage_json, not_stored);
        let delivered = !abandoned
            && Self::sent(Self::send_frame(encoder, abandoned, event.kind, &event, tx).await);
        if delivered && not_stored.is_some() {
            Metrics::record_responses_stream_failure(&self.model, failure_reason(not_stored));
        }
    }

    /// Convert tool entries to JSON values using the shared bridge builder.
    fn tool_entries_to_json(
        tools: &[mcp::ToolEntry],
    ) -> Result<Vec<serde_json::Value>, serde_json::Error> {
        openai_bridge::build_mcp_tool_infos(tools)
            .into_iter()
            .map(serde_json::to_value)
            .collect()
    }

    // ========================================================================
    // MCP Event Emission Methods
    // ========================================================================

    /// The list item's events carry the item's id after the output index,
    /// as the call events do: a client joins them to the item's
    /// `output_item.added` by `item_id`.
    pub fn emit_mcp_list_tools_in_progress(
        &mut self,
        output_index: usize,
        item_id: &str,
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::LIST_TOOLS_IN_PROGRESS,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id
        })
    }

    pub fn emit_mcp_list_tools_completed(
        &mut self,
        output_index: usize,
        item_id: &str,
        tool_items: &[serde_json::Value],
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::LIST_TOOLS_COMPLETED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "tools": tool_items
        })
    }

    /// The failure itself is on the done item (`error`), not on the event.
    pub fn emit_mcp_list_tools_failed(
        &mut self,
        output_index: usize,
        item_id: &str,
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::LIST_TOOLS_FAILED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id
        })
    }

    pub fn emit_mcp_call_arguments_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        arguments: &str,
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::CALL_ARGUMENTS_DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "arguments": arguments
        })
    }

    pub fn emit_mcp_call_failed(
        &mut self,
        output_index: usize,
        item_id: &str,
        error: &str,
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::CALL_FAILED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "error": error
        })
    }

    // ========================================================================
    // Generic Tool Call Event Emission (based on ResponseFormat)
    // ========================================================================

    /// Emit a tool call event with the specified event type.
    /// This is the internal helper used by all tool call event methods.
    fn emit_tool_event(
        &mut self,
        event_type: &'static str,
        output_index: usize,
        item_id: &str,
    ) -> serde_json::Value {
        json!({
            "type": event_type,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id
        })
    }

    pub fn emit_tool_call_in_progress(
        &mut self,
        output_index: usize,
        item_id: &str,
        response_format: ResponseFormat,
    ) -> serde_json::Value {
        let event_type = descriptor(response_format).in_progress_event;
        self.emit_tool_event(event_type, output_index, item_id)
    }

    /// Emit the searching/interpreting/generating event; `None` for formats
    /// with no intermediate phase.
    pub fn emit_tool_call_searching(
        &mut self,
        output_index: usize,
        item_id: &str,
        response_format: ResponseFormat,
    ) -> Option<serde_json::Value> {
        let event_type = descriptor(response_format).searching_event?;
        Some(self.emit_tool_event(event_type, output_index, item_id))
    }

    pub fn emit_tool_call_completed(
        &mut self,
        output_index: usize,
        item_id: &str,
        response_format: ResponseFormat,
    ) -> serde_json::Value {
        let event_type = descriptor(response_format).completed_event;
        self.emit_tool_event(event_type, output_index, item_id)
    }

    pub fn type_str_for_format(response_format: Option<&ResponseFormat>) -> &'static str {
        match response_format {
            Some(format) => descriptor(*format).type_str,
            None => "function_call",
        }
    }

    // ========================================================================
    // Function Call Event Emission Methods
    // ========================================================================

    pub fn emit_function_call_arguments_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        arguments: &str,
    ) -> serde_json::Value {
        json!({
            "type": FunctionCallEvent::ARGUMENTS_DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "arguments": arguments
        })
    }

    // ========================================================================
    // Output Item Wrapper Events
    // ========================================================================

    /// Restore declared namespace identities for streamed function and custom calls.
    fn normalize_function_item<'a>(
        &self,
        item: &'a serde_json::Value,
    ) -> Cow<'a, serde_json::Value> {
        if matches!(
            item["type"].as_str(),
            Some("function_call" | "custom_tool_call")
        ) && item["namespace"].is_null()
        {
            if let Some(name) = item["name"].as_str() {
                let tools = self.effective_tools();
                let (name, namespace) = super::utils::resolve_function_identity(tools, name);
                if let Some(namespace) = namespace {
                    let mut item = item.clone();
                    item["name"] = json!(name);
                    item["namespace"] = json!(namespace);
                    return Cow::Owned(item);
                }
            }
        }
        Cow::Borrowed(item)
    }

    /// The `response.output_item.added` event of `item` under `output_index`,
    /// sent as [`OutputItemFrame`]: the item goes into the frame borrowed,
    /// not through a copy in an event map.
    pub async fn send_output_item_added(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
        tx: &SseSender,
    ) -> Result<(), String> {
        let item = self.normalize_function_item(item);
        let event = OutputItemFrame {
            kind: OutputItemEvent::ADDED,
            sequence_number: self.next_sequence(),
            output_index,
            item: &item,
        };
        Self::send_frame(
            &mut self.encoder,
            self.abandoned,
            OutputItemEvent::ADDED,
            &event,
            tx,
        )
        .await
    }

    /// The `response.output_item.done` event of the item `output_index`
    /// closes with, sent as [`OutputItemFrame`]. When `retain`, the emitter
    /// retains the item for the terminal event and stored object, and the
    /// frame borrows it: a close costs the item once, not the event map's copy
    /// and a retained clone besides, except for a namespaced function call,
    /// whose restored copy is retained. A partial item rejected by the
    /// completed parse closes with `retain` false. An untracked index always
    /// retains nothing.
    pub async fn send_output_item_done(
        &mut self,
        output_index: usize,
        item: serde_json::Value,
        retain: bool,
        tx: &SseSender,
    ) -> Result<(), String> {
        let item = match self.normalize_function_item(&item) {
            Cow::Owned(normalized) => normalized,
            Cow::Borrowed(_) => item,
        };
        let sequence_number = self.next_sequence();
        let kept: &serde_json::Value = if retain {
            match self
                .output_items
                .iter_mut()
                .find(|tracked| tracked.output_index == output_index)
            {
                Some(tracked) => tracked.item_data.insert(item),
                None => &item,
            }
        } else {
            &item
        };
        let event = OutputItemFrame {
            kind: OutputItemEvent::DONE,
            sequence_number,
            output_index,
            item: kept,
        };
        Self::send_frame(
            &mut self.encoder,
            self.abandoned,
            OutputItemEvent::DONE,
            &event,
            tx,
        )
        .await
    }

    /// Keeps the item `output_index` closes with, for a close whose
    /// `response.output_item.done` event is sent later by
    /// [`Self::send_kept_output_item_done`]: a hosted tool call keeps its
    /// item before the completion sends that can fail on a closed channel, so
    /// a client who leaves during the call still has the executed call in the
    /// record. A namespaced function call is kept as its restored copy; an
    /// index the emitter does not track keeps nothing.
    pub fn keep_output_item(&mut self, output_index: usize, item: serde_json::Value) {
        let item = match self.normalize_function_item(&item) {
            Cow::Owned(normalized) => normalized,
            Cow::Borrowed(_) => item,
        };
        if let Some(tracked) = self
            .output_items
            .iter_mut()
            .find(|tracked| tracked.output_index == output_index)
        {
            tracked.item_data = Some(item);
        }
    }

    /// The `response.output_item.done` event of an item kept by
    /// [`Self::keep_output_item`], the frame borrowing the kept item; an
    /// index with nothing kept is an error, not a silent omission.
    pub async fn send_kept_output_item_done(
        &mut self,
        output_index: usize,
        tx: &SseSender,
    ) -> Result<(), String> {
        let sequence_number = self.next_sequence();
        let kept = self
            .output_items
            .iter()
            .find(|tracked| tracked.output_index == output_index)
            .and_then(|tracked| tracked.item_data.as_ref())
            .ok_or_else(|| format!("no kept item at output index {output_index}"))?;
        let event = OutputItemFrame {
            kind: OutputItemEvent::DONE,
            sequence_number,
            output_index,
            item: kept,
        };
        Self::send_frame(
            &mut self.encoder,
            self.abandoned,
            OutputItemEvent::DONE,
            &event,
            tx,
        )
        .await
    }

    /// Generate unique ID for item type
    fn generate_item_id(prefix: &str) -> String {
        format!("{}_{}", prefix, Uuid::now_v7().simple())
    }

    /// Allocate next output index and track item, deriving the item-id from
    /// `id_prefix` (e.g. `"msg"`, `"mcp"`, `"ws"` — without trailing `_`).
    pub fn allocate_output_index_with_prefix(&mut self, id_prefix: &str) -> (usize, String) {
        let index = self.next_output_index;
        self.next_output_index += 1;

        let id = Self::generate_item_id(id_prefix);

        self.output_items.push(OutputItemState {
            output_index: index,
            status: ItemStatus::InProgress,
            item_data: None,
        });

        (index, id)
    }

    /// Convenience: allocate for a non-format kind.
    pub fn allocate_output_index(&mut self, kind: OutputItemKind) -> (usize, String) {
        self.allocate_output_index_with_prefix(kind.id_prefix())
    }

    /// Convenience: allocate for a `ResponseFormat`-driven item, falling back
    /// to `function_call` (`"fc"`) when no format applies. Mirrors the prior
    /// `output_item_type_for_format` mapping but reads the prefix straight
    /// off the format descriptor.
    pub fn allocate_output_index_for_format(
        &mut self,
        response_format: Option<ResponseFormat>,
    ) -> (usize, String) {
        let prefix = response_format
            .map(|f| descriptor(f).id_prefix)
            .unwrap_or("fc");
        self.allocate_output_index_with_prefix(prefix)
    }

    /// Mark output item as completed and store its data
    pub fn complete_output_item(&mut self, output_index: usize) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.status = ItemStatus::Completed;
        }
    }

    /// Finalize and return the complete ResponsesResponse
    ///
    /// This constructs the final ResponsesResponse from all accumulated output items
    /// for persistence. Should be called after streaming is complete.
    /// Reads non-destructively so the terminal event can still drain state afterwards.
    pub fn finalize(&self, usage: Option<ResponseUsage>) -> ResponsesResponse {
        // Build output array from tracked items, read in place (the terminal event drains later)
        let output: Vec<ResponseOutputItem> = self
            .output_items
            .iter()
            .filter_map(|item| {
                let data = item.item_data.as_ref()?;
                match ResponseOutputItem::deserialize(data) {
                    Ok(typed) => Some(typed),
                    Err(error) => {
                        // The terminal event carries the raw item; the stored object carries
                        // what reads as a typed item, so a drop is said rather than silent.
                        warn!(
                            response_id = %self.response_id,
                            output_index = item.output_index,
                            item_type = data["type"].as_str().unwrap_or(""),
                            error_kind = serde_error_kind(&error),
                            "a streamed output item does not read as a typed output item; \
                             the stored object drops it"
                        );
                        None
                    }
                }
            })
            .collect();

        // The response-level usage the terminal event carries.
        let responses_usage = usage.map(ResponsesUsage::Modern);

        // Match the streamed terminal event: a `length` finish reports
        // status=incomplete with the truncation reason, a cut call without one.
        let (status, incomplete_details) =
            super::utils::response_status(self.finish_reason.as_deref(), self.cut_call);

        // Build response using builder
        let mut builder = ResponsesResponse::builder(&self.response_id, &self.model)
            .created_at(self.created_at as i64)
            .status(status)
            .output(output)
            .maybe_copy_from_request(self.original_request.as_ref())
            .maybe_usage(responses_usage);
        if let Some(details) = incomplete_details {
            builder = builder.incomplete_details(details);
        }
        if let Some(error) = generation_failure_error(self.finish_reason.as_deref()) {
            builder = builder.error(error);
        }
        builder.build()
    }

    /// The final object of a run that ends as `failed` before its terminal
    /// event (the tool-call limit): [`finalize`](Self::finalize)'s object
    /// with the status and the error the terminal will carry. Read before
    /// the terminal, which drains the items: the stored object keeps them.
    pub fn finalize_failed(
        &self,
        usage: Option<ResponseUsage>,
        error: serde_json::Value,
    ) -> ResponsesResponse {
        let mut response = self.finalize(usage);
        response.status = ResponseStatus::Failed;
        response.error = Some(error);
        response.incomplete_details = None;
        response
    }

    /// Emit reasoning item wrapper events (added + done)
    ///
    /// Reasoning items in OpenAI format are simple placeholders emitted between tool iterations.
    /// They don't have streaming content - just wrapper events with empty/null content.
    pub async fn emit_reasoning_item(
        &mut self,
        tx: &SseSender,
        reasoning_content: Option<String>,
    ) -> Result<(), String> {
        // Allocate output index and generate ID
        let (output_index, item_id) = self.allocate_output_index(OutputItemKind::Reasoning);

        // Build reasoning item structure. `content` is an array of typed
        // parts on the wire (a bare string breaks clients walking the
        // content list).
        let item = json!({
            "id": item_id,
            "type": "reasoning",
            "summary": [],
            "content": reasoning_content
                .map(|text| json!([{ "type": "reasoning_text", "text": text }]))
                .unwrap_or(json!([])),
            "encrypted_content": null,
            "status": null
        });

        // Emit output_item.added
        self.send_output_item_added(output_index, &item, tx).await?;

        // Immediately emit output_item.done (no streaming for reasoning)
        self.send_output_item_done(output_index, item, RETAIN_OUTPUT_ITEM, tx)
            .await?;

        // Mark as completed
        self.complete_output_item(output_index);

        Ok(())
    }

    fn include_encrypted_reasoning(&self) -> bool {
        self.original_request
            .as_ref()
            .and_then(|r| r.include.as_deref())
            .is_some_and(|f| f.contains(&IncludeField::ReasoningEncryptedContent))
    }

    pub(crate) fn is_custom_tool(&self, name: &str) -> bool {
        super::utils::custom_tool_names(self.effective_tools()).contains(name)
    }

    pub(crate) fn custom_input_events(
        &mut self,
        output_index: usize,
        item_id: &str,
        input: &str,
    ) -> [serde_json::Value; 2] {
        [
            ("response.custom_tool_call_input.delta", "delta"),
            ("response.custom_tool_call_input.done", "input"),
        ]
        .map(|(event_type, field)| {
            json!({
                "type": event_type,
                "sequence_number": self.next_sequence(),
                "output_index": output_index,
                "item_id": item_id,
                field: input,
            })
        })
    }

    fn emit_reasoning_text_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        text: &str,
    ) -> serde_json::Value {
        json!({
            "type": "response.reasoning_text.done",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": 0,
            "text": text
        })
    }

    /// Close the in-flight reasoning item, if any: `reasoning_text.done`
    /// followed by `content_part.done` and `output_item.done` with the full text.
    async fn close_reasoning_item(&mut self, tx: &SseSender) -> Result<(), String> {
        let Some(reasoning) = self.reasoning_item.take() else {
            return Ok(());
        };
        // The done events go out in order; the item is kept by
        // `send_output_item_done` before its frame goes out, so a send that
        // failed because the client left has not lost it (as for the message
        // item), and the first failure is reported once the item is kept.
        let text_done = self.emit_reasoning_text_done(
            reasoning.output_index,
            &reasoning.item_id,
            &reasoning.text,
        );
        let part_done = json!({
            "type": "response.content_part.done",
            "sequence_number": self.next_sequence(),
            "output_index": reasoning.output_index,
            "item_id": reasoning.item_id,
            "content_index": 0,
            "part": { "type": "reasoning_text", "text": reasoning.text }
        });
        let mut item = json!({
            "id": reasoning.item_id,
            "type": "reasoning",
            "summary": [],
            "content": [{ "type": "reasoning_text", "text": reasoning.text }],
            "status": self.message_status()
        });
        if self.include_encrypted_reasoning() {
            item["encrypted_content"] =
                json!(super::utils::encode_reasoning_content(&reasoning.text));
        }
        let mut sent = Ok(());
        for event in [&text_done, &part_done] {
            if sent.is_ok() {
                sent = self.send_event(event, tx).await;
            }
        }
        let item_sent = self
            .send_output_item_done(reasoning.output_index, item, RETAIN_OUTPUT_ITEM, tx)
            .await;
        self.complete_output_item(reasoning.output_index);
        sent.and(item_sent)
    }

    /// The status of the message or reasoning item a stream closes, under the turn's finish
    /// reason: `in_progress` when the generation failed or the stream was
    /// cancelled before its end (closing the event sequence does not imply it
    /// succeeded; the partial message belongs in the terminal output all the
    /// same), else `completed`. A stream that drives the emitter through its
    /// own events and closes its message item itself reads it here, so its
    /// item carries the status the typed read of the stored object requires.
    pub(crate) fn message_status(&self) -> &'static str {
        if matches!(
            self.finish_reason.as_deref(),
            Some("failed" | "error" | "cancelled")
        ) {
            "in_progress"
        } else {
            "completed"
        }
    }

    /// The status of a function call the stream closes, read under the
    /// finish reason the emitter holds, as `message_status` reads the message
    /// item's: a call whose arguments the finish cut reads `incomplete`, and
    /// the response reads it as cut on its terminal event and stored object.
    /// The chat-chunk path and a stream that closes its own call items read
    /// every call's status here, so the cut is recorded wherever it is read.
    pub(crate) fn function_call_status(&mut self, arguments: &str) -> &'static str {
        let status = super::utils::function_call_status(self.finish_reason.as_deref(), arguments);
        self.cut_call |= status == "incomplete";
        // A normal model finish with non-JSON arguments deserves a diagnostic,
        // without changing the established incomplete response contract. A
        // token limit, cancellation, steering or unknown finish is not evidence
        // of a malformed model call. serde's message can quote private values.
        if status == "incomplete"
            && matches!(self.finish_reason.as_deref(), Some("stop" | "tool_calls"))
        {
            if let Err(error) = serde_json::from_str::<serde_json::Value>(arguments) {
                warn!(
                    response_id = %self.response_id,
                    finish_reason = self.finish_reason.as_deref().unwrap_or_default(),
                    argument_bytes = arguments.len(),
                    json_error_class = ?error.classify(),
                    json_error_line = error.line(),
                    json_error_column = error.column(),
                    "Tool-call arguments are not complete JSON at generation end"
                );
            }
        }
        status
    }

    /// Closing the event sequence does not imply generation succeeded: partial
    /// messages remain in_progress on the wire, but belong in terminal output.
    async fn close_message_item(&mut self, tx: &SseSender) -> Result<(), String> {
        let (Some(output_index), Some(item_id)) = (
            self.current_message_output_index.take(),
            self.current_item_id.take(),
        ) else {
            return Ok(());
        };
        // The part's done events borrow the accumulated text, so they go out
        // before the item takes it; a send that fails because the client left
        // does not stop the close: the item is still kept (by
        // `send_output_item_done`, before its frame goes out) and completed,
        // so the record of the departure reads it, and the first failure is
        // reported last. The sequence numbers stay in send order.
        let mut sent = Ok(());
        if std::mem::take(&mut self.has_emitted_content_part_added) {
            sent = match self.send_text_done(output_index, &item_id, 0, tx).await {
                Ok(()) => {
                    self.send_content_part_done(output_index, &item_id, 0, tx)
                        .await
                }
                Err(e) => Err(e),
            };
        }
        if std::mem::take(&mut self.has_emitted_output_item_added) {
            let item = json!({
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": std::mem::take(&mut self.accumulated_text)}],
                "status": self.message_status(),
            });
            let item_sent = self
                .send_output_item_done(output_index, item, RETAIN_OUTPUT_ITEM, tx)
                .await;
            sent = sent.and(item_sent);
        }
        self.complete_output_item(output_index);
        sent
    }

    /// Close every streamed tool-call item: arguments (or custom input) done,
    /// then output_item.done, as the non-streaming path pairs them. Called on
    /// all finishes and terminal errors — grammar-constrained models can
    /// keep emitting valid tool calls until `max_output_tokens` truncates the
    /// turn, and those (possibly partial) calls still belong in the final
    /// output, so they must be closed and collected here.
    async fn close_tool_call_items(&mut self, tx: &SseSender) -> Result<(), String> {
        // Every call is kept and completed whether or not its events reach
        // the client (as for the message item): a send that fails because
        // the client left does not stop the loop, `send_output_item_done`
        // keeps each call before its frame goes out, and the first failure
        // is reported once every call is kept.
        let mut sent = Ok(());
        for item in std::mem::take(&mut self.tool_call_items) {
            // The call's status as a function call, read before a custom
            // tool's call becomes its item.
            let status = self.function_call_status(&item.arguments);
            let full = if item.custom {
                let input = super::utils::custom_tool_input(&item.arguments);
                for event in self.custom_input_events(item.output_index, &item.item_id, &input) {
                    if sent.is_ok() {
                        sent = self.send_event(&event, tx).await;
                    }
                }
                json!({
                    "id": item.item_id,
                    "type": "custom_tool_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "input": input,
                })
            } else {
                let event = self.emit_function_call_arguments_done(
                    item.output_index,
                    &item.item_id,
                    &item.arguments,
                );
                if sent.is_ok() {
                    sent = self.send_event(&event, tx).await;
                }
                json!({
                    "id": item.item_id,
                    "type": "function_call",
                    "call_id": item.call_id,
                    "name": item.name,
                    "arguments": item.arguments,
                    "status": status,
                })
            };
            let item_sent = self
                .send_output_item_done(item.output_index, full, RETAIN_OUTPUT_ITEM, tx)
                .await;
            sent = sent.and(item_sent);
            self.complete_output_item(item.output_index);
        }
        sent
    }

    /// Process a chunk and emit appropriate events: the reasoning delta, the
    /// content delta, the tool-call deltas, then the finish, in that order.
    pub async fn process_chunk(
        &mut self,
        chunk: &impl ChatChunkParts,
        tx: &SseSender,
    ) -> Result<(), String> {
        if let Some(reasoning) = chunk.reasoning().filter(|r| !r.is_empty()) {
            self.process_reasoning_delta(reasoning, tx).await?;
        }
        if let Some(content) = chunk.content().filter(|c| !c.is_empty()) {
            self.process_content_delta(content, tx).await?;
        }
        let mut at = 0;
        while let Some(call) = chunk.tool_call(at) {
            self.process_tool_call_delta(call, tx).await?;
            at += 1;
        }
        // Check for finish_reason to emit completion events
        if let Some(reason) = chunk.finish_reason() {
            self.finish_reason = Some(reason.to_owned());
            self.close_reasoning_item(tx).await?;
            self.close_message_item(tx).await?;
            self.close_tool_call_items(tx).await?;
        }
        Ok(())
    }

    /// A reasoning delta. Reasoning streams as its own output item, opened on
    /// the first delta so it precedes the answer (OpenAI order).
    async fn process_reasoning_delta(
        &mut self,
        reasoning: &str,
        tx: &SseSender,
    ) -> Result<(), String> {
        if self.reasoning_item.is_none() {
            let (output_index, item_id) = self.allocate_output_index(OutputItemKind::Reasoning);
            let item = json!({
                "id": item_id,
                "type": "reasoning",
                "summary": [],
                "content": [],
                "status": "in_progress"
            });
            self.send_output_item_added(output_index, &item, tx).await?;
            let event = json!({
                "type": "response.content_part.added",
                "sequence_number": self.next_sequence(),
                "output_index": output_index,
                "item_id": item_id,
                "content_index": 0,
                "part": { "type": "reasoning_text", "text": "" }
            });
            self.send_event(&event, tx).await?;
            self.reasoning_item = Some(ReasoningStreamItem {
                output_index,
                item_id,
                text: String::new(),
            });
        }
        if let Some(item) = self.reasoning_item.as_mut() {
            item.text.push_str(reasoning);
            // The id stays borrowed, as the text delta's does: the counter
            // advanced on its field, the frame cut from `delta_frames` alone.
            let sequence_number = self.sequence_number;
            self.sequence_number += 1;
            let event = TextDelta {
                head: REASONING_TEXT_DELTA_HEAD,
                sequence_number,
                output_index: item.output_index,
                item_id: &item.item_id,
                content_index: 0,
                delta: reasoning,
            };
            Self::send_text_delta_frame(&mut self.delta_frames, self.abandoned, &event, tx).await?;
        }
        Ok(())
    }

    /// A content delta. The first one closes the reasoning item and opens the
    /// message item and its content part; every one streams its text.
    async fn process_content_delta(&mut self, content: &str, tx: &SseSender) -> Result<(), String> {
        // Allocate output_index and item_id for this message item (once per message)
        if self.current_item_id.is_none() {
            self.close_reasoning_item(tx).await?;
            let (output_index, item_id) = self.allocate_output_index(OutputItemKind::Message);

            // Build message item structure
            let item = json!({
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "content": []
            });

            // Emit output_item.added
            self.send_output_item_added(output_index, &item, tx).await?;
            self.has_emitted_output_item_added = true;

            // Store for subsequent events
            self.current_item_id = Some(item_id);
            self.current_message_output_index = Some(output_index);
        }

        // output_index and item_id are always set in the block above
        // when current_item_id was None and we allocated new ones
        let Some(output_index) = self.current_message_output_index else {
            return Ok(());
        };
        let content_index = 0; // Single content part for now

        // Emit content_part.added before first delta: once per item, so the
        // id's copy for the borrow split stays on this branch.
        if !self.has_emitted_content_part_added {
            let Some(item_id) = self.current_item_id.clone() else {
                return Ok(());
            };
            self.send_content_part_added(output_index, &item_id, content_index, tx)
                .await?;
        }

        // Emit text delta: the id borrowed from the emitter, the counter
        // advanced on its field (the method would take the whole emitter while
        // the id is borrowed, and the guard must not cost a number when it
        // fires) and the frame cut from `delta_frames` alone, so no copy of
        // the id is made per token.
        let Some(item_id) = self.current_item_id.as_deref() else {
            return Ok(());
        };
        let sequence_number = self.sequence_number;
        self.sequence_number += 1;
        self.accumulated_text.push_str(content);
        let event = TextDelta {
            head: OUTPUT_TEXT_DELTA_HEAD,
            sequence_number,
            output_index,
            item_id,
            content_index,
            delta: content,
        };
        Self::send_text_delta_frame(&mut self.delta_frames, self.abandoned, &event, tx).await
    }

    /// A chat tool-call delta as Responses function_call events; without this
    /// the streamed call is dropped entirely. The call's item is found or
    /// opened (closing the reasoning item first), announced once it has a name
    /// or arguments, and its arguments streamed unless the tool is custom.
    async fn process_tool_call_delta(
        &mut self,
        tc: ToolCallParts<'_>,
        tx: &SseSender,
    ) -> Result<(), String> {
        let pos = match self
            .tool_call_items
            .iter()
            .position(|i| i.chat_index == tc.index)
        {
            Some(pos) => pos,
            None => {
                self.close_reasoning_item(tx).await?;
                let (output_index, item_id) =
                    self.allocate_output_index(OutputItemKind::FunctionCall);
                self.tool_call_items.push(ToolCallStreamItem {
                    chat_index: tc.index,
                    output_index,
                    item_id,
                    call_id: tc
                        .id
                        .map_or_else(|| format!("call_{}", Uuid::now_v7()), str::to_owned),
                    name: String::new(),
                    custom: false,
                    arguments: String::new(),
                    added_emitted: false,
                });
                self.tool_call_items.len() - 1
            }
        };
        let name_delta = tc.name.filter(|n| !n.is_empty());
        let args_delta = tc.arguments.filter(|a| !a.is_empty());

        // The name arrives once; whether it names a custom tool is decided
        // then and kept on the item.
        let arriving_name = name_delta.filter(|_| self.tool_call_items[pos].name.is_empty());
        let arriving_custom = arriving_name.is_some_and(|name| self.is_custom_tool(name));
        let item = &mut self.tool_call_items[pos];
        if let Some(name) = arriving_name {
            item.name = name.to_string();
            item.custom = arriving_custom;
            // The id was minted as a function call's before the name arrived;
            // a custom tool's call takes its own kind's id while nothing has
            // announced the first one (`added` waits for a name or arguments).
            if arriving_custom && !item.added_emitted {
                item.item_id = OutputItemKind::CustomToolCall.new_id();
            }
        }
        if let Some(args) = args_delta {
            item.arguments.push_str(args);
        }
        let emit_added =
            !item.added_emitted && (!item.name.is_empty() || !item.arguments.is_empty());
        item.added_emitted |= emit_added;
        // The ids and the name are copied only for the announcement, not on
        // every delta.
        let (output_index, custom) = (item.output_index, item.custom);

        if emit_added {
            let announced = &self.tool_call_items[pos];
            let item = if custom {
                json!({
                    "id": announced.item_id,
                    "type": "custom_tool_call",
                    "call_id": announced.call_id,
                    "name": announced.name,
                    "input": "",
                })
            } else {
                json!({
                    "id": announced.item_id,
                    "type": "function_call",
                    "call_id": announced.call_id,
                    "name": announced.name,
                    "arguments": "",
                    "status": "in_progress",
                })
            };
            self.send_output_item_added(output_index, &item, tx).await?;
        }
        // Custom tool calls carry their payload as the raw `input`
        // string of the final item, not as streamed JSON arguments.
        if let Some(delta) = args_delta.filter(|_| !custom) {
            let sequence_number = self.sequence_number;
            self.sequence_number += 1;
            let event = ArgumentsDelta {
                kind: FunctionCallEvent::ARGUMENTS_DELTA,
                sequence_number,
                output_index,
                item_id: &self.tool_call_items[pos].item_id,
                delta,
            };
            Self::send_frame(
                &mut self.encoder,
                self.abandoned,
                FunctionCallEvent::ARGUMENTS_DELTA,
                &event,
                tx,
            )
            .await?;
        }
        Ok(())
    }

    /// Send `event` as one frame: its `type` on the `event:` line (`message`
    /// when it has none), its JSON on the `data:` line, serialized straight
    /// into the encoder's buffer.
    pub async fn send_event(
        &mut self,
        event: &serde_json::Value,
        tx: &SseSender,
    ) -> Result<(), String> {
        let event_type = event
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("message");
        self.send_typed(event_type, event, tx).await
    }

    /// One event of any serializable shape under `event_type`: the frame
    /// writer over `self`, [`Self::send_frame`] with this emitter's encoder.
    async fn send_typed<T: Serialize>(
        &mut self,
        event_type: &str,
        event: &T,
        tx: &SseSender,
    ) -> Result<(), String> {
        Self::send_frame(&mut self.encoder, self.abandoned, event_type, event, tx).await
    }

    /// The frame writer of this emitter, over the encoder alone: `event`
    /// serialized once into the frame under `event_type`. [`Self::send_typed`]
    /// delegates here; a caller holding a borrow of another field while the
    /// frame goes out (an arguments delta borrows its call's item id)
    /// calls it directly. Nothing is delivered once the client left
    /// (`abandoned`, see `abandon`): the frame is not built, and the caller's
    /// item state and sequence number stand as they would for a sent frame.
    async fn send_frame<T: Serialize>(
        encoder: &mut SseEncoder,
        abandoned: bool,
        event_type: &str,
        event: &T,
        tx: &SseSender,
    ) -> Result<(), String> {
        if abandoned {
            return Ok(());
        }
        let frame = encoder
            .encode_event(event_type, event)
            .map_err(|e| match e {
                SseEncodeError::Json(e) => format!("Failed to serialize event: {e}"),
                SseEncodeError::InvalidEventType => {
                    "Failed to serialize event: its type breaks the frame".to_string()
                }
            })?;
        if tx.send(Ok(frame)).await.is_err() {
            return Err("Client disconnected".to_string());
        }
        Ok(())
    }

    /// [`Self::send_frame`] for the text delta of one token, the one event
    /// that writes its own frame ([`TextDelta::write`]).
    async fn send_text_delta_frame(
        frames: &mut BytesMut,
        abandoned: bool,
        event: &TextDelta<'_>,
        tx: &SseSender,
    ) -> Result<(), String> {
        if abandoned {
            return Ok(());
        }
        // What a frame that failed midway left is not sent ahead of the next.
        frames.clear();
        event
            .write(frames)
            .map_err(|e| format!("Failed to serialize event: {e}"))?;
        if tx.send(Ok(frames.split().freeze())).await.is_err() {
            return Err("Client disconnected".to_string());
        }
        Ok(())
    }

    /// The `response.output_text.delta` event of one token, sent as
    /// [`TextDelta`]; the text accumulates as it did under the map.
    pub async fn send_text_delta(
        &mut self,
        delta: &str,
        output_index: usize,
        item_id: &str,
        content_index: usize,
        tx: &SseSender,
    ) -> Result<(), String> {
        self.accumulated_text.push_str(delta);
        let event = TextDelta {
            head: OUTPUT_TEXT_DELTA_HEAD,
            sequence_number: self.next_sequence(),
            output_index,
            item_id,
            content_index,
            delta,
        };
        Self::send_text_delta_frame(&mut self.delta_frames, self.abandoned, &event, tx).await
    }

    /// [`Self::send_text_delta`] for the streams that log a failed send
    /// instead of ending on it; `true` when the event went out.
    pub async fn send_text_delta_best_effort(
        &mut self,
        delta: &str,
        output_index: usize,
        item_id: &str,
        content_index: usize,
        tx: &SseSender,
    ) -> bool {
        match self
            .send_text_delta(delta, output_index, item_id, content_index, tx)
            .await
        {
            Ok(()) => true,
            Err(e) => {
                tracing::debug!("Failed to send event (likely client disconnect): {}", e);
                false
            }
        }
    }

    /// One fragment of a call's arguments under `kind` (a function call's
    /// event name or an MCP call's), sent as [`ArgumentsDelta`].
    pub async fn send_arguments_delta(
        &mut self,
        kind: &'static str,
        delta: &str,
        output_index: usize,
        item_id: &str,
        tx: &SseSender,
    ) -> Result<(), String> {
        let event = ArgumentsDelta {
            kind,
            sequence_number: self.next_sequence(),
            output_index,
            item_id,
            delta,
        };
        self.send_typed(kind, &event, tx).await
    }

    /// [`Self::send_arguments_delta`] for the streams that log a failed send
    /// instead of ending on it.
    pub async fn send_arguments_delta_best_effort(
        &mut self,
        kind: &'static str,
        delta: &str,
        output_index: usize,
        item_id: &str,
        tx: &SseSender,
    ) {
        if let Err(e) = self
            .send_arguments_delta(kind, delta, output_index, item_id, tx)
            .await
        {
            tracing::debug!("Failed to send event (likely client disconnect): {}", e);
        }
    }

    /// Send event and log any errors (typically client disconnect)
    ///
    /// This is a convenience method for streaming scenarios where client
    /// disconnection is expected and should be logged but not fail the operation.
    /// Returns true if sent successfully, false if client disconnected (or
    /// the stream was abandoned: nothing is delivered then).
    pub async fn send_event_best_effort(
        &mut self,
        event: &serde_json::Value,
        tx: &SseSender,
    ) -> bool {
        if self.abandoned {
            return false;
        }
        match self.send_event(event, tx).await {
            Ok(()) => true,
            Err(e) => {
                tracing::debug!("Failed to send event (likely client disconnect): {}", e);
                false
            }
        }
    }

    /// Emit the `error` event that ends the stream: the one frame every
    /// router writes for a failure of its own
    /// ([`SseEncoder::encode_responses_error`]), at the stream's next
    /// sequence number. Use this for terminal errors that abort the stream.
    pub async fn emit_error(&mut self, error_msg: &str, error_code: Option<&str>, tx: &SseSender) {
        let code = error_code.unwrap_or("internal_error");
        self.last_error = Some(json!({ "code": code, "message": error_msg }));
        let sequence_number = self.next_sequence();
        let frame = self
            .encoder
            .encode_responses_error(code, error_msg, sequence_number);
        let _ = tx.send(Ok(frame)).await;
    }

    /// The error of the last `error` event this emitter sent, if any.
    pub fn take_error(&mut self) -> Option<serde_json::Value> {
        self.last_error.take()
    }

    /// Emit the full mcp_list_tools output-item sequence.
    ///
    /// Allocates an output index, builds the tool-list JSON, then emits the four
    /// standard events (output_item.added, mcp_list_tools.in_progress,
    /// mcp_list_tools.completed, output_item.done) and marks the item complete.
    ///
    /// `server_label` is taken as an explicit parameter so callers can iterate
    /// over multiple MCP servers without mutating the emitter's own label.
    /// The list-tools item's events: `output_item.added`, `in_progress`, then
    /// `completed` with the tools or, when the server's listing failed
    /// (`error`), `failed`, and `output_item.done` with the item as the
    /// completed response carries it (the error on it); every event carries
    /// the item's id.
    pub async fn emit_mcp_list_tools_sequence(
        &mut self,
        server_label: &str,
        tools: &[mcp::ToolEntry],
        error: Option<&str>,
        tx: &SseSender,
    ) -> Result<(), String> {
        let (output_index, item_id) = self.allocate_output_index(OutputItemKind::McpListTools);

        // Build per-tool JSON items
        let tool_items = Self::tool_entries_to_json(tools).unwrap_or_else(|e| {
            warn!("Failed to serialize McpToolInfo to JSON: {e}");
            Vec::new()
        });

        // The added item (no tools yet): the typed item's shape, which has no
        // status; the item's progress is on its events.
        let item_in_progress = json!({
            "id": item_id,
            "type": "mcp_list_tools",
            "server_label": server_label,
            "tools": []
        });

        // Emit output_item.added
        self.send_output_item_added(output_index, &item_in_progress, tx)
            .await?;

        // Emit mcp_list_tools.in_progress
        let event = self.emit_mcp_list_tools_in_progress(output_index, &item_id);
        self.send_event(&event, tx).await?;

        // Emit mcp_list_tools.completed, or .failed when the listing failed
        let event = if error.is_some() {
            self.emit_mcp_list_tools_failed(output_index, &item_id)
        } else {
            self.emit_mcp_list_tools_completed(output_index, &item_id, &tool_items)
        };
        self.send_event(&event, tx).await?;

        // The done item: the tools, and the listing's failure as its error
        let mut item_done = json!({
            "id": item_id,
            "type": "mcp_list_tools",
            "server_label": server_label,
            "tools": tool_items
        });
        if let Some(error) = error {
            item_done["error"] = json!(error);
        }

        // Emit output_item.done (the emitter keeps the item)
        self.send_output_item_done(output_index, item_done, RETAIN_OUTPUT_ITEM, tx)
            .await?;

        self.complete_output_item(output_index);

        Ok(())
    }
}

/// The `x-smg-parser` value of the chat answer a Responses stream relays,
/// which the stream's own answer carries; `None` where no parser read it
/// (the Harmony path).
pub(crate) type StreamParser = Option<HeaderValue>;

/// One-shot signal used by streaming `/v1/responses` handlers to avoid
/// returning HTTP 200 until the first backend request has actually started.
/// A start brings the [`StreamParser`] of the chat answer that opened the
/// stream, so the Responses answer built at the barrier names its reader.
pub(crate) type StreamStartupSender = oneshot::Sender<Result<StreamParser, Response>>;
pub(crate) type StreamStartupReceiver = oneshot::Receiver<Result<StreamParser, Response>>;

pub(crate) fn stream_startup_channel() -> (StreamStartupSender, StreamStartupReceiver) {
    oneshot::channel()
}

#[must_use]
pub(crate) fn signal_stream_startup(
    startup: &mut Option<StreamStartupSender>,
    result: Result<StreamParser, Response>,
) -> bool {
    let Some(startup) = startup.take() else {
        return true;
    };

    startup.send(result).is_ok()
}

pub(crate) async fn await_stream_startup(
    startup: StreamStartupReceiver,
    rx: SseReceiver,
) -> Response {
    match startup.await {
        Ok(Ok(parser)) => naming_parser(build_sse_response(rx), parser),
        Ok(Err(response)) => response,
        Err(_) => error::internal_error(
            "responses_stream_startup_failed",
            "Streaming response failed before backend request startup",
        ),
    }
}

/// `response` naming the parser that read the chat answer it relays, as the
/// chat route names it in `x-smg-parser`: a client reading a Responses turn
/// tells the two readers apart as it does on chat. Unchanged where none did.
pub(crate) fn naming_parser(mut response: Response, parser: StreamParser) -> Response {
    if let Some(parser) = parser {
        response.headers_mut().insert(symphony::HEADER, parser);
    }
    response
}

/// Build a Server-Sent Events (SSE) response
///
/// Creates a Response with proper SSE headers and streaming body.
pub(crate) fn build_sse_response(rx: SseReceiver) -> Response {
    build_sse_response_from_stream(ReceiverStream::new(rx))
}

/// [`build_sse_response`] over any body stream, for a channel that goes
/// through an adapter (re-chunking, say) before it reaches the client.
#[expect(
    clippy::expect_used,
    reason = "Response::builder with static headers and valid status code is infallible"
)]
pub(crate) fn build_sse_response_from_stream<S>(stream: S) -> Response
where
    S: Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
{
    Response::builder()
        .status(StatusCode::OK)
        .header(
            CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        )
        .header("Cache-Control", HeaderValue::from_static("no-cache"))
        .header("Connection", HeaderValue::from_static("keep-alive"))
        .body(Body::from_stream(stream))
        .expect("infallible: static headers and valid status code")
}

/// What a streaming response's body holds for as long as it lives: the
/// worker load guards and the rate-limit reservation's safety net (its
/// `Drop` abandons a reservation the stream did not settle). Both go when
/// the body is dropped, at the stream's end or the client's disconnect.
pub(crate) type BodyGuards = (Option<LoadGuards>, Option<ReservationAttachment>);

/// The [`BodyGuards`] of a stream dispatched with `load_guards` under
/// `reservation`.
pub(crate) fn body_guards(
    load_guards: Option<LoadGuards>,
    reservation: Option<&Arc<SharedReservationHandle>>,
) -> BodyGuards {
    (
        load_guards,
        reservation.map(|handle| ReservationAttachment::new(Arc::clone(handle))),
    )
}

/// Attach `server_label` to an MCP tool-call JSON item.
///
/// Only sets the field when `response_format` indicates a passthrough (mcp_call)
/// type and a server label is provided, since built-in tool types
/// (web_search_call, etc.) do not carry a server label.
pub(crate) fn attach_mcp_server_label(
    item: &mut serde_json::Value,
    server_label: Option<&str>,
    response_format: Option<&ResponseFormat>,
) {
    if let (Some(label), Some(ResponseFormat::Passthrough)) = (server_label, response_format) {
        item["server_label"] = json!(label);
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/common/responses/streaming/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/common/responses/streaming/namespace_tests.rs"]
mod namespace_tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/common/responses/streaming/process_chunk_tests.rs"]
mod process_chunk_tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/common/responses/streaming/serde_error_kind_tests.rs"]
mod serde_error_kind_tests;
