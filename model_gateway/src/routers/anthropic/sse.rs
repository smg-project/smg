//! SSE infrastructure for Anthropic streaming responses
//!
//! Provides SSE frame parsing, event formatting, stream wrappers,
//! and the core stream consumption logic used by the streaming processor.

use std::{borrow::Cow, io};

use axum::{
    body::Body,
    http::{header, HeaderMap, StatusCode},
    response::Response,
};
use bytes::Bytes;
use futures::StreamExt;
use openai_protocol::messages::{ContentBlock, MessageDeltaUsage, StopReason, ToolUseBlock};
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use super::mcp::{IterationResult, McpToolCall};
use crate::routers::{
    common::{
        header_utils::is_smg_owned_error_code_header,
        sse::{SseDecodeError, SseDecoder, SseEncodeError, SseEncoder, SseFrame},
    },
    error::internal_error,
};

// ============================================================================
// Constants
// ============================================================================

/// Sentinel error string returned when the downstream SSE client disconnects.
pub(crate) const CLIENT_DISCONNECTED_ERROR: &str = "Client disconnected";

/// Maximum SSE buffer size (1 MB) to prevent DoS from upstream workers
/// that send data without frame delimiters.
const MAX_SSE_BUFFER_SIZE: usize = 1024 * 1024;

/// Maximum content block index accepted from an upstream worker.
/// Prevents OOM from a malicious worker sending an extremely large index.
const MAX_UPSTREAM_BLOCK_INDEX: u32 = 1024;

/// Maximum accumulated size for a single content block's text/JSON (10 MB).
/// Prevents unbounded memory growth from a stream of deltas.
const MAX_BLOCK_ACCUMULATION_SIZE: usize = 10 * 1024 * 1024;

// ============================================================================
// Public types
// ============================================================================

/// Result from consuming an upstream SSE stream.
pub(crate) struct StreamConsumeResult {
    pub iteration: IterationResult,
    pub usage: Option<MessageDeltaUsage>,
}

// ============================================================================
// SSE Response Builder
// ============================================================================

/// Build an SSE response from a status, upstream headers, and body stream.
pub(crate) fn build_sse_response(
    status: StatusCode,
    upstream_headers: HeaderMap,
    body: Body,
) -> Response {
    let mut builder = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive");

    for (key, value) in &upstream_headers {
        let key_str = key.as_str();
        if !is_smg_owned_error_code_header(key_str)
            && !matches!(
                key_str,
                "content-type"
                    | "cache-control"
                    | "connection"
                    | "transfer-encoding"
                    | "content-length"
            )
        {
            builder = builder.header(key, value);
        }
    }

    builder.body(body).unwrap_or_else(|e| {
        error!("Failed to build streaming response: {}", e);
        internal_error("response_build_failed", "Failed to build response")
    })
}

// ============================================================================
// SSE event formatting and sending
// ============================================================================

/// Format and send an SSE event through the channel.
///
/// Returns `true` if the send succeeded, `false` if the receiver was dropped.
pub(crate) async fn send_event(
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    enc: &mut SseEncoder,
    event_type: &str,
    data: &Value,
) -> bool {
    let bytes = format_sse_event(enc, event_type, data);
    tx.send(Ok(bytes)).await.is_ok()
}

/// Format a `MessageStreamEvent` as SSE bytes: `event: <type>\ndata: <json>\n\n`
fn format_sse_event(enc: &mut SseEncoder, event_type: &str, data: &Value) -> Bytes {
    enc.encode_event(event_type, data)
        .unwrap_or_else(|e| match e {
            SseEncodeError::InvalidEventType => Bytes::from_static(b"event: error\ndata: {}\n\n"),
            SseEncodeError::Json(_) => Bytes::from(format!("event: {event_type}\ndata: {{}}\n\n")),
        })
}

/// Send an SSE error event.
pub(crate) async fn send_error(
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    enc: &mut SseEncoder,
    message: &str,
) -> bool {
    let data = serde_json::json!({
        "type": "error",
        "error": {
            "type": "api_error",
            "message": message
        }
    });
    send_event(tx, enc, "error", &data).await
}

/// Emit `content_block_start` + `content_block_stop` events for an
/// `mcp_tool_result` block.
pub(crate) async fn emit_mcp_tool_result(
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    enc: &mut SseEncoder,
    call: &McpToolCall,
    global_index: &mut u32,
) -> bool {
    let index = *global_index;

    // content_block_start with mcp_tool_result
    let block_start = serde_json::json!({
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "mcp_tool_result",
            "tool_use_id": call.mcp_id,
            "is_error": call.is_error,
            "content": [{
                "type": "text",
                "text": call.result_content
            }]
        }
    });

    if !send_event(tx, enc, "content_block_start", &block_start).await {
        return false;
    }

    // content_block_stop
    let block_stop = serde_json::json!({
        "type": "content_block_stop",
        "index": index
    });

    if !send_event(tx, enc, "content_block_stop", &block_stop).await {
        return false;
    }

    *global_index += 1;
    true
}

/// Emit the final `message_delta` and `message_stop` events.
pub(crate) async fn emit_final(
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    enc: &mut SseEncoder,
    stop_reason: Option<&StopReason>,
    total_input_tokens: u32,
    total_output_tokens: u32,
) {
    let stop_reason_val = stop_reason
        .map(|r| serde_json::to_value(r).unwrap_or(Value::Null))
        .unwrap_or(Value::Null);

    let message_delta = serde_json::json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason_val,
            "stop_sequence": null
        },
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        }
    });

    if !send_event(tx, enc, "message_delta", &message_delta).await {
        debug!("Failed to send final message_delta — channel closed");
    }

    let message_stop = serde_json::json!({
        "type": "message_stop"
    });
    if !send_event(tx, enc, "message_stop", &message_stop).await {
        debug!("Failed to send message_stop — channel closed");
    }
}

// ============================================================================
// Stream consumption
// ============================================================================

/// Consume an upstream SSE byte stream, parsing events and forwarding them
/// (with transformations) to the client.
///
/// The `resolve_server_name` closure maps a tool name to its MCP server label,
/// decoupling this function from `McpToolSession`.
pub(crate) async fn consume_and_forward<F>(
    tx: &mpsc::Sender<Result<Bytes, io::Error>>,
    enc: &mut SseEncoder,
    response: reqwest::Response,
    global_index: &mut u32,
    is_first_iteration: bool,
    resolve_server_name: F,
) -> Result<StreamConsumeResult, String>
where
    F: Fn(&str) -> String,
{
    let mut stream = response.bytes_stream();
    let mut decoder = SseDecoder::with_max_size(MAX_SSE_BUFFER_SIZE);
    let mut processor = EventProcessor::new(
        tx,
        enc,
        global_index,
        is_first_iteration,
        resolve_server_name,
    );

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream read error: {e}"))?;

        decoder.push(&chunk).map_err(|e| match e {
            SseDecodeError::BufferOverflow => format!(
                "SSE buffer exceeded maximum size ({MAX_SSE_BUFFER_SIZE} bytes) — possible malformed upstream stream"
            ),
            other => format!("SSE decode error: {other}"),
        })?;

        while let Some(frame) = decoder.next_frame() {
            let frame = frame.map_err(|e| format!("Invalid UTF-8 in SSE frame: {e}"))?;
            if let Some((event_type, data)) = resolve_event(frame) {
                processor.process(&event_type, &data).await?;
            }
        }
        decoder.compact();
    }

    // Process any trailing data not terminated by a blank line.
    if let Some(frame) = decoder.flush() {
        let frame = frame.map_err(|e| match e {
            SseDecodeError::InvalidUtf8(u) => format!("Invalid UTF-8 in final SSE data: {u}"),
            other => format!("SSE decode error on flush: {other}"),
        })?;
        if let Some((event_type, data)) = resolve_event(frame) {
            processor.process(&event_type, &data).await?;
        }
    }

    Ok(processor.into_result())
}

// ============================================================================
// Internal: Block accumulator
// ============================================================================

/// Accumulator for a content block being streamed.
enum BlockAccumulator {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input_json: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    /// Passthrough for block types that don't need delta accumulation
    /// (e.g. server_tool_use, tool_search_tool_result, tool_reference).
    /// Stores the raw `content_block` JSON from content_block_start.
    Passthrough {
        content_block: Value,
    },
}

impl BlockAccumulator {
    /// Create a new accumulator for the given content block type and
    /// optional raw `content_block` JSON from `content_block_start`.
    fn for_type(block_type: &str, content_block: Option<Value>) -> Self {
        match block_type {
            "thinking" => Self::Thinking {
                thinking: String::new(),
                signature: String::new(),
            },
            "text" => Self::Text {
                text: String::new(),
            },
            // Block types that are forwarded as-is without delta accumulation
            _ => Self::Passthrough {
                content_block: content_block.unwrap_or(Value::Null),
            },
        }
    }

    /// Accumulate a streaming delta into this block.
    fn accumulate_delta(&mut self, delta: &Value) {
        let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match self {
            Self::Text { text } if delta_type == "text_delta" => {
                if let Some(t) = delta.get("text").and_then(|v| v.as_str()) {
                    if text.len() + t.len() <= MAX_BLOCK_ACCUMULATION_SIZE {
                        text.push_str(t);
                    }
                }
            }
            Self::ToolUse { input_json, .. } if delta_type == "input_json_delta" => {
                if let Some(json) = delta.get("partial_json").and_then(|v| v.as_str()) {
                    if input_json.len() + json.len() <= MAX_BLOCK_ACCUMULATION_SIZE {
                        input_json.push_str(json);
                    }
                }
            }
            Self::Thinking {
                thinking,
                signature,
            } => {
                if delta_type == "thinking_delta" {
                    if let Some(t) = delta.get("thinking").and_then(|v| v.as_str()) {
                        if thinking.len() + t.len() <= MAX_BLOCK_ACCUMULATION_SIZE {
                            thinking.push_str(t);
                        }
                    }
                } else if delta_type == "signature_delta" {
                    if let Some(s) = delta.get("signature").and_then(|v| v.as_str()) {
                        if signature.len() + s.len() <= MAX_BLOCK_ACCUMULATION_SIZE {
                            signature.push_str(s);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Finalize this accumulator into a `ContentBlock` and optional `ToolUseBlock`.
    fn finalize(&self) -> (ContentBlock, Option<ToolUseBlock>) {
        match self {
            Self::Text { text } => (
                ContentBlock::Text {
                    text: text.clone(),
                    citations: None,
                },
                None,
            ),
            Self::ToolUse {
                id,
                name,
                input_json,
            } => {
                let input: Value = serde_json::from_str(input_json).unwrap_or_else(|e| {
                    warn!(
                        error = %e,
                        json = %input_json,
                        "Failed to parse tool input JSON, using empty object"
                    );
                    Value::Object(serde_json::Map::new())
                });
                let tool_use = ToolUseBlock {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    cache_control: None,
                };
                (
                    ContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input,
                    },
                    Some(tool_use),
                )
            }
            Self::Thinking {
                thinking,
                signature,
            } => (
                ContentBlock::Thinking {
                    thinking: thinking.clone(),
                    signature: signature.clone(),
                },
                None,
            ),
            Self::Passthrough { content_block } => {
                // Deserialize the raw content_block JSON into a ContentBlock.
                // Falls back to empty text if deserialization fails.
                match serde_json::from_value::<ContentBlock>(content_block.clone()) {
                    Ok(block) => (block, None),
                    Err(e) => {
                        warn!(
                            error = %e,
                            "Failed to deserialize passthrough content block, using empty text"
                        );
                        (
                            ContentBlock::Text {
                                text: String::new(),
                                citations: None,
                            },
                            None,
                        )
                    }
                }
            }
        }
    }
}

// ============================================================================
// Internal: SSE event processor
// ============================================================================

/// Processes SSE events from the upstream worker, transforming and forwarding
/// them to the client with index remapping and tool_use → mcp_tool_use conversion.
///
/// Generic over `F` to decouple from `McpToolSession` — the closure resolves
/// tool names to MCP server labels.
struct EventProcessor<'a, F> {
    tx: &'a mpsc::Sender<Result<Bytes, io::Error>>,
    enc: &'a mut SseEncoder,
    global_index: &'a mut u32,
    index_base: u32,
    is_first_iteration: bool,
    resolve_server_name: F,
    result: IterationResult,
    usage: Option<MessageDeltaUsage>,
    upstream_blocks: Vec<BlockAccumulator>,
}

impl<'a, F> EventProcessor<'a, F>
where
    F: Fn(&str) -> String,
{
    fn new(
        tx: &'a mpsc::Sender<Result<Bytes, io::Error>>,
        enc: &'a mut SseEncoder,
        global_index: &'a mut u32,
        is_first_iteration: bool,
        resolve_server_name: F,
    ) -> Self {
        let index_base = *global_index;
        Self {
            tx,
            enc,
            global_index,
            index_base,
            is_first_iteration,
            resolve_server_name,
            result: IterationResult {
                content_blocks: Vec::new(),
                tool_use_blocks: Vec::new(),
                stop_reason: None,
            },
            usage: None,
            upstream_blocks: Vec::new(),
        }
    }

    /// Consume the accumulated result.
    fn into_result(self) -> StreamConsumeResult {
        StreamConsumeResult {
            iteration: self.result,
            usage: self.usage,
        }
    }

    /// Send an SSE event to the client, returning `Err` on disconnect.
    async fn send(&mut self, event_type: &str, data: &Value) -> Result<(), String> {
        if !send_event(self.tx, self.enc, event_type, data).await {
            return Err(CLIENT_DISCONNECTED_ERROR.into());
        }
        Ok(())
    }

    /// Process a single SSE event from the upstream worker.
    async fn process(&mut self, event_type: &str, data: &str) -> Result<(), String> {
        let mut parsed: Value =
            serde_json::from_str(data).map_err(|e| format!("Failed to parse SSE data: {e}"))?;

        match event_type {
            "message_start" => {
                if self.is_first_iteration {
                    self.send("message_start", &parsed).await?;
                }
            }
            "content_block_start" => self.handle_block_start(&mut parsed).await?,
            "content_block_delta" => self.handle_block_delta(&mut parsed).await?,
            "content_block_stop" => self.handle_block_stop(&parsed).await?,
            "message_delta" => self.handle_message_delta(&parsed),
            "message_stop" => { /* Don't forward — we emit our own at the end */ }
            "ping" => {
                self.send("ping", &serde_json::json!({"type": "ping"}))
                    .await?;
            }
            "error" => {
                self.send("error", &parsed).await?;
            }
            _ => {
                debug!(event_type = %event_type, "Forwarding unknown SSE event type");
                self.send(event_type, &parsed).await?;
            }
        }

        Ok(())
    }

    /// Handle a `content_block_start` event: transform tool_use → mcp_tool_use,
    /// remap index, and initialize the block accumulator.
    async fn handle_block_start(&mut self, parsed: &mut Value) -> Result<(), String> {
        let upstream_index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

        if upstream_index > MAX_UPSTREAM_BLOCK_INDEX {
            return Err(format!(
                "Upstream content block index {upstream_index} exceeds maximum ({MAX_UPSTREAM_BLOCK_INDEX})"
            ));
        }

        let block_type = parsed
            .get("content_block")
            .and_then(|cb| cb.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let client_index = self.index_base + upstream_index;

        while self.upstream_blocks.len() <= upstream_index as usize {
            self.upstream_blocks.push(BlockAccumulator::Text {
                text: String::new(),
            });
        }

        if block_type == "tool_use" {
            let content_block = parsed.get("content_block").cloned().unwrap_or(Value::Null);
            self.emit_mcp_tool_use_start(&content_block, upstream_index, client_index)
                .await?;
        } else {
            // Initialize accumulator before mutating parsed (block_type borrows parsed)
            let raw_block = parsed.get("content_block").cloned();
            self.upstream_blocks[upstream_index as usize] =
                BlockAccumulator::for_type(block_type, raw_block);
            parsed["index"] = Value::from(client_index);
            self.send("content_block_start", parsed).await?;
        }

        Ok(())
    }

    /// Transform an upstream `tool_use` block into `mcp_tool_use` and emit it.
    async fn emit_mcp_tool_use_start(
        &mut self,
        content_block: &Value,
        upstream_index: u32,
        client_index: u32,
    ) -> Result<(), String> {
        let id = content_block
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let name = content_block
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let mcp_id = format!("mcptoolu_{}", id.trim_start_matches("toolu_"));
        let server_name = (self.resolve_server_name)(&name);

        let event = serde_json::json!({
            "type": "content_block_start",
            "index": client_index,
            "content_block": {
                "type": "mcp_tool_use",
                "id": mcp_id,
                "name": name,
                "server_name": server_name,
                "input": {}
            }
        });
        self.send("content_block_start", &event).await?;

        self.upstream_blocks[upstream_index as usize] = BlockAccumulator::ToolUse {
            id,
            name,
            input_json: String::new(),
        };
        Ok(())
    }

    /// Handle a `content_block_delta` event: accumulate content and forward
    /// with remapped index.
    async fn handle_block_delta(&mut self, parsed: &mut Value) -> Result<(), String> {
        let upstream_index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let client_index = self.index_base + upstream_index;

        // Accumulate before mutating (we need to read delta first)
        if let Some(delta) = parsed.get("delta") {
            if let Some(block) = self.upstream_blocks.get_mut(upstream_index as usize) {
                block.accumulate_delta(delta);
            }
        }

        parsed["index"] = Value::from(client_index);
        self.send("content_block_delta", parsed).await
    }

    /// Handle a `content_block_stop` event: finalize the accumulated block
    /// and update the global index.
    async fn handle_block_stop(&mut self, parsed: &Value) -> Result<(), String> {
        let upstream_index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let client_index = self.index_base + upstream_index;

        let event = serde_json::json!({
            "type": "content_block_stop",
            "index": client_index
        });
        self.send("content_block_stop", &event).await?;

        if let Some(block) = self.upstream_blocks.get(upstream_index as usize) {
            let (content_block, tool_use) = block.finalize();
            self.result.content_blocks.push(content_block);
            if let Some(tool_use) = tool_use {
                self.result.tool_use_blocks.push(tool_use);
            }
        }

        *self.global_index = (*self.global_index).max(client_index + 1);
        Ok(())
    }

    /// Handle a `message_delta` event: capture stop_reason and usage
    /// (not forwarded — we emit our own combined delta at the end).
    fn handle_message_delta(&mut self, parsed: &Value) {
        if let Some(delta) = parsed.get("delta") {
            if let Some(stop_str) = delta.get("stop_reason").and_then(|v| v.as_str()) {
                self.result.stop_reason =
                    serde_json::from_value(Value::String(stop_str.to_string())).ok();
            }
        }
        if let Some(usage) = parsed.get("usage") {
            self.usage = serde_json::from_value(usage.clone()).ok();
        }
    }
}

// ============================================================================
// SSE frame resolution
// ============================================================================

/// Resolve a decoded [`SseFrame`] into `(event_type, data)` pair.
fn resolve_event(frame: SseFrame<'_>) -> Option<(Cow<'_, str>, Cow<'_, str>)> {
    let event_type = match frame.event_type {
        Some(e) if !e.is_empty() => e,
        _ => {
            let parsed: Value = serde_json::from_str(&frame.data).ok()?;
            Cow::Owned(parsed.get("type")?.as_str()?.to_string())
        }
    };

    if event_type.is_empty() {
        return None;
    }

    Some((event_type, frame.data))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routers::error::HEADER_X_SMG_ERROR_CODE;

    #[test]
    fn build_sse_response_drops_smg_owned_error_code() {
        let mut upstream_headers = HeaderMap::new();
        upstream_headers.insert(HEADER_X_SMG_ERROR_CODE, "forged".parse().unwrap());
        upstream_headers.insert("x-upstream-id", "worker-a".parse().unwrap());

        let response = build_sse_response(
            StatusCode::SERVICE_UNAVAILABLE,
            upstream_headers,
            Body::empty(),
        );

        assert!(response.headers().get(HEADER_X_SMG_ERROR_CODE).is_none());
        assert_eq!(response.headers().get("x-upstream-id").unwrap(), "worker-a");
    }

    /// Decode `bytes` through the shared `SseDecoder` and resolve each frame
    /// to `(event_type, data)` the way `consume_and_forward` does.
    fn decode_events(bytes: &[u8]) -> Vec<(String, String)> {
        let mut decoder = SseDecoder::new();
        decoder.push(bytes).unwrap();
        let mut out = Vec::new();
        while let Some(frame) = decoder.next_frame() {
            if let Some((event_type, data)) = resolve_event(frame.unwrap()) {
                out.push((event_type.into_owned(), data.into_owned()));
            }
        }
        out
    }

    #[test]
    fn test_resolve_event_basic() {
        let frame = SseFrame {
            event_type: Some(Cow::Borrowed("message_start")),
            data: Cow::Borrowed("{\"type\":\"message_start\"}"),
        };
        let (event_type, data) = resolve_event(frame).unwrap();
        assert_eq!(event_type.as_ref(), "message_start");
        assert_eq!(data, "{\"type\":\"message_start\"}");
    }

    #[test]
    fn test_resolve_event_no_event_type_infers() {
        // No `event:` line -> infer the type from the payload's "type" field.
        let frame = SseFrame {
            event_type: None,
            data: Cow::Borrowed("{\"type\":\"ping\"}"),
        };
        let (event_type, _data) = resolve_event(frame).unwrap();
        assert_eq!(event_type.as_ref(), "ping");
    }

    #[test]
    fn test_resolve_event_uninferable_is_none() {
        // No event type and no "type" field in the payload -> skipped.
        let frame = SseFrame {
            event_type: None,
            data: Cow::Borrowed("{\"foo\":1}"),
        };
        assert!(resolve_event(frame).is_none());
    }

    #[test]
    fn test_decode_events_basic() {
        let events = decode_events(
            b"event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: ping\ndata: {\"type\":\"ping\"}\n\n",
        );
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].0, "message_start");
        assert_eq!(events[1].0, "ping");
    }

    #[test]
    fn test_decode_events_content_block() {
        let events = decode_events(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "content_block_start");
        let parsed: Value = serde_json::from_str(&events[0].1).unwrap();
        assert_eq!(parsed["index"], 0);
    }

    #[test]
    fn test_decode_events_infers_event_type() {
        // A `data:`-only frame (no `event:` line) infers its type from the payload.
        let events = decode_events(b"data: {\"type\":\"ping\"}\n\n");
        assert_eq!(
            events,
            vec![("ping".to_string(), "{\"type\":\"ping\"}".to_string())]
        );
    }

    #[test]
    fn test_format_sse_event() {
        let data = serde_json::json!({"type": "ping"});
        let mut enc = SseEncoder::new();
        let bytes = format_sse_event(&mut enc, "ping", &data);
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.starts_with("event: ping\n"));
        assert!(text.contains("data: "));
        assert!(text.ends_with("\n\n"));
    }

    #[test]
    fn test_decode_events_split_across_chunks() {
        // A multi-byte-safe decoder must reassemble frames split mid-stream.
        let mut decoder = SseDecoder::new();
        decoder
            .push(b"event: content_block_delta\ndata: {\"type\":\"content_block_de")
            .unwrap();
        assert!(decoder.next_frame().is_none()); // incomplete
        decoder
            .push(b"lta\",\"index\":1,\"delta\":{\"partial_json\":\"{}\"}}\n\n")
            .unwrap();
        let frame = decoder.next_frame().unwrap().unwrap();
        let (event_type, data) = resolve_event(frame).unwrap();
        assert_eq!(event_type.as_ref(), "content_block_delta");
        let parsed: Value = serde_json::from_str(&data).unwrap();
        assert_eq!(parsed["index"], 1);
    }
}
