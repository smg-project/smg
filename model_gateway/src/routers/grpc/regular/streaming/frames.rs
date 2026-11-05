//! The SSE frames of the per-token chunks: the chat and completion chunk
//! heads serialized once per stream and copied in front of each chunk, the
//! choice that opens a chat stream, and the delta helpers the stream state
//! machines write through. Moved out of `streaming.rs` as is; the streams
//! keep writing through them by name.

use std::io;

use bytes::BytesMut;
use openai_protocol::{
    chat::{
        ChatMessageDelta, ChatMessageDeltaRef, ChatStreamChoice, ChatStreamChoiceRef,
        DEFAULT_SERVICE_TIER,
    },
    common::{ToolCallDelta, ToolChoice, Usage},
    completion::CompletionStreamChoice,
};
use serde::Serialize;
use serde_json::json;
use tracing::error;

use crate::routers::grpc::spec::ChatResponseSpec;

/// The frame buffer as serde's sink: every write is one `extend_from_slice`,
/// the shape a `Vec<u8>` gives serde. The `bytes` writer adapter instead slices
/// each write to the buffer's remaining room before it appends, which costs the
/// length bookkeeping on every JSON token and turns the one-byte constant tokens
/// into libc calls: about 7 % of a 128-delta stream request's user instructions.
pub(super) struct FrameSink<'a>(pub(super) &'a mut BytesMut);

impl io::Write for FrameSink<'_> {
    #[inline]
    fn write(&mut self, src: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(src);
        Ok(src.len())
    }

    #[inline]
    fn write_all(&mut self, src: &[u8]) -> io::Result<()> {
        self.0.extend_from_slice(src);
        Ok(())
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Buffered chat chunks are flushed to the response channel at this size even
/// while the engine keeps more ready, so one frame stays a bounded burst.
pub(super) const CHUNK_FLUSH_BYTES: usize = 16 * 1024;

/// The SSE frame of the per-token chat chunks of one stream.
///
/// Every chunk of a stream repeats the same `id`, `object`, `created`,
/// `model` and `system_fingerprint`; only its choice and usage differ. The
/// head up to and including `"choices":[` is serialized once per stream and
/// copied in front of each chunk, so a chunk costs the serialization of its
/// choice and usage only. The bytes are exactly those of a serialized
/// [`super::ChatChunkWithUsage`] (the chunk-level `usage: null` placeholder included).
pub(super) struct ChatChunkFrame {
    head: Vec<u8>,
}

impl ChatChunkFrame {
    pub(super) fn new(
        request_id: &str,
        model: &str,
        created: u64,
        system_fingerprint: Option<&str>,
    ) -> Self {
        let mut head = Vec::with_capacity(192);
        head.extend_from_slice(b"data: {\"id\":");
        push_json(&mut head, request_id);
        head.extend_from_slice(b",\"object\":\"chat.completion.chunk\",\"created\":");
        push_json(&mut head, &created);
        head.extend_from_slice(b",\"model\":");
        push_json(&mut head, model);
        head.extend_from_slice(b",\"service_tier\":");
        push_json(&mut head, DEFAULT_SERVICE_TIER);
        if let Some(system_fingerprint) = system_fingerprint {
            head.extend_from_slice(b",\"system_fingerprint\":");
            push_json(&mut head, system_fingerprint);
        }
        head.extend_from_slice(b",\"choices\":[");
        Self { head }
    }

    /// Append `data: {...}\n\n` for `choice` (and `usage`, or the `null`
    /// placeholder when `emit_usage_null`) to `buffer`.
    pub(super) fn write_chunk<C: Serialize>(
        &self,
        buffer: &mut BytesMut,
        choice: &C,
        usage: Option<&Usage>,
        emit_usage_null: bool,
    ) {
        let start = buffer.len();
        buffer.extend_from_slice(&self.head);
        let serialized = serde_json::to_writer(FrameSink(buffer), choice);
        Self::close_chunk(buffer, start, serialized, usage, emit_usage_null);
    }

    /// [`Self::write_chunk`] for a bare text delta ([`text_delta`]): the
    /// choice is `{"index":N,"delta":{"<field>":"<text>"},"logprobs":null,"finish_reason":null}`,
    /// so its constant text is written as literals and serde writes the index
    /// and the text alone, instead of walking the five keys and the two nulls
    /// of the struct on every token. The bytes are those [`Self::write_chunk`]
    /// writes for the same choice (pinned by test).
    pub(super) fn write_text_delta(
        &self,
        buffer: &mut BytesMut,
        index: u32,
        field: TextDelta,
        text: &str,
        usage: Option<&Usage>,
        emit_usage_null: bool,
    ) {
        let start = buffer.len();
        buffer.extend_from_slice(&self.head);
        buffer.extend_from_slice(b"{\"index\":");
        push_json(FrameSink(buffer), &index);
        buffer.extend_from_slice(field.opener());
        push_json(FrameSink(buffer), text);
        buffer.extend_from_slice(b"},\"logprobs\":null,\"finish_reason\":null}");
        Self::close_chunk(buffer, start, Ok(()), usage, emit_usage_null);
    }

    /// The chunk's end after its choice: `]`, the usage or its `null`
    /// placeholder, `}` and the event's blank line; a choice that failed to
    /// serialize leaves an error event in its place.
    fn close_chunk(
        buffer: &mut BytesMut,
        start: usize,
        mut serialized: serde_json::Result<()>,
        usage: Option<&Usage>,
        emit_usage_null: bool,
    ) {
        if serialized.is_ok() {
            buffer.extend_from_slice(b"]");
            if let Some(usage) = usage {
                buffer.extend_from_slice(b",\"usage\":");
                serialized = serde_json::to_writer(FrameSink(buffer), usage);
            } else if emit_usage_null {
                buffer.extend_from_slice(b",\"usage\":null");
            }
        }
        match serialized {
            Ok(()) => buffer.extend_from_slice(b"}"),
            Err(e) => {
                error!("Failed to serialize SSE chunk: {}", e);
                buffer.truncate(start);
                buffer.extend_from_slice(b"data: {\"error\":\"serialization_failed\"}");
            }
        }
        buffer.extend_from_slice(b"\n\n");
    }
}

/// The one text field a bare per-token chat delta carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TextDelta {
    Content,
    Reasoning,
}

impl TextDelta {
    /// The bytes between a choice's index and its text: the delta's opening
    /// and the field's key, as serde writes them for the struct.
    fn opener(self) -> &'static [u8] {
        match self {
            TextDelta::Content => b",\"delta\":{\"content\":",
            TextDelta::Reasoning => b",\"delta\":{\"reasoning_content\":",
        }
    }
}

/// The field and the text of `choice` when it is a bare per-token delta: one
/// text field and nothing else (no role, no tool calls, no logprobs, not
/// finished, no matched stop), the shape [`ChatChunkFrame::write_text_delta`]
/// writes; `None` for every other choice, which stays serde's. Both structs
/// are matched field by field with no rest pattern, so a field added to
/// either is a compile error here until this read says what it does.
pub(super) fn text_delta<'a>(choice: &'a ChatStreamChoiceRef<'_>) -> Option<(TextDelta, &'a str)> {
    let ChatStreamChoiceRef {
        index: _,
        delta,
        logprobs: None,
        finish_reason: None,
        matched_stop: None,
    } = choice
    else {
        return None;
    };
    match delta {
        ChatMessageDeltaRef {
            role: None,
            content: Some(text),
            tool_calls: None,
            reasoning_content: None,
        } => Some((TextDelta::Content, text.as_ref())),
        ChatMessageDeltaRef {
            role: None,
            content: None,
            tool_calls: None,
            reasoning_content: Some(text),
        } => Some((TextDelta::Reasoning, text.as_ref())),
        _ => None,
    }
}

/// Serializing a string or an integer into memory cannot fail.
fn push_json<W: io::Write, T: Serialize + ?Sized>(writer: W, value: &T) {
    let _ = serde_json::to_writer(writer, value);
}

/// The SSE frame of the per-token chunks of one completion stream.
///
/// Every chunk of a stream repeats the same `id`, `object`, `created`,
/// `model` and `system_fingerprint`; only its choice differs, and it sits
/// between them. The head up to and including `"choices":[` and the tail
/// from `]` to the end of the event are serialized once per stream and
/// copied around each chunk's choice, so a chunk costs the serialization of
/// its choice only. The bytes are exactly those of a serialized
/// [`openai_protocol::completion::CompletionStreamResponse`] without `usage`, which a chunk of this
/// stream never carries.
pub(super) struct CompletionChunkFrame {
    head: Vec<u8>,
    tail: Vec<u8>,
}

impl CompletionChunkFrame {
    pub(super) fn new(
        request_id: &str,
        model: &str,
        created: u64,
        system_fingerprint: Option<&str>,
    ) -> Self {
        let mut head = Vec::with_capacity(128);
        head.extend_from_slice(b"data: {\"id\":");
        push_json(&mut head, request_id);
        head.extend_from_slice(b",\"object\":\"text_completion\",\"created\":");
        push_json(&mut head, &created);
        head.extend_from_slice(b",\"choices\":[");
        let mut tail = Vec::with_capacity(96);
        tail.extend_from_slice(b"],\"model\":");
        push_json(&mut tail, model);
        if let Some(system_fingerprint) = system_fingerprint {
            tail.extend_from_slice(b",\"system_fingerprint\":");
            push_json(&mut tail, system_fingerprint);
        }
        tail.extend_from_slice(b"}\n\n");
        Self { head, tail }
    }

    /// Append `data: {...}\n\n` for `choice` to `buffer`.
    pub(super) fn write_chunk(&self, buffer: &mut BytesMut, choice: &CompletionStreamChoice) {
        let start = buffer.len();
        buffer.extend_from_slice(&self.head);
        match serde_json::to_writer(FrameSink(buffer), choice) {
            Ok(()) => buffer.extend_from_slice(&self.tail),
            Err(e) => {
                error!("Failed to serialize completion SSE chunk: {}", e);
                buffer.truncate(start);
                buffer.extend_from_slice(b"data: ");
                let error_msg = json!({
                    "error": {
                        "message": format!("Failed to serialize completion chunk: {e}"),
                        "type": "internal_error"
                    }
                })
                .to_string();
                buffer.extend_from_slice(error_msg.as_bytes());
                buffer.extend_from_slice(b"\n\n");
            }
        }
    }
}

/// The stream's first choice: the role, once, with the delta the OpenAI API
/// opens a stream with. A stream that may answer with text opens with
/// `{"role":"assistant","content":"","refusal":null}`; one the request forces
/// into a tool call opens with `{"role":"assistant","content":null}`. Later
/// deltas carry only what changed ([`assistant_choice`]). Serialized by the
/// same frame as a [`ChatStreamChoice`], so the bytes around the delta are
/// those of every other chunk.
#[derive(Serialize)]
pub(super) struct OpeningChoice {
    index: u32,
    delta: OpeningDelta,
    logprobs: (),
    finish_reason: (),
}

#[derive(Serialize)]
struct OpeningDelta {
    role: &'static str,
    content: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refusal: Option<()>,
}

pub(super) fn opening_choice(index: u32, opens_tool_call: bool) -> OpeningChoice {
    OpeningChoice {
        index,
        delta: if opens_tool_call {
            OpeningDelta {
                role: "assistant",
                content: None,
                refusal: None,
            }
        } else {
            OpeningDelta {
                role: "assistant",
                content: Some(""),
                refusal: Some(()),
            }
        },
        logprobs: (),
        finish_reason: (),
    }
}

/// Whether the request leaves the model no choice but a tool call (a named
/// function, `required`, or allowed tools in `required` mode, with tools to
/// call), so the stream's first delta is a tool call's, not text's.
pub(super) fn forces_tool_call(spec: &ChatResponseSpec) -> bool {
    spec.tools.as_ref().is_some_and(|tools| !tools.is_empty())
        && spec
            .tool_choice
            .as_ref()
            .is_some_and(ToolChoice::forces_call)
}

/// A streamed choice carrying whichever of `content`, `reasoning_content` and
/// `tool_calls` the caller sets; the role is not repeated, the first chunk
/// ([`opening_choice`]) carried it.
pub(super) fn assistant_choice(
    index: u32,
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<ToolCallDelta>>,
) -> ChatStreamChoice {
    ChatStreamChoice {
        index,
        delta: ChatMessageDelta {
            role: None,
            content,
            tool_calls,
            reasoning_content,
        },
        logprobs: None,
        finish_reason: None,
        matched_stop: None,
    }
}
