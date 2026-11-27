//! Shared SSE (Server-Sent Events) codec for encoding and decoding SSE streams.
//!
//! Consolidates duplicate SSE parsing and formatting logic from across the gateway
//! into a single, well-tested module. Two main components:
//!
//! - [`SseEncoder`]: Produces SSE-framed bytes for sending to clients.
//!   Serializes JSON directly into a `BytesMut` block via `serde_json::to_writer`
//!   (no intermediate `String`) and hands each frame out as the block's filled
//!   part, split off and frozen (no copy).
//!
//! - [`SseDecoder`]: Consumes incoming SSE byte streams from upstream workers.
//!   Uses cursor tracking to avoid per-frame memmove. `next_frame` / `flush`
//!   return `SseFrame<'static>` (owned) so callers can hold frames across
//!   subsequent decode calls. [`parse_block`] returns a borrowed frame since
//!   the caller already owns the input string — no allocation for single-line
//!   data, which is the common case.

use std::{borrow::Cow, io};

use axum::body::Body;
use bytes::{Bytes, BytesMut};
use http::{
    header::{HeaderValue, CACHE_CONTROL, CONNECTION, CONTENT_TYPE},
    StatusCode,
};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

// ============================================================================
// SseEncoder
// ============================================================================

/// The first block an encoder writes its frames into; a later block is at
/// least as large (`BytesMut` keeps the original capacity as its floor).
const FRAME_BLOCK: usize = 4096;

/// The block as serde's sink: every write is one `extend_from_slice`, the
/// shape a `Vec<u8>` gives serde. The `bytes` crate's `Writer` adaptor has no
/// `write_all` of its own, so each JSON token goes through std's default
/// `write_all` loop around a `remaining_mut` clamp (measured: more than the
/// copy it was to save, on a stream of a hundred frames).
struct BlockWriter<'a>(&'a mut BytesMut);

impl io::Write for BlockWriter<'_> {
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

/// Reusable SSE encoder. A frame is written into a `BytesMut` block and
/// leaves as the block's filled part, split off and frozen: the bytes the
/// channel sends are the bytes that were written, not a copy. The block's
/// remaining capacity serves the next frames; when it runs out a fresh block
/// is taken, or the old one reclaimed once every frame cut from it has been
/// dropped. After a frame larger than a block (a stream's terminal events
/// carry its whole answer; a reasoning item's or a tool list's done events
/// come before the answer's deltas) the next frame starts in a block of that
/// size and a block more: the frames of a terminal run each wrap the one
/// before (`output_text.done`, then `content_part.done` around it, then the
/// item, then the response) and are a little longer than it, so a run of
/// large frames takes one block each with no growth copies as long as each
/// grows by less than a block over the last, and the deltas after a large
/// frame are cut from the tail of the block the first of them took.
pub struct SseEncoder {
    buf: BytesMut,
    /// The length of the last frame written; the reservation the next frame
    /// starts with when it is above a block. The last frame and not the
    /// largest: a mark that never came down made every delta after one
    /// large frame take a fresh block of that size while the previous frame
    /// was still alive.
    last_len: usize,
}

impl SseEncoder {
    pub fn new() -> Self {
        Self {
            buf: BytesMut::with_capacity(FRAME_BLOCK),
            last_len: 0,
        }
    }

    /// Encode `data: {json}\n\n`.
    ///
    /// Serializes directly into the block via `serde_json::to_writer` over
    /// `BlockWriter` (no intermediate `String`); the frame is split off for
    /// the channel.
    pub fn encode_data<T: Serialize>(&mut self, value: &T) -> Result<Bytes, SseEncodeError> {
        self.begin();
        self.buf.extend_from_slice(b"data: ");
        serde_json::to_writer(BlockWriter(&mut self.buf), value)?;
        self.buf.extend_from_slice(b"\n\n");
        Ok(self.frame())
    }

    /// Encode `event: {type}\ndata: {json}\n\n` (Anthropic/Responses format).
    ///
    /// Returns `Err(SseEncodeError::InvalidEventType)` if `event_type` contains
    /// newline characters, which would break SSE framing.
    pub fn encode_event<T: Serialize>(
        &mut self,
        event_type: &str,
        value: &T,
    ) -> Result<Bytes, SseEncodeError> {
        if event_type.contains(['\r', '\n']) {
            return Err(SseEncodeError::InvalidEventType);
        }
        self.begin();
        self.buf.extend_from_slice(b"event: ");
        self.buf.extend_from_slice(event_type.as_bytes());
        self.buf.extend_from_slice(b"\ndata: ");
        serde_json::to_writer(BlockWriter(&mut self.buf), value)?;
        self.buf.extend_from_slice(b"\n\n");
        Ok(self.frame())
    }

    /// The start of a frame: the block emptied of what a frame whose
    /// serialization failed midway left in it (a split runs only on
    /// success), and, when the last frame was larger than a block, room for
    /// the next one before the first byte. A block nothing else holds any
    /// more (every frame cut from it dropped) is taken back whole, for
    /// nothing. Otherwise a frame of that size and a block more is reserved:
    /// free while the block's tail is that long; one block of that size when
    /// it is not, so a large frame after a large frame is written once
    /// instead of grown and copied by doublings, and the small frames after
    /// it share the block it took. The block more is for the growth from one
    /// large frame to the next: a fresh block has exactly the capacity asked
    /// for, and a frame one byte longer than the last would have doubled it
    /// and copied the frame so far; asked of a block nothing held, short of
    /// it by that growth, it reallocated the dead block instead of taking it
    /// back.
    fn begin(&mut self) {
        self.buf.clear();
        if self.last_len <= FRAME_BLOCK {
            return;
        }
        // A block nothing else holds any more is taken back whole: it held
        // the last frame, so it is at least that long, and a frame grown past
        // it grows once, as a stream's first large frame does. The headroom
        // is for a block that has to be fresh (the last frame still holds
        // this one) or a tail long enough for the last frame but not for its
        // growth.
        if self.buf.capacity() < self.last_len && self.buf.try_reclaim(self.last_len) {
            return;
        }
        self.buf.reserve(self.last_len + FRAME_BLOCK);
    }

    /// The frame written so far, split off the block and frozen; the block
    /// keeps what capacity it has left for the next frame.
    fn frame(&mut self) -> Bytes {
        let frame = self.buf.split().freeze();
        self.last_len = frame.len();
        frame
    }

    /// `data: [DONE]\n\n` — static bytes, zero allocation.
    pub fn done() -> Bytes {
        Bytes::from_static(b"data: [DONE]\n\n")
    }

    /// The `error` event that ends a failed Responses stream: the API's own
    /// event (`type`, `code`, `message`, `param`, `sequence_number`) on a
    /// `data:` line alone, nothing after it. The one frame every router
    /// writes for a failure of its own, so a client reads the same event
    /// whichever router served the stream. The frame names no event, as the
    /// gateway's error frames never have: the SDKs parse the typed event the
    /// same way with or without the name, and the older ones only without it.
    ///
    /// `code` is the failure's class (`stream_error`, `max_tool_calls_exceeded`,
    /// the upstream's own code, ...), `sequence_number` the event's place in
    /// the stream, from the stream's own counter.
    pub fn encode_responses_error(
        &mut self,
        code: &str,
        message: &str,
        sequence_number: u64,
    ) -> Bytes {
        let event = ResponsesErrorEvent {
            r#type: "error",
            code,
            message,
            param: None,
            sequence_number,
        };
        // Unreachable in practice: two strings and an integer always
        // serialize; the fallback keeps the frame well-formed should that
        // ever change.
        self.encode_data(&event)
            .unwrap_or_else(|_| Bytes::from_static(RESPONSES_ERROR_FALLBACK))
    }
}

impl Default for SseEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// The `error` event of a Responses stream, the fields in the order the API
/// writes them.
#[derive(Serialize)]
struct ResponsesErrorEvent<'a> {
    r#type: &'static str,
    code: &'a str,
    message: &'a str,
    param: Option<&'a str>,
    sequence_number: u64,
}

/// The error frame written when the event itself cannot be serialized.
const RESPONSES_ERROR_FALLBACK: &[u8] = b"data: {\"type\":\"error\",\"code\":\"internal_error\",\"message\":\"serialization failed\",\"param\":null}\n\n";

/// Errors that can occur during SSE encoding.
#[derive(Debug, thiserror::Error)]
pub enum SseEncodeError {
    /// The event type contains newline characters, which would break SSE framing.
    #[error("event_type must not contain newline characters")]
    InvalidEventType,
    /// JSON serialization failed.
    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

// ============================================================================
// SseDecoder
// ============================================================================

/// Maximum buffer size before rejecting (DoS protection).
const DEFAULT_MAX_BUFFER_SIZE: usize = 1024 * 1024; // 1 MB

/// Buffers incoming bytes and yields complete SSE frames.
///
/// - Defers UTF-8 validation to complete frames (handles multi-byte splits)
/// - Uses cursor tracking to avoid per-frame memmove
/// - `next_frame` / `flush` return `SseFrame<'static>` — callers can hold
///   frames across subsequent `next_frame()` / `compact()` calls without
///   lifetime concerns
pub struct SseDecoder {
    buf: Vec<u8>,
    consumed: usize,
    max_size: usize,
}

/// A parsed SSE frame. Fields are `Cow` so the same type can represent both
/// a zero-copy view into borrowed input ([`parse_block`]) and fully-owned
/// data released from the decoder buffer ([`SseDecoder::next_frame`]).
#[derive(Debug)]
pub struct SseFrame<'a> {
    /// The `event:` field value, if present.
    pub event_type: Option<Cow<'a, str>>,
    /// The `data:` field value.
    pub data: Cow<'a, str>,
}

impl<'a> SseFrame<'a> {
    /// Deserialize the data as JSON. Called on-demand (lazy deserialization).
    pub fn decode_data<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.data)
    }

    /// Check for `[DONE]` sentinel.
    pub fn is_done(&self) -> bool {
        self.data.as_ref() == "[DONE]"
    }

    /// Convert into an owned frame with `'static` lifetime
    pub fn into_owned(self) -> SseFrame<'static> {
        SseFrame {
            event_type: self.event_type.map(|c| Cow::Owned(c.into_owned())),
            data: Cow::Owned(self.data.into_owned()),
        }
    }
}

/// Errors that can occur during SSE decoding.
#[derive(Debug, thiserror::Error)]
pub enum SseDecodeError {
    /// Buffer exceeded the configured maximum size.
    #[error("SSE buffer overflow")]
    BufferOverflow,
    /// A complete frame contained invalid UTF-8.
    #[error("invalid UTF-8 in SSE frame: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),
    /// `flush()` called while complete frames remain in the buffer.
    /// Drain `next_frame()` to `None` before calling `flush()`.
    #[error("flush() called with complete frames still in the buffer")]
    IncompleteFlush,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::with_max_size(DEFAULT_MAX_BUFFER_SIZE)
    }

    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            buf: Vec::with_capacity(max_size.min(4096)),
            consumed: 0,
            max_size,
        }
    }

    /// Append bytes. Returns error if buffer exceeds max_size.
    ///
    /// Auto-compacts when the total allocation would exceed `max_size` but
    /// the unconsumed data alone would fit. This prevents unbounded growth
    /// when callers drain frames but forget to call `compact()`.
    pub fn push(&mut self, chunk: &[u8]) -> Result<(), SseDecodeError> {
        if self.buf.len() + chunk.len() > self.max_size {
            self.compact();
            if self.buf.len() + chunk.len() > self.max_size {
                return Err(SseDecodeError::BufferOverflow);
            }
        }
        self.buf.extend_from_slice(chunk);
        Ok(())
    }

    /// Yield the next complete SSE frame.
    ///
    /// Returns owned frames (`SseFrame<'static>`). Call in a loop until
    /// `None`, then call `compact()`.
    pub fn next_frame(&mut self) -> Option<Result<SseFrame<'static>, SseDecodeError>> {
        loop {
            let remaining = &self.buf[self.consumed..];
            let (pos, delim_len) = find_frame_boundary(remaining)?;
            let frame_bytes = &remaining[..pos];

            let frame_str = match std::str::from_utf8(frame_bytes) {
                Ok(s) => s,
                Err(e) => {
                    self.consumed += pos + delim_len;
                    return Some(Err(SseDecodeError::InvalidUtf8(e)));
                }
            };

            let owned = parse_frame(frame_str).map(SseFrame::into_owned);
            self.consumed += pos + delim_len;
            if let Some(frame) = owned {
                return Some(Ok(frame));
            }
            // Control-only block (comment, event-only, id/retry) — skip it
        }
    }

    /// Compact: shift unconsumed bytes to front. Call after draining frames.
    /// Single O(n) memmove per batch instead of per-frame.
    pub fn compact(&mut self) {
        if self.consumed > 0 {
            self.buf.drain(..self.consumed);
            self.consumed = 0;
        }
    }

    /// Flush remaining data at end of stream.
    ///
    /// Returns `Err(SseDecodeError::IncompleteFlush)` if complete frames remain
    /// in the buffer. Callers must drain `next_frame()` to `None` before calling
    /// this method.
    pub fn flush(&mut self) -> Option<Result<SseFrame<'static>, SseDecodeError>> {
        if find_frame_boundary(&self.buf[self.consumed..]).is_some() {
            return Some(Err(SseDecodeError::IncompleteFlush));
        }

        let remaining = &self.buf[self.consumed..];
        if remaining.is_empty() {
            return None;
        }
        let frame_str = match std::str::from_utf8(remaining) {
            Ok(s) => s,
            Err(e) => {
                self.consumed = self.buf.len();
                return Some(Err(SseDecodeError::InvalidUtf8(e)));
            }
        };
        // Only trim trailing newlines/CR to detect empty frames, not payload whitespace
        if frame_str.trim_end_matches(['\r', '\n']).is_empty() {
            self.consumed = self.buf.len();
            return None;
        }
        let owned = parse_frame(frame_str.trim_end_matches(['\r', '\n'])).map(SseFrame::into_owned);
        self.consumed = self.buf.len();
        owned.map(Ok)
    }

    /// Unconsumed byte count (for DoS monitoring).
    pub fn buffered_len(&self) -> usize {
        self.buf.len() - self.consumed
    }
}

impl Default for SseDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Standalone block parsing
// ============================================================================

/// Parse a single SSE block (already extracted from the stream) into an [`SseFrame`].
///
/// Use this when you already have a complete SSE block as a string (e.g. from
/// an accumulator or rewrite path) and don't need the full [`SseDecoder`] machinery.
///
/// Returns a frame borrowing from the input — zero allocation for single-line
/// `data:` (the common case). Multi-line `data:` joins into an owned `String`
/// per spec.
pub fn parse_block(block: &str) -> Option<SseFrame<'_>> {
    parse_frame(block.trim_end_matches(['\r', '\n']))
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Find a frame boundary (two consecutive end-of-line sequences) in the buffer.
///
/// The SSE spec recognizes three EOL forms: `\r\n`, `\r`, `\n`. A frame
/// boundary is any two consecutive EOLs (an empty line). This handles all
/// combinations including mixed delimiters like `\n\r\n` or `\r\n\r`.
///
/// Returns `(position, delimiter_length)` where position is the start of
/// the first EOL and delimiter_length covers both EOLs.
///
/// Uses `memchr` to jump to the first CR/LF byte, then inspects locally,
/// so performance is close to the old `memmem` approach for LF-only streams.
#[inline]
fn find_frame_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < buf.len() {
        // Fast-skip to next CR or LF using memchr (avoids scanning ASCII bytes one-by-one).
        let offset = memchr::memchr2(b'\r', b'\n', &buf[i..])?;
        i += offset;

        // Measure the first EOL sequence at position i.
        let eol1_len = eol_len_at(buf, i);
        debug_assert!(eol1_len > 0);

        // Check if a second EOL immediately follows.
        let eol2_start = i + eol1_len;
        let eol2_len = eol_len_at(buf, eol2_start);
        if eol2_len > 0 {
            return Some((i, eol1_len + eol2_len));
        }

        // Not a double-EOL; advance past this single EOL.
        i += eol1_len;
    }
    None
}

/// Length of the EOL sequence at `buf[pos..]`: 2 for `\r\n`, 1 for `\r` or `\n`, 0 if none.
#[inline]
fn eol_len_at(buf: &[u8], pos: usize) -> usize {
    match buf.get(pos) {
        Some(b'\r') => {
            if buf.get(pos + 1) == Some(&b'\n') {
                2 // \r\n
            } else {
                1 // bare \r
            }
        }
        Some(b'\n') => 1,
        _ => 0,
    }
}

/// Split a string into lines using all SSE-spec line endings: `\r\n`, `\r`, `\n`.
///
/// `str::lines()` handles `\n` and `\r\n` but not bare `\r`. The SSE spec
/// requires all three forms to be recognized as line terminators.
fn split_sse_lines(s: &str) -> impl Iterator<Item = &str> {
    let mut remaining = s;
    std::iter::from_fn(move || {
        if remaining.is_empty() {
            return None;
        }
        // Find the next \r or \n (memchr2, as the frame decoder above)
        let pos = memchr::memchr2(b'\r', b'\n', remaining.as_bytes());
        match pos {
            None => {
                let line = remaining;
                remaining = "";
                Some(line)
            }
            Some(i) => {
                let line = &remaining[..i];
                // Consume the EOL: \r\n (2), or \r / \n (1)
                if remaining.as_bytes().get(i) == Some(&b'\r')
                    && remaining.as_bytes().get(i + 1) == Some(&b'\n')
                {
                    remaining = &remaining[i + 2..];
                } else {
                    remaining = &remaining[i + 1..];
                }
                Some(line)
            }
        }
    })
}

/// Parse a complete SSE frame into event type and data.
///
/// Returns an `SseFrame` borrowing from `frame`. Single-line `data:` (the
/// common case) stays as `Cow::Borrowed` — zero allocation. Multi-line
/// `data:` joins into `Cow::Owned` per spec.
///
/// Per the SSE spec, each line is split at the first colon to determine
/// the field name and value. A single leading space after the colon is stripped.
fn parse_frame(frame: &str) -> Option<SseFrame<'_>> {
    let mut event_type: Option<&str> = None;
    let mut data: Option<Cow<'_, str>> = None;

    for line in split_sse_lines(frame) {
        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        // Per spec: split at first colon to get field name and value
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""), // field with no colon = field name, empty value
        };

        match field {
            "data" => match data.take() {
                None => data = Some(Cow::Borrowed(value)),
                Some(Cow::Borrowed(first)) => {
                    let mut joined = String::with_capacity(frame.len());
                    joined.push_str(first);
                    joined.push('\n');
                    joined.push_str(value);
                    data = Some(Cow::Owned(joined));
                }
                Some(Cow::Owned(mut joined)) => {
                    joined.push('\n');
                    joined.push_str(value);
                    data = Some(Cow::Owned(joined));
                }
            },
            "event" => {
                event_type = Some(value);
            }
            _ => {} // id, retry, etc. — silently ignored
        }
    }

    // Per SSE spec: blocks without data lines (comments, event-only, id/retry)
    // should be silently skipped, not surfaced as empty frames.
    Some(SseFrame {
        event_type: event_type.map(Cow::Borrowed),
        data: data?,
    })
}

// ============================================================================
// Bounded SSE body channel
// ============================================================================

/// Buffer size for SSE / streaming HTTP response-body channels.
///
/// Mirrors `STREAM_RELAY_BUFFER` in the gateway's HTTP router (the
/// transcription relay). The channel is deliberately **bounded**: when a client
/// reads the response slowly, [`SseSender::send`](tokio::sync::mpsc::Sender::send)
/// awaits once the buffer fills, propagating backpressure to the backend read
/// loop instead of buffering the entire response in RAM. An unbounded channel
/// would let a slow/stalled client drive the gateway into OOM (slowloris).
pub const SSE_CHANNEL_BUFFER: usize = 32;

/// Sender half of a bounded SSE body channel (see [`SSE_CHANNEL_BUFFER`]).
/// `send` is async: it awaits when the buffer is full, applying backpressure.
pub type SseSender = mpsc::Sender<Result<Bytes, io::Error>>;

/// Receiver half of a bounded SSE body channel, consumed by [`build_sse_response`].
pub type SseReceiver = mpsc::Receiver<Result<Bytes, io::Error>>;

/// Create a bounded SSE body channel (see [`SSE_CHANNEL_BUFFER`]).
pub fn sse_channel() -> (SseSender, SseReceiver) {
    mpsc::channel(SSE_CHANNEL_BUFFER)
}

// ============================================================================
// Build SSE Response
// ============================================================================

/// Build an HTTP response with SSE headers and streaming body from a bounded receiver.
#[expect(
    clippy::expect_used,
    reason = "Response::builder with static headers and valid status code is infallible"
)]
pub fn build_sse_response(rx: SseReceiver) -> axum::response::Response {
    let stream = ReceiverStream::new(rx);
    axum::response::Response::builder()
        .status(StatusCode::OK)
        .header(
            CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        )
        .header(CACHE_CONTROL, HeaderValue::from_static("no-cache"))
        .header(CONNECTION, HeaderValue::from_static("keep-alive"))
        .body(Body::from_stream(stream))
        .expect("infallible: static headers and valid status code")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "../tests/unit/sse/tests.rs"]
mod tests;
