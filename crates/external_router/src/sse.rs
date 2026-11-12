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
        // Find the next \r or \n
        let pos = remaining.find(['\r', '\n']);
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
mod tests {
    use super::*;

    // --- SseEncoder tests ---

    #[test]
    fn test_encode_data_simple() {
        let mut enc = SseEncoder::new();
        let val = serde_json::json!({"type": "ping"});
        let bytes = enc.encode_data(&val).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.starts_with("data: "));
        assert!(text.ends_with("\n\n"));
        assert!(text.contains("\"type\":\"ping\""));
    }

    #[test]
    fn test_encode_event_with_type() {
        let mut enc = SseEncoder::new();
        let val = serde_json::json!({"index": 0, "delta": "hi"});
        let bytes = enc.encode_event("content_block_delta", &val).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.starts_with("event: content_block_delta\n"));
        assert!(text.contains("data: "));
        assert!(text.ends_with("\n\n"));
    }

    #[test]
    fn test_encode_done_sentinel() {
        let bytes = SseEncoder::done();
        assert_eq!(&bytes[..], b"data: [DONE]\n\n");
    }

    /// Twenty small frames are cut from the encoder's first block, one after
    /// another, without a copy or a block of their own: each lies inside the
    /// first frame's block, and a frame already handed out keeps its bytes
    /// while the frames after it are written.
    #[test]
    fn small_frames_are_cut_from_one_block_and_keep_their_bytes() {
        let mut enc = SseEncoder::new();
        let frames: Vec<Bytes> = (0..20)
            .map(|i| enc.encode_data(&serde_json::json!({"i": i})).unwrap())
            .collect();
        let block = frames[0].as_ptr() as usize;
        for (i, frame) in frames.iter().enumerate() {
            let start = frame.as_ptr() as usize;
            assert!(
                start >= block && start + frame.len() <= block + FRAME_BLOCK,
                "frame {i} lies outside the first block"
            );
            assert_eq!(&frame[..], format!("data: {{\"i\":{i}}}\n\n").as_bytes());
        }
    }

    /// A frame larger than a block is whole, middle included, and the
    /// frames after it are written as before.
    #[test]
    fn a_frame_larger_than_a_block_is_whole() {
        let mut enc = SseEncoder::new();
        let long = "x".repeat(3 * FRAME_BLOCK);
        let big = enc.encode_data(&serde_json::json!({"text": long})).unwrap();
        assert_eq!(
            &big[..],
            format!("data: {{\"text\":\"{long}\"}}\n\n").as_bytes()
        );
        let next = enc.encode_data(&serde_json::json!({"i": 1})).unwrap();
        assert_eq!(&next[..], b"data: {\"i\":1}\n\n");
    }

    /// A value whose serialization fails after it has written part of a map.
    struct FailsMidway;

    impl Serialize for FailsMidway {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            use serde::ser::{Error, SerializeMap};
            let mut map = serializer.serialize_map(None)?;
            map.serialize_entry("partial", "bytes")?;
            Err(S::Error::custom("boom"))
        }
    }

    /// A frame whose serialization fails partway leaves nothing in the next
    /// frame: the block is emptied at the start of every encode, and a split
    /// runs only on success.
    #[test]
    fn a_failed_frame_leaves_nothing_in_the_next() {
        let mut enc = SseEncoder::new();
        assert!(enc.encode_data(&FailsMidway).is_err());
        let next = enc.encode_data(&serde_json::json!({"i": 1})).unwrap();
        assert_eq!(&next[..], b"data: {\"i\":1}\n\n");
    }

    /// A large frame sets the next frame's reservation and a small frame
    /// clears it: a second large frame is whole, so is the small frame after
    /// it, and after that small frame the encoder reserves nothing.
    #[test]
    fn a_large_frame_sets_the_next_frames_reservation() {
        let mut enc = SseEncoder::new();
        let long = "y".repeat(5 * FRAME_BLOCK);
        let first = enc.encode_data(&serde_json::json!({"text": long})).unwrap();
        assert_eq!(enc.last_len, first.len());
        let second = enc.encode_data(&serde_json::json!({"text": long})).unwrap();
        assert_eq!(&second[..], &first[..]);
        let next = enc.encode_data(&serde_json::json!({"i": 2})).unwrap();
        assert_eq!(&next[..], b"data: {\"i\":2}\n\n");
        assert_eq!(enc.last_len, next.len());
        assert!(enc.last_len <= FRAME_BLOCK);
    }

    #[test]
    fn test_encode_data_matches_format_pattern() {
        // Verify output matches `format!("data: {json}\n\n")` for compatibility.
        let mut enc = SseEncoder::new();
        let val = serde_json::json!({"type":"message_start","message":{"id":"msg_1"}});
        let bytes = enc.encode_data(&val).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();

        let json_str = serde_json::to_string(&val).unwrap();
        let expected = format!("data: {json_str}\n\n");
        assert_eq!(text, expected);
    }

    #[test]
    fn test_encode_event_matches_format_pattern() {
        // Verify output matches `format!("event: {type}\ndata: {json}\n\n")`.
        let mut enc = SseEncoder::new();
        let val = serde_json::json!({"type":"content_block_start","index":0});
        let bytes = enc.encode_event("content_block_start", &val).unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();

        let json_str = serde_json::to_string(&val).unwrap();
        let expected = format!("event: content_block_start\ndata: {json_str}\n\n");
        assert_eq!(text, expected);
    }

    /// The error event of a failed Responses stream: the API's fields in its
    /// order, the message escaped as JSON, on a data line alone, no event
    /// name and no sentinel.
    #[test]
    fn a_responses_error_frame_is_the_typed_event_on_a_data_line() {
        let mut enc = SseEncoder::new();
        let frame =
            enc.encode_responses_error("stream_error", "Stream error: he said \"bye\"\n", 7);
        assert_eq!(
            frame,
            "data: {\"type\":\"error\",\"code\":\"stream_error\",\"message\":\"Stream error: he \
             said \\\"bye\\\"\\n\",\"param\":null,\"sequence_number\":7}\n\n"
                .as_bytes()
        );
    }

    // --- SseDecoder tests ---

    #[test]
    fn test_decode_single_data_frame() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: {\"type\":\"ping\"}\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.event_type, None);
        assert_eq!(frame.data, "{\"type\":\"ping\"}");
    }

    #[test]
    fn test_decode_event_and_data() {
        let mut dec = SseDecoder::new();
        dec.push(b"event: message_start\ndata: {\"type\":\"message_start\"}\n\n")
            .unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.event_type.as_deref(), Some("message_start"));
        assert_eq!(frame.data, "{\"type\":\"message_start\"}");
    }

    #[test]
    fn test_decode_multiple_frames() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: first\n\ndata: second\n\n").unwrap();

        let f1 = dec.next_frame().unwrap().unwrap();
        assert_eq!(f1.data, "first");

        let f2 = dec.next_frame().unwrap().unwrap();
        assert_eq!(f2.data, "second");

        assert!(dec.next_frame().is_none());
    }

    #[test]
    fn test_decode_split_across_chunks() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: hel").unwrap();
        assert!(dec.next_frame().is_none()); // incomplete
        dec.push(b"lo\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data, "hello");
    }

    #[test]
    fn test_decode_multiline_data() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: line1\ndata: line2\ndata: line3\n\n")
            .unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data, "line1\nline2\nline3");
    }

    #[test]
    fn test_decode_crlf_frame_boundary() {
        // Full CRLF-delimited frame: \r\n\r\n
        let mut dec = SseDecoder::new();
        dec.push(b"event: ping\r\ndata: {}\r\n\r\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.event_type.as_deref(), Some("ping"));
        assert_eq!(frame.data, "{}");
    }

    #[test]
    fn test_decode_crlf_within_frame() {
        // Mixed: CRLF lines but LF-only frame boundary
        let mut dec = SseDecoder::new();
        dec.push(b"event: ping\r\ndata: {}\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.event_type.as_deref(), Some("ping"));
        assert_eq!(frame.data, "{}");
    }

    #[test]
    fn test_decode_cr_cr_frame_boundary() {
        // Standalone CR delimiters: \r\r
        let mut dec = SseDecoder::new();
        dec.push(b"data: hello\r\r").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data, "hello");
    }

    #[test]
    fn test_decode_mixed_eol_frame_boundary() {
        // Mixed: LF then CRLF (\n\r\n)
        let mut dec = SseDecoder::new();
        dec.push(b"data: mixed\n\r\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data, "mixed");
    }

    #[test]
    fn test_decode_comments_ignored() {
        let mut dec = SseDecoder::new();
        dec.push(b": this is a comment\ndata: hello\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data, "hello");
        assert_eq!(frame.event_type, None);
    }

    #[test]
    fn test_decode_event_only_block_skipped() {
        // Per SSE spec, blocks without data lines are control-only and should be skipped
        let mut dec = SseDecoder::new();
        dec.push(b"event: ping\n\n").unwrap();
        assert!(dec.next_frame().is_none());
    }

    #[test]
    fn test_decode_empty_data_value() {
        // A block with an explicit `data:` line (empty value) IS a valid frame
        let mut dec = SseDecoder::new();
        dec.push(b"event: ping\ndata:\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.event_type.as_deref(), Some("ping"));
        assert_eq!(frame.data, "");
    }

    #[test]
    fn test_decode_done_sentinel() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: [DONE]\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert!(frame.is_done());
    }

    #[test]
    fn test_decode_json_deserialization() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: {\"value\":42}\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        let val: serde_json::Value = frame.decode_data().unwrap();
        assert_eq!(val["value"], 42);
    }

    #[test]
    fn test_decode_buffer_overflow() {
        let mut dec = SseDecoder::with_max_size(16);
        let result = dec.push(b"data: this is way too long for the buffer\n\n");
        assert!(matches!(result, Err(SseDecodeError::BufferOverflow)));
    }

    #[test]
    fn test_decode_buffer_overflow_auto_compacts() {
        // Buffer with max_size=32. Push data, drain frames, push more.
        // Without auto-compact, the second push would overflow because
        // consumed bytes still occupy the Vec.
        let mut dec = SseDecoder::with_max_size(32);
        dec.push(b"data: first\n\n").unwrap(); // 13 bytes
        let f = dec.next_frame().unwrap().unwrap();
        assert_eq!(f.data, "first");
        // Without auto-compact: buf.len()=13, consumed=13, unconsumed=0
        // New push of 14 bytes: buf.len()+chunk.len() = 27 <= 32, OK
        // But if we pushed 20 bytes: 13+20 = 33 > 32, auto-compact needed
        dec.push(b"data: second-longer\n\n").unwrap(); // 21 bytes, 13+21=34>32, auto-compacts
        let f = dec.next_frame().unwrap().unwrap();
        assert_eq!(f.data, "second-longer");
    }

    #[test]
    fn test_decode_compact() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: first\n\ndata: second\n\npartial").unwrap();

        let f1 = dec.next_frame().unwrap().unwrap();
        assert_eq!(f1.data, "first");

        let f2 = dec.next_frame().unwrap().unwrap();
        assert_eq!(f2.data, "second");

        assert!(dec.next_frame().is_none());
        assert_eq!(dec.buffered_len(), 7); // "partial"

        dec.compact();
        assert_eq!(dec.consumed, 0);
        assert_eq!(dec.buf, b"partial");
    }

    #[test]
    fn test_decode_flush() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: final").unwrap();
        assert!(dec.next_frame().is_none());
        let frame = dec.flush().unwrap().unwrap();
        assert_eq!(frame.data, "final");
    }

    #[test]
    fn test_decode_flush_empty() {
        let mut dec = SseDecoder::new();
        assert!(dec.flush().is_none());
    }

    #[test]
    fn test_decode_flush_whitespace_only() {
        let mut dec = SseDecoder::new();
        dec.push(b"  \n  ").unwrap();
        // No data: lines, so this is a control-only block — flush returns None
        assert!(dec.flush().is_none());
    }

    #[test]
    fn test_decode_flush_newlines_only() {
        let mut dec = SseDecoder::new();
        dec.push(b"\n\r\n").unwrap();
        // \n\r\n is a frame boundary (LF + CRLF) with empty content.
        // next_frame skips empty frames, so it returns None.
        assert!(dec.next_frame().is_none());
        // After draining, flush also returns None — no meaningful data.
        assert!(dec.flush().is_none());
    }

    #[test]
    fn test_decode_flush_incomplete() {
        let mut dec = SseDecoder::new();
        dec.push(b"data: a\n\ndata: b\n\n").unwrap();
        // Complete frames remain — flush should reject
        let result = dec.flush().unwrap();
        assert!(matches!(result, Err(SseDecodeError::IncompleteFlush)));
    }

    #[test]
    fn test_decode_unknown_fields_ignored() {
        let mut dec = SseDecoder::new();
        dec.push(b"id: 123\nretry: 5000\ndata: hello\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data, "hello");
        assert_eq!(frame.event_type, None);
    }

    #[test]
    fn test_decode_leading_space_strip() {
        let mut dec = SseDecoder::new();
        // Per spec: strip exactly one leading space after the colon
        dec.push(b"data:  two spaces\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        // Strip one space, leaving " two spaces"
        assert_eq!(frame.data, " two spaces");
    }

    #[test]
    fn test_decode_no_space_after_colon() {
        let mut dec = SseDecoder::new();
        dec.push(b"data:nospace\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data, "nospace");
    }

    #[test]
    fn test_decode_field_without_colon() {
        // Per spec: a line with no colon is treated as field name with empty value
        let mut dec = SseDecoder::new();
        dec.push(b"data\n\n").unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data, "");
    }

    #[test]
    fn test_decode_multi_byte_utf8_split() {
        let mut dec = SseDecoder::new();
        // "data: 日本" in UTF-8
        let full = "data: 日本\n\n";
        let bytes = full.as_bytes();
        // Split in the middle of a multi-byte character
        let mid = 8; // "data: 日" is "data: " (6) + 日 (3 bytes) = 9, split at 8
        dec.push(&bytes[..mid]).unwrap();
        assert!(dec.next_frame().is_none()); // incomplete frame
        dec.push(&bytes[mid..]).unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data, "日本");
    }

    #[test]
    fn test_decode_empty_lines_between_fields() {
        let mut dec = SseDecoder::new();
        dec.push(b"event: test\n\ndata: value\n\n").unwrap();
        // First block (event: test) has no data line — skipped per SSE spec
        // Second block (data: value) is returned as the first frame
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.data, "value");
        assert_eq!(frame.event_type, None);
    }

    // --- Roundtrip tests ---

    #[test]
    fn test_roundtrip_data_only() {
        let mut enc = SseEncoder::new();
        let original = serde_json::json!({"type":"ping","id":42});
        let bytes = enc.encode_data(&original).unwrap();

        let mut dec = SseDecoder::new();
        dec.push(&bytes).unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        let decoded: serde_json::Value = frame.decode_data().unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_event_with_type() {
        let mut enc = SseEncoder::new();
        let original = serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}});
        let bytes = enc.encode_event("content_block_delta", &original).unwrap();

        let mut dec = SseDecoder::new();
        dec.push(&bytes).unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert_eq!(frame.event_type.as_deref(), Some("content_block_delta"));
        let decoded: serde_json::Value = frame.decode_data().unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_done() {
        let bytes = SseEncoder::done();
        let mut dec = SseDecoder::new();
        dec.push(&bytes).unwrap();
        let frame = dec.next_frame().unwrap().unwrap();
        assert!(frame.is_done());
    }

    #[test]
    fn test_roundtrip_multiple_events() {
        let mut enc = SseEncoder::new();
        let events = vec![
            ("message_start", serde_json::json!({"type":"message_start"})),
            (
                "content_block_start",
                serde_json::json!({"type":"content_block_start","index":0}),
            ),
            (
                "content_block_delta",
                serde_json::json!({"type":"content_block_delta","index":0,"delta":{"text":"hi"}}),
            ),
            (
                "content_block_stop",
                serde_json::json!({"type":"content_block_stop","index":0}),
            ),
            ("message_stop", serde_json::json!({"type":"message_stop"})),
        ];

        let mut all_bytes = Vec::new();
        for (event_type, val) in &events {
            let bytes = enc.encode_event(event_type, val).unwrap();
            all_bytes.extend_from_slice(&bytes);
        }

        let mut dec = SseDecoder::new();
        dec.push(&all_bytes).unwrap();

        for (expected_type, expected_val) in &events {
            let frame = dec.next_frame().unwrap().unwrap();
            assert_eq!(frame.event_type.as_deref(), Some(*expected_type));
            let decoded: serde_json::Value = frame.decode_data().unwrap();
            assert_eq!(&decoded, expected_val);
        }

        assert!(dec.next_frame().is_none());
    }

    // --- parse_frame unit tests ---

    #[test]
    fn test_parse_frame_basic() {
        let frame =
            parse_frame("event: message_start\ndata: {\"type\":\"message_start\"}").unwrap();
        assert_eq!(frame.event_type.as_deref(), Some("message_start"));
        assert_eq!(frame.data, "{\"type\":\"message_start\"}");
    }

    #[test]
    fn test_parse_frame_data_only() {
        let frame = parse_frame("data: hello").unwrap();
        assert_eq!(frame.event_type, None);
        assert_eq!(frame.data, "hello");
    }

    #[test]
    fn test_parse_frame_empty() {
        assert!(parse_frame("").is_none());
    }

    #[test]
    fn test_parse_frame_multiline_data() {
        let frame = parse_frame("data: line1\ndata: line2").unwrap();
        assert_eq!(frame.data, "line1\nline2");
    }

    #[test]
    fn test_parse_frame_empty_data_lines() {
        // Empty data: lines must produce newlines per SSE spec
        let frame = parse_frame("data:\ndata: hello").unwrap();
        assert_eq!(frame.data, "\nhello");

        // Two consecutive empty data: lines
        let frame = parse_frame("data:\ndata:").unwrap();
        assert_eq!(frame.data, "\n");
    }

    #[test]
    fn test_parse_frame_comment_only() {
        // Comment-only blocks have no data line — should return None
        assert!(parse_frame(": heartbeat").is_none());
    }

    #[test]
    fn test_parse_frame_field_without_colon() {
        // "data" with no colon = field name "data", empty value — still has a data line
        let frame = parse_frame("data").unwrap();
        assert_eq!(frame.data, "");
    }

    #[test]
    fn test_parse_frame_bare_cr_lines() {
        // Lines separated by bare \r — str::lines() would fail here
        let frame = parse_frame("event: ping\rdata: {}").unwrap();
        assert_eq!(frame.event_type.as_deref(), Some("ping"));
        assert_eq!(frame.data, "{}");
    }

    // --- split_sse_lines tests ---

    #[test]
    fn test_split_sse_lines_lf() {
        let lines: Vec<_> = split_sse_lines("a\nb\nc").collect();
        assert_eq!(lines, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_sse_lines_crlf() {
        let lines: Vec<_> = split_sse_lines("a\r\nb\r\nc").collect();
        assert_eq!(lines, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_sse_lines_bare_cr() {
        let lines: Vec<_> = split_sse_lines("a\rb\rc").collect();
        assert_eq!(lines, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_sse_lines_mixed() {
        let lines: Vec<_> = split_sse_lines("a\nb\rc\r\nd").collect();
        assert_eq!(lines, vec!["a", "b", "c", "d"]);
    }

    // --- find_frame_boundary tests ---

    #[test]
    fn test_find_frame_boundary_lf_lf() {
        assert_eq!(find_frame_boundary(b"data: x\n\n"), Some((7, 2)));
    }

    #[test]
    fn test_find_frame_boundary_crlf_crlf() {
        assert_eq!(find_frame_boundary(b"data: x\r\n\r\n"), Some((7, 4)));
    }

    #[test]
    fn test_find_frame_boundary_cr_cr() {
        assert_eq!(find_frame_boundary(b"data: x\r\r"), Some((7, 2)));
    }

    #[test]
    fn test_find_frame_boundary_lf_crlf() {
        // Mixed: LF then CRLF
        assert_eq!(find_frame_boundary(b"data: x\n\r\n"), Some((7, 3)));
    }

    #[test]
    fn test_find_frame_boundary_crlf_lf() {
        // Mixed: CRLF then LF
        assert_eq!(find_frame_boundary(b"data: x\r\n\n"), Some((7, 3)));
    }

    #[test]
    fn test_find_frame_boundary_crlf_cr() {
        // Mixed: CRLF then CR
        assert_eq!(find_frame_boundary(b"data: x\r\n\r"), Some((7, 3)));
    }

    #[test]
    fn test_find_frame_boundary_cr_lf_not_crlf_pair() {
        // \r followed by \n is a single CRLF, not two EOLs.
        // Need another EOL after to form a boundary.
        assert_eq!(find_frame_boundary(b"data: x\r\n"), None);
    }

    #[test]
    fn test_find_frame_boundary_none() {
        assert_eq!(find_frame_boundary(b"data: x\n"), None);
    }

    #[test]
    fn test_find_frame_boundary_earliest_wins() {
        // \n\n at pos 7, \r\r later — \n\n should win
        assert_eq!(find_frame_boundary(b"data: x\n\ndata: y\r\r"), Some((7, 2)));
    }

    #[test]
    fn test_find_frame_boundary_empty() {
        assert_eq!(find_frame_boundary(b""), None);
    }
}
