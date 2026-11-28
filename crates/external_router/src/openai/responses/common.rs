//! Common SSE parsing and processing utilities for OpenAI responses
//!
//! This module contains shared helpers used by both streaming and accumulator modules.

use std::borrow::Cow;

use serde_json::Value;

use crate::sse::parse_block;

/// Extract output_index from a JSON value
#[inline]
pub(crate) fn extract_output_index(value: &Value) -> Option<usize> {
    value.get("output_index")?.as_u64().map(|v| v as usize)
}

/// Get event type from event name or parsed JSON, returning a reference to avoid allocation
#[inline]
pub(crate) fn get_event_type<'a>(event_name: Option<&'a str>, parsed: &'a Value) -> &'a str {
    event_name
        .or_else(|| parsed.get("type").and_then(|v| v.as_str()))
        .unwrap_or("")
}

/// Processes incoming byte chunks into complete SSE blocks.
/// Handles buffering of partial chunks and CRLF normalization.
///
/// Network chunk boundaries are arbitrary: a multi-byte UTF-8 character (or a
/// `\r\n` pair) may be split across two chunks. Such a split must not be
/// decoded on its own, otherwise a large event (e.g. `response.output_text.done`
/// carrying the full text) gets U+FFFD in place of the split character, both
/// in what is forwarded to the client and in what is stored. An incomplete
/// UTF-8 sequence at the end of a chunk is therefore carried over to the next
/// chunk, and a trailing `\r` is held back until we know whether `\n` follows.
pub(super) struct ChunkProcessor {
    pending: String,
    /// Bytes of an incomplete UTF-8 sequence at the end of the last chunk.
    carry: Vec<u8>,
    /// The last decoded text ended with `\r` that has not been emitted yet.
    pending_cr: bool,
    /// Where the next block starts in `pending`: the blocks before it were
    /// handed out already and are dropped at the next chunk.
    read: usize,
}

impl ChunkProcessor {
    pub fn new() -> Self {
        Self {
            pending: String::new(),
            carry: Vec::new(),
            pending_cr: false,
            read: 0,
        }
    }

    /// Drop the blocks handed out so far: one move of the unread tail per
    /// pushed chunk, not one per block.
    fn compact(&mut self) {
        if self.read > 0 {
            self.pending.drain(..self.read);
            self.read = 0;
        }
    }

    /// Append a chunk to the buffer, normalizing line endings
    pub fn push_chunk(&mut self, chunk: &[u8]) {
        let joined;
        let mut rest: &[u8] = if self.carry.is_empty() {
            chunk
        } else {
            let mut buf = std::mem::take(&mut self.carry);
            buf.extend_from_slice(chunk);
            joined = buf;
            &joined
        };
        loop {
            match std::str::from_utf8(rest) {
                Ok(s) => {
                    self.push_text(s);
                    break;
                }
                Err(e) => {
                    let (valid, after) = rest.split_at(e.valid_up_to());
                    // `valid_up_to` guarantees this prefix is valid UTF-8.
                    self.push_text(std::str::from_utf8(valid).unwrap_or_default());
                    match e.error_len() {
                        // Incomplete sequence at the end: wait for the next chunk.
                        None => {
                            self.carry = after.to_vec();
                            break;
                        }
                        // Genuinely invalid bytes: replace them, keep going.
                        Some(n) => {
                            self.push_text("\u{FFFD}");
                            rest = &after[n..];
                        }
                    }
                }
            }
        }
    }

    /// Append decoded text, normalizing `\r\n` to `\n` across chunk boundaries.
    fn push_text(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        self.compact();
        let mut s = Cow::Borrowed(s);
        if std::mem::take(&mut self.pending_cr) {
            if s.starts_with('\n') {
                // `\r` + `\n` split across chunks: the `\r` is dropped.
            } else {
                self.pending.push('\r');
            }
        }
        if s.ends_with('\r') {
            self.pending_cr = true;
            s = Cow::Owned(s[..s.len() - 1].to_string());
        }
        if s.contains('\r') {
            self.pending.push_str(&s.replace("\r\n", "\n"));
        } else {
            self.pending.push_str(&s);
        }
    }

    /// End of stream: whatever is still held back becomes plain text
    /// (an incomplete UTF-8 tail is replaced, a lone `\r` is kept).
    fn flush_held(&mut self) {
        if !self.carry.is_empty() {
            let tail = String::from_utf8_lossy(&std::mem::take(&mut self.carry)).into_owned();
            self.push_text(&tail);
        }
        if std::mem::take(&mut self.pending_cr) {
            self.pending.push('\r');
        }
    }

    /// The next complete SSE block in the buffer, if any, borrowed from it:
    /// no copy of the block and no move of the rest. The borrow ends at the
    /// next call; the block is dropped from the buffer at the next chunk.
    pub fn next_block(&mut self) -> Option<&str> {
        loop {
            let rest = &self.pending[self.read..];
            let pos = memchr::memmem::find(rest.as_bytes(), b"\n\n")?;
            let start = self.read;
            self.read = start + pos + 2;
            let block = &self.pending[start..start + pos];
            if !block.trim().is_empty() {
                return Some(block);
            }
        }
    }

    /// Check if there's remaining content in the buffer
    pub fn has_remaining(&self) -> bool {
        !self.pending[self.read..].trim().is_empty() || !self.carry.is_empty()
    }

    /// Take any remaining content from the buffer
    pub fn take_remaining(&mut self) -> String {
        self.flush_held();
        self.compact();
        std::mem::take(&mut self.pending)
    }
}

/// Parse an SSE block into event name and data.
///
/// Delegates field parsing to the shared [`parse_block`] codec. Returns
/// borrowed strings for the common single-line `data:` case (zero
/// allocation); multi-line `data:` joins into an owned `String`.
///
/// All callers treat an empty `data` as "skip", so the shared codec's
/// behavior of dropping data-less control blocks (returning no frame) is
/// equivalent to the previous `(event_name, "")` result.
pub(super) fn parse_sse_block(block: &str) -> (Option<&str>, Cow<'_, str>) {
    match parse_block(block) {
        Some(frame) => {
            let event_name = frame.event_type.and_then(|e| match e {
                Cow::Borrowed(s) => Some(s.trim()),
                Cow::Owned(_) => {
                    debug_assert!(false, "parse_block returned Cow::Owned for event_type");
                    None
                }
            });
            (event_name, frame.data)
        }
        None => (None, Cow::Borrowed("")),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/openai/responses/common/tests.rs"]
mod tests;
