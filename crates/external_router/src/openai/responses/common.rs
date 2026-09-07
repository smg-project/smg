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
pub(super) struct ChunkProcessor {
    pending: String,
}

impl ChunkProcessor {
    pub fn new() -> Self {
        Self {
            pending: String::new(),
        }
    }

    /// Append a chunk to the buffer, normalizing line endings
    pub fn push_chunk(&mut self, chunk: &[u8]) {
        let chunk_str = match std::str::from_utf8(chunk) {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(String::from_utf8_lossy(chunk).into_owned()),
        };
        if chunk_str.contains('\r') {
            self.pending.push_str(&chunk_str.replace("\r\n", "\n"));
        } else {
            self.pending.push_str(&chunk_str);
        }
    }

    /// Extract the next complete SSE block from the buffer, if available
    pub fn next_block(&mut self) -> Option<String> {
        loop {
            let pos = self.pending.find("\n\n")?;
            let block = self.pending[..pos].to_string();
            self.pending.drain(..pos + 2);

            if !block.trim().is_empty() {
                return Some(block);
            }
        }
    }

    /// Check if there's remaining content in the buffer
    pub fn has_remaining(&self) -> bool {
        !self.pending.trim().is_empty()
    }

    /// Take any remaining content from the buffer
    pub fn take_remaining(&mut self) -> String {
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
mod tests {
    use super::*;

    #[test]
    fn test_parse_sse_block_event_and_data() {
        let (event, data) =
            parse_sse_block("event: response.created\ndata: {\"type\":\"response.created\"}");
        assert_eq!(event, Some("response.created"));
        assert_eq!(data, "{\"type\":\"response.created\"}");
    }

    #[test]
    fn test_parse_sse_block_data_only() {
        let (event, data) = parse_sse_block("data: {\"type\":\"response.output_text.delta\"}");
        assert_eq!(event, None);
        assert_eq!(data, "{\"type\":\"response.output_text.delta\"}");
    }

    #[test]
    fn test_parse_sse_block_multiline_data() {
        let (event, data) = parse_sse_block("event: x\ndata: line1\ndata: line2");
        assert_eq!(event, Some("x"));
        assert_eq!(data, "line1\nline2");
    }

    #[test]
    fn test_parse_sse_block_control_only_has_empty_data() {
        // A block with no `data:` line yields empty data; all callers skip it.
        let (_event, data) = parse_sse_block(": keep-alive");
        assert!(data.is_empty());

        let (_event, data) = parse_sse_block("event: ping");
        assert!(data.is_empty());
    }

    #[test]
    fn test_parse_sse_block_empty_input() {
        let (event, data) = parse_sse_block("");
        assert_eq!(event, None);
        assert!(data.is_empty());
    }
}
