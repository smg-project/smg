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
mod tests {
    use super::*;
    use crate::test_support::counting_alloc;

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

    /// Feed `bytes` split at every position in `cuts` and collect all blocks
    /// plus whatever remains at end of stream.
    fn run_chunks(bytes: &[u8], cuts: &[usize]) -> (Vec<String>, String) {
        let mut cp = ChunkProcessor::new();
        let mut blocks = Vec::new();
        let mut start = 0;
        for &cut in cuts.iter().chain(std::iter::once(&bytes.len())) {
            cp.push_chunk(&bytes[start..cut]);
            while let Some(b) = cp.next_block() {
                blocks.push(b.to_string());
            }
            start = cut;
        }
        let rest = if cp.has_remaining() {
            cp.take_remaining()
        } else {
            String::new()
        };
        (blocks, rest)
    }

    #[test]
    fn test_chunk_processor_multibyte_split_at_every_byte() {
        // "化" is 3 bytes and "😀" is 4: split the stream at every possible
        // position, the decoded block must never contain U+FFFD.
        let text = "热迁移是虚拟化的标志😀能力";
        let event = format!("event: response.output_text.done\ndata: {{\"text\":\"{text}\"}}\n\n");
        let bytes = event.as_bytes();
        for cut in 1..bytes.len() {
            let (blocks, rest) = run_chunks(bytes, &[cut]);
            assert_eq!(blocks.len(), 1, "cut at {cut}");
            assert!(
                !blocks[0].contains('\u{FFFD}'),
                "cut at {cut}: {}",
                blocks[0]
            );
            assert!(blocks[0].contains(text), "cut at {cut}");
            assert!(rest.is_empty(), "cut at {cut}");
        }
    }

    #[test]
    fn test_chunk_processor_one_byte_chunks() {
        let event = "data: {\"delta\":\"云计算架构\"}\n\ndata: {\"text\":\"、与私有子网（\"}\n\n";
        let bytes = event.as_bytes();
        let cuts: Vec<usize> = (1..bytes.len()).collect();
        let (blocks, rest) = run_chunks(bytes, &cuts);
        assert_eq!(
            blocks,
            vec![
                "data: {\"delta\":\"云计算架构\"}".to_string(),
                "data: {\"text\":\"、与私有子网（\"}".to_string()
            ]
        );
        assert!(rest.is_empty());
    }

    #[test]
    fn test_chunk_processor_crlf_split_across_chunks() {
        let bytes = b"event: a\r\ndata: 1\r\n\r\nevent: b\r\ndata: 2\r\n\r\n";
        for cut in 1..bytes.len() {
            let (blocks, rest) = run_chunks(bytes, &[cut]);
            assert_eq!(
                blocks,
                vec![
                    "event: a\ndata: 1".to_string(),
                    "event: b\ndata: 2".to_string()
                ],
                "cut at {cut}"
            );
            assert!(rest.is_empty(), "cut at {cut}");
        }
    }

    #[test]
    fn test_chunk_processor_invalid_bytes_still_replaced() {
        // A byte that can never start a UTF-8 sequence is replaced, the rest kept.
        let (blocks, _) = run_chunks(b"data: a\xffb\n\n", &[]);
        assert_eq!(blocks, vec!["data: a\u{FFFD}b".to_string()]);
    }

    #[test]
    fn test_chunk_processor_truncated_tail_at_end_of_stream() {
        // Stream ends in the middle of a character: the remainder is still returned.
        let bytes = "data: 化".as_bytes();
        let (blocks, rest) = run_chunks(&bytes[..bytes.len() - 1], &[]);
        assert!(blocks.is_empty());
        assert_eq!(rest, "data: \u{FFFD}");
    }

    #[test]
    fn test_parse_sse_block_empty_input() {
        let (event, data) = parse_sse_block("");
        assert_eq!(event, None);
        assert!(data.is_empty());
    }

    /// Blocks handed out are dropped from the buffer at the end of the
    /// stream too: `take_remaining` returns the unread tail alone (what the
    /// record stores), not the blocks already relayed.
    #[test]
    fn test_chunk_processor_take_remaining_skips_handed_out_blocks() {
        let (blocks, rest) = run_chunks(b"data: 1\n\ndata: 2", &[]);
        assert_eq!(blocks, vec!["data: 1".to_string()]);
        assert_eq!(rest, "data: 2");
    }

    /// The blocks handed out from one chunk are dropped when the next chunk
    /// arrives, and a block split across the two chunks comes out whole.
    #[test]
    fn a_block_split_across_chunks_after_handed_out_blocks_comes_out_whole() {
        let mut cp = ChunkProcessor::new();
        cp.push_chunk(b"event: a\ndata: 1\n\nevent: b\ndata: 2");
        assert_eq!(cp.next_block(), Some("event: a\ndata: 1"));
        assert_eq!(cp.next_block(), None);
        cp.push_chunk(b"\n\nevent: c\ndata: 3\n\n");
        assert_eq!(cp.pending, "event: b\ndata: 2\n\nevent: c\ndata: 3\n\n");
        assert_eq!(cp.next_block(), Some("event: b\ndata: 2"));
        assert_eq!(cp.next_block(), Some("event: c\ndata: 3"));
        assert_eq!(cp.next_block(), None);
        assert!(!cp.has_remaining());
        assert_eq!(cp.take_remaining(), "");
    }

    /// Draining one pushed chunk of ten blocks hands out ten borrowed blocks
    /// and allocates nothing: no copy per block, no move of the rest per
    /// block (the buffer is compacted once, at the next chunk). The count is
    /// printed for the Numbers table.
    #[test]
    #[expect(
        clippy::print_stderr,
        reason = "the count a pin reads is printed for the Numbers table"
    )]
    fn draining_a_chunk_of_ten_blocks_allocates_nothing() {
        let chunk: String = (0..10)
            .map(|i| {
                format!(
                    "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"sequence_number\":{i},\"delta\":\"a dozen words of answer text that a model streams per step \"}}\n\n"
                )
            })
            .collect();
        let mut cp = ChunkProcessor::new();
        cp.push_chunk(chunk.as_bytes());

        let before = counting_alloc::calls();
        let mut blocks = 0;
        let mut bytes = 0;
        while let Some(block) = cp.next_block() {
            blocks += 1;
            bytes += block.len();
        }
        let calls = counting_alloc::calls() - before;

        eprintln!("draining a chunk of ten blocks: {calls} allocation calls");
        assert_eq!(blocks, 10);
        assert_eq!(
            bytes,
            chunk.len() - 10 * 2,
            "every block, without its blank line"
        );
        assert!(!cp.has_remaining());
        assert_eq!(
            calls, 0,
            "allocation calls while draining a chunk of ten blocks"
        );
    }
}
