//! DeepSeek-V4.1 reasoning parser.
//!
//! Mirrors the reasoning half of vLLM's `deepseek_v4` parser engine
//! (`vllm/parser/deepseek_v4.py`, inherited by `deepseek_v41.py`, which only
//! swaps in the spaced tool marker): a small state machine driven by the
//! first marker found in the text.
//!
//! | state      | `<think>`       | `</think>`         | `<｜DSML｜ calls>`        |
//! |------------|-----------------|--------------------|---------------------------|
//! | content    | enter reasoning | absorbed (dropped) | enter tool block (kept)   |
//! | reasoning  | absorbed        | end reasoning      | end reasoning, tool block |
//! | tool block | verbatim        | verbatim           | verbatim                  |
//!
//! The tool block and everything after it pass through as normal text for
//! the tool parser; vLLM's tool states never return to content or reasoning,
//! so think markers after a tool block are not interpreted. Only the full
//! spaced marker counts: a bare `<｜DSML｜` (SGLang's cut point) or V4's
//! unspaced spelling is ordinary text, and so is a block-less
//! `<｜DSML｜ invoke ...>` (vLLM's content-state invoke transition only
//! concerns the tool parser downstream). Text is emitted verbatim — no
//! whitespace trimming. Streaming holds back a buffer suffix that could still
//! grow into a marker of the current state and emits everything before it.

use crate::traits::{ParseError, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE};

/// The spaced DSML tool-call block opener; inside reasoning it ends the
/// reasoning block implicitly and is kept at the start of the normal text.
pub const TOOL_BLOCK_START: &str = "<｜DSML｜ calls>";
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";

/// The longest marker, in bytes: a held-back suffix is never longer.
const MAX_MARKER_LEN: usize = TOOL_BLOCK_START.len();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Marker {
    ThinkStart,
    ThinkEnd,
    ToolBlockStart,
}

impl Marker {
    const ALL: [Marker; 3] = [Marker::ThinkStart, Marker::ThinkEnd, Marker::ToolBlockStart];

    fn text(self) -> &'static str {
        match self {
            Marker::ThinkStart => THINK_START,
            Marker::ThinkEnd => THINK_END,
            Marker::ToolBlockStart => TOOL_BLOCK_START,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Content,
    Reasoning,
    ToolBlock,
}

/// Reasoning parser for DeepSeek-V4.1 (`<think>`/`</think>` plus the spaced
/// DSML tool block as an implicit end of reasoning).
#[derive(Debug, Clone)]
pub struct DeepSeekV41Parser {
    state: State,
    /// Streaming holdback: text that may still grow into a marker.
    buffer: String,
}

impl DeepSeekV41Parser {
    /// A chat-mode parser: reasoning starts only when the gateway arms it
    /// (`mark_reasoning_started`) or the output opens with `<think>`.
    pub fn new() -> Self {
        Self {
            state: State::Content,
            buffer: String::new(),
        }
    }

    /// Apply every complete marker in the buffer left to right, then emit
    /// what cannot still become a marker (all of it when `at_end`).
    fn drain(&mut self, at_end: bool) -> ParserResult {
        let mut reasoning = String::new();
        let mut normal = String::new();
        loop {
            let hit = match self.state {
                State::ToolBlock => None,
                State::Content | State::Reasoning => earliest_marker(&self.buffer),
            };
            let Some((index, marker)) = hit else {
                let keep = if at_end || self.state == State::ToolBlock {
                    0
                } else {
                    partial_marker_suffix(&self.buffer)
                };
                let emit_end = self.buffer.len() - keep;
                self.emit(&mut reasoning, &mut normal, emit_end);
                break;
            };
            self.emit(&mut reasoning, &mut normal, index);
            // The transition table from the module docs: the next state, and
            // whether the marker is consumed (dropped) or kept as normal text.
            let (next_state, consume_marker) = match (self.state, marker) {
                (State::Content, Marker::ThinkStart) => (State::Reasoning, true),
                (State::Reasoning, Marker::ThinkEnd) => (State::Content, true),
                // A bare `</think>` in content and a duplicate `<think>` inside
                // reasoning are absorbed.
                (State::Content, Marker::ThinkEnd) | (State::Reasoning, Marker::ThinkStart) => {
                    (self.state, true)
                }
                // Kept: the tool parser consumes the block from its opener.
                (_, Marker::ToolBlockStart) => (State::ToolBlock, false),
                // Verbatim inside a tool block (never reached: that state is
                // drained above without scanning).
                (State::ToolBlock, Marker::ThinkStart | Marker::ThinkEnd) => {
                    (State::ToolBlock, false)
                }
            };
            self.state = next_state;
            if consume_marker {
                self.buffer.drain(..marker.text().len());
            }
        }
        ParserResult::new(normal, reasoning)
    }

    /// Move the first `len` bytes of the buffer into the sink for the
    /// current state.
    fn emit(&mut self, reasoning: &mut String, normal: &mut String, len: usize) {
        let sink = match self.state {
            State::Reasoning => reasoning,
            State::Content | State::ToolBlock => normal,
        };
        sink.push_str(&self.buffer[..len]);
        self.buffer.drain(..len);
    }
}

impl Default for DeepSeekV41Parser {
    fn default() -> Self {
        Self::new()
    }
}

/// The earliest complete marker in `text` and its byte offset.
fn earliest_marker(text: &str) -> Option<(usize, Marker)> {
    Marker::ALL
        .iter()
        .filter_map(|marker| text.find(marker.text()).map(|index| (index, *marker)))
        .min_by_key(|(index, _)| *index)
}

/// Length in bytes of the longest suffix of `text` that is a proper prefix
/// of some marker — the part streaming must hold back.
fn partial_marker_suffix(text: &str) -> usize {
    let earliest_start = text.len().saturating_sub(MAX_MARKER_LEN - 1);
    text.char_indices()
        .map(|(start, _)| start)
        .filter(|start| *start >= earliest_start)
        .find(|start| {
            let suffix = &text[*start..];
            Marker::ALL.iter().any(|marker| {
                marker.text().len() > suffix.len() && marker.text().starts_with(suffix)
            })
        })
        .map_or(0, |start| text.len() - start)
}

impl ReasoningParser for DeepSeekV41Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        if text.len() > DEFAULT_MAX_BUFFER_SIZE {
            return Err(ParseError::BufferOverflow(text.len()));
        }
        self.buffer.clear();
        self.buffer.push_str(text);
        Ok(self.drain(true))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        let buffered = self.buffer.len() + text.len();
        if buffered > DEFAULT_MAX_BUFFER_SIZE {
            return Err(ParseError::BufferOverflow(buffered));
        }
        self.buffer.push_str(text);
        Ok(self.drain(false))
    }

    fn flush(&mut self) -> Result<ParserResult, ParseError> {
        // Whatever is still held back can no longer become a marker: emit it
        // to the current state's side. The buffer is empty afterwards, so a
        // repeated call returns nothing.
        Ok(self.drain(true))
    }

    fn reset(&mut self) {
        self.state = State::Content;
        self.buffer.clear();
    }

    fn model_type(&self) -> &str {
        "deepseek_v41"
    }

    fn is_in_reasoning(&self) -> bool {
        self.state == State::Reasoning
    }

    fn mark_reasoning_started(&mut self) {
        self.state = State::Reasoning;
    }

    /// The prefill consumed `<think>`; nothing to record, because a `<think>`
    /// that still appears inside reasoning is absorbed by the state machine.
    fn mark_think_start_stripped(&mut self) {}
}
