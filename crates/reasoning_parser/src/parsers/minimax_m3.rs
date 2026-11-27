// MiniMax M3 specific reasoning parser.
//
// MiniMax M3 can emit stray '</mm:think>' delimiters before normal content.
// Its markers may also be decoded across arbitrary streaming chunk boundaries.

use crate::traits::{ParseError, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE};

const THINK_START: &str = "<mm:think>";
const THINK_END: &str = "</mm:think>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamingState {
    BeforeReasoning,
    InReasoning,
    AfterReasoning,
    Content,
}

/// MiniMax M3 reasoning parser.
///
/// Unlike the common reasoning parser, this parser deliberately distinguishes
/// a leading/boundary close marker from a close marker embedded in visible
/// content. It also retains the longest marker-prefix suffix between streaming
/// calls, preventing split delimiters from leaking into either output channel.
pub struct MinimaxM3Parser {
    state: StreamingState,
    buffer: String,
    /// Whether close markers can still be treated as leading framing noise.
    /// Whitespace does not end this phase, but substantive normal text does.
    allow_leading_closers: bool,
    reasoning_has_text: bool,
}

impl MinimaxM3Parser {
    /// Create a new MiniMax M3 reasoning parser.
    pub fn new() -> Self {
        Self {
            state: StreamingState::BeforeReasoning,
            buffer: String::new(),
            allow_leading_closers: true,
            reasoning_has_text: false,
        }
    }

    /// Byte length of the longest trailing slice of 'text' that is a strict
    /// prefix of 'marker'.
    fn partial_marker_suffix_len(text: &str, marker: &str) -> usize {
        let max_len = text.len().min(marker.len().saturating_sub(1));
        for len in (1..=max_len).rev() {
            let start = text.len() - len;
            if text.is_char_boundary(start) && marker.starts_with(&text[start..]) {
                return len;
            }
        }
        0
    }

    fn leading_whitespace_len(text: &str) -> usize {
        text.char_indices()
            .find_map(|(index, ch)| (!ch.is_whitespace()).then_some(index))
            .unwrap_or(text.len())
    }

    /// Strip every close delimiter at the beginning of a content segment while
    /// preserving surrounding whitespace. MiniMax M3 may duplicate this
    /// delimiter, so stripping exactly one is insufficient.
    fn strip_repeated_leading_closers(text: &str) -> String {
        let mut remainder = text;
        let mut whitespace = String::new();
        let mut stripped = false;

        loop {
            let whitespace_len = Self::leading_whitespace_len(remainder);
            let after_whitespace = &remainder[whitespace_len..];
            let Some(after_marker) = after_whitespace.strip_prefix(THINK_END) else {
                break;
            };

            whitespace.push_str(&remainder[..whitespace_len]);
            remainder = after_marker;
            stripped = true;
        }

        if stripped {
            whitespace.push_str(remainder);
            whitespace
        } else {
            text.to_string()
        }
    }

    fn parse_complete_outside_reasoning(text: &str) -> ParserResult {
        let text = Self::strip_repeated_leading_closers(text);
        let Some(start_index) = text.find(THINK_START) else {
            return ParserResult::normal(text);
        };

        let content_before = &text[..start_index];
        let after_start = &text[start_index + THINK_START.len()..];
        let Some(end_index) = after_start.find(THINK_END) else {
            return ParserResult::new(content_before.to_string(), after_start.to_string());
        };

        let reasoning = &after_start[..end_index];
        let content_after =
            Self::strip_repeated_leading_closers(&after_start[end_index + THINK_END.len()..]);
        ParserResult::new(
            format!("{content_before}{content_after}"),
            reasoning.to_string(),
        )
    }

    fn parse_complete_inside_reasoning(text: &str) -> ParserResult {
        // A prefilled start marker is normally absent from generated output,
        // but consume it if the model repeats it.
        let text = text.strip_prefix(THINK_START).unwrap_or(text);
        let Some(end_index) = text.find(THINK_END) else {
            return ParserResult::reasoning(text.to_string());
        };

        let reasoning = &text[..end_index];
        let content = Self::strip_repeated_leading_closers(&text[end_index + THINK_END.len()..]);
        ParserResult::new(content, reasoning.to_string())
    }

    fn emit_prefix(buffer: &mut String, len: usize, output: &mut String) {
        output.push_str(&buffer[..len]);
        buffer.drain(..len);
    }

    fn parse_stream_buffer(&mut self) -> ParserResult {
        let mut normal = String::new();
        let mut reasoning = String::new();

        loop {
            if self.buffer.is_empty() {
                break;
            }

            match self.state {
                StreamingState::BeforeReasoning => {
                    if self.allow_leading_closers {
                        // Whitespace is visible, but does not prevent a following
                        // close delimiter from being recognized as leading noise.
                        let whitespace_len = Self::leading_whitespace_len(&self.buffer);
                        if whitespace_len > 0 {
                            Self::emit_prefix(&mut self.buffer, whitespace_len, &mut normal);
                            if self.buffer.is_empty() {
                                break;
                            }
                        }

                        if THINK_END.starts_with(&self.buffer) && self.buffer != THINK_END {
                            break;
                        }
                        if self.buffer.starts_with(THINK_END) {
                            self.buffer.drain(..THINK_END.len());
                            continue;
                        }
                    }

                    if THINK_START.starts_with(&self.buffer) && self.buffer != THINK_START {
                        break;
                    }
                    if self.buffer.starts_with(THINK_START) {
                        self.buffer.drain(..THINK_START.len());
                        self.state = StreamingState::InReasoning;
                        self.reasoning_has_text = false;
                        continue;
                    }

                    if let Some(start_index) = self.buffer.find(THINK_START) {
                        if start_index > 0 {
                            let has_text = self.buffer[..start_index]
                                .chars()
                                .any(|ch| !ch.is_whitespace());
                            Self::emit_prefix(&mut self.buffer, start_index, &mut normal);
                            if has_text {
                                self.allow_leading_closers = false;
                            }
                        }
                        self.buffer.drain(..THINK_START.len());
                        self.state = StreamingState::InReasoning;
                        self.reasoning_has_text = false;
                        continue;
                    }

                    // No complete start marker is present. Emit everything except
                    // a suffix that could become one on the next call.
                    let holdback = Self::partial_marker_suffix_len(&self.buffer, THINK_START);
                    let emit_len = self.buffer.len() - holdback;
                    if emit_len > 0 {
                        let has_text = self.buffer[..emit_len]
                            .chars()
                            .any(|ch| !ch.is_whitespace());
                        Self::emit_prefix(&mut self.buffer, emit_len, &mut normal);
                        if has_text {
                            self.allow_leading_closers = false;
                        }
                    }
                    break;
                }
                StreamingState::InReasoning => {
                    // If thinking was opened by the prompt, tolerate the model
                    // redundantly generating the opening marker as well.
                    if !self.reasoning_has_text {
                        if THINK_START.starts_with(&self.buffer) && self.buffer != THINK_START {
                            break;
                        }
                        if self.buffer.starts_with(THINK_START) {
                            self.buffer.drain(..THINK_START.len());
                            continue;
                        }
                    }

                    if let Some(end_index) = self.buffer.find(THINK_END) {
                        if end_index > 0 {
                            Self::emit_prefix(&mut self.buffer, end_index, &mut reasoning);
                            self.reasoning_has_text = true;
                        }
                        self.buffer.drain(..THINK_END.len());
                        self.state = StreamingState::AfterReasoning;
                        continue;
                    }

                    let holdback = Self::partial_marker_suffix_len(&self.buffer, THINK_END);
                    let emit_len = self.buffer.len() - holdback;
                    if emit_len > 0 {
                        Self::emit_prefix(&mut self.buffer, emit_len, &mut reasoning);
                        self.reasoning_has_text = true;
                    }
                    break;
                }
                StreamingState::AfterReasoning => {
                    // Preserve whitespace between the reasoning boundary and
                    // content, while continuing to suppress duplicate closers.
                    let whitespace_len = Self::leading_whitespace_len(&self.buffer);
                    if whitespace_len > 0 {
                        Self::emit_prefix(&mut self.buffer, whitespace_len, &mut normal);
                        if self.buffer.is_empty() {
                            break;
                        }
                    }

                    if THINK_END.starts_with(&self.buffer) && self.buffer != THINK_END {
                        break;
                    }
                    if self.buffer.starts_with(THINK_END) {
                        self.buffer.drain(..THINK_END.len());
                        continue;
                    }

                    self.state = StreamingState::Content;
                }
                StreamingState::Content => {
                    let len = self.buffer.len();
                    Self::emit_prefix(&mut self.buffer, len, &mut normal);
                    break;
                }
            }
        }

        ParserResult::new(normal, reasoning)
    }
}

impl Default for MinimaxM3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for MinimaxM3Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        if text.len() > DEFAULT_MAX_BUFFER_SIZE {
            return Err(ParseError::BufferOverflow(text.len()));
        }

        Ok(if self.state == StreamingState::InReasoning {
            Self::parse_complete_inside_reasoning(text)
        } else {
            Self::parse_complete_outside_reasoning(text)
        })
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        if self.buffer.len() + text.len() > DEFAULT_MAX_BUFFER_SIZE {
            return Err(ParseError::BufferOverflow(self.buffer.len() + text.len()));
        }

        self.buffer.push_str(text);
        Ok(self.parse_stream_buffer())
    }

    fn flush(&mut self) -> Result<ParserResult, ParseError> {
        let text = std::mem::take(&mut self.buffer);
        Ok(if self.is_in_reasoning() {
            ParserResult::reasoning(text)
        } else {
            ParserResult::normal(text)
        })
    }

    fn reset(&mut self) {
        self.state = StreamingState::BeforeReasoning;
        self.buffer.clear();
        self.allow_leading_closers = true;
        self.reasoning_has_text = false;
    }

    fn model_type(&self) -> &str {
        "minimax_m3"
    }

    fn is_in_reasoning(&self) -> bool {
        self.state == StreamingState::InReasoning
    }

    fn mark_reasoning_started(&mut self) {
        self.state = StreamingState::InReasoning;
        self.reasoning_has_text = false;
    }

    fn mark_think_start_stripped(&mut self) {
        // 'mark_reasoning_started' owns the state transition. The dedicated M3
        // state machine already knows that a generated start marker is optional.
    }
}

#[cfg(test)]
#[path = "../../tests/unit/parsers/minimax_m3/tests.rs"]
mod tests;
