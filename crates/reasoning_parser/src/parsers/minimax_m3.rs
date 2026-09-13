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
mod tests {
    use super::*;

    fn collect_stream(parser: &mut MinimaxM3Parser, chunks: &[&str]) -> ParserResult {
        let mut result = ParserResult::default();
        for chunk in chunks {
            let parsed = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
            result.normal_text.push_str(&parsed.normal_text);
            result.reasoning_text.push_str(&parsed.reasoning_text);
        }
        result
    }

    #[test]
    fn test_model_type() {
        let parser = MinimaxM3Parser::new();
        assert_eq!(parser.model_type(), "minimax_m3");
    }

    #[test]
    fn test_fresh_parser_not_in_reasoning() {
        let parser = MinimaxM3Parser::new();
        // always_in_reasoning=false -> starts outside reasoning.
        assert!(!parser.is_in_reasoning());
    }

    #[test]
    fn test_reasoning_extraction_complete() {
        let mut parser = MinimaxM3Parser::new();
        let result = parser
            .detect_and_parse_reasoning("<mm:think>thinking here</mm:think>normal text")
            .unwrap();
        assert_eq!(result.reasoning_text, "thinking here");
        assert_eq!(result.normal_text, "normal text");
    }

    #[test]
    fn test_no_think_passthrough() {
        let mut parser = MinimaxM3Parser::new();
        // Without a start token, content is treated as normal text.
        let result = parser
            .detect_and_parse_reasoning("just a plain answer")
            .unwrap();
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "just a plain answer");
    }

    #[test]
    fn test_nonstreaming_drops_repeated_leading_closers() {
        let mut parser = MinimaxM3Parser::new();
        for (input, expected) in [
            ("</mm:think>Hello.", "Hello."),
            ("</mm:think></mm:think>tool", "tool"),
        ] {
            let result = parser.detect_and_parse_reasoning(input).unwrap();
            assert_eq!(result, ParserResult::normal(expected.to_string()));
        }
    }

    #[test]
    fn test_nonstreaming_preserves_non_leading_unmatched_closer() {
        let mut parser = MinimaxM3Parser::new();
        let result = parser
            .detect_and_parse_reasoning("visible</mm:think>text")
            .unwrap();
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "visible</mm:think>text");
    }

    #[test]
    fn test_streaming_incremental_across_chunks() {
        let mut parser = MinimaxM3Parser::new();

        let r1 = parser
            .parse_reasoning_streaming_incremental("<mm:think>thinking ")
            .unwrap();
        assert_eq!(r1.reasoning_text, "thinking ");
        assert_eq!(r1.normal_text, "");

        let r2 = parser
            .parse_reasoning_streaming_incremental("about it</mm:think>the answer")
            .unwrap();
        assert_eq!(r2.reasoning_text, "about it");
        assert_eq!(r2.normal_text, "the answer");
    }

    #[test]
    fn test_streaming_holds_end_marker_suffix_after_reasoning_text() {
        let mut parser = MinimaxM3Parser::new();
        let result = collect_stream(&mut parser, &["<mm:think>plan</mm:", "think>ANSWER"]);
        assert_eq!(result.reasoning_text, "plan");
        assert_eq!(result.normal_text, "ANSWER");
    }

    #[test]
    fn test_streaming_drops_repeated_leading_closers_across_chunks() {
        let mut parser = MinimaxM3Parser::new();
        let result = collect_stream(
            &mut parser,
            &["</mm:", "think></mm:think>", "]<]minimax[>[<tool_call>"],
        );
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "]<]minimax[>[<tool_call>");
    }

    #[test]
    fn test_streaming_one_character_chunks() {
        let text = "</mm:think></mm:think><mm:think>思考</mm:think>答案";
        let chunks: Vec<&str> = text
            .char_indices()
            .map(|(start, _)| start)
            .zip(
                text.char_indices()
                    .map(|(start, _)| start)
                    .skip(1)
                    .chain(std::iter::once(text.len())),
            )
            .map(|(start, end)| &text[start..end])
            .collect();
        let mut parser = MinimaxM3Parser::new();
        let result = collect_stream(&mut parser, &chunks);
        assert_eq!(result.reasoning_text, "思考");
        assert_eq!(result.normal_text, "答案");
    }

    #[test]
    fn test_prefilled_thinking_marked_splits_without_start_token() {
        // thinking_mode="enabled" prefills <mm:think> in the prompt, so the
        // output stream carries no start token; the pipeline marks the parser.
        let mut parser = MinimaxM3Parser::new();
        parser.mark_reasoning_started();
        let result = parser
            .detect_and_parse_reasoning("deep thoughts</mm:think>the answer")
            .unwrap();
        assert_eq!(result.reasoning_text, "deep thoughts");
        assert_eq!(result.normal_text, "the answer");

        let mut streaming = MinimaxM3Parser::new();
        streaming.mark_reasoning_started();
        let r1 = streaming
            .parse_reasoning_streaming_incremental("deep ")
            .unwrap();
        assert_eq!(r1.reasoning_text, "deep ");
        assert_eq!(r1.normal_text, "");
        let r2 = streaming
            .parse_reasoning_streaming_incremental("thoughts</mm:think>the answer")
            .unwrap();
        assert_eq!(r2.reasoning_text, "thoughts");
        assert_eq!(r2.normal_text, "the answer");
    }

    #[test]
    fn test_prefilled_thinking_drops_duplicate_closers() {
        let mut parser = MinimaxM3Parser::new();
        parser.mark_reasoning_started();
        let result = collect_stream(
            &mut parser,
            &["<mm:think>deep</mm:", "think></mm:think>answer"],
        );
        assert_eq!(result.reasoning_text, "deep");
        assert_eq!(result.normal_text, "answer");
    }

    #[test]
    fn test_reset_behavior() {
        let mut parser = MinimaxM3Parser::new();
        parser
            .parse_reasoning_streaming_incremental("<mm:think>partial reasoning")
            .unwrap();
        assert!(parser.is_in_reasoning());

        parser.reset();
        // After reset, state returns to the configured initial value (false).
        assert!(!parser.is_in_reasoning());

        let result = parser
            .detect_and_parse_reasoning("<mm:think>fresh</mm:think>done")
            .unwrap();
        assert_eq!(result.reasoning_text, "fresh");
        assert_eq!(result.normal_text, "done");
    }

    #[test]
    fn test_buffer_overflow() {
        let mut parser = MinimaxM3Parser::new();
        let oversized = "x".repeat(DEFAULT_MAX_BUFFER_SIZE + 1);
        assert!(matches!(
            parser.detect_and_parse_reasoning(&oversized),
            Err(ParseError::BufferOverflow(_))
        ));
    }
}
