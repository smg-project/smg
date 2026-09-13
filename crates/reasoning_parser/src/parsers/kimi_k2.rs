// Kimi K2 family reasoning parser (`kimi_k2`).
//
// One parser for the whole Kimi K2 family (K2 / K2-Thinking / K2.5 / K2.6 /
// K2.7), matching vLLM and SGLang's unified `kimi_k2` reasoning parser and
// Moonshot's deploy guidance. K2.5+ chat templates prefill `<think>` at the
// generation prompt (thinking is on by default), so model output *begins
// inside* reasoning:
//
// - Starts in reasoning; a leading `<think>` is consumed if present
//   (prefill-robust both ways, like vLLM's self-correcting start token).
// - Reasoning ends on `</think>` OR `<|tool_calls_section_begin|>` — Kimi can
//   go straight from reasoning into a tool section without closing the think
//   block. The tool-section marker is forwarded as content so the downstream
//   tool parser can parse it.
//
// Replaces the per-SKU `kimi_k25` / `kimi_thinking` pair whose static
// `always_in_reasoning` flag guessed the template behavior (#1873).

use crate::traits::{ParseError, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE};

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL_SECTION_START: &str = "<|tool_calls_section_begin|>";

/// Which end-of-reasoning marker was found.
enum EndKind {
    /// `</think>` — consumed, not forwarded.
    ThinkEnd,
    /// `<|tool_calls_section_begin|>` — forwarded as content for the tool parser.
    ToolSection,
}

/// Unified Kimi K2 reasoning parser.
pub struct KimiK2Parser {
    in_reasoning: bool,
    reasoning_ended: bool,
    start_decided: bool,
    buffer: String,
    max_buffer_size: usize,
}

impl KimiK2Parser {
    pub fn new() -> Self {
        Self {
            in_reasoning: true,
            reasoning_ended: false,
            start_decided: false,
            buffer: String::new(),
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
        }
    }

    /// Earliest end-of-reasoning marker in `text`, if any.
    fn find_reasoning_end(text: &str) -> Option<(usize, EndKind)> {
        match (text.find(THINK_END), text.find(TOOL_SECTION_START)) {
            (None, None) => None,
            (Some(i), None) => Some((i, EndKind::ThinkEnd)),
            (None, Some(i)) => Some((i, EndKind::ToolSection)),
            (Some(a), Some(b)) => Some(if a <= b {
                (a, EndKind::ThinkEnd)
            } else {
                (b, EndKind::ToolSection)
            }),
        }
    }

    /// Length of the longest buffer suffix that is a proper prefix of one of
    /// `tokens` — held back while streaming so split markers aren't emitted
    /// as text.
    fn trailing_prefix_of(buffer: &str, tokens: &[&str]) -> usize {
        let mut longest = 0;
        for token in tokens {
            for n in 1..token.len() {
                if buffer.ends_with(&token[..n]) {
                    longest = longest.max(n);
                }
            }
        }
        longest
    }
}

impl Default for KimiK2Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for KimiK2Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        if text.len() > self.max_buffer_size {
            return Err(ParseError::BufferOverflow(text.len()));
        }
        if self.reasoning_ended {
            return Ok(ParserResult::normal(text.to_string()));
        }

        // Consume a leading <think> if the template left one; otherwise the
        // output already starts inside reasoning (the prefill case).
        let body = text.strip_prefix(THINK_START).unwrap_or(text);
        self.start_decided = true;

        match Self::find_reasoning_end(body) {
            Some((idx, EndKind::ThinkEnd)) => {
                self.in_reasoning = false;
                self.reasoning_ended = true;
                Ok(ParserResult::new(
                    body[idx + THINK_END.len()..].to_string(),
                    body[..idx].to_string(),
                ))
            }
            Some((idx, EndKind::ToolSection)) => {
                self.in_reasoning = false;
                self.reasoning_ended = true;
                Ok(ParserResult::new(
                    body[idx..].to_string(),
                    body[..idx].to_string(),
                ))
            }
            // Assume reasoning was truncated before any end marker.
            None => Ok(ParserResult::reasoning(body.to_string())),
        }
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        if self.buffer.len() + text.len() > self.max_buffer_size {
            return Err(ParseError::BufferOverflow(self.buffer.len() + text.len()));
        }
        self.buffer.push_str(text);

        if self.reasoning_ended {
            // Hold back a trailing partial tool-section marker: downstream
            // tool parsers drain marker-less deltas as user-visible text, so
            // a split marker would never reassemble.
            let hold = Self::trailing_prefix_of(&self.buffer, &[TOOL_SECTION_START]);
            let end = self.buffer.len() - hold;
            let normal: String = self.buffer.drain(..end).collect();
            return Ok(ParserResult::normal(normal));
        }

        // Resolve the leading <think> question exactly once: consume it if
        // present, keep buffering while it could still form, otherwise the
        // template prefilled it and output starts inside reasoning.
        if !self.start_decided {
            if self.buffer.starts_with(THINK_START) {
                self.buffer.drain(..THINK_START.len());
                self.start_decided = true;
            } else if THINK_START.starts_with(self.buffer.as_str()) {
                return Ok(ParserResult::default());
            } else {
                self.start_decided = true;
            }
        }

        if let Some((idx, kind)) = Self::find_reasoning_end(&self.buffer) {
            let reasoning = self.buffer[..idx].to_string();
            let (normal, held) = match kind {
                EndKind::ThinkEnd => {
                    let rest = &self.buffer[idx + THINK_END.len()..];
                    let hold = Self::trailing_prefix_of(rest, &[TOOL_SECTION_START]);
                    (
                        rest[..rest.len() - hold].to_string(),
                        rest[rest.len() - hold..].to_string(),
                    )
                }
                EndKind::ToolSection => (self.buffer[idx..].to_string(), String::new()),
            };
            self.buffer = held;
            self.in_reasoning = false;
            self.reasoning_ended = true;
            return Ok(ParserResult::new(normal, reasoning));
        }

        // Stream everything except a trailing suffix that may be the start of
        // an end marker split across chunks.
        let hold = Self::trailing_prefix_of(&self.buffer, &[THINK_END, TOOL_SECTION_START]);
        let end = self.buffer.len() - hold;
        let reasoning: String = self.buffer.drain(..end).collect();
        Ok(ParserResult::reasoning(reasoning))
    }

    fn reset(&mut self) {
        self.in_reasoning = true;
        self.reasoning_ended = false;
        self.start_decided = false;
        self.buffer.clear();
    }

    fn model_type(&self) -> &str {
        "kimi_k2"
    }

    fn is_in_reasoning(&self) -> bool {
        self.in_reasoning
    }

    fn mark_reasoning_started(&mut self) {
        // A new turn begins in reasoning: clear the terminal latch and any
        // stale buffer from the previous turn, and re-resolve the leading
        // <think> question for the new output.
        self.in_reasoning = true;
        self.reasoning_ended = false;
        self.start_decided = false;
        self.buffer.clear();
    }

    fn mark_think_start_stripped(&mut self) {
        self.start_decided = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frozen Kimi-K2.6-style output: starts mid-reasoning (template prefilled
    /// `<think>`), closes the think block, then emits a tool section.
    const K26_GOLDEN: &str = "The user wants the current weather in Tokyo. I should call the weather tool with the city filled in.</think>\n\n<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Tokyo\"}<|tool_call_end|><|tool_calls_section_end|>";

    const K26_GOLDEN_REASONING: &str =
        "The user wants the current weather in Tokyo. I should call the weather tool with the city filled in.";
    const K26_GOLDEN_CONTENT: &str = "\n\n<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Tokyo\"}<|tool_call_end|><|tool_calls_section_end|>";

    #[test]
    fn kimi_k2_golden_k26_output_split() {
        let mut parser = KimiK2Parser::new();
        let result = parser.detect_and_parse_reasoning(K26_GOLDEN).unwrap();
        assert_eq!(result.reasoning_text, K26_GOLDEN_REASONING);
        assert_eq!(result.normal_text, K26_GOLDEN_CONTENT);
    }

    #[test]
    fn kimi_k2_ends_reasoning_at_tool_section_without_think_end() {
        // Kimi can go straight from reasoning into a tool section without
        // emitting </think>; the marker must be forwarded for the tool parser.
        let mut parser = KimiK2Parser::new();
        let output = "let me think about this<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_end|>";
        let result = parser.detect_and_parse_reasoning(output).unwrap();
        assert_eq!(result.reasoning_text, "let me think about this");
        assert_eq!(
            result.normal_text,
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_end|>"
        );
    }

    #[test]
    fn kimi_k2_consumes_leading_think_start_when_present() {
        // Same model served through a template that does NOT prefill: the
        // model emits <think> itself, and the parser must consume it.
        let mut parser = KimiK2Parser::new();
        let result = parser
            .detect_and_parse_reasoning("<think>reasoning</think>answer")
            .unwrap();
        assert_eq!(result.reasoning_text, "reasoning");
        assert_eq!(result.normal_text, "answer");
    }

    #[test]
    fn kimi_k2_truncated_reasoning_is_all_reasoning() {
        let mut parser = KimiK2Parser::new();
        let result = parser
            .detect_and_parse_reasoning("still thinking, no end token")
            .unwrap();
        assert_eq!(result.reasoning_text, "still thinking, no end token");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn kimi_k2_streaming_chunked_matches_non_streaming() {
        // Feed the golden output in awkward chunks that split both end
        // markers; the streamed split must equal the one-shot split.
        let chunks = [
            "The user",
            " wants the current weather in Tokyo. I should call the weather tool with the city filled in.",
            "</th",
            "ink>\n\n<|tool_calls_se",
            "ction_begin|><|tool_call_begin|>functions.get_weather:0",
            "<|tool_call_argument_begin|>{\"city\": \"Tokyo\"}<|tool_call_end|><|tool_calls_section_end|>",
        ];
        let mut parser = KimiK2Parser::new();
        let mut reasoning = String::new();
        let mut normal = String::new();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
            reasoning.push_str(&result.reasoning_text);
            normal.push_str(&result.normal_text);
        }
        assert_eq!(reasoning, K26_GOLDEN_REASONING);
        assert_eq!(normal, K26_GOLDEN_CONTENT);
    }

    #[test]
    fn kimi_k2_streaming_leading_think_start_consumed() {
        let mut parser = KimiK2Parser::new();
        let r1 = parser
            .parse_reasoning_streaming_incremental("<thi")
            .unwrap();
        assert!(r1.is_empty());
        let r2 = parser
            .parse_reasoning_streaming_incremental("nk>reasoning</think>answer")
            .unwrap();
        assert_eq!(r2.reasoning_text, "reasoning");
        assert_eq!(r2.normal_text, "answer");
    }

    #[test]
    fn kimi_k2_reset_restores_initial_state() {
        let mut parser = KimiK2Parser::new();
        parser.detect_and_parse_reasoning(K26_GOLDEN).unwrap();
        assert!(!parser.is_in_reasoning());
        parser.reset();
        assert!(parser.is_in_reasoning());
        let result = parser.detect_and_parse_reasoning(K26_GOLDEN).unwrap();
        assert_eq!(result.reasoning_text, K26_GOLDEN_REASONING);
    }

    #[test]
    fn kimi_k2_streaming_preserves_tool_marker_split_at_transition() {
        // The chunk boundary falls right after </think>, leaving a partial
        // tool-section marker. The reasoning parser must not emit the
        // fragment: downstream tool parsers drain marker-less deltas as
        // user-visible text, so a split marker would be lost forever.
        let chunks = ["reasoning</think><|tool_calls_se", "ction_begin|>rest"];
        let mut parser = KimiK2Parser::new();

        let r1 = parser
            .parse_reasoning_streaming_incremental(chunks[0])
            .unwrap();
        assert_eq!(r1.reasoning_text, "reasoning");
        assert_eq!(r1.normal_text, "");

        let r2 = parser
            .parse_reasoning_streaming_incremental(chunks[1])
            .unwrap();
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, "<|tool_calls_section_begin|>rest");
    }

    #[test]
    fn kimi_k2_mark_reasoning_started_restarts_after_completion() {
        // Parser reuse across turns: after a completed turn, the runtime marks
        // the next turn as reasoning-from-start (template prefilled <think>).
        // The parser must actually return to reasoning, not stay latched in
        // the terminal state.
        let mut parser = KimiK2Parser::new();
        parser.detect_and_parse_reasoning(K26_GOLDEN).unwrap();
        assert!(!parser.is_in_reasoning());

        parser.mark_reasoning_started();
        assert!(parser.is_in_reasoning());

        let result = parser
            .parse_reasoning_streaming_incremental("new turn thinking</think>new answer")
            .unwrap();
        assert_eq!(result.reasoning_text, "new turn thinking");
        assert_eq!(result.normal_text, "new answer");
    }

    #[test]
    fn kimi_k2_model_type() {
        let parser = KimiK2Parser::new();
        assert_eq!(parser.model_type(), "kimi_k2");
    }
}
