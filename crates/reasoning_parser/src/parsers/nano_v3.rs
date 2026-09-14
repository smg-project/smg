// NanoV3 / Nemotron reasoning parser.
//
// The Nemotron chat template supports an `enable_thinking` toggle that injects
// `<think>\n` in the prefill when ON and `<think></think>` when OFF.
// See: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8?chat_template=default
//
// Uses `always_in_reasoning=false` because the thinking toggle is detected at
// runtime via `ThinkingToggle::DefaultOn` + `think_in_prefill=true`, and
// `mark_reasoning_started()` is called when thinking is effectively ON.

use crate::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE},
};

/// NanoV3 / Nemotron reasoning parser.
pub struct NanoV3Parser {
    base: BaseReasoningParser,
}

impl NanoV3Parser {
    /// Create a new NanoV3 parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            always_in_reasoning: false,
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("nano_v3".to_string()),
        }
    }
}

impl Default for NanoV3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for NanoV3Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        self.base.detect_and_parse_reasoning(text)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        self.base.parse_reasoning_streaming_incremental(text)
    }

    fn flush(&mut self) -> Result<ParserResult, ParseError> {
        self.base.flush()
    }

    fn reset(&mut self) {
        self.base.reset();
    }

    fn model_type(&self) -> &str {
        self.base.model_type()
    }

    fn is_in_reasoning(&self) -> bool {
        self.base.is_in_reasoning()
    }

    fn mark_reasoning_started(&mut self) {
        self.base.mark_reasoning_started();
    }

    fn mark_think_start_stripped(&mut self) {
        self.base.mark_think_start_stripped();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nano_v3_initial_state() {
        let mut parser = NanoV3Parser::new();

        // Without mark_reasoning_started(), text without <think> is normal content
        let result = parser
            .detect_and_parse_reasoning("This is normal content")
            .unwrap();
        assert_eq!(result.normal_text, "This is normal content");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_nano_v3_with_mark_reasoning_started() {
        let mut parser = NanoV3Parser::new();
        parser.mark_reasoning_started();

        // After mark_reasoning_started(), text is treated as reasoning
        let result = parser
            .detect_and_parse_reasoning("reasoning content</think>answer")
            .unwrap();
        assert_eq!(result.normal_text, "answer");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_nano_v3_with_both_tokens() {
        let mut parser = NanoV3Parser::new();

        let result = parser
            .detect_and_parse_reasoning("<think>reasoning content</think>answer")
            .unwrap();
        assert_eq!(result.normal_text, "answer");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_nano_v3_streaming() {
        let mut parser = NanoV3Parser::new();
        parser.mark_reasoning_started();

        let result1 = parser
            .parse_reasoning_streaming_incremental("reasoning text ")
            .unwrap();
        assert_eq!(result1.normal_text, "");
        assert_eq!(result1.reasoning_text, "reasoning text ");

        let result2 = parser
            .parse_reasoning_streaming_incremental("more reasoning</think>answer")
            .unwrap();
        assert_eq!(result2.normal_text, "answer");
        assert_eq!(result2.reasoning_text, "more reasoning");
    }

    #[test]
    fn test_model_type() {
        let parser = NanoV3Parser::new();
        assert_eq!(parser.model_type(), "nano_v3");
    }
}
