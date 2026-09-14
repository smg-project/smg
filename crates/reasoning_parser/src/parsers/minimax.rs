// MiniMax M2 specific reasoning parser.
// The MiniMax-M2 template always injects <think> in the prefill,
// so the parser uses always_in_reasoning=true.

use crate::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE},
};

/// MiniMax M2 reasoning parser.
///
/// The MiniMax-M2 template always injects `<think>` in the prefill, so the model
/// outputs reasoning content directly. Uses `always_in_reasoning=true`.
pub struct MiniMaxParser {
    base: BaseReasoningParser,
}

impl MiniMaxParser {
    /// Create a new MiniMax M2 parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<think>".to_string(),
            think_end_token: "</think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            always_in_reasoning: true,
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("minimax".to_string()),
        }
    }
}

impl Default for MiniMaxParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for MiniMaxParser {
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
    fn test_minimax_always_in_reasoning() {
        let mut parser = MiniMaxParser::new();

        // always_in_reasoning=true: treats all text as reasoning until </think>
        let result = parser
            .detect_and_parse_reasoning("reasoning content</think>normal content")
            .unwrap();
        assert_eq!(result.normal_text, "normal content");
        assert_eq!(result.reasoning_text, "reasoning content");
    }

    #[test]
    fn test_minimax_without_end_token() {
        let mut parser = MiniMaxParser::new();

        // Should treat all content as reasoning when no end token
        let result = parser
            .detect_and_parse_reasoning("all reasoning content")
            .unwrap();
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "all reasoning content");
    }

    #[test]
    fn test_minimax_streaming() {
        let mut parser = MiniMaxParser::new();

        // First chunk — already in reasoning mode
        let result1 = parser
            .parse_reasoning_streaming_incremental("thinking about")
            .unwrap();
        assert_eq!(result1.reasoning_text, "thinking about");
        assert_eq!(result1.normal_text, "");

        // End of reasoning
        let result2 = parser
            .parse_reasoning_streaming_incremental(" the problem</think>answer")
            .unwrap();
        assert_eq!(result2.reasoning_text, " the problem");
        assert_eq!(result2.normal_text, "answer");
    }

    #[test]
    fn test_model_type() {
        let parser = MiniMaxParser::new();
        assert_eq!(parser.model_type(), "minimax");
    }
}
