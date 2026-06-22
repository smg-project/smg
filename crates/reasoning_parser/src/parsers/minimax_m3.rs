// MiniMax M3 specific reasoning parser.
//
// MiniMax M3 emits reasoning inside `<mm:think>...</mm:think>` blocks. The vLLM
// reference parser constructs its delimited state machine with the
// `initial_in_reasoning = false` argument and only flips to "already reasoning"
// when the rendered prompt prefills the start marker (thinking_mode="enabled").
// SMG's `BaseReasoningParser` derives its starting state from a single static
// flag rather than the rendered prompt, so this parser uses
// `always_in_reasoning = false` to match the default (non-prefilled) behavior:
// output is normal text until an explicit `<mm:think>` token appears.

use crate::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE},
};

/// MiniMax M3 reasoning parser.
///
/// Uses the `<mm:think>` / `</mm:think>` delimiters. By default the M3 chat
/// template does not prefill the start marker, so the parser begins outside a
/// reasoning block (`always_in_reasoning = false`) and enters one only when an
/// explicit `<mm:think>` token is seen.
pub struct MinimaxM3Parser {
    base: BaseReasoningParser,
}

impl MinimaxM3Parser {
    /// Create a new MiniMax M3 reasoning parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<mm:think>".to_string(),
            think_end_token: "</mm:think>".to_string(),
            stream_reasoning: true,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            always_in_reasoning: false,
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("minimax_m3".to_string()),
        }
    }
}

impl Default for MinimaxM3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for MinimaxM3Parser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        self.base.detect_and_parse_reasoning(text)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        self.base.parse_reasoning_streaming_incremental(text)
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
}
