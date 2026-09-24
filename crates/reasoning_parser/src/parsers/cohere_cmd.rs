//! Cohere Command model reasoning parser.
//!
//! Parses thinking blocks from `<|START_THINKING|>...<|END_THINKING|>`.
//! Supports CMD3 and CMD4 format.

use crate::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE},
};

/// Cohere Command model reasoning parser.
///
/// Handles `<|START_THINKING|>` and `<|END_THINKING|>` tokens.
/// Unlike DeepSeek-R1, Cohere requires explicit start token (always_in_reasoning=false).
pub struct CohereCmdParser {
    base: BaseReasoningParser,
}

impl CohereCmdParser {
    /// Create a new Cohere Command parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "<|START_THINKING|>".to_string(),
            think_end_token: "<|END_THINKING|>".to_string(),
            stream_reasoning: true,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            always_in_reasoning: false,
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("cohere_cmd".to_string()),
        }
    }
}

impl Default for CohereCmdParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for CohereCmdParser {
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
    fn test_cohere_cmd_no_reasoning() {
        let mut parser = CohereCmdParser::new();

        // Without thinking tags, text should be returned as normal
        let result = parser
            .detect_and_parse_reasoning("This is a normal response.")
            .unwrap();
        assert_eq!(result.normal_text, "This is a normal response.");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_cohere_cmd_with_thinking() {
        let mut parser = CohereCmdParser::new();

        let result = parser
            .detect_and_parse_reasoning(
                "<|START_THINKING|>Let me analyze this step by step.<|END_THINKING|>The answer is 42.",
            )
            .unwrap();
        assert_eq!(result.normal_text, "The answer is 42.");
        assert_eq!(result.reasoning_text, "Let me analyze this step by step.");
    }

    #[test]
    fn test_cohere_cmd_truncated_thinking() {
        let mut parser = CohereCmdParser::new();

        // Thinking block without end token (truncated)
        let result = parser
            .detect_and_parse_reasoning("<|START_THINKING|>Analyzing the problem...")
            .unwrap();
        assert_eq!(result.normal_text, "");
        assert_eq!(result.reasoning_text, "Analyzing the problem...");
    }

    #[test]
    fn test_cohere_cmd_streaming() {
        let mut parser = CohereCmdParser::new();

        // First chunk - start of thinking
        let result1 = parser
            .parse_reasoning_streaming_incremental("<|START_THINKING|>Step 1: ")
            .unwrap();
        assert_eq!(result1.reasoning_text, "Step 1: ");
        assert_eq!(result1.normal_text, "");

        // Second chunk - continue thinking
        let result2 = parser
            .parse_reasoning_streaming_incremental("Analyze inputs. ")
            .unwrap();
        assert_eq!(result2.reasoning_text, "Analyze inputs. ");
        assert_eq!(result2.normal_text, "");

        // Third chunk - end thinking and normal text
        let result3 = parser
            .parse_reasoning_streaming_incremental(
                "Step 2: Check.<|END_THINKING|>Here's the answer.",
            )
            .unwrap();
        assert_eq!(result3.reasoning_text, "Step 2: Check.");
        assert_eq!(result3.normal_text, "Here's the answer.");
    }

    #[test]
    fn test_cohere_cmd_streaming_partial_token() {
        let mut parser = CohereCmdParser::new();

        // Partial start token
        let result1 = parser
            .parse_reasoning_streaming_incremental("<|START_")
            .unwrap();
        assert_eq!(result1.reasoning_text, "");
        assert_eq!(result1.normal_text, "");

        // Complete the start token
        let result2 = parser
            .parse_reasoning_streaming_incremental("THINKING|>reasoning")
            .unwrap();
        assert_eq!(result2.reasoning_text, "reasoning");
        assert_eq!(result2.normal_text, "");
    }

    #[test]
    fn test_cohere_cmd_reset() {
        let mut parser = CohereCmdParser::new();

        // Process some text
        parser
            .parse_reasoning_streaming_incremental("<|START_THINKING|>thinking<|END_THINKING|>done")
            .unwrap();

        // Reset
        parser.reset();
        assert!(!parser.is_in_reasoning());

        // Should work fresh again
        let result = parser
            .detect_and_parse_reasoning("<|START_THINKING|>new<|END_THINKING|>text")
            .unwrap();
        assert_eq!(result.reasoning_text, "new");
        assert_eq!(result.normal_text, "text");
    }

    #[test]
    fn test_model_type() {
        let parser = CohereCmdParser::new();
        assert_eq!(parser.model_type(), "cohere_cmd");
    }

    #[test]
    fn test_cohere_full_response_format() {
        let mut parser = CohereCmdParser::new();

        // Simulate full Cohere response with thinking and response markers
        let input = r"<|START_THINKING|>
Let me analyze this step by step.
1. First, I'll consider the question.
2. Then, I'll formulate a response.
<|END_THINKING|>
<|START_RESPONSE|>The answer is 42.<|END_RESPONSE|>";

        let result = parser.detect_and_parse_reasoning(input).unwrap();
        assert!(result.reasoning_text.contains("step by step"));
        // Note: Response markers are passed through - cleaned by tool_parser or response processor
        assert!(result.normal_text.contains("START_RESPONSE"));
    }

    #[test]
    fn test_cohere_cmd_empty_thinking() {
        let mut parser = CohereCmdParser::new();

        let result = parser
            .detect_and_parse_reasoning("<|START_THINKING|><|END_THINKING|>The answer.")
            .unwrap();
        assert_eq!(result.normal_text, "The answer.");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn test_cohere_cmd_unicode_in_thinking() {
        let mut parser = CohereCmdParser::new();

        let result = parser
            .detect_and_parse_reasoning(
                "<|START_THINKING|>分析这个问题 🤔 emoji test<|END_THINKING|>答案是42。",
            )
            .unwrap();
        assert_eq!(result.reasoning_text, "分析这个问题 🤔 emoji test");
        assert_eq!(result.normal_text, "答案是42。");
    }

    #[test]
    fn test_cohere_cmd_angle_brackets_in_thinking() {
        let mut parser = CohereCmdParser::new();

        // Angle brackets that don't match tokens should pass through
        let result = parser
            .detect_and_parse_reasoning(
                "<|START_THINKING|>check if x < 10 and y > 5<|END_THINKING|>result",
            )
            .unwrap();
        assert_eq!(result.reasoning_text, "check if x < 10 and y > 5");
        assert_eq!(result.normal_text, "result");
    }

    #[test]
    fn test_cohere_cmd_whitespace_only_thinking() {
        let mut parser = CohereCmdParser::new();

        let result = parser
            .detect_and_parse_reasoning("<|START_THINKING|>   \n\t  <|END_THINKING|>answer")
            .unwrap();
        // Whitespace is preserved (no trimming in BaseReasoningParser)
        assert_eq!(result.reasoning_text, "   \n\t  ");
        assert_eq!(result.normal_text, "answer");
    }
}
