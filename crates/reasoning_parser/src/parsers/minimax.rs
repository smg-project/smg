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
#[path = "../../tests/unit/parsers/minimax/tests.rs"]
mod tests;
