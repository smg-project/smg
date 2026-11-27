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
#[path = "../../tests/unit/parsers/cohere_cmd/tests.rs"]
mod tests;
