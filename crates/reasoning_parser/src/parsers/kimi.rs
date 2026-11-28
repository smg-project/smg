// Kimi specific reasoning parser.
// This parser uses Unicode tokens and starts with in_reasoning=false.

use crate::{
    parsers::BaseReasoningParser,
    traits::{ParseError, ParserConfig, ParserResult, ReasoningParser, DEFAULT_MAX_BUFFER_SIZE},
};

/// Kimi reasoning parser.
///
/// This parser uses Unicode tokens (◁think▷ and ◁/think▷) and requires
/// explicit start tokens to enter reasoning mode.
pub struct KimiParser {
    base: BaseReasoningParser,
}

impl KimiParser {
    /// Create a new Kimi parser.
    pub fn new() -> Self {
        let config = ParserConfig {
            think_start_token: "◁think▷".to_string(),
            think_end_token: "◁/think▷".to_string(),
            stream_reasoning: true,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            always_in_reasoning: false,
        };

        Self {
            base: BaseReasoningParser::new(config).with_model_type("kimi".to_string()),
        }
    }
}

impl Default for KimiParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for KimiParser {
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
#[path = "../../tests/unit/parsers/kimi/tests.rs"]
mod tests;
