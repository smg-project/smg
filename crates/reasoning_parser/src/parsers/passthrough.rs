// Passthrough reasoning parser: forwards input as normal_text, no extraction.

use crate::traits::{ParseError, ParserResult, ReasoningParser};

#[derive(Debug, Clone, Default)]
pub struct PassthroughParser;

impl PassthroughParser {
    pub fn new() -> Self {
        Self
    }
}

impl ReasoningParser for PassthroughParser {
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError> {
        Ok(ParserResult::normal(text.to_string()))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, ParseError> {
        Ok(ParserResult::normal(text.to_string()))
    }

    fn reset(&mut self) {}

    fn model_type(&self) -> &str {
        "passthrough"
    }

    fn is_in_reasoning(&self) -> bool {
        false
    }

    fn mark_reasoning_started(&mut self) {}

    fn mark_think_start_stripped(&mut self) {}
}

#[cfg(test)]
#[path = "../../tests/unit/parsers/passthrough/tests.rs"]
mod tests;
