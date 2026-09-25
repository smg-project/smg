// Passthrough reasoning parser: forwards input as normal_text, no extraction.

use crate::traits::{ParseError, ParserResult, PromptReasoning, ReasoningParser};

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

    fn prompt_reasoning(&self, _prompt: &str) -> PromptReasoning {
        PromptReasoning::Absent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_streaming_forwards_input_verbatim() {
        let mut parser = PassthroughParser::new();
        let result = parser
            .detect_and_parse_reasoning("  <think>cot</think>answer  \n")
            .unwrap();
        assert_eq!(result.normal_text, "  <think>cot</think>answer  \n");
        assert_eq!(result.reasoning_text, "");
    }

    #[test]
    fn streaming_forwards_chunks_verbatim() {
        let mut parser = PassthroughParser::new();
        let r1 = parser
            .parse_reasoning_streaming_incremental("  <think>cot")
            .unwrap();
        assert_eq!(r1.normal_text, "  <think>cot");
        assert_eq!(r1.reasoning_text, "");
        let r2 = parser
            .parse_reasoning_streaming_incremental("</think>answer  \n")
            .unwrap();
        assert_eq!(r2.normal_text, "</think>answer  \n");
        assert_eq!(r2.reasoning_text, "");
    }
}
