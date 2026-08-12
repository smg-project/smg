//! Shared test fakes for the regular (non-harmony) gRPC router tests.
//!
//! These fakes exist solely to exercise error-propagation paths in
//! [`processor`] and [`streaming`]. They are compiled only under
//! `#[cfg(test)]` and intentionally fail every operation they implement.
//!
//! [`processor`]: super::processor
//! [`streaming`]: super::streaming

use async_trait::async_trait;
use llm_tokenizer::traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer};
use openai_protocol::common::Tool;
use reasoning_parser::{ParseError, ParserResult, ReasoningParser};
use tool_parser::{
    errors::{ParserError as ToolParserError, ParserResult as ToolParserResult},
    types::{StreamingParseResult, ToolCall as ParsedToolCall},
    ToolParser,
};

pub(crate) struct FailingReasoningParser;

pub(crate) struct FailingToolParser;

#[derive(Default)]
pub(crate) struct FailingTokenizer {
    special_tokens: SpecialTokens,
}

impl Encoder for FailingTokenizer {
    fn encode(&self, _input: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
        Ok(Encoding::Plain(Vec::new()))
    }

    fn encode_batch(
        &self,
        inputs: &[&str],
        _add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        Ok(inputs.iter().map(|_| Encoding::Plain(Vec::new())).collect())
    }
}

impl Decoder for FailingTokenizer {
    fn decode(&self, _token_ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
        Err(anyhow::anyhow!("fake decode failure"))
    }
}

impl Tokenizer for FailingTokenizer {
    fn vocab_size(&self) -> usize {
        0
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }

    fn id_to_token(&self, _id: u32) -> Option<String> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ReasoningParser for FailingReasoningParser {
    fn detect_and_parse_reasoning(
        &mut self,
        _text: &str,
    ) -> Result<ParserResult, ParseError> {
        Err(ParseError::ConfigError("fake failure".to_string()))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        _text: &str,
    ) -> Result<ParserResult, ParseError> {
        Err(ParseError::ConfigError("fake failure".to_string()))
    }

    fn reset(&mut self) {}

    fn model_type(&self) -> &str {
        "failing"
    }

    fn is_in_reasoning(&self) -> bool {
        false
    }

    fn mark_reasoning_started(&mut self) {}

    fn mark_think_start_stripped(&mut self) {}
}

#[async_trait]
impl ToolParser for FailingToolParser {
    async fn parse_complete(
        &self,
        _output: &str,
    ) -> ToolParserResult<(String, Vec<ParsedToolCall>)> {
        Err(ToolParserError::ParsingFailed("fake failure".to_string()))
    }

    async fn parse_incremental(
        &mut self,
        _chunk: &str,
        _tools: &[Tool],
    ) -> ToolParserResult<StreamingParseResult> {
        Err(ToolParserError::ParsingFailed("fake failure".to_string()))
    }

    fn has_tool_markers(&self, _text: &str) -> bool {
        false
    }
}
