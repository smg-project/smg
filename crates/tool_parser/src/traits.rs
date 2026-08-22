use async_trait::async_trait;
use openai_protocol::common::Tool;

use crate::{
    errors::ParserResult,
    types::{StreamingParseResult, ToolCall},
};

/// Core trait for all tool parsers
#[async_trait]
pub trait ToolParser: Send + Sync {
    /// Parse complete tool calls from final output
    /// Returns (remaining_normal_text, tool_calls) tuple
    async fn parse_complete(&self, output: &str) -> ParserResult<(String, Vec<ToolCall>)>;

    /// Like [`Self::parse_complete`], but with the request's tool schemas so
    /// schema-aware parsers coerce arg values by declared type. The default
    /// ignores `tools` and delegates to `parse_complete`.
    async fn parse_complete_with_tools(
        &self,
        output: &str,
        _tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete(output).await
    }

    /// Parse tool calls from model output (streaming)
    /// Parsers now maintain internal state, so self is mutable
    ///
    /// # Arguments
    /// * `chunk` - New text chunk from model output
    /// * `tools` - List of available tools for validation
    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult>;

    /// Check if text contains tool calls in this parser's format
    fn has_tool_markers(&self, text: &str) -> bool;

    /// Get unstreamed tool call arguments
    /// Returns tool call items for arguments that have been parsed but not yet streamed
    fn get_unstreamed_tool_args(&self) -> Option<Vec<crate::types::ToolCallItem>> {
        None
    }

    /// Take any text still buffered by the streaming parser that never became
    /// a tool call, transferring ownership to the caller.
    ///
    /// Streaming consumers call this once at end of stream, alongside
    /// [`Self::get_unstreamed_tool_args`]: text held back as a *prospective*
    /// tool call (a bare `{` prefix, a partial start marker, or tool JSON
    /// that never completed) must be surfaced as normal content instead of
    /// being silently dropped — mirroring the non-streaming fallback that
    /// returns unparsable tool text verbatim. Parsers that announced a tool
    /// call from the buffered text return an empty string (the remaining
    /// arguments are recovered via `get_unstreamed_tool_args`).
    ///
    /// The default returns an empty string (parser holds no buffered text).
    fn take_unstreamed_normal_text(&mut self) -> String {
        String::new()
    }

    /// Reset the parser state for reuse across requests.
    /// This should clear all buffers and reset state to initial values.
    fn reset(&mut self) {
        // Default no-op implementation
    }
}

/// Trait for partial JSON parsing
pub trait PartialJsonParser: Send + Sync {
    /// Parse potentially incomplete JSON
    fn parse(&self, input: &str) -> ParserResult<(serde_json::Value, usize)>;

    /// Check if JSON is complete
    fn is_complete(&self, input: &str) -> bool;

    /// Get the maximum parsing depth
    fn max_depth(&self) -> usize;
}
