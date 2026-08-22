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

    /// Number of structurally complete, valid calls observed by an incremental
    /// parser. `None` means the parser cannot distinguish completion from
    /// provisional deltas and preserves the legacy finish-reason behavior.
    fn completed_tool_call_count(&self) -> Option<usize> {
        None
    }

    /// Finalize buffered incremental input at the Engine terminal boundary.
    /// Parsers use this to release ordinary text that was retained only to
    /// disambiguate a split structural marker.
    fn finish_incremental(&mut self) -> StreamingParseResult {
        StreamingParseResult::default()
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
