//! MCP (Model Context Protocol) module for the OpenAI router.
//!
//! Contains tool loop orchestration and streaming tool call handling,
//! extracted from `responses/` for separation of concerns.

#[cfg(test)]
#[path = "../../../tests/unit/openai/mcp/test_support.rs"]
pub(crate) mod test_support;
mod tool_handler;
mod tool_loop;

// Re-export types used by responses/streaming.rs
pub(crate) use tool_handler::{StreamAction, StreamingToolHandler};
// Re-export functions used by responses/streaming.rs and responses/non_streaming.rs
pub(crate) use tool_loop::{
    close_streaming_open_items, execute_streaming_tool_calls, execute_tool_loop,
    inject_mcp_metadata_last, prepare_mcp_tools_as_functions, resume_payload_in_place,
    retained_output_items, send_mcp_list_tools_events, skip_streaming_tool_calls,
    ToolLoopExecutionContext, ToolLoopState, NOT_EXECUTED_AT_STREAM_BREAK,
    NOT_EXECUTED_AT_TOOL_CALL_LIMIT, TOOL_CALL_LIMIT_REASON,
};
