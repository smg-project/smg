//! MCP (Model Context Protocol) module for the OpenAI router.
//!
//! Contains tool loop orchestration and streaming tool call handling,
//! extracted from `responses/` for separation of concerns.

mod tool_handler;
mod tool_loop;

// Re-export types used by responses/streaming.rs
pub(crate) use tool_handler::{StreamAction, StreamingToolHandler};
// Re-export functions used by responses/streaming.rs and responses/non_streaming.rs
pub(crate) use tool_loop::{
    build_resume_payload, execute_streaming_tool_calls, execute_tool_loop,
    inject_mcp_metadata_streaming, mcp_list_tools_bindings_to_emit, prepare_mcp_tools_as_functions,
    send_mcp_list_tools_events, ToolLoopExecutionContext, ToolLoopState,
};
