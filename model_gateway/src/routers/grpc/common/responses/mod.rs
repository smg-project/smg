//! Shared response functionality used by both regular and harmony implementations

pub(crate) mod context;
pub(crate) mod handlers;
pub(crate) mod streaming;
pub(crate) mod utils;

// Re-export commonly used items
pub(crate) use context::ResponsesContext;
pub(crate) use streaming::{
    await_stream_startup, build_sse_response, signal_stream_startup, stream_startup_channel,
    StreamStartupSender,
};
pub(crate) use utils::{ensure_mcp_connection, persist_response_if_needed};

pub(crate) use crate::routers::common::mcp_utils::collect_user_function_names;
