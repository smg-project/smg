//! OpenAI-compatible responses handling module
//!
//! This module provides comprehensive support for OpenAI Responses API with:
//! - Streaming and non-streaming response handling
//! - MCP (Model Context Protocol) tool interception and execution
//! - SSE (Server-Sent Events) parsing and forwarding
//! - Response accumulation for persistence
//! - Tool call detection and output index remapping
//! - Input history loading from conversations and response chains
//! - Shared helpers for response retrieval-related logic

mod accumulator;
mod common;
pub(crate) mod history;
mod non_streaming;
pub(crate) mod route;
mod streaming;
pub(crate) mod utils;

// Re-exported for openai::mcp::tool_handler (cross-module dependency)
pub(crate) use accumulator::StreamingResponseAccumulator;
pub(crate) use common::{extract_output_index, get_event_type};
pub use non_streaming::handle_non_streaming_response;
pub use streaming::handle_streaming_response;

use crate::metrics::{metrics_labels, Metrics};

/// Counts a router error of the Responses route: its fixed router, backend,
/// connection and endpoint labels, the model, and the kind of error.
fn record_error(model: &str, kind: &'static str) {
    Metrics::record_router_error(
        metrics_labels::ROUTER_OPENAI,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model,
        metrics_labels::ENDPOINT_RESPONSES,
        kind,
    );
}
