//! State machine types for the Gemini Interactions router.

use axum::response::Response;

/// Unified request processing state.
///
/// Covers both streaming and non-streaming flows in a single enum.
/// The `BuildRequest` step decides which branch to enter based on
/// whether the request is streaming and whether it contains MCP tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RequestState {
    // ── Shared initial states ───────────────────────────────────
    /// Entry state for every request.
    SelectWorker,
    /// A healthy worker has been selected for this model.
    LoadPreviousInteraction,
    /// Previous interaction loading is complete (or was not needed).
    BuildRequest,

    // ── Non-streaming path ──────────────────────────────────────
    /// The upstream payload is ready for a non-streaming POST.
    NonStreamRequest,
    /// The upstream response has been received (no more tool calls to execute).
    ProcessResponse,

    // ── Streaming path ──────────────────────────────────────────
    /// Streaming request that contains MCP tools (needs tool interception loop).
    StreamRequestWithTool,
    /// Streaming request without MCP tools (simple passthrough).
    StreamRequest,
}

/// The result of executing a single step.
pub(crate) enum StepResult {
    /// The step updated `ctx.state`. The driver should continue the loop.
    Continue,
    /// Terminal: return this `Response` to the client.
    /// Used by the `ResponseProcessing` step to hand back the final HTTP response.
    Response(Response),
}
