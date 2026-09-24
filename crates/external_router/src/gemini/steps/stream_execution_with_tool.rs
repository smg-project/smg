//! StreamRequestExecutionWithTool step (streaming with MCP tool interception).
//!
//! Terminal step: creates an SSE channel, spawns the streaming + tool-loop
//! work in a background task, and returns the SSE `Response` wrapping the
//! receiver. This follows the same pattern as `process_streaming_response`
//! in the gRPC router.
//!
//! Streaming state (`is_first_iteration`, `sequence_number`,
//! `next_output_index`) lives as local variables inside the spawned task,
//! not on the request context.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::gemini::{context::RequestContext, state::StepResult};

/// Stream events from the upstream worker with MCP tool call interception.
///
/// **Not yet implemented** — currently returns 501. Planned behavior:
///
/// Creates an SSE channel (`tx`, `rx`) and spawns a background task that
/// runs an internal tool-loop:
/// 1. Make a streaming request to upstream.
/// 2. Forward/transform/buffer SSE events to the client via `tx`.
/// 3. If tool calls are detected: execute them, send results as SSE events,
///    build a resume payload, and loop back to step 1.
/// 4. If no tool calls: send the final completion event and break.
///
/// The following state is local to the spawned task (not on context):
/// - `is_first_iteration` — controls dedup of lifecycle events across iterations.
/// - `sequence_number` — monotonically increasing SSE event counter.
/// - `next_output_index` — sequential output item numbering across iterations.
///
/// ## Reads
/// - `ctx.processing.payload` — the JSON body (cloned into the spawned task).
/// - `ctx.processing.upstream_url` — the worker endpoint.
/// - `ctx.input.headers` — forwarded headers.
/// - `ctx.components.mcp_orchestrator` — for tool execution.
///
/// ## Returns (planned)
/// `Ok(StepResult::Response(sse_response))` — the SSE response for the client.
/// Currently returns `Err(501 Not Implemented)`.
pub(crate) async fn stream_request_execution_with_tool(
    _ctx: &mut RequestContext,
) -> Result<StepResult, Response> {
    // TODO: implement streaming with tool interception
    //
    // 1. Create SSE channel: let (tx, rx) = mpsc::unbounded_channel().
    // 2. Clone/move required data from ctx into the spawned task:
    //    - payload, upstream_url, headers, mcp_orchestrator.
    // 3. Spawn a background task that runs:
    //    - Local state: is_first_iteration = true, sequence_number = 0, next_output_index = 0.
    //    - loop {
    //        // ── Stream events from upstream ──────────────────────
    //        //
    //        // a. POST payload to upstream_url with Accept: text/event-stream.
    //        // b. On error: send SSE error event via tx, break.
    //        // c. Create StreamingToolHandler::with_starting_index(next_output_index).
    //        //    If not first iteration, set handler.original_response_id.
    //        // d. Read upstream SSE byte stream. For each event:
    //        //    - Forward: transform, remap output_index, update sequence_number,
    //        //      send via tx. Skip lifecycle events on subsequent iterations.
    //        //    - Buffer: accumulate tool call arguments.
    //        //    - ExecuteTools: forward final event, break inner loop.
    //        // e. Update next_output_index from handler.
    //        // f. Preserve response ID from first iteration.
    //        //
    //        // ── Check for tool calls ─────────────────────────────
    //        //
    //        // g. If no tool calls: send completion event, send [DONE], break.
    //        //
    //        // ── Execute tool calls and resume ────────────────────
    //        //
    //        // h. Execute pending tool calls via mcp_session.
    //        // i. Send tool result SSE events via tx.
    //        // j. Build resume payload, set is_first_iteration = false.
    //        // k. Continue loop.
    //    }
    // 4. Return Ok(StepResult::Response(build_sse_response(rx))).

    Err((
        StatusCode::NOT_IMPLEMENTED,
        "streaming with tool interception not yet implemented",
    )
        .into_response())
}
