//! StreamRequestExecution step (no MCP tools — simple passthrough).
//!
//! Terminal step: creates an SSE channel, spawns the streaming work in a
//! background task, and returns the SSE `Response` wrapping the receiver.
//! This follows the same pattern as `process_streaming_response` in the
//! gRPC router.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::gemini::{context::RequestContext, state::StepResult};

/// Open a streaming connection to the upstream worker and forward SSE events
/// directly to the client without tool interception.
///
/// **Not yet implemented** — currently returns 501. Planned behavior:
///
/// Creates an SSE channel (`tx`, `rx`), spawns a background task that
/// streams events from the upstream worker through `tx`, and returns the
/// SSE `Response` wrapping `rx`. Events are forwarded as-is with minimal
/// transformation (metadata patching only).
///
/// ## Reads
/// - `ctx.processing.payload` — the JSON body to stream.
/// - `ctx.processing.upstream_url` — the worker endpoint.
/// - `ctx.input.headers` — forwarded headers.
///
/// ## Returns (planned)
/// `Ok(StepResult::Response(sse_response))` — the SSE response for the client.
/// Currently returns `Err(501 Not Implemented)`.
pub(crate) async fn stream_request_execution(
    _ctx: &mut RequestContext,
) -> Result<StepResult, Response> {
    // TODO: implement simple streaming passthrough
    //
    // 1. Create SSE channel: let (tx, rx) = mpsc::unbounded_channel().
    // 2. Spawn a background task that:
    //    a. POSTs ctx.processing.payload to ctx.processing.upstream_url
    //       with Accept: text/event-stream. Forward headers from ctx.input.headers.
    //    b. On connection error: send SSE error event via tx, return.
    //    c. On non-success HTTP status: send SSE error event via tx, return.
    //    d. Record circuit breaker success.
    //    e. Read upstream SSE byte stream chunk by chunk.
    //    f. For each SSE event:
    //       - Apply minimal transformations (patch store, previous_interaction_id).
    //       - Accumulate response for persistence if store is true.
    //       - Forward event via tx.
    //    g. After stream completes:
    //       - Persist interaction if store is true.
    //       - Send "data: [DONE]\n\n" via tx.
    // 3. Return Ok(StepResult::Response(build_sse_response(rx))).

    Err((
        StatusCode::NOT_IMPLEMENTED,
        "streaming request execution not yet implemented",
    )
        .into_response())
}
