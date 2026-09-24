//! Single state machine driver for the Gemini Interactions router.
//!
//! One function, one loop, one match. Dispatches the appropriate step
//! based on `ctx.state` and handles the result.

use axum::response::Response;

use super::{
    context::RequestContext,
    state::{RequestState, StepResult},
    steps,
};

/// Execute the state machine to completion and return the final `Response`.
///
/// For **non-streaming** requests the `ResponseProcessing` step produces the
/// HTTP response via `StepResult::Response`.
///
/// For **streaming** requests the SSE response is built by the router before
/// spawning this function; the streaming steps send events through
/// `ctx.streaming.sse_tx` and return `StepResult::Response` when done.
pub(crate) async fn execute(ctx: &mut RequestContext) -> Response {
    loop {
        let result = match ctx.state {
            RequestState::SelectWorker => steps::worker_selection(ctx).await,

            RequestState::LoadPreviousInteraction => steps::previous_interaction_loading(ctx).await,

            RequestState::BuildRequest => steps::request_building(ctx).await,

            RequestState::NonStreamRequest => steps::non_stream_request_execution(ctx).await,

            RequestState::ProcessResponse => steps::response_processing(ctx).await,

            RequestState::StreamRequestWithTool => {
                steps::stream_request_execution_with_tool(ctx).await
            }

            RequestState::StreamRequest => steps::stream_request_execution(ctx).await,
        };

        match result {
            Ok(StepResult::Continue) => continue,
            Ok(StepResult::Response(resp)) => return resp,
            Err(resp) => return resp,
        }
    }
}
