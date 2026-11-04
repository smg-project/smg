//! The frame of the `Embed` RPC every wire that serves it shares: the request
//! id required, the generation runner refused in the engine frontend's own
//! words, and the call registered like a generate stream so an `Abort` (or
//! the drain on shutdown) ends the wait. The wire's module keeps its refusal
//! words and its adapter call.

use std::{future::Future, sync::Arc};

use tonic::Status;

use crate::servicer::WireState;

/// Run `call`, the wire's embed request to the engine, under the frame:
/// `request_id` must be set, a runner for which `is_generation` holds is
/// refused with `refusal` before the engine sees the request, and the call
/// is registered under the id so an abort or the drain ends it (dropping
/// the in-flight call drops its engine stream, which aborts the engine-side
/// request). `call` is built only once the request is admitted.
pub(crate) async fn guarded<S, R, F>(
    state: &Arc<S>,
    request_id: String,
    is_generation: bool,
    refusal: &'static str,
    call: impl FnOnce() -> F,
) -> Result<R, Status>
where
    S: WireState,
    F: Future<Output = Result<R, Status>>,
{
    if request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    if is_generation {
        return Err(Status::invalid_argument(refusal));
    }
    let (_registration, cancel) = state.shared().registry.register(&request_id)?;
    tokio::select! {
        result = call() => result,
        _ = cancel => Err(Status::aborted(format!("embed request {request_id} was aborted"))),
    }
}
