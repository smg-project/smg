//! The `Embed` RPC: one embedding request to the scheduler, answered by its
//! single finished output, and registered so the `Abort` RPC can end it.

use std::sync::Arc;

use smg_grpc_client::sglang_proto as sg;
use tonic::Status;

use super::State;

/// Handle one `Embed` request against the connected scheduler.
pub(super) async fn embed(
    state: &Arc<State>,
    req: sg::EmbedRequest,
) -> Result<sg::EmbedResponse, Status> {
    let client = state.engine()?;
    if req.request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    // A generation scheduler has no pooler to answer with. The Router only
    // sends embeddings to workers advertising `is_generation = false`, so
    // this guards a misrouted request rather than a supported path.
    if state.model.is_generation {
        return Err(Status::invalid_argument(
            "This model does not support embeddings",
        ));
    }
    // Registered like a generate stream, so an `Abort` (or the drain on
    // shutdown) ends the wait: dropping the in-flight call drops its engine
    // stream, which aborts the scheduler-side request.
    let (_registration, cancel) = state.registry.register(&req.request_id)?;
    let request_id = req.request_id.clone();
    tokio::select! {
        result = client.embed_sglang(req) => result,
        _ = cancel => Err(Status::aborted(format!("embed request {request_id} was aborted"))),
    }
}
