//! The `Embed` RPC: one pooling request, answered when the engine's single
//! finished output arrives, and registered so the `Abort` RPC can end it.

use std::sync::Arc;

use engine_zmq_adapter::PoolerDefaults;
use smg_grpc_client::vllm_proto as vllm;
use tonic::Status;

use super::State;

/// Handle one `Embed` request against the connected engine.
pub(super) async fn embed(
    state: &Arc<State>,
    req: vllm::EmbedRequest,
) -> Result<vllm::EmbedResponse, Status> {
    let client = state.engine()?;
    if req.request_id.is_empty() {
        return Err(Status::invalid_argument("request_id is required"));
    }
    // A generation runner serves no pooling task; vLLM's frontend refuses
    // this before the engine sees it, in these words.
    if state.model.is_generation {
        return Err(Status::invalid_argument(
            "This model does not support pooling",
        ));
    }
    let pooler = PoolerDefaults {
        use_activation: state.model.pooler_use_activation,
        dimensions: state.model.pooler_dimensions,
    };
    // Registered like a generate stream, so an `Abort` (or the drain on
    // shutdown) ends the wait: dropping the in-flight call drops its engine
    // stream, which aborts the engine-side request.
    let (_registration, cancel) = state.registry.register(&req.request_id)?;
    let request_id = req.request_id.clone();
    tokio::select! {
        result = client.embed(req, pooler) => result,
        _ = cancel => Err(Status::aborted(format!("embed request {request_id} was aborted"))),
    }
}
