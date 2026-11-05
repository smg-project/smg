//! The `Embed` RPC: one pooling request, answered when the engine's single
//! finished output arrives, under the frame every wire shares
//! ([`crate::embed::guarded`]: registered so the `Abort` RPC can end it).

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
    let client = state.shared.engine()?;
    let pooler = PoolerDefaults {
        use_activation: state.model.pooler_use_activation,
        dimensions: state.model.pooler_dimensions,
    };
    // A generation runner serves no pooling task; vLLM's frontend refuses
    // this before the engine sees it, in these words.
    crate::embed::guarded(
        state,
        req.request_id.clone(),
        state.model.is_generation,
        "This model does not support pooling",
        || client.embed(req, pooler),
    )
    .await
}
