//! The `Embed` RPC: one embedding request to the scheduler, answered by its
//! single finished output, under the frame every wire shares
//! ([`crate::embed::guarded`]: registered so the `Abort` RPC can end it).

use std::sync::Arc;

use smg_grpc_client::sglang_proto as sg;
use tonic::Status;

use super::State;

/// Handle one `Embed` request against the connected scheduler.
pub(super) async fn embed(
    state: &Arc<State>,
    req: sg::EmbedRequest,
) -> Result<sg::EmbedResponse, Status> {
    let client = state.shared.engine()?;
    // A generation scheduler has no pooler to answer with. The Router only
    // sends embeddings to workers advertising `is_generation = false`, so
    // this guards a misrouted request rather than a supported path.
    crate::embed::guarded(
        state,
        req.request_id.clone(),
        state.model.is_generation,
        "This model does not support embeddings",
        || client.embed_sglang(req),
    )
    .await
}
