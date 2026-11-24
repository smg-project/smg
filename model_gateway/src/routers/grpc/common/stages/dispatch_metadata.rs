//! Dispatch metadata: per-attempt response metadata derived from the stamped
//! plan and the attempt's worker selection.

use crate::routers::grpc::context::{DispatchMetadata, ExecutionPlan, WorkerSelection};

/// Metadata for one dispatch attempt. `dispatch_model` was captured from the
/// request at the build boundary (already canonical); the weight version
/// comes from the attempt's selected worker ([`weight_version_of`]);
/// `created` is the request's wall-clock second
/// (`DispatchContext::created`), the same for every attempt.
pub(crate) fn prepare_dispatch_metadata(
    plan: &ExecutionPlan,
    dispatch_model: &str,
    workers: Option<&WorkerSelection>,
    created: u64,
) -> DispatchMetadata {
    let weight_version = weight_version_of(workers);

    DispatchMetadata {
        request_id: plan.request_id().to_string(),
        model: dispatch_model.to_string(),
        created,
        weight_version: Some(weight_version),
    }
}

/// The weight version a response states for the worker its answer comes
/// from: the `weight_version` label of the single worker, or of a pair's
/// decode, and `default` where there is no worker or no such label. An
/// attempt reads it where it is prepared, and again where its decode moved
/// behind the prefill. Always inlined: the attempt's preparation makes the
/// reading in place, with no call.
#[inline(always)]
pub(super) fn weight_version_of(workers: Option<&WorkerSelection>) -> String {
    workers
        .map(|w| match w {
            WorkerSelection::Single { worker } => worker.worker(),
            WorkerSelection::Disaggregated { decode, .. } => decode.worker(),
        })
        .and_then(|w| w.metadata().spec.labels.get("weight_version").cloned())
        .unwrap_or_else(|| "default".to_string())
}
