//! Dispatch metadata: per-attempt response metadata derived from the stamped
//! plan and the attempt's worker selection.

use crate::routers::grpc::context::{DispatchMetadata, ExecutionPlan, WorkerSelection};

/// Metadata for one dispatch attempt. `dispatch_model` was captured from the
/// request at the build boundary (already canonical); the weight version
/// comes from the attempt's selected worker; `created` is the request's
/// wall-clock second (`DispatchContext::created`), the same for every
/// attempt.
pub(crate) fn prepare_dispatch_metadata(
    plan: &ExecutionPlan,
    dispatch_model: &str,
    workers: Option<&WorkerSelection>,
    created: u64,
) -> DispatchMetadata {
    let weight_version = workers
        .map(|w| match w {
            WorkerSelection::Single { worker } => worker,
            WorkerSelection::Disaggregated { decode, .. } => decode,
        })
        .and_then(|w| w.metadata().spec.labels.get("weight_version").cloned())
        .unwrap_or_else(|| "default".to_string());

    DispatchMetadata {
        request_id: plan.request_id().to_string(),
        model: dispatch_model.to_string(),
        created,
        weight_version: Some(weight_version),
    }
}
