//! Dispatch metadata: per-attempt response metadata derived from the stamped
//! plan and the attempt's worker selection.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::routers::grpc::context::{DispatchMetadata, ExecutionPlan, WorkerSelection};

/// Metadata for one dispatch attempt. `dispatch_model` was captured from the
/// request at the build boundary (already canonical); the weight version
/// comes from the attempt's selected worker.
pub(crate) fn prepare_dispatch_metadata(
    plan: &ExecutionPlan,
    dispatch_model: &str,
    workers: Option<&WorkerSelection>,
) -> DispatchMetadata {
    let weight_version = workers
        .map(|w| match w {
            WorkerSelection::Single { worker } => worker,
            WorkerSelection::Disaggregated { decode, .. } => decode,
        })
        .and_then(|w| w.metadata().spec.labels.get("weight_version").cloned())
        .unwrap_or_else(|| "default".to_string());

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    DispatchMetadata {
        request_id: plan.request_id().to_string(),
        model: dispatch_model.to_string(),
        created,
        weight_version: Some(weight_version),
    }
}
