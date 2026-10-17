//! Dispatch metadata: per-attempt response metadata derived from the stamped
//! plan and the attempt's worker selection.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    middleware::TenantKey,
    routers::grpc::context::{DispatchMetadata, ExecutionPlan, WorkerSelection},
};

/// Metadata for one dispatch attempt. `dispatch_model` was captured from the
/// request at the build boundary (already canonical); the weight version
/// comes from the attempt's selected worker; `tenant_key` is the caller's,
/// for the usage row the response is booked to, and `unbilled_prompt_tokens`
/// the stub the rendering added to the prompt, which that row leaves out as
/// the usage block the client reads does.
pub(crate) fn prepare_dispatch_metadata(
    plan: &ExecutionPlan,
    dispatch_model: &str,
    workers: Option<&WorkerSelection>,
    tenant_key: Option<TenantKey>,
    unbilled_prompt_tokens: u32,
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
        tenant_key,
        prompt_tokens: u32::try_from(plan.prompt_len())
            .unwrap_or(u32::MAX)
            .saturating_sub(unbilled_prompt_tokens),
    }
}
