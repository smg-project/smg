//! Dispatch metadata stage: Prepare metadata for dispatch

use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::PipelineStage;
use crate::routers::{
    error,
    grpc::context::{
        ConstrainedDecodingMode, DispatchMetadata, RequestContext, RequestType, WorkerSelection,
    },
};

/// Dispatch metadata stage: Prepare metadata for dispatch
pub(crate) struct DispatchMetadataStage;

#[async_trait]
impl PipelineStage for DispatchMetadataStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let execution_plan = ctx.state.execution_plan.as_ref().ok_or_else(|| {
            error!(
                function = "DispatchMetadataStage::execute",
                "Execution plan not built"
            );
            error::internal_error("execution_plan_not_built", "Execution plan not built")
        })?;

        let request_id = execution_plan.request_id().to_string();
        // The model the response reports. `RequestContext::new` already
        // rewrote every one of these to the canonical model ID, so a request
        // that arrived under an alias is answered under the canonical name.
        let model = match &ctx.input.request_type {
            RequestType::Chat(req) => req.model.clone(),
            RequestType::Completion(req) => req.model.clone(),
            // `GenerateRequest` carries a model field too, but callers of the
            // native `/generate` route may leave it empty, so prefer the
            // model the router resolved.
            RequestType::Generate(_req) => ctx.input.model_id.clone(),
            RequestType::Responses(req) => req.model.clone(),
            RequestType::Embedding(req) => req.model.clone(),
            RequestType::Classify(req) => req.model.clone(),
            RequestType::Messages(req) => req.model.clone(),
        };

        // For PD disaggregation the decode leg generates the tokens, so its
        // labels describe the completion (weight version, decode behavior).
        let generating_worker = ctx.state.workers.as_ref().map(|w| match w {
            WorkerSelection::Single { worker } => worker,
            WorkerSelection::Disaggregated { decode, .. } => decode,
        });

        let weight_version = generating_worker
            .and_then(|w| w.metadata().spec.labels.get("weight_version").cloned())
            .unwrap_or_else(|| "default".to_string());

        let constrained_decoding_mode = generating_worker
            .and_then(|w| w.metadata().spec.labels.get("constrained_decoding_mode"))
            .and_then(|value| ConstrainedDecodingMode::from_label(value));

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        ctx.state.dispatch = Some(DispatchMetadata {
            request_id,
            model,
            created,
            weight_version: Some(weight_version),
            constrained_decoding_mode,
        });

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "DispatchMetadata"
    }
}
