//! WorkerSelection step.
//!
//! Transition: SelectWorker → LoadPreviousInteraction

use axum::response::Response;

use crate::{
    routers::{
        common::worker_selection::{SelectWorkerRequest, WorkerSelector},
        error,
        gemini::{
            context::RequestContext,
            state::{RequestState, StepResult},
        },
    },
    worker::ProviderType,
};

/// Select a healthy upstream worker for the requested model.
pub(crate) async fn worker_selection(ctx: &mut RequestContext) -> Result<StepResult, Response> {
    let model = ctx
        .input
        .model_id
        .as_deref()
        .or(ctx.input.original_request.model.as_deref())
        .or(ctx.input.original_request.agent.as_deref());

    let model = match model {
        Some(m) if !m.is_empty() => m,
        _ => {
            return Err(error::bad_request(
                "invalid_request",
                "No model identifier provided in request".to_string(),
            ));
        }
    };

    let selector = WorkerSelector::new(&ctx.components.worker_registry);
    let worker = selector
        .select_worker(&SelectWorkerRequest {
            model_id: model,
            headers: ctx.input.headers.as_ref(),
            provider: Some(ProviderType::Gemini),
            ..Default::default()
        })
        .await?;

    ctx.processing.upstream_url = Some(format!("{}/v1beta/interactions", worker.url()));
    ctx.processing.worker = Some(worker);
    ctx.state = RequestState::LoadPreviousInteraction;

    Ok(StepResult::Continue)
}
