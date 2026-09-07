//! WorkerSelection step.
//!
//! Transition: SelectWorker → LoadPreviousInteraction

use axum::response::Response;

use crate::{
    error,
    gemini::{
        context::RequestContext,
        state::{RequestState, StepResult},
    },
    known,
    worker::SelectWorkerRequest,
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

    let worker = ctx
        .components
        .workers
        .select(&SelectWorkerRequest {
            model_id: model,
            headers: ctx.input.headers.as_ref(),
            router: Some(known::GEMINI),
            ..Default::default()
        })
        .await?;

    ctx.processing.upstream_url = Some(format!("{}/v1beta/interactions", worker.url()));
    ctx.processing.worker = Some(worker);
    ctx.state = RequestState::LoadPreviousInteraction;

    Ok(StepResult::Continue)
}
