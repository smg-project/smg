//! Resolve trusted Chat Completions contracts before normalization can discard vendor fields.
use std::sync::Arc;

use axum::{
    extract::{FromRequest, Request},
    response::Response,
};
use openai_protocol::{
    chat::ChatCompletionRequest, model_card::ModelCard, profile::ModelProfile,
    validated::ValidatedJson,
};

use crate::{app_context::AppContext, routers::error, server::AppState, worker::Worker};

/// Precise architecture/type metadata wins; model names remain the legacy fallback.
fn profile_for_card(worker: &dyn Worker, card: &ModelCard, single_model: bool) -> ModelProfile {
    if card
        .architectures
        .iter()
        .any(|a| a == "KimiK3ForConditionalGeneration")
        || card.hf_model_type.as_deref() == Some("kimi_k3")
    {
        return ModelProfile::KimiK3;
    }
    let profile = ModelProfile::for_model(&card.id);
    // A worker-level path identifies only a single-model worker. Never apply
    // one model's identity to a sibling card on a multi-model worker.
    if profile == ModelProfile::OpenAi && single_model {
        if let Some(path) = worker.metadata().spec.labels.get("model_path") {
            return ModelProfile::for_model(path);
        }
    }
    profile
}

pub(crate) fn worker_requires_chat_profile(worker: &dyn Worker) -> bool {
    let cards = worker.models();
    if cards.is_empty() {
        return ModelProfile::for_model(worker.model_id()) != ModelProfile::OpenAi;
    }
    cards
        .iter()
        .any(|card| profile_for_card(worker, card, cards.len() == 1) != ModelProfile::OpenAi)
}

/// Resolve only the requested card, including aliases. All registered replicas
/// must agree so validation cannot depend on which replica is selected later.
fn resolve_model_profile(context: &AppContext, requested: &str) -> Result<ModelProfile, Response> {
    let canonical = context.worker_registry.resolve_model_alias(requested);
    let model = canonical.as_deref().unwrap_or(requested);
    let mut resolved = None;
    for worker in context.worker_registry.get_by_model(model).iter() {
        let cards = worker.models();
        let candidate = if let Some(card) = cards.iter().find(|card| card.matches(model)) {
            profile_for_card(worker.as_ref(), card, cards.len() == 1)
        } else if cards.is_empty() {
            ModelProfile::for_model(worker.model_id())
        } else {
            continue;
        };
        if resolved.is_some_and(|current| current != candidate) {
            return Err(error::service_unavailable(
                "model_profile_conflict",
                format!("Workers serving '{model}' advertise conflicting model contracts"),
            ));
        }
        resolved = Some(candidate);
    }
    Ok(resolved.unwrap_or_else(|| ModelProfile::for_model(model)))
}

pub(crate) struct ProfiledChatJson(pub ChatCompletionRequest);

impl FromRequest<Arc<AppState>> for ProfiledChatJson {
    type Rejection = Response;

    async fn from_request(req: Request, state: &Arc<AppState>) -> Result<Self, Self::Rejection> {
        let ValidatedJson(body) =
            ValidatedJson::<ChatCompletionRequest>::from_request_with(req, state, |body| {
                body.resolved_model_profile =
                    Some(resolve_model_profile(&state.context, &body.model)?);
                // Reserved internal names are never backend passthrough options.
                body.other.remove("resolved_model_profile");
                Ok(())
            })
            .await?;
        Ok(Self(body))
    }
}
