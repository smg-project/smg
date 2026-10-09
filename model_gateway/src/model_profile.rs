//! Resolve trusted Chat Completions contracts before normalization can discard vendor fields.
use std::sync::Arc;

use axum::{
    extract::{FromRequest, Request},
    response::Response,
};
use openai_protocol::{
    chat::ChatCompletionRequest, profile::ModelProfile, validated::ValidatedJson,
};

use crate::{app_context::AppContext, server::AppState};

/// Resolve registered aliases for metadata only. The public request name and
/// external-provider pinned model IDs are not rewritten here.
pub(crate) fn resolve_model_profile(
    context: &AppContext,
    requested: &str,
) -> (ModelProfile, String) {
    let canonical = context.worker_registry.resolve_model_alias(requested);
    let model = canonical.as_deref().unwrap_or(requested);
    let profile = context
        .router_config
        .model_profiles
        .get(model)
        .copied()
        .unwrap_or_else(|| ModelProfile::for_model(model));
    (profile, model.to_owned())
}

pub(crate) struct ProfiledChatJson(pub ChatCompletionRequest);

impl FromRequest<Arc<AppState>> for ProfiledChatJson {
    type Rejection = Response;

    async fn from_request(req: Request, state: &Arc<AppState>) -> Result<Self, Self::Rejection> {
        let ValidatedJson(body) =
            ValidatedJson::<ChatCompletionRequest>::from_request_with(req, state, |body| {
                let (profile, canonical) = resolve_model_profile(&state.context, &body.model);
                body.resolved_model_profile = Some(profile);
                body.resolved_model_id = Some(canonical);
                // The reserved internal metadata name is not a backend passthrough option.
                body.other.remove("resolved_model_profile");
                body.other.remove("resolved_model_id");
            })
            .await?;
        Ok(Self(body))
    }
}
