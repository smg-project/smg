//! Which third-party provider routers this build carries.
//!
//! The external-router crate lists what was compiled in. A worker that
//! targets a provider is admitted only when one of those routers takes it;
//! nothing decides this at runtime.

use openai_protocol::worker::{ProviderType, RuntimeType, WorkerModels, WorkerSpec};
use smg_external_router::{home_among, known, ExternalRouterSpec};

/// The router a provider target needs but this build lacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MissingRouter {
    /// Human name for messages.
    pub label: &'static str,
    /// The Cargo feature that compiles the router in.
    pub feature: &'static str,
}

/// The provider a worker spec targets: an explicit `provider`, else a known
/// provider URL. `None` for a self-hosted engine.
fn provider_of(spec: &WorkerSpec) -> Option<ProviderType> {
    spec.provider
        .clone()
        .or_else(|| ProviderType::from_url(&spec.url))
}

/// Whether `spec` describes a third-party provider rather than a self-hosted
/// engine: an explicit external runtime, a provider, or a known provider URL.
fn targets_provider(spec: &WorkerSpec) -> bool {
    spec.runtime_type == RuntimeType::External || provider_of(spec).is_some()
}

/// Every provider a worker's traffic can be dispatched under, judged the way
/// the dispatcher judges it: a model's own provider first, else the worker's.
/// Only the listed models are ever dispatched to a worker that lists any, so
/// the worker's own provider counts alone when it serves every model.
fn providers_needed(spec: &WorkerSpec) -> Vec<Option<ProviderType>> {
    let default = provider_of(spec);
    let cards: &[_] = match &spec.models {
        WorkerModels::Wildcard => &[],
        WorkerModels::Single(card) => std::slice::from_ref(card.as_ref()),
        WorkerModels::Multi(cards) => cards,
    };
    if cards.is_empty() {
        return vec![default];
    }
    let mut needed: Vec<Option<ProviderType>> = cards
        .iter()
        .map(|card| card.provider.clone().or_else(|| default.clone()))
        .collect();
    needed.dedup();
    needed
}

/// The router `spec` needs that this build lacks, judged over `known`, or
/// `None` when the spec is not a provider target or every router it can reach
/// is compiled in. A router that is known but not compiled names its feature.
pub(crate) fn missing_router_among(
    spec: &WorkerSpec,
    known: &[ExternalRouterSpec],
) -> Option<MissingRouter> {
    if !targets_provider(spec) {
        return None;
    }
    providers_needed(spec).into_iter().find_map(|provider| {
        match home_among(known, provider.as_ref()) {
            Some(router) if router.compiled => None,
            Some(router) => Some(MissingRouter {
                label: router.label,
                feature: router.feature,
            }),
            None => Some(MissingRouter {
                label: "external",
                feature: "providers",
            }),
        }
    })
}

/// [`missing_router_among`] against this build.
pub(crate) fn missing_router(spec: &WorkerSpec) -> Option<MissingRouter> {
    missing_router_among(spec, &known::all())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn spec(value: serde_json::Value) -> WorkerSpec {
        serde_json::from_value(value).expect("worker spec")
    }

    fn with_compiled(backends: &[&str]) -> Vec<ExternalRouterSpec> {
        known::all()
            .into_iter()
            .map(|mut spec| {
                spec.compiled = backends.contains(&spec.backend);
                spec
            })
            .collect()
    }

    fn anthropic_only() -> Vec<ExternalRouterSpec> {
        with_compiled(&["anthropic"])
    }

    fn everything() -> Vec<ExternalRouterSpec> {
        with_compiled(&["openai", "anthropic", "gemini"])
    }

    #[test]
    fn self_hosted_engines_need_no_provider_router() {
        let local = spec(json!({"url": "http://10.0.0.5:8000"}));
        assert_eq!(missing_router_among(&local, &with_compiled(&[])), None);
        let sglang = spec(json!({"url": "grpc://10.0.0.5:8000", "runtime_type": "sglang"}));
        assert_eq!(missing_router_among(&sglang, &with_compiled(&[])), None);
    }

    #[test]
    fn a_provider_target_is_matched_to_the_router_it_would_reach() {
        let anthropic = spec(json!({"url": "https://api.anthropic.com"}));
        assert_eq!(missing_router_among(&anthropic, &anthropic_only()), None);
        assert_eq!(
            missing_router_among(&anthropic, &with_compiled(&[])).map(|m| m.feature),
            Some("provider-anthropic")
        );

        // xAI, an explicit external runtime with an unknown host, and any
        // custom provider all ride the OpenAI-compatible router.
        for value in [
            json!({"url": "https://api.x.ai"}),
            json!({"url": "https://llm.internal:8443", "runtime_type": "external"}),
            json!({"url": "https://llm.internal:8443", "provider": "together"}),
        ] {
            let worker = spec(value);
            assert_eq!(
                missing_router_among(&worker, &anthropic_only()).map(|m| m.feature),
                Some("provider-openai")
            );
            assert_eq!(missing_router_among(&worker, &everything()), None);
        }

        let gemini = spec(json!({"url": "https://generativelanguage.googleapis.com"}));
        assert_eq!(
            missing_router_among(&gemini, &anthropic_only()).map(|m| m.label),
            Some("Gemini")
        );
    }

    #[test]
    fn a_model_that_names_its_own_provider_is_judged_by_it() {
        // A proxy on a private host: the worker says nothing about its
        // provider, the model card does. Dispatch keys on the card.
        let proxied = spec(json!({
            "url": "https://llm.internal:8443",
            "runtime_type": "external",
            "models": [{"id": "claude-3-5-sonnet", "provider": "anthropic"}]
        }));
        assert_eq!(missing_router_among(&proxied, &anthropic_only()), None);
        let openai_only = with_compiled(&["openai"]);
        assert_eq!(
            missing_router_among(&proxied, &openai_only).map(|m| m.feature),
            Some("provider-anthropic")
        );

        // A worker serving models of two providers needs both routers.
        let mixed = spec(json!({
            "url": "https://llm.internal:8443",
            "runtime_type": "external",
            "models": [
                {"id": "gpt-4o"},
                {"id": "gemini-2.5-pro", "provider": "gemini"}
            ]
        }));
        assert_eq!(
            missing_router_among(&mixed, &openai_only).map(|m| m.label),
            Some("Gemini")
        );
        assert_eq!(missing_router_among(&mixed, &everything()), None);
    }
}
