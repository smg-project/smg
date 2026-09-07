//! Which third-party provider routers this build carries.
//!
//! The external-router crate lists what was compiled in. A worker that
//! targets a provider is admitted only when one of those routers takes it;
//! nothing decides this at runtime.

use openai_protocol::worker::{ProviderType, RuntimeType, WorkerModels, WorkerSpec};
use smg_external_router::{builtin_routers, ExternalRouterSpec};

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

/// The router a worker of `provider` reaches, named whether or not it is
/// compiled in: Anthropic and Gemini have routers of their own; every other
/// provider rides the OpenAI-compatible router.
fn needed(provider: Option<&ProviderType>) -> MissingRouter {
    match provider {
        Some(ProviderType::Anthropic) => MissingRouter {
            label: "Anthropic",
            feature: "provider-anthropic",
        },
        Some(ProviderType::Gemini) => MissingRouter {
            label: "Gemini",
            feature: "provider-gemini",
        },
        _ => MissingRouter {
            label: "OpenAI-compatible",
            feature: "provider-openai",
        },
    }
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

/// The router `spec` needs that none of `routers` provides, or `None` when
/// the spec is not a provider target or every router it can reach is there.
pub(crate) fn missing_router_among(
    spec: &WorkerSpec,
    routers: &[ExternalRouterSpec],
) -> Option<MissingRouter> {
    if !targets_provider(spec) {
        return None;
    }
    providers_needed(spec)
        .into_iter()
        .find(|provider| !routers.iter().any(|r| r.takes(provider.as_ref())))
        .map(|provider| needed(provider.as_ref()))
}

/// [`missing_router_among`] against this build.
pub(crate) fn missing_router(spec: &WorkerSpec) -> Option<MissingRouter> {
    missing_router_among(spec, &builtin_routers())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use smg_external_router::{BuildFuture, ExternalContext};

    use super::*;

    fn spec(value: serde_json::Value) -> WorkerSpec {
        serde_json::from_value(value).expect("worker spec")
    }

    fn never(_ctx: ExternalContext) -> BuildFuture {
        Box::pin(async { Err("not built in tests".to_string()) })
    }

    fn router(
        backend: &'static str,
        serves: fn(&ProviderType) -> bool,
        fallback: bool,
    ) -> ExternalRouterSpec {
        ExternalRouterSpec {
            router_id: backend,
            backend,
            label: backend,
            feature: backend,
            serves,
            fallback,
            build: never,
        }
    }

    fn anthropic_only() -> Vec<ExternalRouterSpec> {
        vec![router(
            "anthropic",
            |p| matches!(p, ProviderType::Anthropic),
            false,
        )]
    }

    fn everything() -> Vec<ExternalRouterSpec> {
        vec![
            router(
                "openai",
                |p| !matches!(p, ProviderType::Anthropic | ProviderType::Gemini),
                true,
            ),
            router("anthropic", |p| matches!(p, ProviderType::Anthropic), false),
            router("gemini", |p| matches!(p, ProviderType::Gemini), false),
        ]
    }

    #[test]
    fn self_hosted_engines_need_no_provider_router() {
        let local = spec(json!({"url": "http://10.0.0.5:8000"}));
        assert_eq!(missing_router_among(&local, &[]), None);
        let sglang = spec(json!({"url": "grpc://10.0.0.5:8000", "runtime_type": "sglang"}));
        assert_eq!(missing_router_among(&sglang, &[]), None);
    }

    #[test]
    fn a_provider_target_is_matched_to_the_router_it_would_reach() {
        let anthropic = spec(json!({"url": "https://api.anthropic.com"}));
        assert_eq!(missing_router_among(&anthropic, &anthropic_only()), None);
        assert_eq!(
            missing_router_among(&anthropic, &[]).map(|m| m.feature),
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
        let openai_only = vec![router(
            "openai",
            |p| !matches!(p, ProviderType::Anthropic | ProviderType::Gemini),
            true,
        )];
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
