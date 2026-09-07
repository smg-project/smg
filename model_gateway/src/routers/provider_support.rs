//! Which third-party provider routers this build carries.
//!
//! Each provider router sits behind its own Cargo feature. IGW mode registers
//! every router that was compiled in, and a worker that targets a provider is
//! admitted only when the router its traffic would reach exists in this
//! build; nothing decides this at runtime.

use openai_protocol::worker::{ProviderType, RuntimeType, WorkerModels, WorkerSpec};

/// The provider routers a build carries, one flag per feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Compiled {
    pub openai: bool,
    pub anthropic: bool,
    pub gemini: bool,
}

/// This build's set.
pub(crate) const COMPILED: Compiled = Compiled {
    openai: cfg!(feature = "provider-openai"),
    anthropic: cfg!(feature = "provider-anthropic"),
    gemini: cfg!(feature = "provider-gemini"),
};

/// The router family a provider's traffic reaches, mirroring the manager's
/// dispatch: Anthropic and Gemini have routers of their own; every other
/// provider rides the OpenAI-compatible router.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProviderFamily {
    OpenAiCompatible,
    Anthropic,
    Gemini,
}

impl ProviderFamily {
    pub(crate) fn of(provider: Option<&ProviderType>) -> Self {
        match provider {
            Some(ProviderType::Anthropic) => Self::Anthropic,
            Some(ProviderType::Gemini) => Self::Gemini,
            _ => Self::OpenAiCompatible,
        }
    }

    /// Human name for messages.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::OpenAiCompatible => "OpenAI-compatible",
            Self::Anthropic => "Anthropic",
            Self::Gemini => "Gemini",
        }
    }

    /// The Cargo feature that compiles this family's router in.
    pub(crate) fn feature(self) -> &'static str {
        match self {
            Self::OpenAiCompatible => "provider-openai",
            Self::Anthropic => "provider-anthropic",
            Self::Gemini => "provider-gemini",
        }
    }

    pub(crate) fn compiled_in(self, compiled: Compiled) -> bool {
        match self {
            Self::OpenAiCompatible => compiled.openai,
            Self::Anthropic => compiled.anthropic,
            Self::Gemini => compiled.gemini,
        }
    }
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
/// Empty for a self-hosted engine.
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

/// The family whose router `spec` needs but `compiled` lacks, or `None` when
/// the spec is not a provider target or every router it can reach is present.
pub(crate) fn missing_router_in(spec: &WorkerSpec, compiled: Compiled) -> Option<ProviderFamily> {
    if !targets_provider(spec) {
        return None;
    }
    providers_needed(spec)
        .into_iter()
        .map(|provider| ProviderFamily::of(provider.as_ref()))
        .find(|family| !family.compiled_in(compiled))
}

/// [`missing_router_in`] against this build.
pub(crate) fn missing_router(spec: &WorkerSpec) -> Option<ProviderFamily> {
    missing_router_in(spec, COMPILED)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn spec(value: serde_json::Value) -> WorkerSpec {
        serde_json::from_value(value).expect("worker spec")
    }

    const NONE: Compiled = Compiled {
        openai: false,
        anthropic: false,
        gemini: false,
    };
    const ANTHROPIC_ONLY: Compiled = Compiled {
        openai: false,
        anthropic: true,
        gemini: false,
    };

    #[test]
    fn self_hosted_engines_need_no_provider_router() {
        let local = spec(json!({"url": "http://10.0.0.5:8000"}));
        assert_eq!(missing_router_in(&local, NONE), None);
        let sglang = spec(json!({"url": "grpc://10.0.0.5:8000", "runtime_type": "sglang"}));
        assert_eq!(missing_router_in(&sglang, NONE), None);
    }

    #[test]
    fn a_provider_target_is_matched_to_the_router_it_would_reach() {
        let anthropic = spec(json!({"url": "https://api.anthropic.com"}));
        assert_eq!(missing_router_in(&anthropic, ANTHROPIC_ONLY), None);
        assert_eq!(
            missing_router_in(&anthropic, NONE),
            Some(ProviderFamily::Anthropic)
        );

        // xAI, an explicit external runtime with an unknown host, and any
        // custom provider all ride the OpenAI-compatible router.
        for value in [
            json!({"url": "https://api.x.ai"}),
            json!({"url": "https://llm.internal:8443", "runtime_type": "external"}),
            json!({"url": "https://llm.internal:8443", "provider": "together"}),
        ] {
            assert_eq!(
                missing_router_in(&spec(value), ANTHROPIC_ONLY),
                Some(ProviderFamily::OpenAiCompatible)
            );
        }

        let gemini = spec(json!({"url": "https://generativelanguage.googleapis.com"}));
        assert_eq!(
            missing_router_in(&gemini, ANTHROPIC_ONLY),
            Some(ProviderFamily::Gemini)
        );
        assert_eq!(ProviderFamily::Gemini.feature(), "provider-gemini");
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
        assert_eq!(missing_router_in(&proxied, ANTHROPIC_ONLY), None);
        let openai_only = Compiled {
            openai: true,
            anthropic: false,
            gemini: false,
        };
        assert_eq!(
            missing_router_in(&proxied, openai_only),
            Some(ProviderFamily::Anthropic)
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
            missing_router_in(&mixed, openai_only),
            Some(ProviderFamily::Gemini)
        );
    }
}
