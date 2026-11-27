//! Which third-party provider routers this build carries.
//!
//! The external-router crate lists what was compiled in. A worker that
//! targets a provider is admitted only when one of those routers takes it;
//! nothing decides this at runtime.

use openai_protocol::{
    model_card::ModelCard,
    worker::{ProviderType, RuntimeType, WorkerModels, WorkerSpec},
};
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
/// provider URL. `None` for a self-hosted engine, and for an external
/// endpoint that names no provider, which the OpenAI-compatible router takes
/// as its fallback. The models listing reads a worker's provider here too, so
/// admission and the listing judge a worker alike.
pub(crate) fn provider_of(spec: &WorkerSpec) -> Option<ProviderType> {
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

/// Split discovered model cards into the ones a router in `known` takes and
/// the ones nothing compiled in can route, judged as dispatch will judge
/// them: the card's own provider first, else the worker's.
pub(crate) fn partition_routable_among(
    spec: &WorkerSpec,
    cards: Vec<ModelCard>,
    known: &[ExternalRouterSpec],
) -> (Vec<ModelCard>, Vec<(ModelCard, MissingRouter)>) {
    let default = provider_of(spec);
    let mut routable = Vec::with_capacity(cards.len());
    let mut unroutable = Vec::new();
    for card in cards {
        let provider = card.provider.clone().or_else(|| default.clone());
        match home_among(known, provider.as_ref()) {
            Some(router) if router.compiled => routable.push(card),
            Some(router) => unroutable.push((
                card,
                MissingRouter {
                    label: router.label,
                    feature: router.feature,
                },
            )),
            None => unroutable.push((
                card,
                MissingRouter {
                    label: "external",
                    feature: "providers",
                },
            )),
        }
    }
    (routable, unroutable)
}

/// [`partition_routable_among`] against this build.
pub(crate) fn partition_routable(
    spec: &WorkerSpec,
    cards: Vec<ModelCard>,
) -> (Vec<ModelCard>, Vec<(ModelCard, MissingRouter)>) {
    partition_routable_among(spec, cards, &known::all())
}

/// [`missing_router_among`] against this build.
pub(crate) fn missing_router(spec: &WorkerSpec) -> Option<MissingRouter> {
    missing_router_among(spec, &known::all())
}

#[cfg(test)]
#[path = "../../tests/unit/routers/provider_support/tests.rs"]
mod tests;
