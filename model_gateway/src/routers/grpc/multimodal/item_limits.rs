//! The engine's own per-request media limits, advertised by the workers as
//! the `mm_item_limits` label and held to by the router's media pipeline.
//!
//! An engine's front end refuses a prompt above its per-request media limits
//! (vLLM's `--limit-mm-per-prompt`, SGLang's `--limit-mm-data-per-request`)
//! but never sees the precomputed inputs the router sends, so the router holds
//! a request to them itself: the effective per-modality limit is the router's
//! own (`--mm-per-request-image-limit`, else `SMG_*_MAX_COUNT`, else the model
//! spec's), never above the smallest limit the model's workers advertise. The
//! check runs in preparation, before any fetch and before worker selection, so
//! a model whose workers advertise different limits gets the smallest and a
//! request never depends on which worker is picked.

use std::{collections::HashMap, sync::Arc};

use llm_multimodal::Modality;
use smg_worker::identity::basic::{RuntimeType, Worker};

use super::{capability::media_facts, plan::MediaPlan};
use crate::routers::common::snapshot::WorkerRegistry;

/// Worker label carrying the engine's per-request media limits as
/// `<modality>=<count>` pairs (`image=8,video=2`): vLLM's
/// `GetServerInfo.mm_item_limits`, the `mm_item_limits` key of SGLang's and
/// TokenSpeed's `server_args`.
const MM_ITEM_LIMITS_LABEL: &str = "mm_item_limits";

/// The smallest limit a model's workers advertise for one modality, and the
/// runtime of the worker that set it (which names the engine's knob).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EngineItemLimit {
    pub(crate) limit: usize,
    pub(crate) runtime: RuntimeType,
}

/// Parse one worker's label; a pair that is not `<modality>=<positive count>`
/// of a modality the pipeline counts is skipped.
fn parse_item_limits(label: &str) -> HashMap<Modality, usize> {
    label
        .split(',')
        .filter_map(|pair| {
            let (modality, count) = pair.split_once('=')?;
            let modality = match modality.trim() {
                "image" => Modality::Image,
                "video" => Modality::Video,
                "audio" => Modality::Audio,
                _ => return None,
            };
            let count: usize = count.trim().parse().ok()?;
            (count > 0).then_some((modality, count))
        })
        .collect()
}

/// The limits `worker` advertises; empty without the label.
fn worker_item_limits(worker: &dyn Worker) -> HashMap<Modality, usize> {
    worker
        .metadata()
        .spec
        .labels
        .get(MM_ITEM_LIMITS_LABEL)
        .map(|label| parse_item_limits(label))
        .unwrap_or_default()
}

/// Per modality, the smallest limit any registered worker of `model_id`
/// advertises (healthy or not: a health flap must not move a model's limit
/// request to request). Empty when none advertises one. The model's
/// membership snapshot keeps the answer ([`media_facts`]).
pub(crate) fn engine_item_limits(
    registry: &WorkerRegistry,
    model_id: &str,
) -> HashMap<Modality, EngineItemLimit> {
    media_facts(registry, model_id, |facts| facts.item_limits.clone())
}

/// The rule of [`engine_item_limits`] over the members of one snapshot.
pub(super) fn item_limits_over(workers: &[Arc<dyn Worker>]) -> HashMap<Modality, EngineItemLimit> {
    let mut limits: HashMap<Modality, EngineItemLimit> = HashMap::new();
    for worker in workers {
        let runtime = worker.metadata().spec.runtime_type;
        for (modality, limit) in worker_item_limits(worker.as_ref()) {
            let candidate = EngineItemLimit { limit, runtime };
            limits
                .entry(modality)
                .and_modify(|current| {
                    if candidate.limit < current.limit {
                        *current = candidate;
                    }
                })
                .or_insert(candidate);
        }
    }
    limits
}

/// What set a modality's effective per-request item limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ItemLimitSource {
    /// The smallest limit the model's workers advertise for their engine.
    Engine(RuntimeType),
    /// The router's `--mm-per-request-image-limit`.
    Flag,
    /// `SMG_<MODALITY>_MAX_COUNT` in the router's environment.
    Env,
    /// The model spec's declared limit.
    Spec,
}

impl ItemLimitSource {
    fn describe(self, modality: Modality) -> String {
        match self {
            Self::Engine(RuntimeType::Vllm) => "the engine's --limit-mm-per-prompt".to_string(),
            Self::Engine(RuntimeType::Sglang) => {
                "the engine's --limit-mm-data-per-request".to_string()
            }
            Self::Engine(_) => "the engine's per-request media limit".to_string(),
            Self::Flag => "--mm-per-request-image-limit".to_string(),
            Self::Env => format!(
                "SMG_{}_MAX_COUNT",
                modality.to_string().to_ascii_uppercase()
            ),
            Self::Spec => "the model spec".to_string(),
        }
    }
}

/// A modality's effective per-request item limit and what set it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ItemLimit {
    pub(super) limit: usize,
    pub(super) source: ItemLimitSource,
}

/// The per-request item limit of every modality the spec declares: the
/// router's own limit (the flag's override, else the `SMG_*_MAX_COUNT`
/// environment override, else the spec's), never above the smallest limit the
/// model's workers advertise for their engine. The engine sizes its encoder
/// budget by its limit and its own server refuses above it, so the router must
/// not send it more.
pub(super) fn effective_item_limits(
    spec_limits: &HashMap<Modality, usize>,
    engine_limits: &HashMap<Modality, EngineItemLimit>,
    flag_overrides: &HashMap<Modality, usize>,
    env_override: impl Fn(Modality) -> Option<usize>,
) -> HashMap<Modality, ItemLimit> {
    spec_limits
        .iter()
        .map(|(&modality, &spec_limit)| {
            let own = match (flag_overrides.get(&modality), env_override(modality)) {
                (Some(&limit), _) => ItemLimit {
                    limit,
                    source: ItemLimitSource::Flag,
                },
                (None, Some(limit)) => ItemLimit {
                    limit,
                    source: ItemLimitSource::Env,
                },
                (None, None) => ItemLimit {
                    limit: spec_limit,
                    source: ItemLimitSource::Spec,
                },
            };
            let effective = match engine_limits.get(&modality) {
                Some(engine) if engine.limit <= own.limit => ItemLimit {
                    limit: engine.limit,
                    source: ItemLimitSource::Engine(engine.runtime),
                },
                _ => own,
            };
            (modality, effective)
        })
        .collect()
}

/// The effective numbers alone, for the spec's own validation of the plan.
pub(super) fn limit_numbers(limits: &HashMap<Modality, ItemLimit>) -> HashMap<Modality, usize> {
    limits
        .iter()
        .map(|(&modality, limit)| (modality, limit.limit))
        .collect()
}

/// Refuse a plan carrying more items of a modality than its limit allows,
/// naming the count, the limit and what set it; a modality without a limit
/// here is left to the spec's own validation.
pub(super) fn check_item_counts(
    plan: &MediaPlan,
    limits: &HashMap<Modality, ItemLimit>,
) -> Result<(), String> {
    for &modality in plan.modalities() {
        let count = plan.count(modality);
        let Some(limit) = limits.get(&modality) else {
            continue;
        };
        if count > limit.limit {
            return Err(format!(
                "the request carries {count} {modality} items, above this model's limit of {} ({})",
                limit.limit,
                limit.source.describe(modality)
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/unit/routers/grpc/multimodal/item_limits/tests.rs"]
mod tests;
