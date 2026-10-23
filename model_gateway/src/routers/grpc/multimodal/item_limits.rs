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

use std::collections::HashMap;

use llm_multimodal::Modality;

use super::plan::MediaPlan;
use crate::worker::{RuntimeType, Worker, WorkerRegistry};

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
/// request to request). Empty when none advertises one.
pub(crate) fn engine_item_limits(
    registry: &WorkerRegistry,
    model_id: &str,
) -> HashMap<Modality, EngineItemLimit> {
    let mut limits: HashMap<Modality, EngineItemLimit> = HashMap::new();
    for worker in registry.get_by_model(model_id).iter() {
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
mod tests {
    use std::sync::Arc;

    use llm_multimodal::MediaContentPart;
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, WorkerType};

    const MODEL: &str = "vision-model";

    fn worker(url: &str, runtime: RuntimeType, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new(MODEL))
            .worker_type(WorkerType::Regular)
            .runtime_type(runtime)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            });
        for (key, value) in labels {
            builder = builder.label(*key, *value);
        }
        Arc::new(builder.build())
    }

    fn registry_with(workers: Vec<Arc<dyn Worker>>) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        for worker in workers {
            registry.register(worker).expect("register");
        }
        registry
    }

    fn limits(pairs: &[(Modality, usize)]) -> HashMap<Modality, usize> {
        pairs.iter().copied().collect()
    }

    fn engine(pairs: &[(Modality, usize, RuntimeType)]) -> HashMap<Modality, EngineItemLimit> {
        pairs
            .iter()
            .map(|&(modality, limit, runtime)| (modality, EngineItemLimit { limit, runtime }))
            .collect()
    }

    fn images(count: usize) -> MediaPlan {
        MediaPlan::new((0..count).map(|i| MediaContentPart::ImageUrl {
            url: format!("https://example.com/{i}.png"),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }))
    }

    /// The label parses as the servicers write it; anything else in it is
    /// skipped rather than refused, so a label from a newer servicer still
    /// yields what this router understands.
    #[test]
    fn the_label_parses_as_the_servicers_write_it() {
        assert_eq!(
            parse_item_limits("image=8,video=2"),
            limits(&[(Modality::Image, 8), (Modality::Video, 2)])
        );
        assert_eq!(
            parse_item_limits(" image = 4 ,audio=1,pdf=3,video=x,image_embeds=2,=5,image"),
            limits(&[(Modality::Image, 4), (Modality::Audio, 1)])
        );
        assert!(parse_item_limits("").is_empty());
        assert!(parse_item_limits("image=0").is_empty());
    }

    /// A model's engine limit is the smallest its workers advertise, per
    /// modality, whatever their runtime; a worker without the label does not
    /// count, and a model without any has no engine limit.
    #[test]
    fn a_models_engine_limit_is_the_smallest_its_workers_advertise() {
        let registry = registry_with(vec![
            worker(
                "http://w1:50051",
                RuntimeType::Vllm,
                &[(MM_ITEM_LIMITS_LABEL, "image=8,video=2")],
            ),
            worker(
                "http://w2:50051",
                RuntimeType::Sglang,
                &[(MM_ITEM_LIMITS_LABEL, "image=4")],
            ),
            worker("http://w3:50051", RuntimeType::Vllm, &[]),
        ]);
        let found = engine_item_limits(&registry, MODEL);
        assert_eq!(
            found[&Modality::Image],
            EngineItemLimit {
                limit: 4,
                runtime: RuntimeType::Sglang
            }
        );
        assert_eq!(
            found[&Modality::Video],
            EngineItemLimit {
                limit: 2,
                runtime: RuntimeType::Vllm
            }
        );
        assert!(!found.contains_key(&Modality::Audio));
        assert!(engine_item_limits(&registry, "other-model").is_empty());

        let unlabelled = registry_with(vec![worker("http://w4:50051", RuntimeType::Vllm, &[])]);
        assert!(engine_item_limits(&unlabelled, MODEL).is_empty());
    }

    /// The engine's limit caps the router's own (the flag, the environment
    /// override, the spec's), which keep their precedence below it; without
    /// an engine limit nothing changes.
    #[test]
    fn engine_limits_cap_the_routers_own_limits() {
        use ItemLimitSource::{Engine, Env, Flag, Spec};
        let spec = limits(&[(Modality::Image, 128), (Modality::Video, 8)]);
        let at = |limit, source| ItemLimit { limit, source };
        let no_flag = HashMap::new();
        let vllm = |limit| engine(&[(Modality::Image, limit, RuntimeType::Vllm)]);

        let both = engine(&[
            (Modality::Image, 1, RuntimeType::Vllm),
            (Modality::Video, 2, RuntimeType::Sglang),
        ]);
        let effective = effective_item_limits(&spec, &both, &no_flag, |_| None);
        assert_eq!(
            effective[&Modality::Image],
            at(1, Engine(RuntimeType::Vllm))
        );
        assert_eq!(
            effective[&Modality::Video],
            at(2, Engine(RuntimeType::Sglang))
        );

        let flag = limits(&[(Modality::Image, 3)]);
        let effective = effective_item_limits(&spec, &vllm(8), &flag, |_| None);
        assert_eq!(effective[&Modality::Image], at(3, Flag));
        assert_eq!(effective[&Modality::Video], at(8, Spec));
        let loose_flag = limits(&[(Modality::Image, 16)]);
        let effective = effective_item_limits(&spec, &vllm(8), &loose_flag, |_| None);
        assert_eq!(
            effective[&Modality::Image],
            at(8, Engine(RuntimeType::Vllm))
        );

        let env = |modality| (modality == Modality::Image).then_some(5);
        let effective = effective_item_limits(&spec, &vllm(8), &no_flag, env);
        assert_eq!(effective[&Modality::Image], at(5, Env));
        let effective = effective_item_limits(&spec, &vllm(4), &no_flag, env);
        assert_eq!(
            effective[&Modality::Image],
            at(4, Engine(RuntimeType::Vllm))
        );

        let effective = effective_item_limits(&spec, &HashMap::new(), &no_flag, |_| None);
        assert_eq!(effective[&Modality::Image], at(128, Spec));
        assert_eq!(effective[&Modality::Video], at(8, Spec));
        // A modality the spec does not declare stays undeclared, whatever the
        // engine says about it.
        let audio = engine(&[(Modality::Audio, 1, RuntimeType::Vllm)]);
        let effective = effective_item_limits(&spec, &audio, &no_flag, |_| None);
        assert!(!effective.contains_key(&Modality::Audio));
    }

    /// The refusal names the count, the limit and what set it, with the
    /// engine's own knob for the runtime that advertised the limit.
    #[test]
    fn over_limit_requests_are_refused_naming_the_limit_and_its_origin() {
        let refusal = |source| {
            let limits = HashMap::from([(Modality::Image, ItemLimit { limit: 8, source })]);
            check_item_counts(&images(9), &limits).unwrap_err()
        };
        assert_eq!(
            refusal(ItemLimitSource::Engine(RuntimeType::Vllm)),
            "the request carries 9 image items, above this model's limit of 8 \
             (the engine's --limit-mm-per-prompt)"
        );
        assert_eq!(
            refusal(ItemLimitSource::Engine(RuntimeType::Sglang)),
            "the request carries 9 image items, above this model's limit of 8 \
             (the engine's --limit-mm-data-per-request)"
        );
        assert!(refusal(ItemLimitSource::Engine(RuntimeType::TokenSpeed))
            .ends_with("(the engine's per-request media limit)"));
        assert!(refusal(ItemLimitSource::Flag).ends_with("(--mm-per-request-image-limit)"));
        assert!(refusal(ItemLimitSource::Env).ends_with("(SMG_IMAGE_MAX_COUNT)"));
        assert!(refusal(ItemLimitSource::Spec).ends_with("(the model spec)"));

        let limits = HashMap::from([(
            Modality::Image,
            ItemLimit {
                limit: 8,
                source: ItemLimitSource::Spec,
            },
        )]);
        assert!(check_item_counts(&images(8), &limits).is_ok());
        // A modality without a limit here is the spec's to validate.
        assert!(check_item_counts(&images(9), &HashMap::new()).is_ok());
    }
}
