use std::sync::Arc;

use super::{super::plan::tests::image_urls, *};
use crate::routers::common::snapshot::test_support::{grpc_worker, registry_with};

const MODEL: &str = "vision-model";

fn worker(url: &str, runtime: RuntimeType, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
    grpc_worker(url, MODEL, runtime, labels)
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
        check_item_counts(&image_urls(9), &limits).unwrap_err()
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
    assert!(check_item_counts(&image_urls(8), &limits).is_ok());
    // A modality without a limit here is the spec's to validate.
    assert!(check_item_counts(&image_urls(9), &HashMap::new()).is_ok());
}
