use std::{sync::Arc, time::Instant};

use llm_multimodal::{PreprocessedEncoderInputs, TransformError, VisionPreProcessor};
use openai_protocol::model_card::ModelCard;
use smg_worker::{
    identity::basic::{ConnectionMode, Worker, WorkerType},
    test_support::no_health_check,
};

use super::*;
use crate::routers::{
    common::snapshot::test_support::registry_with,
    grpc::{backend_client::BasicWorkerBuilder, mode::Mode},
};

const MODEL: &str = "vision-model";
/// The regular pipeline's pool, where the test fleets' workers are.
const REGULAR: &[RoutingPool] = &[RoutingPool::GrpcPipelineRegular];

fn worker(url: &str, runtime: RuntimeType) -> Arc<dyn Worker> {
    worker_of(url, WorkerType::Regular, runtime)
}

fn worker_of(url: &str, worker_type: WorkerType, runtime: RuntimeType) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(MODEL))
            .worker_type(worker_type)
            .runtime_type(runtime)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .build(),
    )
}

/// One engine family per test: what each loader does with alpha.
#[test]
fn a_vllm_fleet_composites_over_white() {
    let registry = registry_with(vec![
        worker("http://w1:8000", RuntimeType::Vllm),
        worker("http://w2:8000", RuntimeType::Vllm),
    ]);
    assert_eq!(
        engine_transparency_policy(&registry, MODEL, REGULAR),
        TransparencyPolicy::CompositeOverWhite
    );
}

#[test]
fn an_sglang_fleet_drops_the_alpha_channel() {
    let registry = registry_with(vec![worker("http://w1:8000", RuntimeType::Sglang)]);
    assert_eq!(
        engine_transparency_policy(&registry, MODEL, REGULAR),
        TransparencyPolicy::DropAlpha
    );
}

#[test]
fn a_tokenspeed_fleet_keeps_the_frame_as_decoded() {
    let registry = registry_with(vec![worker("http://w1:8000", RuntimeType::TokenSpeed)]);
    assert_eq!(
        engine_transparency_policy(&registry, MODEL, REGULAR),
        TransparencyPolicy::KeepAlpha
    );
}

#[test]
fn an_unmeasured_runtime_keeps_the_frame_as_decoded() {
    for runtime in [
        RuntimeType::Trtllm,
        RuntimeType::Mlx,
        RuntimeType::Unspecified,
    ] {
        assert_eq!(
            runtime_transparency_policy(runtime),
            TransparencyPolicy::KeepAlpha,
            "{runtime:?}"
        );
    }
}

#[test]
fn a_mixed_or_empty_fleet_keeps_the_frame_as_decoded() {
    let mixed = registry_with(vec![
        worker("http://w1:8000", RuntimeType::Vllm),
        worker("http://w2:8000", RuntimeType::Sglang),
    ]);
    assert_eq!(
        engine_transparency_policy(&mixed, MODEL, REGULAR),
        TransparencyPolicy::KeepAlpha
    );
    assert_eq!(
        engine_transparency_policy(&registry_with(vec![]), MODEL, REGULAR),
        TransparencyPolicy::KeepAlpha
    );
    let other_model = registry_with(vec![worker("http://w1:8000", RuntimeType::Vllm)]);
    assert_eq!(
        engine_transparency_policy(&other_model, "another-model", REGULAR),
        TransparencyPolicy::KeepAlpha
    );
}

/// Regular workers of one family next to prefill and decode workers of
/// another: the whole model's workers disagree, the pools a mode's request
/// can reach do not.
fn fleet_with_every_role() -> WorkerRegistry {
    registry_with(vec![
        worker_of(
            "grpc://regular-1:8000",
            WorkerType::Regular,
            RuntimeType::Vllm,
        ),
        worker_of(
            "grpc://regular-2:8000",
            WorkerType::Regular,
            RuntimeType::Vllm,
        ),
        worker_of(
            "grpc://prefill-1:8000",
            WorkerType::Prefill,
            RuntimeType::Sglang,
        ),
        worker_of(
            "grpc://decode-1:8000",
            WorkerType::Decode,
            RuntimeType::Sglang,
        ),
    ])
}

/// The policy is resolved over the pools a mode's request can reach, as
/// the wire dtype is: the regular pipeline sees its regular workers, a
/// disaggregated one its prefill and decode pools; only a mode whose own
/// pools disagree, or an empty one, keeps the frame as decoded.
#[test]
fn the_policy_is_resolved_over_the_pools_a_modes_request_can_reach() {
    let registry = fleet_with_every_role();
    assert_eq!(
        engine_transparency_policy(&registry, MODEL, Mode::Regular.multimodal_wire_pools()),
        TransparencyPolicy::CompositeOverWhite,
        "the regular pipeline reaches the vLLM workers only"
    );
    assert_eq!(
        engine_transparency_policy(
            &registry,
            MODEL,
            Mode::PrefillDecode.multimodal_wire_pools()
        ),
        TransparencyPolicy::DropAlpha,
        "the disaggregated pipeline reaches the SGLang workers only"
    );
    assert_eq!(
        engine_transparency_policy(
            &registry,
            MODEL,
            Mode::EncodePrefillDecode.multimodal_wire_pools()
        ),
        TransparencyPolicy::DropAlpha,
        "no encode worker registered: the prefill and decode workers decide"
    );
    assert_eq!(
        engine_transparency_policy(
            &registry,
            MODEL,
            &[
                RoutingPool::GrpcPipelineRegular,
                RoutingPool::GrpcPrefill,
                RoutingPool::GrpcDecode,
            ]
        ),
        TransparencyPolicy::KeepAlpha,
        "pools of two families keep the frame as decoded"
    );
    assert_eq!(
        engine_transparency_policy(&registry, MODEL, &[RoutingPool::GrpcEncode]),
        TransparencyPolicy::KeepAlpha,
        "an empty pool has nothing to agree on"
    );
}

/// The resolver's cost per request over the four-worker fleet, next to
/// the whole-model walk it replaces; a timing, not a check:
/// `cargo test --release -p smg --lib -- resolution_cost --ignored --nocapture`.
#[test]
#[ignore = "timing; run in release with --nocapture"]
#[expect(clippy::print_stderr, reason = "a timing, read from the test output")]
fn resolution_cost_over_pools_and_over_the_model() {
    let registry = fleet_with_every_role();
    let pools = Mode::PrefillDecode.multimodal_wire_pools();
    let rounds = 200_000u32;
    let started = Instant::now();
    let mut sink = 0usize;
    for _ in 0..rounds {
        sink += engine_transparency_policy(&registry, MODEL, pools) as usize;
    }
    let over_pools = started.elapsed().as_nanos() as f64 / f64::from(rounds);
    let started = Instant::now();
    for _ in 0..rounds {
        let workers = registry.get_by_model(MODEL);
        let mut runtimes = workers
            .iter()
            .map(|worker| worker.metadata().spec.runtime_type);
        sink += runtimes
            .next()
            .filter(|first| runtimes.all(|runtime| runtime == *first))
            .map_or(TransparencyPolicy::KeepAlpha, runtime_transparency_policy)
            as usize;
    }
    let over_model = started.elapsed().as_nanos() as f64 / f64::from(rounds);
    eprintln!(
        "RESOLUTION_COST ns/call: over the mode's pools {over_pools:.0}, over the whole model {over_model:.0} (sink {sink})"
    );
}

/// A processor that takes the image as decoded (`keeps_alpha`), or not.
struct Probe {
    keeps_alpha: bool,
}

impl VisionPreProcessor for Probe {
    fn keeps_alpha(&self) -> bool {
        self.keeps_alpha
    }
    fn default_mean(&self) -> [f64; 3] {
        [0.0; 3]
    }
    fn default_std(&self) -> [f64; 3] {
        [1.0; 3]
    }
    fn calculate_num_tokens(&self, _: u32, _: u32) -> Result<usize, TransformError> {
        Ok(1)
    }
    fn model_name(&self) -> &'static str {
        "probe"
    }
    fn preprocess(
        &self,
        _: &[&image::DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        Err(TransformError::ShapeError("probe".to_string()))
    }
}

const POLICIES: [TransparencyPolicy; 3] = [
    TransparencyPolicy::CompositeOverWhite,
    TransparencyPolicy::DropAlpha,
    TransparencyPolicy::KeepAlpha,
];

/// One per policy: a processor that keeps alpha overrides the fleet's rule.
#[test]
fn a_processor_that_keeps_alpha_decodes_as_stored_under_every_fleet() {
    for engine in POLICIES {
        assert_eq!(
            frame_transparency(Some(&Probe { keeps_alpha: true }), engine),
            TransparencyPolicy::KeepAlpha,
            "{engine:?}"
        );
    }
}

#[test]
fn a_processor_without_the_override_follows_the_fleet() {
    for engine in POLICIES {
        assert_eq!(
            frame_transparency(Some(&Probe { keeps_alpha: false }), engine),
            engine
        );
        assert_eq!(
            frame_transparency(None, engine),
            engine,
            "a model without a processor keeps the fleet's rule"
        );
    }
}
