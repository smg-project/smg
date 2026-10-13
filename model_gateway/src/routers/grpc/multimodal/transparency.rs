//! The transparency policy of a request's image frames: what the destination
//! engine's own image loader does with an alpha channel before its processor
//! runs, resolved from the runtime of the workers the request can reach.

use llm_multimodal::{TransparencyPolicy, VisionProcessorRegistry};

use crate::worker::{RoutingPool, RuntimeType, WorkerRegistry};

/// The policy for the frames of a request to `model_id` served from the
/// routing pools `pools`: the legs of the pipeline's mode, read as the
/// selection reads them, so the workers resolved here are the workers that
/// can be picked (the image wire dtype is resolved over the same pools).
///
/// Every worker of those pools shares one runtime in a deployment the router
/// can match; that runtime's loader rule applies. A model whose workers in
/// the pools mix runtimes, or has none there, keeps the frame as decoded,
/// which is the behaviour before this policy existed: no engine family
/// changes behaviour without being named below.
pub(crate) fn engine_transparency_policy(
    registry: &WorkerRegistry,
    model_id: &str,
    pools: &[RoutingPool],
) -> TransparencyPolicy {
    // One pool's snapshot after the other, nothing allocated: the shared
    // runtime, or the frame as decoded as soon as two disagree.
    let mut shared: Option<RuntimeType> = None;
    for pool in pools {
        for worker in registry.get_routing_pool(model_id, *pool).iter() {
            let runtime = worker.metadata().spec.runtime_type;
            match shared {
                None => shared = Some(runtime),
                Some(existing) if existing == runtime => {}
                Some(_) => return TransparencyPolicy::KeepAlpha,
            }
        }
    }
    shared.map_or(TransparencyPolicy::KeepAlpha, runtime_transparency_policy)
}

/// The policy the frames of a request to `model_id` decode under: `engine`
/// (the destination fleet's rule), unless the model's processor takes the
/// image as decoded and handles alpha itself
/// ([`VisionPreProcessor::keeps_alpha`](llm_multimodal::VisionPreProcessor::keeps_alpha)).
/// The lookup is the one the preprocessing stage makes for the same request.
pub(crate) fn frame_transparency(
    registry: &VisionProcessorRegistry,
    model_id: &str,
    model_type: Option<&str>,
    engine: TransparencyPolicy,
) -> TransparencyPolicy {
    if registry
        .find(model_id, model_type)
        .is_some_and(|processor| processor.keeps_alpha())
    {
        TransparencyPolicy::KeepAlpha
    } else {
        engine
    }
}

/// What one engine's own image loader does with an alpha channel.
pub(crate) fn runtime_transparency_policy(runtime: RuntimeType) -> TransparencyPolicy {
    match runtime {
        // Its media loader opens images in RGB mode and composites an image
        // with transparency over opaque white with a paste through the alpha
        // mask before the processor sees it (`ImageMediaIO(image_mode="RGB")`
        // and `convert_image_mode` / `rgba_to_rgb` in its multimodal image
        // module). A processor it hands the image as stored opts out through
        // `VisionPreProcessor::keeps_alpha`.
        RuntimeType::Vllm => TransparencyPolicy::CompositeOverWhite,
        // Its processors load images through `_load_single_item` with
        // `discard_alpha_channel=True`, which calls Pillow's `convert("RGB")`:
        // the alpha channel is dropped and every pixel keeps its stored
        // colour. (Its GLM-4V processor picks a background from the image's
        // edges instead; that path is not reproduced here.)
        RuntimeType::Sglang => TransparencyPolicy::DropAlpha,
        // No image loader of its own in this repository: its servicer takes
        // the gateway's tensors, so the frame stays as decoded, unchanged.
        RuntimeType::TokenSpeed => TransparencyPolicy::KeepAlpha,
        // Not measured: the frame stays as decoded, unchanged.
        _ => TransparencyPolicy::KeepAlpha,
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Instant};

    use llm_multimodal::{
        PreProcessorConfig, PreprocessedEncoderInputs, TransformError, VisionPreProcessor,
    };
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{
        routers::grpc::mode::Mode,
        worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, Worker, WorkerType},
    };

    const MODEL: &str = "vision-model";
    /// The regular pipeline's pool, where the test fleets' workers are.
    const REGULAR: &[RoutingPool] = &[RoutingPool::GrpcPipelineRegular];

    fn worker(url: &str, runtime: RuntimeType) -> Arc<dyn Worker> {
        leg(url, WorkerType::Regular, runtime)
    }

    fn leg(url: &str, worker_type: WorkerType, runtime: RuntimeType) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(MODEL))
                .worker_type(worker_type)
                .runtime_type(runtime)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    fn registry_with(workers: Vec<Arc<dyn Worker>>) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        for worker in workers {
            registry.register(worker).expect("register");
        }
        registry
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

    /// Regular workers of one family next to prefill and decode legs of
    /// another: the whole model's workers disagree, the legs a mode's request
    /// can reach do not.
    fn fleet_with_legs() -> WorkerRegistry {
        registry_with(vec![
            leg(
                "grpc://regular-1:8000",
                WorkerType::Regular,
                RuntimeType::Vllm,
            ),
            leg(
                "grpc://regular-2:8000",
                WorkerType::Regular,
                RuntimeType::Vllm,
            ),
            leg(
                "grpc://prefill-1:8000",
                WorkerType::Prefill,
                RuntimeType::Sglang,
            ),
            leg(
                "grpc://decode-1:8000",
                WorkerType::Decode,
                RuntimeType::Sglang,
            ),
        ])
    }

    /// The policy is resolved over the legs a mode's request can reach, as
    /// the wire dtype is: the regular pipeline sees its regular workers, a
    /// disaggregated one its prefill and decode legs; only a mode whose own
    /// legs disagree, or an empty one, keeps the frame as decoded.
    #[test]
    fn the_policy_over_a_modes_pools_is_the_legs_the_request_can_reach() {
        let registry = fleet_with_legs();
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
            "the disaggregated pipeline reaches the SGLang legs only"
        );
        assert_eq!(
            engine_transparency_policy(
                &registry,
                MODEL,
                Mode::EncodePrefillDecode.multimodal_wire_pools()
            ),
            TransparencyPolicy::DropAlpha,
            "no encode leg registered: the prefill and decode legs decide"
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
            "legs of two families keep the frame as decoded"
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
        let registry = fleet_with_legs();
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
        fn calculate_num_tokens(
            &self,
            _: u32,
            _: u32,
            _: &PreProcessorConfig,
        ) -> Result<usize, TransformError> {
            Ok(1)
        }
        fn model_name(&self) -> &'static str {
            "probe"
        }
        fn preprocess(
            &self,
            _: &[image::DynamicImage],
            _: &PreProcessorConfig,
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
        let mut registry = VisionProcessorRegistry::new();
        registry.register("keeps-alpha-model", Box::new(Probe { keeps_alpha: true }));
        for engine in POLICIES {
            assert_eq!(
                frame_transparency(&registry, "keeps-alpha-model", None, engine),
                TransparencyPolicy::KeepAlpha,
                "{engine:?}"
            );
        }
    }

    #[test]
    fn a_processor_without_the_override_follows_the_fleet() {
        let mut registry = VisionProcessorRegistry::new();
        registry.register("plain-model", Box::new(Probe { keeps_alpha: false }));
        for engine in POLICIES {
            assert_eq!(
                frame_transparency(&registry, "plain-model", None, engine),
                engine
            );
            assert_eq!(
                frame_transparency(&registry, "no-such-model", None, engine),
                engine,
                "a model without a processor keeps the fleet's rule"
            );
        }
    }
}
