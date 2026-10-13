//! The transparency policy of a request's image frames: what the destination
//! engine's own image loader does with an alpha channel before its processor
//! runs, resolved from the runtime of the model's registered workers.

use llm_multimodal::TransparencyPolicy;

use crate::worker::{RuntimeType, WorkerRegistry};

/// The policy for the frames of a request to `model_id`.
///
/// Every worker of the model shares one runtime in a deployment the router
/// can match; that runtime's loader rule applies. A model whose workers mix
/// runtimes, or has none registered, keeps the frame as decoded, which is the
/// behaviour before this policy existed: no engine family changes behaviour
/// without being named below.
pub(crate) fn engine_transparency_policy(
    registry: &WorkerRegistry,
    model_id: &str,
) -> TransparencyPolicy {
    let workers = registry.get_by_model(model_id);
    let mut runtimes = workers
        .iter()
        .map(|worker| worker.metadata().spec.runtime_type);
    let Some(first) = runtimes.next() else {
        return TransparencyPolicy::KeepAlpha;
    };
    if runtimes.any(|runtime| runtime != first) {
        return TransparencyPolicy::KeepAlpha;
    }
    runtime_transparency_policy(first)
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
    use std::sync::Arc;

    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, Worker, WorkerType};

    const MODEL: &str = "vision-model";

    fn worker(url: &str, runtime: RuntimeType) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(MODEL))
                .worker_type(WorkerType::Regular)
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
            engine_transparency_policy(&registry, MODEL),
            TransparencyPolicy::CompositeOverWhite
        );
    }

    #[test]
    fn an_sglang_fleet_drops_the_alpha_channel() {
        let registry = registry_with(vec![worker("http://w1:8000", RuntimeType::Sglang)]);
        assert_eq!(
            engine_transparency_policy(&registry, MODEL),
            TransparencyPolicy::DropAlpha
        );
    }

    #[test]
    fn a_tokenspeed_fleet_keeps_the_frame_as_decoded() {
        let registry = registry_with(vec![worker("http://w1:8000", RuntimeType::TokenSpeed)]);
        assert_eq!(
            engine_transparency_policy(&registry, MODEL),
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
            engine_transparency_policy(&mixed, MODEL),
            TransparencyPolicy::KeepAlpha
        );
        assert_eq!(
            engine_transparency_policy(&registry_with(vec![]), MODEL),
            TransparencyPolicy::KeepAlpha
        );
        let other_model = registry_with(vec![worker("http://w1:8000", RuntimeType::Vllm)]);
        assert_eq!(
            engine_transparency_policy(&other_model, "another-model"),
            TransparencyPolicy::KeepAlpha
        );
    }
}
