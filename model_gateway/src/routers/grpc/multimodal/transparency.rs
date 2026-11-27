//! The transparency policy of a request's image frames: what the destination
//! engine's own image loader does with an alpha channel before its processor
//! runs, resolved from the runtime of the workers the request can reach.

use llm_multimodal::{TransparencyPolicy, VisionPreProcessor};
use smg_worker::identity::basic::RuntimeType;

use crate::routers::common::snapshot::{RoutingPool, WorkerRegistry, WorkerRegistryExt};

/// The policy for the frames of a request to `model_id` served from the
/// routing pools `pools`: the pools of the pipeline's mode, read as the
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

/// The policy the frames of a request decode under: `engine` (the destination
/// fleet's rule), unless the model's image processor takes the image as
/// decoded and handles alpha itself ([`VisionPreProcessor::keeps_alpha`]).
/// The processor is the one the preprocessing stage runs for the same request.
pub(crate) fn frame_transparency(
    processor: Option<&dyn VisionPreProcessor>,
    engine: TransparencyPolicy,
) -> TransparencyPolicy {
    if processor.is_some_and(|processor| processor.keeps_alpha()) {
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
#[path = "../../../../tests/unit/routers/grpc/multimodal/transparency/tests.rs"]
mod tests;
