//! Backend x modality capability: the single source of truth for which gRPC
//! engine supports which input modality.
//!
//! Previously this was implicit and duplicated across assembly (per-backend
//! `into_single_image_batch` / `into_single_vision_batch` / ad-hoc `bail!`s with
//! divergent messages). Centralizing it here lets the pipeline reject an
//! unsupported (engine, modality) request at worker selection -- once the
//! runtime is known, before request building assembles the payload -- with one
//! consistent message, and lets assembly assert against the same matrix as
//! defense in depth.

use anyhow::Result;
use llm_multimodal::Modality;

use super::MultimodalIntermediate;
use crate::worker::RuntimeType;

/// Whether `runtime` accepts multimodal inputs of `modality`.
///
/// This is the authoritative capability matrix. Truth, transcribed from the
/// per-backend assembly arms:
/// - Image: SGLang, vLLM, TRT-LLM, TokenSpeed
/// - ImageEmbeds: TokenSpeed only (pre-computed embeddings are an EPD feature)
/// - Video: vLLM, TokenSpeed
/// - Audio: TokenSpeed
/// - MLX: none
///
/// `Modality::ImageEmbeds` cannot actually reach the early check today because a
/// [`crate::routers::grpc::multimodal::MediaBatch`] only ever yields
/// Image/Video/Audio; it is kept in the matrix for correctness/defense in depth.
pub(crate) fn runtime_supports_modality(runtime: RuntimeType, modality: Modality) -> bool {
    match modality {
        Modality::Image => matches!(
            runtime,
            RuntimeType::Sglang | RuntimeType::Vllm | RuntimeType::Trtllm | RuntimeType::TokenSpeed
        ),
        Modality::ImageEmbeds => matches!(runtime, RuntimeType::TokenSpeed),
        Modality::Video => matches!(runtime, RuntimeType::Vllm | RuntimeType::TokenSpeed),
        Modality::Audio => matches!(runtime, RuntimeType::TokenSpeed),
    }
}

/// Reject early if the selected backend does not support every modality present
/// in the request. Runs at worker selection, once the runtime is known but
/// before request building assembles the payload, so an unsupported combination
/// fails fast with one clear message instead of dying deep in assembly.
pub(crate) fn ensure_backend_supports_modalities(
    runtime: RuntimeType,
    intermediate: &MultimodalIntermediate,
) -> Result<()> {
    for batch in intermediate.batches() {
        let modality = batch.media.modality();
        anyhow::ensure!(
            runtime_supports_modality(runtime, modality),
            "backend {runtime} does not support {modality} inputs"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full backend x modality matrix, mirroring the pre-refactor assembly
    /// dispatch so a behavior change surfaces here.
    #[test]
    fn capability_matrix_matches_backend_support() {
        use Modality::{Audio, Image, ImageEmbeds, Video};
        use RuntimeType::{Mlx, Sglang, TokenSpeed, Trtllm, Vllm};

        // (runtime, modality, expected_supported)
        let cases = [
            (Sglang, Image, true),
            (Sglang, Video, false),
            (Sglang, Audio, false),
            (Vllm, Image, true),
            (Vllm, Video, true),
            (Vllm, Audio, false),
            (Trtllm, Image, true),
            (Trtllm, Video, false),
            (Trtllm, Audio, false),
            (TokenSpeed, Image, true),
            (TokenSpeed, Video, true),
            (TokenSpeed, Audio, true),
            (Mlx, Image, false),
            (Mlx, Video, false),
            (Mlx, Audio, false),
            // ImageEmbeds is a TokenSpeed-only EPD feature.
            (Sglang, ImageEmbeds, false),
            (Vllm, ImageEmbeds, false),
            (Trtllm, ImageEmbeds, false),
            (TokenSpeed, ImageEmbeds, true),
            (Mlx, ImageEmbeds, false),
        ];

        for (runtime, modality, expected) in cases {
            assert_eq!(
                runtime_supports_modality(runtime, modality),
                expected,
                "runtime={runtime} modality={modality} expected supported={expected}"
            );
        }
    }

    /// Non-gRPC runtimes are never routed to the multimodal gRPC path; they
    /// support nothing here.
    #[test]
    fn non_grpc_runtimes_support_no_modality() {
        for runtime in [RuntimeType::Unspecified, RuntimeType::External] {
            for modality in [Modality::Image, Modality::Video, Modality::Audio] {
                assert!(!runtime_supports_modality(runtime, modality));
            }
        }
    }

    fn single_image_intermediate() -> MultimodalIntermediate {
        use std::{collections::HashMap, sync::Arc};

        use llm_multimodal::{
            EncoderFieldLayouts, ImageDetail, ImageFrame, ImageSource, PlaceholderRange,
            PreprocessedEncoderInputs,
        };
        use ndarray::{ArrayD, IxDyn};

        use super::super::{MediaBatch, PrecomputedMultimodalIntermediate, PromptBinding};

        MultimodalIntermediate::try_new(vec![PrecomputedMultimodalIntermediate {
            preprocessed: PreprocessedEncoderInputs {
                encoder_input: ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap(),
                feature_token_counts: vec![1],
                item_sizes: vec![(1, 1)],
                model_specific: HashMap::new(),
            },
            media: MediaBatch::Images(vec![Arc::new(ImageFrame::new(
                image::DynamicImage::new_rgb8(1, 1),
                bytes::Bytes::from_static(b"image"),
                ImageDetail::Auto,
                ImageSource::InlineBytes,
                "image-hash".to_string(),
            ))]),
            bindings: vec![PromptBinding {
                item_index: 0,
                prompt_ordinal: 0,
                structural: PlaceholderRange {
                    offset: 0,
                    length: 1,
                },
                patches: vec![],
            }],
            placeholder_token_id: Some(10),
            field_layouts: EncoderFieldLayouts::default(),
            keep_on_cpu_keys: vec![],
            encoder_input_key: None,
        }])
        .unwrap()
    }

    /// The early check accepts a supported (backend, modality) pair and rejects
    /// an unsupported one with the single consistent message.
    #[test]
    fn early_check_gates_backend_modality() {
        let intermediate = single_image_intermediate();

        // Image on SGLang is supported.
        assert!(ensure_backend_supports_modalities(RuntimeType::Sglang, &intermediate).is_ok());

        // Image on MLX is not.
        let err = ensure_backend_supports_modalities(RuntimeType::Mlx, &intermediate).unwrap_err();
        assert_eq!(err.to_string(), "backend mlx does not support image inputs");
    }
}
