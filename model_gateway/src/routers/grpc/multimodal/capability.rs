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
use axum::response::Response;
use llm_multimodal::Modality;

use super::{plan::MediaPlan, MultimodalIntermediate};
use crate::{
    routers::error,
    worker::{RuntimeType, Worker, WorkerRegistry},
};

/// Worker label carrying the engine's vision-input capability (the vLLM
/// gRPC servicer's `GetModelInfoResponse.supports_vision`). `"false"` on a
/// multimodal architecture means the engine runs `--language-model-only`:
/// no vision encoder, encoder-cache budget 0, so no multimodal payload may
/// reach it. A multimodal model reports `"true"`; a text-only model also
/// reports `"false"`. The label is tri-state: only the vLLM servicer sets
/// the field on purpose, so for the other runtimes a proto-default `false`
/// is not turned into a label at all (`ModelInfo::to_labels`), and an
/// absent label reads as unknown, never as a refusal.
pub(crate) const SUPPORTS_VISION_LABEL: &str = "supports_vision";

/// Whether the worker's engine accepts no multimodal inputs at all (a vLLM
/// `--language-model-only` worker). Absent label (non-vLLM runtimes, older
/// servicers) reads as multimodal-capable so behavior stays as today.
pub(crate) fn worker_language_model_only(worker: &dyn Worker) -> bool {
    worker
        .metadata()
        .spec
        .labels
        .get(SUPPORTS_VISION_LABEL)
        .is_some_and(|value| value == "false")
}

/// Whether every worker registered for `model_id` runs without a vision
/// encoder (`supports_vision=false`, see [`SUPPORTS_VISION_LABEL`]). Such a
/// pool takes no multimodal payload at all: a vLLM engine admits the request
/// and never schedules it (encoder-cache budget 0), so a request carrying
/// media must be refused before any of it is decoded, preprocessed or
/// dispatched. An empty pool reads as capable: worker selection answers for a
/// model with no workers.
pub(crate) fn model_workers_language_model_only(registry: &WorkerRegistry, model_id: &str) -> bool {
    let workers = registry.get_by_model(model_id);
    !workers.is_empty()
        && workers
            .iter()
            .all(|worker| worker_language_model_only(worker.as_ref()))
}

/// Refuse a request's media when every worker of `model_id` is language model
/// only: the same prompt 400 the public API gives image input to a model
/// without vision, instead of a dispatch the engine never schedules. Runs in
/// preparation, before the media is decoded or preprocessed. A plan without
/// media passes; so does a model with no registered workers (selection
/// answers for it).
pub(crate) fn ensure_model_accepts_media(
    registry: &WorkerRegistry,
    model_id: &str,
    plan: &MediaPlan,
) -> Result<(), Response> {
    if plan.is_empty() || !model_workers_language_model_only(registry, model_id) {
        return Ok(());
    }
    let modalities = plan
        .modalities()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    Err(error::bad_request(
        "multimodal_not_supported",
        format!(
            "{modalities} input is not supported by model '{model_id}': its workers run as a \
             language model only, without a vision encoder"
        ),
    ))
}

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
                encoder_input: ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0])
                    .unwrap()
                    .into(),
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

    mod pool {
        use std::sync::Arc;

        use llm_multimodal::MediaContentPart;
        use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

        use super::*;
        use crate::worker::{BasicWorkerBuilder, ConnectionMode, WorkerType};

        const MODEL: &str = "pool-test-model";

        fn worker(url: &str, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
            let mut builder = BasicWorkerBuilder::new(url)
                .model(ModelCard::new(MODEL))
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::Vllm)
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

        fn image_plan() -> MediaPlan {
            MediaPlan::new([MediaContentPart::ImageUrl {
                url: "data:image/png;base64,AAAA".to_string(),
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            }])
        }

        /// Only a pool whose every worker says `supports_vision=false` is
        /// language model only; an absent label keeps today's reading
        /// (capable), and an empty pool is left to worker selection.
        #[test]
        fn pool_is_language_model_only_when_every_worker_says_so() {
            let lmo = |url| worker(url, &[(SUPPORTS_VISION_LABEL, "false")]);
            let vision = |url| worker(url, &[(SUPPORTS_VISION_LABEL, "true")]);
            let unlabeled = |url| worker(url, &[]);

            let all_lmo = registry_with(vec![lmo("grpc://10.0.0.1:1"), lmo("grpc://10.0.0.2:1")]);
            assert!(model_workers_language_model_only(&all_lmo, MODEL));

            let mixed = registry_with(vec![lmo("grpc://10.0.0.1:1"), vision("grpc://10.0.0.2:1")]);
            assert!(!model_workers_language_model_only(&mixed, MODEL));

            let legacy = registry_with(vec![
                lmo("grpc://10.0.0.1:1"),
                unlabeled("grpc://10.0.0.2:1"),
            ]);
            assert!(!model_workers_language_model_only(&legacy, MODEL));

            let empty = registry_with(vec![]);
            assert!(!model_workers_language_model_only(&empty, MODEL));
            assert!(!model_workers_language_model_only(
                &all_lmo,
                "another-model"
            ));
        }

        /// Media to a language-model-only pool is a 400 naming the modality
        /// and the model; text-only requests and capable pools pass.
        #[test]
        fn media_to_a_language_model_only_pool_is_a_bad_request() {
            let all_lmo = registry_with(vec![worker(
                "grpc://10.0.0.1:1",
                &[(SUPPORTS_VISION_LABEL, "false")],
            )]);
            let response = ensure_model_accepts_media(&all_lmo, MODEL, &image_plan()).unwrap_err();
            assert_eq!(response.status(), http::StatusCode::BAD_REQUEST);
            assert_eq!(
                response.headers()[error::HEADER_X_SMG_ERROR_CODE],
                "multimodal_not_supported"
            );

            assert!(ensure_model_accepts_media(&all_lmo, MODEL, &MediaPlan::default()).is_ok());

            let vision = registry_with(vec![worker(
                "grpc://10.0.0.1:1",
                &[(SUPPORTS_VISION_LABEL, "true")],
            )]);
            assert!(ensure_model_accepts_media(&vision, MODEL, &image_plan()).is_ok());
        }
    }
}
