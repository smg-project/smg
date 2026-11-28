use std::sync::Arc;

use llm_multimodal::PreProcessorConfig;
use llm_tokenizer::mock::MockTokenizer;
use smg_worker::identity::basic::RuntimeType;

use super::{
    super::{
        capability::fleet_reads_format,
        config::{MultimodalConfigRegistry, MultimodalModelConfig},
        settings::MultimodalSettings,
    },
    *,
};
use crate::routers::common::snapshot::WorkerRegistry;

/// The tokens a spec resolves for a modality, as `(text, id)` pairs, the
/// anchor first.
type Tokens = &'static [(&'static str, u32)];

/// `count` image parts, each at its own URL.
pub(crate) fn image_urls(count: usize) -> MediaPlan {
    MediaPlan::new((0..count).map(|i| MediaContentPart::ImageUrl {
        url: format!("https://example.com/{i}.png").into(),
        detail: None,
        uuid: None,
        max_long_side_pixel: None,
    }))
}

/// `count` fetchable parts of `modality`.
fn media_urls(modality: Modality, count: usize) -> MediaPlan {
    match modality {
        Modality::Video => MediaPlan::new((0..count).map(|i| MediaContentPart::VideoUrl {
            url: format!("https://example.com/{i}.mp4").into(),
            uuid: None,
            fps: None,
            max_long_side_pixel: None,
        })),
        Modality::Audio => MediaPlan::new((0..count).map(|i| MediaContentPart::AudioUrl {
            url: format!("https://example.com/{i}.wav").into(),
            uuid: None,
        })),
        Modality::Image | Modality::ImageEmbeds => image_urls(count),
    }
}

/// A three-modality checkpoint's config, its token ids nested as the
/// family's checkpoints nest them.
fn omni_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3_omni_moe",
        "thinker_config": {
            "audio_token_id": 151675,
            "image_token_id": 151655,
            "video_token_id": 151656
        }
    })
}

/// A Gemma 4 checkpoint's config: the image anchor with its begin and end
/// markers, and the video anchor.
fn gemma4_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma4",
        "image_token_id": 258880,
        "boi_token_id": 255999,
        "eoi_token_id": 258882,
        "video_token_id": 258884
    })
}

/// A Nemotron Omni checkpoint's config: the geometry the family's
/// processors implement and the sound encoder's parameters.
fn nemotron_omni_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
        "patch_size": 16,
        "downsample_ratio": 0.5,
        "norm_mean": [0.48145466, 0.4578275, 0.40821073],
        "norm_std": [0.26862954, 0.26130258, 0.27577711],
        "vision_config": {
            "args": {"min_num_patches": 1024, "max_num_patches": 13312},
            "video_target_num_patches": 1024,
            "video_temporal_patch_size": 2,
            "video_maintain_aspect_ratio": true
        },
        "sound_config": {"sampling_rate": 16000, "num_mel_bins": 128, "subsampling_factor": 8}
    })
}

/// The limit the model's workers advertise for their engine refuses a plan
/// above it before any placeholder work, naming the limit and the engine's
/// knob; without one the same plan goes on into the spec's own resolution.
#[tokio::test]
async fn placeholder_preparation_holds_the_plan_to_the_engines_limits() {
    let components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &MultimodalSettings::default(),
    )
    .unwrap();
    components.config_registry.insert(
        "tok".to_string(),
        Arc::new(MultimodalModelConfig::new(
            serde_json::json!({"model_type": "llava", "image_token_index": 32000}),
            PreProcessorConfig::default(),
            None,
        )),
    );
    let tokenizer = MockTokenizer::new();
    let plan = image_urls(2);
    let engine = HashMap::from([(
        Modality::Image,
        EngineItemLimit {
            limit: 1,
            runtime: RuntimeType::Vllm,
        },
    )]);
    let error = prepare_placeholder_tokens_with_env(
        &plan,
        "org/llava",
        &tokenizer,
        &components,
        "tok",
        "unused",
        &engine,
        |_| true,
        |_| None,
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("2 image items")
            && error.contains("limit of 1")
            && error.contains("the engine's --limit-mm-per-prompt"),
        "{error}"
    );
    // Without an engine limit the plan passes the count check and reaches
    // the placeholder resolution (which the mock tokenizer, lacking the
    // token, then refuses).
    let error = prepare_placeholder_tokens_with_env(
        &plan,
        "org/llava",
        &tokenizer,
        &components,
        "tok",
        "unused",
        &HashMap::new(),
        |_| true,
        |_| None,
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(!error.contains("limit of"), "{error}");
}

/// The specs that declare images without a count of their own leave the
/// bound to the engine's advertised limit and the router's knobs: for
/// each family, with neither a request above the count the spec used to
/// declare goes through, and the flag and the engine's limit still refuse
/// it, each naming itself. A row's tokens are the ones the spec resolves
/// for images, the anchor first.
#[tokio::test]
async fn the_specs_without_an_image_count_of_their_own_leave_the_bound_to_the_engine_and_the_knobs()
{
    let families: [(&str, Tokens, serde_json::Value, usize); 9] = [
        (
            "org/qwen3-vl",
            &[("<|image_pad|>", 151655)],
            serde_json::json!({"model_type": "qwen3_vl", "image_token_id": 151655}),
            10,
        ),
        (
            "org/qwen2-vl",
            &[("<|image_pad|>", 151655)],
            serde_json::json!({"model_type": "qwen2_vl", "image_token_id": 151655}),
            10,
        ),
        (
            "org/inkling",
            &[("<|unused_200054|>", 200054)],
            serde_json::json!({
                "model_type": "inkling_mm_model",
                "vision_config": {"decoder_dmodel": 6144}
            }),
            10,
        ),
        (
            "org/muse-spark",
            &[("<|patch|>", 200092)],
            serde_json::json!({"model_type": "muse_spark", "image_token_id": 200092}),
            10,
        ),
        (
            "org/kimi-k3",
            &[("<|media_pad|>", 163605)],
            serde_json::json!({"model_type": "kimi_k3", "media_placeholder_token_id": 163605}),
            10,
        ),
        (
            "org/qwen3-omni",
            &[("<|image_pad|>", 151655)],
            omni_config(),
            10,
        ),
        (
            "org/gemma-4",
            &[("<|image|>", 258880)],
            gemma4_config(),
            128,
        ),
        (
            "org/nemotron-omni",
            &[("<image>", 18), ("<img>", 19), ("</img>", 20)],
            nemotron_omni_config(),
            16,
        ),
        (
            "org/deepseek-v4.1",
            &[("<｜deepseek_image｜>", 129264)],
            serde_json::json!({"model_type": "deepseek_v41", "image_token_id": 129264}),
            128,
        ),
    ];
    for (model_id, tokens, config, cap) in families {
        let (anchor, _) = tokens[0];
        let registry = Arc::new(MultimodalConfigRegistry::new());
        registry.insert(
            "tok".to_string(),
            Arc::new(MultimodalModelConfig::new(
                config,
                PreProcessorConfig::default(),
                None,
            )),
        );
        let tokenizer = MockTokenizer::new().with_tokens(tokens);
        let plan = image_urls(cap + 1);
        let prepare = |image_limit, engine: HashMap<Modality, EngineItemLimit>| {
            let components = MultimodalComponents::new(
                registry.clone(),
                image_limit,
                None,
                &MultimodalSettings::default(),
            )
            .unwrap();
            let plan = &plan;
            let tokenizer = &tokenizer;
            async move {
                prepare_placeholder_tokens_with_env(
                    plan,
                    model_id,
                    tokenizer,
                    &components,
                    "tok",
                    "unused",
                    &engine,
                    |_| true,
                    |_| None,
                )
                .await
            }
        };

        let placeholders = prepare(None, HashMap::new())
            .await
            .unwrap_or_else(|error| panic!("{model_id}: no engine limit and no knob: {error}"));
        assert_eq!(
            placeholders.get(Modality::Image),
            Some(anchor),
            "{model_id}"
        );

        let error = prepare(Some(cap), HashMap::new())
            .await
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(&format!("{} image items", cap + 1))
                && error.contains(&format!("limit of {cap}"))
                && error.contains("(--mm-per-request-image-limit)"),
            "{model_id}: {error}"
        );

        let engine = HashMap::from([(
            Modality::Image,
            EngineItemLimit {
                limit: 8,
                runtime: RuntimeType::Vllm,
            },
        )]);
        let error = prepare(None, engine).await.unwrap_err().to_string();
        assert!(
            error.contains("limit of 8") && error.contains("the engine's"),
            "{model_id}: {error}"
        );
    }
}

/// The specs that declare video or audio without a count of their own
/// leave the bound to the engine's advertised limit and the router's
/// environment override: for each family and modality, with neither a
/// request above the count the spec used to declare goes through and
/// resolves the family's placeholder, and the override and the engine's
/// limit still refuse it, each naming itself. A row's tokens are the ones
/// the spec resolves for the modality, the anchor first.
#[tokio::test]
async fn the_specs_without_a_video_or_audio_count_of_their_own_leave_the_bound_to_the_engine_and_the_env(
) {
    let rows: [(&str, Modality, Tokens, serde_json::Value, usize); 11] = [
        (
            "org/qwen3-vl",
            Modality::Video,
            &[("<|video_pad|>", 151656)],
            serde_json::json!({
                "model_type": "qwen3_vl",
                "image_token_id": 151655,
                "video_token_id": 151656
            }),
            1,
        ),
        (
            "org/qwen2.5-vl",
            Modality::Video,
            &[("<|video_pad|>", 151656)],
            serde_json::json!({
                "model_type": "qwen2_5_vl",
                "image_token_id": 151655,
                "video_token_id": 151656
            }),
            1,
        ),
        (
            "org/muse-spark",
            Modality::Video,
            &[("<|video|>", 200091)],
            serde_json::json!({
                "model_type": "muse_spark",
                "image_token_id": 200092,
                "video_token_id": 200091
            }),
            1,
        ),
        (
            "org/muse-spark",
            Modality::Audio,
            &[("<|audio|>", 200038)],
            serde_json::json!({
                "model_type": "muse_spark",
                "image_token_id": 200092,
                "audio_token_id": 200038,
                "audio_config": {"model_type": "muse_spark_audio", "patch_sizes": [16, 6, 5, 4]}
            }),
            10,
        ),
        (
            "org/inkling",
            Modality::Audio,
            &[("<|unused_200053|>", 200053)],
            serde_json::json!({
                "model_type": "inkling_mm_model",
                "audio_config": {"decoder_dmodel": 6144}
            }),
            10,
        ),
        (
            "org/qwen3-omni",
            Modality::Video,
            &[("<|video_pad|>", 151656)],
            omni_config(),
            1,
        ),
        (
            "org/qwen3-omni",
            Modality::Audio,
            &[("<|audio_pad|>", 151675)],
            omni_config(),
            10,
        ),
        (
            "org/qwen3-asr",
            Modality::Audio,
            &[("<|audio_pad|>", 151676)],
            serde_json::json!({
                "model_type": "qwen3_asr",
                "thinker_config": {"audio_token_id": 151676}
            }),
            10,
        ),
        (
            "org/gemma-4",
            Modality::Video,
            &[("<|video|>", 258884)],
            gemma4_config(),
            8,
        ),
        (
            "org/nemotron-omni",
            Modality::Video,
            &[
                ("<SPECIAL_30>", 30),
                ("<image>", 18),
                ("<img>", 19),
                ("</img>", 20),
            ],
            nemotron_omni_config(),
            2,
        ),
        (
            "org/nemotron-omni",
            Modality::Audio,
            &[("<so_embedding>", 27), ("<so_start>", 28), ("<so_end>", 29)],
            nemotron_omni_config(),
            4,
        ),
    ];
    for (model_id, modality, tokens, config, cap) in rows {
        let (anchor, _) = tokens[0];
        let registry = Arc::new(MultimodalConfigRegistry::new());
        registry.insert(
            "tok".to_string(),
            Arc::new(MultimodalModelConfig::new(
                config,
                PreProcessorConfig::default(),
                None,
            )),
        );
        let tokenizer = MockTokenizer::new().with_tokens(tokens);
        let components =
            MultimodalComponents::new(registry, None, None, &MultimodalSettings::default())
                .unwrap();
        let plan = media_urls(modality, cap + 1);
        // `env` stands in for the modality's `SMG_<MODALITY>_MAX_COUNT`.
        let prepare = |env: Option<usize>, engine: HashMap<Modality, EngineItemLimit>| {
            let plan = &plan;
            let tokenizer = &tokenizer;
            let components = &components;
            async move {
                prepare_placeholder_tokens_with_env(
                    plan,
                    model_id,
                    tokenizer,
                    components,
                    "tok",
                    "unused",
                    &engine,
                    |_| true,
                    move |requested| if requested == modality { env } else { None },
                )
                .await
            }
        };

        let placeholders = prepare(None, HashMap::new()).await.unwrap_or_else(|error| {
            panic!("{model_id} {modality}: no engine limit and no override: {error}")
        });
        assert_eq!(
            placeholders.get(modality),
            Some(anchor),
            "{model_id} {modality}"
        );

        let error = prepare(Some(cap), HashMap::new())
            .await
            .unwrap_err()
            .to_string();
        let variable = format!(
            "SMG_{}_MAX_COUNT",
            modality.to_string().to_ascii_uppercase()
        );
        assert!(
            error.contains(&format!("{} {modality} items", cap + 1))
                && error.contains(&format!("limit of {cap}"))
                && error.contains(&variable),
            "{model_id} {modality}: {error}"
        );

        // The engine whose servicer advertises this modality's limit.
        let runtime = if modality == Modality::Audio {
            RuntimeType::Sglang
        } else {
            RuntimeType::Vllm
        };
        let engine = HashMap::from([(
            modality,
            EngineItemLimit {
                limit: cap,
                runtime,
            },
        )]);
        let error = prepare(None, engine).await.unwrap_err().to_string();
        assert!(
            error.contains(&format!("limit of {cap}")) && error.contains("the engine's"),
            "{model_id} {modality}: {error}"
        );
    }
}

#[test]
fn media_plan_preserves_media_order_and_counts() {
    let plan = MediaPlan::new([
        MediaContentPart::Text {
            text: "ignored".to_string(),
        },
        MediaContentPart::AudioUrl {
            url: "audio".to_string().into(),
            uuid: None,
        },
        MediaContentPart::ImageUrl {
            url: "image".to_string().into(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        },
        MediaContentPart::AudioUrl {
            url: "audio-2".to_string().into(),
            uuid: None,
        },
    ]);

    assert_eq!(plan.modalities(), &[Modality::Audio, Modality::Image]);
    assert_eq!(plan.count(Modality::Audio), 2);
    assert_eq!(plan.count(Modality::Image), 1);

    let parts = plan.into_parts();
    assert_eq!(parts.len(), 3);
    assert!(matches!(
        &parts[0],
        MediaContentPart::AudioUrl { url, .. } if url.as_str() == "audio"
    ));
    assert!(matches!(
        &parts[1],
        MediaContentPart::ImageUrl { url, .. } if url.as_str() == "image"
    ));
    assert!(matches!(
        &parts[2],
        MediaContentPart::AudioUrl { url, .. } if url.as_str() == "audio-2"
    ));
}

/// An image part at `url`.
fn image_at(url: &str) -> MediaContentPart {
    MediaContentPart::ImageUrl {
        url: url.to_string().into(),
        detail: None,
        uuid: None,
        max_long_side_pixel: None,
    }
}

/// An image part that carries its bytes.
fn image_bytes() -> MediaContentPart {
    MediaContentPart::ImageData {
        data: vec![1, 2, 3],
        mime_type: Some("image/png".to_string()),
        uuid: None,
        detail: None,
    }
}

/// An audio part at `url`.
fn audio_at(url: &str) -> MediaContentPart {
    MediaContentPart::AudioUrl {
        url: url.to_string().into(),
        uuid: None,
    }
}

/// A plan's source class is `none` without media, `inline` for `data:`
/// URLs and carried bytes, `remote` for fetched URLs and `hybrid` for a
/// mix, read over every modality: the class is the request's.
#[test]
fn a_plans_source_class_is_none_inline_remote_or_hybrid_over_every_modality() {
    assert_eq!(MediaPlan::new([]).source_class(), "none");
    assert_eq!(
        MediaPlan::new([
            image_at("data:image/png;base64,aGVsbG8="),
            image_bytes(),
            audio_at("data:audio/wav;base64,aGVsbG8="),
        ])
        .source_class(),
        "inline",
        "a data: URL and carried bytes, over two modalities"
    );
    assert_eq!(
        MediaPlan::new([
            image_at("https://example.com/a.png"),
            audio_at("https://example.com/a.wav"),
        ])
        .source_class(),
        "remote",
        "fetched URLs alone, over two modalities"
    );
    assert_eq!(
        MediaPlan::new([image_at("https://example.com/a.png"), image_bytes()]).source_class(),
        "hybrid",
        "one of each in one modality"
    );
    assert_eq!(
        MediaPlan::new([image_bytes(), audio_at("https://example.com/a.wav")]).source_class(),
        "hybrid",
        "inline images beside a fetched clip: the class is not the images' alone"
    );
}

/// The images a request is observed with are the plan's image items:
/// text is no media, and a part of another modality changes the class,
/// not the count.
#[test]
fn a_plan_counts_its_images_apart_from_its_source_class() {
    let text = || MediaContentPart::Text {
        text: "hi".to_string(),
    };

    let plan = MediaPlan::new([image_bytes(), text()]);
    assert_eq!(plan.count(Modality::Image), 1);
    assert_eq!(plan.source_class(), "inline");

    let plan = MediaPlan::new([image_bytes(), audio_at("https://example.com/a.wav")]);
    assert_eq!(plan.count(Modality::Image), 1);
    assert_eq!(plan.source_class(), "hybrid");

    let plan = MediaPlan::new([text()]);
    assert_eq!(plan.count(Modality::Image), 0);
    assert_eq!(plan.source_class(), "none");
}

#[test]
fn rendered_anchor_validation_is_exact_per_modality() {
    let plan = MediaPlan::new([
        MediaContentPart::ImageUrl {
            url: "image".to_string().into(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        },
        MediaContentPart::AudioUrl {
            url: "audio".to_string().into(),
            uuid: None,
        },
    ]);
    let mut placeholders = PlaceholderTokens::default();
    placeholders.insert(Modality::Image, "<|im_start|>".to_string());
    placeholders.insert(Modality::Audio, "<|im_end|>".to_string());
    let tokenizer = MockTokenizer::new();

    validate_rendered_media_anchors(&plan, &placeholders, &tokenizer, &[1001, 7, 1002]).unwrap();

    let error =
        validate_rendered_media_anchors(&plan, &placeholders, &tokenizer, &[1001, 1001, 1002])
            .unwrap_err();
    assert!(error.to_string().contains("image anchor count mismatch"));
}

#[test]
fn rendered_markers_become_their_anchor_in_text_prompts() {
    let mut placeholders = PlaceholderTokens::default();
    placeholders.add_rendered_marker("<video>".to_string(), "<SPECIAL_30>".to_string());
    let text = "a<video>\nb <video><audio>";
    assert_eq!(
        anchor_rendered_markers(
            Some(&placeholders),
            &PromptEncoding::FromText,
            text.to_string()
        ),
        "a<SPECIAL_30>\nb <SPECIAL_30><audio>"
    );
    // A text-only request carries no placeholders.
    assert_eq!(
        anchor_rendered_markers(None, &PromptEncoding::FromText, text.to_string()),
        text
    );
}

fn components(mode: MmProcessingMode) -> MultimodalComponents {
    use super::super::{config::MultimodalConfigRegistry, settings::MultimodalSettings};
    let mut components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &MultimodalSettings::default(),
    )
    .expect("components");
    components.processing = mode;
    components
}

/// A registry holding one worker of `runtime` for model "m".
fn fleet(runtime: RuntimeType) -> WorkerRegistry {
    use openai_protocol::model_card::ModelCard;
    use smg_worker::identity::basic::{ConnectionMode, WorkerType};

    use crate::routers::grpc::backend_client::BasicWorkerBuilder;

    let registry = WorkerRegistry::new();
    registry
        .register(Arc::new(
            BasicWorkerBuilder::new("grpc://10.0.0.1:1")
                .model(ModelCard::new("m"))
                .worker_type(WorkerType::Regular)
                .runtime_type(runtime)
                .connection_mode(ConnectionMode::Grpc)
                .build(),
        ))
        .expect("register");
    registry
}

/// A spec emitting packed inputs applies only to a fleet that reads them:
/// on TokenSpeed it anchors each modality and rewrites the template's
/// `<video>`; behind another engine the model is handled as having no spec.
#[tokio::test]
async fn a_packed_spec_applies_only_to_a_fleet_that_reads_it() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::json!({
            "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
            "patch_size": 16,
            "downsample_ratio": 0.5,
            "norm_mean": [0.48145466, 0.4578275, 0.40821073],
            "norm_std": [0.26862954, 0.26130258, 0.27577711],
            "vision_config": {
                "args": {"min_num_patches": 1024, "max_num_patches": 13312},
                "video_target_num_patches": 1024,
                "video_temporal_patch_size": 2,
                "video_maintain_aspect_ratio": true
            },
            "sound_config": {"sampling_rate": 16000, "num_mel_bins": 128, "subsampling_factor": 8}
        })
        .to_string(),
    )
    .unwrap();
    let source = dir.path().to_str().unwrap();
    let tokenizer = MockTokenizer::new().with_tokens(&[
        ("<image>", 18),
        ("<img>", 19),
        ("</img>", 20),
        ("<so_embedding>", 27),
        ("<so_start>", 28),
        ("<so_end>", 29),
        ("<SPECIAL_30>", 30),
    ]);
    let plan = MediaPlan::new([
        MediaContentPart::ImageUrl {
            url: "https://a/1.png".to_string().into(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        },
        MediaContentPart::VideoUrl {
            url: "https://a/v.mp4".to_string().into(),
            uuid: None,
            fps: None,
            max_long_side_pixel: None,
        },
        MediaContentPart::AudioUrl {
            url: "https://a/a.wav".to_string().into(),
            uuid: None,
        },
    ]);
    let prepare = |mode, runtime| {
        let components = components(mode);
        let plan = &plan;
        let tokenizer = &tokenizer;
        async move {
            prepare_placeholder_tokens_with_env(
                plan,
                "m",
                tokenizer,
                &components,
                "tok",
                source,
                &HashMap::new(),
                |format| fleet_reads_format(&fleet(runtime), "m", format),
                |_| None,
            )
            .await
        }
    };

    let placeholders = prepare(MmProcessingMode::Auto, RuntimeType::TokenSpeed)
        .await
        .expect("served by TokenSpeed, the spec applies");
    assert!(!placeholders.is_worker_only());
    assert_eq!(placeholders.get(Modality::Image), Some("<image>"));
    assert_eq!(placeholders.get(Modality::Video), Some("<SPECIAL_30>"));
    assert_eq!(placeholders.get(Modality::Audio), Some("<so_embedding>"));
    assert_eq!(
        anchor_rendered_markers(
            Some(&placeholders),
            &PromptEncoding::FromText,
            "Frame <video> end".to_string()
        ),
        "Frame <SPECIAL_30> end"
    );

    for runtime in [RuntimeType::Vllm, RuntimeType::Sglang] {
        let placeholders = prepare(MmProcessingMode::Auto, runtime)
            .await
            .expect("another engine's workers process the media themselves");
        assert!(placeholders.is_worker_only());
        assert!(prepare(MmProcessingMode::Router, runtime).await.is_err());
    }
}

/// A model whose family has no registry spec: the router refuses it only
/// where it would process the media itself; under worker (or auto)
/// placement the request is marked worker-only, every modality counts as
/// expandable by the worker, and the router checks no anchors of its own.
#[tokio::test]
async fn a_model_without_a_spec_is_worker_only_unless_the_router_must_process() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}"#,
    )
    .unwrap();
    let source = dir.path().to_str().unwrap();
    let tokenizer = MockTokenizer::default();
    let plan = MediaPlan::new([MediaContentPart::ImageUrl {
        url: "https://a/1.png".to_string().into(),
        detail: None,
        uuid: None,
        max_long_side_pixel: None,
    }]);

    for mode in [MmProcessingMode::Worker, MmProcessingMode::Auto] {
        let placeholders = prepare_placeholder_tokens_with_env(
            &plan,
            "m6",
            &tokenizer,
            &components(mode),
            "tok",
            source,
            &HashMap::new(),
            |_| true,
            |_| None,
        )
        .await
        .unwrap_or_else(|error| panic!("{mode:?} placement: {error:#}"));
        assert!(placeholders.is_worker_only(), "{mode:?}");
        assert!(placeholders.worker_expandable(Modality::Image), "{mode:?}");
        assert!(placeholders.get(Modality::Image).is_none(), "{mode:?}");
        // The template rendered the engine's own placeholders (or none the
        // router knows of): nothing to check here, the engine checks.
        validate_rendered_media_anchors(&plan, &placeholders, &tokenizer, &[1, 2, 3])
            .expect("no anchors of the router's own to verify");
    }

    let error = prepare_placeholder_tokens_with_env(
        &plan,
        "m6",
        &tokenizer,
        &components(MmProcessingMode::Router),
        "tok",
        source,
        &HashMap::new(),
        |_| true,
        |_| None,
    )
    .await
    .expect_err("the router cannot process media it has no spec for");
    assert!(RequestRefusal::is(&error), "{error:#}");
    let message = format!("{error:#}");
    assert!(
        message.contains("multimodal not supported for model: m6"),
        "{message}"
    );
    assert!(message.contains("no media spec"), "{message}");
    assert!(message.contains("--mm-processing worker"), "{message}");
}

/// Registry failures the request brings are typed as the request's: a
/// modality the spec does not take, or more items than its limit. A config
/// field the model's files lack, a token its vocabulary lacks and a
/// modality counted twice (the plan counts each once) are typed as the
/// deployment's.
#[test]
fn registry_failures_are_typed_by_who_can_fix_them() {
    let typed = |error: ModelRegistryError| {
        let error = RequestRefusal::of_registry(&error, error.to_string());
        (RequestRefusal::is(&error), DeploymentFault::is(&error))
    };
    assert_eq!(
        typed(ModelRegistryError::UnsupportedModality {
            spec: "s",
            modality: Modality::Audio,
        }),
        (true, false)
    );
    assert_eq!(
        typed(ModelRegistryError::ModalityLimitExceeded {
            spec: "s",
            modality: Modality::Image,
            limit: 1,
            requested: 2,
        }),
        (true, false)
    );
    assert_eq!(
        typed(ModelRegistryError::MissingConfigField {
            field: "image_token_id".to_string(),
        }),
        (false, true)
    );
    assert_eq!(
        typed(ModelRegistryError::TokenNotFound {
            token: "<image>".to_string(),
        }),
        (false, true)
    );
    assert_eq!(
        typed(ModelRegistryError::DuplicateModality {
            spec: "s",
            modality: Modality::Image,
        }),
        (false, true)
    );
}

/// A chat request with one image part, its `max_long_side_pixel` as `cap`.
fn image_chat_plan(cap: Option<serde_json::Value>) -> MediaPlan {
    let mut image_url = serde_json::json!({"url": "https://invalid.example/image.png"});
    if let Some(cap) = cap {
        image_url["max_long_side_pixel"] = cap;
    }
    let request: openai_protocol::chat::ChatCompletionRequest =
        serde_json::from_value(serde_json::json!({
            "model": "checkpoint",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": image_url}]}]
        }))
        .unwrap();
    super::super::media_plan_chat(&request.messages, &request.other)
}

fn image_components(config: serde_json::Value) -> MultimodalComponents {
    let components = MultimodalComponents::new(
        Arc::new(MultimodalConfigRegistry::new()),
        None,
        None,
        &MultimodalSettings::default(),
    )
    .unwrap();
    components.config_registry.insert(
        "image-plan-test".to_string(),
        Arc::new(MultimodalModelConfig::new(
            config,
            PreProcessorConfig::default(),
            None,
        )),
    );
    components
}

fn image_tokenizer() -> (
    tempfile::TempDir,
    llm_tokenizer::huggingface::HuggingFaceTokenizer,
) {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("tokenizer.json");
    std::fs::write(
        &path,
        r#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"[UNK]":0,"<|image_pad|>":1,"<|unused_200054|>":200054},"unk_token":"[UNK]"}}"#,
    )
    .unwrap();
    let tokenizer =
        llm_tokenizer::huggingface::HuggingFaceTokenizer::from_file(path.to_str().unwrap())
            .unwrap();
    (dir, tokenizer)
}

async fn image_placeholders(
    plan: &MediaPlan,
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    components: &MultimodalComponents,
) -> Result<PlaceholderTokens> {
    prepare_placeholder_tokens_with_env(
        plan,
        model_id,
        tokenizer,
        components,
        "image-plan-test",
        "unused-local-source",
        &HashMap::new(),
        |_| true,
        |_| None,
    )
    .await
}

#[tokio::test]
async fn inkling2_image_plan_rejects_request_resize_before_fetch() {
    let tokenizer = MockTokenizer::new();
    for config in [
        serde_json::json!({"model_type": "inkling2"}),
        serde_json::json!({"architectures": ["Inkling2ForConditionalGeneration"]}),
        serde_json::json!({"model_type": "inkling2_loop"}),
        serde_json::json!({"architectures": ["Inkling2LoopForConditionalGeneration"]}),
    ] {
        let components = image_components(config);
        for cap in [0, 504, u32::MAX] {
            let plan = image_chat_plan(Some(serde_json::json!(cap)));
            let error = image_placeholders(&plan, "checkpoint", &tokenizer, &components)
                .await
                .unwrap_err();
            assert!(error
                .to_string()
                .contains("max_long_side_pixel is unsupported"));
        }
    }
}

#[tokio::test]
async fn inkling2_image_plan_accepts_missing_or_null_resize() {
    let (_dir, tokenizer) = image_tokenizer();
    for config in [
        serde_json::json!({"model_type": "inkling2"}),
        serde_json::json!({"architectures": ["Inkling2ForConditionalGeneration"]}),
        serde_json::json!({"model_type": "inkling2_loop"}),
        serde_json::json!({"architectures": ["Inkling2LoopForConditionalGeneration"]}),
    ] {
        let components = image_components(config);
        for cap in [None, Some(serde_json::Value::Null)] {
            let placeholders =
                image_placeholders(&image_chat_plan(cap), "checkpoint", &tokenizer, &components)
                    .await
                    .unwrap();
            assert_eq!(placeholders.get(Modality::Image), Some("<|unused_200054|>"));
        }
    }
}

#[tokio::test]
async fn image_plan_preserves_resize_for_other_model_specs() {
    let (_dir, tokenizer) = image_tokenizer();
    for (model_id, config, placeholder) in [
        (
            "org/inkling2-name-only",
            serde_json::json!({"model_type": "inkling_mm_model", "vision_config": {"decoder_dmodel": 1}}),
            "<|unused_200054|>",
        ),
        (
            "org/inkling2-name-only",
            serde_json::json!({"architectures": ["Inkling2ForConditionalGenerationSuffix"], "vision_config": {"decoder_dmodel": 1}}),
            "<|unused_200054|>",
        ),
        (
            "qwen2-vl",
            serde_json::json!({"model_type": "qwen2_vl"}),
            "<|image_pad|>",
        ),
    ] {
        let components = image_components(config);
        let placeholders = image_placeholders(
            &image_chat_plan(Some(serde_json::json!(504))),
            model_id,
            &tokenizer,
            &components,
        )
        .await
        .unwrap();
        assert_eq!(placeholders.get(Modality::Image), Some(placeholder));
    }
}
