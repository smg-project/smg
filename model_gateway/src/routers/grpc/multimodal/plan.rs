//! Canonical multimodal request planning shared by protocol adapters, rendering,
//! and preprocessing.

use std::collections::HashMap;

use anyhow::Result;
use llm_multimodal::{
    registry::modality_limit_override, EncoderInputFormat, FetchSourceKind, MediaContentPart,
    MediaPartOrder, Modality, ModelMetadata,
};
use llm_tokenizer::{traits::PromptEncoding, TokenizerTrait};
use openai_protocol::worker::MmProcessingMode;

use super::{
    config::MultimodalComponents,
    item_limits::{check_item_counts, effective_item_limits, limit_numbers, EngineItemLimit},
    RegistryTokenizer,
};

/// Ordered media extracted from an API request.
///
/// This is the single hand-off between protocol-specific parsing and the shared
/// multimodal pipeline. Text remains in the message representation used by the
/// chat template; media is kept here in authored order for fetching and count
/// validation.
#[derive(Debug, Clone, Default)]
pub(crate) struct MediaPlan {
    parts: Vec<MediaContentPart>,
    modalities: Vec<Modality>,
    counts: HashMap<Modality, usize>,
}

impl MediaPlan {
    pub(crate) fn new(parts: impl IntoIterator<Item = MediaContentPart>) -> Self {
        let mut plan = Self::default();
        for part in parts {
            let modality = match &part {
                MediaContentPart::ImageUrl { .. } | MediaContentPart::ImageData { .. } => {
                    Some(Modality::Image)
                }
                MediaContentPart::ImageEmbeds { .. } => Some(Modality::ImageEmbeds),
                MediaContentPart::AudioUrl { .. } | MediaContentPart::AudioData { .. } => {
                    Some(Modality::Audio)
                }
                MediaContentPart::VideoUrl { .. } | MediaContentPart::VideoData { .. } => {
                    Some(Modality::Video)
                }
                MediaContentPart::Text { .. } => None,
            };

            let Some(modality) = modality else {
                continue;
            };
            if !plan.modalities.contains(&modality) {
                plan.modalities.push(modality);
            }
            *plan.counts.entry(modality).or_default() += 1;
            plan.parts.push(part);
        }
        plan
    }

    /// The plan's items of `modality` per source (`url`, `data_url`,
    /// `inline`): what the request carries, before any limit is applied.
    pub(crate) fn source_counts(&self, modality: Modality) -> [(&'static str, usize); 3] {
        let mut counts = [
            (FetchSourceKind::Url.as_str(), 0),
            (FetchSourceKind::DataUrl.as_str(), 0),
            (FetchSourceKind::InlineBytes.as_str(), 0),
        ];
        for part in &self.parts {
            let (of, source) = match part {
                MediaContentPart::ImageUrl { url, .. } => (Modality::Image, url_source(url)),
                MediaContentPart::ImageData { .. } => (Modality::Image, 2),
                MediaContentPart::ImageEmbeds { .. } => (Modality::ImageEmbeds, 2),
                MediaContentPart::AudioUrl { url, .. } => (Modality::Audio, url_source(url)),
                MediaContentPart::AudioData { .. } => (Modality::Audio, 2),
                MediaContentPart::VideoUrl { url, .. } => (Modality::Video, url_source(url)),
                MediaContentPart::VideoData { .. } => (Modality::Video, 2),
                MediaContentPart::Text { .. } => continue,
            };
            if of == modality {
                counts[source].1 += 1;
            }
        }
        counts
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    pub(crate) fn modalities(&self) -> &[Modality] {
        &self.modalities
    }

    pub(crate) fn count(&self, modality: Modality) -> usize {
        self.counts.get(&modality).copied().unwrap_or_default()
    }

    pub(crate) fn into_parts(self) -> Vec<MediaContentPart> {
        self.parts
    }

    pub(crate) fn parts(&self) -> &[MediaContentPart] {
        &self.parts
    }

    /// Whether every part is a plain image/audio/video URL without per-item
    /// processing hints, i.e. something a worker can fetch and process itself.
    pub(crate) fn is_forwardable(&self) -> bool {
        self.parts.iter().all(|part| {
            matches!(
                part,
                MediaContentPart::ImageUrl {
                    max_long_side_pixel: None,
                    ..
                } | MediaContentPart::AudioUrl { .. }
                    | MediaContentPart::VideoUrl {
                        fps: None,
                        max_long_side_pixel: None,
                        ..
                    }
            )
        })
    }
}

/// Model-specific structural anchor strings keyed by modality.
///
/// String-format templates need the actual anchor string, while OpenAI-format
/// templates receive canonical `image` / `audio` / `video` parts. Keeping the
/// mapping typed prevents the former `image_placeholder` argument from being
/// accidentally reused for every modality.
#[derive(Debug, Clone, Default)]
pub(crate) struct PlaceholderTokens {
    tokens: HashMap<Modality, String>,
    /// Modalities whose anchor a vLLM worker can expand itself.
    worker_expandable: HashMap<Modality, bool>,
    /// Multi-token text the template writes in place of an anchor, with that anchor.
    rendered_markers: Vec<(String, String)>,
    /// The model matched no registry spec: the router knows no anchors and
    /// no limits for it, so it writes and checks none; its media can only go
    /// to a worker that processes media itself, whose engine owns the
    /// placeholders, their expansion and the per-prompt limits, as on the
    /// engine's own server.
    worker_only: bool,
}

impl PlaceholderTokens {
    /// For a model without a registry spec (see `worker_only`).
    pub(crate) fn worker_only() -> Self {
        Self {
            worker_only: true,
            ..Self::default()
        }
    }

    pub(crate) fn is_worker_only(&self) -> bool {
        self.worker_only
    }

    pub(crate) fn insert(&mut self, modality: Modality, token: String) {
        self.tokens.insert(modality, token);
    }

    pub(crate) fn get(&self, modality: Modality) -> Option<&str> {
        self.tokens.get(&modality).map(String::as_str)
    }

    pub(crate) fn add_rendered_marker(&mut self, marker: String, anchor: String) {
        self.rendered_markers.push((marker, anchor));
    }

    pub(crate) fn set_worker_expandable(&mut self, modality: Modality, expandable: bool) {
        self.worker_expandable.insert(modality, expandable);
    }

    /// Whether a vLLM worker can expand the rendered anchor for `modality`
    /// itself: the spec's opt-in, or every modality of a model without a
    /// spec, whose template renders the engine's own placeholders.
    pub(crate) fn worker_expandable(&self, modality: Modality) -> bool {
        self.worker_only
            || self
                .worker_expandable
                .get(&modality)
                .copied()
                .unwrap_or(false)
    }
}

/// Validate a multimodal request against the model spec and resolve the
/// structural anchors for its active modalities in one config/spec lookup.
///
/// `engine_limits` are the per-request media limits the model's workers
/// advertise for their engine ([`super::engine_item_limits`]): the request is
/// held to them here, before any fetch, as the engine's own front end would
/// (it never sees the precomputed inputs the router sends).
///
/// A model without a spec is refused only where the router would process
/// the media itself (`--mm-processing router`); otherwise its request is
/// marked worker-only and goes on to a worker that processes media itself
/// (see [`PlaceholderTokens::worker_only`]). A spec whose encoder input
/// format the engines that would receive the request do not read
/// (`reads_format` false) counts as no spec.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is an independent input of the one lookup"
)]
pub(crate) async fn prepare_placeholder_tokens(
    plan: &MediaPlan,
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
    engine_limits: &HashMap<Modality, EngineItemLimit>,
    reads_format: impl Fn(EncoderInputFormat) -> bool,
) -> Result<PlaceholderTokens> {
    anyhow::ensure!(!plan.is_empty(), "multimodal media plan is empty");
    let model_config = components
        .config_registry
        .get_or_load(tokenizer_id, tokenizer_source)
        .await?;
    let registry_tokenizer = RegistryTokenizer(tokenizer);
    let metadata = ModelMetadata {
        model_id,
        tokenizer: &registry_tokenizer,
        config: &model_config.config,
    };
    let Some(spec) = components
        .model_registry
        .lookup(&metadata)
        .filter(|spec| reads_format(spec.encoder_input_format()))
    else {
        anyhow::ensure!(
            components.processing != MmProcessingMode::Router,
            "multimodal not supported for model: {model_id} on the router (no media spec for \
             its family); a vLLM worker with --mm-processor inprocess|redis processes its media \
             itself under --mm-processing worker"
        );
        return Ok(PlaceholderTokens::worker_only());
    };
    let spec_limits = spec.modality_limits(&metadata).map_err(|error| {
        anyhow::anyhow!("reading the {} spec's media limits: {error}", spec.name())
    })?;
    // The router's own limits, capped by the engine's where its workers
    // advertise one; the refusal names the limit and what set it.
    let limits = effective_item_limits(
        &spec_limits,
        engine_limits,
        &components.modality_limit_overrides,
        modality_limit_override,
    );
    check_item_counts(plan, &limits).map_err(|error| anyhow::anyhow!("{error}"))?;
    let requested = plan
        .modalities()
        .iter()
        .map(|&modality| (modality, plan.count(modality)))
        .collect::<Vec<_>>();
    // The same numbers for the spec's own validation, so nothing downstream
    // loosens what the engine takes.
    spec.validate_media_request_with_limits(&metadata, &requested, &limit_numbers(&limits))
        .map_err(|error| {
            anyhow::anyhow!("invalid media request for model {}: {error}", spec.name())
        })?;
    let mut placeholders = PlaceholderTokens::default();
    for &modality in plan.modalities() {
        let token = spec
            .placeholder_token_for(&metadata, modality)
            .map_err(|error| {
                anyhow::anyhow!(
                    "model {} supports {modality} but its placeholder token could not be resolved: {error}",
                    spec.name()
                )
            })?;
        anyhow::ensure!(
            tokenizer.token_to_id(&token).is_some(),
            "{modality} placeholder token '{token}' is missing from the tokenizer vocabulary"
        );
        if let Some(marker) = spec.rendered_marker_for(modality) {
            placeholders.add_rendered_marker(marker.to_string(), token.clone());
        }
        placeholders.insert(modality, token);
        placeholders.set_worker_expandable(modality, spec.worker_expandable(modality));
    }

    Ok(placeholders)
}

/// Resolve the media-part ordering for a model from the same registry that owns
/// its placeholder/prompt logic: where media goes relative to text when the
/// template takes one string per message (a template that takes parts places
/// them itself, in the request's order). Falls back to the engine's media-first
/// placement when the model matches no spec.
pub(crate) async fn resolve_media_part_order(
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
) -> MediaPartOrder {
    let model_config = match components
        .config_registry
        .get_or_load(tokenizer_id, tokenizer_source)
        .await
    {
        Ok(config) => config,
        Err(_) => return MediaPartOrder::MediaFirst,
    };
    let registry_tokenizer = RegistryTokenizer(tokenizer);
    let metadata = ModelMetadata {
        model_id,
        tokenizer: &registry_tokenizer,
        config: &model_config.config,
    };
    components
        .model_registry
        .lookup(&metadata)
        .map(|spec| spec.media_part_order())
        .unwrap_or(MediaPartOrder::MediaFirst)
}

/// The prompt text to tokenize: a text-rendered prompt gets each template
/// marker replaced by its anchor token (see
/// `ModelProcessorSpec::rendered_marker_for`); other prompts pass through.
pub(crate) fn anchor_rendered_markers(
    placeholders: Option<&PlaceholderTokens>,
    encoding: &PromptEncoding,
    text: String,
) -> String {
    match (placeholders, encoding) {
        (Some(placeholders), PromptEncoding::FromText) => placeholders
            .rendered_markers
            .iter()
            .fold(text, |text, (marker, anchor)| text.replace(marker, anchor)),
        _ => text,
    }
}

/// Verify a rendered/tokenized multimodal prompt contains exactly one
/// structural anchor for every planned media item before any media is fetched
/// or preprocessed.
///
/// This catches stale/custom templates, adapter omissions, and literal internal
/// anchors in user text at the cheapest point in the pipeline.
pub(crate) fn validate_rendered_media_anchors(
    plan: &MediaPlan,
    placeholders: &PlaceholderTokens,
    tokenizer: &dyn TokenizerTrait,
    token_ids: &[u32],
) -> Result<()> {
    if placeholders.is_worker_only() {
        // No spec, no anchors of the router's own: the worker's engine checks
        // its placeholders against the media it fetches, as its own server does.
        return Ok(());
    }
    for &modality in plan.modalities() {
        let token = placeholders
            .get(modality)
            .ok_or_else(|| anyhow::anyhow!("missing resolved {modality} placeholder token"))?;
        let token_id = tokenizer.token_to_id(token).ok_or_else(|| {
            anyhow::anyhow!(
                "{modality} placeholder token '{token}' is missing from the tokenizer vocabulary"
            )
        })?;
        let expected = plan.count(modality);
        let actual = token_ids
            .iter()
            .filter(|&&candidate| candidate == token_id)
            .count();
        anyhow::ensure!(
            actual == expected,
            "rendered {modality} anchor count mismatch: expected {expected}, found {actual}; verify the chat template contract and escape literal media anchors in user text"
        );
    }
    Ok(())
}

/// The source index of a URL part in [`MediaPlan::source_counts`], read as
/// the tracker reads the URL at the fetch: a `data:` URL is 1, any other URL
/// 0.
fn url_source(url: &str) -> usize {
    usize::from(FetchSourceKind::of_url(url) == FetchSourceKind::DataUrl)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_multimodal::PreProcessorConfig;
    use llm_tokenizer::mock::MockTokenizer;

    use super::{
        super::{
            capability::fleet_reads_format,
            config::{MultimodalConfigRegistry, MultimodalModelConfig},
            settings::MultimodalSettings,
        },
        *,
    };
    use crate::worker::{RuntimeType, WorkerRegistry};

    fn image_urls(count: usize) -> MediaPlan {
        MediaPlan::new((0..count).map(|i| MediaContentPart::ImageUrl {
            url: format!("https://example.com/{i}.png"),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }))
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
        let error = prepare_placeholder_tokens(
            &plan,
            "org/llava",
            &tokenizer,
            &components,
            "tok",
            "unused",
            &engine,
            |_| true,
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
        let error = prepare_placeholder_tokens(
            &plan,
            "org/llava",
            &tokenizer,
            &components,
            "tok",
            "unused",
            &HashMap::new(),
            |_| true,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(!error.contains("limit of"), "{error}");
    }

    #[test]
    fn media_plan_preserves_media_order_and_counts() {
        let plan = MediaPlan::new([
            MediaContentPart::Text {
                text: "ignored".to_string(),
            },
            MediaContentPart::AudioUrl {
                url: "audio".to_string(),
                uuid: None,
            },
            MediaContentPart::ImageUrl {
                url: "image".to_string(),
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            },
            MediaContentPart::AudioUrl {
                url: "audio-2".to_string(),
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
            MediaContentPart::AudioUrl { url, .. } if url == "audio"
        ));
        assert!(matches!(
            &parts[1],
            MediaContentPart::ImageUrl { url, .. } if url == "image"
        ));
        assert!(matches!(
            &parts[2],
            MediaContentPart::AudioUrl { url, .. } if url == "audio-2"
        ));
    }

    #[test]
    fn rendered_anchor_validation_is_exact_per_modality() {
        let plan = MediaPlan::new([
            MediaContentPart::ImageUrl {
                url: "image".to_string(),
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            },
            MediaContentPart::AudioUrl {
                url: "audio".to_string(),
                uuid: None,
            },
        ]);
        let mut placeholders = PlaceholderTokens::default();
        placeholders.insert(Modality::Image, "<|im_start|>".to_string());
        placeholders.insert(Modality::Audio, "<|im_end|>".to_string());
        let tokenizer = MockTokenizer::new();

        validate_rendered_media_anchors(&plan, &placeholders, &tokenizer, &[1001, 7, 1002])
            .unwrap();

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

        use crate::worker::{BasicWorkerBuilder, ConnectionMode, WorkerType};

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
                url: "https://a/1.png".to_string(),
                detail: None,
                uuid: None,
                max_long_side_pixel: None,
            },
            MediaContentPart::VideoUrl {
                url: "https://a/v.mp4".to_string(),
                uuid: None,
                fps: None,
                max_long_side_pixel: None,
            },
            MediaContentPart::AudioUrl {
                url: "https://a/a.wav".to_string(),
                uuid: None,
            },
        ]);
        let prepare = |mode, runtime| {
            let components = components(mode);
            let plan = &plan;
            let tokenizer = &tokenizer;
            async move {
                prepare_placeholder_tokens(
                    plan,
                    "m",
                    tokenizer,
                    &components,
                    "tok",
                    source,
                    &HashMap::new(),
                    |format| fleet_reads_format(&fleet(runtime), "m", format),
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
            url: "https://a/1.png".to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }]);

        for mode in [MmProcessingMode::Worker, MmProcessingMode::Auto] {
            let placeholders = prepare_placeholder_tokens(
                &plan,
                "m6",
                &tokenizer,
                &components(mode),
                "tok",
                source,
                &HashMap::new(),
                |_| true,
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

        let error = prepare_placeholder_tokens(
            &plan,
            "m6",
            &tokenizer,
            &components(MmProcessingMode::Router),
            "tok",
            source,
            &HashMap::new(),
            |_| true,
        )
        .await
        .expect_err("the router cannot process media it has no spec for");
        let message = format!("{error:#}");
        assert!(
            message.contains("multimodal not supported for model: m6"),
            "{message}"
        );
        assert!(message.contains("no media spec"), "{message}");
        assert!(message.contains("--mm-processing worker"), "{message}");
    }
}
