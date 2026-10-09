//! Canonical multimodal request planning shared by protocol adapters, rendering,
//! and preprocessing.

use std::collections::HashMap;

use anyhow::{Context, Result};
use llm_multimodal::{
    registry::modality_limit_override, MediaContentPart, MediaPartOrder, Modality, ModelMetadata,
};
use llm_tokenizer::TokenizerTrait;

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

    /// Whether every part is a plain image/video URL without per-item
    /// processing hints, i.e. something a worker can fetch and process itself.
    pub(crate) fn is_forwardable(&self) -> bool {
        self.parts.iter().all(|part| {
            matches!(
                part,
                MediaContentPart::ImageUrl {
                    max_long_side_pixel: None,
                    ..
                } | MediaContentPart::VideoUrl {
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
}

impl PlaceholderTokens {
    pub(crate) fn insert(&mut self, modality: Modality, token: String) {
        self.tokens.insert(modality, token);
    }

    pub(crate) fn get(&self, modality: Modality) -> Option<&str> {
        self.tokens.get(&modality).map(String::as_str)
    }

    pub(crate) fn set_worker_expandable(&mut self, modality: Modality, expandable: bool) {
        self.worker_expandable.insert(modality, expandable);
    }

    pub(crate) fn worker_expandable(&self, modality: Modality) -> bool {
        self.worker_expandable
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
pub(crate) async fn prepare_placeholder_tokens(
    plan: &MediaPlan,
    model_id: &str,
    tokenizer: &dyn TokenizerTrait,
    components: &MultimodalComponents,
    tokenizer_id: &str,
    tokenizer_source: &str,
    engine_limits: &HashMap<Modality, EngineItemLimit>,
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
    let spec = components
        .model_registry
        .lookup(&metadata)
        .with_context(|| format!("multimodal not supported for model: {model_id}"))?;
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
        placeholders.insert(modality, token);
        placeholders.set_worker_expandable(modality, spec.worker_expandable(modality));
    }

    Ok(placeholders)
}

/// Resolve the media-part ordering for a model from the same registry that owns
/// its placeholder/prompt logic. Falls back to vLLM-compatible media-first when
/// the model matches no spec.
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm_multimodal::PreProcessorConfig;
    use llm_tokenizer::mock::MockTokenizer;

    use super::{
        super::{
            config::{MultimodalConfigRegistry, MultimodalModelConfig},
            settings::MultimodalSettings,
        },
        *,
    };
    use crate::worker::RuntimeType;

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
            Arc::new(MultimodalModelConfig {
                config: serde_json::json!({"model_type": "llava", "image_token_index": 32000}),
                preprocessor_config: PreProcessorConfig::default(),
                video_preprocessor_config: None,
            }),
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
}
