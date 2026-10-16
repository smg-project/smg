use std::{collections::HashMap, sync::OnceLock};

use serde_json::Value;
use thiserror::Error;

use crate::{
    audio::AudioPreProcessor,
    encoder_inputs::PreprocessedEncoderInputs,
    media::FrameSampling,
    types::{
        EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId, VideoSamplingInfo,
    },
    vision::PreProcessorConfig,
};

/// The format of the encoder inputs a spec's processors emit, which decides
/// the engines whose models can read them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderInputFormat {
    /// The model family's reference processor outputs, which every engine
    /// serving the family reads.
    Reference,
    /// Images and video frames as patch rows with per-item grids, audio as mel
    /// frames with per-clip lengths, for an engine whose model reads them
    /// packed rather than the reference processor's tensors.
    Packed,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModelRegistryError {
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("token '{token}' not found in tokenizer vocabulary")]
    TokenNotFound { token: String },
    #[error("missing config field '{field}'")]
    MissingConfigField { field: String },
    #[error("invalid or missing preprocessed field '{field}'")]
    InvalidPreprocessedField { field: String },
    #[error("config field '{field}' is present but unusable: {reason}")]
    InvalidConfigField { field: String, reason: String },
    #[error("model spec {spec} could not encode '{text}' with the model tokenizer")]
    TextEncodingFailed { spec: &'static str, text: String },
    #[error("modality {modality} is not supported by model spec {spec}")]
    UnsupportedModality {
        spec: &'static str,
        modality: Modality,
    },
    #[error("model spec {spec} supports at most {limit} {modality} inputs; got {requested}")]
    ModalityLimitExceeded {
        spec: &'static str,
        modality: Modality,
        limit: usize,
        requested: usize,
    },
    #[error("modality {modality} appears more than once in the request for model spec {spec}")]
    DuplicateModality {
        spec: &'static str,
        modality: Modality,
    },
}

pub type RegistryResult<T> = Result<T, ModelRegistryError>;

static IMAGE_MAX_COUNT_OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
static VIDEO_MAX_COUNT_OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
static AUDIO_MAX_COUNT_OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();

fn env_count_override(cache: &'static OnceLock<Option<usize>>, env_var: &str) -> Option<usize> {
    *cache.get_or_init(|| {
        std::env::var(env_var)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|count| *count > 0)
    })
}

/// Deployment-wide override of a spec's per-request media-count limit
/// (`SMG_IMAGE_MAX_COUNT`, `SMG_VIDEO_MAX_COUNT`, `SMG_AUDIO_MAX_COUNT`).
///
/// The override replaces the spec's declared limit (it can raise or lower
/// it) but never enables a modality the spec does not declare. Unset,
/// non-numeric, or zero values are ignored.
pub fn modality_limit_override(modality: Modality) -> Option<usize> {
    match modality {
        Modality::Image => env_count_override(&IMAGE_MAX_COUNT_OVERRIDE, "SMG_IMAGE_MAX_COUNT"),
        Modality::Video => env_count_override(&VIDEO_MAX_COUNT_OVERRIDE, "SMG_VIDEO_MAX_COUNT"),
        Modality::Audio => env_count_override(&AUDIO_MAX_COUNT_OVERRIDE, "SMG_AUDIO_MAX_COUNT"),
        Modality::ImageEmbeds => None,
    }
}

fn check_media_counts(
    spec: &'static str,
    limits: &HashMap<Modality, usize>,
    requested: &[(Modality, usize)],
    limit_override: impl Fn(Modality) -> Option<usize>,
) -> RegistryResult<()> {
    let mut active = Vec::with_capacity(requested.len());

    for &(modality, count) in requested {
        if count == 0 {
            continue;
        }
        if active.contains(&modality) {
            return Err(ModelRegistryError::DuplicateModality { spec, modality });
        }
        active.push(modality);

        let Some(&limit) = limits.get(&modality) else {
            return Err(ModelRegistryError::UnsupportedModality { spec, modality });
        };
        let limit = limit_override(modality).unwrap_or(limit);
        if count > limit {
            return Err(ModelRegistryError::ModalityLimitExceeded {
                spec,
                modality,
                limit,
                requested: count,
            });
        }
    }

    Ok(())
}

/// Ordering of media and text parts when rendering a multipart message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaPartOrder {
    /// Emit media parts before text (the serving engine's `interleave_mm_strings=false` order).
    MediaFirst,
    /// Keep parts in request order; part order is protocol-visible.
    Authored,
}

/// The tokenizer surface the registry needs: resolving placeholder and
/// structural token ids, and encoding the short text fragments some families
/// splice into their media wrappers (e.g. an `image {W}x{H}` header).
///
/// Deliberately local: callers adapt whatever tokenizer they hold to this
/// trait, and this crate stays free of any tokenizer crate's dependency tree.
pub trait Tokenizer: Send + Sync {
    /// The token id for a token string, if the vocabulary knows it.
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// The token string for a token id, if the vocabulary knows it.
    fn id_to_token(&self, id: u32) -> Option<String>;

    /// Encode plain text (no special tokens) into token ids; `None` when the
    /// text cannot be encoded.
    fn encode_text(&self, text: &str) -> Option<Vec<u32>>;
}

/// Metadata about the current model used to derive tokenizer/config dependent fields.
pub struct ModelMetadata<'a> {
    pub model_id: &'a str,
    pub tokenizer: &'a dyn Tokenizer,
    pub config: &'a Value,
}

impl<'a> ModelMetadata<'a> {
    pub fn token_id(&self, token: &str) -> RegistryResult<TokenId> {
        self.tokenizer
            .token_to_id(token)
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::TokenNotFound {
                token: token.to_string(),
            })
    }

    /// `text` encoded on its own by the model tokenizer, the way a spec
    /// splices a frame stamp or a dimension line into a replacement. A
    /// tokenizer that cannot encode it is the request's error, never an
    /// empty splice: a replacement missing a stamp would reach the engine
    /// with a layout its reference processor does not write.
    pub fn encode_plain(&self, spec: &'static str, text: &str) -> RegistryResult<Vec<TokenId>> {
        let ids = self.tokenizer.encode_text(text).ok_or_else(|| {
            ModelRegistryError::TextEncodingFailed {
                spec,
                text: text.to_string(),
            }
        })?;
        Ok(ids.into_iter().map(|id| id as TokenId).collect())
    }

    pub fn config_u32(&self, path: &[&str]) -> Option<u32> {
        Self::find_value(self.config, path).and_then(|value| value.as_u64().map(|v| v as u32))
    }

    pub fn config_model_type(&self) -> Option<&str> {
        Self::find_value(self.config, &["model_type"]).and_then(Value::as_str)
    }

    fn find_value<'v>(value: &'v Value, path: &[&str]) -> Option<&'v Value> {
        let mut current = value;
        for key in path {
            current = current.get(*key)?;
        }
        Some(current)
    }
}

/// Decoder-side facts about one media item that prompt replacements may need.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MediaItemInfo {
    /// Sampling behind a decoded clip; `None` for images/audio or when the decoder could not report it.
    pub video_sampling: Option<VideoSamplingInfo>,
}

/// The replacements for a preprocessed batch of one modality, given the
/// decoder-side facts of its media items and the config the modality was
/// preprocessed with (empty and default when the caller has none).
pub type ReplacementsHook = fn(
    &ModelMetadata,
    &PreprocessedEncoderInputs,
    &[MediaItemInfo],
    &PreProcessorConfig,
) -> RegistryResult<Vec<PromptReplacement>>;

/// What one spec knows about one modality: the anchor its template writes and
/// the id behind it, how its replacements are built, and how its tensors are
/// laid out. A spec's table of these drives the trait's per-modality methods
/// through [`ModelProcessorSpec::hooks`], so the Image / Video / Audio dispatch
/// is written once per spec instead of once per method.
pub struct ModalityHooks {
    /// The placeholder the rendered prompt carries for one item.
    pub placeholder: fn(&ModelMetadata) -> RegistryResult<String>,
    /// The token id the placeholder expands around.
    pub placeholder_id: fn(&ModelMetadata) -> RegistryResult<TokenId>,
    /// The replacements for a preprocessed batch of the modality.
    pub replacements: ReplacementsHook,
    /// The primary and side-tensor layouts of the modality's batch.
    pub layouts: fn() -> EncoderFieldLayouts,
    /// Tensor keys that stay on the CPU.
    pub keep_on_cpu: &'static [&'static str],
    /// The engine-side name of the primary encoder input, when it is not the
    /// conventional one.
    pub encoder_input_key: Option<&'static str>,
    /// Whether the serving engine expands the placeholder itself.
    pub worker_expandable: bool,
}

pub trait ModelProcessorSpec: Send + Sync {
    fn name(&self) -> &'static str;
    fn matches(&self, metadata: &ModelMetadata) -> bool;

    /// The spec's hooks for one modality: `None` for a modality it does not
    /// serve, and for a spec that keeps the per-method defaults (an image-only
    /// spec needs no table). The `*_for` methods below read the table first
    /// and fall back to those defaults.
    fn hooks(&self, _modality: Modality) -> Option<ModalityHooks> {
        None
    }

    /// Ordering of media/text parts this model's chat template expects. Most
    /// models use the serving engine's media-first order; families whose template renders
    /// parts positionally override to `Authored`.
    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::MediaFirst
    }

    /// Whether the anchor rendered for `modality` is exactly the single token
    /// the serving engine's prompt updates target, so a worker can expand it itself.
    fn worker_expandable(&self, modality: Modality) -> bool {
        self.hooks(modality)
            .is_some_and(|hooks| hooks.worker_expandable)
    }

    /// The format of the encoder inputs this spec's processors emit. The
    /// gateway applies the spec only while every engine serving the model
    /// reads that format; otherwise the model is handled as having no spec,
    /// and its workers process media themselves.
    fn encoder_input_format(&self) -> EncoderInputFormat {
        EncoderInputFormat::Reference
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String>;
    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId>;
    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match (self.hooks(modality), modality) {
            (Some(hooks), _) => (hooks.placeholder)(metadata),
            (None, Modality::Image) => self.placeholder_token(metadata),
            (None, _) => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }
    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match (self.hooks(modality), modality) {
            (Some(hooks), _) => (hooks.placeholder_id)(metadata),
            (None, Modality::Image) => self.placeholder_token_id(metadata),
            (None, _) => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }
    /// The text the model's chat template writes for one item of `modality`
    /// when that is not the anchor token itself but several tokens (a
    /// literal such as `<video>`). The gateway rewrites each occurrence in
    /// the rendered prompt to the anchor before tokenizing; `None` when the
    /// template writes the anchor.
    fn rendered_marker_for(&self, _modality: Modality) -> Option<&'static str> {
        None
    }

    fn modality_limits(&self, metadata: &ModelMetadata)
        -> RegistryResult<HashMap<Modality, usize>>;

    /// Validate the active modalities and item counts in one media request.
    ///
    /// Any subset of the modalities declared by [`Self::modality_limits`] is
    /// accepted. Each modality may appear once in `requested`; zero-count
    /// entries are ignored. Per-modality limits can be overridden
    /// deployment-wide via `SMG_IMAGE_MAX_COUNT`, `SMG_VIDEO_MAX_COUNT`, and
    /// `SMG_AUDIO_MAX_COUNT`.
    fn validate_media_request(
        &self,
        metadata: &ModelMetadata,
        requested: &[(Modality, usize)],
    ) -> RegistryResult<()> {
        self.validate_media_request_with_limits(metadata, requested, &HashMap::new())
    }

    /// [`Self::validate_media_request`] with caller-supplied per-modality limit
    /// overrides; each replaces the spec limit and beats the env override.
    fn validate_media_request_with_limits(
        &self,
        metadata: &ModelMetadata,
        requested: &[(Modality, usize)],
        limit_overrides: &HashMap<Modality, usize>,
    ) -> RegistryResult<()> {
        let limits = self.modality_limits(metadata)?;
        check_media_counts(self.name(), &limits, requested, |modality| {
            limit_overrides
                .get(&modality)
                .copied()
                .or_else(|| modality_limit_override(modality))
        })
    }

    fn processor_kwargs(&self, metadata: &ModelMetadata) -> RegistryResult<Value>;

    /// Build the audio preprocessor for this model, if it supports audio.
    ///
    /// This is the single source of truth for audio-processor selection: the
    /// same spec that owns a model's prompt/placeholder logic also owns its
    /// audio preprocessor factory, so there is no separate string-keyed
    /// registry to keep in sync. Audio-less specs use the default
    /// (`Ok(None)`); an audio config the spec cannot serve is an error
    /// naming the field, never `None`.
    ///
    /// The processor is built from the current model config because its feature
    /// shapes and quantization parameters can be checkpoint-specific.
    fn audio_processor(
        &self,
        _model_config: &Value,
        _preprocessor_config: &PreProcessorConfig,
    ) -> RegistryResult<Option<Box<dyn AudioPreProcessor>>> {
        Ok(None)
    }

    /// Compute per-media prompt replacement token sequences.
    ///
    /// Receives the full preprocessed output so each model can extract whatever
    /// metadata it needs (e.g. aspect_ratios for tile-based models).  This
    /// mirrors the reference implementation's `_get_prompt_updates(out_mm_kwargs)` pattern.
    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>>;
    /// The replacements for one modality without the media items' facts: a
    /// hooked modality is built with no items and the default config, which
    /// is how the stamped families fall back to their assumed timing.
    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match (self.hooks(modality), modality) {
            (Some(hooks), _) => {
                (hooks.replacements)(metadata, preprocessed, &[], &PreProcessorConfig::default())
            }
            (None, Modality::Image) => self.prompt_replacements(metadata, preprocessed),
            (None, _) => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    /// Prompt replacements given one [`MediaItemInfo`] per media item in batch
    /// order and the config the modality was preprocessed with; a hooked
    /// modality receives both, the default ignores them.
    fn prompt_replacements_with_media(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
        media: &[MediaItemInfo],
        preprocessor_config: &PreProcessorConfig,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match self.hooks(modality) {
            Some(hooks) => (hooks.replacements)(metadata, preprocessed, media, preprocessor_config),
            None => self.prompt_replacements_for(metadata, preprocessed, modality),
        }
    }

    /// Declare how each tensor's first dimension maps to media items.
    ///
    /// Keys not listed are treated as shared (replicated across all media items).
    /// The `"pixel_values"` key mirrors the reference processors' vision kwargs and should be
    /// included when the primary encoder input differs from batched layout.
    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // Default: encoder_input is batched (most models).
        HashMap::from([("pixel_values".to_string(), FieldLayout::Batched)])
    }

    /// Declare the neutral primary/side-tensor layout contract for one modality.
    ///
    /// A hooked modality reads its layouts from [`ModelProcessorSpec::hooks`];
    /// otherwise the default converts the legacy reference-shaped field map so
    /// existing vision specs remain source-compatible. New multimodal specs
    /// declare their layouts in `hooks()` rather than override this, and keep
    /// backend-specific field names at adapters.
    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match self.hooks(modality) {
            Some(hooks) => (hooks.layouts)(),
            None => EncoderFieldLayouts::from_legacy_fields(self.field_layouts()),
        }
    }

    /// Tensor keys that should remain on CPU (not transferred to GPU).
    ///
    /// In the serving engine, certain model-specific tensors are marked `keep_on_cpu=True`
    /// in their `MultiModalFieldConfig`.  This method mirrors that per-model
    /// knowledge so the router can send the hint via gRPC, avoiding the need
    /// for the backend to instantiate a Python processor just to query it.
    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec![]
    }

    /// Tensor keys that should remain on CPU for one modality.
    ///
    /// A hooked modality reads them from [`ModelProcessorSpec::hooks`];
    /// otherwise the default preserves the legacy model-wide declaration.
    fn keep_on_cpu_keys_for(&self, modality: Modality) -> Vec<String> {
        match self.hooks(modality) {
            Some(hooks) => hooks
                .keep_on_cpu
                .iter()
                .map(|key| (*key).to_string())
                .collect(),
            None => self.keep_on_cpu_keys(),
        }
    }

    /// The engine-side name of the primary encoder input for one modality,
    /// when it is not the HF-conventional `pixel_values` / `audio_features`
    /// (DeepSeek-V4.1's forward pops `patches`). `None` keeps the default
    /// name; adapters that build the engine request consult this before
    /// naming the tensor.
    fn encoder_input_key_for(&self, modality: Modality) -> Option<String> {
        self.hooks(modality)
            .and_then(|hooks| hooks.encoder_input_key.map(str::to_string))
    }

    /// Frame rate to sample a video at when the request names no `fps`: the
    /// model's reference processor default, so token counts match it. `None`
    /// keeps the media connector's default.
    fn default_video_sample_fps(&self) -> Option<f32> {
        None
    }

    /// [`Self::default_video_sample_fps`] given the video preprocessor config
    /// the checkpoint ships, for a family whose reference reads the rate from
    /// it; the default ignores the config.
    fn default_video_sample_fps_with(&self, _config: Option<&PreProcessorConfig>) -> Option<f32> {
        self.default_video_sample_fps()
    }

    /// Where a video's sampled frames sit: evenly spread, or one per interval
    /// from the start with the last frame kept, as the model's reference does.
    fn video_frame_sampling(&self) -> FrameSampling {
        FrameSampling::Even
    }

    /// [`Self::video_frame_sampling`] given the video preprocessor config the
    /// checkpoint ships, for a family whose reference reads its frame budget
    /// or grouping from it; the default ignores the config.
    fn video_frame_sampling_with(&self, _config: Option<&PreProcessorConfig>) -> FrameSampling {
        self.video_frame_sampling()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        registry::test_helpers::{test_preprocessed_with_tokens, TestTokenizer},
        types::ImageSize,
    };

    #[test]
    fn encode_plain_maps_the_ids_or_names_the_spec_and_the_text() {
        let config = json!({});
        let encoding = TestTokenizer::new(&[]).with_byte_encoder(1000);
        let metadata = ModelMetadata {
            model_id: "m",
            tokenizer: &encoding,
            config: &config,
        };
        assert_eq!(
            metadata.encode_plain("test", "ab").unwrap(),
            vec![1097 as TokenId, 1098]
        );

        struct NoText;
        impl Tokenizer for NoText {
            fn token_to_id(&self, _: &str) -> Option<u32> {
                None
            }
            fn id_to_token(&self, _: u32) -> Option<String> {
                None
            }
            fn encode_text(&self, _: &str) -> Option<Vec<u32>> {
                None
            }
        }
        let metadata = ModelMetadata {
            model_id: "m",
            tokenizer: &NoText,
            config: &config,
        };
        assert_eq!(
            metadata.encode_plain("test", "<0.2 seconds>").unwrap_err(),
            ModelRegistryError::TextEncodingFailed {
                spec: "test",
                text: "<0.2 seconds>".to_string(),
            }
        );
    }

    struct TestSpec;

    impl ModelProcessorSpec for TestSpec {
        fn name(&self) -> &'static str {
            "test"
        }

        fn matches(&self, _metadata: &ModelMetadata) -> bool {
            true
        }

        fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
            Ok("<image>".to_string())
        }

        fn placeholder_token_id(&self, _metadata: &ModelMetadata) -> RegistryResult<TokenId> {
            Ok(1)
        }

        fn modality_limits(
            &self,
            _metadata: &ModelMetadata,
        ) -> RegistryResult<HashMap<Modality, usize>> {
            Ok(HashMap::from([(Modality::Image, 2), (Modality::Audio, 1)]))
        }

        fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
            Ok(json!({}))
        }

        fn prompt_replacements(
            &self,
            _metadata: &ModelMetadata,
            preprocessed: &PreprocessedEncoderInputs,
        ) -> RegistryResult<Vec<PromptReplacement>> {
            Ok(preprocessed
                .feature_token_counts
                .iter()
                .map(|&count| {
                    PromptReplacement::sequence(Modality::Image, "<image>", vec![1; count])
                })
                .collect())
        }
    }

    fn validate(
        spec: &dyn ModelProcessorSpec,
        requested: &[(Modality, usize)],
    ) -> RegistryResult<()> {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({});
        let metadata = ModelMetadata {
            model_id: "test-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        spec.validate_media_request(&metadata, requested)
    }

    #[test]
    fn media_aware_replacements_default_to_the_modality_hook() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({});
        let metadata = ModelMetadata {
            model_id: "test-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let preprocessed =
            test_preprocessed_with_tokens(&[ImageSize::new(4, 4), ImageSize::new(4, 4)], &[3, 5]);
        let media = vec![
            MediaItemInfo::default(),
            MediaItemInfo {
                video_sampling: Some(VideoSamplingInfo {
                    source_fps: 30.0,
                    frame_indices: vec![0, 15],
                }),
            },
        ];

        let preprocessor_config = PreProcessorConfig::default();
        let with_media = TestSpec
            .prompt_replacements_with_media(
                &metadata,
                &preprocessed,
                Modality::Image,
                &media,
                &preprocessor_config,
            )
            .unwrap();
        let without_media = TestSpec
            .prompt_replacements_for(&metadata, &preprocessed, Modality::Image)
            .unwrap();

        let tokens = |replacements: &[PromptReplacement]| {
            replacements
                .iter()
                .map(|replacement| replacement.tokens.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(tokens(&with_media), vec![vec![1; 3], vec![1; 5]]);
        assert_eq!(tokens(&with_media), tokens(&without_media));
        let unsupported = TestSpec
            .prompt_replacements_with_media(
                &metadata,
                &preprocessed,
                Modality::Video,
                &media,
                &preprocessor_config,
            )
            .unwrap_err();
        assert_eq!(
            unsupported,
            ModelRegistryError::UnsupportedModality {
                spec: "test",
                modality: Modality::Video,
            }
        );
    }

    #[test]
    fn worker_expandable_defaults_to_false() {
        assert!(!TestSpec.worker_expandable(Modality::Image));
        assert!(!TestSpec.worker_expandable(Modality::Video));
    }

    #[test]
    fn validation_accepts_any_declared_modality_subset() {
        assert_eq!(validate(&TestSpec, &[(Modality::Image, 2)]), Ok(()));
        assert_eq!(
            validate(&TestSpec, &[(Modality::Image, 1), (Modality::Audio, 1)]),
            Ok(())
        );
    }

    #[test]
    fn validation_rejects_undeclared_modality() {
        assert_eq!(
            validate(&TestSpec, &[(Modality::Video, 1)]),
            Err(ModelRegistryError::UnsupportedModality {
                spec: "test",
                modality: Modality::Video,
            })
        );
    }

    #[test]
    fn validation_rejects_count_above_limit() {
        assert_eq!(
            validate(&TestSpec, &[(Modality::Image, 3)]),
            Err(ModelRegistryError::ModalityLimitExceeded {
                spec: "test",
                modality: Modality::Image,
                limit: 2,
                requested: 3,
            })
        );
    }

    #[test]
    fn validation_rejects_duplicate_modality_counts() {
        assert_eq!(
            validate(&TestSpec, &[(Modality::Image, 1), (Modality::Image, 1)]),
            Err(ModelRegistryError::DuplicateModality {
                spec: "test",
                modality: Modality::Image,
            })
        );
    }

    #[test]
    fn caller_limit_override_replaces_spec_limit() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({});
        let metadata = ModelMetadata {
            model_id: "test-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let overrides = HashMap::from([(Modality::Image, 5)]);

        // TestSpec declares Image=2; the caller override wins in both directions.
        assert_eq!(
            TestSpec.validate_media_request_with_limits(
                &metadata,
                &[(Modality::Image, 5)],
                &overrides
            ),
            Ok(())
        );
        assert_eq!(
            TestSpec.validate_media_request_with_limits(
                &metadata,
                &[(Modality::Image, 6)],
                &overrides
            ),
            Err(ModelRegistryError::ModalityLimitExceeded {
                spec: "test",
                modality: Modality::Image,
                limit: 5,
                requested: 6,
            })
        );
        // Unoverridden modalities keep the spec limit.
        assert_eq!(
            TestSpec.validate_media_request_with_limits(
                &metadata,
                &[(Modality::Audio, 2)],
                &overrides
            ),
            Err(ModelRegistryError::ModalityLimitExceeded {
                spec: "test",
                modality: Modality::Audio,
                limit: 1,
                requested: 2,
            })
        );
    }

    #[test]
    fn limit_override_raises_declared_limit() {
        let limits = HashMap::from([(Modality::Image, 2)]);
        let raise = |modality| (modality == Modality::Image).then_some(5);
        assert_eq!(
            check_media_counts("test", &limits, &[(Modality::Image, 5)], raise),
            Ok(())
        );
        assert_eq!(
            check_media_counts("test", &limits, &[(Modality::Image, 6)], raise),
            Err(ModelRegistryError::ModalityLimitExceeded {
                spec: "test",
                modality: Modality::Image,
                limit: 5,
                requested: 6,
            })
        );
    }

    #[test]
    fn limit_override_does_not_enable_undeclared_modality() {
        let limits = HashMap::from([(Modality::Image, 2)]);
        assert_eq!(
            check_media_counts("test", &limits, &[(Modality::Video, 1)], |_| Some(5)),
            Err(ModelRegistryError::UnsupportedModality {
                spec: "test",
                modality: Modality::Video,
            })
        );
    }

    #[test]
    fn env_count_override_ignores_invalid_and_zero_values() {
        static ZERO: OnceLock<Option<usize>> = OnceLock::new();
        static INVALID: OnceLock<Option<usize>> = OnceLock::new();
        static UNSET: OnceLock<Option<usize>> = OnceLock::new();
        static VALID: OnceLock<Option<usize>> = OnceLock::new();

        std::env::set_var("SMG_TEST_MAX_COUNT_ZERO", "0");
        std::env::set_var("SMG_TEST_MAX_COUNT_INVALID", "ten");
        std::env::set_var("SMG_TEST_MAX_COUNT_VALID", "20");

        assert_eq!(env_count_override(&ZERO, "SMG_TEST_MAX_COUNT_ZERO"), None);
        assert_eq!(
            env_count_override(&INVALID, "SMG_TEST_MAX_COUNT_INVALID"),
            None
        );
        assert_eq!(env_count_override(&UNSET, "SMG_TEST_MAX_COUNT_UNSET"), None);
        assert_eq!(
            env_count_override(&VALID, "SMG_TEST_MAX_COUNT_VALID"),
            Some(20)
        );
    }

    struct HookedSpec;

    impl ModelProcessorSpec for HookedSpec {
        fn name(&self) -> &'static str {
            "hooked"
        }

        fn matches(&self, _metadata: &ModelMetadata) -> bool {
            true
        }

        fn hooks(&self, modality: Modality) -> Option<ModalityHooks> {
            match modality {
                Modality::Video => Some(ModalityHooks {
                    placeholder: |_| Ok("<video>".to_string()),
                    placeholder_id: |_| Ok(7),
                    replacements: |_, preprocessed, media, config| {
                        Ok(vec![PromptReplacement::sequence(
                            Modality::Video,
                            "<video>",
                            vec![
                                preprocessed.feature_token_counts.len() as TokenId,
                                media.len() as TokenId,
                                config.temporal_patch_size.unwrap_or(0) as TokenId,
                            ],
                        )])
                    },
                    layouts: || {
                        EncoderFieldLayouts::new(FieldLayout::flat("frames"), HashMap::new())
                    },
                    keep_on_cpu: &["frames"],
                    encoder_input_key: Some("pixel_values_videos"),
                    worker_expandable: true,
                }),
                _ => None,
            }
        }

        fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
            Ok("<image>".to_string())
        }

        fn placeholder_token_id(&self, _metadata: &ModelMetadata) -> RegistryResult<TokenId> {
            Ok(1)
        }

        fn modality_limits(
            &self,
            _metadata: &ModelMetadata,
        ) -> RegistryResult<HashMap<Modality, usize>> {
            Ok(HashMap::from([(Modality::Image, 1), (Modality::Video, 1)]))
        }

        fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
            Ok(json!({}))
        }

        fn prompt_replacements(
            &self,
            _metadata: &ModelMetadata,
            preprocessed: &PreprocessedEncoderInputs,
        ) -> RegistryResult<Vec<PromptReplacement>> {
            Ok(vec![PromptReplacement::sequence(
                Modality::Image,
                "<image>",
                vec![1; preprocessed.feature_token_counts.len()],
            )])
        }
    }

    /// A hooked modality is served from the table; the modalities without
    /// hooks keep the image-only defaults the specs without a table rely on.
    #[test]
    fn hooks_drive_the_per_modality_methods_and_the_defaults_stand_without_them() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({});
        let metadata = ModelMetadata {
            model_id: "hooked",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = HookedSpec;
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(8, 8); 2], &[4, 4]);

        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Video)
                .unwrap(),
            "<video>"
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, Modality::Video)
                .unwrap(),
            7
        );
        assert!(spec.worker_expandable(Modality::Video));
        assert_eq!(
            spec.encoder_field_layouts_for(Modality::Video)
                .encoder_input,
            FieldLayout::flat("frames")
        );
        assert_eq!(spec.keep_on_cpu_keys_for(Modality::Video), ["frames"]);
        assert_eq!(
            spec.encoder_input_key_for(Modality::Video).as_deref(),
            Some("pixel_values_videos")
        );
        // Without media the hook sees no items and the default config.
        let plain = spec
            .prompt_replacements_for(&metadata, &preprocessed, Modality::Video)
            .unwrap();
        assert_eq!(plain[0].tokens, [2, 0, 0]);
        let with_media = spec
            .prompt_replacements_with_media(
                &metadata,
                &preprocessed,
                Modality::Video,
                &[MediaItemInfo::default(), MediaItemInfo::default()],
                &PreProcessorConfig {
                    temporal_patch_size: Some(4),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(with_media[0].tokens, [2, 2, 4]);

        // The image path has no hooks: the required methods answer, as for
        // every spec without a table.
        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Image)
                .unwrap(),
            "<image>"
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, Modality::Image)
                .unwrap(),
            1
        );
        assert!(!spec.worker_expandable(Modality::Image));
        assert_eq!(
            spec.encoder_field_layouts_for(Modality::Image),
            EncoderFieldLayouts::from_legacy_fields(spec.field_layouts())
        );
        assert!(spec.keep_on_cpu_keys_for(Modality::Image).is_empty());
        assert_eq!(spec.encoder_input_key_for(Modality::Image), None);
        assert_eq!(
            spec.prompt_replacements_with_media(
                &metadata,
                &preprocessed,
                Modality::Image,
                &[],
                &PreProcessorConfig::default(),
            )
            .unwrap()[0]
                .tokens,
            [1, 1]
        );
        for modality in [Modality::Audio, Modality::ImageEmbeds] {
            assert!(matches!(
                spec.placeholder_token_for(&metadata, modality),
                Err(ModelRegistryError::UnsupportedModality { spec: "hooked", .. })
            ));
            assert!(matches!(
                spec.prompt_replacements_for(&metadata, &preprocessed, modality),
                Err(ModelRegistryError::UnsupportedModality { spec: "hooked", .. })
            ));
        }
    }
}
