use std::{collections::HashMap, sync::OnceLock};

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine};
use serde_json::Value;
use thiserror::Error;

use crate::{
    audio::AudioPreProcessor,
    encoder_inputs::PreprocessedEncoderInputs,
    media::FrameSampling,
    types::{
        EncoderFieldLayouts, FieldLayout, MediaContentPart, Modality, PromptReplacement, TokenId,
        VideoSamplingInfo,
    },
    vision::PreProcessorConfig,
};

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
    #[error("{format} images are not accepted by model spec {spec}")]
    UnsupportedImageFormat { spec: &'static str, format: String },
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

/// Deployment-wide override of a spec's per-request media-count limit.
///
/// The override replaces the spec's declared limit (it can raise or lower
/// it) but never enables a modality the spec does not declare. Unset,
/// non-numeric, or zero values are ignored.
fn modality_limit_override(modality: Modality) -> Option<usize> {
    match modality {
        Modality::Image => env_count_override(&IMAGE_MAX_COUNT_OVERRIDE, "SMG_IMAGE_MAX_COUNT"),
        Modality::Video => env_count_override(&VIDEO_MAX_COUNT_OVERRIDE, "SMG_VIDEO_MAX_COUNT"),
        Modality::Audio => env_count_override(&AUDIO_MAX_COUNT_OVERRIDE, "SMG_AUDIO_MAX_COUNT"),
        Modality::ImageEmbeds => None,
    }
}

/// Base64 characters decoded to sniff a data URL's format: 96 bytes, past
/// every magic signature `image::guess_format` reads.
const SNIFF_BASE64_CHARS: usize = 128;

/// The image format a content part names: sniffed from the bytes (inline,
/// or decoded from a data URL) before the client's media type is believed,
/// else the media type in any case, else the URL path's extension. `None`
/// when nothing says (a URL without an extension, another modality).
fn image_format_of(part: &MediaContentPart) -> Option<image::ImageFormat> {
    let from_media_type = |media_type: &str| {
        image::ImageFormat::from_mime_type(media_type.trim().to_ascii_lowercase())
    };
    match part {
        MediaContentPart::ImageData {
            data, mime_type, ..
        } => image::guess_format(data)
            .ok()
            .or_else(|| mime_type.as_deref().and_then(from_media_type)),
        MediaContentPart::ImageUrl { url, .. } => {
            if let Some(rest) = url.strip_prefix("data:") {
                let (header, payload) = rest.split_once(',').unwrap_or((rest, ""));
                let mut parameters = header.split(';');
                let media_type = parameters.next().unwrap_or_default();
                // The magic bytes sit in the first few dozen bytes: decode a
                // bounded prefix (whole base64 quads), not a copy of the image.
                let sniffed = parameters
                    .any(|parameter| parameter.trim().eq_ignore_ascii_case("base64"))
                    .then(|| {
                        // Bytes, not chars: base64 is ASCII, anything else
                        // fails to decode, and a byte prefix cannot panic.
                        let payload = payload.trim().as_bytes();
                        let end = payload.len().min(SNIFF_BASE64_CHARS);
                        BASE64_STANDARD.decode(&payload[..end - end % 4]).ok()
                    })
                    .flatten()
                    .and_then(|bytes| image::guess_format(&bytes).ok());
                return sniffed.or_else(|| from_media_type(media_type));
            }
            let path = url.split(['?', '#']).next().unwrap_or_default();
            let extension = path.rsplit('/').next()?.rsplit_once('.')?.1;
            image::ImageFormat::from_extension(extension)
        }
        _ => None,
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
    /// Emit media parts before text (vLLM `interleave_mm_strings=false`).
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

pub trait ModelProcessorSpec: Send + Sync {
    fn name(&self) -> &'static str;
    fn matches(&self, metadata: &ModelMetadata) -> bool;

    /// Ordering of media/text parts this model's chat template expects. Most
    /// models use vLLM-compatible media-first; families whose template renders
    /// parts positionally override to `Authored`.
    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::MediaFirst
    }

    /// Whether the anchor rendered for `modality` is exactly the single token
    /// vLLM's prompt updates target, so a worker can expand it itself.
    fn worker_expandable(&self, _modality: Modality) -> bool {
        false
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String>;
    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId>;
    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => self.placeholder_token(metadata),
            _ => Err(ModelRegistryError::UnsupportedModality {
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
        match modality {
            Modality::Image => self.placeholder_token_id(metadata),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
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

    /// Image formats the model's vendor refuses; the gateway refuses them
    /// up front, judged from the bytes, the data URL's type or the URL path.
    fn rejected_image_formats(&self) -> &'static [image::ImageFormat] {
        &[]
    }

    /// Refuse an image part whose format is one of
    /// [`Self::rejected_image_formats`].
    fn validate_image_formats(&self, parts: &[MediaContentPart]) -> RegistryResult<()> {
        let rejected = self.rejected_image_formats();
        if rejected.is_empty() {
            return Ok(());
        }
        for part in parts {
            if let Some(format) = image_format_of(part).filter(|f| rejected.contains(f)) {
                return Err(ModelRegistryError::UnsupportedImageFormat {
                    spec: self.name(),
                    format: format
                        .extensions_str()
                        .first()
                        .map_or("unknown", |s| *s)
                        .to_string(),
                });
            }
        }
        Ok(())
    }

    fn processor_kwargs(&self, metadata: &ModelMetadata) -> RegistryResult<Value>;

    /// Build the audio preprocessor for this model, if it supports audio.
    ///
    /// This is the single source of truth for audio-processor selection: the
    /// same spec that owns a model's prompt/placeholder logic also owns its
    /// audio preprocessor factory, so there is no separate string-keyed
    /// registry to keep in sync. Audio-less specs use the default (`None`).
    ///
    /// The processor is built from the current model config because its feature
    /// shapes and quantization parameters can be checkpoint-specific.
    fn audio_processor(
        &self,
        _model_config: &Value,
        _preprocessor_config: &PreProcessorConfig,
    ) -> Option<Box<dyn AudioPreProcessor>> {
        None
    }

    /// Compute per-media prompt replacement token sequences.
    ///
    /// Receives the full preprocessed output so each model can extract whatever
    /// metadata it needs (e.g. aspect_ratios for tile-based models).  This
    /// mirrors vLLM's `_get_prompt_updates(out_mm_kwargs)` pattern.
    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>>;
    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => self.prompt_replacements(metadata, preprocessed),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    /// Prompt replacements given one [`MediaItemInfo`] per media item in batch order and the config the modality was preprocessed with; the default ignores both.
    fn prompt_replacements_with_media(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
        _media: &[MediaItemInfo],
        _preprocessor_config: &PreProcessorConfig,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        self.prompt_replacements_for(metadata, preprocessed, modality)
    }

    /// Declare how each tensor's first dimension maps to media items.
    ///
    /// Keys not listed are treated as shared (replicated across all media items).
    /// The `"pixel_values"` key mirrors HF/vLLM vision kwargs and should be
    /// included when the primary encoder input differs from batched layout.
    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // Default: encoder_input is batched (most models).
        HashMap::from([("pixel_values".to_string(), FieldLayout::Batched)])
    }

    /// Declare the neutral primary/side-tensor layout contract for one modality.
    ///
    /// The default converts the legacy HF/vLLM-shaped field map so existing
    /// vision specs remain source-compatible. New multimodal specs should
    /// override this method and keep backend-specific field names at adapters.
    fn encoder_field_layouts_for(&self, _modality: Modality) -> EncoderFieldLayouts {
        EncoderFieldLayouts::from_legacy_fields(self.field_layouts())
    }

    /// Tensor keys that should remain on CPU (not transferred to GPU).
    ///
    /// In vLLM, certain model-specific tensors are marked `keep_on_cpu=True`
    /// in their `MultiModalFieldConfig`.  This method mirrors that per-model
    /// knowledge so the router can send the hint via gRPC, avoiding the need
    /// for the backend to instantiate a Python processor just to query it.
    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec![]
    }

    /// Tensor keys that should remain on CPU for one modality.
    ///
    /// The default preserves the legacy model-wide declaration.
    fn keep_on_cpu_keys_for(&self, _modality: Modality) -> Vec<String> {
        self.keep_on_cpu_keys()
    }

    /// The engine-side name of the primary encoder input for one modality,
    /// when it is not the HF-conventional `pixel_values` / `audio_features`
    /// (DeepSeek-V4.1's forward pops `patches`). `None` keeps the default
    /// name; adapters that build the engine request consult this before
    /// naming the tensor.
    fn encoder_input_key_for(&self, _modality: Modality) -> Option<String> {
        None
    }

    /// Frame rate to sample a video at when the request names no `fps`: the
    /// model's reference processor default, so token counts match it. `None`
    /// keeps the media connector's default.
    fn default_video_sample_fps(&self) -> Option<f32> {
        None
    }

    /// Where a video's sampled frames sit: evenly spread, or one per interval
    /// from the start with the last frame kept, as the model's reference does.
    fn video_frame_sampling(&self) -> FrameSampling {
        FrameSampling::Even
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
    fn image_formats_are_accepted_unless_a_spec_rejects_them() {
        assert!(TestSpec.rejected_image_formats().is_empty());
        let bmp = MediaContentPart::ImageData {
            data: b"BM\x00\x00".to_vec(),
            mime_type: None,
            uuid: None,
            detail: None,
        };
        assert!(TestSpec
            .validate_image_formats(std::slice::from_ref(&bmp))
            .is_ok());

        struct NoBmp;
        impl ModelProcessorSpec for NoBmp {
            fn name(&self) -> &'static str {
                "no-bmp"
            }
            fn matches(&self, _metadata: &ModelMetadata) -> bool {
                true
            }
            fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
                Ok("<img>".to_string())
            }
            fn placeholder_token_id(&self, _metadata: &ModelMetadata) -> RegistryResult<TokenId> {
                Ok(1)
            }
            fn modality_limits(
                &self,
                _metadata: &ModelMetadata,
            ) -> RegistryResult<HashMap<Modality, usize>> {
                Ok(HashMap::from([(Modality::Image, 1)]))
            }
            fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
                Ok(Value::Null)
            }
            fn prompt_replacements(
                &self,
                _metadata: &ModelMetadata,
                _preprocessed: &PreprocessedEncoderInputs,
            ) -> RegistryResult<Vec<PromptReplacement>> {
                Ok(Vec::new())
            }
            fn rejected_image_formats(&self) -> &'static [image::ImageFormat] {
                &[image::ImageFormat::Bmp]
            }
        }
        let png = MediaContentPart::ImageUrl {
            url: "https://a/x.png".to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        };
        assert!(NoBmp
            .validate_image_formats(std::slice::from_ref(&png))
            .is_ok());
        // Sniffed from the bytes (inline, or decoded from a data URL, before
        // the client's media type is believed), from a data URL's type in
        // any case, or from the URL path.
        let data_url = |url: &str| MediaContentPart::ImageUrl {
            url: url.to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        };
        let rejected = [
            bmp,
            MediaContentPart::ImageData {
                data: b"zz".to_vec(),
                mime_type: Some("IMAGE/BMP".to_string()),
                uuid: None,
                detail: None,
            },
            data_url("data:image/bmp;base64,Qk0AAA=="),
            data_url("data:image/BMP;base64,Qk0AAA=="),
            // BMP bytes behind a lying media type, short and past the sniffed prefix.
            data_url("data:image/png;base64,Qk0AAAAA"),
            data_url(&format!(
                "data:image/png;base64,{}",
                BASE64_STANDARD.encode([b"BM".as_slice(), &[0u8; 4096]].concat())
            )),
            data_url("https://a/scan.BMP?x=1"),
        ];
        for part in &rejected {
            assert_eq!(
                NoBmp.validate_image_formats(std::slice::from_ref(part)),
                Err(ModelRegistryError::UnsupportedImageFormat {
                    spec: "no-bmp",
                    format: "bmp".to_string(),
                }),
                "{part:?}"
            );
        }
        // Client-supplied payloads that are not base64 at all, multi-byte
        // characters included, are sniffed without panicking and fall back
        // to the media type.
        for url in [
            "data:image/png;base64,aéé",
            "data:image/bmp;base64,ééé",
            "data:image/bmp;base64,",
        ] {
            let expected = url.starts_with("data:image/bmp").then(|| {
                ModelRegistryError::UnsupportedImageFormat {
                    spec: "no-bmp",
                    format: "bmp".to_string(),
                }
            });
            assert_eq!(
                NoBmp
                    .validate_image_formats(std::slice::from_ref(&data_url(url)))
                    .err(),
                expected,
                "{url}"
            );
        }
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
}
