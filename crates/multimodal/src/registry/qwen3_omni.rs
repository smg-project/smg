use std::collections::HashMap;

use serde_json::Value;

use crate::{
    audio::{AudioPreProcessor, Qwen3AudioProcessor},
    encoder_inputs::PreprocessedEncoderInputs,
    media::FrameSampling,
    registry::{
        layouts::grid_patch_layouts_for,
        qwen_vl::{loader_frame_sampling, IMAGE_PAD_TOKEN},
        ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId},
    vision::{PreProcessorConfig, Qwen3OmniVisionProcessor, VisionPreProcessor},
};

/// The video placeholder: the spelling the spec answers when the tokenizer
/// cannot name the thinker config's `video_token_id`.
const VIDEO_PAD_TOKEN: &str = "<|video_pad|>";
/// The audio placeholder, answered the same way for `audio_token_id`;
/// Qwen3-ASR (`super::qwen3_asr`) reads it.
pub(super) const AUDIO_PAD_TOKEN: &str = "<|audio_pad|>";

pub(super) struct Qwen3OmniSpec;

impl Qwen3OmniSpec {
    fn token_id(metadata: &ModelMetadata, field: &str) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["thinker_config", field])
            .or_else(|| metadata.config_u32(&[field]))
            .map(|value| value as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: format!("thinker_config.{field}"),
            })
    }

    fn token(metadata: &ModelMetadata, field: &str, fallback: &str) -> RegistryResult<String> {
        let token_id = Self::token_id(metadata, field)?;
        if let Some(token) = metadata.tokenizer.id_to_token(token_id as u32) {
            return Ok(token);
        }
        metadata.token_id(fallback)?;
        Ok(fallback.to_string())
    }

    fn replacements(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
        field: &str,
        fallback: &str,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let token_id = Self::token_id(metadata, field)?;
        let token = Self::token(metadata, field, fallback)?;
        Ok(PromptReplacement::repeated_per_item(
            modality,
            &token,
            token_id,
            &preprocessed.feature_token_counts,
        ))
    }
}

impl ModelProcessorSpec for Qwen3OmniSpec {
    fn name(&self) -> &'static str {
        "qwen3_omni"
    }

    /// `<|image_pad|>`, `<|video_pad|>` and `<|audio_pad|>` (the thinker
    /// config's token ids) are the single-token targets the serving engine's
    /// prompt updates expand, so a worker can expand them itself.
    fn worker_expandable(&self, modality: Modality) -> bool {
        matches!(
            modality,
            Modality::Image | Modality::Video | Modality::Audio
        )
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let model_id = metadata.model_id.to_ascii_lowercase();
        model_id.contains("qwen3-omni")
            || model_id.contains("qwen3_omni")
            || metadata.config_model_type().is_some_and(|model_type| {
                model_type == "qwen3_omni_moe" || model_type == "qwen3_omni_moe_thinker"
            })
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        self.placeholder_token_for(metadata, Modality::Image)
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        self.placeholder_token_id_for(metadata, Modality::Image)
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => Self::token(metadata, "image_token_id", IMAGE_PAD_TOKEN),
            Modality::Video => Self::token(metadata, "video_token_id", VIDEO_PAD_TOKEN),
            Modality::Audio => Self::token(metadata, "audio_token_id", AUDIO_PAD_TOKEN),
            Modality::ImageEmbeds => Err(ModelRegistryError::UnsupportedModality {
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
            Modality::Image => Self::token_id(metadata, "image_token_id"),
            Modality::Video => Self::token_id(metadata, "video_token_id"),
            Modality::Audio => Self::token_id(metadata, "audio_token_id"),
            Modality::ImageEmbeds => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    /// The family sets no per-request media count of its own for any of its
    /// three modalities: its serving engines read all three as unlimited (the
    /// family's processing info declares `audio: None`, `image: None`,
    /// `video: None`), the checkpoint's processor config bounds an image's
    /// pixels, a clip's frames and a recording's length, not how many of
    /// each, and the context length is the practical bound. The three stay
    /// declared (an undeclared modality is refused as unsupported) without a
    /// bound of the spec's, so the effective limit per modality is the
    /// engine's advertised one, else the deployment's
    /// `--mm-per-request-image-limit` / `SMG_IMAGE_MAX_COUNT`,
    /// `SMG_VIDEO_MAX_COUNT` or `SMG_AUDIO_MAX_COUNT`.
    ///
    /// What still bounds such a request, and where it lives. Before any
    /// fetch: the per-request limit the model's workers advertise for their
    /// engine, which caps the router's own limit in the gateway's limit
    /// resolution, and the deployment's `SMG_IMAGE_MAX_COUNT` /
    /// `SMG_VIDEO_MAX_COUNT` / `SMG_AUDIO_MAX_COUNT` (and
    /// `--mm-per-request-image-limit` for images), read in the same
    /// resolution. At or after the fetch: the gateway's cap on media bytes
    /// in flight (`multimodal_max_inflight_bytes`), the engine's advertised
    /// encoder budget and the context length. A number of the spec's own
    /// would be a declared default again, not a bound of the family's.
    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([
            (Modality::Image, usize::MAX),
            (Modality::Video, usize::MAX),
            (Modality::Audio, usize::MAX),
        ]))
    }

    fn vision_processor(
        &self,
        _model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image | Modality::Video => Ok(Box::new(
                Qwen3OmniVisionProcessor::from_config_for(preprocessor_config, modality),
            )),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn audio_processor(
        &self,
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
    ) -> RegistryResult<Option<Box<dyn AudioPreProcessor>>> {
        Ok(Some(Box::new(Qwen3AudioProcessor::from_configs(
            model_config,
            preprocessor_config,
        ))))
    }

    /// The family's video loader is the Qwen2-VL line's: a clip is sampled
    /// before its processor sees it, `total / fps x 2` frames in whole
    /// temporal groups of two, at most 768 ([`FrameSampling::SteppedGroups`]),
    /// so a 5-frame clip at 1 fps is four frames, two temporal groups,
    /// whatever the processor config states.
    fn video_frame_sampling(&self) -> FrameSampling {
        loader_frame_sampling()
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        self.prompt_replacements_for(metadata, preprocessed, Modality::Image)
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => Self::replacements(
                metadata,
                preprocessed,
                modality,
                "image_token_id",
                IMAGE_PAD_TOKEN,
            ),
            Modality::Video => Self::replacements(
                metadata,
                preprocessed,
                modality,
                "video_token_id",
                VIDEO_PAD_TOKEN,
            ),
            Modality::Audio => Self::replacements(
                metadata,
                preprocessed,
                modality,
                "audio_token_id",
                AUDIO_PAD_TOKEN,
            ),
            Modality::ImageEmbeds => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match modality {
            Modality::Image | Modality::Video => grid_patch_layouts_for(modality),
            Modality::Audio => EncoderFieldLayouts::new(
                FieldLayout::Batched,
                HashMap::from([
                    ("feature_attention_mask".to_string(), FieldLayout::Batched),
                    ("audio_feature_lengths".to_string(), FieldLayout::Batched),
                ]),
            ),
            Modality::ImageEmbeds => EncoderFieldLayouts::default(),
        }
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string(), "video_grid_thw".to_string()]
    }

    fn keep_on_cpu_keys_for(&self, modality: Modality) -> Vec<String> {
        match modality {
            Modality::Image => vec!["image_grid_thw".to_string()],
            Modality::Video => vec!["video_grid_thw".to_string()],
            Modality::Audio | Modality::ImageEmbeds => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        registry::{test_helpers::*, ModelRegistry},
        types::ImageSize,
    };

    fn omni_tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            (AUDIO_PAD_TOKEN, 151675),
            (IMAGE_PAD_TOKEN, 151655),
            (VIDEO_PAD_TOKEN, 151656),
        ])
    }

    #[test]
    fn omni_accepts_mixed_modalities_and_uses_nested_tokens() {
        let tokenizer = omni_tokenizer();
        let config = json!({
            "model_type": "qwen3_omni_moe",
            "thinker_config": {
                "audio_token_id": 151675,
                "image_token_id": 151655,
                "video_token_id": 151656
            }
        });
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-Omni-30B-A3B-Thinking",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).unwrap();
        assert_eq!(spec.name(), "qwen3_omni");
        assert_eq!(
            spec.placeholder_token(&metadata).unwrap(),
            spec.placeholder_token_for(&metadata, Modality::Image)
                .unwrap()
        );
        assert_eq!(
            spec.placeholder_token_id(&metadata).unwrap(),
            spec.placeholder_token_id_for(&metadata, Modality::Image)
                .unwrap()
        );
        spec.validate_media_request(
            &metadata,
            &[
                (Modality::Image, 2),
                (Modality::Video, 1),
                (Modality::Audio, 2),
            ],
        )
        .unwrap();

        for (modality, expected) in [
            (Modality::Image, 151655),
            (Modality::Video, 151656),
            (Modality::Audio, 151675),
        ] {
            let replacements = spec
                .prompt_replacements_for(
                    &metadata,
                    &test_preprocessed_with_tokens(&[ImageSize::new(32, 32)], &[3]),
                    modality,
                )
                .unwrap();
            assert_eq!(replacements[0].tokens, vec![expected; 3]);
        }
    }

    /// The family bounds its media by context, not by count: the spec
    /// declares its three modalities without a limit of its own (the ten, one
    /// and ten it carried were declared defaults, never counts of the
    /// family's), so a deployment knob or the engine's advertised limit is the
    /// only router-side bound for each.
    #[test]
    fn omni_declares_its_media_without_a_count_of_its_own() {
        let tokenizer = omni_tokenizer();
        let config = json!({
            "model_type": "qwen3_omni_moe",
            "thinker_config": {
                "audio_token_id": 151675,
                "image_token_id": 151655,
                "video_token_id": 151656
            }
        });
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = Qwen3OmniSpec;

        assert_eq!(
            spec.modality_limits(&metadata).unwrap(),
            HashMap::from([
                (Modality::Image, usize::MAX),
                (Modality::Video, usize::MAX),
                (Modality::Audio, usize::MAX),
            ])
        );
        // Above the old declared defaults each modality passes the spec's
        // own validation; a deployment's cap (the flag or the environment
        // override, or the engine's advertised limit the gateway passes in)
        // still binds.
        for (modality, cap) in [
            (Modality::Image, 10),
            (Modality::Video, 1),
            (Modality::Audio, 10),
        ] {
            for count in [cap + 1, 1_000] {
                spec.validate_media_request(&metadata, &[(modality, count)])
                    .unwrap_or_else(|error| panic!("{count} {modality} items: {error}"));
            }
            let capped = HashMap::from([(modality, cap)]);
            assert_eq!(
                spec.validate_media_request_with_limits(&metadata, &[(modality, cap + 1)], &capped),
                Err(ModelRegistryError::ModalityLimitExceeded {
                    spec: "qwen3_omni",
                    modality,
                    limit: cap,
                    requested: cap + 1,
                })
            );
        }
    }

    /// Every modality's anchor is the one token the serving engine expands,
    /// so images, video and audio may all be forwarded to a worker that
    /// processes media itself.
    #[test]
    fn omni_media_can_be_expanded_by_a_worker() {
        for modality in [Modality::Image, Modality::Video, Modality::Audio] {
            assert!(Qwen3OmniSpec.worker_expandable(modality), "{modality}");
        }
        assert!(!Qwen3OmniSpec.worker_expandable(Modality::ImageEmbeds));
    }

    /// A clip is sampled the way the family's own video loader samples it:
    /// a 5-frame clip at 1 fps is its first four frames (two temporal groups),
    /// not ten frames at the nominal 2 fps with every frame doubled; the
    /// temporal group is the loader's own, not the processor config's; a
    /// one-frame clip, shorter than one temporal group, gets no frame, as on
    /// the engine's own server.
    #[test]
    fn omni_video_frames_follow_the_qwen2_vl_loader_rule() {
        use crate::media::{sampled_frame_indices, VideoFetchConfig};

        let spec = Qwen3OmniSpec;
        assert_eq!(
            spec.video_frame_sampling(),
            FrameSampling::SteppedGroups {
                temporal_patch_size: 2,
                max_frames: 768,
            }
        );
        let config = PreProcessorConfig {
            temporal_patch_size: Some(4),
            ..PreProcessorConfig::default()
        };
        assert_eq!(
            spec.video_frame_sampling_with(&config),
            FrameSampling::SteppedGroups {
                temporal_patch_size: 2,
                max_frames: 768,
            }
        );
        let fetch = VideoFetchConfig {
            sampling: spec.video_frame_sampling(),
            ..VideoFetchConfig::default()
        };
        assert_eq!(sampled_frame_indices(5, 1.0, fetch), vec![0, 1, 2, 3]);
        assert_eq!(
            sampled_frame_indices(150, 30.0, fetch),
            (0..10).map(|group| group * 15).collect::<Vec<_>>()
        );
        assert!(sampled_frame_indices(1, 30.0, fetch).is_empty());
    }

    #[test]
    fn omni_layouts_are_modality_specific() {
        let image = Qwen3OmniSpec.encoder_field_layouts_for(Modality::Image);
        assert_eq!(image.encoder_input, FieldLayout::flat("patches_per_image"));
        assert!(image.model_specific.contains_key("image_grid_thw"));

        let video = Qwen3OmniSpec.encoder_field_layouts_for(Modality::Video);
        assert_eq!(video.encoder_input, FieldLayout::flat("patches_per_video"));
        assert!(video.model_specific.contains_key("video_grid_thw"));

        let audio = Qwen3OmniSpec.encoder_field_layouts_for(Modality::Audio);
        assert_eq!(audio.encoder_input, FieldLayout::Batched);
        assert!(audio.model_specific.contains_key("feature_attention_mask"));
    }

    #[test]
    fn omni_keep_on_cpu_keys_are_modality_specific() {
        assert_eq!(
            Qwen3OmniSpec.keep_on_cpu_keys_for(Modality::Image),
            vec!["image_grid_thw"]
        );
        assert_eq!(
            Qwen3OmniSpec.keep_on_cpu_keys_for(Modality::Video),
            vec!["video_grid_thw"]
        );
        assert!(Qwen3OmniSpec
            .keep_on_cpu_keys_for(Modality::Audio)
            .is_empty());
    }

    #[test]
    fn omni_spec_builds_qwen_audio_processor() {
        use std::sync::Arc;

        use bytes::Bytes;

        use crate::{
            audio::DecodedAudio,
            types::{AudioClip, AudioSource},
            vision::PreProcessorConfig,
        };

        let tokenizer = omni_tokenizer();
        let config = json!({"model_type": "qwen3_omni_moe"});
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-Omni-30B-A3B-Thinking",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).unwrap();

        let preprocessor_config = PreProcessorConfig::from_json(
            r#"{"feature_size": 16, "sampling_rate": 16000, "n_fft": 400, "hop_length": 160}"#,
        )
        .unwrap();
        let processor = spec
            .audio_processor(&config, &preprocessor_config)
            .unwrap()
            .expect("qwen3_omni spec must provide an audio processor");

        let clip = Arc::new(AudioClip::new(
            Bytes::from_static(b"audio"),
            DecodedAudio {
                samples: vec![0.0; 800],
                sample_rate: 16_000,
            },
            AudioSource::InlineBytes,
            "audio-hash".to_string(),
        ));
        let result = processor.preprocess(&[clip]).unwrap();
        assert_eq!(result.encoder_input.shape(), &[1, 16, 5]);
    }
}
