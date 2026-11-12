use std::collections::HashMap;

use serde_json::Value;

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    media::FrameSampling,
    registry::{
        layouts::{grid_patch_fields, grid_patch_layouts_for},
        ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId},
    vision::{PreProcessorConfig, Qwen2VLProcessor, VisionPreProcessor},
};

/// The image placeholder the chat template writes once per image, the one
/// token the expansion repeats; `image_token_id` in the config names its id.
/// Qwen3-Omni (`super::qwen3_omni`) answers the same spelling.
pub(super) const IMAGE_PAD_TOKEN: &str = "<|image_pad|>";

/// The video placeholder the chat template writes once per video, the one
/// token the expansion repeats; `video_token_id` in the config names its id.
const VIDEO_PAD_TOKEN: &str = "<|video_pad|>";

/// Frames per temporal group of the family's own video loader: the default of
/// the loader's sampling kwargs, which the servicer refuses to override for
/// this family. The loader never reads the processor config's
/// `temporal_patch_size`; that one groups the frames the loader hands on, in
/// the processor, padding the last group.
const LOADER_TEMPORAL_PATCH_SIZE: usize = 2;

/// The most frames the family's own video loader hands its processor.
const LOADER_MAX_FRAMES: usize = 768;

/// The frames the Qwen2-VL line's own video loader hands its processor
/// ([`FrameSampling::SteppedGroups`]): `total / fps x 2` frames clamped to
/// [4, 768] in whole temporal groups of two, at a constant step from the
/// first frame. Qwen3-Omni (`super::qwen3_omni`) is served by the same
/// loader and answers the same rule.
pub(super) fn loader_frame_sampling() -> FrameSampling {
    FrameSampling::SteppedGroups {
        temporal_patch_size: LOADER_TEMPORAL_PATCH_SIZE,
        max_frames: LOADER_MAX_FRAMES,
    }
}

pub(super) struct QwenVLVisionSpec;

impl QwenVLVisionSpec {
    fn pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.config_token_id("image_token_id")
    }

    /// A checkpoint serves video when its config names the video placeholder's
    /// id; one that names none keeps video unsupported on every surface.
    fn names_a_video_token(config: &Value) -> bool {
        config
            .get("video_token_id")
            .and_then(Value::as_u64)
            .is_some()
    }

    fn video_pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.config_token_id("video_token_id")
    }
}

impl ModelProcessorSpec for QwenVLVisionSpec {
    fn name(&self) -> &'static str {
        "qwen_vl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("qwen") && id.contains("vl")
            || metadata
                .config_model_type()
                .is_some_and(|mt| mt == "qwen2_vl")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE_PAD_TOKEN.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        // Must match pad_token_id (image_token_id) — this is the repeated token
        // in the expanded sequence. vision_token_id (<|vision_pad|>) is a distinct
        // token the chat template never emits.
        Self::pad_token_id(metadata)
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => self.placeholder_token(metadata),
            Modality::Video if Self::names_a_video_token(metadata.config) => {
                Ok(VIDEO_PAD_TOKEN.to_string())
            }
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
            Modality::Image => Self::pad_token_id(metadata),
            Modality::Video if Self::names_a_video_token(metadata.config) => {
                Self::video_pad_token_id(metadata)
            }
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    /// The family sets no per-request image count of its own: its serving
    /// engines read the image count as unlimited (the family's processing
    /// info declares `image: None`), the checkpoints' processor configs bound
    /// an image's pixels, not the number of images, and the context length is
    /// the practical bound. Images stay declared (an undeclared modality is
    /// refused as unsupported) without a bound of the spec's, so the effective
    /// limit is the engine's advertised one, else the deployment's
    /// `--mm-per-request-image-limit` / `SMG_IMAGE_MAX_COUNT`. A checkpoint
    /// whose config names a video token serves video too, likewise without a
    /// count of the spec's (the same processing info declares `video: None`),
    /// so `SMG_VIDEO_MAX_COUNT` or the engine's advertised limit bounds it.
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
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        let mut limits = HashMap::from([(Modality::Image, usize::MAX)]);
        if Self::names_a_video_token(metadata.config) {
            limits.insert(Modality::Video, usize::MAX);
        }
        Ok(limits)
    }

    fn vision_processor(
        &self,
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image => Ok(Box::new(Qwen2VLProcessor::from_config_for(
                preprocessor_config,
                modality,
            ))),
            Modality::Video if Self::names_a_video_token(model_config) => Ok(Box::new(
                Qwen2VLProcessor::from_config_for(preprocessor_config, modality),
            )),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let pad_token_id = Self::pad_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        // The chat template already wraps each image with <|vision_start|> ... <|vision_end|>,
        // so we only expand the single <|image_pad|> placeholder to N pad tokens.
        Ok(PromptReplacement::repeated_per_item(
            Modality::Image,
            &placeholder_token,
            pad_token_id,
            &preprocessed.feature_token_counts,
        ))
    }

    /// A video's `<|video_pad|>` expands to one pad token per merged patch
    /// of its grid, as the serving engine's own prompt update does
    /// (`grid_t x grid_h x grid_w / merge_size^2`, the processor's token
    /// count per clip); the chat template wraps it in the vision markers
    /// already, and the family's prompt carries no frame stamps.
    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => self.prompt_replacements(metadata, preprocessed),
            Modality::Video if Self::names_a_video_token(metadata.config) => {
                Ok(PromptReplacement::repeated_per_item(
                    Modality::Video,
                    VIDEO_PAD_TOKEN,
                    Self::video_pad_token_id(metadata)?,
                    &preprocessed.feature_token_counts,
                ))
            }
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    /// The family's own video loader samples a clip before its processor sees
    /// it ([`loader_frame_sampling`]), whatever the processor config states.
    fn video_frame_sampling(&self) -> FrameSampling {
        loader_frame_sampling()
    }

    /// One layout per modality for the gateway's assembly: a video batch
    /// carries `patches_per_video` and `video_grid_thw` and no image tensor,
    /// so its patches are sliced by its own sizes key; the image batch keeps
    /// the image layout. The legacy map below answers every modality with the
    /// image layout, which fails assembly for every video.
    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        grid_patch_layouts_for(modality)
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // encoder_input is patchified: [total_patches, patch_features].
        // patches_per_image tells how many patches belong to each image.
        // image_grid_thw is [num_images, 3]; the video fields follow the same
        // shapes per clip, and video_second_per_grid is one value per clip.
        grid_patch_fields()
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string(), "video_grid_thw".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::{
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        types::ImageSize,
    };

    #[test]
    fn qwen_vision_uses_config_token_ids() {
        let tokenizer =
            TestTokenizer::new(&[("<|vision_pad|>", 151654), ("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "image_token_id": 151655,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen spec");
        // The chat template emits <|image_pad|>; it is the prompt anchor and the
        // token the expansion repeats.
        let placeholder_token = spec.placeholder_token(&metadata).unwrap();
        assert_eq!(metadata.token_id(&placeholder_token).unwrap(), 151655);
        assert_eq!(spec.placeholder_token_id(&metadata).unwrap(), 151655);
        // 448/14 = 32 grid, merge_size=2 => (32*32)/4 = 256 tokens
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(448, 448)], &[256]),
            )
            .unwrap();
        // Only pad tokens — vision_start/vision_end are already in the chat template
        assert_eq!(replacements[0].tokens.len(), 256);
        assert_eq!(replacements[0].tokens[0], 151655); // pad (image_token_id)
        assert_eq!(*replacements[0].tokens.last().unwrap(), 151655);
    }

    /// The family bounds its images by context, not by count: the spec
    /// declares the modality without a limit of its own (the ten it carried
    /// was the first registry's default, never a count of the family's), so
    /// a deployment knob is the only router-side bound; a config that names
    /// no video token keeps video unsupported.
    #[test]
    fn qwen_vl_declares_images_without_a_count_of_its_own() {
        use std::collections::HashMap;

        use crate::{
            registry::{ModelProcessorSpec, ModelRegistryError},
            types::Modality,
        };

        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "image_token_id": 151655
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2-VL-7B-Instruct",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = super::QwenVLVisionSpec;

        assert_eq!(
            spec.modality_limits(&metadata).unwrap(),
            HashMap::from([(Modality::Image, usize::MAX)])
        );
        for count in [11, 1_000] {
            spec.validate_media_request(&metadata, &[(Modality::Image, count)])
                .unwrap_or_else(|error| panic!("{count} images: {error}"));
        }
        // A deployment's own cap (the flag, else the environment override)
        // still binds.
        let capped = HashMap::from([(Modality::Image, 10)]);
        assert_eq!(
            spec.validate_media_request_with_limits(&metadata, &[(Modality::Image, 11)], &capped),
            Err(ModelRegistryError::ModalityLimitExceeded {
                spec: "qwen_vl",
                modality: Modality::Image,
                limit: 10,
                requested: 11,
            })
        );
        assert!(matches!(
            spec.validate_media_request(&metadata, &[(Modality::Video, 1)]),
            Err(ModelRegistryError::UnsupportedModality {
                modality: Modality::Video,
                ..
            })
        ));
    }

    /// A checkpoint whose config names a video token serves video: clips
    /// without a count of the spec's own (a deployment cap still binds), each
    /// clip's `<|video_pad|>` expanded to the processor's token count, the
    /// family's own loader rule for the frames, and the video tensors
    /// declared beside the image ones.
    #[test]
    fn qwen_vl_serves_video_when_the_config_names_a_video_token() {
        use std::collections::HashMap;

        use crate::{
            media::FrameSampling,
            registry::{ModelProcessorSpec, ModelRegistryError},
            types::{FieldLayout, Modality},
            vision::PreProcessorConfig,
        };

        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655), ("<|video_pad|>", 151656)]);
        let config = json!({
            "model_type": "qwen2_5_vl",
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "image_token_id": 151655,
            "video_token_id": 151656
        });
        let metadata = ModelMetadata {
            model_id: "Qwen2.5-VL-7B-Instruct",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = super::QwenVLVisionSpec;

        assert_eq!(
            spec.modality_limits(&metadata).unwrap(),
            HashMap::from([(Modality::Image, usize::MAX), (Modality::Video, usize::MAX)])
        );
        for count in [1, 2, 100] {
            spec.validate_media_request(&metadata, &[(Modality::Video, count)])
                .unwrap_or_else(|error| panic!("{count} clips: {error}"));
        }
        // A deployment's own cap (the environment override, or the engine's
        // advertised limit the gateway passes in) still binds.
        let capped = HashMap::from([(Modality::Video, 1)]);
        assert_eq!(
            spec.validate_media_request_with_limits(&metadata, &[(Modality::Video, 2)], &capped),
            Err(ModelRegistryError::ModalityLimitExceeded {
                spec: "qwen_vl",
                modality: Modality::Video,
                limit: 1,
                requested: 2,
            })
        );
        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Video)
                .unwrap(),
            "<|video_pad|>"
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, Modality::Video)
                .unwrap(),
            151656
        );
        // A clip whose grid is 2 x 4 x 4 at merge size 2 is 8 tokens.
        let replacements = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(56, 56)], &[8]),
                Modality::Video,
            )
            .unwrap();
        assert_eq!(replacements.len(), 1);
        assert_eq!(replacements[0].tokens, vec![151656; 8]);
        assert_eq!(
            spec.video_frame_sampling(),
            FrameSampling::SteppedGroups {
                temporal_patch_size: 2,
                max_frames: 768,
            }
        );
        // The processor config's temporal group does not reach the loader's
        // rule: the loader groups by its own default and the processor pads
        // the frames it is handed to the config's group.
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
        spec.vision_processor(
            metadata.config,
            &PreProcessorConfig::default(),
            Modality::Video,
        )
        .expect("the family's processor serves a clip");
        assert_eq!(
            spec.encoder_field_layouts_for(Modality::Video)
                .encoder_input,
            FieldLayout::flat("patches_per_video")
        );
        assert_eq!(
            spec.encoder_field_layouts_for(Modality::Image)
                .encoder_input,
            FieldLayout::flat("patches_per_image")
        );
        let layouts = spec.field_layouts();
        for key in [
            "video_grid_thw",
            "patches_per_video",
            "video_second_per_grid",
        ] {
            assert!(layouts.contains_key(key), "{key}");
        }
        assert!(spec
            .keep_on_cpu_keys()
            .contains(&"video_grid_thw".to_string()));
        assert!(matches!(
            spec.placeholder_token_for(&metadata, Modality::Audio),
            Err(ModelRegistryError::UnsupportedModality {
                modality: Modality::Audio,
                ..
            })
        ));
    }

    #[test]
    fn qwen_vl_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen2_vl",
            "vision_start_token_id": 151652,
            "vision_token_id": 151654,
            "image_token_id": 151655
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("should match qwen alias");
        assert_eq!(spec.name(), "qwen_vl");
    }
}
