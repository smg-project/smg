use std::collections::HashMap;

use serde_json::Value;

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::{PreProcessorConfig, Qwen2VLProcessor, VisionPreProcessor},
};

/// The image placeholder the chat template writes once per image, the one
/// token the expansion repeats; `image_token_id` in the config names its id.
/// Qwen3-Omni (`super::qwen3_omni`) answers the same spelling.
pub(super) const IMAGE_PAD_TOKEN: &str = "<|image_pad|>";

pub(super) struct QwenVLVisionSpec;

impl QwenVLVisionSpec {
    fn pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
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

    /// The family sets no per-request image count of its own: its serving
    /// engines read the image count as unlimited (the family's processing
    /// info declares `image: None`), the checkpoints' processor configs bound
    /// an image's pixels, not the number of images, and the context length is
    /// the practical bound. Images stay declared (an undeclared modality is
    /// refused as unsupported) without a bound of the spec's, so the effective
    /// limit is the engine's advertised one, else the deployment's
    /// `--mm-per-request-image-limit` / `SMG_IMAGE_MAX_COUNT`.
    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, usize::MAX)]))
    }

    fn vision_processor(
        &self,
        _model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image => Ok(Box::new(Qwen2VLProcessor::from_preprocessor_config(
                preprocessor_config,
            ))),
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

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // encoder_input is patchified: [total_patches, patch_features].
        // patches_per_image tells how many patches belong to each image.
        // image_grid_thw is [num_images, 3].
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string()]
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
    /// a deployment knob is the only router-side bound and video stays
    /// unsupported.
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
