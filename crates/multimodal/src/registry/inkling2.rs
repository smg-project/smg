use std::collections::HashMap;

use serde_json::Value;

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{
        MediaPartOrder, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId},
    vision::{processors::Inkling2ImageProcessor, PreProcessorConfig, VisionPreProcessor},
};

pub(super) struct Inkling2Spec;

impl Inkling2Spec {
    const IMAGE_ANCHOR: &'static str = "<|unused_200054|>";

    fn checked_token(
        metadata: &ModelMetadata,
        token: &str,
        expected: TokenId,
    ) -> RegistryResult<TokenId> {
        let actual = metadata.token_id(token)?;
        if actual != expected {
            return Err(ModelRegistryError::UnsupportedModel(format!(
                "Inkling2 requires token {token} at ID {expected}, not {actual}"
            )));
        }
        Ok(actual)
    }
}

impl ModelProcessorSpec for Inkling2Spec {
    fn name(&self) -> &'static str {
        "inkling2"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata.config_model_type() == Some("inkling2")
            || metadata
                .config
                .get("architectures")
                .and_then(Value::as_array)
                .is_some_and(|items| {
                    items
                        .iter()
                        .any(|item| item.as_str() == Some("Inkling2ForConditionalGeneration"))
                })
    }

    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::Authored
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(Self::IMAGE_ANCHOR.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::checked_token(metadata, Self::IMAGE_ANCHOR, 200054)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 10)]))
    }

    /// Images only, from the checkpoint's exact native policy file; any
    /// other config is refused when the processor is built.
    fn vision_processor(
        &self,
        _model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image => Ok(Box::new(
                Inkling2ImageProcessor::from_preprocessor_config(preprocessor_config).map_err(
                    |error| ModelRegistryError::InvalidPreprocessorConfig {
                        spec: self.name(),
                        modality,
                        reason: error.to_string(),
                    },
                )?,
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
        self.placeholder_token_id(metadata)?;
        let payload = Self::checked_token(metadata, "<|content_image|>", 200005)?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| {
                // The renderer owns the typed header. Only the soft anchor expands.
                PromptReplacement::repeated(Modality::Image, Self::IMAGE_ANCHOR, payload, count)
                    .with_feature_span(0, count)
                    .with_structural_prefix(1)
            })
            .collect())
    }

    fn encoder_field_layouts_for(&self, _modality: Modality) -> EncoderFieldLayouts {
        EncoderFieldLayouts::new(
            FieldLayout::flat("tokens_per_item"),
            HashMap::from([
                ("tokens_per_item".to_string(), FieldLayout::Batched),
                ("vision_grid_thw".to_string(), FieldLayout::Batched),
                ("vision_original_thw".to_string(), FieldLayout::Batched),
            ]),
        )
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec![
            "tokens_per_item".to_string(),
            "vision_grid_thw".to_string(),
            "vision_original_thw".to_string(),
        ]
    }

    fn encoder_input_key_for(&self, modality: Modality) -> Option<String> {
        (modality == Modality::Image).then(|| "vision_patches_bthwc".to_string())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::registry::{test_helpers::TestTokenizer, ModelRegistry};

    #[test]
    fn exact_model_metadata_selects_new_spec_before_old_family() {
        let tokenizer = TestTokenizer::new(&[]);
        for config in [
            json!({"model_type":"inkling2"}),
            json!({"architectures":["Inkling2ForConditionalGeneration"]}),
        ] {
            let metadata = ModelMetadata {
                model_id: "org/inkling-checkpoint",
                tokenizer: &tokenizer,
                config: &config,
            };
            assert_eq!(
                ModelRegistry::new().lookup(&metadata).unwrap().name(),
                "inkling2"
            );
        }
        for config in [
            json!({}),
            json!({"model_type":"inkling_mm_model"}),
            json!({"architectures":["Inkling2ForConditionalGenerationSuffix"]}),
        ] {
            let metadata = ModelMetadata {
                model_id: "org/inkling2",
                tokenizer: &tokenizer,
                config: &config,
            };
            assert!(!Inkling2Spec.matches(&metadata));
        }
    }

    #[test]
    fn image_contract_keeps_header_outside_explicit_features() {
        let tokenizer =
            TestTokenizer::new(&[("<|unused_200054|>", 200054), ("<|content_image|>", 200005)]);
        let config = json!({"model_type":"inkling2"});
        let metadata = ModelMetadata {
            model_id: "checkpoint",
            tokenizer: &tokenizer,
            config: &config,
        };
        let inputs = PreprocessedEncoderInputs::new(
            ndarray::ArrayD::<f32>::zeros(vec![6, 2, 40, 40, 3]),
            vec![2, 4],
            vec![(80, 40), (80, 80)],
        );
        let replacements = Inkling2Spec
            .prompt_replacements(&metadata, &inputs)
            .unwrap();
        for (replacement, count) in replacements.iter().zip([2, 4]) {
            assert_eq!(replacement.placeholder_token, "<|unused_200054|>");
            assert_eq!(replacement.tokens, vec![200005; count]);
            assert_eq!(replacement.structural_prefix, 1);
            assert_eq!(
                replacement.feature_ranges.as_ref().unwrap(),
                &vec![crate::types::PlaceholderRange {
                    offset: 0,
                    length: count
                }]
            );
        }
        assert_eq!(Inkling2Spec.media_part_order(), MediaPartOrder::Authored);
        assert!(Inkling2Spec
            .validate_media_request(&metadata, &[(Modality::Image, 2)])
            .is_ok());
        assert!(Inkling2Spec
            .validate_media_request(&metadata, &[(Modality::Audio, 1)])
            .is_err());
        assert!(Inkling2Spec
            .validate_media_request(&metadata, &[(Modality::Video, 1)])
            .is_err());
        let layouts = Inkling2Spec.encoder_field_layouts_for(Modality::Image);
        assert_eq!(layouts.encoder_input, FieldLayout::flat("tokens_per_item"));
        for key in ["tokens_per_item", "vision_grid_thw", "vision_original_thw"] {
            assert_eq!(layouts.model_specific[key], FieldLayout::Batched);
            assert!(Inkling2Spec.keep_on_cpu_keys().contains(&key.to_string()));
        }
        assert_eq!(
            Inkling2Spec
                .encoder_input_key_for(Modality::Image)
                .as_deref(),
            Some("vision_patches_bthwc")
        );
    }

    #[test]
    fn vision_processor_requires_the_native_policy() {
        let config = json!({"model_type": "inkling2"});
        let native = PreProcessorConfig::from_value(json!({
            "patch_size": 40, "temporal_patch_size": 2,
            "include_extra_column": false,
            "rescale_image_frac": null, "rescale_image_max_upscaled_long_edge": null,
            "stretch_to_padded_grid": false,
            "overlay_rgba_images_on_checkerboard": true, "requires_image_grid_thw": true
        }))
        .unwrap();
        let processor = Inkling2Spec
            .vision_processor(&config, &native, Modality::Image)
            .unwrap();
        assert_eq!(processor.model_name(), "inkling2");
        assert!(matches!(
            Inkling2Spec.vision_processor(&config, &PreProcessorConfig::default(), Modality::Image),
            Err(ModelRegistryError::InvalidPreprocessorConfig { .. })
        ));
        assert!(matches!(
            Inkling2Spec.vision_processor(&config, &native, Modality::Video),
            Err(ModelRegistryError::UnsupportedModality { .. })
        ));
    }

    #[test]
    fn wrong_native_token_ids_fail() {
        let config = json!({});
        for pairs in [
            [("<|unused_200054|>", 1), ("<|content_image|>", 200005)],
            [("<|unused_200054|>", 200054), ("<|content_image|>", 1)],
        ] {
            let tokenizer = TestTokenizer::new(&pairs);
            let metadata = ModelMetadata {
                model_id: "checkpoint",
                tokenizer: &tokenizer,
                config: &config,
            };
            let inputs = PreprocessedEncoderInputs::new(
                ndarray::ArrayD::<f32>::zeros(vec![1]),
                vec![1],
                vec![(1, 1)],
            );
            assert!(Inkling2Spec
                .prompt_replacements(&metadata, &inputs)
                .is_err());
        }
    }
}
