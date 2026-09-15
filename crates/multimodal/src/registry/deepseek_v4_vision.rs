use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{
        MediaPartOrder, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
        ToolResultOrder,
    },
    types::{EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId},
};

pub(crate) const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

pub(super) struct DeepseekV4VisionSpec;

impl DeepseekV4VisionSpec {
    fn config_or(metadata: &ModelMetadata, key: &str, default: u32) -> u32 {
        metadata.config_u32(&[key]).unwrap_or(default)
    }

    fn placeholder_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id(IMAGE_PLACEHOLDER)
    }
}

impl ModelProcessorSpec for DeepseekV4VisionSpec {
    fn name(&self) -> &'static str {
        "deepseek_v4_vision"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata.config_model_type() == Some("deepseek_v4")
            && metadata
                .config_u32(&["vision_n_layers"])
                .is_some_and(|n| n > 0)
    }

    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::Authored
    }

    fn requires_structured_chat_content(&self) -> bool {
        true
    }

    fn tool_result_order(&self) -> ToolResultOrder {
        ToolResultOrder::AssistantCall
    }

    fn offset_dependent_replacements(&self) -> bool {
        true
    }

    fn rebuild_replacement_at_offset(
        &self,
        _metadata: &ModelMetadata,
        replacement: &PromptReplacement,
        start_offset: usize,
    ) -> RegistryResult<(PromptReplacement, Option<u32>)> {
        let compress_pad = 3 - start_offset % 4;
        let placeholder_id = replacement.tokens.first().copied().ok_or_else(|| {
            ModelRegistryError::InvalidPreprocessedField {
                field: "feature_token_counts".to_string(),
            }
        })?;
        let rebuilt = PromptReplacement::repeated(
            Modality::Image,
            IMAGE_PLACEHOLDER,
            placeholder_id,
            compress_pad + replacement.tokens.len(),
        );
        Ok((rebuilt, Some(compress_pad as u32)))
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE_PLACEHOLDER.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::placeholder_id(metadata)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, usize::MAX)]))
    }

    fn processor_kwargs(&self, metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({
            "vision_patch_size": Self::config_or(metadata, "vision_patch_size", 14),
            "vision_downsample_ratio": Self::config_or(metadata, "vision_downsample_ratio", 3),
            "vision_max_n_token": Self::config_or(metadata, "vision_max_n_token", 384),
            "vision_min_pixels": Self::config_or(metadata, "vision_min_pixels", 147456),
            "vision_max_wh_ratio": Self::config_or(metadata, "vision_max_wh_ratio", 8),
        }))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let placeholder_id = Self::placeholder_id(metadata)?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&num_tokens| {
                PromptReplacement::repeated(
                    Modality::Image,
                    IMAGE_PLACEHOLDER,
                    placeholder_id,
                    num_tokens,
                )
            })
            .collect())
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        if modality != Modality::Image {
            return EncoderFieldLayouts::default();
        }
        EncoderFieldLayouts::new(
            FieldLayout::flat("patches_per_image"),
            HashMap::from([
                ("patches_per_image".to_string(), FieldLayout::Batched),
                ("n_vit_h".to_string(), FieldLayout::Batched),
                ("n_vit_w".to_string(), FieldLayout::Batched),
                ("n_llm_h".to_string(), FieldLayout::Batched),
                ("n_llm_w".to_string(), FieldLayout::Batched),
                ("safe_resize_iterations".to_string(), FieldLayout::Batched),
            ]),
        )
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::registry::{test_helpers::TestTokenizer, ModelRegistry};

    fn metadata<'a>(tokenizer: &'a TestTokenizer, config: &'a Value) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
            tokenizer,
            config,
        }
    }

    #[test]
    fn detection_requires_model_type_and_positive_vision_depth() {
        let tokenizer = TestTokenizer::new(&[(IMAGE_PLACEHOLDER, 129281)]);
        let registry = ModelRegistry::new();
        for (model_type, layers, expected) in [
            ("deepseek_v4", 32, true),
            ("deepseek_v4", 0, false),
            ("other", 32, false),
        ] {
            let config = json!({"model_type": model_type, "vision_n_layers": layers});
            let found = registry.lookup(&metadata(&tokenizer, &config));
            assert_eq!(
                found.map(ModelProcessorSpec::name),
                expected.then_some("deepseek_v4_vision")
            );
        }
    }

    #[test]
    fn replacement_rebuild_adds_offset_variant() {
        let tokenizer = TestTokenizer::new(&[(IMAGE_PLACEHOLDER, 129281)]);
        let config = json!({"model_type": "deepseek_v4", "vision_n_layers": 32});
        let metadata = metadata(&tokenizer, &config);
        let spec = DeepseekV4VisionSpec;
        let replacement =
            PromptReplacement::repeated(Modality::Image, IMAGE_PLACEHOLDER, 129281, 114);
        for residue in 0..4 {
            let (rebuilt, variant) = spec
                .rebuild_replacement_at_offset(&metadata, &replacement, residue)
                .unwrap();
            assert_eq!(variant, Some((3 - residue) as u32));
            assert_eq!(rebuilt.tokens.len(), 114 + 3 - residue);
            assert!(rebuilt.feature_ranges.is_none());
        }
    }

    #[test]
    fn dsv4_contract_is_authored_structured_and_offset_dependent() {
        let spec = DeepseekV4VisionSpec;
        assert_eq!(spec.media_part_order(), MediaPartOrder::Authored);
        assert!(spec.requires_structured_chat_content());
        assert_eq!(spec.tool_result_order(), ToolResultOrder::AssistantCall);
        assert!(spec.offset_dependent_replacements());
    }
}
