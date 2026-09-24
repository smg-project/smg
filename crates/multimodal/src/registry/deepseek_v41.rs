//! DeepSeek-V4.1 vision model contract.
//!
//! Mirrors the checkpoint's `inference/image_processor.py` and vLLM's
//! `models/deepseek_v4_1/common/mm_preprocess.py`: each `<｜deepseek_image｜>`
//! placeholder expands to a span where every position carries
//! `image_token_id` (the roles ride in the per-image `types` tensor). The
//! span is spliced verbatim — vLLM dropped its compressor-alignment pad
//! (vllm-project/vllm#56554), so no pad token is prepended.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{
        MediaPartOrder, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
};

/// The placeholder text inlined at each image's position; the multimodal
/// pipeline later expands it into the image span. Public so the renderer can
/// emit it without re-resolving the spec (mirrors the reference encoding's
/// hard-coded `IMAGE_PLACEHOLDER`).
pub const DEEPSEEK_V41_IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

pub(super) struct DeepseekV41VisionSpec;

impl DeepseekV41VisionSpec {
    /// vLLM leaves the image count unlimited for this model
    /// (`get_supported_mm_limits` returns `{"image": None}`); keep a generous
    /// cap until the gateway can express "unlimited". Deployments raise or
    /// lower it with `SMG_IMAGE_MAX_COUNT`.
    const MAX_IMAGES_PER_PROMPT: usize = 128;

    fn image_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }
}

impl ModelProcessorSpec for DeepseekV41VisionSpec {
    fn name(&self) -> &'static str {
        "deepseek_v41"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata
            .config_model_type()
            .is_some_and(|model_type| model_type == "deepseek_v41")
            || metadata
                .model_id
                .to_ascii_lowercase()
                .contains("deepseek-v4.1")
            || metadata
                .model_id
                .to_ascii_lowercase()
                .contains("deepseek_v41")
    }

    /// The V4.1 renderer inlines the placeholder at each image part's
    /// position (vLLM `_normalize_messages`, SGLang `process_image_messages`),
    /// so part order is protocol-visible.
    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::Authored
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(DEEPSEEK_V41_IMAGE_PLACEHOLDER.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_token_id(metadata)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(
            Modality::Image,
            Self::MAX_IMAGES_PER_PROMPT,
        )]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let image_token_id = Self::image_token_id(metadata)?;
        let placeholder = self.placeholder_token(metadata)?;
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| {
                // Every span position carries image_token_id; the roles live
                // in `types`. All of them are embed positions (delimiters get
                // the learned vectors from embed_multimodal, not the embed
                // table).
                PromptReplacement::repeated(Modality::Image, &placeholder, image_token_id, count)
            })
            .collect())
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("vit_grid".to_string(), FieldLayout::Batched),
            ("llm_grid".to_string(), FieldLayout::Batched),
            ("types".to_string(), FieldLayout::flat("types_per_image")),
            ("patches_per_image".to_string(), FieldLayout::Batched),
            ("types_per_image".to_string(), FieldLayout::Batched),
        ])
    }

    /// The model's forward pops `patches` (not the HF-conventional
    /// `pixel_values`); see vLLM's `DeepseekV4VLImagePixelInputs`.
    fn encoder_input_key_for(&self, modality: Modality) -> Option<String> {
        match modality {
            Modality::Image => Some("patches".to_string()),
            _ => None,
        }
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec![
            "vit_grid".to_string(),
            "llm_grid".to_string(),
            "types".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::registry::test_helpers::TestTokenizer;

    const IMAGE_TOKEN_ID: u32 = 129264;

    fn metadata<'a>(tokenizer: &'a TestTokenizer, config: &'a Value) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "/models/local-checkpoint",
            tokenizer,
            config,
        }
    }

    fn spec_inputs(span_lens: &[usize]) -> PreprocessedEncoderInputs {
        PreprocessedEncoderInputs::new(
            Array2::<f32>::zeros((4, 588)),
            span_lens.to_vec(),
            vec![(100, 100); span_lens.len()],
        )
    }

    #[test]
    fn matches_deepseek_v41_model_type_and_ids() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({"model_type": "deepseek_v41", "image_token_id": IMAGE_TOKEN_ID});
        assert!(DeepseekV41VisionSpec.matches(&metadata(&tokenizer, &config)));
        assert_eq!(
            crate::VisionProcessorRegistry::with_defaults()
                .find("/models/local-checkpoint", Some("deepseek_v41"))
                .map(|processor| processor.model_name()),
            Some("deepseek_v41")
        );
        for model_id in ["deepseek-ai/DeepSeek-V4.1-Flash", "DEEPSEEK_V41-local"] {
            let neutral = json!({});
            let by_id = ModelMetadata {
                model_id,
                tokenizer: &tokenizer,
                config: &neutral,
            };
            assert!(DeepseekV41VisionSpec.matches(&by_id), "{model_id}");
        }

        // A text-only DeepSeek model with a neutral model id must not match.
        let other = json!({"model_type": "deepseek_v3"});
        let other_metadata = ModelMetadata {
            model_id: "deepseek-ai/DeepSeek-V3",
            tokenizer: &tokenizer,
            config: &other,
        };
        assert!(!DeepseekV41VisionSpec.matches(&other_metadata));
        let v4 = json!({"model_type": "deepseek_v4"});
        let v4_metadata = ModelMetadata {
            model_id: "deepseek-ai/DeepSeek-V4-Flash",
            tokenizer: &tokenizer,
            config: &v4,
        };
        assert!(!DeepseekV41VisionSpec.matches(&v4_metadata));
    }

    #[test]
    fn prompt_replacements_expand_spans_without_a_pad() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({"model_type": "deepseek_v41", "image_token_id": IMAGE_TOKEN_ID});
        let metadata = metadata(&tokenizer, &config);

        let replacements = DeepseekV41VisionSpec
            .prompt_replacements(&metadata, &spec_inputs(&[5, 8]))
            .unwrap();

        assert_eq!(replacements.len(), 2);
        for (replacement, &count) in replacements.iter().zip([5usize, 8].iter()) {
            assert_eq!(replacement.modality, Modality::Image);
            assert_eq!(replacement.tokens, vec![IMAGE_TOKEN_ID as TokenId; count]);
        }
    }

    #[test]
    fn placeholder_and_contract_hooks() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({"model_type": "deepseek_v41", "image_token_id": IMAGE_TOKEN_ID});
        let metadata = metadata(&tokenizer, &config);
        let spec = DeepseekV41VisionSpec;
        assert_eq!(
            spec.placeholder_token(&metadata).unwrap(),
            "<｜deepseek_image｜>"
        );
        assert_eq!(
            spec.placeholder_token_id(&metadata).unwrap(),
            IMAGE_TOKEN_ID as TokenId
        );
        assert_eq!(spec.media_part_order(), MediaPartOrder::Authored);
        assert_eq!(
            spec.encoder_input_key_for(Modality::Image).as_deref(),
            Some("patches")
        );
        assert_eq!(spec.encoder_input_key_for(Modality::Audio), None);
        assert_eq!(
            spec.keep_on_cpu_keys(),
            vec!["vit_grid", "llm_grid", "types"]
        );
        assert_eq!(
            spec.modality_limits(&metadata).unwrap(),
            HashMap::from([(Modality::Image, 128)])
        );
        let layouts = spec.field_layouts();
        assert_eq!(
            layouts["pixel_values"],
            FieldLayout::flat("patches_per_image")
        );
        assert_eq!(layouts["types"], FieldLayout::flat("types_per_image"));
        assert_eq!(layouts["vit_grid"], FieldLayout::Batched);

        let missing = json!({"model_type": "deepseek_v41"});
        let missing_metadata = ModelMetadata {
            model_id: "/models/local-checkpoint",
            tokenizer: &tokenizer,
            config: &missing,
        };
        assert!(matches!(
            spec.placeholder_token_id(&missing_metadata),
            Err(ModelRegistryError::MissingConfigField { .. })
        ));
    }
}
