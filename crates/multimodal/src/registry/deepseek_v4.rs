//! DeepSeek V4 Flash Vision registry spec: model detection, placeholder
//! handling, and the image-block expansion consumed by the engine.
//!
//! The checkpoint keeps `architectures: ["DeepseekV4ForCausalLM"]` and is a
//! vision model iff `vision_n_layers > 0` in its config. Image placeholders
//! (`<｜deepseek_image｜>`) expand into the official N-layout sentinel block:
//! every position uses the tokenizer-resolved `<｜deepseek_image｜>` id.
//! Sentinel types travel separately; the engine fills the whole block with
//! vision vectors and includes the whole block in image-aware prefix hashing.
//!
//! Each block starts with the maximal `COMPRESS_PAD_TO - 1` leading pads so
//! the replacement layer can trim `block_start % 4` of them at splice time
//! (see `replacement_alignment`); the engine trims `types` by the same rule,
//! keeping the IMAGE data region aligned to a multiple of 4 tokens exactly
//! like the official `build_image_block(start_pos=...)`.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    registry::{
        MediaPartOrder, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::processors::deepseek_v4::COMPRESS_PAD_TO,
};

pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

/// Hard cap on images per request; the official processor has no explicit
/// limit, so this only guards runaway requests.
const MAX_IMAGES: usize = 16;

pub(super) struct DeepseekV4Spec;

impl ModelProcessorSpec for DeepseekV4Spec {
    fn name(&self) -> &'static str {
        "deepseek_v4"
    }

    /// Vision variant of DeepSeek V4 only: same `model_type` as the text
    /// model, distinguished by a non-zero vision tower.
    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata.config_model_type() == Some("deepseek_v4")
            && metadata
                .config_u32(&["vision_n_layers"])
                .is_some_and(|n| n > 0)
    }

    /// DeepSeek's template renders parts in authored order (see
    /// `encoding_dsv4.py`: text and `<｜deepseek_image｜>` interleave).
    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::Authored
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE_PLACEHOLDER.to_string())
    }

    /// The placeholder is a regular vocab token; resolve it through the
    /// tokenizer (the model config declares no `image_token_id` field).
    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id(IMAGE_PLACEHOLDER)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, MAX_IMAGES)]))
    }

    /// Lift the vision geometry from the model config into the preprocessor
    /// config so `DeepseekV4Processor` sees one uniform source.
    fn processor_kwargs(&self, metadata: &ModelMetadata) -> RegistryResult<Value> {
        let keys = [
            "vision_patch_size",
            "vision_downsample_ratio",
            "vision_max_n_token",
            "vision_min_pixels",
            "vision_max_wh_ratio",
        ];
        let mut extra = serde_json::Map::new();
        for key in keys {
            if let Some(v) = metadata.config.get(key) {
                extra.insert(key.to_string(), v.clone());
            }
        }
        Ok(json!({ "extra": extra }))
    }

    /// One replacement per image: the full block of image placeholder ids.
    /// The head carries `COMPRESS_PAD_TO - 1` pads for position trimming.
    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let image_token_id = self.placeholder_token_id(metadata)?;
        let lengths = match preprocessed.model_specific.get("types_lengths") {
            Some(ModelSpecificValue::IntVec(lengths)) => lengths,
            _ => {
                return Err(ModelRegistryError::InvalidPreprocessedField {
                    field: "types_lengths".to_string(),
                })
            }
        };
        let mut remaining = match preprocessed.model_specific.get("types") {
            Some(ModelSpecificValue::IntTensor { data, .. }) => data.len(),
            _ => {
                return Err(ModelRegistryError::InvalidPreprocessedField {
                    field: "types".to_string(),
                })
            }
        };
        lengths
            .iter()
            .map(|&len| {
                let len = usize::try_from(len).map_err(|_| {
                    ModelRegistryError::InvalidPreprocessedField {
                        field: "types_lengths".to_string(),
                    }
                })?;
                remaining = remaining.checked_sub(len).ok_or_else(|| {
                    ModelRegistryError::InvalidPreprocessedField {
                        field: "types_lengths".to_string(),
                    }
                })?;
                Ok(PromptReplacement::sequence(
                    Modality::Image,
                    IMAGE_PLACEHOLDER,
                    vec![image_token_id; len],
                ))
            })
            .collect()
    }

    /// IMAGE data must start at a token index divisible by 4 (the aligner's
    /// compression alignment). The replacement layer trims
    /// `block_start % 4` leading pads from each expansion.
    fn replacement_alignment(&self) -> Option<u32> {
        Some(COMPRESS_PAD_TO as u32)
    }

    /// Patches, types and permutations are concatenated across images;
    /// geometry and lengths are one scalar per image. Types retain all three
    /// leading pads on the wire; the receiver trims by the prompt offset.
    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::Flat {
                    sizes_key: "patch_counts".to_string(),
                },
            ),
            (
                "types".to_string(),
                FieldLayout::Flat {
                    sizes_key: "types_lengths".to_string(),
                },
            ),
            (
                "perm".to_string(),
                FieldLayout::Flat {
                    sizes_key: "perm_lengths".to_string(),
                },
            ),
            ("n_vit_h".to_string(), FieldLayout::Batched),
            ("n_vit_w".to_string(), FieldLayout::Batched),
            ("types_lengths".to_string(), FieldLayout::Batched),
            ("patch_counts".to_string(), FieldLayout::Batched),
            ("perm_lengths".to_string(), FieldLayout::Batched),
        ])
    }

    /// The sentinel metadata is consumed on the engine host when building the
    /// image block; no need to stage it on GPU.
    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec![
            "types".to_string(),
            "perm".to_string(),
            "n_vit_h".to_string(),
            "n_vit_w".to_string(),
            "types_lengths".to_string(),
            "patch_counts".to_string(),
            "perm_lengths".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        registry::test_helpers::TestTokenizer,
        vision::{
            processors::deepseek_v4::{
                DeepseekV4Processor, IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_PAD, IMAGE_START,
            },
            PreProcessorConfig, VisionPreProcessor,
        },
    };

    #[test]
    fn replacements_repeat_tokenizer_image_id_for_the_entire_block() {
        let images = [
            image::DynamicImage::new_rgb8(84, 84),
            image::DynamicImage::new_rgb8(126, 42),
        ];
        let processor_config: PreProcessorConfig = serde_json::from_value(json!({
            "vision_min_pixels": 0
        }))
        .unwrap();
        let preprocessed = DeepseekV4Processor
            .preprocess(&images, &processor_config)
            .unwrap();
        // No vocab_size is needed, and a custom tokenizer must not be ignored.
        for token_id in [129264, 4242] {
            let tokenizer = TestTokenizer::new(&[(IMAGE_PLACEHOLDER, token_id)]);
            let config = json!({"model_type": "deepseek_v4", "vision_n_layers": 1});
            let metadata = ModelMetadata {
                model_id: "deepseek-v4-vision",
                tokenizer: &tokenizer,
                config: &config,
            };
            let replacements = DeepseekV4Spec
                .prompt_replacements(&metadata, &preprocessed)
                .unwrap();
            assert_eq!(replacements.len(), 2);
            for replacement in replacements {
                assert_eq!(replacement.tokens, vec![token_id as TokenId; 13]);
                assert!(replacement.feature_ranges.is_none());
            }
            for lengths in [vec![-1], vec![i64::MAX], vec![13, 14]] {
                let mut invalid = preprocessed.clone();
                invalid.model_specific.insert(
                    "types_lengths".to_string(),
                    ModelSpecificValue::IntVec(lengths),
                );
                assert!(matches!(
                    DeepseekV4Spec.prompt_replacements(&metadata, &invalid),
                    Err(ModelRegistryError::InvalidPreprocessedField { .. })
                ));
            }
        }
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({"vocab_size": 129280});
        let metadata = ModelMetadata {
            model_id: "deepseek-v4-vision",
            tokenizer: &tokenizer,
            config: &config,
        };
        assert!(matches!(
            DeepseekV4Spec.prompt_replacements(&metadata, &preprocessed),
            Err(ModelRegistryError::TokenNotFound { .. })
        ));
    }
    #[test]
    fn sentinel_constants_match_official() {
        assert_eq!(IMAGE_START, 0);
        assert_eq!(IMAGE_PAD, 1);
        assert_eq!(IMAGE, 2);
        assert_eq!(IMAGE_NEW_LINE, 3);
        assert_eq!(IMAGE_END, 4);
    }

    #[test]
    fn placeholder_uses_fullwidth_bars() {
        assert_eq!(IMAGE_PLACEHOLDER, "<｜deepseek_image｜>");
    }
}
