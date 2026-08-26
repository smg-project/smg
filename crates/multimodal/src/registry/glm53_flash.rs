use std::collections::HashMap;

use llm_tokenizer::Encoding;
use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    registry::{
        MediaPartOrder, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{FieldLayout, Modality, PlaceholderRange, PromptReplacement, TokenId},
};

const IMAGE_TOKEN: &str = "<|image|>";
const VIDEO_TOKEN: &str = "<|video|>";
const IMAGE_BEGIN: &str = "<|begin_of_image|>";
const IMAGE_END: &str = "<|end_of_image|>";

pub(super) struct Glm53FlashSpec;

impl Glm53FlashSpec {
    fn image_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }

    fn encode_text(metadata: &ModelMetadata, text: &str) -> RegistryResult<Vec<TokenId>> {
        let encoding = metadata.tokenizer.encode(text, false).map_err(|_| {
            ModelRegistryError::TextEncodingFailed {
                spec: "glm53_flash",
                text: text.to_string(),
            }
        })?;
        Ok(match encoding {
            Encoding::Hf(inner) => inner.get_ids().iter().map(|&id| id as TokenId).collect(),
            Encoding::Plain(ids) | Encoding::Tiktoken(ids) => {
                ids.into_iter().map(|id| id as TokenId).collect()
            }
        })
    }

    fn video_grid(preprocessed: &PreprocessedEncoderInputs) -> Option<Vec<[usize; 3]>> {
        let ModelSpecificValue::IntTensor { data, shape } =
            preprocessed.model_specific.get("video_grid_thw")?
        else {
            return None;
        };
        if shape.len() != 2 || shape[1] != 3 || data.len() != shape[0] * 3 {
            return None;
        }
        data.chunks_exact(3)
            .map(|row| {
                Some([
                    usize::try_from(row[0]).ok()?,
                    usize::try_from(row[1]).ok()?,
                    usize::try_from(row[2]).ok()?,
                ])
            })
            .collect()
    }

    fn seconds_per_grid(preprocessed: &PreprocessedEncoderInputs) -> f32 {
        match preprocessed.model_specific.get("video_second_per_grid") {
            Some(ModelSpecificValue::Tensor { data, .. }) => data.first().copied().unwrap_or(1.0),
            _ => 1.0,
        }
    }
}

impl ModelProcessorSpec for Glm53FlashSpec {
    fn name(&self) -> &'static str {
        "glm53_flash"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        matches!(
            metadata.config_model_type(),
            Some("glm53_flash" | "glm5_next")
        ) || id.contains("glm-5.3-flash")
            || id.contains("glm5.3-flash")
            || id.contains("glm-5-next")
            || id.contains("glm5-next")
    }

    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::Authored
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE_TOKEN.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_token_id(metadata)
    }

    fn placeholder_token_for(
        &self,
        _metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => Ok(IMAGE_TOKEN.to_string()),
            Modality::Video => Ok(VIDEO_TOKEN.to_string()),
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
            // GLM-5.3-Flash represents video patches with image_token_id inside
            // the surrounding video boundary tokens.
            Modality::Image | Modality::Video => Self::image_token_id(metadata),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 10), (Modality::Video, 1)]))
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
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&tokens| {
                PromptReplacement::repeated(Modality::Image, IMAGE_TOKEN, image_token_id, tokens)
            })
            .collect())
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        if modality == Modality::Image {
            return self.prompt_replacements(metadata, preprocessed);
        }
        if modality != Modality::Video {
            return Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            });
        }

        let grids = Self::video_grid(preprocessed).ok_or_else(|| {
            ModelRegistryError::InvalidPreprocessedField {
                field: "video_grid_thw".to_string(),
            }
        })?;
        if grids.len() != preprocessed.feature_token_counts.len() {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "video_grid_thw item count".to_string(),
            });
        }

        let image_token_id = Self::image_token_id(metadata)?;
        let image_begin = metadata.token_id(IMAGE_BEGIN)?;
        let image_end = metadata.token_id(IMAGE_END)?;
        let seconds_per_grid = Self::seconds_per_grid(preprocessed);
        preprocessed
            .feature_token_counts
            .iter()
            .zip(grids)
            .map(|(&num_tokens, [grid_t, _, _])| {
                if grid_t == 0 || !num_tokens.is_multiple_of(grid_t) {
                    return Err(ModelRegistryError::InvalidPreprocessedField {
                        field: "video_grid_thw temporal token layout".to_string(),
                    });
                }
                let tokens_per_grid = num_tokens / grid_t;
                let mut tokens = Vec::new();
                let mut feature_ranges = Vec::with_capacity(grid_t);
                for grid_index in 0..grid_t {
                    tokens.push(image_begin);
                    let feature_offset = tokens.len();
                    tokens.extend(std::iter::repeat_n(image_token_id, tokens_per_grid));
                    feature_ranges.push(PlaceholderRange {
                        offset: feature_offset,
                        length: tokens_per_grid,
                    });
                    tokens.push(image_end);
                    tokens.extend(Self::encode_text(
                        metadata,
                        &format!("{:.1} seconds", grid_index as f32 * seconds_per_grid),
                    )?);
                }
                Ok(
                    PromptReplacement::sequence(Modality::Video, VIDEO_TOKEN, tokens)
                        .with_feature_ranges(feature_ranges),
                )
            })
            .collect()
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("video_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
            ("patches_per_video".to_string(), FieldLayout::Batched),
            ("video_second_per_grid".to_string(), FieldLayout::Batched),
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string(), "video_grid_thw".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::registry::{test_helpers::*, ModelRegistry};

    const IMAGE_ID: u32 = 154854;
    const VIDEO_ID: u32 = 154855;
    const IMAGE_BEGIN_ID: u32 = 154830;
    const IMAGE_END_ID: u32 = 154831;
    const TEXT_BASE: u32 = 1000;

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            (IMAGE_TOKEN, IMAGE_ID),
            (VIDEO_TOKEN, VIDEO_ID),
            (IMAGE_BEGIN, IMAGE_BEGIN_ID),
            (IMAGE_END, IMAGE_END_ID),
        ])
        .with_byte_encoder(TEXT_BASE)
    }

    fn config() -> Value {
        json!({
            "model_type": "glm53_flash",
            "image_token_id": IMAGE_ID,
            "video_token_id": VIDEO_ID,
        })
    }

    #[test]
    fn matches_public_and_legacy_names() {
        let tokenizer = tokenizer();
        let registry = ModelRegistry::new();
        for (model_id, model_type) in [
            ("zai-org/GLM-5.3-Flash", "glm53_flash"),
            ("zai-org/GLM-5-Next", "glm5_next"),
        ] {
            let config = json!({
                "model_type": model_type,
                "image_token_id": IMAGE_ID,
                "video_token_id": VIDEO_ID,
            });
            let metadata = ModelMetadata {
                model_id,
                tokenizer: &tokenizer,
                config: &config,
            };
            let spec = registry.lookup(&metadata).expect("GLM-5.3-Flash spec");
            assert_eq!(spec.name(), "glm53_flash");
            assert_eq!(spec.media_part_order(), MediaPartOrder::Authored);
        }
    }

    #[test]
    fn video_uses_image_pads_with_per_frame_timestamps() {
        let tokenizer = tokenizer();
        let config = config();
        let metadata = ModelMetadata {
            model_id: "zai-org/GLM-5.3-Flash",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).unwrap();
        let mut preprocessed = test_preprocessed_with_tokens(&[], &[4]);
        preprocessed.model_specific.insert(
            "video_grid_thw".to_string(),
            ModelSpecificValue::int_2d(vec![2, 2, 4], 1, 3),
        );
        preprocessed.model_specific.insert(
            "video_second_per_grid".to_string(),
            ModelSpecificValue::Tensor {
                data: vec![1.0],
                shape: vec![1],
            },
        );

        let replacement = spec
            .prompt_replacements_for(&metadata, &preprocessed, Modality::Video)
            .unwrap()
            .pop()
            .unwrap();
        let text = |value: &str| {
            value
                .bytes()
                .map(|byte| (TEXT_BASE + u32::from(byte)) as TokenId)
                .collect::<Vec<_>>()
        };
        let mut expected = vec![
            IMAGE_BEGIN_ID as TokenId,
            IMAGE_ID as TokenId,
            IMAGE_ID as TokenId,
            IMAGE_END_ID as TokenId,
        ];
        expected.extend(text("0.0 seconds"));
        let second_offset = expected.len() + 1;
        expected.extend([
            IMAGE_BEGIN_ID as TokenId,
            IMAGE_ID as TokenId,
            IMAGE_ID as TokenId,
            IMAGE_END_ID as TokenId,
        ]);
        expected.extend(text("1.0 seconds"));

        assert_eq!(replacement.tokens, expected);
        assert_eq!(
            replacement.feature_ranges,
            Some(vec![
                PlaceholderRange {
                    offset: 1,
                    length: 2,
                },
                PlaceholderRange {
                    offset: second_offset,
                    length: 2,
                },
            ])
        );
        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Video)
                .unwrap(),
            VIDEO_TOKEN
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, Modality::Video)
                .unwrap(),
            IMAGE_ID as TokenId
        );
    }
}
