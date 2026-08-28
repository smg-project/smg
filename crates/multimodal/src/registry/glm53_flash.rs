use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    registry::{
        MediaPartOrder, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{FieldLayout, Modality, PlaceholderRange, PromptReplacement, TokenId},
};

const IMAGE: &str = "<|image|>";
const VIDEO: &str = "<|video|>";
const BEGIN: &str = "<|begin_of_image|>";
const END: &str = "<|end_of_image|>";

pub(super) struct Glm53FlashSpec;

impl Glm53FlashSpec {
    fn image_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }

    fn encode(metadata: &ModelMetadata, text: &str) -> RegistryResult<Vec<TokenId>> {
        let ids = metadata.tokenizer.encode_text(text).ok_or_else(|| {
            ModelRegistryError::TextEncodingFailed {
                spec: "glm53_flash",
                text: text.to_string(),
            }
        })?;
        Ok(ids.into_iter().map(|id| id as TokenId).collect())
    }

    fn video_grid_t(input: &PreprocessedEncoderInputs) -> RegistryResult<Vec<usize>> {
        let Some(ModelSpecificValue::IntTensor { data, shape }) =
            input.model_specific.get("video_grid_thw")
        else {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "video_grid_thw".to_string(),
            });
        };
        if shape.len() != 2 || shape[1] != 3 || data.len() != shape[0] * 3 {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "video_grid_thw".to_string(),
            });
        }
        data.as_chunks::<3>()
            .0
            .iter()
            .map(|row| {
                usize::try_from(row[0]).map_err(|_| ModelRegistryError::InvalidPreprocessedField {
                    field: "video_grid_thw".to_string(),
                })
            })
            .collect()
    }

    fn unsupported(modality: Modality) -> ModelRegistryError {
        ModelRegistryError::UnsupportedModality {
            spec: "glm53_flash",
            modality,
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
        ) || ["glm-5.3-flash", "glm5.3-flash", "glm-5-next", "glm5-next"]
            .iter()
            .any(|name| id.contains(name))
    }

    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::Authored
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_id(metadata)
    }

    fn placeholder_token_for(
        &self,
        _metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => Ok(IMAGE.to_string()),
            Modality::Video => Ok(VIDEO.to_string()),
            _ => Err(Self::unsupported(modality)),
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image | Modality::Video => Self::image_id(metadata),
            _ => Err(Self::unsupported(modality)),
        }
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        // Advertise only what the checkpoint can serve, so an incapable
        // derivative is rejected at validate_media_request instead of after
        // a full media fetch + preprocess. Images need the placeholder id;
        // video additionally splices <|begin_of_image|>/<|end_of_image|>
        // frame markers into the prompt.
        let mut limits = HashMap::new();
        if Self::image_id(metadata).is_ok() {
            limits.insert(Modality::Image, 10);
            if metadata.token_id(BEGIN).is_ok() && metadata.token_id(END).is_ok() {
                limits.insert(Modality::Video, 1);
            }
        }
        Ok(limits)
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        input: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let id = Self::image_id(metadata)?;
        Ok(input
            .feature_token_counts
            .iter()
            .map(|&count| PromptReplacement::repeated(Modality::Image, IMAGE, id, count))
            .collect())
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        input: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        if modality == Modality::Image {
            return self.prompt_replacements(metadata, input);
        }
        if modality != Modality::Video {
            return Err(Self::unsupported(modality));
        }

        let grids = Self::video_grid_t(input)?;
        if grids.len() != input.feature_token_counts.len() {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "video_grid_thw item count".to_string(),
            });
        }
        let image_id = Self::image_id(metadata)?;
        let begin = metadata.token_id(BEGIN)?;
        let end = metadata.token_id(END)?;
        // The paired vision processor always emits this; a missing or
        // wrongly-typed value means the pipeline is broken, and defaulting
        // would silently caption every frame with wrong timestamps while
        // the grid field next to it fails loudly.
        let seconds = match input.model_specific.get("video_second_per_grid") {
            Some(ModelSpecificValue::Tensor { data, .. }) if !data.is_empty() => data[0],
            _ => {
                return Err(ModelRegistryError::InvalidPreprocessedField {
                    field: "video_second_per_grid".to_string(),
                })
            }
        };

        input
            .feature_token_counts
            .iter()
            .zip(grids)
            .map(|(&count, grid_t)| {
                if grid_t == 0 || !count.is_multiple_of(grid_t) {
                    return Err(ModelRegistryError::InvalidPreprocessedField {
                        field: "video_grid_thw temporal token layout".to_string(),
                    });
                }
                let per_grid = count / grid_t;
                let mut tokens = Vec::new();
                let mut ranges = Vec::with_capacity(grid_t);
                for index in 0..grid_t {
                    tokens.push(begin);
                    ranges.push(PlaceholderRange {
                        offset: tokens.len(),
                        length: per_grid,
                    });
                    tokens.extend(std::iter::repeat_n(image_id, per_grid));
                    tokens.push(end);
                    tokens.extend(Self::encode(
                        metadata,
                        &format!("{:.1} seconds", index as f32 * seconds),
                    )?);
                }
                Ok(PromptReplacement::sequence(Modality::Video, VIDEO, tokens)
                    .with_feature_ranges(ranges))
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
    const BEGIN_ID: u32 = 154830;
    const END_ID: u32 = 154831;
    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[(IMAGE, IMAGE_ID), (BEGIN, BEGIN_ID), (END, END_ID)])
            .with_byte_encoder(1000)
    }

    #[test]
    fn matches_names_and_builds_video_timestamps() {
        let tokenizer = tokenizer();
        let config = json!({"model_type":"glm53_flash", "image_token_id":IMAGE_ID});
        let metadata = ModelMetadata {
            model_id: "zai-org/GLM-5.3-Flash",
            tokenizer: &tokenizer,
            config: &config,
        };
        let mut input = test_preprocessed_with_tokens(&[], &[4]);
        input.model_specific.insert(
            "video_grid_thw".into(),
            ModelSpecificValue::int_2d(vec![2, 2, 4], 1, 3),
        );
        input.model_specific.insert(
            "video_second_per_grid".into(),
            ModelSpecificValue::Tensor {
                data: vec![1.0],
                shape: vec![1],
            },
        );
        let replacement = ModelRegistry::new()
            .lookup(&metadata)
            .unwrap()
            .prompt_replacements_for(&metadata, &input, Modality::Video)
            .unwrap()
            .pop()
            .unwrap();

        let ranges = replacement.feature_ranges.unwrap();
        assert_eq!(
            ranges
                .iter()
                .map(|range| (range.offset, range.length))
                .collect::<Vec<_>>(),
            vec![(1, 2), (16, 2)]
        );
        assert_eq!(replacement.tokens[0], BEGIN_ID as TokenId);
        assert_eq!(replacement.tokens[1], IMAGE_ID as TokenId);
        assert_eq!(replacement.tokens[3], END_ID as TokenId);
        let timestamp = Glm53FlashSpec::encode(&metadata, "0.0 seconds").unwrap();
        assert_eq!(&replacement.tokens[4..15], timestamp.as_slice());

        let legacy = json!({"model_type":"glm5_next"});
        assert!(Glm53FlashSpec.matches(&ModelMetadata {
            model_id: "legacy",
            tokenizer: &tokenizer,
            config: &legacy,
        }));
    }

    #[test]
    fn modality_adverts_follow_checkpoint_capability() {
        // Full capability: image placeholder id + frame-marker tokens.
        let full = tokenizer();
        let config = json!({"model_type":"glm53_flash", "image_token_id":IMAGE_ID});
        let limits = Glm53FlashSpec
            .modality_limits(&ModelMetadata {
                model_id: "capable",
                tokenizer: &full,
                config: &config,
            })
            .unwrap();
        assert_eq!(limits.get(&Modality::Image), Some(&10));
        assert_eq!(limits.get(&Modality::Video), Some(&1));

        // No frame markers in the vocab: image-only, video rejected up
        // front instead of after a full clip fetch + preprocess.
        let no_markers = TestTokenizer::new(&[(IMAGE, IMAGE_ID)]).with_byte_encoder(1000);
        let limits = Glm53FlashSpec
            .modality_limits(&ModelMetadata {
                model_id: "image-only",
                tokenizer: &no_markers,
                config: &config,
            })
            .unwrap();
        assert_eq!(limits.get(&Modality::Image), Some(&10));
        assert!(!limits.contains_key(&Modality::Video));

        // No image token id in the config: nothing advertised.
        let no_id = json!({"model_type":"glm53_flash"});
        let limits = Glm53FlashSpec
            .modality_limits(&ModelMetadata {
                model_id: "text-only",
                tokenizer: &full,
                config: &no_id,
            })
            .unwrap();
        assert!(limits.is_empty());
    }

    #[test]
    fn video_error_branches_fail_loudly() {
        let tokenizer = tokenizer();
        let config = json!({"model_type":"glm53_flash", "image_token_id":IMAGE_ID});
        let metadata = ModelMetadata {
            model_id: "zai-org/GLM-5.3-Flash",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = Glm53FlashSpec;

        // Missing video_second_per_grid: the paired processor always emits
        // it, so absence is a broken pipeline, not a 1.0 default.
        let mut input = test_preprocessed_with_tokens(&[], &[4]);
        input.model_specific.insert(
            "video_grid_thw".into(),
            ModelSpecificValue::int_2d(vec![2, 2, 4], 1, 3),
        );
        let err = spec
            .prompt_replacements_for(&metadata, &input, Modality::Video)
            .unwrap_err();
        assert!(matches!(
            err,
            ModelRegistryError::InvalidPreprocessedField { ref field }
                if field == "video_second_per_grid"
        ));

        // Wrong grid shape (row width != 3).
        let mut input = test_preprocessed_with_tokens(&[], &[4]);
        input.model_specific.insert(
            "video_grid_thw".into(),
            ModelSpecificValue::int_2d(vec![2, 2], 1, 2),
        );
        assert!(matches!(
            spec.prompt_replacements_for(&metadata, &input, Modality::Video),
            Err(ModelRegistryError::InvalidPreprocessedField { .. })
        ));

        // Token count not divisible by grid_t.
        let mut input = test_preprocessed_with_tokens(&[], &[5]);
        input.model_specific.insert(
            "video_grid_thw".into(),
            ModelSpecificValue::int_2d(vec![2, 2, 4], 1, 3),
        );
        input.model_specific.insert(
            "video_second_per_grid".into(),
            ModelSpecificValue::Tensor {
                data: vec![1.0],
                shape: vec![1],
            },
        );
        assert!(matches!(
            spec.prompt_replacements_for(&metadata, &input, Modality::Video),
            Err(ModelRegistryError::InvalidPreprocessedField { .. })
        ));
    }
}
