//! The Step3-VL family: a global view and sliding-window crops per image,
//! written crops first, then the view, as the serving engine's own prompt
//! update writes them.

use std::collections::HashMap;

use serde_json::Value;

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    registry::{
        ModalityHooks, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId},
    vision::{
        processors::step3_vl::{
            Step3VLProcessor, IMAGE_FEATURE_TOKENS, NUM_PATCHES, PATCH_FEATURE_TOKENS,
            PATCH_NEWLINE_MASK, PATCH_PIXEL_VALUES,
        },
        PreProcessorConfig, VisionPreProcessor,
    },
};

/// The image token: the chat template writes one per image, and it fills
/// every feature position of the expansion (`image_token_id` in the config).
const IMAGE_TOKEN: &str = "<im_patch>";
/// The structural tokens around the view and around each crop.
const IMAGE_START: &str = "<im_start>";
const IMAGE_END: &str = "<im_end>";
const PATCH_START: &str = "<patch_start>";
const PATCH_END: &str = "<patch_end>";
/// The token after every crop row but the last.
const PATCH_NEWLINE: &str = "<patch_newline>";
/// Images per request by default. The serving engine advertises no count
/// for the family; eight (the count of the other global-view-and-tiles
/// family) bounds a request's crops tensor at eight images of the per-image
/// limit. A deployment's `--mm-per-request-image-limit` / `SMG_IMAGE_MAX_COUNT`
/// replaces this number outright (the operator's call: 32 images are up to
/// 3.4 GiB of float32 crops), and the engine's advertised limit caps the
/// result.
const MAX_IMAGES_PER_REQUEST: usize = 8;

/// The Step3-VL image spec (`model_type` `step3_vl`, and `step5_vl`, which
/// the serving engine runs through the same processor and layout).
pub(super) struct Step3VLVisionSpec;

impl Step3VLVisionSpec {
    const NAME: &'static str = "step3_vl";

    fn image_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        match metadata.config_u32(&["image_token_id"]) {
            Some(id) => Ok(id as TokenId),
            None => metadata.token_id(IMAGE_TOKEN),
        }
    }

    fn int_field<'a>(
        preprocessed: &'a PreprocessedEncoderInputs,
        field: &str,
    ) -> RegistryResult<&'a [i64]> {
        match preprocessed.model_specific.get(field) {
            Some(ModelSpecificValue::IntTensor { data, .. }) => Ok(data),
            _ => Err(ModelRegistryError::InvalidPreprocessedField {
                field: field.to_string(),
            }),
        }
    }

    /// One replacement per image: each crop as `<patch_start>`, its feature
    /// tokens and `<patch_end>`, a `<patch_newline>` where the mask says the
    /// crop ends a row that is not the last, then the view as `<im_start>`,
    /// its feature tokens and `<im_end>`.
    fn image_replacements(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let image_token = Self::image_token_id(metadata)?;
        let image_start = metadata.token_id(IMAGE_START)?;
        let image_end = metadata.token_id(IMAGE_END)?;
        let patch_start = metadata.token_id(PATCH_START)?;
        let patch_end = metadata.token_id(PATCH_END)?;
        let patch_newline = metadata.token_id(PATCH_NEWLINE)?;
        let counts = Self::int_field(preprocessed, NUM_PATCHES)?;
        let newlines = Self::int_field(preprocessed, PATCH_NEWLINE_MASK)?;
        let counts: Vec<usize> = counts
            .iter()
            .map(|&count| usize::try_from(count))
            .collect::<Result<_, _>>()
            .map_err(|_| ModelRegistryError::InvalidPreprocessedField {
                field: NUM_PATCHES.to_string(),
            })?;
        if counts.iter().sum::<usize>() != newlines.len() {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: PATCH_NEWLINE_MASK.to_string(),
            });
        }
        let mut first = 0;
        Ok(counts
            .iter()
            .map(|&count| {
                let mask = &newlines[first..first + count];
                first += count;
                let mut tokens = Vec::with_capacity(
                    count * (PATCH_FEATURE_TOKENS + 3) + IMAGE_FEATURE_TOKENS + 2,
                );
                for &newline in mask {
                    tokens.push(patch_start);
                    tokens.extend(std::iter::repeat_n(image_token, PATCH_FEATURE_TOKENS));
                    tokens.push(patch_end);
                    if newline != 0 {
                        tokens.push(patch_newline);
                    }
                }
                tokens.push(image_start);
                tokens.extend(std::iter::repeat_n(image_token, IMAGE_FEATURE_TOKENS));
                tokens.push(image_end);
                PromptReplacement::sequence(Modality::Image, IMAGE_TOKEN, tokens)
            })
            .collect())
    }
}

impl ModelProcessorSpec for Step3VLVisionSpec {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        matches!(metadata.config_model_type(), Some("step3_vl" | "step5_vl"))
    }

    fn hooks(&self, modality: Modality) -> Option<ModalityHooks> {
        match modality {
            Modality::Image => Some(ModalityHooks {
                placeholder: |_| Ok(IMAGE_TOKEN.to_string()),
                placeholder_id: Self::image_token_id,
                replacements: |metadata, preprocessed, _, _| {
                    Self::image_replacements(metadata, preprocessed)
                },
                layouts: || {
                    EncoderFieldLayouts::new(
                        FieldLayout::Batched,
                        HashMap::from([
                            (
                                PATCH_PIXEL_VALUES.to_string(),
                                FieldLayout::flat(NUM_PATCHES),
                            ),
                            (NUM_PATCHES.to_string(), FieldLayout::Batched),
                            (
                                PATCH_NEWLINE_MASK.to_string(),
                                FieldLayout::flat(NUM_PATCHES),
                            ),
                        ]),
                    )
                },
                keep_on_cpu: &[NUM_PATCHES, PATCH_NEWLINE_MASK],
                encoder_input_key: None,
                // The serving engine's prompt update targets the single image
                // token, so a worker expands the anchor itself.
                worker_expandable: true,
            }),
            _ => None,
        }
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE_TOKEN.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_token_id(metadata)
    }

    /// Images per request: [`MAX_IMAGES_PER_REQUEST`] by default; the
    /// deployment's `--mm-per-request-image-limit` / `SMG_IMAGE_MAX_COUNT`
    /// replaces it, and the engine's advertised limit caps the result.
    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, MAX_IMAGES_PER_REQUEST)]))
    }

    /// The processor reads the model config's `vision_config.enable_patch`
    /// and nothing from a preprocessor config: the family ships none and its
    /// serving engine builds the processor from its constants.
    fn vision_processor(
        &self,
        model_config: &Value,
        _preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image => Ok(Box::new(Step3VLProcessor::from_model_config(model_config))),
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
        Self::image_replacements(metadata, preprocessed)
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};
    use serde_json::json;

    use super::*;
    use crate::{
        registry::{test_helpers::TestTokenizer, ModelRegistry},
        types::ImageSize,
    };

    const IMAGE_START_ID: TokenId = 128000;
    const IMAGE_TOKEN_ID: TokenId = 128001;
    const IMAGE_END_ID: TokenId = 128002;
    const PATCH_START_ID: TokenId = 128003;
    const PATCH_NEWLINE_ID: TokenId = 128004;
    const PATCH_END_ID: TokenId = 128005;

    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[
            (IMAGE_START, IMAGE_START_ID as u32),
            (IMAGE_TOKEN, IMAGE_TOKEN_ID as u32),
            (IMAGE_END, IMAGE_END_ID as u32),
            (PATCH_START, PATCH_START_ID as u32),
            (PATCH_NEWLINE, PATCH_NEWLINE_ID as u32),
            (PATCH_END, PATCH_END_ID as u32),
        ])
    }

    fn config(model_type: &str) -> Value {
        json!({
            "model_type": model_type,
            "image_token_id": IMAGE_TOKEN_ID,
            "vision_config": {"image_size": 728, "patch_size": 14},
        })
    }

    fn metadata<'a>(tokenizer: &'a TestTokenizer, config: &'a Value) -> ModelMetadata<'a> {
        ModelMetadata {
            model_id: "/models/checkpoint",
            tokenizer,
            config,
        }
    }

    /// The preprocessed shape of `counts.len()` images with the given crop
    /// counts and newline flags, as the processor emits it.
    fn preprocessed(counts: &[i64], newlines: &[i64]) -> PreprocessedEncoderInputs {
        let mut out = PreprocessedEncoderInputs::new(
            ndarray::Array4::<f32>::zeros((counts.len(), 3, 8, 8)),
            counts
                .iter()
                .map(|&count| count as usize * PATCH_FEATURE_TOKENS + IMAGE_FEATURE_TOKENS)
                .collect(),
            vec![(16, 16); counts.len()],
        );
        out.model_specific.insert(
            NUM_PATCHES.to_string(),
            ModelSpecificValue::IntTensor {
                data: counts.to_vec(),
                shape: vec![counts.len()],
            },
        );
        out.model_specific.insert(
            PATCH_NEWLINE_MASK.to_string(),
            ModelSpecificValue::IntTensor {
                data: newlines.to_vec(),
                shape: vec![newlines.len()],
            },
        );
        out
    }

    /// Both model types of the family resolve to the spec by `model_type`
    /// alone; the id is an opaque path.
    #[test]
    fn the_registry_resolves_both_model_types() {
        let tokenizer = tokenizer();
        let registry = ModelRegistry::new();
        for model_type in ["step3_vl", "step5_vl"] {
            let config = config(model_type);
            let spec = registry
                .lookup(&metadata(&tokenizer, &config))
                .unwrap_or_else(|| panic!("{model_type} has a spec"));
            assert_eq!(spec.name(), "step3_vl", "{model_type}");
            assert_eq!(
                spec.vision_processor(&config, &PreProcessorConfig::default(), Modality::Image)
                    .map(|processor| processor.model_name()),
                Ok("step3_vl")
            );
        }
        let other = json!({"model_type": "step3_text"});
        assert!(!Step3VLVisionSpec.matches(&metadata(&tokenizer, &other)));
    }

    /// An image without crops expands to the view alone: `<im_start>`, 169
    /// image tokens, `<im_end>`.
    #[test]
    fn an_image_without_crops_expands_to_the_view() {
        let tokenizer = tokenizer();
        let config = config("step3_vl");
        let replacements = Step3VLVisionSpec
            .prompt_replacements(&metadata(&tokenizer, &config), &preprocessed(&[0], &[]))
            .unwrap();
        assert_eq!(replacements.len(), 1);
        let tokens = &replacements[0].tokens;
        assert_eq!(tokens.len(), 171);
        assert_eq!(tokens[0], IMAGE_START_ID);
        assert!(tokens[1..170].iter().all(|&token| token == IMAGE_TOKEN_ID));
        assert_eq!(tokens[170], IMAGE_END_ID);
        assert_eq!(replacements[0].placeholder_token, IMAGE_TOKEN);
    }

    /// A 1920x1080 frame's eight crops in two rows: each crop between its
    /// start and end tokens, one newline after the fourth, then the view;
    /// 836 tokens, the engine's count for the size.
    #[test]
    fn crops_come_first_with_a_newline_after_every_row_but_the_last() {
        let tokenizer = tokenizer();
        let config = config("step3_vl");
        let replacements = Step3VLVisionSpec
            .prompt_replacements(
                &metadata(&tokenizer, &config),
                &preprocessed(&[8], &[0, 0, 0, 1, 0, 0, 0, 0]),
            )
            .unwrap();
        let tokens = &replacements[0].tokens;
        assert_eq!(tokens.len(), 836);
        let crop = PATCH_FEATURE_TOKENS + 2;
        for index in 0..8 {
            // The newline after the fourth crop shifts the second row by one.
            let offset = index * crop + usize::from(index >= 4);
            assert_eq!(tokens[offset], PATCH_START_ID, "crop {index} start");
            assert!(
                tokens[offset + 1..offset + 1 + PATCH_FEATURE_TOKENS]
                    .iter()
                    .all(|&token| token == IMAGE_TOKEN_ID),
                "crop {index} features"
            );
            assert_eq!(tokens[offset + crop - 1], PATCH_END_ID, "crop {index} end");
        }
        assert_eq!(tokens[4 * crop], PATCH_NEWLINE_ID);
        let view = 8 * crop + 1;
        assert_eq!(tokens[view], IMAGE_START_ID);
        assert_eq!(tokens[view + 1 + IMAGE_FEATURE_TOKENS], IMAGE_END_ID);
        assert_eq!(
            tokens
                .iter()
                .filter(|&&token| token == IMAGE_TOKEN_ID)
                .count(),
            8 * PATCH_FEATURE_TOKENS + IMAGE_FEATURE_TOKENS
        );
    }

    /// A batch's replacements read each image's own crops from the flat
    /// mask, in batch order.
    #[test]
    fn a_batch_splits_the_flat_mask_per_image() {
        let tokenizer = tokenizer();
        let config = config("step5_vl");
        let replacements = Step3VLVisionSpec
            .prompt_replacements(
                &metadata(&tokenizer, &config),
                &preprocessed(&[0, 2, 4], &[0, 0, 0, 1, 0, 0]),
            )
            .unwrap();
        let lengths: Vec<usize> = replacements
            .iter()
            .map(|replacement| replacement.tokens.len())
            .collect();
        assert_eq!(lengths, vec![171, 337, 504]);
    }

    /// A mask whose length is not the crop count, or a missing side tensor,
    /// is the preprocessed batch's error, never a layout the engine did not
    /// write.
    #[test]
    fn a_mask_that_does_not_match_the_counts_is_refused() {
        let tokenizer = tokenizer();
        let config = config("step3_vl");
        assert_eq!(
            Step3VLVisionSpec
                .prompt_replacements(&metadata(&tokenizer, &config), &preprocessed(&[2], &[0]))
                .unwrap_err(),
            ModelRegistryError::InvalidPreprocessedField {
                field: PATCH_NEWLINE_MASK.to_string(),
            }
        );
        let mut without = preprocessed(&[1], &[0]);
        without.model_specific.remove(NUM_PATCHES);
        assert_eq!(
            Step3VLVisionSpec
                .prompt_replacements(&metadata(&tokenizer, &config), &without)
                .unwrap_err(),
            ModelRegistryError::InvalidPreprocessedField {
                field: NUM_PATCHES.to_string(),
            }
        );
    }

    /// The hooks table: the view batched, the crops and their flags flat by
    /// the crop counts, the counts and flags on the CPU, the anchor expandable
    /// by a worker; nothing for the other modalities.
    #[test]
    fn the_hooks_table_declares_the_layouts_of_the_engines_fields() {
        let spec = Step3VLVisionSpec;
        let layouts = spec.encoder_field_layouts_for(Modality::Image);
        assert_eq!(layouts.encoder_input, FieldLayout::Batched);
        assert_eq!(
            layouts.model_specific.get(PATCH_PIXEL_VALUES),
            Some(&FieldLayout::flat(NUM_PATCHES))
        );
        assert_eq!(
            layouts.model_specific.get(NUM_PATCHES),
            Some(&FieldLayout::Batched)
        );
        assert_eq!(
            layouts.model_specific.get(PATCH_NEWLINE_MASK),
            Some(&FieldLayout::flat(NUM_PATCHES))
        );
        assert_eq!(
            spec.keep_on_cpu_keys_for(Modality::Image),
            [NUM_PATCHES, PATCH_NEWLINE_MASK]
        );
        assert!(spec.worker_expandable(Modality::Image));
        assert_eq!(spec.encoder_input_key_for(Modality::Image), None);
        for modality in [Modality::Video, Modality::Audio, Modality::ImageEmbeds] {
            assert!(spec.hooks(modality).is_none(), "{modality}");
            assert!(!spec.worker_expandable(modality), "{modality}");
        }
        let tokenizer = tokenizer();
        let config = config("step3_vl");
        let metadata = metadata(&tokenizer, &config);
        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Image)
                .unwrap(),
            IMAGE_TOKEN
        );
        assert_eq!(
            spec.placeholder_token_id_for(&metadata, Modality::Image)
                .unwrap(),
            IMAGE_TOKEN_ID
        );
        assert_eq!(
            spec.modality_limits(&metadata).unwrap(),
            HashMap::from([(Modality::Image, MAX_IMAGES_PER_REQUEST)])
        );
        assert!(matches!(
            spec.vision_processor(&config, &PreProcessorConfig::default(), Modality::Video),
            Err(ModelRegistryError::UnsupportedModality {
                spec: "step3_vl",
                modality: Modality::Video,
            })
        ));
    }

    /// The processor the spec builds and the replacements it writes agree:
    /// the replacement of an image carries as many image tokens as the
    /// processor counts features for it, and its length is the layout's,
    /// crops, newlines and the structural tokens included.
    #[test]
    fn the_processors_feature_count_is_the_replacements_image_token_count() {
        let tokenizer = tokenizer();
        let config = config("step3_vl");
        let metadata = metadata(&tokenizer, &config);
        let processor = Step3VLVisionSpec
            .vision_processor(&config, &PreProcessorConfig::default(), Modality::Image)
            .unwrap();
        let wide = image::DynamicImage::ImageRgb8(RgbImage::from_pixel(600, 300, Rgb([9; 3])));
        let frame = image::DynamicImage::ImageRgb8(RgbImage::from_pixel(1920, 1080, Rgb([7; 3])));
        let out = processor.preprocess(&[&wide, &frame]).unwrap();
        let replacements = Step3VLVisionSpec
            .prompt_replacements(&metadata, &out)
            .unwrap();
        let image_tokens: Vec<usize> = replacements
            .iter()
            .map(|replacement| {
                replacement
                    .tokens
                    .iter()
                    .filter(|&&token| token == IMAGE_TOKEN_ID)
                    .count()
            })
            .collect();
        assert_eq!(image_tokens, out.feature_token_counts);
        assert_eq!(image_tokens, vec![331, 817]);
        let lengths: Vec<usize> = replacements
            .iter()
            .map(|replacement| replacement.tokens.len())
            .collect();
        assert_eq!(lengths, vec![337, 836]);
        assert_eq!(
            processor
                .calculate_num_tokens(ImageSize::new(1920, 1080).width, 1080)
                .unwrap(),
            817
        );
    }
}
