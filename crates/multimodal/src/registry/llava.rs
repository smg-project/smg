use std::collections::HashMap;

use serde_json::Value;

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    registry::{
        layouts::batched_with_image_sizes, ModelMetadata, ModelProcessorSpec, ModelRegistryError,
        RegistryResult,
    },
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::{
        LlavaNextProcessor, LlavaProcessor, PixtralProcessor, PreProcessorConfig,
        VisionPreProcessor,
    },
};

pub(super) struct LlavaSpec;
pub(super) struct LlavaNextSpec;

/// The placeholder a chat template renders for an image under a CLIP-style
/// tower.
const IMAGE_TOKEN: &str = "<image>";
/// The image tokens of the Pixtral tower: one per patch, a break after every
/// row of patches, an end after the last row.
const PIXTRAL_IMAGE_TOKEN: &str = "[IMG]";
const PIXTRAL_IMAGE_BREAK_TOKEN: &str = "[IMG_BREAK]";
const PIXTRAL_IMAGE_END_TOKEN: &str = "[IMG_END]";
/// The patch size of the Pixtral tower when `vision_config` leaves it out.
const PIXTRAL_PATCH_SIZE: u32 = 16;

/// The vision tower a LLaVA checkpoint pairs with its language model, read
/// from `vision_config.model_type` as the serving engine reads it: the tower
/// decides which image processor the checkpoint ships and how its image
/// tokens are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VisionTower {
    /// A CLIP-style tower: a fixed-size crop, one feature token per patch,
    /// the `<image>` placeholder repeated.
    Clip,
    /// The Pixtral tower: the image at its own aspect ratio within
    /// `longest_edge`, one `[IMG]` per patch, `[IMG_BREAK]` closing every
    /// row of patches and `[IMG_END]` closing the last.
    Pixtral,
}

impl VisionTower {
    fn of(config: &Value) -> Self {
        match config
            .get("vision_config")
            .and_then(|vision| vision.get("model_type"))
            .and_then(Value::as_str)
        {
            Some("pixtral") => Self::Pixtral,
            _ => Self::Clip,
        }
    }
}

/// The Pixtral tower's image tokens, one replacement per image, in the
/// layout the checkpoint's processor and the serving engine's prompt update
/// both produce: for every row of the patch grid, one `[IMG]` per patch and
/// then `[IMG_BREAK]`, with `[IMG_END]` in place of the last row's break.
/// The grid is the processor's `image_sizes` (the resized height and width)
/// over the tower's patch size, and it must hold exactly the feature tokens
/// the processor counted: the tower's patch size comes from `config.json`
/// and the processor's from the preprocessor config, so a checkpoint whose
/// two files disagree is refused here rather than handed to the engine with
/// the wrong number of `[IMG]` positions. Only the `[IMG]` positions carry
/// encoder features, which the gateway finds by the placeholder id.
fn pixtral_replacements(
    metadata: &ModelMetadata,
    preprocessed: &PreprocessedEncoderInputs,
    token: &str,
    token_id: TokenId,
) -> RegistryResult<Vec<PromptReplacement>> {
    let break_id = metadata.token_id(PIXTRAL_IMAGE_BREAK_TOKEN)?;
    let end_id = metadata.token_id(PIXTRAL_IMAGE_END_TOKEN)?;
    let patch = metadata
        .config_u32(&["vision_config", "patch_size"])
        .unwrap_or(PIXTRAL_PATCH_SIZE)
        .max(1) as usize;
    let counts = &preprocessed.feature_token_counts;
    let sizes = match preprocessed.model_specific.get("image_sizes") {
        Some(ModelSpecificValue::IntTensor { data, shape })
            if shape.len() == 2
                && shape[1] == 2
                && shape[0] == counts.len()
                && data.len() == counts.len() * 2 =>
        {
            data
        }
        _ => {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "image_sizes".to_string(),
            })
        }
    };
    sizes
        .as_chunks::<2>()
        .0
        .iter()
        .zip(counts)
        .map(|(size, &count)| {
            let (height, width) = (size[0] as usize, size[1] as usize);
            let (rows, cols) = (height / patch, width / patch);
            if rows * cols != count {
                return Err(ModelRegistryError::InvalidConfigField {
                    field: "vision_config.patch_size".to_string(),
                    reason: format!(
                        "a {height}x{width} image in {patch}-pixel patches has {rows}x{cols} = {} \
                         tokens, but the processor counted {count}",
                        rows * cols
                    ),
                });
            }
            let mut tokens = Vec::with_capacity(rows * (cols + 1));
            for row in 0..rows {
                tokens.extend(std::iter::repeat_n(token_id, cols));
                tokens.push(if row + 1 == rows { end_id } else { break_id });
            }
            Ok(PromptReplacement::sequence(Modality::Image, token, tokens))
        })
        .collect()
}

impl ModelProcessorSpec for LlavaSpec {
    fn name(&self) -> &'static str {
        "llava"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        // Match plain "llava" but not "llava_next" (handled by LlavaNextSpec).
        let model_type = metadata.config_model_type();
        if model_type.is_some_and(|mt| mt == "llava_next") {
            return false;
        }
        let model_id_lower = metadata.model_id.to_ascii_lowercase();
        if model_id_lower.contains("llava-next") || model_id_lower.contains("llava_next") {
            return false;
        }
        model_id_lower.contains("llava") || model_type.is_some_and(|mt| mt == "llava")
    }

    /// The placeholder (`<image>`, or `[IMG]` under the Pixtral tower; the
    /// config's `image_token_index`) is the single token the serving
    /// engine's prompt updates target, so a worker can expand it itself.
    fn worker_expandable(&self, modality: Modality) -> bool {
        matches!(modality, Modality::Image)
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(match VisionTower::of(metadata.config) {
            VisionTower::Clip => IMAGE_TOKEN,
            VisionTower::Pixtral => PIXTRAL_IMAGE_TOKEN,
        }
        .to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        if let Some(value) = metadata.config_u32(&["image_token_index"]) {
            return Ok(value as TokenId);
        }
        metadata.token_id(&self.placeholder_token(metadata)?)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 4)]))
    }

    /// The processor of the checkpoint's vision tower: the Pixtral processor
    /// for a Pixtral tower (its `longest_edge` and `patch_size` from the
    /// preprocessor config), the LLaVA processor otherwise.
    fn vision_processor(
        &self,
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image => Ok(match VisionTower::of(model_config) {
                VisionTower::Clip => Box::new(LlavaProcessor::from_configs(
                    model_config,
                    preprocessor_config,
                )),
                VisionTower::Pixtral => Box::new(PixtralProcessor::from_preprocessor_config(
                    preprocessor_config,
                )),
            }),
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
        let token_id = self.placeholder_token_id(metadata)?;
        let token = self.placeholder_token(metadata)?;
        match VisionTower::of(metadata.config) {
            VisionTower::Clip => Ok(PromptReplacement::repeated_per_item(
                Modality::Image,
                &token,
                token_id,
                &preprocessed.feature_token_counts,
            )),
            VisionTower::Pixtral => pixtral_replacements(metadata, preprocessed, &token, token_id),
        }
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // pixel_values is [num_images, C, H, W]; the Pixtral tower's
        // image_sizes is [num_images, 2] (a CLIP tower emits none).
        batched_with_image_sizes()
    }
}

impl ModelProcessorSpec for LlavaNextSpec {
    fn name(&self) -> &'static str {
        "llava_next"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        metadata
            .config_model_type()
            .is_some_and(|mt| mt == "llava_next")
    }

    /// `<image>` (the config's `image_token_index`) is the single token the
    /// serving engine's prompt updates target, so a worker can expand it
    /// itself.
    fn worker_expandable(&self, modality: Modality) -> bool {
        matches!(modality, Modality::Image)
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        LlavaSpec.placeholder_token(metadata)
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        LlavaSpec.placeholder_token_id(metadata)
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        LlavaSpec.modality_limits(metadata)
    }

    fn vision_processor(
        &self,
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image => Ok(Box::new(LlavaNextProcessor::from_configs(
                model_config,
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
        // LLaVA-Next token counts differ from plain LLaVA because of
        // anyres multi-crop + spatial_unpad.  The correct per-image counts
        // are already computed by LlavaNextProcessor::calculate_num_tokens
        // and stored in preprocessed.feature_token_counts.
        let token_id = LlavaSpec.placeholder_token_id(metadata)?;
        let token = LlavaSpec.placeholder_token(metadata)?;
        Ok(PromptReplacement::repeated_per_item(
            Modality::Image,
            &token,
            token_id,
            &preprocessed.feature_token_counts,
        ))
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // encoder_input is [num_images, max_patches, C, H, W] (5D, batched).
        // image_sizes is [num_images, 2] (batched), matching the reference processor's kwargs.
        batched_with_image_sizes()
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
    fn llava_prompt_replacement_uses_preprocessed_tokens() {
        let tokenizer = TestTokenizer::new(&[("<image>", 32000)]);
        let config = json!({
            "model_type": "llava",
            "image_token_index": 32000,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "llava-v1.5",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llava spec");
        // Token count comes from preprocessed.feature_token_counts (set by
        // LlavaProcessor::calculate_num_tokens), not from image dimensions.
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(336, 336)], &[576]);
        let replacements = spec.prompt_replacements(&metadata, &preprocessed).unwrap();
        assert_eq!(replacements[0].tokens.len(), 576);
    }

    #[test]
    fn llava_prompt_replacement_uses_per_image_token_counts() {
        let tokenizer = TestTokenizer::new(&[("<image>", 32000)]);
        let config = json!({
            "model_type": "llava",
            "image_token_index": 32000,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "llava-v1.5",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llava spec");
        let preprocessed = test_preprocessed_with_tokens(
            &[ImageSize::new(336, 336), ImageSize::new(448, 448)],
            &[576, 1024],
        );

        let replacements = spec.prompt_replacements(&metadata, &preprocessed).unwrap();

        assert_eq!(replacements.len(), 2);
        assert_eq!(replacements[0].tokens.len(), 576);
        assert_eq!(replacements[1].tokens.len(), 1024);
    }

    /// Both specs render the one token the serving engine expands, so their
    /// images may be forwarded to a worker that processes media itself.
    #[test]
    fn llava_images_can_be_expanded_by_a_worker() {
        use crate::{registry::ModelProcessorSpec, types::Modality};

        for spec in [
            &super::LlavaSpec as &dyn ModelProcessorSpec,
            &super::LlavaNextSpec,
        ] {
            assert!(spec.worker_expandable(Modality::Image), "{}", spec.name());
            assert!(!spec.worker_expandable(Modality::Video), "{}", spec.name());
            assert!(!spec.worker_expandable(Modality::Audio), "{}", spec.name());
        }
    }

    #[test]
    fn llava_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<image>", 32000)]);
        let config = json!({
            "model_type": "llava",
            "image_token_index": 32000,
            "vision_config": {"patch_size": 14}
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llava alias");
        assert_eq!(spec.name(), "llava");
    }

    #[test]
    fn llava_spec_has_no_audio_processor() {
        use crate::vision::PreProcessorConfig;

        let tokenizer = TestTokenizer::new(&[("<image>", 32000)]);
        let config = json!({"model_type": "llava", "image_token_index": 32000});
        let metadata = ModelMetadata {
            model_id: "llava-v1.5",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("llava spec");
        assert!(spec
            .audio_processor(&config, &PreProcessorConfig::default())
            .unwrap()
            .is_none());
    }

    /// A LLaVA checkpoint declares its vision tower in `vision_config`: the
    /// Pixtral tower takes the Pixtral processor, built from the
    /// preprocessor config, and its `[IMG]` placeholder; a CLIP tower keeps
    /// the LLaVA processor and `<image>`.
    #[test]
    fn the_vision_tower_picks_the_processor_and_the_placeholder() {
        use crate::{types::FieldLayout, vision::PreProcessorConfig, Modality};

        let registry = ModelRegistry::new();
        let tokenizer = pixtral_tokenizer();
        let config = pixtral_config();
        let metadata = ModelMetadata {
            model_id: "/models/checkpoint",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = registry.lookup(&metadata).expect("llava spec");
        assert_eq!(spec.name(), "llava");
        let processor = spec
            .vision_processor(&config, &pixtral_preprocessor_config(), Modality::Image)
            .unwrap();
        assert_eq!(processor.model_name(), "pixtral");
        assert_eq!(spec.placeholder_token(&metadata).unwrap(), "[IMG]");
        assert_eq!(spec.placeholder_token_id(&metadata).unwrap(), 10);
        assert!(spec.worker_expandable(Modality::Image));
        assert_eq!(
            spec.field_layouts().get("image_sizes"),
            Some(&FieldLayout::Batched)
        );

        let tokenizer = TestTokenizer::new(&[("<image>", 32000)]);
        let config = json!({
            "model_type": "llava",
            "image_token_index": 32000,
            "vision_config": {"model_type": "clip_vision_model", "patch_size": 14, "image_size": 336}
        });
        let metadata = ModelMetadata {
            model_id: "/models/checkpoint",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = registry.lookup(&metadata).expect("llava spec");
        let processor = spec
            .vision_processor(&config, &PreProcessorConfig::default(), Modality::Image)
            .unwrap();
        assert_eq!(processor.model_name(), "llava");
        assert_eq!(spec.placeholder_token(&metadata).unwrap(), "<image>");
        assert_eq!(spec.placeholder_token_id(&metadata).unwrap(), 32000);
    }

    /// Under the Pixtral tower an image takes the grid of its resized size:
    /// the processor's shape and feature count follow it, and the
    /// replacement is one `[IMG]` per patch, `[IMG_BREAK]` after every row
    /// of patches and `[IMG_END]` after the last.
    #[test]
    fn the_pixtral_tower_lays_out_the_image_tokens_by_rows_of_patches() {
        use image::{DynamicImage, Rgb, RgbImage};

        use crate::Modality;

        let registry = ModelRegistry::new();
        let tokenizer = pixtral_tokenizer();
        let config = pixtral_config();
        let metadata = ModelMetadata {
            model_id: "/models/checkpoint",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = registry.lookup(&metadata).expect("llava spec");
        let processor = spec
            .vision_processor(&config, &pixtral_preprocessor_config(), Modality::Image)
            .unwrap();
        // (width, height) -> the resized (height, width), rows and columns of
        // 16-pixel patches within a longest edge of 1024.
        for (width, height, resized, rows, cols) in [
            (256, 256, (256, 256), 16, 16),
            (1024, 768, (768, 1024), 48, 64),
            (1920, 1080, (576, 1024), 36, 64),
        ] {
            let image =
                DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb([90, 120, 200])));
            let preprocessed = processor.preprocess(&[&image]).unwrap();
            assert_eq!(
                preprocessed.encoder_input.shape(),
                &[1, 3, resized.0, resized.1],
                "{width}x{height}"
            );
            assert_eq!(preprocessed.feature_token_counts, vec![rows * cols]);
            let replacements = spec.prompt_replacements(&metadata, &preprocessed).unwrap();
            assert_eq!(replacements.len(), 1);
            let replacement = &replacements[0];
            assert_eq!(replacement.placeholder_token, "[IMG]");
            assert_eq!(
                replacement.tokens.len(),
                rows * cols + rows,
                "{width}x{height}"
            );
            let count = |id| replacement.tokens.iter().filter(|&&t| t == id).count();
            assert_eq!(count(10), rows * cols);
            assert_eq!(count(12), rows - 1);
            assert_eq!(count(13), 1);
            assert_eq!(replacement.tokens.last(), Some(&13));
            assert!(replacement
                .tokens
                .chunks_exact(cols + 1)
                .all(|row| row[..cols].iter().all(|&t| t == 10) && row[cols] != 10));
        }
    }

    /// Without the processor's `image_sizes` the grid is unknown, and the
    /// spec says which field it misses rather than guessing a layout.
    #[test]
    fn a_pixtral_replacement_without_image_sizes_is_refused() {
        use crate::registry::ModelRegistryError;

        let registry = ModelRegistry::new();
        let tokenizer = pixtral_tokenizer();
        let config = pixtral_config();
        let metadata = ModelMetadata {
            model_id: "/models/checkpoint",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = registry.lookup(&metadata).expect("llava spec");
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(256, 256)], &[256]);
        assert_eq!(
            spec.prompt_replacements(&metadata, &preprocessed)
                .unwrap_err(),
            ModelRegistryError::InvalidPreprocessedField {
                field: "image_sizes".to_string()
            }
        );
    }

    /// The tower's patch size (`config.json`) and the processor's (the
    /// preprocessor config) are two files of one checkpoint; when they
    /// disagree the grid no longer holds the processor's feature tokens and
    /// the spec refuses the request, naming the field, instead of handing
    /// the engine the wrong number of `[IMG]` positions.
    #[test]
    fn a_patch_size_that_disagrees_with_the_processor_is_refused() {
        use image::{DynamicImage, Rgb, RgbImage};

        use crate::{registry::ModelRegistryError, Modality};

        let registry = ModelRegistry::new();
        let tokenizer = pixtral_tokenizer();
        let mut config = pixtral_config();
        config["vision_config"]["patch_size"] = json!(14);
        let metadata = ModelMetadata {
            model_id: "/models/checkpoint",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = registry.lookup(&metadata).expect("llava spec");
        // The processor keeps the preprocessor config's 16-pixel patches.
        let processor = spec
            .vision_processor(&config, &pixtral_preprocessor_config(), Modality::Image)
            .unwrap();
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(256, 256, Rgb([90, 120, 200])));
        let preprocessed = processor.preprocess(&[&image]).unwrap();
        assert_eq!(preprocessed.feature_token_counts, vec![256]);
        let error = spec
            .prompt_replacements(&metadata, &preprocessed)
            .unwrap_err();
        assert!(
            matches!(
                error,
                ModelRegistryError::InvalidConfigField { ref field, .. }
                    if field == "vision_config.patch_size"
            ),
            "{error}"
        );
    }

    /// `image_sizes` describes every image of the batch, one row each; a
    /// row count that differs from the processor's counts is refused.
    #[test]
    fn an_image_sizes_row_count_that_differs_from_the_batch_is_refused() {
        use crate::{encoder_inputs::ModelSpecificValue, registry::ModelRegistryError};

        let registry = ModelRegistry::new();
        let tokenizer = pixtral_tokenizer();
        let config = pixtral_config();
        let metadata = ModelMetadata {
            model_id: "/models/checkpoint",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = registry.lookup(&metadata).expect("llava spec");
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(256, 256)], &[256])
            .with_extra(
                "image_sizes",
                ModelSpecificValue::int_2d(vec![256, 256, 256, 256], 2, 2),
            );
        assert_eq!(
            spec.prompt_replacements(&metadata, &preprocessed)
                .unwrap_err(),
            ModelRegistryError::InvalidPreprocessedField {
                field: "image_sizes".to_string()
            }
        );
    }

    fn pixtral_tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[("[IMG]", 10), ("[IMG_BREAK]", 12), ("[IMG_END]", 13)])
    }

    /// A LLaVA-format checkpoint with the Pixtral tower (`config.json`).
    fn pixtral_config() -> serde_json::Value {
        json!({
            "model_type": "llava",
            "image_token_index": 10,
            "vision_config": {"model_type": "pixtral", "patch_size": 16, "image_size": 1024}
        })
    }

    /// Its `preprocessor_config.json`.
    fn pixtral_preprocessor_config() -> crate::vision::PreProcessorConfig {
        crate::vision::PreProcessorConfig::from_json(
            r#"{"image_processor_type":"PixtralImageProcessor","size":{"longest_edge":1024},
                "patch_size":{"height":16,"width":16},"resample":3,
                "image_mean":[0.48145466,0.4578275,0.40821073],
                "image_std":[0.26862954,0.26130258,0.27577711],
                "do_resize":true,"do_rescale":true,"rescale_factor":0.00392156862745098,
                "do_normalize":true,"do_convert_rgb":true}"#,
        )
        .unwrap()
    }
}
