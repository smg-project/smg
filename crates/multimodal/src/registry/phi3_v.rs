use std::collections::HashMap;

use serde_json::Value;

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{
        layouts::batched_with_image_sizes, ModelMetadata, ModelProcessorSpec, ModelRegistryError,
        RegistryResult,
    },
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::{Phi3VisionProcessor, PreProcessorConfig, VisionPreProcessor},
};

pub(super) struct Phi3VisionSpec;

impl ModelProcessorSpec for Phi3VisionSpec {
    fn name(&self) -> &'static str {
        "phi3_v"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("phi") && id.contains("vision")
            || metadata
                .config_model_type()
                .is_some_and(|mt| mt == "phi3_v")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok("<|image|>".to_owned())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.token_id("<|image|>")
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Image, 4)]))
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        batched_with_image_sizes()
    }

    fn vision_processor(
        &self,
        _model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
        modality: Modality,
    ) -> RegistryResult<Box<dyn VisionPreProcessor>> {
        match modality {
            Modality::Image => Ok(Box::new(Phi3VisionProcessor::from_preprocessor_config(
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
        let token_id = self.placeholder_token_id(metadata)?;
        let token = self.placeholder_token(metadata)?;
        Ok(PromptReplacement::repeated_per_item(
            Modality::Image,
            &token,
            token_id,
            &preprocessed.feature_token_counts,
        ))
    }
}

#[cfg(test)]
mod tests {
    use image::Rgb;
    use serde_json::json;

    use crate::{
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        test_support::create_test_image,
        types::{ImageSize, Modality},
        vision::PreProcessorConfig,
    };

    #[test]
    fn phi3_uses_num_img_tokens() {
        let tokenizer = TestTokenizer::new(&[("<|image|>", 555)]);
        let config = json!({
            "model_type": "phi3_v",
            "img_processor": {"num_img_tokens": 144}
        });
        let metadata = ModelMetadata {
            model_id: "Phi-3-vision",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("phi3 spec");
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(336, 336)], &[144]),
            )
            .unwrap();
        assert_eq!(replacements[0].tokens.len(), 144);
        assert_eq!(replacements[0].tokens[0], 555);
    }

    #[test]
    fn phi3_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image|>", 555)]);
        let config = json!({
            "model_type": "phi3_v",
            "img_processor": {"num_img_tokens": 144}
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("phi3 alias");
        assert_eq!(spec.name(), "phi3_v");
    }

    /// The spec builds its processor from the checkpoint's preprocessor
    /// config: a `num_crops` of 4 yields five tiles (the global view and the
    /// four crops) and the token count of the 672x672 HD image, where the
    /// sixteen-crop default yields seventeen tiles and the 1344x1344 count.
    #[test]
    fn the_crop_count_is_the_preprocessor_configs_num_crops() {
        let tokenizer = TestTokenizer::new(&[("<|image|>", 555)]);
        let config = json!({"model_type": "phi3_v"});
        let metadata = ModelMetadata {
            model_id: "Phi-3.5-vision-instruct",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("phi3 spec");
        let preprocessor_config = PreProcessorConfig::from_json(r#"{"num_crops": 4}"#).unwrap();
        let processor = spec
            .vision_processor(&config, &preprocessor_config, Modality::Image)
            .expect("image processor");

        let output = processor
            .preprocess(&[&create_test_image(256, 256, Rgb([128, 128, 128]))])
            .unwrap();
        assert_eq!(
            output.encoder_input.shape()[1],
            5,
            "the global view and num_crops tiles"
        );
        assert_eq!(
            output.item_sizes,
            vec![(672, 672)],
            "the HD size of a 2x2 grid"
        );
        assert_eq!(
            output.feature_token_counts,
            vec![757],
            "(2 * 2 + 1) * 144 + 1 + (2 + 1) * 12"
        );
        assert_eq!(processor.calculate_num_tokens(256, 256).unwrap(), 757);
    }

    /// A checkpoint whose config names no crop count keeps the family's
    /// sixteen-crop default.
    #[test]
    fn a_config_without_num_crops_keeps_the_sixteen_crop_default() {
        let tokenizer = TestTokenizer::new(&[("<|image|>", 555)]);
        let config = json!({"model_type": "phi3_v"});
        let metadata = ModelMetadata {
            model_id: "Phi-3-vision",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("phi3 spec");
        let processor = spec
            .vision_processor(&config, &PreProcessorConfig::default(), Modality::Image)
            .expect("image processor");

        let output = processor
            .preprocess(&[&create_test_image(256, 256, Rgb([128, 128, 128]))])
            .unwrap();
        assert_eq!(
            output.encoder_input.shape()[1],
            17,
            "the global view and sixteen tiles"
        );
        assert_eq!(output.feature_token_counts, vec![2509], "a 4x4 grid");
    }
}
