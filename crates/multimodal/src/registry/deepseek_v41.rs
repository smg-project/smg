//! DeepSeek-V4.1 vision model contract.
//!
//! Mirrors the checkpoint's `inference/image_processor.py` and vLLM's
//! `models/deepseek_v4_1/common/mm_preprocess.py`: each `<｜deepseek_image｜>`
//! placeholder expands to a span where every position carries
//! `image_token_id` (the roles ride in the per-image `types` tensor). The
//! span is spliced verbatim — vLLM dropped its compressor-alignment pad
//! (vllm-project/vllm#56554), so no pad token is prepended.

use std::collections::HashMap;

use serde_json::Value;

use crate::{
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{
        MediaPartOrder, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    vision::{DeepseekV41Processor, PreProcessorConfig, VisionPreProcessor},
};

/// The placeholder text inlined at each image's position; the multimodal
/// pipeline later expands it into the image span. Public so the renderer can
/// emit it without re-resolving the spec (mirrors the reference encoding's
/// hard-coded `IMAGE_PLACEHOLDER`).
pub const DEEPSEEK_V41_IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

pub(super) struct DeepseekV41VisionSpec;

impl DeepseekV41VisionSpec {
    fn image_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata.config_token_id("image_token_id")
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
    /// position, so part order is protocol-visible: a string template of the
    /// family gets the media parts where the request put them. The family's
    /// own renderer takes the parts and places them itself, whatever this
    /// says, and its profile wins over any template, so this is read only
    /// for a model of the family served without it.
    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::Authored
    }

    /// `<｜deepseek_image｜>` (the config's `image_token_id`) is the single
    /// token the serving engine's prompt updates target, so a worker can
    /// expand it itself.
    fn worker_expandable(&self, modality: Modality) -> bool {
        matches!(modality, Modality::Image)
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(DEEPSEEK_V41_IMAGE_PLACEHOLDER.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_token_id(metadata)
    }

    /// The family sets no per-request image count of its own: its serving
    /// engines read the count as unlimited (the family's processing info
    /// declares `image: None`), the checkpoint's config bounds an image's
    /// tokens and pixels, not the number of images, and the context length
    /// is the practical bound. Images stay declared (an undeclared modality
    /// is refused as unsupported) without a bound of the spec's, so the
    /// effective limit is the engine's advertised one, else the deployment's
    /// `--mm-per-request-image-limit` / `SMG_IMAGE_MAX_COUNT`.
    ///
    /// What still bounds such a request, and where it lives. Before any
    /// fetch: the per-request limit the model's workers advertise for their
    /// engine, which caps the router's own limit in the gateway's limit
    /// resolution, and the deployment's `SMG_IMAGE_MAX_COUNT` /
    /// `SMG_VIDEO_MAX_COUNT` / `SMG_AUDIO_MAX_COUNT` (and
    /// `--mm-per-request-image-limit` for images), read in the same
    /// resolution. At or after the fetch: the gateway's cap on media bytes
    /// in flight (`multimodal_max_inflight_bytes`), the engine's advertised
    /// encoder budget and the context length. A number of the spec's own
    /// would be a declared default again, not a bound of the family's.
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
            Modality::Image => Ok(Box::new(DeepseekV41Processor::from_preprocessor_config(
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
        let image_token_id = Self::image_token_id(metadata)?;
        let placeholder = self.placeholder_token(metadata)?;
        // Every span position carries image_token_id; the roles live in
        // `types`. All of them are embed positions (delimiters get the
        // learned vectors from embed_multimodal, not the embed table).
        Ok(PromptReplacement::repeated_per_item(
            Modality::Image,
            &placeholder,
            image_token_id,
            &preprocessed.feature_token_counts,
        ))
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
    use serde_json::{json, Value};

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

    /// The image anchor is the one token the serving engine expands, so
    /// images may be forwarded to a worker that processes media itself.
    #[test]
    fn deepseek_v41_images_can_be_expanded_by_a_worker() {
        assert!(DeepseekV41VisionSpec.worker_expandable(Modality::Image));
        for modality in [Modality::Video, Modality::Audio, Modality::ImageEmbeds] {
            assert!(
                !DeepseekV41VisionSpec.worker_expandable(modality),
                "{modality}"
            );
        }
    }

    #[test]
    fn matches_deepseek_v41_model_type_and_ids() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({"model_type": "deepseek_v41", "image_token_id": IMAGE_TOKEN_ID});
        assert!(DeepseekV41VisionSpec.matches(&metadata(&tokenizer, &config)));
        assert_eq!(
            DeepseekV41VisionSpec
                .vision_processor(&config, &PreProcessorConfig::default(), Modality::Image)
                .map(|processor| processor.model_name()),
            Ok("deepseek_v41")
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

    /// The family bounds its images by context, not by count: the spec
    /// declares the modality without a limit of its own (the 128 it carried
    /// was a declared default, never a count of the family's), so a
    /// deployment knob or the engine's advertised limit is the only
    /// router-side bound.
    #[test]
    fn deepseek_v41_declares_images_without_a_count_of_its_own() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({"model_type": "deepseek_v41", "image_token_id": IMAGE_TOKEN_ID});
        let metadata = metadata(&tokenizer, &config);
        let spec = DeepseekV41VisionSpec;

        assert_eq!(
            spec.modality_limits(&metadata).unwrap(),
            HashMap::from([(Modality::Image, usize::MAX)])
        );
        for count in [129, 1_000] {
            spec.validate_media_request(&metadata, &[(Modality::Image, count)])
                .unwrap_or_else(|error| panic!("{count} images: {error}"));
        }
        // A deployment's own cap (the flag or the environment override, or
        // the engine's advertised limit the gateway passes in) still binds.
        let capped = HashMap::from([(Modality::Image, 128)]);
        assert_eq!(
            spec.validate_media_request_with_limits(&metadata, &[(Modality::Image, 129)], &capped),
            Err(ModelRegistryError::ModalityLimitExceeded {
                spec: "deepseek_v41",
                modality: Modality::Image,
                limit: 128,
                requested: 129,
            })
        );
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
            HashMap::from([(Modality::Image, usize::MAX)])
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
