//! Native vLLM Nemotron-3.5-Transcribe: one `<|audio|>` anchor expands to
//! one position per audio-tower embedding; the tower receives waveforms.

use std::collections::HashMap;

use serde_json::Value;

use crate::{
    audio::{
        AudioPreProcessor, NemotronTranscribeAudioProcessor, NemotronTranscribeConfigError,
        NEMOTRON_AUDIO_LENGTH_KEY,
    },
    registry::{
        MediaPartOrder, ModalityHooks, ModelMetadata, ModelProcessorSpec, ModelRegistryError,
        RegistryResult,
    },
    types::{EncoderFieldLayouts, FieldLayout, Modality},
    vision::PreProcessorConfig,
};

const AUDIO: &str = "<|audio|>";
const ARCHITECTURE: &str = "Nemotron3_5TranscribeForConditionalGeneration";

pub(super) struct NemotronTranscribeSpec;

impl NemotronTranscribeSpec {
    fn audio_replacements(
        metadata: &ModelMetadata,
        preprocessed: &crate::encoder_inputs::PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<crate::types::PromptReplacement>> {
        let id = metadata.token_id(AUDIO)?;
        preprocessed
            .feature_token_counts
            .iter()
            .map(|&count| {
                if count == 0 {
                    return Err(ModelRegistryError::InvalidPreprocessedField {
                        field: "feature_token_counts".to_string(),
                    });
                }
                Ok(crate::types::PromptReplacement::repeated(
                    Modality::Audio,
                    AUDIO,
                    id,
                    count,
                ))
            })
            .collect()
    }
}

impl ModelProcessorSpec for NemotronTranscribeSpec {
    fn name(&self) -> &'static str {
        "nemotron_transcribe"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let config = metadata.config;
        let named = metadata.model_id.to_ascii_lowercase();
        config
            .get("architectures")
            .and_then(Value::as_array)
            .is_some_and(|values| values.iter().any(|v| v.as_str() == Some(ARCHITECTURE)))
            || metadata.config_model_type() == Some("nemotron3_5_transcribe")
            || config
                .pointer("/audio_config/model_type")
                .and_then(Value::as_str)
                == Some("nemotron3_5_transcribe_audio")
            || named.contains("nemotron-3.5-transcribe")
            || named.contains("nemotron3_5_transcribe")
    }

    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::Authored
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        metadata.token_id(AUDIO)?;
        Ok(AUDIO.to_string())
    }

    fn placeholder_token_id(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<crate::types::TokenId> {
        metadata.token_id(AUDIO)
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        if modality != Modality::Audio {
            return Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            });
        }
        self.placeholder_token(metadata)
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<crate::types::TokenId> {
        if modality != Modality::Audio {
            return Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            });
        }
        self.placeholder_token_id(metadata)
    }

    fn hooks(&self, modality: Modality) -> Option<ModalityHooks> {
        (modality == Modality::Audio).then_some(ModalityHooks {
            placeholder: |metadata| {
                metadata.token_id(AUDIO)?;
                Ok(AUDIO.to_string())
            },
            placeholder_id: |metadata| metadata.token_id(AUDIO),
            replacements: |metadata, preprocessed, _, _| {
                Self::audio_replacements(metadata, preprocessed)
            },
            layouts: || {
                EncoderFieldLayouts::new(
                    FieldLayout::Batched,
                    HashMap::from([(NEMOTRON_AUDIO_LENGTH_KEY.to_string(), FieldLayout::Batched)]),
                )
            },
            keep_on_cpu: &[NEMOTRON_AUDIO_LENGTH_KEY],
            encoder_input_key: Some("audio_signal"),
            worker_expandable: true,
        })
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &crate::encoder_inputs::PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<crate::types::PromptReplacement>> {
        Self::audio_replacements(metadata, preprocessed)
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &crate::encoder_inputs::PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<crate::types::PromptReplacement>> {
        if modality != Modality::Audio {
            return Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            });
        }
        self.prompt_replacements(metadata, preprocessed)
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        // A conservative default; the server's per-modality override can raise it.
        Ok(HashMap::from([(Modality::Audio, 10)]))
    }

    fn audio_processor(
        &self,
        config: &Value,
        _preprocessor: &PreProcessorConfig,
    ) -> RegistryResult<Option<Box<dyn AudioPreProcessor>>> {
        let processor =
            NemotronTranscribeAudioProcessor::from_config(config).map_err(|error| match error {
                NemotronTranscribeConfigError::Missing { field } => {
                    ModelRegistryError::MissingConfigField { field }
                }
                NemotronTranscribeConfigError::Invalid { field, reason } => {
                    ModelRegistryError::InvalidConfigField { field, reason }
                }
            })?;
        Ok(Some(Box::new(processor)))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        registry::{
            test_helpers::{test_preprocessed_with_tokens, TestTokenizer},
            ModelRegistry,
        },
        types::ImageSize,
    };

    #[test]
    fn resolves_only_native_transcribe_family() {
        let tokenizer = TestTokenizer::new(&[(AUDIO, 131072)]);
        for config in [
            json!({"architectures":[ARCHITECTURE],"model_type":"nemo_speechlm"}),
            json!({"model_type":"nemotron3_5_transcribe"}),
            json!({"audio_config":{"model_type":"nemotron3_5_transcribe_audio"}}),
        ] {
            let metadata = ModelMetadata {
                model_id: "local-model",
                tokenizer: &tokenizer,
                config: &config,
            };
            let registry = ModelRegistry::new();
            assert_eq!(
                registry.lookup(&metadata).unwrap().name(),
                "nemotron_transcribe"
            );
        }
        for model_type in [
            "nemo_speechlm",
            "NemotronH_Nano_Omni_Reasoning_V3",
            "qwen3_asr",
        ] {
            let config = json!({"model_type":model_type});
            let metadata = ModelMetadata {
                model_id: "unrelated",
                tokenizer: &tokenizer,
                config: &config,
            };
            assert!(!NemotronTranscribeSpec.matches(&metadata));
        }
    }

    #[test]
    fn preserves_missing_and_invalid_config_diagnostics() {
        let spec = NemotronTranscribeSpec;
        let preprocessor = PreProcessorConfig::default();
        assert!(matches!(
            spec.audio_processor(&json!({}), &preprocessor),
            Err(ModelRegistryError::MissingConfigField { field })
                if field == "audio_config.preprocessor.sample_rate"
        ));
        assert!(matches!(
            spec.audio_processor(
                &json!({"audio_config": {"preprocessor": {"sample_rate": null}}}),
                &preprocessor,
            ),
            Err(ModelRegistryError::InvalidConfigField { field, .. })
                if field == "audio_config.preprocessor.sample_rate"
        ));
    }

    #[test]
    fn exposes_native_tensor_and_prompt_contract() {
        let tokenizer = TestTokenizer::new(&[(AUDIO, 131072)]);
        let config = json!({"model_type":"nemotron3_5_transcribe"});
        let metadata = ModelMetadata {
            model_id: "native",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = NemotronTranscribeSpec;
        assert_eq!(
            spec.encoder_input_key_for(Modality::Audio).as_deref(),
            Some("audio_signal")
        );
        assert_eq!(
            spec.keep_on_cpu_keys_for(Modality::Audio),
            [NEMOTRON_AUDIO_LENGTH_KEY]
        );
        assert!(spec.worker_expandable(Modality::Audio));
        assert!(!spec.worker_expandable(Modality::Image));
        assert!(matches!(
            spec.encoder_field_layouts_for(Modality::Audio)
                .encoder_input,
            FieldLayout::Batched
        ));
        assert_eq!(
            spec.placeholder_token_for(&metadata, Modality::Audio)
                .unwrap(),
            AUDIO
        );
        assert!(spec
            .placeholder_token_for(&metadata, Modality::Image)
            .is_err());
        assert!(spec
            .validate_media_request(&metadata, &[(Modality::Image, 1)])
            .is_err());
        let preprocessed =
            test_preprocessed_with_tokens(&[ImageSize::new(1, 1), ImageSize::new(1, 1)], &[1, 13]);
        let replacements = spec.prompt_replacements(&metadata, &preprocessed).unwrap();
        assert_eq!(replacements.len(), 2);
        assert_eq!(replacements[0].tokens, vec![131072]);
        assert_eq!(replacements[1].tokens, vec![131072; 13]);
        assert!(spec
            .audio_processor(&json!({}), &PreProcessorConfig::default())
            .is_err());
    }
}
