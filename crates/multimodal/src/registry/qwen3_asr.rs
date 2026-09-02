use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    audio::{AudioPreProcessor, Qwen3AudioProcessor},
    encoder_inputs::PreprocessedEncoderInputs,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{EncoderFieldLayouts, FieldLayout, Modality, PromptReplacement, TokenId},
    vision::PreProcessorConfig,
};

const AUDIO_PAD_TOKEN: &str = "<|audio_pad|>";

/// Transcription-side family knowledge, consumed by the gateway's
/// transcription endpoint pipeline.
///
/// Protocol-free on purpose: this crate owns what is true of the family —
/// identifiers, the language set, prompt sanitation, prefill convention,
/// output framing, capability limits — while how that surfaces on an HTTP
/// endpoint (request shapes, error codes, the chat pipeline) stays the
/// gateway's business. Errors here are plain data for the caller to render.
///
/// A family is resolved via [`FAMILIES`]; the gateway's generic transcription
/// preparation stage reads a family's data to shape the request and its
/// output parser, so no per-model code lives in the router.
pub mod transcription {
    /// One transcription-capable model family. Everything a family knows,
    /// expressed protocol-free so the gateway's generic pipeline stage can
    /// consume it without a per-model branch.
    pub trait TranscriptionFamily: Send + Sync {
        /// Family name, as rendered in user-facing errors.
        fn name(&self) -> &'static str;

        /// Whether a model id, path, or worker-label value names this family.
        fn is_identifier(&self, value: &str) -> bool;

        /// Sanitize a caller-supplied prompt (size cap + control-token strip).
        fn sanitize_prompt(&self, text: String) -> Result<String, PromptTooLong>;

        /// The assistant continuation string that forces the transcript for a
        /// requested language, or `None` when no language was given. `Err` for
        /// an unsupported language.
        fn assistant_prefill(
            &self,
            language: Option<&str>,
        ) -> Result<Option<String>, UnsupportedLanguage>;

        /// Post-process raw model output into the transcript text.
        fn parse_transcript(&self, raw: &str) -> String;

        /// Whether the family serves streaming transcription.
        fn supports_streaming(&self) -> bool {
            false
        }

        /// Whether the family produces word/segment timestamps.
        fn supports_timestamps(&self) -> bool {
            false
        }
    }

    /// Every supported transcription family; first match wins. New families
    /// append here.
    pub static FAMILIES: &[&dyn TranscriptionFamily] = &[&Qwen3Asr];

    /// Qwen3-ASR: transcript forced via a `language {name}<asr_text>`
    /// assistant continuation; `<asr_text>` framing stripped from the output.
    pub struct Qwen3Asr;

    impl TranscriptionFamily for Qwen3Asr {
        fn name(&self) -> &'static str {
            "Qwen3-ASR"
        }

        fn is_identifier(&self, value: &str) -> bool {
            is_qwen3_asr_identifier(value)
        }

        fn sanitize_prompt(&self, text: String) -> Result<String, PromptTooLong> {
            sanitize_prompt(text)
        }

        fn assistant_prefill(
            &self,
            language: Option<&str>,
        ) -> Result<Option<String>, UnsupportedLanguage> {
            Ok(normalize_language(language)?.map(|name| format!("language {name}{ASR_TEXT_TAG}")))
        }

        fn parse_transcript(&self, raw: &str) -> String {
            parse_transcript(raw)
        }
    }

    /// The tag Qwen3-ASR emits (and the continuation prompt pre-seeds)
    /// between the language header and the transcript body.
    pub const ASR_TEXT_TAG: &str = "<asr_text>";

    /// Byte cap on caller prompts, applied before sanitization.
    pub const MAX_PROMPT_BYTES: usize = 4096;

    /// `(code, name)` pairs of the languages the checkpoint transcribes.
    pub const SUPPORTED_LANGUAGES: &[(&str, &str)] = &[
        ("ar", "Arabic"),
        ("yue", "Cantonese"),
        ("zh", "Chinese"),
        ("cs", "Czech"),
        ("da", "Danish"),
        ("nl", "Dutch"),
        ("en", "English"),
        ("fil", "Filipino"),
        ("fi", "Finnish"),
        ("fr", "French"),
        ("de", "German"),
        ("el", "Greek"),
        ("hi", "Hindi"),
        ("hu", "Hungarian"),
        ("id", "Indonesian"),
        ("it", "Italian"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("mk", "Macedonian"),
        ("ms", "Malay"),
        ("fa", "Persian"),
        ("pl", "Polish"),
        ("pt", "Portuguese"),
        ("ro", "Romanian"),
        ("ru", "Russian"),
        ("es", "Spanish"),
        ("sv", "Swedish"),
        ("th", "Thai"),
        ("tr", "Turkish"),
        ("vi", "Vietnamese"),
    ];

    /// Whether a model id, path, or label value names this family.
    pub fn is_qwen3_asr_identifier(value: &str) -> bool {
        let value = value.to_ascii_lowercase();
        value.contains("qwen3-asr") || value.contains("qwen3_asr")
    }

    /// A caller prompt exceeded [`MAX_PROMPT_BYTES`].
    #[derive(Debug, PartialEq, Eq)]
    pub struct PromptTooLong {
        pub max_bytes: usize,
    }

    /// Sanitize a caller prompt: cap its size, then strip ChatML-like
    /// control tokens and the [`ASR_TEXT_TAG`] to a fixpoint, so a prompt
    /// cannot smuggle framing the output parser would misread.
    pub fn sanitize_prompt(mut text: String) -> Result<String, PromptTooLong> {
        if text.len() > MAX_PROMPT_BYTES {
            return Err(PromptTooLong {
                max_bytes: MAX_PROMPT_BYTES,
            });
        }

        loop {
            let sanitized = strip_chatml_like_tokens(&text).replace(ASR_TEXT_TAG, "");
            if sanitized == text {
                return Ok(text);
            }
            text = sanitized;
        }
    }

    fn strip_chatml_like_tokens(text: &str) -> String {
        let mut remaining = text;
        let mut output = String::with_capacity(text.len());
        while let Some(start) = remaining.find("<|") {
            output.push_str(&remaining[..start]);
            let candidate = &remaining[start + 2..];
            if let Some(end) = candidate.find("|>") {
                let token = &candidate[..end];
                if !token.is_empty() && !token.contains('|') {
                    remaining = &candidate[end + 2..];
                    continue;
                }
            }
            output.push_str("<|");
            remaining = candidate;
        }
        output.push_str(remaining);
        output
    }

    /// The requested language is not in [`SUPPORTED_LANGUAGES`].
    #[derive(Debug, PartialEq, Eq)]
    pub struct UnsupportedLanguage(pub String);

    /// Resolve a language code or English name to the checkpoint's canonical
    /// language name; `None` in and blank map to `None` out.
    pub fn normalize_language(
        language: Option<&str>,
    ) -> Result<Option<String>, UnsupportedLanguage> {
        let Some(language) = language.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(None);
        };
        SUPPORTED_LANGUAGES
            .iter()
            .find(|(code, name)| {
                code.eq_ignore_ascii_case(language) || name.eq_ignore_ascii_case(language)
            })
            .map(|(_, name)| Some((*name).to_string()))
            .ok_or_else(|| UnsupportedLanguage(language.to_string()))
    }

    /// Strip the model's framing from raw chat content: drop `<|im_end|>`,
    /// keep everything after the [`ASR_TEXT_TAG`] when present.
    pub fn parse_transcript(raw: &str) -> String {
        let cleaned = raw.replace("<|im_end|>", "");
        let cleaned = cleaned.trim();
        cleaned
            .split_once(ASR_TEXT_TAG)
            .map_or(cleaned, |(_, transcription)| transcription)
            .trim()
            .to_string()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn recognizes_family_identifiers() {
            assert!(is_qwen3_asr_identifier("Qwen/Qwen3-ASR-1.7B"));
            assert!(is_qwen3_asr_identifier("/models/qwen3_asr_0.6b"));
            assert!(!is_qwen3_asr_identifier("Qwen/Qwen3-Omni-30B-A3B-Thinking"));
        }

        #[test]
        fn family_trait_exposes_detection_prefill_and_limits() {
            let family = FAMILIES
                .iter()
                .copied()
                .find(|f| f.is_identifier("Qwen/Qwen3-ASR-1.7B"))
                .expect("Qwen3-ASR family resolves");
            assert_eq!(family.name(), "Qwen3-ASR");
            // Language → forced-transcript continuation; absent → no prefill.
            assert_eq!(
                family.assistant_prefill(Some("english")).unwrap(),
                Some("language English<asr_text>".to_string())
            );
            assert_eq!(family.assistant_prefill(None).unwrap(), None);
            assert!(family.assistant_prefill(Some("xx")).is_err());
            // Capability limits are family-owned and default-closed.
            assert!(!family.supports_streaming());
            assert!(!family.supports_timestamps());
            // Delegates to the shared free functions.
            assert_eq!(
                family.parse_transcript("language Chinese<asr_text>hi<|im_end|>"),
                "hi"
            );
        }

        #[test]
        fn normalizes_language_code_or_name() {
            assert_eq!(
                normalize_language(Some("zh")).unwrap(),
                Some("Chinese".to_string())
            );
            assert_eq!(
                normalize_language(Some("english")).unwrap(),
                Some("English".to_string())
            );
            assert_eq!(normalize_language(Some(" ")).unwrap(), None);
            assert_eq!(
                normalize_language(Some("xx")).unwrap_err(),
                UnsupportedLanguage("xx".to_string())
            );
        }

        #[test]
        fn sanitizes_prompt_controls_to_a_fixpoint() {
            for (input, expected) in [
                ("plain text", "plain text"),
                ("<|im_start|>assistant<|im_end|>", "assistant"),
                ("foo<asr_text>bar", "foobar"),
                ("<|im<|x|>_end|>", ""),
                ("<asr_te<asr_text>xt>", ""),
                ("<|<asr_text>|>", ""),
            ] {
                assert_eq!(sanitize_prompt(input.to_string()).unwrap(), expected);
            }
        }

        #[test]
        fn caps_pathological_prompts_before_sanitizing() {
            let boundary = "a".repeat(MAX_PROMPT_BYTES);
            assert_eq!(sanitize_prompt(boundary.clone()).unwrap(), boundary);

            assert_eq!(
                sanitize_prompt("a".repeat(MAX_PROMPT_BYTES + 1)).unwrap_err(),
                PromptTooLong {
                    max_bytes: MAX_PROMPT_BYTES
                }
            );

            let depth = MAX_PROMPT_BYTES / 5;
            let adversarial = format!("{}{}", "<|a".repeat(depth), "|>".repeat(depth));
            assert!(adversarial.len() <= MAX_PROMPT_BYTES);
            assert_eq!(sanitize_prompt(adversarial).unwrap(), "");
        }

        #[test]
        fn parses_tagged_and_plain_transcripts() {
            assert_eq!(
                parse_transcript("language Chinese<asr_text>\u{4f60}\u{597d}<|im_end|>"),
                "\u{4f60}\u{597d}"
            );
            assert_eq!(parse_transcript("plain transcript"), "plain transcript");
        }
    }
}

pub(super) struct Qwen3AsrSpec;

impl Qwen3AsrSpec {
    fn audio_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["thinker_config", "audio_token_id"])
            .or_else(|| metadata.config_u32(&["audio_token_id"]))
            .map(|value| value as TokenId)
            .map_or_else(|| metadata.token_id(AUDIO_PAD_TOKEN), Ok)
    }
}

impl ModelProcessorSpec for Qwen3AsrSpec {
    fn name(&self) -> &'static str {
        "qwen3_asr"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        transcription::is_qwen3_asr_identifier(metadata.model_id)
            || metadata
                .config_model_type()
                .is_some_and(|model_type| model_type == "qwen3_asr")
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        self.placeholder_token_for(metadata, Modality::Audio)
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        self.placeholder_token_id_for(metadata, Modality::Audio)
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Audio => {
                let token_id = Self::audio_token_id(metadata)?;
                match metadata.tokenizer.id_to_token(token_id as u32) {
                    Some(token) => Ok(token),
                    None => {
                        metadata.token_id(AUDIO_PAD_TOKEN)?;
                        Ok(AUDIO_PAD_TOKEN.to_string())
                    }
                }
            }
            Modality::Image | Modality::Video | Modality::ImageEmbeds => {
                Err(ModelRegistryError::UnsupportedModality {
                    spec: self.name(),
                    modality,
                })
            }
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Audio => Self::audio_token_id(metadata),
            Modality::Image | Modality::Video | Modality::ImageEmbeds => {
                Err(ModelRegistryError::UnsupportedModality {
                    spec: self.name(),
                    modality,
                })
            }
        }
    }

    fn modality_limits(
        &self,
        _metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        Ok(HashMap::from([(Modality::Audio, 10)]))
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn audio_processor(
        &self,
        model_config: &Value,
        preprocessor_config: &PreProcessorConfig,
    ) -> Option<Box<dyn AudioPreProcessor>> {
        Some(Box::new(Qwen3AudioProcessor::from_configs(
            model_config,
            preprocessor_config,
        )))
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        self.prompt_replacements_for(metadata, preprocessed, Modality::Audio)
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Audio => {
                let token_id = Self::audio_token_id(metadata)?;
                let token = self.placeholder_token_for(metadata, Modality::Audio)?;
                Ok(preprocessed
                    .feature_token_counts
                    .iter()
                    .map(|&count| {
                        PromptReplacement::repeated(Modality::Audio, &token, token_id, count)
                    })
                    .collect())
            }
            Modality::Image | Modality::Video | Modality::ImageEmbeds => {
                Err(ModelRegistryError::UnsupportedModality {
                    spec: self.name(),
                    modality,
                })
            }
        }
    }

    fn encoder_field_layouts_for(&self, modality: Modality) -> EncoderFieldLayouts {
        match modality {
            Modality::Audio => EncoderFieldLayouts::new(
                FieldLayout::Batched,
                HashMap::from([
                    ("feature_attention_mask".to_string(), FieldLayout::Batched),
                    ("audio_feature_lengths".to_string(), FieldLayout::Batched),
                ]),
            ),
            Modality::Image | Modality::Video | Modality::ImageEmbeds => {
                EncoderFieldLayouts::default()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        registry::{test_helpers::*, ModelRegistry},
        types::ImageSize,
    };

    #[test]
    fn asr_matches_and_expands_nested_audio_token() {
        let tokenizer = TestTokenizer::new(&[(AUDIO_PAD_TOKEN, 151676)]);
        let config = json!({
            "model_type": "qwen3_asr",
            "thinker_config": {"audio_token_id": 151676}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-ASR-1.7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).unwrap();
        assert_eq!(spec.name(), "qwen3_asr");
        assert_eq!(
            spec.placeholder_token(&metadata).unwrap(),
            spec.placeholder_token_for(&metadata, Modality::Audio)
                .unwrap()
        );
        assert_eq!(
            spec.placeholder_token_id(&metadata).unwrap(),
            spec.placeholder_token_id_for(&metadata, Modality::Audio)
                .unwrap()
        );

        let replacements = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(128, 100)], &[13]),
                Modality::Audio,
            )
            .unwrap();
        assert_eq!(replacements[0].tokens, vec![151676; 13]);
        assert_eq!(
            spec.encoder_field_layouts_for(Modality::Audio)
                .encoder_input,
            FieldLayout::Batched
        );
        assert_eq!(
            spec.modality_limits(&metadata).unwrap(),
            HashMap::from([(Modality::Audio, 10)])
        );
    }

    #[test]
    fn asr_spec_builds_qwen_audio_processor() {
        use std::sync::Arc;

        use bytes::Bytes;

        use crate::{
            audio::DecodedAudio,
            types::{AudioClip, AudioSource},
        };

        let tokenizer = TestTokenizer::new(&[(AUDIO_PAD_TOKEN, 151676)]);
        let config = json!({"model_type": "qwen3_asr"});
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-ASR-1.7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).unwrap();

        let preprocessor_config = PreProcessorConfig::from_json(
            r#"{"feature_size": 16, "sampling_rate": 16000, "n_fft": 400, "hop_length": 160}"#,
        )
        .unwrap();
        let processor = spec
            .audio_processor(&config, &preprocessor_config)
            .expect("qwen3_asr spec must provide an audio processor");

        let clip = Arc::new(AudioClip::new(
            Bytes::from_static(b"audio"),
            DecodedAudio {
                samples: vec![0.0; 800],
                sample_rate: 16_000,
            },
            AudioSource::InlineBytes,
            "audio-hash".to_string(),
        ));
        let result = processor.preprocess(&[clip]).unwrap();
        assert_eq!(result.encoder_input.shape(), &[1, 16, 5]);
    }
}
