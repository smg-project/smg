//! Qwen3-ASR endpoint adapter: transcription served as a continued chat.
//!
//! The family's own knowledge — identifiers, language set, prompt
//! sanitation, `<asr_text>` framing — lives in
//! `llm_multimodal::registry::qwen3_asr::transcription`, shared with the
//! audio-processing spec so the two layers cannot drift. This adapter only
//! maps that knowledge onto the HTTP endpoint: worker-label detection, the
//! chat request shape, capability gating, and error codes.

use axum::response::Response;
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use llm_multimodal::registry::qwen3_asr::transcription as family;
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::{ContentPart, InputAudio},
    transcription::{AudioFile, TranscriptionRequest},
};

use super::{audio_format, TranscriptionModelSpec, TranscriptionResponseFormat};
use crate::{routers::error, worker::WorkerRegistry};

/// Worker labels whose values may name the family (deployment-level
/// detection, complementing the model-id check).
const QWEN3_ASR_LABEL_KEYS: &[&str] = &[
    "model",
    "model_path",
    "model_type",
    "hf_model_type",
    "tokenizer",
    "tokenizer_path",
];

/// The Qwen3-ASR family adapter.
pub(super) struct Qwen3AsrSpec;

impl TranscriptionModelSpec for Qwen3AsrSpec {
    fn name(&self) -> &'static str {
        "Qwen3-ASR"
    }

    fn matches(&self, worker_registry: &WorkerRegistry, model_id: &str) -> bool {
        if family::is_qwen3_asr_identifier(model_id) {
            return true;
        }

        worker_registry.get_by_model(model_id).iter().any(|worker| {
            let metadata = worker.metadata();
            family::is_qwen3_asr_identifier(metadata.model_id())
                || metadata
                    .spec
                    .labels
                    .iter()
                    .any(|(key, value)| is_qwen3_asr_metadata_label(key, value))
        })
    }

    fn response_format(
        &self,
        body: &TranscriptionRequest,
    ) -> Result<TranscriptionResponseFormat, Box<Response>> {
        let format = parse_transcription_response_format(body.response_format.as_deref())?;
        if body.stream.unwrap_or(false) {
            return Err(Box::new(error::bad_request(
                "streaming_transcription_not_supported",
                "TokenSpeed Qwen3-ASR currently supports whole-file transcription only",
            )));
        }
        if body
            .timestamp_granularities
            .as_ref()
            .is_some_and(|values| !values.is_empty())
        {
            return Err(Box::new(error::bad_request(
                "transcription_timestamps_not_supported",
                "Qwen3-ASR timestamps require the forced-aligner model and are not supported",
            )));
        }
        Ok(format)
    }

    fn build_chat_request(
        &self,
        body: &TranscriptionRequest,
        audio: &AudioFile,
    ) -> Result<ChatCompletionRequest, Box<Response>> {
        let language =
            family::normalize_language(body.language.as_deref()).map_err(|unsupported| {
                Box::new(error::bad_request(
                    "unsupported_transcription_language",
                    format!("Qwen3-ASR does not support language '{}'", unsupported.0),
                ))
            })?;
        build_qwen3_asr_chat_request(body, audio, language.as_deref())
    }

    fn parse_output(&self, raw: &str) -> String {
        family::parse_transcript(raw)
    }
}

fn is_qwen3_asr_metadata_label(key: &str, value: &str) -> bool {
    QWEN3_ASR_LABEL_KEYS.contains(&key) && family::is_qwen3_asr_identifier(value)
}

fn build_qwen3_asr_chat_request(
    body: &TranscriptionRequest,
    audio: &AudioFile,
    language: Option<&str>,
) -> Result<ChatCompletionRequest, Box<Response>> {
    let mut messages = Vec::with_capacity(3);
    if let Some(prompt) = body.prompt.as_deref() {
        let prompt = family::sanitize_prompt(prompt.to_string()).map_err(|too_long| {
            Box::new(error::bad_request(
                "asr_prompt_too_long",
                format!(
                    "Qwen3-ASR prompt must not exceed {} bytes",
                    too_long.max_bytes
                ),
            ))
        })?;
        let prompt = prompt.trim();
        if !prompt.is_empty() {
            messages.push(ChatMessage::System {
                content: MessageContent::Text(prompt.to_string()),
                name: None,
            });
        }
    }
    messages.push(ChatMessage::User {
        content: MessageContent::Parts(vec![ContentPart::InputAudio {
            input_audio: InputAudio {
                data: BASE64_STANDARD.encode(&audio.bytes),
                format: audio_format(audio),
            },
        }]),
        name: None,
    });

    let continue_final_message = if let Some(language) = language {
        messages.push(ChatMessage::Assistant {
            content: Some(MessageContent::Text(format!(
                "language {language}{}",
                family::ASR_TEXT_TAG
            ))),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        });
        true
    } else {
        false
    };

    Ok(ChatCompletionRequest {
        messages,
        model: body.model.clone(),
        n: Some(1),
        stream: false,
        temperature: Some(body.temperature.unwrap_or(0.0)),
        continue_final_message,
        skip_special_tokens: true,
        separate_reasoning: false,
        stream_reasoning: false,
        ..Default::default()
    })
}

fn parse_transcription_response_format(
    format: Option<&str>,
) -> Result<TranscriptionResponseFormat, Box<Response>> {
    match format.unwrap_or("json").to_ascii_lowercase().as_str() {
        "json" => Ok(TranscriptionResponseFormat::Json),
        "text" => Ok(TranscriptionResponseFormat::Text),
        unsupported => Err(Box::new(error::bad_request(
            "unsupported_transcription_response_format",
            format!(
                "Qwen3-ASR does not provide timestamps required for response format '{unsupported}'"
            ),
        ))),
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use bytes::Bytes;

    use super::*;
    use crate::routers::grpc::transcription::render;

    fn transcription_request() -> TranscriptionRequest {
        TranscriptionRequest {
            model: "Qwen/Qwen3-ASR-1.7B".to_string(),
            ..Default::default()
        }
    }

    fn wav_file() -> AudioFile {
        AudioFile {
            bytes: Bytes::from_static(b"RIFFtest"),
            file_name: "sample.wav".to_string(),
            content_type: Some("audio/wav".to_string()),
        }
    }

    #[test]
    fn recognizes_qwen3_asr_metadata_labels() {
        assert!(is_qwen3_asr_metadata_label("model_type", "qwen3_asr"));
        assert!(is_qwen3_asr_metadata_label("hf_model_type", "qwen3-asr"));
        assert!(!is_qwen3_asr_metadata_label("unrelated", "qwen3_asr"));
    }

    #[test]
    fn builds_audio_chat_request_with_language_continuation() {
        let mut body = transcription_request();
        body.prompt = Some("domain vocabulary".to_string());
        body.temperature = Some(0.2);

        let chat = build_qwen3_asr_chat_request(&body, &wav_file(), Some("English")).unwrap();

        assert_eq!(chat.model, body.model);
        assert_eq!(chat.temperature, Some(0.2));
        assert!(chat.continue_final_message);
        assert_eq!(chat.messages.len(), 3);
        match &chat.messages[1] {
            ChatMessage::User {
                content: MessageContent::Parts(parts),
                ..
            } => match &parts[0] {
                ContentPart::InputAudio { input_audio } => {
                    assert_eq!(input_audio.data, "UklGRnRlc3Q=");
                    assert_eq!(input_audio.format, "wav");
                }
                other => panic!("expected audio content part, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
        match &chat.messages[2] {
            ChatMessage::Assistant {
                content: Some(MessageContent::Text(content)),
                ..
            } => assert_eq!(content, "language English<asr_text>"),
            other => panic!("expected assistant continuation, got {other:?}"),
        }
    }

    #[test]
    fn transcription_chat_defaults_to_greedy_decoding() {
        let chat =
            build_qwen3_asr_chat_request(&transcription_request(), &wav_file(), None).unwrap();

        assert_eq!(chat.temperature, Some(0.0));
        assert!(!chat.continue_final_message);
    }

    #[test]
    fn maps_family_violations_to_http_error_codes() {
        // Over-long prompt → asr_prompt_too_long.
        let mut body = transcription_request();
        body.prompt = Some("a".repeat(family::MAX_PROMPT_BYTES + 1));
        let error = build_qwen3_asr_chat_request(&body, &wav_file(), None).unwrap_err();
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(error.as_ref()),
            "asr_prompt_too_long"
        );

        // Unknown language → unsupported_transcription_language.
        let mut body = transcription_request();
        body.language = Some("xx".to_string());
        let error = Qwen3AsrSpec
            .build_chat_request(&body, &wav_file())
            .unwrap_err();
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(error.as_ref()),
            "unsupported_transcription_language"
        );
    }

    #[test]
    fn transcription_response_rejects_timestamp_formats() {
        assert_eq!(
            parse_transcription_response_format(Some("verbose_json"))
                .unwrap_err()
                .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            parse_transcription_response_format(Some("srt"))
                .unwrap_err()
                .status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            render(
                parse_transcription_response_format(Some("json")).unwrap(),
                "text".to_string(),
            )
            .status(),
            StatusCode::OK
        );
        assert_eq!(
            render(
                parse_transcription_response_format(Some("text")).unwrap(),
                "text".to_string(),
            )
            .status(),
            StatusCode::OK
        );
    }
}
