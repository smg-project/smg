use axum::http::StatusCode;
use bytes::Bytes;
use llm_multimodal::registry::{
    qwen3_asr::transcription::MAX_PROMPT_BYTES, transcription::FAMILIES,
};
use tracing::Level;

use super::*;
use crate::{
    middleware::ReportedRefusal,
    observability::logging::test_support::{test_subscriber, LinesAtWarnOrAbove},
};

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

fn family() -> &'static dyn TranscriptionFamily {
    FAMILIES[0]
}

#[test]
fn builds_audio_chat_request_with_language_continuation() {
    let mut body = transcription_request();
    body.prompt = Some("domain vocabulary".to_string());
    body.temperature = Some(0.2);
    body.language = Some("en".to_string());

    let chat = build_chat_request(&body, &wav_file(), family()).unwrap();

    assert_eq!(chat.model, body.model);
    assert_eq!(chat.temperature, Some(0.2));
    assert!(chat.continue_final_message);
    assert_eq!(chat.messages.len(), 3);
    match &chat.messages[0] {
        ChatMessage::System {
            content: MessageContent::Text(text),
            ..
        } => assert_eq!(text, "domain vocabulary"),
        other => panic!("expected system prompt, got {other:?}"),
    }
    match &chat.messages[1] {
        ChatMessage::User {
            content: MessageContent::Parts(parts),
            ..
        } => match &parts[0] {
            ContentPart::InputAudio { input_audio } => {
                assert_eq!(*input_audio.data, "UklGRnRlc3Q=");
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
fn transcription_chat_defaults_to_greedy_whole_file_decoding() {
    let chat = build_chat_request(&transcription_request(), &wav_file(), family()).unwrap();

    assert_eq!(chat.temperature, Some(0.0));
    assert!(!chat.continue_final_message);
    assert!(!chat.stream);
    assert_eq!(chat.n, Some(1));
    // No prompt and no language: just the audio user turn.
    assert_eq!(chat.messages.len(), 1);
}

#[test]
fn maps_family_validation_errors_to_gateway_codes() {
    let mut body = transcription_request();
    body.prompt = Some("a".repeat(MAX_PROMPT_BYTES + 1));
    let response = build_chat_request(&body, &wav_file(), family()).unwrap_err();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "asr_prompt_too_long"
    );

    let mut body = transcription_request();
    body.language = Some("klingon".to_string());
    let response = build_chat_request(&body, &wav_file(), family()).unwrap_err();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "unsupported_transcription_language"
    );
}

/// A language the family does not transcribe is the client's to fix: the stage writes one
/// WARN and marks the answer, and no ERROR.
#[test]
fn an_unsupported_language_is_one_warn_and_marked() {
    let mut body = transcription_request();
    body.language = Some("klingon".to_string());
    let lines = LinesAtWarnOrAbove::default();
    let _subscriber = test_subscriber(lines.clone());

    let response = build_chat_request(&body, &wav_file(), family()).unwrap_err();

    assert!(response.extensions().get::<ReportedRefusal>().is_some());
    assert_eq!(
        lines.lines(),
        [(
            Level::WARN,
            "unsupported transcription language".to_string()
        )]
    );
}
