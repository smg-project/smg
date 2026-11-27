use std::{collections::HashMap, sync::Arc};

use axum::http::StatusCode;
use bytes::Bytes;
use openai_protocol::model_card::ModelCard;
use smg_worker::identity::basic::Worker;

use super::*;
use crate::routers::grpc::backend_client::BasicWorkerBuilder;

fn audio(file_name: &str, content_type: Option<&str>) -> AudioFile {
    AudioFile {
        bytes: Bytes::from_static(b"RIFFtest"),
        file_name: file_name.to_string(),
        content_type: content_type.map(str::to_string),
    }
}

fn labeled_worker(url: &str, model: &str, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
    let labels: HashMap<String, String> = labels
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect();
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model))
            .labels(labels)
            .build(),
    )
}

#[test]
fn family_registry_resolves_qwen3_asr_by_identifier() {
    assert!(FAMILIES
        .iter()
        .any(|f| f.is_identifier("Qwen/Qwen3-ASR-1.7B")));
    assert!(supported_families().contains("Qwen3-ASR"));
}

#[test]
fn resolves_family_from_model_identity_labels_only() {
    let registry = WorkerRegistry::new();
    registry
        .register(labeled_worker(
            "http://asr:9000",
            "asr-prod",
            &[("model_path", "/models/qwen3_asr_1.7b")],
        ))
        .unwrap();
    registry
        .register(labeled_worker(
            "http://chat:9000",
            "chat-prod",
            &[("description", "distilled from qwen3-asr")],
        ))
        .unwrap();

    // A family-identifying model id needs no worker evidence.
    assert!(resolve_family(&registry, "Qwen/Qwen3-ASR-1.7B").is_some());
    // A neutral alias resolves through a model-identity label…
    assert!(resolve_family(&registry, "asr-prod").is_some());
    // …but a free-text label mentioning the family must not.
    assert!(resolve_family(&registry, "chat-prod").is_none());
    // No workers, no identifier match: rejected.
    assert!(resolve_family(&registry, "unknown-model").is_none());
}

#[test]
fn parse_response_format_accepts_json_text_rejects_timestamps() {
    assert!(matches!(
        parse_response_format(None).unwrap(),
        TranscriptionResponseFormat::Json
    ));
    assert!(matches!(
        parse_response_format(Some("text")).unwrap(),
        TranscriptionResponseFormat::Text
    ));
    assert_eq!(
        parse_response_format(Some("verbose_json"))
            .unwrap_err()
            .status(),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        parse_response_format(Some("srt")).unwrap_err().status(),
        StatusCode::BAD_REQUEST
    );
}

#[test]
fn audio_format_prefers_content_type_then_extension_then_wav() {
    assert_eq!(audio_format(&audio("a.bin", Some("audio/mpeg"))), "mp3");
    assert_eq!(audio_format(&audio("a.bin", Some("audio/x-wav"))), "wav");
    assert_eq!(audio_format(&audio("clip.MP3", None)), "mp3");
    assert_eq!(
        audio_format(&audio("noext", Some("application/octet-stream"))),
        "wav"
    );
}

#[test]
fn render_sets_json_or_plain_text() {
    assert_eq!(
        render(TranscriptionResponseFormat::Json, "hi".to_string()).status(),
        StatusCode::OK
    );
    assert_eq!(
        render(TranscriptionResponseFormat::Text, "hi".to_string()).status(),
        StatusCode::OK
    );
}
