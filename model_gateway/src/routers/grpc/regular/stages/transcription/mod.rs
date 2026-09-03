//! Transcription endpoint pipeline stages.
//!
//! `/v1/audio/transcriptions` served as a first-class pipeline endpoint: the
//! preparation stage turns `(TranscriptionRequest, AudioFile)` into a
//! chat-shaped backend request *inside* the pipeline (using the resolved
//! model family's prompt convention), request building reuses the chat build
//! path, and response processing extracts plain text — no tool/reasoning
//! parsing. All model-specific knowledge lives in the family
//! (`llm_multimodal::registry::transcription`); these stages are
//! model-agnostic.

mod preparation;
mod request_building;
mod response_processing;

use axum::{
    http::header,
    response::{IntoResponse, Response},
};
use llm_multimodal::registry::transcription::{TranscriptionFamily, FAMILIES};
use openai_protocol::transcription::AudioFile;
pub(crate) use preparation::TranscriptionPreparationStage;
pub(crate) use request_building::TranscriptionRequestBuildingStage;
pub(crate) use response_processing::TranscriptionResponseProcessingStage;
use serde_json::json;

use crate::{
    routers::{error, grpc::spec::TranscriptionResponseFormat},
    worker::WorkerRegistry,
};

/// Label keys that carry model identity. Family detection through worker
/// labels consults only these, so a family name mentioned in a free-text
/// label (a description, lineage note, …) cannot flip a deployment into a
/// transcription family.
const MODEL_IDENTITY_LABEL_KEYS: &[&str] = &[
    "model",
    "model_path",
    "model_type",
    "hf_model_type",
    "tokenizer",
    "tokenizer_path",
];

/// Resolve the transcription family serving `model_id`, or `None` when no
/// family matches (the endpoint then rejects the request). Detection is the
/// family's own identifier check against the model id and — for deployments
/// serving under a neutral alias — the workers' model ids and model-identity
/// label values.
pub(crate) fn resolve_family(
    worker_registry: &WorkerRegistry,
    model_id: &str,
) -> Option<&'static dyn TranscriptionFamily> {
    FAMILIES.iter().copied().find(|family| {
        if family.is_identifier(model_id) {
            return true;
        }
        worker_registry.get_by_model(model_id).iter().any(|worker| {
            let metadata = worker.metadata();
            family.is_identifier(metadata.model_id())
                || metadata.spec.labels.iter().any(|(key, value)| {
                    MODEL_IDENTITY_LABEL_KEYS.contains(&key.as_str()) && family.is_identifier(value)
                })
        })
    })
}

/// The supported family names, for the model-not-supported rejection.
pub(crate) fn supported_families() -> String {
    FAMILIES
        .iter()
        .map(|family| family.name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Parse the requested response format, rejecting the timestamp-bearing
/// formats no chat-based transcription family produces.
pub(crate) fn parse_response_format(
    format: Option<&str>,
) -> Result<TranscriptionResponseFormat, Box<Response>> {
    match format.unwrap_or("json").to_ascii_lowercase().as_str() {
        "json" => Ok(TranscriptionResponseFormat::Json),
        "text" => Ok(TranscriptionResponseFormat::Text),
        unsupported => Err(Box::new(error::bad_request(
            "unsupported_transcription_response_format",
            format!("response format '{unsupported}' requires timestamps, which are not supported"),
        ))),
    }
}

/// Render the final transcript in the chosen wire format.
pub(crate) fn render(format: TranscriptionResponseFormat, text: String) -> Response {
    match format {
        TranscriptionResponseFormat::Json => axum::Json(json!({ "text": text })).into_response(),
        TranscriptionResponseFormat::Text => {
            ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response()
        }
    }
}

/// Uploaded-audio wire format: the content type when it names one, else the
/// file extension, else wav.
pub(crate) fn audio_format(audio: &AudioFile) -> String {
    let content_type = audio.content_type.as_deref().unwrap_or_default();
    if content_type.eq_ignore_ascii_case("audio/mpeg")
        || content_type.eq_ignore_ascii_case("audio/mp3")
    {
        return "mp3".to_string();
    }
    if content_type.eq_ignore_ascii_case("audio/wav")
        || content_type.eq_ignore_ascii_case("audio/wave")
        || content_type.eq_ignore_ascii_case("audio/x-wav")
    {
        return "wav".to_string();
    }
    audio
        .file_name
        .rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase())
        .filter(|extension| !extension.is_empty())
        .unwrap_or_else(|| "wav".to_string())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use axum::http::StatusCode;
    use bytes::Bytes;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, ModelCard, Worker};

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
}
