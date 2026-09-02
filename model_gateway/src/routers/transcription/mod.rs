//! Transcription model-family specs — the family seam for the OpenAI
//! `/v1/audio/transcriptions` endpoint.
//!
//! Lives at the shared `routers` level, not under any one backend: audio
//! transcription is a `RouterTrait` method, so a router implementation
//! resolves a family here and drives it through its own execution path.
//!
//! The router stays family-blind: it resolves a spec (fail-closed), lets the
//! spec validate capability limits and build the family's chat request,
//! executes that request like any other chat, and renders the format the
//! spec approved. Everything a family knows — its detection signals, its
//! language set, its prompt shape, its output post-processing, its
//! capability limits — lives on the spec, next to the spec's tests,
//! mirroring `crates/multimodal`'s per-model registry. Supporting another
//! family is a new spec module and one registry row, not another
//! model-conditional in the router.

// One module per family. Adding a transcription model is exactly: a module
// here implementing `TranscriptionModelSpec`, one row in `SPECS`, and the
// family's protocol-free knowledge in that family's `crates/multimodal`
// registry spec — no edit to the router or to anything generic below.
mod qwen3_asr;

use axum::{
    http::header,
    response::{IntoResponse, Response},
};
use openai_protocol::{
    chat::ChatCompletionRequest,
    transcription::{AudioFile, TranscriptionRequest},
};
use serde_json::json;

use crate::worker::WorkerRegistry;

/// One transcription-capable model family.
pub(crate) trait TranscriptionModelSpec: Send + Sync {
    /// Family name as rendered in user-facing errors.
    fn name(&self) -> &'static str;

    /// Whether this family serves `model_id` on this fleet. Detection is
    /// card- and label-driven and fail-closed: no match means the endpoint
    /// rejects the request rather than guessing.
    fn matches(&self, worker_registry: &WorkerRegistry, model_id: &str) -> bool;

    /// Validate the request against the family's capability limits and
    /// resolve the response format.
    fn response_format(
        &self,
        body: &TranscriptionRequest,
    ) -> Result<TranscriptionResponseFormat, Box<Response>>;

    /// Build the family's chat request for one audio file.
    fn build_chat_request(
        &self,
        body: &TranscriptionRequest,
        audio: &AudioFile,
    ) -> Result<ChatCompletionRequest, Box<Response>>;

    /// Post-process raw chat content into the transcription text.
    fn parse_output(&self, raw: &str) -> String;
}

/// Every supported family; first match wins. New families append here.
static SPECS: &[&dyn TranscriptionModelSpec] = &[&qwen3_asr::Qwen3AsrSpec];

/// Resolve the family serving `model_id`, or `None` when no spec matches.
pub(crate) fn resolve(
    worker_registry: &WorkerRegistry,
    model_id: &str,
) -> Option<&'static dyn TranscriptionModelSpec> {
    SPECS
        .iter()
        .copied()
        .find(|spec| spec.matches(worker_registry, model_id))
}

/// The supported family names, for the model-not-supported rejection.
pub(crate) fn supported_families() -> String {
    SPECS
        .iter()
        .map(|spec| spec.name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Wire format of the transcription response body.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TranscriptionResponseFormat {
    Json,
    Text,
}

/// Render the final transcription response in the approved format.
pub(crate) fn render(format: TranscriptionResponseFormat, text: String) -> Response {
    match format {
        TranscriptionResponseFormat::Json => axum::Json(json!({"text": text})).into_response(),
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
