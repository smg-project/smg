//! Audio transcription API protocol definitions.
//!
//! This module defines the request type for the OpenAI-compatible
//! `/v1/audio/transcriptions` endpoint. The wire format is
//! `multipart/form-data` (an audio file plus text form fields), so the
//! struct below carries only the non-file fields; the file bytes travel
//! through the router as a separate payload.

use serde::{Deserialize, Serialize};

use super::common::GenerationRequest;

/// Transcription request - compatible with OpenAI's /v1/audio/transcriptions API.
///
/// The audio file itself is carried out-of-band because the endpoint uses
/// multipart/form-data, not JSON.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize, schemars::JsonSchema)]
pub struct TranscriptionRequest {
    /// ID of the model to use (e.g. "whisper-large-v3").
    pub model: String,

    /// Optional ISO-639-1 language hint for the input audio.
    pub language: Option<String>,

    /// Optional text prompt to guide the model's style / preserve continuity.
    pub prompt: Option<String>,

    /// Response format: `json` (default), `text`, `srt`, `verbose_json`, `vtt`,
    /// `diarized_json`.
    pub response_format: Option<String>,

    /// Sampling temperature (0..=1).
    pub temperature: Option<f32>,

    /// Timestamp granularities for verbose_json: `word`, `segment`.
    pub timestamp_granularities: Option<Vec<String>>,

    /// If true, stream partial transcription events as SSE.
    pub stream: Option<bool>,

    /// The form's other text parts, in order and under the names they came
    /// with: the public API's `include[]`, `chunking_strategy` (or the SDK's
    /// `chunking_strategy[type]` and its other bracketed keys),
    /// `known_speaker_names[]` and the fields the gateway does not model,
    /// forwarded to the worker as sent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub other: Vec<(String, String)>,
}

impl GenerationRequest for TranscriptionRequest {
    fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        // Audio bytes are not visible here; use the optional prompt as a
        // rough cache-aware routing hint when present.
        self.prompt.clone().unwrap_or_default()
    }
}

/// Binary audio payload for `/v1/audio/transcriptions`.
///
/// The transcription endpoint uses multipart/form-data, so the file bytes
/// travel alongside the JSON-like `TranscriptionRequest` rather than inside it.
#[derive(Debug, Clone)]
pub struct AudioFile {
    /// Raw audio bytes (wav/mp3/m4a/etc.).
    pub bytes: bytes::Bytes,
    /// Original filename from the multipart part. Forwarded verbatim to the worker.
    pub file_name: String,
    /// Original content-type of the audio part (e.g. `audio/wav`), if the client supplied one.
    pub content_type: Option<String>,
}
