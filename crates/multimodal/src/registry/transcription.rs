//! Transcription family contract, consumed by the gateway's transcription
//! endpoint pipeline.
//!
//! Protocol-free on purpose: a family module (e.g. [`super::qwen3_asr`]) owns
//! what is true of its model — identifiers, the language set, prompt
//! sanitation, prefill convention, output framing, capability limits — while
//! how that surfaces on an HTTP endpoint (request shapes, error codes, the
//! chat pipeline) stays the gateway's business. Errors here are plain data
//! for the caller to render.
//!
//! A family is resolved via [`FAMILIES`]; the gateway's generic transcription
//! preparation stage reads a family's data to shape the request and its
//! output parser, so no per-model code lives in the router.

use super::qwen3_asr;

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
pub static FAMILIES: &[&dyn TranscriptionFamily] = &[&qwen3_asr::transcription::Qwen3Asr];

/// A caller prompt exceeded the family's prompt byte cap.
#[derive(Debug, PartialEq, Eq)]
pub struct PromptTooLong {
    pub max_bytes: usize,
}

/// The requested language is not in the family's supported set.
#[derive(Debug, PartialEq, Eq)]
pub struct UnsupportedLanguage(pub String);
