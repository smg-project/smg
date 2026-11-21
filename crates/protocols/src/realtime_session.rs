// OpenAI Realtime Session API types
// https://platform.openai.com/docs/api-reference/realtime

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::{Validate, ValidationError};

use crate::{
    common::{Redacted, ResponsePrompt, ToolReference},
    validated::Normalizable,
};

// ============================================================================
// Session Configuration
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[validate(schema(function = "validate_session_create_request"))]
pub struct RealtimeSessionCreateRequest {
    #[serde(rename = "type")]
    pub r#type: RealtimeSessionType,
    pub audio: Option<RealtimeAudioConfig>,
    pub include: Option<Vec<RealtimeIncludeOption>>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<MaxOutputTokens>,
    pub model: Option<String>,
    #[serde(default = "audio")]
    pub output_modalities: Option<Vec<OutputModality>>,
    pub prompt: Option<ResponsePrompt>,
    pub tool_choice: Option<RealtimeToolChoiceConfig>,
    /// The session's tools, a list as the public API takes it: an empty one
    /// clears the tools.
    pub tools: Option<Vec<RealtimeToolsConfig>>,
    pub tracing: Option<RealtimeTracingConfig>,
    pub truncation: Option<RealtimeTruncation>,
}

impl Normalizable for RealtimeSessionCreateRequest {}

fn validate_session_create_request(
    req: &RealtimeSessionCreateRequest,
) -> Result<(), ValidationError> {
    let has_model = req.model.as_deref().is_some_and(|m| !m.trim().is_empty());
    if !has_model {
        return Err(ValidationError::new("model is required"));
    }
    Ok(())
}

// ============================================================================
// Session Object
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeSessionCreateResponse {
    pub client_secret: RealtimeSessionClientSecret,
    #[serde(rename = "type")]
    pub r#type: RealtimeSessionType,
    pub audio: Option<RealtimeAudioConfig>,
    pub include: Option<Vec<RealtimeIncludeOption>>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<MaxOutputTokens>,
    pub model: Option<String>,
    #[serde(default = "audio")]
    pub output_modalities: Option<Vec<OutputModality>>,
    pub prompt: Option<ResponsePrompt>,
    pub tool_choice: Option<RealtimeToolChoiceConfig>,
    pub tools: Option<Vec<RealtimeToolsConfig>>,
    pub tracing: Option<RealtimeTracingConfig>,
    pub truncation: Option<RealtimeTruncation>,
}

// ============================================================================
// Transcription Session Configuration
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[validate(schema(function = "validate_transcription_session_create_request"))]
pub struct RealtimeTranscriptionSessionCreateRequest {
    #[serde(rename = "type")]
    pub r#type: RealtimeTranscriptionSessionType,
    pub audio: Option<RealtimeTranscriptionSessionAudio>,
    pub include: Option<Vec<RealtimeIncludeOption>>,
    pub model: Option<String>,
    pub language: Option<String>,
    pub prompt: Option<String>,
}

impl Normalizable for RealtimeTranscriptionSessionCreateRequest {
    // Use default no-op implementation
}

fn validate_transcription_session_create_request(
    req: &RealtimeTranscriptionSessionCreateRequest,
) -> Result<(), ValidationError> {
    let has_model = req.model.as_deref().is_some_and(|m| !m.trim().is_empty());
    if !has_model {
        return Err(ValidationError::new("model is required"));
    }
    Ok(())
}

// ============================================================================
// Transcription Session Object from Create Response
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTranscriptionSessionCreateResponse {
    pub id: String,
    pub object: String,
    #[serde(rename = "type")]
    pub r#type: RealtimeTranscriptionSessionType,
    pub audio: Option<RealtimeTranscriptionSessionResponseAudio>,
    pub expires_at: Option<i64>,
    pub include: Option<Vec<RealtimeIncludeOption>>,
}

// ============================================================================
// Audio Formats
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RealtimeAudioFormats {
    #[serde(rename = "audio/pcm")]
    Pcm {
        /// Sample rate. Only 24000 is supported.
        #[serde(skip_serializing_if = "Option::is_none")]
        rate: Option<u32>,
    },
    #[serde(rename = "audio/pcmu")]
    Pcmu,
    #[serde(rename = "audio/pcma")]
    Pcma,
}

// ============================================================================
// Audio Transcription
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranscription {
    pub language: Option<String>,
    pub model: Option<String>,
    pub prompt: Option<String>,
}

// ============================================================================
// Noise Reduction
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoiseReductionType {
    NearField,
    FarField,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseReduction {
    #[serde(rename = "type")]
    pub r#type: NoiseReductionType,
}

// ============================================================================
// Turn Detection
// ============================================================================
/// Used only for `semantic_vad` mode. The eagerness of the model to respond.
/// `low` will wait longer for the user to continue speaking, `high` will respond more quickly.
/// `auto` is the default and is equivalent to `medium`. `low`, `medium`, and `high` have max timeouts of 8s, 4s, and 2s respectively.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SemanticVadEagerness {
    Low,
    Medium,
    High,
    #[default]
    Auto,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TurnDetection {
    #[serde(rename = "server_vad")]
    ServerVad {
        create_response: Option<bool>,
        idle_timeout_ms: Option<u32>,
        interrupt_response: Option<bool>,
        prefix_padding_ms: Option<u32>,
        silence_duration_ms: Option<u32>,
        threshold: Option<f64>,
    },
    #[serde(rename = "semantic_vad")]
    SemanticVad {
        create_response: Option<bool>,
        eagerness: Option<SemanticVadEagerness>,
        interrupt_response: Option<bool>,
    },
}

/// Turn detection for transcription sessions. Only `server_vad` is currently supported.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RealtimeTranscriptionSessionTurnDetection {
    #[serde(rename = "server_vad")]
    ServerVad {
        prefix_padding_ms: Option<u32>,
        silence_duration_ms: Option<u32>,
        threshold: Option<f64>,
    },
}

// ============================================================================
// Voice
// ============================================================================

/// Built-in voice name (e.g. "alloy", "ash") or custom voice reference.
///
/// Variant order matters for `#[serde(untagged)]`: a bare JSON string (e.g. `"alloy"`)
/// always deserializes as `BuiltIn`. A JSON object `{"id": "..."}` fails `BuiltIn`
/// and falls through to `Custom`. The two forms are structurally distinct (string vs
/// object) per the OpenAI spec, so there is no ambiguity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Voice {
    VoiceIDsShared(String),
    Custom { id: String },
}

// ============================================================================
// Output Modality
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputModality {
    Text,
    Audio,
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "must return Option to match serde default field type"
)]
fn audio() -> Option<Vec<OutputModality>> {
    Some(vec![OutputModality::Audio])
}

// ============================================================================
// Tracing
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    pub group_id: Option<String>,
    pub metadata: Option<Value>,
    pub workflow_name: Option<String>,
}

/// The tracing mode. Always `"auto"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TracingMode {
    #[serde(rename = "auto")]
    Auto,
}

/// Either the string `"auto"` or a granular tracing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RealtimeTracingConfig {
    Mode(TracingMode),
    Config(TracingConfig),
}

// ============================================================================
// Connector ID
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[expect(
    clippy::enum_variant_names,
    reason = "variant names match OpenAI Realtime API spec"
)]
pub enum ConnectorId {
    ConnectorDropbox,
    ConnectorGmail,
    ConnectorGooglecalendar,
    ConnectorGoogledrive,
    ConnectorMicrosoftteams,
    ConnectorOutlookcalendar,
    ConnectorOutlookemail,
    ConnectorSharepoint,
}

// ============================================================================
// Tools
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RealtimeToolsConfig {
    #[serde(rename = "function")]
    RealtimeFunctionTool {
        description: Option<String>,
        name: Option<String>,
        parameters: Option<Value>,
    },
    #[serde(rename = "mcp")]
    McpTool {
        server_label: String,
        allowed_tools: Option<McpAllowedTools>,
        authorization: Option<Redacted>,
        connector_id: Option<ConnectorId>,
        headers: Option<HashMap<String, Redacted>>,
        require_approval: Option<McpToolApproval>,
        server_description: Option<String>,
        server_url: Option<String>,
    },
}

// ============================================================================
// MCP Tool Filter
// ============================================================================

/// List of allowed tool names or a filter object.
///
/// Variant order matters for `#[serde(untagged)]`: serde tries `List` first
/// (JSON array). A JSON object falls through to `Filter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpAllowedTools {
    List(Vec<String>),
    Filter(McpToolFilter),
}

/// A filter object to specify which tools are allowed.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolFilter {
    pub read_only: Option<bool>,
    pub tool_names: Option<Vec<String>>,
}

/// Approval policy for MCP tools: a filter object or `"always"`/`"never"`.
///
/// Variant order matters for `#[serde(untagged)]`: serde tries `Setting` first
/// (plain string). A JSON object falls through to `Filter`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum McpToolApproval {
    Setting(McpToolApprovalSetting),
    Filter(McpToolApprovalFilter),
}

/// Single approval policy for all tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpToolApprovalSetting {
    Always,
    Never,
}

/// Granular approval filter specifying which tools always/never require approval.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolApprovalFilter {
    pub always: Option<McpToolFilter>,
    pub never: Option<McpToolFilter>,
}

// ============================================================================
// Tool Choice
// ============================================================================

/// `"none"`, `"auto"`, `"required"`, or a specific function/MCP tool reference.
///
/// Variant order matters for `#[serde(untagged)]`: serde tries `Options` first
/// (plain string). A JSON object fails and falls through to `Reference`.
/// Reuses [`ToolReference`] from `common` for the tagged object forms.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RealtimeToolChoiceConfig {
    Options(ToolChoiceOptions),
    Reference(ToolReference),
}

/// Controls which (if any) tool is called by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceOptions {
    None,
    Auto,
    Required,
}

// ============================================================================
// Max Output Tokens
// ============================================================================

/// Integer token limit (1–4096) or `"inf"` for the maximum available tokens.
/// Defaults to `"inf"`.
///
/// Variant order matters for `#[serde(untagged)]`: serde tries `Integer` first.
/// A JSON number succeeds immediately; the string `"inf"` fails `Integer` and
/// falls through to `Inf`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MaxOutputTokens {
    /// An integer between 1 and 4096.
    Integer(u32),
    Inf(InfMarker),
}

impl Default for MaxOutputTokens {
    fn default() -> Self {
        Self::Inf(InfMarker::Inf)
    }
}

/// The literal string `"inf"`. Used by [`MaxOutputTokens::Inf`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InfMarker {
    #[serde(rename = "inf")]
    Inf,
}

// ============================================================================
// Truncation
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationTokenLimits {
    pub post_instructions: Option<u32>,
}

/// The retention ratio truncation type. Always `"retention_ratio"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionRatioTruncationType {
    #[serde(rename = "retention_ratio")]
    RetentionRatio,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRatioTruncation {
    pub retention_ratio: f64,
    #[serde(rename = "type")]
    pub r#type: RetentionRatioTruncationType,
    pub token_limits: Option<TruncationTokenLimits>,
}

/// The truncation mode.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TruncationMode {
    #[default]
    Auto,
    Disabled,
}

/// `"auto"`, `"disabled"`, or a retention ratio configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RealtimeTruncation {
    Mode(TruncationMode),
    RetentionRatio(RetentionRatioTruncation),
}

// ============================================================================
// Client Secret
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[validate(schema(function = "validate_client_secret_create_request"))]
pub struct RealtimeClientSecretCreateRequest {
    /// When the client secret expires; the public API's default is 600
    /// seconds after it is created.
    pub expires_after: Option<RealtimeClientSecretExpiresAfter>,
    pub session: RealtimeSessionCreateRequest,
}

/// A client secret's lifetime: `seconds` after its `anchor`, the secret's
/// creation, as the public API takes it. The allowed range is the
/// upstream's to enforce.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeClientSecretExpiresAfter {
    pub anchor: Option<RealtimeClientSecretExpiresAfterAnchor>,
    pub seconds: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RealtimeClientSecretExpiresAfterAnchor {
    #[serde(rename = "created_at")]
    CreatedAt,
}

impl Normalizable for RealtimeClientSecretCreateRequest {}

fn validate_client_secret_create_request(
    req: &RealtimeClientSecretCreateRequest,
) -> Result<(), ValidationError> {
    let has_model = req
        .session
        .model
        .as_deref()
        .is_some_and(|m| !m.trim().is_empty());
    if !has_model {
        return Err(ValidationError::new("session.model is required"));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeSessionClientSecret {
    pub expires_at: i64,
    pub value: Redacted,
}

// ============================================================================
// Audio Configuration
// ============================================================================

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeAudioConfigInput {
    pub format: Option<RealtimeAudioFormats>,
    pub noise_reduction: Option<NoiseReduction>,
    pub transcription: Option<AudioTranscription>,
    /// Voice activity detection. An explicit `null` turns it off, where an
    /// absent key leaves the public API's server VAD on, so the two are kept
    /// apart (`Some(None)` and `None`) and each is written back as it came.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "serde_with::rust::double_option"
    )]
    pub turn_detection: Option<Option<TurnDetection>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeAudioConfigOutput {
    pub format: Option<RealtimeAudioFormats>,
    pub speed: Option<f64>,
    pub voice: Option<Voice>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeAudioConfig {
    pub input: Option<RealtimeAudioConfigInput>,
    pub output: Option<RealtimeAudioConfigOutput>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTranscriptionSessionAudio {
    pub input: Option<RealtimeAudioConfigInput>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTranscriptionSessionResponseAudio {
    pub input: Option<RealtimeTranscriptionSessionResponseAudioConfigInput>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeTranscriptionSessionResponseAudioConfigInput {
    pub format: Option<RealtimeAudioFormats>,
    pub noise_reduction: Option<NoiseReduction>,
    pub transcription: Option<AudioTranscription>,
    pub turn_detection: Option<RealtimeTranscriptionSessionTurnDetection>,
}

// ============================================================================
// Include Options
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealtimeIncludeOption {
    #[serde(rename = "item.input_audio_transcription.logprobs")]
    InputAudioTranscriptionLogprobs,
}

// ============================================================================
// Session Type
// ============================================================================

/// The type of session. Always `"realtime"` for the Realtime API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealtimeSessionType {
    #[serde(rename = "realtime")]
    Realtime,
}

// ============================================================================
// TranscriptionSession Type
// ============================================================================

/// The type of session. Always `"transcription"` for the Realtime API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealtimeTranscriptionSessionType {
    #[serde(rename = "transcription")]
    Transcription,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A client secret request keeps what the public API takes and is
    /// written back for the upstream as sent: the session's tools as a list,
    /// an empty one to clear them; a `null` turn_detection, which turns VAD
    /// off where an absent one leaves server VAD on; and the secret's
    /// `expires_after`. A single tool object is refused, as the API refuses
    /// it.
    #[test]
    fn a_client_secret_request_keeps_its_tools_turn_detection_and_expiry() {
        let tools = json!([
            {
                "type": "function",
                "name": "get_weather",
                "description": "The weather in a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            },
            {"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.deepwiki.com/mcp"}
        ]);
        let request: RealtimeClientSecretCreateRequest = serde_json::from_value(json!({
            "expires_after": {"anchor": "created_at", "seconds": 3600},
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "audio": {"input": {"turn_detection": null}},
                "tools": tools
            }
        }))
        .unwrap();
        let written = serde_json::to_value(&request).unwrap();
        assert_eq!(written["session"]["tools"], tools);
        assert_eq!(
            written["session"]["audio"]["input"],
            json!({"turn_detection": null})
        );
        assert_eq!(
            written["expires_after"],
            json!({"anchor": "created_at", "seconds": 3600})
        );

        let cleared: RealtimeClientSecretCreateRequest = serde_json::from_value(json!({
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "audio": {"input": {"turn_detection": {"type": "semantic_vad"}}},
                "tools": []
            }
        }))
        .unwrap();
        let written = serde_json::to_value(&cleared).unwrap();
        assert_eq!(written["session"]["tools"], json!([]));
        assert_eq!(
            written["session"]["audio"]["input"]["turn_detection"]["type"],
            "semantic_vad"
        );
        assert!(written.get("expires_after").is_none(), "{written}");

        let single = serde_json::from_value::<RealtimeClientSecretCreateRequest>(json!({
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "tools": {"type": "function", "name": "get_weather"}
            }
        }));
        assert!(single.is_err(), "a single tool object is not a list");
    }

    /// A transcription session shares the input audio config, so its `null`
    /// turn_detection is kept too, and an absent one stays absent.
    #[test]
    fn a_transcription_session_keeps_a_null_turn_detection() {
        let request: RealtimeTranscriptionSessionCreateRequest = serde_json::from_value(json!({
            "type": "transcription",
            "model": "gpt-4o-transcribe",
            "audio": {"input": {"turn_detection": null}}
        }))
        .unwrap();
        let written = serde_json::to_value(&request).unwrap();
        assert_eq!(written["audio"]["input"], json!({"turn_detection": null}));

        let absent: RealtimeTranscriptionSessionCreateRequest = serde_json::from_value(json!({
            "type": "transcription",
            "model": "gpt-4o-transcribe",
            "audio": {"input": {}}
        }))
        .unwrap();
        let written = serde_json::to_value(&absent).unwrap();
        assert_eq!(written["audio"]["input"], json!({}));
    }
}
