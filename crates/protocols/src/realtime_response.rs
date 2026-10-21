// OpenAI Realtime Conversation API types
// https://platform.openai.com/docs/api-reference/realtime
//
// Session configuration and audio types live in `realtime_session`.
// Event type constants live in `event_types`.
// This module covers the realtime response
// object, usage, errors, rate limits.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

use crate::{
    builders::RealtimeResponseBuilder,
    common::{invalid, ResponsePrompt},
    realtime_conversation::{ConversationItemRole, RealtimeContentPart, RealtimeConversationItem},
    realtime_session::{
        MaxOutputTokens, OutputModality, RealtimeAudioFormats, RealtimeToolChoiceConfig,
        RealtimeToolsConfig, Voice,
    },
};

// ============================================================================
// Realtime Response
// ============================================================================

/// A response object in the Realtime API.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeResponse {
    pub id: Option<String>,
    pub audio: Option<RealtimeResponseCreateAudioOutput>,
    pub conversation_id: Option<String>,
    pub max_output_tokens: Option<MaxOutputTokens>,
    pub metadata: Option<HashMap<String, String>>,
    pub object: Option<RealtimeResponseObject>,
    pub output: Option<Vec<RealtimeConversationItem>>,
    pub output_modalities: Option<Vec<OutputModality>>,
    pub status: Option<ResponseStatus>,
    pub status_details: Option<RealtimeResponseStatus>,
    pub usage: Option<RealtimeResponseUsage>,
}

// ============================================================================
// Realtime Response Create Params
// ============================================================================

/// Parameters for creating a realtime response.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[validate(schema(function = "validate_response_create_params"))]
pub struct RealtimeResponseCreateParams {
    pub audio: Option<RealtimeResponseCreateAudioOutput>,
    pub conversation: Option<ResponseConversation>,
    pub input: Option<Vec<RealtimeConversationItem>>,
    pub instructions: Option<String>,
    pub max_output_tokens: Option<MaxOutputTokens>,
    pub metadata: Option<HashMap<String, String>>,
    pub output_modalities: Option<Vec<OutputModality>>,
    pub prompt: Option<ResponsePrompt>,
    pub tool_choice: Option<RealtimeToolChoiceConfig>,
    pub tools: Option<Vec<RealtimeToolsConfig>>,
}

// ============================================================================
// Response Status
// ============================================================================

/// Object type for realtime responses. Always `"realtime.response"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RealtimeResponseObject {
    #[serde(rename = "realtime.response")]
    RealtimeResponse,
}

/// Status of a realtime response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Completed,
    Cancelled,
    Failed,
    Incomplete,
    InProgress,
}

/// The type within status details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StatusDetailsType {
    Completed,
    Cancelled,
    Failed,
    Incomplete,
}

/// Reason the response did not complete.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StatusDetailsReason {
    TurnDetected,
    ClientCancelled,
    MaxOutputTokens,
    ContentFilter,
}

/// Error that caused the response to fail.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseStatusError {
    pub code: Option<String>,
    #[serde(rename = "type")]
    pub r#type: Option<String>,
}

/// Additional details about the response status.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeResponseStatus {
    pub error: Option<ResponseStatusError>,
    pub reason: Option<StatusDetailsReason>,
    #[serde(rename = "type")]
    pub r#type: Option<StatusDetailsType>,
}

// ============================================================================
// Response Audio Configuration
// ============================================================================

/// Audio output configuration for a response.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseAudioOutputConfig {
    pub format: Option<RealtimeAudioFormats>,
    pub voice: Option<Voice>,
}

/// Audio configuration for a response (output only).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeResponseCreateAudioOutput {
    pub output: Option<ResponseAudioOutputConfig>,
}

// ============================================================================
// Usage
// ============================================================================

/// Breakdown of cached token usage by modality.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedTokensDetails {
    pub audio_tokens: Option<u64>,
    pub image_tokens: Option<u64>,
    pub text_tokens: Option<u64>,
}

/// Input token usage details.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeResponseUsageInputTokenDetails {
    pub audio_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub cached_tokens_details: Option<CachedTokensDetails>,
    pub image_tokens: Option<u64>,
    pub text_tokens: Option<u64>,
}

/// Output token usage details.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeResponseUsageOutputTokenDetails {
    pub audio_tokens: Option<u64>,
    pub text_tokens: Option<u64>,
}

/// Token usage for a realtime response.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeResponseUsage {
    pub input_token_details: Option<RealtimeResponseUsageInputTokenDetails>,
    pub input_tokens: Option<u64>,
    pub output_token_details: Option<RealtimeResponseUsageOutputTokenDetails>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

// ============================================================================
// Response Conversation
// ============================================================================

/// `"auto"`, `"none"`, or a conversation ID string.
///
/// Variant order matters for `#[serde(untagged)]`: serde tries `Mode` first.
/// `"auto"` and `"none"` match `Mode`; any other string falls through to `Id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseConversation {
    Mode(ResponseConversationMode),
    Id(String),
}

/// Controls which conversation the response is added to.
/// `auto` is the default.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResponseConversationMode {
    #[default]
    Auto,
    None,
}

/// Schema-level validation
fn validate_response_create_params(
    request: &RealtimeResponseCreateParams,
) -> Result<(), ValidationError> {
    // validate role→content-part constraints per the OpenAI spec
    if let Some(items) = &request.input {
        for item in items {
            validate_conversation_item(item)?;
        }
    }
    Ok(())
}

/// Validates role→content-part constraints per the OpenAI spec.
///
/// - `system`    → `input_text` only
/// - `user`      → `input_text`, `input_audio`, `input_image`
/// - `assistant` → `output_text`, `output_audio`
///
/// Non-`Message` variants are always valid.
fn validate_conversation_item(item: &RealtimeConversationItem) -> Result<(), ValidationError> {
    let (role, content) = match item {
        RealtimeConversationItem::Message { role, content, .. } => (role, content),
        _ => return Ok(()),
    };

    for (i, part) in content.iter().enumerate() {
        let allowed = match role {
            ConversationItemRole::System => {
                matches!(part, RealtimeContentPart::InputText { .. })
            }
            ConversationItemRole::User => matches!(
                part,
                RealtimeContentPart::InputText { .. }
                    | RealtimeContentPart::InputAudio { .. }
                    | RealtimeContentPart::InputImage { .. }
            ),
            ConversationItemRole::Assistant => matches!(
                part,
                RealtimeContentPart::OutputText { .. } | RealtimeContentPart::OutputAudio { .. }
            ),
        };

        if !allowed {
            return Err(invalid(
                "invalid_content_part",
                format!(
                    "content[{}]: {:?} role does not allow \"{}\" content parts",
                    i,
                    role,
                    part.type_name()
                ),
            ));
        }
    }

    Ok(())
}

impl RealtimeResponse {
    /// Create a builder for constructing a `RealtimeResponse`.
    pub fn builder(id: impl Into<String>) -> RealtimeResponseBuilder {
        RealtimeResponseBuilder::new(id)
    }
}
