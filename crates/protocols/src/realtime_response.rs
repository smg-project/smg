// OpenAI Realtime Conversation API types
// https://platform.openai.com/docs/api-reference/realtime
//
// Session configuration and audio types live in `realtime_session`.
// Event type constants live in `event_types`.
// This module covers the realtime response
// object, usage, errors, rate limits.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize};
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
    pub input: Option<Vec<RealtimeResponseInputItem>>,
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

/// An item of a `response.create`'s input, as the public API takes it
/// (`RealtimeResponseCreateParamsWithReferencesParam.input`): a conversation
/// item, or a reference to an item already in the session by its id. A
/// reference is taken here only, not as a new conversation item.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum RealtimeResponseInputItem {
    ItemReference(RealtimeItemReference),
    Item(RealtimeConversationItem),
}

/// The `type` picks the variant, so an item that is not a reference keeps
/// its own refusals, an unknown `type` among them.
impl<'de> Deserialize<'de> for RealtimeResponseInputItem {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        let item = serde_json::Value::deserialize(deserializer)?;
        if item.get("type").and_then(serde_json::Value::as_str) == Some("item_reference") {
            serde_json::from_value(item)
                .map(Self::ItemReference)
                .map_err(D::Error::custom)
        } else {
            serde_json::from_value(item)
                .map(Self::Item)
                .map_err(D::Error::custom)
        }
    }
}

/// A reference to an item already in the session (SDK
/// `ResponseItemReferenceParam`): `{"type": "item_reference", "id": ...}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeItemReference {
    #[serde(rename = "type")]
    pub r#type: RealtimeItemReferenceType,
    /// The id of an item already in the same session
    pub id: String,
}

/// The `type` of an item reference, whose one value is `item_reference`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeItemReferenceType {
    #[default]
    ItemReference,
}

/// Schema-level validation
fn validate_response_create_params(
    request: &RealtimeResponseCreateParams,
) -> Result<(), ValidationError> {
    // validate role→content-part constraints per the OpenAI spec; a
    // reference names an item the session already holds
    if let Some(items) = &request.input {
        for item in items {
            if let RealtimeResponseInputItem::Item(item) = item {
                validate_conversation_item(item)?;
            }
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The public API's `response.create` input may reference an item of the
    /// session by its id beside new items (`openapi.transformed.yml`'s
    /// example starts with one): it reads, passes the rules and is written
    /// back as sent. A new item keeps its own rules, and a reference without
    /// an id is refused.
    #[test]
    fn a_response_create_input_takes_an_item_reference() {
        let params = json!({
            "conversation": "none",
            "input": [
                {"type": "item_reference", "id": "item_12345"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Summarize the call so far."}]
                }
            ],
            "output_modalities": ["text"]
        });
        let request: RealtimeResponseCreateParams =
            serde_json::from_value(params.clone()).expect("a response.create with a reference");
        assert!(request.validate().is_ok(), "a reference passes the rules");
        assert!(matches!(
            request.input.as_deref(),
            Some([
                RealtimeResponseInputItem::ItemReference(_),
                RealtimeResponseInputItem::Item(_)
            ])
        ));
        assert_eq!(serde_json::to_value(&request).expect("serialize"), params);

        let refused: RealtimeResponseCreateParams = serde_json::from_value(json!({
            "input": [{
                "type": "message",
                "role": "system",
                "content": [{"type": "input_audio", "audio": "AAAA"}]
            }]
        }))
        .expect("parses");
        assert!(refused.validate().is_err(), "a new item keeps its rules");

        let no_id = serde_json::from_value::<RealtimeResponseCreateParams>(json!({
            "input": [{"type": "item_reference"}]
        }));
        assert!(no_id.is_err(), "a reference names its item");

        let unknown = serde_json::from_value::<RealtimeResponseCreateParams>(json!({
            "input": [{"type": "item_refrence", "id": "item_12345"}]
        }))
        .expect_err("an unknown type is refused");
        assert!(
            unknown
                .to_string()
                .contains("unknown variant `item_refrence`"),
            "an unknown type keeps the conversation item's refusal: {unknown}"
        );

        // the path a client takes: the params inside a response.create event
        let event: crate::realtime_events::ClientEvent =
            serde_json::from_value(json!({"type": "response.create", "response": params}))
                .expect("a response.create event with a reference");
        assert!(matches!(
            event,
            crate::realtime_events::ClientEvent::ResponseCreate { response: Some(ref r), .. }
                if matches!(
                    r.input.as_deref(),
                    Some([RealtimeResponseInputItem::ItemReference(_), _])
                )
        ));
    }
}
