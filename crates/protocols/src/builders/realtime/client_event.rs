//! Constructors for ClientEvent
//!
//! Associated functions on `ClientEvent` for constructing client-to-server
//! events programmatically. Useful when the gateway needs to inject events
//! (e.g., MCP tool call results) into the upstream connection.
//!
//! ```ignore
//! ClientEvent::session_update(Some("evt_1".into()), config)
//! ClientEvent::conversation_item_create(Some("evt_2".into()), None, item)
//! ClientEvent::response_create(Some("evt_3".into()), Some(params))
//! ```

use crate::{
    realtime_conversation::RealtimeConversationItem,
    realtime_events::{ClientEvent, SessionConfig},
    realtime_response::RealtimeResponseCreateParams,
};

impl ClientEvent {
    // ---- Session ----

    /// Build a `session.update` event.
    pub fn session_update(event_id: Option<String>, session: SessionConfig) -> Self {
        Self::SessionUpdate {
            event_id,
            session: Box::new(session),
        }
    }

    // ---- Conversation items ----

    /// Build a `conversation.item.create` event.
    pub fn conversation_item_create(
        event_id: Option<String>,
        previous_item_id: Option<String>,
        item: RealtimeConversationItem,
    ) -> Self {
        Self::ConversationItemCreate {
            event_id,
            previous_item_id,
            item,
        }
    }

    /// Build a `conversation.item.delete` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn conversation_item_delete(event_id: Option<String>, item_id: impl Into<String>) -> Self {
        Self::ConversationItemDelete {
            event_id,
            item_id: item_id.into(),
        }
    }

    /// Build a `conversation.item.retrieve` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn conversation_item_retrieve(
        event_id: Option<String>,
        item_id: impl Into<String>,
    ) -> Self {
        Self::ConversationItemRetrieve {
            event_id,
            item_id: item_id.into(),
        }
    }

    /// Build a `conversation.item.truncate` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn conversation_item_truncate(
        event_id: Option<String>,
        item_id: impl Into<String>,
        content_index: u32,
        audio_end_ms: u32,
    ) -> Self {
        Self::ConversationItemTruncate {
            event_id,
            item_id: item_id.into(),
            content_index,
            audio_end_ms,
        }
    }

    // ---- Input audio buffer ----

    /// Build an `input_audio_buffer.append` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_buffer_append(event_id: Option<String>, audio: impl Into<String>) -> Self {
        Self::InputAudioBufferAppend {
            event_id,
            audio: audio.into(),
        }
    }

    /// Build an `input_audio_buffer.clear` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_buffer_clear(event_id: Option<String>) -> Self {
        Self::InputAudioBufferClear { event_id }
    }

    /// Build an `input_audio_buffer.commit` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_buffer_commit(event_id: Option<String>) -> Self {
        Self::InputAudioBufferCommit { event_id }
    }

    // ---- Output audio buffer (WebRTC/SIP only) ----

    /// Build an `output_audio_buffer.clear` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn output_audio_buffer_clear(event_id: Option<String>) -> Self {
        Self::OutputAudioBufferClear { event_id }
    }

    // ---- Response ----

    /// Build a `response.cancel` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn response_cancel(event_id: Option<String>, response_id: Option<String>) -> Self {
        Self::ResponseCancel {
            event_id,
            response_id,
        }
    }

    /// Build a `response.create` event.
    pub fn response_create(
        event_id: Option<String>,
        response: Option<RealtimeResponseCreateParams>,
    ) -> Self {
        Self::ResponseCreate {
            event_id,
            response: response.map(Box::new),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime_session::{
        OutputModality, RealtimeSessionCreateRequest, RealtimeSessionType,
    };

    #[test]
    fn test_session_update() {
        let config = SessionConfig::Realtime(Box::new(RealtimeSessionCreateRequest {
            r#type: RealtimeSessionType::Realtime,
            output_modalities: Some(vec![OutputModality::Audio]),
            model: None,
            instructions: None,
            audio: None,
            include: None,
            tracing: None,
            tools: None,
            tool_choice: None,
            max_output_tokens: None,
            truncation: None,
            prompt: None,
        }));

        let event = ClientEvent::session_update(Some("evt_1".into()), config);
        assert_eq!(event.event_type(), "session.update");
    }

    #[test]
    fn test_response_create_empty() {
        let event = ClientEvent::response_create(None, None);
        assert_eq!(event.event_type(), "response.create");
    }
}
