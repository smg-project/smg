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

    // ---- Response ----

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
