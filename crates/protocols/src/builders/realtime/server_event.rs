//! Constructors and streaming context builders for ServerEvent
//!
//! # One-shot events
//!
//! Simple events are constructed via associated functions on `ServerEvent`:
//!
//! ```ignore
//! ServerEvent::session_created("evt_1", config)
//! ServerEvent::error_event("evt_2", error)
//! ```
//!
//! # Streaming events (hierarchical context builders)
//!
//! Response-streaming events share stable fields (`response_id`, `item_id`,
//! `output_index`, `content_index`). Context builders capture this shared
//! state and can be reused across many events — only `event_id` varies per call:
//!
//! ```ignore
//! // One-shot chaining:
//! ResponseEventBuilder::new("resp_1")
//!     .for_item("item_1", 0)
//!     .for_content(0)
//!     .output_text_delta("evt_1", "Hello")
//!
//! // Streaming hot path — reuse the context, no rebuild:
//! let ctx = ResponseEventBuilder::new("resp_1")
//!     .for_item("item_1", 0)
//!     .for_content(0);
//! for chunk in chunks {
//!     send(ctx.output_text_delta(next_event_id(), chunk));
//! }
//! ```

use crate::{
    realtime_conversation::RealtimeConversationItem,
    realtime_events::{RealtimeError, ServerEvent, SessionConfig},
};

// ============================================================================
// ServerEvent associated constructors (one-shot events)
// ============================================================================

impl ServerEvent {
    // ---- Session events ----

    /// Build a `session.created` event.
    pub fn session_created(event_id: impl Into<String>, session: SessionConfig) -> Self {
        Self::SessionCreated {
            event_id: event_id.into(),
            session: Box::new(session),
        }
    }

    // ---- Error ----

    /// Build an `error` event.
    pub fn error_event(event_id: impl Into<String>, error: RealtimeError) -> Self {
        Self::Error {
            event_id: event_id.into(),
            error,
        }
    }
}

// ============================================================================
// Level 2: ResponseEventBuilder (response_id)
// ============================================================================

/// Reusable context for response-scoped server events.
///
/// Holds `response_id`. Terminal methods take `event_id` per call, so the
/// builder can be reused across many events in a streaming loop.
/// Call `.for_item()` to descend into item-scoped events.
#[derive(Clone, Debug)]
pub struct ResponseEventBuilder {
    response_id: String,
}

impl ResponseEventBuilder {
    /// Create a new response-scoped context.
    pub fn new(response_id: impl Into<String>) -> Self {
        Self {
            response_id: response_id.into(),
        }
    }

    // ---- Transition ----

    /// Descend into item-scoped events.
    pub fn for_item(&self, item_id: impl Into<String>, output_index: u32) -> ItemEventBuilder {
        ItemEventBuilder {
            response_id: self.response_id.clone(),
            item_id: item_id.into(),
            output_index,
        }
    }

    // ---- Output audio buffer events (WebRTC/SIP only) ----

    /// Build an `output_audio_buffer.started` event.
    pub fn output_audio_buffer_started(&self, event_id: impl Into<String>) -> ServerEvent {
        ServerEvent::OutputAudioBufferStarted {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
        }
    }
}

// ============================================================================
// Level 3: ItemEventBuilder (response_id + item_id + output_index)
// ============================================================================

/// Reusable context for item-scoped server events.
///
/// Holds `response_id`, `item_id`, and `output_index`. Terminal methods take
/// `event_id` per call. Call `.for_content()` to descend into content-scoped events.
#[derive(Clone, Debug)]
pub struct ItemEventBuilder {
    response_id: String,
    item_id: String,
    output_index: u32,
}

impl ItemEventBuilder {
    /// Create a new item-scoped context directly.
    pub fn new(
        response_id: impl Into<String>,
        item_id: impl Into<String>,
        output_index: u32,
    ) -> Self {
        Self {
            response_id: response_id.into(),
            item_id: item_id.into(),
            output_index,
        }
    }

    // ---- Transition ----

    /// Descend into content-scoped events.
    pub fn for_content(&self, content_index: u32) -> ContentEventBuilder {
        ContentEventBuilder {
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            content_index,
        }
    }

    // ---- Response output item events ----

    /// Build a `response.output_item.done` event.
    pub fn output_item_done(
        &self,
        event_id: impl Into<String>,
        item: RealtimeConversationItem,
    ) -> ServerEvent {
        ServerEvent::ResponseOutputItemDone {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            output_index: self.output_index,
            item,
        }
    }
}

// ============================================================================
// Level 4: ContentEventBuilder
//   (response_id + item_id + output_index + content_index)
// ============================================================================

/// Reusable context for content-scoped server events.
///
/// Holds `response_id`, `item_id`, `output_index`, and `content_index`.
/// Terminal methods take `event_id` per call, enabling reuse in streaming loops.
#[derive(Clone, Debug)]
pub struct ContentEventBuilder {
    response_id: String,
    item_id: String,
    output_index: u32,
    content_index: u32,
}

impl ContentEventBuilder {
    /// Create a new content-scoped context directly.
    pub fn new(
        response_id: impl Into<String>,
        item_id: impl Into<String>,
        output_index: u32,
        content_index: u32,
    ) -> Self {
        Self {
            response_id: response_id.into(),
            item_id: item_id.into(),
            output_index,
            content_index,
        }
    }

    // ---- Response text events ----

    /// Build a `response.output_text.delta` event.
    pub fn output_text_delta(
        &self,
        event_id: impl Into<String>,
        delta: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::ResponseOutputTextDelta {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            content_index: self.content_index,
            delta: delta.into(),
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

    // One-shot events via ServerEvent associated functions
    #[test]
    fn test_session_created() {
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

        let event = ServerEvent::session_created("evt_1", config);
        assert_eq!(event.event_type(), "session.created");
        let json = serde_json::to_string(&event).expect("serialization failed");
        assert!(json.contains("\"type\":\"session.created\""));
    }

    // Full hierarchy — verifies context propagation at every level
    #[test]
    fn test_full_hierarchy() {
        let l2 = ResponseEventBuilder::new("resp_1");

        // Level 2 terminal — only needs response_id
        let event = l2.output_audio_buffer_started("evt_1");
        assert_eq!(event.event_type(), "output_audio_buffer.started");
        if let ServerEvent::OutputAudioBufferStarted { response_id, .. } = &event {
            assert_eq!(response_id, "resp_1");
        } else {
            panic!("Expected OutputAudioBufferStarted");
        }

        // Level 3 terminal — needs response_id + item context
        let l3 = l2.for_item("item_1", 0);
        let item = RealtimeConversationItem::FunctionCallOutput {
            call_id: "call_1".into(),
            output: "result".into(),
            id: None,
            object: None,
            status: None,
        };
        let event = l3.output_item_done("evt_2", item);
        assert_eq!(event.event_type(), "response.output_item.done");
        if let ServerEvent::ResponseOutputItemDone {
            response_id,
            output_index,
            ..
        } = &event
        {
            assert_eq!(response_id, "resp_1");
            assert_eq!(*output_index, 0);
        } else {
            panic!("Expected ResponseOutputItemDone");
        }

        // Level 4 terminal — needs response_id + item + content context
        let l4 = l3.for_content(0);
        let event = l4.output_text_delta("evt_3", "Hello");
        assert_eq!(event.event_type(), "response.output_text.delta");
        if let ServerEvent::ResponseOutputTextDelta {
            response_id,
            item_id,
            output_index,
            content_index,
            delta,
            ..
        } = &event
        {
            assert_eq!(response_id, "resp_1");
            assert_eq!(item_id, "item_1");
            assert_eq!(*output_index, 0);
            assert_eq!(*content_index, 0);
            assert_eq!(delta, "Hello");
        } else {
            panic!("Expected ResponseOutputTextDelta");
        }
    }

    // Streaming reuse test — the main motivation for this refactoring
    #[test]
    fn test_streaming_reuse() {
        let ctx = ResponseEventBuilder::new("resp_1")
            .for_item("item_1", 0)
            .for_content(0);

        let chunks = ["Hello", " ", "world"];
        let events: Vec<_> = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| ctx.output_text_delta(format!("evt_{i}"), *chunk))
            .collect();

        assert_eq!(events.len(), 3);
        for (i, event) in events.iter().enumerate() {
            assert_eq!(event.event_type(), "response.output_text.delta");
            if let ServerEvent::ResponseOutputTextDelta {
                event_id, delta, ..
            } = event
            {
                assert_eq!(event_id, &format!("evt_{i}"));
                assert_eq!(delta, chunks[i]);
            }
        }
    }
}
