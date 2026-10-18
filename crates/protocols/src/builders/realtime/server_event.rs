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
    realtime_events::{
        Conversation, LogProbProperties, RealtimeError, RealtimeRateLimit, ResponseContentPart,
        ServerEvent, SessionConfig, TranscriptionError, TranscriptionUsage,
    },
    realtime_response::RealtimeResponse,
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

    /// Build a `session.updated` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn session_updated(event_id: impl Into<String>, session: SessionConfig) -> Self {
        Self::SessionUpdated {
            event_id: event_id.into(),
            session: Box::new(session),
        }
    }

    // ---- Conversation events ----

    /// Build a `conversation.created` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn conversation_created(event_id: impl Into<String>, conversation: Conversation) -> Self {
        Self::ConversationCreated {
            event_id: event_id.into(),
            conversation,
        }
    }

    /// Build a `conversation.item.created` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn conversation_item_created(
        event_id: impl Into<String>,
        previous_item_id: Option<String>,
        item: RealtimeConversationItem,
    ) -> Self {
        Self::ConversationItemCreated {
            event_id: event_id.into(),
            previous_item_id,
            item,
        }
    }

    /// Build a `conversation.item.added` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn conversation_item_added(
        event_id: impl Into<String>,
        previous_item_id: Option<String>,
        item: RealtimeConversationItem,
    ) -> Self {
        Self::ConversationItemAdded {
            event_id: event_id.into(),
            previous_item_id,
            item,
        }
    }

    /// Build a `conversation.item.done` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn conversation_item_done(
        event_id: impl Into<String>,
        previous_item_id: Option<String>,
        item: RealtimeConversationItem,
    ) -> Self {
        Self::ConversationItemDone {
            event_id: event_id.into(),
            previous_item_id,
            item,
        }
    }

    /// Build a `conversation.item.deleted` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn conversation_item_deleted(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
    ) -> Self {
        Self::ConversationItemDeleted {
            event_id: event_id.into(),
            item_id: item_id.into(),
        }
    }

    /// Build a `conversation.item.retrieved` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn conversation_item_retrieved(
        event_id: impl Into<String>,
        item: RealtimeConversationItem,
    ) -> Self {
        Self::ConversationItemRetrieved {
            event_id: event_id.into(),
            item,
        }
    }

    /// Build a `conversation.item.truncated` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn conversation_item_truncated(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
        content_index: u32,
        audio_end_ms: u32,
    ) -> Self {
        Self::ConversationItemTruncated {
            event_id: event_id.into(),
            item_id: item_id.into(),
            content_index,
            audio_end_ms,
        }
    }

    // ---- Input audio transcription events ----

    /// Build a `conversation.item.input_audio_transcription.completed` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_transcription_completed(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
        content_index: u32,
        transcript: impl Into<String>,
        usage: TranscriptionUsage,
    ) -> Self {
        Self::InputAudioTranscriptionCompleted {
            event_id: event_id.into(),
            item_id: item_id.into(),
            content_index,
            transcript: transcript.into(),
            logprobs: None,
            usage,
        }
    }

    /// Build a `conversation.item.input_audio_transcription.completed` event with logprobs.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_transcription_completed_with_logprobs(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
        content_index: u32,
        transcript: impl Into<String>,
        usage: TranscriptionUsage,
        logprobs: Vec<LogProbProperties>,
    ) -> Self {
        Self::InputAudioTranscriptionCompleted {
            event_id: event_id.into(),
            item_id: item_id.into(),
            content_index,
            transcript: transcript.into(),
            logprobs: Some(logprobs),
            usage,
        }
    }

    /// Build a `conversation.item.input_audio_transcription.delta` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_transcription_delta(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
        content_index: Option<u32>,
        delta: Option<String>,
        logprobs: Option<Vec<LogProbProperties>>,
    ) -> Self {
        Self::InputAudioTranscriptionDelta {
            event_id: event_id.into(),
            item_id: item_id.into(),
            content_index,
            delta,
            logprobs,
        }
    }

    /// Build a `conversation.item.input_audio_transcription.failed` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_transcription_failed(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
        content_index: u32,
        error: TranscriptionError,
    ) -> Self {
        Self::InputAudioTranscriptionFailed {
            event_id: event_id.into(),
            item_id: item_id.into(),
            content_index,
            error,
        }
    }

    /// Build a `conversation.item.input_audio_transcription.segment` event.
    #[expect(clippy::too_many_arguments)]
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_transcription_segment(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
        content_index: u32,
        text: impl Into<String>,
        id: impl Into<String>,
        speaker: impl Into<String>,
        start: f32,
        end: f32,
    ) -> Self {
        Self::InputAudioTranscriptionSegment {
            event_id: event_id.into(),
            item_id: item_id.into(),
            content_index,
            text: text.into(),
            id: id.into(),
            speaker: speaker.into(),
            start,
            end,
        }
    }

    // ---- Input audio buffer events ----

    /// Build an `input_audio_buffer.cleared` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_buffer_cleared(event_id: impl Into<String>) -> Self {
        Self::InputAudioBufferCleared {
            event_id: event_id.into(),
        }
    }

    /// Build an `input_audio_buffer.committed` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_buffer_committed(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
        previous_item_id: Option<String>,
    ) -> Self {
        Self::InputAudioBufferCommitted {
            event_id: event_id.into(),
            previous_item_id,
            item_id: item_id.into(),
        }
    }

    /// Build an `input_audio_buffer.speech_started` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_buffer_speech_started(
        event_id: impl Into<String>,
        audio_start_ms: u32,
        item_id: impl Into<String>,
    ) -> Self {
        Self::InputAudioBufferSpeechStarted {
            event_id: event_id.into(),
            audio_start_ms,
            item_id: item_id.into(),
        }
    }

    /// Build an `input_audio_buffer.speech_stopped` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_buffer_speech_stopped(
        event_id: impl Into<String>,
        audio_end_ms: u32,
        item_id: impl Into<String>,
    ) -> Self {
        Self::InputAudioBufferSpeechStopped {
            event_id: event_id.into(),
            audio_end_ms,
            item_id: item_id.into(),
        }
    }

    /// Build an `input_audio_buffer.timeout_triggered` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn input_audio_buffer_timeout_triggered(
        event_id: impl Into<String>,
        audio_start_ms: u32,
        audio_end_ms: u32,
        item_id: impl Into<String>,
    ) -> Self {
        Self::InputAudioBufferTimeoutTriggered {
            event_id: event_id.into(),
            audio_start_ms,
            audio_end_ms,
            item_id: item_id.into(),
        }
    }

    /// Build an `input_audio_buffer.dtmf_event_received` event.
    ///
    /// DTMF events have no `event_id` per the OpenAI spec.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn dtmf_event_received(event: impl Into<String>, received_at: i64) -> Self {
        Self::InputAudioBufferDtmfEventReceived {
            event: event.into(),
            received_at,
        }
    }

    // ---- MCP list tools events ----

    /// Build an `mcp_list_tools.in_progress` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn mcp_list_tools_in_progress(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
    ) -> Self {
        Self::McpListToolsInProgress {
            event_id: event_id.into(),
            item_id: item_id.into(),
        }
    }

    /// Build an `mcp_list_tools.completed` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn mcp_list_tools_completed(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
    ) -> Self {
        Self::McpListToolsCompleted {
            event_id: event_id.into(),
            item_id: item_id.into(),
        }
    }

    /// Build an `mcp_list_tools.failed` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn mcp_list_tools_failed(event_id: impl Into<String>, item_id: impl Into<String>) -> Self {
        Self::McpListToolsFailed {
            event_id: event_id.into(),
            item_id: item_id.into(),
        }
    }

    // ---- MCP call lifecycle events ----

    /// Build a `response.mcp_call.in_progress` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn mcp_call_in_progress(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
        output_index: u32,
    ) -> Self {
        Self::ResponseMcpCallInProgress {
            event_id: event_id.into(),
            output_index,
            item_id: item_id.into(),
        }
    }

    /// Build a `response.mcp_call.completed` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn mcp_call_completed(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
        output_index: u32,
    ) -> Self {
        Self::ResponseMcpCallCompleted {
            event_id: event_id.into(),
            output_index,
            item_id: item_id.into(),
        }
    }

    /// Build a `response.mcp_call.failed` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn mcp_call_failed(
        event_id: impl Into<String>,
        item_id: impl Into<String>,
        output_index: u32,
    ) -> Self {
        Self::ResponseMcpCallFailed {
            event_id: event_id.into(),
            output_index,
            item_id: item_id.into(),
        }
    }

    // ---- Rate limits ----

    /// Build a `rate_limits.updated` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn rate_limits_updated(
        event_id: impl Into<String>,
        rate_limits: Vec<RealtimeRateLimit>,
    ) -> Self {
        Self::RateLimitsUpdated {
            event_id: event_id.into(),
            rate_limits,
        }
    }

    // ---- Response lifecycle events ----

    /// Build a `response.created` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn response_created(event_id: impl Into<String>, response: RealtimeResponse) -> Self {
        Self::ResponseCreated {
            event_id: event_id.into(),
            response: Box::new(response),
        }
    }

    /// Build a `response.done` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn response_done(event_id: impl Into<String>, response: RealtimeResponse) -> Self {
        Self::ResponseDone {
            event_id: event_id.into(),
            response: Box::new(response),
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

    /// Build an `output_audio_buffer.stopped` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn output_audio_buffer_stopped(&self, event_id: impl Into<String>) -> ServerEvent {
        ServerEvent::OutputAudioBufferStopped {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
        }
    }

    /// Build an `output_audio_buffer.cleared` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn output_audio_buffer_cleared(&self, event_id: impl Into<String>) -> ServerEvent {
        ServerEvent::OutputAudioBufferCleared {
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

    /// Build a `response.output_item.added` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn output_item_added(
        &self,
        event_id: impl Into<String>,
        item: RealtimeConversationItem,
    ) -> ServerEvent {
        ServerEvent::ResponseOutputItemAdded {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            output_index: self.output_index,
            item,
        }
    }

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

    // ---- Response function call events ----

    /// Build a `response.function_call_arguments.delta` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn function_call_arguments_delta(
        &self,
        event_id: impl Into<String>,
        call_id: impl Into<String>,
        delta: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::ResponseFunctionCallArgumentsDelta {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            call_id: call_id.into(),
            delta: delta.into(),
        }
    }

    /// Build a `response.function_call_arguments.done` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn function_call_arguments_done(
        &self,
        event_id: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::ResponseFunctionCallArgumentsDone {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            call_id: call_id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    // ---- Response MCP call events ----

    /// Build a `response.mcp_call_arguments.delta` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn mcp_call_arguments_delta(
        &self,
        event_id: impl Into<String>,
        delta: impl Into<String>,
        obfuscation: Option<String>,
    ) -> ServerEvent {
        ServerEvent::ResponseMcpCallArgumentsDelta {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            delta: delta.into(),
            obfuscation,
        }
    }

    /// Build a `response.mcp_call_arguments.done` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn mcp_call_arguments_done(
        &self,
        event_id: impl Into<String>,
        arguments: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::ResponseMcpCallArgumentsDone {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            arguments: arguments.into(),
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

    // ---- Response content part events ----

    /// Build a `response.content_part.added` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn content_part_added(
        &self,
        event_id: impl Into<String>,
        part: ResponseContentPart,
    ) -> ServerEvent {
        ServerEvent::ResponseContentPartAdded {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            content_index: self.content_index,
            part,
        }
    }

    /// Build a `response.content_part.done` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn content_part_done(
        &self,
        event_id: impl Into<String>,
        part: ResponseContentPart,
    ) -> ServerEvent {
        ServerEvent::ResponseContentPartDone {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            content_index: self.content_index,
            part,
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

    /// Build a `response.output_text.done` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn output_text_done(
        &self,
        event_id: impl Into<String>,
        text: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::ResponseOutputTextDone {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            content_index: self.content_index,
            text: text.into(),
        }
    }

    // ---- Response audio events ----

    /// Build a `response.output_audio.delta` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn output_audio_delta(
        &self,
        event_id: impl Into<String>,
        delta: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::ResponseOutputAudioDelta {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            content_index: self.content_index,
            delta: delta.into(),
        }
    }

    /// Build a `response.output_audio.done` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn output_audio_done(&self, event_id: impl Into<String>) -> ServerEvent {
        ServerEvent::ResponseOutputAudioDone {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            content_index: self.content_index,
        }
    }

    // ---- Response audio transcript events ----

    /// Build a `response.output_audio_transcript.delta` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn output_audio_transcript_delta(
        &self,
        event_id: impl Into<String>,
        delta: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::ResponseOutputAudioTranscriptDelta {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            content_index: self.content_index,
            delta: delta.into(),
        }
    }

    /// Build a `response.output_audio_transcript.done` event.
    #[deprecated(
        since = "1.17.0",
        note = "unused by the gateway; removed in the next major"
    )]
    pub fn output_audio_transcript_done(
        &self,
        event_id: impl Into<String>,
        transcript: impl Into<String>,
    ) -> ServerEvent {
        ServerEvent::ResponseOutputAudioTranscriptDone {
            event_id: event_id.into(),
            response_id: self.response_id.clone(),
            item_id: self.item_id.clone(),
            output_index: self.output_index,
            content_index: self.content_index,
            transcript: transcript.into(),
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
