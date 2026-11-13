// OpenAI Realtime API wire-format event types
// https://platform.openai.com/docs/api-reference/realtime
//
// This module defines the serializable/deserializable event structures
// for both client-to-server and server-to-client messages sent over
// WebSocket, WebRTC, or SIP connections.
//
// Session configuration types live in `realtime_session`.
// Conversation item types live in `realtime_conversation`.
// Response and usage types live in `realtime_response`.
// Event type string constants live in `event_types`.

use serde::{Deserialize, Serialize};

use crate::{
    event_types::{RealtimeClientEvent, RealtimeServerEvent},
    realtime_conversation::RealtimeConversationItem,
    realtime_response::{RealtimeResponse, RealtimeResponseCreateParams},
    realtime_session::{RealtimeSessionCreateRequest, RealtimeTranscriptionSessionCreateRequest},
};

// ============================================================================
// Client Events
// ============================================================================

/// A client-to-server event in the OpenAI Realtime API.
///
/// Sent by the client over WebSocket, WebRTC, or SIP connections.
/// Discriminated by the `type` field in the JSON wire format.
///
/// Large payloads (`SessionConfig` 624 B, `RealtimeResponseCreateParams` 384 B) are
/// `Box`-ed so the enum stays ≈224 bytes instead of ≈648.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientEvent {
    // ---- Session ----
    /// Update the session configuration.
    #[serde(rename = "session.update")]
    SessionUpdate {
        session: Box<SessionConfig>,
        event_id: Option<String>,
    },

    // ---- Conversation items ----
    /// Add a new item to the conversation.
    #[serde(rename = "conversation.item.create")]
    ConversationItemCreate {
        item: RealtimeConversationItem,
        event_id: Option<String>,
        previous_item_id: Option<String>,
    },

    /// Remove an item from the conversation history.
    #[serde(rename = "conversation.item.delete")]
    ConversationItemDelete {
        item_id: String,
        event_id: Option<String>,
    },

    /// Retrieve the server's representation of a conversation item.
    #[serde(rename = "conversation.item.retrieve")]
    ConversationItemRetrieve {
        item_id: String,
        event_id: Option<String>,
    },

    /// Truncate a previous assistant message's audio.
    #[serde(rename = "conversation.item.truncate")]
    ConversationItemTruncate {
        audio_end_ms: u32,
        content_index: u32,
        item_id: String,
        event_id: Option<String>,
    },

    // ---- Input audio buffer ----
    /// Append audio bytes to the input audio buffer.
    ///
    /// WARNING: `audio` contains a base64 audio blob that can be very large.
    /// Avoid logging this variant with `Debug` in production; prefer
    /// `event_type()` for structured logging.
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend {
        audio: String,
        event_id: Option<String>,
    },

    /// Clear the input audio buffer.
    #[serde(rename = "input_audio_buffer.clear")]
    InputAudioBufferClear { event_id: Option<String> },

    /// Commit the input audio buffer as a user message.
    #[serde(rename = "input_audio_buffer.commit")]
    InputAudioBufferCommit { event_id: Option<String> },

    // ---- Output audio buffer (WebRTC/SIP only) ----
    /// Cut off the current audio response.
    #[serde(rename = "output_audio_buffer.clear")]
    OutputAudioBufferClear { event_id: Option<String> },

    // ---- Response ----
    /// Cancel an in-progress response.
    #[serde(rename = "response.cancel")]
    ResponseCancel {
        event_id: Option<String>,
        response_id: Option<String>,
    },

    /// Trigger model inference to create a response.
    #[serde(rename = "response.create")]
    ResponseCreate {
        event_id: Option<String>,
        response: Option<Box<RealtimeResponseCreateParams>>,
    },

    // ---- Unknown ----
    /// Unrecognized event type. Serde automatically deserializes any
    /// unrecognized `type` value into this variant (no data preserved).
    /// For proxy use, forward the raw frame instead of re-serializing.
    #[serde(other)]
    Unknown,
}

impl ClientEvent {
    /// Returns the event type string (e.g. `"session.update"`).
    ///
    /// For unknown events, returns `"unknown"`.
    pub fn event_type(&self) -> &str {
        self.to_event_type()
            .map(|e| e.as_str())
            .unwrap_or("unknown")
    }

    /// Maps this event to its corresponding `RealtimeClientEvent` constant.
    ///
    /// Returns `None` for `Unknown` events.
    pub fn to_event_type(&self) -> Option<RealtimeClientEvent> {
        match self {
            ClientEvent::SessionUpdate { .. } => Some(RealtimeClientEvent::SessionUpdate),
            ClientEvent::ConversationItemCreate { .. } => {
                Some(RealtimeClientEvent::ConversationItemCreate)
            }
            ClientEvent::ConversationItemDelete { .. } => {
                Some(RealtimeClientEvent::ConversationItemDelete)
            }
            ClientEvent::ConversationItemRetrieve { .. } => {
                Some(RealtimeClientEvent::ConversationItemRetrieve)
            }
            ClientEvent::ConversationItemTruncate { .. } => {
                Some(RealtimeClientEvent::ConversationItemTruncate)
            }
            ClientEvent::InputAudioBufferAppend { .. } => {
                Some(RealtimeClientEvent::InputAudioBufferAppend)
            }
            ClientEvent::InputAudioBufferClear { .. } => {
                Some(RealtimeClientEvent::InputAudioBufferClear)
            }
            ClientEvent::InputAudioBufferCommit { .. } => {
                Some(RealtimeClientEvent::InputAudioBufferCommit)
            }
            ClientEvent::OutputAudioBufferClear { .. } => {
                Some(RealtimeClientEvent::OutputAudioBufferClear)
            }
            ClientEvent::ResponseCancel { .. } => Some(RealtimeClientEvent::ResponseCancel),
            ClientEvent::ResponseCreate { .. } => Some(RealtimeClientEvent::ResponseCreate),
            ClientEvent::Unknown => None,
        }
    }
}

// ============================================================================
// Server Events
// ============================================================================

/// A server-to-client event in the OpenAI Realtime API.
///
/// Sent by the server over WebSocket, WebRTC, or SIP connections.
/// Discriminated by the `type` field in the JSON wire format.
///
/// Large payloads (`SessionConfig` 624 B, `RealtimeResponse` 352 B) are
/// `Box`-ed so the enum stays ≈232 bytes instead of ≈656.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerEvent {
    // ---- Session events ----
    /// Emitted when a new connection is established with the default session config.
    #[serde(rename = "session.created")]
    SessionCreated {
        event_id: String,
        session: Box<SessionConfig>,
    },

    /// Emitted after a successful `session.update`.
    #[serde(rename = "session.updated")]
    SessionUpdated {
        event_id: String,
        session: Box<SessionConfig>,
    },

    // ---- Conversation events ----
    /// Emitted when a conversation is created (right after session creation).
    #[serde(rename = "conversation.created")]
    ConversationCreated {
        conversation: Conversation,
        event_id: String,
    },

    /// Emitted when a conversation item is created (legacy event).
    #[serde(rename = "conversation.item.created")]
    ConversationItemCreated {
        event_id: String,
        item: RealtimeConversationItem,
        previous_item_id: Option<String>,
    },

    /// Emitted when an item is added to the default conversation.
    #[serde(rename = "conversation.item.added")]
    ConversationItemAdded {
        event_id: String,
        item: RealtimeConversationItem,
        previous_item_id: Option<String>,
    },

    /// Emitted when a conversation item is finalized.
    #[serde(rename = "conversation.item.done")]
    ConversationItemDone {
        event_id: String,
        item: RealtimeConversationItem,
        previous_item_id: Option<String>,
    },

    /// Emitted when a conversation item is deleted.
    #[serde(rename = "conversation.item.deleted")]
    ConversationItemDeleted { event_id: String, item_id: String },

    /// Emitted in response to `conversation.item.retrieve`.
    #[serde(rename = "conversation.item.retrieved")]
    ConversationItemRetrieved {
        event_id: String,
        item: RealtimeConversationItem,
    },

    /// Emitted when an assistant audio message item is truncated.
    #[serde(rename = "conversation.item.truncated")]
    ConversationItemTruncated {
        audio_end_ms: u32,
        content_index: u32,
        event_id: String,
        item_id: String,
    },

    // ---- Input audio transcription events ----
    /// Emitted when input audio transcription completes.
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    InputAudioTranscriptionCompleted {
        content_index: u32,
        event_id: String,
        item_id: String,
        transcript: String,
        usage: TranscriptionUsage,
        logprobs: Option<Vec<LogProbProperties>>,
        /// The languages detected in the audio; an empty list means none
        /// could be detected reliably.
        languages: Option<Vec<TranscriptionLanguage>>,
    },

    /// Emitted with incremental transcription results.
    #[serde(rename = "conversation.item.input_audio_transcription.delta")]
    InputAudioTranscriptionDelta {
        event_id: String,
        item_id: String,
        content_index: Option<u32>,
        delta: Option<String>,
        logprobs: Option<Vec<LogProbProperties>>,
    },

    /// Emitted when input audio transcription fails.
    #[serde(rename = "conversation.item.input_audio_transcription.failed")]
    InputAudioTranscriptionFailed {
        content_index: u32,
        error: TranscriptionError,
        event_id: String,
        item_id: String,
    },

    /// Emitted when an input audio transcription segment is identified
    /// (used with diarization models).
    #[serde(rename = "conversation.item.input_audio_transcription.segment")]
    InputAudioTranscriptionSegment {
        id: String,
        content_index: u32,
        /// The segment's end, in seconds, kept at the precision the
        /// upstream wrote it
        end: f64,
        event_id: String,
        item_id: String,
        speaker: String,
        /// The segment's start, in seconds
        start: f64,
        text: String,
    },

    // ---- Input audio buffer events ----
    /// Emitted when the input audio buffer is cleared.
    #[serde(rename = "input_audio_buffer.cleared")]
    InputAudioBufferCleared { event_id: String },

    /// Emitted when the input audio buffer is committed.
    #[serde(rename = "input_audio_buffer.committed")]
    InputAudioBufferCommitted {
        event_id: String,
        item_id: String,
        previous_item_id: Option<String>,
    },

    /// Emitted when speech is detected in the audio buffer (server VAD mode).
    #[serde(rename = "input_audio_buffer.speech_started")]
    InputAudioBufferSpeechStarted {
        audio_start_ms: u32,
        event_id: String,
        item_id: String,
    },

    /// Emitted when the end of speech is detected (server VAD mode).
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    InputAudioBufferSpeechStopped {
        audio_end_ms: u32,
        event_id: String,
        item_id: String,
    },

    /// Emitted when the VAD idle timeout triggers.
    #[serde(rename = "input_audio_buffer.timeout_triggered")]
    InputAudioBufferTimeoutTriggered {
        audio_end_ms: u32,
        audio_start_ms: u32,
        event_id: String,
        item_id: String,
    },

    /// **SIP Only:** Emitted when a DTMF keypad event is received.
    ///
    /// NOTE: This is the only server event without an `event_id` field per the
    /// OpenAI spec. Downstream code that generically extracts `event_id` from
    /// all server events must handle this variant as a special case.
    #[serde(rename = "input_audio_buffer.dtmf_event_received")]
    InputAudioBufferDtmfEventReceived { event: String, received_at: i64 },

    // ---- Output audio buffer events (WebRTC/SIP only) ----
    /// Emitted when the server begins streaming audio to the client.
    #[serde(rename = "output_audio_buffer.started")]
    OutputAudioBufferStarted {
        event_id: String,
        response_id: String,
    },

    /// Emitted when the output audio buffer has been completely drained.
    #[serde(rename = "output_audio_buffer.stopped")]
    OutputAudioBufferStopped {
        event_id: String,
        response_id: String,
    },

    /// Emitted when the output audio buffer is cleared (user interrupt or
    /// explicit `output_audio_buffer.clear`).
    #[serde(rename = "output_audio_buffer.cleared")]
    OutputAudioBufferCleared {
        event_id: String,
        response_id: String,
    },

    // ---- Response lifecycle events ----
    /// Emitted when a new response is created (status `in_progress`).
    #[serde(rename = "response.created")]
    ResponseCreated {
        event_id: String,
        response: Box<RealtimeResponse>,
    },

    /// Emitted when a response is done streaming.
    #[serde(rename = "response.done")]
    ResponseDone {
        event_id: String,
        response: Box<RealtimeResponse>,
    },

    // ---- Response output item events ----
    /// Emitted when a new output item is created during response generation.
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded {
        event_id: String,
        item: RealtimeConversationItem,
        output_index: u32,
        response_id: String,
    },

    /// Emitted when an output item is done streaming.
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone {
        event_id: String,
        item: RealtimeConversationItem,
        output_index: u32,
        response_id: String,
    },

    // ---- Response content part events ----
    /// Emitted when a new content part is added to an assistant message.
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded {
        content_index: u32,
        event_id: String,
        item_id: String,
        output_index: u32,
        part: ResponseContentPart,
        response_id: String,
    },

    /// Emitted when a content part is done streaming.
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone {
        content_index: u32,
        event_id: String,
        item_id: String,
        output_index: u32,
        part: ResponseContentPart,
        response_id: String,
    },

    // ---- Response text events ----
    /// Emitted when the text of an output_text content part is updated.
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta {
        content_index: u32,
        delta: String,
        event_id: String,
        item_id: String,
        output_index: u32,
        response_id: String,
    },

    /// Emitted when an output_text content part is done streaming.
    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone {
        content_index: u32,
        event_id: String,
        item_id: String,
        output_index: u32,
        response_id: String,
        text: String,
    },

    // ---- Response audio events ----
    /// Emitted when model-generated audio is updated.
    ///
    /// WARNING: `delta` contains a base64 audio chunk. Avoid logging this
    /// variant with `Debug` in production; prefer `event_type()`.
    #[serde(rename = "response.output_audio.delta")]
    ResponseOutputAudioDelta {
        content_index: u32,
        delta: String,
        event_id: String,
        item_id: String,
        output_index: u32,
        response_id: String,
    },

    /// Emitted when model-generated audio is done.
    #[serde(rename = "response.output_audio.done")]
    ResponseOutputAudioDone {
        content_index: u32,
        event_id: String,
        item_id: String,
        output_index: u32,
        response_id: String,
    },

    // ---- Response audio transcript events ----
    /// Emitted when the transcription of audio output is updated.
    #[serde(rename = "response.output_audio_transcript.delta")]
    ResponseOutputAudioTranscriptDelta {
        content_index: u32,
        delta: String,
        event_id: String,
        item_id: String,
        output_index: u32,
        response_id: String,
    },

    /// Emitted when the transcription of audio output is done.
    #[serde(rename = "response.output_audio_transcript.done")]
    ResponseOutputAudioTranscriptDone {
        content_index: u32,
        event_id: String,
        item_id: String,
        output_index: u32,
        response_id: String,
        transcript: String,
    },

    // ---- Response function call events ----
    /// Emitted when function call arguments are updated.
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta {
        call_id: String,
        delta: String,
        event_id: String,
        item_id: String,
        output_index: u32,
        response_id: String,
    },

    /// Emitted when function call arguments are done streaming.
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone {
        arguments: String,
        call_id: String,
        event_id: String,
        item_id: String,
        name: String,
        output_index: u32,
        response_id: String,
    },

    // ---- Response MCP call events ----
    /// Emitted when MCP tool call arguments are updated.
    #[serde(rename = "response.mcp_call_arguments.delta")]
    ResponseMcpCallArgumentsDelta {
        delta: String,
        event_id: String,
        item_id: String,
        output_index: u32,
        response_id: String,
        obfuscation: Option<String>,
    },

    /// Emitted when MCP tool call arguments are finalized.
    #[serde(rename = "response.mcp_call_arguments.done")]
    ResponseMcpCallArgumentsDone {
        arguments: String,
        event_id: String,
        item_id: String,
        output_index: u32,
        response_id: String,
    },

    /// Emitted when an MCP tool call starts.
    #[serde(rename = "response.mcp_call.in_progress")]
    ResponseMcpCallInProgress {
        event_id: String,
        item_id: String,
        output_index: u32,
    },

    /// Emitted when an MCP tool call completes successfully.
    #[serde(rename = "response.mcp_call.completed")]
    ResponseMcpCallCompleted {
        event_id: String,
        item_id: String,
        output_index: u32,
    },

    /// Emitted when an MCP tool call fails.
    #[serde(rename = "response.mcp_call.failed")]
    ResponseMcpCallFailed {
        event_id: String,
        item_id: String,
        output_index: u32,
    },

    // ---- MCP list tools events ----
    /// Emitted when listing MCP tools is in progress.
    #[serde(rename = "mcp_list_tools.in_progress")]
    McpListToolsInProgress { event_id: String, item_id: String },

    /// Emitted when listing MCP tools has completed.
    #[serde(rename = "mcp_list_tools.completed")]
    McpListToolsCompleted { event_id: String, item_id: String },

    /// Emitted when listing MCP tools has failed.
    #[serde(rename = "mcp_list_tools.failed")]
    McpListToolsFailed { event_id: String, item_id: String },

    // ---- Rate limits ----
    /// Emitted at the beginning of a response with updated rate limit info.
    #[serde(rename = "rate_limits.updated")]
    RateLimitsUpdated {
        event_id: String,
        rate_limits: Vec<RealtimeRateLimit>,
    },

    // ---- Error ----
    /// Emitted when an error occurs. Most errors are recoverable.
    #[serde(rename = "error")]
    Error {
        error: RealtimeError,
        event_id: String,
    },

    // ---- Unknown ----
    /// Unrecognized event type. Serde automatically deserializes any
    /// unrecognized `type` value into this variant (no data preserved).
    /// For proxy use, forward the raw frame instead of re-serializing.
    #[serde(other)]
    Unknown,
}

impl ServerEvent {
    /// Returns the event type string (e.g. `"session.created"`).
    ///
    /// For known events, returns a `&'static str` from the event type constants.
    /// For unknown events, returns `"unknown"`.
    pub fn event_type(&self) -> &str {
        self.to_event_type()
            .map(|e| e.as_str())
            .unwrap_or("unknown")
    }

    /// Maps this event to its corresponding `RealtimeServerEvent` constant.
    ///
    /// Returns `None` for `Unknown` events.
    pub fn to_event_type(&self) -> Option<RealtimeServerEvent> {
        match self {
            ServerEvent::SessionCreated { .. } => Some(RealtimeServerEvent::SessionCreated),
            ServerEvent::SessionUpdated { .. } => Some(RealtimeServerEvent::SessionUpdated),
            ServerEvent::ConversationCreated { .. } => {
                Some(RealtimeServerEvent::ConversationCreated)
            }
            ServerEvent::ConversationItemCreated { .. } => {
                Some(RealtimeServerEvent::ConversationItemCreated)
            }
            ServerEvent::ConversationItemAdded { .. } => {
                Some(RealtimeServerEvent::ConversationItemAdded)
            }
            ServerEvent::ConversationItemDone { .. } => {
                Some(RealtimeServerEvent::ConversationItemDone)
            }
            ServerEvent::ConversationItemDeleted { .. } => {
                Some(RealtimeServerEvent::ConversationItemDeleted)
            }
            ServerEvent::ConversationItemRetrieved { .. } => {
                Some(RealtimeServerEvent::ConversationItemRetrieved)
            }
            ServerEvent::ConversationItemTruncated { .. } => {
                Some(RealtimeServerEvent::ConversationItemTruncated)
            }
            ServerEvent::InputAudioTranscriptionCompleted { .. } => {
                Some(RealtimeServerEvent::ConversationItemInputAudioTranscriptionCompleted)
            }
            ServerEvent::InputAudioTranscriptionDelta { .. } => {
                Some(RealtimeServerEvent::ConversationItemInputAudioTranscriptionDelta)
            }
            ServerEvent::InputAudioTranscriptionFailed { .. } => {
                Some(RealtimeServerEvent::ConversationItemInputAudioTranscriptionFailed)
            }
            ServerEvent::InputAudioTranscriptionSegment { .. } => {
                Some(RealtimeServerEvent::ConversationItemInputAudioTranscriptionSegment)
            }
            ServerEvent::InputAudioBufferCleared { .. } => {
                Some(RealtimeServerEvent::InputAudioBufferCleared)
            }
            ServerEvent::InputAudioBufferCommitted { .. } => {
                Some(RealtimeServerEvent::InputAudioBufferCommitted)
            }
            ServerEvent::InputAudioBufferSpeechStarted { .. } => {
                Some(RealtimeServerEvent::InputAudioBufferSpeechStarted)
            }
            ServerEvent::InputAudioBufferSpeechStopped { .. } => {
                Some(RealtimeServerEvent::InputAudioBufferSpeechStopped)
            }
            ServerEvent::InputAudioBufferTimeoutTriggered { .. } => {
                Some(RealtimeServerEvent::InputAudioBufferTimeoutTriggered)
            }
            ServerEvent::InputAudioBufferDtmfEventReceived { .. } => {
                Some(RealtimeServerEvent::InputAudioBufferDtmfEventReceived)
            }
            ServerEvent::OutputAudioBufferStarted { .. } => {
                Some(RealtimeServerEvent::OutputAudioBufferStarted)
            }
            ServerEvent::OutputAudioBufferStopped { .. } => {
                Some(RealtimeServerEvent::OutputAudioBufferStopped)
            }
            ServerEvent::OutputAudioBufferCleared { .. } => {
                Some(RealtimeServerEvent::OutputAudioBufferCleared)
            }
            ServerEvent::ResponseCreated { .. } => Some(RealtimeServerEvent::ResponseCreated),
            ServerEvent::ResponseDone { .. } => Some(RealtimeServerEvent::ResponseDone),
            ServerEvent::ResponseOutputItemAdded { .. } => {
                Some(RealtimeServerEvent::ResponseOutputItemAdded)
            }
            ServerEvent::ResponseOutputItemDone { .. } => {
                Some(RealtimeServerEvent::ResponseOutputItemDone)
            }
            ServerEvent::ResponseContentPartAdded { .. } => {
                Some(RealtimeServerEvent::ResponseContentPartAdded)
            }
            ServerEvent::ResponseContentPartDone { .. } => {
                Some(RealtimeServerEvent::ResponseContentPartDone)
            }
            ServerEvent::ResponseOutputTextDelta { .. } => {
                Some(RealtimeServerEvent::ResponseOutputTextDelta)
            }
            ServerEvent::ResponseOutputTextDone { .. } => {
                Some(RealtimeServerEvent::ResponseOutputTextDone)
            }
            ServerEvent::ResponseOutputAudioDelta { .. } => {
                Some(RealtimeServerEvent::ResponseOutputAudioDelta)
            }
            ServerEvent::ResponseOutputAudioDone { .. } => {
                Some(RealtimeServerEvent::ResponseOutputAudioDone)
            }
            ServerEvent::ResponseOutputAudioTranscriptDelta { .. } => {
                Some(RealtimeServerEvent::ResponseOutputAudioTranscriptDelta)
            }
            ServerEvent::ResponseOutputAudioTranscriptDone { .. } => {
                Some(RealtimeServerEvent::ResponseOutputAudioTranscriptDone)
            }
            ServerEvent::ResponseFunctionCallArgumentsDelta { .. } => {
                Some(RealtimeServerEvent::ResponseFunctionCallArgumentsDelta)
            }
            ServerEvent::ResponseFunctionCallArgumentsDone { .. } => {
                Some(RealtimeServerEvent::ResponseFunctionCallArgumentsDone)
            }
            ServerEvent::ResponseMcpCallArgumentsDelta { .. } => {
                Some(RealtimeServerEvent::ResponseMcpCallArgumentsDelta)
            }
            ServerEvent::ResponseMcpCallArgumentsDone { .. } => {
                Some(RealtimeServerEvent::ResponseMcpCallArgumentsDone)
            }
            ServerEvent::ResponseMcpCallInProgress { .. } => {
                Some(RealtimeServerEvent::ResponseMcpCallInProgress)
            }
            ServerEvent::ResponseMcpCallCompleted { .. } => {
                Some(RealtimeServerEvent::ResponseMcpCallCompleted)
            }
            ServerEvent::ResponseMcpCallFailed { .. } => {
                Some(RealtimeServerEvent::ResponseMcpCallFailed)
            }
            ServerEvent::McpListToolsInProgress { .. } => {
                Some(RealtimeServerEvent::McpListToolsInProgress)
            }
            ServerEvent::McpListToolsCompleted { .. } => {
                Some(RealtimeServerEvent::McpListToolsCompleted)
            }
            ServerEvent::McpListToolsFailed { .. } => Some(RealtimeServerEvent::McpListToolsFailed),
            ServerEvent::RateLimitsUpdated { .. } => Some(RealtimeServerEvent::RateLimitsUpdated),
            ServerEvent::Error { .. } => Some(RealtimeServerEvent::Error),
            ServerEvent::Unknown => None,
        }
    }

    /// Returns true if this is a `response.function_call_arguments.done` event.
    pub fn is_function_call_done(&self) -> bool {
        matches!(self, ServerEvent::ResponseFunctionCallArgumentsDone { .. })
    }

    /// For `response.function_call_arguments.done`, returns `(call_id, item_id, arguments)`.
    pub fn get_function_call(&self) -> Option<(&str, &str, &str)> {
        match self {
            ServerEvent::ResponseFunctionCallArgumentsDone {
                call_id,
                item_id,
                arguments,
                ..
            } => Some((call_id, item_id, arguments)),
            _ => None,
        }
    }
}

// ============================================================================
// Session Config Union
// ============================================================================

/// Union of realtime and transcription session configurations, as the
/// `session.update`, `session.created` and `session.updated` events carry
/// it.
///
/// The session's own `type`, `"realtime"` or `"transcription"`, picks the
/// variant, and the variant reads the whole object, `type` included, as its
/// struct requires it. It is written the same way, so `type` appears once.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum SessionConfig {
    Realtime(Box<RealtimeSessionCreateRequest>),
    Transcription(Box<RealtimeTranscriptionSessionCreateRequest>),
}

impl<'de> Deserialize<'de> for SessionConfig {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        let session = serde_json::Value::deserialize(deserializer)?;
        match session.get("type").and_then(serde_json::Value::as_str) {
            Some("realtime") => serde_json::from_value(session)
                .map(|config| Self::Realtime(Box::new(config)))
                .map_err(D::Error::custom),
            Some("transcription") => serde_json::from_value(session)
                .map(|config| Self::Transcription(Box::new(config)))
                .map_err(D::Error::custom),
            Some(other) => Err(D::Error::unknown_variant(
                other,
                &["realtime", "transcription"],
            )),
            None => Err(D::Error::missing_field("type")),
        }
    }
}

// ============================================================================
// Supporting Types
// ============================================================================

/// A language detected in transcribed audio.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionLanguage {
    /// The language's code
    pub code: String,
}

/// Conversation metadata returned in `conversation.created` events.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: Option<String>,
    pub object: Option<String>,
}

/// A content part within a response (used in `response.content_part.*` events).
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseContentPart {
    pub audio: Option<String>,
    pub text: Option<String>,
    pub transcript: Option<String>,
    #[serde(rename = "type")]
    pub r#type: Option<String>,
}

/// Log probability entry for input audio transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProbProperties {
    pub token: String,
    /// UTF-8 byte values of the token. Serializes as a JSON array of integers
    /// (e.g. `[104, 101, 108, 108, 111]`), matching the OpenAI spec.
    pub bytes: Vec<u8>,
    pub logprob: f64,
}

/// Input token details for transcription usage.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionTokenInputDetails {
    pub audio_tokens: Option<u32>,
    pub text_tokens: Option<u32>,
}

/// Usage statistics for input audio transcription.
///
/// Discriminated by the `type` field: `"tokens"` or `"duration"`.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TranscriptionUsage {
    /// Token-based usage (e.g. for `gpt-4o-transcribe`).
    #[serde(rename = "tokens")]
    Tokens {
        input_tokens: u32,
        output_tokens: u32,
        total_tokens: u32,
        input_token_details: Option<TranscriptionTokenInputDetails>,
    },
    /// Duration-based usage (e.g. for `whisper-1`).
    #[serde(rename = "duration")]
    Duration { seconds: f64 },
}

/// Error details for a failed input audio transcription.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionError {
    pub code: Option<String>,
    pub message: Option<String>,
    pub param: Option<String>,
    #[serde(rename = "type")]
    pub r#type: Option<String>,
}

/// Rate limit information returned in `rate_limits.updated` events.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeRateLimit {
    pub limit: Option<u32>,
    pub name: Option<String>,
    pub remaining: Option<u32>,
    pub reset_seconds: Option<f64>,
}

/// Error details returned in the `error` server event.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeError {
    pub message: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub code: Option<String>,
    pub event_id: Option<String>,
    pub param: Option<String>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A `session.update`, `session.created` or `session.updated` event reads
    /// its session into the variant its `type` names, as the public API
    /// writes them, and is written back with that `type` once. A `type` no
    /// API names is refused, naming it.
    #[test]
    fn a_session_event_reads_its_session_and_writes_its_type_once() {
        let update = json!({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "You are a creative assistant that helps with design tasks.",
                "tool_choice": "auto"
            }
        });
        let event: ClientEvent = serde_json::from_value(update).unwrap();
        let ClientEvent::SessionUpdate { session, .. } = &event else {
            panic!("a session.update, got {event:?}");
        };
        assert!(matches!(session.as_ref(), SessionConfig::Realtime(_)));
        let written = serde_json::to_string(&event).unwrap();
        assert_eq!(
            written.matches(r#""type":"realtime""#).count(),
            1,
            "{written}"
        );

        let transcription = json!({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {"input": {"transcription": {"model": "gpt-4o-transcribe"}}}
            }
        });
        let event: ClientEvent = serde_json::from_value(transcription).unwrap();
        let ClientEvent::SessionUpdate { session, .. } = &event else {
            panic!("a session.update, got {event:?}");
        };
        assert!(matches!(session.as_ref(), SessionConfig::Transcription(_)));

        let created = json!({
            "type": "session.created",
            "event_id": "event_C9G5RJeJ2gF77mV7f2B1j",
            "session": {"type": "realtime", "object": "realtime.session", "model": "gpt-realtime"}
        });
        let event: ServerEvent = serde_json::from_value(created).unwrap();
        assert_eq!(event.event_type(), "session.created");

        let bogus = serde_json::from_value::<ClientEvent>(json!({
            "type": "session.update",
            "session": {"type": "a_session_of_no_api"}
        }))
        .expect_err("a session type no API names")
        .to_string();
        assert!(bogus.contains("a_session_of_no_api"), "{bogus}");
    }

    /// A transcription event keeps its detected languages, an empty list
    /// included, and its segment times at full precision; a failed
    /// response keeps its error's message. Each is written back as sent.
    #[test]
    fn server_events_keep_languages_segment_times_and_an_error_message() {
        let completed = json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "event_1",
            "item_id": "item_1",
            "content_index": 0,
            "transcript": "Bonjour",
            "usage": {"type": "duration", "seconds": 2},
            "languages": []
        });
        let event: ServerEvent = serde_json::from_value(completed.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(&event).unwrap()["languages"],
            json!([])
        );

        let segment = json!({
            "type": "conversation.item.input_audio_transcription.segment",
            "event_id": "event_2",
            "item_id": "item_1",
            "content_index": 0,
            "id": "seg_1",
            "speaker": "A",
            "text": "Hello",
            "start": 3600.123456,
            "end": 3601.654321
        });
        let event: ServerEvent = serde_json::from_value(segment.clone()).unwrap();
        let written = serde_json::to_value(&event).unwrap();
        assert_eq!(written["start"], json!(3600.123456));
        assert_eq!(written["end"], json!(3601.654321));

        let failed = json!({
            "type": "response.done",
            "event_id": "event_3",
            "response": {
                "object": "realtime.response",
                "id": "resp_1",
                "status": "failed",
                "status_details": {
                    "type": "failed",
                    "error": {"type": "server_error", "code": "internal", "message": "The server had an error."}
                }
            }
        });
        let event: ServerEvent = serde_json::from_value(failed).unwrap();
        let written = serde_json::to_value(&event).unwrap();
        assert_eq!(
            written["response"]["status_details"]["error"]["message"],
            "The server had an error."
        );
    }
}
