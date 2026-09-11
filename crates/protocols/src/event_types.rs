use std::fmt;

/// Response lifecycle events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResponseEvent {
    Created,
    InProgress,
    Completed,
    Incomplete,
}

impl ResponseEvent {
    pub const CREATED: &'static str = "response.created";
    pub const IN_PROGRESS: &'static str = "response.in_progress";
    pub const COMPLETED: &'static str = "response.completed";
    pub const INCOMPLETE: &'static str = "response.incomplete";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Created => Self::CREATED,
            Self::InProgress => Self::IN_PROGRESS,
            Self::Completed => Self::COMPLETED,
            Self::Incomplete => Self::INCOMPLETE,
        }
    }
}

impl fmt::Display for ResponseEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Output item events for streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputItemEvent {
    Added,
    Done,
    Delta,
}

impl OutputItemEvent {
    pub const ADDED: &'static str = "response.output_item.added";
    pub const DONE: &'static str = "response.output_item.done";
    pub const DELTA: &'static str = "response.output_item.delta";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Added => Self::ADDED,
            Self::Done => Self::DONE,
            Self::Delta => Self::DELTA,
        }
    }
}

impl fmt::Display for OutputItemEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Function call argument streaming events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionCallEvent {
    ArgumentsDelta,
    ArgumentsDone,
}

impl FunctionCallEvent {
    pub const ARGUMENTS_DELTA: &'static str = "response.function_call_arguments.delta";
    pub const ARGUMENTS_DONE: &'static str = "response.function_call_arguments.done";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ArgumentsDelta => Self::ARGUMENTS_DELTA,
            Self::ArgumentsDone => Self::ARGUMENTS_DONE,
        }
    }
}

impl fmt::Display for FunctionCallEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Content part streaming events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentPartEvent {
    Added,
    Done,
}

impl ContentPartEvent {
    pub const ADDED: &'static str = "response.content_part.added";
    pub const DONE: &'static str = "response.content_part.done";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Added => Self::ADDED,
            Self::Done => Self::DONE,
        }
    }
}

impl fmt::Display for ContentPartEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Output text streaming events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputTextEvent {
    Delta,
    Done,
}

impl OutputTextEvent {
    pub const DELTA: &'static str = "response.output_text.delta";
    pub const DONE: &'static str = "response.output_text.done";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Delta => Self::DELTA,
            Self::Done => Self::DONE,
        }
    }
}

impl fmt::Display for OutputTextEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// MCP Events
// ============================================================================

/// MCP (Model Context Protocol) call events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum McpEvent {
    CallArgumentsDelta,
    CallArgumentsDone,
    CallInProgress,
    CallCompleted,
    CallFailed,
    ListToolsInProgress,
    ListToolsCompleted,
}

impl McpEvent {
    pub const CALL_ARGUMENTS_DELTA: &'static str = "response.mcp_call_arguments.delta";
    pub const CALL_ARGUMENTS_DONE: &'static str = "response.mcp_call_arguments.done";
    pub const CALL_IN_PROGRESS: &'static str = "response.mcp_call.in_progress";
    pub const CALL_COMPLETED: &'static str = "response.mcp_call.completed";
    pub const CALL_FAILED: &'static str = "response.mcp_call.failed";
    pub const LIST_TOOLS_IN_PROGRESS: &'static str = "response.mcp_list_tools.in_progress";
    pub const LIST_TOOLS_COMPLETED: &'static str = "response.mcp_list_tools.completed";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CallArgumentsDelta => Self::CALL_ARGUMENTS_DELTA,
            Self::CallArgumentsDone => Self::CALL_ARGUMENTS_DONE,
            Self::CallInProgress => Self::CALL_IN_PROGRESS,
            Self::CallCompleted => Self::CALL_COMPLETED,
            Self::CallFailed => Self::CALL_FAILED,
            Self::ListToolsInProgress => Self::LIST_TOOLS_IN_PROGRESS,
            Self::ListToolsCompleted => Self::LIST_TOOLS_COMPLETED,
        }
    }
}

impl fmt::Display for McpEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// Built-in Tool Events
// ============================================================================

/// Web search call events for streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WebSearchCallEvent {
    InProgress,
    Searching,
    Completed,
}

impl WebSearchCallEvent {
    pub const IN_PROGRESS: &'static str = "response.web_search_call.in_progress";
    pub const SEARCHING: &'static str = "response.web_search_call.searching";
    pub const COMPLETED: &'static str = "response.web_search_call.completed";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => Self::IN_PROGRESS,
            Self::Searching => Self::SEARCHING,
            Self::Completed => Self::COMPLETED,
        }
    }
}

impl fmt::Display for WebSearchCallEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Code interpreter call events for streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeInterpreterCallEvent {
    InProgress,
    Interpreting,
    Completed,
}

impl CodeInterpreterCallEvent {
    pub const IN_PROGRESS: &'static str = "response.code_interpreter_call.in_progress";
    pub const INTERPRETING: &'static str = "response.code_interpreter_call.interpreting";
    pub const COMPLETED: &'static str = "response.code_interpreter_call.completed";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => Self::IN_PROGRESS,
            Self::Interpreting => Self::INTERPRETING,
            Self::Completed => Self::COMPLETED,
        }
    }
}

impl fmt::Display for CodeInterpreterCallEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// File search call events for streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileSearchCallEvent {
    InProgress,
    Searching,
    Completed,
}

impl FileSearchCallEvent {
    pub const IN_PROGRESS: &'static str = "response.file_search_call.in_progress";
    pub const SEARCHING: &'static str = "response.file_search_call.searching";
    pub const COMPLETED: &'static str = "response.file_search_call.completed";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => Self::IN_PROGRESS,
            Self::Searching => Self::SEARCHING,
            Self::Completed => Self::COMPLETED,
        }
    }
}

impl fmt::Display for FileSearchCallEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Image generation call events for streaming.
///
/// Mirrors OpenAI Python SDK 2.8.1 `response_image_gen_call_*_event.py` and
/// `.claude/_audit/openai-responses-api-spec.md` §tools (image_generation).
/// `PartialImage` is emitted 0-3 times per call when the tool is configured
/// with `partial_images`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageGenerationCallEvent {
    InProgress,
    Generating,
    PartialImage,
    Completed,
}

impl ImageGenerationCallEvent {
    pub const IN_PROGRESS: &'static str = "response.image_generation_call.in_progress";
    pub const GENERATING: &'static str = "response.image_generation_call.generating";
    pub const PARTIAL_IMAGE: &'static str = "response.image_generation_call.partial_image";
    pub const COMPLETED: &'static str = "response.image_generation_call.completed";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => Self::IN_PROGRESS,
            Self::Generating => Self::GENERATING,
            Self::PartialImage => Self::PARTIAL_IMAGE,
            Self::Completed => Self::COMPLETED,
        }
    }
}

impl fmt::Display for ImageGenerationCallEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Item type discriminators used in output items
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemType {
    FunctionCall,
    FunctionToolCall,
    McpCall,
    Function,
    McpListTools,
    WebSearchCall,
    CodeInterpreterCall,
    FileSearchCall,
    ImageGenerationCall,
}

impl ItemType {
    pub const FUNCTION_CALL: &'static str = "function_call";
    pub const FUNCTION_CALL_OUTPUT: &'static str = "function_call_output";
    pub const FUNCTION_TOOL_CALL: &'static str = "function_tool_call";
    pub const MCP_CALL: &'static str = "mcp_call";
    pub const FUNCTION: &'static str = "function";
    pub const MCP_LIST_TOOLS: &'static str = "mcp_list_tools";
    pub const WEB_SEARCH_CALL: &'static str = "web_search_call";
    pub const CODE_INTERPRETER_CALL: &'static str = "code_interpreter_call";
    pub const FILE_SEARCH_CALL: &'static str = "file_search_call";
    pub const IMAGE_GENERATION_CALL: &'static str = "image_generation_call";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FunctionCall => Self::FUNCTION_CALL,
            Self::FunctionToolCall => Self::FUNCTION_TOOL_CALL,
            Self::McpCall => Self::MCP_CALL,
            Self::Function => Self::FUNCTION,
            Self::McpListTools => Self::MCP_LIST_TOOLS,
            Self::WebSearchCall => Self::WEB_SEARCH_CALL,
            Self::CodeInterpreterCall => Self::CODE_INTERPRETER_CALL,
            Self::FileSearchCall => Self::FILE_SEARCH_CALL,
            Self::ImageGenerationCall => Self::IMAGE_GENERATION_CALL,
        }
    }

    /// Check if this is a function call variant (FunctionCall or FunctionToolCall)
    pub const fn is_function_call(self) -> bool {
        matches!(self, Self::FunctionCall | Self::FunctionToolCall)
    }

    /// Check if this is a builtin tool call variant
    pub const fn is_builtin_tool_call(self) -> bool {
        matches!(
            self,
            Self::WebSearchCall
                | Self::CodeInterpreterCall
                | Self::FileSearchCall
                | Self::ImageGenerationCall
        )
    }
}

impl fmt::Display for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// Realtime Client Events
// ============================================================================

/// Realtime API client events sent over WebSocket/WebRTC/SIP connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RealtimeClientEvent {
    ConversationItemCreate,
    ConversationItemDelete,
    ConversationItemRetrieve,
    ConversationItemTruncate,
    InputAudioBufferAppend,
    InputAudioBufferClear,
    InputAudioBufferCommit,
    OutputAudioBufferClear,
    ResponseCancel,
    ResponseCreate,
    SessionUpdate,
}

impl RealtimeClientEvent {
    pub const CONVERSATION_ITEM_CREATE: &'static str = "conversation.item.create";
    pub const CONVERSATION_ITEM_DELETE: &'static str = "conversation.item.delete";
    pub const CONVERSATION_ITEM_RETRIEVE: &'static str = "conversation.item.retrieve";
    pub const CONVERSATION_ITEM_TRUNCATE: &'static str = "conversation.item.truncate";
    pub const INPUT_AUDIO_BUFFER_APPEND: &'static str = "input_audio_buffer.append";
    pub const INPUT_AUDIO_BUFFER_CLEAR: &'static str = "input_audio_buffer.clear";
    pub const INPUT_AUDIO_BUFFER_COMMIT: &'static str = "input_audio_buffer.commit";
    pub const OUTPUT_AUDIO_BUFFER_CLEAR: &'static str = "output_audio_buffer.clear";
    pub const RESPONSE_CANCEL: &'static str = "response.cancel";
    pub const RESPONSE_CREATE: &'static str = "response.create";
    pub const SESSION_UPDATE: &'static str = "session.update";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConversationItemCreate => Self::CONVERSATION_ITEM_CREATE,
            Self::ConversationItemDelete => Self::CONVERSATION_ITEM_DELETE,
            Self::ConversationItemRetrieve => Self::CONVERSATION_ITEM_RETRIEVE,
            Self::ConversationItemTruncate => Self::CONVERSATION_ITEM_TRUNCATE,
            Self::InputAudioBufferAppend => Self::INPUT_AUDIO_BUFFER_APPEND,
            Self::InputAudioBufferClear => Self::INPUT_AUDIO_BUFFER_CLEAR,
            Self::InputAudioBufferCommit => Self::INPUT_AUDIO_BUFFER_COMMIT,
            Self::OutputAudioBufferClear => Self::OUTPUT_AUDIO_BUFFER_CLEAR,
            Self::ResponseCancel => Self::RESPONSE_CANCEL,
            Self::ResponseCreate => Self::RESPONSE_CREATE,
            Self::SessionUpdate => Self::SESSION_UPDATE,
        }
    }
}

impl fmt::Display for RealtimeClientEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// Realtime Server Events
// ============================================================================

/// Realtime API server events received over WebSocket/WebRTC/SIP connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RealtimeServerEvent {
    // Session events
    SessionCreated,
    SessionUpdated,
    // Conversation events
    ConversationCreated,
    ConversationItemCreated,
    ConversationItemAdded,
    ConversationItemDone,
    ConversationItemDeleted,
    ConversationItemRetrieved,
    ConversationItemTruncated,
    // Input audio transcription events
    ConversationItemInputAudioTranscriptionCompleted,
    ConversationItemInputAudioTranscriptionDelta,
    ConversationItemInputAudioTranscriptionFailed,
    ConversationItemInputAudioTranscriptionSegment,
    // Input audio buffer events
    InputAudioBufferCleared,
    InputAudioBufferCommitted,
    InputAudioBufferSpeechStarted,
    InputAudioBufferSpeechStopped,
    InputAudioBufferTimeoutTriggered,
    InputAudioBufferDtmfEventReceived,
    // Output audio buffer events (WebRTC/SIP only)
    OutputAudioBufferStarted,
    OutputAudioBufferStopped,
    OutputAudioBufferCleared,
    // Response lifecycle events
    ResponseCreated,
    ResponseDone,
    // Response output item events
    ResponseOutputItemAdded,
    ResponseOutputItemDone,
    // Response content part events
    ResponseContentPartAdded,
    ResponseContentPartDone,
    // Response text events
    ResponseOutputTextDelta,
    ResponseOutputTextDone,
    // Response audio events
    ResponseOutputAudioDelta,
    ResponseOutputAudioDone,
    // Response audio transcript events
    ResponseOutputAudioTranscriptDelta,
    ResponseOutputAudioTranscriptDone,
    // Response function call events
    ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone,
    // Response MCP call events
    ResponseMcpCallArgumentsDelta,
    ResponseMcpCallArgumentsDone,
    ResponseMcpCallInProgress,
    ResponseMcpCallCompleted,
    ResponseMcpCallFailed,
    // MCP list tools events
    McpListToolsInProgress,
    McpListToolsCompleted,
    McpListToolsFailed,
    // Rate limits
    RateLimitsUpdated,
    // Error
    Error,
}

impl RealtimeServerEvent {
    // Session events
    pub const SESSION_CREATED: &'static str = "session.created";
    pub const SESSION_UPDATED: &'static str = "session.updated";
    // Conversation events
    pub const CONVERSATION_CREATED: &'static str = "conversation.created";
    pub const CONVERSATION_ITEM_CREATED: &'static str = "conversation.item.created";
    pub const CONVERSATION_ITEM_ADDED: &'static str = "conversation.item.added";
    pub const CONVERSATION_ITEM_DONE: &'static str = "conversation.item.done";
    pub const CONVERSATION_ITEM_DELETED: &'static str = "conversation.item.deleted";
    pub const CONVERSATION_ITEM_RETRIEVED: &'static str = "conversation.item.retrieved";
    pub const CONVERSATION_ITEM_TRUNCATED: &'static str = "conversation.item.truncated";
    // Input audio transcription events
    pub const CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED: &'static str =
        "conversation.item.input_audio_transcription.completed";
    pub const CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA: &'static str =
        "conversation.item.input_audio_transcription.delta";
    pub const CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED: &'static str =
        "conversation.item.input_audio_transcription.failed";
    pub const CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_SEGMENT: &'static str =
        "conversation.item.input_audio_transcription.segment";
    // Input audio buffer events
    pub const INPUT_AUDIO_BUFFER_CLEARED: &'static str = "input_audio_buffer.cleared";
    pub const INPUT_AUDIO_BUFFER_COMMITTED: &'static str = "input_audio_buffer.committed";
    pub const INPUT_AUDIO_BUFFER_SPEECH_STARTED: &'static str = "input_audio_buffer.speech_started";
    pub const INPUT_AUDIO_BUFFER_SPEECH_STOPPED: &'static str = "input_audio_buffer.speech_stopped";
    pub const INPUT_AUDIO_BUFFER_TIMEOUT_TRIGGERED: &'static str =
        "input_audio_buffer.timeout_triggered";
    pub const INPUT_AUDIO_BUFFER_DTMF_EVENT_RECEIVED: &'static str =
        "input_audio_buffer.dtmf_event_received";
    // Output audio buffer events
    pub const OUTPUT_AUDIO_BUFFER_STARTED: &'static str = "output_audio_buffer.started";
    pub const OUTPUT_AUDIO_BUFFER_STOPPED: &'static str = "output_audio_buffer.stopped";
    pub const OUTPUT_AUDIO_BUFFER_CLEARED: &'static str = "output_audio_buffer.cleared";
    // Response lifecycle events
    pub const RESPONSE_CREATED: &'static str = "response.created";
    pub const RESPONSE_DONE: &'static str = "response.done";
    // Response output item events
    pub const RESPONSE_OUTPUT_ITEM_ADDED: &'static str = "response.output_item.added";
    pub const RESPONSE_OUTPUT_ITEM_DONE: &'static str = "response.output_item.done";
    // Response content part events
    pub const RESPONSE_CONTENT_PART_ADDED: &'static str = "response.content_part.added";
    pub const RESPONSE_CONTENT_PART_DONE: &'static str = "response.content_part.done";
    // Response text events
    pub const RESPONSE_OUTPUT_TEXT_DELTA: &'static str = "response.output_text.delta";
    pub const RESPONSE_OUTPUT_TEXT_DONE: &'static str = "response.output_text.done";
    // Response audio events
    pub const RESPONSE_OUTPUT_AUDIO_DELTA: &'static str = "response.output_audio.delta";
    pub const RESPONSE_OUTPUT_AUDIO_DONE: &'static str = "response.output_audio.done";
    // Response audio transcript events
    pub const RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA: &'static str =
        "response.output_audio_transcript.delta";
    pub const RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE: &'static str =
        "response.output_audio_transcript.done";
    // Response function call events
    pub const RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA: &'static str =
        "response.function_call_arguments.delta";
    pub const RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE: &'static str =
        "response.function_call_arguments.done";
    // Response MCP call events
    pub const RESPONSE_MCP_CALL_ARGUMENTS_DELTA: &'static str = "response.mcp_call_arguments.delta";
    pub const RESPONSE_MCP_CALL_ARGUMENTS_DONE: &'static str = "response.mcp_call_arguments.done";
    pub const RESPONSE_MCP_CALL_IN_PROGRESS: &'static str = "response.mcp_call.in_progress";
    pub const RESPONSE_MCP_CALL_COMPLETED: &'static str = "response.mcp_call.completed";
    pub const RESPONSE_MCP_CALL_FAILED: &'static str = "response.mcp_call.failed";
    // MCP list tools events
    pub const MCP_LIST_TOOLS_IN_PROGRESS: &'static str = "mcp_list_tools.in_progress";
    pub const MCP_LIST_TOOLS_COMPLETED: &'static str = "mcp_list_tools.completed";
    pub const MCP_LIST_TOOLS_FAILED: &'static str = "mcp_list_tools.failed";
    // Rate limits
    pub const RATE_LIMITS_UPDATED: &'static str = "rate_limits.updated";
    // Error
    pub const ERROR: &'static str = "error";

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SessionCreated => Self::SESSION_CREATED,
            Self::SessionUpdated => Self::SESSION_UPDATED,
            Self::ConversationCreated => Self::CONVERSATION_CREATED,
            Self::ConversationItemCreated => Self::CONVERSATION_ITEM_CREATED,
            Self::ConversationItemAdded => Self::CONVERSATION_ITEM_ADDED,
            Self::ConversationItemDone => Self::CONVERSATION_ITEM_DONE,
            Self::ConversationItemDeleted => Self::CONVERSATION_ITEM_DELETED,
            Self::ConversationItemRetrieved => Self::CONVERSATION_ITEM_RETRIEVED,
            Self::ConversationItemTruncated => Self::CONVERSATION_ITEM_TRUNCATED,
            Self::ConversationItemInputAudioTranscriptionCompleted => {
                Self::CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED
            }
            Self::ConversationItemInputAudioTranscriptionDelta => {
                Self::CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA
            }
            Self::ConversationItemInputAudioTranscriptionFailed => {
                Self::CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED
            }
            Self::ConversationItemInputAudioTranscriptionSegment => {
                Self::CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_SEGMENT
            }
            Self::InputAudioBufferCleared => Self::INPUT_AUDIO_BUFFER_CLEARED,
            Self::InputAudioBufferCommitted => Self::INPUT_AUDIO_BUFFER_COMMITTED,
            Self::InputAudioBufferSpeechStarted => Self::INPUT_AUDIO_BUFFER_SPEECH_STARTED,
            Self::InputAudioBufferSpeechStopped => Self::INPUT_AUDIO_BUFFER_SPEECH_STOPPED,
            Self::InputAudioBufferTimeoutTriggered => Self::INPUT_AUDIO_BUFFER_TIMEOUT_TRIGGERED,
            Self::InputAudioBufferDtmfEventReceived => Self::INPUT_AUDIO_BUFFER_DTMF_EVENT_RECEIVED,
            Self::OutputAudioBufferStarted => Self::OUTPUT_AUDIO_BUFFER_STARTED,
            Self::OutputAudioBufferStopped => Self::OUTPUT_AUDIO_BUFFER_STOPPED,
            Self::OutputAudioBufferCleared => Self::OUTPUT_AUDIO_BUFFER_CLEARED,
            Self::ResponseCreated => Self::RESPONSE_CREATED,
            Self::ResponseDone => Self::RESPONSE_DONE,
            Self::ResponseOutputItemAdded => Self::RESPONSE_OUTPUT_ITEM_ADDED,
            Self::ResponseOutputItemDone => Self::RESPONSE_OUTPUT_ITEM_DONE,
            Self::ResponseContentPartAdded => Self::RESPONSE_CONTENT_PART_ADDED,
            Self::ResponseContentPartDone => Self::RESPONSE_CONTENT_PART_DONE,
            Self::ResponseOutputTextDelta => Self::RESPONSE_OUTPUT_TEXT_DELTA,
            Self::ResponseOutputTextDone => Self::RESPONSE_OUTPUT_TEXT_DONE,
            Self::ResponseOutputAudioDelta => Self::RESPONSE_OUTPUT_AUDIO_DELTA,
            Self::ResponseOutputAudioDone => Self::RESPONSE_OUTPUT_AUDIO_DONE,
            Self::ResponseOutputAudioTranscriptDelta => {
                Self::RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA
            }
            Self::ResponseOutputAudioTranscriptDone => Self::RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE,
            Self::ResponseFunctionCallArgumentsDelta => {
                Self::RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA
            }
            Self::ResponseFunctionCallArgumentsDone => Self::RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE,
            Self::ResponseMcpCallArgumentsDelta => Self::RESPONSE_MCP_CALL_ARGUMENTS_DELTA,
            Self::ResponseMcpCallArgumentsDone => Self::RESPONSE_MCP_CALL_ARGUMENTS_DONE,
            Self::ResponseMcpCallInProgress => Self::RESPONSE_MCP_CALL_IN_PROGRESS,
            Self::ResponseMcpCallCompleted => Self::RESPONSE_MCP_CALL_COMPLETED,
            Self::ResponseMcpCallFailed => Self::RESPONSE_MCP_CALL_FAILED,
            Self::McpListToolsInProgress => Self::MCP_LIST_TOOLS_IN_PROGRESS,
            Self::McpListToolsCompleted => Self::MCP_LIST_TOOLS_COMPLETED,
            Self::McpListToolsFailed => Self::MCP_LIST_TOOLS_FAILED,
            Self::RateLimitsUpdated => Self::RATE_LIMITS_UPDATED,
            Self::Error => Self::ERROR,
        }
    }
}

impl fmt::Display for RealtimeServerEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Check if an event type string matches any response lifecycle event
pub fn is_response_event(event_type: &str) -> bool {
    matches!(
        event_type,
        ResponseEvent::CREATED
            | ResponseEvent::IN_PROGRESS
            | ResponseEvent::COMPLETED
            | ResponseEvent::INCOMPLETE
    )
}

/// Check if an item type string is a function call variant
pub fn is_function_call_type(item_type: &str) -> bool {
    item_type == ItemType::FUNCTION_CALL || item_type == ItemType::FUNCTION_TOOL_CALL
}
