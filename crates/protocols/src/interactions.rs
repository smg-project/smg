// Gemini Interactions API types
// https://ai.google.dev/gemini-api/docs/interactions

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_with::skip_serializing_none;
use validator::{Validate, ValidationError};

use super::{
    common::{default_true, deserialize_null_as_false, invalid, Function, GenerationRequest},
    sampling_params::validate_top_p_value,
    validated::Normalizable,
};

// ============================================================================
// Request Type
// ============================================================================

#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
#[validate(schema(function = "validate_interactions_request"))]
pub struct InteractionsRequest {
    /// Model identifier (e.g., "gemini-2.5-flash")
    /// Required if agent is not provided
    pub model: Option<String>,

    /// Agent name (e.g., "deep-research-pro-preview-12-2025")
    /// Required if model is not provided
    pub agent: Option<String>,

    /// Input content - can be string or array of Content objects
    #[validate(custom(function = "validate_input"))]
    pub input: InteractionsInput,

    /// System instruction for the model
    pub system_instruction: Option<String>,

    /// Available tools
    #[validate(custom(function = "validate_tools"))]
    pub tools: Option<Vec<InteractionsTool>>,

    /// Response format for structured outputs
    pub response_format: Option<Value>,

    /// MIME type for the response (required if response_format is set)
    pub response_mime_type: Option<String>,

    /// Whether to stream the response
    #[serde(default, deserialize_with = "deserialize_null_as_false")]
    pub stream: bool,

    /// Whether to store the interaction (default: true)
    #[serde(default = "default_true")]
    pub store: bool,

    /// Run request in background (agents only)
    #[serde(default)]
    pub background: bool,

    /// Generation configuration
    #[validate(nested)]
    pub generation_config: Option<GenerationConfig>,

    /// Agent configuration (only applicable when agent is specified)
    pub agent_config: Option<AgentConfig>,

    /// Response modalities (text, image, audio)
    pub response_modalities: Option<Vec<ResponseModality>>,

    /// Link to prior interaction for stateful conversations
    pub previous_interaction_id: Option<String>,
}

impl Default for InteractionsRequest {
    fn default() -> Self {
        Self {
            model: None,
            agent: None,
            agent_config: None,
            input: InteractionsInput::Text(String::new()),
            system_instruction: None,
            previous_interaction_id: None,
            tools: None,
            generation_config: None,
            response_format: None,
            response_mime_type: None,
            response_modalities: None,
            stream: false,
            background: false,
            store: true,
        }
    }
}

impl Normalizable for InteractionsRequest {
    // Use default no-op implementation
}

impl GenerationRequest for InteractionsRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn extract_text_for_routing(&self) -> String {
        fn extract_from_content(content: &Content) -> Option<String> {
            match content {
                Content::Text { text, .. } => text.clone(),
                _ => None,
            }
        }

        fn extract_from_turn(turn: &Turn) -> String {
            match &turn.content {
                Some(TurnContent::Text(text)) => text.clone(),
                Some(TurnContent::Contents(contents)) => contents
                    .iter()
                    .filter_map(extract_from_content)
                    .collect::<Vec<String>>()
                    .join(" "),
                None => String::new(),
            }
        }

        match &self.input {
            InteractionsInput::Text(text) => text.clone(),
            InteractionsInput::Content(content) => {
                extract_from_content(content).unwrap_or_default()
            }
            InteractionsInput::Contents(contents) => contents
                .iter()
                .filter_map(extract_from_content)
                .collect::<Vec<String>>()
                .join(" "),
            InteractionsInput::Turns(turns) => turns
                .iter()
                .map(extract_from_turn)
                .collect::<Vec<String>>()
                .join(" "),
        }
    }
}

// ============================================================================
// Response Type
// ============================================================================

#[skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Interaction {
    /// Object type, always "interaction"
    pub object: Option<String>,

    /// Model used
    pub model: Option<String>,

    /// Agent used
    pub agent: Option<String>,

    /// Interaction ID
    pub id: String,

    /// Interaction status
    pub status: InteractionsStatus,

    /// Creation timestamp (ISO 8601)
    pub created: Option<String>,

    /// Last update timestamp (ISO 8601)
    pub updated: Option<String>,

    /// Role of the interaction
    pub role: Option<String>,

    /// Output content
    pub outputs: Option<Vec<Content>>,

    /// Usage information
    pub usage: Option<InteractionsUsage>,

    /// Previous interaction ID for conversation threading
    pub previous_interaction_id: Option<String>,
}

impl Interaction {
    /// Check if the interaction is complete
    pub fn is_complete(&self) -> bool {
        matches!(self.status, InteractionsStatus::Completed)
    }

    /// Check if the interaction is in progress
    pub fn is_in_progress(&self) -> bool {
        matches!(self.status, InteractionsStatus::InProgress)
    }

    /// Check if the interaction failed
    pub fn is_failed(&self) -> bool {
        matches!(self.status, InteractionsStatus::Failed)
    }

    /// Check if the interaction requires action (tool execution)
    pub fn requires_action(&self) -> bool {
        matches!(self.status, InteractionsStatus::RequiresAction)
    }
}

// ============================================================================
// Streaming Event Types (SSE)
// ============================================================================

/// Server-Sent Event for Interactions API streaming
/// See: <https://ai.google.dev/api/interactions-api#streaming>
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "event_type")]
pub enum InteractionStreamEvent {
    /// Emitted when an interaction begins processing
    #[serde(rename = "interaction.start")]
    InteractionStart {
        /// The interaction object
        interaction: Option<Interaction>,
        /// Event ID for resuming streams
        event_id: Option<String>,
    },

    /// Emitted when an interaction completes
    #[serde(rename = "interaction.complete")]
    InteractionComplete {
        /// The interaction object
        interaction: Option<Interaction>,
        /// Event ID for resuming streams
        event_id: Option<String>,
    },

    /// Emitted when interaction status changes
    #[serde(rename = "interaction.status_update")]
    InteractionStatusUpdate {
        /// The interaction ID
        interaction_id: Option<String>,
        /// The new status
        status: Option<InteractionsStatus>,
        /// Event ID for resuming streams
        event_id: Option<String>,
    },

    /// Signals the beginning of a new content block
    #[serde(rename = "content.start")]
    ContentStart {
        /// Content block index in outputs array
        index: Option<u32>,
        /// The content object
        content: Option<Content>,
        /// Event ID for resuming streams
        event_id: Option<String>,
    },

    /// Streams incremental content updates
    #[serde(rename = "content.delta")]
    ContentDelta {
        /// Content block index in outputs array
        index: Option<u32>,
        /// Event ID for resuming streams
        event_id: Option<String>,
        /// The delta content
        delta: Option<Delta>,
    },

    /// Marks the end of a content block
    #[serde(rename = "content.stop")]
    ContentStop {
        /// Content block index in outputs array
        index: Option<u32>,
        /// Event ID for resuming streams
        event_id: Option<String>,
    },

    /// Error event
    #[serde(rename = "error")]
    Error {
        /// Error information
        error: Option<InteractionsError>,
        /// Event ID for resuming streams
        event_id: Option<String>,
    },
}

/// Delta content for streaming updates
/// See: <https://ai.google.dev/api/interactions-api#ContentDelta>
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Delta {
    /// Text delta
    Text {
        text: Option<String>,
        annotations: Option<Vec<Annotation>>,
    },
    /// Image delta
    Image {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<ImageMimeType>,
        resolution: Option<MediaResolution>,
    },
    /// Audio delta
    Audio {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<AudioMimeType>,
    },
    /// Document delta
    Document {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<DocumentMimeType>,
    },
    /// Video delta
    Video {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<VideoMimeType>,
        resolution: Option<MediaResolution>,
    },
    /// Thought summary delta
    ThoughtSummary {
        content: Option<ThoughtSummaryContent>,
    },
    /// Thought signature delta
    ThoughtSignature { signature: Option<String> },
    /// Function call delta
    FunctionCall {
        name: Option<String>,
        arguments: Option<String>,
        id: Option<String>,
    },
    /// Function result delta
    FunctionResult {
        name: Option<String>,
        is_error: Option<bool>,
        result: Option<Value>,
        call_id: Option<String>,
    },
    /// Code execution call delta
    CodeExecutionCall {
        arguments: Option<CodeExecutionArguments>,
        id: Option<String>,
    },
    /// Code execution result delta
    CodeExecutionResult {
        result: Option<String>,
        is_error: Option<bool>,
        signature: Option<String>,
        call_id: Option<String>,
    },
    /// URL context call delta
    UrlContextCall {
        arguments: Option<UrlContextArguments>,
        id: Option<String>,
    },
    /// URL context result delta
    UrlContextResult {
        signature: Option<String>,
        result: Option<Vec<UrlContextResultData>>,
        is_error: Option<bool>,
        call_id: Option<String>,
    },
    /// Google search call delta
    GoogleSearchCall {
        arguments: Option<GoogleSearchArguments>,
        id: Option<String>,
    },
    /// Google search result delta
    GoogleSearchResult {
        signature: Option<String>,
        result: Option<Vec<GoogleSearchResultData>>,
        is_error: Option<bool>,
        call_id: Option<String>,
    },
    /// File search call delta
    FileSearchCall { id: Option<String> },
    /// File search result delta
    FileSearchResult {
        result: Option<Vec<FileSearchResultData>>,
    },
    /// MCP server tool call delta
    McpServerToolCall {
        name: Option<String>,
        server_name: Option<String>,
        arguments: Option<Value>,
        id: Option<String>,
    },
    /// MCP server tool result delta
    McpServerToolResult {
        name: Option<String>,
        server_name: Option<String>,
        result: Option<Value>,
        call_id: Option<String>,
    },
}

/// Error information in streaming events
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InteractionsError {
    /// Error code
    pub code: Option<String>,
    /// Error message
    pub message: Option<String>,
}

// ============================================================================
// Query Parameters
// ============================================================================

/// Query parameters for GET /interactions/{id}
#[skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InteractionsGetParams {
    /// Whether to stream the response
    pub stream: Option<bool>,
    /// Last event ID for resuming a stream
    pub last_event_id: Option<String>,
    /// API version
    pub api_version: Option<String>,
}

/// Query parameters for DELETE /interactions/{id}
#[skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InteractionsDeleteParams {
    /// API version
    pub api_version: Option<String>,
}

/// Query parameters for POST /interactions/{id}/cancel
#[skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InteractionsCancelParams {
    /// API version
    pub api_version: Option<String>,
}

// ============================================================================
// Interaction Tools
// ============================================================================

/// Interaction tool types
/// See: <https://ai.google.dev/api/interactions-api#Resource:Tool>
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InteractionsTool {
    /// Function tool with function declaration
    Function(Function),
    /// Google Search built-in tool
    GoogleSearch {},
    /// Code Execution built-in tool
    CodeExecution {},
    /// URL Context built-in tool
    UrlContext {},
    /// MCP Server tool
    McpServer {
        name: Option<String>,
        url: Option<String>,
        headers: Option<HashMap<String, String>>,
        allowed_tools: Option<AllowedTools>,
    },
    /// File Search built-in tool
    FileSearch {
        /// Names of file search stores to search
        file_search_store_names: Option<Vec<String>>,
        /// Maximum number of results to return
        top_k: Option<u32>,
        /// Metadata filter for search
        metadata_filter: Option<String>,
    },
}

/// Allowed tools configuration for MCP server
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct AllowedTools {
    /// Tool choice mode: auto, any, none, or validated
    pub mode: Option<ToolChoiceType>,
    /// List of allowed tool names
    pub tools: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceType {
    Auto,
    Any,
    None,
    Validated,
}

// ============================================================================
// Generation Config (Gemini-specific)
// ============================================================================

#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
pub struct GenerationConfig {
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,

    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,

    pub seed: Option<i64>,

    #[validate(custom(function = "validate_stop_sequences"))]
    pub stop_sequences: Option<Vec<String>>,

    pub tool_choice: Option<ToolChoice>,

    pub thinking_level: Option<ThinkingLevel>,

    pub thinking_summaries: Option<ThinkingSummaries>,

    #[validate(range(min = 1))]
    pub max_output_tokens: Option<u32>,

    pub speech_config: Option<Vec<SpeechConfig>>,

    pub image_config: Option<ImageConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingSummaries {
    Auto,
    None,
}

/// Tool choice can be a simple mode or a detailed config
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Type(ToolChoiceType),
    Config(ToolChoiceConfig),
}

#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ToolChoiceConfig {
    pub allowed_tools: Option<AllowedTools>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SpeechConfig {
    pub voice: Option<String>,
    pub language: Option<String>,
    pub speaker: Option<String>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageConfig {
    pub aspect_ratio: Option<AspectRatio>,
    pub image_size: Option<ImageSize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum AspectRatio {
    #[serde(rename = "1:1")]
    Square,
    #[serde(rename = "2:3")]
    Portrait2x3,
    #[serde(rename = "3:2")]
    Landscape3x2,
    #[serde(rename = "3:4")]
    Portrait3x4,
    #[serde(rename = "4:3")]
    Landscape4x3,
    #[serde(rename = "4:5")]
    Portrait4x5,
    #[serde(rename = "5:4")]
    Landscape5x4,
    #[serde(rename = "9:16")]
    Portrait9x16,
    #[serde(rename = "16:9")]
    Landscape16x9,
    #[serde(rename = "21:9")]
    UltraWide,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ImageSize {
    #[serde(rename = "1K")]
    OneK,
    #[serde(rename = "2K")]
    TwoK,
    #[serde(rename = "4K")]
    FourK,
}

/// Agent configuration
/// See: <https://ai.google.dev/api/interactions-api#CreateInteraction-deep_research>
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentConfig {
    /// Dynamic agent configuration
    Dynamic {},
    /// Deep Research agent configuration
    #[serde(rename = "deep-research")]
    DeepResearch {
        /// Whether to include thought summaries ("auto" or "none")
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking_summaries: Option<ThinkingSummaries>,
    },
}

// ============================================================================
// Input/Output Types
// ============================================================================

/// Input can be Content, array of Content, array of Turn, or string
/// See: <https://ai.google.dev/api/interactions-api#request-body>
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InteractionsInput {
    /// Simple text input
    Text(String),
    /// Single content block
    Content(Content),
    /// Array of content blocks
    Contents(Vec<Content>),
    /// Array of turns (conversation history)
    Turns(Vec<Turn>),
}

/// A turn in a conversation with role and content
/// See: <https://ai.google.dev/api/interactions-api#Resource:Turn>
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Turn {
    /// Role: "user" or "model"
    pub role: Option<String>,
    /// Content can be array of Content or string
    pub content: Option<TurnContent>,
}

/// Turn content can be array of Content or a simple string
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TurnContent {
    Contents(Vec<Content>),
    Text(String),
}

/// Content is a polymorphic type representing different content types
/// See: <https://ai.google.dev/api/interactions-api#Resource:Content>
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    /// Text content
    Text {
        text: Option<String>,
        annotations: Option<Vec<Annotation>>,
    },

    /// Image content
    Image {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<ImageMimeType>,
        resolution: Option<MediaResolution>,
    },

    /// Audio content
    Audio {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<AudioMimeType>,
    },

    /// Document content (PDF)
    Document {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<DocumentMimeType>,
    },

    /// Video content
    Video {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<VideoMimeType>,
        resolution: Option<MediaResolution>,
    },

    /// Thought content
    Thought {
        signature: Option<String>,
        summary: Option<Vec<ThoughtSummaryContent>>,
    },

    /// Function call content
    FunctionCall {
        name: String,
        arguments: Value,
        id: String,
    },

    /// Function result content
    FunctionResult {
        name: Option<String>,
        is_error: Option<bool>,
        result: Value,
        call_id: String,
    },

    /// Code execution call content
    CodeExecutionCall {
        arguments: Option<CodeExecutionArguments>,
        id: Option<String>,
    },

    /// Code execution result content
    CodeExecutionResult {
        result: Option<String>,
        is_error: Option<bool>,
        signature: Option<String>,
        call_id: Option<String>,
    },

    /// URL context call content
    UrlContextCall {
        arguments: Option<UrlContextArguments>,
        id: Option<String>,
    },

    /// URL context result content
    UrlContextResult {
        signature: Option<String>,
        result: Option<Vec<UrlContextResultData>>,
        is_error: Option<bool>,
        call_id: Option<String>,
    },

    /// Google search call content
    GoogleSearchCall {
        arguments: Option<GoogleSearchArguments>,
        id: Option<String>,
    },

    /// Google search result content
    GoogleSearchResult {
        signature: Option<String>,
        result: Option<Vec<GoogleSearchResultData>>,
        is_error: Option<bool>,
        call_id: Option<String>,
    },

    /// File search call content
    FileSearchCall { id: Option<String> },

    /// File search result content
    FileSearchResult {
        result: Option<Vec<FileSearchResultData>>,
    },

    /// MCP server tool call content
    McpServerToolCall {
        name: String,
        server_name: String,
        arguments: Value,
        id: String,
    },

    /// MCP server tool result content
    McpServerToolResult {
        name: Option<String>,
        server_name: Option<String>,
        result: Value,
        call_id: String,
    },
}

/// Content types allowed in thought summary (text or image only)
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThoughtSummaryContent {
    /// Text content in thought summary
    Text {
        text: Option<String>,
        annotations: Option<Vec<Annotation>>,
    },
    /// Image content in thought summary
    Image {
        data: Option<String>,
        uri: Option<String>,
        mime_type: Option<ImageMimeType>,
        resolution: Option<MediaResolution>,
    },
}

/// Annotation for text content (citations)
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Annotation {
    /// Start of the attributed segment, measured in bytes
    pub start_index: Option<u32>,
    /// End of the attributed segment, exclusive
    pub end_index: Option<u32>,
    /// Source attributed for a portion of the text (URL, title, or other identifier)
    pub source: Option<String>,
}

/// Arguments for URL context call
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UrlContextArguments {
    /// The URLs to fetch
    pub urls: Option<Vec<String>>,
}

/// Result data for URL context result
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UrlContextResultData {
    /// The URL that was fetched
    pub url: Option<String>,
    /// The status of the URL retrieval
    pub status: Option<UrlContextStatus>,
}

/// Status of URL context retrieval
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum UrlContextStatus {
    Success,
    Error,
    Paywall,
    Unsafe,
}

/// Arguments for Google search call
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GoogleSearchArguments {
    /// Web search queries
    pub queries: Option<Vec<String>>,
}

/// Result data for Google search result
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GoogleSearchResultData {
    /// URI reference of the search result
    pub url: Option<String>,
    /// Title of the search result
    pub title: Option<String>,
    /// Web content snippet
    pub rendered_content: Option<String>,
}

/// Result data for file search result
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FileSearchResultData {
    /// Search result title
    pub title: Option<String>,
    /// Search result text
    pub text: Option<String>,
    /// Name of the file search store
    pub file_search_store: Option<String>,
}

/// Arguments for code execution call
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CodeExecutionArguments {
    /// Programming language (currently only Python is supported)
    pub language: Option<CodeExecutionLanguage>,
    /// The code to be executed
    pub code: Option<String>,
}

/// Supported languages for code execution
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CodeExecutionLanguage {
    Python,
}

/// Image/video resolution options
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MediaResolution {
    Low,
    Medium,
    High,
    UltraHigh,
}

/// Supported image MIME types
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ImageMimeType {
    #[serde(rename = "image/png")]
    Png,
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[serde(rename = "image/webp")]
    Webp,
    #[serde(rename = "image/heic")]
    Heic,
    #[serde(rename = "image/heif")]
    Heif,
}

impl ImageMimeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ImageMimeType::Png => "image/png",
            ImageMimeType::Jpeg => "image/jpeg",
            ImageMimeType::Webp => "image/webp",
            ImageMimeType::Heic => "image/heic",
            ImageMimeType::Heif => "image/heif",
        }
    }
}

/// Supported audio MIME types
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum AudioMimeType {
    #[serde(rename = "audio/wav")]
    Wav,
    #[serde(rename = "audio/mp3")]
    Mp3,
    #[serde(rename = "audio/aiff")]
    Aiff,
    #[serde(rename = "audio/aac")]
    Aac,
    #[serde(rename = "audio/ogg")]
    Ogg,
    #[serde(rename = "audio/flac")]
    Flac,
}

impl AudioMimeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioMimeType::Wav => "audio/wav",
            AudioMimeType::Mp3 => "audio/mp3",
            AudioMimeType::Aiff => "audio/aiff",
            AudioMimeType::Aac => "audio/aac",
            AudioMimeType::Ogg => "audio/ogg",
            AudioMimeType::Flac => "audio/flac",
        }
    }
}

/// Supported document MIME types
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum DocumentMimeType {
    #[serde(rename = "application/pdf")]
    Pdf,
}

impl DocumentMimeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DocumentMimeType::Pdf => "application/pdf",
        }
    }
}

/// Supported video MIME types
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum VideoMimeType {
    #[serde(rename = "video/mp4")]
    Mp4,
    #[serde(rename = "video/mpeg")]
    Mpeg,
    #[serde(rename = "video/mov")]
    Mov,
    #[serde(rename = "video/avi")]
    Avi,
    #[serde(rename = "video/x-flv")]
    Flv,
    #[serde(rename = "video/mpg")]
    Mpg,
    #[serde(rename = "video/webm")]
    Webm,
    #[serde(rename = "video/wmv")]
    Wmv,
    #[serde(rename = "video/3gpp")]
    ThreeGpp,
}

impl VideoMimeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            VideoMimeType::Mp4 => "video/mp4",
            VideoMimeType::Mpeg => "video/mpeg",
            VideoMimeType::Mov => "video/mov",
            VideoMimeType::Avi => "video/avi",
            VideoMimeType::Flv => "video/x-flv",
            VideoMimeType::Mpg => "video/mpg",
            VideoMimeType::Webm => "video/webm",
            VideoMimeType::Wmv => "video/wmv",
            VideoMimeType::ThreeGpp => "video/3gpp",
        }
    }
}

// ============================================================================
// Status Types
// ============================================================================

#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionsStatus {
    #[default]
    InProgress,
    RequiresAction,
    Completed,
    Failed,
    Cancelled,
}

// ============================================================================
// Usage Types
// ============================================================================

/// Token count by modality
#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModalityTokens {
    pub modality: Option<ResponseModality>,
    pub tokens: Option<u32>,
}

#[skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InteractionsUsage {
    pub total_input_tokens: Option<u32>,
    pub input_tokens_by_modality: Option<Vec<ModalityTokens>>,
    pub total_cached_tokens: Option<u32>,
    pub cached_tokens_by_modality: Option<Vec<ModalityTokens>>,
    pub total_output_tokens: Option<u32>,
    pub output_tokens_by_modality: Option<Vec<ModalityTokens>>,
    pub total_tool_use_tokens: Option<u32>,
    pub tool_use_tokens_by_modality: Option<Vec<ModalityTokens>>,
    pub total_thought_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseModality {
    Text,
    Image,
    Audio,
}

fn is_option_blank(v: Option<&String>) -> bool {
    v.map(|s| s.trim().is_empty()).unwrap_or(true)
}

fn validate_interactions_request(req: &InteractionsRequest) -> Result<(), ValidationError> {
    // Exactly one of model or agent must be provided
    if is_option_blank(req.model.as_ref()) && is_option_blank(req.agent.as_ref()) {
        return Err(ValidationError::new("model_or_agent_required"));
    }
    if !is_option_blank(req.model.as_ref()) && !is_option_blank(req.agent.as_ref()) {
        return Err(invalid(
            "model_and_agent_mutually_exclusive",
            "Cannot set both model and agent. Provide exactly one.".to_string(),
        ));
    }

    // response_mime_type is required when response_format is set
    if req.response_format.is_some() && is_option_blank(req.response_mime_type.as_ref()) {
        return Err(ValidationError::new("response_mime_type_required"));
    }

    // background mode is required for agent interactions, and only for agents
    if !is_option_blank(req.agent.as_ref()) && !req.background {
        return Err(invalid(
            "agent_requires_background",
            "Agent interactions require background mode to be enabled.".to_string(),
        ));
    }
    if !is_option_blank(req.model.as_ref()) && req.background {
        return Err(invalid(
            "background_requires_agent",
            "Background mode is only supported for agent interactions.".to_string(),
        ));
    }

    // background and stream are mutually exclusive
    if req.background && req.stream {
        return Err(invalid(
            "background_conflicts_with_stream",
            "Cannot set both background and stream to true.".to_string(),
        ));
    }
    Ok(())
}

fn validate_tools(tools: &[InteractionsTool]) -> Result<(), ValidationError> {
    // FileSearch tool is not supported yet
    if tools
        .iter()
        .any(|t| matches!(t, InteractionsTool::FileSearch { .. }))
    {
        return Err(ValidationError::new("file_search_tool_not_supported"));
    }
    Ok(())
}

fn validate_input(input: &InteractionsInput) -> Result<(), ValidationError> {
    // Reject empty input
    let empty_msg = match input {
        InteractionsInput::Text(s) if s.trim().is_empty() => Some("Input text cannot be empty"),
        InteractionsInput::Content(content) if is_content_empty(content) => {
            Some("Input content cannot be empty")
        }
        InteractionsInput::Contents(contents) if contents.is_empty() => {
            Some("Input content array cannot be empty")
        }
        InteractionsInput::Contents(contents) if contents.iter().any(is_content_empty) => {
            Some("Input content array contains empty content items")
        }
        InteractionsInput::Turns(turns) if turns.is_empty() => {
            Some("Input turns array cannot be empty")
        }
        InteractionsInput::Turns(turns) if turns.iter().any(is_turn_empty) => {
            Some("Input turns array contains empty turn items")
        }
        _ => None,
    };
    if let Some(msg) = empty_msg {
        return Err(invalid("input_cannot_be_empty", msg.to_string()));
    }

    // Reject unsupported file search content
    fn has_file_search_content(content: &Content) -> bool {
        matches!(
            content,
            Content::FileSearchCall { .. } | Content::FileSearchResult { .. }
        )
    }

    fn check_turn(turn: &Turn) -> bool {
        if let Some(content) = &turn.content {
            match content {
                TurnContent::Contents(contents) => contents.iter().any(has_file_search_content),
                TurnContent::Text(_) => false,
            }
        } else {
            false
        }
    }

    let has_file_search = match input {
        InteractionsInput::Text(_) => false,
        InteractionsInput::Content(content) => has_file_search_content(content),
        InteractionsInput::Contents(contents) => contents.iter().any(has_file_search_content),
        InteractionsInput::Turns(turns) => turns.iter().any(check_turn),
    };

    if has_file_search {
        return Err(ValidationError::new("file_search_content_not_supported"));
    }
    Ok(())
}

fn is_content_empty(content: &Content) -> bool {
    match content {
        Content::Text { text, .. } => is_option_blank(text.as_ref()),
        Content::Image { data, uri, .. }
        | Content::Audio { data, uri, .. }
        | Content::Document { data, uri, .. }
        | Content::Video { data, uri, .. } => {
            is_option_blank(data.as_ref()) && is_option_blank(uri.as_ref())
        }
        Content::CodeExecutionCall { id, .. }
        | Content::UrlContextCall { id, .. }
        | Content::GoogleSearchCall { id, .. } => is_option_blank(id.as_ref()),
        Content::CodeExecutionResult { call_id, .. }
        | Content::UrlContextResult { call_id, .. }
        | Content::GoogleSearchResult { call_id, .. } => is_option_blank(call_id.as_ref()),
        _ => false,
    }
}

fn is_turn_empty(turn: &Turn) -> bool {
    match &turn.content {
        None => true,
        Some(TurnContent::Text(s)) => s.trim().is_empty(),
        Some(TurnContent::Contents(contents)) => {
            contents.is_empty() || contents.iter().any(is_content_empty)
        }
    }
}

fn validate_stop_sequences(seqs: &[String]) -> Result<(), ValidationError> {
    if seqs.len() > 5 {
        return Err(invalid(
            "too_many_stop_sequences",
            "Maximum 5 stop sequences allowed".to_string(),
        ));
    }
    if seqs.iter().any(|s| s.trim().is_empty()) {
        return Err(invalid(
            "stop_sequences_cannot_be_empty",
            "Stop sequences cannot contain empty strings".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refusal(result: Result<(), ValidationError>) -> (String, String) {
        let err = result.expect_err("a refusal");
        (
            err.code.to_string(),
            err.message.as_deref().unwrap_or("").to_string(),
        )
    }

    /// Each cross-field refusal keeps its code and its message.
    #[test]
    fn the_cross_field_refusals_keep_their_codes_and_messages() {
        let agent = || InteractionsRequest {
            agent: Some("deep-research".to_string()),
            ..Default::default()
        };
        let cases = [
            (
                InteractionsRequest {
                    model: Some("m".to_string()),
                    agent: Some("a".to_string()),
                    ..Default::default()
                },
                "model_and_agent_mutually_exclusive",
                "Cannot set both model and agent. Provide exactly one.",
            ),
            (
                InteractionsRequest {
                    background: false,
                    ..agent()
                },
                "agent_requires_background",
                "Agent interactions require background mode to be enabled.",
            ),
            (
                InteractionsRequest {
                    model: Some("m".to_string()),
                    background: true,
                    ..Default::default()
                },
                "background_requires_agent",
                "Background mode is only supported for agent interactions.",
            ),
            (
                InteractionsRequest {
                    background: true,
                    stream: true,
                    ..agent()
                },
                "background_conflicts_with_stream",
                "Cannot set both background and stream to true.",
            ),
        ];
        for (request, code, message) in cases {
            assert_eq!(
                refusal(validate_interactions_request(&request)),
                (code.to_string(), message.to_string())
            );
        }
    }

    #[test]
    fn an_empty_input_is_refused_with_its_message() {
        assert_eq!(
            refusal(validate_input(&InteractionsInput::Text("  ".to_string()))),
            (
                "input_cannot_be_empty".to_string(),
                "Input text cannot be empty".to_string()
            )
        );
    }

    #[test]
    fn stop_sequences_are_refused_when_too_many_or_blank() {
        let six: Vec<String> = (0..6).map(|i| format!("s{i}")).collect();
        assert_eq!(
            refusal(validate_stop_sequences(&six)),
            (
                "too_many_stop_sequences".to_string(),
                "Maximum 5 stop sequences allowed".to_string()
            )
        );
        assert_eq!(
            refusal(validate_stop_sequences(&[
                "ok".to_string(),
                " ".to_string()
            ])),
            (
                "stop_sequences_cannot_be_empty".to_string(),
                "Stop sequences cannot contain empty strings".to_string()
            )
        );
    }
}
