//! Anthropic Messages API protocol definitions
//!
//! This module provides Rust types for the Anthropic Messages API.
//! See: https://docs.anthropic.com/en/api/messages

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::Validate;

use crate::{common::GenerationRequest, validated::Normalizable};

// ============================================================================
// Request Types
// ============================================================================

/// Request to create a message using the Anthropic Messages API.
///
/// This is the main request type for `/v1/messages` endpoint.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, Validate, schemars::JsonSchema)]
#[validate(schema(function = "validate_message_request"))]
pub struct CreateMessageRequest {
    /// The model that will complete your prompt.
    #[validate(length(min = 1, message = "model field is required and cannot be empty"))]
    pub model: String,

    /// Input messages for the conversation.
    #[validate(length(min = 1, message = "messages array is required and cannot be empty"))]
    pub messages: Vec<InputMessage>,

    /// The maximum number of tokens to generate before stopping.
    #[validate(range(min = 1, message = "max_tokens must be greater than 0"))]
    pub max_tokens: u32,

    /// An object describing metadata about the request.
    pub metadata: Option<Metadata>,

    /// Service tier for the request (auto or standard_only).
    pub service_tier: Option<ServiceTier>,

    /// Custom text sequences that will cause the model to stop generating.
    pub stop_sequences: Option<Vec<String>>,

    /// Whether to incrementally stream the response using server-sent events.
    pub stream: Option<bool>,

    /// System prompt for providing context and instructions.
    pub system: Option<SystemContent>,

    /// Amount of randomness injected into the response (0.0 to 1.0).
    pub temperature: Option<f64>,

    /// Configuration for extended thinking.
    pub thinking: Option<ThinkingConfig>,

    /// How the model should use the provided tools.
    pub tool_choice: Option<ToolChoice>,

    /// Definitions of tools that the model may use.
    pub tools: Option<Vec<Tool>>,

    /// Only sample from the top K options for each subsequent token.
    pub top_k: Option<u32>,

    /// Use nucleus sampling.
    pub top_p: Option<f64>,

    // Beta features
    /// Container configuration for code execution (beta).
    pub container: Option<ContainerConfig>,

    /// MCP servers to be utilized in this request (beta).
    pub mcp_servers: Option<Vec<McpServerConfig>>,

    /// Request ID forwarded to the backend for log correlation (SGLang extension)
    pub rid: Option<String>,

    /// Additional fields not explicitly defined above (e.g. beta features like
    /// context_management, output_config). Captured and forwarded to backends.
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl Normalizable for CreateMessageRequest {
    // Use default no-op implementation
}

impl CreateMessageRequest {
    /// Check if the request is for streaming
    pub fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    /// Get the model name
    pub fn get_model(&self) -> &str {
        &self.model
    }

    /// Check if the request contains any `mcp_toolset` tool entries.
    pub fn has_mcp_toolset(&self) -> bool {
        self.tools
            .as_ref()
            .is_some_and(|tools| tools.iter().any(|t| matches!(t, Tool::McpToolset(_))))
    }

    /// Return MCP server configs if present and non-empty.
    pub fn mcp_server_configs(&self) -> Option<&[McpServerConfig]> {
        self.mcp_servers
            .as_deref()
            .filter(|servers| !servers.is_empty())
    }
}

impl GenerationRequest for CreateMessageRequest {
    fn rid(&self) -> Option<&str> {
        self.rid.as_deref()
    }

    fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        let mut buffer = String::new();
        let mut has_content = false;

        let push = |s: &str, has_content: &mut bool, buffer: &mut String| {
            if s.is_empty() {
                return;
            }
            if *has_content {
                buffer.push(' ');
            }
            buffer.push_str(s);
            *has_content = true;
        };

        if let Some(system) = &self.system {
            match system {
                SystemContent::String(s) => push(s, &mut has_content, &mut buffer),
                SystemContent::Blocks(blocks) => {
                    for block in blocks {
                        let SystemContentBlock::Text(text_block) = block;
                        push(&text_block.text, &mut has_content, &mut buffer);
                    }
                }
            }
        }

        for msg in &self.messages {
            match &msg.content {
                InputContent::String(s) => push(s, &mut has_content, &mut buffer),
                InputContent::Blocks(blocks) => {
                    for block in blocks {
                        if let InputContentBlock::Text(text_block) = block {
                            push(&text_block.text, &mut has_content, &mut buffer);
                        }
                    }
                }
            }
        }

        buffer
    }
}

impl Tool {
    fn matches_tool_choice_name(&self, name: &str) -> bool {
        match self {
            Self::Custom(tool) => tool.name == name,
            Self::ToolSearch(tool) => tool.name == name,
            Self::Bash(tool) => tool.name == name,
            Self::TextEditor(tool) => tool.name == name,
            Self::WebSearch(tool) => tool.name == name,
            Self::McpToolset(toolset) => {
                let default_enabled = toolset
                    .default_config
                    .as_ref()
                    .and_then(|config| config.enabled)
                    .unwrap_or(true);

                toolset
                    .configs
                    .as_ref()
                    .and_then(|configs| configs.get(name))
                    .and_then(|config| config.enabled)
                    .unwrap_or(default_enabled)
            }
        }
    }
}
/// Validate cross-field constraints for Messages API requests.
fn validate_message_request(req: &CreateMessageRequest) -> Result<(), validator::ValidationError> {
    if req.has_mcp_toolset() && req.mcp_server_configs().is_none() {
        let mut e = validator::ValidationError::new("mcp_servers_required");
        e.message = Some("mcp_servers is required when mcp_toolset tools are present".into());
        return Err(e);
    }

    let Some(tool_choice) = &req.tool_choice else {
        return Ok(());
    };

    let has_tools = req.tools.as_ref().is_some_and(|tools| !tools.is_empty());
    let requires_tools = !matches!(tool_choice, ToolChoice::None);

    if requires_tools && !has_tools {
        let mut e = validator::ValidationError::new("tool_choice_requires_tools");
        e.message = Some(
            "Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified."
                .into(),
        );
        return Err(e);
    }

    if let ToolChoice::Tool { name, .. } = tool_choice {
        let tool_exists = req
            .tools
            .as_ref()
            .is_some_and(|tools| tools.iter().any(|tool| tool.matches_tool_choice_name(name)));

        if !tool_exists {
            let mut e = validator::ValidationError::new("tool_choice_tool_not_found");
            e.message = Some(
                format!("Invalid value for 'tool_choice': tool '{name}' not found in 'tools'.")
                    .into(),
            );
            return Err(e);
        }
    }

    Ok(())
}

/// Request metadata
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Metadata {
    /// An external identifier for the user who is associated with the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// Service tier options
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    Auto,
    StandardOnly,
}

/// System content can be a string or an array of text blocks
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum SystemContent {
    String(String),
    Blocks(Vec<SystemContentBlock>),
}

/// System content block — wraps TextBlock with the required `type` discriminator
/// so it round-trips correctly through serialization.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemContentBlock {
    Text(TextBlock),
}

/// A single input message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct InputMessage {
    /// The role of the message sender (user or assistant)
    pub role: Role,

    /// The content of the message
    pub content: InputContent,
}

/// Role of a message sender
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    /// `system` role inside `messages[]`.
    ///
    /// The Anthropic Messages API carries the system prompt in the top-level
    /// `system` field, but some clients (e.g. Claude Code) additionally send a
    /// `system`-role message *in the array* — often mid-conversation. Accepting
    /// it (instead of 400ing) lets those requests through; the message is
    /// forwarded to the chat template **in place**, so backends that render
    /// `system` inline (e.g. GLM-4.5+/5.x) keep its position, while backends
    /// that only read a leading system (e.g. MiniMax-M2) handle it per their
    /// own template. See https://github.com/smg-project/smg/issues/1795
    System,
}

/// Input content can be a string or an array of content blocks
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum InputContent {
    String(String),
    Blocks(Vec<InputContentBlock>),
}

// ============================================================================
// Input Content Blocks
// ============================================================================

/// Input content block types
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContentBlock {
    /// Text content
    Text(TextBlock),
    /// Image content
    Image(ImageBlock),
    /// Document content
    Document(DocumentBlock),
    /// Tool use block (for assistant messages)
    ToolUse(ToolUseBlock),
    /// Tool result block (for user messages)
    ToolResult(ToolResultBlock),
    /// Thinking block
    Thinking(ThinkingBlock),
    /// Redacted thinking block
    RedactedThinking(RedactedThinkingBlock),
    /// Server tool use block
    ServerToolUse(ServerToolUseBlock),
    /// Search result block
    SearchResult(SearchResultBlock),
    /// Web search tool result block
    WebSearchToolResult(WebSearchToolResultBlock),
    /// Tool search tool result block
    ToolSearchToolResult(ToolSearchToolResultBlock),
    /// Tool reference block
    ToolReference(ToolReferenceBlock),
}

/// Text content block
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TextBlock {
    /// The text content
    pub text: String,

    /// Cache control for this block
    pub cache_control: Option<CacheControl>,

    /// Citations for this text block
    pub citations: Option<Vec<Citation>>,
}

/// Image content block
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImageBlock {
    /// The image source
    pub source: ImageSource,

    /// Cache control for this block
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Image source (base64 or URL)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

/// Document content block
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DocumentBlock {
    /// The document source
    pub source: DocumentSource,

    /// Cache control for this block
    pub cache_control: Option<CacheControl>,

    /// Optional title for the document
    pub title: Option<String>,

    /// Optional context for the document
    pub context: Option<String>,

    /// Citations configuration
    pub citations: Option<CitationsConfig>,
}

/// Document source (base64, text, or URL)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    Base64 { media_type: String, data: String },
    Text { data: String },
    Url { url: String },
    Content { content: Vec<InputContentBlock> },
}

/// Tool use block (in assistant messages)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ToolUseBlock {
    /// Unique identifier for this tool use
    pub id: String,

    /// Name of the tool being used
    pub name: String,

    /// Input arguments for the tool
    pub input: Value,

    /// Cache control for this block
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Tool result block (in user messages)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ToolResultBlock {
    /// The ID of the tool use this is a result for
    pub tool_use_id: String,

    /// The result content (string or blocks)
    pub content: Option<ToolResultContent>,

    /// Whether this result indicates an error
    pub is_error: Option<bool>,

    /// Cache control for this block
    pub cache_control: Option<CacheControl>,
}

/// Tool result content (string or blocks)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ToolResultContent {
    String(String),
    Blocks(Vec<ToolResultContentBlock>),
}

/// Content blocks allowed in tool results
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContentBlock {
    Text(TextBlock),
    Image(ImageBlock),
    Document(DocumentBlock),
    SearchResult(SearchResultBlock),
}

/// Thinking block
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ThinkingBlock {
    /// The thinking content
    pub thinking: String,

    /// Signature for the thinking block
    pub signature: String,
}

/// Redacted thinking block
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RedactedThinkingBlock {
    /// The encrypted/redacted data
    pub data: String,
}

/// Server tool use block
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ServerToolUseBlock {
    /// Unique identifier for this tool use
    pub id: String,

    /// Name of the server tool
    pub name: String,

    /// Input arguments for the tool
    pub input: Value,

    /// Cache control for this block
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Search result block
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SearchResultBlock {
    /// Source URL or identifier
    pub source: String,

    /// Title of the search result
    pub title: String,

    /// Content of the search result
    pub content: Vec<TextBlock>,

    /// Cache control for this block
    pub cache_control: Option<CacheControl>,

    /// Citations configuration
    pub citations: Option<CitationsConfig>,
}

/// Web search tool result block
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WebSearchToolResultBlock {
    /// The tool use ID this result is for
    pub tool_use_id: String,

    /// The search results or error
    pub content: WebSearchToolResultContent,

    /// Cache control for this block
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Web search tool result content
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum WebSearchToolResultContent {
    Results(Vec<WebSearchResultBlock>),
    Error(WebSearchToolResultError),
}

/// Web search result block
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WebSearchResultBlock {
    /// Title of the search result
    pub title: String,

    /// URL of the search result
    pub url: String,

    /// Encrypted content
    pub encrypted_content: String,

    /// Page age (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_age: Option<String>,
}

/// Web search tool result error
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WebSearchToolResultError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error_code: WebSearchToolResultErrorCode,
}

/// Web search tool result error codes
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchToolResultErrorCode {
    InvalidToolInput,
    Unavailable,
    MaxUsesExceeded,
    TooManyRequests,
    QueryTooLong,
}

/// Cache control configuration
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheControl {
    Ephemeral,
}

/// Citations configuration
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CitationsConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

/// Citation types
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[expect(
    clippy::enum_variant_names,
    reason = "variant names match the OpenAI API citation type discriminators (char_location, page_location, etc.)"
)]
pub enum Citation {
    CharLocation(CharLocationCitation),
    PageLocation(PageLocationCitation),
    ContentBlockLocation(ContentBlockLocationCitation),
    WebSearchResultLocation(WebSearchResultLocationCitation),
    SearchResultLocation(SearchResultLocationCitation),
}

/// Character location citation
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CharLocationCitation {
    pub cited_text: String,
    pub document_index: u32,
    pub document_title: Option<String>,
    pub start_char_index: u32,
    pub end_char_index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
}

/// Page location citation
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PageLocationCitation {
    pub cited_text: String,
    pub document_index: u32,
    pub document_title: Option<String>,
    pub start_page_number: u32,
    pub end_page_number: u32,
}

/// Content block location citation
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ContentBlockLocationCitation {
    pub cited_text: String,
    pub document_index: u32,
    pub document_title: Option<String>,
    pub start_block_index: u32,
    pub end_block_index: u32,
}

/// Web search result location citation
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WebSearchResultLocationCitation {
    pub cited_text: String,
    pub url: String,
    pub title: Option<String>,
    pub encrypted_index: String,
}

/// Search result location citation
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SearchResultLocationCitation {
    pub cited_text: String,
    pub search_result_index: u32,
    pub source: String,
    pub title: Option<String>,
    pub start_block_index: u32,
    pub end_block_index: u32,
}

// ============================================================================
// Tool Definitions
// ============================================================================

/// Tool definition
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
#[expect(
    clippy::enum_variant_names,
    reason = "ToolSearch matches Anthropic API naming"
)]
#[schemars(rename = "MessagesTool")]
pub enum Tool {
    /// MCP toolset definition
    McpToolset(McpToolset),
    /// Custom tool definition (must come before ToolSearch: CustomTool requires
    /// `input_schema` which acts as a discriminator — ToolSearchTool JSON lacks
    /// it and falls through, while CustomTool JSON with "type" would incorrectly
    /// match ToolSearchTool's less-restrictive shape if tried first)
    Custom(CustomTool),
    /// Tool search tool
    ToolSearch(ToolSearchTool),
    /// Bash tool (computer use)
    Bash(BashTool),
    /// Text editor tool (computer use)
    TextEditor(TextEditorTool),
    /// Web search tool
    WebSearch(WebSearchTool),
}

/// Custom tool definition
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CustomTool {
    /// Name of the tool
    pub name: String,

    /// Optional type (defaults to "custom")
    #[serde(rename = "type")]
    pub tool_type: Option<String>,

    /// Description of what this tool does
    pub description: Option<String>,

    /// JSON schema for the tool's input
    pub input_schema: InputSchema,

    /// Whether to defer loading this tool
    pub defer_loading: Option<bool>,

    /// Cache control for this tool
    pub cache_control: Option<CacheControl>,
}

/// JSON Schema for tool input
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct InputSchema {
    #[serde(rename = "type")]
    pub schema_type: String,

    pub properties: Option<HashMap<String, Value>>,

    pub required: Option<Vec<String>>,

    /// Additional properties can be stored here
    #[serde(flatten)]
    pub additional: HashMap<String, Value>,
}

/// Bash tool for computer use
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BashTool {
    #[serde(rename = "type")]
    pub tool_type: String, // "bash_20250124"

    pub name: String, // "bash"

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Text editor tool for computer use
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TextEditorTool {
    #[serde(rename = "type")]
    pub tool_type: String, // "text_editor_20250124", etc.

    pub name: String, // "str_replace_editor"

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Web search tool
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WebSearchTool {
    #[serde(rename = "type")]
    pub tool_type: String, // "web_search_20250305"

    pub name: String, // "web_search"

    pub allowed_domains: Option<Vec<String>>,

    pub blocked_domains: Option<Vec<String>>,

    pub max_uses: Option<u32>,

    pub user_location: Option<UserLocation>,

    pub cache_control: Option<CacheControl>,
}

/// User location for web search
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct UserLocation {
    #[serde(rename = "type")]
    pub location_type: String, // "approximate"

    pub city: Option<String>,

    pub region: Option<String>,

    pub country: Option<String>,

    pub timezone: Option<String>,
}

// ============================================================================
// Tool Choice
// ============================================================================

/// How the model should use the provided tools
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[schemars(rename = "MessagesToolChoice")]
pub enum ToolChoice {
    /// The model will automatically decide whether to use tools
    Auto {
        disable_parallel_tool_use: Option<bool>,
    },
    /// The model will use any available tools
    Any {
        disable_parallel_tool_use: Option<bool>,
    },
    /// The model will use the specified tool
    Tool {
        name: String,
        disable_parallel_tool_use: Option<bool>,
    },
    /// The model will not use tools
    None,
}

// ============================================================================
// Thinking Configuration
// ============================================================================

/// Configuration for extended thinking
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingConfig {
    /// Enable extended thinking
    Enabled {
        /// Budget in tokens for thinking (minimum 1024)
        budget_tokens: u32,
        /// How thinking content is returned in the response.
        display: Option<ThinkingDisplay>,
    },
    /// Disable extended thinking
    Disabled,
    /// Let the model decide when and how much to think. Required on Opus 4.7.
    Adaptive {
        /// How thinking content is returned in the response.
        /// Defaults vary by model (Opus 4.7 / Mythos default to `Omitted`; others to `Summarized`).
        display: Option<ThinkingDisplay>,
    },
}

/// How thinking content is returned in API responses
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingDisplay {
    /// Thinking blocks contain summarized reasoning text
    Summarized,
    /// Thinking blocks return empty text; signature still carries encrypted content
    Omitted,
}

// ============================================================================
// Response Types
// ============================================================================

/// Response message from the API
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Message {
    /// Unique object identifier
    pub id: String,

    /// Object type (always "message")
    #[serde(rename = "type")]
    pub message_type: String,

    /// Conversational role (always "assistant")
    pub role: String,

    /// Content generated by the model
    pub content: Vec<ContentBlock>,

    /// The model that generated the message
    pub model: String,

    /// The reason the model stopped generating
    pub stop_reason: Option<StopReason>,

    /// Which custom stop sequence was generated (if any)
    pub stop_sequence: Option<String>,

    /// Billing and rate-limit usage
    pub usage: Usage,
}

/// Output content block types
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<Citation>>,
    },
    /// Tool use by the model
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// Thinking content
    Thinking { thinking: String, signature: String },
    /// Redacted thinking content
    RedactedThinking { data: String },
    /// Server tool use
    ServerToolUse {
        id: String,
        name: String,
        input: Value,
    },
    /// Web search tool result
    WebSearchToolResult {
        tool_use_id: String,
        content: WebSearchToolResultContent,
    },
    /// Tool search tool result
    ToolSearchToolResult {
        tool_use_id: String,
        content: ToolSearchResultContent,
    },
    /// Tool reference (returned by tool search)
    ToolReference {
        tool_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    /// MCP tool use (beta) - model requesting tool execution via MCP
    McpToolUse {
        id: String,
        name: String,
        server_name: String,
        input: Value,
    },
    /// MCP tool result (beta) - result from MCP tool execution
    McpToolResult {
        tool_use_id: String,
        content: Option<ToolResultContent>,
        is_error: Option<bool>,
    },
}

/// Stop reasons
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model reached a natural stopping point
    EndTurn,
    /// We exceeded the requested max_tokens
    MaxTokens,
    /// One of the custom stop_sequences was generated
    StopSequence,
    /// The model invoked one or more tools
    ToolUse,
    /// We paused a long-running turn
    PauseTurn,
    /// Streaming classifiers intervened
    Refusal,
}

/// Billing and rate-limit usage
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[schemars(rename = "MessagesUsage")]
pub struct Usage {
    /// The number of input tokens used
    pub input_tokens: u32,

    /// The number of output tokens used
    pub output_tokens: u32,

    /// The number of input tokens used to create the cache entry
    pub cache_creation_input_tokens: Option<u32>,

    /// The number of input tokens read from the cache
    pub cache_read_input_tokens: Option<u32>,

    /// Breakdown of cached tokens by TTL
    pub cache_creation: Option<CacheCreation>,

    /// Server tool usage information
    pub server_tool_use: Option<ServerToolUsage>,

    /// Service tier used for the request
    pub service_tier: Option<String>,
}

/// Cache creation breakdown
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CacheCreation {
    #[serde(flatten)]
    pub tokens_by_ttl: HashMap<String, u32>,
}

/// Server tool usage information
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ServerToolUsage {
    pub web_search_requests: u32,
}

// ============================================================================
// Streaming Event Types
// ============================================================================

/// Server-sent event wrapper
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageStreamEvent {
    /// Start of a new message
    MessageStart { message: Message },
    /// Update to a message
    MessageDelta {
        delta: MessageDelta,
        usage: MessageDeltaUsage,
    },
    /// End of a message
    MessageStop,
    /// Start of a content block
    ContentBlockStart {
        index: u32,
        content_block: ContentBlock,
    },
    /// Update to a content block
    ContentBlockDelta {
        index: u32,
        delta: ContentBlockDelta,
    },
    /// End of a content block
    ContentBlockStop { index: u32 },
    /// Ping event (for keep-alive)
    Ping,
    /// Error event
    Error { error: ErrorResponse },
}

/// Message delta for streaming updates
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MessageDelta {
    pub stop_reason: Option<StopReason>,

    pub stop_sequence: Option<String>,
}

/// Usage delta for streaming updates
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MessageDeltaUsage {
    pub output_tokens: u32,

    pub input_tokens: Option<u32>,

    pub cache_creation_input_tokens: Option<u32>,

    pub cache_read_input_tokens: Option<u32>,

    pub server_tool_use: Option<ServerToolUsage>,
}

/// Content block delta for streaming updates
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[expect(
    clippy::enum_variant_names,
    reason = "variant names match the OpenAI/Anthropic streaming delta type discriminators (text_delta, input_json_delta, etc.)"
)]
pub enum ContentBlockDelta {
    /// Text delta
    TextDelta { text: String },
    /// JSON input delta (for tool use)
    InputJsonDelta { partial_json: String },
    /// Thinking delta
    ThinkingDelta { thinking: String },
    /// Signature delta
    SignatureDelta { signature: String },
    /// Citations delta
    CitationsDelta { citation: Citation },
}

// ============================================================================
// Error Types
// ============================================================================

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[schemars(rename = "MessagesErrorResponse")]
pub struct ErrorResponse {
    #[serde(rename = "type")]
    pub error_type: String,

    pub message: String,
}

/// API error types
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[expect(
    clippy::enum_variant_names,
    reason = "variant names match the OpenAI API error type discriminators (invalid_request_error, authentication_error, etc.)"
)]
pub enum ApiError {
    InvalidRequestError { message: String },
    AuthenticationError { message: String },
    BillingError { message: String },
    PermissionError { message: String },
    NotFoundError { message: String },
    RateLimitError { message: String },
    TimeoutError { message: String },
    ApiError { message: String },
    OverloadedError { message: String },
}

// ============================================================================
// Count Tokens Types
// ============================================================================

/// Request to count tokens in a message
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CountMessageTokensRequest {
    /// The model to use for token counting
    pub model: String,

    /// Input messages
    pub messages: Vec<InputMessage>,

    /// System prompt
    pub system: Option<SystemContent>,

    /// Thinking configuration
    pub thinking: Option<ThinkingConfig>,

    /// Tool choice
    pub tool_choice: Option<ToolChoice>,

    /// Tool definitions
    pub tools: Option<Vec<Tool>>,
}

/// Response from token counting
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CountMessageTokensResponse {
    pub input_tokens: u32,
}

// ============================================================================
// Model Info Types
// ============================================================================

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ModelInfo {
    /// Object type (always "model")
    #[serde(rename = "type")]
    pub model_type: String,

    /// Model ID
    pub id: String,

    /// Display name
    pub display_name: String,

    /// When the model was created
    pub created_at: String,
}

/// List of models response
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ListModelsResponse {
    pub data: Vec<ModelInfo>,
    pub has_more: bool,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
}

// ============================================================================
// Beta Features - Container & MCP Configuration
// ============================================================================

/// Container configuration for code execution (beta)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ContainerConfig {
    /// Container ID for reuse across requests
    pub id: Option<String>,
}

/// MCP server configuration (beta)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpServerConfig {
    /// Server type (always "url")
    #[serde(rename = "type", default = "McpServerConfig::default_type")]
    pub server_type: String,

    /// Name of the MCP server
    pub name: String,

    /// MCP server URL
    pub url: String,

    /// Authorization token (if required)
    pub authorization_token: Option<String>,

    /// Tool configuration for this server
    pub tool_configuration: Option<McpToolConfiguration>,
}

impl McpServerConfig {
    fn default_type() -> String {
        "url".to_string()
    }
}

/// MCP tool configuration
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpToolConfiguration {
    /// Whether to allow all tools
    pub enabled: Option<bool>,

    /// Allowed tool names
    pub allowed_tools: Option<Vec<String>>,
}

// ============================================================================
// Beta Features - MCP Tool Types
// ============================================================================

/// MCP tool use block (beta) - for assistant messages
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpToolUseBlock {
    /// Unique identifier for this tool use
    pub id: String,

    /// Name of the tool being used
    pub name: String,

    /// Name of the MCP server
    pub server_name: String,

    /// Input arguments for the tool
    pub input: Value,

    /// Cache control for this block
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// MCP tool result block (beta) - for user messages
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpToolResultBlock {
    /// The ID of the tool use this is a result for
    pub tool_use_id: String,

    /// The result content
    pub content: Option<ToolResultContent>,

    /// Whether this result indicates an error
    pub is_error: Option<bool>,

    /// Cache control for this block
    pub cache_control: Option<CacheControl>,
}

/// MCP toolset definition (beta)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpToolset {
    #[serde(rename = "type")]
    pub toolset_type: String, // "mcp_toolset"

    /// Name of the MCP server to configure tools for
    pub mcp_server_name: String,

    /// Default configuration applied to all tools from this server
    pub default_config: Option<McpToolDefaultConfig>,

    /// Configuration overrides for specific tools
    pub configs: Option<HashMap<String, McpToolConfig>>,

    /// Cache control for this toolset
    pub cache_control: Option<CacheControl>,
}

/// Default configuration for MCP tools
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpToolDefaultConfig {
    /// Whether tools are enabled
    pub enabled: Option<bool>,

    /// Whether to defer loading
    pub defer_loading: Option<bool>,
}

/// Per-tool MCP configuration
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpToolConfig {
    /// Whether this tool is enabled
    pub enabled: Option<bool>,

    /// Whether to defer loading
    pub defer_loading: Option<bool>,
}

// ============================================================================
// Beta Features - Code Execution Types
// ============================================================================

/// Code execution tool (beta)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CodeExecutionTool {
    #[serde(rename = "type")]
    pub tool_type: String, // "code_execution_20250522" or "code_execution_20250825"

    pub name: String, // "code_execution"

    /// Allowed callers for this tool
    pub allowed_callers: Option<Vec<String>>,

    /// Whether to defer loading
    pub defer_loading: Option<bool>,

    /// Whether to use strict mode
    pub strict: Option<bool>,

    /// Cache control for this tool
    pub cache_control: Option<CacheControl>,
}

/// Code execution result block (beta)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CodeExecutionResultBlock {
    /// Stdout output
    pub stdout: String,

    /// Stderr output
    pub stderr: String,

    /// Return code
    pub return_code: i32,

    /// Output files
    pub content: Vec<CodeExecutionOutputBlock>,
}

/// Code execution output file reference
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CodeExecutionOutputBlock {
    #[serde(rename = "type")]
    pub block_type: String, // "code_execution_output"

    /// File ID
    pub file_id: String,
}

/// Code execution tool result block (beta)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CodeExecutionToolResultBlock {
    /// The ID of the tool use this is a result for
    pub tool_use_id: String,

    /// The result content (success or error)
    pub content: CodeExecutionToolResultContent,

    /// Cache control for this block
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Code execution tool result content
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum CodeExecutionToolResultContent {
    Success(CodeExecutionResultBlock),
    Error(CodeExecutionToolResultError),
}

/// Code execution tool result error
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CodeExecutionToolResultError {
    #[serde(rename = "type")]
    pub error_type: String, // "code_execution_tool_result_error"

    pub error_code: CodeExecutionToolResultErrorCode,
}

/// Code execution error codes
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CodeExecutionToolResultErrorCode {
    Unavailable,
    CodeExecutionExceededTimeout,
    ContainerExpired,
    InvalidToolInput,
}

/// Bash code execution result block (beta)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BashCodeExecutionResultBlock {
    /// Stdout output
    pub stdout: String,

    /// Stderr output
    pub stderr: String,

    /// Return code
    pub return_code: i32,

    /// Output files
    pub content: Vec<BashCodeExecutionOutputBlock>,
}

/// Bash code execution output file reference
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BashCodeExecutionOutputBlock {
    #[serde(rename = "type")]
    pub block_type: String, // "bash_code_execution_output"

    /// File ID
    pub file_id: String,
}

/// Bash code execution tool result block (beta)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BashCodeExecutionToolResultBlock {
    /// The ID of the tool use this is a result for
    pub tool_use_id: String,

    /// The result content (success or error)
    pub content: BashCodeExecutionToolResultContent,

    /// Cache control for this block
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Bash code execution tool result content
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum BashCodeExecutionToolResultContent {
    Success(BashCodeExecutionResultBlock),
    Error(BashCodeExecutionToolResultError),
}

/// Bash code execution tool result error
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BashCodeExecutionToolResultError {
    #[serde(rename = "type")]
    pub error_type: String, // "bash_code_execution_tool_result_error"

    pub error_code: BashCodeExecutionToolResultErrorCode,
}

/// Bash code execution error codes
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BashCodeExecutionToolResultErrorCode {
    Unavailable,
    CodeExecutionExceededTimeout,
    ContainerExpired,
    InvalidToolInput,
}

/// Text editor code execution tool result block (beta)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TextEditorCodeExecutionToolResultBlock {
    /// The ID of the tool use this is a result for
    pub tool_use_id: String,

    /// The result content
    pub content: TextEditorCodeExecutionToolResultContent,

    /// Cache control for this block
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Text editor code execution result content
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum TextEditorCodeExecutionToolResultContent {
    CreateResult(TextEditorCodeExecutionCreateResultBlock),
    StrReplaceResult(TextEditorCodeExecutionStrReplaceResultBlock),
    ViewResult(TextEditorCodeExecutionViewResultBlock),
    Error(TextEditorCodeExecutionToolResultError),
}

/// Text editor create result block
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TextEditorCodeExecutionCreateResultBlock {
    #[serde(rename = "type")]
    pub block_type: String, // "text_editor_code_execution_create_result"
}

/// Text editor str_replace result block
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TextEditorCodeExecutionStrReplaceResultBlock {
    #[serde(rename = "type")]
    pub block_type: String, // "text_editor_code_execution_str_replace_result"

    /// Snippet of content around the replacement
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

/// Text editor view result block
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TextEditorCodeExecutionViewResultBlock {
    #[serde(rename = "type")]
    pub block_type: String, // "text_editor_code_execution_view_result"

    /// Content of the viewed file
    pub content: String,
}

/// Text editor code execution tool result error
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TextEditorCodeExecutionToolResultError {
    #[serde(rename = "type")]
    pub error_type: String,

    pub error_code: TextEditorCodeExecutionToolResultErrorCode,
}

/// Text editor code execution error codes
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TextEditorCodeExecutionToolResultErrorCode {
    Unavailable,
    InvalidToolInput,
    FileNotFound,
    ContainerExpired,
}

// ============================================================================
// Beta Features - Web Fetch Types
// ============================================================================

/// Web fetch tool (beta)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WebFetchTool {
    #[serde(rename = "type")]
    pub tool_type: String, // "web_fetch_20250305" or similar

    pub name: String, // "web_fetch"

    /// Allowed callers for this tool
    pub allowed_callers: Option<Vec<String>>,

    /// Maximum number of uses
    pub max_uses: Option<u32>,

    /// Cache control for this tool
    pub cache_control: Option<CacheControl>,
}

/// Web fetch result block (beta)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WebFetchResultBlock {
    #[serde(rename = "type")]
    pub block_type: String, // "web_fetch_result"

    /// The URL that was fetched
    pub url: String,

    /// The document content
    pub content: DocumentBlock,

    /// When the content was retrieved
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieved_at: Option<String>,
}

/// Web fetch tool result block (beta)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WebFetchToolResultBlock {
    /// The ID of the tool use this is a result for
    pub tool_use_id: String,

    /// The result content (success or error)
    pub content: WebFetchToolResultContent,

    /// Cache control for this block
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Web fetch tool result content
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum WebFetchToolResultContent {
    Success(WebFetchResultBlock),
    Error(WebFetchToolResultError),
}

/// Web fetch tool result error
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WebFetchToolResultError {
    #[serde(rename = "type")]
    pub error_type: String, // "web_fetch_tool_result_error"

    pub error_code: WebFetchToolResultErrorCode,
}

/// Web fetch error codes
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WebFetchToolResultErrorCode {
    InvalidToolInput,
    Unavailable,
    MaxUsesExceeded,
    TooManyRequests,
    UrlNotAllowed,
    FetchFailed,
    ContentTooLarge,
}

// ============================================================================
// Beta Features - Tool Search Types
// ============================================================================

/// Tool search tool (beta)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ToolSearchTool {
    #[serde(rename = "type")]
    pub tool_type: String, // "tool_search_tool_regex" or "tool_search_tool_bm25"

    pub name: String,

    /// Allowed callers for this tool
    pub allowed_callers: Option<Vec<String>>,

    /// Cache control for this tool
    pub cache_control: Option<CacheControl>,
}

/// Tool reference block (beta) - returned by tool search
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ToolReferenceBlock {
    #[serde(rename = "type")]
    pub block_type: String, // "tool_reference"

    /// Tool name
    pub tool_name: String,

    /// Tool description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Tool search result content — wraps tool references returned by tool search
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ToolSearchResultContent {
    #[serde(rename = "type")]
    pub block_type: String, // "tool_search_tool_search_result"

    /// Tool references found by the search
    pub tool_references: Vec<ToolReferenceBlock>,
}

/// Tool search tool result block (beta)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ToolSearchToolResultBlock {
    /// The ID of the tool use this is a result for
    pub tool_use_id: String,

    /// The search results
    pub content: ToolSearchResultContent,

    /// Cache control for this block
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

// ============================================================================
// Beta Features - Container Upload Types
// ============================================================================

/// Container upload block (beta)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ContainerUploadBlock {
    #[serde(rename = "type")]
    pub block_type: String, // "container_upload"

    /// File ID
    pub file_id: String,

    /// File name
    pub file_name: String,

    /// File path in container
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
}

// ============================================================================
// Beta Features - Memory Tool Types
// ============================================================================

/// Memory tool (beta)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MemoryTool {
    #[serde(rename = "type")]
    pub tool_type: String, // "memory_20250818"

    pub name: String, // "memory"

    /// Allowed callers for this tool
    pub allowed_callers: Option<Vec<String>>,

    /// Whether to defer loading
    pub defer_loading: Option<bool>,

    /// Whether to use strict mode
    pub strict: Option<bool>,

    /// Input examples
    pub input_examples: Option<Vec<Value>>,

    /// Cache control for this tool
    pub cache_control: Option<CacheControl>,
}

// ============================================================================
// Beta Features - Computer Use Tool Types
// ============================================================================

/// Computer use tool (beta)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ComputerUseTool {
    #[serde(rename = "type")]
    pub tool_type: String, // "computer_20241022" or "computer_20250124"

    pub name: String, // "computer"

    /// Display width
    pub display_width_px: u32,

    /// Display height
    pub display_height_px: u32,

    /// Display number (optional)
    pub display_number: Option<u32>,

    /// Allowed callers for this tool
    pub allowed_callers: Option<Vec<String>>,

    /// Cache control for this tool
    pub cache_control: Option<CacheControl>,
}

// ============================================================================
// Beta Features - Extended Input Content Block Enum
// ============================================================================

/// Beta input content block types (extends InputContentBlock)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BetaInputContentBlock {
    // Standard types
    Text(TextBlock),
    Image(ImageBlock),
    Document(DocumentBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Thinking(ThinkingBlock),
    RedactedThinking(RedactedThinkingBlock),
    ServerToolUse(ServerToolUseBlock),
    SearchResult(SearchResultBlock),
    WebSearchToolResult(WebSearchToolResultBlock),

    // Beta MCP types
    McpToolUse(McpToolUseBlock),
    McpToolResult(McpToolResultBlock),

    // Beta code execution types
    CodeExecutionToolResult(CodeExecutionToolResultBlock),
    BashCodeExecutionToolResult(BashCodeExecutionToolResultBlock),
    TextEditorCodeExecutionToolResult(TextEditorCodeExecutionToolResultBlock),

    // Beta web fetch types
    WebFetchToolResult(WebFetchToolResultBlock),

    // Beta tool search types
    ToolSearchToolResult(ToolSearchToolResultBlock),
    ToolReference(ToolReferenceBlock),

    // Beta container types
    ContainerUpload(ContainerUploadBlock),
}

/// Beta output content block types (extends ContentBlock)
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BetaContentBlock {
    // Standard types
    Text {
        text: String,
        citations: Option<Vec<Citation>>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    ServerToolUse {
        id: String,
        name: String,
        input: Value,
    },
    WebSearchToolResult {
        tool_use_id: String,
        content: WebSearchToolResultContent,
    },

    // Beta MCP types
    McpToolUse {
        id: String,
        name: String,
        server_name: String,
        input: Value,
    },
    McpToolResult {
        tool_use_id: String,
        content: Option<ToolResultContent>,
        is_error: Option<bool>,
    },

    // Beta code execution types
    CodeExecutionToolResult {
        tool_use_id: String,
        content: CodeExecutionToolResultContent,
    },
    BashCodeExecutionToolResult {
        tool_use_id: String,
        content: BashCodeExecutionToolResultContent,
    },
    TextEditorCodeExecutionToolResult {
        tool_use_id: String,
        content: TextEditorCodeExecutionToolResultContent,
    },

    // Beta web fetch types
    WebFetchToolResult {
        tool_use_id: String,
        content: WebFetchToolResultContent,
    },

    // Beta tool search types
    ToolSearchToolResult {
        tool_use_id: String,
        content: ToolSearchResultContent,
    },
    ToolReference {
        tool_name: String,
        description: Option<String>,
    },

    // Beta container types
    ContainerUpload {
        file_id: String,
        file_name: String,
        file_path: Option<String>,
    },
}

/// Beta tool definition (extends Tool)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum BetaTool {
    // Standard tools
    Custom(CustomTool),
    Bash(BashTool),
    TextEditor(TextEditorTool),
    WebSearch(WebSearchTool),

    // Beta tools
    CodeExecution(CodeExecutionTool),
    McpToolset(McpToolset),
    WebFetch(WebFetchTool),
    ToolSearch(ToolSearchTool),
    Memory(MemoryTool),
    ComputerUse(ComputerUseTool),
}

/// Server tool names for beta features
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BetaServerToolName {
    WebSearch,
    WebFetch,
    CodeExecution,
    BashCodeExecution,
    TextEditorCodeExecution,
    ToolSearchToolRegex,
    ToolSearchToolBm25,
}

/// Server tool caller types (beta)
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerToolCaller {
    /// Direct caller (the model itself)
    Direct,
    /// Code execution caller
    #[serde(rename = "code_execution_20250825")]
    CodeExecution20250825,
}

#[cfg(test)]
mod tests {
    use serde_json::{self, json};

    use super::*;

    #[test]
    fn test_system_blocks_preserve_type_field() {
        let input = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "system prompt", "cache_control": {"type": "ephemeral"}}
            ]
        });

        let req: CreateMessageRequest = serde_json::from_value(input).expect("should deserialize");
        let reserialized = serde_json::to_value(&req).expect("should serialize");

        let system_blocks = reserialized.get("system").unwrap().as_array().unwrap();
        let first_block = &system_blocks[0];
        assert_eq!(
            first_block.get("type").and_then(|v| v.as_str()),
            Some("text"),
            "system block must retain 'type' field after round-trip: got {first_block:?}",
        );
    }

    #[test]
    fn test_message_content_blocks_preserve_type_field() {
        let input = json!({
            "model": "test",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
                ]
            }],
            "max_tokens": 100
        });

        let req: CreateMessageRequest = serde_json::from_value(input).expect("should deserialize");
        let reserialized = serde_json::to_value(&req).expect("should serialize");

        let msg = &reserialized["messages"][0];
        let content_blocks = msg["content"].as_array().unwrap();
        let first_block = &content_blocks[0];
        assert_eq!(
            first_block.get("type").and_then(|v| v.as_str()),
            Some("text"),
            "content block must retain 'type' field: got {first_block:?}",
        );
    }

    #[test]
    fn test_unknown_fields_preserved_via_flatten() {
        let input = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
            "thinking": {"type": "adaptive"},
            "context_management": {"edits": [{"type": "clear_thinking", "keep": "all"}]},
            "output_config": {"effort": "high"},
            "stream": true
        });

        let req: CreateMessageRequest =
            serde_json::from_value(input.clone()).expect("should deserialize");
        assert!(matches!(
            req.thinking,
            Some(ThinkingConfig::Adaptive { .. })
        ));

        let reserialized = serde_json::to_value(&req).expect("should serialize");
        assert_eq!(
            reserialized.get("context_management"),
            input.get("context_management"),
            "context_management must survive round-trip"
        );
        assert_eq!(
            reserialized.get("output_config"),
            input.get("output_config"),
            "output_config must survive round-trip"
        );
    }

    fn base_request() -> CreateMessageRequest {
        CreateMessageRequest {
            model: "claude-test".to_string(),
            messages: vec![InputMessage {
                role: Role::User,
                content: InputContent::String("hello".to_string()),
            }],
            max_tokens: 16,
            metadata: None,
            service_tier: None,
            stop_sequences: None,
            stream: None,
            system: None,
            temperature: None,
            thinking: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            container: None,
            mcp_servers: None,
            rid: None,
            other: Map::new(),
        }
    }

    fn custom_tool(name: &str) -> Tool {
        Tool::Custom(CustomTool {
            name: name.to_string(),
            tool_type: None,
            description: Some("test tool".to_string()),
            input_schema: InputSchema {
                schema_type: "object".to_string(),
                properties: None,
                required: None,
                additional: HashMap::new(),
            },
            defer_loading: None,
            cache_control: None,
        })
    }

    fn mcp_toolset(configs: Option<HashMap<String, McpToolConfig>>) -> Tool {
        Tool::McpToolset(McpToolset {
            toolset_type: "mcp_toolset".to_string(),
            mcp_server_name: "brave".to_string(),
            default_config: None,
            configs,
            cache_control: None,
        })
    }

    fn mcp_server_config() -> McpServerConfig {
        McpServerConfig {
            server_type: "url".to_string(),
            name: "brave".to_string(),
            url: "https://example.com/mcp".to_string(),
            authorization_token: None,
            tool_configuration: None,
        }
    }
    #[test]
    fn test_tool_mcp_toolset_defer_loading_deserialization() {
        let json = r#"{
            "type": "mcp_toolset",
            "mcp_server_name": "brave",
            "default_config": {"defer_loading": true}
        }"#;

        let tool: Tool = serde_json::from_str(json).expect("Failed to deserialize McpToolset Tool");
        match tool {
            Tool::McpToolset(ts) => {
                assert_eq!(ts.mcp_server_name, "brave");
                let default_config = ts.default_config.expect("default_config should be Some");
                assert_eq!(default_config.defer_loading, Some(true));
            }
            other => panic!(
                "Expected McpToolset, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn test_tool_search_tool_deserialization() {
        let json = r#"{
            "type": "tool_search_tool_regex_20251119",
            "name": "tool_search_tool_regex"
        }"#;

        let tool: Tool = serde_json::from_str(json).expect("Failed to deserialize ToolSearch Tool");
        match tool {
            Tool::ToolSearch(ts) => {
                assert_eq!(ts.name, "tool_search_tool_regex");
                assert_eq!(ts.tool_type, "tool_search_tool_regex_20251119");
            }
            other => panic!(
                "Expected ToolSearch, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn test_content_block_tool_search_tool_result_deserialization() {
        let json = r#"{
            "type": "tool_search_tool_result",
            "tool_use_id": "srvtoolu_015dw5iXvktXLmqwpyzo4Dp2",
            "content": {
                "type": "tool_search_tool_search_result",
                "tool_references": [
                    {"type": "tool_reference", "tool_name": "get_weather"}
                ]
            }
        }"#;

        let block: ContentBlock = serde_json::from_str(json)
            .expect("Failed to deserialize tool_search_tool_result ContentBlock");
        match block {
            ContentBlock::ToolSearchToolResult {
                tool_use_id,
                content,
            } => {
                assert_eq!(tool_use_id, "srvtoolu_015dw5iXvktXLmqwpyzo4Dp2");
                assert_eq!(content.tool_references.len(), 1);
                assert_eq!(content.tool_references[0].tool_name, "get_weather");
            }
            _ => panic!("Expected ToolSearchToolResult variant"),
        }
    }

    #[test]
    fn test_content_block_server_tool_use_deserialization() {
        let json = r#"{
            "type": "server_tool_use",
            "id": "srvtoolu_015dw5iXvktXLmqwpyzo4Dp2",
            "name": "tool_search_tool_regex",
            "input": {"query": "weather"}
        }"#;

        let block: ContentBlock =
            serde_json::from_str(json).expect("Failed to deserialize server_tool_use ContentBlock");
        match block {
            ContentBlock::ServerToolUse { id, name, input: _ } => {
                assert_eq!(id, "srvtoolu_015dw5iXvktXLmqwpyzo4Dp2");
                assert_eq!(name, "tool_search_tool_regex");
            }
            _ => panic!("Expected ServerToolUse variant"),
        }
    }

    #[test]
    fn test_content_block_tool_reference_deserialization() {
        let json = r#"{
            "type": "tool_reference",
            "tool_name": "get_weather",
            "description": "Get the weather for a location"
        }"#;

        let block: ContentBlock =
            serde_json::from_str(json).expect("Failed to deserialize tool_reference ContentBlock");
        match block {
            ContentBlock::ToolReference {
                tool_name,
                description,
            } => {
                assert_eq!(tool_name, "get_weather");
                assert_eq!(description.unwrap(), "Get the weather for a location");
            }
            _ => panic!("Expected ToolReference variant"),
        }
    }

    #[test]
    fn test_tool_choice_auto_requires_tools() {
        let mut request = base_request();
        request.tool_choice = Some(ToolChoice::Auto {
            disable_parallel_tool_use: None,
        });

        assert!(request.validate().is_err());
    }

    #[test]
    fn test_tool_choice_any_requires_tools() {
        let mut request = base_request();
        request.tool_choice = Some(ToolChoice::Any {
            disable_parallel_tool_use: None,
        });

        assert!(request.validate().is_err());
    }

    #[test]
    fn test_tool_choice_auto_with_tools_is_valid() {
        let mut request = base_request();
        request.tool_choice = Some(ToolChoice::Auto {
            disable_parallel_tool_use: None,
        });
        request.tools = Some(vec![custom_tool("get_weather")]);

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_tool_choice_any_with_tools_is_valid() {
        let mut request = base_request();
        request.tool_choice = Some(ToolChoice::Any {
            disable_parallel_tool_use: None,
        });
        request.tools = Some(vec![custom_tool("get_weather")]);

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_tool_choice_specific_tool_requires_tools() {
        let mut request = base_request();
        request.tool_choice = Some(ToolChoice::Tool {
            name: "get_weather".to_string(),
            disable_parallel_tool_use: None,
        });

        assert!(request.validate().is_err());
    }

    #[test]
    fn test_tool_choice_specific_tool_must_exist() {
        let mut request = base_request();
        request.tool_choice = Some(ToolChoice::Tool {
            name: "get_weather".to_string(),
            disable_parallel_tool_use: None,
        });
        request.tools = Some(vec![custom_tool("search_web")]);

        assert!(request.validate().is_err());
    }

    #[test]
    fn test_tool_choice_none_without_tools_is_valid() {
        let mut request = base_request();
        request.tool_choice = Some(ToolChoice::None);

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_tool_choice_specific_tool_is_valid_when_declared() {
        let mut request = base_request();
        request.tool_choice = Some(ToolChoice::Tool {
            name: "get_weather".to_string(),
            disable_parallel_tool_use: None,
        });
        request.tools = Some(vec![custom_tool("get_weather")]);

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_tool_choice_specific_tool_is_valid_with_mcp_toolset() {
        let mut request = base_request();
        request.tool_choice = Some(ToolChoice::Tool {
            name: "get_weather".to_string(),
            disable_parallel_tool_use: None,
        });
        request.tools = Some(vec![mcp_toolset(None)]);
        request.mcp_servers = Some(vec![mcp_server_config()]);

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_tool_choice_specific_tool_uses_mcp_toolset_default_when_override_missing() {
        let mut request = base_request();
        request.tool_choice = Some(ToolChoice::Tool {
            name: "get_weather".to_string(),
            disable_parallel_tool_use: None,
        });
        request.tools = Some(vec![mcp_toolset(Some(HashMap::from([(
            "search_web".to_string(),
            McpToolConfig {
                enabled: Some(false),
                defer_loading: None,
            },
        )])))]);
        request.mcp_servers = Some(vec![mcp_server_config()]);

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_tool_choice_specific_tool_must_be_enabled_in_mcp_toolset_configs() {
        let mut request = base_request();
        request.tool_choice = Some(ToolChoice::Tool {
            name: "get_weather".to_string(),
            disable_parallel_tool_use: None,
        });
        request.tools = Some(vec![mcp_toolset(Some(HashMap::from([(
            "get_weather".to_string(),
            McpToolConfig {
                enabled: Some(false),
                defer_loading: None,
            },
        )])))]);
        request.mcp_servers = Some(vec![mcp_server_config()]);

        assert!(request.validate().is_err());
    }

    #[test]
    fn test_thinking_config_adaptive_minimal() {
        let cfg: ThinkingConfig = serde_json::from_str(r#"{"type":"adaptive"}"#).unwrap();
        match cfg {
            ThinkingConfig::Adaptive { display } => assert_eq!(display, None),
            other => panic!("expected Adaptive, got {other:?}"),
        }
    }

    #[test]
    fn test_thinking_config_adaptive_with_display() {
        let cfg: ThinkingConfig =
            serde_json::from_str(r#"{"type":"adaptive","display":"omitted"}"#).unwrap();
        match cfg {
            ThinkingConfig::Adaptive { display } => {
                assert_eq!(display, Some(ThinkingDisplay::Omitted));
            }
            other => panic!("expected Adaptive, got {other:?}"),
        }

        let cfg: ThinkingConfig =
            serde_json::from_str(r#"{"type":"adaptive","display":"summarized"}"#).unwrap();
        match cfg {
            ThinkingConfig::Adaptive { display } => {
                assert_eq!(display, Some(ThinkingDisplay::Summarized));
            }
            other => panic!("expected Adaptive, got {other:?}"),
        }
    }

    #[test]
    fn test_thinking_config_adaptive_round_trip_omits_null_display() {
        let cfg = ThinkingConfig::Adaptive { display: None };
        let json = serde_json::to_string(&cfg).unwrap();
        assert_eq!(json, r#"{"type":"adaptive"}"#);
    }

    #[test]
    fn test_thinking_config_existing_variants_still_work() {
        let cfg: ThinkingConfig =
            serde_json::from_str(r#"{"type":"enabled","budget_tokens":1024}"#).unwrap();
        assert!(matches!(
            cfg,
            ThinkingConfig::Enabled {
                budget_tokens: 1024,
                display: None
            }
        ));

        let cfg: ThinkingConfig = serde_json::from_str(r#"{"type":"disabled"}"#).unwrap();
        assert!(matches!(cfg, ThinkingConfig::Disabled));
    }

    #[test]
    fn test_thinking_config_enabled_with_display() {
        let cfg: ThinkingConfig = serde_json::from_str(
            r#"{"type":"enabled","budget_tokens":2048,"display":"summarized"}"#,
        )
        .unwrap();
        match cfg {
            ThinkingConfig::Enabled {
                budget_tokens,
                display,
            } => {
                assert_eq!(budget_tokens, 2048);
                assert_eq!(display, Some(ThinkingDisplay::Summarized));
            }
            other => panic!("expected Enabled, got {other:?}"),
        }
    }

    #[test]
    fn test_thinking_config_enabled_round_trip_omits_null_display() {
        let cfg = ThinkingConfig::Enabled {
            budget_tokens: 1024,
            display: None,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert_eq!(json, r#"{"type":"enabled","budget_tokens":1024}"#);
    }

    #[test]
    fn test_full_message_with_tool_search_flow_deserialization() {
        // Simulates the full response from Anthropic API with tool search flow
        let json = r#"{
            "id": "msg_01TEST",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_015dw5iXvktXLmqwpyzo4Dp2",
                    "name": "tool_search_tool_regex",
                    "input": {"query": "weather"}
                },
                {
                    "type": "tool_search_tool_result",
                    "tool_use_id": "srvtoolu_015dw5iXvktXLmqwpyzo4Dp2",
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": [
                            {"type": "tool_reference", "tool_name": "get_weather"}
                        ]
                    }
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01ABC",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"}
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50
            }
        }"#;

        let msg: Message = serde_json::from_str(json)
            .expect("Failed to deserialize Message with tool search flow");
        assert_eq!(msg.content.len(), 3);
        assert!(matches!(msg.content[0], ContentBlock::ServerToolUse { .. }));
        assert!(matches!(
            msg.content[1],
            ContentBlock::ToolSearchToolResult { .. }
        ));
        assert!(matches!(msg.content[2], ContentBlock::ToolUse { .. }));
    }

    #[test]
    fn test_system_role_in_messages_is_accepted_and_preserved() {
        // Claude Code sends a `system`-role message in `messages[]` (in addition
        // to the top-level `system`). It must parse (not 400) and stay in place
        // so inline-`system` templates render it where it was sent.
        let body = json!({
            "model": "m",
            "max_tokens": 16,
            "system": "main prompt",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "mid-conversation system"}
            ]
        });
        let req: CreateMessageRequest = serde_json::from_value(body).unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[1].role, Role::System); // preserved in place
    }
}
