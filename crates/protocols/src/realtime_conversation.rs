// OpenAI Realtime Conversation API types
// https://platform.openai.com/docs/api-reference/realtime
//
// Session configuration and audio types live in `realtime_session`.
// Event type constants live in `event_types`.
// This module covers conversation items, content parts.

use serde::{Deserialize, Serialize};

use crate::common::Redacted;

// ============================================================================
// Conversation Item
// ============================================================================
/// A conversation item in the Realtime API.
///
/// Discriminated by the `type` field:
/// - `message`               — a text/audio/image message (system/user/assistant)
/// - `function_call`         — a function call issued by the model
/// - `function_call_output`  — the result supplied by the client
/// - `mcp_call`              — an MCP tool call issued by the model
/// - `mcp_list_tools`        — MCP list-tools result
/// - `mcp_approval_request`  — server asks client to approve an MCP call
/// - `mcp_approval_response` — client approves/denies an MCP call
///
/// # Content-part / role constraints
///
/// The OpenAI spec restricts which content-part types are valid per role:
/// - `system`    → `input_text` only
/// - `user`      → `input_text`, `input_audio`, `input_image`
/// - `assistant` → `output_text`, `output_audio`
///
/// Serde does not enforce this; these invariants are checked during request
/// validation, not at deserialization.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeConversationItem {
    Message {
        content: Vec<RealtimeContentPart>,
        role: ConversationItemRole,
        id: Option<String>,
        object: Option<ConversationItemObject>,
        status: Option<ConversationItemStatus>,
    },
    FunctionCall {
        arguments: String,
        name: String,
        id: Option<String>,
        call_id: Option<String>,
        object: Option<ConversationItemObject>,
        status: Option<ConversationItemStatus>,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
        id: Option<String>,
        object: Option<ConversationItemObject>,
        status: Option<ConversationItemStatus>,
    },
    McpApprovalResponse {
        id: String,
        approval_request_id: String,
        approve: bool,
        reason: Option<String>,
    },
    McpListTools {
        server_label: String,
        tools: Vec<McpListToolEntry>,
        id: Option<String>,
    },
    McpCall {
        id: String,
        arguments: String,
        name: String,
        server_label: String,
        approval_request_id: Option<String>,
        error: Option<McpCallError>,
        output: Option<String>,
    },
    McpApprovalRequest {
        id: String,
        arguments: String,
        name: String,
        server_label: String,
    },
}

/// Object type for conversation items. Always `"realtime.item"`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConversationItemObject {
    #[serde(rename = "realtime.item")]
    RealtimeItem,
}

/// Status of a conversation item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConversationItemStatus {
    Completed,
    Incomplete,
    InProgress,
}

/// Role for a conversation item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConversationItemRole {
    User,
    Assistant,
    System,
}

// ============================================================================
// Content Parts (Realtime-specific)
// ============================================================================

/// Content part inside a `RealtimeConversationItem::Message`.
///
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeContentPart {
    InputText {
        text: Option<String>,
    },
    InputAudio {
        audio: Option<Redacted>,
        transcript: Option<String>,
    },
    InputImage {
        detail: Option<ImageDetail>,
        image_url: Option<Redacted>,
    },
    OutputText {
        text: Option<String>,
    },
    OutputAudio {
        audio: Option<Redacted>,
        transcript: Option<String>,
    },
}

impl RealtimeContentPart {
    /// Returns the wire-format type name (e.g. `"input_text"`).
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::InputText { .. } => "input_text",
            Self::InputAudio { .. } => "input_audio",
            Self::InputImage { .. } => "input_image",
            Self::OutputText { .. } => "output_text",
            Self::OutputAudio { .. } => "output_audio",
        }
    }
}

/// Detail level for image processing. `auto` will default to `high`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    #[default]
    Auto,
    Low,
    High,
}

// ============================================================================
// MCP Types
// ============================================================================

/// A tool entry returned in `mcp_list_tools` conversation items.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpListToolEntry {
    pub input_schema: serde_json::Value,
    pub name: String,
    pub annotations: Option<serde_json::Value>,
    pub description: Option<String>,
}

/// Error from an MCP tool call.
///
/// One of: protocol error, tool execution error, or HTTP error.
#[expect(
    clippy::enum_variant_names,
    reason = "the variants carry the wire's type names, which all end in error"
)]
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum McpCallError {
    /// MCP protocol-level error.
    ProtocolError { code: i64, message: String },
    /// Error during tool execution on the MCP server.
    ToolExecutionError { message: String },
    /// HTTP-level error communicating with the MCP server.
    HttpError { code: i64, message: String },
}
