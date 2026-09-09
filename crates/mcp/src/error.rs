//! MCP error types.
//!
//! Defines error variants for MCP operations including connection, tool execution,
//! configuration errors, and approval errors.

use thiserror::Error;

pub type McpResult<T> = Result<T, McpError>;

#[derive(Debug, Error)]
pub enum McpError {
    #[error("Server not found: {0}")]
    ServerNotFound(String),

    #[error("Server disconnected: {0}")]
    ServerDisconnected(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Tool name collision: '{tool_name}' exists on servers: {servers:?}")]
    ToolCollision {
        tool_name: String,
        servers: Vec<String>,
    },

    #[error("Transport error: {0}")]
    Transport(String),

    #[error("Tool execution failed: {0}")]
    ToolExecution(String),

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    #[error("Prompt not found: {0}")]
    PromptNotFound(String),

    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    #[error("Approval error: {0}")]
    Approval(#[from] ApprovalError),

    #[error("Server access denied: {0}")]
    ServerAccessDenied(String),

    #[error("Tool execution denied: {0}")]
    ToolDenied(String),

    #[error(
        "Tool call outcome unknown: server '{server}' disconnected while executing '{tool}'; \
         the call was not retried because the tool is not marked idempotent or read-only"
    )]
    OutcomeUnknown { server: String, tool: String },

    #[error(
        "Tool call timed out after {secs}s on server '{server}' while executing '{tool}'; \
         the outcome is unknown and the call was not retried"
    )]
    CallTimeout {
        server: String,
        tool: String,
        secs: u64,
    },

    #[error(transparent)]
    Sdk(#[from] Box<rmcp::RmcpError>),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Http(#[from] reqwest::Error),
}

/// Approval-specific errors.
#[derive(Debug, Error)]
pub enum ApprovalError {
    /// Approval request not found (already resolved or expired).
    #[error("Approval not found: {0}")]
    NotFound(String),

    /// Approval request already pending.
    #[error("Approval already pending: {0}")]
    AlreadyPending(String),

    /// Response channel was closed.
    #[error("Approval channel closed")]
    ChannelClosed,

    /// Approval request timed out.
    #[error("Approval timed out: {0}")]
    Timeout(String),

    /// Policy evaluation failed.
    #[error("Policy evaluation failed: {0}")]
    PolicyError(String),
}
