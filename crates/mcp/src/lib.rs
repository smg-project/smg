//! Model Context Protocol (MCP) client implementation.
//!
//! Crate is OpenAI-protocol-free; gateway-side adapter logic lives in
//! `model_gateway::routers::common::openai_bridge`.
//!
//! Modules:
//! - [`core`] — orchestrator, sessions, transports, config
//! - [`inventory`] — tool registry + qualified naming
//! - [`approval`] — interactive/policy approval engine
//! - [`annotations`], [`tenant`], [`error`] — shared cross-module types

pub mod annotations;
pub mod approval;
pub mod core;
pub mod error;
pub mod inventory;
pub mod tenant;
// Re-export from core
pub use core::{
    ArgMappingConfig, BuiltinToolType, ConfigValidationError, ConnectedServer,
    HandlerRequestContext, LatencySnapshot, McpConfig, McpMetrics, McpOrchestrator,
    McpRequestContext, McpServerBinding, McpServerConfig, McpToolSession, McpTransport,
    MetricsSnapshot, PendingToolExecution, PolicyConfig, PolicyDecisionConfig, PoolKey,
    RefreshRequest, ResponseFormatConfig, ServerPolicyConfig, SmgClientHandler, Tool, ToolConfig,
    ToolExecutionInput, ToolExecutionOutput, ToolExecutionResult, TrustLevelConfig,
    DEFAULT_SERVER_LABEL,
};

// Re-export shared types
pub use annotations::{AnnotationType, ToolAnnotations};
// Re-export from approval. Only `ApprovalMode` is surfaced at the crate root —
// production gateway code never instantiates `ApprovalManager`, `PolicyEngine`,
// `AuditLog`, etc. directly (those are reached through `McpOrchestrator`).
// The remaining symbols stay accessible via `smg_mcp::approval::*` for code
// that genuinely needs them, but they no longer pollute the flat root namespace.
pub use approval::ApprovalMode;
pub use error::{ApprovalError, McpError, McpResult};
// Re-export from inventory
pub use inventory::{
    AliasTarget, ArgMapping, QualifiedToolName, ToolCategory, ToolEntry, ToolInventory,
};
pub use tenant::{SessionId, TenantContext, TenantId};
