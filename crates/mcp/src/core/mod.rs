//! Core MCP client infrastructure.

/// Sentinel value used as the server key when the actual server cannot be determined
/// (e.g. tool not found, ambiguous collision).
pub const UNKNOWN_SERVER_KEY: &str = "unknown";

pub mod config;
pub mod handler;
pub mod metrics;
pub mod orchestrator;
pub mod pool;
pub mod proxy;
pub mod reconnect;
pub mod session;

pub use config::{
    ArgMappingConfig, BuiltinToolType, ConfigValidationError, McpConfig, McpServerConfig,
    McpTransport, PolicyConfig, PolicyDecisionConfig, ResponseFormatConfig, ServerPolicyConfig,
    Tool, ToolConfig, TrustLevelConfig,
};
pub use handler::{HandlerRequestContext, RefreshRequest, SmgClientHandler};
pub use metrics::{LatencySnapshot, McpMetrics, MetricsSnapshot};
pub use orchestrator::{
    ConnectedServer, McpOrchestrator, McpRequestContext, PendingToolExecution, ToolExecutionInput,
    ToolExecutionOutput, ToolExecutionResult,
};
pub use pool::{McpConnectionPool, PoolKey};
pub use reconnect::ReconnectionManager;
pub use session::{McpServerBinding, McpToolSession, DEFAULT_SERVER_LABEL};
