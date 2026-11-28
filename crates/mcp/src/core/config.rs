//! MCP configuration types and utilities.
//!
//! Defines configuration structures for MCP servers, transports, proxies, and inventory.

use std::{collections::HashMap, fmt, time::Duration};

pub use rmcp::model::{Prompt, Resource, Tool};
use serde::{Deserialize, Serialize};

/// Default upper bound for a single tool call (seconds).
pub const DEFAULT_CALL_TIMEOUT_SECS: u64 = 120;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct McpConfig {
    /// Static MCP servers (loaded at startup)
    pub servers: Vec<McpServerConfig>,

    /// Connection pool settings
    #[serde(default)]
    pub pool: McpPoolConfig,

    /// Global MCP proxy configuration (default for all servers)
    /// Can be overridden per-server
    #[serde(default)]
    pub proxy: Option<McpProxyConfig>,

    /// Pre-warm these connections at startup
    #[serde(default)]
    pub warmup: Vec<WarmupServer>,

    /// Tool inventory refresh settings
    #[serde(default)]
    pub inventory: InventoryConfig,

    /// Approval policy configuration
    /// Default: allow all tools
    #[serde(default)]
    pub policy: PolicyConfig,
}

/// Policy configuration for tool approval decisions.
///
/// Evaluation order:
/// 1. Explicit tool policies (server:tool → decision)
/// 2. Server policies with trust levels
/// 3. Default policy (fallback)
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct PolicyConfig {
    /// Default policy when no other rules match.
    /// Default: "allow"
    #[serde(default)]
    pub default: PolicyDecisionConfig,

    /// Per-server policies with trust levels.
    #[serde(default)]
    pub servers: HashMap<String, ServerPolicyConfig>,

    /// Explicit per-tool policies (qualified name: "server:tool").
    #[serde(default)]
    pub tools: HashMap<String, PolicyDecisionConfig>,
}

/// Server-level policy configuration.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ServerPolicyConfig {
    /// Trust level for this server.
    /// - trusted: Allow all tools unconditionally
    /// - standard: Use default policy (default)
    /// - untrusted: Deny destructive operations
    /// - sandboxed: Only allow read-only, no external access
    #[serde(default)]
    pub trust_level: TrustLevelConfig,

    /// Default policy for tools on this server.
    #[serde(default)]
    pub default: PolicyDecisionConfig,
}

/// Trust level for an MCP server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevelConfig {
    Trusted,
    #[default]
    Standard,
    Untrusted,
    Sandboxed,
}

/// Policy decision configuration.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum PolicyDecisionConfig {
    #[default]
    Allow,
    Deny,
    /// Deny with a specific reason message.
    DenyWithReason(String),
}

impl Serialize for PolicyDecisionConfig {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            PolicyDecisionConfig::Allow => serializer.serialize_str("allow"),
            PolicyDecisionConfig::Deny => serializer.serialize_str("deny"),
            PolicyDecisionConfig::DenyWithReason(reason) => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("deny_with_reason", reason)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for PolicyDecisionConfig {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, MapAccess, Visitor};

        struct PolicyDecisionVisitor;

        impl<'de> Visitor<'de> for PolicyDecisionVisitor {
            type Value = PolicyDecisionConfig;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("\"allow\", \"deny\", or {\"deny_with_reason\": \"...\"}")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                match v {
                    "allow" => Ok(PolicyDecisionConfig::Allow),
                    "deny" => Ok(PolicyDecisionConfig::Deny),
                    _ => Err(E::unknown_variant(v, &["allow", "deny"])),
                }
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                if let Some(key) = map.next_key::<&str>()? {
                    if key == "deny_with_reason" {
                        let reason: String = map.next_value()?;
                        return Ok(PolicyDecisionConfig::DenyWithReason(reason));
                    }
                }
                Err(de::Error::custom("expected deny_with_reason key"))
            }
        }

        deserializer.deserialize_any(PolicyDecisionVisitor)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpServerConfig {
    pub name: String,
    #[serde(flatten)]
    pub transport: McpTransport,

    /// Per-server proxy: a mapping here replaces the global proxy for this
    /// server. A missing field and `null` both inherit the global proxy; to
    /// keep a server's host off the global proxy, name the host in the
    /// global proxy's `no_proxy` list.
    #[serde(default)]
    pub proxy: Option<McpProxyConfig>,

    /// Whether this server is required for router startup
    /// - true: Router startup fails if this server cannot be reached
    /// - false: Log warning but continue (default)
    #[serde(default)]
    pub required: bool,

    /// Tool-level configuration (aliases, response formats, arg mappings)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<HashMap<String, ToolConfig>>,

    /// Built-in tool type this server handles.
    ///
    /// When set, requests with `{"type": "web_search_preview"}` (or other built-in types)
    /// will be routed to this MCP server instead of being passed to the model.
    ///
    /// Example:
    /// ```yaml
    /// servers:
    ///   - name: brave
    ///     protocol: sse
    ///     url: "https://mcp.brave.com/sse"
    ///     builtin_type: web_search_preview
    ///     builtin_tool_name: brave_web_search
    /// ```
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub builtin_type: Option<BuiltinToolType>,

    /// Which tool on this server to call for the built-in type.
    ///
    /// Required when `builtin_type` is set. This is the actual MCP tool name
    /// to invoke (e.g., "brave_web_search", "execute", "search").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub builtin_tool_name: Option<String>,

    /// Whether tools from this server should be hidden from standard client-visible output.
    ///
    /// Limitation: `internal: true` does not currently redact or block live streaming
    /// Server-Sent Events (SSE) tool events. Internal tool activity may still appear in
    /// real-time streams until streaming redaction is implemented.
    #[serde(default)]
    pub internal: bool,
}

impl McpServerConfig {
    /// Validate the server configuration.
    ///
    /// Returns an error if:
    /// - `builtin_type` is set but `builtin_tool_name` is not
    /// - `builtin_tool_name` is set but `builtin_type` is not
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        match (&self.builtin_type, &self.builtin_tool_name) {
            (Some(_), None) => Err(ConfigValidationError::MissingBuiltinToolName {
                server: self.name.clone(),
            }),
            (None, Some(tool_name)) => Err(ConfigValidationError::MissingBuiltinType {
                server: self.name.clone(),
                tool_name: tool_name.clone(),
            }),
            _ => Ok(()),
        }
    }
}

/// Configuration validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigValidationError {
    /// `builtin_type` is set but `builtin_tool_name` is missing.
    MissingBuiltinToolName { server: String },
    /// `builtin_tool_name` is set but `builtin_type` is missing.
    MissingBuiltinType { server: String, tool_name: String },
    /// Multiple servers configured for the same `builtin_type`.
    DuplicateBuiltinType {
        builtin_type: BuiltinToolType,
        first_server: String,
        second_server: String,
    },
}

impl fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigValidationError::MissingBuiltinToolName { server } => {
                write!(
                    f,
                    "server '{server}': builtin_type is set but builtin_tool_name is missing"
                )
            }
            ConfigValidationError::MissingBuiltinType { server, tool_name } => {
                write!(
                    f,
                    "server '{server}': builtin_tool_name '{tool_name}' is set but builtin_type is missing"
                )
            }
            ConfigValidationError::DuplicateBuiltinType {
                builtin_type,
                first_server,
                second_server,
            } => {
                write!(
                    f,
                    "duplicate builtin_type '{builtin_type}': configured on both '{first_server}' and '{second_server}'"
                )
            }
        }
    }
}

impl std::error::Error for ConfigValidationError {}

/// Built-in tool types that can be routed to MCP servers.
///
/// When a request includes `{"type": "web_search_preview"}`, and an MCP server
/// is configured with `builtin_type: web_search_preview`, the gateway routes
/// the tool call to that MCP server instead of passing it to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinToolType {
    /// Web search tool (OpenAI: web_search_preview)
    WebSearchPreview,
    /// Code interpreter tool (OpenAI: code_interpreter)
    CodeInterpreter,
    /// File search tool (OpenAI: file_search)
    FileSearch,
    /// Image generation tool (OpenAI: image_generation)
    ImageGeneration,
}

impl BuiltinToolType {
    /// Every builtin tool type. The exhaustive `match` makes a new variant
    /// fail to compile until it is added here, so callers that enumerate all
    /// builtins (e.g. session builtin classification) cannot silently miss it.
    pub fn all() -> [BuiltinToolType; 4] {
        // The match exists solely for its exhaustiveness check.
        let _exhaustive = |t: BuiltinToolType| match t {
            BuiltinToolType::WebSearchPreview => (),
            BuiltinToolType::CodeInterpreter => (),
            BuiltinToolType::FileSearch => (),
            BuiltinToolType::ImageGeneration => (),
        };
        [
            BuiltinToolType::WebSearchPreview,
            BuiltinToolType::CodeInterpreter,
            BuiltinToolType::FileSearch,
            BuiltinToolType::ImageGeneration,
        ]
    }

    /// Get the corresponding response format for this built-in type.
    pub fn response_format(self) -> ResponseFormatConfig {
        match self {
            BuiltinToolType::WebSearchPreview => ResponseFormatConfig::WebSearchCall,
            BuiltinToolType::CodeInterpreter => ResponseFormatConfig::CodeInterpreterCall,
            BuiltinToolType::FileSearch => ResponseFormatConfig::FileSearchCall,
            BuiltinToolType::ImageGeneration => ResponseFormatConfig::ImageGenerationCall,
        }
    }
}

impl fmt::Display for BuiltinToolType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuiltinToolType::WebSearchPreview => write!(f, "web_search_preview"),
            BuiltinToolType::CodeInterpreter => write!(f, "code_interpreter"),
            BuiltinToolType::FileSearch => write!(f, "file_search"),
            BuiltinToolType::ImageGeneration => write!(f, "image_generation"),
        }
    }
}

/// Configuration for a specific tool on an MCP server.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ToolConfig {
    /// Optional alias name (e.g., "web_search" for "brave_web_search")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,

    /// Response format for transformation.
    ///
    /// `None` means "use whatever the surrounding context decides" — for a
    /// builtin-routed tool that becomes the builtin's hosted format, otherwise
    /// it falls through to `Passthrough`. `Some(Passthrough)` is an *explicit*
    /// passthrough request and overrides the builtin default. This distinction
    /// lets users add an `alias` or `arg_mapping` to a builtin tool without
    /// silently disabling its hosted-format wire shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormatConfig>,

    /// Argument mapping configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arg_mapping: Option<ArgMappingConfig>,
}

/// Response format configuration (mirrors ResponseFormat but for config).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatConfig {
    #[default]
    Passthrough,
    WebSearchCall,
    CodeInterpreterCall,
    FileSearchCall,
    ImageGenerationCall,
}

/// Argument mapping configuration for tool aliases.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ArgMappingConfig {
    /// Rename arguments: from -> to
    #[serde(default)]
    pub renames: HashMap<String, String>,

    /// Default values for arguments
    #[serde(default)]
    pub defaults: HashMap<String, serde_json::Value>,

    /// Values that override model-provided arguments
    #[serde(default)]
    pub overrides: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "protocol", rename_all = "lowercase")]
pub enum McpTransport {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        envs: HashMap<String, String>,
    },
    Sse {
        url: String,
        /// Bearer token for Authorization header
        #[serde(skip_serializing_if = "Option::is_none")]
        token: Option<String>,
        /// Additional headers (e.g., X-API-Key, custom auth)
        /// These affect connection identity and will be hashed for pool keying
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        headers: HashMap<String, String>,
    },
    Streamable {
        url: String,
        /// Bearer token for Authorization header
        #[serde(skip_serializing_if = "Option::is_none")]
        token: Option<String>,
        /// Additional headers (e.g., X-API-Key, custom auth)
        /// These affect connection identity and will be hashed for pool keying
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        headers: HashMap<String, String>,
    },
}

impl fmt::Debug for McpTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpTransport::Stdio {
                command,
                args,
                envs,
            } => f
                .debug_struct("Stdio")
                .field("command", command)
                .field("args", args)
                .field("envs", envs)
                .finish(),
            McpTransport::Sse {
                url,
                token,
                headers,
            } => f
                .debug_struct("Sse")
                .field("url", url)
                .field("token", &token.as_ref().map(|_| "****"))
                .field("headers", &format!("{} headers", headers.len()))
                .finish(),
            McpTransport::Streamable {
                url,
                token,
                headers,
            } => f
                .debug_struct("Streamable")
                .field("url", url)
                .field("token", &token.as_ref().map(|_| "****"))
                .field("headers", &format!("{} headers", headers.len()))
                .finish(),
        }
    }
}

/// MCP-specific proxy configuration (does NOT affect LLM API traffic)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpProxyConfig {
    /// HTTP proxy URL (e.g., `http://proxy.internal:8080`)
    pub http: Option<String>,

    /// HTTPS proxy URL
    pub https: Option<String>,

    /// Comma-separated hosts to exclude from proxying
    /// Example: "localhost,127.0.0.1,*.internal,10.*"
    pub no_proxy: Option<String>,

    /// Custom proxy authentication (if needed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
}

/// Connection pool configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpPoolConfig {
    /// Maximum cached connections per server URL
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Idle timeout before closing connection (seconds)
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout: u64,

    /// Upper bound for a single tool call (seconds).
    ///
    /// A call that exceeds it fails with an unknown outcome and is never
    /// retried, because the server may still have executed it. `0` disables
    /// the bound.
    #[serde(default = "default_call_timeout")]
    pub call_timeout: u64,
}

/// Tool inventory refresh configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InventoryConfig {
    /// Enable automatic tool inventory refresh
    #[serde(default = "default_true")]
    pub enable_refresh: bool,

    /// Tool cache TTL (seconds) - how long tools are considered fresh
    #[serde(default = "default_tool_ttl")]
    pub tool_ttl: u64,

    /// Background refresh interval (seconds) - proactive refresh
    #[serde(default = "default_refresh_interval")]
    pub refresh_interval: u64,

    /// Refresh on tool call failure (try refreshing if tool not found)
    #[serde(default = "default_true")]
    pub refresh_on_error: bool,
}

/// Pre-warm server connections at startup
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WarmupServer {
    /// Server URL
    pub url: String,

    /// Server label/name
    pub label: String,

    /// Optional authentication token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
}

// Default value functions
fn default_max_connections() -> usize {
    100
}

fn default_call_timeout() -> u64 {
    DEFAULT_CALL_TIMEOUT_SECS
}

fn default_idle_timeout() -> u64 {
    300 // 5 minutes
}

fn default_true() -> bool {
    true
}

fn default_tool_ttl() -> u64 {
    300 // 5 minutes
}

fn default_refresh_interval() -> u64 {
    60 // 1 minute
}

// Default implementations
impl Default for McpPoolConfig {
    fn default() -> Self {
        Self {
            max_connections: default_max_connections(),
            idle_timeout: default_idle_timeout(),
            call_timeout: default_call_timeout(),
        }
    }
}

impl Default for InventoryConfig {
    fn default() -> Self {
        Self {
            enable_refresh: true,
            tool_ttl: default_tool_ttl(),
            refresh_interval: default_refresh_interval(),
            refresh_on_error: true,
        }
    }
}

impl McpProxyConfig {
    /// Load proxy config from standard environment variables
    pub fn from_env() -> Option<Self> {
        let http = std::env::var("MCP_HTTP_PROXY")
            .ok()
            .or_else(|| std::env::var("HTTP_PROXY").ok());

        let https = std::env::var("MCP_HTTPS_PROXY")
            .ok()
            .or_else(|| std::env::var("HTTPS_PROXY").ok());

        let no_proxy = std::env::var("MCP_NO_PROXY")
            .ok()
            .or_else(|| std::env::var("NO_PROXY").ok());

        if http.is_some() || https.is_some() {
            Some(Self {
                http,
                https,
                no_proxy,
                username: None,
                password: None,
            })
        } else {
            None
        }
    }
}

impl McpConfig {
    /// Upper bound for a single tool call, or `None` when disabled.
    pub fn call_timeout(&self) -> Option<Duration> {
        (self.pool.call_timeout > 0).then(|| Duration::from_secs(self.pool.call_timeout))
    }

    /// Load configuration from a YAML file
    pub async fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = tokio::fs::read_to_string(path).await?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from environment variables (optional)
    pub fn from_env() -> Option<Self> {
        // This could be expanded to read from env vars
        // For now, return None to indicate env config not implemented
        None
    }

    /// Merge with environment-based proxy config
    pub fn with_env_proxy(mut self) -> Self {
        if self.proxy.is_none() {
            self.proxy = McpProxyConfig::from_env();
        }
        self
    }

    /// Validate the entire configuration.
    ///
    /// Checks:
    /// - Each server's builtin_type/builtin_tool_name pairing
    /// - No duplicate builtin_type across servers
    pub fn validate(&self) -> Result<(), ConfigValidationError> {
        let mut builtin_types: HashMap<BuiltinToolType, &str> = HashMap::new();

        for server in &self.servers {
            // Validate individual server config
            server.validate()?;

            // Check for duplicate builtin_type
            if let Some(builtin_type) = &server.builtin_type {
                if let Some(first_server) = builtin_types.get(builtin_type) {
                    return Err(ConfigValidationError::DuplicateBuiltinType {
                        builtin_type: *builtin_type,
                        first_server: (*first_server).to_string(),
                        second_server: server.name.clone(),
                    });
                }
                builtin_types.insert(*builtin_type, &server.name);
            }
        }

        Ok(())
    }
}

/// Server configurations the crate's tests build: an SSE server with the
/// defaults every test shares; a test that differs writes its difference as
/// a struct update over it.
#[cfg(test)]
#[path = "../../tests/unit/core/config/test_support.rs"]
pub(crate) mod test_support;

#[cfg(test)]
#[path = "../../tests/unit/core/config/tests.rs"]
mod tests;
