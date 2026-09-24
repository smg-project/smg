//! MCP configuration types and utilities.
//!
//! Defines configuration structures for MCP servers, transports, proxies, and inventory.

use std::{collections::HashMap, fmt, time::Duration};

pub use rmcp::model::{Prompt, RawResource, Tool};
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
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PolicyConfig {
    /// Default policy when no other rules match.
    /// Default: "allow"
    #[serde(default = "default_allow")]
    pub default: PolicyDecisionConfig,

    /// Per-server policies with trust levels.
    #[serde(default)]
    pub servers: HashMap<String, ServerPolicyConfig>,

    /// Explicit per-tool policies (qualified name: "server:tool").
    #[serde(default)]
    pub tools: HashMap<String, PolicyDecisionConfig>,
}

/// Server-level policy configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerPolicyConfig {
    /// Trust level for this server.
    /// - trusted: Allow all tools unconditionally
    /// - standard: Use default policy (default)
    /// - untrusted: Deny destructive operations
    /// - sandboxed: Only allow read-only, no external access
    #[serde(default)]
    pub trust_level: TrustLevelConfig,

    /// Default policy for tools on this server.
    #[serde(default = "default_allow")]
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

fn default_allow() -> PolicyDecisionConfig {
    PolicyDecisionConfig::Allow
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            default: PolicyDecisionConfig::Allow,
            servers: HashMap::new(),
            tools: HashMap::new(),
        }
    }
}

impl Default for ServerPolicyConfig {
    fn default() -> Self {
        Self {
            trust_level: TrustLevelConfig::Standard,
            default: PolicyDecisionConfig::Allow,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpServerConfig {
    pub name: String,
    #[serde(flatten)]
    pub transport: McpTransport,

    /// Per-server proxy override (overrides global proxy)
    /// Set to `null` in YAML to force direct connection (no proxy)
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
    /// HTTP proxy URL (e.g., "http://proxy.internal:8080")
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

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;

    #[test]
    fn test_call_timeout_default_override_and_disable() {
        let config: McpConfig = serde_yaml::from_str("servers: []").unwrap();
        assert_eq!(
            config.call_timeout(),
            Some(Duration::from_secs(DEFAULT_CALL_TIMEOUT_SECS))
        );

        let yaml = "servers: []
pool:
  call_timeout: 5";
        let config: McpConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.call_timeout(), Some(Duration::from_secs(5)));

        let yaml = "servers: []
pool:
  call_timeout: 0";
        let config: McpConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.call_timeout(), None);
    }

    #[test]
    fn test_default_pool_config() {
        let config = McpPoolConfig::default();
        assert_eq!(config.max_connections, 100);
        assert_eq!(config.idle_timeout, 300);
    }

    #[test]
    fn test_default_inventory_config() {
        let config = InventoryConfig::default();
        assert!(config.enable_refresh);
        assert_eq!(config.tool_ttl, 300);
        assert_eq!(config.refresh_interval, 60);
        assert!(config.refresh_on_error);
    }

    #[test]
    #[serial]
    fn test_proxy_from_env_empty() {
        // Ensure no proxy env vars are set for this test
        std::env::remove_var("MCP_HTTP_PROXY");
        std::env::remove_var("MCP_HTTPS_PROXY");
        std::env::remove_var("HTTP_PROXY");
        std::env::remove_var("HTTPS_PROXY");

        let proxy = McpProxyConfig::from_env();
        assert!(proxy.is_none(), "Should return None when no env vars set");
    }

    #[test]
    #[serial]
    fn test_proxy_from_env_with_vars() {
        std::env::set_var("MCP_HTTP_PROXY", "http://test-proxy:8080");
        std::env::set_var("MCP_NO_PROXY", "localhost,127.0.0.1");

        let proxy = McpProxyConfig::from_env();
        assert!(proxy.is_some(), "Should return Some when env vars set");

        let proxy = proxy.unwrap();
        assert_eq!(proxy.http.as_ref().unwrap(), "http://test-proxy:8080");
        assert_eq!(proxy.no_proxy.as_ref().unwrap(), "localhost,127.0.0.1");

        // Cleanup
        std::env::remove_var("MCP_HTTP_PROXY");
        std::env::remove_var("MCP_NO_PROXY");
    }

    #[tokio::test]
    async fn test_yaml_minimal_config() {
        let yaml = r#"
servers:
  - name: "test-server"
    protocol: sse
    url: "http://localhost:3000/sse"
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");
        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.servers[0].name, "test-server");
        assert!(!config.servers[0].required); // Should default to false
        assert!(config.servers[0].proxy.is_none()); // Should default to None
        assert_eq!(config.pool.max_connections, 100); // Should use default
        assert_eq!(config.inventory.tool_ttl, 300); // Should use default
    }

    #[tokio::test]
    async fn test_yaml_full_config() {
        let yaml = r#"
# Global proxy configuration
proxy:
  http: "http://global-proxy:8080"
  https: "http://global-proxy:8080"
  no_proxy: "localhost,127.0.0.1,*.internal"

# Connection pool settings
pool:
  max_connections: 50
  idle_timeout: 600

# Tool inventory settings
inventory:
  enable_refresh: true
  tool_ttl: 600
  refresh_interval: 120
  refresh_on_error: true

# Static servers
servers:
  - name: "required-server"
    protocol: sse
    url: "https://api.example.com/sse"
    token: "secret-token"
    required: true

  - name: "optional-server"
    protocol: stdio
    command: "mcp-server"
    args: ["--port", "3000"]
    required: false
    proxy:
      http: "http://server-specific-proxy:9090"

# Pre-warm connections
warmup:
  - url: "http://localhost:3000/sse"
    label: "local-dev"
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");

        // Check global proxy
        assert!(config.proxy.is_some());
        let global_proxy = config.proxy.as_ref().unwrap();
        assert_eq!(
            global_proxy.http.as_ref().unwrap(),
            "http://global-proxy:8080"
        );

        // Check pool config
        assert_eq!(config.pool.max_connections, 50);
        assert_eq!(config.pool.idle_timeout, 600);

        // Check inventory config
        assert_eq!(config.inventory.tool_ttl, 600);
        assert_eq!(config.inventory.refresh_interval, 120);

        // Check servers
        assert_eq!(config.servers.len(), 2);

        // Required server
        assert_eq!(config.servers[0].name, "required-server");
        assert!(config.servers[0].required);
        assert!(config.servers[0].proxy.is_none()); // Inherits global proxy

        // Optional server with custom proxy
        assert_eq!(config.servers[1].name, "optional-server");
        assert!(!config.servers[1].required);
        assert!(config.servers[1].proxy.is_some());
        assert_eq!(
            config.servers[1]
                .proxy
                .as_ref()
                .unwrap()
                .http
                .as_ref()
                .unwrap(),
            "http://server-specific-proxy:9090"
        );

        // Check warmup
        assert_eq!(config.warmup.len(), 1);
        assert_eq!(config.warmup[0].label, "local-dev");
    }

    #[tokio::test]
    async fn test_yaml_backward_compatibility() {
        // Old config format should still work
        let yaml = r#"
servers:
  - name: "legacy-server"
    protocol: sse
    url: "http://localhost:3000/sse"
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse old format");
        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.servers[0].name, "legacy-server");
        assert!(!config.servers[0].required); // New field defaults to false
        assert!(config.servers[0].proxy.is_none()); // New field defaults to None
        assert!(config.proxy.is_none()); // New field defaults to None
        assert!(config.warmup.is_empty()); // New field defaults to empty
    }

    #[tokio::test]
    async fn test_yaml_null_proxy_override() {
        // Test that explicit null in YAML sets proxy to None
        let yaml = r#"
proxy:
  http: "http://global-proxy:8080"

servers:
  - name: "direct-connection"
    protocol: sse
    url: "http://localhost:3000/sse"
    proxy: null
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");
        assert!(config.proxy.is_some()); // Global proxy set
        assert_eq!(config.servers.len(), 1);
        assert!(config.servers[0].proxy.is_none()); // Explicitly set to null
    }

    #[test]
    fn test_transport_stdio() {
        let yaml = r#"
name: "test"
protocol: stdio
command: "mcp-server"
args: ["--port", "3000"]
envs:
  VAR1: "value1"
  VAR2: "value2"
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse stdio");
        assert_eq!(config.name, "test");

        match config.transport {
            McpTransport::Stdio {
                command,
                args,
                envs,
            } => {
                assert_eq!(command, "mcp-server");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], "--port");
                assert_eq!(envs.get("VAR1").unwrap(), "value1");
            }
            _ => panic!("Expected Stdio transport"),
        }
    }

    #[test]
    fn test_transport_sse() {
        let yaml = r#"
name: "test"
protocol: sse
url: "http://localhost:3000/sse"
token: "secret"
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse sse");
        assert_eq!(config.name, "test");

        match config.transport {
            McpTransport::Sse { url, token, .. } => {
                assert_eq!(url, "http://localhost:3000/sse");
                assert_eq!(token.unwrap(), "secret");
            }
            _ => panic!("Expected Sse transport"),
        }
    }

    #[test]
    fn test_transport_streamable() {
        let yaml = r#"
name: "test"
protocol: streamable
url: "http://localhost:3000"
"#;

        let config: McpServerConfig =
            serde_yaml::from_str(yaml).expect("Failed to parse streamable");
        assert_eq!(config.name, "test");

        match config.transport {
            McpTransport::Streamable { url, token, .. } => {
                assert_eq!(url, "http://localhost:3000");
                assert!(token.is_none());
            }
            _ => panic!("Expected Streamable transport"),
        }
    }

    #[test]
    fn test_tool_config_with_alias() {
        let yaml = r#"
name: "brave"
protocol: sse
url: "https://mcp.brave.com/sse"
tools:
  brave_web_search:
    alias: web_search
    response_format: web_search_call
    arg_mapping:
      renames:
        q: query
      defaults:
        count: 10
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        assert_eq!(config.name, "brave");
        let tools = config.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        let tool_config = tools.get("brave_web_search").unwrap();
        assert_eq!(tool_config.alias, Some("web_search".to_string()));
        assert_eq!(
            tool_config.response_format,
            Some(ResponseFormatConfig::WebSearchCall)
        );

        let arg_mapping = tool_config.arg_mapping.as_ref().unwrap();
        assert_eq!(arg_mapping.renames.get("q").unwrap(), "query");
        assert_eq!(
            arg_mapping.defaults.get("count").unwrap(),
            &serde_json::json!(10)
        );
    }

    #[test]
    fn test_tool_config_arg_mapping_overrides() {
        let yaml = r#"
name: "web-search"
protocol: sse
url: "http://localhost:8080/sse"
tools:
  search_web:
    response_format: web_search_call
    arg_mapping:
      defaults:
        enable_source: true
      overrides:
        enable_brave: false
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        let tool_config = config
            .tools
            .as_ref()
            .and_then(|tools| tools.get("search_web"))
            .expect("search_web config should exist");

        let arg_mapping = tool_config.arg_mapping.as_ref().unwrap();
        assert_eq!(
            arg_mapping.defaults.get("enable_source").unwrap(),
            &serde_json::json!(true)
        );
        assert_eq!(
            arg_mapping.overrides.get("enable_brave").unwrap(),
            &serde_json::json!(false)
        );
    }

    #[test]
    fn test_tool_config_format_only() {
        let yaml = r#"
name: "filesystem"
protocol: stdio
command: "npx"
args: ["-y", "@anthropic/mcp-server-filesystem"]
tools:
  search:
    response_format: file_search_call
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        let tools = config.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 1);

        let tool_config = tools.get("search").unwrap();
        assert!(tool_config.alias.is_none());
        assert_eq!(
            tool_config.response_format,
            Some(ResponseFormatConfig::FileSearchCall)
        );
        assert!(tool_config.arg_mapping.is_none());
    }

    #[test]
    fn test_tool_config_defaults() {
        let yaml = r#"
name: "test"
protocol: sse
url: "http://localhost:3000/sse"
tools:
  my_tool: {}
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        let tools = config.tools.as_ref().unwrap();
        let tool_config = tools.get("my_tool").unwrap();
        assert!(tool_config.alias.is_none());
        // `my_tool: {}` carries no `response_format` field, so the deserialized
        // value is `None` (meaning "inherit context"), not an explicit
        // `Passthrough`.
        assert_eq!(tool_config.response_format, None);
        assert!(tool_config.arg_mapping.is_none());
    }

    #[test]
    fn test_response_format_config_serde() {
        let formats = vec![
            (ResponseFormatConfig::Passthrough, "\"passthrough\""),
            (ResponseFormatConfig::WebSearchCall, "\"web_search_call\""),
            (
                ResponseFormatConfig::CodeInterpreterCall,
                "\"code_interpreter_call\"",
            ),
            (ResponseFormatConfig::FileSearchCall, "\"file_search_call\""),
            (
                ResponseFormatConfig::ImageGenerationCall,
                "\"image_generation_call\"",
            ),
        ];

        for (format, expected) in formats {
            let serialized = serde_json::to_string(&format).unwrap();
            assert_eq!(serialized, expected);

            let deserialized: ResponseFormatConfig = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, format);
        }
    }

    #[test]
    fn test_multiple_tools_config() {
        let yaml = r#"
name: "multi-tool-server"
protocol: sse
url: "https://example.com/sse"
tools:
  tool_a:
    alias: a
    response_format: web_search_call
  tool_b:
    response_format: file_search_call
  tool_c:
    alias: c
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        let tools = config.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 3);

        let tool_a = tools.get("tool_a").unwrap();
        assert_eq!(tool_a.alias, Some("a".to_string()));
        assert_eq!(
            tool_a.response_format,
            Some(ResponseFormatConfig::WebSearchCall)
        );

        let tool_b = tools.get("tool_b").unwrap();
        assert!(tool_b.alias.is_none());
        assert_eq!(
            tool_b.response_format,
            Some(ResponseFormatConfig::FileSearchCall)
        );

        let tool_c = tools.get("tool_c").unwrap();
        assert_eq!(tool_c.alias, Some("c".to_string()));
        // Alias-only stanza: no `response_format` field → `None` (meaning
        // "inherit context"), not explicit Passthrough.
        assert_eq!(tool_c.response_format, None);
    }

    #[test]
    fn test_policy_config_default() {
        let config = PolicyConfig::default();
        assert_eq!(config.default, PolicyDecisionConfig::Allow);
        assert!(config.servers.is_empty());
        assert!(config.tools.is_empty());
    }

    #[test]
    fn test_policy_config_yaml_minimal() {
        let yaml = r"
servers: []
";

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        assert_eq!(config.policy.default, PolicyDecisionConfig::Allow);
        assert!(config.policy.servers.is_empty());
    }

    #[test]
    fn test_policy_config_yaml_full() {
        let yaml = r#"
servers:
  - name: "test"
    protocol: sse
    url: "http://localhost:3000/sse"

policy:
  default: allow
  servers:
    brave:
      trust_level: trusted
    untrusted_server:
      trust_level: untrusted
      default: deny
    sandbox_server:
      trust_level: sandboxed
  tools:
    "dangerous_server:delete_all": deny
    "risky_server:format_disk":
      deny_with_reason: "This operation is too dangerous"
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse");

        // Check default policy
        assert_eq!(config.policy.default, PolicyDecisionConfig::Allow);

        // Check server policies
        assert_eq!(config.policy.servers.len(), 3);

        let brave = config.policy.servers.get("brave").unwrap();
        assert_eq!(brave.trust_level, TrustLevelConfig::Trusted);
        assert_eq!(brave.default, PolicyDecisionConfig::Allow);

        let untrusted = config.policy.servers.get("untrusted_server").unwrap();
        assert_eq!(untrusted.trust_level, TrustLevelConfig::Untrusted);
        assert_eq!(untrusted.default, PolicyDecisionConfig::Deny);

        let sandbox = config.policy.servers.get("sandbox_server").unwrap();
        assert_eq!(sandbox.trust_level, TrustLevelConfig::Sandboxed);

        // Check tool policies
        assert_eq!(config.policy.tools.len(), 2);
        assert_eq!(
            config.policy.tools.get("dangerous_server:delete_all"),
            Some(&PolicyDecisionConfig::Deny)
        );
        assert_eq!(
            config.policy.tools.get("risky_server:format_disk"),
            Some(&PolicyDecisionConfig::DenyWithReason(
                "This operation is too dangerous".to_string()
            ))
        );
    }

    #[test]
    fn test_trust_level_config_serde() {
        let levels = vec![
            (TrustLevelConfig::Trusted, "\"trusted\""),
            (TrustLevelConfig::Standard, "\"standard\""),
            (TrustLevelConfig::Untrusted, "\"untrusted\""),
            (TrustLevelConfig::Sandboxed, "\"sandboxed\""),
        ];

        for (level, expected) in levels {
            let serialized = serde_json::to_string(&level).unwrap();
            assert_eq!(serialized, expected);

            let deserialized: TrustLevelConfig = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, level);
        }
    }

    #[test]
    fn test_policy_decision_config_serde() {
        let decisions = vec![
            (PolicyDecisionConfig::Allow, "\"allow\""),
            (PolicyDecisionConfig::Deny, "\"deny\""),
        ];

        for (decision, expected) in decisions {
            let serialized = serde_json::to_string(&decision).unwrap();
            assert_eq!(serialized, expected);

            let deserialized: PolicyDecisionConfig = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, decision);
        }

        // Test deny_with_reason
        let deny_with_reason = PolicyDecisionConfig::DenyWithReason("Not allowed".to_string());
        let serialized = serde_json::to_string(&deny_with_reason).unwrap();
        assert!(serialized.contains("deny_with_reason"));
        assert!(serialized.contains("Not allowed"));
    }

    #[test]
    fn test_builtin_tool_type_serde() {
        let types = vec![
            (BuiltinToolType::WebSearchPreview, "\"web_search_preview\""),
            (BuiltinToolType::CodeInterpreter, "\"code_interpreter\""),
            (BuiltinToolType::FileSearch, "\"file_search\""),
            (BuiltinToolType::ImageGeneration, "\"image_generation\""),
        ];

        for (builtin_type, expected) in types {
            let serialized = serde_json::to_string(&builtin_type).unwrap();
            assert_eq!(serialized, expected);

            let deserialized: BuiltinToolType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, builtin_type);
        }
    }

    #[test]
    fn test_builtin_tool_type_all_includes_every_variant() {
        let all = BuiltinToolType::all();
        for variant in [
            BuiltinToolType::WebSearchPreview,
            BuiltinToolType::CodeInterpreter,
            BuiltinToolType::FileSearch,
            BuiltinToolType::ImageGeneration,
        ] {
            assert!(all.contains(&variant), "all() missing {variant}");
        }
    }

    #[test]
    fn test_builtin_tool_type_response_format() {
        assert_eq!(
            BuiltinToolType::WebSearchPreview.response_format(),
            ResponseFormatConfig::WebSearchCall
        );
        assert_eq!(
            BuiltinToolType::CodeInterpreter.response_format(),
            ResponseFormatConfig::CodeInterpreterCall
        );
        assert_eq!(
            BuiltinToolType::FileSearch.response_format(),
            ResponseFormatConfig::FileSearchCall
        );
        assert_eq!(
            BuiltinToolType::ImageGeneration.response_format(),
            ResponseFormatConfig::ImageGenerationCall
        );
    }

    #[test]
    fn test_server_config_with_builtin_type() {
        let yaml = r#"
name: "brave-search"
protocol: sse
url: "https://mcp.brave.com/sse"
token: "${BRAVE_API_KEY}"
builtin_type: web_search_preview
builtin_tool_name: brave_web_search
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        assert_eq!(config.name, "brave-search");
        assert_eq!(config.builtin_type, Some(BuiltinToolType::WebSearchPreview));
        assert_eq!(
            config.builtin_tool_name,
            Some("brave_web_search".to_string())
        );
        assert!(!config.internal);
    }

    #[test]
    fn test_server_config_without_builtin_type() {
        let yaml = r#"
name: "regular-server"
protocol: sse
url: "http://localhost:3000/sse"
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        assert_eq!(config.name, "regular-server");
        assert!(config.builtin_type.is_none());
        assert!(config.builtin_tool_name.is_none());
        assert!(!config.internal);
    }

    #[test]
    fn test_server_config_with_internal_visibility() {
        let yaml = r#"
name: "internal-server"
protocol: sse
url: "http://localhost:3000/sse"
internal: true
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        assert_eq!(config.name, "internal-server");
        assert!(config.internal);
    }

    #[test]
    fn test_full_config_with_builtin_servers() {
        let yaml = r#"
servers:
  - name: brave
    protocol: sse
    url: "https://mcp.brave.com/sse"
    builtin_type: web_search_preview
    builtin_tool_name: brave_web_search
    tools:
      brave_web_search:
        response_format: web_search_call

  - name: code-runner
    protocol: stdio
    command: "code-interpreter"
    builtin_type: code_interpreter
    builtin_tool_name: execute

  - name: regular-mcp
    protocol: sse
    url: "http://localhost:3000/sse"
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
        assert_eq!(config.servers.len(), 3);

        // Brave with web_search_preview
        assert_eq!(
            config.servers[0].builtin_type,
            Some(BuiltinToolType::WebSearchPreview)
        );
        assert_eq!(
            config.servers[0].builtin_tool_name,
            Some("brave_web_search".to_string())
        );

        // Code interpreter
        assert_eq!(
            config.servers[1].builtin_type,
            Some(BuiltinToolType::CodeInterpreter)
        );
        assert_eq!(
            config.servers[1].builtin_tool_name,
            Some("execute".to_string())
        );

        // Regular MCP server (no builtin type)
        assert!(config.servers[2].builtin_type.is_none());
        assert!(config.servers[2].builtin_tool_name.is_none());
    }

    #[test]
    fn test_validate_valid_config_no_builtin() {
        let config = McpServerConfig {
            name: "test".to_string(),
            transport: McpTransport::Sse {
                url: "http://localhost:3000/sse".to_string(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: None,
            builtin_type: None,
            builtin_tool_name: None,
            internal: false,
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_valid_config_with_builtin() {
        let config = McpServerConfig {
            name: "brave".to_string(),
            transport: McpTransport::Sse {
                url: "http://localhost:3000/sse".to_string(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: None,
            builtin_type: Some(BuiltinToolType::WebSearchPreview),
            builtin_tool_name: Some("brave_web_search".to_string()),
            internal: false,
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_missing_builtin_tool_name() {
        let config = McpServerConfig {
            name: "brave".to_string(),
            transport: McpTransport::Sse {
                url: "http://localhost:3000/sse".to_string(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: None,
            builtin_type: Some(BuiltinToolType::WebSearchPreview),
            builtin_tool_name: None, // Missing!
            internal: false,
        };

        let err = config.validate().unwrap_err();
        assert!(matches!(
            err,
            ConfigValidationError::MissingBuiltinToolName { .. }
        ));
        assert!(err.to_string().contains("brave"));
        assert!(err.to_string().contains("builtin_tool_name is missing"));
    }

    #[test]
    fn test_validate_missing_builtin_type() {
        let config = McpServerConfig {
            name: "brave".to_string(),
            transport: McpTransport::Sse {
                url: "http://localhost:3000/sse".to_string(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: None,
            builtin_type: None, // Missing!
            builtin_tool_name: Some("brave_web_search".to_string()),
            internal: false,
        };

        let err = config.validate().unwrap_err();
        assert!(matches!(
            err,
            ConfigValidationError::MissingBuiltinType { .. }
        ));
        assert!(err.to_string().contains("brave"));
        assert!(err.to_string().contains("builtin_type is missing"));
    }

    #[test]
    fn test_mcp_config_validate_no_duplicates() {
        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "brave".to_string(),
                    transport: McpTransport::Sse {
                        url: "http://localhost:3000/sse".to_string(),
                        token: None,
                        headers: HashMap::new(),
                    },
                    proxy: None,
                    required: false,
                    tools: None,
                    builtin_type: Some(BuiltinToolType::WebSearchPreview),
                    builtin_tool_name: Some("search".to_string()),
                    internal: false,
                },
                McpServerConfig {
                    name: "code-runner".to_string(),
                    transport: McpTransport::Sse {
                        url: "http://localhost:3001/sse".to_string(),
                        token: None,
                        headers: HashMap::new(),
                    },
                    proxy: None,
                    required: false,
                    tools: None,
                    builtin_type: Some(BuiltinToolType::CodeInterpreter),
                    builtin_tool_name: Some("execute".to_string()),
                    internal: false,
                },
            ],
            ..Default::default()
        };

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_mcp_config_validate_duplicate_builtin_type() {
        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "brave1".to_string(),
                    transport: McpTransport::Sse {
                        url: "http://localhost:3000/sse".to_string(),
                        token: None,
                        headers: HashMap::new(),
                    },
                    proxy: None,
                    required: false,
                    tools: None,
                    builtin_type: Some(BuiltinToolType::WebSearchPreview),
                    builtin_tool_name: Some("search1".to_string()),
                    internal: false,
                },
                McpServerConfig {
                    name: "brave2".to_string(),
                    transport: McpTransport::Sse {
                        url: "http://localhost:3001/sse".to_string(),
                        token: None,
                        headers: HashMap::new(),
                    },
                    proxy: None,
                    required: false,
                    tools: None,
                    builtin_type: Some(BuiltinToolType::WebSearchPreview), // Duplicate!
                    builtin_tool_name: Some("search2".to_string()),
                    internal: false,
                },
            ],
            ..Default::default()
        };

        let err = config.validate().unwrap_err();
        assert!(matches!(
            err,
            ConfigValidationError::DuplicateBuiltinType { .. }
        ));
        assert!(err.to_string().contains("brave1"));
        assert!(err.to_string().contains("brave2"));
        assert!(err.to_string().contains("web_search_preview"));
    }

    #[test]
    fn test_builtin_tool_type_display() {
        assert_eq!(
            BuiltinToolType::WebSearchPreview.to_string(),
            "web_search_preview"
        );
        assert_eq!(
            BuiltinToolType::CodeInterpreter.to_string(),
            "code_interpreter"
        );
        assert_eq!(BuiltinToolType::FileSearch.to_string(), "file_search");
    }
}
