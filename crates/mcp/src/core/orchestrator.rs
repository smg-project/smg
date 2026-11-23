//! MCP Orchestrator - Main entry point for all MCP operations.
//!
//! `McpOrchestrator` coordinates between:
//! - Server connections (static from config, dynamic from requests)
//! - Tool inventory with qualified names and aliasing
//! - Approval manager (interactive + policy-only modes)
//! - Response transformation (MCP → OpenAI formats)
//! - Metrics and monitoring
//!
//! ## Usage
//!
//! ```ignore
//! // Initialize orchestrator
//! let orchestrator = McpOrchestrator::new(config).await?;
//!
//! // Create per-request context
//! let request_ctx = orchestrator.create_request_context(tenant_ctx);
//!
//! // Call a tool
//! let result = orchestrator.call_tool(
//!     "brave",           // server_key
//!     "web_search",      // tool_name
//!     json!({"query": "rust programming"}),
//!     "brave",           // server_label (user-facing)
//!     &request_ctx,
//! ).await?;
//! ```

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    future::Future,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use parking_lot::RwLock;
use rmcp::{
    model::{CallToolRequestParams, CallToolResult},
    service::{RunningService, ServiceError},
    RoleClient,
};
use rustls::crypto::{ring, CryptoProvider};
use serde_json::Value;
use tokio::{
    sync::mpsc,
    time::{error::Elapsed, Timeout},
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use super::{
    config::{
        ArgMappingConfig, BuiltinToolType, McpConfig, McpProxyConfig, McpServerConfig,
        McpTransport, Prompt, Resource,
    },
    handler::{HandlerRequestContext, RefreshRequest, SmgClientHandler},
    metrics::McpMetrics,
    pool::{Listing, McpConnectionPool, PoolKey, UrlLookup},
    reconnect::ReconnectionManager,
};
use crate::{
    approval::{
        audit::AuditLog, policy::PolicyEngine, ApprovalDecision, ApprovalManager, ApprovalMode,
        ApprovalOutcome, ApprovalParams, McpApprovalRequest,
    },
    error::{McpError, McpResult},
    inventory::{
        AliasTarget, ArgMapping, QualifiedToolName, ToolCategory, ToolEntry, ToolInventory,
        ALIAS_SERVER_KEY,
    },
    tenant::TenantContext,
};

/// A server's tools as inventory entries under `category`: the listing and
/// the entries; the failure's message, after one warning, when the server
/// cannot be listed (the caller keeps it as the server's listing error: a
/// static server's entry, a pooled connection and the request's binding).
async fn tools_of<S: rmcp::service::Service<RoleClient>>(
    client: &RunningService<RoleClient, S>,
    server_key: &str,
    category: ToolCategory,
) -> Result<Vec<ToolEntry>, String> {
    match client.peer().list_all_tools().await {
        Ok(tools) => {
            info!("Discovered {} tools from '{}'", tools.len(), server_key);
            Ok(tools
                .into_iter()
                .enumerate()
                .map(|(position, tool)| {
                    ToolEntry::from_server_tool(server_key, tool)
                        .with_category(category)
                        .with_position(position)
                })
                .collect())
        }
        Err(e) => {
            warn!("Failed to list tools from '{}': {}", server_key, e);
            Err(e.to_string())
        }
    }
}

/// A server's prompts by name: `None`, after a debug line, when the listing
/// failed (a server without prompts answers so).
async fn prompts_of<S: rmcp::service::Service<RoleClient>>(
    client: &RunningService<RoleClient, S>,
    server_key: &str,
) -> Option<Vec<(String, Prompt)>> {
    match client.peer().list_all_prompts().await {
        Ok(prompts) => {
            info!("Discovered {} prompts from '{}'", prompts.len(), server_key);
            Some(
                prompts
                    .into_iter()
                    .map(|prompt| (prompt.name.clone(), prompt))
                    .collect(),
            )
        }
        Err(e) => {
            debug!("No prompts from '{}': {}", server_key, e);
            None
        }
    }
}

/// A server's resources by uri: `None`, after a debug line, when the listing
/// failed (a server without resources answers so).
async fn resources_of<S: rmcp::service::Service<RoleClient>>(
    client: &RunningService<RoleClient, S>,
    server_key: &str,
) -> Option<Vec<(String, Resource)>> {
    match client.peer().list_all_resources().await {
        Ok(resources) => {
            info!(
                "Discovered {} resources from '{}'",
                resources.len(),
                server_key
            );
            Some(
                resources
                    .into_iter()
                    .map(|resource| (resource.uri.clone(), resource))
                    .collect(),
            )
        }
        Err(e) => {
            debug!("No resources from '{}': {}", server_key, e);
            None
        }
    }
}

/// The argument mapping a tool config sets.
fn arg_mapping_of(config: &ArgMappingConfig) -> ArgMapping {
    let mut mapping = ArgMapping::new();
    for (from, to) in &config.renames {
        mapping = mapping.with_rename(from, to);
    }
    for (key, value) in &config.defaults {
        mapping = mapping.with_default(key, value.clone());
    }
    for (key, value) in &config.overrides {
        mapping = mapping.with_override(key, value.clone());
    }
    mapping
}

/// Puts a server's tool listing into the inventory in place: the listing
/// goes in (a tool of the same name is overwritten), then what the server no
/// longer lists goes out with the aliases that target it. A reader finds
/// every tool both listings hold at every moment; an empty listing leaves
/// the server without tools. A tool goes in with the argument mapping
/// `config` sets on the tool itself, so a call never finds it without; a
/// tool with an alias carries its mapping on the alias.
fn swap_server_tools(inventory: &ToolInventory, config: &McpServerConfig, tools: Vec<ToolEntry>) {
    let listed: HashSet<String> = tools
        .iter()
        .map(|entry| entry.tool_name().to_string())
        .collect();
    for mut entry in tools {
        let direct = config
            .tools
            .as_ref()
            .and_then(|configs| configs.get(entry.tool_name()))
            .filter(|tool_config| tool_config.alias.is_none())
            .and_then(|tool_config| tool_config.arg_mapping.as_ref());
        if let Some(mapping) = direct {
            entry.arg_mapping = Some(arg_mapping_of(mapping));
        }
        inventory.insert_entry(entry);
    }
    inventory.retain_server_tool_entries(&config.name, &listed);
}

/// Puts a server's prompts into the inventory.
fn install_prompts(inventory: &ToolInventory, server_key: &str, prompts: Vec<(String, Prompt)>) {
    for (name, prompt) in prompts {
        inventory.insert_prompt(name, server_key.to_string(), prompt);
    }
}

/// Puts a server's resources into the inventory.
fn install_resources(
    inventory: &ToolInventory,
    server_key: &str,
    resources: Vec<(String, Resource)>,
) {
    for (uri, resource) in resources {
        inventory.insert_resource(uri, server_key.to_string(), resource);
    }
}

/// Registers `alias_name` for `target_server:target_tool` in the inventory,
/// with its argument mapping; the target must be in the inventory.
fn register_alias(
    inventory: &ToolInventory,
    alias_name: &str,
    target_server: &str,
    target_tool: &str,
    arg_mapping: Option<ArgMapping>,
) -> McpResult<()> {
    // Verify target exists
    let target_entry = inventory
        .get_entry(target_server, target_tool)
        .ok_or_else(|| McpError::ToolNotFound(format!("{target_server}:{target_tool}")))?;

    // Create alias entry
    let alias_target = AliasTarget {
        target: QualifiedToolName::new(target_server, target_tool),
        arg_mapping,
    };

    let alias_entry = ToolEntry::new(
        QualifiedToolName::new(ALIAS_SERVER_KEY, alias_name),
        target_entry.tool.clone(),
    )
    .with_alias(alias_target);

    inventory.insert_entry(alias_entry);

    // Also register in the aliases index
    inventory.register_alias(
        alias_name.to_string(),
        QualifiedToolName::new(target_server, target_tool),
    );

    info!(
        "Registered alias '{}' → '{}:{}'",
        alias_name, target_server, target_tool
    );
    Ok(())
}

/// Applies a server config's aliases, each with its argument mapping, to the
/// server's entries in the inventory. A tool without an alias took its
/// mapping when its listing went in (`swap_server_tools`).
fn apply_tool_configs(inventory: &ToolInventory, config: &McpServerConfig) {
    let Some(tools) = &config.tools else {
        return;
    };
    for (tool_name, tool_config) in tools {
        // Check if the tool exists
        if !inventory.has_tool_qualified(&config.name, tool_name) {
            warn!(
                "Tool config for '{}:{}' but tool not found on server",
                config.name, tool_name
            );
            continue;
        }
        // If there's an alias, register it
        if let Some(alias_name) = &tool_config.alias {
            let arg_mapping = tool_config.arg_mapping.as_ref().map(arg_mapping_of);
            if let Err(e) =
                register_alias(inventory, alias_name, &config.name, tool_name, arg_mapping)
            {
                warn!(
                    "Failed to register alias '{}' for '{}:{}': {}",
                    alias_name, config.name, tool_name, e
                );
            } else {
                info!(
                    "Registered alias '{}' → '{}:{}'",
                    alias_name, config.name, tool_name
                );
            }
        }
    }
}

/// Build request headers from token and custom headers.
fn build_request_headers(
    token: Option<&str>,
    custom_headers: &HashMap<String, String>,
) -> McpResult<reqwest::header::HeaderMap> {
    let mut headers = reqwest::header::HeaderMap::new();

    if let Some(tok) = token {
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {tok}")
                .parse()
                .map_err(|e| McpError::Transport(format!("auth token: {e}")))?,
        );
    }

    for (key, value) in custom_headers {
        headers.insert(
            reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| McpError::Transport(format!("header name: {e}")))?,
            value
                .parse()
                .map_err(|e| McpError::Transport(format!("header value: {e}")))?,
        );
    }

    Ok(headers)
}

/// Installs `ring` as the process-level TLS crypto provider unless one is
/// installed already. The HTTP client takes the process provider (reqwest's
/// `rustls-no-provider` feature) and panics at build time when there is
/// none, so the factory installs it before building a client; a provider
/// the embedding process installed earlier stays.
fn install_crypto_provider() {
    if CryptoProvider::get_default().is_none() {
        // An error here means another thread installed one in between,
        // which is the outcome wanted.
        let _ = ring::default_provider().install_default();
    }
}

/// A client builder with the process-level TLS crypto provider installed and
/// the crate's request headers: every HTTP client of this crate comes from
/// here.
#[expect(
    clippy::disallowed_methods,
    reason = "the crate's one direct client construction; the process TLS crypto provider is installed first"
)]
pub(crate) fn http_client_builder() -> reqwest::ClientBuilder {
    install_crypto_provider();
    reqwest::Client::builder().default_headers(default_request_headers())
}

/// The headers every client of this crate sends with every request: the
/// client names `identity` as the one content coding it accepts. It is built
/// without decoders, so a coded body would fail to parse as a protocol
/// message; a server that honours the header never sends one (RFC 9110,
/// section 12.5.3).
fn default_request_headers() -> reqwest::header::HeaderMap {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::ACCEPT_ENCODING,
        reqwest::header::HeaderValue::from_static("identity"),
    );
    headers
}

/// Build HTTP client with default headers.
fn build_http_client(
    proxy_config: Option<&McpProxyConfig>,
    token: Option<&str>,
    custom_headers: &HashMap<String, String>,
) -> McpResult<reqwest::Client> {
    let mut builder = http_client_builder().connect_timeout(Duration::from_secs(10));

    if let Some(proxy_cfg) = proxy_config {
        builder = super::proxy::apply_proxy_to_builder(builder, proxy_cfg)?;
    }

    let req_headers = build_request_headers(token, custom_headers)?;
    if !req_headers.is_empty() {
        builder = builder.default_headers(req_headers);
    }

    builder
        .build()
        .map_err(|e| McpError::Transport(format!("build HTTP client: {e}")))
}

/// rmcp 1.7 dropped the standalone SSE client transport (HTTP+SSE was
/// superseded by Streamable HTTP in the MCP spec). Servers still configured
/// with `protocol: sse` get a clear error pointing at the replacement.
fn sse_unsupported(server_name: &str) -> McpError {
    McpError::Transport(format!(
        "server '{server_name}': the SSE client transport was removed in rmcp 1.7; \
         use 'protocol: streamable' instead"
    ))
}

/// Responses checks a label at intake; a Messages server `name` is free text:
/// empty, an address or a line break that would split a log line is `unnamed`.
pub(super) fn printable_label(label: &str) -> &str {
    if label.is_empty() || label.contains("://") || label.contains(char::is_control) {
        "unnamed"
    } else {
        label
    }
}

/// Type alias for MCP client with handler.
type McpClientWithHandler = RunningService<RoleClient, SmgClientHandler>;

/// Server entry with client, handler, and config.
#[derive(Clone)]
struct ServerEntry {
    client: Arc<McpClientWithHandler>,
    handler: Arc<SmgClientHandler>,
    config: McpServerConfig,
    /// Why the server's tool listing failed; `None` once a listing succeeded
    /// (the first load, a refresh).
    list_error: Option<String>,
}

/// A dynamic server as its connect leaves it: its URL (`server_key`, the
/// name its tools are listed under), the key its connection is pooled under
/// (`pool_key`), and why its tools could not be listed (`None` when they
/// were). The caller keeps the outcome with its binding; the pool keeps it
/// with the connection for a later hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectedServer {
    pub server_key: String,
    pub list_error: Option<String>,
    /// The key the connection is pooled under. A binding of this server
    /// carries it, and a tool call takes its connection by it.
    pub pool_key: PoolKey,
}

/// Internal result type for approval-checked execution.
enum ApprovalExecutionResult {
    /// Tool executed successfully, contains raw result.
    Success(CallToolResult),
    /// Pending approval from user (interactive mode only).
    PendingApproval(McpApprovalRequest),
}

// ============================================================================
// Batch Tool Execution Types
// ============================================================================

/// Input for batch tool execution.
///
/// This is a simplified input format that allows routers to batch
/// multiple tool calls without worrying about the underlying execution details.
#[derive(Debug, Clone)]
pub struct ToolExecutionInput {
    /// Unique identifier for this tool call (from LLM response).
    pub call_id: String,
    /// Name of the tool to execute.
    pub tool_name: String,
    /// Tool arguments as JSON.
    pub arguments: Value,
}

/// Output from batch tool execution.
///
/// `#[non_exhaustive]` so additive fields don't break consumers; only
/// `smg-mcp` constructs this in production. External tests use
/// [`ToolExecutionOutput::new_for_test`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolExecutionOutput {
    /// The call_id from the input (for matching).
    pub call_id: String,
    /// Name of the tool that was executed.
    ///
    /// This is the display name used for response transformation/output.
    pub tool_name: String,
    /// Server key where the tool was executed (internal identifier, may be URL).
    pub server_key: String,
    /// User-facing server label for API responses.
    pub server_label: String,
    /// Original arguments JSON string (for conversation history).
    pub arguments_str: String,
    /// Raw tool output as Value (for conversation history and transformation).
    pub output: Value,
    /// Whether the execution resulted in an error.
    pub is_error: bool,
    /// Error message if `is_error` is true.
    pub error_message: Option<String>,
    /// Execution duration.
    pub duration: Duration,
}

impl ToolExecutionOutput {
    /// Test-only constructor for downstream crates blocked by
    /// `#[non_exhaustive]`. One positional arg per public field is
    /// intentional so adding a field forces every test to consider it.
    #[expect(
        clippy::too_many_arguments,
        reason = "Test fixture constructor; one arg per field is intentional."
    )]
    pub fn new_for_test(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        server_key: impl Into<String>,
        server_label: impl Into<String>,
        arguments_str: impl Into<String>,
        output: Value,
        is_error: bool,
        error_message: Option<String>,
        duration: Duration,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            server_key: server_key.into(),
            server_label: server_label.into(),
            arguments_str: arguments_str.into(),
            output,
            is_error,
            error_message,
            duration,
        }
    }
}

/// Result from resolved tool execution that preserves interactive approval state.
#[derive(Debug, Clone)]
pub enum ToolExecutionResult {
    Executed(ToolExecutionOutput),
    PendingApproval(PendingToolExecution),
}

/// Pending approval from resolved tool execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PendingToolExecution {
    pub call_id: String,
    pub tool_name: String,
    pub server_key: String,
    pub server_label: String,
    pub arguments_str: String,
    pub approval_request: McpApprovalRequest,
    pub duration: Duration,
}

impl ToolExecutionResult {
    /// Convert the result to the legacy flattened output shape.
    #[must_use]
    pub fn into_output(self) -> ToolExecutionOutput {
        match self {
            Self::Executed(output) => output,
            Self::PendingApproval(pending) => ToolExecutionOutput {
                call_id: pending.call_id,
                tool_name: pending.tool_name,
                server_key: pending.server_key,
                server_label: pending.server_label,
                arguments_str: pending.arguments_str,
                output: serde_json::json!({
                    "error": "Tool requires approval (not supported in this context)"
                }),
                is_error: true,
                error_message: Some(
                    "Tool requires approval (not supported in this context)".to_string(),
                ),
                duration: pending.duration,
            },
        }
    }
}

/// Main orchestrator for MCP operations.
///
/// Thread-safe and designed for sharing across async tasks.
pub struct McpOrchestrator {
    /// Static servers (from config, never evicted). Shared with the refresh
    /// handler, which must see a server's current client after a recovery
    /// re-inserts it.
    static_servers: Arc<DashMap<String, ServerEntry>>,
    /// The connected server that answers each builtin tool type, with the tool it names: what
    /// a walk of `static_servers` would find, kept by type so a lookup takes no walk (a walk
    /// allocates once per shard). Written where an entry goes into `static_servers`, emptied
    /// where that map is; the first server registered for a type keeps it.
    builtin_servers: RwLock<Vec<(BuiltinToolType, String, String)>>,
    /// Tool inventory with qualified names.
    tool_inventory: Arc<ToolInventory>,
    /// Approval manager for interactive and policy-only modes.
    approval_manager: Arc<ApprovalManager>,
    /// Connection pool for dynamic servers.
    connection_pool: Arc<McpConnectionPool>,
    /// Metrics and monitoring.
    metrics: Arc<McpMetrics>,
    /// Channel for refresh requests from handlers.
    refresh_tx: mpsc::Sender<RefreshRequest>,
    active_executions: Arc<AtomicUsize>,
    shutdown_token: CancellationToken,
    reconnection_locks: DashMap<String, Arc<ReconnectionLock>>,
    /// The retry schedule of one recovery cycle.
    reconnection: ReconnectionManager,
    /// Original config for reference.
    config: McpConfig,
}

impl McpOrchestrator {
    /// Create a new orchestrator with the given configuration.
    ///
    /// Policy is built from `config.policy`. Default policy allows all tools.
    pub async fn new(config: McpConfig) -> McpResult<Self> {
        // Validate configuration (server pairs, no duplicate builtin_types)
        config
            .validate()
            .map_err(|e| McpError::Config(e.to_string()))?;

        let tool_inventory = Arc::new(ToolInventory::new());
        let metrics = Arc::new(McpMetrics::new());

        // Build approval manager from config
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::from_yaml_config(
            &config.policy,
            Arc::clone(&audit_log),
        ));
        let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));

        // Create connection pool with eviction callback
        let mut connection_pool =
            McpConnectionPool::with_full_config(config.pool.max_connections, config.proxy.clone());

        let inventory_clone = Arc::clone(&tool_inventory);
        connection_pool.set_eviction_callback(move |key: &PoolKey| {
            debug!(
                "LRU evicted dynamic server '{}' (tenant: {:?}) - clearing tools from inventory",
                key.url, key.tenant_id
            );
            // Tools are registered by URL, so clear by URL
            inventory_clone.clear_server_tools(&key.url);
        });

        let connection_pool = Arc::new(connection_pool);

        // Create refresh channel
        let (refresh_tx, refresh_rx) = mpsc::channel(100);

        let orchestrator = Self {
            static_servers: Arc::new(DashMap::new()),
            builtin_servers: RwLock::new(Vec::new()),
            tool_inventory,
            approval_manager,
            connection_pool,
            metrics,
            refresh_tx,
            active_executions: Arc::new(AtomicUsize::new(0)),
            shutdown_token: CancellationToken::new(),
            reconnection_locks: DashMap::new(),
            reconnection: ReconnectionManager::default(),
            config: config.clone(),
        };

        // Connect to static servers
        for server_config in &config.servers {
            if let Err(e) = orchestrator.connect_static_server(server_config).await {
                if server_config.required {
                    return Err(e);
                }
                error!(
                    "Failed to connect to optional server '{}': {}",
                    server_config.name, e
                );
            }
        }

        // Start background refresh task
        orchestrator.spawn_refresh_handler(refresh_rx);

        info!(
            "McpOrchestrator initialized with {} static servers",
            orchestrator.static_servers.len()
        );

        Ok(orchestrator)
    }

    /// Create a simplified orchestrator for testing.
    #[cfg(test)]
    pub fn new_test() -> Self {
        Self::new_test_with_config(McpConfig::default())
    }

    /// Create a simplified orchestrator with a specific config for testing.
    #[cfg(test)]
    pub fn new_test_with_config(config: McpConfig) -> Self {
        use crate::approval::{audit::AuditLog, policy::PolicyEngine};

        let (refresh_tx, _) = mpsc::channel(10);
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::new(Arc::clone(&audit_log)));
        let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));

        Self {
            static_servers: Arc::new(DashMap::new()),
            builtin_servers: RwLock::new(Vec::new()),
            tool_inventory: Arc::new(ToolInventory::new()),
            approval_manager,
            connection_pool: Arc::new(McpConnectionPool::new()),
            metrics: Arc::new(McpMetrics::new()),
            refresh_tx,
            active_executions: Arc::new(AtomicUsize::new(0)),
            shutdown_token: CancellationToken::new(),
            reconnection_locks: DashMap::new(),
            // One attempt and no delay: a test's failing recovery fails at once.
            reconnection: ReconnectionManager {
                max_retries: 1,
                base_delay: Duration::ZERO,
                max_delay: Duration::ZERO,
            },
            config,
        }
    }

    // ========================================================================
    // Server Connection
    // ========================================================================

    /// Connect to a static server from config.
    ///
    /// This method:
    /// 1. Establishes a connection to the MCP server
    /// 2. Loads tools, prompts, and resources from the server
    /// 3. Applies tool configurations (aliases, response formats)
    /// 4. Registers the server as a static server
    ///
    /// Static servers are never evicted from the connection pool.
    pub async fn connect_static_server(&self, config: &McpServerConfig) -> McpResult<()> {
        // Skip if already connected (handles duplicate registrations from workflow)
        if self.static_servers.contains_key(&config.name) {
            debug!(
                "Server '{}' already connected, skipping duplicate registration",
                config.name
            );
            return Ok(());
        }
        self.establish_static_server(config).await
    }

    /// Connect a static server and put its entry in place: the connection,
    /// its three listings, then one write-back under the name's guard: the
    /// tools swapped in place, the prompts and the resources replaced, the
    /// tool configurations, and the entry, replacing whatever entry the name
    /// held. Nothing of the server's changes while the listings are on their
    /// way, so through a recovery a reader finds the tools both listings
    /// hold; a tool listing that failed leaves the server without tools and
    /// the failure on its entry. A refresh writes back under the same guard,
    /// after checking the connection there, so a refresh that listed on the
    /// dropped connection lands before this write-back or not at all. A
    /// recovery runs this beside the dropped entry, which stays in the map,
    /// settings and visibility flags included, until the new connection
    /// replaces it (a server registered after start-up has no other copy of
    /// them); an attempt that fails changes nothing.
    async fn establish_static_server(&self, config: &McpServerConfig) -> McpResult<()> {
        info!("Connecting to static server '{}'", config.name);

        let handler = Arc::new(
            SmgClientHandler::new(&config.name, Arc::clone(&self.approval_manager))
                .with_refresh_channel(self.refresh_tx.clone()),
        );

        let client = self.connect_server_impl(config, (*handler).clone()).await?;
        let client = Arc::new(client);

        let key = config.name.as_str();
        let (tools, list_error) = match tools_of(client.as_ref(), key, ToolCategory::Static).await {
            Ok(tools) => (tools, None),
            Err(message) => (Vec::new(), Some(message)),
        };
        let prompts = prompts_of(client.as_ref(), key).await;
        let resources = resources_of(client.as_ref(), key).await;

        // The write-back: nothing awaits between the guard and the entry.
        let slot = self.static_servers.entry(config.name.clone());
        swap_server_tools(&self.tool_inventory, config, tools);
        self.tool_inventory.clear_server_prompts(key);
        if let Some(prompts) = prompts {
            install_prompts(&self.tool_inventory, key, prompts);
        }
        self.tool_inventory.clear_server_resources(key);
        if let Some(resources) = resources {
            install_resources(&self.tool_inventory, key, resources);
        }
        self.apply_tool_configs(config);
        // Store server entry with config for builtin lookups
        slot.insert(ServerEntry {
            client,
            handler,
            config: config.clone(),
            list_error,
        });

        self.metrics.record_connection_opened();
        self.index_builtin_server(config);
        info!("Connected to static server '{}'", config.name);
        Ok(())
    }

    /// Apply tool configurations from server config (aliases with their arg mappings).
    fn apply_tool_configs(&self, config: &McpServerConfig) {
        apply_tool_configs(&self.tool_inventory, config);
    }

    /// Internal server connection logic.
    async fn connect_server_impl(
        &self,
        config: &McpServerConfig,
        handler: SmgClientHandler,
    ) -> McpResult<McpClientWithHandler> {
        use rmcp::{
            transport::{
                streamable_http_client::StreamableHttpClientTransportConfig, ConfigureCommandExt,
                StreamableHttpClientTransport, TokioChildProcess,
            },
            ServiceExt,
        };

        match &config.transport {
            McpTransport::Stdio {
                command,
                args,
                envs,
            } => {
                let transport = TokioChildProcess::new(
                    tokio::process::Command::new(command).configure(|cmd| {
                        cmd.args(args)
                            .envs(envs.iter())
                            .stderr(std::process::Stdio::inherit());
                    }),
                )
                .map_err(|e| McpError::Transport(format!("create stdio transport: {e}")))?;

                Box::pin(handler.serve(transport)).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize stdio client: {e}"))
                })
            }

            McpTransport::Sse { .. } => Err(sse_unsupported(&config.name)),

            McpTransport::Streamable {
                url,
                token,
                headers: custom_headers,
            } => {
                let proxy_config =
                    super::proxy::resolve_proxy_config(config, self.config.proxy.as_ref());
                let http_client =
                    build_http_client(proxy_config, token.as_deref(), custom_headers)?;
                let cfg = StreamableHttpClientTransportConfig::with_uri(url.as_str());

                let transport = StreamableHttpClientTransport::with_client(http_client, cfg);

                Box::pin(handler.serve(transport)).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize streamable client: {e}"))
                })
            }
        }
    }

    /// Spawn background handler for inventory refresh requests.
    fn spawn_refresh_handler(&self, mut rx: mpsc::Receiver<RefreshRequest>) {
        let token = self.shutdown_token.clone();
        let tool_inventory = Arc::clone(&self.tool_inventory);
        let static_servers = Arc::clone(&self.static_servers);

        #[expect(
            clippy::disallowed_methods,
            reason = "background refresh handler runs for orchestrator lifetime; shutdown is coordinated via CancellationToken"
        )]
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    // Stop the loop if the shutdown token is triggered
                    () = token.cancelled() => {
                        debug!("Refresh handler shutting down");
                        break;
                    }
                    // Process refresh requests as they arrive
                    Some(request) = rx.recv() => {
                        debug!("Processing refresh request for '{}'", request.server_key);

                        // The client and the config out of the map's guard
                        // before the listings' round-trips: a removal or an
                        // insert on the same shard must not wait behind them.
                        let found = static_servers
                            .get(&request.server_key)
                            .map(|entry| (Arc::clone(&entry.client), entry.config.clone()));
                        if let Some((client, config)) = found {
                            // The tool listing first: when it fails nothing
                            // changes and the other two listings are not
                            // made. Prompts and resources are cleared only
                            // when their own listing succeeded, so one that
                            // fails keeps what the server had. The tools are
                            // swapped in place, each with the argument
                            // mapping its config sets, so a reader finds
                            // every tool both listings hold, whole. The
                            // config's aliases go over the result, as at a
                            // connect.
                            let key = request.server_key.as_str();
                            let Ok(tools) =
                                tools_of(client.as_ref(), key, ToolCategory::Static).await
                            else {
                                continue;
                            };
                            let prompts = prompts_of(client.as_ref(), key).await;
                            let resources = resources_of(client.as_ref(), key).await;
                            // The write-back is this connection's: a recovery
                            // during the listings replaced the entry (or a
                            // removal dropped it), and its own load's tools
                            // and outcome stand, not a stale listing's. A
                            // recovery writes back under this guard too, so
                            // the check holds until this listing is in. The
                            // entry stays held while the listing goes in and
                            // its error is cleared last: a reader that finds
                            // no listing error finds the tools.
                            let mut server = match static_servers.get_mut(key) {
                                Some(entry) if Arc::ptr_eq(&entry.client, &client) => entry,
                                _ => continue,
                            };
                            swap_server_tools(&tool_inventory, &config, tools);
                            if let Some(prompts) = prompts {
                                tool_inventory.clear_server_prompts(key);
                                install_prompts(&tool_inventory, key, prompts);
                            }
                            if let Some(resources) = resources {
                                tool_inventory.clear_server_resources(key);
                                install_resources(&tool_inventory, key, resources);
                            }
                            apply_tool_configs(&tool_inventory, &config);
                            server.list_error = None;
                            drop(server);
                            info!(
                                "Refreshed '{}'; the inventory now holds {} tools",
                                key,
                                tool_inventory.counts().0
                            );
                        }
                    }

                    else => break,
                }
            }
        });
    }

    // ========================================================================
    // Tool Execution
    // ========================================================================

    /// Find the MCP server configured to handle a built-in tool type.
    ///
    /// Returns `(server_key, tool_name)` for the configured server, or
    /// `None` if no server handles this builtin. Per-tool `ResponseFormat`
    /// lives in the gateway's `FormatRegistry` — callers look it up there.
    pub fn find_builtin_server(&self, builtin_type: BuiltinToolType) -> Option<(String, String)> {
        let extract_builtin = |server_config: &McpServerConfig| {
            if let (Some(cfg_type), Some(tool_name)) = (
                &server_config.builtin_type,
                &server_config.builtin_tool_name,
            ) {
                if *cfg_type == builtin_type {
                    return Some((server_config.name.clone(), tool_name.clone()));
                }
            }
            None
        };

        // A connected server first, one registered after start-up included: read by type from
        // the index its registration wrote, with no walk of the server map.
        let connected = self
            .builtin_servers
            .read()
            .iter()
            .find(|(held, ..)| *held == builtin_type)
            .map(|(_, server, tool)| (server.clone(), tool.clone()));
        if connected.is_some() {
            return connected;
        }

        // Fallback to initial config servers (handles edge cases where servers
        // from the initial config haven't connected yet, and supports tests).
        for server_config in &self.config.servers {
            if let Some(result) = extract_builtin(server_config) {
                return Some(result);
            }
        }

        None
    }

    /// Record a connected server as the answer for its builtin type, unless an earlier server
    /// holds the type. A recovery puts a server's own settings back, so it changes nothing here.
    fn index_builtin_server(&self, config: &McpServerConfig) {
        let (Some(builtin_type), Some(tool_name)) =
            (config.builtin_type, &config.builtin_tool_name)
        else {
            return;
        };
        let mut index = self.builtin_servers.write();
        if !index.iter().any(|(held, ..)| *held == builtin_type) {
            index.push((builtin_type, config.name.clone(), tool_name.clone()));
        }
    }

    /// Returns the set of server names that have `builtin_type` configured.
    pub fn builtin_server_names(&self) -> HashSet<String> {
        let mut names = HashSet::new();

        // Check connected static servers first
        for entry in self.static_servers.iter() {
            if entry.config.builtin_type.is_some() {
                names.insert(entry.config.name.clone());
            }
        }

        // Also check initial config (covers not-yet-connected servers and tests)
        for server_config in &self.config.servers {
            if server_config.builtin_type.is_some() {
                names.insert(server_config.name.clone());
            }
        }

        names
    }

    /// Returns the set of server names that are configured as internal.
    pub fn internal_server_names(&self) -> HashSet<String> {
        let mut names = HashSet::new();

        for entry in self.static_servers.iter() {
            if entry.config.internal {
                names.insert(entry.config.name.clone());
            }
        }

        for server_config in &self.config.servers {
            if server_config.internal {
                names.insert(server_config.name.clone());
            }
        }

        names
    }

    /// Whether a server of this name is configured as internal: the answer
    /// [`Self::internal_server_names`] holds for the name, read by key.
    pub(crate) fn is_internal_server(&self, name: &str) -> bool {
        self.static_servers
            .get(name)
            .is_some_and(|entry| entry.config.internal)
            || self
                .config
                .servers
                .iter()
                .any(|server| server.internal && server.name == name)
    }

    /// Whether a server of this name has `builtin_type` configured: the
    /// answer [`Self::builtin_server_names`] holds for the name, read by key.
    pub(crate) fn is_builtin_server(&self, name: &str) -> bool {
        self.static_servers
            .get(name)
            .is_some_and(|entry| entry.config.builtin_type.is_some())
            || self
                .config
                .servers
                .iter()
                .any(|server| server.builtin_type.is_some() && server.name == name)
    }

    /// Whether the pool holds a connection under `key`: the request's own, or
    /// one pooled since under the same key.
    pub(crate) fn is_pooled(&self, key: &PoolKey) -> bool {
        self.connection_pool.contains(key)
    }

    /// Execute a single tool using an already-resolved qualified binding.
    ///
    /// This path does not perform tool-name reverse lookup. Callers must provide
    /// the exact `server_key` and tool name to execute.
    pub async fn execute_tool_resolved(
        &self,
        input: ToolExecutionInput,
        server_key: &str,
        server_label: &str,
        request_ctx: &McpRequestContext<'_>,
    ) -> ToolExecutionOutput {
        self.execute_tool_resolved_result(input, server_key, server_label, request_ctx)
            .await
            .into_output()
    }

    /// Execute a single resolved tool while preserving pending approval state.
    pub async fn execute_tool_resolved_result(
        &self,
        input: ToolExecutionInput,
        server_key: &str,
        server_label: &str,
        request_ctx: &McpRequestContext<'_>,
    ) -> ToolExecutionResult {
        let entry = self.tool_inventory.get_entry(server_key, &input.tool_name);
        self.execute_resolved_entry_result(entry, input, server_key, server_label, request_ctx)
            .await
    }

    /// The same for the entry a caller read under `server_key` and the
    /// tool's name: the entry it checked is the entry that runs.
    pub(crate) async fn execute_resolved_entry_result(
        &self,
        entry: Option<ToolEntry>,
        input: ToolExecutionInput,
        server_key: &str,
        server_label: &str,
        request_ctx: &McpRequestContext<'_>,
    ) -> ToolExecutionResult {
        let start = Instant::now();
        let arguments_str = input.arguments.to_string();

        let qualified = QualifiedToolName::new(server_key, &input.tool_name);
        let named = if request_ctx.pool_key.is_some() {
            printable_label(server_label)
        } else {
            server_label
        };

        match entry {
            Some(entry) => match self
                .execute_tool_entry_result(
                    &entry,
                    qualified,
                    &input.call_id,
                    input.arguments,
                    named,
                    request_ctx,
                )
                .await
            {
                ToolExecutionResult::Executed(mut output) => {
                    output.call_id = input.call_id;
                    output.tool_name = input.tool_name;
                    output.server_key = server_key.to_string();
                    output.server_label = server_label.to_string();
                    output.arguments_str = arguments_str;
                    ToolExecutionResult::Executed(output)
                }
                ToolExecutionResult::PendingApproval(mut pending) => {
                    pending.call_id = input.call_id;
                    pending.tool_name = input.tool_name;
                    pending.server_key = server_key.to_string();
                    pending.server_label = server_label.to_string();
                    pending.arguments_str = arguments_str;
                    ToolExecutionResult::PendingApproval(pending)
                }
            },
            None => {
                let server = match &request_ctx.pool_key {
                    Some(key) if key.url == server_key => named,
                    _ => server_key,
                };
                let err = format!(
                    "Tool '{}' not found on server '{}'",
                    input.tool_name, server
                );
                ToolExecutionResult::Executed(ToolExecutionOutput {
                    call_id: input.call_id,
                    tool_name: input.tool_name,
                    server_key: server_key.to_string(),
                    server_label: server_label.to_string(),
                    arguments_str,
                    output: serde_json::json!({ "error": &err }),
                    is_error: true,
                    error_message: Some(err),
                    duration: start.elapsed(),
                })
            }
        }
    }

    async fn execute_tool_entry_result(
        &self,
        entry: &ToolEntry,
        qualified: QualifiedToolName,
        call_id: &str,
        arguments: Value,
        server_label: &str,
        request_ctx: &McpRequestContext<'_>,
    ) -> ToolExecutionResult {
        self.active_executions.fetch_add(1, Ordering::SeqCst);
        let _guard = scopeguard::guard(Arc::clone(&self.active_executions), |count| {
            count.fetch_sub(1, Ordering::SeqCst);
        });
        self.metrics.record_call_start(&qualified);
        let call_start_time = Instant::now();

        let result = match self
            .execute_tool_with_approval_raw_internal(
                entry,
                call_id,
                arguments,
                server_label,
                request_ctx,
            )
            .await
        {
            Ok(ApprovalExecutionResult::Success(raw_result)) => {
                ToolExecutionResult::Executed(ToolExecutionOutput {
                    call_id: String::new(),
                    tool_name: entry.tool_name().to_string(),
                    server_key: entry.server_key().to_string(),
                    server_label: entry.server_key().to_string(),
                    arguments_str: String::new(),
                    output: Self::call_result_to_json(&raw_result),
                    is_error: raw_result.is_error.unwrap_or(false),
                    error_message: None,
                    duration: call_start_time.elapsed(),
                })
            }
            Ok(ApprovalExecutionResult::PendingApproval(approval_request)) => {
                ToolExecutionResult::PendingApproval(PendingToolExecution {
                    call_id: String::new(),
                    tool_name: entry.tool_name().to_string(),
                    server_key: entry.server_key().to_string(),
                    server_label: entry.server_key().to_string(),
                    arguments_str: String::new(),
                    approval_request,
                    duration: call_start_time.elapsed(),
                })
            }
            Err(e) => {
                let err = format!("Tool call failed: {e}");
                ToolExecutionResult::Executed(ToolExecutionOutput {
                    call_id: String::new(),
                    tool_name: entry.tool_name().to_string(),
                    server_key: entry.server_key().to_string(),
                    server_label: entry.server_key().to_string(),
                    arguments_str: String::new(),
                    output: serde_json::json!({ "error": &err }),
                    is_error: true,
                    error_message: Some(err),
                    duration: call_start_time.elapsed(),
                })
            }
        };

        let succeeded =
            !matches!(&result, ToolExecutionResult::Executed(output) if output.is_error);
        let duration_ms = call_start_time.elapsed().as_millis() as u64;
        self.metrics
            .record_call_end(&qualified, succeeded, duration_ms);
        result
    }

    /// Internal implementation of approval-checked tool execution.
    /// Returns either the raw `CallToolResult` or a pending approval request.
    async fn execute_tool_with_approval_raw_internal(
        &self,
        entry: &ToolEntry,
        call_id: &str,
        arguments: Value,
        server_label: &str,
        request_ctx: &McpRequestContext<'_>,
    ) -> McpResult<ApprovalExecutionResult> {
        let elicitation_id = Self::approval_elicitation_id(entry.tool_name(), call_id);
        let approval_params = ApprovalParams {
            request_id: &request_ctx.request_id,
            server_key: entry.server_key(),
            elicitation_id: &elicitation_id,
            tool_name: entry.tool_name(),
            hints: &entry.annotations,
            message: &format!("Allow execution of '{}'?", entry.tool_name()),
            tenant_ctx: &request_ctx.tenant_ctx,
        };

        let outcome = self
            .approval_manager
            .handle_approval(request_ctx.approval_mode, approval_params)
            .await?;

        match outcome {
            ApprovalOutcome::Decided(decision) => {
                if !decision.is_allowed() {
                    self.metrics.record_approval_denied();
                    return Err(McpError::ToolDenied(entry.tool_name().to_string()));
                }
                self.metrics.record_approval_granted();
                let result = self
                    .execute_tool_with_reconnect(
                        entry,
                        arguments,
                        server_label,
                        request_ctx.pool_key.as_ref(),
                    )
                    .await?;
                Ok(ApprovalExecutionResult::Success(result))
            }
            ApprovalOutcome::Pending {
                approval_request,
                rx,
                ..
            } => {
                self.metrics.record_approval_requested();

                // In interactive mode, return pending approval
                if request_ctx.approval_mode == ApprovalMode::Interactive {
                    return Ok(ApprovalExecutionResult::PendingApproval(approval_request));
                }

                // In policy-only mode, wait for decision
                match rx.await {
                    Ok(ApprovalDecision::Approved) => {
                        self.metrics.record_approval_granted();
                        let result = self
                            .execute_tool_with_reconnect(
                                entry,
                                arguments,
                                server_label,
                                request_ctx.pool_key.as_ref(),
                            )
                            .await?;
                        Ok(ApprovalExecutionResult::Success(result))
                    }
                    Ok(ApprovalDecision::Denied { reason }) => {
                        self.metrics.record_approval_denied();
                        Err(McpError::ToolDenied(reason))
                    }
                    Err(_) => Err(McpError::ToolDenied("Channel closed".to_string())),
                }
            }
        }
    }
    async fn execute_tool_with_reconnect(
        &self,
        entry: &ToolEntry,
        arguments: Value,
        server_label: &str,
        pool_key: Option<&PoolKey>,
    ) -> McpResult<CallToolResult> {
        // A pool key is its own address's: a tool that targets another server
        // by now (an alias retargeted since its session was made) is called
        // without it.
        let pool_key = pool_key.filter(|key| key.url == Self::target_server_key(entry));

        // One deadline for the whole call: the first attempt and a replay
        // after reconnection share the configured bound instead of each
        // starting a fresh one. Reconnection itself runs under the
        // reconnection manager's own bounded retry budget.
        let deadline = self
            .config
            .call_timeout()
            .map(|bound| Instant::now() + bound);

        // Capture the client the call is about to use, under the server it
        // reaches, to tell a replacement from the one that failed.
        let initial_client = self
            .static_servers
            .get(Self::target_server_key(entry))
            .map(|e| Arc::clone(&e.client));

        match self
            .execute_tool_impl(
                entry,
                arguments.clone(),
                Self::remaining(deadline),
                server_label,
                pool_key,
            )
            .await
        {
            Ok(result) => Ok(result),
            Err(McpError::ServerDisconnected(name)) => {
                // Decide replay eligibility before any recovery work: a tool
                // with side effects gets an unknown outcome whether or not
                // the connection comes back.
                let replayable = Self::replayable(entry);
                // A pooled connection is not recovered: the reconnection is
                // the configured servers'.
                let recovery = match pool_key {
                    Some(_) => Err(McpError::ServerDisconnected(name.clone())),
                    None => self.recover_after_disconnect(&name, initial_client).await,
                };
                Self::after_recovery(replayable, recovery, &name, entry.tool_name())?;
                self.execute_tool_impl(
                    entry,
                    arguments,
                    Self::remaining(deadline),
                    server_label,
                    pool_key,
                )
                .await
            }
            Err(e) => Err(e),
        }
    }

    /// The server a call reaches: an alias's target, else the entry's own.
    /// The client captured before the call and the recovery after a drop
    /// are keyed by it, as the disconnect error names it.
    fn target_server_key(entry: &ToolEntry) -> &str {
        match &entry.alias_target {
            Some(alias) => alias.target.server_key(),
            None => entry.server_key(),
        }
    }

    /// The recovery a dropped transport gets. A server that is not a static
    /// one (no entry under its key when the call began and none configured)
    /// reached the transport through the dynamic pool, which this path does
    /// not rebuild, so the drop is answered as the disconnect it is, and no
    /// reconnection lock is kept under a key a client named in its request.
    /// A static server reconnects under its lock, one task at a time; the
    /// lock's entry leaves the map with the last lease out, so the map's
    /// size follows the servers recovering, not the servers that ever
    /// dropped. The callers that dropped during one outage are one wave: the
    /// first runs the recovery cycle, the ones queued behind it read the
    /// cycle that ran while they waited and, with the client unchanged (the
    /// cycle failed), answer the disconnect without a cycle of their own; a
    /// server that never returns holds no growing queue of cycles.
    async fn recover_after_disconnect(
        &self,
        name: &str,
        initial_client: Option<Arc<McpClientWithHandler>>,
    ) -> McpResult<()> {
        if initial_client.is_none() && !self.config.servers.iter().any(|s| s.name == name) {
            return Err(McpError::ServerDisconnected(name.to_string()));
        }

        // One recovery at a time per server; the lease's drop releases the
        // lock entry on every exit path, a cancelled request's included (the
        // server's own entry is recover_static_server's to put back).
        let lease = ReconnectionLockLease::take(&self.reconnection_locks, name).await;

        // A client differing from the one that failed means another task
        // already reconnected it. Read the verdict and drop the map guard
        // before awaiting anything.
        let already_reconnected =
            self.static_servers
                .get(name)
                .map(|current| match &initial_client {
                    Some(initial) => !Arc::ptr_eq(initial, &current.client),
                    None => true,
                });

        let recovery = match already_reconnected {
            Some(true) => {
                debug!("Server '{}' already reconnected by another task", name);
                Ok(())
            }
            _ if lease.a_cycle_ran_while_waiting() => {
                debug!("Server '{}' recovery already failed for this wave", name);
                Err(McpError::ServerDisconnected(name.to_string()))
            }
            _ => {
                let recovery = self.recover_static_server(name).await;
                lease.cycle_finished();
                recovery
            }
        };
        drop(lease);
        recovery
    }

    /// Rebuild a static server's connection. Runs under the caller's
    /// reconnection lock. The dropped entry stays in the map until the new
    /// connection replaces it, so the server's settings and visibility flags
    /// are never absent while it recovers or after a failed attempt.
    async fn recover_static_server(&self, name: &str) -> McpResult<()> {
        let server_config = Self::recovery_config(
            self.static_servers
                .get(name)
                .map(|entry| entry.config.clone()),
            &self.config.servers,
            name,
        )?;

        warn!(
            "Server '{}' disconnected, initiating thread-safe recovery",
            name
        );

        self.reconnection
            .reconnect(name, || async {
                self.establish_static_server(&server_config).await
            })
            .await
    }

    /// The settings a server's connection is rebuilt with: what its entry was
    /// connected with first (a server registered after start-up, the way the
    /// gateway registers every one of its servers, is not in the
    /// configuration's list), the configuration's list otherwise.
    fn recovery_config(
        connected: Option<McpServerConfig>,
        configured: &[McpServerConfig],
        name: &str,
    ) -> McpResult<McpServerConfig> {
        connected
            .or_else(|| configured.iter().find(|s| s.name == name).cloned())
            .ok_or_else(|| McpError::ServerNotFound(name.to_string()))
    }

    /// Whether a call whose transport dropped may be re-issued: only a tool
    /// that cannot double-execute, one marked idempotent or read-only by the
    /// entry's own annotations or by the hints the server listed with the
    /// tool. The listing is read as it came (`tool.annotations`), not through
    /// the entry's approval set, whose defaults for an absent hint belong to
    /// policy decisions. The MCP spec says the idempotency hint is only
    /// meaningful for tools with side effects, so read-only tools rarely set
    /// it.
    fn replayable(entry: &ToolEntry) -> bool {
        let listed = |hint: fn(&rmcp::model::ToolAnnotations) -> Option<bool>| {
            entry.tool.annotations.as_ref().and_then(hint) == Some(true)
        };
        entry.annotations.idempotent
            || entry.annotations.read_only
            || listed(|hints| hints.idempotent_hint)
            || listed(|hints| hints.read_only_hint)
    }

    /// Time left under `deadline`, or `None` for an unbounded call. A deadline
    /// already passed yields a zero budget, which the bounded call reports as
    /// a timeout rather than running.
    fn remaining(deadline: Option<Instant>) -> Option<Duration> {
        deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    /// What a disconnected call becomes once recovery was attempted: `Ok(())`
    /// means re-issue it. A call that must not be replayed is an unknown
    /// outcome whether recovery succeeded or not; the recovery error then only
    /// gets logged, since later calls are what it matters to.
    fn after_recovery(
        replayable: bool,
        recovery: McpResult<()>,
        server: &str,
        tool: &str,
    ) -> McpResult<()> {
        if !replayable {
            if let Err(err) = recovery {
                warn!(
                    "Server '{}' recovery failed after a non-replayable call to '{}': {}",
                    server, tool, err
                );
            }
            return Err(McpError::OutcomeUnknown {
                server: server.to_string(),
                tool: tool.to_string(),
            });
        }
        recovery
    }

    /// Elicitation id for an interactive approval.
    ///
    /// Keyed by the model's call id so that two calls to the same tool in one
    /// turn get two approvals instead of the second being rejected as a
    /// duplicate. Falls back to the tool name when no call id is known.
    fn approval_elicitation_id(tool_name: &str, call_id: &str) -> String {
        if call_id.is_empty() {
            format!("tool-{tool_name}")
        } else {
            format!("tool-{tool_name}-{call_id}")
        }
    }

    /// Run a tool call under the configured bound, if any.
    ///
    /// A disabled bound becomes a deadline that never fires: tokio clamps an
    /// unrepresentable deadline to "never", so `Duration::MAX` is safe. This
    /// is a plain function so the call future is stored once in the caller's
    /// state machine rather than duplicated through an async-fn argument slot.
    fn bounded<F>(bound: Option<Duration>, call: F) -> Timeout<F>
    where
        F: Future<Output = Result<CallToolResult, ServiceError>>,
    {
        tokio::time::timeout(bound.unwrap_or(Duration::MAX), call)
    }

    /// Map a bounded tool call to the crate error. `configured_secs` is the
    /// operator's bound, reported on expiry even when a replay ran on the
    /// remainder of it.
    fn map_call_result(
        server_key: &str,
        tool: &str,
        configured_secs: u64,
        result: Result<Result<CallToolResult, ServiceError>, Elapsed>,
    ) -> McpResult<CallToolResult> {
        match result {
            Ok(Ok(result)) => Ok(result),
            // Typed detection for transport-level failures
            Ok(Err(ServiceError::TransportClosed | ServiceError::TransportSend(_))) => {
                Err(McpError::ServerDisconnected(server_key.to_string()))
            }
            Ok(Err(e)) => Err(McpError::ToolExecution(format!("MCP call failed: {e}"))),
            Err(_elapsed) => Err(McpError::CallTimeout {
                server: server_key.to_string(),
                tool: tool.to_string(),
                secs: configured_secs,
            }),
        }
    }

    /// Execute tool without approval (internal use).
    async fn execute_tool_impl(
        &self,
        entry: &ToolEntry,
        mut arguments: Value,
        bound: Option<Duration>,
        server_label: &str,
        pool_key: Option<&PoolKey>,
    ) -> McpResult<CallToolResult> {
        // The server the call reaches is the one the recovery is keyed by;
        // the tool and the argument mapping come from the alias's target or
        // the entry itself.
        let target_server = Self::target_server_key(entry);
        let (target_tool, mapping) = match &entry.alias_target {
            Some(alias) => (alias.target.tool_name(), alias.arg_mapping.as_ref()),
            None => (entry.tool_name(), entry.arg_mapping.as_ref()),
        };
        if let Some(mapping) = mapping {
            arguments = Self::apply_arg_mapping(arguments, mapping);
        }

        // Coerce argument types based on tool schema
        // LLMs often return numbers as strings (e.g., "5" instead of 5)
        Self::coerce_arg_types(&mut arguments, &entry.tool.input_schema);

        // Build request. `CallToolRequestParams` is `#[non_exhaustive]` in rmcp
        // 1.7 (added `meta`/`task`), so construct via the builder.
        let mut request = CallToolRequestParams::new(Cow::Owned(target_tool.to_string()));
        if let Value::Object(map) = arguments {
            request = request.with_arguments(map);
        }

        // Execute on server
        self.execute_on_server(target_server, request, bound, server_label, pool_key)
            .await
    }

    /// Coerce argument types based on tool schema.
    ///
    /// LLMs often output numbers as strings, so we convert them based on the schema.
    fn coerce_arg_types(args: &mut Value, schema: &serde_json::Map<String, Value>) {
        let Some(props) = schema.get("properties").and_then(|p| p.as_object()) else {
            return;
        };
        let Some(args_map) = args.as_object_mut() else {
            return;
        };

        for (key, val) in args_map.iter_mut() {
            let should_be_number = props
                .get(key)
                .and_then(|s| s.get("type"))
                .and_then(|t| t.as_str())
                .is_some_and(|t| matches!(t, "number" | "integer"));

            if should_be_number {
                if let Some(s) = val.as_str() {
                    if let Ok(num) = s.parse::<f64>() {
                        *val = serde_json::json!(num);
                    }
                }
            }
        }
    }

    /// Apply argument mapping for tool calls.
    fn apply_arg_mapping(mut args: Value, mapping: &ArgMapping) -> Value {
        if let Value::Object(ref mut map) = args {
            // Apply renames
            for (from, to) in &mapping.renames {
                if let Some(value) = map.remove(from) {
                    map.insert(to.clone(), value);
                }
            }

            // Apply defaults
            for (key, default_value) in &mapping.defaults {
                map.entry(key.clone())
                    .or_insert_with(|| default_value.clone());
            }

            // Apply forced overrides last so config wins over model-produced values
            for (key, override_value) in &mapping.overrides {
                map.insert(key.clone(), override_value.clone());
            }
        }
        args
    }

    /// Convert CallToolResult to JSON value.
    fn call_result_to_json(result: &CallToolResult) -> Value {
        // Serialize the CallToolResult content to JSON
        // The content is a Vec of annotated content items
        serde_json::to_value(&result.content).unwrap_or_else(|e| {
            warn!(
                "Failed to serialize CallToolResult to JSON: {}. Falling back to empty object.",
                e
            );
            Value::Object(serde_json::Map::new())
        })
    }

    /// Execute a tool call on a server.
    async fn execute_on_server(
        &self,
        server_key: &str,
        request: CallToolRequestParams,
        bound: Option<Duration>,
        server_label: &str,
        pool_key: Option<&PoolKey>,
    ) -> McpResult<CallToolResult> {
        // Every call is bounded: a hung server must not hold a tool loop open
        // forever, and a call that outlives the bound is reported as unknown
        // rather than retried. `bound` is what is left of the configured
        // budget; the message reports the configured value.
        let configured_secs = self.config.pool.call_timeout;
        let tool = request.name.to_string();

        // A call made for a binding with a pool key uses the connection
        // pooled under that key (made with its own caller's credentials) and
        // no other, whatever is held under the server's name; when the pool
        // holds none under the key, the call reads the server as not found.
        if let Some(key) = pool_key {
            // The tools of an address are one list for every key pooled
            // under it, so a call on an address pooled under more than one
            // key stays refused.
            if matches!(
                self.connection_pool.get_unique_by_url(server_key),
                UrlLookup::Ambiguous
            ) {
                warn!(
                    "Refusing tool execution on '{}': multiple pooled connections share this \
                     URL with different credentials/tenants; cannot select one safely",
                    server_label
                );
                return Err(McpError::ServerAccessDenied(format!(
                    "ambiguous MCP connection for '{server_label}': multiple credentials or \
                     tenants are pooled for this URL"
                )));
            }
            let Some(client) = self.connection_pool.get(key) else {
                return Err(McpError::ServerNotFound(server_label.to_string()));
            };
            // Note: Pooled connections trigger Disconnected but
            // recovery logic is currently scoped to static servers.
            let result = Self::bounded(bound, client.call_tool(request)).await;
            return Self::map_call_result(server_label, &tool, configured_secs, result);
        }

        // Clone the client out of the map so no shard guard is held across
        // the call: reconnection removes entries from the same map.
        let static_client = self
            .static_servers
            .get(server_key)
            .map(|entry| Arc::clone(&entry.client));
        if let Some(client) = static_client {
            let result = Self::bounded(bound, client.call_tool(request)).await;
            return Self::map_call_result(server_key, &tool, configured_secs, result);
        }

        Err(McpError::ServerNotFound(server_key.to_string()))
    }

    // ========================================================================
    // Alias Registration
    // ========================================================================

    /// Register a tool alias (e.g., `web_search` → `brave:brave_web_search`).
    pub fn register_alias(
        &self,
        alias_name: &str,
        target_server: &str,
        target_tool: &str,
        arg_mapping: Option<ArgMapping>,
    ) -> McpResult<()> {
        register_alias(
            &self.tool_inventory,
            alias_name,
            target_server,
            target_tool,
            arg_mapping,
        )
    }

    // ========================================================================
    // Request Context
    // ========================================================================

    /// Create a per-request context for tool execution.
    pub fn create_request_context<'a>(
        &'a self,
        request_id: impl Into<String>,
        tenant_ctx: TenantContext,
        approval_mode: ApprovalMode,
    ) -> McpRequestContext<'a> {
        self.create_request_context_with_headers(
            request_id,
            tenant_ctx,
            approval_mode,
            HashMap::new(),
        )
    }

    /// Create a per-request context for tool execution with forwarded headers.
    pub fn create_request_context_with_headers<'a>(
        &'a self,
        request_id: impl Into<String>,
        tenant_ctx: TenantContext,
        approval_mode: ApprovalMode,
        forwarded_headers: HashMap<String, String>,
    ) -> McpRequestContext<'a> {
        McpRequestContext::new(
            self,
            request_id.into(),
            tenant_ctx,
            approval_mode,
            forwarded_headers,
        )
    }

    /// Set request context on all static server handlers.
    pub fn set_handler_contexts(&self, ctx: &HandlerRequestContext) {
        for entry in self.static_servers.iter() {
            entry.handler.set_request_context(ctx.clone());
        }
    }

    /// Clear request context from all static server handlers.
    pub fn clear_handler_contexts(&self) {
        for entry in self.static_servers.iter() {
            entry.handler.clear_request_context();
        }
    }

    // ========================================================================
    // Dynamic Server Connection
    // ========================================================================

    /// Generate a unique key for a server configuration.
    ///
    /// The key is based on the transport URL, making it suitable for connection pooling.
    pub fn server_key(config: &McpServerConfig) -> String {
        match &config.transport {
            McpTransport::Streamable { url, .. } | McpTransport::Sse { url, .. } => url.clone(),
            McpTransport::Stdio { command, args, .. } => {
                format!("{}:{}", command, args.join(" "))
            }
        }
    }

    /// Connect to a dynamic server and add it to the connection pool.
    ///
    /// This is used for per-request MCP servers specified in tool configurations.
    /// Returns the server key (URL) that references the connection and the
    /// outcome of its tool listing.
    ///
    /// The connection is keyed by (url, auth_hash, tenant_id) to ensure:
    /// - Different auth tokens get different connections
    /// - Different tenants are isolated
    pub async fn connect_dynamic_server(
        &self,
        config: McpServerConfig,
    ) -> McpResult<ConnectedServer> {
        self.connect_dynamic_server_with_tenant(config, None).await
    }

    /// Connect to a dynamic server with tenant isolation.
    ///
    /// Like `connect_dynamic_server` but includes tenant_id in the pool key
    /// for proper tenant isolation.
    pub async fn connect_dynamic_server_with_tenant(
        &self,
        config: McpServerConfig,
        tenant_id: Option<String>,
    ) -> McpResult<ConnectedServer> {
        use rmcp::{
            transport::{
                streamable_http_client::StreamableHttpClientTransportConfig,
                StreamableHttpClientTransport,
            },
            ServiceExt,
        };

        let pool_key = PoolKey::from_config(&config, tenant_id);
        let server_key = pool_key.url.clone();

        // A connection with the same auth and tenant whose tools are listed
        // is a pool hit. One whose listing failed, or has not returned yet
        // (its opener is still listing), lists again for this caller: a
        // transient refusal does not stand until eviction, and a caller
        // arriving during another's listing takes no pending listing for a
        // success.
        let mut pooled = self.connection_pool.get(&pool_key);
        // A pooled connection whose transport is closed answers no call: it
        // leaves the pool, and this caller connects anew.
        if let Some(closed) = pooled.take_if(|client| client.is_transport_closed()) {
            self.connection_pool.remove(&pool_key, &closed);
        }
        if pooled.is_some() && self.connection_pool.is_listed(&pool_key) {
            return Ok(ConnectedServer {
                server_key,
                list_error: None,
                pool_key,
            });
        }

        // Connect via the pool
        let inventory_clone = Arc::clone(&self.tool_inventory);
        let global_proxy = self.config.proxy.clone();

        let client = match pooled {
            Some(client) => client,
            None => {
                self.connection_pool
                    .get_or_create(pool_key.clone(), config.clone(), |cfg, _proxy| async move {
                        match &cfg.transport {
                            McpTransport::Streamable {
                                url,
                                token,
                                headers: custom_headers,
                            } => {
                                let proxy_config =
                                    super::proxy::resolve_proxy_config(&cfg, global_proxy.as_ref());
                                let http_client = build_http_client(
                                    proxy_config,
                                    token.as_deref(),
                                    custom_headers,
                                )?;
                                let cfg_http =
                                    StreamableHttpClientTransportConfig::with_uri(url.as_str());

                                let transport = StreamableHttpClientTransport::with_client(
                                    http_client,
                                    cfg_http,
                                );

                                Box::pin(().serve(transport)).await.map_err(|e| {
                                    McpError::ConnectionFailed(format!("streamable: {e}"))
                                })
                            }
                            McpTransport::Sse { .. } => Err(sse_unsupported(&cfg.name)),
                            McpTransport::Stdio { .. } => Err(McpError::Transport(
                                "Stdio not supported for dynamic connections".to_string(),
                            )),
                        }
                    })
                    .await?
            }
        };

        // The tools go to the inventory by URL, while this connection is
        // still pooled under its key; the listing's outcome stays with this
        // connection (its pool key) and goes to the caller's binding.
        let (entries, list_error) =
            match tools_of(client.as_ref(), &server_key, ToolCategory::Dynamic).await {
                Ok(entries) => (entries, None),
                Err(message) => (Vec::new(), Some(message)),
            };
        let still_pooled = self.connection_pool.set_listing(
            &pool_key,
            &client,
            match &list_error {
                Some(message) => Listing::Failed(message.clone()),
                None => Listing::Listed,
            },
            || {
                for entry in entries {
                    inventory_clone.insert_entry(entry);
                }
            },
        );
        if !still_pooled {
            debug!(
                "Dynamic server '{}' (tenant: {:?}) left the pool during its listing - its \
                 tools stay out of the inventory",
                pool_key.url, pool_key.tenant_id
            );
        }

        Ok(ConnectedServer {
            server_key,
            list_error,
            pool_key,
        })
    }

    // ========================================================================
    // Queries
    // ========================================================================

    /// List all tools visible to a tenant.
    pub fn list_tools(&self, _tenant_ctx: Option<&TenantContext>) -> Vec<ToolEntry> {
        self.tool_inventory
            .list_tools()
            .into_iter()
            .filter_map(|(tool_name, server_key, _)| {
                self.tool_inventory.get_entry(&server_key, &tool_name)
            })
            .collect()
    }

    /// List tools for specific servers: the servers in the order given, each
    /// server's tools in its listing order (the inventory's maps keep neither),
    /// an entry without a position after the listed ones, the name settling
    /// ties. Read by server key, cloning only these servers' entries.
    pub fn list_tools_for_servers(&self, server_keys: &[String]) -> Vec<ToolEntry> {
        let mut ranked = self.tool_inventory.entries_of_servers(server_keys);
        ranked.sort_by(|(rank_a, a), (rank_b, b)| {
            (rank_a, a.position.is_none(), a.position, a.tool_name()).cmp(&(
                rank_b,
                b.position.is_none(),
                b.position,
                b.tool_name(),
            ))
        });
        ranked.into_iter().map(|(_, entry)| entry).collect()
    }

    /// Get a tool by qualified name.
    pub fn get_tool(&self, server_key: &str, tool_name: &str) -> Option<ToolEntry> {
        self.tool_inventory.get_entry(server_key, tool_name)
    }

    /// Check if a tool exists.
    pub fn has_tool(&self, server_key: &str, tool_name: &str) -> bool {
        self.tool_inventory
            .has_tool_qualified(server_key, tool_name)
    }

    /// List all connected servers.
    pub fn list_servers(&self) -> Vec<String> {
        let mut servers: Vec<_> = self
            .static_servers
            .iter()
            .map(|e| e.key().clone())
            .collect();
        // Extract URLs from pool keys (may have duplicates for different auth/tenants)
        servers.extend(self.connection_pool.list_keys().into_iter().map(|k| k.url));
        servers.sort();
        servers.dedup();
        servers
    }

    /// Get the tool inventory.
    pub fn tool_inventory(&self) -> Arc<ToolInventory> {
        Arc::clone(&self.tool_inventory)
    }

    /// Why a static server's tools could not be listed, when its last listing
    /// failed; a dynamic server answers that at its connect
    /// (`ConnectedServer::list_error`) and the request's binding keeps it.
    pub fn list_error_for_server(&self, server_key: &str) -> Option<String> {
        self.static_servers
            .get(server_key)
            .and_then(|entry| entry.list_error.clone())
    }

    /// Whether a configured server is held under `server_key`: one that
    /// connected, at construction or by a later registration, and is held
    /// until shutdown, a dropped transport included (a server whose first
    /// connect failed, or whose registration is still retrying, is not held).
    pub fn has_static_server(&self, server_key: &str) -> bool {
        self.static_servers.contains_key(server_key)
    }

    /// Get the approval manager.
    pub fn approval_manager(&self) -> Arc<ApprovalManager> {
        Arc::clone(&self.approval_manager)
    }

    /// Get metrics.
    pub fn metrics(&self) -> Arc<McpMetrics> {
        Arc::clone(&self.metrics)
    }

    // ========================================================================
    // Interactive Mode API (Issue #103)
    // ========================================================================

    /// Resolve a pending approval request.
    ///
    /// Called when the client responds to an approval request in interactive mode.
    /// This matches the OpenAI Responses API `mcp_approval_response` format.
    pub async fn resolve_approval(
        &self,
        request_id: &str,
        server_key: &str,
        elicitation_id: &str,
        approved: bool,
        reason: Option<String>,
        tenant_ctx: &TenantContext,
    ) -> McpResult<()> {
        self.approval_manager
            .resolve(
                request_id,
                server_key,
                elicitation_id,
                approved,
                reason,
                tenant_ctx,
            )
            .await
    }

    /// Get the count of pending approvals for a request.
    pub fn pending_approval_count(&self) -> usize {
        self.approval_manager.pending_count()
    }

    /// Determine the approval mode based on API type.
    ///
    /// | API                      | Mode         |
    /// |--------------------------|--------------|
    /// | OpenAI Responses API     | Interactive  |
    /// | OpenAI Chat Completions  | PolicyOnly   |
    /// | Anthropic Messages API   | PolicyOnly   |
    /// | Batch processing         | PolicyOnly   |
    pub fn determine_approval_mode(supports_mcp_approval: bool) -> ApprovalMode {
        if supports_mcp_approval {
            ApprovalMode::Interactive
        } else {
            ApprovalMode::PolicyOnly
        }
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /// Shutdown the orchestrator gracefully.
    pub async fn shutdown(&self) {
        tracing::info!("Starting graceful shutdown of McpOrchestrator");

        self.shutdown_token.cancel();

        // Wait for active executions (30s timeout)
        let start = Instant::now();
        let timeout = Duration::from_secs(30);
        while self.active_executions.load(Ordering::SeqCst) > 0 {
            if start.elapsed() >= timeout {
                tracing::warn!(
                    "Shutdown timeout reached; {} executions still active",
                    self.active_executions.load(Ordering::SeqCst)
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Cancel pending approvals
        self.approval_manager.cancel_all_pending();

        for _ in self.static_servers.iter() {
            self.metrics.record_connection_closed();
        }
        self.static_servers.clear();
        self.builtin_servers.write().clear();
        self.connection_pool.clear();

        self.tool_inventory.clear_all();

        tracing::info!("McpOrchestrator shutdown complete");
    }
}

/// Per-request context for MCP operations.
///
/// Holds request-specific state and provides access to the orchestrator
/// for tool execution with proper tenant isolation.
pub struct McpRequestContext<'a> {
    orchestrator: &'a McpOrchestrator,
    pub request_id: String,
    pub tenant_ctx: TenantContext,
    pub approval_mode: ApprovalMode,
    pub forwarded_headers: HashMap<String, String>,
    /// The pool key a call is made with: a server named by `server_url` is
    /// called on the connection pooled under it (`ConnectedServer::pool_key`).
    /// A session sets it from its binding; `None` calls a configured server.
    pub pool_key: Option<PoolKey>,
    /// Dynamic tools added for this request only.
    dynamic_tools: DashMap<QualifiedToolName, ToolEntry>,
    /// Dynamic server clients for this request.
    dynamic_clients: DashMap<String, Arc<McpClientWithHandler>>,
}

impl<'a> McpRequestContext<'a> {
    fn new(
        orchestrator: &'a McpOrchestrator,
        request_id: String,
        tenant_ctx: TenantContext,
        approval_mode: ApprovalMode,
        forwarded_headers: HashMap<String, String>,
    ) -> Self {
        Self {
            orchestrator,
            request_id,
            tenant_ctx,
            approval_mode,
            forwarded_headers,
            pool_key: None,
            dynamic_tools: DashMap::new(),
            dynamic_clients: DashMap::new(),
        }
    }

    /// Get the handler request context for setting on handlers.
    pub fn handler_context(&self) -> HandlerRequestContext {
        HandlerRequestContext::new(
            &self.request_id,
            self.approval_mode,
            self.tenant_ctx.clone(),
            self.forwarded_headers.clone(),
        )
    }

    /// Add a dynamic server for this request.
    pub async fn add_dynamic_server(&self, config: &McpServerConfig) -> McpResult<()> {
        let handler = SmgClientHandler::new(
            &config.name,
            Arc::clone(&self.orchestrator.approval_manager),
        );

        let client = self
            .orchestrator
            .connect_server_impl(config, handler)
            .await?;
        let client = Arc::new(client);

        // Load tools
        for entry in tools_of(client.as_ref(), &config.name, ToolCategory::Dynamic)
            .await
            .unwrap_or_default()
        {
            self.dynamic_tools
                .insert(entry.qualified_name.clone(), entry);
        }

        self.dynamic_clients.insert(config.name.clone(), client);
        Ok(())
    }

    /// List all tools visible in this request context.
    pub fn list_tools(&self) -> Vec<ToolEntry> {
        let mut tools = self.orchestrator.list_tools(Some(&self.tenant_ctx));

        // Add dynamic tools
        for entry in &self.dynamic_tools {
            tools.push(entry.value().clone());
        }

        tools
    }

    /// Get server keys for dynamic clients.
    pub fn dynamic_server_keys(&self) -> Vec<String> {
        self.dynamic_clients
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }
}

impl<'a> Drop for McpRequestContext<'a> {
    fn drop(&mut self) {
        // Cleanup dynamic clients
        if !self.dynamic_clients.is_empty() {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                // Collect keys first, then remove to get exclusive ownership of Arc
                let keys: Vec<_> = self
                    .dynamic_clients
                    .iter()
                    .map(|e| e.key().clone())
                    .collect();
                for key in keys {
                    if let Some((_, client)) = self.dynamic_clients.remove(&key) {
                        if let Some(client) = Arc::into_inner(client) {
                            handle.spawn(async move {
                                if let Err(e) = client.cancel().await {
                                    warn!("Error closing dynamic client: {}", e);
                                }
                            });
                        }
                    }
                }
            }
        }
    }
}
/// A static server's reconnection state: the lock one recovery holds at a
/// time, and the number of recovery cycles finished under it, by which a
/// waiter tells a cycle that ran during its wait from none.
struct ReconnectionLock {
    lock: Arc<tokio::sync::Mutex<()>>,
    cycles: AtomicU64,
}

/// A static server's reconnection lock, held for one recovery. The map's
/// entry leaves with the last lease out, on every exit path: the release is
/// the lease's drop, so a task dropped mid-await (a cancelled request)
/// releases too, and a lease still waiting keeps the entry with its clone.
struct ReconnectionLockLease<'a> {
    locks: &'a DashMap<String, Arc<ReconnectionLock>>,
    name: &'a str,
    state: Option<Arc<ReconnectionLock>>,
    cycles_at_request: u64,
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
}

impl<'a> ReconnectionLockLease<'a> {
    /// Take the server's lock, creating its entry if no lease holds one. The
    /// lease exists before the wait, so a waiter dropped before it acquires
    /// still runs the release; the cycle count is read before the wait.
    async fn take(locks: &'a DashMap<String, Arc<ReconnectionLock>>, name: &'a str) -> Self {
        let state = Arc::clone(
            locks
                .entry(name.to_string())
                .or_insert_with(|| {
                    Arc::new(ReconnectionLock {
                        lock: Arc::new(tokio::sync::Mutex::new(())),
                        cycles: AtomicU64::new(0),
                    })
                })
                .value(),
        );
        let cycles_at_request = state.cycles.load(Ordering::Acquire);
        let lock = Arc::clone(&state.lock);
        let mut lease = Self {
            locks,
            name,
            state: Some(state),
            cycles_at_request,
            guard: None,
        };
        lease.guard = Some(lock.lock_owned().await);
        lease
    }

    /// Whether a recovery cycle finished between this lease's request and
    /// its acquisition: the wave this caller belongs to has had its cycle.
    fn a_cycle_ran_while_waiting(&self) -> bool {
        self.state
            .as_ref()
            .is_some_and(|state| state.cycles.load(Ordering::Acquire) != self.cycles_at_request)
    }

    /// Count the cycle this lease ran, for the leases waiting behind it.
    fn cycle_finished(&self) {
        if let Some(state) = &self.state {
            state.cycles.fetch_add(1, Ordering::AcqRel);
        }
    }
}

impl Drop for ReconnectionLockLease<'_> {
    fn drop(&mut self) {
        // The guard and this lease's own clone go first; what is left is the
        // map's entry and the clones of leases still waiting.
        drop(self.guard.take());
        drop(self.state.take());
        self.locks
            .remove_if(self.name, |_, held| Arc::strong_count(held) == 1);
    }
}

#[cfg(test)]
mod integration_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[tokio::test]
    async fn test_reconnection_logic_flow() {
        let _orchestrator = McpOrchestrator::new_test();
        let name = "test-server";

        // simulate a server in the registry
        let manager = ReconnectionManager {
            base_delay: Duration::from_millis(1),
            max_retries: 3,
            ..Default::default()
        };

        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_clone = Arc::clone(&attempts);

        let result = manager
            .reconnect(name, || {
                let count = attempts_clone.fetch_add(1, Ordering::SeqCst);
                async move {
                    if count < 1 {
                        Err(McpError::Transport("Handshake failed".to_string()))
                    } else {
                        Ok(())
                    }
                }
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, time::Duration};

    use tokio::time::timeout;

    use super::*;
    use crate::{
        annotations::ToolAnnotations,
        core::config::{
            test_support::sse_server, ArgMappingConfig, ResponseFormatConfig, Tool as McpTool,
            ToolConfig,
        },
    };

    fn create_test_tool(name: &str) -> McpTool {
        use std::sync::Arc;

        McpTool::new(
            Cow::Owned(name.to_string()),
            Cow::Owned(format!("Test tool: {name}")),
            Arc::new(serde_json::Map::new()),
        )
    }
    /// The tools of the request's servers come in the request's server order,
    /// then each server's listing order (the position the listing recorded),
    /// an entry without a position last, however the inventory's maps hash
    /// them.
    #[test]
    fn tools_are_listed_in_the_servers_listing_order() {
        let orchestrator = McpOrchestrator::new_test();
        let listed = [
            "echo",
            "sleep",
            "fail",
            "big",
            "nested",
            "multi",
            "rpc_error",
            "zeta",
        ];
        for (position, name) in listed.iter().enumerate() {
            orchestrator.tool_inventory().insert_entry(
                ToolEntry::from_server_tool("s1", create_test_tool(name)).with_position(position),
            );
        }
        // made otherwise than by the listing (no position): last, whatever its name
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool("s1", create_test_tool("aaa")));
        for (position, name) in ["second", "first"].iter().enumerate() {
            orchestrator.tool_inventory().insert_entry(
                ToolEntry::from_server_tool("s2", create_test_tool(name)).with_position(position),
            );
        }
        let names = |keys: &[&str]| -> Vec<String> {
            let keys: Vec<String> = keys.iter().map(ToString::to_string).collect();
            orchestrator
                .list_tools_for_servers(&keys)
                .iter()
                .map(|entry| format!("{}/{}", entry.server_key(), entry.tool_name()))
                .collect()
        };

        let mut s1: Vec<String> = listed.iter().map(|name| format!("s1/{name}")).collect();
        s1.push("s1/aaa".to_string());
        let s2 = ["s2/second".to_string(), "s2/first".to_string()];
        assert_eq!(names(&["s1"]), s1);
        assert_eq!(names(&["s2", "s1"]), [&s2[..], &s1[..]].concat());
        assert_eq!(names(&["s1", "s2"]), [&s1[..], &s2[..]].concat());
        assert!(names(&["s3"]).is_empty());
    }

    #[test]
    fn approval_elicitation_id_is_keyed_by_call_id() {
        assert_eq!(
            McpOrchestrator::approval_elicitation_id("order", "call_1"),
            "tool-order-call_1"
        );
        assert_ne!(
            McpOrchestrator::approval_elicitation_id("order", "call_1"),
            McpOrchestrator::approval_elicitation_id("order", "call_2")
        );
        assert_eq!(
            McpOrchestrator::approval_elicitation_id("order", ""),
            "tool-order"
        );
    }

    #[test]
    fn only_side_effect_free_tools_are_replayed() {
        let mut entry = ToolEntry::new(
            QualifiedToolName::new("srv", "order"),
            create_test_tool("order"),
        );
        assert!(!McpOrchestrator::replayable(&entry));

        // Read-only tools rarely carry the idempotency hint; they cannot
        // double-execute anything, so they are replayed.
        entry.annotations.read_only = true;
        assert!(McpOrchestrator::replayable(&entry));

        entry.annotations.read_only = false;
        entry.annotations.idempotent = true;
        assert!(McpOrchestrator::replayable(&entry));
    }

    #[test]
    fn a_tool_listed_read_only_or_idempotent_is_replayable() {
        // The server's own hints, as tools/list carries them, decide replay
        // without any hand-set annotation; a listing that declares neither
        // (a title alone, or read-only false) is no declaration.
        let listed = |hints: rmcp::model::ToolAnnotations| {
            let mut tool = create_test_tool("lookup");
            tool.annotations = Some(hints);
            ToolEntry::from_server_tool("srv", tool)
        };
        let hints = rmcp::model::ToolAnnotations::new;
        assert!(McpOrchestrator::replayable(&listed(
            hints().read_only(true)
        )));
        assert!(McpOrchestrator::replayable(&listed(
            hints().idempotent(true)
        )));
        assert!(!McpOrchestrator::replayable(&listed(
            hints().read_only(false)
        )));
        assert!(!McpOrchestrator::replayable(&listed(
            rmcp::model::ToolAnnotations::with_title("Look up")
        )));

        let unhinted = ToolEntry::from_server_tool("srv", create_test_tool("order"));
        assert!(!McpOrchestrator::replayable(&unhinted));
    }

    #[test]
    fn a_server_is_rebuilt_from_what_it_was_connected_with() {
        // A server registered after start-up is not in the configuration's
        // list; its entry's settings are what recovery reads first.
        let late = sse_server("late", "http://localhost:1/sse");
        let configured = vec![sse_server("listed", "http://localhost:2/sse")];
        let found = McpOrchestrator::recovery_config(Some(late), &[], "late");
        assert_eq!(found.expect("the entry's settings").name, "late");
        let found = McpOrchestrator::recovery_config(None, &configured, "listed");
        assert_eq!(found.expect("the configuration's entry").name, "listed");
        assert!(matches!(
            McpOrchestrator::recovery_config(None, &configured, "late"),
            Err(McpError::ServerNotFound(name)) if name == "late"
        ));
        // Both hold the name: the settings the entry was connected with win
        // (a configured server that failed its optional connect at start-up
        // and was registered later with other settings recovers the live one).
        let connected = sse_server("listed", "http://localhost:3/sse");
        let found = McpOrchestrator::recovery_config(Some(connected), &configured, "listed")
            .expect("the entry's settings");
        assert!(matches!(
            found.transport,
            McpTransport::Sse { ref url, .. } if url == "http://localhost:3/sse"
        ));
    }

    #[tokio::test]
    async fn a_failed_connection_attempt_leaves_the_server_map_as_it_was() {
        // The entry is written last, by a connection that came up: an attempt
        // that fails (nothing listens on the port) leaves the map, and the
        // visibility it answers for, as they were.
        let orchestrator = McpOrchestrator::new_test();
        let mut late = sse_server("late", "http://127.0.0.1:1/mcp");
        late.transport = McpTransport::Streamable {
            url: "http://127.0.0.1:1/mcp".to_string(),
            token: None,
            headers: HashMap::new(),
        };
        late.internal = true;
        orchestrator
            .establish_static_server(&late)
            .await
            .expect_err("nothing listens on the port");
        assert!(!orchestrator.static_servers.contains_key("late"));
        assert!(orchestrator.internal_server_names().is_empty());
    }

    #[test]
    fn a_non_replayable_call_is_unknown_whether_or_not_recovery_succeeds() {
        let failed = || Err(McpError::ConnectionFailed("handshake".to_string()));

        assert!(matches!(
            McpOrchestrator::after_recovery(false, failed(), "srv", "order"),
            Err(McpError::OutcomeUnknown { .. })
        ));
        assert!(matches!(
            McpOrchestrator::after_recovery(false, Ok(()), "srv", "order"),
            Err(McpError::OutcomeUnknown { .. })
        ));

        // A replayable call surfaces the recovery error itself, and proceeds
        // to the replay once the connection is back.
        assert!(matches!(
            McpOrchestrator::after_recovery(true, failed(), "srv", "order"),
            Err(McpError::ConnectionFailed(_))
        ));
        assert!(McpOrchestrator::after_recovery(true, Ok(()), "srv", "order").is_ok());
    }

    #[tokio::test]
    async fn an_expired_call_maps_to_the_timeout_error() {
        let pending = std::future::pending::<Result<CallToolResult, ServiceError>>();
        let result = McpOrchestrator::bounded(Some(Duration::from_millis(5)), pending).await;
        assert!(result.is_err(), "the bound must fire");

        let mapped = McpOrchestrator::map_call_result("srv", "order", 7, result);
        assert!(matches!(mapped, Err(McpError::CallTimeout { secs: 7, .. })));
    }

    #[test]
    fn a_passed_deadline_leaves_no_budget() {
        assert_eq!(McpOrchestrator::remaining(None), None);
        let passed = Instant::now() - Duration::from_secs(1);
        assert_eq!(
            McpOrchestrator::remaining(Some(passed)),
            Some(Duration::ZERO)
        );
        let ahead = Instant::now() + Duration::from_secs(60);
        assert!(McpOrchestrator::remaining(Some(ahead))
            .is_some_and(|left| left > Duration::from_secs(50)));
    }

    #[tokio::test]
    async fn test_orchestrator_graceful_shutdown_flow() {
        let orchestrator = McpOrchestrator::new_test();
        let tenant = TenantContext::new("test-tenant");
        let ctx = orchestrator.create_request_context(
            "req-shutdown-test",
            tenant,
            ApprovalMode::Interactive,
        );

        let hints = ToolAnnotations::new();
        let params = ApprovalParams {
            request_id: "req-shutdown-test",
            server_key: "test-server",
            elicitation_id: "elicit-1",
            tool_name: "test_tool",
            hints: &hints,
            message: "Allow tool execution?",
            tenant_ctx: &ctx.tenant_ctx,
        };

        // Use the approval manager directly to simulate a real pending state
        let outcome = orchestrator
            .approval_manager()
            .handle_approval(ApprovalMode::Interactive, params)
            .await
            .expect("Failed to create pending approval");

        let rx = match outcome {
            ApprovalOutcome::Pending { rx, .. } => rx,
            ApprovalOutcome::Decided(_) => {
                panic!("Expected the outcome to be Pending for interactive mode")
            }
        };

        let shutdown_result = timeout(Duration::from_secs(5), orchestrator.shutdown()).await;
        assert!(shutdown_result.is_ok(), "Orchestrator shutdown timed out");

        let decision = rx
            .await
            .expect("The approval channel should have received a response before closing");
        match decision {
            ApprovalDecision::Denied { reason } => {
                assert_eq!(
                    reason, "System shutdown",
                    "Denial reason should be 'System shutdown'"
                );
            }
            ApprovalDecision::Approved => {
                panic!("Expected the tool call to be Denied, but it was Approved")
            }
        }

        assert_eq!(
            orchestrator.active_executions.load(Ordering::SeqCst),
            0,
            "Active execution counter must be zero"
        );
        assert_eq!(
            orchestrator.pending_approval_count(),
            0,
            "Pending approvals were not cleared"
        );
        assert!(
            orchestrator.list_servers().is_empty(),
            "Server registry was not cleared"
        );
        assert_eq!(
            orchestrator.tool_inventory().counts().0,
            0,
            "Tool inventory was not cleared"
        );
    }

    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = McpOrchestrator::new_test();
        assert!(orchestrator.list_servers().is_empty());
    }

    #[test]
    fn test_request_context_creation() {
        let orchestrator = McpOrchestrator::new_test();
        let ctx = orchestrator.create_request_context(
            "req-1",
            TenantContext::new("tenant-1"),
            ApprovalMode::PolicyOnly,
        );

        assert_eq!(ctx.request_id, "req-1");
        assert_eq!(ctx.tenant_ctx.tenant_id.as_str(), "tenant-1");
        assert!(ctx.forwarded_headers.is_empty());
    }

    #[test]
    fn test_handler_context() {
        let orchestrator = McpOrchestrator::new_test();
        let ctx = orchestrator.create_request_context(
            "req-1",
            TenantContext::new("tenant-1"),
            ApprovalMode::Interactive,
        );

        let handler_ctx = ctx.handler_context();
        assert_eq!(handler_ctx.request_id, "req-1");
        assert_eq!(handler_ctx.approval_mode, ApprovalMode::Interactive);
        assert!(handler_ctx.forwarded_headers.is_empty());
    }

    #[test]
    fn test_tool_inventory_access() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert a test tool
        let tool = create_test_tool("test_tool");
        let entry = ToolEntry::from_server_tool("test_server", tool);
        orchestrator.tool_inventory.insert_entry(entry);

        assert!(orchestrator.has_tool("test_server", "test_tool"));
        assert!(!orchestrator.has_tool("other_server", "test_tool"));
    }

    #[test]
    fn test_alias_registration() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert target tool
        let tool = create_test_tool("brave_web_search");
        let entry = ToolEntry::from_server_tool("brave", tool);
        orchestrator.tool_inventory.insert_entry(entry);

        // Register alias
        let result = orchestrator.register_alias("web_search", "brave", "brave_web_search", None);

        assert!(result.is_ok());
        assert!(orchestrator
            .tool_inventory
            .resolve_alias("web_search")
            .is_some());
    }

    #[test]
    fn test_alias_registration_missing_target() {
        let orchestrator = McpOrchestrator::new_test();

        let result =
            orchestrator.register_alias("web_search", "missing_server", "missing_tool", None);

        assert!(result.is_err());
    }

    #[test]
    fn test_arg_mapping() {
        let mapping = ArgMapping::new()
            .with_rename("query", "search_query")
            .with_default("limit", serde_json::json!(10));

        let args = serde_json::json!({
            "query": "rust programming"
        });

        let result = McpOrchestrator::apply_arg_mapping(args, &mapping);

        let obj = result.as_object().unwrap();
        assert!(obj.contains_key("search_query"));
        assert!(!obj.contains_key("query"));
        assert_eq!(obj.get("limit"), Some(&serde_json::json!(10)));
    }

    #[test]
    fn test_arg_mapping_overrides_take_precedence() {
        let mapping = ArgMapping::new()
            .with_default("enable_source", serde_json::json!(true))
            .with_override("enable_brave", serde_json::json!(false));

        let args = serde_json::json!({
            "query": "rust programming",
            "enable_brave": true
        });

        let result = McpOrchestrator::apply_arg_mapping(args, &mapping);

        let obj = result.as_object().unwrap();
        assert_eq!(obj.get("enable_source"), Some(&serde_json::json!(true)));
        assert_eq!(obj.get("enable_brave"), Some(&serde_json::json!(false)));
    }

    #[test]
    fn a_listed_tool_goes_in_with_its_direct_arg_mapping() {
        let orchestrator = McpOrchestrator::new_test();

        let mut tools = HashMap::new();
        tools.insert(
            "search_web".to_string(),
            ToolConfig {
                alias: None,
                response_format: Some(ResponseFormatConfig::WebSearchCall),
                arg_mapping: Some(ArgMappingConfig {
                    renames: HashMap::new(),
                    defaults: HashMap::from([(
                        "enable_source".to_string(),
                        serde_json::json!(true),
                    )]),
                    overrides: HashMap::from([(
                        "enable_brave".to_string(),
                        serde_json::json!(false),
                    )]),
                }),
            },
        );

        let config = McpServerConfig {
            tools: Some(tools),
            ..sse_server("web-search", "http://localhost:8080/sse")
        };

        swap_server_tools(
            &orchestrator.tool_inventory,
            &config,
            vec![ToolEntry::from_server_tool(
                "web-search",
                create_test_tool("search_web"),
            )],
        );

        let entry = orchestrator
            .tool_inventory
            .get_entry("web-search", "search_web")
            .expect("tool entry should exist");
        let mapping = entry.arg_mapping.expect("direct arg mapping should be set");
        assert_eq!(
            mapping.defaults,
            vec![("enable_source".to_string(), serde_json::json!(true))]
        );
        assert_eq!(
            mapping.overrides,
            vec![("enable_brave".to_string(), serde_json::json!(false))]
        );
    }

    #[test]
    fn an_aliased_tool_keeps_its_arg_mapping_on_the_alias() {
        let orchestrator = McpOrchestrator::new_test();

        let mut tools = HashMap::new();
        tools.insert(
            "search_web".to_string(),
            ToolConfig {
                alias: Some("web_search".to_string()),
                response_format: None,
                arg_mapping: Some(ArgMappingConfig {
                    renames: HashMap::new(),
                    defaults: HashMap::new(),
                    overrides: HashMap::from([(
                        "enable_brave".to_string(),
                        serde_json::json!(false),
                    )]),
                }),
            },
        );

        let config = McpServerConfig {
            tools: Some(tools),
            ..sse_server("web-search", "http://localhost:8080/sse")
        };

        swap_server_tools(
            &orchestrator.tool_inventory,
            &config,
            vec![ToolEntry::from_server_tool(
                "web-search",
                create_test_tool("search_web"),
            )],
        );
        orchestrator.apply_tool_configs(&config);

        let own = orchestrator
            .tool_inventory
            .get_entry("web-search", "search_web")
            .expect("tool entry should exist");
        assert!(
            own.arg_mapping.is_none(),
            "the tool's own entry takes no mapping: the alias carries it"
        );
        let alias = orchestrator
            .tool_inventory
            .get_entry(ALIAS_SERVER_KEY, "web_search")
            .expect("alias entry should exist");
        let mapping = alias
            .alias_target
            .and_then(|target| target.arg_mapping)
            .expect("the alias carries the arg mapping");
        assert_eq!(
            mapping.overrides,
            vec![("enable_brave".to_string(), serde_json::json!(false))]
        );
    }

    /// A tool listing that ends after its connection left the pool puts no
    /// tool into the inventory: no pooled connection is left to clear it.
    #[tokio::test]
    async fn a_listing_that_ends_after_its_connection_left_the_pool_lists_nothing() {
        use rmcp::{
            model::{ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig},
            service::RequestContext,
            RoleServer, ServerHandler, ServiceExt,
        };
        use tokio::sync::Notify;

        /// A server whose listing says it was asked, then waits to be let go.
        struct Slow {
            asked: Arc<Notify>,
            go: Arc<Notify>,
        }
        impl ServerHandler for Slow {
            fn get_info(&self) -> ServerConfig {
                ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            }

            async fn list_tools(
                &self,
                _request: Option<PaginatedRequestParams>,
                _context: RequestContext<RoleServer>,
            ) -> Result<ListToolsResult, rmcp::ErrorData> {
                self.asked.notify_one();
                self.go.notified().await;
                Ok(ListToolsResult::with_all_items(vec![create_test_tool(
                    "late",
                )]))
            }
        }

        let mut config = McpConfig::default();
        config.pool.max_connections = 1;
        let orchestrator = &McpOrchestrator::new(config).await.unwrap();
        let (asked, go) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
        let server_at = |url: &str| McpServerConfig {
            transport: McpTransport::Streamable {
                url: url.to_string(),
                token: None,
                headers: HashMap::new(),
            },
            ..sse_server("tools", url)
        };
        let pool = |url: &str, client| {
            orchestrator.connection_pool.get_or_create(
                PoolKey::from_config(&server_at(url), None),
                server_at(url),
                |_, _| std::future::ready(Ok(client)),
            )
        };
        let slow = Slow {
            asked: Arc::clone(&asked),
            go: Arc::clone(&go),
        };
        let (server_io, client_io) = tokio::io::duplex(4096);
        let (_server, client) = tokio::join!(slow.serve(server_io), ().serve(client_io));
        let url = "http://slow.example/mcp";
        pool(url, client.unwrap()).await.unwrap();
        // The connection that takes the pool's one place is never asked.
        let (_idle_io, client_io) = tokio::io::duplex(64);
        let other = pool(
            "http://other.example/mcp",
            rmcp::service::serve_directly((), client_io, None),
        );
        let (listed, ()) = timeout(Duration::from_secs(5), async {
            tokio::join!(orchestrator.connect_dynamic_server(server_at(url)), async {
                asked.notified().await;
                other.await.unwrap();
                go.notify_one();
            })
        })
        .await
        .unwrap();
        assert_eq!(listed.unwrap().list_error, None);

        let key = PoolKey::from_config(&server_at(url), None);
        assert!(orchestrator.connection_pool.get(&key).is_none());
        assert!(orchestrator
            .tool_inventory()
            .get_entry(url, "late")
            .is_none());
    }

    /// A request's own server whose connection left the pool before the
    /// session is built lists with an error, not as a server of no tools.
    #[tokio::test]
    async fn a_server_whose_connection_left_the_pool_lists_with_an_error() {
        use crate::core::session::{McpServerBinding, McpToolSession};

        let mut config = McpConfig::default();
        config.pool.max_connections = 1;
        let orchestrator = &McpOrchestrator::new(config).await.unwrap();
        // Pools a connection that is never asked and answers its key.
        let pool = |url: &str| {
            let config = McpServerConfig {
                transport: McpTransport::Streamable {
                    url: url.to_string(),
                    token: None,
                    headers: HashMap::new(),
                },
                ..sse_server("tools", url)
            };
            let key = PoolKey::from_config(&config, None);
            let (_, client_io) = tokio::io::duplex(64);
            let client = rmcp::service::serve_directly((), client_io, None);
            let pooling =
                orchestrator
                    .connection_pool
                    .get_or_create(key.clone(), config, |_, _| std::future::ready(Ok(client)));
            async move {
                pooling.await.unwrap();
                key
            }
        };
        let url = "http://tools.example/mcp";
        let mut binding = McpServerBinding {
            label: "tools".to_string(),
            server_key: url.to_string(),
            allowed_tools: None,
            list_error: None,
            pool_key: Some(pool(url).await),
        };
        let list_error = |binding: &McpServerBinding| {
            McpToolSession::new(orchestrator, vec![binding.clone()], "request")
                .list_error_for_binding(&binding.label, url)
        };
        assert_eq!(list_error(&binding), None);

        // Another server's connection takes the pool's one place.
        pool("http://other.example/mcp").await;
        assert_eq!(
            list_error(&binding).as_deref(),
            Some("Server not found: tools")
        );
        binding.label.clear();
        assert_eq!(
            list_error(&binding).as_deref(),
            Some("Server not found: unnamed")
        );
        // A listing that failed keeps its own error.
        binding.list_error = Some("boom".to_string());
        assert_eq!(list_error(&binding).as_deref(), Some("boom"));
    }

    #[test]
    fn test_metrics_access() {
        let orchestrator = McpOrchestrator::new_test();
        let metrics = orchestrator.metrics();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_calls, 0);
    }

    #[test]
    fn test_determine_approval_mode() {
        // OpenAI Responses API supports MCP approval
        assert_eq!(
            McpOrchestrator::determine_approval_mode(true),
            ApprovalMode::Interactive
        );

        // Other APIs don't support MCP approval
        assert_eq!(
            McpOrchestrator::determine_approval_mode(false),
            ApprovalMode::PolicyOnly
        );
    }

    #[test]
    fn test_pending_approval_count() {
        let orchestrator = McpOrchestrator::new_test();
        assert_eq!(orchestrator.pending_approval_count(), 0);
    }

    #[test]
    fn test_call_tool_by_name_finds_unique_tool() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert a tool on server1
        let tool = create_test_tool("unique_tool");
        let entry = ToolEntry::from_server_tool("server1", tool);
        orchestrator.tool_inventory.insert_entry(entry);

        // Check that the tool is found in inventory
        let tools = orchestrator.tool_inventory.list_tools();
        let matching: Vec<_> = tools
            .into_iter()
            .filter(|(name, server_key, _)| name == "unique_tool" && server_key == "server1")
            .collect();

        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].1, "server1");
    }

    #[test]
    fn test_call_tool_by_name_collision_detection() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert same tool name on two different servers
        let tool1 = create_test_tool("shared_tool");
        let entry1 = ToolEntry::from_server_tool("server1", tool1);
        orchestrator.tool_inventory.insert_entry(entry1);

        let tool2 = create_test_tool("shared_tool");
        let entry2 = ToolEntry::from_server_tool("server2", tool2);
        orchestrator.tool_inventory.insert_entry(entry2);

        // Check collision: both servers allowed
        let tools = orchestrator.tool_inventory.list_tools();
        let allowed_servers = ["server1", "server2"];
        let matching: Vec<_> = tools
            .into_iter()
            .filter(|(name, server_key, _)| {
                name == "shared_tool" && allowed_servers.contains(&server_key.as_str())
            })
            .collect();

        // Should find 2 matches (collision)
        assert_eq!(matching.len(), 2);
    }

    #[test]
    fn test_call_tool_by_name_no_collision_with_single_server() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert same tool name on two different servers
        let tool1 = create_test_tool("shared_tool");
        let entry1 = ToolEntry::from_server_tool("server1", tool1);
        orchestrator.tool_inventory.insert_entry(entry1);

        let tool2 = create_test_tool("shared_tool");
        let entry2 = ToolEntry::from_server_tool("server2", tool2);
        orchestrator.tool_inventory.insert_entry(entry2);

        // Check no collision: only one server allowed
        let tools = orchestrator.tool_inventory.list_tools();
        let allowed_servers = ["server1"];
        let matching: Vec<_> = tools
            .into_iter()
            .filter(|(name, server_key, _)| {
                name == "shared_tool" && allowed_servers.contains(&server_key.as_str())
            })
            .collect();

        // Should find only 1 match (no collision)
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0].1, "server1");
    }

    /// A dynamic server (no static entry, none configured) whose transport
    /// dropped is answered as the disconnect it is, not the configured-server
    /// lookup's "not found", and no reconnection lock is kept under its key.
    #[tokio::test]
    async fn a_dynamic_servers_dropped_transport_is_a_disconnect_and_leaves_no_lock() {
        let orchestrator = McpOrchestrator::new_test();
        let key = "https://tools.example/mcp";
        let err = orchestrator
            .recover_after_disconnect(key, None)
            .await
            .expect_err("a dynamic server has no static recovery");
        assert!(
            matches!(&err, McpError::ServerDisconnected(name) if name == key),
            "{err:?}"
        );
        assert!(orchestrator.reconnection_locks.is_empty());
    }

    /// A pooled connection whose transport closed is no hit at the connect:
    /// it leaves the pool with its address's tools, and the caller connects
    /// anew.
    #[tokio::test]
    async fn a_closed_pooled_connection_leaves_the_pool_at_the_next_connect() {
        let orchestrator = McpOrchestrator::new(McpConfig::default()).await.unwrap();
        // A dead local port: a new connect fails at once.
        let url = "http://127.0.0.1:1/mcp";
        let config = McpServerConfig {
            transport: McpTransport::Streamable {
                url: url.to_string(),
                token: None,
                headers: HashMap::new(),
            },
            ..sse_server("tools", url)
        };
        let key = PoolKey::from_config(&config, None);
        let pool = &orchestrator.connection_pool;
        let (server_io, client_io) = tokio::io::duplex(64);
        let client = rmcp::service::serve_directly((), client_io, None);
        let client = pool
            .get_or_create(key.clone(), config.clone(), |_, _| {
                std::future::ready(Ok(client))
            })
            .await
            .unwrap();
        pool.set_listing(&key, &client, Listing::Listed, || {
            orchestrator
                .tool_inventory
                .insert_entry(ToolEntry::from_server_tool(url, create_test_tool("find")));
        });

        // The server's end closes.
        drop(server_io);
        timeout(Duration::from_secs(5), async {
            while !client.is_transport_closed() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the client reads its transport closed");
        let err = orchestrator
            .connect_dynamic_server(config)
            .await
            .unwrap_err();
        assert!(matches!(err, McpError::ConnectionFailed(_)), "{err:?}");
        assert!(pool.is_empty());
        // No other pooled connection holds the address: its tools left too.
        assert!(orchestrator.tool_inventory.get_entry(url, "find").is_none());
    }

    /// A dynamic server's tool call uses the connection pooled under its own
    /// key, through a session and made directly with the key on the caller's
    /// context; while its address is pooled under two keys it stays refused;
    /// a call whose key the pool holds no more reads the server as not found
    /// while its tool is still listed; a pooled connection that drops starts
    /// no reconnection.
    #[tokio::test]
    async fn a_dynamic_call_uses_the_connection_its_own_caller_made() {
        use rmcp::{
            model::{CallToolResponse, ContentBlock, ServerCapabilities, ServerConfig},
            service::RequestContext,
            RoleServer, ServerHandler, ServiceExt,
        };

        use crate::core::session::{McpServerBinding, McpToolSession};

        /// A server that answers every call with its own name.
        struct Named(&'static str);
        impl ServerHandler for Named {
            fn get_info(&self) -> ServerConfig {
                ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            }

            async fn call_tool(
                &self,
                _request: CallToolRequestParams,
                _context: RequestContext<RoleServer>,
            ) -> Result<CallToolResponse, rmcp::ErrorData> {
                Ok(CallToolResult::success(vec![ContentBlock::text(self.0)]).into())
            }
        }

        let url = "http://tools.example/mcp";
        let mut config = McpConfig::default();
        config.pool.max_connections = 2;
        let orchestrator = &McpOrchestrator::new(config).await.unwrap();
        // Read-only, so that the dropped call reads the disconnect: a call
        // that may not be replayed reads an unknown outcome.
        let mut tool = ToolEntry::from_server_tool(url, create_test_tool("whoami"));
        tool.annotations.read_only = true;
        orchestrator.tool_inventory().insert_entry(tool);
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(url, create_test_tool("write")));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "conf",
                create_test_tool("whoami"),
            ));
        orchestrator
            .register_alias("who", "conf", "whoami", None)
            .unwrap();

        // One connection for each token; its server answers with the token.
        let mut servers = Vec::new();
        let mut clients = Vec::new();
        for token in ["three", "two", "one"] {
            let (server_io, client_io) = tokio::io::duplex(4096);
            let (server, client) = tokio::join!(Named(token).serve(server_io), ().serve(client_io));
            servers.push(server.unwrap());
            clients.push(client.unwrap());
        }
        // A configured server held under the address as its name, and under a
        // name of its own: a call made for a binding with a pool key still
        // uses its own connection.
        let handler = Arc::new(SmgClientHandler::new(
            url,
            Arc::clone(&orchestrator.approval_manager),
        ));
        let (server_io, client_io) = tokio::io::duplex(4096);
        let (server, client) = tokio::join!(
            Named("configured").serve(server_io),
            (*handler).clone().serve(client_io)
        );
        servers.push(server.unwrap());
        let configured = Arc::new(client.unwrap());
        for name in [url, "conf"] {
            orchestrator.static_servers.insert(
                name.to_string(),
                ServerEntry {
                    client: Arc::clone(&configured),
                    handler: Arc::clone(&handler),
                    config: sse_server(name, url),
                    list_error: None,
                },
            );
        }
        // Pools the next connection as the one `token` made to `url`; the
        // binding holds that connection's key.
        let mut connect = |url: &str, token: &str| {
            let config = McpServerConfig {
                transport: McpTransport::Streamable {
                    url: url.to_string(),
                    token: Some(token.to_string()),
                    headers: HashMap::new(),
                },
                ..sse_server("tools", url)
            };
            let key = PoolKey::from_config(&config, None);
            let client = clients.pop().unwrap();
            let pooling =
                orchestrator
                    .connection_pool
                    .get_or_create(key.clone(), config, |_, _| std::future::ready(Ok(client)));
            let binding = McpServerBinding {
                label: "tools".to_string(),
                server_key: url.to_string(),
                allowed_tools: None,
                list_error: None,
                pool_key: Some(key),
            };
            async move {
                pooling.await.unwrap();
                binding
            }
        };
        let session = |bindings| McpToolSession::new(orchestrator, bindings, "request");
        let input = || ToolExecutionInput {
            call_id: "call".to_string(),
            tool_name: "whoami".to_string(),
            arguments: serde_json::json!({}),
        };
        // What a call reads: the text of its answer, or its error.
        let read = |output: ToolExecutionOutput| match output.error_message {
            Some(error) => error,
            None => output.output[0]["text"].as_str().unwrap().to_string(),
        };

        let one = connect(url, "one").await;
        let first = session(vec![one.clone()]);
        assert_eq!(read(first.execute_tool(input()).await), "one");

        // The address pooled under a second key: a call on it is refused.
        let two = connect(url, "two").await;
        assert_eq!(
            first.execute_tool(input()).await.error_message.as_deref(),
            Some(
                "Tool call failed: Server access denied: ambiguous MCP connection for 'tools': \
                 multiple credentials or tenants are pooled for this URL"
            )
        );
        // Two bindings of the label, of which only the second admits the
        // tool: the call uses the second's connection.
        let narrowed = McpServerBinding {
            allowed_tools: Some(vec!["other".to_string()]),
            ..one
        };
        let second = session(vec![narrowed, two.clone()]);

        // A third connection takes the first one's place in the pool of two.
        connect("http://other.example/mcp", "three").await;
        assert_eq!(
            first.execute_tool(input()).await.error_message.as_deref(),
            Some("Tool call failed: Server not found: tools")
        );
        assert_eq!(read(second.execute_tool(input()).await), "two");
        // The same call made directly: the key goes on the caller's context.
        let mut ctx = orchestrator.create_request_context(
            "request",
            TenantContext::new("tenant"),
            ApprovalMode::PolicyOnly,
        );
        ctx.pool_key = two.pool_key;
        assert_eq!(
            read(
                orchestrator
                    .execute_tool_resolved(input(), url, "tools", &ctx)
                    .await
            ),
            "two"
        );
        for (server, label, named) in [
            (url, "tools", "tools"),
            (url, url, "unnamed"),
            (url, "two\nlines", "unnamed"),
            ("conf", "tools", "conf"),
        ] {
            let missing = ToolExecutionInput {
                tool_name: "missing".to_string(),
                ..input()
            };
            let output = orchestrator
                .execute_tool_resolved(missing, server, label, &ctx)
                .await;
            assert_eq!(
                output.error_message,
                Some(format!("Tool 'missing' not found on server '{named}'"))
            );
        }
        // A key serves its own address alone: an alias whose target is
        // another server is called there, without the key.
        let who = ToolExecutionInput {
            tool_name: "who".to_string(),
            ..input()
        };
        assert_eq!(
            read(
                orchestrator
                    .execute_tool_resolved(who, ALIAS_SERVER_KEY, "tools", &ctx)
                    .await
            ),
            "configured"
        );

        // The second one's server goes away.
        servers.remove(1).cancel().await.unwrap();
        assert_eq!(
            second.execute_tool(input()).await.error_message.as_deref(),
            Some("Tool call failed: Server disconnected: tools")
        );
        let write = ToolExecutionInput {
            tool_name: "write".to_string(),
            ..input()
        };
        assert_eq!(
            second.execute_tool(write).await.error_message.as_deref(),
            Some(
                "Tool call failed: Tool call outcome unknown: server 'tools' disconnected while \
                 executing 'write'; the call was not retried because the tool is not marked \
                 idempotent or read-only"
            )
        );
        assert_eq!(
            orchestrator
                .execute_tool_resolved(input(), url, "", &ctx)
                .await
                .error_message
                .as_deref(),
            Some("Tool call failed: Server disconnected: unnamed")
        );
    }

    /// A static server's reconnection lock leaves the map with the last
    /// lease out: a lease waiting on the first keeps the entry, the last one
    /// removes it, and a waiter dropped before it acquires (a cancelled
    /// request) leaves nothing behind.
    #[tokio::test]
    async fn a_reconnection_lock_leaves_the_map_with_its_last_holder() {
        let orchestrator = McpOrchestrator::new_test();
        let locks = &orchestrator.reconnection_locks;

        let first = ReconnectionLockLease::take(locks, "srv").await;
        let mut second = Box::pin(ReconnectionLockLease::take(locks, "srv"));
        assert!(futures::poll!(second.as_mut()).is_pending());
        drop(first);
        assert!(locks.contains_key("srv"), "a waiting lease keeps the entry");
        drop(second.await);
        assert!(locks.is_empty(), "the last lease out removes the entry");

        let first = ReconnectionLockLease::take(locks, "srv").await;
        let mut cancelled = Box::pin(ReconnectionLockLease::take(locks, "srv"));
        assert!(futures::poll!(cancelled.as_mut()).is_pending());
        drop(cancelled);
        assert!(locks.contains_key("srv"), "the holder keeps the entry");
        drop(first);
        assert!(locks.is_empty(), "a cancelled waiter leaves nothing behind");
    }

    /// The callers that dropped during one outage are one wave: the first
    /// runs the recovery cycle, the ones queued behind it read the cycle
    /// that ran while they waited and answer the disconnect without a cycle
    /// of their own.
    #[tokio::test]
    async fn one_recovery_cycle_serves_every_caller_queued_behind_it() {
        let mut config = McpConfig::default();
        let mut listed = sse_server("listed", "http://127.0.0.1:1/mcp");
        listed.transport = McpTransport::Streamable {
            url: "http://127.0.0.1:1/mcp".to_string(),
            token: None,
            headers: HashMap::new(),
        };
        config.servers.push(listed);
        let orchestrator = Arc::new(McpOrchestrator::new_test_with_config(config));

        // The test holds the lock while the wave queues up behind it; its own
        // clone of the entry outlives the leases, so the count can be read.
        let held = ReconnectionLockLease::take(&orchestrator.reconnection_locks, "listed").await;
        let state = Arc::clone(
            orchestrator
                .reconnection_locks
                .get("listed")
                .expect("the held entry")
                .value(),
        );
        #[expect(
            clippy::disallowed_methods,
            reason = "test concurrency: the four callers are the test's own tasks, each joined below"
        )]
        let waiters: Vec<_> = (0..4)
            .map(|_| {
                let orchestrator = Arc::clone(&orchestrator);
                tokio::spawn(
                    async move { orchestrator.recover_after_disconnect("listed", None).await },
                )
            })
            .collect();
        // the map's entry, this clone, the held lease and the four waiters
        for _ in 0..10_000 {
            if Arc::strong_count(&state) >= 7 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(Arc::strong_count(&state), 7, "four callers queued");
        drop(held);

        for waiter in waiters {
            let outcome = waiter.await.expect("the caller's task ran");
            assert!(outcome.is_err(), "nothing listens on the port");
        }
        assert_eq!(
            state.cycles.load(Ordering::SeqCst),
            1,
            "one cycle for the wave"
        );

        // The wave has drained: the next caller starts a cycle of its own, so a
        // server whose first recovery failed is still retried.
        assert!(orchestrator
            .recover_after_disconnect("listed", None)
            .await
            .is_err());
        assert_eq!(
            state.cycles.load(Ordering::SeqCst),
            2,
            "a later caller's own cycle"
        );
    }

    /// The client captured before a call, and the recovery after its drop,
    /// are keyed by the server the call reaches: an alias's target, else
    /// the entry's own server, as the disconnect error names it.
    #[test]
    fn a_calls_recovery_is_keyed_by_the_server_it_reaches() {
        let plain = ToolEntry::new(
            QualifiedToolName::new("brave", "web_search"),
            create_test_tool("web_search"),
        );
        assert_eq!(McpOrchestrator::target_server_key(&plain), "brave");

        let alias = ToolEntry::new(
            QualifiedToolName::new(ALIAS_SERVER_KEY, "search"),
            create_test_tool("web_search"),
        )
        .with_alias(AliasTarget {
            target: QualifiedToolName::new("brave", "web_search"),
            arg_mapping: None,
        });
        assert_eq!(McpOrchestrator::target_server_key(&alias), "brave");
    }

    #[test]
    fn test_call_tool_by_name_tool_not_found() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert a tool
        let tool = create_test_tool("existing_tool");
        let entry = ToolEntry::from_server_tool("server1", tool);
        orchestrator.tool_inventory.insert_entry(entry);

        // Search for non-existent tool
        let tools = orchestrator.tool_inventory.list_tools();
        let allowed_servers = ["server1"];
        let matching: Vec<_> = tools
            .into_iter()
            .filter(|(name, server_key, _)| {
                name == "nonexistent_tool" && allowed_servers.contains(&server_key.as_str())
            })
            .collect();

        // Should find 0 matches
        assert_eq!(matching.len(), 0);
    }

    /// The index of connected builtin servers: the first server registered for a type keeps it,
    /// the same settings put back (what a recovery does) change nothing, a connected server
    /// answers before a configured one, and a shutdown leaves the configured one.
    #[tokio::test]
    async fn the_first_connected_server_of_a_builtin_type_answers_until_shutdown() {
        let search = |name: &str, tool: &str| McpServerConfig {
            builtin_type: Some(BuiltinToolType::WebSearchPreview),
            builtin_tool_name: Some(tool.to_string()),
            ..sse_server(name, "http://localhost:1/sse")
        };
        let orchestrator = McpOrchestrator::new_test_with_config(McpConfig {
            servers: vec![search("configured", "configured_search")],
            ..Default::default()
        });
        let found = || orchestrator.find_builtin_server(BuiltinToolType::WebSearchPreview);
        let answer = |server: &str, tool: &str| Some((server.to_string(), tool.to_string()));
        assert_eq!(found(), answer("configured", "configured_search"));

        orchestrator.index_builtin_server(&search("first", "search"));
        orchestrator.index_builtin_server(&search("second", "other_search"));
        orchestrator.index_builtin_server(&search("first", "search"));
        assert_eq!(found(), answer("first", "search"));
        assert_eq!(
            orchestrator.find_builtin_server(BuiltinToolType::CodeInterpreter),
            None
        );

        orchestrator.shutdown().await;
        assert_eq!(found(), answer("configured", "configured_search"));
    }

    #[test]
    fn test_find_builtin_server_web_search() {
        // Create config with a server configured for web_search_preview
        let config = McpConfig {
            servers: vec![McpServerConfig {
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
                ..sse_server("brave", "https://mcp.brave.com/sse")
            }],
            ..Default::default()
        };

        let orchestrator = McpOrchestrator::new_test_with_config(config);

        // Should find the brave server for web_search_preview
        let result = orchestrator.find_builtin_server(BuiltinToolType::WebSearchPreview);
        assert!(result.is_some());

        let (server_key, tool_name) = result.unwrap();
        assert_eq!(server_key, "brave");
        assert_eq!(tool_name, "brave_web_search");

        // Should NOT find a server for code_interpreter
        let result = orchestrator.find_builtin_server(BuiltinToolType::CodeInterpreter);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_builtin_server_with_custom_response_format() {
        // Create config with custom response_format for the tool
        let mut tools = HashMap::new();
        tools.insert(
            "my_search".to_string(),
            ToolConfig {
                alias: None,
                response_format: Some(ResponseFormatConfig::Passthrough), // Override default
                arg_mapping: None,
            },
        );

        let config = McpConfig {
            servers: vec![McpServerConfig {
                tools: Some(tools),
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("my_search".to_string()),
                ..sse_server("custom-search", "http://localhost:3000/sse")
            }],
            ..Default::default()
        };

        let orchestrator = McpOrchestrator::new_test_with_config(config);

        let result = orchestrator.find_builtin_server(BuiltinToolType::WebSearchPreview);
        assert!(result.is_some());

        let (server_key, tool_name) = result.unwrap();
        assert_eq!(server_key, "custom-search");
        assert_eq!(tool_name, "my_search");
    }

    #[test]
    fn test_find_builtin_server_no_config() {
        let orchestrator = McpOrchestrator::new_test();

        // No servers configured for any builtin type
        assert!(orchestrator
            .find_builtin_server(BuiltinToolType::WebSearchPreview)
            .is_none());
        assert!(orchestrator
            .find_builtin_server(BuiltinToolType::CodeInterpreter)
            .is_none());
        assert!(orchestrator
            .find_builtin_server(BuiltinToolType::FileSearch)
            .is_none());
    }

    /// Every client of the crate names `identity` as the content coding it
    /// accepts, so a server that compresses every request naming no coding
    /// answers this one in the clear and the body parses. Without the header
    /// the server sends a gzip body, which the client, built without decoders,
    /// cannot read.
    #[tokio::test]
    async fn every_client_names_identity_so_a_server_compressing_unasked_answers_in_the_clear() {
        use std::{
            io::{Read, Write},
            net::TcpListener,
            sync::mpsc,
            thread,
        };
        // `{"ok":true}` under gzip: what the server sends a request that names no coding.
        const GZIPPED_BODY: &[u8] = &[
            0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x03, 0xab, 0x56, 0xca, 0xcf,
            0x56, 0xb2, 0x2a, 0x29, 0x2a, 0x4d, 0xad, 0x05, 0x00, 0x90, 0x5f, 0xd4, 0xa7, 0x0b,
            0x00, 0x00, 0x00,
        ];
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel::<String>();
        thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut head = Vec::new();
            let mut buf = [0u8; 1024];
            loop {
                let n = socket.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    break;
                }
                head.extend_from_slice(&buf[..n]);
                if head.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            let head = String::from_utf8_lossy(&head).into_owned();
            let names_identity = head
                .lines()
                .any(|line| line.eq_ignore_ascii_case("accept-encoding: identity"));
            let _ = tx.send(head);
            let response_head: &[u8] = if names_identity {
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n"
            } else {
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Encoding: gzip\r\nContent-Length: 31\r\nConnection: close\r\n\r\n"
            };
            let body: &[u8] = if names_identity {
                b"{\"ok\":true}"
            } else {
                GZIPPED_BODY
            };
            let _ = socket
                .write_all(response_head)
                .and_then(|()| socket.write_all(body));
        });

        let client = build_http_client(None, Some("token"), &HashMap::new()).unwrap();
        let response = client.get(format!("http://{addr}/")).send().await.unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body, serde_json::json!({ "ok": true }));
        let head = rx.recv().unwrap();
        assert!(
            head.lines()
                .any(|line| line.eq_ignore_ascii_case("accept-encoding: identity")),
            "the request head names no identity coding:\n{head}"
        );
    }
}
