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
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use rmcp::{
    model::{CallToolRequestParams, CallToolResult},
    service::{RunningService, ServiceError},
    RoleClient,
};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use super::{
    config::{BuiltinToolType, McpConfig, McpProxyConfig, McpServerConfig, McpTransport},
    handler::{HandlerRequestContext, RefreshRequest, SmgClientHandler},
    metrics::McpMetrics,
    pool::{McpConnectionPool, PoolKey, UrlLookup},
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

/// Build HTTP client with default headers.
fn build_http_client(
    proxy_config: Option<&McpProxyConfig>,
    token: Option<&str>,
    custom_headers: &HashMap<String, String>,
) -> McpResult<reqwest::Client> {
    let mut builder = reqwest::Client::builder().connect_timeout(Duration::from_secs(10));

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

/// Type alias for MCP client with handler.
type McpClientWithHandler = RunningService<RoleClient, SmgClientHandler>;

/// Server entry with client, handler, and config.
#[derive(Clone)]
struct ServerEntry {
    client: Arc<McpClientWithHandler>,
    handler: Arc<SmgClientHandler>,
    config: McpServerConfig,
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
    /// Static servers (from config, never evicted).
    static_servers: DashMap<String, ServerEntry>,
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
    reconnection_locks: DashMap<String, Arc<tokio::sync::Mutex<()>>>,
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
            static_servers: DashMap::new(),
            tool_inventory,
            approval_manager,
            connection_pool,
            metrics,
            refresh_tx,
            active_executions: Arc::new(AtomicUsize::new(0)),
            shutdown_token: CancellationToken::new(),
            reconnection_locks: DashMap::new(),
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
            static_servers: DashMap::new(),
            tool_inventory: Arc::new(ToolInventory::new()),
            approval_manager,
            connection_pool: Arc::new(McpConnectionPool::new()),
            metrics: Arc::new(McpMetrics::new()),
            refresh_tx,
            active_executions: Arc::new(AtomicUsize::new(0)),
            shutdown_token: CancellationToken::new(),
            reconnection_locks: DashMap::new(),
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

        info!("Connecting to static server '{}'", config.name);

        let handler = Arc::new(
            SmgClientHandler::new(
                &config.name,
                Arc::clone(&self.approval_manager),
                Arc::clone(&self.tool_inventory),
            )
            .with_refresh_channel(self.refresh_tx.clone()),
        );

        let client = self.connect_server_impl(config, (*handler).clone()).await?;
        let client = Arc::new(client);
        self.tool_inventory.clear_server_tools(&config.name);

        // Load tools from server
        self.load_server_inventory(&config.name, &client).await;

        self.apply_tool_configs(config);

        // Store server entry with config for builtin lookups
        self.static_servers.insert(
            config.name.clone(),
            ServerEntry {
                client,
                handler,
                config: config.clone(),
            },
        );

        self.metrics.record_connection_opened();
        info!("Connected to static server '{}'", config.name);
        Ok(())
    }

    /// Apply tool configurations from server config (aliases and arg mappings).
    fn apply_tool_configs(&self, config: &McpServerConfig) {
        let Some(tools) = &config.tools else {
            return;
        };

        for (tool_name, tool_config) in tools {
            // Check if the tool exists
            if !self
                .tool_inventory
                .has_tool_qualified(&config.name, tool_name)
            {
                warn!(
                    "Tool config for '{}:{}' but tool not found on server",
                    config.name, tool_name
                );
                continue;
            }

            let arg_mapping = tool_config.arg_mapping.as_ref().map(|cfg| {
                let mut mapping = ArgMapping::new();
                for (from, to) in &cfg.renames {
                    mapping = mapping.with_rename(from, to);
                }
                for (key, value) in &cfg.defaults {
                    mapping = mapping.with_default(key, value.clone());
                }
                for (key, value) in &cfg.overrides {
                    mapping = mapping.with_override(key, value.clone());
                }
                mapping
            });

            // If there's an alias, register it
            if let Some(alias_name) = &tool_config.alias {
                if let Err(e) =
                    self.register_alias(alias_name, &config.name, tool_name, arg_mapping)
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
            } else if let Some(mapping) = arg_mapping {
                // No alias but has arg_mapping - update entry directly
                if let Some(mut entry) = self.tool_inventory.get_entry(&config.name, tool_name) {
                    entry.arg_mapping = Some(mapping);
                    self.tool_inventory.insert_entry(entry);
                }
            }
        }
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

                handler.serve(transport).await.map_err(|e| {
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

                handler.serve(transport).await.map_err(|e| {
                    McpError::ConnectionFailed(format!("initialize streamable client: {e}"))
                })
            }
        }
    }

    /// Load tools, prompts, and resources from a server into the inventory.
    async fn load_server_inventory(&self, server_key: &str, client: &Arc<McpClientWithHandler>) {
        // Load tools
        match client.peer().list_all_tools().await {
            Ok(tools) => {
                info!("Discovered {} tools from '{}'", tools.len(), server_key);
                for tool in tools {
                    let entry = ToolEntry::from_server_tool(server_key, tool)
                        .with_category(ToolCategory::Static);
                    self.tool_inventory.insert_entry(entry);
                }
            }
            Err(e) => warn!("Failed to list tools from '{}': {}", server_key, e),
        }

        // Load prompts
        match client.peer().list_all_prompts().await {
            Ok(prompts) => {
                info!("Discovered {} prompts from '{}'", prompts.len(), server_key);
                for prompt in prompts {
                    self.tool_inventory.insert_prompt(
                        prompt.name.clone(),
                        server_key.to_string(),
                        prompt,
                    );
                }
            }
            Err(e) => debug!("No prompts from '{}': {}", server_key, e),
        }

        // Load resources
        match client.peer().list_all_resources().await {
            Ok(resources) => {
                info!(
                    "Discovered {} resources from '{}'",
                    resources.len(),
                    server_key
                );
                for resource in resources {
                    self.tool_inventory.insert_resource(
                        resource.uri.clone(),
                        server_key.to_string(),
                        resource.raw,
                    );
                }
            }
            Err(e) => debug!("No resources from '{}': {}", server_key, e),
        }
    }

    /// Spawn background handler for inventory refresh requests.
    fn spawn_refresh_handler(&self, mut rx: mpsc::Receiver<RefreshRequest>) {
        let token = self.shutdown_token.clone();
        let tool_inventory = Arc::clone(&self.tool_inventory);
        let static_servers = self.static_servers.clone();

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

                        if let Some(entry) = static_servers.get(&request.server_key) {
                            // Clear existing tools for this server
                            tool_inventory.clear_server_tools(&request.server_key);

                            // Reload tools
                            match entry.client.peer().list_all_tools().await {
                                Ok(tools) => {
                                    for tool in tools {
                                        let entry = ToolEntry::from_server_tool(&request.server_key, tool)
                                            .with_category(ToolCategory::Static);
                                        tool_inventory.insert_entry(entry);
                                    }
                                    info!(
                                        "Refreshed inventory for '{}': {} tools",
                                        request.server_key,
                                        tool_inventory.counts().0
                                    );
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to refresh tools for '{}': {}",
                                        request.server_key, e
                                    );
                                }
                            }
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

        // First, search connected static servers (dynamically registered via connect_static_server).
        // This handles servers registered after orchestrator initialization.
        for entry in &self.static_servers {
            if let Some(result) = extract_builtin(&entry.config) {
                return Some(result);
            }
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

    /// Returns the set of server names that have `builtin_type` configured.
    pub fn builtin_server_names(&self) -> HashSet<String> {
        let mut names = HashSet::new();

        // Check connected static servers first
        for entry in &self.static_servers {
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

        for entry in &self.static_servers {
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
        let start = Instant::now();
        let arguments_str = input.arguments.to_string();

        let qualified = QualifiedToolName::new(server_key, &input.tool_name);
        let entry = self.tool_inventory.get_entry(server_key, &input.tool_name);

        match entry {
            Some(entry) => match self
                .execute_tool_entry_result(&entry, qualified, input.arguments, request_ctx)
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
                let err = format!(
                    "Tool '{}' not found on server '{}'",
                    input.tool_name, server_key
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
        arguments: Value,
        request_ctx: &McpRequestContext<'_>,
    ) -> ToolExecutionResult {
        self.active_executions.fetch_add(1, Ordering::SeqCst);
        let _guard = scopeguard::guard(Arc::clone(&self.active_executions), |count| {
            count.fetch_sub(1, Ordering::SeqCst);
        });
        self.metrics.record_call_start(&qualified);
        let call_start_time = Instant::now();

        let result = match self
            .execute_tool_with_approval_raw_internal(entry, arguments, request_ctx)
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
        arguments: Value,
        request_ctx: &McpRequestContext<'_>,
    ) -> McpResult<ApprovalExecutionResult> {
        let approval_params = ApprovalParams {
            request_id: &request_ctx.request_id,
            server_key: entry.server_key(),
            elicitation_id: &format!("tool-{}", entry.tool_name()),
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
                let result = self.execute_tool_with_reconnect(entry, arguments).await?;
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
                        let result = self.execute_tool_with_reconnect(entry, arguments).await?;
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
    ) -> McpResult<CallToolResult> {
        let server_name = entry.server_key();

        // Capture the client instance we are about to use (to detect if it gets replaced)
        let initial_client = self
            .static_servers
            .get(server_name)
            .map(|e| Arc::clone(&e.client));

        match self.execute_tool_impl(entry, arguments.clone()).await {
            Ok(result) => Ok(result),
            Err(McpError::ServerDisconnected(name)) => {
                // Acquire/Create the mutex for this server to prevent concurrent reconnects
                let lock = self
                    .reconnection_locks
                    .entry(name.clone())
                    .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                    .value()
                    .clone();

                let _guard = lock.lock().await;

                // A client differing from the one that failed means another task
                // already reconnected it.
                if let Some(current_entry) = self.static_servers.get(&name) {
                    let already_reconnected = match &initial_client {
                        Some(initial) => !Arc::ptr_eq(initial, &current_entry.client),
                        None => true,
                    };

                    if already_reconnected {
                        debug!(
                            "Server '{}' already reconnected by another task, retrying call",
                            name
                        );
                        return self.execute_tool_impl(entry, arguments).await;
                    }
                }

                let server_config = self
                    .config
                    .servers
                    .iter()
                    .find(|s| s.name == name)
                    .cloned()
                    .ok_or_else(|| McpError::ServerNotFound(name.clone()))?;

                warn!(
                    "Server '{}' disconnected, initiating thread-safe recovery",
                    name
                );

                ReconnectionManager::default()
                    .reconnect(&name, || async {
                        self.static_servers.remove(&name);
                        self.connect_static_server(&server_config).await
                    })
                    .await?;

                // Retry execution after successful reconnection
                self.execute_tool_impl(entry, arguments).await
            }
            Err(e) => Err(e),
        }
    }

    /// Execute tool without approval (internal use).
    async fn execute_tool_impl(
        &self,
        entry: &ToolEntry,
        mut arguments: Value,
    ) -> McpResult<CallToolResult> {
        // Resolve alias if needed
        let (target_server, target_tool) = if let Some(alias) = &entry.alias_target {
            // Apply argument mapping
            if let Some(mapping) = &alias.arg_mapping {
                arguments = Self::apply_arg_mapping(arguments, mapping);
            }
            (
                alias.target.server_key().to_string(),
                alias.target.tool_name().to_string(),
            )
        } else {
            if let Some(mapping) = &entry.arg_mapping {
                arguments = Self::apply_arg_mapping(arguments, mapping);
            }
            (
                entry.server_key().to_string(),
                entry.tool_name().to_string(),
            )
        };

        // Coerce argument types based on tool schema
        // LLMs often return numbers as strings (e.g., "5" instead of 5)
        Self::coerce_arg_types(&mut arguments, &entry.tool.input_schema);

        // Build request. `CallToolRequestParams` is `#[non_exhaustive]` in rmcp
        // 1.7 (added `meta`/`task`), so construct via the builder.
        let mut request = CallToolRequestParams::new(Cow::Owned(target_tool));
        if let Value::Object(map) = arguments {
            request = request.with_arguments(map);
        }

        // Execute on server
        self.execute_on_server(&target_server, request).await
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
    ) -> McpResult<CallToolResult> {
        if let Some(entry) = self.static_servers.get(server_key) {
            return entry.client.call_tool(request).await.map_err(|e| match e {
                // Typed detection for transport-level failures
                ServiceError::TransportClosed | ServiceError::TransportSend(_) => {
                    McpError::ServerDisconnected(server_key.to_string())
                }
                _ => McpError::ToolExecution(format!("MCP call failed: {e}")),
            });
        }

        match self.connection_pool.get_unique_by_url(server_key) {
            UrlLookup::Found(client) => {
                return client.call_tool(request).await.map_err(|e| match e {
                    ServiceError::TransportClosed | ServiceError::TransportSend(_) => {
                        // Note: Pooled connections trigger Disconnected but
                        // recovery logic is currently scoped to static servers.
                        McpError::ServerDisconnected(server_key.to_string())
                    }
                    _ => McpError::ToolExecution(format!("MCP call failed: {e}")),
                });
            }
            UrlLookup::Ambiguous => {
                // Multiple pooled connections share this URL under different
                // credentials/tenants. Picking one would run the call over
                // another caller's authenticated connection, so fail closed.
                warn!(
                    "Refusing tool execution on '{}': multiple pooled connections share this \
                     URL with different credentials/tenants; cannot select one safely",
                    server_key
                );
                return Err(McpError::ServerAccessDenied(format!(
                    "ambiguous MCP connection for '{server_key}': multiple credentials or \
                     tenants are pooled for this URL"
                )));
            }
            UrlLookup::NotFound => {}
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
        // Verify target exists
        let target_entry = self
            .tool_inventory
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

        self.tool_inventory.insert_entry(alias_entry);

        // Also register in the aliases index
        self.tool_inventory.register_alias(
            alias_name.to_string(),
            QualifiedToolName::new(target_server, target_tool),
        );

        info!(
            "Registered alias '{}' → '{}:{}'",
            alias_name, target_server, target_tool
        );
        Ok(())
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
        for entry in &self.static_servers {
            entry.handler.set_request_context(ctx.clone());
        }
    }

    /// Clear request context from all static server handlers.
    pub fn clear_handler_contexts(&self) {
        for entry in &self.static_servers {
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
            McpTransport::Streamable { url, .. } => url.clone(),
            McpTransport::Sse { url, .. } => url.clone(),
            McpTransport::Stdio { command, args, .. } => {
                format!("{}:{}", command, args.join(" "))
            }
        }
    }

    /// Connect to a dynamic server and add it to the connection pool.
    ///
    /// This is used for per-request MCP servers specified in tool configurations.
    /// Returns the server key (URL) that can be used to reference the connection.
    ///
    /// The connection is keyed by (url, auth_hash, tenant_id) to ensure:
    /// - Different auth tokens get different connections
    /// - Different tenants are isolated
    pub async fn connect_dynamic_server(&self, config: McpServerConfig) -> McpResult<String> {
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
    ) -> McpResult<String> {
        use rmcp::{
            transport::{
                streamable_http_client::StreamableHttpClientTransportConfig,
                StreamableHttpClientTransport,
            },
            ServiceExt,
        };

        let pool_key = PoolKey::from_config(&config, tenant_id);

        // Check if already connected with same auth/tenant
        if self.connection_pool.contains(&pool_key) {
            return Ok(pool_key.url.clone());
        }

        // Extract server_key from pool_key to avoid double URL extraction
        let server_key = pool_key.url.clone();

        // Connect via the pool
        let inventory_clone = Arc::clone(&self.tool_inventory);
        let global_proxy = self.config.proxy.clone();

        let client = self
            .connection_pool
            .get_or_create(pool_key, config.clone(), |cfg, _proxy| async move {
                match &cfg.transport {
                    McpTransport::Streamable {
                        url,
                        token,
                        headers: custom_headers,
                    } => {
                        let proxy_config =
                            super::proxy::resolve_proxy_config(&cfg, global_proxy.as_ref());
                        let http_client =
                            build_http_client(proxy_config, token.as_deref(), custom_headers)?;
                        let cfg_http = StreamableHttpClientTransportConfig::with_uri(url.as_str());

                        let transport =
                            StreamableHttpClientTransport::with_client(http_client, cfg_http);

                        ().serve(transport)
                            .await
                            .map_err(|e| McpError::ConnectionFailed(format!("streamable: {e}")))
                    }
                    McpTransport::Sse { .. } => Err(sse_unsupported(&cfg.name)),
                    McpTransport::Stdio { .. } => Err(McpError::Transport(
                        "Stdio not supported for dynamic connections".to_string(),
                    )),
                }
            })
            .await?;

        // Pooled connections are keyed by URL.
        match client.peer().list_all_tools().await {
            Ok(tools) => {
                info!(
                    "Discovered {} tools from dynamic server '{}'",
                    tools.len(),
                    server_key
                );
                for tool in tools {
                    let entry = ToolEntry::from_server_tool(&server_key, tool)
                        .with_category(ToolCategory::Dynamic);
                    inventory_clone.insert_entry(entry);
                }
            }
            Err(e) => warn!("Failed to list tools from '{}': {}", server_key, e),
        }

        self.metrics.record_connection_opened();
        Ok(server_key)
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

    /// List tools for specific servers.
    pub fn list_tools_for_servers(&self, server_keys: &[String]) -> Vec<ToolEntry> {
        // For small server lists (typical case: 1-5), linear scan is faster than HashSet
        let is_allowed = |server_key: &str| -> bool { server_keys.iter().any(|s| s == server_key) };

        self.tool_inventory
            .list_tools()
            .into_iter()
            .filter_map(|(tool_name, server_key, _)| {
                if is_allowed(&server_key) {
                    self.tool_inventory.get_entry(&server_key, &tool_name)
                } else {
                    None
                }
            })
            .collect()
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

        for _ in &self.static_servers {
            self.metrics.record_connection_closed();
        }
        self.static_servers.clear();
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
            Arc::clone(&self.orchestrator.tool_inventory),
        );

        let client = self
            .orchestrator
            .connect_server_impl(config, handler)
            .await?;
        let client = Arc::new(client);

        // Load tools
        match client.peer().list_all_tools().await {
            Ok(tools) => {
                for tool in tools {
                    let entry = ToolEntry::from_server_tool(&config.name, tool)
                        .with_category(ToolCategory::Dynamic);
                    self.dynamic_tools
                        .insert(entry.qualified_name.clone(), entry);
                }
            }
            Err(e) => warn!("Failed to list tools from '{}': {}", config.name, e),
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
    use crate::core::config::{
        ArgMappingConfig, ResponseFormatConfig, Tool as McpTool, ToolConfig,
    };

    fn create_test_tool(name: &str) -> McpTool {
        use std::sync::Arc;

        McpTool::new(
            Cow::Owned(name.to_string()),
            Cow::Owned(format!("Test tool: {name}")),
            Arc::new(serde_json::Map::new()),
        )
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

        let hints = crate::annotations::ToolAnnotations::new();
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
    fn test_apply_tool_configs_sets_direct_arg_mapping() {
        let orchestrator = McpOrchestrator::new_test();
        orchestrator
            .tool_inventory
            .insert_entry(ToolEntry::from_server_tool(
                "web-search",
                create_test_tool("search_web"),
            ));

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
            name: "web-search".to_string(),
            transport: McpTransport::Sse {
                url: "http://localhost:8080/sse".to_string(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: Some(tools),
            builtin_type: None,
            builtin_tool_name: None,
            internal: false,
        };

        orchestrator.apply_tool_configs(&config);

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

    #[test]
    fn test_find_builtin_server_web_search() {
        use std::collections::HashMap;

        use crate::approval::{audit::AuditLog, policy::PolicyEngine};

        // Create config with a server configured for web_search_preview
        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "brave".to_string(),
                transport: McpTransport::Sse {
                    url: "https://mcp.brave.com/sse".to_string(),
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
                internal: false,
            }],
            ..Default::default()
        };

        let (refresh_tx, _) = mpsc::channel(10);
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::new(Arc::clone(&audit_log)));
        let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));

        let orchestrator = McpOrchestrator {
            static_servers: DashMap::new(),
            tool_inventory: Arc::new(ToolInventory::new()),
            approval_manager,
            connection_pool: Arc::new(McpConnectionPool::new()),
            metrics: Arc::new(McpMetrics::new()),
            refresh_tx,
            active_executions: Arc::new(AtomicUsize::new(0)),
            shutdown_token: CancellationToken::new(),
            reconnection_locks: DashMap::new(),
            config,
        };

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
        use std::collections::HashMap;

        use crate::{
            approval::{audit::AuditLog, policy::PolicyEngine},
            core::config::{ResponseFormatConfig, ToolConfig},
        };

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
                name: "custom-search".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: Some(tools),
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("my_search".to_string()),
                internal: false,
            }],
            ..Default::default()
        };

        let (refresh_tx, _) = mpsc::channel(10);
        let audit_log = Arc::new(AuditLog::new());
        let policy_engine = Arc::new(PolicyEngine::new(Arc::clone(&audit_log)));
        let approval_manager = Arc::new(ApprovalManager::new(policy_engine, audit_log));

        let orchestrator = McpOrchestrator {
            static_servers: DashMap::new(),
            tool_inventory: Arc::new(ToolInventory::new()),
            approval_manager,
            connection_pool: Arc::new(McpConnectionPool::new()),
            metrics: Arc::new(McpMetrics::new()),
            refresh_tx,
            active_executions: Arc::new(AtomicUsize::new(0)),
            shutdown_token: CancellationToken::new(),
            reconnection_locks: DashMap::new(),
            config,
        };

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
}
