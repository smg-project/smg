//! MCP Tool Session — bundles all MCP execution state for a single request.
//!
//! Instead of threading `orchestrator`, `request_ctx`, `mcp_servers`,
//! and `mcp_tools` through every function, callers create one `McpToolSession` and
//! pass `&session` everywhere. When an MCP parameter changes (e.g. `mcp_servers`
//! representation), only this struct and its constructor need updating — not every
//! router function signature.

use std::collections::{HashMap, HashSet};

use futures::stream::{self, StreamExt};

use super::{
    config::BuiltinToolType,
    orchestrator::{
        printable_label, McpOrchestrator, McpRequestContext, ToolExecutionInput,
        ToolExecutionOutput, ToolExecutionResult,
    },
    pool::PoolKey,
    UNKNOWN_SERVER_KEY,
};
use crate::{
    approval::ApprovalMode,
    error::McpError,
    inventory::{QualifiedToolName, ToolCategory, ToolEntry},
    tenant::TenantContext,
};

/// Default user-facing label for MCP servers when no explicit label is provided.
pub const DEFAULT_SERVER_LABEL: &str = "mcp";

/// Named pair of `(label, server_key)` for a connected MCP server.
///
/// Replaces the opaque `(String, String)` tuple that was threaded through
/// ~20 call sites, improving readability and preventing field-swap bugs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerBinding {
    /// User-facing label (e.g. the `server_label` from the request).
    pub label: String,
    /// Internal key used to look up the server in the orchestrator.
    pub server_key: String,
    /// Optional per-server tool allowlist.
    ///
    /// When `Some`, only the listed tool names are exposed for this server.
    /// When `None`, all tools from the server are exposed.
    pub allowed_tools: Option<Vec<String>>,
    /// Why the request cannot read the server's tools; `None` when it can.
    /// The connect sets its listing's failure; a session sets `Server not
    /// found` for a listed binding whose key the pool holds no more when the
    /// session is built.
    pub list_error: Option<String>,
    /// The pool key a dynamic server was connected with; `None` for a
    /// configured server. A tool call of the binding on the key's own
    /// address uses the connection pooled under that key and no other.
    pub pool_key: Option<PoolKey>,
}

#[derive(Debug, Clone)]
struct ExposedToolBinding {
    // INVARIANT:
    // - server_key: execution/runtime key used to invoke the tool.
    // - associated_server_key: classification/authorization key for alias mapping and privacy checks.
    // For alias entries these intentionally diverge. Privacy/classification logic MUST use
    // associated_server_key and MUST NOT reuse server_key.
    server_key: String,
    associated_server_key: String,
    server_label: String,
    /// The binding the tool is attributed to, by its place in the session's
    /// bindings: the label is its label, a call takes its pool key.
    binding_index: Option<usize>,
    resolved_tool_name: String,
    /// The alias target the session listed the tool with; `None` for a
    /// server's own tool. A call runs an entry that still holds it.
    alias_target: Option<QualifiedToolName>,
    is_builtin_routed: bool,
    approval_mode: ApprovalMode,
}

/// Bundles all MCP execution state for a single request.
///
/// Created once per request, then passed by reference to every function
/// that needs MCP infrastructure. This eliminates repeated parameter
/// threading of `orchestrator`, `request_ctx`, `mcp_servers`,
/// and `mcp_tools`.
pub struct McpToolSession<'a> {
    orchestrator: &'a McpOrchestrator,
    request_id: String,
    tenant_ctx: TenantContext,
    forwarded_headers: HashMap<String, String>,
    /// All MCP servers in this session (including builtin).
    all_mcp_servers: Vec<McpServerBinding>,
    /// Non-builtin MCP servers only — used for `mcp_list_tools` output.
    mcp_servers: Vec<McpServerBinding>,
    mcp_tools: Vec<ToolEntry>,
    exposed_name_map: HashMap<String, ExposedToolBinding>,
    exposed_name_by_qualified: HashMap<QualifiedToolName, String>,
    /// Internal server keys for this request snapshot.
    internal_server_keys: HashSet<String>,
    /// Builtin-routed server keys for this request snapshot.
    builtin_server_keys: HashSet<String>,
    /// Internal, non-builtin server labels for this request snapshot.
    internal_non_builtin_server_labels: HashSet<String>,
}

/// A binding with its allowlist as a set (`None` for an open binding), built
/// once per pass over the session's tools.
type LabelledAllowlist<'b> = (&'b McpServerBinding, Option<HashSet<&'b str>>);

impl<'a> McpToolSession<'a> {
    /// Create a new session by performing the setup every path currently repeats:
    /// 1. Create request context with default tenant and policy-only approval
    /// 2. List tools for the selected servers
    /// 3. Apply per-server allowed_tools filtering from bindings
    pub fn new(
        orchestrator: &'a McpOrchestrator,
        mcp_servers: Vec<McpServerBinding>,
        request_id: impl Into<String>,
    ) -> Self {
        Self::new_with_headers(orchestrator, mcp_servers, request_id, HashMap::new())
    }

    /// Create a new session with forwarded request headers preserved in the
    /// request context for downstream execution paths.
    pub fn new_with_headers(
        orchestrator: &'a McpOrchestrator,
        mut mcp_servers: Vec<McpServerBinding>,
        request_id: impl Into<String>,
        forwarded_headers: HashMap<String, String>,
    ) -> Self {
        let request_id = request_id.into();
        let tenant_ctx = TenantContext::default();
        let server_keys: Vec<String> = mcp_servers.iter().map(|b| b.server_key.clone()).collect();
        let mut mcp_tools = Self::collect_visible_mcp_tools(orchestrator, &server_keys);
        // A request's own server whose key the pool holds no more: its calls
        // find no server, and its listing says so, where it read as a listing
        // of no tools. Asked after the tools are read: a connection that
        // leaves between the two is told as well.
        for binding in &mut mcp_servers {
            if binding.list_error.is_none()
                && binding
                    .pool_key
                    .as_ref()
                    .is_some_and(|key| !orchestrator.is_pooled(key))
            {
                binding.list_error = Some(
                    McpError::ServerNotFound(printable_label(&binding.label).to_string())
                        .to_string(),
                );
            }
        }

        // The model's view of a server is the union of its bindings' allowlists
        // (a server bound under several labels lists each label's own set,
        // `list_tools_for_binding`); a binding without one leaves the server open.
        let mut open_server_keys: HashSet<&str> = HashSet::new();
        let mut allowed_tools_by_server_key: HashMap<&str, HashSet<&str>> = HashMap::new();
        for b in &mcp_servers {
            match b.allowed_tools.as_deref() {
                None => {
                    open_server_keys.insert(b.server_key.as_str());
                }
                Some(tools) => allowed_tools_by_server_key
                    .entry(b.server_key.as_str())
                    .or_default()
                    .extend(Self::allowlist(tools)),
            }
        }
        allowed_tools_by_server_key.retain(|key, _| !open_server_keys.contains(key));

        if !allowed_tools_by_server_key.is_empty() {
            mcp_tools.retain(|entry| {
                match allowed_tools_by_server_key.get(Self::associated_server_key(entry)) {
                    None => true,
                    Some(allowed) => Self::matches_allowed_tool_name(entry, allowed),
                }
            });
        }
        // The orchestrator is asked about the bound servers by name: a walk of
        // its server map takes one lock round and one allocation per shard,
        // and the shards grow with the host's cores.
        let internal_server_keys: HashSet<String> = mcp_servers
            .iter()
            .filter(|binding| orchestrator.is_internal_server(&binding.server_key))
            .map(|binding| binding.server_key.clone())
            .collect();
        let builtin_server_keys: HashSet<String> = mcp_servers
            .iter()
            .filter(|binding| orchestrator.is_builtin_server(&binding.server_key))
            .map(|binding| binding.server_key.clone())
            .collect();
        // A builtin binding names a tool of a builtin server, and a tool of
        // this session is a tool of a bound server, an alias by its target:
        // only an alias whose target's server is not bound can match without
        // a builtin server among the bindings.
        let may_route_builtin = !builtin_server_keys.is_empty()
            || mcp_tools.iter().any(|entry| {
                entry.alias_target.as_ref().is_some_and(|alias| {
                    !server_keys
                        .iter()
                        .any(|key| key == alias.target.server_key())
                })
            });
        let builtin_tool_bindings = if may_route_builtin {
            Self::builtin_tool_bindings(orchestrator)
        } else {
            HashSet::new()
        };
        let (exposed_name_map, exposed_name_by_qualified) =
            Self::build_exposed_function_tools(&mcp_tools, &mcp_servers, &builtin_tool_bindings);
        let internal_non_builtin_server_labels: HashSet<String> = mcp_servers
            .iter()
            .filter(|binding| {
                internal_server_keys.contains(&binding.server_key)
                    && !builtin_server_keys.contains(&binding.server_key)
            })
            .map(|binding| binding.label.clone())
            .collect();
        // Filter out servers configured with builtin_type from the visible list.
        let visible_mcp_servers: Vec<McpServerBinding> = mcp_servers
            .iter()
            .filter(|b| !builtin_server_keys.contains(&b.server_key))
            .cloned()
            .collect();

        Self {
            orchestrator,
            request_id,
            tenant_ctx,
            forwarded_headers,
            all_mcp_servers: mcp_servers,
            mcp_servers: visible_mcp_servers,
            mcp_tools,
            exposed_name_map,
            exposed_name_by_qualified,
            internal_server_keys,
            builtin_server_keys,
            internal_non_builtin_server_labels,
        }
    }

    // --- Accessors ---

    pub fn orchestrator(&self) -> &McpOrchestrator {
        self.orchestrator
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Returns only non-builtin MCP servers
    pub fn mcp_servers(&self) -> &[McpServerBinding] {
        &self.mcp_servers
    }

    /// Returns all MCP servers including builtin ones.
    pub fn all_mcp_servers(&self) -> &[McpServerBinding] {
        &self.all_mcp_servers
    }

    pub fn mcp_tools(&self) -> &[ToolEntry] {
        &self.mcp_tools
    }

    /// Returns true if the name is exposed to the model for this session.
    pub fn has_exposed_tool(&self, tool_name: &str) -> bool {
        self.exposed_name_map.contains_key(tool_name)
    }

    /// Returns the session's qualified-name -> exposed-name mapping.
    ///
    /// Router adapters should use this with response bridge builders.
    pub fn exposed_name_by_qualified(&self) -> &HashMap<QualifiedToolName, String> {
        &self.exposed_name_by_qualified
    }

    // --- Delegation methods ---

    /// Execute multiple tools concurrently using this session's exposed-name mapping.
    ///
    /// Uses `buffered()` to cap in-flight requests while preserving input ordering.
    pub async fn execute_tools(&self, inputs: Vec<ToolExecutionInput>) -> Vec<ToolExecutionOutput> {
        self.execute_tool_results(inputs)
            .await
            .into_iter()
            .map(ToolExecutionResult::into_output)
            .collect()
    }

    /// Execute multiple tools concurrently while preserving pending approval state.
    pub async fn execute_tool_results(
        &self,
        inputs: Vec<ToolExecutionInput>,
    ) -> Vec<ToolExecutionResult> {
        const MAX_IN_FLIGHT_TOOL_CALLS: usize = 8;
        stream::iter(inputs)
            .map(|input| self.execute_tool_result(input))
            .buffered(MAX_IN_FLIGHT_TOOL_CALLS)
            .collect()
            .await
    }

    /// Execute a single tool using this session's exposed-name mapping.
    pub async fn execute_tool(&self, input: ToolExecutionInput) -> ToolExecutionOutput {
        self.execute_tool_result(input).await.into_output()
    }

    /// Execute a single tool while preserving pending approval state.
    pub async fn execute_tool_result(&self, input: ToolExecutionInput) -> ToolExecutionResult {
        let invoked_name = input.tool_name.clone();
        let exposed = self.exposed_name_map.get(&invoked_name);
        // A session calls what it listed: the entry read here is the one the
        // call runs, and an entry whose alias target is not the listed one is
        // not this session's tool.
        let entry = exposed.and_then(|binding| {
            self.orchestrator
                .tool_inventory()
                .get_entry(&binding.server_key, &binding.resolved_tool_name)
        });
        let admitted = exposed.filter(|binding| {
            entry.as_ref().is_none_or(|now| {
                now.alias_target.as_ref().map(|alias| &alias.target)
                    == binding.alias_target.as_ref()
            })
        });

        if let Some(binding) = admitted {
            let resolved_tool_name = binding.resolved_tool_name.clone();
            let mut request_ctx = self.request_ctx_for(binding.approval_mode);
            // The call carries the pool key of the binding its tool is
            // attributed to.
            request_ctx.pool_key = binding
                .binding_index
                .and_then(|index| self.all_mcp_servers.get(index))
                .and_then(|b| b.pool_key.clone());
            let mut result = self
                .orchestrator
                .execute_resolved_entry_result(
                    entry,
                    ToolExecutionInput {
                        call_id: input.call_id,
                        tool_name: resolved_tool_name.clone(),
                        arguments: input.arguments,
                    },
                    &binding.server_key,
                    &binding.server_label,
                    &request_ctx,
                )
                .await;

            match &mut result {
                ToolExecutionResult::Executed(output) => {
                    output.tool_name = invoked_name;
                }
                ToolExecutionResult::PendingApproval(pending) => {
                    pending.tool_name = invoked_name;
                }
            }

            result
        } else {
            // A tool the session listed is refused under its own binding.
            let (server_key, server_label) = match exposed {
                Some(binding) => (binding.server_key.as_str(), binding.server_label.as_str()),
                None => (
                    UNKNOWN_SERVER_KEY,
                    self.all_mcp_servers
                        .first()
                        .map_or(DEFAULT_SERVER_LABEL, |b| b.label.as_str()),
                ),
            };
            let err = format!("Tool '{invoked_name}' is not in this session's exposed tool map");
            ToolExecutionResult::Executed(ToolExecutionOutput {
                call_id: input.call_id,
                tool_name: invoked_name.clone(),
                server_key: server_key.to_string(),
                server_label: server_label.to_string(),
                arguments_str: input.arguments.to_string(),
                output: serde_json::json!({ "error": &err }),
                is_error: true,
                error_message: Some(err),
                duration: std::time::Duration::default(),
            })
        }
    }

    /// Resolve the user-facing server label for a tool.
    ///
    /// Uses the orchestrator inventory to find the tool's server key, then maps
    /// it to the request's MCP server label. Falls back to the first server
    /// label (or [`DEFAULT_SERVER_LABEL`]).
    pub fn resolve_tool_server_label(&self, tool_name: &str) -> String {
        let fallback_label = self
            .all_mcp_servers
            .first()
            .map(|b| b.label.as_str())
            .unwrap_or(DEFAULT_SERVER_LABEL);

        self.exposed_name_map
            .get(tool_name)
            .map(|binding| binding.server_label.clone())
            .unwrap_or_else(|| fallback_label.to_string())
    }

    /// Set the approval mode of every tool that a binding labelled
    /// `server_label` admits, optionally narrowed by a second allowlist of
    /// tool names. A server bound under several labels attributes each tool
    /// to one of them, and every label that admits a tool still gates it: the
    /// stricter mode stays.
    pub fn set_approval_mode(
        &mut self,
        server_label: &str,
        allowed_tool_names: Option<&[String]>,
        mode: ApprovalMode,
    ) {
        let bindings: Vec<LabelledAllowlist<'_>> = self
            .all_mcp_servers
            .iter()
            .filter(|b| b.label == server_label)
            .map(|b| (b, b.allowed_tools.as_deref().map(Self::allowlist)))
            .collect();
        let names = allowed_tool_names.map(Self::allowlist);
        for entry in &self.mcp_tools {
            let key = Self::associated_server_key(entry);
            let admitted = bindings.iter().any(|(b, allowed)| {
                b.server_key == key
                    && allowed
                        .as_ref()
                        .is_none_or(|allowed| Self::matches_allowed_tool_name(entry, allowed))
            });
            if !admitted
                || names
                    .as_ref()
                    .is_some_and(|names| !Self::matches_allowed_tool_name(entry, names))
            {
                continue;
            }
            let Some(name) = self.exposed_name_by_qualified.get(&entry.qualified_name) else {
                continue;
            };
            if let Some(binding) = self
                .exposed_name_map
                .get_mut(name)
                .filter(|binding| binding.approval_mode == ApprovalMode::PolicyOnly)
            {
                binding.approval_mode = mode;
            }
        }
    }

    /// Returns true if the bound server label belongs to an internal server.
    pub fn is_internal_server_label(&self, server_label: &str) -> bool {
        self.all_mcp_servers.iter().any(|binding| {
            binding.label == server_label && self.is_internal_server_key(&binding.server_key)
        })
    }

    /// Returns true if the bound server label is internal and not builtin-routed.
    ///
    /// Use this helper in redaction paths so internal filtering behavior stays
    /// consistent across response assembly code paths.
    pub fn is_internal_non_builtin_server_label(&self, server_label: &str) -> bool {
        self.internal_non_builtin_server_labels
            .contains(server_label)
    }

    /// Returns true if the given tool resolves to an internal server.
    pub fn is_internal_tool(&self, tool_name: &str) -> bool {
        self.exposed_name_map
            .get(tool_name)
            .is_some_and(|binding| self.is_internal_server_key(&binding.associated_server_key))
    }

    /// Returns true if the bound server label belongs to a builtin-routed server.
    pub fn is_builtin_server_label(&self, server_label: &str) -> bool {
        self.all_mcp_servers.iter().any(|binding| {
            binding.label == server_label && self.builtin_server_keys.contains(&binding.server_key)
        })
    }

    /// Returns true if the given tool resolves to a builtin-routed server.
    pub fn is_builtin_tool(&self, tool_name: &str) -> bool {
        self.exposed_name_map
            .get(tool_name)
            .is_some_and(|binding| binding.is_builtin_routed)
    }

    /// Returns true if the given tool resolves to an internal, non-builtin server.
    pub fn is_internal_non_builtin_tool(&self, tool_name: &str) -> bool {
        self.is_internal_tool(tool_name) && !self.is_builtin_tool(tool_name)
    }

    fn is_internal_server_key(&self, server_key: &str) -> bool {
        self.internal_server_keys.contains(server_key)
    }

    /// List tools for a single server key: every label's, as the model sees
    /// them (`list_tools_for_binding` for one label's own).
    pub fn list_tools_for_server(&self, server_key: &str) -> Vec<ToolEntry> {
        // Use the session's pre-filtered tool snapshot for consistency.
        self.mcp_tools
            .iter()
            .filter(|entry| Self::associated_server_key(entry) == server_key)
            .cloned()
            .collect()
    }

    /// Why this request cannot read the tools of `server_label` (bound to
    /// `server_key`), for its `mcp_list_tools` item's `error`: the failure of
    /// the listing made at its connect, or `Server not found` when its key
    /// left the pool before the session was built; `None` otherwise.
    pub fn list_error_for_binding(&self, server_label: &str, server_key: &str) -> Option<String> {
        self.all_mcp_servers
            .iter()
            .find(|binding| binding.label == server_label && binding.server_key == server_key)
            .and_then(|binding| binding.list_error.clone())
    }

    /// The tools a binding lists: its server's visible tools narrowed to the
    /// binding's own allowlist (a server bound under several labels lists
    /// each label's own set; the model sees their union).
    pub fn list_tools_for_binding(&self, binding: &McpServerBinding) -> Vec<ToolEntry> {
        let allowed = binding.allowed_tools.as_deref().map(Self::allowlist);
        self.list_tools_for_server(&binding.server_key)
            .into_iter()
            .filter(|entry| {
                allowed
                    .as_ref()
                    .is_none_or(|allowed| Self::matches_allowed_tool_name(entry, allowed))
            })
            .collect()
    }

    /// The tools listed under `server_label` for `server_key`: the binding's
    /// own set when that label is bound, the server's tools otherwise.
    pub fn list_tools_for_label(&self, server_label: &str, server_key: &str) -> Vec<ToolEntry> {
        match self
            .all_mcp_servers
            .iter()
            .find(|b| b.label == server_label && b.server_key == server_key)
        {
            Some(binding) => self.list_tools_for_binding(binding),
            None => self.list_tools_for_server(server_key),
        }
    }

    /// Resolve an exposed tool name (post-alias) to its `QualifiedToolName`.
    pub fn qualified_name_for_exposed(&self, tool_name: &str) -> Option<QualifiedToolName> {
        let binding = self.exposed_name_map.get(tool_name)?;
        Some(QualifiedToolName::new(
            &binding.server_key,
            &binding.resolved_tool_name,
        ))
    }

    /// True when a tool name should be hidden because the underlying server
    /// is internal-non-builtin and the user didn't explicitly declare it as a
    /// function tool.
    pub fn should_hide_function_call_like(
        &self,
        name: &str,
        user_function_names: &HashSet<String>,
    ) -> bool {
        self.is_internal_tool(name) && !user_function_names.contains(name)
    }

    /// True when an `mcp_call`/`mcp_approval_request` item should be hidden.
    /// `name` is the tool name; `server_label` is the user-facing label.
    pub fn should_hide_mcp_call_like_by_label(&self, name: &str, server_label: &str) -> bool {
        let matches_internal_server = self.is_internal_non_builtin_server_label(server_label);
        if self.has_exposed_tool(name) {
            self.is_internal_non_builtin_tool(name)
        } else {
            matches_internal_server
        }
    }

    /// Variant of `should_hide_mcp_call_like_by_label` that takes the
    /// pre-resolved internal-server flag, used by JSON-shape filters that
    /// already inspected the `server_label` field.
    pub fn should_hide_mcp_call_like_by_server_flag(
        &self,
        name: &str,
        matches_internal_server: bool,
    ) -> bool {
        if self.has_exposed_tool(name) {
            self.is_internal_non_builtin_tool(name)
        } else {
            matches_internal_server
        }
    }

    fn build_exposed_function_tools(
        tools: &[ToolEntry],
        mcp_servers: &[McpServerBinding],
        builtin_tool_bindings: &HashSet<QualifiedToolName>,
    ) -> (
        HashMap<String, ExposedToolBinding>,
        HashMap<QualifiedToolName, String>,
    ) {
        let mut name_counts: HashMap<&str, usize> = HashMap::new();
        for entry in tools {
            *name_counts.entry(entry.tool_name()).or_insert(0) += 1;
        }

        let mut used_exposed_names: HashSet<String> = HashSet::with_capacity(tools.len());
        let mut name_suffixes: HashMap<String, usize> = HashMap::with_capacity(tools.len());
        let mut exposed_name_map: HashMap<String, ExposedToolBinding> =
            HashMap::with_capacity(tools.len());
        let mut exposed_name_by_qualified: HashMap<QualifiedToolName, String> =
            HashMap::with_capacity(tools.len());
        // Each binding's allowlist built once; the attribution reads it per tool.
        let bindings: Vec<LabelledAllowlist<'_>> = mcp_servers
            .iter()
            .map(|b| (b, b.allowed_tools.as_deref().map(Self::allowlist)))
            .collect();

        for entry in tools {
            let server_key = entry.server_key().to_string();
            let associated_server_key = Self::associated_server_key(entry);
            let binding_index = Self::binding_for_entry(entry, &bindings);
            let server_label = binding_index
                .and_then(|index| mcp_servers.get(index))
                .map_or(associated_server_key, |b| b.label.as_str())
                .to_string();
            let resolved_tool_name = entry.tool_name().to_string();
            let is_builtin_routed = Self::builtin_binding_for_entry(entry, builtin_tool_bindings);

            let base_exposed_name = if name_counts.get(entry.tool_name()).copied().unwrap_or(0) <= 1
            {
                resolved_tool_name.clone()
            } else {
                format!(
                    "mcp_{}_{}",
                    sanitize_tool_token(&server_label),
                    sanitize_tool_token(&resolved_tool_name)
                )
            };

            let suffix = name_suffixes.entry(base_exposed_name.clone()).or_insert(0);
            let mut exposed_name = if *suffix == 0 {
                base_exposed_name.clone()
            } else {
                format!("{base_exposed_name}_{suffix}")
            };
            while used_exposed_names.contains(&exposed_name) {
                *suffix += 1;
                exposed_name = format!("{base_exposed_name}_{suffix}");
            }
            used_exposed_names.insert(exposed_name.clone());

            exposed_name_by_qualified.insert(entry.qualified_name.clone(), exposed_name.clone());

            exposed_name_map.insert(
                exposed_name,
                ExposedToolBinding {
                    server_key,
                    associated_server_key: associated_server_key.to_string(),
                    server_label,
                    binding_index,
                    resolved_tool_name,
                    alias_target: entry.alias_target.as_ref().map(|a| a.target.clone()),
                    is_builtin_routed,
                    approval_mode: ApprovalMode::PolicyOnly,
                },
            );
        }

        (exposed_name_map, exposed_name_by_qualified)
    }

    fn collect_visible_mcp_tools(
        orchestrator: &McpOrchestrator,
        server_keys: &[String],
    ) -> Vec<ToolEntry> {
        let direct_tools = orchestrator.list_tools_for_servers(server_keys);
        let server_key_set: HashSet<&str> = server_keys.iter().map(String::as_str).collect();

        let mut aliases_by_target: HashMap<QualifiedToolName, Vec<ToolEntry>> = HashMap::new();
        for alias_entry in orchestrator
            .tool_inventory()
            .list_by_category(ToolCategory::Alias)
        {
            let Some(target) = alias_entry
                .alias_target
                .as_ref()
                .map(|alias| alias.target.clone())
            else {
                continue;
            };
            if !server_key_set.contains(target.server_key()) {
                continue;
            }
            aliases_by_target
                .entry(target)
                .or_default()
                .push(alias_entry);
        }

        let mut visible_tools = Vec::with_capacity(
            direct_tools.len() + aliases_by_target.values().map(Vec::len).sum::<usize>(),
        );

        for direct_entry in direct_tools {
            if let Some(mut alias_entries) = aliases_by_target.remove(&direct_entry.qualified_name)
            {
                visible_tools.append(&mut alias_entries);
            } else {
                visible_tools.push(direct_entry);
            }
        }

        for (_, mut alias_entries) in aliases_by_target {
            visible_tools.append(&mut alias_entries);
        }

        visible_tools
    }

    fn associated_server_key(entry: &ToolEntry) -> &str {
        entry
            .alias_target
            .as_ref()
            .map(|alias| alias.target.server_key())
            .unwrap_or_else(|| entry.server_key())
    }

    /// The binding a tool is attributed to, by its place in `bindings`: the
    /// first binding of its server whose allowlist admits it (a server bound
    /// under several labels lists each label's own tools); failing that, the
    /// server's first binding. `bindings` pairs each binding with its
    /// allowlist, built once.
    fn binding_for_entry(entry: &ToolEntry, bindings: &[LabelledAllowlist<'_>]) -> Option<usize> {
        let key = Self::associated_server_key(entry);
        let mut of_server = bindings
            .iter()
            .enumerate()
            .filter(|(_, (b, _))| b.server_key == key)
            .peekable();
        let first = of_server.peek().map(|(index, _)| *index);
        of_server
            .find(|(_, (_, allowed))| {
                allowed
                    .as_ref()
                    .is_none_or(|allowed| Self::matches_allowed_tool_name(entry, allowed))
            })
            .map(|(index, _)| index)
            .or(first)
    }

    /// A binding's allowlist as a set: the names trimmed, the empty dropped.
    fn allowlist(tools: &[String]) -> HashSet<&str> {
        tools
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn matches_allowed_tool_name(entry: &ToolEntry, allowed: &HashSet<&str>) -> bool {
        allowed.contains(entry.tool_name())
            || entry
                .alias_target
                .as_ref()
                .is_some_and(|alias| allowed.contains(alias.target.tool_name()))
    }

    fn builtin_tool_bindings(orchestrator: &McpOrchestrator) -> HashSet<QualifiedToolName> {
        BuiltinToolType::all()
            .into_iter()
            .filter_map(|builtin_type| orchestrator.find_builtin_server(builtin_type))
            .map(|(server_key, tool_name)| QualifiedToolName::new(server_key, tool_name))
            .collect()
    }

    fn builtin_binding_for_entry(
        entry: &ToolEntry,
        builtin_tool_bindings: &HashSet<QualifiedToolName>,
    ) -> bool {
        let target = entry
            .alias_target
            .as_ref()
            .map(|alias| &alias.target)
            .unwrap_or(&entry.qualified_name);
        builtin_tool_bindings.contains(target)
    }

    fn request_ctx_for(&self, approval_mode: ApprovalMode) -> McpRequestContext<'a> {
        self.orchestrator.create_request_context_with_headers(
            self.request_id.clone(),
            self.tenant_ctx.clone(),
            approval_mode,
            self.forwarded_headers.clone(),
        )
    }
}

fn sanitize_tool_token(input: &str) -> String {
    let mut out = String::with_capacity(input.len().max(1));
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let out = out.trim_matches('_');
    if out.is_empty() {
        "tool".to_string()
    } else {
        out.to_string()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/core/session/tests.rs"]
mod tests;
