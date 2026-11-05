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
        McpOrchestrator, McpRequestContext, ToolExecutionInput, ToolExecutionOutput,
        ToolExecutionResult,
    },
    UNKNOWN_SERVER_KEY,
};
use crate::{
    approval::ApprovalMode,
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
    resolved_tool_name: String,
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
        mcp_servers: Vec<McpServerBinding>,
        request_id: impl Into<String>,
        forwarded_headers: HashMap<String, String>,
    ) -> Self {
        let request_id = request_id.into();
        let tenant_ctx = TenantContext::default();
        let server_keys: Vec<String> = mcp_servers.iter().map(|b| b.server_key.clone()).collect();
        let mut mcp_tools = Self::collect_visible_mcp_tools(orchestrator, &server_keys);

        // Build per-server allowlists from bindings that specify allowed_tools.
        let allowed_tools_by_server_key: HashMap<&str, HashSet<&str>> = mcp_servers
            .iter()
            .filter_map(|b| {
                b.allowed_tools.as_ref().map(|tools| {
                    let set: HashSet<&str> = tools
                        .iter()
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    (b.server_key.as_str(), set)
                })
            })
            .collect();

        if !allowed_tools_by_server_key.is_empty() {
            mcp_tools.retain(|entry| {
                match allowed_tools_by_server_key.get(Self::associated_server_key(entry)) {
                    None => true,
                    Some(allowed) => Self::matches_allowed_tool_name(entry, allowed),
                }
            });
        }
        let builtin_tool_bindings = Self::builtin_tool_bindings(orchestrator);
        let (exposed_name_map, exposed_name_by_qualified) =
            Self::build_exposed_function_tools(&mcp_tools, &mcp_servers, &builtin_tool_bindings);
        let configured_internal_servers = orchestrator.internal_server_names();
        let configured_builtin_servers = orchestrator.builtin_server_names();
        let internal_server_keys: HashSet<String> = mcp_servers
            .iter()
            .filter_map(|binding| {
                if configured_internal_servers.contains(&binding.server_key) {
                    Some(binding.server_key.clone())
                } else {
                    None
                }
            })
            .collect();
        let builtin_server_keys: HashSet<String> = mcp_servers
            .iter()
            .filter_map(|binding| {
                if configured_builtin_servers.contains(&binding.server_key) {
                    Some(binding.server_key.clone())
                } else {
                    None
                }
            })
            .collect();
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
            .filter(|b| !configured_builtin_servers.contains(&b.server_key))
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

        if let Some(binding) = self.exposed_name_map.get(&invoked_name) {
            let resolved_tool_name = binding.resolved_tool_name.clone();
            let request_ctx = self.request_ctx_for(binding.approval_mode);
            let mut result = self
                .orchestrator
                .execute_tool_resolved_result(
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
            let fallback_label = self
                .all_mcp_servers
                .first()
                .map(|b| b.label.as_str())
                .unwrap_or(DEFAULT_SERVER_LABEL)
                .to_string();
            let err = format!("Tool '{invoked_name}' is not in this session's exposed tool map");
            ToolExecutionResult::Executed(ToolExecutionOutput {
                call_id: input.call_id,
                tool_name: invoked_name.clone(),
                server_key: UNKNOWN_SERVER_KEY.to_string(),
                server_label: fallback_label,
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

    /// Set the approval mode for every binding matching `server_label`,
    /// optionally narrowed to a subset of resolved tool names.
    pub fn set_approval_mode(
        &mut self,
        server_label: &str,
        allowed_tool_names: Option<&[String]>,
        mode: ApprovalMode,
    ) {
        for binding in self.exposed_name_map.values_mut() {
            if binding.server_label != server_label {
                continue;
            }
            if let Some(allowed) = allowed_tool_names {
                if !allowed.iter().any(|n| n == &binding.resolved_tool_name) {
                    continue;
                }
            }
            binding.approval_mode = mode;
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

    /// List tools for a single server key.
    ///
    /// Useful for emitting per-server `mcp_list_tools` items.
    pub fn list_tools_for_server(&self, server_key: &str) -> Vec<ToolEntry> {
        // Use the session's pre-filtered tool snapshot for consistency.
        self.mcp_tools
            .iter()
            .filter(|entry| Self::associated_server_key(entry) == server_key)
            .cloned()
            .collect()
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
        let server_labels: HashMap<&str, &str> = mcp_servers
            .iter()
            .map(|b| (b.server_key.as_str(), b.label.as_str()))
            .collect();

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

        for entry in tools {
            let server_key = entry.server_key().to_string();
            let associated_server_key = Self::associated_server_key(entry);
            let server_label = server_labels
                .get(associated_server_key)
                .copied()
                .unwrap_or(associated_server_key)
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
                    resolved_tool_name,
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
mod tests {
    use serde_json::json;

    use super::*;
    use crate::core::config::{test_support::sse_server, Tool as McpTool};

    #[test]
    fn test_session_creation_keeps_servers() {
        let orchestrator = McpOrchestrator::new_test();
        let mcp_servers = vec![
            McpServerBinding {
                label: "label1".to_string(),
                server_key: "key1".to_string(),
                allowed_tools: None,
            },
            McpServerBinding {
                label: "label2".to_string(),
                server_key: "key2".to_string(),
                allowed_tools: None,
            },
        ];

        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        assert_eq!(session.mcp_servers().len(), 2);
        assert_eq!(session.mcp_servers()[0].label, "label1");
        assert_eq!(session.mcp_servers()[0].server_key, "key1");
    }

    #[test]
    fn test_session_empty_servers() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");

        assert!(session.mcp_servers().is_empty());
        assert!(session.mcp_tools().is_empty());
    }

    #[test]
    fn test_session_creation_keeps_request_id() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");

        assert_eq!(session.request_id(), "test-request");
    }

    #[test]
    fn test_resolve_tool_server_label_fallback() {
        let orchestrator = McpOrchestrator::new_test();
        let mcp_servers = vec![McpServerBinding {
            label: "my_label".to_string(),
            server_key: "my_key".to_string(),
            allowed_tools: None,
        }];
        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        // Tool doesn't exist, should fall back to first label
        let label = session.resolve_tool_server_label("nonexistent_tool");
        assert_eq!(label, "my_label");
    }

    #[test]
    fn test_resolve_tool_server_label_no_servers() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new(&orchestrator, vec![], "test-request");

        // No servers, should fall back to DEFAULT_SERVER_LABEL
        let label = session.resolve_tool_server_label("nonexistent_tool");
        assert_eq!(label, DEFAULT_SERVER_LABEL);
    }

    fn create_test_tool(name: &str) -> McpTool {
        use std::{borrow::Cow, sync::Arc};

        McpTool::new(
            Cow::Owned(name.to_string()),
            Cow::Owned(format!("Test tool: {name}")),
            Arc::new(serde_json::Map::new()),
        )
    }

    #[test]
    fn test_has_exposed_tool_with_inventory() {
        let orchestrator = McpOrchestrator::new_test();

        // Insert a tool into the inventory
        let tool = create_test_tool("test_tool");
        let entry = ToolEntry::from_server_tool("server1", tool);
        orchestrator.tool_inventory().insert_entry(entry);

        let mcp_servers = vec![McpServerBinding {
            label: "label1".to_string(),
            server_key: "server1".to_string(),
            allowed_tools: None,
        }];
        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        assert!(session.has_exposed_tool("test_tool"));
        assert_eq!(session.mcp_tools().len(), 1);
    }

    #[tokio::test]
    async fn test_execute_tool_result_preserves_pending_approval() {
        let orchestrator = McpOrchestrator::new_test();

        let tool = create_test_tool("test_tool");
        let entry = ToolEntry::from_server_tool("server1", tool);
        orchestrator.tool_inventory().insert_entry(entry);

        let mut session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "label1".to_string(),
                server_key: "server1".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        session.set_approval_mode("label1", None, ApprovalMode::Interactive);

        let result = session
            .execute_tool_result(ToolExecutionInput {
                call_id: "call-1".to_string(),
                tool_name: "test_tool".to_string(),
                arguments: json!({"hello": "world"}),
            })
            .await;

        match result {
            ToolExecutionResult::PendingApproval(pending) => {
                assert_eq!(pending.call_id, "call-1");
                assert_eq!(pending.tool_name, "test_tool");
                assert_eq!(pending.server_key, "server1");
                assert_eq!(pending.server_label, "label1");
                assert_eq!(pending.approval_request.server_key, "server1");
                assert_eq!(pending.approval_request.tool_name, "test_tool");
            }
            ToolExecutionResult::Executed(output) => {
                panic!("expected pending approval, got executed result: {output:?}")
            }
        }
    }

    #[test]
    fn test_resolve_label_with_inventory() {
        let orchestrator = McpOrchestrator::new_test();

        let tool = create_test_tool("test_tool");
        let entry = ToolEntry::from_server_tool("server1", tool);
        orchestrator.tool_inventory().insert_entry(entry);

        let mcp_servers = vec![McpServerBinding {
            label: "my_server".to_string(),
            server_key: "server1".to_string(),
            allowed_tools: None,
        }];
        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        let label = session.resolve_tool_server_label("test_tool");
        assert_eq!(label, "my_server");
    }

    #[test]
    fn test_exposed_names_are_unique_for_tool_name_collisions() {
        let orchestrator = McpOrchestrator::new_test();

        let tool_a = create_test_tool("shared_tool");
        let tool_b = create_test_tool("shared_tool");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool("server1", tool_a));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool("server2", tool_b));

        let session = McpToolSession::new(
            &orchestrator,
            vec![
                McpServerBinding {
                    label: "alpha".to_string(),
                    server_key: "server1".to_string(),
                    allowed_tools: None,
                },
                McpServerBinding {
                    label: "beta".to_string(),
                    server_key: "server2".to_string(),
                    allowed_tools: None,
                },
            ],
            "test-request",
        );

        let name_a = session
            .exposed_name_by_qualified()
            .get(&QualifiedToolName::new("server1", "shared_tool"))
            .cloned()
            .expect("missing exposed name for server1 tool");
        let name_b = session
            .exposed_name_by_qualified()
            .get(&QualifiedToolName::new("server2", "shared_tool"))
            .cloned()
            .expect("missing exposed name for server2 tool");

        assert_ne!(name_a, name_b);
        assert_ne!(name_a, "shared_tool");
        assert_ne!(name_b, "shared_tool");
        assert!(session.has_exposed_tool(&name_a));
        assert!(session.has_exposed_tool(&name_b));
    }

    #[test]
    fn test_exposed_names_handle_pre_suffixed_name_conflicts() {
        let orchestrator = McpOrchestrator::new_test();

        let tool_base = create_test_tool("foo");
        let tool_suffixed = create_test_tool("foo_1");
        let tool_dup = create_test_tool("foo");

        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool("s1", tool_base));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool("s2", tool_suffixed));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool("s3", tool_dup));

        let session = McpToolSession::new(
            &orchestrator,
            vec![
                McpServerBinding {
                    label: "a".to_string(),
                    server_key: "s1".to_string(),
                    allowed_tools: None,
                },
                McpServerBinding {
                    label: "b".to_string(),
                    server_key: "s2".to_string(),
                    allowed_tools: None,
                },
                McpServerBinding {
                    label: "c".to_string(),
                    server_key: "s3".to_string(),
                    allowed_tools: None,
                },
            ],
            "test-request",
        );

        let exposed_names: HashSet<String> = session
            .exposed_name_by_qualified()
            .values()
            .cloned()
            .collect();
        assert_eq!(exposed_names.len(), 3);
    }

    // --- Builtin server filtering tests ---

    fn create_builtin_orchestrator() -> McpOrchestrator {
        use crate::core::config::{BuiltinToolType, McpConfig, McpServerConfig};

        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    builtin_type: Some(BuiltinToolType::WebSearchPreview),
                    builtin_tool_name: Some("brave_web_search".to_string()),
                    ..sse_server("brave-builtin", "http://localhost:8001/sse")
                },
                sse_server("regular-server", "http://localhost:3000/sse"),
            ],
            ..Default::default()
        };

        McpOrchestrator::new_test_with_config(config)
    }

    #[test]
    fn test_mcp_servers_filters_builtin() {
        let orchestrator = create_builtin_orchestrator();
        let mcp_servers = vec![
            McpServerBinding {
                label: "brave".to_string(),
                server_key: "brave-builtin".to_string(),
                allowed_tools: None,
            },
            McpServerBinding {
                label: "regular".to_string(),
                server_key: "regular-server".to_string(),
                allowed_tools: None,
            },
        ];

        let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

        // mcp_servers() should only return non-builtin servers
        let visible = session.mcp_servers();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].label, "regular");
        assert_eq!(visible[0].server_key, "regular-server");

        // all_mcp_servers() should return everything
        assert_eq!(session.all_mcp_servers().len(), 2);
    }

    #[test]
    fn test_is_builtin_tool_scoped_to_builtin_binding_only() {
        use crate::core::config::{BuiltinToolType, McpConfig, McpServerConfig};

        let orchestrator = McpOrchestrator::new_test_with_config(McpConfig {
            servers: vec![McpServerConfig {
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
                internal: true,
                ..sse_server("internal-server", "http://localhost:3000/sse")
            }],
            ..Default::default()
        });

        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                create_test_tool("brave_web_search"),
            ));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                create_test_tool("internal_non_builtin_tool"),
            ));

        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );

        assert!(session.is_builtin_tool("brave_web_search"));
        assert!(!session.is_builtin_tool("internal_non_builtin_tool"));
        assert!(session.is_internal_non_builtin_tool("internal_non_builtin_tool"));
    }

    #[test]
    fn test_is_builtin_tool_classifies_image_generation() {
        use crate::core::config::{BuiltinToolType, McpConfig, McpServerConfig};

        let orchestrator = McpOrchestrator::new_test_with_config(McpConfig {
            servers: vec![McpServerConfig {
                builtin_type: Some(BuiltinToolType::ImageGeneration),
                builtin_tool_name: Some("generate_image".to_string()),
                ..sse_server("image-server", "http://localhost:3000/sse")
            }],
            ..Default::default()
        });

        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "image-server",
                create_test_tool("generate_image"),
            ));

        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "image".to_string(),
                server_key: "image-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );

        assert!(session.is_builtin_tool("generate_image"));
        assert!(session.is_builtin_server_label("image"));
    }

    #[test]
    fn test_allowed_tools_filters_inventory_and_list_tools() {
        let orchestrator = McpOrchestrator::new_test();

        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server1",
                create_test_tool("brave_web_search"),
            ));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server1",
                create_test_tool("brave_local_search"),
            ));

        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "mock".to_string(),
                server_key: "server1".to_string(),
                allowed_tools: Some(vec!["brave_web_search".to_string()]),
            }],
            "test-request",
        );

        assert!(session.has_exposed_tool("brave_web_search"));
        assert!(!session.has_exposed_tool("brave_local_search"));
        assert_eq!(session.mcp_tools().len(), 1);

        let listed = session.list_tools_for_server("server1");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].tool_name(), "brave_web_search");
    }

    #[test]
    fn test_alias_tools_replace_target_tool_in_session_inventory() {
        let orchestrator = McpOrchestrator::new_test();

        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server1",
                create_test_tool("brave_web_search"),
            ));

        orchestrator
            .register_alias(
                "web_search",
                "server1",
                "brave_web_search",
                Some(
                    crate::inventory::ArgMapping::new()
                        .with_override("enable_brave", serde_json::json!(false)),
                ),
            )
            .expect("alias registration should succeed");

        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "brave".to_string(),
                server_key: "server1".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );

        assert!(session.has_exposed_tool("web_search"));
        assert!(!session.has_exposed_tool("brave_web_search"));
        assert_eq!(session.mcp_tools().len(), 1);
        assert_eq!(session.mcp_tools()[0].tool_name(), "web_search");
        assert_eq!(session.resolve_tool_server_label("web_search"), "brave");

        let listed = session.list_tools_for_server("server1");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].tool_name(), "web_search");
    }

    #[test]
    fn test_allowed_tools_accepts_alias_name() {
        let orchestrator = McpOrchestrator::new_test();

        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server1",
                create_test_tool("brave_web_search"),
            ));

        orchestrator
            .register_alias("web_search", "server1", "brave_web_search", None)
            .expect("alias registration should succeed");

        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "brave".to_string(),
                server_key: "server1".to_string(),
                allowed_tools: Some(vec!["web_search".to_string()]),
            }],
            "test-request",
        );

        assert!(session.has_exposed_tool("web_search"));
        assert_eq!(session.mcp_tools().len(), 1);
        assert_eq!(session.mcp_tools()[0].tool_name(), "web_search");
    }

    #[test]
    fn test_is_internal_tool_for_internal_server() {
        use crate::core::config::{McpConfig, McpServerConfig};

        let config = McpConfig {
            servers: vec![McpServerConfig {
                internal: true,
                ..sse_server("internal-server", "http://localhost:3000/sse")
            }],
            ..Default::default()
        };

        let orchestrator = McpOrchestrator::new_test_with_config(config);
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                create_test_tool("internal_search"),
            ));

        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );

        assert!(session.has_exposed_tool("internal_search"));
        assert!(session.is_internal_tool("internal_search"));
        assert!(session.is_internal_server_label("internal-label"));
    }

    #[test]
    fn test_is_internal_tool_for_alias_targeting_internal_server() {
        use crate::core::config::{McpConfig, McpServerConfig};

        let config = McpConfig {
            servers: vec![McpServerConfig {
                internal: true,
                ..sse_server("internal-server", "http://localhost:3000/sse")
            }],
            ..Default::default()
        };

        let orchestrator = McpOrchestrator::new_test_with_config(config);
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                create_test_tool("internal_search"),
            ));

        orchestrator
            .register_alias("alias_search", "internal-server", "internal_search", None)
            .expect("alias registration should succeed");

        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );

        assert!(session.has_exposed_tool("alias_search"));
        assert!(session.is_internal_tool("alias_search"));
    }

    #[test]
    fn test_is_internal_server_label_checks_all_bindings_for_shared_label() {
        use crate::core::config::{McpConfig, McpServerConfig};

        let config = McpConfig {
            servers: vec![
                sse_server("public-server", "http://localhost:3000/sse"),
                McpServerConfig {
                    internal: true,
                    ..sse_server("internal-server", "http://localhost:3001/sse")
                },
            ],
            ..Default::default()
        };

        let orchestrator = McpOrchestrator::new_test_with_config(config);
        let session = McpToolSession::new(
            &orchestrator,
            vec![
                McpServerBinding {
                    label: "shared-label".to_string(),
                    server_key: "public-server".to_string(),
                    allowed_tools: None,
                },
                McpServerBinding {
                    label: "shared-label".to_string(),
                    server_key: "internal-server".to_string(),
                    allowed_tools: None,
                },
            ],
            "test-request",
        );

        assert!(
            session.is_internal_server_label("shared-label"),
            "shared labels should be internal when any matching binding is internal"
        );
    }

    #[test]
    fn test_allowed_tools_filters_only_target_server() {
        let orchestrator = McpOrchestrator::new_test();

        // server1 has two tools
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server1",
                create_test_tool("brave_web_search"),
            ));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server1",
                create_test_tool("brave_local_search"),
            ));

        // server2 has two tools
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server2",
                create_test_tool("deepwiki_search"),
            ));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "server2",
                create_test_tool("deepwiki_read"),
            ));

        let session = McpToolSession::new(
            &orchestrator,
            vec![
                McpServerBinding {
                    label: "brave".to_string(),
                    server_key: "server1".to_string(),
                    allowed_tools: Some(vec!["brave_web_search".to_string()]),
                },
                McpServerBinding {
                    label: "deepwiki".to_string(),
                    server_key: "server2".to_string(),
                    allowed_tools: None,
                },
            ],
            "test-request",
        );

        // server1 is filtered
        assert!(session.has_exposed_tool("brave_web_search"));
        assert!(!session.has_exposed_tool("brave_local_search"));
        let listed_server1 = session.list_tools_for_server("server1");
        assert_eq!(listed_server1.len(), 1);
        assert_eq!(listed_server1[0].tool_name(), "brave_web_search");

        // server2 is unfiltered
        assert!(session.has_exposed_tool("deepwiki_search"));
        assert!(session.has_exposed_tool("deepwiki_read"));
        let listed_server2 = session.list_tools_for_server("server2");
        assert_eq!(listed_server2.len(), 2);
    }

    #[test]
    fn test_session_preserves_forwarded_headers_in_request_context() {
        let orchestrator = McpOrchestrator::new_test();
        let session = McpToolSession::new_with_headers(
            &orchestrator,
            vec![],
            "test-request",
            HashMap::from([
                ("openai-project".to_string(), "project-123".to_string()),
                ("opc-request-id".to_string(), "req-123".to_string()),
            ]),
        );
        let request_ctx = session.request_ctx_for(ApprovalMode::PolicyOnly);

        assert_eq!(
            request_ctx.forwarded_headers.get("openai-project"),
            Some(&"project-123".to_string())
        );
        assert_eq!(
            request_ctx.forwarded_headers.get("opc-request-id"),
            Some(&"req-123".to_string())
        );
    }
}
