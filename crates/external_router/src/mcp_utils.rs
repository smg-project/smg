//! Shared MCP utilities for routers.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use axum::response::Response;
use openai_protocol::responses::{McpAllowedTools, ResponseTool, ResponsesRequest};
use serde_json::{json, Value};
use smg_mcp::{BuiltinToolType, McpOrchestrator, McpServerBinding, McpServerConfig, McpTransport};
use tracing::{debug, warn};

use crate::{
    error,
    openai_bridge::{
        self, apply_hosted_tool_overrides, extract_hosted_tool_overrides, FormatRegistry,
        ResponseFormat,
    },
    worker::redacted_url,
};

/// Default maximum tool loop iterations (safety limit).
pub const DEFAULT_MAX_ITERATIONS: usize = 10;

/// Project the T11 `McpAllowedTools` union into the flat name list consumed by
/// the router-side `McpServerInput` and `McpServerBinding` allowlist paths.
///
/// Mapping (documented on the calling sites):
///   * `None` → `None` (no constraint; all tools exposed)
///   * `Some(List(names))` → `Some(names)`
///   * `Some(Filter { tool_names: Some(v), read_only: None })` → `Some(v)`
///   * `Some(Filter { read_only: Some(_), .. })` → `Some(vec![])` —
///     fail-closed whenever the caller specified a `read_only` restriction,
///     regardless of `tool_names`. smg has no `readOnlyHint`-based filter
///     implementation yet, so honoring only the `tool_names` half would
///     silently broaden exposure past caller intent (e.g. `{read_only: true,
///     tool_names: ["mutating_tool"]}` must NOT expose `mutating_tool`).
///     Narrow to nothing so the downstream retain path exposes none.
///   * `Some(Filter { tool_names: None, read_only: None })` → `None`
pub fn project_allowed_tools(value: Option<&McpAllowedTools>) -> Option<Vec<String>> {
    value.and_then(|at| match at {
        McpAllowedTools::List(names) => Some(names.clone()),
        McpAllowedTools::Filter(filter) => match (&filter.tool_names, &filter.read_only) {
            // Any `read_only` restriction fails closed: readOnlyHint-based
            // filtering is unimplemented, so we cannot safely project a subset.
            (_, Some(_)) => Some(Vec::new()),
            (Some(names), None) => Some(names.clone()),
            (None, None) => None,
        },
    })
}

/// Protocol-agnostic MCP server descriptor for connection setup.
///
/// Contains only the fields needed by [`connect_mcp_servers`]. Each router
/// converts its protocol-specific type into this struct.
pub struct McpServerInput {
    pub label: String,
    pub url: Option<String>,
    pub authorization: Option<String>,
    pub headers: HashMap<String, String>,
    /// Optional per-server tool allowlist.
    pub allowed_tools: Option<Vec<String>>,
}

/// Connect to MCP servers described by protocol-agnostic inputs.
///
/// For each input:
/// - If `url` is present, connects a dynamic MCP server (SSE or Streamable).
/// - If `url` is absent, binds the configured server held under the label; a
///   label no connected server is held under binds nothing (a warning names
///   it).
///
/// Returns a list of [`McpServerBinding`]s for successfully connected servers.
pub async fn connect_mcp_servers(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    format_registry: &FormatRegistry,
    inputs: &[McpServerInput],
) -> Vec<McpServerBinding> {
    let mut mcp_servers: Vec<McpServerBinding> = Vec::new();

    for input in inputs {
        // Case A: Dynamic Server (Has URL)
        if let Some(server_url) = input
            .url
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            if !(server_url.starts_with("http://") || server_url.starts_with("https://")) {
                warn!(
                    "Ignoring MCP server_url with unsupported scheme: {}",
                    redacted_url(server_url)
                );
                continue;
            }

            let token = input.authorization.clone();
            let headers = input.headers.clone();
            let server_url = server_url.to_string();

            let transport = if server_url.contains("/sse") {
                McpTransport::Sse {
                    url: server_url,
                    token,
                    headers,
                }
            } else {
                McpTransport::Streamable {
                    url: server_url,
                    token,
                    headers,
                }
            };

            let server_config = McpServerConfig {
                name: input.label.clone(),
                transport,
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: false,
            };

            let server_key = McpOrchestrator::server_key(&server_config);

            match openai_bridge::connect_dynamic_server(
                mcp_orchestrator,
                format_registry,
                server_config,
            )
            .await
            {
                Ok(connected) => {
                    // One binding per label: a server named under two labels
                    // shares its pooled client and lists under each label.
                    if !mcp_servers
                        .iter()
                        .any(|b| b.server_key == server_key && b.label == input.label)
                    {
                        mcp_servers.push(McpServerBinding {
                            label: input.label.clone(),
                            server_key,
                            allowed_tools: input.allowed_tools.clone(),
                            list_error: connected.list_error,
                            pool_key: Some(connected.pool_key),
                        });
                    }
                }
                Err(err) => {
                    warn!("Failed to connect MCP server {}: {}", server_key, err);
                }
            }
        }
        // Case B: a configured server named by its label (no URL). A label no
        // connected server is held under binds nothing: next to servers that
        // bind it is dropped with the warning, alone it leaves the request to
        // fail on the missing connection (`ensure_request_mcp_routing`),
        // instead of listing no tools under a key that resolves to nothing.
        else if !mcp_orchestrator.has_static_server(&input.label) {
            warn!(
                "Ignoring MCP tool '{}': no server_url and no connected MCP server of that name",
                input.label
            );
        } else if !mcp_servers.iter().any(|b| b.server_key == input.label) {
            mcp_servers.push(McpServerBinding {
                label: input.label.clone(),
                server_key: input.label.clone(),
                allowed_tools: input.allowed_tools.clone(),
                list_error: mcp_orchestrator.list_error_for_server(&input.label),
                pool_key: None,
            });
        }
    }

    mcp_servers
}

/// Routing information for a built-in tool type.
///
/// When a built-in tool type (web_search_preview, code_interpreter, file_search)
/// is configured to route to an MCP server, this struct holds the routing details.
#[derive(Debug, Clone)]
pub struct BuiltinToolRouting {
    /// The built-in tool type being routed.
    pub builtin_type: BuiltinToolType,
    /// The MCP server name to route to.
    pub server_name: String,
    /// The MCP tool name to call on the server.
    pub tool_name: String,
    /// The response format for transforming the output.
    pub response_format: ResponseFormat,
}

/// Collect routing information for built-in tools in a request.
///
/// Scans request tools for built-in types (web_search_preview, code_interpreter, file_search)
/// and looks up configured MCP servers to handle them.
///
/// # Arguments
/// * `mcp_orchestrator` - The MCP orchestrator with server configuration
/// * `tools` - Request tools to scan for built-in types
///
/// # Returns
/// Vector of routing information for built-in tools that have configured MCP servers.
/// Empty if no built-in tools are found or none have MCP server configurations.
pub fn collect_builtin_routing(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    format_registry: &FormatRegistry,
    tools: Option<&[ResponseTool]>,
) -> Vec<BuiltinToolRouting> {
    let Some(tools) = tools else {
        return Vec::new();
    };

    let mut routing = Vec::new();

    for tool in tools {
        let Some(builtin_type) = openai_bridge::builtin_type_for_response_tool(tool) else {
            continue;
        };

        if let Some((server_name, tool_name)) = mcp_orchestrator.find_builtin_server(builtin_type) {
            debug!(
                builtin_type = ?builtin_type,
                server = %server_name,
                tool = %tool_name,
                "Found MCP server for built-in tool type"
            );

            // ResponseFormat for the routed (server, tool) pair lives in the
            // gateway-side registry — populated at server-registration time
            // (see FormatRegistry::populate_from_server_config).
            let response_format = format_registry.lookup_by_names(&server_name, &tool_name);

            routing.push(BuiltinToolRouting {
                builtin_type,
                server_name,
                tool_name,
                response_format,
            });
        } else {
            warn!(
                builtin_type = %builtin_type,
                "Request includes built-in tool but no MCP server is configured for it"
            );
        }
    }

    routing
}

/// Extract builtin tool types from an OpenAI `ResponseTool` array.
///
/// Used by routers to determine which built-in tool types are present in
/// a request, for passing to [`ensure_mcp_servers`].
pub fn extract_builtin_types(tools: &[ResponseTool]) -> Vec<BuiltinToolType> {
    tools
        .iter()
        .filter_map(openai_bridge::builtin_type_for_response_tool)
        .collect()
}

/// True if `tools` carries any MCP-routed entry (declared MCP server or a
/// builtin family that the gateway intercepts via MCP).
///
/// Derived from [`extract_builtin_types`] so the predicate and the actual
/// routing path can't drift — adding a new builtin to the routing path
/// (e.g. `file_search`) makes this predicate cover it too.
pub fn request_uses_mcp_routing(tools: &[ResponseTool]) -> bool {
    tools.iter().any(|t| matches!(t, ResponseTool::Mcp(_)))
        || !extract_builtin_types(tools).is_empty()
}

/// Collect user-declared function tool names from a Responses request.
pub fn collect_user_function_names(request: &ResponsesRequest) -> HashSet<String> {
    request
        .tools
        .as_deref()
        .unwrap_or_default()
        .iter()
        .flat_map(|tool| match tool {
            ResponseTool::Function(function_tool) => vec![function_tool.function.name.clone()],
            ResponseTool::Namespace(namespace) => namespace
                .tools
                .iter()
                .filter_map(|member| {
                    if let openai_protocol::responses::NamespaceTool::Function(ft) = member {
                        Some(format!("{}.{}", namespace.name, ft.function.name))
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        })
        .collect()
}

/// Unified MCP server connection logic shared by all routers.
///
/// Connects dynamic/static MCP servers described by `inputs`, then adds
/// any static builtin servers for the given `builtin_types`.
///
/// Returns `Some(servers)` if at least one server is available, `None` otherwise.
pub async fn ensure_mcp_servers(
    orchestrator: &Arc<McpOrchestrator>,
    format_registry: &FormatRegistry,
    inputs: &[McpServerInput],
    builtin_types: &[BuiltinToolType],
) -> Option<Vec<McpServerBinding>> {
    let mut mcp_servers = connect_mcp_servers(orchestrator, format_registry, inputs).await;

    // Add builtin tool routing servers
    for &builtin_type in builtin_types {
        if let Some((server_name, tool_name)) = orchestrator.find_builtin_server(builtin_type) {
            debug!(
                builtin_type = ?builtin_type,
                server = %server_name,
                tool = %tool_name,
                "Adding static server for built-in tool routing"
            );
            if !mcp_servers.iter().any(|b| b.server_key == server_name) {
                let list_error = orchestrator.list_error_for_server(&server_name);
                mcp_servers.push(McpServerBinding {
                    label: server_name.clone(),
                    server_key: server_name,
                    allowed_tools: None,
                    list_error,
                    pool_key: None,
                });
            }
        } else {
            warn!(
                builtin_type = %builtin_type,
                "Request includes built-in tool but no MCP server is configured for it"
            );
        }
    }

    if mcp_servers.is_empty() {
        None
    } else {
        Some(mcp_servers)
    }
}

/// Convenience wrapper for OpenAI Responses API routers.
///
/// Extracts MCP server inputs and builtin types from `ResponseTool` array,
/// then delegates to [`ensure_mcp_servers`].
pub async fn ensure_request_mcp_client(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    format_registry: &FormatRegistry,
    tools: &[ResponseTool],
) -> Option<Vec<McpServerBinding>> {
    let inputs: Vec<McpServerInput> = tools
        .iter()
        .filter_map(|tool| match tool {
            ResponseTool::Mcp(mcp) => Some(McpServerInput {
                label: mcp.server_label.clone(),
                url: mcp.server_url.clone(),
                authorization: mcp.authorization.clone(),
                headers: mcp.headers.clone().unwrap_or_default(),
                // T11: project the `allowed_tools` union (List | Filter) into
                // the flat name list `McpServerInput` still expects. See
                // [`project_allowed_tools`] for the mapping table and the
                // rationale for the `read_only`-only fail-closed case.
                allowed_tools: project_allowed_tools(mcp.allowed_tools.as_ref()),
            }),
            _ => None,
        })
        .collect();

    let builtin_types = extract_builtin_types(tools);

    ensure_mcp_servers(mcp_orchestrator, format_registry, &inputs, &builtin_types).await
}

/// The Responses routes' MCP routing, one rule for the HTTP and the gRPC
/// routers: a request that names an `mcp` tool and binds no server is refused
/// on the missing connection (424 `connect_mcp_server_failed`) before anything
/// goes upstream; built-in tools without MCP routing pass through to the model
/// (`Ok(None)`).
pub async fn ensure_request_mcp_routing(
    mcp_orchestrator: &Arc<McpOrchestrator>,
    format_registry: &FormatRegistry,
    tools: &[ResponseTool],
) -> Result<Option<Vec<McpServerBinding>>, Response> {
    for (idx, tool) in tools.iter().enumerate() {
        if tool.allows_direct_calls() {
            continue;
        }
        let gateway_managed = matches!(tool, ResponseTool::Mcp(_))
            || openai_bridge::builtin_type_for_response_tool(tool)
                .is_some_and(|kind| mcp_orchestrator.find_builtin_server(kind).is_some());
        if gateway_managed {
            return Err(error::bad_request(
                "tool_not_supported",
                format!(
                    "Invalid 'tools[{idx}].allowed_callers': gateway-managed tools must allow the 'direct' caller."
                ),
            ));
        }
    }
    match ensure_request_mcp_client(mcp_orchestrator, format_registry, tools).await {
        Some(servers) => Ok(Some(servers)),
        None if tools
            .iter()
            .any(|tool| matches!(tool, ResponseTool::Mcp(_))) =>
        {
            warn!("Failed to connect to MCP servers: the request names an mcp tool and none bound");
            Err(error::failed_dependency(
                "connect_mcp_server_failed",
                "Failed to connect to MCP servers. Check server_url and authorization, or that \
                 a server_label without server_url names a configured server.",
            ))
        }
        None => Ok(None),
    }
}

/// Forward the caller's `user` identifier into hosted-tool dispatch arguments.
///
/// OpenAI's Responses API takes a top-level `user` field for end-user
/// attribution. We mirror that into the MCP dispatch payload for hosted
/// tools (image_generation, web_search_preview, web_search, code_interpreter,
/// file_search) so a downstream MCP server can attribute usage and enforce
/// per-user quotas.
///
/// Scope is hosted-tools only — `ResponseFormat::Passthrough` (plain MCP
/// function tools) is a no-op because plain tool schemas are caller-defined
/// and may not expect a `user` key. Surprising those servers with an
/// unsolicited field is worse than missing the feature.
///
/// Behavior:
/// - No-op if `response_format` is `Passthrough` (non-hosted).
/// - No-op if `user` is `None` or an empty string.
/// - No-op if `arguments` is not a JSON object.
/// - No-op if `arguments` already contains a `user` key — model-supplied
///   values win over the request-level identifier.
/// - Otherwise inserts `arguments["user"] = json!(user)`.
fn inject_user_into_hosted_args(
    arguments: &mut Value,
    response_format: ResponseFormat,
    user: Option<&str>,
) {
    if response_format.to_builtin_tool_type().is_none() {
        return;
    }
    let Some(user_value) = user.filter(|u| !u.is_empty()) else {
        return;
    };
    let Value::Object(args_map) = arguments else {
        return;
    };
    // Preserve a real model-supplied identifier, but treat an explicit
    // null (or absent key) as "no value" — when the synthesized function
    // tool exposes `user` as a parameter the model dutifully emits
    // `{"user": null}` even though the caller provided no value, and that
    // null should not block the request-level forwarding.
    if let Some(existing) = args_map.get("user") {
        if !existing.is_null() {
            debug!(
                "Hosted-tool dispatch args already include a non-null 'user'; \
                 preserving model-supplied value over request-level identifier"
            );
            return;
        }
    }
    args_map.insert("user".to_string(), json!(user_value));
}

/// Prepare an MCP dispatch payload for a hosted-tool call.
///
/// One call collapses the two per-dispatch mutations every router used to
/// repeat:
/// 1. Merge caller-declared hosted-tool overrides from `request_tools` into
///    `arguments` (e.g. an `image_generation` request's `size`/`quality`
///    pinning the model's tool-call args).
/// 2. Forward the request-level `user` identifier into `arguments` for
///    hosted tools so a downstream MCP server can attribute usage.
///
/// Both steps are no-ops when `response_format` is `Passthrough` (plain MCP
/// function tools); they are also no-ops on individual missing inputs
/// (no overrides, no `user`, non-object args, pre-existing `user`).
///
/// Routers should call this in place of the inline override + injection
/// pair. Keeping it here, in the external-router crate, avoids leaking
/// gateway-side concerns into `crates/mcp` while still giving every router
/// a single chokepoint.
pub fn prepare_hosted_dispatch_args(
    arguments: &mut Value,
    response_format: ResponseFormat,
    request_tools: &[ResponseTool],
    request_user: Option<&str>,
) {
    if let Some(kind) = response_format.to_builtin_tool_type() {
        if let Some(overrides) = extract_hosted_tool_overrides(request_tools, kind) {
            apply_hosted_tool_overrides(arguments, &overrides);
        }
    }
    inject_user_into_hosted_args(arguments, response_format, request_user);
}

#[cfg(test)]
#[path = "../tests/unit/mcp_utils/tests.rs"]
mod tests;
