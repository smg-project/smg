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
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc,
        },
        time::Duration,
    };

    use axum::http::StatusCode;
    use openai_protocol::{
        common::Function,
        responses::{
            CodeInterpreterTool, FunctionTool, ImageGenerationTool, McpTool, ResponseOutputItem,
            ResponseTool, WebSearchPreviewTool,
        },
    };
    use serde_json::json;
    use smg_mcp::{
        McpConfig, McpToolSession, ResponseFormatConfig, ToolConfig, ToolExecutionInput,
    };
    use tokio::sync::{oneshot, Notify};

    use super::*;

    /// Create a test orchestrator with a built-in server configuration,
    /// plus a `FormatRegistry` mirroring the same per-tool ResponseFormat
    /// values that production code populates at server-registration time.
    async fn create_test_orchestrator_with_builtin() -> (Arc<McpOrchestrator>, FormatRegistry) {
        let mut tools_config = HashMap::new();
        tools_config.insert(
            "web_search".to_string(),
            ToolConfig {
                response_format: Some(ResponseFormatConfig::WebSearchCall),
                ..Default::default()
            },
        );

        let server = McpServerConfig {
            name: "search-server".to_string(),
            transport: McpTransport::Streamable {
                url: "http://localhost:9999/mcp".to_string(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: Some(tools_config),
            builtin_type: Some(BuiltinToolType::WebSearchPreview),
            builtin_tool_name: Some("web_search".to_string()),
            internal: false,
        };

        let config = McpConfig {
            servers: vec![server.clone()],
            pool: Default::default(),
            proxy: None,
            warmup: Vec::new(),
            inventory: Default::default(),
            policy: Default::default(),
        };

        let registry = FormatRegistry::new();
        registry.populate_from_server_config(&server);

        // Note: This will fail to connect but still create the orchestrator with config
        (
            Arc::new(McpOrchestrator::new(config).await.unwrap()),
            registry,
        )
    }

    /// Create a test orchestrator without built-in server configuration
    async fn create_test_orchestrator_no_builtin() -> (Arc<McpOrchestrator>, FormatRegistry) {
        let config = McpConfig {
            servers: vec![],
            pool: Default::default(),
            proxy: None,
            warmup: Vec::new(),
            inventory: Default::default(),
            policy: Default::default(),
        };

        (
            Arc::new(McpOrchestrator::new(config).await.unwrap()),
            FormatRegistry::new(),
        )
    }

    /// What the local MCP server answers to `tools/list`: `tools` (each with
    /// an empty schema), or a JSON-RPC error when `tools` is `None`, when the
    /// request carries the `refused_bearer` token, or while `refusing` is
    /// set. `arrived` is signalled when a `tools/list` arrives, after its
    /// refusal is decided; a listing arriving while `hold` is holding waits
    /// for its release; the answer comes after `list_delay`. With `changed`
    /// the server opens a session, and its notification stream sends
    /// `notifications/tools/list_changed` at each signal. With `numbered`
    /// each tool lists under its name and the number held (`echo-1`), so a
    /// listing shows its moment. The server has no prompts and answers so; a
    /// `prompts/list` arriving while `prompts_gate` holds signals the gate
    /// and waits for its release first. With `drops_calls` a `tools/call`
    /// fails at the transport (an HTTP 500), which its caller takes as a
    /// dropped connection.
    #[derive(Clone, Default)]
    struct LocalServer {
        tools: Option<&'static [&'static str]>,
        /// While set, a listing leaves out the first of `tools` and
        /// describes the others anew.
        relisted: Option<Arc<AtomicBool>>,
        refused_bearer: Option<&'static str>,
        refusing: Option<Arc<AtomicBool>>,
        list_delay: Duration,
        arrived: Option<Arc<Notify>>,
        hold: Option<Arc<Hold>>,
        changed: Option<Arc<Notify>>,
        numbered: Option<Arc<AtomicUsize>>,
        prompts_gate: Option<Arc<Gate>>,
        drops_calls: bool,
    }

    /// A gate the local server's `tools/list` waits at while `holding` is
    /// set, until `release` is signalled.
    #[derive(Default)]
    struct Hold {
        holding: AtomicBool,
        release: Notify,
    }

    /// A [`Hold`] that tells when a listing reached it: each listing it
    /// stops signals `arrived` first.
    #[derive(Default)]
    struct Gate {
        arrived: Notify,
        hold: Hold,
    }

    /// A local MCP server over HTTP that initializes and answers `tools/list`
    /// as `server` says; the URL it serves on.
    async fn serve_local_mcp_server(server: LocalServer) -> String {
        use axum::{
            extract::State,
            http::{header::AUTHORIZATION, HeaderMap, HeaderValue, StatusCode},
            routing::post,
            Json, Router,
        };

        async fn mcp(
            State(server): State<LocalServer>,
            headers: HeaderMap,
            Json(message): Json<Value>,
        ) -> Response {
            use axum::response::IntoResponse;
            let id = message.get("id").cloned().unwrap_or(Value::Null);
            if message["method"] == "prompts/list" {
                let gate = server.prompts_gate.as_ref();
                if let Some(gate) = gate.filter(|gate| gate.hold.holding.load(Ordering::SeqCst)) {
                    gate.arrived.notify_one();
                    gate.hold.release.notified().await;
                }
            }
            let mut response = match message.get("method").and_then(Value::as_str) {
                Some("initialize") => Json(json!({
                    "jsonrpc": "2.0", "id": id,
                    "result": {
                        "protocolVersion": message["params"]["protocolVersion"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "listing-fails", "version": "1"}
                    }
                }))
                .into_response(),
                Some("tools/list") => {
                    let refused = server
                        .refusing
                        .as_ref()
                        .is_some_and(|flag| flag.load(Ordering::SeqCst))
                        || server.refused_bearer.is_some_and(|token| {
                            headers
                                .get(AUTHORIZATION)
                                .and_then(|value| value.to_str().ok())
                                .is_some_and(|value| value == format!("Bearer {token}"))
                        });
                    let held = server
                        .hold
                        .as_ref()
                        .filter(|hold| hold.holding.load(Ordering::SeqCst));
                    if let Some(arrived) = &server.arrived {
                        arrived.notify_one();
                    }
                    if let Some(hold) = held {
                        hold.release.notified().await;
                    }
                    if !server.list_delay.is_zero() {
                        tokio::time::sleep(server.list_delay).await;
                    }
                    let relisted = server
                        .relisted
                        .as_ref()
                        .is_some_and(|flag| flag.load(Ordering::SeqCst));
                    let note = if relisted { ", relisted" } else { "" };
                    let moment = server
                        .numbered
                        .as_ref()
                        .map(|moment| moment.load(Ordering::SeqCst));
                    let numbered = |name: &str| match moment {
                        Some(moment) => format!("{name}-{moment}"),
                        None => name.to_string(),
                    };
                    match server.tools {
                        Some(names) if !refused => Json(json!({
                            "jsonrpc": "2.0", "id": id,
                            "result": {"tools": names.iter().skip(usize::from(relisted)).map(|name| json!({
                                "name": numbered(name),
                                "description": format!("the {name} tool{note}"),
                                "inputSchema": {"type": "object", "properties": {}}
                            })).collect::<Vec<_>>()}
                        }))
                        .into_response(),
                        _ => Json(json!({
                            "jsonrpc": "2.0", "id": id,
                            "error": {"code": -32603, "message": "tools/list refused by the server"}
                        }))
                        .into_response(),
                    }
                }
                Some("tools/call") if server.drops_calls => {
                    StatusCode::INTERNAL_SERVER_ERROR.into_response()
                }
                Some(method) if method.starts_with("notifications/") => {
                    StatusCode::ACCEPTED.into_response()
                }
                _ => Json(json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": {"code": -32601, "message": "method not found"}
                }))
                .into_response(),
            };
            // A session makes the client open the notification stream.
            if server.changed.is_some() {
                response
                    .headers_mut()
                    .insert("mcp-session-id", HeaderValue::from_static("local"));
            }
            response
        }

        /// The notification stream: a `tools/list_changed` at each signal.
        async fn changes(State(server): State<LocalServer>) -> Response {
            use axum::response::{
                sse::{Event, Sse},
                IntoResponse,
            };
            let Some(changed) = server.changed else {
                return StatusCode::METHOD_NOT_ALLOWED.into_response();
            };
            Sse::new(futures::stream::unfold(changed, |changed| async move {
                changed.notified().await;
                let notice =
                    json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"});
                Some((Event::default().json_data(notice), changed))
            }))
            .into_response()
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a loopback port of the system's choice");
        let port = listener.local_addr().expect("the bound address").port();
        #[expect(
            clippy::disallowed_methods,
            reason = "a test server that lives as long as the test's runtime; nothing waits for it"
        )]
        tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new()
                    .route("/mcp", post(mcp).get(changes))
                    .with_state(server),
            )
            .await
            .expect("the local MCP server serves");
        });
        format!("http://127.0.0.1:{port}/mcp")
    }

    /// The `error` of the `mcp_list_tools` item a session over `bindings`
    /// builds for `label`, as a request's response carries it.
    fn list_item_error(
        orchestrator: &McpOrchestrator,
        bindings: Vec<McpServerBinding>,
        label: &str,
    ) -> Option<Value> {
        let server_key = bindings
            .iter()
            .find(|binding| binding.label == label)
            .expect("the label is bound")
            .server_key
            .clone();
        let session = McpToolSession::new(orchestrator, bindings, "test-request");
        openai_bridge::mcp_list_tools_json(&session, label, &server_key)
            .get("error")
            .cloned()
    }

    /// A server that connects but cannot list its tools is a binding whose
    /// `mcp_list_tools` item carries the listing's failure as its `error`,
    /// in the typed item and in the JSON form alike, with no tools; the
    /// request is not refused (the server is reachable) and the model is
    /// told nothing of tools that could not be listed.
    #[tokio::test]
    async fn a_failed_tools_list_is_the_list_items_error() {
        let (orchestrator, format_registry) = create_test_orchestrator_no_builtin().await;
        let url = serve_local_mcp_server(LocalServer::default()).await;
        let inputs = [McpServerInput {
            label: "conf".to_string(),
            url: Some(url),
            authorization: None,
            headers: HashMap::new(),
            allowed_tools: None,
        }];

        let bindings = connect_mcp_servers(&orchestrator, &format_registry, &inputs).await;
        assert_eq!(bindings.len(), 1, "a reachable server is bound");
        let server_key = bindings[0].server_key.clone();
        let session = McpToolSession::new(&orchestrator, bindings, "test-request");

        let item = openai_bridge::mcp_list_tools_item(&session, "conf", &server_key);
        let ResponseOutputItem::McpListTools { tools, error, .. } = &item else {
            panic!("a list-tools item, got {item:?}");
        };
        assert!(tools.is_empty());
        let error = error
            .as_deref()
            .expect("the listing's failure is the item's error");
        assert!(
            error.contains("tools/list refused by the server"),
            "the error names the server's message: {error}"
        );

        let json = openai_bridge::mcp_list_tools_json(&session, "conf", &server_key);
        assert_eq!(json["error"].as_str(), Some(error));
        assert_eq!(json["tools"], json!([]));
    }

    /// A binding keeps the outcome of its own connect: with a pool of one, the
    /// second server's connect evicts the first's connection, and the first's
    /// item still carries its listing's error while the second's carries
    /// none. The read goes through the item builder, as a request's does.
    #[tokio::test]
    async fn a_bindings_listing_failure_survives_its_connections_eviction() {
        let mut config = McpConfig::default();
        config.pool.max_connections = 1;
        let orchestrator = Arc::new(McpOrchestrator::new(config).await.unwrap());
        let format_registry = FormatRegistry::new();
        let refusing = serve_local_mcp_server(LocalServer::default()).await;
        let listing = serve_local_mcp_server(LocalServer {
            tools: Some(&["echo"]),
            ..Default::default()
        })
        .await;
        let input = |label: &str, url: &str| McpServerInput {
            label: label.to_string(),
            url: Some(url.to_string()),
            authorization: None,
            headers: HashMap::new(),
            allowed_tools: None,
        };
        let item_error = |bindings, label| list_item_error(&orchestrator, bindings, label);

        let first =
            connect_mcp_servers(&orchestrator, &format_registry, &[input("a", &refusing)]).await;
        assert_eq!(first.len(), 1);
        assert!(
            item_error(first.clone(), "a").is_some_and(|error| error.is_string()),
            "the failure stands while its connection is pooled"
        );

        let second =
            connect_mcp_servers(&orchestrator, &format_registry, &[input("b", &listing)]).await;
        assert_eq!(second.len(), 1);
        let bindings = [first, second].concat();
        assert_eq!(item_error(bindings.clone(), "b"), None);
        assert!(
            item_error(bindings, "a").is_some_and(|error| error.is_string()),
            "the first binding keeps the failure its own connect saw"
        );
    }

    /// One URL under two bearer tokens is two connections (the pool key takes
    /// the auth): the server refuses `tools/list` for one token and lists for
    /// the other, and the request that listed reads no error on its item
    /// while the refused one reads its own; neither reads the other's.
    #[tokio::test]
    async fn a_listing_refused_for_one_caller_is_not_anothers_error() {
        let (orchestrator, format_registry) = create_test_orchestrator_no_builtin().await;
        let url = serve_local_mcp_server(LocalServer {
            tools: Some(&["echo"]),
            refused_bearer: Some("refused-token"),
            ..Default::default()
        })
        .await;
        let input = |label: &str, token: &str| McpServerInput {
            label: label.to_string(),
            url: Some(url.clone()),
            authorization: Some(token.to_string()),
            headers: HashMap::new(),
            allowed_tools: None,
        };
        let item_error = |bindings: Vec<McpServerBinding>| {
            let label = bindings.first().expect("one binding").label.clone();
            list_item_error(&orchestrator, bindings, &label)
        };

        let refused = connect_mcp_servers(
            &orchestrator,
            &format_registry,
            &[input("a", "refused-token")],
        )
        .await;
        let listed =
            connect_mcp_servers(&orchestrator, &format_registry, &[input("b", "good-token")]).await;
        assert_eq!((refused.len(), listed.len()), (1, 1));
        assert_eq!(
            refused[0].server_key, listed[0].server_key,
            "one URL, two connections"
        );
        assert!(
            item_error(refused.clone()).is_some_and(|error| error.is_string()),
            "the refused caller's item carries the refusal"
        );
        assert_eq!(
            item_error(listed),
            None,
            "a refusal another caller met is not this caller's error"
        );
    }

    /// A pooled connection whose listing failed lists again for the next
    /// caller with the same key: the server refuses the first request's
    /// `tools/list` and lists for the second, whose item carries no error.
    /// A transient refusal lasts one request, not until eviction.
    #[tokio::test]
    async fn a_hit_on_a_connection_whose_listing_failed_lists_again() {
        let (orchestrator, format_registry) = create_test_orchestrator_no_builtin().await;
        let refusing = Arc::new(AtomicBool::new(true));
        let url = serve_local_mcp_server(LocalServer {
            tools: Some(&["echo"]),
            refusing: Some(Arc::clone(&refusing)),
            ..Default::default()
        })
        .await;
        let input = || McpServerInput {
            label: "conf".to_string(),
            url: Some(url.clone()),
            authorization: None,
            headers: HashMap::new(),
            allowed_tools: None,
        };

        let first = connect_mcp_servers(&orchestrator, &format_registry, &[input()]).await;
        assert!(
            list_item_error(&orchestrator, first, "conf").is_some_and(|error| error.is_string()),
            "the first request's listing was refused"
        );

        refusing.store(false, Ordering::SeqCst);
        let second = connect_mcp_servers(&orchestrator, &format_registry, &[input()]).await;
        assert_eq!(
            list_item_error(&orchestrator, second, "conf"),
            None,
            "the pooled connection listed again for the second request"
        );
    }

    /// A caller that hits the pool while the connection's opener is still
    /// listing takes no pending listing for a success: it lists for itself.
    /// Two requests to one refusing server under one key, the second arriving
    /// during the first's slowed listing: both items carry the refusal.
    #[tokio::test]
    async fn a_hit_during_anothers_listing_lists_for_itself() {
        let (orchestrator, format_registry) = create_test_orchestrator_no_builtin().await;
        let url = serve_local_mcp_server(LocalServer {
            list_delay: Duration::from_millis(400),
            ..Default::default()
        })
        .await;
        let input = || McpServerInput {
            label: "conf".to_string(),
            url: Some(url.clone()),
            authorization: None,
            headers: HashMap::new(),
            allowed_tools: None,
        };

        let (first_input, second_input) = ([input()], [input()]);
        let (first, second) = tokio::join!(
            connect_mcp_servers(&orchestrator, &format_registry, &first_input),
            async {
                tokio::time::sleep(Duration::from_millis(150)).await;
                connect_mcp_servers(&orchestrator, &format_registry, &second_input).await
            }
        );
        for (which, bindings) in [("first", first), ("second", second)] {
            assert_eq!(bindings.len(), 1, "the {which} request binds the server");
            assert!(
                list_item_error(&orchestrator, bindings, "conf")
                    .is_some_and(|error| error.is_string()),
                "the {which} request's item carries the refusal"
            );
        }
    }

    /// A listing's record is the listing connection's. With a pool of one,
    /// the first connection is evicted while its listing is held at the
    /// server, a second is pooled under the same key and its listing
    /// refused; the first's late success does not stand for the second: the
    /// next caller under the key lists again and carries the refusal instead
    /// of a hit's silence. The server's signals order the steps, no timer.
    #[tokio::test]
    async fn an_evicted_connections_late_listing_is_not_its_successors() {
        let mut config = McpConfig::default();
        config.pool.max_connections = 1;
        let orchestrator = Arc::new(McpOrchestrator::new(config).await.unwrap());
        let format_registry = FormatRegistry::new();
        let refusing = Arc::new(AtomicBool::new(false));
        let arrived = Arc::new(Notify::new());
        let hold = Arc::new(Hold {
            holding: AtomicBool::new(true),
            release: Notify::new(),
        });
        let url = serve_local_mcp_server(LocalServer {
            tools: Some(&["echo"]),
            refusing: Some(Arc::clone(&refusing)),
            arrived: Some(Arc::clone(&arrived)),
            hold: Some(Arc::clone(&hold)),
            ..Default::default()
        })
        .await;
        let input = |label: &str, token: Option<&str>| McpServerInput {
            label: label.to_string(),
            url: Some(url.clone()),
            authorization: token.map(str::to_string),
            headers: HashMap::new(),
            allowed_tools: None,
        };
        let item_error = |bindings: Vec<McpServerBinding>| {
            let label = bindings.first().expect("one binding").label.clone();
            list_item_error(&orchestrator, bindings, &label)
        };

        let (first_input, evicting_input, second_input) = (
            [input("first", None)],
            [input("other", Some("other-token"))],
            [input("second", None)],
        );
        let (first, second) = tokio::join!(
            connect_mcp_servers(&orchestrator, &format_registry, &first_input),
            async {
                // The first connection's listing has arrived and is held:
                // another key's connect evicts the connection, the same
                // key's connect pools a second whose listing the server
                // refuses at once; the held listing is released after that
                // and completes last.
                // Bounded: a first connection that never lists fails here and
                // says so, not at the job's timeout.
                tokio::time::timeout(Duration::from_secs(10), arrived.notified())
                    .await
                    .expect("the first connection's tools/list reaches the server");
                hold.holding.store(false, Ordering::SeqCst);
                refusing.store(true, Ordering::SeqCst);
                connect_mcp_servers(&orchestrator, &format_registry, &evicting_input).await;
                let second =
                    connect_mcp_servers(&orchestrator, &format_registry, &second_input).await;
                hold.release.notify_one();
                second
            }
        );
        assert_eq!(item_error(first), None, "the first connection listed");
        assert!(
            item_error(second).is_some_and(|error| error.is_string()),
            "the second connection's listing was refused"
        );

        // The server still refuses: the pooled connection's listing failed,
        // so this caller lists again and reads the refusal. The evicted
        // connection's late success recorded under the key would make this
        // a hit that reads nothing.
        let next_input = [input("next", None)];
        let next = connect_mcp_servers(&orchestrator, &format_registry, &next_input).await;
        assert!(
            item_error(next).is_some_and(|error| error.is_string()),
            "the evicted connection's listing is not the pooled connection's record"
        );
    }

    /// One server URL named under two labels is two bindings, each with its
    /// own allowlist (the pooled client shared), and two `mcp_list_tools`
    /// items in the response, each listing its label's own tools.
    #[tokio::test]
    async fn a_server_url_bound_under_two_labels_is_two_bindings_and_two_list_items() {
        let (orchestrator, format_registry) = create_test_orchestrator_no_builtin().await;
        let url = serve_local_mcp_server(LocalServer {
            tools: Some(&["echo", "big"]),
            ..Default::default()
        })
        .await;
        let inputs = [
            McpServerInput {
                label: "conf".to_string(),
                url: Some(url.clone()),
                authorization: None,
                headers: HashMap::new(),
                allowed_tools: None,
            },
            McpServerInput {
                label: "conf-b".to_string(),
                url: Some(url),
                authorization: None,
                headers: HashMap::new(),
                allowed_tools: Some(vec!["echo".to_string()]),
            },
        ];

        let bindings = connect_mcp_servers(&orchestrator, &format_registry, &inputs).await;
        assert_eq!(
            bindings
                .iter()
                .map(|b| b.label.as_str())
                .collect::<Vec<_>>(),
            ["conf", "conf-b"],
            "one binding per label"
        );
        assert_eq!(bindings[0].server_key, bindings[1].server_key);
        assert_eq!(
            bindings[1].allowed_tools.as_deref(),
            Some(&["echo".to_string()][..])
        );

        let session = McpToolSession::new(&orchestrator, bindings, "test-request");
        let mut output = Vec::new();
        openai_bridge::inject_client_visible_mcp_output_items(
            &session,
            &mut output,
            Vec::new(),
            &HashSet::new(),
        );
        let lists: Vec<(&str, Vec<&str>)> = output
            .iter()
            .filter_map(|item| match item {
                ResponseOutputItem::McpListTools {
                    server_label,
                    tools,
                    ..
                } => {
                    let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
                    names.sort_unstable();
                    Some((server_label.as_str(), names))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            lists,
            [("conf", vec!["big", "echo"]), ("conf-b", vec!["echo"])]
        );
    }

    /// A server registered after start-up, the way the gateway registers every one of its
    /// servers, answers for its builtin type; a second server that claims the type does not
    /// take it; after a shutdown no connected server answers.
    #[tokio::test]
    async fn a_builtin_server_registered_after_start_up_is_found_by_its_type() {
        let builtin = |name: &str, url: String| McpServerConfig {
            name: name.to_string(),
            transport: McpTransport::Streamable {
                url,
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: None,
            builtin_type: Some(BuiltinToolType::WebSearchPreview),
            builtin_tool_name: Some("echo".to_string()),
            internal: false,
        };
        let orchestrator = McpOrchestrator::new(McpConfig::default()).await.unwrap();
        let found = || orchestrator.find_builtin_server(BuiltinToolType::WebSearchPreview);
        assert_eq!(found(), None);

        for name in ["first", "second"] {
            let url = serve_local_mcp_server(LocalServer::default()).await;
            orchestrator
                .connect_static_server(&builtin(name, url))
                .await
                .expect("the server connects");
        }
        assert_eq!(found(), Some(("first".to_string(), "echo".to_string())));

        orchestrator.shutdown().await;
        assert_eq!(found(), None);
    }

    /// A refresh publishes its tools before its success. The server refuses
    /// the first listing, then says its tools changed. A reader on its own
    /// thread takes the tools at the first moment it finds no listing error,
    /// as a request that binds the server does, and finds them all. Two
    /// thousand tools make the write-back long enough to be read half done.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_refreshed_listing_is_published_before_its_success_is() {
        let names: &'static [&'static str] = (0..2000)
            .map(|n| &*format!("tool-{n}").leak())
            .collect::<Vec<_>>()
            .leak();
        let refusing = Arc::new(AtomicBool::new(true));
        let changed = Arc::new(Notify::new());
        let url = serve_local_mcp_server(LocalServer {
            tools: Some(names),
            refusing: Some(Arc::clone(&refusing)),
            changed: Some(Arc::clone(&changed)),
            ..Default::default()
        })
        .await;
        let orchestrator = Arc::new(McpOrchestrator::new(McpConfig::default()).await.unwrap());
        orchestrator
            .connect_static_server(&McpServerConfig {
                name: "conf".to_string(),
                transport: McpTransport::Streamable {
                    url,
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: false,
            })
            .await
            .expect("the server connects");
        assert!(
            orchestrator.list_error_for_server("conf").is_some(),
            "the first listing was refused"
        );

        let (spinning, spins) = oneshot::channel();
        let reader = tokio::task::spawn_blocking({
            let orchestrator = Arc::clone(&orchestrator);
            move || {
                let _ = spinning.send(());
                let end = std::time::Instant::now() + Duration::from_secs(10);
                while std::time::Instant::now() < end {
                    if orchestrator.list_error_for_server("conf").is_none() {
                        let conf = ["conf".to_string()];
                        return Some(orchestrator.list_tools_for_servers(&conf).len());
                    }
                }
                None
            }
        });
        spins.await.expect("the reader runs");
        refusing.store(false, Ordering::SeqCst);
        changed.notify_one();
        assert_eq!(
            reader.await.expect("the reader ends"),
            Some(names.len()),
            "what a reader finds once the listing error is gone (None: no refresh in ten seconds)"
        );
    }

    /// A refresh drops what the server no longer lists and keeps the rest.
    /// The server lists two tools, each with a configured alias, and the
    /// kept one has an alias registered through the orchestrator too. The
    /// second listing leaves the first tool out and describes the other
    /// anew. The dropped tool and its alias are gone; the kept tool and its
    /// aliases stay, each with the new description.
    #[tokio::test]
    async fn a_refresh_drops_the_tool_the_server_no_longer_lists() {
        let relisted = Arc::new(AtomicBool::new(false));
        let changed = Arc::new(Notify::new());
        let arrived = Arc::new(Notify::new());
        let url = serve_local_mcp_server(LocalServer {
            tools: Some(&["gone", "kept"]),
            relisted: Some(Arc::clone(&relisted)),
            changed: Some(Arc::clone(&changed)),
            arrived: Some(Arc::clone(&arrived)),
            ..Default::default()
        })
        .await;
        let alias = |name: &str| ToolConfig {
            alias: Some(name.to_string()),
            ..Default::default()
        };
        let orchestrator = Arc::new(McpOrchestrator::new(McpConfig::default()).await.unwrap());
        orchestrator
            .connect_static_server(&McpServerConfig {
                name: "conf".to_string(),
                transport: McpTransport::Streamable {
                    url,
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: Some(HashMap::from([
                    ("gone".to_string(), alias("gone_alias")),
                    ("kept".to_string(), alias("kept_alias")),
                ])),
                builtin_type: None,
                builtin_tool_name: None,
                internal: false,
            })
            .await
            .expect("the server connects");
        // The first load's listing.
        arrived.notified().await;
        orchestrator
            .register_alias("kept_too", "conf", "kept", None)
            .expect("the kept tool is listed");
        let inventory = orchestrator.tool_inventory();
        let described = |name: &str| {
            inventory
                .get_tool_or_alias(name)
                .and_then(|(_, tool)| tool.description.map(|text| text.into_owned()))
        };
        assert_eq!(described("gone_alias").as_deref(), Some("the gone tool"));
        assert_eq!(described("kept_too").as_deref(), Some("the kept tool"));

        relisted.store(true, Ordering::SeqCst);
        // The second refresh's listing reaches the server only after the
        // first's write-back.
        for _ in 0..2 {
            changed.notify_one();
            tokio::time::timeout(Duration::from_secs(10), arrived.notified())
                .await
                .expect("a refresh's listing reaches the server");
        }

        assert!(inventory.get_entry("conf", "gone").is_none());
        assert!(inventory.resolve_alias("gone_alias").is_none());
        assert_eq!(described("gone_alias"), None);
        for name in ["kept", "kept_alias", "kept_too"] {
            assert_eq!(
                described(name).as_deref(),
                Some("the kept tool, relisted"),
                "the description under {name} after the refresh"
            );
        }
    }

    /// A refresh never shows a server with fewer tools than both listings
    /// hold: the refreshed listing goes in before what is gone goes out. A
    /// reader on its own thread lists in a loop through one whole refresh of
    /// two thousand tools. The second refresh's listing reaches the server
    /// only after the first's write-back, and ends the read.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_refresh_keeps_the_tools_both_listings_hold() {
        let names: &'static [&'static str] = (0..2000)
            .map(|n| &*format!("tool-{n}").leak())
            .collect::<Vec<_>>()
            .leak();
        let changed = Arc::new(Notify::new());
        let arrived = Arc::new(Notify::new());
        let url = serve_local_mcp_server(LocalServer {
            tools: Some(names),
            changed: Some(Arc::clone(&changed)),
            arrived: Some(Arc::clone(&arrived)),
            ..Default::default()
        })
        .await;
        let orchestrator = Arc::new(McpOrchestrator::new(McpConfig::default()).await.unwrap());
        orchestrator
            .connect_static_server(&McpServerConfig {
                name: "conf".to_string(),
                transport: McpTransport::Streamable {
                    url,
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: false,
            })
            .await
            .expect("the server connects");
        // The first load's listing.
        arrived.notified().await;

        let (spinning, spins) = oneshot::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let reader = tokio::task::spawn_blocking({
            let (orchestrator, stop) = (Arc::clone(&orchestrator), Arc::clone(&stop));
            move || {
                let conf = ["conf".to_string()];
                let mut fewest = orchestrator.list_tools_for_servers(&conf).len();
                let _ = spinning.send(());
                // Its own end: were a wait below to time out, a reader still
                // looping would hold the runtime open and hang the test.
                let end = std::time::Instant::now() + Duration::from_secs(30);
                while !stop.load(Ordering::SeqCst) && std::time::Instant::now() < end {
                    fewest = fewest.min(orchestrator.list_tools_for_servers(&conf).len());
                }
                fewest
            }
        });
        spins.await.expect("the reader runs");
        for _ in 0..2 {
            changed.notify_one();
            tokio::time::timeout(Duration::from_secs(10), arrived.notified())
                .await
                .expect("a refresh's listing reaches the server");
        }
        stop.store(true, Ordering::SeqCst);
        assert_eq!(
            reader.await.expect("the reader ends"),
            names.len(),
            "the fewest tools a reader found through a refresh"
        );
    }

    /// A refreshed tool is never in the inventory without the argument
    /// mapping its configuration sets: the listing's tools take their
    /// mappings before they go in. Two thousand tools are configured with a
    /// mapping each, and a reader on its own thread reads the first of them
    /// in a loop through one whole refresh. The second refresh's listing
    /// reaches the server only after the first's write-back, and ends the
    /// read.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_refreshed_tool_keeps_its_argument_mapping() {
        let names: &'static [&'static str] = (0..2000)
            .map(|n| &*format!("tool-{n}").leak())
            .collect::<Vec<_>>()
            .leak();
        let changed = Arc::new(Notify::new());
        let arrived = Arc::new(Notify::new());
        let url = serve_local_mcp_server(LocalServer {
            tools: Some(names),
            changed: Some(Arc::clone(&changed)),
            arrived: Some(Arc::clone(&arrived)),
            ..Default::default()
        })
        .await;
        let with_mapping = || ToolConfig {
            arg_mapping: Some(smg_mcp::ArgMappingConfig {
                defaults: HashMap::from([("limit".to_string(), json!(5))]),
                ..Default::default()
            }),
            ..Default::default()
        };
        let orchestrator = Arc::new(McpOrchestrator::new(McpConfig::default()).await.unwrap());
        orchestrator
            .connect_static_server(&McpServerConfig {
                name: "conf".to_string(),
                transport: McpTransport::Streamable {
                    url,
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: Some(
                    names
                        .iter()
                        .map(|name| (name.to_string(), with_mapping()))
                        .collect(),
                ),
                builtin_type: None,
                builtin_tool_name: None,
                internal: false,
            })
            .await
            .expect("the server connects");
        // The first load's listing.
        arrived.notified().await;

        let (spinning, spins) = oneshot::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let reader = tokio::task::spawn_blocking({
            let (orchestrator, stop) = (Arc::clone(&orchestrator), Arc::clone(&stop));
            move || {
                let inventory = orchestrator.tool_inventory();
                let mapped = || {
                    inventory
                        .get_entry("conf", names[0])
                        .is_some_and(|entry| entry.arg_mapping.is_some())
                };
                let mut always = mapped();
                let _ = spinning.send(());
                // Its own end: were a wait below to time out, a reader still
                // looping would hold the runtime open and hang the test.
                let end = std::time::Instant::now() + Duration::from_secs(30);
                while !stop.load(Ordering::SeqCst) && std::time::Instant::now() < end {
                    always &= mapped();
                }
                always
            }
        });
        spins.await.expect("the reader runs");
        for _ in 0..2 {
            changed.notify_one();
            tokio::time::timeout(Duration::from_secs(10), arrived.notified())
                .await
                .expect("a refresh's listing reaches the server");
        }
        stop.store(true, Ordering::SeqCst);
        assert!(
            reader.await.expect("the reader ends"),
            "a reader found the tool with its argument mapping at every read through a refresh"
        );
    }

    /// Waits for one of the local server's signals, ten seconds at most;
    /// `step` names what did not happen.
    async fn reached(signal: &Notify, step: &str) {
        tokio::time::timeout(Duration::from_secs(10), signal.notified())
            .await
            .expect(step);
    }

    /// An orchestrator holding one static server, `conf`, registered after
    /// start-up and connected to the local server at `url`.
    async fn orchestrator_connected_to(url: String) -> Arc<McpOrchestrator> {
        let orchestrator = Arc::new(McpOrchestrator::new(McpConfig::default()).await.unwrap());
        orchestrator
            .connect_static_server(&McpServerConfig {
                name: "conf".to_string(),
                transport: McpTransport::Streamable {
                    url,
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: false,
            })
            .await
            .expect("the server connects");
        orchestrator
    }

    /// One call of `tool` on the static server `conf`, through a session
    /// that binds it; whether the call failed. A local server that drops its
    /// calls fails the call's transport: the call is answered after the
    /// recovery that follows, as a failure (the tool is not replayed).
    async fn call_on_conf(orchestrator: &McpOrchestrator, tool: &str) -> bool {
        let conf = McpServerBinding {
            label: "conf".to_string(),
            server_key: "conf".to_string(),
            allowed_tools: None,
            list_error: None,
            pool_key: None,
        };
        McpToolSession::new(orchestrator, vec![conf], "test-request")
            .execute_tool(ToolExecutionInput {
                call_id: "call-1".to_string(),
                tool_name: tool.to_string(),
                arguments: json!({}),
            })
            .await
            .is_error
    }

    /// A recovery keeps the server's tools until its new listing is in. The
    /// local server drops a tool call, so the call's recovery connects
    /// again, and holds the new connection's listing. A request that binds
    /// the server at that moment reads no listing error, because the dropped
    /// connection's entry stands until the new one replaces it, and finds
    /// the tools.
    #[tokio::test]
    async fn a_recovery_keeps_the_tools_until_its_listing_is_in() {
        let arrived = Arc::new(Notify::new());
        let hold = Arc::new(Hold::default());
        let url = serve_local_mcp_server(LocalServer {
            tools: Some(&["echo"]),
            arrived: Some(Arc::clone(&arrived)),
            hold: Some(Arc::clone(&hold)),
            drops_calls: true,
            ..Default::default()
        })
        .await;
        let orchestrator = orchestrator_connected_to(url).await;
        // The first load's listing.
        arrived.notified().await;
        hold.holding.store(true, Ordering::SeqCst);

        let conf = ["conf".to_string()];
        let (dropped, read) = tokio::join!(call_on_conf(&orchestrator, "echo"), async {
            reached(&arrived, "the dropped call's recovery lists again").await;
            let read = (
                orchestrator.list_error_for_server("conf"),
                orchestrator.list_tools_for_servers(&conf).len(),
            );
            hold.holding.store(false, Ordering::SeqCst);
            hold.release.notify_one();
            read
        });
        assert!(dropped, "the call the server dropped fails");
        assert_eq!(
            read,
            (None, 1),
            "the listing error and the tools a request reads while the recovery lists"
        );
        assert_eq!(
            orchestrator.list_tools_for_servers(&conf).len(),
            1,
            "the tools after the recovery"
        );
    }

    /// A refresh that listed on a dropped connection does not stand over the
    /// recovery's listing. Each listing of the local server carries the
    /// number of its moment. A refresh takes its tools at moment 1 on the
    /// first connection and waits in its prompts listing. A dropped call's
    /// recovery connects again, takes its tools at moment 2 and waits in its
    /// own prompts listing. The refresh is released and writes back, as the
    /// entry still holds its connection; then the recovery. The server keeps
    /// the tools of moment 2.
    #[tokio::test]
    async fn a_dropped_connections_refresh_does_not_stand_over_the_recovery() {
        let moment = Arc::new(AtomicUsize::new(0));
        let changed = Arc::new(Notify::new());
        let gate = Arc::new(Gate::default());
        let url = serve_local_mcp_server(LocalServer {
            tools: Some(&["echo"]),
            changed: Some(Arc::clone(&changed)),
            numbered: Some(Arc::clone(&moment)),
            prompts_gate: Some(Arc::clone(&gate)),
            drops_calls: true,
            ..Default::default()
        })
        .await;
        let orchestrator = orchestrator_connected_to(url).await;
        let conf = ["conf".to_string()];
        let names = || -> Vec<String> {
            orchestrator
                .list_tools_for_servers(&conf)
                .iter()
                .map(|tool| tool.tool_name().to_string())
                .collect()
        };
        assert_eq!(names(), ["echo-0"], "the first load's listing");

        gate.hold.holding.store(true, Ordering::SeqCst);
        moment.store(1, Ordering::SeqCst);
        changed.notify_one();
        reached(&gate.arrived, "the refresh reaches its prompts listing").await;

        moment.store(2, Ordering::SeqCst);
        let (dropped, ()) = tokio::join!(call_on_conf(&orchestrator, "echo-0"), async {
            reached(&gate.arrived, "the recovery reaches its prompts listing").await;
            // The refresh first: its write-back lands while the recovery waits.
            gate.hold.release.notify_one();
            tokio::time::timeout(Duration::from_secs(10), async {
                while names() != ["echo-1"] {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("the refresh writes back while the entry holds its connection");
            gate.hold.release.notify_one();
        });
        assert!(dropped, "the call the server dropped fails");
        assert_eq!(
            names(),
            ["echo-2"],
            "the recovery's listing stands, not the dropped connection's"
        );
    }

    /// An orchestrator holding one configured server, `conf-static`, a
    /// local MCP server with one tool; and a registry.
    async fn orchestrator_holding_conf_static() -> (Arc<McpOrchestrator>, FormatRegistry) {
        let url = serve_local_mcp_server(LocalServer {
            tools: Some(&["echo"]),
            ..Default::default()
        })
        .await;
        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "conf-static".to_string(),
                transport: McpTransport::Streamable {
                    url,
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: false,
            }],
            pool: Default::default(),
            proxy: None,
            warmup: Vec::new(),
            inventory: Default::default(),
            policy: Default::default(),
        };
        let orchestrator = Arc::new(McpOrchestrator::new(config).await.unwrap());
        let format_registry = FormatRegistry::new();
        (orchestrator, format_registry)
    }

    /// A tool with no `server_url` names a configured server by its label
    /// and binds it; a label no configured server has binds nothing (the
    /// request then fails on the missing connection) instead of a server
    /// key that resolves to no tools.
    #[tokio::test]
    async fn a_label_without_a_server_url_binds_only_a_configured_server() {
        let (orchestrator, format_registry) = orchestrator_holding_conf_static().await;
        let label_only = |label: &str| McpServerInput {
            label: label.to_string(),
            url: None,
            authorization: None,
            headers: HashMap::new(),
            allowed_tools: None,
        };

        let bindings = connect_mcp_servers(
            &orchestrator,
            &format_registry,
            &[label_only("conf-static")],
        )
        .await;
        assert_eq!(
            bindings.len(),
            1,
            "the configured server binds: {bindings:?}"
        );
        assert_eq!(bindings[0].server_key, "conf-static");
        let session = McpToolSession::new(&orchestrator, bindings, "test-request");
        assert_eq!(session.list_tools_for_server("conf-static").len(), 1);

        let bindings =
            connect_mcp_servers(&orchestrator, &format_registry, &[label_only("conf")]).await;
        assert!(
            bindings.is_empty(),
            "no configured server is named conf: {bindings:?}"
        );
    }

    /// The Responses routes' rule: a request whose only `mcp` tool names no
    /// held server is refused on the missing connection (424
    /// `connect_mcp_server_failed`) before anything goes upstream; next to a
    /// server that binds, the unknown label is dropped and the request
    /// routes with the one binding.
    #[tokio::test]
    async fn an_mcp_tool_that_binds_nothing_is_refused_before_the_forward() {
        let (orchestrator, format_registry) = orchestrator_holding_conf_static().await;
        let mcp = |label: &str| -> ResponseTool {
            serde_json::from_value(json!({"type": "mcp", "server_label": label}))
                .expect("an mcp tool with no server_url")
        };

        let refused =
            ensure_request_mcp_routing(&orchestrator, &format_registry, &[mcp("conf")]).await;
        let response = refused.expect_err("a lone unknown label is refused");
        assert_eq!(response.status(), StatusCode::FAILED_DEPENDENCY);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the refusal's body");
        let body: Value = serde_json::from_slice(&body).expect("a JSON error");
        assert_eq!(body["error"]["code"], "connect_mcp_server_failed");
        assert_eq!(
            body["error"]["message"],
            "Failed to connect to MCP servers. Check server_url and authorization, or that a \
             server_label without server_url names a configured server."
        );

        let routed = ensure_request_mcp_routing(
            &orchestrator,
            &format_registry,
            &[mcp("conf"), mcp("conf-static")],
        )
        .await
        .expect("the configured server routes the request");
        let bindings = routed.expect("one binding");
        assert_eq!(bindings.len(), 1, "{bindings:?}");
        assert_eq!(bindings[0].server_key, "conf-static");
    }

    /// A server URL of a scheme the connector does not take is refused with
    /// a line that names it masked: credentials in it stay out of the log.
    #[tracing_test::traced_test]
    #[tokio::test]
    async fn a_refused_server_url_is_named_masked() {
        let (orchestrator, format_registry) = create_test_orchestrator_no_builtin().await;
        let inputs = [McpServerInput {
            label: "tools".to_string(),
            url: Some("wss://user:secret@127.0.0.1:1/sse".to_string()),
            authorization: None,
            headers: HashMap::new(),
            allowed_tools: None,
        }];

        let bindings = connect_mcp_servers(&orchestrator, &format_registry, &inputs).await;

        assert!(!logs_contain("secret"));
        assert!(bindings.is_empty());
        assert!(logs_contain(
            "Ignoring MCP server_url with unsupported scheme: wss://***@127.0.0.1:1/sse"
        ));
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_with_configured_server() {
        let (orchestrator, format_registry) = create_test_orchestrator_with_builtin().await;

        let tools = vec![ResponseTool::WebSearchPreview(
            WebSearchPreviewTool::default(),
        )];

        let routing = collect_builtin_routing(&orchestrator, &format_registry, Some(&tools));

        assert_eq!(routing.len(), 1);
        assert_eq!(routing[0].builtin_type, BuiltinToolType::WebSearchPreview);
        assert_eq!(routing[0].server_name, "search-server");
        assert_eq!(routing[0].tool_name, "web_search");
        assert_eq!(routing[0].response_format, ResponseFormat::WebSearchCall);
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_no_configured_server() {
        let (orchestrator, format_registry) = create_test_orchestrator_no_builtin().await;

        let tools = vec![ResponseTool::WebSearchPreview(
            WebSearchPreviewTool::default(),
        )];

        let routing = collect_builtin_routing(&orchestrator, &format_registry, Some(&tools));

        // No routing because no server configured for this built-in type
        assert!(routing.is_empty());
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_ignores_mcp_tools() {
        let (orchestrator, format_registry) = create_test_orchestrator_with_builtin().await;

        let tools = vec![ResponseTool::Mcp(McpTool {
            server_url: Some("http://example.com/mcp".to_string()),
            authorization: None,
            headers: None,
            server_label: "mcp".to_string(),
            server_description: None,
            require_approval: None,
            allowed_tools: None,
            connector_id: None,
            defer_loading: None,
        })];

        let routing = collect_builtin_routing(&orchestrator, &format_registry, Some(&tools));

        // MCP tools are not built-in types, should be empty
        assert!(routing.is_empty());
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_ignores_function_tools() {
        let (orchestrator, format_registry) = create_test_orchestrator_with_builtin().await;

        let tools = vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: "dummy".to_string(),
                description: None,
                parameters: Some(json!({})),
                strict: None,
                extra: Default::default(),
            },
        })];

        let routing = collect_builtin_routing(&orchestrator, &format_registry, Some(&tools));

        // Function tools are not built-in types, should be empty
        assert!(routing.is_empty());
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_none_tools() {
        let (orchestrator, format_registry) = create_test_orchestrator_no_builtin().await;

        let routing = collect_builtin_routing(&orchestrator, &format_registry, None);

        assert!(routing.is_empty());
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_multiple_builtin_tools() {
        // Create orchestrator with both web search and code interpreter
        let mut web_search_tools = HashMap::new();
        web_search_tools.insert(
            "web_search".to_string(),
            ToolConfig {
                response_format: Some(ResponseFormatConfig::WebSearchCall),
                ..Default::default()
            },
        );

        let mut code_interp_tools = HashMap::new();
        code_interp_tools.insert(
            "run_code".to_string(),
            ToolConfig {
                response_format: Some(ResponseFormatConfig::CodeInterpreterCall),
                ..Default::default()
            },
        );

        let config = McpConfig {
            servers: vec![
                McpServerConfig {
                    name: "search-server".to_string(),
                    transport: McpTransport::Streamable {
                        url: "http://localhost:9999/search".to_string(),
                        token: None,
                        headers: HashMap::new(),
                    },
                    proxy: None,
                    required: false,
                    tools: Some(web_search_tools),
                    builtin_type: Some(BuiltinToolType::WebSearchPreview),
                    builtin_tool_name: Some("web_search".to_string()),
                    internal: false,
                },
                McpServerConfig {
                    name: "code-server".to_string(),
                    transport: McpTransport::Streamable {
                        url: "http://localhost:9998/code".to_string(),
                        token: None,
                        headers: HashMap::new(),
                    },
                    proxy: None,
                    required: false,
                    tools: Some(code_interp_tools),
                    builtin_type: Some(BuiltinToolType::CodeInterpreter),
                    builtin_tool_name: Some("run_code".to_string()),
                    internal: false,
                },
            ],
            pool: Default::default(),
            proxy: None,
            warmup: Vec::new(),
            inventory: Default::default(),
            policy: Default::default(),
        };

        let format_registry = FormatRegistry::new();
        for server in &config.servers {
            format_registry.populate_from_server_config(server);
        }
        let orchestrator = Arc::new(McpOrchestrator::new(config).await.unwrap());

        let tools = vec![
            ResponseTool::WebSearchPreview(WebSearchPreviewTool::default()),
            ResponseTool::CodeInterpreter(CodeInterpreterTool::default()),
        ];

        let routing = collect_builtin_routing(&orchestrator, &format_registry, Some(&tools));

        assert_eq!(routing.len(), 2);

        // Find web search routing
        let web_routing = routing
            .iter()
            .find(|r| r.builtin_type == BuiltinToolType::WebSearchPreview)
            .expect("Should have web search routing");
        assert_eq!(web_routing.server_name, "search-server");
        assert_eq!(web_routing.tool_name, "web_search");
        assert_eq!(web_routing.response_format, ResponseFormat::WebSearchCall);

        // Find code interpreter routing
        let code_routing = routing
            .iter()
            .find(|r| r.builtin_type == BuiltinToolType::CodeInterpreter)
            .expect("Should have code interpreter routing");
        assert_eq!(code_routing.server_name, "code-server");
        assert_eq!(code_routing.tool_name, "run_code");
        assert_eq!(
            code_routing.response_format,
            ResponseFormat::CodeInterpreterCall
        );
    }

    #[tokio::test]
    async fn test_collect_builtin_routing_image_generation() {
        // Image generation is wired through the same hosted-tool MCP plumbing
        // as web_search / code_interpreter / file_search. This test proves
        // BuiltinToolType::ImageGeneration → ResponseFormat::ImageGenerationCall
        // so per-router wiring can rely on the infrastructure.
        let mut image_gen_tools = HashMap::new();
        image_gen_tools.insert(
            "generate_image".to_string(),
            ToolConfig {
                response_format: Some(ResponseFormatConfig::ImageGenerationCall),
                ..Default::default()
            },
        );

        let config = McpConfig {
            servers: vec![McpServerConfig {
                name: "image-server".to_string(),
                transport: McpTransport::Streamable {
                    url: "http://localhost:9997/image".to_string(),
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: Some(image_gen_tools),
                builtin_type: Some(BuiltinToolType::ImageGeneration),
                builtin_tool_name: Some("generate_image".to_string()),
                internal: false,
            }],
            pool: Default::default(),
            proxy: None,
            warmup: Vec::new(),
            inventory: Default::default(),
            policy: Default::default(),
        };

        let format_registry = FormatRegistry::new();
        for server in &config.servers {
            format_registry.populate_from_server_config(server);
        }
        let orchestrator = Arc::new(McpOrchestrator::new(config).await.unwrap());

        let tools = vec![ResponseTool::ImageGeneration(ImageGenerationTool::default())];

        let routing = collect_builtin_routing(&orchestrator, &format_registry, Some(&tools));

        assert_eq!(routing.len(), 1);
        assert_eq!(routing[0].builtin_type, BuiltinToolType::ImageGeneration);
        assert_eq!(routing[0].server_name, "image-server");
        assert_eq!(routing[0].tool_name, "generate_image");
        assert_eq!(
            routing[0].response_format,
            ResponseFormat::ImageGenerationCall
        );
    }

    // =========================================================================
    // ensure_request_mcp_client tests
    // =========================================================================

    #[tokio::test]
    async fn test_ensure_request_mcp_client_with_builtin_routing() {
        // Create orchestrator with a built-in server configured
        let (orchestrator, format_registry) = create_test_orchestrator_with_builtin().await;

        // Request has web_search_preview tool (no server_url, not MCP type)
        let tools = vec![ResponseTool::WebSearchPreview(
            WebSearchPreviewTool::default(),
        )];

        let result = ensure_request_mcp_client(&orchestrator, &format_registry, &tools).await;

        // Should return Some because built-in routing is configured
        assert!(result.is_some());

        let mcp_servers = result.unwrap();
        assert_eq!(mcp_servers.len(), 1);

        // The server key should be the static server name
        assert_eq!(mcp_servers[0].label, "search-server");
        assert_eq!(mcp_servers[0].server_key, "search-server");
    }

    #[tokio::test]
    async fn test_ensure_request_mcp_client_no_builtin_routing() {
        // Create orchestrator WITHOUT built-in server configured
        let (orchestrator, format_registry) = create_test_orchestrator_no_builtin().await;

        // Request has web_search_preview tool
        let tools = vec![ResponseTool::WebSearchPreview(
            WebSearchPreviewTool::default(),
        )];

        let result = ensure_request_mcp_client(&orchestrator, &format_registry, &tools).await;

        // Should return None because no MCP or built-in routing is available
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_ensure_request_mcp_client_function_tools_only() {
        let (orchestrator, format_registry) = create_test_orchestrator_with_builtin().await;

        // Request has only function tools (no MCP, no built-in)
        let tools = vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: "dummy".to_string(),
                description: None,
                parameters: Some(json!({})),
                strict: None,
                extra: Default::default(),
            },
        })];

        let result = ensure_request_mcp_client(&orchestrator, &format_registry, &tools).await;

        // Should return None - function tools don't need MCP processing
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_ensure_request_mcp_client_mixed_tools() {
        // Create orchestrator with built-in server
        let (orchestrator, format_registry) = create_test_orchestrator_with_builtin().await;

        // Request has mixed tools: function + web_search_preview
        let tools = vec![
            ResponseTool::Function(FunctionTool {
                function: Function {
                    name: "dummy".to_string(),
                    description: None,
                    parameters: Some(json!({})),
                    strict: None,
                    extra: Default::default(),
                },
            }),
            ResponseTool::WebSearchPreview(WebSearchPreviewTool::default()),
        ];

        let result = ensure_request_mcp_client(&orchestrator, &format_registry, &tools).await;

        // Should return Some because web_search_preview has built-in routing
        assert!(result.is_some());

        let mcp_servers = result.unwrap();
        assert_eq!(mcp_servers.len(), 1);
        assert_eq!(mcp_servers[0].label, "search-server");
    }

    // ---- T11 projection ---------------------------------------------------

    use openai_protocol::responses::McpToolFilter;

    #[test]
    fn test_project_allowed_tools_none() {
        assert_eq!(project_allowed_tools(None), None);
    }

    #[test]
    fn test_project_allowed_tools_list_variant() {
        let value = McpAllowedTools::List(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(
            project_allowed_tools(Some(&value)),
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn test_project_allowed_tools_filter_with_tool_names() {
        let value = McpAllowedTools::Filter(McpToolFilter {
            read_only: None,
            tool_names: Some(vec!["x".to_string()]),
        });
        assert_eq!(
            project_allowed_tools(Some(&value)),
            Some(vec!["x".to_string()])
        );
    }

    /// `Filter { read_only: Some(true) }` with no `tool_names` must project
    /// to `Some(vec![])` (fail-closed) so downstream does not expose the full
    /// tool surface when the caller explicitly asked to restrict. See
    /// [`project_allowed_tools`] rationale.
    #[test]
    fn test_project_allowed_tools_filter_read_only_only_is_fail_closed() {
        let value = McpAllowedTools::Filter(McpToolFilter {
            read_only: Some(true),
            tool_names: None,
        });
        assert_eq!(project_allowed_tools(Some(&value)), Some(Vec::new()));
    }

    /// `Filter { read_only: None, tool_names: None }` — both sub-fields
    /// absent — is indistinguishable from an absent allowlist and projects
    /// to `None` so the downstream retain path treats it as unconstrained.
    #[test]
    fn test_project_allowed_tools_filter_empty_is_unconstrained() {
        let value = McpAllowedTools::Filter(McpToolFilter::default());
        assert_eq!(project_allowed_tools(Some(&value)), None);
    }

    /// `Filter { read_only: Some(_), tool_names: Some(_) }` — both set — must
    /// still fail-closed. Any `read_only` restriction disables the normal
    /// name-list projection because `readOnlyHint`-based filtering is
    /// unimplemented; honoring only the `tool_names` half would broaden
    /// exposure past caller intent (e.g. `tool_names: ["mutating_tool"]` with
    /// `read_only: true` must not expose `mutating_tool`).
    #[test]
    fn test_project_allowed_tools_filter_read_only_plus_names_is_fail_closed() {
        let value = McpAllowedTools::Filter(McpToolFilter {
            read_only: Some(true),
            tool_names: Some(vec!["mutating_tool".to_string()]),
        });
        assert_eq!(project_allowed_tools(Some(&value)), Some(Vec::new()));
    }

    /// When the synthesized function tool exposes `user` as a parameter,
    /// the model emits `{"user": null}` even though no value is supplied.
    /// That null must not block request-level forwarding — the helper
    /// treats absent and null as equivalent for injection purposes.
    #[test]
    fn inject_user_hosted_format_overwrites_model_supplied_null() {
        let mut args = json!({"prompt": "a cat", "user": null});
        inject_user_into_hosted_args(
            &mut args,
            ResponseFormat::ImageGenerationCall,
            Some("user-123"),
        );
        assert_eq!(args.get("user"), Some(&json!("user-123")));
    }

    #[test]
    fn inject_user_hosted_format_inserts_user_into_clean_args() {
        let mut args = json!({"prompt": "a cat"});
        inject_user_into_hosted_args(
            &mut args,
            ResponseFormat::ImageGenerationCall,
            Some("user-123"),
        );
        assert_eq!(args.get("user"), Some(&json!("user-123")));
        assert_eq!(args.get("prompt"), Some(&json!("a cat")));
    }

    #[test]
    fn inject_user_hosted_format_preserves_existing_user_key() {
        // Model-supplied `user` wins over the request-level identifier; the
        // helper must not clobber it.
        let mut args = json!({"prompt": "a cat", "user": "model-supplied"});
        inject_user_into_hosted_args(
            &mut args,
            ResponseFormat::ImageGenerationCall,
            Some("request-level"),
        );
        assert_eq!(args.get("user"), Some(&json!("model-supplied")));
    }

    #[test]
    fn inject_user_passthrough_format_is_noop() {
        // Plain MCP function tools have caller-defined schemas; injecting
        // `user` could surprise tools that don't expect that key.
        let mut args = json!({"q": "weather"});
        inject_user_into_hosted_args(&mut args, ResponseFormat::Passthrough, Some("user-123"));
        assert!(
            !args.as_object().unwrap().contains_key("user"),
            "passthrough format must not receive an injected user key"
        );
    }

    #[test]
    fn inject_user_empty_or_missing_user_is_noop() {
        let mut args_none = json!({"prompt": "x"});
        inject_user_into_hosted_args(&mut args_none, ResponseFormat::WebSearchCall, None);
        assert!(!args_none.as_object().unwrap().contains_key("user"));

        let mut args_empty = json!({"prompt": "x"});
        inject_user_into_hosted_args(&mut args_empty, ResponseFormat::WebSearchCall, Some(""));
        assert!(!args_empty.as_object().unwrap().contains_key("user"));
    }

    #[test]
    fn inject_user_non_object_args_is_noop() {
        let mut args = json!("not-an-object");
        inject_user_into_hosted_args(&mut args, ResponseFormat::FileSearchCall, Some("u1"));
        assert_eq!(args, json!("not-an-object"));
    }

    #[test]
    fn inject_user_covers_every_hosted_format() {
        // All four hosted formats should accept the injection identically.
        for format in [
            ResponseFormat::ImageGenerationCall,
            ResponseFormat::WebSearchCall,
            ResponseFormat::CodeInterpreterCall,
            ResponseFormat::FileSearchCall,
        ] {
            let mut args = json!({});
            inject_user_into_hosted_args(&mut args, format, Some("user-xyz"));
            assert_eq!(
                args.get("user"),
                Some(&json!("user-xyz")),
                "expected user injection for format {format:?}"
            );
        }
    }
    #[test]
    fn collect_namespace_function_names_preserves_user_routing() {
        let request: ResponsesRequest = serde_json::from_value(json!({
            "model": "test-model", "input": "Check the weather",
            "tools": [
                {"type": "namespace", "name": "weather", "description": "Weather tools", "tools": [
                    {"type": "function", "name": "lookup", "parameters": {}}
                ]},
                {"type": "namespace", "name": "travel", "description": "Travel tools", "tools": [
                    {"type": "function", "name": "lookup", "parameters": {}}
                ]}
            ]
        }))
        .unwrap();
        let names = collect_user_function_names(&request);
        assert_eq!(
            names,
            HashSet::from(["weather.lookup".into(), "travel.lookup".into()])
        );
    }
}
