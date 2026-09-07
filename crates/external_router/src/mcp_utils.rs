//! Shared MCP utilities for routers.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use openai_protocol::responses::{McpAllowedTools, ResponseTool, ResponsesRequest};
use serde_json::{json, Value};
use smg_mcp::{BuiltinToolType, McpOrchestrator, McpServerBinding, McpServerConfig, McpTransport};
use tracing::{debug, warn};

use crate::openai_bridge::{
    self, apply_hosted_tool_overrides, extract_hosted_tool_overrides, FormatRegistry,
    ResponseFormat,
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
/// - If `url` is absent, registers the label as a static server reference.
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
                    server_url
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
                Ok(_) => {
                    if !mcp_servers.iter().any(|b| b.server_key == server_key) {
                        mcp_servers.push(McpServerBinding {
                            label: input.label.clone(),
                            server_key,
                            allowed_tools: input.allowed_tools.clone(),
                        });
                    }
                }
                Err(err) => {
                    warn!("Failed to connect MCP server {}: {}", server_key, err);
                }
            }
        }
        // Case B: Static Server (No URL)
        else if !mcp_servers.iter().any(|b| b.server_key == input.label) {
            mcp_servers.push(McpServerBinding {
                label: input.label.clone(),
                server_key: input.label.clone(),
                allowed_tools: input.allowed_tools.clone(),
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
        .filter_map(|tool| match tool {
            ResponseTool::Function(function_tool) => Some(function_tool.function.name.clone()),
            _ => None,
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
                mcp_servers.push(McpServerBinding {
                    label: server_name.clone(),
                    server_key: server_name,
                    allowed_tools: None,
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
    use std::{collections::HashMap, sync::Arc};

    use openai_protocol::{
        common::Function,
        responses::{
            CodeInterpreterTool, FunctionTool, ImageGenerationTool, McpTool, ResponseTool,
            WebSearchPreviewTool,
        },
    };
    use serde_json::json;
    use smg_mcp::{McpConfig, ResponseFormatConfig, ToolConfig};

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
                parameters: json!({}),
                strict: None,
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
                parameters: json!({}),
                strict: None,
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
                    parameters: json!({}),
                    strict: None,
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
}
