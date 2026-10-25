//! What the crate's tests build an MCP session from: an orchestrator over
//! one internal SSE server, the binding a request names that server by, and
//! the session over the two.

use smg_mcp::{
    BuiltinToolType, McpConfig, McpOrchestrator, McpServerBinding, McpServerConfig, McpToolSession,
    McpTransport, Tool, ToolEntry,
};

/// The server `internal_orchestrator` configures and `internal_binding` names.
pub(crate) const INTERNAL_SERVER: &str = "internal-server";

/// An orchestrator over one internal server, [`INTERNAL_SERVER`], reached over
/// SSE; `builtin` routes the server as that built-in tool, under the given
/// tool name. The server is optional, so nothing needs to listen: the tests
/// fill the tool inventory by hand.
pub(crate) async fn internal_orchestrator(
    builtin: Option<(BuiltinToolType, &str)>,
) -> McpOrchestrator {
    let (builtin_type, builtin_tool_name) = match builtin {
        Some((kind, tool_name)) => (Some(kind), Some(tool_name.to_string())),
        None => (None, None),
    };
    McpOrchestrator::new(McpConfig {
        servers: vec![McpServerConfig {
            name: INTERNAL_SERVER.to_string(),
            transport: McpTransport::Sse {
                url: "http://localhost:3000/sse".to_string(),
                token: None,
                headers: Default::default(),
            },
            proxy: None,
            required: false,
            tools: None,
            builtin_type,
            builtin_tool_name,
            internal: true,
        }],
        ..Default::default()
    })
    .await
    .expect("orchestrator")
}

/// The binding a request names [`INTERNAL_SERVER`] by: label `internal-label`,
/// no tool allow-list.
pub(crate) fn internal_binding() -> McpServerBinding {
    McpServerBinding {
        label: "internal-label".to_string(),
        server_key: INTERNAL_SERVER.to_string(),
        allowed_tools: None,
    }
}

/// An orchestrator from [`internal_orchestrator`] whose inventory holds
/// `tools`, each one a tool of [`INTERNAL_SERVER`].
pub(crate) async fn internal_orchestrator_serving(
    builtin: Option<(BuiltinToolType, &str)>,
    tools: Vec<Tool>,
) -> McpOrchestrator {
    let orchestrator = internal_orchestrator(builtin).await;
    for tool in tools {
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(INTERNAL_SERVER, tool));
    }
    orchestrator
}

/// The session a request that names [`INTERNAL_SERVER`] through
/// [`internal_binding`] opens over `orchestrator`.
pub(crate) fn internal_session(orchestrator: &McpOrchestrator) -> McpToolSession<'_> {
    McpToolSession::new(orchestrator, vec![internal_binding()], "test-request")
}
