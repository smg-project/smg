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
    assert!(
        McpOrchestrator::remaining(Some(ahead)).is_some_and(|left| left > Duration::from_secs(50))
    );
}

#[tokio::test]
async fn test_orchestrator_graceful_shutdown_flow() {
    let orchestrator = McpOrchestrator::new_test();
    let tenant = TenantContext::new("test-tenant");
    let ctx =
        orchestrator.create_request_context("req-shutdown-test", tenant, ApprovalMode::Interactive);

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

    let result = orchestrator.register_alias("web_search", "missing_server", "missing_tool", None);

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
                defaults: HashMap::from([("enable_source".to_string(), serde_json::json!(true))]),
                overrides: HashMap::from([("enable_brave".to_string(), serde_json::json!(false))]),
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
                overrides: HashMap::from([("enable_brave".to_string(), serde_json::json!(false))]),
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
        let pooling = orchestrator
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
        let pooling = orchestrator
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
            tokio::spawn(async move { orchestrator.recover_after_disconnect("listed", None).await })
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
        0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x03, 0xab, 0x56, 0xca, 0xcf, 0x56,
        0xb2, 0x2a, 0x29, 0x2a, 0x4d, 0xad, 0x05, 0x00, 0x90, 0x5f, 0xd4, 0xa7, 0x0b, 0x00, 0x00,
        0x00,
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
