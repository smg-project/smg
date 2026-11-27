use serde_json::json;

use super::*;
use crate::{
    core::config::{test_support::sse_server, Tool as McpTool},
    inventory::ArgMapping,
};

#[test]
fn test_session_creation_keeps_servers() {
    let orchestrator = McpOrchestrator::new_test();
    let mcp_servers = vec![
        McpServerBinding {
            label: "label1".to_string(),
            server_key: "key1".to_string(),
            allowed_tools: None,
            list_error: None,
            pool_key: None,
        },
        McpServerBinding {
            label: "label2".to_string(),
            server_key: "key2".to_string(),
            allowed_tools: None,
            list_error: None,
            pool_key: None,
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
        list_error: None,
        pool_key: None,
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

/// Puts the test tool `name` of `server` into `orchestrator`'s inventory.
fn insert_test_tool(orchestrator: &McpOrchestrator, server: &str, name: &str) {
    orchestrator
        .tool_inventory()
        .insert_entry(ToolEntry::from_server_tool(server, create_test_tool(name)));
}

#[test]
fn test_has_exposed_tool_with_inventory() {
    let orchestrator = McpOrchestrator::new_test();

    insert_test_tool(&orchestrator, "server1", "test_tool");

    let mcp_servers = vec![McpServerBinding {
        label: "label1".to_string(),
        server_key: "server1".to_string(),
        allowed_tools: None,
        list_error: None,
        pool_key: None,
    }];
    let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

    assert!(session.has_exposed_tool("test_tool"));
    assert_eq!(session.mcp_tools().len(), 1);
}

#[tokio::test]
async fn test_execute_tool_result_preserves_pending_approval() {
    let orchestrator = McpOrchestrator::new_test();

    insert_test_tool(&orchestrator, "server1", "test_tool");

    let mut session = McpToolSession::new(
        &orchestrator,
        vec![McpServerBinding {
            label: "label1".to_string(),
            server_key: "server1".to_string(),
            allowed_tools: None,
            list_error: None,
            pool_key: None,
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
fn a_labels_approval_reaches_a_tool_attributed_to_another_label_of_its_server() {
    fn mode(session: &McpToolSession<'_>, tool: &str) -> Option<ApprovalMode> {
        session.exposed_name_map.get(tool).map(|b| b.approval_mode)
    }
    let orchestrator = McpOrchestrator::new_test();
    insert_test_tool(&orchestrator, "server1", "echo");
    insert_test_tool(&orchestrator, "server1", "big");
    let open = McpServerBinding {
        label: "conf".to_string(),
        server_key: "server1".to_string(),
        allowed_tools: None,
        list_error: None,
        pool_key: None,
    };
    let narrow = McpServerBinding {
        label: "conf-b".to_string(),
        allowed_tools: Some(vec!["echo".to_string()]),
        ..open.clone()
    };

    // `echo` is attributed to `conf`, the first label that admits it.
    let mut session = McpToolSession::new(
        &orchestrator,
        vec![open.clone(), narrow.clone()],
        "test-request",
    );
    session.set_approval_mode("conf-b", None, ApprovalMode::Interactive);
    assert_eq!(mode(&session, "echo"), Some(ApprovalMode::Interactive));
    assert_eq!(
        mode(&session, "big"),
        Some(ApprovalMode::PolicyOnly),
        "`conf-b` does not admit `big`"
    );

    // The other order attributes `echo` to `conf-b`.
    let mut session = McpToolSession::new(&orchestrator, vec![narrow, open], "test-request");
    session.set_approval_mode("conf", None, ApprovalMode::Interactive);
    assert_eq!(mode(&session, "echo"), Some(ApprovalMode::Interactive));
    assert_eq!(mode(&session, "big"), Some(ApprovalMode::Interactive));

    session.set_approval_mode("conf-b", None, ApprovalMode::PolicyOnly);
    assert_eq!(
        mode(&session, "echo"),
        Some(ApprovalMode::Interactive),
        "the stricter mode stays"
    );
}

#[test]
fn a_labels_approval_reaches_an_alias_and_a_padded_name_its_allowlist_admits() {
    let orchestrator = McpOrchestrator::new_test();
    insert_test_tool(&orchestrator, "server1", "brave_web_search");
    insert_test_tool(&orchestrator, "server1", "echo");
    orchestrator
        .register_alias("web_search", "server1", "brave_web_search", None)
        .expect("alias registration should succeed");
    // The alias is listed by its target's name, `echo` with a space.
    let listed = vec!["brave_web_search".to_string(), " echo".to_string()];
    let mut session = McpToolSession::new(
        &orchestrator,
        vec![McpServerBinding {
            label: "brave".to_string(),
            server_key: "server1".to_string(),
            allowed_tools: Some(listed.clone()),
            list_error: None,
            pool_key: None,
        }],
        "test-request",
    );
    session.set_approval_mode("brave", Some(&listed), ApprovalMode::Interactive);
    for tool in ["web_search", "echo"] {
        assert_eq!(
            session.exposed_name_map.get(tool).map(|b| b.approval_mode),
            Some(ApprovalMode::Interactive),
            "{tool}"
        );
    }
}

/// A session calls what it listed: an alias registered again with
/// another target is not this session's tool.
#[tokio::test]
async fn an_alias_registered_again_with_another_target_is_not_called() {
    let orchestrator = McpOrchestrator::new_test();
    insert_test_tool(&orchestrator, "server1", "search");
    insert_test_tool(&orchestrator, "server2", "search");
    orchestrator
        .register_alias("find", "server1", "search", None)
        .unwrap();
    let binding = |label: &str, server_key: &str| McpServerBinding {
        label: label.to_string(),
        server_key: server_key.to_string(),
        allowed_tools: None,
        list_error: None,
        pool_key: None,
    };
    let session = McpToolSession::new(
        &orchestrator,
        vec![binding("zero", "server0"), binding("one", "server1")],
        "test-request",
    );
    assert!(session.has_exposed_tool("find"));
    let call = || {
        session.execute_tool(ToolExecutionInput {
            call_id: "call".to_string(),
            tool_name: "find".to_string(),
            arguments: json!({}),
        })
    };

    // Registered again with the target it had, the alias is still called.
    orchestrator
        .register_alias("find", "server1", "search", None)
        .unwrap();
    assert_eq!(
        call().await.error_message.as_deref(),
        Some("Tool call failed: Server not found: server1")
    );

    orchestrator
        .register_alias("find", "server2", "search", None)
        .unwrap();
    let refused = call().await;
    assert_eq!(
        refused.error_message.as_deref(),
        Some("Tool 'find' is not in this session's exposed tool map")
    );
    // The refusal reads under the label the alias was listed with.
    assert_eq!(refused.server_label, "one");
}

#[test]
fn test_resolve_label_with_inventory() {
    let orchestrator = McpOrchestrator::new_test();

    insert_test_tool(&orchestrator, "server1", "test_tool");

    let mcp_servers = vec![McpServerBinding {
        label: "my_server".to_string(),
        server_key: "server1".to_string(),
        allowed_tools: None,
        list_error: None,
        pool_key: None,
    }];
    let session = McpToolSession::new(&orchestrator, mcp_servers, "test-request");

    let label = session.resolve_tool_server_label("test_tool");
    assert_eq!(label, "my_server");
}

#[test]
fn test_exposed_names_are_unique_for_tool_name_collisions() {
    let orchestrator = McpOrchestrator::new_test();

    insert_test_tool(&orchestrator, "server1", "shared_tool");
    insert_test_tool(&orchestrator, "server2", "shared_tool");

    let session = McpToolSession::new(
        &orchestrator,
        vec![
            McpServerBinding {
                label: "alpha".to_string(),
                server_key: "server1".to_string(),
                allowed_tools: None,
                list_error: None,
                pool_key: None,
            },
            McpServerBinding {
                label: "beta".to_string(),
                server_key: "server2".to_string(),
                allowed_tools: None,
                list_error: None,
                pool_key: None,
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

    insert_test_tool(&orchestrator, "s1", "foo");
    insert_test_tool(&orchestrator, "s2", "foo_1");
    insert_test_tool(&orchestrator, "s3", "foo");

    let session = McpToolSession::new(
        &orchestrator,
        vec![
            McpServerBinding {
                label: "a".to_string(),
                server_key: "s1".to_string(),
                allowed_tools: None,
                list_error: None,
                pool_key: None,
            },
            McpServerBinding {
                label: "b".to_string(),
                server_key: "s2".to_string(),
                allowed_tools: None,
                list_error: None,
                pool_key: None,
            },
            McpServerBinding {
                label: "c".to_string(),
                server_key: "s3".to_string(),
                allowed_tools: None,
                list_error: None,
                pool_key: None,
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
            list_error: None,
            pool_key: None,
        },
        McpServerBinding {
            label: "regular".to_string(),
            server_key: "regular-server".to_string(),
            allowed_tools: None,
            list_error: None,
            pool_key: None,
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

    insert_test_tool(&orchestrator, "internal-server", "brave_web_search");
    insert_test_tool(
        &orchestrator,
        "internal-server",
        "internal_non_builtin_tool",
    );

    let session = McpToolSession::new(
        &orchestrator,
        vec![McpServerBinding {
            label: "internal-label".to_string(),
            server_key: "internal-server".to_string(),
            allowed_tools: None,
            list_error: None,
            pool_key: None,
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

    insert_test_tool(&orchestrator, "image-server", "generate_image");

    let session = McpToolSession::new(
        &orchestrator,
        vec![McpServerBinding {
            label: "image".to_string(),
            server_key: "image-server".to_string(),
            allowed_tools: None,
            list_error: None,
            pool_key: None,
        }],
        "test-request",
    );

    assert!(session.is_builtin_tool("generate_image"));
    assert!(session.is_builtin_server_label("image"));
}

#[test]
fn test_allowed_tools_filters_inventory_and_list_tools() {
    let orchestrator = McpOrchestrator::new_test();

    insert_test_tool(&orchestrator, "server1", "brave_web_search");
    insert_test_tool(&orchestrator, "server1", "brave_local_search");

    let session = McpToolSession::new(
        &orchestrator,
        vec![McpServerBinding {
            label: "mock".to_string(),
            server_key: "server1".to_string(),
            allowed_tools: Some(vec!["brave_web_search".to_string()]),
            list_error: None,
            pool_key: None,
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

/// A server bound under two labels lists each label's own tools, the
/// model sees the union, and a tool is attributed to the first label
/// whose allowlist admits it.
#[test]
fn a_server_bound_under_two_labels_lists_per_label_and_attributes_the_first() {
    fn names(entries: &[ToolEntry]) -> Vec<&str> {
        let mut names: Vec<&str> = entries.iter().map(ToolEntry::tool_name).collect();
        names.sort_unstable();
        names
    }
    let orchestrator = McpOrchestrator::new_test();
    insert_test_tool(&orchestrator, "server1", "echo");
    insert_test_tool(&orchestrator, "server1", "big");
    let open = McpServerBinding {
        label: "conf".to_string(),
        server_key: "server1".to_string(),
        allowed_tools: None,
        list_error: None,
        pool_key: None,
    };
    let narrow = McpServerBinding {
        label: "conf-b".to_string(),
        server_key: "server1".to_string(),
        allowed_tools: Some(vec!["echo".to_string()]),
        list_error: None,
        pool_key: None,
    };

    let session = McpToolSession::new(
        &orchestrator,
        vec![open.clone(), narrow.clone()],
        "test-request",
    );
    assert_eq!(session.mcp_servers().len(), 2, "one binding per label");
    assert_eq!(
        names(session.mcp_tools()),
        ["big", "echo"],
        "the model sees the union"
    );
    assert_eq!(
        names(&session.list_tools_for_binding(&open)),
        ["big", "echo"]
    );
    assert_eq!(names(&session.list_tools_for_binding(&narrow)), ["echo"]);
    assert_eq!(
        names(&session.list_tools_for_label("conf-b", "server1")),
        ["echo"]
    );
    assert_eq!(
        names(&session.list_tools_for_label("unbound", "server1")),
        ["big", "echo"]
    );
    assert_eq!(session.resolve_tool_server_label("echo"), "conf");
    assert_eq!(session.resolve_tool_server_label("big"), "conf");

    // Two allowlists: the model still sees both tools, each attributed
    // to the first label admitting it.
    let session = McpToolSession::new(
        &orchestrator,
        vec![
            narrow,
            McpServerBinding {
                allowed_tools: Some(vec!["big".to_string()]),
                ..open
            },
        ],
        "test-request",
    );
    assert_eq!(names(session.mcp_tools()), ["big", "echo"]);
    assert_eq!(session.resolve_tool_server_label("echo"), "conf-b");
    assert_eq!(session.resolve_tool_server_label("big"), "conf");
}

#[test]
fn test_alias_tools_replace_target_tool_in_session_inventory() {
    let orchestrator = McpOrchestrator::new_test();

    insert_test_tool(&orchestrator, "server1", "brave_web_search");

    orchestrator
        .register_alias(
            "web_search",
            "server1",
            "brave_web_search",
            Some(ArgMapping::new().with_override("enable_brave", serde_json::json!(false))),
        )
        .expect("alias registration should succeed");

    let session = McpToolSession::new(
        &orchestrator,
        vec![McpServerBinding {
            label: "brave".to_string(),
            server_key: "server1".to_string(),
            allowed_tools: None,
            list_error: None,
            pool_key: None,
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

    insert_test_tool(&orchestrator, "server1", "brave_web_search");

    orchestrator
        .register_alias("web_search", "server1", "brave_web_search", None)
        .expect("alias registration should succeed");

    let session = McpToolSession::new(
        &orchestrator,
        vec![McpServerBinding {
            label: "brave".to_string(),
            server_key: "server1".to_string(),
            allowed_tools: Some(vec!["web_search".to_string()]),
            list_error: None,
            pool_key: None,
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
    insert_test_tool(&orchestrator, "internal-server", "internal_search");

    let session = McpToolSession::new(
        &orchestrator,
        vec![McpServerBinding {
            label: "internal-label".to_string(),
            server_key: "internal-server".to_string(),
            allowed_tools: None,
            list_error: None,
            pool_key: None,
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
    insert_test_tool(&orchestrator, "internal-server", "internal_search");

    orchestrator
        .register_alias("alias_search", "internal-server", "internal_search", None)
        .expect("alias registration should succeed");

    let session = McpToolSession::new(
        &orchestrator,
        vec![McpServerBinding {
            label: "internal-label".to_string(),
            server_key: "internal-server".to_string(),
            allowed_tools: None,
            list_error: None,
            pool_key: None,
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
                list_error: None,
                pool_key: None,
            },
            McpServerBinding {
                label: "shared-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
                list_error: None,
                pool_key: None,
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
    insert_test_tool(&orchestrator, "server1", "brave_web_search");
    insert_test_tool(&orchestrator, "server1", "brave_local_search");

    // server2 has two tools
    insert_test_tool(&orchestrator, "server2", "deepwiki_search");
    insert_test_tool(&orchestrator, "server2", "deepwiki_read");

    let session = McpToolSession::new(
        &orchestrator,
        vec![
            McpServerBinding {
                label: "brave".to_string(),
                server_key: "server1".to_string(),
                allowed_tools: Some(vec!["brave_web_search".to_string()]),
                list_error: None,
                pool_key: None,
            },
            McpServerBinding {
                label: "deepwiki".to_string(),
                server_key: "server2".to_string(),
                allowed_tools: None,
                list_error: None,
                pool_key: None,
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

/// An alias listed through the alias index's own key, its target's server
/// not among the bindings, is still told from the builtin bindings: the
/// session looks them up for it.
#[test]
fn an_alias_listed_without_its_targets_server_keeps_its_builtin_routing() {
    let orchestrator = create_builtin_orchestrator();
    insert_test_tool(&orchestrator, "brave-builtin", "brave_web_search");
    orchestrator
        .register_alias("search", "brave-builtin", "brave_web_search", None)
        .expect("alias registration should succeed");

    let session = McpToolSession::new(
        &orchestrator,
        vec![McpServerBinding {
            label: "aliases".to_string(),
            server_key: crate::inventory::ALIAS_SERVER_KEY.to_string(),
            allowed_tools: None,
            list_error: None,
            pool_key: None,
        }],
        "test-request",
    );

    assert!(session.is_builtin_tool("search"));
}
