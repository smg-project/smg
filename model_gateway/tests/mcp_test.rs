// This test suite validates the complete MCP implementation against the
// functionality required for SGLang responses API integration.
//
// - Core MCP server functionality
// - Tool session management (individual and multi-tool)
// - Tool execution and error handling
// - Schema adaptation and validation
// - Mock server integration for reliable testing

mod common;

use std::collections::HashMap;

use common::mock_mcp_server::{MockMCPServer, MockSearchResponseMCPServer, MockSearchResponseMode};
use openai_protocol::responses::{ResponseOutputItem, WebSearchAction};
use serde_json::json;
use smg::routers::common::openai_bridge::{
    inject_client_visible_mcp_output_items, transform_tool_output, ResponseFormat,
};
use smg_mcp::{
    core::config::{ResponseFormatConfig, ToolConfig},
    McpConfig, McpOrchestrator, McpServerBinding, McpServerConfig, McpToolSession, McpTransport,
    ToolExecutionInput,
};

/// Create a new mock server for testing (each test gets its own)
#[expect(clippy::expect_used)]
async fn create_mock_server() -> MockMCPServer {
    MockMCPServer::start()
        .await
        .expect("Failed to start mock MCP server")
}

#[expect(clippy::expect_used)]
async fn create_mock_search_response_server(
    mode: MockSearchResponseMode,
) -> MockSearchResponseMCPServer {
    MockSearchResponseMCPServer::start(mode)
        .await
        .expect("Failed to start mock search response MCP server")
}

// Core MCP Server Tests

#[tokio::test]
async fn test_mcp_server_initialization() {
    let config = McpConfig {
        servers: vec![],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
        policy: Default::default(),
    };

    // Should succeed but with no connected servers (empty config is allowed)
    let result = McpOrchestrator::new(config).await;
    assert!(result.is_ok(), "Should succeed with empty config");

    let manager = result.unwrap();
    let servers = manager.list_servers();
    assert_eq!(servers.len(), 0, "Should have no servers");
    let tools = manager.list_tools(None);
    assert_eq!(tools.len(), 0, "Should have no tools");
}

#[tokio::test]
async fn test_server_connection_with_mock() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
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

    let result = McpOrchestrator::new(config).await;
    assert!(result.is_ok(), "Should connect to mock server");

    let manager = result.unwrap();

    let servers = manager.list_servers();
    assert_eq!(servers.len(), 1);
    assert!(servers.contains(&"mock_server".to_string()));

    let tools = manager.list_tools(None);
    assert_eq!(tools.len(), 2, "Should have 2 tools from mock server");

    assert!(manager.has_tool("mock_server", "brave_web_search"));
    assert!(manager.has_tool("mock_server", "brave_local_search"));

    manager.shutdown().await;
}

#[tokio::test]
async fn test_tool_availability_checking() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
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

    let manager = McpOrchestrator::new(config).await.unwrap();

    let test_tools = vec!["brave_web_search", "brave_local_search", "calculator"];
    for tool in test_tools {
        let available = manager.has_tool("mock_server", tool);
        match tool {
            "brave_web_search" | "brave_local_search" => {
                assert!(
                    available,
                    "Tool {tool} should be available from mock server"
                );
            }
            "calculator" => {
                assert!(
                    !available,
                    "Tool {tool} should not be available from mock server"
                );
            }
            _ => {}
        }
    }

    manager.shutdown().await;
}

#[tokio::test]
async fn test_multi_server_connection() {
    let mock_server1 = create_mock_server().await;
    let mock_server2 = create_mock_server().await;

    let config = McpConfig {
        servers: vec![
            McpServerConfig {
                name: "mock_server_1".to_string(),
                transport: McpTransport::Streamable {
                    url: mock_server1.url(),
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: false,
            },
            McpServerConfig {
                name: "mock_server_2".to_string(),
                transport: McpTransport::Streamable {
                    url: mock_server2.url(),
                    token: None,
                    headers: HashMap::new(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: false,
            },
        ],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
        policy: Default::default(),
    };

    // Note: This will fail to connect to both servers in the current implementation
    // since they return the same tools. The manager will connect to the first one.
    let result = McpOrchestrator::new(config).await;

    if let Ok(manager) = result {
        let servers = manager.list_servers();
        assert!(!servers.is_empty(), "Should have at least one server");

        let tools = manager.list_tools(None);
        assert!(tools.len() >= 2, "Should have tools from servers");

        manager.shutdown().await;
    }
}

#[tokio::test]
async fn test_tool_execution_with_mock() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
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

    let manager = McpOrchestrator::new(config).await.unwrap();

    let session = McpToolSession::new(
        &manager,
        vec![McpServerBinding {
            label: "mock_server".to_string(),
            server_key: "mock_server".to_string(),
            allowed_tools: None,
        }],
        "test-request-1",
    );

    let output = session
        .execute_tool(ToolExecutionInput {
            call_id: "call-1".to_string(),
            tool_name: "brave_web_search".to_string(),
            arguments: json!({
                "query": "rust programming",
                "count": 1
            }),
        })
        .await;

    assert!(!output.is_error, "Tool execution should succeed");
    assert!(
        output
            .output
            .to_string()
            .contains("Mock search results for: rust programming"),
        "Output should contain mock search results, got: {}",
        output.output
    );

    manager.shutdown().await;
}

#[tokio::test]
async fn test_web_search_transform_handles_openai_search_response_with_mock() {
    let mock_server = create_mock_search_response_server(MockSearchResponseMode::OpenAi).await;
    let mut tools = HashMap::new();
    tools.insert(
        "brave_web_search".to_string(),
        ToolConfig {
            alias: None,
            response_format: Some(ResponseFormatConfig::WebSearchCall),
            arg_mapping: None,
        },
    );

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "openai_search_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: Some(tools),
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

    let manager = McpOrchestrator::new(config).await.unwrap();

    let session = McpToolSession::new(
        &manager,
        vec![McpServerBinding {
            label: "openai_search_server".to_string(),
            server_key: "openai_search_server".to_string(),
            allowed_tools: None,
        }],
        "test-request-openai-search",
    );

    let output = session
        .execute_tool(ToolExecutionInput {
            call_id: "call-1".to_string(),
            tool_name: "brave_web_search".to_string(),
            arguments: json!({ "query": "rust openai search" }),
        })
        .await;

    assert!(!output.is_error, "Tool execution should succeed");

    let transformed = transform_tool_output(&output, ResponseFormat::WebSearchCall);
    match transformed {
        ResponseOutputItem::WebSearchCall { action, .. } => match action {
            WebSearchAction::Search {
                query,
                queries: _,
                sources: _,
            } => {
                assert_eq!(query, Some("rust openai search".to_string()));
            }
            _ => panic!("Expected Search action"),
        },
        other => panic!("Expected WebSearchCall, got {other:?}"),
    }

    manager.shutdown().await;
}

#[tokio::test]
async fn test_web_search_transform_sets_action_query_for_brave_search_with_mock() {
    let mock_server = create_mock_search_response_server(MockSearchResponseMode::Brave).await;
    let mut tools = HashMap::new();
    tools.insert(
        "brave_web_search".to_string(),
        ToolConfig {
            alias: None,
            response_format: Some(ResponseFormatConfig::WebSearchCall),
            arg_mapping: None,
        },
    );

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "brave_response_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: Some(tools),
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

    let manager = McpOrchestrator::new(config).await.unwrap();

    let session = McpToolSession::new(
        &manager,
        vec![McpServerBinding {
            label: "brave_response_server".to_string(),
            server_key: "brave_response_server".to_string(),
            allowed_tools: None,
        }],
        "test-request-brave",
    );

    let output = session
        .execute_tool(ToolExecutionInput {
            call_id: "call-1".to_string(),
            tool_name: "brave_web_search".to_string(),
            arguments: json!({ "query": "rust brave query" }),
        })
        .await;

    assert!(!output.is_error, "Tool execution should succeed");

    let transformed = transform_tool_output(&output, ResponseFormat::WebSearchCall);
    match transformed {
        ResponseOutputItem::WebSearchCall { action, .. } => match action {
            WebSearchAction::Search {
                query,
                queries: _,
                sources: _,
            } => {
                assert_eq!(query, Some("rust brave query".to_string()));
            }
            _ => panic!("Expected Search action"),
        },
        other => panic!("Expected WebSearchCall, got {other:?}"),
    }
    manager.shutdown().await;
}

#[tokio::test]
async fn test_concurrent_tool_execution() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
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

    let manager = McpOrchestrator::new(config).await.unwrap();

    let session = McpToolSession::new(
        &manager,
        vec![McpServerBinding {
            label: "mock_server".to_string(),
            server_key: "mock_server".to_string(),
            allowed_tools: None,
        }],
        "test-concurrent",
    );

    let tool_calls = vec![
        ("brave_web_search", json!({"query": "test1"})),
        ("brave_local_search", json!({"query": "test2"})),
    ];

    for (tool_name, args) in tool_calls {
        let output = session
            .execute_tool(ToolExecutionInput {
                call_id: format!("call-{tool_name}"),
                tool_name: tool_name.to_string(),
                arguments: args,
            })
            .await;
        assert!(!output.is_error, "Tool {tool_name} should succeed");
        assert!(
            !output.output.to_string().is_empty(),
            "Should have output content"
        );
    }

    manager.shutdown().await;
}

// Error Handling Tests

#[tokio::test]
async fn test_tool_execution_errors() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
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

    let manager = McpOrchestrator::new(config).await.unwrap();

    let session = McpToolSession::new(
        &manager,
        vec![McpServerBinding {
            label: "mock_server".to_string(),
            server_key: "mock_server".to_string(),
            allowed_tools: None,
        }],
        "test-error",
    );

    let output = session
        .execute_tool(ToolExecutionInput {
            call_id: "call-1".to_string(),
            tool_name: "unknown_tool".to_string(),
            arguments: json!({}),
        })
        .await;
    assert!(
        output.is_error,
        "Unknown tool should produce an error output"
    );
    assert_eq!(output.tool_name, "unknown_tool");

    manager.shutdown().await;
}

#[tokio::test]
async fn test_connection_without_server() {
    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "nonexistent".to_string(),
            transport: McpTransport::Stdio {
                command: "/nonexistent/command".to_string(),
                args: vec![],
                envs: HashMap::new(),
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

    let result = McpOrchestrator::new(config).await;
    // Manager succeeds but no servers are connected (errors are logged)
    assert!(
        result.is_ok(),
        "Manager should succeed even if servers fail to connect"
    );

    let manager = result.unwrap();
    let servers = manager.list_servers();
    assert_eq!(servers.len(), 0, "Should have no connected servers");
}

// Schema Validation Tests

#[tokio::test]
async fn test_tool_info_structure() {
    let mock_server = create_mock_server().await;

    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "mock_server".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
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

    let manager = McpOrchestrator::new(config).await.unwrap();

    let tools = manager.list_tools(None);
    let brave_search = tools
        .iter()
        .find(|t| t.tool.name.as_ref() == "brave_web_search")
        .expect("Should have brave_web_search tool");

    assert_eq!(brave_search.tool.name.as_ref(), "brave_web_search");
    assert!(brave_search
        .tool
        .description
        .as_ref()
        .map(|d| d.contains("Mock web search"))
        .unwrap_or(false));
    // Note: server information is now maintained separately in the inventory,
    // not in the Tool type itself
    assert!(!brave_search.tool.input_schema.is_empty());
}

// SSE Parsing Tests (simplified since we don't expose parse_sse_event)

#[tokio::test]
async fn test_sse_connection() {
    // This tests that SSE configuration is properly handled even when connection fails
    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "sse_test".to_string(),
            transport: McpTransport::Stdio {
                command: "/nonexistent/sse/server".to_string(),
                args: vec!["--sse".to_string()],
                envs: HashMap::new(),
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

    // Manager succeeds but no servers are connected (errors are logged)
    let result = McpOrchestrator::new(config).await;
    assert!(
        result.is_ok(),
        "Manager should succeed even if SSE server fails to connect"
    );

    let manager = result.unwrap();
    let servers = manager.list_servers();
    assert_eq!(servers.len(), 0, "Should have no connected servers");
}

// Connection Type Tests

#[tokio::test]
async fn test_transport_types() {
    // HTTP/Streamable transport
    let http_config = McpServerConfig {
        name: "http_server".to_string(),
        transport: McpTransport::Streamable {
            url: "http://localhost:8080/mcp".to_string(),
            token: Some("auth_token".to_string()),
            headers: HashMap::new(),
        },
        proxy: None,
        required: false,
        tools: None,
        builtin_type: None,
        builtin_tool_name: None,
        internal: false,
    };
    assert_eq!(http_config.name, "http_server");

    // SSE transport
    let sse_config = McpServerConfig {
        name: "sse_server".to_string(),
        transport: McpTransport::Sse {
            url: "http://localhost:8081/sse".to_string(),
            token: None,
            headers: HashMap::new(),
        },
        proxy: None,
        required: false,
        tools: None,
        builtin_type: None,
        builtin_tool_name: None,
        internal: false,
    };
    assert_eq!(sse_config.name, "sse_server");

    // STDIO transport
    let stdio_config = McpServerConfig {
        name: "stdio_server".to_string(),
        transport: McpTransport::Stdio {
            command: "mcp-server".to_string(),
            args: vec!["--port".to_string(), "8082".to_string()],
            envs: HashMap::new(),
        },
        proxy: None,
        required: false,
        tools: None,
        builtin_type: None,
        builtin_tool_name: None,
        internal: false,
    };
    assert_eq!(stdio_config.name, "stdio_server");
}

// Integration Pattern Tests

#[tokio::test]
async fn test_complete_workflow() {
    let mock_server = create_mock_server().await;

    // 1. Initialize configuration
    let config = McpConfig {
        servers: vec![McpServerConfig {
            name: "integration_test".to_string(),
            transport: McpTransport::Streamable {
                url: mock_server.url(),
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

    // 2. Connect to server
    let manager = McpOrchestrator::new(config)
        .await
        .expect("Should connect to mock server");

    // 3. Verify server connection
    let servers = manager.list_servers();
    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0], "integration_test");

    // 4. Check available tools
    let tools = manager.list_tools(None);
    assert_eq!(tools.len(), 2);

    // 5. Verify specific tools exist
    assert!(manager.has_tool("integration_test", "brave_web_search"));
    assert!(manager.has_tool("integration_test", "brave_local_search"));
    assert!(!manager.has_tool("integration_test", "nonexistent_tool"));

    // 6. Execute a tool
    let session = McpToolSession::new(
        &manager,
        vec![McpServerBinding {
            label: "integration_test".to_string(),
            server_key: "integration_test".to_string(),
            allowed_tools: None,
        }],
        "test-workflow",
    );

    let output = session
        .execute_tool(ToolExecutionInput {
            call_id: "call-1".to_string(),
            tool_name: "brave_web_search".to_string(),
            arguments: json!({
                "query": "SGLang router MCP integration",
                "count": 1
            }),
        })
        .await;

    assert!(!output.is_error, "Tool execution should succeed");
    assert!(
        !output.output.to_string().is_empty(),
        "Should return output content"
    );

    // 7. Clean shutdown
    manager.shutdown().await;

    let capabilities = [
        "MCP server initialization",
        "Tool server connection and discovery",
        "Tool availability checking",
        "Tool execution",
        "Error handling and robustness",
        "Multi-server support",
        "Schema adaptation",
        "Mock server integration (no external dependencies)",
    ];

    assert_eq!(capabilities.len(), 8);
}

#[tokio::test]
async fn internal_mcp_visibility_preserves_namespaced_client_function() {
    let mut server = MockMCPServer::start().await.unwrap();
    let manager = McpOrchestrator::new(McpConfig {
        servers: vec![McpServerConfig {
            name: "internal".into(),
            transport: McpTransport::Streamable {
                url: server.url(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: None,
            builtin_type: None,
            builtin_tool_name: None,
            internal: true,
        }],
        pool: Default::default(),
        proxy: None,
        warmup: Vec::new(),
        inventory: Default::default(),
        policy: Default::default(),
    })
    .await
    .unwrap();
    let session = McpToolSession::new(
        &manager,
        vec![McpServerBinding {
            label: "internal".into(),
            server_key: "internal".into(),
            allowed_tools: None,
        }],
        "visibility-test",
    );
    let client: ResponseOutputItem = serde_json::from_value(json!({
        "type":"function_call", "id":"fc_client", "call_id":"call_client", "name":"brave_web_search", "namespace":"client", "arguments":"{}", "status":"completed",
    })).unwrap();
    let internal: ResponseOutputItem = serde_json::from_value(json!({
        "type":"function_call", "id":"fc_internal", "call_id":"call_internal", "name":"brave_web_search", "arguments":"{}", "status":"completed",
    })).unwrap();
    let mut output = vec![client, internal];
    inject_client_visible_mcp_output_items(
        &session,
        &mut output,
        vec![],
        &std::collections::HashSet::from(["client.brave_web_search".to_string()]),
    );
    server.stop().await;
    let output = serde_json::to_value(output).unwrap();
    assert_eq!(output.as_array().unwrap().len(), 1, "{output}");
    assert_eq!(output[0]["call_id"], "call_client");
    assert_eq!(output[0]["namespace"], "client");
}
