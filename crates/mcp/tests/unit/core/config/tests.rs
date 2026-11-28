use serial_test::serial;

use super::*;
use crate::core::config::test_support::sse_server;

#[test]
fn test_call_timeout_default_override_and_disable() {
    let config: McpConfig = serde_yaml::from_str("servers: []").unwrap();
    assert_eq!(
        config.call_timeout(),
        Some(Duration::from_secs(DEFAULT_CALL_TIMEOUT_SECS))
    );

    let yaml = "servers: []
pool:
  call_timeout: 5";
    let config: McpConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(config.call_timeout(), Some(Duration::from_secs(5)));

    let yaml = "servers: []
pool:
  call_timeout: 0";
    let config: McpConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(config.call_timeout(), None);
}

#[test]
fn test_default_pool_config() {
    let config = McpPoolConfig::default();
    assert_eq!(config.max_connections, 100);
    assert_eq!(config.idle_timeout, 300);
}

#[test]
fn test_default_inventory_config() {
    let config = InventoryConfig::default();
    assert!(config.enable_refresh);
    assert_eq!(config.tool_ttl, 300);
    assert_eq!(config.refresh_interval, 60);
    assert!(config.refresh_on_error);
}

#[test]
#[serial]
fn test_proxy_from_env_empty() {
    // Ensure no proxy env vars are set for this test
    std::env::remove_var("MCP_HTTP_PROXY");
    std::env::remove_var("MCP_HTTPS_PROXY");
    std::env::remove_var("HTTP_PROXY");
    std::env::remove_var("HTTPS_PROXY");

    let proxy = McpProxyConfig::from_env();
    assert!(proxy.is_none(), "Should return None when no env vars set");
}

#[test]
#[serial]
fn test_proxy_from_env_with_vars() {
    std::env::set_var("MCP_HTTP_PROXY", "http://test-proxy:8080");
    std::env::set_var("MCP_NO_PROXY", "localhost,127.0.0.1");

    let proxy = McpProxyConfig::from_env();
    assert!(proxy.is_some(), "Should return Some when env vars set");

    let proxy = proxy.unwrap();
    assert_eq!(proxy.http.as_ref().unwrap(), "http://test-proxy:8080");
    assert_eq!(proxy.no_proxy.as_ref().unwrap(), "localhost,127.0.0.1");

    // Cleanup
    std::env::remove_var("MCP_HTTP_PROXY");
    std::env::remove_var("MCP_NO_PROXY");
}

#[tokio::test]
async fn test_yaml_minimal_config() {
    let yaml = r#"
servers:
  - name: "test-server"
    protocol: sse
    url: "http://localhost:3000/sse"
"#;

    let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");
    assert_eq!(config.servers.len(), 1);
    assert_eq!(config.servers[0].name, "test-server");
    assert!(!config.servers[0].required); // Should default to false
    assert!(config.servers[0].proxy.is_none()); // Should default to None
    assert_eq!(config.pool.max_connections, 100); // Should use default
    assert_eq!(config.inventory.tool_ttl, 300); // Should use default
}

#[tokio::test]
async fn test_yaml_full_config() {
    let yaml = r#"
# Global proxy configuration
proxy:
  http: "http://global-proxy:8080"
  https: "http://global-proxy:8080"
  no_proxy: "localhost,127.0.0.1,*.internal"

# Connection pool settings
pool:
  max_connections: 50
  idle_timeout: 600

# Tool inventory settings
inventory:
  enable_refresh: true
  tool_ttl: 600
  refresh_interval: 120
  refresh_on_error: true

# Static servers
servers:
  - name: "required-server"
    protocol: sse
    url: "https://api.example.com/sse"
    token: "secret-token"
    required: true

  - name: "optional-server"
    protocol: stdio
    command: "mcp-server"
    args: ["--port", "3000"]
    required: false
    proxy:
      http: "http://server-specific-proxy:9090"

# Pre-warm connections
warmup:
  - url: "http://localhost:3000/sse"
    label: "local-dev"
"#;

    let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");

    // Check global proxy
    assert!(config.proxy.is_some());
    let global_proxy = config.proxy.as_ref().unwrap();
    assert_eq!(
        global_proxy.http.as_ref().unwrap(),
        "http://global-proxy:8080"
    );

    // Check pool config
    assert_eq!(config.pool.max_connections, 50);
    assert_eq!(config.pool.idle_timeout, 600);

    // Check inventory config
    assert_eq!(config.inventory.tool_ttl, 600);
    assert_eq!(config.inventory.refresh_interval, 120);

    // Check servers
    assert_eq!(config.servers.len(), 2);

    // Required server
    assert_eq!(config.servers[0].name, "required-server");
    assert!(config.servers[0].required);
    assert!(config.servers[0].proxy.is_none()); // Inherits global proxy

    // Optional server with custom proxy
    assert_eq!(config.servers[1].name, "optional-server");
    assert!(!config.servers[1].required);
    assert!(config.servers[1].proxy.is_some());
    assert_eq!(
        config.servers[1]
            .proxy
            .as_ref()
            .unwrap()
            .http
            .as_ref()
            .unwrap(),
        "http://server-specific-proxy:9090"
    );

    // Check warmup
    assert_eq!(config.warmup.len(), 1);
    assert_eq!(config.warmup[0].label, "local-dev");
}

#[tokio::test]
async fn test_yaml_backward_compatibility() {
    // Old config format should still work
    let yaml = r#"
servers:
  - name: "legacy-server"
    protocol: sse
    url: "http://localhost:3000/sse"
"#;

    let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse old format");
    assert_eq!(config.servers.len(), 1);
    assert_eq!(config.servers[0].name, "legacy-server");
    assert!(!config.servers[0].required); // New field defaults to false
    assert!(config.servers[0].proxy.is_none()); // New field defaults to None
    assert!(config.proxy.is_none()); // New field defaults to None
    assert!(config.warmup.is_empty()); // New field defaults to empty
}

/// A server's `proxy: null` is the missing field: the server inherits
/// the global proxy, it is not forced direct.
#[test]
fn a_null_server_proxy_inherits_the_global_proxy() {
    use crate::core::proxy::resolve_proxy_config;

    let yaml = r#"
proxy:
  http: "http://global-proxy:8080"

servers:
  - name: "inherits-the-global-proxy"
    protocol: sse
    url: "http://localhost:3000/sse"
    proxy: null
"#;

    let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");
    assert_eq!(config.servers.len(), 1);
    assert!(config.servers[0].proxy.is_none());
    let resolved =
        resolve_proxy_config(&config.servers[0], config.proxy.as_ref()).expect("the global proxy");
    assert_eq!(resolved.http.as_deref(), Some("http://global-proxy:8080"));
}

#[test]
fn test_transport_stdio() {
    let yaml = r#"
name: "test"
protocol: stdio
command: "mcp-server"
args: ["--port", "3000"]
envs:
  VAR1: "value1"
  VAR2: "value2"
"#;

    let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse stdio");
    assert_eq!(config.name, "test");

    match config.transport {
        McpTransport::Stdio {
            command,
            args,
            envs,
        } => {
            assert_eq!(command, "mcp-server");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], "--port");
            assert_eq!(envs.get("VAR1").unwrap(), "value1");
        }
        _ => panic!("Expected Stdio transport"),
    }
}

#[test]
fn test_transport_sse() {
    let yaml = r#"
name: "test"
protocol: sse
url: "http://localhost:3000/sse"
token: "secret"
"#;

    let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse sse");
    assert_eq!(config.name, "test");

    match config.transport {
        McpTransport::Sse { url, token, .. } => {
            assert_eq!(url, "http://localhost:3000/sse");
            assert_eq!(token.unwrap(), "secret");
        }
        _ => panic!("Expected Sse transport"),
    }
}

#[test]
fn test_transport_streamable() {
    let yaml = r#"
name: "test"
protocol: streamable
url: "http://localhost:3000"
"#;

    let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse streamable");
    assert_eq!(config.name, "test");

    match config.transport {
        McpTransport::Streamable { url, token, .. } => {
            assert_eq!(url, "http://localhost:3000");
            assert!(token.is_none());
        }
        _ => panic!("Expected Streamable transport"),
    }
}

#[test]
fn test_tool_config_with_alias() {
    let yaml = r#"
name: "brave"
protocol: sse
url: "https://mcp.brave.com/sse"
tools:
  brave_web_search:
    alias: web_search
    response_format: web_search_call
    arg_mapping:
      renames:
        q: query
      defaults:
        count: 10
"#;

    let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
    assert_eq!(config.name, "brave");
    let tools = config.tools.as_ref().unwrap();
    assert_eq!(tools.len(), 1);

    let tool_config = tools.get("brave_web_search").unwrap();
    assert_eq!(tool_config.alias, Some("web_search".to_string()));
    assert_eq!(
        tool_config.response_format,
        Some(ResponseFormatConfig::WebSearchCall)
    );

    let arg_mapping = tool_config.arg_mapping.as_ref().unwrap();
    assert_eq!(arg_mapping.renames.get("q").unwrap(), "query");
    assert_eq!(
        arg_mapping.defaults.get("count").unwrap(),
        &serde_json::json!(10)
    );
}

#[test]
fn test_tool_config_arg_mapping_overrides() {
    let yaml = r#"
name: "web-search"
protocol: sse
url: "http://localhost:8080/sse"
tools:
  search_web:
    response_format: web_search_call
    arg_mapping:
      defaults:
        enable_source: true
      overrides:
        enable_brave: false
"#;

    let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
    let tool_config = config
        .tools
        .as_ref()
        .and_then(|tools| tools.get("search_web"))
        .expect("search_web config should exist");

    let arg_mapping = tool_config.arg_mapping.as_ref().unwrap();
    assert_eq!(
        arg_mapping.defaults.get("enable_source").unwrap(),
        &serde_json::json!(true)
    );
    assert_eq!(
        arg_mapping.overrides.get("enable_brave").unwrap(),
        &serde_json::json!(false)
    );
}

#[test]
fn test_tool_config_format_only() {
    let yaml = r#"
name: "filesystem"
protocol: stdio
command: "npx"
args: ["-y", "@anthropic/mcp-server-filesystem"]
tools:
  search:
    response_format: file_search_call
"#;

    let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
    let tools = config.tools.as_ref().unwrap();
    assert_eq!(tools.len(), 1);

    let tool_config = tools.get("search").unwrap();
    assert!(tool_config.alias.is_none());
    assert_eq!(
        tool_config.response_format,
        Some(ResponseFormatConfig::FileSearchCall)
    );
    assert!(tool_config.arg_mapping.is_none());
}

#[test]
fn test_tool_config_defaults() {
    let yaml = r#"
name: "test"
protocol: sse
url: "http://localhost:3000/sse"
tools:
  my_tool: {}
"#;

    let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
    let tools = config.tools.as_ref().unwrap();
    let tool_config = tools.get("my_tool").unwrap();
    assert!(tool_config.alias.is_none());
    // `my_tool: {}` carries no `response_format` field, so the deserialized
    // value is `None` (meaning "inherit context"), not an explicit
    // `Passthrough`.
    assert_eq!(tool_config.response_format, None);
    assert!(tool_config.arg_mapping.is_none());
}

#[test]
fn test_response_format_config_serde() {
    let formats = vec![
        (ResponseFormatConfig::Passthrough, "\"passthrough\""),
        (ResponseFormatConfig::WebSearchCall, "\"web_search_call\""),
        (
            ResponseFormatConfig::CodeInterpreterCall,
            "\"code_interpreter_call\"",
        ),
        (ResponseFormatConfig::FileSearchCall, "\"file_search_call\""),
        (
            ResponseFormatConfig::ImageGenerationCall,
            "\"image_generation_call\"",
        ),
    ];

    for (format, expected) in formats {
        let serialized = serde_json::to_string(&format).unwrap();
        assert_eq!(serialized, expected);

        let deserialized: ResponseFormatConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, format);
    }
}

#[test]
fn test_multiple_tools_config() {
    let yaml = r#"
name: "multi-tool-server"
protocol: sse
url: "https://example.com/sse"
tools:
  tool_a:
    alias: a
    response_format: web_search_call
  tool_b:
    response_format: file_search_call
  tool_c:
    alias: c
"#;

    let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
    let tools = config.tools.as_ref().unwrap();
    assert_eq!(tools.len(), 3);

    let tool_a = tools.get("tool_a").unwrap();
    assert_eq!(tool_a.alias, Some("a".to_string()));
    assert_eq!(
        tool_a.response_format,
        Some(ResponseFormatConfig::WebSearchCall)
    );

    let tool_b = tools.get("tool_b").unwrap();
    assert!(tool_b.alias.is_none());
    assert_eq!(
        tool_b.response_format,
        Some(ResponseFormatConfig::FileSearchCall)
    );

    let tool_c = tools.get("tool_c").unwrap();
    assert_eq!(tool_c.alias, Some("c".to_string()));
    // Alias-only stanza: no `response_format` field → `None` (meaning
    // "inherit context"), not explicit Passthrough.
    assert_eq!(tool_c.response_format, None);
}

#[test]
fn test_policy_config_default() {
    let config = PolicyConfig::default();
    assert_eq!(config.default, PolicyDecisionConfig::Allow);
    assert!(config.servers.is_empty());
    assert!(config.tools.is_empty());
}

#[test]
fn test_policy_config_yaml_minimal() {
    let yaml = r"
servers: []
";

    let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
    assert_eq!(config.policy.default, PolicyDecisionConfig::Allow);
    assert!(config.policy.servers.is_empty());
}

#[test]
fn test_policy_config_yaml_full() {
    let yaml = r#"
servers:
  - name: "test"
    protocol: sse
    url: "http://localhost:3000/sse"

policy:
  default: allow
  servers:
    brave:
      trust_level: trusted
    untrusted_server:
      trust_level: untrusted
      default: deny
    sandbox_server:
      trust_level: sandboxed
  tools:
    "dangerous_server:delete_all": deny
    "risky_server:format_disk":
      deny_with_reason: "This operation is too dangerous"
"#;

    let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse");

    // Check default policy
    assert_eq!(config.policy.default, PolicyDecisionConfig::Allow);

    // Check server policies
    assert_eq!(config.policy.servers.len(), 3);

    let brave = config.policy.servers.get("brave").unwrap();
    assert_eq!(brave.trust_level, TrustLevelConfig::Trusted);
    assert_eq!(brave.default, PolicyDecisionConfig::Allow);

    let untrusted = config.policy.servers.get("untrusted_server").unwrap();
    assert_eq!(untrusted.trust_level, TrustLevelConfig::Untrusted);
    assert_eq!(untrusted.default, PolicyDecisionConfig::Deny);

    let sandbox = config.policy.servers.get("sandbox_server").unwrap();
    assert_eq!(sandbox.trust_level, TrustLevelConfig::Sandboxed);

    // Check tool policies
    assert_eq!(config.policy.tools.len(), 2);
    assert_eq!(
        config.policy.tools.get("dangerous_server:delete_all"),
        Some(&PolicyDecisionConfig::Deny)
    );
    assert_eq!(
        config.policy.tools.get("risky_server:format_disk"),
        Some(&PolicyDecisionConfig::DenyWithReason(
            "This operation is too dangerous".to_string()
        ))
    );
}

#[test]
fn test_trust_level_config_serde() {
    let levels = vec![
        (TrustLevelConfig::Trusted, "\"trusted\""),
        (TrustLevelConfig::Standard, "\"standard\""),
        (TrustLevelConfig::Untrusted, "\"untrusted\""),
        (TrustLevelConfig::Sandboxed, "\"sandboxed\""),
    ];

    for (level, expected) in levels {
        let serialized = serde_json::to_string(&level).unwrap();
        assert_eq!(serialized, expected);

        let deserialized: TrustLevelConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, level);
    }
}

#[test]
fn test_policy_decision_config_serde() {
    let decisions = vec![
        (PolicyDecisionConfig::Allow, "\"allow\""),
        (PolicyDecisionConfig::Deny, "\"deny\""),
    ];

    for (decision, expected) in decisions {
        let serialized = serde_json::to_string(&decision).unwrap();
        assert_eq!(serialized, expected);

        let deserialized: PolicyDecisionConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, decision);
    }

    // Test deny_with_reason
    let deny_with_reason = PolicyDecisionConfig::DenyWithReason("Not allowed".to_string());
    let serialized = serde_json::to_string(&deny_with_reason).unwrap();
    assert!(serialized.contains("deny_with_reason"));
    assert!(serialized.contains("Not allowed"));
}

#[test]
fn test_builtin_tool_type_serde() {
    let types = vec![
        (BuiltinToolType::WebSearchPreview, "\"web_search_preview\""),
        (BuiltinToolType::CodeInterpreter, "\"code_interpreter\""),
        (BuiltinToolType::FileSearch, "\"file_search\""),
        (BuiltinToolType::ImageGeneration, "\"image_generation\""),
    ];

    for (builtin_type, expected) in types {
        let serialized = serde_json::to_string(&builtin_type).unwrap();
        assert_eq!(serialized, expected);

        let deserialized: BuiltinToolType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, builtin_type);
    }
}

#[test]
fn test_builtin_tool_type_all_includes_every_variant() {
    let all = BuiltinToolType::all();
    for variant in [
        BuiltinToolType::WebSearchPreview,
        BuiltinToolType::CodeInterpreter,
        BuiltinToolType::FileSearch,
        BuiltinToolType::ImageGeneration,
    ] {
        assert!(all.contains(&variant), "all() missing {variant}");
    }
}

#[test]
fn test_builtin_tool_type_response_format() {
    assert_eq!(
        BuiltinToolType::WebSearchPreview.response_format(),
        ResponseFormatConfig::WebSearchCall
    );
    assert_eq!(
        BuiltinToolType::CodeInterpreter.response_format(),
        ResponseFormatConfig::CodeInterpreterCall
    );
    assert_eq!(
        BuiltinToolType::FileSearch.response_format(),
        ResponseFormatConfig::FileSearchCall
    );
    assert_eq!(
        BuiltinToolType::ImageGeneration.response_format(),
        ResponseFormatConfig::ImageGenerationCall
    );
}

#[test]
fn test_server_config_with_builtin_type() {
    let yaml = r#"
name: "brave-search"
protocol: sse
url: "https://mcp.brave.com/sse"
token: "${BRAVE_API_KEY}"
builtin_type: web_search_preview
builtin_tool_name: brave_web_search
"#;

    let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
    assert_eq!(config.name, "brave-search");
    assert_eq!(config.builtin_type, Some(BuiltinToolType::WebSearchPreview));
    assert_eq!(
        config.builtin_tool_name,
        Some("brave_web_search".to_string())
    );
    assert!(!config.internal);
}

#[test]
fn test_server_config_without_builtin_type() {
    let yaml = r#"
name: "regular-server"
protocol: sse
url: "http://localhost:3000/sse"
"#;

    let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
    assert_eq!(config.name, "regular-server");
    assert!(config.builtin_type.is_none());
    assert!(config.builtin_tool_name.is_none());
    assert!(!config.internal);
}

#[test]
fn test_server_config_with_internal_visibility() {
    let yaml = r#"
name: "internal-server"
protocol: sse
url: "http://localhost:3000/sse"
internal: true
"#;

    let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
    assert_eq!(config.name, "internal-server");
    assert!(config.internal);
}

#[test]
fn test_full_config_with_builtin_servers() {
    let yaml = r#"
servers:
  - name: brave
    protocol: sse
    url: "https://mcp.brave.com/sse"
    builtin_type: web_search_preview
    builtin_tool_name: brave_web_search
    tools:
      brave_web_search:
        response_format: web_search_call

  - name: code-runner
    protocol: stdio
    command: "code-interpreter"
    builtin_type: code_interpreter
    builtin_tool_name: execute

  - name: regular-mcp
    protocol: sse
    url: "http://localhost:3000/sse"
"#;

    let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse");
    assert_eq!(config.servers.len(), 3);

    // Brave with web_search_preview
    assert_eq!(
        config.servers[0].builtin_type,
        Some(BuiltinToolType::WebSearchPreview)
    );
    assert_eq!(
        config.servers[0].builtin_tool_name,
        Some("brave_web_search".to_string())
    );

    // Code interpreter
    assert_eq!(
        config.servers[1].builtin_type,
        Some(BuiltinToolType::CodeInterpreter)
    );
    assert_eq!(
        config.servers[1].builtin_tool_name,
        Some("execute".to_string())
    );

    // Regular MCP server (no builtin type)
    assert!(config.servers[2].builtin_type.is_none());
    assert!(config.servers[2].builtin_tool_name.is_none());
}

#[test]
fn test_validate_valid_config_no_builtin() {
    let config = sse_server("test", "http://localhost:3000/sse");

    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_valid_config_with_builtin() {
    let config = McpServerConfig {
        builtin_type: Some(BuiltinToolType::WebSearchPreview),
        builtin_tool_name: Some("brave_web_search".to_string()),
        ..sse_server("brave", "http://localhost:3000/sse")
    };

    assert!(config.validate().is_ok());
}

#[test]
fn test_validate_missing_builtin_tool_name() {
    let config = McpServerConfig {
        builtin_type: Some(BuiltinToolType::WebSearchPreview),
        builtin_tool_name: None, // Missing!
        ..sse_server("brave", "http://localhost:3000/sse")
    };

    let err = config.validate().unwrap_err();
    assert!(matches!(
        err,
        ConfigValidationError::MissingBuiltinToolName { .. }
    ));
    assert!(err.to_string().contains("brave"));
    assert!(err.to_string().contains("builtin_tool_name is missing"));
}

#[test]
fn test_validate_missing_builtin_type() {
    let config = McpServerConfig {
        builtin_type: None, // Missing!
        builtin_tool_name: Some("brave_web_search".to_string()),
        ..sse_server("brave", "http://localhost:3000/sse")
    };

    let err = config.validate().unwrap_err();
    assert!(matches!(
        err,
        ConfigValidationError::MissingBuiltinType { .. }
    ));
    assert!(err.to_string().contains("brave"));
    assert!(err.to_string().contains("builtin_type is missing"));
}

#[test]
fn test_mcp_config_validate_no_duplicates() {
    let config = McpConfig {
        servers: vec![
            McpServerConfig {
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("search".to_string()),
                ..sse_server("brave", "http://localhost:3000/sse")
            },
            McpServerConfig {
                builtin_type: Some(BuiltinToolType::CodeInterpreter),
                builtin_tool_name: Some("execute".to_string()),
                ..sse_server("code-runner", "http://localhost:3001/sse")
            },
        ],
        ..Default::default()
    };

    assert!(config.validate().is_ok());
}

#[test]
fn test_mcp_config_validate_duplicate_builtin_type() {
    let config = McpConfig {
        servers: vec![
            McpServerConfig {
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("search1".to_string()),
                ..sse_server("brave1", "http://localhost:3000/sse")
            },
            McpServerConfig {
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("search2".to_string()),
                ..sse_server("brave2", "http://localhost:3001/sse")
            },
        ],
        ..Default::default()
    };

    let err = config.validate().unwrap_err();
    assert!(matches!(
        err,
        ConfigValidationError::DuplicateBuiltinType { .. }
    ));
    assert!(err.to_string().contains("brave1"));
    assert!(err.to_string().contains("brave2"));
    assert!(err.to_string().contains("web_search_preview"));
}

#[test]
fn test_builtin_tool_type_display() {
    assert_eq!(
        BuiltinToolType::WebSearchPreview.to_string(),
        "web_search_preview"
    );
    assert_eq!(
        BuiltinToolType::CodeInterpreter.to_string(),
        "code_interpreter"
    );
    assert_eq!(BuiltinToolType::FileSearch.to_string(), "file_search");
}
