//! Side-map of `QualifiedToolName → ResponseFormat`, populated when MCP
//! servers register and queried by router code at request time.

use std::sync::Arc;

use dashmap::DashMap;
use smg_mcp::{inventory::ALIAS_SERVER_KEY, McpServerConfig, QualifiedToolName};
use tracing::debug;

use super::ResponseFormat;

/// Resolve an exposed tool name's `ResponseFormat` via the session's name map
/// and the registry. Returns `Passthrough` for unknown tools.
///
/// Lives next to `FormatRegistry` because it's a thin lookup helper that
/// composes the session's name map with `FormatRegistry::lookup`. Reuses the
/// `QualifiedToolName` returned by `qualified_name_for_exposed` rather than
/// rebuilding one, so we pay the two `Arc<str>` allocations once instead of
/// twice per call.
///
/// Telemetry: when the session knows the tool but the registry doesn't, we
/// log a `debug` event with the qualified name. That asymmetric miss is the
/// fingerprint of a registration-path bug (the orchestrator added the tool
/// but `populate_from_server_config` was never called for the same server),
/// and would otherwise dispatch silently as `mcp_call`.
pub fn lookup_tool_format(
    session: &smg_mcp::McpToolSession<'_>,
    registry: &FormatRegistry,
    exposed_name: &str,
) -> ResponseFormat {
    let Some(qn) = session.qualified_name_for_exposed(exposed_name) else {
        return ResponseFormat::Passthrough;
    };
    let format = registry.lookup(&qn);
    if format == ResponseFormat::Passthrough && !registry.contains(&qn) {
        debug!(
            exposed_name = %exposed_name,
            server_key = %qn.server_key(),
            tool_name = %qn.tool_name(),
            "FormatRegistry miss for session-exposed tool — dispatching as Passthrough; \
             check that populate_from_server_config ran for this server"
        );
    }
    format
}

#[derive(Default, Debug, Clone)]
pub struct FormatRegistry {
    formats: Arc<DashMap<QualifiedToolName, ResponseFormat>>,
}

impl FormatRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lookup(&self, qualified: &QualifiedToolName) -> ResponseFormat {
        self.formats
            .get(qualified)
            .map(|r| *r.value())
            .unwrap_or(ResponseFormat::Passthrough)
    }

    pub fn lookup_by_names(&self, server_key: &str, tool_name: &str) -> ResponseFormat {
        self.lookup(&QualifiedToolName::new(server_key, tool_name))
    }

    /// True iff the registry has an explicit entry for `qualified`. Used by
    /// [`lookup_tool_format`] to distinguish a registered Passthrough entry
    /// from a missing entry that defaulted to Passthrough.
    pub fn contains(&self, qualified: &QualifiedToolName) -> bool {
        self.formats.contains_key(qualified)
    }

    fn insert(&self, qualified: QualifiedToolName, format: ResponseFormat) {
        self.formats.insert(qualified, format);
    }

    fn remove(&self, qualified: &QualifiedToolName) {
        self.formats.remove(qualified);
    }

    /// Populate from a server config. Safe to call repeatedly.
    ///
    /// `McpToolSession::collect_visible_mcp_tools` replaces a direct tool
    /// entry with its alias entry, so production session lookup of an
    /// aliased tool resolves through `("alias", alias_name)`. Direct
    /// dispatch still uses `(server_key, tool_name)`. Both keys must carry
    /// the same format, so non-Passthrough formats are mirrored on both.
    pub fn populate_from_server_config(&self, config: &McpServerConfig) {
        if let Some(tools) = &config.tools {
            for (tool_name, tool_config) in tools {
                let direct_key = QualifiedToolName::new(&config.name, tool_name);
                let alias_key = tool_config
                    .alias
                    .as_deref()
                    .map(|alias| QualifiedToolName::new(ALIAS_SERVER_KEY, alias));

                let Some(format_config) = tool_config.response_format else {
                    continue;
                };
                let format: ResponseFormat = format_config.into();
                if format == ResponseFormat::Passthrough {
                    self.remove(&direct_key);
                    if let Some(alias_key) = &alias_key {
                        self.remove(alias_key);
                    }
                    continue;
                }

                self.insert(direct_key, format);
                if let Some(alias_key) = alias_key {
                    self.insert(alias_key, format);
                }
            }
        }

        if let (Some(builtin_type), Some(tool_name)) =
            (&config.builtin_type, &config.builtin_tool_name)
        {
            let stanza = config.tools.as_ref().and_then(|tools| tools.get(tool_name));
            let has_explicit_format = stanza.is_some_and(|cfg| cfg.response_format.is_some());
            if !has_explicit_format {
                let format: ResponseFormat = builtin_type.response_format().into();
                self.insert(QualifiedToolName::new(&config.name, tool_name), format);
                if let Some(alias) = stanza.and_then(|cfg| cfg.alias.as_deref()) {
                    self.insert(QualifiedToolName::new(ALIAS_SERVER_KEY, alias), format);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;
    use smg_mcp::{
        BuiltinToolType, McpConfig, McpOrchestrator, McpServerBinding, McpServerConfig,
        McpToolSession, McpTransport, ResponseFormatConfig, Tool, ToolConfig, ToolEntry,
    };

    use super::*;

    fn server(name: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.to_string(),
            transport: McpTransport::Streamable {
                url: "http://x".to_string(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: None,
            builtin_type: None,
            builtin_tool_name: None,
            internal: false,
        }
    }

    fn test_tool(name: &str) -> Tool {
        let mut schema = serde_json::Map::new();
        schema.insert("type".to_string(), json!("object"));
        schema.insert("properties".to_string(), json!({}));
        Tool::new(name.to_string(), "test", schema)
    }

    #[test]
    fn lookup_unknown_returns_passthrough() {
        let r = FormatRegistry::new();
        assert_eq!(
            r.lookup_by_names("any", "tool"),
            ResponseFormat::Passthrough
        );
    }

    #[test]
    fn alias_format_mirrored_on_both_keys() {
        let mut tools = HashMap::new();
        tools.insert(
            "brave_web_search".to_string(),
            ToolConfig {
                alias: Some("web_search".to_string()),
                response_format: Some(ResponseFormatConfig::WebSearchCall),
                arg_mapping: None,
            },
        );
        let mut cfg = server("brave");
        cfg.tools = Some(tools);

        let r = FormatRegistry::new();
        r.populate_from_server_config(&cfg);

        assert_eq!(
            r.lookup_by_names("alias", "web_search"),
            ResponseFormat::WebSearchCall,
        );
        assert_eq!(
            r.lookup_by_names("brave", "brave_web_search"),
            ResponseFormat::WebSearchCall,
        );
    }

    #[test]
    fn non_aliased_tool_stores_format_under_server_tool_pair() {
        let mut tools = HashMap::new();
        tools.insert(
            "search".to_string(),
            ToolConfig {
                alias: None,
                response_format: Some(ResponseFormatConfig::WebSearchCall),
                arg_mapping: None,
            },
        );
        let mut cfg = server("brave");
        cfg.tools = Some(tools);

        let r = FormatRegistry::new();
        r.populate_from_server_config(&cfg);

        assert_eq!(
            r.lookup_by_names("brave", "search"),
            ResponseFormat::WebSearchCall
        );
    }

    #[test]
    fn builtin_default_applies_when_no_explicit_tool_config() {
        let mut cfg = server("search");
        cfg.builtin_type = Some(BuiltinToolType::WebSearchPreview);
        cfg.builtin_tool_name = Some("do_search".to_string());

        let r = FormatRegistry::new();
        r.populate_from_server_config(&cfg);

        assert_eq!(
            r.lookup_by_names("search", "do_search"),
            ResponseFormat::WebSearchCall
        );
    }

    #[test]
    fn explicit_per_tool_override_wins_over_builtin_default() {
        let mut tools = HashMap::new();
        tools.insert(
            "do_search".to_string(),
            ToolConfig {
                alias: None,
                response_format: Some(ResponseFormatConfig::Passthrough),
                arg_mapping: None,
            },
        );
        let mut cfg = server("search");
        cfg.tools = Some(tools);
        cfg.builtin_type = Some(BuiltinToolType::WebSearchPreview);
        cfg.builtin_tool_name = Some("do_search".to_string());

        let r = FormatRegistry::new();
        r.populate_from_server_config(&cfg);

        assert_eq!(
            r.lookup_by_names("search", "do_search"),
            ResponseFormat::Passthrough
        );
    }

    #[test]
    fn alias_only_stanza_preserves_builtin_default_on_both_keys() {
        let mut tools = HashMap::new();
        tools.insert(
            "do_search".to_string(),
            ToolConfig {
                alias: Some("web_search".to_string()),
                response_format: None,
                arg_mapping: None,
            },
        );
        let mut cfg = server("search");
        cfg.tools = Some(tools);
        cfg.builtin_type = Some(BuiltinToolType::WebSearchPreview);
        cfg.builtin_tool_name = Some("do_search".to_string());

        let r = FormatRegistry::new();
        r.populate_from_server_config(&cfg);

        // Alias key — what production session lookup hits.
        assert_eq!(
            r.lookup_by_names("alias", "web_search"),
            ResponseFormat::WebSearchCall,
        );
        // Direct key — what direct dispatch hits.
        assert_eq!(
            r.lookup_by_names("search", "do_search"),
            ResponseFormat::WebSearchCall,
        );
    }

    #[test]
    fn explicit_passthrough_downgrade_clears_prior_hosted_entry() {
        let r = FormatRegistry::new();

        let mut hosted = HashMap::new();
        hosted.insert(
            "brave_web_search".to_string(),
            ToolConfig {
                alias: Some("web_search".to_string()),
                response_format: Some(ResponseFormatConfig::WebSearchCall),
                arg_mapping: None,
            },
        );
        let mut hosted_cfg = server("brave");
        hosted_cfg.tools = Some(hosted);
        r.populate_from_server_config(&hosted_cfg);
        assert_eq!(
            r.lookup_by_names("alias", "web_search"),
            ResponseFormat::WebSearchCall,
        );
        assert_eq!(
            r.lookup_by_names("brave", "brave_web_search"),
            ResponseFormat::WebSearchCall,
        );

        let mut downgraded = HashMap::new();
        downgraded.insert(
            "brave_web_search".to_string(),
            ToolConfig {
                alias: Some("web_search".to_string()),
                response_format: Some(ResponseFormatConfig::Passthrough),
                arg_mapping: None,
            },
        );
        let mut downgraded_cfg = server("brave");
        downgraded_cfg.tools = Some(downgraded);
        r.populate_from_server_config(&downgraded_cfg);

        assert_eq!(
            r.lookup_by_names("alias", "web_search"),
            ResponseFormat::Passthrough,
        );
        assert_eq!(
            r.lookup_by_names("brave", "brave_web_search"),
            ResponseFormat::Passthrough,
        );
    }

    async fn orchestrator_with_tool(server_name: &str, tool_name: &str) -> McpOrchestrator {
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![server(server_name)],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                server_name,
                test_tool(tool_name),
            ));
        orchestrator
    }

    fn binding(server_name: &str) -> Vec<McpServerBinding> {
        vec![McpServerBinding {
            label: server_name.to_string(),
            server_key: server_name.to_string(),
            allowed_tools: None,
        }]
    }

    #[tokio::test]
    async fn lookup_tool_format_returns_passthrough_when_session_unknown() {
        let orchestrator = orchestrator_with_tool("brave", "brave_web_search").await;
        let session = McpToolSession::new(&orchestrator, binding("brave"), "test-request");
        let registry = FormatRegistry::new();
        // `not_a_tool` was never registered with the session, so the lookup
        // short-circuits at `qualified_name_for_exposed`.
        assert_eq!(
            lookup_tool_format(&session, &registry, "not_a_tool"),
            ResponseFormat::Passthrough
        );
    }

    #[tokio::test]
    async fn lookup_tool_format_returns_passthrough_for_session_known_but_registry_missing() {
        // Asymmetric-miss branch: the session exposes the tool but the
        // registry has no entry — production fingerprint of a server whose
        // `populate_from_server_config` was skipped. Must dispatch as
        // Passthrough rather than panicking or making up a hosted format.
        let orchestrator = orchestrator_with_tool("brave", "brave_web_search").await;
        let session = McpToolSession::new(&orchestrator, binding("brave"), "test-request");
        let registry = FormatRegistry::new();
        assert_eq!(
            lookup_tool_format(&session, &registry, "brave_web_search"),
            ResponseFormat::Passthrough
        );
    }

    #[tokio::test]
    async fn lookup_tool_format_returns_registry_value_for_session_known_and_registered() {
        // Happy path: session knows the tool AND registry has a hosted entry —
        // composed lookup returns the hosted format.
        let orchestrator = orchestrator_with_tool("brave", "brave_web_search").await;
        let session = McpToolSession::new(&orchestrator, binding("brave"), "test-request");
        let mut tools = HashMap::new();
        tools.insert(
            "brave_web_search".to_string(),
            ToolConfig {
                alias: None,
                response_format: Some(ResponseFormatConfig::WebSearchCall),
                arg_mapping: None,
            },
        );
        let mut cfg = server("brave");
        cfg.tools = Some(tools);
        let registry = FormatRegistry::new();
        registry.populate_from_server_config(&cfg);

        assert_eq!(
            lookup_tool_format(&session, &registry, "brave_web_search"),
            ResponseFormat::WebSearchCall
        );
    }
}
