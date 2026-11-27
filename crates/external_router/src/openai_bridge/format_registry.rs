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
#[path = "../../tests/unit/openai_bridge/format_registry/tests.rs"]
mod tests;
