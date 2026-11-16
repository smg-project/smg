//! Core types for the tool inventory.

use std::{fmt, sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use crate::{annotations::ToolAnnotations, core::config::Tool, tenant::TenantId};

/// Category of a tool for filtering and visibility control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ToolCategory {
    #[default]
    Static,
    Alias,
    Dynamic,
    Custom,
    /// Built-in tools (web_search, code_interpreter, file_search) exposed via aliasing.
    Builtin,
}

/// Synthetic `server_key` used by alias entries created via
/// `McpOrchestrator::register_alias`. Exported so gateway-side code that
/// indexes by `QualifiedToolName` (e.g. the response-format registry) can
/// reconstruct the same key without hardcoding the literal.
pub const ALIAS_SERVER_KEY: &str = "alias";

/// Unique tool identifier: `server_key:tool_name`.
///
/// Uses `Arc<str>` internally for cheap cloning in hot paths.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QualifiedToolName {
    server_key: Arc<str>,
    tool_name: Arc<str>,
}

impl QualifiedToolName {
    pub fn new(server_key: impl AsRef<str>, tool_name: impl AsRef<str>) -> Self {
        Self {
            server_key: Arc::from(server_key.as_ref()),
            tool_name: Arc::from(tool_name.as_ref()),
        }
    }

    /// Parse from "server:tool" format.
    pub fn parse(s: &str) -> Option<Self> {
        let (server, tool) = s.split_once(':')?;
        Some(Self::new(server, tool))
    }

    pub fn server_key(&self) -> &str {
        &self.server_key
    }

    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }
}

impl fmt::Display for QualifiedToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.server_key, self.tool_name)
    }
}

impl Serialize for QualifiedToolName {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for QualifiedToolName {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).ok_or_else(|| serde::de::Error::custom("expected format: server:tool"))
    }
}

/// Target of a tool alias (e.g., `web_search` → `brave:brave_web_search`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AliasTarget {
    pub target: QualifiedToolName,
    pub arg_mapping: Option<ArgMapping>,
}

impl AliasTarget {
    pub fn new(target: QualifiedToolName) -> Self {
        Self {
            target,
            arg_mapping: None,
        }
    }

    #[must_use]
    pub fn with_arg_mapping(mut self, mapping: ArgMapping) -> Self {
        self.arg_mapping = Some(mapping);
        self
    }
}

/// Argument mapping for tool aliases (renames and defaults).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ArgMapping {
    pub renames: Vec<(String, String)>,
    pub defaults: Vec<(String, serde_json::Value)>,
    pub overrides: Vec<(String, serde_json::Value)>,
}

impl ArgMapping {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_rename(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.renames.push((from.into(), to.into()));
        self
    }

    #[must_use]
    pub fn with_default(mut self, name: impl Into<String>, value: serde_json::Value) -> Self {
        self.defaults.push((name.into(), value));
        self
    }

    #[must_use]
    pub fn with_override(mut self, name: impl Into<String>, value: serde_json::Value) -> Self {
        self.overrides.push((name.into(), value));
        self
    }
}

/// Tool entry with metadata for approval, caching, and multi-tenancy.
#[derive(Debug, Clone)]
pub struct ToolEntry {
    pub qualified_name: QualifiedToolName,
    pub tool: Tool,
    pub category: ToolCategory,
    pub annotations: ToolAnnotations,
    pub tenant_id: Option<TenantId>,
    pub alias_target: Option<AliasTarget>,
    pub arg_mapping: Option<ArgMapping>,
    pub cached_at: Instant,
    pub ttl: Option<Duration>,
    /// The tool's position in its server's listing (`tools/list` order): the
    /// order the server's tools are listed and offered in. `None` for an
    /// entry made otherwise than by a listing; such an entry sorts after
    /// the listed ones.
    pub position: Option<usize>,
}

impl ToolEntry {
    pub fn new(qualified_name: QualifiedToolName, tool: Tool) -> Self {
        Self {
            qualified_name,
            tool,
            category: ToolCategory::default(),
            annotations: ToolAnnotations::default(),
            tenant_id: None,
            alias_target: None,
            arg_mapping: None,
            cached_at: Instant::now(),
            ttl: None,
            position: None,
        }
    }

    pub fn from_server_tool(server_key: impl AsRef<str>, tool: Tool) -> Self {
        let name = tool.name.to_string();
        Self::new(QualifiedToolName::new(server_key, name), tool)
    }

    #[must_use]
    pub fn with_category(mut self, category: ToolCategory) -> Self {
        self.category = category;
        self
    }

    #[must_use]
    pub fn with_annotations(mut self, annotations: ToolAnnotations) -> Self {
        self.annotations = annotations;
        self
    }

    #[must_use]
    pub fn with_tenant(mut self, tenant_id: TenantId) -> Self {
        self.tenant_id = Some(tenant_id);
        self
    }

    #[must_use]
    pub fn with_alias(mut self, target: AliasTarget) -> Self {
        self.alias_target = Some(target);
        self.category = ToolCategory::Alias;
        self
    }

    #[must_use]
    pub fn with_arg_mapping(mut self, mapping: ArgMapping) -> Self {
        self.arg_mapping = Some(mapping);
        self
    }

    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    #[must_use]
    pub fn with_position(mut self, position: usize) -> Self {
        self.position = Some(position);
        self
    }

    pub fn is_expired(&self) -> bool {
        self.ttl
            .map(|ttl| self.cached_at.elapsed() > ttl)
            .unwrap_or(false)
    }

    pub fn server_key(&self) -> &str {
        &self.qualified_name.server_key
    }

    pub fn tool_name(&self) -> &str {
        &self.qualified_name.tool_name
    }
}

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, sync::Arc};

    use super::*;

    fn create_test_tool(name: &str) -> Tool {
        let schema_obj = serde_json::json!({
            "type": "object",
            "properties": {}
        });
        let schema_map = if let serde_json::Value::Object(m) = schema_obj {
            m
        } else {
            serde_json::Map::new()
        };

        Tool::new(
            Cow::Owned(name.to_string()),
            Cow::Owned(format!("Test tool: {name}")),
            Arc::new(schema_map),
        )
    }

    #[test]
    fn test_qualified_tool_name() {
        let name = QualifiedToolName::new("server", "tool");
        assert_eq!(name.server_key(), "server");
        assert_eq!(name.tool_name(), "tool");
        assert_eq!(format!("{name}"), "server:tool");
    }

    #[test]
    fn test_qualified_tool_name_parse() {
        let parsed = QualifiedToolName::parse("brave:web_search").unwrap();
        assert_eq!(parsed.server_key(), "brave");
        assert_eq!(parsed.tool_name(), "web_search");

        assert!(QualifiedToolName::parse("no_colon").is_none());
    }

    #[test]
    fn test_tool_entry_creation() {
        let tool = create_test_tool("my_tool");
        let entry = ToolEntry::from_server_tool("my_server", tool);

        assert_eq!(entry.server_key(), "my_server");
        assert_eq!(entry.tool_name(), "my_tool");
        assert_eq!(entry.category, ToolCategory::Static);
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_tool_entry_with_alias() {
        let tool = create_test_tool("web_search");
        let target = AliasTarget::new(QualifiedToolName::new("brave", "brave_web_search"));
        let entry = ToolEntry::from_server_tool("aliases", tool).with_alias(target);

        assert_eq!(entry.category, ToolCategory::Alias);
        assert!(entry.alias_target.is_some());
        assert_eq!(
            entry.alias_target.unwrap().target.tool_name(),
            "brave_web_search"
        );
    }

    #[test]
    fn test_tool_entry_expiration() {
        let tool = create_test_tool("expiring_tool");
        let entry = ToolEntry::from_server_tool("server", tool).with_ttl(Duration::from_millis(1));

        // Should not be expired immediately
        assert!(!entry.is_expired());

        // After waiting, should be expired
        std::thread::sleep(Duration::from_millis(5));
        assert!(entry.is_expired());
    }

    #[test]
    fn test_arg_mapping() {
        let mapping = ArgMapping::new()
            .with_rename("query", "search_query")
            .with_default("limit", serde_json::json!(10));

        assert_eq!(mapping.renames.len(), 1);
        assert_eq!(mapping.defaults.len(), 1);
    }
}
