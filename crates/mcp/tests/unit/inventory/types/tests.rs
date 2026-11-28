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
