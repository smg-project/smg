use std::{borrow::Cow, sync::Arc};

use rmcp::model::{Tool, ToolAnnotations as RmcpToolAnnotations};
use smg_mcp::ToolEntry;

use super::*;

// `Tool` and `ToolAnnotations` are `#[non_exhaustive]` in rmcp 1.7, so build
// them via the constructor/setters rather than struct literals.
fn entry_with_rmcp_annotations(annotations: Option<RmcpToolAnnotations>) -> ToolEntry {
    let mut tool = Tool::new(
        Cow::Owned("widget".to_string()),
        Cow::Owned("widget description".to_string()),
        Arc::new(serde_json::Map::new()),
    );
    tool.annotations = annotations;
    ToolEntry::from_server_tool("srv", tool)
}

fn read_only_hint(value: bool) -> RmcpToolAnnotations {
    RmcpToolAnnotations::new().read_only(value)
}

#[test]
fn build_mcp_tool_infos_surfaces_rmcp_read_only_hint() {
    let entries = vec![entry_with_rmcp_annotations(Some(read_only_hint(true)))];
    let infos = build_mcp_tool_infos(&entries);
    let serialized = serde_json::to_value(&infos[0])
        .expect("McpToolInfo must serialize")
        .get("annotations")
        .cloned()
        .expect("annotations must serialize");
    assert_eq!(serialized, json!({ "read_only": true }));
}

#[test]
fn build_mcp_tool_infos_defaults_to_false_when_hint_absent() {
    let entries = vec![entry_with_rmcp_annotations(None)];
    let infos = build_mcp_tool_infos(&entries);
    let serialized = serde_json::to_value(&infos[0])
        .expect("McpToolInfo must serialize")
        .get("annotations")
        .cloned()
        .expect("annotations must serialize");
    assert_eq!(serialized, json!({ "read_only": false }));
}

/// The JSON tool info is the typed `McpToolInfo` serialised, byte for
/// byte: the hint read the same way, the description present or absent,
/// the schema's keys in their order.
#[test]
fn the_json_tool_info_is_the_typed_one() {
    let mut described = entry_with_rmcp_annotations(Some(read_only_hint(true)));
    described.tool.input_schema = Arc::new(
        json!({
            "type": "object",
            "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
            "required": ["q"]
        })
        .as_object()
        .cloned()
        .expect("an object schema"),
    );
    let mut bare = entry_with_rmcp_annotations(None);
    bare.tool.description = None;
    for entry in [described, bare] {
        let typed = serde_json::to_string(&build_mcp_tool_infos(std::slice::from_ref(&entry))[0])
            .expect("McpToolInfo must serialize");
        assert_eq!(
            serde_json::to_string(&mcp_tool_info_json(&entry)).expect("a JSON info"),
            typed
        );
    }
}
