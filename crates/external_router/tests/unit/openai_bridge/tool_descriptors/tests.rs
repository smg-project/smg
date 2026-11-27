use std::{borrow::Cow, sync::Arc};

use rmcp::model::{Tool, ToolAnnotations as RmcpToolAnnotations};
use smg_mcp::{
    McpConfig, McpOrchestrator, McpServerBinding, ToolEntry, ToolExecutionInput,
    ToolExecutionResult,
};

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

/// Whether a call of `echo` and a call of `write` wait for an approval, in
/// that order, once the `require_approval` of a request's one server entry
/// is applied to a session that lists both tools.
async fn waits_for_approval(require_approval: Value) -> [bool; 2] {
    let orchestrator = McpOrchestrator::new(McpConfig::default())
        .await
        .expect("an orchestrator");
    for name in ["echo", "write"] {
        let tool = Tool::new(
            Cow::Borrowed(name),
            Cow::Borrowed("a tool"),
            Arc::new(serde_json::Map::new()),
        );
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool("srv", tool));
    }
    let binding = McpServerBinding {
        label: "tools".to_string(),
        server_key: "srv".to_string(),
        allowed_tools: None,
        list_error: None,
        pool_key: None,
    };
    let mut session = McpToolSession::new(&orchestrator, vec![binding], "test-request");
    let entry: ResponseTool = serde_json::from_value(json!({
        "type": "mcp",
        "server_label": "tools",
        "require_approval": require_approval,
    }))
    .expect("a server entry");
    configure_response_tools_approval(&mut session, &[entry]);

    let mut waits = [false; 2];
    for (waits, name) in waits.iter_mut().zip(["echo", "write"]) {
        let result = session
            .execute_tool_result(ToolExecutionInput {
                call_id: format!("call-{name}"),
                tool_name: name.to_string(),
                arguments: json!({}),
            })
            .await;
        *waits = matches!(result, ToolExecutionResult::PendingApproval(_));
    }
    waits
}

/// The object form of `require_approval`: an `always` filter of names asks
/// an approval of the named tools alone; one that reads `read_only`, or
/// names nothing, asks it of every tool; a tool also under `never` is still
/// asked for.
#[tokio::test]
async fn an_always_filter_asks_approval_of_the_tools_it_names() {
    for (require_approval, waits) in [
        (json!({"always": {"tool_names": ["echo"]}}), [true, false]),
        (json!({"always": {"read_only": true}}), [true, true]),
        (
            json!({"always": {"read_only": false, "tool_names": ["echo"]}}),
            [true, true],
        ),
        (json!({"always": {}}), [true, true]),
        (
            json!({"always": {"tool_names": ["echo"]}, "never": {"tool_names": ["echo", "write"]}}),
            [true, false],
        ),
    ] {
        assert_eq!(
            waits_for_approval(require_approval.clone()).await,
            waits,
            "{require_approval}"
        );
    }
}

/// What asked no approval asks none: a `never` filter alone and the string
/// `never`; the string `always` asks it of every tool.
#[tokio::test]
async fn a_never_filter_asks_no_approval_and_the_strings_ask_as_before() {
    for (require_approval, waits) in [
        (json!({"never": {"tool_names": ["echo"]}}), [false, false]),
        (json!({"never": {}}), [false, false]),
        (json!("never"), [false, false]),
        (json!("always"), [true, true]),
    ] {
        assert_eq!(
            waits_for_approval(require_approval.clone()).await,
            waits,
            "{require_approval}"
        );
    }
}
