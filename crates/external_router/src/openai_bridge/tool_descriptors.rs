//! Builders for Responses/Chat tool payloads derived from MCP tool inventory.
//!
//! `ToolEntry` stores its JSON Schema as `Arc<serde_json::Map>`. Downstream
//! protocol types require an owned `serde_json::Map` inside `Value::Object`,
//! so every builder deep-clones the schema once per tool per call. Schema
//! maps are typically small; if this ever profiles hot, cache the
//! materialised `Value::Object` alongside the `Arc<Map>` on `ToolEntry`.

use openai_protocol::{
    common::{Function, Tool},
    responses::{
        generate_id, FunctionTool, McpAllowedTools, McpToolInfo, RequireApproval,
        RequireApprovalMode, ResponseOutputItem, ResponseTool,
    },
};
use serde_json::{json, Value};
use smg_mcp::{ApprovalMode, BuiltinToolType, McpToolSession};

/// Map a single `ResponseTool` variant to the `BuiltinToolType` the MCP
/// router uses for resolution. `None` for non-builtin tool kinds (e.g.
/// `Function`, `Mcp`, `Computer`). Centralizing here keeps the four
/// hosted families (web_search_preview / code_interpreter / file_search /
/// image_generation) in lockstep — every caller that previously matched
/// on `ResponseTool::*` independently was at risk of forgetting a variant
/// when a new builtin lands.
pub fn builtin_type_for_response_tool(tool: &ResponseTool) -> Option<BuiltinToolType> {
    match tool {
        ResponseTool::WebSearchPreview(_) => Some(BuiltinToolType::WebSearchPreview),
        ResponseTool::CodeInterpreter(_) => Some(BuiltinToolType::CodeInterpreter),
        ResponseTool::FileSearch(_) => Some(BuiltinToolType::FileSearch),
        ResponseTool::ImageGeneration(_) => Some(BuiltinToolType::ImageGeneration),
        _ => None,
    }
}

#[inline]
fn schema_to_value(schema: &serde_json::Map<String, Value>) -> Value {
    Value::Object(schema.clone())
}

/// `(exposed_name, description, &schema)` triples for every tool exposed by
/// `session`. Centralizes name-resolution (alias / disambiguation) so the
/// per-protocol builders below stay one-liners.
fn exposed_tool_fields<'a>(
    session: &'a McpToolSession<'a>,
) -> impl Iterator<Item = (&'a str, Option<&'a str>, &'a serde_json::Map<String, Value>)> + 'a {
    let exposed = session.exposed_name_by_qualified();
    session.mcp_tools().iter().map(move |entry| {
        let name = exposed
            .get(&entry.qualified_name)
            .map(String::as_str)
            .unwrap_or_else(|| entry.tool_name());
        let description = entry.tool.description.as_deref();
        (name, description, &*entry.tool.input_schema)
    })
}

/// Function-tool JSON payloads (`{"type": "function", ...}`) for upstream model calls.
pub fn function_tools_json(session: &McpToolSession<'_>) -> Vec<Value> {
    exposed_tool_fields(session)
        .map(|(name, description, parameters)| {
            json!({
                "type": "function",
                "name": name,
                "description": description,
                "parameters": schema_to_value(parameters)
            })
        })
        .collect()
}

/// Chat API function tools.
pub fn chat_function_tools(session: &McpToolSession<'_>) -> Vec<Tool> {
    exposed_tool_fields(session)
        .map(|(name, description, parameters)| Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: name.to_string(),
                description: description.map(str::to_string),
                parameters: schema_to_value(parameters),
                strict: None,
            },
        })
        .collect()
}

/// Responses API function tools. MCP tools surface to the model as
/// `{"type": "function", ...}` entries.
pub fn response_tools(session: &McpToolSession<'_>) -> Vec<ResponseTool> {
    exposed_tool_fields(session)
        .map(|(name, description, parameters)| {
            ResponseTool::Function(FunctionTool {
                function: Function {
                    name: name.to_string(),
                    description: description.map(str::to_string),
                    parameters: schema_to_value(parameters),
                    strict: None,
                },
            })
        })
        .collect()
}

/// `McpToolInfo` records used inside `mcp_list_tools` output items.
///
/// `annotations` is narrowed to `{"read_only": …}` to match OpenAI's
/// Responses API shape. The hint is read straight off the rmcp tool
/// (`tool.annotations.read_only_hint`) rather than the SMG
/// `ToolAnnotations` wrapper — the wrapper applies conservative policy
/// defaults (destructive=true on absent hint) that are intended for the
/// approval pipeline, not the wire surface, and reading them here would
/// surface the wrong `read_only` for tools the server didn't annotate.
pub fn build_mcp_tool_infos(entries: &[smg_mcp::ToolEntry]) -> Vec<McpToolInfo> {
    entries
        .iter()
        .map(|entry| {
            let read_only = entry
                .tool
                .annotations
                .as_ref()
                .and_then(|a| a.read_only_hint)
                .unwrap_or(false);
            McpToolInfo {
                name: entry.tool_name().to_string(),
                description: entry.tool.description.as_ref().map(|d| d.to_string()),
                input_schema: schema_to_value(&entry.tool.input_schema),
                annotations: Some(json!({ "read_only": read_only })),
            }
        })
        .collect()
}

/// Typed `mcp_list_tools` output item for one server's exposed tools.
pub fn mcp_list_tools_item(
    session: &McpToolSession<'_>,
    server_label: &str,
    server_key: &str,
) -> ResponseOutputItem {
    let tools = session.list_tools_for_server(server_key);
    ResponseOutputItem::McpListTools {
        id: generate_id("mcpl"),
        server_label: server_label.to_string(),
        tools: build_mcp_tool_infos(&tools),
        error: None,
    }
}

/// JSON form of `mcp_list_tools_item`. Falls back to a minimal stub if the
/// typed item fails to serialise (should be infallible for well-formed input).
pub fn mcp_list_tools_json(
    session: &McpToolSession<'_>,
    server_label: &str,
    server_key: &str,
) -> Value {
    serde_json::to_value(mcp_list_tools_item(session, server_label, server_key)).unwrap_or_else(
        |_| json!({ "type": "mcp_list_tools", "server_label": server_label, "tools": [] }),
    )
}

/// Inject only client-visible MCP metadata and call items into a response output array.
///
/// Visibility policy:
/// - Hide builtin `mcp_list_tools` (builtin tools surface under their own type).
/// - Hide internal non-builtin `mcp_list_tools`.
/// - Hide internal non-builtin passthrough `mcp_call`/`mcp_approval_request`.
/// - Keep builtin-routed call items visible.
/// - Keep user-defined function calls visible even on name collisions.
pub fn inject_client_visible_mcp_output_items(
    session: &McpToolSession<'_>,
    output: &mut Vec<ResponseOutputItem>,
    tool_call_items: Vec<ResponseOutputItem>,
    user_function_names: &std::collections::HashSet<String>,
) {
    let existing = std::mem::take(output);
    let servers = session.mcp_servers();
    output.reserve(servers.len() + tool_call_items.len() + existing.len());

    for binding in servers {
        if !session.is_internal_non_builtin_server_label(&binding.label) {
            output.push(mcp_list_tools_item(
                session,
                &binding.label,
                &binding.server_key,
            ));
        }
    }

    for item in tool_call_items {
        if is_client_visible_output_item(session, &item, user_function_names) {
            output.push(item);
        }
    }

    for item in existing {
        if is_client_visible_output_item(session, &item, user_function_names) {
            output.push(item);
        }
    }
}

/// Apply request-time approval configuration to exposed tools in `session`.
///
/// Parses `ResponseTool::Mcp::require_approval`/`allowed_tools` and forwards
/// the resolved approval mode + scoping to `session.set_approval_mode`.
///
/// `McpAllowedTools` projection (T11):
/// - `None` or `Filter { None, None }` → no name constraint (every binding
///   on the server inherits the explicit approval mode).
/// - `List(names)` / `Filter { tool_names: Some(v), .. }` → constrain by
///   explicit names.
/// - `Filter { tool_names: None, read_only: Some(_) }` → `None`. The
///   `readOnlyHint`-based filter is unimplemented; the safe direction for
///   approval scoping is "over-gate" — return `None` so the requested mode
///   applies to every binding.
pub fn configure_response_tools_approval(session: &mut McpToolSession<'_>, tools: &[ResponseTool]) {
    for tool in tools {
        let ResponseTool::Mcp(mcp_tool) = tool else {
            continue;
        };

        let approval_mode = match mcp_tool.require_approval.as_ref() {
            Some(RequireApproval::Mode(RequireApprovalMode::Always)) => ApprovalMode::Interactive,
            _ => ApprovalMode::PolicyOnly,
        };

        if approval_mode == ApprovalMode::PolicyOnly {
            continue;
        }

        let allowed_tool_names: Option<&[String]> =
            mcp_tool.allowed_tools.as_ref().and_then(|at| match at {
                McpAllowedTools::List(names) => Some(names.as_slice()),
                McpAllowedTools::Filter(filter) => filter.tool_names.as_deref(),
            });
        session.set_approval_mode(&mcp_tool.server_label, allowed_tool_names, approval_mode);
    }
}

/// True when a JSON tool entry should be hidden from client-facing responses.
/// Used by OpenAI non-streaming response normalization (tools handled as
/// `serde_json::Value` payloads).
pub fn should_hide_tool_json(
    session: &McpToolSession<'_>,
    tool: &Value,
    user_function_names: &std::collections::HashSet<String>,
) -> bool {
    match tool.get("type").and_then(|v| v.as_str()) {
        Some("function") => function_tool_name_json(tool)
            .is_some_and(|name| session.should_hide_function_call_like(name, user_function_names)),
        // MCP tool entries are keyed by server metadata; function-name
        // collision handling does not apply.
        Some("mcp") => tool
            .get("server_label")
            .and_then(|v| v.as_str())
            .is_some_and(|label| session.is_internal_non_builtin_server_label(label)),
        _ => false,
    }
}

/// True when a JSON output item should be hidden from client-facing responses.
/// Mirrors `is_client_visible_output_item` for the non-streaming path that
/// operates on raw JSON instead of typed `ResponseOutputItem`s.
pub fn should_hide_output_item_json(
    session: &McpToolSession<'_>,
    item: &Value,
    user_function_names: &std::collections::HashSet<String>,
) -> bool {
    match item.get("type").and_then(|v| v.as_str()) {
        Some("mcp_list_tools") => item
            .get("server_label")
            .and_then(|v| v.as_str())
            .is_some_and(|label| {
                session.is_builtin_server_label(label)
                    || session.is_internal_non_builtin_server_label(label)
            }),
        Some("mcp_call") | Some("mcp_approval_request") => {
            let matches_internal = item
                .get("server_label")
                .and_then(|v| v.as_str())
                .is_some_and(|label| session.is_internal_non_builtin_server_label(label));
            match item.get("name").and_then(|v| v.as_str()) {
                Some(name) => {
                    session.should_hide_mcp_call_like_by_server_flag(name, matches_internal)
                }
                None => matches_internal,
            }
        }
        Some("function_call") | Some("function_tool_call") => item
            .get("name")
            .and_then(|v| v.as_str())
            .is_some_and(|name| session.should_hide_function_call_like(name, user_function_names)),
        _ => false,
    }
}

fn function_tool_name_json(tool: &Value) -> Option<&str> {
    if tool.get("type").and_then(|v| v.as_str()) != Some("function") {
        return None;
    }
    tool.get("name").and_then(|v| v.as_str()).or_else(|| {
        tool.get("function")
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str())
    })
}

fn is_client_visible_output_item(
    session: &McpToolSession<'_>,
    item: &ResponseOutputItem,
    user_function_names: &std::collections::HashSet<String>,
) -> bool {
    match item {
        ResponseOutputItem::McpListTools { server_label, .. } => {
            !session.is_builtin_server_label(server_label)
                && !session.is_internal_non_builtin_server_label(server_label)
        }
        ResponseOutputItem::McpCall {
            server_label, name, ..
        }
        | ResponseOutputItem::McpApprovalRequest {
            server_label, name, ..
        } => !session.should_hide_mcp_call_like_by_label(name, server_label),
        ResponseOutputItem::FunctionToolCall { name, .. } => {
            !session.should_hide_function_call_like(name, user_function_names)
        }
        ResponseOutputItem::WebSearchCall { .. }
        | ResponseOutputItem::CodeInterpreterCall { .. }
        | ResponseOutputItem::FileSearchCall { .. }
        | ResponseOutputItem::ImageGenerationCall { .. }
        | ResponseOutputItem::ComputerCall { .. }
        | ResponseOutputItem::ComputerCallOutput { .. }
        | ResponseOutputItem::ShellCall { .. }
        | ResponseOutputItem::ShellCallOutput { .. }
        | ResponseOutputItem::ApplyPatchCall { .. }
        | ResponseOutputItem::ApplyPatchCallOutput { .. }
        | ResponseOutputItem::Message { .. }
        | ResponseOutputItem::Reasoning { .. }
        | ResponseOutputItem::Compaction { .. }
        | ResponseOutputItem::LocalShellCall { .. }
        | ResponseOutputItem::LocalShellCallOutput { .. } => true,
    }
}

#[cfg(test)]
mod tests {
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
}
