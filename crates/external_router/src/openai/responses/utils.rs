//! Response patching and transformation utilities for OpenAI responses

use std::collections::HashSet;

use openai_protocol::{
    event_types::is_response_event,
    responses::{ResponseTool, ResponsesRequest},
};
use serde::Serialize;
use serde_json::{json, Map, Value};
use smg_mcp::McpToolSession;
use tracing::warn;

use super::common::parse_sse_block;
use crate::{mcp_utils::collect_user_function_names, openai_bridge};

/// Check if a JSON value is missing, null, or an empty string
fn is_missing_or_empty(value: Option<&Value>) -> bool {
    match value {
        None => true,
        Some(v) => v.is_null() || v.as_str().is_some_and(|s| s.is_empty()),
    }
}

/// Insert a string value into a JSON object if the condition is met
fn insert_if<F>(obj: &mut Map<String, Value>, key: &str, value: &str, condition: F)
where
    F: FnOnce(&Map<String, Value>) -> bool,
{
    if condition(obj) {
        obj.insert(key.to_string(), Value::String(value.to_string()));
    }
}

/// Patch response JSON with metadata from original request
///
/// The upstream response may be missing fields that were in the original request.
/// This function ensures these fields are preserved in the final response:
/// - `previous_response_id` - conversation threading
/// - `instructions` - system instructions
/// - `metadata` - user-provided metadata
/// - `store` - whether to persist the response
/// - `model` - model identifier
/// - `safety_identifier` - user identifier for safety
pub(super) fn patch_response_with_request_metadata(
    response_json: &mut Value,
    original_body: &ResponsesRequest,
    original_previous_response_id: Option<&str>,
) {
    let Some(obj) = response_json.as_object_mut() else {
        return;
    };

    // Set previous_response_id if missing/empty
    if let Some(prev_id) = original_previous_response_id {
        insert_if(obj, "previous_response_id", prev_id, |o| {
            is_missing_or_empty(o.get("previous_response_id"))
        });
    }

    // Set instructions if missing/null
    if let Some(instructions) = &original_body.instructions {
        insert_if(obj, "instructions", instructions, |o| {
            is_missing_or_empty(o.get("instructions"))
        });
    }

    // Set metadata if missing/null
    if is_missing_or_empty(obj.get("metadata")) {
        if let Some(metadata) = &original_body.metadata {
            let metadata_map: Map<String, Value> = metadata
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            obj.insert("metadata".to_string(), Value::Object(metadata_map));
        }
    }

    // Always set store
    obj.insert(
        "store".to_string(),
        Value::Bool(original_body.store.unwrap_or(true)),
    );

    // Set model if missing/empty
    insert_if(obj, "model", &original_body.model, |o| {
        is_missing_or_empty(o.get("model"))
    });

    // Set safety_identifier if null (but key exists)
    if let Some(user) = &original_body.user {
        if obj
            .get("safety_identifier")
            .is_some_and(|v: &Value| v.is_null())
        {
            obj.insert("safety_identifier".to_string(), Value::String(user.clone()));
        }
    }

    // Attach conversation id for client response. Unwrap via `as_id()` so we
    // emit a flat `{ "id": "conv_..." }` object regardless of whether the
    // original request used the string or object variant of `ConversationRef`.
    if let Some(conv_ref) = &original_body.conversation {
        obj.insert(
            "conversation".to_string(),
            json!({ "id": conv_ref.as_id() }),
        );
    }
}

/// Rebuild SSE block with new data payload
fn rebuild_sse_block(block: &str, new_payload: &str) -> String {
    let mut rebuilt_lines = Vec::new();
    let mut data_written = false;

    for line in block.lines() {
        if line.starts_with("data:") {
            if !data_written {
                rebuilt_lines.push(format!("data: {new_payload}"));
                data_written = true;
            }
        } else {
            rebuilt_lines.push(line.to_string());
        }
    }

    if !data_written {
        rebuilt_lines.push(format!("data: {new_payload}"));
    }

    rebuilt_lines.join("\n")
}

/// Rewrite streaming SSE block to include metadata from original request
pub(super) fn rewrite_streaming_block(
    block: &str,
    original_body: &ResponsesRequest,
    original_previous_response_id: Option<&str>,
) -> Option<String> {
    let trimmed = block.trim();
    if trimmed.is_empty() {
        return None;
    }

    let (_, data) = parse_sse_block(trimmed);
    if data.is_empty() {
        return None;
    }
    let mut parsed: Value = serde_json::from_str(&data)
        .map_err(|e| warn!("Failed to parse streaming JSON payload: {}", e))
        .ok()?;

    let event_type = parsed
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    if !is_response_event(event_type) {
        return None;
    }

    let response_obj = parsed.get_mut("response").and_then(|v| v.as_object_mut())?;
    let mut changed = false;

    // Update store value if different
    let desired_store = Value::Bool(original_body.store.unwrap_or(true));
    if response_obj.get("store") != Some(&desired_store) {
        response_obj.insert("store".to_string(), desired_store);
        changed = true;
    }

    // Set previous_response_id if missing/empty
    if let Some(prev_id) = original_previous_response_id {
        if is_missing_or_empty(response_obj.get("previous_response_id")) {
            response_obj.insert("previous_response_id".to_string(), json!(prev_id));
            changed = true;
        }
    }

    // Attach conversation id. Unwrap via `as_id()` so the SSE payload emits a
    // flat `{ "id": "conv_..." }` even when the original request used the
    // object form of `ConversationRef`.
    if let Some(conv_ref) = &original_body.conversation {
        response_obj.insert(
            "conversation".to_string(),
            json!({ "id": conv_ref.as_id() }),
        );
        changed = true;
    }

    if !changed {
        return None;
    }

    let new_payload = serde_json::to_string(&parsed)
        .map_err(|e| warn!("Failed to serialize modified streaming payload: {}", e))
        .ok()?;

    Some(rebuild_sse_block(trimmed, &new_payload))
}

/// Helper to insert an optional serializable field into a JSON map.
pub(super) fn insert_optional_value<T: Serialize>(
    map: &mut Map<String, Value>,
    key: &str,
    value: Option<&T>,
) {
    if let Some(v) = value {
        match serde_json::to_value(v) {
            Ok(val) => {
                map.insert(key.to_string(), val);
            }
            Err(e) => {
                warn!(field = key, error = %e, "Failed to serialize optional field");
            }
        }
    }
}

/// Convert a single ResponseTool back to its original JSON representation.
///
/// Handles MCP tools (with server metadata), web_search, web_search_preview,
/// file_search, and code_interpreter. Returns None for function tools and other
/// types that don't need restoration.
pub(super) fn response_tool_to_value(tool: &ResponseTool) -> Option<Value> {
    match tool {
        ResponseTool::Mcp(mcp) => {
            let mut m = Map::new();
            m.insert("type".to_string(), json!("mcp"));
            m.insert("server_label".to_string(), json!(&mcp.server_label));
            insert_optional_value(&mut m, "server_url", mcp.server_url.as_ref());
            insert_optional_value(
                &mut m,
                "server_description",
                mcp.server_description.as_ref(),
            );
            insert_optional_value(&mut m, "require_approval", mcp.require_approval.as_ref());
            // T11: `allowed_tools` is now an untagged union (`List(Vec<String>)`
            // or `Filter { read_only?, tool_names? }`). Delegate to serde so both
            // wire shapes serialize identically to the input payload.
            insert_optional_value(&mut m, "allowed_tools", mcp.allowed_tools.as_ref());
            // T11: surface the new `connector_id` and `defer_loading` fields so
            // the echoed tools[] payload round-trips losslessly for clients that
            // rely on connector-based MCP setups (server_url XOR connector_id)
            // or the deferred-loading hint.
            insert_optional_value(&mut m, "connector_id", mcp.connector_id.as_ref());
            insert_optional_value(&mut m, "defer_loading", mcp.defer_loading.as_ref());
            Some(Value::Object(m))
        }
        ResponseTool::WebSearchPreview(_) => serde_json::to_value(tool).ok(),
        ResponseTool::WebSearch(_) => serde_json::to_value(tool).ok(),
        ResponseTool::CodeInterpreter(_) => serde_json::to_value(tool).ok(),
        ResponseTool::FileSearch(_) => serde_json::to_value(tool).ok(),
        ResponseTool::ImageGeneration(_) => serde_json::to_value(tool).ok(),
        ResponseTool::Computer | ResponseTool::ComputerUsePreview(_) => {
            serde_json::to_value(tool).ok()
        }
        ResponseTool::Custom(_) => serde_json::to_value(tool).ok(),
        // Namespace groups Function/Custom elements; serialize through serde so
        // the full {type, name, description, tools} payload round-trips back
        // to the client unchanged (T9).
        ResponseTool::Namespace(_) => serde_json::to_value(tool).ok(),
        ResponseTool::Shell(_) => serde_json::to_value(tool).ok(),
        ResponseTool::ApplyPatch => serde_json::to_value(tool).ok(),
        ResponseTool::Function(_) => None,
        // T5 schema-only: forced-cascade arm, no behavior.
        ResponseTool::LocalShell => serde_json::to_value(tool).ok(),
    }
}

/// Restore original tools (MCP and builtin) in response for client.
///
/// The model receives function tools, but the response should mirror the original
/// request's tool format (MCP tools with server_url, builtin tools like web_search_preview).
/// Also scrubs any internal-MCP artifacts that leaked into the upstream payload.
pub(super) fn restore_original_tools(
    resp: &mut Value,
    original_body: &ResponsesRequest,
    session: Option<&McpToolSession<'_>>,
) {
    let user_function_names = collect_user_function_names(original_body);
    strip_internal_mcp_output_items(resp, session, &user_function_names);
    strip_internal_mcp_tools(resp, session, &user_function_names);
    restore_client_tool_view(resp, original_body, session, &user_function_names);
}

fn restore_client_tool_view(
    resp: &mut Value,
    original_body: &ResponsesRequest,
    session: Option<&McpToolSession<'_>>,
    user_function_names: &HashSet<String>,
) {
    let Some(original_tools) = original_body.tools.as_ref() else {
        return;
    };

    // Pure function-tool requests: keep upstream tools/tool_choice payload as-is.
    if original_tools
        .iter()
        .all(|tool| matches!(tool, ResponseTool::Function(_)))
    {
        return;
    }

    let mut restored_tools: Vec<Value> = Vec::with_capacity(original_tools.len());
    for original_tool in original_tools {
        match original_tool {
            ResponseTool::Function(_) => {
                if let Ok(value) = serde_json::to_value(original_tool) {
                    restored_tools.push(value);
                }
            }
            _ => {
                if let Some(value) = response_tool_to_value(original_tool) {
                    let should_hide = session.is_some_and(|s| {
                        openai_bridge::should_hide_tool_json(s, &value, user_function_names)
                    });
                    if !should_hide {
                        restored_tools.push(value);
                    }
                }
            }
        }
    }

    let Some(obj) = resp.as_object_mut() else {
        return;
    };
    if restored_tools.is_empty() {
        obj.remove("tools");
        obj.remove("tool_choice");
        return;
    }

    obj.insert("tools".to_string(), Value::Array(restored_tools));

    if let Some(tool_choice) = obj.get("tool_choice") {
        let selected_name = tool_choice
            .as_object()
            .and_then(|choice| {
                choice
                    .get("type")
                    .and_then(Value::as_str)
                    .zip(choice.get("name").and_then(Value::as_str))
            })
            .and_then(|(choice_type, name)| (choice_type == "function").then_some(name));
        if let Some(name) = selected_name {
            let has_selected_tool =
                obj.get("tools")
                    .and_then(Value::as_array)
                    .is_some_and(|tools| {
                        tools
                            .iter()
                            .filter_map(function_tool_name)
                            .any(|tool_name| tool_name == name)
                    });
            if !has_selected_tool {
                obj.insert("tool_choice".to_string(), json!("auto"));
            }
        }
    }

    obj.entry("tool_choice").or_insert(json!("auto"));
}

fn strip_internal_mcp_tools(
    resp: &mut Value,
    session: Option<&McpToolSession<'_>>,
    user_function_names: &HashSet<String>,
) {
    let Some(obj) = resp.as_object_mut() else {
        return;
    };

    let Some(tools) = obj.get_mut("tools").and_then(|value| value.as_array_mut()) else {
        return;
    };

    tools.retain(|tool| {
        !session.is_some_and(|s| openai_bridge::should_hide_tool_json(s, tool, user_function_names))
    });
}

fn strip_internal_mcp_output_items(
    resp: &mut Value,
    session: Option<&McpToolSession<'_>>,
    user_function_names: &HashSet<String>,
) {
    let Some(obj) = resp.as_object_mut() else {
        return;
    };

    let Some(output) = obj.get_mut("output").and_then(|value| value.as_array_mut()) else {
        return;
    };

    output.retain(|item| {
        !session.is_some_and(|s| {
            openai_bridge::should_hide_output_item_json(s, item, user_function_names)
        })
    });
}

pub(crate) fn function_tool_name(tool: &Value) -> Option<&str> {
    let tool_type = tool.get("type").and_then(|value| value.as_str());
    if tool_type != Some("function") {
        return None;
    }
    tool.get("name")
        .and_then(|value| value.as_str())
        .or_else(|| {
            tool.get("function")
                .and_then(|function| function.get("name"))
                .and_then(|value| value.as_str())
        })
}

#[cfg(test)]
mod tests {
    use openai_protocol::{
        common::{ConversationRef, Function},
        responses::{FunctionTool, McpTool, ResponseInput, ResponseTool, ResponsesRequest},
    };
    use serde_json::json;
    use smg_mcp::{
        BuiltinToolType, McpConfig, McpOrchestrator, McpServerBinding, McpServerConfig,
        McpToolSession, McpTransport, Tool, ToolEntry,
    };

    use super::{
        patch_response_with_request_metadata, restore_original_tools, rewrite_streaming_block,
    };

    fn test_tool(name: &str) -> Tool {
        let mut schema = serde_json::Map::new();
        schema.insert("type".to_string(), json!("object"));
        schema.insert(
            "properties".to_string(),
            json!({
                "query": { "type": "string" }
            }),
        );

        Tool::new(name.to_string(), "internal", schema)
    }

    #[tokio::test]
    async fn restore_original_tools_strips_injected_internal_tool_when_request_had_no_tools() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("internal_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "tools": [{
                "type": "function",
                "name": "internal_search",
                "description": "internal",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": ["query"]
                }
            }],
            "tool_choice": "auto"
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(response["tools"], serde_json::json!([]));
        assert_eq!(response["tool_choice"], "auto");
    }

    #[tokio::test]
    async fn restore_original_tools_strips_injected_internal_nested_function_tool_shape() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("internal_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "internal_search",
                    "description": "internal",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": ["query"]
                    }
                }
            }]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(response["tools"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn restore_original_tools_strips_internal_mcp_output_items() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("internal_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "output": [
                {
                    "type": "mcp_list_tools",
                    "server_label": "internal-label",
                    "tools": []
                },
                {
                    "type": "mcp_call",
                    "name": "internal_search",
                    "server_label": "internal-label"
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(
            response["output"],
            serde_json::json!([{
                "type": "message",
                "content": [{"type": "output_text", "text": "visible"}]
            }])
        );
    }

    #[tokio::test]
    async fn restore_original_tools_strips_internal_function_tool_call_items() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("internal_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "output": [
                {
                    "type": "function_tool_call",
                    "call_id": "call_123",
                    "name": "internal_search",
                    "arguments": "{\"query\":\"secret\"}"
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(
            response["output"],
            serde_json::json!([{
                "type": "message",
                "content": [{"type": "output_text", "text": "visible"}]
            }])
        );
    }

    #[tokio::test]
    async fn restore_original_tools_strips_internal_mcp_approval_request_items() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("internal_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "output": [
                {
                    "type": "mcp_approval_request",
                    "name": "internal_search",
                    "server_label": "internal-label"
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(
            response["output"],
            serde_json::json!([{
                "type": "message",
                "content": [{"type": "output_text", "text": "visible"}]
            }])
        );
    }

    #[tokio::test]
    async fn restore_original_tools_strips_internal_mcp_items_by_server_label_when_name_unknown() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "output": [
                {
                    "type": "mcp_call",
                    "name": "unknown_tool_name",
                    "server_label": "internal-label"
                },
                {
                    "type": "mcp_approval_request",
                    "name": "unknown_tool_name",
                    "server_label": "internal-label"
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(
            response["output"],
            serde_json::json!([{
                "type": "message",
                "content": [{"type": "output_text", "text": "visible"}]
            }])
        );
    }

    #[tokio::test]
    async fn restore_original_tools_keeps_builtin_output_items_visible() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("brave_web_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "output": [
                {
                    "type": "web_search_call",
                    "id": "ws_call_123",
                    "server_label": "internal-label",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "private query",
                        "queries": ["private query"],
                        "sources": []
                    }
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(
            response["output"],
            serde_json::json!([
                {
                    "type": "web_search_call",
                    "id": "ws_call_123",
                    "server_label": "internal-label",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "query": "private query",
                        "queries": ["private query"],
                        "sources": []
                    }
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ])
        );
    }

    #[tokio::test]
    async fn restore_original_tools_keeps_builtin_passthrough_mcp_call_visible() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("brave_web_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "output": [
                {
                    "type": "mcp_call",
                    "name": "brave_web_search",
                    "server_label": "internal-label",
                    "output": "{\"results\":[]}"
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(
            response["output"],
            serde_json::json!([
                {
                    "type": "mcp_call",
                    "name": "brave_web_search",
                    "server_label": "internal-label",
                    "output": "{\"results\":[]}"
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ])
        );
    }

    #[tokio::test]
    async fn restore_original_tools_keeps_builtin_passthrough_mcp_call_visible_with_mixed_tools() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("brave_web_search"),
            ));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("internal_non_builtin_tool"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "output": [
                {
                    "type": "mcp_call",
                    "name": "brave_web_search",
                    "server_label": "internal-label",
                    "output": "{\"results\":[]}"
                },
                {
                    "type": "mcp_call",
                    "name": "internal_non_builtin_tool",
                    "server_label": "internal-label",
                    "output": "{\"private\":true}"
                }
            ]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(
            response["output"],
            serde_json::json!([{
                "type": "mcp_call",
                "name": "brave_web_search",
                "server_label": "internal-label",
                "output": "{\"results\":[]}"
            }])
        );
    }

    #[tokio::test]
    async fn restore_original_tools_keeps_user_function_call_on_name_collision() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            tools: Some(vec![
                ResponseTool::Function(FunctionTool {
                    function: Function {
                        name: "internal_search".to_string(),
                        description: Some("user function".to_string()),
                        parameters: json!({
                            "type": "object",
                            "properties": { "query": { "type": "string" } },
                        }),
                        strict: None,
                    },
                }),
                ResponseTool::Mcp(McpTool {
                    server_url: Some("http://localhost:3000/sse".to_string()),
                    authorization: None,
                    headers: None,
                    server_label: "internal-label".to_string(),
                    server_description: None,
                    require_approval: None,
                    allowed_tools: Some(openai_protocol::responses::McpAllowedTools::List(vec![
                        "internal_search".to_string(),
                    ])),
                    connector_id: None,
                    defer_loading: None,
                }),
            ]),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("internal_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "output": [
                {
                    "type": "function_call",
                    "name": "internal_search",
                    "arguments": "{\"query\":\"client\"}"
                },
                {
                    "type": "mcp_call",
                    "name": "internal_search",
                    "server_label": "internal-label"
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        assert_eq!(
            response["output"],
            serde_json::json!([
                {
                    "type": "function_call",
                    "name": "internal_search",
                    "arguments": "{\"query\":\"client\"}"
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ])
        );
    }

    #[tokio::test]
    async fn restore_original_tools_hides_builtin_list_tools_keeps_passthrough_call() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            ..Default::default()
        };
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("brave_web_search"),
            ));
        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );
        let mut response = serde_json::json!({
            "output": [
                {
                    "type": "mcp_list_tools",
                    "server_label": "internal-label",
                    "tools": []
                },
                {
                    "type": "mcp_call",
                    "name": "brave_web_search",
                    "server_label": "internal-label",
                    "output": "{\"results\":[]}"
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ]
        });

        restore_original_tools(&mut response, &original_body, Some(&session));

        // Builtin mcp_list_tools is hidden — clients don't see the underlying
        // MCP server for builtin-routed tools like web_search_preview.
        // Builtin passthrough mcp_call remains visible.
        assert_eq!(
            response["output"],
            serde_json::json!([
                {
                    "type": "mcp_call",
                    "name": "brave_web_search",
                    "server_label": "internal-label",
                    "output": "{\"results\":[]}"
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "visible"}]
                }
            ])
        );
    }

    /// Regression: when the client sends `conversation` as an object
    /// (`ConversationRef::Object { id }`), the non-streaming response patcher
    /// must emit a flat `{ "conversation": { "id": "conv_..." } }`. Before the
    /// `as_id()` fix, the old code interpolated the whole `ConversationRef`,
    /// serializing the `Object` variant as `{"id": "conv_..."}` and producing
    /// a nested `{"conversation": {"id": {"id": "conv_..."}}}`.
    #[test]
    fn patch_response_flattens_object_conversation_ref() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            conversation: Some(ConversationRef::Object {
                id: "conv_abc".to_string(),
            }),
            ..Default::default()
        };
        let mut response = json!({});
        patch_response_with_request_metadata(&mut response, &original_body, None);

        assert_eq!(
            response.get("conversation"),
            Some(&json!({ "id": "conv_abc" })),
            "object-form conversation must emit flat {{id}}, not nested {{id:{{id}}}}"
        );
        // Negative guard: the inner value must be a string, not an object.
        let inner = response
            .get("conversation")
            .and_then(|v| v.get("id"))
            .expect("conversation.id present");
        assert!(
            inner.is_string(),
            "conversation.id must be a plain string, got {inner:?}"
        );
    }

    /// Regression: same invariant for the streaming SSE rewriter.
    #[test]
    fn rewrite_streaming_block_flattens_object_conversation_ref() {
        let original_body = ResponsesRequest {
            model: "gpt-5.4".to_string(),
            input: ResponseInput::Text("hello".to_string()),
            conversation: Some(ConversationRef::Object {
                id: "conv_xyz".to_string(),
            }),
            ..Default::default()
        };

        // Minimal response.created SSE block with a `response` object to patch.
        let block = "event: response.created\n\
                     data: {\"type\":\"response.created\",\"response\":{}}\n";

        let rewritten =
            rewrite_streaming_block(block, &original_body, None).expect("block rewritten");

        // Extract the `data:` line payload and re-parse it.
        let data_line = rewritten
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("data line");
        let payload: serde_json::Value =
            serde_json::from_str(data_line.trim_start_matches("data: ")).expect("json");
        let conv = payload
            .get("response")
            .and_then(|r| r.get("conversation"))
            .expect("conversation attached");

        assert_eq!(
            conv,
            &json!({ "id": "conv_xyz" }),
            "streaming payload must flatten object-form conversation"
        );
        assert!(
            conv.get("id").unwrap().is_string(),
            "conversation.id must be a plain string, got {conv:?}"
        );
    }
}
