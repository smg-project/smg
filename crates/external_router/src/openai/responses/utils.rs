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
use crate::{
    json_utils::is_missing_or_empty, mcp_utils::collect_user_function_names, openai_bridge,
};

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
            insert_optional_value(&mut m, "allowed_callers", mcp.allowed_callers.as_ref());
            insert_optional_value(&mut m, "tunnel_id", mcp.tunnel_id.as_ref());
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
        ResponseTool::ApplyPatch(_) => serde_json::to_value(tool).ok(),
        ResponseTool::Function(_) => None,
        // T5 schema-only: forced-cascade arm, no behavior.
        ResponseTool::LocalShell => serde_json::to_value(tool).ok(),
        ResponseTool::ToolSearch(_) => serde_json::to_value(tool).ok(),
        ResponseTool::ProgrammaticToolCalling => serde_json::to_value(tool).ok(),
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
#[path = "../../../tests/unit/openai/responses/utils/tests.rs"]
mod tests;
