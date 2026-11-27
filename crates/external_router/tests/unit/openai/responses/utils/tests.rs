use openai_protocol::{
    common::{ConversationRef, Function},
    responses::{FunctionTool, McpTool, ResponseInput, ResponseTool, ResponsesRequest},
};
use serde_json::json;
use smg_mcp::{BuiltinToolType, Tool};

use super::{
    patch_response_with_request_metadata, restore_original_tools, rewrite_streaming_block,
};
use crate::openai::mcp::test_support;

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

/// The request the tests start from: one text input and, unless a test
/// adds them, no tools of its own.
fn hello_request() -> ResponsesRequest {
    ResponsesRequest {
        model: "gpt-5.4".to_string(),
        input: ResponseInput::Text("hello".to_string()),
        ..Default::default()
    }
}

#[tokio::test]
async fn restore_original_tools_strips_injected_internal_tool_when_request_had_no_tools() {
    let original_body = hello_request();
    let orchestrator =
        test_support::internal_orchestrator_serving(None, vec![test_tool("internal_search")]).await;
    let session = test_support::internal_session(&orchestrator);
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
    let original_body = hello_request();
    let orchestrator =
        test_support::internal_orchestrator_serving(None, vec![test_tool("internal_search")]).await;
    let session = test_support::internal_session(&orchestrator);
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
    let original_body = hello_request();
    let orchestrator =
        test_support::internal_orchestrator_serving(None, vec![test_tool("internal_search")]).await;
    let session = test_support::internal_session(&orchestrator);
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
    let original_body = hello_request();
    let orchestrator =
        test_support::internal_orchestrator_serving(None, vec![test_tool("internal_search")]).await;
    let session = test_support::internal_session(&orchestrator);
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
    let original_body = hello_request();
    let orchestrator =
        test_support::internal_orchestrator_serving(None, vec![test_tool("internal_search")]).await;
    let session = test_support::internal_session(&orchestrator);
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
    let original_body = hello_request();
    let orchestrator = test_support::internal_orchestrator(None).await;
    let session = test_support::internal_session(&orchestrator);
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
    let original_body = hello_request();
    let orchestrator = test_support::internal_orchestrator_serving(
        Some((BuiltinToolType::WebSearchPreview, "brave_web_search")),
        vec![test_tool("brave_web_search")],
    )
    .await;
    let session = test_support::internal_session(&orchestrator);
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
    let original_body = hello_request();
    let orchestrator = test_support::internal_orchestrator_serving(
        Some((BuiltinToolType::WebSearchPreview, "brave_web_search")),
        vec![test_tool("brave_web_search")],
    )
    .await;
    let session = test_support::internal_session(&orchestrator);
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
    let original_body = hello_request();
    let orchestrator = test_support::internal_orchestrator_serving(
        Some((BuiltinToolType::WebSearchPreview, "brave_web_search")),
        vec![
            test_tool("brave_web_search"),
            test_tool("internal_non_builtin_tool"),
        ],
    )
    .await;
    let session = test_support::internal_session(&orchestrator);
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
        tools: Some(vec![
            ResponseTool::Function(FunctionTool {
                function: Function {
                    name: "internal_search".to_string(),
                    description: Some("user function".to_string()),
                    parameters: Some(json!({
                        "type": "object",
                        "properties": { "query": { "type": "string" } },
                    })),
                    strict: None,
                    extra: Default::default(),
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
                allowed_callers: None,
                tunnel_id: None,
            }),
        ]),
        ..hello_request()
    };
    let orchestrator =
        test_support::internal_orchestrator_serving(None, vec![test_tool("internal_search")]).await;
    let session = test_support::internal_session(&orchestrator);
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
    let original_body = hello_request();
    let orchestrator = test_support::internal_orchestrator_serving(
        Some((BuiltinToolType::WebSearchPreview, "brave_web_search")),
        vec![test_tool("brave_web_search")],
    )
    .await;
    let session = test_support::internal_session(&orchestrator);
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
        conversation: Some(ConversationRef::Object {
            id: "conv_abc".to_string(),
        }),
        ..hello_request()
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
        conversation: Some(ConversationRef::Object {
            id: "conv_xyz".to_string(),
        }),
        ..hello_request()
    };

    // Minimal response.created SSE block with a `response` object to patch.
    let block = "event: response.created\n\
                     data: {\"type\":\"response.created\",\"response\":{}}\n";

    let rewritten = rewrite_streaming_block(block, &original_body, None).expect("block rewritten");

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
