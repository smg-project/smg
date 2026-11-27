use axum::http::StatusCode;
use openai_protocol::{
    responses::{
        ResponseInput, ResponseInputOutputItem, ResponsesRequest, ResponsesToolChoice,
        ToolChoiceOptions,
    },
    validated::Normalizable,
};
use serde_json::{json, Value};
use validator::Validate;

use super::*;
use crate::routers::grpc::common::responses::utils::extract_tools_from_response_tools;

fn parse_request(body: Value) -> ResponsesRequest {
    serde_json::from_value(body).expect("request should deserialize")
}

#[test]
fn lowers_codex_subagent_request_with_additional_tools_and_agent_message() {
    let mut req = parse_request(json!({
        "model": "test-model",
        "stream": true,
        "tools": [],
        "input": [
            {
                "type": "additional_tools",
                "id": "at_1",
                "role": "developer",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "functions",
                        "description": "",
                        "tools": [
                            {
                                "type": "function",
                                "name": "exec_command",
                                "description": "Run shell command",
                                "parameters": {"type": "object", "properties": {}},
                                "strict": false
                            },
                            {
                                "type": "custom",
                                "name": "apply_patch",
                                "description": "Patch files"
                            }
                        ]
                    },
                    {
                        "type": "namespace",
                        "name": "clock",
                        "description": "Clock tools",
                        "tools": [{
                            "type": "function",
                            "name": "now",
                            "description": "Current time",
                            "parameters": {"type": "object", "properties": {}}
                        }]
                    },
                    {
                        "type": "namespace",
                        "name": "collaboration",
                        "description": "Multi-agent tools",
                        "tools": [{
                            "type": "function",
                            "name": "send_message",
                            "description": "Send message",
                            "parameters": {"type": "object", "properties": {}}
                        }]
                    }
                ]
            },
            {
                "type": "message",
                "id": "msg_sys",
                "role": "developer",
                "content": [{"type": "input_text", "text": "Base instructions"}]
            },
            {
                "type": "agent_message",
                "id": "am_1",
                "author": "/root",
                "recipient": "/root/worker_1",
                "content": [{"type": "input_text", "text": "Reply with OK-123."}]
            }
        ]
    }));

    // ValidatedJson runs normalize() before routing; empty top-level tools: []
    // must default to auto (not none) when inline additional_tools exist.
    req.normalize();
    assert!(matches!(
        req.tool_choice,
        Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto))
    ));

    // Custom tool_choice targeting an unwrapped default-namespace tool must
    // strip the `functions.` qualifier to match the bare offered tool name.
    let mut custom_choice_req = req.clone();
    custom_choice_req.tool_choice = Some(
        serde_json::from_value(json!({"type": "custom", "name": "functions.apply_patch"})).unwrap(),
    );
    lower_responses_request_for_local_backend(&mut custom_choice_req)
        .expect("lowering succeeds with custom tool_choice");
    assert_eq!(
        serde_json::to_value(&custom_choice_req.tool_choice).unwrap(),
        json!({"type": "custom", "name": "apply_patch"})
    );

    // Literal top-level custom names with a `functions.` prefix must stay
    // intact when an unrelated inline `functions` namespace is unwrapped.
    let mut literal_custom_req = parse_request(json!({
        "model": "test-model",
        "tools": [{"type": "custom", "name": "functions.apply_patch"}],
        "tool_choice": {"type": "custom", "name": "functions.apply_patch"},
        "input": [
            {
                "type": "additional_tools",
                "tools": [{
                    "type": "namespace",
                    "name": "functions",
                    "description": "",
                    "tools": [{"type": "function", "name": "exec_command", "parameters": {"type": "object"}}]
                }]
            },
            {"role": "user", "content": "patch it"}
        ]
    }));
    lower_responses_request_for_local_backend(&mut literal_custom_req)
        .expect("lowering preserves literal dotted custom tool");
    assert_eq!(
        serde_json::to_value(&literal_custom_req.tool_choice).unwrap(),
        json!({"type": "custom", "name": "functions.apply_patch"})
    );

    // An unwrapped `functions.apply_patch` member must not block rewriting
    // `tool_choice: functions.apply_patch` to the unwrapped `apply_patch`.
    let mut collision_req = parse_request(json!({
        "model": "test-model",
        "tool_choice": {"type": "custom", "name": "functions.apply_patch"},
        "input": [
            {
                "type": "additional_tools",
                "tools": [{
                    "type": "namespace",
                    "name": "functions",
                    "description": "",
                    "tools": [
                        {"type": "custom", "name": "apply_patch"},
                        {"type": "custom", "name": "functions.apply_patch"}
                    ]
                }]
            },
            {"role": "user", "content": "patch it"}
        ]
    }));
    lower_responses_request_for_local_backend(&mut collision_req)
        .expect("lowering rewrites choice to unwrapped apply_patch");
    assert_eq!(
        serde_json::to_value(&collision_req.tool_choice).unwrap(),
        json!({"type": "custom", "name": "apply_patch"})
    );

    lower_responses_request_for_local_backend(&mut req).expect("lowering succeeds");

    let wire = serde_json::to_value(&req).unwrap();
    let tools = wire["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 4);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["name"], "exec_command");
    assert_eq!(tools[1]["type"], "custom");
    assert_eq!(tools[1]["name"], "apply_patch");
    assert_eq!(tools[2]["type"], "namespace");
    assert_eq!(tools[2]["name"], "clock");
    assert_eq!(tools[3]["type"], "namespace");
    assert_eq!(tools[3]["name"], "collaboration");

    let input = wire["input"].as_array().expect("input array");
    assert_eq!(input.len(), 2);
    assert_eq!(input[0]["type"], "message");
    assert_eq!(input[0]["role"], "developer");

    assert_eq!(input[1]["type"], "message");
    assert_eq!(input[1]["id"], "am_1");
    assert_eq!(input[1]["role"], "user");
    let text = input[1]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("Sender: /root"), "{text}");
    assert!(text.contains("Recipient: /root/worker_1"), "{text}");
    assert!(text.ends_with("Reply with OK-123."), "{text}");
}

#[test]
fn inline_tool_declarations_alone_leave_no_local_model_input() {
    let mut req = parse_request(json!({
        "model":"test-model",
        "input":[{"type":"additional_tools", "role":"developer",
            "tools":[{"type":"function","name":"exec_command"}]}]
    }));
    let response = lower_responses_request_for_local_backend(&mut req).unwrap_err();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn rejects_encrypted_agent_message_for_local_backends() {
    let mut req = parse_request(json!({
        "model": "test-model",
        "input": [{
            "type": "agent_message",
            "id": "am_enc",
            "author": "/root",
            "recipient": "/root/worker_1",
            "content": [
                {"type": "input_text", "text": "Header"},
                {"type": "encrypted_content", "encrypted_content": "gAAAAB-secret"}
            ]
        }]
    }));

    let response = lower_responses_request_for_local_backend(&mut req)
        .expect_err("encrypted_content must be refused");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn coalesces_duplicate_tools_and_keeps_explicit_functions_namespace_consistent() {
    // 1. Repeated inline declarations deduplicate by name (latest wins).
    let mut dedup = parse_request(json!({
        "model": "test-model",
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [{
                    "type": "namespace",
                    "name": "functions",
                    "description": "",
                    "tools": [{
                        "type": "function",
                        "name": "exec_command",
                        "description": "v1",
                        "parameters": {"type": "object"}
                    }]
                }, {
                    "type": "namespace",
                    "name": "collaboration",
                    "description": "Collab",
                    "tools": [{
                        "type": "function",
                        "name": "send_message",
                        "description": "v1",
                        "parameters": {"type": "object"}
                    }]
                }]
            },
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [{
                    "type": "namespace",
                    "name": "functions",
                    "description": "",
                    "tools": [{
                        "type": "function",
                        "name": "exec_command",
                        "description": "v2",
                        "parameters": {"type": "object"}
                    }]
                }, {
                    "type": "namespace",
                    "name": "collaboration",
                    "description": "Collab",
                    "tools": [{
                        "type": "function",
                        "name": "send_message",
                        "description": "v2",
                        "parameters": {"type": "object"}
                    }]
                }]
            },
            {"role": "user", "content": "run"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec_command",
                "namespace": "functions",
                "arguments": "{}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok"
            }
        ]
    }));
    lower_responses_request_for_local_backend(&mut dedup).unwrap();
    let wire = serde_json::to_value(&dedup).unwrap();
    let tools = wire["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0]["name"], "exec_command");
    assert_eq!(tools[0]["description"], "v2");
    assert_eq!(tools[1]["tools"].as_array().unwrap().len(), 1);
    assert_eq!(tools[1]["tools"][0]["description"], "v2");
    let ResponseInput::Items(items) = &dedup.input else {
        panic!("expected items");
    };
    assert!(matches!(
        &items[1],
        ResponseInputOutputItem::FunctionToolCall { namespace: None, name, .. } if name == "exec_command"
    ));

    // 2. When top-level tools already declare a `functions` namespace, inline
    // `functions` members coalesce into it so `tool_choice` targeting
    // `functions.<new>` remains valid.
    let mut mixed = parse_request(json!({
        "model": "test-model",
        "tools": [{
            "type": "namespace",
            "name": "functions",
            "description": "Explicit functions namespace",
            "tools": [{
                "type": "function",
                "name": "write_stdin",
                "parameters": {"type": "object"}
            }]
        }],
        "tool_choice": {"type": "function", "name": "exec_command", "namespace": "functions"},
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [{
                    "type": "namespace",
                    "name": "functions",
                    "description": "",
                    "tools": [{
                        "type": "function",
                        "name": "exec_command",
                        "parameters": {"type": "object"}
                    }, {"type": "custom", "name": "apply_patch"}]
                }]
            },
            {"role": "user", "content": "Run git status"}
        ]
    }));
    for (choice, expected) in [
        (
            json!({"type": "function", "name": "exec_command"}),
            "functions.exec_command",
        ),
        (
            json!({"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "exec_command"}]}),
            "functions.exec_command",
        ),
        (
            json!({"type": "custom", "name": "apply_patch"}),
            "functions.apply_patch",
        ),
    ] {
        let mut request = mixed.clone();
        request.tool_choice = Some(serde_json::from_value(choice).unwrap());
        request.validate().unwrap();
        lower_responses_request_for_local_backend(&mut request).unwrap();
        let tools = extract_tools_from_response_tools(request.tools.as_deref());
        let offered = request
            .tool_choice
            .unwrap()
            .to_chat_tool_choice()
            .narrow_tools(&tools)
            .unwrap();
        assert_eq!(offered.len(), 1);
        assert_eq!(offered[0].function.name, expected);
    }
    lower_responses_request_for_local_backend(&mut mixed).unwrap();
    let wire = serde_json::to_value(&mixed).unwrap();
    assert_eq!(wire["tools"].as_array().unwrap().len(), 1);
    assert_eq!(wire["tools"][0]["type"], "namespace");
    assert_eq!(wire["tools"][0]["name"], "functions");
    let members = wire["tools"][0]["tools"].as_array().unwrap();
    assert_eq!(members.len(), 3);
    assert_eq!(members[0]["name"], "write_stdin");
    assert_eq!(members[1]["name"], "exec_command");
    assert_eq!(wire["tool_choice"]["namespace"], "functions");
    assert_eq!(wire["tool_choice"]["name"], "exec_command");
}

#[test]
fn lowers_missing_reasoning_id_for_local_backends() {
    let mut req = parse_request(json!({
        "model": "test-model",
        "input": [
            {"role": "user", "content": "check status"},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking"}],
                "content": null
            },
            {
                "type": "reasoning",
                "id": null,
                "summary": []
            },
            {
                "type": "reasoning",
                "id": "rs_existing",
                "summary": [],
                "encrypted_content": "opaque"
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec_command",
                "arguments": "{}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok"
            }
        ]
    }));

    lower_responses_request_for_local_backend(&mut req)
        .expect("missing/null reasoning ids should be lowered");

    let wire = serde_json::to_value(&req).unwrap();
    let items = wire["input"].as_array().unwrap();
    let generated_1 = items[1]["id"].as_str().expect("missing id synthesized");
    let generated_2 = items[2]["id"].as_str().expect("null id synthesized");
    assert!(generated_1.starts_with("rs_"), "{generated_1}");
    assert!(generated_2.starts_with("rs_"), "{generated_2}");
    assert_ne!(generated_1, generated_2);
    assert!(items[1].get("content").is_none());
    assert_eq!(items[3]["id"], "rs_existing");
    assert_eq!(items[3]["encrypted_content"], "opaque");
}

#[test]
fn lowers_call_id_less_function_call_output_for_local_backends() {
    let mut req = parse_request(json!({
        "model": "test-model",
        "input": [
            {"role": "user", "content": "open the page"},
            {
                "type": "function_call_output",
                "name": "codex_apps_open_page",
                "namespace": "agent",
                "output": "{\"page_id\":null}"
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec_command",
                "arguments": "{}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok"
            }
        ]
    }));

    lower_responses_request_for_local_backend(&mut req)
        .expect("call_id-less function_call_output should lower cleanly");

    let wire = serde_json::to_value(&req).unwrap();
    let items = wire["input"].as_array().unwrap();
    assert_eq!(items[1]["type"], "message");
    assert_eq!(items[1]["role"], "user");
    assert_eq!(
        items[1]["content"][0]["text"],
        "[tool_output agent.codex_apps_open_page]\n{\"page_id\":null}"
    );
    // Normal call_id-bearing function_call_output stays untouched.
    assert_eq!(items[3]["type"], "function_call_output");
    assert_eq!(items[3]["call_id"], "call_1");
    assert_eq!(items[3]["output"], "ok");
}

#[test]
fn lowers_tool_search_and_web_search_items_for_local_backends() {
    let mut req = parse_request(json!({
        "model": "test-model",
        "tools": [],
        "input": [
            {"role": "user", "content": "check weather"},
            {
                "type": "tool_search_call",
                "call_id": "ts_1",
                "arguments": {"query": "weather"},
                "execution": "client"
            },
            {
                "type": "tool_search_output",
                "call_id": "ts_1",
                "execution": "client",
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"}
                }]
            },
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "sf weather"}
            }
        ]
    }));

    let mut explicit = req.clone();
    explicit.tool_choice = Some(ResponsesToolChoice::Options(ToolChoiceOptions::None));
    explicit.normalize();
    lower_responses_request_for_local_backend(&mut explicit).unwrap();
    assert!(matches!(
        explicit.tool_choice,
        Some(ResponsesToolChoice::Options(ToolChoiceOptions::None))
    ));

    let mut named = req.clone();
    named.tool_choice = Some(
        serde_json::from_value(json!({
            "type": "function",
            "name": "get_weather"
        }))
        .unwrap(),
    );
    named
        .validate()
        .expect("a discovered function can be selected by name");

    req.validate().unwrap();
    req.normalize();
    assert!(matches!(
        req.tool_choice,
        Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto))
    ));
    lower_responses_request_for_local_backend(&mut req)
        .expect("tool_search and web_search_call should lower cleanly");

    let wire = serde_json::to_value(&req).unwrap();
    let tools = wire["tools"].as_array().expect("discovered tool lifted");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "get_weather");
    let items = wire["input"].as_array().unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["role"], "user");
}

#[tokio::test]
async fn rejects_invalid_discovered_tools_with_their_input_path() {
    for (tool, code) in [
        (json!({"type": "function"}), "invalid_tool"),
        (
            json!({"type": "function", "name": "lookup", "parameters": []}),
            "invalid_type",
        ),
        (json!({"type": "local_shell"}), "tool_not_supported"),
    ] {
        let mut req = parse_request(json!({
            "model": "test-model",
            "input": [
                {"role": "user", "content": "hi"},
                {"type": "tool_search_output", "tools": [
                    {"type": "function", "name": "valid", "parameters": {}},
                    tool
                ]}
            ]
        }));
        let response = lower_responses_request_for_local_backend(&mut req).unwrap_err();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let payload: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(payload["error"]["code"], code);
        assert!(payload["error"]["message"]
            .as_str()
            .unwrap()
            .contains("input[1].tools[1]"));
    }
}
