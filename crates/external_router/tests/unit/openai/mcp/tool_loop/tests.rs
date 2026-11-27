use std::collections::HashSet;

use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::responses::ResponsesRequest;
use serde_json::json;
use smg_mcp::{
    BuiltinToolType, McpConfig, McpOrchestrator, McpServerBinding, McpServerConfig, McpToolSession,
    McpTransport, Tool, ToolEntry,
};
use tokio::sync::mpsc;

use super::{
    build_transformed_mcp_call_item, extract_openai_response_output_items,
    is_internal_mcp_response_item, mcp_list_tools_bindings_to_emit, not_executed_tool_call_item,
    upstream_round, ItemType, McpEvent, OutputItemEvent, SseEncoder, ToolLoopState,
    NOT_EXECUTED_AT_TOOL_CALL_LIMIT,
};
use crate::{
    header_utils::ApiProvider,
    openai::mcp::test_support,
    openai_bridge::{FormatRegistry, ResponseFormat},
    test_support::{answering, counting_alloc, http_client, RefusedAddress},
};

fn test_tool(name: &str) -> Tool {
    let mut schema = serde_json::Map::new();
    schema.insert("type".to_string(), json!("object"));
    schema.insert("properties".to_string(), json!({}));

    Tool::new(name.to_string(), "internal", schema)
}

/// A call left over the tool-call allowance is rendered in its own
/// format: an `mcp_call` that failed with the not-executed error and the
/// id an executed call would have had, or the hosted item type the stream
/// announces for the tool, `failed`, with that format's id.
#[test]
fn a_not_executed_call_item_keeps_the_format_the_stream_announced() {
    let plain = not_executed_tool_call_item(
        ResponseFormat::Passthrough,
        "fc_1",
        "mock",
        "brave_web_search",
        "{}",
    );
    assert_eq!(plain["type"], "mcp_call");
    assert_eq!(plain["status"], "failed");
    assert_eq!(plain["id"], "mcp_1");
    assert_eq!(plain["server_label"], "mock");
    assert!(
        plain["error"]
            .to_string()
            .contains(NOT_EXECUTED_AT_TOOL_CALL_LIMIT),
        "{plain}"
    );

    let hosted = not_executed_tool_call_item(
        ResponseFormat::WebSearchCall,
        "1",
        "mock",
        "brave_web_search",
        r#"{"query":"q"}"#,
    );
    assert_eq!(hosted["type"], "web_search_call");
    assert_eq!(hosted["status"], "failed");
    assert_eq!(hosted["id"], "ws_1");
    assert!(hosted.get("server_label").is_none());
}

#[test]
fn build_transformed_mcp_call_item_does_not_add_server_label_for_builtin_formats() {
    let item = build_transformed_mcp_call_item(
        &json!({
            "queries": ["private query"],
            "results": [
                { "url": "https://example.com" }
            ]
        }),
        ResponseFormat::WebSearchCall,
        "call_123",
        "internal-label",
        "brave_web_search",
        r#"{"query":"private query"}"#,
    );

    assert_eq!(
        item.get("type").and_then(|value| value.as_str()),
        Some("web_search_call")
    );
    assert!(item.get("server_label").is_none());
}

#[tokio::test]
async fn internal_filter_keeps_builtin_passthrough_mcp_call_items() {
    let orchestrator = test_support::internal_orchestrator_serving(
        Some((BuiltinToolType::WebSearchPreview, "brave_web_search")),
        vec![test_tool("brave_web_search")],
    )
    .await;
    let session = test_support::internal_session(&orchestrator);

    let item = json!({
        "type": "mcp_call",
        "name": "brave_web_search",
        "server_label": "internal-label"
    });

    assert!(!is_internal_mcp_response_item(&item, &session));
}

#[tokio::test]
async fn internal_filter_keeps_builtin_passthrough_mcp_call_items_with_mixed_tools() {
    let orchestrator = test_support::internal_orchestrator_serving(
        Some((BuiltinToolType::WebSearchPreview, "brave_web_search")),
        vec![
            test_tool("brave_web_search"),
            test_tool("internal_non_builtin_tool"),
        ],
    )
    .await;
    let session = test_support::internal_session(&orchestrator);

    let builtin_item = json!({
        "type": "mcp_call",
        "name": "brave_web_search",
        "server_label": "internal-label"
    });
    let internal_non_builtin_item = json!({
        "type": "mcp_call",
        "name": "internal_non_builtin_tool",
        "server_label": "internal-label"
    });

    assert!(!is_internal_mcp_response_item(&builtin_item, &session));
    assert!(is_internal_mcp_response_item(
        &internal_non_builtin_item,
        &session
    ));
}

#[test]
fn emits_only_new_binding_when_resume_adds_second_tool_block() {
    let existing_labels = HashSet::from(["deepwiki_ask".to_string()]);
    let bindings = vec![
        McpServerBinding {
            label: "deepwiki_ask".to_string(),
            server_key: "server-ask".to_string(),
            allowed_tools: Some(vec!["ask_question".to_string()]),
            list_error: None,
            pool_key: None,
        },
        McpServerBinding {
            label: "deepwiki_read".to_string(),
            server_key: "server-read".to_string(),
            allowed_tools: Some(vec!["read_wiki_structure".to_string()]),
            list_error: None,
            pool_key: None,
        },
    ];

    let bindings_to_emit = mcp_list_tools_bindings_to_emit(&existing_labels, &bindings);

    assert_eq!(
        bindings_to_emit,
        vec![("deepwiki_read".to_string(), "server-read".to_string())]
    );
}

#[test]
fn emits_all_bindings_when_no_prior_mcp_list_tools_exist() {
    let existing_labels = HashSet::new();
    let bindings = vec![McpServerBinding {
        label: "deepwiki_ask".to_string(),
        server_key: "server-ask".to_string(),
        allowed_tools: Some(vec!["ask_question".to_string()]),
        list_error: None,
        pool_key: None,
    }];

    let bindings_to_emit = mcp_list_tools_bindings_to_emit(&existing_labels, &bindings);

    assert_eq!(
        bindings_to_emit,
        vec![("deepwiki_ask".to_string(), "server-ask".to_string())]
    );
}

#[test]
fn extract_openai_response_output_items_from_embedded_text_json() {
    let output = r#"[{"type":"text","text":"{\"execution_id\":\"abc\",\"openai_response\":{\"content\":{\"type\":\"output_text\",\"annotations\":[{\"type\":\"url_citation\",\"title\":\"Example citation\",\"url\":\"https://example.com/openai-result\",\"start_index\":0,\"end_index\":10}],\"logprobs\":[],\"text\":\"intermediate summary\"}}}"}]"#;

    let extracted = extract_openai_response_output_items(output);
    assert_eq!(extracted.len(), 1);
    assert_eq!(extracted[0]["type"], "message");
    assert_eq!(extracted[0]["role"], "assistant");
    assert_eq!(extracted[0]["content"][0]["type"], "output_text");
    assert_eq!(extracted[0]["content"][0]["text"], "intermediate summary");
    assert_eq!(
        extracted[0]["content"][0]["annotations"][0]["type"],
        "url_citation"
    );
    assert_eq!(
        extracted[0]["content"][0]["annotations"][0]["title"],
        "Example citation"
    );
    assert_eq!(
        extracted[0]["content"][0]["annotations"][0]["url"],
        "https://example.com/openai-result"
    );
    assert_eq!(
        extracted[0]["content"][0]["annotations"][0]["start_index"],
        0
    );
    assert_eq!(
        extracted[0]["content"][0]["annotations"][0]["end_index"],
        10
    );
}

#[test]
fn record_call_appends_openai_response_output_after_tool_item() {
    let mut state = ToolLoopState::new(Vec::new());
    let transformed = json!({
        "type": "web_search_call",
        "id": "ws_test",
        "status": "completed",
        "action": {"type": "search"}
    });
    let output = r#"[{"type":"text","text":"{\"openai_response\":{\"content\":{\"type\":\"output_text\",\"annotations\":[{\"type\":\"url_citation\",\"title\":\"Example citation\",\"url\":\"https://example.com/openai-result\",\"start_index\":0,\"end_index\":10}],\"logprobs\":[],\"text\":\"intermediate\"}}}"}]"#;

    state.record_call(
        true,
        "call_123".to_string(),
        "search_web".to_string(),
        "{\"query\":\"x\"}".to_string(),
        output.to_string(),
        transformed,
    );

    assert_eq!(state.mcp_call_items.len(), 2);
    assert_eq!(state.mcp_call_items[0]["type"], "web_search_call");
    assert_eq!(state.mcp_call_items[1]["type"], "message");
    assert_eq!(
        state.mcp_call_items[1]["content"][0]["text"],
        "intermediate"
    );
    assert_eq!(
        state.mcp_call_items[1]["content"][0]["annotations"][0]["type"],
        "url_citation"
    );
    assert_eq!(
        state.mcp_call_items[1]["content"][0]["annotations"][0]["title"],
        "Example citation"
    );
    assert_eq!(
        state.mcp_call_items[1]["content"][0]["annotations"][0]["url"],
        "https://example.com/openai-result"
    );
    assert_eq!(
        state.mcp_call_items[1]["content"][0]["annotations"][0]["start_index"],
        0
    );
    assert_eq!(
        state.mcp_call_items[1]["content"][0]["annotations"][0]["end_index"],
        10
    );
}

#[test]
fn record_call_does_not_append_openai_response_output_for_non_builtin_tools() {
    let mut state = ToolLoopState::new(Vec::new());
    let transformed = json!({
        "type": "web_search_call",
        "id": "ws_test",
        "status": "completed",
        "action": {"type": "search"}
    });
    let output = r#"[{"type":"text","text":"{\"openai_response\":{\"content\":{\"type\":\"output_text\",\"annotations\":[],\"logprobs\":[],\"text\":\"intermediate\"}}}"}]"#;

    state.record_call(
        false,
        "call_123".to_string(),
        "internal_search_web".to_string(),
        "{\"query\":\"x\"}".to_string(),
        output.to_string(),
        transformed,
    );

    assert_eq!(state.mcp_call_items.len(), 1);
    assert_eq!(state.mcp_call_items[0]["type"], "web_search_call");
}

#[test]
fn extract_openai_response_output_items_ignores_null_openai_response() {
    let output = r#"[{"type":"text","text":"{\"openai_response\":null}"}]"#;
    let extracted = extract_openai_response_output_items(output);
    assert!(extracted.is_empty());
}

// ========================================================================
// Streaming emission ordering invariant.
//
// `send_tool_call_completion_events` MUST emit
// `response.<tool>.completed` BEFORE `response.output_item.done` so the
// umbrella event terminates the item's sub-events per spec (see
// `.claude/_audit/openai-responses-api-spec.md` §streaming, events
// L1054-L1072).
// ========================================================================

fn drain_channel(rx: &mut mpsc::Receiver<Result<bytes::Bytes, std::io::Error>>) -> Vec<String> {
    let mut events = Vec::new();
    while let Ok(chunk) = rx.try_recv() {
        let bytes = chunk.expect("no io errors in unit-test channel");
        events.push(String::from_utf8(bytes.to_vec()).expect("utf-8 sse block"));
    }
    events
}

fn event_type_from_sse_block(block: &str) -> String {
    // SSE block shape: `event: <name>\ndata: {...}\n\n`. We parse only
    // the leading `event: …` line to get the wire type without pulling
    // serde_json into this tiny assertion helper.
    for line in block.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            return rest.trim().to_string();
        }
    }
    block.to_string()
}

#[tokio::test]
async fn image_generation_completion_events_fire_before_output_item_done() {
    // Gap B lock test: after the tool executes, the completion
    // emitter must push `response.image_generation_call.completed`
    // onto the wire BEFORE `response.output_item.done`. If a future
    // refactor reverses the order this asserts and fails loudly.
    let call = super::FunctionCallInProgress {
        call_id: "call_img".to_string(),
        name: "image_generation".to_string(),
        arguments_buffer: "{}".to_string(),
        item_id: Some("fc_img".to_string()),
        output_index: 0,
        last_obfuscation: None,
        assigned_output_index: Some(0),
    };

    let tool_call_item = json!({
        "type": "image_generation_call",
        "id": "ig_img",
        "status": "completed",
        "result": "BASE64",
    });

    let (tx, mut rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
    let mut sequence_number: u64 = 0;

    let ok = super::send_tool_call_completion_events(
        &tx,
        &call,
        &tool_call_item,
        ResponseFormat::ImageGenerationCall,
        &mut sequence_number,
    )
    .await;
    assert!(ok, "send_tool_call_completion_events should not disconnect");
    drop(tx);

    let events = drain_channel(&mut rx);
    let types: Vec<String> = events
        .iter()
        .map(|b| event_type_from_sse_block(b))
        .collect();

    let completed_idx = types
        .iter()
        .position(|t| t == "response.image_generation_call.completed")
        .expect("response.image_generation_call.completed must be emitted");
    let done_idx = types
        .iter()
        .position(|t| t == "response.output_item.done")
        .expect("response.output_item.done must be emitted");

    assert!(
        completed_idx < done_idx,
        "`response.image_generation_call.completed` (index {completed_idx}) must come \
             before `response.output_item.done` (index {done_idx}); full sequence: {types:?}"
    );
}

#[tokio::test]
async fn web_search_completion_events_fire_before_output_item_done() {
    // Same ordering contract for the pre-existing web_search_call path,
    // so the invariant applies uniformly across every hosted-tool
    // ResponseFormat we emit.
    let call = super::FunctionCallInProgress {
        call_id: "call_ws".to_string(),
        name: "web_search".to_string(),
        arguments_buffer: "{}".to_string(),
        item_id: Some("fc_ws".to_string()),
        output_index: 0,
        last_obfuscation: None,
        assigned_output_index: Some(0),
    };

    let tool_call_item = json!({
        "type": "web_search_call",
        "id": "ws_ws",
        "status": "completed",
        "action": {"type": "search"}
    });

    let (tx, mut rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
    let mut sequence_number: u64 = 0;

    let ok = super::send_tool_call_completion_events(
        &tx,
        &call,
        &tool_call_item,
        ResponseFormat::WebSearchCall,
        &mut sequence_number,
    )
    .await;
    assert!(ok);
    drop(tx);

    let events = drain_channel(&mut rx);
    let types: Vec<String> = events
        .iter()
        .map(|b| event_type_from_sse_block(b))
        .collect();

    let completed_idx = types
        .iter()
        .position(|t| t == "response.web_search_call.completed")
        .expect("web_search_call.completed must be emitted");
    let done_idx = types
        .iter()
        .position(|t| t == "response.output_item.done")
        .expect("output_item.done must be emitted");
    assert!(completed_idx < done_idx);
}

#[tokio::test]
async fn code_interpreter_completion_events_fire_before_output_item_done() {
    // The suppression gate in `tool_handler.rs` drops the upstream
    // umbrella `output_item.done` for every tool-call item type. This
    // test locks the downstream half of the contract for
    // `code_interpreter_call` — the tool loop's completion emitter
    // must push `response.code_interpreter_call.completed` BEFORE
    // `response.output_item.done` so the gate's suppression lines up
    // with a correctly-ordered wire sequence.
    let call = super::FunctionCallInProgress {
        call_id: "call_ci".to_string(),
        name: "code_interpreter".to_string(),
        arguments_buffer: "{}".to_string(),
        item_id: Some("fc_ci".to_string()),
        output_index: 0,
        last_obfuscation: None,
        assigned_output_index: Some(0),
    };

    let tool_call_item = json!({
        "type": "code_interpreter_call",
        "id": "ci_ci",
        "status": "completed",
        "code": "print('hi')",
    });

    let (tx, mut rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
    let mut sequence_number: u64 = 0;

    let ok = super::send_tool_call_completion_events(
        &tx,
        &call,
        &tool_call_item,
        ResponseFormat::CodeInterpreterCall,
        &mut sequence_number,
    )
    .await;
    assert!(ok);
    drop(tx);

    let events = drain_channel(&mut rx);
    let types: Vec<String> = events
        .iter()
        .map(|b| event_type_from_sse_block(b))
        .collect();

    let completed_idx = types
        .iter()
        .position(|t| t == "response.code_interpreter_call.completed")
        .expect("code_interpreter_call.completed must be emitted");
    let done_idx = types
        .iter()
        .position(|t| t == "response.output_item.done")
        .expect("output_item.done must be emitted");
    assert!(
        completed_idx < done_idx,
        "`response.code_interpreter_call.completed` (index {completed_idx}) must come \
             before `response.output_item.done` (index {done_idx}); full sequence: {types:?}"
    );

    // No duplicate `output_item.done` — the tool loop emits exactly
    // one umbrella (the upstream copy is dropped by the suppression
    // gate in `tool_handler.rs`).
    let done_count = types
        .iter()
        .filter(|t| *t == "response.output_item.done")
        .count();
    assert_eq!(
        done_count, 1,
        "exactly one `output_item.done` expected, got {done_count}: {types:?}"
    );
}

#[tokio::test]
async fn file_search_completion_events_fire_before_output_item_done() {
    // Lock the ordering contract for the hosted `file_search_call`
    // format so the gate's suppression covers every tool-call item
    // type listed in `TOOL_CALL_ITEM_TYPES`.
    let call = super::FunctionCallInProgress {
        call_id: "call_fs".to_string(),
        name: "file_search".to_string(),
        arguments_buffer: "{}".to_string(),
        item_id: Some("fc_fs".to_string()),
        output_index: 0,
        last_obfuscation: None,
        assigned_output_index: Some(0),
    };

    let tool_call_item = json!({
        "type": "file_search_call",
        "id": "fs_fs",
        "status": "completed",
        "queries": ["needle"],
    });

    let (tx, mut rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
    let mut sequence_number: u64 = 0;

    let ok = super::send_tool_call_completion_events(
        &tx,
        &call,
        &tool_call_item,
        ResponseFormat::FileSearchCall,
        &mut sequence_number,
    )
    .await;
    assert!(ok);
    drop(tx);

    let events = drain_channel(&mut rx);
    let types: Vec<String> = events
        .iter()
        .map(|b| event_type_from_sse_block(b))
        .collect();

    let completed_idx = types
        .iter()
        .position(|t| t == "response.file_search_call.completed")
        .expect("file_search_call.completed must be emitted");
    let done_idx = types
        .iter()
        .position(|t| t == "response.output_item.done")
        .expect("output_item.done must be emitted");
    assert!(
        completed_idx < done_idx,
        "`response.file_search_call.completed` (index {completed_idx}) must come \
             before `response.output_item.done` (index {done_idx}); full sequence: {types:?}"
    );

    // No duplicate `output_item.done` — the tool loop emits exactly
    // one umbrella (the upstream copy is dropped by the suppression
    // gate in `tool_handler.rs`).
    let done_count = types
        .iter()
        .filter(|t| *t == "response.output_item.done")
        .count();
    assert_eq!(
        done_count, 1,
        "exactly one `output_item.done` expected, got {done_count}: {types:?}"
    );
}

#[test]
fn approval_request_item_id_source_uses_single_underscore() {
    // Regression: `normalize_tool_item_id_with_prefix` appends `_`
    // internally, so the prefix argument must be the bare token. A
    // prior version passed `"mcpr_"` here and emitted `mcpr__...` ids,
    // breaking the approval wire format.
    assert_eq!(super::approval_request_item_id_source("fc_abc"), "mcpr_abc");
    assert_eq!(
        super::approval_request_item_id_source("call_xyz"),
        "mcpr_xyz"
    );
    assert_eq!(
        super::approval_request_item_id_source("mcpr_already"),
        "mcpr_already"
    );
    assert_eq!(
        super::approval_request_item_id_source("raw_id"),
        "mcpr_raw_id"
    );
}

async fn function_tool_session() -> (McpOrchestrator, Vec<McpServerBinding>) {
    let orchestrator = McpOrchestrator::new(McpConfig {
        servers: vec![McpServerConfig {
            name: "wiki-server".to_string(),
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
            internal: false,
        }],
        ..Default::default()
    })
    .await
    .expect("orchestrator");
    orchestrator
        .tool_inventory()
        .insert_entry(ToolEntry::from_server_tool(
            "wiki-server",
            test_tool("ask_wiki"),
        ));
    let bindings = vec![McpServerBinding {
        label: "wiki".to_string(),
        server_key: "wiki-server".to_string(),
        allowed_tools: None,
        list_error: None,
        pool_key: None,
    }];
    (orchestrator, bindings)
}

#[tokio::test]
async fn prepare_preserves_explicit_tool_choice() {
    let (orchestrator, bindings) = function_tool_session().await;
    let session = McpToolSession::new(&orchestrator, bindings, "test-request");

    let mut payload = json!({
        "model": "m",
        "input": "q",
        "tools": [{"type": "mcp", "server_label": "wiki"}],
        "tool_choice": "required"
    });
    super::prepare_mcp_tools_as_functions(&mut payload, &session);

    assert_eq!(payload["tool_choice"], json!("required"));
    assert_eq!(payload["tools"][0]["type"], json!("function"));
}

#[tokio::test]
async fn prepare_remaps_hosted_tool_choice_to_function() {
    let (orchestrator, bindings) = function_tool_session().await;
    orchestrator
        .tool_inventory()
        .insert_entry(ToolEntry::from_server_tool(
            "wiki-server",
            test_tool("image_generation"),
        ));
    let session = McpToolSession::new(&orchestrator, bindings, "test-request");

    let mut payload = json!({
        "model": "m",
        "input": "q",
        "tools": [{"type": "image_generation"}],
        "tool_choice": {"type": "image_generation"}
    });
    super::prepare_mcp_tools_as_functions(&mut payload, &session);

    assert_eq!(
        payload["tool_choice"],
        json!({"type": "function", "name": "image_generation"})
    );
}

#[tokio::test]
async fn prepare_downgrades_unmappable_tool_choice_to_auto() {
    let (orchestrator, bindings) = function_tool_session().await;
    let session = McpToolSession::new(&orchestrator, bindings, "test-request");

    let mut payload = json!({
        "model": "m",
        "input": "q",
        "tools": [{"type": "mcp", "server_label": "wiki"}],
        "tool_choice": {"type": "web_search_preview"}
    });
    super::prepare_mcp_tools_as_functions(&mut payload, &session);

    assert_eq!(payload["tool_choice"], json!("auto"));
}

#[tokio::test]
async fn prepare_defaults_missing_tool_choice_to_auto() {
    let (orchestrator, bindings) = function_tool_session().await;
    let session = McpToolSession::new(&orchestrator, bindings, "test-request");

    let mut payload = json!({
        "model": "m",
        "input": "q",
        "tools": [{"type": "mcp", "server_label": "wiki"}]
    });
    super::prepare_mcp_tools_as_functions(&mut payload, &session);

    assert_eq!(payload["tool_choice"], json!("auto"));
}

/// The resumed payload keeps the base payload's keys where they were:
/// `input` is taken out in place and filled again, not removed and
/// appended (the crate's maps keep insertion order on the wire); the
/// loop's own `store` follows them.
#[test]
fn resume_payload_keeps_the_base_keys_in_place() {
    let base = json!({"model": "test-model", "input": "hi", "stream": false, "metadata": {}});
    let mut resumed = base;
    super::resume_payload_in_place(&mut resumed, &[], false).unwrap();
    let keys: Vec<&str> = resumed
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["model", "input", "stream", "metadata", "store"]);
    assert_eq!(resumed["input"][0]["content"][0]["text"], "hi");
}

#[test]
fn resume_payload_forces_auto_tool_choice() {
    let base = json!({
        "model": "m",
        "input": "q",
        "tools": [{"type": "function", "name": "ask_wiki"}],
        "tool_choice": "required"
    });
    let mut resumed = base;
    super::resume_payload_in_place(&mut resumed, &[], false).expect("resume payload");

    assert_eq!(resumed["tool_choice"], json!("auto"));
    assert_eq!(
        resumed["input"],
        json!([{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]}])
    );
}

/// The payload the loop started from carries the history the request
/// continues (a stored turn's function call and the client's answer to
/// it); the resume payload keeps it in front of the calls this loop ran.
#[test]
fn resume_payload_keeps_the_loaded_history_in_front_of_the_loops_calls() {
    let loaded_input = json!([
        {"type": "message", "id": "msg_1", "role": "user", "content": [{"type": "input_text", "text": "What is 6*7?"}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_A", "name": "calc", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_A", "output": "42"}
    ]);
    let base = json!({
        "model": "m",
        "input": loaded_input,
        "tools": [{"type": "function", "name": "search"}]
    });
    let executed = [
        json!({"type": "function_call", "call_id": "call_B", "name": "search", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "call_B", "output": "found"}),
    ];

    let mut resumed = base;
    super::resume_payload_in_place(&mut resumed, &executed, false).expect("resume payload");

    let kinds: Vec<(&str, &str)> = resumed["input"]
        .as_array()
        .expect("an item list")
        .iter()
        .map(|item| {
            (
                item["type"].as_str().unwrap_or("?"),
                item["call_id"].as_str().unwrap_or(""),
            )
        })
        .collect();
    assert_eq!(
        kinds,
        [
            ("message", ""),
            ("function_call", "call_A"),
            ("function_call_output", "call_A"),
            ("function_call", "call_B"),
            ("function_call_output", "call_B"),
        ]
    );
    assert_eq!(resumed["store"], json!(false));
}

/// The argument preparation both loops share: a payload that is not an
/// object becomes `{}` before the hosted-tool merge, the request's `user`
/// is forwarded for a hosted format and left out for a plain MCP tool,
/// and an object payload under a plain MCP tool goes through untouched.
#[test]
fn prepared_arguments_coerce_merge_and_forward_the_user() {
    let hosted = super::prepared_arguments("42", ResponseFormat::WebSearchCall, &[], Some("u-1"))
        .expect("a scalar payload parses");
    assert_eq!(hosted, json!({ "user": "u-1" }));
    let plain = super::prepared_arguments("42", ResponseFormat::Passthrough, &[], Some("u-1"))
        .expect("a scalar payload parses");
    assert_eq!(plain, json!({}));
    let object = super::prepared_arguments(
        r#"{"query": "weather", "user": null}"#,
        ResponseFormat::Passthrough,
        &[],
        Some("u-1"),
    )
    .expect("an object payload parses");
    assert_eq!(object, json!({ "query": "weather", "user": null }));
    let overwritten = super::prepared_arguments(
        r#"{"query": "weather", "user": null}"#,
        ResponseFormat::WebSearchCall,
        &[],
        Some("u-1"),
    )
    .expect("an object payload parses");
    assert_eq!(overwritten, json!({ "query": "weather", "user": "u-1" }));
    assert!(super::prepared_arguments("not json", ResponseFormat::Passthrough, &[], None).is_err());
}

/// The `output` of the one `function_call_output` item a loop recorded
/// for the model's next round.
fn function_call_output(state: &ToolLoopState) -> String {
    let item = state
        .conversation_history
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("a function_call_output item");
    item["output"]
        .as_str()
        .expect("the output is a string")
        .to_string()
}

/// A call whose arguments are not JSON is counted as an errored call in
/// both loops (the streaming one counted nothing before), and both loops
/// word the error on the call's item the same way.
#[tokio::test]
async fn a_call_with_non_json_arguments_is_counted_and_worded_alike_in_both_loops() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);
    let (orchestrator, bindings) = function_tool_session().await;
    let session = McpToolSession::new(&orchestrator, bindings, "test-request");
    let registry = FormatRegistry::new();

    let (tx, mut rx) = mpsc::channel(16);
    let mut state = ToolLoopState::new(Vec::new());
    let mut sequence_number = 0u64;
    let mut call = super::FunctionCallInProgress::new("call_1".to_string(), 0);
    call.name = "wiki_search".to_string();
    call.arguments_buffer = "not json".to_string();
    let client_stayed = super::execute_streaming_tool_calls(
        vec![call],
        &session,
        &registry,
        &tx,
        &mut state,
        &mut sequence_number,
        "m",
        &[],
        None,
    )
    .await;
    assert!(client_stayed);
    let streamed = state.mcp_call_items[0].to_string();
    assert!(
        streamed.contains("Invalid tool arguments: "),
        "the streaming loop's item: {streamed}"
    );
    let streamed_output = function_call_output(&state);
    drop(tx);
    while rx.recv().await.is_some() {}

    let mut state = ToolLoopState::new(Vec::new());
    let body: ResponsesRequest =
        serde_json::from_value(json!({ "model": "m", "input": "q" })).expect("a request");
    let approval = super::dispatch_call(
        super::ExtractedFunctionCall {
            call_id: "call_2".to_string(),
            item_id: "fc_2".to_string(),
            name: "wiki_search".to_string(),
            arguments: "not json".to_string(),
        },
        &mut state,
        &session,
        &registry,
        &body,
    )
    .await;
    assert!(approval.is_none());
    let direct = state.mcp_call_items[0].to_string();
    assert!(
        direct.contains("Invalid tool arguments: "),
        "the non-streaming loop's item: {direct}"
    );
    let direct_output = function_call_output(&state);
    assert!(
        direct_output.starts_with("{\"error\":\"Invalid tool arguments: ") && direct_output == streamed_output,
        "the function_call_output the model reads back, both loops: {direct_output} vs {streamed_output}"
    );

    let series: Vec<String> = handle
        .render()
        .lines()
        .filter(|line| line.contains("smg_mcp_tool_calls_total") && line.contains("wiki_search"))
        .map(str::to_owned)
        .collect();
    assert_eq!(series.len(), 1, "one errored-call series: {series:?}");
    assert!(
        series[0].contains("result=\"error\"") && series[0].ends_with(" 2"),
        "both loops counted: {}",
        series[0]
    );
}

/// The sub-event a dispatched call closes under, read from its item: an
/// `mcp_call` that failed closes under `response.mcp_call.failed`, the
/// event a call past the allowance closes under; an `mcp_call` that
/// completed, and a hosted-format call whatever its status (the API has
/// no failed sub-event for those item types), close under the format's
/// completed event. The item's `response.output_item.done` follows each.
#[tokio::test]
async fn a_dispatched_call_closes_under_the_sub_event_its_item_reads() {
    for (item_type, status, format, sub_event) in [
        (
            "mcp_call",
            "completed",
            ResponseFormat::Passthrough,
            "response.mcp_call.completed",
        ),
        (
            "mcp_call",
            "failed",
            ResponseFormat::Passthrough,
            "response.mcp_call.failed",
        ),
        (
            "web_search_call",
            "failed",
            ResponseFormat::WebSearchCall,
            "response.web_search_call.completed",
        ),
    ] {
        let call = super::FunctionCallInProgress {
            call_id: "call_1".to_string(),
            name: "search".to_string(),
            arguments_buffer: "{}".to_string(),
            item_id: Some("fc_1".to_string()),
            output_index: 0,
            last_obfuscation: None,
            assigned_output_index: Some(0),
        };
        let item = json!({"type": item_type, "id": "item_1", "status": status});
        let (tx, mut rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
        let mut sequence_number: u64 = 0;
        let ok = super::send_tool_call_completion_events(
            &tx,
            &call,
            &item,
            format,
            &mut sequence_number,
        )
        .await;
        assert!(ok);
        drop(tx);
        let types: Vec<String> = drain_channel(&mut rx)
            .iter()
            .map(|block| event_type_from_sse_block(block))
            .collect();
        assert_eq!(
            types,
            [sub_event, "response.output_item.done"],
            "{item_type} with status {status}"
        );
    }
}

/// A streamed call the session cannot run is closed on the wire as a
/// call that failed: `response.mcp_call.failed`, then the item's
/// `response.output_item.done`, in sequence, under the id the stream
/// announced for the call; the loop's state records the call with its
/// error output and its errored-call metric counts it once.
#[tokio::test]
async fn a_streamed_call_is_closed_on_the_wire_under_its_announced_id_and_recorded() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);
    let (orchestrator, bindings) = function_tool_session().await;
    let session = McpToolSession::new(&orchestrator, bindings, "test-request");
    let registry = FormatRegistry::new();

    let (tx, mut rx) = mpsc::channel(16);
    let mut state = ToolLoopState::new(Vec::new());
    let mut sequence_number = 7u64;
    let mut call = super::FunctionCallInProgress::new("call_9".to_string(), 3);
    call.name = "not_exposed".to_string();
    call.arguments_buffer = "{\"q\":1}".to_string();
    let client_stayed = super::execute_streaming_tool_calls(
        vec![call],
        &session,
        &registry,
        &tx,
        &mut state,
        &mut sequence_number,
        "m",
        &[],
        None,
    )
    .await;
    assert!(client_stayed);
    drop(tx);

    let events = drain_channel(&mut rx);
    let types: Vec<String> = events
        .iter()
        .map(|e| event_type_from_sse_block(e))
        .collect();
    assert_eq!(
        types,
        ["response.mcp_call.failed", "response.output_item.done"],
        "the call's events, in order"
    );
    assert_eq!(sequence_number, 9, "one sequence number per event");
    let failed: serde_json::Value = events[0]
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .and_then(|data| serde_json::from_str(data).ok())
        .expect("the failed event's data");
    assert_eq!(failed["sequence_number"], 7);
    assert_eq!(failed["output_index"], 3);
    assert_eq!(failed["item_id"], "mcp_9", "the id the stream announced");
    let done: serde_json::Value = events[1]
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .and_then(|data| serde_json::from_str(data).ok())
        .expect("the done event's data");
    assert_eq!(done["sequence_number"], 8);
    assert_eq!(done["output_index"], 3);
    assert_eq!(done["item"]["id"], "mcp_9", "the id the stream announced");
    assert_eq!(done["item"]["type"], "mcp_call");
    assert_eq!(done["item"]["status"], "failed");

    assert_eq!(state.mcp_call_items.len(), 1);
    assert_eq!(
        state.mcp_call_items[0], done["item"],
        "the recorded item is the streamed one"
    );
    assert!(
        function_call_output(&state).contains("is not in this session's exposed tool map"),
        "the output the model reads back: {}",
        function_call_output(&state)
    );
    let series: Vec<String> = handle
        .render()
        .lines()
        .filter(|line| line.contains("smg_mcp_tool_calls_total") && line.contains("not_exposed"))
        .map(str::to_owned)
        .collect();
    assert_eq!(series.len(), 1, "one errored-call series: {series:?}");
    assert!(
        series[0].contains("result=\"error\"") && series[0].ends_with(" 1"),
        "counted once: {}",
        series[0]
    );
}

/// The list item's third event names the listing's outcome as the gRPC
/// stream does: `response.mcp_list_tools.failed` for an item carrying
/// the listing's `error`, `completed` for one without; the event's `type`
/// and its SSE `event:` line are one kind, the payload carries the
/// item's id and no error (the done item does).
#[tokio::test]
async fn a_failed_listings_item_is_announced_as_failed() {
    for (error, kind) in [
        (None, McpEvent::LIST_TOOLS_COMPLETED),
        (
            Some("tools/list refused by the server"),
            McpEvent::LIST_TOOLS_FAILED,
        ),
    ] {
        let mut item = json!({
            "type": ItemType::MCP_LIST_TOOLS,
            "id": "mcpl_test",
            "server_label": "conf",
            "tools": []
        });
        if let Some(error) = error {
            item["error"] = json!(error);
        }
        let (tx, mut rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(8);
        let mut enc = SseEncoder::new();
        let mut sequence_number: u64 = 0;

        let ok =
            super::send_mcp_list_tools_events(&tx, &mut enc, &item, 0, &mut sequence_number).await;
        assert!(ok, "the client is connected");
        drop(tx);

        let events = drain_channel(&mut rx);
        let types: Vec<String> = events
            .iter()
            .map(|block| event_type_from_sse_block(block))
            .collect();
        assert_eq!(
            types,
            [
                OutputItemEvent::ADDED,
                McpEvent::LIST_TOOLS_IN_PROGRESS,
                kind,
                OutputItemEvent::DONE,
            ],
            "the event lines for error {error:?}"
        );
        let data = events[2]
            .split("data: ")
            .nth(1)
            .expect("the third event's data line")
            .trim();
        let payload: serde_json::Value = serde_json::from_str(data).expect("json");
        assert_eq!(payload["type"], kind);
        assert_eq!(payload["item_id"], "mcpl_test");
        assert_eq!(payload["output_index"], 0);
        assert_eq!(payload.get("error"), None, "the failure is on the item");
    }
}

/// The loop announces only what a client may see: an internal server's
/// `mcp_list_tools` item is left out of the kept items, so the stream's
/// start, the completed event and the stored record agree; a visible
/// server keeps its item when an internal builtin-routed binding shares
/// its label (the predicate the other renders use, not the label-wide one).
#[tokio::test]
async fn an_internal_servers_list_tools_item_is_not_announced() {
    let server = |name: &str, port: u16, internal: bool| McpServerConfig {
        name: name.to_string(),
        transport: McpTransport::Sse {
            url: format!("http://localhost:{port}/sse"),
            token: None,
            headers: Default::default(),
        },
        proxy: None,
        required: false,
        tools: None,
        builtin_type: None,
        builtin_tool_name: None,
        internal,
    };
    let orchestrator = McpOrchestrator::new(McpConfig {
        servers: vec![
            server("wiki-server", 3000, false),
            server("ledger-server", 3001, true),
            McpServerConfig {
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
                ..server("search-server", 3002, true)
            },
        ],
        ..Default::default()
    })
    .await
    .expect("orchestrator");
    orchestrator
        .tool_inventory()
        .insert_entry(ToolEntry::from_server_tool(
            "wiki-server",
            test_tool("ask_wiki"),
        ));
    orchestrator
        .tool_inventory()
        .insert_entry(ToolEntry::from_server_tool(
            "ledger-server",
            test_tool("post_entry"),
        ));
    let binding = |label: &str, server_key: &str| McpServerBinding {
        label: label.to_string(),
        server_key: server_key.to_string(),
        allowed_tools: None,
        list_error: None,
        pool_key: None,
    };
    let session = McpToolSession::new(
        &orchestrator,
        vec![
            binding("wiki", "wiki-server"),
            binding("ledger", "ledger-server"),
            // An internal builtin-routed binding under the visible server's label.
            binding("wiki", "search-server"),
        ],
        "test-request",
    );
    assert!(session.is_internal_non_builtin_server_label("ledger"));
    // The label-wide predicate would drop the visible server's item.
    assert!(session.is_internal_server_label("wiki"));
    assert!(!session.is_internal_non_builtin_server_label("wiki"));

    let mut state = ToolLoopState::new(Vec::new());
    let labels: Vec<&str> = state
        .mcp_list_tools_items(&session)
        .iter()
        .map(|item| item["server_label"].as_str().unwrap_or("?"))
        .collect();
    assert_eq!(
        labels,
        ["wiki"],
        "the internal server's item is not announced"
    );
}

/// Turn N appends only the calls recorded since turn N-1: the cursor
/// advances with every resume, so a call never rides upstream twice and
/// a turn with nothing new appends nothing.
#[test]
fn a_resume_appends_only_the_calls_recorded_since_the_last_turn() {
    let mut state = ToolLoopState::new(Vec::new());
    let mut payload = json!({
        "model": "m",
        "input": [{"type": "message", "role": "user", "content": "find it"}],
        "tools": [{"type": "function", "name": "search"}]
    });
    let call = |state: &mut ToolLoopState, id: &str| {
        state.record_call(
            false,
            id.to_string(),
            "search".to_string(),
            "{}".to_string(),
            "found".to_string(),
            json!({"type": "mcp_call", "id": id}),
        );
    };
    let kinds = |payload: &serde_json::Value| -> Vec<(String, String)> {
        payload["input"]
            .as_array()
            .expect("an item list")
            .iter()
            .map(|item| {
                (
                    item["type"].as_str().unwrap_or("?").to_string(),
                    item["call_id"].as_str().unwrap_or("").to_string(),
                )
            })
            .collect()
    };

    call(&mut state, "call_A");
    super::resume_payload_in_place(&mut payload, state.history_to_append(), false)
        .expect("the first resume");
    call(&mut state, "call_B");
    super::resume_payload_in_place(&mut payload, state.history_to_append(), false)
        .expect("the second resume");
    let expected: Vec<(String, String)> = [
        ("message", ""),
        ("function_call", "call_A"),
        ("function_call_output", "call_A"),
        ("function_call", "call_B"),
        ("function_call_output", "call_B"),
    ]
    .iter()
    .map(|(kind, id)| (kind.to_string(), id.to_string()))
    .collect();
    assert_eq!(
        kinds(&payload),
        expected,
        "each call once, in order, behind the input"
    );

    super::resume_payload_in_place(&mut payload, state.history_to_append(), false)
        .expect("a resume with nothing new");
    assert_eq!(
        kinds(&payload),
        expected,
        "a turn with nothing new appends nothing"
    );
}

/// The resume payload of a request of ten input items, one tool and four
/// history items: its allocation calls are pinned. The payload the loop
/// sent is edited in place (the four items are copied behind its input),
/// where a copy of the whole payload, its tools included, was made per
/// turn. The count is printed for the Numbers table.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]
fn a_resume_payload_appends_the_new_history_in_place() {
    let items: Vec<serde_json::Value> = (0..10)
        .map(|i| {
            json!({"type": "message", "role": "user", "content": [{"type": "input_text",
                "text": format!("turn {i}: a sentence long enough to live on the heap")}]})
        })
        .collect();
    let tools = json!([{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]);
    let mut payload = json!({"model": "m", "input": items, "tools": tools, "stream": false});
    let first_item = payload["input"][0].clone();
    let history: Vec<serde_json::Value> = (0..4)
        .map(|i| json!({"type": "function_call_output", "call_id": format!("call_{i}"), "output": format!("result {i}")}))
        .collect();

    let before = counting_alloc::calls();
    super::resume_payload_in_place(&mut payload, &history, false).expect("a resume payload");
    let calls = counting_alloc::calls() - before;

    eprintln!("a resume payload of 10 input items and 4 history items: {calls} allocation calls");
    assert_eq!(payload["input"].as_array().map(Vec::len), Some(14));
    assert_eq!(payload["input"][0], first_item);
    assert_eq!(payload["input"][13], history[3]);
    assert_eq!(payload["tool_choice"], json!("auto"));
    assert_eq!(
        calls, PINNED_RESUME_CALLS,
        "the resume payload's allocation calls changed"
    );
}

/// Allocation calls of the resume payload above, today.
const PINNED_RESUME_CALLS: usize = 38;

/// A tool whose JSON Schema is about a kilobyte: eight string arguments
/// with a sentence of description each.
fn kilobyte_tool(name: &str) -> Tool {
    let mut properties = serde_json::Map::new();
    for field in [
        "query", "locale", "region", "format", "cursor", "filter", "since", "until",
    ] {
        properties.insert(
            field.to_string(),
            json!({"type": "string", "description": format!("the {field} of the lookup, a sentence long enough to make the schema about a kilobyte of JSON")}),
        );
    }
    let mut schema = serde_json::Map::new();
    schema.insert("type".to_string(), json!("object"));
    schema.insert(
        "properties".to_string(),
        serde_json::Value::Object(properties),
    );
    schema.insert("required".to_string(), json!(["query"]));
    Tool::new(name.to_string(), "a tool of a kilobyte of schema", schema)
}

/// The model's tool list and the `mcp_list_tools` item of a session of
/// 30 tools with 1 KB schemas: their allocation calls are pinned. Each
/// schema is copied once into each (the copy the payload owns), where
/// `json!` and the typed item's serialisation copied it twice. The count
/// is printed for the Numbers table.
#[tokio::test]
#[expect(
    clippy::print_stderr,
    reason = "the count a pin reads is printed for the Numbers table"
)]
async fn the_tool_descriptors_copy_each_schema_once() {
    let (orchestrator, bindings) = function_tool_session().await;
    for i in 0..30 {
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "wiki-server",
                kilobyte_tool(&format!("lookup_{i:02}")),
            ));
    }
    let session = McpToolSession::new(&orchestrator, bindings, "test-request");
    assert!(
        serde_json::to_string(&*kilobyte_tool("probe").input_schema)
            .expect("a schema")
            .len()
            >= 1024
    );
    let mut payload = json!({
        "model": "m",
        "input": "q",
        "tools": [{"type": "mcp", "server_label": "wiki"}]
    });

    // UUID's fast-rng feature can warm rand's thread-local generator during
    // session setup. Exclude that one-time setup under either feature set.
    let _ = rand::rng();
    let before = counting_alloc::calls();
    super::prepare_mcp_tools_as_functions(&mut payload, &session);
    let item = crate::openai_bridge::mcp_list_tools_json(&session, "wiki", "wiki-server");
    let calls = counting_alloc::calls() - before;

    eprintln!(
        "the model's tool list and the list-tools item of 30 tools with 1 KB schemas: {calls} allocation calls"
    );
    assert_eq!(payload["tools"].as_array().map(Vec::len), Some(31));
    assert_eq!(item["tools"].as_array().map(Vec::len), Some(31));
    assert_eq!(
        calls, PINNED_DESCRIPTOR_CALLS,
        "the tool descriptors' allocation calls changed"
    );
}

/// A session reads the orchestrator's maps by key. A walk of one of them
/// allocates once per shard, four times at the least and more with every
/// core of the host. No read here has room for one: the tool listing of a
/// server without tools allocates nothing, and a session makes only its
/// own calls, 4 when the request binds no server and 12 when it binds
/// one.
#[tokio::test]
async fn a_session_walks_no_sharded_map() {
    let orchestrator = McpOrchestrator::new(McpConfig::default())
        .await
        .expect("an orchestrator with no servers");

    let before = counting_alloc::calls();
    let session = McpToolSession::new(&orchestrator, Vec::new(), "test-request");
    let unbound = counting_alloc::calls() - before;

    let bindings = vec![McpServerBinding {
        label: "a-label".to_string(),
        server_key: "a-server".to_string(),
        allowed_tools: None,
        list_error: None,
        pool_key: None,
    }];
    let before = counting_alloc::calls();
    let bound_session = McpToolSession::new(&orchestrator, bindings, "test-request");
    let bound = counting_alloc::calls() - before;

    let keys = ["a-server".to_string()];
    let before = counting_alloc::calls();
    let listed = orchestrator.list_tools_for_servers(&keys);
    let listing = counting_alloc::calls() - before;

    assert!(session.mcp_tools().is_empty() && listed.is_empty());
    assert!(bound_session.mcp_tools().is_empty() && bound_session.mcp_servers().len() == 1);
    assert!(
        unbound < 8 && bound < 16 && listing < 4,
        "allocation calls: {unbound} for a session without a server, {bound} for a session that binds one, {listing} for a server's tool listing"
    );
}

/// A session that may route a builtin tool asks for the server of each builtin type, four
/// lookups. The servers here are configured and not connected (the SSE transport is refused
/// at connect), so each lookup reads the index, finds nothing and answers from the
/// configured list. None walks the server map, where even an empty walk allocates once per
/// shard, so the session's allocation calls are the same on a host of any width.
#[tokio::test]
async fn a_session_that_routes_a_builtin_tool_walks_no_sharded_map() {
    let server = |name: &str, port: u16| McpServerConfig {
        name: name.to_string(),
        transport: McpTransport::Sse {
            url: format!("http://localhost:{port}/sse"),
            token: None,
            headers: Default::default(),
        },
        proxy: None,
        required: false,
        tools: None,
        builtin_type: None,
        builtin_tool_name: None,
        internal: false,
    };
    let orchestrator = McpOrchestrator::new(McpConfig {
        servers: vec![
            server("wiki-server", 3000),
            McpServerConfig {
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
                ..server("search-server", 3002)
            },
        ],
        ..Default::default()
    })
    .await
    .expect("orchestrator");
    orchestrator
        .tool_inventory()
        .insert_entry(ToolEntry::from_server_tool(
            "search-server",
            test_tool("brave_web_search"),
        ));
    let bindings = vec![McpServerBinding {
        label: "search".to_string(),
        server_key: "search-server".to_string(),
        allowed_tools: None,
        list_error: None,
        pool_key: None,
    }];

    let before = counting_alloc::calls();
    let session = McpToolSession::new(&orchestrator, bindings, "test-request");
    let calls = counting_alloc::calls() - before;

    assert_eq!(session.mcp_tools().len(), 1);
    assert!(
        calls < 40,
        "allocation calls: {calls} for a session that binds a builtin server"
    );
}

/// Steady-state allocation calls of the tool descriptors, excluding RNG setup.
const PINNED_DESCRIPTOR_CALLS: usize = 4642;

/// The JSON list-tools item is the typed item serialised, byte for byte
/// (its id aside: each build draws one).
#[tokio::test]
async fn the_json_list_tools_item_is_the_typed_one() {
    let (orchestrator, bindings) = function_tool_session().await;
    let session = McpToolSession::new(&orchestrator, bindings, "test-request");
    let mut built = crate::openai_bridge::mcp_list_tools_json(&session, "wiki", "wiki-server");
    let mut typed = serde_json::to_value(crate::openai_bridge::mcp_list_tools_item(
        &session,
        "wiki",
        "wiki-server",
    ))
    .expect("the typed item serialises");
    for item in [&mut built, &mut typed] {
        item["id"] = json!("mcpl_x");
    }
    assert_eq!(
        serde_json::to_string(&built).expect("JSON"),
        serde_json::to_string(&typed).expect("JSON")
    );
}

/// A round nothing answers fails without the request URL in its text. The
/// HTTP client's own error text ends with it, its query kept.
#[tokio::test]
async fn a_round_nothing_answers_goes_without_the_clients_url() {
    let refused = RefusedAddress::reserve();
    let url = format!("http://{}/v1/responses?key=s3cret", refused.addr());

    let Err(text) = upstream_round(
        &http_client(),
        &url,
        &ApiProvider::Generic,
        None,
        &json!({}),
    )
    .await
    else {
        panic!("a failed send");
    };

    assert!(!text.contains("s3cret"), "{text}");
    assert!(!text.contains(&refused.addr().to_string()), "{text}");
}

/// A round answered with no JSON fails without the request URL in its
/// text. The HTTP client's text of a failed body read ends with it, its
/// query kept.
#[tokio::test]
async fn a_round_answered_with_no_json_goes_without_the_clients_url() {
    let addr = answering(200, "not json", false).await;
    let url = format!("http://{addr}/v1/responses?key=s3cret");

    let Err(text) = upstream_round(
        &http_client(),
        &url,
        &ApiProvider::Generic,
        None,
        &json!({}),
    )
    .await
    else {
        panic!("an answer that is no JSON");
    };

    assert!(text.starts_with("parse response"), "{text}");
    assert!(!text.contains("s3cret"), "{text}");
}
