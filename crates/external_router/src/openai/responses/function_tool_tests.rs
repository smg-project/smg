//! A caller-declared function tool must never reach the MCP executor.
//!
//! The model sees MCP tools and the caller's function tools as one flat list
//! of functions. Only the calls whose name the MCP session exposes are run by
//! the router; every other `function_call` belongs to the caller and has to
//! come back in the response unchanged. These tests drive the real
//! non-streaming loop and the real streaming loop against a scripted upstream.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use axum::{extract::State, http::header, response::IntoResponse, routing::post, Json, Router};
use openai_protocol::{
    common::Function,
    responses::{FunctionTool, ResponseInput, ResponseTool, ResponsesRequest},
};
use rustls::crypto::ring;
use serde_json::{json, Value};
use smg_data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
use smg_mcp::{
    McpConfig, McpOrchestrator, McpServerBinding, McpServerConfig, McpToolSession, McpTransport,
    Tool, ToolEntry,
};

use super::streaming::handle_streaming_with_tool_interception;
use crate::{
    openai::{
        context::{StorageHandles, StreamingRequest},
        mcp::{execute_tool_loop, ToolLoopExecutionContext},
    },
    openai_bridge::FormatRegistry,
};

/// The tests' HTTP client, the process-level TLS crypto provider installed
/// first (reqwest is built without a bundled one; installing twice is harmless).
#[expect(
    clippy::disallowed_methods,
    reason = "a test binary's client factory; the process TLS crypto provider is installed first"
)]
fn client() -> reqwest::Client {
    let _ = ring::default_provider().install_default();
    reqwest::Client::new()
}

const MCP_TOOL: &str = "maps_search";
const FN_TOOL: &str = "get_weather";

#[derive(Clone)]
struct Upstream {
    scripted: Arc<Mutex<VecDeque<String>>>,
    seen: Arc<Mutex<Vec<Value>>>,
}

async fn upstream_handler(
    State(up): State<Upstream>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    up.seen.lock().unwrap().push(body);
    let next = up
        .scripted
        .lock()
        .unwrap()
        .pop_front()
        .expect("upstream called more times than scripted");
    let content_type = if streaming {
        "text/event-stream"
    } else {
        "application/json"
    };
    ([(header::CONTENT_TYPE, content_type)], next)
}

/// Serve `scripted` bodies, one per upstream request, on a random local port.
async fn spawn_upstream(scripted: Vec<String>) -> (String, Upstream) {
    let up = Upstream {
        scripted: Arc::new(Mutex::new(scripted.into())),
        seen: Arc::new(Mutex::new(Vec::new())),
    };
    let app = Router::new()
        .route("/v1/responses", post(upstream_handler))
        .with_state(up.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    #[expect(clippy::disallowed_methods, reason = "test server lives for the test")]
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}/v1/responses"), up)
}

async fn orchestrator_with_mcp_tool() -> McpOrchestrator {
    let orchestrator = McpOrchestrator::new(McpConfig {
        servers: vec![McpServerConfig {
            name: "maps-server".to_string(),
            transport: McpTransport::Sse {
                url: "http://127.0.0.1:1/sse".to_string(),
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

    let mut schema = serde_json::Map::new();
    schema.insert("type".to_string(), json!("object"));
    schema.insert("properties".to_string(), json!({}));
    orchestrator
        .tool_inventory()
        .insert_entry(ToolEntry::from_server_tool(
            "maps-server",
            Tool::new(MCP_TOOL.to_string(), "search places", schema),
        ));
    orchestrator
}

fn binding() -> McpServerBinding {
    McpServerBinding {
        label: "maps".to_string(),
        server_key: "maps-server".to_string(),
        allowed_tools: None,
    }
}

fn request(max_tool_calls: Option<u32>) -> ResponsesRequest {
    ResponsesRequest {
        model: "gpt-test".to_string(),
        input: ResponseInput::Text("plan my day".to_string()),
        tools: Some(vec![ResponseTool::Function(FunctionTool {
            function: Function {
                name: FN_TOOL.to_string(),
                description: Some("caller function".to_string()),
                parameters: json!({"type": "object", "properties": {}}),
                strict: None,
                extra: Default::default(),
            },
        })]),
        max_tool_calls,
        ..Default::default()
    }
}

fn payload(streaming: bool) -> Value {
    json!({
        "model": "gpt-test",
        "input": "plan my day",
        "stream": streaming,
        "tools": [
            {"type": "function", "name": FN_TOOL, "parameters": {"type": "object", "properties": {}}},
            {"type": "function", "name": MCP_TOOL, "parameters": {"type": "object", "properties": {}}},
        ],
    })
}

fn function_call(name: &str, call_id: &str) -> Value {
    json!({
        "type": "function_call",
        "id": format!("fc_{call_id}"),
        "call_id": call_id,
        "name": name,
        "arguments": "{}",
        "status": "completed",
    })
}

fn message(text: &str) -> Value {
    json!({
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    })
}

fn response_json(output: Vec<Value>) -> String {
    json!({
        "id": "resp_up",
        "object": "response",
        "status": "completed",
        "model": "gpt-test",
        "output": output,
    })
    .to_string()
}

fn items_of_type<'a>(output: &'a [Value], item_type: &str) -> Vec<&'a Value> {
    output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some(item_type))
        .collect()
}

fn names(items: &[&Value]) -> Vec<String> {
    items
        .iter()
        .map(|item| item["name"].as_str().unwrap_or_default().to_string())
        .collect()
}

async fn run_non_streaming(
    scripted: Vec<String>,
    max_tool_calls: Option<u32>,
) -> (Value, Upstream) {
    let (url, up) = spawn_upstream(scripted).await;
    let orchestrator = orchestrator_with_mcp_tool().await;
    let session = McpToolSession::new(&orchestrator, vec![binding()], "req-test");
    let body = request(max_tool_calls);
    let registry = FormatRegistry::new();
    let result = execute_tool_loop(
        &client(),
        &url,
        None,
        None,
        payload(false),
        ToolLoopExecutionContext {
            original_body: &body,
            existing_mcp_list_tools_labels: &[],
            session: &session,
            format_registry: &registry,
        },
    )
    .await
    .expect("tool loop");
    (result, up)
}

#[tokio::test]
async fn non_streaming_function_only_turn_returns_the_call_without_touching_mcp() {
    let (result, up) = run_non_streaming(
        vec![response_json(vec![function_call(FN_TOOL, "call_f")])],
        None,
    )
    .await;

    let output = result["output"].as_array().unwrap();
    assert_eq!(names(&items_of_type(output, "function_call")), [FN_TOOL]);
    assert!(items_of_type(output, "mcp_call").is_empty(), "{output:?}");
    assert_eq!(up.seen.lock().unwrap().len(), 1, "the loop must end");
}

#[tokio::test]
async fn non_streaming_mcp_only_turn_executes_and_resumes() {
    let (result, up) = run_non_streaming(
        vec![
            response_json(vec![function_call(MCP_TOOL, "call_m")]),
            response_json(vec![message("done")]),
        ],
        None,
    )
    .await;

    let output = result["output"].as_array().unwrap();
    assert_eq!(names(&items_of_type(output, "mcp_call")), [MCP_TOOL]);
    assert!(
        items_of_type(output, "function_call").is_empty(),
        "{output:?}"
    );
    assert_eq!(items_of_type(output, "message").len(), 1);

    let seen = up.seen.lock().unwrap();
    assert_eq!(seen.len(), 2, "one resume after the MCP call");
    let resumed = seen[1]["input"].as_array().unwrap();
    assert_eq!(items_of_type(resumed, "function_call_output").len(), 1);
}

#[tokio::test]
async fn non_streaming_mixed_turn_runs_mcp_and_returns_the_function_call() {
    let (result, up) = run_non_streaming(
        vec![response_json(vec![
            function_call(MCP_TOOL, "call_m"),
            function_call(FN_TOOL, "call_f"),
        ])],
        None,
    )
    .await;

    let output = result["output"].as_array().unwrap();
    assert_eq!(names(&items_of_type(output, "mcp_call")), [MCP_TOOL]);
    let calls = items_of_type(output, "function_call");
    assert_eq!(names(&calls), [FN_TOOL], "only the caller's call stays");
    assert_eq!(calls[0]["call_id"], "call_f");
    assert_eq!(
        up.seen.lock().unwrap().len(),
        1,
        "the caller must answer its own call before the model continues"
    );
    let mcp_call = items_of_type(output, "mcp_call")[0];
    assert!(
        mcp_call["error"]
            .as_str()
            .is_none_or(|e| !e.contains("exposed tool map")),
        "{mcp_call}"
    );
}

#[tokio::test]
async fn non_streaming_function_call_does_not_count_against_max_tool_calls() {
    let (result, _) = run_non_streaming(
        vec![response_json(vec![
            function_call(MCP_TOOL, "call_m"),
            function_call(FN_TOOL, "call_f"),
        ])],
        Some(1),
    )
    .await;

    assert!(result.get("incomplete_details").is_none_or(Value::is_null));
    let output = result["output"].as_array().unwrap();
    assert_eq!(names(&items_of_type(output, "mcp_call")), [MCP_TOOL]);
    assert_eq!(names(&items_of_type(output, "function_call")), [FN_TOOL]);
}

fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {}\n\n",
                event["type"].as_str().unwrap(),
                event
            )
        })
        .collect()
}

fn created() -> Vec<Value> {
    let response = json!({"id": "resp_up", "object": "response", "status": "in_progress",
        "model": "gpt-test", "output": []});
    vec![
        json!({"type": "response.created", "sequence_number": 0, "response": response}),
        json!({"type": "response.in_progress", "sequence_number": 1, "response": response}),
    ]
}

fn completed(output: Vec<Value>) -> Value {
    json!({"type": "response.completed", "sequence_number": 99, "response": {
        "id": "resp_up", "object": "response", "status": "completed",
        "model": "gpt-test", "output": output}})
}

/// Stream one function call as the upstream does: added, delta, args done, done.
fn streamed_call(name: &str, call_id: &str, index: usize) -> Vec<Value> {
    let item = function_call(name, call_id);
    let mut added = item.clone();
    added["arguments"] = json!("");
    added["status"] = json!("in_progress");
    vec![
        json!({"type": "response.output_item.added", "sequence_number": 2, "output_index": index, "item": added}),
        json!({"type": "response.function_call_arguments.delta", "sequence_number": 3,
            "output_index": index, "item_id": item["id"], "delta": "{}"}),
        json!({"type": "response.function_call_arguments.done", "sequence_number": 4,
            "output_index": index, "item_id": item["id"], "arguments": "{}"}),
        json!({"type": "response.output_item.done", "sequence_number": 5, "output_index": index, "item": item}),
    ]
}

fn streamed_message(index: usize) -> Vec<Value> {
    vec![
        json!({"type": "response.output_item.added", "sequence_number": 2, "output_index": index,
            "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []}}),
        json!({"type": "response.output_item.done", "sequence_number": 5, "output_index": index,
            "item": message("done")}),
    ]
}

fn stream_body(parts: Vec<Vec<Value>>, final_output: Vec<Value>) -> String {
    let mut events = created();
    for part in parts {
        events.extend(part);
    }
    events.push(completed(final_output));
    sse(&events)
}

/// Run the streaming loop and return the client-visible SSE events.
async fn run_streaming(scripted: Vec<String>) -> (Vec<Value>, Upstream) {
    let (url, up) = spawn_upstream(scripted).await;
    let orchestrator = Arc::new(orchestrator_with_mcp_tool().await);
    let req = StreamingRequest {
        url,
        payload: payload(true),
        original_body: request(None),
        previous_response_id: None,
        existing_mcp_list_tools_labels: Vec::new(),
        storage: StorageHandles {
            response: Arc::new(MemoryResponseStorage::new()),
            conversation: Arc::new(MemoryConversationStorage::new()),
            conversation_item: Arc::new(MemoryConversationItemStorage::new()),
            request_context: None,
            tenant: None,
        },
    };
    let response = handle_streaming_with_tool_interception(
        &client(),
        None,
        None,
        req,
        &orchestrator,
        FormatRegistry::new(),
        vec![binding()],
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let events = text
        .split("\n\n")
        .filter_map(|block| block.lines().find_map(|l| l.strip_prefix("data: ")))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .collect();
    (events, up)
}

fn event_types(events: &[Value]) -> Vec<String> {
    events
        .iter()
        .map(|e| e["type"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn final_output(events: &[Value]) -> Vec<Value> {
    events
        .iter()
        .rev()
        .find(|e| e["type"] == "response.completed")
        .and_then(|e| e["response"]["output"].as_array().cloned())
        .expect("response.completed with output")
}

#[tokio::test]
async fn streaming_function_only_turn_forwards_the_call_untouched() {
    let call = function_call(FN_TOOL, "call_f");
    let (events, up) = run_streaming(vec![stream_body(
        vec![streamed_call(FN_TOOL, "call_f", 0)],
        vec![call],
    )])
    .await;

    let types = event_types(&events);
    assert!(
        types
            .iter()
            .any(|t| t == "response.function_call_arguments.done"),
        "{types:?}"
    );
    assert!(
        types.iter().all(|t| !t.contains("mcp_call")),
        "a caller function must not become an mcp_call: {types:?}"
    );
    let output = final_output(&events);
    assert_eq!(names(&items_of_type(&output, "function_call")), [FN_TOOL]);
    assert_eq!(up.seen.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn streaming_mcp_only_turn_executes_and_resumes() {
    let (events, up) = run_streaming(vec![
        stream_body(
            vec![streamed_call(MCP_TOOL, "call_m", 0)],
            vec![function_call(MCP_TOOL, "call_m")],
        ),
        stream_body(vec![streamed_message(0)], vec![message("done")]),
    ])
    .await;

    let output = final_output(&events);
    assert_eq!(names(&items_of_type(&output, "mcp_call")), [MCP_TOOL]);
    assert!(
        items_of_type(&output, "function_call").is_empty(),
        "{output:?}"
    );
    assert_eq!(items_of_type(&output, "message").len(), 1);
    assert_eq!(up.seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn streaming_mixed_turn_runs_mcp_and_returns_the_function_call() {
    // Both orders: the caller's call can stream before or after the MCP call.
    for caller_first in [false, true] {
        let mut parts = vec![
            streamed_call(MCP_TOOL, "call_m", if caller_first { 1 } else { 0 }),
            streamed_call(FN_TOOL, "call_f", if caller_first { 0 } else { 1 }),
        ];
        if caller_first {
            parts.reverse();
        }
        let (events, up) = run_streaming(vec![stream_body(
            parts,
            vec![
                function_call(MCP_TOOL, "call_m"),
                function_call(FN_TOOL, "call_f"),
            ],
        )])
        .await;

        let types = event_types(&events);
        let output = final_output(&events);
        assert_eq!(
            names(&items_of_type(&output, "mcp_call")),
            [MCP_TOOL],
            "caller_first={caller_first}: {output:?}"
        );
        let calls = items_of_type(&output, "function_call");
        assert_eq!(
            names(&calls),
            [FN_TOOL],
            "caller_first={caller_first}: {output:?}"
        );
        assert_eq!(calls[0]["call_id"], "call_f");
        assert_eq!(
            up.seen.lock().unwrap().len(),
            1,
            "caller_first={caller_first}: the caller must answer first: {types:?}"
        );
        // The caller's call keeps its own streaming events.
        let fn_args_done: Vec<_> = events
            .iter()
            .filter(|e| e["type"] == "response.function_call_arguments.done")
            .collect();
        assert_eq!(
            fn_args_done.len(),
            1,
            "caller_first={caller_first}: {types:?}"
        );
        assert_eq!(fn_args_done[0]["item_id"], "fc_call_f");
    }
}
