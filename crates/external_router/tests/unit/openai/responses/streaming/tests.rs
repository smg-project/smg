use openai_protocol::responses::ResponseInput;

use super::*;

/// The terminal event says `completed` on the wire whatever the shaped
/// value carries, and the value is left as it was once the event is out:
/// an upstream `incomplete` stays `incomplete` for the record, a value
/// without a status gets none.
#[tokio::test]
async fn the_terminal_event_writes_completed_on_the_wire_and_leaves_the_value_its_own_status() {
    for (own_status, expected_after) in [
        (Some("incomplete"), Some(json!("incomplete"))),
        (None, None),
    ] {
        let mut final_response = json!({"id": "resp_1", "object": "response", "output": []});
        if let Some(status) = own_status {
            final_response["status"] = json!(status);
        }
        let (tx, mut rx) = sse_channel();
        let mut enc = SseEncoder::new();
        let mut sequence_number = 3;
        assert!(
            send_final_response_event(&mut final_response, &tx, &mut enc, &mut sequence_number)
                .await
        );
        drop(tx);
        let frame = rx.recv().await.expect("the terminal").expect("a frame");
        let frame = String::from_utf8(frame.to_vec()).expect("utf-8");
        let data = frame
            .strip_prefix("event: response.completed\ndata: ")
            .expect("the completed event");
        let event: Value = serde_json::from_str(data.trim_end()).expect("json");
        assert_eq!(
            event["response"]["status"],
            json!("completed"),
            "{own_status:?}"
        );
        assert_eq!(sequence_number, 4, "{own_status:?}");
        assert_eq!(
            final_response.get("status").cloned(),
            expected_after,
            "the value keeps its own status once the event is out"
        );
    }
}

/// The terminal of a loop stopped at the request's tool-call allowance
/// is the terminal of an answered loop, `response.completed` with the
/// response `completed` at the stream's own sequence number, plus the
/// note the non-streamed loop leaves on the same stop,
/// `incomplete_details.reason` `max_tool_calls`; nothing else differs
/// between the two frames.
#[tokio::test]
async fn the_tool_call_limit_terminal_is_the_completed_terminal_with_the_non_streamed_note() {
    let request = ResponsesRequest {
        model: "m".to_string(),
        input: ResponseInput::Text("hi".to_string()),
        ..Default::default()
    };
    let ctx = StreamingEventContext {
        original_request: &request,
        previous_response_id: None,
        session: None,
        mcp_format_registry: None,
    };
    let mut state = ToolLoopState::new(Vec::new());
    let created =
        "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\
             \"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1,\
             \"model\":\"m\",\"status\":\"in_progress\",\"output\":[]}}";

    let mut frames = Vec::new();
    for end in [ToolLoopEnd::Answered, ToolLoopEnd::ToolCallLimit] {
        let (tx, mut rx) = sse_channel();
        let mut enc = SseEncoder::new();
        let mut sequence_number = 7;
        let mut handler = StreamingToolHandler::with_starting_index(0);
        handler.accumulator.ingest_block(created);
        let preserved_id = handler.original_response_id().map(str::to_string);
        let mut final_response = handler
            .accumulator
            .into_final_response()
            .expect("the final response");
        shape_final_response(
            &mut final_response,
            preserved_id.as_deref(),
            &mut state,
            &ctx,
            end,
        );
        assert!(
            send_final_response_event(&mut final_response, &tx, &mut enc, &mut sequence_number,)
                .await
        );
        drop(tx);
        assert_eq!(sequence_number, 8, "{end:?}");
        let frame = rx.recv().await.expect("the terminal").expect("a frame");
        assert!(rx.recv().await.is_none(), "one frame, {end:?}");
        frames.push(String::from_utf8(frame.to_vec()).expect("utf-8"));
    }

    let data = |frame: &str| -> Value {
        let data = frame
            .strip_prefix("event: response.completed\ndata: ")
            .expect("the completed event");
        serde_json::from_str(data.trim_end()).expect("json")
    };
    let answered = data(&frames[0]);
    let mut limited = data(&frames[1]);
    assert_eq!(limited["type"], "response.completed");
    assert_eq!(limited["sequence_number"], 7);
    assert_eq!(limited["response"]["status"], "completed");
    assert_eq!(
        limited["response"]["incomplete_details"],
        json!({ "reason": "max_tool_calls" })
    );
    assert!(answered["response"].get("incomplete_details").is_none());
    limited["response"]
        .as_object_mut()
        .expect("an object")
        .remove("incomplete_details");
    assert_eq!(answered, limited, "the note is the only difference");
}

/// The error event that ends a failed stream is the frame every router
/// writes for one, at the stream's own sequence number, which it
/// advances; nothing follows it.
#[tokio::test]
async fn the_error_event_is_the_shared_frame_at_the_streams_sequence_number() {
    let (tx, mut rx) = sse_channel();
    let mut enc = SseEncoder::new();
    let mut sequence_number = 4;
    send_error_event(
        &tx,
        &mut enc,
        &mut sequence_number,
        STREAM_ERROR,
        "Stream error: connection reset",
    )
    .await;
    drop(tx);
    assert_eq!(sequence_number, 5);
    let frame = rx.recv().await.expect("the error frame").expect("a frame");
    assert_eq!(
        frame,
        SseEncoder::new().encode_responses_error(
            "stream_error",
            "Stream error: connection reset",
            4
        )
    );
    assert!(rx.recv().await.is_none(), "nothing follows the error event");
}

/// An upstream status answered after the stream began: the body's own
/// code (else its type) and message when it carries the error envelope;
/// the status's error type, with the status and the body, when it does
/// not.
#[test]
fn an_upstream_status_keeps_the_bodys_code_and_message_when_it_has_them() {
    let cases = [
        (
            StatusCode::SERVICE_UNAVAILABLE,
            r#"{"error":{"code":"overloaded","type":"server_error","message":"busy"}}"#,
            "overloaded",
            "busy",
        ),
        (
            StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"type":"rate_limit_error","message":"slow down"}}"#,
            "rate_limit_error",
            "slow down",
        ),
        (
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"bad input"}}"#,
            "invalid_request_error",
            "bad input",
        ),
        (
            StatusCode::BAD_GATEWAY,
            "bad gateway",
            "server_error",
            "Upstream error 502 Bad Gateway: bad gateway",
        ),
    ];
    for (status, body, code, message) in cases {
        assert_eq!(
            upstream_status_error(status, body),
            (code.to_string(), message.to_string()),
            "{status} {body}"
        );
    }
}

/// The routed-worker header of a streaming answer names the worker
/// masked: the header reaches every client of the stream and the access
/// log copies it, so a credential in a worker URL never reaches either.
#[tokio::test]
async fn the_routed_worker_header_names_the_worker_masked() {
    use std::sync::Arc;

    use axum::{
        http::{header::CONTENT_TYPE, StatusCode},
        routing::any,
        Router,
    };
    use openai_protocol::responses::ResponsesRequest;
    use serde_json::json;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };

    use crate::{
        openai::context::StorageHandles,
        test_support::{http_client, StubWorker},
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a listener");
    let port = listener.local_addr().expect("an address").port();
    #[expect(
        clippy::disallowed_methods,
        reason = "test stub upstream answering one event stream; aborted at the test's end"
    )]
    let upstream = tokio::spawn(async move {
        let app = Router::new().fallback(any(|| async {
            (
                [(CONTENT_TYPE, "text/event-stream")],
                "event: response.completed\ndata: {}\n\n",
            )
        }));
        axum::serve(listener, app).await.expect("the stub serves");
    });
    let worker_url = format!("http://user:secret@127.0.0.1:{port}");
    let request: ResponsesRequest =
        serde_json::from_value(json!({"model": "m", "input": "hi", "stream": true}))
            .expect("a request");
    let req = StreamingRequest {
        url: format!("{worker_url}/v1/responses"),
        payload: json!({}),
        original_body: request,
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

    let response = handle_simple_streaming_passthrough(
        &http_client(),
        &StubWorker::at(&worker_url),
        None,
        req,
    )
    .await;
    upstream.abort();

    assert_eq!(response.status(), StatusCode::OK);
    let header = response
        .headers()
        .get("x-smg-routed-worker-id")
        .and_then(|value| value.to_str().ok())
        .expect("the routed-worker header");
    assert!(!header.contains("secret"), "{header}");
    assert_eq!(header, format!("http://***@127.0.0.1:{port}"));
}

/// A streamed forward nothing answers: the caller's 502 says why without
/// the worker's address. The HTTP client's own error text ends with the
/// request URL.
#[tokio::test]
async fn a_failed_streamed_forward_answers_without_the_workers_address() {
    use axum::http::StatusCode;

    use crate::test_support::{text_of, RefusedAddress};

    let refused = RefusedAddress::reserve();

    let response = streamed_answer_from(&format!("http://{}", refused.addr())).await;

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = text_of(response).await;
    assert!(!body.contains(&refused.addr().to_string()), "{body}");
}

/// A refusal whose body is lost mid-answer: the caller's answer says so
/// without the worker's address. The HTTP client's text of a failed body
/// read ends with the request URL.
#[tokio::test]
async fn an_unread_refusal_is_answered_without_the_workers_address() {
    use axum::http::StatusCode;

    use crate::test_support::{answering, text_of};

    let addr = answering(500, "short", true).await;

    let response = streamed_answer_from(&format!("http://{addr}")).await;

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = text_of(response).await;
    assert_eq!(
        body,
        "Failed to read upstream error body: error decoding response body"
    );
}

/// The plain streaming path's answer to a streamed request for the worker
/// at `worker_url`, over empty stores.
async fn streamed_answer_from(worker_url: &str) -> Response {
    use std::sync::Arc;

    use openai_protocol::responses::ResponsesRequest;
    use serde_json::json;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };

    use crate::{
        openai::context::StorageHandles,
        test_support::{http_client, StubWorker},
    };

    let request: ResponsesRequest =
        serde_json::from_value(json!({"model": "m", "input": "hi", "stream": true}))
            .expect("a request");
    let req = StreamingRequest {
        url: format!("{worker_url}/v1/responses"),
        payload: json!({}),
        original_body: request,
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

    handle_simple_streaming_passthrough(&http_client(), &StubWorker::at(worker_url), None, req)
        .await
}

/// A header the upstream sent more than once reaches the client of a
/// streamed answer with every value, in order: the relay's copy of the
/// preserved headers appends a pair per value instead of keeping the last.
#[tokio::test]
async fn the_streamed_answer_keeps_every_value_of_a_repeated_upstream_header() {
    use std::sync::Arc;

    use axum::{
        http::{
            header::{CONTENT_TYPE, SET_COOKIE},
            StatusCode,
        },
        response::AppendHeaders,
        routing::any,
        Router,
    };
    use openai_protocol::responses::ResponsesRequest;
    use serde_json::json;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };

    use crate::{
        openai::context::StorageHandles,
        test_support::{http_client, StubWorker},
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a listener");
    let port = listener.local_addr().expect("an address").port();
    #[expect(
        clippy::disallowed_methods,
        reason = "test stub upstream answering one event stream; aborted at the test's end"
    )]
    let upstream = tokio::spawn(async move {
        // `AppendHeaders`: the plain header array would insert and keep one cookie.
        let app = Router::new().fallback(any(|| async {
            (
                AppendHeaders([
                    (CONTENT_TYPE, "text/event-stream"),
                    (SET_COOKIE, "a=1"),
                    (SET_COOKIE, "b=2"),
                ]),
                "event: response.completed\ndata: {}\n\n",
            )
        }));
        axum::serve(listener, app).await.expect("the stub serves");
    });
    let worker_url = format!("http://127.0.0.1:{port}");
    let request: ResponsesRequest =
        serde_json::from_value(json!({"model": "m", "input": "hi", "stream": true}))
            .expect("a request");
    let req = StreamingRequest {
        url: format!("{worker_url}/v1/responses"),
        payload: json!({}),
        original_body: request,
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

    let response = handle_simple_streaming_passthrough(
        &http_client(),
        &StubWorker::at(&worker_url),
        None,
        req,
    )
    .await;
    upstream.abort();

    assert_eq!(response.status(), StatusCode::OK);
    let cookies: Vec<_> = response
        .headers()
        .get_all(SET_COOKIE)
        .iter()
        .map(|value| value.to_str().expect("an ascii cookie"))
        .collect();
    assert_eq!(cookies, ["a=1", "b=2"]);
}

/// The relay's inputs: a stream of chunks, one SSE block per chunk.
fn chunks(blocks: &[&str]) -> impl Stream<Item = reqwest::Result<Bytes>> + Unpin {
    let frames: Vec<reqwest::Result<Bytes>> = blocks
        .iter()
        .map(|block| Ok(Bytes::from(format!("{block}\n\n"))))
        .collect();
    futures_util::stream::iter(frames)
}

const CREATED: &str = "event: response.created\ndata: {\"type\":\"response.created\",\
                           \"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\
                           \"response\",\"created_at\":1,\"model\":\"m\",\"status\":\
                           \"in_progress\",\"output\":[]}}";
const IN_PROGRESS: &str = "event: response.in_progress\ndata: {\"type\":\
                               \"response.in_progress\",\"sequence_number\":1,\"response\":{\"id\":\
                               \"resp_1\",\"object\":\"response\",\"created_at\":1,\"model\":\"m\",\
                               \"status\":\"in_progress\",\"output\":[]}}";
const DELTA: &str = "event: response.output_text.delta\ndata: {\"type\":\
                         \"response.output_text.delta\",\"sequence_number\":2,\"item_id\":\"msg_1\",\
                         \"output_index\":0,\"content_index\":0,\"delta\":\"hi\"}";
const COMPLETED: &str = "event: response.completed\ndata: {\"type\":\"response.completed\",\
                             \"sequence_number\":3,\"response\":{\"id\":\"resp_1\",\"object\":\
                             \"response\",\"created_at\":1,\"model\":\"m\",\"status\":\
                             \"completed\",\"output\":[]}}";
const CALL_ADDED: &str = "event: response.output_item.added\ndata: {\"type\":\
                              \"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\
                              \"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\
                              \"call_1\",\"name\":\"lookup\",\"arguments\":\"\"}}";
const CALL_DONE: &str = "event: response.function_call_arguments.done\ndata: {\"type\":\
                             \"response.function_call_arguments.done\",\"sequence_number\":2,\
                             \"item_id\":\"fc_1\",\"output_index\":0,\"arguments\":\"{}\"}";

/// Relays `blocks` through a fresh context with no session; the client
/// reads the frames as they come, or is gone before the first one. Returns
/// the relay's end, the frames, the sequence number and the count of
/// blocks the relay left unread.
async fn relayed(
    blocks: &[&str],
    flags: &mut RelayFlags,
    handler: &mut StreamingToolHandler,
    connected: bool,
) -> (RelayEnd, Vec<String>, u64, usize) {
    let request = ResponsesRequest {
        model: "m".to_string(),
        input: ResponseInput::Text("hi".to_string()),
        ..Default::default()
    };
    let ctx = StreamingEventContext {
        original_request: &request,
        previous_response_id: None,
        session: None,
        mcp_format_registry: None,
    };
    let (tx, rx) = sse_channel();
    let rx = connected.then_some(rx);
    let mut encoder = SseEncoder::new();
    let mut sequence_number = 0;
    let mut upstream = chunks(blocks);
    let relay = async {
        let end = relay_upstream_stream(
            &mut upstream,
            handler,
            flags,
            &[],
            &tx,
            &mut encoder,
            &ctx,
            &mut sequence_number,
        )
        .await;
        drop(tx);
        end
    };
    let read = async {
        let mut frames = Vec::new();
        if let Some(mut rx) = rx {
            while let Some(frame) = rx.recv().await {
                let frame = frame.expect("a frame");
                frames.push(String::from_utf8(frame.to_vec()).expect("utf-8"));
            }
        }
        frames
    };
    let (end, frames) = tokio::join!(relay, read);
    let unread = upstream.count().await;
    (end, frames, sequence_number, unread)
}

/// The first stream's events reach the client in order, re-stamped from
/// the stream's own sequence number; the upstream's `response.completed`
/// is not among them (the loop's end sends the terminal); the relay ends
/// answered with the response id kept and the list-tools step taken.
#[tokio::test]
async fn the_first_stream_is_forwarded_and_ends_answered() {
    let mut flags = RelayFlags {
        first_iteration: true,
        mcp_list_tools_sent: false,
    };
    let mut handler = StreamingToolHandler::with_starting_index(0);
    let (end, frames, sequence_number, unread) = relayed(
        &[CREATED, IN_PROGRESS, DELTA, COMPLETED],
        &mut flags,
        &mut handler,
        true,
    )
    .await;
    assert_eq!(end, RelayEnd::Answered);
    assert_eq!(unread, 0, "the stream is read to its end");
    let events: Vec<&str> = frames
        .iter()
        .map(|frame| frame.split('\n').next().expect("an event line"))
        .collect();
    assert_eq!(
        events,
        [
            "event: response.created",
            "event: response.in_progress",
            "event: response.output_text.delta"
        ]
    );
    assert_eq!(sequence_number, 3);
    assert_eq!(handler.original_response_id(), Some("resp_1"));
    assert!(flags.mcp_list_tools_sent);
}

/// A resumed stream (after a tool round) skips its own `response.created`
/// and `response.in_progress`: the client saw the first stream's.
#[tokio::test]
async fn a_resumed_stream_skips_created_and_in_progress() {
    let mut flags = RelayFlags {
        first_iteration: false,
        mcp_list_tools_sent: true,
    };
    let mut handler = StreamingToolHandler::with_starting_index(1);
    let (end, frames, sequence_number, unread) = relayed(
        &[CREATED, IN_PROGRESS, DELTA, COMPLETED],
        &mut flags,
        &mut handler,
        true,
    )
    .await;
    assert_eq!(end, RelayEnd::Answered);
    assert_eq!(unread, 0, "the stream is read to its end");
    assert_eq!(frames.len(), 1, "{frames:?}");
    assert!(frames[0].starts_with("event: response.output_text.delta"));
    assert_eq!(sequence_number, 1);
}

/// A function call the upstream completes ends the relay with the call
/// pending in the handler: the loop's tool phase runs it, and the blocks
/// after the trigger are not read by this relay.
#[tokio::test]
async fn a_completed_tool_call_ends_the_relay_with_the_call_pending() {
    let mut flags = RelayFlags {
        first_iteration: true,
        mcp_list_tools_sent: false,
    };
    let mut handler = StreamingToolHandler::with_starting_index(0);
    let (end, _frames, _, unread) = relayed(
        &[CREATED, CALL_ADDED, CALL_DONE, COMPLETED],
        &mut flags,
        &mut handler,
        true,
    )
    .await;
    assert_eq!(end, RelayEnd::ToolCalls);
    assert_eq!(unread, 1, "the completed after the trigger is not read");
    assert_eq!(handler.take_pending_calls().len(), 1);
}

/// A client that is gone before the first frame stops the relay at that
/// frame: the frame is numbered, its send fails, nothing is read further
/// and nothing reaches the client.
#[tokio::test]
async fn a_gone_client_stops_the_relay() {
    let mut flags = RelayFlags {
        first_iteration: true,
        mcp_list_tools_sent: false,
    };
    let mut handler = StreamingToolHandler::with_starting_index(0);
    let (end, frames, sequence_number, unread) = relayed(
        &[CREATED, DELTA, COMPLETED],
        &mut flags,
        &mut handler,
        false,
    )
    .await;
    assert_eq!(end, RelayEnd::Stopped);
    assert_eq!(unread, 2, "the relay stops at the first refused send");
    assert!(frames.is_empty());
    assert_eq!(sequence_number, 1);
}

/// The whole interception path against a stub upstream that answers one
/// four-event stream and its own end sentinel: the client reads the
/// stream's events, the loop's terminal and one sentinel, the loop's.
#[tokio::test]
async fn a_tool_interception_stream_ends_with_the_terminal_and_the_sentinel() {
    use std::sync::Arc;

    use axum::{
        http::{header::CONTENT_TYPE, StatusCode},
        routing::any,
        Router,
    };
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };
    use smg_mcp::McpConfig;

    use crate::{
        openai::context::StorageHandles, openai_bridge::FormatRegistry, test_support::http_client,
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a listener");
    let port = listener.local_addr().expect("an address").port();
    let body = [CREATED, IN_PROGRESS, DELTA, COMPLETED, "data: [DONE]"].join("\n\n") + "\n\n";
    #[expect(
        clippy::disallowed_methods,
        reason = "test stub upstream answering one event stream per request; aborted at the test's end"
    )]
    let upstream = tokio::spawn(async move {
        let app = Router::new().fallback(any(move || {
            let body = body.clone();
            async move { ([(CONTENT_TYPE, "text/event-stream")], body) }
        }));
        axum::serve(listener, app).await.expect("the stub serves");
    });
    let orchestrator = Arc::new(
        McpOrchestrator::new(McpConfig::default())
            .await
            .expect("an orchestrator with no servers"),
    );
    let client = http_client();
    let storage = StorageHandles {
        response: Arc::new(MemoryResponseStorage::new()),
        conversation: Arc::new(MemoryConversationStorage::new()),
        conversation_item: Arc::new(MemoryConversationItemStorage::new()),
        request_context: None,
        tenant: None,
    };
    let request: ResponsesRequest =
        serde_json::from_value(json!({"model": "m", "input": "hi", "stream": true}))
            .expect("a request");
    let req = StreamingRequest {
        url: format!("http://127.0.0.1:{port}/v1/responses"),
        payload: json!({"model": "m", "input": "hi", "stream": true}),
        original_body: request,
        previous_response_id: None,
        existing_mcp_list_tools_labels: Vec::new(),
        storage: StorageHandles {
            response: Arc::clone(&storage.response),
            conversation: Arc::clone(&storage.conversation),
            conversation_item: Arc::clone(&storage.conversation_item),
            request_context: None,
            tenant: None,
        },
    };
    let response = handle_streaming_with_tool_interception(
        &client,
        None,
        None,
        req,
        &orchestrator,
        FormatRegistry::new(),
        Vec::new(),
    );
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the stream to its end");
    let text = String::from_utf8(bytes.to_vec()).expect("utf-8");
    assert!(text.contains("event: response.created\n"), "{text}");
    assert!(
        text.contains("event: response.output_text.delta\n"),
        "{text}"
    );
    assert!(text.contains("event: response.completed\n"), "{text}");
    assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
    // A reader that stops at the first sentinel has read the terminal.
    assert_eq!(text.matches("data: [DONE]").count(), 1, "{text}");
    upstream.abort();
}

/// The streaming MCP relay hands its request payload to the loop it
/// spawns instead of copying it: what the call itself allocates before
/// the loop runs a single turn (the task, the channel, the loop's own
/// state) does not grow with the request body, where a copy of the whole
/// JSON tree would grow with it byte for byte. The stub upstream's
/// received body is the one serialization the turn needs, as before. The
/// counts are printed for the Numbers table.
#[tokio::test]
#[expect(
    clippy::print_stderr,
    reason = "the exchange's allocation counts, read from the test's output by the measurement of the relay"
)]
async fn a_streaming_mcp_request_hands_its_payload_to_the_loop_without_a_copy() {
    use std::sync::{Arc, Mutex};

    use axum::{
        extract::Request,
        http::{header::CONTENT_TYPE, StatusCode},
        routing::any,
        Router,
    };
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };
    use smg_mcp::McpConfig;

    use crate::{
        openai::context::StorageHandles,
        openai_bridge::FormatRegistry,
        test_support::{counting_alloc, http_client},
    };

    let orchestrator = Arc::new(
        McpOrchestrator::new(McpConfig::default())
            .await
            .expect("an orchestrator with no servers"),
    );
    let client = http_client();
    let answer = [CREATED, IN_PROGRESS, COMPLETED].join("\n\n") + "\n\n";

    // (input length, body length, bytes the call allocated) per request.
    let mut by_the_call = Vec::new();
    for input_len in [2usize, 64 * 1024] {
        let input = "x".repeat(input_len);
        let payload = json!({"model": "m", "input": input, "stream": true});
        let body_len = serde_json::to_vec(&payload).expect("the body").len();
        let request: ResponsesRequest = serde_json::from_value(payload.clone()).expect("a request");

        // The stub keeps the body of the one request it answers.
        let received: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a listener");
        let port = listener.local_addr().expect("an address").port();
        let (bodies, answer) = (Arc::clone(&received), answer.clone());
        #[expect(
            clippy::disallowed_methods,
            reason = "test stub upstream answering one event stream per request; aborted at the test's end"
        )]
        let upstream = tokio::spawn(async move {
            let app = Router::new().fallback(any(move |req: Request| {
                let (bodies, answer) = (Arc::clone(&bodies), answer.clone());
                async move {
                    let body = axum::body::to_bytes(req.into_body(), usize::MAX)
                        .await
                        .expect("the request body");
                    bodies.lock().expect("the bodies").push(body.to_vec());
                    ([(CONTENT_TYPE, "text/event-stream")], answer)
                }
            }));
            axum::serve(listener, app).await.expect("the stub serves");
        });
        let req = StreamingRequest {
            url: format!("http://127.0.0.1:{port}/v1/responses"),
            payload,
            original_body: request,
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

        // The call spawns the loop and returns; on this current-thread
        // runtime the loop has not run yet, so the counters move only by
        // what the call itself allocated.
        let (calls_before, bytes_before) = (counting_alloc::calls(), counting_alloc::bytes());
        let response = handle_streaming_with_tool_interception(
            &client,
            None,
            None,
            req,
            &orchestrator,
            FormatRegistry::new(),
            Vec::new(),
        );
        let bytes_by_the_call = counting_alloc::bytes() - bytes_before;
        assert_eq!(response.status(), StatusCode::OK);
        let stream = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("the stream to its end");
        let (exchange_calls, exchange_bytes) = (
            counting_alloc::calls() - calls_before,
            counting_alloc::bytes() - bytes_before,
        );
        upstream.abort();

        eprintln!(
            "input {input_len} bytes (body {body_len} bytes): the call allocated {bytes_by_the_call} bytes before the loop ran; the exchange {exchange_calls} allocation calls, {exchange_bytes} bytes"
        );
        let stream = String::from_utf8(stream.to_vec()).expect("utf-8");
        assert!(stream.ends_with("data: [DONE]\n\n"), "{stream}");
        let bodies = received.lock().expect("the bodies");
        assert_eq!(bodies.len(), 1, "one turn, one upstream request");
        let on_the_wire: Value = serde_json::from_slice(&bodies[0]).expect("the body is JSON");
        assert_eq!(on_the_wire["input"].as_str().map(str::len), Some(input_len));
        assert_eq!(on_the_wire["model"], json!("m"));
        by_the_call.push((input_len, body_len, bytes_by_the_call));
    }

    let [(_, _, small), (_, large_body, large)] = by_the_call[..] else {
        panic!("two requests measured, {by_the_call:?}");
    };
    let growth = large.saturating_sub(small);
    assert!(
        growth * 4 < large_body,
        "the call's allocation grew by {growth} bytes for a {large_body}-byte body (small request {small} bytes, large {large}): the payload was copied, not moved"
    );
}

/// A tool loop whose upstream answers with a large final response (64
/// message items of 2 KiB) ends with the same items on the wire, in the
/// `response.completed` event, and in the stored record, the record
/// carrying the upstream's status. The allocation count of the whole
/// exchange is printed for the measurement of the loop's end.
#[tokio::test]
async fn a_large_final_response_is_streamed_and_stored_with_the_same_items() {
    use std::sync::Arc;

    use axum::{
        http::{header::CONTENT_TYPE, StatusCode},
        routing::any,
        Router,
    };
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage, ResponseId,
    };
    use smg_mcp::McpConfig;

    use crate::{
        openai::context::StorageHandles,
        openai_bridge::FormatRegistry,
        test_support::{counting_alloc, http_client},
    };

    let text = "x".repeat(2048);
    let items: Vec<Value> = (0..64)
        .map(|i| {
            json!({
                "type": "message", "id": format!("msg_{i}"), "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]
            })
        })
        .collect();
    let completed = format!(
        "event: response.completed\ndata: {}",
        json!({
            "type": "response.completed", "sequence_number": 3,
            "response": {
                "id": "resp_1", "object": "response", "created_at": 1, "model": "m",
                "status": "incomplete", "output": items
            }
        })
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a listener");
    let port = listener.local_addr().expect("an address").port();
    let body = [CREATED, IN_PROGRESS, completed.as_str()].join("\n\n") + "\n\n";
    #[expect(
        clippy::disallowed_methods,
        reason = "test stub upstream answering one event stream per request; aborted at the test's end"
    )]
    let upstream = tokio::spawn(async move {
        let app = Router::new().fallback(any(move || {
            let body = body.clone();
            async move { ([(CONTENT_TYPE, "text/event-stream")], body) }
        }));
        axum::serve(listener, app).await.expect("the stub serves");
    });
    let orchestrator = Arc::new(
        McpOrchestrator::new(McpConfig::default())
            .await
            .expect("an orchestrator with no servers"),
    );
    let client = http_client();
    let storage = StorageHandles {
        response: Arc::new(MemoryResponseStorage::new()),
        conversation: Arc::new(MemoryConversationStorage::new()),
        conversation_item: Arc::new(MemoryConversationItemStorage::new()),
        request_context: None,
        tenant: None,
    };
    let request: ResponsesRequest =
        serde_json::from_value(json!({"model": "m", "input": "hi", "stream": true}))
            .expect("a request");
    let req = StreamingRequest {
        url: format!("http://127.0.0.1:{port}/v1/responses"),
        payload: json!({"model": "m", "input": "hi", "stream": true}),
        original_body: request,
        previous_response_id: None,
        existing_mcp_list_tools_labels: Vec::new(),
        storage: StorageHandles {
            response: Arc::clone(&storage.response),
            conversation: Arc::clone(&storage.conversation),
            conversation_item: Arc::clone(&storage.conversation_item),
            request_context: None,
            tenant: None,
        },
    };

    let allocations_before = counting_alloc::calls();
    let response = handle_streaming_with_tool_interception(
        &client,
        None,
        None,
        req,
        &orchestrator,
        FormatRegistry::new(),
        Vec::new(),
    );
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the stream to its end");
    let stored = storage
        .response
        .get_response(&ResponseId::from("resp_1"))
        .await
        .expect("the storage answers")
        .expect("the record");
    let allocations = counting_alloc::calls() - allocations_before;
    #[expect(
        clippy::print_stderr,
        reason = "the exchange's allocation count, read from the test's output by the measurement of the loop's end"
    )]
    {
        eprintln!("allocations during the streamed loop with 64 items: {allocations}");
    }
    upstream.abort();

    let stream = String::from_utf8(bytes.to_vec()).expect("utf-8");
    let completed_frame = stream
        .split("\n\n")
        .find(|frame| frame.starts_with("event: response.completed\n"))
        .expect("the terminal event");
    let data = completed_frame
        .strip_prefix("event: response.completed\ndata: ")
        .expect("the terminal's data");
    let event: Value = serde_json::from_str(data).expect("the terminal's JSON");
    let on_the_wire = &event["response"];
    assert_eq!(on_the_wire["id"], json!("resp_1"));
    assert_eq!(on_the_wire["status"], json!("completed"));
    assert_eq!(on_the_wire["output"].as_array().map(Vec::len), Some(64));
    assert_eq!(
        stored.raw_response["output"], on_the_wire["output"],
        "the record and the wire carry the same items"
    );
    assert_eq!(
        stored.raw_response["status"],
        json!("incomplete"),
        "the record keeps the status the upstream reported; the wire said completed"
    );
    assert!(stream.ends_with("data: [DONE]\n\n"), "{stream}");
}
