use smg_external_router::retry::REQUEST_TIMEOUT_CODE;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::*;
use crate::routers::grpc::common::responses::utils::{
    assert_responses_api_usage, namespace_test_request, responses_api_test_usage,
};

/// Runs chat SSE frames through the Responses transform as an unstored
/// request over empty memory storages (no tenant, no request context) and
/// hands back the client's receiver once the transform has returned.
async fn transformed_chat_stream(
    frames: Vec<String>,
) -> mpsc::Receiver<Result<Bytes, std::io::Error>> {
    let body = Body::from_stream(futures_util::stream::iter(
        frames.into_iter().map(Ok::<_, std::convert::Infallible>),
    ));
    let (tx, rx) = mpsc::channel(64);
    process_and_transform_sse_stream(
        body,
        Arc::new(ResponsesRequest {
            store: Some(false),
            ..Default::default()
        }),
        None,
        None,
        Arc::new(smg_data_connector::MemoryResponseStorage::new()),
        Arc::new(smg_data_connector::MemoryConversationStorage::new()),
        Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
        None,
        None,
        tx,
    )
    .await
    .unwrap();
    rx
}

#[tokio::test]
async fn terminal_stream_preserves_cache_and_reasoning_usage() {
    for cached in [Some(0), Some(6), None] {
        let mut usage = Usage::from_counts(12, 7).with_reasoning_tokens(2);
        if let Some(cached) = cached {
            usage = usage.with_cached_tokens(cached);
        }
        let chunk = ChatCompletionStreamResponse::builder("chat_test", "test-model")
            .usage(usage)
            .build();
        let frames = vec![
            format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()),
            "data: [DONE]\n\n".into(),
        ];
        let mut rx = transformed_chat_stream(frames).await;
        let mut terminal = None;
        while let Some(Ok(bytes)) = rx.recv().await {
            let text = std::str::from_utf8(&bytes).unwrap();
            let data = text
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap();
            let event: Value = serde_json::from_str(data).unwrap();
            if event["type"] == "response.completed" {
                terminal = Some(event);
            }
        }
        let terminal = terminal.expect("completed event must be emitted");
        let usage = &terminal["response"]["usage"];
        assert_eq!(usage["input_tokens"], 12);
        assert_eq!(usage["output_tokens"], 7);
        assert_eq!(usage["total_tokens"], 19);
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 2);
        assert_eq!(
            usage["input_tokens_details"]["cached_tokens"],
            cached.unwrap_or(0)
        );
    }
}

#[tokio::test]
async fn upstream_error_closes_reasoning_and_emits_failed_response() {
    let reasoning = json!({"id":"chat_test","object":"chat.completion.chunk","created":0,"model":"test-model",
        "choices":[{"index":0,"delta":{"reasoning_content":"Partial reasoning"},"finish_reason":null}]});
    let frames = vec![
        format!("data: {reasoning}\n\n"),
        "data: {\"error\":{\"message\":\"engine unavailable\",\"type\":\"internal_error\"}}\n\n"
            .into(),
        "data: [DONE]\n\n".into(),
    ];
    let mut rx = transformed_chat_stream(frames).await;
    let mut events: Vec<Value> = Vec::new();
    while let Some(Ok(bytes)) = rx.recv().await {
        let text = std::str::from_utf8(&bytes).unwrap();
        let data = text
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap();
        events.push(serde_json::from_str(data).unwrap());
    }
    let added = events
        .iter()
        .find(|e| e["type"] == "response.content_part.added")
        .unwrap();
    let done = events
        .iter()
        .find(|e| e["type"] == "response.content_part.done")
        .expect("open reasoning must close even on error");
    assert_eq!(added["item_id"], done["item_id"]);
    assert_eq!(done["part"]["text"], "Partial reasoning");
    let terminal = events.last().unwrap();
    assert_eq!(terminal["type"], "response.failed");
    assert_eq!(terminal["response"]["status"], "failed");
    assert_eq!(
        terminal["response"]["error"]["message"],
        "engine unavailable"
    );
    assert!(!events.iter().any(|e| e["type"] == "response.completed"));
}

/// The terminal of a Responses stream reports the code of the chat
/// stream's error event: the timeout's code for a stream the gateway's
/// request timeout cut, and the event's type for an event without a code.
#[tokio::test]
async fn a_stream_cut_by_the_request_timeout_fails_the_response_with_the_timeouts_code() {
    for (error, code) in [
        (
            json!({"message": "the request timed out", "type": "stream_error", "code": REQUEST_TIMEOUT_CODE}),
            REQUEST_TIMEOUT_CODE,
        ),
        (
            json!({"message": "the worker went away", "type": "stream_error"}),
            "stream_error",
        ),
    ] {
        let cut = json!({"error": error});
        let frames = vec![format!("data: {cut}\n\n"), "data: [DONE]\n\n".into()];
        let mut rx = transformed_chat_stream(frames).await;
        let events = drained_events(&mut rx).await;
        let terminal = events.last().expect("a terminal event");
        assert_eq!(terminal["type"], "response.failed", "{cut}");
        assert_eq!(terminal["response"]["error"]["code"], code, "{cut}");
    }
}

/// The `data:` payloads the client read, in order, once the transform has
/// dropped its sender.
async fn drained_events(rx: &mut mpsc::Receiver<Result<Bytes, std::io::Error>>) -> Vec<Value> {
    let mut events = Vec::new();
    while let Some(Ok(bytes)) = rx.recv().await {
        let text = std::str::from_utf8(&bytes).unwrap();
        events.extend(
            text.lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .map(|data| serde_json::from_str::<Value>(data).unwrap()),
        );
    }
    events
}

/// Runs chat SSE frames through the Responses transform as a STORED
/// request over memory storages (no tenant, no request context) and hands
/// back the events the client read and the object the store holds under
/// the id `response.created` announced.
async fn transformed_and_stored(
    frames: Vec<String>,
    request: ResponsesRequest,
) -> (Vec<Value>, Value) {
    let body = Body::from_stream(futures_util::stream::iter(
        frames.into_iter().map(Ok::<_, std::convert::Infallible>),
    ));
    let storage = Arc::new(smg_data_connector::MemoryResponseStorage::new());
    let (tx, mut rx) = mpsc::channel(64);
    process_and_transform_sse_stream(
        body,
        Arc::new(ResponsesRequest {
            store: Some(true),
            ..request
        }),
        None,
        None,
        storage.clone(),
        Arc::new(smg_data_connector::MemoryConversationStorage::new()),
        Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
        None,
        None,
        tx,
    )
    .await
    .unwrap();
    let events = drained_events(&mut rx).await;
    let id = events[0]["response"]["id"]
        .as_str()
        .expect("response.created names the id");
    let stored = storage
        .get_response(&smg_data_connector::ResponseId::from(id))
        .await
        .unwrap()
        .expect("a stored request's record")
        .raw_response;
    (events, stored)
}

/// The item ids of a response object's output, in order.
fn output_ids(response: &Value) -> Vec<String> {
    response["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["id"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn the_stored_streamed_object_serializes_responses_api_usage() {
    let chunk = ChatCompletionStreamResponse::builder("chat_test", "test-model")
        .usage(responses_api_test_usage())
        .build();
    let frames = vec![
        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()),
        "data: [DONE]\n\n".into(),
    ];
    let (events, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
    assert_responses_api_usage(&stored);
    // A turn without items: the terminal event and the stored object both
    // read `output: []`; no message is invented for it.
    let terminal = events.last().expect("a terminal event");
    assert_eq!(terminal["response"]["output"], json!([]), "{terminal}");
    assert_eq!(terminal["response"]["output"], stored["output"]);
}

/// The stored object of a stream is the one its events announced: the
/// items in their streamed order, each under the id its
/// `response.output_item.added` carried (the reasoning item `rs_`, the
/// message `msg_`, the function call `fc_`), a function call's own id
/// beside the `call_id` the client echoes, and the terminal event names
/// the same items.
#[tokio::test]
async fn the_stored_streamed_object_carries_the_items_the_events_announced() {
    let frames = vec![
        chat_event(json!({"reasoning_content":"think"}), None),
        chat_event(json!({"content":"Hello"}), None),
        chat_event(
            json!({"tool_calls":[{"index":0,"id":"call_streamed","type":"function",
                "function":{"name":"lookup","arguments":"{}"}}]}),
            None,
        ),
        chat_event(json!({}), Some("tool_calls")),
        "data: [DONE]\n\n".into(),
    ];
    let (events, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
    let output = stored["output"].as_array().unwrap();
    let kinds: Vec<&str> = output
        .iter()
        .map(|item| item["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["reasoning", "message", "function_call"], "{stored}");
    for (item, prefix) in output.iter().zip(["rs_", "msg_", "fc_"]) {
        let id = item["id"].as_str().unwrap();
        assert!(id.starts_with(prefix), "{item}");
        for event_type in ["response.output_item.added", "response.output_item.done"] {
            assert_eq!(
                events
                    .iter()
                    .filter(|e| e["type"] == event_type && e["item"]["id"] == id)
                    .count(),
                1,
                "{event_type} of {id}"
            );
        }
    }
    let call = &output[2];
    assert_eq!(call["call_id"], "call_streamed");
    assert_ne!(call["id"], call["call_id"], "{call}");
    let terminal = events.last().unwrap();
    assert_eq!(terminal["type"], "response.completed");
    assert_eq!(output_ids(&terminal["response"]), output_ids(&stored));
}

#[tokio::test]
async fn namespace_stored_streamed_call_keeps_its_structured_identity() {
    let first = ChatCompletionStreamResponse::builder("chat_test", "test-model")
        .add_choice_tool_name(0, "call_test", "weather.lookup")
        .build();
    let frames = vec![
        format!("data: {}\n\n", serde_json::to_string(&first).unwrap()),
        chat_event(
            json!({"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}),
            Some("tool_calls"),
        ),
        "data: [DONE]\n\n".into(),
    ];
    let (_, stored) = transformed_and_stored(frames, namespace_test_request()).await;
    assert_eq!(stored["output"][0]["name"], "lookup");
    assert_eq!(stored["output"][0]["namespace"], "weather");
    assert_eq!(stored["output"][0]["arguments"], "{}");
    assert_eq!(stored["output"][0]["call_id"], "call_test");
}

#[tokio::test]
async fn stored_streamed_tool_calls_have_terminal_status() {
    for (finish_reason, expected_status) in [("tool_calls", "completed"), ("length", "incomplete")]
    {
        let frames = vec![
            chat_event(
                json!({"tool_calls":[{"index":0,"id":"call_weather","type":"function",
                    "function":{"name":"weather","arguments":"{}"}}]}),
                Some(finish_reason),
            ),
            "data: [DONE]\n\n".into(),
        ];
        let (_, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
        assert_eq!(stored["status"], expected_status);
        assert_eq!(stored["output"][0]["status"], "completed");
        assert_eq!(stored["output"][0]["call_id"], "call_weather");
        assert_eq!(stored["output"][0]["arguments"], "{}");
        if finish_reason == "length" {
            assert_eq!(stored["incomplete_details"]["reason"], "max_output_tokens");
        }
    }
}

#[tokio::test]
async fn failed_generation_does_not_complete_stored_partial_messages_or_tool_calls() {
    for finish_reason in ["failed", "error"] {
        let frames = vec![
            chat_event(
                json!({"content":"Partial reply","tool_calls":[{"index":0,"id":"call_weather",
                    "type":"function","function":{"name":"weather","arguments":"{\"city\":"}}]}),
                None,
            ),
            chat_event(json!({}), Some(finish_reason)),
            "data: [DONE]\n\n".into(),
        ];
        let (_, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
        assert_eq!(stored["status"], "failed", "{finish_reason}");
        assert_eq!(stored["output"].as_array().unwrap().len(), 2);
        assert_eq!(stored["output"][0]["type"], "message");
        assert_eq!(
            stored["output"][0]["status"], "in_progress",
            "{finish_reason}"
        );
        assert_eq!(stored["output"][0]["content"][0]["text"], "Partial reply");
        let item = &stored["output"][1];
        assert_eq!(item["status"], "in_progress", "{finish_reason}");
        assert_eq!(item["call_id"], "call_weather");
        assert_eq!(item["arguments"], "{\"city\":");
    }
}

#[tokio::test]
async fn length_truncated_stored_tool_call_preserves_earlier_completed_calls() {
    let frames = vec![
        chat_event(
            json!({"tool_calls":[
                {"index":0,"id":"call_weather","type":"function","function":{"name":"weather","arguments":"{}"}},
                {"index":1,"id":"call_time","type":"function","function":{"name":"time","arguments":"{\"tz\":"}}
            ]}),
            None,
        ),
        chat_event(json!({}), Some("length")),
        "data: [DONE]\n\n".into(),
    ];
    let (_, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
    assert_eq!(stored["status"], "incomplete");
    assert_eq!(stored["incomplete_details"]["reason"], "max_output_tokens");
    assert_eq!(stored["output"].as_array().unwrap().len(), 2);
    assert_eq!(stored["output"][0]["status"], "completed");
    assert_eq!(stored["output"][0]["arguments"], "{}");
    assert_eq!(stored["output"][1]["status"], "incomplete");
    assert_eq!(stored["output"][1]["call_id"], "call_time");
    assert_eq!(stored["output"][1]["arguments"], "{\"tz\":");
}

/// A call cut before its arguments were whole reads `incomplete` in the
/// stored object whatever ended the stream: an abort, a stop, or a body
/// that ended with no finish at all (the item is closed when the stream
/// ends, so the record keeps it).
#[tokio::test]
async fn a_stored_streamed_turn_whose_call_was_cut_reads_incomplete() {
    for finish_reason in [Some("abort"), Some("stop"), None] {
        let frames = vec![
            chat_event(
                json!({"tool_calls":[{"index":0,"id":"call_time","type":"function",
                    "function":{"name":"time","arguments":"{\"tz\":"}}]}),
                None,
            ),
            chat_event(json!({}), finish_reason),
            "data: [DONE]\n\n".into(),
        ];
        let (_, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
        assert_eq!(stored["status"], "incomplete", "{finish_reason:?}");
        assert!(stored["incomplete_details"].is_null(), "{finish_reason:?}");
        assert_eq!(
            stored["output"][0]["status"], "incomplete",
            "{finish_reason:?}"
        );
        assert_eq!(
            stored["output"][0]["arguments"], "{\"tz\":",
            "{finish_reason:?}"
        );
    }
}

#[tokio::test]
async fn a_stored_streamed_turn_whose_custom_call_was_cut_reads_incomplete_and_keeps_its_item() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "test-model", "input": "hi",
        "tools": [{"type": "custom", "name": "emit_command"}]
    }))
    .unwrap();
    for finish_reason in [Some("abort"), Some("stop"), None] {
        let frames = vec![
            chat_event(
                json!({"tool_calls":[{"index":0,"id":"call_c","type":"function",
                    "function":{"name":"emit_command","arguments":"{\"input\": \"pw"}}]}),
                None,
            ),
            chat_event(json!({}), finish_reason),
            "data: [DONE]\n\n".into(),
        ];
        let (_, stored) = transformed_and_stored(frames, request.clone()).await;
        assert_eq!(stored["status"], "incomplete", "{finish_reason:?}");
        assert!(stored["incomplete_details"].is_null(), "{finish_reason:?}");
        assert_eq!(
            stored["output"][0]["type"], "custom_tool_call",
            "{finish_reason:?}"
        );
        assert_eq!(
            stored["output"][0]["call_id"], "call_c",
            "{finish_reason:?}"
        );
    }
}

/// A stream that fails after a message and a call were partly streamed
/// closes both (in progress) before `response.failed`, and the stored
/// object carries the same items under the same ids with the same error,
/// whether the failure is the generation's finish or an upstream error
/// event before any finish.
#[tokio::test]
async fn failed_terminal_retains_and_closes_partial_message_and_function() {
    for finish in [None, Some("failed"), Some("error")] {
        let mut frames = vec![chat_event(
            json!({"content":"Partial answer","tool_calls":[{"index":0,"id":"call_test",
                "type":"function","function":{"name":"lookup","arguments":"{\"city\":"}}]}),
            None,
        )];
        match finish {
            Some(finish) => frames.push(chat_event(json!({}), Some(finish))),
            None => frames.push(
                "data: {\"error\":{\"code\":\"internal_error\",\"message\":\"engine stopped\"}}\n\n"
                    .into(),
            ),
        }
        frames.push("data: [DONE]\n\n".into());
        let (events, stored) = transformed_and_stored(frames, ResponsesRequest::default()).await;
        let terminal = events.last().unwrap();
        assert_eq!(terminal["type"], "response.failed", "{finish:?}");
        assert_eq!(terminal["response"]["output"].as_array().unwrap().len(), 2);
        assert_eq!(stored["status"], "failed", "{finish:?}");
        assert_eq!(stored["error"], terminal["response"]["error"], "{finish:?}");
        if finish.is_some() {
            assert_eq!(stored["error"]["code"], "server_error");
            assert!(stored["error"]["message"]
                .as_str()
                .is_some_and(|s| !s.is_empty()));
        } else {
            assert_eq!(stored["error"]["code"], "internal_error");
            assert_eq!(stored["error"]["message"], "engine stopped");
        }
        for kind in ["message", "function_call"] {
            let output = terminal["response"]["output"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["type"] == kind)
                .unwrap();
            let saved = stored["output"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["type"] == kind)
                .unwrap();
            assert_eq!(output["status"], "in_progress", "{output}");
            assert_eq!(saved["id"], output["id"]);
            assert_eq!(saved["status"], output["status"]);
            assert_eq!(saved["arguments"], output["arguments"]);
            assert_eq!(saved["content"][0]["text"], output["content"][0]["text"]);
            for event_type in ["response.output_item.added", "response.output_item.done"] {
                assert_eq!(
                    events
                        .iter()
                        .filter(|e| e["type"] == event_type && e["item"]["id"] == output["id"])
                        .count(),
                    1
                );
            }
        }
        for kind in [
            "response.output_text.done",
            "response.content_part.done",
            "response.function_call_arguments.done",
        ] {
            assert_eq!(
                events.iter().filter(|e| e["type"] == kind).count(),
                1,
                "{events:?}"
            );
        }
    }
}

#[tokio::test]
async fn mcp_stream_preserves_structured_upstream_errors() {
    for (error, expected_code) in [
        (
            json!({"code":"overloaded","type":"server_error","message":"busy"}),
            "overloaded",
        ),
        (
            json!({"type":"internal_error","message":"busy"}),
            "internal_error",
        ),
        (
            json!({"code":null,"type":"internal_error","message":"busy"}),
            "internal_error",
        ),
        (json!({"message":"busy"}), "server_error"),
    ] {
        let frame = format!("data: {}\n\n", json!({"error":error}));
        let body = Body::from(frame);
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
        // The client stays (its receiver lives), so the body is read.
        let (tx, _rx) = mpsc::channel(64);
        let error = convert_and_accumulate_stream(body, &mut emitter, &tx, |_| false)
            .await
            .unwrap_err();
        let wire = serde_json::to_value(error).unwrap();
        assert_eq!(wire["code"], expected_code, "{wire}");
        assert_eq!(wire["message"], "busy");
    }
}

#[tokio::test]
async fn mcp_converter_hides_server_calls_but_streams_interleaved_client_calls() {
    let chunks: Vec<ChatCompletionStreamResponse> = [
        (json!({"tool_calls":[{"index":0,"id":"call_server","type":"function"},
            {"index":1,"id":"call_client","type":"function"}]}), None),
        (json!({"tool_calls":[{"index":0,"function":{"name":"server_tool","arguments":"{\"q\":"}},
            {"index":1,"function":{"name":"client_tool","arguments":"{\"q\":"}}]}), None),
        (json!({"tool_calls":[{"index":1,"function":{"arguments":"1}"}},
            {"index":0,"function":{"arguments":"2}"}}]}), None),
        (json!({}), Some("tool_calls")),
    ].into_iter().map(|(delta, finish)| serde_json::from_value(json!({
        "id":"chat_test","object":"chat.completion.chunk","created":0,"model":"test-model",
        "choices":[{"index":0,"delta":delta,"finish_reason":finish}],
    })).unwrap()).collect();
    let frames = chunks.into_iter().map(|chunk| {
        Ok::<_, std::convert::Infallible>(format!(
            "data: {}\n\n",
            serde_json::to_string(&chunk).unwrap()
        ))
    });
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    let (tx, mut rx) = mpsc::channel(64);
    let response = convert_and_accumulate_stream(
        Body::from_stream(futures_util::stream::iter(frames)),
        &mut emitter,
        &tx,
        |name| name == "server_tool",
    )
    .await
    .unwrap();
    assert_eq!(
        response.choices[0]
            .message
            .tool_calls
            .as_ref()
            .unwrap()
            .len(),
        2
    );
    emitter.emit_terminal(None, None, &tx).await.unwrap();
    drop(tx);
    let mut events: Vec<Value> = Vec::new();
    while let Some(Ok(bytes)) = rx.recv().await {
        let text = std::str::from_utf8(&bytes).unwrap();
        events.push(
            serde_json::from_str(
                text.lines()
                    .find_map(|line| line.strip_prefix("data: "))
                    .unwrap(),
            )
            .unwrap(),
        );
    }
    let output = events.last().unwrap()["response"]["output"]
        .as_array()
        .unwrap();
    assert_eq!(output.len(), 1, "{events:?}");
    assert_eq!(output[0]["call_id"], "call_client");
    assert_eq!(output[0]["name"], "client_tool");
    assert_eq!(output[0]["arguments"], "{\"q\":1}");
    for kind in [
        "response.output_item.added",
        "response.output_item.done",
        "response.function_call_arguments.done",
    ] {
        assert_eq!(events.iter().filter(|e| e["type"] == kind).count(), 1);
    }
    assert_eq!(
        events
            .iter()
            .filter(|e| e["type"] == "response.function_call_arguments.delta")
            .map(|e| e["delta"].as_str().unwrap())
            .collect::<String>(),
        "{\"q\":1}"
    );
    assert!(!events.iter().any(|e| e["item"]["name"] == "server_tool"));
}

#[tokio::test]
async fn mcp_transport_error_keeps_stream_error_code() {
    let body = Body::from_stream(futures_util::stream::iter([Err::<Bytes, _>(
        std::io::Error::other("connection reset"),
    )]));
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    // The client stays (its receiver lives), so the body is read.
    let (tx, _rx) = mpsc::channel(64);
    let error = convert_and_accumulate_stream(body, &mut emitter, &tx, |_| false)
        .await
        .unwrap_err();
    assert_eq!(error["code"], "stream_error");
    assert!(error["message"]
        .as_str()
        .unwrap()
        .contains("connection reset"));
}

/// A stored foreground stream whose client leaves mid-stream keeps the
/// promise `response.created` made: its record is written with the items
/// streamed so far (closed in progress) and status `cancelled`, while the
/// chat body is dropped with the read, as before; an unstored one records
/// nothing. The transform answers the departure either way.
#[tokio::test]
async fn a_stored_stream_whose_client_leaves_is_recorded_as_cancelled() {
    for store in [true, false] {
        let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        let body = Body::from_stream(ReceiverStream::new(body_rx));
        let storage = Arc::new(smg_data_connector::MemoryResponseStorage::new());
        let (tx, mut rx) = mpsc::channel(64);
        let transform = process_and_transform_sse_stream(
            body,
            Arc::new(ResponsesRequest {
                store: Some(store),
                ..Default::default()
            }),
            None,
            None,
            storage.clone(),
            Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
            None,
            None,
            tx,
        );
        let client = async move {
            body_tx
                .send(Ok(Bytes::from(chat_event(
                    json!({"content":"Hello"}),
                    None,
                ))))
                .await
                .unwrap();
            // The client reads up to the text it was streamed, then leaves.
            let mut id = None;
            'read: while let Some(Ok(bytes)) = rx.recv().await {
                let text = std::str::from_utf8(&bytes).unwrap();
                for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
                    let event: Value = serde_json::from_str(data).unwrap();
                    if event["type"] == "response.created" {
                        id = event["response"]["id"].as_str().map(str::to_string);
                    }
                    if event["type"] == "response.output_text.delta" {
                        break 'read;
                    }
                }
            }
            drop(rx);
            (id.expect("response.created names the id"), body_tx)
        };
        // The joined futures are large; boxed, they stay off the test's stack.
        let (outcome, (id, body_tx)) = Box::pin(tokio::time::timeout(
            std::time::Duration::from_secs(5),
            async { tokio::join!(transform, client) },
        ))
        .await
        .expect("the body never ends; the client's leaving ends the read");
        assert_eq!(outcome.unwrap_err(), "Client disconnected");
        assert!(
            body_tx.is_closed(),
            "the chat body is dropped with the read"
        );
        let record = storage
            .get_response(&smg_data_connector::ResponseId::from(id.as_str()))
            .await
            .unwrap();
        if !store {
            assert!(record.is_none(), "an unstored response records nothing");
            continue;
        }
        let stored = record.expect("the stored stream's record").raw_response;
        assert_eq!(stored["status"], "cancelled", "{stored}");
        assert!(stored["error"].is_null(), "{stored}");
        assert!(stored["incomplete_details"].is_null(), "{stored}");
        assert_eq!(stored["output"][0]["type"], "message", "{stored}");
        assert_eq!(stored["output"][0]["status"], "in_progress", "{stored}");
        assert_eq!(stored["output"][0]["content"][0]["text"], "Hello");
        assert!(
            stored["output"][0]["id"]
                .as_str()
                .is_some_and(|id| id.starts_with("msg_")),
            "{stored}"
        );
    }
}

/// Reads the client's SSE frames up to the first event of `event_type`
/// and returns the id `response.created` announced.
async fn read_until<E>(rx: &mut mpsc::Receiver<Result<Bytes, E>>, event_type: &str) -> String {
    let mut id = None;
    'read: while let Some(Ok(bytes)) = rx.recv().await {
        let text = std::str::from_utf8(&bytes).unwrap();
        for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
            let event: Value = serde_json::from_str(data).unwrap();
            if event["type"] == "response.created" {
                id = event["response"]["id"].as_str().map(str::to_string);
            }
            if event["type"] == event_type {
                break 'read;
            }
        }
    }
    id.expect("response.created names the id")
}

/// The stored object of the response `id`.
async fn stored_record(storage: &smg_data_connector::MemoryResponseStorage, id: &str) -> Value {
    storage
        .get_response(&smg_data_connector::ResponseId::from(id))
        .await
        .unwrap()
        .expect("the stored stream's record")
        .raw_response
}

/// The client leaves while the message's closing events are being sent
/// (an SSE channel of one slot: the client reads up to the text delta and
/// leaves, so the first closing event's send meets the dead client). The
/// item was stored before that send, and the finish had been read: the
/// record keeps the message, completed, under the stream's own finish.
#[tokio::test]
async fn a_client_that_leaves_during_the_closing_events_keeps_the_streamed_message() {
    let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
    let body = Body::from_stream(ReceiverStream::new(body_rx));
    let storage = Arc::new(smg_data_connector::MemoryResponseStorage::new());
    let (tx, mut rx) = mpsc::channel(1);
    let transform = process_and_transform_sse_stream(
        body,
        Arc::new(ResponsesRequest {
            store: Some(true),
            ..Default::default()
        }),
        None,
        None,
        storage.clone(),
        Arc::new(smg_data_connector::MemoryConversationStorage::new()),
        Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
        None,
        None,
        tx,
    );
    let client = async move {
        for frame in [
            chat_event(json!({"content":"Hello"}), None),
            chat_event(json!({}), Some("stop")),
        ] {
            body_tx.send(Ok(Bytes::from(frame))).await.unwrap();
        }
        let id = read_until(&mut rx, "response.output_text.delta").await;
        drop(rx);
        (id, body_tx)
    };
    let (outcome, (id, _body_tx)) = Box::pin(tokio::time::timeout(
        std::time::Duration::from_secs(5),
        async { tokio::join!(transform, client) },
    ))
    .await
    .expect("the client's leaving ends the read");
    assert_eq!(outcome.unwrap_err(), "Client disconnected");
    let stored = stored_record(&storage, &id).await;
    assert_eq!(stored["status"], "completed", "{stored}");
    assert_eq!(stored["output"][0]["type"], "message", "{stored}");
    assert_eq!(stored["output"][0]["status"], "completed", "{stored}");
    assert_eq!(
        stored["output"][0]["content"][0]["text"], "Hello",
        "{stored}"
    );
}

/// A finish the stream had already read stands when the client leaves
/// before the body ends: a `length` finish is recorded `incomplete` with
/// its `incomplete_details`, not `cancelled`, and the message it closed
/// keeps its text.
#[tokio::test]
async fn a_client_that_leaves_after_the_finish_keeps_the_finish() {
    let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
    let body = Body::from_stream(ReceiverStream::new(body_rx));
    let storage = Arc::new(smg_data_connector::MemoryResponseStorage::new());
    let (tx, mut rx) = mpsc::channel(64);
    let transform = process_and_transform_sse_stream(
        body,
        Arc::new(ResponsesRequest {
            store: Some(true),
            ..Default::default()
        }),
        None,
        None,
        storage.clone(),
        Arc::new(smg_data_connector::MemoryConversationStorage::new()),
        Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
        None,
        None,
        tx,
    );
    let client = async move {
        for frame in [
            chat_event(json!({"content":"Hello"}), None),
            chat_event(json!({}), Some("length")),
        ] {
            body_tx.send(Ok(Bytes::from(frame))).await.unwrap();
        }
        // The message closed under the finish; the client leaves before
        // the body ends (no usage chunk, no `[DONE]` yet).
        let id = read_until(&mut rx, "response.output_item.done").await;
        drop(rx);
        (id, body_tx)
    };
    let (outcome, (id, _body_tx)) = Box::pin(tokio::time::timeout(
        std::time::Duration::from_secs(5),
        async { tokio::join!(transform, client) },
    ))
    .await
    .expect("the client's leaving ends the read");
    assert_eq!(outcome.unwrap_err(), "Client disconnected");
    let stored = stored_record(&storage, &id).await;
    assert_eq!(stored["status"], "incomplete", "{stored}");
    assert_eq!(
        stored["incomplete_details"]["reason"], "max_output_tokens",
        "{stored}"
    );
    assert_eq!(
        stored["output"][0]["content"][0]["text"], "Hello",
        "{stored}"
    );
    assert_ne!(stored["output"][0]["status"], "in_progress", "{stored}");
}

/// Under `cancelled` every item still open when the client left reads
/// `in_progress`: a reasoning item, a function call with partial
/// arguments, as the message does.
#[tokio::test]
async fn a_cancelled_record_closes_the_open_reasoning_and_call_items_in_progress() {
    for (delta, leave_at, item_type) in [
        (
            json!({"reasoning_content":"think"}),
            "response.reasoning_text.delta",
            "reasoning",
        ),
        (
            json!({"tool_calls":[{"index":0,"id":"call_open","type":"function",
                "function":{"name":"lookup","arguments":"{\"q\":"}}]}),
            "response.function_call_arguments.delta",
            "function_call",
        ),
    ] {
        let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        let body = Body::from_stream(ReceiverStream::new(body_rx));
        let storage = Arc::new(smg_data_connector::MemoryResponseStorage::new());
        let (tx, mut rx) = mpsc::channel(64);
        let transform = process_and_transform_sse_stream(
            body,
            Arc::new(ResponsesRequest {
                store: Some(true),
                ..Default::default()
            }),
            None,
            None,
            storage.clone(),
            Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
            None,
            None,
            tx,
        );
        let client = async move {
            body_tx
                .send(Ok(Bytes::from(chat_event(delta, None))))
                .await
                .unwrap();
            let id = read_until(&mut rx, leave_at).await;
            drop(rx);
            (id, body_tx)
        };
        let (outcome, (id, _body_tx)) = Box::pin(tokio::time::timeout(
            std::time::Duration::from_secs(5),
            async { tokio::join!(transform, client) },
        ))
        .await
        .expect("the client's leaving ends the read");
        assert_eq!(outcome.unwrap_err(), "Client disconnected");
        let stored = stored_record(&storage, &id).await;
        assert_eq!(stored["status"], "cancelled", "{item_type}: {stored}");
        assert_eq!(stored["output"][0]["type"], item_type, "{stored}");
        assert_eq!(stored["output"][0]["status"], "in_progress", "{stored}");
    }
}

/// The reader parked on a chat body that stalled (nothing more on it)
/// ends the moment its client leaves, with the error it ends with when
/// it meets the dead client at the next frame, and the chat body goes
/// with it: its sender sees the drop, as the chat layer does.
#[tokio::test]
async fn a_client_that_leaves_while_the_chat_body_is_stalled_ends_the_read_at_once() {
    let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
    let body = Body::from_stream(ReceiverStream::new(body_rx));
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    let (tx, rx) = mpsc::channel(64);
    let read = convert_and_accumulate_stream(body, &mut emitter, &tx, |_| false);
    let leave = async move {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        drop(rx);
    };
    let (outcome, ()) = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        tokio::join!(read, leave)
    })
    .await
    .expect("the body never yields; the client's leaving ends the read");
    let error = outcome.unwrap_err();
    assert_eq!(error["code"], "stream_error");
    assert_eq!(error["message"], "Client disconnected");
    assert!(
        body_tx.is_closed(),
        "the chat body is dropped with the read"
    );
}

/// The client leaves between two reads of the chat body (while a call is
/// named): only a wait the relay kept from its parked read wakes it then.
#[test]
fn the_wait_a_parked_read_registered_wakes_the_reader_after_that_read() {
    use std::{
        cell::Cell,
        future::Future,
        sync::atomic::{AtomicUsize, Ordering},
        task::{Context, Poll, Wake, Waker},
    };

    struct Wakes(AtomicUsize);
    impl Wake for Wakes {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let wakes = Arc::new(Wakes(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&wakes));
    let mut cx = Context::from_waker(&waker);

    let (body_tx, body_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
    let body = Body::from_stream(ReceiverStream::new(body_rx));
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    let (tx, rx) = mpsc::channel(64);
    let rx = Cell::new(Some(rx));
    let mut read = std::pin::pin!(convert_and_accumulate_stream(
        body,
        &mut emitter,
        &tx,
        |_| {
            drop(rx.take());
            true
        }
    ));
    assert!(read.as_mut().poll(&mut cx).is_pending());
    let call = json!({"tool_calls":[{"index":0,"id":"call_server","type":"function",
        "function":{"name":"server_tool","arguments":""}}]});
    // A keep-alive ends the parked read. The next read finds the call's
    // frame ready and registers no wait; the call is named between reads.
    body_tx
        .try_send(Ok(Bytes::from_static(b": keep-alive\n\n")))
        .unwrap();
    body_tx
        .try_send(Ok(Bytes::from(chat_event(call, None))))
        .unwrap();
    assert_eq!(wakes.0.load(Ordering::SeqCst), 1, "the frames' one wake");
    let Poll::Ready(Err(error)) = read.as_mut().poll(&mut cx) else {
        panic!("the read ends at the first read after the client left");
    };
    assert_eq!(error["message"], "Client disconnected");
    assert_eq!(
        wakes.0.load(Ordering::SeqCst),
        2,
        "the client's leaving wakes the reader through the kept wait"
    );
}

/// One reader for the chat stream's events: chunks, the error event as
/// the terminal's error object (`code`, else `type`, else `server_error`;
/// `message`, else the generic wording), the terminator, and every other
/// data event left to the caller; lines that are not data events are
/// nobody's.
#[test]
fn the_chat_sse_reader_sorts_a_frames_events() {
    let frame = concat!(
        ": keep-alive\n\n",
        "data: {\"id\":\"chat_test\",\"object\":\"chat.completion.chunk\",\"created\":0,\
             \"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\
             \"finish_reason\":null}]}\n\n",
        "data: {\"unexpected\":true}\n\n",
        "data: {\"error\":{\"type\":\"internal_error\",\"message\":\"engine unavailable\"}}\n\n",
        "data: {\"error\":{\"code\":\"rate_limited\"}}\n\n",
        "data: [DONE]\n\n",
        "data: {\"after\":\"done\"}\n\n",
    );
    let read: Vec<String> = chat_sse_events::<ChatCompletionStreamResponse>(frame)
        .map(|event| match event {
            ChatSseEvent::Chunk(chunk) => format!(
                "chunk:{}",
                chunk.choices[0].delta.content.as_deref().unwrap_or("")
            ),
            ChatSseEvent::UpstreamError(error) => format!("error:{error}"),
            ChatSseEvent::Done => "done".to_string(),
            ChatSseEvent::Other(event) => format!("other:{event}"),
        })
        .collect();
    assert_eq!(
        read,
        [
            "chunk:hi",
            "other:data: {\"unexpected\":true}",
            "error:{\"code\":\"internal_error\",\"message\":\"engine unavailable\"}",
            "error:{\"code\":\"rate_limited\",\"message\":\"Upstream generation failed\"}",
            "done",
            "other:data: {\"after\":\"done\"}",
        ]
    );
}

/// What the stream transform reads of each data event of `body` with the
/// chunks read as `C`: the parts `process_chunk` asks a chunk for and the
/// chunk's usage, or the event's kind.
fn chunk_reads<'a, C: Deserialize<'a> + ChatChunkParts>(
    body: &'a str,
    usage: impl Fn(&C) -> Option<&Usage>,
) -> Vec<String> {
    chat_sse_events::<C>(body)
        .map(|event| match event {
            ChatSseEvent::Chunk(chunk) => {
                let calls: Vec<_> = (0..)
                    .map_while(|at| chunk.tool_call(at))
                    .map(|call| (call.index, call.id, call.name, call.arguments))
                    .collect();
                format!(
                    "chunk: {:?} {:?} {calls:?} {:?} {:?}",
                    chunk.reasoning(),
                    chunk.content(),
                    chunk.finish_reason(),
                    usage(&chunk).map(|usage| json!(usage))
                )
            }
            ChatSseEvent::UpstreamError(error) => format!("error: {error}"),
            ChatSseEvent::Done => "done".to_string(),
            ChatSseEvent::Other(event) => format!("other: {event}"),
        })
        .collect()
}

/// The borrowed view reads every data event of a chat body as the owned
/// chunk reads it: the same events are chunks, each chunk hands
/// `process_chunk` the same parts and the loop the same usage, a chunk
/// with two choices the parts of its first; a text with no escape is
/// borrowed from the event, one with an escape copied.
#[test]
fn the_borrowed_chunk_view_reads_what_the_owned_chunk_reads() {
    let body = [
        ": keep-alive\n\n".to_string(),
        chat_event(json!({"role": "assistant", "content": ""}), None),
        chat_event(json!({"reasoning_content": "th\u{e9} \"why\"\n"}), None),
        chat_event(json!({"content": "Hello"}), None),
        chat_event(json!({"content": " \"quoted\"\n\ttab \u{1F600}"}), None),
        "data: {\"id\":\"chat_test\",\"object\":\"chat.completion.chunk\",\"created\":0,\
             \"model\":\"test-model\",\"choices\":[\
             {\"index\":0,\"delta\":{\"content\":\"first\"},\"finish_reason\":null},\
             {\"index\":1,\"delta\":{\"content\":\"second\"},\"logprobs\":null,\
             \"finish_reason\":null}]}\n\n"
            .to_string(),
        chat_event(
            json!({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":"}}]}),
            None,
        ),
        chat_event(
            json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"Paris\"}"}},
                {"index": 1, "id": "call_2", "function": {"name": "get_time"}}]}),
            None,
        ),
        "data: {\"unexpected\":true}\n\n".to_string(),
        chat_event(json!({}), Some("tool_calls")),
        "data: {\"id\":\"chat_test\",\"object\":\"chat.completion.chunk\",\"created\":0,\
             \"model\":\"test-model\",\"choices\":[],\
             \"usage\":{\"prompt_tokens\":3,\"completion_tokens\":5,\"total_tokens\":8}}\n\n"
            .to_string(),
        "data: {\"error\":{\"type\":\"internal_error\",\"message\":\"engine unavailable\"}}\n\n"
            .to_string(),
        "data: [DONE]\n\n".to_string(),
        "data: {\"after\":\"done\"}\n\n".to_string(),
    ]
    .concat();
    let owned = chunk_reads::<ChatCompletionStreamResponse>(&body, |chunk| chunk.usage.as_ref());
    let view = chunk_reads::<ChatChunkView<'_>>(&body, |chunk| chunk.usage.as_ref());
    assert_eq!(view, owned);
    assert_eq!(owned.len(), 13);
    let chunks = owned.iter().filter(|read| read.starts_with("chunk: "));
    assert_eq!(chunks.count(), 9);
    assert!(owned[4].contains("first") && !owned[4].contains("second"));
    assert_eq!(owned[7], "other: data: {\"unexpected\":true}");
    assert_eq!(
        owned[10],
        "error: {\"code\":\"internal_error\",\"message\":\"engine unavailable\"}"
    );
    assert_eq!(owned[11], "done");
    assert_eq!(owned[12], "other: data: {\"after\":\"done\"}");

    let read = |event: &str| -> bool {
        let data = event.trim().strip_prefix("data: ").unwrap();
        let view: ChatChunkView<'_> = serde_json::from_str(data).unwrap();
        let content = view
            .choices
            .0
            .as_ref()
            .and_then(|choice| choice.delta.content.as_ref());
        matches!(content, Some(JsonText(Cow::Borrowed(_))))
    };
    assert!(read(&chat_event(json!({"content": "Hello"}), None)));
    assert!(!read(&chat_event(json!({"content": "a\"b"}), None)));
}

/// One chat chunk as the SSE event the chat stream writes for it.
fn chat_event(delta: Value, finish_reason: Option<&str>) -> String {
    format!(
        "data: {}\n\n",
        json!({
            "id":"chat_test","object":"chat.completion.chunk","created":0,"model":"test-model",
            "choices":[{"index":0,"delta":delta,"finish_reason":finish_reason}]
        })
    )
}

/// The chat stream hands the body every chunk the engine had ready as one
/// frame. Each event of such a frame is converted, and a `[DONE]` that
/// shares its frame with the finish chunk still ends the stream.
#[tokio::test]
async fn drained_frame_feeds_every_chat_chunk_to_the_responses_stream() {
    let frames = vec![
        [
            chat_event(json!({"role":"assistant","content":""}), None),
            chat_event(json!({"content":"Hel"}), None),
            chat_event(json!({"content":"lo"}), None),
        ]
        .concat(),
        [
            chat_event(json!({}), Some("stop")),
            "data: [DONE]\n\n".to_string(),
        ]
        .concat(),
    ];
    let mut rx = transformed_chat_stream(frames).await;
    let mut events: Vec<Value> = Vec::new();
    while let Some(Ok(bytes)) = rx.recv().await {
        let text = std::str::from_utf8(&bytes).unwrap();
        for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
            // Only the chat stream's `[DONE]` is not JSON; keep it visible.
            events.push(serde_json::from_str(data).unwrap_or(Value::String(data.to_string())));
        }
    }
    assert!(
        events.iter().all(|e| e["sequence_number"].is_u64()),
        "a chat frame reached the client unconverted: {events:?}"
    );
    let text: String = events
        .iter()
        .filter(|e| e["type"] == "response.output_text.delta")
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert_eq!(text, "Hello");
    let terminal = events.last().unwrap();
    assert_eq!(terminal["type"], "response.completed");
    assert_eq!(
        terminal["response"]["output"][0]["content"][0]["text"],
        "Hello"
    );
}

/// The MCP converter reads every event of a frame as well: a server call
/// whose name, arguments and finish arrive in one frame is accumulated
/// for execution.
#[tokio::test]
async fn drained_frame_feeds_every_chat_chunk_to_the_mcp_converter() {
    let frame = [
        chat_event(
            json!({"tool_calls":[{"index":0,"id":"call_server","type":"function",
                "function":{"name":"server_tool","arguments":""}}]}),
            None,
        ),
        chat_event(
            json!({"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":1}"}}]}),
            None,
        ),
        chat_event(json!({}), Some("tool_calls")),
        "data: [DONE]\n\n".to_string(),
    ]
    .concat();
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    let (tx, _rx) = mpsc::channel(64);
    let response = convert_and_accumulate_stream(Body::from(frame), &mut emitter, &tx, |name| {
        name == "server_tool"
    })
    .await
    .unwrap();
    let choice = &response.choices[0];
    assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    let calls = choice
        .message
        .tool_calls
        .as_ref()
        .expect("the server call of the frame is accumulated");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_server");
    assert_eq!(calls[0].function.name, "server_tool");
    assert_eq!(calls[0].function.arguments.as_deref(), Some("{\"q\":1}"));
}

/// An upstream error that shares a frame with the chunks before it fails
/// the response instead of completing it.
#[tokio::test]
async fn drained_frame_error_fails_the_mcp_stream() {
    let frame = [
        chat_event(json!({"role":"assistant","content":"Part"}), None),
        "data: {\"error\":{\"message\":\"engine unavailable\",\"type\":\"internal_error\"}}\n\n"
            .to_string(),
    ]
    .concat();
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    let (tx, _rx) = mpsc::channel(64);
    let error = convert_and_accumulate_stream(Body::from(frame), &mut emitter, &tx, |_| false)
        .await
        .unwrap_err();
    assert_eq!(error["code"], "internal_error");
    assert_eq!(error["message"], "engine unavailable");
}

/// A chat body whose read fails mid-stream ends the response as failed,
/// not completed: the text before the cut reaches the client, the
/// terminal event is `response.failed` with the read error, and the
/// stored record reads failed and carries it.
#[tokio::test]
async fn a_body_read_error_fails_the_responses_stream_and_its_record() {
    let frames: Vec<Result<String, std::io::Error>> = vec![
        Ok(chat_event(
            json!({"role":"assistant","content":"Part"}),
            None,
        )),
        Err(std::io::Error::other("the engine reset the connection")),
    ];
    let body = Body::from_stream(futures_util::stream::iter(frames));
    let response_storage = Arc::new(smg_data_connector::MemoryResponseStorage::new());
    let (tx, mut rx) = mpsc::channel(64);
    process_and_transform_sse_stream(
        body,
        Arc::new(ResponsesRequest {
            store: Some(true),
            ..Default::default()
        }),
        None,
        None,
        response_storage.clone(),
        Arc::new(smg_data_connector::MemoryConversationStorage::new()),
        Arc::new(smg_data_connector::MemoryConversationItemStorage::new()),
        None,
        None,
        tx,
    )
    .await
    .unwrap();
    let mut events: Vec<Value> = Vec::new();
    while let Some(Ok(bytes)) = rx.recv().await {
        let text = std::str::from_utf8(&bytes).unwrap();
        for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
            events.push(serde_json::from_str(data).unwrap());
        }
    }
    let text: String = events
        .iter()
        .filter(|e| e["type"] == "response.output_text.delta")
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert_eq!(text, "Part", "the text before the cut reaches the client");
    let terminal = events.last().expect("a terminal event");
    assert_eq!(terminal["type"], "response.failed", "{terminal}");
    assert_eq!(terminal["response"]["status"], "failed");
    assert_eq!(terminal["response"]["error"]["code"], "stream_error");
    let message = terminal["response"]["error"]["message"]
        .as_str()
        .expect("the error's message");
    assert!(
        message.starts_with("Stream read error: ")
            && message.contains("the engine reset the connection"),
        "{message}"
    );
    assert!(!events.iter().any(|e| e["type"] == "response.completed"));
    let id = terminal["response"]["id"]
        .as_str()
        .expect("the response id");
    let stored = response_storage
        .get_response(&smg_data_connector::ResponseId::from(id))
        .await
        .unwrap()
        .expect("the failed response is stored");
    assert_eq!(stored.raw_response["status"], "failed");
    assert_eq!(
        stored.raw_response["error"], terminal["response"]["error"],
        "the record carries the error the client saw"
    );
}

#[test]
fn the_events_of_a_frame_are_split_at_the_blank_line_as_before() {
    let frame = "data: {\"a\":1}\n\ndata: {\"b\":2}\n\n\n\n: keep-alive\n\ndata: [DONE]\n\n";
    let events: Vec<&str> = sse_events(frame).collect();
    let split: Vec<&str> = frame
        .split("\n\n")
        .map(str::trim)
        .filter(|event| !event.is_empty())
        .collect();
    assert_eq!(events, split);
    assert_eq!(
        events,
        [
            "data: {\"a\":1}",
            "data: {\"b\":2}",
            ": keep-alive",
            "data: [DONE]"
        ]
    );
    assert_eq!(sse_events("").count(), 0);
    assert_eq!(sse_events("\n\n").count(), 0);
    assert_eq!(sse_events("data: x").collect::<Vec<_>>(), ["data: x"]);
}

#[test]
fn a_frame_of_valid_bytes_is_borrowed_and_an_invalid_one_is_replaced() {
    let valid = b"data: {\"id\":\"chatcmpl-1\"}\n\n";
    let frame = frame_text(valid);
    assert!(matches!(frame, Cow::Borrowed(_)));
    assert_eq!(frame.as_ref(), std::str::from_utf8(valid).unwrap());

    let invalid = b"data: \xff\n\n";
    let frame = frame_text(invalid);
    assert!(matches!(frame, Cow::Owned(_)));
    assert_eq!(frame.as_ref(), "data: \u{FFFD}\n\n");
}
