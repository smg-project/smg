#[test]
fn invalid_tool_call_warning_is_correlated_and_content_free() {
    use std::{
        io::{self, Write},
        sync::{Arc, Mutex},
    };

    #[derive(Clone)]
    struct Capture(Arc<Mutex<Vec<u8>>>);
    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let bytes = Arc::new(Mutex::new(Vec::new()));
    let capture = Capture(bytes.clone());
    let subscriber = tracing_subscriber::fmt()
        .json()
        .without_time()
        .with_target(false)
        .with_writer(move || capture.clone())
        .finish();
    let _subscriber = tracing::subscriber::set_default(subscriber);
    let secret = r#"{"private_argument":"must-not-be-logged", "broken": private_token}"#;
    let mut emitter =
        ResponseStreamEventEmitter::new("resp_diagnostic".into(), "fixture-model".into(), 0);
    emitter.set_finish_reason(Some("tool_calls"));
    assert_eq!(emitter.function_call_status(secret), "incomplete");
    let warning: serde_json::Value =
        serde_json::from_slice(&bytes.lock().unwrap()).expect("one structured warning");
    assert_eq!(warning["fields"]["response_id"], "resp_diagnostic");
    assert_eq!(warning["fields"]["finish_reason"], "tool_calls");
    assert_eq!(warning["fields"]["argument_bytes"], secret.len());
    assert_eq!(warning["fields"]["json_error_class"], "Syntax");
    let logged = warning.to_string();
    for private in ["private_argument", "must-not-be-logged", "private_token"] {
        assert!(
            !logged.contains(private),
            "private argument content in warning"
        );
    }
    let terminal = emitter.emit_completed(None);
    assert_eq!(terminal["type"], "response.incomplete");
    assert!(terminal["response"]["incomplete_details"].is_null());
    assert!(terminal["response"]["error"].is_null());

    bytes.lock().unwrap().clear();
    for finish in [
        Some("length"),
        Some("abort"),
        Some("cancelled"),
        Some("steered"),
        Some("failed"),
        Some("error"),
        None,
    ] {
        emitter.set_finish_reason(finish);
        emitter.function_call_status(secret);
    }
    emitter.set_finish_reason(Some("tool_calls"));
    emitter.function_call_status(r#"{"valid":true}"#);
    assert!(
        bytes.lock().unwrap().is_empty(),
        "truncation, cancellation, steering and valid calls are not malformed-call warnings"
    );
}

use openai_protocol::common::Usage;

use super::*;
use crate::routers::grpc::common::responses::utils::{
    assert_responses_api_usage, responses_api_test_usage,
};

/// The terminal's error for a stream the request timeout cut counts under
/// `stream_error` in both shapes: with the event's type as its code (an
/// event without a code) and with the timeout's own code. The label set
/// does not grow with the code.
#[test]
fn a_stream_cut_by_the_request_timeout_counts_as_a_stream_error() {
    let without_a_code = json!({"code": "stream_error"});
    let with_the_timeouts_code = json!({"code": REQUEST_TIMEOUT_CODE});
    assert_eq!(failure_reason(Some(&without_a_code)), "stream_error");
    assert_eq!(
        failure_reason(Some(&with_the_timeouts_code)),
        "stream_error"
    );
    assert_eq!(
        failure_reason(Some(&json!({"code": "a_code_it_does_not_know"}))),
        "other"
    );
}

/// The events of one list-tools sequence for a server with no tools whose
/// listing failed with `error` (or succeeded, `None`), read back from the
/// stream in order.
async fn list_tools_sequence_events(error: Option<&str>) -> Vec<serde_json::Value> {
    let (tx, mut rx) = sse_channel();
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    emitter
        .emit_mcp_list_tools_sequence("conf", &[], error, &tx)
        .await
        .unwrap();
    drop(tx);
    let mut events = Vec::new();
    while let Some(Ok(bytes)) = rx.recv().await {
        let frame = String::from_utf8(bytes.to_vec()).unwrap();
        let data = frame.split("data: ").nth(1).unwrap().trim();
        events.push(serde_json::from_str::<serde_json::Value>(data).unwrap());
    }
    events
}

/// The list item's `in_progress` and `completed` (or `failed`) events
/// carry the item's id and output index, as the call events do: a
/// client joins them to the item's `output_item.added` by `item_id`.
#[tokio::test]
async fn the_list_tools_events_carry_the_items_id() {
    for error in [None, Some("Mcp error: -32603: tools/list refused")] {
        let events = list_tools_sequence_events(error).await;
        assert_eq!(events.len(), 4);
        let item_id = events[0]["item"]["id"]
            .as_str()
            .expect("the added item's id");
        assert!(!item_id.is_empty());
        for event in &events[1..3] {
            assert_eq!(event["item_id"], item_id, "{}", event["type"]);
            assert_eq!(event["output_index"], 0, "{}", event["type"]);
        }
        assert_eq!(events[3]["item"]["id"], item_id);
    }
}

/// A server whose listing failed streams its list item as `added`,
/// `mcp_list_tools.in_progress`, `mcp_list_tools.failed`, then
/// `output_item.done` whose item carries the error and no tools; a
/// listed server streams `completed` and an item without an error.
#[tokio::test]
async fn a_failed_listing_streams_the_failed_event_and_the_items_error() {
    let failed = list_tools_sequence_events(Some("Mcp error: -32603: tools/list refused")).await;
    assert_eq!(
        failed
            .iter()
            .map(|e| e["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "response.output_item.added",
            "response.mcp_list_tools.in_progress",
            "response.mcp_list_tools.failed",
            "response.output_item.done",
        ]
    );
    assert_eq!(failed[2].get("error"), None, "the failure is on the item");
    assert_eq!(failed[2]["output_index"], 0);
    let item = &failed[3]["item"];
    assert_eq!(item["type"], "mcp_list_tools");
    assert_eq!(item["server_label"], "conf");
    assert_eq!(item["error"], "Mcp error: -32603: tools/list refused");
    assert_eq!(item["tools"], json!([]));

    let listed = list_tools_sequence_events(None).await;
    assert_eq!(
        listed
            .iter()
            .map(|e| e["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "response.output_item.added",
            "response.mcp_list_tools.in_progress",
            "response.mcp_list_tools.completed",
            "response.output_item.done",
        ]
    );
    assert_eq!(listed[3]["item"].get("error"), None);
}

/// The streamed list item has the typed item's shape: the added and the
/// done item carry `id`, `type`, `server_label`, `tools` (and `error`
/// when the listing failed) and no `status`, as the non-streaming form
/// does; both parse as the typed item.
#[tokio::test]
async fn the_streamed_list_item_has_the_typed_items_shape() {
    fn keys(item: &serde_json::Value) -> Vec<&str> {
        let mut keys: Vec<&str> = item
            .as_object()
            .expect("an item object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        keys
    }
    for error in [None, Some("Mcp error: -32603: tools/list refused")] {
        let events = list_tools_sequence_events(error).await;
        let (added, done) = (&events[0]["item"], &events[3]["item"]);
        assert_eq!(keys(added), ["id", "server_label", "tools", "type"]);
        let mut done_keys = vec!["id", "server_label", "tools", "type"];
        if error.is_some() {
            done_keys.insert(0, "error");
        }
        assert_eq!(keys(done), done_keys);
        for item in [added, done] {
            let typed: ResponseOutputItem =
                serde_json::from_value(item.clone()).expect("the typed item");
            assert!(matches!(typed, ResponseOutputItem::McpListTools { .. }));
            assert_eq!(
                &serde_json::to_value(&typed).expect("the typed item serialises"),
                item,
                "the streamed item is the typed item's own serialisation"
            );
        }
    }
}

/// The typed text-delta event writes the frame the `json!` event wrote,
/// byte for byte (the same keys in the same order, the same escaping, the
/// same sequence number), and accumulates the text the same way.
#[tokio::test]
async fn the_typed_text_delta_is_the_json_event_frame_byte_for_byte() {
    let frames = |mut emitter: ResponseStreamEventEmitter, typed: bool| async move {
        let (tx, mut rx) = sse_channel();
        for (index, delta) in ["héllo \"there\"\n", "", "\u{1F600} tab\t"]
            .into_iter()
            .enumerate()
        {
            if typed {
                emitter
                    .send_text_delta(delta, 1, "msg_1", index, &tx)
                    .await
                    .unwrap();
            } else {
                // The event as the `json!` map wrote it before the typed form.
                emitter.accumulated_text.push_str(delta);
                let event = json!({
                    "type": OutputTextEvent::DELTA,
                    "sequence_number": emitter.next_sequence(),
                    "output_index": 1,
                    "item_id": "msg_1",
                    "content_index": index,
                    "delta": delta
                });
                emitter.send_event(&event, &tx).await.unwrap();
            }
        }
        drop(tx);
        let mut frames = Vec::new();
        while let Some(Ok(bytes)) = rx.recv().await {
            frames.push(bytes);
        }
        (frames, emitter.accumulated_text.clone())
    };
    let new = |seq| ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), seq);
    let (json_frames, json_text) = frames(new(7), false).await;
    let (typed_frames, typed_text) = frames(new(7), true).await;
    assert_eq!(typed_frames, json_frames);
    assert_eq!(typed_text, json_text);
    assert!(std::str::from_utf8(&typed_frames[0])
        .unwrap()
        .starts_with("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":"));
}

/// A call's arguments fragment sent typed writes the frame the `json!`
/// event wrote, byte for byte, under the function call's event name and
/// under the MCP call's.
#[tokio::test]
async fn the_typed_arguments_delta_is_the_json_event_frame_byte_for_byte() {
    let frames = |mut emitter: ResponseStreamEventEmitter, typed: bool| async move {
        let (tx, mut rx) = sse_channel();
        for kind in [
            FunctionCallEvent::ARGUMENTS_DELTA,
            McpEvent::CALL_ARGUMENTS_DELTA,
        ] {
            for delta in ["{\"city\":\"Pa", "", "r\u{ed}s\"}\n\u{1F600}"] {
                if typed {
                    emitter
                        .send_arguments_delta_best_effort(kind, delta, 1, "fc_1", &tx)
                        .await;
                } else {
                    // The event as the `json!` map wrote it before the typed form.
                    let event = json!({
                        "type": kind,
                        "sequence_number": emitter.next_sequence(),
                        "output_index": 1,
                        "item_id": "fc_1",
                        "delta": delta
                    });
                    emitter.send_event(&event, &tx).await.unwrap();
                }
            }
        }
        drop(tx);
        let mut frames = Vec::new();
        while let Some(Ok(bytes)) = rx.recv().await {
            frames.push(bytes);
        }
        frames
    };
    let new = || ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    let typed_frames = frames(new(), true).await;
    assert_eq!(typed_frames, frames(new(), false).await);
    assert!(std::str::from_utf8(&typed_frames[3])
        .unwrap()
        .starts_with("event: response.mcp_call_arguments.delta\ndata: {\"type\":\"response.mcp_call_arguments.delta\",\"sequence_number\":3,"));
}

/// The five typed lifecycle events write the frames the `json!` events
/// wrote, byte for byte (the same keys in the same order, the same
/// escaping, the same sequence numbers), the text part's text included,
/// for a foreground response and for a background one (whose created
/// frame says `queued`).
#[tokio::test]
async fn the_typed_lifecycle_events_are_the_json_event_frames_byte_for_byte() {
    let text = "h\u{e9}llo \"there\"\n\u{1F600}";
    let frames = |mut emitter: ResponseStreamEventEmitter, typed: bool, background: bool| async move {
        let (tx, mut rx) = sse_channel();
        emitter.background = background;
        emitter.accumulated_text.push_str(text);
        if typed {
            emitter.send_created(&tx).await.unwrap();
            emitter.send_in_progress(&tx).await.unwrap();
            emitter
                .send_content_part_added(2, "msg_1", 0, &tx)
                .await
                .unwrap();
            emitter.send_text_done(2, "msg_1", 0, &tx).await.unwrap();
            emitter
                .send_content_part_done(2, "msg_1", 0, &tx)
                .await
                .unwrap();
        } else {
            // The events as the `json!` maps wrote them before the typed forms.
            let events = [
                json!({
                    "type": ResponseEvent::CREATED,
                    "sequence_number": emitter.next_sequence(),
                    "response": {
                        "id": emitter.response_id,
                        "object": "response",
                        "created_at": emitter.created_at,
                        "status": if emitter.background { "queued" } else { "in_progress" },
                        "background": emitter.background,
                        "model": emitter.model,
                        "output": []
                    }
                }),
                json!({
                    "type": ResponseEvent::IN_PROGRESS,
                    "sequence_number": emitter.next_sequence(),
                    "response": {
                        "id": emitter.response_id,
                        "object": "response",
                        "status": "in_progress"
                    }
                }),
                json!({
                    "type": ContentPartEvent::ADDED,
                    "sequence_number": emitter.next_sequence(),
                    "output_index": 2,
                    "item_id": "msg_1",
                    "content_index": 0,
                    "part": { "type": "output_text", "text": "" }
                }),
                json!({
                    "type": OutputTextEvent::DONE,
                    "sequence_number": emitter.next_sequence(),
                    "output_index": 2,
                    "item_id": "msg_1",
                    "content_index": 0,
                    "text": text
                }),
                json!({
                    "type": ContentPartEvent::DONE,
                    "sequence_number": emitter.next_sequence(),
                    "output_index": 2,
                    "item_id": "msg_1",
                    "content_index": 0,
                    "part": { "type": "output_text", "text": text }
                }),
            ];
            for event in &events {
                emitter.send_event(event, &tx).await.unwrap();
            }
        }
        drop(tx);
        let mut frames = Vec::new();
        while let Some(Ok(bytes)) = rx.recv().await {
            frames.push(bytes);
        }
        frames
    };
    let new = |created_at| {
        ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), created_at)
    };
    let json_frames = frames(new(3), false, false).await;
    let typed_frames = frames(new(3), true, false).await;
    assert_eq!(typed_frames.len(), 5);
    assert_eq!(typed_frames, json_frames);
    // A background response: the created frame says queued, in the same place.
    let json_queued = frames(new(3), false, true).await;
    let typed_queued = frames(new(3), true, true).await;
    assert_eq!(typed_queued, json_queued);
    assert!(std::str::from_utf8(&typed_queued[0])
        .unwrap()
        .contains("\"status\":\"queued\",\"background\":true,\"model\""));
    assert!(std::str::from_utf8(&typed_frames[0])
        .unwrap()
        .starts_with("event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_test\",\"object\":\"response\",\"created_at\":3,\"status\":\"in_progress\""));
}

/// The terminal event's keys stay in the order the `json!` maps wrote
/// them with the moved-in values where the copies were: `output` last of
/// the response object's base keys, before the usage and the request
/// fields; `response` last of the event's keys. The kept item is the one
/// the event carries.
#[tokio::test]
async fn the_terminal_event_keeps_its_key_order_with_the_moved_in_values() {
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    emitter.set_original_request(
        serde_json::from_value(json!({"model": "test-model", "input": "hi"})).unwrap(),
    );
    let (tx, _rx) = sse_channel();
    let (index, id) = emitter.allocate_output_index(OutputItemKind::Message);
    let item = json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hello"}],
        "status": "completed"
    });
    emitter
        .send_output_item_done(index, item.clone(), RETAIN_OUTPUT_ITEM, &tx)
        .await
        .unwrap();
    emitter.complete_output_item(index);
    let usage = json!({"input_tokens": 3, "output_tokens": 2, "total_tokens": 5});

    let event = emitter.emit_completed(Some(usage.clone()));

    let keys = |value: &serde_json::Value| -> Vec<String> {
        value.as_object().unwrap().keys().cloned().collect()
    };
    assert_eq!(keys(&event), ["type", "sequence_number", "response"]);
    let response_keys = keys(&event["response"]);
    assert_eq!(
        response_keys[..8],
        [
            "id",
            "object",
            "created_at",
            "status",
            "background",
            "model",
            "output",
            "usage"
        ]
    );
    // The request's fields follow, as the builder adds them.
    assert!(
        response_keys[8..].iter().any(|key| key == "store")
            && response_keys[8..].iter().any(|key| key == "tool_choice"),
        "{response_keys:?}"
    );
    assert_eq!(event["response"]["output"], json!([item]));
    assert_eq!(event["response"]["usage"], usage);
    assert_eq!(event["type"], ResponseEvent::COMPLETED);
}

/// The two typed output item events write the frames the `json!` events
/// wrote, byte for byte (the same keys in the same order, the same
/// escaping, the same sequence numbers), and the item the done event
/// keeps is the one the terminal event carries.
#[tokio::test]
async fn the_output_item_frames_are_the_json_event_frames_byte_for_byte() {
    let item = json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "h\u{e9}llo \"there\"\n\u{1F600}"}],
        "status": "completed"
    });
    let new = |seq| ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), seq);
    let frames_of = |mut rx: SseReceiver| async move {
        let mut frames: Vec<Bytes> = Vec::new();
        while let Some(Ok(bytes)) = rx.recv().await {
            frames.push(bytes);
        }
        frames
    };

    let (tx, rx) = sse_channel();
    let mut emitter = new(7);
    let (index, _) = emitter.allocate_output_index(OutputItemKind::Message);
    emitter
        .send_output_item_added(index, &item, &tx)
        .await
        .unwrap();
    emitter
        .send_output_item_done(index, item.clone(), RETAIN_OUTPUT_ITEM, &tx)
        .await
        .unwrap();
    emitter.complete_output_item(index);
    drop(tx);
    let typed_frames = frames_of(rx).await;

    // The events as the `json!` maps wrote them before the typed forms.
    let (tx, rx) = sse_channel();
    let mut json_emitter = new(7);
    for kind in [OutputItemEvent::ADDED, OutputItemEvent::DONE] {
        let event = json!({
            "type": kind,
            "sequence_number": json_emitter.next_sequence(),
            "output_index": index,
            "item": &item
        });
        json_emitter.send_event(&event, &tx).await.unwrap();
    }
    drop(tx);
    let json_frames = frames_of(rx).await;

    assert_eq!(typed_frames.len(), 2);
    assert_eq!(typed_frames, json_frames);
    let terminal = emitter.emit_completed(None);
    assert_eq!(terminal["response"]["output"][0], item);
}

/// The frame a text delta writes for itself is the frame the encoder
/// writes for the same event serialized: both types, numbers of every
/// width, strings that take every kind of escape.
#[test]
fn text_delta_frames_write_the_bytes_of_the_serialized_event() {
    let mut encoder = SseEncoder::new();
    for (head, kind) in [
        (OUTPUT_TEXT_DELTA_HEAD, OutputTextEvent::DELTA),
        (REASONING_TEXT_DELTA_HEAD, "response.reasoning_text.delta"),
    ] {
        for (sequence_number, output_index, item_id, content_index, delta) in [
            (0, 0, "msg_1", 0, "Hello"),
            (u64::MAX, usize::MAX, "", usize::MAX, ""),
            (7, 1, "m\"\\", 2, "a\n\nb\"\\\u{1}\u{7f}\u{e9}\u{1f600}"),
        ] {
            let event = TextDelta {
                head,
                sequence_number,
                output_index,
                item_id,
                content_index,
                delta,
            };
            let mut written = BytesMut::new();
            event.write(&mut written).unwrap();
            let serialized = encoder
                .encode_event(
                    kind,
                    &json!({
                        "type": kind,
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "item_id": item_id,
                        "content_index": content_index,
                        "delta": delta,
                    }),
                )
                .unwrap();
            assert_eq!(written, serialized, "{kind} {sequence_number}");
        }
    }
}

/// The frame of an event: its `type` on the `event:` line and the JSON on
/// the `data:` line, the text escaped as JSON; an error frame has no
/// `event:` line.
#[tokio::test]
async fn an_event_frame_names_its_type_and_an_error_frame_does_not() {
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    let (tx, mut rx) = sse_channel();
    emitter
        .send_text_delta("héllo \"there\"\n", 1, "msg_1", 0, &tx)
        .await
        .unwrap();
    emitter
        .emit_error("engine unavailable", Some("server_error"), &tx)
        .await;
    drop(tx);
    let mut frames = Vec::new();
    while let Some(Ok(bytes)) = rx.recv().await {
        frames.push(String::from_utf8(bytes.to_vec()).unwrap());
    }
    let delta = OutputTextEvent::DELTA;
    assert_eq!(
        frames,
        [
            format!(
                "event: {delta}\ndata: {{\"type\":\"{delta}\",\"sequence_number\":0,\
                     \"output_index\":1,\"item_id\":\"msg_1\",\"content_index\":0,\
                     \"delta\":\"héllo \\\"there\\\"\\n\"}}\n\n"
            ),
            "data: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"engine \
                 unavailable\",\"param\":null,\"sequence_number\":1}\n\n"
                .to_string(),
        ]
    );
}

/// A run that ends as failed before its terminal (the tool-call limit)
/// stores what the stream carries: `failed`, the error, the items still
/// held by the emitter, the usage; the terminal drains them afterwards.
#[tokio::test]
async fn the_failed_final_object_keeps_the_items_the_error_and_the_usage() {
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    let (index, id) = emitter.allocate_output_index(OutputItemKind::FunctionCall);
    let item = json!({"id": id, "type": "function_call", "call_id": "call_test", "name": "lookup", "arguments": "{}", "status": "completed"});
    let (tx, _rx) = sse_channel();
    emitter
        .send_output_item_added(index, &item, &tx)
        .await
        .unwrap();
    emitter
        .send_output_item_done(index, item, RETAIN_OUTPUT_ITEM, &tx)
        .await
        .unwrap();
    emitter.complete_output_item(index);
    let error = json!({
        "code": "max_tool_calls_exceeded",
        "message": "Reached the max_tool_calls limit (1) before executing the remaining tool calls.",
    });

    let response = emitter.finalize_failed(
        Some(ResponseUsage::from(&Usage::from_counts(3, 2))),
        error.clone(),
    );

    assert_eq!(response.status, ResponseStatus::Failed);
    assert_eq!(response.error, Some(error.clone()));
    assert!(response.incomplete_details.is_none());
    let stored = serde_json::to_value(&response).unwrap();
    assert_eq!(stored["output"][0]["name"], "lookup", "{stored}");
    assert_eq!(stored["usage"]["total_tokens"], 5, "{stored}");
    // The terminal built afterwards still carries the item.
    let terminal = emitter.emit_completed(None);
    assert_eq!(terminal["response"]["output"][0]["name"], "lookup");
}

#[tokio::test]
async fn the_error_event_is_remembered_for_the_stream_runner() {
    let mut emitter =
        ResponseStreamEventEmitter::new("resp_test".to_string(), "test-model".to_string(), 1);
    assert!(emitter.take_error().is_none());
    let (tx, mut rx) = sse_channel();
    emitter
        .emit_error("Pipeline execution failed", Some("pipeline_error"), &tx)
        .await;
    let frame = rx.recv().await.expect("the error frame").expect("a frame");
    assert!(frame.starts_with(b"data: "), "{frame:?}");
    let error = emitter.take_error().expect("remembered");
    assert_eq!(error["code"], "pipeline_error");
    assert_eq!(error["message"], "Pipeline execution failed");
    assert!(emitter.take_error().is_none(), "taken once");
}

/// The emitter's error event is the frame every router writes for a
/// failed Responses stream, at the emitter's own sequence number.
#[tokio::test]
async fn the_error_frame_is_the_shared_responses_error_frame() {
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    let (tx, mut rx) = sse_channel();
    emitter.send_created(&tx).await.unwrap();
    emitter
        .emit_error(
            "Stream read error: connection reset",
            Some("stream_error"),
            &tx,
        )
        .await;
    drop(tx);
    let _created = rx.recv().await.expect("the created frame");
    let frame = rx.recv().await.expect("the error frame").expect("a frame");
    assert_eq!(
        frame,
        SseEncoder::new().encode_responses_error(
            "stream_error",
            "Stream read error: connection reset",
            1
        )
    );
    assert!(rx.recv().await.is_none(), "nothing follows the error event");
}

/// The terminal event echoes the request's sampling numbers unwidened
/// (`0.7`, not `0.699999988079071`) and in the spelling of a 64-bit JSON
/// number: a `top_p` of 0.000001 reads `1e-6`, not `0.000001`.
#[tokio::test]
async fn the_terminal_event_echoes_the_request_numbers_as_written() {
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    emitter.set_original_request(
        serde_json::from_value(json!({
            "model": "test-model",
            "input": "hi",
            "temperature": 0.7,
            "top_p": 0.000001
        }))
        .unwrap(),
    );
    let (tx, mut rx) = sse_channel();
    emitter.emit_terminal(None, None, &tx).await.unwrap();
    let frame = rx
        .recv()
        .await
        .expect("the terminal frame")
        .expect("a frame");
    let text = String::from_utf8(frame.to_vec()).unwrap();
    assert!(
        text.contains(r#""temperature":0.7,"top_p":1e-6,"#),
        "{text}"
    );
}

#[test]
fn finalized_streaming_response_serializes_responses_api_usage() {
    let emitter =
        ResponseStreamEventEmitter::new("resp_test".to_string(), "test-model".to_string(), 1);

    let usage = ResponseUsage::from(&responses_api_test_usage());
    let wire = serde_json::to_value(emitter.finalize(Some(usage)))
        .expect("finalized response should serialize");
    assert_responses_api_usage(&wire);
}

#[tokio::test]
async fn await_stream_startup_returns_backend_error_response() {
    let (startup_tx, startup_rx) = stream_startup_channel();
    let (_sse_tx, sse_rx) = sse_channel();
    let response = Response::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .body(Body::empty())
        .expect("test response should build");

    assert!(startup_tx.send(Err(response)).is_ok());

    let response = await_stream_startup(startup_rx, sse_rx).await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn signal_stream_startup_returns_sse_after_success() {
    let (startup_tx, startup_rx) = stream_startup_channel();
    let (_sse_tx, sse_rx) = sse_channel();
    let mut startup = Some(startup_tx);

    assert!(signal_stream_startup(&mut startup, Ok(None)));
    assert!(startup.is_none());

    let response = await_stream_startup(startup_rx, sse_rx).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(CONTENT_TYPE),
        Some(&HeaderValue::from_static(
            "text/event-stream; charset=utf-8"
        ))
    );
}

/// A started stream's answer names the parser the barrier brought, as the
/// chat answer it relays does; a start without one names none.
#[tokio::test]
async fn a_started_stream_names_the_parser_the_barrier_brought() {
    for parser in [Some(HeaderValue::from_static("symphony")), None] {
        let (startup_tx, startup_rx) = stream_startup_channel();
        let (_sse_tx, sse_rx) = sse_channel();
        let mut startup = Some(startup_tx);

        assert!(signal_stream_startup(&mut startup, Ok(parser.clone())));

        let response = await_stream_startup(startup_rx, sse_rx).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers().get(symphony::HEADER), parser.as_ref());
    }
}
