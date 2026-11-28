use tokio::sync::mpsc;

use super::*;
use crate::routers::grpc::common::responses::utils::decode_reasoning_content;

fn chunk(delta: serde_json::Value, finish_reason: Option<&str>) -> ChatCompletionStreamResponse {
    serde_json::from_value(json!({
        "id": "chat_test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    }))
    .unwrap()
}

/// Drive `chunks` through a fresh emitter; return the emitted events in
/// order plus the terminal `response.completed`/`response.incomplete`
/// event.
async fn stream_with_terminal(
    request: serde_json::Value,
    chunks: &[ChatCompletionStreamResponse],
) -> (Vec<serde_json::Value>, serde_json::Value) {
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    emitter.set_original_request(serde_json::from_value(request).unwrap());
    let (tx, mut rx) = mpsc::channel(256);
    for chunk in chunks {
        emitter.process_chunk(chunk, &tx).await.unwrap();
    }
    drop(tx);
    let mut events = Vec::new();
    while let Some(Ok(bytes)) = rx.recv().await {
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let data = text.lines().find_map(|l| l.strip_prefix("data: ")).unwrap();
        events.push(serde_json::from_str(data).unwrap());
    }
    let terminal = emitter.emit_completed(None);
    (events, terminal)
}

/// Drive `chunks` through a fresh emitter; return the emitted events in
/// order plus the `output` array of the terminal event.
async fn stream(
    request: serde_json::Value,
    chunks: &[ChatCompletionStreamResponse],
) -> (Vec<serde_json::Value>, serde_json::Value) {
    let (events, terminal) = stream_with_terminal(request, chunks).await;
    (events, terminal["response"]["output"].clone())
}

fn types(events: &[serde_json::Value]) -> Vec<&str> {
    events.iter().map(|e| e["type"].as_str().unwrap()).collect()
}

#[tokio::test]
async fn reasoning_then_function_call_stream_as_paired_items() {
    let (events, output) = stream(
        json!({"model": "test-model", "input": "hi", "include": ["reasoning.encrypted_content"]}),
        &[
            chunk(json!({"reasoning_content": "thi"}), None),
            chunk(json!({"reasoning_content": "nk"}), None),
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": ""}}]}),
                None,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":"}}]}),
                None,
            ),
            chunk(
                json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"Paris\"}"}}]}),
                None,
            ),
            chunk(json!({}), Some("tool_calls")),
        ],
    )
    .await;

    assert_eq!(
        types(&events),
        [
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
        ]
    );
    assert_eq!(
        events[1]["part"],
        json!({"type": "reasoning_text", "text": ""})
    );
    assert_eq!(events[5]["part"], output[0]["content"][0]);
    for event in &events[1..6] {
        assert_eq!(event["item_id"], output[0]["id"]);
        assert_eq!(event["output_index"], 0);
        assert_eq!(event["content_index"], 0);
    }
    let seq: Vec<u64> = events
        .iter()
        .map(|e| e["sequence_number"].as_u64().unwrap())
        .collect();
    assert!(seq.windows(2).all(|w| w[0] < w[1]));

    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(
        output[0]["content"],
        json!([{"type": "reasoning_text", "text": "think"}])
    );
    assert_eq!(
        decode_reasoning_content(output[0]["encrypted_content"].as_str().unwrap()).as_deref(),
        Some("think")
    );
    assert_eq!(output[1]["type"], "function_call");
    assert_eq!(output[1]["call_id"], "call_1");
    assert_eq!(output[1]["name"], "get_weather");
    assert_eq!(output[1]["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(output[1]["status"], "completed");
    // Each output_item.done carries the item response.completed reports.
    assert_eq!(events[6]["item"], output[0]);
    assert_eq!(events[11]["item"], output[1]);
    assert_eq!(events[10]["arguments"], "{\"city\":\"Paris\"}");
}

/// The reasoning text delta and the function-call arguments delta of a
/// chunk write the frames their `json!` events wrote, byte for byte: the
/// same keys in the same order, the same escaping, the same sequence
/// numbers.
#[tokio::test]
async fn the_typed_reasoning_and_arguments_deltas_are_the_json_event_frames_byte_for_byte() {
    let new = || ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    let mut emitter = new();
    let (tx, mut rx) = mpsc::channel(256);
    for chunk in [
        chunk(json!({"reasoning_content": "th\u{e9} \"why\"\n"}), None),
        chunk(json!({"reasoning_content": "\u{1F600} tab\t"}), None),
        chunk(
            json!({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":"}}]}),
            None,
        ),
        chunk(
            json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"Paris\"}\n"}}]}),
            None,
        ),
    ] {
        emitter.process_chunk(&chunk, &tx).await.unwrap();
    }
    drop(tx);
    let mut typed = Vec::new();
    while let Some(Ok(bytes)) = rx.recv().await {
        typed.push(bytes);
    }
    assert_eq!(typed.len(), 10);
    let item_id = |frame: usize| {
        let text = std::str::from_utf8(&typed[frame]).unwrap();
        let data = text.lines().find_map(|l| l.strip_prefix("data: ")).unwrap();
        serde_json::from_str::<serde_json::Value>(data).unwrap()["item_id"].clone()
    };

    // The four deltas as the `json!` maps wrote them before the typed form.
    let mut emitter = new();
    let (tx, mut rx) = mpsc::channel(8);
    for event in [
        json!({"type": "response.reasoning_text.delta", "sequence_number": 2,
            "output_index": 0, "item_id": item_id(2), "content_index": 0,
            "delta": "th\u{e9} \"why\"\n"}),
        json!({"type": "response.reasoning_text.delta", "sequence_number": 3,
            "output_index": 0, "item_id": item_id(3), "content_index": 0,
            "delta": "\u{1F600} tab\t"}),
        json!({"type": "response.function_call_arguments.delta", "sequence_number": 8,
            "output_index": 1, "item_id": item_id(8), "delta": "{\"city\":"}),
        json!({"type": "response.function_call_arguments.delta", "sequence_number": 9,
            "output_index": 1, "item_id": item_id(9), "delta": "\"Paris\"}\n"}),
    ] {
        emitter.send_event(&event, &tx).await.unwrap();
    }
    drop(tx);
    let mut json_frames = Vec::new();
    while let Some(Ok(bytes)) = rx.recv().await {
        json_frames.push(bytes);
    }
    assert_eq!(
        [&typed[2], &typed[3], &typed[8], &typed[9]],
        [
            &json_frames[0],
            &json_frames[1],
            &json_frames[2],
            &json_frames[3]
        ]
    );
}

#[tokio::test]
async fn one_chunk_carrying_reasoning_content_and_a_call_emits_them_in_that_order() {
    // One chunk with every kind of delta at once: the reasoning item
    // opens first, the content closes it and opens the message, the
    // call opens its item, and the finish closes the message and the
    // call, in that order.
    let (events, output) = stream(
        json!({"model": "test-model", "input": "hi"}),
        &[chunk(
            json!({
                "reasoning_content": "think",
                "content": "answer",
                "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"}}]
            }),
            Some("tool_calls"),
        )],
    )
    .await;
    assert_eq!(
        types(&events),
        [
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.function_call_arguments.done",
            "response.output_item.done",
        ]
    );
    assert_eq!(
        output
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["reasoning", "message", "function_call"]
    );
    assert_eq!(output[0]["content"][0]["text"], "think");
    assert_eq!(output[1]["content"][0]["text"], "answer");
    assert_eq!(output[2]["arguments"], "{}");
}

#[tokio::test]
async fn custom_tool_call_streams_as_custom_input() {
    let (events, output) = stream(
        json!({"model": "test-model", "input": "hi",
            "tools": [{"type": "custom", "name": "emit_command"}]}),
        &[
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "call_c", "type": "function",
                    "function": {"name": "emit_command", "arguments": "{\"input\": \"pwd\"}"}}]}),
                None,
            ),
            chunk(json!({}), Some("tool_calls")),
        ],
    )
    .await;

    assert_eq!(
        types(&events),
        [
            "response.output_item.added",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
            "response.output_item.done",
        ]
    );
    assert_eq!(events[0]["item"]["type"], "custom_tool_call");
    assert_eq!(events[1]["delta"], "pwd");
    assert_eq!(events[2]["input"], "pwd");
    assert_eq!(output[0]["type"], "custom_tool_call");
    assert_eq!(output[0]["input"], "pwd");
    // The item's id is a custom tool call's own, the same on the event and
    // in the final output.
    let id = events[0]["item"]["id"].as_str().unwrap_or_default();
    assert!(id.starts_with("ctc_"), "{}", events[0]);
    assert_eq!(output[0]["id"], id);
    assert_eq!(output[0]["call_id"], "call_c");
    assert_eq!(events[3]["item"], output[0]);
}

#[tokio::test]
async fn text_after_reasoning_closes_the_reasoning_item_first() {
    let (events, output) = stream(
        json!({"model": "test-model", "input": "hi"}),
        &[
            chunk(json!({"reasoning_content": "why"}), None),
            chunk(json!({"content": "Hel"}), None),
            chunk(json!({"content": "lo"}), None),
            chunk(json!({}), Some("stop")),
        ],
    )
    .await;

    assert_eq!(
        types(&events),
        [
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ]
    );
    assert_eq!(events[1]["part"]["type"], "reasoning_text");
    assert_eq!(events[4]["part"], output[0]["content"][0]);
    assert_eq!(events[7]["part"]["type"], "output_text");
    assert_eq!(events[11]["part"], output[1]["content"][0]);
    assert_eq!(events[1]["item_id"], output[0]["id"]);
    assert_eq!(events[7]["item_id"], output[1]["id"]);
    assert_eq!(output[0]["type"], "reasoning");
    assert!(output[0].get("encrypted_content").is_none());
    assert_eq!(output[1]["type"], "message");
    assert_eq!(output[1]["content"][0]["text"], "Hello");
    // The deltas through the emitter's own path carry the item's id, the
    // text of their chunk and consecutive sequence numbers.
    assert_eq!(events[8]["item_id"], output[1]["id"]);
    assert_eq!(events[9]["item_id"], output[1]["id"]);
    assert_eq!(
        (events[8]["delta"].as_str(), events[9]["delta"].as_str()),
        (Some("Hel"), Some("lo"))
    );
    for pair in events.windows(2) {
        assert_eq!(
            pair[1]["sequence_number"].as_u64(),
            pair[0]["sequence_number"].as_u64().map(|s| s + 1)
        );
    }
}

#[tokio::test]
async fn reasoning_truncation_closes_content_part_before_incomplete() {
    let (events, terminal) = stream_with_terminal(
        json!({"model": "test-model", "input": "hi"}),
        &[
            chunk(json!({"reasoning_content": "still thinking"}), None),
            chunk(json!({}), Some("length")),
        ],
    )
    .await;
    assert_eq!(
        types(&events),
        [
            "response.output_item.added",
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ]
    );
    assert_eq!(terminal["type"], "response.incomplete");
    assert_eq!(terminal["response"]["status"], "incomplete");
    assert_eq!(
        terminal["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(
        events[4]["part"],
        json!({"type": "reasoning_text", "text": "still thinking"})
    );
    assert_eq!(events[4]["part"], events[5]["item"]["content"][0]);
    assert_eq!(events[1]["item_id"], events[4]["item_id"]);
    assert_eq!(events[5]["item"], terminal["response"]["output"][0]);
}

#[tokio::test]
async fn length_truncation_closes_tool_calls_and_marks_incomplete() {
    // Grammar-constrained turn: one full call, a second call truncated
    // mid-arguments by max_output_tokens.
    let (events, terminal) = stream_with_terminal(
        json!({"model": "test-model", "input": "hi"}),
        &[
            chunk(
                json!({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]}),
                None,
            ),
            chunk(
                json!({"tool_calls": [{"index": 1, "id": "call_b", "type": "function",
                    "function": {"name": "get_time", "arguments": "{\"tz\":"}}]}),
                None,
            ),
            chunk(json!({}), Some("length")),
        ],
    )
    .await;

    // Both streamed calls close even though the turn ended truncated.
    assert_eq!(
        types(&events),
        [
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.function_call_arguments.done",
            "response.output_item.done",
        ]
    );

    assert_eq!(terminal["type"], "response.incomplete");
    assert_eq!(terminal["response"]["status"], "incomplete");
    assert_eq!(
        terminal["response"]["incomplete_details"],
        json!({"reason": "max_output_tokens"})
    );

    let output = &terminal["response"]["output"];
    assert_eq!(output.as_array().map(Vec::len), Some(2));
    assert_eq!(output[0]["type"], "function_call");
    assert_eq!(output[0]["name"], "get_weather");
    assert_eq!(output[0]["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(output[1]["type"], "function_call");
    assert_eq!(output[1]["name"], "get_time");
    assert_eq!(output[1]["arguments"], "{\"tz\":");
    assert_eq!(output[0]["status"], "completed");
    assert_eq!(output[1]["status"], "incomplete");
    assert_eq!(events[5]["item"], output[0]);
    assert_eq!(events[7]["item"], output[1]);
}

#[tokio::test]
async fn length_finish_keeps_complete_streamed_tool_arguments_completed() {
    let (events, terminal) = stream_with_terminal(
        json!({"model":"test-model","input":"hi"}),
        &[
            chunk(
                json!({"tool_calls":[{"index":0,"id":"call_a","type":"function",
            "function":{"name":"weather","arguments":"{}"}}]}),
                None,
            ),
            chunk(json!({}), Some("length")),
        ],
    )
    .await;
    assert_eq!(terminal["type"], "response.incomplete");
    assert_eq!(terminal["response"]["output"][0]["status"], "completed");
    let done = events
        .iter()
        .find(|e| e["type"] == "response.output_item.done")
        .unwrap();
    assert_eq!(done["item"], terminal["response"]["output"][0]);
}

#[tokio::test]
async fn length_truncated_text_ends_with_response_incomplete() {
    let (events, terminal) = stream_with_terminal(
        json!({"model": "test-model", "input": "hi"}),
        &[
            chunk(json!({"content": "He"}), None),
            chunk(json!({"content": "y"}), None),
            chunk(json!({}), Some("length")),
        ],
    )
    .await;

    // The text item still closes normally; only the terminal event and
    // response status change.
    assert_eq!(
        types(&events),
        [
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
        ]
    );
    assert_eq!(terminal["type"], "response.incomplete");
    assert_eq!(terminal["response"]["status"], "incomplete");
    assert_eq!(
        terminal["response"]["incomplete_details"],
        json!({"reason": "max_output_tokens"})
    );
    assert_eq!(
        terminal["response"]["output"][0]["content"][0]["text"],
        "Hey"
    );
}

#[tokio::test]
async fn length_truncated_finalize_persists_incomplete_status() {
    let mut emitter = ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
    emitter.set_original_request(
        serde_json::from_value(json!({"model": "test-model", "input": "hi"})).unwrap(),
    );
    let (tx, mut rx) = mpsc::channel(256);
    emitter
        .process_chunk(&chunk(json!({"content": "par"}), None), &tx)
        .await
        .unwrap();
    emitter
        .process_chunk(&chunk(json!({}), Some("length")), &tx)
        .await
        .unwrap();
    drop(tx);
    while rx.recv().await.is_some() {}

    let wire = serde_json::to_value(emitter.finalize(None)).unwrap();
    assert_eq!(wire["status"], "incomplete");
    assert_eq!(
        wire["incomplete_details"],
        json!({"reason": "max_output_tokens"})
    );
}

#[tokio::test]
async fn a_cut_custom_tool_call_reads_incomplete_on_the_terminal_event_and_the_stored_object() {
    for finish in [Some("abort"), Some("stop")] {
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_test".into(), "test-model".into(), 0);
        emitter.set_original_request(
            serde_json::from_value(json!({"model": "test-model", "input": "hi",
                "tools": [{"type": "custom", "name": "emit_command"}]}))
            .unwrap(),
        );
        let (tx, mut rx) = mpsc::channel(256);
        emitter
            .process_chunk(
                &chunk(
                    json!({"tool_calls": [{"index": 0, "id": "call_c", "type": "function",
                        "function": {"name": "emit_command", "arguments": "{\"input\": \"pw"}}]}),
                    None,
                ),
                &tx,
            )
            .await
            .unwrap();
        emitter
            .process_chunk(&chunk(json!({}), finish), &tx)
            .await
            .unwrap();
        drop(tx);
        while rx.recv().await.is_some() {}

        let stored = serde_json::to_value(emitter.finalize(None)).unwrap();
        assert_eq!(stored["status"], "incomplete", "{finish:?}");
        assert!(stored["incomplete_details"].is_null(), "{finish:?}");
        assert_eq!(
            stored["output"][0]["type"], "custom_tool_call",
            "{finish:?}"
        );
        let terminal = emitter.emit_completed(None);
        assert_eq!(terminal["type"], "response.incomplete", "{finish:?}");
        assert_eq!(terminal["response"]["status"], "incomplete", "{finish:?}");
        assert!(
            terminal["response"]["incomplete_details"].is_null(),
            "{finish:?}"
        );
        assert_eq!(
            terminal["response"]["output"][0]["type"], "custom_tool_call",
            "{finish:?}"
        );
    }
}
