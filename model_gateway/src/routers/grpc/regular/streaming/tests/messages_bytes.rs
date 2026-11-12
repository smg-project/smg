//! The exact SSE bytes of Messages streams, pinned per scenario: text, text
//! under Symphony (the answer over several chunks: the first delta opens the
//! block through the content-block machine, the following ones go straight
//! from the borrowed text to the frame), thinking then text, a tool call and a
//! length stop, driven over scripted
//! engine frames and compared byte for byte with
//! `fixtures/messages_<scenario>.sse`. The fixtures hold the bytes the
//! per-event copies produced, so any change to the wire format of a Messages
//! event, intended or not, fails here. Set `SMG_RECORD_SSE_FIXTURES=1` to
//! rewrite them from the current code.

use std::sync::Arc;

use futures::StreamExt as _;
use smg_grpc_client::vllm_engine::proto;
use symphony::{formats::qwen3, CallSyntax};
use tokio::sync::oneshot;

use super::{
    super::*,
    eof::{
        chat_spec, chunk, complete, complete_with_prompt, dispatch, messages_spec, processor, pull,
        scripted_stream, staged_stream, CharacterTokenizer,
    },
    sse_bytes::assert_recorded_bytes,
};
use crate::routers::grpc::regular::symphony::Assignment;

struct MessagesScenario {
    name: &'static str,
    with_tools: bool,
    thinking: bool,
    /// Symphony's Qwen3 table reads the message (the content-block machine writes the events)
    /// instead of the reasoning and tool parsers.
    symphony: bool,
    /// The served model; a Kimi name gives the tool call its deterministic id.
    model: &'static str,
    frames: Vec<proto::GenerateResponse>,
    /// A substring the scenario's bytes must carry, so the scenario keeps
    /// exercising the path it was written for.
    must_contain: &'static str,
}

fn scenarios() -> Vec<MessagesScenario> {
    vec![
        MessagesScenario {
            name: "messages_text",
            with_tools: false,
            thinking: false,
            symphony: false,
            model: "fixture-model",
            frames: vec![
                chunk(0, "Hello"),
                chunk(0, ", world"),
                complete_with_prompt(0, "stop", 10),
            ],
            must_contain: r#""text":", world""#,
        },
        MessagesScenario {
            name: "messages_symphony_text",
            with_tools: false,
            thinking: false,
            symphony: true,
            model: "fixture-model",
            frames: vec![
                chunk(0, "Hello"),
                chunk(0, ", wor"),
                chunk(0, "ld"),
                chunk(0, " "),
                chunk(0, "caf\u{e9}"),
                chunk(0, " \"again\""),
                complete_with_prompt(0, "stop", 10),
            ],
            must_contain: r#""text":"ld""#,
        },
        MessagesScenario {
            name: "messages_thinking_then_text",
            with_tools: false,
            thinking: true,
            symphony: false,
            model: "fixture-model",
            frames: vec![
                chunk(0, "<think>because"),
                chunk(0, " of this</think>the answer"),
                complete(0, "stop"),
            ],
            must_contain: r#""thinking":"because""#,
        },
        MessagesScenario {
            name: "messages_tool_use",
            with_tools: true,
            thinking: false,
            symphony: false,
            model: "kimi-fixture",
            frames: vec![
                chunk(0, r#"{"name": "lookup", "#),
                chunk(0, r#""parameters": {"q": "x"}}"#),
                complete(0, "stop"),
            ],
            must_contain: r#""type":"tool_use""#,
        },
        MessagesScenario {
            name: "messages_length",
            with_tools: false,
            thinking: false,
            symphony: false,
            model: "fixture-model",
            frames: vec![chunk(0, "a"), complete(0, "length")],
            must_contain: r#""stop_reason":"max_tokens""#,
        },
    ]
}

/// The SSE bytes of one scenario.
async fn stream_bytes(scenario: &MessagesScenario) -> Vec<u8> {
    let (stream, server) = scripted_stream(scenario.frames.clone(), "0").await;
    let dispatch = context::DispatchMetadata {
        request_id: "msg-bytes".to_string(),
        model: scenario.model.to_string(),
        created: 1,
        weight_version: None,
    };
    let spec = MessagesResponseSpec {
        prompt_tokens: 0,
        // The `length` scenario's one token is its whole budget: the recorded
        // bytes say `max_tokens`.
        max_tokens: 1,
        context_window: None,
        thinking_requested: scenario.thinking,
        thinking_text_returned: true,
        reasoning_in_prefill: false,
        tool_choice: None,
        has_tools: scenario.with_tools,
        history_tool_calls_count: 0,
        chat_tools: if scenario.with_tools {
            chat_spec(true).tools.unwrap()
        } else {
            Vec::new().into()
        },
        stop_sequences: None,
        symphony: scenario.symphony.then(|| Assignment {
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: "<|im_start|>assistant\n".to_string(),
            forbids_calls: false,
            past_the_end: false,
            span: Span::none(),
            marked: Arc::default(),
        }),
        first_tool_call_only: false,
        parser_names: None,
    };
    let (tx, mut rx) = sse_channel();
    let result = processor(scenario.with_tools)
        .process_messages_streaming_chunks(
            stream,
            dispatch,
            Arc::new(CharacterTokenizer::default()),
            (None, None, false, false, true),
            spec,
            &tx,
            None,
        )
        .await;
    drop(tx);
    let mut bytes = Vec::new();
    while let Some(frame) = rx.recv().await {
        bytes.extend_from_slice(&frame.expect("successful SSE write"));
    }
    server.abort();
    assert!(result.is_ok(), "{}: {result:?}", scenario.name);
    bytes
}

#[tokio::test]
async fn messages_stream_bytes_match_the_recorded_fixtures() {
    for scenario in scenarios() {
        let actual = stream_bytes(&scenario).await;
        assert_recorded_bytes(scenario.name, scenario.must_contain, &actual);
    }
}

/// As the chat stream packs its frames: `message_start` leaves at once, before
/// the engine is read (as it always has); the frame carrying the first token
/// (the block's start and its delta) leaves on its own as soon as the message
/// is processed, and after it the events the engine has ready at one poll
/// leave as one frame: the second delta, ready with the first, leaves alone
/// once the first frame is out (the loop does not wait for more); the two
/// after the gate share a frame; the stream's tail (the block's stop,
/// `message_delta`, `message_stop`), written after the read, is one frame.
#[tokio::test]
async fn ready_engine_messages_share_a_frame_after_the_first_token_on_the_messages_stream() {
    let (gate, gated) = oneshot::channel();
    let (stream, server) = staged_stream(
        vec![
            (None, vec![chunk(0, "a"), chunk(0, "b")]),
            (
                Some(gated),
                vec![chunk(0, "c"), chunk(0, "d"), complete(0, "stop")],
            ),
        ],
        "0",
    )
    .await;
    let (tx, mut rx) = sse_channel();
    let dispatch = context::DispatchMetadata {
        request_id: "msg-frames".to_string(),
        model: "fixture-model".to_string(),
        created: 1,
        weight_version: None,
    };
    let processor = processor(false);
    let reading = async {
        let result = processor
            .process_messages_streaming_chunks(
                stream,
                dispatch,
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, true),
                messages_spec(false),
                &tx,
                None,
            )
            .await;
        drop(tx);
        result
    };
    let reader = async {
        let mut frames = Vec::new();
        // The start event, then the two messages ready at once: the reader
        // parks at the gate only after their three frames left.
        for _ in 0..3 {
            frames.push(rx.recv().await.expect("a frame").expect("a write"));
        }
        gate.send(()).expect("the reader waits at the gate");
        while let Some(frame) = rx.recv().await {
            frames.push(frame.expect("a write"));
        }
        frames
    };
    let (result, frames) = tokio::join!(reading, reader);
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    let events: Vec<(usize, String)> = frames
        .iter()
        .map(|frame| {
            let text = String::from_utf8_lossy(frame).to_string();
            (text.matches("event: ").count(), text)
        })
        .collect();
    let shape: Vec<usize> = events.iter().map(|(count, _)| *count).collect();
    assert_eq!(
        shape,
        [1, 2, 1, 2, 3],
        "message_start alone, the block's start with the first token, the second token alone, the two after the gate together, the tail: {events:?}"
    );
    assert!(
        events[0].1.starts_with("event: message_start\n"),
        "{events:?}"
    );
    assert!(
        events[1].1.starts_with("event: content_block_start\n")
            && events[1].1.contains(r#""text":"a""#),
        "{events:?}"
    );
    assert!(events[2].1.contains(r#""text":"b""#), "{events:?}");
    assert!(
        events[3].1.contains(r#""text":"c""#) && events[3].1.contains(r#""text":"d""#),
        "{events:?}"
    );
    assert!(
        events[4].1.starts_with("event: content_block_stop\n")
            && events[4]
                .1
                .ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
        "{events:?}"
    );
}

/// The first token's frame reaches the connection before the reading runs
/// on into a burst: the body that drives the reading hands the frame to the
/// connection and answers the connection's next pull with pending, because
/// the packed read yields twice at the hand-off (`next_packed`). With one
/// yield that pull would run the reading on through the engine messages
/// ready behind the first token and return the burst's frame instead: the
/// first text delta would reach the client behind the burst.
#[tokio::test]
async fn the_first_tokens_frame_is_pending_to_the_connection_before_the_burst_is_read() {
    // Four chunks ready at once: the first token, then a burst behind it.
    let (stream, server) = scripted_stream(
        vec![
            chunk(0, "a"),
            chunk(0, "b"),
            chunk(0, "c"),
            chunk(0, "d"),
            complete(0, "stop"),
        ],
        "0",
    )
    .await;
    let (tx, rx) = driven_channel();
    let processor = processor(false);
    let spec = messages_spec(false);
    let reading = async move {
        let result = processor
            .process_messages_streaming_chunks(
                stream,
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, false),
                spec,
                &tx,
                None,
            )
            .await;
        assert!(result.is_ok(), "{result:?}");
        ReadingEnd::Complete
    };
    let mut body = DrivenBody::new(Box::pin(reading), rx);
    // `message_start` (the body's own yield follows it), then the frame
    // carrying the block's start and the first delta.
    let mut frames = Vec::new();
    loop {
        let frame = body
            .next()
            .await
            .expect("the body ends only after the first token")
            .expect("a write");
        let text = String::from_utf8(frame.to_vec()).expect("text");
        let first_token = text.contains(r#""text":"a""#);
        frames.push(text);
        if first_token {
            break;
        }
    }
    assert_eq!(
        frames.len(),
        2,
        "message_start, then the first token's frame: {frames:?}"
    );
    assert!(
        frames[1].contains("event: content_block_start"),
        "the first token's frame opens the block: {frames:?}"
    );
    // The pull after it: pending, so the connection writes the first
    // token's frame before the reading runs on. With one yield at the
    // hand-off it is the burst's frame.
    let next = pull(&mut body).await;
    assert!(
        next.is_pending(),
        "the pull after the first token's frame must be pending, not a frame: {next:?}"
    );
    let rest: Vec<String> = body
        .map(|frame| String::from_utf8(frame.expect("a write").to_vec()).expect("text"))
        .collect()
        .await;
    server.abort();
    assert!(
        rest.iter().any(|frame| frame.contains(r#""text":"b""#)),
        "the burst follows: {rest:?}"
    );
    assert!(
        rest.last()
            .is_some_and(|frame| frame.contains("event: message_stop")),
        "the stream ends after the burst: {rest:?}"
    );
}
