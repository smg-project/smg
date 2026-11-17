//! The exact SSE bytes of chat and completion streams, pinned per scenario.
//!
//! Each scenario drives `process_streaming_chunks` (chat) or
//! `process_completion_streaming_chunks` (completions) over scripted engine
//! frames and compares the concatenated SSE events, byte for byte, with
//! `fixtures/<scenario>.sse`. The fixtures hold the bytes the plain
//! serialization of whole chunk structs produced, so any change to the wire
//! format of a streamed chunk, intended or not, fails here. Set
//! `SMG_RECORD_SSE_FIXTURES=1` to rewrite the fixtures from the current code.

use std::{fs, path::PathBuf};

use openai_protocol::{
    chat::ChatCompletionRequest, common::StringOrArray, completion::CompletionRequest,
};
use smg_grpc_client::vllm_engine::proto;
use tokio::sync::oneshot;

use super::{
    super::*,
    eof::{
        chunk, chunk_with_logprobs, complete, complete_with_prompt, processor, scripted_stream,
        staged_stream, CharacterTokenizer,
    },
};
use crate::routers::common::sse::SseReceiver;

struct Scenario {
    name: &'static str,
    with_tools: bool,
    request: Value,
    weight_version: Option<&'static str>,
    stop: Option<&'static str>,
    frames: Vec<proto::GenerateResponse>,
    /// A substring the scenario's bytes must carry, so the scenario keeps
    /// exercising the path it was written for.
    must_contain: &'static str,
}

fn request(model: &str, extra: Value) -> Value {
    let mut request = json!({"model": model, "messages": [], "stream": true});
    if let (Some(base), Some(extra)) = (request.as_object_mut(), extra.as_object()) {
        base.extend(extra.iter().map(|(k, v)| (k.clone(), v.clone())));
    }
    request
}

fn tools() -> Value {
    json!([{
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}
    }])
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "content",
            with_tools: false,
            request: request("fixture-model", json!({})),
            weight_version: None,
            stop: None,
            frames: vec![chunk(0, "Hello"), chunk(0, ", world"), complete(0, "stop")],
            must_contain: r#""content":", world""#,
        },
        Scenario {
            name: "reasoning",
            with_tools: false,
            request: request("fixture-model", json!({"separate_reasoning": true})),
            weight_version: None,
            stop: None,
            frames: vec![
                chunk(0, "<think>because"),
                chunk(0, " of this</think>the answer"),
                complete(0, "stop"),
            ],
            must_contain: r#""reasoning_content":"because""#,
        },
        Scenario {
            name: "two_choices_with_logprobs",
            with_tools: false,
            request: request("fixture-model", json!({"n": 2, "logprobs": true})),
            weight_version: None,
            stop: None,
            // Only one choice completes: finish chunks of several choices are
            // emitted in hash-map order, which would make the bytes vary.
            frames: vec![
                chunk_with_logprobs(0, "left"),
                chunk_with_logprobs(1, "right"),
                chunk_with_logprobs(0, " one"),
                complete(0, "length"),
            ],
            must_contain: r#""logprob":-0.5"#,
        },
        Scenario {
            name: "usage_chunk",
            with_tools: false,
            request: request(
                "fixture-model",
                json!({"stream_options": {"include_usage": true}}),
            ),
            weight_version: None,
            stop: None,
            frames: vec![chunk(0, "ok"), complete_with_prompt(0, "stop", 10)],
            must_contain: r#""prompt_tokens":10"#,
        },
        Scenario {
            name: "continuous_usage",
            with_tools: false,
            request: request(
                "fixture-model",
                json!({"stream_options": {"include_usage": true, "continuous_usage_stats": true}}),
            ),
            weight_version: None,
            stop: None,
            frames: vec![
                chunk(0, "a"),
                chunk(0, "b"),
                complete_with_prompt(0, "stop", 10),
            ],
            must_contain: r#""completion_tokens":1,"#,
        },
        Scenario {
            name: "deepseek_usage_placeholder",
            with_tools: false,
            request: request(
                "deepseek-flash",
                json!({"stream_options": {"include_usage": true}}),
            ),
            weight_version: None,
            stop: None,
            frames: vec![chunk(0, "hi"), complete_with_prompt(0, "stop", 10)],
            must_contain: r#""usage":null"#,
        },
        Scenario {
            name: "system_fingerprint",
            with_tools: false,
            request: request("fixture-model", json!({})),
            weight_version: Some("v7"),
            stop: None,
            frames: vec![chunk(0, "x"), complete(0, "stop")],
            must_contain: r#""system_fingerprint":"v7""#,
        },
        Scenario {
            name: "stop_sequence",
            with_tools: false,
            request: request("fixture-model", json!({})),
            weight_version: None,
            stop: Some("END"),
            frames: vec![chunk(0, "keep this ENDnot this"), complete(0, "length")],
            must_contain: r#""matched_stop":"END""#,
        },
        Scenario {
            name: "tool_calls",
            with_tools: true,
            request: request("kimi-fixture", json!({"tools": tools()})),
            weight_version: None,
            stop: None,
            frames: vec![
                chunk(0, r#"{"name": "lookup", "#),
                chunk(0, r#""parameters": {"q": "x"}}"#),
                complete(0, "stop"),
            ],
            must_contain: r#""tool_calls""#,
        },
        Scenario {
            name: "specific_function",
            with_tools: true,
            request: request(
                "kimi-fixture",
                json!({
                    "tools": tools(),
                    "tool_choice": {"type": "function", "function": {"name": "lookup"}}
                }),
            ),
            weight_version: None,
            stop: None,
            frames: vec![
                chunk(0, r#"{"q": "#),
                chunk(0, r#""x"}"#),
                complete(0, "stop"),
            ],
            must_contain: r#""name":"lookup""#,
        },
    ]
}

async fn stream_bytes(scenario: &Scenario) -> Vec<u8> {
    let (stream, server) = scripted_stream(scenario.frames.clone(), "0").await;
    let (tx, mut rx) = sse_channel();
    let request: ChatCompletionRequest =
        serde_json::from_value(scenario.request.clone()).expect("chat request");
    let dispatch = context::DispatchMetadata {
        request_id: "chatcmpl-fixture".to_string(),
        model: request.model.clone(),
        created: 1_700_000_000,
        weight_version: scenario.weight_version.map(str::to_string),
    };
    let stop = scenario
        .stop
        .map(|sequence| StringOrArray::String(sequence.to_string()));
    let result = processor(scenario.with_tools)
        .process_streaming_chunks(
            stream,
            dispatch,
            Arc::new(CharacterTokenizer::default()),
            (stop, None, false, false, false),
            ChatResponseSpec::from(&request),
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

/// Compare (or, under `SMG_RECORD_SSE_FIXTURES`, record) the bytes of one
/// scenario against `fixtures/<name>.sse`.
pub(super) fn assert_recorded_bytes(name: &str, must_contain: &str, actual: &[u8]) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/routers/grpc/regular/streaming/tests/fixtures");
    let record = std::env::var_os("SMG_RECORD_SSE_FIXTURES").is_some();
    let text = String::from_utf8_lossy(actual);
    assert!(
        text.contains(must_contain),
        "{name}: {must_contain:?} missing from the stream:\n{text}"
    );
    let path = dir.join(format!("{name}.sse"));
    if record {
        fs::write(&path, actual).expect("write the fixture");
        return;
    }
    let expected = fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert!(
        actual == expected,
        "{name}: the SSE bytes differ from {}\n--- recorded\n{}\n--- now\n{text}",
        path.display(),
        String::from_utf8_lossy(&expected)
    );
}

#[tokio::test]
async fn chat_stream_bytes_match_the_recorded_fixtures() {
    for scenario in scenarios() {
        let actual = stream_bytes(&scenario).await;
        assert_recorded_bytes(scenario.name, scenario.must_contain, &actual);
    }
}

struct CompletionScenario {
    name: &'static str,
    request: Value,
    weight_version: Option<&'static str>,
    stop: Option<&'static str>,
    /// The prompt's text, echoed into the first chunk when the request asks.
    prompt_text: &'static str,
    frames: Vec<proto::GenerateResponse>,
    must_contain: &'static str,
}

fn completion_scenarios() -> Vec<CompletionScenario> {
    let request = |extra: Value| {
        let mut request = json!({"model": "fixture-model", "prompt": "test", "stream": true});
        if let (Some(base), Some(extra)) = (request.as_object_mut(), extra.as_object()) {
            base.extend(extra.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        request
    };
    vec![
        CompletionScenario {
            name: "completion_text",
            request: request(json!({})),
            weight_version: None,
            stop: None,
            prompt_text: "",
            frames: vec![chunk(0, "Hello"), chunk(0, ", world"), complete(0, "stop")],
            must_contain: r#""text":", world""#,
        },
        CompletionScenario {
            name: "completion_fingerprint_and_stop_sequence",
            request: request(json!({})),
            weight_version: Some("v7"),
            stop: Some("END"),
            prompt_text: "",
            frames: vec![chunk(0, "keep this ENDnot this"), complete(0, "length")],
            must_contain: r#""system_fingerprint":"v7""#,
        },
        CompletionScenario {
            name: "completion_echo_and_suffix",
            request: request(json!({"prompt": "prompt: ", "echo": true, "suffix": "tail"})),
            weight_version: None,
            stop: None,
            prompt_text: "prompt: ",
            frames: vec![chunk(0, "a"), complete(0, "stop")],
            must_contain: r#""text":"tail""#,
        },
    ]
}

async fn completion_stream_bytes(scenario: &CompletionScenario) -> Vec<u8> {
    let (stream, server) = scripted_stream(scenario.frames.clone(), "0").await;
    let (tx, mut rx) = sse_channel();
    let request: CompletionRequest =
        serde_json::from_value(scenario.request.clone()).expect("completion request");
    let spec = CompletionResponseSpec::new(&request, Vec::new());
    let dispatch = context::DispatchMetadata {
        request_id: "cmpl-fixture".to_string(),
        model: request.model.clone(),
        created: 1_700_000_000,
        weight_version: scenario.weight_version.map(str::to_string),
    };
    let stop = scenario
        .stop
        .map(|sequence| StringOrArray::String(sequence.to_string()));
    let result = processor(false)
        .process_completion_streaming_chunks(
            stream,
            dispatch,
            Arc::new(CharacterTokenizer::default()),
            (stop, None, false, false, false),
            &spec,
            scenario.prompt_text,
            0,
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
    assert!(
        result.is_ok(),
        "{}: the completion stream failed",
        scenario.name
    );
    bytes
}

#[tokio::test]
async fn completion_stream_bytes_match_the_recorded_fixtures() {
    for scenario in completion_scenarios() {
        let actual = completion_stream_bytes(&scenario).await;
        assert_recorded_bytes(scenario.name, scenario.must_contain, &actual);
    }
}

/// Read `rx` to its end: the `before` frames of the messages ready before
/// the gate first (the reader parks at the gate only after they all left),
/// then `gate` opens and the rest are read as they come.
async fn frames_through_the_gate(
    mut rx: SseReceiver,
    before: usize,
    gate: oneshot::Sender<()>,
) -> Vec<Bytes> {
    let mut frames = Vec::new();
    for _ in 0..before {
        frames.push(rx.recv().await.expect("a frame").expect("a write"));
    }
    gate.send(()).expect("the reader waits at the gate");
    while let Some(frame) = rx.recv().await {
        frames.push(frame.expect("a write"));
    }
    frames
}

/// The frame carrying the first token leaves on its own as soon as the
/// message is processed, whatever the engine has ready behind it (the role
/// chunk shares it: both are written before the engine is polled again).
/// After it, the messages the engine has ready at one poll leave as one
/// frame, and a message the reader had to wait for leaves as soon as it is
/// processed, with what came with it: the second message, ready with the
/// first, leaves alone once the first frame is out; the two after the gate
/// share a frame; the finish chunk, written after the read, is its own.
#[tokio::test]
async fn ready_engine_messages_share_a_frame_after_the_first_token() {
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
    let (tx, rx) = sse_channel();
    let request: ChatCompletionRequest =
        serde_json::from_value(request("fixture-model", json!({}))).expect("chat request");
    let dispatch = context::DispatchMetadata {
        request_id: "chatcmpl-fixture".to_string(),
        model: request.model.clone(),
        created: 1_700_000_000,
        weight_version: None,
    };
    let processor = processor(false);
    let reading = async {
        let result = processor
            .process_streaming_chunks(
                stream,
                dispatch,
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, false),
                ChatResponseSpec::from(&request),
                &tx,
                None,
            )
            .await;
        drop(tx);
        result
    };
    let (result, frames) = tokio::join!(reading, frames_through_the_gate(rx, 2, gate));
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    let events: Vec<(usize, String)> = frames
        .iter()
        .map(|frame| {
            let text = String::from_utf8_lossy(frame).to_string();
            (text.matches("data: ").count(), text)
        })
        .collect();
    let shape: Vec<usize> = events.iter().map(|(count, _)| *count).collect();
    assert_eq!(
        shape,
        [2, 1, 2, 1],
        "the role with the first token, the second token alone, the two after the gate together, the finish chunk: {events:?}"
    );
    assert!(events[0].1.contains(r#""content":"a""#), "{events:?}");
    assert!(events[1].1.contains(r#""content":"b""#), "{events:?}");
    assert!(
        events[2].1.contains(r#""content":"c""#) && events[2].1.contains(r#""content":"d""#),
        "{events:?}"
    );
    assert!(
        events[3].1.contains(r#""finish_reason":"stop""#),
        "{events:?}"
    );
}

/// As the chat stream packs its frames: the chunk carrying the first token
/// leaves on its own as soon as its message is processed; after it, the
/// chunks of the messages the engine has ready at one poll leave as one
/// frame: the second token, ready with the first, leaves alone once the
/// first frame is out (the loop does not wait for more); the two after the
/// gate share a frame with the finish the Complete behind them writes.
#[tokio::test]
async fn ready_engine_chunks_share_a_frame_after_the_first_token_on_the_completion_stream() {
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
    let (tx, rx) = sse_channel();
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": "fixture-model", "prompt": "frames", "stream": true
    }))
    .expect("completion request");
    let spec = CompletionResponseSpec::new(&request, Vec::new());
    let dispatch = context::DispatchMetadata {
        request_id: "cmpl-frames".to_string(),
        model: "fixture-model".to_string(),
        created: 1,
        weight_version: None,
    };
    let processor = processor(false);
    let reading = async {
        let result = processor
            .process_completion_streaming_chunks(
                stream,
                dispatch,
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, false),
                &spec,
                "",
                0,
                &tx,
                None,
            )
            .await;
        drop(tx);
        result
    };
    let (result, frames) = tokio::join!(reading, frames_through_the_gate(rx, 2, gate));
    server.abort();
    assert!(result.is_ok(), "the completion stream failed");
    let chunks: Vec<(usize, String)> = frames
        .iter()
        .map(|frame| {
            let text = String::from_utf8_lossy(frame).to_string();
            (text.matches("data: ").count(), text)
        })
        .collect();
    let shape: Vec<usize> = chunks.iter().map(|(count, _)| *count).collect();
    assert_eq!(
        shape.len(),
        3,
        "the first token alone, the second alone, everything after the gate together: {chunks:?}"
    );
    assert_eq!(&shape[..2], [1, 1], "{chunks:?}");
    assert!(chunks[0].1.contains(r#""text":"a""#), "{chunks:?}");
    assert!(chunks[1].1.contains(r#""text":"b""#), "{chunks:?}");
    assert!(
        chunks[2].1.contains(r#""text":"c""#)
            && chunks[2].1.contains(r#""text":"d""#)
            && chunks[2].1.contains(r#""finish_reason":"stop""#)
            && chunks[2].1.ends_with("\n\n"),
        "{chunks:?}"
    );
}

/// The content deltas of the events in `frame`, in order; the opening
/// chunk's empty content is no delta.
fn contents_of(frame: &[u8]) -> Vec<String> {
    std::str::from_utf8(frame)
        .expect("UTF-8 SSE")
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter_map(|json| {
            json["choices"][0]["delta"]["content"]
                .as_str()
                .filter(|content| !content.is_empty())
                .map(str::to_owned)
        })
        .collect()
}

/// The frames until `deltas` content deltas have arrived (the engine quiet
/// behind them), then the gate opened and the frames after it: the two
/// lists.
async fn frames_until_the_deltas_then_the_rest(
    mut rx: SseReceiver,
    deltas: usize,
    gate: oneshot::Sender<()>,
) -> (Vec<Bytes>, Vec<Bytes>) {
    let mut before = Vec::new();
    let mut seen = 0;
    while seen < deltas {
        let frame = rx.recv().await.expect("a frame").expect("a write");
        seen += contents_of(&frame).len();
        before.push(frame);
    }
    gate.send(()).expect("the reader waits at the gate");
    let mut after = Vec::new();
    while let Some(frame) = rx.recv().await {
        after.push(frame.expect("a write"));
    }
    (before, after)
}

/// The room one event may add past the bound: a chat chunk event of a
/// delta of up to three characters is under 200 bytes.
const ONE_EVENT_ROOM: usize = 512;

/// A burst of ready engine messages behind the first token leaves in frames
/// bounded by `CHUNK_FLUSH_BYTES`: the frame being written goes out once it
/// holds the bound, so a burst past it leaves as more than one frame, each
/// at most the bound plus the event that crossed it, the deltas in order,
/// before the engine's next message is read; the finish behind the gate is
/// its own frame. `generate_frames` pins the same bound on the generate
/// stream.
#[tokio::test]
async fn a_burst_past_the_flush_bound_leaves_in_bounded_frames_on_the_chat_stream() {
    const BURST: usize = 200;
    let (gate, gated) = oneshot::channel();
    let burst: Vec<_> = (0..BURST).map(|i| chunk(0, &i.to_string())).collect();
    let (stream, server) = staged_stream(
        vec![(None, burst), (Some(gated), vec![complete(0, "stop")])],
        "0",
    )
    .await;
    let (tx, rx) = sse_channel();
    let request: ChatCompletionRequest =
        serde_json::from_value(request("fixture-model", json!({}))).expect("chat request");
    let dispatch = context::DispatchMetadata {
        request_id: "chatcmpl-fixture".to_string(),
        model: request.model.clone(),
        created: 1_700_000_000,
        weight_version: None,
    };
    let processor = processor(false);
    let reading = async {
        let result = processor
            .process_streaming_chunks(
                stream,
                dispatch,
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, false),
                ChatResponseSpec::from(&request),
                &tx,
                None,
            )
            .await;
        drop(tx);
        result
    };
    let (result, (burst_frames, rest)) = tokio::join!(
        reading,
        frames_until_the_deltas_then_the_rest(rx, BURST, gate)
    );
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    let deltas: Vec<String> = burst_frames
        .iter()
        .flat_map(|frame| contents_of(frame))
        .collect();
    let expected: Vec<String> = (0..BURST).map(|i| i.to_string()).collect();
    assert_eq!(deltas, expected, "the burst's deltas in order");
    let sizes: Vec<usize> = burst_frames.iter().map(Bytes::len).collect();
    let largest = sizes.iter().copied().max().expect("a frame");
    assert!(
        largest >= CHUNK_FLUSH_BYTES,
        "a full frame left at the bound: {sizes:?}"
    );
    assert!(
        largest < CHUNK_FLUSH_BYTES + ONE_EVENT_ROOM,
        "no frame past the bound by more than the event that crossed it: {sizes:?}"
    );
    assert!(
        sizes.len() > 2,
        "the burst behind the first token's frame leaves in more than one frame: {sizes:?}"
    );
    let finish: Vec<String> = rest
        .iter()
        .map(|frame| String::from_utf8_lossy(frame).to_string())
        .collect();
    assert_eq!(
        finish.len(),
        1,
        "the finish behind the gate is its own frame: {finish:?}"
    );
    assert!(
        finish[0].contains(r#""finish_reason":"stop""#),
        "{finish:?}"
    );
}
