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

use super::{
    eof_tests::{
        chunk, chunk_with_logprobs, complete, complete_with_prompt, processor, scripted_stream,
        CharacterTokenizer,
    },
    *,
};

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
fn assert_recorded_bytes(name: &str, must_contain: &str, actual: &[u8]) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/routers/grpc/regular/streaming/fixtures");
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
    let spec = CompletionResponseSpec::from(&request);
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

/// Every engine message leaves as its own frame as soon as it is processed,
/// even when the engine delivered them all at once: here the role chunk
/// shares the first content chunk's frame (both are written before the
/// stream is polled again), the two other content chunks and the finish
/// chunk follow one frame each.
#[tokio::test]
async fn each_engine_message_leaves_as_its_own_frame() {
    let scenario = Scenario {
        name: "ready_chunks",
        with_tools: false,
        request: request("fixture-model", json!({})),
        weight_version: None,
        stop: None,
        frames: vec![
            chunk(0, "a"),
            chunk(0, "b"),
            chunk(0, "c"),
            complete(0, "stop"),
        ],
        must_contain: r#""content":"c""#,
    };
    let (stream, server) = scripted_stream(scenario.frames.clone(), "0").await;
    let (tx, mut rx) = sse_channel();
    let request: ChatCompletionRequest =
        serde_json::from_value(scenario.request.clone()).expect("chat request");
    let dispatch = context::DispatchMetadata {
        request_id: "chatcmpl-fixture".to_string(),
        model: request.model.clone(),
        created: 1_700_000_000,
        weight_version: None,
    };
    let result = processor(false)
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
    let mut frames = 0usize;
    let mut events = 0usize;
    while let Some(frame) = rx.recv().await {
        let frame = frame.expect("successful SSE write");
        frames += 1;
        events += String::from_utf8_lossy(&frame).matches("data: ").count();
    }
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(events, 5, "role, three content chunks and the finish chunk");
    assert_eq!(
        frames, 4,
        "{events} events in {frames} frames: the role with the first content chunk, then one frame per message"
    );
}
