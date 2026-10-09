//! The exact SSE bytes of chat streams, pinned per scenario.
//!
//! Each scenario drives `process_streaming_chunks` over scripted engine frames
//! and compares the concatenated SSE events, byte for byte, with
//! `fixtures/<scenario>.sse`. The fixtures hold the bytes the plain
//! serialization of whole chunk structs produced, so any change to the wire
//! format of a streamed chunk, intended or not, fails here. Set
//! `SMG_RECORD_SSE_FIXTURES=1` to rewrite the fixtures from the current code.

use std::{fs, path::PathBuf};

use openai_protocol::{chat::ChatCompletionRequest, common::StringOrArray};
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

#[tokio::test]
async fn chat_stream_bytes_match_the_recorded_fixtures() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/routers/grpc/regular/streaming/fixtures");
    let record = std::env::var_os("SMG_RECORD_SSE_FIXTURES").is_some();
    for scenario in scenarios() {
        let actual = stream_bytes(&scenario).await;
        let text = String::from_utf8_lossy(&actual);
        assert!(
            text.contains(scenario.must_contain),
            "{}: {:?} missing from the stream:\n{text}",
            scenario.name,
            scenario.must_contain
        );
        let path = dir.join(format!("{}.sse", scenario.name));
        if record {
            fs::write(&path, &actual).expect("write the fixture");
            continue;
        }
        let expected = fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(
            actual == expected,
            "{}: the SSE bytes differ from {}\n--- recorded\n{}\n--- now\n{text}",
            scenario.name,
            path.display(),
            String::from_utf8_lossy(&expected)
        );
    }
}
