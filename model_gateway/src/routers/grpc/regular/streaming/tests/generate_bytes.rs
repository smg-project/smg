//! The exact SSE bytes of `/generate` streams, pinned per scenario: the plain
//! loop and the loop that carries logprobs, driven over scripted engine frames
//! and compared byte for byte with `fixtures/generate_<scenario>.sse`, the
//! finish event's `e2e_latency` (the stream's clock) written as `0` on both
//! sides. The fixtures hold the bytes the per-chunk JSON trees produced, so
//! any change to the wire format of a generate event, intended or not, fails
//! here. Set `SMG_RECORD_SSE_FIXTURES=1` to rewrite them from the current code.

use smg_grpc_client::vllm_engine::proto;

use super::{
    super::*,
    eof::{
        chunk, chunk_with_logprobs, complete, complete_with_prompt, scripted_stream,
        CharacterTokenizer,
    },
    sse_bytes::assert_recorded_bytes,
};

struct GenerateScenario {
    name: &'static str,
    /// Drive the loop that carries logprobs (the prefill-decode and logprob
    /// streams' loop) instead of the plain one.
    logprobs_loop: bool,
    input_token_logprobs: Option<Vec<Vec<Option<f64>>>>,
    weight_version: &'static str,
    expected_choices: u32,
    frames: Vec<proto::GenerateResponse>,
    /// A substring the scenario's bytes must carry, so the scenario keeps
    /// exercising the path it was written for.
    must_contain: &'static str,
}

fn scenarios() -> Vec<GenerateScenario> {
    vec![
        GenerateScenario {
            name: "generate_text",
            logprobs_loop: false,
            input_token_logprobs: None,
            weight_version: "default",
            expected_choices: 1,
            frames: vec![
                chunk(0, "Hello"),
                chunk(0, ", world"),
                complete_with_prompt(0, "stop", 10),
            ],
            must_contain: r#""text":"Hello, world""#,
        },
        GenerateScenario {
            name: "generate_two_choices_length",
            logprobs_loop: false,
            input_token_logprobs: None,
            weight_version: "v7",
            expected_choices: 2,
            frames: vec![
                chunk(0, "left"),
                chunk(1, "right"),
                chunk(0, " one"),
                complete(0, "length"),
                complete(1, "length"),
            ],
            must_contain: r#""weight_version":"v7""#,
        },
        GenerateScenario {
            name: "generate_logprobs",
            logprobs_loop: true,
            input_token_logprobs: Some(vec![vec![None], vec![Some(-0.25)]]),
            weight_version: "default",
            expected_choices: 1,
            frames: vec![
                chunk_with_logprobs(0, "ab"),
                chunk_with_logprobs(0, "c"),
                complete(0, "stop"),
            ],
            must_contain: r#""output_token_logprobs":[["#,
        },
        GenerateScenario {
            name: "generate_logprobs_loop_without_logprobs",
            logprobs_loop: true,
            input_token_logprobs: None,
            weight_version: "default",
            expected_choices: 1,
            frames: vec![chunk(0, "hi"), complete(0, "stop")],
            must_contain: r#""input_token_logprobs":null"#,
        },
    ]
}

/// The SSE bytes of one scenario, its `e2e_latency` values written as `0`.
async fn stream_bytes(scenario: &GenerateScenario) -> Vec<u8> {
    let (stream, server) = scripted_stream(scenario.frames.clone(), "0").await;
    let ctx = GenerateStreamContext {
        request_id: "gen-bytes".to_string(),
        weight_version: scenario.weight_version.to_string(),
        return_logprob: scenario.logprobs_loop,
        backend_type: "test",
        model: "fixture-model".to_string(),
        expected_choices: scenario.expected_choices,
    };
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let (tx, mut rx) = sse_channel();
    let result = if scenario.logprobs_loop {
        StreamingProcessor::process_generate_streaming_with_input_logprobs(
            tokenizer,
            stream,
            ctx,
            scenario.input_token_logprobs.clone(),
            &tx,
            None,
            None,
        )
        .await
    } else {
        StreamingProcessor::process_generate_streaming(tokenizer, stream, ctx, &tx, None).await
    };
    drop(tx);
    let mut bytes = Vec::new();
    while let Some(frame) = rx.recv().await {
        bytes.extend_from_slice(&frame.expect("successful SSE write"));
    }
    server.abort();
    assert!(result.is_ok(), "{}: {result:?}", scenario.name);
    without_latency(&bytes)
}

/// The bytes with every `e2e_latency` value written as `0`: the field is the
/// stream's clock, the one part of a finish event that differs run to run.
fn without_latency(bytes: &[u8]) -> Vec<u8> {
    const KEY: &[u8] = b"\"e2e_latency\":";
    let mut out = Vec::with_capacity(bytes.len());
    let mut rest = bytes;
    while let Some(at) = rest.windows(KEY.len()).position(|window| window == KEY) {
        let value = at + KEY.len();
        out.extend_from_slice(&rest[..value]);
        out.push(b'0');
        let end = rest[value..]
            .iter()
            .position(|b| !matches!(b, b'0'..=b'9' | b'.' | b'-' | b'+' | b'e' | b'E'))
            .map_or(rest.len(), |n| value + n);
        rest = &rest[end..];
    }
    out.extend_from_slice(rest);
    out
}

#[test]
fn the_latency_value_is_written_as_zero_wherever_it_appears() {
    let bytes = br#"{"meta_info":{"e2e_latency":0.0123,"x":1}}{"e2e_latency":1e-5}"#;
    assert_eq!(
        without_latency(bytes),
        br#"{"meta_info":{"e2e_latency":0,"x":1}}{"e2e_latency":0}"#
    );
}

#[tokio::test]
async fn generate_stream_bytes_match_the_recorded_fixtures() {
    for scenario in scenarios() {
        let actual = stream_bytes(&scenario).await;
        assert_recorded_bytes(scenario.name, scenario.must_contain, &actual);
    }
}

/// Text that needs escaping reaches the client as serde writes it, on both
/// loops: every event's `text` is the choice's text so far, written as the
/// JSON string literal serde makes of it, and it reads back as that text.
#[tokio::test]
async fn generate_events_write_text_that_needs_escaping_as_serde_does() {
    let chunks = [
        "say \"hi\"",
        "\n\ttab\\",
        " caf\u{e9} \u{1f600}",
        " \u{0}\u{1f}",
    ];
    let mut texts_so_far = Vec::new();
    let mut text = String::new();
    for chunk in chunks {
        text.push_str(chunk);
        texts_so_far.push(text.clone());
    }
    // The finish event carries the whole text once more.
    texts_so_far.push(text);
    for logprobs_loop in [false, true] {
        let mut frames: Vec<_> = chunks.iter().map(|text| chunk(0, text)).collect();
        frames.push(complete(0, "stop"));
        let scenario = GenerateScenario {
            name: "generate_escapes",
            logprobs_loop,
            input_token_logprobs: None,
            weight_version: "default",
            expected_choices: 1,
            frames,
            must_contain: "",
        };
        let bytes = stream_bytes(&scenario).await;
        let events: Vec<&str> = std::str::from_utf8(&bytes)
            .expect("UTF-8 SSE")
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect();
        assert_eq!(
            events.len(),
            texts_so_far.len(),
            "logprobs loop {logprobs_loop}: one event per chunk and the finish"
        );
        for (event, text) in events.iter().zip(&texts_so_far) {
            let literal = serde_json::to_string(text).expect("a string serializes");
            assert!(
                event.starts_with(&format!("{{\"text\":{literal},\"output_ids\":")),
                "logprobs loop {logprobs_loop}: {event}"
            );
            let parsed: Value = serde_json::from_str(event).expect("the event parses");
            assert_eq!(parsed["text"].as_str(), Some(text.as_str()));
        }
    }
}
