//! The shape of a streamed chat completion per provider profile: the delta
//! re-slicing of `routers::common::sse_rechunk` is MiniMax's stream contract
//! and applies to MiniMax models only. Under every other model id, and under
//! a served model aliased under another vendor's name, every engine chunk
//! leaves as its own SSE event before the engine's next chunk exists. A
//! guard: these tests pass on the code they were written against.

use std::time::Duration;

use axum::{body::Body, http::StatusCode};
use http_body_util::BodyExt;
use openai_protocol::chat::ChatCompletionRequest;
use tokio::{sync::oneshot, time::timeout};

use super::{
    eof_tests::{chunk, complete, gated_stream, processor, CharacterTokenizer},
    *,
};

/// Chunks per stream: two-character deltas, so a re-slicer holds all of them
/// (their total stays under its slice threshold) until an idle flush or the
/// end of the stream.
const CHUNKS: usize = 12;
/// The engine parks after this many chunks until the client has read their
/// text: whatever the client reads meanwhile left before the next chunk
/// existed, with no clock in the assertion.
const PARK_AFTER: usize = CHUNKS / 2;
/// The client's patience per half; a path that held the deltas back fails
/// here instead of hanging the suite.
const PATIENCE: Duration = Duration::from_secs(10);

/// The text of chunk `i`.
fn text(i: usize) -> String {
    format!("{i:02}")
}

/// The spec of a streamed chat request for `model`; the model id selects the
/// provider profile.
fn spec_for(model: &str) -> ChatResponseSpec {
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": model, "messages": [], "stream": true
    }))
    .expect("chat request");
    ChatResponseSpec::from(&request)
}

/// The next non-empty content delta of the SSE body, `None` at its end.
async fn next_content_delta(body: &mut Body, pending: &mut Vec<u8>) -> Option<String> {
    loop {
        while let Some(end) = pending.windows(2).position(|w| w == b"\n\n") {
            let event: Vec<u8> = pending.drain(..end + 2).collect();
            let event = String::from_utf8_lossy(&event);
            let Some(payload) = event.strip_prefix("data: ") else {
                continue;
            };
            let Ok(json) = serde_json::from_str::<Value>(payload.trim()) else {
                continue;
            };
            if let Some(content) = json["choices"][0]["delta"]["content"].as_str() {
                if !content.is_empty() {
                    return Some(content.to_string());
                }
            }
        }
        let frame = body.frame().await?.expect("an SSE frame");
        if let Some(data) = frame.data_ref() {
            pending.extend_from_slice(data);
        }
    }
}

/// The content deltas of one streamed chat completion for `model` through
/// the chat path, in two halves: those read while the engine was parked
/// after its `PARK_AFTER`th chunk, and those read after it resumed.
async fn content_deltas_around_the_park(model: &str) -> (Vec<String>, Vec<String>) {
    let first: Vec<_> = (0..PARK_AFTER).map(|i| chunk(0, &text(i))).collect();
    let mut rest: Vec<_> = (PARK_AFTER..CHUNKS).map(|i| chunk(0, &text(i))).collect();
    rest.push(complete(0, "stop"));
    let (resume, gate) = oneshot::channel();
    let (stream, server) = gated_stream(first, gate, rest, "0").await;
    let dispatch = context::DispatchMetadata {
        request_id: "chatcmpl-parked".to_string(),
        model: model.to_string(),
        created: 1_700_000_000,
        weight_version: None,
    };
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_CHAT);
    let response = Arc::new(processor(false))
        .process_streaming_response(
            context::ExecutionResult::Single { stream },
            spec_for(model),
            dispatch,
            Arc::new(CharacterTokenizer::default()),
            false,
            None,
            source,
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK, "{model}");
    let mut body = response.into_body();
    let mut pending = Vec::new();

    let parked_text: String = (0..PARK_AFTER).map(text).collect();
    let mut parked = Vec::new();
    let read = timeout(PATIENCE, async {
        while parked.concat() != parked_text {
            assert!(
                parked_text.starts_with(&parked.concat()),
                "{model}: the parked half read as {parked:?}"
            );
            let delta = next_content_delta(&mut body, &mut pending)
                .await
                .unwrap_or_else(|| panic!("{model}: the stream ended at {parked:?}"));
            parked.push(delta);
        }
    })
    .await;
    assert!(
        read.is_ok(),
        "{model}: the parked half did not reach the client within {PATIENCE:?}: {parked:?}"
    );

    resume.send(()).expect("the engine resumes");
    let mut resumed = Vec::new();
    let read = timeout(PATIENCE, async {
        while let Some(delta) = next_content_delta(&mut body, &mut pending).await {
            resumed.push(delta);
        }
    })
    .await;
    assert!(
        read.is_ok(),
        "{model}: the stream did not end within {PATIENCE:?} of the engine resuming: {resumed:?}"
    );
    server.abort();
    (parked, resumed)
}

/// Every chunk left as its own content event before the next chunk existed,
/// on both sides of the park.
async fn assert_one_delta_per_chunk(model: &str) {
    let (parked, resumed) = content_deltas_around_the_park(model).await;
    let expected = |chunks: std::ops::Range<usize>| chunks.map(text).collect::<Vec<_>>();
    assert_eq!(
        parked,
        expected(0..PARK_AFTER),
        "{model}: one event per chunk while the engine was parked"
    );
    assert_eq!(
        resumed,
        expected(PARK_AFTER..CHUNKS),
        "{model}: one event per chunk after the engine resumed"
    );
}

#[tokio::test]
async fn a_deepseek_stream_keeps_one_delta_per_engine_chunk() {
    assert_one_delta_per_chunk("deepseek-v4.1").await;
}

#[tokio::test]
async fn a_kimi_stream_keeps_one_delta_per_engine_chunk() {
    assert_one_delta_per_chunk("kimi-k3").await;
}

#[tokio::test]
async fn a_glm_stream_keeps_one_delta_per_engine_chunk() {
    assert_one_delta_per_chunk("glm-5").await;
}

#[tokio::test]
async fn a_plain_served_model_stream_keeps_one_delta_per_engine_chunk() {
    assert_one_delta_per_chunk("served-plain-model").await;
}

/// A served model aliased under a Kimi name takes Kimi's profile for either
/// of its names (the alias table is process-wide: names unique to this test),
/// and Kimi's profile re-slices nothing.
#[tokio::test]
async fn a_served_model_aliased_under_a_kimi_name_keeps_one_delta_per_engine_chunk() {
    ProviderProfile::register_model_aliases([(
        "kimi-k3-slicing-guard",
        "served-model-slicing-guard",
    )]);
    assert_eq!(
        ProviderProfile::for_model("served-model-slicing-guard"),
        ProviderProfile::Kimi
    );
    assert_one_delta_per_chunk("served-model-slicing-guard").await;
    assert_one_delta_per_chunk("kimi-k3-slicing-guard").await;
}

/// MiniMax's stream contract: the short deltas are held and re-sliced into
/// fewer, larger events (the parked half leaves on the idle flush), the text
/// unchanged.
#[tokio::test]
async fn a_minimax_stream_is_re_sliced_as_the_contract_asks() {
    let (parked, resumed) = content_deltas_around_the_park("MiniMax-M3").await;
    assert!(
        parked.len() < PARK_AFTER,
        "re-sliced into fewer events than engine chunks, got {parked:?}"
    );
    let all: String = parked.iter().chain(&resumed).map(String::as_str).collect();
    let expected: String = (0..CHUNKS).map(text).collect();
    assert_eq!(
        all, expected,
        "the re-slicing changes the cuts, not the text"
    );
}
