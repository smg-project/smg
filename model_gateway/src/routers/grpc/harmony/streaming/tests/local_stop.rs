//! A stop the router enforces itself (a stop string of the request's, which
//! the engine does not see over the core link) ends the client's stream and
//! the engine's request alike: the engine stream is not read past the chunk
//! the stop matched in, and it is released un-marked, which is the abort path
//! a client disconnect takes. Proven on scripted frames whose Complete waits
//! behind a gate that never opens: a stream read to the engine's end would
//! wait on it.

use std::{sync::Arc, time::Duration};

use openai_protocol::chat::ChatCompletionRequest;
use serde_json::{json, Value};
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};
use tokio::{sync::oneshot, time::timeout};

use super::{super::HarmonyStreamingProcessor, chunk, usage::harmony_tokens};
use crate::{
    observability::metrics::metrics_labels,
    routers::{
        common::placement::{tests::heard_request, RequestEnd},
        grpc::{context, regular::streaming::tests::eof::gated_stream, utils},
    },
};

fn complete(output_ids: Vec<u32>) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            index: 0,
            finish_reason: "length".to_string(),
            prompt_tokens: 3,
            completion_tokens: output_ids.len() as u32,
            output_ids,
            ..Default::default()
        })),
    }
}

/// The chat turn of these cases, read to its end: the engine streams on past
/// the router's stop, and its Complete waits behind a gate that never opens.
/// The engine's stream tells its end through `end`; an empty handle leaves it
/// as it is. Answers the engine's tokens and the events the client read.
async fn stopped_turn(end: RequestEnd) -> (Vec<u32>, Vec<Value>) {
    let tokens = harmony_tokens(None, "Hello STOP and on and on and on and on");
    let chunks: Vec<_> = tokens.chunks(3).map(|ids| chunk(ids.to_vec())).collect();
    let (gate_tx, gate) = oneshot::channel::<()>();
    let (mut stream, server) =
        gated_stream(chunks, gate, vec![complete(tokens.clone())], "0").await;
    stream.report_through(end);
    let dispatch = context::DispatchMetadata {
        request_id: "harmony-local-stop".to_string(),
        model: "harmony-test".to_string(),
        created: 1,
        weight_version: None,
    };
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "harmony-test", "messages": [], "stream": true,
        "stream_options": {"include_usage": true},
    }))
    .expect("chat request");
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_CHAT);
    let response = Arc::new(HarmonyStreamingProcessor::new())
        .process_streaming_chat_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: std::time::Instant::now(),
            },
            Arc::new(request),
            dispatch,
            context::HarmonyReadOptions {
                stop_strings: vec!["STOP".to_string()],
                ..Default::default()
            },
            None,
            None,
            source,
        )
        .await;
    let body = timeout(
        Duration::from_secs(5),
        axum::body::to_bytes(response.into_body(), 1 << 20),
    )
    .await
    .expect("the stream ends at the stop, without the engine's Complete")
    .expect("the stream's body");
    server.abort();
    drop(gate_tx);
    let events = std::str::from_utf8(&body)
        .expect("UTF-8 SSE")
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE JSON"))
        .collect();
    (tokens, events)
}

#[tokio::test]
async fn a_router_stop_ends_the_read_and_the_engine_request() {
    let (tokens, events) = stopped_turn(RequestEnd::default()).await;
    let content: String = events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
        .collect();
    // The router's scanner emits the text through the match, as it did before.
    assert_eq!(content, "Hello STOP", "{events:?}");
    let finishes: Vec<_> = events
        .iter()
        .filter(|event| !event["choices"][0]["finish_reason"].is_null())
        .collect();
    assert_eq!(finishes.len(), 1, "{events:?}");
    assert_eq!(finishes[0]["choices"][0]["finish_reason"], "stop");
    assert_eq!(finishes[0]["choices"][0]["matched_stop"], "STOP");
    // The usage counts through the chunk the stop matched in, with that
    // chunk's prompt count; the engine's tail is nobody's.
    let usage = events
        .iter()
        .find(|event| event["usage"].is_object())
        .map(|event| event["usage"].clone())
        .expect("the usage chunk");
    assert_eq!(usage["prompt_tokens"], 3, "{usage}");
    let completion = usage["completion_tokens"]
        .as_u64()
        .expect("completion tokens");
    assert!(
        completion > 0 && completion < tokens.len() as u64,
        "{completion} of {} tokens",
        tokens.len()
    );
}

/// The stop leaves a whole answer. The reader lets the engine go there, and
/// Grand Central hears the worker's 200, though the stream is dropped
/// before its own end.
#[tokio::test]
async fn a_router_stop_is_heard_as_the_workers_good_end() {
    let (heard, guard, end) = heard_request();
    let (_tokens, events) = stopped_turn(end).await;
    assert!(events
        .iter()
        .any(|event| event["choices"][0]["finish_reason"] == "stop"));

    drop(guard);
    assert_eq!(
        *heard.lock().expect("the stage heard"),
        ["issued", "dispatched Single", "completed Ok 200"]
    );
}
