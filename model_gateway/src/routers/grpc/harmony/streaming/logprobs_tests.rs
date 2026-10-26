//! The logprobs of a streamed Harmony chat completion: every engine step's
//! entries reach the client once, on the step's delta chunk or, when the
//! step's tokens made no delta (a channel header the parser consumed), on a
//! chunk with an empty delta, as the engine's own server sends them.

use std::{sync::Arc, time::Duration};

use openai_protocol::chat::ChatCompletionRequest;
use serde_json::{json, Value};
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};
use tokio::time::timeout;

use super::{usage_tests::harmony_tokens, HarmonyStreamingProcessor};
use crate::{
    observability::metrics::metrics_labels,
    routers::grpc::{context, regular::streaming::eof_tests::scripted_stream, utils},
};

/// One token per engine step, with its logprob.
fn chunk_with_logprobs(token_id: u32) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
            index: 0,
            prompt_tokens: 3,
            completion_tokens: 1,
            token_ids: vec![token_id],
            output_logprobs: Some(proto::OutputLogProbs {
                token_logprobs: vec![-0.5],
                token_ids: vec![token_id],
                top_logprobs: Vec::new(),
            }),
            ..Default::default()
        })),
    }
}

fn complete(output_ids: Vec<u32>) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            index: 0,
            finish_reason: "stop".to_string(),
            prompt_tokens: 3,
            completion_tokens: output_ids.len() as u32,
            output_ids,
            ..Default::default()
        })),
    }
}

#[tokio::test]
async fn every_steps_logprobs_reach_the_client_once() {
    let tokens = harmony_tokens(Some("Think."), "Hello");
    let mut frames: Vec<_> = tokens.iter().map(|&id| chunk_with_logprobs(id)).collect();
    frames.push(complete(tokens.clone()));
    let (stream, server) = scripted_stream(frames, "0").await;
    let dispatch = context::DispatchMetadata {
        request_id: "harmony-logprobs".to_string(),
        model: "harmony-test".to_string(),
        created: 1,
        weight_version: None,
    };
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "harmony-test", "messages": [], "stream": true, "logprobs": true,
    }))
    .expect("chat request");
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_CHAT);
    let response = Arc::new(HarmonyStreamingProcessor::new())
        .process_streaming_chat_response(
            context::ExecutionResult::Single { stream },
            Arc::new(request),
            dispatch,
            Vec::new(),
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
    .expect("the stream ends")
    .expect("the stream's body");
    server.abort();
    let events: Vec<Value> = std::str::from_utf8(&body)
        .expect("UTF-8 SSE")
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE JSON"))
        .collect();
    let mut entries = 0;
    let mut empty_deltas_with_logprobs = 0;
    for event in &events {
        let choice = &event["choices"][0];
        let Some(items) = choice["logprobs"]["content"].as_array() else {
            continue;
        };
        entries += items.len();
        if choice["delta"]
            .as_object()
            .is_some_and(|delta| delta.is_empty())
        {
            empty_deltas_with_logprobs += 1;
        }
    }
    // One entry per generated token, the channel headers' and the stop token's
    // on empty deltas; the text is what it was.
    assert_eq!(entries, tokens.len(), "{events:?}");
    assert!(empty_deltas_with_logprobs > 0, "{events:?}");
    let text = |field: &str| -> String {
        events
            .iter()
            .filter_map(|event| event["choices"][0]["delta"][field].as_str())
            .collect()
    };
    assert_eq!(text("reasoning_content"), "Think.");
    assert_eq!(text("content"), "Hello");
}
