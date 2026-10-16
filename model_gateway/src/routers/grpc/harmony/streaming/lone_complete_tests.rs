//! A Harmony chat choice whose only engine output is its `Complete` (a
//! completion with no tokens) opens with its role chunk and ends with its
//! finish chunk, as a choice with output does: the chat path makes a choice's
//! parser on its first `Chunk`, and a `Complete` with no parser used to leave
//! the choice without any chunk at all.

use std::sync::Arc;

use openai_protocol::chat::ChatCompletionRequest;
use serde_json::{json, Value};
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};

use super::{usage_tests::harmony_tokens, HarmonyStreamingProcessor};
use crate::{
    observability::metrics::metrics_labels,
    routers::grpc::{context, regular::streaming::eof_tests::scripted_stream, utils},
};

fn chunk(index: u32, token_ids: Vec<u32>) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
            index,
            completion_tokens: token_ids.len() as u32,
            token_ids,
            ..Default::default()
        })),
    }
}

fn complete(index: u32, output_ids: Vec<u32>) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            index,
            finish_reason: "stop".to_string(),
            prompt_tokens: 3,
            completion_tokens: output_ids.len() as u32,
            output_ids,
            ..Default::default()
        })),
    }
}

/// The frames of a choice that answers "Hi" on the final channel.
fn answering_choice(index: u32) -> Vec<proto::GenerateResponse> {
    let tokens = harmony_tokens(None, "Hi");
    vec![chunk(index, tokens.clone()), complete(index, tokens)]
}

/// The `data:` events of the streamed chat completion for `frames`, `[DONE]`
/// left out.
async fn streamed_events(frames: Vec<proto::GenerateResponse>, choices: u32) -> Vec<Value> {
    let (stream, server) = scripted_stream(frames, "0").await;
    let dispatch = context::DispatchMetadata {
        request_id: "harmony-lone-complete".to_string(),
        model: "harmony-test".to_string(),
        created: 1,
        weight_version: None,
    };
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "harmony-test", "messages": [], "n": choices, "stream": true,
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
            source,
        )
        .await;
    let body = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("the stream's body");
    server.abort();
    let events: Vec<Value> = std::str::from_utf8(&body)
        .expect("UTF-8 SSE")
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE JSON"))
        .collect();
    assert!(
        events.iter().all(|event| event.get("error").is_none()),
        "the stream completes without an error event: {events:?}"
    );
    events
}

/// Per choice, the role and finish reason of each of its deltas, in stream
/// order.
fn choice_trail(events: &[Value], index: u32) -> Vec<(Option<&str>, Option<&str>)> {
    events
        .iter()
        .filter_map(|event| {
            let choice = event["choices"].get(0)?;
            (choice["index"] == index).then(|| {
                (
                    choice["delta"]["role"].as_str(),
                    choice["finish_reason"].as_str(),
                )
            })
        })
        .collect()
}

#[tokio::test]
async fn harmony_lone_complete_opens_the_choice_before_its_finish() {
    let events = streamed_events(vec![complete(0, Vec::new())], 1).await;
    assert_eq!(
        choice_trail(&events, 0),
        vec![(Some("assistant"), None), (None, Some("stop"))],
        "{events:?}"
    );
}

#[tokio::test]
async fn harmony_lone_complete_opens_its_choice_among_others() {
    let mut frames = answering_choice(0);
    frames.push(complete(1, Vec::new()));
    let events = streamed_events(frames, 2).await;
    assert_eq!(
        choice_trail(&events, 1),
        vec![(Some("assistant"), None), (None, Some("stop"))],
        "{events:?}"
    );
    let trail = choice_trail(&events, 0);
    assert_eq!(
        trail.first(),
        Some(&(Some("assistant"), None)),
        "{events:?}"
    );
    assert_eq!(trail.last(), Some(&(None, Some("stop"))), "{events:?}");
    assert_eq!(trail.iter().filter(|(role, _)| role.is_some()).count(), 1);
    let content: String = events
        .iter()
        .filter_map(|event| {
            let choice = event["choices"].get(0)?;
            (choice["index"] == 0)
                .then(|| choice["delta"]["content"].as_str())
                .flatten()
        })
        .collect();
    assert_eq!(content, "Hi");
}
