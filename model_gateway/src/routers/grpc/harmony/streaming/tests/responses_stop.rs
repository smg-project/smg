//! A Harmony Responses iteration takes the router's string stops as a chat
//! turn does, where the engine cannot match them (a token-only wire or
//! worker): the channel text is cut after the first match, which stays as
//! the suffix as the chat turn keeps it; what the model wrote past it is
//! dropped; the turn's finish is `stop` and its object `completed`. On the
//! stream the engine request ends with the match: the stream is not read
//! past the chunk the stop matched in and is released un-marked, the abort
//! path a client disconnect takes (proven on scripted frames whose Complete
//! waits behind a gate that never opens); an engine that ends the turn on
//! the very step of the match sends its Complete next, which adds only the
//! counts the stop could not know. A stop inside the analysis channel ends
//! the streamed turn before any message. Text a scanner held back as a
//! possible stop is released when the turn ends without one, and the whole
//! chat turn keeps its cut through the helper both arms share.

use std::{sync::Arc, time::Duration};

use openai_protocol::{chat::ChatCompletionRequest, responses::ResponsesRequest};
use serde_json::{json, Value};
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};
use tokio::{sync::oneshot, time::timeout};

use super::{super::HarmonyStreamingProcessor, chunk, usage::harmony_tokens};
use crate::routers::{
    common::sse::sse_channel,
    grpc::{
        common::{
            response_formatting::TokenCounts, responses::streaming::ResponseStreamEventEmitter,
        },
        context,
        harmony::processor::{HarmonyResponseProcessor, ResponsesIterationResult},
        regular::streaming::tests::eof::{gated_stream, scripted_stream},
    },
};

const STOP: &str = "STOP";

/// The one stop the router enforces in these turns.
fn stops() -> Vec<String> {
    vec![STOP.to_string()]
}

fn complete(output_ids: Vec<u32>, finish_reason: &str) -> proto::GenerateResponse {
    complete_with_spec(output_ids, finish_reason, 0, 0)
}

/// The terminal frame with the speculative counters only it knows.
fn complete_with_spec(
    output_ids: Vec<u32>,
    finish_reason: &str,
    spec_accepted: u32,
    spec_drafted: u32,
) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            index: 0,
            finish_reason: finish_reason.to_string(),
            prompt_tokens: 3,
            completion_tokens: output_ids.len() as u32,
            spec_accepted_tokens: spec_accepted,
            spec_draft_tokens: spec_drafted,
            output_ids,
            ..Default::default()
        })),
    }
}

/// The frames of a turn: `ids` in chunks of three, then its terminal frame
/// with `finish_reason`.
fn frames(ids: &[u32], finish_reason: &str) -> Vec<proto::GenerateResponse> {
    let mut frames: Vec<proto::GenerateResponse> =
        ids.chunks(3).map(|ids| chunk(ids.to_vec())).collect();
    frames.push(complete(ids.to_vec(), finish_reason));
    frames
}

/// The request of these turns: its `stop` is the one the router enforces.
fn request() -> ResponsesRequest {
    serde_json::from_value(json!({"model": "harmony-test", "input": "hi", "stop": [STOP]}))
        .expect("responses request")
}

fn dispatch() -> context::DispatchMetadata {
    context::DispatchMetadata {
        request_id: "harmony-responses-stop".to_string(),
        model: "harmony-test".to_string(),
        created: 1,
        weight_version: None,
    }
}

/// The non-streamed iteration over `frames` under `router_stops`: the
/// response object of the completed turn.
async fn whole(frames: Vec<proto::GenerateResponse>, router_stops: &[String]) -> Value {
    let (stream, server) = scripted_stream(frames, "0").await;
    let (result, _) = HarmonyResponseProcessor::new()
        .process_responses_iteration(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: std::time::Instant::now(),
            },
            Arc::new(request()),
            dispatch(),
            router_stops,
        )
        .await
        .unwrap_or_else(|response| panic!("{}", response.status()));
    server.abort();
    let ResponsesIterationResult::Completed { response, .. } = result else {
        panic!("a turn without calls completes the iteration");
    };
    serde_json::to_value(*response).expect("a JSON object")
}

/// What a streamed iteration left: the events the stream wrote, the stored
/// object of the turn, the terminal event, and the counts the iteration
/// settles with.
struct Streamed {
    events: Vec<Value>,
    stored: Value,
    terminal: Value,
    counts: Option<TokenCounts>,
}

impl Streamed {
    /// The final channel's text as the deltas streamed it.
    fn text(&self) -> String {
        self.events
            .iter()
            .filter(|event| event["type"] == "response.output_text.delta")
            .filter_map(|event| event["delta"].as_str())
            .collect()
    }
}

/// The streamed iteration over `execution` under `router_stops`.
async fn streamed(execution: context::ExecutionResult, router_stops: &[String]) -> Streamed {
    let mut emitter = ResponseStreamEventEmitter::new(
        "resp_harmony-responses-stop".to_string(),
        "harmony-test".to_string(),
        1,
    );
    emitter.set_original_request(request());
    let (tx, mut rx) = sse_channel();
    // The frames are read as they come, so the processor never waits on the
    // channel.
    let collect = async {
        let mut frames = Vec::new();
        while let Some(frame) = rx.recv().await {
            frames.push(String::from_utf8_lossy(&frame.expect("an SSE frame")).into_owned());
        }
        frames
    };
    let iteration = async {
        let result = HarmonyStreamingProcessor::process_responses_iteration_stream(
            execution,
            &mut emitter,
            &tx,
            None,
            None,
            router_stops,
        )
        .await;
        drop(tx);
        result
    };
    let (result, frames) = tokio::join!(iteration, collect);
    let (_, counts) = result.expect("the streamed iteration");
    let events = frames
        .iter()
        .flat_map(|frame| frame.lines())
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).expect("SSE JSON"))
        .collect();
    let stored = serde_json::to_value(emitter.finalize(None)).expect("a JSON object");
    let terminal = emitter.emit_completed(None);
    Streamed {
        events,
        stored,
        terminal,
        counts,
    }
}

#[tokio::test]
async fn a_router_stop_cuts_the_whole_turn_at_the_match() {
    let ids = harmony_tokens(None, "Hello STOP and on and on");

    // Without a stop of the router's the turn stands as the engine ended it.
    let wire = whole(frames(&ids, "length"), &[]).await;
    assert_eq!(
        wire["output"][0]["content"][0]["text"], "Hello STOP and on and on",
        "{wire}"
    );
    assert_eq!(wire["status"], "incomplete", "{wire}");

    let wire = whole(frames(&ids, "length"), &stops()).await;
    assert_eq!(wire["output"][0]["type"], "message", "{wire}");
    assert_eq!(
        wire["output"][0]["content"][0]["text"], "Hello STOP",
        "{wire}"
    );
    assert_eq!(wire["status"], "completed", "{wire}");
    assert!(wire["incomplete_details"].is_null(), "{wire}");
}

#[tokio::test]
async fn a_router_stop_inside_the_analysis_channel_ends_the_turn_before_the_final_channel() {
    let ids = harmony_tokens(Some("Think STOP more"), "Answer");
    let wire = whole(frames(&ids, "stop"), &stops()).await;
    let output = wire["output"].as_array().expect("the output items");
    assert_eq!(
        output.len(),
        1,
        "no message item: the turn ended in the analysis: {wire}"
    );
    assert_eq!(output[0]["type"], "reasoning", "{wire}");
    assert_eq!(output[0]["content"][0]["text"], "Think STOP", "{wire}");
    assert_eq!(wire["status"], "completed", "{wire}");
}

#[tokio::test]
async fn a_router_stop_ends_the_streamed_turn_and_the_engine_request() {
    let ids = harmony_tokens(None, "Hello STOP and on and on and on and on");
    let chunks: Vec<_> = ids.chunks(3).map(|ids| chunk(ids.to_vec())).collect();
    // The engine's Complete waits behind a gate that never opens: a stream
    // read to the engine's end would wait on it.
    let (gate_tx, gate) = oneshot::channel::<()>();
    let (stream, server) =
        gated_stream(chunks, gate, vec![complete(ids.clone(), "length")], "0").await;

    let streamed = timeout(
        Duration::from_secs(5),
        streamed(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: std::time::Instant::now(),
            },
            &stops(),
        ),
    )
    .await
    .expect("the stream ends at the stop, without the engine's Complete");
    server.abort();
    drop(gate_tx);

    assert_eq!(streamed.text(), "Hello STOP", "{:?}", streamed.events);
    assert_eq!(
        streamed.terminal["type"], "response.completed",
        "{}",
        streamed.terminal
    );
    assert_eq!(
        streamed.terminal["response"]["output"][0]["content"][0]["text"], "Hello STOP",
        "{}",
        streamed.terminal
    );
    assert_eq!(
        streamed.stored["status"], "completed",
        "{}",
        streamed.stored
    );
    assert_eq!(
        streamed.stored["output"][0]["content"][0]["text"], "Hello STOP",
        "{}",
        streamed.stored
    );
    // The usage counts through the chunk the stop matched in, with that
    // chunk's prompt count; the engine's tail is nobody's.
    let counts = streamed
        .counts
        .expect("the turn's counts, from the chunk the stop matched in");
    assert_eq!(counts.prompt, 3);
    assert!(
        counts.completion > 0 && (counts.completion as usize) < ids.len(),
        "{} of {} tokens",
        counts.completion,
        ids.len()
    );
}

#[tokio::test]
async fn a_router_stop_inside_the_analysis_channel_streams_no_message() {
    // One chunk carries the whole turn: the final channel's text arrives in the
    // same delta as the analysis the stop is in, and the engine's Complete comes
    // right after the chunk the stop matched in.
    let ids = harmony_tokens(Some("Think STOP more"), "Answer");
    let frames = vec![chunk(ids.clone()), complete_with_spec(ids, "stop", 2, 3)];
    let (stream, server) = scripted_stream(frames, "0").await;
    let streamed = timeout(
        Duration::from_secs(5),
        streamed(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: std::time::Instant::now(),
            },
            &stops(),
        ),
    )
    .await
    .expect("the stream ends with the engine's Complete");
    server.abort();

    assert!(
        streamed
            .events
            .iter()
            .all(|event| event["type"] != "response.output_text.delta"),
        "nothing of the final channel streams past a stop in the analysis: {:?}",
        streamed.events
    );
    let output = streamed.stored["output"]
        .as_array()
        .expect("the output items");
    assert!(
        output.iter().any(|item| item["type"] == "reasoning"),
        "the analysis channel opened its item: {}",
        streamed.stored
    );
    assert!(
        output.iter().all(|item| item["type"] != "message"),
        "no message item: the turn ended in the analysis: {}",
        streamed.stored
    );
    assert_eq!(
        streamed.stored["status"], "completed",
        "{}",
        streamed.stored
    );
    // The Complete after the stop chunk adds the counts the stop could not know.
    let counts = streamed.counts.expect("the turn's counts");
    assert_eq!(
        (counts.prompt, counts.spec_accepted, counts.spec_drafted),
        (3, 2, 3)
    );
}

#[tokio::test]
async fn the_engine_s_complete_right_after_the_stop_chunk_adds_only_the_counts_the_stop_could_not_know(
) {
    // The stop matches in the last chunk before the Complete: the engine ended
    // the turn on that step (its own finish is `length`), nothing is aborted,
    // and the turn stays the router's `stop`.
    let ids = harmony_tokens(None, "Hello STOP");
    let frames = vec![
        chunk(ids.clone()),
        complete_with_spec(ids.clone(), "length", 2, 3),
    ];
    let (stream, server) = scripted_stream(frames, "0").await;
    let streamed = timeout(
        Duration::from_secs(5),
        streamed(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: std::time::Instant::now(),
            },
            &stops(),
        ),
    )
    .await
    .expect("the stream ends with the engine's Complete");
    server.abort();

    assert_eq!(streamed.text(), "Hello STOP", "{:?}", streamed.events);
    assert_eq!(
        streamed.stored["status"], "completed",
        "{}",
        streamed.stored
    );
    assert!(
        streamed.stored["incomplete_details"].is_null(),
        "the engine's length finish does not overrule the router's stop: {}",
        streamed.stored
    );
    assert_eq!(
        streamed.terminal["type"], "response.completed",
        "{}",
        streamed.terminal
    );
    let counts = streamed.counts.expect("the turn's counts");
    assert_eq!(
        (
            counts.prompt,
            counts.completion as usize,
            counts.spec_accepted,
            counts.spec_drafted
        ),
        (3, ids.len(), 2, 3)
    );
}

#[tokio::test]
async fn the_text_a_scanner_held_back_is_released_when_no_stop_comes() {
    // "S" could still grow into the stop: held back until the turn ends.
    let ids = harmony_tokens(None, "Hello S");
    let (stream, server) = scripted_stream(frames(&ids, "stop"), "0").await;
    let streamed = streamed(
        context::ExecutionResult::Single {
            stream,
            dispatch_start: std::time::Instant::now(),
        },
        &stops(),
    )
    .await;
    server.abort();

    assert_eq!(streamed.text(), "Hello S", "{:?}", streamed.events);
    assert_eq!(
        streamed.stored["output"][0]["content"][0]["text"], "Hello S",
        "{}",
        streamed.stored
    );
    assert_eq!(
        streamed.stored["status"], "completed",
        "{}",
        streamed.stored
    );
    assert!(streamed.counts.is_some(), "the terminal frame's counts");
}

#[tokio::test]
async fn the_whole_chat_turn_keeps_its_cut_at_the_router_s_stop() {
    let ids = harmony_tokens(None, "Hello STOP and on");
    let (stream, server) = scripted_stream(frames(&ids, "length"), "0").await;
    let request: ChatCompletionRequest =
        serde_json::from_value(json!({"model": "harmony-test", "messages": []}))
            .expect("chat request");
    let (response, _) = HarmonyResponseProcessor::new()
        .process_non_streaming_chat_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: std::time::Instant::now(),
            },
            Arc::new(request),
            dispatch(),
            &stops(),
        )
        .await
        .unwrap_or_else(|response| panic!("{}", response.status()));
    server.abort();

    let wire = serde_json::to_value(response).expect("a JSON response");
    assert_eq!(
        wire["choices"][0]["message"]["content"], "Hello STOP",
        "{wire}"
    );
    assert_eq!(wire["choices"][0]["finish_reason"], "stop", "{wire}");
    assert_eq!(wire["choices"][0]["matched_stop"], STOP, "{wire}");
}
