//! The status of a Harmony Responses turn is read from its finish reason on
//! both transports, as the regular paths read theirs: a turn cut by
//! `max_output_tokens` (a `length` finish) is `incomplete` with the reason
//! `max_output_tokens` on the non-streamed object, on the streamed terminal
//! event and on the stored object of a streamed turn; a call the stream
//! found cut (its arguments not complete JSON under a finish other than
//! `tool_calls`, or no finish at all) leaves the response `incomplete`
//! without a reason and the call item `incomplete`; a turn that ended on its
//! closed calls stays `completed`; and a later iteration of one request reads
//! its own finish, not the previous turn's.

use openai_protocol::responses::ResponsesRequest;
use serde_json::{json, Value};
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};

use super::super::HarmonyStreamingProcessor;
use crate::routers::{
    common::sse::sse_channel,
    grpc::{
        common::responses::streaming::ResponseStreamEventEmitter,
        context,
        harmony::{
            processor::{HarmonyResponseProcessor, ResponsesIterationResult},
            try_harmony_encoding,
        },
        regular::streaming::tests::eof::scripted_stream,
    },
};

/// A final-channel answer the token limit cut: no `<|return|>`.
const CUT_ANSWER: &str = "<|channel|>final<|message|>Hello wor";
/// A call to a function the token limit cut inside its arguments.
const CUT_CALL: &str =
    "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\": \"Par";
/// A call the turn ended on.
const CLOSED_CALL: &str = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\": \"Paris\"}<|call|>";

/// The token ids of `turn`, its special tokens included.
fn tokens(turn: &str) -> Vec<u32> {
    try_harmony_encoding()
        .expect("harmony encoding must load")
        .tokenizer()
        .encode_with_special_tokens(turn)
}

/// The frames of a turn: `ids` in chunks of three, then the terminal frame
/// with `finish_reason` and the whole output; no terminal frame when
/// `finish_reason` is `None` (the stream ends before it).
fn frames(ids: &[u32], finish_reason: Option<&str>) -> Vec<proto::GenerateResponse> {
    let mut frames: Vec<proto::GenerateResponse> = ids
        .chunks(3)
        .map(|chunk| proto::GenerateResponse {
            response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
                index: 0,
                completion_tokens: chunk.len() as u32,
                token_ids: chunk.to_vec(),
                ..Default::default()
            })),
        })
        .collect();
    if let Some(finish_reason) = finish_reason {
        frames.push(proto::GenerateResponse {
            response: Some(GenerationEvent::Complete(proto::GenerateComplete {
                index: 0,
                finish_reason: finish_reason.to_string(),
                prompt_tokens: 3,
                completion_tokens: ids.len() as u32,
                output_ids: ids.to_vec(),
                ..Default::default()
            })),
        });
    }
    frames
}

fn request() -> ResponsesRequest {
    serde_json::from_value(json!({
        "model": "harmony-test", "input": "hi", "max_output_tokens": 8,
        "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]
    }))
    .expect("responses request")
}

fn dispatch() -> context::DispatchMetadata {
    context::DispatchMetadata {
        request_id: "harmony-status".to_string(),
        model: "harmony-test".to_string(),
        created: 1,
        weight_version: None,
    }
}

/// The non-streamed iteration over `frames`: its result.
async fn whole(frames: Vec<proto::GenerateResponse>) -> ResponsesIterationResult {
    let (stream, server) = scripted_stream(frames, "0").await;
    let (result, _) = HarmonyResponseProcessor::new()
        .process_responses_iteration(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: std::time::Instant::now(),
            },
            std::sync::Arc::new(request()),
            dispatch(),
            &[],
        )
        .await
        .unwrap_or_else(|response| panic!("{}", response.status()));
    server.abort();
    result
}

fn emitter() -> ResponseStreamEventEmitter {
    let mut emitter = ResponseStreamEventEmitter::new(
        "resp_harmony-status".to_string(),
        "harmony-test".to_string(),
        1,
    );
    emitter.set_original_request(request());
    emitter
}

/// One streamed iteration over `frames` through `emitter`: its result.
async fn iterate(
    emitter: &mut ResponseStreamEventEmitter,
    frames: Vec<proto::GenerateResponse>,
) -> ResponsesIterationResult {
    let (stream, server) = scripted_stream(frames, "0").await;
    let (tx, mut rx) = sse_channel();
    // The events are read as they come, so the processor never waits on the
    // channel.
    let drain = async { while rx.recv().await.is_some() {} };
    let iteration = async {
        let result = HarmonyStreamingProcessor::process_responses_iteration_stream(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: std::time::Instant::now(),
            },
            emitter,
            &tx,
            None,
            None,
            &[],
        )
        .await;
        drop(tx);
        result
    };
    let (streamed, ()) = tokio::join!(iteration, drain);
    server.abort();
    let (result, _) = streamed.expect("the streamed iteration");
    result
}

/// The stored object of the turn and the terminal event the stream ends with,
/// as the serving paths read them: the stored object before the terminal,
/// which drains the items.
fn objects(emitter: &mut ResponseStreamEventEmitter) -> (Value, Value) {
    let stored = serde_json::to_value(emitter.finalize(None)).expect("a JSON object");
    let terminal = emitter.emit_completed(None);
    (stored, terminal)
}

/// The streamed iteration over `frames` on a fresh emitter: the stored object
/// of the turn, the terminal event the stream ends with, and the result.
async fn streamed(
    frames: Vec<proto::GenerateResponse>,
) -> (Value, Value, ResponsesIterationResult) {
    let mut emitter = emitter();
    let result = iterate(&mut emitter, frames).await;
    let (stored, terminal) = objects(&mut emitter);
    (stored, terminal, result)
}

fn assert_incomplete(object: &Value, reason: Option<&str>, context: &str) {
    assert_eq!(object["status"], "incomplete", "{context}: {object}");
    match reason {
        Some(reason) => assert_eq!(
            object["incomplete_details"]["reason"], reason,
            "{context}: {object}"
        ),
        None => assert!(
            object["incomplete_details"].is_null(),
            "{context}: {object}"
        ),
    }
    assert!(object["error"].is_null(), "{context}: {object}");
}

#[tokio::test]
async fn a_turn_cut_by_max_output_tokens_reads_incomplete_on_both_transports() {
    let ids = tokens(CUT_ANSWER);

    let ResponsesIterationResult::Completed { response, .. } =
        whole(frames(&ids, Some("length"))).await
    else {
        panic!("a turn without calls completes the iteration");
    };
    let wire = serde_json::to_value(*response).expect("a JSON object");
    assert_incomplete(&wire, Some("max_output_tokens"), "the non-streamed object");
    assert_eq!(wire["output"][0]["type"], "message", "{wire}");
    assert_eq!(
        wire["output"][0]["content"][0]["text"], "Hello wor",
        "{wire}"
    );

    let (stored, terminal, _) = streamed(frames(&ids, Some("length"))).await;
    assert_incomplete(&stored, Some("max_output_tokens"), "the stored object");
    assert_eq!(terminal["type"], "response.incomplete", "{terminal}");
    assert_incomplete(
        &terminal["response"],
        Some("max_output_tokens"),
        "the terminal event",
    );
    assert_eq!(
        terminal["response"]["output"][0]["content"][0]["text"],
        "Hello wor"
    );
}

#[tokio::test]
async fn a_streamed_call_the_finish_cut_reads_incomplete() {
    let ids = tokens(CUT_CALL);
    for (finish_reason, reason) in [
        (Some("length"), Some("max_output_tokens")),
        (Some("abort"), None),
        (None, None),
    ] {
        let context = format!("finish {finish_reason:?}");
        let (stored, terminal, result) = streamed(frames(&ids, finish_reason)).await;
        let ResponsesIterationResult::ToolCallsFound { tool_calls, .. } = result else {
            panic!("{context}: the cut call is recovered");
        };
        assert_eq!(
            tool_calls[0].function.arguments.as_deref(),
            Some("{\"city\": \"Par"),
            "{context}"
        );
        assert_incomplete(&stored, reason, &format!("{context}, the stored object"));
        assert_eq!(stored["output"][0]["type"], "function_call", "{context}");
        assert_eq!(stored["output"][0]["status"], "incomplete", "{context}");
        assert_eq!(
            terminal["type"], "response.incomplete",
            "{context}: {terminal}"
        );
        assert_incomplete(
            &terminal["response"],
            reason,
            &format!("{context}, the terminal event"),
        );
        assert_eq!(
            terminal["response"]["output"][0]["status"], "incomplete",
            "{context}"
        );
    }
}

#[tokio::test]
async fn a_turn_that_ended_on_its_closed_calls_stays_completed() {
    let ids = tokens(CLOSED_CALL);

    assert!(
        matches!(
            whole(frames(&ids, Some("stop"))).await,
            ResponsesIterationResult::ToolCallsFound { .. }
        ),
        "a turn that ended on its call returns it"
    );

    let (stored, terminal, _) = streamed(frames(&ids, Some("stop"))).await;
    assert_eq!(stored["status"], "completed", "{stored}");
    assert!(stored["incomplete_details"].is_null(), "{stored}");
    assert_eq!(stored["output"][0]["status"], "completed", "{stored}");
    assert_eq!(terminal["type"], "response.completed", "{terminal}");
    assert_eq!(terminal["response"]["status"], "completed", "{terminal}");
}

/// One request's emitter serves every iteration of a tool loop: a later
/// iteration whose stream ends before its terminal frame reads its own
/// (absent) finish, not the `tool_calls` the previous turn ended on, so a
/// call it cut reads `incomplete` and the response with it.
#[tokio::test]
async fn a_later_iteration_reads_its_own_finish_not_the_previous_turns() {
    let mut emitter = emitter();
    let first = iterate(&mut emitter, frames(&tokens(CLOSED_CALL), Some("stop"))).await;
    assert!(
        matches!(first, ResponsesIterationResult::ToolCallsFound { .. }),
        "the first turn ended on its call"
    );
    let second = iterate(&mut emitter, frames(&tokens(CUT_CALL), None)).await;
    let ResponsesIterationResult::ToolCallsFound { tool_calls, .. } = second else {
        panic!("the second turn's cut call is recovered");
    };
    assert_eq!(
        tool_calls[0].function.arguments.as_deref(),
        Some("{\"city\": \"Par"),
    );
    let (stored, terminal) = objects(&mut emitter);
    assert_eq!(stored["output"][0]["status"], "completed", "{stored}");
    assert_eq!(stored["output"][1]["status"], "incomplete", "{stored}");
    assert_incomplete(&stored, None, "the stored object");
    assert_eq!(terminal["type"], "response.incomplete", "{terminal}");
    assert_eq!(
        terminal["response"]["output"][1]["status"], "incomplete",
        "{terminal}"
    );
    assert_incomplete(&terminal["response"], None, "the terminal event");
}
