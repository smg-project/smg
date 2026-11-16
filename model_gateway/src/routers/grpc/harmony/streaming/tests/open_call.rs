//! The call a turn ended inside is the turn's call on the non-streamed reads,
//! as it is on the streams. A worker may stop at a call's closing token
//! without delivering it, and the token limit may cut a call's arguments: the
//! chat answer carries that call in `tool_calls` and the Responses iteration
//! returns it to the loop, on any channel, as far as it came. Its arguments
//! are neither reasoning nor answer text. A turn that stopped inside a whole
//! call finished on it: its `stop` reads `tool_calls`, on the non-streamed
//! answer and on the chat stream's final chunk alike; a turn the token limit
//! ended keeps `length`, and one that stopped inside a call's arguments
//! keeps `stop`.

use std::{sync::Arc, time::Instant};

use openai_protocol::{chat::ChatCompletionRequest, responses::ResponsesRequest};
use serde_json::{json, Value};
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};

use super::super::HarmonyStreamingProcessor;
use crate::{
    observability::metrics::metrics_labels,
    routers::grpc::{
        context,
        harmony::{
            processor::{HarmonyResponseProcessor, ResponsesIterationResult},
            try_harmony_encoding,
        },
        regular::streaming::tests::eof::scripted_stream,
        utils,
    },
};

/// A call's whole arguments, and the same arguments cut by the token limit.
const WHOLE: &str = r#"{"city": "Paris"}"#;
const CUT: &str = r#"{"city": "Par"#;

/// The ways a turn ends inside a call: the channel the call is written on,
/// the engine's finish, the arguments as far as they came, and the finish the
/// reads report.
const TURNS: [(&str, &str, &str, &str); 6] = [
    // The worker stopped at the call's closing token without delivering it.
    ("commentary", "stop", WHOLE, "tool_calls"),
    ("analysis", "stop", WHOLE, "tool_calls"),
    // The token limit ended the turn right behind whole arguments.
    ("commentary", "length", WHOLE, "length"),
    // The token limit cut the arguments.
    ("commentary", "length", CUT, "length"),
    ("analysis", "length", CUT, "length"),
    // The worker stopped inside the arguments: the turn did not end on a call.
    ("commentary", "stop", CUT, "stop"),
];

/// The engine's frames for one such turn: its tokens in chunks of three (the
/// streamed reads parse them), then the terminal frame with the whole output
/// (the non-streamed reads parse it), the call's closing token in neither.
fn frames(channel: &str, finish_reason: &str, arguments: &str) -> Vec<proto::GenerateResponse> {
    let turn = format!(
        "<|channel|>{channel} to=functions.get_weather <|constrain|>json<|message|>{arguments}"
    );
    let output_ids = try_harmony_encoding()
        .expect("harmony encoding must load")
        .tokenizer()
        .encode_with_special_tokens(&turn);
    let mut frames: Vec<proto::GenerateResponse> = output_ids
        .chunks(3)
        .map(|ids| super::chunk(ids.to_vec()))
        .collect();
    frames.push(proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            index: 0,
            finish_reason: finish_reason.to_string(),
            prompt_tokens: 3,
            completion_tokens: output_ids.len() as u32,
            output_ids,
            ..Default::default()
        })),
    });
    frames
}

fn dispatch() -> context::DispatchMetadata {
    context::DispatchMetadata {
        request_id: "harmony-open-call".to_string(),
        model: "harmony-test".to_string(),
        created: 1,
        weight_version: None,
    }
}

fn chat_request(stream: bool) -> Arc<ChatCompletionRequest> {
    Arc::new(
        serde_json::from_value(json!({
            "model": "harmony-test", "messages": [], "n": 1, "stream": stream,
        }))
        .expect("chat request"),
    )
}

#[tokio::test]
async fn a_non_streamed_chat_answer_carries_the_call_its_turn_ended_inside() {
    let request = chat_request(false);
    let mut read = Vec::new();
    let mut due = Vec::new();
    for (channel, finish_reason, arguments, finish) in TURNS {
        let (stream, server) =
            scripted_stream(frames(channel, finish_reason, arguments), "0").await;
        let (response, _) = HarmonyResponseProcessor::new()
            .process_non_streaming_chat_response(
                context::ExecutionResult::Single {
                    stream,
                    dispatch_start: Instant::now(),
                },
                Arc::clone(&request),
                dispatch(),
                &[],
            )
            .await
            .unwrap_or_else(|response| panic!("{}", response.status()));
        server.abort();
        let wire = serde_json::to_value(response).expect("a JSON object");
        let choice = &wire["choices"][0];
        let calls: Vec<&Value> = choice["message"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|call| &call["function"])
            .collect();
        read.push(json!({
            "turn": [channel, finish_reason, arguments],
            "calls": calls,
            "content": choice["message"]["content"],
            "reasoning_content": choice["message"]["reasoning_content"],
            "finish_reason": choice["finish_reason"],
        }));
        due.push(json!({
            "turn": [channel, finish_reason, arguments],
            "calls": [{"name": "get_weather", "arguments": arguments}],
            "content": null,
            "reasoning_content": null,
            "finish_reason": finish,
        }));
    }
    assert_eq!(read, due);
}

#[tokio::test]
async fn a_non_streamed_responses_iteration_returns_the_call_its_turn_ended_inside() {
    let request: Arc<ResponsesRequest> = Arc::new(
        serde_json::from_value(json!({
            "model": "harmony-test", "input": "hi",
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]
        }))
        .expect("responses request"),
    );
    let mut read = Vec::new();
    let mut due = Vec::new();
    for (channel, finish_reason, arguments, finish) in TURNS {
        let (stream, server) =
            scripted_stream(frames(channel, finish_reason, arguments), "0").await;
        let (result, _) = HarmonyResponseProcessor::new()
            .process_responses_iteration(
                context::ExecutionResult::Single {
                    stream,
                    dispatch_start: Instant::now(),
                },
                Arc::clone(&request),
                dispatch(),
                &[],
            )
            .await
            .unwrap_or_else(|response| panic!("{}", response.status()));
        server.abort();
        read.push(match result {
            ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis,
                partial_text,
                finish_reason: found_finish,
                ..
            } => json!({
                "turn": [channel, finish_reason, arguments],
                "calls": tool_calls.iter().map(|call| &call.function).collect::<Vec<_>>(),
                "reasoning": analysis,
                "text": partial_text,
                "finish_reason": found_finish,
            }),
            ResponsesIterationResult::Completed { response, .. } => {
                let wire = serde_json::to_value(*response).expect("a JSON object");
                json!({
                    "turn": [channel, finish_reason, arguments],
                    "completed without the call, output": wire["output"],
                })
            }
        });
        due.push(json!({
            "turn": [channel, finish_reason, arguments],
            "calls": [{"name": "get_weather", "arguments": arguments}],
            "reasoning": null,
            "text": "",
            "finish_reason": finish,
        }));
    }
    assert_eq!(read, due);
}

/// The chat stream's final chunk reads the same finish as the non-streamed
/// answer: `tool_calls` for a stream that stopped inside a whole call, the
/// engine's `length` for one the token limit ended.
#[tokio::test]
async fn a_chat_stream_that_stopped_inside_a_whole_call_finishes_on_its_call() {
    let request = chat_request(true);
    let mut read = Vec::new();
    let mut due = Vec::new();
    for (channel, finish_reason, arguments, finish) in TURNS {
        let (stream, server) =
            scripted_stream(frames(channel, finish_reason, arguments), "0").await;
        let dispatch = dispatch();
        let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_CHAT);
        let response = Arc::new(HarmonyStreamingProcessor::new())
            .process_streaming_chat_response(
                context::ExecutionResult::Single {
                    stream,
                    dispatch_start: Instant::now(),
                },
                Arc::clone(&request),
                dispatch,
                Vec::new(),
                None,
                None,
                source,
            )
            .await;
        let body = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("the stream's body");
        server.abort();
        let finishes: Vec<Value> = std::str::from_utf8(&body)
            .expect("an event stream")
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str::<Value>(data).expect("a JSON event"))
            .map(|event| event["choices"][0]["finish_reason"].clone())
            .filter(|finish| !finish.is_null())
            .collect();
        read.push(json!({"turn": [channel, finish_reason, arguments], "finishes": finishes}));
        due.push(json!({"turn": [channel, finish_reason, arguments], "finishes": [finish]}));
    }
    assert_eq!(read, due);
}
