//! The streaming processor opens a server call's item only within the turn's
//! allowance (the calls the request may still run under its cap; none on the
//! final turn after it): a call past the allowance is parsed for the loop,
//! which refuses it, and never reaches the client or the stored object as a
//! completed `mcp_call`; a call to the client's function is not counted.

use std::{borrow::Cow, sync::Arc};

use serde_json::{json, Value};
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};
use smg_mcp::{McpConfig, McpOrchestrator, McpServerBinding, McpToolSession, Tool, ToolEntry};

use super::super::HarmonyStreamingProcessor;
use crate::routers::{
    common::{openai_bridge::FormatRegistry, sse::sse_channel},
    grpc::{
        common::responses::streaming::ResponseStreamEventEmitter,
        context,
        harmony::{processor::ResponsesIterationResult, try_harmony_encoding},
        regular::streaming::tests::eof::scripted_stream,
    },
};

/// A call to the session's server tool the turn ended on.
const SERVER_CALL: &str = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\": \"Paris\"}<|call|>";
/// A call to a function the session does not expose: the client's.
const CLIENT_CALL: &str = "<|channel|>commentary to=functions.get_time <|constrain|>json<|message|>{\"city\": \"Paris\"}<|call|>";
/// A second message of the turn, a call to the session's other server tool,
/// in two pieces: up to a point inside its arguments, and the rest.
const SECOND_SERVER_CALL: [&str; 2] = [
    "<|start|>assistant<|channel|>commentary to=functions.get_forecast <|constrain|>json<|message|>{\"city\"",
    ": \"Paris\"}<|call|>",
];

fn ids(text: &str) -> Vec<u32> {
    try_harmony_encoding()
        .expect("harmony encoding must load")
        .tokenizer()
        .encode_with_special_tokens(text)
}

fn in_threes(turn: &str) -> Vec<Vec<u32>> {
    ids(turn).chunks(3).map(<[u32]>::to_vec).collect()
}

/// The frames of a turn: one per chunk of its tokens, then the terminal
/// frame.
fn frames(chunks: Vec<Vec<u32>>) -> Vec<proto::GenerateResponse> {
    let ids = chunks.concat();
    let mut frames: Vec<proto::GenerateResponse> = chunks
        .into_iter()
        .map(|chunk| proto::GenerateResponse {
            response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
                index: 0,
                completion_tokens: chunk.len() as u32,
                token_ids: chunk,
                ..Default::default()
            })),
        })
        .collect();
    frames.push(proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            index: 0,
            finish_reason: "stop".to_string(),
            prompt_tokens: 3,
            completion_tokens: ids.len() as u32,
            output_ids: ids,
            ..Default::default()
        })),
    });
    frames
}

/// An orchestrator whose inventory holds the server tools `get_weather` and
/// `get_forecast` of the server `weather`; no server is live, the processor
/// reads the exposure only.
async fn orchestrator() -> McpOrchestrator {
    let orchestrator = McpOrchestrator::new(McpConfig::default())
        .await
        .expect("mcp orchestrator");
    for name in ["get_weather", "get_forecast"] {
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "weather",
                Tool::new(
                    Cow::Borrowed(name),
                    Cow::Borrowed("The weather of a city"),
                    Arc::new(serde_json::Map::new()),
                ),
            ));
    }
    orchestrator
}

fn binding() -> Vec<McpServerBinding> {
    vec![McpServerBinding {
        label: "weather".to_string(),
        server_key: "weather".to_string(),
        allowed_tools: None,
        list_error: None,
        pool_key: None,
    }]
}

/// One streamed turn, its tokens in `chunks`, through a fresh emitter under
/// `allowance`: the iteration's result, the SSE frames the client received,
/// and the stored object.
async fn streamed(
    session: &McpToolSession<'_>,
    chunks: Vec<Vec<u32>>,
    allowance: usize,
) -> (ResponsesIterationResult, Vec<String>, Value) {
    let (stream, server) = scripted_stream(frames(chunks), "0").await;
    let mut emitter = ResponseStreamEventEmitter::new(
        "resp_harmony-allowance".to_string(),
        "harmony-test".to_string(),
        1,
    );
    let registry = FormatRegistry::new();
    let (tx, mut rx) = sse_channel();
    // The frames are read as they come, so the processor never waits on the
    // channel.
    let read = async {
        let mut frames = Vec::new();
        while let Some(frame) = rx.recv().await {
            if let Ok(frame) = frame {
                frames.push(String::from_utf8_lossy(&frame).into_owned());
            }
        }
        frames
    };
    let iteration = async {
        let result = HarmonyStreamingProcessor::process_responses_iteration_stream(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: std::time::Instant::now(),
            },
            &mut emitter,
            &tx,
            Some(session),
            Some(&registry),
            &[],
            Some(allowance),
        )
        .await;
        drop(tx);
        result
    };
    let (streamed, frames) = tokio::join!(iteration, read);
    server.abort();
    let (result, _) = streamed.expect("the streamed iteration");
    let stored = serde_json::to_value(emitter.finalize(None)).expect("a JSON object");
    (result, frames, stored)
}

/// The item types the turn's `output_item.added` events opened.
fn opened(frames: &[String]) -> Vec<&'static str> {
    frames
        .iter()
        .filter(|frame| frame.contains("response.output_item.added"))
        .map(|frame| {
            if frame.contains("\"mcp_call\"") {
                "mcp_call"
            } else if frame.contains("\"function_call\"") {
                "function_call"
            } else {
                "other"
            }
        })
        .collect()
}

#[tokio::test]
async fn a_server_call_past_the_turns_allowance_opens_no_item() {
    let orchestrator = orchestrator().await;
    let session = McpToolSession::new(&orchestrator, binding(), "resp_harmony-allowance");
    assert!(
        session.has_exposed_tool("get_weather"),
        "the session exposes the server tool"
    );

    let (result, frames, stored) = streamed(&session, in_threes(SERVER_CALL), 0).await;
    let ResponsesIterationResult::ToolCallsFound { tool_calls, .. } = result else {
        panic!("the call is parsed for the loop, which refuses it");
    };
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "get_weather");
    assert!(
        !frames
            .iter()
            .any(|frame| frame.contains("response.output_item.added") || frame.contains("mcp_call")),
        "no item reached the client: {frames:?}"
    );
    assert_eq!(stored["output"], json!([]), "{stored}");
}

#[tokio::test]
async fn a_server_call_within_the_allowance_opens_its_item_as_before() {
    let orchestrator = orchestrator().await;
    let session = McpToolSession::new(&orchestrator, binding(), "resp_harmony-allowance");

    let (result, frames, stored) = streamed(&session, in_threes(SERVER_CALL), 1).await;
    assert!(
        matches!(result, ResponsesIterationResult::ToolCallsFound { .. }),
        "the call is returned to the loop"
    );
    assert_eq!(opened(&frames), ["mcp_call"], "{frames:?}");
    assert!(
        frames.iter().any(|frame| {
            frame.contains("response.output_item.done")
                && frame.contains("\"mcp_call\"")
                && frame.contains("\"status\":\"completed\"")
        }),
        "the call's item closes as completed, for the loop to fill with its output: {frames:?}"
    );
    // The stored object lists a server call once the loop sets its output
    // (`update_mcp_call_outputs`), not within the iteration that announced it.
    assert_eq!(stored["output"], json!([]), "{stored}");
}

/// The allowance is the call's place among the turn's server calls, as the
/// loop cuts its permitted prefix, whatever deltas the chunks bring: a chunk
/// that holds the whole first call and the head of the second brings a delta
/// for the second only, and the whole turn in one chunk brings none.
#[tokio::test]
async fn a_server_call_past_the_allowance_opens_no_item_however_the_chunks_fall() {
    let orchestrator = orchestrator().await;
    let session = McpToolSession::new(&orchestrator, binding(), "resp_harmony-allowance");
    let [head, tail] = SECOND_SERVER_CALL;
    let turn = format!("{SERVER_CALL}{head}{tail}");

    for chunks in [
        vec![ids(&format!("{SERVER_CALL}{head}")), ids(tail)],
        vec![ids(&turn)],
        in_threes(&turn),
    ] {
        let sizes: Vec<usize> = chunks.iter().map(Vec::len).collect();
        let (result, frames, stored) = streamed(&session, chunks, 1).await;
        let ResponsesIterationResult::ToolCallsFound { tool_calls, .. } = result else {
            panic!("both calls are parsed for the loop, chunks {sizes:?}");
        };
        let called: Vec<&str> = tool_calls
            .iter()
            .map(|call| call.function.name.as_str())
            .collect();
        assert_eq!(called, ["get_weather", "get_forecast"], "chunks {sizes:?}");
        assert!(
            !frames.iter().any(|frame| frame.contains("get_forecast")),
            "the call past the allowance reached the client, chunks {sizes:?}: {frames:?}"
        );
        assert_eq!(stored["output"], json!([]), "chunks {sizes:?}: {stored}");
    }
}

#[tokio::test]
async fn a_clients_function_call_is_not_counted_against_the_allowance() {
    let orchestrator = orchestrator().await;
    let session = McpToolSession::new(&orchestrator, binding(), "resp_harmony-allowance");

    let (result, frames, stored) = streamed(&session, in_threes(CLIENT_CALL), 0).await;
    assert!(
        matches!(result, ResponsesIterationResult::ToolCallsFound { .. }),
        "the call is returned to the loop"
    );
    assert_eq!(opened(&frames), ["function_call"], "{frames:?}");
    assert_eq!(stored["output"][0]["type"], "function_call", "{stored}");
}
