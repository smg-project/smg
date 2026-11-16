//! The streaming processor opens a server call's item only within the turn's
//! allowance (the calls the request may still run under its cap; none on the
//! final turn after it): a call past the allowance is parsed for the loop,
//! which refuses it, and never reaches the client or the stored object as a
//! completed `mcp_call`; a call to the client's function is not counted.
//! Every other call of the turn opens its item with all of its arguments,
//! however the turn's tokens are cut into chunks. A server call the turn
//! stopped inside is not run: its item, if it has one, closes in a status
//! its type defines, and the response reads the turn as cut with or without
//! the item.

use std::{borrow::Cow, collections::HashMap, sync::Arc};

use openai_protocol::responses::{
    CodeInterpreterCallStatus, FileSearchCallStatus, ImageGenerationCallStatus, WebSearchCallStatus,
};
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};
use smg_mcp::{
    McpConfig, McpOrchestrator, McpServerBinding, McpServerConfig, McpToolSession, McpTransport,
    ResponseFormatConfig, Tool, ToolConfig, ToolEntry,
};

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
/// A call to the session's server tool that the turn stopped inside: its
/// arguments are cut, and no closing token came.
const CUT_SERVER_CALL: &str =
    "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\"";

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
    streamed_as(session, &FormatRegistry::new(), chunks, allowance).await
}

/// [`streamed`], with the formats of the session's tools read from
/// `registry`.
async fn streamed_as(
    session: &McpToolSession<'_>,
    registry: &FormatRegistry,
    chunks: Vec<Vec<u32>>,
    allowance: usize,
) -> (ResponsesIterationResult, Vec<String>, Value) {
    let (stream, server) = scripted_stream(frames(chunks), "0").await;
    let mut emitter = ResponseStreamEventEmitter::new(
        "resp_harmony-allowance".to_string(),
        "harmony-test".to_string(),
        1,
    );
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
            Some(registry),
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

/// The arguments the turn's `function_call_arguments.delta` events add up to.
fn streamed_arguments(frames: &[String]) -> String {
    frames
        .iter()
        .filter_map(|frame| serde_json::from_str::<Value>(frame[frame.find('{')?..].trim()).ok())
        .filter(|event| event["type"] == "response.function_call_arguments.delta")
        .filter_map(|event| event["delta"].as_str().map(str::to_owned))
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
/// loop cuts its permitted prefix, however the chunks fall: the first call
/// opens its item and the second none, whether a chunk holds the whole first
/// call and the head of the second, the whole turn, or three tokens.
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
        assert_eq!(
            opened(&frames),
            ["mcp_call"],
            "chunks {sizes:?}: {frames:?}"
        );
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

#[tokio::test]
async fn a_clients_call_is_announced_with_all_its_arguments_however_the_chunks_fall() {
    let orchestrator = orchestrator().await;
    let session = McpToolSession::new(&orchestrator, binding(), "resp_harmony-allowance");
    let tokens = ids(CLIENT_CALL);

    // The last size holds the whole call in one chunk.
    for size in (1..=8).chain([tokens.len()]) {
        let chunks = tokens.chunks(size).map(<[u32]>::to_vec).collect();
        let (result, frames, stored) = streamed(&session, chunks, 0).await;
        assert!(
            matches!(result, ResponsesIterationResult::ToolCallsFound { .. }),
            "the call is returned to the loop, chunks of {size}"
        );
        assert_eq!(
            opened(&frames),
            ["function_call"],
            "chunks of {size}: {frames:?}"
        );
        assert_eq!(
            streamed_arguments(&frames),
            r#"{"city": "Paris"}"#,
            "chunks of {size}: {frames:?}"
        );
        assert_eq!(
            stored["output"][0]["arguments"], r#"{"city": "Paris"}"#,
            "chunks of {size}: {stored}"
        );
    }
}

#[tokio::test]
async fn a_second_call_to_the_same_tool_opens_its_own_item() {
    let orchestrator = orchestrator().await;
    let session = McpToolSession::new(&orchestrator, binding(), "resp_harmony-allowance");
    let turn = format!("{CLIENT_CALL}<|start|>assistant{CLIENT_CALL}");

    let (result, frames, stored) = streamed(&session, in_threes(&turn), 0).await;
    let ResponsesIterationResult::ToolCallsFound { tool_calls, .. } = result else {
        panic!("both calls are returned to the loop");
    };
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(
        opened(&frames),
        ["function_call", "function_call"],
        "{frames:?}"
    );
    assert_eq!(
        stored["output"].as_array().map(Vec::len),
        Some(2),
        "{stored}"
    );
}

#[tokio::test]
async fn a_call_on_the_analysis_channel_opens_a_call_item_not_a_reasoning_item() {
    let orchestrator = orchestrator().await;
    let session = McpToolSession::new(&orchestrator, binding(), "resp_harmony-allowance");
    let turn = CLIENT_CALL.replace("commentary", "analysis");

    let (result, frames, stored) = streamed(&session, in_threes(&turn), 0).await;
    assert!(
        matches!(result, ResponsesIterationResult::ToolCallsFound { .. }),
        "the call is returned to the loop"
    );
    assert_eq!(opened(&frames), ["function_call"], "{frames:?}");
    assert_eq!(stored["output"][0]["type"], "function_call", "{stored}");
}

/// A worker may stop at a call's closing token without delivering it: the final
/// read then takes the call from the parser's open message, on any channel, as
/// the stream announced it.
#[tokio::test]
async fn a_call_cut_before_its_closing_token_is_a_call_on_any_channel() {
    let orchestrator = orchestrator().await;
    let session = McpToolSession::new(&orchestrator, binding(), "resp_harmony-allowance");
    for channel in ["commentary", "analysis"] {
        let turn = CLIENT_CALL.replace("commentary", channel);
        let cut = turn
            .strip_suffix("<|call|>")
            .expect("the call's closing token");

        let (result, frames, stored) = streamed(&session, in_threes(cut), 0).await;
        assert!(
            matches!(result, ResponsesIterationResult::ToolCallsFound { .. }),
            "{channel}: the call is returned to the loop"
        );
        assert_eq!(opened(&frames), ["function_call"], "{channel}: {frames:?}");
        let output = stored["output"].as_array().expect("the stored output");
        assert_eq!(output.len(), 1, "{channel}: {stored}");
        assert_eq!(output[0]["type"], "function_call", "{channel}: {stored}");
    }
}

/// The items the turn's `output_item.done` events closed.
fn closed(frames: &[String]) -> Vec<Value> {
    frames
        .iter()
        .filter_map(|frame| serde_json::from_str::<Value>(frame[frame.find('{')?..].trim()).ok())
        .filter(|event| event["type"] == "response.output_item.done")
        .map(|event| event["item"].clone())
        .collect()
}

/// A turn that stopped inside a server call leaves it un-run, and the stored
/// object reads the turn as cut whether or not the call had an item: one past
/// the turn's allowance opened none.
#[tokio::test]
async fn a_cut_server_call_reads_the_response_as_cut_with_or_without_its_item() {
    let orchestrator = orchestrator().await;
    let session = McpToolSession::new(&orchestrator, binding(), "resp_harmony-allowance");

    for allowance in [1, 0] {
        let (_, frames, stored) = streamed(&session, in_threes(CUT_SERVER_CALL), allowance).await;
        assert_eq!(
            opened(&frames).len(),
            allowance,
            "allowance {allowance}: {frames:?}"
        );
        assert_eq!(
            stored["status"], "incomplete",
            "allowance {allowance}: {stored}"
        );
        assert_eq!(
            stored["output"],
            json!([]),
            "allowance {allowance}: {stored}"
        );
    }
}

/// Whether `status` is one the status type of `item_type` defines.
fn defined(item_type: &str, status: &Value) -> bool {
    fn of<S: DeserializeOwned>(status: &Value) -> bool {
        serde_json::from_value::<S>(status.clone()).is_ok()
    }
    match item_type {
        "web_search_call" => of::<WebSearchCallStatus>(status),
        "code_interpreter_call" => of::<CodeInterpreterCallStatus>(status),
        "file_search_call" => of::<FileSearchCallStatus>(status),
        "image_generation_call" => of::<ImageGenerationCallStatus>(status),
        // An `mcp_call` item's status is a string.
        _ => status.is_string(),
    }
}

/// A server call the turn stopped inside closes its item in a status the
/// item's type defines, whatever the tool's format: `incomplete`, and
/// `failed` for an image generation call, whose type has no `incomplete`. No
/// event says the call completed, and the response reads the turn as cut.
#[tokio::test]
async fn a_cut_server_calls_item_closes_in_a_status_its_type_defines() {
    let orchestrator = orchestrator().await;
    let session = McpToolSession::new(&orchestrator, binding(), "resp_harmony-allowance");

    // Every format is read before any is judged, so one run shows them all.
    let mut misread = Vec::new();
    for (format, item_type, status) in [
        (ResponseFormatConfig::Passthrough, "mcp_call", "incomplete"),
        (
            ResponseFormatConfig::WebSearchCall,
            "web_search_call",
            "incomplete",
        ),
        (
            ResponseFormatConfig::CodeInterpreterCall,
            "code_interpreter_call",
            "incomplete",
        ),
        (
            ResponseFormatConfig::FileSearchCall,
            "file_search_call",
            "incomplete",
        ),
        (
            ResponseFormatConfig::ImageGenerationCall,
            "image_generation_call",
            "failed",
        ),
    ] {
        let registry = FormatRegistry::new();
        registry.populate_from_server_config(&McpServerConfig {
            name: "weather".to_string(),
            transport: McpTransport::Streamable {
                url: "http://weather.invalid".to_string(),
                token: None,
                headers: HashMap::new(),
            },
            proxy: None,
            required: false,
            tools: Some(HashMap::from([(
                "get_weather".to_string(),
                ToolConfig {
                    alias: None,
                    response_format: Some(format),
                    arg_mapping: None,
                },
            )])),
            builtin_type: None,
            builtin_tool_name: None,
            internal: false,
        });
        let (_, frames, stored) =
            streamed_as(&session, &registry, in_threes(CUT_SERVER_CALL), 1).await;
        let items = closed(&frames);
        let completed_event = format!("{item_type}.completed");
        let read = json!({
            "closed": items
                .iter()
                .map(|item| json!([item["type"], item["status"]]))
                .collect::<Vec<_>>(),
            "a status of the type": items.iter().all(|item| defined(item_type, &item["status"])),
            "completed events": frames
                .iter()
                .filter(|frame| frame.contains(&completed_event))
                .count(),
            "stored status": stored["status"],
            "stored output": stored["output"],
        });
        let due = json!({
            "closed": [[item_type, status]],
            "a status of the type": true,
            "completed events": 0,
            "stored status": "incomplete",
            "stored output": [],
        });
        if read != due {
            misread.push(format!("{item_type}: {read}, due {due}"));
        }
    }
    assert!(misread.is_empty(), "{}", misread.join("\n"));
}
