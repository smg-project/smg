//! MiniMax's wire contract for a tool call the reader could not read
//! (`wire::minimax::InvalidToolCallFormat`), on the chat stream and the whole
//! answer, with the MiniMax M3 table reading the output. A request that
//! expects tool calls fails on a call the token limit cut, on one the model
//! stopped inside, and on a block with malformed structure. A stream has sent
//! the answer as it was read by then, the call as far as the model wrote it,
//! exactly as a request the contract does not cover is sent it: the contract
//! changes how the stream ends, not what streams (at the model's own stop
//! the close the table would write for the model is left out). And the
//! contract's limits: a request without tools, another wire, and another
//! table on the same wire end as they did.

use std::{sync::Arc, time::Duration};

use axum::http::StatusCode;
use http_body_util::BodyExt;
use llm_tokenizer::StopSequenceConfig;
use openai_protocol::{
    chat::{ChatChoice, ChatCompletionRequest},
    profile::ProviderProfile,
};
use serde_json::{json, Value};
use smg_grpc_client::{tokenspeed_proto::GenerateComplete, vllm_engine::proto};
use smg_policy_controller::TransportClass;
use symphony::{formats::Table, grammar::GrammarStart};
use tokio::{sync::oneshot, time::timeout};

use super::{
    super::*,
    eof::{
        chunk, collect_events, complete, complete_with_prompt, gated_stream, placed_stream_context,
        processor, response_processor, scripted_stream, CharacterTokenizer,
    },
};
use crate::{
    rate_limit::{test_support::RecordingBackend, UsageSettlement},
    routers::{
        common::rate_limit_cell::{RateLimitCell, RateLimitOutcome},
        error::HEADER_X_SMG_ERROR_CODE,
        grpc::{
            context::test_support::assert_end,
            proto_wrapper::ProtoGenerateComplete,
            regular::{
                processor::{ChoiceFailure, ResponseProcessor},
                stages::chat::ChatResponseProcessingStage,
                symphony::{assign_for, Asks, Reading, UnreadableCall},
            },
            spec::ResponseSpec,
            utils::{StreamFailure, StreamSource},
            wire::minimax::InvalidToolCallFormat,
        },
    },
};

const BEFORE: &str = "Checking.";
/// The content of an answer whose call the limit cut: the text, then the
/// newline the invoke was holding when the cut came, which the table returns
/// as text, the contract or not.
const BEFORE_AND_THE_CUTS_NEWLINE: &str = "Checking.\n";
/// A MiniMax M3 call to `lookup`, as the template writes it, ended after its
/// one parameter.
const CUT_CALL: &str = "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"lookup\">\n]<]minimax[>[<city>Paris]<]minimax[>[</city>\n";
/// What closes [`CUT_CALL`].
const CLOSE: &str = "]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>";
/// A block whose structure is not the syntax's: text where a parameter's tag
/// should open, inside an invoke the model closed.
const MALFORMED_CALL: &str = "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"lookup\">\n]<]minimax[>[Paris\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>";
/// A MiniMax M2 call to `lookup`, ended after its one parameter.
const M2_CUT_CALL: &str =
    "<minimax:tool_call>\n<invoke name=\"lookup\">\n<parameter name=\"city\">Paris</parameter>\n";

/// A model on MiniMax's wire that the M3 table reads, one on MiniMax's wire
/// that the M2 table reads, and a served name on no vendor's wire.
const M3: &str = "MiniMax-M3";
const M2: &str = "MiniMax-M2";
const SERVED: &str = "served-model";

const CUT_BY_THE_LIMIT: &str = "{\"error\":{\"message\":\"Treat this response as failed. The model stopped due to max_tokens before it completed the tool call. Retry the request.\",\"type\":\"invalid_tool_call_format\",\"code\":\"invalid_tool_call_format\",\"finish_reason\":\"length\"}}";

/// The spec of a chat request for `model`, read by `table`: with the `lookup`
/// tool (a string parameter, `city`, and an integer, `days`) and the `other`
/// tool declared or none, asking for the usage or not, under `tool_choice` or none. The model
/// id selects the provider profile, and so the wire.
fn spec_of(
    model: &str,
    table: Table,
    tools: bool,
    usage: bool,
    tool_choice: Option<Value>,
) -> ChatResponseSpec {
    let mut request = json!({
        "model": model, "messages": [{"role": "user", "content": "hi"}], "stream": true,
        "separate_reasoning": true,
    });
    if tools {
        request["tools"] = json!([
            {"type": "function", "function": {
                "name": "lookup", "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"}, "days": {"type": "integer"}}}}},
            {"type": "function", "function": {
                "name": "other", "parameters": {"type": "object", "properties": {}}}},
        ]);
    }
    if usage {
        request["stream_options"] = json!({"include_usage": true});
    }
    if let Some(tool_choice) = tool_choice {
        request["tool_choice"] = tool_choice;
    }
    let request: ChatCompletionRequest = serde_json::from_value(request).expect("a chat request");
    let mut spec = ChatResponseSpec::from(&request);
    spec.symphony = assign_for(
        Reading::Table(table.shared(), false),
        Asks::Chat(&request),
        None,
        "",
        ProviderProfile::for_model(&request.model),
        || GrammarStart::Unknown,
    );
    assert!(spec.symphony.is_some(), "a table reads");
    spec
}

/// A MiniMax M3 request, with the `lookup` tool declared or none, asking for
/// the usage or not.
fn spec(tools: bool, usage: bool) -> ChatResponseSpec {
    spec_of(M3, Table::MinimaxM3, tools, usage, None)
}

fn dispatch(model: &str) -> context::DispatchMetadata {
    context::DispatchMetadata {
        request_id: "chatcmpl-unreadable".to_string(),
        model: model.to_string(),
        created: 1,
        weight_version: None,
    }
}

/// `output` one character a frame, then the engine's `finish`.
fn by_character(output: &str, finish: &str) -> Vec<proto::GenerateResponse> {
    let mut frames: Vec<_> = output.chars().map(|ch| chunk(0, &ch.to_string())).collect();
    frames.push(complete(0, finish));
    frames
}

/// The read of the stream of `frames` under `spec`: how it ended, and its
/// `data:` events.
async fn streamed(
    frames: Vec<proto::GenerateResponse>,
    spec: ChatResponseSpec,
) -> (Result<(), StreamFailure>, Vec<Value>) {
    let (stream, server) = scripted_stream(frames, "0").await;
    let (tx, rx) = sse_channel();
    let result = processor(true)
        .process_streaming_chunks(
            stream,
            dispatch(M3),
            Arc::new(CharacterTokenizer::default()),
            (None, None, false, false, false),
            spec,
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    (result, events)
}

fn finish_reasons(events: &[Value]) -> Vec<String> {
    events
        .iter()
        .flat_map(|event| event["choices"].as_array().into_iter().flatten())
        .filter_map(|choice| choice["finish_reason"].as_str())
        .map(str::to_owned)
        .collect()
}

fn content(events: &[Value]) -> String {
    events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
        .collect()
}

/// The tool-call deltas of choice 0, in order.
fn call_deltas(events: &[Value]) -> Vec<&Value> {
    events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["tool_calls"].as_array())
        .flatten()
        .collect()
}

fn tool_names(events: &[Value]) -> Vec<String> {
    call_deltas(events)
        .into_iter()
        .filter_map(|delta| delta["function"]["name"].as_str())
        .map(str::to_owned)
        .collect()
}

/// The arguments sent for the stream's calls, their fragments joined.
fn tool_arguments(events: &[Value]) -> String {
    call_deltas(events)
        .into_iter()
        .filter_map(|delta| delta["function"]["arguments"].as_str())
        .collect()
}

/// The `lookup` call of [`CUT_CALL`] went out as far as the model wrote it:
/// its name, and its arguments without their close.
fn assert_the_call_went_out_cut(events: &[Value]) {
    assert_eq!(tool_names(events), vec!["lookup"], "{events:?}");
    let arguments = tool_arguments(events);
    assert!(
        arguments.contains("Paris") && serde_json::from_str::<Value>(&arguments).is_err(),
        "the arguments without their close: {arguments:?}"
    );
}

fn the_contracts_failure(result: Result<(), StreamFailure>) -> InvalidToolCallFormat {
    match result {
        Err(StreamFailure::ToolCallFormat(failure)) => failure,
        Ok(()) => panic!("the stream ended well"),
        Err(other) => panic!("another failure: {other}"),
    }
}

/// What the client was sent of the answer, chunk by chunk: each chunk's
/// `choices`, without the finish chunks and the usage chunk, a call's id left
/// out (every output draws its ids afresh).
fn deliveries(events: &[Value]) -> Vec<Value> {
    events
        .iter()
        .filter_map(|event| event["choices"].as_array())
        .filter(|choices| {
            !choices.is_empty()
                && choices
                    .iter()
                    .all(|choice| choice["finish_reason"].is_null())
        })
        .map(|choices| {
            let mut choices = choices.clone();
            let calls = choices
                .iter_mut()
                .filter_map(|choice| choice.get_mut("delta"))
                .filter_map(|delta| delta.get_mut("tool_calls"))
                .filter_map(Value::as_array_mut)
                .flatten();
            for call in calls {
                if let Some(call) = call.as_object_mut() {
                    call.remove("id");
                }
            }
            Value::Array(choices)
        })
        .collect()
}

/// The events of the stream of `frames` where the contract does not apply,
/// under the same tools and `tool_choice`: the M3 table reading for a served
/// name on no vendor's wire, which ends with the engine's finish.
async fn without_the_contract(
    frames: Vec<proto::GenerateResponse>,
    tool_choice: Option<Value>,
) -> Vec<Value> {
    let served = spec_of(SERVED, Table::MinimaxM3, true, false, tool_choice);
    assert!(!served.fails_on_unreadable_call());
    let (result, events) = streamed(frames, served).await;
    assert!(result.is_ok(), "{:?}", result.err().map(|e| e.to_string()));
    assert_eq!(finish_reasons(&events).len(), 1, "{events:?}");
    events
}

/// A call the token limit cut, under a request that declares tools: the text
/// before it went out and so did the call, as far as it was read (its name,
/// its arguments without their close), chunk for chunk what a request the
/// contract does not cover is sent; no finish chunk follows, the usage chunk
/// does when the request asked for one, and the stream's outcome is the
/// contract's failure with the `length` finish. One frame for the call, or a
/// character a frame: the same.
#[tokio::test]
async fn a_call_the_limit_cut_ends_the_stream_with_the_usage_frame_and_no_finish() {
    for usage in [true, false] {
        for frames in [
            vec![chunk(0, BEFORE), chunk(0, CUT_CALL), complete(0, "length")],
            by_character(&format!("{BEFORE}{CUT_CALL}"), "length"),
        ] {
            let (result, events) = streamed(frames.clone(), spec(true, usage)).await;
            let failure = the_contracts_failure(result);
            assert_eq!(
                failure,
                InvalidToolCallFormat::new(UnreadableCall::Incomplete, true)
            );
            assert_eq!(failure.finish_reason(), "length");
            assert_eq!(content(&events), BEFORE_AND_THE_CUTS_NEWLINE, "{events:?}");
            assert!(finish_reasons(&events).is_empty(), "{events:?}");
            assert_the_call_went_out_cut(&events);
            assert_eq!(
                deliveries(&events),
                deliveries(&without_the_contract(frames, None).await)
            );
            let last = events.last().expect("the text went out");
            if usage {
                assert!(
                    last["usage"].is_object()
                        && last["choices"].as_array().is_some_and(Vec::is_empty),
                    "the usage chunk last: {last}"
                );
            } else {
                assert!(
                    events.iter().all(|event| !event["usage"].is_object()),
                    "{events:?}"
                );
            }
        }
    }
}

/// The model's own stop inside a call is the `error` finish with the
/// incomplete-call sentence: the token limit is blamed only for a cut. The
/// call went out as far as the model wrote it, its arguments without their
/// close: where the contract does not apply the table closes them for the
/// model at its stop, and that one fragment is what the failing stream leaves
/// out. Everything else is sent chunk for chunk the same.
#[tokio::test]
async fn the_models_own_stop_inside_a_call_is_an_error_finish() {
    for frames in [
        vec![chunk(0, CUT_CALL), complete(0, "stop")],
        by_character(CUT_CALL, "stop"),
    ] {
        let (result, events) = streamed(frames.clone(), spec(true, false)).await;
        let failure = the_contracts_failure(result);
        assert_eq!(
            failure,
            InvalidToolCallFormat::new(UnreadableCall::Incomplete, false)
        );
        assert_eq!(failure.finish_reason(), "error");
        assert_eq!(
            failure.message(),
            "Treat this response as failed. The model generated an incomplete tool call. Retry the request."
        );
        assert!(finish_reasons(&events).is_empty(), "{events:?}");
        assert_the_call_went_out_cut(&events);
        assert_eq!(
            tool_arguments(&events),
            "{\"city\": \"Paris\"",
            "{events:?}"
        );

        let as_before = without_the_contract(frames, None).await;
        assert_eq!(finish_reasons(&as_before), vec!["tool_calls"]);
        assert_eq!(
            tool_arguments(&as_before),
            "{\"city\": \"Paris\"}",
            "closed for the model: {as_before:?}"
        );
        let (closing, before_it) = deliveries(&as_before)
            .split_last()
            .map(|(last, rest)| (last.clone(), rest.to_vec()))
            .expect("the answer's chunks");
        assert_eq!(
            closing[0]["delta"]["tool_calls"][0]["function"]["arguments"], "}",
            "{as_before:?}"
        );
        assert_eq!(deliveries(&events), before_it);
    }
}

/// A value the model's stop found open went out as far as the model wrote
/// it: a string that streams has its text in the arguments, and neither its
/// closing quote nor the arguments' brace follows, on the stream and in the
/// body the client reads.
#[tokio::test]
async fn a_value_the_models_stop_found_open_goes_out_without_a_close() {
    const OPEN: &str =
        "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"lookup\">\n]<]minimax[>[<city>Paris, Fr";
    let frames = vec![chunk(0, BEFORE), chunk(0, OPEN), complete(0, "stop")];
    let (result, events) = streamed(frames.clone(), spec(true, false)).await;
    assert_eq!(
        the_contracts_failure(result),
        InvalidToolCallFormat::new(UnreadableCall::Incomplete, false)
    );
    assert_eq!(tool_names(&events), vec!["lookup"], "{events:?}");
    assert_eq!(
        tool_arguments(&events),
        "{\"city\": \"Paris, Fr",
        "{events:?}"
    );
    let as_before = without_the_contract(frames.clone(), None).await;
    assert_eq!(tool_arguments(&as_before), "{\"city\": \"Paris, Fr\"}");

    let body = response_body(frames, spec(true, true)).await;
    let (frames, events) = frames_and_events(&body);
    assert_eq!(tool_arguments(&events), "{\"city\": \"Paris, Fr", "{body}");
    let contract = InvalidToolCallFormat::new(UnreadableCall::Incomplete, false).sse_frame();
    assert_eq!(
        frames.last().copied(),
        std::str::from_utf8(&contract).ok().map(str::trim_end),
        "{body}"
    );
}

/// A block with malformed structure is an `error` finish whatever the
/// engine's: the invoke opened with its name, the reader met text where a
/// parameter's tag goes, and the model closed the invoke. The call's name had
/// gone out when the reader met the text; what follows it is what the table
/// delivers of such a block where the contract does not apply, and the
/// failure ends the stream.
#[tokio::test]
async fn a_block_with_malformed_structure_is_an_error_finish() {
    for finish in ["length", "stop"] {
        for frames in [
            vec![
                chunk(0, BEFORE),
                chunk(0, MALFORMED_CALL),
                complete(0, finish),
            ],
            by_character(&format!("{BEFORE}{MALFORMED_CALL}"), finish),
        ] {
            let (result, events) = streamed(frames.clone(), spec(true, false)).await;
            let failure = the_contracts_failure(result);
            assert_eq!(
                failure,
                InvalidToolCallFormat::new(UnreadableCall::Malformed, finish == "length"),
                "{finish}"
            );
            assert_eq!(failure.finish_reason(), "error", "{finish}");
            assert_eq!(
                failure.message(),
                "Treat this response as failed. The model generated a tool call with malformed structure. Retry the request."
            );
            assert!(finish_reasons(&events).is_empty(), "{events:?}");
            assert_eq!(tool_names(&events), vec!["lookup"], "{events:?}");
            assert_eq!(
                deliveries(&events),
                deliveries(&without_the_contract(frames, None).await)
            );
        }
    }
}

/// The output goes on after a block with malformed structure: the text the
/// model wrote after it streams as it does where the contract does not
/// apply, a whole call after it goes out too, and the failure waits for the
/// output's end, where it stands in place of the finish.
#[tokio::test]
async fn the_output_after_a_malformed_block_streams_and_the_failure_ends_it() {
    const AFTER: &str = "Done.";
    for (rest, names) in [
        (AFTER.to_string(), vec!["lookup"]),
        (format!("{CUT_CALL}{CLOSE}"), vec!["lookup", "lookup"]),
    ] {
        let output = format!("{BEFORE}{MALFORMED_CALL}{rest}");
        let frames = by_character(&output, "stop");
        let (result, events) = streamed(frames.clone(), spec(true, true)).await;
        assert_eq!(
            the_contracts_failure(result),
            InvalidToolCallFormat::new(UnreadableCall::Malformed, false)
        );
        assert!(finish_reasons(&events).is_empty(), "{events:?}");
        assert_eq!(tool_names(&events), names, "{events:?}");
        let as_before = without_the_contract(frames, None).await;
        assert_eq!(deliveries(&events), deliveries(&as_before));
        assert!(
            events.last().is_some_and(|last| last["usage"].is_object()),
            "{events:?}"
        );
    }
}

/// One block with one `lookup` invoke whose elements are `body`, the
/// template's separator before every tag.
fn lookup_block(body: &str) -> String {
    format!("<tool_call>\n<invoke name=\"lookup\">{body}</invoke>\n</tool_call>")
        .replace('<', "]<]minimax[>[<")
}

/// What the M3 table reads past a slip of the model's is a whole call, and is
/// delivered under the contract as outside it: a closing tag that names
/// nothing open (a value's close misnamed, a second close, one between two
/// invokes), and an element left open where the element around it, the
/// invoke, the block or the next invoke ends. The stream ends with the
/// `tool_calls` finish, the arguments read as JSON, and the whole answer is
/// the same call.
#[tokio::test]
async fn a_call_the_table_reads_past_a_slip_is_delivered() {
    let paris = json!({"city": "Paris"});
    for (output, arguments) in [
        (lookup_block("<city>Paris</town>"), paris.clone()),
        (lookup_block("<city>Paris</town></city>"), paris.clone()),
        (lookup_block("<city>Paris</city></town>"), paris.clone()),
        (lookup_block("<days>3</day>"), json!({"days": 3})),
        (lookup_block("<days>3"), json!({"days": 3})),
        (lookup_block("<city>Paris"), paris.clone()),
        (
            lookup_block("<where><city>Paris</city><zip></where>"),
            json!({"where": {"city": "Paris", "zip": ""}}),
        ),
        (
            lookup_block("<where><city>Paris</where><days>3</days>"),
            json!({"where": {"city": "Paris"}, "days": 3}),
        ),
        (
            "<tool_call>\n<invoke name=\"lookup\"><days>3</tool_call>".replace('<', "]<]minimax[>[<"),
            json!({"days": 3}),
        ),
        (
            "<tool_call>\n<invoke name=\"lookup\"><city>Paris</city></invoke>\n</invoke>\n</tool_call>"
                .replace('<', "]<]minimax[>[<"),
            paris.clone(),
        ),
    ] {
        for frames in [
            vec![chunk(0, &output), complete(0, "stop")],
            by_character(&output, "stop"),
        ] {
            let covered = spec(true, false);
            assert!(covered.fails_on_unreadable_call());
            let (result, events) = streamed(frames.clone(), covered).await;
            assert!(
                result.is_ok(),
                "{output}: {:?}",
                result.err().map(|e| e.to_string())
            );
            assert_eq!(finish_reasons(&events), vec!["tool_calls"], "{output}");
            assert_eq!(tool_names(&events), vec!["lookup"], "{output}");
            assert_eq!(
                serde_json::from_str::<Value>(&tool_arguments(&events)).expect("whole arguments"),
                arguments,
                "{output}"
            );
            assert_eq!(content(&events), "", "{output}");
            assert_eq!(
                deliveries(&events),
                deliveries(&without_the_contract(frames, None).await),
                "{output}"
            );
        }
        let choice = whole_answer(M3, &output, "stop", &spec(true, false))
            .await
            .unwrap_or_else(|failure| panic!("{output}: {failure}"));
        let calls = choice.message.tool_calls.expect("the call");
        assert_eq!(calls.len(), 1, "{output}");
        assert_eq!(
            serde_json::from_str::<Value>(calls[0].function.arguments.as_deref().expect("arguments"))
                .expect("whole arguments"),
            arguments,
            "{output}"
        );
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    }
}

/// What the table cannot read past still fails the request. The end of the
/// output closes nothing for the model: the element the invoke's closing tag
/// would have closed is an incomplete call where the output ends in it, at
/// the model's stop and at the limit. Text between a call's elements, and a
/// tag cut short there, are malformed structure.
#[tokio::test]
async fn what_the_table_cannot_read_past_still_fails_the_request() {
    let open = "<tool_call>\n<invoke name=\"lookup\"><days>3".replace('<', "]<]minimax[>[<");
    for (finish, at_limit) in [("stop", false), ("length", true)] {
        let failure = InvalidToolCallFormat::new(UnreadableCall::Incomplete, at_limit);
        let frames = vec![chunk(0, &open), complete(0, finish)];
        let (result, events) = streamed(frames, spec(true, false)).await;
        assert_eq!(the_contracts_failure(result), failure, "{finish}");
        assert!(finish_reasons(&events).is_empty(), "{events:?}");
        let read = whole_answer(M3, &open, finish, &spec(true, false)).await;
        let Err(ChoiceFailure::ToolCallFormat(whole)) = read else {
            panic!("the contract's failure under {finish}");
        };
        assert_eq!(whole, failure, "{finish}");
    }
    for body in [
        "<city>Paris</city> and <days>3</days>",
        "<city>Paris</city><da",
        "<days>3<b</day>",
    ] {
        let output = lookup_block(body);
        let failure = InvalidToolCallFormat::new(UnreadableCall::Malformed, false);
        let frames = by_character(&output, "stop");
        let (result, events) = streamed(frames, spec(true, false)).await;
        assert_eq!(the_contracts_failure(result), failure, "{body}");
        assert!(finish_reasons(&events).is_empty(), "{events:?}");
        let read = whole_answer(M3, &output, "stop", &spec(true, false)).await;
        let Err(ChoiceFailure::ToolCallFormat(whole)) = read else {
            panic!("the contract's failure for {body}");
        };
        assert_eq!(whole, failure, "{body}");
    }
}

/// An invoke the output ended inside before its name closed starts no call:
/// the block is the one the output ended in, so the failure is the
/// incomplete one, blamed on the limit where the limit cut it. No call delta
/// went out, and the invoke's bytes went out as content before the failure,
/// as they do where the contract does not apply.
#[tokio::test]
async fn a_block_the_output_ended_inside_before_its_call_started_is_incomplete() {
    const OPENED: &str = "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"look";
    for (finish, at_limit) in [("length", true), ("stop", false)] {
        let frames = vec![chunk(0, BEFORE), chunk(0, OPENED), complete(0, finish)];
        let (result, events) = streamed(frames.clone(), spec(true, false)).await;
        assert_eq!(
            the_contracts_failure(result),
            InvalidToolCallFormat::new(UnreadableCall::Incomplete, at_limit),
            "{finish}"
        );
        assert_eq!(
            content(&events),
            format!("{BEFORE}<invoke name=\"look"),
            "{events:?}"
        );
        assert!(call_deltas(&events).is_empty(), "{events:?}");
        assert!(finish_reasons(&events).is_empty(), "{events:?}");
        assert_eq!(
            deliveries(&events),
            deliveries(&without_the_contract(frames, None).await)
        );
    }
}

/// An invoke the output ended in right after its opener, before any byte of
/// its name, is a call the model began: the request fails as for the invoke
/// one byte further, incomplete, blamed on the limit where the limit cut it.
/// On another wire the same output ends as it did, with the engine's finish
/// and no call, and what went out is the same.
#[tokio::test]
async fn an_invoke_the_output_ended_in_before_any_byte_of_its_name_is_incomplete() {
    const OPENED: &str = "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"";
    for (finish, at_limit) in [("length", true), ("stop", false)] {
        for frames in [
            vec![chunk(0, BEFORE), chunk(0, OPENED), complete(0, finish)],
            by_character(&format!("{BEFORE}{OPENED}"), finish),
        ] {
            let (result, events) = streamed(frames.clone(), spec(true, false)).await;
            assert_eq!(
                the_contracts_failure(result),
                InvalidToolCallFormat::new(UnreadableCall::Incomplete, at_limit),
                "{finish}"
            );
            assert_eq!(content(&events), BEFORE, "{events:?}");
            assert!(finish_reasons(&events).is_empty(), "{events:?}");
            assert!(call_deltas(&events).is_empty(), "{events:?}");

            let as_before = without_the_contract(frames, None).await;
            assert_eq!(content(&as_before), BEFORE, "{as_before:?}");
            assert_eq!(finish_reasons(&as_before), vec![finish]);
            assert!(call_deltas(&as_before).is_empty(), "{as_before:?}");
            assert_eq!(deliveries(&events), deliveries(&as_before));
        }
        let read = whole_answer(M3, &format!("{BEFORE}{OPENED}"), finish, &spec(true, false)).await;
        let Err(ChoiceFailure::ToolCallFormat(failure)) = read else {
            panic!("the contract's failure under {finish}");
        };
        assert_eq!(
            failure,
            InvalidToolCallFormat::new(UnreadableCall::Incomplete, at_limit)
        );
    }
}

/// The spec of a request for `model` that names `lookup`, whose one parameter
/// schema is a bare integer, read as the gateway's JSON schema makes the M3
/// table read a forced call: the output is the call's arguments alone, from
/// its first byte, with no marker to close them.
fn forced_spec_of(model: &str) -> ChatResponseSpec {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": model, "messages": [{"role": "user", "content": "hi"}], "stream": true,
        "tools": [{"type": "function", "function": {
            "name": "lookup", "parameters": {"type": "integer"}}}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
    }))
    .expect("a chat request");
    let schema = (
        "json_schema".to_string(),
        "{\"type\":\"integer\"}".to_string(),
    );
    let mut spec = ChatResponseSpec::from(&request);
    spec.symphony = assign_for(
        Reading::Table(Table::MinimaxM3.shared(), false),
        Asks::Chat(&request),
        Some(&schema),
        "",
        ProviderProfile::for_model(&request.model),
        || GrammarStart::FirstToken,
    );
    assert!(spec.symphony.is_some(), "a table reads");
    spec
}

/// The forced request on MiniMax's wire, which the contract covers.
fn forced_spec() -> ChatResponseSpec {
    let spec = forced_spec_of(M3);
    assert!(spec.fails_on_unreadable_call(), "the M3 table reads");
    spec
}

/// A forced call whose arguments are a bare scalar has no byte to close it:
/// the value is whole where the output ends, and the call is delivered, at
/// the model's stop for a number and a literal, streamed and whole. Where the
/// limit ended the output a literal is whole all the same, and a number is
/// not: `12` may be what the limit left of `123`, so the request fails as for
/// a call the limit cut, the digits sent as far as they went. A forced value
/// the output ended inside is the incomplete call, as any other: the request
/// fails, and the stream has sent the call's name and the value as far as it
/// went, as it does where the contract does not apply.
#[tokio::test]
async fn a_forced_scalar_whole_where_the_output_ends_is_delivered_and_an_unfinished_value_fails() {
    for (whole, finish) in [
        ("7", "stop"),
        ("-12", "stop"),
        ("true", "stop"),
        ("true", "length"),
        ("null", "length"),
    ] {
        for frames in [
            vec![chunk(0, whole), complete(0, finish)],
            by_character(whole, finish),
        ] {
            let (result, events) = streamed(frames, forced_spec()).await;
            assert!(
                result.is_ok(),
                "{whole} under {finish}: {:?}",
                result.err().map(|e| e.to_string())
            );
            assert_eq!(tool_names(&events), vec!["lookup"], "{events:?}");
            assert_eq!(tool_arguments(&events), whole, "{events:?}");
        }
        let choice = whole_answer(M3, whole, finish, &forced_spec())
            .await
            .unwrap_or_else(|failure| panic!("{whole} under {finish}: {failure}"));
        let calls = choice.message.tool_calls.expect("the call");
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(calls[0].function.arguments.as_deref(), Some(whole));
    }
    for (finish, at_limit, values) in [
        ("stop", false, &["{\"city\": \"Par", "[1, 2", "tru"][..]),
        (
            "length",
            true,
            &["{\"city\": \"Par", "[1, 2", "tru", "7", "-12"][..],
        ),
    ] {
        for unfinished in values {
            let frames = vec![chunk(0, unfinished), complete(0, finish)];
            let (result, events) = streamed(frames.clone(), forced_spec()).await;
            assert_eq!(
                the_contracts_failure(result),
                InvalidToolCallFormat::new(UnreadableCall::Incomplete, at_limit),
                "{unfinished} under {finish}"
            );
            assert!(finish_reasons(&events).is_empty(), "{events:?}");
            assert_eq!(tool_names(&events), vec!["lookup"], "{events:?}");
            assert_eq!(tool_arguments(&events), *unfinished, "{events:?}");
            let served = forced_spec_of(SERVED);
            assert!(!served.fails_on_unreadable_call());
            let (result, as_before) = streamed(frames, served).await;
            assert!(result.is_ok(), "{:?}", result.err().map(|e| e.to_string()));
            assert_eq!(deliveries(&events), deliveries(&as_before));
            let read = whole_answer(M3, unfinished, finish, &forced_spec()).await;
            let Err(ChoiceFailure::ToolCallFormat(failure)) = read else {
                panic!("the contract's failure for {unfinished} under {finish}");
            };
            assert_eq!(
                failure,
                InvalidToolCallFormat::new(UnreadableCall::Incomplete, at_limit)
            );
        }
    }
}

/// Without a tool inventory the contract does not apply and the stream ends
/// as it did: the engine's finish, the text kept, and the cut call out as far
/// as the model wrote it. `tool_choice: none` over declared tools switches
/// the contract off as well: the call is dropped whole, as before.
#[tokio::test]
async fn without_tools_the_cut_call_ends_as_before() {
    assert!(!spec(false, false).fails_on_unreadable_call());
    let frames = vec![chunk(0, BEFORE), chunk(0, CUT_CALL), complete(0, "length")];
    let (result, events) = streamed(frames.clone(), spec(false, false)).await;
    assert!(result.is_ok(), "{:?}", result.err().map(|e| e.to_string()));
    assert_eq!(content(&events), BEFORE_AND_THE_CUTS_NEWLINE, "{events:?}");
    assert_eq!(finish_reasons(&events), vec!["length"]);
    assert_the_call_went_out_cut(&events);

    let none = spec_of(M3, Table::MinimaxM3, true, false, Some(json!("none")));
    assert!(!none.fails_on_unreadable_call());
    let (result, events) = streamed(frames, none).await;
    assert!(result.is_ok(), "{:?}", result.err().map(|e| e.to_string()));
    assert_eq!(content(&events), BEFORE, "{events:?}");
    assert_eq!(finish_reasons(&events), vec!["length"]);
    assert!(call_deltas(&events).is_empty(), "{events:?}");
}

/// A whole call is delivered as before: what a request the contract covers
/// is sent is what one it does not cover is sent for the same output, chunk
/// for chunk, the call's id aside, then the `tool_calls` finish.
#[tokio::test]
async fn a_whole_call_is_delivered_as_before() {
    for frames in [
        vec![
            chunk(0, BEFORE),
            chunk(0, CUT_CALL),
            chunk(0, CLOSE),
            complete(0, "stop"),
        ],
        by_character(&format!("{BEFORE}{CUT_CALL}{CLOSE}"), "stop"),
    ] {
        let covered = spec(true, false);
        assert!(covered.fails_on_unreadable_call());
        let (result, events) = streamed(frames.clone(), covered).await;
        assert!(result.is_ok(), "{:?}", result.err().map(|e| e.to_string()));
        assert_eq!(content(&events), BEFORE, "{events:?}");
        assert_eq!(tool_names(&events), vec!["lookup"]);
        assert_eq!(
            serde_json::from_str::<Value>(&tool_arguments(&events)).expect("the arguments, closed"),
            json!({"city": "Paris"}),
            "{events:?}"
        );
        assert_eq!(finish_reasons(&events), vec!["tool_calls"]);

        let as_before = without_the_contract(frames, None).await;
        assert_eq!(deliveries(&events), deliveries(&as_before));
        assert_eq!(finish_reasons(&as_before), vec!["tool_calls"]);
    }
}

/// A call before the one that failed was whole and went out; the one the
/// limit cut after it went out as far as it was read, and the stream ends as
/// the contract's failure.
#[tokio::test]
async fn a_whole_call_before_the_failed_one_went_out_and_the_failed_one_as_far_as_it_was_read() {
    let second = "]<]minimax[>[<invoke name=\"lookup\">\n]<]minimax[>[<city>Rome";
    let output = format!("{CUT_CALL}]<]minimax[>[</invoke>\n{second}");
    let frames = by_character(&output, "length");
    let (result, events) = streamed(frames.clone(), spec(true, false)).await;
    assert_eq!(
        the_contracts_failure(result),
        InvalidToolCallFormat::new(UnreadableCall::Incomplete, true)
    );
    assert_eq!(tool_names(&events), vec!["lookup", "lookup"], "{events:?}");
    assert_eq!(
        tool_arguments(&events),
        "{\"city\": \"Paris\"}{\"city\": \"Rome",
        "{events:?}"
    );
    assert!(finish_reasons(&events).is_empty(), "{events:?}");
    assert_eq!(
        deliveries(&events),
        deliveries(&without_the_contract(frames, None).await)
    );
}

/// A call the request's own choice drops is a call the model failed to write
/// all the same: under a choice that names another function, the cut call to
/// `lookup` fails the request, and it is dropped as the choice drops it where
/// the contract does not apply. Whole, the same call is dropped as before
/// and the stream ends with the engine's finish.
#[tokio::test]
async fn a_cut_call_the_choice_forbids_fails_the_request_all_the_same() {
    let other = json!({"type": "function", "function": {"name": "other"}});
    let named = || spec_of(M3, Table::MinimaxM3, true, false, Some(other.clone()));
    let frames = vec![chunk(0, CUT_CALL), complete(0, "length")];
    let (result, events) = streamed(frames.clone(), named()).await;
    assert_eq!(
        the_contracts_failure(result),
        InvalidToolCallFormat::new(UnreadableCall::Incomplete, true)
    );
    assert!(call_deltas(&events).is_empty(), "{events:?}");
    assert_eq!(
        deliveries(&events),
        deliveries(&without_the_contract(frames, Some(other.clone())).await)
    );

    let frames = vec![chunk(0, CUT_CALL), chunk(0, CLOSE), complete(0, "stop")];
    let (result, events) = streamed(frames, named()).await;
    assert!(result.is_ok(), "{:?}", result.err().map(|e| e.to_string()));
    assert!(call_deltas(&events).is_empty(), "{events:?}");
    assert_eq!(finish_reasons(&events), vec!["stop"]);
}

/// One failing choice fails the whole request: of two choices, the first
/// whole and the second cut, the stream's outcome is the contract's failure
/// and no choice gets a finish chunk. Both choices' calls went out as they
/// were read, the second's without its close.
#[tokio::test]
async fn one_failing_choice_fails_the_whole_stream() {
    let two = |model: &str| {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": model, "messages": [{"role": "user", "content": "hi"}], "stream": true,
            "n": 2, "separate_reasoning": true,
            "tools": [{"type": "function", "function": {
                "name": "lookup", "parameters": {"type": "object", "properties": {}}}}],
        }))
        .expect("a chat request");
        let mut two = ChatResponseSpec::from(&request);
        two.symphony = assign_for(
            Reading::Table(Table::MinimaxM3.shared(), false),
            Asks::Chat(&request),
            None,
            "",
            ProviderProfile::for_model(&request.model),
            || GrammarStart::Unknown,
        );
        two
    };
    let frames = vec![
        chunk(0, CUT_CALL),
        chunk(1, CUT_CALL),
        chunk(0, CLOSE),
        complete(0, "stop"),
        complete(1, "length"),
    ];
    let (result, events) = streamed(frames.clone(), two(M3)).await;
    assert_eq!(
        the_contracts_failure(result),
        InvalidToolCallFormat::new(UnreadableCall::Incomplete, true)
    );
    assert!(finish_reasons(&events).is_empty(), "{events:?}");
    let arguments_of = |index: u64| -> String {
        events
            .iter()
            .flat_map(|event| event["choices"].as_array().into_iter().flatten())
            .filter(|choice| choice["index"] == index)
            .filter_map(|choice| choice["delta"]["tool_calls"][0]["function"]["arguments"].as_str())
            .collect()
    };
    assert_eq!(arguments_of(0), "{\"city\": \"Paris\"}", "{events:?}");
    assert_eq!(arguments_of(1), "{\"city\": \"Paris\"", "{events:?}");

    let (result, as_before) = streamed(frames, two(SERVED)).await;
    assert!(result.is_ok(), "{:?}", result.err().map(|e| e.to_string()));
    assert_eq!(finish_reasons(&as_before), vec!["tool_calls", "length"]);
    assert_eq!(deliveries(&events), deliveries(&as_before));
}

/// The stream's end: `[DONE]` after a read that ended well and after any
/// other failure's event; the contract's event is the last frame, and the
/// body is told the reading failed either way a failure ended it.
#[tokio::test]
async fn the_error_event_is_the_streams_last_frame_and_no_done_follows() {
    let source = StreamSource::new(None, &dispatch(M3), metrics_labels::ENDPOINT_CHAT);
    let contract =
        StreamFailure::ToolCallFormat(InvalidToolCallFormat::new(UnreadableCall::Incomplete, true));
    for (result, expected, end) in [
        (
            Ok(()),
            vec!["data: [DONE]\n\n".to_string()],
            ReadingEnd::Complete,
        ),
        (
            Err(contract),
            vec![format!("event: error\ndata: {CUT_BY_THE_LIMIT}\n\n")],
            ReadingEnd::Failed,
        ),
        (
            Err(StreamFailure::Internal("the gateway's own".to_string())),
            vec![
                "data: {\"error\":{\"message\":\"the gateway's own\",\"type\":\"internal_error\"}}\n\n"
                    .to_string(),
                "data: [DONE]\n\n".to_string(),
            ],
            ReadingEnd::Failed,
        ),
    ] {
        let (tx, mut rx) = sse_channel();
        let ended =
            StreamingProcessor::close_chat_stream(&tx, &result, &source, Span::current()).await;
        drop(tx);
        let mut frames = Vec::new();
        while let Some(frame) = rx.recv().await {
            frames.push(String::from_utf8(frame.expect("a frame").to_vec()).expect("UTF-8"));
        }
        assert_eq!(frames, expected);
        assert_eq!(ended, end);
    }
}

/// The engine completed the answer the request fails on, so its dispatch
/// ended whole: through the chat stage, a placed dispatch whose stream ends
/// as the contract's failure is told a 200 with the answer's token counts,
/// as for an answer that went out, and the body ends with the contract's
/// event.
#[tokio::test]
async fn a_placed_dispatch_is_told_the_answer_ended_whole() {
    let output = format!("{BEFORE}{CUT_CALL}");
    let frames = vec![chunk(0, &output), complete_with_prompt(0, "length", 11)];
    let (stream, server) = scripted_stream(frames, "0").await;
    let (mut ctx, store) = placed_stream_context(stream, dispatch(M3)).await;
    let placement = ctx
        .load_guards
        .as_ref()
        .and_then(context::LoadGuards::placement)
        .expect("the placement")
        .clone();
    let response =
        ChatResponseProcessingStage::new(response_processor(), Arc::new(processor(true)))
            .process_result(&mut ctx, ResponseSpec::Chat(Box::new(spec(true, false))))
            .await
            .expect("the stage succeeded")
            .expect("a streaming response");
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("the stream's body");
    server.abort();
    assert!(
        body.ends_with(format!("event: error\ndata: {CUT_BY_THE_LIMIT}\n\n").as_bytes()),
        "{body:?}"
    );
    assert_end(
        &placement.ends(),
        &store,
        200,
        TransportClass::Ok,
        11,
        output.chars().count() as u32,
    );
}

/// The body of the streamed response to `spec` over an engine stream, as the
/// client reads it: the re-slicing of MiniMax's delta-size contract included.
async fn response_of(stream: ProtoStream, spec: ChatResponseSpec) -> axum::body::Body {
    let dispatch = dispatch(M3);
    let source = StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_CHAT);
    let response = Arc::new(processor(true))
        .process_streaming_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: Instant::now(),
            },
            spec,
            dispatch,
            Arc::new(CharacterTokenizer::default()),
            false,
            None,
            None,
            source,
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    response.into_body()
}

/// The whole body of the streamed response to `spec` over `frames`.
async fn response_body(frames: Vec<proto::GenerateResponse>, spec: ChatResponseSpec) -> String {
    let (stream, server) = scripted_stream(frames, "0").await;
    let body = response_of(stream, spec)
        .await
        .collect()
        .await
        .expect("the body")
        .to_bytes();
    server.abort();
    String::from_utf8(body.to_vec()).expect("UTF-8")
}

/// The frames of a body, and those of them that are JSON `data:` events.
fn frames_and_events(body: &str) -> (Vec<&str>, Vec<Value>) {
    let frames: Vec<&str> = body.split("\n\n").filter(|f| !f.is_empty()).collect();
    let events = frames
        .iter()
        .filter_map(|frame| frame.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect();
    (frames, events)
}

/// A failed call went out as far as it was read, read off the bytes of the
/// response's body: for a call the limit cut, one the model stopped inside
/// and a block with malformed structure, a character a frame, the body
/// carries the text before the call and the call's name in a tool-call
/// delta; the usage chunk follows the last of the answer, and the contract's
/// event is the last frame, with no finish chunk ahead of it and no `[DONE]`
/// after it.
#[tokio::test]
async fn a_failed_call_went_out_as_far_as_it_was_read_and_the_contracts_event_ends_the_body() {
    for (call, finish, failure) in [
        (
            CUT_CALL,
            "length",
            InvalidToolCallFormat::new(UnreadableCall::Incomplete, true),
        ),
        (
            CUT_CALL,
            "stop",
            InvalidToolCallFormat::new(UnreadableCall::Incomplete, false),
        ),
        (
            MALFORMED_CALL,
            "stop",
            InvalidToolCallFormat::new(UnreadableCall::Malformed, false),
        ),
    ] {
        let frames = by_character(&format!("{BEFORE}{call}"), finish);
        let body = response_body(frames, spec(true, true)).await;
        assert!(!body.contains("[DONE]"), "{body}");
        let (frames, events) = frames_and_events(&body);
        let contract = failure.sse_frame();
        assert_eq!(
            frames.last().copied(),
            std::str::from_utf8(&contract).ok().map(str::trim_end),
            "{body}"
        );
        assert_eq!(
            frames
                .iter()
                .filter(|frame| frame.starts_with("event: "))
                .count(),
            1,
            "{body}"
        );
        assert!(content(&events).starts_with(BEFORE), "{body}");
        assert_eq!(tool_names(&events), vec!["lookup"], "{body}");
        assert!(finish_reasons(&events).is_empty(), "{body}");
        let (last, answer) = events.split_last().expect("the answer's events");
        assert!(
            last["usage"].is_object() && last["choices"].as_array().is_some_and(Vec::is_empty),
            "the usage chunk ahead of the event: {body}"
        );
        assert!(
            answer.iter().all(|event| !event["usage"].is_object()),
            "{body}"
        );
    }
}

/// A call streams as it is read, the contract or not: with the engine silent
/// inside the call's value, the call's name and the value so far are in the
/// body's frames already (all but the short tail MiniMax's delta-size
/// contract keeps back from every stream of its wire), and no failure is; the
/// rest of the value and the limit's cut then end the body with the
/// contract's event.
#[tokio::test]
async fn a_call_streams_as_it_is_read_before_the_output_ends() {
    const PATIENCE: Duration = Duration::from_secs(10);
    const OPENED: &str = "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"lookup\">\n]<]minimax[>[<city>Paris, France";
    let (resume, gate) = oneshot::channel();
    let (stream, server) = gated_stream(
        vec![chunk(0, BEFORE), chunk(0, OPENED)],
        gate,
        vec![chunk(0, " and on"), complete(0, "length")],
        "0",
    )
    .await;
    let mut body = response_of(stream, spec(true, false)).await;
    let mut early = String::new();
    timeout(PATIENCE, async {
        loop {
            let (_, events) = frames_and_events(&early);
            if tool_arguments(&events).contains("Paris") {
                break;
            }
            let frame = body.frame().await.expect("a frame").expect("a body frame");
            early.push_str(std::str::from_utf8(frame.data_ref().expect("data")).expect("UTF-8"));
        }
    })
    .await
    .expect("the call streams while the engine is silent");
    let (_, events) = frames_and_events(&early);
    assert_eq!(content(&events), BEFORE, "{early}");
    assert_eq!(tool_names(&events), vec!["lookup"], "{early}");
    assert!(
        tool_arguments(&events).starts_with("{\"city\": \"Paris"),
        "{early}"
    );
    assert!(!early.contains("event: error"), "{early}");
    resume.send(()).expect("the engine goes on");
    let rest = body.collect().await.expect("the body").to_bytes();
    server.abort();
    let rest = std::str::from_utf8(&rest).expect("UTF-8");
    assert!(
        rest.ends_with(&format!("event: error\ndata: {CUT_BY_THE_LIMIT}\n\n")),
        "{rest}"
    );
    assert!(!rest.contains("[DONE]"), "{rest}");
    let whole = format!("{early}{rest}");
    let (_, events) = frames_and_events(&whole);
    assert_eq!(
        tool_arguments(&events),
        "{\"city\": \"Paris, France and on",
        "{whole}"
    );
    assert!(finish_reasons(&events).is_empty(), "{whole}");
}

/// The whole answer of `model` whose output is `output`, ended by the engine
/// with `finish`, under `spec`.
async fn whole_answer(
    model: &str,
    output: &str,
    finish: &str,
    spec: &ChatResponseSpec,
) -> Result<ChatChoice, ChoiceFailure> {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let engine = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
        output_ids: output.chars().map(u32::from).collect(),
        finish_reason: finish.to_string(),
        ..Default::default()
    });
    let mut decoder =
        StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), false);
    ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    )
    .process_single_choice(
        "chatcmpl-unreadable",
        &engine,
        0,
        spec,
        model,
        &tokenizer,
        &mut decoder,
        0,
        false,
        false,
        None,
        None,
    )
    .await
    .map(|(choice, _)| choice)
}

/// The whole (non-streamed) answer with a cut call is the 400 with the
/// contract's body: the finish blamed on the limit only where the engine's
/// finish says so, a block with malformed structure the `error` finish, and
/// the response the contract's header, status and body. Without tools the
/// answer is a message as before, and a whole call is one too.
#[tokio::test]
async fn the_whole_answer_with_a_cut_call_is_the_400_with_the_contracts_body() {
    let output = format!("{BEFORE}{CUT_CALL}");
    for (finish, ended_at_limit) in [("length", true), ("stop", false)] {
        let read = whole_answer(M3, &output, finish, &spec(true, false)).await;
        let Err(ChoiceFailure::ToolCallFormat(failure)) = read else {
            panic!("the contract's failure under {finish}");
        };
        assert_eq!(
            failure,
            InvalidToolCallFormat::new(UnreadableCall::Incomplete, ended_at_limit)
        );
    }
    let read = whole_answer(M3, MALFORMED_CALL, "length", &spec(true, false)).await;
    let Err(ChoiceFailure::ToolCallFormat(failure)) = read else {
        panic!("the contract's failure for a malformed block");
    };
    assert_eq!(
        failure,
        InvalidToolCallFormat::new(UnreadableCall::Malformed, true)
    );
    assert_eq!(failure.finish_reason(), "error");

    let read = whole_answer(M3, &output, "length", &spec(true, false)).await;
    let Err(ChoiceFailure::ToolCallFormat(failure)) = read else {
        panic!("the contract's failure");
    };
    let response = failure.response();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response
            .headers()
            .get(HEADER_X_SMG_ERROR_CODE)
            .and_then(|value| value.to_str().ok()),
        Some("invalid_tool_call_format")
    );
    let body = response
        .into_body()
        .collect()
        .await
        .expect("the body")
        .to_bytes();
    assert_eq!(std::str::from_utf8(&body).expect("UTF-8"), CUT_BY_THE_LIMIT);

    let choice = whole_answer(M3, &output, "length", &spec(false, false))
        .await
        .expect("a message as before");
    assert_eq!(
        choice.message.content.as_deref(),
        Some(BEFORE_AND_THE_CUTS_NEWLINE)
    );
    assert_eq!(choice.finish_reason.as_deref(), Some("length"));
    let whole = format!("{BEFORE}{CUT_CALL}{CLOSE}");
    let choice = whole_answer(M3, &whole, "stop", &spec(true, false))
        .await
        .expect("a whole call is a message");
    let calls = choice.message.tool_calls.expect("the call");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "lookup");
    assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
}

/// The whole request fails on one failing choice: the non-streamed answer of
/// an engine that cut the call is the 400 itself, with the contract's header
/// and body, in place of a response.
#[tokio::test]
async fn the_whole_request_is_answered_with_the_400() {
    let output = format!("{BEFORE}{CUT_CALL}");
    let mut answer = complete(0, "length");
    if let Some(proto::generate_response::Response::Complete(complete)) = &mut answer.response {
        complete.output_ids = output.chars().map(u32::from).collect();
        complete.completion_tokens = complete.output_ids.len() as u32;
    }
    let (stream, server) = scripted_stream(vec![answer], "0").await;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
    let answered = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    )
    .process_non_streaming_chat_response(
        context::ExecutionResult::Single {
            stream,
            dispatch_start: Instant::now(),
        },
        spec(true, false),
        dispatch(M3),
        tokenizer,
        &mut decoder,
        false,
    )
    .await;
    server.abort();
    let Err(response) = answered else {
        panic!("the request's failure");
    };
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response
            .headers()
            .get(HEADER_X_SMG_ERROR_CODE)
            .and_then(|value| value.to_str().ok()),
        Some("invalid_tool_call_format")
    );
    assert_eq!(
        utils::error_type_for_response(&response),
        "invalid_tool_call_format"
    );
    assert_eq!(utils::error_reason_for_response(&response), "bad_response");
    let body = response
        .into_body()
        .collect()
        .await
        .expect("the body")
        .to_bytes();
    assert_eq!(std::str::from_utf8(&body).expect("UTF-8"), CUT_BY_THE_LIMIT);
}

/// The engine's counters of an answer it completed are what the request
/// used, whether the request then fails or not: the whole request's 400
/// settles the reservation with them, as the stream that fails the same way
/// does, and the close of an unsettled reservation that follows a failed
/// answer finds it settled.
#[tokio::test]
async fn a_failed_requests_reservation_is_settled_with_the_engines_counts() {
    let output = format!("{BEFORE}{CUT_CALL}");
    let tokens = output.chars().count() as u32;
    let settled = vec![UsageSettlement {
        actual_input_tokens: 11,
        completion_tokens: tokens,
        cached_input_tokens: 0,
        reasoning_tokens: 0,
    }];

    // Streamed: the stream's own settlement, at its end.
    let frames = vec![chunk(0, &output), complete_with_prompt(0, "length", 11)];
    let (stream, server) = scripted_stream(frames, "0").await;
    let (tx, rx) = sse_channel();
    let (handle, backend) = RecordingBackend::reservation();
    let result = processor(true)
        .process_streaming_chunks(
            stream,
            dispatch(M3),
            Arc::new(CharacterTokenizer::default()),
            (None, None, false, false, false),
            spec(true, false),
            &tx,
            Some(handle),
        )
        .await;
    drop(tx);
    collect_events(rx).await;
    server.abort();
    the_contracts_failure(result);
    assert_eq!(backend.settlements(), settled);
    assert_eq!(backend.closes(), 0);

    // Whole: the chat stage settles before it answers the failure.
    let mut answer = complete_with_prompt(0, "length", 11);
    if let Some(proto::generate_response::Response::Complete(complete)) = &mut answer.response {
        complete.output_ids = output.chars().map(u32::from).collect();
        complete.completion_tokens = tokens;
    }
    let (stream, server) = scripted_stream(vec![answer], "0").await;
    let (mut ctx, _store) = placed_stream_context(stream, dispatch(M3)).await;
    ctx.streaming = false;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    ctx.response.stop_decoder = Some(utils::create_stop_decoder(
        &tokenizer, None, None, false, false, false,
    ));
    let (handle, backend) = RecordingBackend::reservation();
    let cell = Arc::new(RateLimitCell::new());
    cell.set_for_test(RateLimitOutcome::Admitted(handle));
    ctx.rate_limit_cell = Some(Arc::clone(&cell));
    let answered =
        ChatResponseProcessingStage::new(response_processor(), Arc::new(processor(true)))
            .process_result(&mut ctx, ResponseSpec::Chat(Box::new(spec(true, false))))
            .await;
    server.abort();
    let Err(response) = answered else {
        panic!("the request's failure");
    };
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(backend.settlements(), settled);
    assert_eq!(backend.closes(), 0);
    // What the router does after a failed answer: the reservation is settled already.
    cell.close_unsettled().await;
    assert_eq!(backend.settlements(), settled);
    assert_eq!(backend.closes(), 0);
}

/// Another wire with the same output ends as before: the M3 table under a
/// served name that is not MiniMax's ends a cut call with the engine's
/// finish, streamed (the call out as far as the model wrote it) and whole (a
/// message), and a call the model stopped inside goes out closed for it.
#[tokio::test]
async fn another_wire_with_the_same_output_ends_as_before() {
    let served = spec_of(SERVED, Table::MinimaxM3, true, false, None);
    assert!(!served.fails_on_unreadable_call());

    let frames = vec![chunk(0, BEFORE), chunk(0, CUT_CALL), complete(0, "length")];
    let (result, events) = streamed(frames, served.clone()).await;
    assert!(result.is_ok(), "{:?}", result.err().map(|e| e.to_string()));
    assert_eq!(content(&events), BEFORE_AND_THE_CUTS_NEWLINE, "{events:?}");
    assert_eq!(finish_reasons(&events), vec!["length"]);
    assert_the_call_went_out_cut(&events);

    let frames = vec![chunk(0, CUT_CALL), complete(0, "stop")];
    let (result, events) = streamed(frames, served.clone()).await;
    assert!(result.is_ok(), "{:?}", result.err().map(|e| e.to_string()));
    assert_eq!(tool_names(&events), vec!["lookup"]);
    assert_eq!(
        serde_json::from_str::<Value>(&tool_arguments(&events)).expect("closed for the model"),
        json!({"city": "Paris"})
    );
    assert_eq!(finish_reasons(&events), vec!["tool_calls"]);

    let output = format!("{BEFORE}{CUT_CALL}");
    let choice = whole_answer(SERVED, &output, "length", &served)
        .await
        .expect("a message as before");
    assert_eq!(
        choice.message.content.as_deref(),
        Some(BEFORE_AND_THE_CUTS_NEWLINE)
    );
    assert_eq!(choice.finish_reason.as_deref(), Some("length"));
    assert_eq!(
        choice.message.tool_calls.map(|calls| calls.len()),
        Some(1),
        "the cut call, as before"
    );
}

/// And another table on MiniMax's wire ends as before: a MiniMax M2 answer
/// cut inside a call, under the same tools, is read by the M2 table, which
/// the contract does not cover, streamed and whole.
#[tokio::test]
async fn a_minimax_m2_answer_cut_inside_a_call_ends_as_before() {
    let m2 = spec_of(M2, Table::MinimaxM2, true, false, None);
    assert!(m2.expects_tool_calls, "the request expects calls");
    assert!(!m2.fails_on_unreadable_call(), "the M2 table's answers");

    let frames = vec![
        chunk(0, BEFORE),
        chunk(0, M2_CUT_CALL),
        complete(0, "length"),
    ];
    let (result, events) = streamed(frames, m2.clone()).await;
    assert!(result.is_ok(), "{:?}", result.err().map(|e| e.to_string()));
    assert_eq!(content(&events), BEFORE, "{events:?}");
    assert_eq!(finish_reasons(&events), vec!["length"]);
    assert_the_call_went_out_cut(&events);

    let output = format!("{BEFORE}{M2_CUT_CALL}");
    let choice = whole_answer(M2, &output, "length", &m2)
        .await
        .expect("a message as before");
    assert_eq!(choice.finish_reason.as_deref(), Some("length"));
}
