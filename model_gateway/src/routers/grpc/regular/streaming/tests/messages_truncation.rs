//! The stop reason of an engine `length` finish on the Messages paths. The
//! engines report the request's budget and the model's window alike as
//! `length`; the gateway tells them apart from the counts
//! (`MessagesResponseSpec::length_stop_reason`) on the stream, on the whole
//! answer, and under Symphony's reader, whose own stop reason is the
//! engine's finish.

use serde_json::Value;
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};
use symphony::{formats::qwen3, CallSyntax};

use super::{
    super::*,
    eof::{
        chunk, collect_events, complete, dispatch, messages_spec, processor, response_processor,
        scripted_stream, CharacterTokenizer,
    },
    symphony_chat::{assigned_message, message_events},
};
use crate::routers::grpc::spec::MessagesResponseSpec;

/// A `length` finish of `output` as the engine reports it: its ids, its count
/// and the prompt's count as the engine held it.
fn length_finish(prompt_tokens: u32, output: &str) -> proto::GenerateResponse {
    let mut done = complete(0, "length");
    if let Some(GenerationEvent::Complete(complete)) = done.response.as_mut() {
        complete.output_ids = output.chars().map(u32::from).collect();
        complete.completion_tokens = complete.output_ids.len() as u32;
        complete.prompt_tokens = prompt_tokens;
    }
    done
}

/// A plain spec of `max_tokens` with the worker's window and a rendered
/// count two short of the engine's `prompt_tokens`: the window is judged on
/// the engine's count, as media expanded on the worker would require.
fn spec(max_tokens: u32, prompt_tokens: u32, context_window: Option<u32>) -> MessagesResponseSpec {
    let mut spec = messages_spec(false);
    spec.max_tokens = max_tokens;
    spec.prompt_tokens = prompt_tokens.saturating_sub(2);
    spec.context_window = context_window;
    spec
}

/// The `message_delta` stop reason of a stream's events.
fn streamed_stop_reason(events: &[Value]) -> String {
    let delta = events
        .iter()
        .find(|event| event["type"] == "message_delta")
        .expect("message_delta");
    delta["delta"]["stop_reason"]
        .as_str()
        .expect("a stop reason")
        .to_string()
}

/// The stream's stop reason for a three-token answer the engine cut at
/// `length`, holding a prompt of `prompt_tokens`.
async fn stream_cut_at_length(spec: MessagesResponseSpec, prompt_tokens: u32) -> String {
    let (stream, server) = scripted_stream(
        vec![chunk(0, "abc"), length_finish(prompt_tokens, "abc")],
        "0",
    )
    .await;
    let (tx, rx) = sse_channel();
    let result = processor(false)
        .process_messages_streaming_chunks(
            stream,
            dispatch(),
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
    assert!(result.is_ok(), "{result:?}");
    streamed_stop_reason(&events)
}

/// The whole answer's stop reason for the same cut, read by the older parsers
/// or by Symphony's reader (the spec's assignment decides).
async fn whole_answer_cut_at_length(spec: MessagesResponseSpec, prompt_tokens: u32) -> String {
    let (stream, server) = scripted_stream(vec![length_finish(prompt_tokens, "abc")], "0").await;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
    let (message, _) = response_processor()
        .process_non_streaming_messages_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: Instant::now(),
            },
            spec,
            dispatch(),
            tokenizer,
            &mut decoder,
        )
        .await
        .unwrap_or_else(|error| panic!("{}", error.status()));
    server.abort();
    let message = serde_json::to_value(message).expect("a message serializes");
    message["stop_reason"]
        .as_str()
        .expect("a stop reason")
        .to_string()
}

/// The cases: the budget used whole is `max_tokens`; a cut short of the
/// budget, or one that fills the worker's window with the prompt, is
/// `model_context_window_exceeded`.
fn cases() -> [(&'static str, u32, u32, Option<u32>, &'static str); 4] {
    [
        (
            "the budget used whole, no window known",
            3,
            10,
            None,
            "max_tokens",
        ),
        (
            "short of the budget",
            64,
            10,
            None,
            "model_context_window_exceeded",
        ),
        (
            "the budget used whole inside the window",
            3,
            10,
            Some(14),
            "max_tokens",
        ),
        (
            "the budget used whole at the window",
            3,
            10,
            Some(13),
            "model_context_window_exceeded",
        ),
    ]
}

#[tokio::test]
async fn a_length_finish_reads_the_budget_against_the_window_on_the_stream() {
    for (name, max_tokens, prompt_tokens, window, expected) in cases() {
        let got =
            stream_cut_at_length(spec(max_tokens, prompt_tokens, window), prompt_tokens).await;
        assert_eq!(got, expected, "{name}");
    }
}

#[tokio::test]
async fn a_length_finish_reads_the_budget_against_the_window_on_the_whole_answer() {
    for (name, max_tokens, prompt_tokens, window, expected) in cases() {
        let got =
            whole_answer_cut_at_length(spec(max_tokens, prompt_tokens, window), prompt_tokens)
                .await;
        assert_eq!(got, expected, "{name}");
    }
}

/// Symphony's reader reports the engine's `length` as `max_tokens`; the whole
/// answer reads it against the budget and the window like the other paths.
#[tokio::test]
async fn symphony_s_length_finish_reads_the_budget_against_the_window_on_the_whole_answer() {
    for (name, max_tokens, prompt_tokens, window, expected) in cases() {
        let mut spec = assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
        spec.max_tokens = max_tokens;
        spec.prompt_tokens = prompt_tokens.saturating_sub(2);
        spec.context_window = window;
        let got = whole_answer_cut_at_length(spec, prompt_tokens).await;
        assert_eq!(got, expected, "{name}");
    }
}

/// The same under Symphony on the stream: the `Finish` event's `max_tokens`
/// is read against the budget and the window before the `message_delta`.
#[tokio::test]
async fn symphony_s_length_finish_reads_the_budget_against_the_window_on_the_stream() {
    for (name, max_tokens, prompt_tokens, window, expected) in cases() {
        let mut spec = assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false);
        spec.max_tokens = max_tokens;
        spec.prompt_tokens = prompt_tokens.saturating_sub(2);
        spec.context_window = window;
        let (result, events) = message_events(
            vec![chunk(0, "abc"), length_finish(prompt_tokens, "abc")],
            spec,
            None,
        )
        .await;
        assert!(result.is_ok(), "{name}: {result:?}");
        assert_eq!(streamed_stop_reason(&events), expected, "{name}");
    }
}
