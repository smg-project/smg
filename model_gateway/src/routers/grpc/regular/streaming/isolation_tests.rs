//! A panic in the reading of a response is that request's failure, on every
//! path: the chat stream ends with the error event under `reader_panicked`
//! and `[DONE]`, the Messages stream with its error event, the whole
//! responses are a 500 under the same code, the output is counted under the
//! table that read it and one line names the request; nothing of the task or
//! the process is touched (`utils::isolation`). The panic is the test trap's,
//! sprung by a marker in the output ([`arm_test_trap`]).

use std::{sync::Arc, time::Duration};

use axum::{
    body::{to_bytes, Body},
    http::StatusCode,
};
use llm_tokenizer::StopSequenceConfig;
use serde_json::Value;
use smg_grpc_client::vllm_engine::proto;
use symphony::{formats::qwen3, CallSyntax};

use super::{
    eof_tests::{
        chat_spec, chunk, complete, dispatch, processor, scripted_stream, CharacterTokenizer,
    },
    symphony_tests::{assigned, assigned_message, scraped, series_line},
    *,
};
use crate::routers::{
    common::retry::is_retryable_response,
    grpc::{
        regular::processor::ResponseProcessor,
        utils::isolation::{arm_test_trap, READER_PANICKED_CODE},
    },
};

/// The text that springs the trap: no other test's output carries it.
const MARKER: &str = "TRAP-7f3a-READER";
const PROMPT: &str = "<|im_start|>assistant\n";

fn output() -> String {
    format!("Hello {MARKER} world")
}

/// A `Complete` frame whose output is `text`, one token per character.
fn whole(text: &str) -> proto::GenerateResponse {
    let mut frame = complete(0, "stop");
    if let Some(proto::generate_response::Response::Complete(complete)) = &mut frame.response {
        complete.output_ids = text.chars().map(u32::from).collect();
        complete.completion_tokens = complete.output_ids.len() as u32;
    }
    frame
}

/// The `data:` payloads of the SSE `body`, read to its end within five
/// seconds, in order.
async fn sse_payloads(body: Body) -> Vec<String> {
    let bytes = tokio::time::timeout(Duration::from_secs(5), to_bytes(body, 1 << 20))
        .await
        .expect("the body ends in time")
        .expect("the body");
    let text = String::from_utf8(bytes.to_vec()).expect("text");
    text.split("\n\n")
        .filter_map(|event| {
            event
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .map(str::to_string)
        })
        .collect()
}

/// The JSON body of a 500 under `reader_panicked`, the failure named and
/// nothing of the panic's message in it; the answer is final (the pipeline
/// retries it on no worker).
async fn reader_panicked_body(response: Response) -> Value {
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        !is_retryable_response(&response),
        "a panic is never retried"
    );
    let bytes = to_bytes(response.into_body(), 65536)
        .await
        .expect("the body");
    let body: Value = serde_json::from_slice(&bytes).expect("a JSON body");
    assert_eq!(body["error"]["code"], READER_PANICKED_CODE, "{body}");
    assert_eq!(
        body["error"]["message"], "the gateway's reader of the response panicked",
        "{body}"
    );
    body
}

fn whole_reader() -> ResponseProcessor {
    ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    )
}

/// The one `panicked` output of the qwen3 table in `scrape`.
fn assert_counted_once(scrape: &str) {
    let outputs = series_line(
        scrape,
        "smg_symphony_outputs_total",
        &["table=\"qwen3\"", "outcome=\"panicked\""],
    )
    .unwrap_or_else(|| panic!("the panicked outcome:\n{scrape}"));
    assert!(outputs.ends_with(" 1"), "{outputs}");
}

#[tracing_test::traced_test]
#[tokio::test]
async fn a_reader_panic_ends_the_chat_stream_with_an_error_event_and_done() {
    arm_test_trap(MARKER);
    let (stream, server) =
        scripted_stream(vec![chunk(0, &output()), complete(0, "stop")], "0").await;
    let dispatch = dispatch();
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_CHAT);
    let (payloads, scrape) = scraped(async {
        let response = Arc::new(processor(true))
            .process_streaming_response(
                context::ExecutionResult::Single { stream },
                assigned(chat_spec(true), qwen3(CallSyntax::Json), PROMPT),
                dispatch,
                Arc::new(CharacterTokenizer::default()),
                false,
                None,
                None,
                source,
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        sse_payloads(response.into_body()).await
    })
    .await;
    server.abort();
    // The terminal frames: the error event under its code, then `[DONE]`.
    let [.., error, done] = payloads.as_slice() else {
        panic!("two terminal frames: {payloads:?}");
    };
    assert_eq!(done.trim(), "[DONE]");
    let error: Value = serde_json::from_str(error).expect("the error event");
    assert_eq!(error["error"]["type"], "internal_error", "{error}");
    assert_eq!(error["error"]["code"], READER_PANICKED_CODE, "{error}");
    // The event names the failure; the panic's message is the log line's.
    assert_eq!(
        error["error"]["message"], "the gateway's reader of the response panicked",
        "{error}"
    );
    assert_counted_once(&scrape);
    assert!(logs_contain(
        "The response's reader panicked; the request ends with an error"
    ));
    assert!(logs_contain("request_id=\"chatcmpl-eof\""));
    assert!(logs_contain("table=\"qwen3\""));
    assert!(logs_contain("panic.message="));
    // The marker is quoted in the trap's own message, and quoted spans are
    // elided from the line.
    assert!(!logs_contain(MARKER));
}

#[tracing_test::traced_test]
#[tokio::test]
async fn a_reader_panic_answers_the_whole_chat_with_a_500_under_its_code() {
    arm_test_trap(MARKER);
    let (stream, server) = scripted_stream(vec![whole(&output())], "0").await;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let mut decoder =
        StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), false);
    let (result, scrape) = scraped(whole_reader().process_non_streaming_chat_response(
        context::ExecutionResult::Single { stream },
        assigned(chat_spec(true), qwen3(CallSyntax::Json), PROMPT),
        dispatch(),
        tokenizer,
        &mut decoder,
        false,
    ))
    .await;
    server.abort();
    let response = match result {
        Ok((response, _)) => panic!("an answer: {response:?}"),
        Err(response) => response,
    };
    reader_panicked_body(response).await;
    assert_counted_once(&scrape);
    assert!(logs_contain(
        "The response's reader panicked; the request ends with an error"
    ));
    assert!(logs_contain("request_id=\"chatcmpl-eof\""));
    assert!(!logs_contain(MARKER));
}

#[tracing_test::traced_test]
#[tokio::test]
async fn a_reader_panic_ends_the_messages_stream_with_its_error_event() {
    arm_test_trap(MARKER);
    let (stream, server) =
        scripted_stream(vec![chunk(0, &output()), complete(0, "stop")], "0").await;
    let dispatch = dispatch();
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_MESSAGES);
    let (payloads, scrape) = scraped(async {
        let response = Arc::new(processor(true))
            .process_messages_streaming_response(
                context::ExecutionResult::Single { stream },
                assigned_message(qwen3(CallSyntax::Json), PROMPT, false),
                dispatch,
                Arc::new(CharacterTokenizer::default()),
                false,
                None,
                None,
                source,
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        sse_payloads(response.into_body()).await
    })
    .await;
    server.abort();
    // The Messages API's terminal event for a failure on the server's side.
    let last: Value =
        serde_json::from_str(payloads.last().expect("an event")).expect("the error event");
    assert_eq!(last["type"], "error", "{last}");
    assert_eq!(last["error"]["type"], "api_error", "{last}");
    // The event names the failure; the panic's message is the log line's.
    assert_eq!(
        last["error"]["message"], "the gateway's reader of the response panicked",
        "{last}"
    );
    assert_counted_once(&scrape);
    assert!(logs_contain("panic.message="));
    assert!(!logs_contain(MARKER));
}

#[tokio::test]
async fn a_reader_panic_answers_the_whole_message_with_a_500_under_its_code() {
    arm_test_trap(MARKER);
    let (stream, server) = scripted_stream(vec![whole(&output())], "0").await;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, true);
    let (result, scrape) = scraped(whole_reader().process_non_streaming_messages_response(
        context::ExecutionResult::Single { stream },
        assigned_message(qwen3(CallSyntax::Json), PROMPT, false),
        dispatch(),
        tokenizer,
        &mut decoder,
    ))
    .await;
    server.abort();
    let response = match result {
        Ok((message, _)) => panic!("a message with {:?}", message.content),
        Err(response) => response,
    };
    reader_panicked_body(response).await;
    assert_counted_once(&scrape);
}

/// An output without the marker reads as before with the trap armed: the
/// trap is one branch per step, and a request whose reading does not panic
/// is untouched by the isolation.
#[tokio::test]
async fn a_reading_that_does_not_panic_is_untouched_by_the_isolation() {
    arm_test_trap(MARKER);
    let (stream, server) =
        scripted_stream(vec![chunk(0, "Hello world"), complete(0, "stop")], "0").await;
    let dispatch = dispatch();
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_CHAT);
    let response = Arc::new(processor(true))
        .process_streaming_response(
            context::ExecutionResult::Single { stream },
            assigned(chat_spec(true), qwen3(CallSyntax::Json), PROMPT),
            dispatch,
            Arc::new(CharacterTokenizer::default()),
            false,
            None,
            None,
            source,
        )
        .await;
    let payloads = sse_payloads(response.into_body()).await;
    server.abort();
    assert_eq!(payloads.last().map(|s| s.trim()), Some("[DONE]"));
    let content: String = payloads
        .iter()
        .filter_map(|payload| serde_json::from_str::<Value>(payload).ok())
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_string)
        })
        .collect();
    assert_eq!(content, "Hello world");
    assert!(
        payloads
            .iter()
            .all(|payload| !payload.contains("\"error\"")),
        "{payloads:?}"
    );
}
