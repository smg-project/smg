//! The tenant usage row of a streamed response at its end, whichever way the
//! stream ends: read to its last token, abandoned by the client mid-stream,
//! failed by the engine mid-stream, failed on its prefill leg before the
//! first token. Each books the stream exactly once: the normal end with the
//! usage block's numbers, every other end with the prompt as tokenized at
//! dispatch and the completion tokens streamed up to the end.

use std::time::Duration;

use http_body_util::BodyExt;
use tokio::{sync::oneshot, time::sleep};

use super::{
    eof_tests::{
        chat_spec, chunk, complete_with_prompt, cut_stream, gated_stream, processor,
        scripted_stream, CharacterTokenizer,
    },
    *,
};
use crate::{
    middleware::TenantKey,
    observability::tenant_usage::{self, TenantUsage, Totals},
};

/// The prompt length the dispatch counted; no engine message of these
/// streams repeats it.
const PROMPT_AT_DISPATCH: u32 = 100;

/// The process-wide table the streams book to, installed by the first test
/// of this binary that needs it (first call wins, as at start-up); every
/// test books a tenant key of its own. A stream's guard books to the table
/// installed when the stream was dispatched (none: the guard is inert), so
/// [`dispatch`] installs it first.
fn table() -> &'static TenantUsage {
    tenant_usage::install(Arc::new(TenantUsage::new(1024)));
    tenant_usage::installed().expect("a table is installed")
}

fn dispatch(tenant: &str) -> context::DispatchMetadata {
    table();
    context::DispatchMetadata {
        request_id: "chatcmpl-usage-row".to_string(),
        model: "usage-row-test".to_string(),
        created: 1,
        weight_version: None,
        tenant_key: Some(TenantKey::new(tenant)),
        prompt_tokens: PROMPT_AT_DISPATCH,
    }
}

fn row(tenant: &str) -> Totals {
    table()
        .snapshot()
        .tenants
        .into_iter()
        .find(|row| row.tenant == tenant)
        .map(|row| row.totals)
        .unwrap_or_default()
}

/// The tenant's row once its request is booked (the stream's task ends on
/// its own time), read again a moment later so a second booking would show.
async fn booked(tenant: &str) -> Totals {
    let deadline = Instant::now() + Duration::from_secs(10);
    while row(tenant).requests == 0 {
        assert!(
            Instant::now() < deadline,
            "no usage row for {tenant} within 10 s"
        );
        sleep(Duration::from_millis(5)).await;
    }
    sleep(Duration::from_millis(50)).await;
    row(tenant)
}

/// Requests, prompt tokens and completion tokens of a row.
fn counts(totals: &Totals) -> (u64, u64, u64) {
    (
        totals.requests,
        totals.prompt_tokens,
        totals.completion_tokens,
    )
}

fn character_tokenizer() -> Arc<dyn Tokenizer> {
    Arc::new(CharacterTokenizer::default())
}

async fn chat_response(stream: ProtoStream, dispatch: context::DispatchMetadata) -> Response {
    chat_response_of(stream, dispatch, chat_spec(false)).await
}

async fn chat_response_of(
    stream: ProtoStream,
    dispatch: context::DispatchMetadata,
    spec: ChatResponseSpec,
) -> Response {
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_CHAT);
    Arc::new(processor(false))
        .process_streaming_response(
            context::ExecutionResult::Single { stream },
            spec,
            dispatch,
            character_tokenizer(),
            false,
            None,
            None,
            source,
        )
        .await
}

/// The whole SSE body of `response`, as text.
async fn read_to_end(response: Response) -> String {
    let body = response
        .into_body()
        .collect()
        .await
        .expect("the SSE body")
        .to_bytes();
    String::from_utf8(body.to_vec()).expect("UTF-8 SSE")
}

/// The first frame of `body`, as text.
async fn first_frame(body: &mut axum::body::Body) -> String {
    let frame = body.frame().await.expect("a frame").expect("an SSE frame");
    String::from_utf8(frame.data_ref().expect("a data frame").to_vec()).expect("UTF-8 SSE")
}

#[tokio::test]
async fn a_stream_read_to_its_end_books_the_usage_block_once() {
    let tenant = "auth:usage-row-normal-end";
    let (stream, server) = scripted_stream(
        vec![
            chunk(0, "Hel"),
            chunk(0, "lo"),
            complete_with_prompt(0, "stop", 7),
        ],
        "0",
    )
    .await;
    let text = read_to_end(chat_response(stream, dispatch(tenant)).await).await;
    assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
    server.abort();

    let totals = booked(tenant).await;
    assert_eq!(
        counts(&totals),
        (1, 7, 5),
        "one request, the engine's prompt count, the tokens the Complete confirmed"
    );
    assert!(totals.request_seconds > 0.0);
}

#[tokio::test]
async fn a_stream_the_client_abandons_books_the_dispatch_prompt_and_the_tokens_streamed_once() {
    let tenant = "auth:usage-row-client-gone";
    let (gate, held) = oneshot::channel();
    let (stream, server) = gated_stream(
        vec![chunk(0, "Hel")],
        held,
        vec![chunk(0, "lo"), complete_with_prompt(0, "stop", 7)],
        "0",
    )
    .await;
    let mut body = chat_response(stream, dispatch(tenant)).await.into_body();

    // The first delta reaches the client ...
    let first = first_frame(&mut body).await;
    assert!(first.contains("Hel"), "{first}");
    // ... then the client goes away: its response body is dropped. The engine
    // produces the next token, the gateway cannot deliver it, the stream ends.
    drop(body);
    let _ = gate.send(());

    assert_eq!(
        counts(&booked(tenant).await),
        (1, u64::from(PROMPT_AT_DISPATCH), 5),
        "one request; the prompt as tokenized at dispatch (the engine's count never \
         arrived); every token the engine delivered before the send failed"
    );
    server.abort();
}

#[tokio::test]
async fn a_stream_the_client_leaves_after_the_engines_count_books_that_count_once() {
    let tenant = "auth:usage-row-client-gone-after-complete";
    let (gate, held) = oneshot::channel();
    let (stream, server) = gated_stream(
        vec![chunk(0, "Hel"), chunk(0, "lo")],
        held,
        vec![complete_with_prompt(0, "stop", 7)],
        "0",
    )
    .await;
    let mut body = chat_response(stream, dispatch(tenant)).await.into_body();

    // The client reads the whole text ...
    assert!(first_frame(&mut body).await.contains("Hel"));
    assert!(first_frame(&mut body).await.contains("lo"));
    // ... and leaves. The engine's Complete arrives, the finish chunk cannot
    // be sent: the row takes the engine's counts, not the dispatch estimate.
    drop(body);
    let _ = gate.send(());

    assert_eq!(
        counts(&booked(tenant).await),
        (1, 7, 5),
        "one request; the engine's prompt count (its Complete arrived before the          client left); the tokens it confirmed"
    );
    server.abort();
}

#[tokio::test]
async fn a_stream_whose_first_message_is_the_engines_error_books_the_prompt_alone_once() {
    let tenant = "auth:usage-row-error-first";
    let (stream, server) =
        cut_stream(vec![], tonic::Status::unavailable("engine overloaded")).await;
    let text = read_to_end(chat_response(stream, dispatch(tenant)).await).await;
    assert!(text.contains("stream_error"), "{text}");
    server.abort();

    assert_eq!(
        counts(&booked(tenant).await),
        (1, u64::from(PROMPT_AT_DISPATCH), 0),
        "one request of the prompt as dispatched: the gateway cannot tell a rejection          from an engine that failed after the prefill"
    );
}

#[tokio::test]
async fn a_stream_books_the_prompt_without_the_unbilled_stub() {
    let tenant = "auth:usage-row-unbilled-stub";
    let mut spec = chat_spec(false);
    spec.unbilled_prompt_tokens = 2;
    let (stream, server) = scripted_stream(
        vec![chunk(0, "Hel"), complete_with_prompt(0, "stop", 7)],
        "0",
    )
    .await;
    let text = read_to_end(chat_response_of(stream, dispatch(tenant), spec).await).await;
    assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
    server.abort();

    assert_eq!(
        counts(&booked(tenant).await),
        (1, 5, 3),
        "the engine's 7 less the 2-token stub the client is not billed for, as its          usage block shows"
    );
}

#[tokio::test]
async fn a_stream_the_engine_fails_books_the_tokens_delivered_once() {
    let tenant = "auth:usage-row-engine-failed";
    let (stream, server) = cut_stream(
        vec![chunk(0, "Hel")],
        tonic::Status::internal("h2 protocol error: error reading a body from connection"),
    )
    .await;
    let text = read_to_end(chat_response(stream, dispatch(tenant)).await).await;
    assert!(
        text.contains("stream_error") && text.ends_with("data: [DONE]\n\n"),
        "{text}"
    );
    server.abort();

    assert_eq!(
        counts(&booked(tenant).await),
        (1, u64::from(PROMPT_AT_DISPATCH), 3)
    );
}

#[tokio::test]
async fn a_prefill_leg_that_fails_before_the_first_token_books_the_prompt_once() {
    let tenant = "auth:usage-row-prefill-failed";
    let (prefill, prefill_server) = scripted_stream(vec![], "13").await;
    let (decode, decode_server) = scripted_stream(
        vec![chunk(0, "Hi"), complete_with_prompt(0, "stop", 7)],
        "0",
    )
    .await;
    let (tx, rx) = sse_channel();
    let result = processor(false)
        .process_prefill_decode_streaming_chunks(
            prefill,
            decode,
            dispatch(tenant),
            character_tokenizer(),
            (None, None, false, false, false),
            chat_spec(false),
            &tx,
            context::PdTiming {
                prefill_start: Instant::now(),
                decode_sent: Instant::now(),
                runtime: "test",
            },
            vec![None],
            None,
        )
        .await;
    drop(tx);
    drop(rx);
    prefill_server.abort();
    decode_server.abort();
    assert!(result.is_err(), "the prefill leg failed");

    assert_eq!(
        counts(&booked(tenant).await),
        (1, u64::from(PROMPT_AT_DISPATCH), 0)
    );
}

#[tokio::test]
async fn a_generate_stream_the_engine_fails_books_the_tokens_delivered_once() {
    let tenant = "auth:usage-row-generate-failed";
    let (stream, server) = cut_stream(
        vec![chunk(0, "Hel")],
        tonic::Status::unavailable("the engine went away"),
    )
    .await;
    let dispatch = dispatch(tenant);
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_GENERATE);
    let response = Arc::new(processor(false))
        .process_streaming_generate(
            context::ExecutionResult::Single { stream },
            GenerateResponseSpec {
                return_logprob: false,
                expected_choices: 1,
            },
            dispatch,
            character_tokenizer(),
            None,
            None,
            source,
        )
        .await;
    let text = read_to_end(response).await;
    assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
    server.abort();

    assert_eq!(
        counts(&booked(tenant).await),
        (1, u64::from(PROMPT_AT_DISPATCH), 3)
    );
}

#[tokio::test]
async fn a_completions_stream_the_client_abandons_books_what_its_units_streamed_once() {
    let tenant = "auth:usage-row-completions-client-gone";
    let request: openai_protocol::completion::CompletionRequest = serde_json::from_value(
        serde_json::json!({"model": "usage-row-test", "prompt": "test", "stream": true}),
    )
    .expect("a completion request");
    let (gate, held) = oneshot::channel();
    let (stream, server) = gated_stream(
        vec![chunk(0, "Hel")],
        held,
        vec![chunk(0, "lo"), complete_with_prompt(0, "stop", 7)],
        "0",
    )
    .await;
    let dispatch = dispatch(tenant);
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_COMPLETIONS);
    let mut body = Arc::new(processor(false))
        .process_completion_streaming_response(
            context::ExecutionResult::Single { stream },
            CompletionResponseSpec::from(&request),
            dispatch,
            character_tokenizer(),
            None,
            None,
            source,
        )
        .await
        .into_body();

    let first = first_frame(&mut body).await;
    assert!(first.contains("Hel"), "{first}");
    drop(body);
    let _ = gate.send(());

    assert_eq!(
        counts(&booked(tenant).await),
        (1, u64::from(PROMPT_AT_DISPATCH), 5),
        "the request's one guard, counted into by its prompt's unit"
    );
    server.abort();
}
