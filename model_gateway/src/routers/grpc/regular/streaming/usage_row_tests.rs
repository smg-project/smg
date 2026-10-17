//! The tenant usage row of a streamed response at its end, whichever way the
//! stream ends: read to its last token, abandoned by the client mid-stream,
//! failed by the engine mid-stream, failed on its prefill leg before the
//! first token. Each books the stream exactly once: the normal end with the
//! usage block's numbers, every other end with the prompt as tokenized at
//! dispatch and the completion tokens streamed up to the end.

use std::time::Duration;

use futures::{stream, StreamExt as _};
use http_body::Frame;
use http_body_util::BodyExt;
use smg_grpc_client::sglang_scheduler::{
    proto as scheduler, proto::generate_response::Response as SchedulerEvent,
    AbortOnDropStream as SchedulerStream, SglangSchedulerClient,
};
use tokio::{sync::oneshot, task::JoinHandle, time::sleep};

use super::{
    eof_tests::{
        chat_spec, chunk, collect_events, complete_with_prompt, cut_stream, data_frame,
        decoded_stream, gated_stream, mock_worker_endpoint, processor, scripted_stream,
        trailers_frame, CharacterTokenizer,
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

/// A chunk of a cumulative-semantics engine (the scheduler proto): `text` with
/// the running count `generated`, which the tracker only reports; the count it
/// books comes from the `Complete`.
fn scheduler_chunk(index: u32, text: &str, generated: u32) -> scheduler::GenerateResponse {
    scheduler::GenerateResponse {
        request_id: String::new(),
        response: Some(SchedulerEvent::Chunk(scheduler::GenerateStreamChunk {
            index,
            completion_tokens: generated,
            token_ids: text.chars().map(u32::from).collect(),
            ..Default::default()
        })),
    }
}

/// The `Complete` of a cumulative-semantics engine: the prompt count and the
/// whole output's ids, whose length is the completion count.
fn scheduler_complete(index: u32, prompt_tokens: u32, output: &str) -> scheduler::GenerateResponse {
    let output_ids: Vec<u32> = output.chars().map(u32::from).collect();
    scheduler::GenerateResponse {
        request_id: String::new(),
        response: Some(SchedulerEvent::Complete(scheduler::GenerateComplete {
            index,
            finish_reason: "stop".to_string(),
            prompt_tokens,
            completion_tokens: output_ids.len() as u32,
            output_ids,
            ..Default::default()
        })),
    }
}

/// [`gated_stream`] over the scheduler proto: a cumulative-semantics stream
/// whose frames after `first` are held back until `gate` fires.
async fn gated_scheduler_stream(
    first: Vec<scheduler::GenerateResponse>,
    gate: oneshot::Receiver<()>,
    rest: Vec<scheduler::GenerateResponse>,
) -> (ProtoStream, JoinHandle<()>) {
    let head: Vec<Result<Frame<Bytes>, tonic::Status>> =
        first.iter().map(|frame| Ok(data_frame(frame))).collect();
    let tail: Vec<Result<Frame<Bytes>, tonic::Status>> = rest
        .iter()
        .map(|frame| Ok(data_frame(frame)))
        .chain(std::iter::once(Ok(trailers_frame("0"))))
        .collect();
    let (url, server) = mock_worker_endpoint().await;
    let client = SglangSchedulerClient::connect(&url)
        .await
        .expect("connect mock worker");
    let gated = stream::once(async move {
        let _ = gate.await;
        stream::iter(tail)
    })
    .flatten();
    let stream = SchedulerStream::new(
        decoded_stream(stream::iter(head).chain(gated)),
        "usage-row-test".to_string(),
        client,
    );
    // No generation was sent to the mock, so there is nothing to cancel.
    stream.mark_completed();
    (ProtoStream::Sglang(stream), server)
}

fn messages_spec() -> MessagesResponseSpec {
    MessagesResponseSpec {
        thinking_requested: false,
        thinking_text_returned: true,
        reasoning_in_prefill: false,
        tool_choice: None,
        has_tools: false,
        history_tool_calls_count: 0,
        chat_tools: Vec::new().into(),
        stop_sequences: None,
        symphony: None,
        first_tool_call_only: false,
    }
}

/// The Messages streaming path over `stream` for `tenant`, its events sent to
/// `tx`; the caller reads them, or leaves.
fn messages_task(
    stream: ProtoStream,
    tenant: &str,
    tx: SseSender,
) -> JoinHandle<Result<(), utils::StreamFailure>> {
    let dispatch = dispatch(tenant);
    #[expect(
        clippy::disallowed_methods,
        reason = "bounded test fixture; the task ends with the stream"
    )]
    tokio::spawn(async move {
        processor(false)
            .process_messages_streaming_chunks(
                stream,
                dispatch,
                character_tokenizer(),
                (None, None, false, false, false),
                messages_spec(),
                &tx,
                None,
            )
            .await
    })
}

#[tokio::test]
async fn a_messages_stream_the_client_leaves_at_the_finish_books_the_engines_counts_once() {
    let tenant = "auth:usage-row-messages-client-gone-after-complete";
    // A cumulative-semantics engine: the chunks carry the running count, which
    // the tracker only reports; the count it books arrives with the Complete.
    let (gate, held) = oneshot::channel();
    let (stream, server) = gated_scheduler_stream(
        vec![scheduler_chunk(0, "Hel", 3), scheduler_chunk(0, "lo", 5)],
        held,
        vec![scheduler_complete(0, 7, "Hello")],
    )
    .await;
    let (tx, mut rx) = sse_channel();
    let task = messages_task(stream, tenant, tx);

    // The client reads the whole text ...
    let mut seen = String::new();
    while !seen.contains(r#""text":"lo""#) {
        let frame = rx
            .recv()
            .await
            .expect("a frame before the client leaves")
            .expect("successful SSE write");
        seen.push_str(std::str::from_utf8(&frame).expect("UTF-8 SSE"));
    }
    // ... and leaves. The engine's Complete arrives; the events after it (the
    // block's stop, the message delta and stop) cannot be sent: the row takes
    // the engine's counts, the completion count included.
    drop(rx);
    let _ = gate.send(());
    let result = task.await.expect("the stream task");
    assert!(
        result.is_err(),
        "the send after the client left fails: {result:?}"
    );

    assert_eq!(
        counts(&booked(tenant).await),
        (1, 7, 5),
        "one request; the engine's prompt count and the completion count its Complete \
         carried (the chunks of a cumulative engine book nothing on their own)"
    );
    server.abort();
}

#[tokio::test]
async fn a_messages_stream_that_ends_without_a_complete_books_the_dispatch_prompt_once() {
    let tenant = "auth:usage-row-messages-no-complete";
    let (stream, server) = scripted_stream(vec![chunk(0, "Hel")], "0").await;
    let (tx, rx) = sse_channel();
    let task = messages_task(stream, tenant, tx);
    let events = collect_events(rx).await;
    let result = task.await.expect("the stream task");
    assert!(result.is_ok(), "{result:?}");
    assert!(
        events.iter().any(|event| event["type"] == "message_stop"),
        "the stream ends on its own: {events:?}"
    );
    server.abort();

    assert_eq!(
        counts(&booked(tenant).await),
        (1, u64::from(PROMPT_AT_DISPATCH), 3),
        "a clean end without the engine's count keeps the prompt as tokenized at dispatch, \
         on the message_stop exit as on the message_delta one"
    );
}

#[tokio::test]
async fn a_chat_stream_that_ends_without_a_complete_books_the_dispatch_prompt_once() {
    let tenant = "auth:usage-row-chat-no-complete";
    let (stream, server) = scripted_stream(vec![chunk(0, "Hel")], "0").await;
    let text = read_to_end(chat_response(stream, dispatch(tenant)).await).await;
    assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
    server.abort();

    assert_eq!(
        counts(&booked(tenant).await),
        (1, u64::from(PROMPT_AT_DISPATCH), 3),
        "no Complete populated the prompt count: the dispatch count stands, not zero"
    );
}

#[tokio::test]
async fn a_completions_stream_that_ends_without_a_complete_books_the_dispatch_prompt_once() {
    let tenant = "auth:usage-row-completions-no-complete";
    let request: openai_protocol::completion::CompletionRequest = serde_json::from_value(
        serde_json::json!({"model": "usage-row-test", "prompt": "test", "stream": true}),
    )
    .expect("a completion request");
    let (stream, server) = scripted_stream(vec![chunk(0, "Hel")], "0").await;
    let dispatch = dispatch(tenant);
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_COMPLETIONS);
    let response = Arc::new(processor(false))
        .process_completion_streaming_response(
            context::ExecutionResult::Single { stream },
            CompletionResponseSpec::from(&request),
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
        (1, u64::from(PROMPT_AT_DISPATCH), 3),
        "a unit without a Complete leaves the dispatch count in place, as the settle before the \
         usage chunk has it"
    );
}
