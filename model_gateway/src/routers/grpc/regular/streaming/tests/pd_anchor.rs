//! Through a prefill/decode pair the stream task starts only once the decode
//! side exists (for sequential dispatch after the prefill side completed, with
//! the first decode chunk already buffered), so a clock the task starts itself
//! reads the prefill time away. Every streaming route therefore anchors its
//! time to first token and its generation duration at the prefill dispatch,
//! as the chat stream does, and records the PD first-token pair at the first
//! decode token: the chat, Messages, completions (single and batched) and
//! generate streams, and the openai-harmony pipeline's chat and Responses
//! streams.
//! Each case drives the route's `PrefillDecode` arm over scripted sides whose
//! dispatch instants lie a known time before the stream, under a thread-local
//! Prometheus recorder, and reads the rendered families.
//!
//! The request's wait ahead of that dispatch is recorded with the route's
//! time to first token. Each case's entry point runs under an entry stamp
//! taken a known time before the dispatch, as the edge runs a request; the
//! body is read outside it, as the connection reads it.
use std::{future::Future, time::Duration};

use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
};
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};
use tokio::task::JoinHandle;

use super::{
    super::*,
    eof::{
        chunk, complete, messages_spec, processor, runtime, scripted_stream, CharacterTokenizer,
    },
};
use crate::{
    middleware::RequestStarted,
    observability::metrics::{test_support::render_with_recorder, DISPATCH_WAIT_SECONDS},
    routers::grpc::{
        common::responses::streaming::ResponseStreamEventEmitter,
        context::ExecutionResult,
        harmony::{try_harmony_encoding, HarmonyStreamingProcessor},
    },
};

const MODEL: &str = "pd-anchor-test";
/// How long before the stream task the prefill side was dispatched.
const PREFILL_OFFSET: Duration = Duration::from_millis(300);
/// How long before the stream task the decode side was dispatched.
const DECODE_OFFSET: Duration = Duration::from_millis(100);
/// How long before its dispatch a case's request came in at the edge.
const DISPATCH_WAIT: Duration = Duration::from_millis(40);

/// The instant a case's stream clocks start at: the one worker's dispatch,
/// or the prefill dispatch of a pair. Read off the result's own fields, not
/// through the gateway's helper for it; a batch's case names its earliest
/// member itself.
fn dispatched_at(result: &ExecutionResult) -> Instant {
    match result {
        ExecutionResult::Single { dispatch_start, .. } => *dispatch_start,
        ExecutionResult::PrefillDecode { pd_timing, .. } => pd_timing.prefill_start,
        ExecutionResult::Batch { .. } | ExecutionResult::Embedding { .. } => {
            panic!("a case stamps one dispatch or one pair")
        }
    }
}

/// The entry stamp of a request that came in `DISPATCH_WAIT` before the
/// dispatch at `dispatched`: the stamp is taken off the dispatch instant
/// itself, so the wait is `DISPATCH_WAIT` to the nanosecond.
fn entered_before(dispatched: Instant) -> RequestStarted {
    let entered = dispatched
        .checked_sub(DISPATCH_WAIT)
        .expect("the process is older than the wait");
    RequestStarted::at(entered)
}

/// A route's entry point run under an entry stamp, as the edge runs a
/// request's handler.
trait Entered: Future + Sized {
    fn entered(self, at: RequestStarted) -> impl Future<Output = Self::Output> {
        at.scope(self)
    }
}

impl<F: Future> Entered for F {}

fn dispatch() -> context::DispatchMetadata {
    context::DispatchMetadata {
        request_id: "pd-anchor".to_string(),
        model: MODEL.to_string(),
        created: 1,
        weight_version: None,
    }
}

/// The dispatch instants of a pair whose prefill side was sent
/// `prefill_offset` ago and whose decode side `DECODE_OFFSET` ago.
fn pd_timing(prefill_offset: Duration) -> context::PdTiming {
    let now = Instant::now();
    let before = |offset: Duration| {
        now.checked_sub(offset)
            .expect("the process is older than the offset")
    };
    context::PdTiming {
        prefill_start: before(prefill_offset),
        decode_sent: before(DECODE_OFFSET),
        runtime: "test",
    }
}

/// A disaggregated execution result: a prefill side that completed, a decode
/// side streaming `decode_frames`, dispatched `prefill_offset` ago.
async fn pd_result(
    decode_frames: Vec<proto::GenerateResponse>,
    prefill_offset: Duration,
) -> (ExecutionResult, Vec<JoinHandle<()>>) {
    let (prefill, prefill_server) = scripted_stream(vec![complete(0, "stop")], "0").await;
    let (decode, decode_server) = scripted_stream(decode_frames, "0").await;
    let result = ExecutionResult::PrefillDecode {
        prefill,
        decode: Box::new(decode),
        prefill_guards: vec![None],
        pd_timing: pd_timing(prefill_offset),
        prefill_input_logprobs: None,
    };
    (result, vec![prefill_server, decode_server])
}

/// The decode side's frames for the character tokenizer: one chunk, then the
/// completion.
fn text_frames() -> Vec<proto::GenerateResponse> {
    vec![chunk(0, "hi"), complete(0, "stop")]
}

/// The decode side's frames of one openai-harmony turn: the final channel's
/// text in one chunk, then the completion.
fn harmony_frames() -> Vec<proto::GenerateResponse> {
    let token_ids = try_harmony_encoding()
        .expect("harmony encoding")
        .tokenizer()
        .encode_with_special_tokens("<|channel|>final<|message|>Hello<|return|>");
    vec![
        proto::GenerateResponse {
            response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
                index: 0,
                completion_tokens: token_ids.len() as u32,
                token_ids: token_ids.clone(),
                ..Default::default()
            })),
        },
        proto::GenerateResponse {
            response: Some(GenerationEvent::Complete(proto::GenerateComplete {
                index: 0,
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
                completion_tokens: token_ids.len() as u32,
                output_ids: token_ids,
                ..Default::default()
            })),
        },
    ]
}

fn chat_request() -> ChatCompletionRequest {
    serde_json::from_value(json!({"model": MODEL, "messages": [], "stream": true}))
        .expect("chat request")
}

/// A completion request over `prompt` (one string, or one per batched pair).
fn completion_spec(prompt: Value) -> CompletionResponseSpec {
    let request: CompletionRequest =
        serde_json::from_value(json!({"model": MODEL, "prompt": prompt, "stream": true}))
            .expect("completion request");
    CompletionResponseSpec::new(&request, Vec::new())
}

/// A generate request over one text, streamed.
fn generate_spec() -> GenerateResponseSpec {
    let request: GenerateRequest =
        serde_json::from_value(json!({"model": MODEL, "text": "test", "stream": true}))
            .expect("generate request");
    GenerateResponseSpec::from(&request)
}

fn source(endpoint: &'static str) -> utils::StreamSource {
    utils::StreamSource::new(None, &dispatch(), endpoint)
}

/// A route's SSE response read to its end, so that its stream task has
/// recorded its metrics.
async fn read_to_end(response: Response) {
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("the stream's body");
    let body = String::from_utf8(body.to_vec()).expect("UTF-8 SSE");
    assert_eq!(status, http::StatusCode::OK, "{body}");
    assert!(
        !body.contains("\"error\""),
        "the stream completes without an error event: {body}"
    );
}

/// Run `case` on a current-thread runtime under a thread-local recorder and
/// return the rendered `/metrics` text: the route's stream task runs on this
/// thread, so its records land in the recorder. `case` returns the scripted
/// sides' servers, stopped afterwards.
fn render_after<F, Fut>(case: F) -> String
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Vec<JoinHandle<()>>>,
{
    let rt = runtime();
    render_with_recorder(|| {
        for server in rt.block_on(case()) {
            server.abort();
        }
    })
}

/// Print one route's readings: the measured anchors are the evidence this
/// suite produces, next to its pass/fail.
#[expect(
    clippy::print_stderr,
    reason = "the measured anchors are the evidence this suite produces"
)]
fn report(route: &str, readings: &str) {
    eprintln!("pd anchor ({route}): {readings}");
}

/// The `_sum` or `_count` sample of `family` whose labels contain `label`.
fn sample(rendered: &str, family: &str, field: &str, label: &str) -> Option<f64> {
    let prefix = format!("{family}_{field}{{");
    rendered
        .lines()
        .find(|line| line.starts_with(&prefix) && line.contains(label))
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
}

/// The label set of `family`'s `_count` line that carries `label`.
fn count_labels<'a>(rendered: &'a str, family: &str, label: &str) -> Option<&'a str> {
    let prefix = format!("{family}_count{{");
    rendered
        .lines()
        .find(|line| line.starts_with(&prefix) && line.contains(label))
        .and_then(|line| line.strip_prefix(&prefix))
        .and_then(|rest| rest.split('}').next())
}

/// The route's stream metrics anchor at dispatch: one record, a
/// TTFT of at least the dispatch offset (a clock the task started itself
/// would read milliseconds) and a generation duration on the same base.
/// The request's wait ahead of that dispatch is recorded with the TTFT:
/// once, under the TTFT's labels, and it is the time from the entry stamp to
/// the dispatch.
fn assert_stream_anchor(rendered: &str, route: &str, endpoint: &'static str) {
    let label = format!("endpoint=\"{endpoint}\"");
    let count = sample(rendered, "smg_router_ttft_seconds", "count", &label);
    let ttft = sample(rendered, "smg_router_ttft_seconds", "sum", &label);
    let generation = sample(
        rendered,
        "smg_router_generation_duration_seconds",
        "sum",
        &label,
    );
    let (Some(count), Some(ttft), Some(generation)) = (count, ttft, generation) else {
        panic!("{endpoint}: the stream metrics are missing:\n{rendered}");
    };
    report(
        route,
        &format!("router ttft={ttft} s generation={generation} s records={count}"),
    );
    assert_eq!(count, 1.0, "{endpoint}: one TTFT record:\n{rendered}");
    let floor = PREFILL_OFFSET.as_secs_f64();
    assert!(
        ttft >= floor,
        "{endpoint}: TTFT {ttft} s anchors after dispatch ({floor} s before)"
    );
    assert!(
        generation >= ttft,
        "{endpoint}: generation duration {generation} s shares the TTFT's base ({ttft} s)"
    );
    let waits = sample(rendered, DISPATCH_WAIT_SECONDS, "count", &label);
    let wait = sample(rendered, DISPATCH_WAIT_SECONDS, "sum", &label);
    report(
        route,
        &format!("dispatch wait={wait:?} s records={waits:?}"),
    );
    assert_eq!(
        waits,
        Some(1.0),
        "{endpoint}: one dispatch wait, as there is one TTFT:\n{rendered}"
    );
    assert_eq!(
        count_labels(rendered, DISPATCH_WAIT_SECONDS, &label),
        count_labels(rendered, "smg_router_ttft_seconds", &label),
        "{endpoint}: the dispatch wait carries the TTFT's labels:\n{rendered}"
    );
    let wait = wait.expect("the dispatch wait");
    assert!(
        (wait - DISPATCH_WAIT.as_secs_f64()).abs() < 1e-9,
        "{endpoint}: dispatch wait {wait} s is the time from the entry stamp to the dispatch ({DISPATCH_WAIT:?})"
    );
}

/// The PD first-token pair was recorded `records` times against the dispatch
/// instants: the end-to-end values add up to at least `prefill_offsets`, the
/// decode phases to at least their offsets, and each end-to-end value exceeds
/// its decode phase.
fn assert_pd_pair(rendered: &str, route: &str, records: f64, prefill_offsets: Duration) {
    let label = "runtime=\"test\"";
    let count = |family: &str| sample(rendered, family, "count", label);
    let sum = |family: &str| sample(rendered, family, "sum", label);
    report(
        route,
        &format!(
            "pd ttft={:?} s decode ttft={:?} s records={:?}",
            sum("smg_pd_ttft_seconds"),
            sum("smg_pd_decode_ttft_seconds"),
            count("smg_pd_ttft_seconds")
        ),
    );
    assert_eq!(
        count("smg_pd_ttft_seconds"),
        Some(records),
        "the PD TTFT records:\n{rendered}"
    );
    assert_eq!(
        count("smg_pd_decode_ttft_seconds"),
        Some(records),
        "the PD decode TTFT records:\n{rendered}"
    );
    let ttft = sum("smg_pd_ttft_seconds").expect("the PD TTFT");
    let decode = sum("smg_pd_decode_ttft_seconds").expect("the PD decode TTFT");
    assert!(
        ttft >= prefill_offsets.as_secs_f64(),
        "PD TTFT {ttft} s over {records} records, prefill offsets {prefill_offsets:?}"
    );
    assert!(
        decode >= records * DECODE_OFFSET.as_secs_f64(),
        "PD decode TTFT {decode} s over {records} records, decode offset {DECODE_OFFSET:?}"
    );
    assert!(
        ttft > decode,
        "the end-to-end PD TTFT {ttft} s exceeds the decode phase {decode} s"
    );
}

/// The chat stream: the reference the other routes are held to.
#[test]
fn a_pd_chat_stream_anchors_at_the_prefill_dispatch() {
    let rendered = render_after(|| async {
        let (result, servers) = pd_result(text_frames(), PREFILL_OFFSET).await;
        let entered = entered_before(dispatched_at(&result));
        let response = Arc::new(processor(false))
            .process_streaming_response(
                result,
                ChatResponseSpec::from(&chat_request()),
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                false,
                None,
                None,
                source(metrics_labels::ENDPOINT_CHAT),
            )
            .entered(entered)
            .await;
        read_to_end(response).await;
        servers
    });
    assert_stream_anchor(&rendered, "chat", metrics_labels::ENDPOINT_CHAT);
    assert_pd_pair(&rendered, "chat", 1.0, PREFILL_OFFSET);
}

#[test]
fn a_pd_messages_stream_anchors_at_the_prefill_dispatch() {
    let rendered = render_after(|| async {
        let (result, servers) = pd_result(text_frames(), PREFILL_OFFSET).await;
        let entered = entered_before(dispatched_at(&result));
        let response = Arc::new(processor(false))
            .process_messages_streaming_response(
                result,
                messages_spec(false),
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                false,
                None,
                None,
                source(metrics_labels::ENDPOINT_MESSAGES),
            )
            .entered(entered)
            .await;
        read_to_end(response).await;
        servers
    });
    assert_stream_anchor(&rendered, "messages", metrics_labels::ENDPOINT_MESSAGES);
    assert_pd_pair(&rendered, "messages", 1.0, PREFILL_OFFSET);
}

#[test]
fn a_pd_completion_stream_anchors_at_the_prefill_dispatch() {
    let rendered = render_after(|| async {
        let (result, servers) = pd_result(text_frames(), PREFILL_OFFSET).await;
        let entered = entered_before(dispatched_at(&result));
        let response = Arc::new(processor(false))
            .process_completion_streaming_response(
                result,
                completion_spec(json!("test")),
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                None,
                None,
                source(metrics_labels::ENDPOINT_COMPLETIONS),
            )
            .entered(entered)
            .await;
        read_to_end(response).await;
        servers
    });
    assert_stream_anchor(
        &rendered,
        "completions",
        metrics_labels::ENDPOINT_COMPLETIONS,
    );
    assert_pd_pair(&rendered, "completions", 1.0, PREFILL_OFFSET);
}

/// A batched completion fans out into one pair per prompt; the request's
/// metrics anchor at the earliest prefill dispatch of them, and each pair
/// records its own PD first-token pair.
#[test]
fn a_batched_pd_completion_stream_anchors_at_its_earliest_prefill_dispatch() {
    let later = PREFILL_OFFSET - Duration::from_millis(100);
    let rendered = render_after(|| async {
        let (first, mut servers) = pd_result(text_frames(), PREFILL_OFFSET).await;
        let (second, more) = pd_result(text_frames(), later).await;
        servers.extend(more);
        // The first pair's prefill is the earliest of the batch.
        let entered = entered_before(dispatched_at(&first));
        let response = Arc::new(processor(false))
            .process_completion_streaming_response(
                ExecutionResult::Batch {
                    results: vec![first, second],
                },
                completion_spec(json!(["one", "two"])),
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                None,
                None,
                source(metrics_labels::ENDPOINT_COMPLETIONS),
            )
            .entered(entered)
            .await;
        read_to_end(response).await;
        servers
    });
    assert_stream_anchor(
        &rendered,
        "batched completions",
        metrics_labels::ENDPOINT_COMPLETIONS,
    );
    assert_pd_pair(
        &rendered,
        "batched completions",
        2.0,
        PREFILL_OFFSET + later,
    );
}

/// The generate stream of a pair (the one that carries the prefill side's
/// input logprobs) derives its anchor from the pair's prefill dispatch once
/// at its start; a stream anchored at its own start would read the
/// prefill time away.
#[test]
fn a_pd_generate_stream_anchors_at_the_prefill_dispatch() {
    let rendered = render_after(|| async {
        let (result, servers) = pd_result(text_frames(), PREFILL_OFFSET).await;
        let entered = entered_before(dispatched_at(&result));
        let response = Arc::new(processor(false))
            .process_streaming_generate(
                result,
                generate_spec(),
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                None,
                None,
                source(metrics_labels::ENDPOINT_GENERATE),
            )
            .entered(entered)
            .await;
        read_to_end(response).await;
        servers
    });
    assert_stream_anchor(&rendered, "generate", metrics_labels::ENDPOINT_GENERATE);
    assert_pd_pair(&rendered, "generate", 1.0, PREFILL_OFFSET);
}

/// The openai-harmony pipeline's chat stream.
#[test]
fn a_pd_harmony_chat_stream_anchors_at_the_prefill_dispatch() {
    let rendered = render_after(|| async {
        let (result, servers) = pd_result(harmony_frames(), PREFILL_OFFSET).await;
        let entered = entered_before(dispatched_at(&result));
        let response = Arc::new(HarmonyStreamingProcessor::new())
            .process_streaming_chat_response(
                result,
                Arc::new(chat_request()),
                dispatch(),
                Vec::new(),
                None,
                None,
                source(metrics_labels::ENDPOINT_CHAT),
            )
            .entered(entered)
            .await;
        read_to_end(response).await;
        servers
    });
    assert_stream_anchor(&rendered, "harmony chat", metrics_labels::ENDPOINT_CHAT);
    assert_pd_pair(&rendered, "harmony chat", 1.0, PREFILL_OFFSET);
}

/// The openai-harmony pipeline's Responses iteration records no stream timing
/// of its own; through a pair it records the PD first-token pair at its first
/// decode frame. With no time to first token there is no dispatch wait either,
/// though the iteration runs under an entry stamp.
#[test]
fn a_pd_harmony_responses_iteration_records_the_pd_pair_at_its_first_decode_frame() {
    let rendered = render_after(|| async {
        let (result, servers) = pd_result(harmony_frames(), PREFILL_OFFSET).await;
        let entered = entered_before(dispatched_at(&result));
        let mut emitter =
            ResponseStreamEventEmitter::new("resp_pd-anchor".to_string(), MODEL.to_string(), 1);
        let (tx, mut rx) = sse_channel();
        // The events are read as they come, so the processor never waits on
        // the channel.
        let drain = async { while rx.recv().await.is_some() {} };
        let iteration = async {
            let result = HarmonyStreamingProcessor::process_responses_iteration_stream(
                result,
                &mut emitter,
                &tx,
                None,
                None,
                &[],
                None,
            )
            .await;
            drop(tx);
            result
        };
        let (iteration, ()) = tokio::join!(iteration.entered(entered), drain);
        iteration.expect("the streamed iteration");
        servers
    });
    assert_pd_pair(&rendered, "harmony responses", 1.0, PREFILL_OFFSET);
    for family in ["smg_router_ttft_seconds", DISPATCH_WAIT_SECONDS] {
        assert_eq!(
            sample(&rendered, family, "count", "endpoint="),
            None,
            "{family}:\n{rendered}"
        );
    }
}

/// A Single RPC can have its first chunk buffered when the stream starts.
#[test]
fn a_single_harmony_chat_stream_anchors_at_dispatch() {
    let rendered = render_after(|| async {
        let (stream, server) = scripted_stream(harmony_frames(), "0").await;
        let dispatch_start = pd_timing(PREFILL_OFFSET).prefill_start;
        let entered = entered_before(dispatch_start);
        let response = Arc::new(HarmonyStreamingProcessor::new())
            .process_streaming_chat_response(
                ExecutionResult::Single {
                    stream,
                    dispatch_start,
                },
                Arc::new(chat_request()),
                dispatch(),
                Vec::new(),
                None,
                None,
                source(metrics_labels::ENDPOINT_CHAT),
            )
            .entered(entered)
            .await;
        read_to_end(response).await;
        vec![server]
    });
    assert_stream_anchor(
        &rendered,
        "single harmony chat",
        metrics_labels::ENDPOINT_CHAT,
    );
}

/// The earliest dispatch belongs to the second prompt, not the first one.
#[test]
fn a_batched_single_completion_stream_anchors_at_its_earliest_dispatch() {
    let rendered = render_after(|| async {
        let (first, first_server) = scripted_stream(text_frames(), "0").await;
        let (second, second_server) = scripted_stream(text_frames(), "0").await;
        let timing = pd_timing(PREFILL_OFFSET);
        let entered = entered_before(timing.prefill_start);
        let response = Arc::new(processor(false))
            .process_completion_streaming_response(
                ExecutionResult::Batch {
                    results: vec![
                        ExecutionResult::Single {
                            stream: first,
                            dispatch_start: timing.decode_sent,
                        },
                        ExecutionResult::Single {
                            stream: second,
                            dispatch_start: timing.prefill_start,
                        },
                    ],
                },
                completion_spec(json!(["one", "two"])),
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                None,
                None,
                source(metrics_labels::ENDPOINT_COMPLETIONS),
            )
            .entered(entered)
            .await;
        read_to_end(response).await;
        vec![first_server, second_server]
    });
    assert_stream_anchor(
        &rendered,
        "batched single completions",
        metrics_labels::ENDPOINT_COMPLETIONS,
    );
}

/// A single dispatch's result: one stream of `frames`, dispatched
/// `PREFILL_OFFSET` ago.
async fn single_result(
    frames: Vec<proto::GenerateResponse>,
) -> (ExecutionResult, Vec<JoinHandle<()>>) {
    let (stream, server) = scripted_stream(frames, "0").await;
    let result = ExecutionResult::Single {
        stream,
        dispatch_start: pd_timing(PREFILL_OFFSET).prefill_start,
    };
    (result, vec![server])
}

/// One worker's chat stream records the request's wait ahead of its dispatch
/// with its TTFT, as a pair's does.
#[test]
fn a_single_chat_stream_records_its_dispatch_wait_with_its_ttft() {
    let rendered = render_after(|| async {
        let (result, servers) = single_result(text_frames()).await;
        let entered = entered_before(dispatched_at(&result));
        let response = Arc::new(processor(false))
            .process_streaming_response(
                result,
                ChatResponseSpec::from(&chat_request()),
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                false,
                None,
                None,
                source(metrics_labels::ENDPOINT_CHAT),
            )
            .entered(entered)
            .await;
        read_to_end(response).await;
        servers
    });
    assert_stream_anchor(&rendered, "single chat", metrics_labels::ENDPOINT_CHAT);
}

#[test]
fn a_single_messages_stream_records_its_dispatch_wait_with_its_ttft() {
    let rendered = render_after(|| async {
        let (result, servers) = single_result(text_frames()).await;
        let entered = entered_before(dispatched_at(&result));
        let response = Arc::new(processor(false))
            .process_messages_streaming_response(
                result,
                messages_spec(false),
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                false,
                None,
                None,
                source(metrics_labels::ENDPOINT_MESSAGES),
            )
            .entered(entered)
            .await;
        read_to_end(response).await;
        servers
    });
    assert_stream_anchor(
        &rendered,
        "single messages",
        metrics_labels::ENDPOINT_MESSAGES,
    );
}

#[test]
fn a_single_generate_stream_records_its_dispatch_wait_with_its_ttft() {
    let rendered = render_after(|| async {
        let (result, servers) = single_result(text_frames()).await;
        let entered = entered_before(dispatched_at(&result));
        let response = Arc::new(processor(false))
            .process_streaming_generate(
                result,
                generate_spec(),
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                None,
                None,
                source(metrics_labels::ENDPOINT_GENERATE),
            )
            .entered(entered)
            .await;
        read_to_end(response).await;
        servers
    });
    assert_stream_anchor(
        &rendered,
        "single generate",
        metrics_labels::ENDPOINT_GENERATE,
    );
}

/// A stream whose entry point runs outside any entry stamp, as a pipeline run
/// bare does, has no entry to count from: it records its TTFT and no dispatch
/// wait.
#[test]
fn a_stream_outside_an_entry_stamp_records_its_ttft_and_no_dispatch_wait() {
    let rendered = render_after(|| async {
        let (result, servers) = single_result(text_frames()).await;
        let response = Arc::new(processor(false))
            .process_streaming_response(
                result,
                ChatResponseSpec::from(&chat_request()),
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                false,
                None,
                None,
                source(metrics_labels::ENDPOINT_CHAT),
            )
            .await;
        read_to_end(response).await;
        servers
    });
    let label = format!("endpoint=\"{}\"", metrics_labels::ENDPOINT_CHAT);
    assert_eq!(
        sample(&rendered, "smg_router_ttft_seconds", "count", &label),
        Some(1.0),
        "{rendered}"
    );
    assert_eq!(
        sample(&rendered, DISPATCH_WAIT_SECONDS, "count", &label),
        None,
        "{rendered}"
    );
}

/// The stamp is read where the stream's entry point runs, not where its
/// first token is recorded: a body read under a later stamp of its own
/// changes nothing. The wait stays the time from the entry point's stamp to
/// the dispatch; one read at the record would count from the later stamp and
/// read zero.
#[test]
fn the_dispatch_wait_is_read_at_the_stream_s_entry_not_where_it_is_recorded() {
    let rendered = render_after(|| async {
        let (result, servers) = single_result(text_frames()).await;
        let entered = entered_before(dispatched_at(&result));
        let response = Arc::new(processor(false))
            .process_streaming_response(
                result,
                ChatResponseSpec::from(&chat_request()),
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                false,
                None,
                None,
                source(metrics_labels::ENDPOINT_CHAT),
            )
            .entered(entered)
            .await;
        read_to_end(response).entered(RequestStarted::now()).await;
        servers
    });
    assert_stream_anchor(&rendered, "single chat", metrics_labels::ENDPOINT_CHAT);
}
