//! Regular gRPC latency starts at dispatch, including a worker that holds
//! response headers until its first chunk: every streaming route and the
//! collected ones, the batched completion stream, the Harmony pipeline, a
//! dispatch that answers on its retry, and the client that leaves after the
//! first chunk (which records nothing). Each case captures metrics on a
//! current-thread runtime, as the PD TTFT regressions do.
//!
//! The request's wait ahead of that dispatch
//! (`smg_router_dispatch_wait_seconds`) is recorded once per answer, under
//! the labels of its other records, counted from the entry stamp the
//! request's handler runs under, so a failed attempt is inside it: with a
//! streamed route's time to first token, and alone for a collected answer,
//! which has no first token. The body is read outside the stamp, as the
//! connection reads it.
//!
//! A Responses tool loop dispatches one chat generation per turn (the
//! streamed loop from a task of its own). Every turn records a wait, a
//! streamed turn with its TTFT: the first from the request's entry stamp, a
//! later one from where that turn starts.

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};

use common::{
    decode_stream::HeldHeadersEngine,
    engine_answers::{VllmAnswers, VllmGenerateStream},
    grpc_harness::{
        build_grpc_router, grpc_router, grpc_router_config, grpc_test_context_with_tokenizer,
        grpc_worker, grpc_workers, serve_engine,
    },
    metrics_capture::{self, render_after},
    mock_mcp_server::MockMCPServer,
    scripted_tokenizer::ScriptedTokenizer,
};
use http_body_util::BodyExt;
use llm_tokenizer::{
    traits::{MediaHandling, RendererCapabilities},
    MockTokenizer,
};
use openai_protocol::{
    model_card::ModelCard,
    transcription::{AudioFile, TranscriptionRequest},
};
use smg::{
    app_context::AppContext,
    config::{RetryConfig, RouterConfigBuilder},
    middleware::{RequestStarted, TenantRequestMeta},
    routers::RouterTrait,
    tenant::TenantKey,
};
use smg_grpc_client::vllm_proto as proto;
use smg_test_harness as common;
use smg_worker::identity::basic::{RuntimeType, Worker, WorkerType};
use tonic::{Request, Response, Status};

const MODEL: &str = "regular-dispatch-ttft-test-model";
/// A model id the router reads as a Harmony model: its chat runs the Harmony pipeline.
const HARMONY_MODEL: &str = "gpt-oss-dispatch-ttft-test-model";
/// A model id the one transcription family recognises: a transcription is a
/// chat-shaped generation to the engine.
const ASR_MODEL: &str = "qwen3-asr-dispatch-ttft-test-model";
/// An alias a case's workers give `ASR_MODEL`.
const ASR_ALIAS: &str = "dispatch-ttft-test-transcription-alias";
/// The audio of every transcription request here.
const CLIP: &[u8] = b"not real audio";
/// The ids the transcription case's engine answers, words of its tokenizer.
const TRANSCRIPT_IDS: [u32; 3] = [100, 101, 102];
/// An id the transcription case's tokenizer decodes to a blank.
const BLANK_ID: u32 = 103;
const OUTPUT_TOKENS: u32 = 3;
const HELD_HEADERS_DELAY: Duration = Duration::from_millis(300);
const POST_FIRST_DELAY: Duration = Duration::from_millis(600);
/// The most a first token may carry over the hold on a loaded machine: the
/// first chunk arrives with the headers, so the in-stream delay
/// (`POST_FIRST_DELAY`, longer than this slack) is never part of it. The
/// worst first-token stall measured under a heavy parallel build load was
/// 283 ms; at a moderate load the stalls stay under 100 ms.
const FIRST_TOKEN_SLACK: Duration = Duration::from_millis(500);
const _: () = assert!(FIRST_TOKEN_SLACK.as_millis() < POST_FIRST_DELAY.as_millis());
/// What a worker that fails its attempt spends before failing it: longer
/// than the slack, so the retry case tells a TTFT anchored at the answering
/// attempt from one anchored at the request's entry.
const FAILED_ATTEMPT_DELAY: Duration = Duration::from_millis(700);
const _: () = assert!(FIRST_TOKEN_SLACK.as_millis() < FAILED_ATTEMPT_DELAY.as_millis());
/// How long before its handler runs a stamped request entered at the edge:
/// the dispatch wait must reach it, so a wait counted from anywhere at or
/// after the dispatch fails its lower bound.
const ENTRY_LEAD: Duration = Duration::from_millis(100);
const DISPATCH_WAIT: &str = "smg_router_dispatch_wait_seconds";

/// The engine of these cases: the RPC await absorbs the hold, with the first
/// chunk already buffered when it returns, like a Python servicer holding
/// headers; the rest of the answer follows after `POST_FIRST_DELAY`.
fn held_headers_engine(output_ids: Option<Vec<u32>>) -> HeldHeadersEngine {
    HeldHeadersEngine {
        hold: HELD_HEADERS_DELAY,
        first_chunk_buffered: true,
        in_stream_delay: POST_FIRST_DELAY,
        output_tokens: OUTPUT_TOKENS,
        output_ids,
    }
}

/// A regular gRPC router over the workers of `ports`, all serving `model`.
async fn router_over(
    ports: &[u16],
    model: &str,
    retries: Option<RetryConfig>,
) -> Box<dyn RouterTrait> {
    let mut config = grpc_router_config().regular_mode(vec![]).random_policy();
    if let Some(retries) = retries {
        config = config.retry_config(retries);
    }
    let workers: Vec<(u16, WorkerType)> = ports
        .iter()
        .map(|&port| (port, WorkerType::Regular))
        .collect();
    build_grpc_router(
        config.build_unchecked(),
        model,
        grpc_workers(&workers, RuntimeType::Vllm, model),
    )
    .await
}

async fn regular_router() -> Box<dyn RouterTrait> {
    let port = serve_engine(held_headers_engine(None).into_server()).await;
    router_over(&[port], MODEL, None).await
}

/// A regular router serving `ASR_MODEL` over one held-headers engine.
async fn transcription_router() -> Box<dyn RouterTrait> {
    let engine = held_headers_engine(Some(TRANSCRIPT_IDS.to_vec()));
    let port = serve_engine(engine.into_server()).await;
    transcription_router_over(&[port], None).await
}

/// The test context of a regular router under `config` that serves
/// `ASR_MODEL`. Its mock tokenizer renders media as template text, so the
/// audio turn needs no media processing, and it names the engines' ids, so
/// the transcript is text.
fn transcription_context(
    config: RouterConfigBuilder,
) -> impl std::future::Future<Output = Arc<AppContext>> {
    let tokenizer = MockTokenizer::new()
        .with_renderer_capabilities(RendererCapabilities {
            media_handling: MediaHandling::TemplateText,
            ..RendererCapabilities::default()
        })
        .with_tokens(&[
            ("spoken", TRANSCRIPT_IDS[0]),
            ("words", TRANSCRIPT_IDS[1]),
            ("here", TRANSCRIPT_IDS[2]),
            (" ", BLANK_ID),
        ]);
    grpc_test_context_with_tokenizer(config.build_unchecked(), ASR_MODEL, Arc::new(tokenizer))
}

/// A regular router serving `ASR_MODEL` over the engines of `ports`.
async fn transcription_router_over(
    ports: &[u16],
    retries: Option<RetryConfig>,
) -> Box<dyn RouterTrait> {
    let mut config = grpc_router_config().regular_mode(vec![]).random_policy();
    if let Some(retries) = retries {
        config = config.retry_config(retries);
    }
    let app_context = transcription_context(config).await;
    let workers: Vec<(u16, WorkerType)> = ports
        .iter()
        .map(|&port| (port, WorkerType::Regular))
        .collect();
    grpc_router(
        &app_context,
        grpc_workers(&workers, RuntimeType::Vllm, ASR_MODEL),
    )
    .await
}

#[derive(Clone, Copy)]
enum Route {
    Chat,
    Messages,
    Generate,
    Completions,
    BatchedCompletions,
    Transcriptions,
}

impl Route {
    fn endpoint(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Messages => "messages",
            Self::Generate => "generate",
            Self::Completions | Self::BatchedCompletions => "completions",
            Self::Transcriptions => "audio_transcriptions",
        }
    }

    /// The tokens the route's answer carries: every prompt of a batch answers
    /// `OUTPUT_TOKENS`.
    fn output_tokens(self) -> u32 {
        match self {
            Self::BatchedCompletions => 2 * OUTPUT_TOKENS,
            _ => OUTPUT_TOKENS,
        }
    }

    /// The model the route's request names and its router serves.
    fn model(self) -> &'static str {
        match self {
            Self::Transcriptions => ASR_MODEL,
            _ => MODEL,
        }
    }

    /// The router of the route's case, over one held-headers engine.
    async fn router(self) -> Box<dyn RouterTrait> {
        match self {
            Self::Transcriptions => transcription_router().await,
            _ => regular_router().await,
        }
    }
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("regular-dispatch-ttft-test-tenant"))
}

fn chat_request(model: &str, streaming: bool) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": streaming,
    })
}

/// The answer of one transcription request for `model`, its body unread.
async fn transcribe(router: &dyn RouterTrait, model: &str) -> axum::response::Response {
    let request = TranscriptionRequest {
        model: model.to_string(),
        ..Default::default()
    };
    let audio = AudioFile {
        bytes: bytes::Bytes::from_static(CLIP),
        file_name: "clip.wav".to_string(),
        content_type: Some("audio/wav".to_string()),
    };
    router
        .route_audio_transcriptions(None, &tenant(), &request, audio, model)
        .await
}

/// The answer of one request on `route`: what the route's handler returns,
/// its body unread.
#[expect(clippy::expect_used, reason = "test helper")]
async fn route_request(
    router: &dyn RouterTrait,
    route: Route,
    streaming: bool,
) -> axum::response::Response {
    let tenant = tenant();
    let model = route.model();
    match route {
        Route::Chat => {
            let request =
                serde_json::from_value(chat_request(model, streaming)).expect("chat request");
            router
                .route_chat(None, &tenant, Arc::new(request), model)
                .await
        }
        Route::Transcriptions => transcribe(router, model).await,
        Route::Messages => {
            let request = serde_json::from_value(serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "Hello world"}],
                "max_tokens": 8,
                "stream": streaming,
            }))
            .expect("Messages request");
            router
                .route_messages(None, &tenant, Arc::new(request), model)
                .await
        }
        Route::Generate => {
            let request = serde_json::from_value(serde_json::json!({
                "model": model,
                "text": "Hello world",
                "sampling_params": {"max_new_tokens": 8},
                "stream": streaming,
            }))
            .expect("generate request");
            router.route_generate(None, &tenant, request, model).await
        }
        Route::Completions | Route::BatchedCompletions => {
            let prompt = if matches!(route, Route::BatchedCompletions) {
                serde_json::json!(["Hello world", "Hello again"])
            } else {
                serde_json::json!("Hello world")
            };
            let request = serde_json::from_value(serde_json::json!({
                "model": model,
                "prompt": prompt,
                "max_tokens": 8,
                "stream": streaming,
            }))
            .expect("completions request");
            router.route_completion(None, &tenant, request, model).await
        }
    }
}

/// One request on `route`, read to its end: the status and the body are
/// checked, and a generate answer's `e2e_latency` (its processor-entry clock)
/// is returned for the comparison with the dispatch-anchored record.
async fn request(router: &dyn RouterTrait, route: Route, streaming: bool) -> Option<f64> {
    request_entered(router, route, streaming, None).await
}

/// [`request`], its handler run under the entry stamp `entered` when there is
/// one, as the edge runs every request; the body is read outside it either
/// way, as the connection reads it.
#[expect(clippy::expect_used, reason = "test helper")]
async fn request_entered(
    router: &dyn RouterTrait,
    route: Route,
    streaming: bool,
    entered: Option<RequestStarted>,
) -> Option<f64> {
    let handler = route_request(router, route, streaming);
    let response = match entered {
        Some(entered) => entered.scope(handler).await,
        None => handler.await,
    };
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    let body = String::from_utf8(bytes.to_vec()).expect("UTF-8 body");
    assert_eq!(status, http::StatusCode::OK, "{body}");
    if streaming {
        let ended = match route {
            Route::Messages => body.contains("event: message_stop"),
            _ => body.contains("data: [DONE]"),
        };
        assert!(ended, "{body}");
    } else {
        let response: serde_json::Value = serde_json::from_str(&body).expect("JSON response");
        // A generate answer is one object, or one per prompt.
        let answer = response.get(0).unwrap_or(&response);
        let output_tokens = match route {
            Route::Messages => response["usage"]["output_tokens"].as_u64(),
            Route::Generate => answer["meta_info"]["completion_tokens"].as_u64(),
            Route::Chat | Route::Completions | Route::BatchedCompletions => {
                response["usage"]["completion_tokens"].as_u64()
            }
            Route::Transcriptions => {
                assert!(response["text"].is_string(), "{body}");
                return None;
            }
        };
        assert_eq!(
            output_tokens,
            Some(u64::from(route.output_tokens())),
            "{body}"
        );
        if matches!(route, Route::Generate) {
            return Some(
                answer["meta_info"]["e2e_latency"]
                    .as_f64()
                    .expect("the generate answer carries e2e_latency"),
            );
        }
    }
    if matches!(route, Route::Generate) && streaming {
        return Some(
            body.lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|data| *data != "[DONE]")
                .map(|data| {
                    serde_json::from_str::<serde_json::Value>(data).expect("generate event")
                })
                .find_map(|event| event["meta_info"]["e2e_latency"].as_f64())
                .expect("the terminal generate event carries e2e_latency"),
        );
    }
    None
}

/// The regular router's `_sum` or `_count` sample of `family` on `endpoint`.
fn sample(rendered: &str, family: &str, field: &str, endpoint: &str) -> Option<f64> {
    let endpoint = format!("endpoint=\"{endpoint}\"");
    metrics_capture::sample(
        rendered,
        family,
        field,
        &["backend_type=\"regular\"", endpoint.as_str()],
    )
}

#[expect(clippy::expect_used, reason = "test helper")]
fn observation(rendered: &str, family: &str, endpoint: &str) -> f64 {
    assert_eq!(
        sample(rendered, family, "count", endpoint),
        Some(1.0),
        "{endpoint}: one {family} observation:\n{rendered}"
    );
    sample(rendered, family, "sum", endpoint).expect("histogram sum")
}

/// The stream records of `endpoint`, read against the held-header floor:
/// TTFT carries the hold and not the in-stream delay, generation shares its
/// base, and TPOT is the mean gap over `tokens - 1` intervals.
fn assert_stream_records(rendered: &str, endpoint: &str, generation: f64, tokens: u32) {
    let ttft = observation(rendered, "smg_router_ttft_seconds", endpoint);
    let tpot = observation(rendered, "smg_router_tpot_seconds", endpoint);
    assert!(
        ttft >= HELD_HEADERS_DELAY.as_secs_f64(),
        "{endpoint}: TTFT {ttft} must include the held-header delay"
    );
    let ceiling = (HELD_HEADERS_DELAY + FIRST_TOKEN_SLACK).as_secs_f64();
    assert!(
        ttft < ceiling,
        "{endpoint}: TTFT {ttft} past the hold plus slack ({ceiling}): the first chunk arrived with the headers"
    );
    assert!(
        generation >= ttft,
        "{endpoint}: generation {generation} < TTFT {ttft}"
    );
    let expected = (generation - ttft) / f64::from(tokens - 1);
    assert!(
        (tpot - expected).abs() < 1e-6,
        "{endpoint}: TPOT {tpot} != (generation {generation} - TTFT {ttft}) / {}",
        tokens - 1
    );
    assert!(
        tpot.is_finite() && tpot >= 0.0,
        "{endpoint}: TPOT {tpot} must be finite and nonnegative"
    );
}

#[expect(
    clippy::print_stderr,
    reason = "report the measured intervals alongside the regression result"
)]
fn held_headers_are_included(route: Route, streaming: bool) {
    let mut e2e_latency = None;
    let rendered = render_after(|| async {
        let router = route.router().await;
        e2e_latency = request(router.as_ref(), route, streaming).await;
    });
    let endpoint = route.endpoint();
    let generation = observation(
        &rendered,
        "smg_router_generation_duration_seconds",
        endpoint,
    );
    let ttft = sample(&rendered, "smg_router_ttft_seconds", "sum", endpoint);
    let tpot = sample(&rendered, "smg_router_tpot_seconds", "sum", endpoint);
    eprintln!(
        "regular {endpoint} (stream={streaming}): ttft={ttft:?} generation={generation} tpot={tpot:?}"
    );
    assert!(
        generation >= (HELD_HEADERS_DELAY + POST_FIRST_DELAY).as_secs_f64(),
        "{endpoint}: generation {generation} must include the held-header and decode delays"
    );
    if let Some(e2e_latency) = e2e_latency {
        assert!(
            generation - e2e_latency >= HELD_HEADERS_DELAY.as_secs_f64(),
            "generate: response e2e_latency {e2e_latency} keeps its processor-entry clock, \
             while Prometheus generation {generation} includes the held-header delay"
        );
    }
    if streaming {
        assert_stream_records(&rendered, endpoint, generation, route.output_tokens());
    } else {
        // A collected response shows no first token: no TTFT and no TPOT.
        assert!(ttft.is_none(), "{endpoint}: collected TTFT {ttft:?}");
        assert!(tpot.is_none(), "{endpoint}: collected TPOT {tpot:?}");
    }
    // The request ran outside an entry stamp: its other records are made
    // alone, the dispatch wait has no start to count from.
    assert!(
        sample(&rendered, DISPATCH_WAIT, "count", endpoint).is_none(),
        "{endpoint}: a dispatch wait outside an entry stamp:\n{rendered}"
    );
}

#[test]
fn streaming_chat_includes_the_held_header_delay() {
    held_headers_are_included(Route::Chat, true);
}

#[test]
fn streaming_messages_includes_the_held_header_delay() {
    held_headers_are_included(Route::Messages, true);
}

#[test]
fn streaming_generate_includes_the_held_header_delay() {
    held_headers_are_included(Route::Generate, true);
}

#[test]
fn streaming_completions_includes_the_held_header_delay() {
    held_headers_are_included(Route::Completions, true);
}

#[test]
fn collected_chat_includes_the_held_header_delay() {
    held_headers_are_included(Route::Chat, false);
}

#[test]
fn collected_messages_includes_the_held_header_delay() {
    held_headers_are_included(Route::Messages, false);
}

#[test]
fn collected_completions_includes_the_held_header_delay() {
    held_headers_are_included(Route::Completions, false);
}

#[test]
fn collected_batched_completions_includes_the_held_header_delay() {
    held_headers_are_included(Route::BatchedCompletions, false);
}

/// A batched completion stream anchors at its prompts' dispatch too, its
/// TPOT read over the batch's tokens.
#[test]
fn streaming_batched_completions_includes_the_held_header_delay() {
    held_headers_are_included(Route::BatchedCompletions, true);
}

/// A collected generate answer keeps its own `e2e_latency` on the
/// processor-entry clock while its Prometheus generation includes the hold.
#[test]
fn collected_generate_includes_the_held_header_delay() {
    held_headers_are_included(Route::Generate, false);
}

/// A transcription decodes through the same processor: its generation record
/// includes the hold.
#[test]
fn collected_transcription_includes_the_held_header_delay() {
    held_headers_are_included(Route::Transcriptions, false);
}

/// The entry stamp of a request that came in at the edge `ENTRY_LEAD` ago.
fn entered_a_lead_ago() -> RequestStarted {
    RequestStarted::at(std::time::Instant::now() - ENTRY_LEAD)
}

/// The label set of `family`'s `_count` series whose labels hold every
/// fragment of `labels`.
fn count_labels<'a>(rendered: &'a str, family: &str, labels: &[&str]) -> Option<&'a str> {
    let prefix = format!("{family}_count{{");
    rendered
        .lines()
        .find(|line| line.starts_with(&prefix) && labels.iter().all(|label| line.contains(label)))
        .and_then(|line| line.strip_prefix(&prefix))
        .and_then(|rest| rest.split('}').next())
}

/// The dispatch wait of the one stamped stream whose series carry `labels`,
/// in seconds: recorded once, as its TTFT is, and under the TTFT's labels.
fn dispatch_wait_beside_the_ttft(rendered: &str, labels: &[&str]) -> f64 {
    dispatch_waits_beside_the_ttfts(rendered, labels, 1)
}

/// The dispatch waits of the `streams` stamped streams whose series carry
/// `labels`, summed, in seconds: one per stream, as the TTFT has, and under
/// the TTFT's labels.
#[expect(clippy::expect_used, reason = "test helper")]
fn dispatch_waits_beside_the_ttfts(rendered: &str, labels: &[&str], streams: u32) -> f64 {
    for family in [DISPATCH_WAIT, "smg_router_ttft_seconds"] {
        assert_eq!(
            metrics_capture::sample(rendered, family, "count", labels),
            Some(f64::from(streams)),
            "{labels:?}: {streams} {family} observation(s):\n{rendered}"
        );
    }
    assert_eq!(
        count_labels(rendered, DISPATCH_WAIT, labels),
        count_labels(rendered, "smg_router_ttft_seconds", labels),
        "{labels:?}: the dispatch wait carries the TTFT's labels:\n{rendered}"
    );
    metrics_capture::sample(rendered, DISPATCH_WAIT, "sum", labels).expect("histogram sum")
}

/// A request whose handler runs under an entry stamp taken `ENTRY_LEAD` ago.
/// A streamed answer records its dispatch wait once, with its TTFT and under
/// its labels: the wait reaches the stamp's lead, and the two add up to the
/// time to first token from the request's entry, the worker's hold included.
/// A collected answer records its wait once too, under its generation
/// record's labels and with no first token: the wait reaches the stamp's lead
/// and ends where the generation clock starts, so the two add up to no more
/// than the request's own time since its entry. A wait read after the
/// worker's hold, or where the answer is built, counts that time twice and
/// passes it.
#[expect(
    clippy::print_stderr,
    reason = "report the measured intervals alongside the regression result"
)]
fn under_an_entry_stamp(route: Route, streaming: bool) {
    let mut since_entry = Duration::ZERO;
    let rendered = render_after(|| async {
        let router = route.router().await;
        let entered = std::time::Instant::now() - ENTRY_LEAD;
        request_entered(
            router.as_ref(),
            route,
            streaming,
            Some(RequestStarted::at(entered)),
        )
        .await;
        since_entry = entered.elapsed();
    });
    let endpoint = route.endpoint();
    let endpoint_label = format!("endpoint=\"{endpoint}\"");
    let labels = ["backend_type=\"regular\"", endpoint_label.as_str()];
    if !streaming {
        assert!(
            sample(&rendered, "smg_router_ttft_seconds", "count", endpoint).is_none(),
            "{endpoint}: a collected answer recorded a first token:\n{rendered}"
        );
        let generation_family = "smg_router_generation_duration_seconds";
        let wait = observation(&rendered, DISPATCH_WAIT, endpoint);
        let generation = observation(&rendered, generation_family, endpoint);
        assert_eq!(
            count_labels(&rendered, DISPATCH_WAIT, &labels),
            count_labels(&rendered, generation_family, &labels),
            "{endpoint}: the dispatch wait carries the generation record's labels:\n{rendered}"
        );
        let since_entry = since_entry.as_secs_f64();
        eprintln!(
            "regular {endpoint} (collected): dispatch wait={wait} generation={generation} since entry={since_entry}"
        );
        assert!(
            wait >= ENTRY_LEAD.as_secs_f64(),
            "{endpoint}: dispatch wait {wait} runs from the entry stamp ({ENTRY_LEAD:?} before the handler)"
        );
        assert!(
            wait + generation <= since_entry,
            "{endpoint}: dispatch wait {wait} ends where the generation clock ({generation}) starts: the two fit in the request's time since its entry ({since_entry})"
        );
        return;
    }
    let wait = dispatch_wait_beside_the_ttft(&rendered, &labels);
    let ttft = observation(&rendered, "smg_router_ttft_seconds", endpoint);
    eprintln!("regular {endpoint}: dispatch wait={wait} ttft={ttft}");
    assert!(
        wait >= ENTRY_LEAD.as_secs_f64(),
        "{endpoint}: dispatch wait {wait} runs from the entry stamp ({ENTRY_LEAD:?} before the handler)"
    );
    let ceiling = (ENTRY_LEAD + FIRST_TOKEN_SLACK).as_secs_f64();
    assert!(
        wait < ceiling,
        "{endpoint}: dispatch wait {wait} past the stamp's lead plus slack ({ceiling})"
    );
    let floor = (ENTRY_LEAD + HELD_HEADERS_DELAY).as_secs_f64();
    assert!(
        wait + ttft >= floor,
        "{endpoint}: dispatch wait {wait} and TTFT {ttft} add up to the time to first token from the request's entry ({floor})"
    );
}

#[test]
fn streaming_chat_records_its_dispatch_wait_with_its_ttft() {
    under_an_entry_stamp(Route::Chat, true);
}

#[test]
fn streaming_messages_records_its_dispatch_wait_with_its_ttft() {
    under_an_entry_stamp(Route::Messages, true);
}

#[test]
fn streaming_generate_records_its_dispatch_wait_with_its_ttft() {
    under_an_entry_stamp(Route::Generate, true);
}

#[test]
fn streaming_completions_records_its_dispatch_wait_with_its_ttft() {
    under_an_entry_stamp(Route::Completions, true);
}

/// A batched completion stream has one TTFT, anchored at its earliest
/// dispatch, and one dispatch wait ahead of it.
#[test]
fn streaming_batched_completions_records_one_dispatch_wait() {
    under_an_entry_stamp(Route::BatchedCompletions, true);
}

#[test]
fn collected_chat_records_its_dispatch_wait_and_no_ttft() {
    under_an_entry_stamp(Route::Chat, false);
}

#[test]
fn collected_messages_records_its_dispatch_wait_and_no_ttft() {
    under_an_entry_stamp(Route::Messages, false);
}

#[test]
fn collected_generate_records_its_dispatch_wait_and_no_ttft() {
    under_an_entry_stamp(Route::Generate, false);
}

#[test]
fn collected_completions_records_its_dispatch_wait_and_no_ttft() {
    under_an_entry_stamp(Route::Completions, false);
}

/// A collected batch has one wait, ahead of its earliest dispatch.
#[test]
fn collected_batched_completions_records_one_dispatch_wait() {
    under_an_entry_stamp(Route::BatchedCompletions, false);
}

/// A transcription decodes through the generate processor and records its
/// wait under its own endpoint label.
#[test]
fn collected_transcription_records_its_dispatch_wait_and_no_ttft() {
    under_an_entry_stamp(Route::Transcriptions, false);
}

/// The ids of one Harmony final-channel turn, as the engine streams them.
#[expect(clippy::expect_used, reason = "test helper")]
fn harmony_output_ids() -> Vec<u32> {
    openai_harmony::load_harmony_encoding(openai_harmony::HarmonyEncodingName::HarmonyGptOss)
        .expect("the Harmony encoding")
        .tokenizer()
        .encode_with_special_tokens("<|channel|>final<|message|>Hello<|return|>")
}

/// A Harmony collected chat anchors its generation at the dispatch: the
/// Harmony pipeline's processor reads the attempt's dispatch, not its own
/// entry. Its handler runs under an entry stamp, and it records its dispatch
/// wait once, with no first token, under the Harmony pipeline's backend
/// label: the wait ends where the generation clock starts, so the two fit in
/// the request's time since its entry.
#[test]
fn harmony_collected_chat_includes_the_held_header_delay() {
    let ids = harmony_output_ids();
    let tokens = u64::try_from(ids.len()).expect("a small count");
    let mut since_entry = Duration::ZERO;
    let rendered = render_after(|| async {
        let port = serve_engine(held_headers_engine(Some(ids)).into_server()).await;
        let router = router_over(&[port], HARMONY_MODEL, None).await;
        let request =
            serde_json::from_value(chat_request(HARMONY_MODEL, false)).expect("chat request");
        let tenant = tenant();
        let entered = std::time::Instant::now() - ENTRY_LEAD;
        let response = RequestStarted::at(entered)
            .scope(router.route_chat(None, &tenant, Arc::new(request), HARMONY_MODEL))
            .await;
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("response body");
        since_entry = entered.elapsed();
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON response");
        assert_eq!(status, http::StatusCode::OK, "{body}");
        assert_eq!(
            body["usage"]["completion_tokens"].as_u64(),
            Some(tokens),
            "{body}"
        );
        assert_eq!(body["choices"][0]["message"]["content"], "Hello", "{body}");
    });
    // The Harmony pipeline records under its own backend label.
    let labels = ["backend_type=\"harmony\"", "endpoint=\"chat\""];
    let family = "smg_router_generation_duration_seconds";
    assert_eq!(
        metrics_capture::sample(&rendered, family, "count", &labels),
        Some(1.0),
        "harmony chat: one generation observation:\n{rendered}"
    );
    let generation =
        metrics_capture::sample(&rendered, family, "sum", &labels).expect("histogram sum");
    assert!(
        generation >= (HELD_HEADERS_DELAY + POST_FIRST_DELAY).as_secs_f64(),
        "harmony chat: generation {generation} must include the held-header and decode delays"
    );
    assert_eq!(
        metrics_capture::sample(&rendered, DISPATCH_WAIT, "count", &labels),
        Some(1.0),
        "harmony chat: one dispatch wait:\n{rendered}"
    );
    assert_eq!(
        metrics_capture::sample(&rendered, "smg_router_ttft_seconds", "count", &labels),
        None,
        "harmony chat: a collected answer recorded a first token:\n{rendered}"
    );
    let wait =
        metrics_capture::sample(&rendered, DISPATCH_WAIT, "sum", &labels).expect("histogram sum");
    assert!(
        wait >= ENTRY_LEAD.as_secs_f64(),
        "harmony chat: dispatch wait {wait} runs from the entry stamp ({ENTRY_LEAD:?} before the handler)"
    );
    let since_entry = since_entry.as_secs_f64();
    assert!(
        wait + generation <= since_entry,
        "harmony chat: dispatch wait {wait} ends where the generation clock ({generation}) starts: the two fit in the request's time since its entry ({since_entry})"
    );
}

/// A Harmony streamed chat records its dispatch wait with its TTFT, under the
/// Harmony pipeline's backend label.
#[test]
fn harmony_streaming_chat_records_its_dispatch_wait_with_its_ttft() {
    // Read before the runtime starts: loading the encoding may fetch its vocabulary with a
    // blocking client, which cannot be dropped inside the runtime.
    let ids = harmony_output_ids();
    let rendered = render_after(|| async {
        let engine = held_headers_engine(Some(ids));
        let port = serve_engine(engine.into_server()).await;
        let router = router_over(&[port], HARMONY_MODEL, None).await;
        let request =
            serde_json::from_value(chat_request(HARMONY_MODEL, true)).expect("chat request");
        let tenant = tenant();
        let response = entered_a_lead_ago()
            .scope(router.route_chat(None, &tenant, Arc::new(request), HARMONY_MODEL))
            .await;
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("response body");
        let body = String::from_utf8(bytes.to_vec()).expect("UTF-8 body");
        assert_eq!(status, http::StatusCode::OK, "{body}");
        assert!(body.contains("data: [DONE]"), "{body}");
    });
    let labels = ["backend_type=\"harmony\"", "endpoint=\"chat\""];
    let wait = dispatch_wait_beside_the_ttft(&rendered, &labels);
    assert!(
        wait >= ENTRY_LEAD.as_secs_f64(),
        "harmony chat: dispatch wait {wait} runs from the entry stamp ({ENTRY_LEAD:?} before the handler)"
    );
}

/// An engine whose first generate call, across every worker it serves, fails
/// after `FAILED_ATTEMPT_DELAY`; every later call answers as the held-headers
/// engine does.
#[derive(Clone)]
struct FailsFirstEngine {
    calls: Arc<AtomicU32>,
    answering: HeldHeadersEngine,
}

#[tonic::async_trait]
impl VllmAnswers for FailsFirstEngine {
    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            tokio::time::sleep(FAILED_ATTEMPT_DELAY).await;
            return Err(Status::unavailable("the first attempt's worker"));
        }
        self.answering.generate(request).await
    }
}

/// A dispatch that answers on its retry anchors at the answering attempt:
/// the TTFT carries the answering worker's hold and not the failed attempt's
/// wait, which a clock started at the request's entry would include. That
/// wait is in the request's dispatch wait, which runs from its entry stamp to
/// the answering attempt's dispatch and is recorded once, not per attempt.
#[test]
fn a_retried_dispatch_anchors_at_the_answering_attempt() {
    let calls = Arc::new(AtomicU32::new(0));
    let rendered = render_after(|| async {
        let engine = FailsFirstEngine {
            calls: Arc::clone(&calls),
            answering: held_headers_engine(None),
        };
        let first = serve_engine(engine.clone().into_server()).await;
        let second = serve_engine(engine.into_server()).await;
        let retries = RetryConfig {
            max_retries: 2,
            initial_backoff_ms: 1,
            max_backoff_ms: 2,
            ..Default::default()
        };
        let router = router_over(&[first, second], MODEL, Some(retries)).await;
        request_entered(
            router.as_ref(),
            Route::Chat,
            true,
            Some(RequestStarted::now()),
        )
        .await;
    });
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "one failed attempt, one answer"
    );
    let generation = observation(&rendered, "smg_router_generation_duration_seconds", "chat");
    assert_stream_records(&rendered, "chat", generation, OUTPUT_TOKENS);
    let ttft = observation(&rendered, "smg_router_ttft_seconds", "chat");
    assert!(
        ttft < (FAILED_ATTEMPT_DELAY + HELD_HEADERS_DELAY).as_secs_f64(),
        "chat: TTFT {ttft} must not include the failed attempt's {FAILED_ATTEMPT_DELAY:?}"
    );
    let labels = ["backend_type=\"regular\"", "endpoint=\"chat\""];
    let wait = dispatch_wait_beside_the_ttft(&rendered, &labels);
    assert!(
        wait >= FAILED_ATTEMPT_DELAY.as_secs_f64(),
        "chat: dispatch wait {wait} must include the failed attempt's {FAILED_ATTEMPT_DELAY:?}"
    );
    let floor = (FAILED_ATTEMPT_DELAY + HELD_HEADERS_DELAY).as_secs_f64();
    assert!(
        wait + ttft >= floor,
        "chat: dispatch wait {wait} and TTFT {ttft} add up to the time to first token from the request's entry ({floor})"
    );
}

/// A transcription whose first attempt fails with a retryable status answers
/// from the other worker, as a chat does: the route hands the pipeline the
/// retry settings it resolved.
#[tokio::test]
async fn a_transcription_answers_on_its_retry() {
    let calls = Arc::new(AtomicU32::new(0));
    let engine = FailsFirstEngine {
        calls: Arc::clone(&calls),
        answering: held_headers_engine(Some(TRANSCRIPT_IDS.to_vec())),
    };
    let first = serve_engine(engine.clone().into_server()).await;
    let second = serve_engine(engine.into_server()).await;
    let retries = RetryConfig {
        max_retries: 2,
        initial_backoff_ms: 1,
        max_backoff_ms: 2,
        ..Default::default()
    };
    let router = transcription_router_over(&[first, second], Some(retries)).await;
    // A 200 with the transcript, or this fails on the first attempt's status.
    request_entered(router.as_ref(), Route::Transcriptions, false, None).await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "one failed attempt, one answer"
    );
}

/// An engine that counts its generate calls, across every worker it serves,
/// and answers each as the held-headers engine does.
#[derive(Clone)]
struct CountedEngine {
    calls: Arc<AtomicU32>,
    answering: HeldHeadersEngine,
}

#[tonic::async_trait]
impl VllmAnswers for CountedEngine {
    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<VllmGenerateStream>, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.answering.generate(request).await
    }
}

/// A transcript of nothing but blanks is refused by the gateway itself, on a
/// whole engine answer: the caller reads that 500 behind one dispatch, and no
/// retry runs an engine again on the same audio for the same reading.
#[tokio::test]
async fn a_blank_transcript_is_refused_behind_one_dispatch() {
    let calls = Arc::new(AtomicU32::new(0));
    let engine = CountedEngine {
        calls: Arc::clone(&calls),
        answering: held_headers_engine(Some(vec![BLANK_ID; TRANSCRIPT_IDS.len()])),
    };
    let first = serve_engine(engine.clone().into_server()).await;
    let second = serve_engine(engine.into_server()).await;
    let retries = RetryConfig {
        max_retries: 3,
        initial_backoff_ms: 1,
        max_backoff_ms: 2,
        ..Default::default()
    };
    let router = transcription_router_over(&[first, second], Some(retries)).await;
    let response = route_request(router.as_ref(), Route::Transcriptions, false).await;
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap_or_default();
    let body = String::from_utf8_lossy(&bytes);
    assert_eq!(status, http::StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert!(body.contains("empty_transcription_response"), "{body}");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "one dispatch and no retry: {body}"
    );
}

/// A transcription asked for by a model's alias retries under that model's
/// own retry settings, as a chat does: the router's allow one attempt here
/// and the model's two, and the second attempt answers.
#[tokio::test]
async fn a_transcription_by_an_alias_retries_under_its_model_s_settings() {
    let calls = Arc::new(AtomicU32::new(0));
    let engine = FailsFirstEngine {
        calls: Arc::clone(&calls),
        answering: held_headers_engine(Some(TRANSCRIPT_IDS.to_vec())),
    };
    let first = serve_engine(engine.clone().into_server()).await;
    let second = serve_engine(engine.into_server()).await;
    let one_attempt = RetryConfig {
        max_retries: 1,
        ..Default::default()
    };
    let config = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .retry_config(one_attempt);
    let app_context = transcription_context(config).await;
    let workers = [first, second]
        .into_iter()
        .map(|port| {
            let card = ModelCard::new(ASR_MODEL).with_alias(ASR_ALIAS);
            Arc::new(grpc_worker(port, WorkerType::Regular, RuntimeType::Vllm, card).build())
                as Arc<dyn Worker>
        })
        .collect();
    let router = grpc_router(&app_context, workers).await;
    let two_attempts = RetryConfig {
        max_retries: 2,
        initial_backoff_ms: 1,
        max_backoff_ms: 2,
        ..Default::default()
    };
    app_context
        .worker_registry
        .set_model_retry_config(ASR_MODEL, two_attempts, true);

    let response = transcribe(router.as_ref(), ASR_ALIAS).await;
    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "one failed attempt, one answer"
    );
}

/// A transcription whose audio is larger than `max_buffered_request_bytes`
/// gets one attempt, as on the HTTP router: the caller reads the first
/// attempt's failure, and no second dispatch runs.
#[tokio::test]
async fn a_transcription_above_the_buffer_bound_gets_one_attempt() {
    let calls = Arc::new(AtomicU32::new(0));
    let engine = FailsFirstEngine {
        calls: Arc::clone(&calls),
        answering: held_headers_engine(Some(TRANSCRIPT_IDS.to_vec())),
    };
    let first = serve_engine(engine.clone().into_server()).await;
    let second = serve_engine(engine.into_server()).await;
    let retries = RetryConfig {
        max_retries: 2,
        initial_backoff_ms: 1,
        max_backoff_ms: 2,
        ..Default::default()
    };
    let config = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .retry_config(retries)
        .max_buffered_request_bytes(CLIP.len() as u64 - 1);
    let app_context = transcription_context(config).await;
    let workers = [(first, WorkerType::Regular), (second, WorkerType::Regular)];
    let router = grpc_router(
        &app_context,
        grpc_workers(&workers, RuntimeType::Vllm, ASR_MODEL),
    )
    .await;

    let response = transcribe(router.as_ref(), ASR_MODEL).await;
    assert!(response.status().is_server_error(), "the first failure");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "one attempt and no retry");
}

/// A client that leaves after the first chunk records nothing: no TTFT,
/// generation or TPOT sample for the route, and no dispatch wait though its
/// handler ran under an entry stamp (the wait is recorded with the answer's
/// other records, at the stream's end, and this stream makes none).
#[test]
fn a_client_leaving_after_the_first_chunk_records_no_stream_metrics() {
    let rendered = render_after(|| async {
        let router = regular_router().await;
        let request = serde_json::from_value(chat_request(MODEL, true)).expect("chat request");
        let tenant = tenant();
        let response = RequestStarted::now()
            .scope(router.route_chat(None, &tenant, Arc::new(request), MODEL))
            .await;
        assert_eq!(response.status(), http::StatusCode::OK);
        let mut body = response.into_body();
        let first = body
            .frame()
            .await
            .expect("the first frame")
            .expect("a frame");
        assert!(first.is_data(), "the first frame carries the first chunk");
        drop(body);
        // The stream learns of the departed client at its next send, after
        // the engine's post-first delay; the records, if any, would follow.
        tokio::time::sleep(POST_FIRST_DELAY * 3).await;
    });
    for family in [
        "smg_router_ttft_seconds",
        DISPATCH_WAIT,
        "smg_router_generation_duration_seconds",
        "smg_router_tpot_seconds",
    ] {
        assert!(
            sample(&rendered, family, "count", "chat").is_none(),
            "a client that left recorded {family}:\n{rendered}"
        );
    }
}

/// The tool-loop engine's first generation, as its tokenizer decodes it: a
/// call of the MCP server's search tool. The generation after the tool's
/// result decodes to `FINAL_ANSWER`.
const TOOL_CALL: &str =
    "<tool_call>\n{\"name\":\"brave_web_search\",\"arguments\":{\"query\":\"test\"}}\n</tool_call>";
const FINAL_ANSWER: &str = "Final answer after tools";
/// The ids the tool-loop engine answers every generation with: the scripted
/// tokenizer reads the generation's text off id 100, which arrives with the
/// headers.
const TOOL_LOOP_IDS: [u32; 2] = [100, 101];

/// A regular router over one held-headers engine whose tokenizer decodes a
/// request's first generation to a call of the MCP server's search tool and
/// the generation after one tool result to a plain answer, under the `qwen`
/// tool-call parser: a Responses request with that server as its tool runs a
/// loop of two turns, each a generation the engine holds and stretches as it
/// does any other.
async fn tool_loop_router() -> Box<dyn RouterTrait> {
    let engine = held_headers_engine(Some(TOOL_LOOP_IDS.to_vec()));
    let port = serve_engine(engine.into_server()).await;
    let tokenizer = ScriptedTokenizer::new(TOOL_CALL).with_final_answer(1, FINAL_ANSWER);
    let mut config = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .build_unchecked();
    config.tool_call_parser = Some("qwen".into());
    let app_context = grpc_test_context_with_tokenizer(config, MODEL, Arc::new(tokenizer)).await;
    grpc_router(
        &app_context,
        grpc_workers(&[(port, WorkerType::Regular)], RuntimeType::Vllm, MODEL),
    )
    .await
}

/// One Responses request, read to its end: the body. Its handler runs under
/// the entry stamp `entered`, and the body is read outside it, as the
/// connection reads it. With `mcp_server` the request names the MCP server
/// at that URL as its tool, which makes it a tool loop.
#[expect(clippy::expect_used, reason = "test helper")]
async fn responses_request(
    router: &dyn RouterTrait,
    mcp_server: Option<&str>,
    streaming: bool,
    entered: RequestStarted,
) -> String {
    let mut request = serde_json::json!({
        "model": MODEL,
        "input": "Hello",
        "stream": streaming,
        "max_output_tokens": 16,
    });
    if let Some(server_url) = mcp_server {
        request["tools"] = serde_json::json!([{
            "type": "mcp",
            "server_label": "test-tools",
            "server_url": server_url,
            "require_approval": "never",
        }]);
    }
    let request = serde_json::from_value(request).expect("Responses request");
    let tenant = tenant();
    let response = entered
        .scope(router.route_responses(None, &tenant, request, MODEL))
        .await;
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    let body = String::from_utf8(bytes.to_vec()).expect("UTF-8 body");
    assert_eq!(status, http::StatusCode::OK, "{body}");
    body
}

/// One streamed Responses request, read to its end (see [`responses_request`]).
async fn streamed_responses(
    router: &dyn RouterTrait,
    mcp_server: Option<&str>,
    entered: RequestStarted,
) -> String {
    let body = responses_request(router, mcp_server, true, entered).await;
    assert!(body.contains("event: response.completed"), "{body}");
    body
}

/// One Responses tool loop of two turns against `mcp`, streamed or collected,
/// read to its end: the first generation calls the server's tool once, the
/// second answers.
async fn tool_loop(
    router: &dyn RouterTrait,
    mcp: &MockMCPServer,
    streaming: bool,
    entered: RequestStarted,
) {
    let calls = mcp.call_count();
    let body = if streaming {
        streamed_responses(router, Some(&mcp.url()), entered).await
    } else {
        responses_request(router, Some(&mcp.url()), false, entered).await
    };
    assert_eq!(
        mcp.call_count(),
        calls + 1,
        "one tool call between the two turns: {body}"
    );
    assert!(
        body.contains(FINAL_ANSWER),
        "the turn after the tool call answers: {body}"
    );
}

/// A streamed Responses request whose tool loop runs two turns records a
/// dispatch wait with each turn's TTFT, under the TTFT's labels, though the
/// loop dispatches from a task of its own. The first turn's wait counts from
/// the request's entry stamp. The second turn's counts from where that turn
/// starts: counted from the request's entry it would hold the whole first
/// generation and the tool call, and the two waits together stay under the
/// first generation alone.
#[expect(
    clippy::print_stderr,
    reason = "report the measured intervals alongside the regression result"
)]
#[test]
fn a_streamed_tool_loop_records_a_dispatch_wait_with_every_turn_s_ttft() {
    let rendered = render_after(|| async {
        let router = tool_loop_router().await;
        let mut mcp = MockMCPServer::start().await.expect("the MCP server");
        tool_loop(router.as_ref(), &mcp, true, entered_a_lead_ago()).await;
        mcp.stop().await;
    });
    // A turn of the loop is a chat generation: it records under the chat
    // route's labels.
    let labels = ["backend_type=\"regular\"", "endpoint=\"chat\""];
    let waits = dispatch_waits_beside_the_ttfts(&rendered, &labels, 2);
    let ttfts = metrics_capture::sample(&rendered, "smg_router_ttft_seconds", "sum", &labels)
        .expect("histogram sum");
    eprintln!("tool loop, two turns: dispatch waits={waits} ttfts={ttfts}");
    assert!(
        ttfts >= (2 * HELD_HEADERS_DELAY).as_secs_f64(),
        "both turns' TTFTs ({ttfts}) carry the engine's hold"
    );
    let lead = ENTRY_LEAD.as_secs_f64();
    assert!(
        waits >= lead,
        "the first turn's dispatch wait runs from the request's entry stamp ({ENTRY_LEAD:?} before the handler): {waits}"
    );
    // The first turn's wait is the stamp's lead at least, so the rest of the
    // sum bounds the second turn's from above.
    let second_turn_at_most = waits - lead;
    let first_generation = (HELD_HEADERS_DELAY + POST_FIRST_DELAY).as_secs_f64();
    assert!(
        second_turn_at_most < first_generation,
        "the second turn's dispatch wait (at most {second_turn_at_most}) runs from that turn's start, not from the request's entry: the first generation alone took {first_generation}"
    );
}

/// A chat stream, a streamed Responses answer and a turn of a streamed tool
/// loop are all chat generations and record under the same labels. On a
/// router that served one of each (the loop two turns), the dispatch wait
/// has as many observations as the TTFT: four.
#[test]
fn chat_and_responses_streams_record_as_many_dispatch_waits_as_ttfts() {
    let rendered = render_after(|| async {
        let router = tool_loop_router().await;
        let mut mcp = MockMCPServer::start().await.expect("the MCP server");
        request_entered(
            router.as_ref(),
            Route::Chat,
            true,
            Some(entered_a_lead_ago()),
        )
        .await;
        streamed_responses(router.as_ref(), None, entered_a_lead_ago()).await;
        tool_loop(router.as_ref(), &mcp, true, entered_a_lead_ago()).await;
        mcp.stop().await;
    });
    let labels = ["backend_type=\"regular\"", "endpoint=\"chat\""];
    dispatch_waits_beside_the_ttfts(&rendered, &labels, 4);
}

/// A collected Responses request whose tool loop runs two turns records a
/// dispatch wait for each turn's generation, under the chat route's labels,
/// and no first token. The first turn's wait counts from the request's entry
/// stamp. The second turn's counts from where that turn starts: counted from
/// the request's entry it would hold the whole first generation and the tool
/// call, and the two waits together stay under the first generation alone.
#[expect(
    clippy::print_stderr,
    reason = "report the measured intervals alongside the regression result"
)]
#[test]
fn a_collected_tool_loop_records_a_dispatch_wait_for_every_turn() {
    let rendered = render_after(|| async {
        let router = tool_loop_router().await;
        let mut mcp = MockMCPServer::start().await.expect("the MCP server");
        tool_loop(router.as_ref(), &mcp, false, entered_a_lead_ago()).await;
        mcp.stop().await;
    });
    let labels = ["backend_type=\"regular\"", "endpoint=\"chat\""];
    for (family, observations) in [
        (DISPATCH_WAIT, Some(2.0)),
        ("smg_router_generation_duration_seconds", Some(2.0)),
        ("smg_router_ttft_seconds", None),
    ] {
        assert_eq!(
            metrics_capture::sample(&rendered, family, "count", &labels),
            observations,
            "{family}: one record per collected turn, and no first token:\n{rendered}"
        );
    }
    let waits =
        metrics_capture::sample(&rendered, DISPATCH_WAIT, "sum", &labels).expect("histogram sum");
    eprintln!("collected tool loop, two turns: dispatch waits={waits}");
    let lead = ENTRY_LEAD.as_secs_f64();
    assert!(
        waits >= lead,
        "the first turn's dispatch wait runs from the request's entry stamp ({ENTRY_LEAD:?} before the handler): {waits}"
    );
    // The first turn's wait is the stamp's lead at least, so the rest of the
    // sum bounds the second turn's from above.
    let second_turn_at_most = waits - lead;
    let first_generation = (HELD_HEADERS_DELAY + POST_FIRST_DELAY).as_secs_f64();
    assert!(
        second_turn_at_most < first_generation,
        "the second turn's dispatch wait (at most {second_turn_at_most}) runs from that turn's start, not from the request's entry: the first generation alone took {first_generation}"
    );
}
