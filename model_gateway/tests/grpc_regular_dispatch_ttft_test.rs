//! Regular gRPC latency starts at dispatch, including a worker that holds
//! response headers until its first chunk: every streaming route and the
//! collected ones, the batched completion stream, the Harmony pipeline, a
//! dispatch that answers on its retry, and the client that leaves after the
//! first chunk (which records nothing). Each case captures metrics on a
//! current-thread runtime, as the PD TTFT regressions do.

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
        grpc_workers, serve_engine,
    },
    metrics_capture::{self, render_after},
};
use http_body_util::BodyExt;
use llm_tokenizer::{
    traits::{MediaHandling, RendererCapabilities},
    MockTokenizer,
};
use openai_protocol::transcription::{AudioFile, TranscriptionRequest};
use smg::{
    config::RetryConfig,
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};
use smg_grpc_client::vllm_proto as proto;
use smg_test_harness as common;
use tonic::{Request, Response, Status};

const MODEL: &str = "regular-dispatch-ttft-test-model";
/// A model id the router reads as a Harmony model: its chat runs the Harmony pipeline.
const HARMONY_MODEL: &str = "gpt-oss-dispatch-ttft-test-model";
/// A model id the one transcription family recognises: a transcription is a
/// chat-shaped generation to the engine.
const ASR_MODEL: &str = "qwen3-asr-dispatch-ttft-test-model";
/// The ids the transcription case's engine answers, words of its tokenizer.
const TRANSCRIPT_IDS: [u32; 3] = [100, 101, 102];
const OUTPUT_TOKENS: u32 = 3;
const HELD_HEADERS_DELAY: Duration = Duration::from_millis(300);
const POST_FIRST_DELAY: Duration = Duration::from_millis(120);
/// The most a first token may carry over the hold: the first chunk arrives
/// with the headers, so the in-stream delay (`POST_FIRST_DELAY`, longer than
/// this slack) is never part of it.
const FIRST_TOKEN_SLACK: Duration = Duration::from_millis(100);
/// What a worker that fails its attempt spends before failing it.
const FAILED_ATTEMPT_DELAY: Duration = Duration::from_millis(400);

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

/// A regular router serving `ASR_MODEL` over one held-headers engine. Its
/// mock tokenizer renders media as template text, so the audio turn needs no
/// media processing, and it names the engine's ids, so the transcript is text.
async fn transcription_router() -> Box<dyn RouterTrait> {
    let engine = held_headers_engine(Some(TRANSCRIPT_IDS.to_vec()));
    let port = serve_engine(engine.into_server()).await;
    let tokenizer = MockTokenizer::new()
        .with_renderer_capabilities(RendererCapabilities {
            media_handling: MediaHandling::TemplateText,
            ..RendererCapabilities::default()
        })
        .with_tokens(&[
            ("spoken", TRANSCRIPT_IDS[0]),
            ("words", TRANSCRIPT_IDS[1]),
            ("here", TRANSCRIPT_IDS[2]),
        ]);
    let config = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .build_unchecked();
    let app_context =
        grpc_test_context_with_tokenizer(config, ASR_MODEL, Arc::new(tokenizer)).await;
    grpc_router(
        &app_context,
        grpc_workers(&[(port, WorkerType::Regular)], RuntimeType::Vllm, ASR_MODEL),
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

/// One request on `route`, read to its end: the status and the body are
/// checked, and a generate answer's `e2e_latency` (its processor-entry clock)
/// is returned for the comparison with the dispatch-anchored record.
#[expect(clippy::expect_used, reason = "test helper")]
async fn request(router: &dyn RouterTrait, route: Route, streaming: bool) -> Option<f64> {
    let tenant = tenant();
    let model = route.model();
    let response = match route {
        Route::Chat => {
            let request =
                serde_json::from_value(chat_request(model, streaming)).expect("chat request");
            router.route_chat(None, &tenant, request, model).await
        }
        Route::Transcriptions => {
            let request = TranscriptionRequest {
                model: model.to_string(),
                ..Default::default()
            };
            let audio = AudioFile {
                bytes: bytes::Bytes::from_static(b"not real audio"),
                file_name: "clip.wav".to_string(),
                content_type: Some("audio/wav".to_string()),
            };
            router
                .route_audio_transcriptions(None, &tenant, &request, audio, model)
                .await
        }
        Route::Messages => {
            let request = serde_json::from_value(serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content": "Hello world"}],
                "max_tokens": 8,
                "stream": streaming,
            }))
            .expect("Messages request");
            router.route_messages(None, &tenant, request, model).await
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
        // Collected responses retain their existing generation-only semantics.
        assert!(ttft.is_none(), "{endpoint}: collected TTFT {ttft:?}");
        assert!(tpot.is_none(), "{endpoint}: collected TPOT {tpot:?}");
    }
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
/// entry.
#[test]
fn harmony_collected_chat_includes_the_held_header_delay() {
    let ids = harmony_output_ids();
    let tokens = u64::try_from(ids.len()).expect("a small count");
    let rendered = render_after(|| async {
        let port = serve_engine(held_headers_engine(Some(ids)).into_server()).await;
        let router = router_over(&[port], HARMONY_MODEL, None).await;
        let request =
            serde_json::from_value(chat_request(HARMONY_MODEL, false)).expect("chat request");
        let response = router
            .route_chat(None, &tenant(), request, HARMONY_MODEL)
            .await;
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .expect("response body");
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
/// wait, which a clock started at the request's entry would include.
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
        request(router.as_ref(), Route::Chat, true).await;
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
}

/// A client that leaves after the first chunk records nothing: no TTFT,
/// generation or TPOT sample for the route.
#[test]
fn a_client_leaving_after_the_first_chunk_records_no_stream_metrics() {
    let rendered = render_after(|| async {
        let router = regular_router().await;
        let request = serde_json::from_value(chat_request(MODEL, true)).expect("chat request");
        let response = router.route_chat(None, &tenant(), request, MODEL).await;
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
        "smg_router_generation_duration_seconds",
        "smg_router_tpot_seconds",
    ] {
        assert!(
            sample(&rendered, family, "count", "chat").is_none(),
            "a client that left recorded {family}:\n{rendered}"
        );
    }
}
