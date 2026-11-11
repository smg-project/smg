//! A parallel-dispatch prefill side whose stream opens with a failure while
//! the decode call is still pending is answered at once as the prefill
//! failing first, instead of holding the client (and the decode room the
//! admission gate claimed for it) until the decode call returns: with a
//! servicer that sends its headers with the first message, that is when the
//! decode engine gives up on the handoff itself.
//!
//! The mock engines speak the TokenSpeed proto. The prefill returns its
//! stream at once and fails it with its first item (the engine's own
//! `INTERNAL`); the decode holds its call's headers for good. The answer is
//! the prefill side's start failure under the shared vocabulary, in hand
//! well before the case cap, with the prefill worker charged one failure and
//! the decode worker nothing, on the plain and the streamed form. A prefill
//! that answers before the decode call returns is read in full, as before.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use common::{
    engine_answers::{TokenSpeedAnswers, TokenSpeedGenerateStream},
    grpc_harness::{build_grpc_router, grpc_router_config, grpc_worker, serve_engine},
    metrics_capture::render_after,
};
use openai_protocol::model_card::ModelCard;
use smg::{
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        RuntimeType, Worker, WorkerType,
    },
};
use smg_grpc_client::tokenspeed_scheduler::tokenspeed_proto as ts;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::common;

const MODEL: &str = "pd-prefill-failure-held-test-model";
/// Tokens an engine that answers produces; ids 1..=3 are in the mock
/// tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// Longer than any case: a request held to the decode call's return fails
/// the case here instead of hanging it for the request timeout.
const CASE_CAP: Duration = Duration::from_secs(30);
/// How soon a failure already in hand is answered: scheduling slack on a
/// loaded machine, nothing else.
const PROMPT: Duration = Duration::from_secs(3);
/// The delay of a call that must never return within a case.
const FOREVER: Duration = Duration::from_secs(3_600);
/// How long a late decode holds its headers before it answers.
const LATE_HEADERS: Duration = Duration::from_secs(1);
/// The words the failing engine puts on its status.
const REASON: &str = "the engine says no";

/// What a mock engine does with a generate call.
#[derive(Clone, Copy)]
enum Behaviour {
    /// The stream is returned at once and its first item is the failure: a
    /// side admitted, then failed before its first token.
    FailsFirst,
    /// The call never returns its stream: a servicer that sends its headers
    /// with the first message, in front of a decode engine waiting on a
    /// handoff that never comes.
    HoldsHeaders,
    /// The stream is returned after the delay, with the tokens and the
    /// completion on it at once.
    Answers(Duration),
}

/// A parallel-dispatch mock engine (the TokenSpeed proto) with one
/// behaviour, counting its generate calls.
#[derive(Clone)]
struct Engine {
    behaviour: Behaviour,
    calls: Arc<AtomicUsize>,
}

impl Engine {
    fn new(behaviour: Behaviour) -> Self {
        Self {
            behaviour,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

fn ts_chunk(request_id: &str, id: u32) -> ts::GenerateResponse {
    ts::GenerateResponse {
        request_id: request_id.to_string(),
        response: Some(ts::generate_response::Response::Chunk(
            ts::GenerateStreamChunk {
                token_ids: vec![id],
                prompt_tokens: 1,
                completion_tokens: 1,
                index: 0,
                ..Default::default()
            },
        )),
    }
}

fn ts_complete(request_id: &str) -> ts::GenerateResponse {
    ts::GenerateResponse {
        request_id: request_id.to_string(),
        response: Some(ts::generate_response::Response::Complete(
            ts::GenerateComplete {
                output_ids: (1..=OUTPUT_TOKENS).collect(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
                completion_tokens: OUTPUT_TOKENS,
                index: 0,
                ..Default::default()
            },
        )),
    }
}

#[tonic::async_trait]
impl TokenSpeedAnswers for Engine {
    #[expect(
        clippy::disallowed_methods,
        reason = "test mock; the producer task ends with its stream"
    )]
    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<TokenSpeedGenerateStream>, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let request_id = request.into_inner().request_id;
        let (tx, rx) = mpsc::channel(8);
        match self.behaviour {
            Behaviour::HoldsHeaders => tokio::time::sleep(FOREVER).await,
            Behaviour::FailsFirst => {
                tokio::spawn(async move {
                    let _ = tx.send(Err(Status::internal(REASON))).await;
                });
            }
            Behaviour::Answers(delay) => {
                tokio::time::sleep(delay).await;
                tokio::spawn(async move {
                    for id in 1..=OUTPUT_TOKENS {
                        if tx.send(Ok(ts_chunk(&request_id, id))).await.is_err() {
                            return;
                        }
                    }
                    let _ = tx.send(Ok(ts_complete(&request_id))).await;
                });
            }
        }
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

/// A parallel-shape PD gRPC router over `prefill` and `decode`, retries off
/// so the first attempt's answer is the client's, every breaker set to open
/// on one failure so the charge of a failure is read from the breaker; the
/// registered workers come back with the router, prefill first.
async fn build_pd_router(
    prefill: &Engine,
    decode: &Engine,
) -> (Box<dyn RouterTrait>, Vec<Arc<dyn Worker>>) {
    let prefill_port = serve_engine(prefill.clone().into_server()).await;
    let decode_port = serve_engine(decode.clone().into_server()).await;
    let config = grpc_router_config()
        .prefill_decode_mode(vec![], vec![])
        .round_robin_policy()
        .disable_retries()
        .build_unchecked();
    let breaker = CircuitBreakerConfig {
        failure_threshold: 1,
        ..Default::default()
    };
    let workers: Vec<Arc<dyn Worker>> = [
        (prefill_port, WorkerType::Prefill),
        (decode_port, WorkerType::Decode),
    ]
    .into_iter()
    .map(|(port, worker_type)| {
        Arc::new(
            grpc_worker(
                port,
                worker_type,
                RuntimeType::TokenSpeed,
                ModelCard::new(MODEL),
            )
            .circuit_breaker_config(breaker.clone())
            .build(),
        ) as Arc<dyn Worker>
    })
    .collect();
    let router = build_grpc_router(config, MODEL, workers.clone()).await;
    (router, workers)
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-prefill-failure-held-test-tenant"))
}

/// One chat through `router`, streamed or not: its status, body and duration.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn one_chat(
    router: &dyn RouterTrait,
    streamed: bool,
) -> (http::StatusCode, String, Duration) {
    let request = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": streamed,
    }))
    .unwrap();
    let started = Instant::now();
    let response =
        tokio::time::timeout(CASE_CAP, router.route_chat(None, &tenant(), request, MODEL))
            .await
            .expect(
                "the request must be answered within the case cap, not held to the decode call",
            );
    let status = response.status();
    let bytes = tokio::time::timeout(
        CASE_CAP,
        axum::body::to_bytes(response.into_body(), 1 << 20),
    )
    .await
    .expect("the response body must end within the case cap")
    .expect("response body");
    (
        status,
        String::from_utf8(bytes.to_vec()).expect("utf-8 body"),
        started.elapsed(),
    )
}

/// The envelope's code and message, from a JSON error body; `none` and an
/// empty message for a body that is not the envelope.
fn envelope(body: &str) -> (String, String) {
    let json = serde_json::from_str::<serde_json::Value>(body).unwrap_or_default();
    (
        json["error"]["code"].as_str().unwrap_or("none").to_string(),
        json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
    )
}

/// The value of the `smg_worker_errors_total` sample for `worker_type` and
/// `error_type` (any connection), zero when it was not emitted.
fn worker_errors(rendered: &str, worker_type: &str, error_type: &str) -> f64 {
    let labels = [
        format!("worker_type=\"{worker_type}\""),
        format!("error_type=\"{error_type}\""),
    ];
    rendered
        .lines()
        .filter(|line| line.starts_with("smg_worker_errors_total{"))
        .filter(|line| labels.iter().all(|label| line.contains(label.as_str())))
        .filter_map(|line| line.rsplit(' ').next())
        .filter_map(|value| value.parse::<f64>().ok())
        .sum()
}

/// The prefill stream fails at its first item while the decode call is still
/// pending: the request is answered at once with the prefill side's start
/// failure (the worker's own status and words under the side's code), the
/// prefill worker charged once and the decode worker not at all.
fn held_prefill_failure_case(streamed: bool) {
    let prefill = Engine::new(Behaviour::FailsFirst);
    let decode = Engine::new(Behaviour::HoldsHeaders);
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let (router, workers) = build_pd_router(&prefill, &decode).await;

        let (status, body, elapsed) = one_chat(router.as_ref(), streamed).await;

        assert_eq!(
            status,
            http::StatusCode::INTERNAL_SERVER_ERROR,
            "streamed={streamed}: {body}"
        );
        let (code, message) = envelope(&body);
        assert_eq!(code, "prefill_worker_failed_to_start", "{body}");
        assert!(
            message.starts_with("Prefill worker failed to start: ") && message.ends_with(REASON),
            "{body}"
        );
        assert!(
            elapsed < PROMPT,
            "a failure in hand is answered at once, not at the decode call's return: {elapsed:?}"
        );
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Open,
            "the prefill worker is charged its one failure"
        );
        assert_eq!(
            workers[1].circuit_breaker_state(),
            CircuitState::Closed,
            "the decode worker, whose call is still pending, is charged nothing"
        );
    });
    assert_eq!(
        prefill_calls.calls(),
        1,
        "the prefill side was dispatched once"
    );
    assert_eq!(
        decode_calls.calls(),
        1,
        "the decode side was dispatched once, and its call is left to be retired"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "backend_error"),
        1.0,
        "one engine error on the prefill worker:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "decode", "backend_error")
            + worker_errors(&rendered, "decode", "timeout"),
        0.0,
        "nothing on the decode worker:\n{rendered}"
    );
}

#[test]
fn a_prefill_that_fails_at_its_first_item_while_the_decode_call_is_pending_is_answered_at_once() {
    held_prefill_failure_case(false);
}

#[test]
fn a_streamed_request_whose_prefill_fails_at_its_first_item_while_the_decode_call_is_pending_is_answered_whole(
) {
    held_prefill_failure_case(true);
}

/// The prefill answers before the decode call returns its stream (a decode
/// servicer that sends its headers with the first message, late): the
/// prefill's first item, pulled ahead while the decode call was pending, is
/// read by its reader as it was, and the request is answered in full.
#[test]
fn a_prefill_that_answers_before_the_decode_call_returns_is_read_in_full() {
    let prefill = Engine::new(Behaviour::Answers(Duration::ZERO));
    let decode = Engine::new(Behaviour::Answers(LATE_HEADERS));
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    render_after(|| async move {
        let (router, workers) = build_pd_router(&prefill, &decode).await;

        let (status, body, elapsed) = one_chat(router.as_ref(), false).await;

        assert_eq!(status, http::StatusCode::OK, "{body}");
        let json: serde_json::Value = serde_json::from_str(&body).expect("a JSON answer");
        assert_eq!(
            json["usage"]["completion_tokens"].as_u64(),
            Some(u64::from(OUTPUT_TOKENS)),
            "the answer carries the decode's completion: {body}"
        );
        assert!(
            elapsed >= LATE_HEADERS && elapsed < LATE_HEADERS + PROMPT,
            "answered once the decode call returned: {elapsed:?}"
        );
        for worker in &workers {
            assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
        }
    });
    assert_eq!(prefill_calls.calls(), 1);
    assert_eq!(decode_calls.calls(), 1);
}
