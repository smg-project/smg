//! A disaggregated prefill leg that produces nothing is given up on within
//! its silence bound, on both dispatch shapes, instead of holding the request
//! (and the decode room admission claimed for it) until the request's own
//! timeout: the attempt answers 504 `prefill_worker_silent`, the prefill
//! worker is charged one `timeout` on the worker-error family, the decode
//! leg is never asked to serve what the prefill never produced, and with
//! another prefill worker available the request is retried there, never on
//! the silent one again.
//!
//! The mock engines of the sequential shape speak the vLLM proto: one holds
//! its response headers for good (a servicer that flushes headers with the
//! first message, in front of a frozen engine), one returns a stream at once
//! and never writes to it (a frozen engine behind a servicer that flushes
//! headers eagerly), one answers after a delay. The parallel shape uses the
//! mock-worker crate's canned worker, whose delay runs before its stream is
//! returned, set longer than the test.
//!
//! The bound is the default wedge threshold (three seconds) for these short
//! prompts; every case is capped well below the request timeout, so a
//! gateway without the bound fails the case instead of hanging it.
//!
//! A prefill that answers late is another matter: on the parallel shape a
//! leg's first response also waits for the decode side to admit it, so a
//! worker that keeps answering legs one by one is backed up, not frozen.
//! Two cases queue the legs at the mock engine's gate (one every 1.2 s,
//! five at once: the last answers well past the bound) and expect every
//! request answered and nothing charged to the prefill worker. One more
//! lets the call of a leg given up on fail late, and expects the worker
//! charged once, at the verdict.

#[path = "common/mod.rs"]
mod common;

use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use futures::future::join_all;
use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use proto::generate_response::Response as GenResp;
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType,
    },
};
use smg_grpc_client::{
    common_proto as grpc_common, tokenspeed_scheduler::tokenspeed_proto as ts, vllm_proto as proto,
};
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{transport::Server, Request, Response, Status};
use ts::token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer};

const MODEL: &str = "pd-prefill-silence-test-model";
/// Tokens every answering mock produces; ids 1..=3 are in the mock tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// The default wedge threshold: the silence bound of a short prompt on an
/// idle prefill worker.
const BOUND: Duration = Duration::from_secs(3);
/// Scheduling slack for a loaded machine, on top of the bound.
const SLACK: Duration = Duration::from_secs(3);
/// Longer than any case: a gateway that does not bound the leg's silence
/// fails the case here instead of hanging it for the request timeout.
const CASE_CAP: Duration = Duration::from_secs(30);
/// The delay of a mock that must never answer within a case.
const FOREVER: Duration = Duration::from_secs(3_600);
/// How often a queued engine's gate lets a leg through: five legs at once
/// put the last one well past the bound, with an answer every turn.
const TURN: Duration = Duration::from_millis(1_200);
/// How many legs queue at once in the queued cases.
const QUEUED_LEGS: usize = 5;
/// How long after the bound a held call fails, in the late-failure case.
const LATE: Duration = Duration::from_secs(2);

type GenerateTx = mpsc::Sender<Result<proto::GenerateResponse, Status>>;

/// What a sequential-dispatch mock engine does with a generate call.
#[derive(Clone, Copy)]
enum Behaviour {
    /// The call never returns its stream.
    HoldHeaders,
    /// The stream is returned at once and nothing is ever written to it.
    EmptyStream,
    /// Tokens and a completion after the delay, the stream returned at once.
    Answer(Duration),
    /// The call returns its stream when the leg's turn comes at the engine's
    /// gate (one leg every `TURN`), the tokens at once after that.
    Queued,
    /// The call holds its headers for the delay, then fails.
    HoldHeadersThenFail(Duration),
}

/// A gate that lets one leg through every turn: legs queue in arrival order,
/// as they do in front of a decode side that pre-allocates for one at a time.
#[derive(Clone, Default)]
struct Gate {
    next: Arc<tokio::sync::Mutex<Option<tokio::time::Instant>>>,
}

impl Gate {
    /// Wait for this leg's turn.
    async fn turn(&self) {
        let at = {
            let mut next = self.next.lock().await;
            let now = tokio::time::Instant::now();
            let at = next.map_or(now, |next| next.max(now));
            *next = Some(at + TURN);
            at
        };
        tokio::time::sleep_until(at).await;
    }
}

/// A sequential-dispatch mock engine, counting its generate calls.
#[derive(Clone)]
struct Engine {
    behaviour: Behaviour,
    gate: Gate,
    calls: Arc<AtomicUsize>,
}

impl Engine {
    fn new(behaviour: Behaviour) -> Self {
        Self {
            behaviour,
            gate: Gate::default(),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

fn chunk(id: u32) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenResp::Chunk(proto::GenerateStreamChunk {
            token_ids: vec![id],
            prompt_tokens: 1,
            completion_tokens: 1,
            index: 0,
            ..Default::default()
        })),
    }
}

fn complete() -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenResp::Complete(proto::GenerateComplete {
            output_ids: (1..=OUTPUT_TOKENS).collect(),
            finish_reason: "stop".to_string(),
            prompt_tokens: 1,
            completion_tokens: OUTPUT_TOKENS,
            index: 0,
            ..Default::default()
        })),
    }
}

/// Feed one generate stream: the tokens after `delay`, then the completion.
/// With `delay` at [`FOREVER`] the sender is held and nothing is written.
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
fn produce_after(tx: GenerateTx, delay: Duration) {
    tokio::spawn(async move {
        tokio::time::sleep(delay).await;
        for id in 1..=OUTPUT_TOKENS {
            if tx.send(Ok(chunk(id))).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Ok(complete())).await;
    });
}

#[tonic::async_trait]
impl proto::vllm_engine_server::VllmEngine for Engine {
    type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::channel(8);
        match self.behaviour {
            Behaviour::HoldHeaders => {
                tokio::time::sleep(FOREVER).await;
            }
            Behaviour::EmptyStream => produce_after(tx, FOREVER),
            Behaviour::Answer(delay) => produce_after(tx, delay),
            Behaviour::Queued => {
                self.gate.turn().await;
                produce_after(tx, Duration::ZERO);
            }
            Behaviour::HoldHeadersThenFail(delay) => {
                tokio::time::sleep(delay).await;
                return Err(Status::internal("the engine gave up on the request"));
            }
        }
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn abort(
        &self,
        _request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        Ok(Response::new(proto::AbortResponse::default()))
    }

    async fn flush_cache(
        &self,
        _request: Request<smg_grpc_client::common_proto::FlushCacheRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::FlushCacheResponse>, Status> {
        Ok(Response::new(
            smg_grpc_client::common_proto::FlushCacheResponse::default(),
        ))
    }

    async fn embed(
        &self,
        _request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        Ok(Response::new(proto::HealthCheckResponse {
            healthy: true,
            ..Default::default()
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        Ok(Response::new(proto::GetModelInfoResponse::default()))
    }

    async fn get_server_info(
        &self,
        _request: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        Ok(Response::new(proto::GetServerInfoResponse::default()))
    }

    async fn get_loads(
        &self,
        _request: Request<proto::GetLoadsRequest>,
    ) -> Result<Response<proto::GetLoadsResponse>, Status> {
        Ok(Response::new(proto::GetLoadsResponse::default()))
    }

    type GetTokenizerStream =
        ReceiverStream<Result<smg_grpc_client::common_proto::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: Request<smg_grpc_client::common_proto::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    type SubscribeKvEventsStream =
        ReceiverStream<Result<smg_grpc_client::common_proto::KvEventBatch, Status>>;

    async fn subscribe_kv_events(
        &self,
        _request: Request<smg_grpc_client::common_proto::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
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

/// Feed one parallel-dispatch generate stream once the leg's turn at `gate`
/// comes: the tokens, then the completion.
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
fn produce_at_turn(
    tx: mpsc::Sender<Result<ts::GenerateResponse, Status>>,
    gate: Gate,
    request_id: String,
) {
    tokio::spawn(async move {
        gate.turn().await;
        for id in 1..=OUTPUT_TOKENS {
            if tx.send(Ok(ts_chunk(&request_id, id))).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Ok(ts_complete(&request_id))).await;
    });
}

/// A parallel-dispatch mock engine (the TokenSpeed proto, as the mock-worker
/// crate's worker speaks it): the generate call returns its stream at once,
/// and the leg's first chunk waits for its turn at the engine's gate (one
/// leg every `TURN`: the decode side admitting the pair's legs one by one),
/// then the tokens and the completion follow. Counts its generate calls.
#[derive(Clone)]
struct QueuedEngine {
    gate: Gate,
    calls: Arc<AtomicUsize>,
}

impl QueuedEngine {
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[tonic::async_trait]
impl TokenSpeedScheduler for QueuedEngine {
    type GenerateStream = ReceiverStream<Result<ts::GenerateResponse, Status>>;
    type GetTokenizerStream = ReceiverStream<Result<grpc_common::GetTokenizerChunk, Status>>;
    type SubscribeKvEventsStream = ReceiverStream<Result<grpc_common::KvEventBatch, Status>>;

    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::channel(8);
        produce_at_turn(tx, self.gate.clone(), request.into_inner().request_id);
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn health_check(
        &self,
        _request: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        Ok(Response::new(ts::HealthCheckResponse {
            healthy: true,
            message: "ok".to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        Ok(Response::new(ts::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        Ok(Response::new(ts::GetModelInfoResponse::default()))
    }

    async fn get_server_info(
        &self,
        _request: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Ok(Response::new(ts::GetServerInfoResponse::default()))
    }

    async fn get_loads(
        &self,
        _request: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        Ok(Response::new(ts::GetLoadsResponse::default()))
    }

    async fn flush_cache(
        &self,
        _request: Request<grpc_common::FlushCacheRequest>,
    ) -> Result<Response<grpc_common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn start_profile(
        &self,
        _request: Request<grpc_common::StartProfileRequest>,
    ) -> Result<Response<grpc_common::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn stop_profile(
        &self,
        _request: Request<grpc_common::StopProfileRequest>,
    ) -> Result<Response<grpc_common::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn get_tokenizer(
        &self,
        _request: Request<grpc_common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn subscribe_kv_events(
        &self,
        _request: Request<grpc_common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }
}

/// Serve a queued parallel-dispatch mock engine in-process; returns its port
/// and the engine (for its call count).
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_queued_grpc_worker() -> (u16, QueuedEngine) {
    let engine = QueuedEngine {
        gate: Gate::default(),
        calls: Arc::new(AtomicUsize::new(0)),
    };
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind queued mock engine");
    let port = listener
        .local_addr()
        .expect("queued mock engine address")
        .port();
    tokio::spawn(
        Server::builder()
            .add_service(TokenSpeedSchedulerServer::new(engine.clone()))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    (port, engine)
}

/// Serve a sequential-dispatch mock engine in-process; returns its port.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_mock_engine(engine: Engine) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock engine");
    let port = listener.local_addr().expect("mock engine address").port();
    tokio::spawn(
        Server::builder()
            .add_service(proto::vllm_engine_server::VllmEngineServer::new(engine))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    port
}

/// Serve the mock-worker crate's canned worker in-process (a parallel-dispatch
/// runtime once registered as one): one up-front `delay` before its stream is
/// returned, then `OUTPUT_TOKENS` tokens; returns its port.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_mock_grpc_worker(delay: Duration) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock gRPC worker");
    let port = listener
        .local_addr()
        .expect("mock gRPC worker address")
        .port();
    let cfg = Arc::new(mock_worker::config::Config {
        admin_port: None,
        context_length: 32768,
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        gen_delay: delay,
        output_tokens: OUTPUT_TOKENS,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
        ..mock_worker::config::Config::default()
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    port
}

/// A PD gRPC router over the given legs, registered as `runtime` (which picks
/// the dispatch shape), with round-robin leg policies so the first request
/// goes to the first registered worker of each leg.
async fn build_pd_router(legs: &[(u16, WorkerType)], runtime: RuntimeType) -> Box<dyn RouterTrait> {
    build_pd_router_with(legs, runtime, CircuitBreakerConfig::default())
        .await
        .0
}

/// [`build_pd_router`] with every worker's circuit breaker configured as
/// `breaker`; the registered workers come back with the router.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_pd_router_with(
    legs: &[(u16, WorkerType)],
    runtime: RuntimeType,
    breaker: CircuitBreakerConfig,
) -> (Box<dyn RouterTrait>, Vec<Arc<dyn Worker>>) {
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        })
        .grpc_connection()
        .round_robin_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;

    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    let app_context =
        common::create_test_context_with_tokenizer_registry(config, tokenizer_registry).await;
    let mut workers = Vec::with_capacity(legs.len());
    for (port, worker_type) in legs {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
                .worker_type(*worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(runtime)
                .model(ModelCard::new(MODEL))
                .circuit_breaker_config(breaker.clone())
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        app_context
            .worker_registry
            .register(Arc::clone(&worker))
            .unwrap();
        workers.push(worker);
    }
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("PD gRPC router should build");
    (router, workers)
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-prefill-silence-test-tenant"))
}

/// One streaming chat through `router`: its status, body and duration.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn one_streaming_chat(router: &dyn RouterTrait) -> (http::StatusCode, String, Duration) {
    let request = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": true,
    }))
    .unwrap();
    let started = Instant::now();
    let response =
        tokio::time::timeout(CASE_CAP, router.route_chat(None, &tenant(), request, MODEL))
            .await
            .expect(
                "the request must be answered within the case cap, not held to the request timeout",
            );
    let status = response.status();
    let bytes = tokio::time::timeout(
        CASE_CAP,
        axum::body::to_bytes(response.into_body(), 1 << 20),
    )
    .await
    .expect("the response body must end within the case cap")
    .expect("response body");
    let body = String::from_utf8(bytes.to_vec()).expect("utf-8 body");
    (status, body, started.elapsed())
}

/// `n` streaming chats through `router` at once.
async fn concurrent_streaming_chats(
    router: &dyn RouterTrait,
    n: usize,
) -> Vec<(http::StatusCode, String, Duration)> {
    join_all((0..n).map(|_| one_streaming_chat(router))).await
}

/// Every chat answered in full, and the slowest of them later than the
/// bound (a case whose legs never waited past the bound proves nothing),
/// though within the turns its queue needed.
fn assert_all_answered_past_the_bound(answers: &[(http::StatusCode, String, Duration)]) {
    for (status, body, elapsed) in answers {
        assert_eq!(*status, http::StatusCode::OK, "{body} ({elapsed:?})");
        assert!(body.contains("data: [DONE]"), "{body}");
    }
    let slowest = answers
        .iter()
        .map(|(_, _, elapsed)| *elapsed)
        .max()
        .unwrap_or_default();
    assert!(
        slowest >= BOUND,
        "the last leg's turn comes past the bound: {slowest:?}"
    );
    assert!(
        slowest < TURN * u32::try_from(QUEUED_LEGS).unwrap_or(u32::MAX) + SLACK,
        "answered once its turn came: {slowest:?}"
    );
}

/// Run `case` on a current-thread runtime under a thread-local recorder and
/// return the rendered `/metrics` text.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn render_after<F, Fut>(case: F) -> String
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = ()>,
{
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime");
        rt.block_on(case());
    });
    handle.render()
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

/// What a given-up-on request looks like: a 504 under the leg's own code,
/// answered within the bound plus slack.
fn assert_given_up_on(status: http::StatusCode, body: &str, elapsed: Duration) {
    assert_eq!(status, http::StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert!(body.contains("prefill_worker_silent"), "{body}");
    assert!(
        elapsed >= BOUND && elapsed < BOUND + SLACK,
        "given up on at the bound ({BOUND:?}), not before and not much after: {elapsed:?}"
    );
}

/// Sequential dispatch, the prefill call never returns its stream: given up
/// on at the bound, counted as the prefill worker's timeout, the decode never
/// dispatched.
#[test]
fn a_sequential_prefill_that_returns_no_stream_is_given_up_on_within_the_bound() {
    let prefill = Engine::new(Behaviour::HoldHeaders);
    let decode = Engine::new(Behaviour::Answer(Duration::ZERO));
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
        )
        .await;
        let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
        assert_given_up_on(status, &body, elapsed);
    });
    assert_eq!(
        prefill_calls.calls(),
        1,
        "not replayed on the silent prefill"
    );
    assert_eq!(
        decode_calls.calls(),
        0,
        "the decode leg was never dispatched"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        1.0,
        "one timeout on the prefill worker:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "backend_error"),
        0.0,
        "counted once, under one class:\n{rendered}"
    );
}

/// Sequential dispatch, the prefill returns its stream at once and never
/// writes to it: the same verdict, from the first-response phase.
#[test]
fn a_sequential_prefill_whose_stream_stays_empty_is_given_up_on_within_the_bound() {
    let prefill = Engine::new(Behaviour::EmptyStream);
    let decode = Engine::new(Behaviour::Answer(Duration::ZERO));
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
        )
        .await;
        let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
        assert_given_up_on(status, &body, elapsed);
    });
    assert_eq!(
        prefill_calls.calls(),
        1,
        "not replayed on the silent prefill"
    );
    assert_eq!(
        decode_calls.calls(),
        0,
        "the decode leg was never dispatched"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        1.0,
        "one timeout on the prefill worker:\n{rendered}"
    );
}

/// Parallel dispatch (both legs sent together), the prefill never returns
/// its stream: given up on at the bound; the decode leg, dispatched with it,
/// is dropped with the attempt. Two such requests are two consecutive
/// failures of the prefill worker's circuit breaker (its dispatch is not
/// sampled as a success first), so a breaker set to two opens; the decode
/// worker's stays closed.
#[test]
fn a_parallel_prefill_that_returns_no_stream_is_given_up_on_within_the_bound() {
    let rendered = render_after(|| async {
        let prefill_port = start_mock_grpc_worker(FOREVER).await;
        let decode_port = start_mock_grpc_worker(Duration::ZERO).await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::TokenSpeed,
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        )
        .await;
        for _ in 0..2 {
            let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
            assert_given_up_on(status, &body, elapsed);
        }
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Open,
            "two silences are two consecutive failures of the prefill worker"
        );
        assert_eq!(
            workers[1].circuit_breaker_state(),
            CircuitState::Closed,
            "the decode worker did nothing wrong"
        );
    });
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        2.0,
        "one timeout per request on the prefill worker:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "decode", "backend_error"),
        0.0,
        "the decode leg did nothing wrong:\n{rendered}"
    );
}

/// Parallel dispatch, a prefill whose legs queue at the engine's gate (the
/// decode side admitting them one by one): the later legs' first responses
/// come well past the bound, but the worker answers a leg every turn, so
/// none is given up on. Every request is answered, nothing is charged to the
/// prefill worker, and its breaker (set to two) stays closed.
#[test]
fn a_parallel_prefill_whose_legs_queue_behind_the_decode_side_is_not_given_up_on() {
    let rendered = render_after(|| async {
        let (prefill_port, prefill) = start_queued_grpc_worker().await;
        let decode_port = start_mock_grpc_worker(Duration::ZERO).await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::TokenSpeed,
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        )
        .await;
        let answers = concurrent_streaming_chats(router.as_ref(), QUEUED_LEGS).await;
        assert_all_answered_past_the_bound(&answers);
        assert_eq!(prefill.calls(), QUEUED_LEGS, "each leg dispatched once");
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Closed,
            "a backed-up prefill is not a failing one"
        );
    });
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        0.0,
        "no leg was given up on:\n{rendered}"
    );
}

/// Sequential dispatch, the same queue in front of the prefill call (its
/// headers held until the leg's turn): no leg is given up on, every request
/// is answered by the decode, nothing is charged to the prefill worker.
#[test]
fn a_sequential_prefill_whose_legs_queue_is_not_given_up_on() {
    let prefill = Engine::new(Behaviour::Queued);
    let decode = Engine::new(Behaviour::Answer(Duration::ZERO));
    let (prefill_calls, decode_calls) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        )
        .await;
        // The legs are booked on the prefill worker at their dispatch, while
        // their calls are still held at the gate: sampled through the run,
        // the backlog shows them waiting (a booking at the call's return
        // would show nothing: the span from the call's return to the first
        // message, which is the same message, is too short to be sampled or
        // to teach a rate).
        let done = Arc::new(AtomicBool::new(false));
        let sampler = {
            let done = Arc::clone(&done);
            let prefill = Arc::clone(&workers[0]);
            async move {
                let mut largest = Duration::ZERO;
                while !done.load(Ordering::Acquire) {
                    largest = largest.max(prefill.prefill_backlog());
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                largest
            }
        };
        let chats = async {
            let answers = concurrent_streaming_chats(router.as_ref(), QUEUED_LEGS).await;
            done.store(true, Ordering::Release);
            answers
        };
        let (answers, largest_backlog) = tokio::join!(chats, sampler);
        assert!(
            largest_backlog > Duration::ZERO,
            "the held legs are pending prefill on their worker from the dispatch"
        );
        assert_all_answered_past_the_bound(&answers);
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Closed,
            "a backed-up prefill is not a failing one"
        );
    });
    assert_eq!(
        prefill_calls.calls(),
        QUEUED_LEGS,
        "each leg dispatched once"
    );
    assert_eq!(
        decode_calls.calls(),
        QUEUED_LEGS,
        "every request reached its decode"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        0.0,
        "no leg was given up on:\n{rendered}"
    );
}

/// Sequential dispatch, the prefill holds its headers past the bound and
/// then fails: the request charged the worker once, at the verdict. The late
/// failure of the call it gave up on is not a second breaker failure (the
/// breaker, set to two, stays closed) and not a `backend_error`.
#[test]
fn the_late_failure_of_a_leg_given_up_on_is_not_charged_again() {
    let prefill = Engine::new(Behaviour::HoldHeadersThenFail(BOUND + LATE));
    let decode = Engine::new(Behaviour::Answer(Duration::ZERO));
    let prefill_calls = prefill.clone();
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let decode_port = start_mock_engine(decode).await;
        let (router, workers) = build_pd_router_with(
            &[
                (prefill_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
            CircuitBreakerConfig {
                failure_threshold: 2,
                ..Default::default()
            },
        )
        .await;
        let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
        assert_given_up_on(status, &body, elapsed);
        // The retired call lands, and fails, after the verdict.
        tokio::time::sleep(LATE + SLACK).await;
        assert_eq!(
            workers[0].circuit_breaker_state(),
            CircuitState::Closed,
            "one failure, the verdict's; the late one is not a second"
        );
    });
    assert_eq!(
        prefill_calls.calls(),
        1,
        "not replayed on the silent prefill"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        1.0,
        "one timeout on the prefill worker:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "backend_error"),
        0.0,
        "the late failure is not counted again:\n{rendered}"
    );
}

/// With another prefill worker available the request is retried there and
/// answered; the silent prefill is never asked again, and its decode partner
/// is still the one that serves the retry.
#[test]
fn the_request_is_retried_on_another_prefill_worker_and_never_on_the_silent_one_again() {
    let silent = Engine::new(Behaviour::HoldHeaders);
    let healthy = Engine::new(Behaviour::Answer(Duration::from_millis(50)));
    let decode = Engine::new(Behaviour::Answer(Duration::from_millis(50)));
    let (silent_calls, healthy_calls, decode_calls) =
        (silent.clone(), healthy.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let silent_port = start_mock_engine(silent).await;
        let healthy_port = start_mock_engine(healthy).await;
        let decode_port = start_mock_engine(decode).await;
        let router = build_pd_router(
            &[
                (silent_port, WorkerType::Prefill),
                (healthy_port, WorkerType::Prefill),
                (decode_port, WorkerType::Decode),
            ],
            RuntimeType::Vllm,
        )
        .await;
        let (status, body, elapsed) = one_streaming_chat(router.as_ref()).await;
        assert_eq!(status, http::StatusCode::OK, "{body}");
        assert!(body.contains("data: [DONE]"), "{body}");
        assert!(
            elapsed < BOUND + SLACK,
            "one bound on the silent prefill, then the answer: {elapsed:?}"
        );
    });
    assert_eq!(
        silent_calls.calls(),
        1,
        "given up on once, never asked again"
    );
    assert_eq!(
        healthy_calls.calls(),
        1,
        "the retry went to the other prefill"
    );
    assert_eq!(
        decode_calls.calls(),
        1,
        "the decode partner served the retry"
    );
    assert_eq!(
        worker_errors(&rendered, "prefill", "timeout"),
        1.0,
        "one timeout on the silent prefill worker:\n{rendered}"
    );
}
