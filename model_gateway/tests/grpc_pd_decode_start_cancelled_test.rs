//! A disaggregated decode leg whose start the worker's side cancelled (its
//! connection closed under the call: a worker dying as the request arrived)
//! is a failure of that worker, not a rejection of the request: with another
//! decode worker available the request is retried there and answered in
//! full, the cancelled worker is charged a backend error, and nothing counts
//! as a validation error; with no other decode worker the client reads a
//! 502 under the leg's own code, never a 400 it would take for its own
//! mistake and not retry.
//!
//! The mock engines speak the vLLM proto (sequential dispatch): the prefill
//! leg completes, one decode leg cancels every call before its stream.

#[path = "common/mod.rs"]
mod common;

use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use proto::generate_response::Response as GenResp;
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType},
};
use smg_grpc_client::vllm_proto as proto;
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{transport::Server, Request, Response, Status};

const MODEL: &str = "pd-decode-start-cancelled-test-model";
/// Tokens an engine that completes produces; ids 1..=3 are in the mock
/// tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// Longer than any case: a request held past it fails the case instead of
/// hanging it for the request timeout.
const CASE_CAP: Duration = Duration::from_secs(30);
/// The words tonic puts on a call whose connection closed under it.
const CANCEL_MESSAGE: &str = "operation was canceled";

type GenerateTx = mpsc::Sender<Result<proto::GenerateResponse, Status>>;

/// What a mock engine does with a generate call.
#[derive(Clone, Copy)]
enum Behaviour {
    /// `OUTPUT_TOKENS` tokens, then a `stop` completion.
    Completes,
    /// The call fails with `CANCELLED` before any stream: what the gateway's
    /// client sees when the worker's connection closes under the call.
    CancelsStart,
}

/// A mock engine counting its generate calls.
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

fn complete(finish_reason: &str, tokens: u32) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenResp::Complete(proto::GenerateComplete {
            output_ids: (1..=tokens).collect(),
            finish_reason: finish_reason.to_string(),
            prompt_tokens: 1,
            completion_tokens: tokens,
            index: 0,
            ..Default::default()
        })),
    }
}

/// Feed one generate stream: the tokens, then the completion.
#[expect(
    clippy::disallowed_methods,
    reason = "test mock; the producer task ends with its stream"
)]
fn produce(tx: GenerateTx) {
    tokio::spawn(async move {
        for id in 1..=OUTPUT_TOKENS {
            if tx.send(Ok(chunk(id))).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Ok(complete("stop", OUTPUT_TOKENS))).await;
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
        if matches!(self.behaviour, Behaviour::CancelsStart) {
            return Err(Status::cancelled(CANCEL_MESSAGE));
        }
        let (tx, rx) = mpsc::channel(8);
        produce(tx);
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

/// Serve a mock engine in-process; returns its port.
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

/// A PD gRPC router over the given legs (sequential dispatch).
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_pd_router(legs: &[(u16, WorkerType)]) -> Box<dyn RouterTrait> {
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
    for (port, worker_type) in legs {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
                .worker_type(*worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::Vllm)
                .model(ModelCard::new(MODEL))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        app_context.worker_registry.register(worker).unwrap();
    }
    RouterFactory::create_router(&app_context)
        .await
        .expect("PD gRPC router should build")
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-decode-start-cancelled-test-tenant"))
}

/// One streaming chat through `router`: its status and body.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn one_streaming_chat(router: &dyn RouterTrait) -> (http::StatusCode, String) {
    let request = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": true,
    }))
    .unwrap();
    let response =
        tokio::time::timeout(CASE_CAP, router.route_chat(None, &tenant(), request, MODEL))
            .await
            .expect("the request must be answered within the case cap");
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
    )
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

/// The sum of a counter's samples carrying every one of `labels`; zero when
/// none was emitted.
fn counter(rendered: &str, name: &str, labels: &[String]) -> f64 {
    let prefix = format!("{name}{{");
    rendered
        .lines()
        .filter(|line| line.starts_with(&prefix))
        .filter(|line| labels.iter().all(|label| line.contains(label.as_str())))
        .filter_map(|line| line.rsplit(' ').next())
        .filter_map(|value| value.parse::<f64>().ok())
        .sum()
}

/// `smg_worker_errors_total` for `worker_type` and `error_type`.
fn worker_errors(rendered: &str, worker_type: &str, error_type: &str) -> f64 {
    counter(
        rendered,
        "smg_worker_errors_total",
        &[
            format!("worker_type=\"{worker_type}\""),
            format!("error_type=\"{error_type}\""),
        ],
    )
}

/// `smg_router_request_errors_total` for `error_type` on the PD backend.
fn router_errors(rendered: &str, error_type: &str) -> f64 {
    counter(
        rendered,
        "smg_router_request_errors_total",
        &[
            "backend_type=\"pd\"".to_string(),
            format!("error_type=\"{error_type}\""),
        ],
    )
}

/// The first decode worker cancels the leg's start; the request is retried
/// on the second and answered in full. The cancelled worker is charged one
/// backend error and never asked again; nothing counts as a validation
/// error.
#[test]
fn a_decode_start_the_worker_cancelled_is_retried_on_another_decode_worker() {
    let prefill = Engine::new(Behaviour::Completes);
    let cancelling = Engine::new(Behaviour::CancelsStart);
    let healthy = Engine::new(Behaviour::Completes);
    let (cancelling_calls, healthy_calls) = (cancelling.clone(), healthy.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let cancelling_port = start_mock_engine(cancelling).await;
        let healthy_port = start_mock_engine(healthy).await;
        let router = build_pd_router(&[
            (prefill_port, WorkerType::Prefill),
            (cancelling_port, WorkerType::Decode),
            (healthy_port, WorkerType::Decode),
        ])
        .await;
        let (status, body) = one_streaming_chat(router.as_ref()).await;
        assert_eq!(status, http::StatusCode::OK, "{body}");
        assert!(body.contains(r#""finish_reason":"stop""#), "{body}");
        assert!(body.contains("data: [DONE]"), "{body}");
        assert!(!body.contains(r#""error""#), "{body}");
    });
    assert_eq!(
        cancelling_calls.calls(),
        1,
        "cancelled once, never asked again"
    );
    assert_eq!(
        healthy_calls.calls(),
        1,
        "the retry went to the other decode worker"
    );
    assert_eq!(
        worker_errors(&rendered, "decode", "backend_error"),
        1.0,
        "the cancelled worker is charged one backend error:\n{rendered}"
    );
    assert_eq!(
        worker_errors(&rendered, "decode", "validation_error"),
        0.0,
        "{rendered}"
    );
    assert_eq!(
        router_errors(&rendered, "validation_error"),
        0.0,
        "nothing about the request was wrong:\n{rendered}"
    );
}

/// With no other decode worker, the cancelled start reaches the client as
/// an upstream failure under the leg's own code: a 502 it may retry, never a
/// 400 it would take for its own mistake.
#[test]
fn a_decode_start_the_worker_cancelled_with_no_other_worker_is_a_502_not_a_400() {
    let prefill = Engine::new(Behaviour::Completes);
    let cancelling = Engine::new(Behaviour::CancelsStart);
    let cancelling_calls = cancelling.clone();
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill).await;
        let cancelling_port = start_mock_engine(cancelling).await;
        let router = build_pd_router(&[
            (prefill_port, WorkerType::Prefill),
            (cancelling_port, WorkerType::Decode),
        ])
        .await;
        let (status, body) = one_streaming_chat(router.as_ref()).await;
        assert_eq!(status, http::StatusCode::BAD_GATEWAY, "{body}");
        assert!(body.contains("decode_worker_failed_to_start"), "{body}");
        assert!(body.contains(CANCEL_MESSAGE), "{body}");
        assert!(
            !body.contains("invalid_request_error"),
            "not the request's fault:\n{body}"
        );
    });
    assert!(
        cancelling_calls.calls() >= 1,
        "the only decode worker was asked at least once"
    );
    assert!(
        worker_errors(&rendered, "decode", "backend_error") >= 1.0,
        "the cancelled worker is charged:\n{rendered}"
    );
    assert_eq!(
        router_errors(&rendered, "validation_error"),
        0.0,
        "{rendered}"
    );
    assert!(
        router_errors(&rendered, "backend_error") >= 1.0,
        "the request counts as a backend error on the PD backend:\n{rendered}"
    );
}
