//! The PD path's time to first token splits into phases the gateway bounds
//! itself, checked against mock prefill and decode workers whose delays are
//! known.
//!
//! Sequential dispatch (the gateway relays the legs): the prefill, relay and
//! decode phases add up to `smg_pd_ttft_seconds` exactly and each phase carries
//! the delay its own worker added, whether the decode worker returns its
//! response headers at once or holds them until its first token. Parallel
//! dispatch (both legs sent together): the decode phase is the whole TTFT.
//!
//! Each case runs on a current-thread runtime under a thread-local Prometheus
//! recorder, so the request path's emissions land in the rendered text.

#[path = "common/mod.rs"]
mod common;

use std::{future::Future, sync::Arc, time::Duration};

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use proto::generate_response::Response as GenResp;
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use smg_grpc_client::vllm_proto as proto;
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{transport::Server, Request, Response, Status};

const MODEL: &str = "pd-ttft-phases-test-model";
/// Tokens every mock answers with; ids 1..=3 are in the mock tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// Delay the prefill worker adds before it answers.
const PREFILL_DELAY: Duration = Duration::from_millis(200);
/// Delay the decode worker adds before its first token.
const DECODE_DELAY: Duration = Duration::from_millis(400);
/// Scheduling slack for a loaded machine. Below either delay, so a phase that
/// absorbed the other worker's delay fails its upper bound.
const SLACK: Duration = Duration::from_millis(250);

type GenerateTx = mpsc::Sender<Result<proto::GenerateResponse, Status>>;

/// A sequential-dispatch mock engine: every generate call answers with
/// `OUTPUT_TOKENS` chunks and a completion after `delay`. With `hold_headers`
/// the call itself waits out the delay before it returns the stream, like an
/// engine server that flushes its response headers with the first message;
/// without it the stream is returned at once and the delay runs inside it.
#[derive(Clone)]
struct DelayedEngine {
    delay: Duration,
    hold_headers: bool,
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
impl proto::vllm_engine_server::VllmEngine for DelayedEngine {
    type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let (tx, rx) = mpsc::channel(8);
        let in_stream_delay = if self.hold_headers {
            tokio::time::sleep(self.delay).await;
            Duration::ZERO
        } else {
            self.delay
        };
        produce_after(tx, in_stream_delay);
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

/// Serve a sequential-dispatch mock engine in-process; returns its port.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_mock_engine(engine: DelayedEngine) -> u16 {
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
/// runtime once registered as one): one up-front `delay`, then `OUTPUT_TOKENS`
/// tokens; returns its port.
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

/// A PD gRPC router over one prefill and one decode worker registered as
/// `runtime`, which picks the dispatch shape.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_pd_router(
    prefill_port: u16,
    decode_port: u16,
    runtime: RuntimeType,
) -> Box<dyn RouterTrait> {
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        })
        .grpc_connection()
        .random_policy()
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
    for (port, worker_type) in [
        (prefill_port, WorkerType::Prefill),
        (decode_port, WorkerType::Decode),
    ] {
        let worker = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(runtime)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        app_context
            .worker_registry
            .register(Arc::new(worker))
            .unwrap();
    }
    RouterFactory::create_router(&app_context)
        .await
        .expect("PD gRPC router should build")
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-ttft-phases-test-tenant"))
}

/// One streaming chat through `router`, read to its `[DONE]`.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn one_streaming_chat(router: &dyn RouterTrait) {
    let request = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 8,
        "stream": true,
    }))
    .unwrap();
    let response = router.route_chat(None, &tenant(), request, MODEL).await;
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    let body = String::from_utf8(bytes.to_vec()).expect("utf-8 body");
    assert_eq!(status, http::StatusCode::OK, "{body}");
    assert!(body.contains("data: [DONE]"), "{body}");
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

/// The `_sum` or `_count` sample of `family` for `runtime`, if it was emitted.
fn sample(rendered: &str, family: &str, field: &str, runtime: &str) -> Option<f64> {
    let prefix = format!("{family}_{field}{{");
    let label = format!("runtime=\"{runtime}\"");
    rendered
        .lines()
        .find(|line| line.starts_with(&prefix) && line.contains(&label))
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
}

/// The four PD TTFT families of one request, in seconds.
struct Phases {
    prefill: Option<f64>,
    relay: Option<f64>,
    decode: Option<f64>,
    ttft: Option<f64>,
}

impl Phases {
    fn of(rendered: &str, runtime: &str) -> Self {
        let sum = |family: &str| sample(rendered, family, "sum", runtime);
        for family in [
            "smg_pd_prefill_duration_seconds",
            "smg_pd_kv_transfer_duration_seconds",
            "smg_pd_decode_ttft_seconds",
            "smg_pd_ttft_seconds",
        ] {
            let count = sample(rendered, family, "count", runtime);
            assert!(
                count.is_none_or(|count| count == 1.0),
                "{family} recorded {count:?} times for runtime {runtime}:\n{rendered}"
            );
        }
        Self {
            prefill: sum("smg_pd_prefill_duration_seconds"),
            relay: sum("smg_pd_kv_transfer_duration_seconds"),
            decode: sum("smg_pd_decode_ttft_seconds"),
            ttft: sum("smg_pd_ttft_seconds"),
        }
    }

    /// Share of the TTFT the prefill and relay families alone explain, in percent.
    fn explained_without_decode(&self) -> f64 {
        let (Some(ttft), Some(prefill)) = (self.ttft, self.prefill) else {
            return 0.0;
        };
        (prefill + self.relay.unwrap_or(0.0)) / ttft * 100.0
    }
}

fn within(value: f64, at_least: Duration, slack: Duration) -> bool {
    value >= at_least.as_secs_f64() && value < (at_least + slack).as_secs_f64()
}

/// Print one case's phases: the measured values are the evidence this suite
/// produces, next to its pass/fail.
#[expect(
    clippy::print_stderr,
    reason = "the measured phases are the evidence this suite produces"
)]
fn report(label: &str, phases: &Phases) {
    eprintln!(
        "pd phases ({label}): prefill={:?} relay={:?} decode={:?} ttft={:?} \
         explained without the decode phase={:.1}%",
        phases.prefill,
        phases.relay,
        phases.decode,
        phases.ttft,
        phases.explained_without_decode()
    );
}

/// Sequential dispatch with a worker that returns its stream at once: the
/// prefill phase carries the prefill worker's delay, the relay phase is the
/// gateway's own work, the decode phase carries the decode worker's delay,
/// and the three add up to the TTFT exactly.
#[test]
fn sequential_dispatch_phases_add_up_to_the_pd_ttft() {
    let rendered = render_after(|| async {
        let prefill = start_mock_engine(DelayedEngine {
            delay: PREFILL_DELAY,
            hold_headers: false,
        })
        .await;
        let decode = start_mock_engine(DelayedEngine {
            delay: DECODE_DELAY,
            hold_headers: false,
        })
        .await;
        let router = build_pd_router(prefill, decode, RuntimeType::Vllm).await;
        one_streaming_chat(router.as_ref()).await;
    });
    let phases = Phases::of(&rendered, RuntimeType::Vllm.as_str());
    report("sequential, eager headers", &phases);
    let (Some(prefill), Some(relay), Some(decode), Some(ttft)) =
        (phases.prefill, phases.relay, phases.decode, phases.ttft)
    else {
        panic!("a PD phase family is missing:\n{rendered}");
    };
    assert!(
        within(prefill, PREFILL_DELAY, SLACK),
        "prefill phase {prefill}"
    );
    assert!(relay < SLACK.as_secs_f64(), "relay phase {relay}");
    assert!(within(decode, DECODE_DELAY, SLACK), "decode phase {decode}");
    assert!(
        (ttft - (prefill + relay + decode)).abs() < 1e-6,
        "ttft {ttft} != prefill {prefill} + relay {relay} + decode {decode}"
    );
}

/// A decode worker that holds its response headers until its first token puts
/// that wait in the decode phase, not in the relay phase.
#[test]
fn a_held_header_decode_leg_stays_in_the_decode_phase() {
    let rendered = render_after(|| async {
        let prefill = start_mock_engine(DelayedEngine {
            delay: PREFILL_DELAY,
            hold_headers: false,
        })
        .await;
        let decode = start_mock_engine(DelayedEngine {
            delay: DECODE_DELAY,
            hold_headers: true,
        })
        .await;
        let router = build_pd_router(prefill, decode, RuntimeType::Vllm).await;
        one_streaming_chat(router.as_ref()).await;
    });
    let phases = Phases::of(&rendered, RuntimeType::Vllm.as_str());
    report("sequential, held headers", &phases);
    let (Some(prefill), Some(relay), Some(decode), Some(ttft)) =
        (phases.prefill, phases.relay, phases.decode, phases.ttft)
    else {
        panic!("a PD phase family is missing:\n{rendered}");
    };
    assert!(
        within(prefill, PREFILL_DELAY, SLACK),
        "prefill phase {prefill}"
    );
    assert!(relay < SLACK.as_secs_f64(), "relay phase {relay}");
    assert!(within(decode, DECODE_DELAY, SLACK), "decode phase {decode}");
    assert!(
        (ttft - (prefill + relay + decode)).abs() < 1e-6,
        "ttft {ttft} != prefill {prefill} + relay {relay} + decode {decode}"
    );
}

/// Parallel dispatch sends both legs together: the decode phase is the TTFT
/// itself, and the prefill and relay phases of the sequential shape are not
/// recorded.
#[test]
fn parallel_dispatch_reports_the_decode_phase_as_the_whole_ttft() {
    let rendered = render_after(|| async {
        let prefill = start_mock_grpc_worker(DECODE_DELAY).await;
        let decode = start_mock_grpc_worker(DECODE_DELAY).await;
        let router = build_pd_router(prefill, decode, RuntimeType::TokenSpeed).await;
        one_streaming_chat(router.as_ref()).await;
    });
    let phases = Phases::of(&rendered, RuntimeType::TokenSpeed.as_str());
    report("parallel", &phases);
    let (Some(decode), Some(ttft)) = (phases.decode, phases.ttft) else {
        panic!("the decode phase or the TTFT is missing:\n{rendered}");
    };
    assert!(ttft >= DECODE_DELAY.as_secs_f64(), "ttft {ttft}");
    assert!(
        (decode - ttft).abs() < 1e-9,
        "decode phase {decode} != ttft {ttft}"
    );
    assert!(
        phases.prefill.is_none() && phases.relay.is_none(),
        "parallel dispatch has no prefill or relay phase:\n{rendered}"
    );
}
