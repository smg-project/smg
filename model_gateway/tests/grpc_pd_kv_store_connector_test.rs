//! A disaggregated pair whose legs run a KV-store connector: the prefill
//! writes its blocks to a shared store and the decode reads them back by
//! prefix hash, so the gateway tags neither leg, mints no handoff and runs
//! the decode after the prefill. A store leg never pairs with a rendezvous
//! leg of the same connector family.
//!
//! Each case runs on a current-thread runtime under a thread-local Prometheus
//! recorder, so the request path's mode decision lands in the rendered text.

#[path = "common/mod.rs"]
mod common;

use std::{
    future::Future,
    net::TcpListener as StdTcpListener,
    sync::{Arc, Mutex},
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
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use smg_grpc_client::vllm_proto as proto;
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{transport::Server, Request, Response, Status};

const MODEL: &str = "pd-kv-store-connector-test-model";
/// Tokens every mock answers with; ids 1..=3 are in the mock tokenizer's vocabulary.
const OUTPUT_TOKENS: u32 = 3;
/// The store connector of the Mooncake family, and the family's rendezvous
/// connector.
const STORE_CONNECTOR: &str = "MooncakeStoreConnector";
const RENDEZVOUS_CONNECTOR: &str = "MooncakeConnector";

/// The KV-transfer params one generate call carried: the JSON field, and
/// whether the legacy typed field was set.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SeenParams {
    json: Option<String>,
    typed: bool,
}

impl SeenParams {
    fn none() -> Self {
        Self {
            json: None,
            typed: false,
        }
    }
}

/// A sequential-dispatch mock engine that records the KV-transfer params of
/// every generate call it receives and answers with `OUTPUT_TOKENS` tokens.
#[derive(Clone, Default)]
struct RecordingEngine {
    seen: Arc<Mutex<Vec<SeenParams>>>,
}

impl RecordingEngine {
    #[expect(
        clippy::expect_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    fn seen(&self) -> Vec<SeenParams> {
        self.seen.lock().expect("recording lock").clone()
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

#[tonic::async_trait]
impl proto::vllm_engine_server::VllmEngine for RecordingEngine {
    type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;

    #[expect(
        clippy::expect_used,
        reason = "test mock - panicking on failure is intentional"
    )]
    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request = request.into_inner();
        self.seen.lock().expect("recording lock").push(SeenParams {
            json: request.kv_transfer_params_json.clone(),
            typed: request.kv_transfer_params.is_some(),
        });
        // Room for the tokens and the completion: the stream is filled before
        // it is returned.
        let (tx, rx) = mpsc::channel(usize::try_from(OUTPUT_TOKENS).unwrap_or(8) + 1);
        for id in 1..=OUTPUT_TOKENS {
            let _ = tx.send(Ok(chunk(id))).await;
        }
        let _ = tx.send(Ok(complete())).await;
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

/// Serve a recording mock engine in-process; returns its port.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_mock_engine(engine: RecordingEngine) -> u16 {
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

/// A loopback port nothing listens on: the bootstrap port of the rendezvous
/// recipe, registered on a leg that never rendezvous.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn closed_port() -> u16 {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind");
    listener.local_addr().expect("address").port()
}

/// One leg of the pair as it is registered: its mock engine's port, its type
/// and its KV connector; a prefill leg also carries the engine id and the
/// bootstrap port the rendezvous recipe registers.
struct Leg {
    port: u16,
    worker_type: WorkerType,
    connector: &'static str,
    bootstrap_port: Option<u16>,
}

/// A PD gRPC router over the given legs, all of the sequential-dispatch
/// runtime, with pairing in its default mode.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_pd_router(legs: &[Leg]) -> Box<dyn RouterTrait> {
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
    for leg in legs {
        let mut builder = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{}", leg.port))
            .worker_type(leg.worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Vllm)
            .kv_connector(leg.connector)
            .bootstrap_port(leg.bootstrap_port)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            });
        if leg.worker_type == WorkerType::Prefill {
            builder = builder.kv_engine_id("prefill-0");
        }
        app_context
            .worker_registry
            .register(Arc::new(builder.build()))
            .unwrap();
    }
    RouterFactory::create_router(&app_context)
        .await
        .expect("PD gRPC router should build")
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-kv-store-connector-test-tenant"))
}

/// One streaming chat through `router`: its status and the whole body.
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
    let response = router.route_chat(None, &tenant(), request, MODEL).await;
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
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

/// The `smg_pd_kv_connector_mode_total` sample for `mode`, if emitted.
fn mode_count(rendered: &str, mode: &str) -> Option<u64> {
    let label = format!("mode=\"{mode}\"");
    rendered
        .lines()
        .find(|line| line.starts_with("smg_pd_kv_connector_mode_total{") && line.contains(&label))
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
}

/// Both legs on the store, the prefill registered with the bootstrap port
/// the rendezvous recipe carries: the pair is placed, neither leg's request
/// carries KV-transfer params (the decode reads the store by prefix hash),
/// the answer streams, and the mode is counted as `store`.
#[test]
fn a_store_pair_runs_prefill_then_decode_with_no_handoff_params() {
    let prefill = RecordingEngine::default();
    let decode = RecordingEngine::default();
    let (prefill_engine, decode_engine) = (prefill.clone(), decode.clone());
    let rendered = render_after(|| async move {
        let prefill_port = start_mock_engine(prefill_engine).await;
        let decode_port = start_mock_engine(decode_engine).await;
        let router = build_pd_router(&[
            Leg {
                port: prefill_port,
                worker_type: WorkerType::Prefill,
                connector: STORE_CONNECTOR,
                bootstrap_port: Some(closed_port()),
            },
            Leg {
                port: decode_port,
                worker_type: WorkerType::Decode,
                connector: STORE_CONNECTOR,
                bootstrap_port: None,
            },
        ])
        .await;
        let (status, body) = one_streaming_chat(router.as_ref()).await;
        assert_eq!(status, http::StatusCode::OK, "{body}");
        assert!(body.contains("data: [DONE]"), "{body}");
    });
    assert_eq!(prefill.seen(), [SeenParams::none()], "the prefill leg");
    assert_eq!(decode.seen(), [SeenParams::none()], "the decode leg");
    assert_eq!(mode_count(&rendered, "store"), Some(1), "{rendered}");
    assert_eq!(mode_count(&rendered, "passthrough"), None, "{rendered}");
}

/// A rendezvous prefill and a store decode of the same connector family
/// cannot hand KV to each other: the pair is refused at placement, instead of
/// the decode leg being handed a bootstrap address its connector never dials
/// while the prefill pins its blocks for a pull that never comes.
#[test]
fn a_store_leg_never_pairs_with_a_rendezvous_leg_of_the_same_family() {
    let prefill = RecordingEngine::default();
    let decode = RecordingEngine::default();
    let (prefill_engine, decode_engine) = (prefill.clone(), decode.clone());
    render_after(|| async move {
        let prefill_port = start_mock_engine(prefill_engine).await;
        let decode_port = start_mock_engine(decode_engine).await;
        let router = build_pd_router(&[
            Leg {
                port: prefill_port,
                worker_type: WorkerType::Prefill,
                connector: RENDEZVOUS_CONNECTOR,
                bootstrap_port: Some(closed_port()),
            },
            Leg {
                port: decode_port,
                worker_type: WorkerType::Decode,
                connector: STORE_CONNECTOR,
                bootstrap_port: None,
            },
        ])
        .await;
        let (status, body) = one_streaming_chat(router.as_ref()).await;
        assert_eq!(status, http::StatusCode::SERVICE_UNAVAILABLE, "{body}");
        assert!(body.contains("no_compatible_pd_pair"), "{body}");
    });
    assert!(prefill.seen().is_empty(), "{:?}", prefill.seen());
    assert!(decode.seen().is_empty(), "{:?}", decode.seen());
}
