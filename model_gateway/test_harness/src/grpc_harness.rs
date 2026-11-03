//! The in-process gRPC harness the gateway's integration binaries share: the
//! mock-worker crate's engine (or a binary's own tonic engine) served on an
//! ephemeral loopback port, the gateway's record of such a worker with its
//! health checks off, and a gRPC router over those records with the mock
//! tokenizer loaded for the model. A binary keeps only what its cases set
//! differently.

use std::{convert::Infallible, future::Future, pin::Pin, sync::Arc, time::Duration};

use axum::response::IntoResponse;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use smg::{
    app_context::AppContext,
    config::{RouterConfig, RouterConfigBuilder},
    routers::{RouterFactory, RouterTrait},
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType},
};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{body::Body, server::NamedService, transport::Server};
use tower::Service;

/// The canned mock engine's answer to every request, whatever it asks for:
/// `output_tokens` tokens of `model`, after `gen_delay`.
pub struct Canned {
    pub model: &'static str,
    pub gen_delay: Duration,
    pub output_tokens: u32,
}

/// An ephemeral loopback port, bound, with the port it got. Bound first and
/// the port read back: picking a free port and binding it later would race
/// the other tests of the binary doing the same.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn ephemeral_listener() -> (TcpListener, u16) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock gRPC worker");
    let port = listener
        .local_addr()
        .expect("mock gRPC worker address")
        .port();
    (listener, port)
}

/// Serve the mock-worker crate's gRPC engine in-process, configured by
/// `configure` from the ephemeral port it was given; returns the port. The
/// server task runs for the test process's lifetime.
#[expect(
    clippy::disallowed_methods,
    reason = "test helper - the spawned mock server task is fire-and-forget for the test \
              process's lifetime"
)]
pub async fn serve_mock_worker_with(
    configure: impl FnOnce(u16) -> mock_worker::config::Config,
) -> u16 {
    let (listener, port) = ephemeral_listener().await;
    tokio::spawn(mock_worker::grpc::serve_with_listener(
        Arc::new(configure(port)),
        listener,
    ));
    port
}

/// [`serve_mock_worker_with`] in canned mode: one gRPC engine and no HTTP
/// server, `canned` its answer to everything.
pub async fn serve_canned_worker(canned: Canned) -> u16 {
    serve_mock_worker_with(|port| mock_worker::config::Config {
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        model_id: canned.model.to_string(),
        tokenizer_path: canned.model.to_string(),
        gen_delay: canned.gen_delay,
        output_tokens: canned.output_tokens,
        realistic: false,
        ..mock_worker::config::Config::default()
    })
    .await
}

/// Serve a binary's own tonic engine `service` in-process; returns its port.
/// The server task runs for the test process's lifetime.
#[expect(
    clippy::disallowed_methods,
    reason = "test helper - the spawned mock server task is fire-and-forget for the test \
              process's lifetime"
)]
pub async fn serve_engine<S>(service: S) -> u16
where
    S: Service<http::Request<Body>, Error = Infallible>
        + NamedService
        + Clone
        + Send
        + Sync
        + 'static,
    S::Response: IntoResponse,
    S::Future: Send + 'static,
{
    let (listener, port) = ephemeral_listener().await;
    tokio::spawn(
        Server::builder()
            .add_service(service)
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    port
}

/// Health checks off: the worker is ready as registered.
pub fn health_checks_off() -> HealthCheckConfig {
    HealthCheckConfig {
        disable_health_check: true,
        ..Default::default()
    }
}

/// The gateway's record of the gRPC worker at loopback `port`, its health
/// checks off; the caller adds what its case sets, then builds it.
pub fn grpc_worker(
    port: u16,
    worker_type: WorkerType,
    runtime: RuntimeType,
    model: ModelCard,
) -> BasicWorkerBuilder {
    BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
        .worker_type(worker_type)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(runtime)
        .model(model)
        .health_config(health_checks_off())
}

/// The records of the workers at `workers`, every one of `runtime` and serving
/// `model`.
pub fn grpc_workers(
    workers: &[(u16, WorkerType)],
    runtime: RuntimeType,
    model: &str,
) -> Vec<Arc<dyn Worker>> {
    workers
        .iter()
        .map(|&(port, worker_type)| {
            Arc::new(grpc_worker(port, worker_type, runtime, ModelCard::new(model)).build())
                as Arc<dyn Worker>
        })
        .collect()
}

/// The configuration the binaries' routers start from: a gRPC gateway on an
/// ephemeral loopback port; the caller sets the mode and the policy.
pub fn grpc_router_config() -> RouterConfigBuilder {
    RouterConfig::builder()
        .grpc_connection()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
}

/// The test context a gRPC router is built from: `config` with its health
/// checks off, and `tokenizer` loaded for `model` through `common`'s one
/// registry load. Boxed, as the context builders of `common` are: a
/// `RouterConfig` held across an await would put the caller's future over
/// the crate's size limit.
pub fn grpc_test_context_with_tokenizer(
    mut config: RouterConfig,
    model: &str,
    tokenizer: Arc<dyn Tokenizer>,
) -> Pin<Box<dyn Future<Output = Arc<AppContext>> + Send>> {
    config.health_check.disable_health_check = true;
    let model = model.to_owned();
    Box::pin(async move {
        let tokenizer_registry = super::tokenizer_registry_with(&model, tokenizer).await;
        super::create_test_context_with_tokenizer_registry(config, tokenizer_registry).await
    })
}

/// [`grpc_test_context_with_tokenizer`] with a tenant rate limiter of the
/// caller's (a manager over a backend of the test's) in place of the one
/// `config` would build.
pub fn grpc_test_context_with_rate_limiter(
    mut config: RouterConfig,
    model: &str,
    tokenizer: Arc<dyn Tokenizer>,
    rate_limit_manager: Arc<smg::rate_limit::RateLimitManager>,
) -> Pin<Box<dyn Future<Output = Arc<AppContext>> + Send>> {
    config.health_check.disable_health_check = true;
    let model = model.to_owned();
    Box::pin(async move {
        let tokenizer_registry = super::tokenizer_registry_with(&model, tokenizer).await;
        super::create_test_context_with_rate_limiter(
            config,
            tokenizer_registry,
            Some(rate_limit_manager),
        )
        .await
    })
}

/// [`grpc_test_context_with_tokenizer`] with the mock tokenizer: `common`'s
/// `create_test_context_with_mock_tokenizer`, the health checks off.
pub fn grpc_test_context(
    mut config: RouterConfig,
    model: &str,
) -> Pin<Box<dyn Future<Output = Arc<AppContext>> + Send>> {
    config.health_check.disable_health_check = true;
    super::create_test_context_with_mock_tokenizer(config, model)
}

/// The gRPC router of `app_context` over `workers`, registered in order.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
pub async fn grpc_router(
    app_context: &Arc<AppContext>,
    workers: Vec<Arc<dyn Worker>>,
) -> Box<dyn RouterTrait> {
    for worker in workers {
        app_context
            .worker_registry
            .register(worker)
            .expect("register the worker");
    }
    RouterFactory::create_router(app_context)
        .await
        .expect("gRPC router should build")
}

/// A gRPC router over `workers`, from `config` through [`grpc_test_context`].
pub fn build_grpc_router(
    config: RouterConfig,
    model: &str,
    workers: Vec<Arc<dyn Worker>>,
) -> Pin<Box<dyn Future<Output = Box<dyn RouterTrait>> + Send>> {
    let app_context = grpc_test_context(config, model);
    Box::pin(async move { grpc_router(&app_context.await, workers).await })
}
