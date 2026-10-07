//! A fleet of eight realistic mock gRPC workers reporting loads the way the
//! vLLM servicer does (no queued token-work): with load polling held off,
//! their loads still reach the gateway through the KV-event streams, as the
//! `EngineLoad` record on every batch and as `load_only` heartbeats while
//! the engines are idle, and land in the same `smg_engine_*` gauges a poll
//! fills.

mod common;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use llm_tokenizer::{mock::MockTokenizer, traits::Tokenizer, TokenizerRegistry};
use openai_protocol::worker::{ConnectionMode, HealthCheckConfig, RuntimeType, WorkerType};
use smg::{
    config::RouterConfig,
    observability::{
        metrics::{start_prometheus, PrometheusConfig},
        metrics_server::start_metrics_server,
    },
    worker::{BasicWorkerBuilder, KvEventMonitor, ModelCard},
};
use tokio::net::TcpListener;

const MODEL: &str = "mock-model";
const WORKERS: usize = 8;

/// A realistic mock gRPC worker with prefix caching (so it streams KV events)
/// and vLLM-like load reports.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the mock server task is fire-and-forget"
)]
async fn start_realistic_worker() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock gRPC worker");
    let port = listener
        .local_addr()
        .expect("mock gRPC worker address")
        .port();
    let cfg = Arc::new(mock_worker::config::Config {
        host: "127.0.0.1".to_string(),
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        realistic: true,
        engine: mock_worker::engine::EngineParams {
            prefix_cache: true,
            ..mock_worker::engine::EngineParams::default()
        },
        loads_like: mock_worker::engine::LoadsLike::Vllm,
        ..mock_worker::config::Config::default()
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    port
}

#[tokio::test]
async fn eight_workers_loads_reach_the_gateway_through_the_event_streams() {
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    });
    let (metrics_addr, _metrics_server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");

    let mut ports = Vec::with_capacity(WORKERS);
    for _ in 0..WORKERS {
        ports.push(start_realistic_worker().await);
    }

    // Cache-aware routing (the KV-event subscriptions) with the load poll held
    // off: the monitor's loops are never started, so every load the gateway
    // learns came through an event stream.
    let mut config = RouterConfig::builder()
        .grpc_connection()
        .cache_aware_policy(0.5, 32, 1.1, 60, 1_000_000)
        .load_monitor_interval_secs(3600)
        .host("127.0.0.1")
        .port(0)
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
    let worker_monitor = app_context
        .worker_monitor
        .clone()
        .expect("the test context builds a worker monitor");
    let kv_monitor = Arc::new(KvEventMonitor::new(None));
    kv_monitor.set_load_sink(&worker_monitor);

    let mut urls = Vec::with_capacity(WORKERS);
    for port in &ports {
        let url = format!("grpc://127.0.0.1:{port}");
        let worker = BasicWorkerBuilder::new(url.clone())
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        let worker: Arc<dyn smg::worker::Worker> = Arc::new(worker);
        worker.set_status(openai_protocol::worker::WorkerStatus::Ready);
        app_context
            .worker_registry
            .register(Arc::clone(&worker))
            .unwrap();
        kv_monitor.on_worker_added(&worker).await;
        urls.push(url);
    }

    // Idle engines publish no KV events; their streams still send the load
    // record as heartbeats within a second or so of the subscription.
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("test client");
    let deadline = Instant::now() + Duration::from_secs(15);
    let body = loop {
        let body = client
            .get(format!("http://{metrics_addr}/metrics"))
            .send()
            .await
            .expect("metrics endpoint reachable")
            .text()
            .await
            .expect("metrics body");
        let reported = urls
            .iter()
            .filter(|url| {
                body.lines().any(|line| {
                    line.starts_with("smg_engine_running_requests{")
                        && line.contains(&format!("worker=\"{url}\""))
                })
            })
            .count();
        // The records come as load_only batches (the engines are idle) that
        // the gateway counts as such, admitting none into the index. A
        // worker's first load may reach the gateway by its poll before its
        // stream's heartbeat, so the batch count is waited for as well.
        let load_only: f64 = body
            .lines()
            .filter(|line| {
                line.starts_with("smg_kv_event_batches_total{")
                    && line.contains("disposition=\"load_only\"")
            })
            .filter_map(|line| line.rsplit(' ').next()?.parse::<f64>().ok())
            .sum();
        if reported == WORKERS && load_only >= WORKERS as f64 {
            break body;
        }
        assert!(
            Instant::now() < deadline,
            "{reported} of {WORKERS} workers reported a load and {load_only} load_only \
             batches arrived through the event streams:\n{body}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    for url in &urls {
        let usage = body
            .lines()
            .find(|line| {
                line.starts_with("smg_engine_token_usage{")
                    && line.contains(&format!("worker=\"{url}\""))
            })
            .unwrap_or_else(|| panic!("token usage gauge for {url} missing:\n{body}"));
        let value: f64 = usage.rsplit(' ').next().unwrap().parse().unwrap();
        assert!((0.0..=1.0).contains(&value), "{usage}");
    }
    kv_monitor.stop().await;
}
