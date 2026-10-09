//! `smg_worker_requests_total{worker, model}` moves once per dispatch.
//!
//! The per-worker gauges (`smg_worker_requests_active`, the engine's running
//! and waiting counts) miss short requests and cannot be summed over a
//! window, and `smg_worker_selection_total` has no worker label, so the share
//! of traffic per worker could not be read from a scrape. The counter exists
//! from a worker's registration and counts every send to it, on the HTTP and
//! on the gRPC path; this drives both through the real routers against
//! in-process upstreams and reads the counter from the metrics endpoint.
//!
//! This file is its own test binary (process), so installing the global
//! Prometheus recorder here is safe and does not collide with other suites.

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test failures should panic; the upstream tasks live for the test process"
)]

#[path = "common/mod.rs"]
mod common;

use std::{net::SocketAddr, sync::Arc};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
    routing::post,
    Json, Router,
};
use common::test_app::{create_test_app_context, create_test_app_with_context};
use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{
    completion::CompletionRequest, model_card::ModelCard, worker::HealthCheckConfig,
};
use serde_json::json;
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    observability::{
        metrics::{start_prometheus, PrometheusConfig},
        metrics_server::start_metrics_server,
    },
    routers::{
        factory::router_ids, gateway::Gateway, http::router::Router as HttpRouter, RouterFactory,
        RouterTrait,
    },
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType},
};
use tokio::net::TcpListener;
use tower::ServiceExt;

const MODEL: &str = "counter-test-model";

async fn scrape(addr: SocketAddr) -> String {
    reqwest::get(format!("http://{addr}/metrics"))
        .await
        .expect("metrics endpoint reachable")
        .text()
        .await
        .expect("metrics body")
}

/// The value of the worker's request counter in a scrape, if the series exists.
fn requests_total(body: &str, worker_url: &str) -> Option<u64> {
    let label = format!("worker=\"{worker_url}\"");
    body.lines()
        .find(|line| line.starts_with("smg_worker_requests_total{") && line.contains(&label))
        .map(|line| {
            assert!(
                line.contains(&format!("model=\"{MODEL}\"")),
                "counter without the model label: {line}"
            );
            line.rsplit(' ').next().unwrap().parse().unwrap()
        })
}

fn health_checks_off() -> HealthCheckConfig {
    HealthCheckConfig {
        disable_health_check: true,
        ..Default::default()
    }
}

/// An HTTP upstream that answers every `/generate` with a completion.
async fn start_http_upstream() -> String {
    let app = Router::new().route(
        "/generate",
        post(|| async { Json(json!({"text": "ok", "meta_info": {"finish_reason": "stop"}})) }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    url
}

/// The production HTTP app with one regular worker at `upstream`.
async fn http_app(upstream: &str) -> Router {
    let ctx = create_test_app_context().await;
    ctx.worker_registry
        .register(Arc::new(
            BasicWorkerBuilder::new(upstream)
                .model(ModelCard::new(MODEL))
                .health_config(health_checks_off())
                .build(),
        ))
        .unwrap();
    let router = Arc::new(HttpRouter::new(&ctx).await.unwrap());
    let gateway = Arc::new(Gateway::new(ctx.worker_registry.clone()));
    gateway.register_router(router_ids::HTTP_REGULAR, router);
    create_test_app_with_context(gateway, ctx)
}

fn generate_request() -> Request {
    Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"text": "hello", "sampling_params": {"max_new_tokens": 2}}).to_string(),
        ))
        .unwrap()
}

/// An in-process mock gRPC engine that answers any prompt.
async fn start_grpc_upstream() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let cfg = Arc::new(mock_worker::config::Config {
        admin_port: None,
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        zmq_count: 0,
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        output_tokens: 2,
        realistic: false,
        ..mock_worker::config::Config::default()
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    format!("grpc://127.0.0.1:{port}")
}

/// The production gRPC router with one regular worker at `upstream`.
async fn grpc_router(upstream: &str) -> Box<dyn RouterTrait> {
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
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
    let ctx = common::create_test_context_with_tokenizer_registry(config, tokenizer_registry).await;
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(upstream)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(health_checks_off())
            .build(),
    );
    ctx.worker_registry.register(worker).unwrap();
    RouterFactory::create_router(&ctx)
        .await
        .expect("gRPC router should build")
}

fn completion() -> CompletionRequest {
    serde_json::from_value(json!({
        "model": MODEL,
        "prompt": "Hello Hello Hello",
        "max_tokens": 2,
        "stream": false,
    }))
    .unwrap()
}

#[tokio::test]
async fn every_dispatch_moves_the_worker_request_counter() {
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    });
    let (metrics_addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");

    // HTTP: the counter exists at zero from the registration, and moves once
    // per request sent to the worker.
    let http_upstream = start_http_upstream().await;
    let app = http_app(&http_upstream).await;
    let body = scrape(metrics_addr).await;
    assert_eq!(
        requests_total(&body, &http_upstream),
        Some(0),
        "counter missing or non-zero after registration:\n{body}"
    );
    for _ in 0..3 {
        let response = app.clone().oneshot(generate_request()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
    let body = scrape(metrics_addr).await;
    assert_eq!(
        requests_total(&body, &http_upstream),
        Some(3),
        "three HTTP dispatches:\n{body}"
    );

    // gRPC: the same counter, on the request-execution stage.
    let grpc_upstream = start_grpc_upstream().await;
    let router = grpc_router(&grpc_upstream).await;
    let tenant = TenantRequestMeta::new(TenantKey::new("counter-test-tenant"));
    for _ in 0..2 {
        let response = router
            .route_completion(None, &tenant, completion(), MODEL)
            .await;
        assert_eq!(response.status(), StatusCode::OK);
    }
    let body = scrape(metrics_addr).await;
    assert_eq!(
        requests_total(&body, &grpc_upstream),
        Some(2),
        "two gRPC dispatches:\n{body}"
    );
    assert_eq!(
        requests_total(&body, &http_upstream),
        Some(3),
        "the HTTP worker's count is its own:\n{body}"
    );
}
