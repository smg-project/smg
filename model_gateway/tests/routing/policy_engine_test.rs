//! Regular placement through the policy engine: under `--policy-engine
//! engine` each preset the engine serves routes every request of the HTTP
//! router, as it does under the legacy default.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{PolicyConfig, PolicyEngineMode, RouterConfig};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

/// The harness's regular configuration under `policy`, decided through the
/// engine.
fn engine_config(port: u16, policy: PolicyConfig) -> RouterConfig {
    let mut config = TestRouterConfig::round_robin(port);
    config.policy = policy;
    config.policy_engine = PolicyEngineMode::Engine;
    config
}

/// `workers` healthy workers behind a gateway deciding `policy` through the
/// engine take twenty generate requests, every one placed by a decision of
/// the engine.
#[expect(clippy::unwrap_used, clippy::expect_used, reason = "test helper")]
async fn every_request_is_placed(port: u16, worker_port: u16, workers: u16, policy: PolicyConfig) {
    let name = policy.name();
    let ctx = AppTestContext::new_with_config(
        engine_config(port, policy),
        TestWorkerConfig::healthy_workers(worker_port, workers),
    )
    .await;
    let app = ctx.create_app();
    let requests = 20;
    let mut placed = 0;
    for i in 0..requests {
        let payload = json!({ "text": format!("Request {i}"), "stream": false });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        if resp.status() == StatusCode::OK {
            placed += 1;
        }
    }
    assert_eq!(placed, requests, "{name}: every request is placed");
    let decider = ctx
        .app_context
        .policy_registry
        .decider()
        .expect("the engine is hosted under the switch");
    assert_eq!(
        decider.decisions(),
        requests as u64,
        "{name}: every request was decided through the engine"
    );
    ctx.shutdown().await;
}

#[tokio::test]
async fn round_robin_routes_through_the_engine() {
    every_request_is_placed(3920, 19920, 3, PolicyConfig::RoundRobin).await;
}

#[tokio::test]
async fn random_routes_through_the_engine() {
    every_request_is_placed(3921, 19924, 2, PolicyConfig::Random).await;
}

#[tokio::test]
async fn passthrough_routes_through_the_engine() {
    every_request_is_placed(3922, 19927, 2, PolicyConfig::Passthrough).await;
}

#[tokio::test]
async fn power_of_two_routes_through_the_engine() {
    every_request_is_placed(
        3923,
        19930,
        2,
        PolicyConfig::PowerOfTwo {
            load_metric: Default::default(),
        },
    )
    .await;
}

#[tokio::test]
async fn least_load_routes_through_the_engine() {
    every_request_is_placed(
        3924,
        19933,
        2,
        PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        },
    )
    .await;
}

#[tokio::test]
async fn cache_aware_routes_through_the_engine() {
    every_request_is_placed(
        3929,
        19948,
        2,
        PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096),
    )
    .await;
}
