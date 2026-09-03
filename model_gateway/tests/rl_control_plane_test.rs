//! Gateway-level tests for the RL control plane (`/v1/rl/*`).

mod common;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{
    mock_worker::{HealthStatus, MockWorkerConfig, RequestRecorder, WorkerType},
    AppTestContext, TestRouterConfig,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

fn mock(port: u16, fail_rate: f32) -> MockWorkerConfig {
    MockWorkerConfig {
        port,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate,
    }
}

async fn ctx(enabled: bool, workers: Vec<MockWorkerConfig>) -> AppTestContext {
    let mut config = TestRouterConfig::round_robin(0);
    config.rl.enabled = enabled;
    config.rl.control_timeout_secs = 5;
    AppTestContext::new_with_config(config, workers).await
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn json_of(resp: axum::response::Response) -> Value {
    serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes())
        .unwrap_or(Value::Null)
}

#[tokio::test]
async fn flag_off_leaves_v1_rl_unmounted() {
    let ctx = ctx(false, vec![mock(18901, 0.0)]).await;
    let app = ctx.create_app();
    for (method, uri) in [
        ("GET", "/v1/rl/workers"),
        ("POST", "/v1/rl/engine/flush_cache?selector=engine%3Dsglang"),
    ] {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND, "{method} {uri}");
        assert!(resp
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .is_empty());
    }
    assert!(ctx.app_context.rl.is_none());
    ctx.shutdown().await;
}

#[tokio::test]
async fn discovery_lists_mock_workers() {
    let ctx = ctx(true, vec![mock(18902, 0.0), mock(18903, 0.0)]).await;
    let app = ctx.create_app();
    let resp = app
        .oneshot(Request::get("/v1/rl/workers").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_of(resp).await;
    assert_eq!(body["total"], 2);
    for w in body["workers"].as_array().unwrap() {
        assert_eq!(w["engine"], "sglang");
        assert_eq!(w["tp_size"], 1);
        assert_eq!(w["health"], "ready");
        assert_eq!(w["connection_mode"], "http");
        assert_eq!(w["capabilities"]["source"], "static");
        assert!(w["id"].as_str().unwrap().len() >= 32);
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn proxy_forwards_body_verbatim() {
    let recorder = RequestRecorder::new();
    common::mock_worker::set_request_recorder(18904, recorder.clone());
    let ctx = ctx(true, vec![mock(18904, 0.0)]).await;
    let app = ctx.create_app();
    let id = {
        let resp = app
            .clone()
            .oneshot(Request::get("/v1/rl/workers").body(Body::empty()).unwrap())
            .await
            .unwrap();
        json_of(resp).await["workers"][0]["id"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let payload = json!({"model_path": "/models/x", "weight_version": "42", "flush_cache": true});
    let resp = app
        .oneshot(
            Request::post(format!(
                "/v1/rl/workers/{id}/engine/update_weights_from_disk"
            ))
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_of(resp).await;
    assert_eq!(body["worker_id"], id);
    assert_eq!(body["body"]["success"], true);
    assert_eq!(recorder.only_body(), payload);
    ctx.shutdown().await;
}

#[tokio::test]
async fn fanout_hits_every_worker_and_reports_failures_without_touching_breakers() {
    let mut ctx = ctx(true, vec![mock(18905, 0.0), mock(18906, 0.0)]).await;
    // Take the second engine down *after* registration: a mock with
    // `fail_rate: 1.0` would also fail its registration probes, so that
    // worker would never reach the registry at all.
    ctx.workers[1].stop().await;
    let app = ctx.create_app();
    let resp = app
        .oneshot(
            Request::post("/v1/rl/engine/pause_generation?selector=engine%3Dsglang")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"mode":"abort"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::MULTI_STATUS);
    let body = json_of(resp).await;
    assert_eq!(body["total"], 2);
    assert_eq!(body["succeeded"], 1);
    assert_eq!(body["failed"].as_array().unwrap().len(), 1);
    assert!(body["failed"][0]["url"].as_str().unwrap().contains("18906"));
    assert_eq!(body["failed"][0]["error"], "upstream_unreachable");

    // The control plane owns its own client: no data-plane breaker trips and
    // no load counter moves, however the engine answered.
    for worker in ctx.app_context.worker_registry.get_all() {
        assert!(worker.circuit_breaker_can_execute(), "{}", worker.url());
        assert_eq!(worker.load(), 0, "{}", worker.url());
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn control_plane_auth_guards_v1_rl() {
    let mut config = TestRouterConfig::round_robin(0);
    config.rl.enabled = true;
    config.api_key = Some("admin-secret".to_string());
    let ctx = AppTestContext::new_with_config(config, vec![mock(18907, 0.0)]).await;
    let app = ctx.create_app();

    let resp = app
        .clone()
        .oneshot(Request::get("/v1/rl/workers").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let resp = app
        .oneshot(
            Request::get("/v1/rl/workers")
                .header("authorization", "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    ctx.shutdown().await;
}
