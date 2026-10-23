//! Integration tests for the priority scheduler wiring (M6).
//!
//! These are regression/wiring guards: they boot the real `build_app` via the
//! test harness with `--priority-scheduler-enabled` on and off and assert the
//! gateway still serves requests through the admission middleware, the
//! permit on the edge's body wrapper, and the handler `PreemptionGuard` — and
//! that a scheduler that can't start (reservations exceed capacity) degrades
//! to legacy admission rather than failing requests.
//!
//! The scheduler's behavioral logic (admit / queue / preempt / starvation /
//! capacity) is covered deterministically by the engine unit tests in
//! `middleware::scheduler::*`; driving those end-to-end here would depend on
//! `WorkerCapacity` derivation and request timing, which would be flaky.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};
use serde_json::json;
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common;

/// Build a test config matching `AppTestContext::new`'s defaults, with the
/// priority-scheduler flag and concurrency cap overridden.
fn scheduler_config(enabled: bool, max_concurrent_requests: i32) -> RouterConfig {
    let mut config = RouterConfig::builder()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(3002)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(max_concurrent_requests)
        .queue_timeout_secs(60)
        .priority_scheduler_enabled(enabled)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    config
}

fn mock(port: u16) -> MockWorkerConfig {
    MockWorkerConfig {
        port,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn generate_status(app: axum::Router) -> StatusCode {
    let payload = json!({ "text": "hello", "stream": false });
    let req = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();
    app.oneshot(req).await.unwrap().status()
}

/// Scheduler enabled with capacity comfortably above the default reservation
/// sum (160): the priority admission path serves a normal request.
#[tokio::test]
async fn scheduler_enabled_serves_generate() {
    let ctx = AppTestContext::new_with_config(scheduler_config(true, 512), vec![mock(19090)]).await;
    assert_eq!(generate_status(ctx.create_app()).await, StatusCode::OK);
    assert!(
        ctx.app_context.inflight_tracker.is_empty(),
        "slot/permit must be released after the response completes"
    );
    ctx.shutdown().await;
}

/// A request carrying the priority header still serves cleanly (header parse +
/// tenant clamp path does not break admission).
#[tokio::test]
async fn scheduler_enabled_honors_priority_header() {
    let ctx = AppTestContext::new_with_config(scheduler_config(true, 512), vec![mock(19091)]).await;
    let payload = json!({ "text": "hi", "stream": false });
    let req = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .header("x-smg-priority", "interactive")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap();
    let resp = ctx.create_app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    ctx.shutdown().await;
}

/// Legacy mode (flag off, the default): the concurrency-limit middleware path
/// serves requests exactly as before.
#[tokio::test]
async fn scheduler_disabled_legacy_serves_generate() {
    let ctx = AppTestContext::new_with_config(scheduler_config(false, 64), vec![mock(19092)]).await;
    assert_eq!(generate_status(ctx.create_app()).await, StatusCode::OK);
    ctx.shutdown().await;
}

/// Scheduler enabled but the capacity fallback is below the default
/// reservation sum (160): `AdmissionMode::from_config` fails to build the
/// scheduler and degrades to legacy admission. The request must still serve —
/// a misconfigured optional feature never takes the data plane down.
#[tokio::test]
async fn scheduler_enabled_low_capacity_falls_back_and_serves() {
    let ctx = AppTestContext::new_with_config(scheduler_config(true, 8), vec![mock(19093)]).await;
    assert_eq!(generate_status(ctx.create_app()).await, StatusCode::OK);
    ctx.shutdown().await;
}
