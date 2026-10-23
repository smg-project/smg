use std::time::Duration;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;

use crate::common;

#[tokio::test]
#[expect(clippy::disallowed_methods)]
async fn test_multiple_concurrent_requests_tracking() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 19002,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 50,
        fail_rate: 0.0,
    }])
    .await;

    let tracker = ctx.app_context.inflight_tracker.clone();

    let mut handles = vec![];
    for i in 0..5 {
        let app = ctx.create_app();
        handles.push(tokio::spawn(async move {
            let payload = json!({
                "text": format!("Request {i}"),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            app.oneshot(req).await.unwrap()
        }));
    }

    for handle in handles {
        let resp = handle.await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    assert!(tracker.is_empty());

    ctx.shutdown().await;
}

#[tokio::test]
#[expect(clippy::disallowed_methods)]
async fn test_inflight_request_appears_in_bucket() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 19004,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 2000,
        fail_rate: 0.0,
    }])
    .await;

    let tracker = &ctx.app_context.inflight_tracker;
    assert!(tracker.is_empty(), "Tracker should start empty");

    let app = ctx.create_app();
    let payload = json!({
        "text": "Long running request",
        "stream": false
    });

    let req = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap();

    let tracker_clone = ctx.app_context.inflight_tracker.clone();
    let response_future = tokio::spawn(async move { app.oneshot(req).await });

    tokio::time::sleep(Duration::from_millis(500)).await;

    let inflight_count = tracker_clone.len();
    assert!(
        inflight_count > 0,
        "Should have at least one in-flight request, got {inflight_count}"
    );

    let buckets = tracker_clone.compute_bucket_counts();
    assert!(buckets[0] > 0, "first bucket (<=30s) should have requests");

    let resp = response_future.await.unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Snapshot while the response body is still held. A drain observer must
    // retain this request even though its handler has already returned.
    let before = inflight_snapshot(&ctx).await;
    let repeated = inflight_snapshot(&ctx).await;
    // The snapshot requests themselves finish between calls; the held
    // response is the request that must appear in both observations.
    let previous: Vec<_> = before["request_ids"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|id| repeated["request_ids"].as_array().unwrap().contains(id))
        .collect();
    assert!(!previous.is_empty());

    // The in-flight guard rides the response BODY (streaming responses stay
    // counted while they generate), so the request deregisters when the body
    // is consumed to completion, not when the handler returns.
    let _ = resp.into_body().collect().await.unwrap();
    assert!(
        tracker_clone.is_empty(),
        "Request should be deregistered after the body completes"
    );

    // Later requests must not keep the original cohort from draining.
    let newer = tracker.track();
    let after = inflight_snapshot(&ctx).await;
    assert_eq!(before["instance_id"], after["instance_id"]);
    assert!(after["instance_id"].as_str().is_some_and(|s| !s.is_empty()));
    let current = after["request_ids"].as_array().unwrap();
    assert!(!current.is_empty());
    assert!(previous.iter().all(|id| !current.contains(id)));
    drop(newer);

    ctx.shutdown().await;
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper: malformed responses fail the test"
)]
async fn inflight_snapshot(ctx: &AppTestContext) -> serde_json::Value {
    let response = ctx
        .create_app()
        .oneshot(
            Request::builder()
                .uri("/inflight")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
}

#[tokio::test]
async fn test_failed_request_still_deregisters() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 19003,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 1.0,
    }])
    .await;

    let tracker = &ctx.app_context.inflight_tracker;
    assert!(tracker.is_empty());

    let app = ctx.create_app();

    let payload = json!({
        "text": "This should fail",
        "stream": false
    });

    let req = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

    // Dropping the response without reading its body (client-disconnect
    // shape) must also release the body-held guard.
    drop(resp);
    assert!(tracker.is_empty());

    ctx.shutdown().await;
}
