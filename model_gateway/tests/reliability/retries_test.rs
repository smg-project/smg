//! Retry mechanism integration tests
//!
//! Tests for retry behavior: exponential backoff, max retries, and retry-on-failure scenarios.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{RetryConfig, RouterConfig};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[cfg(test)]
mod retry_tests {
    use super::*;

    /// Test that retries succeed when at least one worker is healthy
    #[tokio::test]
    async fn test_retry_succeeds_with_healthy_fallback() {
        let retry_config = RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 10,
            max_backoff_ms: 100,
            ..Default::default()
        };
        let config = TestRouterConfig::round_robin_with_retry(3300, retry_config);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19200, 1.0), // First worker always fails
                TestWorkerConfig::healthy(19201),    // Second worker always succeeds
            ],
        )
        .await;

        let app = ctx.create_app();

        // Request should succeed via retry to healthy worker
        let payload = json!({
            "text": "Test retry to healthy worker",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request should succeed via retry"
        );

        ctx.shutdown().await;
    }

    /// Test that retries are disabled when configured
    #[tokio::test]
    async fn test_retries_disabled() {
        let mut config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3301)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .disable_retries()
            .build_unchecked();
        config.health_check.disable_health_check = true;

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::flaky(19202, 1.0)], // Always fail
        )
        .await;

        let app = ctx.create_app();

        // With retries disabled, request should fail immediately
        let payload = json!({
            "text": "Test no retries",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "Request should fail without retries"
        );

        ctx.shutdown().await;
    }

    /// A worker that answers every request with its own 500 is not asked
    /// again for the same request: with one worker there is nothing to retry
    /// on, so the answer comes back after one attempt (the retry budget on
    /// transient failures is covered by the router's unit tests).
    #[tokio::test]
    async fn test_max_retries_limit() {
        let retry_config = RetryConfig {
            max_retries: 2,
            initial_backoff_ms: 10,
            max_backoff_ms: 50,
            ..Default::default()
        };
        let config = TestRouterConfig::round_robin_with_retry(3302, retry_config);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::flaky(19203, 1.0)], // Always fail
        )
        .await;

        let app = ctx.create_app();

        // All retries will fail, should return error after exhausting retries
        let payload = json!({
            "text": "Test max retries",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();

        // The worker's own answer is what the client gets.
        assert_eq!(
            resp.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "the failing worker's own 500 is returned, got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    /// Test retry with multiple workers - should eventually find healthy one
    #[tokio::test]
    async fn test_retry_finds_healthy_worker() {
        let retry_config = RetryConfig {
            max_retries: 5,
            initial_backoff_ms: 5,
            max_backoff_ms: 50,
            ..Default::default()
        };
        let config = TestRouterConfig::round_robin_with_retry(3303, retry_config);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19204, 1.0), // Fail
                TestWorkerConfig::flaky(19205, 1.0), // Fail
                TestWorkerConfig::healthy(19206),    // Succeed
            ],
        )
        .await;

        let app = ctx.create_app();

        // With round robin and retries, should eventually hit the healthy worker
        let payload = json!({
            "text": "Test find healthy worker",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Should succeed by retrying until finding healthy worker"
        );

        ctx.shutdown().await;
    }
}
