//! Circuit breaker integration tests
//!
//! Tests for circuit breaker behavior: opening, half-open state, recovery, and isolation.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{CircuitBreakerConfig, RetryConfig, RouterConfig};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[cfg(test)]
mod circuit_breaker_tests {
    use super::*;

    /// One worker that fails every request, a breaker that trips at three failures and stays
    /// open longer than the test runs, and one attempt per request, so that the sequence a
    /// client sees is the breaker's own: the first three answers are the worker's 500, every
    /// one after them the open breaker's 503, and the worker is never asked again. With the
    /// default retries every request would retry its 500 and then the 503, and end as 503, so
    /// the 500 the breaker counts would never reach the client.
    #[tokio::test]
    async fn test_circuit_breaker_opens_after_failures() {
        const THRESHOLD: u32 = 3;
        const REQUESTS: usize = 10;
        let config = TestRouterConfig::round_robin_with_reliability(
            3200,
            RetryConfig {
                max_retries: 1,
                initial_backoff_ms: 0,
                max_backoff_ms: 0,
                backoff_multiplier: 1.0,
                jitter_factor: 0.0,
            },
            CircuitBreakerConfig {
                failure_threshold: THRESHOLD,
                success_threshold: 2,
                timeout_duration_secs: 60,
                window_duration_secs: 60,
            },
        );

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::flaky(19100, 1.0)], // Always fail
        )
        .await;

        let app = ctx.create_app();

        let mut statuses = Vec::with_capacity(REQUESTS);
        for _ in 0..REQUESTS {
            let payload = json!({
                "text": "Test circuit breaker",
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            statuses.push(app.clone().oneshot(req).await.unwrap().status());
        }

        let (before_the_trip, after_the_trip) = statuses.split_at(THRESHOLD as usize);
        assert!(
            before_the_trip
                .iter()
                .all(|status| *status == StatusCode::INTERNAL_SERVER_ERROR),
            "the first {THRESHOLD} answers are the worker's own failures: {statuses:?}"
        );
        assert!(
            after_the_trip
                .iter()
                .all(|status| *status == StatusCode::SERVICE_UNAVAILABLE),
            "after {THRESHOLD} failures the breaker is open and every answer is its 503: {statuses:?}"
        );

        ctx.shutdown().await;
    }

    /// Test circuit breaker with disabled flag
    #[tokio::test]
    async fn test_circuit_breaker_disabled() {
        let mut config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3201)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .disable_circuit_breaker()
            .disable_retries()
            .build_unchecked();
        config.health_check.disable_health_check = true;

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::flaky(19101, 1.0)], // Always fail
        )
        .await;

        let app = ctx.create_app();

        // With circuit breaker disabled, should always see 500 (never 503)
        for _ in 0..5 {
            let payload = json!({
                "text": "Test disabled CB",
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            // With CB disabled, we expect 500 errors (not 503 from CB)
            assert_eq!(
                resp.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "Should get 500 with CB disabled, not CB-related 503"
            );
        }
        ctx.shutdown().await;
    }

    /// Test circuit breaker per-worker isolation
    #[tokio::test]
    async fn test_circuit_breaker_per_worker_isolation() {
        let config = TestRouterConfig::round_robin_with_circuit_breaker(
            3202,
            CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 1,
                timeout_duration_secs: 2,
                window_duration_secs: 10,
            },
        );

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19102, 1.0), // Always fail
                TestWorkerConfig::healthy(19103),    // Always succeed
            ],
        )
        .await;

        let app = ctx.create_app();

        // Send enough requests to trigger CB on the failing worker
        // The healthy worker should continue to serve requests
        let mut success_count = 0;
        for _ in 0..20 {
            let payload = json!({
                "text": "Test isolation",
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        // After CB opens on failing worker, healthy worker should handle all requests
        // We should see at least some successes
        assert!(
            success_count > 0,
            "Should have some successful requests from healthy worker"
        );

        ctx.shutdown().await;
    }

    /// Test circuit breaker with retries enabled
    #[tokio::test]
    async fn test_circuit_breaker_with_retries() {
        let mut config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3203)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            })
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 1,
                timeout_duration_secs: 2,
                window_duration_secs: 10,
            })
            .build_unchecked();
        config.health_check.disable_health_check = true;

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19104, 1.0), // Always fail
                TestWorkerConfig::healthy(19105),    // Always succeed
            ],
        )
        .await;

        let app = ctx.create_app();

        // With retries, requests should succeed by retrying on healthy worker
        let payload = json!({
            "text": "Test with retries",
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
            "Request should succeed via retry on healthy worker"
        );

        ctx.shutdown().await;
    }
}
