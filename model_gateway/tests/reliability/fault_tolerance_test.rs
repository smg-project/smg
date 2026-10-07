//! Fault tolerance integration tests
//!
//! Tests for system resilience: worker failures, network issues, and recovery.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{CircuitBreakerConfig, RetryConfig};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{set_request_recorder, RequestRecorder},
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};

#[cfg(test)]
mod fault_tolerance_tests {
    use super::*;

    /// Test that requests are rerouted when a worker fails
    #[tokio::test]
    async fn test_worker_failure_reroute() {
        let config = TestRouterConfig::round_robin_with_reliability(
            4100,
            RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 100,
                ..Default::default()
            },
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
                TestWorkerConfig::flaky(20100, 1.0), // Always fails
                TestWorkerConfig::healthy(20101),    // Always succeeds
            ],
        )
        .await;

        let app = ctx.create_app();

        // Requests should succeed via retry to healthy worker
        for i in 0..10 {
            let payload = json!({
                "text": format!("Fault tolerance test {i}"),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "Request should succeed via reroute to healthy worker"
            );
        }

        ctx.shutdown().await;
    }

    /// Test behavior when all workers are temporarily unavailable
    #[tokio::test]
    async fn test_all_workers_temporarily_failing() {
        let config = TestRouterConfig::round_robin_with_retry(
            4101,
            RetryConfig {
                max_retries: 2,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            },
        );

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(20102, 1.0), // Always fails
                TestWorkerConfig::flaky(20103, 1.0), // Always fails
            ],
        )
        .await;

        let app = ctx.create_app();

        let payload = json!({
            "text": "Test with all failing workers",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Should fail when all workers are failing
        assert!(
            resp.status() == StatusCode::INTERNAL_SERVER_ERROR
                || resp.status() == StatusCode::SERVICE_UNAVAILABLE,
            "Request should fail when all workers are failing, got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    /// Test graceful handling of slow workers
    #[tokio::test]
    #[expect(clippy::disallowed_methods)]
    async fn test_slow_worker_handling() {
        let config = TestRouterConfig::round_robin(4102);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::slow(20104, 500), // Slow worker
                TestWorkerConfig::healthy(20105),   // Fast worker
            ],
        )
        .await;

        let app = ctx.create_app();

        // Send concurrent requests
        let mut handles = Vec::new();
        let success_count = Arc::new(AtomicUsize::new(0));

        for i in 0..10 {
            let app_clone = app.clone();
            let success_clone = Arc::clone(&success_count);

            let handle = tokio::spawn(async move {
                let payload = json!({
                    "text": format!("Slow worker test {i}"),
                    "stream": false
                });

                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();

                let resp = app_clone.oneshot(req).await.unwrap();
                if resp.status() == StatusCode::OK {
                    success_clone.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // All requests should eventually succeed
        assert_eq!(
            success_count.load(Ordering::SeqCst),
            10,
            "All requests should succeed despite slow worker"
        );

        ctx.shutdown().await;
    }

    /// Test circuit breaker prevents cascading failures
    #[tokio::test]
    async fn test_circuit_breaker_prevents_cascade() {
        let config = TestRouterConfig::round_robin_with_reliability(
            4103,
            RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            },
            CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 1,
                timeout_duration_secs: 5,
                window_duration_secs: 10,
            },
        );

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(20106, 1.0), // Failing worker
                TestWorkerConfig::healthy(20107),    // Healthy worker
            ],
        )
        .await;

        let app = ctx.create_app();
        let mut success_count = 0;

        // Send many requests - after CB opens, all should route to healthy worker
        for i in 0..20 {
            let payload = json!({
                "text": format!("Circuit breaker cascade test {i}"),
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

        // Most requests should succeed after CB opens on failing worker
        assert!(
            success_count >= 15,
            "Most requests should succeed after circuit breaker opens, got {success_count} successes"
        );

        ctx.shutdown().await;
    }

    /// Test recovery after worker comes back online (simulated via healthy worker)
    #[tokio::test]
    async fn test_system_stability_under_partial_failure() {
        let config = TestRouterConfig::round_robin_with_reliability(
            4104,
            RetryConfig {
                max_retries: 2,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            },
            CircuitBreakerConfig {
                failure_threshold: 3,
                success_threshold: 1,
                timeout_duration_secs: 2,
                window_duration_secs: 10,
            },
        );

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(20108, 0.5), // 50% failure rate
                TestWorkerConfig::healthy(20109),    // Always succeeds
            ],
        )
        .await;

        let app = ctx.create_app();
        let mut success_count = 0;

        // System should maintain stability with partial failures
        for i in 0..30 {
            let payload = json!({
                "text": format!("Stability test {i}"),
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

        // With retries and one healthy worker, most should succeed
        assert!(
            success_count >= 25,
            "System should maintain stability under partial failure, got {success_count} successes"
        );

        ctx.shutdown().await;
    }

    /// Sticky routing (manual policy) with the circuit breaker and health checks
    /// disabled: nothing ever marks a failing worker unavailable, so retry
    /// attempts must route around the worker that already failed the request
    /// instead of re-selecting it until retries are exhausted.
    #[tokio::test]
    async fn test_sticky_retry_reroutes_around_failed_worker_without_circuit_breaker() {
        let mut config = TestRouterConfig::manual(4104);
        config.retry = RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 10,
            max_backoff_ms: 50,
            ..Default::default()
        };
        config.disable_circuit_breaker = true;

        // Records every request the failing worker receives, including ones it
        // fails, so the test can prove healed keys stop being routed to it.
        let failing_recorder = RequestRecorder::new();
        set_request_recorder(20160, failing_recorder.clone());

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(20160, 1.0), // Always fails, never marked unavailable
                TestWorkerConfig::healthy(20161),
            ],
        )
        .await;

        let app = ctx.create_app();

        // Fresh routing keys: with random assignment some keys land on the
        // failing worker first. Every request must still succeed via a retry
        // on the other worker.
        for i in 0..16 {
            let payload = json!({
                "text": format!("Sticky failover test {i}"),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .header("X-SMG-Routing-Key", format!("sticky-key-{i}"))
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "Request with routing key sticky-key-{i} should succeed via retry on the healthy worker"
            );
        }

        // The failover must also rewrite each key's sticky assignment: repeat
        // requests go straight to the healthy worker, without first burning an
        // attempt (and a retry backoff) on the failed one. The recorder count
        // freezing proves it — the response header alone only shows the final
        // attempt's worker.
        let attempts_on_failed_worker = failing_recorder.bodies().len();
        assert!(
            attempts_on_failed_worker > 0,
            "Setup check: some routing keys should have been assigned to the failing worker first"
        );

        for i in 0..16 {
            let payload = json!({
                "text": format!("Sticky failover repeat {i}"),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .header("X-SMG-Routing-Key", format!("sticky-key-{i}"))
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let routed = resp
                .headers()
                .get("x-smg-routed-worker-id")
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_string();
            assert!(
                routed.contains(":20161"),
                "sticky-key-{i} should be remapped to the healthy worker, got {routed}"
            );
        }

        assert_eq!(
            failing_recorder.bodies().len(),
            attempts_on_failed_worker,
            "Healed routing keys must not send any further attempts to the failed worker"
        );

        ctx.shutdown().await;
    }

    /// X-SMG-Target-Worker pins a request to a worker by index. The pin must
    /// resolve to a stable worker identity for the whole request: per-attempt
    /// selection filtering (retry exclusion of workers that failed the request,
    /// and availability once the worker's circuit breaker opens) used to shift
    /// the index, silently routing pinned traffic to a worker the caller never
    /// asked for. A pinned request fails on its worker, it is never rerouted.
    ///
    /// The breaker's failure_threshold (2) is below the attempt count (3), so
    /// this covers both regimes in one run: exclusion churn within the first
    /// request, and an open breaker for every request after it.
    #[tokio::test]
    async fn test_target_worker_pin_is_never_rerouted() {
        let mut config = TestRouterConfig::consistent_hashing(4105);
        config.retry = RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 10,
            max_backoff_ms: 50,
            ..Default::default()
        };
        config.circuit_breaker = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration_secs: 60,
            window_duration_secs: 60,
        };

        let failing_recorder = RequestRecorder::new();
        set_request_recorder(20162, failing_recorder.clone());
        let healthy_recorder = RequestRecorder::new();
        set_request_recorder(20163, healthy_recorder.clone());

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(20162, 1.0), // Always fails
                TestWorkerConfig::healthy(20163),
            ],
        )
        .await;

        let app = ctx.create_app();

        let pinned_request = |idx: usize, text: String| {
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .header("X-SMG-Target-Worker", idx.to_string())
                .body(Body::from(
                    serde_json::to_string(&json!({"text": text, "stream": false})).unwrap(),
                ))
                .unwrap()
        };

        // Find the failing worker's index in the gateway's selection order by
        // probing both pins; registration order is not deterministic. The
        // failing probe burns through the breaker's failure threshold, so the
        // failing worker's circuit is open for everything below.
        let mut failing_index = None;
        for idx in 0..2 {
            let resp = app
                .clone()
                .oneshot(pinned_request(idx, "pin probe".to_string()))
                .await
                .unwrap();
            if resp.status() != StatusCode::OK {
                failing_index = Some(idx);
            }
        }
        let failing_index = failing_index.expect("one pinned worker must be the failing one");
        let healthy_requests_before = healthy_recorder.bodies().len();
        let failing_requests_before = failing_recorder.bodies().len();

        // Requests pinned to the failing worker must fail there, never be
        // silently served by the healthy worker via a shifted index.
        for i in 0..4 {
            let resp = app
                .clone()
                .oneshot(pinned_request(failing_index, format!("pinned {i}")))
                .await
                .unwrap();
            assert!(
                !resp.status().is_success(),
                "Request pinned to the failing worker must not succeed elsewhere"
            );
        }
        assert_eq!(
            healthy_recorder.bodies().len(),
            healthy_requests_before,
            "Pinned requests must never be rerouted to the healthy worker"
        );
        // Every retry attempt goes to the pinned worker, even with the breaker
        // open (an explicit pin overrides it). Under index shifting the pinned
        // worker got at most one attempt per request before the retry either
        // misrouted (caught above) or went out of bounds.
        let pinned_attempts = failing_recorder.bodies().len() - failing_requests_before;
        assert!(
            pinned_attempts > 4,
            "Retries of a pinned request must stay on the pinned worker \
             (got {pinned_attempts} attempts for 4 requests)"
        );

        ctx.shutdown().await;
    }
}
