//! Prefill/Decode (PD) routing integration tests
//!
//! Tests for prefill-decode disaggregation routing mode.

use std::sync::Arc;

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use openai_protocol::model_card::ModelCard;
use serde_json::json;
use smg::{config::RouterConfig, worker::BasicWorkerBuilder};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{
        set_request_recorder, HealthStatus, MockWorkerConfig, RequestRecorder, WorkerType,
    },
    AppTestContext, TestWorkerConfig,
};

#[cfg(test)]
mod pd_routing_tests {
    use super::*;

    const CANONICAL_MODEL: &str = "GLM-5.2";
    const MODEL_ALIAS: &str = "GLM-5.2-Coding";

    /// Test basic PD mode routing with prefill and decode workers
    #[tokio::test]
    async fn test_pd_mode_basic_routing() {
        let mut config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![
                    ("http://127.0.0.1:19800".to_string(), None),
                    ("http://127.0.0.1:19801".to_string(), None),
                ],
                vec![
                    "http://127.0.0.1:19802".to_string(),
                    "http://127.0.0.1:19803".to_string(),
                ],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3800)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        config.health_check.disable_health_check = true;

        // Note: For PD mode tests, we need to start prefill and decode workers separately
        // The test context will need to handle this specially
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                // Prefill workers
                TestWorkerConfig::prefill(19800),
                TestWorkerConfig::prefill(19801),
                // Decode workers
                TestWorkerConfig::decode(19802),
                TestWorkerConfig::decode(19803),
            ],
        )
        .await;

        let app = ctx.create_app();

        // Send requests and verify they succeed
        for i in 0..10 {
            let payload = json!({
                "text": format!("PD mode request {}", i),
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
                "PD mode request should succeed"
            );
        }

        ctx.shutdown().await;
    }

    /// Test PD mode with round robin policy
    #[tokio::test]
    async fn test_pd_mode_round_robin() {
        let mut config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19810".to_string(), None)],
                vec![
                    "http://127.0.0.1:19811".to_string(),
                    "http://127.0.0.1:19812".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3801)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        config.health_check.disable_health_check = true;

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19810),
                TestWorkerConfig::decode(19811),
                TestWorkerConfig::decode(19812),
            ],
        )
        .await;

        let app = ctx.create_app();
        let mut success_count = 0;

        for i in 0..20 {
            let payload = json!({
                "text": format!("PD round robin {}", i),
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

        assert_eq!(
            success_count, 20,
            "All requests should succeed in PD mode with round robin"
        );

        ctx.shutdown().await;
    }

    /// A request addressed to a model alias must reach both PD workers under
    /// the canonical model ID. The workers were registered under the canonical
    /// ID and have never heard of the alias, so forwarding the alias would
    /// hand them a model they cannot serve.
    #[tokio::test]
    async fn test_pd_model_alias() {
        let prefill_url = "http://127.0.0.1:19840".to_string();
        let decode_url = "http://127.0.0.1:19841".to_string();

        let prefill_recorder = RequestRecorder::new();
        let decode_recorder = RequestRecorder::new();
        set_request_recorder(19840, Arc::clone(&prefill_recorder));
        set_request_recorder(19841, Arc::clone(&decode_recorder));

        let mut config = RouterConfig::builder()
            .prefill_decode_mode(vec![(prefill_url.clone(), None)], vec![decode_url.clone()])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3804)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        config.health_check.disable_health_check = true;

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19840),
                TestWorkerConfig::decode(19841),
            ],
        )
        .await;
        let app = ctx.create_app();

        // The workers bound ephemeral ports; the registry knows them by those.
        let registry = &ctx.app_context.worker_registry;
        for url in [ctx.worker_url_for(19840), ctx.worker_url_for(19841)] {
            let worker_id = registry.get_id_by_url(url).unwrap();
            let worker = registry.get(&worker_id).unwrap();
            let mut spec = worker.metadata().spec.as_ref().clone();
            spec.models = vec![ModelCard::new(CANONICAL_MODEL).with_alias(MODEL_ALIAS)].into();
            let replacement = BasicWorkerBuilder::from_spec(spec)
                .health_config(worker.metadata().health_config.clone())
                .health_endpoint(&worker.metadata().health_endpoint)
                .build();
            assert!(registry.replace(&worker_id, Arc::new(replacement)));
        }

        let alias_request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": MODEL_ALIAS,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": false
                })
                .to_string(),
            ))
            .unwrap();
        let alias_response = app.clone().oneshot(alias_request).await.unwrap();
        assert_eq!(alias_response.status(), StatusCode::OK);
        let alias_body = to_bytes(alias_response.into_body(), usize::MAX)
            .await
            .unwrap();
        let alias_json: serde_json::Value = serde_json::from_slice(&alias_body).unwrap();
        assert_eq!(alias_json["model"], "mock-model");

        // The part the response body cannot show: what the gateway forwarded.
        for (leg, recorder) in [("prefill", &prefill_recorder), ("decode", &decode_recorder)] {
            let forwarded = recorder.only_body();
            assert_eq!(
                forwarded["model"], CANONICAL_MODEL,
                "{leg} worker must receive the canonical model ID, got {}",
                forwarded["model"]
            );
        }

        ctx.shutdown().await;
    }

    /// A non-streaming PD request must emit the SMG-only PD metrics, including
    /// the honest `smg_pd_ttft_seconds`. Runs on a current-thread runtime so the
    /// thread-local Prometheus recorder captures emissions from the request path.
    #[test]
    fn test_pd_metrics_emitted_on_request() {
        use metrics_exporter_prometheus::PrometheusBuilder;

        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();

        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let mut config = RouterConfig::builder()
                    .prefill_decode_mode(
                        vec![("http://127.0.0.1:19830".to_string(), None)],
                        vec!["http://127.0.0.1:19831".to_string()],
                    )
                    .round_robin_policy()
                    .host("127.0.0.1")
                    .port(3803)
                    .max_payload_size(256 * 1024 * 1024)
                    .request_timeout_secs(600)
                    .worker_startup_timeout_secs(5)
                    .worker_startup_check_interval_secs(1)
                    .max_concurrent_requests(64)
                    .queue_timeout_secs(60)
                    .build_unchecked();
                config.health_check.disable_health_check = true;

                let ctx = AppTestContext::new_with_config(
                    config,
                    vec![
                        TestWorkerConfig::prefill(19830),
                        TestWorkerConfig::decode(19831),
                    ],
                )
                .await;

                let app = ctx.create_app();
                let payload = json!({ "text": "PD metrics request", "stream": false });
                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();

                let resp = app.oneshot(req).await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK, "PD request should succeed");

                ctx.shutdown().await;
            });
        });

        let rendered = handle.render();
        assert!(
            rendered.contains("smg_pd_prefill_duration_seconds_count"),
            "smg_pd_prefill_duration_seconds not emitted; rendered:\n{rendered}"
        );
        assert!(
            rendered.contains("smg_pd_ttft_seconds_count"),
            "smg_pd_ttft_seconds not emitted; rendered:\n{rendered}"
        );
    }

    /// Test PD mode handles worker failures gracefully
    #[tokio::test]
    async fn test_pd_mode_with_failing_decode_worker() {
        use smg::config::RetryConfig;

        let mut config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19820".to_string(), None)],
                vec![
                    "http://127.0.0.1:19821".to_string(),
                    "http://127.0.0.1:19822".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3802)
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
            .build_unchecked();
        config.health_check.disable_health_check = true;

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19820),
                MockWorkerConfig {
                    port: 19821,
                    worker_type: WorkerType::Decode,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: 0,
                    fail_rate: 1.0, // Failing decode worker
                },
                TestWorkerConfig::decode(19822), // Healthy decode worker
            ],
        )
        .await;

        let app = ctx.create_app();

        // Request should succeed via retry to healthy decode worker
        let payload = json!({
            "text": "Test with failing decode worker",
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
            "Request should succeed via retry to healthy decode worker"
        );

        ctx.shutdown().await;
    }
}
