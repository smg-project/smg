//! Completed short requests remain visible as worker-scoped dispatch totals.

mod common;

use std::sync::{Arc, OnceLock};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use common::{AppTestContext, TestRouterConfig, TestWorkerConfig};
use smg::{
    config::RouterConfig,
    observability::metrics::{start_prometheus, MetricsHandle, PrometheusConfig},
    routers::external::GatewayWorker,
    worker::{BasicWorkerBuilder, ModelCard, WorkerRegistry},
};
use tower::ServiceExt;

fn handle() -> &'static MetricsHandle {
    static HANDLE: OnceLock<MetricsHandle> = OnceLock::new();
    HANDLE.get_or_init(|| start_prometheus(PrometheusConfig::default()))
}

fn requests(url: &str) -> Option<u64> {
    handle()
        .render()
        .lines()
        .find(|line| {
            line.starts_with("smg_worker_requests_total{")
                && line.contains(&format!("worker=\"{url}\""))
        })
        .and_then(|line| {
            line.rsplit_once(' ')
                .and_then(|(_, value)| value.parse().ok())
        })
}

#[expect(
    clippy::unwrap_used,
    reason = "test request construction and body consumption must succeed"
)]
async fn generate(app: &axum::Router) {
    let request = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"text":"short request","stream":false}"#))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    to_bytes(response.into_body(), usize::MAX).await.unwrap();
}

#[tokio::test]
async fn short_http_requests_accumulate_per_worker_after_load_returns_to_zero() {
    handle();
    let ctx = AppTestContext::new_with_config(
        TestRouterConfig::round_robin(0),
        TestWorkerConfig::healthy_workers(19101, 2),
    )
    .await;
    let app = ctx.create_app();
    for _ in 0..6 {
        generate(&app).await;
    }
    for url in &ctx.worker_urls {
        assert_eq!(
            requests(url),
            Some(3),
            "completed requests were not retained for {url}"
        );
        let worker = ctx.app_context.worker_registry.get_by_url(url).unwrap();
        assert_eq!(worker.load(), 0);
        assert!(handle()
            .render()
            .lines()
            .any(|line| line.starts_with("smg_worker_requests_total{")
                && line.contains(&format!("worker=\"{url}\""))
                && line.contains(&format!("model=\"{}\"", worker.model_id()))));
    }
    let invalid = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from("invalid JSON"))
        .unwrap();
    assert_eq!(
        app.oneshot(invalid).await.unwrap().status(),
        StatusCode::BAD_REQUEST
    );
    for url in &ctx.worker_urls {
        assert_eq!(requests(url), Some(3));
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn pd_requests_count_each_actual_upstream_leg() {
    handle();
    let mut config = RouterConfig::builder()
        .prefill_decode_mode(
            vec![("http://127.0.0.1:19111".to_string(), None)],
            vec!["http://127.0.0.1:19112".to_string()],
        )
        .round_robin_policy()
        .build_unchecked();
    config.health_check.disable_health_check = true;
    let ctx = AppTestContext::new_with_config(
        config,
        vec![
            TestWorkerConfig::prefill(19111),
            TestWorkerConfig::decode(19112),
        ],
    )
    .await;
    let app = ctx.create_app();
    for _ in 0..2 {
        generate(&app).await;
    }
    for url in &ctx.worker_urls {
        assert_eq!(requests(url), Some(2), "missing PD leg {url}");
    }
    ctx.shutdown().await;
}

#[test]
fn gateway_external_hook_uses_registered_identity_and_retires_with_worker() {
    handle();
    let registry = WorkerRegistry::new();
    let url = "http://request-counter-external";
    let worker = Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new("registered-model"))
            .build(),
    );
    let id = registry.register(worker.clone()).unwrap();
    assert_eq!(requests(url), Some(0));
    let rendered = handle().render();
    assert!(rendered.contains("# HELP smg_worker_requests_total "));
    assert!(rendered.contains("# TYPE smg_worker_requests_total counter\n"));
    let external = GatewayWorker::handle(worker);
    external.record_request();
    external.record_request();
    assert_eq!(requests(url), Some(2));
    assert!(registry.replace(
        &id,
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("registered-model"))
                .build()
        )
    ));
    assert_eq!(requests(url), Some(2), "replacement reset live counter");
    registry.remove(&id).unwrap();
    external.record_request();
    assert_eq!(
        requests(url),
        None,
        "late retired-worker dispatch resurrected its series"
    );
    let id = registry
        .register(Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("registered-model"))
                .build(),
        ))
        .unwrap();
    assert_eq!(requests(url), Some(0));
    registry.remove(&id).unwrap();
}

#[tokio::test]
async fn each_outer_http_retry_counts_even_when_every_upstream_response_fails() {
    handle();
    let config = TestRouterConfig::round_robin_with_retry(
        0,
        smg::config::RetryConfig {
            // RetryConfig caps total attempts, including the initial dispatch.
            max_retries: 3,
            initial_backoff_ms: 1,
            max_backoff_ms: 2,
            ..Default::default()
        },
    );
    let ctx =
        AppTestContext::new_with_config(config, vec![TestWorkerConfig::flaky(19121, 1.0)]).await;
    let request = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"text":"retry","stream":false}"#))
        .unwrap();
    let response = ctx.create_app().oneshot(request).await.unwrap();
    assert!(response.status().is_server_error());
    assert_eq!(requests(&ctx.worker_urls[0]), Some(3));
    ctx.shutdown().await;
}
