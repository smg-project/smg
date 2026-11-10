//! A router request under tracing reaches the OTLP collector. The gateway's tracer is
//! process-wide and initialized once, and the exporter's connection lives on the runtime of the
//! test that initialized it; in a binary with other tracing tests the first test by name owns it
//! and a later test's collector hears nothing, so the export assertion lives in a binary of its
//! own, where its test is always the initializer and the count is always asserted.

use std::{sync::Arc, time::Duration};

use axum::{body::Body, extract::Request, http::StatusCode};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use portpicker::pick_unused_port;
use serde_json::json;
use smg::{
    config::{RouterConfig, TraceConfig},
    observability::{logging, otel_trace},
    routers::RouterFactory,
    workflow::Job,
};
use smg_test_harness as common;
use tokio::sync::oneshot;
use tower::ServiceExt;
use tracing::info_span;

use crate::collector::start_collector;

#[path = "otel/collector.rs"]
mod collector;

/// The one test of this binary: it initializes the tracer itself, so the exporter's connection
/// lives on this test's runtime and the spans of its request reach its own collector, where the
/// count is asserted. (In a binary with other tracing tests the first test by name initializes the
/// tracer and the connection dies with that test's runtime, so a later test's collector hears
/// nothing, which is why this assertion used to be skipped when another test had gone first.)
#[expect(clippy::print_stdout, reason = "test diagnostic output")]
#[tokio::test]
async fn a_router_request_under_tracing_reaches_the_collector_as_two_spans() {
    // 1. Start the OTLP collector
    let port = pick_unused_port().expect("Failed to pick unused port");
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let collector = start_collector(port, shutdown_rx)
        .await
        .expect("Failed to start collector");
    let collector_endpoint = format!("0.0.0.0:{port}");
    println!("OTLP Collector started on: {collector_endpoint}");

    // 2. create the mock worker
    let mut mock_worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });

    let worker_url = mock_worker.start().await.unwrap();
    println!("Mock worker started on: {worker_url}");

    // 3. create router config and enable tracing
    let mut router_config = RouterConfig::builder()
        .regular_mode(vec![worker_url.clone()])
        .random_policy()
        .host("0.0.0.0")
        .port(0)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .enable_trace(&collector_endpoint)
        .build_unchecked();
    router_config.health_check.disable_health_check = true;

    // 4. Initialize the OTLP client: this binary's one test is the first and only initializer.
    assert!(
        !otel_trace::is_otel_enabled(),
        "nothing in this binary initializes the tracer before this test"
    );
    otel_trace::otel_tracing_init(true, Some(&collector_endpoint))
        .expect("the tracer initializes against this test's collector");
    println!("OpenTelemetry initialized successfully");

    let trace_config = TraceConfig {
        enable_trace: true,
        otlp_traces_endpoint: collector_endpoint.clone(),
    };
    // `--log-level warn` with the default targets: the spans export without a log directive for them.
    let _log_guard = logging::init_logging(
        logging::LoggingConfig {
            level: tracing::Level::WARN,
            json_format: false,
            log_dir: None,
            colorize: false,
            log_file_name: "test-otel".to_string(),
            log_targets: None,
        },
        Some(trace_config),
    );
    println!("Logging initialized with OTEL layer");

    // 5. The test's own span; its duration is nobody's assertion.
    let _span = info_span!(target: "smg::otel-trace", "test_router_with_tracing");
    drop(_span);

    // 6. create app context and router
    let app_context = common::create_test_context(router_config.clone()).await;

    // 7. initialize worker
    let job_queue = app_context
        .worker_job_queue
        .get()
        .expect("JobQueue should be initialized");

    let job = Job::InitializeWorkersFromConfig {
        router_config: Box::new(router_config.clone()),
    };

    job_queue
        .submit(job)
        .await
        .expect("Failed to submit worker init job");

    // 8. The worker's registration: the init job submits one AddWorker job per
    // configured worker, whose workflow detects, registers and activates it.
    common::wait_for_worker_registration(&app_context, &worker_url, Duration::from_secs(10)).await;
    println!("Workers initialized");

    // 9. create router
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("Failed to create router");

    println!("Router created");

    // 10. create app (middleware::create_logging_layer() will use the already initialized OTEL layer)
    let app =
        common::test_app::create_test_app_with_context(Arc::from(router), app_context.clone());

    println!("App created with logging middleware");

    // 10. send request
    let request_body = json!({
        "model": "mock-model",
        "messages": [
            {"role": "user", "content": "Hello, test OpenTelemetry tracing!"}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(request_body.to_string()))
        .unwrap();

    println!("Sending request to router...");
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK, "Request should succeed");

    println!("Request completed successfully");
    drop(response);

    // 11. Wait for spans to be exported
    match otel_trace::flush_spans_async().await {
        Ok(()) => println!("Spans flushed successfully"),
        Err(e) => println!("Failed to flush spans: {e:?}"),
    }

    // 12. The spans reached the collector: the test's own span and the request's `http_request`.
    let span_count = collector.get_span_count();
    println!("Total spans received by collector: {span_count}");
    assert_eq!(
        span_count, 2,
        "the collector receives exactly the test's span and the request's http_request span; \
         fewer means spans are not exported to the OTLP collector, more means a new span joined \
         the request path and this count needs updating"
    );

    // 13. cleanup
    let _ = shutdown_tx.send(());
    mock_worker.stop().await;

    println!("Cleanup completed");
}
