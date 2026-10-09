//! The request's trace id reaches the request log lines and the response.
//!
//! Its own test binary: the logging subscriber is installed once per process,
//! and this test needs it writing JSON lines to a directory it can read back.

mod common;

use std::{fs, path::PathBuf, time::Duration};

use axum::{body::Body, extract::Request, http::StatusCode};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use portpicker::pick_unused_port;
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, TraceConfig},
    observability::{logging, otel_trace},
    routers::RouterFactory,
    workflow::Job,
};
use tower::ServiceExt;

/// The trace id the caller chose; the gateway must log and echo this one.
const TRACE_ID: &str = "4bf92f3577b34da6a3ce929d0e0e4736";

#[tokio::test]
async fn request_log_lines_and_the_response_carry_the_trace_id() {
    // Tracing on, exporting towards a port nobody listens on: the spans get
    // their ids all the same, and the exporter's failures are its own.
    let otlp_port = pick_unused_port().expect("free port");
    let otlp_endpoint = format!("127.0.0.1:{otlp_port}");
    otel_trace::otel_tracing_init(true, Some(&otlp_endpoint)).expect("otel init");

    let log_dir = std::env::temp_dir().join(format!("smg-trace-context-{}", std::process::id()));
    fs::create_dir_all(&log_dir).expect("log dir");
    let _log_guard = logging::init_logging(
        logging::LoggingConfig {
            level: tracing::Level::INFO,
            json_format: true,
            log_dir: Some(log_dir.to_string_lossy().into_owned()),
            colorize: false,
            log_file_name: "trace-context-test".to_string(),
            log_targets: Some(vec!["smg".to_string()]),
        },
        Some(TraceConfig {
            enable_trace: true,
            otlp_traces_endpoint: otlp_endpoint,
        }),
    );

    let mut mock_worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let worker_url = mock_worker.start().await.expect("mock worker");

    let mut router_config = RouterConfig::builder()
        .regular_mode(vec![worker_url])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked();
    router_config.health_check.disable_health_check = true;
    let app_context = common::create_test_context(router_config.clone()).await;
    app_context
        .worker_job_queue
        .get()
        .expect("job queue")
        .submit(Job::InitializeWorkersFromConfig {
            router_config: Box::new(router_config),
        })
        .await
        .expect("worker init job");
    tokio::time::sleep(Duration::from_millis(3000)).await;
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("router");
    let app = common::test_app::create_test_app_with_context(router.into(), app_context);

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("traceparent", format!("00-{TRACE_ID}-00f067aa0ba902b7-01"))
        .body(Body::from(
            json!({
                "model": "mock-model",
                "messages": [{"role": "user", "content": "trace me"}],
                "max_tokens": 8
            })
            .to_string(),
        ))
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);

    // The reply carries the trace context it was served under.
    let traceparent = response
        .headers()
        .get("traceparent")
        .expect("the response carries traceparent")
        .to_str()
        .expect("ascii traceparent");
    let parts: Vec<&str> = traceparent.split('-').collect();
    assert_eq!(parts.len(), 4, "traceparent {traceparent}");
    assert_eq!(parts[1], TRACE_ID, "traceparent {traceparent}");
    assert!(
        response.headers().get("tracestate").is_none(),
        "no tracestate was sent, none is echoed"
    );
    drop(response);
    mock_worker.stop().await;

    // The request log lines carry the same id as a span field.
    logging::close_logging();
    let lines = request_log_lines(&log_dir);
    fs::remove_dir_all(&log_dir).expect("remove log dir");
    assert!(
        lines
            .iter()
            .any(|line| line["message"] == "started processing request"),
        "no request start line among {lines:?}"
    );
    assert!(
        lines
            .iter()
            .any(|line| line["message"] == "finished processing request"),
        "no request end line among {lines:?}"
    );
    for line in &lines {
        assert_eq!(line["span"]["trace_id"], TRACE_ID, "log line {line}");
    }
}

/// The JSON lines on the `smg::request` / `smg::response` targets.
#[expect(
    clippy::expect_used,
    reason = "test helper; a missing log file fails the test"
)]
fn request_log_lines(log_dir: &PathBuf) -> Vec<Value> {
    let mut lines = Vec::new();
    for entry in fs::read_dir(log_dir).expect("read log dir") {
        let path = entry.expect("dir entry").path();
        let text = fs::read_to_string(&path).expect("read log file");
        lines.extend(
            text.lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .filter(|line| {
                    line["target"] == "smg::request" || line["target"] == "smg::response"
                }),
        );
    }
    lines
}
