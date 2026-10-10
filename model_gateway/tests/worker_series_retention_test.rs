//! A removed worker's series leave `/metrics`.
//!
//! Drives the real Prometheus exporter, the metrics HTTP server and the
//! worker registry end to end: a worker's series are scraped while it is
//! registered, are gone from the next scrape after its removal, stay gone
//! when a write for the address arrives late, and come back when the address
//! is registered again. With pod-based discovery every replaced engine pod
//! is a new address, so without this the scrape grew by one worker's set of
//! series per replacement.
//!
//! This file is its own test binary (process), so installing the global
//! Prometheus recorder here is safe and does not collide with other suites.

#![expect(clippy::expect_used, reason = "test failures should panic")]

use std::{net::SocketAddr, sync::Arc};

use smg::{
    observability::{
        metrics::{start_prometheus, Metrics, PrometheusConfig},
        metrics_server::start_metrics_server,
    },
    worker::{BasicWorkerBuilder, Worker, WorkerRegistry, WorkerType},
};

const URL: &str = "grpc://[fd00::1]:50051";

async fn scrape(addr: SocketAddr) -> String {
    reqwest::get(format!("http://{addr}/metrics"))
        .await
        .expect("metrics endpoint reachable")
        .text()
        .await
        .expect("metrics body")
}

/// The sample lines of `body` that carry the worker's label.
fn worker_lines(body: &str) -> Vec<&str> {
    let label = format!("worker=\"{URL}\"");
    body.lines().filter(|line| line.contains(&label)).collect()
}

fn has_family(lines: &[&str], prefix: &str) -> bool {
    lines.iter().any(|line| line.starts_with(prefix))
}

fn worker() -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(URL)
            .worker_type(WorkerType::Regular)
            .build(),
    )
}

#[tokio::test]
async fn removed_worker_series_leave_the_scrape() {
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    });
    let (addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");
    let registry = WorkerRegistry::new();

    let worker_id = registry.register(worker()).expect("a fresh URL registers");
    // Activity on the worker: a breaker transition, an applied KV event
    // batch (histogram) and its lag (summary).
    Metrics::record_worker_cb_transition(URL, "closed", "open");
    Metrics::record_kv_event_apply(URL, 0.000_01);
    Metrics::record_kv_event_lag(URL, 0.25);

    let body = scrape(addr).await;
    let lines = worker_lines(&body);
    for prefix in [
        "smg_worker_health{",
        "smg_worker_http2{",
        "smg_worker_cb_state{",
        "smg_worker_cb_transitions_total{",
        "smg_kv_event_apply_seconds_bucket{",
        "smg_kv_event_lag_seconds_count{",
    ] {
        assert!(
            has_family(&lines, prefix),
            "{prefix} missing for the registered worker:\n{body}"
        );
    }
    assert_eq!(
        body.matches("# TYPE smg_worker_health ").count(),
        1,
        "one family header per metric name:\n{body}"
    );

    assert!(registry.remove(&worker_id).is_some());
    // Writes that arrive after the removal: a health check that raced it,
    // the outcome of a request that was in flight.
    Metrics::set_worker_health(URL, true);
    Metrics::record_worker_cb_outcome(URL, "failure");
    let body = scrape(addr).await;
    assert!(
        worker_lines(&body).is_empty(),
        "series of the removed worker still in the scrape:\n{body}"
    );

    // The same address registered again (a pod IP reused) records again,
    // from fresh series.
    registry
        .register(worker())
        .expect("the retired URL registers again");
    let body = scrape(addr).await;
    let lines = worker_lines(&body);
    assert!(
        has_family(&lines, "smg_worker_health{") && has_family(&lines, "smg_worker_cb_state{"),
        "series missing after re-registration:\n{body}"
    );
    assert!(
        !has_family(&lines, "smg_worker_cb_transitions_total{"),
        "the removed registration's counter came back:\n{body}"
    );
}
