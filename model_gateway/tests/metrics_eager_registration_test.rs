//! The protection, retry and breaker-transition families are on the first
//! scrape, at zero, before any request, retry or breaker trip.
//!
//! Drives the recorder `start_prometheus` installs and the same baseline call
//! the gateway makes right after it, and reads the families the way a scrape
//! does, through the metrics HTTP server (the per-worker series render there,
//! not in the exporter's handle). This file is its own test binary (process),
//! so installing the global Prometheus recorder here does not collide with
//! other suites.

#![expect(clippy::expect_used, reason = "test failures should panic")]

use std::net::SocketAddr;

use smg::{
    observability::{
        metrics::{init_startup_series, start_prometheus, PrometheusConfig},
        metrics_server::start_metrics_server,
    },
    worker::{CircuitBreaker, CircuitBreakerConfig},
};

async fn scrape(addr: SocketAddr) -> String {
    reqwest::get(format!("http://{addr}/metrics"))
        .await
        .expect("metrics endpoint reachable")
        .text()
        .await
        .expect("metrics body")
}

/// The rendered sample of `family` whose label set contains every pair in
/// `labels`, as `(line, value)`.
fn sample<'a>(body: &'a str, family: &str, labels: &[(&str, &str)]) -> Option<(&'a str, f64)> {
    let prefix = format!("{family}{{");
    let line = body.lines().find(|line| {
        line.starts_with(&prefix)
            && labels
                .iter()
                .all(|(key, value)| line.contains(&format!("{key}=\"{value}\"")))
    })?;
    let value = line.rsplit(' ').next()?.parse().ok()?;
    Some((line, value))
}

#[tokio::test]
async fn protection_retry_and_breaker_families_start_at_zero() {
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    });
    let (addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");
    init_startup_series();

    let body = scrape(addr).await;
    for (family, labels) in [
        (
            "smg_worker_overload_shed_total",
            vec![("stage", "selection")],
        ),
        (
            "smg_worker_overload_shed_total",
            vec![("stage", "dispatch")],
        ),
        (
            "smg_worker_overload_shed_total",
            vec![("stage", "pd_admission")],
        ),
        (
            "smg_worker_overload_fallback_total",
            vec![("stage", "selection")],
        ),
        (
            "smg_worker_liveness_fallback_total",
            vec![("stage", "selection")],
        ),
        (
            "smg_worker_retries_total",
            vec![("worker_type", "regular"), ("endpoint", "chat")],
        ),
        (
            "smg_worker_retries_total",
            vec![("worker_type", "decode"), ("endpoint", "generate")],
        ),
        (
            "smg_worker_retries_exhausted_total",
            vec![("worker_type", "prefill"), ("endpoint", "responses")],
        ),
        (
            "smg_worker_retry_backoff_seconds_count",
            vec![("attempt", "1")],
        ),
    ] {
        let (line, value) = sample(&body, family, &labels)
            .unwrap_or_else(|| panic!("no {family} sample with {labels:?}:\n{body}"));
        assert_eq!(value, 0.0, "{line}");
    }

    // A breaker trips long after start-up; its transition counters still
    // exist from the moment its worker is built.
    let worker = "grpc://worker-a:50051";
    let _breaker =
        CircuitBreaker::with_config_and_label(CircuitBreakerConfig::default(), worker.to_string());
    let body = scrape(addr).await;
    for (from, to) in [
        ("closed", "open"),
        ("open", "half_open"),
        ("open", "closed"),
        ("half_open", "closed"),
        ("half_open", "open"),
    ] {
        let labels = [("worker", worker), ("from", from), ("to", to)];
        let (line, value) = sample(&body, "smg_worker_cb_transitions_total", &labels)
            .unwrap_or_else(|| panic!("no transition sample with {labels:?}:\n{body}"));
        assert_eq!(value, 0.0, "{line}");
    }
}
