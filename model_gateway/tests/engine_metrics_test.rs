//! Integration test for the `smg_engine_*` re-export (W2).
//!
//! Drives the real Prometheus exporter and the :29000-style metrics HTTP
//! server end to end: a worker load snapshot carrying a `disagg` section is
//! recorded through the same `Metrics::record_engine_load` call the load
//! monitor makes, then scraped over HTTP to assert the PD gauge is exposed.
//!
//! This file is its own test binary (process), so installing the global
//! Prometheus recorder here is safe and does not collide with other suites.

use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};
use smg::observability::{
    metrics::{start_prometheus, Metrics, PrometheusConfig},
    metrics_server::start_metrics_server,
};

#[tokio::test]
async fn engine_pd_gauge_appears_on_metrics_endpoint() {
    // The recorder is global and may only be installed once per process;
    // `start_prometheus` does that install and hands back the handle the
    // server renders. Port 0 binds an ephemeral port with no reservation race.
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    });
    let (addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");

    let response = WorkerLoadResponse {
        timestamp: "t".to_string(),
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: 0,
            num_running_reqs: 5,
            disagg_mode: Some("prefill".to_string()),
            kv_transfer_latency_ms: Some(2.5),
            kv_transfer_speed_gb_s: Some(8.0),
            prefill_queue_reqs: Some(6),
            decode_queue_reqs: Some(2),
            ..Default::default()
        }],
        ..Default::default()
    };
    Metrics::record_engine_load("grpc://prefill-0:30000", "test-model", &response);

    let body = reqwest::get(format!("http://{addr}/metrics"))
        .await
        .expect("metrics endpoint reachable")
        .text()
        .await
        .expect("metrics body");

    // Match the metric name AND its labels on the SAME rendered line, so the
    // name and the role/dp_rank labels can't be satisfied by different samples.
    let pd_latency_line = body
        .lines()
        .find(|l| l.starts_with("smg_engine_pd_kv_transfer_latency_ms{"))
        .unwrap_or_else(|| panic!("PD KV transfer latency sample missing from /metrics:\n{body}"));
    assert!(
        pd_latency_line.contains("role=\"prefill\"") && pd_latency_line.contains("dp_rank=\"0\""),
        "PD latency sample missing role/dp_rank labels: {pd_latency_line}"
    );
    assert!(
        body.lines()
            .any(|l| l.starts_with("smg_engine_running_requests{")),
        "core engine sample missing from /metrics:\n{body}"
    );
    assert!(
        !body.contains("smg_allocator_"),
        "an rlib consumer that did not register jemalloc exposed allocator gauges:\n{body}"
    );
}
