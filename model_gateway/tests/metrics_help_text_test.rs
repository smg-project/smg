//! Every `smg_*` family on `/metrics` carries its `# HELP` text.
//!
//! Drives the real exporter the way the gateway does: `start_prometheus`
//! installs the global recorder (so the descriptions have to land on the
//! installed recorder, not the no-op one that precedes it), the metrics server
//! serves the scrape, and a sample of families from every layer is recorded
//! before the scrape is parsed.
//!
//! This file is its own test binary (process), so installing the global
//! Prometheus recorder here does not collide with other suites.

use std::collections::BTreeSet;

use smg::observability::{
    metrics::{start_prometheus, Metrics, PrometheusConfig},
    metrics_server::start_metrics_server,
};

/// Names of the `smg_` families announced by `# <marker> <name> ...` lines.
fn families(body: &str, marker: &str) -> BTreeSet<String> {
    body.lines()
        .filter_map(|line| line.strip_prefix(marker))
        .filter_map(|rest| rest.split_whitespace().next())
        .filter(|name| name.starts_with("smg_"))
        .map(str::to_owned)
        .collect()
}

#[tokio::test]
async fn every_smg_family_has_help_text() {
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    });
    let (addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");

    // One family per layer: router, policy, PD, worker protection, retry,
    // circuit breaker, KV events. The tokenizer families are recorded by the
    // scrape handler itself.
    let worker = "grpc://worker-a:50051";
    Metrics::record_upstream_send_retry("http");
    Metrics::record_routing_key_source("header");
    Metrics::record_worker_manual_policy_branch("vacant");
    Metrics::record_worker_consistent_hashing_policy_branch("routing_key_hit");
    Metrics::record_worker_prefix_hash_policy_branch("ring_hit");
    Metrics::set_worker_routing_keys_active(worker, 1);
    Metrics::record_pd_admission_wait();
    Metrics::record_pd_admission_shed();
    Metrics::record_worker_overload_shed("selection");
    Metrics::record_worker_retry("regular", "chat");
    Metrics::set_worker_health(worker, true);
    Metrics::set_worker_cb_state(worker, 0);
    Metrics::record_worker_cb_transition(worker, "closed", "open");
    Metrics::record_kv_event_lag(worker, 0.004);

    let body = reqwest::get(format!("http://{addr}/metrics"))
        .await
        .expect("metrics endpoint reachable")
        .text()
        .await
        .expect("metrics body");

    let typed = families(&body, "# TYPE ");
    let helped = families(&body, "# HELP ");
    assert!(
        typed.len() >= 15,
        "expected the recorded families on the scrape, got {}:\n{body}",
        typed.len()
    );
    let missing: Vec<&String> = typed.difference(&helped).collect();
    assert!(
        missing.is_empty(),
        "{} of {} smg_ families have no # HELP line: {missing:?}\n{body}",
        missing.len(),
        typed.len()
    );
}
