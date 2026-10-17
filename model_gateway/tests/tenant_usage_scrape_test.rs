//! The per-tenant usage families reach `/metrics` through the metrics server,
//! with their `# HELP` and `# TYPE` lines, bounded by the configured maximum.
//!
//! Installs the process-wide usage table the way start-up does, drives the
//! real exporter and metrics server, and reads the scrape as Prometheus
//! would. This file is its own test binary (process), so installing the
//! global recorder and the table here collides with no other suite.

use std::{sync::Arc, time::Duration};

use smg::observability::{
    metrics::{start_prometheus, PrometheusConfig},
    metrics_server::start_metrics_server,
    tenant_usage::{self, RequestUsage, TenantUsage},
};

fn usage(prompt: u64, completion: u64, millis: u64) -> RequestUsage {
    RequestUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        elapsed: Duration::from_millis(millis),
    }
}

#[tokio::test]
async fn usage_families_render_in_the_scrape_with_help_text_and_the_other_row() {
    let handle = start_prometheus(PrometheusConfig {
        port: 0,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    });
    let (addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");

    let table = Arc::new(TenantUsage::new(2));
    tenant_usage::install(Arc::clone(&table));
    let installed = tenant_usage::installed().expect("the table is installed");
    assert!(Arc::ptr_eq(installed, &table));

    // Two tenants take the two rows; the third is booked under `other`.
    tenant_usage::record("header:acme", "model-a", usage(100, 20, 300));
    tenant_usage::record("header:acme", "model-a", usage(50, 10, 200));
    tenant_usage::record("header:bravo", "model-b", usage(7, 3, 100));
    tenant_usage::record("ip:203.0.113.9", "model-a", usage(1000, 1, 1000));

    let body = smg::http_client::client()
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("metrics endpoint reachable")
        .text()
        .await
        .expect("metrics body");

    for family in [
        "smg_tenant_usage_requests_total",
        "smg_tenant_usage_tokens_total",
        "smg_tenant_usage_request_seconds_total",
        "smg_tenant_usage_tracked_tenants",
        "smg_tenant_usage_max_tenants",
    ] {
        assert!(
            body.contains(&format!("# HELP {family} ")),
            "{family} carries its HELP line:\n{body}"
        );
        assert!(
            body.contains(&format!("# TYPE {family} ")),
            "{family} carries its TYPE line:\n{body}"
        );
    }
    let has = |line: &str| {
        assert!(
            body.contains(&format!("{line}\n")),
            "{line} is in the scrape:\n{body}"
        );
    };
    has("smg_tenant_usage_requests_total{tenant=\"header:acme\",model=\"model-a\"} 2");
    has("smg_tenant_usage_tokens_total{tenant=\"header:acme\",model=\"model-a\",token_type=\"input\"} 150");
    has("smg_tenant_usage_tokens_total{tenant=\"header:acme\",model=\"model-a\",token_type=\"output\"} 30");
    has("smg_tenant_usage_request_seconds_total{tenant=\"header:acme\",model=\"model-a\"} 0.5");
    has("smg_tenant_usage_requests_total{tenant=\"header:bravo\",model=\"model-b\"} 1");
    has("smg_tenant_usage_requests_total{tenant=\"other\",model=\"model-a\"} 1");
    has("smg_tenant_usage_tokens_total{tenant=\"other\",model=\"model-a\",token_type=\"input\"} 1000");
    has("smg_tenant_usage_tracked_tenants 2");
    has("smg_tenant_usage_max_tenants 2");
    assert!(
        !body.contains("tenant=\"ip:203.0.113.9\""),
        "a tenant past the maximum has no row of its own:\n{body}"
    );

    // The snapshot the admin route serves agrees with the scrape.
    let snapshot = table.snapshot();
    assert_eq!(snapshot.tracked_tenants, 2);
    assert_eq!(snapshot.tenants.len(), 2);
    assert_eq!(snapshot.other.totals.requests, 1);
    assert_eq!(snapshot.other.totals.prompt_tokens, 1000);
}
