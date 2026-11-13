//! The `/metrics` read's response shape: a ratio-only snapshot, a snapshot with capacity and a
//! fleet rollup against the rank cache. The parser and the gauge tables are the controller crate's
//! (`smg_policy_controller::ingest::scrape`), tested there; the monitor's own fetch is read here
//! against a body whose usage gauge is not a number.

use tokio::{net::TcpListener, task::JoinSet};

use super::super::*;
use crate::worker::{
    test_support::no_health_check, BasicWorkerBuilder, ConnectionMode, WorkerType,
};

#[test]
fn metric_derived_snapshot_has_no_absolute_token_data() {
    // A ratio-only snapshot (from `/metrics`) must not be treated as
    // carrying absolute token counts: it stays out of the DP cache,
    // even at high KV usage.
    let resp = WorkerMonitor::single_rank(SchedulerLoadSnapshot {
        token_usage: 0.75,
        num_running_reqs: 3,
        ..Default::default()
    });
    assert!(!resp.has_absolute_token_data());
    assert_eq!(resp.total_used_tokens(), 0);
    assert_eq!(resp.effective_token_usage(), 0.75);
}

#[test]
fn real_snapshot_with_capacity_has_absolute_token_data() {
    // A snapshot from gRPC/`/v1/loads` reports KV capacity, so its
    // absolute token fields are meaningful (even when idle).
    let resp = WorkerMonitor::single_rank(SchedulerLoadSnapshot {
        max_total_num_tokens: 8192,
        num_used_tokens: 1024,
        ..Default::default()
    });
    assert!(resp.has_absolute_token_data());
    assert_eq!(resp.total_used_tokens(), 1024);
}

#[test]
fn fleet_rollup_is_not_dp_ranks() {
    // A gateway worker answers `/loads` with one entry per downstream
    // worker, all at `dp_rank: 0`. Keying those by rank collapses them
    // onto a single entry, so the DP cache must reject the response.
    let fleet = WorkerLoadResponse {
        dp_rank_count: 2,
        loads: vec![
            SchedulerLoadSnapshot {
                worker: Some("http://a:8000".to_string()),
                max_total_num_tokens: 8192,
                num_used_tokens: 1024,
                ..Default::default()
            },
            SchedulerLoadSnapshot {
                worker: Some("http://b:8000".to_string()),
                max_total_num_tokens: 8192,
                num_used_tokens: 4096,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    assert!(fleet.has_absolute_token_data());
    assert!(!fleet.ranks_are_dp_ranks());
    assert_eq!(fleet.dp_rank_loads().len(), 1);

    // One engine's own ranks are distinct and stay eligible.
    let engine = WorkerLoadResponse {
        dp_rank_count: 2,
        loads: vec![
            SchedulerLoadSnapshot {
                dp_rank: 0,
                max_total_num_tokens: 8192,
                num_used_tokens: 1024,
                ..Default::default()
            },
            SchedulerLoadSnapshot {
                dp_rank: 1,
                max_total_num_tokens: 8192,
                num_used_tokens: 4096,
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    assert!(engine.ranks_are_dp_ranks());
    assert_eq!(engine.dp_rank_loads().len(), 2);
}

/// A metrics endpoint that serves `body`: its base address, and the set
/// whose task serves it until the set is dropped.
async fn metrics_endpoint(body: &'static str) -> (String, JoinSet<()>) {
    let app =
        axum::Router::new().route("/metrics", axum::routing::get(move || async move { body }));
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a loopback port");
    let base = format!(
        "http://{}",
        listener.local_addr().expect("the bound address")
    );
    let mut served = JoinSet::new();
    served.spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("the endpoint serves until the test ends");
    });
    (base, served)
}

/// A regular HTTP worker of `runtime` at `base`, with no health checker
/// behind it.
fn http_worker(base: &str, runtime: RuntimeType) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(base)
            .worker_type(WorkerType::Regular)
            .runtime_type(runtime)
            .connection_mode(ConnectionMode::Http)
            .health_config(no_health_check())
            .build(),
    )
}

/// The gauge table's rule at the monitor's own fetch: a body whose KV-usage
/// gauge is not a number is a worker without the gauge, and the fetch has no
/// report. Kept as a sample it would publish zero usage, an idle worker's
/// figure, for a worker nobody knows the pressure of.
#[tokio::test]
async fn a_body_whose_kv_usage_is_not_a_number_is_no_report_to_the_monitors_fetch() {
    let families = [
        (
            RuntimeType::Vllm,
            "vllm:num_requests_running 3\nvllm:kv_cache_usage_perc NaN\n",
            "vllm:num_requests_running 3\nvllm:kv_cache_usage_perc 0.5\n",
        ),
        (
            RuntimeType::Sglang,
            "sglang:num_running_reqs 3\nsglang:token_usage NaN\n",
            "sglang:num_running_reqs 3\nsglang:token_usage 0.5\n",
        ),
    ];
    for (runtime, not_a_number, a_number) in families {
        let (base, _served) = metrics_endpoint(not_a_number).await;
        let unknown = WorkerMonitor::fetch_http_gauges(&http_worker(&base, runtime)).await;
        assert!(unknown.is_none(), "{runtime:?}: {unknown:?}");

        let (base, _served) = metrics_endpoint(a_number).await;
        let known = WorkerMonitor::fetch_http_gauges(&http_worker(&base, runtime))
            .await
            .expect("a report from a body with the gauge");
        assert_eq!(
            (known.loads[0].num_running_reqs, known.loads[0].token_usage),
            (3, 0.5),
            "{runtime:?}"
        );
    }
}
