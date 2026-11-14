//! The `/metrics` read's response shape: a ratio-only snapshot, a snapshot with capacity and a
//! fleet rollup against the rank cache. The parser and the gauge tables are the controller crate's
//! (`smg_policy_controller::ingest::scrape`), tested there.

use super::super::*;

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
