//! The `/metrics` scrape read for a worker's load: the engine families' gauges, a body with none of
//! them, and the rank loads a data-parallel engine reports.

use super::super::*;

// Trimmed sample mirroring the fields observed on a live vLLM v0.7.3
// `/metrics` endpoint (dev ORD cluster).
const VLLM_METRICS: &str = r#"
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="llama"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="llama"} 5.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="llama"} 0.75
vllm:generation_tokens_total{model_name="llama"} 123456.0
"#;

const SGLANG_METRICS: &str = r#"
sglang:num_running_reqs{model="llama"} 2.0
sglang:num_queue_reqs{model="llama"} 4.0
sglang:token_usage{model="llama"} 0.42
sglang:utilization{model="llama"} 0.9
"#;

#[test]
fn parses_gauges_ignoring_comments_and_labels() {
    let m = PromScrape::parse(VLLM_METRICS);
    assert!(m.has("vllm:num_requests_running"));
    assert_eq!(m.sum("vllm:num_requests_running"), 3.0);
    assert_eq!(m.sum("vllm:num_requests_waiting"), 5.0);
    assert_eq!(m.mean("vllm:gpu_cache_usage_perc"), 0.75);
    assert!(!m.has("vllm:does_not_exist"));
    assert_eq!(m.sum("vllm:does_not_exist"), 0.0);
    assert_eq!(m.mean("vllm:does_not_exist"), 0.0);
}

#[test]
fn sum_adds_ranks_mean_averages_ratios() {
    // Two DP-rank series for the same gauge name.
    let text = "g{dp=\"0\"} 10\ng{dp=\"1\"} 30\nr{dp=\"0\"} 0.2\nr{dp=\"1\"} 0.8\n";
    let m = PromScrape::parse(text);
    assert_eq!(m.sum("g"), 40.0); // counts add
    assert_eq!(m.mean("r"), 0.5); // ratios average
}

#[test]
fn vllm_metrics_map_onto_token_usage_snapshot() {
    let m = PromScrape::parse(VLLM_METRICS);
    let snap = SchedulerLoadSnapshot {
        num_running_reqs: m.sum("vllm:num_requests_running") as i32,
        num_waiting_reqs: m.sum("vllm:num_requests_waiting") as i32,
        token_usage: m.mean("vllm:gpu_cache_usage_perc"),
        ..Default::default()
    };
    let resp = WorkerMonitor::single_rank(snap);
    assert_eq!(resp.dp_rank_count, 1);
    assert_eq!(resp.effective_token_usage(), 0.75);
    assert_eq!(resp.loads[0].num_running_reqs, 3);
    assert_eq!(resp.loads[0].num_waiting_reqs, 5);
}

#[test]
fn sglang_metrics_map_onto_token_usage_snapshot() {
    let m = PromScrape::parse(SGLANG_METRICS);
    assert_eq!(m.mean("sglang:token_usage"), 0.42);
    assert_eq!(m.sum("sglang:num_running_reqs"), 2.0);
    assert_eq!(m.sum("sglang:num_queue_reqs"), 4.0);
}

#[test]
fn sglang_v054_underscore_prefix_is_recognized() {
    // SGLang v0.5.4+ renamed the metric prefix `sglang:` -> `sglang_`.
    let v054 = "sglang_token_usage{model=\"llama\"} 0.5\n\
                sglang_num_running_reqs{model=\"llama\"} 7\n\
                sglang_num_queue_reqs{model=\"llama\"} 1\n";
    let m = PromScrape::parse(v054);
    assert!(!m.has("sglang:token_usage"));
    let prefix = ["sglang:", "sglang_"]
        .into_iter()
        .find(|p| m.has(&format!("{p}token_usage")));
    assert_eq!(prefix, Some("sglang_"));
    assert_eq!(m.mean("sglang_token_usage"), 0.5);
    assert_eq!(m.sum("sglang_num_running_reqs"), 7.0);
}

#[test]
fn parse_ignores_trailing_timestamp_and_label_spaces() {
    // Optional trailing timestamp must not be read as the value, and a
    // space inside a quoted label value must not split the head early.
    let text = "vllm:gpu_cache_usage_perc{model_name=\"my model\"} 0.75 1719849600000\n";
    let m = PromScrape::parse(text);
    assert_eq!(m.mean("vllm:gpu_cache_usage_perc"), 0.75);
}

#[test]
fn vllm_v1_kv_cache_usage_metric_is_recognized() {
    // vLLM v1 renamed the gauge from `gpu_cache_usage_perc` to
    // `kv_cache_usage_perc`; the fetcher accepts either name.
    let v1 = "vllm:kv_cache_usage_perc{model_name=\"llama\"} 0.6\n";
    let m = PromScrape::parse(v1);
    assert!(!m.has("vllm:gpu_cache_usage_perc"));
    let kv_usage = ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"]
        .into_iter()
        .find(|name| m.has(name));
    assert_eq!(kv_usage, Some("vllm:kv_cache_usage_perc"));
    assert_eq!(m.mean("vllm:kv_cache_usage_perc"), 0.6);
}

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
