//! A host that takes the handle `start_prometheus` returns and renders it by
//! itself reads every histogram family with no `worker` label in that text.
//!
//! The gateway's own start asks the per-worker store to keep these families,
//! because its scrape appends the store's text; a host that never asks keeps
//! them with the exporter. No test of this binary asks, and the recorder is
//! this binary's, installed once for all its tests (see `recorder()` in the
//! group's `mod.rs`).

use metrics::histogram;

use crate::metrics::recorder;

/// The histogram families the gateway records with no `worker` label.
const FAMILIES: [&str; 29] = [
    "router_mesh_sync_round_duration_seconds",
    "smg_cache_aware_match_ratio",
    "smg_discovery_sync_duration_seconds",
    "smg_http_request_duration_seconds",
    "smg_kv_index_lookup_seconds",
    "smg_mcp_tool_duration_seconds",
    "smg_mm_media_fetch_seconds",
    "smg_mm_media_per_request",
    "smg_mm_processing_seconds",
    "smg_mm_request_images",
    "smg_pd_decode_ttft_seconds",
    "smg_pd_kv_transfer_duration_seconds",
    "smg_pd_prefill_admission_wait_seconds",
    "smg_pd_prefill_duration_seconds",
    "smg_pd_ttft_seconds",
    "smg_rl_control_call_duration_seconds",
    "smg_rl_fanout_duration_seconds",
    "smg_router_dispatch_wait_after_body_seconds",
    "smg_router_dispatch_wait_seconds",
    "smg_router_generation_duration_seconds",
    "smg_router_request_duration_seconds",
    "smg_router_request_time_per_output_token_seconds",
    "smg_router_time_to_first_token_seconds",
    "smg_router_tpot_seconds",
    "smg_router_ttft_seconds",
    "smg_scheduler_queue_wait_seconds",
    "smg_symphony_malformed_bytes",
    "smg_tokio_event_loop_delay_seconds",
    "smg_worker_retry_backoff_seconds",
];

#[test]
fn a_host_that_renders_the_handle_alone_reads_every_histogram_family() {
    let handle = recorder();

    for family in FAMILIES {
        histogram!(family, "reader" => "handle").record(0.5);
    }

    let body = handle.render();
    for family in FAMILIES {
        let count = format!("{family}_count{{reader=\"handle\"}} 1");
        assert!(
            body.lines().any(|line| line == count),
            "{family} left the exporter's text"
        );
    }
}
