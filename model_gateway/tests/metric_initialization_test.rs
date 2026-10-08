//! First-scrape coverage for metric families before any protection event occurs.

use std::sync::{Arc, OnceLock};

use smg::{
    observability::metrics::{start_prometheus, Metrics, MetricsHandle, PrometheusConfig},
    worker::{
        kv_index_backend::KvIndexKind, BasicWorkerBuilder, ConnectionMode, KvEventMonitor,
        ModelCard, Worker, WorkerRegistry,
    },
};

fn handle() -> &'static MetricsHandle {
    static HANDLE: OnceLock<MetricsHandle> = OnceLock::new();
    HANDLE.get_or_init(|| start_prometheus(PrometheusConfig::default()))
}

fn sample(rendered: &str, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
    rendered.lines().find_map(|line| {
        let rest = line.strip_prefix(name)?;
        if !rest.starts_with('{')
            || !labels
                .iter()
                .all(|(key, value)| rest.contains(&format!("{key}=\"{value}\"")))
        {
            return None;
        }
        line.rsplit_once(' ')?.1.parse().ok()
    })
}

fn assert_sample(rendered: &str, name: &str, labels: &[(&str, &str)], expected: f64) {
    assert_eq!(
        sample(rendered, name, labels),
        Some(expected),
        "missing/wrong {name} {labels:?}:\n{rendered}"
    );
}

#[test]
fn startup_exports_static_families_before_their_first_event() {
    let rendered = handle().render();
    for stage in ["selection", "dispatch"] {
        for name in [
            "smg_worker_overload_shed_total",
            "smg_worker_overload_fallback_total",
            "smg_worker_liveness_fallback_total",
        ] {
            assert_sample(&rendered, name, &[("stage", stage)], 0.0);
        }
    }
    for source in ["header", "rid"] {
        assert_sample(
            &rendered,
            "smg_routing_key_source_total",
            &[("source", source)],
            0.0,
        );
    }
    assert_sample(
        &rendered,
        "smg_policy_inflight_reconciled_total",
        &[("policy", "cache_aware")],
        0.0,
    );
    for name in [
        "smg_worker_retries_total",
        "smg_worker_retries_exhausted_total",
    ] {
        assert_sample(
            &rendered,
            name,
            &[("worker_type", "regular"), ("endpoint", "chat")],
            0.0,
        );
    }
    assert_sample(
        &rendered,
        "smg_worker_retry_backoff_seconds_count",
        &[("attempt", "1")],
        0.0,
    );
    assert_sample(
        &rendered,
        "smg_worker_retry_backoff_seconds_sum",
        &[("attempt", "1")],
        0.0,
    );
    assert_sample(
        &rendered,
        "smg_router_upstream_responses_total",
        &[
            ("router_type", "http"),
            ("status_code", "200"),
            ("error_code", ""),
        ],
        0.0,
    );
    assert!(
        !rendered
            .lines()
            .any(|line| line.starts_with("smg_discovery_")),
        "disabled discovery acquired series"
    );
}

#[test]
fn worker_registration_exports_zero_series_and_preserves_live_counters() {
    let handle = handle();
    let url = "http://protection-initialization";
    let model = "protection-initialization";
    let registry = WorkerRegistry::new();
    let worker = Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model))
            .build(),
    );
    registry
        .register(worker.clone())
        .expect("fresh worker registration");
    let rendered = handle.render();
    assert_sample(
        &rendered,
        "smg_workers_overloaded",
        &[("model", model)],
        0.0,
    );
    for reason in ["unreachable", "wedged"] {
        for name in ["smg_worker_stalled", "smg_worker_stall_transitions_total"] {
            assert_sample(&rendered, name, &[("worker", url), ("reason", reason)], 0.0);
        }
    }
    assert_sample(
        &rendered,
        "smg_worker_cb_transitions_total",
        &[("worker", url), ("from", "closed"), ("to", "open")],
        0.0,
    );
    assert_sample(
        &rendered,
        "smg_worker_cb_outcomes_total",
        &[("worker", url), ("outcome", "failure")],
        0.0,
    );
    assert_sample(
        &rendered,
        "smg_worker_cb_consecutive_failures",
        &[("worker", url)],
        0.0,
    );
    assert_sample(
        &rendered,
        "smg_worker_errors_total",
        &[
            ("worker_type", "regular"),
            ("connection_mode", "http"),
            ("error_type", "backend_error"),
        ],
        0.0,
    );

    Metrics::record_worker_cb_transition(url, "closed", "open");
    Metrics::set_worker_stalled(url, "unreachable", true);
    Metrics::set_workers_overloaded(model, 3);
    let replacement = Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model))
            .build(),
    );
    registry.register_or_replace(replacement);
    let rendered = handle.render();
    assert_sample(
        &rendered,
        "smg_worker_cb_transitions_total",
        &[("worker", url), ("from", "closed"), ("to", "open")],
        1.0,
    );
    assert_sample(
        &rendered,
        "smg_worker_stall_transitions_total",
        &[("worker", url), ("reason", "unreachable")],
        1.0,
    );
    assert_sample(
        &rendered,
        "smg_worker_stalled",
        &[("worker", url), ("reason", "unreachable")],
        1.0,
    );
    assert_sample(
        &rendered,
        "smg_workers_overloaded",
        &[("model", model)],
        3.0,
    );
}

#[tokio::test]
async fn kv_registration_exports_only_eligible_worker_and_index_series() {
    let handle = handle();
    let monitor = KvEventMonitor::with_kind(KvIndexKind::Chain, None);
    let registry = WorkerRegistry::new();
    let make_worker = |url, mode, model| -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .connection_mode(mode)
                .model(ModelCard::new(model))
                .build(),
        )
    };
    let url = "http://127.0.0.1:1";
    let model = "kv-initialization";
    let grpc = make_worker(url, ConnectionMode::Grpc, model);
    registry.register(grpc.clone()).unwrap();
    monitor.on_worker_added(&grpc).await;
    let rendered = handle.render();
    for name in [
        "smg_kv_event_missed_batches_total",
        "smg_kv_event_parentless_stores_total",
        "smg_kv_event_parentless_blocks_total",
        "smg_kv_event_degraded_ranks",
        "smg_kv_event_tail_depth",
        "smg_kv_index_blocks",
    ] {
        assert_sample(&rendered, name, &[("worker", url)], 0.0);
    }
    assert_sample(
        &rendered,
        "smg_kv_event_subscription_failures_total",
        &[("worker", url), ("reason", "panic")],
        0.0,
    );
    assert_sample(
        &rendered,
        "smg_kv_event_gaps_total",
        &[("worker", url), ("outcome", "replay_requested")],
        0.0,
    );
    assert_sample(
        &rendered,
        "smg_kv_event_resyncs_total",
        &[("worker", url), ("reason", "snapshot")],
        0.0,
    );
    for name in [
        "smg_kv_index_blocks_live",
        "smg_kv_index_runs_live",
        "smg_kv_index_memberships",
        "smg_kv_index_entries",
    ] {
        assert_sample(&rendered, name, &[("model", model)], 0.0);
    }
    for name in ["smg_kv_index_arena_bytes", "smg_kv_index_slab_bytes"] {
        assert!(
            sample(&rendered, name, &[("model", model)]).is_some(),
            "missing initial {name}"
        );
    }
    Metrics::record_kv_event_gap(url, "replay_requested", 4);
    Metrics::record_kv_event_parentless(url, 3, 5);
    Metrics::set_kv_event_tail_depth(url, 7);
    monitor.on_worker_added(&grpc).await;
    Metrics::set_kv_index_size(model, 9, 4);
    let second = make_worker("http://127.0.0.1:2", ConnectionMode::Grpc, model);
    registry.register(second.clone()).unwrap();
    monitor.on_worker_added(&second).await;
    for (url, mode) in [
        ("http://skip-http", ConnectionMode::Http),
        ("ipc:///skip-zmq", ConnectionMode::Zmq),
    ] {
        let worker = make_worker(url, mode, "skipped-index");
        registry.register(worker.clone()).unwrap();
        monitor.on_worker_added(&worker).await;
        assert!(
            !handle
                .render()
                .lines()
                .any(|line| line.starts_with("smg_kv_")
                    && line.contains(&format!("worker=\"{url}\"")))
        );
    }
    let rendered = handle.render();
    assert_sample(
        &rendered,
        "smg_kv_index_memberships",
        &[("model", model)],
        9.0,
    );
    assert_sample(&rendered, "smg_kv_index_entries", &[("model", model)], 4.0);
    assert_sample(
        &rendered,
        "smg_kv_event_missed_batches_total",
        &[("worker", url)],
        4.0,
    );
    assert_sample(
        &rendered,
        "smg_kv_event_tail_depth",
        &[("worker", url)],
        7.0,
    );
    monitor.stop().await;
    assert_sample(
        &rendered,
        "smg_kv_event_parentless_stores_total",
        &[("worker", url)],
        3.0,
    );
    assert_sample(
        &rendered,
        "smg_kv_event_parentless_blocks_total",
        &[("worker", url)],
        5.0,
    );
}
