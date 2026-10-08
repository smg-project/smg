//! Exercise the installed recorder so startup ordering and bucket configuration
//! are covered by the same scrape output production serves.

use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use metrics::{counter, gauge, histogram};
use smg::{
    observability::metrics::{start_prometheus, Metrics, MetricsHandle, PrometheusConfig},
    worker::{BasicWorkerBuilder, WorkerRegistry},
};

fn prometheus_handle() -> &'static MetricsHandle {
    static HANDLE: OnceLock<MetricsHandle> = OnceLock::new();
    HANDLE.get_or_init(|| start_prometheus(PrometheusConfig::default()))
}

#[test]
fn installed_recorder_exports_metric_descriptions() {
    let handle = prometheus_handle();
    let registry = WorkerRegistry::new();
    registry
        .register(Arc::new(
            BasicWorkerBuilder::new("http://metadata-regression").build(),
        ))
        .unwrap();
    Metrics::record_http_request("GET", "/metadata-regression");
    Metrics::set_worker_health("http://metadata-regression", true);
    Metrics::record_http_duration("GET", "/metadata-regression", Duration::from_millis(5));
    histogram!("smg_tokio_event_loop_delay_seconds").record(0.001);
    gauge!("router_mesh_peer_connections", "peer" => "metadata-regression").set(1.0);
    counter!("smg_rl_fanout_total", "result" => "success").increment(1);
    counter!("smg_scheduler_admit_total", "class" => "default", "outcome" => "admitted")
        .increment(1);

    let rendered = handle.render();
    for (name, kind) in [
        ("smg_http_requests_total", "counter"),
        ("smg_worker_health", "gauge"),
        ("smg_http_request_duration_seconds", "histogram"),
        ("smg_tokio_event_loop_delay_seconds", "histogram"),
        ("router_mesh_peer_connections", "gauge"),
        ("smg_rl_fanout_total", "counter"),
        ("smg_scheduler_admit_total", "counter"),
    ] {
        assert!(
            rendered
                .lines()
                .any(|line| line.starts_with(&format!("# HELP {name} "))),
            "missing description for {name}:\n{rendered}"
        );
        assert!(rendered.contains(&format!("# TYPE {name} {kind}\n")));
    }
}

#[test]
fn installed_recorder_exports_kv_event_lag_as_histogram() {
    let handle = prometheus_handle();
    let worker = "http://kv-lag-regression";
    let registry = WorkerRegistry::new();
    registry
        .register(Arc::new(BasicWorkerBuilder::new(worker).build()))
        .unwrap();
    for lag in [0.0, 0.001_5, 0.01, 0.2, 2.0, 20.0, 60.0, 90.0] {
        Metrics::record_kv_event_lag(worker, lag);
    }
    Metrics::record_kv_event_apply(worker, 0.000_001_5);
    Metrics::record_kv_index_lookup("lag-bucket-isolation", 0.000_001_5);
    let rendered = handle.render();

    for (name, label) in [
        ("smg_kv_event_apply_seconds", format!("worker=\"{worker}\"")),
        (
            "smg_kv_index_lookup_seconds",
            "index=\"lag-bucket-isolation\"".to_string(),
        ),
    ] {
        assert!(
            rendered
                .lines()
                .any(|line| line.starts_with(&format!("{name}_bucket{{"))
                    && line.contains(&label)
                    && line.contains("le=\"0.000002\"")
                    && line.ends_with(" 1")),
            "lookup/apply must retain microsecond buckets: {rendered}"
        );
    }
    assert!(rendered.contains("# TYPE smg_kv_event_lag_seconds histogram\n"));
    for (bound, count) in [
        ("0.001", 1),
        ("0.005", 2),
        ("0.01", 3),
        ("0.25", 4),
        ("2.5", 5),
        ("30", 6),
        ("60", 7),
        ("+Inf", 8),
    ] {
        assert!(
            rendered.lines().any(|line| {
                line.starts_with("smg_kv_event_lag_seconds_bucket{")
                    && line.contains(&format!("worker=\"{worker}\""))
                    && line.contains(&format!("le=\"{bound}\""))
                    && line.ends_with(&format!(" {count}"))
            }),
            "missing lag bucket {bound} with count {count}:\n{rendered}"
        );
    }
    assert!(rendered.lines().any(|line| {
        line.starts_with("smg_kv_event_lag_seconds_count{")
            && line.contains(&format!("worker=\"{worker}\""))
            && line.ends_with(" 8")
    }));
    let sum = rendered
        .lines()
        .find(|line| {
            line.starts_with("smg_kv_event_lag_seconds_sum{")
                && line.contains(&format!("worker=\"{worker}\""))
        })
        .unwrap()
        .rsplit_once(' ')
        .unwrap()
        .1
        .parse::<f64>()
        .unwrap();
    assert!((sum - 172.2115).abs() < 1e-9);
    assert!(!rendered.lines().any(|line| {
        line.starts_with("smg_kv_event_lag_seconds{") && line.contains("quantile=")
    }));
}
