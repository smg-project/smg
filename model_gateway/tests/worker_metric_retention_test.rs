//! Production-recorder coverage for registered worker metric ownership.

use std::sync::{Arc, OnceLock};

use smg::{
    observability::metrics::{start_prometheus, Metrics, MetricsHandle, PrometheusConfig},
    worker::{BasicWorkerBuilder, ModelCard, WorkerRegistry},
};

fn handle() -> &'static MetricsHandle {
    static HANDLE: OnceLock<MetricsHandle> = OnceLock::new();
    HANDLE.get_or_init(|| start_prometheus(PrometheusConfig::default()))
}

fn worker(url: &str) -> Arc<dyn smg::worker::Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new("retention-test"))
            .build(),
    )
}

fn has_worker(rendered: &str, url: &str) -> bool {
    rendered
        .lines()
        .any(|line| line.contains(&format!("worker=\"{url}\"")))
}

#[test]
fn final_removal_retires_all_series_and_late_updates_cannot_resurrect_them() {
    let handle = handle();
    let registry = WorkerRegistry::new();
    let url = "http://retention-removed";
    let id = registry.register(worker(url)).unwrap();
    Metrics::set_worker_requests_active(url, 7);
    Metrics::record_kv_event_lag(url, 0.000_001);
    Metrics::record_worker_cb_outcome(url, "failure");
    assert!(has_worker(&handle.render(), url));

    registry.remove(&id).unwrap();
    assert!(
        !has_worker(&handle.render(), url),
        "removed worker remained in scrape"
    );
    Metrics::set_worker_health(url, true);
    Metrics::record_kv_event_lag(url, 0.5);
    Metrics::record_worker_cb_outcome(url, "failure");
    assert!(
        !has_worker(&handle.render(), url),
        "late absent-URL update recreated series"
    );

    let id = registry.register(worker(url)).unwrap();
    let rendered = handle.render();
    assert!(
        rendered.lines().any(|line| {
            line.starts_with("smg_worker_cb_outcomes_total{")
                && line.contains(&format!("worker=\"{url}\""))
                && line.contains("outcome=\"failure\"")
                && line.ends_with(" 0")
        }),
        "reused URL retained retired counter: {rendered}"
    );
    registry.remove(&id).unwrap();
}

#[test]
fn quiet_active_metrics_survive_upkeep_and_metadata_is_emitted_once() {
    let handle = handle();
    let registry = WorkerRegistry::new();
    for url in ["http://retention-quiet-a", "http://retention-quiet-b"] {
        registry.register(worker(url)).unwrap();
        Metrics::set_worker_requests_active(url, 3);
        Metrics::record_worker_cb_outcome(url, "failure");
        Metrics::record_kv_event_lag(url, 0.000_001);
    }
    for _ in 0..3 {
        handle.run_upkeep();
    }
    let rendered = handle.render();
    for url in ["http://retention-quiet-a", "http://retention-quiet-b"] {
        assert!(rendered
            .lines()
            .any(|line| line.starts_with("smg_worker_requests_active{")
                && line.contains(&format!("worker=\"{url}\""))
                && line.ends_with(" 3")));
        assert!(rendered
            .lines()
            .any(|line| line.starts_with("smg_kv_event_lag_seconds_count{")
                && line.contains(&format!("worker=\"{url}\""))
                && line.ends_with(" 1")));
    }
    for family in [
        "smg_worker_health",
        "smg_worker_cb_outcomes_total",
        "smg_kv_event_lag_seconds",
    ] {
        for directive in ["HELP", "TYPE"] {
            assert_eq!(
                rendered
                    .lines()
                    .filter(|line| line.starts_with(&format!("# {directive} {family} ")))
                    .count(),
                1,
                "duplicate/missing {directive} for {family}: {rendered}"
            );
        }
    }
}

#[test]
fn registry_owners_replacements_and_duplicates_preserve_scope_until_final_release() {
    let handle = handle();
    let url = "http://retention-owners";
    let first = Arc::new(WorkerRegistry::new());
    let id = first.register(worker(url)).unwrap();
    assert!(first.register(worker(url)).is_none());
    Metrics::record_worker_cb_outcome(url, "failure");
    assert!(first.replace(&id, worker(url)));
    let second = WorkerRegistry::new();
    let second_id = second.register(worker(url)).unwrap();
    first.remove(&id).unwrap();
    assert!(
        has_worker(&handle.render(), url),
        "other registry ownership was lost"
    );
    second.remove(&second_id).unwrap();
    assert!(
        !has_worker(&handle.render(), url),
        "final owner did not retire scope"
    );

    let url = "http://retention-registry-drop";
    first.register(worker(url)).unwrap();
    let clone = first.clone();
    drop(first);
    assert!(
        has_worker(&handle.render(), url),
        "an Arc clone still owns the registry"
    );
    drop(clone);
    assert!(
        !has_worker(&handle.render(), url),
        "final registry Drop leaked ownership"
    );
}

#[test]
fn repeated_worker_churn_does_not_accumulate_retired_scrape_series() {
    let handle = handle();
    let registry = WorkerRegistry::new();
    for n in 0..50 {
        let url = format!("http://retention-churn-{n}");
        let id = registry.register(worker(&url)).unwrap();
        Metrics::record_kv_event_lag(&url, 0.000_001);
        registry.remove(&id).unwrap();
    }
    assert!(!handle.render().contains("worker=\"http://retention-churn-"));
}
