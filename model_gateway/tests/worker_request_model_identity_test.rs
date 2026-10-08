//! Dispatch counters follow the current live registration, including stale handles.

use std::sync::{Arc, OnceLock};

use smg::{
    observability::metrics::{start_prometheus, Metrics, MetricsHandle, PrometheusConfig},
    routers::external::GatewayWorker,
    worker::{BasicWorkerBuilder, ModelCard, Worker, WorkerRegistry},
};

fn handle() -> &'static MetricsHandle {
    static HANDLE: OnceLock<MetricsHandle> = OnceLock::new();
    HANDLE.get_or_init(|| start_prometheus(PrometheusConfig::default()))
}

fn worker(url: &str, model: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model))
            .build(),
    )
}

fn total(url: &str, model: &str) -> Option<u64> {
    handle()
        .render()
        .lines()
        .find(|line| {
            line.starts_with("smg_worker_requests_total{")
                && line.contains(&format!("worker=\"{url}\""))
                && line.contains(&format!("model=\"{model}\""))
        })
        .and_then(|line| {
            line.rsplit_once(' ')
                .and_then(|(_, value)| value.parse().ok())
        })
}

#[test]
fn retained_external_and_internal_dispatches_use_replacement_model() {
    handle();
    let registry = WorkerRegistry::new();
    let url = "http://request-model-replacement";
    let retained = worker(url, "before");
    let external = GatewayWorker::handle(retained.clone());
    let id = registry.register(retained.clone()).unwrap();
    external.record_request();
    assert_eq!(total(url, "before"), Some(1));
    assert!(registry.replace(&id, worker(url, "after")));
    external.record_request();
    Metrics::record_worker_request(retained.url(), retained.model_id());
    assert_eq!(
        total(url, "before"),
        Some(1),
        "retained identity changed prior model total"
    );
    assert_eq!(
        total(url, "after"),
        Some(2),
        "current registration did not receive both dispatches"
    );
}

#[test]
fn retained_dispatch_after_url_reuse_uses_new_registration_model() {
    handle();
    let registry = WorkerRegistry::new();
    let url = "http://request-model-reuse";
    let retained = GatewayWorker::handle(worker(url, "retired"));
    let id = registry.register(worker(url, "retired")).unwrap();
    retained.record_request();
    registry.remove(&id).unwrap();
    retained.record_request();
    assert_eq!(total(url, "retired"), None);
    registry.register(worker(url, "reused")).unwrap();
    retained.record_request();
    assert_eq!(
        total(url, "retired"),
        None,
        "stale dispatch recreated retired model series"
    );
    assert_eq!(total(url, "reused"), Some(1));
}

#[test]
fn latest_live_owner_model_survives_replacement_and_owner_drop() {
    handle();
    let first = WorkerRegistry::new();
    let second = WorkerRegistry::new();
    let url = "http://request-model-shared-owners";
    let retained = GatewayWorker::handle(worker(url, "first"));
    let first_id = first.register(worker(url, "first")).unwrap();
    retained.record_request();
    second.register(worker(url, "second")).unwrap();
    retained.record_request();
    assert_eq!(total(url, "first"), Some(1));
    assert_eq!(total(url, "second"), Some(1));
    assert!(first.replace(&first_id, worker(url, "updated-first")));
    retained.record_request();
    assert_eq!(total(url, "updated-first"), Some(1));
    drop(first);
    retained.record_request();
    assert_eq!(
        total(url, "second"),
        Some(2),
        "latest surviving owner was not restored"
    );
    assert_eq!(total(url, "updated-first"), Some(1));
    drop(second);
    retained.record_request();
    assert_eq!(total(url, "second"), None);
}

#[test]
fn rejected_duplicate_registration_cannot_change_counter_identity() {
    handle();
    let registry = WorkerRegistry::new();
    let url = "http://request-model-rejected-duplicate";
    registry.register(worker(url, "accepted")).unwrap();
    assert!(registry.register(worker(url, "rejected")).is_none());
    Metrics::record_worker_request(url, "rejected");
    assert_eq!(total(url, "accepted"), Some(1));
    assert_eq!(total(url, "rejected"), None);
}

#[test]
fn direct_dispatch_counter_uses_the_private_scope_identity() {
    handle();
    let registry = WorkerRegistry::new();
    let url = "https://private-user:private-password@example.com:8443/v1?token=private-query";
    registry
        .register(worker(url, "private-dispatch-model"))
        .unwrap();
    Metrics::record_worker_request(url, "private-dispatch-model");
    let rendered = handle().render();
    for secret in ["private-user", "private-password", "private-query"] {
        assert!(
            !rendered.contains(secret),
            "direct dispatch exposed sensitive identity"
        );
    }
    let samples: Vec<_> = rendered
        .lines()
        .filter(|line| {
            line.starts_with("smg_worker_requests_total{")
                && line.contains("model=\"private-dispatch-model\"")
        })
        .collect();
    assert_eq!(samples.len(), 1);
    assert!(samples[0].contains("worker=\"worker:"));
    assert!(samples[0].ends_with(" 1"));
}
