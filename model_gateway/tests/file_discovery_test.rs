//! File discovery against a real registry and job queue: a manifest on disk,
//! mock engines behind it, and the provider's loop rereading it.

#![allow(clippy::unwrap_used, clippy::allow_attributes)]

mod common;

use std::{path::Path, sync::Arc, time::Duration};

use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use smg::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    service_discovery::{
        start_service_discovery, FileProviderConfig, RuntimeDiscoveryConfig, DISCOVERY_ID_LABEL,
        DISCOVERY_PROVIDER_LABEL,
    },
};
use tokio::task::JoinHandle;

/// Short enough that a test sees several rereads a second.
const REREAD: Duration = Duration::from_millis(200);
const CONVERGE: Duration = Duration::from_secs(15);

async fn test_context() -> Arc<AppContext> {
    let mut config = RouterConfig::builder()
        .worker_startup_timeout_secs(2)
        .build_unchecked();
    // Remove at once rather than after the drain window.
    config.health_check.drain_settle_secs = 0;
    common::create_test_context(config).await
}

async fn start_mock_engine() -> (MockWorker, u16) {
    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let url = worker.start().await.unwrap();
    let port = url.rsplit(':').next().unwrap().parse().unwrap();
    (worker, port)
}

/// Replace the manifest as the provider asks writers to: write a temporary
/// file beside it, then rename it over.
fn write_manifest(path: &Path, json: &str) {
    let staged = path.with_extension("staged");
    std::fs::write(&staged, json).unwrap();
    std::fs::rename(&staged, path).unwrap();
}

/// A manifest listing one regular worker per port, each with an explicit id.
fn manifest(ports: &[u16]) -> String {
    let workers: Vec<String> = ports
        .iter()
        .map(|port| format!(r#"{{"id": "engine-{port}", "url": "127.0.0.1:{port}"}}"#))
        .collect();
    format!(r#"{{"version": 1, "workers": [{}]}}"#, workers.join(", "))
}

async fn start_file_discovery(path: &Path, app_context: &Arc<AppContext>) -> JoinHandle<()> {
    let config = FileProviderConfig::new(
        path.to_path_buf(),
        REREAD,
        &RoutingMode::Regular {
            worker_urls: vec![],
        },
    )
    .unwrap();
    start_service_discovery(
        RuntimeDiscoveryConfig::File(config),
        Arc::clone(app_context),
    )
    .await
    .unwrap()
}

/// Poll until the registry satisfies `predicate` or fail with its contents.
async fn wait_for(app_context: &AppContext, what: &str, predicate: impl Fn(&[String]) -> bool) {
    let deadline = tokio::time::Instant::now() + CONVERGE;
    loop {
        let urls = app_context.worker_registry.get_all_urls();
        if predicate(&urls) {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {what}; registry: {urls:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn has_url(urls: &[String], port: u16) -> bool {
    urls.iter()
        .any(|url| url.ends_with(&format!("127.0.0.1:{port}")))
}

/// Several rereads, so a wrong reaction to the current file has had its
/// chance to happen.
async fn let_rereads_pass() {
    tokio::time::sleep(REREAD * 5).await;
}

/// The registry follows the manifest: listed workers register as the file
/// provider's, and leave when it stops listing them, down to an intentionally
/// empty fleet.
#[tokio::test]
async fn workers_follow_the_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.json");
    let app_context = test_context().await;
    let (_engine_a, port_a) = start_mock_engine().await;
    let (_engine_b, port_b) = start_mock_engine().await;

    write_manifest(&path, &manifest(&[port_a, port_b]));
    let handle = start_file_discovery(&path, &app_context).await;
    wait_for(&app_context, "both manifest workers registered", |urls| {
        has_url(urls, port_a) && has_url(urls, port_b)
    })
    .await;

    for worker in app_context.worker_registry.get_all() {
        let labels = &worker.metadata().spec.labels;
        assert_eq!(
            labels.get(DISCOVERY_PROVIDER_LABEL).map(String::as_str),
            Some("file"),
            "{}",
            worker.url()
        );
        assert!(
            labels
                .get(DISCOVERY_ID_LABEL)
                .is_some_and(|id| id.starts_with("engine-")),
            "{} carries the manifest id",
            worker.url()
        );
    }

    write_manifest(&path, &manifest(&[port_a]));
    wait_for(&app_context, "the unlisted worker removed", |urls| {
        has_url(urls, port_a) && !has_url(urls, port_b)
    })
    .await;

    write_manifest(&path, &manifest(&[]));
    wait_for(
        &app_context,
        "an empty manifest removing every worker",
        |urls| urls.is_empty(),
    )
    .await;

    handle.abort();
}

/// A manifest that fails to parse is reported and changes nothing: the last
/// good manifest's workers stay until a valid rewrite drops them.
#[tokio::test]
async fn an_invalid_manifest_keeps_the_last_good_workers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.json");
    let app_context = test_context().await;
    let (_engine, port) = start_mock_engine().await;

    write_manifest(&path, &manifest(&[port]));
    let handle = start_file_discovery(&path, &app_context).await;
    wait_for(&app_context, "the manifest worker registered", |urls| {
        has_url(urls, port)
    })
    .await;

    for broken in [
        // A partial write.
        r#"{"version": 1, "workers": ["#,
        // A misspelled field, which a lenient parser would drop.
        r#"{"version": 1, "workers": [{"url": "127.0.0.1:1", "worker_typ": "regular"}]}"#,
    ] {
        write_manifest(&path, broken);
        let_rereads_pass().await;
        assert!(
            has_url(&app_context.worker_registry.get_all_urls(), port),
            "worker dropped after the invalid manifest {broken}"
        );
    }

    write_manifest(&path, &manifest(&[]));
    wait_for(&app_context, "the valid rewrite applied", |urls| {
        !has_url(urls, port)
    })
    .await;

    handle.abort();
}

/// An absent manifest is retried, never read as an empty fleet: discovery
/// waits for a manifest that does not exist yet, and keeps its workers when
/// the manifest disappears.
#[tokio::test]
async fn a_missing_manifest_is_retried_not_emptied() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.json");
    let app_context = test_context().await;
    let (_engine, port) = start_mock_engine().await;

    let handle = start_file_discovery(&path, &app_context).await;
    let_rereads_pass().await;
    assert!(app_context.worker_registry.get_all_urls().is_empty());

    write_manifest(&path, &manifest(&[port]));
    wait_for(
        &app_context,
        "the late manifest's worker registered",
        |urls| has_url(urls, port),
    )
    .await;

    std::fs::remove_file(&path).unwrap();
    let_rereads_pass().await;
    assert!(
        has_url(&app_context.worker_registry.get_all_urls(), port),
        "worker dropped after the manifest was removed"
    );

    handle.abort();
}
