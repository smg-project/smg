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
/// Far longer than any test: the startup read is the only reread a tick
/// gives, so anything read later was read because the watcher saw it change.
const NO_REREAD: Duration = Duration::from_secs(3600);
const CONVERGE: Duration = Duration::from_secs(15);

async fn test_context() -> Arc<AppContext> {
    // Remove at once rather than after the drain window.
    test_context_with_drain(0).await
}

async fn test_context_with_drain(drain_settle_secs: u64) -> Arc<AppContext> {
    let mut config = RouterConfig::builder()
        .worker_startup_timeout_secs(2)
        .build_unchecked();
    config.health_check.drain_settle_secs = drain_settle_secs;
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
    start_file_discovery_every(path, REREAD, app_context).await
}

async fn start_file_discovery_every(
    path: &Path,
    interval: Duration,
    app_context: &Arc<AppContext>,
) -> JoinHandle<()> {
    let config = FileProviderConfig::new(
        path.to_path_buf(),
        interval,
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

/// `(discovery id, KV connector, Ready)` of every registered worker at `port`.
fn registrations_at(app_context: &AppContext, port: u16) -> Vec<(String, Option<String>, bool)> {
    app_context
        .worker_registry
        .get_all()
        .into_iter()
        .filter(|worker| worker.url().ends_with(&format!(":{port}")))
        .map(|worker| {
            let spec = &worker.metadata().spec;
            let id = spec
                .labels
                .get(DISCOVERY_ID_LABEL)
                .cloned()
                .unwrap_or_default();
            let ready = worker.status() == openai_protocol::worker::WorkerStatus::Ready;
            (id, spec.kv_connector.clone(), ready)
        })
        .collect()
}

/// Editing a record but keeping its id changes its configuration, not its
/// instance: the worker is updated to the new settings rather than left as it
/// was registered. This is the common way to fix a manifest.
///
/// It must stay registered throughout. The drain window is real here and the
/// worker is Ready before the edit: a removal submitted beside the update
/// would drain the worker, then delete whatever sits at its URL once the drain
/// ends, which by then is the updated worker.
#[tokio::test]
async fn an_in_place_edit_updates_the_worker_in_place() {
    const DRAIN: Duration = Duration::from_secs(2);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.json");
    let app_context = test_context_with_drain(DRAIN.as_secs()).await;
    let (_engine, port) = start_mock_engine().await;
    let record = |connector: &str| {
        format!(
            r#"{{"version": 1, "workers": [{{"id": "engine-a", "url": "127.0.0.1:{port}", "kv_connector": "{connector}"}}]}}"#
        )
    };
    let connector_of = |app_context: &AppContext| -> Vec<(String, Option<String>)> {
        registrations_at(app_context, port)
            .into_iter()
            .map(|(id, connector, _)| (id, connector))
            .collect()
    };
    let registered = |connector: &str| vec![("engine-a".to_string(), Some(connector.to_string()))];

    write_manifest(&path, &record("MooncakeConnector"));
    let handle = start_file_discovery(&path, &app_context).await;
    let deadline = tokio::time::Instant::now() + CONVERGE;
    while registrations_at(&app_context, port)
        != vec![(
            "engine-a".to_string(),
            Some("MooncakeConnector".to_string()),
            true,
        )]
    {
        assert!(
            tokio::time::Instant::now() < deadline,
            "never became Ready: {:?}",
            registrations_at(&app_context, port)
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    write_manifest(&path, &record("NixlConnector"));
    let deadline = tokio::time::Instant::now() + CONVERGE;
    loop {
        let current = connector_of(&app_context);
        if current == registered("NixlConnector") {
            break;
        }
        assert!(
            current == registered("MooncakeConnector"),
            "the worker left the registry while being updated: {current:?}"
        );
        assert!(
            tokio::time::Instant::now() < deadline,
            "the edit never reached the registry: {current:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // Past the drain window and several rereads, it is still there.
    let settled = tokio::time::Instant::now() + DRAIN + Duration::from_secs(1);
    while tokio::time::Instant::now() < settled {
        assert_eq!(
            connector_of(&app_context),
            registered("NixlConnector"),
            "the updated worker did not stay registered"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    handle.abort();
}

/// A change is read as it happens, not on the next tick: with no tick due for
/// an hour, an atomic replace and an in-place rewrite each apply at once.
#[tokio::test]
async fn a_change_is_read_without_waiting_for_the_reread() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.json");
    let app_context = test_context().await;
    let (_engine_a, port_a) = start_mock_engine().await;
    let (_engine_b, port_b) = start_mock_engine().await;

    write_manifest(&path, &manifest(&[port_a]));
    let handle = start_file_discovery_every(&path, NO_REREAD, &app_context).await;
    wait_for(&app_context, "the startup read registering a", |urls| {
        has_url(urls, port_a)
    })
    .await;

    // A writer renaming a complete file over the manifest.
    write_manifest(&path, &manifest(&[port_a, port_b]));
    wait_for(&app_context, "the renamed manifest adding b", |urls| {
        has_url(urls, port_a) && has_url(urls, port_b)
    })
    .await;

    // An editor writing the manifest in place.
    std::fs::write(&path, manifest(&[port_b])).unwrap();
    wait_for(&app_context, "the rewritten manifest removing a", |urls| {
        !has_url(urls, port_a) && has_url(urls, port_b)
    })
    .await;

    handle.abort();
}

/// A manifest that a writer creates after the gateway started is read when it
/// appears, not a tick later.
#[tokio::test]
async fn a_manifest_created_after_startup_is_read_when_it_appears() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.json");
    let app_context = test_context().await;
    let (_engine, port) = start_mock_engine().await;

    let handle = start_file_discovery_every(&path, NO_REREAD, &app_context).await;
    // Past the startup read, which found no manifest.
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(app_context.worker_registry.get_all_urls().is_empty());

    write_manifest(&path, &manifest(&[port]));
    wait_for(
        &app_context,
        "the new manifest's worker registered",
        |urls| has_url(urls, port),
    )
    .await;

    handle.abort();
}

/// A directory removed and recreated under the manifest, as a deploy tool or
/// a sidecar may do, is watched again: with no tick due for an hour, the
/// manifest written into the new directory applies, and so does a later edit,
/// which only the new watch can report.
#[tokio::test]
async fn a_recreated_directory_is_watched_again() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("smg");
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("workers.json");
    let app_context = test_context().await;
    let (_engine_a, port_a) = start_mock_engine().await;
    let (_engine_b, port_b) = start_mock_engine().await;

    write_manifest(&path, &manifest(&[port_a]));
    let handle = start_file_discovery_every(&path, NO_REREAD, &app_context).await;
    wait_for(&app_context, "the startup read registering a", |urls| {
        has_url(urls, port_a)
    })
    .await;

    std::fs::remove_dir_all(&dir).unwrap();
    // Gone for longer than the settle window, so the provider's first try to
    // watch it again finds nothing, and it keeps retrying.
    tokio::time::sleep(Duration::from_millis(500)).await;
    std::fs::create_dir(&dir).unwrap();
    write_manifest(&path, &manifest(&[port_a, port_b]));
    // Usually written before a retry succeeds, and read by the read that
    // follows it.
    wait_for(
        &app_context,
        "the manifest in the new directory read",
        |urls| has_url(urls, port_a) && has_url(urls, port_b),
    )
    .await;

    // The new watch is in place by now, and with no tick due, it is the only
    // way this edit can arrive.
    write_manifest(&path, &manifest(&[port_b]));
    wait_for(&app_context, "an edit the new watch reports", |urls| {
        !has_url(urls, port_a) && has_url(urls, port_b)
    })
    .await;

    handle.abort();
}

/// The watch sees only the manifest's own directory. A manifest that is a
/// symlink into another directory changes without an event there, and the
/// tick still reads that change.
#[cfg(unix)]
#[tokio::test]
async fn a_change_the_watch_cannot_see_is_read_on_the_next_tick() {
    use std::os::unix::fs::symlink;

    let watched = tempfile::tempdir().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    let target = elsewhere.path().join("workers.json");
    let path = watched.path().join("workers.json");
    symlink(&target, &path).unwrap();
    let app_context = test_context().await;
    let (_engine_a, port_a) = start_mock_engine().await;
    let (_engine_b, port_b) = start_mock_engine().await;

    write_manifest(&target, &manifest(&[port_a]));
    let handle = start_file_discovery(&path, &app_context).await;
    wait_for(&app_context, "a registered through the symlink", |urls| {
        has_url(urls, port_a)
    })
    .await;

    write_manifest(&target, &manifest(&[port_a, port_b]));
    wait_for(
        &app_context,
        "the next tick reading the new target",
        |urls| has_url(urls, port_a) && has_url(urls, port_b),
    )
    .await;

    handle.abort();
}
