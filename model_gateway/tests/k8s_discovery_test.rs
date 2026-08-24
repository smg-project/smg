//! K8s service discovery integration tests against a scripted fake API
//! server: the real watcher/reflector/reconciler, JobQueue, registration
//! workflow, and worker registry run end to end; only the Kubernetes API
//! and the engine servers are test doubles.
#![allow(clippy::unwrap_used, clippy::allow_attributes)]

mod common;

use std::{
    collections::BTreeMap,
    convert::Infallible,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use axum::{
    body::{Body, Bytes},
    extract::{RawQuery, State},
    http::header,
    response::Response,
    routing::any,
    Router,
};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use k8s_openapi::{
    api::core::v1::{Pod, PodCondition, PodStatus},
    apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time},
};
use rustls::crypto::ring;
use smg::{
    app_context::AppContext,
    config::RouterConfig,
    service_discovery::{
        start_service_discovery_with_client, ServiceDiscoveryConfig, POD_UID_LABEL,
    },
    worker::BasicWorkerBuilder,
};
use tokio::sync::broadcast;

// ========== Fake Kubernetes API server ==========

#[derive(Clone)]
struct FakeState {
    pods: Arc<Mutex<BTreeMap<String, Pod>>>,
    resource_version: Arc<AtomicU64>,
    watch_tx: Arc<Mutex<broadcast::Sender<String>>>,
}

/// Minimal pods endpoint: LIST returns the current pod set, WATCH streams
/// scripted line-delimited events; a 410 ERROR event forces a re-LIST.
struct FakeK8s {
    state: FakeState,
    base_url: String,
}

impl FakeK8s {
    #[expect(clippy::disallowed_methods, reason = "test infrastructure")]
    async fn start() -> Self {
        let state = FakeState {
            pods: Arc::new(Mutex::new(BTreeMap::new())),
            resource_version: Arc::new(AtomicU64::new(1)),
            watch_tx: Arc::new(Mutex::new(broadcast::channel(64).0)),
        };
        let app = Router::new()
            .route("/api/v1/pods", any(pods_handler))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            state,
            base_url: format!("http://{addr}"),
        }
    }

    fn client(&self) -> kube::Client {
        let _ = ring::default_provider().install_default();
        let config = kube::Config::new(self.base_url.parse().unwrap());
        kube::Client::try_from(config).unwrap()
    }

    fn apply_pod(&self, pod: Pod) {
        let mut pod = pod;
        pod.metadata.resource_version = Some(self.next_rv());
        let name = pod.metadata.name.clone().unwrap();
        let replaced = self
            .state
            .pods
            .lock()
            .unwrap()
            .insert(name, pod.clone())
            .is_some();
        self.send_event(if replaced { "MODIFIED" } else { "ADDED" }, &pod);
    }

    fn delete_pod(&self, name: &str) {
        let removed = self.state.pods.lock().unwrap().remove(name);
        if let Some(mut pod) = removed {
            pod.metadata.resource_version = Some(self.next_rv());
            self.send_event("DELETED", &pod);
        }
    }

    fn next_rv(&self) -> String {
        (self.state.resource_version.fetch_add(1, Ordering::SeqCst) + 1).to_string()
    }

    /// Remove a pod without emitting a watch event (only visible via re-LIST).
    fn silently_remove_pod(&self, name: &str) {
        self.state.pods.lock().unwrap().remove(name);
        self.state.resource_version.fetch_add(1, Ordering::SeqCst);
    }

    /// Sever every open watch stream without a 410: the watcher re-WATCHes
    /// (no re-LIST). Returns once a new watch has subscribed.
    async fn sever_watch_and_await_reconnect(&self) {
        *self.state.watch_tx.lock().unwrap() = broadcast::channel(64).0;
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if self.state.watch_tx.lock().unwrap().receiver_count() > 0 {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "watcher never reconnected after severed stream"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    /// Expire every open watch: send a 410 Gone ERROR event (the "too old
    /// resource version" desync) and end the streams, forcing a re-LIST.
    fn expire_watch(&self) {
        let status = serde_json::json!({
            "type": "ERROR",
            "object": {
                "kind": "Status",
                "apiVersion": "v1",
                "metadata": {},
                "status": "Failure",
                "message": "too old resource version",
                "reason": "Expired",
                "code": 410,
            },
        });
        let _ = self
            .state
            .watch_tx
            .lock()
            .unwrap()
            .send(format!("{status}\n"));
        *self.state.watch_tx.lock().unwrap() = broadcast::channel(64).0;
    }

    fn send_event(&self, event_type: &str, pod: &Pod) {
        let line = format!(
            "{}\n",
            serde_json::json!({ "type": event_type, "object": pod })
        );
        let _ = self.state.watch_tx.lock().unwrap().send(line);
    }
}

async fn pods_handler(State(state): State<FakeState>, RawQuery(query): RawQuery) -> Response {
    let query = query.unwrap_or_default();
    if query.contains("watch=true") {
        let rx = state.watch_tx.lock().unwrap().subscribe();
        let stream = futures::stream::unfold(rx, |mut rx| async move {
            loop {
                match rx.recv().await {
                    Ok(line) => return Some((Ok::<Bytes, Infallible>(Bytes::from(line)), rx)),
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => return None,
                }
            }
        });
        Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        let items: Vec<Pod> = state.pods.lock().unwrap().values().cloned().collect();
        let list = serde_json::json!({
            "kind": "PodList",
            "apiVersion": "v1",
            "metadata": {
                "resourceVersion": state.resource_version.load(Ordering::SeqCst).to_string()
            },
            "items": items,
        });
        Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(list.to_string()))
            .unwrap()
    }
}

// ========== Test fixtures ==========

fn worker_pod(name: &str, uid: &str, ports: &str, ready: bool) -> Pod {
    let labels: BTreeMap<String, String> = [("app".to_string(), "smg-it".to_string())].into();
    let annotations: BTreeMap<String, String> =
        [("smg.ai/worker-ports".to_string(), ports.to_string())].into();
    Pod {
        metadata: ObjectMeta {
            name: Some(name.to_string()),
            uid: Some(uid.to_string()),
            labels: Some(labels),
            annotations: Some(annotations),
            ..Default::default()
        },
        spec: None,
        status: Some(PodStatus {
            pod_ip: Some("127.0.0.1".to_string()),
            phase: Some("Running".to_string()),
            conditions: Some(vec![PodCondition {
                type_: "Ready".to_string(),
                status: if ready { "True" } else { "False" }.to_string(),
                last_probe_time: None,
                last_transition_time: None,
                message: None,
                reason: None,
                observed_generation: None,
            }]),
            ..Default::default()
        }),
    }
}

fn discovery_config() -> ServiceDiscoveryConfig {
    ServiceDiscoveryConfig {
        enabled: true,
        selector: [("app".to_string(), "smg-it".to_string())].into(),
        check_interval: Duration::from_millis(300),
        ..Default::default()
    }
}

async fn test_context_with_drain(drain_settle_secs: u64) -> Arc<AppContext> {
    let mut config = RouterConfig::builder()
        .worker_startup_timeout_secs(2)
        .build_unchecked();
    config.health_check.drain_settle_secs = drain_settle_secs;
    common::create_test_context(config).await
}

async fn test_context() -> Arc<AppContext> {
    test_context_with_drain(0).await
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

/// Poll until the registry satisfies `predicate` or fail with its contents.
async fn wait_for(
    app_context: &AppContext,
    what: &str,
    timeout: Duration,
    predicate: impl Fn(&[String]) -> bool,
) {
    let deadline = tokio::time::Instant::now() + timeout;
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

// ========== Scenarios ==========

#[tokio::test]
async fn multi_port_pod_registers_and_removes_all_workers() {
    let fake = FakeK8s::start().await;
    let app_context = test_context().await;
    let (_engine_a, port_a) = start_mock_engine().await;
    let (_engine_b, port_b) = start_mock_engine().await;

    fake.apply_pod(worker_pod(
        "multi-0",
        "uid-multi-0",
        &format!("{port_a},{port_b}"),
        true,
    ));

    let handle = start_service_discovery_with_client(
        fake.client(),
        discovery_config(),
        Arc::clone(&app_context),
        None,
        None,
    );

    wait_for(
        &app_context,
        "both engine workers registered",
        Duration::from_secs(15),
        |urls| has_url(urls, port_a) && has_url(urls, port_b),
    )
    .await;

    // Every discovery-created worker carries the pod ownership labels.
    for worker in app_context.worker_registry.get_all() {
        assert_eq!(
            worker.metadata().spec.labels.get(POD_UID_LABEL),
            Some(&"uid-multi-0".to_string()),
            "worker {} missing pod-uid label",
            worker.url()
        );
    }

    fake.delete_pod("multi-0");
    wait_for(
        &app_context,
        "workers removed after pod deletion",
        Duration::from_secs(15),
        |urls| !has_url(urls, port_a) && !has_url(urls, port_b),
    )
    .await;

    handle.abort();
}

#[tokio::test]
async fn terminating_pod_drains_before_final_deletion() {
    let fake = FakeK8s::start().await;
    let app_context = test_context().await;
    let (_engine, port) = start_mock_engine().await;

    fake.apply_pod(worker_pod("term-0", "uid-term-0", &port.to_string(), true));
    let handle = start_service_discovery_with_client(
        fake.client(),
        discovery_config(),
        Arc::clone(&app_context),
        None,
        None,
    );
    wait_for(
        &app_context,
        "worker registered",
        Duration::from_secs(15),
        |urls| has_url(urls, port),
    )
    .await;

    // Graceful termination: deletion_timestamp set, pod object still live and
    // Ready. The worker must drain now, not when the object finally goes.
    let mut terminating = worker_pod("term-0", "uid-term-0", &port.to_string(), true);
    terminating.metadata.deletion_timestamp = Some(Time(k8s_openapi::jiff::Timestamp::now()));
    fake.apply_pod(terminating);

    wait_for(
        &app_context,
        "worker removed at grace-period start",
        Duration::from_secs(15),
        |urls| !has_url(urls, port),
    )
    .await;
    assert!(
        fake.state.pods.lock().unwrap().contains_key("term-0"),
        "pod object should still exist while its worker is already gone"
    );

    handle.abort();
}

#[tokio::test]
async fn zombie_worker_removed_and_manual_worker_kept() {
    let fake = FakeK8s::start().await;
    let app_context = test_context().await;

    // A worker registered for a pod that no longer exists (e.g. by a
    // registration workflow that outlived its pod) — carries the pod-uid
    // label, so the reconciler owns it.
    let zombie = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:19")
            .label(POD_UID_LABEL, "uid-ghost")
            .build(),
    );
    // An operator-added worker without the label — never touched.
    let manual = Arc::new(BasicWorkerBuilder::new("http://127.0.0.1:21").build());
    app_context.worker_registry.register(zombie).unwrap();
    app_context.worker_registry.register(manual).unwrap();

    let handle = start_service_discovery_with_client(
        fake.client(),
        discovery_config(),
        Arc::clone(&app_context),
        None,
        None,
    );

    wait_for(
        &app_context,
        "zombie worker removed",
        Duration::from_secs(15),
        |urls| !has_url(urls, 19),
    )
    .await;

    // Give the reconciler further passes to prove the manual worker survives.
    tokio::time::sleep(Duration::from_millis(900)).await;
    assert!(
        has_url(&app_context.worker_registry.get_all_urls(), 21),
        "manually added worker must never be removed by discovery"
    );

    handle.abort();
}

#[tokio::test]
async fn watch_interruption_relists_and_converges() {
    let fake = FakeK8s::start().await;
    let app_context = test_context().await;
    let (_engine, port) = start_mock_engine().await;

    fake.apply_pod(worker_pod("re-0", "uid-re-0", &port.to_string(), true));
    let handle = start_service_discovery_with_client(
        fake.client(),
        discovery_config(),
        Arc::clone(&app_context),
        None,
        None,
    );
    wait_for(
        &app_context,
        "worker registered",
        Duration::from_secs(15),
        |urls| has_url(urls, port),
    )
    .await;

    // Force-delete while the watch stream is down: the deletion is only
    // observable through the re-LIST the watcher performs after the 410.
    fake.silently_remove_pod("re-0");
    fake.expire_watch();

    wait_for(
        &app_context,
        "worker removed after re-list",
        Duration::from_secs(20),
        |urls| !has_url(urls, port),
    )
    .await;

    handle.abort();
}

#[tokio::test]
async fn drain_window_is_honored_despite_reconcile_passes() {
    use openai_protocol::worker::WorkerStatus;

    let fake = FakeK8s::start().await;
    let app_context = test_context_with_drain(2).await;
    let (_engine, port) = start_mock_engine().await;

    fake.apply_pod(worker_pod(
        "drain-0",
        "uid-drain-0",
        &port.to_string(),
        true,
    ));
    let handle = start_service_discovery_with_client(
        fake.client(),
        discovery_config(),
        Arc::clone(&app_context),
        None,
        None,
    );
    wait_for(
        &app_context,
        "worker registered",
        Duration::from_secs(15),
        |urls| has_url(urls, port),
    )
    .await;

    // The drain path only settles Ready workers; promote if the registration
    // left it in another state (no health-check loop runs in tests).
    let registry = &app_context.worker_registry;
    let worker = registry
        .get_all()
        .into_iter()
        .find(|worker| worker.url().ends_with(&format!(":{port}")))
        .unwrap();
    if worker.status() != WorkerStatus::Ready {
        let worker_id = registry.get_id_by_url(worker.url()).unwrap();
        registry.transition_status_if_revision(&worker_id, worker.revision(), WorkerStatus::Ready);
    }
    assert_eq!(
        registry.get_by_url(worker.url()).unwrap().status(),
        WorkerStatus::Ready,
        "worker must be Ready to exercise the drain path"
    );

    // Ready=False must enter the same drain workflow as deletion. The 300ms
    // check interval guarantees several reconcile passes during the 2s drain;
    // none may cut it short. Anchor on the observed Draining transition so
    // discovery latency does not pad the measurement.
    fake.apply_pod(worker_pod(
        "drain-0",
        "uid-drain-0",
        &port.to_string(),
        false,
    ));
    let url = worker.url().to_string();
    let drain_started = {
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            match registry.get_by_url(&url) {
                Some(current) if current.status() == WorkerStatus::Draining => {
                    break std::time::Instant::now();
                }
                Some(_) => {}
                None => panic!("worker removed without ever draining"),
            }
            assert!(
                std::time::Instant::now() < deadline,
                "worker never started draining"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    wait_for(
        &app_context,
        "worker removed after drain",
        Duration::from_secs(20),
        |urls| !has_url(urls, port),
    )
    .await;
    let drained_for = drain_started.elapsed();
    assert!(
        drained_for >= Duration::from_millis(1800),
        "drain window collapsed: worker removed {drained_for:?} after draining began (settle is 2s)"
    );
    assert!(
        fake.state.pods.lock().unwrap().contains_key("drain-0"),
        "Ready=False must drain workers while the Pod still exists"
    );

    handle.abort();
}

/// Poll until the worker at `port` carries the given pod-uid label.
async fn wait_for_pod_uid(app_context: &AppContext, port: u16, uid: &str, what: &str) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        let found = app_context.worker_registry.get_all().into_iter().any(|w| {
            w.url().ends_with(&format!(":{port}"))
                && w.metadata()
                    .spec
                    .labels
                    .get(POD_UID_LABEL)
                    .map(String::as_str)
                    == Some(uid)
        });
        if found {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {what}; registry: {:?}",
            app_context.worker_registry.get_all_urls()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_ready_pod_uid(app_context: &AppContext, port: u16, uid: &str, what: &str) {
    use openai_protocol::worker::WorkerStatus;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        let found = app_context.worker_registry.get_all().into_iter().any(|w| {
            w.url().ends_with(&format!(":{port}"))
                && w.status() == WorkerStatus::Ready
                && w.metadata()
                    .spec
                    .labels
                    .get(POD_UID_LABEL)
                    .map(String::as_str)
                    == Some(uid)
        });
        if found {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for {what}; registry: {:?}",
            app_context.worker_registry.get_all_urls()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn failed_registration_retries_until_engine_appears() {
    let fake = FakeK8s::start().await;
    let app_context = test_context().await;

    // Reserve a free port, then release it so nothing listens there yet.
    let reserved = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = reserved.local_addr().unwrap().port();
    drop(reserved);

    fake.apply_pod(worker_pod("late-0", "uid-late-0", &port.to_string(), true));
    let handle = start_service_discovery_with_client(
        fake.client(),
        discovery_config(),
        Arc::clone(&app_context),
        None,
        None,
    );

    // Many reconcile passes with the engine down: registration keeps failing
    // and must never leave a phantom worker behind.
    tokio::time::sleep(Duration::from_secs(4)).await;
    assert!(
        !has_url(&app_context.worker_registry.get_all_urls(), port),
        "no worker may exist while the engine is unreachable"
    );

    // Engine comes up on the annotated port: a later pass must register it.
    let mut engine = MockWorker::new(MockWorkerConfig {
        port,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    engine.start().await.unwrap();
    wait_for(
        &app_context,
        "worker registered once the engine came up",
        Duration::from_secs(30),
        |urls| has_url(urls, port),
    )
    .await;

    handle.abort();
}

#[tokio::test]
async fn pod_added_after_start_registers_only_once_ready() {
    let fake = FakeK8s::start().await;
    let app_context = test_context().await;
    let (_engine, port) = start_mock_engine().await;

    // Discovery starts against an empty cluster; everything below arrives
    // through watch events, not the initial LIST.
    let handle = start_service_discovery_with_client(
        fake.client(),
        discovery_config(),
        Arc::clone(&app_context),
        None,
        None,
    );

    fake.apply_pod(worker_pod("flip-0", "uid-flip-0", &port.to_string(), false));
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert!(
        !has_url(&app_context.worker_registry.get_all_urls(), port),
        "unready pod must not register"
    );

    fake.apply_pod(worker_pod("flip-0", "uid-flip-0", &port.to_string(), true));
    wait_for(
        &app_context,
        "worker registered after pod became Ready",
        Duration::from_secs(15),
        |urls| has_url(urls, port),
    )
    .await;

    handle.abort();
}

#[tokio::test]
async fn ready_pod_becoming_unready_removes_all_workers_and_recovers() {
    let fake = FakeK8s::start().await;
    let app_context = test_context().await;
    let (_engine_a, port_a) = start_mock_engine().await;
    let (_engine_b, port_b) = start_mock_engine().await;
    let ports = format!("{port_a},{port_b}");

    fake.apply_pod(worker_pod("ready-flip-0", "uid-ready-flip-0", &ports, true));
    let handle = start_service_discovery_with_client(
        fake.client(),
        discovery_config(),
        Arc::clone(&app_context),
        None,
        None,
    );
    wait_for_ready_pod_uid(
        &app_context,
        port_a,
        "uid-ready-flip-0",
        "first worker routable before readiness change",
    )
    .await;
    wait_for_ready_pod_uid(
        &app_context,
        port_b,
        "uid-ready-flip-0",
        "second worker routable before readiness change",
    )
    .await;

    fake.apply_pod(worker_pod(
        "ready-flip-0",
        "uid-ready-flip-0",
        &ports,
        false,
    ));
    wait_for(
        &app_context,
        "all workers removed while pod remains unready",
        Duration::from_secs(15),
        |urls| !has_url(urls, port_a) && !has_url(urls, port_b),
    )
    .await;
    assert!(
        fake.state.pods.lock().unwrap().contains_key("ready-flip-0"),
        "unready pod should still exist after its workers are removed"
    );

    fake.apply_pod(worker_pod("ready-flip-0", "uid-ready-flip-0", &ports, true));
    wait_for(
        &app_context,
        "all workers re-registered after readiness recovery",
        Duration::from_secs(15),
        |urls| has_url(urls, port_a) && has_url(urls, port_b),
    )
    .await;
    wait_for_ready_pod_uid(
        &app_context,
        port_a,
        "uid-ready-flip-0",
        "first recovered worker routable",
    )
    .await;
    wait_for_ready_pod_uid(
        &app_context,
        port_b,
        "uid-ready-flip-0",
        "second recovered worker routable",
    )
    .await;

    handle.abort();
}

#[tokio::test]
async fn same_url_pod_replacement_reregisters_with_new_uid() {
    let fake = FakeK8s::start().await;
    let app_context = test_context().await;
    let (_engine, port) = start_mock_engine().await;

    fake.apply_pod(worker_pod("stable-0", "uid-gen1", &port.to_string(), true));
    let handle = start_service_discovery_with_client(
        fake.client(),
        discovery_config(),
        Arc::clone(&app_context),
        None,
        None,
    );
    wait_for_pod_uid(
        &app_context,
        port,
        "uid-gen1",
        "first generation registered",
    )
    .await;

    // Stable-IP restart: same name and URL, new uid.
    fake.delete_pod("stable-0");
    fake.apply_pod(worker_pod("stable-0", "uid-gen2", &port.to_string(), true));

    wait_for_pod_uid(&app_context, port, "uid-gen2", "replacement re-registered").await;

    handle.abort();
}

#[tokio::test]
async fn annotation_port_change_reconciles_worker_set() {
    let fake = FakeK8s::start().await;
    let app_context = test_context().await;
    let (_engine_a, port_a) = start_mock_engine().await;
    let (_engine_b, port_b) = start_mock_engine().await;
    let (_engine_c, port_c) = start_mock_engine().await;

    fake.apply_pod(worker_pod(
        "resize-0",
        "uid-resize-0",
        &format!("{port_a},{port_b}"),
        true,
    ));
    let handle = start_service_discovery_with_client(
        fake.client(),
        discovery_config(),
        Arc::clone(&app_context),
        None,
        None,
    );
    wait_for(
        &app_context,
        "initial two workers",
        Duration::from_secs(15),
        |urls| has_url(urls, port_a) && has_url(urls, port_b),
    )
    .await;

    // Shrink then grow the annotated port set on the live pod.
    fake.apply_pod(worker_pod(
        "resize-0",
        "uid-resize-0",
        &port_a.to_string(),
        true,
    ));
    wait_for(
        &app_context,
        "removed worker for dropped port",
        Duration::from_secs(15),
        |urls| has_url(urls, port_a) && !has_url(urls, port_b),
    )
    .await;

    fake.apply_pod(worker_pod(
        "resize-0",
        "uid-resize-0",
        &format!("{port_a},{port_c}"),
        true,
    ));
    wait_for(
        &app_context,
        "added worker for new port",
        Duration::from_secs(15),
        |urls| has_url(urls, port_a) && has_url(urls, port_c) && !has_url(urls, port_b),
    )
    .await;

    handle.abort();
}

#[tokio::test]
async fn severed_watch_reconnects_and_converges_without_relist() {
    let fake = FakeK8s::start().await;
    let app_context = test_context().await;
    let (_engine, port) = start_mock_engine().await;

    fake.apply_pod(worker_pod("re2-0", "uid-re2-0", &port.to_string(), true));
    let handle = start_service_discovery_with_client(
        fake.client(),
        discovery_config(),
        Arc::clone(&app_context),
        None,
        None,
    );
    wait_for(
        &app_context,
        "worker registered",
        Duration::from_secs(15),
        |urls| has_url(urls, port),
    )
    .await;

    // Drop the stream mid-watch (no 410): the watcher must reconnect and
    // process events delivered on the new stream.
    fake.sever_watch_and_await_reconnect().await;
    fake.delete_pod("re2-0");

    wait_for(
        &app_context,
        "worker removed after reconnect",
        Duration::from_secs(20),
        |urls| !has_url(urls, port),
    )
    .await;

    handle.abort();
}

#[tokio::test]
async fn router_discovery_updates_mesh_cluster_state() {
    use smg_mesh::gossip::NodeStatus;

    let fake = FakeK8s::start().await;
    let app_context = test_context().await;

    let mut config = discovery_config();
    config
        .router_selector
        .insert("role".to_string(), "router".to_string());
    let cluster_state: smg_mesh::ClusterState = Arc::default();

    let mut router = worker_pod("r-0", "uid-r-0", "1", true);
    router.metadata.labels = Some([("role".to_string(), "router".to_string())].into());
    router.metadata.annotations = Some(
        [("sglang.ai/mesh-port".to_string(), "7100".to_string())]
            .into_iter()
            .collect(),
    );
    fake.apply_pod(router.clone());

    let handle = start_service_discovery_with_client(
        fake.client(),
        config,
        Arc::clone(&app_context),
        Some(cluster_state.clone()),
        Some(7000),
    );

    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        let alive = cluster_state
            .read()
            .get("r-0")
            .map(|node| (node.status, node.address.clone()));
        if let Some((status, address)) = alive {
            assert_eq!(address, "127.0.0.1:7100");
            if status == NodeStatus::Alive as i32 {
                break;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "router node never became Alive: {:?}",
            cluster_state.read().get("r-0")
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    fake.delete_pod("r-0");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        if cluster_state.read().get("r-0").map(|node| node.status) == Some(NodeStatus::Down as i32)
        {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "router node never marked Down: {:?}",
            cluster_state.read().get("r-0")
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    handle.abort();
}
