use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
};

/// The re-read gate: after a failure always, while pending only without
/// an id, while Ready only until a re-read has confirmed the id, never
/// while draining.
#[test]
fn the_engine_id_is_re_read_after_a_failure_while_pending_without_one_or_until_confirmed() {
    let worker = |kv_engine_id: Option<&str>| -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new("grpc://p:1")
            .worker_type(WorkerType::Prefill)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Vllm)
            .kv_connector("MooncakeConnector");
        if let Some(id) = kv_engine_id {
            builder = builder.kv_engine_id(id);
        }
        Arc::new(builder.build())
    };
    let with_id = worker(Some("eng"));
    let without_id = worker(None);
    let unconfirmed = worker(Some("eng"));
    unconfirmed.set_kv_engine_id_confirmed(false);
    let cases = [
        (WorkerStatus::Failed, &with_id, true),
        (WorkerStatus::NotReady, &with_id, true),
        (WorkerStatus::Pending, &with_id, false),
        (WorkerStatus::Pending, &without_id, true),
        (WorkerStatus::Ready, &with_id, false),
        (WorkerStatus::Ready, &without_id, false),
        (WorkerStatus::Ready, &unconfirmed, true),
        (WorkerStatus::Draining, &unconfirmed, false),
    ];
    for (status, worker, expected) in cases {
        assert_eq!(
            engine_id_may_be_stale(status, worker.as_ref()),
            expected,
            "{status:?} with id {:?}",
            worker.kv_engine_id()
        );
    }
}

/// A recovered worker whose engine accepts the connection but never
/// answers the metadata read must not hold the probe slot: the read
/// expires and the previous engine id stays (#2491).
#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "the held connections live for the duration of the test"
)]
async fn a_hanging_engine_id_read_expires_and_keeps_the_previous_id() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a loopback listener");
    let port = listener.local_addr().expect("listener address").port();
    // Accept every connection and never answer on it.
    let hold = tokio::spawn(async move {
        let mut held = Vec::new();
        loop {
            if let Ok((socket, _)) = listener.accept().await {
                held.push(socket);
            }
        }
    });
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Prefill)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Vllm)
            .kv_connector("MooncakeConnector")
            .kv_engine_id("eng-old")
            .build(),
    );

    let started = std::time::Instant::now();
    refresh_kv_engine_id_after_recovery(&worker, Duration::from_millis(300)).await;

    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the re-read must be bounded by the timeout"
    );
    assert_eq!(worker.kv_engine_id().as_deref(), Some("eng-old"));
    // Not confirmed: a later Ready probe re-reads instead of giving up.
    assert!(!worker.kv_engine_id_confirmed());
    assert!(engine_id_may_be_stale(WorkerStatus::Ready, worker.as_ref()));
    hold.abort();
}

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use http::{header::AUTHORIZATION, HeaderMap};
use openai_protocol::{
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerStatus},
};

use super::*;
use crate::worker::{
    circuit_breaker::CircuitState, test_support::RefusedAddress, BasicWorkerBuilder,
    ConnectionMode, RuntimeType, Worker, WorkerError, WorkerRegistry, WorkerType,
};

fn make_worker(url: &str, success_threshold: u32, failure_threshold: u32) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .health_config(HealthCheckConfig {
                success_threshold,
                failure_threshold,
                timeout_secs: 1,
                check_interval_secs: 1,
                disable_health_check: false,
                drain_settle_secs: 0,
            })
            .build(),
    )
}

fn test_config() -> WorkerManagerConfig {
    WorkerManagerConfig {
        default_check_interval_secs: 5,
        remove_unhealthy: UnhealthyRemovalScope::Disabled,
    }
}

/// `test_config()` removing the unhealthy workers of `scope`.
fn config_removing(scope: UnhealthyRemovalScope) -> WorkerManagerConfig {
    WorkerManagerConfig {
        remove_unhealthy: scope,
        ..test_config()
    }
}

/// A registry holding one worker whose probes have failed, and that worker's
/// id: what a bootstrap reconcile finds after a restart.
fn registry_with_a_failed_worker() -> (Arc<WorkerRegistry>, WorkerId) {
    let registry = Arc::new(WorkerRegistry::new());
    let failed_worker = make_worker("http://failed:1", 2, 3);
    failed_worker.set_status(WorkerStatus::Failed);
    let failed_id = registry.register(failed_worker).unwrap();
    (registry, failed_id)
}

/// The completion of a passing probe of `worker`, launched while the worker
/// was not ready: the result that brings a worker into service.
fn passing_probe(worker_id: WorkerId, worker: &Arc<dyn Worker>) -> ProbeCompletion {
    ProbeCompletion {
        worker_id,
        worker: Arc::clone(worker),
        expected_revision: worker.revision(),
        launched_status: WorkerStatus::NotReady,
        probe_result: Ok(()),
    }
}

/// The probes promote a demoted worker through `apply_probe_completion`,
/// not through `transition_status`: that path has to put the worker's
/// open circuit breaker on trial too, or a restarted engine is Ready and
/// still unroutable until the breaker's timeout.
#[tokio::test]
#[tracing_test::traced_test]
async fn a_passing_probe_run_readmits_a_demoted_worker_with_its_breaker_on_trial() {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = make_worker("http://w:1", 1, 1);
    let worker_id = registry.register(worker.clone()).unwrap();
    worker.set_status(WorkerStatus::Ready);

    // The engine died: its requests failed until the breaker opened, and
    // the probes demoted it.
    for _ in 0..8 {
        worker.record_circuit_breaker_outcome(false);
    }
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    worker.set_status(WorkerStatus::NotReady);
    assert!(!worker.is_available());

    // The engine is back on the same address: the first passing probe
    // (success threshold 1) promotes the worker.
    let completion = passing_probe(worker_id, &worker);
    let applied = apply_probe_completion(&registry, completion, None, None).await;
    assert!(matches!(
        applied,
        ProbeApplyResult::Applied(Some((WorkerStatus::NotReady, WorkerStatus::Ready)))
    ));
    // The transition is the INFO line an outage is read from.
    assert!(logs_contain("Worker status transition"));
    assert!(logs_contain("new=Ready"));
    assert_eq!(worker.status(), WorkerStatus::Ready);
    assert_eq!(
        worker.circuit_breaker_state(),
        CircuitState::HalfOpen,
        "the promotion puts the open breaker on trial instead of leaving it to its timeout"
    );
    assert!(
        worker.is_available(),
        "a worker promoted by its probes is routable at once, its breaker on trial"
    );
}

/// Labels follow the engine for a worker this node registered itself over
/// a transport with metadata endpoints; a mesh import carries its owner's
/// labels and a ZMQ engine reports its geometry at the handshake.
fn discovered_worker(url: &str) -> Arc<dyn Worker> {
    discovered_worker_with(
        url,
        HealthCheckConfig {
            success_threshold: 1,
            failure_threshold: 1,
            timeout_secs: 1,
            check_interval_secs: 1,
            disable_health_check: false,
            drain_settle_secs: 0,
        },
    )
}

fn discovered_worker_with(url: &str, health: HealthCheckConfig) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .health_config(health)
            .labels(HashMap::from([(
                "max_num_seqs".to_string(),
                "256".to_string(),
            )]))
            .discovered_labels(HashMap::from([(
                "max_num_seqs".to_string(),
                "256".to_string(),
            )]))
            .build(),
    )
}

#[test]
fn labels_follow_the_engine_for_local_self_hosted_workers_only() {
    let registry = Arc::new(WorkerRegistry::new());
    let local = discovered_worker("http://local:1");
    let local_id = registry.register(local.clone()).unwrap();
    assert!(labels_follow_the_engine(&registry, &local_id, &local));

    // Registration discovered nothing (its read failed, or the engine
    // offers no metadata): the labels follow once the engine answers.
    let undiscovered = make_worker("http://local:2", 1, 1);
    let undiscovered_id = registry.register(undiscovered.clone()).unwrap();
    assert!(labels_follow_the_engine(
        &registry,
        &undiscovered_id,
        &undiscovered
    ));

    // An external provider is never read for engine metadata.
    let external: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("https://provider.example")
            .runtime_type(RuntimeType::External)
            .build(),
    );
    let external_id = registry.register(external.clone()).unwrap();
    assert!(!labels_follow_the_engine(
        &registry,
        &external_id,
        &external
    ));

    // A ZMQ engine reports its geometry at the handshake: the worker's own
    // link holds it, and the refresh step reads the link.
    let zmq: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("ipc:///tmp/engine.sock")
            .connection_mode(ConnectionMode::Zmq)
            .build(),
    );
    let zmq_id = registry.register(zmq.clone()).unwrap();
    assert!(labels_follow_the_engine(&registry, &zmq_id, &zmq));

    registry.on_remote_worker_state(&smg_mesh::WorkerState {
        worker_id: "peer-w1".to_string(),
        model_id: "m".to_string(),
        url: "http://remote:1".to_string(),
        health: true,
        load: 0.0,
        version: 1,
        spec: vec![],
        owner: "peer".to_string(),
    });
    let mesh_id = registry.get_id_by_url("http://remote:1").unwrap();
    let mesh = registry.get(&mesh_id).unwrap();
    assert!(!labels_follow_the_engine(&registry, &mesh_id, &mesh));
}

/// A transition into Ready submits a label refresh for the worker's URL,
/// with no removal queue (the shape of `remove_unhealthy = false`).
#[tokio::test]
async fn a_transition_into_ready_submits_a_label_refresh() {
    use crate::{service_discovery::testing::create_test_app_context, workflow::JobQueueConfig};
    let app_context = create_test_app_context();
    let registry = Arc::clone(&app_context.worker_registry);
    let job_queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&app_context));

    let worker = discovered_worker("http://local:7");
    worker.set_status(WorkerStatus::NotReady);
    let worker_id = registry.register(worker.clone()).unwrap();
    let completion = passing_probe(worker_id, &worker);
    let applied = apply_probe_completion(&registry, completion, None, Some(&job_queue)).await;
    assert!(matches!(
        applied,
        ProbeApplyResult::Applied(Some((WorkerStatus::NotReady, WorkerStatus::Ready)))
    ));
    assert!(
        job_queue.get_status("http://local:7").is_some(),
        "a refresh job was submitted for the worker that came into service"
    );
    assert_eq!(worker.status(), WorkerStatus::Ready);
    assert!(
        worker.is_available(),
        "a worker promoted by its probes is routable at once"
    );
}

/// Registration's read can fail while an engine is still loading, and the
/// step registers the worker with no labels rather than failing; the first
/// transition into service is then what fills them in.
#[tokio::test]
async fn a_worker_registered_without_labels_is_refreshed_when_it_comes_into_service() {
    use crate::{service_discovery::testing::create_test_app_context, workflow::JobQueueConfig};
    let app_context = create_test_app_context();
    let registry = Arc::clone(&app_context.worker_registry);
    let job_queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&app_context));

    let worker = make_worker("http://local:8", 1, 1);
    assert!(worker.metadata().discovered_labels.is_empty());
    worker.set_status(WorkerStatus::NotReady);
    let worker_id = registry.register(worker.clone()).unwrap();
    let completion = passing_probe(worker_id, &worker);
    let applied = apply_probe_completion(&registry, completion, None, Some(&job_queue)).await;
    assert!(matches!(
        applied,
        ProbeApplyResult::Applied(Some((WorkerStatus::NotReady, WorkerStatus::Ready)))
    ));
    assert!(
        job_queue.get_status("http://local:8").is_some(),
        "a worker whose registration discovered nothing is refreshed all the same"
    );
}

fn make_zmq_worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Zmq)
            .build(),
    )
}

/// A worker with a health probe on the given wire, an hour between
/// probes: the probes a test sees are the ones it brings forward.
fn probed_worker(url: &str, connection_mode: ConnectionMode) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(connection_mode)
            .health_config(HealthCheckConfig {
                success_threshold: 1,
                failure_threshold: 1,
                timeout_secs: 1,
                check_interval_secs: 3600,
                disable_health_check: false,
                drain_settle_secs: 0,
            })
            .build(),
    )
}

/// A probed worker promoted on `success_threshold` reads, probed every
/// `check_interval_secs`, with a one-second probe timeout.
fn probed_worker_with(
    url: &str,
    connection_mode: ConnectionMode,
    success_threshold: u32,
    check_interval_secs: u64,
) -> Arc<dyn Worker> {
    probed_worker_with_timeout(
        url,
        connection_mode,
        success_threshold,
        check_interval_secs,
        1,
    )
}

fn probed_worker_with_timeout(
    url: &str,
    connection_mode: ConnectionMode,
    success_threshold: u32,
    check_interval_secs: u64,
    timeout_secs: u64,
) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(connection_mode)
            .health_config(HealthCheckConfig {
                success_threshold,
                failure_threshold: 1,
                timeout_secs,
                check_interval_secs,
                disable_health_check: false,
                drain_settle_secs: 0,
            })
            .build(),
    )
}

/// A worker whose health checks are off: its contacts are all the
/// gateway will hear from it.
fn unprobed_worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..HealthCheckConfig::default()
            })
            .build(),
    )
}

/// A `/health` that answers 200 while `serving` is set and 503 otherwise,
/// counting its hits; returns its base URL.
async fn start_health_stub(serving: Arc<AtomicBool>, hits: Arc<AtomicUsize>) -> String {
    let app = axum::Router::new().route(
        "/health",
        axum::routing::get(move || {
            let serving = serving.clone();
            let hits = hits.clone();
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                if serving.load(Ordering::SeqCst) {
                    StatusCode::OK
                } else {
                    StatusCode::SERVICE_UNAVAILABLE
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind stub");
    let addr = listener.local_addr().expect("stub address");
    #[expect(
        clippy::disallowed_methods,
        reason = "test stub server lives for the duration of the test process"
    )]
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serve");
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn maybe_start_skips_when_health_disabled_and_no_pending_zmq() {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = make_worker("http://w:1", 2, 3);
    worker.set_status(WorkerStatus::Ready);
    registry.register(worker).unwrap();

    let manager = WorkerManager::maybe_start(
        registry,
        WorkerManagerConfig {
            default_check_interval_secs: 3600,
            remove_unhealthy: UnhealthyRemovalScope::Disabled,
        },
        None,
        false,
        false,
    );
    assert!(
        manager.is_none(),
        "no health polling and no connect-signal consumers needed"
    );
}

#[tokio::test]
async fn maybe_start_runs_when_health_disabled_but_zmq_pending() {
    let registry = Arc::new(WorkerRegistry::new());
    let zmq = make_zmq_worker("ipc:///tmp/w.ipc");
    assert_eq!(zmq.status(), WorkerStatus::Pending);
    registry.register(zmq).unwrap();

    let mut manager = WorkerManager::maybe_start(
        registry,
        WorkerManagerConfig {
            default_check_interval_secs: 3600,
            remove_unhealthy: UnhealthyRemovalScope::Disabled,
        },
        None,
        false,
        false,
    )
    .expect("manager must run to consume the connect signal for the pending ZMQ worker");
    manager.shutdown().await;
}

#[tokio::test]
async fn maybe_start_runs_when_health_enabled() {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = make_worker("http://w:1", 2, 3);
    worker.set_status(WorkerStatus::Ready);
    registry.register(worker).unwrap();

    let mut manager = WorkerManager::maybe_start(
        registry,
        WorkerManagerConfig {
            default_check_interval_secs: 3600,
            remove_unhealthy: UnhealthyRemovalScope::Disabled,
        },
        None,
        true,
        false,
    )
    .expect("manager must run when health checks are enabled");
    manager.shutdown().await;
}

#[tokio::test]
async fn maybe_start_runs_when_zmq_transport_enabled_even_with_empty_registry() {
    // Config workers register in the background after startup, so the
    // registry is empty when maybe_start runs. With ZMQ configured the
    // manager must still start — otherwise those late ZMQ workers would
    // fire the connect signal at a manager that never took the receiver.
    let registry = Arc::new(WorkerRegistry::new());
    let mut manager = WorkerManager::maybe_start(
        registry,
        WorkerManagerConfig {
            default_check_interval_secs: 3600,
            remove_unhealthy: UnhealthyRemovalScope::Disabled,
        },
        None,
        false,
        true,
    )
    .expect("manager must run when the ZMQ transport is configured");
    manager.shutdown().await;
}

#[tokio::test]
async fn manager_loop_promotes_zmq_worker_on_connect_signal() {
    // End-to-end promotion through the running loop: the signal travels the
    // registry sender -> manager select! -> apply_connect_signal, flipping
    // the pending ZMQ worker to Ready. The worker starts Pending and its
    // health probe (against a socket with no backend) can only fail, so the
    // connect signal is the sole path to Ready under test.
    let registry = Arc::new(WorkerRegistry::new());
    let zmq = make_zmq_worker("ipc:///tmp/connect.ipc");
    let revision = zmq.revision();
    registry.register(zmq.clone()).unwrap();
    assert_eq!(zmq.status(), WorkerStatus::Pending);

    let mut manager = WorkerManager::maybe_start(
        registry.clone(),
        WorkerManagerConfig {
            default_check_interval_secs: 3600,
            remove_unhealthy: UnhealthyRemovalScope::Disabled,
        },
        None,
        false,
        true,
    )
    .expect("manager must run to consume the connect signal");

    registry
        .connect_signal_sender()
        .send(WorkerConnected {
            url: "ipc:///tmp/connect.ipc".to_string(),
            revision,
        })
        .expect("connect signal must reach the manager loop");

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while zmq.status() != WorkerStatus::Ready {
        assert!(
            tokio::time::Instant::now() < deadline,
            "manager loop did not promote the ZMQ worker in time"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    manager.shutdown().await;
}

fn make_dp_worker(base: &str, rank: usize, size: usize) -> Arc<dyn Worker> {
    let spec: openai_protocol::worker::WorkerSpec = serde_json::from_value(serde_json::json!({
        "url": format!("{base}@{rank}"),
        "dp_base_url": base,
        "dp_rank": rank,
        "dp_size": size,
    }))
    .expect("dp worker spec");
    Arc::new(BasicWorkerBuilder::from_spec(spec).build())
}

/// The one decision behind every slot: a Failed worker that a removal
/// will take has no slot whatever its health settings, one the removal
/// skips keeps its probe, a worker whose checks are off or whose drain is
/// the removal workflow's is never probed, everyone else is probed at its
/// own interval or the default. Whether a removal takes a worker
/// (`removable`) is the removal scope's word: off, a mesh import, or a
/// static worker under the discovery default all come to `false`.
#[test]
fn the_probe_slot_follows_the_status_and_the_health_settings() {
    let config = config_removing(UnhealthyRemovalScope::AllLocal);
    let probe = |secs| ProbeSlot::Probe(Duration::from_secs(secs));
    // (status, checks disabled, own interval, removable, slot)
    let cases = [
        (WorkerStatus::Pending, false, 0, true, probe(5)),
        (WorkerStatus::Pending, false, 7, true, probe(7)),
        (WorkerStatus::Ready, false, 7, true, probe(7)),
        (WorkerStatus::NotReady, false, 7, true, probe(7)),
        (WorkerStatus::Failed, false, 7, true, ProbeSlot::Removal),
        (WorkerStatus::Failed, true, 7, true, ProbeSlot::Removal),
        (WorkerStatus::Failed, false, 7, false, probe(7)),
        (WorkerStatus::Ready, true, 7, true, ProbeSlot::Unprobed),
        (WorkerStatus::Draining, false, 7, true, ProbeSlot::Unprobed),
        (WorkerStatus::Draining, false, 7, false, ProbeSlot::Unprobed),
    ];
    for (status, disabled, interval, removable, expected) in cases {
        assert_eq!(
            ProbeSlot::of(status, disabled, interval, &config, removable),
            expected,
            "{status:?} disabled={disabled} interval={interval} removable={removable}"
        );
    }
}

/// A mesh import that the local probes took to Failed is owner-managed:
/// `--remove-unhealthy-workers` never removes it, so it must keep its
/// probe slot (the reconcile and the due-probe pass alike), take no
/// removal, and answer its connect signal with a probe now, or it stays
/// Failed and unroutable on this node until its owner re-publishes it.
#[tokio::test]
async fn a_failed_mesh_import_under_removal_keeps_its_probe_slot() {
    let registry = Arc::new(WorkerRegistry::new());
    registry.on_remote_worker_state(&smg_mesh::WorkerState {
        worker_id: "peer-w1".to_string(),
        model_id: "m".to_string(),
        url: "http://remote:1".to_string(),
        health: true,
        load: 0.0,
        version: 1,
        spec: vec![],
        owner: "peer".to_string(),
    });
    let worker_id = registry.get_id_by_url("http://remote:1").unwrap();
    let worker = registry.get(&worker_id).unwrap();
    worker.set_status(WorkerStatus::Failed);
    let config = config_removing(UnhealthyRemovalScope::AllLocal);

    let mut next_check = HashMap::new();
    reconcile_from_registry(&registry, &mut next_check, &config);
    assert!(
        next_check.contains_key(&worker_id),
        "the reconcile keeps a Failed import on the schedule"
    );

    next_check.insert(worker_id.clone(), tokio::time::Instant::now());
    let mut in_flight = HashSet::new();
    let mut probes: ProbeFutures = FuturesUnordered::new();
    let removals = queue_due_probes(
        &registry,
        &config,
        &mut next_check,
        &mut in_flight,
        &mut probes,
        tokio::time::Instant::now(),
    );
    assert!(
        removals.is_empty(),
        "no removal: the import is owner-managed"
    );
    assert!(
        in_flight.contains(&worker_id) && next_check.contains_key(&worker_id),
        "the due pass probes the import and re-arms its slot"
    );

    let before = tokio::time::Instant::now();
    apply_connect_signal(
        &registry,
        WorkerConnected {
            url: "http://remote:1".to_string(),
            revision: worker.revision(),
        },
        &mut next_check,
        &config,
    );
    assert_eq!(
        registry.get(&worker_id).unwrap().status(),
        WorkerStatus::Failed,
        "a contact is not a health read: the import stays Failed until its probe passes"
    );
    let due = next_check
        .get(&worker_id)
        .copied()
        .expect("the import keeps its slot");
    assert!(
        due >= before && due <= tokio::time::Instant::now(),
        "the contact is not a last poll: the import's probe runs now"
    );
}

#[test]
fn apply_connect_signal_promotes_pending_worker() {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = make_zmq_worker("ipc:///tmp/w.ipc");
    assert_eq!(worker.status(), WorkerStatus::Pending);
    let revision = worker.revision();
    registry.register(worker.clone()).unwrap();

    let mut next_check = HashMap::new();
    apply_connect_signal(
        &registry,
        WorkerConnected {
            url: "ipc:///tmp/w.ipc".to_string(),
            revision,
        },
        &mut next_check,
        &test_config(),
    );

    assert_eq!(
        worker.status(),
        WorkerStatus::Ready,
        "a matching handshake must promote a Pending ZMQ worker immediately"
    );
}

/// A worker with a health probe is not promoted by its contact: a
/// servicer answers its load poll and relays KV events through a warm-up
/// whose health it holds at not-serving, and a worker promoted on that
/// contact took requests before its engine served (forty seconds before,
/// on the bed the defect was found on). The signal moves the probe to
/// now and leaves the status to it, on every wire with a health probe.
#[test]
fn a_contact_from_a_demoted_probed_worker_moves_its_probe_to_now_and_promotes_nothing() {
    for (url, connection_mode, demoted) in [
        ("grpc://w:1", ConnectionMode::Grpc, WorkerStatus::Failed),
        ("grpc://w:1", ConnectionMode::Grpc, WorkerStatus::NotReady),
        ("http://w:1", ConnectionMode::Http, WorkerStatus::NotReady),
    ] {
        let registry = Arc::new(WorkerRegistry::new());
        let worker = probed_worker(url, connection_mode);
        worker.set_status(demoted);
        let revision = worker.revision();
        let worker_id = registry.register(worker.clone()).unwrap();

        let mut next_check = HashMap::new();
        let before = tokio::time::Instant::now();
        let moved = apply_connect_signal(
            &registry,
            WorkerConnected {
                url: url.to_string(),
                revision,
            },
            &mut next_check,
            &test_config(),
        );

        assert!(
            moved.is_none(),
            "{connection_mode:?}: nothing moves into service on a contact"
        );
        assert_eq!(
            worker.status(),
            demoted,
            "{connection_mode:?}: the status is the probe's to change"
        );
        let due = next_check
            .get(&worker_id)
            .copied()
            .expect("the probe is on the schedule");
        assert!(
            due >= before && due <= tokio::time::Instant::now(),
            "{connection_mode:?}: the probe is due now, not after its interval"
        );
    }
}

/// A signal from a replaced worker moves nothing: not the replacement's
/// status and not its probe.
#[test]
fn a_stale_contact_moves_no_probe_of_the_replacement() {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = probed_worker("http://w:1", ConnectionMode::Http);
    worker.set_status(WorkerStatus::NotReady);
    let stale_revision = worker.revision();
    let worker_id = registry.register(worker).unwrap();
    let replacement = probed_worker("http://w:1", ConnectionMode::Http);
    assert!(registry.replace(&worker_id, replacement));

    let mut next_check = HashMap::new();
    apply_connect_signal(
        &registry,
        WorkerConnected {
            url: "http://w:1".to_string(),
            revision: stale_revision,
        },
        &mut next_check,
        &test_config(),
    );

    assert!(
        next_check.is_empty(),
        "a stale signal schedules no probe of the replacement"
    );
    assert_eq!(
        registry.get(&worker_id).unwrap().status(),
        WorkerStatus::NotReady,
        "the replacement keeps the status it inherited: nothing promoted it"
    );
}

/// The signal promotes outright where it is all the gateway will hear: a
/// worker whose health checks are off has nothing to read.
#[test]
fn a_contact_promotes_a_worker_without_health_checks_outright() {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = unprobed_worker("http://u:1");
    worker.set_status(WorkerStatus::NotReady);
    let revision = worker.revision();
    let worker_id = registry.register(worker.clone()).unwrap();

    let mut next_check = HashMap::new();
    let moved = apply_connect_signal(
        &registry,
        WorkerConnected {
            url: "http://u:1".to_string(),
            revision,
        },
        &mut next_check,
        &test_config(),
    );

    assert!(moved.is_some_and(|(id, _)| id == worker_id));
    assert_eq!(worker.status(), WorkerStatus::Ready);
    assert!(
        next_check.is_empty(),
        "a worker without health checks holds no probe slot"
    );
}

/// Starts the manager over `registry` with an hour's default interval.
fn manager_over(registry: &Arc<WorkerRegistry>) -> WorkerManager {
    WorkerManager::maybe_start(
        registry.clone(),
        WorkerManagerConfig {
            default_check_interval_secs: 3600,
            remove_unhealthy: UnhealthyRemovalScope::Disabled,
        },
        None,
        true,
        false,
    )
    .expect("health checks on: the manager runs")
}

/// Polls every 10 ms until `done` holds or `deadline` passes.
async fn wait_until(deadline: tokio::time::Instant, what: &str, done: impl Fn() -> bool) {
    while !done() {
        assert!(tokio::time::Instant::now() < deadline, "{what}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Registers `worker` demoted under the running manager and waits for the
/// registration's probe to read the health stub (`hits`), which is held:
/// the worker is still demoted once that read has landed.
async fn registered_and_held(
    registry: &WorkerRegistry,
    worker: &Arc<dyn Worker>,
    hits: &AtomicUsize,
) {
    worker.set_status(WorkerStatus::NotReady);
    registry.register(worker.clone()).unwrap();
    let registered = tokio::time::Instant::now();
    wait_until(
        registered + Duration::from_secs(5),
        "the registration's probe did not read the health in time",
        || hits.load(Ordering::SeqCst) > 0,
    )
    .await;
    assert_eq!(worker.status(), WorkerStatus::NotReady);
}

/// Through the running loop, against a `/health` that is held: the probe
/// the contact brings forward reads the hold and the worker stays
/// demoted; the contact's probe re-arms the deadline an interval out: no
/// further probe without a signal. Once the health answers, the scheduled
/// probe promotes it.
#[tokio::test]
async fn a_demoted_worker_stays_out_on_its_contacts_read_and_returns_on_the_scheduled_one() {
    let serving = Arc::new(AtomicBool::new(false));
    let hits = Arc::new(AtomicUsize::new(0));
    let url = start_health_stub(serving.clone(), hits.clone()).await;
    let registry = Arc::new(WorkerRegistry::new());
    // Two seconds between probes: the first probe this test sees is the
    // one the signal brings forward, the second the scheduled one.
    let worker = probed_worker_with(&url, ConnectionMode::Http, 1, 2);
    worker.set_status(WorkerStatus::NotReady);
    let revision = worker.revision();
    registry.register(worker.clone()).unwrap();
    let mut manager = manager_over(&registry);

    let contact = tokio::time::Instant::now();
    registry
        .connect_signal_sender()
        .send(WorkerConnected {
            url: url.clone(),
            revision,
        })
        .unwrap();
    wait_until(
        contact + Duration::from_secs(5),
        "the contact's probe did not read the health in time",
        || hits.load(Ordering::SeqCst) > 0,
    )
    .await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        worker.status(),
        WorkerStatus::NotReady,
        "a held health keeps the worker demoted whatever its transport says"
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "the contact's probe re-arms the deadline an interval out: no further probe without a signal"
    );

    serving.store(true, Ordering::SeqCst);
    wait_until(
        contact + Duration::from_secs(6),
        "the scheduled probe did not promote the worker in time",
        || worker.status() == WorkerStatus::Ready,
    )
    .await;
    assert_eq!(
        hits.load(Ordering::SeqCst),
        2,
        "the hold and the scheduled pass, with no second signal"
    );
    assert!(
        tokio::time::Instant::now() >= contact + Duration::from_secs(2),
        "the promoting probe was the scheduled one, an interval after the launch"
    );

    manager.shutdown().await;
}

/// A demoted worker whose health answers again returns on consecutive
/// reads, not consecutive intervals. Through the running loop, a
/// threshold of two reads, three seconds between probes and a one-second
/// probe timeout: the stub is held until the registration's probe has
/// read it, so the next probe is an interval away; it passes, and the
/// read that promotes the worker follows a probe's timeout later (three
/// reads, one interval and a second) instead of an interval later (three
/// reads, two intervals).
#[tokio::test]
async fn a_demoted_worker_returns_on_consecutive_reads_not_consecutive_intervals() {
    let serving = Arc::new(AtomicBool::new(false));
    let hits = Arc::new(AtomicUsize::new(0));
    let url = start_health_stub(serving.clone(), hits.clone()).await;
    let registry = Arc::new(WorkerRegistry::new());
    let mut manager = manager_over(&registry);

    // Registered under the running manager: probed at once, and held.
    let worker = probed_worker_with(&url, ConnectionMode::Http, 2, 3);
    registered_and_held(&registry, &worker, &hits).await;

    // The engine is back; the next probe is a whole interval away.
    serving.store(true, Ordering::SeqCst);
    let back = tokio::time::Instant::now();
    wait_until(
        back + Duration::from_millis(5500),
        "the worker did not return within one interval and a probe timeout",
        || worker.status() == WorkerStatus::Ready,
    )
    .await;
    let returned = tokio::time::Instant::now();
    assert!(
        returned >= back + Duration::from_millis(3500),
        "the first passing read was the scheduled one, an interval after the launch, and the \
         promoting read a probe timeout after it"
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        3,
        "the held read, the scheduled pass and the pass brought forward"
    );

    manager.shutdown().await;
}

/// The read a pass brings forward never lands after the one the schedule
/// already holds. With two seconds between probes and a five-second probe
/// timeout, a demoted worker's promoting read follows the interval, not
/// the timeout: three reads over two intervals (the held read, the
/// scheduled pass, the pass an interval later), not two intervals and a
/// timeout.
#[tokio::test]
async fn a_brought_forward_read_never_lands_after_the_scheduled_one() {
    let serving = Arc::new(AtomicBool::new(false));
    let hits = Arc::new(AtomicUsize::new(0));
    let url = start_health_stub(serving.clone(), hits.clone()).await;
    let registry = Arc::new(WorkerRegistry::new());
    let mut manager = manager_over(&registry);

    let worker = probed_worker_with_timeout(&url, ConnectionMode::Http, 2, 2, 5);
    registered_and_held(&registry, &worker, &hits).await;

    serving.store(true, Ordering::SeqCst);
    let back = tokio::time::Instant::now();
    wait_until(
        back + Duration::from_millis(5500),
        "the worker did not return within two intervals: the brought-forward read landed \
         after the scheduled one",
        || worker.status() == WorkerStatus::Ready,
    )
    .await;
    assert!(
        tokio::time::Instant::now() >= back + Duration::from_millis(3500),
        "the scheduled pass an interval after the launch, the promoting read an interval later"
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        3,
        "the held read, the scheduled pass and the pass an interval later"
    );

    manager.shutdown().await;
}

#[test]
fn apply_connect_signal_reschedules_a_promoted_worker() {
    // A worker that hit the Pending cap is dropped from the schedule. A
    // late handshake promotes it to Ready, so it must re-enter the probe
    // schedule — otherwise it would serve traffic unmonitored forever.
    let registry = Arc::new(WorkerRegistry::new());
    let worker = make_zmq_worker("ipc:///tmp/w.ipc");
    worker.set_status(WorkerStatus::Failed);
    let revision = worker.revision();
    let worker_id = registry.register(worker.clone()).unwrap();

    let mut next_check = HashMap::new();
    apply_connect_signal(
        &registry,
        WorkerConnected {
            url: "ipc:///tmp/w.ipc".to_string(),
            revision,
        },
        &mut next_check,
        &test_config(),
    );

    assert_eq!(worker.status(), WorkerStatus::Ready);
    assert!(
        next_check.contains_key(&worker_id),
        "a promoted worker must be probed again"
    );
}

/// The signal's promotion is reported for what follows a transition into
/// service (the label refresh): the worker it moved, and nothing when the
/// signal moved nothing (already Ready, or an unknown URL).
#[test]
fn apply_connect_signal_reports_the_promoted_worker() {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = unprobed_worker("http://w:1");
    worker.set_status(WorkerStatus::NotReady);
    let revision = worker.revision();
    let worker_id = registry.register(worker.clone()).unwrap();
    let mut next_check = HashMap::new();
    let signal = || WorkerConnected {
        url: "http://w:1".to_string(),
        revision,
    };

    let (promoted_id, promoted) =
        apply_connect_signal(&registry, signal(), &mut next_check, &test_config())
            .expect("a NotReady worker without health checks answering a contact is promoted");
    assert!(promoted_id == worker_id);
    assert_eq!(promoted.url(), "http://w:1");
    assert!(
        apply_connect_signal(&registry, signal(), &mut next_check, &test_config()).is_none(),
        "a second signal moves nothing: the worker is Ready already"
    );
    assert!(
        apply_connect_signal(
            &registry,
            WorkerConnected {
                url: "http://elsewhere:1".to_string(),
                revision,
            },
            &mut next_check,
            &test_config(),
        )
        .is_none(),
        "an unknown URL moves nothing"
    );
}

/// A promotion by contact is a transition into service like one by probe:
/// the label refresh is submitted for a locally registered worker.
#[tokio::test]
async fn a_connect_signal_promotion_submits_a_label_refresh() {
    use crate::{service_discovery::testing::create_test_app_context, workflow::JobQueueConfig};
    let app_context = create_test_app_context();
    let registry = Arc::clone(&app_context.worker_registry);
    let job_queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&app_context));

    let worker = discovered_worker_with(
        "http://local:11",
        HealthCheckConfig {
            disable_health_check: true,
            ..HealthCheckConfig::default()
        },
    );
    worker.set_status(WorkerStatus::NotReady);
    let revision = worker.revision();
    registry.register(worker.clone()).unwrap();
    let mut next_check = HashMap::new();
    apply_connect_signal_and_refresh(
        &registry,
        WorkerConnected {
            url: "http://local:11".to_string(),
            revision,
        },
        &mut next_check,
        &test_config(),
        Some(&job_queue),
    )
    .await;

    assert_eq!(worker.status(), WorkerStatus::Ready);
    assert!(
        job_queue.get_status("http://local:11").is_some(),
        "a refresh job was submitted for the worker the contact promoted"
    );
}

/// A handshake can complete while the worker's registration job is still
/// running its last steps (the worker is in the registry from the register
/// step on). The refresh that signal asks for is kept behind that job and
/// run when it ends, never dropped: on the ZMQ wire it is what gives the
/// worker its window, and a worker that stays in service has no next
/// transition to retry on. The registration's status stays its own.
#[tokio::test]
async fn a_connect_signal_during_the_registration_job_keeps_the_refresh() {
    use crate::{service_discovery::testing::create_test_app_context, workflow::JobQueueConfig};
    let app_context = create_test_app_context();
    let registry = Arc::clone(&app_context.worker_registry);
    let job_queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&app_context));

    let worker = discovered_worker_with(
        "http://local:12",
        HealthCheckConfig {
            disable_health_check: true,
            ..HealthCheckConfig::default()
        },
    );
    worker.set_status(WorkerStatus::NotReady);
    let revision = worker.revision();
    registry.register(worker.clone()).unwrap();
    job_queue.hold_url_for_tests("http://local:12", "AddWorker");

    apply_connect_signal_and_refresh(
        &registry,
        WorkerConnected {
            url: "http://local:12".to_string(),
            revision,
        },
        &mut HashMap::new(),
        &test_config(),
        Some(&job_queue),
    )
    .await;

    assert_eq!(worker.status(), WorkerStatus::Ready);
    assert!(
        job_queue.kept_for("http://local:12"),
        "the refresh is kept behind the registration job"
    );
    assert_eq!(
        job_queue
            .get_status("http://local:12")
            .expect("the registration's status")
            .job_type,
        "AddWorker",
        "the registration keeps its own status"
    );
}

#[test]
fn apply_connect_signal_ignores_stale_revision() {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = make_zmq_worker("ipc:///tmp/w.ipc");
    let stale_revision = worker.revision();
    let worker_id = registry.register(worker).unwrap();

    // A same-URL replace bumps the revision; a handshake that started
    // against the old worker must not promote its replacement.
    let replacement = make_zmq_worker("ipc:///tmp/w.ipc");
    assert!(registry.replace(&worker_id, replacement));
    let current = registry.get(&worker_id).unwrap();
    assert_eq!(current.revision(), stale_revision + 1);
    assert_eq!(current.status(), WorkerStatus::Pending);

    apply_connect_signal(
        &registry,
        WorkerConnected {
            url: "ipc:///tmp/w.ipc".to_string(),
            revision: stale_revision,
        },
        &mut HashMap::new(),
        &test_config(),
    );

    assert_eq!(
        registry.get(&worker_id).unwrap().status(),
        WorkerStatus::Pending,
        "a stale connect signal must not promote a replaced worker"
    );
}

#[test]
fn apply_connect_signal_ignores_unknown_url() {
    // A signal for a worker that was removed before its handshake landed
    // must be a silent no-op, not a panic.
    let registry = Arc::new(WorkerRegistry::new());
    apply_connect_signal(
        &registry,
        WorkerConnected {
            url: "http://ghost:1".to_string(),
            revision: 0,
        },
        &mut HashMap::new(),
        &test_config(),
    );
}

const STUB_METRICS: &str = "# HELP test_metric stub\n# TYPE test_metric gauge\ntest_metric 1\n";

/// Tiny HTTP stub counting GET /metrics hits; returns its base URL.
async fn start_metrics_stub(hits: Arc<AtomicUsize>) -> String {
    let app = axum::Router::new().route(
        "/metrics",
        axum::routing::get(move || {
            let hits = hits.clone();
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                STUB_METRICS
            }
        }),
    );
    serve_stub(app).await
}

/// The same stub behind basic authentication: a request whose
/// `Authorization` header is not the given user and password is answered
/// 401. Every request counts, answered or refused.
async fn start_metrics_stub_with_basic_auth(
    hits: Arc<AtomicUsize>,
    user: &str,
    password: &str,
) -> String {
    let expected = format!(
        "Basic {}",
        BASE64_STANDARD.encode(format!("{user}:{password}"))
    );
    let app = axum::Router::new().route(
        "/metrics",
        axum::routing::get(move |headers: HeaderMap| {
            let hits = hits.clone();
            let expected = expected.clone();
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                let presented = headers
                    .get(AUTHORIZATION)
                    .and_then(|value| value.to_str().ok());
                if presented == Some(expected.as_str()) {
                    (StatusCode::OK, STUB_METRICS)
                } else {
                    (StatusCode::UNAUTHORIZED, "")
                }
            }
        }),
    );
    serve_stub(app).await
}

/// A stub whose engine is broken: every GET /metrics is counted and
/// answered 500.
async fn start_failing_stub(hits: Arc<AtomicUsize>) -> String {
    let app = axum::Router::new().route(
        "/metrics",
        axum::routing::get(move || {
            let hits = hits.clone();
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                (StatusCode::INTERNAL_SERVER_ERROR, "")
            }
        }),
    );
    serve_stub(app).await
}

/// Serves a stub router on a free local port; returns its base URL.
async fn serve_stub(app: axum::Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind stub");
    let addr = listener.local_addr().expect("stub address");
    #[expect(
        clippy::disallowed_methods,
        reason = "test stub server lives for the duration of the test process"
    )]
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("stub serve");
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn engine_metrics_scrapes_dp_workers_via_base_url_once() {
    // Two DP ranks share one base worker. The `@{rank}` suffix in
    // worker.url() is not a valid endpoint; metrics must be scraped
    // from base_url(), once per base worker (#1993).
    let hits = Arc::new(AtomicUsize::new(0));
    let base = start_metrics_stub(hits.clone()).await;

    let registry = WorkerRegistry::new();
    registry.register(make_dp_worker(&base, 0, 2)).unwrap();
    registry.register(make_dp_worker(&base, 1, 2)).unwrap();

    let result = WorkerManager::get_engine_metrics(&registry).await;

    assert!(
        matches!(result, EngineMetricsResult::Ok(_)),
        "expected metrics scrape to succeed"
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "base worker must be scraped once, not per DP rank"
    );
}

#[tokio::test]
async fn engine_metrics_scrapes_each_distinct_base_worker() {
    let hits_a = Arc::new(AtomicUsize::new(0));
    let base_a = start_metrics_stub(hits_a.clone()).await;
    let hits_b = Arc::new(AtomicUsize::new(0));
    let base_b = start_metrics_stub(hits_b.clone()).await;

    let registry = WorkerRegistry::new();
    registry.register(make_dp_worker(&base_a, 0, 2)).unwrap();
    registry.register(make_dp_worker(&base_a, 1, 2)).unwrap();
    registry.register(make_dp_worker(&base_b, 0, 1)).unwrap();

    let result = WorkerManager::get_engine_metrics(&registry).await;

    assert!(matches!(result, EngineMetricsResult::Ok(_)));
    assert_eq!(hits_a.load(Ordering::SeqCst), 1);
    assert_eq!(hits_b.load(Ordering::SeqCst), 1);
}

fn grpc_worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .build(),
    )
}

/// A mixed fleet: the HTTP worker's `/metrics` is scraped and aggregated
/// under its `worker_addr`, the gRPC worker gets no HTTP request (its base
/// URL is a gRPC listener) and is named in a comment line of the answer.
/// Neither worker's credentials reach the answer: the label and the comment
/// line carry the redacted address.
#[tokio::test]
async fn engine_metrics_scrapes_the_http_workers_and_names_the_others() {
    let hits = Arc::new(AtomicUsize::new(0));
    let base = start_metrics_stub(hits.clone()).await;
    let http_with_secret = base.replacen("http://", "http://user:secret@", 1);

    let registry = WorkerRegistry::new();
    registry
        .register(make_worker(&http_with_secret, 1, 1))
        .unwrap();
    registry
        .register(grpc_worker("grpc://user:secret@engine-a:50051"))
        .unwrap();

    let text = match WorkerManager::get_engine_metrics(&registry).await {
        EngineMetricsResult::Ok(text) => text,
        EngineMetricsResult::Err(err) => panic!("a mixed fleet is scraped: {err}"),
    };

    assert_eq!(hits.load(Ordering::SeqCst), 1, "the HTTP worker, once");
    assert!(text.contains("test_metric"), "{text}");
    let masked_http = base.replacen("http://", "http://***@", 1);
    assert!(
        text.contains(&format!("worker_addr=\"{masked_http}\"")),
        "the label carries the redacted address:\n{text}"
    );
    assert!(
        text.lines().any(|line| {
            line.starts_with("# not scraped: grpc://***@engine-a:50051 (grpc worker)")
        }),
        "{text}"
    );
    assert!(!text.contains("secret"), "{text}");
}

/// Two registrations of one address under other credentials are one scrape
/// target: the first in URL order answers, the second gets no request, and
/// a comment line names the registration that answered, so the aggregate
/// carries one `worker_addr` per address and no credential.
#[tokio::test]
async fn engine_metrics_scrapes_an_address_once_whatever_its_credentials() {
    let hits = Arc::new(AtomicUsize::new(0));
    let base = start_metrics_stub(hits.clone()).await;
    let first = base.replacen("http://", "http://user:secret@", 1);
    let second = base.replacen("http://", "http://other:hunter2@", 1);

    let registry = WorkerRegistry::new();
    registry.register(make_worker(&first, 1, 1)).unwrap();
    registry.register(make_worker(&second, 1, 1)).unwrap();

    let text = match WorkerManager::get_engine_metrics(&registry).await {
        EngineMetricsResult::Ok(text) => text,
        EngineMetricsResult::Err(err) => {
            panic!("one address under two credentials is scraped: {err}")
        }
    };

    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "one request for the address"
    );
    let masked = base.replacen("http://", "http://***@", 1);
    assert_eq!(
        text.matches(&format!("worker_addr=\"{masked}\"")).count(),
        1,
        "one series per address:\n{text}"
    );
    let named = format!(
        "# scraped once: {masked} (2 registrations under other credentials, tried in URL order): registration 1 answered"
    );
    assert!(text.lines().any(|line| line == named), "{text}");
    for secret in ["secret", "hunter2", "user:", "other:"] {
        assert!(
            !text.contains(secret),
            "{secret} reached the answer:\n{text}"
        );
    }
}

/// One address under two credentials, the first stale: the registrations
/// are tried in URL order, so the stale one is scraped first and answers
/// 401, the valid one is tried next and answers. The address's series are
/// present once, the failed scrape is a warn line and the comment line names
/// the registration that answered. Registration order plays no part: the
/// valid registration is registered first and still tried second.
#[tokio::test]
#[tracing_test::traced_test]
async fn engine_metrics_tries_the_next_registration_when_stale_credentials_get_a_401() {
    let hits = Arc::new(AtomicUsize::new(0));
    let base = start_metrics_stub_with_basic_auth(hits.clone(), "valid", "p4ss-valid").await;
    let stale = base.replacen("http://", "http://stale:p4ss-stale@", 1);
    let valid = base.replacen("http://", "http://valid:p4ss-valid@", 1);

    let registry = WorkerRegistry::new();
    registry.register(make_worker(&valid, 1, 1)).unwrap();
    registry.register(make_worker(&stale, 1, 1)).unwrap();

    let text = match WorkerManager::get_engine_metrics(&registry).await {
        EngineMetricsResult::Ok(text) => text,
        EngineMetricsResult::Err(err) => {
            panic!("a registration with valid credentials is scraped: {err}")
        }
    };

    assert_eq!(
        hits.load(Ordering::SeqCst),
        2,
        "the stale registration first, then the valid one"
    );
    assert!(text.contains("test_metric"), "{text}");
    let masked = base.replacen("http://", "http://***@", 1);
    assert_eq!(
        text.matches(&format!("worker_addr=\"{masked}\"")).count(),
        1,
        "one series per address:\n{text}"
    );
    let named = format!(
        "# scraped once: {masked} (2 registrations under other credentials, tried in URL order): registration 2 answered"
    );
    assert!(text.lines().any(|line| line == named), "{text}");
    assert!(
        logs_contain("GET /metrics of an HTTP worker failed"),
        "the refused scrape leaves a warn line"
    );
    for secret in ["p4ss", "stale:", "valid:"] {
        assert!(
            !text.contains(secret),
            "{secret} reached the answer:\n{text}"
        );
    }
}

/// A credentialed registration whose engine is down: the scrape fails, the
/// warn line names the masked address and the failure, and the credentials
/// reach neither the line nor the answer. The HTTP client moves a URL's
/// credentials into a header as it builds the request, and the line logs
/// the error without its URL: neither path can carry them.
#[tokio::test]
#[tracing_test::traced_test]
async fn engine_metrics_logs_a_failed_scrape_without_its_credentials() {
    // The down engine's port: held and refused for the whole case.
    let down = RefusedAddress::reserve();
    let closed = down.port();
    let registry = WorkerRegistry::new();
    registry
        .register(make_worker(
            &format!("http://scraper:p4ss-secret@127.0.0.1:{closed}"),
            1,
            1,
        ))
        .unwrap();

    let answer = match WorkerManager::get_engine_metrics(&registry).await {
        EngineMetricsResult::Ok(text) => panic!("a down engine is a failed scrape: {text}"),
        EngineMetricsResult::Err(err) => err,
    };

    assert_eq!(answer, "All backend requests failed");
    assert!(logs_contain("GET /metrics of an HTTP worker failed"));
    assert!(logs_contain(&format!("http://***@127.0.0.1:{closed}")));
    logs_assert(|lines: &[&str]| {
        match lines
            .iter()
            .find(|line| line.contains("p4ss-secret") || line.contains("scraper:"))
        {
            Some(line) => Err(format!("a credential reached the log: {line}")),
            None => Ok(()),
        }
    });
}

/// Two registrations of one address at a host that refuses the connection
/// (the engine down behind a port nothing listens on): a transport error
/// says nothing of the credentials, so the first registration's failure
/// closes the address in its round, one request and one warn line, and the
/// comment line says so. The healthy worker beside it is answered as usual.
#[tokio::test]
#[tracing_test::traced_test]
async fn engine_metrics_closes_an_address_on_a_transport_error() {
    // The down engine's port: held and refused for the whole case, so no
    // later bind, the healthy stub's included, can draw it.
    let down = RefusedAddress::reserve();
    let closed = down.port();
    let healthy_hits = Arc::new(AtomicUsize::new(0));
    let healthy = start_metrics_stub(healthy_hits.clone()).await;

    let registry = WorkerRegistry::new();
    registry
        .register(make_worker(
            &format!("http://alpha:p4ss-alpha@127.0.0.1:{closed}"),
            1,
            1,
        ))
        .unwrap();
    registry
        .register(make_worker(
            &format!("http://beta:p4ss-beta@127.0.0.1:{closed}"),
            1,
            1,
        ))
        .unwrap();
    registry.register(make_worker(&healthy, 1, 1)).unwrap();

    let text = match WorkerManager::get_engine_metrics(&registry).await {
        EngineMetricsResult::Ok(text) => text,
        EngineMetricsResult::Err(err) => panic!("the healthy worker is scraped: {err}"),
    };

    assert_eq!(healthy_hits.load(Ordering::SeqCst), 1);
    let masked = format!("# not scraped: http://***@127.0.0.1:{closed}");
    let closed_line =
        "registration 1 failed and closed the address, only a 401 or 403 tries the next";
    assert!(
        text.lines()
            .any(|line| line.starts_with(&masked) && line.ends_with(closed_line)),
        "{text}"
    );
    logs_assert(|lines: &[&str]| {
        let failures = lines
            .iter()
            .filter(|line| line.contains("GET /metrics of an HTTP worker failed"))
            .count();
        if failures == 1 {
            Ok(())
        } else {
            Err(format!(
                "{failures} failed-scrape lines: a transport error closes the address after one request"
            ))
        }
    });
}

/// Two registrations of one address whose engine answers 500: the group
/// shares the host, so the first failure that is not a credential refusal
/// closes the address in its round (one request, one warn line) and the
/// comment line says so; the healthy worker beside it is answered as usual.
#[tokio::test]
#[tracing_test::traced_test]
async fn engine_metrics_closes_an_address_on_a_failure_other_than_a_credential_refusal() {
    let failing_hits = Arc::new(AtomicUsize::new(0));
    let failing = start_failing_stub(failing_hits.clone()).await;
    let healthy_hits = Arc::new(AtomicUsize::new(0));
    let healthy = start_metrics_stub(healthy_hits.clone()).await;
    let first = failing.replacen("http://", "http://alpha:p4ss-alpha@", 1);
    let second = failing.replacen("http://", "http://beta:p4ss-beta@", 1);

    let registry = WorkerRegistry::new();
    registry.register(make_worker(&first, 1, 1)).unwrap();
    registry.register(make_worker(&second, 1, 1)).unwrap();
    registry.register(make_worker(&healthy, 1, 1)).unwrap();

    let text = match WorkerManager::get_engine_metrics(&registry).await {
        EngineMetricsResult::Ok(text) => text,
        EngineMetricsResult::Err(err) => panic!("the healthy worker is scraped: {err}"),
    };

    assert_eq!(
        failing_hits.load(Ordering::SeqCst),
        1,
        "a 500 closes the address: the second registration is not tried"
    );
    assert_eq!(healthy_hits.load(Ordering::SeqCst), 1);
    let masked = failing.replacen("http://", "http://***@", 1);
    let named = format!(
        "# not scraped: {masked} (2 registrations under other credentials): registration 1 failed and closed the address, only a 401 or 403 tries the next"
    );
    assert!(text.lines().any(|line| line == named), "{text}");
    logs_assert(|lines: &[&str]| {
        let failures = lines
            .iter()
            .filter(|line| line.contains("GET /metrics of an HTTP worker failed"))
            .count();
        if failures == 1 {
            Ok(())
        } else {
            Err(format!("{failures} warn lines for one closed address"))
        }
    });
    for secret in ["p4ss", "alpha:", "beta:"] {
        assert!(
            !text.contains(secret),
            "{secret} reached the answer:\n{text}"
        );
    }
}

/// A fleet with no HTTP worker is not a failed scrape: the route answers
/// the comment lines alone, a 200 with no series.
#[tokio::test]
async fn engine_metrics_answers_a_fleet_without_http_workers_as_an_empty_aggregate() {
    let registry = WorkerRegistry::new();
    registry
        .register(grpc_worker("grpc://engine-a:50051"))
        .unwrap();

    let text = match WorkerManager::get_engine_metrics(&registry).await {
        EngineMetricsResult::Ok(text) => text,
        EngineMetricsResult::Err(err) => panic!("no HTTP worker is not a server error: {err}"),
    };

    assert!(text.starts_with("# no HTTP worker registered"), "{text}");
    assert!(
        text.contains("# not scraped: grpc://engine-a:50051 (grpc worker)"),
        "{text}"
    );
    assert!(
        text.lines().all(|line| line.starts_with('#')),
        "no series in the aggregate of zero HTTP workers:\n{text}"
    );
}

fn cfg(success_threshold: u32, failure_threshold: u32) -> HealthCheckConfig {
    HealthCheckConfig {
        success_threshold,
        failure_threshold,
        timeout_secs: 1,
        check_interval_secs: 1,
        disable_health_check: false,
        drain_settle_secs: 0,
    }
}

/// A failed worker's removal follows the scope: every local worker under
/// `AllLocal`, only one a discovery provider registered (its provider
/// label) under `DiscoveryOwned`, none under `Disabled`. A mesh import
/// never goes: it is owner-managed.
#[test]
fn a_failed_workers_removal_follows_the_scope() {
    use UnhealthyRemovalScope::{AllLocal, Disabled, DiscoveryOwned};
    let registry = Arc::new(WorkerRegistry::new());

    let static_id = registry
        .register(make_worker("http://static:1", 1, 1))
        .unwrap();
    let discovered: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://discovered:1")
            .worker_type(WorkerType::Regular)
            .label(DISCOVERY_PROVIDER_LABEL, "consul")
            .build(),
    );
    let discovered_id = registry.register(discovered).unwrap();
    registry.on_remote_worker_state(&smg_mesh::WorkerState {
        worker_id: "peer-w1".to_string(),
        model_id: "m".to_string(),
        url: "http://remote:1".to_string(),
        health: true,
        load: 0.0,
        version: 1,
        spec: vec![],
        owner: "peer".to_string(),
    });
    let mesh_id = registry.get_id_by_url("http://remote:1").unwrap();

    // (scope, static or API-added, discovered, mesh-imported)
    for (scope, static_goes, discovered_goes) in [
        (Disabled, false, false),
        (DiscoveryOwned, false, true),
        (AllLocal, true, true),
    ] {
        assert_eq!(
            should_remove_failed(&registry, &static_id, scope),
            static_goes,
            "static worker under {scope:?}"
        );
        assert_eq!(
            should_remove_failed(&registry, &discovered_id, scope),
            discovered_goes,
            "discovered worker under {scope:?}"
        );
        assert!(
            !should_remove_failed(&registry, &mesh_id, scope),
            "mesh-imported worker under {scope:?}"
        );
    }
}

#[test]
fn test_state_machine_pending_to_ready_after_success_threshold() {
    let worker = make_worker("http://w:1", 2, 3);
    assert_eq!(worker.status(), WorkerStatus::Pending);

    // First success: not yet promoted (1 < 2)
    assert_eq!(compute_next_status(&worker, true, &cfg(2, 3)), None);
    assert_eq!(worker.status(), WorkerStatus::Pending);

    // Second success: promoted Pending → Ready
    let next = compute_next_status(&worker, true, &cfg(2, 3));
    assert_eq!(next, Some(WorkerStatus::Ready));
}

#[test]
fn test_state_machine_ready_to_notready_after_failure_threshold() {
    let worker = make_worker("http://w:1", 2, 3);
    worker.set_status(WorkerStatus::Ready);

    // 1 fail, 2 fail: still Ready
    assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
    assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
    assert_eq!(worker.status(), WorkerStatus::Ready);

    // 3rd fail: Ready → NotReady
    assert_eq!(
        compute_next_status(&worker, false, &cfg(2, 3)),
        Some(WorkerStatus::NotReady)
    );
}

#[test]
fn test_state_machine_notready_to_failed_after_liveness_threshold() {
    let worker = make_worker("http://w:1", 2, 3);
    worker.set_status(WorkerStatus::NotReady);

    // liveness_threshold = 3 × failure_threshold = 9
    for i in 1..9 {
        assert_eq!(
            compute_next_status(&worker, false, &cfg(2, 3)),
            None,
            "iteration {i}"
        );
    }

    // 9th consecutive failure → Failed
    assert_eq!(
        compute_next_status(&worker, false, &cfg(2, 3)),
        Some(WorkerStatus::Failed)
    );
}

#[test]
fn test_state_machine_pending_to_failed_after_max_pending_probes() {
    let worker = make_worker("http://w:1", 2, 3);
    // max_pending_probes = 10 × failure_threshold = 30

    // Simulate 30 failed probes — increment counter manually since the
    // loop usually does this before calling compute_next_status.
    for _ in 0..29 {
        worker.total_pending_probes_increment();
        assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
    }
    worker.total_pending_probes_increment();
    // 30th: Pending → Failed
    assert_eq!(
        compute_next_status(&worker, false, &cfg(2, 3)),
        Some(WorkerStatus::Failed)
    );
}

#[test]
fn test_state_machine_pending_to_failed_on_success_when_cap_exceeded() {
    // Flapping pattern: a Pending worker that flaps F,S,F,S,... and
    // never reaches success_threshold should still hit max_pending_probes.
    let worker = make_worker("http://w:1", 2, 3);

    // Simulate 30 attempts with the counter, then call compute on success.
    for _ in 0..30 {
        worker.total_pending_probes_increment();
    }
    // Even on success, the cap fires.
    assert_eq!(
        compute_next_status(&worker, true, &cfg(2, 3)),
        Some(WorkerStatus::Failed)
    );
}

#[test]
fn test_state_machine_failed_recovers_after_success_threshold() {
    let worker = make_worker("http://w:1", 2, 3);
    worker.set_status(WorkerStatus::Failed);

    // Failures keep it Failed.
    assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
    assert_eq!(worker.status(), WorkerStatus::Failed);

    // The engine came back on the same address: success_threshold
    // consecutive successes promote it exactly like NotReady, because a
    // static fleet has no other way to rejoin.
    assert_eq!(compute_next_status(&worker, true, &cfg(2, 3)), None);
    assert_eq!(
        compute_next_status(&worker, true, &cfg(2, 3)),
        Some(WorkerStatus::Ready)
    );
}

#[test]
fn test_state_machine_notready_to_ready_on_success_threshold() {
    let worker = make_worker("http://w:1", 2, 3);
    worker.set_status(WorkerStatus::NotReady);

    assert_eq!(compute_next_status(&worker, true, &cfg(2, 3)), None);
    assert_eq!(
        compute_next_status(&worker, true, &cfg(2, 3)),
        Some(WorkerStatus::Ready)
    );
}

#[test]
fn test_state_machine_success_resets_failure_counter() {
    let worker = make_worker("http://w:1", 2, 3);
    worker.set_status(WorkerStatus::Ready);

    // 2 failures (not yet at threshold)
    assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
    assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);

    // Single success resets the counter
    assert_eq!(compute_next_status(&worker, true, &cfg(2, 3)), None);

    // Now 2 failures again — still no transition because counter was reset
    assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
    assert_eq!(compute_next_status(&worker, false, &cfg(2, 3)), None);
    assert_eq!(worker.status(), WorkerStatus::Ready);

    // 3rd failure now triggers transition
    assert_eq!(
        compute_next_status(&worker, false, &cfg(2, 3)),
        Some(WorkerStatus::NotReady)
    );
}

#[tokio::test]
async fn test_removal_candidate_strips_dp_rank_suffix() {
    // Regression test: a DP-aware worker is registered under
    // "{base}@{rank}", but the removal workflow prefix-matches
    // "{url}@" in dp_aware mode. Submitting the registered (suffixed)
    // URL made the prefix "{base}@{rank}@", which matched nothing, so
    // unhealthy DP workers were never removed.
    let registry = Arc::new(WorkerRegistry::new());
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://10.130.99.80:30000")
            .worker_type(WorkerType::Regular)
            .health_config(cfg(2, 3))
            .dp_config(1, 4)
            .build(),
    );
    assert_eq!(worker.url(), "http://10.130.99.80:30000@1");
    worker.set_status(WorkerStatus::Failed);
    let worker_id = registry.register(worker).unwrap();

    let mut next_check = HashMap::new();
    next_check.insert(worker_id, tokio::time::Instant::now());
    let mut in_flight = HashSet::new();
    let mut probes: ProbeFutures = FuturesUnordered::new();

    let removals = queue_due_probes(
        &registry,
        &config_removing(UnhealthyRemovalScope::AllLocal),
        &mut next_check,
        &mut in_flight,
        &mut probes,
        tokio::time::Instant::now(),
    );

    assert_eq!(removals.len(), 1);
    assert_eq!(removals[0].url, "http://10.130.99.80:30000");
}

/// Under the discovery default, a failed static or API-added worker is
/// not removed: nothing would register it again, so it keeps its probe
/// and rejoins when its engine answers. A failed discovered worker beside
/// it is removed, and its provider registers it again.
#[tokio::test]
async fn the_discovery_default_removes_only_the_discovered_failed_worker() {
    let registry = Arc::new(WorkerRegistry::new());
    let static_worker = make_worker("http://static:1", 2, 3);
    static_worker.set_status(WorkerStatus::Failed);
    let static_id = registry.register(static_worker).unwrap();
    let discovered: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://discovered:1")
            .worker_type(WorkerType::Regular)
            .health_config(cfg(2, 3))
            .label(DISCOVERY_PROVIDER_LABEL, "file")
            .build(),
    );
    discovered.set_status(WorkerStatus::Failed);
    let discovered_id = registry.register(discovered).unwrap();

    let now = tokio::time::Instant::now();
    let mut next_check = HashMap::from([(static_id.clone(), now), (discovered_id, now)]);
    let mut in_flight = HashSet::new();
    let mut probes: ProbeFutures = FuturesUnordered::new();
    let removals = queue_due_probes(
        &registry,
        &config_removing(UnhealthyRemovalScope::DiscoveryOwned),
        &mut next_check,
        &mut in_flight,
        &mut probes,
        now,
    );

    let removed: Vec<&str> = removals
        .iter()
        .map(|removal| removal.url.as_str())
        .collect();
    assert_eq!(removed, ["http://discovered:1"]);
    assert!(
        next_check.contains_key(&static_id) && in_flight.contains(&static_id),
        "the static worker keeps being probed"
    );
}

#[test]
fn test_reconcile_from_registry_skips_failed_workers_on_bootstrap() {
    let (registry, failed_id) = registry_with_a_failed_worker();

    let mut next_check = HashMap::new();
    reconcile_from_registry(
        &registry,
        &mut next_check,
        &config_removing(UnhealthyRemovalScope::AllLocal),
    );

    assert!(
        !next_check.contains_key(&failed_id),
        "bootstrap reconcile must not reschedule failed workers"
    );
}

#[test]
fn test_reconcile_from_registry_keeps_failed_workers_probed_without_removal() {
    // Without --remove-unhealthy-workers a Failed worker is kept on the
    // schedule: the probe is how a restart on the same address rejoins.
    let (registry, failed_id) = registry_with_a_failed_worker();
    let mut next_check = HashMap::new();
    reconcile_from_registry(&registry, &mut next_check, &test_config());
    assert!(
        next_check.contains_key(&failed_id),
        "a failed worker that is not being removed must keep being probed"
    );
}

#[test]
fn test_reconcile_from_registry_captures_pending_and_ready_workers() {
    // Positive complement to the "skips failed" test: the startup
    // reconcile must pick up Pending (not-yet-probed) and Ready
    // workers that existed at registry snapshot time. This is what
    // makes WorkerManager::start() deterministic — the schedule is
    // captured on the caller's thread, not whenever the spawned task
    // happens to run.
    let registry = Arc::new(WorkerRegistry::new());

    let pending_worker = make_worker("http://pending:1", 2, 3);
    assert_eq!(pending_worker.status(), WorkerStatus::Pending);
    let pending_id = registry.register(pending_worker).unwrap();

    let ready_worker = make_worker("http://ready:1", 2, 3);
    ready_worker.set_status(WorkerStatus::Ready);
    let ready_id = registry.register(ready_worker).unwrap();

    let mut next_check = HashMap::new();
    reconcile_from_registry(
        &registry,
        &mut next_check,
        &config_removing(UnhealthyRemovalScope::AllLocal),
    );

    assert!(
        next_check.contains_key(&pending_id),
        "pending worker must be in the bootstrap schedule"
    );
    assert!(
        next_check.contains_key(&ready_id),
        "ready worker must be in the bootstrap schedule"
    );
}

/// The bootstrap schedules a worker without a verdict as a registration
/// would: due now, not an interval out. A worker that carries a verdict
/// keeps its interval.
#[test]
fn the_bootstrap_schedules_a_pending_worker_now_and_a_ready_one_an_interval_out() {
    let registry = Arc::new(WorkerRegistry::new());
    let pending_worker = make_worker("http://pending:1", 2, 3);
    assert_eq!(pending_worker.status(), WorkerStatus::Pending);
    let pending_id = registry.register(pending_worker).unwrap();
    let ready_worker = make_worker("http://ready:1", 2, 3);
    ready_worker.set_status(WorkerStatus::Ready);
    let ready_id = registry.register(ready_worker).unwrap();

    let before = tokio::time::Instant::now();
    let mut next_check = HashMap::new();
    reconcile_from_registry(&registry, &mut next_check, &test_config());
    let after = tokio::time::Instant::now();

    assert!(
        next_check[&pending_id] <= after,
        "a worker without a verdict is due the moment the loop starts, not an interval later"
    );
    assert!(
        next_check[&ready_id] >= before + Duration::from_secs(1),
        "a worker that carries a verdict keeps its interval"
    );
}

/// Through the running loop: a worker registered before the manager started
/// (a start-up worker whose registration won the race with the manager's
/// start) is probed on the loop's first tick, as one registered under the
/// running loop is. With an hour between probes, the only probe this test
/// can see is that first one.
#[tokio::test]
async fn a_worker_registered_before_the_manager_starts_is_probed_at_once_not_an_interval_later() {
    let serving = Arc::new(AtomicBool::new(true));
    let hits = Arc::new(AtomicUsize::new(0));
    let url = start_health_stub(serving, hits.clone()).await;
    let registry = Arc::new(WorkerRegistry::new());
    let worker = probed_worker(&url, ConnectionMode::Http);
    assert_eq!(worker.status(), WorkerStatus::Pending);
    registry.register(worker.clone()).unwrap();

    let started = tokio::time::Instant::now();
    let mut manager = manager_over(&registry);
    wait_until(
        started + Duration::from_secs(5),
        "a worker already in the registry at the manager's start was not probed at once: its \
         first read waits a whole interval",
        || hits.load(Ordering::SeqCst) > 0,
    )
    .await;
    wait_until(
        started + Duration::from_secs(5),
        "the first read's pass did not promote the worker",
        || worker.status() == WorkerStatus::Ready,
    )
    .await;

    manager.shutdown().await;
}

/// A worker the bootstrap finds with a verdict is not read at the start: its
/// interval stands; only the workers without one are brought forward.
#[tokio::test]
async fn a_ready_worker_registered_before_the_manager_starts_keeps_its_interval() {
    let serving = Arc::new(AtomicBool::new(true));
    let hits = Arc::new(AtomicUsize::new(0));
    let url = start_health_stub(serving, hits.clone()).await;
    let registry = Arc::new(WorkerRegistry::new());
    let worker = probed_worker(&url, ConnectionMode::Http);
    worker.set_status(WorkerStatus::Ready);
    registry.register(worker).unwrap();

    let mut manager = manager_over(&registry);
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        hits.load(Ordering::SeqCst),
        0,
        "a worker that carries a verdict is read an interval after the start, not at once"
    );

    manager.shutdown().await;
}

#[tokio::test]
async fn test_worker_manager_start_is_deterministic_with_preexisting_workers() {
    // End-to-end contract for fix #2: a worker that exists in the
    // registry before WorkerManager::start() returns must be on the
    // schedule the moment the spawned task begins running — no race
    // with task scheduling. We can't observe `next_check` directly,
    // so we run the full start/shutdown lifecycle with a very long
    // probe interval (so no probe actually fires) and verify the
    // happy path doesn't panic. Together with the reconcile unit
    // test above, this covers both the "reconcile captures workers"
    // and "start() calls reconcile synchronously" invariants.
    let registry = Arc::new(WorkerRegistry::new());
    let worker = make_worker("http://pre-existing:1", 2, 3);
    worker.set_status(WorkerStatus::Ready);
    registry.register(worker).unwrap();

    let mut manager = WorkerManager::start(
        registry,
        WorkerManagerConfig {
            default_check_interval_secs: 3600,
            remove_unhealthy: UnhealthyRemovalScope::Disabled,
        },
        None,
    );
    manager.shutdown().await;
}

#[tokio::test]
async fn test_apply_probe_completion_discards_stale_probe_after_replace() {
    let registry = Arc::new(WorkerRegistry::new());
    let worker = make_worker("http://w:1", 1, 1);
    worker.set_status(WorkerStatus::Ready);
    let worker_id = registry.register(worker.clone()).unwrap();
    let expected_revision = worker.revision();

    let completion = ProbeCompletion {
        worker_id: worker_id.clone(),
        worker: worker.clone(),
        expected_revision,
        launched_status: WorkerStatus::Ready,
        probe_result: Err(WorkerError::HealthCheckFailed {
            url: worker.url().to_string(),
            reason: "stale probe".to_string(),
        }),
    };

    let replacement = make_worker("http://w:1", 1, 1);
    assert!(registry.replace(&worker_id, replacement));

    let current = registry.get(&worker_id).unwrap();
    assert_eq!(current.status(), WorkerStatus::Ready);
    assert_eq!(current.revision(), expected_revision + 1);

    let result = apply_probe_completion(&registry, completion, None, None).await;
    assert!(matches!(result, ProbeApplyResult::Stale));
    assert_eq!(
        registry.get(&worker_id).unwrap().status(),
        WorkerStatus::Ready
    );
}

/// Spawn a loopback HTTP server stubbing the engine admin endpoints,
/// answering every admin POST with the given status.
async fn spawn_admin_stub(status: StatusCode) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = axum::Router::new()
        .route(
            "/flush_cache",
            axum::routing::post(move || async move { status }),
        )
        .route(
            "/start_profile",
            axum::routing::post(move || async move { status }),
        )
        .route(
            "/stop_profile",
            axum::routing::post(move || async move { status }),
        );
    #[expect(
        clippy::disallowed_methods,
        reason = "test stub server lives for the duration of the test process"
    )]
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn test_flush_cache_all_aggregates_mixed_results() {
    let ok_url = spawn_admin_stub(StatusCode::OK).await;
    let fail_url = spawn_admin_stub(StatusCode::INTERNAL_SERVER_ERROR).await;

    let registry = WorkerRegistry::new();
    registry.register(make_worker(&ok_url, 1, 1)).unwrap();
    registry.register(make_worker(&fail_url, 1, 1)).unwrap();

    let result = WorkerManager::flush_cache_all(&registry).await;

    assert_eq!(result.total_workers, 2);
    assert_eq!(result.http_workers, 2);
    assert_eq!(result.grpc_workers, 0);
    assert_eq!(result.successful, vec![ok_url]);
    assert_eq!(result.failed.len(), 1);
    assert_eq!(result.failed[0].0, fail_url);
    assert!(
        result.failed[0].1.contains("500"),
        "failure reason should carry the HTTP status: {}",
        result.failed[0].1
    );
}

#[tokio::test]
async fn test_flush_cache_all_no_workers() {
    let registry = WorkerRegistry::new();
    let result = WorkerManager::flush_cache_all(&registry).await;
    assert_eq!(result.total_workers, 0);
    assert!(result.successful.is_empty());
    assert!(result.failed.is_empty());
    assert_eq!(result.message, "No workers available for cache flush");
}

#[tokio::test]
async fn test_profile_all_targets_single_worker_via_url_filter() {
    let url_a = spawn_admin_stub(StatusCode::OK).await;
    let url_b = spawn_admin_stub(StatusCode::OK).await;

    let registry = WorkerRegistry::new();
    registry.register(make_worker(&url_a, 1, 1)).unwrap();
    registry.register(make_worker(&url_b, 1, 1)).unwrap();

    let options = ProfileOptions::default();
    let result = WorkerManager::start_profile_all(&registry, &options, Some(url_a.as_str())).await;
    assert_eq!(result.total_workers, 1);
    assert_eq!(result.successful, vec![url_a.clone()]);
    assert!(result.failed.is_empty());
    assert_eq!(
        result.message,
        "Successfully started profiling on all 1 workers"
    );

    let result = WorkerManager::stop_profile_all(&registry, Some(url_a.as_str())).await;
    assert_eq!(result.total_workers, 1);
    assert_eq!(result.successful, vec![url_a]);
    assert_eq!(
        result.message,
        "Successfully stopped profiling on all 1 workers"
    );

    // Neither call finds a worker the fleet does not have, and both say so the same way.
    let none = WorkerManager::stop_profile_all(&registry, Some("http://nowhere:1")).await;
    assert_eq!(none.total_workers, 0);
    assert_eq!(none.message, "No worker matching url 'http://nowhere:1'");
    let empty = WorkerRegistry::new();
    let none = WorkerManager::start_profile_all(&empty, &options, None).await;
    assert_eq!(none.message, "No workers available for profiling");
}

fn load_report(ranks: &[(i32, i32, i32, f64)]) -> WorkerLoadResponse {
    let loads: Vec<SchedulerLoadSnapshot> = ranks
        .iter()
        .map(
            |&(dp_rank, num_running_reqs, num_waiting_reqs, token_usage)| SchedulerLoadSnapshot {
                dp_rank,
                num_running_reqs,
                num_waiting_reqs,
                token_usage,
                ..Default::default()
            },
        )
        .collect();
    WorkerLoadResponse {
        timestamp: "2026-09-03T00:00:00Z".to_string(),
        version: "engine-test".to_string(),
        dp_rank_count: loads.len() as i32,
        aggregate: None,
        loads,
        sampled_at: None,
    }
}

fn make_model_worker(url: &str, model: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .models(vec![ModelCard::new(model)])
            .build(),
    )
}

#[test]
fn fleet_loads_tags_ranks_and_aggregates() {
    let registry = WorkerRegistry::new();
    // Registered out of order to prove the response sorts by URL.
    registry.register(make_worker("http://w:2", 1, 1)).unwrap();
    registry.register(make_worker("http://w:1", 1, 1)).unwrap();

    let snapshot = LoadSnapshot::from_loads_for_test(vec![
        (
            "http://w:1".to_string(),
            load_report(&[(0, 4, 1, 0.25), (1, 6, 3, 0.75)]),
        ),
        ("http://w:2".to_string(), load_report(&[(0, 2, 0, 0.5)])),
    ]);

    let body = WorkerManager::fleet_loads(&registry, &snapshot, None);

    assert_eq!(body.dp_rank_count, 3);
    assert_eq!(body.loads.len(), 3);
    assert!(body.version.starts_with("smg-"));
    assert!(!body.timestamp.is_empty());

    let tagged: Vec<(&str, &str, i32)> = body
        .loads
        .iter()
        .map(|rank| {
            (
                rank.worker.as_deref().expect("worker url"),
                rank.worker_type.as_deref().expect("worker type"),
                rank.dp_rank,
            )
        })
        .collect();
    assert_eq!(
        tagged,
        vec![
            ("http://w:1", "regular", 0),
            ("http://w:1", "regular", 1),
            ("http://w:2", "regular", 0),
        ]
    );

    let aggregate = body.aggregate.expect("aggregate");
    assert_eq!(aggregate.total_running_reqs, 12);
    assert_eq!(aggregate.total_waiting_reqs, 4);
    assert_eq!(aggregate.total_reqs, 16);
    assert_eq!(aggregate.avg_token_usage, 0.5);
}

#[test]
fn fleet_loads_filters_by_model() {
    let registry = WorkerRegistry::new();
    registry
        .register(make_model_worker("http://a:1", "gpt-4o"))
        .unwrap();
    registry
        .register(make_model_worker("http://b:1", "o3"))
        .unwrap();

    let snapshot = LoadSnapshot::from_loads_for_test(vec![
        ("http://a:1".to_string(), load_report(&[(0, 5, 0, 0.5)])),
        ("http://b:1".to_string(), load_report(&[(0, 9, 0, 0.9)])),
    ]);

    let body = WorkerManager::fleet_loads(&registry, &snapshot, Some("gpt-4o"));

    assert_eq!(body.dp_rank_count, 1);
    assert_eq!(body.loads[0].worker.as_deref(), Some("http://a:1"));
    assert_eq!(body.aggregate.expect("aggregate").total_running_reqs, 5);
}

#[test]
fn fleet_loads_omits_workers_without_a_report() {
    let registry = WorkerRegistry::new();
    registry.register(make_worker("http://w:1", 1, 1)).unwrap();
    registry.register(make_worker("http://w:2", 1, 1)).unwrap();

    // Only one worker has been polled: the other must be absent rather
    // than reported as idle.
    let snapshot = LoadSnapshot::from_loads_for_test(vec![(
        "http://w:2".to_string(),
        load_report(&[(0, 7, 2, 0.4)]),
    )]);

    let body = WorkerManager::fleet_loads(&registry, &snapshot, None);

    assert_eq!(body.dp_rank_count, 1);
    assert_eq!(body.loads[0].worker.as_deref(), Some("http://w:2"));
}

#[test]
fn fleet_loads_with_no_polled_workers_has_no_aggregate() {
    let registry = WorkerRegistry::new();
    registry.register(make_worker("http://w:1", 1, 1)).unwrap();

    let body = WorkerManager::fleet_loads(&registry, &LoadSnapshot::default(), None);

    assert_eq!(body.dp_rank_count, 0);
    assert!(body.loads.is_empty());
    assert!(body.aggregate.is_none());
}

/// A fleet where one worker's admin endpoint fails: both calls count the failed worker in
/// their summary and name it in the failed list.
#[tokio::test]
async fn test_profile_all_over_a_mixed_fleet_counts_the_failed() {
    let url_ok = spawn_admin_stub(StatusCode::OK).await;
    let url_failing = spawn_admin_stub(StatusCode::INTERNAL_SERVER_ERROR).await;

    let registry = WorkerRegistry::new();
    registry.register(make_worker(&url_ok, 1, 1)).unwrap();
    registry.register(make_worker(&url_failing, 1, 1)).unwrap();

    let options = ProfileOptions::default();
    let result = WorkerManager::start_profile_all(&registry, &options, None).await;
    assert_eq!(result.total_workers, 2);
    assert_eq!(result.successful, vec![url_ok.clone()]);
    assert_eq!(result.failed.len(), 1);
    assert_eq!(result.failed[0].0, url_failing);
    assert_eq!(result.message, "Profile start: 1 succeeded, 1 failed");

    let result = WorkerManager::stop_profile_all(&registry, None).await;
    assert_eq!(result.total_workers, 2);
    assert_eq!(result.successful, vec![url_ok]);
    assert_eq!(result.failed.len(), 1);
    assert_eq!(result.failed[0].0, url_failing);
    assert_eq!(result.message, "Profile stop: 1 succeeded, 1 failed");
}

#[tokio::test]
async fn test_profile_all_url_filter_without_match() {
    let registry = WorkerRegistry::new();
    registry.register(make_worker("http://w:1", 1, 1)).unwrap();

    let options = ProfileOptions::default();
    let result =
        WorkerManager::start_profile_all(&registry, &options, Some("http://absent:1")).await;
    assert_eq!(result.total_workers, 0);
    assert!(result.successful.is_empty());
    assert!(result.failed.is_empty());
    assert!(result.message.contains("No worker matching"));
}
