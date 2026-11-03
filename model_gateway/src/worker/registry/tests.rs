use std::collections::HashMap;

use openai_protocol::model_card::ModelCard;

use super::*;
use crate::worker::{
    circuit_breaker::{CircuitBreakerConfig, CircuitState},
    test_support::no_health_check,
    BasicWorkerBuilder, WorkerLoadGuard,
};

/// A worker whose breaker opens on one failed request and would stay
/// open for an hour on its own.
fn worker_with_a_hair_trigger_breaker() -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new("http://worker:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 1,
                success_threshold: 1,
                timeout_duration: std::time::Duration::from_secs(3600),
                window_duration: std::time::Duration::from_secs(3600),
            })
            .build(),
    )
}

/// The outage ended before the breaker opened (fewer failures than the
/// threshold): the promotion forgets that streak, on either path, so the
/// first failure after the return does not open the breaker on the
/// outage's account.
#[test]
fn a_promotion_to_ready_forgets_the_failure_streak_of_a_closed_breaker() {
    for by_probe in [true, false] {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .circuit_breaker_config(CircuitBreakerConfig {
                    failure_threshold: 5,
                    success_threshold: 1,
                    timeout_duration: std::time::Duration::from_secs(3600),
                    window_duration: std::time::Duration::from_secs(3600),
                })
                .build(),
        );
        let worker_id = registry.register(worker.clone()).unwrap();
        for _ in 0..4 {
            worker.record_outcome(503);
        }
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
        registry.transition_status(&worker_id, WorkerStatus::NotReady);

        if by_probe {
            registry.apply_if_revision(&worker_id, worker.revision(), |_| {
                ((), Some(WorkerStatus::Ready))
            });
        } else {
            registry.transition_status(&worker_id, WorkerStatus::Ready);
        }
        worker.record_outcome(503);
        assert_eq!(
            worker.circuit_breaker_state(),
            CircuitState::Closed,
            "by_probe={by_probe}: the outage's four failures are forgotten"
        );
        assert!(worker.is_available(), "by_probe={by_probe}");
    }
}

/// The two promotion paths, `apply_if_revision` (the health probes) and
/// `transition_status` (connect signals), both put a promoted worker's
/// open circuit breaker on trial: a worker whose health returned is
/// routable at once, not at the breaker's timeout.
#[test]
fn a_promotion_to_ready_puts_an_open_circuit_breaker_on_trial() {
    for by_probe in [true, false] {
        let registry = WorkerRegistry::new();
        let worker = worker_with_a_hair_trigger_breaker();
        let worker_id = registry.register(worker.clone()).unwrap();

        // The outage: a request fails, the breaker opens, the probes demote.
        worker.record_outcome(503);
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
        assert_eq!(
            registry.transition_status(&worker_id, WorkerStatus::NotReady),
            Some((WorkerStatus::Ready, WorkerStatus::NotReady))
        );
        assert!(!worker.is_available());

        // The return, by either path.
        let transition = if by_probe {
            registry
                .apply_if_revision(&worker_id, worker.revision(), |_| {
                    ((), Some(WorkerStatus::Ready))
                })
                .and_then(|((), transition)| transition)
        } else {
            registry.transition_status(&worker_id, WorkerStatus::Ready)
        };
        assert_eq!(
            transition,
            Some((WorkerStatus::NotReady, WorkerStatus::Ready)),
            "by_probe={by_probe}"
        );
        assert_eq!(
            worker.circuit_breaker_state(),
            CircuitState::HalfOpen,
            "by_probe={by_probe}: the promotion puts the open breaker on trial"
        );
        assert!(worker.is_available(), "by_probe={by_probe}");
    }
}

/// A flapping engine: its probes pass, so it is promoted and its breaker
/// goes on trial; the first failed request reopens the breaker, and
/// nothing short of another demotion and promotion (or the breaker's own
/// timeout) starts another trial.
#[test]
fn a_breaker_reopened_during_the_trial_waits_for_the_next_promotion() {
    let registry = WorkerRegistry::new();
    let worker = worker_with_a_hair_trigger_breaker();
    let worker_id = registry.register(worker.clone()).unwrap();
    worker.record_outcome(503);
    registry.transition_status(&worker_id, WorkerStatus::NotReady);
    registry.transition_status(&worker_id, WorkerStatus::Ready);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);

    // The trial fails.
    worker.record_outcome(503);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    assert!(!worker.is_available());

    // Passing probes on a Ready worker are no promotion: no new trial.
    assert_eq!(
        registry.apply_if_revision(&worker_id, worker.revision(), |_| {
            ((), Some(WorkerStatus::Ready))
        }),
        Some(((), None))
    );
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    assert!(!worker.is_available());

    // The next demotion and promotion earn the next trial.
    registry.transition_status(&worker_id, WorkerStatus::NotReady);
    registry.transition_status(&worker_id, WorkerStatus::Ready);
    assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);
    assert!(worker.is_available());
}

fn worker_with_sampling_defaults(
    url: &str,
    model_id: &str,
    worker_type: WorkerType,
    connection_mode: ConnectionMode,
    defaults: Option<&str>,
) -> Arc<dyn Worker> {
    let mut builder = BasicWorkerBuilder::new(url)
        .model(ModelCard::new(model_id))
        .worker_type(worker_type)
        .connection_mode(connection_mode);

    if let Some(defaults) = defaults {
        let mut labels = HashMap::new();
        labels.insert(
            DEFAULT_SAMPLING_PARAMS_LABEL.to_string(),
            defaults.to_string(),
        );
        builder = builder.labels(labels);
    }

    Arc::new(builder.build())
}

fn worker_with_model_aliases(
    url: &str,
    canonical_id: &str,
    aliases: &[&str],
    worker_type: WorkerType,
) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(canonical_id).with_aliases(aliases.iter().copied()))
            .worker_type(worker_type)
            .health_config(no_health_check())
            .build(),
    )
}

fn assert_sampling_defaults_group_values(
    registry: &WorkerRegistry,
    model_id: &str,
    worker_type: WorkerType,
    expected: &[&str],
) {
    let expected: Vec<String> = expected.iter().map(|value| value.to_string()).collect();
    assert_eq!(
        registry.sampling_defaults_values_for_group(model_id, worker_type),
        expected
    );
}

#[test]
fn connect_signal_receiver_is_taken_exactly_once() {
    let registry = WorkerRegistry::new();
    assert!(
        registry.take_connect_signal_receiver().is_some(),
        "first take (the manager) must get the receiver"
    );
    assert!(
        registry.take_connect_signal_receiver().is_none(),
        "a second take must return None so a second manager runs poll-only"
    );
}

#[tokio::test]
async fn connect_signal_sender_delivers_to_the_receiver() {
    let registry = WorkerRegistry::new();
    let tx = registry.connect_signal_sender();
    let mut rx = registry
        .take_connect_signal_receiver()
        .expect("receiver present");

    tx.send(WorkerConnected {
        url: "ipc:///tmp/w.ipc".to_string(),
        revision: 7,
    })
    .expect("send on a live channel");

    let got = rx.recv().await.expect("a delivered signal");
    assert_eq!(got.url, "ipc:///tmp/w.ipc");
    assert_eq!(got.revision, 7);
}

#[test]
fn has_workers_awaiting_connect_signal_tracks_pending_zmq_workers() {
    let registry = WorkerRegistry::new();
    assert!(
        !registry.has_workers_awaiting_connect_signal(),
        "empty registry awaits nothing"
    );

    // An HTTP worker never waits on the connect signal.
    let http: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w:1")
            .connection_mode(ConnectionMode::Http)
            .build(),
    );
    registry.register(http).unwrap();
    assert!(!registry.has_workers_awaiting_connect_signal());

    // A Pending ZMQ worker is promoted only by the connect signal.
    let zmq: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("ipc:///tmp/w.ipc")
            .connection_mode(ConnectionMode::Zmq)
            .build(),
    );
    assert_eq!(zmq.status(), WorkerStatus::Pending);
    let zmq_url = zmq.url().to_string();
    registry.register(zmq).unwrap();
    assert!(registry.has_workers_awaiting_connect_signal());

    // Once promoted to Ready it no longer awaits the signal.
    registry
        .get_by_url(&zmq_url)
        .unwrap()
        .set_status(WorkerStatus::Ready);
    assert!(!registry.has_workers_awaiting_connect_signal());
}

#[test]
fn test_worker_registry() {
    let registry = WorkerRegistry::new();

    let mut labels = HashMap::new();
    labels.insert("model_id".to_string(), "llama-3-8b".to_string());
    labels.insert("priority".to_string(), "50".to_string());
    labels.insert("cost".to_string(), "0.8".to_string());

    let worker: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .labels(labels)
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .api_key("test_api_key")
            .build(),
    );

    // Register worker
    let worker_id = registry.register(Arc::from(worker)).unwrap();

    assert!(registry.get(&worker_id).is_some());
    assert!(registry.get_by_url("http://worker1:8080").is_some());
    assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
    assert_eq!(registry.get_by_type(WorkerType::Regular).len(), 1);
    assert_eq!(registry.get_by_connection(ConnectionMode::Http).len(), 1);

    let stats = registry.stats();
    assert_eq!(stats.total_workers, 1);
    assert_eq!(stats.total_models, 1);

    registry.remove(&worker_id);
    assert!(registry.get(&worker_id).is_none());
}

#[test]
fn test_stats_counts_encode_workers() {
    let registry = WorkerRegistry::new();

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://encode-worker:8080")
            .worker_type(WorkerType::Encode)
            .connection_mode(ConnectionMode::Grpc)
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .build(),
    );

    registry.register(worker).unwrap();

    let stats = registry.stats();
    assert_eq!(stats.total_workers, 1);
    assert_eq!(stats.encode_workers, 1);
    assert_eq!(stats.prefill_workers, 0);
    assert_eq!(stats.decode_workers, 0);
    assert_eq!(stats.regular_workers, 0);
    assert_eq!(stats.grpc_workers, 1);
    assert_eq!(stats.zmq_workers, 0);
}

#[test]
fn test_stats_counts_zmq_workers_separately_from_grpc() {
    // ZMQ rides the gRPC request pipeline but is its own transport;
    // folding it into grpc_workers hid it from observability output.
    let registry = WorkerRegistry::new();

    for (url, mode) in [
        ("grpc://worker:8080", ConnectionMode::Grpc),
        ("ipc:///tmp/smg-zmq/engine.ipc", ConnectionMode::Zmq),
        ("http://worker:8080", ConnectionMode::Http),
    ] {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .connection_mode(mode)
                .build(),
        );
        registry.register(worker).unwrap();
    }

    let stats = registry.stats();
    assert_eq!(stats.total_workers, 3);
    assert_eq!(stats.http_workers, 1);
    assert_eq!(stats.grpc_workers, 1);
    assert_eq!(stats.zmq_workers, 1);
}

#[test]
fn origin_tracks_local_and_mesh_registrations() {
    let registry = WorkerRegistry::new();

    let local: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://local:8080")
            .model(ModelCard::new("llama-3"))
            .build(),
    );
    let local_id = registry.register(local).unwrap();
    assert_eq!(registry.origin_of(&local_id), Some(WorkerOrigin::Local));

    registry.on_remote_worker_state(&smg_mesh::WorkerState {
        worker_id: "peer-w1".to_string(),
        model_id: "llama-3".to_string(),
        url: "http://remote:8080".to_string(),
        health: true,
        load: 0.0,
        version: 1,
        spec: vec![],
        owner: "peer".to_string(),
    });
    let mesh_id = registry.get_id_by_url("http://remote:8080").unwrap();
    assert_eq!(registry.origin_of(&mesh_id), Some(WorkerOrigin::Mesh));
}

fn remote_state(worker_id: &str, url: &str, health: bool, spec: Vec<u8>) -> smg_mesh::WorkerState {
    smg_mesh::WorkerState {
        worker_id: worker_id.to_string(),
        model_id: "llama-3".to_string(),
        url: url.to_string(),
        health,
        load: 0.0,
        version: 1,
        spec,
        owner: "peer".to_string(),
    }
}

#[test]
fn mesh_import_adopts_publisher_worker_id() {
    let registry = WorkerRegistry::new();
    registry.on_remote_worker_state(&remote_state("peer-w1", "http://remote:8080", true, vec![]));
    assert_eq!(
        registry.get_id_by_url("http://remote:8080"),
        Some(WorkerId::from_string("peer-w1".to_string())),
        "import keys under the publisher's id so its tombstone resolves"
    );
}

#[test]
fn mesh_state_for_zmq_worker_is_never_imported() {
    // ZMQ is same-host: an `ipc://` endpoint published by a peer names a
    // socket path on that peer's machine, so importing it would advertise
    // an unroutable worker.
    let registry = WorkerRegistry::new();
    registry.on_remote_worker_state(&remote_state(
        "peer-w1",
        "ipc:///tmp/smg-peer.sock",
        true,
        vec![],
    ));
    assert!(
        registry.get_by_url("ipc:///tmp/smg-peer.sock").is_none(),
        "a host-local ZMQ worker must not be imported from the mesh"
    );

    // Same rejection when the transport is declared only by the spec.
    let spec: openai_protocol::worker::WorkerSpec = serde_json::from_value(serde_json::json!({
        "url": "http://remote:8080",
        "connection_mode": "zmq"
    }))
    .unwrap();
    registry.on_remote_worker_state(&remote_state(
        "peer-w2",
        "http://remote:8080",
        true,
        serde_json::to_vec(&spec).unwrap(),
    ));
    assert!(
        registry.get_by_url("http://remote:8080").is_none(),
        "a spec-declared ZMQ worker must not be imported from the mesh"
    );
}

#[test]
fn rejected_zmq_state_leaves_no_url_to_id_residue() {
    // Both transport gates run before the id reservation. A leftover
    // `url_to_id` entry would be invisible to `get_id_by_url` (which
    // skips ids with no live worker) yet still win the `Entry::Occupied`
    // arm in `register_inner`, handing the next legitimate worker at
    // this URL the rejected publisher's id — so a peer tombstone for
    // that id would delete a worker that never came from the mesh.
    let registry = WorkerRegistry::new();

    // Rejected by the URL scheme.
    registry.on_remote_worker_state(&remote_state(
        "peer-w1",
        "ipc:///tmp/smg-peer.sock",
        true,
        vec![],
    ));
    assert!(
        registry.url_to_id.get("ipc:///tmp/smg-peer.sock").is_none(),
        "a URL-scheme rejection must not reserve an id"
    );

    // Rejected by the spec's declared connection mode.
    let spec: openai_protocol::worker::WorkerSpec = serde_json::from_value(serde_json::json!({
        "url": "http://remote:8080",
        "connection_mode": "zmq"
    }))
    .unwrap();
    registry.on_remote_worker_state(&remote_state(
        "peer-w2",
        "http://remote:8080",
        true,
        serde_json::to_vec(&spec).unwrap(),
    ));
    assert!(
        registry.url_to_id.get("http://remote:8080").is_none(),
        "a spec rejection must not reserve an id"
    );

    // A legitimate worker later arriving at the same URL gets its own id.
    let local: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://remote:8080")
            .model(ModelCard::new("llama-3"))
            .build(),
    );
    let local_id = registry.register(local).expect("registers");
    assert_ne!(
        local_id,
        WorkerId::from_string("peer-w2".to_string()),
        "a later worker must not inherit the rejected publisher's id"
    );
    assert_eq!(registry.get_id_by_url("http://remote:8080"), Some(local_id));
}

#[test]
fn mesh_state_never_mutates_locally_owned_worker() {
    let registry = WorkerRegistry::new();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://local:8080")
            .model(ModelCard::new("m"))
            .build(),
    );
    let id = registry.register(worker.clone()).unwrap();
    worker.set_status(WorkerStatus::Ready);

    registry.on_remote_worker_state(&remote_state("peer-x", "http://local:8080", false, vec![]));
    assert_eq!(
        registry.get(&id).unwrap().status(),
        WorkerStatus::Ready,
        "a peer's echo must not demote a locally-owned worker"
    );
}

#[test]
fn unhealthy_import_with_disabled_health_check_is_not_ready() {
    // The builder defaults disable_health_check workers to Ready; an
    // explicitly-unhealthy import must still land unroutable.
    let spec: openai_protocol::worker::WorkerSpec = serde_json::from_value(serde_json::json!({
        "url": "http://remote:8080",
        "health": { "disable_health_check": true }
    }))
    .unwrap();
    let registry = WorkerRegistry::new();
    registry.on_remote_worker_state(&remote_state(
        "peer-w1",
        "http://remote:8080",
        false,
        serde_json::to_vec(&spec).unwrap(),
    ));
    let worker = registry.get_by_url("http://remote:8080").expect("imported");
    assert_ne!(
        worker.status(),
        WorkerStatus::Ready,
        "an explicitly-unhealthy import must not be routable"
    );
}

#[test]
fn stale_healthy_state_never_resurrects_probe_failed_import() {
    // A dead owner's key keeps health=true forever; the periodic
    // reconcile replays it. A probe-failed import must stay failed,
    // not flap back into rotation every pass.
    let registry = WorkerRegistry::new();
    let state = remote_state("peer-w1", "http://remote:8080", true, vec![]);
    registry.on_remote_worker_state(&state);
    let worker = registry.get_by_url("http://remote:8080").unwrap();
    assert_eq!(worker.status(), WorkerStatus::Ready);

    worker.set_status(WorkerStatus::Failed);
    registry.on_remote_worker_state(&state);
    assert_eq!(
        registry.get_by_url("http://remote:8080").unwrap().status(),
        WorkerStatus::Failed,
        "mesh hints must not resurrect probe-owned terminal states"
    );

    // NotReady is still promotable: the owner saying healthy again
    // is the legitimate recovery signal.
    worker.set_status(WorkerStatus::NotReady);
    registry.on_remote_worker_state(&state);
    assert_eq!(
        registry.get_by_url("http://remote:8080").unwrap().status(),
        WorkerStatus::Ready
    );
}

/// One mesh health hint against one status written in between. The
/// import is registered healthy or not (`imported_healthy`), the hint
/// says `hint_health`, and `written_between` lands through the registry
/// after the hint looked at the import and before it writes. The hint
/// reads the status and decides under the worker's mutation lock, in
/// one step with its write, so the status written in between stands
/// and the only event is its own.
fn mesh_hint_keeps_the_status_written_in_between(
    hint_health: bool,
    imported_healthy: bool,
    written_between: WorkerStatus,
) {
    let url = "http://remote:8080";
    let registry = WorkerRegistry::new();
    registry.on_remote_worker_state(&remote_state("peer-w1", url, imported_healthy, vec![]));
    let worker = registry.get_by_url(url).expect("imported");
    let worker_id = registry.get_id_by_url(url).expect("imported");
    let found = worker.status();
    assert_eq!(
        found,
        if imported_healthy {
            WorkerStatus::Ready
        } else {
            WorkerStatus::NotReady
        }
    );
    let revision = worker.revision();
    let mut rx = registry.subscribe_events();

    let hint = remote_state("peer-w1", url, hint_health, vec![]);
    registry.with_another_write_before_the_lock(
        |registry| registry.on_remote_worker_state(&hint),
        |registry| {
            let ((), transition) = registry
                .apply_if_revision(&worker_id, revision, |_| ((), Some(written_between)))
                .expect("the import is there and current");
            assert_eq!(transition, Some((found, written_between)));
        },
    );

    assert_eq!(
        worker.status(),
        written_between,
        "the status written in between stands"
    );
    match rx.try_recv() {
        Ok(WorkerEvent::StatusChanged {
            old_status,
            new_status,
            ..
        }) => assert_eq!(
            (old_status, new_status),
            (found, written_between),
            "the in-between write's event"
        ),
        other => panic!("expected the in-between write's StatusChanged, got {other:?}"),
    }
    assert!(rx.try_recv().is_err(), "the hint published nothing");
}

/// A `health=true` hint that found a mesh import NotReady does not write
/// `Ready` over the `Failed` a probe reached in between, and publishes
/// nothing. Decided outside the lock, the hint wrote `Failed -> Ready`
/// here: a probe-failed import came back into rotation on a stale hint.
#[test]
fn mesh_hint_keeps_a_probe_failure_written_between_its_read_and_its_write() {
    mesh_hint_keeps_the_status_written_in_between(true, false, WorkerStatus::Failed);
}

/// A `health=false` hint that found a mesh import Ready does not write
/// `NotReady` over the `Draining` the drain step wrote in between, and
/// publishes nothing. Decided outside the lock, the hint wrote
/// `Draining -> NotReady` here.
#[test]
fn mesh_hint_keeps_a_drain_written_between_its_read_and_its_write() {
    mesh_hint_keeps_the_status_written_in_between(false, true, WorkerStatus::Draining);
}

/// A `health=true` hint that found a mesh import `NotReady` does not
/// write `Ready` over the local worker a `register_or_replace` installed
/// at the URL in between: the claim replaces the worker under the same
/// mutation lock and promotes its origin to `Local`, which the hint reads
/// again under the lock and stands down. Checked before the lock alone,
/// the hint promoted a locally-owned worker no local probe had passed.
#[test]
fn mesh_hint_stands_down_for_a_local_claim_landed_between_its_read_and_its_write() {
    let url = "http://remote:8080";
    let registry = WorkerRegistry::new();
    registry.on_remote_worker_state(&remote_state("peer-w1", url, false, vec![]));
    let worker_id = registry.get_id_by_url(url).expect("imported");
    assert_eq!(registry.origin_of(&worker_id), Some(WorkerOrigin::Mesh));
    let mut rx = registry.subscribe_events();

    let hint = remote_state("peer-w1", url, true, vec![]);
    // The status the claim leaves the local worker with: not `Ready`
    // (its own prober promotes it), so the hint would promote it if it
    // still looked.
    let after_claim = std::cell::Cell::new(None);
    registry.with_another_write_before_the_lock(
        |registry| registry.on_remote_worker_state(&hint),
        |registry| {
            let local: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(url)
                    .worker_type(WorkerType::Regular)
                    .build(),
            );
            assert_eq!(
                registry.register_or_replace(local),
                worker_id,
                "the claim keeps the URL's id"
            );
            assert_eq!(registry.origin_of(&worker_id), Some(WorkerOrigin::Local));
            let status = registry.get_by_url(url).expect("the local worker").status();
            assert_ne!(status, WorkerStatus::Ready, "the local prober promotes it");
            after_claim.set(Some(status));
        },
    );

    assert_eq!(registry.origin_of(&worker_id), Some(WorkerOrigin::Local));
    assert_eq!(
        Some(registry.get_by_url(url).expect("the local worker").status()),
        after_claim.get(),
        "the hint did not promote the locally-owned worker"
    );
    while let Ok(event) = rx.try_recv() {
        assert!(
            !matches!(event, WorkerEvent::StatusChanged { .. }),
            "the hint published nothing; the events are the claim's own: {event:?}"
        );
    }
}

#[test]
fn remove_remote_only_removes_mesh_origin_workers() {
    let registry = WorkerRegistry::new();

    registry.on_remote_worker_state(&remote_state("peer-w1", "http://remote:8080", true, vec![]));
    let mesh_id = registry.get_id_by_url("http://remote:8080").unwrap();
    assert!(registry.remove_remote(&mesh_id).is_some());
    assert!(registry.get_by_url("http://remote:8080").is_none());

    let local: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://local:8080")
            .model(ModelCard::new("m"))
            .build(),
    );
    let local_id = registry.register(local).unwrap();
    assert!(registry.remove_remote(&local_id).is_none());
    assert!(
        registry.get(&local_id).is_some(),
        "a locally-owned worker survives a remote tombstone"
    );
}

#[test]
fn register_or_replace_promotes_mesh_origin_to_local() {
    // Restart race: a mesh import wins the URL before the local
    // workflow registers it. The local claim must take ownership back,
    // else the node never publishes its own worker and a peer tombstone
    // could delete it.
    let registry = WorkerRegistry::new();
    registry.on_remote_worker_state(&remote_state("peer-w1", "http://w:8080", true, vec![]));
    let id = registry.get_id_by_url("http://w:8080").unwrap();
    assert_eq!(registry.origin_of(&id), Some(WorkerOrigin::Mesh));

    let local: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w:8080")
            .model(ModelCard::new("llama-3"))
            .build(),
    );
    let claimed_id = registry.register_or_replace(local);
    assert_eq!(claimed_id, id, "claim reuses the adopted id");
    assert_eq!(
        registry.origin_of(&id),
        Some(WorkerOrigin::Local),
        "local registration over a mesh import takes ownership"
    );
    assert!(
        registry.remove_remote(&id).is_none(),
        "a peer tombstone can no longer delete the claimed worker"
    );
}

#[test]
fn remove_inner_aborts_when_origin_changed_before_lock() {
    // The TOCTOU guard: remove_remote's pre-check can observe Mesh,
    // then lose the lock race to a local claim that promotes the
    // origin. The under-lock recheck must abort the removal.
    let registry = WorkerRegistry::new();
    let local: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w:8080")
            .model(ModelCard::new("m"))
            .build(),
    );
    let id = registry.register(local).unwrap();

    // Models the post-promotion state: expecting Mesh, finding Local.
    assert!(
        registry
            .remove_inner(&id, Some(WorkerOrigin::Mesh))
            .is_none(),
        "removal must abort when the origin no longer matches"
    );
    assert!(
        registry.get(&id).is_some(),
        "the claimed worker survives the racing tombstone"
    );
}

#[test]
fn origin_survives_replace_and_clears_on_remove() {
    let registry = WorkerRegistry::new();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w:8080")
            .model(ModelCard::new("m"))
            .build(),
    );
    let id = registry.register(worker).unwrap();

    let replacement: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w:8080")
            .model(ModelCard::new("m2"))
            .build(),
    );
    assert!(registry.replace(&id, replacement));
    assert_eq!(
        registry.origin_of(&id),
        Some(WorkerOrigin::Local),
        "replace keeps the same id, so origin is untouched"
    );

    registry.remove(&id);
    assert_eq!(registry.origin_of(&id), None, "remove clears the origin");
}

#[test]
fn test_model_index_fast_lookup() {
    let registry = WorkerRegistry::new();

    let mut labels1 = HashMap::new();
    labels1.insert("model_id".to_string(), "llama-3".to_string());
    let worker1: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .labels(labels1)
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .api_key("test_api_key")
            .build(),
    );

    let mut labels2 = HashMap::new();
    labels2.insert("model_id".to_string(), "llama-3".to_string());
    let worker2: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://worker2:8080")
            .worker_type(WorkerType::Regular)
            .labels(labels2)
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .api_key("test_api_key")
            .build(),
    );

    let mut labels3 = HashMap::new();
    labels3.insert("model_id".to_string(), "gpt-4".to_string());
    let worker3: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://worker3:8080")
            .worker_type(WorkerType::Regular)
            .labels(labels3)
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .api_key("test_api_key")
            .build(),
    );

    // Register workers
    registry.register(Arc::from(worker1)).unwrap();
    registry.register(Arc::from(worker2)).unwrap();
    registry.register(Arc::from(worker3)).unwrap();

    let llama_workers = registry.get_by_model("llama-3");
    assert_eq!(llama_workers.len(), 2);
    let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
    assert!(urls.contains(&"http://worker1:8080".to_string()));
    assert!(urls.contains(&"http://worker2:8080".to_string()));

    let gpt_workers = registry.get_by_model("gpt-4");
    assert_eq!(gpt_workers.len(), 1);
    assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

    let unknown_workers = registry.get_by_model("unknown-model");
    assert_eq!(unknown_workers.len(), 0);

    registry.remove_by_url("http://worker1:8080");
    let llama_workers_after = registry.get_by_model("llama-3");
    assert_eq!(llama_workers_after.len(), 1);
    assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
}

#[test]
fn routing_pool_reuses_uniform_model_snapshot() {
    let registry = WorkerRegistry::new();
    for url in ["http://worker1:8080", "http://worker2:8080"] {
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new("llama-3"))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Http)
                    .health_config(no_health_check())
                    .build(),
            ))
            .unwrap();
    }

    let all = registry.get_by_model("llama-3");
    let first = registry.get_routing_pool("llama-3", RoutingPool::HttpRegular);
    let second = registry.get_routing_pool("llama-3", RoutingPool::HttpRegular);

    assert!(Arc::ptr_eq(&all, &first));
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn routing_pool_is_cached_and_replaced_with_membership_snapshot() {
    let registry = WorkerRegistry::new();
    let http_id = registry
        .register(Arc::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .model(ModelCard::new("llama-3"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .health_config(no_health_check())
                .build(),
        ))
        .unwrap();
    registry
        .register(Arc::new(
            BasicWorkerBuilder::new("grpc://worker2:8080")
                .model(ModelCard::new("llama-3"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        ))
        .unwrap();

    let first = registry.get_routing_pool("llama-3", RoutingPool::HttpRegular);
    let second = registry.get_routing_pool("llama-3", RoutingPool::HttpRegular);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].url(), "http://worker1:8080");
    assert!(Arc::ptr_eq(&first, &second));

    registry
        .transition_status(&http_id, WorkerStatus::NotReady)
        .unwrap();
    let after_health_change = registry.get_routing_pool("llama-3", RoutingPool::HttpRegular);
    assert!(Arc::ptr_eq(&first, &after_health_change));
    assert!(!after_health_change[0].is_available());

    registry.remove(&http_id).unwrap();
    let after_remove = registry.get_routing_pool("llama-3", RoutingPool::HttpRegular);
    assert!(after_remove.is_empty());
    assert!(!Arc::ptr_eq(&first, &after_remove));
    assert_eq!(first.len(), 1, "held snapshots remain immutable");
}

#[test]
fn wildcard_routing_pool_reuses_global_snapshot() {
    let registry = WorkerRegistry::new();
    let http_worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker1:8080")
            .model(ModelCard::new("model-a"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .health_config(no_health_check())
            .build(),
    );
    let http_id = registry.register(http_worker.clone()).unwrap();
    registry
        .register(Arc::new(
            BasicWorkerBuilder::new("grpc://worker2:8080")
                .model(ModelCard::new("model-b"))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        ))
        .unwrap();

    let before = Arc::strong_count(&http_worker);
    let first = registry.get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::HttpRegular);
    let after_first = Arc::strong_count(&http_worker);
    let second = registry.get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::HttpRegular);

    assert_eq!(first.len(), 1);
    assert_eq!(first[0].url(), "http://worker1:8080");
    assert!(Arc::ptr_eq(&first, &second));
    assert!(after_first > before);
    assert_eq!(Arc::strong_count(&http_worker), after_first);

    registry
        .transition_status(&http_id, WorkerStatus::NotReady)
        .unwrap();
    let after_health_change = registry.get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::HttpRegular);
    assert!(Arc::ptr_eq(&first, &after_health_change));
    assert!(!after_health_change[0].is_available());

    let replacement: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker1:8080")
            .model(ModelCard::new("model-a"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .build(),
    );
    assert!(registry.replace(&http_id, replacement));
    let after_replace = registry.get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::HttpRegular);
    assert!(after_replace.is_empty());
    assert!(!Arc::ptr_eq(&first, &after_replace));
    assert_eq!(
        registry
            .get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::GrpcPipelineRegular)
            .len(),
        2
    );
    assert_eq!(first.len(), 1, "held snapshots remain immutable");
}

#[test]
fn concurrent_global_snapshot_updates_preserve_all_workers() {
    let registry = Arc::new(WorkerRegistry::new());
    std::thread::scope(|scope| {
        for i in 0..8 {
            let registry = Arc::clone(&registry);
            scope.spawn(move || {
                registry
                    .register(Arc::new(
                        BasicWorkerBuilder::new(format!("http://worker{i}:8080"))
                            .worker_type(WorkerType::Regular)
                            .connection_mode(ConnectionMode::Http)
                            .health_config(no_health_check())
                            .build(),
                    ))
                    .unwrap();
            });
        }
    });

    let workers = registry.get_routing_pool(UNKNOWN_MODEL_ID, RoutingPool::HttpRegular);
    assert_eq!(workers.len(), 8);
    assert_eq!(
        workers
            .iter()
            .map(|worker| worker.url())
            .collect::<HashSet<_>>()
            .len(),
        8
    );
}

/// The writer replaces the prefill worker before the decode worker on
/// every generation, so in every membership state that ever truly
/// existed, generation(prefill) >= generation(decode). A rebuild scan
/// that interleaved with the writer could publish the reverse — an old
/// prefill paired with a newer decode — which is exactly the torn pair a
/// live PD selection must never draw from.
#[test]
fn global_snapshot_never_publishes_a_torn_generation() {
    use std::sync::atomic::AtomicBool;

    fn pd_worker(url: &str, worker_type: WorkerType, generation: usize) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("model"))
                .worker_type(worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .label("generation", generation.to_string())
                .health_config(no_health_check())
                .build(),
        )
    }

    fn generation(pool: &[Arc<dyn Worker>]) -> usize {
        pool[0]
            .metadata()
            .spec
            .labels
            .get("generation")
            .and_then(|generation| generation.parse().ok())
            .expect("every test worker carries a generation label")
    }

    let registry = WorkerRegistry::new();
    registry.register_or_replace(pd_worker("grpc://prefill:1", WorkerType::Prefill, 0));
    registry.register_or_replace(pd_worker("grpc://decode:1", WorkerType::Decode, 0));

    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for generation in 1..=300 {
                registry.register_or_replace(pd_worker(
                    "grpc://prefill:1",
                    WorkerType::Prefill,
                    generation,
                ));
                registry.register_or_replace(pd_worker(
                    "grpc://decode:1",
                    WorkerType::Decode,
                    generation,
                ));
            }
            done.store(true, Ordering::Release);
        });
        scope.spawn(|| {
            while !done.load(Ordering::Acquire) {
                let snapshot = registry.get_routing_snapshot(UNKNOWN_MODEL_ID);
                let prefill = snapshot.pool(RoutingPool::GrpcPrefill);
                let decode = snapshot.pool(RoutingPool::GrpcDecode);
                if prefill.is_empty() || decode.is_empty() {
                    continue;
                }
                assert!(
                    generation(&prefill) >= generation(&decode),
                    "torn snapshot: decode generation passed prefill"
                );
            }
        });
    });
}

#[test]
fn routing_pools_filter_regular_workers_by_transport() {
    let registry = WorkerRegistry::new();
    let (sglang, tokenspeed) = (RuntimeType::Sglang, RuntimeType::TokenSpeed);
    let workers = [
        (
            "http://regular",
            WorkerType::Regular,
            ConnectionMode::Http,
            sglang,
        ),
        (
            "grpc://regular",
            WorkerType::Regular,
            ConnectionMode::Grpc,
            sglang,
        ),
        (
            "zmq://regular",
            WorkerType::Regular,
            ConnectionMode::Zmq,
            sglang,
        ),
        (
            "http://prefill",
            WorkerType::Prefill,
            ConnectionMode::Http,
            sglang,
        ),
        (
            "grpc://prefill",
            WorkerType::Prefill,
            ConnectionMode::Grpc,
            sglang,
        ),
        (
            "zmq://prefill",
            WorkerType::Prefill,
            ConnectionMode::Zmq,
            sglang,
        ),
        (
            "zmq://ts-prefill",
            WorkerType::Prefill,
            ConnectionMode::Zmq,
            tokenspeed,
        ),
        (
            "http://decode",
            WorkerType::Decode,
            ConnectionMode::Http,
            sglang,
        ),
        (
            "grpc://decode",
            WorkerType::Decode,
            ConnectionMode::Grpc,
            sglang,
        ),
        (
            "zmq://ts-decode",
            WorkerType::Decode,
            ConnectionMode::Zmq,
            tokenspeed,
        ),
        (
            "grpc://encode",
            WorkerType::Encode,
            ConnectionMode::Grpc,
            sglang,
        ),
        (
            "http://encode",
            WorkerType::Encode,
            ConnectionMode::Http,
            sglang,
        ),
        (
            "zmq://ts-encode",
            WorkerType::Encode,
            ConnectionMode::Zmq,
            tokenspeed,
        ),
    ];
    for (url, worker_type, connection_mode, runtime_type) in workers {
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new("model"))
                    .worker_type(worker_type)
                    .connection_mode(connection_mode)
                    .runtime_type(runtime_type)
                    .health_config(no_health_check())
                    .build(),
            ))
            .unwrap();
    }

    let urls = |pool| {
        registry
            .get_routing_pool("model", pool)
            .iter()
            .map(|worker| worker.url().to_string())
            .collect::<HashSet<_>>()
    };

    assert_eq!(
        urls(RoutingPool::HttpRegular),
        HashSet::from(["http://regular".to_string()])
    );
    assert_eq!(
        urls(RoutingPool::GrpcPipelineRegular),
        HashSet::from(["grpc://regular".to_string(), "zmq://regular".to_string()])
    );
    // The disaggregated pools hold gRPC workers and TokenSpeed's direct-ZMQ
    // engines, whose wire carries the rendezvous and the encode request:
    // another runtime's ZMQ prefill worker that slipped past registration
    // must not appear.
    assert_eq!(
        urls(RoutingPool::GrpcPrefill),
        HashSet::from(["grpc://prefill".to_string(), "zmq://ts-prefill".to_string()])
    );
    assert_eq!(
        urls(RoutingPool::GrpcDecode),
        HashSet::from(["grpc://decode".to_string(), "zmq://ts-decode".to_string()])
    );
    assert_eq!(
        urls(RoutingPool::GrpcEncode),
        HashSet::from(["grpc://encode".to_string(), "zmq://ts-encode".to_string()])
    );
    // The HTTP PD pools are HTTP-only: the router proxies plain HTTP to
    // the selected URL, so gRPC and ZMQ workers must never be selectable.
    assert_eq!(
        urls(RoutingPool::HttpPrefill),
        HashSet::from(["http://prefill".to_string()])
    );
    assert_eq!(
        urls(RoutingPool::HttpDecode),
        HashSet::from(["http://decode".to_string()])
    );
}

#[test]
fn routing_pool_alias_reuses_canonical_snapshot() {
    let registry = WorkerRegistry::new();
    registry
        .register(worker_with_model_aliases(
            "http://worker:8080",
            "canonical-model",
            &["model-alias"],
            WorkerType::Regular,
        ))
        .unwrap();

    let canonical = registry.get_routing_pool("canonical-model", RoutingPool::HttpRegular);
    let alias = registry.get_routing_pool("model-alias", RoutingPool::HttpRegular);
    assert!(Arc::ptr_eq(&canonical, &alias));
}

// Health-checker integration tests moved to worker/manager.rs along with
// the loop itself. The registry is now a pure collection — see
// `worker::manager::WorkerManager` tests.

#[test]
fn test_transition_status_emits_event_and_changes_status() {
    let registry = WorkerRegistry::new();
    let mut rx = registry.subscribe_events();

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w1:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build(),
    );
    let worker_id = registry.register(worker.clone()).unwrap();
    // Drain Registered event
    let _ = rx.try_recv().unwrap();

    // Initial status is Ready (disable_health_check). Transition to NotReady.
    let result = registry.transition_status(&worker_id, WorkerStatus::NotReady);
    assert_eq!(result, Some((WorkerStatus::Ready, WorkerStatus::NotReady)));
    assert_eq!(worker.status(), WorkerStatus::NotReady);

    let event = rx.try_recv().unwrap();
    match event {
        WorkerEvent::StatusChanged {
            old_status,
            new_status,
            ..
        } => {
            assert_eq!(old_status, WorkerStatus::Ready);
            assert_eq!(new_status, WorkerStatus::NotReady);
        }
        other => panic!("Expected StatusChanged, got {other:?}"),
    }
}

#[test]
fn test_transition_status_no_op_when_status_unchanged() {
    let registry = WorkerRegistry::new();
    let mut rx = registry.subscribe_events();

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w1:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build(),
    );
    let worker_id = registry.register(worker).unwrap();
    let _ = rx.try_recv().unwrap();

    // Already Ready — transition to Ready is a no-op
    assert_eq!(
        registry.transition_status(&worker_id, WorkerStatus::Ready),
        None
    );
    assert!(rx.try_recv().is_err(), "no event should be emitted");
}

#[test]
fn test_transition_status_returns_none_for_missing_worker() {
    let registry = WorkerRegistry::new();
    let missing = WorkerId::from_string("nonexistent".to_string());
    assert_eq!(
        registry.transition_status(&missing, WorkerStatus::Ready),
        None
    );
}

#[test]
fn test_get_id_by_url_returns_id_for_registered_worker() {
    let registry = WorkerRegistry::new();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w-by-url:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build(),
    );
    let worker_id = registry.register(worker).unwrap();
    assert_eq!(
        registry.get_id_by_url("http://w-by-url:8080"),
        Some(worker_id)
    );
}

#[test]
fn test_get_id_by_url_returns_none_for_unknown_url() {
    let registry = WorkerRegistry::new();
    assert!(registry.get_id_by_url("http://missing:8080").is_none());
}

#[test]
fn test_url_lookup_canonicalizes_bare_host_port_to_http_form() {
    let registry = WorkerRegistry::new();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://canon:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build(),
    );
    let worker_id = registry.register(worker).unwrap();

    assert_eq!(registry.get_id_by_url("canon:8080"), Some(worker_id));
    assert!(registry.get_by_url("canon:8080").is_some());
}

#[test]
fn test_url_lookup_canonicalizes_bare_host_port_to_grpc_form() {
    let registry = WorkerRegistry::new();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://canon-grpc:8080")
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .build(),
    );
    let worker_id = registry.register(worker).unwrap();

    assert_eq!(registry.get_id_by_url("canon-grpc:8080"), Some(worker_id));
    assert!(registry.get_by_url("canon-grpc:8080").is_some());
}

#[test]
fn test_remove_by_url_canonicalizes_bare_host_port() {
    let registry = WorkerRegistry::new();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://to-remove:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build(),
    );
    let worker_id = registry.register(worker).unwrap();

    assert!(registry.remove_by_url("to-remove:8080").is_some());
    assert!(registry.get(&worker_id).is_none());
    assert!(registry.get_by_url("http://to-remove:8080").is_none());
}

#[test]
fn test_url_lookup_does_not_cross_match_explicit_schemes() {
    let registry = WorkerRegistry::new();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://strict:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build(),
    );
    registry.register(worker).unwrap();

    assert!(registry.get_by_url("grpc://strict:8080").is_none());
    assert!(registry.get_id_by_url("grpcs://strict:8080").is_none());
}

#[test]
fn test_url_lookup_returns_none_for_bare_host_port_with_no_match() {
    let registry = WorkerRegistry::new();
    assert!(registry.get_by_url("nothing:8080").is_none());
    assert!(registry.get_id_by_url("nothing:8080").is_none());
    assert!(registry.remove_by_url("nothing:8080").is_none());
}

/// Regression for the bare-URL reservation shadowing the canonical
/// scheme-prefixed entry. `WorkerService::create_worker` reserves an
/// ID against `config.url` *before* `normalize_url` runs in the
/// AddWorker workflow, so when service discovery submits bare
/// `host:port` the registry briefly holds two `url_to_id` entries:
/// the bare reservation (no live worker) and the canonical
/// scheme-prefixed entry (live worker). Lookups for the bare URL
/// must skip the reservation and resolve to the live worker.
#[test]
fn test_url_lookup_skips_orphan_reservation_for_bare_host_port() {
    let registry = WorkerRegistry::new();
    let _reserved = registry.reserve_id_for_url("orphan:8080");

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://orphan:8080")
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .build(),
    );
    let live_id = registry.register(worker).unwrap();

    assert_eq!(registry.get_id_by_url("orphan:8080"), Some(live_id.clone()));
    assert!(registry
        .get_by_url("orphan:8080")
        .is_some_and(|w| w.url() == "grpc://orphan:8080"));
    assert!(registry.remove_by_url("orphan:8080").is_some());
    assert!(registry.get(&live_id).is_none());
}

#[test]
fn test_url_lookup_returns_none_when_only_reservation_exists() {
    let registry = WorkerRegistry::new();
    registry.reserve_id_for_url("pending:8080");
    assert!(registry.get_by_url("pending:8080").is_none());
    assert!(registry.get_id_by_url("pending:8080").is_none());
    assert!(registry.remove_by_url("pending:8080").is_none());
}

/// `transition_status_inner` (the shared backend used by both
/// `transition_status` and `transition_status_if_revision`) emits a
/// single `WorkerEvent::StatusChanged` event for every status mutation,
/// regardless of which target status is being installed. The mesh
/// adapter subscribes to that event stream, so this proves Draining
/// transitions propagate through the same path as Ready/NotReady.
#[test]
fn test_transition_to_draining_emits_status_changed_event() {
    let registry = WorkerRegistry::new();
    let mut rx = registry.subscribe_events();

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w-drain:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build(),
    );
    let worker_id = registry.register(worker.clone()).unwrap();
    let _ = rx.try_recv().unwrap();

    let result = registry.transition_status(&worker_id, WorkerStatus::Draining);
    assert_eq!(result, Some((WorkerStatus::Ready, WorkerStatus::Draining)));
    assert_eq!(worker.status(), WorkerStatus::Draining);

    match rx.try_recv().unwrap() {
        WorkerEvent::StatusChanged {
            old_status,
            new_status,
            ..
        } => {
            assert_eq!(old_status, WorkerStatus::Ready);
            assert_eq!(new_status, WorkerStatus::Draining);
        }
        other => panic!("expected StatusChanged, got {other:?}"),
    }
}

/// The primitives section's transition table, walked through the one
/// write. Whichever public path carries it (`transition_status`, the
/// revision-checked `transition_status_if_revision`, a probe's
/// `apply_if_revision`), an edge between two different statuses lands
/// as one status write and one `StatusChanged` naming it, for every
/// pair of statuses: the registry has no opinion on which edges a
/// writer may take. The same status again is no write and no event. A
/// stale revision or a worker that is gone is no write and no event,
/// and a stale probe's closure does not run. The removal's demotion
/// writes NotReady over Ready, nothing over any other status, and
/// publishes nothing beside `Removed`.
#[test]
fn every_status_edge_lands_as_one_write_and_one_event_through_any_path() {
    use WorkerStatus::{Draining, Failed, NotReady, Pending, Ready};
    const STATUSES: [WorkerStatus; 5] = [Pending, Ready, NotReady, Failed, Draining];

    #[derive(Clone, Copy, Debug)]
    enum Path {
        Plain,
        IfRevision,
        Probe,
    }

    fn fresh() -> (
        WorkerRegistry,
        broadcast::Receiver<WorkerEvent>,
        Arc<dyn Worker>,
        WorkerId,
    ) {
        let registry = WorkerRegistry::new();
        let mut rx = registry.subscribe_events();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        );
        let worker_id = registry.register(worker.clone()).unwrap();
        assert!(matches!(rx.try_recv(), Ok(WorkerEvent::Registered { .. })));
        (registry, rx, worker, worker_id)
    }

    fn write(
        registry: &WorkerRegistry,
        path: Path,
        worker_id: &WorkerId,
        revision: u64,
        to: WorkerStatus,
    ) -> Option<(WorkerStatus, WorkerStatus)> {
        match path {
            Path::Plain => registry.transition_status(worker_id, to),
            Path::IfRevision => registry.transition_status_if_revision(worker_id, revision, to),
            Path::Probe => {
                let (tag, transition) = registry
                    .apply_if_revision(worker_id, revision, |_| ("probe", Some(to)))
                    .expect("the worker is there and current");
                assert_eq!(tag, "probe", "the probe's own result comes back");
                transition
            }
        }
    }

    // Every edge, through every path.
    for path in [Path::Plain, Path::IfRevision, Path::Probe] {
        for from in STATUSES {
            for to in STATUSES {
                let (registry, mut rx, worker, worker_id) = fresh();
                worker.set_status(from);
                let case = format!("{path:?} {from:?} -> {to:?}");

                let transition = write(&registry, path, &worker_id, worker.revision(), to);

                assert_eq!(worker.status(), to, "{case}: the status is written");
                if from == to {
                    assert_eq!(transition, None, "{case}: the same status is no transition");
                    assert!(rx.try_recv().is_err(), "{case}: and no event");
                    continue;
                }
                assert_eq!(transition, Some((from, to)), "{case}");
                match rx.try_recv() {
                    Ok(WorkerEvent::StatusChanged {
                        worker_id: id,
                        old_status,
                        new_status,
                        ..
                    }) => {
                        assert_eq!(id, worker_id, "{case}");
                        assert_eq!(
                            (old_status, new_status),
                            (from, to),
                            "{case}: the event names the edge"
                        );
                    }
                    other => panic!("{case}: expected one StatusChanged, got {other:?}"),
                }
                assert!(rx.try_recv().is_err(), "{case}: exactly one event");
            }
        }
    }

    // A probe that declines to move the status: its result, no write, no
    // event.
    {
        let (registry, mut rx, worker, worker_id) = fresh();
        let before = worker.status();
        let outcome = registry.apply_if_revision(&worker_id, worker.revision(), |_| (7, None));
        assert_eq!(outcome, Some((7, None)));
        assert_eq!(worker.status(), before);
        assert!(rx.try_recv().is_err());
    }

    // A stale revision: no write, no event, and the probe's closure does
    // not run.
    {
        let (registry, mut rx, worker, worker_id) = fresh();
        let before = worker.status();
        let stale = worker.revision().wrapping_add(1);
        assert_eq!(
            registry.transition_status_if_revision(&worker_id, stale, Failed),
            None
        );
        let ran = std::cell::Cell::new(false);
        let outcome = registry.apply_if_revision(&worker_id, stale, |_| {
            ran.set(true);
            ((), Some(Failed))
        });
        assert_eq!(outcome, None);
        assert!(!ran.get(), "a stale probe's closure does not run");
        assert_eq!(worker.status(), before, "no write");
        assert!(rx.try_recv().is_err(), "no event");
    }

    // A worker that is gone: no write, no event, through any path.
    {
        let (registry, mut rx, worker, worker_id) = fresh();
        let revision = worker.revision();
        assert!(registry.remove(&worker_id).is_some());
        assert!(matches!(rx.try_recv(), Ok(WorkerEvent::Removed { .. })));
        assert_eq!(registry.transition_status(&worker_id, Failed), None);
        assert_eq!(
            registry.transition_status_if_revision(&worker_id, revision, Failed),
            None
        );
        assert_eq!(
            registry.apply_if_revision(&worker_id, revision, |_| ((), Some(Failed))),
            None
        );
        assert!(rx.try_recv().is_err(), "a worker that is gone: no event");
    }

    // The removal's demotion: Ready becomes NotReady on the way out, any
    // other status stays, and the only event is `Removed`.
    for from in STATUSES {
        let (registry, mut rx, worker, worker_id) = fresh();
        worker.set_status(from);
        let removed = registry.remove(&worker_id).expect("registered");
        assert!(Arc::ptr_eq(&removed, &worker));
        let after = if from == Ready { NotReady } else { from };
        assert_eq!(
            worker.status(),
            after,
            "removal from {from:?}: Ready alone is demoted"
        );
        match rx.try_recv() {
            Ok(WorkerEvent::Removed { worker_id: id, .. }) => assert_eq!(id, worker_id),
            other => panic!("removal from {from:?}: expected Removed alone, got {other:?}"),
        }
        assert!(
            rx.try_recv().is_err(),
            "removal from {from:?}: no StatusChanged beside Removed"
        );
    }
}

#[test]
fn test_transition_status_if_revision_rejects_stale_transition() {
    let registry = WorkerRegistry::new();

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w1:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build(),
    );
    let worker_id = registry.register(worker.clone()).unwrap();
    let stale_revision = worker.revision();

    let replacement: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w1:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .priority(99)
            .build(),
    );
    assert!(registry.replace(&worker_id, replacement));

    assert_eq!(
        registry.transition_status_if_revision(&worker_id, stale_revision, WorkerStatus::NotReady),
        None
    );

    let current = registry.get(&worker_id).unwrap();
    assert_eq!(current.status(), WorkerStatus::Ready);
    assert_eq!(current.priority(), 99);
    assert_eq!(current.revision(), stale_revision + 1);
}

#[test]
fn test_promotion_back_to_ready_closes_the_circuit_breaker() {
    let registry = WorkerRegistry::new();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w1:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .build(),
    );
    let worker_id = registry.register(worker.clone()).unwrap();
    let revision = worker.revision();
    assert!(registry
        .transition_status_if_revision(&worker_id, revision, WorkerStatus::NotReady)
        .is_some());
    for _ in 0..8 {
        worker.record_circuit_breaker_outcome(false);
    }
    assert!(!worker.circuit_breaker_can_execute());

    assert!(registry
        .transition_status_if_revision(&worker_id, revision, WorkerStatus::Ready)
        .is_some());
    assert!(
        worker.circuit_breaker_can_execute(),
        "a worker that returns starts with a closed breaker"
    );
}

#[test]
fn test_multi_model_worker_is_indexed_for_each_model() {
    let registry = WorkerRegistry::new();

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("https://api.openai.com")
            .worker_type(WorkerType::Regular)
            .models(vec![
                ModelCard::new("gpt-4o"),
                ModelCard::new("text-embedding-3-large"),
            ])
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .build(),
    );

    let worker_id = registry.register(worker).unwrap();

    assert!(registry.get(&worker_id).is_some());
    assert_eq!(registry.get_by_model("gpt-4o").len(), 1);
    assert_eq!(registry.get_by_model("text-embedding-3-large").len(), 1);
    assert_eq!(
        registry.get_by_model("gpt-4o")[0].url(),
        "https://api.openai.com"
    );
    assert_eq!(
        registry.get_by_model("text-embedding-3-large")[0].url(),
        "https://api.openai.com"
    );

    let mut models = registry.get_models();
    models.sort();
    assert_eq!(
        models,
        vec!["gpt-4o".to_string(), "text-embedding-3-large".to_string()]
    );

    let stats = registry.stats();
    assert_eq!(stats.total_workers, 1);
    assert_eq!(stats.total_models, 2);
}

#[test]
fn test_model_alias_resolves_to_canonical_model() {
    let registry = WorkerRegistry::new();
    let aliased = worker_with_model_aliases(
        "http://aliased:8080",
        "GLM-5.2",
        &["GLM-5.2-Coding"],
        WorkerType::Prefill,
    );
    let canonical_only =
        worker_with_model_aliases("http://canonical:8080", "GLM-5.2", &[], WorkerType::Decode);

    registry.register(aliased).unwrap();
    registry.register(canonical_only).unwrap();

    let canonical_workers = registry.get_by_model("GLM-5.2");
    assert_eq!(canonical_workers.len(), 2);
    let alias_workers = registry.get_by_model("GLM-5.2-Coding");
    assert_eq!(alias_workers.len(), 2);
    assert_eq!(
        registry
            .get_workers_filtered(
                Some("GLM-5.2-Coding"),
                Some(WorkerType::Prefill),
                None,
                None,
                false,
            )
            .len(),
        1
    );
    assert_eq!(
        registry
            .get_workers_filtered(
                Some("GLM-5.2-Coding"),
                Some(WorkerType::Decode),
                None,
                None,
                false,
            )
            .len(),
        1
    );

    assert_eq!(
        registry.resolve_model_alias("GLM-5.2-Coding").as_deref(),
        Some("GLM-5.2")
    );
    assert!(registry.resolve_model_alias("GLM-5.2").is_none());
    assert!(registry.get_by_model("glm-5.2-coding").is_empty());
    assert_eq!(registry.get_models(), vec!["GLM-5.2"]);
    assert_eq!(registry.stats().total_models, 1);
}

#[test]
fn test_model_alias_tracks_shared_workers_through_removal() {
    let registry = WorkerRegistry::new();
    let first = worker_with_model_aliases(
        "http://first:8080",
        "GLM-5.2",
        &["GLM-5.2-Coding"],
        WorkerType::Prefill,
    );
    let second = worker_with_model_aliases(
        "http://second:8080",
        "GLM-5.2",
        &["GLM-5.2-Coding"],
        WorkerType::Decode,
    );
    let first_id = registry.register(first).unwrap();
    let second_id = registry.register(second).unwrap();

    assert_eq!(registry.get_by_model("GLM-5.2-Coding").len(), 2);
    registry.remove(&first_id).unwrap();
    let remaining = registry.get_by_model("GLM-5.2-Coding");
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].url(), "http://second:8080");

    registry.remove(&second_id).unwrap();
    assert!(registry.get_by_model("GLM-5.2-Coding").is_empty());
    assert!(registry.resolve_model_alias("GLM-5.2-Coding").is_none());
}

#[test]
fn test_model_alias_replace_refreshes_kept_and_changed_aliases() {
    let registry = WorkerRegistry::new();
    let original: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker:8080")
            .model(ModelCard::new("GLM-5.2").with_alias("old"))
            .build(),
    );
    let worker_id = registry.register(original).unwrap();
    let replacement: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker:8080")
            .model(ModelCard::new("GLM-5.2").with_alias("new"))
            .build(),
    );

    assert!(registry.replace(&worker_id, replacement));

    assert!(registry.get_by_model("old").is_empty());
    assert_eq!(registry.get_by_model("new").len(), 1);
}

#[test]
fn test_contains_model_accepts_canonical_id_and_alias() {
    let registry = WorkerRegistry::new();
    let worker = worker_with_model_aliases(
        "http://worker:8080",
        "GLM-5.2",
        &["GLM-5.2-Coding"],
        WorkerType::Regular,
    );
    let worker_id = registry.register(worker).unwrap();

    assert!(registry.contains_model("GLM-5.2"));
    assert!(registry.contains_model("GLM-5.2-Coding"));
    assert!(!registry.contains_model("GLM-5.2-Unknown"));
    // The wildcard is a routing sentinel, never a registered name.
    assert!(!registry.contains_model(UNKNOWN_MODEL_ID));

    assert!(registry.remove(&worker_id).is_some());
    assert!(!registry.contains_model("GLM-5.2"));
    assert!(!registry.contains_model("GLM-5.2-Coding"));
}

/// smg-lab #288: the request metric families take the served model's id,
/// or `other` for a name no worker serves, as their `model` label.
#[test]
fn metric_model_label_is_the_served_id_or_other() {
    let registry = WorkerRegistry::new();
    assert_eq!(registry.metric_model_label("ghost-1"), "other");

    let worker = worker_with_model_aliases(
        "http://worker:8080",
        "GLM-5.2",
        &["GLM-5.2-Coding"],
        WorkerType::Regular,
    );
    let worker_id = registry.register(worker).unwrap();
    assert_eq!(registry.metric_model_label("GLM-5.2"), "GLM-5.2");
    assert_eq!(
        registry.metric_model_label("GLM-5.2-Coding"),
        "GLM-5.2-Coding"
    );
    assert_eq!(
        registry.metric_model_label("ghost-1"),
        metrics_labels::MODEL_OTHER
    );
    // The gateway's own wildcard is one fixed value, kept as it is.
    assert_eq!(
        registry.metric_model_label(UNKNOWN_MODEL_ID),
        UNKNOWN_MODEL_ID
    );

    assert!(registry.remove(&worker_id).is_some());
    assert_eq!(registry.metric_model_label("GLM-5.2"), "other");
}

#[test]
fn test_model_alias_is_handed_over_when_its_owner_leaves() {
    // Two models declare the same alias. `add_model_alias` keeps the
    // first, so removing the first must not strand the second, which
    // still advertises the alias.
    let registry = WorkerRegistry::new();
    let owner = worker_with_model_aliases(
        "http://owner:8080",
        "GLM-5.2",
        &["shared-alias"],
        WorkerType::Regular,
    );
    let loser = worker_with_model_aliases(
        "http://loser:8080",
        "Qwen3",
        &["shared-alias"],
        WorkerType::Regular,
    );
    let owner_id = registry.register(owner).unwrap();
    registry.register(loser).unwrap();

    assert_eq!(
        registry.resolve_model_alias("shared-alias").as_deref(),
        Some("GLM-5.2")
    );

    assert!(registry.remove(&owner_id).is_some());

    assert_eq!(
        registry.resolve_model_alias("shared-alias").as_deref(),
        Some("Qwen3"),
        "alias must move to the remaining model that declares it"
    );
    let workers = registry.get_by_model("shared-alias");
    assert_eq!(workers.len(), 1);
    assert_eq!(workers[0].url(), "http://loser:8080");
}

#[test]
fn test_model_alias_never_shadows_a_registered_model_id() {
    // Registered later than the model it collides with.
    let registry = WorkerRegistry::new();
    let canonical =
        worker_with_model_aliases("http://canonical:8080", "GLM-5.2", &[], WorkerType::Regular);
    let colliding = worker_with_model_aliases(
        "http://colliding:8080",
        "Qwen3",
        &["GLM-5.2"],
        WorkerType::Regular,
    );
    let canonical_id = registry.register(canonical).unwrap();
    registry.register(colliding).unwrap();

    assert!(registry.resolve_model_alias("GLM-5.2").is_none());
    // The real model has gone, so the name must resolve to nothing at all
    // rather than start pointing at Qwen3.
    assert!(registry.remove(&canonical_id).is_some());
    assert!(registry.resolve_model_alias("GLM-5.2").is_none());
    assert!(registry.get_by_model("GLM-5.2").is_empty());
}

#[test]
fn test_registered_model_id_evicts_a_colliding_alias() {
    // Registered earlier than the model it collides with: same invariant,
    // resolved from the other side.
    let registry = WorkerRegistry::new();
    let colliding = worker_with_model_aliases(
        "http://colliding:8080",
        "Qwen3",
        &["GLM-5.2"],
        WorkerType::Regular,
    );
    registry.register(colliding).unwrap();
    assert_eq!(
        registry.resolve_model_alias("GLM-5.2").as_deref(),
        Some("Qwen3")
    );

    let canonical =
        worker_with_model_aliases("http://canonical:8080", "GLM-5.2", &[], WorkerType::Regular);
    let canonical_id = registry.register(canonical).unwrap();

    assert!(registry.resolve_model_alias("GLM-5.2").is_none());
    assert_eq!(registry.get_by_model("GLM-5.2").len(), 1);
    assert_eq!(
        registry.get_by_model("GLM-5.2")[0].url(),
        "http://canonical:8080"
    );

    assert!(registry.remove(&canonical_id).is_some());
    assert!(registry.get_by_model("GLM-5.2").is_empty());
}

/// Two workers at one URL, as a replacement sees them: the first serving one
/// model, the second two others.
fn replacement_pair() -> (Arc<dyn Worker>, Arc<dyn Worker>) {
    let first: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("https://api.openai.com")
            .worker_type(WorkerType::Regular)
            .models(vec![ModelCard::new("gpt-4o")])
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .build(),
    );
    let second: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("https://api.openai.com")
            .worker_type(WorkerType::Regular)
            .models(vec![ModelCard::new("o3"), ModelCard::new("o4-mini")])
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .build(),
    );
    (first, second)
}

#[test]
fn test_replace_same_url_refreshes_all_model_indexes() {
    let registry = WorkerRegistry::new();

    let (first, second) = replacement_pair();

    // First registration creates the worker
    let first_id = registry.register(first).unwrap();

    // Second registration with same URL should be rejected
    assert!(registry.register(second.clone()).is_none());

    // Use replace() to update the worker
    assert!(registry.replace(&first_id, second));

    assert_eq!(registry.len(), 1);
    assert!(registry.get_by_model("gpt-4o").is_empty());
    assert_eq!(registry.get_by_model("o3").len(), 1);
    assert_eq!(registry.get_by_model("o4-mini").len(), 1);
    assert_eq!(registry.get_by_type(WorkerType::Regular).len(), 1);
    assert_eq!(registry.get_by_connection(ConnectionMode::Http).len(), 1);

    let mut models = registry.get_models();
    models.sort();
    assert_eq!(models, vec!["o3".to_string(), "o4-mini".to_string()]);
}

#[test]
fn test_register_or_replace_upsert() {
    let registry = WorkerRegistry::new();

    let (first, second) = replacement_pair();

    let first_id = registry.register_or_replace(first);
    let second_id = registry.register_or_replace(second);

    // Same URL → same ID (upsert)
    assert_eq!(first_id, second_id);
    assert_eq!(registry.len(), 1);
    // Old model gone, new models present
    assert!(registry.get_by_model("gpt-4o").is_empty());
    assert_eq!(registry.get_by_model("o3").len(), 1);
    assert_eq!(registry.get_by_model("o4-mini").len(), 1);
}

#[test]
fn test_builder_status_override_on_replace() {
    // Regression test: metadata-only updates must not kick a healthy
    // worker back to Pending. The builder exposes a `.status()` setter
    // so callers (e.g. UpdateWorkerPropertiesStep) can pass the old
    // worker's status when constructing the replacement.
    let registry = WorkerRegistry::new();

    // First worker starts Pending (health checks enabled by default),
    // then gets promoted to Ready (simulating what the health checker does).
    let first: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker:8080")
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("llama-3"))
            .build(),
    );
    assert_eq!(first.status(), WorkerStatus::Pending);
    first.set_status(WorkerStatus::Ready);

    let first_id = registry.register(first.clone()).unwrap();
    assert_eq!(
        registry.get(&first_id).unwrap().status(),
        WorkerStatus::Ready
    );

    // Caller (e.g. UpdateWorkerPropertiesStep) reads the old status and
    // passes it to the builder. The builder honors the override instead
    // of defaulting to Pending.
    let preserved_status = first.status();
    let second: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker:8080")
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("llama-3"))
            .priority(99)
            .status(preserved_status)
            .build(),
    );
    assert_eq!(
        second.status(),
        WorkerStatus::Ready,
        "builder must honor explicit status override"
    );

    assert!(registry.replace(&first_id, second));

    let after = registry.get(&first_id).unwrap();
    assert_eq!(after.status(), WorkerStatus::Ready);
    assert_eq!(after.priority(), 99, "new priority should be applied");
}

#[test]
fn test_replace_preserves_runtime_state_and_circuit_breaker() {
    let registry = WorkerRegistry::new();
    let mut headers = http::HeaderMap::new();
    headers.insert("x-smg-routing-key", "sticky-key".parse().unwrap());

    let first: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build(),
    );
    let first_id = registry.register(first.clone()).unwrap();
    let initial_revision = first.revision();

    first.set_status(WorkerStatus::NotReady);
    first.increment_processed();
    let load_guard = WorkerLoadGuard::new(first.clone(), Some(&headers));

    for _ in 0..5 {
        first.record_outcome(503);
    }
    assert_eq!(first.circuit_breaker_state(), CircuitState::Open);

    let replacement: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .priority(99)
            .build(),
    );
    assert!(registry.replace(&first_id, replacement));

    let current = registry.get(&first_id).unwrap();
    assert_eq!(current.priority(), 99);
    assert_eq!(current.status(), WorkerStatus::NotReady);
    assert_eq!(current.load(), 1);
    assert_eq!(current.routing_key_load(), 1);
    assert_eq!(current.processed_requests(), 1);
    assert_eq!(current.circuit_breaker_state(), CircuitState::Open);
    assert_eq!(current.revision(), initial_revision + 1);

    first.increment_processed();
    assert_eq!(current.processed_requests(), 2);

    drop(load_guard);
    assert_eq!(current.load(), 0);
    assert_eq!(current.routing_key_load(), 0);
}

#[test]
fn test_builder_default_status_is_pending() {
    // Without an explicit override, health-checked workers start Pending.
    let worker = BasicWorkerBuilder::new("http://worker:8080")
        .worker_type(WorkerType::Regular)
        .build();
    assert_eq!(worker.status(), WorkerStatus::Pending);
}

#[test]
fn test_register_rejects_duplicate_url() {
    let registry = WorkerRegistry::new();

    let first: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build(),
    );
    let second: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build(),
    );

    assert!(registry.register(first).is_some());
    assert!(registry.register(second).is_none());
    assert_eq!(registry.len(), 1);
}

#[test]
fn test_model_retry_config_last_write_wins() {
    let registry = WorkerRegistry::new();

    assert!(registry.get_retry_config("llama-3").is_none());

    let config1 = RetryConfig {
        max_retries: 3,
        ..RetryConfig::default()
    };
    registry.set_model_retry_config("llama-3", config1, true);

    let stored = registry.get_retry_config("llama-3").unwrap();
    assert_eq!(stored.max_retries, 3);

    // Last write wins — overwrite with different config
    let config2 = RetryConfig {
        max_retries: 10,
        initial_backoff_ms: 200,
        ..RetryConfig::default()
    };
    registry.set_model_retry_config("llama-3", config2, true);

    let stored = registry.get_retry_config("llama-3").unwrap();
    assert_eq!(stored.max_retries, 10);
    assert_eq!(stored.initial_backoff_ms, 200);

    // Disable retries — max_retries should be set to 1
    let config3 = RetryConfig {
        max_retries: 10,
        ..RetryConfig::default()
    };
    registry.set_model_retry_config("llama-3", config3, false);

    let stored = registry.get_retry_config("llama-3").unwrap();
    assert_eq!(stored.max_retries, 1); // disabled → max_retries = 1
}

#[test]
fn test_model_retry_config_cleanup_on_last_worker_removal() {
    let registry = WorkerRegistry::new();

    let worker1 = Arc::new(
        BasicWorkerBuilder::new("http://worker1:8080")
            .model(ModelCard::new("llama-3"))
            .build(),
    ) as Arc<dyn Worker>;

    let worker2 = Arc::new(
        BasicWorkerBuilder::new("http://worker2:8080")
            .model(ModelCard::new("llama-3"))
            .build(),
    ) as Arc<dyn Worker>;

    let id1 = registry.register(worker1).unwrap();
    let id2 = registry.register(worker2).unwrap();

    registry.set_model_retry_config(
        "llama-3",
        RetryConfig {
            max_retries: 5,
            ..RetryConfig::default()
        },
        true,
    );
    assert!(registry.get_retry_config("llama-3").is_some());

    // Remove first worker — config should still exist
    registry.remove(&id1);
    assert!(registry.get_retry_config("llama-3").is_some());

    // Remove last worker — config should be cleaned up
    registry.remove(&id2);
    assert!(registry.get_retry_config("llama-3").is_none());
}

#[test]
fn test_sampling_defaults_group_warning_scan_tracks_distinct_values() {
    let registry = WorkerRegistry::new();
    let defaults_a = r#"{"temperature":0.6}"#;
    let defaults_b = r#"{"temperature":0.7}"#;

    let id1 = registry
        .register(worker_with_sampling_defaults(
            "http://worker1:8080",
            "llama-3",
            WorkerType::Regular,
            ConnectionMode::Grpc,
            Some(defaults_a),
        ))
        .unwrap();
    let id2 = registry
        .register(worker_with_sampling_defaults(
            "http://worker2:8080",
            "llama-3",
            WorkerType::Regular,
            ConnectionMode::Grpc,
            Some(defaults_a),
        ))
        .unwrap();

    assert_sampling_defaults_group_values(&registry, "llama-3", WorkerType::Regular, &[defaults_a]);

    let id3 = registry
        .register(worker_with_sampling_defaults(
            "http://worker3:8080",
            "llama-3",
            WorkerType::Regular,
            ConnectionMode::Grpc,
            Some(defaults_b),
        ))
        .unwrap();

    assert_sampling_defaults_group_values(
        &registry,
        "llama-3",
        WorkerType::Regular,
        &[defaults_a, defaults_b],
    );

    registry.remove(&id3);
    assert_sampling_defaults_group_values(&registry, "llama-3", WorkerType::Regular, &[defaults_a]);

    registry.remove(&id1);
    registry.remove(&id2);
    assert_sampling_defaults_group_values(&registry, "llama-3", WorkerType::Regular, &[]);
}

#[test]
fn test_sampling_defaults_group_warning_scan_updates_on_replace() {
    let registry = WorkerRegistry::new();
    let defaults_a = r#"{"temperature":0.6}"#;
    let defaults_b = r#"{"temperature":0.7}"#;

    let id = registry
        .register(worker_with_sampling_defaults(
            "http://worker:8080",
            "llama-3",
            WorkerType::Decode,
            ConnectionMode::Grpc,
            Some(defaults_a),
        ))
        .unwrap();

    assert!(registry.replace(
        &id,
        worker_with_sampling_defaults(
            "http://worker:8080",
            "llama-3",
            WorkerType::Decode,
            ConnectionMode::Grpc,
            Some(defaults_b),
        ),
    ));

    assert_sampling_defaults_group_values(&registry, "llama-3", WorkerType::Decode, &[defaults_b]);
}

#[test]
fn test_sampling_defaults_group_warning_scan_ignores_non_applicable_workers() {
    let registry = WorkerRegistry::new();
    let defaults = r#"{"temperature":0.6}"#;

    registry.register(worker_with_sampling_defaults(
        "http://prefill:8080",
        "llama-3",
        WorkerType::Prefill,
        ConnectionMode::Grpc,
        Some(defaults),
    ));
    registry.register(worker_with_sampling_defaults(
        "http://http-worker:8080",
        "llama-3",
        WorkerType::Regular,
        ConnectionMode::Http,
        Some(defaults),
    ));
    registry.register(worker_with_sampling_defaults(
        "http://missing-label:8080",
        "llama-3",
        WorkerType::Regular,
        ConnectionMode::Grpc,
        None,
    ));

    assert_sampling_defaults_group_values(&registry, "llama-3", WorkerType::Regular, &[]);
}

#[test]
fn test_worker_event_broadcast() {
    let registry = WorkerRegistry::new();
    let mut rx = registry.subscribe_events();

    let mut labels = HashMap::new();
    labels.insert("model_id".to_string(), "test-model".to_string());

    let worker: Box<dyn Worker> = Box::new(
        BasicWorkerBuilder::new("http://event-worker:8080")
            .worker_type(WorkerType::Regular)
            .labels(labels)
            .circuit_breaker_config(CircuitBreakerConfig::default())
            .api_key("test_api_key")
            .build(),
    );

    let worker_id = registry.register(Arc::from(worker)).unwrap();

    // Should receive Registered event
    let event = rx.try_recv().unwrap();
    match event {
        WorkerEvent::Registered { worker, .. } => {
            assert_eq!(worker.url(), "http://event-worker:8080");
        }
        other => panic!("Expected Registered event, got: {other:?}"),
    }

    registry.remove(&worker_id);

    // Should receive Removed event
    let event = rx.try_recv().unwrap();
    match event {
        WorkerEvent::Removed { worker, .. } => {
            assert_eq!(worker.url(), "http://event-worker:8080");
        }
        other => panic!("Expected Removed event, got: {other:?}"),
    }
}

fn worker_serving(url: &str, model_ids: &[&str]) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .models(
                model_ids
                    .iter()
                    .map(|id| ModelCard::new(*id))
                    .collect::<Vec<_>>(),
            )
            .health_config(no_health_check())
            .build(),
    )
}

/// Pause one real membership operation after its model snapshot, then let
/// another operation reach the rebuild. No sleep determines the ordering.
fn overlapping_hash_ring_updates(
    registry: &Arc<WorkerRegistry>,
    first: impl FnOnce(&WorkerRegistry) + Send + 'static,
    second: impl FnOnce(&WorkerRegistry) + Send + 'static,
) -> bool {
    use std::{sync::mpsc, time::Duration};

    let (snapshot_tx, snapshot_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();
    registry.hash_ring_test_hooks.lock().after_snapshot = Some((snapshot_tx, resume_rx));

    let (first_done_tx, first_done_rx) = mpsc::channel();
    let first_registry = Arc::clone(registry);
    let first_thread = std::thread::spawn(move || {
        first(&first_registry);
        first_done_tx.send(()).expect("first completion receiver");
    });
    snapshot_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("first update reaches its snapshot");

    let (started_tx, started_rx) = mpsc::channel();
    registry.hash_ring_test_hooks.lock().before_rebuild = Some(started_tx);
    let (second_done_tx, second_done_rx) = mpsc::channel();
    let second_registry = Arc::clone(registry);
    let second_thread = std::thread::spawn(move || {
        second(&second_registry);
        second_done_tx.send(()).expect("second completion receiver");
    });
    started_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("second update changes membership and reaches the rebuild");

    // The unfixed implementation completes the second publication here,
    // before the first overwrites it with its stale snapshot. The fixed
    // implementation must wait for the first publication to finish.
    let blocked = matches!(
        second_done_rx.recv_timeout(Duration::from_secs(1)),
        Err(mpsc::RecvTimeoutError::Timeout)
    );
    resume_tx.send(()).expect("release first update");
    first_done_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("first membership operation completes");
    if blocked {
        second_done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("second membership operation completes");
    }
    first_thread.join().expect("first update did not panic");
    second_thread.join().expect("second update did not panic");
    blocked
}

#[test]
fn concurrent_hash_ring_registration_keeps_encode_selectable() {
    use crate::policies::{ConsistentHashingPolicy, LoadBalancingPolicy, SelectWorkerInfo};

    let registry = Arc::new(WorkerRegistry::new());
    let prefill: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://prefill:8080")
            .model(ModelCard::new("epd-model"))
            .worker_type(WorkerType::Prefill)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .build(),
    );
    let encode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://encode:8080")
            .model(ModelCard::new("epd-model"))
            .worker_type(WorkerType::Encode)
            .connection_mode(ConnectionMode::Grpc)
            .health_config(no_health_check())
            .build(),
    );
    let second_encode = Arc::clone(&encode);
    let blocked = overlapping_hash_ring_updates(
        &registry,
        move |registry| {
            registry.register(prefill).expect("register prefill");
        },
        move |registry| {
            registry.register(second_encode).expect("register encode");
        },
    );

    assert_eq!(registry.get_by_model("epd-model").len(), 2);
    assert!(encode.is_healthy_and_eligible());
    let info = SelectWorkerInfo {
        routing_key: Some("image-hash"),
        hash_ring: registry.get_hash_ring("epd-model"),
        ..Default::default()
    };
    assert_eq!(
        ConsistentHashingPolicy::new().select_worker(&[encode], &info),
        Some(0),
        "a registered, healthy encode worker must remain selectable"
    );
    assert_eq!(info.hash_ring.expect("model ring").worker_count(), 2);
    assert_eq!(
        registry
            .get_hash_ring(UNKNOWN_MODEL_ID)
            .expect("wildcard ring")
            .worker_count(),
        2
    );
    assert!(blocked, "ring publications must be serialized");
}

fn assert_hash_ring_urls(registry: &WorkerRegistry, model_id: &str, urls: &[&str]) {
    let ring = registry.get_hash_ring(model_id);
    if urls.is_empty() {
        assert!(ring.is_none(), "removed model {model_id} has no ring");
        return;
    }
    let ring = ring.expect("nonempty model has a ring");
    assert_eq!(ring.worker_count(), urls.len(), "model {model_id}");
    for &url in urls {
        assert_eq!(
            ring.find_healthy_url("key", |candidate| candidate == url),
            Some(url),
            "model {model_id} must contain {url}"
        );
    }
}

#[test]
fn concurrent_hash_ring_registration_preserves_wildcard_union() {
    let registry = Arc::new(WorkerRegistry::new());
    let blocked = overlapping_hash_ring_updates(
        &registry,
        |registry| {
            registry
                .register(worker_serving("http://first:8080", &["model-a", "model-b"]))
                .expect("register multi-model worker");
        },
        |registry| {
            registry
                .register(worker_serving("http://second:8080", &["model-b"]))
                .expect("register other worker");
        },
    );
    assert_hash_ring_urls(&registry, "model-a", &["http://first:8080"]);
    assert_hash_ring_urls(
        &registry,
        "model-b",
        &["http://first:8080", "http://second:8080"],
    );
    assert_hash_ring_urls(
        &registry,
        UNKNOWN_MODEL_ID,
        &["http://first:8080", "http://second:8080"],
    );
    assert!(blocked, "different models also serialize wildcard updates");
}

#[test]
fn concurrent_hash_ring_removal_does_not_restore_the_last_worker() {
    let registry = Arc::new(WorkerRegistry::new());
    let first_id = registry
        .register(worker_serving("http://first:8080", &["model"]))
        .expect("first worker");
    let second_id = registry
        .register(worker_serving("http://second:8080", &["model"]))
        .expect("second worker");
    let blocked = overlapping_hash_ring_updates(
        &registry,
        move |registry| {
            registry.remove(&first_id).expect("remove first worker");
        },
        move |registry| {
            registry.remove(&second_id).expect("remove last worker");
        },
    );
    assert!(registry.get_by_model("model").is_empty());
    assert_hash_ring_urls(&registry, "model", &[]);
    assert_hash_ring_urls(&registry, UNKNOWN_MODEL_ID, &[]);
    assert!(blocked, "removals serialize ring publications");
}

#[test]
fn concurrent_hash_ring_removal_does_not_erase_a_new_registration() {
    let registry = Arc::new(WorkerRegistry::new());
    let old_id = registry
        .register(worker_serving("http://old:8080", &["model"]))
        .expect("old worker");
    let blocked = overlapping_hash_ring_updates(
        &registry,
        move |registry| {
            registry.remove(&old_id).expect("remove old worker");
        },
        |registry| {
            registry
                .register(worker_serving("http://new:8080", &["model"]))
                .expect("new worker");
        },
    );
    assert_eq!(registry.get_by_model("model").len(), 1);
    assert_hash_ring_urls(&registry, "model", &["http://new:8080"]);
    assert_hash_ring_urls(&registry, UNKNOWN_MODEL_ID, &["http://new:8080"]);
    assert!(blocked, "registration waits for the older removal");
}

#[test]
fn concurrent_hash_ring_replace_preserves_changed_models() {
    let registry = Arc::new(WorkerRegistry::new());
    let first_id = registry
        .register(worker_serving("http://first:8080", &["old", "kept"]))
        .expect("first worker");
    let second_id = registry
        .register(worker_serving("http://second:8080", &["old"]))
        .expect("second worker");
    let blocked = overlapping_hash_ring_updates(
        &registry,
        move |registry| {
            assert!(registry.replace(
                &first_id,
                worker_serving("http://first:8080", &["new", "kept"]),
            ));
        },
        move |registry| {
            registry
                .remove(&second_id)
                .expect("remove last old-model worker");
        },
    );
    assert!(registry.get_by_model("old").is_empty());
    assert_hash_ring_urls(&registry, "old", &[]);
    assert_hash_ring_urls(&registry, "new", &["http://first:8080"]);
    assert_hash_ring_urls(&registry, "kept", &["http://first:8080"]);
    assert_hash_ring_urls(&registry, UNKNOWN_MODEL_ID, &["http://first:8080"]);
    assert!(blocked, "replace and remove serialize ring publications");
}

#[test]
fn test_wildcard_hash_ring_matches_the_only_model() {
    let registry = WorkerRegistry::new();
    registry
        .register(worker_serving("http://w1:8080", &["llama-3"]))
        .unwrap();
    registry
        .register(worker_serving("http://w2:8080", &["llama-3"]))
        .unwrap();

    let wildcard = registry
        .get_hash_ring(UNKNOWN_MODEL_ID)
        .expect("requests naming no model need a ring");
    let model_ring = registry.get_hash_ring("llama-3").expect("per-model ring");

    assert_eq!(wildcard.worker_count(), 2);
    for key in ["alpha", "beta", "gamma"] {
        assert_eq!(
            wildcard.find_healthy_url(key, |_| true),
            model_ring.find_healthy_url(key, |_| true),
            "wildcard and single-model rings must agree on {key}"
        );
    }
}

#[test]
fn test_wildcard_hash_ring_unions_models() {
    let registry = WorkerRegistry::new();
    registry
        .register(worker_serving("http://w1:8080", &["llama-3"]))
        .unwrap();
    registry
        .register(worker_serving("http://w2:8080", &["gpt-4"]))
        .unwrap();

    let wildcard = registry.get_hash_ring(UNKNOWN_MODEL_ID).expect("ring");
    assert_eq!(wildcard.worker_count(), 2);
    assert_eq!(
        wildcard.find_healthy_url("key", |url| url == "http://w1:8080"),
        Some("http://w1:8080")
    );
    assert_eq!(
        wildcard.find_healthy_url("key", |url| url == "http://w2:8080"),
        Some("http://w2:8080")
    );

    // Per-model rings stay scoped to their own workers.
    assert_eq!(
        registry
            .get_hash_ring("llama-3")
            .expect("ring")
            .worker_count(),
        1
    );
}

#[test]
fn test_wildcard_hash_ring_weights_multi_model_worker_once() {
    let registry = WorkerRegistry::new();
    registry
        .register(worker_serving("http://w1:8080", &["llama-3", "gpt-4"]))
        .unwrap();
    registry
        .register(worker_serving("http://w2:8080", &["gpt-4"]))
        .unwrap();

    let wildcard = registry.get_hash_ring(UNKNOWN_MODEL_ID).expect("ring");
    assert_eq!(
        wildcard.worker_count(),
        2,
        "a worker serving two models must not take a double share of the ring"
    );
}

#[test]
fn test_wildcard_hash_ring_follows_removals() {
    let registry = WorkerRegistry::new();
    registry
        .register(worker_serving("http://w1:8080", &["llama-3"]))
        .unwrap();
    registry
        .register(worker_serving("http://w2:8080", &["gpt-4"]))
        .unwrap();

    registry.remove_by_url("http://w2:8080");
    let wildcard = registry.get_hash_ring(UNKNOWN_MODEL_ID).expect("ring");
    assert_eq!(wildcard.worker_count(), 1);
    assert_eq!(
        wildcard.find_healthy_url("key", |_| true),
        Some("http://w1:8080")
    );

    registry.remove_by_url("http://w1:8080");
    assert!(
        registry.get_hash_ring(UNKNOWN_MODEL_ID).is_none(),
        "an empty registry has no ring to route against"
    );
}

/// A ZMQ worker whose handshake driver is in flight, holding the socket
/// binds derived from `dir`. The handshake takes an OS-assigned loopback
/// port: the port derived from a temporary ipc path is a hash into a band
/// other listeners on the host may hold, and a collision there fails the
/// handshake bind after the data-plane sockets are already bound.
async fn connecting_zmq_worker(dir: &std::path::Path) -> Arc<crate::worker::BasicWorker> {
    let worker = Arc::new(
        BasicWorkerBuilder::new(format!("ipc://{}", dir.join("ts0.ipc").display()))
            .connection_mode(ConnectionMode::Zmq)
            .zmq_handshake_address("tcp://127.0.0.1:0")
            .health_config(no_health_check())
            .build(),
    );
    assert!(
        !worker.zmq_health_check().await.unwrap(),
        "worker is not ready until the handshake lands"
    );
    assert!(
        worker.zmq_connect_abort.load_full().is_some(),
        "probe must start the background handshake driver"
    );
    worker
}

/// A removed worker's ZMQ handshake driver must not outlive its registry
/// entry: it holds the worker's socket binds until it lands, which would
/// fail a re-registration at the same URL.
#[tokio::test]
async fn remove_aborts_the_zmq_handshake_driver() {
    let dir = tempfile::tempdir().unwrap();
    let registry = WorkerRegistry::new();
    let worker = connecting_zmq_worker(dir.path()).await;

    let id = registry
        .register(worker.clone() as Arc<dyn Worker>)
        .unwrap();
    registry.remove(&id).expect("worker removed");

    assert!(
        worker.zmq_connect_abort.load_full().is_none(),
        "removal must abort the in-flight handshake driver"
    );
}

/// Same for a replacement: it brings its own backend-client slot, so the
/// old instance's driver could only collide with the new worker's connect.
#[tokio::test]
async fn replace_aborts_the_replaced_workers_zmq_handshake_driver() {
    let dir = tempfile::tempdir().unwrap();
    let registry = WorkerRegistry::new();
    let worker = connecting_zmq_worker(dir.path()).await;
    let url = worker.url().to_string();

    let id = registry
        .register(worker.clone() as Arc<dyn Worker>)
        .unwrap();
    let replacement: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .connection_mode(ConnectionMode::Zmq)
            .health_config(no_health_check())
            .build(),
    );
    assert!(registry.replace(&id, replacement));

    assert!(
        worker.zmq_connect_abort.load_full().is_none(),
        "replacement must abort the replaced worker's handshake driver"
    );
}

fn overload_worker(url: &str, model_id: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model_id))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .health_config(no_health_check())
            .build(),
    )
}

/// The counters and the gauge must move on edges only, so a repeated
/// verdict from the same poll cannot inflate them.
#[test]
fn overload_counters_move_only_on_transitions() {
    let registry = WorkerRegistry::new();
    let a = overload_worker("http://127.0.0.1:9401", "m");
    let b = overload_worker("http://127.0.0.1:9402", "m");
    registry.register(Arc::clone(&a)).unwrap();
    registry.register(Arc::clone(&b)).unwrap();

    assert_eq!(registry.overloaded_worker_count("m"), 0);

    assert!(
        registry.set_worker_overloaded(&a, true),
        "first set is an edge"
    );
    assert!(
        !registry.set_worker_overloaded(&a, true),
        "re-asserting the same verdict is not an edge"
    );
    assert_eq!(registry.overloaded_worker_count("m"), 1);

    registry.set_worker_overloaded(&b, true);
    assert_eq!(registry.overloaded_worker_count("m"), 2);

    // Recovery re-admits.
    assert!(registry.set_worker_overloaded(&a, false));
    assert!(!registry.set_worker_overloaded(&a, false));
    assert_eq!(registry.overloaded_worker_count("m"), 1);
    assert!(a.is_available(), "a recovered worker is routable again");
}

/// A model with no workers has nothing flagged — the shed helper reads
/// "empty pool", which its callers answer with a 404, not a shed.
#[test]
fn empty_model_has_no_overloaded_workers() {
    let registry = WorkerRegistry::new();
    assert_eq!(registry.overloaded_worker_count("nobody"), 0);
}

/// Removing a flagged worker must give its count back, or the model would
/// stay stuck at "all overloaded" with live workers left.
#[test]
fn removing_a_flagged_worker_releases_its_count() {
    let registry = WorkerRegistry::new();
    let a = overload_worker("http://127.0.0.1:9411", "m");
    let b = overload_worker("http://127.0.0.1:9412", "m");
    let a_id = registry.register(Arc::clone(&a)).unwrap();
    registry.register(Arc::clone(&b)).unwrap();
    registry.set_worker_overloaded(&a, true);
    registry.set_worker_overloaded(&b, true);
    assert_eq!(registry.overloaded_worker_count("m"), 2);

    registry.remove(&a_id);

    // b is still flagged, so the model is still shedding — but on one
    // worker's count, not two.
    assert_eq!(registry.overloaded_worker_count("m"), 1);

    registry.set_worker_overloaded(&b, false);
    assert_eq!(registry.overloaded_worker_count("m"), 0);
}

/// The monitor snapshots worker `Arc`s at tick start and writes its verdict
/// after a multi-second fetch await. A worker removed inside that window
/// must not be able to move the live model's counter through its detached
/// handle: nothing would ever decrement it again, since every reset path
/// walks the registry's own worker map.
#[test]
fn a_removed_worker_cannot_move_the_counter_through_a_stale_handle() {
    let registry = WorkerRegistry::new();
    let a = overload_worker("http://127.0.0.1:9421", "m");
    let b = overload_worker("http://127.0.0.1:9422", "m");
    let a_id = registry.register(Arc::clone(&a)).unwrap();
    registry.register(Arc::clone(&b)).unwrap();

    registry.remove(&a_id);
    assert!(
        !registry.set_worker_overloaded(&a, true),
        "a detached handle records no edge"
    );
    assert_eq!(registry.overloaded_worker_count("m"), 0);

    // And the same in the clearing direction, which would otherwise
    // under-count a live worker's flag.
    registry.set_worker_overloaded(&b, true);
    assert!(!registry.set_worker_overloaded(&a, false));
    assert_eq!(registry.overloaded_worker_count("m"), 1);
}

/// The counter is written from the monitor's per-report loop, which runs
/// concurrently across groups. Flipping the same worker from many threads
/// must leave the count consistent with the flag.
#[test]
fn overload_counter_is_consistent_under_concurrent_transitions() {
    let registry = Arc::new(WorkerRegistry::new());
    let workers: Vec<Arc<dyn Worker>> = (0..8)
        .map(|i| {
            let w = overload_worker(&format!("http://127.0.0.1:{}", 9500 + i), "m");
            registry.register(Arc::clone(&w)).unwrap();
            w
        })
        .collect();

    let mut handles = Vec::new();
    for worker in &workers {
        for _ in 0..4 {
            let registry = Arc::clone(&registry);
            let worker = Arc::clone(worker);
            handles.push(std::thread::spawn(move || {
                for i in 0..200 {
                    registry.set_worker_overloaded(&worker, i % 2 == 0);
                }
            }));
        }
    }
    for handle in handles {
        handle.join().expect("worker flip thread");
    }

    // Whatever the interleaving settled on, the counter must equal the
    // number of workers actually carrying the flag.
    let flagged = workers.iter().filter(|w| w.is_overloaded()).count();
    assert_eq!(registry.overloaded_worker_count("m"), flagged);

    for worker in &workers {
        registry.set_worker_overloaded(worker, false);
    }
    assert_eq!(registry.overloaded_worker_count("m"), 0);
}

/// A same-URL replacement shares its runtime with the worker it replaces,
/// so a verdict left set would be carried onto a possibly different model
/// set — and the monitor's `Replaced` eviction, which clears through the
/// *old* handle, could not put the counter back. The verdict is therefore
/// released before the handover and re-derived by the next poll.
#[test]
fn replacement_releases_the_verdict_and_leaves_no_counter_behind() {
    let registry = WorkerRegistry::new();
    let original = overload_worker("http://127.0.0.1:9601", "old");
    let id = registry.register(Arc::clone(&original)).unwrap();
    registry.set_worker_overloaded(&original, true);
    assert_eq!(registry.overloaded_worker_count("old"), 1);

    let replacement = overload_worker("http://127.0.0.1:9601", "new");
    assert!(registry.replace(&id, Arc::clone(&replacement)));

    assert!(
        !replacement.is_overloaded(),
        "the replacement starts unvetoed and is routable until its next poll"
    );
    assert_eq!(registry.overloaded_worker_count("old"), 0);
    assert_eq!(registry.overloaded_worker_count("new"), 0);

    // The monitor's `Replaced` handler clears through the old handle. That
    // handle is detached now, so it must be a no-op rather than a second
    // decrement against a model it no longer belongs to.
    registry.set_worker_overloaded(&replacement, true);
    assert!(!registry.set_worker_overloaded(&original, false));
    assert_eq!(registry.overloaded_worker_count("new"), 1);
    assert_eq!(registry.overloaded_worker_count("old"), 0);
}

#[test]
fn clear_all_overload_flags_resets_every_model() {
    let registry = WorkerRegistry::new();
    let a = overload_worker("http://127.0.0.1:9701", "m1");
    let b = overload_worker("http://127.0.0.1:9702", "m2");
    registry.register(Arc::clone(&a)).unwrap();
    registry.register(Arc::clone(&b)).unwrap();
    registry.set_worker_overloaded(&a, true);
    registry.set_worker_overloaded(&b, true);

    assert_eq!(registry.clear_all_overload_flags(), 2);

    assert!(!a.is_overloaded());
    assert!(!b.is_overloaded());
    assert_eq!(registry.overloaded_worker_count("m1"), 0);
    assert_eq!(registry.overloaded_worker_count("m2"), 0);
}
