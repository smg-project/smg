use std::collections::HashMap;

use metrics_exporter_prometheus::PrometheusBuilder;

use super::*;
// The cases run over the test support's registry: its generations keep their
// members alone.
use crate::test_support::WorkerRegistry;
use crate::{
    health::circuit_breaker::{CircuitBreakerConfig, CircuitState},
    identity::basic::WorkerLoadGuard,
    metrics::{series as worker_series, series::WorkerSeriesRecorder},
    model_card::ModelCard,
    registry::event::WorkerConnected,
    test_support::{no_health_check, worker_with_model_aliases, BasicWorkerBuilder},
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
fn connect_signal_receiver_is_held_by_one_taker_at_a_time() {
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

/// The receiver handed back is taken again, with what was sent while
/// nobody read: a signal waits in the channel for the next loop. The
/// hand-back wakes a waiter that asks after it too.
#[tokio::test]
async fn a_returned_connect_signal_receiver_is_taken_again_with_what_was_sent_meanwhile() {
    let registry = WorkerRegistry::new();
    let held = registry
        .take_connect_signal_receiver()
        .expect("the first take");
    registry
        .connect_signal_sender()
        .send(WorkerSignal::Connected(WorkerConnected {
            url: "ipc:///tmp/w.ipc".to_string(),
            revision: 3,
        }))
        .expect("send on a live channel");
    registry.return_connect_signal_receiver(held);
    // A lost wake fails here by its message, not by a hang.
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        registry.connect_signal_receiver_returned(),
    )
    .await
    .expect("a hand-back nobody awaited is kept for the next await");

    let mut again = registry
        .take_connect_signal_receiver()
        .expect("the receiver is back");
    let got = again.recv().await.expect("the signal sent meanwhile");
    assert!(
        matches!(
            &got,
            WorkerSignal::Connected(WorkerConnected { revision: 3, .. })
        ),
        "{got:?}"
    );
}

#[tokio::test]
async fn connect_signal_sender_delivers_to_the_receiver() {
    let registry = WorkerRegistry::new();
    let tx = registry.connect_signal_sender();
    let mut rx = registry
        .take_connect_signal_receiver()
        .expect("receiver present");

    tx.send(WorkerSignal::Connected(WorkerConnected {
        url: "ipc:///tmp/w.ipc".to_string(),
        revision: 7,
    }))
    .expect("send on a live channel");

    let got = rx.recv().await.expect("a delivered signal");
    assert!(
        matches!(
            &got,
            WorkerSignal::Connected(WorkerConnected { url, revision: 7 }) if url == "ipc:///tmp/w.ipc"
        ),
        "{got:?}"
    );
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

    registry.on_remote_worker_state(&RemoteWorkerState {
        worker_id: "peer-w1".to_string(),
        model_id: "llama-3".to_string(),
        url: "http://remote:8080".to_string(),
        health: true,
        version: 1,
        spec: vec![],
    });
    let mesh_id = registry.get_id_by_url("http://remote:8080").unwrap();
    assert_eq!(registry.origin_of(&mesh_id), Some(WorkerOrigin::Mesh));
}

#[test]
fn a_model_is_served_locally_only_by_a_worker_this_node_owns() {
    let registry = WorkerRegistry::new();
    assert!(!registry.serves_model_locally("llama-3"));

    registry.on_remote_worker_state(&remote_state("peer-w1", "http://remote:8080", true, vec![]));
    assert!(!registry.serves_model_locally("llama-3"));

    let local: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://local:8080")
            .model(ModelCard::new("llama-3"))
            .build(),
    );
    registry.register(local).unwrap();
    assert!(registry.serves_model_locally("llama-3"));
    registry.remove_by_url("http://local:8080");
    assert!(!registry.serves_model_locally("llama-3"));
}

#[test]
fn a_worker_serves_locally_the_models_it_lists_now() {
    // A refresh of an endpoint's model list rewrites the worker's models and
    // leaves the model index as the registration wrote it. Its primary model,
    // the first card, counts no more than the others once left out.
    let registry = WorkerRegistry::new();
    let endpoint: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://endpoint:8080")
            .models(vec![ModelCard::new("qwen"), ModelCard::new("llama")])
            .build(),
    );
    registry.register(Arc::clone(&endpoint)).unwrap();
    assert!(registry.serves_model_locally("qwen"));
    assert!(registry.serves_model_locally("llama"));

    endpoint.set_models(vec![ModelCard::new("llama")]);
    assert!(!registry.serves_model_locally("qwen"));
    assert!(registry.serves_model_locally("llama"));
}

#[test]
fn a_worker_registered_without_model_cards_serves_its_primary_model() {
    // Indexed under its primary model alone, the `model_id` label or the
    // wildcard id: a model list it discovers later leaves that out, and it
    // still counts for it.
    let registry = WorkerRegistry::new();
    let labelled: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://labelled:8080")
            .label("model_id", "llama")
            .build(),
    );
    let unlabelled: Arc<dyn Worker> =
        Arc::new(BasicWorkerBuilder::new("http://unlabelled:8080").build());
    registry.register(Arc::clone(&labelled)).unwrap();
    registry.register(Arc::clone(&unlabelled)).unwrap();
    assert!(registry.serves_model_locally("llama"));
    assert!(registry.serves_model_locally(unlabelled.model_id()));

    labelled.set_models(vec![ModelCard::new("qwen")]);
    unlabelled.set_models(vec![ModelCard::new("qwen")]);
    assert!(registry.serves_model_locally("llama"));
    assert!(registry.serves_model_locally(unlabelled.model_id()));
}

#[test]
fn an_alias_is_not_a_model_served_locally() {
    // Policies are keyed by canonical ids, which the routers resolve every
    // alias to first.
    let registry = WorkerRegistry::new();
    registry
        .register(worker_with_model_aliases(
            "http://w1:8080",
            "llama",
            &["llama-latest"],
            WorkerType::Regular,
        ))
        .unwrap();
    assert!(registry.serves_model_locally("llama"));
    assert!(!registry.serves_model_locally("llama-latest"));
}

#[test]
fn a_worker_keeps_the_policy_of_every_model_it_serves_and_of_its_primary_model() {
    let endpoint: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://endpoint:8080")
            .models(vec![ModelCard::new("qwen"), ModelCard::new("llama")])
            .build(),
    );
    assert_eq!(
        WorkerRegistry::policy_model_ids(&endpoint),
        ["qwen", "llama"]
    );
    // A refresh of its model list leaves llama in the model index, and its
    // removal asks after llama all the same.
    endpoint.set_models(vec![ModelCard::new("qwen"), ModelCard::new("mistral")]);
    assert_eq!(
        WorkerRegistry::policy_model_ids(&endpoint),
        ["qwen", "mistral", "llama"]
    );

    // A worker registered before its models were known names its policy by
    // its primary model, which the models it discovers later leave out.
    let lazy: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://lazy:8080").build());
    let primary = lazy.model_id().to_string();
    lazy.set_models(vec![ModelCard::new("qwen")]);
    assert_eq!(
        WorkerRegistry::policy_model_ids(&lazy),
        vec!["qwen".to_string(), primary]
    );
}

/// The record a rebuild from a decoded spec is for.
fn record(version: u64) -> ImportRecord {
    ImportRecord {
        version,
        from_spec: true,
    }
}

fn remote_state(worker_id: &str, url: &str, health: bool, spec: Vec<u8>) -> RemoteWorkerState {
    RemoteWorkerState {
        worker_id: worker_id.to_string(),
        model_id: "llama-3".to_string(),
        url: url.to_string(),
        health,
        version: 1,
        spec,
    }
}

/// The address a masked-line case registers: a password in its userinfo.
const ADDRESS_WITH_A_PASSWORD: &str = "http://user:s3cret@remote:8080";

/// Every captured line that says one of `messages` names the address masked,
/// each message is said at least once, and the password is in no line.
fn assert_the_lines_name_the_address_masked(
    lines: &[&str],
    messages: &[&str],
) -> Result<(), String> {
    if let Some(line) = lines.iter().find(|line| line.contains("s3cret")) {
        return Err(format!("a line carries the password: {line}"));
    }
    for message in messages {
        let mut said = lines
            .iter()
            .filter(|line| line.contains(message))
            .peekable();
        if said.peek().is_none() {
            return Err(format!("no line says: {message}"));
        }
        if let Some(line) = said.find(|line| !line.contains("://***@remote:8080")) {
            return Err(format!("a line does not name the address masked: {line}"));
        }
    }
    Ok(())
}

/// A mesh import's lines name the address masked, at every level: each
/// branch of the import that names the address is taken once with a state
/// whose address carries a password.
#[test]
#[tracing_test::traced_test]
fn a_mesh_imports_lines_name_the_address_masked() {
    let url = ADDRESS_WITH_A_PASSWORD;
    let spec = |model: &str| spec_bytes(url, model, WorkerType::Regular);
    let undecodable = || b"not a worker spec".to_vec();
    let state = |version: u64, health: bool, spec: Vec<u8>| RemoteWorkerState {
        version,
        ..remote_state("peer-w1", url, health, spec)
    };

    // A first import from a spec, its health hint, its owner's replacement,
    // and a replacement whose spec does not decode here.
    let registry = WorkerRegistry::new();
    registry.on_remote_worker_state(&state(1, false, spec("llama-3")));
    registry.on_remote_worker_state(&state(1, true, spec("llama-3")));
    registry.on_remote_worker_state(&state(2, true, spec("llama-4")));
    registry.on_remote_worker_state(&state(3, true, undecodable()));

    // A first import whose spec does not decode, and its owner's
    // replacement with no spec at all.
    let registry = WorkerRegistry::new();
    registry.on_remote_worker_state(&state(1, true, undecodable()));
    registry.on_remote_worker_state(&state(2, true, Vec::new()));

    // A spec that declares the same-host transport.
    let registry = WorkerRegistry::new();
    let same_host = serde_json::json!({ "url": url, "connection_mode": "zmq" });
    registry.on_remote_worker_state(&state(1, true, serde_json::to_vec(&same_host).unwrap()));

    // An address of the same-host transport itself.
    let registry = WorkerRegistry::new();
    let same_host_address = "tcp://user:s3cret@remote:8080";
    registry.on_remote_worker_state(&remote_state(
        "peer-w2",
        same_host_address,
        true,
        Vec::new(),
    ));

    // A peer's echo of a worker this node owns.
    let registry = WorkerRegistry::new();
    let local: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new("m"))
            .build(),
    );
    registry.register(local).unwrap();
    registry.on_remote_worker_state(&state(1, false, Vec::new()));

    logs_assert(|lines: &[&str]| {
        assert_the_lines_name_the_address_masked(
            lines,
            &[
                "Registered mesh-synced worker into local registry",
                "Updated health for existing mesh-synced worker",
                "Rebuilt mesh-synced worker from its owner's replacement",
                "carries a spec this node cannot decode",
                "undecodable WorkerSpec in mesh state; importing minimal worker",
                "Rebuilt the minimal mesh-synced worker from its owner's replacement",
                "Ignoring mesh state for host-local ZMQ worker",
                "Ignoring mesh state for locally-owned worker",
            ],
        )?;
        if lines
            .iter()
            .any(|line| line.contains("url=tcp://***@remote:8080"))
        {
            Ok(())
        } else {
            Err("no line names the same-host address masked".to_string())
        }
    });
}

/// The lines of a refused replacement and of divergent sampling defaults
/// name the address masked.
#[test]
#[tracing_test::traced_test]
fn a_replacements_lines_name_the_address_masked() {
    let url = ADDRESS_WITH_A_PASSWORD;
    let at = |url: &str| -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("m"))
                .build(),
        )
    };
    let registry = WorkerRegistry::new();
    let id = registry.register(at(url)).unwrap();
    let revision = registry.get(&id).unwrap().revision();
    assert!(
        !registry.replace_claiming_local(&id, at(url), revision + 1),
        "a replacement at a revision the worker is not at is refused"
    );
    assert!(
        !registry.replace(&id, at("http://user:s3cret@remote:8081")),
        "a replacement at another address is refused"
    );

    // Two gRPC workers of one routing group whose sampling defaults differ.
    let registry = WorkerRegistry::new();
    for (url, defaults) in [
        (url, r#"{"temperature":0.1}"#),
        ("grpc://user:s3cret@remote:8081", r#"{"temperature":0.9}"#),
    ] {
        registry.register(worker_with_sampling_defaults(
            url,
            "m",
            WorkerType::Regular,
            ConnectionMode::Grpc,
            Some(defaults),
        ));
    }

    logs_assert(|lines: &[&str]| {
        if let Some(line) = lines.iter().find(|line| line.contains("s3cret")) {
            return Err(format!("a line carries the password: {line}"));
        }
        for message in [
            "Aborting replacement: worker was replaced before the lock was acquired",
            "replace() does not support URL changes",
            "Divergent default sampling params reported by workers in the same routing group",
        ] {
            let said = lines.iter().find(|line| line.contains(message));
            match said {
                None => return Err(format!("no line says: {message}")),
                Some(line) if !line.contains("://***@remote:808") => {
                    return Err(format!("a line does not name the address masked: {line}"));
                }
                Some(_) => {}
            }
        }
        Ok(())
    });
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
    let spec: crate::worker::WorkerSpec = serde_json::from_value(serde_json::json!({
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
    let spec: crate::worker::WorkerSpec = serde_json::from_value(serde_json::json!({
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
    let spec: crate::worker::WorkerSpec = serde_json::from_value(serde_json::json!({
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

/// The spec a publisher's state carries, as the outbound sync encodes it.
fn spec_bytes(url: &str, model: &str, worker_type: WorkerType) -> Vec<u8> {
    let worker = BasicWorkerBuilder::new(url)
        .model(ModelCard::new(model))
        .worker_type(worker_type)
        .build();
    serde_json::to_vec(&worker.metadata().spec).unwrap()
}

#[test]
fn a_newer_state_on_the_imports_own_key_rebuilds_the_import_from_its_spec() {
    // The owner replaced the worker in place (a new model and role under the
    // same address) and published the replacement under the same key with a
    // higher version. Before, such a state refreshed the import's health
    // only, and the import kept the specification it was first built from.
    let registry = WorkerRegistry::new();
    let url = "http://remote:8080";
    let mut state = remote_state(
        "peer-w1",
        url,
        true,
        spec_bytes(url, "m1", WorkerType::Regular),
    );
    state.version = 3;
    registry.on_remote_worker_state(&state);
    let id = registry.get_id_by_url(url).unwrap();
    let import = registry.get(&id).unwrap();
    assert_eq!(import.model_id(), "m1");
    let built = import.revision();

    let mut replaced = remote_state(
        "peer-w1",
        url,
        true,
        spec_bytes(url, "m2", WorkerType::Decode),
    );
    replaced.version = 4;
    registry.on_remote_worker_state(&replaced);
    let rebuilt = registry.get(&id).expect("the import keeps its id");
    assert_eq!(
        rebuilt.model_id(),
        "m2",
        "the import follows the owner's new model"
    );
    assert_eq!(
        *rebuilt.worker_type(),
        WorkerType::Decode,
        "and its new role"
    );
    assert_eq!(
        registry.origin_of(&id),
        Some(WorkerOrigin::Mesh),
        "a rebuilt import is still an import"
    );
    assert!(
        rebuilt.revision() > built,
        "the rebuild went through the replacement path"
    );
    assert_eq!(
        registry.get_id_by_url(url),
        Some(id.clone()),
        "under the same id, so the owner's tombstone still resolves"
    );

    // The reconcile pass replays the key's current state every interval:
    // the same version is a health hint, not another rebuild.
    let rebuilt_at = rebuilt.revision();
    registry.on_remote_worker_state(&replaced);
    assert_eq!(
        registry.get(&id).unwrap().revision(),
        rebuilt_at,
        "a replay of the version the import has rebuilds nothing"
    );

    // The health hint still reaches the rebuilt import.
    replaced.health = false;
    registry.on_remote_worker_state(&replaced);
    assert_eq!(
        registry.get(&id).unwrap().status(),
        WorkerStatus::NotReady,
        "an unhealthy state demotes the rebuilt import"
    );
}

#[test]
fn a_rebuild_aborts_when_the_import_was_claimed_before_the_lock() {
    // The rebuild decides on the origin before the per-worker lock and can
    // lose the race to a local claim (`register_or_replace`, a PUT) that
    // promotes the worker in between; the origin is read again under the
    // lock and the replacement aborts, as a removal does.
    let registry = WorkerRegistry::new();
    let url = "http://w:8080";
    registry.on_remote_worker_state(&remote_state("peer-w1", url, true, vec![]));
    let id = registry.get_id_by_url(url).unwrap();
    let mine: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new("mine"))
            .build(),
    );
    assert_eq!(registry.register_or_replace(mine), id);

    // Models the late rebuild, to a publication at version 2: expecting an
    // import, finding a local worker.
    let late: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new("theirs"))
            .build(),
    );
    assert!(
        !registry.replace_inner(&id, late, None, None, Some(record(2))),
        "the replacement must abort when the worker is no longer an import"
    );
    assert_eq!(
        registry.get(&id).unwrap().model_id(),
        "mine",
        "the claimed worker keeps the operator's specification"
    );
}

#[test]
fn a_rebuild_to_a_version_the_import_already_passed_aborts_under_the_lock() {
    // Two publications of one key on two tasks (the subscription with the
    // newer one, a reconcile pass with a state it read just before): both
    // pass the check outside the lock with the record as it was. Whichever
    // takes the lock second finds the record at or past its version and
    // aborts, so the import never ends on the older specification with the
    // newer version recorded.
    let registry = WorkerRegistry::new();
    let url = "http://remote:8080";
    let mut state = remote_state(
        "peer-w1",
        url,
        true,
        spec_bytes(url, "m1", WorkerType::Regular),
    );
    state.version = 3;
    registry.on_remote_worker_state(&state);
    let id = registry.get_id_by_url(url).unwrap();
    let mut newest = remote_state(
        "peer-w1",
        url,
        true,
        spec_bytes(url, "m5", WorkerType::Regular),
    );
    newest.version = 5;
    registry.on_remote_worker_state(&newest);
    assert_eq!(registry.get(&id).unwrap().model_id(), "m5");

    // Models the slower task, whose check passed with the record at 3.
    let late: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new("m4"))
            .build(),
    );
    assert!(
        !registry.replace_inner(&id, late, None, None, Some(record(4))),
        "a publication the record already passed rebuilds nothing"
    );
    assert_eq!(
        registry.get(&id).unwrap().model_id(),
        "m5",
        "the import keeps the newer specification"
    );
    assert_eq!(
        registry
            .import_records
            .get(&id)
            .map(|built_from| built_from.version),
        Some(5),
        "and its record"
    );
}

#[test]
fn a_rebuild_that_finds_no_record_under_the_lock_aborts() {
    // The import was removed and imported again from the live key between
    // a task's check outside the lock and its rebuild: the fresh import is
    // registered first and its record inserted after, so under the lock the
    // worker is an import with no record. The late rebuild aborts; it must
    // not write its older state over the fresh import and leave the record
    // the other task inserts describing a worker that is not there.
    let registry = WorkerRegistry::new();
    let url = "http://remote:8080";
    let mut fresh = remote_state(
        "peer-w1",
        url,
        true,
        spec_bytes(url, "m3", WorkerType::Regular),
    );
    fresh.version = 3;
    registry.on_remote_worker_state(&fresh);
    let id = registry.get_id_by_url(url).unwrap();
    registry.import_records.remove(&id);

    let late: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new("m2"))
            .build(),
    );
    assert!(
        !registry.replace_inner(&id, late, None, None, Some(record(2))),
        "a rebuild that finds no record under the lock must abort"
    );
    assert_eq!(
        registry.get(&id).unwrap().model_id(),
        "m3",
        "the fresh import keeps its specification"
    );
    assert!(
        registry.import_records.get(&id).is_none(),
        "the aborted rebuild wrote a record of its own"
    );
}

#[test]
fn a_newer_state_whose_spec_does_not_decode_here_keeps_the_import_as_built() {
    // A rolling upgrade: the owner's replacement carries a spec this build
    // cannot decode. The minimal fallback is right for a brand-new import
    // (there is nothing better) and a downgrade for an existing one: the
    // import keeps its specification, the health hint still applies, and
    // the version is recorded (the bytes never decode later on this node,
    // and the owner's next replacement carries a higher version), so a
    // replay reads nothing again and a later decodable publication still
    // rebuilds the import.
    let registry = WorkerRegistry::new();
    let url = "http://remote:8080";
    let mut state = remote_state(
        "peer-w1",
        url,
        true,
        spec_bytes(url, "m1", WorkerType::Decode),
    );
    state.version = 1;
    registry.on_remote_worker_state(&state);
    let id = registry.get_id_by_url(url).unwrap();
    let built = registry.get(&id).unwrap().revision();

    let mut undecodable = remote_state("peer-w1", url, false, b"{not a spec".to_vec());
    undecodable.version = 2;
    registry.on_remote_worker_state(&undecodable);
    let kept = registry.get(&id).unwrap();
    assert_eq!(kept.model_id(), "m1", "the import keeps its model");
    assert_eq!(*kept.worker_type(), WorkerType::Decode, "and its role");
    assert_eq!(kept.revision(), built, "nothing was rebuilt");
    assert_eq!(
        kept.status(),
        WorkerStatus::NotReady,
        "the health hint of the undecodable state still applied"
    );
    assert_eq!(
        registry
            .import_records
            .get(&id)
            .map(|built_from| built_from.version),
        Some(2),
        "the undecodable publication's version is recorded"
    );

    let mut decodable = remote_state(
        "peer-w1",
        url,
        true,
        spec_bytes(url, "m2", WorkerType::Decode),
    );
    decodable.version = 3;
    registry.on_remote_worker_state(&decodable);
    assert_eq!(
        registry.get(&id).unwrap().model_id(),
        "m2",
        "a later publication that decodes rebuilds the import"
    );
}

#[test]
fn a_newer_state_without_a_usable_spec_rebuilds_a_minimal_import() {
    // The import is the minimal worker already (the first state this node
    // saw carried no spec): a replacement without a usable spec rebuilds it
    // all the same, so it follows the owner's new model; a minimal-to-minimal
    // rebuild loses nothing. Only an import built from a spec keeps its
    // specification (the test above).
    let registry = WorkerRegistry::new();
    let url = "http://remote:8080";
    let mut state = remote_state("peer-w1", url, true, vec![]);
    state.model_id = "m1".to_string();
    state.version = 1;
    registry.on_remote_worker_state(&state);
    let id = registry.get_id_by_url(url).unwrap();
    let import = registry.get(&id).unwrap();
    assert_eq!(import.model_id(), "m1");
    let built = import.revision();

    let mut undecodable = remote_state("peer-w1", url, true, b"{not a spec".to_vec());
    undecodable.model_id = "m2".to_string();
    undecodable.version = 2;
    registry.on_remote_worker_state(&undecodable);
    let rebuilt = registry.get(&id).expect("the import keeps its id");
    assert_eq!(
        rebuilt.model_id(),
        "m2",
        "a minimal import follows the owner's new model past a spec it cannot decode"
    );
    assert!(
        rebuilt.revision() > built,
        "the rebuild went through the replacement path"
    );
    assert_eq!(
        registry.origin_of(&id),
        Some(WorkerOrigin::Mesh),
        "a rebuilt import is still an import"
    );
    assert!(
        registry.get_by_model("m1").is_empty(),
        "the old model no longer routes to it"
    );
    assert_eq!(registry.get_by_model("m2").len(), 1, "the new one does");
    assert_eq!(
        registry
            .import_records
            .get(&id)
            .map(|built_from| (built_from.version, built_from.from_spec)),
        Some((2, false)),
        "the record says minimal, at the new version"
    );

    let mut absent = remote_state("peer-w1", url, true, vec![]);
    absent.model_id = "m3".to_string();
    absent.version = 3;
    registry.on_remote_worker_state(&absent);
    assert_eq!(
        registry.get(&id).unwrap().model_id(),
        "m3",
        "and past a publication with no spec at all"
    );
}

#[test]
fn a_state_from_another_key_for_an_imports_address_stays_a_health_hint() {
    // Only the import's own key rebuilds it: a state under another id for
    // the same address (a second publisher of the endpoint, or an orphan
    // key of the owner's earlier life) carries a version from another
    // sequence, and is read as it always was, for its health alone.
    let registry = WorkerRegistry::new();
    let url = "http://remote:8080";
    let mut state = remote_state(
        "peer-w1",
        url,
        true,
        spec_bytes(url, "m1", WorkerType::Regular),
    );
    state.version = 1;
    registry.on_remote_worker_state(&state);
    let id = registry.get_id_by_url(url).unwrap();
    let built = registry.get(&id).unwrap().revision();

    let mut other = remote_state(
        "peer-w2",
        url,
        false,
        spec_bytes(url, "m2", WorkerType::Decode),
    );
    other.version = 9;
    registry.on_remote_worker_state(&other);
    let import = registry.get(&id).expect("the import stays");
    assert_eq!(import.model_id(), "m1", "another key's spec is not applied");
    assert_eq!(import.revision(), built, "and nothing was rebuilt");
    assert_eq!(
        import.status(),
        WorkerStatus::NotReady,
        "its health hint applied, as before"
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
            .remove_inner(&id, Some(WorkerOrigin::Mesh), None)
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

/// The model index is iterated on request paths and the map's iterator pays
/// for every shard, so its shard count is fixed: not the library's default
/// of four shards a CPU.
#[test]
fn the_model_index_keeps_a_fixed_shard_count() {
    use dashmap::Map as _;

    let registry = WorkerRegistry::new();
    assert_eq!(
        registry.model_index._shard_count(),
        16,
        "the model index's shard count follows the CPUs"
    );
}

/// The per-model retry overrides are scanned on a request path, as the model
/// index is counted on one, so their map keeps the same fixed shard count.
#[test]
fn the_retry_override_map_keeps_a_fixed_shard_count() {
    use dashmap::Map as _;

    let registry = WorkerRegistry::new();
    assert_eq!(
        registry.model_retry_configs._shard_count(),
        16,
        "the retry override map's shard count follows the CPUs"
    );
}

// Health-checker integration tests moved to health/manager.rs along with
// the loop itself. The registry is now a pure collection — see
// `health::manager::WorkerManager` tests.

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
    let load_guard = WorkerLoadGuard::with_key(first.clone(), Some("sticky-key"));

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

/// The registry counts its overload edges for a host's read: one a veto
/// that changed, either way, a clear of every flag among them. A verdict
/// said again counts nothing, and a worker the registry does not hold
/// counts nothing. The replacement and the removal of an overloaded worker
/// lower its flag, an edge each, and those of a worker with no flag count
/// nothing.
#[test]
fn an_overload_edge_is_counted_once_and_a_verdict_said_again_is_not() {
    let registry = WorkerRegistry::new();
    let a = overload_worker("http://127.0.0.1:9711", "m");
    let b = overload_worker("http://127.0.0.1:9712", "m");
    let a_id = registry.register(Arc::clone(&a)).unwrap();
    let b_id = registry.register(Arc::clone(&b)).unwrap();
    assert_eq!(registry.overload_edges(), 0);

    assert!(registry.set_worker_overloaded(&a, true));
    assert!(!registry.set_worker_overloaded(&a, true));
    assert_eq!(registry.overload_edges(), 1, "said again: no edge");
    assert!(registry.set_worker_overloaded(&b, true));
    assert!(registry.set_worker_overloaded(&a, false));
    assert_eq!(registry.overload_edges(), 3, "either way is an edge");

    assert_eq!(registry.clear_all_overload_flags(), 1);
    assert_eq!(registry.overload_edges(), 4, "a clear is an edge a flag");

    let stranger = overload_worker("http://127.0.0.1:9713", "m");
    assert!(!registry.set_worker_overloaded(&stranger, true));
    assert_eq!(registry.overload_edges(), 4, "no worker of the registry's");

    assert!(registry.set_worker_overloaded(&a, true));
    assert!(registry.set_worker_overloaded(&b, true));
    let again = overload_worker("http://127.0.0.1:9711", "m");
    assert!(registry.replace(&a_id, again));
    assert_eq!(
        registry.overload_edges(),
        7,
        "a replacement lowers the flag"
    );
    assert!(registry.remove(&b_id).is_some());
    assert_eq!(registry.overload_edges(), 8, "a removal lowers the flag");
    assert!(registry.remove(&a_id).is_some());
    assert_eq!(registry.overload_edges(), 8, "no flag to lower: no edge");
}

/// A removed worker's address stays retired in the series store while a
/// handle to the worker lives (a late write of that handle lands nowhere),
/// and the store's upkeep forgets it once the last handle is gone: the
/// registry hands the store the removed worker's own witness.
#[test]
fn a_removed_workers_address_is_retired_until_its_last_handle_is_gone() {
    let series = worker_series::installed_or_default();
    let registry = WorkerRegistry::new();
    let url = "http://retired-seam:8080";
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new("m"))
            .health_config(no_health_check())
            .build(),
    );
    let held = Arc::clone(&worker);
    let id = registry.register(worker).unwrap();
    assert!(!series.is_retired(url), "a registered address is live");

    registry.remove(&id);
    assert!(series.is_retired(url), "the removal retires the address");
    series.run_upkeep();
    assert!(
        series.is_retired(url),
        "a held worker keeps its address retired through the upkeep"
    );

    drop(held);
    series.run_upkeep();
    assert!(
        !series.is_retired(url),
        "the upkeep forgets the address once the removed worker is gone"
    );
}

/// A worker replaced at its URL stays alive in the handles taken before the
/// replacement (a load guard) and writes under the URL when they go: the
/// registry hands the store its witness at the replacement, so once the URL
/// is removed the upkeep keeps the address retired until that instance is
/// gone too, and the guard's release lands nowhere.
#[test]
fn a_replaced_workers_guard_keeps_the_removed_address_retired_until_it_drops() {
    let series = worker_series::installed_or_default();
    let recorder = WorkerSeriesRecorder::new(
        PrometheusBuilder::new().build_recorder(),
        Arc::clone(series),
    );
    let url = "http://replaced-seam:8080";
    let worker = || -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("m"))
                .health_config(no_health_check())
                .build(),
        )
    };
    let listed = || {
        let mut rendered = String::new();
        series.render_into(&mut rendered);
        rendered.lines().any(|line| {
            line.starts_with("smg_worker_requests_active{")
                && line.contains("worker=\"http://replaced-seam:8080\"")
        })
    };
    metrics::with_local_recorder(&recorder, || {
        let registry = WorkerRegistry::new();
        let first = worker();
        let id = registry.register(Arc::clone(&first)).unwrap();
        let guard = WorkerLoadGuard::with_key(first, None);
        assert!(
            listed(),
            "the guard's load is a series of the registered address"
        );
        assert!(registry.replace(&id, worker()), "a same-URL replacement");

        registry.remove(&id);
        assert!(series.is_retired(url), "the removal retires the address");
        series.run_upkeep();
        assert!(
            series.is_retired(url),
            "the replaced instance in its guard keeps the address retired"
        );

        drop(guard);
        assert!(
            !listed(),
            "the guard's release after the removal lands nowhere"
        );
        series.run_upkeep();
        assert!(
            !series.is_retired(url),
            "the upkeep forgets the address once the replaced instance is gone"
        );
    });
}

/// `remove_if` reads its condition of the worker as it stands and removes
/// the worker in the same step under its lock: a condition that reads false
/// leaves the worker, one that reads true takes it, and a worker gone reads
/// `None`.
#[test]
fn remove_if_removes_the_worker_only_while_its_condition_holds() {
    let registry = WorkerRegistry::new();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w1:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build(),
    );
    let id = registry
        .register(Arc::clone(&worker))
        .expect("the worker registers");
    let revision = worker.revision();

    assert!(
        registry
            .remove_if(&id, |w| w.revision() == revision + 1)
            .is_none(),
        "a condition that reads false leaves the worker"
    );
    assert!(registry.get(&id).is_some(), "the worker stands");
    let removed = registry
        .remove_if(&id, |w| w.revision() == revision)
        .expect("a condition that reads true takes the worker");
    assert!(
        Arc::ptr_eq(&removed, &worker),
        "the worker removed is the one registered"
    );
    assert!(registry.get(&id).is_none(), "the worker is gone");
    assert!(
        registry.remove_if(&id, |_| true).is_none(),
        "a worker gone reads None"
    );
}

/// A call that names a worker already removed keeps no lock for it: only the
/// removal that found the worker drops its entry, so a later one would stay.
#[test]
fn a_call_that_names_a_removed_worker_keeps_no_lock_for_it() {
    let registry = WorkerRegistry::new();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://w1:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build(),
    );
    let id = registry
        .register(Arc::clone(&worker))
        .expect("the worker registers");
    registry.remove(&id).expect("the worker is removed");
    assert!(registry.worker_mutation_locks.is_empty());

    assert_eq!(
        registry.transition_status_if_revision(&id, worker.revision(), WorkerStatus::NotReady),
        None
    );
    assert!(registry.worker_mutation_locks.is_empty(), "a transition");
    assert!(!registry.replace(&id, Arc::clone(&worker)));
    assert!(registry.worker_mutation_locks.is_empty(), "a replacement");
    assert!(registry.remove_if(&id, |_| true).is_none());
    assert!(
        registry.worker_mutation_locks.is_empty(),
        "a conditional removal"
    );
    assert!(registry.remove(&id).is_none());
    assert!(registry.worker_mutation_locks.is_empty(), "a removal");
}

/// The lines a rendered scrape holds for the worker at `url`, sorted.
fn series_of(rendered: &str, url: &str) -> Vec<String> {
    let label = format!("worker=\"{url}\"");
    let mut lines: Vec<String> = rendered
        .lines()
        .filter(|line| line.contains(&label))
        .map(str::to_owned)
        .collect();
    lines.sort_unstable();
    lines
}

/// The sample of `family` that carries the label of the worker at `url`
/// alone.
fn worker_sample(rendered: &str, family: &str, url: &str) -> Option<f64> {
    let prefix = format!("{family}{{worker=\"{url}\"}} ");
    rendered
        .lines()
        .find_map(|line| line.strip_prefix(prefix.as_str()))
        .map(|value| value.parse().unwrap())
}

/// A breaker that opens on one failed request and stays open for an hour.
fn opens_on_one_failure() -> CircuitBreakerConfig {
    CircuitBreakerConfig {
        failure_threshold: 1,
        success_threshold: 1,
        timeout_duration: std::time::Duration::from_secs(3600),
        window_duration: std::time::Duration::from_secs(3600),
    }
}

/// Building a worker writes no series. Its registration starts them: its
/// health, its HTTP/2 flag, its breaker's state and two counts, and at zero
/// the breaker's five transition counters and its health readmissions.
#[test]
fn a_workers_series_start_at_its_registration_and_not_at_its_build() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let url = "http://series-start:8080";
    metrics::with_local_recorder(&recorder, || {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .http2(true)
                .build(),
        );
        assert_eq!(
            series_of(&handle.render(), url),
            Vec::<String>::new(),
            "a build writes no series"
        );

        let registry = WorkerRegistry::new();
        registry.register(worker).expect("a fresh URL registers");
        let rendered = handle.render();
        assert_eq!(
            worker_sample(&rendered, "smg_worker_health", url),
            Some(1.0)
        );
        assert_eq!(worker_sample(&rendered, "smg_worker_http2", url), Some(1.0));
        assert_eq!(
            worker_sample(&rendered, "smg_worker_cb_state", url),
            Some(0.0)
        );
        // The breaker's two counts are on the scrape from the registration,
        // ahead of the worker's first outcome.
        for count in [
            "smg_worker_cb_consecutive_failures",
            "smg_worker_cb_consecutive_successes",
        ] {
            assert_eq!(worker_sample(&rendered, count, url), Some(0.0), "{count}");
        }
        let zero_counters: Vec<String> = series_of(&rendered, url)
            .into_iter()
            .filter(|line| {
                line.starts_with("smg_worker_cb_transitions_total{")
                    || line.starts_with("smg_worker_cb_health_readmissions_total{")
            })
            .collect();
        assert_eq!(zero_counters.len(), 6, "{rendered}");
        assert!(
            zero_counters.iter().all(|line| line.ends_with(" 0")),
            "{rendered}"
        );
    });
}

/// A replacement is built for an address a live worker serves, and its host
/// refuses it before it enters the registry. Its build wrote no series, so
/// every series of the address reads as the live worker left it: healthy,
/// on HTTP/2, its breaker open.
#[test]
fn a_replacement_that_never_enters_leaves_the_live_workers_series_as_they_stood() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let url = "http://refused-replacement:8080";
    metrics::with_local_recorder(&recorder, || {
        let registry = WorkerRegistry::new();
        let live: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .http2(true)
                .circuit_breaker_config(opens_on_one_failure())
                .build(),
        );
        registry
            .register(Arc::clone(&live))
            .expect("a fresh URL registers");
        live.record_outcome(503);
        let rendered = handle.render();
        let stood = series_of(&rendered, url);
        assert_eq!(
            worker_sample(&rendered, "smg_worker_health", url),
            Some(1.0)
        );
        assert_eq!(worker_sample(&rendered, "smg_worker_http2", url), Some(1.0));
        assert_eq!(
            worker_sample(&rendered, "smg_worker_cb_state", url),
            Some(1.0)
        );

        // The replacement differs in each of them: its health checks are on,
        // so it starts pending; it speaks HTTP/1.1; its breaker is closed.
        let refused = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .build();
        assert!(!refused.is_healthy());
        drop(refused);

        assert_eq!(series_of(&handle.render(), url), stood);
    });
}

/// A replacement's series are its own from the moment it enters: its HTTP/2
/// flag, and the breaker it ends with. A replacement with the old worker's
/// breaker settings adopts the live breaker, whose open state and failure
/// count stand. One with other settings keeps its own breaker: the state
/// reads closed and the failure count zero. The status is the address's:
/// a replacement built pending serves as ready, and reads healthy. The
/// counter of the trip keeps its count through both.
#[test]
fn a_replacement_publishes_its_own_series_where_it_enters() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let url = "http://replacement-series:8080";
    let replacement = |http2: bool, breaker: CircuitBreakerConfig| -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .http2(http2)
                .circuit_breaker_config(breaker)
                .build(),
        )
    };
    metrics::with_local_recorder(&recorder, || {
        let read = |family: &str| worker_sample(&handle.render(), family, url);
        let registry = WorkerRegistry::new();
        let live: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .circuit_breaker_config(opens_on_one_failure())
                .build(),
        );
        let id = registry
            .register(Arc::clone(&live))
            .expect("a fresh URL registers");
        live.record_outcome(503);
        assert_eq!(read("smg_worker_http2"), Some(0.0));
        assert_eq!(read("smg_worker_cb_state"), Some(1.0));
        assert_eq!(read("smg_worker_cb_consecutive_failures"), Some(1.0));

        assert!(registry.replace(&id, replacement(true, opens_on_one_failure())));
        assert_eq!(read("smg_worker_http2"), Some(1.0));
        assert_eq!(read("smg_worker_cb_state"), Some(1.0));
        assert_eq!(read("smg_worker_cb_consecutive_failures"), Some(1.0));
        assert_eq!(read("smg_worker_health"), Some(1.0));

        let other = CircuitBreakerConfig {
            failure_threshold: 2,
            ..opens_on_one_failure()
        };
        assert!(registry.replace(&id, replacement(false, other)));
        assert_eq!(read("smg_worker_http2"), Some(0.0));
        assert_eq!(read("smg_worker_cb_state"), Some(0.0));
        assert_eq!(read("smg_worker_cb_consecutive_failures"), Some(0.0));
        assert_eq!(read("smg_worker_health"), Some(1.0));

        let trips = series_of(&handle.render(), url)
            .into_iter()
            .find(|line| {
                line.starts_with("smg_worker_cb_transitions_total{")
                    && line.contains("from=\"closed\",to=\"open\"")
            })
            .expect("the trip's counter");
        assert!(trips.ends_with(" 1"), "{trips}");
    });
}

/// A refresh of a worker's model list leaves the model index as the
/// registration wrote it. So a removal takes the worker out of every entry
/// that holds it, and not out of the models it lists that minute alone: the
/// lookup by a model its refresh left out must not hand out a removed worker.
#[test]
fn a_removed_worker_leaves_every_model_the_index_holds_it_under() {
    let registry = WorkerRegistry::new();
    let endpoint: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://endpoint:8080")
            .models(vec![ModelCard::new("qwen"), ModelCard::new("llama")])
            .build(),
    );
    let id = registry.register(Arc::clone(&endpoint)).unwrap();
    assert!(registry.refresh_models(&endpoint, vec![ModelCard::new("llama")]));
    assert_eq!(registry.get_by_model("qwen").len(), 1);
    registry.set_model_retry_config("qwen", RetryConfig::default(), true);

    assert!(registry.remove(&id).is_some());
    assert!(registry.get_by_model("llama").is_empty());
    assert!(registry.get_by_model("qwen").is_empty());
    // The registry holds no worker now, so it lists no model, and the
    // left-out model's retry settings went with its last worker.
    assert!(registry.get_models().is_empty());
    assert!(registry.get_retry_config("qwen").is_none());
}

/// The same for a worker registered without model cards: it is indexed
/// under its primary model alone, and a list it discovers later leaves
/// that model out.
#[test]
fn a_removed_worker_registered_without_cards_leaves_its_primary_model() {
    let registry = WorkerRegistry::new();
    let labelled: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://labelled:8080")
            .label("model_id", "llama")
            .build(),
    );
    let id = registry.register(Arc::clone(&labelled)).unwrap();
    assert!(registry.refresh_models(&labelled, vec![ModelCard::new("qwen")]));

    assert!(registry.remove(&id).is_some());
    assert!(registry.get_by_model("llama").is_empty());
}

/// A replacement takes the old handle out of every entry that holds it
/// too: the worker that takes its place stands under the models it lists,
/// and the old handle under none.
#[test]
fn a_replaced_worker_leaves_every_model_the_index_holds_it_under() {
    let registry = WorkerRegistry::new();
    let endpoint: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://endpoint:8080")
            .models(vec![ModelCard::new("qwen"), ModelCard::new("llama")])
            .build(),
    );
    let id = registry.register(Arc::clone(&endpoint)).unwrap();
    assert!(registry.refresh_models(&endpoint, vec![ModelCard::new("llama")]));
    let next: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://endpoint:8080")
            .models(vec![ModelCard::new("llama")])
            .build(),
    );

    assert!(registry.replace(&id, Arc::clone(&next)));
    assert!(registry.get_by_model("qwen").is_empty());
    let llama = registry.get_by_model("llama");
    assert_eq!(llama.len(), 1);
    assert!(Arc::ptr_eq(&llama[0], &next));
}

/// An alias goes with its model: the alias of a model the refresh left out
/// resolves to nothing once the worker is removed.
#[test]
fn a_removed_workers_alias_goes_with_a_model_its_refresh_left_out() {
    let registry = WorkerRegistry::new();
    let endpoint: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://endpoint:8080")
            .models(vec![
                ModelCard::new("qwen").with_alias("qwen-latest"),
                ModelCard::new("llama"),
            ])
            .build(),
    );
    let id = registry.register(Arc::clone(&endpoint)).unwrap();
    assert!(registry.refresh_models(&endpoint, vec![ModelCard::new("llama")]));

    assert!(registry.remove(&id).is_some());
    assert!(registry.resolve_model_alias("qwen-latest").is_none());
    assert!(registry.get_by_model("qwen-latest").is_empty());
}

/// A replacement reads the aliases from the alias index too: the alias of
/// a model the old worker's refresh left out goes with the old handle.
#[test]
fn a_replaced_workers_alias_goes_with_a_model_its_refresh_left_out() {
    let registry = WorkerRegistry::new();
    let endpoint: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://endpoint:8080")
            .models(vec![
                ModelCard::new("qwen").with_alias("qwen-latest"),
                ModelCard::new("llama"),
            ])
            .build(),
    );
    let id = registry.register(Arc::clone(&endpoint)).unwrap();
    assert!(registry.refresh_models(&endpoint, vec![ModelCard::new("llama")]));
    let next: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://endpoint:8080")
            .models(vec![ModelCard::new("llama")])
            .build(),
    );

    assert!(registry.replace(&id, next));
    assert!(registry.resolve_model_alias("qwen-latest").is_none());
    assert!(registry.get_by_model("qwen-latest").is_empty());
}

/// An alias stays while a peer still serves its model, though a refresh
/// wrote the peer's cards with no alias: the cards of the minute cannot
/// tell an alias the peer registered from a stale one. The alias goes
/// with the model's last worker.
#[test]
fn an_alias_stays_while_a_refreshed_peer_still_serves_its_model() {
    let registry = WorkerRegistry::new();
    let (first, second) = two_refreshed_workers_of_an_aliased_model(&registry);

    assert!(registry.remove(&first).is_some());
    assert_eq!(
        registry.resolve_model_alias("qwen-latest").as_deref(),
        Some("qwen")
    );
    assert!(registry.remove(&second).is_some());
    assert!(registry.resolve_model_alias("qwen-latest").is_none());
}

/// The same through a replacement whose cards do not declare the alias:
/// the refreshed peer registered it, and it stays.
#[test]
fn an_alias_stays_through_a_replacement_while_a_refreshed_peer_serves_its_model() {
    let registry = WorkerRegistry::new();
    let (first, _second) = two_refreshed_workers_of_an_aliased_model(&registry);
    let next: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://first:8080")
            .model(ModelCard::new("qwen"))
            .build(),
    );

    assert!(registry.replace(&first, next));
    assert_eq!(
        registry.resolve_model_alias("qwen-latest").as_deref(),
        Some("qwen")
    );
}

/// Two workers registered with one aliased model, each refreshed since:
/// a refresh writes the cards an endpoint's model list names, with no
/// alias. Returns their ids.
fn two_refreshed_workers_of_an_aliased_model(registry: &WorkerRegistry) -> (WorkerId, WorkerId) {
    let registered = |url: &str| {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("qwen").with_alias("qwen-latest"))
                .build(),
        );
        let id = registry.register(Arc::clone(&worker)).unwrap();
        assert!(registry.refresh_models(&worker, vec![ModelCard::new("qwen")]));
        id
    };
    (
        registered("http://first:8080"),
        registered("http://second:8080"),
    )
}

/// A removal frees its URL ahead of its walk of the index, so a
/// registration of the same URL can stand in a model's slice already.
/// The eviction takes the handle it is given, and the newcomer keeps its
/// place.
#[test]
fn an_eviction_from_the_index_takes_the_handle_and_not_its_address() {
    let registry = WorkerRegistry::new();
    let built = || -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new("http://endpoint:8080")
                .model(ModelCard::new("llama"))
                .build(),
        )
    };
    let (leaving, newcomer) = (built(), built());
    // What the slice holds once the newcomer registered at the freed URL:
    // its handle, in the place of the one that leaves.
    registry.add_worker_to_model_index("llama", Arc::clone(&newcomer));

    registry.remove_worker_from_model_index("llama", &leaving);
    let llama = registry.get_by_model("llama");
    assert_eq!(llama.len(), 1);
    assert!(Arc::ptr_eq(&llama[0], &newcomer));

    registry.remove_worker_from_model_index("llama", &newcomer);
    assert!(registry.get_by_model("llama").is_empty());
}
