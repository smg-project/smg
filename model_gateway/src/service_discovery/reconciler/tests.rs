use openai_protocol::worker::WorkerType;

use super::*;
use crate::{
    server::http_client, service_discovery::testing::create_test_app_context, worker::endpoint_key,
};

fn key(url: &str) -> EndpointKey {
    endpoint_key(url).expect(url)
}

/// A published worker with nothing but an endpoint and an instance id.
fn published(url: &str, discovery_id: &str) -> DiscoveredWorker {
    DiscoveredWorker {
        discovery_id: discovery_id.to_string(),
        endpoint: Endpoint::parse_with_rank(url).expect(url).0,
        worker_type: WorkerType::Regular,
        bootstrap_port: None,
        model_id_override: None,
        kv_connector: None,
        kv_role: None,
        kv_engine_id: None,
        compat_labels: BTreeMap::new(),
    }
}

fn desired_state_of(workers: &[DiscoveredWorker]) -> DesiredState {
    DesiredState::from_workers(workers.iter().cloned())
}

fn owned(url: &str, discovery_id: &str) -> OwnedWorker {
    owned_rank(url, discovery_id, url, 1)
}

/// One rank of a DP group: same endpoint and instance, its own registry id
/// and revision. Registered from the record `published(url, discovery_id)`
/// builds, so it is current against that record.
fn owned_rank(url: &str, discovery_id: &str, worker_id: &str, revision: u64) -> OwnedWorker {
    OwnedWorker {
        worker_id: WorkerId::from_string(worker_id.to_string()),
        discovery_id: Some(discovery_id.to_string()),
        spec_hash: Some(published(url, discovery_id).fingerprint()),
        endpoint: Endpoint::parse_with_rank(url).expect(url).0,
        revision,
    }
}

/// What `canonical_host_port` used to assert, now served by the shared
/// parser — plus the spellings it got wrong: an uppercase scheme kept its
/// prefix, and `@` inside a path or userinfo truncated the key.
#[test]
fn ownership_grouping_uses_the_shared_canonical_key() {
    for spelling in [
        "10.0.0.1:8080",
        "http://10.0.0.1:8080",
        "grpc://10.0.0.1:8080@2",
        "10.0.0.1:8080@0",
        "HTTPS://10.0.0.1:8080",
    ] {
        assert_eq!(key(spelling).as_str(), "10.0.0.1:8080", "{spelling}");
    }
    assert_eq!(key("ipc:///tmp/a@b.sock").as_str(), "/tmp/a@b.sock");
    assert_eq!(key("http://user@host:8080").as_str(), "user@host:8080");
}

#[test]
fn desired_state_keeps_the_first_record_for_an_endpoint() {
    let state = desired_state_of(&[
        published("10.0.0.1:8080", "first"),
        published("http://10.0.0.1:8080", "second"),
    ]);
    assert_eq!(state.workers.len(), 1);
    assert_eq!(state.workers[&key("10.0.0.1:8080")].discovery_id, "first");
}

#[test]
fn test_compute_actions_adds_missing_workers() {
    let desired = desired_state_of(&[
        published("10.0.0.1:8080", "a:8080"),
        published("10.0.0.1:8081", "a:8081"),
    ]);
    let actions = compute_actions(&desired, &[]);
    assert_eq!(actions.add.len(), 2);
    assert!(actions.remove.is_empty());
}

#[test]
fn test_compute_actions_unchanged_instance_is_kept() {
    let desired = desired_state_of(&[published("10.0.0.1:8080", "u1")]);
    let registered = [owned("10.0.0.1:8080", "u1")];
    let actions = compute_actions(&desired, &registered);
    assert!(actions.add.is_empty());
    assert!(actions.remove.is_empty());
}

/// The same instance republished with another configuration is updated:
/// a corrected bootstrap port, model or KV setting reaches the registry
/// instead of waiting for the instance itself to be recreated. Only the
/// add is submitted. It is an Upsert that replaces the registration in
/// place; a removal beside it would delete the updated worker once its
/// drain ended.
#[test]
fn test_compute_actions_same_instance_configuration_change_is_updated_in_place() {
    let registered = [owned("10.0.0.1:8080", "u1")];
    for change in [
        |w: &mut DiscoveredWorker| w.kv_connector = Some("NixlConnector".to_string()),
        |w: &mut DiscoveredWorker| w.model_id_override = Some("llama".to_string()),
        |w: &mut DiscoveredWorker| w.kv_engine_id = Some("engine-1".to_string()),
    ] {
        let mut worker = published("10.0.0.1:8080", "u1");
        change(&mut worker);
        let actions = compute_actions(&desired_state_of(&[worker.clone()]), &registered);
        assert!(actions.remove.is_empty(), "{worker:?}");
        assert_eq!(actions.add, vec![worker]);
    }
}

/// Every rank of a DP group carries the same labels, so a configuration
/// change updates the group with the one add for its endpoint.
#[test]
fn test_compute_actions_configuration_change_updates_a_dp_group_with_one_add() {
    let registered = [
        owned_rank("10.0.0.1:8080", "u1", "w@0", 4),
        owned_rank("10.0.0.1:8080", "u1", "w@1", 9),
    ];
    let mut worker = published("10.0.0.1:8080", "u1");
    worker.kv_connector = Some("NixlConnector".to_string());
    let actions = compute_actions(&desired_state_of(&[worker]), &registered);
    assert!(actions.remove.is_empty());
    assert_eq!(actions.add.len(), 1);
}

/// An update that lands at a different URL (here, the published spelling
/// gained an explicit scheme) leaves the old registration beside the new.
/// Once the new one is current, the old one is removed, and its URL is not
/// the current one's, so the removal cannot reach the current worker.
#[test]
fn test_compute_actions_removes_a_stale_registration_beside_a_current_one() {
    let desired = desired_state_of(&[published("grpc://10.0.0.1:8080", "u1")]);
    let registered = [
        owned("grpc://10.0.0.1:8080", "u1"),
        owned("http://10.0.0.1:8080", "u1"),
    ];
    let actions = compute_actions(&desired, &registered);
    assert!(actions.add.is_empty());
    assert_eq!(actions.remove.len(), 1);
    assert_eq!(actions.remove[0].endpoint.render(), "http://10.0.0.1:8080");
}

#[test]
fn test_compute_actions_removes_workers_no_longer_published() {
    let desired = desired_state_of(&[published("10.0.0.1:8080", "u1")]);
    let registered = [owned("10.0.0.1:8080", "u1"), owned("10.0.0.2:8080", "u2")];
    let actions = compute_actions(&desired, &registered);
    assert!(actions.add.is_empty());
    assert_eq!(actions.remove.len(), 1);
    assert_eq!(actions.remove[0].endpoint.key().as_str(), "10.0.0.2:8080");
}

#[test]
fn test_compute_actions_new_instance_at_same_endpoint_is_replaced() {
    // A new instance at an unchanged address — for Kubernetes, a Pod
    // recreated at a stable IP, not a container restarting inside one — is
    // removed and the new one registered.
    // The removal also covers a scheme-flipped sibling the Upsert cannot
    // replace.
    let desired = desired_state_of(&[published("10.0.0.1:8080", "new")]);
    let registered = [owned("10.0.0.1:8080", "old")];
    let actions = compute_actions(&desired, &registered);
    assert_eq!(actions.add.len(), 1);
    assert_eq!(actions.add[0].discovery_id, "new");
    assert_eq!(actions.remove.len(), 1);
    assert_eq!(actions.remove[0].discovery_id.as_deref(), Some("old"));
}

#[test]
fn test_compute_actions_dp_ranks_removed_once() {
    let registered = [
        owned_rank("10.0.0.1:8080", "u1", "w@0", 1),
        owned_rank("10.0.0.1:8080", "u1", "w@1", 1),
    ];
    let actions = compute_actions(&DesiredState::default(), &registered);
    assert_eq!(actions.remove.len(), 1);
    assert_eq!(actions.remove[0].endpoint.key().as_str(), "10.0.0.1:8080");
    assert_eq!(actions.remove[0].guards.len(), 2);
    assert!(actions.add.is_empty());
}

/// The group collapses to one removal job, but every rank must keep its
/// own revision. Carrying a single revision retained only the ranks that
/// happened to share it and left the others registered against an instance
/// that was already gone.
#[test]
fn dp_ranks_keep_their_own_revision_when_diverged() {
    let registered = [
        owned_rank("10.0.0.1:8080", "u1", "w@0", 7),
        owned_rank("10.0.0.1:8080", "u1", "w@1", 2),
    ];
    let actions = compute_actions(&DesiredState::default(), &registered);
    assert_eq!(actions.remove.len(), 1, "one job per canonical endpoint");
    let guards: Vec<(&str, u64)> = actions.remove[0]
        .guards
        .iter()
        .map(|(id, revision)| (id.as_str(), *revision))
        .collect();
    assert_eq!(guards, vec![("w@0", 7), ("w@1", 2)]);
}

#[test]
fn diverged_ranks_do_not_collapse_on_equal_revisions() {
    let registered = [
        owned_rank("10.0.0.1:8080", "u1", "w@0", 3),
        owned_rank("10.0.0.1:8080", "u1", "w@1", 3),
        owned_rank("10.0.0.1:8081", "u1", "x@0", 3),
    ];
    let actions = compute_actions(&DesiredState::default(), &registered);
    assert_eq!(actions.remove.len(), 2, "one job per canonical endpoint");
    assert_eq!(actions.remove[0].guards.len(), 2);
    assert_eq!(actions.remove[1].guards.len(), 1);
}

fn register_with_labels(app_context: &AppContext, url: &str, labels: &[(&str, &str)]) {
    use openai_protocol::model_card::ModelCard;

    use crate::worker::BasicWorkerBuilder;

    let labels: HashMap<String, String> = labels
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let worker = Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new("m"))
            .labels(labels)
            .build(),
    );
    app_context.worker_registry.register(worker).unwrap();
}

/// Ownership is the provider label alone: a manual worker carries none,
/// and a worker another provider registered is not this provider's to
/// remove.
#[test]
fn owned_workers_are_scoped_by_provider_label_and_origin() {
    let app_context = create_test_app_context();
    register_with_labels(
        &app_context,
        "http://10.0.0.1:8080",
        &[
            (DISCOVERY_PROVIDER_LABEL, "kubernetes"),
            (DISCOVERY_ID_LABEL, "uid-1:8080"),
        ],
    );
    register_with_labels(
        &app_context,
        "http://10.0.0.2:8080",
        &[
            (DISCOVERY_PROVIDER_LABEL, "file"),
            (DISCOVERY_ID_LABEL, "f-1"),
        ],
    );
    register_with_labels(&app_context, "http://10.0.0.3:8080", &[]);

    let owned = owned_workers(&app_context, DiscoveryKind::Kubernetes);
    assert_eq!(owned.len(), 1);
    assert_eq!(owned[0].endpoint.key().as_str(), "10.0.0.1:8080");
    assert_eq!(owned[0].discovery_id.as_deref(), Some("uid-1:8080"));
}

/// The fingerprint stamped at registration is read back, and a worker
/// carrying the current one is kept.
#[test]
fn owned_workers_read_back_the_spec_hash() {
    let app_context = create_test_app_context();
    let worker = published("10.0.0.1:8080", "uid-1:8080");
    register_with_labels(
        &app_context,
        "http://10.0.0.1:8080",
        &[
            (DISCOVERY_PROVIDER_LABEL, "kubernetes"),
            (DISCOVERY_ID_LABEL, "uid-1:8080"),
            (DISCOVERY_SPEC_HASH_LABEL, &worker.fingerprint()),
        ],
    );
    let owned = owned_workers(&app_context, DiscoveryKind::Kubernetes);
    assert_eq!(owned[0].spec_hash, Some(worker.fingerprint()));

    let actions = compute_actions(&desired_state_of(&[worker]), &owned);
    assert!(actions.add.is_empty());
    assert!(actions.remove.is_empty());
}

/// An owned worker missing its fingerprint label cannot be shown current,
/// so it is updated in place and comes back labelled properly.
#[test]
fn an_owned_worker_without_a_spec_hash_is_updated_in_place() {
    let app_context = create_test_app_context();
    register_with_labels(
        &app_context,
        "http://10.0.0.1:8080",
        &[
            (DISCOVERY_PROVIDER_LABEL, "kubernetes"),
            (DISCOVERY_ID_LABEL, "uid-1:8080"),
        ],
    );
    let owned = owned_workers(&app_context, DiscoveryKind::Kubernetes);
    assert_eq!(owned[0].spec_hash, None);

    let desired = desired_state_of(&[published("10.0.0.1:8080", "uid-1:8080")]);
    let actions = compute_actions(&desired, &owned);
    assert!(actions.remove.is_empty());
    assert_eq!(actions.add.len(), 1);
}

/// An owned worker missing its id label cannot match anything published,
/// so it is replaced and comes back labelled properly.
#[test]
fn an_owned_worker_without_an_id_is_replaced() {
    let app_context = create_test_app_context();
    register_with_labels(
        &app_context,
        "http://10.0.0.1:8080",
        &[(DISCOVERY_PROVIDER_LABEL, "kubernetes")],
    );
    let owned = owned_workers(&app_context, DiscoveryKind::Kubernetes);
    assert_eq!(owned[0].discovery_id, None);

    let desired = desired_state_of(&[published("10.0.0.1:8080", "uid-1:8080")]);
    let actions = compute_actions(&desired, &owned);
    assert_eq!(actions.remove.len(), 1);
    assert_eq!(actions.add.len(), 1);
}

/// The case a defaulted `""` got wrong. Kubernetes never publishes an empty
/// id, but a provider with an optional id field could, and `"" == ""` then
/// matched it to an owned worker whose label was missing — so that worker
/// was never replaced or relabelled. Snapshot validation will reject an
/// empty id outright; this keeps the diff correct even without it.
#[test]
fn an_empty_published_id_does_not_match_a_missing_label() {
    let app_context = create_test_app_context();
    register_with_labels(
        &app_context,
        "http://10.0.0.1:8080",
        &[(DISCOVERY_PROVIDER_LABEL, "kubernetes")],
    );
    let owned = owned_workers(&app_context, DiscoveryKind::Kubernetes);

    let desired = desired_state_of(&[published("10.0.0.1:8080", "")]);
    let actions = compute_actions(&desired, &owned);
    assert_eq!(actions.remove.len(), 1, "the unlabelled worker is replaced");
    assert_eq!(actions.add.len(), 1);
}

#[test]
fn owned_workers_exclude_mesh_imported_workers() {
    // A peer's discovered worker arrives via mesh sync carrying the peer's
    // provider label; it is absent from this node's snapshot, so without
    // the Local-origin filter the reconciler would remove it every pass.
    let app_context = create_test_app_context();

    let mut spec = WorkerSpec::new("http://10.0.0.3:8080");
    spec.labels.insert(
        DISCOVERY_PROVIDER_LABEL.to_string(),
        "kubernetes".to_string(),
    );
    let state = crate::worker::RemoteWorkerState {
        worker_id: "peer-w1".to_string(),
        model_id: "m".to_string(),
        url: "http://10.0.0.3:8080".to_string(),
        health: true,
        version: 0,
        spec: serde_json::to_vec(&spec).unwrap(),
    };
    app_context.worker_registry.on_remote_worker_state(&state);
    assert!(app_context
        .worker_registry
        .get_by_url("http://10.0.0.3:8080")
        .is_some());

    assert!(owned_workers(&app_context, DiscoveryKind::Kubernetes).is_empty());
}

#[test]
fn build_worker_spec_carries_the_record_and_stamps_provenance() {
    let app_context = create_test_app_context();
    let worker = DiscoveredWorker {
        worker_type: WorkerType::Prefill,
        bootstrap_port: Some(9080),
        model_id_override: Some("llama".to_string()),
        kv_connector: Some("MooncakeConnector".to_string()),
        kv_role: Some("kv_producer".to_string()),
        kv_engine_id: Some("engine-1".to_string()),
        ..published("10.0.0.1:8081", "uid-1:8081")
    };
    let spec = build_worker_spec(&worker, DiscoveryKind::Kubernetes, &app_context);

    assert_eq!(spec.url, "10.0.0.1:8081");
    assert_eq!(spec.worker_type, WorkerType::Prefill);
    assert_eq!(spec.bootstrap_port, Some(9080));
    assert_eq!(spec.kv_connector.as_deref(), Some("MooncakeConnector"));
    assert_eq!(spec.kv_role.as_deref(), Some("kv_producer"));
    assert_eq!(spec.kv_engine_id.as_deref(), Some("engine-1"));
    assert_eq!(
        spec.labels.get("served_model_name").map(String::as_str),
        Some("llama")
    );
    assert_eq!(
        spec.labels
            .get(DISCOVERY_PROVIDER_LABEL)
            .map(String::as_str),
        Some("kubernetes")
    );
    assert_eq!(
        spec.labels.get(DISCOVERY_ID_LABEL).map(String::as_str),
        Some("uid-1:8081")
    );
    // The discovered budget: the context's default success threshold
    // of 2, twenty attempts per success.
    assert_eq!(
        spec.max_connection_attempts,
        app_context
            .router_config
            .connection_attempt_budget(RouterConfig::DISCOVERED_WORKER_ATTEMPTS_PER_SUCCESS)
    );
    assert_eq!(spec.max_connection_attempts, 40);
    // Taken from the record, so the API key and retry budget injected into
    // the spec cannot move it.
    assert_eq!(
        spec.labels.get(DISCOVERY_SPEC_HASH_LABEL),
        Some(&worker.fingerprint())
    );
}

/// Compatibility labels are opaque here: any key a provider supplies is
/// written verbatim. The keys below are deliberately not Kubernetes' — the
/// reconciler has no notion of which provider's labels it is carrying.
#[test]
fn compat_labels_are_written_verbatim_but_cannot_claim_provenance() {
    let app_context = create_test_app_context();
    let worker = DiscoveredWorker {
        compat_labels: BTreeMap::from([
            ("example.com/source-ref".to_string(), "rec-7".to_string()),
            (DISCOVERY_PROVIDER_LABEL.to_string(), "forged".to_string()),
            (
                format!("{DISCOVERY_LABEL_PREFIX}extra"),
                "forged".to_string(),
            ),
        ]),
        ..published("10.0.0.1:8080", "rec-7")
    };
    let spec = build_worker_spec(&worker, DiscoveryKind::Kubernetes, &app_context);

    assert_eq!(
        spec.labels
            .get("example.com/source-ref")
            .map(String::as_str),
        Some("rec-7")
    );
    assert_eq!(
        spec.labels
            .get(DISCOVERY_PROVIDER_LABEL)
            .map(String::as_str),
        Some("kubernetes"),
        "a provider cannot forge ownership through its compatibility labels"
    );
    assert!(!spec
        .labels
        .contains_key(&format!("{DISCOVERY_LABEL_PREFIX}extra")));
}

/// `?actions` is how a reconcile pass gets debugged, and these structs
/// derive `Debug`. A provider-published credential must not come out.
#[test]
fn debug_output_masks_a_provider_published_credential() {
    let desired = desired_state_of(&[published("http://user:pass@10.0.0.1:8080", "u1")]);
    let actions = compute_actions(&desired, &[]);
    for rendered in [format!("{desired:?}"), format!("{actions:?}")] {
        assert!(!rendered.contains("pass"), "{rendered}");
    }
}

#[test]
fn test_deregistration_reconciled_metric_label() {
    assert_eq!(metrics_labels::DEREGISTRATION_RECONCILED, "reconciled");
}

#[tokio::test]
async fn test_reconcile_without_job_queue_is_safe() {
    let app_context = create_test_app_context();
    let desired = desired_state_of(&[published("10.0.0.1:8080", "u1")]);
    reconcile(
        &desired,
        DiscoveryKind::Kubernetes,
        &app_context,
        time::Instant::now(),
    )
    .await;
}

/// A pass that waits for a queue slot with an addition in hand holds that
/// URL's claim while it waits: a `POST` of the URL during the wait is told
/// a registration is in flight. The pass used to read the URL's status
/// first and submit without a claim, so such a `POST` was admitted and the
/// pass then queued a second registration of the URL beside it.
#[tokio::test]
async fn an_addition_waiting_for_a_queue_slot_holds_its_urls_claim() {
    use crate::workflow::{JobQueue, JobQueueConfig};

    let app_context = create_test_app_context();
    // One slot and no runner: the dispatcher takes one job out and waits for
    // a permit that never comes, and the next job fills the slot.
    let queue = JobQueue::new(
        JobQueueConfig {
            queue_capacity: 1,
            max_concurrent_jobs: 0,
        },
        Arc::downgrade(&app_context),
    );
    app_context.worker_job_queue.set(Arc::clone(&queue)).ok();
    let registration = |url: &str, registration_mode| Job::AddWorker {
        config: Box::new(WorkerSpec::new(url)),
        registration_mode,
    };
    for other in 1.. {
        tokio::task::yield_now().await;
        if queue.get_load_info().0 == 1 {
            break;
        }
        let url = format!("http://10.0.0.{other}:8080");
        queue
            .submit(registration(&url, WorkerRegistrationMode::Upsert))
            .await
            .expect("a slot");
    }

    let worker = published("http://10.0.9.1:8080", "u1");
    let url = worker.endpoint.render();
    let desired = desired_state_of(&[worker]);
    let mut pass = std::pin::pin!(reconcile(
        &desired,
        DiscoveryKind::Kubernetes,
        &app_context,
        time::Instant::now(),
    ));
    assert!(
        matches!(futures::poll!(pass.as_mut()), std::task::Poll::Pending),
        "the pass waits for a queue slot"
    );

    assert!(
        !queue.claim_if_idle(&registration(&url, WorkerRegistrationMode::CreateOnly)),
        "the waiting addition holds its URL: a POST of it is told to poll"
    );
}

/// A prefill worker discovered without a bootstrap annotation carries no
/// bootstrap port into its spec and is handed the default port on the
/// request path (here a vLLM worker on the Mooncake connector, as the
/// provider reports it). Registered, it is probed on that default port:
/// with nothing listening there, the failed probe is counted like a
/// registered port's.
#[tokio::test]
async fn a_discovered_prefill_worker_without_a_bootstrap_annotation_is_probed_on_the_default_port()
{
    use std::{
        net::{SocketAddr, TcpStream},
        sync::Arc,
        time::Duration,
    };

    use openai_protocol::worker::RuntimeType;
    use wfaas::{StepExecutor, StepResult, WorkflowContext, WorkflowInstanceId};

    use crate::{
        worker::{ConnectionMode, DEFAULT_BOOTSTRAP_PORT},
        workflow::{
            data::WorkerKind,
            steps::{create_worker_workflow_data, CreateLocalWorkerStep, RegisterWorkersStep},
        },
    };

    // A listener on the default port would make a passed probe and an
    // absent one look alike, so the test stands down then.
    let default_port = SocketAddr::from(([127, 0, 0, 1], DEFAULT_BOOTSTRAP_PORT));
    if TcpStream::connect_timeout(&default_port, Duration::from_millis(500)).is_ok() {
        return;
    }

    let app_context = create_test_app_context();
    let worker = DiscoveredWorker {
        worker_type: WorkerType::Prefill,
        kv_connector: Some("MooncakeConnector".to_string()),
        kv_role: Some("kv_producer".to_string()),
        kv_engine_id: Some("engine-1".to_string()),
        ..published("127.0.0.1:18081", "uid-1:18081")
    };
    let mut spec = build_worker_spec(&worker, DiscoveryKind::Kubernetes, &app_context);
    assert_eq!(spec.bootstrap_port, None, "no annotation, no port");
    // The runtime the worker's engine reports when the registration detects it.
    spec.runtime_type = RuntimeType::Vllm;

    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);

    let mut data = create_worker_workflow_data(
        spec,
        WorkerRegistrationMode::Upsert,
        Arc::clone(&app_context),
    );
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(ConnectionMode::Http);
    data.http_client_handle = Some(Arc::new(http_client()));
    let mut ctx = WorkflowContext::new(WorkflowInstanceId::new(), data);
    assert_eq!(
        CreateLocalWorkerStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    assert_eq!(
        RegisterWorkersStep.execute(&mut ctx).await.unwrap(),
        StepResult::Success
    );
    let registered = &ctx.data.actual_workers.as_ref().unwrap()[0];
    assert_eq!(
        registered.bootstrap_port(),
        None,
        "the spec keeps what was discovered"
    );

    // The probe runs off the path, within its own timeout.
    let deadline = time::Instant::now() + Duration::from_secs(5);
    let counted = loop {
        let rendered = handle.render();
        if let Some(line) = rendered
            .lines()
            .find(|line| line.starts_with("smg_pd_rendezvous_failures_total{"))
        {
            break line.to_string();
        }
        assert!(
            time::Instant::now() < deadline,
            "the failed probe of the default port is counted within its timeout:\n{rendered}"
        );
        time::sleep(Duration::from_millis(20)).await;
    };
    assert!(
        counted.contains(r#"leg="prefill""#)
            && counted.contains(r#"phase="probe""#)
            && counted.ends_with(" 1"),
        "{counted}"
    );
}
