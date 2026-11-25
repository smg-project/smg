use std::{slice, sync::Arc};

use openai_protocol::worker::{RuntimeType, WorkerStatus, WorkerType};
use smg_worker::{
    identity::endpoint::endpoint_key,
    pd::pairing::{PairingMismatch, PdPairing, PdPairingMode},
    test_support::{assert_metric, render_with_recorder},
};
use tokio::runtime::Builder;

use super::*;
use crate::testing::{Queued, TestHost, STAMPED_API_KEY, STAMPED_ATTEMPTS};

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
        pairing_protocol: None,
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

fn register_with_labels(host: &TestHost, url: &str, labels: &[(&str, &str)]) {
    use openai_protocol::model_card::ModelCard;
    use smg_worker::test_support::BasicWorkerBuilder;

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
    host.registry.register(worker).unwrap();
}

/// Ownership is the provider label alone: a manual worker carries none,
/// and a worker another provider registered is not this provider's to
/// remove.
#[test]
fn owned_workers_are_scoped_by_provider_label_and_origin() {
    let host = TestHost::without_a_queue();
    register_with_labels(
        &host,
        "http://10.0.0.1:8080",
        &[
            (DISCOVERY_PROVIDER_LABEL, "kubernetes"),
            (DISCOVERY_ID_LABEL, "uid-1:8080"),
        ],
    );
    register_with_labels(
        &host,
        "http://10.0.0.2:8080",
        &[
            (DISCOVERY_PROVIDER_LABEL, "file"),
            (DISCOVERY_ID_LABEL, "f-1"),
        ],
    );
    register_with_labels(&host, "http://10.0.0.3:8080", &[]);

    let owned = owned_workers(&*host, DiscoveryKind::Kubernetes);
    assert_eq!(owned.len(), 1);
    assert_eq!(owned[0].endpoint.key().as_str(), "10.0.0.1:8080");
    assert_eq!(owned[0].discovery_id.as_deref(), Some("uid-1:8080"));
}

/// The fingerprint stamped at registration is read back, and a worker
/// carrying the current one is kept.
#[test]
fn owned_workers_read_back_the_spec_hash() {
    let host = TestHost::without_a_queue();
    let worker = published("10.0.0.1:8080", "uid-1:8080");
    register_with_labels(
        &host,
        "http://10.0.0.1:8080",
        &[
            (DISCOVERY_PROVIDER_LABEL, "kubernetes"),
            (DISCOVERY_ID_LABEL, "uid-1:8080"),
            (DISCOVERY_SPEC_HASH_LABEL, &worker.fingerprint()),
        ],
    );
    let owned = owned_workers(&*host, DiscoveryKind::Kubernetes);
    assert_eq!(owned[0].spec_hash, Some(worker.fingerprint()));

    let actions = compute_actions(&desired_state_of(&[worker]), &owned);
    assert!(actions.add.is_empty());
    assert!(actions.remove.is_empty());
}

/// Today's behaviour, held here so that a change of it is a choice and not
/// an accident: a pass reads no worker state. A worker that left the
/// published list and is listed again while its removal still drains it
/// counts as registered, so the pass queues nothing for it. The removal
/// goes on and removes it, and the first pass after that registers it again.
#[test]
fn a_worker_listed_again_while_it_drains_counts_as_registered() {
    let host = TestHost::without_a_queue();
    let worker = published("10.0.0.1:8080", "f-1");
    register_with_labels(
        &host,
        "http://10.0.0.1:8080",
        &[
            (DISCOVERY_PROVIDER_LABEL, "file"),
            (DISCOVERY_ID_LABEL, "f-1"),
            (DISCOVERY_SPEC_HASH_LABEL, &worker.fingerprint()),
        ],
    );
    // What the removal's drain step writes while its window runs.
    for registered in host.registry.get_all() {
        registered.set_status(WorkerStatus::Draining);
    }

    let owned = owned_workers(&*host, DiscoveryKind::File);
    assert_eq!(owned.len(), 1);
    let actions = compute_actions(&desired_state_of(&[worker]), &owned);
    assert!(actions.add.is_empty());
    assert!(actions.remove.is_empty());
}

/// An owned worker missing its fingerprint label cannot be shown current,
/// so it is updated in place and comes back labelled properly.
#[test]
fn an_owned_worker_without_a_spec_hash_is_updated_in_place() {
    let host = TestHost::without_a_queue();
    register_with_labels(
        &host,
        "http://10.0.0.1:8080",
        &[
            (DISCOVERY_PROVIDER_LABEL, "kubernetes"),
            (DISCOVERY_ID_LABEL, "uid-1:8080"),
        ],
    );
    let owned = owned_workers(&*host, DiscoveryKind::Kubernetes);
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
    let host = TestHost::without_a_queue();
    register_with_labels(
        &host,
        "http://10.0.0.1:8080",
        &[(DISCOVERY_PROVIDER_LABEL, "kubernetes")],
    );
    let owned = owned_workers(&*host, DiscoveryKind::Kubernetes);
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
    let host = TestHost::without_a_queue();
    register_with_labels(
        &host,
        "http://10.0.0.1:8080",
        &[(DISCOVERY_PROVIDER_LABEL, "kubernetes")],
    );
    let owned = owned_workers(&*host, DiscoveryKind::Kubernetes);

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
    let host = TestHost::without_a_queue();

    let mut spec = WorkerSpec::new("http://10.0.0.3:8080");
    spec.labels.insert(
        DISCOVERY_PROVIDER_LABEL.to_string(),
        "kubernetes".to_string(),
    );
    let state = smg_worker::registry::RemoteWorkerState {
        worker_id: "peer-w1".to_string(),
        model_id: "m".to_string(),
        url: "http://10.0.0.3:8080".to_string(),
        health: true,
        version: 0,
        spec: serde_json::to_vec(&spec).unwrap(),
    };
    host.registry.on_remote_worker_state(&state);
    assert!(host.registry.get_by_url("http://10.0.0.3:8080").is_some());

    assert!(owned_workers(&*host, DiscoveryKind::Kubernetes).is_empty());
}

/// Workers published under different pairing protocols never pair under
/// strict mode, and workers under one protocol do. Nothing else tells the
/// two rollouts here apart: without the protocol a prefill of one would pair
/// with a decode of the other.
#[test]
fn workers_of_different_pairing_protocols_never_pair_under_strict_mode() {
    let host = TestHost::without_a_queue();
    let pairing = |url: &str, worker_type: WorkerType, protocol: Option<&str>| {
        let worker = DiscoveredWorker {
            worker_type,
            kv_connector: Some("NixlConnector".to_string()),
            pairing_protocol: protocol.map(str::to_string),
            ..published(url, url)
        };
        let mut spec = build_worker_spec(&worker, DiscoveryKind::Kubernetes, &*host);
        // As registration completes the spec: the runtime its probe detects.
        spec.runtime_type = RuntimeType::Vllm;
        PdPairing::derive(&spec)
    };
    let prefill = |protocol| pairing("10.0.1.1:8080", WorkerType::Prefill, protocol);
    let decode = |protocol| pairing("10.0.2.1:8080", WorkerType::Decode, protocol);
    let strict = PdPairingMode::Strict;

    for same in ["v1", "v2"] {
        let (prefill, decode) = (prefill(Some(same)), decode(Some(same)));
        assert_eq!(prefill.mismatch(&decode, strict), None, "{same}");
    }
    for (old, new) in [("v1", "v2"), ("v2", "v1")] {
        let (prefill, decode) = (prefill(Some(old)), decode(Some(new)));
        assert_eq!(
            prefill.mismatch(&decode, strict),
            Some(PairingMismatch::Explicit),
            "a prefill of {old} and a decode of {new}"
        );
    }
    assert_eq!(prefill(None).mismatch(&decode(None), strict), None);
}

#[test]
fn build_worker_spec_carries_the_record_and_stamps_provenance() {
    let host = TestHost::without_a_queue();
    let worker = DiscoveredWorker {
        worker_type: WorkerType::Prefill,
        bootstrap_port: Some(9080),
        model_id_override: Some("llama".to_string()),
        kv_connector: Some("MooncakeConnector".to_string()),
        kv_role: Some("kv_producer".to_string()),
        kv_engine_id: Some("engine-1".to_string()),
        pairing_protocol: Some("rollout-7".to_string()),
        ..published("10.0.0.1:8081", "uid-1:8081")
    };
    let spec = build_worker_spec(&worker, DiscoveryKind::Kubernetes, &*host);

    assert_eq!(spec.url, "10.0.0.1:8081");
    assert_eq!(spec.worker_type, WorkerType::Prefill);
    assert_eq!(spec.bootstrap_port, Some(9080));
    assert_eq!(spec.kv_connector.as_deref(), Some("MooncakeConnector"));
    assert_eq!(spec.kv_role.as_deref(), Some("kv_producer"));
    assert_eq!(spec.kv_engine_id.as_deref(), Some("engine-1"));
    assert_eq!(spec.pairing_protocol.as_deref(), Some("rollout-7"));
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
    // What the host stamps: its credential and its budget of attempts.
    assert_eq!(spec.api_key.as_deref(), Some(STAMPED_API_KEY));
    assert_eq!(spec.max_connection_attempts, STAMPED_ATTEMPTS);
    // Taken from the record, so the API key and retry budget injected into
    // the spec cannot move it.
    assert_eq!(
        spec.labels.get(DISCOVERY_SPEC_HASH_LABEL),
        Some(&worker.fingerprint())
    );
}

#[test]
fn build_worker_spec_omits_a_blank_model_override() {
    let host = TestHost::without_a_queue();
    for model_id_override in ["", " \t"] {
        let worker = DiscoveredWorker {
            model_id_override: Some(model_id_override.to_string()),
            ..published("10.0.0.1:8081", "uid-1:8081")
        };
        let spec = build_worker_spec(&worker, DiscoveryKind::Kubernetes, &*host);

        assert!(
            !spec.labels.contains_key("served_model_name"),
            "override {model_id_override:?} was stamped as served_model_name"
        );
    }
}

#[test]
fn build_worker_spec_trims_a_padded_model_override() {
    let host = TestHost::without_a_queue();
    let worker = DiscoveredWorker {
        model_id_override: Some(" llama \t".to_string()),
        ..published("10.0.0.1:8081", "uid-1:8081")
    };
    let spec = build_worker_spec(&worker, DiscoveryKind::Kubernetes, &*host);

    assert_eq!(
        spec.labels.get("served_model_name").map(String::as_str),
        Some("llama")
    );
}

/// Compatibility labels are opaque here: any key a provider supplies is
/// written verbatim. The keys below are deliberately not Kubernetes' — the
/// reconciler has no notion of which provider's labels it is carrying.
#[test]
fn compat_labels_are_written_verbatim_but_cannot_claim_provenance() {
    let host = TestHost::without_a_queue();
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
    let spec = build_worker_spec(&worker, DiscoveryKind::Kubernetes, &*host);

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
    assert_eq!(series::DEREGISTRATION_RECONCILED, "reconciled");
}

#[tokio::test]
async fn test_reconcile_without_job_queue_is_safe() {
    let host = TestHost::without_a_queue();
    let desired = desired_state_of(&[published("10.0.0.1:8080", "u1")]);
    reconcile(
        &desired,
        DiscoveryKind::Kubernetes,
        &*host,
        time::Instant::now(),
    )
    .await;
}

/// A prefill worker discovered without a bootstrap annotation carries no
/// bootstrap port into its spec: its host hands it the default port on the
/// request path.
#[test]
fn a_discovered_prefill_worker_without_a_bootstrap_annotation_carries_no_port() {
    let host = TestHost::without_a_queue();
    let worker = DiscoveredWorker {
        worker_type: WorkerType::Prefill,
        ..published("127.0.0.1:18081", "uid-1:18081")
    };
    let spec = build_worker_spec(&worker, DiscoveryKind::Kubernetes, &*host);
    assert_eq!(spec.worker_type, WorkerType::Prefill);
    assert_eq!(spec.bootstrap_port, None, "no annotation, no port");
}

/// One pass of the Kubernetes provider over `desired`, against `host`.
async fn pass(desired: &DesiredState, host: &TestHost) {
    reconcile(
        desired,
        DiscoveryKind::Kubernetes,
        host,
        time::Instant::now(),
    )
    .await;
}

/// Run `scenario` to its end and return the scrape of what it recorded.
fn scrape_of(scenario: impl std::future::Future<Output = ()>) -> String {
    render_with_recorder(|| {
        Builder::new_current_thread()
            .build()
            .expect("a runtime")
            .block_on(scenario);
    })
}

/// A worker the Kubernetes provider registered at `url` and no longer
/// publishes.
fn register_an_unpublished_worker(host: &TestHost, url: &str) {
    register_with_labels(
        host,
        url,
        &[
            (DISCOVERY_PROVIDER_LABEL, "kubernetes"),
            (DISCOVERY_ID_LABEL, "gone"),
        ],
    );
}

#[tokio::test]
async fn the_reconciler_asks_as_discovery() {
    let (host, jobs) = TestHost::recording();
    register_an_unpublished_worker(&host, "http://10.0.0.2:8080");
    let desired = desired_state_of(&[published("http://10.0.0.1:8080", "u1")]);

    pass(&desired, &host).await;

    assert!(matches!(
        jobs.queued().as_slice(),
        [
            Queued::Remove {
                asker: WorkerJobAsker::Discovery,
                ..
            },
            Queued::Add {
                asker: WorkerJobAsker::Discovery,
                ..
            },
        ]
    ));
}

/// What a pass hands the queue is the spec its host stamped, the credential
/// and the budget of attempts on it, under an Upsert.
#[tokio::test]
async fn a_pass_registers_the_spec_its_host_stamped() {
    let (host, jobs) = TestHost::recording();
    let worker = published("10.0.0.1:8080", "u1");
    pass(&desired_state_of(slice::from_ref(&worker)), &host).await;

    let queued = jobs.queued();
    let [Queued::Add { spec, mode, .. }] = queued.as_slice() else {
        panic!("one addition and nothing else: {queued:?}");
    };
    assert_eq!(*mode, WorkerRegistrationMode::Upsert);
    assert_eq!(spec.url, "10.0.0.1:8080");
    assert_eq!(spec.api_key.as_deref(), Some(STAMPED_API_KEY));
    assert_eq!(spec.max_connection_attempts, STAMPED_ATTEMPTS);
    assert_eq!(
        spec.labels.get(DISCOVERY_SPEC_HASH_LABEL),
        Some(&worker.fingerprint())
    );
}

/// A pass asks nothing for an address whose job is in flight, and asks once
/// that job has ended. It reads a removal's address by its lookup form and
/// an addition's as the provider published it: the forms the two jobs are
/// queued under. Neither hold is a registration's, so the queue would take
/// the addition: that nothing is asked is the pass's own read.
#[tokio::test]
async fn a_pass_leaves_an_address_whose_job_is_in_flight() {
    let (host, jobs) = TestHost::recording();
    register_an_unpublished_worker(&host, "http://10.0.0.2:8080");
    let desired = desired_state_of(&[published("http://10.0.0.1:8080", "u1")]);
    jobs.hold_for_another_job("10.0.0.2:8080");
    jobs.hold_for_another_job("http://10.0.0.1:8080");

    pass(&desired, &host).await;
    assert!(jobs.queued().is_empty(), "{:?}", jobs.queued());
    assert!(jobs.refused().is_empty(), "a held address is not asked for");

    jobs.release("10.0.0.2:8080");
    jobs.release("http://10.0.0.1:8080");
    pass(&desired, &host).await;
    let queued = jobs.queued();
    let [Queued::Remove { url, guards, .. }, Queued::Add { spec, .. }] = queued.as_slice() else {
        panic!("the removal, then the addition: {queued:?}");
    };
    assert_eq!(url, "10.0.0.2:8080");
    assert_eq!(guards.len(), 1);
    assert_eq!(spec.url, "http://10.0.0.1:8080");
}

/// A registration that takes the address after the pass asked about it keeps
/// it: the queue refuses the pass's addition, the pass counts it as neither
/// a success nor a failure, and the next pass adds what is still missing.
#[test]
fn an_addition_the_queue_refuses_is_left_to_the_next_pass() {
    let (host, jobs) = TestHost::recording();
    let desired = desired_state_of(&[published("10.0.0.1:8080", "u1")]);
    jobs.hold_after_the_next_read("10.0.0.1:8080");

    let rendered = scrape_of(pass(&desired, &host));
    assert_eq!(jobs.refused(), ["10.0.0.1:8080"]);
    assert!(jobs.queued().is_empty(), "{:?}", jobs.queued());
    assert!(
        !rendered.contains("smg_discovery_registrations_total"),
        "{rendered}"
    );

    jobs.release("10.0.0.1:8080");
    let rendered = scrape_of(pass(&desired, &host));
    assert_eq!(jobs.queued().len(), 1, "{:?}", jobs.queued());
    assert_metric(
        &rendered,
        "smg_discovery_registrations_total",
        &[r#"source="kubernetes""#, r#"result="success""#],
        "1",
    );
}

/// A pass counts what the queue took, by source: a registration, a removal,
/// and the workers the provider published.
#[test]
fn a_pass_counts_what_the_queue_took() {
    let (host, jobs) = TestHost::recording();
    register_an_unpublished_worker(&host, "http://10.0.0.2:8080");
    let desired = desired_state_of(&[published("10.0.0.1:8080", "u1")]);

    let rendered = scrape_of(pass(&desired, &host));
    assert_eq!(jobs.queued().len(), 2, "{:?}", jobs.queued());
    let kubernetes = r#"source="kubernetes""#;
    assert_metric(
        &rendered,
        "smg_discovery_registrations_total",
        &[kubernetes, r#"result="success""#],
        "1",
    );
    assert_metric(
        &rendered,
        "smg_discovery_deregistrations_total",
        &[kubernetes, r#"reason="reconciled""#],
        "1",
    );
    assert_metric(
        &rendered,
        "smg_discovery_workers_discovered",
        &[kubernetes],
        "1",
    );
}
