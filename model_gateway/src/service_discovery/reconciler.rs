//! Level-triggered worker reconciliation, shared by every discovery provider.
//!
//! Takes the workers a provider published, diffs them against the registry
//! entries this reconciler owns, and submits the gap as `AddWorker`/
//! `RemoveWorker` jobs. Failed or missed work is retried on the next pass by
//! construction.
//!
//! Nothing here is specific to a provider. A provider hands over
//! [`DiscoveredWorker`]s and its [`DiscoveryKind`], and ownership and matching
//! run on the provenance labels this module stamps on every worker it
//! registers:
//!
//! - [`DISCOVERY_PROVIDER_LABEL`] decides which workers a provider owns.
//! - [`DISCOVERY_ID_LABEL`] decides whether an owned worker is still the
//!   instance the provider is describing.
//! - [`DISCOVERY_SPEC_HASH_LABEL`] records the configuration it was
//!   registered with.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use openai_protocol::worker::WorkerSpec;
use tokio::time;
use tracing::{error, info, warn};

use super::provider::{DiscoveredWorker, DiscoveryKind};
use crate::{
    app_context::AppContext,
    observability::metrics::{metrics_labels, Metrics},
    worker::{endpoint::Endpoint, registry::WorkerId, EndpointKey, WorkerOrigin},
    workflow::{Job, WorkerRegistrationMode},
};

/// Prefix reserved for discovery provenance. Only this module writes labels
/// under it; a provider's compatibility labels may not.
pub const DISCOVERY_LABEL_PREFIX: &str = "smg.ai/discovery-";
/// Which provider owns the worker: `kubernetes`, `file`, `slurm` or `consul`.
pub const DISCOVERY_PROVIDER_LABEL: &str = "smg.ai/discovery-provider";
/// The provider's stable identity for the worker instance.
pub const DISCOVERY_ID_LABEL: &str = "smg.ai/discovery-id";
/// BLAKE3 fingerprint of the [`DiscoveredWorker`] it was registered from.
pub const DISCOVERY_SPEC_HASH_LABEL: &str = "smg.ai/discovery-spec-hash";

/// What a provider currently publishes, by canonical endpoint.
///
/// One set. A provider publishes only the workers it considers eligible, so
/// there is no "present but not yet addable" state for the reconciler to hold
/// a worker in.
#[derive(Debug, Default)]
pub(super) struct DesiredState {
    pub(super) workers: BTreeMap<EndpointKey, DiscoveredWorker>,
}

impl DesiredState {
    /// Index the published workers by canonical endpoint.
    ///
    /// When two records claim one endpoint the first is kept, which preserves
    /// what Kubernetes did before this moved here. Rejecting such a snapshot
    /// outright is a separate, deliberate change.
    pub(super) fn from_workers(workers: impl IntoIterator<Item = DiscoveredWorker>) -> Self {
        let mut state = Self::default();
        for worker in workers {
            state.workers.entry(worker.endpoint.key()).or_insert(worker);
        }
        state
    }
}

/// A registry worker this provider owns.
#[derive(Debug, Clone)]
pub(super) struct OwnedWorker {
    /// The registry's id for this one registration. A DP group has several
    /// under one endpoint, and the removal guard pins each to its own revision.
    pub(super) worker_id: WorkerId,
    /// The provider's id for the instance, read back from
    /// [`DISCOVERY_ID_LABEL`]. Every rank of a DP group carries the same one.
    ///
    /// `None` when the label is missing. That is kept distinct from any string
    /// rather than defaulted to `""`, because a provider is free to publish an
    /// empty id and the two would then compare equal — leaving an unlabelled
    /// worker matched forever instead of replaced.
    pub(super) discovery_id: Option<String>,
    /// The registered address, parsed. Its [`Endpoint::key`] is the identity
    /// used for grouping; the endpoint itself is kept so a removal can submit
    /// a form that parses back (an IPC key is a bare socket path).
    pub(super) endpoint: Endpoint,
    /// Revision guard for removal: a concurrently replaced worker is skipped
    /// and re-evaluated on the next pass instead of removed blindly.
    pub(super) revision: u64,
}

/// One canonical endpoint to remove, carrying every registry worker that
/// shares it.
///
/// DP-rank expansions collapse to one canonical endpoint but hold independent
/// revisions, so a single scalar cannot guard the group: it would silently drop
/// the ranks whose revision differs, leaving them registered against an
/// instance that is already gone.
#[derive(Debug, Clone)]
pub(super) struct RemovalTarget {
    /// One member's parsed address; every member shares its key.
    pub(super) endpoint: Endpoint,
    /// The provider id of whichever member the registry happened to yield
    /// first. Ranks of one DP group do share it, but a stale-scheme sibling
    /// may not: `grpc://h:p` and `http://h:p` canonicalize alike, so two
    /// registrations of different instances can land in one target. Logged
    /// only, never matched on — the removal is decided by [`Self::guards`].
    pub(super) discovery_id: Option<String>,
    /// `(worker_id, revision)` as observed in this snapshot, one per rank.
    pub(super) guards: Vec<(WorkerId, u64)>,
}

/// Snapshot the registry workers this provider owns: locally registered (never
/// mesh-imported) and stamped with this provider's label. Manually added and
/// statically configured workers carry no provider label and are never
/// touched, and neither are workers another provider owns.
fn owned_workers(app_context: &AppContext, kind: DiscoveryKind) -> Vec<OwnedWorker> {
    app_context
        .worker_registry
        .get_all_with_ids()
        .into_iter()
        .filter_map(|(worker_id, worker)| {
            if app_context.worker_registry.origin_of(&worker_id) != Some(WorkerOrigin::Local) {
                return None;
            }
            let labels = &worker.metadata().spec.labels;
            if labels.get(DISCOVERY_PROVIDER_LABEL).map(String::as_str) != Some(kind.as_label()) {
                return None;
            }
            // A missing id stays `None`, which never equals a published id, so
            // an owned worker without one is replaced on this pass and comes
            // back labelled properly.
            let discovery_id = labels.get(DISCOVERY_ID_LABEL).cloned();
            // A registered address the shared parser rejects is one this
            // reconciler cannot safely match against a published endpoint, so
            // it is left alone rather than guessed at.
            let endpoint = match Endpoint::parse_with_rank(worker.url()) {
                Ok((endpoint, _)) => endpoint,
                Err(e) => {
                    warn!(
                        worker_url = %worker.url(),
                        error = %e,
                        "Skipping discovery-owned worker with an unparsable address"
                    );
                    return None;
                }
            };
            Some(OwnedWorker {
                worker_id,
                discovery_id,
                endpoint,
                revision: worker.revision(),
            })
        })
        .collect()
}

#[derive(Debug, Default)]
pub(super) struct ReconcileActions {
    /// Workers to register: endpoints this provider does not own yet, plus
    /// endpoints whose owned worker is a different instance than the one
    /// published.
    pub(super) add: Vec<DiscoveredWorker>,
    /// Workers to remove: endpoint no longer published, or held by a different
    /// instance than the one published (which also covers a stale-scheme
    /// sibling the same-URL Upsert cannot replace). One entry per canonical
    /// endpoint, carrying every rank that shares it.
    pub(super) remove: Vec<RemovalTarget>,
}

/// Diff what a provider publishes against what it owns, by canonical endpoint.
///
/// An endpoint is left alone when the owned worker there is the instance the
/// provider describes (same `discovery_id`). It is removed when no longer
/// published or held by a different instance, and added when the provider
/// owns nothing there or owns a different instance. A DP group yields one
/// removal per endpoint, carrying a guard for every rank.
pub(super) fn compute_actions(
    desired: &DesiredState,
    registered: &[OwnedWorker],
) -> ReconcileActions {
    let mut actions = ReconcileActions::default();

    let mut registered_id: HashMap<EndpointKey, Option<&str>> = HashMap::new();
    // DP-rank expansions share one canonical endpoint: remove it once, but keep
    // every rank's own `(worker_id, revision)` so the guard cannot drop the
    // ranks whose revision happens to differ from an arbitrarily chosen one.
    let mut remove_by_key: HashMap<EndpointKey, RemovalTarget> = HashMap::new();
    for worker in registered {
        let key = worker.endpoint.key();
        registered_id.insert(key.clone(), worker.discovery_id.as_deref());
        match desired.workers.get(&key) {
            Some(published)
                if worker.discovery_id.as_deref() == Some(published.discovery_id.as_str()) => {}
            _ => remove_by_key
                .entry(key)
                .or_insert_with(|| RemovalTarget {
                    endpoint: worker.endpoint.clone(),
                    discovery_id: worker.discovery_id.clone(),
                    guards: Vec::new(),
                })
                .guards
                .push((worker.worker_id.clone(), worker.revision)),
        }
    }
    actions.remove = remove_by_key.into_values().collect();
    // `HashMap` iteration order is unspecified; sort so a pass submits jobs
    // and logs them in a stable order.
    actions
        .remove
        .sort_unstable_by_key(|target| target.endpoint.key());
    for target in &mut actions.remove {
        target
            .guards
            .sort_unstable_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
    }

    for (key, worker) in &desired.workers {
        match registered_id.get(key) {
            Some(Some(id)) if *id == worker.discovery_id => {}
            _ => actions.add.push(worker.clone()),
        }
    }
    actions
}

/// Turn a published worker into the spec its registration job will carry.
///
/// Compatibility labels go in first, then provenance — so a provider cannot
/// overwrite ownership — then the router-controlled settings. The fingerprint
/// is taken from the record, never from this spec, so none of what is added
/// here can move it.
fn build_worker_spec(
    worker: &DiscoveredWorker,
    kind: DiscoveryKind,
    app_context: &AppContext,
) -> WorkerSpec {
    let mut spec = WorkerSpec::new(worker.endpoint.render());
    spec.worker_type = worker.worker_type;
    spec.bootstrap_port = worker.bootstrap_port;
    // A provider's compatibility labels go in first and verbatim, so the
    // provenance written after them always wins. Nothing under the reserved
    // prefix is taken from a provider: ownership comes from this module alone.
    for (key, value) in &worker.compat_labels {
        if key.starts_with(DISCOVERY_LABEL_PREFIX) {
            warn!(
                label = %key,
                "Ignoring a provider label under the reserved discovery prefix"
            );
            continue;
        }
        spec.labels.insert(key.clone(), value.clone());
    }
    spec.labels.insert(
        DISCOVERY_PROVIDER_LABEL.to_string(),
        kind.as_label().to_string(),
    );
    spec.labels
        .insert(DISCOVERY_ID_LABEL.to_string(), worker.discovery_id.clone());
    // Taken from the record, never from this spec, so nothing added here — the
    // API key, the retry budget, compatibility labels — can move it.
    spec.labels
        .insert(DISCOVERY_SPEC_HASH_LABEL.to_string(), worker.fingerprint());
    // served_model_name is priority #2 in create_worker's model_id chain.
    if let Some(ref model_id) = worker.model_id_override {
        spec.labels
            .insert("served_model_name".to_string(), model_id.clone());
    }
    spec.kv_connector.clone_from(&worker.kv_connector);
    spec.kv_role.clone_from(&worker.kv_role);
    spec.kv_engine_id.clone_from(&worker.kv_engine_id);
    spec.api_key.clone_from(&app_context.router_config.api_key);
    spec.max_connection_attempts = app_context
        .router_config
        .health_check
        .success_threshold
        .max(1)
        * 20;
    spec
}

/// One reconcile pass: diff the workers a provider published against the
/// registry entries it owns and submit Add/Remove jobs for the gap. Failed or
/// missed work is retried on the next pass by construction.
///
/// `started_at` is taken by the caller so the sync-duration metric still
/// covers the provider's own snapshot conversion.
pub(super) async fn reconcile(
    desired: &DesiredState,
    kind: DiscoveryKind,
    app_context: &Arc<AppContext>,
    started_at: time::Instant,
) {
    let registered = owned_workers(app_context, kind);
    let actions = compute_actions(desired, &registered);

    let desired_count = desired.workers.len();
    if actions.add.is_empty() && actions.remove.is_empty() {
        Metrics::set_discovery_workers_discovered(kind.metric_label(), desired_count);
        return;
    }

    let Some(job_queue) = app_context.worker_job_queue.get() else {
        warn!(
            "JobQueue not initialized; deferring {} addition(s), {} removal(s)",
            actions.add.len(),
            actions.remove.len()
        );
        return;
    };

    // One in-flight job per URL: a pending/processing job owns that worker's
    // transition (a duplicate removal would find it already Draining, skip
    // the settle sleep, and collapse the drain window). Completed/failed
    // statuses do not block, so failures retry on the next pass.
    let in_flight = |url: &str| {
        job_queue
            .get_status(url)
            .is_some_and(|status| status.status == "pending" || status.status == "processing")
    };
    let removals: Vec<&RemovalTarget> = actions
        .remove
        .iter()
        .filter(|target| !in_flight(&target.endpoint.lookup_form()))
        .collect();
    let additions: Vec<&DiscoveredWorker> = actions
        .add
        .iter()
        .filter(|worker| !in_flight(&worker.endpoint.render()))
        .collect();

    if removals.is_empty() && additions.is_empty() {
        Metrics::set_discovery_workers_discovered(kind.metric_label(), desired_count);
        return;
    }

    info!(
        "Reconciling workers: {} to add, {} to remove ({} desired)",
        additions.len(),
        removals.len(),
        desired_count
    );

    for target in removals {
        info!(
            "Removing worker {} ({} registration(s), {}): no longer published, or a \
             different instance now holds the address",
            target.endpoint.redacted(),
            target.guards.len(),
            target.discovery_id.as_deref().unwrap_or("no discovery id")
        );
        // Scheme-less for a network address, so `find_workers_by_url` reaches
        // every spelling the group was registered under. An IPC endpoint keeps
        // its scheme, because its key is a bare socket path that would not
        // parse back.
        let job = Job::RemoveWorker {
            url: target.endpoint.lookup_form(),
            expected_revisions: Some(
                target
                    .guards
                    .iter()
                    .map(|(id, revision)| (id.as_str().to_string(), *revision))
                    .collect(),
            ),
        };
        match job_queue.submit(job).await {
            Ok(()) => Metrics::record_discovery_deregistration(
                kind.metric_label(),
                metrics_labels::DEREGISTRATION_RECONCILED,
            ),
            Err(e) => error!(
                "Failed to submit worker removal for {}: {}",
                target.endpoint.redacted(),
                e
            ),
        }
    }

    for worker in additions {
        info!(
            "Registering worker {} ({:?}) as {}",
            worker.endpoint, worker.worker_type, worker.discovery_id
        );
        let job = Job::AddWorker {
            config: Box::new(build_worker_spec(worker, kind, app_context)),
            registration_mode: WorkerRegistrationMode::Upsert,
        };
        match job_queue.submit(job).await {
            Ok(()) => Metrics::record_discovery_registration(
                kind.metric_label(),
                metrics_labels::REGISTRATION_SUCCESS,
            ),
            Err(e) => {
                error!(
                    "Failed to submit worker addition for {}: {}",
                    worker.endpoint, e
                );
                Metrics::record_discovery_registration(
                    kind.metric_label(),
                    metrics_labels::REGISTRATION_FAILED,
                );
            }
        }
    }

    Metrics::set_discovery_workers_discovered(kind.metric_label(), desired_count);
    Metrics::record_discovery_sync_duration(kind.metric_label(), started_at.elapsed());
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::WorkerType;

    use super::*;
    use crate::service_discovery::testing::create_test_app_context;

    fn key(url: &str) -> EndpointKey {
        crate::worker::endpoint_key(url).expect(url)
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
    /// and revision.
    fn owned_rank(url: &str, discovery_id: &str, worker_id: &str, revision: u64) -> OwnedWorker {
        OwnedWorker {
            worker_id: WorkerId::from_string(worker_id.to_string()),
            discovery_id: Some(discovery_id.to_string()),
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
    fn test_compute_actions_same_instance_metadata_change_is_noop() {
        let mut worker = published("10.0.0.1:8080", "u1");
        worker.kv_connector = Some("NixlConnector".to_string());
        let desired = desired_state_of(&[worker]);
        let registered = [owned("10.0.0.1:8080", "u1")];
        let actions = compute_actions(&desired, &registered);
        assert!(actions.add.is_empty());
        assert!(actions.remove.is_empty());
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
        let state = smg_mesh::WorkerState {
            worker_id: "peer-w1".to_string(),
            model_id: "m".to_string(),
            url: "http://10.0.0.3:8080".to_string(),
            health: true,
            load: 0.0,
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
}
