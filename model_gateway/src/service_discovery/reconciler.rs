//! Level-triggered worker reconciliation.
//!
//! Takes a desired-worker snapshot, diffs it against the registry entries this
//! reconciler owns, and submits the gap as `AddWorker`/`RemoveWorker` jobs.
//! Failed or missed work is retried on the next pass by construction.
//!
//! The module boundary is provider-neutral — it never sees a Kubernetes `Pod`
//! — but the ownership model is still Kubernetes-specific: identity is a Pod
//! UID carried in [`POD_UID_LABEL`]. Generalizing that to a provider-agnostic
//! `(provider, id, spec-hash)` triple is deliberately left to a follow-up so
//! this extraction stays behavior-preserving.

use std::{collections::HashMap, sync::Arc};

use openai_protocol::worker::{WorkerSpec, WorkerType};
use tokio::time;
use tracing::{error, info, warn};

use crate::{
    app_context::AppContext,
    observability::metrics::{metrics_labels, Metrics},
    worker::{registry::WorkerId, WorkerOrigin},
    workflow::{Job, WorkerRegistrationMode},
};

/// Labels stamped on every discovery-created worker; workers carrying
/// [`POD_UID_LABEL`] are owned (added/removed) by the K8s reconciler.
pub const POD_NAME_LABEL: &str = "smg.ai/pod-name";
pub const POD_UID_LABEL: &str = "smg.ai/pod-uid";

/// One worker the reconciler wants registered: a single engine server
/// (pod IP + data port) plus the metadata needed to build its spec.
#[derive(Debug, Clone)]
pub(super) struct DesiredWorker {
    /// Bare host:port so DetectConnectionModeStep dual-probes HTTP and gRPC.
    pub(super) url: String,
    pub(super) worker_type: WorkerType,
    pub(super) bootstrap_port: Option<u16>,
    pub(super) pod_name: String,
    pub(super) pod_uid: String,
    pub(super) model_id_override: Option<String>,
    pub(super) kv_connector: Option<String>,
    pub(super) kv_engine_id: Option<String>,
}

/// Desired view of the cluster derived from the store snapshot.
#[derive(Debug, Default)]
pub(super) struct DesiredState {
    /// Owning pod uid per worker URL for Ready, non-terminating Pods.
    /// Registered workers whose URL is absent — or owned by a different Pod
    /// uid — enter the existing drain/remove workflow.
    pub(super) uid_by_url: HashMap<String, String>,
    /// Workers on Running, Ready Pods — registration candidates.
    pub(super) addable: Vec<DesiredWorker>,
}

/// A registry worker owned by K8s discovery (stamped with [`POD_UID_LABEL`]).
#[derive(Debug, Clone)]
pub(super) struct OwnedWorker {
    /// The registry id. A DP group shares one canonical [`Self::url`], so the
    /// id is what distinguishes its ranks — and what the removal guard needs
    /// to pin each rank to its own revision.
    pub(super) id: WorkerId,
    /// Scheme- and DP-rank-stripped `host:port`.
    pub(super) url: String,
    pub(super) pod_uid: String,
    /// Revision guard for removal: a concurrently replaced worker is skipped
    /// and re-evaluated on the next pass instead of removed blindly.
    pub(super) revision: u64,
}

/// One canonical URL to remove, carrying every registry worker that shares it.
///
/// DP-rank expansions collapse to one canonical URL but hold independent
/// revisions, so a single scalar cannot guard the group: it would silently
/// drop the ranks whose revision differs, leaving them registered against a
/// Pod that is already gone.
#[derive(Debug, Clone)]
pub(super) struct RemovalTarget {
    pub(super) url: String,
    /// Pod uid of whichever member the registry happened to yield first.
    /// Ranks of one DP group do share it, but a stale-scheme sibling can not:
    /// `grpc://h:p` and `http://h:p` canonicalize alike, so two registrations
    /// from different Pods can land in one target. Logged only, never matched
    /// on — the removal is decided by [`Self::guards`].
    pub(super) pod_uid: String,
    /// `(id, revision)` as observed in this snapshot, one entry per rank.
    pub(super) guards: Vec<(WorkerId, u64)>,
}

/// `http://10.0.0.1:8080@2` → `10.0.0.1:8080`.
fn canonical_host_port(url: &str) -> &str {
    let stripped = ["http://", "https://", "grpc://", "grpcs://", "ipc://"]
        .iter()
        .find_map(|scheme| url.strip_prefix(scheme))
        .unwrap_or(url);
    stripped.split('@').next().unwrap_or(stripped)
}

/// Snapshot the registry workers this reconciler owns: locally registered
/// (never mesh-imported) and stamped with the pod-uid label. Manually added
/// workers lack the label and are never touched.
fn k8s_owned_workers(app_context: &AppContext) -> Vec<OwnedWorker> {
    app_context
        .worker_registry
        .get_all_with_ids()
        .into_iter()
        .filter_map(|(id, worker)| {
            if app_context.worker_registry.origin_of(&id) != Some(WorkerOrigin::Local) {
                return None;
            }
            let pod_uid = worker.metadata().spec.labels.get(POD_UID_LABEL)?.clone();
            Some(OwnedWorker {
                id,
                url: canonical_host_port(worker.url()).to_string(),
                pod_uid,
                revision: worker.revision(),
            })
        })
        .collect()
}

#[derive(Debug, Default)]
pub(super) struct ReconcileActions {
    /// Workers to register: new URLs, plus same-URL pods whose uid changed
    /// (restart with a stable IP).
    pub(super) add: Vec<DesiredWorker>,
    /// Workers to remove: URL gone from the desired set, or owned by a pod
    /// uid that no longer holds the URL (covers a stale-scheme sibling the
    /// same-URL Upsert cannot replace). One entry per canonical URL, carrying
    /// every rank that shares it.
    pub(super) remove: Vec<RemovalTarget>,
}

pub(super) fn compute_actions(
    desired: &DesiredState,
    registered: &[OwnedWorker],
) -> ReconcileActions {
    let mut actions = ReconcileActions::default();

    let mut registered_uid: HashMap<&str, &str> = HashMap::new();
    // DP-rank expansions share one canonical URL: remove it once, but keep
    // every rank's own `(id, revision)` so the guard cannot drop the ranks
    // whose revision happens to differ from an arbitrarily chosen one.
    let mut remove_by_url: HashMap<&str, RemovalTarget> = HashMap::new();
    for worker in registered {
        registered_uid.insert(worker.url.as_str(), worker.pod_uid.as_str());
        match desired.uid_by_url.get(worker.url.as_str()) {
            Some(uid) if *uid == worker.pod_uid => {}
            _ => remove_by_url
                .entry(worker.url.as_str())
                .or_insert_with(|| RemovalTarget {
                    url: worker.url.clone(),
                    pod_uid: worker.pod_uid.clone(),
                    guards: Vec::new(),
                })
                .guards
                .push((worker.id.clone(), worker.revision)),
        }
    }
    actions.remove = remove_by_url.into_values().collect();
    // `HashMap` iteration order is unspecified; sort so a pass submits jobs
    // and logs them in a stable order.
    actions.remove.sort_unstable_by(|a, b| a.url.cmp(&b.url));
    for target in &mut actions.remove {
        target
            .guards
            .sort_unstable_by(|a, b| a.0.as_str().cmp(b.0.as_str()));
    }

    for worker in &desired.addable {
        match registered_uid.get(worker.url.as_str()) {
            Some(uid) if *uid == worker.pod_uid => {}
            _ => actions.add.push(worker.clone()),
        }
    }
    actions
}

fn build_worker_spec(desired: &DesiredWorker, app_context: &AppContext) -> WorkerSpec {
    let mut spec = WorkerSpec::new(desired.url.clone());
    spec.worker_type = desired.worker_type;
    spec.bootstrap_port = desired.bootstrap_port;
    spec.labels
        .insert(POD_NAME_LABEL.to_string(), desired.pod_name.clone());
    spec.labels
        .insert(POD_UID_LABEL.to_string(), desired.pod_uid.clone());
    // served_model_name is priority #2 in create_worker's model_id chain.
    if let Some(ref model_id) = desired.model_id_override {
        spec.labels
            .insert("served_model_name".to_string(), model_id.clone());
    }
    spec.kv_connector.clone_from(&desired.kv_connector);
    spec.kv_engine_id.clone_from(&desired.kv_engine_id);
    spec.api_key.clone_from(&app_context.router_config.api_key);
    spec.max_connection_attempts = app_context
        .router_config
        .health_check
        .success_threshold
        .max(1)
        * 20;
    spec
}

/// One reconcile pass: diff the desired workers a provider published against
/// the registry entries this reconciler owns and submit Add/Remove jobs for
/// the gap. Failed or missed work is retried on the next pass by construction.
///
/// `started_at` is taken by the caller so the sync-duration metric still
/// covers the provider's own snapshot conversion, as it did when this
/// function read the reflector store itself.
pub(super) async fn reconcile(
    desired: &DesiredState,
    app_context: &Arc<AppContext>,
    started_at: time::Instant,
) {
    let registered = k8s_owned_workers(app_context);
    let actions = compute_actions(desired, &registered);

    let desired_count = desired.uid_by_url.len();
    if actions.add.is_empty() && actions.remove.is_empty() {
        Metrics::set_discovery_workers_discovered(
            metrics_labels::DISCOVERY_KUBERNETES,
            desired_count,
        );
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
        .filter(|worker| !in_flight(&worker.url))
        .collect();
    let additions: Vec<&DesiredWorker> = actions
        .add
        .iter()
        .filter(|worker| !in_flight(&worker.url))
        .collect();

    if removals.is_empty() && additions.is_empty() {
        Metrics::set_discovery_workers_discovered(
            metrics_labels::DISCOVERY_KUBERNETES,
            desired_count,
        );
        return;
    }

    info!(
        "Reconciling workers: {} to add, {} to remove ({} desired)",
        additions.len(),
        removals.len(),
        desired_count
    );

    for worker in removals {
        info!(
            "Removing worker {} ({} registration(s), pod {}): pod unready, gone, \
             terminating, or replaced",
            worker.url,
            worker.guards.len(),
            worker.pod_uid
        );
        let job = Job::RemoveWorker {
            url: worker.url.clone(),
            expected_revisions: Some(
                worker
                    .guards
                    .iter()
                    .map(|(id, revision)| (id.as_str().to_string(), *revision))
                    .collect(),
            ),
        };
        match job_queue.submit(job).await {
            Ok(()) => Metrics::record_discovery_deregistration(
                metrics_labels::DISCOVERY_KUBERNETES,
                metrics_labels::DEREGISTRATION_RECONCILED,
            ),
            Err(e) => error!("Failed to submit worker removal for {}: {}", worker.url, e),
        }
    }

    for worker in additions {
        info!(
            "Registering worker {} ({:?}) for pod {}",
            worker.url, worker.worker_type, worker.pod_name
        );
        let job = Job::AddWorker {
            config: Box::new(build_worker_spec(worker, app_context)),
            registration_mode: WorkerRegistrationMode::Upsert,
        };
        match job_queue.submit(job).await {
            Ok(()) => Metrics::record_discovery_registration(
                metrics_labels::DISCOVERY_KUBERNETES,
                metrics_labels::REGISTRATION_SUCCESS,
            ),
            Err(e) => {
                error!("Failed to submit worker addition for {}: {}", worker.url, e);
                Metrics::record_discovery_registration(
                    metrics_labels::DISCOVERY_KUBERNETES,
                    metrics_labels::REGISTRATION_FAILED,
                );
            }
        }
    }

    Metrics::set_discovery_workers_discovered(metrics_labels::DISCOVERY_KUBERNETES, desired_count);
    Metrics::record_discovery_sync_duration(
        metrics_labels::DISCOVERY_KUBERNETES,
        started_at.elapsed(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service_discovery::testing::create_test_app_context;

    #[test]
    fn test_canonical_host_port() {
        assert_eq!(canonical_host_port("10.0.0.1:8080"), "10.0.0.1:8080");
        assert_eq!(canonical_host_port("http://10.0.0.1:8080"), "10.0.0.1:8080");
        assert_eq!(
            canonical_host_port("grpc://10.0.0.1:8080@2"),
            "10.0.0.1:8080"
        );
        assert_eq!(canonical_host_port("10.0.0.1:8080@0"), "10.0.0.1:8080");
    }

    fn desired_worker(url: &str, uid: &str) -> DesiredWorker {
        DesiredWorker {
            url: url.to_string(),
            worker_type: WorkerType::Regular,
            bootstrap_port: None,
            pod_name: "w".to_string(),
            pod_uid: uid.to_string(),
            model_id_override: None,
            kv_connector: None,
            kv_engine_id: None,
        }
    }

    fn desired_state_of(workers: &[DesiredWorker]) -> DesiredState {
        let mut state = DesiredState::default();
        for worker in workers {
            state
                .uid_by_url
                .insert(worker.url.clone(), worker.pod_uid.clone());
            state.addable.push(worker.clone());
        }
        state
    }

    fn owned(url: &str, uid: &str) -> OwnedWorker {
        owned_rank(url, uid, url, 1)
    }

    /// One rank of a DP group: same canonical `url`, distinct id and revision.
    fn owned_rank(url: &str, uid: &str, id: &str, revision: u64) -> OwnedWorker {
        OwnedWorker {
            id: WorkerId::from_string(id.to_string()),
            url: url.to_string(),
            pod_uid: uid.to_string(),
            revision,
        }
    }

    #[test]
    fn test_compute_actions_adds_missing_workers() {
        let desired = desired_state_of(&[
            desired_worker("10.0.0.1:8080", "u1"),
            desired_worker("10.0.0.1:8081", "u1"),
        ]);
        let actions = compute_actions(&desired, &[]);
        assert_eq!(actions.add.len(), 2);
        assert!(actions.remove.is_empty());
    }

    #[test]
    fn test_compute_actions_same_uid_metadata_change_is_noop() {
        let mut worker = desired_worker("10.0.0.1:8080", "u1");
        worker.kv_connector = Some("NixlConnector".to_string());
        let desired = desired_state_of(&[worker]);
        let registered = [owned("10.0.0.1:8080", "u1")];
        let actions = compute_actions(&desired, &registered);
        assert!(actions.add.is_empty());
        assert!(actions.remove.is_empty());
    }

    #[test]
    fn test_compute_actions_removes_workers_for_gone_pods() {
        let desired = desired_state_of(&[desired_worker("10.0.0.1:8080", "u1")]);
        let registered = [owned("10.0.0.1:8080", "u1"), owned("10.0.0.2:8080", "u2")];
        let actions = compute_actions(&desired, &registered);
        assert!(actions.add.is_empty());
        assert_eq!(actions.remove.len(), 1);
        assert_eq!(actions.remove[0].url, "10.0.0.2:8080");
    }

    #[test]
    fn test_compute_actions_uid_change_removes_and_reregisters_same_url() {
        // Same-IP pod restart (hostNetwork / stable IP): URL unchanged but
        // uid differs → the stale worker is removed (covers a scheme-flipped
        // sibling the Upsert cannot replace) and the new one registered.
        let desired = desired_state_of(&[desired_worker("10.0.0.1:8080", "uid-new")]);
        let registered = [owned("10.0.0.1:8080", "uid-old")];
        let actions = compute_actions(&desired, &registered);
        assert_eq!(actions.add.len(), 1);
        assert_eq!(actions.add[0].pod_uid, "uid-new");
        assert_eq!(actions.remove.len(), 1);
        assert_eq!(actions.remove[0].pod_uid, "uid-old");
    }

    #[test]
    fn test_compute_actions_dp_ranks_removed_once() {
        let registered = [
            owned_rank("10.0.0.1:8080", "u1", "w@0", 1),
            owned_rank("10.0.0.1:8080", "u1", "w@1", 1),
        ];
        let actions = compute_actions(&DesiredState::default(), &registered);
        assert_eq!(actions.remove.len(), 1);
        assert_eq!(actions.remove[0].url, "10.0.0.1:8080");
        assert_eq!(actions.remove[0].guards.len(), 2);
        assert!(actions.add.is_empty());
    }

    /// The group collapses to one removal job, but every rank must keep its
    /// own revision. Carrying a single revision retained only the ranks that
    /// happened to share it and left the others registered against a Pod that
    /// was already gone.
    #[test]
    fn dp_ranks_keep_their_own_revision_when_diverged() {
        let registered = [
            owned_rank("10.0.0.1:8080", "u1", "w@0", 7),
            owned_rank("10.0.0.1:8080", "u1", "w@1", 2),
        ];
        let actions = compute_actions(&DesiredState::default(), &registered);
        assert_eq!(actions.remove.len(), 1, "one job per canonical URL");
        let guards: Vec<(&str, u64)> = actions.remove[0]
            .guards
            .iter()
            .map(|(id, revision)| (id.as_str(), *revision))
            .collect();
        assert_eq!(guards, vec![("w@0", 7), ("w@1", 2)]);
    }

    /// Two pods behind one canonical URL keep separate guards per rank, so a
    /// shared revision value cannot make one pod's rank stand in for another's.
    #[test]
    fn diverged_ranks_do_not_collapse_on_equal_revisions() {
        let registered = [
            owned_rank("10.0.0.1:8080", "u1", "w@0", 3),
            owned_rank("10.0.0.1:8080", "u1", "w@1", 3),
            owned_rank("10.0.0.1:8081", "u1", "x@0", 3),
        ];
        let actions = compute_actions(&DesiredState::default(), &registered);
        assert_eq!(actions.remove.len(), 2, "one job per canonical URL");
        assert_eq!(actions.remove[0].guards.len(), 2);
        assert_eq!(actions.remove[1].guards.len(), 1);
    }

    #[test]
    fn test_k8s_owned_workers_scoped_by_label_and_origin() {
        use openai_protocol::model_card::ModelCard;

        use crate::worker::BasicWorkerBuilder;

        let app_context = create_test_app_context();

        let mut labels = HashMap::new();
        labels.insert(POD_UID_LABEL.to_string(), "uid-1".to_string());
        let discovered = Arc::new(
            BasicWorkerBuilder::new("http://10.0.0.1:8080")
                .model(ModelCard::new("m"))
                .labels(labels)
                .build(),
        );
        let manual = Arc::new(
            BasicWorkerBuilder::new("http://10.0.0.2:8080")
                .model(ModelCard::new("m"))
                .build(),
        );
        app_context.worker_registry.register(discovered).unwrap();
        app_context.worker_registry.register(manual).unwrap();

        let owned = k8s_owned_workers(&app_context);
        assert_eq!(owned.len(), 1);
        assert_eq!(owned[0].url, "10.0.0.1:8080");
        assert_eq!(owned[0].pod_uid, "uid-1");
    }

    #[test]
    fn test_k8s_owned_workers_excludes_mesh_imported_workers() {
        // A peer's discovered worker arrives via mesh sync carrying the
        // pod-uid label; its pod is absent from this node's store, so without
        // the Local-origin filter the reconciler would remove it every pass.
        let app_context = create_test_app_context();

        let mut spec = WorkerSpec::new("http://10.0.0.3:8080");
        spec.labels
            .insert(POD_UID_LABEL.to_string(), "uid-mesh".to_string());
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

        assert!(k8s_owned_workers(&app_context).is_empty());
    }

    #[test]
    fn test_build_worker_spec_stamps_ownership_labels() {
        let app_context = create_test_app_context();
        let desired = DesiredWorker {
            url: "10.0.0.1:8081".to_string(),
            worker_type: WorkerType::Prefill,
            bootstrap_port: Some(9080),
            pod_name: "prefill-0".to_string(),
            pod_uid: "uid-1".to_string(),
            model_id_override: Some("llama".to_string()),
            kv_connector: Some("MooncakeConnector".to_string()),
            kv_engine_id: Some("engine-1".to_string()),
        };
        let spec = build_worker_spec(&desired, &app_context);
        assert_eq!(spec.url, "10.0.0.1:8081");
        assert_eq!(spec.worker_type, WorkerType::Prefill);
        assert_eq!(spec.bootstrap_port, Some(9080));
        assert_eq!(spec.kv_connector.as_deref(), Some("MooncakeConnector"));
        assert_eq!(spec.kv_engine_id.as_deref(), Some("engine-1"));
        assert_eq!(
            spec.labels.get(POD_NAME_LABEL),
            Some(&"prefill-0".to_string())
        );
        assert_eq!(spec.labels.get(POD_UID_LABEL), Some(&"uid-1".to_string()));
        assert_eq!(
            spec.labels.get("served_model_name"),
            Some(&"llama".to_string())
        );
    }

    #[test]
    fn test_deregistration_reconciled_metric_label() {
        // Verify the metric label constant exists and has expected value
        assert_eq!(metrics_labels::DEREGISTRATION_RECONCILED, "reconciled");
    }

    #[tokio::test]
    async fn test_reconcile_without_job_queue_is_safe() {
        let app_context = create_test_app_context();
        let desired = desired_state_of(&[desired_worker("10.0.0.1:8080", "u1")]);
        reconcile(&desired, &app_context, time::Instant::now()).await;
    }
}
