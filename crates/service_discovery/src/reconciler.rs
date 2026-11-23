//! Level-triggered worker reconciliation, shared by every discovery provider.
//!
//! Takes the workers a provider published, diffs them against the registry
//! entries this reconciler owns, and hands the gap to the host's job queue as
//! registrations and removals. Failed or missed work is retried on the next
//! pass by construction.
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
//!   registered with. A worker whose configuration no longer matches what is
//!   published is updated in place.

use std::collections::{BTreeMap, HashMap, HashSet};

use openai_protocol::worker::WorkerSpec;
use smg_worker::{
    identity::endpoint::{Endpoint, EndpointKey},
    jobs::{WorkerJobAsker, WorkerRegistrationMode},
    registry::{WorkerId, WorkerOrigin},
};
use tokio::time;
use tracing::{error, info, warn};

use super::{
    host::DiscoveryHost,
    provider::{DiscoveredWorker, DiscoveryKind},
    series,
};

/// Prefix reserved for discovery provenance. Only this module writes labels
/// under it; a provider's compatibility labels may not, and neither may the
/// worker API (`WorkerService` refuses them).
pub const DISCOVERY_LABEL_PREFIX: &str = "smg.ai/discovery-";
/// Which provider owns the worker: `kubernetes`, `file`, `slurm` or `consul`.
/// It lives in the worker crate, whose health checker reads it.
pub use smg_worker::labels::DISCOVERY_PROVIDER_LABEL;
/// The provider's stable identity for the worker instance.
pub const DISCOVERY_ID_LABEL: &str = "smg.ai/discovery-id";
/// BLAKE3 fingerprint of the `DiscoveredWorker` it was registered from. A
/// change to it updates the worker in place.
pub const DISCOVERY_SPEC_HASH_LABEL: &str = "smg.ai/discovery-spec-hash";

/// What a provider currently publishes, by canonical endpoint.
///
/// One set. A provider publishes only the workers it considers eligible, so
/// there is no "present but not yet addable" state for the reconciler to hold
/// a worker in.
#[derive(Debug, Default, PartialEq)]
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
    /// The fingerprint of the configuration it was registered with, read back
    /// from [`DISCOVERY_SPEC_HASH_LABEL`]. `None` when the label is missing,
    /// which matches nothing, so such a worker is updated and relabelled.
    pub(super) spec_hash: Option<String>,
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
fn owned_workers(host: &dyn DiscoveryHost, kind: DiscoveryKind) -> Vec<OwnedWorker> {
    host.workers()
        .into_iter()
        .filter_map(|(worker_id, worker)| {
            if host.origin_of(&worker_id) != Some(WorkerOrigin::Local) {
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
            let spec_hash = labels.get(DISCOVERY_SPEC_HASH_LABEL).cloned();
            // A registered address the shared parser rejects is one this
            // reconciler cannot safely match against a published endpoint, so
            // it is left alone rather than guessed at.
            let endpoint = match Endpoint::parse_with_rank(worker.url()) {
                Ok((endpoint, _)) => endpoint,
                Err(e) => {
                    // The address the parser refused reads as its marker, so the
                    // id says which worker was skipped.
                    warn!(
                        worker_id = %worker_id.as_str(),
                        worker_url = %worker.redacted_url(),
                        error = %e,
                        "Skipping discovery-owned worker with an unparsable address"
                    );
                    return None;
                }
            };
            Some(OwnedWorker {
                worker_id,
                discovery_id,
                spec_hash,
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
    /// published, or the same one registered with a different configuration.
    /// Registration is an Upsert, so the last case updates the worker in place.
    pub(super) add: Vec<DiscoveredWorker>,
    /// Workers to remove: endpoint no longer published, held by a different
    /// instance than the one published (which also covers a stale-scheme
    /// sibling the same-URL Upsert cannot replace), or a stale registration of
    /// the published instance left beside its current one. One entry per canonical
    /// endpoint, carrying every rank that shares it.
    pub(super) remove: Vec<RemovalTarget>,
}

/// Diff what a provider publishes against what it owns, by canonical endpoint.
///
/// An owned worker is current when it is the instance the provider describes
/// (same `discovery_id`) registered with the configuration it now publishes
/// (same fingerprint). An endpoint with a current worker is left alone, and
/// one without is added. Owned workers that are not current are handled by
/// what they are:
///
/// - **A different instance** (or none recorded) is removed, and the published
///   one added in its place.
/// - **The same instance with a stale configuration** is not removed: the add
///   is an Upsert, which replaces the registration at its URL in place. A
///   removal beside it would drain the worker, then delete whatever sits at
///   the URL when the drain ends, which by then is the updated worker. It is
///   removed only as a leftover beside a current registration, such as the old
///   spelling after a scheme change, whose URL differs from the current one.
///
/// An endpoint no longer published is removed. A DP group yields one removal
/// per endpoint, carrying a guard for every rank.
pub(super) fn compute_actions(
    desired: &DesiredState,
    registered: &[OwnedWorker],
) -> ReconcileActions {
    let mut actions = ReconcileActions::default();

    // Taken once per published record, not once per owned rank.
    let fingerprints: HashMap<&EndpointKey, String> = desired
        .workers
        .iter()
        .map(|(key, worker)| (key, worker.fingerprint()))
        .collect();

    // Endpoints with a current registration, and the owned workers that are
    // not current, each marked with whether it is the published instance.
    let mut has_current: HashSet<EndpointKey> = HashSet::new();
    let mut not_current: Vec<(EndpointKey, &OwnedWorker, bool)> = Vec::new();
    for worker in registered {
        let key = worker.endpoint.key();
        match desired.workers.get(&key) {
            Some(published)
                if worker.discovery_id.as_deref() == Some(published.discovery_id.as_str()) =>
            {
                if worker.spec_hash.as_deref() == fingerprints.get(&key).map(String::as_str) {
                    has_current.insert(key);
                } else {
                    not_current.push((key, worker, true));
                }
            }
            _ => not_current.push((key, worker, false)),
        }
    }

    // DP-rank expansions share one canonical endpoint: remove it once, but keep
    // every rank's own `(worker_id, revision)` so the guard cannot drop the
    // ranks whose revision happens to differ from an arbitrarily chosen one.
    let mut remove_by_key: HashMap<EndpointKey, RemovalTarget> = HashMap::new();
    for (key, worker, same_instance) in not_current {
        if same_instance && !has_current.contains(&key) {
            // Updated in place by the add below.
            continue;
        }
        remove_by_key
            .entry(key)
            .or_insert_with(|| RemovalTarget {
                endpoint: worker.endpoint.clone(),
                discovery_id: worker.discovery_id.clone(),
                guards: Vec::new(),
            })
            .guards
            .push((worker.worker_id.clone(), worker.revision));
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
        if !has_current.contains(key) {
            actions.add.push(worker.clone());
        }
    }
    actions
}

/// Turn a published worker into the spec its registration job will carry.
///
/// Compatibility labels go in first, then provenance — so a provider cannot
/// overwrite ownership — then the settings the host controls. The fingerprint
/// is taken from the record, never from this spec, so none of what is added
/// here can move it.
fn build_worker_spec(
    worker: &DiscoveredWorker,
    kind: DiscoveryKind,
    host: &dyn DiscoveryHost,
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
    // A nonblank served_model_name is priority #2 in create_worker's model_id chain.
    // It is stamped without the blanks around it: a model id is matched exactly.
    if let Some(model_id) = worker
        .model_id_override
        .as_deref()
        .map(str::trim)
        .filter(|model_id| !model_id.is_empty())
    {
        spec.labels
            .insert("served_model_name".to_string(), model_id.to_string());
    }
    spec.kv_connector.clone_from(&worker.kv_connector);
    spec.kv_role.clone_from(&worker.kv_role);
    spec.kv_engine_id.clone_from(&worker.kv_engine_id);
    // On the spec it is the explicit protocol, which the pairing prefers to
    // the one an engine reports about itself (a worker label): what the
    // source states for its rollout outranks what the engine was started
    // with.
    spec.pairing_protocol.clone_from(&worker.pairing_protocol);
    host.stamp(&mut spec);
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
    host: &dyn DiscoveryHost,
    started_at: time::Instant,
) {
    let registered = owned_workers(host, kind);
    let actions = compute_actions(desired, &registered);

    let desired_count = desired.workers.len();
    if actions.add.is_empty() && actions.remove.is_empty() {
        series::set_workers_discovered(kind.metric_label(), desired_count);
        return;
    }

    let Some(jobs) = host.jobs() else {
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
    let in_flight = |url: &str| jobs.job_in_flight(url);
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
        series::set_workers_discovered(kind.metric_label(), desired_count);
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
            "Removing worker {} ({} registration(s), {}): no longer published, a \
             different instance now holds the address, or it is a stale registration \
             beside the current one",
            target.endpoint.redacted(),
            target.guards.len(),
            target.discovery_id.as_deref().unwrap_or("no discovery id")
        );
        // Scheme-less for a network address, so `find_workers_by_url` reaches
        // every spelling the group was registered under. An IPC endpoint keeps
        // its scheme, because its key is a bare socket path that would not
        // parse back.
        let removal = jobs.remove_workers_at(
            target.endpoint.lookup_form(),
            target.guards.clone(),
            WorkerJobAsker::Discovery,
        );
        match removal.await {
            Ok(()) => series::record_deregistration(
                kind.metric_label(),
                series::DEREGISTRATION_RECONCILED,
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
        let spec = build_worker_spec(worker, kind, host);
        // The status read above and this submission are two steps, and the
        // pass can wait for a queue slot between them: the claim is taken
        // with the check, so a registration that took the URL since (a
        // `POST`'s) keeps it, and the next pass adds what is still missing.
        match jobs
            .add_worker(
                spec,
                WorkerRegistrationMode::Upsert,
                WorkerJobAsker::Discovery,
            )
            .await
        {
            Ok(true) => {
                series::record_registration(kind.metric_label(), series::REGISTRATION_SUCCESS);
            }
            Ok(false) => info!(
                "Worker {} has a registration in flight; its addition is left to the next pass",
                worker.endpoint
            ),
            Err(e) => {
                error!(
                    "Failed to submit worker addition for {}: {}",
                    worker.endpoint, e
                );
                series::record_registration(kind.metric_label(), series::REGISTRATION_FAILED);
            }
        }
    }

    series::set_workers_discovered(kind.metric_label(), desired_count);
    series::record_sync_duration(kind.metric_label(), started_at.elapsed());
}

#[cfg(test)]
mod tests;
