//! The policy side of a worker's registration and removal.
//!
//! A worker in the registry is not yet a worker the policies place requests
//! on: its model needs a policy and a worker count in the [`PolicyRegistry`],
//! the worker needs the completion sink the load guard fires at every
//! request's end, a cache-aware policy wants it in its worker set, and a gRPC
//! worker streams its KV events (under a policy that reads the KV-event
//! index) or its load records (otherwise) into the [`KvEventMonitor`]. A
//! removal undoes the four. The registration workflow runs [`PolicyRegistration::admit`] after
//! its insert and the removal workflow [`PolicyRegistration::release`] before
//! its teardown; the mesh runs both for the workers it imports from peers,
//! which enter and leave the registry outside the workflows
//! (`mesh::adapters::ImportPolicySync`).

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use tokio::sync::Semaphore;
use tracing::error;

use super::PolicyRegistry;
use crate::{
    app_context::AppContext,
    worker::{
        ConnectionMode, KvEventMonitor, SubscriptionKind, Worker, WorkerMonitor, WorkerRegistry,
    },
};

/// Bound CPU-heavy radix-tree purges during a fleet-wide scale-down. The
/// worker has already left the routing registry before any task waits here.
const MAX_CONCURRENT_CACHE_REMOVALS: usize = 4;
static CACHE_REMOVAL_PERMITS: Semaphore = Semaphore::const_new(MAX_CONCURRENT_CACHE_REMOVALS);

/// The registries and monitors the policy side of a registration touches.
#[derive(Clone)]
pub struct PolicyRegistration {
    pub policy_registry: Arc<PolicyRegistry>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub kv_event_monitor: Option<Arc<KvEventMonitor>>,
    pub worker_monitor: Option<Arc<WorkerMonitor>>,
}

impl PolicyRegistration {
    /// The application's registries and monitors.
    pub fn of(app_context: &AppContext) -> Self {
        Self {
            policy_registry: Arc::clone(&app_context.policy_registry),
            worker_registry: Arc::clone(&app_context.worker_registry),
            kv_event_monitor: app_context.kv_event_monitor.clone(),
            worker_monitor: app_context.worker_monitor.clone(),
        }
    }

    /// Admit `worker` to the policies: its model's worker count and policy
    /// (chosen by `policy_hint`, the registration's `policy` label, when the
    /// model has none yet), the completion sink, the cache-aware policy's
    /// worker set, and its stream into the KV event monitor. The stream
    /// follows the policy that places the worker, the role's policy for a
    /// prefill, decode or encode worker rather than the model's: its KV
    /// events into the model's index under a policy that reads the index,
    /// its load records alone otherwise, when the worker monitor wants loads
    /// at all (always by default; see `WorkerMonitor::loads_wanted`); only
    /// gRPC workers stream. Returns whether the policy that places `worker`
    /// reads the KV-event index.
    pub async fn admit(&self, worker: &Arc<dyn Worker>, policy_hint: Option<&str>) -> bool {
        let model_id = worker.model_id();
        self.policy_registry.on_worker_added(model_id, policy_hint);
        // Policies learn of the request's end through the worker's load
        // guard; the registry routes it to the policy that placed it.
        worker.set_completion_sink(Some(self.policy_registry.completion_sink()));
        let cache_aware = self
            .policy_registry
            .get_policy(model_id)
            .is_some_and(|policy| policy.name() == "cache_aware");
        if cache_aware {
            let all_workers = self.worker_registry.get_by_model(model_id);
            self.policy_registry
                .init_cache_aware_policy(model_id, &all_workers);
        }
        let needs_kv_events = self
            .policy_registry
            .policy_for_worker(worker.as_ref())
            .needs_kv_events();
        self.start_stream(worker, needs_kv_events).await;
        needs_kv_events
    }

    /// The PD policies' worker sets, read again from the registry: the
    /// prefill and decode cache-aware policies' trees and the prefill bucket
    /// policy's per-model URL partition (`init_pd_cache_aware_policies`,
    /// `init_pd_bucket_policies`). The registration step runs it after its
    /// admissions; the import sync after it admits an import with a prefill
    /// or decode role, so a model served by imports alone has its bucket
    /// from the first import on.
    pub fn init_pd_policies(&self) {
        let prefill_workers = self.worker_registry.get_prefill_workers();
        let decode_workers = self.worker_registry.get_decode_workers();
        self.policy_registry
            .init_pd_cache_aware_policies(&prefill_workers, &decode_workers);
        if !prefill_workers.is_empty()
            && self.policy_registry.get_prefill_policy().name() == "bucket"
        {
            self.policy_registry
                .init_pd_bucket_policies(&prefill_workers);
        }
    }

    /// Hand the stream under `from`'s address to `to`, the registry's worker
    /// at that address now (a local claim of an import's address, or a
    /// registration that took it after the import's tombstone), and release
    /// `from`'s share of its model's count. The stream was started for
    /// `from` and holds its worker object: a stream carries its object into
    /// the pushed loads, the overload verdict and the liveness it reports,
    /// the load state drops a report whose object the registry no longer
    /// holds (`is_current_incarnation`) while the records' receipts suppress
    /// the poll, so `to`'s load would freeze. The stream stops here, and is
    /// started again for `to` when `to`'s model has a policy, under the kind
    /// the policy that places `to` needs. When the model has none yet,
    /// `to`'s own admission is still to come and starts the right kind: it
    /// keeps a running stream, so one started here under a guessed kind
    /// would stick. Either order
    /// with the registration workflow's admission ends with one stream,
    /// `to`'s (`on_worker_added` keeps an active stream and waits for one
    /// being removed). The URL's dispatch tally is `to`'s now and stays; the
    /// tree tenants `from` learned under models `to` does not serve are
    /// purged. `from`'s count goes last, so when the two share a model the
    /// policy still decides the kind.
    pub async fn hand_over(&self, from: &Arc<dyn Worker>, to: &Arc<dyn Worker>) {
        if let Some(monitor) = &self.kv_event_monitor {
            monitor.on_worker_removed(from.url()).await;
            if self.policy_registry.get_policy(to.model_id()).is_some() {
                let needs_kv_events = self
                    .policy_registry
                    .policy_for_worker(to.as_ref())
                    .needs_kv_events();
                self.start_stream(to, needs_kv_events).await;
            }
        }
        let kept: HashSet<String> = WorkerRegistry::worker_model_ids(to).into_iter().collect();
        let mut removals: HashMap<String, HashSet<String>> = HashMap::new();
        for model_id in WorkerRegistry::worker_model_ids(from) {
            if !kept.contains(&model_id) {
                removals
                    .entry(model_id)
                    .or_default()
                    .insert(from.url().to_string());
            }
        }
        self.remove_cache_state(removals).await;
        self.policy_registry.on_worker_removed(from.model_id());
    }

    /// The worker's stream into the KV event monitor: its KV events into the
    /// model's index when the policy that places it reads the index
    /// (`needs_kv_events`), its load records alone otherwise, when the worker
    /// monitor wants loads at all (always by default; see
    /// `WorkerMonitor::loads_wanted`). Only gRPC workers stream.
    async fn start_stream(&self, worker: &Arc<dyn Worker>, needs_kv_events: bool) {
        if *worker.connection_mode() != ConnectionMode::Grpc {
            return;
        }
        let Some(monitor) = &self.kv_event_monitor else {
            return;
        };
        let kind = if needs_kv_events {
            Some(SubscriptionKind::Events)
        } else {
            self.worker_monitor
                .as_ref()
                .is_some_and(|loads| loads.loads_wanted(std::slice::from_ref(worker)))
                .then_some(SubscriptionKind::Loads)
        };
        if let Some(kind) = kind {
            monitor.on_worker_added(worker, kind).await;
        }
    }

    /// Release `worker` from the policies: its stream ends and the monitor
    /// takes it out of the index; the policies drop the state they keep on
    /// it (a load-aware policy's dispatch tally, so it does not leak under
    /// churn; a cache-aware policy's tree tenants, purged on the blocking
    /// pool under a bound); its model's worker count falls, and the policy
    /// goes with the last worker.
    pub async fn release(&self, worker: &Arc<dyn Worker>) {
        self.release_all(std::slice::from_ref(worker)).await;
    }

    /// [`Self::release`] for a removal batch: every worker's stream ends
    /// and its per-worker policy state goes, the batch's tree tenants are
    /// purged in one task under one permit (a fleet-wide scale-down costs
    /// one purge per batch, not one per worker), then every model's count
    /// falls.
    pub async fn release_all(&self, workers: &[Arc<dyn Worker>]) {
        for worker in workers {
            if let Some(monitor) = &self.kv_event_monitor {
                monitor.on_worker_removed(worker.url()).await;
            }
            self.policy_registry
                .remove_worker_from_policies(worker.url());
        }
        let mut cache_removals: HashMap<String, HashSet<String>> = HashMap::new();
        for worker in workers {
            for model_id in WorkerRegistry::worker_model_ids(worker) {
                cache_removals
                    .entry(model_id)
                    .or_default()
                    .insert(worker.url().to_string());
            }
        }
        self.remove_cache_state(cache_removals).await;
        for worker in workers {
            self.policy_registry.on_worker_removed(worker.model_id());
        }
    }

    async fn remove_cache_state(&self, removals: HashMap<String, HashSet<String>>) {
        if removals.is_empty() {
            return;
        }
        let Ok(permit) = CACHE_REMOVAL_PERMITS.acquire().await else {
            error!("Cache-removal semaphore closed");
            return;
        };
        let policy_registry = Arc::clone(&self.policy_registry);
        let result = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            for (model_id, worker_urls) in removals {
                policy_registry.remove_workers_from_cache_aware(&model_id, &worker_urls);
            }
        })
        .await;
        if let Err(error) = result {
            error!(%error, "Cache-aware worker cleanup task failed");
        }
    }
}
