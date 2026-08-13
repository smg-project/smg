use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use dashmap::DashMap;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

/// Policy Registry for managing model-to-policy mappings
///
/// This registry manages the dynamic assignment of load balancing policies to models.
/// When the first worker of a new model is added, it determines the policy for that model.
/// All subsequent workers of the same model use the established policy.
/// When the last worker of a model is removed, the policy mapping is cleaned up.
use super::{
    BucketPolicy, CacheAwarePolicy, DPRankLoadPolicy, LoadBalancingPolicy, ManualConfig,
    ManualPolicy, PolicyFactory, SelectWorkerInfo,
};
use crate::{
    config::types::{PolicyConfig, RoutingKeyOverrideConfig},
    mesh::adapters::TreeSyncAdapter,
    policies::cache_aware::LoadReceiver,
    routers::common::header_utils::extract_routing_key,
    worker::{KvEventMonitor, Worker},
};

/// Registry for managing model-to-policy mappings
#[derive(Clone)]
pub struct PolicyRegistry {
    /// Model ID -> Policy instance mapping (lock-free reads via DashMap)
    model_policies: Arc<DashMap<String, Arc<dyn LoadBalancingPolicy>>>,

    /// Model ID -> Worker count for cleanup tracking (lock-free reads via DashMap)
    model_worker_counts: Arc<DashMap<String, usize>>,

    /// Default policy instance (cached, immutable after creation)
    default_policy: Arc<dyn LoadBalancingPolicy>,

    /// Operator config the default policy was built from; hinted per-model
    /// policies of the same type are built from it so they inherit the
    /// operator's tunables.
    default_policy_config: PolicyConfig,

    /// Prefill policy for PD mode (set once at startup, lock-free reads via OnceLock)
    prefill_policy: Arc<OnceLock<Arc<dyn LoadBalancingPolicy>>>,

    /// Decode policy for PD mode (set once at startup, lock-free reads via OnceLock)
    decode_policy: Arc<OnceLock<Arc<dyn LoadBalancingPolicy>>>,

    /// Encode policy for EPD mode (set once at startup, lock-free reads via OnceLock)
    encode_policy: Arc<OnceLock<Arc<dyn LoadBalancingPolicy>>>,

    /// Optional KV event monitor for event-driven cache-aware routing.
    /// When set, new CacheAwarePolicy instances are injected with this monitor.
    kv_event_monitor: Arc<RwLock<Option<Arc<KvEventMonitor>>>>,

    /// Optional backend load-snapshot receiver from the `WorkerMonitor`. When
    /// set, new CacheAwarePolicy instances are injected with it for the KV-usage
    /// imbalance trigger.
    load_rx: Arc<RwLock<Option<LoadReceiver>>>,

    /// Optional mesh outbound bridge. When set, every cache-aware policy
    /// created here is attached to this adapter so its local tree inserts
    /// join the next gossip round. Absence means mesh is disabled and
    /// cache-aware policies stay local.
    mesh_tree_sync: Arc<RwLock<Option<Arc<TreeSyncAdapter>>>>,

    // DP-rank policy: Supports the selection of dp-rank outside the engine.
    dp_rank_policy: Arc<OnceLock<Arc<dyn DPRankLoadPolicy>>>,

    /// Shared sticky selector for the `X-SMG-Routing-Key` override. `Some` when the
    /// override is enabled; consulted (instead of the configured policy) for keyed
    /// requests via [`PolicyRegistry::select_worker`].
    routing_key_sticky: Option<Arc<ManualPolicy>>,
}

impl PolicyRegistry {
    /// Create a new PolicyRegistry with a default policy (no routing-key override).
    pub fn new(default_policy_config: PolicyConfig) -> Self {
        Self::with_override(default_policy_config, RoutingKeyOverrideConfig::default())
    }

    /// Create a PolicyRegistry. When `routing_key_override.enabled`, builds a shared
    /// sticky selector consulted for keyed requests in [`Self::select_worker`].
    pub fn with_override(
        default_policy_config: PolicyConfig,
        routing_key_override: RoutingKeyOverrideConfig,
    ) -> Self {
        let default_policy = Self::create_policy_from_config(&default_policy_config);
        let routing_key_sticky = routing_key_override.enabled.then(|| {
            Arc::new(ManualPolicy::with_config(ManualConfig {
                eviction_interval_secs: routing_key_override.eviction_interval_secs,
                max_idle_secs: routing_key_override.max_idle_secs,
                assignment_mode: routing_key_override.assignment_mode,
            }))
        });

        Self {
            model_policies: Arc::new(DashMap::new()),
            model_worker_counts: Arc::new(DashMap::new()),
            default_policy,
            default_policy_config,
            prefill_policy: Arc::new(OnceLock::new()),
            decode_policy: Arc::new(OnceLock::new()),
            encode_policy: Arc::new(OnceLock::new()),
            kv_event_monitor: Arc::new(RwLock::new(None)),
            load_rx: Arc::new(RwLock::new(None)),
            mesh_tree_sync: Arc::new(RwLock::new(None)),
            dp_rank_policy: Arc::new(OnceLock::new()),
            routing_key_sticky,
        }
    }

    /// Select a worker, applying the `X-SMG-Routing-Key` sticky override when it is
    /// enabled, the request carries the header, and the configured policy does not
    /// already honor the key (`manual` / `consistent_hashing`). Otherwise delegates
    /// to `policy`. `policy.name()` stays the real policy (for metrics).
    pub fn select_worker(
        &self,
        policy: &Arc<dyn LoadBalancingPolicy>,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        if let Some(sticky) = self.routing_key_sticky.as_ref() {
            if Self::routing_key_override_applies(policy.name())
                && extract_routing_key(info.headers).is_some()
            {
                return sticky.select_worker(workers, info);
            }
        }
        policy.select_worker(workers, info)
    }

    /// Policies that already honor `X-SMG-Routing-Key` keep their own handling; all
    /// others (cache_aware, least_load, prefix_hash, ...) get the sticky override.
    fn routing_key_override_applies(name: &str) -> bool {
        !matches!(name, "manual" | "consistent_hashing")
    }

    /// Set KV event monitor (thread-safe, can be called after initialization).
    /// Propagates to all existing cache-aware policies (including default, prefill, decode).
    pub fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        {
            let mut guard = self.kv_event_monitor.write();
            guard.clone_from(&monitor);
        }

        // Propagate to existing cache-aware policies so they don't miss the monitor.
        // This covers the default_policy (created before the monitor was available)
        // and any model/PD policies that were already set up.
        Self::maybe_inject_monitor(&self.default_policy, monitor.as_ref());
        if let Some(p) = self.prefill_policy.get() {
            Self::maybe_inject_monitor(p, monitor.as_ref());
        }
        if let Some(p) = self.decode_policy.get() {
            Self::maybe_inject_monitor(p, monitor.as_ref());
        }
        if let Some(p) = self.encode_policy.get() {
            Self::maybe_inject_monitor(p, monitor.as_ref());
        }
        for entry in self.model_policies.iter() {
            Self::maybe_inject_monitor(entry.value(), monitor.as_ref());
        }
    }

    /// Inject KV event monitor into a policy if it's cache-aware.
    fn maybe_inject_monitor(
        policy: &Arc<dyn LoadBalancingPolicy>,
        monitor: Option<&Arc<KvEventMonitor>>,
    ) {
        if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
            cache_aware.set_kv_event_monitor(monitor.cloned());
        }
    }

    /// Set the backend load-snapshot receiver (thread-safe, can be called after
    /// initialization). Propagates to all existing cache-aware policies.
    pub fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        {
            let mut guard = self.load_rx.write();
            guard.clone_from(&rx);
        }
        Self::maybe_inject_load_rx(&self.default_policy, rx.as_ref());
        if let Some(p) = self.prefill_policy.get() {
            Self::maybe_inject_load_rx(p, rx.as_ref());
        }
        if let Some(p) = self.decode_policy.get() {
            Self::maybe_inject_load_rx(p, rx.as_ref());
        }
        if let Some(p) = self.encode_policy.get() {
            Self::maybe_inject_load_rx(p, rx.as_ref());
        }
        for entry in self.model_policies.iter() {
            Self::maybe_inject_load_rx(entry.value(), rx.as_ref());
        }
    }

    /// Inject the load receiver into a policy if it's cache-aware.
    fn maybe_inject_load_rx(policy: &Arc<dyn LoadBalancingPolicy>, rx: Option<&LoadReceiver>) {
        if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
            cache_aware.set_load_receiver(rx.cloned());
        }
    }

    /// Attach the mesh outbound bridge (thread-safe, can be called after
    /// initialization). Every existing cache-aware policy gets the adapter
    /// wired in and its `populate_hash_index` flag flipped on (both flip
    /// atomically inside [`CacheAwarePolicy::set_mesh_tree_sync`]); every
    /// future cache-aware policy created here inherits both. Pass `None`
    /// to detach.
    pub fn set_mesh_tree_sync(&self, adapter: Option<Arc<TreeSyncAdapter>>) {
        {
            let mut guard = self.mesh_tree_sync.write();
            guard.clone_from(&adapter);
        }
        Self::maybe_inject_mesh_tree_sync(&self.default_policy, adapter.as_ref());
        if let Some(p) = self.prefill_policy.get() {
            Self::maybe_inject_mesh_tree_sync(p, adapter.as_ref());
        }
        if let Some(p) = self.decode_policy.get() {
            Self::maybe_inject_mesh_tree_sync(p, adapter.as_ref());
        }
        if let Some(p) = self.encode_policy.get() {
            Self::maybe_inject_mesh_tree_sync(p, adapter.as_ref());
        }
        for entry in self.model_policies.iter() {
            Self::maybe_inject_mesh_tree_sync(entry.value(), adapter.as_ref());
        }
    }

    /// Inject the mesh adapter into a policy if it's cache-aware.
    /// The setter also flips `populate_hash_index` to match adapter
    /// presence, so callers do not need to touch that flag directly.
    fn maybe_inject_mesh_tree_sync(
        policy: &Arc<dyn LoadBalancingPolicy>,
        adapter: Option<&Arc<TreeSyncAdapter>>,
    ) {
        if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
            cache_aware.set_mesh_tree_sync(adapter.cloned());
        }
    }

    /// Called when a worker is added
    /// Returns the policy that should be used for this worker's model
    pub fn on_worker_added(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // Increment worker count using DashMap entry API
        let count = self
            .model_worker_counts
            .entry(model_id.to_string())
            .and_modify(|c| *c += 1)
            .or_insert(1);
        debug!("Worker added for model {}, count: {}", model_id, *count);
        drop(count); // Release the entry lock

        // Check if model already has a policy (lock-free read via DashMap)
        if let Some(existing_policy) = self.model_policies.get(model_id) {
            debug!(
                "Model {} already has policy: {}",
                model_id,
                existing_policy.name()
            );
            return Arc::clone(&existing_policy);
        }

        // New model - determine policy
        let policy = self.determine_policy_for_model(model_id, policy_hint);

        info!(
            "Assigning policy {} to new model {}",
            policy.name(),
            model_id
        );

        // Inject and publish under the integration guards so a concurrent
        // setter cannot fall between them: it either wrote before the reads
        // here, or its propagation scan runs after the insert and finds
        // this policy.
        {
            let monitor = self.kv_event_monitor.read();
            let load_rx = self.load_rx.read();
            let mesh = self.mesh_tree_sync.read();
            Self::maybe_inject_monitor(&policy, monitor.as_ref());
            Self::maybe_inject_load_rx(&policy, load_rx.as_ref());
            Self::maybe_inject_mesh_tree_sync(&policy, mesh.as_ref());
            self.model_policies
                .insert(model_id.to_string(), Arc::clone(&policy));
        }

        policy
    }

    /// Called when a worker is removed
    pub fn on_worker_removed(&self, model_id: &str) {
        // Decrement worker count and check if cleanup needed
        let should_cleanup = if let Some(mut count_ref) = self.model_worker_counts.get_mut(model_id)
        {
            *count_ref = count_ref.saturating_sub(1);
            debug!(
                "Worker removed for model {}, count: {}",
                model_id, *count_ref
            );
            if *count_ref == 0 {
                drop(count_ref); // Release before remove
                self.model_worker_counts.remove(model_id);
                true
            } else {
                false
            }
        } else {
            warn!(
                "Attempted to remove worker for model {} with no registered workers",
                model_id
            );
            false
        };

        // Clean up policy if this was the last worker
        if should_cleanup {
            if let Some((_, policy)) = self.model_policies.remove(model_id) {
                info!(
                    "Removed policy {} for model {} (last worker removed)",
                    policy.name(),
                    model_id
                );
            }
        }
    }

    /// Get the policy for a model (lock-free via DashMap)
    pub fn get_policy(&self, model_id: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        self.model_policies.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Get the default policy
    pub fn get_default_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        Arc::clone(&self.default_policy)
    }

    /// Get policy for a model, or default if not found
    pub fn get_policy_or_default(&self, model_id: &str) -> Arc<dyn LoadBalancingPolicy> {
        self.get_policy(model_id)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Every distinct policy this registry might dispatch requests
    /// through for `model_id`, ordered from most-specific to least
    /// (per-model → default → PD/EPD legs). Deduplicated by `Arc`
    /// pointer identity so a policy that appears in multiple slots
    /// (e.g. the same `CacheAwarePolicy` as both default and prefill
    /// leg) is returned once.
    ///
    /// The tree-sync bridge uses this to reach every `CacheAwarePolicy`
    /// that could have produced a delta for `model_id` — reaching
    /// only `model_policies` misses PD-leg and default-fallback
    /// deployments, and inbound deltas never resolve.
    pub(crate) fn policies_for_model(&self, model_id: &str) -> Vec<Arc<dyn LoadBalancingPolicy>> {
        let mut out: Vec<Arc<dyn LoadBalancingPolicy>> = Vec::new();
        let mut push = |candidate: Arc<dyn LoadBalancingPolicy>| {
            let ptr = Arc::as_ptr(&candidate) as *const ();
            if !out
                .iter()
                .any(|existing| Arc::as_ptr(existing) as *const () == ptr)
            {
                out.push(candidate);
            }
        };
        if let Some(policy) = self.get_policy(model_id) {
            push(policy);
        }
        push(Arc::clone(&self.default_policy));
        if let Some(p) = self.prefill_policy.get() {
            push(Arc::clone(p));
        }
        if let Some(p) = self.decode_policy.get() {
            push(Arc::clone(p));
        }
        if let Some(p) = self.encode_policy.get() {
            push(Arc::clone(p));
        }
        out
    }

    /// Determine policy for a new model
    fn determine_policy_for_model(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
        // 1. Check policy hint from worker
        if let Some(policy_type) = policy_hint {
            debug!("Using policy hint '{}' for model {}", policy_type, model_id);
            return self.create_policy_from_type(policy_type);
        }

        // 2. Use default policy
        debug!("Using default policy for model {}", model_id);
        Arc::clone(&self.default_policy)
    }

    /// Create a policy from a type string. A hint naming the operator's
    /// default policy type is built from that config so it inherits the
    /// operator's tunables; other types are built with their defaults.
    /// Integration injection happens at publication in
    /// [`Self::on_worker_added`].
    fn create_policy_from_type(&self, policy_type: &str) -> Arc<dyn LoadBalancingPolicy> {
        if Self::hint_matches_config(policy_type, self.default_policy_config.name()) {
            Self::create_policy_from_config(&self.default_policy_config)
        } else if let Some(policy) = PolicyFactory::create_by_name(policy_type) {
            policy
        } else {
            warn!("Unknown policy type '{}', using default", policy_type);
            Arc::clone(&self.default_policy)
        }
    }

    /// Whether a policy hint names `config_name`, accepting the same
    /// underscore-stripped aliases as [`PolicyFactory::create_by_name`].
    fn hint_matches_config(policy_type: &str, config_name: &str) -> bool {
        let hint = policy_type.to_lowercase();
        hint == config_name || hint == config_name.replace('_', "")
    }

    /// Create a policy from a PolicyConfig (delegates to PolicyFactory)
    fn create_policy_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        PolicyFactory::create_from_config(config)
    }

    /// Get current model->policy mappings (for debugging/monitoring)
    pub fn get_all_mappings(&self) -> HashMap<String, String> {
        self.model_policies
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().name().to_string()))
            .collect()
    }

    /// Get worker counts per model
    pub fn get_worker_counts(&self) -> HashMap<String, usize> {
        self.model_worker_counts
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }

    /// Clear all policies (useful for testing)
    pub fn clear(&self) {
        self.model_policies.clear();
        self.model_worker_counts.clear();
    }

    /// Set the prefill policy for PD mode (lock-free, set once at startup)
    pub fn set_prefill_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        let _ = self.prefill_policy.set(policy);
    }

    pub fn set_dp_rank_policy(&self, policy: Arc<dyn DPRankLoadPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        debug!("set dp rank policy");
        let _ = self.dp_rank_policy.set(policy);
    }

    pub fn get_dp_rank_policy(&self) -> Option<Arc<dyn DPRankLoadPolicy>> {
        self.dp_rank_policy.get().map(Arc::clone)
    }

    /// Set the decode policy for PD mode (lock-free, set once at startup)
    pub fn set_decode_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        let _ = self.decode_policy.set(policy);
    }

    /// Set the encode policy for EPD mode (lock-free, set once at startup)
    pub fn set_encode_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        // OnceLock::set returns Err if already set, which we ignore since
        // the policy should only be set once at startup
        let _ = self.encode_policy.set(policy);
    }

    /// Get the prefill policy for PD mode, or default if not set (lock-free)
    pub fn get_prefill_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.prefill_policy
            .get()
            .map(Arc::clone)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Get the decode policy for PD mode, or default if not set (lock-free)
    pub fn get_decode_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.decode_policy
            .get()
            .map(Arc::clone)
            .unwrap_or_else(|| self.get_default_policy())
    }

    /// Get the encode policy for EPD mode. Falls back to consistent_hashing so
    /// repeated multimodal items keep stable affinity even when the main policy is
    /// load-oriented or random.
    pub fn get_encode_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.encode_policy
            .get()
            .map(Arc::clone)
            .unwrap_or_else(|| PolicyFactory::create_from_config(&PolicyConfig::ConsistentHashing))
    }

    /// Get all load-aware policies that need periodic load updates (lock-free).
    ///
    /// These are policies whose routing depends on the worker load monitor:
    /// `power_of_two` and `least_load`.
    pub fn get_all_load_aware_policies(&self) -> Vec<Arc<dyn LoadBalancingPolicy>> {
        fn is_load_aware(name: &str) -> bool {
            name == "power_of_two" || name == "least_load"
        }

        let mut policies = Vec::new();

        if is_load_aware(self.default_policy.name()) {
            policies.push(Arc::clone(&self.default_policy));
        }

        // Get prefill, decode, and encode policies (lock-free via OnceLock::get)
        let prefill_policy_opt = self.prefill_policy.get();
        let decode_policy_opt = self.decode_policy.get();
        let encode_policy_opt = self.encode_policy.get();

        if let Some(policy) = prefill_policy_opt {
            if is_load_aware(policy.name()) && !Arc::ptr_eq(policy, &self.default_policy) {
                policies.push(Arc::clone(policy));
            }
        }

        if let Some(policy) = decode_policy_opt {
            if is_load_aware(policy.name())
                && !Arc::ptr_eq(policy, &self.default_policy)
                && !prefill_policy_opt.is_some_and(|p| Arc::ptr_eq(p, policy))
            {
                policies.push(Arc::clone(policy));
            }
        }

        if let Some(policy) = encode_policy_opt {
            if is_load_aware(policy.name())
                && !Arc::ptr_eq(policy, &self.default_policy)
                && !prefill_policy_opt.is_some_and(|p| Arc::ptr_eq(p, policy))
                && !decode_policy_opt.is_some_and(|p| Arc::ptr_eq(p, policy))
            {
                policies.push(Arc::clone(policy));
            }
        }

        for entry in self.model_policies.iter() {
            let policy = entry.value();
            if is_load_aware(policy.name()) {
                let already_added = policies.iter().any(|p| Arc::ptr_eq(p, policy));
                if !already_added {
                    policies.push(Arc::clone(policy));
                }
            }
        }

        policies
    }

    /// Initialize cache-aware policy with workers if applicable
    /// This should be called after workers are registered for a model
    pub fn init_cache_aware_policy(&self, model_id: &str, workers: &[Arc<dyn Worker>]) {
        // Get the policy for this model
        if let Some(policy) = self.get_policy(model_id) {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    debug!(
                        "Initializing cache-aware policy with {} workers for model {}",
                        workers.len(),
                        model_id
                    );
                    cache_aware.init_workers(workers);
                }
            }
        }
    }

    /// Remove a worker from cache-aware policy if applicable
    /// This should be called when a worker is being removed
    pub fn remove_worker_from_cache_aware(&self, model_id: &str, worker_url: &str) {
        // Get the policy for this model
        if let Some(policy) = self.get_policy(model_id) {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    cache_aware.remove_worker_by_url(worker_url);
                    debug!(
                        "Removed worker {} from cache-aware policy for model {}",
                        worker_url, model_id
                    );
                }
            }
        }
    }

    /// Remove a worker from PD cache-aware policies if applicable
    /// This should be called when a prefill or decode worker is being removed
    pub fn remove_worker_from_pd_cache_aware(&self, worker_url: &str) {
        for (worker_type, policy) in [
            ("prefill", self.prefill_policy.get()),
            ("decode", self.decode_policy.get()),
            ("encode", self.encode_policy.get()),
        ] {
            if let Some(policy) = policy {
                if policy.name() == "cache_aware" {
                    if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                        cache_aware.remove_worker_by_url(worker_url);
                        debug!(
                            "Removed worker {} from {} cache-aware policy",
                            worker_url, worker_type
                        );
                    }
                }
            }
        }
    }

    /// Drop a removed worker's cached load report from all load-aware policies
    /// (`power_of_two`, `least_load`).
    ///
    /// These policies cache per-worker load reports keyed by URL; without this
    /// their caches would grow unbounded under worker churn. Called on worker
    /// removal alongside the cache-aware cleanup above.
    pub fn remove_worker_from_load_aware(&self, worker_url: &str) {
        for policy in self.get_all_load_aware_policies() {
            policy.remove_worker(worker_url);
        }
    }

    /// Initialize cache-aware policies for PD mode (prefill and decode) - lock-free
    pub fn init_pd_cache_aware_policies(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
    ) {
        // Initialize prefill policy if it's cache-aware (lock-free via OnceLock::get)
        if let Some(prefill_policy) = self.prefill_policy.get() {
            if prefill_policy.name() == "cache_aware" {
                if let Some(cache_aware) =
                    prefill_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    if !prefill_workers.is_empty() {
                        debug!(
                            "Initializing prefill cache-aware policy with {} workers",
                            prefill_workers.len()
                        );
                        cache_aware.init_workers(prefill_workers);
                    }
                }
            }
        }

        // Initialize decode policy if it's cache-aware (lock-free via OnceLock::get)
        if let Some(decode_policy) = self.decode_policy.get() {
            if decode_policy.name() == "cache_aware" {
                if let Some(cache_aware) = decode_policy.as_any().downcast_ref::<CacheAwarePolicy>()
                {
                    if !decode_workers.is_empty() {
                        debug!(
                            "Initializing decode cache-aware policy with {} workers",
                            decode_workers.len()
                        );
                        cache_aware.init_workers(decode_workers);
                    }
                }
            }
        }
    }

    /// Initialize bucket policies for PD mode - lock-free
    pub fn init_pd_bucket_policies(&self, prefill_workers: &[Arc<dyn Worker>]) {
        // Initialize prefill policy if it's bucket (lock-free via OnceLock::get)
        if let Some(prefill_policy) = self.prefill_policy.get() {
            if prefill_policy.name() == "bucket" {
                if let Some(bucket) = prefill_policy.as_any().downcast_ref::<BucketPolicy>() {
                    if !prefill_workers.is_empty() {
                        debug!(
                            "Initializing prefill bucket policy with {} workers",
                            prefill_workers.len()
                        );
                        bucket.init_prefill_worker_urls(prefill_workers);
                    }
                }
            }
        }
    }
}

impl std::fmt::Debug for PolicyRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyRegistry")
            .field("model_policies", &self.model_policies)
            .field("model_worker_counts", &self.model_worker_counts)
            .field("default_policy", &self.default_policy.name())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{
        policies::{CacheAwareConfig, LeastLoadPolicy, SelectWorkerInfo},
        worker::{BasicWorkerBuilder, Worker, WorkerType},
    };

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn worker(url: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(worker_type)
                .health_config(no_health_check())
                .build(),
        )
    }

    fn cache_aware_policy() -> Arc<dyn LoadBalancingPolicy> {
        Arc::new(CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        }))
    }

    fn headers_with_key(key: &str) -> http::HeaderMap {
        let mut h = http::HeaderMap::new();
        h.insert("x-smg-routing-key", key.parse().unwrap());
        h
    }

    #[test]
    fn override_eligibility_skips_key_native_policies() {
        // Policies that already honor X-SMG-Routing-Key are skipped; others (incl.
        // prefix_hash, which routes by tokens) get the sticky override.
        assert!(PolicyRegistry::routing_key_override_applies("cache_aware"));
        assert!(PolicyRegistry::routing_key_override_applies("prefix_hash"));
        assert!(PolicyRegistry::routing_key_override_applies("least_load"));
        assert!(!PolicyRegistry::routing_key_override_applies("manual"));
        assert!(!PolicyRegistry::routing_key_override_applies(
            "consistent_hashing"
        ));
    }

    #[test]
    fn override_routes_keyed_request_stickily() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                enabled: true,
                ..Default::default()
            },
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
            worker("http://w3", WorkerType::Regular),
        ];
        let headers = headers_with_key("session-A");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let first = reg.select_worker(&policy, &workers, &info).unwrap();
        for _ in 0..5 {
            assert_eq!(reg.select_worker(&policy, &workers, &info), Some(first));
        }
    }

    #[test]
    fn override_without_key_uses_configured_policy() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                enabled: true,
                ..Default::default()
            },
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let info = SelectWorkerInfo::default(); // no key header
                                                // RoundRobin alternates -> proves the configured policy is used, not sticky.
        let a = reg.select_worker(&policy, &workers, &info).unwrap();
        let b = reg.select_worker(&policy, &workers, &info).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn override_disabled_ignores_key() {
        let reg = PolicyRegistry::new(PolicyConfig::RoundRobin); // override off
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let headers = headers_with_key("session-A");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        // Override off -> the key is ignored, RoundRobin alternates.
        let a = reg.select_worker(&policy, &workers, &info).unwrap();
        let b = reg.select_worker(&policy, &workers, &info).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_policy_registry_basic() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // First worker of a model sets the policy
        let policy1 = registry.on_worker_added("llama-3", Some("cache_aware"));
        assert_eq!(policy1.name(), "cache_aware");

        // Second worker of same model uses existing policy
        let policy2 = registry.on_worker_added("llama-3", Some("round_robin"));
        assert_eq!(policy2.name(), "cache_aware"); // Ignores hint, uses existing

        // Different model can have different policy
        let policy3 = registry.on_worker_added("gpt-4", Some("random"));
        assert_eq!(policy3.name(), "random");

        // Check mappings
        let mappings = registry.get_all_mappings();
        assert_eq!(mappings.get("llama-3").unwrap(), "cache_aware");
        assert_eq!(mappings.get("gpt-4").unwrap(), "random");

        // Check worker counts
        let counts = registry.get_worker_counts();
        assert_eq!(*counts.get("llama-3").unwrap(), 2);
        assert_eq!(*counts.get("gpt-4").unwrap(), 1);
    }

    #[test]
    fn test_policy_registry_cleanup() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // Add workers
        registry.on_worker_added("llama-3", Some("cache_aware"));
        registry.on_worker_added("llama-3", None);
        assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&2));

        // Remove one worker - policy should remain
        registry.on_worker_removed("llama-3");
        assert!(registry.get_policy("llama-3").is_some());
        assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&1));

        // Remove last worker - policy should be cleaned up
        registry.on_worker_removed("llama-3");
        assert!(registry.get_policy("llama-3").is_none());
        assert_eq!(registry.get_worker_counts().get("llama-3"), None);
    }

    #[test]
    fn test_passthrough_is_not_load_aware() {
        // Passthrough must not be polled by the WorkerMonitor: with it as the
        // default policy (and a passthrough model policy too), the load-aware
        // set stays empty.
        let registry = PolicyRegistry::new(PolicyConfig::Passthrough);
        registry.on_worker_added("m", Some("passthrough"));
        assert!(registry.get_all_load_aware_policies().is_empty());
    }

    #[test]
    fn test_default_policy() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // No hint, no template - uses default
        let policy = registry.on_worker_added("unknown-model", None);
        assert_eq!(policy.name(), "round_robin");

        // Get default directly
        let default = registry.get_default_policy();
        assert_eq!(default.name(), "round_robin");
    }

    #[test]
    fn hinted_policy_inherits_operator_cache_aware_config() {
        let registry = PolicyRegistry::new(PolicyConfig::CacheAware {
            cache_threshold: 0.9,
            balance_abs_threshold: 64,
            balance_rel_threshold: 2.5,
            eviction_interval_secs: 0,
            max_tree_size: 4096,
            block_size: 32,
            balance_token_usage_threshold: 0.5,
            overload_token_usage_threshold: 0.8,
        });

        // Hinted policy is a fresh per-model instance, not the shared default.
        let policy = registry.on_worker_added("llama-3", Some("cache_aware"));
        assert!(!Arc::ptr_eq(&policy, &registry.get_default_policy()));

        let config = policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .unwrap()
            .config_for_test();
        assert_eq!(config.cache_threshold, 0.9);
        assert_eq!(config.balance_abs_threshold, 64);
        assert_eq!(config.balance_rel_threshold, 2.5);
        assert_eq!(config.eviction_interval_secs, 0);
        assert_eq!(config.max_tree_size, 4096);
        assert_eq!(config.block_size, 32);
        assert_eq!(config.balance_token_usage_threshold, 0.5);
        assert_eq!(config.overload_token_usage_threshold, 0.8);

        // Aliases accepted by create_by_name inherit too.
        let alias = registry.on_worker_added("llama-4", Some("CacheAware"));
        let alias_config = alias
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .unwrap()
            .config_for_test();
        assert_eq!(alias_config.max_tree_size, 4096);
    }

    #[test]
    fn hinted_policy_inherits_operator_least_load_config() {
        let registry = PolicyRegistry::new(PolicyConfig::LeastLoad {
            load_check_interval_secs: 10,
            kv_pressure_weight: 0.4,
            mean_prefill_tokens: 2048,
            default_throughput: 555.0,
        });

        let policy = registry.on_worker_added("llama-3", Some("least_load"));
        assert!(!Arc::ptr_eq(&policy, &registry.get_default_policy()));
        let least_load = policy.as_any().downcast_ref::<LeastLoadPolicy>().unwrap();
        assert_eq!(least_load.params_for_test(), (0.4, 2048, 555.0));
    }

    /// An integration setter racing policy publication must never be
    /// missed: publication injects and inserts under the integration
    /// guards, so the setter either wrote first (its value is injected)
    /// or its propagation scan runs after the insert.
    #[test]
    fn hinted_policy_publication_races_integration_setter() {
        use std::sync::Barrier;

        use crate::worker::KvEventMonitor;

        const ADDERS: usize = 4;

        let registry = Arc::new(PolicyRegistry::new(PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 0,
            max_tree_size: 128,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
        }));

        for round in 0..64 {
            let monitor = Arc::new(KvEventMonitor::new(Some(4)));
            let barrier = Arc::new(Barrier::new(ADDERS + 1));
            let handles: Vec<_> = (0..ADDERS)
                .map(|k| {
                    let registry = Arc::clone(&registry);
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        registry
                            .on_worker_added(&format!("model-{round}-{k}"), Some("cache_aware"));
                    })
                })
                .collect();
            barrier.wait();
            registry.set_kv_event_monitor(Some(monitor));
            for handle in handles {
                handle.join().unwrap();
            }
            for entry in registry.model_policies.iter() {
                let cache_aware = entry
                    .value()
                    .as_any()
                    .downcast_ref::<CacheAwarePolicy>()
                    .unwrap();
                assert!(
                    cache_aware.kv_event_monitor_is_set_for_test(),
                    "policy for {} missed the concurrently-set monitor",
                    entry.key()
                );
            }
            registry.set_kv_event_monitor(None);
            registry.clear();
        }
    }

    #[test]
    fn hinted_policy_without_operator_config_uses_type_defaults() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let policy = registry.on_worker_added("llama-3", Some("cache_aware"));
        let config = policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .unwrap()
            .config_for_test();
        let defaults = CacheAwareConfig::default();
        assert_eq!(config.cache_threshold, defaults.cache_threshold);
        assert_eq!(config.max_tree_size, defaults.max_tree_size);
        assert_eq!(
            config.eviction_interval_secs,
            defaults.eviction_interval_secs
        );
    }

    #[test]
    fn test_pd_cache_aware_policy_initialization() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        registry.set_prefill_policy(cache_aware_policy());
        registry.set_decode_policy(cache_aware_policy());

        let prefill_workers = vec![
            worker("http://prefill-1:8000", WorkerType::Prefill),
            worker("http://prefill-2:8000", WorkerType::Prefill),
        ];
        let decode_workers = vec![
            worker("http://decode-1:8000", WorkerType::Decode),
            worker("http://decode-2:8000", WorkerType::Decode),
        ];

        registry.init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

        let prefill_policy = registry.get_prefill_policy();
        let decode_policy = registry.get_decode_policy();
        let info = SelectWorkerInfo {
            request_text: Some("shared prefix request"),
            ..Default::default()
        };

        let prefill_first = prefill_policy.select_worker(&prefill_workers, &info);
        let prefill_second = prefill_policy.select_worker(&prefill_workers, &info);
        assert!(prefill_first.is_some());
        assert_eq!(prefill_first, prefill_second);

        let decode_first = decode_policy.select_worker(&decode_workers, &info);
        let decode_second = decode_policy.select_worker(&decode_workers, &info);
        assert!(decode_first.is_some());
        assert_eq!(decode_first, decode_second);

        registry.remove_worker_from_pd_cache_aware("http://prefill-1:8000");
        registry.remove_worker_from_pd_cache_aware("http://decode-1:8000");
    }
}
