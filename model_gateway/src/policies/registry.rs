use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use dashmap::DashMap;
use http::{header::HeaderName, HeaderMap};
use parking_lot::RwLock;
use tracing::{debug, info, warn};

/// Policy Registry for managing model-to-policy mappings
///
/// This registry manages the dynamic assignment of load balancing policies to models.
/// When the first worker of a new model is added, it determines the policy for that model.
/// All subsequent workers of the same model use the established policy.
/// When the last worker of a model is removed, the policy mapping is cleaned up.
use super::{
    get_healthy_worker_indices,
    manual::{ExecutionBranch, PinState},
    BucketPolicy, CacheAwareLengthPolicy, CacheAwarePolicy, DPRankLoadPolicy, LoadBalancingPolicy,
    ManualConfig, ManualPolicy, PolicyFactory, SelectWorkerInfo, WorkerLeg,
};
use crate::{
    config::types::{ManualAssignmentMode, PolicyConfig, RoutingKeyOverrideConfig},
    mesh::adapters::TreeSyncAdapter,
    observability::metrics::Metrics,
    policies::cache_aware::LoadReceiver,
    routers::common::header_utils::{
        extract_routing_key_hint_named, parse_routing_tokens_hint, ROUTING_KEY_HINT_MAX_BYTES,
    },
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

    /// Shared sticky selector for the routing-key override. `Some` when the
    /// override is enabled; consulted (instead of the configured policy) for keyed
    /// requests via [`PolicyRegistry::select_worker`].
    routing_key_sticky: Option<Arc<ManualPolicy>>,

    /// Ordered routing-key header names, parsed once from
    /// `routing_key_override.headers`; the first header present with a valid
    /// value wins.
    routing_key_headers: Arc<Vec<HeaderName>>,
}

/// A sticky key with this many of its own requests already in flight on its
/// pinned worker (router-local, counted per replica) is bypassed and the
/// request reassigned. Guards one conversation stacking onto its pin; total
/// worker load is not a respill trigger.
const STICKY_INFLIGHT_CAP: usize = 2;

/// `conv_t2_r1` -> `conv`: strip one trailing `_r<n>` retry suffix, then one
/// trailing `_t<n>` turn suffix. A bare retry suffix is stripped too: a retry
/// shares identity with its original regardless of turn structure.
fn strip_lineage_suffixes(rid: &str) -> &str {
    strip_suffix_tag(strip_suffix_tag(rid, 'r'), 't')
}

fn strip_suffix_tag(s: &str, tag: char) -> &str {
    let Some((head, tail)) = s.rsplit_once('_') else {
        return s;
    };
    let digits = match tail.strip_prefix(tag) {
        Some(d) => d,
        None => return s,
    };
    if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
        head
    } else {
        s
    }
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
        // ConfigValidator rejects invalid names at startup; skipping here
        // covers direct construction.
        let routing_key_headers = routing_key_override
            .headers
            .iter()
            .filter_map(|name| match HeaderName::try_from(name.as_str()) {
                Ok(parsed) => Some(parsed),
                Err(_) => {
                    warn!("Ignoring invalid routing-key header name: {name:?}");
                    None
                }
            })
            .collect();
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
            routing_key_headers: Arc::new(routing_key_headers),
        }
    }

    /// Derive the session key from a request id: trailing `_r<n>` retry and
    /// `_t<n>` turn suffixes are stripped so every turn of a conversation
    /// shares one key. `None` when the override is disabled, there is no rid,
    /// or the key exceeds the routing-key byte cap; a rid that is nothing but
    /// suffix keys as itself.
    pub fn derive_rid_key<'a>(&self, rid: Option<&'a str>) -> Option<&'a str> {
        self.routing_key_sticky.as_ref()?;
        let rid = rid?;
        let stripped = strip_lineage_suffixes(rid);
        let key = if stripped.is_empty() { rid } else { stripped };
        (!key.is_empty() && key.len() <= ROUTING_KEY_HINT_MAX_BYTES).then_some(key)
    }

    /// Resolve the routing key from the configured header names: the first
    /// header present with a valid value (non-empty UTF-8 within the byte
    /// cap) wins.
    pub fn resolve_routing_key<'a>(&self, headers: Option<&'a HeaderMap>) -> Option<&'a str> {
        self.routing_key_headers
            .iter()
            .find_map(|name| extract_routing_key_hint_named(headers, name))
    }

    /// A key that is nothing but suffix keys as itself, like the rid path.
    fn strip_header_key(raw: &str) -> &str {
        let stripped = strip_lineage_suffixes(raw);
        if stripped.is_empty() {
            raw
        } else {
            stripped
        }
    }

    /// The header-derived sticky key: the resolved routing key,
    /// lineage-stripped when the override is enabled (matching selection),
    /// raw otherwise.
    pub fn sticky_header_key<'a>(&self, headers: Option<&'a HeaderMap>) -> Option<&'a str> {
        let raw = self.resolve_routing_key(headers)?;
        Some(if self.routing_key_sticky.is_some() {
            Self::strip_header_key(raw)
        } else {
            raw
        })
    }

    /// Resolve the effective sticky key: the rid-derived key wins, the
    /// configured routing-key headers are the fallback when no rid is
    /// present. Header keys get the same lineage stripping as rid keys, so a
    /// proxy forwarding `conv_t2` as a header pins the entry `conv`.
    fn effective_sticky_key<'a>(
        &self,
        info: &SelectWorkerInfo<'a>,
    ) -> Option<(&'a str, &'static str)> {
        if let Some(key) = info.rid_key {
            return Some((key, "rid"));
        }
        let raw = info
            .routing_key
            .or_else(|| self.resolve_routing_key(info.headers))?;
        Some((Self::strip_header_key(raw), "header"))
    }

    /// Select a worker, applying the sticky routing-key override when it is
    /// enabled, the request carries a key from the configured source, and the
    /// configured policy does not already honor the key (`manual` /
    /// `consistent_hashing`). Otherwise delegates to `policy`. `policy.name()`
    /// stays the real policy (for metrics).
    pub fn select_worker(
        &self,
        policy: &Arc<dyn LoadBalancingPolicy>,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        if let Some(sticky) = self.routing_key_sticky.as_ref() {
            if Self::routing_key_override_applies(policy.name()) {
                if let Some((key, source)) = self.effective_sticky_key(info) {
                    return Self::select_sticky(sticky, policy, workers, info, key, source);
                }
            }
        }
        policy.select_worker(workers, info)
    }

    /// Keyed selection: honor an existing pin under the in-flight cap;
    /// otherwise assign — via the underlying policy in `delegate` mode, via
    /// the sticky map's own assignment mode otherwise — and pin the result.
    fn select_sticky(
        sticky: &Arc<ManualPolicy>,
        policy: &Arc<dyn LoadBalancingPolicy>,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        key: &str,
        source: &'static str,
    ) -> Option<usize> {
        Metrics::record_routing_key_source(source);

        // Keyed-load guards track the un-namespaced key on each worker.
        let load_key = key;

        // PD legs namespace so prefill and decode stick independently.
        let namespaced;
        let key = if info.leg == WorkerLeg::Single {
            key
        } else {
            namespaced = format!("{}{}", info.leg.routing_id_prefix(), key);
            &namespaced
        };

        let over_cap =
            |idx: usize| workers[idx].routing_key_inflight(load_key) >= STICKY_INFLIGHT_CAP;
        let finish = |result: Option<usize>, branch: ExecutionBranch| {
            Metrics::record_worker_manual_policy_branch(branch.as_str());
            Metrics::set_manual_policy_cache_entries(sticky.map_len());
            debug!(
                source,
                key,
                branch = branch.as_str(),
                worker = result.map_or("none", |idx| workers[idx].url()),
                model_id = result.map_or("none", |idx| workers[idx].model_id()),
                "Sticky routing decision"
            );
            result
        };

        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return finish(None, ExecutionBranch::NoHealthyWorkers);
        }
        let delegate = sticky.assignment_mode() == ManualAssignmentMode::Delegate;

        let pin = sticky.peek_pin(workers, key, &healthy);
        if let PinState::Pinned(idx) = pin {
            if !over_cap(idx) {
                return finish(Some(idx), ExecutionBranch::OccupiedHit);
            }
            // Over the cap: reassign, but re-pin only on a strict improvement.
            // Re-picking the pinned worker (a prefix-affine policy often will)
            // or another saturated worker keeps the pin, so fleet-wide
            // pressure cannot random-walk it off the prefix owner.
            let new_idx = if delegate {
                match policy.select_worker(workers, info) {
                    Some(idx) => idx,
                    None => return finish(None, ExecutionBranch::CapRespill),
                }
            } else {
                sticky.assign_index(workers, &healthy)
            };
            if new_idx != idx && !over_cap(new_idx) {
                sticky.pin_front(key, workers[new_idx].url());
            }
            return finish(Some(new_idx), ExecutionBranch::CapRespill);
        }

        if delegate {
            let branch = match pin {
                PinState::Stale => ExecutionBranch::OccupiedMiss,
                _ => ExecutionBranch::Vacant,
            };
            let idx = match policy.select_worker(workers, info) {
                Some(idx) => idx,
                None => return finish(None, branch),
            };
            sticky.pin_front(key, workers[idx].url());
            return finish(Some(idx), branch);
        }

        let (idx, branch) = sticky.assign_pin(workers, key, &healthy);
        finish(Some(idx), branch)
    }

    /// Policies that already honor the routing key keep their own handling; all
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

    /// Whether any registered policy (default or per-model) routes on
    /// request text this request cannot supply. Content-blind dispatch must
    /// stay buffered when one does: the model inside the unread body could
    /// select that policy. A routing hint lifts the requirement — with a
    /// valid `x-smg-routing-tokens` the buffered path never extracts text
    /// either (the hint wins over body-derived routing), and a valid
    /// routing-key header under the sticky override supersedes the policy
    /// before it reads text. Hints are validated by the same extractors
    /// selection uses, so malformed or over-cap values lift nothing.
    pub fn any_policy_needs_request_text(&self, headers: Option<&HeaderMap>) -> bool {
        if parse_routing_tokens_hint(headers).is_some() {
            return false;
        }
        let keyed_override =
            self.routing_key_sticky.is_some() && self.resolve_routing_key(headers).is_some();
        let needs_text = |policy: &Arc<dyn LoadBalancingPolicy>| {
            policy.needs_request_text()
                && !(keyed_override && Self::routing_key_override_applies(policy.name()))
        };
        needs_text(&self.default_policy)
            || self
                .model_policies
                .iter()
                .any(|entry| needs_text(entry.value()))
    }

    /// Get all load-aware policies that need periodic load updates (lock-free).
    ///
    /// Membership is policy-reported via
    /// [`LoadBalancingPolicy::needs_backend_loads`]: `power_of_two` and
    /// `least_load` always consume load reports; `cache_aware` does when its
    /// pressure knobs are configured.
    pub fn get_all_load_aware_policies(&self) -> Vec<Arc<dyn LoadBalancingPolicy>> {
        let mut policies = Vec::new();

        if self.default_policy.needs_backend_loads() {
            policies.push(Arc::clone(&self.default_policy));
        }

        // Get prefill, decode, and encode policies (lock-free via OnceLock::get)
        let prefill_policy_opt = self.prefill_policy.get();
        let decode_policy_opt = self.decode_policy.get();
        let encode_policy_opt = self.encode_policy.get();

        if let Some(policy) = prefill_policy_opt {
            if policy.needs_backend_loads() && !Arc::ptr_eq(policy, &self.default_policy) {
                policies.push(Arc::clone(policy));
            }
        }

        if let Some(policy) = decode_policy_opt {
            if policy.needs_backend_loads()
                && !Arc::ptr_eq(policy, &self.default_policy)
                && !prefill_policy_opt.is_some_and(|p| Arc::ptr_eq(p, policy))
            {
                policies.push(Arc::clone(policy));
            }
        }

        if let Some(policy) = encode_policy_opt {
            if policy.needs_backend_loads()
                && !Arc::ptr_eq(policy, &self.default_policy)
                && !prefill_policy_opt.is_some_and(|p| Arc::ptr_eq(p, policy))
                && !decode_policy_opt.is_some_and(|p| Arc::ptr_eq(p, policy))
            {
                policies.push(Arc::clone(policy));
            }
        }

        for entry in self.model_policies.iter() {
            let policy = entry.value();
            if policy.needs_backend_loads() {
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
            } else if policy.name() == "cache_aware_length" {
                if let Some(cal) = policy.as_any().downcast_ref::<CacheAwareLengthPolicy>() {
                    debug!(
                        "Initializing cache_aware_length policy with {} workers for model {}",
                        workers.len(),
                        model_id
                    );
                    cal.init_workers(workers);
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
            } else if policy.name() == "cache_aware_length" {
                if let Some(cal) = policy.as_any().downcast_ref::<CacheAwareLengthPolicy>() {
                    cal.remove_worker_by_url(worker_url);
                    debug!(
                        "Removed worker {} from cache_aware_length policy for model {}",
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
                } else if policy.name() == "cache_aware_length" {
                    if let Some(cal) = policy.as_any().downcast_ref::<CacheAwareLengthPolicy>() {
                        cal.remove_worker_by_url(worker_url);
                        debug!(
                            "Removed worker {} from {} cache_aware_length policy",
                            worker_url, worker_type
                        );
                    }
                }
            }
        }
    }

    /// Drop a removed worker's cached load report from all load-aware
    /// policies.
    ///
    /// Push-model policies cache per-worker load reports keyed by URL;
    /// without this their caches would grow unbounded under worker churn.
    /// Called on worker removal alongside the cache-aware cleanup above.
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
            } else if prefill_policy.name() == "cache_aware_length" {
                if let Some(cal) = prefill_policy
                    .as_any()
                    .downcast_ref::<CacheAwareLengthPolicy>()
                {
                    if !prefill_workers.is_empty() {
                        debug!(
                            "Initializing prefill cache_aware_length policy with {} workers",
                            prefill_workers.len()
                        );
                        cal.init_workers(prefill_workers);
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
            } else if decode_policy.name() == "cache_aware_length" {
                if let Some(cal) = decode_policy
                    .as_any()
                    .downcast_ref::<CacheAwareLengthPolicy>()
                {
                    if !decode_workers.is_empty() {
                        debug!(
                            "Initializing decode cache_aware_length policy with {} workers",
                            decode_workers.len()
                        );
                        cal.init_workers(decode_workers);
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
    use tracing_test::traced_test;

    use super::*;
    use crate::{
        policies::{CacheAwareConfig, LeastLoadPolicy, SelectWorkerInfo},
        worker::{BasicWorkerBuilder, Worker, WorkerLoadGuard, WorkerType},
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

    fn headers_with_key(key: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
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

    fn cache_aware_config() -> PolicyConfig {
        PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 4096,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
        }
    }

    fn headers_with_tokens(value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("x-smg-routing-tokens", value.parse().unwrap());
        h
    }

    fn enabled_override() -> RoutingKeyOverrideConfig {
        RoutingKeyOverrideConfig {
            enabled: true,
            ..Default::default()
        }
    }

    #[test]
    fn request_text_scan_is_false_for_text_free_policies() {
        let reg = PolicyRegistry::new(PolicyConfig::RoundRobin);
        assert!(!reg.any_policy_needs_request_text(None));
        assert!(!reg.any_policy_needs_request_text(Some(&headers_with_key("k"))));
    }

    #[test]
    fn tokens_hint_lifts_text_requirement_only_when_valid() {
        let reg = PolicyRegistry::new(cache_aware_config());
        assert!(reg.any_policy_needs_request_text(None));
        assert!(!reg.any_policy_needs_request_text(Some(&headers_with_tokens("1,2,3"))));
        // Malformed and over-cap hints are ignored, exactly like selection.
        assert!(reg.any_policy_needs_request_text(Some(&headers_with_tokens("1,,3"))));
        let over_cap = vec!["7"; 513].join(",");
        assert!(reg.any_policy_needs_request_text(Some(&headers_with_tokens(&over_cap))));
    }

    #[test]
    fn key_hint_lifts_text_requirement_only_under_enabled_override() {
        let without = PolicyRegistry::new(cache_aware_config());
        assert!(without.any_policy_needs_request_text(Some(&headers_with_key("session-A"))));

        let with = PolicyRegistry::with_override(cache_aware_config(), enabled_override());
        assert!(with.any_policy_needs_request_text(None));
        assert!(!with.any_policy_needs_request_text(Some(&headers_with_key("session-A"))));
        let over_cap = "k".repeat(129);
        assert!(with.any_policy_needs_request_text(Some(&headers_with_key(&over_cap))));
    }

    #[test]
    fn per_model_text_policy_scanned_with_the_same_hint_rules() {
        let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
        assert!(!reg.any_policy_needs_request_text(None));
        reg.on_worker_added("llama-3", Some("cache_aware"));
        assert!(reg.any_policy_needs_request_text(None));
        assert!(!reg.any_policy_needs_request_text(Some(&headers_with_tokens("1,2,3"))));
        assert!(!reg.any_policy_needs_request_text(Some(&headers_with_key("session-A"))));
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

    fn override_with_headers(names: &[&str]) -> RoutingKeyOverrideConfig {
        RoutingKeyOverrideConfig {
            enabled: true,
            headers: names.iter().map(|n| n.to_string()).collect(),
            ..Default::default()
        }
    }

    fn headers_with(name: &str, key: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(HeaderName::try_from(name).unwrap(), key.parse().unwrap());
        h
    }

    #[test]
    fn resolve_routing_key_first_present_and_valid_wins() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            override_with_headers(&["x-routing-key", "x-smg-routing-key"]),
        );

        let mut both = headers_with("x-routing-key", "primary");
        both.insert("x-smg-routing-key", "legacy".parse().unwrap());
        assert_eq!(reg.resolve_routing_key(Some(&both)), Some("primary"));

        let legacy_only = headers_with_key("legacy");
        assert_eq!(reg.resolve_routing_key(Some(&legacy_only)), Some("legacy"));

        // An invalid value under the first name falls through to the next.
        let mut over_cap_first = headers_with("x-routing-key", &"k".repeat(129));
        over_cap_first.insert("x-smg-routing-key", "legacy".parse().unwrap());
        assert_eq!(
            reg.resolve_routing_key(Some(&over_cap_first)),
            Some("legacy")
        );

        assert_eq!(reg.resolve_routing_key(None), None);
    }

    #[test]
    fn default_headers_ignore_unconfigured_names() {
        let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
        assert_eq!(
            reg.resolve_routing_key(Some(&headers_with("x-routing-key", "primary"))),
            None
        );
        assert_eq!(
            reg.resolve_routing_key(Some(&headers_with_key("legacy"))),
            Some("legacy")
        );
    }

    #[test]
    fn invalid_configured_header_names_are_skipped() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            override_with_headers(&["not a header", "x-routing-key"]),
        );
        assert_eq!(
            reg.resolve_routing_key(Some(&headers_with("x-routing-key", "primary"))),
            Some("primary")
        );
    }

    #[test]
    fn header_keys_share_rid_lineage_stripping() {
        let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
        assert_eq!(
            reg.sticky_header_key(Some(&headers_with_key("conv_t2"))),
            Some("conv")
        );
        assert_eq!(
            reg.sticky_header_key(Some(&headers_with_key("conv_t2_r1"))),
            Some("conv")
        );
        // A key that is nothing but suffix keys as itself.
        assert_eq!(
            reg.sticky_header_key(Some(&headers_with_key("_t1"))),
            Some("_t1")
        );

        // Override disabled: raw key, no stripping.
        let disabled = PolicyRegistry::new(PolicyConfig::RoundRobin);
        assert_eq!(
            disabled.sticky_header_key(Some(&headers_with_key("conv_t2"))),
            Some("conv_t2")
        );
    }

    #[test]
    fn header_turns_pin_the_same_sticky_entry() {
        let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
            worker("http://w3", WorkerType::Regular),
        ];
        let t1 = headers_with_key("conv_t1");
        let info_t1 = SelectWorkerInfo {
            headers: Some(&t1),
            ..Default::default()
        };
        let first = reg.select_worker(&policy, &workers, &info_t1).unwrap();
        let t2 = headers_with_key("conv_t2");
        let info_t2 = SelectWorkerInfo {
            headers: Some(&t2),
            ..Default::default()
        };
        for _ in 0..5 {
            assert_eq!(reg.select_worker(&policy, &workers, &info_t2), Some(first));
        }
    }

    #[test]
    fn alternate_header_routes_stickily() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            override_with_headers(&["x-routing-key", "x-smg-routing-key"]),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
            worker("http://w3", WorkerType::Regular),
        ];
        let headers = headers_with("x-routing-key", "session-A");
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
    fn alternate_header_lifts_text_gate() {
        let reg = PolicyRegistry::with_override(
            cache_aware_config(),
            override_with_headers(&["x-routing-key", "x-smg-routing-key"]),
        );
        assert!(
            !reg.any_policy_needs_request_text(Some(&headers_with("x-routing-key", "session-A")))
        );
        // A name outside the configured list lifts nothing.
        assert!(reg.any_policy_needs_request_text(Some(&headers_with("x-other-key", "session-A"))));
    }

    fn rid_override(assignment_mode: ManualAssignmentMode) -> RoutingKeyOverrideConfig {
        RoutingKeyOverrideConfig {
            enabled: true,
            assignment_mode,
            ..Default::default()
        }
    }

    #[test]
    fn derive_rid_key_strips_turn_and_retry_suffixes() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        assert_eq!(reg.derive_rid_key(Some("conv123_t1")), Some("conv123"));
        assert_eq!(reg.derive_rid_key(Some("conv123_t12")), Some("conv123"));
        assert_eq!(reg.derive_rid_key(Some("conv123_t2_r1")), Some("conv123"));
        assert_eq!(reg.derive_rid_key(Some("conv123_r1")), Some("conv123"));
        assert_eq!(reg.derive_rid_key(Some("conv_t1_t2")), Some("conv_t1"));
        assert_eq!(
            reg.derive_rid_key(Some("under_scored_id")),
            Some("under_scored_id")
        );
        assert_eq!(reg.derive_rid_key(Some("no-suffix")), Some("no-suffix"));
        assert_eq!(reg.derive_rid_key(Some("conv_tx1")), Some("conv_tx1"));
        assert_eq!(reg.derive_rid_key(Some("conv_t")), Some("conv_t"));
        // A rid that is nothing but suffix keys as itself.
        assert_eq!(reg.derive_rid_key(Some("_t1")), Some("_t1"));
        assert_eq!(reg.derive_rid_key(None), None);
        let over_cap = format!("{}_t1", "k".repeat(200));
        assert_eq!(reg.derive_rid_key(Some(&over_cap)), None);

        // Override disabled: rid never yields a key.
        let disabled = PolicyRegistry::new(PolicyConfig::RoundRobin);
        assert_eq!(disabled.derive_rid_key(Some("conv123_t1")), None);
    }

    #[test]
    fn rid_key_wins_over_header_and_header_is_fallback() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];

        // Unique per-request header keys must not fragment the rid pin.
        let poison_a = headers_with_key("req-aaa");
        let poison_b = headers_with_key("req-bbb");
        let first = reg
            .select_worker(
                &policy,
                &workers,
                &SelectWorkerInfo {
                    headers: Some(&poison_a),
                    rid_key: Some("conv42"),
                    ..Default::default()
                },
            )
            .unwrap();
        for headers in [&poison_b, &poison_a] {
            assert_eq!(
                reg.select_worker(
                    &policy,
                    &workers,
                    &SelectWorkerInfo {
                        headers: Some(headers),
                        rid_key: Some("conv42"),
                        ..Default::default()
                    },
                ),
                Some(first)
            );
        }

        // No rid: the header key gets its own stable pin (fallback works).
        let session = headers_with_key("session-H");
        let header_info = SelectWorkerInfo {
            headers: Some(&session),
            ..Default::default()
        };
        let pinned = reg.select_worker(&policy, &workers, &header_info).unwrap();
        assert_eq!(
            reg.select_worker(&policy, &workers, &header_info),
            Some(pinned)
        );
    }

    #[test]
    fn delegate_assignment_routes_via_underlying_policy_then_pins() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 0,
                max_tree_size: 10000,
                block_size: 16,
                balance_token_usage_threshold: 1.0,
                overload_token_usage_threshold: 1.0,
                overlap_decay: 0.0,
                selection_temperature: 0.0,
                cache_index: Default::default(),
                cache_ttl_secs: 180,
                cache_boundaries: Vec::new(),
            },
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .unwrap()
            .init_workers(&workers);

        let turn1_text = "conversation opening with a long shared instruction block";
        let seeded = reg
            .select_worker(
                &policy,
                &workers,
                &SelectWorkerInfo {
                    request_text: Some(turn1_text),
                    ..Default::default()
                },
            )
            .unwrap();

        // First keyed request delegates to the policy: the tree match must
        // send it to the seeded worker, not a random pick.
        let key = reg.derive_rid_key(Some("conv42_t1")).unwrap();
        let keyed_turn1 = SelectWorkerInfo {
            request_text: Some(turn1_text),
            rid_key: Some(key),
            ..Default::default()
        };
        assert_eq!(
            reg.select_worker(&policy, &workers, &keyed_turn1),
            Some(seeded)
        );

        // The follow-up shares no routable text; only the pin can send it
        // back. The policy alone would pick the other worker (its tie-break
        // counts prior selections), so equality proves the pin decided.
        let keyed_turn2 = SelectWorkerInfo {
            request_text: Some("unrelated follow-up tail"),
            rid_key: reg.derive_rid_key(Some("conv42_t2")),
            ..Default::default()
        };
        assert_eq!(
            reg.select_worker(&policy, &workers, &keyed_turn2),
            Some(seeded)
        );
    }

    #[test]
    #[traced_test]
    fn sticky_selection_emits_decision_line_per_request() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let info = SelectWorkerInfo {
            rid_key: Some("conv1"),
            ..Default::default()
        };
        reg.select_worker(&policy, &workers, &info).unwrap();
        assert!(logs_contain("Sticky routing decision"));
        assert!(logs_contain("vacant"));
        reg.select_worker(&policy, &workers, &info).unwrap();
        assert!(logs_contain("occupied_hit"));
    }

    /// An overloaded pin must respill through the existing reassignment path.
    ///
    /// Every router hands `select_worker` a slice already filtered by
    /// `is_available()`, so a vetoed worker is simply absent from the
    /// candidates — the pin resolves to nothing and takes the same
    /// `occupied_miss` respill a stale pin takes. The test mirrors that call
    /// shape rather than passing a raw pool no router builds.
    #[test]
    #[traced_test]
    fn overloaded_sticky_pin_respills_through_the_stale_path() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let fleet = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let available = |fleet: &[Arc<dyn Worker>]| -> Vec<Arc<dyn Worker>> {
            fleet.iter().filter(|w| w.is_available()).cloned().collect()
        };
        let info = SelectWorkerInfo {
            rid_key: Some("convOverload"),
            ..Default::default()
        };

        let workers = available(&fleet);
        let pinned_url = {
            let idx = reg.select_worker(&policy, &workers, &info).unwrap();
            assert_eq!(
                reg.select_worker(&policy, &workers, &info),
                Some(idx),
                "the pin holds while the worker is eligible"
            );
            workers[idx].url().to_string()
        };

        fleet
            .iter()
            .find(|w| w.url() == pinned_url)
            .unwrap()
            .set_overloaded(true);
        let workers = available(&fleet);
        assert_eq!(workers.len(), 1, "the veto removes the pinned worker");
        let respilled = reg.select_worker(&policy, &workers, &info).unwrap();
        assert_ne!(
            workers[respilled].url(),
            pinned_url,
            "an overloaded pin must be abandoned"
        );
        assert!(logs_contain("occupied_miss"));

        // Recovery re-admits the worker as a candidate; the pin now follows the
        // respill target, so only eligibility is asserted here.
        fleet
            .iter()
            .find(|w| w.url() == pinned_url)
            .unwrap()
            .set_overloaded(false);
        let workers = available(&fleet);
        assert!(reg.select_worker(&policy, &workers, &info).is_some());
    }

    /// Every worker vetoed leaves the sticky selector with nothing to pin.
    #[test]
    #[traced_test]
    fn all_overloaded_leaves_sticky_selection_empty() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        for w in &workers {
            w.set_overloaded(true);
        }
        let info = SelectWorkerInfo {
            rid_key: Some("convShed"),
            ..Default::default()
        };

        assert_eq!(reg.select_worker(&policy, &workers, &info), None);
        assert!(logs_contain("no_healthy_workers"));
    }

    #[test]
    fn cap_respill_reassigns_and_pin_follows_strict_improvement() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let info = SelectWorkerInfo {
            rid_key: Some("convA"),
            ..Default::default()
        };

        let pinned = reg.select_worker(&policy, &workers, &info).unwrap();
        assert_eq!(reg.select_worker(&policy, &workers, &info), Some(pinned));

        // At the cap (2 of this key in flight on its pin) the key reassigns
        // to the idle worker and the pin follows it, even after the original
        // drains.
        let guards = [
            WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convA")),
            WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convA")),
        ];
        let respilled = reg.select_worker(&policy, &workers, &info).unwrap();
        assert_ne!(respilled, pinned);
        drop(guards);
        assert_eq!(reg.select_worker(&policy, &workers, &info), Some(respilled));
    }

    #[test]
    fn cap_respill_keeps_pin_when_no_strictly_better_pick() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let info = SelectWorkerInfo {
            rid_key: Some("convB"),
            ..Default::default()
        };

        let pinned = reg.select_worker(&policy, &workers, &info).unwrap();
        // Saturate BOTH workers with this key: the reassignment cannot
        // strictly improve (every pick is at the key cap), so the request
        // dispatches wherever the policy says but the pin must survive.
        let mut guards = Vec::new();
        for w in &workers {
            guards.push(WorkerLoadGuard::with_key(w.clone(), Some("convB")));
            guards.push(WorkerLoadGuard::with_key(w.clone(), Some("convB")));
        }
        for _ in 0..4 {
            reg.select_worker(&policy, &workers, &info).unwrap();
        }
        drop(guards);
        assert_eq!(
            reg.select_worker(&policy, &workers, &info),
            Some(pinned),
            "pin must not move to an equally saturated worker"
        );
    }

    #[test]
    fn cap_respill_under_legacy_assignment() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::MinLoad),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let info = SelectWorkerInfo {
            rid_key: Some("convC"),
            ..Default::default()
        };

        let pinned = reg.select_worker(&policy, &workers, &info).unwrap();
        let guards = [
            WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convC")),
            WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convC")),
        ];
        let respilled = reg.select_worker(&policy, &workers, &info).unwrap();
        assert_ne!(respilled, pinned);
        drop(guards);
        assert_eq!(reg.select_worker(&policy, &workers, &info), Some(respilled));
    }

    #[test]
    #[traced_test]
    fn busy_worker_does_not_respill_idle_key() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let info = SelectWorkerInfo {
            rid_key: Some("convD"),
            ..Default::default()
        };

        let pinned = reg.select_worker(&policy, &workers, &info).unwrap();
        // Unrelated traffic on the pin, total load far past the cap: the
        // returning key has nothing in flight and must keep its pin.
        let _unrelated = [
            WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convZ")),
            WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convZ")),
            WorkerLoadGuard::new(workers[pinned].clone(), None),
            WorkerLoadGuard::new(workers[pinned].clone(), None),
        ];
        assert_eq!(reg.select_worker(&policy, &workers, &info), Some(pinned));
        assert!(logs_contain("occupied_hit"));
        assert!(!logs_contain("cap_respill"));
    }

    #[test]
    #[traced_test]
    fn two_inflight_same_key_respill_the_third() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let info = SelectWorkerInfo {
            rid_key: Some("convE"),
            ..Default::default()
        };

        let pinned = reg.select_worker(&policy, &workers, &info).unwrap();
        let _g1 = WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convE"));
        let _g2 = WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convE"));
        assert_ne!(reg.select_worker(&policy, &workers, &info), Some(pinned));
        assert!(logs_contain("cap_respill"));
    }

    #[test]
    #[traced_test]
    fn cap_lifts_when_key_inflight_drains() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        // Single worker: the respill has nowhere to go, so the branch flips
        // purely on the key's in-flight count.
        let workers = vec![worker("http://w1", WorkerType::Regular)];
        let info = SelectWorkerInfo {
            rid_key: Some("convF"),
            ..Default::default()
        };

        assert_eq!(reg.select_worker(&policy, &workers, &info), Some(0));
        let guards = [
            WorkerLoadGuard::with_key(workers[0].clone(), Some("convF")),
            WorkerLoadGuard::with_key(workers[0].clone(), Some("convF")),
        ];
        assert_eq!(reg.select_worker(&policy, &workers, &info), Some(0));
        assert!(logs_contain("cap_respill"));
        drop(guards);
        assert_eq!(reg.select_worker(&policy, &workers, &info), Some(0));
        assert!(logs_contain("occupied_hit"));
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
    fn cache_aware_is_load_aware_only_when_pressure_configured() {
        fn cache_aware(
            overlap_decay: f32,
            balance_token_usage_threshold: f32,
            overload_token_usage_threshold: f32,
        ) -> PolicyConfig {
            PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 0,
                max_tree_size: 4096,
                block_size: 16,
                balance_token_usage_threshold,
                overload_token_usage_threshold,
                overlap_decay,
                selection_temperature: 0.0,
                cache_index: Default::default(),
                cache_ttl_secs: 180,
                cache_boundaries: Vec::new(),
            }
        }

        // Plain cache_aware consumes no backend loads — the monitor must
        // keep skipping load polling for it.
        let plain = PolicyRegistry::new(cache_aware(0.0, 1.0, 1.0));
        assert!(plain.get_all_load_aware_policies().is_empty());

        // Any pressure knob makes it load-aware: waiting-prefill decay,
        // KV-usage balance spread, or the KV-usage overload ceiling.
        for pressured in [
            cache_aware(1.0, 1.0, 1.0),
            cache_aware(0.0, 0.5, 1.0),
            cache_aware(0.0, 1.0, 0.9),
        ] {
            let registry = PolicyRegistry::new(pressured);
            assert_eq!(registry.get_all_load_aware_policies().len(), 1);
        }
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
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
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
            max_waiting_requests: 48,
        });

        let policy = registry.on_worker_added("llama-3", Some("least_load"));
        assert!(!Arc::ptr_eq(&policy, &registry.get_default_policy()));
        let least_load = policy.as_any().downcast_ref::<LeastLoadPolicy>().unwrap();
        assert_eq!(least_load.params_for_test(), (0.4, 2048, 555.0, 48));
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
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
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
