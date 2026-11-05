use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc, OnceLock, Weak,
    },
};

use dashmap::DashMap;
use http::{header::HeaderName, HeaderMap};
use parking_lot::{Mutex, RwLock};
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
    normalize_model_key, BucketPolicy, CacheAwarePolicy, DPRankLoadPolicy, LoadBalancingPolicy,
    ManualConfig, ManualPolicy, PolicyFactory, SelectWorkerInfo, WorkerRole,
};
use crate::{
    config::types::{ManualAssignmentMode, PdPairingMode, PolicyConfig, RoutingKeyOverrideConfig},
    mesh::adapters::TreeSyncAdapter,
    observability::{cache_trace, metrics::Metrics},
    routers::common::header_utils::{
        extract_routing_key_hint_named, parse_routing_tokens_hint, ROUTING_KEY_HINT_MAX_BYTES,
    },
    worker::{load_state::LoadReceiver, KvEventMonitor, RequestCompletionSink, Worker, WorkerType},
};

/// Routes a request completion to the policy that placed the request: the
/// prefill, decode or encode policy by worker type, else the model's policy.
/// Holds the registry weakly so a worker outliving its registry reports to
/// nobody instead of keeping the registry alive.
#[derive(Debug)]
struct PolicyCompletionSink {
    registry: Weak<PolicyRegistry>,
}

impl RequestCompletionSink for PolicyCompletionSink {
    fn request_completed(&self, worker: &dyn Worker) {
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        registry
            .policy_for_worker(worker)
            .on_request_complete(worker.url(), true);
    }
}

/// What the policies a registry answers the text questions for (its default
/// and its per-model policies) say, as four bits: each of the two questions
/// ([`LoadBalancingPolicy::needs_request_text`] and the with-tokens form)
/// over every policy, and over the policies the routing-key override does
/// not supersede, which is what a request with a valid routing key asks. A
/// policy's answers are constants, so the bits change only where a policy is
/// added or removed, and a request reads one byte instead of walking the
/// per-model map.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TextNeeds(u8);

impl TextNeeds {
    const TEXT: u8 = 1;
    const TEXT_UNDER_KEY: u8 = 1 << 1;
    const TEXT_WITH_TOKENS: u8 = 1 << 2;
    const TEXT_WITH_TOKENS_UNDER_KEY: u8 = 1 << 3;

    /// What `policy` answers.
    fn of(policy: &dyn LoadBalancingPolicy) -> Self {
        let keeps_text_under_key = !PolicyRegistry::routing_key_override_applies(policy.name());
        let mut bits = 0;
        for (wants, any, under_key) in [
            (
                policy.needs_request_text(),
                Self::TEXT,
                Self::TEXT_UNDER_KEY,
            ),
            (
                policy.needs_request_text_with_tokens(),
                Self::TEXT_WITH_TOKENS,
                Self::TEXT_WITH_TOKENS_UNDER_KEY,
            ),
        ] {
            if wants {
                bits |= any;
                if keeps_text_under_key {
                    bits |= under_key;
                }
            }
        }
        Self(bits)
    }

    fn join(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether some policy wants the text of a request that carries token
    /// ids (`with_tokens`) or not, under a valid routing key
    /// (`keyed_override`) or without one.
    fn answer(self, with_tokens: bool, keyed_override: bool) -> bool {
        let bit = match (with_tokens, keyed_override) {
            (false, false) => Self::TEXT,
            (false, true) => Self::TEXT_UNDER_KEY,
            (true, false) => Self::TEXT_WITH_TOKENS,
            (true, true) => Self::TEXT_WITH_TOKENS_UNDER_KEY,
        };
        self.0 & bit != 0
    }
}

/// Registry for managing model-to-policy mappings
#[derive(Clone)]
pub struct PolicyRegistry {
    /// Model ID -> Policy instance mapping (lock-free reads via DashMap)
    model_policies: Arc<DashMap<String, Arc<dyn LoadBalancingPolicy>>>,

    /// The text questions' answers over the default and the per-model
    /// policies ([`TextNeeds`]), recomputed where one is added or removed
    /// ([`Self::refresh_text_needs`]).
    text_needs: Arc<AtomicU8>,

    /// Serializes the recomputations of `text_needs`: a removal's walk that
    /// started before a concurrent addition's insert must not publish over
    /// the bits that addition published.
    text_needs_refresh: Arc<Mutex<()>>,

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
    /// requests via [`PolicyRegistry::select_worker_for_model`].
    routing_key_sticky: Option<Arc<ManualPolicy>>,

    /// Ordered routing-key header names, parsed once from
    /// `routing_key_override.headers`; the first header present with a valid
    /// value wins.
    routing_key_headers: Arc<Vec<HeaderName>>,
    /// How strictly PD placement pairs prefill and decode descriptors.
    pd_pairing_mode: PdPairingMode,
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
    /// sticky selector consulted for keyed requests in [`Self::select_worker_for_model`].
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
        let text_needs = TextNeeds::of(default_policy.as_ref());
        Self {
            model_policies: Arc::new(DashMap::new()),
            text_needs: Arc::new(AtomicU8::new(text_needs.0)),
            text_needs_refresh: Arc::new(Mutex::new(())),
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
            pd_pairing_mode: PdPairingMode::default(),
        }
    }

    /// Set how strictly PD placement pairs prefill and decode descriptors.
    #[must_use]
    pub fn with_pd_pairing_mode(mut self, mode: PdPairingMode) -> Self {
        self.pd_pairing_mode = mode;
        self
    }

    /// How strictly PD placement pairs prefill and decode descriptors.
    pub fn pd_pairing_mode(&self) -> PdPairingMode {
        self.pd_pairing_mode
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

    /// Use the same header-first key for load accounting and worker selection.
    pub(crate) fn sticky_key<'a>(
        &self,
        headers: Option<&'a HeaderMap>,
        rid_key: Option<&'a str>,
    ) -> Option<&'a str> {
        self.sticky_header_key(headers).or(rid_key)
    }

    /// Whether sticky routing may derive its fallback key from the request
    /// body's `rid`, requiring automatic body-path selection to keep the body
    /// readable.
    pub(crate) fn routing_key_override_enabled(&self) -> bool {
        self.routing_key_sticky.is_some()
    }

    /// Resolve the effective sticky key: a valid configured routing-key header
    /// wins, with the rid-derived key as fallback. Header keys get the same
    /// lineage stripping as rid keys, so a
    /// proxy forwarding `conv_t2` as a header pins the entry `conv`.
    fn effective_sticky_key<'a>(
        &self,
        info: &SelectWorkerInfo<'a>,
    ) -> Option<(&'a str, &'static str)> {
        info.routing_key
            .or_else(|| self.resolve_routing_key(info.headers))
            .map(|raw| (Self::strip_header_key(raw), "header"))
            .or_else(|| info.rid_key.map(|key| (key, "rid")))
    }

    /// Select a worker, applying the sticky routing-key override when it is
    /// enabled, the request carries a key from the configured source, and the
    /// configured policy does not already honor the key (`manual` /
    /// `consistent_hashing`, which read `rid_key` and the header themselves
    /// with the same header-first precedence). Otherwise delegates to `policy`.
    /// `policy.name()` stays the real policy (for metrics).
    pub fn select_worker_for_model(
        &self,
        policy: &Arc<dyn LoadBalancingPolicy>,
        model_id: &str,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        cache_trace::begin_selection(workers);
        if let Some(sticky) = self.routing_key_sticky.as_ref() {
            if Self::routing_key_override_applies(policy.name()) {
                if let Some((key, source)) = self.effective_sticky_key(info) {
                    return Self::select_sticky(
                        sticky, policy, model_id, workers, info, key, source,
                    );
                }
            }
        }
        let selected = policy.select_worker(workers, info);
        cache_trace::selection(policy.name(), "policy", workers, selected);
        selected
    }

    #[cfg(test)]
    fn select_worker(
        &self,
        policy: &Arc<dyn LoadBalancingPolicy>,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let model_id = workers
            .first()
            .map(|worker| worker.model_id())
            .unwrap_or(crate::worker::UNKNOWN_MODEL_ID);
        self.select_worker_for_model(policy, model_id, workers, info)
    }

    /// Keyed selection: honor an existing pin under the in-flight cap;
    /// otherwise assign — via the underlying policy in `delegate` mode, via
    /// the sticky map's own assignment mode otherwise — and pin the result.
    fn select_sticky(
        sticky: &Arc<ManualPolicy>,
        policy: &Arc<dyn LoadBalancingPolicy>,
        model_id: &str,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        key: &str,
        source: &'static str,
    ) -> Option<usize> {
        Metrics::record_routing_key_source(source);

        // WorkerLoadGuard records the raw session key, so the cap must query
        // that same key. Model/role-scoped cap accounting is a separate contract.
        let load_key = key;

        // Models and PD roles stick independently while sharing one bounded map.
        // Length-prefix the model so delimiters in model IDs cannot cross fields.
        let model = normalize_model_key(model_id);
        let role = match info.role {
            WorkerRole::Single => 's',
            WorkerRole::Prefill => 'p',
            WorkerRole::Decode => 'd',
        };
        let namespaced = format!("{}:{model}:{role}:{key}", model.len());
        let key = namespaced.as_str();

        let over_cap =
            |idx: usize| workers[idx].routing_key_inflight(load_key) >= STICKY_INFLIGHT_CAP;
        let finish = |result: Option<usize>, branch: ExecutionBranch| {
            cache_trace::selection(policy.name(), branch.as_str(), workers, result);
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

        let healthy = get_healthy_worker_indices(workers, info);
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

    /// Policies that already honor the routing key keep their own handling
    /// (they consult the header before `rid_key`, like the override does); all
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

    /// Hand every policy the worker monitor's load feed (thread-safe, can be
    /// called after initialization); a policy created later takes it from
    /// the registry (see [`Self::publish_with_shared_state`]).
    pub(crate) fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        {
            let mut guard = self.load_rx.write();
            guard.clone_from(&rx);
        }
        self.each_policy(|policy| policy.set_load_receiver(rx.clone()));
    }

    /// Visit every policy the registry holds once: the default, the PD and
    /// EPD roles, then the per-model policies. A policy shared between slots
    /// (a model with no hint takes the default) is visited for the first
    /// slot it stands in only, so a visit that does work, a worker's
    /// removal, is not repeated per model.
    fn each_policy(&self, mut visit: impl FnMut(&Arc<dyn LoadBalancingPolicy>)) {
        let mut seen: Vec<*const ()> = Vec::new();
        let mut once = |policy: &Arc<dyn LoadBalancingPolicy>| {
            let ptr = Arc::as_ptr(policy) as *const ();
            if !seen.contains(&ptr) {
                seen.push(ptr);
                visit(policy);
            }
        };
        once(&self.default_policy);
        for role in [
            &self.prefill_policy,
            &self.decode_policy,
            &self.encode_policy,
        ] {
            if let Some(policy) = role.get() {
                once(policy);
            }
        }
        for entry in self.model_policies.iter() {
            once(entry.value());
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
    /// The request-completion observer to install on every worker (see
    /// [`RequestCompletionSink`]): policies that book state at dispatch
    /// release it when the request's load guard drops.
    pub fn completion_sink(self: &Arc<Self>) -> Arc<dyn RequestCompletionSink> {
        Arc::new(PolicyCompletionSink {
            registry: Arc::downgrade(self),
        })
    }

    /// The policy that places requests on `worker`: the prefill, decode or
    /// encode policy by worker type, else the model's policy. Completion
    /// reports and in-flight reconciliation both go there.
    fn policy_for_worker(&self, worker: &dyn Worker) -> Arc<dyn LoadBalancingPolicy> {
        match worker.worker_type() {
            WorkerType::Prefill => self.get_prefill_policy(),
            WorkerType::Decode => self.get_decode_policy(),
            WorkerType::Encode => self.get_encode_policy(),
            WorkerType::Regular => self.get_policy_or_default(worker.model_id()),
        }
    }

    /// Reconcile the policy that places requests on `worker` against the
    /// router's live in-flight count there (see
    /// [`LoadBalancingPolicy::reconcile_in_flight`]). The worker monitor
    /// calls this once per load poll for every polled worker.
    pub fn reconcile_in_flight(&self, worker: &dyn Worker) {
        self.policy_for_worker(worker)
            .reconcile_in_flight(worker.url(), worker.load());
    }

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

        self.publish_with_shared_state(&policy, || {
            self.model_policies
                .insert(model_id.to_string(), Arc::clone(&policy));
        });
        self.refresh_text_needs();

        policy
    }

    /// Hand `policy` the shared state a cache-aware policy consumes (the KV
    /// event monitor, the backend load feed, the mesh tree bridge), then run
    /// `publish` while the three guards are still held. Holding them across
    /// the publish is what makes this race-free against the setters: a
    /// setter either wrote before these reads, or its propagation scan runs
    /// after `publish` and finds the policy in place.
    fn publish_with_shared_state(
        &self,
        policy: &Arc<dyn LoadBalancingPolicy>,
        publish: impl FnOnce(),
    ) {
        let monitor = self.kv_event_monitor.read();
        let load_rx = self.load_rx.read();
        let mesh = self.mesh_tree_sync.read();
        Self::maybe_inject_monitor(policy, monitor.as_ref());
        policy.set_load_receiver(load_rx.as_ref().cloned());
        Self::maybe_inject_mesh_tree_sync(policy, mesh.as_ref());
        publish();
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
                self.refresh_text_needs();
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
    /// (per-model → default → PD/EPD roles). Deduplicated by `Arc`
    /// pointer identity so a policy that appears in multiple slots
    /// (e.g. the same `CacheAwarePolicy` as both default and prefill
    /// role) is returned once.
    ///
    /// The tree-sync bridge uses this to reach every `CacheAwarePolicy`
    /// that could have produced a delta for `model_id` — reaching
    /// only `model_policies` misses PD-role and default-fallback
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
        self.refresh_text_needs();
    }

    /// Set the prefill policy for PD mode (set once at startup).
    ///
    /// The router factory sets the prefill policy after the app context wired the
    /// KV event monitor, the load feed and the mesh tree bridge into the
    /// registry, and those setters only reach policies that already exist,
    /// so the policy takes that state here, under the shared-state read guards.
    pub fn set_prefill_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        self.publish_with_shared_state(&policy, || {
            // OnceLock::set returns Err if already set, which we ignore since
            // the policy should only be set once at startup
            let _ = self.prefill_policy.set(Arc::clone(&policy));
        });
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

    /// Set the decode policy for PD mode (set once at startup).
    ///
    /// The router factory sets the decode policy after the app context wired the
    /// KV event monitor, the load feed and the mesh tree bridge into the
    /// registry, and those setters only reach policies that already exist,
    /// so the policy takes that state here, under the shared-state read guards.
    pub fn set_decode_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        self.publish_with_shared_state(&policy, || {
            // OnceLock::set returns Err if already set, which we ignore since
            // the policy should only be set once at startup
            let _ = self.decode_policy.set(Arc::clone(&policy));
        });
    }

    /// Set the encode policy for EPD mode (set once at startup).
    ///
    /// The router factory sets the encode policy after the app context wired the
    /// KV event monitor, the load feed and the mesh tree bridge into the
    /// registry, and those setters only reach policies that already exist,
    /// so the policy takes that state here, under the shared-state read guards.
    pub fn set_encode_policy(&self, policy: Arc<dyn LoadBalancingPolicy>) {
        self.publish_with_shared_state(&policy, || {
            // OnceLock::set returns Err if already set, which we ignore since
            // the policy should only be set once at startup
            let _ = self.encode_policy.set(Arc::clone(&policy));
        });
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
        self.any_policy_wants_text(headers, false)
    }

    /// [`Self::any_policy_needs_request_text`] for a request whose token ids
    /// are available: a policy that routes on the ids and reads text only
    /// without them ([`LoadBalancingPolicy::needs_request_text_with_tokens`])
    /// does not count, so the ids go into the selection snapshot alone.
    pub fn any_policy_needs_request_text_with_tokens(&self, headers: Option<&HeaderMap>) -> bool {
        self.any_policy_wants_text(headers, true)
    }

    /// The two text questions' shared rules: a valid routing-tokens hint
    /// lifts the requirement, and a valid routing key under the sticky
    /// override lifts it for the policies the override supersedes; what the
    /// policies answer is read from [`Self::text_needs`], kept current where
    /// a policy is added or removed, so a request walks nothing.
    fn any_policy_wants_text(&self, headers: Option<&HeaderMap>, with_tokens: bool) -> bool {
        if parse_routing_tokens_hint(headers).is_some() {
            return false;
        }
        let keyed_override =
            self.routing_key_sticky.is_some() && self.resolve_routing_key(headers).is_some();
        TextNeeds(self.text_needs.load(Ordering::Acquire)).answer(with_tokens, keyed_override)
    }

    /// Recompute [`Self::text_needs`] over the policies the text questions
    /// range over: the default policy and the per-model policies. Called
    /// where the per-model map changes; one recomputation at a time, so the
    /// bits published last are those of a walk that saw every earlier change.
    fn refresh_text_needs(&self) {
        let _one_at_a_time = self.text_needs_refresh.lock();
        let mut needs = TextNeeds::of(self.default_policy.as_ref());
        for entry in self.model_policies.iter() {
            needs = needs.join(TextNeeds::of(entry.value().as_ref()));
        }
        self.text_needs.store(needs.0, Ordering::Release);
    }

    /// Whether any policy routes on backend load (see
    /// [`LoadBalancingPolicy::needs_backend_loads`]): with load monitoring
    /// off, the worker monitor still polls a group while this holds.
    pub fn needs_backend_loads(&self) -> bool {
        let mut needs = false;
        self.each_policy(|policy| needs |= policy.needs_backend_loads());
        needs
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

    /// Remove a batch of workers from every cache-aware policy that may route
    /// the model. Shared policies are deduplicated by Arc identity.
    pub(crate) fn remove_workers_from_cache_aware(
        &self,
        model_id: &str,
        worker_urls: &HashSet<String>,
    ) {
        // A model can route through its explicit, default, or PD/EPD policy.
        // `policies_for_model` deduplicates shared policy instances so each
        // tree is cleaned at most once.
        for policy in self.policies_for_model(model_id) {
            if policy.name() == "cache_aware" {
                if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                    cache_aware.remove_workers_from_model(model_id, worker_urls);
                    debug!(
                        "Removed {} workers from cache-aware policy for model {}",
                        worker_urls.len(),
                        model_id
                    );
                }
            }
        }
    }

    /// Drop the state every policy keeps on a removed worker (a load-aware
    /// policy's dispatch tally; see [`LoadBalancingPolicy::remove_worker`]),
    /// which would otherwise grow without bound under worker churn. Called
    /// on worker removal alongside the cache-aware cleanup above.
    pub fn remove_worker_from_policies(&self, worker_url: &str) {
        self.each_policy(|policy| policy.remove_worker(worker_url));
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
            .field(
                "text_needs",
                &TextNeeds(self.text_needs.load(Ordering::Relaxed)),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests;
