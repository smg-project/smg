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
    normalize_model_key,
    traffic_split::{self, Split},
    BucketPolicy, CacheAwarePolicy, DPRankLoadPolicy, LoadBalancingPolicy, ManualConfig,
    ManualPolicy, PolicyFactory, SelectWorkerInfo, WorkerRole,
};
use crate::{
    config::types::{ManualAssignmentMode, PdPairingMode, PolicyConfig, RoutingKeyOverrideConfig},
    mesh::adapters::TreeSyncAdapter,
    observability::{cache_trace, metrics::Metrics},
    routers::common::header_utils::{
        extract_routing_key_hint_named, extract_target_worker, parse_routing_tokens_hint,
        ROUTING_KEY_HINT_MAX_BYTES,
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
    /// Whether the weighted traffic split across worker groups runs before
    /// the policy (`--traffic-split`); see [`super::traffic_split`].
    traffic_split: bool,
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
            traffic_split: false,
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

    /// Run the weighted traffic split across worker groups before every
    /// selection (`--traffic-split`).
    #[must_use]
    pub fn with_traffic_split(mut self, enabled: bool) -> Self {
        self.traffic_split = enabled;
        self
    }

    /// Whether the weighted traffic split is on.
    pub fn traffic_split_enabled(&self) -> bool {
        self.traffic_split
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

    /// Select a worker: under `--traffic-split`, first confine the choice to
    /// one worker group picked by weight (see [`super::traffic_split`]), then
    /// apply the sticky routing-key override when it is enabled, the request
    /// carries a key from the configured source, and the configured policy
    /// does not already honor the key (`manual` / `consistent_hashing`, which
    /// read `rid_key` and the header themselves with the same header-first
    /// precedence); otherwise delegate to `policy`. `policy.name()` stays the
    /// real policy (for metrics). The returned index is into `workers`.
    pub fn select_worker_for_model(
        &self,
        policy: &Arc<dyn LoadBalancingPolicy>,
        model_id: &str,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        self.select_worker_for_model_split(policy, model_id, workers, info, None)
            .map(|(selected, _)| selected)
    }

    /// [`Self::select_worker_for_model`] for one leg of a request, reporting
    /// the group the pick is in: the one the split confined it to, or, when
    /// the split did not run (exempt, or one group among `workers`), the
    /// pick's own group if it is a named one; `None` for the default group,
    /// and whenever the split is off. `follow` is the group an earlier leg
    /// reported (the decode leg after the prefill's): under `--traffic-split`
    /// the choice stays in it while it has an available member of positive
    /// weight among `workers`, so a prefill-decode pair never spans two
    /// groups (two engine builds, which lenient pairing would otherwise
    /// pair). A first leg picked unsplit from the default group reports
    /// `None`, and the later leg then decides for itself, so a canary only
    /// the decode pool carries keeps its share; one picked unsplit from a
    /// named group (its slice shrunk to that group, the others unavailable
    /// or at capacity) still pins the later leg to it; a followed group
    /// absent or drained among `workers` leaves the key or the draw to decide
    /// as for a first leg. Every confined selection counts in
    /// `smg_traffic_split_selections_total`, one per leg.
    pub fn select_worker_for_model_split<'w>(
        &self,
        policy: &Arc<dyn LoadBalancingPolicy>,
        model_id: &str,
        workers: &'w [Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        follow: Option<&str>,
    ) -> Option<(usize, Option<&'w str>)> {
        if self.traffic_split && Self::traffic_split_applies(policy.name(), info) {
            if let Some(split) = Split::plan(workers) {
                // The group an earlier leg was split into is followed;
                // otherwise the same key the sticky override would pin by
                // keeps a session in one group, and a keyless request draws
                // at random.
                let arm = match follow.and_then(|name| split.arm_named(name)) {
                    Some(arm) => arm,
                    None => match self.effective_sticky_key(info) {
                        Some((key, _)) => split.arm_for_key(key),
                        None => split.arm_at_random(&mut rand::rng()),
                    },
                };
                Metrics::record_traffic_split_selection(model_id, arm.name);
                let members: Vec<Arc<dyn Worker>> = arm
                    .members
                    .iter()
                    .map(|&idx| Arc::clone(&workers[idx]))
                    .collect();
                let picked = self.select_within(policy, model_id, &members, info)?;
                return Some((arm.members[picked], Some(arm.name)));
            }
        }
        let selected = self.select_within(policy, model_id, workers, info)?;
        // Not split: a pick in a named group still reports it, so a later
        // leg stays in that group when the slice shrank to it; a pick in the
        // default group reports nothing and leaves the later leg to decide
        // for itself. Off, the labels are inert and nothing is reported.
        let group = self
            .traffic_split
            .then(|| traffic_split::group_of(workers[selected].as_ref()))
            .filter(|group| *group != traffic_split::DEFAULT_GROUP);
        Some((selected, group))
    }

    /// Whether the traffic split may confine this selection: `bucket` places
    /// prefill by request size over boundaries spanning its whole fleet,
    /// which one group cannot represent; and an explicit
    /// `X-SMG-Target-Worker` under `consistent_hashing`, the policy that
    /// reads it, indexes the whole pool and outranks the split (as it
    /// outranks the routing key). Every other policy ignores the header, so
    /// the split holds there: a drained group takes nothing from a request
    /// that carries it.
    fn traffic_split_applies(policy_name: &str, info: &SelectWorkerInfo) -> bool {
        policy_name != "bucket"
            && !(policy_name == "consistent_hashing"
                && extract_target_worker(info.headers).is_some())
    }

    /// The selection proper over `workers`: the sticky override when it
    /// applies, else the policy.
    fn select_within(
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
mod tests {
    use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};
    use tokio::sync::watch;
    use tracing_test::traced_test;

    use super::*;
    use crate::{
        policies::{BucketConfig, CacheAwareConfig, LeastLoadPolicy, SelectWorkerInfo},
        worker::{
            load_state::LoadSnapshot, test_support::no_health_check, BasicWorkerBuilder, HashRing,
            Worker, WorkerLoadGuard, WorkerType,
        },
    };

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
    fn routing_key_override_keeps_model_pins_independent() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                enabled: true,
                assignment_mode: ManualAssignmentMode::Delegate,
                ..Default::default()
            },
        );
        let policy = reg.get_default_policy();
        let pools = [
            vec![
                worker("http://a1", WorkerType::Regular),
                worker("http://a2", WorkerType::Regular),
            ],
            vec![worker("http://b1", WorkerType::Regular)],
            vec![worker("http://c1", WorkerType::Regular)],
        ];
        let info = SelectWorkerInfo {
            rid_key: Some("session-42"),
            ..Default::default()
        };

        let first_a = reg
            .select_worker_for_model(&policy, "model-a", &pools[0], &info)
            .unwrap();
        reg.select_worker_for_model(&policy, "model-b", &pools[1], &info)
            .unwrap();
        reg.select_worker_for_model(&policy, "model-c", &pools[2], &info)
            .unwrap();

        assert_eq!(
            reg.select_worker_for_model(&policy, "model-a", &pools[0], &info),
            Some(first_a),
            "other models must not consume this model's bounded failover slots"
        );
    }

    /// `--routing-key-override` means the same thing under every policy:
    /// the routing-key header outranks the body rid. Key-native policies
    /// skip the sticky override, so they must apply that precedence themselves.
    #[test]
    fn header_key_outranks_rid_under_key_native_policies() {
        for config in [
            PolicyConfig::Manual {
                eviction_interval_secs: 60,
                max_idle_secs: 3600,
                assignment_mode: ManualAssignmentMode::Random,
            },
            PolicyConfig::ConsistentHashing,
        ] {
            let reg = PolicyRegistry::with_override(
                config,
                RoutingKeyOverrideConfig {
                    enabled: true,
                    ..Default::default()
                },
            );
            let policy = reg.get_default_policy();
            let name = policy.name();
            assert!(!PolicyRegistry::routing_key_override_applies(name));
            let workers = vec![
                worker("http://w1", WorkerType::Regular),
                worker("http://w2", WorkerType::Regular),
                worker("http://w3", WorkerType::Regular),
                worker("http://w4", WorkerType::Regular),
            ];
            let hash_ring = Some(Arc::new(HashRing::new(workers.iter().map(|w| w.url()))));
            let headers = headers_with_key("conversation-42");
            let pinned = reg
                .select_worker(
                    &policy,
                    &workers,
                    &SelectWorkerInfo {
                        headers: Some(&headers),
                        routing_key: reg.resolve_routing_key(Some(&headers)),
                        hash_ring: hash_ring.clone(),
                        ..Default::default()
                    },
                )
                .unwrap();

            for request in 0..4 {
                let rid = format!("request:group:request-{request}");
                let info = SelectWorkerInfo {
                    headers: Some(&headers),
                    routing_key: reg.resolve_routing_key(Some(&headers)),
                    rid_key: reg.derive_rid_key(Some(&rid)),
                    hash_ring: hash_ring.clone(),
                    ..Default::default()
                };
                assert_eq!(
                    reg.select_worker(&policy, &workers, &info),
                    Some(pinned),
                    "{name}: header must outrank body rid {rid}"
                );
            }
            if let Some(manual) = policy.as_any().downcast_ref::<ManualPolicy>() {
                assert_eq!(manual.map_len(), 1);
            }
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
        PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096)
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

    /// The answer follows the per-model map: a text policy's model lifts the
    /// requirement while one of its workers stands, its last worker's removal
    /// drops it, and so does clearing the registry. A model on the text-free
    /// default policy changes nothing either way.
    #[test]
    fn text_requirement_follows_the_per_model_policies() {
        let reg = PolicyRegistry::new(PolicyConfig::RoundRobin);
        reg.on_worker_added("plain", None);
        assert!(!reg.any_policy_needs_request_text(None));

        reg.on_worker_added("affine", Some("cache_aware"));
        reg.on_worker_added("affine", Some("cache_aware"));
        assert!(reg.any_policy_needs_request_text(None));
        reg.on_worker_removed("affine");
        assert!(reg.any_policy_needs_request_text(None));
        reg.on_worker_removed("affine");
        assert!(!reg.any_policy_needs_request_text(None));
        reg.on_worker_removed("plain");
        assert!(!reg.any_policy_needs_request_text(None));

        reg.on_worker_added("affine", Some("cache_aware"));
        assert!(reg.any_policy_needs_request_text(None));
        reg.clear();
        assert!(!reg.any_policy_needs_request_text(None));
    }

    /// The override's exemption is per policy: a sticky-exempt text policy
    /// (`consistent_hashing` honours the key itself and reads no text, so the
    /// stand-in here is a test policy by that name) keeps the requirement
    /// under a valid key where `cache_aware` loses it, as a per-model policy
    /// beside the default; the requirement leaves with it. (No hint names
    /// the stand-in, so it goes into the map directly and the test refreshes
    /// what registration and removal refresh.)
    #[test]
    fn text_requirement_under_a_key_counts_the_exempt_policies_only() {
        let reg = PolicyRegistry::with_override(cache_aware_config(), enabled_override());
        let keyed = headers_with_key("session-A");
        assert!(reg.any_policy_needs_request_text(None));
        assert!(!reg.any_policy_needs_request_text(Some(&keyed)));
        assert!(!reg.any_policy_needs_request_text_with_tokens(None));
        assert_eq!(
            TextNeeds(reg.text_needs.load(Ordering::Relaxed)),
            TextNeeds(TextNeeds::TEXT)
        );

        reg.model_policies
            .insert("keyed-text".to_string(), Arc::new(TextBesideIdsExempt));
        reg.refresh_text_needs();
        assert!(reg.any_policy_needs_request_text(Some(&keyed)));
        assert!(reg.any_policy_needs_request_text_with_tokens(Some(&keyed)));
        assert!(reg.any_policy_needs_request_text_with_tokens(None));
        assert!(!reg.any_policy_needs_request_text(Some(&headers_with_tokens("1,2,3"))));
        assert_eq!(
            TextNeeds(reg.text_needs.load(Ordering::Relaxed)),
            TextNeeds(
                TextNeeds::TEXT
                    | TextNeeds::TEXT_UNDER_KEY
                    | TextNeeds::TEXT_WITH_TOKENS
                    | TextNeeds::TEXT_WITH_TOKENS_UNDER_KEY
            )
        );

        reg.model_policies.remove("keyed-text");
        reg.refresh_text_needs();
        assert!(!reg.any_policy_needs_request_text(Some(&keyed)));
        assert!(!reg.any_policy_needs_request_text_with_tokens(None));
        assert_eq!(
            TextNeeds(reg.text_needs.load(Ordering::Relaxed)),
            TextNeeds(TextNeeds::TEXT)
        );
    }

    /// A text policy the routing-key override leaves alone (by name).
    #[derive(Debug)]
    struct TextBesideIdsExempt;

    impl LoadBalancingPolicy for TextBesideIdsExempt {
        fn select_worker(
            &self,
            workers: &[Arc<dyn Worker>],
            _info: &SelectWorkerInfo,
        ) -> Option<usize> {
            (!workers.is_empty()).then_some(0)
        }

        fn name(&self) -> &'static str {
            "consistent_hashing"
        }

        fn needs_request_text(&self) -> bool {
            true
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// A policy that reads the text beside the ids; none of the built-in
    /// ones does, so the trait's default is what such a policy gets.
    #[derive(Debug)]
    struct TextBesideIds;

    impl LoadBalancingPolicy for TextBesideIds {
        fn select_worker(
            &self,
            workers: &[Arc<dyn Worker>],
            _info: &SelectWorkerInfo,
        ) -> Option<usize> {
            (!workers.is_empty()).then_some(0)
        }

        fn name(&self) -> &'static str {
            "text_beside_ids"
        }

        fn needs_request_text(&self) -> bool {
            true
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Beside token ids the text is wanted only by a policy that reads it
    /// there: `cache_aware` and `bucket` route on the ids and answer no, as
    /// the default and as a per-model policy, with the same hint rules; a
    /// policy that says nothing about the ids keeps its text answer.
    #[test]
    fn text_beside_token_ids_is_wanted_only_by_a_policy_that_reads_it_there() {
        let reg = PolicyRegistry::new(cache_aware_config());
        assert!(reg.any_policy_needs_request_text(None));
        assert!(!reg.any_policy_needs_request_text_with_tokens(None));
        assert!(!reg.any_policy_needs_request_text_with_tokens(Some(&headers_with_tokens("1,2,3"))));
        for policy in ["cache_aware", "bucket"] {
            let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
            reg.on_worker_added("llama-3", Some(policy));
            assert!(reg.any_policy_needs_request_text(None), "{policy}");
            assert!(
                !reg.any_policy_needs_request_text_with_tokens(None),
                "{policy}"
            );
        }
        // A per-model policy that reads text beside the ids makes the
        // registry ask for it, under the same hint rule. (The policy goes
        // into the map directly, as no hint names it, so the test refreshes
        // what `on_worker_added` refreshes after its insert.)
        let reg = PolicyRegistry::new(PolicyConfig::RoundRobin);
        reg.model_policies
            .insert("llama-text".to_string(), Arc::new(TextBesideIds));
        reg.refresh_text_needs();
        assert!(reg.any_policy_needs_request_text_with_tokens(None));
        assert!(!reg.any_policy_needs_request_text_with_tokens(Some(&headers_with_tokens("1,2,3"))));
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
    fn header_key_wins_over_rid_and_rid_is_fallback() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];

        let headers = headers_with_key("conversation-42");
        let first = reg
            .select_worker(
                &policy,
                &workers,
                &SelectWorkerInfo {
                    headers: Some(&headers),
                    rid_key: Some("request:group:request-1"),
                    ..Default::default()
                },
            )
            .unwrap();
        for rid in ["request:group:request-2", "request:group:request-3"] {
            assert_eq!(
                reg.select_worker(
                    &policy,
                    &workers,
                    &SelectWorkerInfo {
                        headers: Some(&headers),
                        rid_key: Some(rid),
                        ..Default::default()
                    },
                ),
                Some(first)
            );
        }

        assert_eq!(reg.routing_key_sticky.as_ref().unwrap().map_len(), 1);

        // No header: request-ID lineage still gets its own stable pin.
        let rid_info = SelectWorkerInfo {
            rid_key: reg.derive_rid_key(Some("conv42_t1")),
            ..Default::default()
        };
        let pinned = reg.select_worker(&policy, &workers, &rid_info).unwrap();
        assert_ne!(pinned, first);
        assert_eq!(
            reg.select_worker(
                &policy,
                &workers,
                &SelectWorkerInfo {
                    rid_key: reg.derive_rid_key(Some("conv42_t2_r1")),
                    ..Default::default()
                },
            ),
            Some(pinned)
        );
    }

    #[test]
    fn invalid_header_keys_fall_back_to_rid_for_selection_and_accounting() {
        let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
        let rid_key = reg.derive_rid_key(Some("conversation_t2_r1"));
        let mut invalid_utf8 = HeaderMap::new();
        invalid_utf8.insert(
            "x-smg-routing-key",
            http::HeaderValue::from_bytes(&[0xff]).unwrap(),
        );
        for headers in [
            HeaderMap::new(),
            headers_with_key(""),
            headers_with_key(&"k".repeat(ROUTING_KEY_HINT_MAX_BYTES + 1)),
            invalid_utf8,
        ] {
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                rid_key,
                ..Default::default()
            };
            assert_eq!(
                reg.effective_sticky_key(&info),
                Some(("conversation", "rid"))
            );
            assert_eq!(
                reg.sticky_key(Some(&headers), rid_key),
                Some("conversation")
            );
        }
    }

    #[test]
    fn header_priority_accounts_inflight_requests_under_the_selected_key() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
        ];
        let headers = headers_with_key("conversation_t1");
        let mut guards = Vec::new();
        for rid in ["request-1", "request-2", "request-3"] {
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                rid_key: Some(rid),
                ..Default::default()
            };
            assert_eq!(
                reg.effective_sticky_key(&info),
                Some(("conversation", "header"))
            );
            let selected = reg.select_worker(&policy, &workers, &info).unwrap();
            assert_eq!(selected, usize::from(guards.len() == STICKY_INFLIGHT_CAP));
            guards.push(WorkerLoadGuard::with_key(
                workers[selected].clone(),
                reg.sticky_key(Some(&headers), info.rid_key),
            ));
            assert_eq!(workers[selected].routing_key_inflight(rid), 0);
        }
        assert_eq!(workers[0].routing_key_inflight("conversation"), 2);
        assert_eq!(workers[1].routing_key_inflight("conversation"), 1);
        drop(guards);
        assert_eq!(workers[0].routing_key_inflight("conversation"), 0);
        assert_eq!(workers[1].routing_key_inflight("conversation"), 0);
    }

    #[test]
    fn delegate_assignment_routes_via_underlying_policy_then_pins() {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 10000),
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

    /// A policy shared between slots is visited once: a model with no hint
    /// takes the registry's default, so one worker's removal reaches that
    /// policy once, however many models stand on it, not once per model.
    #[test]
    fn a_policy_shared_between_slots_is_visited_once() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        for model in ["m1", "m2", "m3"] {
            registry.on_worker_added(model, None);
        }
        registry.on_worker_added("hinted", Some("least_load"));
        let mut visits: Vec<*const ()> = Vec::new();
        registry.each_policy(|policy| visits.push(Arc::as_ptr(policy) as *const ()));
        assert_eq!(
            visits.len(),
            2,
            "the shared default once, the hinted policy once"
        );
        let default = Arc::as_ptr(&registry.default_policy) as *const ();
        assert_eq!(visits.iter().filter(|p| **p == default).count(), 1);
    }

    #[test]
    fn test_passthrough_is_not_load_aware() {
        // Passthrough must not be polled by the WorkerMonitor: with it as the
        // default policy (and a passthrough model policy too), the load-aware
        // set stays empty.
        let registry = PolicyRegistry::new(PolicyConfig::Passthrough);
        registry.on_worker_added("m", Some("passthrough"));
        assert!(!registry.needs_backend_loads());
    }

    /// The load feed reaches every policy that scores on it, not only the
    /// cache-aware ones: a power-of-two default routes on the snapshot the
    /// registry handed it, and a model policy created afterwards takes the
    /// feed from the registry.
    #[test]
    fn the_load_feed_reaches_every_policy_that_scores_on_it() {
        fn report(queued_tokens: i32) -> WorkerLoadResponse {
            WorkerLoadResponse {
                loads: vec![SchedulerLoadSnapshot {
                    num_waiting_uncached_tokens: queued_tokens,
                    gen_throughput: 100.0,
                    ..Default::default()
                }],
                ..Default::default()
            }
        }

        let registry = PolicyRegistry::new(PolicyConfig::PowerOfTwo {
            load_metric: Default::default(),
        });
        let (_tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
            ("http://a:8000".to_string(), report(100_000)),
            ("http://b:8000".to_string(), report(0)),
        ]));
        registry.set_load_receiver(Some(rx));
        let later = registry.on_worker_added("m", Some("least_load"));

        let workers = vec![
            worker("http://a:8000", WorkerType::Regular),
            worker("http://b:8000", WorkerType::Regular),
        ];
        for policy in [registry.get_default_policy(), later] {
            for _ in 0..20 {
                assert_eq!(
                    policy.select_worker(&workers, &SelectWorkerInfo::default()),
                    Some(1),
                    "{} did not route on the feed",
                    policy.name()
                );
            }
        }
    }

    #[test]
    fn cache_aware_is_always_load_aware_for_expected_wait() {
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
                selection_policy: None,
                selection_accounting_ttl_ms: 0,
                balance_prefix_margin_tokens: 0,
                balance_uncached_credit: false,
            }
        }

        // Even without pressure tuning, misses, spills, and equal-affinity
        // ties use LeastLoad's expected-wait snapshots.
        let plain = PolicyRegistry::new(cache_aware(0.0, 1.0, 1.0));
        assert!(plain.needs_backend_loads());

        // Pressure knobs do not change that load-aware membership.
        for pressured in [
            cache_aware(1.0, 1.0, 1.0),
            cache_aware(0.0, 0.5, 1.0),
            cache_aware(0.0, 1.0, 0.9),
        ] {
            let registry = PolicyRegistry::new(pressured);
            assert!(registry.needs_backend_loads());
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
            selection_policy: None,
            selection_accounting_ttl_ms: 0,
            balance_prefix_margin_tokens: 0,
            balance_uncached_credit: false,
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

        let registry = Arc::new(PolicyRegistry::new(PolicyConfig::cache_aware(
            0.5, 32, 1.5, 0, 128,
        )));

        for round in 0..64 {
            let monitor = Arc::new(KvEventMonitor::new());
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

    /// The PD/EPD policies are set after the app context wired the shared state
    /// into the registry (the KV event monitor, the load feed, and the mesh
    /// tree bridge that `MeshAdapters::start` attaches), so the policy setters
    /// hand all of it over themselves.
    #[tokio::test]
    async fn pd_role_policies_receive_the_shared_state_set_before_them() {
        use std::collections::BTreeMap;

        use smg_mesh::MeshKV;

        use crate::{mesh::MeshAdapters, worker::WorkerRegistry};

        let registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        registry.set_kv_event_monitor(Some(Arc::new(KvEventMonitor::new())));
        let (_load_tx, load_rx) = watch::channel(LoadSnapshot::from_loads_for_test(Vec::new()));
        registry.set_load_receiver(Some(load_rx));
        let mesh = MeshKV::new("node-a".into());
        let _adapters = MeshAdapters::start(
            &mesh,
            "node-a".into(),
            Arc::new(WorkerRegistry::new()),
            Arc::new(RwLock::new(BTreeMap::new())),
            Arc::clone(&registry),
        );

        let prefill = cache_aware_policy();
        let decode = cache_aware_policy();
        let encode = cache_aware_policy();
        registry.set_prefill_policy(Arc::clone(&prefill));
        registry.set_decode_policy(Arc::clone(&decode));
        registry.set_encode_policy(Arc::clone(&encode));

        for (role, policy) in [("prefill", prefill), ("decode", decode), ("encode", encode)] {
            let cache_aware = policy.as_any().downcast_ref::<CacheAwarePolicy>().unwrap();
            assert!(
                cache_aware.kv_event_monitor_is_set_for_test(),
                "{role} policy missed the KV event monitor"
            );
            assert!(
                cache_aware.has_load_receiver_for_test(),
                "{role} policy missed the load feed"
            );
            assert!(
                cache_aware.should_populate_hash_index_for_test(),
                "{role} policy missed the mesh tree bridge"
            );
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

    /// A policy that remembers completions and reconciliation ticks.
    #[derive(Debug, Default)]
    struct CompletionRecorder {
        completed: std::sync::Mutex<Vec<String>>,
        reconciled: std::sync::Mutex<Vec<(String, usize)>>,
    }

    impl LoadBalancingPolicy for CompletionRecorder {
        fn select_worker(
            &self,
            workers: &[Arc<dyn Worker>],
            _info: &SelectWorkerInfo,
        ) -> Option<usize> {
            (!workers.is_empty()).then_some(0)
        }

        fn on_request_complete(&self, worker_url: &str, _success: bool) {
            self.completed.lock().unwrap().push(worker_url.to_string());
        }

        fn reconcile_in_flight(&self, worker_url: &str, in_flight: usize) {
            self.reconciled
                .lock()
                .unwrap()
                .push((worker_url.to_string(), in_flight));
        }

        fn name(&self) -> &'static str {
            "completion_recorder"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn completion_sink_reports_to_the_policy_that_owns_the_worker_type() {
        use crate::worker::{BasicWorkerBuilder, WorkerLoadGuard};

        let registry = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
        let prefill_policy = Arc::new(CompletionRecorder::default());
        let decode_policy = Arc::new(CompletionRecorder::default());
        registry.set_prefill_policy(Arc::clone(&prefill_policy) as Arc<dyn LoadBalancingPolicy>);
        registry.set_decode_policy(Arc::clone(&decode_policy) as Arc<dyn LoadBalancingPolicy>);
        let sink = registry.completion_sink();

        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://prefill:9000")
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:9000")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        let regular: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://regular:9000")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        for worker in [&prefill, &decode, &regular] {
            worker.set_completion_sink(Some(Arc::clone(&sink)));
        }

        drop(WorkerLoadGuard::new(Arc::clone(&prefill), None));
        drop(WorkerLoadGuard::new(Arc::clone(&decode), None));
        // The regular worker's model policy (random) accepts the completion silently.
        drop(WorkerLoadGuard::new(Arc::clone(&regular), None));

        assert_eq!(
            prefill_policy.completed.lock().unwrap().as_slice(),
            ["grpc://prefill:9000"]
        );
        assert_eq!(
            decode_policy.completed.lock().unwrap().as_slice(),
            ["grpc://decode:9000"]
        );

        // A worker that outlives its registry reports to nobody.
        drop(registry);
        drop(WorkerLoadGuard::new(Arc::clone(&prefill), None));
        assert_eq!(prefill_policy.completed.lock().unwrap().len(), 1);
    }

    #[test]
    fn reconcile_in_flight_reaches_the_policy_that_owns_the_worker_type() {
        use crate::worker::{BasicWorkerBuilder, WorkerLoadGuard};

        let registry = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
        let decode_policy = Arc::new(CompletionRecorder::default());
        registry.set_decode_policy(Arc::clone(&decode_policy) as Arc<dyn LoadBalancingPolicy>);
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://decode:9000")
                .worker_type(WorkerType::Decode)
                .build(),
        );

        // The count handed over is the router's live one: two guards held,
        // then none.
        let held = [
            WorkerLoadGuard::new(Arc::clone(&decode), None),
            WorkerLoadGuard::new(Arc::clone(&decode), None),
        ];
        registry.reconcile_in_flight(decode.as_ref());
        drop(held);
        registry.reconcile_in_flight(decode.as_ref());
        assert_eq!(
            decode_policy.reconciled.lock().unwrap().as_slice(),
            [
                ("grpc://decode:9000".to_string(), 2),
                ("grpc://decode:9000".to_string(), 0)
            ]
        );
    }

    /// A policy that remembers which workers it was told to forget and
    /// consumes no load reports.
    #[derive(Debug, Default)]
    struct RemovalRecorder {
        removed: std::sync::Mutex<Vec<String>>,
    }

    impl LoadBalancingPolicy for RemovalRecorder {
        fn select_worker(
            &self,
            workers: &[Arc<dyn Worker>],
            _info: &SelectWorkerInfo,
        ) -> Option<usize> {
            (!workers.is_empty()).then_some(0)
        }

        fn remove_worker(&self, url: &str) {
            self.removed.lock().unwrap().push(url.to_string());
        }

        fn name(&self) -> &'static str {
            "removal_recorder"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// A worker's removal reaches every policy instance once: the ones that
    /// consume load reports and the ones that do not, a shared instance once
    /// however many roles it serves.
    #[test]
    fn worker_removal_reaches_every_policy_once() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let shared = Arc::new(RemovalRecorder::default());
        registry.set_prefill_policy(Arc::clone(&shared) as Arc<dyn LoadBalancingPolicy>);
        registry.set_decode_policy(Arc::clone(&shared) as Arc<dyn LoadBalancingPolicy>);
        let per_model = Arc::new(RemovalRecorder::default());
        registry.model_policies.insert(
            "m1".to_string(),
            Arc::clone(&per_model) as Arc<dyn LoadBalancingPolicy>,
        );

        registry.remove_worker_from_policies("http://w2:8000");

        assert_eq!(
            shared.removed.lock().unwrap().as_slice(),
            ["http://w2:8000"]
        );
        assert_eq!(
            per_model.removed.lock().unwrap().as_slice(),
            ["http://w2:8000"]
        );
    }

    /// A prefill worker's removal reaches the bucket policy, which keeps the
    /// prefill workers' size partition without consuming load reports: every
    /// request size still selects one of the workers left.
    #[test]
    fn removing_a_prefill_worker_reaches_the_bucket_policy() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        registry.set_prefill_policy(Arc::new(BucketPolicy::with_config(BucketConfig {
            balance_abs_threshold: usize::MAX,
            bucket_adjust_interval_secs: 3600,
            ..Default::default()
        })));
        let prefill = vec![
            worker("http://prefill-1:8000", WorkerType::Prefill),
            worker("http://prefill-2:8000", WorkerType::Prefill),
            worker("http://prefill-3:8000", WorkerType::Prefill),
        ];
        registry.init_pd_bucket_policies(&prefill);

        registry.remove_worker_from_policies("http://prefill-2:8000");

        let remaining = [Arc::clone(&prefill[0]), Arc::clone(&prefill[2])];
        let policy = registry.get_prefill_policy();
        for size in [0, 1364, 1365, 2000, 2729, 2730, 10_000] {
            let text = "a".repeat(size);
            let info = SelectWorkerInfo {
                request_text: Some(&text),
                ..Default::default()
            };
            let selected = policy
                .select_worker(&remaining, &info)
                .unwrap_or_else(|| panic!("a request of {size} chars selects no prefill worker"));
            assert_ne!(
                remaining[selected].url(),
                "http://prefill-2:8000",
                "size {size}"
            );
        }
    }

    /// The weighted traffic split (`--traffic-split`) runs before the policy
    /// and the sticky override, inside [`PolicyRegistry::select_worker_for_model`].
    mod traffic_split {
        use openai_protocol::worker::WorkerStatus;

        use super::*;
        use crate::{
            config::types::RoutingKeyOverrideConfig,
            policies::traffic_split::{GROUP_LABEL, WEIGHT_LABEL},
        };

        fn grouped(url: &str, group: Option<&str>, weight: Option<&str>) -> Arc<dyn Worker> {
            let mut labels = HashMap::new();
            if let Some(group) = group {
                labels.insert(GROUP_LABEL.to_string(), group.to_string());
            }
            if let Some(weight) = weight {
                labels.insert(WEIGHT_LABEL.to_string(), weight.to_string());
            }
            let worker = BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .labels(labels)
                .build();
            worker.set_status(WorkerStatus::Ready);
            Arc::new(worker)
        }

        /// Two stable workers at weight 90 around one canary at weight 10:
        /// indices 0 and 2 are stable, 1 is the canary.
        fn fleet(canary_weight: &str) -> Vec<Arc<dyn Worker>> {
            vec![
                grouped("http://s1:8000", Some("stable"), Some("90")),
                grouped("http://c1:8000", Some("canary"), Some(canary_weight)),
                grouped("http://s2:8000", Some("stable"), Some("90")),
            ]
        }

        fn registry(split: bool) -> PolicyRegistry {
            PolicyRegistry::new(PolicyConfig::RoundRobin).with_traffic_split(split)
        }

        fn pick(
            reg: &PolicyRegistry,
            workers: &[Arc<dyn Worker>],
            info: &SelectWorkerInfo,
        ) -> usize {
            reg.select_worker_for_model(&reg.get_default_policy(), "m", workers, info)
                .expect("a worker is selected")
        }

        fn picks(reg: &PolicyRegistry, workers: &[Arc<dyn Worker>], n: usize) -> Vec<usize> {
            let info = SelectWorkerInfo::default();
            (0..n).map(|_| pick(reg, workers, &info)).collect()
        }

        #[test]
        fn off_by_default_the_policy_runs_over_the_whole_pool() {
            let reg = registry(false);
            assert!(!reg.traffic_split_enabled());
            assert_eq!(picks(&reg, &fleet("10"), 6), vec![0, 1, 2, 0, 1, 2]);
        }

        #[test]
        fn one_group_runs_the_policy_over_the_whole_pool() {
            let reg = registry(true);
            let unlabelled = vec![
                grouped("http://a:8000", None, None),
                grouped("http://b:8000", None, None),
                grouped("http://c:8000", None, None),
            ];
            assert_eq!(picks(&reg, &unlabelled, 6), vec![0, 1, 2, 0, 1, 2]);
        }

        #[test]
        fn keyless_selections_follow_the_weights_and_the_policy_runs_inside_the_arm() {
            let reg = registry(true);
            let workers = fleet("10");
            let n = 3_000;
            let chosen = picks(&reg, &workers, n);
            let canary = chosen.iter().filter(|&&idx| idx == 1).count();
            // Mean 300, standard deviation 16.4: a 5-sigma band.
            assert!((218..=382).contains(&canary), "canary took {canary} of {n}");
            let s1 = chosen.iter().filter(|&&idx| idx == 0).count();
            let s2 = chosen.iter().filter(|&&idx| idx == 2).count();
            // Round robin alternates inside the stable arm.
            assert!(s1.abs_diff(s2) <= 1, "stable arm split {s1} / {s2}");
            assert_eq!(s1 + s2 + canary, n);
        }

        #[test]
        fn a_routing_key_keeps_a_session_in_one_group() {
            let reg = registry(true);
            let workers = fleet("30");
            let group_of = |idx: usize| if idx == 1 { "canary" } else { "stable" };
            let mut seen_canary = false;
            let mut seen_stable = false;
            for i in 0..200 {
                let headers = headers_with_key(&format!("conv-{i}"));
                let info = SelectWorkerInfo {
                    headers: Some(&headers),
                    ..Default::default()
                };
                let first = group_of(pick(&reg, &workers, &info));
                for _ in 0..5 {
                    assert_eq!(group_of(pick(&reg, &workers, &info)), first, "conv-{i}");
                }
                seen_canary |= first == "canary";
                seen_stable |= first == "stable";
            }
            assert!(seen_canary && seen_stable, "keys spread over both groups");
        }

        #[test]
        fn a_drained_group_takes_nothing_until_its_weight_returns() {
            let reg = registry(true);
            let drained = fleet("0");
            assert!(picks(&reg, &drained, 300).iter().all(|&idx| idx != 1));
            let ramped = fleet("100");
            let canary = picks(&reg, &ramped, 2_000)
                .iter()
                .filter(|&&idx| idx == 1)
                .count();
            // Weight 100 against 90: about 53 %.
            assert!(
                (900..=1_200).contains(&canary),
                "canary took {canary} of 2000"
            );
        }

        #[test]
        fn the_selected_index_is_into_the_callers_slice() {
            let reg = registry(true);
            // The canary is last, so an index into the arm's member list and
            // an index into the caller's slice differ.
            let workers = vec![
                grouped("http://s1:8000", Some("stable"), Some("0")),
                grouped("http://s2:8000", Some("stable"), Some("0")),
                grouped("http://c1:8000", Some("canary"), Some("10")),
            ];
            assert!(picks(&reg, &workers, 20).iter().all(|&idx| idx == 2));
        }

        #[test]
        fn an_explicit_target_worker_bypasses_the_split() {
            let reg = PolicyRegistry::new(PolicyConfig::ConsistentHashing).with_traffic_split(true);
            // The canary is drained; an explicit index still reaches it.
            let workers = fleet("0");
            let mut headers = HeaderMap::new();
            headers.insert("x-smg-target-worker", "1".parse().unwrap());
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            for _ in 0..10 {
                assert_eq!(pick(&reg, &workers, &info), 1);
            }
            // Under round robin the policy does not read the header: the
            // drained canary stays drained.
            let round_robin = registry(true);
            for _ in 0..30 {
                assert_ne!(pick(&round_robin, &workers, &info), 1);
            }
        }

        #[test]
        fn bucket_and_targeted_requests_are_exempt() {
            let plain = SelectWorkerInfo::default();
            assert!(PolicyRegistry::traffic_split_applies("round_robin", &plain));
            assert!(PolicyRegistry::traffic_split_applies("cache_aware", &plain));
            assert!(!PolicyRegistry::traffic_split_applies("bucket", &plain));
            let mut headers = HeaderMap::new();
            headers.insert("x-smg-target-worker", "0".parse().unwrap());
            let targeted = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            // Only consistent hashing reads the header; every other policy
            // ignores it, and the split holds.
            assert!(!PolicyRegistry::traffic_split_applies(
                "consistent_hashing",
                &targeted
            ));
            assert!(PolicyRegistry::traffic_split_applies(
                "round_robin",
                &targeted
            ));
        }

        #[test]
        fn the_sticky_override_pins_inside_the_keys_group() {
            let reg = PolicyRegistry::with_override(
                PolicyConfig::RoundRobin,
                RoutingKeyOverrideConfig {
                    enabled: true,
                    ..Default::default()
                },
            )
            .with_traffic_split(true);
            let workers = fleet("50");
            let group_of = |idx: usize| if idx == 1 { "canary" } else { "stable" };
            let mut pinned_in_canary = 0;
            for i in 0..100 {
                let headers = headers_with_key(&format!("session-{i}"));
                let info = SelectWorkerInfo {
                    headers: Some(&headers),
                    ..Default::default()
                };
                let first = pick(&reg, &workers, &info);
                // The pin holds, and it lies in the key's group.
                for _ in 0..4 {
                    assert_eq!(pick(&reg, &workers, &info), first, "session-{i}");
                }
                if group_of(first) == "canary" {
                    pinned_in_canary += 1;
                }
            }
            assert!(
                (10..=60).contains(&pinned_in_canary),
                "{pinned_in_canary} of 100 keys pinned in the canary"
            );
        }

        #[test]
        fn an_unavailable_group_is_left_out_of_the_draw() {
            let reg = registry(true);
            let workers = fleet("100");
            workers[1].set_status(WorkerStatus::Failed);
            // Only the stable arm is live: the policy runs over the pool and
            // never picks the failed canary.
            assert!(picks(&reg, &workers, 50).iter().all(|&idx| idx != 1));
        }

        /// Under a disaggregated pair the decode leg follows the group the
        /// prefill leg was split into: the pair never spans two groups while
        /// the decode pool has that group live, and a group drained on the
        /// decode side is not followed into.
        #[test]
        fn the_decode_leg_follows_the_prefills_group() {
            use crate::policies::traffic_split::group_of;

            let reg = registry(true);
            let policy = reg.get_default_policy();
            let prefill = fleet("50");
            let decode = fleet("50");
            let info = SelectWorkerInfo::default();
            let mut canary_pairs = 0;
            for _ in 0..400 {
                let (prefill_idx, group) = reg
                    .select_worker_for_model_split(&policy, "m", &prefill, &info, None)
                    .expect("a prefill worker is selected");
                let group = group.expect("the prefill leg was split");
                assert_eq!(group_of(prefill[prefill_idx].as_ref()), group);
                let (decode_idx, followed) = reg
                    .select_worker_for_model_split(&policy, "m", &decode, &info, Some(group))
                    .expect("a decode worker is selected");
                assert_eq!(group_of(decode[decode_idx].as_ref()), group);
                assert_eq!(followed, Some(group));
                if group == "canary" {
                    canary_pairs += 1;
                }
            }
            // Weight 50 against 90: about 36 % of the pairs, a wide band.
            assert!(
                (80..=220).contains(&canary_pairs),
                "{canary_pairs} canary pairs of 400"
            );
            let drained = fleet("0");
            for _ in 0..50 {
                let (decode_idx, _) = reg
                    .select_worker_for_model_split(&policy, "m", &drained, &info, Some("canary"))
                    .expect("a decode worker is selected");
                assert_ne!(decode_idx, 1, "a drained group was followed into");
            }
        }

        /// A prefill slice that has shrunk to one named group (its other
        /// groups unavailable or at capacity) is not split, yet still pins
        /// the decode leg to that group: a canary prefill keeps a canary
        /// decode rather than pairing across two builds. Off, no group is
        /// reported.
        #[test]
        fn a_prefill_slice_of_one_named_group_still_pins_the_decode_leg() {
            let reg = registry(true);
            let policy = reg.get_default_policy();
            let prefill = vec![grouped("http://c1:8000", Some("canary"), Some("10"))];
            let decode = fleet("10");
            let info = SelectWorkerInfo::default();
            for _ in 0..100 {
                let (prefill_idx, group) = reg
                    .select_worker_for_model_split(&policy, "m", &prefill, &info, None)
                    .expect("a prefill worker is selected");
                assert_eq!(prefill_idx, 0);
                assert_eq!(group, Some("canary"), "an unsplit named group is reported");
                let (decode_idx, followed) = reg
                    .select_worker_for_model_split(&policy, "m", &decode, &info, group)
                    .expect("a decode worker is selected");
                assert_eq!(decode_idx, 1, "the decode leg left the canary");
                assert_eq!(followed, Some("canary"));
            }
            let off = registry(false);
            let (_, group) = off
                .select_worker_for_model_split(
                    &off.get_default_policy(),
                    "m",
                    &prefill,
                    &info,
                    None,
                )
                .expect("a prefill worker is selected");
            assert_eq!(group, None, "off, the labels are inert");
        }

        /// A prefill pool of the default group alone is not split and reports
        /// no group, so the decode leg decides for itself: a canary only the
        /// decode pool carries keeps its share next to an unlabelled fleet.
        #[test]
        fn a_decode_only_canary_keeps_its_share_next_to_an_unsplit_prefill_pool() {
            let reg = registry(true);
            let policy = reg.get_default_policy();
            let prefill: Vec<Arc<dyn Worker>> = (0..3)
                .map(|i| grouped(&format!("http://p{i}:8000"), None, None))
                .collect();
            let decode = vec![
                grouped("http://d1:8000", None, None),
                grouped("http://c1:8000", Some("canary"), Some("50")),
                grouped("http://d2:8000", None, None),
            ];
            let info = SelectWorkerInfo::default();
            let mut canary_decodes = 0;
            for _ in 0..400 {
                let (_, group) = reg
                    .select_worker_for_model_split(&policy, "m", &prefill, &info, None)
                    .expect("a prefill worker is selected");
                assert_eq!(group, None, "a single-group pool was split");
                let (decode_idx, _) = reg
                    .select_worker_for_model_split(&policy, "m", &decode, &info, group)
                    .expect("a decode worker is selected");
                if decode_idx == 1 {
                    canary_decodes += 1;
                }
            }
            // Weight 50 against the default group's 100: about a third of
            // the decodes, a wide band.
            assert!(
                (70..=200).contains(&canary_decodes),
                "{canary_decodes} canary decodes of 400"
            );
        }
    }
}
