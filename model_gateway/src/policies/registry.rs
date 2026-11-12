use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, AtomicU8, Ordering},
        Arc, OnceLock, Weak,
    },
};

use dashmap::DashMap;
use http::{header::HeaderName, HeaderMap};
use parking_lot::{Mutex, RwLock};
use smg_policy_controller::{Needs, Phase};
use tracing::{debug, error, info, warn};

/// Policy Registry for managing model-to-policy mappings
///
/// This registry manages the dynamic assignment of load balancing policies to models.
/// When the first worker of a new model is added, it determines the policy for that model.
/// All subsequent workers of the same model use the established policy.
/// When the last worker of a model is removed, the policy mapping is cleaned up.
use super::{
    as_bucket, as_cache_aware,
    engine_host::{wants_text, Decider},
    get_healthy_worker_indices,
    manual::{ExecutionBranch, PinState},
    normalize_model_key, DPRankLoadPolicy, LoadBalancingPolicy, ManualConfig, ManualPolicy,
    PolicyFactory, SelectWorkerInfo, WorkerRole,
};
use crate::{
    config::types::{
        ManualAssignmentMode, PdPairingMode, PolicyConfig, PolicyEngineMode,
        RoutingKeyOverrideConfig, RoutingMode, DEFAULT_ENCODE_POLICY,
    },
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
            .on_request_complete(worker.url());
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
        Self::from_answers(
            policy.needs_request_text(),
            policy.needs_request_text_with_tokens(),
            !PolicyRegistry::routing_key_override_applies(policy.name()),
        )
    }

    /// What the engine answers for the policies it serves, from its needs
    /// for a phase: the request text or either kind of prefix hash wants the
    /// body, and the token hashes alone are computed from a request's token
    /// ids when it carries them. `keeps_text_under_key` is the routing-key
    /// override's rule for the served policy's name.
    fn of_needs(needs: Needs, keeps_text_under_key: bool) -> Self {
        Self::from_answers(
            wants_text(needs),
            needs.text || needs.text_hashes,
            keeps_text_under_key,
        )
    }

    fn from_answers(text: bool, text_with_tokens: bool, keeps_text_under_key: bool) -> Self {
        let mut bits = 0;
        for (wants, any, under_key) in [
            (text, Self::TEXT, Self::TEXT_UNDER_KEY),
            (
                text_with_tokens,
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

    /// Whether a placement is handed the model's hash ring under
    /// `--policy-engine engine` ([`Self::hash_ring_wanted`]), recomputed
    /// with `text_needs`; the legacy path reads it for every policy.
    ring_wanted: Arc<AtomicBool>,

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
    /// The sticky routing-key override as configured: the policy engine's
    /// default binding takes its sessions rule from it.
    routing_key_override: RoutingKeyOverrideConfig,
    /// How strictly PD placement pairs prefill and decode descriptors.
    pd_pairing_mode: PdPairingMode,
    /// The policy engine, when `--policy-engine engine` selected it: the one
    /// handle a placement site decides through (none does yet). A later
    /// step's remote mode puts the controller client and the local fallback
    /// engine behind this same handle.
    decider: Option<Arc<Decider>>,
}

/// A sticky key with this many of its own requests already in flight on its
/// pinned worker (router-local, counted per replica) is bypassed and the
/// request reassigned. Guards one conversation stacking onto its pin; total
/// worker load is not a respill trigger. The policy engine's override rule
/// carries the same cap.
pub(crate) const STICKY_INFLIGHT_CAP: u16 = 2;

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
            ring_wanted: Arc::new(AtomicBool::new(true)),
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
            routing_key_override,
            pd_pairing_mode: PdPairingMode::default(),
            decider: None,
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

    /// Select the implementation that decides (`--policy-engine`): under
    /// `engine` the policy engine is built from the default policy and the
    /// two pair-side policies. The validator builds the same decider at
    /// start and refuses what the engine refuses, so a refusal here is a
    /// construction that skipped validation; it keeps the legacy path and
    /// says so at error level.
    #[must_use]
    pub fn with_policy_engine(
        self,
        mode: PolicyEngineMode,
        prefill: &PolicyConfig,
        decode: &PolicyConfig,
    ) -> Self {
        self.with_policy_engines(mode, prefill, decode, None)
    }

    /// [`Self::with_policy_engine`] for a routing mode: the pair sides as
    /// the mode resolves them over `default` and, under
    /// encode-prefill-decode, the mode's encode policy (the one default when
    /// unset), so the engine binds the encode profile the encode items are
    /// decided through.
    #[must_use]
    pub fn with_policy_engine_for(
        self,
        mode: PolicyEngineMode,
        routing: &RoutingMode,
        default: &PolicyConfig,
    ) -> Self {
        let encode = matches!(routing, RoutingMode::EncodePrefillDecode { .. })
            .then(|| routing.get_encode_policy(&DEFAULT_ENCODE_POLICY));
        self.with_policy_engines(
            mode,
            routing.get_prefill_policy(default),
            routing.get_decode_policy(default),
            encode,
        )
    }

    fn with_policy_engines(
        mut self,
        mode: PolicyEngineMode,
        prefill: &PolicyConfig,
        decode: &PolicyConfig,
        encode: Option<&PolicyConfig>,
    ) -> Self {
        if mode == PolicyEngineMode::Engine {
            match Decider::from_configs_with_encode(
                &self.default_policy_config,
                prefill,
                decode,
                encode,
                &self.routing_key_override,
            ) {
                Ok(decider) => self.host(decider),
                Err(error) => {
                    error!(%error, "The policy engine refused a configuration the validator did not see; the legacy policies decide");
                }
            }
        }
        self
    }

    /// Host `decider` as the implementation that decides. This is the apply:
    /// the engine's needs, read once here, set what the host feeds before
    /// any request exists (the text questions and the ring bit through
    /// [`Self::refresh_text_needs`]; the load poll and each worker's stream
    /// kind read them through [`Self::needs_backend_loads`] and
    /// [`Self::worker_needs_kv_events`]).
    fn host(&mut self, decider: Decider) {
        info!(
            needs = ?decider.engine().needs(),
            "The policy engine is hosted behind --policy-engine engine"
        );
        decider.set_load_receiver(self.load_rx.read().clone());
        self.decider = Some(Arc::new(decider));
        self.refresh_text_needs();
    }

    /// A registry hosting `decider` as `--policy-engine engine` does, for an
    /// engine built from a document instead of the policy configuration.
    #[cfg(test)]
    pub(crate) fn with_decider(mut self, decider: Decider) -> Self {
        self.host(decider);
        self
    }

    /// The policy engine, when it decides.
    pub fn decider(&self) -> Option<&Arc<Decider>> {
        self.decider.as_ref()
    }

    /// Whether the engine, when it decides, stands for `policy`: the default
    /// policy by name (what [`Self::engine_for`] serves) and the plan's
    /// sides, whose profiles the engine was built from. A model whose
    /// registration hint named another policy keeps that policy's legacy
    /// instance, and the static needs are answered for it by the trait.
    fn served_by_engine(&self, policy: &dyn LoadBalancingPolicy) -> bool {
        self.decider.is_some()
            && (policy.name() == self.default_policy.name()
                || [
                    &self.prefill_policy,
                    &self.decode_policy,
                    &self.encode_policy,
                ]
                .into_iter()
                .filter_map(|side| side.get())
                .any(|side| std::ptr::addr_eq(Arc::as_ptr(side), policy)))
    }

    /// The policy engine for a regular placement under `policy`
    /// (`--policy-engine engine`), or `None` when the legacy path decides.
    /// The engine is built from the default policy and serves it by name: a
    /// model whose registration hint named another policy keeps that policy's
    /// legacy instance, and a hint naming the default's own preset shares the
    /// engine's one instance of it. A keyed request under the default policy
    /// is the engine's too: its binding carries the sticky routing-key
    /// override's sessions rule, or the manual policy's own.
    pub(crate) fn engine_for(&self, policy: &dyn LoadBalancingPolicy) -> Option<&Arc<Decider>> {
        let decider = self.decider.as_ref()?;
        (policy.name() == self.default_policy.name()).then_some(decider)
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

    /// The header the gateway sets a key under itself: the first configured
    /// routing-key name, the one [`Self::resolve_routing_key`] reads first.
    pub(crate) fn routing_key_header(&self) -> Option<&HeaderName> {
        self.routing_key_headers.first()
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

    /// The one header-derived key: the resolved routing key, lineage-stripped
    /// when the override is enabled, raw otherwise. Placement hands it to
    /// every policy as `SelectWorkerInfo::routing_key` and the load guard
    /// counts under it, so selection and accounting agree by construction.
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
    /// wins, with the rid-derived key as fallback. `info.routing_key` is the
    /// header key as placement derived it ([`Self::sticky_header_key`]:
    /// lineage-stripped under the override, so a proxy forwarding `conv_t2`
    /// pins the entry `conv`); a caller that left it empty gets the same
    /// derivation from the headers here. Nothing strips twice.
    fn effective_sticky_key<'a>(
        &self,
        info: &SelectWorkerInfo<'a>,
    ) -> Option<(&'a str, &'static str)> {
        info.routing_key
            .or_else(|| self.sticky_header_key(info.headers))
            .map(|key| (key, "header"))
            .or_else(|| info.rid_key.map(|key| (key, "rid")))
    }

    /// The sticky routing-key override's selector, with the request's key and
    /// its source, when the override places this request: it is enabled,
    /// `policy` does not honor the key itself (`manual` and
    /// `consistent_hashing` read `rid_key` and the header themselves with the
    /// same header-first precedence), and the request carries a key from a
    /// configured source. The one predicate the keyed path is entered by; the
    /// policy's name is read only when the override is enabled.
    fn sticky_for<'a>(
        &self,
        policy: &dyn LoadBalancingPolicy,
        info: &SelectWorkerInfo<'a>,
    ) -> Option<(&Arc<ManualPolicy>, &'a str, &'static str)> {
        let sticky = self.routing_key_sticky.as_ref()?;
        if !Self::routing_key_override_applies(policy.name()) {
            return None;
        }
        let (key, source) = self.effective_sticky_key(info)?;
        Some((sticky, key, source))
    }

    /// Select a worker, applying the sticky routing-key override when it
    /// places the request (`sticky_for`); otherwise delegates to
    /// `policy`. `policy.name()` stays the real policy (for metrics).
    pub fn select_worker_for_model(
        &self,
        policy: &Arc<dyn LoadBalancingPolicy>,
        model_id: &str,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        cache_trace::begin_selection(workers);
        if let Some((sticky, key, source)) = self.sticky_for(policy.as_ref(), info) {
            return Self::select_sticky(sticky, policy, model_id, workers, info, key, source);
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

        let over_cap = |idx: usize| {
            workers[idx].routing_key_inflight(load_key) >= usize::from(STICKY_INFLIGHT_CAP)
        };
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

        let delegate = sticky.assignment_mode() == ManualAssignmentMode::Delegate;

        // A hit reads the pin's candidates alone; the eligible list is built
        // only where a worker has to be chosen from it.
        let pin = sticky.peek_pin(workers, key, |idx| {
            info.admits(workers[idx].routing_state())
        });
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
                // The pin was admitted by a live read above; the pool can lose
                // every eligible worker before this second read.
                let healthy = get_healthy_worker_indices(workers, info);
                if healthy.is_empty() {
                    return finish(None, ExecutionBranch::NoHealthyWorkers);
                }
                sticky.assign_index(workers, &healthy)
            };
            if new_idx != idx && !over_cap(new_idx) {
                sticky.pin_front(key, workers[new_idx].url());
            }
            return finish(Some(new_idx), ExecutionBranch::CapRespill);
        }

        let healthy = get_healthy_worker_indices(workers, info);
        if healthy.is_empty() {
            return finish(None, ExecutionBranch::NoHealthyWorkers);
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
        if let Some(cache_aware) = as_cache_aware(policy.as_ref()) {
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
        if let Some(decider) = &self.decider {
            decider.set_load_receiver(rx);
        }
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
    /// wired in and its mesh flag raised (both flip
    /// atomically inside [`super::CacheAwarePolicy::set_mesh_tree_sync`]); every
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
    /// The setter also flips the policy's mesh flag to match adapter
    /// presence, so callers do not need to touch that flag directly.
    fn maybe_inject_mesh_tree_sync(
        policy: &Arc<dyn LoadBalancingPolicy>,
        adapter: Option<&Arc<TreeSyncAdapter>>,
    ) {
        if let Some(cache_aware) = as_cache_aware(policy.as_ref()) {
            cache_aware.set_mesh_tree_sync(adapter.cloned());
        }
    }

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
    /// reports and in-flight reconciliation both go there, and the worker's
    /// stream is chosen by what it needs (see `UpdatePoliciesStep`).
    pub(crate) fn policy_for_worker(&self, worker: &dyn Worker) -> Arc<dyn LoadBalancingPolicy> {
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

    /// The policy for `model_id`'s requests: built for the model's first
    /// worker (from `policy_hint`, its `policy` label) and returned as it is
    /// for every later one, until no worker of this node serves the model
    /// ([`Self::on_model_unserved`]). Registering or replacing a worker of
    /// the model calls this as often as it likes.
    pub fn on_worker_added(
        &self,
        model_id: &str,
        policy_hint: Option<&str>,
    ) -> Arc<dyn LoadBalancingPolicy> {
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

    /// Called once no worker of this node serves `model_id` any more (see
    /// `WorkerRegistry::serves_model_locally`): the model's policy goes with
    /// its last worker, so a later registration builds the policy its own
    /// labels name.
    pub fn on_model_unserved(&self, model_id: &str) {
        if let Some((_, policy)) = self.model_policies.remove(model_id) {
            info!(
                "Removed policy {} for model {} (no worker of this node serves it)",
                policy.name(),
                model_id
            );
            self.refresh_text_needs();
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

    /// Clear all policies (useful for testing)
    pub fn clear(&self) {
        self.model_policies.clear();
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
            .unwrap_or_else(|| PolicyFactory::create_from_config(&DEFAULT_ENCODE_POLICY))
    }

    /// The policy engine for the encode items of an encode-prefill-decode
    /// request (`--policy-engine engine` with the encode profile bound), or
    /// `None` when the legacy encode policy decides them.
    pub(crate) fn encode_engine_for(&self) -> Option<&Arc<Decider>> {
        self.decider
            .as_ref()
            .filter(|decider| decider.serves_encode())
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
    /// range over: the default policy and the per-model policies. Under
    /// `--policy-engine engine` the engine answers for the policies it
    /// serves, from its needs for a regular placement, and a per-model
    /// policy it does not serve keeps the trait's answer; the ring bit
    /// ([`Self::hash_ring_wanted`]) is recomputed with them. Called where
    /// the per-model map changes; one recomputation at a time, so the bits
    /// published last are those of a walk that saw every earlier change.
    fn refresh_text_needs(&self) {
        let _one_at_a_time = self.text_needs_refresh.lock();
        let (mut needs, mut ring) = match &self.decider {
            Some(decider) => {
                let engine = decider.engine();
                (
                    TextNeeds::of_needs(
                        engine.needs_for_phase(Phase::Single),
                        !Self::routing_key_override_applies(self.default_policy.name()),
                    ),
                    engine.needs().ring,
                )
            }
            None => (TextNeeds::of(self.default_policy.as_ref()), true),
        };
        for entry in self.model_policies.iter() {
            let policy = entry.value().as_ref();
            if !self.served_by_engine(policy) {
                needs = needs.join(TextNeeds::of(policy));
                // The trait has no ring question: a policy the engine does
                // not serve keeps the ring, as every policy does today.
                ring = true;
            }
        }
        self.text_needs.store(needs.0, Ordering::Release);
        self.ring_wanted.store(ring, Ordering::Release);
    }

    /// Whether any policy routes on backend load (see
    /// [`LoadBalancingPolicy::needs_backend_loads`]): with load monitoring
    /// off, the worker monitor still polls a group while this holds. Under
    /// `--policy-engine engine` the engine's needs answer for the policies
    /// it serves, the trait for the rest.
    pub fn needs_backend_loads(&self) -> bool {
        let mut needs = self
            .decider
            .as_ref()
            .is_some_and(|decider| decider.engine().needs().loads);
        self.each_policy(|policy| {
            if !self.served_by_engine(policy.as_ref()) {
                needs |= policy.needs_backend_loads();
            }
        });
        needs
    }

    /// Whether the prefill or the decode policy reads the request text: the
    /// pair routers' body question. Under `--policy-engine engine` the
    /// engine's needs for the two pair phases answer, else the two policies.
    pub(crate) fn pair_needs_request_text(&self) -> bool {
        match &self.decider {
            Some(decider) => [Phase::Prefill, Phase::Decode]
                .into_iter()
                .any(|phase| wants_text(decider.engine().needs_for_phase(phase))),
            None => {
                self.get_prefill_policy().needs_request_text()
                    || self.get_decode_policy().needs_request_text()
            }
        }
    }

    /// Whether `worker`'s stream carries its KV events into its model's
    /// index: what the policy that places requests on it reads
    /// ([`Self::policy_for_worker`]), from the engine's needs for the
    /// worker's model and phase when the engine serves that policy.
    pub(crate) fn worker_needs_kv_events(&self, worker: &dyn Worker) -> bool {
        let policy = self.policy_for_worker(worker);
        match &self.decider {
            Some(decider) if self.served_by_engine(policy.as_ref()) => {
                let phase = match worker.worker_type() {
                    WorkerType::Prefill => Phase::Prefill,
                    WorkerType::Decode => Phase::Decode,
                    WorkerType::Encode => Phase::Encode,
                    WorkerType::Regular => Phase::Single,
                };
                decider.engine().needs_for(worker.model_id(), phase).events
            }
            _ => policy.needs_kv_events(),
        }
    }

    /// Whether a placement is handed its model's hash ring: always on the
    /// legacy path, whose trait has no question for it; under
    /// `--policy-engine engine` when the engine's needs say so or a policy
    /// the engine does not serve may read it.
    pub(crate) fn hash_ring_wanted(&self) -> bool {
        self.decider.is_none() || self.ring_wanted.load(Ordering::Acquire)
    }

    /// Initialize cache-aware policy with workers if applicable
    /// This should be called after workers are registered for a model
    pub fn init_cache_aware_policy(&self, model_id: &str, workers: &[Arc<dyn Worker>]) {
        // Get the policy for this model
        if let Some(policy) = self.get_policy(model_id) {
            if let Some(cache_aware) = as_cache_aware(policy.as_ref()) {
                debug!(
                    "Initializing cache-aware policy with {} workers for model {}",
                    workers.len(),
                    model_id
                );
                cache_aware.init_workers(workers);
            }
        }
    }

    /// Remove a worker from cache-aware policy if applicable
    /// This should be called when a worker is being removed
    pub fn remove_worker_from_cache_aware(&self, model_id: &str, worker_url: &str) {
        // Get the policy for this model
        if let Some(policy) = self.get_policy(model_id) {
            if let Some(cache_aware) = as_cache_aware(policy.as_ref()) {
                cache_aware.remove_worker_by_url(worker_url);
                debug!(
                    "Removed worker {} from cache-aware policy for model {}",
                    worker_url, model_id
                );
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
                if let Some(cache_aware) = as_cache_aware(policy.as_ref()) {
                    cache_aware.remove_worker_by_url(worker_url);
                    debug!(
                        "Removed worker {} from {} cache-aware policy",
                        worker_url, worker_type
                    );
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
            if let Some(cache_aware) = as_cache_aware(policy.as_ref()) {
                cache_aware.remove_workers_from_model(model_id, worker_urls);
                debug!(
                    "Removed {} workers from cache-aware policy for model {}",
                    worker_urls.len(),
                    model_id
                );
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

    /// Drop what a removed worker left behind: the policies' per-worker
    /// state by URL (`remove_worker_from_policies`) and, under
    /// `--policy-engine engine`, its rows in the engine's accounting
    /// (`Decider::on_worker_removed`).
    pub fn on_worker_removed(&self, worker: &dyn Worker) {
        self.remove_worker_from_policies(worker.url());
        if let Some(decider) = &self.decider {
            decider.on_worker_removed(worker);
        }
    }

    /// A same-URL replacement keeps the policies' per-URL state (the
    /// replacement inherits the shared runtime) and, under `--policy-engine
    /// engine`, retires the old handle's rows in the engine's accounting
    /// (`Decider::on_worker_replaced`).
    pub fn on_worker_replaced(&self, old: &dyn Worker, new: &dyn Worker) {
        if let Some(decider) = &self.decider {
            decider.on_worker_replaced(old, new);
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
            if let Some(cache_aware) = as_cache_aware(prefill_policy.as_ref()) {
                if !prefill_workers.is_empty() {
                    debug!(
                        "Initializing prefill cache-aware policy with {} workers",
                        prefill_workers.len()
                    );
                    cache_aware.init_workers(prefill_workers);
                }
            }
        }

        // Initialize decode policy if it's cache-aware (lock-free via OnceLock::get)
        if let Some(decode_policy) = self.decode_policy.get() {
            if let Some(cache_aware) = as_cache_aware(decode_policy.as_ref()) {
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

    /// Initialize bucket policies for PD mode - lock-free
    pub fn init_pd_bucket_policies(&self, prefill_workers: &[Arc<dyn Worker>]) {
        // Initialize prefill policy if it's bucket (lock-free via OnceLock::get)
        if let Some(prefill_policy) = self.prefill_policy.get() {
            if let Some(bucket) = as_bucket(prefill_policy.as_ref()) {
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

impl std::fmt::Debug for PolicyRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyRegistry")
            .field("model_policies", &self.model_policies)
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
        policies::{
            engine_host::HostRequest,
            test_support::{decider_over, headers_with_routing_key},
            BucketConfig, BucketPolicy, CacheAwareConfig, CacheAwarePolicy, LeastLoadPolicy,
            SelectWorkerInfo,
        },
        worker::{
            load_state::{one_rank_load, LoadSnapshot},
            test_support::{ready_worker, worker_of},
            HashRing, Worker, WorkerLoadGuard, WorkerType,
        },
    };

    /// Under `--policy-engine engine` the static needs the host answers
    /// before any request exists come from the engine, not from the legacy
    /// policy's flags: a cache-aware default policy reads text, loads and KV
    /// events; the engine it is served by here reads none of them. A model
    /// whose hint names a policy the engine does not serve keeps the trait's
    /// answers beside the engine's: the ring for any such policy, the text
    /// where it reads text.
    #[test]
    fn under_the_engine_the_static_needs_are_the_engines_not_the_legacy_flags() {
        let cache_aware = PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096);
        let legacy = PolicyRegistry::new(cache_aware.clone());
        let engine = PolicyRegistry::new(cache_aware).with_decider(decider_over(&[], &[], &[]));
        let regular = worker_of("http://w1:8000", WorkerType::Regular);
        for (registry, expected) in [(&legacy, true), (&engine, false)] {
            assert_eq!(registry.any_policy_needs_request_text(None), expected);
            assert_eq!(registry.needs_backend_loads(), expected);
            assert_eq!(registry.worker_needs_kv_events(regular.as_ref()), expected);
            assert_eq!(registry.pair_needs_request_text(), expected);
            assert_eq!(registry.hash_ring_wanted(), expected);
        }

        engine.on_worker_added("other", Some("random"));
        assert!(
            engine.hash_ring_wanted(),
            "a policy the engine does not serve may read the ring"
        );
        assert!(
            !engine.any_policy_needs_request_text(None),
            "random reads no text"
        );
        engine.on_worker_added("third", Some("bucket"));
        assert!(
            engine.any_policy_needs_request_text(None),
            "bucket reads the text"
        );
        assert!(!engine.needs_backend_loads(), "neither reads loads");
    }

    /// The engine's needs are read per phase and per worker role: a
    /// prefix-affinity default profile wants the body for its token hashes,
    /// which a request's token ids satisfy, and the KV events of a regular
    /// worker; a least-load prefill profile wants the load poll and no text
    /// on the pair paths, and a prefill or decode worker streams no events.
    #[test]
    fn the_engines_needs_are_read_per_phase_and_per_worker_role() {
        let registry = PolicyRegistry::new(PolicyConfig::Random).with_decider(decider_over(
            &["prefix-affinity"],
            &["least-load"],
            &[],
        ));
        assert!(registry.any_policy_needs_request_text(None));
        assert!(!registry.any_policy_needs_request_text_with_tokens(None));
        assert!(!registry.pair_needs_request_text());
        assert!(registry.needs_backend_loads());
        assert!(!registry.hash_ring_wanted());
        let events = |worker_type| {
            registry.worker_needs_kv_events(worker_of("http://w1:8000", worker_type).as_ref())
        };
        assert!(events(WorkerType::Regular));
        assert!(!events(WorkerType::Prefill));
        assert!(!events(WorkerType::Decode));
    }

    /// `--policy-engine engine` puts the engine behind `decider()`; the legacy
    /// path leaves it empty, and a configuration the engine cannot serve keeps
    /// the legacy path (the validator refuses it at start).
    #[test]
    fn the_engine_switch_builds_the_decider_and_the_legacy_path_leaves_it_empty() {
        let round_robin = PolicyConfig::RoundRobin;
        let legacy = PolicyRegistry::new(round_robin.clone()).with_policy_engine(
            PolicyEngineMode::Legacy,
            &round_robin,
            &round_robin,
        );
        assert!(legacy.decider().is_none());
        let engine = PolicyRegistry::new(round_robin.clone()).with_policy_engine(
            PolicyEngineMode::Engine,
            &round_robin,
            &round_robin,
        );
        assert!(engine.decider().is_some());
        let unserved = PolicyRegistry::new(round_robin.clone()).with_policy_engine(
            PolicyEngineMode::Engine,
            &PolicyConfig::ConsistentHashing,
            &round_robin,
        );
        assert!(unserved.decider().is_none());
    }

    /// The load feed handed to the registry reaches the engine whether the
    /// decider exists before or after the hand-off. Worker 1 carries two
    /// in-flight requests, so a dark pool (no feed) takes worker 0 by the
    /// host count and only the reports flip the pick to the idle worker 1.
    #[test]
    fn the_load_feed_hand_off_reaches_the_decider() {
        let least_load = PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        };
        let workers = vec![
            ready_worker("http://w0:8000", "m"),
            ready_worker("http://w1:8000", "m"),
        ];
        workers[1].increment_load();
        workers[1].increment_load();
        let (_tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
            (
                workers[0].url().to_string(),
                one_rank_load(4096, 0.5, 2000.0),
            ),
            (workers[1].url().to_string(), one_rank_load(0, 0.1, 2000.0)),
        ]));
        let pick = |registry: &PolicyRegistry| {
            let decider = registry.decider().expect("the engine");
            let request = HostRequest {
                model: "m",
                ..HostRequest::default()
            };
            let (decision, _) = decider.decide(&request, &workers, &[]);
            let first = decision.pick().expect("a pick").targets.first().copied();
            first.map(|t| t.position)
        };
        let idle_wins = |registry: &PolicyRegistry| {
            assert_eq!(pick(registry), Some(1), "the reports flip the pick");
        };

        let dark = PolicyRegistry::new(least_load.clone()).with_policy_engine(
            PolicyEngineMode::Engine,
            &least_load,
            &least_load,
        );
        assert_eq!(
            pick(&dark),
            Some(0),
            "without the feed the host count decides"
        );

        let handed_first = PolicyRegistry::new(least_load.clone());
        handed_first.set_load_receiver(Some(rx.clone()));
        let handed_first =
            handed_first.with_policy_engine(PolicyEngineMode::Engine, &least_load, &least_load);
        idle_wins(&handed_first);

        let handed_after = PolicyRegistry::new(least_load.clone()).with_policy_engine(
            PolicyEngineMode::Engine,
            &least_load,
            &least_load,
        );
        handed_after.set_load_receiver(Some(rx));
        idle_wins(&handed_after);
    }

    fn cache_aware_policy() -> Arc<dyn LoadBalancingPolicy> {
        Arc::new(CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        }))
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
            worker_of("http://w1", WorkerType::Regular),
            worker_of("http://w2", WorkerType::Regular),
            worker_of("http://w3", WorkerType::Regular),
        ];
        let headers = headers_with_routing_key("session-A");
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
                worker_of("http://a1", WorkerType::Regular),
                worker_of("http://a2", WorkerType::Regular),
            ],
            vec![worker_of("http://b1", WorkerType::Regular)],
            vec![worker_of("http://c1", WorkerType::Regular)],
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
                worker_of("http://w1", WorkerType::Regular),
                worker_of("http://w2", WorkerType::Regular),
                worker_of("http://w3", WorkerType::Regular),
                worker_of("http://w4", WorkerType::Regular),
            ];
            let hash_ring = Some(Arc::new(HashRing::new(workers.iter().map(|w| w.url()))));
            let headers = headers_with_routing_key("conversation-42");
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
            worker_of("http://w1", WorkerType::Regular),
            worker_of("http://w2", WorkerType::Regular),
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
        assert!(!reg.any_policy_needs_request_text(Some(&headers_with_routing_key("k"))));
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
        assert!(without.any_policy_needs_request_text(Some(&headers_with_routing_key("session-A"))));

        let with = PolicyRegistry::with_override(cache_aware_config(), enabled_override());
        assert!(with.any_policy_needs_request_text(None));
        assert!(!with.any_policy_needs_request_text(Some(&headers_with_routing_key("session-A"))));
        let over_cap = "k".repeat(129);
        assert!(with.any_policy_needs_request_text(Some(&headers_with_routing_key(&over_cap))));
    }

    #[test]
    fn per_model_text_policy_scanned_with_the_same_hint_rules() {
        let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
        assert!(!reg.any_policy_needs_request_text(None));
        reg.on_worker_added("llama-3", Some("cache_aware"));
        assert!(reg.any_policy_needs_request_text(None));
        assert!(!reg.any_policy_needs_request_text(Some(&headers_with_tokens("1,2,3"))));
        assert!(!reg.any_policy_needs_request_text(Some(&headers_with_routing_key("session-A"))));
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
        reg.on_model_unserved("affine");
        assert!(!reg.any_policy_needs_request_text(None));
        reg.on_model_unserved("plain");
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
        let keyed = headers_with_routing_key("session-A");
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
            worker_of("http://w1", WorkerType::Regular),
            worker_of("http://w2", WorkerType::Regular),
        ];
        let headers = headers_with_routing_key("session-A");
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

        let legacy_only = headers_with_routing_key("legacy");
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
            reg.resolve_routing_key(Some(&headers_with_routing_key("legacy"))),
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
            reg.sticky_header_key(Some(&headers_with_routing_key("conv_t2"))),
            Some("conv")
        );
        assert_eq!(
            reg.sticky_header_key(Some(&headers_with_routing_key("conv_t2_r1"))),
            Some("conv")
        );
        // A key that is nothing but suffix keys as itself.
        assert_eq!(
            reg.sticky_header_key(Some(&headers_with_routing_key("_t1"))),
            Some("_t1")
        );

        // Override disabled: raw key, no stripping.
        let disabled = PolicyRegistry::new(PolicyConfig::RoundRobin);
        assert_eq!(
            disabled.sticky_header_key(Some(&headers_with_routing_key("conv_t2"))),
            Some("conv_t2")
        );
    }

    #[test]
    fn header_turns_pin_the_same_sticky_entry() {
        let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
        let policy = reg.get_default_policy();
        let workers = vec![
            worker_of("http://w1", WorkerType::Regular),
            worker_of("http://w2", WorkerType::Regular),
            worker_of("http://w3", WorkerType::Regular),
        ];
        let t1 = headers_with_routing_key("conv_t1");
        let info_t1 = SelectWorkerInfo {
            headers: Some(&t1),
            ..Default::default()
        };
        let first = reg.select_worker(&policy, &workers, &info_t1).unwrap();
        let t2 = headers_with_routing_key("conv_t2");
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
            worker_of("http://w1", WorkerType::Regular),
            worker_of("http://w2", WorkerType::Regular),
            worker_of("http://w3", WorkerType::Regular),
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

    /// A round-robin registry whose routing-key override delegates the assignment, its
    /// default policy and the two regular workers the sticky-selection tests route over.
    fn delegating_registry_with_two_workers() -> (
        PolicyRegistry,
        Arc<dyn LoadBalancingPolicy>,
        Vec<Arc<dyn Worker>>,
    ) {
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let workers = vec![
            worker_of("http://w1", WorkerType::Regular),
            worker_of("http://w2", WorkerType::Regular),
        ];
        (reg, policy, workers)
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
        let (reg, policy, workers) = delegating_registry_with_two_workers();

        let headers = headers_with_routing_key("conversation-42");
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
            headers_with_routing_key(""),
            headers_with_routing_key(&"k".repeat(ROUTING_KEY_HINT_MAX_BYTES + 1)),
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
        let (reg, policy, workers) = delegating_registry_with_two_workers();
        let headers = headers_with_routing_key("conversation_t1");
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
            assert_eq!(
                selected,
                usize::from(guards.len() == usize::from(STICKY_INFLIGHT_CAP))
            );
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
            worker_of("http://w1", WorkerType::Regular),
            worker_of("http://w2", WorkerType::Regular),
        ];
        as_cache_aware(policy.as_ref())
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
        let (reg, policy, workers) = delegating_registry_with_two_workers();
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
        let (reg, policy, fleet) = delegating_registry_with_two_workers();
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
        let (reg, policy, workers) = delegating_registry_with_two_workers();
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
        let (reg, policy, workers) = delegating_registry_with_two_workers();
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
        let (reg, policy, workers) = delegating_registry_with_two_workers();
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
            worker_of("http://w1", WorkerType::Regular),
            worker_of("http://w2", WorkerType::Regular),
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
        let (reg, policy, workers) = delegating_registry_with_two_workers();
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
        let (reg, policy, workers) = delegating_registry_with_two_workers();
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
        let workers = vec![worker_of("http://w1", WorkerType::Regular)];
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
    }

    #[test]
    fn test_policy_registry_cleanup() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // A model's policy outlives any number of adds: a registration or a
        // replacement of a worker adds again.
        registry.on_worker_added("llama-3", Some("cache_aware"));
        registry.on_worker_added("llama-3", None);
        registry.on_worker_added("llama-3", None);
        assert!(registry.get_policy("llama-3").is_some());

        // It goes once no worker serves the model, and the next first
        // worker's hint builds the next one.
        registry.on_model_unserved("llama-3");
        assert!(registry.get_policy("llama-3").is_none());
        let policy = registry.on_worker_added("llama-3", Some("round_robin"));
        assert_eq!(policy.name(), "round_robin");
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
            worker_of("http://a:8000", WorkerType::Regular),
            worker_of("http://b:8000", WorkerType::Regular),
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

        let config = as_cache_aware(policy.as_ref()).unwrap().config_for_test();
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
        let alias_config = as_cache_aware(alias.as_ref()).unwrap().config_for_test();
        assert_eq!(alias_config.max_tree_size, 4096);
    }

    #[test]
    fn hinted_policy_inherits_operator_least_load_config() {
        let registry = PolicyRegistry::new(PolicyConfig::LeastLoad {
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
                let cache_aware = as_cache_aware(entry.value().as_ref()).unwrap();
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
            let cache_aware = as_cache_aware(policy.as_ref()).unwrap();
            assert!(
                cache_aware.kv_event_monitor_is_set_for_test(),
                "{role} policy missed the KV event monitor"
            );
            assert!(
                cache_aware.has_load_receiver_for_test(),
                "{role} policy missed the load feed"
            );
            assert!(
                cache_aware.mesh_attached_for_test(),
                "{role} policy missed the mesh tree bridge"
            );
        }
    }

    #[test]
    fn hinted_policy_without_operator_config_uses_type_defaults() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let policy = registry.on_worker_added("llama-3", Some("cache_aware"));
        let config = as_cache_aware(policy.as_ref()).unwrap().config_for_test();
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
            worker_of("http://prefill-1:8000", WorkerType::Prefill),
            worker_of("http://prefill-2:8000", WorkerType::Prefill),
        ];
        let decode_workers = vec![
            worker_of("http://decode-1:8000", WorkerType::Decode),
            worker_of("http://decode-2:8000", WorkerType::Decode),
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

    /// The worker URLs a cache-aware policy's string tree routes to for `model`.
    fn tree_of(policy: &Arc<dyn LoadBalancingPolicy>, model: &str) -> Vec<String> {
        as_cache_aware(policy.as_ref())
            .expect("a cache-aware policy")
            .tree_tenants_for_test(model)
    }

    /// A registry whose policy for `llama-3` is cache-aware, and two of that
    /// model's workers, routed by `init_cache_aware_policy`.
    fn cache_aware_model() -> (
        PolicyRegistry,
        Arc<dyn LoadBalancingPolicy>,
        Vec<Arc<dyn Worker>>,
    ) {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let policy = registry.on_worker_added("llama-3", Some("cache_aware"));
        let workers = vec![
            ready_worker("http://w1:8000", "llama-3"),
            ready_worker("http://w2:8000", "llama-3"),
        ];
        assert!(
            tree_of(&policy, "llama-3").is_empty(),
            "a fresh tree is empty"
        );
        registry.init_cache_aware_policy("llama-3", &workers);
        (registry, policy, workers)
    }

    #[test]
    fn init_cache_aware_policy_fills_the_models_tree() {
        let (_registry, policy, _workers) = cache_aware_model();
        assert_eq!(
            tree_of(&policy, "llama-3"),
            ["http://w1:8000", "http://w2:8000"]
        );
    }

    #[test]
    fn remove_worker_from_cache_aware_drops_the_worker_from_the_tree() {
        let (registry, policy, _workers) = cache_aware_model();
        registry.remove_worker_from_cache_aware("llama-3", "http://w1:8000");
        assert_eq!(tree_of(&policy, "llama-3"), ["http://w2:8000"]);
    }

    #[test]
    fn remove_workers_from_cache_aware_empties_the_models_tree() {
        let (registry, policy, workers) = cache_aware_model();
        let urls: HashSet<String> = workers.iter().map(|w| w.url().to_string()).collect();
        registry.remove_workers_from_cache_aware("llama-3", &urls);
        assert!(tree_of(&policy, "llama-3").is_empty());
    }

    #[test]
    fn init_pd_cache_aware_policies_fills_both_role_trees() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        registry.set_prefill_policy(cache_aware_policy());
        registry.set_decode_policy(cache_aware_policy());
        let prefill = vec![worker_of("http://prefill-1:8000", WorkerType::Prefill)];
        let decode = vec![worker_of("http://decode-1:8000", WorkerType::Decode)];
        registry.init_pd_cache_aware_policies(&prefill, &decode);
        assert_eq!(
            tree_of(&registry.get_prefill_policy(), prefill[0].model_id()),
            ["http://prefill-1:8000"]
        );
        assert_eq!(
            tree_of(&registry.get_decode_policy(), decode[0].model_id()),
            ["http://decode-1:8000"]
        );
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

        fn on_request_complete(&self, worker_url: &str) {
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

    /// The PD bucket init reaches a bucket prefill policy by its type: the
    /// partition exists after it, so the largest size is placed in the last
    /// range and not by the no-partition fallback.
    #[traced_test]
    #[test]
    fn init_pd_bucket_policies_reaches_the_bucket_policy() {
        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        registry.set_prefill_policy(Arc::new(BucketPolicy::with_config(BucketConfig {
            balance_abs_threshold: usize::MAX,
            bucket_adjust_interval_secs: 3600,
            ..Default::default()
        })));
        let prefill = vec![
            worker_of("http://prefill-1:8000", WorkerType::Prefill),
            worker_of("http://prefill-2:8000", WorkerType::Prefill),
            worker_of("http://prefill-3:8000", WorkerType::Prefill),
        ];

        registry.init_pd_bucket_policies(&prefill);

        assert!(logs_contain(
            "Initializing prefill bucket policy with 3 workers"
        ));
        let text = "a".repeat(10_000);
        let info = SelectWorkerInfo {
            request_text: Some(&text),
            ..Default::default()
        };
        let selected = registry
            .get_prefill_policy()
            .select_worker(&prefill, &info)
            .unwrap_or_else(|| panic!("a request of 10000 chars selects no prefill worker"));
        assert_eq!(
            prefill[selected].url(),
            "http://prefill-3:8000",
            "the largest size falls in the last range of the even split"
        );
        assert!(!logs_contain("No bucket found for model"));
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
            worker_of("http://prefill-1:8000", WorkerType::Prefill),
            worker_of("http://prefill-2:8000", WorkerType::Prefill),
            worker_of("http://prefill-3:8000", WorkerType::Prefill),
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

    /// The keyed decisions of a replay under the delegating override: eight
    /// regular workers behind a round-robin registry, sixteen session keys
    /// picked over three rounds, two workers vetoed for the second round and
    /// back for the third (a pin whose worker is vetoed respills and
    /// follows the respill). The sequence was recorded from the override
    /// that built its eligible list before reading the pin and copied the
    /// key per lookup; the override that reads the pin borrowed and lists
    /// only to assign must pick the same.
    #[test]
    fn the_replayed_keyed_requests_pick_the_recorded_sequence() {
        const RECORDED: [usize; 48] = [
            0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 1, 3, 4, 3, 6, 7, 0, 1, 4, 3, 4,
            6, 6, 7, 0, 1, 1, 3, 4, 3, 6, 7, 0, 1, 4, 3, 4, 6, 6, 7,
        ];
        let reg = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let policy = reg.get_default_policy();
        let workers: Vec<Arc<dyn Worker>> = (0..8)
            .map(|i| worker_of(&format!("http://w{i}"), WorkerType::Regular))
            .collect();
        let keys: Vec<String> = (0..16).map(|i| format!("conv{i}_t{}", i % 3)).collect();
        let mut picks = Vec::with_capacity(RECORDED.len());
        for round in 0..3 {
            for vetoed in [2, 5] {
                workers[vetoed].set_overloaded(round == 1);
            }
            for key in &keys {
                let info = SelectWorkerInfo {
                    rid_key: Some(key),
                    ..Default::default()
                };
                picks.push(
                    reg.select_worker(&policy, &workers, &info)
                        .expect("six workers stay eligible"),
                );
            }
        }
        assert_eq!(
            picks, RECORDED,
            "the replay picked {picks:?}; the recorded sequence is the decision"
        );
    }

    /// The engine decides the default policy's regular placements, keyed or
    /// not: under the sticky routing-key override the default binding pins
    /// sessions; without it a key is only a fact in the view. A model whose
    /// hint named another policy keeps the legacy path.
    #[test]
    fn the_engine_decides_the_default_policy_keyed_or_not() {
        use smg_policy_controller::Phase;

        let round_robin = PolicyConfig::RoundRobin;
        let policies = PolicyRegistry::with_override(
            round_robin.clone(),
            RoutingKeyOverrideConfig {
                enabled: true,
                ..Default::default()
            },
        )
        .with_policy_engine(PolicyEngineMode::Engine, &round_robin, &round_robin);
        let default = policies.get_default_policy();
        let pins = |policies: &PolicyRegistry| {
            let decider = policies
                .engine_for(policies.get_default_policy().as_ref())
                .expect("the engine serves the default policy");
            decider
                .engine()
                .binding("m", Phase::Single)
                .is_some_and(|binding| binding.sessions.is_some())
        };
        assert!(
            pins(&policies),
            "the override is the default binding's rule"
        );

        let hinted = policies.on_worker_added("hinted", Some("random"));
        assert_eq!(hinted.name(), "random");
        assert!(
            policies.engine_for(hinted.as_ref()).is_none(),
            "a hinted policy keeps its legacy instance"
        );
        assert!(policies.engine_for(default.as_ref()).is_some());

        let no_override = PolicyRegistry::new(round_robin.clone()).with_policy_engine(
            PolicyEngineMode::Engine,
            &round_robin,
            &round_robin,
        );
        assert!(!pins(&no_override), "without the override a key is a fact");
    }

    /// Under encode-prefill-decode the engine binds the mode's encode policy
    /// (consistent hashing when unset, the one default the legacy fallback
    /// reads too) and serves the encode items; under another mode, and under
    /// `legacy`, nothing but the legacy encode policy decides them.
    #[test]
    fn the_engine_serves_the_encode_items_of_an_encode_prefill_decode_deployment() {
        let round_robin = PolicyConfig::RoundRobin;
        let epd = |encode_policy: Option<PolicyConfig>| RoutingMode::EncodePrefillDecode {
            encode_urls: vec![],
            prefill_urls: vec![],
            decode_urls: vec![],
            encode_policy,
            prefill_policy: None,
            decode_policy: None,
        };
        let policies = PolicyRegistry::new(round_robin.clone()).with_policy_engine_for(
            PolicyEngineMode::Engine,
            &epd(None),
            &round_robin,
        );
        let decider = policies
            .encode_engine_for()
            .expect("the encode items are the engine's");
        assert!(decider.serves_encode());
        assert_eq!(policies.get_encode_policy().name(), "consistent_hashing");
        let explicit = PolicyRegistry::new(round_robin.clone()).with_policy_engine_for(
            PolicyEngineMode::Engine,
            &epd(Some(PolicyConfig::Random)),
            &round_robin,
        );
        assert!(explicit.encode_engine_for().is_some());
        let regular = PolicyRegistry::new(round_robin.clone()).with_policy_engine_for(
            PolicyEngineMode::Engine,
            &RoutingMode::Regular {
                worker_urls: vec![],
            },
            &round_robin,
        );
        assert!(regular.decider().is_some());
        assert!(
            regular.encode_engine_for().is_none(),
            "no encode side to serve"
        );
        let legacy = PolicyRegistry::new(round_robin).with_policy_engine_for(
            PolicyEngineMode::Legacy,
            &epd(None),
            &PolicyConfig::RoundRobin,
        );
        assert!(legacy.encode_engine_for().is_none());
    }

    /// Under encode-prefill-decode the engine is built from the encode side
    /// too, so an encode worker's static needs are the engine's answer for
    /// its Encode phase: the encode policy is served by the engine and the
    /// worker's KV-event question reads the engine, not the legacy policy.
    /// Under `legacy` the encode policy's own answer stands.
    #[test]
    fn an_encode_workers_static_needs_are_the_engines_under_encode_prefill_decode() {
        use crate::worker::test_support::worker_of;

        let round_robin = PolicyConfig::RoundRobin;
        let epd = RoutingMode::EncodePrefillDecode {
            encode_urls: vec![],
            prefill_urls: vec![],
            decode_urls: vec![],
            encode_policy: None,
            prefill_policy: None,
            decode_policy: None,
        };
        let worker = worker_of("grpc://encode:9000", WorkerType::Encode);
        let policies = PolicyRegistry::new(round_robin.clone()).with_policy_engine_for(
            PolicyEngineMode::Engine,
            &epd,
            &round_robin,
        );
        // The encode router publishes the mode's encode policy at start-up.
        policies.set_encode_policy(PolicyFactory::create_from_config(
            &PolicyConfig::ConsistentHashing,
        ));
        let encode = policies.get_encode_policy();
        assert!(
            policies.served_by_engine(encode.as_ref()),
            "the encode side is the engine's"
        );
        let decider = policies
            .encode_engine_for()
            .expect("the encode items are the engine's");
        assert_eq!(
            policies.worker_needs_kv_events(worker.as_ref()),
            decider
                .engine()
                .needs_for(worker.model_id(), Phase::Encode)
                .events
        );
        let legacy = PolicyRegistry::new(round_robin).with_policy_engine_for(
            PolicyEngineMode::Legacy,
            &epd,
            &PolicyConfig::RoundRobin,
        );
        legacy.set_encode_policy(PolicyFactory::create_from_config(
            &PolicyConfig::ConsistentHashing,
        ));
        let encode = legacy.get_encode_policy();
        assert!(!legacy.served_by_engine(encode.as_ref()));
        assert_eq!(
            legacy.worker_needs_kv_events(worker.as_ref()),
            encode.needs_kv_events()
        );
    }
}
