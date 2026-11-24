use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU8, AtomicUsize, Ordering},
        Arc, OnceLock, Weak,
    },
};

use dashmap::DashMap;
use http::{header::HeaderName, HeaderMap};
#[cfg(test)]
use openai_protocol::UNKNOWN_MODEL_ID;
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxBuildHasher;
use smg_policy_controller::{
    document::{Needs, PolicySpec},
    engine::{Engine, Phase, PolicyEngine},
};
use smg_worker::{
    identity::{
        basic::{RequestCompletionSink, Worker, WorkerType},
        endpoint::Endpoint,
    },
    load::{monitor::LoadConsumer, state::LoadReceiver},
};
use tracing::{debug, info, warn};

/// Policy Registry for managing model-to-policy mappings
///
/// This registry manages the dynamic assignment of load balancing policies to models.
/// When the first worker of a new model is added, it determines the policy for that model.
/// All subsequent workers of the same model use the established policy.
/// When the last worker of a model is removed, the policy mapping is cleaned up.
use super::{
    as_bucket, as_cache_aware, get_healthy_worker_indices,
    hash_ring::HashRing,
    kept_rings::KeptRings,
    manual::{ExecutionBranch, PinState},
    normalize_model_key, DPRankLoadPolicy, LentTrees, LoadBalancingPolicy, ManualConfig,
    ManualPolicy, PolicyFactory, SelectWorkerInfo, WorkerRole,
};
use crate::{
    config::types::{
        ManualAssignmentMode, PdPairingMode, PolicyConfig, RouterConfig, RoutingKeyOverrideConfig,
    },
    mesh::adapters::TreeSyncAdapter,
    observability::{cache_trace, metrics::Metrics},
    policies::kv_index::event_monitor::KvEventMonitor,
    routers::common::{
        header_utils::{
            extract_routing_key_hint_named, parse_routing_tokens_hint, ROUTING_KEY_HINT_MAX_BYTES,
        },
        snapshot::{ModelWorkerSnapshot, WorkerRegistry, WorkerRegistryExt},
    },
};

/// A side of a pair deployment: the place of one of the registry's side
/// policies ([`PolicyRegistry::set_side_policy`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Prefill,
    Decode,
    Encode,
}

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

/// What stands where a classic policy would in a registry whose gateway
/// runs the policy engine ([`PolicyRegistry::for_policy_engine`]): in the
/// default slot and the encode slot, so every getter hands it out. It is no
/// classic policy, and the policy factory never makes it.
///
/// It places nothing. A request that reaches its placing call is answered
/// that no worker is available, and is never sent: a path that wrongly asks
/// a classic policy under the engine shows as a failure, and not as a
/// classic placement. Each such request counts on the registry's count
/// ([`PolicyRegistry::classic_made_or_asked`]).
///
/// It answers the two text questions from the registry's stored bits
/// ([`TextNeeds`]), so a reader that asks the object a getter handed it
/// gets the registry's answer. The one in the two side slots answers from
/// the pairs' own bits: the pair routes ask those two objects, as they ask
/// a classic registry's two side policies. It reads no load report and no
/// cache event, and it keeps no state of its own.
#[derive(Debug)]
struct NoClassicPolicy {
    /// The registry's text bits: the document's, or the pairs' own.
    text_needs: Arc<AtomicU8>,
    /// The registry's count.
    asked: Arc<AtomicUsize>,
}

impl NoClassicPolicy {
    fn text_needs(&self) -> TextNeeds {
        TextNeeds(self.text_needs.load(Ordering::Acquire))
    }
}

impl LoadBalancingPolicy for NoClassicPolicy {
    fn select_worker(&self, _: &[Arc<dyn Worker>], _: &SelectWorkerInfo) -> Option<usize> {
        self.asked.fetch_add(1, Ordering::Relaxed);
        None
    }

    fn name(&self) -> &'static str {
        "policy_controller"
    }

    fn needs_request_text(&self) -> bool {
        self.text_needs().answer(false, false)
    }

    fn needs_request_text_with_tokens(&self) -> bool {
        self.text_needs().answer(true, false)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
///
/// Where the policy engine places every request the bits are its
/// document's answers ([`Self::of_document`]), written where the engine is
/// set, and no policy is asked. A second value there holds what a pair's
/// two decisions read ([`Self::of_pairs`]).
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
            policy.honors_routing_key(),
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

    /// What a document that needs `needs` answers, where the policy engine
    /// places every request. The text of a request that carries no token
    /// ids is read where a stage hashes the text's windows or places by the
    /// prompt's size, and the text of a request that carries them where a
    /// stage reads the text itself. A routing key lifts neither: the
    /// document's stages place a key that follows no session yet, and read
    /// of it what they read of any request.
    fn of_document(needs: Needs) -> Self {
        let with_tokens = needs.text;
        let without_tokens = with_tokens || needs.text_hashes || needs.prompt_size;
        Self::from_answers(without_tokens, with_tokens, true)
    }

    /// What a pair's two decisions read of a request's text under the
    /// document `engine` runs: what some decision of the prefill phase and
    /// some decision of the decode phase may read
    /// ([`Engine::needs_for_phase`]). The engine keeps that answer by the
    /// bindings' order, and a binding behind a catch-all one adds nothing
    /// to a phase: a binding of single requests behind the pairs' asks no
    /// text of a pair. That is the grain of the classic answer, the two
    /// side policies' and not the main policy's. It names no model, so it
    /// is one answer for every pair request: a text one model's pair reads
    /// is built for every model's.
    fn of_pairs(engine: &Engine) -> Self {
        let read = |phase| Self::of_document(engine.needs_for_phase(phase));
        read(Phase::Prefill).join(read(Phase::Decode))
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
    /// Model ID -> Policy instance mapping (lock-free reads via DashMap; the
    /// Fx hasher over the operator's model ids, read on every request)
    model_policies: Arc<DashMap<String, Arc<dyn LoadBalancingPolicy>, FxBuildHasher>>,

    /// The text questions' answers over the default and the per-model
    /// policies ([`TextNeeds`]), recomputed where one is added or removed
    /// ([`Self::refresh_text_needs`]).
    text_needs: Arc<AtomicU8>,

    /// Serializes the recomputations of `text_needs`: a removal's walk that
    /// started before a concurrent addition's insert must not publish over
    /// the bits that addition published.
    text_needs_refresh: Arc<Mutex<()>>,

    /// What a pair's two decisions read of a request's text under the
    /// policy engine ([`TextNeeds::of_pairs`]): the answer of the stand-in
    /// a registry made for the engine keeps in its two side slots, written
    /// where the engine is set ([`Self::set_engine`]). A classic registry
    /// never reads it: its two side policies answer for themselves.
    pair_text_needs: Arc<AtomicU8>,

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

    /// The policy engine, where the gateway started one
    /// (`policy_controller: on`): set once at start
    /// ([`Self::set_engine`]). The placement functions ask it then, for
    /// every request.
    engine: Arc<OnceLock<&'static PolicyEngine>>,

    /// The token trees the policy engine is lent, where its start asked
    /// for them ([`Self::lend_token_trees`]). A worker the registry lost is
    /// taken out of them by their own task
    /// ([`LentTrees::follow_removals`]).
    lent_trees: Arc<OnceLock<Arc<LentTrees>>>,

    /// Shared sticky selector for the routing-key override. `Some` when the
    /// override is enabled; consulted (instead of the configured policy) for keyed
    /// requests via [`PolicyRegistry::select_worker_for_model`]. `None` in a
    /// registry made for the policy engine, whatever the settings say.
    routing_key_sticky: Option<Arc<ManualPolicy>>,

    /// Whether the settings enable the routing-key override: what the
    /// registry answers a route that derives a request's key. Under the
    /// classic policies it is `routing_key_sticky.is_some()`.
    routing_key_override: bool,

    /// Whether the classic policies place this gateway's requests. `false`
    /// in a registry made for the policy engine
    /// ([`Self::for_policy_engine`]): it makes no classic policy object,
    /// and what feeds one does nothing. Read at the start and at a
    /// worker's registration, never by a decision.
    classic: bool,

    /// How many classic policy objects this registry made or was handed,
    /// and how many requests reached its stand-in's placing call
    /// ([`Self::classic_made_or_asked`]).
    made_or_asked: Arc<AtomicUsize>,

    /// The name the settings give each side's policy, by the side's place
    /// in [`Side`]: kept where the side is set
    /// ([`Self::set_side_policy`]), the first name set standing as the
    /// first policy does. A registry made for the policy engine keeps this
    /// of a side and nothing else ([`Self::side_policy_name`]).
    side_names: Arc<[OnceLock<&'static str>; 3]>,

    /// Ordered routing-key header names, parsed once from
    /// `routing_key_override.headers`; the first header present with a valid
    /// value wins.
    routing_key_headers: Arc<Vec<HeaderName>>,
    /// How strictly PD placement pairs prefill and decode descriptors.
    pd_pairing_mode: PdPairingMode,
    /// The policy engine decides under a document the gateway was supplied
    /// ([`Self::runs_supplied_document`]).
    supplied_document: bool,
    /// The hash rings kept between membership generations ([`KeptRings`]).
    rings: Arc<KeptRings>,
}

/// A sticky key with this many of its own requests already in flight on its
/// pinned worker (router-local, counted per replica) is bypassed and the
/// request reassigned. Guards one conversation stacking onto its pin; total
/// worker load is not a respill trigger.
const STICKY_INFLIGHT_CAP: u16 = 2;

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
        // The default policy, and the override's pin table where it is on.
        let made = 1 + usize::from(routing_key_sticky.is_some());
        Self::assembled(
            default_policy,
            default_policy_config,
            routing_key_sticky,
            &routing_key_override,
            true,
            made,
        )
    }

    /// The registry of a gateway whose requests the policy engine places
    /// (`policy_controller: on`). It makes NO classic policy object, here or
    /// later: its default slot and its encode slot hold a stand-in that
    /// places nothing, it keeps no pin table for the routing-key override,
    /// a worker's registration and a router's build add no policy to it,
    /// and what would feed a policy (the load feed, the cache-event
    /// monitor, the mesh bridge, a request's end) does nothing. The set-up
    /// and the removal of a policy's trees and tables find no policy in
    /// it. A request that reached the stand-in would be refused, and
    /// counted ([`Self::classic_made_or_asked`]). The stand-in answers the
    /// two text questions from the registry's stored answers, and the two
    /// side slots hold one that answers what a pair's own decisions read.
    /// What a route asks of the registry beside a policy stands as the
    /// settings say: the routing key's headers, whether the override is on,
    /// the pairing mode, the hash rings.
    pub fn for_policy_engine(
        default_policy_config: PolicyConfig,
        routing_key_override: RoutingKeyOverrideConfig,
    ) -> Self {
        let text_needs = Arc::new(AtomicU8::new(TextNeeds::default().0));
        let pair_text_needs = Arc::new(AtomicU8::new(TextNeeds::default().0));
        let asked = Arc::new(AtomicUsize::new(0));
        let stand_in_over = |text_needs: &Arc<AtomicU8>| -> Arc<dyn LoadBalancingPolicy> {
            Arc::new(NoClassicPolicy {
                text_needs: Arc::clone(text_needs),
                asked: Arc::clone(&asked),
            })
        };
        let stand_in = stand_in_over(&text_needs);
        // The pair routes ask the two side slots' objects what they read,
        // as they ask a classic registry's two side policies. Their
        // stand-in answers for a pair's own decisions, and not for the
        // whole document.
        let pair_side = stand_in_over(&pair_text_needs);
        Self {
            // The stand-in reads the registry's text bits and counts on
            // the registry's count.
            text_needs,
            pair_text_needs,
            made_or_asked: asked,
            prefill_policy: Arc::new(OnceLock::from(Arc::clone(&pair_side))),
            decode_policy: Arc::new(OnceLock::from(pair_side)),
            // The encode slot is filled: empty, its getter would build a
            // policy at each call.
            encode_policy: Arc::new(OnceLock::from(Arc::clone(&stand_in))),
            ..Self::assembled(
                stand_in,
                default_policy_config,
                None,
                &routing_key_override,
                false,
                0,
            )
        }
    }

    /// The registry of a gateway of `config`: the classic one
    /// ([`Self::with_override`]), or the one that makes no classic policy
    /// where the policy engine places the gateway's requests
    /// (`RouterConfig::policy_engine_decides`, [`Self::for_policy_engine`]),
    /// under the pairing mode in force
    /// (`RouterConfig::pd_pairing_in_force`): a supplied document's own
    /// where the policy controller runs one, else the configuration's. It
    /// knows whether the engine decides under a supplied document
    /// (`Self::runs_supplied_document`). The one place that reads the
    /// option for the registry.
    pub fn of_config(config: &RouterConfig) -> Self {
        let (policy, routing_key_override) =
            (config.policy.clone(), config.routing_key_override.clone());
        let decides = config.policy_engine_decides();
        let mut registry = if decides {
            Self::for_policy_engine(policy, routing_key_override)
        } else {
            Self::with_override(policy, routing_key_override)
        };
        registry.supplied_document = decides && config.policy_spec.is_some();
        registry.with_pd_pairing_mode(config.pd_pairing_in_force())
    }

    /// The registry's fields from what a constructor made: `made` classic
    /// policy objects, the default policy and the pin table among them.
    fn assembled(
        default_policy: Arc<dyn LoadBalancingPolicy>,
        default_policy_config: PolicyConfig,
        routing_key_sticky: Option<Arc<ManualPolicy>>,
        routing_key_override: &RoutingKeyOverrideConfig,
        classic: bool,
        made: usize,
    ) -> Self {
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
            model_policies: Arc::new(DashMap::with_hasher(FxBuildHasher)),
            text_needs: Arc::new(AtomicU8::new(text_needs.0)),
            text_needs_refresh: Arc::new(Mutex::new(())),
            pair_text_needs: Arc::default(),
            default_policy,
            default_policy_config,
            prefill_policy: Arc::new(OnceLock::new()),
            decode_policy: Arc::new(OnceLock::new()),
            encode_policy: Arc::new(OnceLock::new()),
            kv_event_monitor: Arc::new(RwLock::new(None)),
            load_rx: Arc::new(RwLock::new(None)),
            mesh_tree_sync: Arc::new(RwLock::new(None)),
            dp_rank_policy: Arc::new(OnceLock::new()),
            engine: Arc::new(OnceLock::new()),
            lent_trees: Arc::new(OnceLock::new()),
            routing_key_sticky,
            routing_key_override: routing_key_override.enabled,
            classic,
            made_or_asked: Arc::new(AtomicUsize::new(made)),
            side_names: Arc::default(),
            routing_key_headers: Arc::new(routing_key_headers),
            pd_pairing_mode: PdPairingMode::default(),
            supplied_document: false,
            rings: Arc::new(KeptRings::default()),
        }
    }

    /// The consistent-hash ring `model_id`'s requests are placed on: the
    /// ring of the generation the registry holds for the model, `None` for a
    /// model nobody serves. The model-less id reads the ring over every
    /// worker, the candidates of a request that names no model. While that
    /// generation's ring is filled it is the ring kept from another
    /// generation, and `None` before the model's first ring
    /// ([`KeptRings::of`]). A caller holding a client's name for the model
    /// resolves it at request entry, as for the registry's other reads by
    /// model.
    pub(crate) fn hash_ring(
        &self,
        registry: &WorkerRegistry,
        model_id: &str,
    ) -> Option<Arc<HashRing>> {
        self.rings
            .of(model_id, &registry.get_routing_snapshot(model_id))
    }

    /// [`Self::hash_ring`] for a generation the caller holds: no second read
    /// of the registry, and that generation's own ring once it is filled.
    pub(crate) fn hash_ring_of(
        &self,
        model_id: &str,
        generation: &ModelWorkerSnapshot,
    ) -> Option<Arc<HashRing>> {
        self.rings.of(model_id, generation)
    }

    /// Whether the hash rings have followed the membership: a request
    /// placed now walks the ring of its own generation. After a membership
    /// change, that holds again within one fill of the task that keeps the
    /// rings warm. For a caller that changed the membership and waits for
    /// the placements to follow, as a test bed does.
    #[cfg(any(test, feature = "test-util"))]
    pub fn hash_rings_are_current(&self, registry: &WorkerRegistry) -> bool {
        self.rings.are_current(registry)
    }

    /// Starts the task that fills the hash rings off the request path
    /// ([`KeptRings::keep_warm`]).
    pub(crate) fn keep_rings_warm(
        &self,
        registry: &Arc<WorkerRegistry>,
        runtime: &tokio::runtime::Handle,
    ) {
        self.rings.keep_warm(registry, runtime);
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
        if !self.routing_key_override {
            return None;
        }
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
        Some(if self.routing_key_override {
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
        self.routing_key_override
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
    /// `policy` does not honor the key itself
    /// ([`LoadBalancingPolicy::honors_routing_key`]: `manual` and
    /// `consistent_hashing` read `rid_key` and the header themselves with the
    /// same header-first precedence), and the request carries a key from a
    /// configured source. The one predicate the keyed path is entered by; the
    /// policy is asked only when the override is enabled.
    fn sticky_for<'a>(
        &self,
        policy: &dyn LoadBalancingPolicy,
        info: &SelectWorkerInfo<'a>,
    ) -> Option<(&Arc<ManualPolicy>, &'a str, &'static str)> {
        let sticky = self.routing_key_sticky.as_ref()?;
        if policy.honors_routing_key() {
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
            .unwrap_or(UNKNOWN_MODEL_ID);
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
            debug!(
                source,
                key,
                branch = branch.as_str(),
                worker = result.map_or("none", |idx| workers[idx].redacted_url()),
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

    /// Set KV event monitor (thread-safe, can be called after initialization).
    /// Propagates to all existing cache-aware policies (including default, prefill, decode).
    pub fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        if !self.classic {
            return;
        }
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

    /// The policy engine that places every request, where the gateway
    /// started one. `None` under the classic policies, which is every
    /// gateway that does not set `policy_controller: on`.
    #[inline]
    pub(crate) fn engine(&self) -> Option<&'static PolicyEngine> {
        self.engine.get().copied()
    }

    /// Hands the registry the engine the gateway started, once: a second
    /// engine is not taken. From here on the engine's document answers the
    /// text questions ([`TextNeeds::of_document`]): the stored answer is
    /// written here, and a request reads it as before. What a pair's two
    /// decisions read is settled here once, of the engine that is taken
    /// ([`TextNeeds::of_pairs`]): an engine keeps one document for its life.
    pub(crate) fn set_engine(&self, engine: &'static PolicyEngine) {
        if self.engine.set(engine).is_err() {
            warn!("a policy engine is already set: the second one is not taken");
        } else {
            let pairs = TextNeeds::of_pairs(engine.spec().engine());
            self.pair_text_needs.store(pairs.0, Ordering::Release);
        }
        self.refresh_text_needs();
    }

    /// Whether the policy engine decides under a document the gateway was
    /// supplied, and not under the one its routing settings mean. The names
    /// the settings give the policies ([`Self::policy_name`],
    /// [`Self::side_policy_name`]) then name nothing that runs: a row of a
    /// traced request's decision record reads none of them. False for a
    /// registry made of settings alone, as a classic one is, and beside a
    /// document's path whose document was not loaded: that configuration
    /// runs the settings' document.
    pub(crate) fn runs_supplied_document(&self) -> bool {
        self.supplied_document
    }

    /// The token trees the policy engine is lent, made at the first call
    /// under the numbers of `document`, the document the engine runs, whose
    /// block size is `page` tokens ([`LentTrees::of_document`]), and kept
    /// here. The engine's start calls it, and only where its document reads
    /// the prefixes sent to each worker: with the engine on no classic
    /// policy is made that would keep such trees.
    ///
    /// The trees are made with the task that takes a worker `workers` lost
    /// out of them ([`LentTrees::follow_removals`]), on the runtime the
    /// call is made on. The start runs on one, the engine's own task needs
    /// it. Where none is entered no task is started, nothing takes a worker
    /// out of the trees, and a debug line says so.
    pub(crate) fn lend_token_trees(
        &self,
        document: &PolicySpec,
        page: u32,
        workers: &WorkerRegistry,
    ) -> Arc<LentTrees> {
        let made = || {
            let trees = Arc::new(LentTrees::of_document(document.clone(), page));
            match tokio::runtime::Handle::try_current() {
                Ok(runtime) => trees.follow_removals(workers, &runtime),
                Err(_) => debug!("No runtime is entered: no task takes a worker that leaves out of the lent token trees"),
            }
            trees
        };
        Arc::clone(self.lent_trees.get_or_init(made))
    }

    /// The token trees the policy engine is lent, where its start asked
    /// for them: for a case that reads what the start made.
    #[cfg(test)]
    pub(crate) fn lent_trees(&self) -> Option<Arc<LentTrees>> {
        self.lent_trees.get().cloned()
    }

    /// The name the settings give the main policy: what a classic selection
    /// counts a request under where its model has no policy of its own. An
    /// answer of the policy engine that names no profile is counted under
    /// it, and no policy object is asked for the name.
    pub(crate) fn policy_name(&self) -> &'static str {
        self.default_policy_config.name()
    }

    /// The name the settings give the policy of a pair's `side`
    /// ([`Self::set_side_policy`]), and where they give the side none the
    /// name of the policy its getter falls back to: the main policy for a
    /// prefill or a decode side, `consistent_hashing` for an encode side.
    /// A side of the policy engine's answer that names no profile is
    /// counted under it. No policy object is asked for the name, and a
    /// policy handed whole ([`Self::set_prefill_policy`]) names nothing
    /// here.
    pub(crate) fn side_policy_name(&self, side: Side) -> &'static str {
        let kept = self.side_names[side as usize].get().copied();
        kept.unwrap_or_else(|| match side {
            Side::Prefill | Side::Decode => self.policy_name(),
            Side::Encode => PolicyConfig::ConsistentHashing.name(),
        })
    }

    /// Hand every policy the worker monitor's load feed (thread-safe, can be
    /// called after initialization); a policy created later takes it from
    /// the registry (see [`Self::publish_with_shared_state`]).
    pub(crate) fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        if !self.classic {
            return;
        }
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
    /// wired in and its mesh flag raised (both flip
    /// atomically inside [`super::CacheAwarePolicy::set_mesh_tree_sync`]); every
    /// future cache-aware policy created here inherits both. Pass `None`
    /// to detach.
    pub fn set_mesh_tree_sync(&self, adapter: Option<Arc<TreeSyncAdapter>>) {
        if !self.classic {
            return;
        }
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

    /// The request-completion observer a worker of this gateway is given
    /// at its registration: the registry's own ([`Self::completion_sink`])
    /// under the classic policies, and none where the policy engine places
    /// the requests. The engine hears a request's end from its placement,
    /// and a guard's drop then reaches no policy.
    pub fn worker_completion_sink(self: &Arc<Self>) -> Option<Arc<dyn RequestCompletionSink>> {
        self.classic.then(|| self.completion_sink())
    }

    /// What this registry counts for the rule that a gateway which runs
    /// the policy engine uses no classic policy: the classic policy
    /// objects it made or was handed, and the requests that reached its
    /// stand-in's placing call.
    ///
    /// A made or handed object is the default policy, the routing-key
    /// override's pin table, a model's own policy built from a worker's
    /// label, a side's policy of a pair or an encode deployment, a policy
    /// handed whole, or the rank policy. A policy counts once, with what
    /// its own constructor builds inside it. The gateway makes a classic
    /// policy nowhere else: the policy factory is called from this file
    /// alone, and the router factory builds the rank policy only for a
    /// registry that keeps classic policies. One object is not counted,
    /// the consistent-hashing policy [`Self::get_encode_policy`] builds at
    /// each call where no encode policy was set: a registry made for the
    /// policy engine never builds it.
    ///
    /// A registry made for the policy engine makes none of these. It
    /// reads zero until a request reaches its stand-in's placing call,
    /// which refuses the request and counts it. The stand-in a getter
    /// hands out is no classic object and does not count by itself: a
    /// reader may ask it its name and its text answers.
    pub fn classic_made_or_asked(&self) -> usize {
        self.made_or_asked.load(Ordering::Relaxed)
    }

    /// Whether the classic policies place this gateway's requests:
    /// `false` for a registry made for the policy engine. The router
    /// factory reads it before it builds a policy to hand over.
    pub(crate) fn keeps_classic_policies(&self) -> bool {
        self.classic
    }

    /// One more classic policy object was made or handed.
    fn count_classic_policy(&self) {
        self.made_or_asked.fetch_add(1, Ordering::Relaxed);
    }

    /// What the document in force needs of the host, where the policy
    /// engine stands: the load poll and the stored text answers are asked of
    /// it in a registry made for the engine. They are a group's poll and one
    /// stored byte, read with no model at hand.
    fn document_needs(&self) -> Option<Needs> {
        self.engine().map(|engine| engine.spec().needs())
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
        if !self.classic {
            return;
        }
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
        if !self.classic {
            // No model has a policy of its own under the engine: the
            // stand-in is answered and nothing is kept.
            return Arc::clone(&self.default_policy);
        }
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
            self.count_classic_policy();
            Self::create_policy_from_config(&self.default_policy_config)
        } else if let Some(policy) = PolicyFactory::create_by_name(policy_type) {
            self.count_classic_policy();
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

    /// Builds the policy of a pair's `side` from the side's own `config`
    /// and sets it, once at startup, as [`Self::set_prefill_policy`] sets
    /// a policy handed whole. The registry keeps one policy a side. A
    /// gateway of many routers asks this once from each of its pair
    /// routers: the first policy set stands, and a later call builds
    /// nothing.
    pub fn set_side_policy(&self, side: Side, config: &PolicyConfig) {
        // The side's name stands whether or not a policy is built for it.
        let _ = self.side_names[side as usize].set(config.name());
        let kept = match side {
            Side::Prefill => &self.prefill_policy,
            Side::Decode => &self.decode_policy,
            Side::Encode => &self.encode_policy,
        };
        if !self.classic || kept.get().is_some() {
            return;
        }
        let policy = Self::create_policy_from_config(config);
        match side {
            Side::Prefill => self.set_prefill_policy(policy),
            Side::Decode => self.set_decode_policy(policy),
            Side::Encode => self.set_encode_policy(policy),
        }
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
        if !self.classic {
            return;
        }
        self.count_classic_policy();
        self.publish_with_shared_state(&policy, || {
            // OnceLock::set returns Err if already set, which we ignore since
            // the policy should only be set once at startup
            let _ = self.prefill_policy.set(Arc::clone(&policy));
        });
    }

    pub fn set_dp_rank_policy(&self, policy: Arc<dyn DPRankLoadPolicy>) {
        if !self.classic {
            return;
        }
        self.count_classic_policy();
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
        if !self.classic {
            return;
        }
        self.count_classic_policy();
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
        if !self.classic {
            return;
        }
        self.count_classic_policy();
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
            self.routing_key_override && self.resolve_routing_key(headers).is_some();
        TextNeeds(self.text_needs.load(Ordering::Acquire)).answer(with_tokens, keyed_override)
    }

    /// Recompute [`Self::text_needs`] over the policies the text questions
    /// range over: the default policy and the per-model policies. Called
    /// where the per-model map changes; one recomputation at a time, so the
    /// bits published last are those of a walk that saw every earlier change.
    /// Where the policy engine places every request its document answers
    /// instead, and no policy is walked: called where the engine is set too.
    fn refresh_text_needs(&self) {
        let _one_at_a_time = self.text_needs_refresh.lock();
        let needs = match self.document_needs() {
            Some(needs) => TextNeeds::of_document(needs),
            None => self.policies_text_needs(),
        };
        self.text_needs.store(needs.0, Ordering::Release);
    }

    /// What the default policy and the per-model policies answer the text
    /// questions.
    fn policies_text_needs(&self) -> TextNeeds {
        let mut needs = TextNeeds::of(self.default_policy.as_ref());
        for entry in self.model_policies.iter() {
            let policy = entry.value().as_ref();
            needs = needs.join(TextNeeds::of(policy));
        }
        needs
    }

    /// Whether any policy routes on backend load (see
    /// [`LoadBalancingPolicy::needs_backend_loads`]): with load monitoring
    /// off, the worker monitor still polls a group while this holds.
    pub fn needs_backend_loads(&self) -> bool {
        if !self.classic {
            // The document's word, as the engine keeps the load channel by
            // it: a load report or a named signal read.
            return self
                .document_needs()
                .is_some_and(|needs| needs.loads || needs.signal_cells);
        }
        let mut needs = false;
        self.each_policy(|policy| needs |= policy.needs_backend_loads());
        needs
    }

    /// Whether the prefill or the decode policy reads the request text: the
    /// pair routers' body question. Under the policy engine the two side
    /// slots hold a stand-in that answers what a pair's own decisions read
    /// ([`TextNeeds::of_pairs`]), settled where the engine is set: a
    /// request looks nothing up, and no kind of registry is tested here.
    ///
    /// The count route asks it too, though its request is decided as a
    /// single one over the prefill pool: the classic pass asks the model's
    /// main policy there (`select_pass`). Beside a main policy that reads
    /// the text and sides that read none, either kind places a count with
    /// no text.
    pub(crate) fn pair_needs_request_text(&self) -> bool {
        self.get_prefill_policy().needs_request_text()
            || self.get_decode_policy().needs_request_text()
    }

    /// Whether `worker`'s stream carries its KV events into its model's
    /// index: what the policy that places requests on it reads
    /// ([`Self::policy_for_worker`]). Under the policy engine it is what a
    /// decision of the worker's kind for its model reads
    /// ([`Engine::needs_for`]), and not the document's whole list: a binding
    /// that another kind of worker runs under opens no stream to this one.
    /// The answer is no before an engine stands.
    pub(crate) fn worker_needs_kv_events(&self, worker: &dyn Worker) -> bool {
        if !self.classic {
            let phase = match worker.worker_type() {
                WorkerType::Regular => Phase::Single,
                WorkerType::Prefill => Phase::Prefill,
                WorkerType::Decode => Phase::Decode,
                WorkerType::Encode => Phase::Encode,
            };
            return self.engine().is_some_and(|engine| {
                let reads = engine.spec().engine().needs_for(worker.model_id(), phase);
                reads.events
            });
        }
        self.policy_for_worker(worker).needs_kv_events()
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
                    Endpoint::redacted_url(worker_url),
                    model_id
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
                        Endpoint::redacted_url(worker_url),
                        worker_type
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

impl LoadConsumer for PolicyRegistry {
    fn needs_loads(&self) -> bool {
        self.needs_backend_loads() || self.get_dp_rank_policy().is_some()
    }

    fn reconcile_in_flight(&self, worker: &dyn Worker) {
        PolicyRegistry::reconcile_in_flight(self, worker);
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};
    use smg_policy_controller::document::{AppliedSpec, Pairing, PolicySpec};
    use smg_worker::{
        identity::basic::{Worker, WorkerLoadGuard, WorkerType},
        load::state::LoadSnapshot,
        test_support::{ready_worker, worker_with_model_aliases},
    };
    use tokio::sync::watch;
    use tracing_test::traced_test;

    use super::*;
    use crate::{
        config::RoutingMode,
        policies::{
            hash_ring::HashRing, test_support::headers_with_routing_key, BucketConfig,
            BucketPolicy, CacheAwareConfig, CacheAwarePolicy, LeastLoadPolicy, SelectWorkerInfo,
        },
        routers::common::snapshot::test_support::worker_of,
    };

    /// Every step that makes a classic policy or hands the registry one: a
    /// model's own policy by a worker's label, a model that takes the
    /// default, each side's policy, a policy handed whole and the rank
    /// policy.
    fn every_making_step(registry: &PolicyRegistry) {
        registry.on_worker_added("labelled", Some("power_of_two"));
        registry.on_worker_added("plain", None);
        for side in [Side::Prefill, Side::Decode, Side::Encode] {
            registry.set_side_policy(side, &PolicyConfig::RoundRobin);
        }
        registry.set_decode_policy(PolicyFactory::create_from_config(&PolicyConfig::Random));
        registry.set_dp_rank_policy(Arc::new(crate::policies::MinimumTokensPolicy::new(None)));
    }

    /// Where the policy engine places the requests the registry makes no
    /// classic policy object and keeps none it is handed. The same steps
    /// count eight under the classic policies: the default and the
    /// override's pin table at the start, then the labelled model's policy,
    /// the three sides', the one handed whole and the rank policy.
    #[test]
    fn a_registry_made_for_the_policy_engine_makes_no_classic_policy_object() {
        let classic = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
        assert_eq!(classic.classic_made_or_asked(), 2);
        every_making_step(&classic);
        assert_eq!(classic.classic_made_or_asked(), 8);

        let engine =
            PolicyRegistry::for_policy_engine(PolicyConfig::RoundRobin, enabled_override());
        every_making_step(&engine);
        assert_eq!(engine.classic_made_or_asked(), 0);
        assert!(engine.get_policy("labelled").is_none());
        assert!(engine.get_dp_rank_policy().is_none());
        // The settings' word on the routing-key override stands, with no
        // pin table behind it.
        assert!(engine.routing_key_override_enabled());
        assert_eq!(engine.derive_rid_key(Some("session-A")), Some("session-A"));
        assert_eq!(classic.derive_rid_key(Some("session-A")), Some("session-A"));
        // A worker's guard tells a policy of the request's end under the
        // classic policies alone.
        assert!(Arc::new(classic).worker_completion_sink().is_some());
        assert!(Arc::new(engine).worker_completion_sink().is_none());
    }

    /// A request that reaches the stand-in is placed nowhere, whichever
    /// accessor handed the stand-in out: its route answers that no worker
    /// is available and sends nothing, and the registry counts it. A
    /// classic policy places the same workers.
    #[test]
    fn the_stand_in_of_a_registry_made_for_the_policy_engine_places_no_request() {
        let workers = vec![
            worker_of("http://w1", WorkerType::Regular),
            worker_of("http://w2", WorkerType::Regular),
        ];
        let info = SelectWorkerInfo::default();
        let classic = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let placed = classic.get_default_policy().select_worker(&workers, &info);
        assert!(placed.is_some());

        let engine =
            PolicyRegistry::for_policy_engine(PolicyConfig::RoundRobin, enabled_override());
        for policy in [
            engine.get_default_policy(),
            engine.get_policy_or_default("any-model"),
            engine.get_prefill_policy(),
            engine.get_decode_policy(),
            engine.get_encode_policy(),
            engine.on_worker_added("any-model", Some("round_robin")),
        ] {
            let before = engine.classic_made_or_asked();
            assert_eq!(policy.select_worker(&workers, &info), None);
            assert_eq!(engine.select_worker(&policy, &workers, &info), None);
            assert_eq!(engine.classic_made_or_asked(), before + 2);
        }
    }

    /// The stand-in answers the two text questions from the registry's
    /// stored bits, so a reader that asks the object a getter handed it
    /// reads what the registry stored. The one in the two side slots reads
    /// the pairs' own bits: the pair's text question, which asks the two
    /// side getters' objects, follows those and not the document's.
    #[test]
    fn the_stand_in_answers_the_text_questions_from_the_registrys_stored_bits() {
        let engine = PolicyRegistry::for_policy_engine(
            PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096),
            RoutingKeyOverrideConfig::default(),
        );
        let stand_in = engine.get_default_policy();
        let sides = [engine.get_prefill_policy(), engine.get_decode_policy()];
        let answers = |policy: &Arc<dyn LoadBalancingPolicy>| {
            (
                policy.needs_request_text(),
                policy.needs_request_text_with_tokens(),
            )
        };
        assert_eq!(answers(&stand_in), (false, false));
        assert!(!engine.pair_needs_request_text());

        // The document's bits: a pair's two sides do not read them.
        engine.text_needs.store(TextNeeds::TEXT, Ordering::Release);
        assert_eq!(answers(&stand_in), (true, false));
        assert!(engine.any_policy_needs_request_text(None));
        assert_eq!(sides.each_ref().map(answers), [(false, false); 2]);
        assert!(!engine.pair_needs_request_text());

        // The pairs' own bits: each side's object reads them.
        let text = TextNeeds::TEXT;
        engine.pair_text_needs.store(text, Ordering::Release);
        assert_eq!(sides.each_ref().map(answers), [(true, false); 2]);
        assert!(engine.pair_needs_request_text());

        let both = TextNeeds::TEXT | TextNeeds::TEXT_WITH_TOKENS;
        engine.text_needs.store(both, Ordering::Release);
        assert_eq!(answers(&engine.get_encode_policy()), (true, true));
        assert_eq!(engine.classic_made_or_asked(), 0);
    }

    /// Under the policy engine the load poll follows what the document in
    /// force reads and a worker's cache-event stream what its own binding
    /// reads, whatever policy the classic settings name, and nothing is
    /// asked before an engine stands.
    #[test]
    fn under_the_policy_engine_loads_and_cache_events_follow_the_document() {
        let asked = |settings: PolicyConfig, preset: &str| {
            let registry =
                PolicyRegistry::for_policy_engine(settings, RoutingKeyOverrideConfig::default());
            let worker = worker_of("http://w1:8000", WorkerType::Regular);
            let ask = |registry: &PolicyRegistry| {
                (
                    LoadConsumer::needs_loads(registry),
                    registry.worker_needs_kv_events(worker.as_ref()),
                )
            };
            assert_eq!(ask(&registry), (false, false));
            let document = PolicySpec::preset(preset).expect("a stock preset");
            let spec = AppliedSpec::validate(&document).expect("a valid document");
            registry.set_engine(Box::leak(Box::new(PolicyEngine::new(spec))));
            assert_eq!(registry.classic_made_or_asked(), 0);
            ask(&registry)
        };
        let cache_aware = PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096);
        assert_eq!(asked(cache_aware, "round_robin"), (false, false));
        assert_eq!(asked(PolicyConfig::RoundRobin, "cache_aware"), (true, true));
    }

    /// Under the policy engine a supplied document's pairing is the mode
    /// the registry answers, by which every pair route reads its pair
    /// index: the engine pairs within the classes that index states. The
    /// configuration's own mode stands with the option off, and under the
    /// document the routing settings mean, which states that mode.
    #[test]
    fn under_the_policy_engine_a_supplied_documents_pairing_is_the_registrys() {
        let pair_mode = || RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        };
        assert_eq!(
            RouterConfig::default().pd_pairing_mode,
            PdPairingMode::Lenient
        );
        for (pairing, mode) in [
            (Pairing::Off, PdPairingMode::Off),
            (Pairing::Lenient, PdPairingMode::Lenient),
            (Pairing::Strict, PdPairingMode::Strict),
        ] {
            let mut document =
                PolicySpec::pair_presets("round_robin", "round_robin").expect("paired presets");
            document.plans.pd_pair.pairing = pairing;
            let supplied = RouterConfig {
                mode: pair_mode(),
                policy_controller: true,
                policy_spec: Some(document.clone()),
                ..RouterConfig::default()
            };
            // The document is the one in force beside the default settings.
            assert_eq!(supplied.policy_document_in_force(), Ok(document));
            assert_eq!(PolicyRegistry::of_config(&supplied).pd_pairing_mode(), mode);
            // With the option off no document is in force.
            let off = RouterConfig {
                policy_controller: false,
                ..supplied
            };
            assert_eq!(
                PolicyRegistry::of_config(&off).pd_pairing_mode(),
                PdPairingMode::Lenient
            );
        }
        let settings = RouterConfig {
            mode: pair_mode(),
            policy_controller: true,
            pd_pairing_mode: PdPairingMode::Strict,
            ..RouterConfig::default()
        };
        assert_eq!(
            PolicyRegistry::of_config(&settings).pd_pairing_mode(),
            PdPairingMode::Strict
        );
    }

    /// The registry of a gateway that was supplied a policy document knows
    /// it ([`PolicyRegistry::runs_supplied_document`]), with the document's
    /// path beside it or without. One made of the routing settings alone
    /// does not, with the policy controller on or off. Neither does one
    /// beside a document that the option leaves unread, nor one beside a
    /// path whose document was not loaded: the document in force there is
    /// the one the routing settings mean.
    #[test]
    fn a_registry_knows_whether_its_gateway_runs_a_supplied_document() {
        let on = RouterConfig {
            policy_controller: true,
            ..RouterConfig::default()
        };
        let supplied = RouterConfig {
            policy_spec: Some(PolicySpec::preset("round_robin").expect("a stock preset")),
            ..on.clone()
        };
        let path = || Some("policy.yaml".to_owned());
        let read_from_its_path = RouterConfig {
            policy_spec_path: path(),
            ..supplied.clone()
        };
        let not_loaded = RouterConfig {
            policy_spec_path: path(),
            ..on.clone()
        };
        assert_eq!(
            not_loaded.policy_document_in_force(),
            not_loaded.policy_document(),
            "the settings' document"
        );
        let unread = RouterConfig {
            policy_controller: false,
            ..supplied.clone()
        };
        let knows =
            |config: &RouterConfig| PolicyRegistry::of_config(config).runs_supplied_document();
        assert_eq!([&supplied, &read_from_its_path].map(knows), [true, true]);
        let settings_alone = [&on, &RouterConfig::default(), &unread, &not_loaded].map(knows);
        assert_eq!(settings_alone, [false; 4]);
        assert!(!PolicyRegistry::new(PolicyConfig::RoundRobin).runs_supplied_document());
    }

    /// The static needs the host answers before any request exists: a
    /// cache-aware default policy reads the request text, the load poll and
    /// a regular worker's KV events, and the pair's text question is the two
    /// pair policies'.
    #[test]
    fn a_cache_aware_default_needs_the_text_the_load_poll_and_the_kv_events() {
        let registry = PolicyRegistry::new(PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096));
        let regular = worker_of("http://w1:8000", WorkerType::Regular);
        assert!(registry.any_policy_needs_request_text(None));
        assert!(registry.needs_backend_loads());
        assert!(registry.worker_needs_kv_events(regular.as_ref()));
        assert!(registry.pair_needs_request_text());
    }

    /// Under the policy engine a worker's stream and a pair request's text
    /// follow what their own decisions read, a side at a time, as the
    /// classic answers follow the side policies. Under a document of two
    /// families the pair's binding asks nothing of a regular worker, and the
    /// binding behind it nothing of a pair. The stored text answer, which a
    /// single route reads with no model at hand, stays the document's.
    #[test]
    fn under_the_policy_engine_a_stream_and_a_pairs_text_follow_their_own_decisions() {
        let cache_aware = || PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 67_108_864);
        let round_robin = || PolicyConfig::RoundRobin;
        let under = |mode: RoutingMode, main: PolicyConfig, pairs_of: Option<&str>| {
            let settings = RouterConfig {
                mode,
                policy: main,
                ..RouterConfig::default()
            };
            let registry = PolicyRegistry::for_policy_engine(
                settings.policy.clone(),
                settings.routing_key_override.clone(),
            );
            let mut document = settings.policy_document().expect("the settings' document");
            if let Some(model) = pairs_of {
                document.bindings[0].matcher.model = model.to_owned();
            }
            let spec = AppliedSpec::validate(&document).expect("a valid document");
            registry.set_engine(Box::leak(Box::new(PolicyEngine::new(spec))));
            registry
        };
        let pairs = |prefill: PolicyConfig, decode: PolicyConfig| RoutingMode::PrefillDecode {
            prefill_urls: Vec::new(),
            decode_urls: Vec::new(),
            prefill_policy: Some(prefill),
            decode_policy: Some(decode),
        };
        let asked = |registry: &PolicyRegistry| {
            let stream =
                |kind| registry.worker_needs_kv_events(worker_of("grpc://w:9000", kind).as_ref());
            (
                [
                    stream(WorkerType::Prefill),
                    stream(WorkerType::Decode),
                    stream(WorkerType::Regular),
                    stream(WorkerType::Encode),
                ],
                registry.pair_needs_request_text(),
                registry.any_policy_needs_request_text(None),
            )
        };
        // Round-robin sides under a cache-aware main policy: the pair reads
        // no cache event and no text. The binding behind it is every other
        // phase's, and reads both.
        let quiet_pairs = under(pairs(round_robin(), round_robin()), cache_aware(), None);
        let streams = [false, false, true, true];
        assert_eq!(asked(&quiet_pairs), (streams, false, true));
        // The other way round.
        let reading_pairs = under(pairs(cache_aware(), cache_aware()), round_robin(), None);
        let streams = [true, true, false, false];
        assert_eq!(asked(&reading_pairs), (streams, true, true));
        assert_eq!(quiet_pairs.classic_made_or_asked(), 0);
        // One side that reads: its own workers stream, and the pair's text is
        // read whichever side it is.
        let reading_prefill = under(pairs(cache_aware(), round_robin()), round_robin(), None);
        let streams = [true, false, false, false];
        assert_eq!(asked(&reading_prefill), (streams, true, true));
        let reading_decode = under(pairs(round_robin(), cache_aware()), round_robin(), None);
        let streams = [false, true, false, false];
        assert_eq!(asked(&reading_decode), (streams, true, true));
        // A second engine is not taken: every answer stays the first one's.
        let quiet = RouterConfig {
            mode: pairs(round_robin(), round_robin()),
            policy: cache_aware(),
            ..RouterConfig::default()
        };
        let document = quiet.policy_document().expect("the settings' document");
        let spec = AppliedSpec::validate(&document).expect("a valid document");
        reading_decode.set_engine(Box::leak(Box::new(PolicyEngine::new(spec))));
        assert_eq!(asked(&reading_decode), (streams, true, true));

        // A pair binding of one model that reads, ahead of a quiet binding
        // of every model. The pair's answer names no model, so the text is
        // read. A stream is its worker's model's: a prefill worker of that
        // model streams, and one of another model does not.
        let one_models = under(
            pairs(cache_aware(), cache_aware()),
            round_robin(),
            Some("m"),
        );
        assert_eq!(asked(&one_models), ([false; 4], true, true));
        let its_own = worker_with_model_aliases("grpc://m:9000", "m", &[], WorkerType::Prefill);
        assert!(one_models.worker_needs_kv_events(its_own.as_ref()));

        // Three sides are one binding, and its encode phase is the encode
        // side's own: an encode worker alone streams.
        let three_sides = RoutingMode::EncodePrefillDecode {
            encode_urls: Vec::new(),
            prefill_urls: Vec::new(),
            decode_urls: Vec::new(),
            encode_policy: Some(cache_aware()),
            prefill_policy: Some(round_robin()),
            decode_policy: Some(round_robin()),
        };
        let reading_encode = under(three_sides, round_robin(), None);
        let streams = [false, false, false, true];
        assert_eq!(asked(&reading_encode), (streams, false, true));
    }

    /// An encode worker's KV-event question is the encode policy's own
    /// answer, not the default policy's: the encode policy places the
    /// requests on it.
    #[test]
    fn an_encode_workers_kv_event_need_is_the_encode_policys_own() {
        let encode = worker_of("grpc://encode:9000", WorkerType::Encode);
        let regular = worker_of("http://w1:8000", WorkerType::Regular);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        // The encode router publishes the mode's encode policy at start-up.
        policies.set_encode_policy(cache_aware_policy());
        assert!(
            policies.worker_needs_kv_events(encode.as_ref()),
            "the encode policy reads the KV events"
        );
        assert!(
            !policies.worker_needs_kv_events(regular.as_ref()),
            "the round-robin default reads none"
        );
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
        let honors = |name: &str| {
            PolicyFactory::create_by_name(name)
                .expect("a policy's name")
                .honors_routing_key()
        };
        assert!(!honors("cache_aware"));
        assert!(!honors("prefix_hash"));
        assert!(!honors("least_load"));
        assert!(honors("manual"));
        assert!(honors("consistent_hashing"));
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
            assert!(policy.honors_routing_key(), "{name}");
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
                assert_eq!(manual.entry_count(), 1);
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
    /// stand-in here is a test policy that says the same of itself) keeps the
    /// requirement under a valid key where `cache_aware` loses it, as a
    /// per-model policy beside the default; the requirement leaves with it.
    /// (No hint names the stand-in, so it goes into the map directly and the
    /// test refreshes what registration and removal refresh.)
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

    /// A text policy the routing-key override leaves alone: it says it
    /// places a keyed request by the key itself.
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
            "text_by_its_own_key"
        }

        fn honors_routing_key(&self) -> bool {
            true
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

        assert_eq!(reg.routing_key_sticky.as_ref().unwrap().entry_count(), 1);

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

    /// The sticky decision line and the two removal lines name a worker
    /// registered with a password and a query token by its masked address.
    #[test]
    #[traced_test]
    fn the_debug_lines_name_a_worker_masked() {
        // What tells each line among the test's lines.
        const LINES: [(&str, &str); 3] = [
            ("the sticky decision", "Sticky routing decision"),
            (
                "the removal from a model's policy",
                "from cache-aware policy for model",
            ),
            (
                "the removal from a role's policy",
                "from prefill cache-aware policy",
            ),
        ];
        let address = |host: &str| format!("http://user:s3cret@{host}:8000/v1?token=s3cret");

        let registry = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            rid_override(ManualAssignmentMode::Delegate),
        );
        let workers = vec![
            worker_of(&address("w1"), WorkerType::Regular),
            worker_of(&address("w2"), WorkerType::Regular),
        ];
        let info = SelectWorkerInfo {
            rid_key: Some("conv1"),
            ..Default::default()
        };
        registry
            .select_worker(&registry.get_default_policy(), &workers, &info)
            .unwrap();

        let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
        registry.on_worker_added("llama-3", Some("cache_aware"));
        registry.remove_worker_from_cache_aware("llama-3", &address("w3"));
        registry.set_prefill_policy(cache_aware_policy());
        registry.remove_worker_from_pd_cache_aware(&address("w4"));

        logs_assert(|lines: &[&str]| {
            let written = |told_by: &str| -> Vec<&str> {
                lines
                    .iter()
                    .copied()
                    .filter(|line| line.contains(told_by))
                    .collect()
            };
            let unwritten: Vec<&str> = LINES
                .iter()
                .filter(|(_, told_by)| written(told_by).is_empty())
                .map(|(line, _)| *line)
                .collect();
            if !unwritten.is_empty() {
                return Err(format!("lines nothing wrote: {unwritten:#?}"));
            }
            let whole: Vec<&str> = LINES
                .iter()
                .filter(|(_, told_by)| {
                    written(told_by)
                        .iter()
                        .any(|line| !line.contains("http://***@") || line.contains("s3cret"))
                })
                .map(|(line, _)| *line)
                .collect();
            if whole.is_empty() {
                Ok(())
            } else {
                Err(format!(
                    "lines that do not name their worker masked: {whole:#?}"
                ))
            }
        });
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
            WorkerLoadGuard::with_key(workers[pinned].clone(), None),
            WorkerLoadGuard::with_key(workers[pinned].clone(), None),
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

    /// What the load monitor is told through `LoadConsumer`: loads are wanted
    /// when a policy reads them, and when a DP-rank policy is set though no
    /// policy reads them.
    #[test]
    fn loads_are_wanted_for_a_policy_that_reads_them_or_a_dp_rank_policy() {
        let plain = PolicyRegistry::new(PolicyConfig::RoundRobin);
        assert!(!LoadConsumer::needs_loads(&plain));

        let reading = PolicyRegistry::new(PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096));
        assert!(LoadConsumer::needs_loads(&reading));

        let ranked = PolicyRegistry::new(PolicyConfig::RoundRobin);
        ranked.set_dp_rank_policy(Arc::new(crate::policies::MinimumTokensPolicy::new(None)));
        assert!(!ranked.needs_backend_loads());
        assert!(LoadConsumer::needs_loads(&ranked));
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

        use crate::policies::kv_index::event_monitor::KvEventMonitor;

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

    /// The registry builds a pair's side policies, from the side and the
    /// side's settings.
    #[test]
    fn the_registry_builds_a_pairs_sides_from_their_settings() {
        use crate::policies::RoundRobinPolicy;

        let classic = PolicyRegistry::new(PolicyConfig::RoundRobin);
        classic.set_side_policy(Side::Prefill, &PolicyConfig::RoundRobin);
        classic.set_side_policy(Side::Decode, &PolicyConfig::Random);
        assert!(classic
            .get_prefill_policy()
            .as_any()
            .is::<RoundRobinPolicy>());
        assert_eq!(classic.get_decode_policy().name(), "random");
    }

    /// The PD/EPD policies are set after the app context wired the shared state
    /// into the registry (the KV event monitor, the load feed, and the mesh
    /// tree bridge that `MeshAdapters::start` attaches), so the policy setters
    /// hand all of it over themselves.
    #[tokio::test]
    async fn pd_role_policies_receive_the_shared_state_set_before_them() {
        use std::collections::BTreeMap;

        use smg_mesh::MeshKV;

        use crate::{mesh::MeshAdapters, routers::common::snapshot::WorkerRegistry};

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

    /// A registry made for the policy engine keeps the token trees the
    /// engine is lent: none until the start asks, one owner however often
    /// it asks, under the numbers of the document the start hands it. The
    /// trees are made with their task: a worker the worker registry lost is
    /// taken out of them, and the call that takes a worker out of a
    /// cache-aware policy's trees finds no policy in such a registry.
    #[tokio::test]
    async fn a_worker_that_left_is_taken_out_of_the_lent_trees() {
        let config = RouterConfig {
            policy: cache_aware_config(),
            ..RouterConfig::default()
        };
        let document = config.policy_document().expect("the settings' document");
        let registry = PolicyRegistry::for_policy_engine(
            config.policy.clone(),
            RoutingKeyOverrideConfig::default(),
        );
        let workers = WorkerRegistry::new();
        for url in ["http://w1:8000", "http://w2:8000"] {
            workers.register(ready_worker(url, "llama-3"));
        }
        assert!(registry.lent_trees().is_none());
        let trees = registry.lend_token_trees(&document, 8, &workers);
        assert!(Arc::ptr_eq(
            &trees,
            &registry.lend_token_trees(&document, 8, &workers)
        ));
        let tokens: Vec<u32> = (0..64).collect();
        let tree = trees.token_tree("llama-3");
        assert_eq!(tree.page_size(), 8, "the block size the start handed");
        tree.insert_tokens(&tokens, "http://w1:8000");
        tree.insert_tokens(&tokens, "http://w2:8000");

        workers
            .remove_by_url("http://w1:8000")
            .expect("the removal");
        let bound = Instant::now() + Duration::from_secs(60);
        while tree.get_tenant_token_counts().len() > 1 {
            assert!(Instant::now() < bound, "the trees' task never purged");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let kept: Vec<String> = tree.get_tenant_token_counts().into_keys().collect();
        assert_eq!(kept, ["http://w2:8000"]);
        // The removal step makes its call with the engine on too.
        let left = HashSet::from(["http://w1:8000".to_owned()]);
        registry.remove_workers_from_cache_aware("llama-3", &left);
        assert_eq!(registry.classic_made_or_asked(), 0);
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
        use smg_worker::identity::basic::WorkerLoadGuard;

        use crate::routers::grpc::backend_client::BasicWorkerBuilder;

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

        drop(WorkerLoadGuard::with_key(Arc::clone(&prefill), None));
        drop(WorkerLoadGuard::with_key(Arc::clone(&decode), None));
        // The regular worker's model policy (random) accepts the completion silently.
        drop(WorkerLoadGuard::with_key(Arc::clone(&regular), None));

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
        drop(WorkerLoadGuard::with_key(Arc::clone(&prefill), None));
        assert_eq!(prefill_policy.completed.lock().unwrap().len(), 1);
    }

    #[test]
    fn reconcile_in_flight_reaches_the_policy_that_owns_the_worker_type() {
        use smg_worker::identity::basic::WorkerLoadGuard;

        use crate::routers::grpc::backend_client::BasicWorkerBuilder;

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
            WorkerLoadGuard::with_key(Arc::clone(&decode), None),
            WorkerLoadGuard::with_key(Arc::clone(&decode), None),
        ];
        registry.reconcile_in_flight(decode.as_ref());
        drop(held);
        // The second time as the worker's monitor asks it: through the trait.
        let consumer: &dyn LoadConsumer = registry.as_ref();
        consumer.reconcile_in_flight(decode.as_ref());
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

    /// Hands `reg` an engine over a stock preset's document, as the
    /// gateway's start hands the one it started.
    fn preset_engine_on(reg: &PolicyRegistry, preset: &str) {
        use smg_policy_controller::document::{AppliedSpec, PolicySpec};

        use crate::routers::common::placement::tests::engine_on;

        let document = PolicySpec::preset(preset).expect("a stock preset");
        engine_on(
            reg,
            AppliedSpec::validate(&document).expect("a valid document"),
        );
    }

    /// Where the policy engine places every request its document answers
    /// the text questions, from the call that sets the engine on, and the
    /// settings' policy is asked nothing: each row's settings say the
    /// opposite of its document. A document reads the text of a request
    /// that carries no token ids where a stage hashes the text's windows
    /// (`cache_aware`, `prefix_hash`) or places by the prompt's size
    /// (`bucket`), and no stock stage reads the text beside token ids. A
    /// routing key under the sticky override lifts nothing: the document's
    /// stages place a key that follows no session yet. A valid
    /// routing-tokens hint lifts the need as it does under a classic
    /// policy, and a change of the membership keeps the document's answer.
    ///
    /// Both registries answer so: the one a gateway makes where the policy
    /// controller runs, and a classic one handed an engine. The first
    /// answers nothing before its engine is set, hands the pair's question
    /// the pairs' own stored answer through its stand-in, and makes no
    /// classic policy object. Under one binding of every phase that answer
    /// is the document's. The second asks its two side policies.
    #[test]
    fn an_engines_document_answers_the_text_questions() {
        type Made = fn(PolicyConfig, RoutingKeyOverrideConfig) -> PolicyRegistry;
        let kinds: [(Made, bool); 2] = [
            (PolicyRegistry::for_policy_engine, false),
            (PolicyRegistry::with_override, true),
        ];
        let keyed = headers_with_routing_key("session-A");
        let hinted = headers_with_tokens("1,2,3");
        for (preset, reads) in [
            ("round_robin", false),
            ("consistent_hashing", false),
            ("cache_aware", true),
            ("prefix_hash", true),
            ("bucket", true),
        ] {
            for (made, classic) in kinds {
                let settings = if reads {
                    PolicyConfig::RoundRobin
                } else {
                    cache_aware_config()
                };
                let reg = made(settings, enabled_override());
                let before = classic && !reads;
                assert_eq!(reg.any_policy_needs_request_text(None), before, "{preset}");
                preset_engine_on(&reg, preset);
                let answers = |reg: &PolicyRegistry| {
                    [
                        reg.any_policy_needs_request_text(None),
                        reg.any_policy_needs_request_text(Some(&keyed)),
                        reg.any_policy_needs_request_text(Some(&hinted)),
                        reg.any_policy_needs_request_text_with_tokens(None),
                        reg.any_policy_needs_request_text_with_tokens(Some(&keyed)),
                    ]
                };
                let documents = [reads, reads, false, false, false];
                assert_eq!(answers(&reg), documents, "{preset}");
                // A model's first worker and its model's leaving refresh
                // the stored answer: it stays the document's.
                let _ = reg.on_worker_added("m", Some("cache_aware"));
                assert_eq!(answers(&reg), documents, "{preset}: a worker added");
                reg.on_model_unserved("m");
                assert_eq!(answers(&reg), documents, "{preset}: its model unserved");
                // A classic registry asks its two side policies, here the
                // settings' one: the pairs' stored answer is never its own.
                let pairs = if classic { !reads } else { reads };
                assert_eq!(reg.pair_needs_request_text(), pairs, "{preset}");
                if !classic {
                    assert_eq!(reg.classic_made_or_asked(), 0, "{preset}");
                }
            }
        }
    }

    /// The name the registry answers for a pair's side is the one the
    /// settings give the side's policy, kept where the side is set, and the
    /// first name set stands as the first policy does. A side the settings
    /// give no policy answers what its getter falls back to: the main
    /// policy, and `consistent_hashing` for an encode side. Under the
    /// classic policies the name is the one the side's policy object
    /// answers for itself. A registry made for the policy engine answers
    /// the same names and makes no policy object for them.
    #[test]
    fn the_settings_name_each_side_of_a_pair() {
        let sides = [Side::Prefill, Side::Decode, Side::Encode];
        let names = |reg: &PolicyRegistry| sides.map(|side| reg.side_policy_name(side));
        let classic = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let engine = PolicyRegistry::for_policy_engine(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig::default(),
        );
        for reg in [&classic, &engine] {
            assert_eq!(
                names(reg),
                ["round_robin", "round_robin", "consistent_hashing"]
            );
            reg.set_side_policy(Side::Prefill, &PolicyConfig::Random);
            reg.set_side_policy(Side::Decode, &PolicyConfig::ConsistentHashing);
            reg.set_side_policy(Side::Encode, &PolicyConfig::Passthrough);
            let set = ["random", "consistent_hashing", "passthrough"];
            assert_eq!(names(reg), set);
            for side in sides {
                reg.set_side_policy(side, &PolicyConfig::RoundRobin);
            }
            assert_eq!(names(reg), set, "the first name set stands");
        }
        let objects = [
            classic.get_prefill_policy().name(),
            classic.get_decode_policy().name(),
            classic.get_encode_policy().name(),
        ];
        assert_eq!(names(&classic), objects);
        assert_eq!(engine.classic_made_or_asked(), 0);
    }

    /// The name the registry answers for the settings' main policy is the
    /// name that policy's object answers for itself, for every policy the
    /// settings can name: a selection the policy engine counts under the
    /// settings' name is counted where a classic selection counts it.
    #[test]
    fn the_settings_name_each_policy_as_it_names_itself() {
        for name in [
            "random",
            "round_robin",
            "passthrough",
            "cache_aware",
            "power_of_two",
            "least_load",
            "bucket",
            "manual",
            "consistent_hashing",
            "prefix_hash",
        ] {
            // The two policies whose settings name fields with no default.
            let mut settings = serde_json::json!({
                "type": name,
                "balance_abs_threshold": 32,
                "balance_rel_threshold": 1.1,
            });
            match name {
                "cache_aware" => {
                    settings["cache_threshold"] = 0.5.into();
                    settings["eviction_interval_secs"] = 0.into();
                    settings["max_tree_size"] = 4096.into();
                }
                "bucket" => settings["bucket_adjust_interval_secs"] = 5.into(),
                _ => settings = serde_json::json!({ "type": name }),
            }
            let settings: PolicyConfig =
                serde_json::from_value(settings).expect("the policy's settings");
            let reg = PolicyRegistry::new(settings);
            assert_eq!(reg.policy_name(), name);
            assert_eq!(reg.policy_name(), reg.get_default_policy().name());
        }
    }
}
