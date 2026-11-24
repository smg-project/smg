//! Worker placement shared by the HTTP and gRPC families.
//!
//! Every family used to carry its own copy of the same sequences: take the
//! routing pool for the model, narrow it to the retained wire on a retry,
//! drop unavailable workers unless the policy does that itself, hand the
//! survivors to the policy registry, record the selection; and for a
//! disaggregated request, do that per role from one snapshot under one
//! runtime. This module is those sequences written once. A family still
//! decides which pools it draws from and what a failure means on its wire.
//!
//! Distinct from [`super::worker_selection`], the least-load selector with
//! refresh-on-miss that the provider and realtime paths use; this is the
//! policy-registry path over the self-hosted routing pools.

use std::{fmt, ops::Deref, sync::Arc, time::Duration};

use axum::{
    http::{header::RETRY_AFTER, HeaderMap, HeaderValue},
    response::Response,
};
use parking_lot::Mutex;
use rand::RngExt;
use smg_policy_controller::{
    document::DecodePick,
    engine::{
        LentList, PairAdmission, PairLists, Phase, PlaceRequest, PolicyEngine, Unplaced,
        WorkerLists,
    },
    policies::consistent_hashing::LentRing,
    tracking::{Placement, PlacementSide},
};
use smg_worker::{
    health::overload::{
        BRANCH_ALL_OVERLOADED_FALLBACK, BRANCH_ALL_STALLED_FALLBACK, STAGE_SELECTION,
    },
    identity::basic::{
        ConnectionMode, ConnectionModeExt, RuntimeType, Worker, WorkerLoadGuard, WorkerType,
    },
};
use tracing::{debug, warn};

use crate::{
    middleware::ReportedRefusal,
    observability::{
        cache_trace,
        log_throttle::KeyedLogThrottle,
        metrics::{metrics_labels, Metrics},
    },
    policies::{
        as_consistent_hashing,
        hash_ring::{HashRing, Unfilled},
        policy_filters_unavailable_workers, policy_reads_hash_ring, CacheNamespace,
        LoadBalancingPolicy, NamedList, PolicyRegistry, SelectWorkerInfo, WorkerRole,
    },
    routers::{
        common::{
            header_utils, overload,
            pair_index::PdPairIndex,
            pd_admission,
            place_request::{place_request, tried_ids, TriedIds},
            prefill_admission::{PrefillCandidateError, PrefillSelectionContext},
            retry::mark_non_retryable,
            snapshot::{ModelWorkerSnapshot, RoutingPool, WorkerRegistry, WorkerRegistryExt},
        },
        error,
    },
};

/// The wire a retained plan was built for. Retry re-selection filters
/// candidates to this (runtime, transport): the plan's proto flavor and its
/// stop-resolution are wire-specific and cannot be rebuilt post-drop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WireConstraint {
    pub runtime: RuntimeType,
    pub connection: ConnectionMode,
    /// The retained plan carries media references. Placement pins only the
    /// runtime and transport; the gRPC selection helpers derive their
    /// candidate predicate from this flag.
    pub requires_media_refs: bool,
    /// With `requires_media_refs`: which of the workers that process them the
    /// plan may use; the gRPC selection helpers derive their candidate
    /// predicate from it as from the flag above.
    pub pipeline: PipelinePin,
}

/// Which of the workers that process media references a forwarded plan may
/// use: those running the gateway's own pipeline alone (a family whose anchor
/// only that pipeline expands), those running the engine's own processor
/// alone (audio, which the pipeline does not take), or either.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PipelinePin {
    #[default]
    Any,
    Only,
    Never,
}

impl WireConstraint {
    /// Whether `worker` is on `wire`: its runtime and its transport when a
    /// retry pins one, any worker when none is pinned. The one predicate
    /// every narrowing to a retained plan's wire goes through, so the pools
    /// a placement searches and the ones it judges cannot drift apart.
    pub(crate) fn admits(wire: Option<Self>, worker: &dyn Worker) -> bool {
        wire.is_none_or(|wire| {
            worker.metadata().spec.runtime_type == wire.runtime
                && *worker.connection_mode() == wire.connection
        })
    }
}

/// Whether an explicit target under consistent hashing holds the request
/// to that worker: it indexes the unfiltered candidate set and stays
/// strict, so the request waits for that worker (checked after selection,
/// or at the gate) instead of moving to another one. The one spelling of
/// the rule for the admission pre-filters and the retry's narrowing.
pub(crate) fn strict_target(policy: &dyn LoadBalancingPolicy, headers: Option<&HeaderMap>) -> bool {
    as_consistent_hashing(policy).is_some()
        && header_utils::extract_target_worker(headers).is_some()
}

/// Whether an admission pre-filter may drop full workers from a role's
/// candidate set before `policy` runs over it (Prefill admission's capacity,
/// the decode pick's rooms): never under a [`strict_target`].
pub(crate) fn admission_prefilters(
    policy: &dyn LoadBalancingPolicy,
    headers: Option<&HeaderMap>,
) -> bool {
    !strict_target(policy, headers)
}

/// A per-request candidate predicate: a worker it rejects is never selected.
pub(crate) type CandidateFilter<'a> = &'a (dyn Fn(&dyn Worker) -> bool + Sync);

/// The read of a model's pair index as it stands now, for a pair's decode
/// asked again behind its prefill ([`decode_again`]). The read costs a
/// snapshot and an index, so a route hands it as a call: the policy
/// engine's arm alone makes it, once, and a classic pair reads nothing.
pub(crate) type PairsAgain<'a> = &'a dyn Fn() -> Arc<PdPairIndex>;

/// What a route states of its request for the policy engine alone: the
/// facts no classic policy reads, which [`PlacementInputs`] does not hold.
/// A route hands a placement function a call that states them. The function
/// hands the call on unmade to the engine's arm, which stands out of line
/// and makes it, as it reads a pair index again ([`PairsAgain`]): a classic
/// placement reads none of it, and the call's own reads of the body are
/// not made for one. A route's call takes what it reads by value (`move`),
/// so that nothing of the request is kept in memory for it on the way to
/// a classic placement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Stated {
    /// The most tokens the request's answer may hold, as its body states
    /// the limit: one answer's, whatever the request's count of samples or
    /// prompts. Zero states none.
    pub max_output_tokens: u32,
}

impl Stated {
    /// Nothing stated: for a caller that parsed no body, and for a case.
    #[inline]
    pub(crate) fn nothing() -> Self {
        Self::default()
    }

    /// The body's output limit as its reader answers it
    /// (`GenerationRequest::max_output_tokens`), and nothing else.
    #[inline]
    pub(crate) fn of_limit(limit: Option<u32>) -> Self {
        Self {
            max_output_tokens: limit.unwrap_or(0),
        }
    }
}

/// Everything a single-worker placement reads from the request.
#[derive(Clone, Copy, Default)]
pub(crate) struct PlacementInputs<'a> {
    /// Request text for cache-aware routing.
    pub text: Option<&'a str>,
    /// Tokenized request, or a valid routing-tokens hint, for prefix hashing.
    pub tokens: Option<&'a [u32]>,
    /// Request headers for header-based policies and the routing-key hint.
    pub headers: Option<&'a HeaderMap>,
    /// Session key derived from the body's `rid`.
    pub rid_key: Option<&'a str>,
    /// The body's `rid` as the request states it, its retry and turn
    /// suffixes with it: the request's own id. No classic policy reads it.
    /// The policy engine is handed it and derives the routing value by its
    /// document's own rule, so it is stated whether or not the sticky
    /// override derives a key. The HTTP pair route alone states the derived
    /// key here, and so states an id only where the override derives one.
    pub rid: Option<&'a str>,
    /// The request's cache partition, when set.
    pub cache_namespace: Option<CacheNamespace>,
    /// Extra per-request candidate predicate; a worker it rejects is never
    /// selected (e.g. only workers that process media references themselves,
    /// or none that already answered this request definitively). It is asked
    /// once a candidate in a pass of the selection, which runs once more when
    /// it found nobody available and a candidate has come free since (see
    /// `select_among`), and while it refuses nobody the selection runs over
    /// the pool as drawn.
    pub candidate_filter: Option<CandidateFilter<'a>>,
    /// Engines (by base URL) a failed attempt of this request already
    /// dispatched to. A retry skips them while another available candidate
    /// is left to take the request, so a policy that would re-select the
    /// same worker deterministically (a routing key, a cached prefix) moves
    /// on instead of replaying on the worker that just failed. A pool with
    /// nothing else left is used whole, so a single worker is retried as
    /// before. A pair placement judges each side on its own: the prefill
    /// side skips them while another open prefill is left, the decode side
    /// while the picked prefill has another open partner, each after its
    /// admission filter, so a tried worker with a place for the request is
    /// not given up for one that has none.
    pub tried: &'a [String],
}

/// The pool a placement draws from, before the availability filter.
///
/// Borrowed straight from the registry when no wire is pinned, so the hot
/// path does not clone the snapshot per request.
pub(crate) enum Candidates {
    Shared(Arc<[Arc<dyn Worker>]>),
    Pinned(Vec<Arc<dyn Worker>>),
}

impl Candidates {
    /// `pool` whole, or its part on `wire` when a retry pins the retained
    /// plan's runtime and transport.
    pub(crate) fn of(pool: Arc<[Arc<dyn Worker>]>, wire: Option<WireConstraint>) -> Self {
        match wire {
            None => Self::Shared(pool),
            Some(_) => Self::Pinned(
                pool.iter()
                    .filter(|w| WireConstraint::admits(wire, w.as_ref()))
                    .cloned()
                    .collect(),
            ),
        }
    }

    pub(crate) fn as_slice(&self) -> &[Arc<dyn Worker>] {
        match self {
            Self::Shared(shared) => shared,
            Self::Pinned(pinned) => pinned,
        }
    }

    /// The snapshot's own list, when the candidates are that list whole.
    #[cfg(test)]
    fn shared(&self) -> Option<&Arc<[Arc<dyn Worker>]>> {
        match self {
            Self::Shared(shared) => Some(shared),
            Self::Pinned(_) => None,
        }
    }
}

/// The candidates of one selection as a part of a list the routing snapshot
/// owns: each kept worker beside its position in that list. Every narrowing
/// goes through here, so a position never names another worker than the
/// handle beside it, and the policy is told which list its candidates are
/// cut from ([`NamedList`]).
pub(crate) struct Narrowed<'a> {
    list: &'a Arc<[Arc<dyn Worker>]>,
    positions: Vec<usize>,
    workers: Vec<Arc<dyn Worker>>,
}

impl<'a> Narrowed<'a> {
    /// The workers of `list` at `positions`, which ascend: a policy finds a
    /// candidate by its position, and nothing downstream sorts them.
    fn at(list: &'a Arc<[Arc<dyn Worker>]>, positions: Vec<usize>) -> Self {
        debug_assert!(
            positions.is_sorted_by(|a, b| a < b),
            "a list's part is named in the list's order"
        );
        let workers = positions.iter().map(|&at| Arc::clone(&list[at])).collect();
        Self {
            list,
            positions,
            workers,
        }
    }

    /// The workers of `list` that `keep` accepts, in the list's order.
    pub(crate) fn of(
        list: &'a Arc<[Arc<dyn Worker>]>,
        mut keep: impl FnMut(&Arc<dyn Worker>) -> bool,
    ) -> Self {
        // Sized once for the whole list: two allocations a narrowing, however
        // many workers it keeps.
        let mut positions = Vec::with_capacity(list.len());
        let mut workers = Vec::with_capacity(list.len());
        for (at, worker) in list.iter().enumerate() {
            if keep(worker) {
                positions.push(at);
                workers.push(Arc::clone(worker));
            }
        }
        Self {
            list,
            positions,
            workers,
        }
    }

    /// The candidates, in the list's order.
    pub(crate) fn workers(&self) -> &[Arc<dyn Worker>] {
        &self.workers
    }

    /// The position in the list of candidate `candidate`.
    fn position(&self, candidate: usize) -> usize {
        self.positions[candidate]
    }

    /// The candidates as a policy is told of them.
    fn named(&self) -> NamedList<'_> {
        NamedList {
            list: self.list,
            positions: Some(&self.positions),
        }
    }

    /// Keep the candidates `keep` accepts, when it accepts some and refuses
    /// some; otherwise the candidates stay as they are. `keep` is asked
    /// once a candidate, the kept ones stay in their order, and nothing is
    /// allocated.
    fn narrow_to(&mut self, mut keep: impl FnMut(&dyn Worker) -> bool) {
        // The kept candidates move to the front, in their order; the others
        // end up behind them and are cut off only when there is someone to
        // keep and someone to cut.
        let mut kept = 0;
        for at in 0..self.workers.len() {
            if keep(self.workers[at].as_ref()) {
                self.workers.swap(kept, at);
                self.positions.swap(kept, at);
                kept += 1;
            }
        }
        if kept > 0 && kept < self.workers.len() {
            self.workers.truncate(kept);
            self.positions.truncate(kept);
        }
    }
}

/// Why a placement produced nothing, judged from exactly the pool selection
/// drew from.
#[derive(Debug)]
pub(crate) enum PlacementFailure {
    /// Nothing serves the model on this wire.
    NoCandidates,
    /// Every candidate vetoed the request as overloaded; carries the shed.
    AllOverloaded(Response),
    /// Candidates exist but none is available (health, circuit breaker).
    Unavailable,
    /// Available candidates existed but the named policy picked none.
    PolicyDeclined(&'static str),
    /// Both pools have workers, whatever their availability, but no prefill
    /// shares a KV transfer protocol with any decode (#2483): judged from
    /// membership alone, since it cannot change until membership does.
    /// Carries each pool's pairing keys and the components the pools
    /// disagreed on.
    NoCompatiblePair {
        prefill: Vec<String>,
        decode: Vec<String>,
        mismatches: Vec<String>,
    },
    /// Every prefill worker that could serve the request is at its Prefill
    /// admission limit. Selection was not run; the request waits for a slot.
    PrefillAtCapacity,
}

impl PrefillCandidateError for PlacementFailure {
    fn is_at_capacity(&self) -> bool {
        matches!(self, Self::PrefillAtCapacity)
    }
}

/// A selected prefill/decode pair.
pub(crate) struct Pair {
    pub prefill: Arc<dyn Worker>,
    pub decode: Arc<dyn Worker>,
    /// The selected prefill worker's runtime, which under a homogeneous
    /// caller is the runtime both pools run.
    pub runtime: RuntimeType,
}

/// A selected prefill/decode pair as the pair placement answers it
/// ([`select_pair`]): each worker as a [`Placed`]. A route makes each
/// side's load guard from that side's answer, in the place it made one from
/// the worker: the prefill's by [`Placed::guard`], and the decode's, which
/// it makes ahead of the decode's send, by [`Placed::ahead`] with
/// [`Placed::sent`] at that send.
///
/// Under the classic policies the two guards are the ones a route made
/// before. Where the policy engine placed the pair (`policy_controller:
/// on`), each guard keeps its own side of the engine's one placement: the
/// engine hears a side's dispatch where that side is sent and its end where
/// that side's guard is dropped, in whichever order the two fall.
pub(crate) struct PlacedPair {
    pub prefill: Placed,
    pub decode: Placed,
    /// The selected prefill worker's runtime ([`Pair::runtime`]).
    pub runtime: RuntimeType,
}

impl From<Pair> for PlacedPair {
    /// A pair the classic policies picked: the two handles, each with its
    /// one empty word.
    #[inline(always)]
    fn from(pair: Pair) -> Self {
        Self {
            prefill: pair.prefill.into(),
            decode: pair.decode.into(),
            runtime: pair.runtime,
        }
    }
}

impl PlacedPair {
    /// The pair the policy engine placed: each worker with its own side of
    /// the placement, both counted under the routing value the engine
    /// names. The decode's answer keeps what the request stated for the
    /// engine alone: it is the one asked again. `None` for a placement
    /// that names no decode, which the engine answers no pair's lists with.
    fn by_engine(placement: Placement<'_>, stated: Stated) -> Option<Self> {
        let routing_key = placement.routing_key;
        let placement = placement.kept();
        let prefill = Arc::clone(&placement.worker);
        let decode = placement.decode.clone()?;
        let runtime = prefill.metadata().spec.runtime_type;
        let (prefill_side, decode_side) = placement.sides();
        Some(Self {
            // A prefill's answer is asked nothing again.
            prefill: Placed::by_side(prefill, prefill_side, routing_key, Stated::nothing()),
            decode: Placed::by_side(decode, decode_side?, routing_key, stated),
            runtime,
        })
    }
}

/// Which role of a pair placement failed, and why. Boxed at the `Err`
/// site: a shed verdict carries a whole response.
#[derive(Debug)]
pub(crate) struct PairFailure {
    pub role: WorkerRole,
    pub verdict: PlacementFailure,
}

impl PrefillCandidateError for PairFailure {
    fn is_at_capacity(&self) -> bool {
        self.verdict.is_at_capacity()
    }
}

/// The candidates for `model_id` in `pool`, narrowed to `wire` when a retry
/// pins the retained plan's runtime and transport.
pub(crate) fn candidates(
    registry: &WorkerRegistry,
    model_id: &str,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
) -> Candidates {
    Candidates::of(registry.get_routing_pool(model_id, pool), wire)
}

/// The part of `candidates` a request's `filter` accepts, in their order,
/// where it refuses one of them. `None` where no filter is stated or it
/// refuses nobody (a media or context demand every worker meets): the
/// candidates stay as they were drawn and nothing is collected. Each
/// candidate is asked once.
///
/// The one narrowing by a request's own rule. The selection hands its policy
/// what is left.
#[inline]
pub(crate) fn accepted_by(
    candidates: &[Arc<dyn Worker>],
    filter: Option<CandidateFilter<'_>>,
) -> Option<Vec<Arc<dyn Worker>>> {
    let accepts = filter?;
    let first = candidates.iter().position(|w| !accepts(w.as_ref()))?;
    let (passed, rest) = candidates.split_at(first);
    let rest = rest[1..].iter().filter(|w| accepts(w.as_ref()));
    Some(passed.iter().chain(rest).cloned().collect())
}

/// The worker a single-request placement picked, as the placement functions
/// answer it: the request's worker, and what the request's load guard takes
/// from the answer ([`Placed::guard`]).
///
/// It reads as the worker: a route calls the worker's methods on it, and
/// takes the worker's shared handle by [`Placed::worker`]. A route makes the
/// request's load guard from it, in the place it made one from the worker.
///
/// Under the classic policies that guard is the one a route made before,
/// counted under the route's own key. Where the policy engine placed the
/// request (`policy_controller: on`), the guard is counted under the routing
/// value the engine names, and under the route's own key where it names
/// none. It keeps the engine's placement to the request's end: the engine
/// hears the dispatch where the guard is made and the end where the guard
/// is dropped. It is told no outcome: the worker object has it.
///
/// A classic answer is the worker's handle and one empty word: that word is
/// all a classic placement pays for the engine's side of this type.
#[derive(Debug)]
pub(crate) struct Placed {
    worker: Arc<dyn Worker>,
    /// What the request's load guard takes where the policy engine placed
    /// the request. `None` for a worker a classic policy picked, or no
    /// policy at all. `None` too for an answer of the engine whose guard is
    /// the classic one: the engine names no routing value for the request
    /// and is owed none of its events.
    engine: Option<Box<ByEngine>>,
}

/// What a load guard takes from an answer of the policy engine.
#[derive(Debug)]
struct ByEngine {
    /// The engine's side of the request, where the engine is owed the
    /// request's events, until the request's load guard takes it. One that
    /// no guard took tells the engine a release where the answer is
    /// dropped.
    side: Mutex<Option<Box<PlacementSide>>>,
    /// The routing value the engine names for the request, which the guard
    /// counts it under. `None` where the engine names none: a value it
    /// keeps as a fingerprint alone, a document that reads no value. The
    /// guard then counts under its route's key.
    routing_key: Option<Box<str>>,
    /// What the request stated for the engine alone at the decision that
    /// made this answer, kept by the answer of a pair's decode. Asked again
    /// behind its prefill, that decode states it again from here: its
    /// route may hold the body no more by then.
    stated: Stated,
}

impl From<Arc<dyn Worker>> for Placed {
    /// A worker picked by a classic policy, or with no policy at all.
    #[inline]
    fn from(worker: Arc<dyn Worker>) -> Self {
        Self {
            worker,
            engine: None,
        }
    }
}

/// A worker a case holds by its own type, as [`Placed`] answers a worker
/// picked with no policy at all.
#[cfg(test)]
impl<W: Worker + 'static> From<Arc<W>> for Placed {
    fn from(worker: Arc<W>) -> Self {
        let worker: Arc<dyn Worker> = worker;
        Self::from(worker)
    }
}

impl Deref for Placed {
    type Target = dyn Worker;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.worker.as_ref()
    }
}

impl Placed {
    /// The answer for a request the policy engine placed: its worker, the
    /// engine's side of the request where the engine is owed the request's
    /// events, and the routing value, copied out of the request's own
    /// text. An answer that keeps neither is the answer of a classic pick.
    fn by_engine(
        worker: Arc<dyn Worker>,
        side: Option<Box<PlacementSide>>,
        routing_key: Option<&str>,
    ) -> Self {
        let routing_key = routing_key.map(Box::from);
        if side.is_none() && routing_key.is_none() {
            return Self::from(worker);
        }
        Self {
            worker,
            engine: Some(Box::new(ByEngine {
                side: Mutex::new(side),
                routing_key,
                // A single worker's answer is asked nothing again.
                stated: Stated::nothing(),
            })),
        }
    }

    /// The answer for one worker of a pair the policy engine placed: the
    /// worker with its own side of the pair's placement, counted under the
    /// routing value the engine names for the pair. `stated` is what the
    /// request stated for the engine alone at that decision.
    fn by_side(
        worker: Arc<dyn Worker>,
        side: PlacementSide,
        routing_key: Option<&str>,
        stated: Stated,
    ) -> Self {
        Self {
            worker,
            engine: Some(Box::new(ByEngine {
                side: Mutex::new(Some(Box::new(side))),
                routing_key: routing_key.map(Box::from),
                stated,
            })),
        }
    }

    /// The placed worker's shared handle: the one call that hands it. A
    /// load guard made from the handle keeps no placement: the request's
    /// guard is made by [`Self::guard`].
    #[inline]
    pub(crate) fn worker(&self) -> &Arc<dyn Worker> {
        &self.worker
    }

    /// The request's load guard on the placed worker: it raises the worker's
    /// load now and lowers it where it is dropped.
    ///
    /// `key` is the route's own key for the request, which a classic
    /// policy's answer is counted under. An answer of the policy engine is
    /// counted under the routing value the engine names, and under `key`
    /// where the engine names none: the count a classic policy that names
    /// no key leaves on the worker. The first guard made from an engine's
    /// answer takes the engine's side of the request and tells the engine
    /// the dispatch. A later one is a plain guard under the same value.
    #[inline]
    pub(crate) fn guard(&self, key: Option<&str>) -> WorkerLoadGuard {
        let worker = Arc::clone(&self.worker);
        match &self.engine {
            None => WorkerLoadGuard::with_key(worker, key),
            Some(engine) => engine.guard(worker, key),
        }
    }

    /// [`Self::guard`] for a caller that keeps nothing else of the answer:
    /// the worker's handle moves into the guard. The decision bench's routes
    /// and tests call it. A route keeps its answer and calls [`Self::guard`].
    #[cfg(any(test, feature = "test-util"))]
    #[inline]
    pub(crate) fn into_guard(self, key: Option<&str>) -> WorkerLoadGuard {
        match self.engine {
            None => WorkerLoadGuard::with_key(self.worker, key),
            Some(engine) => engine.guard(self.worker, key),
        }
    }

    /// This answer's load guard, made ahead of its send: the worker is
    /// counted from now, under the key [`Self::guard`] counts under, and
    /// the engine hears nothing yet. For a pair's decode, whose guard a
    /// route makes at the selection and whose request goes out behind its
    /// prefill. [`Self::sent`] says the send. A guard dropped unsent says
    /// nothing: the answer's own drop lets its side go, which the engine
    /// hears as a release.
    #[inline]
    pub(crate) fn ahead(&self, key: Option<&str>) -> WorkerLoadGuard {
        let worker = Arc::clone(&self.worker);
        match &self.engine {
            None => WorkerLoadGuard::with_key(worker, key),
            Some(engine) => WorkerLoadGuard::with_key(worker, engine.key(key)),
        }
    }

    /// The request went out under `ahead`, a guard [`Self::ahead`] made:
    /// the engine hears the dispatch now, and the guard keeps the
    /// placement's side, so the guard's drop is the side's end.
    /// [`Self::guard`] is `sent(ahead(key))`. A debug build stops where
    /// `ahead` counts another worker than this answer's.
    #[inline]
    pub(crate) fn sent(&self, ahead: WorkerLoadGuard) -> WorkerLoadGuard {
        debug_assert!(
            ahead.counts(&self.worker),
            "a guard is sent on the answer it was made ahead of"
        );
        match &self.engine {
            None => ahead,
            Some(engine) => engine.sent(ahead),
        }
    }

    /// Whether a dispatch that sends this answer, a pair's decode, behind
    /// its prefill owes it anything there: the ask ([`decode_again`]), and
    /// its send said apart from its guard ([`Self::ahead`], [`Self::sent`]).
    /// True where the policy engine placed the request. A classic answer is
    /// owed neither: asked again it stands, and [`Self::guard`] makes its
    /// guard whole. A route asks this where it decides its dispatch's
    /// shape, and sends a decode that is owed nothing as it does with no
    /// engine. This test of the answer's one word is then all the pair pays
    /// behind its prefill.
    #[inline]
    pub(crate) fn owed_behind_prefill(&self) -> bool {
        self.engine.is_some()
    }

    /// The worker alone, for a caller that sends it no request: a bench's
    /// placement, a test. An engine's side of the request is let go here,
    /// which the engine hears as a release. A route never calls this: it
    /// makes its guard by [`Self::guard`].
    #[cfg(any(test, feature = "test-util"))]
    #[inline]
    pub(crate) fn into_worker(self) -> Arc<dyn Worker> {
        self.worker
    }

    /// The worker of an answer whose load guard was made: the engine's
    /// side of the request is the guard's by now, so the answer has nothing
    /// left to give up, and the worker's handle moves on with no count. For
    /// a route that hands a pair's prefill on behind its guard. A debug
    /// build stops where the answer still holds its side: the request
    /// would go out under no guard of its own, and the engine would hear a
    /// release.
    #[inline]
    pub(crate) fn into_guarded_worker(self) -> Arc<dyn Worker> {
        debug_assert!(
            self.engine
                .as_ref()
                .is_none_or(|engine| engine.side.lock().is_none()),
            "a worker is handed on behind its guard"
        );
        self.worker
    }
}

impl ByEngine {
    /// The key a guard of this answer counts under: the routing value the
    /// engine names, and the route's own `key` where it names none.
    fn key<'a>(&'a self, key: Option<&'a str>) -> Option<&'a str> {
        self.routing_key.as_deref().or(key)
    }

    /// The load guard of a request the policy engine placed, out of line:
    /// a classic placement never comes here.
    #[inline(never)]
    fn guard(&self, worker: Arc<dyn Worker>, key: Option<&str>) -> WorkerLoadGuard {
        self.sent(WorkerLoadGuard::with_key(worker, self.key(key)))
    }

    /// The send of a request the policy engine placed, out of line as
    /// [`Self::guard`] is: the first guard sent takes the engine's side of
    /// the request and tells the engine the dispatch.
    #[inline(never)]
    fn sent(&self, guard: WorkerLoadGuard) -> WorkerLoadGuard {
        match self.side.lock().take() {
            Some(mut side) => {
                side.sent();
                guard.keeping(side)
            }
            None => guard,
        }
    }

    /// [`decode_again`] for a decode the policy engine placed, out of line.
    /// The engine decides the decode again on this answer's side of the
    /// pair, which is held for the time of that decision: the route's rule
    /// is asked under it, and touches neither of the pair's answers. What
    /// the request stated for the engine alone at the first decision is
    /// stated again as this answer kept it ([`Stated`]). A side
    /// that a guard took already was sent: it is asked nothing, and the
    /// pair index is not read.
    #[inline(never)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the pair placement's own arguments, and the decode's guard"
    )]
    fn decode_again(
        &self,
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        model_id: &str,
        pairs: PairsAgain<'_>,
        wire: Option<WireConstraint>,
        inputs: PlacementInputs<'_>,
        ahead: WorkerLoadGuard,
    ) -> (Option<Placed>, WorkerLoadGuard) {
        let Some(engine) = policies.engine() else {
            return (None, ahead);
        };
        let mut held = self.side.lock();
        let Some(side) = held.as_mut() else {
            return (None, ahead);
        };
        let pairs = pairs();
        let asked = with_pair_request(
            engine,
            registry,
            policies,
            model_id,
            &pairs,
            wire,
            None,
            None,
            inputs,
            self.stated,
            |request| side.decode_again(request),
        );
        let worker = match asked {
            Ok(worker) if !ahead.counts(&worker) => worker,
            Ok(_) => return (None, ahead),
            Err(unplaced) => {
                debug!(
                    model_id,
                    ?unplaced,
                    "A pair's decode stands as first decided"
                );
                return (None, ahead);
            }
        };
        let Some(side) = held.take() else {
            return (None, ahead);
        };
        drop(held);
        debug!(decode = %worker.redacted_url(), "A pair's decode moved behind its prefill");
        let guard = ahead.onto(Arc::clone(&worker));
        // The decode the request moved to is asked nothing again: the
        // engine decides a pair's decode again once, so its answer keeps
        // nothing of what the request stated.
        let routing_key = self.routing_key.as_deref();
        let moved = Placed::by_side(worker, *side, routing_key, Stated::nothing());
        (Some(moved), guard)
    }
}

/// The model's hash ring as one request lends it to the policy engine
/// ([`LentRing`]), read where the engine's document places by a ring and
/// nowhere else.
///
/// A classic ring policy is handed the model's ring for every list a
/// request is placed over: the pool's own, the part of it on a retry's
/// pinned wire, a caller's own slice (`select_pass`). The engine is lent
/// that ring for the same requests, a part of a pool held to by the
/// request's rule, and walks it as the classic policy does
/// ([`HashRing::find_healthy_url`]), so a routing value keeps the worker it
/// has.
enum ModelRing<'a> {
    /// The document places by no ring: none is read.
    Unread,
    /// The ring of the generation the request's list is read from, once it
    /// is filled. It is borrowed from that snapshot: the request takes no
    /// count of the ring.
    Own(&'a HashRing),
    /// The ring the registry answers until then, kept from another
    /// generation ([`PolicyRegistry::hash_ring_of`]).
    Kept(Arc<HashRing>),
    /// No ring is kept for the model, as for its first requests. A classic
    /// ring policy places such a request by its key's position over the
    /// eligible workers, and the engine places it so over a ring that holds
    /// no worker ([`Unfilled`]).
    Unfilled,
}

impl<'a> ModelRing<'a> {
    /// The ring for a request of `model_id` whose list is read from
    /// `generation`.
    fn of(
        engine: &PolicyEngine,
        policies: &PolicyRegistry,
        model_id: &str,
        generation: &'a ModelWorkerSnapshot,
    ) -> Self {
        if !engine.spec().needs().ring {
            return Self::Unread;
        }
        match generation.hash_ring() {
            Some(own) => Self::Own(own),
            None => Self::standing(policies.hash_ring_of(model_id, generation)),
        }
    }

    /// The ring for a request whose caller holds no snapshot: the one the
    /// registry answers for `model_id` now, as the classic selection reads
    /// it for such a list.
    fn read(
        engine: &PolicyEngine,
        policies: &PolicyRegistry,
        registry: &WorkerRegistry,
        model_id: &str,
    ) -> Self {
        if !engine.spec().needs().ring {
            return Self::Unread;
        }
        Self::standing(policies.hash_ring(registry, model_id))
    }

    /// The ring the registry answered for a generation whose own is not
    /// filled.
    fn standing(ring: Option<Arc<HashRing>>) -> Self {
        ring.map_or(Self::Unfilled, Self::Kept)
    }

    /// The ring as the engine's request carries it.
    fn lent(&self) -> Option<&dyn LentRing> {
        match self {
            Self::Unread => None,
            Self::Own(ring) => Some(*ring),
            Self::Kept(ring) => Some(&**ring),
            Self::Unfilled => Some(&Unfilled),
        }
    }
}

/// The lines that say the policy engine refused a decision, a model at a
/// time under the refusal lines' own allowance: an engine that refuses one
/// request may refuse the next ones alike.
static ENGINE_REFUSED_LOG: KeyedLogThrottle = KeyedLogThrottle::new(
    REFUSAL_LINES_PER_SECOND,
    Duration::from_secs(1),
    REFUSAL_LOG_MODELS,
);

/// Writes why the policy engine placed nothing for a request. A decision
/// the engine refused ([`Unplaced::Refused`]) is the engine's own fault and
/// not the fleet's state, a stage that panicked among them: its line is
/// written at `warn`. Every other answer says what the fleet could not
/// take, which the route's own refusal line reports, and is written at
/// `debug` under `placed_none`.
fn report_unplaced(model_id: &str, unplaced: &Unplaced, placed_none: &'static str) {
    if let Unplaced::Refused(refused) = unplaced {
        if let Some(admitted) = ENGINE_REFUSED_LOG.admit(model_id) {
            warn!(
                model_id,
                ?refused,
                suppressed = admitted.suppressed,
                "The policy engine refused a decision"
            );
        }
    } else {
        debug!(model_id, ?unplaced, "{placed_none}");
    }
}

/// The origin a row of a traced request's decision record states where the
/// policy engine picked. The gateway writes such a row behind the engine's
/// answer, and it holds no prediction, score or gate: the engine's stages
/// write none into the record.
const BY_ENGINE: &str = "engine";

/// One row of a traced request's decision record for a list the policy
/// engine was asked over: `seen`, the list as it stood ahead of the engine's
/// call ([`cache_trace::observe`]), under the name `policy` answers, with
/// the worker the engine picked from it, or none. Where the request is not
/// traced nothing is written and the name is not asked.
fn engine_row(
    seen: Option<cache_trace::Observed>,
    policy: impl FnOnce() -> &'static str,
    picked: Option<&Arc<dyn Worker>>,
) {
    if let Some(seen) = seen {
        cache_trace::selection_of(policy(), BY_ENGINE, seen, picked);
    }
}

/// A pair's side as a row of the decision record names it: the side as the
/// settings name its policy, and the side as a document names it.
type RowSide = (crate::policies::Side, Phase);

/// The prefill side of a pair's rows.
const PREFILL_ROW: RowSide = (crate::policies::Side::Prefill, Phase::Prefill);

/// The decode side of a pair's rows.
const DECODE_ROW: RowSide = (crate::policies::Side::Decode, Phase::Decode);

/// The name a row of a traced request's decision record states: `label`,
/// the label of the profile that picked. `side` is the pair's side the row
/// is of, `None` for a single request's row.
///
/// Where the gateway runs the document its routing settings mean, a row
/// reads as a classic one in two cases. The engine names no profile: the
/// row is under the name the settings give the main policy, or the side's
/// ([`PolicyRegistry::policy_name`], [`PolicyRegistry::side_policy_name`]).
/// The profile is named for its side, as that document names a pair's two
/// profiles, each the side's policy as the settings write it: the row is
/// under the side's policy too.
///
/// Under a supplied document ([`PolicyRegistry::runs_supplied_document`])
/// the settings name nothing that runs: the label stands as the engine
/// gives it, and a row whose profile the engine does not name states no
/// policy. The selection metric is not renamed here: it counts a pick
/// under its profile's label.
fn row_name(policies: &PolicyRegistry, side: Option<RowSide>, label: &'static str) -> &'static str {
    let for_its_side = side.is_some_and(|(_, own)| label == own.name());
    if policies.runs_supplied_document() || !(label.is_empty() || for_its_side) {
        return label;
    }
    side.map_or_else(
        || policies.policy_name(),
        |(side, _)| policies.side_policy_name(side),
    )
}

/// The policy engine's answer for a single request over `lent`, the
/// request's list with its cell: what `inputs` state of the request
/// is handed as the engine reads it ([`place_request`]), with `ring`, the
/// model's hash ring ([`ModelRing`]), and the engine places the request
/// with one call. `None` where the engine names no worker.
///
/// Records the selection metric under the chosen worker's own transport
/// label, as a classic selection does, and under the label of the profile
/// that picked. A decision made ahead of every profile names none: it is
/// counted under the name the settings give the main policy
/// ([`PolicyRegistry::policy_name`]), which is what a classic selection
/// counts a request under. The name is read off the settings: no policy
/// object is asked.
///
/// A traced request's decision record takes one row, as a classic
/// selection's does ([`PolicyRegistry::select_worker_for_model`]): the
/// list as it stands ahead of the engine's call, and the worker picked, or
/// no worker, under the row's name ([`row_name`]). Its origin is
/// [`BY_ENGINE`].
fn placed_by_engine(
    engine: &'static PolicyEngine,
    policies: &PolicyRegistry,
    model_id: &str,
    lent: LentList<'_>,
    ring: Option<&dyn LentRing>,
    inputs: PlacementInputs<'_>,
    stated: Stated,
) -> Option<Placed> {
    let mut tried = TriedIds::new();
    tried_ids(lent.workers, inputs.tried, &mut tried);
    let mut request = place_request(model_id, WorkerLists::Single(lent), &inputs, &tried, stated);
    request.ring = ring;
    let seen = cache_trace::observe(lent.workers);
    let answer = engine.place(&request);
    let placement = match &answer {
        Ok(placement) => placement,
        Err(unplaced) => {
            report_unplaced(model_id, unplaced, "The policy engine placed no worker");
            engine_row(seen, || row_name(policies, None, ""), None);
            return None;
        }
    };
    let counted_under = if placement.profile.is_empty() {
        policies.policy_name()
    } else {
        placement.profile
    };
    Metrics::record_worker_selection(
        metrics_labels::WORKER_REGULAR,
        placement.worker.connection_mode().as_metric_label(),
        model_id,
        counted_under,
    );
    let named = || row_name(policies, None, placement.profile);
    engine_row(seen, named, Some(&placement.worker));
    // A placement is moved whole only where the engine is owed the request's
    // events and keeps its side: of every other the worker is taken in place.
    let owed = placement.owes_events();
    match answer {
        Ok(placement) if owed => {
            let (worker, key) = (Arc::clone(&placement.worker), placement.routing_key);
            let side = Box::new(placement.kept().sides().0);
            Some(Placed::by_engine(worker, Some(side), key))
        }
        Ok(Placement {
            worker,
            routing_key,
            ..
        }) => Some(Placed::by_engine(worker, None, routing_key)),
        // Answered above, with its report: this arm only completes the match.
        Err(_) => None,
    }
}

/// [`select_single`] where the gateway started the policy engine, out of
/// line: the engine places the request over the pool's list of the model's
/// snapshot as it stands now. `stated` is the route's call, handed on
/// unmade.
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "the placement's own arguments, the engine it asks and what the request states for it"
)]
fn single_by_engine(
    engine: &'static PolicyEngine,
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
    inputs: PlacementInputs<'_>,
    stated: impl FnOnce() -> Stated,
) -> Option<Placed> {
    let snapshot = registry.get_routing_snapshot(model_id);
    single_of_by_engine(
        engine, policies, model_id, &snapshot, pool, wire, inputs, stated,
    )
}

/// [`select_single_of`] where the gateway started the policy engine, out of
/// line: the engine places the request over the pool's list as the snapshot
/// lends it ([`ModelWorkerSnapshot::lent`]), on the hash ring of that
/// snapshot ([`ModelRing::of`]). A retry pinned to its retained plan's wire
/// is placed over the same list, held to the wire by the request's rule
/// ([`WireConstraint::admits`], beside the request's own): the engine keeps
/// a list's turn and its records with the list, so they are the pool's.
/// The route's call that states the request ([`Stated`]) is made here.
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "the placement's own arguments, the engine it asks and what the request states for it"
)]
fn single_of_by_engine(
    engine: &'static PolicyEngine,
    policies: &PolicyRegistry,
    model_id: &str,
    snapshot: &ModelWorkerSnapshot,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
    inputs: PlacementInputs<'_>,
    stated: impl FnOnce() -> Stated,
) -> Option<Placed> {
    let stated = stated();
    let ring = ModelRing::of(engine, policies, model_id, snapshot);
    let ring = ring.lent();
    let lent = snapshot.lent(pool);
    if wire.is_none() {
        return placed_by_engine(engine, policies, model_id, lent, ring, inputs, stated);
    }
    let own = inputs.candidate_filter;
    let on_wire = move |worker: &dyn Worker| {
        WireConstraint::admits(wire, worker) && own.is_none_or(|accepts| accepts(worker))
    };
    let inputs = PlacementInputs {
        candidate_filter: Some(&on_wire),
        ..inputs
    };
    placed_by_engine(engine, policies, model_id, lent, ring, inputs, stated)
}

/// [`select_from`] where the gateway started the policy engine, out of
/// line: the engine places the request over `pool`'s own list of the model's
/// snapshot as it stands now, on that snapshot's hash ring
/// ([`ModelRing::of`]), held to the caller's slice by the request's rule: a
/// worker is taken where the slice holds that worker object, beside the
/// request's own rule. The engine keeps a list's turn and its records with
/// the list, so they are the pool's. A worker that joined the pool behind
/// the caller's read is in no slice and takes no such request, and a
/// slice's worker object that the pool's list does not hold, as one
/// replaced between the two reads, takes none either. The route's call
/// that states the request ([`Stated`]) is made here.
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "the placement's own arguments, the engine it asks and what the request states for it"
)]
fn from_by_engine(
    engine: &'static PolicyEngine,
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pool: RoutingPool,
    candidates: &[Arc<dyn Worker>],
    inputs: PlacementInputs<'_>,
    stated: impl FnOnce() -> Stated,
) -> Option<Placed> {
    let stated = stated();
    let snapshot = registry.get_routing_snapshot(model_id);
    let ring = ModelRing::of(engine, policies, model_id, &snapshot);
    let ring = ring.lent();
    let lent = snapshot.lent(pool);
    let own = inputs.candidate_filter;
    let in_slice = move |worker: &dyn Worker| {
        let at = std::ptr::from_ref(worker);
        let held = |kept: &Arc<dyn Worker>| std::ptr::addr_eq(Arc::as_ptr(kept), at);
        candidates.iter().any(held) && own.is_none_or(|accepts| accepts(worker))
    };
    let inputs = PlacementInputs {
        candidate_filter: Some(&in_slice),
        ..inputs
    };
    placed_by_engine(engine, policies, model_id, lent, ring, inputs, stated)
}

/// Pick one worker for `model_id` from `pool`, or `None` when nothing is
/// selectable. Records the selection metric under the chosen worker's own
/// transport label.
///
/// This function and its two siblings are thin fronts, always inlined
/// where a route calls them. Each reads the registry's engine once. Where
/// the gateway started none it calls the classic selection, which stands
/// as its own function as it did before the engine and answers the
/// worker's handle alone. Where it started one the engine places the
/// request, out of line.
///
/// `stated` states what the engine alone reads of the request
/// ([`Stated`]). The front hands the call on unmade to the engine's arm,
/// which stands out of line and makes it: a classic placement is handed
/// nothing of it, and a caller that states nothing compiles to the code
/// it had with no such argument.
#[inline(always)]
pub(crate) fn select_single(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
    inputs: PlacementInputs<'_>,
    stated: impl FnOnce() -> Stated,
) -> Option<Placed> {
    if let Some(engine) = policies.engine() {
        return single_by_engine(
            engine, registry, policies, model_id, pool, wire, inputs, stated,
        );
    }
    classic_single(registry, policies, model_id, pool, wire, inputs).map(Placed::from)
}

/// [`select_single`] by the classic policies. Never inlined into its front:
/// the function a classic placement runs is this one, compiled as it was
/// before the engine, whatever stands beside its call.
#[inline(never)]
fn classic_single(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
    inputs: PlacementInputs<'_>,
) -> Option<Arc<dyn Worker>> {
    let snapshot = registry.get_routing_snapshot(model_id);
    classic_single_of(registry, policies, model_id, &snapshot, pool, wire, inputs)
}

/// [`select_single`] over `pool` of a snapshot the caller read something
/// else from too (the pool's window floor), so that both come from one
/// membership generation. The pool is borrowed from the snapshot: the
/// selection takes no count of the list. It keeps its name on the way to
/// the policy.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the placement's own arguments, and what the request states for the engine"
)]
pub(crate) fn select_single_of(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    snapshot: &ModelWorkerSnapshot,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
    inputs: PlacementInputs<'_>,
    stated: impl FnOnce() -> Stated,
) -> Option<Placed> {
    if let Some(engine) = policies.engine() {
        return single_of_by_engine(
            engine, policies, model_id, snapshot, pool, wire, inputs, stated,
        );
    }
    classic_single_of(registry, policies, model_id, snapshot, pool, wire, inputs).map(Placed::from)
}

/// [`select_single_of`] by the classic policies.
fn classic_single_of(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    snapshot: &ModelWorkerSnapshot,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
    inputs: PlacementInputs<'_>,
) -> Option<Arc<dyn Worker>> {
    let list = snapshot.pool(pool);
    if wire.is_none() {
        let whole = NamedList {
            list,
            positions: None,
        };
        return select_among(registry, policies, model_id, list, Some(whole), inputs);
    }
    // A retry pinned to its retained plan's wire: a list of its own, which
    // nobody names.
    let pinned = Candidates::of(Arc::clone(list), wire);
    select_among(
        registry,
        policies,
        model_id,
        pinned.as_slice(),
        None,
        inputs,
    )
}

/// [`select_single`] over an explicit candidate slice: the entry for a caller
/// with a candidate rule the pools cannot express. Its one caller, the
/// multipart transcription, hands its pool whole. Nobody names such a list
/// for a classic policy. `candidates` are the caller's own part of `pool`,
/// which the policy engine alone reads: it places the request over the
/// pool's own list, held to that part.
#[inline(always)]
pub(crate) fn select_from(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pool: RoutingPool,
    candidates: &[Arc<dyn Worker>],
    inputs: PlacementInputs<'_>,
    stated: impl FnOnce() -> Stated,
) -> Option<Placed> {
    if let Some(engine) = policies.engine() {
        return from_by_engine(
            engine, registry, policies, model_id, pool, candidates, inputs, stated,
        );
    }
    select_among(registry, policies, model_id, candidates, None, inputs).map(Placed::from)
}

/// Why a pass of the single-worker selection placed nobody.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Miss {
    /// Nobody was available at the pass's read: the placement's own filter
    /// left no candidate, or a policy that filters availability itself
    /// found none.
    NobodyAvailable,
    /// The policy declined among the available candidates it was handed.
    Declined,
}

/// The selection behind both single-worker entries: one pass, and one more
/// when the pass found nobody available while a candidate the request may
/// use is available now. The pass and the veto fallback each read every
/// worker's vetoes on their own, so a veto that lifts between the two reads
/// leaves the pass with nobody and the fallback with a pool that is not all
/// vetoed: a refusal on a pool that can take the request. A pool with
/// nobody available keeps its miss, and a policy that declined among
/// available candidates is not asked again (a strict target on a worker
/// that is not available, beside workers that are). The second pass repeats
/// what a pass records on its way (the routing key's source, the manual
/// policy's branch, round robin's step), on this path alone.
fn select_among(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    candidates: &[Arc<dyn Worker>],
    whole: Option<NamedList<'_>>,
    inputs: PlacementInputs<'_>,
) -> Option<Arc<dyn Worker>> {
    // At most two rounds. The pass has this one caller, so it compiles into
    // the selection and a placement pays for no call between the two.
    let mut asked_again = false;
    loop {
        match select_pass(registry, policies, model_id, candidates, whole, inputs) {
            Ok(selected) => return Some(selected),
            Err(Miss::NobodyAvailable) if !asked_again => {}
            Err(Miss::NobodyAvailable | Miss::Declined) => return None,
        }
        let came_free = candidates.iter().any(|worker| {
            worker.is_available()
                && inputs
                    .candidate_filter
                    .is_none_or(|accepts| accepts(worker.as_ref()))
        });
        if !came_free {
            return None;
        }
        asked_again = true;
    }
}

/// One pass of the single-worker selection. `whole` names the
/// snapshot's own list when `candidates` is that list as the registry holds
/// it: the policy is told the list's name when the candidates reach it as
/// that list, and reads which workers it was handed once something narrowed
/// them (a candidate filter that refuses a worker, a policy that needs its
/// candidates filtered for it, the engines a retry already tried). A miss
/// says why ([`Miss`]).
fn select_pass(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    candidates: &[Arc<dyn Worker>],
    whole: Option<NamedList<'_>>,
    inputs: PlacementInputs<'_>,
) -> Result<Arc<dyn Worker>, Miss> {
    let policy = policies.get_policy_or_default(model_id);

    // The per-request predicate applies to every entry point, so a caller
    // that sets it can never dispatch to a worker it rejects. A predicate
    // that refuses nobody leaves the candidates as they were drawn, and the
    // pool keeps its name.
    let accepted = accepted_by(candidates, inputs.candidate_filter);
    let candidates = accepted.as_deref().unwrap_or(candidates);

    let mut info = SelectWorkerInfo {
        request_text: inputs.text,
        tokens: inputs.tokens,
        headers: inputs.headers,
        routing_key: policies.sticky_header_key(inputs.headers),
        rid_key: inputs.rid_key,
        cache_namespace: inputs.cache_namespace,
        // The cached hash ring (O(log n) lookup), for a policy that reads
        // it: no other placement pays the registry's keyed read and the
        // ring's shared count.
        hash_ring: policy_reads_hash_ring(policy.as_ref())
            .then(|| policies.hash_ring(registry, model_id))
            .flatten(),
        role: WorkerRole::Single,
        overload_lifted: false,
        named: None,
    };
    // A request its policy places by session key keeps a regular owner over
    // the overload thresholds (`LoadBalancingPolicy::routes_by_session_key`);
    // `--worker-overload-shed` keeps the veto.
    info.overload_lifted = !registry.overload_shed_enabled()
        && policy.routes_by_session_key(&info)
        && candidates
            .iter()
            .all(|w| *w.worker_type() == WorkerType::Regular);
    let eligible = |worker: &Arc<dyn Worker>| {
        worker.is_available()
            || (info.overload_lifted && overload::vetoed_by_overload_alone(worker.as_ref()))
    };

    // Most policies already apply the complete availability predicate. Give
    // them the shared snapshot directly instead of cloning every available
    // worker into a second per-request Vec. Hash policies use a weaker health
    // predicate and keep the pre-filter.
    let filtered;
    let available: &[Arc<dyn Worker>] = if policy_filters_unavailable_workers(policy.as_ref()) {
        candidates
    } else {
        filtered = candidates
            .iter()
            .filter(|worker| eligible(worker))
            .cloned()
            .collect::<Vec<_>>();
        &filtered
    };
    // The engines this request already tried step aside for its retry while
    // another available candidate remains; otherwise the pool is used whole.
    // A strict target indexes this slice (as `admission_prefilters` keeps
    // it), so it is never narrowed: the retry goes back to the targeted
    // worker.
    let keeps_target = strict_target(policy.as_ref(), inputs.headers);
    let untried;
    let available: &[Arc<dyn Worker>] = if inputs.tried.is_empty() || keeps_target {
        available
    } else {
        untried = available
            .iter()
            .filter(|worker| eligible(worker) && !was_tried(worker.as_ref(), inputs.tried))
            .cloned()
            .collect::<Vec<_>>();
        if untried.is_empty() {
            available
        } else {
            &untried
        }
    };
    if available.is_empty() {
        return overload_fallback(registry, policy.name(), model_id, candidates)
            .ok_or(Miss::NobodyAvailable);
    }
    // The name goes with the list itself, read here and nowhere above:
    // whatever narrowed the candidates handed on a list of its own, at
    // another address, and nobody names that one.
    info.named = whole.filter(|named| std::ptr::eq(&named.list[..], available));

    // The registry applies the routing-key sticky override when enabled and
    // otherwise delegates to the configured policy.
    let Some(idx) = policies.select_worker_for_model(&policy, model_id, available, &info) else {
        // A policy that filters availability itself misses on an
        // all-overloaded pool the same way the pre-filter empties. Any other
        // policy was handed available candidates and declined.
        let miss = if policy_filters_unavailable_workers(policy.as_ref()) {
            Miss::NobodyAvailable
        } else {
            Miss::Declined
        };
        return overload_fallback(registry, policy.name(), model_id, candidates).ok_or(miss);
    };
    let selected = available[idx].clone();
    if info.overload_lifted && !candidates.iter().any(|w| w.is_available()) {
        Metrics::record_worker_overload_fallback(STAGE_SELECTION);
        debug!(
            branch = BRANCH_ALL_OVERLOADED_FALLBACK,
            worker = selected.redacted_url(),
            model_id,
            "Veto fallback: a pool over the overload thresholds placed by session key"
        );
    }

    Metrics::record_worker_selection(
        metrics_labels::WORKER_REGULAR,
        selected.connection_mode().as_metric_label(),
        model_id,
        policy.name(),
    );

    Ok(selected)
}

/// The steering default for a pool whose every worker is vetoed, by the
/// overload thresholds or by the liveness tracker: its least-loaded routable
/// worker, recorded as a selection under `policy`. Under
/// `--worker-overload-shed` an overloaded worker is never taken (the caller
/// sheds an all-overloaded pool); `None` when the pool is not all-vetoed or
/// has nothing routable.
fn overload_fallback(
    registry: &WorkerRegistry,
    policy: &'static str,
    model_id: &str,
    candidates: &[Arc<dyn Worker>],
) -> Option<Arc<dyn Worker>> {
    let selected = overload::fallback_if_all_vetoed(
        candidates,
        model_id,
        STAGE_SELECTION,
        registry.overload_shed_enabled(),
    )?;
    Metrics::record_worker_selection(
        metrics_labels::WORKER_REGULAR,
        selected.connection_mode().as_metric_label(),
        model_id,
        policy,
    );
    Some(selected)
}

/// Classify a failed single-worker placement from the same pool it drew from.
pub(crate) fn single_failure(
    registry: &WorkerRegistry,
    model_id: &str,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
) -> PlacementFailure {
    let candidates = candidates(registry, model_id, pool, wire);
    failure_from(
        candidates.as_slice(),
        model_id,
        registry.overload_shed_enabled(),
    )
}

/// Classify a failed placement from the candidates it drew from. `shed` is
/// `--worker-overload-shed`: without it an all-overloaded pool was already
/// steered by the placement, so reaching here means nothing was routable.
pub(crate) fn failure_from(
    candidates: &[Arc<dyn Worker>],
    model_id: &str,
    shedding: bool,
) -> PlacementFailure {
    if candidates.is_empty() {
        return PlacementFailure::NoCandidates;
    }
    if let Some(shed) = overload::shed_if_all_overloaded(candidates, model_id, shedding) {
        return PlacementFailure::AllOverloaded(shed);
    }
    PlacementFailure::Unavailable
}

/// Seconds a client is told to wait before retrying a request every worker
/// refused. A pacing hint, not a recovery estimate: the breaker timeout and
/// the health interval are far longer, and the fleet may be back sooner.
pub(crate) const NO_AVAILABLE_WORKERS_RETRY_AFTER_SECS: u32 = 1;

/// The client-visible code of [`no_available_workers`], which the loggers
/// downstream recognise the refusal by.
pub(crate) const NO_AVAILABLE_WORKERS_CODE: &str = "no_available_workers";

/// The client-visible code of the refusal for a model whose registered
/// workers include none that processes media references, under a placement
/// that forwards a request's media to the worker: preparation's verdict over
/// the pool (`multimodal::resolve_mm_processing`) and worker selection's
/// backstop of it answer under this code.
pub(crate) const NO_MEDIA_REF_CAPABLE_WORKER_CODE: &str = "no_media_ref_capable_worker";

/// The 503 for a pool whose every worker is unavailable (unhealthy or its
/// circuit breaker open). Terminal for the retry layer: the verdict clears at
/// the health interval or the breaker timeout, never inside a backoff window,
/// so retrying it in-process only adds the backoffs to every client's latency
/// while the pool is down. The client retries on its own schedule, paced by
/// `Retry-After` like the gateway's other refusals (admission sheds, the
/// overload shed, tenant limits): a client that honours the header does not
/// hammer a fleet that is down and gets the same bounded hint on every path.
pub(crate) fn no_available_workers(message: impl Into<String>) -> Response {
    let message = message.into();
    let reason = message.clone();
    no_available_workers_because(message, reason)
}

/// [`no_available_workers`] whose line names a reason other than the
/// client's message: the message stays the one the client has always read
/// under the code, the line gets the verdict the router has (the policy
/// that declined a role, the role at its admission limit), so the reason on
/// the line never contradicts the pool state next to it.
pub(crate) fn no_available_workers_because(
    message: impl Into<String>,
    reason: impl Into<String>,
) -> Response {
    let mut response = error::service_unavailable(NO_AVAILABLE_WORKERS_CODE, message.into());
    response
        .extensions_mut()
        .insert(RefusalReason(reason.into()));
    mark_non_retryable(&mut response);
    response.headers_mut().insert(
        RETRY_AFTER,
        HeaderValue::from(NO_AVAILABLE_WORKERS_RETRY_AFTER_SECS),
    );
    response
}

/// The 503 for a model whose registered workers include none that processes
/// media references, under worker-side media placement. A capability
/// verdict, not a pool state: terminal for the retry layer (what a worker
/// advertises does not change within a backoff window) and without a
/// `Retry-After` (a client's pacing fixes nothing here; the deployment's
/// configuration does). The message is the caller's; the line's reason is
/// the verdict.
pub(crate) fn no_media_ref_capable_worker(message: impl Into<String>) -> Response {
    let mut response = error::service_unavailable(NO_MEDIA_REF_CAPABLE_WORKER_CODE, message.into());
    response.extensions_mut().insert(RefusalReason(
        "no worker of the model processes media references".to_string(),
    ));
    mark_non_retryable(&mut response);
    response
}

/// The reason a refusal's line names, when the router gave one (a
/// `no_available_workers` 503; a 404 names none).
pub(crate) fn refusal_reason(response: &Response) -> Option<&str> {
    response
        .extensions()
        .get::<RefusalReason>()
        .map(|reason| reason.0.as_str())
}

/// The 503 for a fleet whose prefill and decode pools hold workers and no
/// prefill of them pairs with a decode: `mismatches` are the components the
/// pools disagree on, `prefill` and `decode` each pool's pairing keys. The
/// verdict reads membership alone.
pub(crate) fn no_compatible_pair(
    model: &str,
    mismatches: &[String],
    prefill: &[String],
    decode: &[String],
) -> Response {
    error::service_unavailable(
        "no_compatible_pd_pair",
        format!(
            "No prefill/decode pair for model '{model}' shares a KV transfer protocol (mismatch \
             on {mismatches:?}; prefill: {prefill:?}, decode: {decode:?})"
        ),
    )
}

/// The workers of `pool` a placement may use: those on `wire` when a retry
/// pins one, narrowed to the ones the request's candidate filter admits. A
/// failed pool's veto fallback and its verdict are judged over these, so a
/// filter that admits only vetoed workers reads as a vetoed pool (steered,
/// or shed under `--worker-overload-shed`) and not as an unavailable one
/// because a worker the request may not use is idle. A narrowing that
/// admits nothing is left out: a wire no worker of the pool is on keeps the
/// pool's own verdict as before, judged over every worker and without the
/// filter (a filter narrowed over workers the retry cannot use would turn
/// that verdict into a shed over another wire's overloaded workers), and a
/// filter that admits none of the workers on the wire is the caller's to
/// name.
fn admitted_pool(
    pool: &[Arc<dyn Worker>],
    wire: Option<WireConstraint>,
    inputs: &PlacementInputs<'_>,
) -> Vec<Arc<dyn Worker>> {
    if wire.is_none() && inputs.candidate_filter.is_none() {
        return pool.to_vec();
    }
    let narrowed = |workers: &[Arc<dyn Worker>], keep: &dyn Fn(&Arc<dyn Worker>) -> bool| {
        let kept: Vec<Arc<dyn Worker>> = workers.iter().filter(|w| keep(w)).cloned().collect();
        (!kept.is_empty()).then_some(kept)
    };
    let Some(on_wire) = narrowed(pool, &|w| WireConstraint::admits(wire, w.as_ref())) else {
        return pool.to_vec();
    };
    narrowed(&on_wire, &|w| {
        inputs
            .candidate_filter
            .is_none_or(|accepts| accepts(w.as_ref()))
    })
    .unwrap_or(on_wire)
}

/// The client-visible code of [`error::model_not_found`], which the refusal
/// line recognises the 404 by.
pub(crate) const MODEL_NOT_FOUND_CODE: &str = "model_not_found";

/// The reason a `no_available_workers` refusal names (the role that is down,
/// the policy that declined), carried on the response for its log line.
#[derive(Clone, Debug)]
struct RefusalReason(String);

/// Lines per second the gateway writes for each refusal it answers on its
/// own: a model whose workers are all unavailable (`no_available_workers`),
/// a model nobody serves (`model_not_found`) and a model none of whose
/// workers processes the media references forwarded to it
/// (`no_media_ref_capable_worker`). One line per refusal, with
/// the request id, the model and the pool's state, shows the condition and
/// whom it hits; past this rate the lines are counted and the next one
/// carries the count, while `smg_router_request_errors_total` keeps the
/// volume. A client retrying at once drew four thousand refusals a second,
/// up to three ERROR lines each, before. The allowance is per model, so one
/// storming model never silences another's first line.
pub(crate) const REFUSAL_LINES_PER_SECOND: u32 = 10;

/// Models a refusal throttle tracks at once; past this the model seen least
/// recently is forgotten (a model that is down keeps refusing, so it stays).
pub(crate) const REFUSAL_LOG_MODELS: usize = 64;

static NO_AVAILABLE_WORKERS_LOG: KeyedLogThrottle = KeyedLogThrottle::new(
    REFUSAL_LINES_PER_SECOND,
    Duration::from_secs(1),
    REFUSAL_LOG_MODELS,
);

static MODEL_NOT_FOUND_LOG: KeyedLogThrottle = KeyedLogThrottle::new(
    REFUSAL_LINES_PER_SECOND,
    Duration::from_secs(1),
    REFUSAL_LOG_MODELS,
);

static NO_MEDIA_REF_CAPABLE_WORKER_LOG: KeyedLogThrottle = KeyedLogThrottle::new(
    REFUSAL_LINES_PER_SECOND,
    Duration::from_secs(1),
    REFUSAL_LOG_MODELS,
);

/// The router answering a refusal, as its line names it.
#[derive(Clone, Copy)]
pub(crate) struct RefusalSite<'a> {
    /// The refusing function, e.g. `WorkerSelectionStage::execute`.
    pub function: &'static str,
    /// The router's mode: `Regular`, `PrefillDecode` or `EncodePrefillDecode`.
    pub mode: &'static str,
    /// The pools the router draws from for the model, named per role.
    pub pools: &'a [(&'static str, RoutingPool)],
}

/// A refusal the gateway answers on its own, with the line it gets and the
/// cap the line is under.
#[derive(Clone, Copy)]
enum Refusal {
    NoAvailableWorkers,
    ModelNotFound,
    /// The model's workers process no media references, under a placement
    /// that forwards them ([`no_media_ref_capable_worker`]): a configuration
    /// verdict that holds until a capable worker registers, so it is keyed
    /// and capped like the pool's 503.
    NoMediaRefCapableWorker,
}

impl Refusal {
    fn of(response: &Response) -> Option<Self> {
        match error::extract_error_code_from_response(response) {
            NO_AVAILABLE_WORKERS_CODE => Some(Self::NoAvailableWorkers),
            MODEL_NOT_FOUND_CODE => Some(Self::ModelNotFound),
            NO_MEDIA_REF_CAPABLE_WORKER_CODE => Some(Self::NoMediaRefCapableWorker),
            _ => None,
        }
    }

    fn throttle(self) -> &'static KeyedLogThrottle {
        match self {
            Self::NoAvailableWorkers => &NO_AVAILABLE_WORKERS_LOG,
            Self::ModelNotFound => &MODEL_NOT_FOUND_LOG,
            Self::NoMediaRefCapableWorker => &NO_MEDIA_REF_CAPABLE_WORKER_LOG,
        }
    }

    fn message(self) -> &'static str {
        match self {
            Self::NoAvailableWorkers => "No available workers for model",
            Self::ModelNotFound => "No worker serves model",
            Self::NoMediaRefCapableWorker => "No worker processes media references for model",
        }
    }

    /// The key the refusal's line is throttled under. A 503 for a model with
    /// workers registered on some role is keyed by the model: those keys are
    /// bounded by the registry, and one storming model never silences
    /// another's first line. A 404's "model" is whatever name the client
    /// sent, so every unknown name shares one allowance: a client varying
    /// the name gets no fresh lines per name, and the one line still names
    /// one of them while `suppressed` carries the rest. A 503 for a name
    /// with no workers on any role (the HTTP PD router answers a name nobody
    /// serves with its role's 503, not a 404) is an unknown name too and
    /// shares one allowance with every other such name, for the same reason.
    fn key(self, model_id: &str, registered: bool) -> &str {
        match self {
            Self::NoAvailableWorkers | Self::NoMediaRefCapableWorker if registered => model_id,
            Self::NoAvailableWorkers | Self::NoMediaRefCapableWorker | Self::ModelNotFound => "",
        }
    }
}

/// The log line of a refusal the gateway answers on its own, and its marker.
///
/// A `no_available_workers` 503, a `no_media_ref_capable_worker` 503 or a
/// `model_not_found` 404 gets one WARN with the request id, the model and
/// the state of the pools `site` draws from, and the reason the refusal
/// names when it has one (the role that is down, the policy that declined,
/// the media references no worker processes), capped at
/// [`REFUSAL_LINES_PER_SECOND`] per model with workers for the 503s and for
/// all unknown names together
/// (the 404, and a 503 for a name with no workers at all; see
/// [`Refusal::key`]); a line past the cap is counted and the next
/// admitted line of its key carries the count as `suppressed`. The response
/// is marked [`ReportedRefusal`], so the
/// pipeline's stage line and the access log's response line drop to DEBUG
/// for it: a refusal is the gateway's normal answer to "no worker for this
/// request", not a fault of its own, and up to three ERROR lines per refusal
/// made a client's retry loop a log storm. The error counter
/// (`smg_router_request_errors_total`) keeps the volume. A refusal already
/// marked, and every other response, passes through unchanged.
pub(crate) fn report_refusal(
    registry: &WorkerRegistry,
    site: RefusalSite<'_>,
    request_id: Option<&str>,
    model_id: &str,
    refusal: Response,
) -> Response {
    match Refusal::of(&refusal) {
        Some(kind) if refusal.extensions().get::<ReportedRefusal>().is_none() => {
            report_refusal_under(
                kind.throttle(),
                registry,
                site,
                request_id,
                model_id,
                refusal,
            )
        }
        _ => refusal,
    }
}

/// [`report_refusal`] under an explicit throttle (a test's own, or the
/// static of the refusal's kind), keyed as [`Refusal::key`] says.
pub(crate) fn report_refusal_under(
    throttle: &KeyedLogThrottle,
    registry: &WorkerRegistry,
    site: RefusalSite<'_>,
    request_id: Option<&str>,
    model_id: &str,
    mut refusal: Response,
) -> Response {
    let Some(kind) = Refusal::of(&refusal) else {
        return refusal;
    };
    let message = kind.message();
    refusal.extensions_mut().insert(ReportedRefusal);
    // Whether some role has workers for the model (see `Refusal::key`): one
    // pool lookup per role, no walk over the workers; the 404 needs none.
    let registered = matches!(
        kind,
        Refusal::NoAvailableWorkers | Refusal::NoMediaRefCapableWorker
    ) && site
        .pools
        .iter()
        .any(|(_, pool)| !registry.get_routing_pool(model_id, *pool).is_empty());
    if let Some(admitted) = throttle.admit(kind.key(model_id, registered)) {
        let pool = PoolState::of(registry, model_id, site.pools);
        let reason = refusal_reason(&refusal);
        // The request's span carries the id too; the field is for a line
        // written outside it (a retry's re-selection) and is omitted when
        // the router has no id of its own to give.
        // The model is the caller's text: as a string field it is written
        // quoted, so a line break in it never starts a line of its own.
        warn!(
            function = site.function,
            request_id = request_id,
            mode = %site.mode,
            model_id = model_id,
            reason = reason,
            workers = pool.workers(),
            available = pool.available(),
            pool = %pool,
            suppressed = admitted.suppressed,
            "{message}"
        );
    }
    refusal
}

/// What a refusal line says about the pools it drew from: per role, the
/// workers registered for the model, those healthy, and those routable
/// (healthy, circuit closed, not vetoed by load or liveness).
struct PoolState {
    roles: Vec<RoleState>,
}

struct RoleState {
    role: &'static str,
    workers: usize,
    healthy: usize,
    available: usize,
}

impl PoolState {
    fn of(
        registry: &WorkerRegistry,
        model_id: &str,
        pools: &[(&'static str, RoutingPool)],
    ) -> Self {
        PoolState {
            roles: pools
                .iter()
                .map(|(role, pool)| {
                    let workers = registry.get_routing_pool(model_id, *pool);
                    RoleState {
                        role,
                        workers: workers.len(),
                        healthy: workers.iter().filter(|w| w.is_healthy()).count(),
                        available: workers.iter().filter(|w| w.is_available()).count(),
                    }
                })
                .collect(),
        }
    }

    fn workers(&self) -> usize {
        self.roles.iter().map(|role| role.workers).sum()
    }

    fn available(&self) -> usize {
        self.roles.iter().map(|role| role.available).sum()
    }
}

impl fmt::Display for PoolState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, role) in self.roles.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(
                f,
                "{} {}/{} available ({} healthy)",
                role.role, role.available, role.workers, role.healthy
            )?;
        }
        Ok(())
    }
}

/// The steering default for a disaggregated placement whose failed pool is
/// vetoed on every worker the request may use (`failed_pool`, see
/// [`admitted_pool`]), by the overload thresholds or by the liveness
/// tracker, where the pool's policy does not place over a lifted overload
/// veto (see [`select_pair`]): the least-loaded routable prefill that has a
/// routable partner, paired with its least-loaded routable partner. Pairs
/// vetoed by overload alone come first; a liveness veto is crossed only when
/// nothing else pairs, and never onto an overloaded worker under
/// `--worker-overload-shed`.
/// `None` when the failed pool is not all-vetoed (an unhealthy pool keeps its
/// unavailable answer), when it is all-overloaded under shedding (the caller
/// sheds), or when nothing pairs; a prefill admission gate still has the last
/// word.
fn relaxed_pair(
    registry: &WorkerRegistry,
    model_id: &str,
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    prefill_capacity: Option<&PrefillSelectionContext<'_>>,
    inputs: &PlacementInputs<'_>,
    failed_pool: &[Arc<dyn Worker>],
) -> Option<Result<Pair, Box<PairFailure>>> {
    let shedding = registry.overload_shed_enabled();
    if !overload::all_vetoed(failed_pool) || (shedding && overload::all_overloaded(failed_pool)) {
        return None;
    }
    let eligible = |w: &Arc<dyn Worker>| {
        w.is_healthy()
            && w.circuit_breaker_can_execute()
            && WireConstraint::admits(wire, w.as_ref())
            && inputs
                .candidate_filter
                .is_none_or(|accepts| accepts(w.as_ref()))
    };
    // Equal loads draw uniformly (partner and pair alike), so an equally
    // loaded pool does not send every fallback to its first pair.
    let mut rng = rand::rng();
    let mut search = |routable: &dyn Fn(&Arc<dyn Worker>) -> bool| {
        let mut best: Option<(Arc<dyn Worker>, Arc<dyn Worker>)> = None;
        let mut tied = 0u32;
        for (i, prefill) in pairs.prefill.iter().enumerate() {
            if !routable(prefill) {
                continue;
            }
            let Some(decode) = overload::least_loaded_uniform(
                pairs.partners[i].iter().filter(|d| routable(d)),
                &mut rng,
            ) else {
                continue;
            };
            let loads = (prefill.load(), decode.load());
            match best.as_ref().map(|(p, d)| (p.load(), d.load())) {
                Some(best_loads) if loads > best_loads => {}
                Some(best_loads) if loads == best_loads => {
                    tied += 1;
                    if rng.random_range(0..=tied) == 0 {
                        best = Some((Arc::clone(prefill), Arc::clone(decode)));
                    }
                }
                _ => {
                    best = Some((Arc::clone(prefill), Arc::clone(decode)));
                    tied = 0;
                }
            }
        }
        best
    };
    let overloaded_first = (!shedding)
        .then(|| search(&|w| eligible(w) && w.stall_reason().is_none()))
        .flatten();
    let ((prefill, decode), branch) = match overloaded_first {
        Some(pair) => (pair, BRANCH_ALL_OVERLOADED_FALLBACK),
        None => (
            search(&|w| eligible(w) && !(shedding && w.is_overloaded()))?,
            BRANCH_ALL_STALLED_FALLBACK,
        ),
    };
    if prefill_capacity.is_some_and(|capacity| !capacity.has_capacity(&prefill)) {
        return Some(Err(Box::new(PairFailure {
            role: WorkerRole::Prefill,
            verdict: PlacementFailure::PrefillAtCapacity,
        })));
    }
    if branch == BRANCH_ALL_OVERLOADED_FALLBACK {
        Metrics::record_worker_overload_fallback(STAGE_SELECTION);
    } else {
        Metrics::record_worker_liveness_fallback(STAGE_SELECTION);
    }
    debug!(
        branch,
        prefill = %prefill.redacted_url(),
        decode = %decode.redacted_url(),
        model_id,
        "Veto fallback"
    );
    let runtime = prefill.metadata().spec.runtime_type;
    Some(Ok(Pair {
        prefill,
        decode,
        runtime,
    }))
}

/// Whether `worker`'s engine (its base URL, which the ranks of a DP-aware
/// worker share) is one a failed attempt of the request already dispatched
/// to ([`PlacementInputs::tried`]).
pub(super) fn was_tried(worker: &dyn Worker, tried: &[String]) -> bool {
    tried.iter().any(|engine| engine == worker.base_url())
}

/// Pick a prefill/decode pair for `model_id` over `pairs`, one worker per
/// role, each as the placement answers a worker ([`PlacedPair`]). A miss
/// names the role and carries the verdict.
///
/// A thin front, always inlined where a route calls it, as the
/// single-request ones are ([`select_single`]). It reads the registry's
/// engine once. Where the gateway started none it calls the classic
/// placement ([`classic_pair`]), which stands as its own function as it did
/// before the engine and answers the two workers' handles alone. Where it
/// started one the engine places the pair, out of line
/// ([`pair_by_engine`]).
///
/// `stated` states what the engine alone reads of the request
/// ([`Stated`]): the call is handed on unmade to the engine's arm, which
/// makes it. The answer of the pair's decode keeps what it stated, and the
/// decode asked again states it again ([`decode_again`]).
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the pair placement's own arguments, and what the request states for the engine"
)]
pub(crate) fn select_pair(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    prefill_capacity: Option<&PrefillSelectionContext<'_>>,
    inputs: PlacementInputs<'_>,
    stated: impl FnOnce() -> Stated,
) -> Result<PlacedPair, Box<PairFailure>> {
    if let Some(engine) = policies.engine() {
        return pair_by_engine(
            engine,
            registry,
            policies,
            model_id,
            pairs,
            wire,
            prefill_capacity,
            inputs,
            stated,
        );
    }
    classic_pair(
        registry,
        policies,
        model_id,
        pairs,
        wire,
        prefill_capacity,
        inputs,
    )
    .map(PlacedPair::from)
}

/// Pick a prefill/decode pair for `model_id`, one worker per role, each under
/// its own policy and sticky namespace.
///
/// `pairs` is the snapshot's compatibility index (see [`PdPairIndex`]):
/// which decodes each prefill may hand off to was decided when membership
/// last changed, so this path compares no descriptors. Every pool is judged
/// live for availability, and a prefill counts only while one of its
/// partners is available, so the policy's pick can always be paired; `wire`
/// pins both pools to the retained plan's runtime and transport on a retry.
/// Where the index's wire is runtime-specific (`PdWire::runtime_specific`),
/// every prefill open with a partner of its own known runtime is a
/// candidate, and its decode is drawn from its partners of that runtime; a
/// pair with a worker whose runtime is unknown is placed only while no such
/// prefill is open. A miss names the role and carries the verdict judged
/// from that role's candidates the request may use. A prefill pool whose
/// policy balances load against cache affinity itself is never steered by
/// the overload veto, and is placed by that policy even when every worker
/// in it is over the thresholds. Any other pool whose every worker the
/// request may use is over them is steered by load as before: a prefill
/// pool by the least-loaded default ([`relaxed_pair`]), a decode pool to
/// the least-loaded partner of the prefill its own policy picked. Only the
/// placement by policy puts a pair on one known runtime before a pair with
/// an unknown one: the default pairs by load over the index's lists, as before.
///
/// A retry's placement by policy steps aside from the engines the request
/// already tried ([`PlacementInputs::tried`]), a side at a time: the open
/// prefills the admission left, then the pick's open partners the rooms
/// left, each kept to the untried ones while one is left and whole
/// otherwise. It is a preference and refuses nobody: a worker that is the
/// only one open on its side, or the only one with a place for the request,
/// takes the retry. An explicit target under consistent hashing holds its
/// side to the targeted worker ([`strict_target`]), and the default pairs by
/// load over the whole pools, tried or not.
///
/// The placement is one pass, and one more when the pass found nobody open
/// on a side while a worker of that side the request may use is available
/// now. The pass and the default each read the vetoes on their own, so a
/// veto that lifts between the two reads leaves the pass with nobody open
/// and the default with a pool that is not all vetoed: a refusal on a pool
/// that can take the request. A side with no admitted worker available keeps
/// its miss. On decode, this check is coarser than the pass: it scans the
/// admitted decode pool rather than only partners of eligible prefills, so an
/// available non-partner can cause the one bounded extra pass. A miss of any
/// other kind (a policy that declined, a prefill at its limit) is not asked
/// again, and neither is an overload shed: building that verdict already
/// counted the request in `smg_worker_overload_shed_total`.
///
/// Never inlined into its front ([`select_pair`]): the function a classic
/// pair placement runs is this one, compiled as it was before the engine,
/// whatever stands beside its call.
#[inline(never)]
fn classic_pair(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    prefill_capacity: Option<&PrefillSelectionContext<'_>>,
    inputs: PlacementInputs<'_>,
) -> Result<Pair, Box<PairFailure>> {
    // At most two rounds. The pass has this one caller, so it compiles into
    // the placement and a placement pays for no call between the two.
    let mut asked_again = false;
    loop {
        let failure = match pair_pass(
            registry,
            policies,
            model_id,
            pairs,
            wire,
            prefill_capacity,
            inputs,
        ) {
            Ok(pair) => return Ok(pair),
            Err(PairMiss::NobodyOpen(failure))
                if !asked_again
                    && !matches!(&failure.verdict, PlacementFailure::AllOverloaded(_)) =>
            {
                failure
            }
            Err(PairMiss::NobodyOpen(failure) | PairMiss::Final(failure)) => return Err(failure),
        };
        let failed_pool: &[Arc<dyn Worker>] = if matches!(failure.role, WorkerRole::Prefill) {
            &pairs.prefill
        } else {
            &pairs.decode_pool
        };
        let came_free = failed_pool.iter().any(|w| {
            w.is_available()
                && WireConstraint::admits(wire, w.as_ref())
                && inputs
                    .candidate_filter
                    .is_none_or(|accepts| accepts(w.as_ref()))
        });
        if !came_free {
            return Err(failure);
        }
        asked_again = true;
    }
}

/// What membership alone refuses a pair by, as the classic pass names it
/// ahead of its pairing ([`pair_pass`]): a role with no worker at all names
/// itself, and two pools no prefill of which shares a KV transfer protocol
/// with a decode name both. Stated here again for the engine's arm, so the
/// classic pass stands as it did.
fn unpairable(
    policies: &PolicyRegistry,
    model_id: &str,
    pairs: &PdPairIndex,
) -> Option<Box<PairFailure>> {
    let fail =
        |role: WorkerRole, verdict: PlacementFailure| Box::new(PairFailure { role, verdict });
    if pairs.prefill_pool.is_empty() {
        debug!("No prefill workers");
        return Some(fail(WorkerRole::Prefill, PlacementFailure::NoCandidates));
    }
    if pairs.decode_pool.is_empty() {
        debug!("No decode workers");
        return Some(fail(WorkerRole::Decode, PlacementFailure::NoCandidates));
    }
    let refusal = pairs.refusal.as_ref()?;
    warn!(
        model_id,
        mode = policies.pd_pairing_mode().as_str(),
        mismatches = ?refusal.mismatches,
        prefill = ?refusal.prefill,
        decode = ?refusal.decode,
        "No prefill/decode pair shares a KV transfer protocol"
    );
    Some(fail(
        WorkerRole::Decode,
        PlacementFailure::NoCompatiblePair {
            prefill: refusal.prefill.clone(),
            decode: refusal.decode.clone(),
            mismatches: refusal.mismatches.clone(),
        },
    ))
}

/// A pair's admission as the policy engine reads it of this gateway, a
/// worker at a time and live: a prefill's capacity by the request's prefill
/// admission, where its route set one, and a decode's room by its gate
/// ([`pd_admission::has_free_room`]).
struct PairGates<'a> {
    prefills: &'a [Arc<dyn Worker>],
    decodes: &'a [Arc<dyn Worker>],
    capacity: Option<&'a PrefillSelectionContext<'a>>,
}

impl fmt::Debug for PairGates<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PairGates")
            .field("prefills", &self.prefills.len())
            .field("decodes", &self.decodes.len())
            .field("capacity", &self.capacity.is_some())
            .finish()
    }
}

impl PairAdmission for PairGates<'_> {
    fn has_capacity(&self, prefill: usize) -> bool {
        let Some(capacity) = self.capacity else {
            return true;
        };
        self.prefills
            .get(prefill)
            .is_some_and(|worker| capacity.has_capacity(worker))
    }

    fn has_room(&self, decode: usize) -> bool {
        self.decodes
            .get(decode)
            .is_some_and(|worker| pd_admission::has_free_room(worker.as_ref()))
    }
}

/// The policy engine's request for a pair over the two pools of `pairs`,
/// handed to `ask`: each pool's list with its cell
/// ([`PdPairIndex::lent`]), the index's pair classes
/// ([`PdPairIndex::classes`]) and this gateway's two admission reads
/// ([`PairGates`]), on the hash ring the registry answers for the model now
/// ([`ModelRing::read`]). One function for a pair's two decisions: the
/// first ([`pair_by_engine`]), and the decode asked again behind its
/// prefill ([`decode_again`]), which reads no prefill's capacity.
///
/// What `inputs` and `stated` state of the request is handed as the engine
/// reads it ([`place_request`]), the route's own rule among it, which the
/// engine asks of the workers it picked. Four things are the pair's own:
///
/// - A retry pinned to its retained plan's wire is held to the places of
///   each pool on that wire. The lists stay the pools' own.
/// - A pair asked again behind a call that answered no worker, where the
///   route's rule takes a pair that can be sent to, is held to the places
///   of the workers `held_to` takes too, that rule ([`pair_by_engine`]). A
///   first ask states none, and neither does a decode asked again: its
///   route's rule claims the decode it is asked of.
/// - The retry number is stated as half the engines the request tried,
///   rounded up: a pair's failed attempt names both its engines.
/// - The request states that its decode may be asked again behind its
///   prefill's answer. A route that sends both sides at once sends the
///   decode the first decision named, which the engine books at that send.
#[expect(
    clippy::too_many_arguments,
    reason = "the pair placement's own arguments, and the engine it asks"
)]
fn with_pair_request<R>(
    engine: &'static PolicyEngine,
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    held_to: Option<CandidateFilter<'_>>,
    prefill_capacity: Option<&PrefillSelectionContext<'_>>,
    inputs: PlacementInputs<'_>,
    stated: Stated,
    ask: impl FnOnce(&PlaceRequest<'_>) -> R,
) -> R {
    let [prefill, decode] = pairs.lent();
    let held = |pool: &[Arc<dyn Worker>]| -> Option<Vec<u32>> {
        (wire.is_some() || held_to.is_some()).then(|| {
            (0u32..)
                .zip(pool)
                .filter(|(_, worker)| {
                    WireConstraint::admits(wire, worker.as_ref())
                        && held_to.is_none_or(|takes| takes(worker.as_ref()))
                })
                .map(|(place, _)| place)
                .collect()
        })
    };
    let (prefill_places, decode_places) = (held(prefill.workers), held(decode.workers));
    let gates = PairGates {
        prefills: prefill.workers,
        decodes: decode.workers,
        capacity: prefill_capacity,
    };
    let lists = PairLists {
        prefill,
        decode,
        classes: pairs.classes(),
        admission: &gates,
        prefill_places: prefill_places.as_deref(),
        decode_places: decode_places.as_deref(),
        // The gateway's standing vetoes are on its worker objects, which
        // the engine reads: none is stated beside a list.
        prefill_stalled: &[],
        prefill_held: &[],
        decode_stalled: &[],
        decode_held: &[],
    };
    // A failed attempt lists both its engines, whichever side failed.
    let (mut tried, mut decodes_tried) = (TriedIds::new(), TriedIds::new());
    tried_ids(prefill.workers, inputs.tried, &mut tried);
    tried_ids(decode.workers, inputs.tried, &mut decodes_tried);
    tried.append(&mut decodes_tried);
    let ring = ModelRing::read(engine, policies, registry, model_id);
    let mut request = place_request(model_id, WorkerLists::Pair(lists), &inputs, &tried, stated);
    request.ring = ring.lent();
    request.attempt = u8::try_from(inputs.tried.len().div_ceil(2)).unwrap_or(u8::MAX);
    request.decode_again = true;
    ask(&request)
}

/// [`select_pair`] where the gateway started the policy engine, out of
/// line: the engine places the pair with one call, over the request
/// [`with_pair_request`] states, and with one more, held to the route's
/// rule, where the first answers no worker and the rule takes a pair that
/// can be sent to. Its document picks both sides: no classic policy runs,
/// and the load default of the classic pass pairs nobody here.
///
/// Prefills all at the request's admission limit answer
/// [`PlacementFailure::PrefillAtCapacity`], on which the route waits for a
/// slot and asks again. Any other miss is judged from the side that has
/// nobody the request may use and that is available, the prefills first, as
/// a classic pass judges its own ([`failure_from`]).
///
/// Records the selection metric of each side under that worker's transport
/// label and the label of the profile that picked the side, and under the
/// name the settings give the side's policy where the engine names none
/// ([`PolicyRegistry::side_policy_name`]): no policy object is asked.
///
/// The route's call that states the request ([`Stated`]) is made here,
/// for a pair that can be placed.
///
/// A traced request's decision record takes the rows a classic pair's two
/// selections write, the prefill's and then the decode's: each pool as it
/// stands ahead of the engine's first call, and the side's worker under the
/// row's name ([`row_name`]). Under a document that picks the decode behind
/// its prefill the decode's row is not written: that decode may be decided
/// again behind the prefill's answer ([`decode_again`]), after the route
/// wrote the record, so the record holds the prefill's row alone. A pair
/// the engine did not place leaves one row, with no worker: the side
/// judged to have nobody. Prefills all at the admission limit leave none,
/// as a classic pass leaves none where they leave its candidates ahead of
/// its policy: the route asks again for every slot it waits for. An
/// untraced pair pays one lookup for both.
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "the pair placement's own arguments, and the engine it asks"
)]
fn pair_by_engine(
    engine: &'static PolicyEngine,
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    prefill_capacity: Option<&PrefillSelectionContext<'_>>,
    inputs: PlacementInputs<'_>,
    stated: impl FnOnce() -> Stated,
) -> Result<PlacedPair, Box<PairFailure>> {
    if let Some(failure) = unpairable(policies, model_id, pairs) {
        return Err(failure);
    }
    let stated = stated();
    let fail =
        |role: WorkerRole, verdict: PlacementFailure| Box::new(PairFailure { role, verdict });
    let admitted = |worker: &Arc<dyn Worker>| {
        WireConstraint::admits(wire, worker.as_ref())
            && inputs
                .candidate_filter
                .is_none_or(|accepts| accepts(worker.as_ref()))
    };
    let place = |held_to: Option<CandidateFilter<'_>>| {
        with_pair_request(
            engine,
            registry,
            policies,
            model_id,
            pairs,
            wire,
            held_to,
            prefill_capacity,
            inputs,
            stated,
            |request| {
                engine.place(request).map(|placement| {
                    let labels = (
                        placement.profile,
                        placement.decode_profile.unwrap_or(placement.profile),
                    );
                    (labels, PlacedPair::by_engine(placement, stated))
                })
            },
        )
    };
    // The engine asks a pair's rule of its picks alone, a few a decision,
    // which miss the workers a rule takes of a long pool: the pair is then
    // decided once more among the workers the rule takes now, where it
    // takes a prefill and a decode that prefill pairs with, each ready and
    // let through by its circuit breaker: the overload veto and the liveness
    // veto are the engine's to lift. Prefills of one descriptor share one
    // partner list: the rule is asked of a list once.
    let takes_a_pair = || {
        let takes = |worker: &Arc<dyn Worker>| {
            let state = worker.routing_state();
            state.healthy && state.can_execute && admitted(worker)
        };
        let mut refused: Vec<&Arc<[Arc<dyn Worker>]>> = Vec::new();
        for (prefill, partners) in pairs.prefill.iter().zip(&pairs.partners) {
            if refused.iter().any(|&list| Arc::ptr_eq(list, partners)) || !takes(prefill) {
                continue;
            }
            if partners.iter().any(takes) {
                return true;
            }
            refused.push(partners);
        }
        false
    };
    let side_row = |side: RowSide, label| row_name(policies, Some(side), label);
    let prefill_seen = cache_trace::observe(&pairs.prefill_pool);
    let decode_seen = prefill_seen
        .as_ref()
        .and_then(|_| cache_trace::observe(&pairs.decode_pool));
    let placed = match place(None) {
        Err(Unplaced::NoWorkers) if inputs.candidate_filter.is_some() && takes_a_pair() => {
            place(inputs.candidate_filter)
        }
        placed => placed,
    };
    let (labels, pair) = match placed {
        Ok(placed) => placed,
        Err(Unplaced::PrefillsFull) => {
            return Err(fail(
                WorkerRole::Prefill,
                PlacementFailure::PrefillAtCapacity,
            ));
        }
        Err(unplaced) => {
            report_unplaced(model_id, &unplaced, "The policy engine placed no pair");
            let open =
                |pool: &[Arc<dyn Worker>]| pool.iter().any(|w| admitted(w) && w.is_available());
            let (role, pool) = if open(&pairs.prefill) {
                engine_row(decode_seen, || side_row(DECODE_ROW, ""), None);
                (WorkerRole::Decode, &pairs.decode_pool)
            } else {
                engine_row(prefill_seen, || side_row(PREFILL_ROW, ""), None);
                (WorkerRole::Prefill, &pairs.prefill)
            };
            let judged = admitted_pool(pool, wire, &inputs);
            let shedding = registry.overload_shed_enabled();
            return Err(fail(role, failure_from(&judged, model_id, shedding)));
        }
    };
    let Some(pair) = pair else {
        return Err(fail(WorkerRole::Decode, PlacementFailure::Unavailable));
    };
    let configured =
        |label: &'static str, policy: &'static str| if label.is_empty() { policy } else { label };
    for (role, worker, side, label, seen) in [
        (
            metrics_labels::WORKER_PREFILL,
            &pair.prefill,
            PREFILL_ROW,
            labels.0,
            prefill_seen,
        ),
        (
            metrics_labels::WORKER_DECODE,
            &pair.decode,
            DECODE_ROW,
            labels.1,
            decode_seen,
        ),
    ] {
        let transport = worker.connection_mode().as_metric_label();
        let counted_under = configured(label, policies.side_policy_name(side.0));
        Metrics::record_worker_selection(role, transport, model_id, counted_under);
        // A decode the document picks behind its prefill may be decided
        // again after the record is written: no row names the one decided
        // first.
        let later = || engine.spec().decode_pick() == DecodePick::AfterPrefill;
        let seen = seen.filter(|_| side != DECODE_ROW || !later());
        engine_row(seen, || side_row(side, label), Some(worker.worker()));
    }
    debug!(
        prefill = %pair.prefill.redacted_url(),
        decode = %pair.decode.redacted_url(),
        "The policy engine placed a PD pair"
    );
    Ok(pair)
}

/// A pair's decode asked again behind its prefill's answer: the placement
/// function a pair route calls once its prefill has answered and before it
/// sends the decode, where it sends the two sides one behind the other.
///
/// `decode` is the pair's decode as [`select_pair`] answered it, and `ahead`
/// its load guard, made ahead of its send ([`Placed::ahead`]). `pairs`,
/// `wire` and `inputs` state the request again, over the pair index as it
/// stands now ([`PairsAgain`]): the route's own rule in `inputs` is asked of
/// the decode the engine picks. What the request stated for the engine
/// alone ([`Stated`]) is stated again as the decode's answer kept it, so a
/// route states it once. The function answers the guard to keep, and
/// the decode the request goes to now where that is another worker than
/// `decode`'s:
///
/// - `(None, ahead)` wherever the first decision's decode stands: under the
///   classic policies, under a document that picked the decode with its
///   prefill, where the second decision names the same worker, and where it
///   takes no worker, whatever the reason. The route sends the decode it
///   has.
/// - That worker, with the decode's side of the engine's placement, and a
///   guard that counts it under the key `ahead` counted under. `ahead` is
///   let go here: the first decode is counted no more, and its answer
///   keeps nothing of the engine.
///
/// The route says [`Placed::sent`] with the answered guard, on the decode
/// the request goes to, at that decode's send.
///
/// A route makes this one call for every decode it holds ahead of a
/// prefill's answer. Whether it holds one it asks the decode's answer where
/// it decides its dispatch's shape ([`Placed::owed_behind_prefill`]), so a
/// classic pair states no request again. The front is whole without that:
/// always inlined, a classic answer pays the one test of its pointer here,
/// and its pair index is not read.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the pair placement's own arguments, the decode's answer and its guard"
)]
pub(crate) fn decode_again(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pairs: PairsAgain<'_>,
    wire: Option<WireConstraint>,
    inputs: PlacementInputs<'_>,
    decode: &Placed,
    ahead: WorkerLoadGuard,
) -> (Option<Placed>, WorkerLoadGuard) {
    match &decode.engine {
        None => (None, ahead),
        Some(first) => first.decode_again(registry, policies, model_id, pairs, wire, inputs, ahead),
    }
}

/// Why a pass of the pair placement placed no pair.
#[derive(Debug)]
enum PairMiss {
    /// Nobody was open on the failure's side at the pass's read, and the
    /// veto default did not take that side's pool: unless the verdict is an
    /// overload shed, the placement may look once more when its retry check
    /// sees a worker of the side available. Building an overload-shed verdict
    /// already counted the request in `smg_worker_overload_shed_total`.
    NobodyOpen(Box<PairFailure>),
    /// Any other miss: the pass's answer stands.
    Final(Box<PairFailure>),
}

/// One pass of the pair placement ([`select_pair`]). A miss says whether
/// the pass found nobody open on a side ([`PairMiss`]).
fn pair_pass(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    prefill_capacity: Option<&PrefillSelectionContext<'_>>,
    inputs: PlacementInputs<'_>,
) -> Result<Pair, PairMiss> {
    let admitted = |w: &Arc<dyn Worker>| {
        WireConstraint::admits(wire, w.as_ref())
            && inputs
                .candidate_filter
                .is_none_or(|accepts| accepts(w.as_ref()))
    };
    // The overload thresholds steer a pool only while one of the workers the
    // request may use in it is under them. A pool whose every such worker
    // is over them (and some otherwise routable) has nowhere to be steered:
    // the default steers it by load, as it steers a single-worker placement,
    // and a decode pool over them pairs by load with the prefill its own
    // policy picked rather than re-pick a prefill its pool never vetoed.
    // Under `--worker-overload-shed` such a pool is shed instead.
    let shedding = registry.overload_shed_enabled();
    let vetoed_alone = |pool: &[Arc<dyn Worker>]| {
        pool.iter()
            .any(|w| admitted(w) && overload::vetoed_by_overload_alone(w.as_ref()))
    };
    let lifted = |pool: &[Arc<dyn Worker>]| {
        !shedding && !pool.iter().any(|w| admitted(w) && w.is_available()) && vetoed_alone(pool)
    };
    // Independent prefill/decode policies so stateful ones (round robin) do
    // not share a counter; each role tags the sticky key with its own prefix.
    let prefill_policy = policies.get_prefill_policy();
    let decode_policy = policies.get_decode_policy();
    let prefill_over = lifted(&pairs.prefill);
    // The veto does not steer a prefill pool whose policy weighs load
    // against cache affinity itself (`SelectWorkerInfo::overload_lifted`):
    // under sustained load the prefill workers take turns crossing the
    // thresholds, and every crossing would send the sessions whose prefixes
    // that worker holds to another one, discarding their cache and pushing
    // that worker over in turn; a least-loaded pick over a pool all over
    // them discards the same. The policy's own gate moves a request off a
    // holder loaded well past the pool
    // (`LoadBalancingPolicy::balances_load_against_affinity`). A policy
    // without that affinity (load-blind, or one that hashes only over
    // workers under the thresholds) keeps the veto and the default.
    let lift_prefill = !shedding
        && prefill_policy.balances_load_against_affinity()
        && vetoed_alone(&pairs.prefill);
    let lift_decode = lifted(&pairs.decode_pool);
    let eligible = |w: &Arc<dyn Worker>, lift: bool| {
        admitted(w)
            && (w.is_available() || (lift && overload::vetoed_by_overload_alone(w.as_ref())))
    };
    let prefill_eligible = |w: &Arc<dyn Worker>| eligible(w, lift_prefill);
    let fail = |role: WorkerRole, verdict: PlacementFailure| {
        PairMiss::Final(Box::new(PairFailure { role, verdict }))
    };
    let nobody_open = |role: WorkerRole, verdict: PlacementFailure| {
        PairMiss::NobodyOpen(Box::new(PairFailure { role, verdict }))
    };

    // A role with no worker at all names itself before pairing is judged.
    if pairs.prefill_pool.is_empty() {
        debug!("No prefill workers");
        return Err(fail(WorkerRole::Prefill, PlacementFailure::NoCandidates));
    }
    if pairs.decode_pool.is_empty() {
        debug!("No decode workers");
        return Err(fail(WorkerRole::Decode, PlacementFailure::NoCandidates));
    }
    if let Some(refusal) = &pairs.refusal {
        warn!(
            model_id,
            mode = policies.pd_pairing_mode().as_str(),
            mismatches = ?refusal.mismatches,
            prefill = ?refusal.prefill,
            decode = ?refusal.decode,
            "No prefill/decode pair shares a KV transfer protocol"
        );
        return Err(fail(
            WorkerRole::Decode,
            PlacementFailure::NoCompatiblePair {
                prefill: refusal.prefill.clone(),
                decode: refusal.decode.clone(),
                mismatches: refusal.mismatches.clone(),
            },
        ));
    }

    // Live state: a prefill is open when it is available and so is one of
    // its partners, so the policy's pick can always be paired. The index
    // keeps known runtimes apart on a runtime-specific wire in every
    // pairing mode (`PdWire::mismatch`), so a pair its lists hold on one
    // known runtime is a pair the wire carries, and the pairs of every
    // runtime serve. One order is still decided here: a pair on one known
    // runtime goes before a pair with a worker whose runtime is unknown.
    let partner_open = |d: &Arc<dyn Worker>, runtime: Option<RuntimeType>| {
        eligible(d, lift_decode) && runtime.is_none_or(|r| d.metadata().spec.runtime_type == r)
    };
    let can_pair_on = |i: usize, runtime: Option<RuntimeType>| {
        pairs.partners[i].iter().any(|d| partner_open(d, runtime))
    };
    if !pairs.prefill.iter().any(&prefill_eligible) {
        let admitted = admitted_pool(&pairs.prefill, wire, &inputs);
        if let Some(pair) = relaxed_pair(
            registry,
            model_id,
            pairs,
            wire,
            prefill_capacity,
            &inputs,
            &admitted,
        ) {
            return pair.map_err(PairMiss::Final);
        }
        debug!(role = ?WorkerRole::Prefill, "No available PD pair");
        return Err(nobody_open(
            WorkerRole::Prefill,
            failure_from(&admitted, model_id, registry.overload_shed_enabled()),
        ));
    }
    // The candidates, in one pass. On a runtime-specific wire a prefill
    // pairs on its own runtime when that runtime is known and one of its
    // partners on it is open: such prefills are the candidates, whatever
    // their runtimes. Until the pass meets the first of them every open
    // prefill stands, with any open partner, and those are the candidates
    // when it meets none.
    let runtime_specific = pairs.wire.runtime_specific();
    let mut on_own_runtime = false;
    let mut open: Vec<usize> = Vec::new();
    for (i, p) in pairs.prefill.iter().enumerate() {
        if !prefill_eligible(p) {
            continue;
        }
        let runtime = p.metadata().spec.runtime_type;
        if runtime_specific && runtime.is_specified() && can_pair_on(i, Some(runtime)) {
            if !on_own_runtime {
                on_own_runtime = true;
                open.clear();
            }
            open.push(i);
        } else if !on_own_runtime && can_pair_on(i, None) {
            open.push(i);
        }
    }
    if open.is_empty() {
        // A lifted prefill pool with every worker vetoed had no prefill free
        // of the vetoes: the default is judged over it, as it was before the
        // lift. Otherwise the decode pool left the open prefills nothing to
        // pair with.
        let (role, failed_pool) = if prefill_over {
            (WorkerRole::Prefill, &pairs.prefill)
        } else {
            (WorkerRole::Decode, &pairs.decode_pool)
        };
        let admitted = admitted_pool(failed_pool, wire, &inputs);
        if let Some(pair) = relaxed_pair(
            registry,
            model_id,
            pairs,
            wire,
            prefill_capacity,
            &inputs,
            &admitted,
        ) {
            return pair.map_err(PairMiss::Final);
        }
        debug!(?role, "No available PD pair");
        return Err(nobody_open(
            role,
            failure_from(&admitted, model_id, registry.overload_shed_enabled()),
        ));
    }

    // Under Prefill admission, workers at their limit leave the candidate set
    // before the policy runs, so a stateful policy never commits a choice
    // the reservation then refuses.
    if let Some(capacity) =
        prefill_capacity.filter(|_| admission_prefilters(prefill_policy.as_ref(), inputs.headers))
    {
        open.retain(|&i| capacity.has_capacity(&pairs.prefill[i]));
        if open.is_empty() {
            return Err(fail(
                WorkerRole::Prefill,
                PlacementFailure::PrefillAtCapacity,
            ));
        }
    }
    // The open prefills as a part of the index's own list: the policy is
    // told the list, so what it keeps for the pool does not move with the
    // workers that are open for this request.
    let mut prefill = Narrowed::at(&pairs.prefill, open);
    // A retry steps aside from the prefill engines the request already
    // tried while another open prefill is left: a policy that places the
    // request the same way every time (a cached prefix, a routing key)
    // would pick the engine that just failed again. With every open prefill
    // tried they all stay. A strict target indexes this set and is never
    // narrowed, as in a single placement. A first attempt has tried nobody
    // and pays the one comparison.
    if !inputs.tried.is_empty() && !strict_target(prefill_policy.as_ref(), inputs.headers) {
        prefill.narrow_to(|p| !was_tried(p, inputs.tried));
    }
    // As for a single placement: the ring is read for the policies that walk
    // it alone. No other pair pays the registry's keyed read and the ring's
    // shared count, or fills the ring of a generation it never reads.
    let hash_ring = (policy_reads_hash_ring(prefill_policy.as_ref())
        || policy_reads_hash_ring(decode_policy.as_ref()))
    .then(|| policies.hash_ring(registry, model_id))
    .flatten();
    let mut info = SelectWorkerInfo {
        request_text: inputs.text,
        tokens: inputs.tokens,
        headers: inputs.headers,
        routing_key: policies.sticky_header_key(inputs.headers),
        rid_key: inputs.rid_key,
        cache_namespace: inputs.cache_namespace,
        hash_ring,
        role: WorkerRole::Prefill,
        overload_lifted: lift_prefill,
        named: Some(prefill.named()),
    };
    // Both pools were filtered for availability above (a pool the overload
    // veto cannot steer counting as available), so a miss here is the
    // policy's own decision.
    let declined = |role: WorkerRole, policy: &'static str| {
        fail(role, PlacementFailure::PolicyDeclined(policy))
    };
    let Some(prefill_idx) =
        policies.select_worker_for_model(&prefill_policy, model_id, prefill.workers(), &info)
    else {
        // A policy that declines a lifted pool all over the thresholds: the
        // least-loaded default steers it, as it steers any other such pool.
        if lift_prefill {
            let admitted = admitted_pool(&pairs.prefill, wire, &inputs);
            if let Some(pair) = relaxed_pair(
                registry,
                model_id,
                pairs,
                wire,
                prefill_capacity,
                &inputs,
                &admitted,
            ) {
                return pair.map_err(PairMiss::Final);
            }
        }
        return Err(declined(WorkerRole::Prefill, prefill_policy.name()));
    };
    let selected_prefill = prefill.workers()[prefill_idx].clone();
    if prefill_capacity.is_some_and(|capacity| !capacity.has_capacity(&selected_prefill)) {
        return Err(fail(
            WorkerRole::Prefill,
            PlacementFailure::PrefillAtCapacity,
        ));
    }
    // The pick's open partners, as a part of its partner list: non-empty by
    // construction, short of an availability flip between the two reads.
    let decode_runtime = on_own_runtime.then(|| selected_prefill.metadata().spec.runtime_type);
    let mut decode = Narrowed::of(&pairs.partners[prefill.position(prefill_idx)], |d| {
        partner_open(d, decode_runtime)
    });
    if decode.workers().is_empty() {
        debug!(
            ?decode_runtime,
            "The selected prefill's partners went unavailable"
        );
        return Err(fail(WorkerRole::Decode, PlacementFailure::Unavailable));
    }
    retain_decodes_with_a_room(decode_policy.as_ref(), inputs.headers, &mut decode);
    // The decode side steps aside from the tried engines the same way, among
    // the partners the rooms left, as the prefill side does among those the
    // admission left. A failed attempt lists both its engines whichever side
    // failed, and its decode has just given its room back: a tried partner
    // with a free room keeps the retry while the untried ones are full,
    // where the retry would wait at a gate and could be shed there.
    if !inputs.tried.is_empty() && !strict_target(decode_policy.as_ref(), inputs.headers) {
        decode.narrow_to(|d| !was_tried(d, inputs.tried));
    }
    // A decode pool all over the thresholds pairs by load, as the default
    // paired it: the least-loaded of the pick's partners, counted as a veto
    // fallback and not as a selection of the decode policy. Any other keeps
    // the veto for its policy.
    let selected_decode = if lift_decode {
        overload::least_loaded_uniform(decode.workers().iter(), &mut rand::rng())
            .cloned()
            .ok_or_else(|| fail(WorkerRole::Decode, PlacementFailure::Unavailable))?
    } else {
        info.role = WorkerRole::Decode;
        info.overload_lifted = false;
        info.named = Some(decode.named());
        match policies.select_worker_for_model(&decode_policy, model_id, decode.workers(), &info) {
            Some(decode_idx) => decode.workers()[decode_idx].clone(),
            None => return Err(declined(WorkerRole::Decode, decode_policy.name())),
        }
    };
    if prefill_over || lift_decode {
        Metrics::record_worker_overload_fallback(STAGE_SELECTION);
        debug!(
            branch = BRANCH_ALL_OVERLOADED_FALLBACK,
            prefill_pool = prefill_over,
            decode_pool = lift_decode,
            prefill = %selected_prefill.redacted_url(),
            decode = %selected_decode.redacted_url(),
            model_id,
            "Veto fallback: a prefill pool over the overload thresholds placed by its policy, or a decode pool by load"
        );
    }
    let runtime = selected_prefill.metadata().spec.runtime_type;
    Metrics::record_worker_selection(
        metrics_labels::WORKER_PREFILL,
        selected_prefill.connection_mode().as_metric_label(),
        model_id,
        prefill_policy.name(),
    );
    if !lift_decode {
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            selected_decode.connection_mode().as_metric_label(),
            model_id,
            decode_policy.name(),
        );
    }

    debug!(
        prefill = %selected_prefill.redacted_url(),
        decode = %selected_decode.redacted_url(),
        "Selected PD pair"
    );
    Ok(Pair {
        prefill: selected_prefill,
        decode: selected_decode,
        runtime,
    })
}

/// Narrow a decode pick's candidates to the workers whose admission gate
/// would seat one more room now ([`pd_admission::has_free_room`]), when some
/// have one and some do not; with every candidate full, or none reporting a
/// window, the list is left as it is and the policy chooses as before, the
/// dispatch waiting on its pick. An explicit target under consistent hashing
/// indexes the unfiltered list and is never narrowed ([`admission_prefilters`]),
/// so it waits at its own gate as before.
///
/// The pick reads the rooms the gate claims against because the policies
/// cannot: a load report never carries the places parked at the gateway, and
/// the router's in-flight count rises only on admission, so a load-scoring
/// policy kept picking the full side whose queue it was growing while its
/// siblings had rooms. Each candidate's gate is read once (a room taken
/// between two reads of the same worker could otherwise narrow the list to
/// nothing), the kept candidates stay in their order, and nothing is
/// allocated.
pub(crate) fn retain_decodes_with_a_room(
    decode_policy: &dyn LoadBalancingPolicy,
    headers: Option<&HeaderMap>,
    decode: &mut Narrowed<'_>,
) {
    if !admission_prefilters(decode_policy, headers) {
        return;
    }
    decode.narrow_to(pd_admission::has_free_room);
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        collections::{BTreeMap, HashMap},
        sync::Mutex,
        time::Duration,
    };

    use axum::http::StatusCode;
    use openai_protocol::{
        model_card::ModelCard,
        worker::{HealthCheckConfig, WorkerStatus},
    };
    use smg_policy_controller::{
        document::{AppliedSpec, DecodePick, KeySourceSpec, PolicySpec, RoutingKeySpec},
        engine::{Candidate, Decision, RequestFacts, Scorer, Target, TieBand},
        stages::{Lifecycle, StageRegistry},
        tracking::{EventSet, Lease, Outcome, ReleaseReason, Side},
    };
    use smg_worker::{
        connection::engine::EngineConnection,
        health::{circuit_breaker::CircuitState, resilience::ResolvedResilience},
        identity::{
            basic::{BasicWorker, WorkerMetadata, WorkerType},
            error::WorkerResult,
        },
        load::state::{one_rank_load, TestFeed},
        runtime::{RoutingState, StallReason},
        test_support::no_health_check,
    };
    use tracing_subscriber::Layer;

    use super::*;
    use crate::{
        config::types::{
            ManualAssignmentMode, PdPairingMode, PolicyConfig, RoutingKeyOverrideConfig,
        },
        observability::logging::test_support::test_subscriber,
        policies::{
            hash_ring::HashRing, CacheAwareConfig, CacheAwarePolicy, ConsistentHashingPolicy,
            LeastLoadPolicy, PowerOfTwoPolicy, RoundRobinPolicy,
        },
        routers::{
            common::{
                pair_index::PdWire,
                prefill_admission::{PrefillAdmission, PrefillAdmissionAttempt},
            },
            grpc::backend_client::BasicWorkerBuilder,
        },
    };

    pub(crate) const MODEL: &str = "m";

    /// A worker of a vLLM gRPC PD fleet, carrying the given KV connector
    /// (`None` leaves the transport unknown).
    fn pd_worker(url: &str, worker_type: WorkerType, connector: Option<&str>) -> BasicWorker {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new(MODEL))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Vllm)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            });
        if let Some(connector) = connector {
            builder = builder.kv_connector(connector);
        }
        builder.build()
    }

    /// A vLLM gRPC PD fleet of such workers.
    fn pd_registry(workers: &[(&str, WorkerType, Option<&str>)]) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        for (url, worker_type, connector) in workers {
            registry
                .register(Arc::new(pd_worker(url, *worker_type, *connector)))
                .expect("worker registers");
        }
        registry
    }

    pub(crate) fn pairs_of(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
    ) -> Arc<PdPairIndex> {
        registry
            .get_routing_snapshot(MODEL)
            .pd_pairs(PdWire::Grpc, policies.pd_pairing_mode())
    }

    fn pair_from(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
    ) -> Result<PlacedPair, Box<PairFailure>> {
        select_pair(
            registry,
            policies,
            MODEL,
            &pairs_of(registry, policies),
            None,
            None,
            PlacementInputs::default(),
            Stated::nothing,
        )
    }

    /// Two PD cohorts (here by KV transport) under round robin on both roles,
    /// as production configures them: one policy instance per role.
    fn cohort_policies() -> PolicyRegistry {
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        policies.set_prefill_policy(Arc::new(RoundRobinPolicy::new()));
        policies.set_decode_policy(Arc::new(RoundRobinPolicy::new()));
        policies
    }

    /// Selections per worker URL over `requests` sequential placements.
    fn hits(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        requests: usize,
    ) -> BTreeMap<String, usize> {
        let pairs = pairs_of(registry, policies);
        let mut hits = BTreeMap::new();
        for _ in 0..requests {
            let pair = select_pair(
                registry,
                policies,
                MODEL,
                &pairs,
                None,
                None,
                PlacementInputs::default(),
                Stated::nothing,
            )
            .expect("a pair is open");
            *hits.entry(pair.prefill.url().to_string()).or_insert(0) += 1;
            *hits.entry(pair.decode.url().to_string()).or_insert(0) += 1;
        }
        hits
    }

    /// One prefill and two decode workers, each decode with a running window
    /// of `window` rooms, so the admission gate's rooms count in the pick.
    fn gated_pd_registry(window: u16) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        for (url, worker_type) in [
            ("grpc://p:1", WorkerType::Prefill),
            ("grpc://d:1", WorkerType::Decode),
            ("grpc://d:2", WorkerType::Decode),
        ] {
            let mut labels = HashMap::new();
            if matches!(worker_type, WorkerType::Decode) {
                labels.insert("max_running_requests".to_string(), window.to_string());
            }
            registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(url)
                        .model(ModelCard::new(MODEL))
                        .worker_type(worker_type)
                        .connection_mode(ConnectionMode::Grpc)
                        .runtime_type(RuntimeType::Vllm)
                        .kv_connector("NixlConnector")
                        .labels(labels)
                        .health_config(HealthCheckConfig {
                            disable_health_check: true,
                            ..Default::default()
                        })
                        .build(),
                ))
                .expect("worker registers");
        }
        registry
    }

    /// A decode partner whose window is full, or whose free room a place
    /// waiting at its gate is in line for, is left out of the pick while its
    /// sibling has a room; with both full the policy chooses as before.
    #[test]
    fn a_full_decode_window_sends_the_pick_to_the_partner_with_a_room() {
        let registry = gated_pd_registry(2);
        let policies = cohort_policies();
        let full = registry.get_by_url("grpc://d:1").expect("registered");
        assert!(full.try_admit_pd(2, 2), "both rooms claimed");
        let picks = hits(&registry, &policies, 10);
        assert_eq!(
            picks.get("grpc://d:1").copied().unwrap_or(0),
            0,
            "{picks:?}"
        );
        assert_eq!(picks["grpc://d:2"], 10, "{picks:?}");

        // One room freed, but a place already waits for it.
        full.release_pd(1);
        let place = full.pd_enqueue(1);
        let picks = hits(&registry, &policies, 10);
        assert_eq!(
            picks.get("grpc://d:1").copied().unwrap_or(0),
            0,
            "{picks:?}"
        );
        full.pd_dequeue(place);
        let picks = hits(&registry, &policies, 10);
        assert_eq!(picks["grpc://d:1"], 5, "{picks:?}");
        assert_eq!(picks["grpc://d:2"], 5, "{picks:?}");

        // Every partner full: the policy's own rotation, and the dispatch
        // waits on its pick as before.
        let other = registry.get_by_url("grpc://d:2").expect("registered");
        assert!(full.try_admit_pd(1, 2) && other.try_admit_pd(2, 2));
        let picks = hits(&registry, &policies, 10);
        assert_eq!(picks["grpc://d:1"], 5, "{picks:?}");
        assert_eq!(picks["grpc://d:2"], 5, "{picks:?}");
    }

    /// The narrowing keeps the candidates with a room in their order (a
    /// policy that rotates over its list, or indexes it, sees the same
    /// sequence minus the full ones) and leaves a list alone when every
    /// candidate is full or every one has a room.
    #[test]
    fn the_narrowing_keeps_the_candidates_with_a_room_in_their_order() {
        let decode = |url: &str| -> Arc<dyn Worker> {
            Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new(MODEL))
                    .worker_type(WorkerType::Decode)
                    .connection_mode(ConnectionMode::Grpc)
                    .labels(HashMap::from([(
                        "max_running_requests".to_string(),
                        "1".to_string(),
                    )]))
                    .build(),
            )
        };
        let policy = RoundRobinPolicy::new();
        let list: Arc<[Arc<dyn Worker>]> = ["grpc://d:1", "grpc://d:2", "grpc://d:3", "grpc://d:4"]
            .map(decode)
            .into();
        let urls = |candidates: &Narrowed<'_>| -> Vec<String> {
            candidates
                .workers()
                .iter()
                .map(|d| d.url().to_string())
                .collect()
        };
        let mut candidates = Narrowed::of(&list, |_| true);
        assert!(list[0].try_admit_pd(1, 1) && list[2].try_admit_pd(1, 1));
        retain_decodes_with_a_room(&policy, None, &mut candidates);
        assert_eq!(urls(&candidates), ["grpc://d:2", "grpc://d:4"]);
        // Each kept candidate keeps its own position in the list beside it.
        assert_eq!(candidates.named().positions, Some(&[1, 3][..]));

        // Every candidate full, then every one with a room: the list as it was.
        assert!(list[1].try_admit_pd(1, 1) && list[3].try_admit_pd(1, 1));
        retain_decodes_with_a_room(&policy, None, &mut candidates);
        assert_eq!(
            candidates.workers().len(),
            2,
            "nothing to keep: the policy's list"
        );
        list[1].release_pd(1);
        list[3].release_pd(1);
        retain_decodes_with_a_room(&policy, None, &mut candidates);
        assert_eq!(
            urls(&candidates),
            ["grpc://d:2", "grpc://d:4"],
            "nothing to cut"
        );
        assert_eq!(candidates.named().positions, Some(&[1, 3][..]));
    }

    /// What a policy was told of the list its candidates are cut from: the
    /// role, the positions, and whether each candidate is the list's own
    /// worker at its position.
    type Told = (WorkerRole, Option<Vec<usize>>, bool);

    /// A policy that records what it is told, per selection, and takes its
    /// first candidate.
    #[derive(Debug, Default)]
    struct NameRecorder {
        seen: Mutex<Vec<Told>>,
    }

    impl LoadBalancingPolicy for NameRecorder {
        fn select_worker(
            &self,
            workers: &[Arc<dyn Worker>],
            info: &SelectWorkerInfo,
        ) -> Option<usize> {
            let told = info.named.map(|named| {
                let positions: Vec<usize> = match named.positions {
                    Some(positions) => positions.to_vec(),
                    None => (0..named.list.len()).collect(),
                };
                let agrees = positions.len() == workers.len()
                    && positions
                        .iter()
                        .zip(workers)
                        .all(|(&at, worker)| Arc::ptr_eq(&named.list[at], worker));
                (positions, agrees)
            });
            let (positions, agrees) = match told {
                Some((positions, agrees)) => (Some(positions), agrees),
                None => (None, false),
            };
            self.seen
                .lock()
                .unwrap()
                .push((info.role, positions, agrees));
            (!workers.is_empty()).then_some(0)
        }

        fn name(&self) -> &'static str {
            "name_recorder"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// A pair placement tells each role's policy the list its candidates are
    /// cut from and each candidate's position in it: the index's prefill
    /// list with the open prefills, the pick's partner list with its open
    /// partners. A worker that closes leaves the candidates and the
    /// positions together.
    #[test]
    fn a_pair_placement_names_both_sides_with_their_positions() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:3", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let prefill = Arc::new(NameRecorder::default());
        let decode = Arc::new(NameRecorder::default());
        policies.set_prefill_policy(prefill.clone());
        policies.set_decode_policy(decode.clone());
        let pairs = pairs_of(&registry, &policies);
        let place = || {
            select_pair(
                &registry,
                &policies,
                MODEL,
                &pairs,
                None,
                None,
                PlacementInputs::default(),
                Stated::nothing,
            )
            .map(|_| ())
        };
        place().expect("a pair");
        // The list's first prefill and its first decode close: both open
        // sets start at position 1.
        pairs.prefill[0].set_status(WorkerStatus::NotReady);
        pairs.partners[1][0].set_status(WorkerStatus::NotReady);
        place().expect("a pair");
        assert_eq!(
            *prefill.seen.lock().unwrap(),
            [
                (WorkerRole::Prefill, Some(vec![0, 1, 2]), true),
                (WorkerRole::Prefill, Some(vec![1, 2]), true),
            ]
        );
        assert_eq!(
            *decode.seen.lock().unwrap(),
            [
                (WorkerRole::Decode, Some(vec![0, 1]), true),
                (WorkerRole::Decode, Some(vec![1]), true),
            ]
        );
    }

    /// The regular entry names the registry's own pool to its policy, so
    /// the cache-aware policy keeps the pool's one table by the list's
    /// name, under a candidate filter too while it refuses nobody. A filter
    /// that refuses a worker and a retry's tried engines hand the policy
    /// lists of their own, which nobody names: each has a table by its
    /// workers. The filter is asked once a candidate either way.
    #[test]
    fn the_regular_entry_names_its_pool_until_something_narrows_it() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::policies::{as_cache_aware, test_support::hold_prompt};

        let registry = registry_with(&[
            ("http://w1:8000", ConnectionMode::Http, RuntimeType::Vllm),
            ("http://w2:8000", ConnectionMode::Http, RuntimeType::Vllm),
            ("http://w3:8000", ConnectionMode::Http, RuntimeType::Vllm),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096));
        let pool = registry.get_routing_pool(MODEL, RoutingPool::HttpRegular);
        assert_eq!(pool.len(), 3);
        // The pool's first worker holds the prompt in the model's index.
        let tokens = hold_prompt(&policies, pool[0].as_ref());
        let default_policy = policies.get_default_policy();
        let cache_aware = as_cache_aware(default_policy.as_ref()).expect("cache-aware by default");
        let place = |inputs: PlacementInputs<'_>| {
            select_single(
                &registry,
                &policies,
                MODEL,
                RoutingPool::HttpRegular,
                None,
                inputs,
                Stated::nothing,
            )
            .expect("a worker")
        };
        let inputs = PlacementInputs {
            tokens: Some(&tokens),
            ..Default::default()
        };
        assert_eq!(place(inputs).url(), pool[0].url(), "the holder");
        place(inputs);
        assert_eq!(
            cache_aware.kept_tables(),
            (1, 0),
            "the pool's one table, by the list's name"
        );
        // A filter that refuses nobody (the HTTP router's ledger on a first
        // attempt) leaves the pool as it was drawn, with its name.
        let asked = AtomicUsize::new(0);
        let everyone = |_: &dyn Worker| {
            asked.fetch_add(1, Ordering::Relaxed);
            true
        };
        let picked = place(PlacementInputs {
            candidate_filter: Some(&everyone),
            ..inputs
        });
        assert_eq!(picked.url(), pool[0].url(), "the holder, under the filter");
        assert_eq!(
            cache_aware.kept_tables(),
            (1, 0),
            "a filter that refuses nobody leaves the pool its name"
        );
        assert_eq!(asked.load(Ordering::Relaxed), 3, "each candidate once");
        // A filter that refuses a worker hands the policy a list of its own:
        // the workers before the refused one and those after it that pass,
        // each asked once.
        let (second_url, holder_url) = (pool[1].url().to_string(), pool[0].url().to_string());
        let not_the_second = |worker: &dyn Worker| {
            asked.fetch_add(1, Ordering::Relaxed);
            worker.url() != second_url
        };
        let picked = place(PlacementInputs {
            candidate_filter: Some(&not_the_second),
            ..inputs
        });
        assert_eq!(
            picked.url(),
            pool[0].url(),
            "the holder stands before the refused worker and stays a candidate"
        );
        assert_eq!(
            cache_aware.kept_tables(),
            (1, 1),
            "a list a filter narrowed is nobody's"
        );
        assert_eq!(asked.load(Ordering::Relaxed), 6, "each candidate once");
        let not_the_holder = |worker: &dyn Worker| {
            asked.fetch_add(1, Ordering::Relaxed);
            worker.url() != holder_url
        };
        let picked = place(PlacementInputs {
            candidate_filter: Some(&not_the_holder),
            ..inputs
        });
        assert_ne!(picked.url(), pool[0].url(), "never the refused worker");
        assert_eq!(cache_aware.kept_tables(), (1, 2), "nor is the next one");
        assert_eq!(asked.load(Ordering::Relaxed), 9, "each candidate once");
        let tried = [pool[2].base_url().to_string()];
        place(PlacementInputs {
            tried: &tried,
            ..inputs
        });
        assert_eq!(cache_aware.kept_tables(), (1, 3), "nor is a retry's");
    }

    /// An explicit target under consistent hashing indexes the partners the
    /// policy is given, so a full partner is never dropped from under it: the
    /// targeted full worker is picked and waits at its gate as before, and a
    /// target past the full one keeps its index.
    #[test]
    fn an_explicit_target_under_consistent_hashing_keeps_its_full_decode_partner() {
        let registry = gated_pd_registry(2);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        policies.set_prefill_policy(Arc::new(RoundRobinPolicy::new()));
        policies.set_decode_policy(Arc::new(ConsistentHashingPolicy::new()));
        let full = registry.get_by_url("grpc://d:1").expect("registered");
        assert!(full.try_admit_pd(2, 2), "both rooms claimed");
        let pairs = pairs_of(&registry, &policies);
        for (index, expected) in [(0, "grpc://d:1"), (1, "grpc://d:2")] {
            let mut headers = HeaderMap::new();
            headers.insert("x-smg-target-worker", index.to_string().parse().unwrap());
            let pair = select_pair(
                &registry,
                &policies,
                MODEL,
                &pairs,
                None,
                None,
                PlacementInputs {
                    headers: Some(&headers),
                    ..Default::default()
                },
                Stated::nothing,
            )
            .expect("the targeted pair is open");
            assert_eq!(pair.decode.url(), expected, "target {index}");
        }
    }

    /// The pair a routing key is placed on in `registry` under consistent
    /// hashing on both sides, with `tried` as the engines the request
    /// already tried.
    fn keyed_pair(registry: &WorkerRegistry, tried: &[String]) -> (String, String) {
        let policies = PolicyRegistry::new(PolicyConfig::ConsistentHashing);
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "session-42".parse().unwrap());
        let pair = select_pair(
            registry,
            &policies,
            MODEL,
            &pairs_of(registry, &policies),
            None,
            None,
            PlacementInputs {
                headers: Some(&headers),
                tried,
                ..Default::default()
            },
            Stated::nothing,
        )
        .expect("a pair is open");
        (
            pair.prefill.url().to_string(),
            pair.decode.url().to_string(),
        )
    }

    /// The engine of the worker registered at `url`, as a retry lists it.
    fn engine_of(registry: &WorkerRegistry, url: &str) -> String {
        let worker = registry.get_by_url(url).expect("registered");
        worker.base_url().to_string()
    }

    /// A pair's retry steps aside from the engines the request already
    /// tried, a side at a time and only while that side has another open
    /// candidate. A routing key names the same pair every time; the tried
    /// prefill engine moves the prefill side alone and the tried decode
    /// engine the decode side alone; a side whose every engine was tried is
    /// placed over all of them again, on the worker the key names.
    #[test]
    fn a_pairs_retry_steps_aside_from_the_tried_engines_a_side_at_a_time() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, None),
            ("grpc://p:2", WorkerType::Prefill, None),
            ("grpc://d:1", WorkerType::Decode, None),
            ("grpc://d:2", WorkerType::Decode, None),
        ]);
        let engine = |url: &str| engine_of(&registry, url);

        let (prefill, decode) = keyed_pair(&registry, &[]);
        assert_eq!(
            keyed_pair(&registry, &[]),
            (prefill.clone(), decode.clone()),
            "with nobody tried the key names one pair"
        );

        let (moved, kept) = keyed_pair(&registry, &[engine(&prefill)]);
        assert_ne!(moved, prefill, "the retry leaves the tried prefill engine");
        assert_eq!(kept, decode, "and keeps the decode nobody tried");

        let (kept, moved) = keyed_pair(&registry, &[engine(&decode)]);
        assert_eq!(kept, prefill, "the prefill nobody tried stays");
        assert_ne!(moved, decode, "the retry leaves the tried decode engine");

        let (other_prefill, other_decode) =
            keyed_pair(&registry, &[engine(&prefill), engine(&decode)]);
        assert!(
            other_prefill != prefill && other_decode != decode,
            "a tried pair moves on both sides"
        );

        let every_prefill = [engine("grpc://p:1"), engine("grpc://p:2")];
        assert_eq!(
            keyed_pair(&registry, &every_prefill),
            (prefill.clone(), decode.clone()),
            "with every prefill tried the prefill side is placed over all of them"
        );
        let everyone = [
            engine("grpc://p:1"),
            engine("grpc://p:2"),
            engine("grpc://d:1"),
            engine("grpc://d:2"),
        ];
        assert_eq!(
            keyed_pair(&registry, &everyone),
            (prefill, decode),
            "with every engine tried the key names its pair again"
        );
    }

    /// The narrowing is a preference and refuses nobody: a tried worker that
    /// is the only one open on its side takes the retry, whether it is the
    /// only one registered or the untried one beside it is not ready.
    #[test]
    fn a_pairs_retry_goes_back_to_a_tried_worker_that_is_the_only_one_open() {
        let alone = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, None),
            ("grpc://d:1", WorkerType::Decode, None),
        ]);
        let tried = [
            engine_of(&alone, "grpc://p:1"),
            engine_of(&alone, "grpc://d:1"),
        ];
        assert_eq!(
            keyed_pair(&alone, &tried),
            ("grpc://p:1".to_string(), "grpc://d:1".to_string())
        );

        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, None),
            ("grpc://p:2", WorkerType::Prefill, None),
            ("grpc://d:1", WorkerType::Decode, None),
            ("grpc://d:2", WorkerType::Decode, None),
        ]);
        for untried in ["grpc://p:2", "grpc://d:2"] {
            let worker = registry.get_by_url(untried).expect("registered");
            worker.set_status(WorkerStatus::NotReady);
        }
        let tried = [
            engine_of(&registry, "grpc://p:1"),
            engine_of(&registry, "grpc://d:1"),
        ];
        assert_eq!(
            keyed_pair(&registry, &tried),
            ("grpc://p:1".to_string(), "grpc://d:1".to_string())
        );
    }

    /// A retry's decode side reads the rooms before it steps aside from the
    /// tried engines: a failed attempt lists both its engines whichever side
    /// failed, and a tried decode with a free room keeps the retry while the
    /// untried partner is full, where the retry would wait at a gate. With a
    /// room on each, or on neither, the retry leaves the tried decode.
    #[test]
    fn a_pairs_retry_keeps_a_tried_decode_with_a_room_while_the_untried_one_is_full() {
        let registry = gated_pd_registry(2);
        let policies = cohort_policies();
        let pairs = pairs_of(&registry, &policies);
        let tried = [engine_of(&registry, "grpc://d:1")];
        let retried_decode = || {
            let pair = select_pair(
                &registry,
                &policies,
                MODEL,
                &pairs,
                None,
                None,
                PlacementInputs {
                    tried: &tried,
                    ..Default::default()
                },
                Stated::nothing,
            )
            .expect("a pair is open");
            pair.decode.url().to_string()
        };
        let tried_decode = registry.get_by_url("grpc://d:1").expect("registered");
        let untried_decode = registry.get_by_url("grpc://d:2").expect("registered");

        for _ in 0..4 {
            assert_eq!(retried_decode(), "grpc://d:2", "a room on each");
        }

        assert!(untried_decode.try_admit_pd(2, 2), "both rooms claimed");
        for _ in 0..4 {
            assert_eq!(
                retried_decode(),
                "grpc://d:1",
                "the tried decode has the only room"
            );
        }

        assert!(tried_decode.try_admit_pd(2, 2), "both rooms claimed");
        for _ in 0..4 {
            assert_eq!(
                retried_decode(),
                "grpc://d:2",
                "a room on neither: the retry waits at the untried decode's gate"
            );
        }
    }

    /// An explicit target under consistent hashing holds a pair's retry to
    /// the targeted worker on the side whose policy it is, though that
    /// worker was tried: the target indexes the candidates, and a narrowed
    /// list would put another worker at its index. A side under another
    /// policy still steps aside.
    #[test]
    fn an_explicit_target_under_consistent_hashing_keeps_its_side_of_a_pair_on_retry() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, None),
            ("grpc://p:2", WorkerType::Prefill, None),
            ("grpc://d:1", WorkerType::Decode, None),
            ("grpc://d:2", WorkerType::Decode, None),
        ]);
        let place = |policies: &PolicyRegistry, index: usize, tried: &[String]| {
            let mut headers = HeaderMap::new();
            headers.insert("x-smg-target-worker", index.to_string().parse().unwrap());
            let pair = select_pair(
                &registry,
                policies,
                MODEL,
                &pairs_of(&registry, policies),
                None,
                None,
                PlacementInputs {
                    headers: Some(&headers),
                    tried,
                    ..Default::default()
                },
                Stated::nothing,
            )
            .expect("the targeted pair is open");
            (pair.prefill.into_worker(), pair.decode.into_worker())
        };

        let both = PolicyRegistry::new(PolicyConfig::ConsistentHashing);
        for index in 0..2 {
            let (prefill, decode) = place(&both, index, &[]);
            let tried = [
                prefill.base_url().to_string(),
                decode.base_url().to_string(),
            ];
            let (again_prefill, again_decode) = place(&both, index, &tried);
            assert_eq!(
                (again_prefill.url(), again_decode.url()),
                (prefill.url(), decode.url()),
                "target {index}: the retry stays on the targeted pair"
            );
        }

        // The prefill side alone is held: the decode side, under a policy
        // that takes no target, leaves the tried engine on every retry.
        let prefill_only = PolicyRegistry::new(PolicyConfig::ConsistentHashing);
        prefill_only.set_decode_policy(Arc::new(RoundRobinPolicy::new()));
        let (prefill, decode) = place(&prefill_only, 0, &[]);
        let tried = [
            prefill.base_url().to_string(),
            decode.base_url().to_string(),
        ];
        for _ in 0..4 {
            let (again_prefill, again_decode) = place(&prefill_only, 0, &tried);
            assert_eq!(again_prefill.url(), prefill.url());
            assert_ne!(again_decode.url(), decode.url());
        }
    }

    /// A pair's ring read is the single placement's: for the policies that
    /// walk the ring alone. A pair of other policies reads none, and so
    /// fills the ring of no generation.
    #[test]
    fn a_pair_reads_the_hash_ring_for_a_policy_that_walks_it_alone() {
        // A pair is placed under `prefill` and `decode`: whether the
        // generation's ring was filled for it.
        type Policy = Arc<dyn LoadBalancingPolicy>;
        let fills_the_ring = |prefill: Policy, decode: Policy| {
            let registry = gated_pd_registry(2);
            let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
            policies.set_prefill_policy(prefill);
            policies.set_decode_policy(decode);
            let pairs = pairs_of(&registry, &policies);
            select_pair(
                &registry,
                &policies,
                MODEL,
                &pairs,
                None,
                None,
                PlacementInputs::default(),
                Stated::nothing,
            )
            .expect("a pair is open");
            registry.get_routing_snapshot(MODEL).hash_ring_is_filled()
        };

        let walks_none = || -> Policy { Arc::new(RoundRobinPolicy::new()) };
        let walks_the_ring = || -> Policy { Arc::new(ConsistentHashingPolicy::new()) };
        assert!(
            !fills_the_ring(walks_none(), walks_none()),
            "no policy of the pair walks the ring"
        );
        assert!(
            fills_the_ring(walks_none(), walks_the_ring()),
            "the decode policy walks the ring"
        );
        assert!(
            fills_the_ring(walks_the_ring(), walks_none()),
            "the prefill policy walks the ring"
        );
    }

    /// The load-scoring decode policies price a side by its report and
    /// their own dispatch credit, neither of which holds a request parked at
    /// the gate, and a fuller batch reports the higher aggregate rate, so a
    /// full decode side read as the cheaper pick while its queue grew. With
    /// the gate's rooms in the selection the full side is not offered.
    #[test]
    fn the_load_scoring_policies_are_not_offered_a_full_decode_window() {
        let scorers: [Arc<dyn LoadBalancingPolicy>; 2] = [
            Arc::new(LeastLoadPolicy::new()),
            Arc::new(PowerOfTwoPolicy::new()),
        ];
        for decode_policy in scorers {
            let name = decode_policy.name();
            let registry = gated_pd_registry(8);
            let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
            policies.set_prefill_policy(Arc::new(RoundRobinPolicy::new()));
            let feed = TestFeed::attached_to(|rx| decode_policy.as_ref().set_load_receiver(rx));
            policies.set_decode_policy(decode_policy);
            // d:1 runs its whole window of eight at 160 tokens/s and holds
            // every room; d:2 runs one request at 20 tokens/s.
            let mut full_report = one_rank_load(0, 0.0, 160.0);
            full_report.loads[0].num_running_reqs = 8;
            feed.publish(&HashMap::from([
                ("grpc://d:1".to_string(), full_report),
                ("grpc://d:2".to_string(), one_rank_load(0, 0.0, 20.0)),
            ]));
            let full = registry.get_by_url("grpc://d:1").expect("registered");
            assert!(full.try_admit_pd(8, 8));
            let tokens: Vec<u32> = (0..512).collect();
            let pairs = pairs_of(&registry, &policies);
            for _ in 0..20 {
                let pair = select_pair(
                    &registry,
                    &policies,
                    MODEL,
                    &pairs,
                    None,
                    None,
                    PlacementInputs {
                        tokens: Some(&tokens),
                        ..Default::default()
                    },
                    Stated::nothing,
                )
                .expect("a pair is open");
                assert_eq!(
                    pair.decode.url(),
                    "grpc://d:2",
                    "{name} picked the full side"
                );
            }
        }
    }

    #[test]
    fn two_cohorts_of_one_prefill_and_two_decodes_use_every_decode() {
        // 1P2D beside 1P2D. Round robin over the two prefills alternates the
        // cohorts; the decode rotation must be each cohort's own, or the
        // alternation pins every cohort to one of its two decodes (reported
        // from a rollout: 120 requests, decodes 60/0 and 60/0).
        let registry = pd_registry(&[
            ("grpc://p:a", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:b", WorkerType::Prefill, Some("MooncakeConnector")),
            ("grpc://d:a1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:a2", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:b1", WorkerType::Decode, Some("MooncakeConnector")),
            ("grpc://d:b2", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        let hits = hits(&registry, &cohort_policies(), 120);
        assert_eq!(hits["grpc://p:a"], 60, "{hits:?}");
        assert_eq!(hits["grpc://p:b"], 60, "{hits:?}");
        for decode in ["grpc://d:a1", "grpc://d:a2", "grpc://d:b1", "grpc://d:b2"] {
            assert_eq!(
                hits.get(decode).copied().unwrap_or(0),
                30,
                "{decode}: {hits:?}"
            );
        }
    }

    #[test]
    fn a_two_prefill_cohort_beside_a_one_prefill_cohort_keeps_each_decode_pool_even() {
        // 2P1D beside 1P2D. Today the prefill policy sees all three prefills,
        // so the cohorts split 2:1 by prefill count (the decode pools then
        // carry 120 and 30/30 of 180); weighting cohorts by their bottleneck
        // pool is a separate change. Whatever the split, a cohort's decodes
        // must share its traffic evenly and the pools must balance.
        let registry = pd_registry(&[
            ("grpc://p:a1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:a2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:b", WorkerType::Prefill, Some("MooncakeConnector")),
            ("grpc://d:a", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:b1", WorkerType::Decode, Some("MooncakeConnector")),
            ("grpc://d:b2", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        let hits = hits(&registry, &cohort_policies(), 180);
        let cohort_a = hits["grpc://p:a1"] + hits["grpc://p:a2"];
        let cohort_b = hits["grpc://p:b"];
        assert_eq!(cohort_a + cohort_b, 180, "{hits:?}");
        assert_eq!(hits["grpc://p:a1"], hits["grpc://p:a2"], "{hits:?}");
        assert_eq!(hits["grpc://d:a"], cohort_a, "{hits:?}");
        assert_eq!(hits["grpc://d:b1"], cohort_b / 2, "{hits:?}");
        assert_eq!(hits["grpc://d:b2"], cohort_b / 2, "{hits:?}");
    }

    #[test]
    fn a_pair_shares_its_kv_transport() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("MooncakeConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);

        // The snapshot's index pairs each prefill with the decode that
        // speaks its transport, once, for every request to read.
        let pairs = pairs_of(&registry, &policies);
        assert_eq!(pairs.prefill.len(), 2);
        for (prefill, partners) in pairs.prefill.iter().zip(&pairs.partners) {
            assert_eq!(partners.len(), 1, "{}", prefill.url());
            assert_eq!(
                prefill.pd_pairing().transport(),
                partners[0].pd_pairing().transport()
            );
        }
        assert!(Arc::ptr_eq(&pairs, &pairs_of(&registry, &policies)));

        // And a placement lands on a matching pair.
        let pair = pair_from(&registry, &policies).expect("the fleet has matching pairs");
        assert_eq!(
            pair.prefill.pd_pairing().transport(),
            pair.decode.pd_pairing().transport()
        );
    }

    #[test]
    fn a_pair_on_one_known_runtime_goes_before_a_pair_with_an_unknown_one() {
        // P1 (NIXL) pairs only with D1, whose runtime probe failed; P2
        // (Mooncake) pairs with D1 and D2. On the gRPC wire a pair on one
        // known runtime goes first: while P2/D2 is open P1 is no candidate
        // and D1 no partner, and every placement lands on P2/D2.
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("MooncakeConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new("grpc://d:1")
                    .model(ModelCard::new(MODEL))
                    .worker_type(WorkerType::Decode)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::Unspecified)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ))
            .expect("worker registers");
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);

        for _ in 0..4 {
            let pair = pair_from(&registry, &policies).expect("P2 and D2 pair on one runtime");
            assert_eq!(pair.prefill.url(), "grpc://p:2");
            assert_eq!(pair.decode.url(), "grpc://d:2");
            assert_eq!(pair.runtime, RuntimeType::Vllm);
        }

        // With D2 down no pair on a known runtime is open, and the pairs
        // with D1 serve: both prefills take a turn.
        registry
            .get_by_url("grpc://d:2")
            .expect("registered")
            .set_status(WorkerStatus::NotReady);
        let mut prefills: Vec<String> = (0..4)
            .map(|_| {
                let pair = pair_from(&registry, &policies).expect("D1 pairs with either prefill");
                assert_eq!(pair.decode.url(), "grpc://d:1");
                pair.prefill.url().to_string()
            })
            .collect();
        prefills.sort();
        prefills.dedup();
        assert_eq!(
            prefills,
            ["grpc://p:1", "grpc://p:2"],
            "P1 and P2 both pair with D1"
        );
    }

    /// A fleet with a complete pair on each of two runtimes: both pairs
    /// serve, under the lenient pairing mode and under off, and no pair
    /// crosses the runtimes. The selection used to hand every request to
    /// the runtime of the first open prefill.
    #[test]
    fn pairs_on_two_runtimes_both_take_requests() {
        let registry = registry_of(&[
            (
                "grpc://p:vllm",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
            (
                "grpc://p:sglang",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
            (
                "grpc://d:vllm",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
            (
                "grpc://d:sglang",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
        ]);
        for mode in [PdPairingMode::Lenient, PdPairingMode::Off] {
            let policies = cohort_policies().with_pd_pairing_mode(mode);
            let pairs = pairs_of(&registry, &policies);
            let mut served = BTreeMap::new();
            for _ in 0..8 {
                let pair = select_pair(
                    &registry,
                    &policies,
                    MODEL,
                    &pairs,
                    None,
                    None,
                    PlacementInputs::default(),
                    Stated::nothing,
                )
                .expect("a pair is open");
                assert_eq!(
                    pair.decode.metadata().spec.runtime_type,
                    pair.runtime,
                    "{mode:?}: a pair is on one runtime"
                );
                *served.entry(pair.runtime.as_str()).or_insert(0) += 1;
            }
            assert_eq!(
                served,
                BTreeMap::from([("sglang", 4), ("vllm", 4)]),
                "{mode:?}: both pairs serve"
            );
        }
    }

    /// A pair of two workers whose runtimes are unknown waits behind a pair
    /// on a known runtime, wherever the pool lists it.
    #[test]
    fn a_pair_of_unknown_runtimes_waits_behind_a_pair_on_a_known_one() {
        let unknown = [
            (
                "grpc://p:unknown",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Unspecified,
            ),
            (
                "grpc://d:unknown",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Unspecified,
            ),
        ];
        let known = [
            (
                "grpc://p:vllm",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
            (
                "grpc://d:vllm",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
        ];
        // Listed before the known pair, then after it.
        for listed in [[unknown, known].concat(), [known, unknown].concat()] {
            let registry = registry_of(&listed);
            let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
            for _ in 0..4 {
                let pair = pair_from(&registry, &policies).expect("the vLLM pair is open");
                assert_eq!(pair.prefill.url(), "grpc://p:vllm");
                assert_eq!(pair.decode.url(), "grpc://d:vllm");
            }
        }
    }

    #[test]
    fn a_pair_whose_partners_are_down_does_not_shut_out_a_healthy_pair_on_another_runtime() {
        // One model served by a vLLM pair and an SGLang pair mid-migration.
        // The vLLM decode goes down: the vLLM prefill, first in pool order,
        // must not dictate the runtime, so the SGLang pair keeps serving.
        let registry = registry_of(&[
            (
                "grpc://p:vllm",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
            (
                "grpc://p:sglang",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
            (
                "grpc://d:vllm",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
            (
                "grpc://d:sglang",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let vllm_decode = registry
            .get_all()
            .into_iter()
            .find(|w| w.url() == "grpc://d:vllm")
            .expect("registered");
        vllm_decode.set_status(WorkerStatus::NotReady);

        for _ in 0..3 {
            let pair = pair_from(&registry, &policies).expect("the SGLang pair is healthy");
            assert_eq!(pair.prefill.url(), "grpc://p:sglang");
            assert_eq!(pair.decode.url(), "grpc://d:sglang");
            assert_eq!(pair.runtime, RuntimeType::Sglang);
        }
    }

    #[test]
    fn a_fleet_without_a_shared_transport_names_both_roles() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);

        let failure = pair_from(&registry, &policies)
            .err()
            .expect("no pair shares a transport");
        assert_eq!(failure.role, WorkerRole::Decode);
        match failure.verdict {
            PlacementFailure::NoCompatiblePair {
                prefill,
                decode,
                mismatches,
            } => {
                assert_eq!(prefill, ["vllm/nixl/?"]);
                assert_eq!(decode, ["vllm/mooncake/?"]);
                assert_eq!(mismatches, ["transport"]);
            }
            _ => panic!("expected NoCompatiblePair"),
        }

        // `off` restores pre-pairing placement for the same fleet.
        let off =
            PolicyRegistry::new(PolicyConfig::RoundRobin).with_pd_pairing_mode(PdPairingMode::Off);
        assert!(pair_from(&registry, &off).is_ok());
    }

    /// A fleet with no pair on one runtime is refused by name under pairing
    /// off, in the words the lenient mode refuses it with, and the
    /// least-loaded default has no pair to steer to either once the prefill
    /// is over the thresholds.
    #[test]
    fn pairing_off_refuses_a_fleet_with_no_shared_runtime_by_name() {
        let registry = registry_of(&[
            (
                "grpc://p:1",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
            (
                "grpc://d:1",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
        ]);
        let off =
            PolicyRegistry::new(PolicyConfig::RoundRobin).with_pd_pairing_mode(PdPairingMode::Off);
        let lenient = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let refused = |policies: &PolicyRegistry, path: &str| {
            let failure = pair_from(&registry, policies)
                .err()
                .unwrap_or_else(|| panic!("{path}: no pair shares a runtime"));
            assert_eq!(failure.role, WorkerRole::Decode, "{path}");
            match failure.verdict {
                PlacementFailure::NoCompatiblePair {
                    prefill,
                    decode,
                    mismatches,
                } => {
                    assert_eq!(prefill, ["vllm/?/?"], "{path}");
                    assert_eq!(decode, ["sglang/?/?"], "{path}");
                    assert_eq!(mismatches, ["runtime"], "{path}");
                }
                _ => panic!("{path}: expected NoCompatiblePair"),
            }
        };
        refused(&lenient, "lenient");
        refused(&off, "off, the open path");

        let prefill = registry.get_by_url("grpc://p:1").expect("registered");
        registry.set_worker_overloaded(&prefill, true);
        refused(&off, "off, the prefill over the thresholds");
    }

    /// The HTTP wire forwards one body to both sides, so no pair on it has
    /// to share a runtime: with pairing off a prefill pairs with a decode of
    /// another runtime, and keeps its turn. The selection reads the wire's
    /// fact from the index.
    #[test]
    fn the_http_wire_does_not_narrow_its_pairs_to_one_runtime() {
        let registry = registry_of(&[
            (
                "http://p:vllm",
                WorkerType::Prefill,
                ConnectionMode::Http,
                RuntimeType::Vllm,
            ),
            (
                "http://p:sglang",
                WorkerType::Prefill,
                ConnectionMode::Http,
                RuntimeType::Sglang,
            ),
            (
                "http://d:vllm",
                WorkerType::Decode,
                ConnectionMode::Http,
                RuntimeType::Vllm,
            ),
        ]);
        let policies = cohort_policies().with_pd_pairing_mode(PdPairingMode::Off);
        let pairs = registry
            .get_routing_snapshot(MODEL)
            .pd_pairs(PdWire::Http, policies.pd_pairing_mode());
        let mut prefills: Vec<String> = (0..4)
            .map(|_| {
                select_pair(
                    &registry,
                    &policies,
                    MODEL,
                    &pairs,
                    None,
                    None,
                    PlacementInputs::default(),
                    Stated::nothing,
                )
                .map(|pair| pair.prefill.url().to_string())
                .unwrap_or_default()
            })
            .collect();
        prefills.sort();
        prefills.dedup();
        assert_eq!(
            prefills,
            ["http://p:sglang", "http://p:vllm"],
            "both prefills take a turn"
        );
    }

    /// With pairing off a prefill of a runtime-specific wire is still listed
    /// only the decodes of its runtime, and every placement reads that list:
    /// the open path, and the least-loaded default once the prefill is over
    /// the thresholds, though the decode on the other runtime is the idle
    /// one. With its one partner not ready the request has no pair on
    /// either path; the other runtime's decode is never handed out. Every
    /// ordered pair of the wire's runtimes is a row.
    #[test]
    fn pairing_off_never_pairs_two_known_runtimes_on_either_path() {
        let runtimes = [
            RuntimeType::Vllm,
            RuntimeType::Sglang,
            RuntimeType::TokenSpeed,
            RuntimeType::Trtllm,
            RuntimeType::Mlx,
        ];
        for own in runtimes {
            for other in runtimes.into_iter().filter(|other| *other != own) {
                let registry = registry_of(&[
                    (
                        "grpc://p:own",
                        WorkerType::Prefill,
                        ConnectionMode::Grpc,
                        own,
                    ),
                    (
                        "grpc://d:own",
                        WorkerType::Decode,
                        ConnectionMode::Grpc,
                        own,
                    ),
                    (
                        "grpc://d:other",
                        WorkerType::Decode,
                        ConnectionMode::Grpc,
                        other,
                    ),
                ]);
                let off = PolicyRegistry::new(PolicyConfig::RoundRobin)
                    .with_pd_pairing_mode(PdPairingMode::Off);
                let decode_of = |path: &str| {
                    let pair = pair_from(&registry, &off)
                        .unwrap_or_else(|_| panic!("{path} pairs ({own:?}, {other:?})"));
                    assert_eq!(pair.runtime, own, "{path} ({own:?}, {other:?})");
                    pair.decode.url().to_string()
                };
                for _ in 0..4 {
                    assert_eq!(decode_of("the open path"), "grpc://d:own");
                }

                let prefill = registry.get_by_url("grpc://p:own").expect("registered");
                registry.set_worker_overloaded(&prefill, true);
                let partner = registry.get_by_url("grpc://d:own").expect("registered");
                for _ in 0..10 {
                    partner.increment_load();
                }
                for _ in 0..4 {
                    assert_eq!(decode_of("the default"), "grpc://d:own");
                }

                partner.set_status(WorkerStatus::NotReady);
                assert!(
                    pair_from(&registry, &off).is_err(),
                    "the default has no pair ({own:?}, {other:?})"
                );
                registry.set_worker_overloaded(&prefill, false);
                assert!(
                    pair_from(&registry, &off).is_err(),
                    "the open path has no pair ({own:?}, {other:?})"
                );
            }
        }
    }

    /// A pair over a pool whose only decode is not ready is the decode
    /// role's unavailability.
    #[test]
    fn a_pair_whose_only_decode_is_not_ready_is_the_decode_roles_unavailability() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let decode = registry.get_by_url("grpc://d:1").expect("registered");
        decode.set_status(WorkerStatus::NotReady);
        let failure = pair_from(&registry, &cohort_policies())
            .err()
            .expect("no partner is open");
        assert_eq!(failure.role, WorkerRole::Decode);
        assert!(matches!(failure.verdict, PlacementFailure::Unavailable));
    }

    #[test]
    fn an_unknown_transport_pairs_leniently_and_fails_strictly() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, None),
        ]);

        let lenient = PolicyRegistry::new(PolicyConfig::RoundRobin);
        assert!(pair_from(&registry, &lenient).is_ok());

        let strict = PolicyRegistry::new(PolicyConfig::RoundRobin)
            .with_pd_pairing_mode(PdPairingMode::Strict);
        let failure = pair_from(&registry, &strict)
            .err()
            .expect("strict mode refuses an unknown transport");
        assert!(matches!(
            failure.verdict,
            PlacementFailure::NoCompatiblePair { .. }
        ));
    }

    /// Beside the default settings, whose pairing mode is lenient, the
    /// registry of a gateway that runs a supplied document reads its pair
    /// index by that document's pairing. Under a document that says
    /// `strict`, the pair of an unknown transport, which the lenient rule
    /// forms, is not formed.
    #[test]
    fn a_supplied_documents_strict_pairing_forms_no_pair_of_an_unknown_transport() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, None),
        ]);
        let can_pair = |pairing: smg_policy_controller::document::Pairing| {
            let mut document =
                PolicySpec::pair_presets("round_robin", "round_robin").expect("paired presets");
            document.plans.pd_pair.pairing = pairing;
            let settings = crate::config::RouterConfig {
                mode: crate::config::RoutingMode::PrefillDecode {
                    prefill_urls: Vec::new(),
                    decode_urls: Vec::new(),
                    prefill_policy: None,
                    decode_policy: None,
                },
                policy_controller: true,
                policy_spec: Some(document),
                ..Default::default()
            };
            pairs_of(&registry, &PolicyRegistry::of_config(&settings)).can_pair()
        };
        assert!(can_pair(smg_policy_controller::document::Pairing::Lenient));
        assert!(!can_pair(smg_policy_controller::document::Pairing::Strict));
    }

    fn registry_with(workers: &[(&str, ConnectionMode, RuntimeType)]) -> WorkerRegistry {
        let typed: Vec<_> = workers
            .iter()
            .map(|(url, connection, runtime)| (*url, WorkerType::Regular, *connection, *runtime))
            .collect();
        registry_of(&typed)
    }

    fn registry_of(workers: &[(&str, WorkerType, ConnectionMode, RuntimeType)]) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        for (url, worker_type, connection, runtime) in workers {
            registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .model(ModelCard::new(MODEL))
                        .worker_type(*worker_type)
                        .connection_mode(*connection)
                        .runtime_type(*runtime)
                        .health_config(HealthCheckConfig {
                            disable_health_check: true,
                            ..Default::default()
                        })
                        .build(),
                ))
                .expect("worker registers");
        }
        registry
    }

    fn urls(candidates: &Candidates) -> Vec<String> {
        let mut urls: Vec<String> = candidates
            .as_slice()
            .iter()
            .map(|w| w.url().to_string())
            .collect();
        urls.sort();
        urls
    }

    #[test]
    fn each_pool_sees_only_its_own_transport() {
        let registry = registry_with(&[
            ("http://h:1", ConnectionMode::Http, RuntimeType::Sglang),
            ("grpc://g:1", ConnectionMode::Grpc, RuntimeType::Sglang),
            ("zmq://z:1", ConnectionMode::Zmq, RuntimeType::Vllm),
        ]);

        assert_eq!(
            urls(&candidates(
                &registry,
                MODEL,
                RoutingPool::HttpRegular,
                None
            )),
            ["http://h:1"]
        );
        // The gRPC pipeline serves both gRPC and direct-ZMQ workers.
        assert_eq!(
            urls(&candidates(
                &registry,
                MODEL,
                RoutingPool::GrpcPipelineRegular,
                None
            )),
            ["grpc://g:1", "zmq://z:1"]
        );
    }

    #[test]
    fn a_pinned_wire_narrows_to_its_runtime_and_transport() {
        let registry = registry_with(&[
            ("grpc://g:1", ConnectionMode::Grpc, RuntimeType::Sglang),
            ("grpc://g:2", ConnectionMode::Grpc, RuntimeType::Vllm),
            ("zmq://z:1", ConnectionMode::Zmq, RuntimeType::Vllm),
        ]);
        let wire = Some(WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: ConnectionMode::Grpc,
            requires_media_refs: false,
            pipeline: PipelinePin::Any,
        });

        assert_eq!(
            urls(&candidates(
                &registry,
                MODEL,
                RoutingPool::GrpcPipelineRegular,
                wire
            )),
            ["grpc://g:2"]
        );
    }

    #[test]
    fn a_pair_shares_the_first_prefill_runtime_and_names_a_missing_role() {
        let registry = registry_of(&[
            (
                "grpc://p:1",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
            (
                "grpc://p:2",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
            (
                "grpc://d:1",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Vllm,
            ),
            (
                "grpc://d:2",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);

        let pair = pair_from(&registry, &policies).expect("a pair exists under either runtime");
        assert_eq!(pair.prefill.metadata().spec.runtime_type, pair.runtime);
        assert_eq!(pair.decode.metadata().spec.runtime_type, pair.runtime);

        // A model with no decode worker at all names the role.
        let only_prefill = registry_of(&[(
            "grpc://p:1",
            WorkerType::Prefill,
            ConnectionMode::Grpc,
            RuntimeType::Sglang,
        )]);
        let failure = pair_from(&only_prefill, &policies)
            .err()
            .expect("no decode worker exists");
        assert_eq!(failure.role, WorkerRole::Decode);
        assert!(matches!(failure.verdict, PlacementFailure::NoCandidates));
    }

    #[test]
    fn a_pinned_pair_keeps_the_retained_runtime_and_transport() {
        // Same runtime on both pools, but the only decode worker speaks ZMQ.
        let registry = registry_of(&[
            (
                "grpc://p:1",
                WorkerType::Prefill,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
            (
                "zmq://d:1",
                WorkerType::Decode,
                ConnectionMode::Zmq,
                RuntimeType::Sglang,
            ),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let by_type = |worker_type: WorkerType| -> Vec<Arc<dyn Worker>> {
            registry
                .get_all()
                .into_iter()
                .filter(|w| *w.worker_type() == worker_type)
                .collect()
        };
        // Built from raw pools, as a caller with its own candidate lists would.
        let pairs = PdPairIndex::build(
            by_type(WorkerType::Prefill).into(),
            by_type(WorkerType::Decode).into(),
            PdWire::Http,
            policies.pd_pairing_mode(),
        );

        // Unpinned, the ZMQ decode worker is a candidate like any other.
        assert!(select_pair(
            &registry,
            &policies,
            MODEL,
            &pairs,
            None,
            None,
            PlacementInputs::default(),
            Stated::nothing,
        )
        .is_ok());

        // A retry that retained a gRPC plan must not land on it.
        let failure = select_pair(
            &registry,
            &policies,
            MODEL,
            &pairs,
            Some(WireConstraint {
                runtime: RuntimeType::Sglang,
                connection: ConnectionMode::Grpc,
                requires_media_refs: false,
                pipeline: PipelinePin::Any,
            }),
            None,
            PlacementInputs::default(),
            Stated::nothing,
        )
        .err()
        .expect("the retained transport has no decode worker");
        assert_eq!(failure.role, WorkerRole::Decode);
        assert!(matches!(failure.verdict, PlacementFailure::Unavailable));
    }

    #[test]
    fn a_caller_can_narrow_the_candidates_itself() {
        let registry = registry_with(&[
            ("http://h:1", ConnectionMode::Http, RuntimeType::Sglang),
            ("http://h:2", ConnectionMode::Http, RuntimeType::Sglang),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let pool = registry.get_routing_pool(MODEL, RoutingPool::HttpRegular);
        let only_second: Vec<Arc<dyn Worker>> = pool
            .iter()
            .filter(|w| w.url() == "http://h:2")
            .cloned()
            .collect();

        let selected = select_from(
            &registry,
            &policies,
            MODEL,
            RoutingPool::HttpRegular,
            &only_second,
            PlacementInputs::default(),
            Stated::nothing,
        )
        .expect("the narrowed slice still has a worker");
        assert_eq!(selected.url(), "http://h:2");
        assert!(matches!(
            failure_from(&[], MODEL, false),
            PlacementFailure::NoCandidates
        ));
    }

    /// A caller's filter narrows the veto fallback and the verdict too: a
    /// decode pool whose only admitted worker is over the thresholds is
    /// steered to it by default and shed under `--worker-overload-shed`,
    /// instead of being refused as unavailable because a worker the request
    /// may not use is idle.
    #[test]
    fn a_filtered_decode_pool_is_judged_over_the_workers_the_request_may_use() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let d1 = registry.get_by_url("grpc://d:1").expect("registered");
        let not_d2 = |w: &dyn Worker| w.url() != "grpc://d:2";
        let inputs = PlacementInputs {
            candidate_filter: Some(&not_d2),
            ..Default::default()
        };
        let place = |inputs: PlacementInputs<'_>| {
            select_pair(
                &registry,
                &policies,
                MODEL,
                &pairs_of(&registry, &policies),
                None,
                None,
                inputs,
                Stated::nothing,
            )
        };

        let pair = place(inputs).expect("the admitted decode worker is open");
        assert_eq!(pair.decode.url(), "grpc://d:1");

        registry.set_worker_overloaded(&d1, true);
        let pair = place(inputs).expect("the only admitted decode worker is vetoed: steered");
        assert_eq!(pair.decode.url(), "grpc://d:1");
        let pair = place(PlacementInputs::default())
            .expect("the idle worker takes a request without the filter");
        assert_eq!(pair.decode.url(), "grpc://d:2");

        registry.set_overload_shed(true);
        let failure = place(inputs)
            .err()
            .expect("under shedding the admitted pool is all-overloaded");
        assert_eq!(failure.role, WorkerRole::Decode);
        assert!(
            matches!(failure.verdict, PlacementFailure::AllOverloaded(_)),
            "{:?}",
            failure.verdict
        );
        let pair = place(PlacementInputs::default())
            .expect("the idle worker still takes a request without the filter");
        assert_eq!(pair.decode.url(), "grpc://d:2");
    }

    /// The prefill pool's mirror of the case above.
    #[test]
    fn a_filtered_prefill_pool_is_judged_over_the_workers_the_request_may_use() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let p1 = registry.get_by_url("grpc://p:1").expect("registered");
        let not_p2 = |w: &dyn Worker| w.url() != "grpc://p:2";
        let inputs = PlacementInputs {
            candidate_filter: Some(&not_p2),
            ..Default::default()
        };
        let place = |inputs: PlacementInputs<'_>| {
            select_pair(
                &registry,
                &policies,
                MODEL,
                &pairs_of(&registry, &policies),
                None,
                None,
                inputs,
                Stated::nothing,
            )
        };

        registry.set_worker_overloaded(&p1, true);
        let pair = place(inputs).expect("the only admitted prefill worker is vetoed: steered");
        assert_eq!(pair.prefill.url(), "grpc://p:1");
        let pair = place(PlacementInputs::default())
            .expect("the idle prefill worker takes a request without the filter");
        assert_eq!(pair.prefill.url(), "grpc://p:2");

        registry.set_overload_shed(true);
        let failure = place(inputs)
            .err()
            .expect("under shedding the admitted pool is all-overloaded");
        assert_eq!(failure.role, WorkerRole::Prefill);
        assert!(
            matches!(failure.verdict, PlacementFailure::AllOverloaded(_)),
            "{:?}",
            failure.verdict
        );
    }

    /// A retry whose pinned wire no prefill worker is on keeps the pool's own
    /// verdict (unavailable: nothing the retry can use), whatever the filter
    /// would keep of the other wire's workers: an overloaded worker the retry
    /// cannot use is no reason to shed.
    #[test]
    fn a_pool_with_no_worker_on_the_pinned_wire_keeps_its_verdict_under_a_filter() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let p1 = registry.get_by_url("grpc://p:1").expect("registered");
        registry.set_worker_overloaded(&p1, true);
        registry.set_overload_shed(true);
        // The retry's plan was built for a runtime none of the workers run.
        let other_wire = Some(WireConstraint {
            runtime: RuntimeType::Sglang,
            connection: ConnectionMode::Grpc,
            requires_media_refs: false,
            pipeline: PipelinePin::Any,
        });
        let only_p1 = |w: &dyn Worker| w.url() == "grpc://p:1";
        let failure = select_pair(
            &registry,
            &policies,
            MODEL,
            &pairs_of(&registry, &policies),
            other_wire,
            None,
            PlacementInputs {
                candidate_filter: Some(&only_p1),
                ..Default::default()
            },
            Stated::nothing,
        )
        .err()
        .expect("no worker on the wire: the pair fails");
        assert_eq!(failure.role, WorkerRole::Prefill);
        assert!(
            matches!(failure.verdict, PlacementFailure::Unavailable),
            "the verdict is the pool's own, not a shed over the other wire's overloaded worker: {:?}",
            failure.verdict
        );
    }

    /// The single-worker placement already judges its fallback over the
    /// candidates the filter admits: the only admitted worker, vetoed, is
    /// taken by default.
    #[test]
    fn a_filtered_single_placement_steers_to_its_only_admitted_worker() {
        let registry = registry_with(&[
            ("http://h:1", ConnectionMode::Http, RuntimeType::Sglang),
            ("http://h:2", ConnectionMode::Http, RuntimeType::Sglang),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let h1 = registry.get_by_url("http://h:1").expect("registered");
        registry.set_worker_overloaded(&h1, true);
        let only_h1 = |w: &dyn Worker| w.url() == "http://h:1";
        let selected = select_single(
            &registry,
            &policies,
            MODEL,
            RoutingPool::HttpRegular,
            None,
            PlacementInputs {
                candidate_filter: Some(&only_h1),
                ..Default::default()
            },
            Stated::nothing,
        )
        .expect("steered to the admitted worker");
        assert_eq!(selected.url(), "http://h:1");
    }

    /// A worker whose overload veto lifts right after selection has read its
    /// routing state a set number of times, as a load record that lands
    /// between the pass's read and the veto fallback's read does. Everything
    /// else is the real worker's.
    #[derive(Debug)]
    struct VetoLiftsAfterARead {
        worker: BasicWorker,
        /// The reads left before the veto lifts.
        reads_left: Mutex<usize>,
    }

    impl VetoLiftsAfterARead {
        /// A ready worker under the overload veto, which lifts behind the
        /// first read.
        fn vetoed(url: &str) -> Arc<dyn Worker> {
            let worker = BasicWorkerBuilder::new(url)
                .model(ModelCard::new(MODEL))
                .worker_type(WorkerType::Regular)
                .build();
            Self::under_veto(worker, 1)
        }

        /// `worker`, ready and under the overload veto, which lifts behind
        /// its `reads`-th read.
        fn under_veto(worker: BasicWorker, reads: usize) -> Arc<dyn Worker> {
            worker.set_status(WorkerStatus::Ready);
            worker.set_overloaded(true);
            Arc::new(Self {
                worker,
                reads_left: Mutex::new(reads),
            })
        }

        /// What `read` sees of the worker; the veto lifts behind the last
        /// read it has left.
        fn read<T>(&self, read: impl FnOnce(&BasicWorker) -> T) -> T {
            let seen = read(&self.worker);
            let mut left = self
                .reads_left
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if *left == 1 {
                self.worker.set_overloaded(false);
            }
            *left = left.saturating_sub(1);
            seen
        }
    }

    #[async_trait::async_trait]
    impl Worker for VetoLiftsAfterARead {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn url(&self) -> &str {
            self.worker.url()
        }
        fn api_key(&self) -> Option<&String> {
            self.worker.api_key()
        }
        fn worker_type(&self) -> &WorkerType {
            self.worker.worker_type()
        }
        fn connection_mode(&self) -> &ConnectionMode {
            self.worker.connection_mode()
        }
        fn status(&self) -> WorkerStatus {
            self.worker.status()
        }
        fn set_status(&self, status: WorkerStatus) {
            self.worker.set_status(status);
        }
        fn load(&self) -> usize {
            self.worker.load()
        }
        fn increment_load(&self) {
            self.worker.increment_load();
        }
        fn decrement_load(&self) {
            self.worker.decrement_load();
        }
        fn routing_key_load(&self) -> usize {
            self.worker.routing_key_load()
        }
        fn routing_key_inflight(&self, routing_key: &str) -> usize {
            self.worker.routing_key_inflight(routing_key)
        }
        fn increment_routing_key_load(&self, routing_key: &str) {
            self.worker.increment_routing_key_load(routing_key);
        }
        fn decrement_routing_key_load(&self, routing_key: &str) {
            self.worker.decrement_routing_key_load(routing_key);
        }
        fn processed_requests(&self) -> usize {
            self.worker.processed_requests()
        }
        fn increment_processed(&self) {
            self.worker.increment_processed();
        }
        fn metadata(&self) -> &WorkerMetadata {
            self.worker.metadata()
        }
        fn circuit_breaker_state(&self) -> CircuitState {
            self.worker.circuit_breaker_state()
        }
        fn circuit_breaker_can_execute(&self) -> bool {
            self.worker.circuit_breaker_can_execute()
        }
        fn record_circuit_breaker_outcome(&self, success: bool) {
            self.worker.record_circuit_breaker_outcome(success);
        }
        fn resilience(&self) -> &ResolvedResilience {
            self.worker.resilience()
        }
        fn http_client(&self) -> &reqwest::Client {
            self.worker.http_client()
        }
        fn http_client_handle_if_initialized(&self) -> Option<Arc<reqwest::Client>> {
            self.worker.http_client_handle_if_initialized()
        }
        async fn check_health_async(&self) -> WorkerResult<()> {
            self.worker.check_health_async().await
        }
        async fn engine_connection(&self) -> WorkerResult<Option<Arc<dyn EngineConnection>>> {
            self.worker.engine_connection().await
        }
        fn is_available(&self) -> bool {
            self.read(|worker| worker.is_available())
        }
        fn routing_state(&self) -> RoutingState {
            self.read(|worker| worker.routing_state())
        }
        fn is_overloaded(&self) -> bool {
            self.read(|worker| worker.is_overloaded())
        }
        fn stall_reason(&self) -> Option<StallReason> {
            self.worker.stall_reason()
        }
    }

    /// An overload veto that lifts between the pass's read and the veto
    /// fallback's read is no refusal. The pass saw its only worker vetoed,
    /// the fallback then finds a pool that is not all vetoed, and one more
    /// pass takes the worker that came free. Two rows: a policy that filters
    /// availability itself, as the default policy does, and a policy the
    /// placement filters the candidates for.
    #[test]
    fn a_veto_that_lifts_between_the_two_reads_is_no_refusal() {
        let registry = WorkerRegistry::new();
        let mut refused = Vec::new();
        for config in [PolicyConfig::RoundRobin, PolicyConfig::ConsistentHashing] {
            let policies = PolicyRegistry::new(config);
            // One pass alone must find nobody available: were the veto to
            // lift before the pass's own read, the test would pass without
            // the gap between that read and the fallback's.
            let alone = [VetoLiftsAfterARead::vetoed("http://w:1")];
            assert_eq!(
                select_pass(
                    &registry,
                    &policies,
                    MODEL,
                    &alone,
                    None,
                    PlacementInputs::default()
                )
                .err(),
                Some(Miss::NobodyAvailable),
                "one pass alone, under {}",
                policies.get_policy_or_default(MODEL).name()
            );
            let candidates = [VetoLiftsAfterARead::vetoed("http://w:1")];
            let selected = select_from(
                &registry,
                &policies,
                MODEL,
                RoutingPool::HttpRegular,
                &candidates,
                PlacementInputs::default(),
                Stated::nothing,
            );
            if selected.is_none() {
                refused.push(policies.get_policy_or_default(MODEL).name());
            }
        }
        assert!(
            refused.is_empty(),
            "refused while its only worker was available, under: {refused:?}"
        );
    }

    /// The pair placement has the same two reads, the pass's and the veto
    /// default's, and a veto that lifts between them is no refusal there
    /// either. Two rows, the prefill side and the decode side: a pool of one
    /// worker under the veto beside a free partner. The pairs are on the
    /// HTTP wire, where the veto lifts behind the worker's third read: two
    /// reads judge its pool's lift, the third is the pass's availability
    /// read. The fourth sees the veto gone: the default's read of the pool
    /// on the prefill side, the vetoed-alone read of the same eligibility
    /// test on the decode side, whose pool the first two reads had lifted.
    #[test]
    fn a_pair_veto_that_lifts_between_the_two_reads_is_no_refusal() {
        let registry = WorkerRegistry::new();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let mut refused = Vec::new();
        for vetoed in [WorkerType::Prefill, WorkerType::Decode] {
            // A fresh pair for every selection: the veto lifts once.
            let pair = || {
                let side = |role: WorkerType, url: &str| -> Arc<[Arc<dyn Worker>]> {
                    let worker = pd_worker(url, role, Some("NixlConnector"));
                    if role == vetoed {
                        Arc::from([VetoLiftsAfterARead::under_veto(worker, 3)])
                    } else {
                        worker.set_status(WorkerStatus::Ready);
                        Arc::from([Arc::new(worker) as Arc<dyn Worker>])
                    }
                };
                PdPairIndex::build(
                    side(WorkerType::Prefill, "grpc://p:1"),
                    side(WorkerType::Decode, "grpc://d:1"),
                    PdWire::Http,
                    policies.pd_pairing_mode(),
                )
            };
            // One pass alone must find nobody open on that side: were the
            // veto to lift before the pass's last read of the worker, the
            // test would pass without the gap between that read and the
            // default's.
            assert!(
                matches!(
                    pair_pass(
                        &registry,
                        &policies,
                        MODEL,
                        &pair(),
                        None,
                        None,
                        PlacementInputs::default(),
                    ),
                    Err(PairMiss::NobodyOpen(_))
                ),
                "one pass alone, the {vetoed:?} side"
            );
            let placed = select_pair(
                &registry,
                &policies,
                MODEL,
                &pair(),
                None,
                None,
                PlacementInputs::default(),
                Stated::nothing,
            );
            if placed.is_err() {
                refused.push(vetoed);
            }
        }
        assert!(
            refused.is_empty(),
            "refused while the side's only worker was available: {refused:?}"
        );
    }

    /// Building an overload-shed verdict records the shed, so that verdict
    /// must stand even if the load flag clears immediately afterwards. Both
    /// sides cover the gap between the verdict's read and the retry check.
    #[test]
    fn an_overload_shed_is_final_for_pair_placement() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let registry = WorkerRegistry::new();
        registry.set_overload_shed(true);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let mut served = Vec::new();
        metrics::with_local_recorder(&recorder, || {
            for vetoed in [WorkerType::Prefill, WorkerType::Decode] {
                let side = |role: WorkerType, url: &str| -> Arc<[Arc<dyn Worker>]> {
                    let worker = pd_worker(url, role, Some("NixlConnector"));
                    if role == vetoed {
                        // The pass and fallback see the veto. It clears just
                        // after failure classification records the shed.
                        Arc::from([VetoLiftsAfterARead::under_veto(worker, 4)])
                    } else {
                        worker.set_status(WorkerStatus::Ready);
                        Arc::from([Arc::new(worker) as Arc<dyn Worker>])
                    }
                };
                let pairs = PdPairIndex::build(
                    side(WorkerType::Prefill, "grpc://p:1"),
                    side(WorkerType::Decode, "grpc://d:1"),
                    PdWire::Http,
                    policies.pd_pairing_mode(),
                );
                match select_pair(
                    &registry,
                    &policies,
                    MODEL,
                    &pairs,
                    None,
                    None,
                    PlacementInputs::default(),
                    Stated::nothing,
                ) {
                    Err(failure)
                        if matches!(failure.verdict, PlacementFailure::AllOverloaded(_)) => {}
                    Err(failure) => panic!("unexpected failure: {failure:?}"),
                    Ok(_) => served.push(vetoed),
                }
                let vetoed_worker = if vetoed == WorkerType::Prefill {
                    &pairs.prefill[0]
                } else {
                    &pairs.decode_pool[0]
                };
                assert!(
                    !vetoed_worker.is_overloaded(),
                    "the veto lifted inside the placement, the {vetoed:?} side"
                );
            }
        });
        assert!(
            served.is_empty(),
            "served after recording a shed: {served:?}"
        );
        assert!(
            handle.render().lines().any(|line| {
                line.starts_with("smg_worker_overload_shed_total{")
                    && line.contains("stage=\"selection\"")
                    && line.ends_with(" 2")
            }),
            "each refused request is counted once:\n{}",
            handle.render()
        );
    }

    #[test]
    fn selection_and_failure_read_the_same_pool() {
        let registry = registry_with(&[("http://h:1", ConnectionMode::Http, RuntimeType::Sglang)]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);

        let selected = select_single(
            &registry,
            &policies,
            MODEL,
            RoutingPool::HttpRegular,
            None,
            PlacementInputs::default(),
            Stated::nothing,
        )
        .expect("the HTTP worker is selectable from its own pool");
        assert_eq!(selected.url(), "http://h:1");

        // The gRPC pool holds nothing for this model, so selection fails and
        // the verdict says why.
        assert!(select_single(
            &registry,
            &policies,
            MODEL,
            RoutingPool::GrpcPipelineRegular,
            None,
            PlacementInputs::default(),
            Stated::nothing,
        )
        .is_none());
        assert!(matches!(
            single_failure(&registry, MODEL, RoutingPool::GrpcPipelineRegular, None),
            PlacementFailure::NoCandidates
        ));
    }

    fn cache_aware_policy() -> PolicyConfig {
        PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096)
    }

    fn http_registry(urls: &[&str]) -> WorkerRegistry {
        let typed: Vec<_> = urls
            .iter()
            .map(|url| (*url, ConnectionMode::Http, RuntimeType::Sglang))
            .collect();
        registry_with(&typed)
    }

    fn pick(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        inputs: PlacementInputs<'_>,
    ) -> Arc<dyn Worker> {
        answered(registry, policies, inputs).into_worker()
    }

    /// [`pick`] as the placement answers it: the worker with what its load
    /// guard takes from the answer.
    fn answered(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        inputs: PlacementInputs<'_>,
    ) -> Placed {
        select_single(
            registry,
            policies,
            MODEL,
            RoutingPool::HttpRegular,
            None,
            inputs,
            Stated::nothing,
        )
        .expect("a worker is selectable")
    }

    /// A routing key pins consistent hashing to one worker; once that worker
    /// was tried, the retry falls through to the next node on the ring, and
    /// a pool with nothing untried left is used whole.
    #[test]
    fn a_retry_under_consistent_hashing_falls_through_to_the_next_ring_node() {
        let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
        let policies = PolicyRegistry::new(PolicyConfig::ConsistentHashing);
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "session-7".parse().unwrap());
        let inputs = PlacementInputs {
            headers: Some(&headers),
            ..Default::default()
        };

        let owner = pick(&registry, &policies, inputs);
        assert_eq!(
            pick(&registry, &policies, inputs).url(),
            owner.url(),
            "the key re-selects its owner deterministically"
        );

        let tried = vec![owner.base_url().to_string()];
        let moved = pick(
            &registry,
            &policies,
            PlacementInputs {
                tried: &tried,
                ..inputs
            },
        );
        assert_ne!(
            moved.url(),
            owner.url(),
            "the retry leaves the tried worker"
        );
        assert_eq!(
            pick(
                &registry,
                &policies,
                PlacementInputs {
                    tried: &tried,
                    ..inputs
                },
            )
            .url(),
            moved.url(),
            "the fall-through is deterministic too"
        );

        let all: Vec<String> = ["http://h:1", "http://h:2", "http://h:3"]
            .iter()
            .map(|url| (*url).to_string())
            .collect();
        assert_eq!(
            pick(
                &registry,
                &policies,
                PlacementInputs {
                    tried: &all,
                    ..inputs
                },
            )
            .url(),
            owner.url(),
            "with every worker tried the pool is used whole again"
        );
    }

    /// A policy that places by the model's hash ring is handed it. Under
    /// each of the two a keyed request lands where the policy itself puts it
    /// with the registry's ring in hand, and some key lands elsewhere
    /// without the ring (consistent hashing then counts the key over the
    /// list, prefix hash takes the least loaded worker): a placement that
    /// withheld the ring would move that key.
    #[test]
    fn a_policy_that_reads_the_hash_ring_is_handed_it() {
        let urls: Vec<String> = (1..=8).map(|n| format!("http://h:{n}")).collect();
        let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
        let registry = http_registry(&urls);
        let ring = PolicyRegistry::new(PolicyConfig::ConsistentHashing)
            .hash_ring(&registry, MODEL)
            .expect("the model's ring");
        let pool = candidates(&registry, MODEL, RoutingPool::HttpRegular, None);
        let pool = pool.as_slice();
        let prefix_hash = PolicyConfig::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: 10,
            cache_boundaries: Vec::new(),
        };
        for config in [PolicyConfig::ConsistentHashing, prefix_hash] {
            let policies = PolicyRegistry::new(config);
            let policy = policies.get_policy_or_default(MODEL);
            let mut moved_without_the_ring = 0;
            for session in 0..32 {
                let mut headers = HeaderMap::new();
                headers.insert(
                    "x-smg-routing-key",
                    format!("session-{session}").parse().unwrap(),
                );
                let own_pick = |hash_ring: Option<Arc<HashRing>>| {
                    let info = SelectWorkerInfo {
                        headers: Some(&headers),
                        routing_key: policies.sticky_header_key(Some(&headers)),
                        hash_ring,
                        ..Default::default()
                    };
                    policy
                        .select_worker(pool, &info)
                        .map(|idx| pool[idx].url().to_string())
                };
                let with_the_ring = own_pick(Some(Arc::clone(&ring)));
                let placed = pick(
                    &registry,
                    &policies,
                    PlacementInputs {
                        headers: Some(&headers),
                        ..Default::default()
                    },
                );
                assert_eq!(
                    Some(placed.url().to_string()),
                    with_the_ring,
                    "{} places session {session} by the ring",
                    policy.name()
                );
                moved_without_the_ring += usize::from(own_pick(None) != with_the_ring);
            }
            assert!(
                moved_without_the_ring > 0,
                "{}: no key of the 32 tells a withheld ring",
                policy.name()
            );
        }
    }

    /// Cache-aware routing pins a prompt to the worker holding its prefix;
    /// a retry moves to the next-best worker and comes back only when no
    /// other worker is left.
    #[test]
    fn a_retry_under_cache_aware_routing_moves_to_the_next_best_worker() {
        let registry = http_registry(&["http://h:1", "http://h:2"]);
        let policies = PolicyRegistry::new(cache_aware_policy());
        let text = "the one prompt whose prefix lives on a single worker after the first request";
        let inputs = PlacementInputs {
            text: Some(text),
            ..Default::default()
        };

        let holder = pick(&registry, &policies, inputs);
        assert_eq!(
            pick(&registry, &policies, inputs).url(),
            holder.url(),
            "the prefix pins the prompt to its holder"
        );

        let tried = vec![holder.base_url().to_string()];
        let moved = pick(
            &registry,
            &policies,
            PlacementInputs {
                tried: &tried,
                ..inputs
            },
        );
        assert_ne!(moved.url(), holder.url(), "the retry leaves the holder");

        let both = vec![holder.base_url().to_string(), moved.base_url().to_string()];
        assert!(select_single(
            &registry,
            &policies,
            MODEL,
            RoutingPool::HttpRegular,
            None,
            PlacementInputs {
                tried: &both,
                ..inputs
            },
            Stated::nothing,
        )
        .is_some());
    }

    /// The holder of a prompt's prefix keeps it over repeated placements, and
    /// a retry leaves the holder.
    #[test]
    fn a_cache_aware_prefix_keeps_its_holder() {
        let registry = http_registry(&["http://h:1", "http://h:2"]);
        let policies = PolicyRegistry::new(cache_aware_policy());
        let text = "a prompt long enough to fill a few blocks of the tree on its first request";
        let inputs = PlacementInputs {
            text: Some(text),
            ..Default::default()
        };
        let holder = pick(&registry, &policies, inputs);
        for _ in 0..8 {
            assert_eq!(
                pick(&registry, &policies, inputs).url(),
                holder.url(),
                "the prefix pins the prompt to its holder"
            );
        }
        let tried = vec![holder.base_url().to_string()];
        let moved = pick(
            &registry,
            &policies,
            PlacementInputs {
                tried: &tried,
                ..inputs
            },
        );
        assert_ne!(moved.url(), holder.url(), "the retry leaves the holder");
    }

    /// `select_from` under round robin rotates the pool it is given: three
    /// candidates, nine picks, three each.
    #[test]
    fn select_from_rotates_the_pool_it_is_given() {
        let registry = registry_with(&[
            ("grpc://r:1", ConnectionMode::Grpc, RuntimeType::Vllm),
            ("grpc://r:2", ConnectionMode::Grpc, RuntimeType::Vllm),
            ("grpc://r:3", ConnectionMode::Grpc, RuntimeType::Vllm),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let candidates: Vec<Arc<dyn Worker>> = ["grpc://r:1", "grpc://r:2", "grpc://r:3"]
            .iter()
            .map(|url| registry.get_by_url(url).expect("registered"))
            .collect();
        let mut picks = BTreeMap::new();
        for _ in 0..9 {
            let worker = select_from(
                &registry,
                &policies,
                MODEL,
                RoutingPool::HttpRegular,
                &candidates,
                PlacementInputs::default(),
                Stated::nothing,
            )
            .expect("a worker is picked");
            *picks.entry(worker.url().to_string()).or_insert(0) += 1;
        }
        assert_eq!(picks.into_values().collect::<Vec<usize>>(), [3, 3, 3]);
    }

    /// An explicit target under consistent hashing indexes the unfiltered
    /// set and stays strict: its retry goes back to the targeted worker
    /// (never to whichever worker a narrowed slice puts at that index), even
    /// though that worker was tried.
    #[test]
    fn an_explicit_target_under_consistent_hashing_keeps_its_worker_on_retry() {
        let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
        let policies = PolicyRegistry::new(PolicyConfig::ConsistentHashing);
        for index in 0..3 {
            let mut headers = HeaderMap::new();
            headers.insert("x-smg-target-worker", index.to_string().parse().unwrap());
            let inputs = PlacementInputs {
                headers: Some(&headers),
                ..Default::default()
            };

            let target = pick(&registry, &policies, inputs);
            let tried = vec![target.base_url().to_string()];
            let retried = pick(
                &registry,
                &policies,
                PlacementInputs {
                    tried: &tried,
                    ..inputs
                },
            );
            assert_eq!(
                retried.url(),
                target.url(),
                "target {index}: the retry stays on the targeted worker"
            );
        }
    }

    /// An explicit target counts the workers available when the request
    /// arrives. With the worker at index 1 out (not ready, overloaded, or
    /// its breaker open), index 1 names the worker that was at index 2 and
    /// index 2 names nobody: the policy declines, and the selection leaves
    /// it at that one pass. Back in, the indexes read as before.
    #[test]
    fn an_explicit_target_indexes_the_available_workers() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
        let policies = PolicyRegistry::new(PolicyConfig::ConsistentHashing);
        let target = |index: usize| {
            let mut headers = HeaderMap::new();
            headers.insert("x-smg-target-worker", index.to_string().parse().unwrap());
            select_single(
                &registry,
                &policies,
                MODEL,
                RoutingPool::HttpRegular,
                None,
                PlacementInputs {
                    headers: Some(&headers),
                    ..Default::default()
                },
                Stated::nothing,
            )
        };
        let by_index: Vec<Arc<dyn Worker>> = (0..3)
            .map(|index| target(index).expect("a healthy pool answers every index"))
            .map(Placed::into_worker)
            .collect();
        assert_ne!(by_index[0].url(), by_index[1].url());
        assert_ne!(by_index[1].url(), by_index[2].url());
        assert_ne!(by_index[0].url(), by_index[2].url());

        let out = &by_index[1];
        let shifted = |veto: &str| {
            assert_eq!(
                target(1).map(|worker| worker.url().to_string()),
                Some(by_index[2].url().to_string()),
                "{veto}: index 1 names the next available worker"
            );
            assert!(
                target(2).is_none(),
                "{veto}: index 2 is past the available workers"
            );
            // That miss is the policy's decline among available workers, and
            // a decline takes one pass: the whole selection asks the filter
            // as often as one pass does.
            let asked = AtomicUsize::new(0);
            let counted = |_: &dyn Worker| {
                asked.fetch_add(1, Ordering::Relaxed);
                true
            };
            let mut headers = HeaderMap::new();
            headers.insert("x-smg-target-worker", "2".parse().unwrap());
            let inputs = PlacementInputs {
                headers: Some(&headers),
                candidate_filter: Some(&counted),
                ..Default::default()
            };
            let pool = candidates(&registry, MODEL, RoutingPool::HttpRegular, None);
            let drawn = pool.as_slice();
            let whole = pool.shared().map(|list| NamedList {
                list,
                positions: None,
            });
            assert_eq!(
                select_pass(&registry, &policies, MODEL, drawn, whole, inputs).err(),
                Some(Miss::Declined),
                "{veto}: one pass declines a target past the available workers"
            );
            let one_pass = asked.swap(0, Ordering::Relaxed);
            let selected = select_among(&registry, &policies, MODEL, drawn, whole, inputs);
            assert!(selected.is_none(), "{veto}: index 2 names nobody");
            assert_eq!(
                asked.load(Ordering::Relaxed),
                one_pass,
                "{veto}: a decline is not asked again"
            );
            assert_eq!(target(0).unwrap().url(), by_index[0].url(), "{veto}");
        };
        out.set_status(WorkerStatus::NotReady);
        shifted("not ready");
        out.set_status(WorkerStatus::Ready);
        assert!(registry.set_worker_overloaded(out, true));
        shifted("overloaded");
        assert!(registry.set_worker_overloaded(out, false));
        for _ in 0..16 {
            out.record_circuit_breaker_outcome(false);
        }
        assert!(!out.circuit_breaker_can_execute());
        shifted("breaker open");
        out.reset_circuit_breaker();
        assert!(out.circuit_breaker_can_execute());
        assert_eq!(target(1).unwrap().url(), by_index[1].url());
        assert_eq!(target(2).unwrap().url(), by_index[2].url());

        // The documented exception: a pool with every worker vetoed is served
        // by the all-vetoed fallback, which does not read the header.
        for worker in &by_index {
            assert!(registry.set_worker_overloaded(worker, true));
        }
        assert!(
            target(5).is_some(),
            "an all-vetoed pool serves an index past the pool through the fallback"
        );
        for worker in &by_index {
            assert!(registry.set_worker_overloaded(worker, false));
        }
        assert_eq!(target(1).unwrap().url(), by_index[1].url());
    }

    /// Round robin never re-selects a tried worker while another is left,
    /// and a single-worker pool keeps retrying its one worker.
    #[test]
    fn a_retry_under_round_robin_skips_the_tried_worker_and_a_lone_worker_is_kept() {
        let registry = http_registry(&["http://h:1", "http://h:2"]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let tried = vec!["http://h:1".to_string()];
        for _ in 0..4 {
            let selected = pick(
                &registry,
                &policies,
                PlacementInputs {
                    tried: &tried,
                    ..Default::default()
                },
            );
            assert_eq!(selected.url(), "http://h:2");
        }

        let lone = http_registry(&["http://h:9"]);
        let tried = vec!["http://h:9".to_string()];
        let selected = pick(
            &lone,
            &policies,
            PlacementInputs {
                tried: &tried,
                ..Default::default()
            },
        );
        assert_eq!(
            selected.url(),
            "http://h:9",
            "nothing else is left: the one worker takes the retry"
        );
    }

    fn fleet(workers: usize) -> (Arc<WorkerRegistry>, Vec<Arc<dyn Worker>>) {
        let registry = Arc::new(WorkerRegistry::new());
        let fleet: Vec<Arc<dyn Worker>> = (0..workers)
            .map(|at| {
                let worker: Arc<dyn Worker> = Arc::new(
                    BasicWorkerBuilder::new(format!("http://asked-{at:03}:8000"))
                        .model(ModelCard::new(MODEL))
                        .health_config(no_health_check())
                        .build(),
                );
                worker.set_status(WorkerStatus::Ready);
                assert!(registry.register(Arc::clone(&worker)).is_some());
                worker
            })
            .collect();
        (registry, fleet)
    }

    /// The worker one request is placed on, by its place in `among`.
    fn placed(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        among: &[Arc<dyn Worker>],
        inputs: PlacementInputs<'_>,
    ) -> Option<usize> {
        let pool = RoutingPool::HttpRegular;
        let worker = select_single(
            registry,
            policies,
            MODEL,
            pool,
            None,
            inputs,
            Stated::nothing,
        )?;
        among
            .iter()
            .position(|member| Arc::ptr_eq(member, worker.worker()))
    }

    /// Callers that pick at the same moment do not pile on one worker.
    /// Four callers pick together, a barrier a round, over 256 idle
    /// workers, 64 rounds: every worker takes exactly one request. No lock
    /// orders the callers: a rotation is one atomic step.
    #[test]
    fn callers_that_pick_at_once_take_each_idle_worker_once() {
        use std::{sync::Barrier, thread};

        const CALLERS: usize = 4;
        const ROUNDS: usize = 64;
        let (registry, fleet) = fleet(CALLERS * ROUNDS);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let round = Barrier::new(CALLERS);
        let taken: Vec<usize> = thread::scope(|scope| {
            let callers: Vec<_> = (0..CALLERS)
                .map(|_| {
                    scope.spawn(|| {
                        let pick = |_| {
                            round.wait();
                            placed(&registry, &policies, &fleet, PlacementInputs::default())
                                .expect("an idle fleet takes every request")
                        };
                        (0..ROUNDS).map(pick).collect::<Vec<_>>()
                    })
                })
                .collect();
            callers
                .into_iter()
                .flat_map(|caller| caller.join().expect("a caller ends"))
                .collect()
        });
        let mut times = vec![0usize; fleet.len()];
        for at in taken {
            times[at] += 1;
        }
        let piled = times.iter().filter(|&&times| times != 1).count();
        assert_eq!(piled, 0, "workers that took no request or more than one");
    }

    /// The all-unavailable 503 tells the client when to come back, like the
    /// gateway's other refusals.
    #[test]
    fn no_available_workers_answers_503_with_retry_after() {
        let response = no_available_workers("every worker is down");

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            "no_available_workers"
        );
        assert_eq!(
            response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some("1")
        );
    }

    // ---- the refusal line -------------------------------------------------------------------

    /// Collects, on this thread's subscriber, the events of one level as
    /// `field=value` renderings (the message among them).
    struct EventLines(tracing::Level, Arc<Mutex<Vec<String>>>);

    impl<S: tracing::Subscriber> Layer<S> for EventLines {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if *event.metadata().level() != self.0 {
                return;
            }
            struct Render(String);
            impl tracing::field::Visit for Render {
                fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
                    use fmt::Write;
                    let _ = write!(self.0, "{}={:?} ", field.name(), value);
                }
            }
            let mut rendered = Render(String::new());
            event.record(&mut rendered);
            self.1.lock().unwrap().push(rendered.0);
        }
    }

    /// The model in the line is the caller's own text, so it is written as a
    /// string field: quoted, a line break in it spelled out, the line one
    /// line.
    #[test]
    fn the_refusal_line_writes_the_callers_model_name_quoted() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let _subscriber = test_subscriber(EventLines(tracing::Level::WARN, Arc::clone(&lines)));
        let registry = WorkerRegistry::new();
        let site = RefusalSite {
            function: "Router::route_typed_request",
            mode: "Regular",
            pools: &[("regular", RoutingPool::HttpRegular)],
        };
        let throttle = KeyedLogThrottle::new(2, Duration::from_millis(100), REFUSAL_LOG_MODELS);
        let model = "nope\n2026-01-01 00:00:00  WARN forged";

        let refused = report_refusal_under(
            &throttle,
            &registry,
            site,
            None,
            model,
            error::model_not_found(model),
        );

        assert_eq!(refused.status(), StatusCode::NOT_FOUND);
        let lines = lines.lock().unwrap();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(!lines[0].contains('\n'), "{:?}", lines[0]);
        assert!(
            lines[0].contains(r#"model_id="nope\n2026-01-01 00:00:00  WARN forged""#),
            "{:?}",
            lines[0]
        );
    }

    /// A refusal the gateway answers on its own is marked for the loggers
    /// downstream and written once, at WARN, with the refusing router, the
    /// request id, the model, its reason and the pool's state; past the cap
    /// the lines are counted onto the key's next admitted one: a model's own
    /// for the 503, one for all unknown names for the 404; a refusal already
    /// reported and any other answer pass through silently.
    #[test]
    fn a_refusal_is_marked_and_written_once_at_warn_with_the_pools_state() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let _subscriber = test_subscriber(EventLines(tracing::Level::WARN, Arc::clone(&lines)));
        let registry = WorkerRegistry::new();
        let down = BasicWorkerBuilder::new("http://worker-down:8080")
            .model(ModelCard::new("m1"))
            .worker_type(WorkerType::Regular)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        down.set_status(WorkerStatus::NotReady);
        registry.register(Arc::new(down)).unwrap();
        let site = RefusalSite {
            function: "Router::route_typed_request",
            mode: "Regular",
            pools: &[("regular", RoutingPool::HttpRegular)],
        };
        let window = Duration::from_millis(100);
        let throttle = KeyedLogThrottle::new(2, window, REFUSAL_LOG_MODELS);
        let refuse = |model: &str, refusal: Response| {
            report_refusal_under(&throttle, &registry, site, Some("req-1"), model, refusal)
        };
        let count = || lines.lock().unwrap().len();
        let line = |i: usize| lines.lock().unwrap()[i].clone();

        // The 503, five times in one window: two lines, the model's own.
        for _ in 0..5 {
            let refused = refuse("m1", no_available_workers("every worker is down"));
            assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert!(
                refused.extensions().get::<ReportedRefusal>().is_some(),
                "the refusal is marked as reported"
            );
        }
        assert_eq!(count(), 2, "two lines per window, not five: {:?}", line(1));
        for expected in [
            "message=No available workers for model",
            "function=\"Router::route_typed_request\"",
            "request_id=\"req-1\"",
            "mode=Regular",
            "model_id=\"m1\"",
            "reason=\"every worker is down\"",
            "workers=1",
            "available=0",
            "pool=regular 0/1 available (0 healthy)",
            "suppressed=0",
        ] {
            assert!(
                line(0).contains(expected),
                "{expected} missing from {:?}",
                line(0)
            );
        }

        // The 404s share one key: m2 and m3 take the window's two lines, m4
        // is counted, whatever the name.
        let gone = report_refusal_under(
            &throttle,
            &registry,
            site,
            None,
            "m2",
            error::model_not_found("m2"),
        );
        assert_eq!(gone.status(), StatusCode::NOT_FOUND);
        assert!(gone.extensions().get::<ReportedRefusal>().is_some());
        assert_eq!(count(), 3, "{:?}", line(2));
        assert!(
            line(2).contains("message=No worker serves model"),
            "{:?}",
            line(2)
        );
        assert!(line(2).contains("model_id=\"m2\""), "{:?}", line(2));
        assert!(
            !line(2).contains("request_id="),
            "no id of its own, no field: {:?}",
            line(2)
        );
        assert!(
            !line(2).contains("reason="),
            "a 404 names no reason: {:?}",
            line(2)
        );
        assert!(
            line(2).contains("pool=regular 0/0 available (0 healthy)"),
            "{:?}",
            line(2)
        );
        refuse("m3", error::model_not_found("m3"));
        assert_eq!(count(), 4);
        refuse("m4", error::model_not_found("m4"));
        assert_eq!(count(), 4, "a new unknown name opens no fresh allowance");

        // The next window: the model's 503 line carries its three hidden
        // refusals, the unknown names' line carries the one of theirs. A full
        // window from here: both keys' windows opened before this point, so
        // both are over however long the assertions above took.
        std::thread::sleep(window + Duration::from_millis(10));
        refuse("m1", no_available_workers("every worker is down"));
        assert!(line(4).contains("suppressed=3"), "{:?}", line(4));
        refuse("m5", error::model_not_found("m5"));
        assert!(line(5).contains("model_id=\"m5\""), "{:?}", line(5));
        assert!(line(5).contains("suppressed=1"), "{:?}", line(5));

        // Already reported: no second line, whatever the throttle says.
        let again = report_refusal(&registry, site, None, "m2", gone);
        assert!(again.extensions().get::<ReportedRefusal>().is_some());
        assert_eq!(count(), 6, "a reported refusal is not reported again");

        // Another answer is neither marked nor written.
        let unrelated = report_refusal(
            &registry,
            site,
            None,
            "m1",
            error::service_unavailable("no_compatible_pd_pair", "no pair"),
        );
        assert!(unrelated.extensions().get::<ReportedRefusal>().is_none());
        assert_eq!(count(), 6);
    }

    /// A `no_available_workers` 503 for a name with no workers on any role
    /// (the HTTP PD router's answer for a name nobody serves) is an unknown
    /// name: it shares one allowance with every other such name, as the 404
    /// does, so a client rotating names opens no fresh lines per name and
    /// the counts are not lost to the key eviction; a model with workers
    /// keeps a key of its own.
    #[test]
    fn a_503_for_a_name_without_workers_shares_one_allowance_with_the_other_unknown_names() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let _subscriber = test_subscriber(EventLines(tracing::Level::WARN, Arc::clone(&lines)));
        let registry = WorkerRegistry::new();
        let down = BasicWorkerBuilder::new("http://worker-down:8080")
            .model(ModelCard::new("m1"))
            .worker_type(WorkerType::Regular)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        down.set_status(WorkerStatus::NotReady);
        registry.register(Arc::new(down)).unwrap();
        let site = RefusalSite {
            function: "PDRouter::execute_dual_dispatch_attempt",
            mode: "PrefillDecode",
            pools: &[("regular", RoutingPool::HttpRegular)],
        };
        let window = Duration::from_millis(100);
        let throttle = KeyedLogThrottle::new(2, window, REFUSAL_LOG_MODELS);
        let refuse = |model: &str| {
            report_refusal_under(
                &throttle,
                &registry,
                site,
                None,
                model,
                no_available_workers("No available servers: No prefill workers available"),
            )
        };
        let count = || lines.lock().unwrap().len();
        let line = |i: usize| lines.lock().unwrap()[i].clone();

        // Three names nobody serves in one window: two lines, the third counted.
        for name in ["x1", "x2", "x3"] {
            let refused = refuse(name);
            assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert!(refused.extensions().get::<ReportedRefusal>().is_some());
        }
        assert_eq!(
            count(),
            2,
            "a new name opens no fresh allowance: {:?}",
            line(1)
        );
        assert!(line(0).contains("model_id=\"x1\""), "{:?}", line(0));
        assert!(line(1).contains("model_id=\"x2\""), "{:?}", line(1));
        assert!(
            line(0).contains("pool=regular 0/0 available (0 healthy)"),
            "{:?}",
            line(0)
        );

        // The model with a worker registered keeps an allowance of its own.
        refuse("m1");
        assert_eq!(count(), 3, "{:?}", line(2));
        assert!(line(2).contains("model_id=\"m1\""), "{:?}", line(2));
        assert!(
            line(2).contains("pool=regular 0/1 available (0 healthy)"),
            "{:?}",
            line(2)
        );

        // The next window: the unknown names' line carries the one counted.
        std::thread::sleep(window + Duration::from_millis(10));
        refuse("x4");
        assert!(line(3).contains("model_id=\"x4\""), "{:?}", line(3));
        assert!(line(3).contains("suppressed=1"), "{:?}", line(3));
    }

    /// A refusal's line names the reason the router gives, which may differ
    /// from the message the client reads: the message stays the one under
    /// the code, the line gets the verdict (here the policy that declined a
    /// role), so the reason never contradicts the pool state next to it.
    #[test]
    fn the_refusal_line_names_the_reason_the_router_gives_not_the_clients_message() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let _subscriber = test_subscriber(EventLines(tracing::Level::WARN, Arc::clone(&lines)));
        let registry = WorkerRegistry::new();
        let idle = BasicWorkerBuilder::new("http://worker-idle:8080")
            .model(ModelCard::new("m1"))
            .worker_type(WorkerType::Regular)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        registry.register(Arc::new(idle)).unwrap();
        let site = RefusalSite {
            function: "WorkerSelectionStage::execute",
            mode: "PrefillDecode",
            pools: &[("regular", RoutingPool::HttpRegular)],
        };
        let throttle = KeyedLogThrottle::new(2, Duration::from_secs(60), REFUSAL_LOG_MODELS);
        let refused = no_available_workers_because(
            "All workers for model 'm1' are unavailable (unhealthy or circuit breaker open)",
            "Policy cache_aware failed to select a prefill worker",
        );
        assert_eq!(
            refusal_reason(&refused),
            Some("Policy cache_aware failed to select a prefill worker")
        );
        assert_eq!(
            refusal_reason(&no_available_workers("every worker is down")),
            Some("every worker is down"),
            "without a reason of its own the line names the message"
        );
        assert_eq!(refusal_reason(&error::model_not_found("m2")), None);

        report_refusal_under(&throttle, &registry, site, Some("req-2"), "m1", refused);
        let line = lines.lock().unwrap()[0].clone();
        assert!(
            line.contains("reason=\"Policy cache_aware failed to select a prefill worker\""),
            "{line:?}"
        );
        assert!(line.contains("workers=1"), "{line:?}");
    }

    /// A 2P1D gRPC fleet under cache-aware routing on both roles, as the
    /// disaggregated benchmarks run it.
    fn cache_aware_pd() -> (WorkerRegistry, PolicyRegistry) {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        policies.set_prefill_policy(Arc::new(CacheAwarePolicy::new()));
        policies.set_decode_policy(Arc::new(CacheAwarePolicy::new()));
        (registry, policies)
    }

    fn place_prompt(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        tokens: &[u32],
    ) -> PlacedPair {
        select_pair(
            registry,
            policies,
            MODEL,
            &pairs_of(registry, policies),
            None,
            None,
            PlacementInputs {
                tokens: Some(tokens),
                ..Default::default()
            },
            Stated::nothing,
        )
        .expect("a pair")
    }

    /// A session's first turn placed, and its prefill worker made the busier
    /// of the two by three requests: not enough for cache-aware's own spill
    /// gate (a margin of four above a mean of 1.5), so the policy keeps the
    /// next turn there, while a least-loaded pick would take the other
    /// prefill. Returns the session's next turn and its prefill worker.
    fn a_session_on_the_busier_prefill(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
    ) -> (Vec<u32>, Arc<dyn Worker>) {
        let first_turn: Vec<u32> = (0..8_192).collect();
        let next_turn: Vec<u32> = (0..9_216).collect();
        let home = place_prompt(registry, policies, &first_turn)
            .prefill
            .into_worker();
        for _ in 0..3 {
            home.increment_load();
        }
        assert_eq!(
            place_prompt(registry, policies, &next_turn).prefill.url(),
            home.url(),
            "the prefix keeps the session on its prefill worker"
        );
        (next_turn, home)
    }

    /// The only decode worker over the overload thresholds vetoes nothing it
    /// could steer around: the pair still needs it. The veto must not take
    /// the prefill pick from the prefill policy either: the session stays
    /// on the prefill worker holding its prefix, paired with the decode.
    #[test]
    fn an_overloaded_decode_pool_leaves_the_prefill_pick_to_the_prefill_policy() {
        let (registry, policies) = cache_aware_pd();
        let (next_turn, home) = a_session_on_the_busier_prefill(&registry, &policies);
        let decode = registry.get_by_url("grpc://d:1").expect("registered");
        registry.set_worker_overloaded(&decode, true);
        for _ in 0..4 {
            let pair = place_prompt(&registry, &policies, &next_turn);
            assert_eq!(
                pair.prefill.url(),
                home.url(),
                "a decode-side veto re-picked the prefill"
            );
            assert_eq!(pair.decode.url(), "grpc://d:1");
        }
    }

    /// Every prefill worker over the overload thresholds: the veto has
    /// nowhere to steer, so the prefill pool is placed by its policy as if
    /// none were over them, and the session stays on the prefill worker
    /// holding its prefix rather than going to the least-loaded one.
    #[test]
    fn an_all_overloaded_prefill_pool_is_placed_by_the_prefill_policy() {
        let (registry, policies) = cache_aware_pd();
        let (next_turn, home) = a_session_on_the_busier_prefill(&registry, &policies);
        for url in ["grpc://p:1", "grpc://p:2"] {
            let worker = registry.get_by_url(url).expect("registered");
            registry.set_worker_overloaded(&worker, true);
        }
        for _ in 0..4 {
            assert_eq!(
                place_prompt(&registry, &policies, &next_turn).prefill.url(),
                home.url(),
                "an all-overloaded prefill pool was re-picked by load"
            );
        }
        // Both pools over the thresholds: the same.
        let decode = registry.get_by_url("grpc://d:1").expect("registered");
        registry.set_worker_overloaded(&decode, true);
        assert_eq!(
            place_prompt(&registry, &policies, &next_turn).prefill.url(),
            home.url()
        );
    }

    /// One prefill worker over the thresholds beside one under them, as the
    /// two take turns under sustained load: cache-aware prefill placement is
    /// not steered by the veto, so the session stays on the prefill worker
    /// holding its prefix.
    #[test]
    fn an_overloaded_prefill_holding_the_prefix_keeps_the_session_under_cache_aware() {
        let (registry, policies) = cache_aware_pd();
        let (next_turn, home) = a_session_on_the_busier_prefill(&registry, &policies);
        registry.set_worker_overloaded(&home, true);
        for _ in 0..4 {
            assert_eq!(
                place_prompt(&registry, &policies, &next_turn).prefill.url(),
                home.url(),
                "the veto moved the session off the prefill holding its prefix"
            );
        }
    }

    /// The session's prefill worker and the only decode worker over the
    /// thresholds, the other prefill under them: the decode pool pairs by
    /// load, and the cache-aware prefill pick is not steered by the veto, so
    /// the session stays on the prefill worker holding its prefix.
    #[test]
    fn an_overloaded_prefill_holding_the_prefix_keeps_the_session_beside_an_overloaded_decode() {
        let (registry, policies) = cache_aware_pd();
        let (next_turn, home) = a_session_on_the_busier_prefill(&registry, &policies);
        registry.set_worker_overloaded(&home, true);
        let decode = registry.get_by_url("grpc://d:1").expect("registered");
        registry.set_worker_overloaded(&decode, true);
        for _ in 0..4 {
            let pair = place_prompt(&registry, &policies, &next_turn);
            assert_eq!(
                pair.prefill.url(),
                home.url(),
                "the veto moved the session off the prefill holding its prefix"
            );
            assert_eq!(pair.decode.url(), "grpc://d:1");
        }
    }

    /// Cache-aware's own gate still moves a request off an overloaded holder
    /// loaded well past the pool: the spill the veto no longer forces.
    #[test]
    fn a_saturated_overloaded_holder_still_spills_under_cache_aware() {
        let (registry, policies) = cache_aware_pd();
        let (next_turn, home) = a_session_on_the_busier_prefill(&registry, &policies);
        registry.set_worker_overloaded(&home, true);
        for _ in 0..27 {
            home.increment_load();
        }
        assert_ne!(
            place_prompt(&registry, &policies, &next_turn).prefill.url(),
            home.url(),
            "the count-pressure gate spills off the saturated holder"
        );
    }

    /// A prefill policy that does not weigh load against affinity keeps the
    /// veto: with one prefill worker over the thresholds and the other under
    /// them, every pick goes to the one under them.
    #[test]
    fn an_overloaded_prefill_beside_one_under_the_thresholds_is_still_steered_around() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let over = registry.get_by_url("grpc://p:1").expect("registered");
        registry.set_worker_overloaded(&over, true);
        let prompt: Vec<u32> = (0..8_192).collect();
        for _ in 0..4 {
            assert_eq!(
                place_prompt(&registry, &policies, &prompt).prefill.url(),
                "grpc://p:2",
                "the veto steers to the prefill under the thresholds"
            );
        }
    }

    /// A prefill pool the liveness tracker vetoes in full keeps the
    /// least-loaded fallback: a suspected worker is crossed by load, not by
    /// the policy.
    #[test]
    fn a_prefill_pool_vetoed_by_liveness_keeps_the_least_loaded_fallback() {
        let (registry, policies) = cache_aware_pd();
        let (next_turn, home) = a_session_on_the_busier_prefill(&registry, &policies);
        for url in ["grpc://p:1", "grpc://p:2"] {
            let worker = registry.get_by_url(url).expect("registered");
            worker.set_stall(Some(StallReason::Wedged));
        }
        assert_ne!(
            place_prompt(&registry, &policies, &next_turn).prefill.url(),
            home.url(),
            "the least-loaded stalled prefill serves"
        );
    }

    /// Every prefill worker over the thresholds under a load-blind prefill
    /// policy: it has no affinity a load pick would discard, so the pool is
    /// steered to its least-loaded worker as before, and the busier worker
    /// takes none of the placements.
    #[test]
    fn an_all_overloaded_prefill_pool_under_a_load_blind_policy_goes_to_the_least_loaded() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = cohort_policies();
        let busier = registry.get_by_url("grpc://p:1").expect("registered");
        for _ in 0..10 {
            busier.increment_load();
        }
        for url in ["grpc://p:1", "grpc://p:2"] {
            let worker = registry.get_by_url(url).expect("registered");
            registry.set_worker_overloaded(&worker, true);
        }
        let prompt: Vec<u32> = (0..8_192).collect();
        for _ in 0..4 {
            assert_eq!(
                place_prompt(&registry, &policies, &prompt).prefill.url(),
                "grpc://p:2",
                "the least-loaded prefill"
            );
        }
    }

    /// The same under a hash policy, which hashes only over workers under
    /// the thresholds: it is not asked to place a pool all over them (no
    /// `no_healthy_workers` decision of its own), and the least-loaded
    /// default steers the request.
    #[test]
    fn an_all_overloaded_prefill_pool_under_a_hash_policy_goes_to_the_least_loaded() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        policies.set_prefill_policy(Arc::new(ConsistentHashingPolicy::new()));
        registry
            .get_by_url("grpc://p:1")
            .expect("registered")
            .increment_load();
        for url in ["grpc://p:1", "grpc://p:2"] {
            let worker = registry.get_by_url(url).expect("registered");
            registry.set_worker_overloaded(&worker, true);
        }
        let pair = metrics::with_local_recorder(&recorder, || pair_from(&registry, &policies))
            .expect("steered by load");
        assert_eq!(pair.prefill.url(), "grpc://p:2", "the less loaded prefill");
        assert_eq!(pair.decode.url(), "grpc://d:1");
        let rendered = handle.render();
        assert!(
            !rendered.contains("no_healthy_workers"),
            "the hash policy was asked to place a pool all over the thresholds:\n{rendered}"
        );
    }

    /// Both decode workers over the thresholds, the session's decode the
    /// busier: the decode pool is paired by load with the prefill pick, as
    /// before, not by the decode policy's affinity, so every placement goes
    /// to the less loaded decode worker.
    #[test]
    fn an_all_overloaded_decode_pool_pairs_the_prefill_pick_with_its_least_loaded_partner() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        policies.set_prefill_policy(Arc::new(CacheAwarePolicy::new()));
        policies.set_decode_policy(Arc::new(CacheAwarePolicy::new()));
        let first_turn: Vec<u32> = (0..8_192).collect();
        let next_turn: Vec<u32> = (0..9_216).collect();
        let home = place_prompt(&registry, &policies, &first_turn).decode;
        for _ in 0..3 {
            home.increment_load();
        }
        for url in ["grpc://d:1", "grpc://d:2"] {
            let worker = registry.get_by_url(url).expect("registered");
            registry.set_worker_overloaded(&worker, true);
        }
        for _ in 0..4 {
            let pair = place_prompt(&registry, &policies, &next_turn);
            assert_eq!(pair.prefill.url(), "grpc://p:1");
            assert_ne!(
                pair.decode.url(),
                home.url(),
                "the busier decode kept its session over the thresholds"
            );
        }
    }

    /// A cache-aware prefill pool whose only worker is over the thresholds,
    /// and pairs only with a decode worker over them too, while the decode
    /// worker free of them is on another KV transport: no pair is open, and
    /// the least-loaded default pairs the two over the thresholds, as it did
    /// before the veto was lifted, instead of the placement failing on the
    /// decode pool.
    #[test]
    fn a_lifted_prefill_pool_without_an_open_pair_keeps_the_least_loaded_fallback() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        policies.set_prefill_policy(Arc::new(CacheAwarePolicy::new()));
        for url in ["grpc://p:1", "grpc://d:1"] {
            let worker = registry.get_by_url(url).expect("registered");
            registry.set_worker_overloaded(&worker, true);
        }
        let pair = pair_from(&registry, &policies).expect("steered to the overloaded pair");
        assert_eq!(pair.prefill.url(), "grpc://p:1");
        assert_eq!(pair.decode.url(), "grpc://d:1");
    }

    /// A long conversation's next turn, its ~99,000-token context held by
    /// its prefill worker, at 25 requests in flight there against 7 on the
    /// other prefill worker, under the launcher's spill thresholds (64
    /// requests and 1.5 times the mean): the cache-aware prefill policy
    /// keeps the turn with its cache. Priced at 0, the swing-sized margin (8
    /// at a mean of 16) moves it to the other prefill worker, which prefills
    /// the whole context again.
    #[test]
    fn a_cache_aware_prefill_pool_keeps_a_long_session_at_a_small_skew() {
        for (price, stays) in [(1_024, true), (0, false)] {
            let registry = pd_registry(&[
                ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
                ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
                ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ]);
            let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
            let config = CacheAwareConfig {
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                balance_prefix_margin_tokens: price,
                eviction_interval_secs: 0,
                ..Default::default()
            };
            policies.set_prefill_policy(Arc::new(CacheAwarePolicy::with_config(config.clone())));
            policies.set_decode_policy(Arc::new(CacheAwarePolicy::with_config(config)));
            let first_turn: Vec<u32> = (0..99_000).collect();
            let next_turn: Vec<u32> = (0..100_000).collect();
            let home = place_prompt(&registry, &policies, &first_turn).prefill;
            let other_url = if home.url() == "grpc://p:1" {
                "grpc://p:2"
            } else {
                "grpc://p:1"
            };
            let other = registry.get_by_url(other_url).expect("registered");
            for _ in 0..25 {
                home.increment_load();
            }
            for _ in 0..7 {
                other.increment_load();
            }
            let pair = place_prompt(&registry, &policies, &next_turn);
            assert_eq!(
                pair.prefill.url() == home.url(),
                stays,
                "price {price}: the turn went to {}",
                pair.prefill.url()
            );
        }
    }

    fn keyed_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", "session-7".parse().unwrap());
        headers
    }

    fn with_headers(headers: &HeaderMap) -> PlacementInputs<'_> {
        PlacementInputs {
            headers: Some(headers),
            ..Default::default()
        }
    }

    /// A two-worker fleet and the worker that owns the session in `inputs`.
    fn a_keyed_owner(
        policies: &PolicyRegistry,
        inputs: PlacementInputs<'_>,
    ) -> (WorkerRegistry, Arc<dyn Worker>) {
        let registry = http_registry(&["http://h:1", "http://h:2"]);
        let owner = pick(&registry, policies, inputs);
        (registry, owner)
    }

    /// One worker over the overload thresholds beside one under them, as the
    /// two take turns under sustained load: a session key (the routing-key
    /// header, a configured routing-key header, or the rid-derived key) keeps
    /// its ring owner without counting a fallback, while a request without a
    /// key is still steered around the owner, and so is the session under
    /// `--worker-overload-shed`.
    #[test]
    fn a_session_key_keeps_its_overloaded_owner_under_consistent_hashing() {
        let routing_key = keyed_headers();
        let mut configured = HeaderMap::new();
        configured.insert("x-session-id", "session-7".parse().unwrap());
        let configured_policies = PolicyRegistry::with_override(
            PolicyConfig::ConsistentHashing,
            RoutingKeyOverrideConfig {
                headers: vec!["x-session-id".to_string()],
                ..Default::default()
            },
        );
        let cases = [
            (
                PolicyRegistry::new(PolicyConfig::ConsistentHashing),
                with_headers(&routing_key),
            ),
            (configured_policies, with_headers(&configured)),
            (
                PolicyRegistry::new(PolicyConfig::ConsistentHashing),
                PlacementInputs {
                    rid_key: Some("session-7"),
                    ..Default::default()
                },
            ),
        ];
        for (policies, inputs) in cases {
            let (registry, owner) = a_keyed_owner(&policies, inputs);
            registry.set_worker_overloaded(&owner, true);
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            let selected =
                metrics::with_local_recorder(&recorder, || pick(&registry, &policies, inputs));
            assert_eq!(selected.url(), owner.url(), "the veto moved the session");
            assert!(!handle
                .render()
                .contains("smg_worker_overload_fallback_total"));
            let unkeyed = pick(&registry, &policies, PlacementInputs::default());
            assert_ne!(
                unkeyed.url(),
                owner.url(),
                "an unkeyed request reached the owner"
            );
            registry.set_overload_shed(true);
            assert_ne!(pick(&registry, &policies, inputs).url(), owner.url());
        }
    }

    /// A header key with turn and retry suffixes under the routing-key
    /// override: placement derives the key once, lineage-stripped, so the
    /// policies that place by the key themselves (consistent hashing, the
    /// sticky map) see the key the load guard counts, and every turn of
    /// `conv` lands on the owner of `conv`. Without the override the key is
    /// raw: a new turn is a new key.
    #[test]
    fn a_header_keys_turns_share_one_owner_under_the_override() {
        let keyed = |key: &str| {
            let mut headers = HeaderMap::new();
            headers.insert("x-smg-routing-key", key.parse().unwrap());
            headers
        };
        let manual = || PolicyConfig::Manual {
            eviction_interval_secs: 0,
            max_idle_secs: 0,
            assignment_mode: ManualAssignmentMode::MinLoad,
        };
        let turns = ["conv_t1", "conv_t2", "conv_t3_r1", "conv_r2"];
        for config in [PolicyConfig::ConsistentHashing, manual()] {
            let policies = PolicyRegistry::with_override(
                config,
                RoutingKeyOverrideConfig {
                    enabled: true,
                    ..Default::default()
                },
            );
            let name = policies.get_default_policy().name();
            let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
            let base = keyed("conv");
            let owner = pick(&registry, &policies, with_headers(&base));
            // A vacant sticky key would go to a lighter worker.
            owner.increment_load();
            for turn in turns {
                let headers = keyed(turn);
                assert_eq!(
                    pick(&registry, &policies, with_headers(&headers)).url(),
                    owner.url(),
                    "{name}: {turn} left the owner of conv"
                );
            }
        }

        // Override off: the sticky map pins each raw key on its own.
        let policies = PolicyRegistry::new(manual());
        let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
        let first = keyed("conv_t1");
        let owner = pick(&registry, &policies, with_headers(&first));
        owner.increment_load();
        let second = keyed("conv_t2");
        assert_ne!(
            pick(&registry, &policies, with_headers(&second)).url(),
            owner.url(),
            "without the override a new turn is a new key"
        );
    }

    /// Under the override a round-robin default keeps every turn of a
    /// conversation on the owner of its key.
    #[test]
    fn a_round_robin_defaults_turns_share_one_owner_under_the_override() {
        let keyed = |key: &str| {
            let mut headers = HeaderMap::new();
            headers.insert("x-smg-routing-key", key.parse().unwrap());
            headers
        };
        let turns = ["conv_t1", "conv_t2", "conv_t3_r1", "conv_r2"];
        let policies = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                enabled: true,
                ..Default::default()
            },
        );
        let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
        let base = keyed("conv");
        let owner = pick(&registry, &policies, with_headers(&base));
        // A vacant key would go to a lighter worker.
        owner.increment_load();
        for turn in turns {
            let headers = keyed(turn);
            assert_eq!(
                pick(&registry, &policies, with_headers(&headers)).url(),
                owner.url(),
                "{turn} left the owner of conv"
            );
        }
    }

    /// The manual policy reads the `x-smg-routing-key` hint under an
    /// override whose configured list names other headers: a conversation
    /// keyed by the hint alone keeps its owner across turns with fresh
    /// request ids, and across a turn with none.
    #[test]
    fn the_manual_policy_reads_the_hint_under_a_configured_list() {
        let manual = PolicyConfig::Manual {
            eviction_interval_secs: 0,
            max_idle_secs: 0,
            assignment_mode: ManualAssignmentMode::MinLoad,
        };
        let hinted = keyed_headers();
        let policies = PolicyRegistry::with_override(
            manual,
            RoutingKeyOverrideConfig {
                enabled: true,
                headers: vec!["x-session-id".into()],
                ..Default::default()
            },
        );
        let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
        let turn = |rid_key: Option<&'static str>| PlacementInputs {
            headers: Some(&hinted),
            rid_key,
            ..Default::default()
        };
        let owner = pick(&registry, &policies, turn(Some("request-1")));
        // A vacant key would go to a lighter worker.
        owner.increment_load();
        for rid_key in [Some("request-2"), Some("request-3"), None] {
            assert_eq!(
                pick(&registry, &policies, turn(rid_key)).url(),
                owner.url(),
                "the turn keyed {rid_key:?} left the owner of the hint"
            );
        }
    }

    /// The hint keys nothing for a default the override places: with the
    /// override's configured list naming other headers, a round-robin
    /// request carrying the hint alone rotates.
    #[test]
    fn the_hint_keys_nothing_under_the_override() {
        let hinted = keyed_headers();
        let policies = PolicyRegistry::with_override(
            PolicyConfig::RoundRobin,
            RoutingKeyOverrideConfig {
                enabled: true,
                headers: vec!["x-session-id".into()],
                ..Default::default()
            },
        );
        let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..3 {
            seen.insert(
                pick(&registry, &policies, with_headers(&hinted))
                    .url()
                    .to_string(),
            );
        }
        assert_eq!(seen.len(), 3, "the hint alone rotates: {seen:?}");
    }

    /// A key whose owner is not ready takes a failover. Once that worker is
    /// back the key returns to it under the manual policy; under the
    /// override it stays on the failover, as the override pins the newest
    /// worker first.
    #[test]
    fn a_key_recovers_under_manual_and_stays_on_the_failover_under_the_override() {
        let headers = keyed_headers();
        let manual = PolicyConfig::Manual {
            eviction_interval_secs: 0,
            max_idle_secs: 0,
            assignment_mode: ManualAssignmentMode::Random,
        };
        let cases = [
            (manual, false, true),
            (PolicyConfig::RoundRobin, true, false),
        ];
        for (config, override_on, returns) in cases {
            let policies = PolicyRegistry::with_override(
                config.clone(),
                RoutingKeyOverrideConfig {
                    enabled: override_on,
                    ..Default::default()
                },
            );
            let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
            let owner = pick(&registry, &policies, with_headers(&headers));
            owner.set_status(WorkerStatus::NotReady);
            let failover = pick(&registry, &policies, with_headers(&headers));
            assert_ne!(failover.url(), owner.url(), "{}", config.name());
            for _ in 0..3 {
                assert_eq!(
                    pick(&registry, &policies, with_headers(&headers)).url(),
                    failover.url(),
                    "{}: the failover holds the key",
                    config.name()
                );
            }
            owner.set_status(WorkerStatus::Ready);
            let back = pick(&registry, &policies, with_headers(&headers));
            assert_eq!(
                back.url() == owner.url(),
                returns,
                "{}: the key went to {}",
                config.name(),
                back.url()
            );
        }
    }

    /// Every worker over the thresholds, the owner the busier: the session
    /// stays on its owner instead of going to the least-loaded worker, and
    /// the placement counts as the overload fallback it replaces.
    #[test]
    fn a_pool_all_over_the_thresholds_keeps_a_routing_key_on_its_owner() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let headers = keyed_headers();
        let inputs = with_headers(&headers);
        let policies = PolicyRegistry::new(PolicyConfig::ConsistentHashing);
        let (registry, owner) = a_keyed_owner(&policies, inputs);
        owner.increment_load();
        for worker in registry.get_all() {
            registry.set_worker_overloaded(&worker, true);
        }
        let selected =
            metrics::with_local_recorder(&recorder, || pick(&registry, &policies, inputs));
        assert_eq!(selected.url(), owner.url(), "the session went by load");
        assert_one_overload_fallback(&handle.render());
    }

    fn assert_one_overload_fallback(rendered: &str) {
        assert!(
            rendered.lines().any(|line| {
                line.starts_with("smg_worker_overload_fallback_total{")
                    && line.contains("stage=\"selection\"")
                    && line.ends_with(" 1")
            }),
            "the placement is not counted as the overload fallback:\n{rendered}"
        );
    }

    /// No worker is fully available, but one is only over the thresholds,
    /// beside one that is not ready: a keyed request takes it, whether it is
    /// the owner or the owner's ring successor, and counts as the overload
    /// fallback.
    #[test]
    fn a_keyed_request_takes_a_worker_only_over_the_thresholds() {
        let headers = keyed_headers();
        let inputs = with_headers(&headers);
        let policies = PolicyRegistry::new(PolicyConfig::ConsistentHashing);
        for owner_over_the_thresholds in [true, false] {
            let (registry, owner) = a_keyed_owner(&policies, inputs);
            let other = registry
                .get_all()
                .into_iter()
                .find(|worker| worker.url() != owner.url())
                .expect("a second worker");
            let (over, not_ready) = if owner_over_the_thresholds {
                (&owner, &other)
            } else {
                (&other, &owner)
            };
            registry.set_worker_overloaded(over, true);
            not_ready.set_status(WorkerStatus::NotReady);
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            let selected =
                metrics::with_local_recorder(&recorder, || pick(&registry, &policies, inputs));
            assert_eq!(selected.url(), over.url());
            assert_one_overload_fallback(&handle.render());
        }
    }

    /// Only the overload veto is lifted: an owner over the thresholds that is
    /// also not ready, behind an open circuit breaker, or stalled loses the
    /// session.
    #[test]
    fn a_session_key_still_leaves_an_owner_vetoed_for_another_reason() {
        let headers = keyed_headers();
        let inputs = with_headers(&headers);
        let policies = PolicyRegistry::new(PolicyConfig::ConsistentHashing);
        for veto in ["not ready", "circuit open", "stalled"] {
            let (registry, owner) = a_keyed_owner(&policies, inputs);
            registry.set_worker_overloaded(&owner, true);
            match veto {
                "not ready" => owner.set_status(WorkerStatus::NotReady),
                "circuit open" => {
                    for _ in 0..5 {
                        owner.record_circuit_breaker_outcome(false);
                    }
                    assert!(!owner.circuit_breaker_can_execute());
                }
                _ => {
                    owner.set_stall(Some(StallReason::Wedged));
                }
            }
            assert_ne!(
                pick(&registry, &policies, inputs).url(),
                owner.url(),
                "{veto}"
            );
        }
    }

    /// A keyed retry leaves the owner it was tried on for the next worker
    /// even when that worker is over the thresholds, instead of replaying on
    /// the owner.
    #[test]
    fn a_keyed_retry_moves_to_a_worker_over_the_thresholds() {
        let headers = keyed_headers();
        let inputs = with_headers(&headers);
        let policies = PolicyRegistry::new(PolicyConfig::ConsistentHashing);
        let (registry, owner) = a_keyed_owner(&policies, inputs);
        for worker in registry.get_all() {
            if worker.url() != owner.url() {
                registry.set_worker_overloaded(&worker, true);
            }
        }
        let tried = vec![owner.base_url().to_string()];
        let retry = PlacementInputs {
            tried: &tried,
            ..inputs
        };
        assert_ne!(pick(&registry, &policies, retry).url(), owner.url());
    }

    /// The lift is for regular workers: a keyed placement over a prefill pool
    /// (the HTTP PD router's token count) keeps the overload veto.
    #[test]
    fn a_session_key_over_a_prefill_pool_keeps_the_overload_veto() {
        let registry = registry_of(&[
            (
                "http://p:1",
                WorkerType::Prefill,
                ConnectionMode::Http,
                RuntimeType::Sglang,
            ),
            (
                "http://p:2",
                WorkerType::Prefill,
                ConnectionMode::Http,
                RuntimeType::Sglang,
            ),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::ConsistentHashing);
        let headers = keyed_headers();
        let inputs = with_headers(&headers);
        let place = || {
            select_single(
                &registry,
                &policies,
                MODEL,
                RoutingPool::HttpPrefill,
                None,
                inputs,
                Stated::nothing,
            )
            .expect("a prefill worker")
        };
        let owner = place();
        registry.set_worker_overloaded(owner.worker(), true);
        assert_ne!(place().url(), owner.url());
    }

    // ---- the policy engine ------------------------------------------------------------------

    /// What a stage heard of the leases it was told, in order.
    pub(crate) type Heard = Arc<Mutex<Vec<String>>>;

    /// A scorer that ranks every worker alike and takes every word of a
    /// lease, so the engine leases each pick under it. Where the request it
    /// ranks for states an output limit, it says `limit <tokens>` ahead of
    /// that decision's words: a request that states none adds no word.
    #[derive(Debug)]
    struct Hears(Heard);

    impl Hears {
        fn note(&self, word: String) {
            self.0.lock().expect("the stage heard").push(word);
        }
    }

    impl Lifecycle for Hears {
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }

        fn on_decision_issued(&self, _: &Lease, _: &Decision) {
            self.note("issued".to_owned());
        }

        fn on_dispatched(&self, _: &Lease, _: &Target, side: Side, _: u32) {
            self.note(format!("dispatched {side:?}"));
        }

        fn on_completed(&self, _: &Lease, outcome: &Outcome) {
            self.note(format!("completed {:?}", outcome.class));
        }

        fn on_released(&self, _: &Lease, reason: ReleaseReason) {
            self.note(format!("released {reason:?}"));
        }
    }

    impl Scorer for Hears {
        fn name(&self) -> &'static str {
            "hears"
        }

        fn score(
            &self,
            facts: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            if facts.max_output_tokens != 0 {
                self.note(format!("limit {}", facts.max_output_tokens));
            }
            out.fill(0.0);
            TieBand::NORMALIZED
        }
    }

    /// Hands `policies` an engine for the life of the test process, as the
    /// gateway's start hands one.
    pub(crate) fn engine_on(policies: &PolicyRegistry, spec: AppliedSpec) {
        policies.set_engine(Box::leak(Box::new(PolicyEngine::new(spec))));
    }

    /// Hands `policies` an engine over a stock preset.
    fn preset_on(policies: &PolicyRegistry, preset: &str) {
        let document = PolicySpec::preset(preset).expect("a stock preset");
        engine_on(
            policies,
            AppliedSpec::validate(&document).expect("a valid document"),
        );
    }

    /// Hands `policies` an engine whose one profile ranks by [`Hears`], and
    /// what that stage heard.
    pub(crate) fn hearing_engine_on(policies: &PolicyRegistry) -> Heard {
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        let mut stages = StageRegistry::builtin();
        stages
            .scorer("hears", move || Box::new(Hears(Arc::clone(&kept))))
            .expect("a free name");
        let document: PolicySpec = serde_json::from_str(
            r#"{"apiVersion":"smg.policy/v1","generation":1,"profiles":{"p":{"score":["hears"],"pick":"best"}}}"#,
        )
        .expect("a policy document");
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        engine_on(policies, spec.expect("the stages hold the scorer"));
        heard
    }

    /// With an engine handed to the registry the placement asks it, and the
    /// engine hears the request by its load guard: the dispatch where the
    /// guard is made and the end where it is dropped. A second guard and the
    /// answer's own drop tell it nothing. This document names no routing
    /// value, so the guard counts the request under its route's own key, as
    /// the guard of a classic policy that names no key does
    /// (`a_classic_answer_counts_under_its_routes_key`).
    #[test]
    fn the_engine_hears_a_placed_request_by_its_load_guard() {
        let registry = http_registry(&["http://h:1", "http://h:2"]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_engine_on(&policies);
        let said = || heard.lock().expect("the stage heard").clone();

        let placed = answered(&registry, &policies, PlacementInputs::default());
        assert_eq!(said(), ["issued"]);
        let guard = placed.guard(Some("the route's own key"));
        let counted = placed.routing_key_inflight("the route's own key");
        assert_eq!((placed.load(), counted), (1, 1));
        assert_eq!(said(), ["issued", "dispatched Single"]);

        let second = placed.guard(None);
        let worker = Arc::clone(placed.worker());
        drop(placed);
        drop(second);
        assert_eq!(worker.load(), 1);
        assert_eq!(said(), ["issued", "dispatched Single"]);

        drop(guard);
        assert_eq!(worker.load(), 0);
        assert_eq!(
            said(),
            ["issued", "dispatched Single", "completed Cancelled"],
            "the end is told at the guard's drop, with no outcome"
        );
    }

    /// An answer of the engine that no guard was made from gives its lease
    /// up where it is dropped.
    #[test]
    fn an_answer_of_the_engine_that_no_guard_took_is_released() {
        let registry = http_registry(&["http://h:1"]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_engine_on(&policies);
        drop(answered(&registry, &policies, PlacementInputs::default()));
        assert_eq!(
            *heard.lock().expect("the stage heard"),
            ["issued", "released Abandoned"]
        );
    }

    /// What a route states for the engine alone reaches the engine's request
    /// through each single-worker front: the output limit the request states
    /// is the limit its decision reads, and the call that states it is made
    /// once a placement. A classic placement never makes the call.
    #[test]
    fn the_engine_reads_the_output_limit_a_single_request_states() {
        let registry = http_registry(&["http://h:1", "http://h:2"]);
        let pool = RoutingPool::HttpRegular;
        let inputs = PlacementInputs::default();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_engine_on(&policies);
        let stated = std::cell::Cell::new(0);
        let limit = |tokens: u32| {
            let stated = &stated;
            move || {
                stated.set(stated.get() + 1);
                Stated {
                    max_output_tokens: tokens,
                }
            }
        };

        let snapshot = registry.get_routing_snapshot(MODEL);
        let slice = registry.get_routing_pool(MODEL, pool);
        drop(select_single(&registry, &policies, MODEL, pool, None, inputs, limit(64)).unwrap());
        drop(
            select_single_of(
                &registry,
                &policies,
                MODEL,
                &snapshot,
                pool,
                None,
                inputs,
                limit(65),
            )
            .unwrap(),
        );
        drop(select_from(&registry, &policies, MODEL, pool, &slice, inputs, limit(66)).unwrap());
        // A request that states none adds no word.
        drop(
            select_single(
                &registry,
                &policies,
                MODEL,
                pool,
                None,
                inputs,
                Stated::nothing,
            )
            .unwrap(),
        );
        let released = "released Abandoned";
        assert_eq!(
            *heard.lock().expect("the stage heard"),
            [
                "limit 64", "issued", released, "limit 65", "issued", released, "limit 66",
                "issued", released, "issued", released,
            ]
        );
        assert_eq!(stated.get(), 3, "one call a placement the engine decided");

        let classic = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let unread = || -> Stated { panic!("a classic placement is handed nothing of it") };
        assert!(select_single(&registry, &classic, MODEL, pool, None, inputs, unread).is_some());
        assert!(select_single_of(
            &registry, &classic, MODEL, &snapshot, pool, None, inputs, unread
        )
        .is_some());
        assert!(select_from(&registry, &classic, MODEL, pool, &slice, inputs, unread).is_some());
        let pairs = one_pair();
        let index = pairs_of(&pairs, &classic);
        let unread = || -> Stated { panic!("a classic pair is handed nothing of it") };
        assert!(select_pair(&pairs, &classic, MODEL, &index, None, None, inputs, unread).is_ok());
    }

    /// Under a document whose `has-room` names the request's output limit,
    /// the limit a route states reserves room for the answer: a worker the
    /// answer would fill is passed over for one that holds it. A request
    /// that states no limit reserves none and takes its turn on both.
    #[test]
    fn a_stated_output_limit_reserves_room_under_a_rule_that_names_it() {
        let registry = http_registry(&["http://h:1", "http://h:2"]);
        let pool = RoutingPool::HttpRegular;
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        // A worker holds 100 a request in flight and the answer's limit, and
        // has room up to 150.
        let document: PolicySpec = serde_json::from_str(
            r#"{"apiVersion":"smg.policy/v1","generation":1,"profiles":{"p":{
                "filter":[{"has-room":{"used":{"candidate.in_flight":100,"request.max_output_tokens":1},"capacity":150}}],
                "pick":"round-robin"}}}"#,
        )
        .expect("a policy document");
        engine_on(
            &policies,
            AppliedSpec::validate(&document).expect("a valid document"),
        );
        let place = |tokens: u32| {
            let stated = move || Stated {
                max_output_tokens: tokens,
            };
            let inputs = PlacementInputs::default();
            select_single(&registry, &policies, MODEL, pool, None, inputs, stated)
                .expect("a worker is selectable")
        };

        // One worker holds a request.
        let busy = place(0);
        let _held = busy.guard(None);
        assert_eq!(busy.load(), 1);

        // An answer of up to 100 tokens fits the idle worker alone.
        for _ in 0..4 {
            assert_ne!(place(100).url(), busy.url(), "the busy worker has no room");
        }
        // With no limit stated both have room, and the turn visits both.
        let taken: std::collections::HashSet<String> =
            (0..4).map(|_| place(0).url().to_owned()).collect();
        assert_eq!(
            taken.len(),
            2,
            "a request that states no limit reserves none"
        );
    }

    /// A scorer that panics where the engine asks it: the engine refuses
    /// that decision as its own fault.
    #[derive(Debug)]
    struct Faults;

    impl Lifecycle for Faults {
        fn hears(&self) -> EventSet {
            EventSet::LEASE
        }
    }

    impl Scorer for Faults {
        fn name(&self) -> &'static str {
            "faults"
        }

        fn score(
            &self,
            _: &RequestFacts<'_>,
            _: &[Candidate],
            _: &[u32],
            _: &mut [f32],
        ) -> TieBand {
            panic!("the test's own stage fails")
        }
    }

    /// A decision the engine refuses is the engine's own fault, not the
    /// fleet's state. The placement answers no worker, as for every request
    /// the engine does not place, and writes the refusal at `warn` with the
    /// model and the engine's reason.
    #[test]
    fn a_decision_the_engine_refuses_is_written_at_warn() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let _subscriber = test_subscriber(EventLines(tracing::Level::WARN, Arc::clone(&lines)));
        let registry = http_registry(&["http://h:1", "http://h:2"]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let mut stages = StageRegistry::builtin();
        stages
            .scorer("faults", || Box::new(Faults))
            .expect("a free name");
        let document: PolicySpec = serde_json::from_str(
            r#"{"apiVersion":"smg.policy/v1","generation":1,"profiles":{"p":{"score":["faults"],"pick":"best"}}}"#,
        )
        .expect("a policy document");
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        engine_on(&policies, spec.expect("the stages hold the scorer"));

        let placed = select_single(
            &registry,
            &policies,
            MODEL,
            RoutingPool::HttpRegular,
            None,
            PlacementInputs::default(),
            Stated::nothing,
        );
        assert!(placed.is_none(), "a refused decision places no worker");
        let lines = lines.lock().unwrap();
        let refused: Vec<_> = lines
            .iter()
            .filter(|line| line.contains("message=The policy engine refused a decision"))
            .collect();
        assert_eq!(refused.len(), 1, "{lines:?}");
        for expected in ["model_id=\"m\"", "refused=Faulted", "suppressed=0"] {
            assert!(
                refused[0].contains(expected),
                "{expected} missing from {:?}",
                refused[0]
            );
        }
    }

    /// The guard of an engine's answer counts the request under the routing
    /// value the engine names, whatever key its route reads. The document
    /// here reads the value from the body's own id, which the engine keeps
    /// in the clear. The value keeps its worker.
    #[test]
    fn the_engines_routing_value_counts_the_request_on_its_worker() {
        let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let mut document = PolicySpec::preset("consistent_hashing").expect("a stock preset");
        document.bindings[0].routing_key = Some(RoutingKeySpec {
            sources: vec![KeySourceSpec {
                rid: true,
                ..KeySourceSpec::default()
            }],
            strip_lineage: false,
        });
        let spec = AppliedSpec::validate(&document);
        engine_on(&policies, spec.expect("a valid document"));
        let keyed = PlacementInputs {
            rid: Some("conversation-1"),
            ..Default::default()
        };
        let placed = answered(&registry, &policies, keyed);
        let guard = placed.guard(Some("the route's own key"));
        let under = |key: &str| placed.routing_key_inflight(key);
        assert_eq!(placed.load(), 1);
        assert_eq!(
            (under("conversation-1"), under("the route's own key")),
            (1, 0)
        );
        for _ in 0..4 {
            assert_eq!(pick(&registry, &policies, keyed).url(), placed.url());
        }
        drop(guard);
        assert_eq!((placed.load(), placed.routing_key_load()), (0, 0));
    }

    /// A classic policy's answer makes the guard a route made before: under
    /// the route's own key, and with nothing kept.
    #[test]
    fn a_classic_answer_counts_under_its_routes_key() {
        let registry = http_registry(&["http://h:1"]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let placed = answered(&registry, &policies, PlacementInputs::default());
        let guard = placed.guard(Some("the route's own key"));
        assert_eq!((placed.load(), placed.routing_key_load()), (1, 1));
        let worker = Arc::clone(placed.worker());
        drop(guard);
        let moved = placed.into_guard(None);
        assert_eq!((worker.load(), worker.routing_key_load()), (1, 0));
        drop(moved);
        assert_eq!(worker.load(), 0);
    }

    /// Under the engine a part of a pool is placed over the pool's own list,
    /// held to the part by the request's rule: a caller's own slice, and
    /// the part on a retry's pinned wire. Every worker of the part takes
    /// requests and no other worker does. The list's turn is the pool's: a
    /// part that is the whole pool gives each worker its exact turn. A pool
    /// none of whose workers is ready has no answer.
    #[test]
    fn the_engine_places_a_part_of_a_pool_over_the_pools_own_list() {
        let registry = registry_with(&[
            ("http://s:1", ConnectionMode::Http, RuntimeType::Sglang),
            ("http://v:1", ConnectionMode::Http, RuntimeType::Vllm),
            ("http://v:2", ConnectionMode::Http, RuntimeType::Vllm),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        preset_on(&policies, "round_robin");
        let pool = RoutingPool::HttpRegular;

        let two: Vec<Arc<dyn Worker>> = ["http://s:1", "http://v:2"]
            .iter()
            .map(|url| registry.get_by_url(url).expect("registered"))
            .collect();
        let pinned = WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: ConnectionMode::Http,
            requires_media_refs: false,
            pipeline: PipelinePin::Any,
        };
        let mut from_slice = BTreeMap::new();
        let mut on_wire = BTreeMap::new();
        let inputs = PlacementInputs::default();
        for _ in 0..16 {
            let placed = select_from(
                &registry,
                &policies,
                MODEL,
                pool,
                &two,
                inputs,
                Stated::nothing,
            );
            let placed = placed.expect("a ready candidate");
            *from_slice.entry(placed.url().to_owned()).or_insert(0) += 1;
            let placed = select_single(
                &registry,
                &policies,
                MODEL,
                pool,
                Some(pinned),
                inputs,
                Stated::nothing,
            );
            let placed = placed.expect("a ready worker on the wire");
            *on_wire.entry(placed.url().to_owned()).or_insert(0) += 1;
        }
        let urls = |seen: BTreeMap<String, i32>| seen.into_keys().collect::<Vec<_>>();
        assert_eq!(urls(from_slice), ["http://s:1", "http://v:2"]);
        assert_eq!(urls(on_wire), ["http://v:1", "http://v:2"]);

        // The request's own rule holds beside the part: where it refuses
        // one of the wire's two workers, and one of the slice's two, the
        // other takes every request.
        let refuses_one = |worker: &dyn Worker| worker.url() != "http://v:2";
        let ruled = PlacementInputs {
            candidate_filter: Some(&refuses_one),
            ..Default::default()
        };
        for _ in 0..8 {
            let placed = select_single(
                &registry,
                &policies,
                MODEL,
                pool,
                Some(pinned),
                ruled,
                Stated::nothing,
            );
            let placed = placed.expect("the wire's other worker");
            assert_eq!(placed.url(), "http://v:1");
            let placed = select_from(
                &registry,
                &policies,
                MODEL,
                pool,
                &two,
                ruled,
                Stated::nothing,
            );
            let placed = placed.expect("the slice's other worker");
            assert_eq!(placed.url(), "http://s:1");
        }

        let whole: Vec<Arc<dyn Worker>> = registry.get_routing_pool(MODEL, pool).to_vec();
        let mut turns = BTreeMap::new();
        for _ in 0..9 {
            let placed = select_from(
                &registry,
                &policies,
                MODEL,
                pool,
                &whole,
                inputs,
                Stated::nothing,
            );
            let placed = placed.expect("a ready candidate");
            *turns.entry(placed.url().to_owned()).or_insert(0) += 1;
        }
        assert_eq!(turns.into_values().collect::<Vec<_>>(), [3, 3, 3]);

        for worker in &registry.get_all() {
            worker.set_status(WorkerStatus::NotReady);
        }
        assert!(select_single(
            &registry,
            &policies,
            MODEL,
            pool,
            None,
            inputs,
            Stated::nothing
        )
        .is_none());
    }

    /// A part of one worker in a pool of sixteen is found at every request:
    /// where the request's rule refuses the engine's own pick, the engine
    /// asks the rule of the whole list and places within what it admits.
    #[test]
    fn the_engine_finds_a_part_of_one_worker_in_a_pool_of_sixteen() {
        let urls: Vec<String> = (1..=16).map(|at| format!("http://h:{at}")).collect();
        let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
        let registry = http_registry(&urls);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        preset_on(&policies, "round_robin");
        let pool = RoutingPool::HttpRegular;
        let one = [registry.get_by_url("http://h:11").expect("registered")];
        for _ in 0..16 {
            let inputs = PlacementInputs::default();
            let placed = select_from(
                &registry,
                &policies,
                MODEL,
                pool,
                &one,
                inputs,
                Stated::nothing,
            );
            let placed = placed.expect("the part's one worker");
            assert_eq!(placed.url(), "http://h:11");
        }
    }

    /// The requests a ring places, each as the one header that keys it: the
    /// routing key header for a session, and the two headers that tell one
    /// client from another.
    fn ring_keyed() -> Vec<HeaderMap> {
        (0..16)
            .flat_map(|at| {
                [
                    ("x-smg-routing-key", format!("session-{at}")),
                    ("x-forwarded-for", format!("10.0.0.{at}")),
                    ("cookie", format!("sid={at}")),
                ]
            })
            .map(|(name, value)| {
                let mut headers = HeaderMap::new();
                headers.insert(name, value.parse().expect("a header value"));
                headers
            })
            .collect()
    }

    /// The classic ring policy's registry beside one whose engine decides
    /// under the stock preset of the same name.
    fn ring_policies() -> (PolicyRegistry, PolicyRegistry) {
        // The engine's registry as a gateway makes it where the policy
        // controller runs: it makes no classic policy object.
        let engine = PolicyRegistry::for_policy_engine(
            PolicyConfig::ConsistentHashing,
            RoutingKeyOverrideConfig::default(),
        );
        preset_on(&engine, "consistent_hashing");
        (PolicyRegistry::new(PolicyConfig::ConsistentHashing), engine)
    }

    /// Under the engine a request a ring places goes to the worker the
    /// classic ring policy sends it to
    /// (`ConsistentHashingPolicy::find_by_consistent_hash`): the gateway
    /// lends the engine the model's hash ring with the request, and the
    /// engine walks it from the place the ring gives the request's key. So
    /// does the request's retry, which leaves the worker it tried for the
    /// next one the ring reaches.
    #[test]
    fn the_engine_places_a_keyed_request_where_the_classic_ring_does() {
        let urls: Vec<String> = (1..=7).map(|at| format!("http://h:{at}")).collect();
        let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
        let registry = http_registry(&urls);
        let (classic, engine) = ring_policies();

        let mut owners = BTreeMap::new();
        for headers in ring_keyed() {
            let inputs = with_headers(&headers);
            let owner = pick(&registry, &classic, inputs);
            let placed = pick(&registry, &engine, inputs);
            assert_eq!(placed.url(), owner.url(), "{headers:?}");
            *owners.entry(owner.url().to_owned()).or_insert(0) += 1;

            let tried = vec![owner.base_url().to_string()];
            let retry = PlacementInputs {
                tried: &tried,
                ..inputs
            };
            let next = pick(&registry, &classic, retry);
            assert_ne!(next.url(), owner.url(), "{headers:?}");
            let placed = pick(&registry, &engine, retry);
            assert_eq!(placed.url(), next.url(), "{headers:?}, its retry");
        }
        assert!(owners.len() > 3, "the keys spread: {owners:?}");
        assert!(registry.get_routing_snapshot(MODEL).hash_ring_is_filled());
        // The model's ring was read from the registry's kept rings: no
        // classic policy object was made for it, and none was asked.
        assert_eq!(engine.classic_made_or_asked(), 0);
    }

    /// The ring is the model's whatever part of a pool a request is held
    /// to, as it is for a classic ring policy over such a list: a caller's
    /// own slice in an order of its own, and the part of a pool on a retry's
    /// pinned wire. A worker of the ring that the part does not hold is
    /// passed over.
    #[test]
    fn the_engine_walks_the_models_ring_over_a_part_of_a_pool() {
        let registry = registry_with(&[
            ("http://s:1", ConnectionMode::Http, RuntimeType::Sglang),
            ("http://v:1", ConnectionMode::Http, RuntimeType::Vllm),
            ("http://s:2", ConnectionMode::Http, RuntimeType::Sglang),
            ("http://v:2", ConnectionMode::Http, RuntimeType::Vllm),
            ("http://v:3", ConnectionMode::Http, RuntimeType::Vllm),
            ("http://s:3", ConnectionMode::Http, RuntimeType::Sglang),
        ]);
        let (classic, engine) = ring_policies();
        let slice: Vec<Arc<dyn Worker>> = ["http://v:3", "http://s:1", "http://v:1", "http://s:3"]
            .iter()
            .map(|url| registry.get_by_url(url).expect("registered"))
            .collect();
        let pinned = WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: ConnectionMode::Http,
            requires_media_refs: false,
            pipeline: PipelinePin::Any,
        };

        let (mut from_slice, mut on_wire) = (BTreeMap::new(), BTreeMap::new());
        for headers in ring_keyed() {
            let inputs = with_headers(&headers);
            let pool = RoutingPool::HttpRegular;
            let of = |policies: &PolicyRegistry| {
                select_from(
                    &registry,
                    policies,
                    MODEL,
                    pool,
                    &slice,
                    inputs,
                    Stated::nothing,
                )
                .expect("a ready candidate")
            };
            let owner = of(&classic);
            assert_eq!(of(&engine).url(), owner.url(), "{headers:?}, a slice");
            *from_slice.entry(owner.url().to_owned()).or_insert(0) += 1;

            let on = |policies: &PolicyRegistry| {
                select_single(
                    &registry,
                    policies,
                    MODEL,
                    pool,
                    Some(pinned),
                    inputs,
                    Stated::nothing,
                )
                .expect("a ready worker on the wire")
            };
            let owner = on(&classic);
            assert_eq!(on(&engine).url(), owner.url(), "{headers:?}, a wire");
            *on_wire.entry(owner.url().to_owned()).or_insert(0) += 1;
        }
        assert!(from_slice.len() > 2, "the keys spread: {from_slice:?}");
        assert!(on_wire.len() > 2, "the keys spread: {on_wire:?}");
    }

    /// A model's first requests, before a ring is kept for it, go where the
    /// classic ring policy sends them with no ring: to the worker at the
    /// key's position over the eligible workers. The engine is lent a ring
    /// that holds no worker and gives that position
    /// ([`Unfilled`](crate::policies::hash_ring::Unfilled)). The task that
    /// fills the rings lives here and never runs: nothing drives its
    /// runtime.
    #[test]
    fn the_engine_places_a_models_first_requests_as_with_no_ring() {
        let urls: Vec<String> = (1..=7).map(|at| format!("http://h:{at}")).collect();
        let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
        let registry = Arc::new(http_registry(&urls));
        let (classic, engine) = ring_policies();
        let runtime = tokio::runtime::Builder::new_current_thread().build();
        let runtime = runtime.expect("a runtime");
        classic.keep_rings_warm(&registry, runtime.handle());
        engine.keep_rings_warm(&registry, runtime.handle());

        let mut owners = BTreeMap::new();
        for headers in ring_keyed() {
            let inputs = with_headers(&headers);
            let owner = pick(&registry, &classic, inputs);
            let placed = pick(&registry, &engine, inputs);
            assert_eq!(placed.url(), owner.url(), "{headers:?}");
            *owners.entry(owner.url().to_owned()).or_insert(0) += 1;
        }
        assert!(owners.len() > 3, "the keys spread: {owners:?}");
        assert!(
            !registry.get_routing_snapshot(MODEL).hash_ring_is_filled(),
            "no request fills a ring while the task lives"
        );
    }

    /// The classic `prefix_hash` policy's registry beside one whose engine
    /// decides under the document the same settings mean, with the cache
    /// boundaries given.
    fn prefix_policies(cache_boundaries: &[usize]) -> (PolicyRegistry, PolicyRegistry) {
        use crate::config::RouterConfig;

        let settings = RouterConfig {
            policy: PolicyConfig::PrefixHash {
                prefix_token_count: 256,
                load_factor: 1.25,
                balance_abs_threshold: 10,
                cache_boundaries: cache_boundaries.to_vec(),
            },
            ..Default::default()
        };
        let engine = PolicyRegistry::for_policy_engine(
            settings.policy.clone(),
            settings.routing_key_override.clone(),
        );
        let document = settings.policy_document().expect("the settings' document");
        let spec = AppliedSpec::validate(&document).expect("a valid document");
        engine_on(&engine, spec);
        (PolicyRegistry::new(settings.policy), engine)
    }

    /// A request the `prefix_hash` policy places by a ring: its name, its
    /// token ids, its text and its headers.
    type Prefixed = (String, Option<Vec<u32>>, Option<String>, HeaderMap);

    /// The requests the `prefix_hash` policy places by a ring. Prompts of
    /// token ids: the policy's own recorded set of 70 to 400, and four
    /// around the first boundary. Prompts of text alone, measured at four
    /// characters a token, around both boundaries. A routing key over a
    /// prompt, which the key leads.
    fn prefix_placed() -> Vec<Prefixed> {
        let mut requests = Vec::new();
        for p in 0..32u32 {
            let len = 70 + (p * 37) % 331;
            let step = 1 + p % 3;
            let tokens: Vec<u32> = (0..len).map(|i| (p / 4) * 1_000 + i * step).collect();
            requests.push((
                format!("{len} tokens"),
                Some(tokens),
                None,
                HeaderMap::new(),
            ));
        }
        for len in [5u32, 40, 63, 64] {
            let tokens: Vec<u32> = (0..len).map(|i| 9_000 + i).collect();
            requests.push((
                format!("{len} tokens"),
                Some(tokens),
                None,
                HeaderMap::new(),
            ));
        }
        let lengths = [100usize, 255, 256, 300, 1_023, 1_024, 1_500];
        for (at, len) in lengths.into_iter().enumerate() {
            let letters = "abcdefghijklmnopqrstuvwxy\u{e9}".chars().cycle();
            let text: String = letters.skip(at).take(len).collect();
            requests.push((
                format!("a text of {len}"),
                None,
                Some(text),
                HeaderMap::new(),
            ));
        }
        for at in 0..6 {
            let mut headers = HeaderMap::new();
            let key = format!("session-{at}").parse().expect("a header value");
            headers.insert("x-smg-routing-key", key);
            let tokens: Vec<u32> = (0..300).collect();
            requests.push((format!("key {at}"), Some(tokens), None, headers));
        }
        requests
    }

    /// Under the engine a request the `prefix_hash` policy places by a
    /// ring goes to the worker that policy sends it to. The gateway's ring
    /// answers the engine where each head of the request stands, by the
    /// policy's own recipe (`PrefixHashPolicy::head_position`), and the
    /// engine cuts the request into the heads the policy cuts it into: a
    /// prompt at each cache boundary it reaches, else at the policy's
    /// depth, and a routing key whole. So does the request's retry, which
    /// leaves the worker it tried for the next one the ring reaches, and a
    /// request placed while three workers are not ready. With and without
    /// cache boundaries.
    #[test]
    fn the_engine_places_a_prompt_where_the_classic_prefix_ring_does() {
        let urls: Vec<String> = (0..16).map(|at| format!("http://w{at:02}:8000")).collect();
        let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
        let down = ["http://w03:8000", "http://w07:8000", "http://w11:8000"];
        for boundaries in [&[64usize, 256][..], &[][..]] {
            let registry = http_registry(&urls);
            let (classic, engine) = prefix_policies(boundaries);
            let compared = |round: &str| {
                let mut owners = BTreeMap::new();
                for (name, tokens, text, headers) in prefix_placed() {
                    let inputs = PlacementInputs {
                        tokens: tokens.as_deref(),
                        text: text.as_deref(),
                        headers: Some(&headers),
                        ..Default::default()
                    };
                    let owner = pick(&registry, &classic, inputs);
                    let placed = pick(&registry, &engine, inputs);
                    assert_eq!(placed.url(), owner.url(), "{name}, {round}");
                    *owners.entry(owner.url().to_owned()).or_insert(0) += 1;

                    let tried = vec![owner.base_url().to_string()];
                    let retry = PlacementInputs {
                        tried: &tried,
                        ..inputs
                    };
                    let next = pick(&registry, &classic, retry);
                    assert_ne!(next.url(), owner.url(), "{name}, {round}");
                    let placed = pick(&registry, &engine, retry);
                    assert_eq!(placed.url(), next.url(), "{name}, {round}, its retry");
                }
                owners
            };
            let owners = compared("every worker ready");
            assert!(owners.len() > 8, "the requests spread: {owners:?}");
            for url in down {
                let worker = registry.get_by_url(url).expect("registered");
                worker.set_status(WorkerStatus::NotReady);
            }
            let owners = compared("three workers not ready");
            assert!(owners.len() > 8, "the requests spread: {owners:?}");
            assert!(down.iter().all(|url| !owners.contains_key(*url)));
            assert!(registry.get_routing_snapshot(MODEL).hash_ring_is_filled());
            // The model's ring was read from the registry's kept rings: no
            // classic policy object was made for it, and none was asked.
            assert_eq!(engine.classic_made_or_asked(), 0);
        }
    }

    /// A request the `prefix_hash` policy places keeps classic's worker
    /// through a change of the model's workers: one that leaves, one that
    /// joins, and one replaced by a new worker at its address. The engine
    /// walks the ring the gateway keeps for the model, so after each change
    /// it is lent the ring the classic policy reads. A request whose worker
    /// stayed keeps it, a new worker takes requests from the others and
    /// moves none between them, and a worker replaced at its address hands
    /// its requests to the new one.
    #[test]
    fn the_engine_follows_the_classic_prefix_ring_through_a_membership_change() {
        let urls: Vec<String> = (0..16).map(|at| format!("http://w{at:02}:8000")).collect();
        let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
        let registry = http_registry(&urls);
        let (classic, engine) = prefix_policies(&[64, 256]);
        let register = |url: &str| {
            let worker = BasicWorkerBuilder::new(url)
                .model(ModelCard::new(MODEL))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .runtime_type(RuntimeType::Sglang)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build();
            registry
                .register(Arc::new(worker))
                .expect("worker registers");
        };
        // Each request's worker under the engine, which is classic's.
        let placed = |round: &str| -> Vec<Arc<dyn Worker>> {
            let place = |(name, tokens, text, headers): Prefixed| {
                let inputs = PlacementInputs {
                    tokens: tokens.as_deref(),
                    text: text.as_deref(),
                    headers: Some(&headers),
                    ..Default::default()
                };
                let owner = pick(&registry, &classic, inputs);
                let by_engine = pick(&registry, &engine, inputs);
                assert_eq!(by_engine.url(), owner.url(), "{name}, {round}");
                assert!(Arc::ptr_eq(&by_engine, &owner), "{name}, {round}");
                by_engine
            };
            prefix_placed().into_iter().map(place).collect()
        };
        let first = placed("sixteen workers");

        // The first request's worker leaves: its requests move, no other's.
        let left = first[0].url().to_owned();
        registry.remove_by_url(&left).expect("registered");
        let without = placed("one worker left");
        for (before, after) in first.iter().zip(&without) {
            assert_ne!(after.url(), left);
            let stays = before.url() == left || Arc::ptr_eq(after, before);
            assert!(stays, "{} to {}", before.url(), after.url());
        }

        let joined = "http://w16:8000";
        register(joined);
        let with = placed("one worker joined");
        let takes = with.iter().filter(|after| after.url() == joined).count();
        assert!(takes > 0, "the new worker takes a request");
        for (before, after) in without.iter().zip(&with) {
            let stays = after.url() == joined || Arc::ptr_eq(after, before);
            assert!(stays, "{} to {}", before.url(), after.url());
        }

        // A new worker at the second request's address: each request stays
        // at its address, and that address's go to the new worker.
        let old = registry.remove_by_url(with[1].url()).expect("registered");
        register(old.url());
        let again = placed("one worker replaced at its address");
        for (before, after) in with.iter().zip(&again) {
            assert_eq!(after.url(), before.url());
            let kept = Arc::ptr_eq(after, before);
            assert_eq!(kept, !Arc::ptr_eq(before, &old), "{}", before.url());
        }
        assert_eq!(engine.classic_made_or_asked(), 0);
    }

    /// The bounded-load rule under the engine is the classic policy's. A
    /// prompt whose deepest head's owner holds more requests than the bound
    /// goes to the owner of its next head: classic tries the next boundary's
    /// key, and the engine the next place its request states. Where every
    /// head's owner is past the bound, it goes to the worker with the
    /// fewest requests, which the loads here leave one of.
    #[test]
    fn the_engine_steps_past_a_loaded_owner_as_the_classic_prefix_ring_does() {
        let urls: Vec<String> = (0..16).map(|at| format!("http://w{at:02}:8000")).collect();
        let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
        let idle = "http://w09:8000";
        let mut stepped = 0;
        for (name, tokens, _, headers) in prefix_placed() {
            // A prompt that reaches both boundaries has two heads, and no
            // routing key leads it.
            let Some(tokens) = tokens.filter(|tokens| tokens.len() >= 256 && headers.is_empty())
            else {
                continue;
            };
            let registry = http_registry(&urls);
            let (classic, engine) = prefix_policies(&[64, 256]);
            let placed = |tokens: &[u32], round: &str| {
                let inputs = PlacementInputs {
                    tokens: Some(tokens),
                    ..Default::default()
                };
                let owner = pick(&registry, &classic, inputs);
                let by_engine = pick(&registry, &engine, inputs);
                assert!(Arc::ptr_eq(&by_engine, &owner), "{name}, {round}");
                owner
            };
            let load = |worker: &Arc<dyn Worker>, requests: usize| {
                for _ in worker.load()..requests {
                    worker.increment_load();
                }
            };
            // With no request in flight the deepest head's owner takes the
            // prompt, and its first boundary's owner takes that head alone.
            let owner = placed(&tokens, "no load");
            let next = placed(&tokens[..64], "the first boundary alone");
            if Arc::ptr_eq(&owner, &next) || owner.url() == idle || next.url() == idle {
                continue;
            }
            // Eleven requests on the owner are past the bound of sixteen
            // workers: the mean with this request, and ten.
            load(&owner, 11);
            let after = placed(&tokens, "the deepest head's owner past the bound");
            assert!(Arc::ptr_eq(&after, &next), "{name}: {}", after.url());
            // Both owners past the bound, and every other worker but one
            // holds a request.
            for worker in &registry.get_all() {
                let own = Arc::ptr_eq(worker, &owner) || Arc::ptr_eq(worker, &next);
                load(
                    worker,
                    if own {
                        13
                    } else {
                        usize::from(worker.url() != idle)
                    },
                );
            }
            let after = placed(&tokens, "every head's owner past the bound");
            assert_eq!(after.url(), idle, "{name}");
            assert_eq!(engine.classic_made_or_asked(), 0);
            stepped += 1;
        }
        assert!(stepped >= 8, "{stepped} prompts stepped past their owner");
    }

    /// A pair under the engine is placed where the classic `prefix_hash`
    /// policies place its two sides. The gateway lends a pair's decision
    /// the model's ring as it lends a single request's, and the ring
    /// answers for the pair's heads too: each side is found among its own
    /// role's workers from the place the classic policy of that role
    /// starts at.
    #[test]
    fn the_engine_places_a_pair_where_the_classic_prefix_ring_does() {
        use crate::{
            config::{RouterConfig, RoutingMode},
            policies::PolicyFactory,
        };

        let prefix = PolicyConfig::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: 10,
            cache_boundaries: vec![64, 256],
        };
        let settings = RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: Some(prefix.clone()),
                decode_policy: Some(prefix.clone()),
            },
            policy: prefix.clone(),
            ..Default::default()
        };
        let fleet: Vec<(String, WorkerType)> = (0..8)
            .flat_map(|at| {
                [
                    (format!("grpc://p{at}:1"), WorkerType::Prefill),
                    (format!("grpc://d{at}:1"), WorkerType::Decode),
                ]
            })
            .collect();
        let listed: Vec<_> = fleet
            .iter()
            .map(|(url, role)| (url.as_str(), *role, Some("NixlConnector")))
            .collect();
        let registry = pd_registry(&listed);
        let classic = PolicyRegistry::new(prefix.clone());
        classic.set_prefill_policy(PolicyFactory::create_from_config(&prefix));
        classic.set_decode_policy(PolicyFactory::create_from_config(&prefix));
        let engine = PolicyRegistry::for_policy_engine(
            settings.policy.clone(),
            settings.routing_key_override.clone(),
        );
        let document = settings.policy_document().expect("the settings' document");
        let spec = AppliedSpec::validate(&document).expect("a valid document");
        engine_on(&engine, spec);
        let placed = |policies: &PolicyRegistry, inputs: PlacementInputs<'_>| {
            let pairs = pairs_of(&registry, policies);
            let pair = select_pair(
                &registry,
                policies,
                MODEL,
                &pairs,
                None,
                None,
                inputs,
                Stated::nothing,
            );
            let pair = pair.expect("a pair is open");
            (pair.prefill.url().to_owned(), pair.decode.url().to_owned())
        };
        let mut sides = std::collections::BTreeSet::new();
        for (name, tokens, text, headers) in prefix_placed() {
            let inputs = PlacementInputs {
                tokens: tokens.as_deref(),
                text: text.as_deref(),
                headers: Some(&headers),
                ..Default::default()
            };
            let by_classic = placed(&classic, inputs);
            assert_eq!(placed(&engine, inputs), by_classic, "{name}");
            sides.insert(by_classic);
        }
        assert!(sides.len() > 8, "the pairs spread: {sides:?}");
        assert!(registry.get_routing_snapshot(MODEL).hash_ring_is_filled());
        assert_eq!(engine.classic_made_or_asked(), 0);
    }

    /// A model's first prompts, before a ring is kept for it, go where the
    /// `prefix_hash` policy sends them with no ring: to the worker with the
    /// fewest requests. The engine is lent a ring that holds no worker and
    /// places a head ([`Unfilled`](crate::policies::hash_ring::Unfilled)),
    /// so its ring stage ranks nobody and the fewest requests decide. The
    /// task that fills the rings lives here and never runs: nothing drives
    /// its runtime.
    #[test]
    fn the_engine_places_a_models_first_prompts_by_the_fewest_requests() {
        let urls: Vec<String> = (0..8).map(|at| format!("http://w{at:02}:8000")).collect();
        let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
        let registry = Arc::new(http_registry(&urls));
        let (classic, engine) = prefix_policies(&[64, 256]);
        let runtime = tokio::runtime::Builder::new_current_thread().build();
        let runtime = runtime.expect("a runtime");
        classic.keep_rings_warm(&registry, runtime.handle());
        engine.keep_rings_warm(&registry, runtime.handle());
        // Every worker but one holds a request.
        let idle = "http://w05:8000";
        for worker in &registry.get_all() {
            if worker.url() != idle {
                worker.increment_load();
            }
        }

        for (name, tokens, text, headers) in prefix_placed() {
            let inputs = PlacementInputs {
                tokens: tokens.as_deref(),
                text: text.as_deref(),
                headers: Some(&headers),
                ..Default::default()
            };
            assert_eq!(pick(&registry, &classic, inputs).url(), idle, "{name}");
            assert_eq!(pick(&registry, &engine, inputs).url(), idle, "{name}");
        }
        assert!(
            !registry.get_routing_snapshot(MODEL).hash_ring_is_filled(),
            "no request fills a ring while the task lives"
        );
        assert_eq!(engine.classic_made_or_asked(), 0);
    }

    /// The engine's request reads the model's ring only under a document
    /// that places by one, as a classic placement reads it only for a
    /// policy that walks it: a ring nobody walks is not filled.
    #[test]
    fn the_engine_reads_the_ring_under_a_document_that_places_by_one_alone() {
        let fills_the_ring = |preset: &str| {
            let registry = http_registry(&["http://h:1", "http://h:2"]);
            let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
            preset_on(&policies, preset);
            let headers = keyed_headers();
            pick(&registry, &policies, with_headers(&headers));
            registry.get_routing_snapshot(MODEL).hash_ring_is_filled()
        };
        assert!(!fills_the_ring("round_robin"));
        assert!(fills_the_ring("consistent_hashing"));
    }

    /// The selection metric's lines for [`MODEL`] as a recorder renders
    /// them: each line's policy label with its count.
    fn selections_by_policy(rendered: &str) -> BTreeMap<String, u64> {
        let label = "policy=\"";
        let mut counted = BTreeMap::new();
        for line in rendered.lines() {
            if !line.starts_with("smg_worker_selection_total{") {
                continue;
            }
            let (labels, count) = line.rsplit_once(' ').expect("a sample's value");
            let from = labels.find(label).expect("the policy label") + label.len();
            let policy = labels[from..].split('"').next().expect("the label's value");
            let count: u64 = count.parse().expect("a count");
            *counted.entry(policy.to_owned()).or_insert(0) += count;
        }
        counted
    }

    /// The proof's two requests beside the count of classic policy objects:
    /// a keyed request under a ring policy, and a request whose text the
    /// document reads. The registry is the one a gateway makes where the
    /// policy controller runs, and the engine's document is the one the
    /// same settings mean. Each selection is counted under the name the
    /// settings give the policy, which is the label of the profile their
    /// document makes of it. The text questions are the document's. No
    /// classic policy object was made, and none was asked to place.
    #[test]
    fn a_keyed_request_and_a_request_whose_text_is_read_ask_no_classic_policy() {
        use crate::config::RouterConfig;

        let cache_aware = PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096);
        for (policy, reads_text) in [
            (PolicyConfig::ConsistentHashing, false),
            (cache_aware, true),
        ] {
            let name = policy.name();
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
            let settings = RouterConfig {
                policy,
                ..Default::default()
            };
            let policies = PolicyRegistry::for_policy_engine(
                settings.policy.clone(),
                settings.routing_key_override.clone(),
            );
            let document = settings.policy_document().expect("the settings' document");
            let spec = AppliedSpec::validate(&document);
            engine_on(&policies, spec.expect("a valid document"));
            assert_eq!(
                policies.any_policy_needs_request_text(None),
                reads_text,
                "{name}"
            );
            assert_eq!(policies.pair_needs_request_text(), reads_text, "{name}");

            let mut headers = HeaderMap::new();
            headers.insert("x-smg-routing-key", HeaderValue::from_static("session-A"));
            let inputs = PlacementInputs {
                text: Some("the words of a prompt its route read for the placement"),
                headers: Some(&headers),
                ..Default::default()
            };
            metrics::with_local_recorder(&recorder, || {
                let first = pick(&registry, &policies, inputs);
                let again = pick(&registry, &policies, inputs);
                assert_eq!(again.url(), first.url(), "{name}: the key's worker");
            });
            let counted = selections_by_policy(&handle.render());
            assert_eq!(counted, BTreeMap::from([(name.to_owned(), 2)]), "{name}");
            assert_eq!(policies.classic_made_or_asked(), 0, "{name}");
        }
    }

    // ---- a traced request under the policy engine --------------------------------------------

    /// One row of a decision record as these cases read it: its policy, its
    /// origin, its worker (empty where the row names none) and its
    /// candidates.
    type TracedRow = (String, String, String, Vec<String>);

    /// The rows of the decision record in scope.
    fn traced_rows() -> Vec<TracedRow> {
        let text = |value: &serde_json::Value| value.as_str().unwrap_or_default().to_owned();
        let rows = cache_trace::captured_selections();
        rows.iter()
            .map(|row| {
                let candidates = row["candidates"].as_array().into_iter().flatten();
                let candidates = candidates.map(|candidate| text(&candidate["worker"]));
                (
                    text(&row["policy"]),
                    text(&row["origin"]),
                    text(&row["worker"]),
                    candidates.collect(),
                )
            })
            .collect()
    }

    /// A row the policy engine's placement writes: under `policy`, with
    /// `worker` picked from `candidates`.
    fn by_engine(policy: &str, worker: &str, candidates: &[&str]) -> TracedRow {
        let candidates = candidates.iter().map(|&url| url.to_owned()).collect();
        (
            policy.to_owned(),
            "engine".to_owned(),
            worker.to_owned(),
            candidates,
        )
    }

    /// The registry of a gateway that was supplied `document` beside its
    /// default settings, as its start makes it of that configuration
    /// (`PolicyRegistry::of_config`); `pair` makes the configuration a pair
    /// gateway's.
    fn supplied_with(document: &PolicySpec, pair: bool) -> PolicyRegistry {
        use crate::config::{RouterConfig, RoutingMode};

        let mut config = RouterConfig {
            policy_controller: true,
            policy_spec: Some(document.clone()),
            ..Default::default()
        };
        if pair {
            config.mode = RoutingMode::PrefillDecode {
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                prefill_policy: None,
                decode_policy: None,
            };
        }
        let policies = PolicyRegistry::of_config(&config);
        assert!(policies.runs_supplied_document());
        policies
    }

    /// A traced single request the policy engine placed records one row, as
    /// a classic selection does (`PolicyRegistry::select_worker_for_model`,
    /// `model_gateway/src/policies/registry.rs:711-716`): the pool's list as
    /// the engine was lent it, and the worker picked under the label of the
    /// profile that picked. A request the engine places nowhere records a
    /// row with no worker, where classic writes its policy's row with no
    /// worker: under the name the settings give the main policy in a
    /// gateway of settings alone, and under no name in one that was
    /// supplied its document, where the settings name nothing that runs.
    /// The rows' origin is `engine`, where classic writes `policy`, and
    /// they hold no prediction, score or gate.
    #[tokio::test]
    async fn a_traced_single_of_the_engine_records_its_pick_under_its_profiles_label() {
        let (one, two) = ("http://h:1", "http://h:2");
        let of_settings = PolicyRegistry::new(PolicyConfig::RoundRobin);
        assert!(!of_settings.runs_supplied_document());
        let document = PolicySpec::preset("round_robin").expect("a stock preset");
        let supplied = supplied_with(&document, false);
        assert_ne!(
            supplied.policy_name(),
            "round_robin",
            "the default settings"
        );
        for (policies, preset, unnamed) in [
            (of_settings, "random", "round_robin"),
            (supplied, "round_robin", ""),
        ] {
            let registry = http_registry(&[one, two]);
            preset_on(&policies, preset);
            let nobody = |_: &dyn Worker| false;
            let refused = PlacementInputs {
                candidate_filter: Some(&nobody),
                ..Default::default()
            };
            let (picked, rows, whole) = cache_trace::scope(true, async {
                let picked = pick(&registry, &policies, PlacementInputs::default());
                let pool = RoutingPool::HttpRegular;
                let stated = Stated::nothing;
                let none = select_single(&registry, &policies, MODEL, pool, None, refused, stated);
                assert!(none.is_none(), "the request's rule takes no worker");
                (picked, traced_rows(), cache_trace::captured_selections())
            })
            .await;
            assert_eq!(
                rows,
                [
                    by_engine(preset, picked.url(), &[one, two]),
                    by_engine(unnamed, "", &[one, two]),
                ],
                "{preset}"
            );
            for row in &whole {
                let stated = (&row["prediction"], &row["scores"], &row["gates"]);
                let nothing = serde_json::json!([]);
                assert_eq!(stated, (&serde_json::Value::Null, &nothing, &nothing));
            }
        }
    }

    /// A traced pair the policy engine placed records the two rows a
    /// classic pair's selections write, the prefill's and then the decode's
    /// (`PolicyRegistry::select_worker_for_model`,
    /// `model_gateway/src/policies/registry.rs:711-716`, asked once a side):
    /// each side's pool as it stood ahead of the engine's call, and the
    /// side's worker. The rows' origin is `engine`, where classic writes
    /// `policy`.
    ///
    /// In a gateway of settings alone the document is the one the settings
    /// mean, whose two profiles are named for their sides: each row is
    /// under the name the settings give its side's policy, as classic's.
    /// Under profiles with names of their own, each row is under its
    /// profile's name. In a gateway that was supplied its document, the
    /// same profiles named for their sides stand under their own labels:
    /// the settings name nothing that runs. Under a supplied document that
    /// picks the decode behind its prefill the record holds the prefill's
    /// row alone, because that decode may be decided again after the record
    /// is written.
    #[tokio::test]
    async fn a_traced_pair_of_the_engine_records_both_picks_each_under_its_sides_policy() {
        use crate::{
            config::{RouterConfig, RoutingMode},
            policies::Side as PairSide,
        };

        let settings = RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                prefill_policy: Some(PolicyConfig::Random),
                decode_policy: Some(PolicyConfig::RoundRobin),
            },
            ..Default::default()
        };
        // A gateway of these settings, its sides named as its routers name
        // them at their build.
        let of_settings = || {
            let (main, mode) = (&settings.policy, &settings.mode);
            let policies = PolicyRegistry::new(main.clone());
            policies.set_side_policy(PairSide::Prefill, mode.get_prefill_policy(main));
            policies.set_side_policy(PairSide::Decode, mode.get_decode_policy(main));
            assert!(!policies.runs_supplied_document());
            policies
        };
        let stock = settings.policy_document().expect("the settings' document");
        let mut renamed = stock.clone();
        for (side, name) in [("prefill", "first"), ("decode", "second")] {
            let profile = renamed.profiles.remove(side).expect("the side's profile");
            renamed.profiles.insert(name.to_owned(), profile);
            for binding in &mut renamed.bindings {
                if let Some(of_side) = binding.profiles.get_mut(side) {
                    name.clone_into(of_side);
                }
            }
        }
        let mut later = stock.clone();
        later.plans.pd_pair.decode = DecodePick::AfterPrefill;
        let (prefill, decode) = ("grpc://p:1", "grpc://d:1");
        let both = |names: [&str; 2]| {
            vec![
                by_engine(names[0], prefill, &[prefill]),
                by_engine(names[1], decode, &[decode]),
            ]
        };
        for (document, policies, expected) in [
            (&stock, of_settings(), both(["random", "round_robin"])),
            (&renamed, of_settings(), both(["first", "second"])),
            (
                &stock,
                supplied_with(&stock, true),
                both(["prefill", "decode"]),
            ),
            (
                &later,
                supplied_with(&later, true),
                vec![by_engine("prefill", prefill, &[prefill])],
            ),
        ] {
            let registry = one_pair();
            let spec = AppliedSpec::validate(document);
            engine_on(&policies, spec.expect("a valid document"));
            registry
                .get_by_url(prefill)
                .expect("registered")
                .increment_load();
            let (rows, whole) = cache_trace::scope(true, async {
                let pair = pair_from(&registry, &policies).expect("the engine places the pair");
                assert_eq!((pair.prefill.url(), pair.decode.url()), (prefill, decode));
                (traced_rows(), cache_trace::captured_selections())
            })
            .await;
            assert_eq!(rows, expected);
            let loads = whole.iter().map(|row| row["candidates"][0]["load"].clone());
            let loads: Vec<_> = loads.collect();
            assert_eq!(loads, [1, 0][..expected.len()], "each pool as it stood");
        }
    }

    /// A traced pair the policy engine did not place records one row, with
    /// no worker: the pool of the side judged to have nobody the request
    /// may use. In a gateway of settings alone it is under the name the
    /// settings give that side's policy, and in one that was supplied its
    /// document under no name. A classic pair whose decode selection names
    /// nobody leaves the prefill's row and then the decode's
    /// (`model_gateway/src/policies/registry.rs:711-716`, once a side): the
    /// engine's call names no prefill where it names no pair.
    #[tokio::test]
    async fn a_traced_pair_the_engine_did_not_place_records_the_side_that_had_nobody() {
        let (prefill, decode) = ("grpc://p:1", "grpc://d:1");
        let document = PolicySpec::pair_presets("random", "round_robin");
        let document = document.expect("the pair presets");
        let of_settings = PolicyRegistry::new(PolicyConfig::RoundRobin);
        of_settings.set_side_policy(crate::policies::Side::Prefill, &PolicyConfig::Random);
        for (policies, names) in [
            (of_settings, ["round_robin", "random"]),
            (supplied_with(&document, true), ["", ""]),
        ] {
            let registry = one_pair();
            let spec = AppliedSpec::validate(&document);
            engine_on(&policies, spec.expect("a valid document"));
            let no_decode = |worker: &dyn Worker| worker.url() != decode;
            let no_prefill = |worker: &dyn Worker| worker.url() != prefill;
            let refused = |rule: CandidateFilter<'_>| {
                let inputs = PlacementInputs {
                    candidate_filter: Some(rule),
                    ..Default::default()
                };
                let pairs = pairs_of(&registry, &policies);
                let stated = Stated::nothing;
                let placed = select_pair(
                    &registry, &policies, MODEL, &pairs, None, None, inputs, stated,
                );
                placed.err().expect("the engine places no pair").role
            };
            let (roles, rows) = cache_trace::scope(true, async {
                let roles = [refused(&no_decode), refused(&no_prefill)];
                (roles, traced_rows())
            })
            .await;
            assert_eq!(roles, [WorkerRole::Decode, WorkerRole::Prefill]);
            assert_eq!(
                rows,
                [
                    by_engine(names[0], "", &[decode]),
                    by_engine(names[1], "", &[prefill]),
                ],
                "{names:?}"
            );
        }
    }

    // ---- a pair under the policy engine ------------------------------------------------------

    /// One prefill and one decode on one KV transport.
    pub(crate) fn one_pair() -> WorkerRegistry {
        pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ])
    }

    /// Hands `policies` an engine under the stock pair document with
    /// [`Hears`] ranking its decode side, which hears the words of both
    /// sides, and what that stage heard. The document picks a pair's decode
    /// with its prefill, or behind it where `later`.
    pub(crate) fn hearing_pairs_on(policies: &PolicyRegistry, later: bool) -> Heard {
        let heard = Heard::default();
        let kept = Arc::clone(&heard);
        let mut stages = StageRegistry::builtin();
        stages
            .scorer("hears", move || Box::new(Hears(Arc::clone(&kept))))
            .expect("a free name");
        let document = PolicySpec::pair_presets("round_robin", "round_robin");
        let mut document = document.expect("the pair presets");
        document.generation = 1;
        let decode = document.profiles.get_mut("decode");
        let decode = decode.expect("the decode side's profile");
        decode.score = vec!["hears".into()];
        decode.pick = "best".into();
        if later {
            document.plans.pd_pair.decode = DecodePick::AfterPrefill;
        }
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        engine_on(policies, spec.expect("the stages hold the scorer"));
        heard
    }

    /// With an engine handed to the registry the pair placement asks it,
    /// and the engine hears each side of the pair by that side's own guard.
    /// The prefill's guard says its dispatch where it is made. The decode's
    /// is made ahead of its send: it counts the decode from then, under the
    /// route's own key where the engine names no routing value, and tells
    /// the engine nothing until the send is said. Each guard's drop is its
    /// side's end, the answers' own drop says nothing, and nothing is
    /// released.
    #[test]
    fn the_engine_hears_each_side_of_a_placed_pair_by_that_sides_guard() {
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_pairs_on(&policies, false);
        let said = || heard.lock().expect("the stage heard").clone();

        let pair = pair_from(&registry, &policies).expect("the engine places the pair");
        assert_eq!(
            (pair.prefill.url(), pair.decode.url()),
            ("grpc://p:1", "grpc://d:1")
        );
        assert_eq!(said(), ["issued", "issued"]);

        let prefill = pair.prefill.guard(Some("the route's own key"));
        assert_eq!(said()[2..], ["dispatched Prefill"]);
        let ahead = pair.decode.ahead(Some("the route's own key"));
        let counted = pair.decode.routing_key_inflight("the route's own key");
        assert_eq!((pair.decode.load(), counted), (1, 1));
        assert_eq!(
            said().len(),
            3,
            "a guard made ahead of its send says nothing"
        );
        let decode = pair.decode.sent(ahead);
        assert_eq!(pair.decode.load(), 1, "the send counts nothing more");
        assert_eq!(said()[3..], ["dispatched Decode"]);

        let workers = [&pair.prefill, &pair.decode].map(|side| Arc::clone(side.worker()));
        drop(pair);
        assert_eq!(said().len(), 4, "each side is with its guard");
        drop(prefill);
        assert_eq!(said()[4..], ["completed Cancelled"]);
        drop(decode);
        assert_eq!(said()[4..], ["completed Cancelled", "completed Cancelled"]);
        assert_eq!(workers.map(|worker| worker.load()), [0, 0]);
    }

    /// A pair of the engine that no guard was made from gives each side up
    /// where that side's answer is dropped. A decode's guard made ahead and
    /// dropped unsent says nothing of its own: the decode's answer still
    /// holds the side, and gives it up.
    #[test]
    fn a_pair_of_the_engine_that_no_guard_took_is_released_a_side_at_a_time() {
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_pairs_on(&policies, false);
        let said = || heard.lock().expect("the stage heard").clone();

        let pair = pair_from(&registry, &policies).expect("the engine places the pair");
        drop(pair.decode.ahead(None));
        assert_eq!(pair.decode.load(), 0);
        assert_eq!(said(), ["issued", "issued"]);
        let PlacedPair {
            prefill, decode, ..
        } = pair;
        drop(decode);
        assert_eq!(said()[2..], ["released Abandoned"]);
        drop(prefill);
        assert_eq!(said()[2..], ["released Abandoned", "released Abandoned"]);
    }

    /// A pair's decision the engine refuses is written at `warn` as a
    /// single request's is, with the model and the engine's reason, and the
    /// pair placement answers no pair. The stage that fails here ranks the
    /// decode side.
    #[test]
    fn a_pairs_decision_the_engine_refuses_is_written_at_warn() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let _subscriber = test_subscriber(EventLines(tracing::Level::WARN, Arc::clone(&lines)));
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let mut stages = StageRegistry::builtin();
        stages
            .scorer("faults", || Box::new(Faults))
            .expect("a free name");
        let document = PolicySpec::pair_presets("round_robin", "round_robin");
        let mut document = document.expect("the pair presets");
        document.generation = 1;
        let decode = document.profiles.get_mut("decode");
        let decode = decode.expect("the decode side's profile");
        decode.score = vec!["faults".into()];
        decode.pick = "best".into();
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        engine_on(&policies, spec.expect("the stages hold the scorer"));

        assert!(
            pair_from(&registry, &policies).is_err(),
            "a refused decision places no pair"
        );
        let lines = lines.lock().unwrap();
        let refused: Vec<_> = lines
            .iter()
            .filter(|line| line.contains("message=The policy engine refused a decision"))
            .collect();
        assert_eq!(refused.len(), 1, "{lines:?}");
        for expected in ["model_id=\"m\"", "refused=Faulted"] {
            assert!(
                refused[0].contains(expected),
                "{expected} missing from {:?}",
                refused[0]
            );
        }
    }

    /// Under a document that picks a pair's decode behind its prefill the
    /// placement books the prefill alone, and names the decode all the
    /// same. A route that asks nothing more sends that decode, and the
    /// engine books it at that send. A prefill's guard dropped ahead of
    /// that send holds its end back for it, so a stage hears the pair's
    /// words in their order.
    #[test]
    fn a_pair_whose_decode_is_picked_later_books_it_at_the_decodes_send() {
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_pairs_on(&policies, true);
        let said = || heard.lock().expect("the stage heard").clone();

        let pair = pair_from(&registry, &policies).expect("the engine places the pair");
        assert_eq!(pair.decode.url(), "grpc://d:1");
        assert_eq!(said(), ["issued"], "the prefill alone is booked");
        let prefill = pair.prefill.guard(None);
        let ahead = pair.decode.ahead(None);
        drop(prefill);
        assert_eq!(said()[1..], ["dispatched Prefill"], "its end waits");
        let decode = pair.decode.sent(ahead);
        assert_eq!(
            said()[2..],
            ["issued", "dispatched Decode", "completed Cancelled"],
            "the decode is booked at its send, and the prefill's end follows"
        );
        drop(decode);
        assert_eq!(said()[5..], ["completed Cancelled"]);
    }

    /// A retry pinned to its retained plan's wire, and its route's own rule,
    /// hold the engine's pair as they hold a classic one: both workers are
    /// on the wire, and neither is one the rule refuses.
    #[test]
    fn the_engines_pair_keeps_to_a_retrys_wire_and_to_its_routes_rule() {
        let grpc = ConnectionMode::Grpc;
        let registry = registry_of(&[
            ("grpc://p:1", WorkerType::Prefill, grpc, RuntimeType::Sglang),
            ("grpc://d:1", WorkerType::Decode, grpc, RuntimeType::Sglang),
            ("grpc://p:2", WorkerType::Prefill, grpc, RuntimeType::Vllm),
            ("grpc://d:2", WorkerType::Decode, grpc, RuntimeType::Vllm),
            ("grpc://d:3", WorkerType::Decode, grpc, RuntimeType::Vllm),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let document = PolicySpec::pair_presets("round_robin", "round_robin");
        let spec = AppliedSpec::validate(&document.expect("the pair presets"));
        engine_on(&policies, spec.expect("a valid document"));
        let pairs = pairs_of(&registry, &policies);
        let pinned = WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: grpc,
            requires_media_refs: false,
            pipeline: PipelinePin::Any,
        };
        let refuses_one = |worker: &dyn Worker| worker.url() != "grpc://d:2";
        let placed_over = |wire: Option<WireConstraint>, rule: Option<CandidateFilter<'_>>| {
            let mut seen = std::collections::BTreeSet::new();
            for _ in 0..12 {
                let inputs = PlacementInputs {
                    candidate_filter: rule,
                    ..Default::default()
                };
                let pair = select_pair(
                    &registry,
                    &policies,
                    MODEL,
                    &pairs,
                    wire,
                    None,
                    inputs,
                    Stated::nothing,
                );
                let pair = pair.expect("a pair the request may use");
                seen.insert((pair.prefill.url().to_owned(), pair.decode.url().to_owned()));
            }
            seen.into_iter().collect::<Vec<_>>()
        };
        let pair = |prefill: &str, decode: &str| (prefill.to_owned(), decode.to_owned());

        assert_eq!(
            placed_over(Some(pinned), Some(&refuses_one)),
            [pair("grpc://p:2", "grpc://d:3")]
        );
        assert_eq!(
            placed_over(Some(pinned), None),
            [
                pair("grpc://p:2", "grpc://d:2"),
                pair("grpc://p:2", "grpc://d:3")
            ]
        );
        let unpinned = placed_over(None, Some(&refuses_one));
        assert!(
            unpinned.contains(&pair("grpc://p:1", "grpc://d:1")),
            "{unpinned:?}"
        );
        assert!(
            unpinned.iter().all(|(_, decode)| decode != "grpc://d:2"),
            "{unpinned:?}"
        );
    }

    /// A route's rule that takes one worker of each pool finds that pair.
    /// The engine's call names twelve pairs at most, four a decision: where
    /// none of them holds the rule's two workers, the pair is decided once
    /// more among the workers the rule takes, a stalled prefill or decode
    /// among them. That ask answers as a first one does: with the rule's
    /// prefill at its admission limit the request waits for a slot. A rule
    /// whose decode is not ready takes no second ask: the first answer
    /// stands.
    #[test]
    fn the_engines_pair_finds_the_one_decode_its_routes_rule_takes() {
        let grpc = ConnectionMode::Grpc;
        let side = |name: &str| -> Vec<String> {
            (0..13).map(|at| format!("grpc://{name}:{at}")).collect()
        };
        let (prefills, decodes) = (side("p"), side("d"));
        let mut workers: Vec<_> = prefills
            .iter()
            .map(|url| (url.as_str(), WorkerType::Prefill, grpc, RuntimeType::Sglang))
            .collect();
        workers.extend(
            decodes
                .iter()
                .map(|url| (url.as_str(), WorkerType::Decode, grpc, RuntimeType::Sglang)),
        );
        let registry = registry_of(&workers);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let document = PolicySpec::pair_presets("passthrough", "passthrough");
        let spec = AppliedSpec::validate(&document.expect("the pair presets"));
        engine_on(&policies, spec.expect("a valid document"));
        let pairs = pairs_of(&registry, &policies);
        let (prefill, decode) = (&pairs.prefill[12], &pairs.decode_pool[12]);
        let takes = |worker: &dyn Worker| [prefill.url(), decode.url()].contains(&worker.url());
        let inputs = PlacementInputs {
            candidate_filter: Some(&takes),
            ..Default::default()
        };
        let pick = |limit: Option<&PrefillSelectionContext<'_>>| {
            select_pair(
                &registry,
                &policies,
                MODEL,
                &pairs,
                None,
                limit,
                inputs,
                Stated::nothing,
            )
        };
        let placed = pick(None).ok().map(|pair| pair.decode.url().to_owned());
        assert_eq!(placed.as_deref(), Some(decode.url()));

        // A stalled prefill is found too, and so is a stalled decode: the
        // engine lifts the liveness veto of a side whose workers are all
        // stalled.
        let placed = || {
            let pair = pick(None).ok()?;
            Some((pair.prefill.url().to_owned(), pair.decode.url().to_owned()))
        };
        let expected = Some((prefill.url().to_owned(), decode.url().to_owned()));
        prefill.set_stall(Some(StallReason::Wedged));
        assert_eq!(placed(), expected);
        prefill.set_stall(None);
        decode.set_stall(Some(StallReason::Wedged));
        assert_eq!(placed(), expected);
        decode.set_stall(None);

        let admission = PrefillAdmission::new(1, 0, Duration::ZERO);
        let runtime = tokio::runtime::Builder::new_current_thread().build();
        let runtime = runtime.expect("a runtime");
        // Which side refused the request, and whether it waits for a slot.
        let refused = || {
            let mut failure = None;
            let asked = admission.admit(None, |limit: &PrefillSelectionContext<'_>| {
                failure = pick(Some(limit)).err();
                PrefillAdmissionAttempt::<()>::Unavailable
            });
            assert!(runtime.block_on(asked).is_err());
            failure.map(|failure| (failure.role, failure.is_at_capacity()))
        };
        prefill.increment_load();
        assert_eq!(refused(), Some((WorkerRole::Prefill, true)));
        decode.set_status(WorkerStatus::NotReady);
        assert_eq!(refused(), Some((WorkerRole::Decode, false)));
    }

    /// A guard sent on another answer than the one it was made ahead of
    /// stops a debug build: it counts another worker than the answer's.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "a guard is sent on the answer it was made ahead of")]
    fn a_guard_sent_on_another_answer_stops_a_debug_build() {
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let pair = pair_from(&registry, &policies).expect("a classic pair");
        let ahead = pair.decode.ahead(None);
        drop(pair.prefill.sent(ahead));
    }

    /// A pair's prefill is handed on behind its guard by a call of its own
    /// name. The side is the guard's by then, so the engine hears nothing
    /// of the hand-over, and a classic answer passes it for nothing.
    #[test]
    fn a_worker_handed_on_behind_its_guard_tells_the_engine_nothing() {
        let registry = one_pair();
        let classic = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let pair = pair_from(&registry, &classic).expect("a classic pair");
        assert_eq!(pair.prefill.into_guarded_worker().url(), "grpc://p:1");

        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_pairs_on(&policies, false);
        let said = || heard.lock().expect("the stage heard").clone();
        let pair = pair_from(&registry, &policies).expect("the engine places the pair");
        let guard = pair.prefill.guard(None);
        assert_eq!(said()[2..], ["dispatched Prefill"]);
        let worker = pair.prefill.into_guarded_worker();
        assert_eq!((worker.url(), worker.load()), ("grpc://p:1", 1));
        assert_eq!(said().len(), 3, "the hand-over says nothing");
        drop(guard);
        assert_eq!(said()[3..], ["completed Cancelled"]);
    }

    /// An answer of the engine handed on before its guard was made stops a
    /// debug build: its request would go out under no guard of its own, and
    /// the engine would hear a release.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "a worker is handed on behind its guard")]
    fn a_worker_handed_on_ahead_of_its_guard_stops_a_debug_build() {
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let _heard = hearing_pairs_on(&policies, false);
        let pair = pair_from(&registry, &policies).expect("the engine places the pair");
        drop(pair.prefill.into_guarded_worker());
    }

    /// One prefill and two decodes on one KV transport, under an engine
    /// whose document picks a pair's decode behind its prefill, with the
    /// pair's first placement: its prefill sent and its guard let go, as a
    /// route that sends the two sides one behind the other leaves it, and
    /// the decode's guard made ahead of its send under the key `"key"`.
    fn a_pair_behind_its_prefill() -> (
        WorkerRegistry,
        PolicyRegistry,
        Heard,
        PlacedPair,
        WorkerLoadGuard,
    ) {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_pairs_on(&policies, true);
        let pair = pair_from(&registry, &policies).expect("the engine places the pair");
        assert!(
            pair.decode.owed_behind_prefill(),
            "a route holds the engine's decode for its ask and its send"
        );
        drop(pair.prefill.guard(None));
        let ahead = pair.decode.ahead(Some("key"));
        assert_eq!(
            *heard.lock().expect("the stage heard"),
            ["issued", "dispatched Prefill"],
            "the prefill's end waits for the decode's word"
        );
        (registry, policies, heard, pair, ahead)
    }

    /// The decode of `pair` asked again with `rule` as the route's own.
    fn asked_again(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        rule: Option<CandidateFilter<'_>>,
        pair: &PlacedPair,
        ahead: WorkerLoadGuard,
    ) -> (Option<Placed>, WorkerLoadGuard) {
        let inputs = PlacementInputs {
            candidate_filter: rule,
            ..Default::default()
        };
        decode_again(
            registry,
            policies,
            MODEL,
            &|| pairs_of(registry, policies),
            None,
            inputs,
            &pair.decode,
            ahead,
        )
    }

    /// A pair's decode asked again behind its prefill moves the request to
    /// the decode the engine takes then: here the first decode is no longer
    /// ready. The answer is that decode with the engine's decode side, and
    /// a guard that counts it under the key the first guard counted under,
    /// while the first decode is counted no more. A stage hears the
    /// decode's issue at the ask, then the prefill's end that was held
    /// back, and the decode's dispatch where the route says its send. The
    /// first decode's answer keeps nothing of the engine.
    #[test]
    fn a_decode_asked_again_moves_the_request_and_its_guard_to_the_decode_the_engine_takes() {
        let (registry, policies, heard, pair, ahead) = a_pair_behind_its_prefill();
        let said = || heard.lock().expect("the stage heard").clone();
        let first = Arc::clone(pair.decode.worker());
        first.set_status(WorkerStatus::NotReady);

        let (moved, guard) = asked_again(&registry, &policies, None, &pair, ahead);
        let moved = moved.expect("the decode moved");
        assert_ne!(moved.url(), first.url());
        assert_eq!((first.load(), moved.load()), (0, 1));
        assert_eq!(
            (
                first.routing_key_inflight("key"),
                moved.routing_key_inflight("key")
            ),
            (0, 1)
        );
        assert_eq!(said()[2..], ["issued", "completed Cancelled"]);

        let guard = moved.sent(guard);
        assert_eq!(said()[4..], ["dispatched Decode"]);
        drop(pair);
        assert_eq!(said().len(), 5, "the first decode's answer says nothing");
        drop(guard);
        assert_eq!(said()[5..], ["completed Cancelled"]);
        assert_eq!(moved.load(), 0);
    }

    /// The first decision's decode stands wherever the second ask takes no
    /// other worker, and the guard handed in is the guard to keep: where
    /// the engine names the same decode again, where the route's own rule
    /// refuses every decode, and where the decode's send was said already.
    /// The route sends the decode it has, and the engine books it at that
    /// send.
    #[test]
    fn a_decode_asked_again_that_takes_no_other_worker_leaves_the_first_standing() {
        // The engine names the first decode again: the other is not ready.
        let (registry, policies, heard, pair, ahead) = a_pair_behind_its_prefill();
        let said = || heard.lock().expect("the stage heard").clone();
        for worker in &registry.get_all() {
            let decode = *worker.worker_type() == WorkerType::Decode;
            if decode && worker.url() != pair.decode.url() {
                worker.set_status(WorkerStatus::NotReady);
            }
        }
        let (moved, guard) = asked_again(&registry, &policies, None, &pair, ahead);
        assert!(moved.is_none());
        assert!(guard.counts(pair.decode.worker()));
        assert_eq!(said()[2..], ["issued", "completed Cancelled"]);
        let guard = pair.decode.sent(guard);
        assert_eq!(said()[4..], ["dispatched Decode"]);
        drop(guard);
        assert_eq!(said()[5..], ["completed Cancelled"]);

        // The route's rule refuses every decode: nothing is taken.
        let (registry, policies, heard, pair, ahead) = a_pair_behind_its_prefill();
        let said = || heard.lock().expect("the stage heard").clone();
        let nobody = |_: &dyn Worker| false;
        let (moved, guard) = asked_again(&registry, &policies, Some(&nobody), &pair, ahead);
        assert!(moved.is_none());
        assert!(guard.counts(pair.decode.worker()));
        let before = said().len();
        let guard = pair.decode.sent(guard);
        let told = said()[before..].to_vec();
        assert!(
            told.contains(&"dispatched Decode".to_owned()),
            "the first decision's decode is booked at its send: {told:?}"
        );
        drop(guard);

        // An answer whose send was said is asked nothing: its pair index
        // is not read.
        let (registry, policies, heard, pair, ahead) = a_pair_behind_its_prefill();
        let sent = pair.decode.sent(ahead);
        let before = heard.lock().expect("the stage heard").len();
        let unread: PairsAgain<'_> = &|| panic!("a sent decode reads no pair index");
        let inputs = PlacementInputs::default();
        let (moved, guard) = decode_again(
            &registry,
            &policies,
            MODEL,
            unread,
            None,
            inputs,
            &pair.decode,
            sent,
        );
        assert!(moved.is_none());
        assert!(guard.counts(pair.decode.worker()));
        assert_eq!(heard.lock().expect("the stage heard").len(), before);
        drop(guard);
    }

    /// A pair's request states its output limit once, at its first
    /// decision, and the decode's answer keeps it. Under a document that
    /// picks the decode with its prefill that decision reads the limit.
    /// Under one that picks it behind the prefill the first decision reads
    /// it where it ranks the decodes, and the decode asked again reads the
    /// same limit. The ask is handed no statement: its route may hold the
    /// body no more by then.
    #[test]
    fn a_pairs_second_ask_reads_the_output_limit_its_first_decision_stated() {
        let stated = || Stated {
            max_output_tokens: 64,
        };
        let inputs = PlacementInputs::default();

        // The decode picked with its prefill: the one decision reads it.
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_pairs_on(&policies, false);
        let pairs = pairs_of(&registry, &policies);
        let pair = select_pair(
            &registry, &policies, MODEL, &pairs, None, None, inputs, stated,
        );
        drop(pair.expect("the engine places the pair"));
        assert_eq!(
            heard.lock().expect("the stage heard")[..2],
            ["limit 64", "issued"]
        );

        // The decode picked behind its prefill: each of the two decisions
        // reads it.
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_pairs_on(&policies, true);
        let said = || heard.lock().expect("the stage heard").clone();
        let pairs = pairs_of(&registry, &policies);
        let pair = select_pair(
            &registry, &policies, MODEL, &pairs, None, None, inputs, stated,
        );
        let pair = pair.expect("the engine places the pair");
        drop(pair.prefill.guard(None));
        let ahead = pair.decode.ahead(None);
        assert_eq!(said(), ["limit 64", "issued", "dispatched Prefill"]);

        let (_moved, guard) = asked_again(&registry, &policies, None, &pair, ahead);
        assert_eq!(said()[3..5], ["limit 64", "issued"]);
        drop(guard);
    }

    /// A classic pair's decode is owed nothing behind its prefill, so a
    /// route that asks its answer first holds nothing for it. Asked again
    /// all the same, it keeps the guard it was handed, by the one test of
    /// the front: the route's read of the pair index is never made.
    #[test]
    fn a_classic_decode_asked_again_reads_no_pair_index() {
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let pair = pair_from(&registry, &policies).expect("a classic pair");
        assert!(
            !pair.decode.owed_behind_prefill(),
            "a classic decode is owed no ask and no send of its own"
        );
        let ahead = pair.decode.ahead(None);
        let unread: PairsAgain<'_> = &|| panic!("a classic pair reads no pair index");
        let inputs = PlacementInputs::default();
        let (moved, guard) = decode_again(
            &registry,
            &policies,
            MODEL,
            unread,
            None,
            inputs,
            &pair.decode,
            ahead,
        );
        assert!(moved.is_none());
        assert!(guard.counts(pair.decode.worker()));
        assert_eq!(pair.decode.load(), 1);
        drop(guard);
        assert_eq!(pair.decode.load(), 0);
    }

    /// A document that picks a pair's decode with its prefill booked the
    /// decode at the pair's one decision. The engine refuses the ask as a
    /// decode that is not chosen, which is no failure of the request: the
    /// guard handed in is the guard to keep, the ask says nothing, and the
    /// route sends the first decode. The engine hears that decode's
    /// dispatch at its send and its end at the guard's drop. Every pair is
    /// placed so under the stock document.
    #[test]
    fn a_decode_booked_with_its_prefill_answers_the_ask_as_the_first_decode_standing() {
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_pairs_on(&policies, false);
        let said = || heard.lock().expect("the stage heard").clone();
        let pair = pair_from(&registry, &policies).expect("the engine places the pair");
        drop(pair.prefill.guard(None));
        let ahead = pair.decode.ahead(None);
        assert_eq!(
            said(),
            [
                "issued",
                "issued",
                "dispatched Prefill",
                "completed Cancelled"
            ]
        );

        let (moved, guard) = asked_again(&registry, &policies, None, &pair, ahead);
        assert!(moved.is_none());
        assert!(guard.counts(pair.decode.worker()));
        assert_eq!(said().len(), 4, "the ask says nothing");

        let guard = pair.decode.sent(guard);
        assert_eq!(said()[4..], ["dispatched Decode"]);
        drop(guard);
        assert_eq!(said()[5..], ["completed Cancelled"]);
        assert_eq!(pair.decode.load(), 0);
    }
}
