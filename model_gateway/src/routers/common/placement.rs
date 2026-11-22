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

use std::{
    cell::Cell,
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    http::{header::RETRY_AFTER, HeaderMap, HeaderValue},
    response::Response,
};
use rand::RngExt;
use smg_policy_controller::engine::{ShedReason, WireModes, WorkerId};
use smg_worker::{
    health::overload::{
        BRANCH_ALL_OVERLOADED_FALLBACK, BRANCH_ALL_STALLED_FALLBACK, STAGE_SELECTION,
    },
    identity::basic::{ConnectionMode, ConnectionModeExt, RuntimeType, Worker, WorkerType},
};
use tracing::{debug, warn};

use crate::{
    config::{PolicyConfig, RoutingMode},
    middleware::ReportedRefusal,
    observability::{
        log_throttle::KeyedLogThrottle,
        metrics::{metrics_labels, Metrics},
    },
    policies::{
        as_consistent_hashing, policy_filters_unavailable_workers, policy_reads_hash_ring,
        CacheNamespace, ConsistentHashingPolicy, LoadBalancingPolicy, NamedList, PolicyRegistry,
        SelectWorkerInfo, WorkerRole,
    },
    policy_controller::engine::{self, Ask, Hashes, Placed, PolicyEngine, Prompt, Unplaced},
    routers::{
        common::{
            header_utils, overload,
            pair_index::PdPairIndex,
            pd_admission,
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

/// The worker an explicit target holds a request to, among `candidates`, as
/// the consistent hashing policy finds it
/// ([`ConsistentHashingPolicy::target_among`]): the header's number counts
/// the candidates that are available when the request arrives. `None` where
/// the request names no worker, and where no candidate is available: such a
/// pool is steered without a reading of the header. `Some(None)` is a miss,
/// and the request moves to no other worker.
///
/// For a route that narrows its candidates to the target ahead of a policy
/// engine, where [`strict_target`] holds.
pub(crate) fn targeted(
    candidates: &[Arc<dyn Worker>],
    headers: Option<&HeaderMap>,
) -> Option<Option<Arc<dyn Worker>>> {
    header_utils::extract_target_worker(headers)?;
    let available: Vec<Arc<dyn Worker>> = candidates
        .iter()
        .filter(|worker| worker.is_available())
        .cloned()
        .collect();
    if available.is_empty() {
        return None;
    }
    let target = ConsistentHashingPolicy::target_among(&available, headers)?;
    Some(target.and_then(|place| available.get(place).cloned()))
}

/// The policy half of [`strict_target`] read off a pair's settings: the
/// prefill or the decode policy, each the `main` one where unset, is
/// consistent hashing. Under the policy controller no role policy is built,
/// so a pair's route asks its settings what the classic would keep.
pub(crate) fn pair_keeps_target(mode: &RoutingMode, main: &PolicyConfig) -> bool {
    [mode.get_prefill_policy(main), mode.get_decode_policy(main)]
        .into_iter()
        .any(|policy| matches!(policy, PolicyConfig::ConsistentHashing))
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
    /// The request's own id, the `rid` of its body. The policy engine's keys
    /// read it where its document names it. The classic policies read
    /// `rid_key`.
    pub rid: Option<&'a str>,
    /// Whether the answer is streamed. The policy engine's request states
    /// it.
    pub streaming: bool,
    /// The retry number, `0` for a first attempt. The policy engine's
    /// request states it.
    pub attempt: u8,
    /// The prompt's hashes, where the route made them
    /// ([`Placer::hashes`]): the waiting form reads them and nothing else
    /// of the prompt. `None` in the call that does not wait: the policy
    /// engine's side makes them from `text` and `tokens`.
    pub hashes: Option<&'a Hashes>,
    /// The snapshot of the model's workers the route read for this attempt,
    /// where it read one for a rule of its own: the placement then draws
    /// its pool from it, and the registry is read once.
    pub snapshot: Option<&'a Arc<ModelWorkerSnapshot>>,
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
            members: None,
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
/// what is left, and a route whose policy engine decides in the gateway's
/// own process states it as the workers that engine decides over.
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

/// Why a placement placed nobody ([`Placer::place`],
/// [`Placer::place_waiting`]). No answer is built and no counter is written
/// here: the route judges the pool it chooses, where it answers.
#[derive(Debug)]
pub(crate) enum Missed {
    /// Who decides took no worker. The route judges its pool
    /// ([`single_failure`], [`failure_from`]): one judgment for both sides.
    Nobody,
    /// A verdict only the policy engine has: its document declined or queued
    /// the request, a queue is full or out of time, or no decision was made.
    /// The route answers it with [`Unplaced::answer`].
    Engine(Unplaced),
}

/// Who places a router's requests. Chosen once, where the router is built,
/// and asked at each of its sites through the one call ([`Placer::place`]).
pub(crate) enum Placer {
    /// The classic policies decide.
    Classic(Arc<PolicyRegistry>),
    /// The policy engine decides. The classic registry stands beside it for
    /// what a placement still reads of it: the session key of a load guard
    /// ([`Placer::session_key`]), whether the model's policy holds a request
    /// to the worker it names ([`strict_target`]), and the name a selection
    /// is counted under.
    Engine(Arc<PolicyEngine>, Arc<PolicyRegistry>),
}

impl Placer {
    /// The placer of a router whose requests `engine` places, and the
    /// classic one where the router is handed none.
    pub(crate) fn of(policies: &Arc<PolicyRegistry>, engine: Option<&Arc<PolicyEngine>>) -> Self {
        match engine {
            Some(engine) => Self::Engine(Arc::clone(engine), Arc::clone(policies)),
            None => Self::Classic(Arc::clone(policies)),
        }
    }

    /// Whether a placement waits: the gateway follows a controller, whose
    /// engine is asked over its store. A family takes the waiting form
    /// ([`Self::place_waiting`]) by this one question. In this process both
    /// sides answer at once ([`Self::place`]).
    #[inline]
    pub(crate) fn waits(&self) -> bool {
        match self {
            Self::Classic(_) => false,
            Self::Engine(engine, _) => !engine.in_process(),
        }
    }

    /// One worker of `pool` for one attempt of a request of `model`, in one
    /// pass on the calling thread: the pool on `wire`, the request's own
    /// rule, the engines its retry tried, then whoever decides
    /// ([`Decides`]). A route that holds a borrow no wait may hold places
    /// here.
    ///
    /// The classic side is [`select_single`], over the snapshot the inputs
    /// state where they state one ([`select_single_of`]), and its placement
    /// owes nothing ([`Placed::of_worker`]). The engine's side decides over
    /// the same candidates ([`EngineSide`]). A miss comes back unjudged
    /// ([`Missed`]).
    #[inline]
    pub(crate) fn place(
        &self,
        registry: &WorkerRegistry,
        model: &str,
        pool: RoutingPool,
        wire: Option<WireConstraint>,
        inputs: &PlacementInputs<'_>,
    ) -> Result<Placed, Missed> {
        match self {
            Self::Classic(policies) => {
                let worker = match inputs.snapshot {
                    Some(snapshot) => {
                        select_single_of(registry, policies, model, snapshot, pool, wire, *inputs)
                    }
                    None => select_single(registry, policies, model, pool, wire, *inputs),
                };
                worker.map(Placed::of_worker).ok_or(Missed::Nobody)
            }
            Self::Engine(engine, policies) => {
                place_by_engine(engine, policies, registry, model, pool, wire, inputs)
            }
        }
    }

    /// The value a load guard counts a session under. Asked at the
    /// placement, where the inputs stand: a family keeps the value and hands
    /// it to its guards. Both sides read the classic registry's
    /// ([`PolicyRegistry::sticky_key`]).
    #[inline]
    pub(crate) fn session_key<'a>(
        &self,
        headers: Option<&'a HeaderMap>,
        rid_key: Option<&'a str>,
    ) -> Option<&'a str> {
        let (Self::Classic(policies) | Self::Engine(_, policies)) = self;
        policies.sticky_key(headers, rid_key)
    }

    /// The prompt's hashes for `model`, for a route that states them to the
    /// waiting form ([`PlacementInputs::hashes`]): made while the route holds
    /// the prompt, read behind it. `None` under the classic policies, which
    /// read the prompt itself.
    pub(crate) fn hashes(
        &self,
        model: &str,
        text: Option<&str>,
        tokens: Option<&[u32]>,
        cache_namespace: Option<CacheNamespace>,
    ) -> Option<Hashes> {
        match self {
            Self::Classic(_) => None,
            Self::Engine(engine, _) => {
                let prompt = Prompt {
                    tokens,
                    text,
                    namespace: cache_namespace,
                };
                Some(engine.hashes(model, &prompt))
            }
        }
    }

    /// The placement that waits, under a controller's address
    /// ([`Self::waits`]): the engine is asked over the controller's store,
    /// three asks at most, and `rule` is asked of each worker picked.
    ///
    /// It stands outside a route's borrow of its request. It reads the
    /// inputs' headers, request id, streaming flag and hashes, and nothing
    /// of the prompt itself. `wires` are the wires the route dials, and
    /// `tried_rows` the rows the controller's placements named for the
    /// attempts that failed: a store's row cannot be read off a worker.
    ///
    /// Every verdict comes back as the engine's own ([`Missed::Engine`]).
    /// The classic policies never wait: such a placer places nobody here.
    pub(crate) async fn place_waiting(
        &self,
        model: &str,
        pool: RoutingPool,
        wires: WireModes,
        inputs: &PlacementInputs<'_>,
        rule: impl Fn(&dyn Worker) -> bool + Send + Sync,
        tried_rows: &[WorkerId],
    ) -> Result<Placed, Missed> {
        let Self::Engine(engine, _) = self else {
            return Err(Missed::Nobody);
        };
        let id = engine::mark();
        let ask = Ask {
            streaming: inputs.streaming,
            headers: inputs.headers,
            rid: inputs.rid,
            wires,
            tried: tried_rows,
            hashes: inputs.hashes,
            pool: Some(pool),
            ..Ask::of(&id, model)
        };
        engine
            .place_at(&ask, rule, Instant::now)
            .await
            .map_err(Missed::Engine)
    }
}

/// The policy engine's side of [`Placer::place`]: the pass every placement
/// runs ([`select_of`]) with the engine as who decides ([`EngineSide`]),
/// over one read of the registry, the snapshot the inputs state or else the
/// model's as the registry holds it now.
///
/// The engine's verdict on a request it did not place stands beside the
/// pass. Where it says the engine found no worker to send to, the miss is
/// the route's to judge, as where a classic policy took none: no endpoint
/// is left, no target holds, or the pool is saturated under the gateway's
/// own shed setting, whose answer the route's judgment builds and counts.
/// Any other verdict keeps its own answer.
fn place_by_engine(
    engine: &PolicyEngine,
    policies: &PolicyRegistry,
    registry: &WorkerRegistry,
    model: &str,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
    inputs: &PlacementInputs<'_>,
) -> Result<Placed, Missed> {
    debug_assert!(
        engine.in_process(),
        "a followed controller's engine is asked through the waiting form"
    );
    let read;
    let snapshot = match inputs.snapshot {
        Some(stated) => stated,
        None => {
            read = registry.get_routing_snapshot(model);
            &read
        }
    };
    let side = EngineSide {
        engine,
        policies,
        snapshot,
        pool,
        inputs,
        verdict: Cell::new(None),
    };
    if let Some(placed) = select_of(registry, &side, model, snapshot, pool, wire, *inputs) {
        return Ok(placed);
    }
    Err(match side.verdict.take() {
        None
        | Some(
            Unplaced::NoTargetHolds
            | Unplaced::Shed {
                reason: ShedReason::NoEndpoints,
                ..
            },
        ) => Missed::Nobody,
        Some(Unplaced::Shed {
            reason: ShedReason::Saturated,
            ..
        }) if registry.overload_shed_enabled() => Missed::Nobody,
        Some(verdict) => Missed::Engine(verdict),
    })
}

/// Pick one worker for `model_id` from `pool`, or `None` when nothing is
/// selectable. Records the selection metric under the chosen worker's own
/// transport label.
pub(crate) fn select_single(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
    inputs: PlacementInputs<'_>,
) -> Option<Arc<dyn Worker>> {
    let snapshot = registry.get_routing_snapshot(model_id);
    select_single_of(registry, policies, model_id, &snapshot, pool, wire, inputs)
}

/// [`select_single`] over `pool` of a snapshot the caller read something
/// else from too (the pool's window floor), so that both come from one
/// membership generation. The pool is borrowed from the snapshot: the
/// selection takes no count of the list. It keeps its name on the way to
/// the policy, and beside the name the snapshot's cell for what the policy
/// engine keeps of the list ([`NamedList::members`]).
pub(crate) fn select_single_of(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    snapshot: &ModelWorkerSnapshot,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
    inputs: PlacementInputs<'_>,
) -> Option<Arc<dyn Worker>> {
    select_of(registry, policies, model_id, snapshot, pool, wire, inputs)
}

/// [`select_single_of`] for whoever decides ([`Decides`]): the pool of the
/// snapshot under its name, or its part on a pinned wire, put to the
/// selection.
fn select_of<D: Decides>(
    registry: &WorkerRegistry,
    who: &D,
    model_id: &str,
    snapshot: &ModelWorkerSnapshot,
    pool: RoutingPool,
    wire: Option<WireConstraint>,
    inputs: PlacementInputs<'_>,
) -> Option<D::Picked> {
    let list = snapshot.pool(pool);
    if wire.is_none() {
        let whole = NamedList {
            list,
            positions: None,
            members: Some(snapshot.pool_members_cell(pool)),
        };
        return select_among(registry, who, model_id, list, Some(whole), inputs);
    }
    // A retry pinned to its retained plan's wire: a list of its own, which
    // nobody names.
    let pinned = Candidates::of(Arc::clone(list), wire);
    select_among(registry, who, model_id, pinned.as_slice(), None, inputs)
}

/// [`select_single`] over an explicit candidate slice: the entry for a caller
/// with a candidate rule the pools cannot express, such as dropping DP-aware
/// workers from a path that cannot pin a rank. Nobody names such a list.
pub(crate) fn select_from(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    candidates: &[Arc<dyn Worker>],
    inputs: PlacementInputs<'_>,
) -> Option<Arc<dyn Worker>> {
    select_among(registry, policies, model_id, candidates, None, inputs)
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
fn select_among<D: Decides>(
    registry: &WorkerRegistry,
    who: &D,
    model_id: &str,
    candidates: &[Arc<dyn Worker>],
    whole: Option<NamedList<'_>>,
    inputs: PlacementInputs<'_>,
) -> Option<D::Picked> {
    // At most two rounds. The pass has this one caller, so it compiles into
    // the selection and a placement pays for no call between the two.
    let mut asked_again = false;
    loop {
        match select_pass(registry, who, model_id, candidates, whole, inputs) {
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

/// What who decides answered a pass ([`Decides::decide`]).
pub(crate) enum Answered<P> {
    /// It took a worker of the candidates.
    Took(P),
    /// It took none of them. The pass's own default for a pool whose every
    /// worker is vetoed runs behind it.
    Nobody,
    /// It ended the request with a verdict of its own, which it keeps. No
    /// default runs behind it and the selection is not asked again.
    Refused,
}

/// Who decides the worker of a pass: the one thing the two implementations
/// of a placement differ in.
///
/// A pass ([`select_pass`]) asks it three questions ahead of the decision:
/// whether it tests a worker's availability itself, whether it holds the
/// request to a worker the request names, and whether it lifts the overload
/// veto for the request. It then asks for the decision over the candidates
/// the pass narrowed. Every other step of a pass is written once, for both:
/// the request's own rule, the engines a retry tried, the default for a pool
/// whose every worker is vetoed and the selection's counters.
///
/// The classic policies decide through the [`PolicyRegistry`], and the
/// policy engine through its own side ([`EngineSide`]). The compiler
/// resolves which: no call on a request's path goes through a pointer for
/// it.
pub(crate) trait Decides {
    /// What it keeps of one model for a pass: the model's policy object, or
    /// nothing.
    type Policy;
    /// The request as its decision reads it.
    type Request<'a>
    where
        Self: 'a;
    /// What a pass hands back for a request it placed: the worker, or the
    /// placement.
    type Picked;

    /// What it keeps of `model_id`.
    fn policy(&self, model_id: &str) -> Self::Policy;

    /// The name its selections are counted under.
    fn name(&self, policy: &Self::Policy) -> &'static str;

    /// Whether it tests a worker's availability itself, so the pass hands
    /// it the candidates with no test of its own.
    fn filters_unavailable(policy: &Self::Policy) -> bool;

    /// Whether it holds the request to a worker the request names
    /// ([`strict_target`]): the engines a retry tried then stay in.
    fn keeps_target(policy: &Self::Policy, headers: Option<&HeaderMap>) -> bool;

    /// The request of `inputs` over `candidates`, the pool as the request's
    /// own rule left it.
    fn request<'a>(
        &'a self,
        registry: &WorkerRegistry,
        policy: &Self::Policy,
        model_id: &str,
        candidates: &[Arc<dyn Worker>],
        inputs: &PlacementInputs<'a>,
    ) -> Self::Request<'a>;

    /// Whether the overload veto is lifted for `request`: a worker over the
    /// thresholds and otherwise routable is then a candidate.
    fn lifts_overload(request: &Self::Request<'_>) -> bool;

    /// The decision over `available`, the candidates the pass narrowed.
    /// `named` is the snapshot's own list where `available` is that list as
    /// the registry holds it.
    fn decide<'a>(
        &self,
        policy: &Self::Policy,
        model_id: &str,
        available: &[Arc<dyn Worker>],
        named: Option<NamedList<'a>>,
        request: &mut Self::Request<'a>,
    ) -> Answered<Self::Picked>;

    /// The worker of a placed request.
    fn worker(picked: &Self::Picked) -> &Arc<dyn Worker>;

    /// A request the pass's own default placed ([`overload_fallback`]).
    fn by_default(worker: Arc<dyn Worker>) -> Self::Picked;
}

/// The classic policies decide: the model's policy object, asked through
/// the registry, which applies the sticky routing-key override.
impl Decides for PolicyRegistry {
    type Policy = Arc<dyn LoadBalancingPolicy>;
    type Request<'a> = SelectWorkerInfo<'a>;
    type Picked = Arc<dyn Worker>;

    #[inline]
    fn policy(&self, model_id: &str) -> Self::Policy {
        self.get_policy_or_default(model_id)
    }

    #[inline]
    fn name(&self, policy: &Self::Policy) -> &'static str {
        policy.name()
    }

    #[inline]
    fn filters_unavailable(policy: &Self::Policy) -> bool {
        policy_filters_unavailable_workers(policy.as_ref())
    }

    #[inline]
    fn keeps_target(policy: &Self::Policy, headers: Option<&HeaderMap>) -> bool {
        strict_target(policy.as_ref(), headers)
    }

    #[inline]
    fn request<'a>(
        &'a self,
        registry: &WorkerRegistry,
        policy: &Self::Policy,
        model_id: &str,
        candidates: &[Arc<dyn Worker>],
        inputs: &PlacementInputs<'a>,
    ) -> SelectWorkerInfo<'a> {
        let mut info = SelectWorkerInfo {
            request_text: inputs.text,
            tokens: inputs.tokens,
            headers: inputs.headers,
            routing_key: self.sticky_header_key(inputs.headers),
            rid_key: inputs.rid_key,
            cache_namespace: inputs.cache_namespace,
            // The cached hash ring (O(log n) lookup), for a policy that reads
            // it: no other placement pays the registry's keyed read and the
            // ring's shared count.
            hash_ring: policy_reads_hash_ring(policy.as_ref())
                .then(|| self.hash_ring(registry, model_id))
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
        info
    }

    #[inline]
    fn lifts_overload(request: &SelectWorkerInfo<'_>) -> bool {
        request.overload_lifted
    }

    #[inline]
    fn decide<'a>(
        &self,
        policy: &Self::Policy,
        model_id: &str,
        available: &[Arc<dyn Worker>],
        named: Option<NamedList<'a>>,
        request: &mut SelectWorkerInfo<'a>,
    ) -> Answered<Arc<dyn Worker>> {
        // The name goes with the list itself: whatever narrowed the
        // candidates handed on a list of its own, at another address, and
        // nobody names that one.
        request.named = named;
        // The registry applies the routing-key sticky override when enabled
        // and otherwise delegates to the configured policy.
        match self.select_worker_for_model(policy, model_id, available, request) {
            Some(idx) => Answered::Took(available[idx].clone()),
            None => Answered::Nobody,
        }
    }

    #[inline]
    fn worker(picked: &Arc<dyn Worker>) -> &Arc<dyn Worker> {
        picked
    }

    #[inline]
    fn by_default(worker: Arc<dyn Worker>) -> Arc<dyn Worker> {
        worker
    }
}

/// The policy engine as who decides a pass ([`Decides`]), for one attempt:
/// the engine, the pool the pass draws with the snapshot it is drawn from,
/// and the attempt's inputs, of which the engine's request is stated.
///
/// Its three answers ahead of the decision. It tests a worker's availability
/// itself, so the pass hands it the candidates untested, and its own rule
/// steers a pool whose every worker is vetoed: the pass's default never runs
/// behind it. It holds a request to the worker the request names where the
/// model's classic policy does ([`strict_target`]): the engine's document
/// states no such rule yet, so the candidates are narrowed to that worker
/// ahead of the decision ([`targeted`]), as that policy finds it among its
/// own. It lifts no overload veto ahead of the pass: a session's owner is
/// kept inside the decision.
struct EngineSide<'a> {
    engine: &'a PolicyEngine,
    policies: &'a PolicyRegistry,
    snapshot: &'a Arc<ModelWorkerSnapshot>,
    pool: RoutingPool,
    inputs: &'a PlacementInputs<'a>,
    /// The engine's verdict on a request it did not place.
    verdict: Cell<Option<Unplaced>>,
}

impl Decides for EngineSide<'_> {
    /// Whether the request is held to the worker it names.
    type Policy = bool;
    type Request<'a>
        = ()
    where
        Self: 'a;
    type Picked = Placed;

    fn policy(&self, model_id: &str) -> bool {
        let headers = self.inputs.headers;
        // A request that names no worker pays the header's probe alone.
        header_utils::extract_target_worker(headers).is_some()
            && strict_target(
                self.policies.get_policy_or_default(model_id).as_ref(),
                headers,
            )
    }

    /// The name of the policy the routing settings name: the engine's
    /// document is written from it, and its selections are counted under
    /// it as the classic side's are.
    fn name(&self, _held: &bool) -> &'static str {
        self.policies.get_default_policy().name()
    }

    #[inline]
    fn filters_unavailable(_held: &bool) -> bool {
        true
    }

    #[inline]
    fn keeps_target(held: &bool, _headers: Option<&HeaderMap>) -> bool {
        *held
    }

    #[inline]
    fn request<'a>(
        &'a self,
        _registry: &WorkerRegistry,
        _held: &bool,
        _model_id: &str,
        _candidates: &[Arc<dyn Worker>],
        _inputs: &PlacementInputs<'a>,
    ) {
    }

    #[inline]
    fn lifts_overload(_request: &()) -> bool {
        false
    }

    /// The engine's decision over `available`: the pool's own list by its
    /// name where nothing narrowed the candidates (`named`), and what the
    /// engine keeps of that list is read beside it. A narrowed list is
    /// stated as the workers the request may take ([`Ask::among`]).
    ///
    /// The engines a retry tried stepped aside in the pass, by their
    /// address, so the request names no tried row and states its retry
    /// number. The prompt is lent as the inputs hold it, and its hashes are
    /// made here where the route made none.
    fn decide<'a>(
        &self,
        held: &bool,
        model_id: &str,
        available: &[Arc<dyn Worker>],
        named: Option<NamedList<'a>>,
        _request: &mut (),
    ) -> Answered<Placed> {
        let inputs = self.inputs;
        let target = held.then(|| targeted(available, inputs.headers)).flatten();
        let among: Option<Arc<[Arc<dyn Worker>]>> = match target {
            Some(Some(target)) => Some(Arc::from([target])),
            // The request names none of its candidates, and moves to no
            // other worker.
            Some(None) => return Answered::Refused,
            None => named.is_none().then(|| available.iter().cloned().collect()),
        };
        let prompt = Prompt {
            tokens: inputs.tokens,
            text: inputs.text,
            namespace: inputs.cache_namespace,
        };
        let made;
        let hashes = match inputs.hashes {
            Some(stated) => stated,
            None => {
                made = self.engine.hashes(model_id, &prompt);
                &made
            }
        };
        let id = engine::mark();
        let ask = Ask {
            streaming: inputs.streaming,
            headers: inputs.headers,
            rid: inputs.rid,
            attempt: Some(inputs.attempt),
            hashes: Some(hashes),
            prompt: Some(&prompt),
            pool: Some(self.pool),
            among: among.as_ref(),
            snapshot: Some(self.snapshot),
            ..Ask::of(&id, model_id)
        };
        // The request's own rule is asked of the worker picked too: a later
        // ask of the attempt reads the registry's list again, and that list
        // can hold a worker the pass did not put to the rule.
        let rule = |worker: &dyn Worker| {
            inputs
                .candidate_filter
                .is_none_or(|accepts| accepts(worker))
        };
        match self.engine.place_here(&ask, rule, Instant::now) {
            Some(Ok(placed)) => Answered::Took(placed),
            Some(Err(verdict)) => {
                self.verdict.set(Some(verdict));
                Answered::Refused
            }
            // A followed controller's engine answers no ask at once: its
            // routes place through the waiting form.
            None => Answered::Refused,
        }
    }

    #[inline]
    fn worker(placed: &Placed) -> &Arc<dyn Worker> {
        placed.worker()
    }

    #[inline]
    fn by_default(worker: Arc<dyn Worker>) -> Placed {
        Placed::of_worker(worker)
    }
}

/// One pass of the single-worker selection, for whoever decides
/// ([`Decides`]). `whole` names the snapshot's own list when `candidates`
/// is that list as the registry holds it: who decides is told the list's
/// name when the candidates reach it as that list, and reads which workers
/// it was handed once something narrowed them (a candidate filter that
/// refuses a worker, a policy that needs its candidates filtered for it, the
/// engines a retry already tried). A miss says why ([`Miss`]).
fn select_pass<D: Decides>(
    registry: &WorkerRegistry,
    who: &D,
    model_id: &str,
    candidates: &[Arc<dyn Worker>],
    whole: Option<NamedList<'_>>,
    inputs: PlacementInputs<'_>,
) -> Result<D::Picked, Miss> {
    let policy = who.policy(model_id);

    // The per-request predicate applies to every entry point, so a caller
    // that sets it can never dispatch to a worker it rejects. A predicate
    // that refuses nobody leaves the candidates as they were drawn, and the
    // pool keeps its name.
    let accepted = accepted_by(candidates, inputs.candidate_filter);
    let candidates = accepted.as_deref().unwrap_or(candidates);

    let mut request = who.request(registry, &policy, model_id, candidates, &inputs);
    let overload_lifted = D::lifts_overload(&request);
    let eligible = |worker: &Arc<dyn Worker>| {
        worker.is_available()
            || (overload_lifted && overload::vetoed_by_overload_alone(worker.as_ref()))
    };

    // Most policies already apply the complete availability predicate. Give
    // them the shared snapshot directly instead of cloning every available
    // worker into a second per-request Vec. Hash policies use a weaker health
    // predicate and keep the pre-filter.
    let filtered;
    let available: &[Arc<dyn Worker>] = if D::filters_unavailable(&policy) {
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
    let keeps_target = D::keeps_target(&policy, inputs.headers);
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
        return overload_fallback(registry, who.name(&policy), model_id, candidates)
            .map(D::by_default)
            .ok_or(Miss::NobodyAvailable);
    }
    // The name goes with the list itself, read here and nowhere above:
    // whatever narrowed the candidates handed on a list of its own, at
    // another address, and nobody names that one.
    let named = whole.filter(|named| std::ptr::eq(&named.list[..], available));

    let picked = match who.decide(&policy, model_id, available, named, &mut request) {
        Answered::Took(picked) => picked,
        Answered::Nobody => {
            // A policy that filters availability itself misses on an
            // all-overloaded pool the same way the pre-filter empties. Any other
            // policy was handed available candidates and declined.
            let miss = if D::filters_unavailable(&policy) {
                Miss::NobodyAvailable
            } else {
                Miss::Declined
            };
            return overload_fallback(registry, who.name(&policy), model_id, candidates)
                .map(D::by_default)
                .ok_or(miss);
        }
        Answered::Refused => return Err(Miss::Declined),
    };
    let selected = D::worker(&picked);
    if overload_lifted && !candidates.iter().any(|w| w.is_available()) {
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
        who.name(&policy),
    );

    Ok(picked)
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

/// The client-visible code of [`policy_declined`].
pub(crate) const POLICY_DECLINED_CODE: &str = "policy_declined";

const POLICY_DECLINED_MESSAGE: &str = "The policy declined the request for model";

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

/// The reason a refusal's line names, when the router gave one.
pub(crate) fn refusal_reason(response: &Response) -> Option<&str> {
    response
        .extensions()
        .get::<RefusalReason>()
        .map(|reason| reason.0.as_str())
}

/// The 503 for a declined request. The line names where the policy
/// declined. A retry reads the same rows, so the answer gives no retry hint.
pub(crate) fn policy_declined(model: &str, declined_at: &str) -> Response {
    let mut response = error::service_unavailable(
        POLICY_DECLINED_CODE,
        format!("{POLICY_DECLINED_MESSAGE} '{model}'"),
    );
    let reason = format!("the policy declined at {declined_at}");
    response.extensions_mut().insert(RefusalReason(reason));
    mark_non_retryable(&mut response);
    response
}

/// The 503 for a fleet whose prefill and decode pools hold workers and no
/// prefill of them pairs with a decode: `mismatches` are the components the
/// pools disagree on, `prefill` and `decode` each pool's pairing keys. The
/// verdict reads membership alone, so the classic selection and the policy
/// engine give one answer for one fleet.
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

/// The reason carried on a refusal response for its log line.
#[derive(Clone, Debug)]
struct RefusalReason(String);

/// Lines per second the gateway writes for each refusal it answers on its
/// own: a model whose workers are all unavailable (`no_available_workers`),
/// a model nobody serves (`model_not_found`), a model none of whose
/// workers processes the media references forwarded to it
/// (`no_media_ref_capable_worker`), or a request the policy declined
/// (`policy_declined`). One line per refusal, with
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

static POLICY_DECLINED_LOG: KeyedLogThrottle = KeyedLogThrottle::new(
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
    PolicyDeclined,
}

impl Refusal {
    fn of(response: &Response) -> Option<Self> {
        match error::extract_error_code_from_response(response) {
            NO_AVAILABLE_WORKERS_CODE => Some(Self::NoAvailableWorkers),
            MODEL_NOT_FOUND_CODE => Some(Self::ModelNotFound),
            NO_MEDIA_REF_CAPABLE_WORKER_CODE => Some(Self::NoMediaRefCapableWorker),
            POLICY_DECLINED_CODE => Some(Self::PolicyDeclined),
            _ => None,
        }
    }

    fn throttle(self) -> &'static KeyedLogThrottle {
        match self {
            Self::NoAvailableWorkers => &NO_AVAILABLE_WORKERS_LOG,
            Self::ModelNotFound => &MODEL_NOT_FOUND_LOG,
            Self::NoMediaRefCapableWorker => &NO_MEDIA_REF_CAPABLE_WORKER_LOG,
            Self::PolicyDeclined => &POLICY_DECLINED_LOG,
        }
    }

    fn message(self) -> &'static str {
        match self {
            Self::NoAvailableWorkers => "No available workers for model",
            Self::ModelNotFound => "No worker serves model",
            Self::NoMediaRefCapableWorker => "No worker processes media references for model",
            Self::PolicyDeclined => POLICY_DECLINED_MESSAGE,
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
            Self::NoAvailableWorkers | Self::NoMediaRefCapableWorker | Self::PolicyDeclined
                if registered =>
            {
                model_id
            }
            Self::NoAvailableWorkers
            | Self::NoMediaRefCapableWorker
            | Self::PolicyDeclined
            | Self::ModelNotFound => "",
        }
    }
}

/// The log line of a refusal the gateway answers on its own, and its marker.
///
/// A `no_available_workers`, `no_media_ref_capable_worker` or
/// `policy_declined` 503, or a `model_not_found` 404, gets one WARN with
/// the request id, the model, the state of the pools `site` draws from,
/// and the reason the refusal names when it has one, capped at
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
        Refusal::NoAvailableWorkers | Refusal::NoMediaRefCapableWorker | Refusal::PolicyDeclined
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
fn was_tried(worker: &dyn Worker, tried: &[String]) -> bool {
    tried.iter().any(|engine| engine == worker.base_url())
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
pub(crate) fn select_pair(
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
mod placer_tests;

#[cfg(test)]
mod tests {
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
    use smg_worker::{
        connection::engine::EngineConnection,
        health::{circuit_breaker::CircuitState, resilience::ResolvedResilience},
        identity::{
            basic::{BasicWorker, WorkerMetadata, WorkerType},
            error::WorkerResult,
        },
        load::state::{one_rank_load, TestFeed},
        runtime::{RoutingState, StallReason},
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
        routers::{common::pair_index::PdWire, grpc::backend_client::BasicWorkerBuilder},
    };

    const MODEL: &str = "m";

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

    fn pairs_of(registry: &WorkerRegistry, policies: &PolicyRegistry) -> Arc<PdPairIndex> {
        registry
            .get_routing_snapshot(MODEL)
            .pd_pairs(PdWire::Grpc, policies.pd_pairing_mode())
    }

    fn pair_from(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
    ) -> Result<Pair, Box<PairFailure>> {
        select_pair(
            registry,
            policies,
            MODEL,
            &pairs_of(registry, policies),
            None,
            None,
            PlacementInputs::default(),
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
            )
            .expect("the targeted pair is open");
            (pair.prefill, pair.decode)
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
            &only_second,
            PlacementInputs::default(),
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
                &candidates,
                PlacementInputs::default(),
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
        select_single(
            registry,
            policies,
            MODEL,
            RoutingPool::HttpRegular,
            None,
            inputs,
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
                &candidates,
                PlacementInputs::default(),
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
            )
        };
        let by_index: Vec<Arc<dyn Worker>> = (0..3)
            .map(|index| target(index).expect("a healthy pool answers every index"))
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
                members: None,
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

    #[test]
    fn a_declined_request_is_reported_once_as_a_refusal() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let _subscriber = test_subscriber(EventLines(tracing::Level::WARN, Arc::clone(&lines)));
        let registry = http_registry(&["http://worker:8080"]);
        let site = RefusalSite {
            function: "Router::route_typed_request",
            mode: "Regular",
            pools: &[("regular", RoutingPool::HttpRegular)],
        };
        let refusal = policy_declined(MODEL, "binding");
        let reported = report_refusal(&registry, site, None, MODEL, refusal);
        assert!(reported.extensions().get::<ReportedRefusal>().is_some());
        let reported = report_refusal(&registry, site, None, MODEL, reported);
        assert!(reported.extensions().get::<ReportedRefusal>().is_some());
        {
            let lines = lines.lock().expect("the refusal's lines");
            assert_eq!(lines.len(), 1, "{lines:?}");
            assert!(lines[0].contains("message=The policy declined the request for model"));
            assert!(lines[0].contains("reason=\"the policy declined at binding\""));
        }

        let kind = Refusal::of(&reported).expect("the reported refusal's kind");
        for other in [
            Refusal::NoAvailableWorkers,
            Refusal::ModelNotFound,
            Refusal::NoMediaRefCapableWorker,
        ] {
            assert!(!std::ptr::eq(kind.throttle(), other.throttle()));
        }

        let throttle = KeyedLogThrottle::new(1, Duration::from_secs(60), REFUSAL_LOG_MODELS);
        for model in ["missing-a", "missing-b", MODEL] {
            report_refusal_under(
                &throttle,
                &registry,
                site,
                None,
                model,
                policy_declined(model, "binding"),
            );
        }
        let lines = lines.lock().expect("the refusal's lines");
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(lines[2].contains(&format!("model_id=\"{MODEL}\"")));
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

    fn place_prompt(registry: &WorkerRegistry, policies: &PolicyRegistry, tokens: &[u32]) -> Pair {
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
        let home = place_prompt(registry, policies, &first_turn).prefill;
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
            )
            .expect("a prefill worker")
        };
        let owner = place();
        registry.set_worker_overloaded(&owner, true);
        assert_ne!(place().url(), owner.url());
    }
}
