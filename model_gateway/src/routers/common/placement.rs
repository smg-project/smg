//! Worker placement shared by the HTTP and gRPC families.
//!
//! Every family used to carry its own copy of the same sequences: take the
//! routing pool for the model, narrow it to the retained wire on a retry,
//! drop unavailable workers unless the policy does that itself, hand the
//! survivors to the policy registry, record the selection; and for a
//! disaggregated request, do that per leg from one snapshot under one
//! runtime. This module is those sequences written once. A family still
//! decides which pools it draws from and what a failure means on its wire.
//!
//! Distinct from [`super::worker_selection`], the least-load selector with
//! refresh-on-miss that the provider and realtime paths use; this is the
//! policy-registry path over the self-hosted routing pools.

use std::{fmt, sync::Arc, time::Duration};

use axum::{
    http::{header::RETRY_AFTER, HeaderMap, HeaderValue},
    response::Response,
};
use rand::RngExt;
use tracing::{debug, warn};

use crate::{
    middleware::ReportedRefusal,
    observability::{
        log_throttle::KeyedLogThrottle,
        metrics::{metrics_labels, Metrics},
    },
    policies::{
        policy_filters_unavailable_workers, CacheNamespace, LoadBalancingPolicy, PolicyRegistry,
        SelectWorkerInfo, WorkerLeg,
    },
    routers::{
        common::{header_utils, overload, retry::mark_non_retryable},
        error,
    },
    worker::{
        overload::{BRANCH_ALL_OVERLOADED_FALLBACK, BRANCH_ALL_STALLED_FALLBACK, STAGE_SELECTION},
        ConnectionMode, ConnectionModeExt, PdPairIndex, PrefillCandidateError,
        PrefillSelectionContext, RoutingPool, RuntimeType, Worker, WorkerRegistry,
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
}

impl WireConstraint {
    /// Whether `worker` is on `wire`: its runtime and its transport when a
    /// retry pins one, any worker when none is pinned. The one predicate
    /// every narrowing to a retained plan's wire goes through, so the legs
    /// a placement searches and the ones it judges cannot drift apart.
    pub(crate) fn admits(wire: Option<Self>, worker: &dyn Worker) -> bool {
        wire.is_none_or(|wire| {
            worker.metadata().spec.runtime_type == wire.runtime
                && *worker.connection_mode() == wire.connection
        })
    }
}

/// Whether Prefill admission may drop full workers from the candidate set
/// before the policy runs. An explicit target under consistent hashing is
/// exempt: it indexes the unfiltered set and stays strict, so the request
/// waits for that worker (checked after selection) instead of moving to
/// another one.
pub(crate) fn admission_prefilters(
    prefill_policy: &dyn LoadBalancingPolicy,
    headers: Option<&HeaderMap>,
) -> bool {
    prefill_policy.name() != "consistent_hashing"
        || header_utils::extract_target_worker(headers).is_none()
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
    /// or none that already answered this request definitively).
    pub candidate_filter: Option<CandidateFilter<'a>>,
    /// Engines (by base URL) a failed attempt of this request already
    /// dispatched to. A retry skips them while another available candidate
    /// is left to take the request, so a policy that would re-select the
    /// same worker deterministically (a routing key, a cached prefix) moves
    /// on instead of replaying on the worker that just failed. A pool with
    /// nothing else left is used whole, so a single worker is retried as
    /// before.
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
    pub(crate) fn as_slice(&self) -> &[Arc<dyn Worker>] {
        match self {
            Self::Shared(shared) => shared,
            Self::Pinned(pinned) => pinned,
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
    /// Both legs have workers, whatever their availability, but no prefill
    /// shares a KV transfer protocol with any decode (#2483): judged from
    /// membership alone, since it cannot change until membership does.
    /// Carries each leg's pairing keys and the components the legs
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
    /// caller is the runtime both legs run.
    pub runtime: RuntimeType,
}

/// Which leg of a pair placement failed, and why. Boxed at the `Err`
/// site: a shed verdict carries a whole response.
#[derive(Debug)]
pub(crate) struct PairFailure {
    pub leg: WorkerLeg,
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
    let pool = registry.get_routing_pool(model_id, pool);
    match wire {
        None => Candidates::Shared(pool),
        Some(_) => Candidates::Pinned(
            pool.iter()
                .filter(|w| WireConstraint::admits(wire, w.as_ref()))
                .cloned()
                .collect(),
        ),
    }
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
    let candidates = candidates(registry, model_id, pool, wire);
    select_from(registry, policies, model_id, candidates.as_slice(), inputs)
}

/// [`select_single`] over an explicit candidate slice: the entry for a caller
/// with a candidate rule the pools cannot express, such as dropping DP-aware
/// workers from a path that cannot pin a rank.
pub(crate) fn select_from(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    candidates: &[Arc<dyn Worker>],
    inputs: PlacementInputs<'_>,
) -> Option<Arc<dyn Worker>> {
    let policy = policies.get_policy_or_default(model_id);

    // The per-request predicate applies to every entry point, so a caller
    // that sets it can never dispatch to a worker it rejects.
    let accepted;
    let candidates: &[Arc<dyn Worker>] = match inputs.candidate_filter {
        Some(accepts) => {
            accepted = candidates
                .iter()
                .filter(|w| accepts(w.as_ref()))
                .cloned()
                .collect::<Vec<_>>();
            &accepted
        }
        None => candidates,
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
            .filter(|worker| worker.is_available())
            .cloned()
            .collect::<Vec<_>>();
        &filtered
    };
    // The engines this request already tried step aside for its retry while
    // another available candidate remains; otherwise the pool is used whole.
    // An explicit target under consistent hashing indexes this slice and
    // stays strict (as `admission_prefilters` keeps it), so it is never
    // narrowed: the retry goes back to the targeted worker.
    let strict_target = policy.name() == "consistent_hashing"
        && header_utils::extract_target_worker(inputs.headers).is_some();
    let untried;
    let available: &[Arc<dyn Worker>] = if inputs.tried.is_empty() || strict_target {
        available
    } else {
        untried = available
            .iter()
            .filter(|worker| {
                worker.is_available()
                    && !inputs
                        .tried
                        .iter()
                        .any(|engine| engine == worker.base_url())
            })
            .cloned()
            .collect::<Vec<_>>();
        if untried.is_empty() {
            available
        } else {
            &untried
        }
    };
    if available.is_empty() {
        return overload_fallback(registry, policy.name(), model_id, candidates);
    }

    // Cached hash ring for consistent hashing (O(log n) lookup).
    let hash_ring = registry.get_hash_ring(model_id);

    // The registry applies the routing-key sticky override when enabled and
    // otherwise delegates to the configured policy.
    let Some(idx) = policies.select_worker_for_model(
        &policy,
        model_id,
        available,
        &SelectWorkerInfo {
            request_text: inputs.text,
            tokens: inputs.tokens,
            headers: inputs.headers,
            routing_key: policies.resolve_routing_key(inputs.headers),
            rid_key: inputs.rid_key,
            cache_namespace: inputs.cache_namespace,
            hash_ring,
            leg: WorkerLeg::Single,
        },
    ) else {
        // A policy that filters availability itself misses on an
        // all-overloaded pool the same way the pre-filter empties.
        return overload_fallback(registry, policy.name(), model_id, candidates);
    };
    let selected = available[idx].clone();

    Metrics::record_worker_selection(
        metrics_labels::WORKER_REGULAR,
        selected.connection_mode().as_metric_label(),
        model_id,
        policy.name(),
    );

    Some(selected)
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
/// that declined a leg, the leg at its admission limit), so the reason on
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

/// The reason a refusal's line names, when the router gave one (a
/// `no_available_workers` 503; a 404 names none).
pub(crate) fn refusal_reason(response: &Response) -> Option<&str> {
    response
        .extensions()
        .get::<RefusalReason>()
        .map(|reason| reason.0.as_str())
}

/// The workers of `leg` a placement may use: those on `wire` when a retry
/// pins one, narrowed to the ones the request's candidate filter admits. A
/// failed leg's veto fallback and its verdict are judged over these, so a
/// filter that admits only vetoed workers reads as a vetoed leg (steered,
/// or shed under `--worker-overload-shed`) and not as an unavailable one
/// because a worker the request may not use is idle. A narrowing that
/// admits nothing is left out: a wire no worker of the leg is on keeps the
/// leg's own verdict as before, judged over every worker and without the
/// filter (a filter narrowed over workers the retry cannot use would turn
/// that verdict into a shed over another wire's overloaded workers), and a
/// filter that admits none of the workers on the wire is the caller's to
/// name.
fn admitted_leg(
    leg: &[Arc<dyn Worker>],
    wire: Option<WireConstraint>,
    inputs: &PlacementInputs<'_>,
) -> Vec<Arc<dyn Worker>> {
    if wire.is_none() && inputs.candidate_filter.is_none() {
        return leg.to_vec();
    }
    let narrowed = |workers: &[Arc<dyn Worker>], keep: &dyn Fn(&Arc<dyn Worker>) -> bool| {
        let kept: Vec<Arc<dyn Worker>> = workers.iter().filter(|w| keep(w)).cloned().collect();
        (!kept.is_empty()).then_some(kept)
    };
    let Some(on_wire) = narrowed(leg, &|w| WireConstraint::admits(wire, w.as_ref())) else {
        return leg.to_vec();
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

/// The reason a `no_available_workers` refusal names (the leg that is down,
/// the policy that declined), carried on the response for its log line.
#[derive(Clone, Debug)]
struct RefusalReason(String);

/// Lines per second the gateway writes for each refusal it answers on its
/// own: a model whose workers are all unavailable (`no_available_workers`)
/// and a model nobody serves (`model_not_found`). One line per refusal, with
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

/// The router answering a refusal, as its line names it.
#[derive(Clone, Copy)]
pub(crate) struct RefusalSite<'a> {
    /// The refusing function, e.g. `WorkerSelectionStage::execute`.
    pub function: &'static str,
    /// The router's mode: `Regular`, `PrefillDecode` or `EncodePrefillDecode`.
    pub mode: &'static str,
    /// The pools the router draws from for the model, named per leg.
    pub legs: &'a [(&'static str, RoutingPool)],
}

/// A refusal the gateway answers on its own, with the line it gets and the
/// cap the line is under.
#[derive(Clone, Copy)]
enum Refusal {
    NoAvailableWorkers,
    ModelNotFound,
}

impl Refusal {
    fn of(response: &Response) -> Option<Self> {
        match error::extract_error_code_from_response(response) {
            NO_AVAILABLE_WORKERS_CODE => Some(Self::NoAvailableWorkers),
            MODEL_NOT_FOUND_CODE => Some(Self::ModelNotFound),
            _ => None,
        }
    }

    fn throttle(self) -> &'static KeyedLogThrottle {
        match self {
            Self::NoAvailableWorkers => &NO_AVAILABLE_WORKERS_LOG,
            Self::ModelNotFound => &MODEL_NOT_FOUND_LOG,
        }
    }

    fn message(self) -> &'static str {
        match self {
            Self::NoAvailableWorkers => "No available workers for model",
            Self::ModelNotFound => "No worker serves model",
        }
    }

    /// The key the refusal's line is throttled under. A 503 for a model with
    /// workers registered on some leg is keyed by the model: those keys are
    /// bounded by the registry, and one storming model never silences
    /// another's first line. A 404's "model" is whatever name the client
    /// sent, so every unknown name shares one allowance: a client varying
    /// the name gets no fresh lines per name, and the one line still names
    /// one of them while `suppressed` carries the rest. A 503 for a name
    /// with no workers on any leg (the HTTP PD router answers a name nobody
    /// serves with its leg's 503, not a 404) is an unknown name too and
    /// shares one allowance with every other such name, for the same reason.
    fn key(self, model_id: &str, registered: bool) -> &str {
        match self {
            Self::NoAvailableWorkers if registered => model_id,
            Self::NoAvailableWorkers | Self::ModelNotFound => "",
        }
    }
}

/// The log line of a refusal the gateway answers on its own, and its marker.
///
/// A `no_available_workers` 503 or a `model_not_found` 404 gets one WARN
/// with the request id, the model and the state of the pools `site` draws
/// from, and the reason the refusal names when it has one (the leg that is
/// down, the policy that declined), capped at [`REFUSAL_LINES_PER_SECOND`]
/// per model with workers for the 503 and for all unknown names together
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
    // Whether some leg has workers for the model (see `Refusal::key`): one
    // pool lookup per leg, no walk over the workers; the 404 needs none.
    let registered = matches!(kind, Refusal::NoAvailableWorkers)
        && site
            .legs
            .iter()
            .any(|(_, pool)| !registry.get_routing_pool(model_id, *pool).is_empty());
    if let Some(admitted) = throttle.admit(kind.key(model_id, registered)) {
        let pool = PoolState::of(registry, model_id, site.legs);
        let reason = refusal_reason(&refusal);
        // The request's span carries the id too; the field is for a line
        // written outside it (a retry's re-selection) and is omitted when
        // the router has no id of its own to give.
        warn!(
            function = site.function,
            request_id = request_id,
            mode = %site.mode,
            model_id = %model_id,
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

/// What a refusal line says about the pools it drew from: per leg, the
/// workers registered for the model, those healthy, and those routable
/// (healthy, circuit closed, not vetoed by load or liveness).
struct PoolState {
    legs: Vec<LegState>,
}

struct LegState {
    leg: &'static str,
    workers: usize,
    healthy: usize,
    available: usize,
}

impl PoolState {
    fn of(registry: &WorkerRegistry, model_id: &str, legs: &[(&'static str, RoutingPool)]) -> Self {
        PoolState {
            legs: legs
                .iter()
                .map(|(leg, pool)| {
                    let workers = registry.get_routing_pool(model_id, *pool);
                    LegState {
                        leg,
                        workers: workers.len(),
                        healthy: workers.iter().filter(|w| w.is_healthy()).count(),
                        available: workers.iter().filter(|w| w.is_available()).count(),
                    }
                })
                .collect(),
        }
    }

    fn workers(&self) -> usize {
        self.legs.iter().map(|leg| leg.workers).sum()
    }

    fn available(&self) -> usize {
        self.legs.iter().map(|leg| leg.available).sum()
    }
}

impl fmt::Display for PoolState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, leg) in self.legs.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(
                f,
                "{} {}/{} available ({} healthy)",
                leg.leg, leg.available, leg.workers, leg.healthy
            )?;
        }
        Ok(())
    }
}

/// The steering default for a disaggregated placement whose failed leg is
/// vetoed on every worker the request may use (`failed_leg`, see
/// [`admitted_leg`]), by the overload thresholds or by the liveness
/// tracker: the least-loaded routable prefill that has a routable partner,
/// paired with its least-loaded routable partner. Pairs vetoed by overload
/// alone come first; a liveness veto is crossed only when nothing else pairs,
/// and never onto an overloaded worker under `--worker-overload-shed`.
/// `None` when the failed leg is not all-vetoed (an unhealthy leg keeps its
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
    failed_leg: &[Arc<dyn Worker>],
) -> Option<Result<Pair, Box<PairFailure>>> {
    let shedding = registry.overload_shed_enabled();
    if !overload::all_vetoed(failed_leg) || (shedding && overload::all_overloaded(failed_leg)) {
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
    // loaded leg does not send every fallback to its first pair.
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
            leg: WorkerLeg::Prefill,
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
        prefill = %prefill.url(),
        decode = %decode.url(),
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

/// Pick a prefill/decode pair for `model_id`, one worker per leg, each under
/// its own policy and sticky namespace.
///
/// `pairs` is the snapshot's compatibility index (see [`PdPairIndex`]):
/// which decodes each prefill may hand off to was decided when membership
/// last changed, so this path compares no descriptors. Every leg is judged
/// live for availability, and a prefill counts only while one of its
/// partners is available, so the policy's pick can always be paired; `wire`
/// pins both legs to the retained plan's runtime and transport on a retry;
/// `homogeneous_runtime` narrows both legs to the runtime of the first
/// prefill worker open under its own runtime, which the gRPC wire needs
/// because its rendezvous is runtime-specific. A miss names the leg and carries the verdict judged
/// from that leg's candidates the request may use.
#[expect(
    clippy::too_many_arguments,
    reason = "pair placement threads every routing input plus the admission view"
)]
pub(crate) fn select_pair(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    model_id: &str,
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    homogeneous_runtime: bool,
    prefill_capacity: Option<&PrefillSelectionContext<'_>>,
    inputs: PlacementInputs<'_>,
) -> Result<Pair, Box<PairFailure>> {
    let eligible = |w: &Arc<dyn Worker>| {
        w.is_available()
            && WireConstraint::admits(wire, w.as_ref())
            && inputs
                .candidate_filter
                .is_none_or(|accepts| accepts(w.as_ref()))
    };
    let fail = |leg: WorkerLeg, verdict: PlacementFailure| Box::new(PairFailure { leg, verdict });

    // A leg with no worker at all names itself before pairing is judged.
    if pairs.prefill_pool.is_empty() {
        debug!("No prefill workers");
        return Err(fail(WorkerLeg::Prefill, PlacementFailure::NoCandidates));
    }
    if pairs.decode_pool.is_empty() {
        debug!("No decode workers");
        return Err(fail(WorkerLeg::Decode, PlacementFailure::NoCandidates));
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
            WorkerLeg::Decode,
            PlacementFailure::NoCompatiblePair {
                prefill: refusal.prefill.clone(),
                decode: refusal.decode.clone(),
                mismatches: refusal.mismatches.clone(),
            },
        ));
    }

    // Live state: a prefill is open when it is available and so is one of
    // its partners, so the policy's pick can always be paired. Where the
    // wire's rendezvous is runtime-specific, both legs must also share a
    // runtime: that of the first prefill worker open under its own runtime,
    // so a prefill whose partners are all down never shuts out a healthy
    // pair on another runtime. The index keeps runtimes apart already
    // unless pairing is off or a runtime is unknown.
    let partner_open = |d: &Arc<dyn Worker>, runtime: Option<RuntimeType>| {
        eligible(d) && runtime.is_none_or(|r| d.metadata().spec.runtime_type == r)
    };
    let can_pair_on = |i: usize, runtime: Option<RuntimeType>| {
        pairs.partners[i].iter().any(|d| partner_open(d, runtime))
    };
    if !pairs.prefill.iter().any(&eligible) {
        let admitted = admitted_leg(&pairs.prefill, wire, &inputs);
        if let Some(pair) = relaxed_pair(
            registry,
            model_id,
            pairs,
            wire,
            prefill_capacity,
            &inputs,
            &admitted,
        ) {
            return pair;
        }
        debug!("No available prefill workers");
        return Err(fail(
            WorkerLeg::Prefill,
            failure_from(&admitted, model_id, registry.overload_shed_enabled()),
        ));
    }
    let leg_runtime = homogeneous_runtime
        .then(|| {
            pairs.prefill.iter().enumerate().find_map(|(i, p)| {
                let runtime = p.metadata().spec.runtime_type;
                (eligible(p) && can_pair_on(i, Some(runtime))).then_some(runtime)
            })
        })
        .flatten();
    // One pass: the open prefills on the leg's runtime, counting the
    // pairable ones the narrowing excluded so the exclusion leaves a trace.
    let mut excluded = 0usize;
    let mut open: Vec<usize> = (0..pairs.prefill.len())
        .filter(|&i| {
            let p = &pairs.prefill[i];
            if !eligible(p) {
                return false;
            }
            let own = p.metadata().spec.runtime_type;
            if leg_runtime.is_some_and(|runtime| own != runtime) {
                if can_pair_on(i, Some(own)) {
                    excluded += 1;
                }
                return false;
            }
            can_pair_on(i, leg_runtime)
        })
        .collect();
    if excluded > 0 {
        warn!(
            model_id,
            ?leg_runtime,
            excluded,
            "Mixed runtime types in PD workers; pairable prefill workers on another runtime were excluded"
        );
    }
    if open.is_empty() {
        let admitted = admitted_leg(&pairs.decode_pool, wire, &inputs);
        if let Some(pair) = relaxed_pair(
            registry,
            model_id,
            pairs,
            wire,
            prefill_capacity,
            &inputs,
            &admitted,
        ) {
            return pair;
        }
        debug!(?leg_runtime, "No available PD pair");
        return Err(fail(
            WorkerLeg::Decode,
            failure_from(&admitted, model_id, registry.overload_shed_enabled()),
        ));
    }

    // Independent prefill/decode policies so stateful ones (round robin) do
    // not share a counter; each leg tags the sticky key with its own prefix.
    let prefill_policy = policies.get_prefill_policy();
    let decode_policy = policies.get_decode_policy();

    // Under Prefill admission, workers at their limit leave the candidate set
    // before the policy runs, so a stateful policy never commits a choice
    // the reservation then refuses.
    if let Some(capacity) =
        prefill_capacity.filter(|_| admission_prefilters(prefill_policy.as_ref(), inputs.headers))
    {
        open.retain(|&i| capacity.has_capacity(&pairs.prefill[i]));
        if open.is_empty() {
            return Err(fail(
                WorkerLeg::Prefill,
                PlacementFailure::PrefillAtCapacity,
            ));
        }
    }
    let prefill: Vec<Arc<dyn Worker>> = open
        .iter()
        .map(|&i| Arc::clone(&pairs.prefill[i]))
        .collect();
    let hash_ring = registry.get_hash_ring(model_id);
    let mut info = SelectWorkerInfo {
        request_text: inputs.text,
        tokens: inputs.tokens,
        headers: inputs.headers,
        routing_key: policies.resolve_routing_key(inputs.headers),
        rid_key: inputs.rid_key,
        cache_namespace: inputs.cache_namespace,
        hash_ring,
        leg: WorkerLeg::Prefill,
    };
    // Both legs were filtered for availability above, so a miss here is the
    // policy's own decision, never an overloaded pool.
    let declined = |leg: WorkerLeg, policy: &'static str| {
        Box::new(PairFailure {
            leg,
            verdict: PlacementFailure::PolicyDeclined(policy),
        })
    };
    let Some(prefill_idx) =
        policies.select_worker_for_model(&prefill_policy, model_id, &prefill, &info)
    else {
        return Err(declined(WorkerLeg::Prefill, prefill_policy.name()));
    };
    let selected_prefill = prefill[prefill_idx].clone();
    if prefill_capacity.is_some_and(|capacity| !capacity.has_capacity(&selected_prefill)) {
        return Err(fail(
            WorkerLeg::Prefill,
            PlacementFailure::PrefillAtCapacity,
        ));
    }
    // The pick's open partners: non-empty by construction, short of an
    // availability flip between the two reads.
    let decode: Vec<Arc<dyn Worker>> = pairs.partners[open[prefill_idx]]
        .iter()
        .filter(|d| partner_open(d, leg_runtime))
        .cloned()
        .collect();
    if decode.is_empty() {
        debug!(
            ?leg_runtime,
            "The selected prefill's partners went unavailable"
        );
        return Err(fail(WorkerLeg::Decode, PlacementFailure::Unavailable));
    }
    info.leg = WorkerLeg::Decode;
    let Some(decode_idx) =
        policies.select_worker_for_model(&decode_policy, model_id, &decode, &info)
    else {
        return Err(declined(WorkerLeg::Decode, decode_policy.name()));
    };
    let selected_decode = decode[decode_idx].clone();
    let runtime = selected_prefill.metadata().spec.runtime_type;
    Metrics::record_worker_selection(
        metrics_labels::WORKER_PREFILL,
        selected_prefill.connection_mode().as_metric_label(),
        model_id,
        prefill_policy.name(),
    );
    Metrics::record_worker_selection(
        metrics_labels::WORKER_DECODE,
        selected_decode.connection_mode().as_metric_label(),
        model_id,
        decode_policy.name(),
    );

    debug!(
        prefill = %selected_prefill.url(),
        decode = %selected_decode.url(),
        "Selected PD pair"
    );
    Ok(Pair {
        prefill: selected_prefill,
        decode: selected_decode,
        runtime,
    })
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Mutex, time::Duration};

    use axum::http::StatusCode;
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};
    use tracing_subscriber::{layer::SubscriberExt, Layer, Registry};

    use super::*;
    use crate::{
        config::types::{PdPairingMode, PolicyConfig},
        policies::RoundRobinPolicy,
        worker::{BasicWorkerBuilder, ModelCard, PdWire, WorkerType},
    };

    const MODEL: &str = "m";

    /// A vLLM gRPC PD fleet whose legs carry the given KV connectors
    /// (`None` leaves the transport unknown).
    fn pd_registry(workers: &[(&str, WorkerType, Option<&str>)]) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        for (url, worker_type, connector) in workers {
            let mut builder = BasicWorkerBuilder::new(*url)
                .model(ModelCard::new(MODEL))
                .worker_type(*worker_type)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::Vllm)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                });
            if let Some(connector) = connector {
                builder = builder.kv_connector(*connector);
            }
            registry
                .register(Arc::new(builder.build()))
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
            true,
            None,
            PlacementInputs::default(),
        )
    }

    /// Two PD cohorts (here by KV transport) under round robin on both legs,
    /// as production configures them: one policy instance per leg.
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
                true,
                None,
                PlacementInputs::default(),
            )
            .expect("a pair is open");
            *hits.entry(pair.prefill.url().to_string()).or_insert(0) += 1;
            *hits.entry(pair.decode.url().to_string()).or_insert(0) += 1;
        }
        hits
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
    fn a_two_prefill_cohort_beside_a_one_prefill_cohort_keeps_each_decode_leg_even() {
        // 2P1D beside 1P2D. Today the prefill policy sees all three prefills,
        // so the cohorts split 2:1 by prefill count (the decode legs then
        // carry 120 and 30/30 of 180); weighting cohorts by their bottleneck
        // leg is a separate change. Whatever the split, a cohort's decodes
        // must share its traffic evenly and the legs must balance.
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
    fn a_homogeneous_pair_skips_a_prefill_whose_only_partner_runs_another_runtime() {
        // P1 (NIXL) pairs only with D1, whose runtime probe failed; P2
        // (Mooncake) pairs with D1 and D2. On the gRPC wire both legs must
        // share a runtime, so P1 is never open and every placement lands on
        // P2/D2 instead of failing every other request.
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
    fn a_fleet_without_a_shared_transport_names_both_legs() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);

        let failure = pair_from(&registry, &policies)
            .err()
            .expect("no pair shares a transport");
        assert_eq!(failure.leg, WorkerLeg::Decode);
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
    fn a_pair_shares_the_first_prefill_runtime_and_names_a_missing_leg() {
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

        // A model with no decode worker at all names the leg.
        let only_prefill = registry_of(&[(
            "grpc://p:1",
            WorkerType::Prefill,
            ConnectionMode::Grpc,
            RuntimeType::Sglang,
        )]);
        let failure = pair_from(&only_prefill, &policies)
            .err()
            .expect("no decode worker exists");
        assert_eq!(failure.leg, WorkerLeg::Decode);
        assert!(matches!(failure.verdict, PlacementFailure::NoCandidates));
    }

    #[test]
    fn a_pinned_pair_keeps_the_retained_runtime_and_transport() {
        // Same runtime on both legs, but the only decode worker speaks ZMQ.
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
        // Built from raw legs, as a caller with its own candidate lists would.
        let pairs = PdPairIndex::build(
            by_type(WorkerType::Prefill).into(),
            by_type(WorkerType::Decode).into(),
            policies.pd_pairing_mode(),
        );

        // Unpinned, the ZMQ decode worker is a candidate like any other.
        assert!(select_pair(
            &registry,
            &policies,
            MODEL,
            &pairs,
            None,
            false,
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
            }),
            false,
            None,
            PlacementInputs::default(),
        )
        .err()
        .expect("the retained transport has no decode worker");
        assert_eq!(failure.leg, WorkerLeg::Decode);
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
    /// decode leg whose only admitted worker is over the thresholds is
    /// steered to it by default and shed under `--worker-overload-shed`,
    /// instead of being refused as unavailable because a worker the request
    /// may not use is idle.
    #[test]
    fn a_filtered_decode_leg_is_judged_over_the_workers_the_request_may_use() {
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
                true,
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
            .expect("under shedding the admitted leg is all-overloaded");
        assert_eq!(failure.leg, WorkerLeg::Decode);
        assert!(
            matches!(failure.verdict, PlacementFailure::AllOverloaded(_)),
            "{:?}",
            failure.verdict
        );
        let pair = place(PlacementInputs::default())
            .expect("the idle worker still takes a request without the filter");
        assert_eq!(pair.decode.url(), "grpc://d:2");
    }

    /// The prefill leg's mirror of the case above.
    #[test]
    fn a_filtered_prefill_leg_is_judged_over_the_workers_the_request_may_use() {
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
                true,
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
            .expect("under shedding the admitted leg is all-overloaded");
        assert_eq!(failure.leg, WorkerLeg::Prefill);
        assert!(
            matches!(failure.verdict, PlacementFailure::AllOverloaded(_)),
            "{:?}",
            failure.verdict
        );
    }

    /// A retry whose pinned wire no prefill worker is on keeps the leg's own
    /// verdict (unavailable: nothing the retry can use), whatever the filter
    /// would keep of the other wire's workers: an overloaded worker the retry
    /// cannot use is no reason to shed.
    #[test]
    fn a_leg_with_no_worker_on_the_pinned_wire_keeps_its_verdict_under_a_filter() {
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
        });
        let only_p1 = |w: &dyn Worker| w.url() == "grpc://p:1";
        let failure = select_pair(
            &registry,
            &policies,
            MODEL,
            &pairs_of(&registry, &policies),
            other_wire,
            true,
            None,
            PlacementInputs {
                candidate_filter: Some(&only_p1),
                ..Default::default()
            },
        )
        .err()
        .expect("no worker on the wire: the pair fails");
        assert_eq!(failure.leg, WorkerLeg::Prefill);
        assert!(
            matches!(failure.verdict, PlacementFailure::Unavailable),
            "the verdict is the leg's own, not a shed over the other wire's overloaded worker: {:?}",
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
            selection_policy: None,
            selection_accounting_ttl_ms: 0,
        }
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

    /// A refusal the gateway answers on its own is marked for the loggers
    /// downstream and written once, at WARN, with the refusing router, the
    /// request id, the model, its reason and the pool's state; past the cap
    /// the lines are counted onto the key's next admitted one: a model's own
    /// for the 503, one for all unknown names for the 404; a refusal already
    /// reported and any other answer pass through silently.
    #[test]
    fn a_refusal_is_marked_and_written_once_at_warn_with_the_pools_state() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let _guard = tracing::subscriber::set_default(
            Registry::default().with(EventLines(tracing::Level::WARN, Arc::clone(&lines))),
        );
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
            legs: &[("regular", RoutingPool::HttpRegular)],
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
            "model_id=m1",
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
        assert!(line(2).contains("model_id=m2"), "{:?}", line(2));
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
        assert!(line(5).contains("model_id=m5"), "{:?}", line(5));
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

    /// A `no_available_workers` 503 for a name with no workers on any leg
    /// (the HTTP PD router's answer for a name nobody serves) is an unknown
    /// name: it shares one allowance with every other such name, as the 404
    /// does, so a client rotating names opens no fresh lines per name and
    /// the counts are not lost to the key eviction; a model with workers
    /// keeps a key of its own.
    #[test]
    fn a_503_for_a_name_without_workers_shares_one_allowance_with_the_other_unknown_names() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let _guard = tracing::subscriber::set_default(
            Registry::default().with(EventLines(tracing::Level::WARN, Arc::clone(&lines))),
        );
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
            legs: &[("regular", RoutingPool::HttpRegular)],
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
        assert!(line(0).contains("model_id=x1"), "{:?}", line(0));
        assert!(line(1).contains("model_id=x2"), "{:?}", line(1));
        assert!(
            line(0).contains("pool=regular 0/0 available (0 healthy)"),
            "{:?}",
            line(0)
        );

        // The model with a worker registered keeps an allowance of its own.
        refuse("m1");
        assert_eq!(count(), 3, "{:?}", line(2));
        assert!(line(2).contains("model_id=m1"), "{:?}", line(2));
        assert!(
            line(2).contains("pool=regular 0/1 available (0 healthy)"),
            "{:?}",
            line(2)
        );

        // The next window: the unknown names' line carries the one counted.
        std::thread::sleep(window + Duration::from_millis(10));
        refuse("x4");
        assert!(line(3).contains("model_id=x4"), "{:?}", line(3));
        assert!(line(3).contains("suppressed=1"), "{:?}", line(3));
    }

    /// A refusal's line names the reason the router gives, which may differ
    /// from the message the client reads: the message stays the one under
    /// the code, the line gets the verdict (here the policy that declined a
    /// leg), so the reason never contradicts the pool state next to it.
    #[test]
    fn the_refusal_line_names_the_reason_the_router_gives_not_the_clients_message() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let _guard = tracing::subscriber::set_default(
            Registry::default().with(EventLines(tracing::Level::WARN, Arc::clone(&lines))),
        );
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
            legs: &[("regular", RoutingPool::HttpRegular)],
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
}
