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
        policy_filters_unavailable_workers, traffic_split, CacheNamespace, LoadBalancingPolicy,
        PolicyRegistry, SelectWorkerInfo, WorkerRole,
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
    // Under the traffic split, untried candidates that are all in drained
    // groups count as none left: the retry goes back to the tried workers
    // rather than to the drained build or to a miss. An explicit target
    // under consistent hashing indexes this slice and stays strict (as
    // `admission_prefilters` keeps it), so it is never narrowed: the retry
    // goes back to the targeted worker.
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
        if untried.is_empty()
            || (policies.traffic_split_enabled()
                && traffic_split::drained(&untried, |worker| worker.is_available()))
        {
            available
        } else {
            &untried
        }
    };
    // Cached hash ring for consistent hashing (O(log n) lookup).
    let hash_ring = registry.get_hash_ring(model_id);
    let info = SelectWorkerInfo {
        request_text: inputs.text,
        tokens: inputs.tokens,
        headers: inputs.headers,
        routing_key: policies.resolve_routing_key(inputs.headers),
        rid_key: inputs.rid_key,
        cache_namespace: inputs.cache_namespace,
        hash_ring,
        role: WorkerRole::Single,
        overload_lifted: false,
    };
    if available.is_empty() {
        return overload_fallback(
            registry,
            policies,
            policy.name(),
            model_id,
            candidates,
            &info,
        );
    }

    // The registry applies the routing-key sticky override when enabled and
    // otherwise delegates to the configured policy.
    let Some(idx) = policies.select_worker_for_model(&policy, model_id, available, &info) else {
        // A policy that filters availability itself misses on an
        // all-overloaded pool the same way the pre-filter empties.
        return overload_fallback(
            registry,
            policies,
            policy.name(),
            model_id,
            candidates,
            &info,
        );
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
/// has nothing routable. Under `--traffic-split` the pool is narrowed to one
/// group first, by the request's key where it has one and at random
/// otherwise, so a canary takes its weight's share of a spike and not the
/// least-loaded worker's; the pool is judged whole first, as without the
/// split, so the split chooses where, not whether: the arms admit what the fallback itself can take
/// (a ready worker, healthy with its circuit closed; under
/// `--worker-overload-shed` not an overloaded one), so a stalled member joins
/// its group's arm, counts in the group's weight and is the arm's answer when
/// every worker is stalled or the shed leaves it alone, while a group the
/// shed empties drops out of the draw as a down group does; a pool whose
/// groups all weigh 0 among those is declined rather than served from the
/// drained build; a pool of one group keeps the least-loaded pick. The
/// split's selection is counted once the arm has answered.
fn overload_fallback(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    policy: &'static str,
    model_id: &str,
    candidates: &[Arc<dyn Worker>],
    info: &SelectWorkerInfo,
) -> Option<Arc<dyn Worker>> {
    let shedding = registry.overload_shed_enabled();
    let whole =
        || overload::fallback_if_all_vetoed(candidates, model_id, STAGE_SELECTION, shedding);
    let selected = if policies.traffic_split_enabled() {
        // The pool is judged whole first, as it is without the split: a pool
        // with a worker that is down but not vetoed keeps its unavailable
        // answer whichever arm the draw would take. The split then only
        // chooses where, over arms that are all vetoed by construction.
        if !overload::all_vetoed(candidates) {
            return None;
        }
        // The arms admit what the fallback can take: a ready worker (healthy,
        // circuit closed), and under the shed not an overloaded one, so a group
        // the shed empties drops out of the draw as a down group does.
        let takes = |worker: &dyn Worker| {
            worker.is_healthy()
                && worker.circuit_breaker_can_execute()
                && !(shedding && worker.is_overloaded())
        };
        if traffic_split::drained(candidates, takes) {
            return None;
        }
        match traffic_split::Split::plan(candidates, takes) {
            Some(split) => {
                let arm = match policies.effective_sticky_key(info) {
                    Some((key, _)) => split.arm_for_key(key),
                    None => split.arm_at_random(&mut rand::rng()),
                };
                // The arm's group whole: the pool is all vetoed (checked above),
                // so the group is too, and the fallback's readiness and shed
                // filters pick inside it as they would over the pool; the other
                // groups stay out, a drained one among them.
                let members: Vec<Arc<dyn Worker>> = candidates
                    .iter()
                    .filter(|worker| traffic_split::group_of(worker.as_ref()) == arm.name)
                    .cloned()
                    .collect();
                let selected = overload::fallback_if_all_vetoed(
                    &members,
                    model_id,
                    STAGE_SELECTION,
                    shedding,
                )?;
                Metrics::record_traffic_split_selection(model_id, arm.name);
                selected
            }
            None => whole()?,
        }
    } else {
        whole()?
    };
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

/// Pick a prefill/decode pair for `model_id`, one worker per role, each under
/// its own policy and sticky namespace.
///
/// `pairs` is the snapshot's compatibility index (see [`PdPairIndex`]):
/// which decodes each prefill may hand off to was decided when membership
/// last changed, so this path compares no descriptors. Every pool is judged
/// live for availability, and a prefill counts only while one of its
/// partners is available, so the policy's pick can always be paired; `wire`
/// pins both pools to the retained plan's runtime and transport on a retry;
/// `homogeneous_runtime` narrows both pools to the runtime of the first
/// prefill worker open under its own runtime, which the gRPC wire needs
/// because its rendezvous is runtime-specific. A miss names the role and carries the verdict judged
/// from that role's candidates the request may use. A prefill pool whose
/// policy balances load against cache affinity itself is never steered by
/// the overload veto, and is placed by that policy even when every worker
/// in it is over the thresholds. Any other pool whose every worker the
/// request may use is over them is steered by load as before: a prefill
/// pool by the least-loaded default ([`relaxed_pair`]), a decode pool to
/// the least-loaded partner of the prefill its own policy picked.
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
    let fail =
        |role: WorkerRole, verdict: PlacementFailure| Box::new(PairFailure { role, verdict });

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
    // its partners, so the policy's pick can always be paired. Where the
    // wire's rendezvous is runtime-specific, both pools must also share a
    // runtime: that of the first prefill worker open under its own runtime,
    // so a prefill whose partners are all down never shuts out a healthy
    // pair on another runtime. The index keeps runtimes apart already
    // unless pairing is off or a runtime is unknown.
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
            return pair;
        }
        debug!("No available prefill workers");
        return Err(fail(
            WorkerRole::Prefill,
            failure_from(&admitted, model_id, registry.overload_shed_enabled()),
        ));
    }
    let pair_runtime = homogeneous_runtime
        .then(|| {
            pairs.prefill.iter().enumerate().find_map(|(i, p)| {
                let runtime = p.metadata().spec.runtime_type;
                (prefill_eligible(p) && can_pair_on(i, Some(runtime))).then_some(runtime)
            })
        })
        .flatten();
    // One pass: the open prefills on the pair's runtime, counting the
    // pairable ones the narrowing excluded so the exclusion leaves a trace.
    let mut excluded = 0usize;
    let mut open: Vec<usize> = (0..pairs.prefill.len())
        .filter(|&i| {
            let p = &pairs.prefill[i];
            if !prefill_eligible(p) {
                return false;
            }
            let own = p.metadata().spec.runtime_type;
            if pair_runtime.is_some_and(|runtime| own != runtime) {
                if can_pair_on(i, Some(own)) {
                    excluded += 1;
                }
                return false;
            }
            can_pair_on(i, pair_runtime)
        })
        .collect();
    if excluded > 0 {
        warn!(
            model_id,
            ?pair_runtime,
            excluded,
            "Mixed runtime types in PD workers; pairable prefill workers on another runtime were excluded"
        );
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
            return pair;
        }
        debug!(?pair_runtime, "No available PD pair");
        return Err(fail(
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
        role: WorkerRole::Prefill,
        overload_lifted: lift_prefill,
    };
    // Both pools were filtered for availability above (a pool the overload
    // veto cannot steer counting as available), so a miss here is the
    // policy's own decision.
    let declined = |role: WorkerRole, policy: &'static str| {
        Box::new(PairFailure {
            role,
            verdict: PlacementFailure::PolicyDeclined(policy),
        })
    };
    let Some((prefill_idx, prefill_group)) =
        policies.select_worker_for_model_split(&prefill_policy, model_id, &prefill, &info, None)
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
                return pair;
            }
        }
        return Err(declined(WorkerRole::Prefill, prefill_policy.name()));
    };
    let selected_prefill = prefill[prefill_idx].clone();
    if prefill_capacity.is_some_and(|capacity| !capacity.has_capacity(&selected_prefill)) {
        return Err(fail(
            WorkerRole::Prefill,
            PlacementFailure::PrefillAtCapacity,
        ));
    }
    // The pick's open partners: non-empty by construction, short of an
    // availability flip between the two reads.
    let decode: Vec<Arc<dyn Worker>> = pairs.partners[open[prefill_idx]]
        .iter()
        .filter(|d| partner_open(d, pair_runtime))
        .cloned()
        .collect();
    if decode.is_empty() {
        debug!(
            ?pair_runtime,
            "The selected prefill's partners went unavailable"
        );
        return Err(fail(WorkerRole::Decode, PlacementFailure::Unavailable));
    }
    // A decode pool all over the thresholds pairs by load, as the default
    // paired it: the least-loaded of the pick's partners, counted as a veto
    // fallback and not as a selection of the decode policy. Any other keeps
    // the veto for its policy. Either way the decode follows the traffic
    // group the prefill side was split into, or the named group an unsplit
    // prefill was picked from, so a pair never spans two groups; an unsplit
    // prefill from the default group leaves it to decide for itself (see
    // `PolicyRegistry::select_worker_for_model_split`).
    let selected_decode = if lift_decode {
        overload::least_loaded_uniform(
            decode.iter().filter(|d| {
                prefill_group.is_none_or(|group| traffic_split::group_of(d.as_ref()) == group)
            }),
            &mut rand::rng(),
        )
        .cloned()
        .ok_or_else(|| fail(WorkerRole::Decode, PlacementFailure::Unavailable))?
    } else {
        info.role = WorkerRole::Decode;
        info.overload_lifted = false;
        match policies.select_worker_for_model_split(
            &decode_policy,
            model_id,
            &decode,
            &info,
            prefill_group,
        ) {
            Some((decode_idx, _)) => decode[decode_idx].clone(),
            None => return Err(declined(WorkerRole::Decode, decode_policy.name())),
        }
    };
    if prefill_over || lift_decode {
        Metrics::record_worker_overload_fallback(STAGE_SELECTION);
        debug!(
            branch = BRANCH_ALL_OVERLOADED_FALLBACK,
            prefill_pool = prefill_over,
            decode_pool = lift_decode,
            prefill = %selected_prefill.url(),
            decode = %selected_decode.url(),
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
    use tracing_subscriber::Layer;

    use super::*;
    use crate::{
        config::types::{PdPairingMode, PolicyConfig},
        observability::logging::test_support::test_subscriber,
        policies::{CacheAwareConfig, CacheAwarePolicy, ConsistentHashingPolicy, RoundRobinPolicy},
        worker::{runtime::StallReason, BasicWorkerBuilder, ModelCard, PdWire, WorkerType},
    };

    const MODEL: &str = "m";

    /// A vLLM gRPC PD fleet whose workers carry the given KV connectors
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
    fn a_homogeneous_pair_skips_a_prefill_whose_only_partner_runs_another_runtime() {
        // P1 (NIXL) pairs only with D1, whose runtime probe failed; P2
        // (Mooncake) pairs with D1 and D2. On the gRPC wire both pools must
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
                pipeline: PipelinePin::Any,
            }),
            false,
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
            true,
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

    /// Under the traffic split, a retry whose untried candidates are all in
    /// a drained group goes back to the tried workers, as it does with
    /// nothing untried left: never to the drained build, and not to a miss
    /// while a healthy tried worker stands.
    #[test]
    fn a_retry_with_only_a_drained_group_untried_goes_back_to_the_tried_workers() {
        let registry = WorkerRegistry::new();
        for (url, group, weight) in [
            ("http://s:1", "stable", "90"),
            ("http://c:1", "canary", "0"),
        ] {
            registry
                .register(Arc::new(
                    BasicWorkerBuilder::new(url)
                        .model(ModelCard::new(MODEL))
                        .worker_type(WorkerType::Regular)
                        .connection_mode(ConnectionMode::Http)
                        .runtime_type(RuntimeType::Sglang)
                        .label("traffic_group", group)
                        .label("traffic_weight", weight)
                        .health_config(HealthCheckConfig {
                            disable_health_check: true,
                            ..Default::default()
                        })
                        .build(),
                ))
                .expect("worker registers");
        }
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin).with_traffic_split(true);
        let first = pick(&registry, &policies, PlacementInputs::default());
        assert_eq!(
            first.base_url(),
            "http://s:1",
            "the drained canary takes nothing"
        );
        let tried = vec![first.base_url().to_string()];
        let retried = pick(
            &registry,
            &policies,
            PlacementInputs {
                tried: &tried,
                ..Default::default()
            },
        );
        assert_eq!(
            retried.base_url(),
            "http://s:1",
            "the retry goes back to the tried stable worker, not to the drained canary"
        );
    }

    /// Every worker of a split pool over the overload thresholds: the steering
    /// default narrows the pool to one group first, so the canary takes its
    /// weight's share of the spike and not the least-loaded worker's, a canary
    /// at weight 0 takes nothing, and a pool whose groups all weigh 0 is
    /// declined rather than served from the drained build.
    #[test]
    fn an_all_overloaded_pool_under_the_split_keeps_the_groups_weights() {
        let grouped = |stable_weight: &str, canary_weight: &str| {
            let registry = WorkerRegistry::new();
            for (url, group, weight) in [
                ("http://s:1", "stable", stable_weight),
                ("http://s:2", "stable", stable_weight),
                ("http://c:1", "canary", canary_weight),
            ] {
                registry
                    .register(Arc::new(
                        BasicWorkerBuilder::new(url)
                            .model(ModelCard::new(MODEL))
                            .worker_type(WorkerType::Regular)
                            .connection_mode(ConnectionMode::Http)
                            .runtime_type(RuntimeType::Sglang)
                            .label("traffic_group", group)
                            .label("traffic_weight", weight)
                            .health_config(HealthCheckConfig {
                                disable_health_check: true,
                                ..Default::default()
                            })
                            .build(),
                    ))
                    .expect("worker registers");
            }
            for worker in registry.get_all() {
                registry.set_worker_overloaded(&worker, true);
            }
            registry
        };
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin).with_traffic_split(true);
        let registry = grouped("90", "10");
        let canary = (0..100)
            .filter(|_| {
                pick(&registry, &policies, PlacementInputs::default()).base_url() == "http://c:1"
            })
            .count();
        assert!(canary <= 25, "the canary took {canary} of 100 at weight 10");
        let registry = grouped("90", "0");
        for _ in 0..50 {
            assert_ne!(
                pick(&registry, &policies, PlacementInputs::default()).base_url(),
                "http://c:1",
                "a drained canary took a request from the steering default"
            );
        }
        let registry = grouped("0", "0");
        assert!(
            select_single(
                &registry,
                &policies,
                MODEL,
                RoutingPool::HttpRegular,
                None,
                PlacementInputs::default(),
            )
            .is_none(),
            "a pool whose groups all weigh 0 is declined, not served"
        );
        // A stalled member still joins its group's arm. Under the shed, where
        // an overloaded worker is never taken, the shed empties the canary
        // group here, so the pool is one group and the least-loaded pick is
        // the wedged stable worker, not a miss.
        let registry = grouped("90", "10");
        registry.set_overload_shed(true);
        let wedged = registry.get_by_url("http://s:2").expect("registered");
        registry.set_worker_overloaded(&wedged, false);
        wedged.set_stall(Some(StallReason::Wedged));
        for _ in 0..20 {
            assert_eq!(
                pick(&registry, &policies, PlacementInputs::default()).base_url(),
                "http://s:2",
                "the shed left the pool without its stalled worker"
            );
        }
        // With a wedged worker in each group the shed leaves two groups, and
        // the arm's answer is its wedged worker, the canary's at its weight.
        let registry = grouped("90", "10");
        registry
            .register(Arc::new(
                BasicWorkerBuilder::new("http://c:2")
                    .model(ModelCard::new(MODEL))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Http)
                    .runtime_type(RuntimeType::Sglang)
                    .label("traffic_group", "canary")
                    .label("traffic_weight", "10")
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            ))
            .expect("worker registers");
        registry.set_overload_shed(true);
        for url in ["http://s:2", "http://c:2"] {
            let worker = registry.get_by_url(url).expect("registered");
            registry.set_worker_overloaded(&worker, false);
            worker.set_stall(Some(StallReason::Wedged));
        }
        let mut canary = 0;
        for _ in 0..100 {
            let picked = pick(&registry, &policies, PlacementInputs::default());
            assert!(
                picked.base_url() == "http://s:2" || picked.base_url() == "http://c:2",
                "the arm answered with an overloaded worker under the shed: {}",
                picked.base_url()
            );
            if picked.base_url() == "http://c:2" {
                canary += 1;
            }
        }
        assert!(
            canary <= 25,
            "the canary took {canary} of 100 at weight 10 under the shed"
        );
        // A group with a worker that is down but not vetoed keeps the pool's
        // unavailable answer, as the single-group and the split-off paths do.
        let registry = grouped("90", "10");
        let down = registry.get_by_url("http://s:2").expect("registered");
        registry.set_worker_overloaded(&down, false);
        down.set_status(WorkerStatus::NotReady);
        assert!(
            select_single(
                &registry,
                &policies,
                MODEL,
                RoutingPool::HttpRegular,
                None,
                PlacementInputs::default(),
            )
            .is_none(),
            "a not-ready worker in the stable group was hidden from the all-vetoed check"
        );
        let registry = grouped("90", "0");
        for url in ["http://s:1", "http://s:2"] {
            let worker = registry.get_by_url(url).expect("registered");
            registry.set_worker_overloaded(&worker, false);
            worker.set_stall(Some(StallReason::Wedged));
        }
        for _ in 0..20 {
            assert_ne!(
                pick(&registry, &policies, PlacementInputs::default()).base_url(),
                "http://c:1",
                "a drained canary served a pool whose stable fleet is only suspected"
            );
        }
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
            true,
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

    /// Two traffic groups on both pools under `--traffic-split`, every worker
    /// over the overload thresholds: the veto is lifted on both pools, and the
    /// split still holds. The pair never spans two groups (the decode follows
    /// the prefill's), and a canary at weight 0 takes nothing, lifted or not.
    #[test]
    fn a_pool_placed_with_the_veto_lifted_is_still_split_and_its_pair_shares_a_group() {
        let grouped = |canary_weight: &str| {
            let registry = WorkerRegistry::new();
            for (url, worker_type, group, weight) in [
                ("grpc://ps:1", WorkerType::Prefill, "stable", "90"),
                ("grpc://ps:2", WorkerType::Prefill, "stable", "90"),
                ("grpc://pc:1", WorkerType::Prefill, "canary", canary_weight),
                ("grpc://ds:1", WorkerType::Decode, "stable", "90"),
                ("grpc://ds:2", WorkerType::Decode, "stable", "90"),
                ("grpc://dc:1", WorkerType::Decode, "canary", canary_weight),
            ] {
                registry
                    .register(Arc::new(
                        BasicWorkerBuilder::new(url)
                            .model(ModelCard::new(MODEL))
                            .worker_type(worker_type)
                            .connection_mode(ConnectionMode::Grpc)
                            .runtime_type(RuntimeType::Vllm)
                            .label("traffic_group", group)
                            .label("traffic_weight", weight)
                            .health_config(HealthCheckConfig {
                                disable_health_check: true,
                                ..Default::default()
                            })
                            .build(),
                    ))
                    .expect("worker registers");
            }
            for worker in registry.get_all() {
                registry.set_worker_overloaded(&worker, true);
            }
            registry
        };
        let policies = cohort_policies().with_traffic_split(true);
        let registry = grouped("10");
        let pairs = pairs_of(&registry, &policies);
        let mut canary = 0;
        for _ in 0..100 {
            let pair = select_pair(
                &registry,
                &policies,
                MODEL,
                &pairs,
                None,
                true,
                None,
                PlacementInputs::default(),
            )
            .expect("a pair");
            let prefill = traffic_split::group_of(pair.prefill.as_ref());
            assert_eq!(
                prefill,
                traffic_split::group_of(pair.decode.as_ref()),
                "a lifted pair spans two groups"
            );
            if prefill == "canary" {
                canary += 1;
            }
        }
        assert!(canary <= 25, "the canary took {canary} of 100 at weight 10");
        // The canary drained: nothing lands on it, lifted or not.
        let registry = grouped("0");
        let pairs = pairs_of(&registry, &policies);
        for _ in 0..50 {
            let pair = select_pair(
                &registry,
                &policies,
                MODEL,
                &pairs,
                None,
                true,
                None,
                PlacementInputs::default(),
            )
            .expect("a pair");
            assert_eq!(traffic_split::group_of(pair.prefill.as_ref()), "stable");
            assert_eq!(traffic_split::group_of(pair.decode.as_ref()), "stable");
        }
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
}
