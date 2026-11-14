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
use smg_policy_controller::{
    Decision, Lease, LeaseId, Phase, Shed, ShedReason, Side, Target, WorkerId, MAX_TARGETS,
};
use tracing::{debug, warn};

use crate::{
    config::types::PdPairingMode,
    middleware::ReportedRefusal,
    observability::{
        cache_trace,
        log_throttle::KeyedLogThrottle,
        metrics::{metrics_labels, Metrics},
    },
    policies::{
        as_cache_aware, as_consistent_hashing,
        engine_host::{Affinity, HostRequest, PairPartners},
        policy_filters_unavailable_workers, CacheNamespace, Decider, LoadBalancingPolicy,
        PolicyRegistry, SelectWorkerInfo, WorkerRole,
    },
    routers::{
        common::{header_utils, overload, pd_admission, retry::mark_non_retryable},
        error,
    },
    worker::{
        overload::{BRANCH_ALL_OVERLOADED_FALLBACK, BRANCH_ALL_STALLED_FALLBACK, STAGE_SELECTION},
        ConnectionMode, ConnectionModeExt, PdPairIndex, PrefillCandidateError,
        PrefillSelectionContext, RoutingPool, RuntimeType, Worker, WorkerRegistry, WorkerType,
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
    /// The lease sequence the caller drew ([`Decider::begin`]) before it
    /// captured the candidates it hands [`select_from`], so a worker retired
    /// between that capture and the decision is booked by no commit of the
    /// request. `None` numbers the lease when its view is built;
    /// [`select_single`] draws the sequence at its own capture.
    pub sequence: Option<LeaseId>,
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
    // The request's sequence comes before the pool read: a worker retired
    // from here on is booked by no commit of this request, however the
    // retirement interleaves with the capture and the decision.
    let sequence = policies.decider().map(|decider| decider.begin());
    let candidates = candidates(registry, model_id, pool, wire);
    select_from(
        registry,
        policies,
        model_id,
        candidates.as_slice(),
        PlacementInputs { sequence, ..inputs },
    )
}

/// [`select_single`] over an explicit candidate slice: the entry for a caller
/// with a candidate rule the pools cannot express, such as dropping DP-aware
/// workers from a path that cannot pin a rank. Such a caller draws the
/// request's sequence before its capture ([`PlacementInputs::sequence`]).
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

    let mut info = SelectWorkerInfo {
        request_text: inputs.text,
        tokens: inputs.tokens,
        headers: inputs.headers,
        routing_key: policies.sticky_header_key(inputs.headers),
        rid_key: inputs.rid_key,
        cache_namespace: inputs.cache_namespace,
        // Cached hash ring for consistent hashing (O(log n) lookup), when a
        // policy of this registry may read one (`hash_ring_wanted`).
        hash_ring: policies
            .hash_ring_wanted()
            .then(|| registry.get_hash_ring(model_id))
            .flatten(),
        role: WorkerRole::Single,
        overload_lifted: false,
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

    // Under `--policy-engine engine` the engine decides over the bounded view
    // of the pool and the commit re-validates each target under this host's
    // eligibility; a model on another policy than the default keeps the
    // legacy selection below. A keyed request is a session of the engine's
    // binding (the sticky override's rule, or the manual policy's own), and
    // the lift travels with it as the legacy keyed path lifts the veto.
    if let Some(decider) = policies.engine_for(policy.as_ref()) {
        return select_from_decided(RegularDecision {
            decider,
            registry,
            model_id,
            candidates,
            request: HostRequest {
                model: model_id,
                prompt_tokens: decider.prompt_tokens(inputs.tokens),
                // A key-native default's own key, else the override's
                // (`PolicyRegistry::engine_session_key`).
                session_key: PolicyRegistry::engine_session_key(policy.as_ref(), &info),
                tried: inputs.tried,
                overload_lifted: info.overload_lifted,
                phase: Phase::Single,
                has_capacity: None,
                sequence: inputs.sequence,
                // A cache-aware decision reads its affinity through the
                // policy's own index and trees.
                affinity: as_cache_aware(policy.as_ref()).map(|policy| Affinity {
                    policy,
                    info: &info,
                }),
            },
            eligible: &eligible,
            policy: policy.name(),
        });
    }

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
            .filter(|worker| {
                eligible(worker)
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

    // The registry applies the routing-key sticky override when enabled and
    // otherwise delegates to the configured policy.
    let Some(idx) = policies.select_worker_for_model(&policy, model_id, available, &info) else {
        // A policy that filters availability itself misses on an
        // all-overloaded pool the same way the pre-filter empties.
        return overload_fallback(registry, policy.name(), model_id, candidates);
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

    Some(selected)
}

/// What [`select_from`] hands the engine seam: the pool after the host's
/// per-request narrowing, the request as the host reads it, and the
/// eligibility the commit re-validates each target under.
struct RegularDecision<'a> {
    decider: &'a Decider,
    registry: &'a WorkerRegistry,
    model_id: &'a str,
    candidates: &'a [Arc<dyn Worker>],
    request: HostRequest<'a>,
    eligible: &'a dyn Fn(&Arc<dyn Worker>) -> bool,
    /// The configured policy's name: the selection metric's label, as under
    /// the legacy path, until the observability step relabels it.
    policy: &'static str,
}

/// [`select_from`] through the policy engine: one decision over the view of
/// the candidates, outside any lock of this host; the commit walks the
/// ordered targets and takes the first one the eligibility re-validates.
/// When every target misses, one re-decision with the misses excluded, then
/// the host's own all-vetoed fallback, as the legacy path falls back.
fn select_from_decided(rd: RegularDecision<'_>) -> Option<Arc<dyn Worker>> {
    let RegularDecision {
        decider,
        registry,
        model_id,
        candidates,
        request,
        eligible,
        policy,
    } = rd;
    cache_trace::begin_selection(candidates);
    let accept = |target: &Target| eligible(&candidates[target.position as usize]);
    let (decision, lease) = decider.decide(&request, candidates, &[]);
    let committed = decider
        .commit(&lease, &decision, Side::Single, accept)
        .or_else(|| {
            let (missed, count) = target_ids(&decision);
            if count == 0 {
                return None;
            }
            let (again, lease) = decider.decide(&request, candidates, &missed[..count]);
            decider.commit(&lease, &again, Side::Single, accept)
        });
    let Some(target) = committed else {
        cache_trace::selection(policy, "policy", candidates, None);
        return overload_fallback(registry, policy, model_id, candidates);
    };
    let position = target.position as usize;
    // The route goes into the policy's tree where the legacy selection
    // records it (the event index is the engines').
    if let Some(source) = request.affinity {
        source
            .policy
            .record_route(candidates, position, source.info);
    }
    cache_trace::selection(policy, "policy", candidates, Some(position));
    let selected = Arc::clone(&candidates[position]);
    Metrics::record_worker_selection(
        metrics_labels::WORKER_REGULAR,
        selected.connection_mode().as_metric_label(),
        model_id,
        policy,
    );
    Some(selected)
}

/// The ids of a decision's targets, for the one re-decision after every one
/// of them missed its commit; none for a shed.
pub(crate) fn target_ids(decision: &Decision) -> ([WorkerId; MAX_TARGETS], usize) {
    let targets = decision
        .pick()
        .map_or(&[][..], |pick| pick.targets.as_slice());
    let mut ids = [WorkerId::default(); MAX_TARGETS];
    for (id, target) in ids.iter_mut().zip(targets) {
        *id = target.id;
    }
    (ids, targets.len().min(MAX_TARGETS))
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
    has_capacity: &dyn Fn(&Arc<dyn Worker>) -> bool,
    inputs: &PlacementInputs<'_>,
    failed_pool: &[Arc<dyn Worker>],
) -> Option<Result<Pair, Box<PairFailure>>> {
    let shedding = registry.overload_shed_enabled();
    if !steerable(shedding, failed_pool) {
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
    if !has_capacity(&prefill) {
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

/// Whether a pool with nothing open is the steering default's to pair
/// ([`relaxed_pair`]): vetoed on every worker, and not all-overloaded under
/// `--worker-overload-shed` (`shedding`), where the caller sheds instead.
fn steerable(shedding: bool, pool: &[Arc<dyn Worker>]) -> bool {
    overload::all_vetoed(pool) && !(shedding && overload::all_overloaded(pool))
}

/// Whether a worker is open: available, or vetoed by the overload thresholds
/// alone while its pool's veto is lifted.
fn open_under(worker: &Arc<dyn Worker>, lift: bool) -> bool {
    worker.is_available() || (lift && overload::vetoed_by_overload_alone(worker.as_ref()))
}

/// The decode side's live checks, which the engine's commit repeats on the
/// partner a decision chose: open under the decode pool's lift, and on the
/// pair's runtime where one is pinned.
fn partner_live(decode: &Arc<dyn Worker>, lift_decode: bool, runtime: Option<RuntimeType>) -> bool {
    open_under(decode, lift_decode)
        && runtime.is_none_or(|r| decode.metadata().spec.runtime_type == r)
}

/// The live predicates of one pair placement: what the request may use (the
/// retained plan's wire, its own candidate filter) and what counts as open
/// on each side (available, or vetoed by the overload thresholds alone where
/// the pool's veto is lifted). Settled once by [`pair_prelude`], read by the
/// legacy selection and by the engine's decision alike.
#[derive(Clone, Copy)]
struct PairLive<'a> {
    wire: Option<WireConstraint>,
    candidate_filter: Option<CandidateFilter<'a>>,
    /// The prefill pool is over the thresholds on every worker the request
    /// may use, and steered by load unless its policy places over the veto.
    prefill_over: bool,
    /// The prefill policy places over the lifted veto
    /// ([`LoadBalancingPolicy::balances_load_against_affinity`]).
    lift_prefill: bool,
    /// The decode pool is over the thresholds on every worker the request
    /// may use: its veto is lifted and the pick pairs by load.
    lift_decode: bool,
}

impl PairLive<'_> {
    fn admitted(&self, worker: &Arc<dyn Worker>) -> bool {
        WireConstraint::admits(self.wire, worker.as_ref())
            && self
                .candidate_filter
                .is_none_or(|accepts| accepts(worker.as_ref()))
    }

    fn eligible(&self, worker: &Arc<dyn Worker>, lift: bool) -> bool {
        self.admitted(worker) && open_under(worker, lift)
    }

    fn prefill_eligible(&self, worker: &Arc<dyn Worker>) -> bool {
        self.eligible(worker, self.lift_prefill)
    }

    /// Whether a decode partner is open for the pair: eligible under the
    /// decode pool's lift, and on the pair's runtime where one is pinned.
    fn partner_open(&self, decode: &Arc<dyn Worker>, runtime: Option<RuntimeType>) -> bool {
        self.admitted(decode) && partner_live(decode, self.lift_decode, runtime)
    }
}

/// What the prelude of a pair placement settles from membership and
/// availability alone, before any policy runs.
enum PairPrelude<'a> {
    /// A verdict from membership: a role with no worker, or the pairing
    /// refusal.
    Failed(Box<PairFailure>),
    /// No prefill is open with an open partner: the role names the pool that
    /// left nothing open, for the steering default or the role's verdict.
    AllVetoed(WorkerRole),
    /// The open prefills (positions into `pairs.prefill`), each with an open
    /// partner on the pair's runtime, for the policies.
    Open {
        live: PairLive<'a>,
        open: Vec<usize>,
        pair_runtime: Option<RuntimeType>,
    },
}

/// The prelude of [`select_pair`], shared with the engine seam: the live
/// predicates, the roles' membership verdicts, and the open prefills on the
/// pair's runtime. See [`select_pair`] for the rules.
#[expect(
    clippy::too_many_arguments,
    reason = "the prelude reads every routing input of the pair placement"
)]
fn pair_prelude<'a>(
    registry: &WorkerRegistry,
    prefill_policy: &dyn LoadBalancingPolicy,
    pairing_mode: PdPairingMode,
    model_id: &str,
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    homogeneous_runtime: bool,
    inputs: &PlacementInputs<'a>,
) -> PairPrelude<'a> {
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
    let live = PairLive {
        wire,
        candidate_filter: inputs.candidate_filter,
        prefill_over: lifted(&pairs.prefill),
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
        lift_prefill: !shedding
            && prefill_policy.balances_load_against_affinity()
            && vetoed_alone(&pairs.prefill),
        lift_decode: lifted(&pairs.decode_pool),
    };
    let fail = |role: WorkerRole, verdict: PlacementFailure| {
        PairPrelude::Failed(Box::new(PairFailure { role, verdict }))
    };

    // A role with no worker at all names itself before pairing is judged.
    if pairs.prefill_pool.is_empty() {
        debug!("No prefill workers");
        return fail(WorkerRole::Prefill, PlacementFailure::NoCandidates);
    }
    if pairs.decode_pool.is_empty() {
        debug!("No decode workers");
        return fail(WorkerRole::Decode, PlacementFailure::NoCandidates);
    }
    if let Some(refusal) = &pairs.refusal {
        warn!(
            model_id,
            mode = pairing_mode.as_str(),
            mismatches = ?refusal.mismatches,
            prefill = ?refusal.prefill,
            decode = ?refusal.decode,
            "No prefill/decode pair shares a KV transfer protocol"
        );
        return fail(
            WorkerRole::Decode,
            PlacementFailure::NoCompatiblePair {
                prefill: refusal.prefill.clone(),
                decode: refusal.decode.clone(),
                mismatches: refusal.mismatches.clone(),
            },
        );
    }

    // Live state: a prefill is open when it is available and so is one of
    // its partners, so the policy's pick can always be paired. Where the
    // wire's rendezvous is runtime-specific, both pools must also share a
    // runtime: that of the first prefill worker open under its own runtime,
    // so a prefill whose partners are all down never shuts out a healthy
    // pair on another runtime. The index keeps runtimes apart already
    // unless pairing is off or a runtime is unknown.
    let can_pair_on = |i: usize, runtime: Option<RuntimeType>| {
        pairs.partners[i]
            .iter()
            .any(|d| live.partner_open(d, runtime))
    };
    if !pairs.prefill.iter().any(|p| live.prefill_eligible(p)) {
        return PairPrelude::AllVetoed(WorkerRole::Prefill);
    }
    let pair_runtime = homogeneous_runtime
        .then(|| {
            pairs.prefill.iter().enumerate().find_map(|(i, p)| {
                let runtime = p.metadata().spec.runtime_type;
                (live.prefill_eligible(p) && can_pair_on(i, Some(runtime))).then_some(runtime)
            })
        })
        .flatten();
    // One pass: the open prefills on the pair's runtime, counting the
    // pairable ones the narrowing excluded so the exclusion leaves a trace.
    let mut excluded = 0usize;
    let open: Vec<usize> = (0..pairs.prefill.len())
        .filter(|&i| {
            let p = &pairs.prefill[i];
            if !live.prefill_eligible(p) {
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
        return PairPrelude::AllVetoed(if live.prefill_over {
            WorkerRole::Prefill
        } else {
            WorkerRole::Decode
        });
    }
    PairPrelude::Open {
        live,
        open,
        pair_runtime,
    }
}

/// The workers of the pool `role` names (the pairable prefills, or the decode
/// pool), as the request may use them ([`admitted_pool`]).
fn admitted_role_pool(
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    inputs: &PlacementInputs<'_>,
    role: WorkerRole,
) -> Vec<Arc<dyn Worker>> {
    let pool: &[Arc<dyn Worker>] = match role {
        WorkerRole::Decode => &pairs.decode_pool,
        WorkerRole::Prefill | WorkerRole::Single => &pairs.prefill,
    };
    admitted_pool(pool, wire, inputs)
}

/// A pair pool with nothing open, named by `role`: the steering default's
/// pair when the pool is vetoed on every worker the request may use
/// ([`relaxed_pair`]), else the role's verdict judged from those workers.
fn settle_vetoed(
    registry: &WorkerRegistry,
    model_id: &str,
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    has_capacity: &dyn Fn(&Arc<dyn Worker>) -> bool,
    inputs: &PlacementInputs<'_>,
    role: WorkerRole,
) -> Result<Pair, Box<PairFailure>> {
    let admitted = admitted_role_pool(pairs, wire, inputs, role);
    if let Some(pair) = relaxed_pair(
        registry,
        model_id,
        pairs,
        wire,
        has_capacity,
        inputs,
        &admitted,
    ) {
        return pair;
    }
    debug!(?role, "No available PD pair");
    Err(Box::new(PairFailure {
        role,
        verdict: failure_from(&admitted, model_id, registry.overload_shed_enabled()),
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
    // Independent prefill/decode policies so stateful ones (round robin) do
    // not share a counter; each role tags the sticky key with its own prefix.
    let prefill_policy = policies.get_prefill_policy();
    let decode_policy = policies.get_decode_policy();
    let has_capacity =
        |w: &Arc<dyn Worker>| prefill_capacity.is_none_or(|capacity| capacity.has_capacity(w));
    let (live, mut open, pair_runtime) = match pair_prelude(
        registry,
        prefill_policy.as_ref(),
        policies.pd_pairing_mode(),
        model_id,
        pairs,
        wire,
        homogeneous_runtime,
        &inputs,
    ) {
        PairPrelude::Failed(failure) => return Err(failure),
        PairPrelude::AllVetoed(role) => {
            return settle_vetoed(
                registry,
                model_id,
                pairs,
                wire,
                &has_capacity,
                &inputs,
                role,
            )
        }
        PairPrelude::Open {
            live,
            open,
            pair_runtime,
        } => (live, open, pair_runtime),
    };
    let fail =
        |role: WorkerRole, verdict: PlacementFailure| Box::new(PairFailure { role, verdict });

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
    let hash_ring = policies
        .hash_ring_wanted()
        .then(|| registry.get_hash_ring(model_id))
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
        overload_lifted: live.lift_prefill,
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
    let Some(prefill_idx) =
        policies.select_worker_for_model(&prefill_policy, model_id, &prefill, &info)
    else {
        // A policy that declines a lifted pool all over the thresholds: the
        // least-loaded default steers it, as it steers any other such pool.
        if live.lift_prefill {
            let admitted = admitted_role_pool(pairs, wire, &inputs, WorkerRole::Prefill);
            if let Some(pair) = relaxed_pair(
                registry,
                model_id,
                pairs,
                wire,
                &has_capacity,
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
    let mut decode: Vec<Arc<dyn Worker>> = pairs.partners[open[prefill_idx]]
        .iter()
        .filter(|d| live.partner_open(d, pair_runtime))
        .cloned()
        .collect();
    if decode.is_empty() {
        debug!(
            ?pair_runtime,
            "The selected prefill's partners went unavailable"
        );
        return Err(fail(WorkerRole::Decode, PlacementFailure::Unavailable));
    }
    retain_decodes_with_a_room(decode_policy.as_ref(), inputs.headers, &mut decode);
    // A decode pool all over the thresholds pairs by load, as the default
    // paired it: the least-loaded of the pick's partners, counted as a veto
    // fallback and not as a selection of the decode policy. Any other keeps
    // the veto for its policy.
    let selected_decode = if live.lift_decode {
        overload::least_loaded_uniform(decode.iter(), &mut rand::rng())
            .cloned()
            .ok_or_else(|| fail(WorkerRole::Decode, PlacementFailure::Unavailable))?
    } else {
        info.role = WorkerRole::Decode;
        info.overload_lifted = false;
        match policies.select_worker_for_model(&decode_policy, model_id, &decode, &info) {
            Some(decode_idx) => decode[decode_idx].clone(),
            None => return Err(declined(WorkerRole::Decode, decode_policy.name())),
        }
    };
    if live.prefill_over || live.lift_decode {
        Metrics::record_worker_overload_fallback(STAGE_SELECTION);
        debug!(
            branch = BRANCH_ALL_OVERLOADED_FALLBACK,
            prefill_pool = live.prefill_over,
            decode_pool = live.lift_decode,
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
    if !live.lift_decode {
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

/// The policy engine for a pair placement, or `None` when the legacy path
/// decides: hosted under `--policy-engine engine`, and not placing a request
/// the sticky routing-key override keys on either side or a key-native side
/// pins by the key placement derives, a configured routing-key header
/// included ([`PolicyRegistry::pair_engine_for`]).
pub(crate) fn pair_decider<'a>(
    policies: &'a PolicyRegistry,
    inputs: &PlacementInputs<'_>,
) -> Option<&'a Arc<Decider>> {
    policies.pair_engine_for(&SelectWorkerInfo {
        headers: inputs.headers,
        routing_key: policies.sticky_header_key(inputs.headers),
        rid_key: inputs.rid_key,
        ..SelectWorkerInfo::default()
    })
}

/// A pair placement decided through the engine, as [`decide_pair`] hands it
/// to [`commit_pair`] under the same hold of the admission lock.
#[expect(
    clippy::large_enum_variant,
    reason = "the decided side carries the engine's decision by value across the one hop from the decision to the commit; boxing it would cost one allocation per pair placement"
)]
pub(crate) enum PairProposal {
    /// The prelude's own pair (the steering default over an all-vetoed pool):
    /// the caller re-validates the prefill's window alone.
    Settled(Pair),
    /// The engine's decision, committed over its ordered targets.
    Decided(PairDecision),
}

/// The engine's pair decision with what its commit indexes and re-validates.
pub(crate) struct PairDecision {
    decision: Decision,
    lease: Lease,
    /// The decode side's booking of a request of unknown size: the decode
    /// policy's own estimate, where the lease carries the prefill's.
    decode_tokens: u32,
    /// The open prefills the decision's targets index by position.
    prefills: Vec<Arc<dyn Worker>>,
    /// The decode rows the targets' pair sides index by position.
    decodes: Vec<Arc<dyn Worker>>,
    pair_runtime: Option<RuntimeType>,
    prefill_over: bool,
    lift_prefill: bool,
    lift_decode: bool,
}

/// [`select_pair`] through the policy engine, the decision half: the prelude
/// settles what membership and availability settle, then one pair decision
/// over the open prefills with their capacity bits (`has_capacity`: the
/// admission window as the lock reads it) and their open partners with
/// their room bits; `excluded` are the ids a commit of this request just
/// refused, for the one re-decision. A prefill picker that declines a
/// lifted pool all over the thresholds is steered by the least-loaded
/// default, as the legacy path steers it. Runs inside the admission
/// callback, under the lock, as the legacy selection does: an open pool
/// with no capacity is the at-capacity verdict before any engine call, so
/// a newcomer behind the queue builds no view and moves no policy state.
#[expect(
    clippy::too_many_arguments,
    reason = "the engine seam threads every routing input plus the window"
)]
pub(crate) fn decide_pair(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    decider: &Decider,
    model_id: &str,
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    homogeneous_runtime: bool,
    has_capacity: &dyn Fn(&Arc<dyn Worker>) -> bool,
    inputs: PlacementInputs<'_>,
    excluded: &[WorkerId],
) -> Result<PairProposal, Box<PairFailure>> {
    let prefill_policy = policies.get_prefill_policy();
    let (live, open, pair_runtime) = match pair_prelude(
        registry,
        prefill_policy.as_ref(),
        policies.pd_pairing_mode(),
        model_id,
        pairs,
        wire,
        homogeneous_runtime,
        &inputs,
    ) {
        PairPrelude::Failed(failure) => return Err(failure),
        PairPrelude::AllVetoed(role) => {
            return settle_vetoed(registry, model_id, pairs, wire, has_capacity, &inputs, role)
                .map(PairProposal::Settled)
        }
        PairPrelude::Open {
            live,
            open,
            pair_runtime,
        } => (live, open, pair_runtime),
    };
    // The at-capacity verdict is the prelude's, before any engine call: no
    // open prefill under the window means the request waits for a slot (the
    // FIFO's own answer), with no view built and no policy state moved, which
    // keeps capacity hidden behind the queue for the policies whose selection
    // commits routing state.
    if !open.iter().any(|&i| has_capacity(&pairs.prefill[i])) {
        return Err(Box::new(PairFailure {
            role: WorkerRole::Prefill,
            verdict: PlacementFailure::PrefillAtCapacity,
        }));
    }
    let prefills: Vec<Arc<dyn Worker>> = open
        .iter()
        .map(|&i| Arc::clone(&pairs.prefill[i]))
        .collect();
    // Each open prefill's partner list as the index shares it between the
    // prefills of one pairing descriptor, and the live filter over a list,
    // which reads the decode and the pair's runtime alone: the engine keys
    // one pair class per list and builds its rows once.
    let list = |i: usize| &pairs.partners[open[i]];
    let partner_open = |d: &Arc<dyn Worker>| live.partner_open(d, pair_runtime);
    let request = HostRequest {
        model: model_id,
        prompt_tokens: decider.prefill_prompt_tokens(inputs.tokens),
        session_key: policies.sticky_key(inputs.headers, inputs.rid_key),
        tried: inputs.tried,
        overload_lifted: live.lift_prefill,
        phase: Phase::Prefill,
        has_capacity: None,
        sequence: inputs.sequence,
        affinity: None,
    };
    // The capture observes the open prefills before the engine reads them,
    // as the single seam does; the row follows the shed or the commit.
    cache_trace::begin_selection(&prefills);
    let (decision, lease, decodes) = decider.decide_pair(
        &request,
        &prefills,
        has_capacity,
        PairPartners {
            list: &list,
            open: &partner_open,
            lift_decode: live.lift_decode,
        },
        excluded,
    );
    if let Decision::Shed(shed) = &decision {
        // A decision that places nothing leaves its row with no worker, as
        // the legacy pick does when its policy declines.
        cache_trace::selection(prefill_policy.name(), "policy", &prefills, None);
        let failure = pair_shed(policies, decider, model_id, shed);
        if live.lift_prefill
            && failure.role == WorkerRole::Prefill
            && matches!(failure.verdict, PlacementFailure::PolicyDeclined(_))
        {
            let admitted = admitted_role_pool(pairs, wire, &inputs, WorkerRole::Prefill);
            if let Some(pair) = relaxed_pair(
                registry,
                model_id,
                pairs,
                wire,
                has_capacity,
                &inputs,
                &admitted,
            ) {
                return pair.map(PairProposal::Settled);
            }
        }
        return Err(failure);
    }
    Ok(PairProposal::Decided(PairDecision {
        decision,
        lease,
        decode_tokens: decider.decode_prompt_tokens(inputs.tokens),
        prefills,
        decodes,
        pair_runtime,
        prefill_over: live.prefill_over,
        lift_prefill: live.lift_prefill,
        lift_decode: live.lift_decode,
    }))
}

/// The host's verdict for a pair the engine shed: a prefill pool its
/// capacity filter emptied waits for a slot; a decode side with no open
/// partner is the decode role's unavailability; a picker's decline names
/// the policy of the side that declined; anything else is the prefill
/// role's unavailability.
fn pair_shed(
    policies: &PolicyRegistry,
    decider: &Decider,
    model_id: &str,
    shed: &Shed,
) -> Box<PairFailure> {
    let (role, verdict) = match shed.emptied_by {
        "has-capacity" => (WorkerRole::Prefill, PlacementFailure::PrefillAtCapacity),
        "partnered" | "partners" | "has-room" => {
            (WorkerRole::Decode, PlacementFailure::Unavailable)
        }
        _ if shed.reason == ShedReason::PolicyDeclined => {
            let decode_side = decider
                .engine()
                .profile_for(model_id, Phase::Decode)
                .is_some_and(|(index, _)| index == shed.trace.profile);
            if decode_side {
                (
                    WorkerRole::Decode,
                    PlacementFailure::PolicyDeclined(policies.get_decode_policy().name()),
                )
            } else {
                (
                    WorkerRole::Prefill,
                    PlacementFailure::PolicyDeclined(policies.get_prefill_policy().name()),
                )
            }
        }
        _ => (WorkerRole::Prefill, PlacementFailure::Unavailable),
    };
    Box::new(PairFailure { role, verdict })
}

/// The decode rows a pair decision read for the committed pair: the slice of
/// `decodes` that is the committed prefill's pair class, and the committed
/// decode's position in it. The decision hands back the rows of every pair
/// class in one slice, each class's rows together and in its partner list's
/// order, one row for every open partner and none for a closed one. So the
/// class is read outward from the committed row (`committed`) along
/// `partners`, the committed prefill's list: the row before belongs while
/// it is an earlier partner, the row after while it is a later one. A row
/// of another class never matches, because the partners past either end of
/// the class are closed and a closed decode has no row in any class. One
/// pass over the list, nothing allocated; a committed decode the list does
/// not hold is its own row alone. The reading does not hold under the
/// lifted decode veto, where a list keeps its least-loaded partners alone:
/// no decode row is written there.
fn partner_rows<'d>(
    partners: &[Arc<dyn Worker>],
    decodes: &'d [Arc<dyn Worker>],
    committed: usize,
) -> (&'d [Arc<dyn Worker>], usize) {
    let is_row = |partner: &Arc<dyn Worker>, row: usize| Arc::ptr_eq(partner, &decodes[row]);
    let Some(at) = partners.iter().position(|p| is_row(p, committed)) else {
        return (&decodes[committed..=committed], 0);
    };
    // Leftward, each earlier row of the class is an earlier partner.
    let (mut first, mut before) = (committed, at);
    while first > 0 {
        let earlier = partners[..before]
            .iter()
            .rposition(|p| is_row(p, first - 1));
        let Some(found) = earlier else { break };
        (first, before) = (first - 1, found);
    }
    // Rightward, each later row is a later partner.
    let (mut end, mut after) = (committed + 1, at + 1);
    while end < decodes.len() {
        let later = partners[after..].iter().position(|p| is_row(p, end));
        let Some(found) = later else { break };
        (end, after) = (end + 1, after + found + 1);
    }
    (&decodes[first..end], committed - first)
}

/// [`select_pair`] through the policy engine, the commit half, right after
/// the decision under the same hold of the admission lock: `has_capacity`
/// is the window re-read, since a regular or an unbounded dispatch moves a
/// worker's load outside this lock. The first target whose
/// prefill has capacity and is still open and whose decode partner is still
/// open is the pair, booked on both sides, the decode at its own estimate;
/// when every target misses, the missed ids go to
/// `missed` for the one re-decision and the request waits for a slot. The
/// room itself is claimed at dispatch (`pd_admission::admit_decode`), as
/// under the legacy path. A traced pair leaves its rows here: the prefill's
/// at the commit or the miss, the decode's over the committed prefill's
/// partners in `pairs`, the index the decision read.
pub(crate) fn commit_pair(
    policies: &PolicyRegistry,
    decider: &Decider,
    model_id: &str,
    pairs: &PdPairIndex,
    proposal: PairProposal,
    has_capacity: &dyn Fn(&Arc<dyn Worker>) -> bool,
    missed: &mut Vec<WorkerId>,
) -> Result<Pair, Box<PairFailure>> {
    let pd = match proposal {
        PairProposal::Settled(pair) => return Ok(pair),
        PairProposal::Decided(pd) => pd,
    };
    let fail =
        |role: WorkerRole, verdict: PlacementFailure| Box::new(PairFailure { role, verdict });
    let decode_of = |target: &Target| {
        target
            .pair
            .map(|pair| &pd.decodes[pair.decode_position as usize])
    };
    // Both sides are re-read under the lock, as the single seam re-reads
    // availability: the prefill open under its pool's lift beside its
    // window, the decode partner live.
    let accept = |target: &Target| {
        let prefill = &pd.prefills[target.position as usize];
        has_capacity(prefill)
            && open_under(prefill, pd.lift_prefill)
            && decode_of(target).is_some_and(|d| partner_live(d, pd.lift_decode, pd.pair_runtime))
    };
    // The policy the trace rows and the selection metric name.
    let prefill_policy = policies.get_prefill_policy().name();
    let Some(target) = decider.commit_pair(&pd.lease, pd.decode_tokens, &pd.decision, accept)
    else {
        // Every target missed: the decision's row names no worker, and the
        // one re-decision writes its own.
        cache_trace::selection(prefill_policy, "policy", &pd.prefills, None);
        if let Some(pick) = pd.decision.pick() {
            for target in pick.targets.iter() {
                missed.push(target.id);
                missed.extend(target.pair.map(|pair| pair.decode_id));
            }
        }
        return Err(fail(
            WorkerRole::Prefill,
            PlacementFailure::PrefillAtCapacity,
        ));
    };
    cache_trace::selection(
        prefill_policy,
        "policy",
        &pd.prefills,
        Some(target.position as usize),
    );
    let prefill = Arc::clone(&pd.prefills[target.position as usize]);
    let decode_row = target
        .pair
        .map(|pair| pair.decode_position as usize)
        .ok_or_else(|| fail(WorkerRole::Decode, PlacementFailure::Unavailable))?;
    let decode = Arc::clone(&pd.decodes[decode_row]);
    if pd.prefill_over || pd.lift_decode {
        Metrics::record_worker_overload_fallback(STAGE_SELECTION);
        debug!(
            branch = BRANCH_ALL_OVERLOADED_FALLBACK,
            prefill_pool = pd.prefill_over,
            decode_pool = pd.lift_decode,
            prefill = %prefill.redacted_url(),
            decode = %decode.redacted_url(),
            model_id,
            "Veto fallback: a pool over the overload thresholds decided with the veto lifted"
        );
    }
    let runtime = prefill.metadata().spec.runtime_type;
    Metrics::record_worker_selection(
        metrics_labels::WORKER_PREFILL,
        prefill.connection_mode().as_metric_label(),
        model_id,
        prefill_policy,
    );
    // A decode paired by load under the lift is the veto fallback's pick,
    // counted above, not a selection of the decode policy.
    if !pd.lift_decode {
        let decode_policy = policies.get_decode_policy().name();
        if cache_trace::enabled() {
            // The decode side's row, as the legacy pair leaves it: the
            // committed prefill's partners among the decode rows the
            // decision read, observed here at the commit, under the same
            // hold of the lock.
            let partners: &[Arc<dyn Worker>] =
                match pairs.prefill.iter().position(|p| Arc::ptr_eq(p, &prefill)) {
                    Some(at) => &pairs.partners[at],
                    None => &[],
                };
            let (rows, picked) = partner_rows(partners, &pd.decodes, decode_row);
            cache_trace::begin_selection(rows);
            cache_trace::selection(decode_policy, "policy", rows, Some(picked));
        }
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            decode.connection_mode().as_metric_label(),
            model_id,
            decode_policy,
        );
    }
    debug!(
        prefill = %prefill.redacted_url(),
        decode = %decode.redacted_url(),
        "Selected PD pair"
    );
    Ok(Pair {
        prefill,
        decode,
        runtime,
    })
}

/// [`select_pair`] through the policy engine inside the admission callback,
/// as the routers run it under `--policy-engine engine`: the decision over
/// the window as the lock reads it (`has_capacity`: the routers' read of the
/// admission context, every prefill open with admission off), its commit
/// under the same hold, and one re-decision with the ids a refused commit
/// missed before the at-capacity verdict queues the request. The window is
/// read again at each step, never remembered from the one before.
#[expect(
    clippy::too_many_arguments,
    reason = "the engine seam threads every routing input plus the window"
)]
pub(crate) fn place_pair_through_engine(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    decider: &Decider,
    model_id: &str,
    pairs: &PdPairIndex,
    wire: Option<WireConstraint>,
    homogeneous_runtime: bool,
    has_capacity: &dyn Fn(&Arc<dyn Worker>) -> bool,
    inputs: PlacementInputs<'_>,
) -> Result<Pair, Box<PairFailure>> {
    let decide = |excluded: &[WorkerId]| {
        decide_pair(
            registry,
            policies,
            decider,
            model_id,
            pairs,
            wire,
            homogeneous_runtime,
            has_capacity,
            inputs,
            excluded,
        )
    };
    let mut missed = Vec::new();
    let refused = match commit_pair(
        policies,
        decider,
        model_id,
        pairs,
        decide(&[])?,
        has_capacity,
        &mut missed,
    ) {
        Ok(pair) => return Ok(pair),
        Err(failure) => failure,
    };
    if missed.is_empty() {
        return Err(refused);
    }
    commit_pair(
        policies,
        decider,
        model_id,
        pairs,
        decide(&missed)?,
        has_capacity,
        &mut Vec::new(),
    )
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
    decode: &mut Vec<Arc<dyn Worker>>,
) {
    if !admission_prefilters(decode_policy, headers) {
        return;
    }
    // The candidates with a room move to the front, in their order; the
    // ones without end up behind them and are cut off only when there is
    // someone to keep and someone to cut.
    let mut kept = 0;
    for at in 0..decode.len() {
        if pd_admission::has_free_room(decode[at].as_ref()) {
            decode.swap(kept, at);
            kept += 1;
        }
    }
    if kept > 0 && kept < decode.len() {
        decode.truncate(kept);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::{Cell, RefCell},
        collections::{BTreeMap, HashMap},
        sync::Mutex,
        time::Duration,
    };

    use axum::http::StatusCode;
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};
    use tracing_subscriber::Layer;

    use super::*;
    use crate::{
        config::types::{
            ManualAssignmentMode, PdPairingMode, PolicyConfig, PolicyEngineMode,
            RoutingKeyOverrideConfig,
        },
        observability::logging::test_support::test_subscriber,
        policies::{
            CacheAwareConfig, CacheAwarePolicy, ConsistentHashingPolicy, LeastLoadPolicy,
            PowerOfTwoPolicy, RoundRobinPolicy,
        },
        worker::{
            load_state::{one_rank_load, TestFeed},
            runtime::StallReason,
            BasicWorkerBuilder, ModelCard, PdWire, WorkerType,
        },
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

    /// One pair through the engine seam, its two halves apart: the decision
    /// with `has_capacity` as one read of the window, the commit with
    /// `capacity` as a later read of it (the routers take both under one
    /// hold of the lock), the misses of a refused commit excluded from the
    /// next decision through `missed`.
    fn decided_pair(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        has_capacity: &dyn Fn(&Arc<dyn Worker>) -> bool,
        capacity: &dyn Fn(&Arc<dyn Worker>) -> bool,
        missed: &mut Vec<WorkerId>,
    ) -> Result<Pair, Box<PairFailure>> {
        let decider = pair_decider(policies, &PlacementInputs::default())
            .expect("the engine serves the pair");
        let pairs = pairs_of(registry, policies);
        let excluded = std::mem::take(missed);
        let proposal = decide_pair(
            registry,
            policies,
            decider,
            MODEL,
            &pairs,
            None,
            true,
            has_capacity,
            PlacementInputs::default(),
            &excluded,
        )?;
        commit_pair(policies, decider, MODEL, &pairs, proposal, capacity, missed)
    }

    /// Selections per worker URL over `requests` pairs decided through the
    /// engine with an open window.
    fn decided_hits(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        requests: usize,
    ) -> BTreeMap<String, usize> {
        let mut hits = BTreeMap::new();
        let mut missed = Vec::new();
        for _ in 0..requests {
            let pair = decided_pair(registry, policies, &|_| true, &|_| true, &mut missed)
                .expect("a pair is open");
            *hits.entry(pair.prefill.url().to_string()).or_insert(0) += 1;
            *hits.entry(pair.decode.url().to_string()).or_insert(0) += 1;
        }
        hits
    }

    /// The two PD cohorts' policies with the engine hosted behind the switch.
    fn engine_cohort_policies() -> PolicyRegistry {
        let round_robin = PolicyConfig::RoundRobin;
        cohort_policies().with_policy_engine(PolicyEngineMode::Engine, &round_robin, &round_robin)
    }

    /// Under `--policy-engine engine` the pair is decided through the engine:
    /// round robin on both roles rotates the prefills and, within the chosen
    /// prefill's partners, the decodes; every pair is one decision.
    #[test]
    fn with_the_engine_switch_a_pair_rotates_both_pools() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let picks = decided_hits(&registry, &policies, 8);
        assert_eq!(picks["grpc://p:1"], 4, "{picks:?}");
        assert_eq!(picks["grpc://p:2"], 4, "{picks:?}");
        assert_eq!(picks["grpc://d:1"], 4, "{picks:?}");
        assert_eq!(picks["grpc://d:2"], 4, "{picks:?}");
        let decider = policies.decider().expect("hosted");
        assert_eq!(decider.decisions(), 8, "one decision per pair");
    }

    /// The at-capacity verdict is the prelude's: with no open prefill under
    /// the window the engine decides nothing (no view built, no policy state
    /// moved) and the failure is the one the FIFO queues on; the locked
    /// placement answers the same.
    #[test]
    fn a_pool_with_no_capacity_queues_before_the_engine_decides() {
        use crate::worker::PrefillCandidateError as _;

        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let decider = policies.decider().expect("hosted");
        let pairs = pairs_of(&registry, &policies);
        let none = |_: &Arc<dyn Worker>| false;
        let failure = decide_pair(
            &registry,
            &policies,
            decider,
            MODEL,
            &pairs,
            None,
            true,
            &none,
            PlacementInputs::default(),
            &[],
        )
        .err()
        .expect("no prefill has capacity");
        assert!(
            failure.is_at_capacity(),
            "the verdict is the at-capacity one"
        );
        assert_eq!(decider.decisions(), 0, "no decision was made");
        let mut missed = Vec::new();
        let placed = decided_pair(&registry, &policies, &|_| true, &|_| true, &mut missed)
            .expect("an open pair with capacity is placed");
        assert_eq!(placed.prefill.url(), "grpc://p:1");
        assert_eq!(decider.decisions(), 1);
    }

    /// Under the switch a decode partner whose window is full is kept out of
    /// the pair by the engine's room bit, and the rooms stay the host's to
    /// claim at dispatch (the engine books nothing).
    #[test]
    fn with_the_engine_switch_a_full_decode_window_is_kept_out_of_the_pair() {
        let registry = gated_pd_registry(2);
        let policies = engine_cohort_policies();
        let full = registry.get_by_url("grpc://d:1").expect("registered");
        assert!(full.try_admit_pd(2, 2), "both rooms claimed");
        let picks = decided_hits(&registry, &policies, 10);
        assert_eq!(
            picks.get("grpc://d:1").copied().unwrap_or(0),
            0,
            "{picks:?}"
        );
        assert_eq!(picks["grpc://d:2"], 10, "{picks:?}");
        assert_eq!(picks["grpc://p:1"], 10, "{picks:?}");

        full.release_pd(2);
        let picks = decided_hits(&registry, &policies, 10);
        assert_eq!(picks["grpc://d:1"], 5, "{picks:?}");
        assert_eq!(picks["grpc://d:2"], 5, "{picks:?}");
    }

    /// The commit re-validates the prefill's window under the lock: a window
    /// that closed between the decision and the commit refuses every target,
    /// the request waits for a slot, and the missed pair (its prefill and its
    /// decode) is recorded for the one re-decision, which the next commit
    /// clears.
    #[test]
    fn the_commit_re_validates_the_window_and_records_the_missed_pair() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let decider = policies.decider().expect("hosted");
        let mut missed = Vec::new();
        let refused = decided_pair(&registry, &policies, &|_| true, &|_| false, &mut missed)
            .err()
            .expect("the window closed under the lock");
        assert_eq!(refused.role, WorkerRole::Prefill);
        assert!(matches!(
            refused.verdict,
            PlacementFailure::PrefillAtCapacity
        ));
        assert_eq!(missed.len(), 2, "the prefill and its decode: {missed:?}");
        assert_eq!(decider.decisions(), 1);
        let pair = decided_pair(&registry, &policies, &|_| true, &|_| true, &mut missed)
            .expect("the re-decision pairs");
        assert_eq!(pair.decode.url(), "grpc://d:1");
        assert_eq!(decider.decisions(), 2, "one re-decision");
        assert!(missed.is_empty(), "a committed pair leaves no miss behind");
    }

    /// The seam the routers call, with the window closing between the
    /// decision and the commit. The prelude reads the window once and the
    /// engine's view once per prefill, so its fourth read is the first
    /// commit's, and that read finds the decided prefill's window closed:
    /// the commit refuses, the miss is recorded, the one re-decision pairs
    /// the other prefill, and one call places the pair with two decisions.
    #[test]
    fn the_seam_re_decides_once_when_the_window_closes_before_the_commit() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let decider = policies.decider().expect("hosted");
        let pairs = pairs_of(&registry, &policies);
        let reads = Cell::new(0_u32);
        let closed: RefCell<Option<String>> = RefCell::new(None);
        let has_capacity = |worker: &Arc<dyn Worker>| {
            reads.set(reads.get() + 1);
            if reads.get() == 4 {
                *closed.borrow_mut() = Some(worker.url().to_string());
            }
            closed.borrow().as_deref() != Some(worker.url())
        };
        let pair = place_pair_through_engine(
            &registry,
            &policies,
            decider,
            MODEL,
            &pairs,
            None,
            true,
            &has_capacity,
            PlacementInputs::default(),
        )
        .expect("the re-decision pairs the other prefill");
        let closed = closed
            .borrow()
            .clone()
            .expect("the first commit read the window");
        assert_ne!(
            pair.prefill.url(),
            closed,
            "the closed prefill is not paired"
        );
        assert_eq!(pair.decode.url(), "grpc://d:1");
        assert_eq!(decider.decisions(), 2, "one re-decision");
    }

    /// A pair decided through the engine scopes the cache-trace capture as
    /// the single path does: the prefill decision is one selection record,
    /// so a traced request's dispatch header names the policy and the worker
    /// instead of an empty `selections` list.
    #[tokio::test]
    async fn a_pair_decided_through_the_engine_records_its_selection_in_the_cache_trace() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let decider = policies.decider().expect("hosted");
        let pairs = pairs_of(&registry, &policies);
        let (pair, header) = cache_trace::scope(true, async {
            let pair = place_pair_through_engine(
                &registry,
                &policies,
                decider,
                MODEL,
                &pairs,
                None,
                true,
                &|_| true,
                PlacementInputs::default(),
            )
            .expect("the engine pairs");
            (pair, cache_trace::dispatch(None, 1, vec![], true, "pd"))
        })
        .await;
        let header = header.expect("the request asked for its decision record");
        let record: serde_json::Value =
            serde_json::from_str(&header).expect("the header is the record's JSON");
        let selections = record["selections"]
            .as_array()
            .expect("the record lists its selections");
        assert!(
            !selections.is_empty(),
            "a pair decided through the engine left no selection in the trace: {record}"
        );
        assert_eq!(
            selections[0]["worker"],
            serde_json::json!(pair.prefill.redacted_url()),
            "the first selection is the prefill decision: {record}"
        );
    }

    /// One pair placed through the engine seam under a capture the request
    /// asked for: the placement's result and the rows the capture holds, as
    /// the evidence line lists them.
    async fn traced_pair(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        has_capacity: &dyn Fn(&Arc<dyn Worker>) -> bool,
    ) -> (Result<Pair, Box<PairFailure>>, Vec<serde_json::Value>) {
        let decider = policies.decider().expect("hosted");
        let pairs = pairs_of(registry, policies);
        cache_trace::scope(true, async {
            let placed = place_pair_through_engine(
                registry,
                policies,
                decider,
                MODEL,
                &pairs,
                None,
                true,
                has_capacity,
                PlacementInputs::default(),
            );
            (placed, cache_trace::captured_selections())
        })
        .await
    }

    /// The workers a row observed, by URL, sorted.
    fn observed(row: &serde_json::Value) -> Vec<&str> {
        let mut urls: Vec<&str> = row["candidates"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|candidate| candidate["worker"].as_str())
            .collect();
        urls.sort_unstable();
        urls
    }

    /// A traced pair under the engine leaves the two rows the legacy pair
    /// leaves: the prefill pick under the prefill policy's name over the
    /// open prefills, then the decode pick under the decode policy's name
    /// over the decode rows the decision read. A prediction left in the
    /// capture before the decision does not reach either row.
    #[tokio::test]
    async fn a_traced_pair_names_each_side_under_its_own_policy() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let round_robin = PolicyConfig::RoundRobin;
        let power_of_two = PolicyConfig::PowerOfTwo {
            load_metric: Default::default(),
        };
        let policies = PolicyRegistry::new(round_robin.clone());
        policies.set_prefill_policy(Arc::new(RoundRobinPolicy::new()));
        policies.set_decode_policy(Arc::new(PowerOfTwoPolicy::new()));
        let policies =
            policies.with_policy_engine(PolicyEngineMode::Engine, &round_robin, &power_of_two);
        let decider = policies.decider().expect("hosted");
        let pairs = pairs_of(&registry, &policies);
        let (pair, rows) = cache_trace::scope(true, async {
            cache_trace::prediction(serde_json::json!({"left": "by an earlier pick"}));
            let pair = place_pair_through_engine(
                &registry,
                &policies,
                decider,
                MODEL,
                &pairs,
                None,
                true,
                &|_| true,
                PlacementInputs::default(),
            )
            .expect("the engine pairs");
            (pair, cache_trace::captured_selections())
        })
        .await;
        assert_eq!(rows.len(), 2, "a prefill row and a decode row: {rows:?}");
        let (prefill, decode) = (&rows[0], &rows[1]);
        assert_eq!(prefill["policy"], "round_robin");
        assert_eq!(prefill["origin"], "policy");
        assert_eq!(
            prefill["worker"],
            serde_json::json!(pair.prefill.redacted_url())
        );
        assert_eq!(observed(prefill), ["grpc://p:1", "grpc://p:2"]);
        assert!(
            prefill["prediction"].is_null(),
            "the decision's row starts clean: {prefill}"
        );
        assert_eq!(decode["policy"], "power_of_two");
        assert_eq!(decode["origin"], "policy");
        assert_eq!(
            decode["worker"],
            serde_json::json!(pair.decode.redacted_url())
        );
        assert_eq!(observed(decode), ["grpc://d:1", "grpc://d:2"]);
        assert!(decode["prediction"].is_null(), "{decode}");
    }

    /// A pair the engine sheds leaves one row that names no worker, under
    /// the prefill policy's name, so a refused traced request still carries
    /// its decision.
    #[tokio::test]
    async fn a_pair_the_engine_sheds_leaves_one_row_without_a_worker() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        // The prelude reads an open window; the engine's own read finds it
        // closed and sheds on its capacity filter.
        let first_read = Cell::new(true);
        let has_capacity = |_: &Arc<dyn Worker>| first_read.replace(false);
        let (shed, rows) = traced_pair(&registry, &policies, &has_capacity).await;
        let shed = shed.err().expect("the engine shed the pool");
        assert!(matches!(shed.verdict, PlacementFailure::PrefillAtCapacity));
        assert_eq!(rows.len(), 1, "the shed decision's row alone: {rows:?}");
        assert_eq!(rows[0]["policy"], "round_robin");
        assert!(rows[0]["worker"].is_null(), "{rows:?}");
        assert_eq!(observed(&rows[0]), ["grpc://p:1"]);
    }

    /// A commit that refuses every target leaves the decision's row with no
    /// worker, and the one re-decision writes its own rows after it: the
    /// prefill it pairs, then the decode.
    #[tokio::test]
    async fn a_refused_pair_commit_leaves_its_row_before_the_re_decisions() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        // The fourth read of the window is the first commit's: it closes the
        // decided prefill, as in the seam's own re-decision test.
        let reads = Cell::new(0_u32);
        let closed: RefCell<Option<String>> = RefCell::new(None);
        let has_capacity = |worker: &Arc<dyn Worker>| {
            reads.set(reads.get() + 1);
            if reads.get() == 4 {
                *closed.borrow_mut() = Some(worker.url().to_string());
            }
            closed.borrow().as_deref() != Some(worker.url())
        };
        let (pair, rows) = traced_pair(&registry, &policies, &has_capacity).await;
        let pair = pair.expect("the re-decision pairs the other prefill");
        assert_eq!(
            rows.len(),
            3,
            "the refused row, then the pair's two: {rows:?}"
        );
        assert!(rows[0]["worker"].is_null(), "{rows:?}");
        assert_eq!(observed(&rows[0]), ["grpc://p:1", "grpc://p:2"]);
        assert_eq!(
            rows[1]["worker"],
            serde_json::json!(pair.prefill.redacted_url())
        );
        assert_eq!(
            rows[2]["worker"],
            serde_json::json!(pair.decode.redacted_url())
        );
    }

    /// A decode paired by load under the lift is the veto fallback's pick
    /// and no selection of the decode policy: the trace carries the prefill
    /// row alone, as the selection metric counts the prefill alone.
    #[tokio::test]
    async fn a_decode_paired_by_load_under_the_lift_leaves_no_decode_row() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        for url in ["grpc://d:1", "grpc://d:2"] {
            let worker = registry.get_by_url(url).expect("registered");
            registry.set_worker_overloaded(&worker, true);
        }
        let (pair, rows) = traced_pair(&registry, &policies, &|_| true).await;
        let pair = pair.expect("a pair is open");
        assert_eq!(rows.len(), 1, "the prefill row alone: {rows:?}");
        assert_eq!(
            rows[0]["worker"],
            serde_json::json!(pair.prefill.redacted_url())
        );
    }

    /// The decode row lists the committed prefill's partners, as the legacy
    /// row lists the pick's: with two pairing descriptors in the fleet the
    /// decision reads the decode rows of both, and the other prefill's
    /// partners stay out of the row.
    #[tokio::test]
    async fn a_decode_row_lists_the_committed_prefills_partners_alone() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("MooncakeConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:3", WorkerType::Decode, Some("MooncakeConnector")),
            ("grpc://d:4", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        let policies = engine_cohort_policies();
        // Round robin takes the two prefills in turn: one pair per class.
        let mut prefills = Vec::new();
        for _ in 0..2 {
            let (pair, rows) = traced_pair(&registry, &policies, &|_| true).await;
            let pair = pair.expect("a pair is open");
            assert_eq!(rows.len(), 2, "{rows:?}");
            let partners = match pair.prefill.url() {
                "grpc://p:1" => ["grpc://d:1", "grpc://d:2"],
                _ => ["grpc://d:3", "grpc://d:4"],
            };
            assert_eq!(observed(&rows[0]), ["grpc://p:1", "grpc://p:2"]);
            assert_eq!(observed(&rows[1]), partners, "{rows:?}");
            assert_eq!(
                rows[1]["worker"],
                serde_json::json!(pair.decode.redacted_url())
            );
            prefills.push(pair.prefill.url().to_owned());
        }
        prefills.sort_unstable();
        assert_eq!(prefills, ["grpc://p:1", "grpc://p:2"]);
    }

    /// The decode rows of `pairs` traced pairs over `registry` under round
    /// robin on both roles: each pair's prefill, its decode, and the workers
    /// its decode row lists and names.
    async fn decode_rows(
        registry: &WorkerRegistry,
        pairs: usize,
    ) -> Vec<(String, String, Vec<String>, serde_json::Value)> {
        let policies = engine_cohort_policies();
        let mut read = Vec::new();
        for _ in 0..pairs {
            let (pair, rows) = traced_pair(registry, &policies, &|_| true).await;
            let pair = pair.expect("a pair is open");
            assert_eq!(rows.len(), 2, "a prefill row and a decode row: {rows:?}");
            read.push((
                pair.prefill.url().to_owned(),
                pair.decode.url().to_owned(),
                observed(&rows[1]).into_iter().map(str::to_owned).collect(),
                rows[1]["worker"].clone(),
            ));
        }
        read
    }

    /// The decode row is its class's rows whichever decode of the class is
    /// the pick: the first, with the class's other row behind it, or the
    /// last, with the other class's rows beside it. Round robin takes the
    /// two prefills in turn and, within a class, its two decodes in turn, so
    /// four pairs pick every decode once.
    #[tokio::test]
    async fn a_decode_row_is_its_class_whichever_decode_is_the_pick() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("MooncakeConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:3", WorkerType::Decode, Some("MooncakeConnector")),
            ("grpc://d:4", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        let mut picked = Vec::new();
        for (prefill, decode, listed, named) in decode_rows(&registry, 4).await {
            let class = match prefill.as_str() {
                "grpc://p:1" => ["grpc://d:1", "grpc://d:2"],
                _ => ["grpc://d:3", "grpc://d:4"],
            };
            assert_eq!(listed, class, "{prefill} with {decode}");
            assert_eq!(named, serde_json::json!(decode), "the row names the pick");
            picked.push(decode);
        }
        picked.sort_unstable();
        assert_eq!(
            picked,
            ["grpc://d:1", "grpc://d:2", "grpc://d:3", "grpc://d:4"]
        );
    }

    /// Two classes that share a decode keep to their own rows: a decode
    /// that reports no connector pairs with both prefills, so it has a row
    /// in each class, and each class's row ends at its own copy.
    #[tokio::test]
    async fn a_decode_row_keeps_to_its_class_beside_a_shared_decode() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("MooncakeConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, None),
            ("grpc://d:3", WorkerType::Decode, Some("MooncakeConnector")),
        ]);
        let mut pairs = Vec::new();
        for (prefill, decode, listed, named) in decode_rows(&registry, 4).await {
            let class = match prefill.as_str() {
                "grpc://p:1" => ["grpc://d:1", "grpc://d:2"],
                _ => ["grpc://d:2", "grpc://d:3"],
            };
            assert_eq!(listed, class, "{prefill} with {decode}");
            assert_eq!(named, serde_json::json!(decode), "the row names the pick");
            pairs.push((prefill, decode));
        }
        pairs.sort_unstable();
        let pair = |prefill: &str, decode: &str| (prefill.to_owned(), decode.to_owned());
        assert_eq!(
            pairs,
            [
                pair("grpc://p:1", "grpc://d:1"),
                pair("grpc://p:1", "grpc://d:2"),
                pair("grpc://p:2", "grpc://d:2"),
                pair("grpc://p:2", "grpc://d:3"),
            ],
            "each class picks its own two decodes, the shared one from both"
        );
    }

    /// A partner that is not ready has no row in the decision, so it stays
    /// out of the decode row: the row is the open partners on either side
    /// of it, whichever of them is the pick.
    #[tokio::test]
    async fn a_decode_row_leaves_a_closed_partner_out() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:3", WorkerType::Decode, Some("NixlConnector")),
        ]);
        registry
            .get_by_url("grpc://d:2")
            .expect("registered")
            .set_status(WorkerStatus::NotReady);
        let mut picked = Vec::new();
        for (_, decode, listed, named) in decode_rows(&registry, 2).await {
            assert_eq!(listed, ["grpc://d:1", "grpc://d:3"], "the open partners");
            assert_eq!(named, serde_json::json!(decode), "the row names the pick");
            picked.push(decode);
        }
        picked.sort_unstable();
        assert_eq!(picked, ["grpc://d:1", "grpc://d:3"]);
    }

    /// A committed decode its prefill's list does not hold is its own row
    /// alone, never the rows of every class. The index and the decision are
    /// one snapshot on the request path, so this is the reading of a list
    /// that is not the decision's.
    #[test]
    fn a_decode_its_list_does_not_hold_is_its_own_row_alone() {
        let registry = pd_registry(&[
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:3", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let decodes: Vec<Arc<dyn Worker>> = ["grpc://d:1", "grpc://d:2", "grpc://d:3"]
            .iter()
            .map(|url| registry.get_by_url(url).expect("registered"))
            .collect();
        for list in [&decodes[..0], &decodes[..1]] {
            let (rows, picked) = partner_rows(list, &decodes, 2);
            assert_eq!(rows.len(), 1, "the committed row alone");
            assert!(Arc::ptr_eq(&rows[0], &decodes[2]));
            assert_eq!(picked, 0);
        }
    }

    /// The pair is one decision: a prediction the engine leaves for it
    /// between the decision and the commit rides the prefill row, and the
    /// decode row carries none. The prefill row is written before the decode
    /// side's observation begins, so that beginning never clears it.
    #[tokio::test]
    async fn the_pairs_prediction_rides_the_prefill_row() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let decider = policies.decider().expect("hosted");
        let pairs = pairs_of(&registry, &policies);
        let rows = cache_trace::scope(true, async {
            let proposal = decide_pair(
                &registry,
                &policies,
                decider,
                MODEL,
                &pairs,
                None,
                true,
                &|_| true,
                PlacementInputs::default(),
                &[],
            )
            .expect("a pair is open");
            // Where a record of the decision itself would land.
            cache_trace::prediction(serde_json::json!({"decision": 7}));
            commit_pair(
                &policies,
                decider,
                MODEL,
                &pairs,
                proposal,
                &|_| true,
                &mut Vec::new(),
            )
            .expect("the commit pairs");
            cache_trace::captured_selections()
        })
        .await;
        assert_eq!(rows.len(), 2, "{rows:?}");
        assert_eq!(rows[0]["prediction"], serde_json::json!({"decision": 7}));
        assert!(rows[1]["prediction"].is_null(), "{rows:?}");
    }

    /// The prelude's own verdicts are settled before any engine call and
    /// write no row, as the legacy path writes none there: a pool with no
    /// capacity queues the request with an empty trace.
    #[tokio::test]
    async fn a_pair_the_prelude_settles_writes_no_row() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let (queued, rows) = traced_pair(&registry, &policies, &|_| false).await;
        let queued = queued.err().expect("no prefill has capacity");
        assert!(matches!(
            queued.verdict,
            PlacementFailure::PrefillAtCapacity
        ));
        assert_eq!(policies.decider().expect("hosted").decisions(), 0);
        assert!(
            rows.is_empty(),
            "the prelude's verdict writes no row: {rows:?}"
        );
    }

    /// A window that closes between the prelude's read and the engine's:
    /// the prelude saw a prefill with capacity, the engine's capacity filter
    /// then empties the pool, and the host's verdict is still the one the
    /// request waits on (`pair_shed` maps the capacity shed to it).
    #[test]
    fn a_capacity_shed_past_the_prelude_is_the_at_capacity_verdict() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let decider = policies.decider().expect("hosted");
        let pairs = pairs_of(&registry, &policies);
        let first_read = Cell::new(true);
        let has_capacity = |_: &Arc<dyn Worker>| first_read.replace(false);
        let shed = place_pair_through_engine(
            &registry,
            &policies,
            decider,
            MODEL,
            &pairs,
            None,
            true,
            &has_capacity,
            PlacementInputs::default(),
        )
        .err()
        .expect("the engine shed the pool on its capacity filter");
        assert_eq!(shed.role, WorkerRole::Prefill);
        assert!(matches!(shed.verdict, PlacementFailure::PrefillAtCapacity));
        assert_eq!(decider.decisions(), 1, "the engine decided once");
    }

    /// A prefill pool the window admits nothing of waits for a slot (the
    /// engine's capacity stage empties it), and a decode pool with nothing
    /// open is the decode role's unavailability, as the legacy path judges
    /// both.
    #[test]
    fn the_engines_pair_sheds_map_to_the_roles_verdicts() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let mut missed = Vec::new();
        let at_capacity = decided_pair(&registry, &policies, &|_| false, &|_| true, &mut missed)
            .err()
            .expect("no prefill has capacity");
        assert_eq!(at_capacity.role, WorkerRole::Prefill);
        assert!(matches!(
            at_capacity.verdict,
            PlacementFailure::PrefillAtCapacity
        ));

        let decode = registry.get_by_url("grpc://d:1").expect("registered");
        decode.set_status(WorkerStatus::NotReady);
        let down = decided_pair(&registry, &policies, &|_| true, &|_| true, &mut missed)
            .err()
            .expect("no partner is open");
        assert_eq!(down.role, WorkerRole::Decode);
        assert!(matches!(down.verdict, PlacementFailure::Unavailable));
        let legacy = pair_from(&registry, &cohort_policies())
            .err()
            .expect("legacy agrees");
        assert_eq!(legacy.role, WorkerRole::Decode);
        assert!(matches!(legacy.verdict, PlacementFailure::Unavailable));
    }

    /// A request the sticky routing-key override keys stays on the legacy
    /// pair path; without a key the engine decides.
    #[test]
    fn a_keyed_pair_keeps_the_legacy_path_under_the_sticky_override() {
        let round_robin = PolicyConfig::RoundRobin;
        let policies = PolicyRegistry::with_override(
            round_robin.clone(),
            RoutingKeyOverrideConfig {
                enabled: true,
                ..Default::default()
            },
        )
        .with_policy_engine(PolicyEngineMode::Engine, &round_robin, &round_robin);
        assert!(pair_decider(&policies, &PlacementInputs::default()).is_some());
        let keyed = PlacementInputs {
            rid_key: Some("session-7"),
            ..PlacementInputs::default()
        };
        assert!(pair_decider(&policies, &keyed).is_none());
    }

    /// A manual side pins the key placement derives from a configured
    /// routing-key header, override off: that keyed request stays on the
    /// legacy pair path, the unkeyed one takes the engine, and sides that
    /// pin nothing take it keyed or not.
    #[test]
    fn a_keyed_pair_with_a_manual_side_keeps_the_legacy_path_by_a_configured_header() {
        let configured = || RoutingKeyOverrideConfig {
            enabled: false,
            headers: vec!["x-session-id".into()],
            ..Default::default()
        };
        let manual = PolicyConfig::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: 3600,
            assignment_mode: ManualAssignmentMode::Random,
        };
        let round_robin = PolicyConfig::RoundRobin;
        let mut headers = HeaderMap::new();
        headers.insert("x-session-id", "conv-7".parse().unwrap());
        let keyed = PlacementInputs {
            headers: Some(&headers),
            ..PlacementInputs::default()
        };
        let pinning = PolicyRegistry::with_override(manual.clone(), configured())
            .with_policy_engine(PolicyEngineMode::Engine, &manual, &round_robin);
        assert!(pair_decider(&pinning, &PlacementInputs::default()).is_some());
        assert!(pair_decider(&pinning, &keyed).is_none());
        let plain = PolicyRegistry::with_override(round_robin.clone(), configured())
            .with_policy_engine(PolicyEngineMode::Engine, &round_robin, &round_robin);
        assert!(pair_decider(&plain, &keyed).is_some());
    }

    /// The engine's proposal of one pair, decided with an open window and
    /// nothing excluded, for a test that reads the decision itself.
    fn proposed_pair(registry: &WorkerRegistry, policies: &PolicyRegistry) -> PairDecision {
        let decider = pair_decider(policies, &PlacementInputs::default())
            .expect("the engine serves the pair");
        let pairs = pairs_of(registry, policies);
        let proposal = decide_pair(
            registry,
            policies,
            decider,
            MODEL,
            &pairs,
            None,
            true,
            &|_| true,
            PlacementInputs::default(),
            &[],
        )
        .expect("a pair is open");
        match proposal {
            PairProposal::Decided(pd) => pd,
            PairProposal::Settled(_) => panic!("the engine decides the pair"),
        }
    }

    /// The decode of a proposal's first target.
    fn proposed_decode(pd: &PairDecision) -> Arc<dyn Worker> {
        let target = pd
            .decision
            .pick()
            .expect("a pair")
            .targets
            .first()
            .copied()
            .expect("a target");
        Arc::clone(&pd.decodes[target.pair.expect("the decode side").decode_position as usize])
    }

    /// Prefills of one pairing descriptor share one partner list, so the
    /// decision builds one set of decode rows for all of them (P + D rows,
    /// never P x D) and still rotates both pools in full.
    #[test]
    fn with_the_engine_switch_prefills_sharing_a_partner_list_share_one_pair_class() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:3", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let pairs = pairs_of(&registry, &policies);
        assert!(
            pairs
                .partners
                .iter()
                .all(|list| Arc::ptr_eq(list, &pairs.partners[0])),
            "one descriptor, one shared list"
        );
        let pd = proposed_pair(&registry, &policies);
        assert_eq!(pd.prefills.len(), 3);
        assert_eq!(
            pd.decodes.len(),
            2,
            "one row per decode, not one per (prefill, decode)"
        );
        let picks = decided_hits(&registry, &policies, 12);
        for prefill in ["grpc://p:1", "grpc://p:2", "grpc://p:3"] {
            assert_eq!(picks[prefill], 4, "{picks:?}");
        }
        for decode in ["grpc://d:1", "grpc://d:2"] {
            assert_eq!(picks[decode], 6, "{picks:?}");
        }
    }

    /// The commit re-checks the prefill's availability under the lock as it
    /// does the decode partner's: a prefill that went unavailable between
    /// the decision and the commit refuses its targets, the miss is recorded
    /// for the one re-decision, and the re-decision pairs the other prefill.
    #[test]
    fn the_pair_commit_re_checks_the_prefill_as_it_does_the_decode() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://p:2", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let decider = policies.decider().expect("hosted");
        let pd = proposed_pair(&registry, &policies);
        let chosen = Arc::clone(
            &pd.prefills[pd
                .decision
                .pick()
                .expect("a pair")
                .targets
                .first()
                .expect("a target")
                .position as usize],
        );
        chosen.set_status(WorkerStatus::NotReady);
        let mut missed = Vec::new();
        let refused = commit_pair(
            &policies,
            decider,
            MODEL,
            &pairs_of(&registry, &policies),
            PairProposal::Decided(pd),
            &|_| true,
            &mut missed,
        )
        .err()
        .expect("the prefill went unavailable under the lock");
        assert_eq!(refused.role, WorkerRole::Prefill);
        assert!(matches!(
            refused.verdict,
            PlacementFailure::PrefillAtCapacity
        ));
        assert_eq!(missed.len(), 2, "the prefill and its decode: {missed:?}");
        let pair = decided_pair(&registry, &policies, &|_| true, &|_| true, &mut missed)
            .expect("the re-decision pairs the other prefill");
        assert_ne!(pair.prefill.url(), chosen.url());
        assert_eq!(pair.decode.url(), "grpc://d:1");
        assert_eq!(decider.decisions(), 2, "one re-decision");
    }

    /// A round-robin prefill beside a least-load decode, no token count: the
    /// decode side is booked at its own policy's mean, so the booked decode
    /// carries the credit and every next decision takes the other decode
    /// (booked at the prefill's estimate of nothing, the two would tie).
    #[test]
    fn a_mixed_policy_pair_books_the_decode_at_the_decode_policies_estimate() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let round_robin = PolicyConfig::RoundRobin;
        let least_load = PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        };
        let policies = PolicyRegistry::new(round_robin.clone()).with_policy_engine(
            PolicyEngineMode::Engine,
            &round_robin,
            &least_load,
        );
        let decider = policies.decider().expect("hosted");
        let feed = TestFeed::attached_to(policies.get_decode_policy().as_ref());
        let idle = || one_rank_load(0, 0.1, 2000.0);
        feed.publish(&HashMap::from([
            ("grpc://p:1".to_string(), idle()),
            ("grpc://d:1".to_string(), idle()),
            ("grpc://d:2".to_string(), idle()),
        ]));
        decider.set_load_receiver(Some(feed.receiver()));
        assert_eq!(
            decider.prefill_prompt_tokens(None),
            0,
            "round robin prices no tokens"
        );
        assert_eq!(decider.decode_prompt_tokens(None), 100);
        let mut missed = Vec::new();
        let booked = decided_pair(&registry, &policies, &|_| true, &|_| true, &mut missed)
            .expect("a pair is open")
            .decode;
        for _ in 0..20 {
            let next = proposed_decode(&proposed_pair(&registry, &policies));
            assert_ne!(
                next.url(),
                booked.url(),
                "the booked decode carries its own credit"
            );
        }
    }

    /// Both decode partners over the thresholds with shedding off, one of
    /// them busier: under the switch every placement takes the least-loaded
    /// partner, counted as the veto fallback and not as a selection of the
    /// decode policy, as the legacy path pairs such a pool.
    #[test]
    fn with_the_engine_switch_an_all_overloaded_decode_pool_pairs_the_least_loaded_partner() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
            ("grpc://d:2", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let busier = registry.get_by_url("grpc://d:1").expect("registered");
        for _ in 0..3 {
            busier.increment_load();
        }
        for url in ["grpc://d:1", "grpc://d:2"] {
            let worker = registry.get_by_url(url).expect("registered");
            registry.set_worker_overloaded(&worker, true);
        }
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let mut missed = Vec::new();
        let pair = metrics::with_local_recorder(&recorder, || {
            decided_pair(&registry, &policies, &|_| true, &|_| true, &mut missed)
        })
        .expect("a pair is open");
        assert_eq!(pair.decode.url(), "grpc://d:2", "the least-loaded partner");
        let rendered = handle.render();
        assert_one_overload_fallback(&rendered);
        assert!(
            !rendered.lines().any(|line| {
                line.starts_with("smg_worker_selection_total{")
                    && line.contains("worker_type=\"decode\"")
            }),
            "the fallback's decode is not a selection of the decode policy:\n{rendered}"
        );
        let picks = decided_hits(&registry, &policies, 4);
        assert_eq!(
            picks.get("grpc://d:1").copied().unwrap_or(0),
            0,
            "{picks:?}"
        );
        assert_eq!(picks["grpc://d:2"], 4, "{picks:?}");
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
        let mut candidates: Vec<Arc<dyn Worker>> =
            ["grpc://d:1", "grpc://d:2", "grpc://d:3", "grpc://d:4"]
                .map(decode)
                .into();
        assert!(candidates[0].try_admit_pd(1, 1) && candidates[2].try_admit_pd(1, 1));
        retain_decodes_with_a_room(&policy, None, &mut candidates);
        let urls: Vec<&str> = candidates.iter().map(|d| d.url()).collect();
        assert_eq!(urls, ["grpc://d:2", "grpc://d:4"]);

        // Every candidate full, then every one with a room: the list as it was.
        assert!(candidates[0].try_admit_pd(1, 1) && candidates[1].try_admit_pd(1, 1));
        retain_decodes_with_a_room(&policy, None, &mut candidates);
        assert_eq!(candidates.len(), 2, "nothing to keep: the policy's list");
        candidates[0].release_pd(1);
        candidates[1].release_pd(1);
        retain_decodes_with_a_room(&policy, None, &mut candidates);
        let urls: Vec<&str> = candidates.iter().map(|d| d.url()).collect();
        assert_eq!(urls, ["grpc://d:2", "grpc://d:4"], "nothing to cut");
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
                true,
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
            let feed = TestFeed::attached_to(decode_policy.as_ref());
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
                    true,
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

    /// The engine serves cache_aware as the legacy path does: the holder of
    /// a prompt's prefix keeps it under both switch values, a retry leaves
    /// the holder, and under the engine every placement is its decision.
    #[test]
    fn a_cache_aware_prefix_keeps_its_holder_under_both_switch_values() {
        for mode in [PolicyEngineMode::Legacy, PolicyEngineMode::Engine] {
            let registry = http_registry(&["http://h:1", "http://h:2"]);
            let policies = PolicyRegistry::new(cache_aware_policy()).with_policy_engine(
                mode,
                &cache_aware_policy(),
                &cache_aware_policy(),
            );
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
                    "{mode:?}: the prefix pins the prompt to its holder"
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
            assert_ne!(
                moved.url(),
                holder.url(),
                "{mode:?}: the retry leaves the holder"
            );
            let decided = policies.decider().map_or(0, |decider| decider.decisions());
            assert_eq!(
                decided,
                if mode == PolicyEngineMode::Engine {
                    10
                } else {
                    0
                }
            );
        }
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
    /// index 2 names nobody. Back in, the indexes read as before.
    #[test]
    fn an_explicit_target_indexes_the_available_workers() {
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

    /// Round robin under `--policy-engine engine`, the shape every test of the
    /// switch starts from.
    fn engine_round_robin() -> PolicyRegistry {
        let round_robin = PolicyConfig::RoundRobin;
        let policies = PolicyRegistry::new(round_robin.clone()).with_policy_engine(
            PolicyEngineMode::Engine,
            &round_robin,
            &round_robin,
        );
        assert!(
            policies.decider().is_some(),
            "the engine serves round robin"
        );
        policies
    }

    /// Under `--policy-engine engine` a preset the engine serves decides over
    /// the engine's view and commits under the host's eligibility: round robin
    /// rotates the pool it is given, and a worker that is not ready is never
    /// placed. Under `legacy`, the default, the engine is not built.
    #[test]
    fn with_the_engine_switch_select_from_rotates_the_pool_it_is_given() {
        let registry = registry_with(&[
            ("grpc://r:1", ConnectionMode::Grpc, RuntimeType::Vllm),
            ("grpc://r:2", ConnectionMode::Grpc, RuntimeType::Vllm),
            ("grpc://r:3", ConnectionMode::Grpc, RuntimeType::Vllm),
        ]);
        let round_robin = PolicyConfig::RoundRobin;
        let legacy = PolicyRegistry::new(round_robin.clone()).with_policy_engine(
            PolicyEngineMode::Legacy,
            &round_robin,
            &round_robin,
        );
        assert!(
            legacy.decider().is_none(),
            "the default leaves the engine out"
        );
        let policies = engine_round_robin();
        let candidates: Vec<Arc<dyn Worker>> = ["grpc://r:1", "grpc://r:2", "grpc://r:3"]
            .iter()
            .map(|url| registry.get_by_url(url).expect("registered"))
            .collect();
        let picks = |policies: &PolicyRegistry| {
            let mut picks = BTreeMap::new();
            for _ in 0..9 {
                let worker = select_from(
                    &registry,
                    policies,
                    MODEL,
                    &candidates,
                    PlacementInputs::default(),
                )
                .expect("a worker is picked");
                *picks.entry(worker.url().to_string()).or_insert(0) += 1;
            }
            picks
        };
        let counts = |picks: BTreeMap<String, usize>| picks.into_values().collect::<Vec<_>>();
        assert_eq!(counts(picks(&policies)), [3, 3, 3]);
        assert_eq!(
            policies.decider().expect("the engine").decisions(),
            9,
            "every placement was decided through the engine"
        );
        assert_eq!(counts(picks(&legacy)), [3, 3, 3]);

        candidates[1].set_status(WorkerStatus::NotReady);
        let picks = picks(&policies);
        assert_eq!(picks.get("grpc://r:2"), None, "{picks:?}");
        assert!(
            picks["grpc://r:1"] > 0 && picks["grpc://r:3"] > 0,
            "{picks:?}"
        );
    }

    /// Under the switch the engines a retry already tried step aside while
    /// another candidate remains, and a lone tried worker takes the retry, as
    /// the legacy narrowing does; the HTTP entry decides through the engine
    /// too.
    #[test]
    fn with_the_engine_switch_a_tried_worker_steps_aside_while_another_remains() {
        let policies = engine_round_robin();
        let registry = http_registry(&["http://h:1", "http://h:2"]);
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

    /// The HTTP entry draws the request's sequence before it reads the pool:
    /// a worker retired after that read and before the decision (here from
    /// the per-request candidate filter, which runs after the read) is booked
    /// by no commit of the request, so the rows its retirement pruned stay
    /// gone.
    #[test]
    fn with_the_engine_switch_select_single_draws_the_sequence_before_its_capture() {
        use std::{
            collections::HashMap,
            sync::atomic::{AtomicBool, Ordering},
        };

        let least_load = PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.0,
            mean_prefill_tokens: 100,
            default_throughput: 1000.0,
        };
        let policies = PolicyRegistry::new(least_load.clone()).with_policy_engine(
            PolicyEngineMode::Engine,
            &least_load,
            &least_load,
        );
        let decider = policies.decider().expect("the engine serves least load");
        let registry = http_registry(&["http://h:1", "http://h:2"]);
        let workers: Vec<Arc<dyn Worker>> = ["http://h:1", "http://h:2"]
            .iter()
            .map(|url| registry.get_by_url(url).expect("registered"))
            .collect();
        // Two workers under the same idle report, so only the engine's own
        // credit tells them apart.
        let feed = TestFeed::attached_to(policies.get_policy_or_default(MODEL).as_ref());
        decider.set_load_receiver(Some(feed.receiver()));
        feed.publish(&HashMap::from([
            ("http://h:1".to_string(), one_rank_load(0, 0.1, 2000.0)),
            ("http://h:2".to_string(), one_rank_load(0, 0.1, 2000.0)),
        ]));
        let leader = || {
            let request = HostRequest {
                model: MODEL,
                ..HostRequest::default()
            };
            let (decision, _) = decider.decide(&request, &workers, &[]);
            decision
                .pick()
                .expect("a pick")
                .targets
                .first()
                .map(|t| t.position)
        };
        // One small booking on the second worker first, so the first one
        // leads strictly and the request below is placed on it.
        let small = HostRequest {
            model: MODEL,
            prompt_tokens: 1,
            ..HostRequest::default()
        };
        let (decision, lease) = decider.decide(&small, &workers, &[]);
        decider
            .commit(&lease, &decision, Side::Single, |t| t.position == 1)
            .expect("the second worker is booked");
        assert_eq!(leader(), Some(0), "the first worker leads while idle");
        // The filter runs after the pool read: the first worker's retirement
        // lands there, once, and the captured pool still holds its handle.
        let retired = AtomicBool::new(false);
        let retire_the_first = |_: &dyn Worker| {
            if !retired.swap(true, Ordering::Relaxed) {
                policies.on_worker_removed(workers[0].as_ref());
            }
            true
        };
        let selected = pick(
            &registry,
            &policies,
            PlacementInputs {
                candidate_filter: Some(&retire_the_first),
                ..Default::default()
            },
        );
        assert!(
            retired.load(Ordering::Relaxed),
            "the retirement landed after the capture"
        );
        assert_eq!(
            selected.url(),
            "http://h:1",
            "the request is placed on the leading worker, retired or not"
        );
        // The retired handle still leads: the request's mean-size booking
        // would have lost it to the one-token booking on the other worker.
        assert_eq!(
            leader(),
            Some(0),
            "the retired handle reads idle: the request's commit booked nothing on it"
        );
    }

    /// Under the switch a keyed request is a session of the engine's default
    /// binding, pinned to its owner by the sticky routing-key override's
    /// rule, while an unkeyed one rotates through the engine.
    #[test]
    fn with_the_engine_switch_the_sticky_override_keeps_its_owner() {
        let round_robin = PolicyConfig::RoundRobin;
        let policies = PolicyRegistry::with_override(
            round_robin.clone(),
            RoutingKeyOverrideConfig {
                enabled: true,
                ..Default::default()
            },
        )
        .with_policy_engine(PolicyEngineMode::Engine, &round_robin, &round_robin);
        assert!(policies.decider().is_some());
        let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
        let headers = keyed_headers();
        let owner = pick(&registry, &policies, with_headers(&headers));
        // A vacant sticky key would go to a lighter worker.
        owner.increment_load();
        for _ in 0..4 {
            assert_eq!(
                pick(&registry, &policies, with_headers(&headers)).url(),
                owner.url(),
                "the key left its owner"
            );
        }
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..3 {
            seen.insert(
                pick(&registry, &policies, PlacementInputs::default())
                    .url()
                    .to_string(),
            );
        }
        assert_eq!(seen.len(), 3, "an unkeyed request rotates: {seen:?}");
        assert_eq!(
            policies.decider().expect("the engine").decisions(),
            8,
            "the keyed placements were decided through the engine too"
        );
    }

    /// Under the switch the manual policy and the sticky override decide
    /// through the engine with the legacy outcomes, under the same inputs: a
    /// key pins its first worker, every turn of a conversation shares that
    /// owner under the override, and a new turn is a new key without it.
    #[test]
    fn with_the_engine_switch_the_manual_policy_and_the_override_pin_like_legacy() {
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
        let modes = [PolicyEngineMode::Legacy, PolicyEngineMode::Engine];
        for (config, mode) in [PolicyConfig::RoundRobin, manual()]
            .iter()
            .flat_map(|config| modes.iter().map(move |mode| (config, *mode)))
        {
            let policies = PolicyRegistry::with_override(
                config.clone(),
                RoutingKeyOverrideConfig {
                    enabled: true,
                    ..Default::default()
                },
            )
            .with_policy_engine(mode, config, config);
            assert_eq!(
                policies.decider().is_some(),
                mode == PolicyEngineMode::Engine
            );
            let name = policies.get_default_policy().name();
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
                    "{name} under {mode:?}: {turn} left the owner of conv"
                );
            }
            if let Some(decider) = policies.decider() {
                assert_eq!(
                    decider.decisions(),
                    1 + turns.len() as u64,
                    "{name}: every keyed placement was decided through the engine"
                );
            }
        }

        // Override off: the manual policy pins each raw key on its own.
        for mode in modes {
            let policies =
                PolicyRegistry::new(manual()).with_policy_engine(mode, &manual(), &manual());
            let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
            let first = keyed("conv_t1");
            let owner = pick(&registry, &policies, with_headers(&first));
            owner.increment_load();
            let second = keyed("conv_t2");
            assert_ne!(
                pick(&registry, &policies, with_headers(&second)).url(),
                owner.url(),
                "{mode:?}: without the override a new turn is a new key"
            );
        }
    }

    /// Under the switch the manual policy reads the `x-smg-routing-key`
    /// hint the legacy manual policy reads second, under an override whose
    /// configured list names other headers: a conversation keyed by the
    /// hint alone keeps its owner across turns with fresh request ids, and
    /// across a turn with none, under both switch values.
    #[test]
    fn with_the_engine_switch_the_manual_policy_reads_the_hint_under_a_configured_list() {
        let manual = || PolicyConfig::Manual {
            eviction_interval_secs: 0,
            max_idle_secs: 0,
            assignment_mode: ManualAssignmentMode::MinLoad,
        };
        let hinted = keyed_headers();
        for mode in [PolicyEngineMode::Legacy, PolicyEngineMode::Engine] {
            let policies = PolicyRegistry::with_override(
                manual(),
                RoutingKeyOverrideConfig {
                    enabled: true,
                    headers: vec!["x-session-id".into()],
                    ..Default::default()
                },
            )
            .with_policy_engine(mode, &manual(), &manual());
            assert_eq!(
                policies.decider().is_some(),
                mode == PolicyEngineMode::Engine
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
                    "{mode:?}: the turn keyed {rid_key:?} left the owner of the hint"
                );
            }
        }
    }
    /// Under the switch the hint keys nothing for a default the override
    /// places: with the override's configured list naming other headers, a
    /// round-robin request carrying the hint alone rotates under both switch
    /// values, as the legacy override has no hint fallback.
    #[test]
    fn with_the_engine_switch_the_hint_keys_nothing_under_the_override() {
        let round_robin = PolicyConfig::RoundRobin;
        let hinted = keyed_headers();
        for mode in [PolicyEngineMode::Legacy, PolicyEngineMode::Engine] {
            let policies = PolicyRegistry::with_override(
                round_robin.clone(),
                RoutingKeyOverrideConfig {
                    enabled: true,
                    headers: vec!["x-session-id".into()],
                    ..Default::default()
                },
            )
            .with_policy_engine(mode, &round_robin, &round_robin);
            let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
            let mut seen = std::collections::BTreeSet::new();
            for _ in 0..3 {
                seen.insert(
                    pick(&registry, &policies, with_headers(&hinted))
                        .url()
                        .to_string(),
                );
            }
            assert_eq!(seen.len(), 3, "{mode:?}: the hint alone rotates: {seen:?}");
        }
    }
    /// Under the switch a key whose owner is not ready takes a failover and
    /// returns to its first owner once that worker is back under the manual
    /// policy, as under legacy; under the override it stays on the failover,
    /// as the legacy override pins the newest worker first.
    #[test]
    fn with_the_engine_switch_a_key_recovers_as_the_legacy_paths_do() {
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
            for mode in [PolicyEngineMode::Legacy, PolicyEngineMode::Engine] {
                let policies = PolicyRegistry::with_override(
                    config.clone(),
                    RoutingKeyOverrideConfig {
                        enabled: override_on,
                        ..Default::default()
                    },
                )
                .with_policy_engine(mode, &config, &config);
                let registry = http_registry(&["http://h:1", "http://h:2", "http://h:3"]);
                let owner = pick(&registry, &policies, with_headers(&headers));
                owner.set_status(WorkerStatus::NotReady);
                let failover = pick(&registry, &policies, with_headers(&headers));
                assert_ne!(failover.url(), owner.url(), "{mode:?}");
                for _ in 0..3 {
                    assert_eq!(
                        pick(&registry, &policies, with_headers(&headers)).url(),
                        failover.url(),
                        "{mode:?}: the failover holds the key"
                    );
                }
                owner.set_status(WorkerStatus::Ready);
                let back = pick(&registry, &policies, with_headers(&headers));
                assert_eq!(
                    back.url() == owner.url(),
                    returns,
                    "{} under {mode:?}: the key went to {}",
                    config.name(),
                    back.url()
                );
            }
        }
    }

    /// Under the switch a pool whose every worker is over the overload
    /// thresholds is steered to its least-loaded worker by the host's fallback
    /// and counted as such, as the legacy path does once its pre-filter
    /// empties; a placement that commits keeps the configured policy's name
    /// as the selection metric's label.
    #[test]
    fn with_the_engine_switch_an_all_overloaded_pool_keeps_the_least_loaded_fallback() {
        let policies = engine_round_robin();
        let registry = http_registry(&["http://h:1", "http://h:2"]);
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            pick(&registry, &policies, PlacementInputs::default())
        });
        assert!(
            handle.render().lines().any(|line| {
                line.starts_with("smg_worker_selection_total{")
                    && line.contains("policy=\"round_robin\"")
                    && line.ends_with(" 1")
            }),
            "the selection keeps the policy label:\n{}",
            handle.render()
        );

        let busier = registry.get_by_url("http://h:1").expect("registered");
        for _ in 0..10 {
            busier.increment_load();
        }
        for worker in registry.get_all() {
            registry.set_worker_overloaded(&worker, true);
        }
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let selected = metrics::with_local_recorder(&recorder, || {
            pick(&registry, &policies, PlacementInputs::default())
        });
        assert_eq!(
            selected.url(),
            "http://h:2",
            "the least-loaded worker of the vetoed pool"
        );
        assert_one_overload_fallback(&handle.render());
    }
    /// The engine's view already leaves out a worker the host vetoes, so a
    /// commit miss needs the eligibility to turn between the view and the
    /// commit. Handed one that refuses targets the view kept, the seam takes
    /// the next target of the same decision; when every target misses, one
    /// re-decision runs with the misses excluded; when that misses too, the
    /// host's own fallback answers, nothing here since the host vetoed no
    /// one, and no third decision runs.
    #[test]
    fn with_the_engine_switch_a_commit_miss_walks_the_targets_then_re_decides_once() {
        let passthrough = PolicyConfig::Passthrough;
        let policies = PolicyRegistry::new(passthrough.clone()).with_policy_engine(
            PolicyEngineMode::Engine,
            &passthrough,
            &passthrough,
        );
        let decider = policies.decider().expect("the engine serves passthrough");
        let urls = [
            "grpc://p:1",
            "grpc://p:2",
            "grpc://p:3",
            "grpc://p:4",
            "grpc://p:5",
        ];
        let registry = registry_with(
            &urls
                .iter()
                .map(|url| (*url, ConnectionMode::Grpc, RuntimeType::Vllm))
                .collect::<Vec<_>>(),
        );
        let candidates: Vec<Arc<dyn Worker>> = urls
            .iter()
            .map(|url| registry.get_by_url(url).expect("registered"))
            .collect();
        let place = |refused: &[&str]| {
            let eligible = |worker: &Arc<dyn Worker>| !refused.contains(&worker.url());
            select_from_decided(RegularDecision {
                decider,
                registry: &registry,
                model_id: MODEL,
                candidates: &candidates,
                request: HostRequest {
                    model: MODEL,
                    phase: Phase::Single,
                    ..HostRequest::default()
                },
                eligible: &eligible,
                policy: "passthrough",
            })
        };

        // Passthrough lists the pool in order: the first target misses, the
        // second one of the same decision is taken.
        assert_eq!(
            place(&["grpc://p:1"]).expect("a worker").url(),
            "grpc://p:2"
        );
        assert_eq!(decider.decisions(), 1, "the walk needs no second decision");

        // Four misses exhaust the decision's targets: one re-decision with the
        // misses excluded reaches the fifth worker.
        let four = ["grpc://p:1", "grpc://p:2", "grpc://p:3", "grpc://p:4"];
        assert_eq!(place(&four).expect("a worker").url(), "grpc://p:5");
        assert_eq!(decider.decisions(), 3, "one decision and one re-decision");

        // Every target refused: the re-decision misses too, the host's
        // fallback has nothing vetoed to steer to, and no third decision runs.
        assert!(
            place(&urls).is_none(),
            "no fallback outside the host's veto"
        );
        assert_eq!(decider.decisions(), 5, "one re-decision, never a second");
    }

    /// The commit re-reads the decode partner's liveness as it re-reads the
    /// prefill's window. With one prefill the window's third read is the
    /// first commit's; the decided decode goes down at that read, the
    /// commit misses the pair, and the re-decision finds no open partner
    /// left: the seam refuses with the decode role's unavailability, as
    /// `pair_shed` maps a decode side with nothing open. A commit that only
    /// checks the partner exists places the request on the dead decode.
    #[test]
    fn the_commit_refuses_a_decode_partner_that_died_after_the_decision() {
        let registry = pd_registry(&[
            ("grpc://p:1", WorkerType::Prefill, Some("NixlConnector")),
            ("grpc://d:1", WorkerType::Decode, Some("NixlConnector")),
        ]);
        let policies = engine_cohort_policies();
        let decider = policies.decider().expect("hosted");
        let pairs = pairs_of(&registry, &policies);
        let decode = registry
            .get_by_url("grpc://d:1")
            .expect("the decode registers");
        let reads = Cell::new(0_u32);
        let has_capacity = |_: &Arc<dyn Worker>| {
            reads.set(reads.get() + 1);
            if reads.get() == 3 {
                decode.set_status(WorkerStatus::NotReady);
            }
            true
        };
        let refused = place_pair_through_engine(
            &registry,
            &policies,
            decider,
            MODEL,
            &pairs,
            None,
            true,
            &has_capacity,
            PlacementInputs::default(),
        )
        .err()
        .expect("no live decode is left for the pair");
        assert!(reads.get() >= 3, "the commit read the window");
        assert_eq!(refused.role, WorkerRole::Decode);
        assert!(matches!(refused.verdict, PlacementFailure::Unavailable));
    }
}
