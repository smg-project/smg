//! The answers to a candidate pool whose every routable worker is vetoed, by the
//! overload thresholds or by the liveness tracker: by default the request
//! steers to the least-loaded of them ([`fallback_if_all_vetoed`]); under
//! `--worker-overload-shed` an all-overloaded pool is refused with a distinct
//! 503 ([`shed_if_all_overloaded`]) while a liveness veto still steers.
//! Nothing here runs while some routable worker is free of both vetoes: such workers
//! are simply left out of selection.
//!
//! The verdict is taken from the candidate pool the caller selected over, never
//! from the model index: selection narrows by worker type and transport first,
//! and a whole-model predicate would miss a saturated PD side, a mixed-transport
//! model, or the model-less wildcard.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use axum::{
    http::{header::RETRY_AFTER, HeaderValue},
    response::Response,
};
use rand::RngExt;
use smg_worker::{
    health::overload::{
        BRANCH_ALL_OVERLOADED_FALLBACK, BRANCH_ALL_OVERLOADED_SHED, BRANCH_ALL_STALLED_FALLBACK,
        BRANCH_OVERLOADED_AT_DISPATCH, BRANCH_PD_ADMISSION_SHED, STAGE_DISPATCH,
        STAGE_PD_ADMISSION, STAGE_SELECTION,
    },
    identity::basic::Worker,
};
use tracing::debug;

use crate::{
    observability::metrics::Metrics,
    routers::{common::retry::mark_non_retryable, error},
};

/// Retry-After seconds advertised on every shed: the load-monitor poll
/// interval, since the veto provably cannot clear faster. Process-wide because
/// the shed helpers are free functions called from every router; the value is
/// a client hint, not a correctness input. Default matches the config default.
static SHED_RETRY_AFTER_SECS: AtomicU64 = AtomicU64::new(10);

/// Client-visible, gateway-owned code for a worker-overload-protection shed.
///
/// This deliberately covers both an all-overloaded candidate pool and the
/// selection-to-dispatch re-check, where another worker may still be eligible.
pub(crate) const WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE: &str =
    "worker_overload_protection_shed";

/// Latch the poll interval the shed responses advertise. Called once at
/// startup when the load monitor is built.
pub fn set_shed_retry_after_secs(secs: u64) {
    SHED_RETRY_AFTER_SECS.store(secs.max(1), Ordering::Relaxed);
}

/// Ready and permitted by the circuit breaker, before overload/liveness vetoes.
pub(crate) fn routable_but_for_veto(worker: &dyn Worker) -> bool {
    worker.is_healthy() && worker.circuit_breaker_can_execute()
}

/// Ready, permitted by the circuit breaker, and not stalled. Overload is not
/// checked.
pub(crate) fn routable_but_for_overload(worker: &dyn Worker) -> bool {
    worker.routing_state().eligible_ignoring_overload()
}

/// Whether every otherwise-routable worker in a non-empty pool is overloaded.
///
/// `candidates` is the pool *before* the `is_available()` filter, narrowed by
/// exactly the worker-type / connection-mode filter selection used. Unhealthy
/// or circuit-blocked workers must not suppress the verdict on the ready pool.
pub fn all_overloaded(candidates: &[Arc<dyn Worker>]) -> bool {
    let mut ready = candidates
        .iter()
        .filter(|w| routable_but_for_veto(w.as_ref()))
        .peekable();
    ready.peek().is_some() && ready.all(|w| w.is_overloaded())
}

/// Shed when every worker selection could have used is flagged overloaded
/// and shedding is on (`--worker-overload-shed`, `shedding`).
///
/// `None` means the pool is not all-overloaded, or shedding is off and the
/// caller steers through [`fallback_if_all_vetoed`] instead; either way
/// the caller's existing not-found / unavailable answer stands when nothing
/// is routable.
pub fn shed_if_all_overloaded(
    candidates: &[Arc<dyn Worker>],
    model_id: &str,
    shedding: bool,
) -> Option<Response> {
    if !shedding || !all_overloaded(candidates) {
        return None;
    }
    Some(shed(
        BRANCH_ALL_OVERLOADED_SHED,
        STAGE_SELECTION,
        "none",
        format!("All workers for model '{model_id}' are overloaded"),
    ))
}

/// Whether every otherwise-routable worker in a non-empty pool is vetoed, by
/// the overload flag or by the liveness tracker (`Worker::stall_reason`).
///
/// `candidates` is the pool *before* the `is_available()` filter, narrowed by
/// exactly the worker-type / connection-mode filter selection used. A dead
/// worker without either veto must not prevent fallback to the ready pool.
pub fn all_vetoed(candidates: &[Arc<dyn Worker>]) -> bool {
    let mut ready = candidates
        .iter()
        .filter(|w| routable_but_for_veto(w.as_ref()))
        .peekable();
    ready.peek().is_some() && ready.all(|w| w.is_overloaded() || w.stall_reason().is_some())
}

/// Whether `worker` is kept out of selection by the overload thresholds
/// alone: over them, and otherwise routable (ready, circuit closed, not
/// vetoed by the liveness tracker).
pub fn vetoed_by_overload_alone(worker: &dyn Worker) -> bool {
    let state = worker.routing_state();
    state.overloaded && state.eligible_ignoring_overload()
}

/// The steering answer to a pool whose every worker is vetoed: the
/// least-loaded worker that is routable but for its veto (ready, circuit
/// closed). A fleet that is uniformly over the thresholds, or whose last
/// worker the liveness tracker doubts, is still a fleet; refusing the request
/// would turn a load signal or a suspicion into an outage, and a worker that
/// really is gone fails the request as fast as a refusal would.
///
/// Workers vetoed by overload alone are taken first, so a liveness veto still
/// steers while any other worker can take the request. Under
/// `--worker-overload-shed` (`shedding`) an overloaded worker is never taken:
/// the shed is that flag's answer to overload, and the callers shed an
/// all-overloaded pool before asking here. `None` when the pool is not
/// all-vetoed, or when no worker in it is routable at all (then the caller's
/// unavailable answer stands).
pub fn fallback_if_all_vetoed(
    candidates: &[Arc<dyn Worker>],
    model_id: &str,
    stage: &'static str,
    shedding: bool,
) -> Option<Arc<dyn Worker>> {
    if !all_vetoed(candidates) {
        return None;
    }
    let mut rng = rand::rng();
    let ready = |w: &Arc<dyn Worker>| routable_but_for_veto(w.as_ref());
    let overloaded_first = (!shedding)
        .then(|| {
            least_loaded_uniform(
                candidates
                    .iter()
                    .filter(|w| ready(w) && w.stall_reason().is_none()),
                &mut rng,
            )
        })
        .flatten();
    let (worker, branch) = match overloaded_first {
        Some(worker) => (worker, BRANCH_ALL_OVERLOADED_FALLBACK),
        None => (
            least_loaded_uniform(
                candidates
                    .iter()
                    .filter(|w| ready(w) && !(shedding && w.is_overloaded())),
                &mut rng,
            )?,
            BRANCH_ALL_STALLED_FALLBACK,
        ),
    };
    if branch == BRANCH_ALL_OVERLOADED_FALLBACK {
        Metrics::record_worker_overload_fallback(stage);
    } else {
        Metrics::record_worker_liveness_fallback(stage);
    }
    debug!(
        branch,
        stage,
        worker = worker.redacted_url(),
        model_id,
        "Veto fallback"
    );
    Some(Arc::clone(worker))
}

/// Dispatch-time re-check under `--worker-overload-shed` (`shedding`): one atomic
/// read on the already-chosen worker, covering the selection→dispatch window.
/// Deliberately sheds rather than re-selecting — the flag moves at the poll
/// interval, so the window is rare — and reports only what it knows: this
/// worker went over, not the fleet. Without shedding the dispatch stands: the
/// worker was the right choice when it was made.
pub fn shed_if_worker_overloaded(
    worker: &dyn Worker,
    model_id: &str,
    shedding: bool,
) -> Option<Response> {
    if !shedding || !worker.is_overloaded() {
        return None;
    }
    let url = worker.redacted_url();
    Some(shed(
        BRANCH_OVERLOADED_AT_DISPATCH,
        STAGE_DISPATCH,
        url,
        format!("Worker '{url}' for model '{model_id}' became overloaded before dispatch"),
    ))
}

/// Shed a disaggregated dispatch the decode side cannot admit: the `rooms`
/// bootstrap rooms it needs do not fit in the engine's running `window`, and
/// none freed inside the admission wait.
///
/// Same client-visible answer as the two vetoes above — the request was never
/// sent, and the wait already outlived any backoff a retry would add — under
/// its own decision branch, because here the *pair* is full rather than a
/// threshold being crossed. The counts are in the message because the two
/// causes need different operator responses: a transient full window versus a
/// batched request that is permanently wider than the pair.
pub(crate) fn shed_pd_admission(
    worker: &str,
    model_id: &str,
    window: usize,
    rooms: usize,
) -> Response {
    shed(
        BRANCH_PD_ADMISSION_SHED,
        STAGE_PD_ADMISSION,
        worker,
        format!(
            "Decode worker '{worker}' for model '{model_id}' could not admit {rooms} \
             request(s) within its running window of {window}"
        ),
    )
}

/// One decision line, one counter, one response — marked non-retryable: the
/// veto clears at the poll interval, which no backoff window outlives, and a
/// terminal shed is what keeps the counter per-request rather than per-attempt.
/// Retry-After carries that interval so clients and proxies pace themselves;
/// internal retries stay off regardless.
fn shed(branch: &'static str, stage: &'static str, worker: &str, message: String) -> Response {
    Metrics::record_worker_overload_shed(stage);
    debug!(branch, stage, worker, "Overload shed");
    let mut response =
        error::service_unavailable(WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE, message);
    response.headers_mut().insert(
        RETRY_AFTER,
        HeaderValue::from(SHED_RETRY_AFTER_SECS.load(Ordering::Relaxed)),
    );
    mark_non_retryable(&mut response);
    response
}

/// The least-loaded of `candidates`; one of them drawn uniformly when
/// several share the lowest load, so a fleet whose workers are all equally
/// loaded does not send every fallback to the first of them.
pub(crate) fn least_loaded_uniform<'a, R: RngExt>(
    candidates: impl Iterator<Item = &'a Arc<dyn Worker>>,
    rng: &mut R,
) -> Option<&'a Arc<dyn Worker>> {
    let mut best: Option<(&'a Arc<dyn Worker>, usize)> = None;
    let mut tied = 0u32;
    for worker in candidates {
        let load = worker.load();
        match best {
            Some((_, best_load)) if load > best_load => {}
            Some((_, best_load)) if load == best_load => {
                // The k-th tied worker replaces the pick with probability 1/k.
                tied += 1;
                if rng.random_range(0..=tied) == 0 {
                    best = Some((worker, load));
                }
            }
            _ => {
                best = Some((worker, load));
                tied = 0;
            }
        }
    }
    best.map(|(worker, _)| worker)
}

#[cfg(test)]
#[path = "../../../tests/unit/routers/common/overload/tests.rs"]
mod tests;
