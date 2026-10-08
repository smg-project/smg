//! The answers to a candidate pool whose every worker is vetoed, by the
//! overload thresholds or by the liveness tracker: by default the request
//! steers to the least-loaded of them ([`fallback_if_all_vetoed`]); under
//! `--worker-overload-shed` an all-overloaded pool is refused with a distinct
//! 503 ([`shed_if_all_overloaded`]) while a liveness veto still steers.
//! Nothing here runs while some worker is free of both vetoes: such workers
//! are simply left out of selection.
//!
//! The verdict is taken from the candidate pool the caller selected over, never
//! from the model index: selection narrows by worker type and transport first,
//! and a whole-model predicate would miss a saturated PD leg, a mixed-transport
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
use tracing::debug;

use crate::{
    observability::metrics::Metrics,
    routers::{common::retry::mark_non_retryable, error},
    worker::{
        overload::{
            BRANCH_ALL_OVERLOADED_FALLBACK, BRANCH_ALL_OVERLOADED_SHED,
            BRANCH_ALL_STALLED_FALLBACK, BRANCH_OVERLOADED_AT_DISPATCH, BRANCH_PD_ADMISSION_SHED,
            STAGE_DISPATCH, STAGE_PD_ADMISSION, STAGE_SELECTION,
        },
        Worker,
    },
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

/// Whether every worker in a non-empty candidate pool is vetoed.
///
/// `candidates` is the pool *before* the `is_available()` filter, narrowed by
/// exactly the worker-type / connection-mode filter selection used.
pub fn all_overloaded(candidates: &[Arc<dyn Worker>]) -> bool {
    !candidates.is_empty() && candidates.iter().all(|w| w.is_overloaded())
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

/// Whether every worker in a non-empty candidate pool is vetoed, by the
/// overload flag or by the liveness tracker (`Worker::stall_reason`).
///
/// `candidates` is the pool *before* the `is_available()` filter, narrowed by
/// exactly the worker-type / connection-mode filter selection used.
pub fn all_vetoed(candidates: &[Arc<dyn Worker>]) -> bool {
    !candidates.is_empty()
        && candidates
            .iter()
            .all(|w| w.is_overloaded() || w.stall_reason().is_some())
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
    let ready = |w: &Arc<dyn Worker>| w.is_healthy() && w.circuit_breaker_can_execute();
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
        worker = worker.url(),
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
    let url = worker.url();
    Some(shed(
        BRANCH_OVERLOADED_AT_DISPATCH,
        STAGE_DISPATCH,
        url,
        format!("Worker '{url}' for model '{model_id}' became overloaded before dispatch"),
    ))
}

/// Shed a disaggregated dispatch the decode leg cannot admit: the `rooms`
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
mod tests {
    use axum::http::StatusCode;
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

    use super::*;
    use crate::{
        routers::{
            common::retry::{is_retryable_response, is_retryable_status},
            error::extract_error_code_from_response,
        },
        worker::{BasicWorkerBuilder, ConnectionMode, WorkerType},
    };

    fn worker(url: &str, model_id: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model_id))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    /// The shed is a distinct 503, taken from the candidate pool rather than
    /// the model index.
    #[test]
    fn all_overloaded_sheds_with_distinct_503_code() {
        let a = worker("http://127.0.0.1:9801", "m");
        let b = worker("http://127.0.0.1:9802", "m");
        let pool = vec![Arc::clone(&a), Arc::clone(&b)];

        a.set_overloaded(true);
        assert!(
            shed_if_all_overloaded(&pool, "m", true).is_none(),
            "one eligible worker left is not a shed"
        );

        b.set_overloaded(true);
        assert!(
            shed_if_all_overloaded(&pool, "m", false).is_none(),
            "without shedding an all-overloaded pool is steered, not refused"
        );
        let response = shed_if_all_overloaded(&pool, "m", true).expect("shed");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            extract_error_code_from_response(&response),
            "worker_overload_protection_shed"
        );
        assert_eq!(
            response
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .expect("gateway error code header"),
            "worker_overload_protection_shed"
        );
    }

    /// A shed must be terminal for the retry layer, or "shed immediately"
    /// becomes six pipeline runs and ~400 ms of backoff per request.
    #[test]
    fn shed_responses_are_not_retried() {
        let a = worker("http://127.0.0.1:9811", "m");
        a.set_overloaded(true);

        let selection = shed_if_all_overloaded(std::slice::from_ref(&a), "m", true).expect("shed");
        assert!(
            is_retryable_status(selection.status()),
            "the status stays the retryable 503 clients already understand"
        );
        assert!(
            !is_retryable_response(&selection),
            "but the retry layer must decline it"
        );

        let dispatch = shed_if_worker_overloaded(a.as_ref(), "m", true).expect("shed");
        assert!(!is_retryable_response(&dispatch));
    }

    /// Both shed kinds advertise the poll interval as Retry-After — the veto
    /// cannot clear faster — while staying terminal for the retry layer.
    #[test]
    fn shed_responses_carry_retry_after() {
        let a = worker("http://127.0.0.1:9821", "m");
        a.set_overloaded(true);

        let selection = shed_if_all_overloaded(std::slice::from_ref(&a), "m", true).expect("shed");
        let dispatch = shed_if_worker_overloaded(a.as_ref(), "m", true).expect("shed");
        for response in [&selection, &dispatch] {
            let value = response
                .headers()
                .get(RETRY_AFTER)
                .expect("Retry-After present")
                .to_str()
                .expect("ascii");
            assert!(
                value.parse::<u64>().is_ok_and(|secs| secs >= 1),
                "Retry-After must be whole seconds >= 1, got {value}"
            );
            assert!(!is_retryable_response(response));
        }
    }

    /// An empty pool is a 404/unavailable question for the caller, not a shed
    /// and not a fallback.
    #[test]
    fn empty_pool_is_not_a_shed() {
        assert!(shed_if_all_overloaded(&[], "nobody", true).is_none());
        assert!(fallback_if_all_vetoed(&[], "nobody", STAGE_SELECTION, false).is_none());
        assert!(!all_overloaded(&[]));
    }

    /// The steering default: an all-overloaded pool routes to its least-loaded
    /// routable worker; a pool with an eligible worker left is not touched,
    /// and a worker that is also unhealthy is never the fallback.
    #[test]
    fn equally_loaded_workers_share_the_fallback_and_a_lighter_one_wins() {
        use rand::{rngs::StdRng, SeedableRng};
        let pool: Vec<Arc<dyn Worker>> = (0..128)
            .map(|i| worker(&format!("http://127.0.0.1:{}", 20000 + i), "m"))
            .collect();
        let mut rng = StdRng::seed_from_u64(7);
        let mut hits = vec![0usize; pool.len()];
        for _ in 0..1000 {
            let picked = least_loaded_uniform(pool.iter(), &mut rng).expect("a worker");
            hits[pool.iter().position(|w| Arc::ptr_eq(w, picked)).unwrap()] += 1;
        }
        assert!(hits.iter().all(|&h| h >= 1), "{hits:?}");
        assert!(*hits.iter().max().unwrap() <= 24, "{hits:?}");
        for (i, w) in pool.iter().enumerate() {
            if i != 5 {
                w.increment_load();
            }
        }
        for _ in 0..100 {
            let picked = least_loaded_uniform(pool.iter(), &mut rng).expect("a worker");
            assert!(Arc::ptr_eq(picked, &pool[5]));
        }
    }

    #[test]
    fn all_overloaded_falls_back_to_the_least_loaded_routable_worker() {
        let a = worker("http://127.0.0.1:9831", "m");
        let b = worker("http://127.0.0.1:9832", "m");
        let c = worker("http://127.0.0.1:9833", "m");
        let pool = vec![Arc::clone(&a), Arc::clone(&b), Arc::clone(&c)];
        for _ in 0..3 {
            a.increment_load();
        }
        b.increment_load();

        a.set_overloaded(true);
        b.set_overloaded(true);
        assert!(
            fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).is_none(),
            "an eligible worker left means selection handles it"
        );

        c.set_overloaded(true);
        let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).expect("fallback");
        assert_eq!(
            picked.url(),
            "http://127.0.0.1:9833",
            "the least-loaded wins"
        );

        c.set_status(openai_protocol::worker::WorkerStatus::NotReady);
        let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).expect("fallback");
        assert_eq!(
            picked.url(),
            "http://127.0.0.1:9832",
            "an unhealthy worker is not routable even as the fallback"
        );

        a.set_status(openai_protocol::worker::WorkerStatus::NotReady);
        b.set_status(openai_protocol::worker::WorkerStatus::NotReady);
        assert!(
            fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).is_none(),
            "nothing routable: the caller's unavailable answer stands"
        );
    }

    /// A liveness veto steers but never refuses: when every ready worker is
    /// vetoed by the tracker the request goes to the least-loaded of them,
    /// after any worker vetoed by overload alone, and under shedding never
    /// to an overloaded one.
    #[test]
    fn all_stalled_falls_back_to_the_least_loaded_ready_worker() {
        use crate::worker::worker::StallReason;

        let a = worker("http://127.0.0.1:9841", "m");
        let b = worker("http://127.0.0.1:9842", "m");
        let pool = vec![Arc::clone(&a), Arc::clone(&b)];
        for _ in 0..2 {
            a.increment_load();
        }

        assert!(a.set_stall(Some(StallReason::Wedged)));
        assert!(
            fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).is_none(),
            "a worker free of both vetoes means selection handles it"
        );

        assert!(b.set_stall(Some(StallReason::Unreachable)));
        let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).expect("fallback");
        assert_eq!(
            picked.url(),
            "http://127.0.0.1:9842",
            "the least-loaded of the vetoed is taken"
        );
        let picked =
            fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, true).expect("under shedding");
        assert_eq!(
            picked.url(),
            "http://127.0.0.1:9842",
            "a liveness veto steers under shedding too"
        );

        assert!(b.set_stall(None));
        b.set_overloaded(true);
        let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).expect("fallback");
        assert_eq!(
            picked.url(),
            "http://127.0.0.1:9842",
            "overload alone outranks a liveness veto"
        );
        let picked =
            fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, true).expect("under shedding");
        assert_eq!(
            picked.url(),
            "http://127.0.0.1:9841",
            "under shedding the overloaded worker is never taken"
        );

        a.set_overloaded(true);
        assert!(
            fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, true).is_none(),
            "all overloaded under shedding: the caller sheds"
        );

        b.set_status(openai_protocol::worker::WorkerStatus::NotReady);
        a.set_overloaded(false);
        let picked = fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).expect("fallback");
        assert_eq!(
            picked.url(),
            "http://127.0.0.1:9841",
            "an unhealthy worker is never the fallback; the stalled ready one is"
        );

        a.set_status(openai_protocol::worker::WorkerStatus::NotReady);
        assert!(
            fallback_if_all_vetoed(&pool, "m", STAGE_SELECTION, false).is_none(),
            "nothing ready: the caller's unavailable answer stands"
        );
    }

    #[test]
    fn dispatch_recheck_sheds_only_for_a_flagged_worker() {
        let w = worker("http://127.0.0.1:9803", "m");
        assert!(shed_if_worker_overloaded(w.as_ref(), "m", true).is_none());

        w.set_overloaded(true);
        let response = shed_if_worker_overloaded(w.as_ref(), "m", true).expect("shed");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            extract_error_code_from_response(&response),
            "worker_overload_protection_shed"
        );
        assert_eq!(
            response
                .headers()
                .get(error::HEADER_X_SMG_ERROR_CODE)
                .expect("gateway error code header"),
            "worker_overload_protection_shed"
        );
    }
}
