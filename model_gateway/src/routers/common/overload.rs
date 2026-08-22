//! Shed responses for the absolute worker-overload guard. Failure paths only —
//! nothing here runs for a served request.
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
use tracing::debug;

use crate::{
    observability::metrics::Metrics,
    routers::{common::retry::mark_non_retryable, error},
    worker::{
        overload::{
            BRANCH_ALL_OVERLOADED_SHED, BRANCH_OVERLOADED_AT_DISPATCH, STAGE_DISPATCH,
            STAGE_SELECTION,
        },
        Worker,
    },
};

/// Retry-After seconds advertised on every shed: the load-monitor poll
/// interval, since the veto provably cannot clear faster. Process-wide because
/// the shed helpers are free functions called from every router; the value is
/// a client hint, not a correctness input. Default matches the config default.
static SHED_RETRY_AFTER_SECS: AtomicU64 = AtomicU64::new(10);

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

/// Shed when every worker selection could have used is flagged overloaded.
///
/// `None` means the empty pool has some other cause, and the caller's existing
/// not-found / unavailable answer stands.
pub fn shed_if_all_overloaded(candidates: &[Arc<dyn Worker>], model_id: &str) -> Option<Response> {
    if !all_overloaded(candidates) {
        return None;
    }
    Some(shed(
        BRANCH_ALL_OVERLOADED_SHED,
        STAGE_SELECTION,
        "none",
        format!("All workers for model '{model_id}' are overloaded"),
    ))
}

/// Dispatch-time re-check: one atomic read on the already-chosen worker,
/// covering the selection→dispatch window. Deliberately sheds rather than
/// re-selecting — the flag moves at the poll interval, so the window is rare —
/// and reports only what it knows: this worker went over, not the fleet.
pub fn shed_if_worker_overloaded(worker: &dyn Worker, model_id: &str) -> Option<Response> {
    if !worker.is_overloaded() {
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

/// One decision line, one counter, one response — marked non-retryable: the
/// veto clears at the poll interval, which no backoff window outlives, and a
/// terminal shed is what keeps the counter per-request rather than per-attempt.
/// Retry-After carries that interval so clients and proxies pace themselves;
/// internal retries stay off regardless.
fn shed(branch: &'static str, stage: &'static str, worker: &str, message: String) -> Response {
    Metrics::record_worker_overload_shed(stage);
    debug!(branch, stage, worker, "Overload shed");
    let mut response = error::service_unavailable("no_available_workers", message);
    response.headers_mut().insert(
        RETRY_AFTER,
        HeaderValue::from(SHED_RETRY_AFTER_SECS.load(Ordering::Relaxed)),
    );
    mark_non_retryable(&mut response);
    response
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

    /// The shed is the existing `no_available_workers` 503, taken from the
    /// candidate pool rather than the model index.
    #[test]
    fn all_overloaded_sheds_with_no_available_workers_503() {
        let a = worker("http://127.0.0.1:9801", "m");
        let b = worker("http://127.0.0.1:9802", "m");
        let pool = vec![Arc::clone(&a), Arc::clone(&b)];

        a.set_overloaded(true);
        assert!(
            shed_if_all_overloaded(&pool, "m").is_none(),
            "one eligible worker left is not a shed"
        );

        b.set_overloaded(true);
        let response = shed_if_all_overloaded(&pool, "m").expect("shed");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            extract_error_code_from_response(&response),
            "no_available_workers"
        );
    }

    /// A shed must be terminal for the retry layer, or "shed immediately"
    /// becomes six pipeline runs and ~400 ms of backoff per request.
    #[test]
    fn shed_responses_are_not_retried() {
        let a = worker("http://127.0.0.1:9811", "m");
        a.set_overloaded(true);

        let selection = shed_if_all_overloaded(std::slice::from_ref(&a), "m").expect("shed");
        assert!(
            is_retryable_status(selection.status()),
            "the status stays the retryable 503 clients already understand"
        );
        assert!(
            !is_retryable_response(&selection),
            "but the retry layer must decline it"
        );

        let dispatch = shed_if_worker_overloaded(a.as_ref(), "m").expect("shed");
        assert!(!is_retryable_response(&dispatch));
    }

    /// Both shed kinds advertise the poll interval as Retry-After — the veto
    /// cannot clear faster — while staying terminal for the retry layer.
    #[test]
    fn shed_responses_carry_retry_after() {
        let a = worker("http://127.0.0.1:9821", "m");
        a.set_overloaded(true);

        let selection = shed_if_all_overloaded(std::slice::from_ref(&a), "m").expect("shed");
        let dispatch = shed_if_worker_overloaded(a.as_ref(), "m").expect("shed");
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

    /// An empty pool is a 404/unavailable question for the caller, not a shed.
    #[test]
    fn empty_pool_is_not_a_shed() {
        assert!(shed_if_all_overloaded(&[], "nobody").is_none());
        assert!(!all_overloaded(&[]));
    }

    #[test]
    fn dispatch_recheck_sheds_only_for_a_flagged_worker() {
        let w = worker("http://127.0.0.1:9803", "m");
        assert!(shed_if_worker_overloaded(w.as_ref(), "m").is_none());

        w.set_overloaded(true);
        let response = shed_if_worker_overloaded(w.as_ref(), "m").expect("shed");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            extract_error_code_from_response(&response),
            "no_available_workers"
        );
    }
}
