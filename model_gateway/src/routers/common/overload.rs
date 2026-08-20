//! Shed responses for the absolute worker-overload guard.
//!
//! Both helpers run only after routing has already failed to produce a worker,
//! so nothing here is on the path of a served request. Their whole job is to
//! turn "the pool emptied because every worker is vetoed" into the immediate
//! 503 both transports use for `no_available_workers`, instead of the
//! circuit-breaker wording (HTTP) or a misleading 404 (gRPC).
//!
//! The verdict is taken from the candidate pool the caller selected over, never
//! from the model index: every selection site narrows by worker type and
//! connection mode first, so a whole-model predicate would miss exactly the
//! cases that matter — a saturated PD prefill leg, one transport of a
//! mixed-transport model, or the model-less `/generate` wildcard, which selects
//! across every model and has no model-index entry of its own.

use std::sync::Arc;

use axum::response::Response;
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

/// The dispatch-time re-check: one relaxed atomic read on the single worker
/// already chosen, covering the window between selection and dispatch in which
/// a load report can land.
///
/// Deliberately sheds rather than re-selecting. The flag moves once per
/// `load_monitor_interval_secs`, so a flip landing inside one selection→dispatch
/// gap is rare enough that the simpler answer costs nothing in aggregate — and
/// unlike the selection shed, this one says only what it knows: *this* worker
/// went over, not the fleet.
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

/// One decision line, one counter, one response.
///
/// The response is marked non-retryable: the condition it reports clears at the
/// load-poll interval, so the retry layer's backoff window cannot outlive it,
/// and retrying would re-run the entire pipeline (rate limiting included) five
/// more times to reach the same 503 ~400 ms later. Marking it terminal is also
/// what makes `smg_worker_overload_shed_total` count sheds per request rather
/// than per attempt.
fn shed(branch: &'static str, stage: &'static str, worker: &str, message: String) -> Response {
    Metrics::record_worker_overload_shed(stage);
    debug!(branch, stage, worker, "Overload shed");
    let mut response = error::service_unavailable("no_available_workers", message);
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
