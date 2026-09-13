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
    http::{header::RETRY_AFTER, HeaderValue, StatusCode},
    response::Response,
};
use tracing::debug;

use crate::{
    observability::metrics::Metrics,
    routers::{common::retry::mark_non_retryable, error},
    worker::{
        overload::{
            BRANCH_ALL_OVERLOADED_SHED, BRANCH_OVERLOADED_AT_DISPATCH, BRANCH_PD_ADMISSION_SHED,
            STAGE_DISPATCH, STAGE_PD_ADMISSION, STAGE_SELECTION,
        },
        Worker, WorkerRegistry,
    },
};

/// Retry-After seconds advertised on capacity responses. The worker-overload
/// veto cannot clear faster than the load-monitor poll interval; the same
/// process-wide value is a conservative fallback for temporarily empty pools
/// and upstream 429s without their own hint.
static SHED_RETRY_AFTER_SECS: AtomicU64 = AtomicU64::new(10);

/// Client-visible, gateway-owned code for a worker-overload-protection shed.
///
/// This deliberately covers both an all-overloaded candidate pool and a
/// selection-to-dispatch re-check after compatible alternatives are exhausted.
pub(crate) const WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE: &str =
    "worker_overload_protection_shed";

/// Client-visible code for a known model with no currently usable worker.
pub(crate) const NO_AVAILABLE_WORKERS_ERROR_CODE: &str = "no_available_workers";

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

/// Dispatch-time re-check: one atomic read on the already-chosen worker.
/// Callers that still own an unsent request should attempt compatible
/// reselection before returning this response.
pub fn shed_if_worker_overloaded(worker: &dyn Worker, model_id: &str) -> Option<Response> {
    if !worker.is_overloaded() {
        return None;
    }
    Some(shed_worker_overloaded(worker, model_id))
}

/// Build the dispatch-time overload response for a worker already observed
/// overloaded. Keeping response construction separate from the atomic read
/// lets reselection paths avoid counting a shed unless no alternative exists.
pub(crate) fn shed_worker_overloaded(worker: &dyn Worker, model_id: &str) -> Response {
    let url = worker.url();
    shed(
        BRANCH_OVERLOADED_AT_DISPATCH,
        STAGE_DISPATCH,
        url,
        format!("Worker '{url}' for model '{model_id}' became overloaded before dispatch"),
    )
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

/// Apply the public capacity contract to an existing 429 without replacing
/// an upstream response body or an upstream `Retry-After` value.
pub(crate) fn apply_capacity_contract(response: &mut Response) {
    if response.status() != StatusCode::TOO_MANY_REQUESTS {
        return;
    }
    if !response.headers().contains_key(RETRY_AFTER) {
        response.headers_mut().insert(
            RETRY_AFTER,
            HeaderValue::from(SHED_RETRY_AFTER_SECS.load(Ordering::Relaxed)),
        );
    }
    mark_non_retryable(response);
}

/// Return capacity pressure for a known model while preserving 404 for an
/// identity this router has never observed.
pub(crate) fn unavailable_or_not_found(
    registry: &WorkerRegistry,
    model_id: &str,
    message: impl Into<String>,
) -> Response {
    if registry.is_known_model(model_id) {
        no_available_workers(message)
    } else {
        error::model_not_found(model_id)
    }
}

/// Stable response for a known model whose workers cannot currently accept
/// work.
pub(crate) fn no_available_workers(message: impl Into<String>) -> Response {
    capacity_response(NO_AVAILABLE_WORKERS_ERROR_CODE, message)
}

fn capacity_response(code: &'static str, message: impl Into<String>) -> Response {
    let mut response = error::too_many_requests(code, message);
    apply_capacity_contract(&mut response);
    response
}

/// One decision line, one counter, one response — marked non-retryable: the
/// veto clears at the poll interval, which no backoff window outlives, and a
/// terminal shed is what keeps the counter per-request rather than per-attempt.
/// Retry-After carries that interval so clients and proxies pace themselves;
/// internal retries stay off regardless.
fn shed(branch: &'static str, stage: &'static str, worker: &str, message: String) -> Response {
    Metrics::record_worker_overload_shed(stage);
    debug!(branch, stage, worker, "Overload shed");
    capacity_response(WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE, message)
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

    /// The shed is a distinct 429, taken from the candidate pool rather than
    /// the model index.
    #[test]
    fn all_overloaded_sheds_with_distinct_429_code() {
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
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
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

        let selection = shed_if_all_overloaded(std::slice::from_ref(&a), "m").expect("shed");
        assert!(
            is_retryable_status(selection.status()),
            "429 is ordinarily retryable before the terminal marker is considered"
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
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
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

    #[test]
    fn known_model_without_workers_is_capacity_but_unknown_model_is_not_found() {
        let registry = WorkerRegistry::new();

        let unknown = unavailable_or_not_found(&registry, "unknown", "no worker");
        assert_eq!(unknown.status(), StatusCode::NOT_FOUND);

        registry.remember_model("known");
        let known = unavailable_or_not_found(&registry, "known", "no worker");
        assert_eq!(known.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            extract_error_code_from_response(&known),
            NO_AVAILABLE_WORKERS_ERROR_CODE
        );
        assert!(known.headers().contains_key(RETRY_AFTER));
        assert!(!is_retryable_response(&known));
    }

    #[test]
    fn capacity_contract_preserves_upstream_retry_after_and_adds_a_fallback() {
        let mut hinted = error::too_many_requests("upstream_busy", "busy");
        hinted
            .headers_mut()
            .insert(RETRY_AFTER, HeaderValue::from_static("17"));
        apply_capacity_contract(&mut hinted);
        assert_eq!(hinted.headers().get(RETRY_AFTER).unwrap(), "17");
        assert!(!is_retryable_response(&hinted));

        let mut unhinted = error::too_many_requests("upstream_busy", "busy");
        apply_capacity_contract(&mut unhinted);
        assert!(unhinted
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            .is_some_and(|seconds| seconds >= 1));
        assert!(!is_retryable_response(&unhinted));

        let mut unrelated = error::service_unavailable("unavailable", "down");
        apply_capacity_contract(&mut unrelated);
        assert!(!unrelated.headers().contains_key(RETRY_AFTER));
        assert!(is_retryable_response(&unrelated));
    }
}
