//! One request's bookkeeping across its retry attempts.
//!
//! The retry loops (the HTTP proxy's executor, the gRPC pipeline's dispatch
//! loop) treat every attempt alike; the circuit breaker must not. Two rules
//! live here so both loops apply them the same way:
//!
//! - A worker is charged at most one breaker failure per request, however
//!   many attempts the request made on it. `cb_failure_threshold` counts
//!   failed requests, and its meaning no longer moves with
//!   `retry_max_retries` (five retries used to turn two failed requests into
//!   the ten failures the default threshold needs).
//! - An answer the worker itself produced for the request (an upstream 500
//!   with its body on the HTTP wire, a gRPC `INTERNAL` status) is definitive
//!   for that payload: a replay on the same worker costs the same work for
//!   the same answer. The request may still move to another available
//!   worker; when none is left, the answer is returned as it is.
//!
//! Connect failures, timeouts and the "not served now" statuses (408, 502,
//! 503, 504) stay transient: they are replayed as before, charged once.
//!
//! One side of a disaggregated pair can also be given up on alone: a prefill
//! side that produced nothing within its silence bound has, for this request,
//! told its worker's side of the story, and a replay there would wait out
//! the same bound. The response names the side ([`FailedSide`]); the retry
//! window closes on that worker only, its partner stays selectable, and
//! when no other worker of the side's kind is available the answer stands.

use std::sync::{Arc, OnceLock};

use axum::{http::StatusCode, response::Response};
use parking_lot::Mutex;
use smg_worker::identity::basic::{Worker, WorkerType};

use crate::routers::common::{placement::CandidateFilter, retry::mark_non_retryable};

/// Response extension: the worker answered the request itself. The status is
/// the worker's own, or the router's judgement of an answer the worker gave
/// (the Decisions adapter's 502 for an answer it cannot map). A transport
/// failure carries no mark.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UpstreamAnswer;

/// Mark `response` as answered by the worker (see [`UpstreamAnswer`]).
pub(crate) fn mark_upstream_answer(response: &mut Response) {
    response.extensions_mut().insert(UpstreamAnswer);
}

/// Whether the worker itself answered `response` (see [`UpstreamAnswer`]).
pub(crate) fn is_upstream_answer(response: &Response) -> bool {
    response.extensions().get::<UpstreamAnswer>().is_some()
}

/// Error code of the 502 a route answers for a worker's body over the
/// limit. The worker answered, and the route refused what it gave: the
/// refusal bears no mark ([`UpstreamAnswer`]) and is known by this code.
pub(crate) const WORKER_BODY_TOO_LARGE: &str = "upstream_response_too_large";

/// Response extension: the attempt failed on one side of a disaggregated pair,
/// named by its engine, and that side alone is what the request must not be
/// replayed on (see [`AttemptLedger::settle`]).
#[derive(Debug, Clone)]
pub(crate) struct FailedSide {
    engine: String,
    worker_type: WorkerType,
}

/// Mark `response` as the failure of `worker`'s side alone (see [`FailedSide`]).
pub(crate) fn mark_failed_side(response: &mut Response, worker: &dyn Worker) {
    response.extensions_mut().insert(FailedSide {
        engine: worker.base_url().to_string(),
        worker_type: *worker.worker_type(),
    });
}

/// Response extension: the worker's stream failed after the worker took the
/// call, and its breaker is not charged yet. The stream's reader has no
/// ledger: the dispatch loop charges it ([`AttemptLedger::charge_pending`]).
#[derive(Clone)]
pub(crate) struct PendingCharge {
    worker: Arc<dyn Worker>,
    status_code: u16,
}

/// Mark `response` as the failure of `worker`'s stream, which its breaker
/// samples as `status_code` (see [`PendingCharge`]).
pub(crate) fn mark_pending_charge(
    response: &mut Response,
    worker: &Arc<dyn Worker>,
    status_code: u16,
) {
    response.extensions_mut().insert(PendingCharge {
        worker: Arc::clone(worker),
        status_code,
    });
}

/// The kind of the worker whose side alone failed, where `response` names
/// one ([`FailedSide`]).
pub(crate) fn failed_side_type(response: &Response) -> Option<WorkerType> {
    let side = response.extensions().get::<FailedSide>();
    side.map(|side| side.worker_type)
}

/// Whether `response` is a failure the answering worker must not see again
/// for this request: the worker's own 500, the one retryable status a worker
/// gives a request it processed and failed on. Its "not served now" statuses
/// (502, 503, 504) stay replayable.
pub(crate) fn is_definitive_failure(response: &Response) -> bool {
    response.status() == StatusCode::INTERNAL_SERVER_ERROR && is_upstream_answer(response)
}

/// Whether `response` closes the retry window: it names a failed side
/// ([`FailedSide`]) or is a definitive failure. [`AttemptLedger::settle`]
/// passes any other response untouched, so a caller that asks this first
/// takes its pool only for a response `settle` reads it for.
pub(crate) fn closes_retry_window(response: &Response) -> bool {
    failed_side_type(response).is_some() || is_definitive_failure(response)
}

/// What one request's attempts have told the worker pool so far.
#[derive(Debug, Default)]
pub(crate) struct AttemptLedger {
    inner: Mutex<Ledger>,
    /// Made at the request's first failure, or for its first holder.
    charges: OnceLock<Charges>,
}

/// The workers (by URL) one request already charged a circuit-breaker
/// failure. A task that outlives its attempt holds a clone, so the outcome it
/// meets late is counted by the request's rule.
#[derive(Clone, Debug, Default)]
pub(crate) struct Charges(Arc<Mutex<Vec<String>>>);

impl Charges {
    /// [`AttemptLedger::record_outcome`], through the request's charges.
    pub(crate) fn record_outcome(&self, worker: &dyn Worker, status_code: u16) {
        if worker.is_breaker_failure(status_code) {
            let mut charged = self.0.lock();
            if charged.iter().any(|url| url == worker.url()) {
                return;
            }
            charged.push(worker.url().to_string());
        }
        worker.record_outcome(status_code);
    }
}

#[derive(Debug, Default)]
struct Ledger {
    /// Engines (by base URL, so one DP rank's answer stands for its
    /// siblings) that answered the request definitively.
    answered: Vec<String>,
}

impl AttemptLedger {
    /// Record an attempt's outcome on `worker`'s circuit breaker: a success or
    /// a capacity pushback as always, a failure only the first time this
    /// request fails on the worker.
    pub(crate) fn record_outcome(&self, worker: &dyn Worker, status_code: u16) {
        if worker.is_breaker_failure(status_code) {
            self.charges().record_outcome(worker, status_code);
        } else {
            worker.record_outcome(status_code);
        }
    }

    /// The request's charges, for a task that outlives the attempt it
    /// reports on.
    pub(crate) fn charges(&self) -> &Charges {
        self.charges.get_or_init(Charges::default)
    }

    /// Charge the worker whose stream failed under the attempt
    /// ([`PendingCharge`]), by the rule of [`Self::record_outcome`].
    pub(crate) fn charge_pending(&self, response: &Response) {
        if let Some(charge) = response.extensions().get::<PendingCharge>() {
            self.record_outcome(charge.worker.as_ref(), charge.status_code);
        }
    }

    /// Whether a retry attempt may select `worker`: not after it answered the
    /// request definitively.
    pub(crate) fn admits(&self, worker: &dyn Worker) -> bool {
        !self
            .inner
            .lock()
            .answered
            .iter()
            .any(|url| url == worker.base_url())
    }

    /// Runs `select` with the candidate filter the request's next attempt
    /// needs. Until a worker answered the request definitively there is
    /// none, and nothing is asked of each candidate here: a first attempt
    /// places over the pool as drawn, and a retry after transient failures
    /// alone does too, but for the engines it tried, which the selection
    /// steps past by itself while another candidate is left. After that the
    /// filter is [`Self::admits`].
    ///
    /// The check and the selection are two critical sections. That stands on
    /// a request's attempts running one after another: no answer is recorded
    /// between the two.
    pub(crate) fn with_candidate_filter<R>(
        &self,
        select: impl FnOnce(Option<CandidateFilter<'_>>) -> R,
    ) -> R {
        let nobody_answered = self.inner.lock().answered.is_empty();
        if nobody_answered {
            return select(None);
        }
        let admits = |worker: &dyn Worker| self.admits(worker);
        select(Some(&admits))
    }

    /// Close the retry window on a definitive failure. The workers that gave
    /// the answer take no further attempt of this request; when no available
    /// worker in `pool` is left to take one, the answer is marked terminal and
    /// goes back to the client as it is. A failure of one side alone
    /// ([`FailedSide`]) closes the window on that side's worker only, and the
    /// alternative is sought among the workers of its kind. Any other
    /// response passes untouched.
    pub(crate) fn settle<'a>(
        &self,
        response: &mut Response,
        answered_by: impl IntoIterator<Item = &'a Arc<dyn Worker>>,
        pool: impl IntoIterator<Item = &'a Arc<dyn Worker>>,
    ) {
        self.settle_with(response, answered_by, pool, |worker| worker.is_available());
    }

    /// As `settle`, using the route's availability check. A PD placement can
    /// admit an overloaded prefill when shedding is disabled.
    pub(crate) fn settle_with<'a>(
        &self,
        response: &mut Response,
        answered_by: impl IntoIterator<Item = &'a Arc<dyn Worker>>,
        pool: impl IntoIterator<Item = &'a Arc<dyn Worker>>,
        available: impl Fn(&dyn Worker) -> bool,
    ) {
        if !closes_retry_window(response) {
            return;
        }
        let side = response.extensions().get::<FailedSide>().cloned();
        {
            let mut ledger = self.inner.lock();
            for worker in answered_by {
                let engine = worker.base_url();
                if side.as_ref().is_some_and(|side| side.engine != engine) {
                    continue;
                }
                if !ledger.answered.iter().any(|url| url == engine) {
                    ledger.answered.push(engine.to_string());
                }
            }
        }
        let alternative = pool.into_iter().any(|worker| {
            side.as_ref()
                .is_none_or(|side| *worker.worker_type() == side.worker_type)
                && available(worker.as_ref())
                && self.admits(worker.as_ref())
        });
        if !alternative {
            mark_non_retryable(response);
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/routers/common/attempt_ledger/tests.rs"]
mod tests;
