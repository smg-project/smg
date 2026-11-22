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

/// Response extension: the worker answered the request itself, so the status
/// is the worker's verdict on this payload rather than a transport failure or
/// a status the router synthesized.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UpstreamAnswer;

/// Mark `response` as answered by the worker (see [`UpstreamAnswer`]).
pub(crate) fn mark_upstream_answer(response: &mut Response) {
    response.extensions_mut().insert(UpstreamAnswer);
}

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
    response.status() == StatusCode::INTERNAL_SERVER_ERROR
        && response.extensions().get::<UpstreamAnswer>().is_some()
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
        let side = response.extensions().get::<FailedSide>().cloned();
        if side.is_none() && !is_definitive_failure(response) {
            return;
        }
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
                && worker.is_available()
                && self.admits(worker.as_ref())
        });
        if !alternative {
            mark_non_retryable(response);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::response::IntoResponse;
    use smg_worker::{
        health::circuit_breaker::{CircuitBreakerConfig, CircuitState},
        test_support::no_health_check,
    };

    use super::*;
    use crate::routers::{
        common::{retry::is_retryable_response, snapshot::test_support::worker_of},
        grpc::backend_client::BasicWorkerBuilder,
    };

    fn worker(url: &str, failure_threshold: u32) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .circuit_breaker_config(CircuitBreakerConfig {
                    failure_threshold,
                    success_threshold: 1,
                    timeout_duration: Duration::from_secs(60),
                    window_duration: Duration::from_secs(60),
                })
                .health_config(no_health_check())
                .build(),
        )
    }

    fn answered(status: StatusCode) -> Response {
        let mut response = (status, "upstream").into_response();
        mark_upstream_answer(&mut response);
        response
    }

    #[test]
    fn a_request_charges_a_worker_once_however_often_it_retries_on_it() {
        let worker = worker("http://w1:8080", 2);

        let request = AttemptLedger::default();
        for _ in 0..5 {
            request.record_outcome(worker.as_ref(), 503);
        }
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);

        // The next request is a new observation: the threshold counts requests.
        AttemptLedger::default().record_outcome(worker.as_ref(), 503);
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    }

    #[test]
    fn successes_and_capacity_pushback_pass_through_unchanged() {
        let worker = worker("http://w1:8080", 2);
        let ledger = AttemptLedger::default();

        ledger.record_outcome(worker.as_ref(), 503);
        ledger.record_outcome(worker.as_ref(), 200);
        // The success reset the failure streak; the next failure is the
        // first of a new streak and still only the request's one charge.
        ledger.record_outcome(worker.as_ref(), 503);
        ledger.record_outcome(worker.as_ref(), 429);
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
    }

    #[test]
    fn a_definitive_answer_excludes_its_worker_and_ends_the_window_when_alone() {
        let w1 = worker("http://w1:8080", 5);
        let w2 = worker("http://w2:8080", 5);
        let pool = [Arc::clone(&w1), Arc::clone(&w2)];
        let ledger = AttemptLedger::default();

        let mut first = answered(StatusCode::INTERNAL_SERVER_ERROR);
        ledger.settle(&mut first, [&w1], pool.iter());
        assert!(!ledger.admits(w1.as_ref()));
        assert!(ledger.admits(w2.as_ref()));
        assert!(
            is_retryable_response(&first),
            "another worker can still take the request"
        );

        let mut second = answered(StatusCode::INTERNAL_SERVER_ERROR);
        ledger.settle(&mut second, [&w2], pool.iter());
        assert!(!ledger.admits(w2.as_ref()));
        assert!(
            !is_retryable_response(&second),
            "no worker is left: the answer is terminal"
        );
    }

    #[test]
    fn only_the_workers_own_500_is_definitive() {
        let w1 = worker("http://w1:8080", 5);
        let pool = [Arc::clone(&w1)];
        let ledger = AttemptLedger::default();

        // A 503 the worker sent is "not served now": replayable.
        let mut busy = answered(StatusCode::SERVICE_UNAVAILABLE);
        ledger.settle(&mut busy, [&w1], pool.iter());
        assert!(ledger.admits(w1.as_ref()));
        assert!(is_retryable_response(&busy));

        // A 500 the router synthesized (connect failure) is not the worker's answer.
        let mut transport = (StatusCode::INTERNAL_SERVER_ERROR, "connect").into_response();
        ledger.settle(&mut transport, [&w1], pool.iter());
        assert!(ledger.admits(w1.as_ref()));
        assert!(is_retryable_response(&transport));
    }

    #[test]
    fn a_candidate_filter_is_handed_only_after_a_definitive_answer() {
        let w1 = worker("http://w1:8080", 5);
        let w2 = worker("http://w2:8080", 5);
        let pool = [Arc::clone(&w1), Arc::clone(&w2)];
        let ledger = AttemptLedger::default();

        assert!(
            ledger.with_candidate_filter(|filter| filter.is_none()),
            "a first attempt asks nothing of a candidate"
        );

        // A transient failure charges the breaker and answers nothing.
        ledger.record_outcome(w1.as_ref(), 503);
        let mut busy = answered(StatusCode::SERVICE_UNAVAILABLE);
        ledger.settle(&mut busy, [&w1], pool.iter());
        assert!(
            ledger.with_candidate_filter(|filter| filter.is_none()),
            "a retry after transient failures alone asks nothing either"
        );

        let mut failed = answered(StatusCode::INTERNAL_SERVER_ERROR);
        ledger.settle(&mut failed, [&w1], pool.iter());
        let (admits_w1, admits_w2) = ledger.with_candidate_filter(|filter| {
            let admits = filter.expect("a definitive answer hands the filter");
            (admits(w1.as_ref()), admits(w2.as_ref()))
        });
        assert!(
            !admits_w1,
            "the worker that answered takes no later attempt"
        );
        assert!(admits_w2, "the other worker still can");

        // A second engine's answer keeps the filter in hand, and it now
        // refuses both.
        let mut failed_again = answered(StatusCode::INTERNAL_SERVER_ERROR);
        ledger.settle(&mut failed_again, [&w2], pool.iter());
        let (admits_w1, admits_w2) = ledger.with_candidate_filter(|filter| {
            let admits = filter.expect("a second definitive answer keeps the filter");
            (admits(w1.as_ref()), admits(w2.as_ref()))
        });
        assert!(
            !admits_w1 && !admits_w2,
            "neither worker that answered takes a later attempt"
        );
    }

    #[test]
    fn a_ranks_answer_stands_for_its_engine() {
        // One engine's ranks share a base address, and the ledger keeps an
        // answer by it: a definitive answer from one rank takes its sibling
        // out of the request's later attempts too, and no other engine.
        let rank = |rank: usize| -> Arc<dyn Worker> {
            Arc::new(
                BasicWorkerBuilder::new("http://engine:8080")
                    .dp_config(rank, 2)
                    .health_config(no_health_check())
                    .build(),
            )
        };
        let (first, sibling) = (rank(0), rank(1));
        let other = worker("http://other:8080", 5);
        let pool = [Arc::clone(&first), Arc::clone(&sibling), Arc::clone(&other)];
        let ledger = AttemptLedger::default();

        let mut failed = answered(StatusCode::INTERNAL_SERVER_ERROR);
        ledger.settle(&mut failed, [&first], pool.iter());
        assert!(!ledger.admits(first.as_ref()));
        assert!(
            !ledger.admits(sibling.as_ref()),
            "the sibling rank is the same engine"
        );
        assert!(ledger.admits(other.as_ref()));
        assert!(
            is_retryable_response(&failed),
            "another engine can still take the request"
        );
    }

    #[test]
    fn a_failed_side_excludes_its_worker_alone_and_ends_the_window_without_another_of_its_kind() {
        // A silent prefill side: the pair's decode did nothing wrong and stays
        // selectable; the retry goes to the other prefill worker, and once
        // that one has failed too the 504 stands even though a decode worker
        // is still available.
        let p1 = worker_of("grpc://p1:8000", WorkerType::Prefill);
        let p2 = worker_of("grpc://p2:8000", WorkerType::Prefill);
        let d1 = worker_of("grpc://d1:8000", WorkerType::Decode);
        let pool = [Arc::clone(&p1), Arc::clone(&p2), Arc::clone(&d1)];
        let ledger = AttemptLedger::default();

        let mut first = (StatusCode::GATEWAY_TIMEOUT, "silent").into_response();
        mark_failed_side(&mut first, p1.as_ref());
        ledger.settle(&mut first, [&p1, &d1], pool.iter());
        assert!(!ledger.admits(p1.as_ref()), "the silent prefill is out");
        assert!(ledger.admits(d1.as_ref()), "its decode partner is not");
        assert!(ledger.admits(p2.as_ref()));
        assert!(
            is_retryable_response(&first),
            "another prefill worker can take the request"
        );

        let mut second = (StatusCode::GATEWAY_TIMEOUT, "silent").into_response();
        mark_failed_side(&mut second, p2.as_ref());
        ledger.settle(&mut second, [&p2, &d1], pool.iter());
        assert!(!ledger.admits(p2.as_ref()));
        assert!(ledger.admits(d1.as_ref()));
        assert!(
            !is_retryable_response(&second),
            "no prefill worker is left: the answer is terminal, whatever the decode pool holds"
        );
    }
}
