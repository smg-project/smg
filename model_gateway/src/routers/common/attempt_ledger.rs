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

use std::sync::Arc;

use axum::{http::StatusCode, response::Response};
use parking_lot::Mutex;

use crate::{
    routers::common::retry::mark_non_retryable,
    worker::{Worker, WorkerType},
};

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

/// Whether `response` is a failure the answering worker must not see again
/// for this request: the worker's own 500, the one retryable status a worker
/// gives a request it processed and failed on. Its "not served now" statuses
/// (502, 503, 504) stay replayable.
pub(crate) fn is_definitive_failure(response: &Response) -> bool {
    response.status() == StatusCode::INTERNAL_SERVER_ERROR
        && response.extensions().get::<UpstreamAnswer>().is_some()
}

/// What one request's attempts have told the worker pool so far. A clone is
/// a handle to the same record: the dispatch stage's result carries the
/// attempt's ledger to the plain collector, which samples a prefill side's
/// breaker for it (see `ExecutionResult::PrefillDecode`).
#[derive(Debug, Default, Clone)]
pub(crate) struct AttemptLedger {
    inner: Arc<Mutex<Ledger>>,
}

/// The phase of the prefill side a dispatch waits on alone, or waited on
/// when it was dropped (see [`AttemptLedger::prefill_phase_pending`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefillPhase {
    /// Its call, before its answer: the sequential shape's dispatch and
    /// first-response wait.
    Call,
    /// Its read to its end, after it answered: the sequential shape's drain,
    /// the parallel shape's early read once the decode side answered.
    Read,
}

#[derive(Debug, Default)]
struct Ledger {
    /// Workers (by URL) already charged a circuit-breaker failure.
    charged: Vec<String>,
    /// Engines (by base URL, so one DP rank's answer stands for its
    /// siblings) that answered the request definitively.
    answered: Vec<String>,
    /// Prefill sides this attempt's dispatch waits on alone, before their
    /// answer (see [`AttemptLedger::prefill_call`]).
    prefill_calls: usize,
    /// Prefill sides this attempt's dispatch waits on alone, read to their
    /// end after their answer (see [`AttemptLedger::prefill_drain`]).
    prefill_drains: usize,
}

impl AttemptLedger {
    /// Record an attempt's outcome on `worker`'s circuit breaker: a success or
    /// a capacity pushback as always, a failure only the first time this
    /// request fails on the worker.
    pub(crate) fn record_outcome(&self, worker: &dyn Worker, status_code: u16) {
        if worker.is_breaker_failure(status_code) {
            let mut ledger = self.inner.lock();
            if ledger.charged.iter().any(|url| url == worker.url()) {
                return;
            }
            ledger.charged.push(worker.url().to_string());
        }
        worker.record_outcome(status_code);
    }

    /// The start of an attempt's dispatch: no call or read of an earlier
    /// attempt's is in flight.
    pub(crate) fn attempt_begins(&self) {
        let mut ledger = self.inner.lock();
        ledger.prefill_calls = 0;
        ledger.prefill_drains = 0;
    }

    /// A prefill side's call the dispatch waits on alone begins (the
    /// sequential shape, whose decode side is not dispatched before the
    /// prefill's harvest). It is counted until [`PrefillCall::answered`]
    /// turns it into the side's read, and a dispatch dropped by the attempt's
    /// deadline leaves it counted, so the deadline is charged to the prefill
    /// side the dispatch was waiting on, not to the decode side its answer
    /// otherwise waits on (see `dispatch_timed_out`).
    pub(crate) fn prefill_call(&self) -> PrefillCall<'_> {
        self.inner.lock().prefill_calls += 1;
        PrefillCall { ledger: self }
    }

    /// A prefill side's read to its end, after it answered, begins: the
    /// parallel shape's early read once the decode side answered (the
    /// sequential shape's drain comes through [`PrefillCall::answered`]). It
    /// is counted until [`PrefillDrain::ended`], and a dispatch dropped by
    /// the attempt's deadline leaves it counted, as a call is.
    pub(crate) fn prefill_drain(&self) -> PrefillDrain<'_> {
        self.inner.lock().prefill_drains += 1;
        PrefillDrain { ledger: self }
    }

    /// The phase of the prefill side the dispatch waits on alone, or waited
    /// on when it was dropped: a read when one is in flight (a batch may
    /// hold calls and reads at once), else a call, else `None`.
    pub(crate) fn prefill_phase_pending(&self) -> Option<PrefillPhase> {
        let ledger = self.inner.lock();
        if ledger.prefill_drains > 0 {
            Some(PrefillPhase::Read)
        } else if ledger.prefill_calls > 0 {
            Some(PrefillPhase::Call)
        } else {
            None
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

/// A prefill side's call before its answer, counted on its attempt's ledger
/// (see [`AttemptLedger::prefill_call`]) until the side answers; a call
/// dropped before that, with the dispatch the deadline cut, stays counted by
/// design.
#[must_use = "the call is counted until the side answers"]
pub(crate) struct PrefillCall<'a> {
    ledger: &'a AttemptLedger,
}

impl<'a> PrefillCall<'a> {
    /// The side answered: its count is its read's from here, until
    /// [`PrefillDrain::ended`].
    pub(crate) fn answered(self) -> PrefillDrain<'a> {
        {
            let mut ledger = self.ledger.inner.lock();
            ledger.prefill_calls = ledger.prefill_calls.saturating_sub(1);
            ledger.prefill_drains += 1;
        }
        PrefillDrain {
            ledger: self.ledger,
        }
    }
}

/// A prefill side's read to its end after it answered, counted on its
/// attempt's ledger (see [`AttemptLedger::prefill_drain`] and
/// [`PrefillCall::answered`]) until it ends; a read dropped before that, with
/// the dispatch the deadline cut, stays counted by design.
#[must_use = "the read is counted until it is ended"]
pub(crate) struct PrefillDrain<'a> {
    ledger: &'a AttemptLedger,
}

impl PrefillDrain<'_> {
    /// The read reached the side's end, or its failure: no longer in flight.
    pub(crate) fn ended(self) {
        let mut ledger = self.ledger.inner.lock();
        ledger.prefill_drains = ledger.prefill_drains.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::response::IntoResponse;

    use super::*;
    use crate::{
        routers::common::retry::is_retryable_response,
        worker::{
            circuit_breaker::{CircuitBreakerConfig, CircuitState},
            test_support::{no_health_check, worker_of},
            BasicWorkerBuilder,
        },
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

    #[test]
    fn a_prefill_side_stays_counted_in_its_phase_until_it_ends_or_the_next_attempt_begins() {
        let ledger = AttemptLedger::default();
        assert_eq!(ledger.prefill_phase_pending(), None);
        // The sequential shape: the call, then the read once the side answered.
        let call = ledger.prefill_call();
        assert_eq!(ledger.prefill_phase_pending(), Some(PrefillPhase::Call));
        let read = call.answered();
        assert_eq!(ledger.prefill_phase_pending(), Some(PrefillPhase::Read));
        read.ended();
        assert_eq!(
            ledger.prefill_phase_pending(),
            None,
            "an ended read is out of flight"
        );
        // A read dropped with its dispatch (the deadline passed) stays counted,
        // and names the phase over a call in flight at the same time.
        {
            let _read = ledger.prefill_drain();
            let _call = ledger.prefill_call();
        }
        assert_eq!(
            ledger.prefill_phase_pending(),
            Some(PrefillPhase::Read),
            "a read dropped mid-way is the deadline's to charge"
        );
        ledger.attempt_begins();
        assert_eq!(
            ledger.prefill_phase_pending(),
            None,
            "the next attempt starts clean"
        );
        // A call dropped before its answer stays counted as a call.
        {
            let _call = ledger.prefill_call();
        }
        assert_eq!(
            ledger.prefill_phase_pending(),
            Some(PrefillPhase::Call),
            "a call dropped before the side answered is the deadline's to charge as a call"
        );
        ledger.attempt_begins();
        assert_eq!(ledger.prefill_phase_pending(), None);
    }
}
