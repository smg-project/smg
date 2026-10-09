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

use std::sync::Arc;

use axum::{http::StatusCode, response::Response};
use parking_lot::Mutex;

use crate::{routers::common::retry::mark_non_retryable, worker::Worker};

/// Response extension: the worker answered the request itself, so the status
/// is the worker's verdict on this payload rather than a transport failure or
/// a status the router synthesized.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UpstreamAnswer;

/// Mark `response` as answered by the worker (see [`UpstreamAnswer`]).
pub(crate) fn mark_upstream_answer(response: &mut Response) {
    response.extensions_mut().insert(UpstreamAnswer);
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
}

#[derive(Debug, Default)]
struct Ledger {
    /// Workers (by URL) already charged a circuit-breaker failure.
    charged: Vec<String>,
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
            let mut ledger = self.inner.lock();
            if ledger.charged.iter().any(|url| url == worker.url()) {
                return;
            }
            ledger.charged.push(worker.url().to_string());
        }
        worker.record_outcome(status_code);
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
    /// goes back to the client as it is. Any other response passes untouched.
    pub(crate) fn settle<'a>(
        &self,
        response: &mut Response,
        answered_by: impl IntoIterator<Item = &'a Arc<dyn Worker>>,
        pool: impl IntoIterator<Item = &'a Arc<dyn Worker>>,
    ) {
        if !is_definitive_failure(response) {
            return;
        }
        {
            let mut ledger = self.inner.lock();
            for worker in answered_by {
                let engine = worker.base_url();
                if !ledger.answered.iter().any(|url| url == engine) {
                    ledger.answered.push(engine.to_string());
                }
            }
        }
        let alternative = pool
            .into_iter()
            .any(|worker| worker.is_available() && self.admits(worker.as_ref()));
        if !alternative {
            mark_non_retryable(response);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::response::IntoResponse;
    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{
        routers::common::retry::is_retryable_response,
        worker::{
            circuit_breaker::{CircuitBreakerConfig, CircuitState},
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
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
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
}
