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
