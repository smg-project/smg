//! The gateway's cases over the worker's health modules: each reads something
//! of the gateway beside them (the gRPC client's transport question, the
//! routers' retry rule), so it stays with the gateway.

use axum::http::StatusCode;
use openai_protocol::worker::ResilienceUpdate;
use smg_grpc_client::{h2, is_transport_failure};
use smg_worker::settings::RetryConfig;

use crate::{
    routers::common::retry::is_retryable_status,
    worker::health::{
        circuit_breaker::CircuitBreakerConfig, liveness::is_connection_code,
        resilience::resolve_resilience,
    },
};

/// The gRPC client's `is_transport_failure` answers another question
/// (is this status the HTTP/2 layer's doing rather than the worker's
/// answer?) from the status's source chain, and the two disagree by
/// design in both directions.
#[test]
fn a_connection_code_and_a_transport_failure_are_different_questions() {
    // An Unavailable the worker answered with: unreachable for work, yet
    // not the transport's doing.
    let answered = tonic::Status::unavailable("shutting down");
    assert!(is_connection_code(answered.code()));
    assert!(!is_transport_failure(&answered));

    // An Internal tonic made of a protocol fault: the transport's doing,
    // yet its code is the one a worker's own failure carries.
    let fault = tonic::Status::from_error(Box::new(h2::Error::from(h2::Reason::PROTOCOL_ERROR)));
    assert_eq!(fault.code(), tonic::Code::Internal);
    assert!(is_transport_failure(&fault));
    assert!(!is_connection_code(fault.code()));

    // A refused stream is both: an Unavailable with an HTTP/2 cause.
    let refused = tonic::Status::from_error(Box::new(h2::Error::from(h2::Reason::REFUSED_STREAM)));
    assert_eq!(refused.code(), tonic::Code::Unavailable);
    assert!(is_transport_failure(&refused));
    assert!(is_connection_code(refused.code()));
}

#[test]
fn partial_retryable_override_leaves_capacity_pushback_retryable() {
    // Pins why the capacity set is not merged into the retryable set: a
    // per-worker retryable override that drops 429 cannot make pushback
    // non-retryable, because the retry gate is router-global.
    let overrides = ResilienceUpdate {
        retryable_status_codes: Some(vec![500, 502]),
        ..Default::default()
    };
    let (resolved, _cb) = resolve_resilience(
        &RetryConfig::default(),
        &CircuitBreakerConfig::default(),
        true,
        true,
        &overrides,
    );

    // The override replaces the default set verbatim - 429 is not merged in.
    assert!(!resolved.retryable_status_codes.contains(&429));
    // Yet 429 is still retried, because retryability never consults this set.
    assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
    // And it still records no circuit-breaker sample, because
    // `record_outcome` returns on the capacity set before it ever reads
    // `retryable_status_codes`.
    assert!(resolved.capacity_status_codes.contains(&429));
}
