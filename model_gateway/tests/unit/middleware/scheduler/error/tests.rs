use super::*;

#[test]
fn test_queue_full_maps_to_429_with_retry_after() {
    let resp = SchedulerError::QueueFull.into_response();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        resp.headers().get(RETRY_AFTER),
        Some(&HeaderValue::from(SHED_RETRY_AFTER_SECS))
    );
}

#[test]
fn test_queue_timeout_maps_to_503_with_retry_after() {
    let resp = SchedulerError::QueueTimeout.into_response();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        resp.headers().get(RETRY_AFTER),
        Some(&HeaderValue::from(SHED_RETRY_AFTER_SECS))
    );
}

#[test]
fn test_preempted_maps_to_503_with_headers() {
    let resp = SchedulerError::Preempted.into_response();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        resp.headers().get(RETRY_AFTER).map(|v| v.to_str().unwrap()),
        Some("1")
    );
    assert_eq!(
        resp.headers()
            .get(HEADER_X_SMG_PREEMPTED)
            .map(|v| v.to_str().unwrap()),
        Some("true")
    );
}

#[test]
fn test_client_cancelled_maps_to_499() {
    let resp = SchedulerError::ClientCancelled.into_response();
    assert_eq!(resp.status().as_u16(), 499);
}

#[test]
fn test_non_preempt_has_no_preempt_header() {
    let resp = SchedulerError::QueueFull.into_response();
    assert!(resp.headers().get(HEADER_X_SMG_PREEMPTED).is_none());
}

#[test]
fn test_client_cancelled_has_no_retry_after() {
    let resp = SchedulerError::ClientCancelled.into_response();
    assert!(resp.headers().get(RETRY_AFTER).is_none());
}

#[test]
fn test_from_rejection_reason_is_one_to_one() {
    assert_eq!(
        SchedulerError::from(RejectionReason::QueueFull),
        SchedulerError::QueueFull
    );
    assert_eq!(
        SchedulerError::from(RejectionReason::QueueTimeout),
        SchedulerError::QueueTimeout
    );
    assert_eq!(
        SchedulerError::from(RejectionReason::Preempted),
        SchedulerError::Preempted
    );
    assert_eq!(
        SchedulerError::from(RejectionReason::ClientCancelled),
        SchedulerError::ClientCancelled
    );
}
