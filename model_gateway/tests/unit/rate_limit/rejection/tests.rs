use axum::http::StatusCode;

use super::*;

#[test]
fn sets_status_code_and_retry_after() {
    let response = rejection_response(30);
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers().get(RETRY_AFTER).unwrap(), "30");
    assert_eq!(
        response
            .headers()
            .get(error::HEADER_X_SMG_ERROR_CODE)
            .unwrap(),
        RATE_LIMIT_ERROR_CODE
    );
}

#[test]
fn zero_retry_after_omits_header() {
    let response = rejection_response(0);
    assert!(response.headers().get(RETRY_AFTER).is_none());
}

#[test]
fn impossible_retry_after_sentinel_omits_header() {
    let response = rejection_response(u64::MAX);
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(response.headers().get(RETRY_AFTER).is_none());
}
