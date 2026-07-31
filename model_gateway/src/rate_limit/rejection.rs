//! Builds the tenant-rate-limit rejection response: `429`, the standard
//! SMG error envelope, and `Retry-After` when a wait time is known.

use axum::{
    http::{header::RETRY_AFTER, HeaderValue},
    response::Response,
};

use crate::routers::error;

pub const RATE_LIMIT_ERROR_CODE: &str = "tenant_rate_limit_exceeded";

/// `retry_after_secs` of `0` omits the `Retry-After` header (nothing to
/// wait for is a caller bug, not a real rejection state). `u64::MAX` is
/// `ScopeBucket::dry_run`'s "this request exceeds the scope's total
/// capacity and can never be admitted, no matter how long the caller
/// waits" sentinel -- surfacing that literally would tell the client to
/// wait about 584 billion years, so it's omitted the same way; the error
/// body still explains why.
pub fn rejection_response(retry_after_secs: u64) -> Response {
    let mut response = error::too_many_requests(
        RATE_LIMIT_ERROR_CODE,
        "Tenant rate limit exceeded for this request",
    );
    if retry_after_secs > 0 && retry_after_secs != u64::MAX {
        if let Ok(value) = HeaderValue::from_str(&retry_after_secs.to_string()) {
            response.headers_mut().insert(RETRY_AFTER, value);
        }
    }
    response
}

#[cfg(test)]
mod tests {
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
}
