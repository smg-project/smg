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
#[path = "../../tests/unit/rate_limit/rejection/tests.rs"]
mod tests;
