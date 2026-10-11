//! Metrics helper functions (shared by HTTP routers and gRPC pipeline).

use http::StatusCode;

use crate::observability::metrics::metrics_labels;

/// Map route path to endpoint label for metrics
pub(crate) fn route_to_endpoint(route: &str) -> &'static str {
    match route {
        "/v1/chat/completions" => metrics_labels::ENDPOINT_CHAT,
        "/generate" => metrics_labels::ENDPOINT_GENERATE,
        "/v1/completions" => metrics_labels::ENDPOINT_COMPLETIONS,
        "/v1/rerank" => metrics_labels::ENDPOINT_RERANK,
        "/v1/responses" => metrics_labels::ENDPOINT_RESPONSES,
        "/v1/decisions" => metrics_labels::ENDPOINT_DECISIONS,
        "/v1/systemone" => metrics_labels::ENDPOINT_SYSTEMONE,
        "/v1/messages" => metrics_labels::ENDPOINT_MESSAGES,
        "/v1/audio/transcriptions" => metrics_labels::ENDPOINT_AUDIO_TRANSCRIPTIONS,
        _ => "other",
    }
}

/// Map HTTP status code to error type label for metrics.
///
/// A 429 is capacity pushback, counted as the `rate_limit_error` the client
/// reads in the error body, not as an internal error of the router.
pub(crate) fn error_type_from_status(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 => metrics_labels::ERROR_VALIDATION,
        404 => metrics_labels::ERROR_NO_WORKERS,
        408 | 504 => metrics_labels::ERROR_TIMEOUT,
        429 => metrics_labels::ERROR_RATE_LIMIT,
        500..=599 => metrics_labels::ERROR_BACKEND,
        _ => metrics_labels::ERROR_INTERNAL,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routers::error::error_type_for_status;

    #[test]
    fn a_429_counts_as_the_rate_limit_error_the_client_reads() {
        let status = StatusCode::TOO_MANY_REQUESTS;
        assert_eq!(error_type_from_status(status), "rate_limit_error");
        // The label the counter carries is the type the error body carries.
        assert_eq!(
            error_type_from_status(status),
            error_type_for_status(status)
        );
        // The rest of the map stands.
        for (status, error_type) in [
            (StatusCode::BAD_REQUEST, "validation_error"),
            (StatusCode::NOT_FOUND, "no_workers"),
            (StatusCode::REQUEST_TIMEOUT, "timeout"),
            (StatusCode::GATEWAY_TIMEOUT, "timeout"),
            (StatusCode::INTERNAL_SERVER_ERROR, "backend_error"),
            (StatusCode::SERVICE_UNAVAILABLE, "backend_error"),
            (StatusCode::CONFLICT, "internal_error"),
        ] {
            assert_eq!(error_type_from_status(status), error_type, "{status}");
        }
    }
}
