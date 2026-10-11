//! Metrics helper functions (shared by HTTP routers and gRPC pipeline).

use axum::response::Response;
use http::StatusCode;

use crate::{
    observability::metrics::metrics_labels,
    routers::{
        common::{
            overload::WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE,
            placement::NO_AVAILABLE_WORKERS_CODE,
        },
        error::extract_error_code_from_response,
        grpc::common::stages::SELECTION_QUEUE_TIMEOUT_ERROR_CODE,
    },
};

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

/// The codes of the refusals the gateway answers on its own ahead of any
/// worker, each paced by a `Retry-After`: every worker of the model
/// unavailable (unhealthy or behind an open circuit breaker), the overload
/// shed, a wait past the selection queue's bound. The gateway's other
/// self-made 503s, `no_compatible_pd_pair` and `no_media_ref_capable_worker`,
/// are left out on purpose: they are configuration and capability
/// mismatches that no wait clears (no `Retry-After`), not the capacity
/// refusals this label names, and they keep their class by status until
/// one of their own is decided.
const GATEWAY_REFUSAL_CODES: &[&str] = &[
    NO_AVAILABLE_WORKERS_CODE,
    WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE,
    SELECTION_QUEUE_TIMEOUT_ERROR_CODE,
];

/// The `error_type` label of a response the gateway answers.
///
/// One of the gateway's own refusals ([`GATEWAY_REFUSAL_CODES`], recognised
/// by the code the response carries) counts as `unavailable`: the gateway
/// minted the code, no worker took the request, so it is not a backend's
/// error whatever its status. The one relay that copies a worker's own code
/// into the response, the Decisions path, counts a worker that answers with
/// one of these codes the same way; every other response counts by its
/// status through [`error_type_from_status`],
/// the map a worker's status takes at dispatch and after the first chunk
/// alike. The client's `error.type` is untouched: a 503 reads `server_error`
/// as before.
pub(crate) fn error_type_for_response(response: &Response) -> &'static str {
    if GATEWAY_REFUSAL_CODES.contains(&extract_error_code_from_response(response)) {
        return metrics_labels::ERROR_UNAVAILABLE;
    }
    error_type_from_status(response.status())
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

    /// A refusal the gateway answers on its own counts as `unavailable`,
    /// whatever its status; a worker's answer with the same status keeps its
    /// class; the client-facing type of either is the status's, unchanged.
    #[test]
    fn a_refusal_the_gateway_answers_on_its_own_counts_as_unavailable() {
        use crate::routers::{
            common::{overload, placement},
            error,
        };

        let refusals = [
            placement::no_available_workers("every worker is down"),
            overload::shed_pd_admission("decode-1", "m", 4, 5),
            error::service_unavailable(SELECTION_QUEUE_TIMEOUT_ERROR_CODE, "waited too long"),
        ];
        for refusal in &refusals {
            let code = extract_error_code_from_response(refusal);
            assert_eq!(refusal.status(), StatusCode::SERVICE_UNAVAILABLE, "{code}");
            assert_eq!(error_type_for_response(refusal), "unavailable", "{code}");
            assert_eq!(
                error_type_for_status(refusal.status()),
                "server_error",
                "the client reads the status's type, as before"
            );
        }
        // A worker's answers keep their classes, by status.
        for (response, error_type) in [
            (
                error::service_unavailable("engine_unavailable", "the worker is overloaded"),
                "backend_error",
            ),
            (
                error::internal_error("engine_failed", "the worker failed"),
                "backend_error",
            ),
            (
                error::too_many_requests("engine_busy", "kv cache full"),
                "rate_limit_error",
            ),
            (error::model_not_found("m"), "no_workers"),
            (
                error::bad_request("invalid_request", "bad"),
                "validation_error",
            ),
        ] {
            assert_eq!(
                error_type_for_response(&response),
                error_type,
                "{}",
                extract_error_code_from_response(&response)
            );
        }
    }
}
