//! Metrics helper functions (shared by HTTP routers and gRPC pipeline).

use axum::response::Response;
use http::StatusCode;
use llm_multimodal::MediaErrorCode;
use smg_external_router::retry::REQUEST_TIMEOUT_CODE;

use super::tonic_ext::CONSTRAINT_FAILED_CODE;
use crate::{
    observability::metrics::{error_reason_from_status, metrics_labels},
    rate_limit::RATE_LIMIT_ERROR_CODE,
    routers::{
        common::{
            overload::WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE,
            placement::{
                MODEL_NOT_FOUND_CODE, NO_AVAILABLE_WORKERS_CODE, NO_MEDIA_REF_CAPABLE_WORKER_CODE,
            },
        },
        error::extract_error_code_from_response,
        grpc::common::stages::SELECTION_QUEUE_TIMEOUT_ERROR_CODE,
        PD_PREFILL_QUEUE_FULL,
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
/// one of these codes the same way. A failure of the request's media
/// (a response under one of the [`MediaErrorCode`]s, from the router's own
/// media pipeline or a worker's report) counts as the request's error,
/// `validation_error`, whichever status tells the client what to do about
/// it: a 400 to fix the reference, a 502 to try the host again later; the
/// reference is the request's input, and a host a client names must not
/// count as a backend's failure. Every other response counts by its status
/// through [`error_type_from_status`], the map a worker's status takes at
/// dispatch and after the first chunk alike. The client's `error.type` is
/// untouched: a 503 reads `server_error` as before.
pub(crate) fn error_type_for_response(response: &Response) -> &'static str {
    let code = extract_error_code_from_response(response);
    if GATEWAY_REFUSAL_CODES.contains(&code) {
        return metrics_labels::ERROR_UNAVAILABLE;
    }
    // The engine could not continue the request under its constraint: the
    // request's outcome, counted on its own (see
    // `TonicStatusExt::constraint_failure_response`).
    if code == CONSTRAINT_FAILED_CODE {
        return metrics_labels::ERROR_CONSTRAINT;
    }
    // The request's media failed (see `TonicStatusExt::media_fault`,
    // `multimodal::processing_failed_response`): the request's error whichever
    // status tells the client what to do about it.
    if MediaErrorCode::parse(code).is_some() {
        return metrics_labels::ERROR_VALIDATION;
    }
    error_type_from_status(response.status())
}

/// Response extension: the `reason` label of the router error this response
/// ends a request with, decided where the cause was known (the worker's
/// status, see `TonicStatusExt::error_reason`) and read where the error is
/// counted (see [`error_reason_for_response`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ErrorReason(pub(crate) &'static str);

/// Carry `reason` on `response` for the router that counts the error.
pub(crate) fn mark_error_reason(response: &mut Response, reason: &'static str) {
    response.extensions_mut().insert(ErrorReason(reason));
}

/// The `reason` label of a response the gateway answers: the cause behind
/// [`error_type_for_response`]'s class, one of the `metrics_labels::REASON_*`
/// values.
///
/// A response made of a worker's status carries the reason the status named
/// ([`ErrorReason`], set by `TonicStatusExt::to_http_error`). Else the
/// gateway's own code names the refusal: a pool with no available worker (the
/// placement's verdict, the HTTP routers' and the prefill side's own words
/// for it), the overload shed and the prefill admission queue, the selection
/// queue's bound, a model nobody serves, the tenant limit, a constraint the
/// engine could not continue under, a pool with no capable worker, an
/// upstream call that timed out or could not connect, a worker's answer that
/// could not be read. A response with no gateway code is a worker's answer
/// relayed as it came, classed by its status; one with another gateway code
/// is the gateway's own: its 4xx is the request's fault, its 504 a deadline
/// (the silence verdicts), any other 5xx its own processing.
pub(crate) fn error_reason_for_response(response: &Response) -> &'static str {
    use metrics_labels::*;
    if let Some(ErrorReason(reason)) = response.extensions().get::<ErrorReason>() {
        return reason;
    }
    let status = response.status();
    match extract_error_code_from_response(response) {
        // routers/http/router.rs and pd_router.rs answer their empty pools
        // under `no_workers` and `no_prefill_servers`.
        NO_AVAILABLE_WORKERS_CODE | "no_workers" | "no_prefill_servers" => {
            REASON_NO_AVAILABLE_WORKERS
        }
        WORKER_OVERLOAD_PROTECTION_SHED_ERROR_CODE | PD_PREFILL_QUEUE_FULL => REASON_OVERLOAD_SHED,
        SELECTION_QUEUE_TIMEOUT_ERROR_CODE => REASON_SELECTION_QUEUE_TIMEOUT,
        MODEL_NOT_FOUND_CODE => REASON_MODEL_NOT_FOUND,
        RATE_LIMIT_ERROR_CODE => REASON_RATE_LIMITED,
        CONSTRAINT_FAILED_CODE => REASON_CONSTRAINT_FAILED,
        // The disaggregated routers' pair verdict (`no_compatible_pd_pair`).
        NO_MEDIA_REF_CAPABLE_WORKER_CODE | "no_compatible_pd_pair" => REASON_NO_CAPABLE_WORKER,
        REQUEST_TIMEOUT_CODE => REASON_DEADLINE,
        // The HTTP router's codes for a call that never reached the worker.
        "call_upstream_connection_failed" => REASON_TRANSPORT_DISCONNECT,
        // The decisions route's code for an answer it could not read.
        "invalid_decisions_response" => REASON_BAD_RESPONSE,
        "" => error_reason_from_status(status),
        _ => match status.as_u16() {
            408 | 504 => REASON_DEADLINE,
            429 => REASON_OVERLOADED,
            400..=499 => REASON_INVALID_REQUEST,
            _ => REASON_INTERNAL,
        },
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

    /// A worker's report that the engine could not continue the request under
    /// its constraint counts as `constraint_error`, under the code the client
    /// reads; a 422 of another making keeps the status's class.
    #[test]
    fn a_constraint_failure_counts_as_a_constraint_error() {
        use smg_grpc_client::constraint_failed_status;

        use crate::routers::{error, grpc::utils::tonic_ext::TonicStatusExt};

        let response = constraint_failed_status("the scheduler aborted the request: Aborted")
            .constraint_failure_response()
            .expect("the report is answered");
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(error_type_for_response(&response), "constraint_error");
        let other = error::create_error(StatusCode::UNPROCESSABLE_ENTITY, "other", "other");
        assert_eq!(error_type_for_response(&other), "internal_error");
    }

    /// A failure of the request's media counts as the request's error on both
    /// paths that answer it before any stream exists, the router's media
    /// pipeline and a worker's report, whichever status the client reads: the
    /// host's trouble (a 502) as much as the reference the request got wrong
    /// (a 400); a 502 of another making keeps the status's class.
    #[test]
    fn a_failure_of_the_requests_media_counts_as_the_requests_error_whatever_its_status() {
        use std::time::Duration;

        use llm_multimodal::{MediaConnectorError, MultiModalError};

        use crate::routers::{
            error,
            grpc::{multimodal::processing_failed_response, utils::tonic_ext::TonicStatusExt},
        };

        // The router's own pipeline: the host's trouble and the request's fault.
        for (media, status) in [
            (
                MediaConnectorError::Timeout {
                    url: "https://media.example/clip.mp4".to_string(),
                    budget: Duration::from_secs(30),
                },
                StatusCode::BAD_GATEWAY,
            ),
            (
                MediaConnectorError::UnsupportedScheme("ftp".to_string()),
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let failed = anyhow::Error::new(MultiModalError::Media(media))
                .context("Failed to finalize multimodal tracker");
            let response = processing_failed_response(&failed);
            assert_eq!(response.status(), status);
            assert_eq!(error_type_for_response(&response), "validation_error");
        }
        // A worker's report, in the pre-trailer form: answered a 502, counted
        // as the request's.
        let report = tonic::Status::unavailable(
            "media fetch of https://media.example/a.png timed out after 10s",
        );
        let response = report.to_http_error(report.media_code().as_str(), report.message().into());
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(error_type_for_response(&response), "validation_error");
        // A 502 of another making keeps the status's class.
        let other = error::bad_gateway("upstream_failed", "the worker's connection reset");
        assert_eq!(error_type_for_response(&other), "backend_error");
    }

    /// The reason behind a response's class: the status's own reason when
    /// the response was made of a worker's status; else the gateway's code
    /// (its refusals, the HTTP router's upstream-call codes, an answer it
    /// could not read); else the status, read as a relayed worker's answer
    /// when the response carries no gateway code and as the gateway's own
    /// when it does.
    #[test]
    fn a_response_names_the_reason_behind_its_class() {
        use metrics_labels::*;
        use smg_grpc_client::constraint_failed_status;

        use crate::routers::{
            common::{overload, placement},
            error,
            grpc::utils::tonic_ext::TonicStatusExt,
            prefill_queue_full,
        };

        // A worker's status, whatever code the answer is minted under.
        let from_status = |status: tonic::Status| {
            status.to_http_error("start_generation_failed", "failed".to_string())
        };
        for (response, reason) in [
            (
                from_status(tonic::Status::unavailable("draining")),
                REASON_ENGINE_UNAVAILABLE,
            ),
            (
                from_status(tonic::Status::internal("boom")),
                REASON_ENGINE_INTERNAL,
            ),
            (
                from_status(tonic::Status::cancelled("reset")),
                REASON_TRANSPORT_DISCONNECT,
            ),
            (
                from_status(tonic::Status::resource_exhausted("full")),
                REASON_OVERLOADED,
            ),
            (
                constraint_failed_status("the scheduler aborted the request: Aborted")
                    .constraint_failure_response()
                    .expect("answered"),
                REASON_CONSTRAINT_FAILED,
            ),
        ] {
            assert_eq!(
                error_reason_for_response(&response),
                reason,
                "{}",
                extract_error_code_from_response(&response)
            );
        }
        // The gateway's own refusals, by their code.
        for (response, reason) in [
            (
                placement::no_available_workers("every worker is down"),
                REASON_NO_AVAILABLE_WORKERS,
            ),
            (
                error::service_unavailable("no_workers", "No available workers"),
                REASON_NO_AVAILABLE_WORKERS,
            ),
            (
                error::service_unavailable("no_prefill_servers", "none"),
                REASON_NO_AVAILABLE_WORKERS,
            ),
            (
                overload::shed_pd_admission("decode-1", "m", 4, 5),
                REASON_OVERLOAD_SHED,
            ),
            (prefill_queue_full(), REASON_OVERLOAD_SHED),
            (
                error::service_unavailable(SELECTION_QUEUE_TIMEOUT_ERROR_CODE, "waited too long"),
                REASON_SELECTION_QUEUE_TIMEOUT,
            ),
            (error::model_not_found("m"), REASON_MODEL_NOT_FOUND),
            (
                crate::rate_limit::rejection_response(3),
                REASON_RATE_LIMITED,
            ),
            (
                placement::no_media_ref_capable_worker("no vision worker"),
                REASON_NO_CAPABLE_WORKER,
            ),
            (
                error::service_unavailable("no_compatible_pd_pair", "no pair"),
                REASON_NO_CAPABLE_WORKER,
            ),
            (
                smg_external_router::retry::request_timeout_response("upstream timed out"),
                REASON_DEADLINE,
            ),
            (
                error::internal_error("call_upstream_connection_failed", "refused"),
                REASON_TRANSPORT_DISCONNECT,
            ),
            (
                error::bad_gateway("invalid_decisions_response", "unreadable"),
                REASON_BAD_RESPONSE,
            ),
            // The gateway's other codes: by the status's side of the fault.
            (
                error::bad_request("invalid_request", "bad"),
                REASON_INVALID_REQUEST,
            ),
            (
                error::create_error(StatusCode::GATEWAY_TIMEOUT, "decode_worker_silent", "quiet"),
                REASON_DEADLINE,
            ),
            (
                error::internal_error("no_response_produced", "none"),
                REASON_INTERNAL,
            ),
            (
                error::service_unavailable("tokenizer_not_ready", "loading"),
                REASON_INTERNAL,
            ),
        ] {
            assert_eq!(
                error_reason_for_response(&response),
                reason,
                "{}",
                extract_error_code_from_response(&response)
            );
        }
        // A worker's answer relayed as it came carries no gateway code: the
        // status is the worker's.
        for (status, reason) in [
            (StatusCode::INTERNAL_SERVER_ERROR, REASON_ENGINE_INTERNAL),
            (StatusCode::BAD_GATEWAY, REASON_TRANSPORT_DISCONNECT),
            (StatusCode::SERVICE_UNAVAILABLE, REASON_ENGINE_UNAVAILABLE),
            (StatusCode::GATEWAY_TIMEOUT, REASON_DEADLINE),
        ] {
            let relayed = Response::builder()
                .status(status)
                .body(axum::body::Body::empty())
                .expect("a response");
            assert_eq!(error_reason_for_response(&relayed), reason, "{status}");
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
