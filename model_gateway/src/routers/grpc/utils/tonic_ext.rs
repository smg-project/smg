//! Extension traits for tonic gRPC types.

use axum::response::Response;
use http::StatusCode;
use llm_multimodal::MediaErrorCode;
use smg_grpc_client::{
    is_constraint_failure, is_request_timeout, is_transport_failure, request_timeout_kind,
    TimeoutKind, WorkerMediaFault,
};
use tonic::Code;

use super::metrics::mark_error_reason;
use crate::{
    observability::metrics::metrics_labels,
    routers::{
        common::retry::{mark_non_retryable, request_timeout_response},
        error,
        grpc::proto_wrapper::ENGINE_ERROR_MESSAGE,
    },
};

/// The code the public API answers when the engine could not continue the
/// request under the request's own constraint (see
/// [`TonicStatusExt::constraint_failure_response`]).
pub(crate) const CONSTRAINT_FAILED_CODE: &str = "constraint_failed";

/// Extension methods for `tonic::Status`.
pub(crate) trait TonicStatusExt {
    /// Map gRPC status code to the corresponding HTTP status code. A failure
    /// of the request's media (see [`TonicStatusExt::media_fault`]) is
    /// answered for the media: a 400 for a reference the request got wrong,
    /// a 502 for an origin that does not deliver within the fetch budget,
    /// answers 5xx or breaks the transfer, so the client knows whether to
    /// retry; the worker's own trouble with the media stays the worker's 5xx
    /// (it carries no media fault). A
    /// `CANCELLED` the worker answered is the worker's side ending the
    /// call (its connection closed under it, its server reset the stream):
    /// a bad gateway, never the request's fault. So is a status tonic made
    /// of an HTTP/2 fault under the call (the connection went away, a stream
    /// reset below gRPC, a protocol fault), whatever code tonic picked from
    /// the frame's reason, and one made of the socket under the call
    /// failing (the connection reset or closed with the call in flight),
    /// which tonic has no code for.
    fn http_status(&self) -> StatusCode;

    /// Convert this gRPC error into an HTTP error response with the appropriate
    /// status code. The response carries the status's
    /// [`error_reason`](TonicStatusExt::error_reason), so the router that
    /// counts it reads the cause the status named (see
    /// `error_reason_for_response`).
    fn to_http_error(&self, code: &str, msg: String) -> Response;

    /// The `reason` label of a router error this status ends a request with:
    /// the cause behind the class [`http_status`](TonicStatusExt::http_status)
    /// maps it to, one of the `metrics_labels::REASON_*` values. The same
    /// predicates, in the same order: a constraint the engine could not
    /// continue under, a fault of the request's media, the worker's
    /// connection failing under the call (an HTTP/2 or socket fault, or the
    /// worker's side cancelling), then the code the worker answered: its
    /// deadline, its capacity pushback, its unavailability, the request's
    /// own fault, the engine's own report that the request failed, or a
    /// failure inside the worker.
    fn error_reason(&self) -> &'static str;

    /// The worker's report that the request's media failed, when the status
    /// carries one: the `smg-media-fault` trailer, or, from a worker that
    /// predates the trailer, the media connector's own words on
    /// `UNAVAILABLE`.
    fn media_fault(&self) -> Option<WorkerMediaFault>;

    /// The code the public API answers for a failure of the request's media
    /// (see [`TonicStatusExt::media_fault`]): the one the worker names in its
    /// `smg-media-code` trailer (`media_too_large` for a size refusal), or
    /// `media_fetch_failed` from a worker that names none or one this router
    /// does not know.
    fn media_code(&self) -> MediaErrorCode;

    /// The HTTP status the circuit breaker samples for this failure; `None`
    /// for a failure of the request's media, which says nothing about the
    /// worker that fetched it, and for a constraint the engine could not
    /// continue the request under, which says nothing about the worker
    /// either.
    fn breaker_status(&self) -> Option<u16>;

    /// The client's answer when the status is a worker's report that the
    /// engine could not continue the request under the request's own
    /// constraint (the `smg-constraint-failed` trailer, see
    /// `smg_grpc_client::constraint_failure`): a 422 under
    /// [`CONSTRAINT_FAILED_CODE`] with the worker's words, terminal for the
    /// retry layer. The request was well formed and the engine admitted it,
    /// so it is not a 400; the same tokens fail the same way on a replay, on
    /// this worker or another of its kind, each replay a prefill, so it is
    /// not a transient 5xx either. `None` for any other status.
    fn constraint_failure_response(&self) -> Option<Response>;
}

/// Whether a status message is the media connector's account of a fetch that
/// failed: a connection that could not be made or broke, or a fetch that ran
/// out of its budget. A worker that predates the `smg-media-fault` trailer
/// reports these on `UNAVAILABLE`, in the connector's words.
fn names_a_fetch_failure(message: &str) -> bool {
    message.contains("HTTP error while fetching media")
        || (message.contains("media fetch") && message.contains("timed out after"))
}

impl TonicStatusExt for tonic::Status {
    fn http_status(&self) -> StatusCode {
        if is_constraint_failure(self) {
            return StatusCode::UNPROCESSABLE_ENTITY;
        }
        // The request's media failed, not the worker (the trailer says whose
        // fault): the request's own reference is a 400, its host's trouble for
        // now a 502, so a client knows whether to retry.
        if let Some(fault) = self.media_fault() {
            return match fault {
                WorkerMediaFault::Client => StatusCode::BAD_REQUEST,
                WorkerMediaFault::Transient => StatusCode::BAD_GATEWAY,
            };
        }
        // tonic makes a status of a failure in the HTTP/2 layer under the
        // call (the connection went away: a GOAWAY of either side, the one
        // the gateway's own client issues over too many locally-reset
        // streams included; a stream reset below gRPC; a protocol fault) and
        // picks its code from the frame's reason: ENHANCE_YOUR_CALM reads as
        // RESOURCE_EXHAUSTED, which would answer the client a 429 rate limit
        // it never hit, PROTOCOL_ERROR as INTERNAL, the code of a worker's
        // own answer; a failure of the socket under the call (a reset, a
        // close with the call in flight) gets UNKNOWN, no worker's code at
        // all. The worker's connection failed under the call: a 502 the
        // breaker samples and the retry layer replays elsewhere.
        if is_transport_failure(self) {
            return StatusCode::BAD_GATEWAY;
        }
        match self.code() {
            Code::Ok => StatusCode::OK,
            Code::InvalidArgument | Code::FailedPrecondition | Code::OutOfRange => {
                StatusCode::BAD_REQUEST
            }
            // The worker's side ended the call: what a connection that closed
            // under it or a server that reset the stream produce, nothing the
            // request did (the liveness monitor reads the code the same way).
            Code::Cancelled => StatusCode::BAD_GATEWAY,
            Code::Unauthenticated => StatusCode::UNAUTHORIZED,
            Code::PermissionDenied => StatusCode::FORBIDDEN,
            Code::NotFound => StatusCode::NOT_FOUND,
            Code::AlreadyExists | Code::Aborted => StatusCode::CONFLICT,
            Code::ResourceExhausted => StatusCode::TOO_MANY_REQUESTS,
            Code::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            Code::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
            Code::Unimplemented => StatusCode::NOT_IMPLEMENTED,
            // Internal, Unknown, DataLoss
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn to_http_error(&self, code: &str, msg: String) -> Response {
        // The gateway's own request timeout (`--request-timeout-secs`): the
        // answer every path gives an upstream call its request timeout cut
        // (the 504 under `call_upstream_timeout`, terminal for the retry
        // layer: a replay would hold the client for a second timeout),
        // whatever the caller's code.
        let mut response = if is_request_timeout(self) {
            request_timeout_response(msg)
        } else {
            error::create_error(self.http_status(), code, msg)
        };
        mark_error_reason(&mut response, self.error_reason());
        response
    }

    fn error_reason(&self) -> &'static str {
        use metrics_labels::*;
        // The gateway's own bounds on the call, by kind: the total reads as
        // any deadline, the two silence bounds under their own names.
        if let Some(kind) = request_timeout_kind(self) {
            return match kind {
                TimeoutKind::Total => REASON_DEADLINE,
                TimeoutKind::FirstToken => REASON_FIRST_TOKEN_TIMEOUT,
                TimeoutKind::InterToken => REASON_INTER_TOKEN_TIMEOUT,
            };
        }
        if is_constraint_failure(self) {
            return REASON_CONSTRAINT_FAILED;
        }
        if self.media_fault().is_some() {
            return REASON_MEDIA_FAULT;
        }
        if is_transport_failure(self) {
            return REASON_TRANSPORT_DISCONNECT;
        }
        match self.code() {
            Code::Cancelled => REASON_TRANSPORT_DISCONNECT,
            Code::DeadlineExceeded => REASON_DEADLINE,
            Code::ResourceExhausted => REASON_OVERLOADED,
            Code::Unavailable => REASON_ENGINE_UNAVAILABLE,
            Code::InvalidArgument | Code::FailedPrecondition | Code::OutOfRange => {
                REASON_INVALID_REQUEST
            }
            Code::Internal if is_the_engines_own_report(self) => REASON_ENGINE_ERROR,
            Code::Internal | Code::Unknown | Code::DataLoss => REASON_ENGINE_INTERNAL,
            _ => REASON_OTHER,
        }
    }

    fn media_fault(&self) -> Option<WorkerMediaFault> {
        WorkerMediaFault::from_status(self).or_else(|| {
            (self.code() == Code::Unavailable && names_a_fetch_failure(self.message()))
                .then_some(WorkerMediaFault::Transient)
        })
    }

    fn media_code(&self) -> MediaErrorCode {
        smg_grpc_client::media_code(self)
            .and_then(MediaErrorCode::parse)
            .unwrap_or(MediaErrorCode::FetchFailed)
    }

    fn breaker_status(&self) -> Option<u16> {
        if is_constraint_failure(self) || self.media_fault().is_some() {
            return None;
        }
        Some(self.http_status().as_u16())
    }

    fn constraint_failure_response(&self) -> Option<Response> {
        if !is_constraint_failure(self) {
            return None;
        }
        let mut response = self.to_http_error(CONSTRAINT_FAILED_CODE, self.message().to_string());
        mark_non_retryable(&mut response);
        Some(response)
    }
}

/// Whether `status` is the engine's own report that the request failed (its
/// `Complete` had finish reason `error`; the ZMQ lane's adapters end their
/// streams with the same words), as opposed to a failed call to the worker.
/// The one status that carries no reason of the engine's.
pub(crate) fn is_the_engines_own_report(status: &tonic::Status) -> bool {
    status.code() == Code::Internal && status.message() == ENGINE_ERROR_MESSAGE
}

/// Extension for `Result<T, tonic::Status>`: what the circuit breaker samples.
pub(crate) trait TonicResultExt {
    /// The HTTP status the circuit breaker samples for this outcome: `Ok` →
    /// 200, `Err(status)` → its mapped HTTP status, or `None` for a failure of
    /// the request's media (see [`TonicStatusExt::breaker_status`]).
    fn breaker_status(&self) -> Option<u16>;
}

impl<T> TonicResultExt for Result<T, tonic::Status> {
    fn breaker_status(&self) -> Option<u16> {
        match self {
            Ok(_) => Some(200),
            Err(status) => status.breaker_status(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::resilience::{
        DEFAULT_CAPACITY_STATUS_CODES, DEFAULT_RETRYABLE_STATUS_CODES,
    };

    /// A status tonic made of an HTTP/2 fault under the call (the h2 error in
    /// its source chain) is the worker's connection failing, whatever code
    /// tonic picked from the frame's reason: a 502 the breaker samples and
    /// the retry layer replays elsewhere, never the 429 a GOAWAY
    /// ENHANCE_YOUR_CALM reads as (a rate limit the client never hit) nor the
    /// 500 of a worker's own answer. The worker's own RESOURCE_EXHAUSTED, in
    /// its trailers, stays capacity pushback.
    #[test]
    fn an_http2_fault_under_the_call_is_a_retryable_502_whatever_its_code() {
        use smg_grpc_client::h2::{Error, Reason};

        for (reason, code) in [
            (Reason::ENHANCE_YOUR_CALM, Code::ResourceExhausted),
            (Reason::PROTOCOL_ERROR, Code::Internal),
            (Reason::FLOW_CONTROL_ERROR, Code::Internal),
            (Reason::REFUSED_STREAM, Code::Unavailable),
            (Reason::CANCEL, Code::Cancelled),
        ] {
            let status = tonic::Status::from_error(Box::new(Error::from(reason)));
            assert_eq!(status.code(), code, "tonic's own code for {reason:?}");
            assert_eq!(status.http_status(), StatusCode::BAD_GATEWAY, "{reason:?}");
            let result: Result<(), tonic::Status> = Err(status);
            assert_eq!(result.breaker_status(), Some(502), "{reason:?}");
        }
        assert!(DEFAULT_RETRYABLE_STATUS_CODES.contains(&502));
        assert_eq!(
            tonic::Status::resource_exhausted("busy").http_status(),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    /// A worker's report that the engine could not continue the request under
    /// its constraint is answered for the constraint: a 422 under its own
    /// code with the worker's words, terminal for the retry layer, sampled by
    /// no breaker; the same code without the report keeps its 400.
    #[test]
    fn a_constraint_failure_is_a_terminal_422_the_breaker_does_not_sample() {
        use smg_grpc_client::constraint_failed_status;

        use crate::routers::{common::retry::is_retryable_response, error};

        let status = constraint_failed_status("the scheduler aborted the request: Aborted");
        assert_eq!(status.http_status(), StatusCode::UNPROCESSABLE_ENTITY);
        let result: Result<(), tonic::Status> = Err(status.clone());
        assert_eq!(result.breaker_status(), None);
        let response = status
            .constraint_failure_response()
            .expect("the report is answered");
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            CONSTRAINT_FAILED_CODE
        );
        assert!(!is_retryable_response(&response));
        let plain = tonic::Status::failed_precondition("the scheduler aborted the request");
        assert_eq!(plain.http_status(), StatusCode::BAD_REQUEST);
        assert!(plain.constraint_failure_response().is_none());
    }

    /// The reason a status names for the router error family, by the same
    /// predicates that class it: the worker's connection failing under the
    /// call (an HTTP/2 fault of any code, a socket fault, the worker's side
    /// cancelling), its deadline, its capacity pushback, its unavailability,
    /// the request's own fault, the engine's own report, a failure inside
    /// the worker, a constraint, the request's media; a code outside them is
    /// `other`.
    #[test]
    fn a_status_names_the_reason_behind_its_class() {
        use metrics_labels::*;
        use smg_grpc_client::{constraint_failed_status, h2::Reason};

        for reason in [
            Reason::ENHANCE_YOUR_CALM,
            Reason::PROTOCOL_ERROR,
            Reason::REFUSED_STREAM,
            Reason::CANCEL,
        ] {
            let status =
                tonic::Status::from_error(Box::new(smg_grpc_client::h2::Error::from(reason)));
            assert_eq!(
                status.error_reason(),
                REASON_TRANSPORT_DISCONNECT,
                "{reason:?}"
            );
        }
        for (status, reason) in [
            (
                tonic::Status::cancelled("operation was canceled"),
                REASON_TRANSPORT_DISCONNECT,
            ),
            (
                tonic::Status::deadline_exceeded("too slow"),
                REASON_DEADLINE,
            ),
            // The gateway's own bounds on the call, by kind.
            (
                smg_grpc_client::RequestDeadline::after(std::time::Duration::from_secs(15))
                    .status(TimeoutKind::Total),
                REASON_DEADLINE,
            ),
            (
                smg_grpc_client::RequestDeadline::after(std::time::Duration::from_secs(15))
                    .status(TimeoutKind::FirstToken),
                REASON_FIRST_TOKEN_TIMEOUT,
            ),
            (
                smg_grpc_client::RequestDeadline::after(std::time::Duration::from_secs(15))
                    .status(TimeoutKind::InterToken),
                REASON_INTER_TOKEN_TIMEOUT,
            ),
            (
                tonic::Status::resource_exhausted("kv cache full"),
                REASON_OVERLOADED,
            ),
            (
                tonic::Status::unavailable("draining"),
                REASON_ENGINE_UNAVAILABLE,
            ),
            (
                tonic::Status::invalid_argument("bad"),
                REASON_INVALID_REQUEST,
            ),
            (
                tonic::Status::failed_precondition("bad"),
                REASON_INVALID_REQUEST,
            ),
            (tonic::Status::out_of_range("bad"), REASON_INVALID_REQUEST),
            (
                tonic::Status::internal(ENGINE_ERROR_MESSAGE),
                REASON_ENGINE_ERROR,
            ),
            (
                tonic::Status::internal("the servicer failed"),
                REASON_ENGINE_INTERNAL,
            ),
            (tonic::Status::unknown("no idea"), REASON_ENGINE_INTERNAL),
            (tonic::Status::data_loss("lost"), REASON_ENGINE_INTERNAL),
            (
                constraint_failed_status("the scheduler aborted the request: Aborted"),
                REASON_CONSTRAINT_FAILED,
            ),
            (
                tonic::Status::unavailable(
                    "HTTP error while fetching media from https://example.invalid/a.png",
                ),
                REASON_MEDIA_FAULT,
            ),
            (tonic::Status::not_found("gone"), REASON_OTHER),
            (tonic::Status::unimplemented("no"), REASON_OTHER),
            (tonic::Status::permission_denied("no"), REASON_OTHER),
        ] {
            assert_eq!(status.error_reason(), reason, "{status:?}");
        }
    }

    /// The HTTP answer made of a status carries the status's reason, so the
    /// router that counts the answer (which sees no status) counts the cause
    /// the status named.
    #[test]
    fn the_http_answer_of_a_status_carries_its_reason() {
        use super::super::metrics::ErrorReason;

        let response = tonic::Status::deadline_exceeded("too slow").to_http_error(
            "start_generation_failed",
            "Failed to start generation".to_string(),
        );
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            response.extensions().get::<ErrorReason>().map(|r| r.0),
            Some(metrics_labels::REASON_DEADLINE)
        );
        let plain = error::internal_error("no_response_produced", "No response produced");
        assert!(plain.extensions().get::<ErrorReason>().is_none());
    }

    #[test]
    fn invalid_argument_maps_to_400_and_is_not_a_circuit_breaker_failure() {
        for status in [
            tonic::Status::invalid_argument("bad"),
            tonic::Status::failed_precondition("bad"),
            tonic::Status::out_of_range("bad"),
        ] {
            assert_eq!(status.http_status(), StatusCode::BAD_REQUEST);
            let result: Result<(), tonic::Status> = Err(status);
            assert_eq!(result.breaker_status(), Some(400));
        }
        assert!(!DEFAULT_RETRYABLE_STATUS_CODES.contains(&400));
    }

    /// A call the worker's side cancelled is the worker's failure: a 502 the
    /// retry layer replays elsewhere and the breaker samples, never a 400.
    #[test]
    fn a_cancelled_call_maps_to_502_and_is_a_retryable_circuit_breaker_failure() {
        let status = tonic::Status::cancelled("operation was canceled");
        assert_eq!(status.http_status(), StatusCode::BAD_GATEWAY);
        let result: Result<(), tonic::Status> = Err(status);
        assert_eq!(result.breaker_status(), Some(502));
        assert!(DEFAULT_RETRYABLE_STATUS_CODES.contains(&502));
        assert!(!DEFAULT_RETRYABLE_STATUS_CODES.contains(&400));
    }

    #[test]
    fn internal_and_unknown_map_to_500() {
        for status in [
            tonic::Status::internal("boom"),
            tonic::Status::unknown("boom"),
            tonic::Status::data_loss("boom"),
        ] {
            assert_eq!(status.http_status(), StatusCode::INTERNAL_SERVER_ERROR);
            let result: Result<(), tonic::Status> = Err(status);
            assert_eq!(result.breaker_status(), Some(500));
        }
        assert!(DEFAULT_RETRYABLE_STATUS_CODES.contains(&500));
    }

    #[test]
    fn resource_exhausted_maps_to_429_capacity_pushback() {
        assert_eq!(
            tonic::Status::resource_exhausted("busy").http_status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            tonic::Status::unavailable("down").http_status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert!(DEFAULT_CAPACITY_STATUS_CODES.contains(&429));
        let ok: Result<(), tonic::Status> = Ok(());
        assert_eq!(ok.breaker_status(), Some(200));
    }

    /// A worker's report that the request's media failed is answered for the
    /// media (a 400 for the request's own fault, a 502 for the media host's)
    /// and is no circuit-breaker sample, whichever code carries it. The API's
    /// code is the one the worker names, `media_fetch_failed` when it names
    /// none or one this router does not know.
    #[test]
    fn a_media_fault_answers_for_the_media_and_is_no_breaker_sample() {
        let client = WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument(
            "HTTP error while fetching media: HTTP status client error (404 Not Found) for \
             url (https://media.example/missing.png)",
        ));
        assert_eq!(client.media_fault(), Some(WorkerMediaFault::Client));
        assert_eq!(client.http_status(), StatusCode::BAD_REQUEST);
        assert_eq!(client.media_code(), MediaErrorCode::FetchFailed);
        assert_eq!(client.breaker_status(), None);

        let transient = WorkerMediaFault::Transient.stamp(tonic::Status::unavailable(
            "media fetch of https://media.example/a.png timed out after 10s",
        ));
        assert_eq!(transient.media_fault(), Some(WorkerMediaFault::Transient));
        assert_eq!(transient.http_status(), StatusCode::BAD_GATEWAY);
        assert_eq!(transient.media_code(), MediaErrorCode::FetchFailed);
        let result: Result<(), tonic::Status> = Err(transient);
        assert_eq!(result.breaker_status(), None);

        let too_large = smg_grpc_client::stamp_media_code(
            WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument(
                "decoded video RGB payload would be 2208384000 bytes, exceeding \
                 SMG_VIDEO_MAX_DECODED_BYTES=1073741824",
            )),
            "media_too_large",
        );
        assert_eq!(too_large.media_fault(), Some(WorkerMediaFault::Client));
        assert_eq!(too_large.http_status(), StatusCode::BAD_REQUEST);
        assert_eq!(too_large.media_code(), MediaErrorCode::TooLarge);
        assert_eq!(too_large.breaker_status(), None);

        let unknown = smg_grpc_client::stamp_media_code(
            WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument("new words")),
            "media_from_the_future",
        );
        assert_eq!(unknown.media_code(), MediaErrorCode::FetchFailed);
    }

    /// A worker that predates the trailer reports a fetch failure on
    /// `UNAVAILABLE` in the connector's words; it is read as the media host's
    /// fault and answered for the media (a 502). The worker's own trouble, and
    /// a caller fault the worker already answers 4xx, are read as before.
    #[test]
    fn a_fetch_failure_in_the_connectors_words_is_read_without_the_trailer() {
        for message in [
            "Failed to finalize multimodal tracker: HTTP error while fetching media: error \
             sending request for url (https://media.example/a.png)",
            "Failed to finalize multimodal tracker: media fetch timed out after 10s",
            "Failed to finalize multimodal tracker: media fetch of https://media.example/a.png \
             timed out after 30s",
        ] {
            let legacy = tonic::Status::unavailable(message);
            assert_eq!(
                legacy.media_fault(),
                Some(WorkerMediaFault::Transient),
                "{message}"
            );
            assert_eq!(legacy.http_status(), StatusCode::BAD_GATEWAY);
            assert_eq!(legacy.media_code(), MediaErrorCode::FetchFailed);
            assert_eq!(legacy.breaker_status(), None);
        }
        let sidecar = tonic::Status::unavailable("sidecar_timeout: no result");
        assert_eq!(sidecar.media_fault(), None);
        assert_eq!(sidecar.breaker_status(), Some(503));
        let engine = tonic::Status::unavailable("engine is restarting");
        assert_eq!(engine.breaker_status(), Some(503));
        let decode = tonic::Status::invalid_argument(
            "Failed to finalize multimodal tracker: image decode error: The image format \
             could not be determined",
        );
        assert_eq!(decode.media_fault(), None);
        assert_eq!(decode.breaker_status(), Some(400));
    }
}
