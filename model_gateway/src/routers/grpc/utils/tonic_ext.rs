//! Extension traits for tonic gRPC types.

use axum::response::Response;
use http::StatusCode;
use llm_multimodal::MediaErrorCode;
use smg_grpc_client::{
    dial_failure_without_peer, is_constraint_failure, is_constraint_refusal, is_keep_alive_timeout,
    is_request_timeout, is_transport_failure, WorkerMediaFault,
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

/// The code the public API answers when the worker refused the request's
/// constraint as one its engine cannot enforce (see
/// [`TonicStatusExt::constraint_refusal_response`]).
pub(crate) const UNSUPPORTED_CONSTRAINT_CODE: &str = "unsupported_constraint";

/// The client's words for a media fetch that failed for the reference's own
/// reason (see [`TonicStatusExt::media_message`]). They name no single
/// cause: the fault holds every way a reference fails on its own account.
const MEDIA_FETCH_FAILED_CLIENT: &str =
    "The media could not be fetched from the reference the request gives: check the reference and what it leads to";

/// The client's words for a media fetch whose host has trouble that may pass
/// (see [`TonicStatusExt::media_message`]).
const MEDIA_FETCH_FAILED_TRANSIENT: &str =
    "The media could not be fetched from its URL for now: its host did not answer in time or failed; a retry may succeed";

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

    /// Whether this status is no answer of the worker's: its connection
    /// failed under the call, its side ended the call, the dial failed, or
    /// it answered no keep-alive ping.
    fn is_no_answer(&self) -> bool;

    /// This status's message as an answer to a client carries it: the
    /// message itself, except for a dial that failed. That status names the
    /// worker's address, for the gateway's own log line, and a client reads
    /// the same failure without it.
    fn client_message(&self) -> &str;

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
    /// `UNAVAILABLE`; or the gRPC framing's own refusal of a message the
    /// request's media made longer than its four-byte length prefix carries
    /// (`RESOURCE_EXHAUSTED` in tonic's words, [`names_the_frame_cap`], or
    /// `OUT_OF_RANGE` under a lower configured limit): the request's own
    /// size, whichever side's encoder met it.
    fn media_fault(&self) -> Option<WorkerMediaFault>;

    /// The code the public API answers for a failure of the request's media
    /// (see [`TonicStatusExt::media_fault`]): the one the worker names in its
    /// `smg-media-code` trailer (`invalid_multimodal_request` for media refused
    /// as the request sent it, `media_too_large` for a size refusal),
    /// `media_too_large` for the framing's refusal of the message's size, or
    /// `media_fetch_failed` from a worker that names none or one this router
    /// does not know.
    fn media_code(&self) -> MediaErrorCode;

    /// The client's words for a failure of the request's media (see
    /// [`TonicStatusExt::media_fault`]). A fetch that failed gets the
    /// gateway's sentence for whose fault it is; the worker's words for it
    /// stay on the gateway's log line. The sentence stands only where the
    /// status names a fetch: its `smg-media-code` trailer reads
    /// `media_fetch_failed`, or it carries no code trailer and its message
    /// is in the media connector's fetch words ([`names_a_fetch_failure`]).
    /// Every other media failure keeps the worker's own words, which
    /// describe the request's payload: media refused as the request sent
    /// it, a size refusal, a status that names the fault alone in other
    /// words (a worker that predates the code trailer), and a code this
    /// router does not know.
    fn media_message(&self, fault: WorkerMediaFault) -> String;

    /// The HTTP status the circuit breaker samples for this failure; `None`
    /// for a failure of the request's media, which says nothing about the
    /// worker that fetched it, and for a constraint the engine could not
    /// continue the request under, or one the worker refused as
    /// unenforceable, which say nothing about the worker
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

    /// The client's answer when the status is a worker's refusal of a
    /// constraint its engine cannot enforce (the `smg-constraint-refused`
    /// trailer, see `smg_grpc_client::constraint_refused_status`): a 400
    /// under [`UNSUPPORTED_CONSTRAINT_CODE`] with the worker's words, which
    /// name the keyword and where it stands, terminal for the retry layer.
    /// The request's own schema is refused, not the worker that read it, so
    /// the answer carries no side's code or words (on a disaggregated pair,
    /// not `prefill_worker_failed_to_start`): every worker of the kind
    /// refuses it alike. `None` for any other status.
    fn constraint_refusal_response(&self) -> Option<Response>;
}

/// Whether a status message is the media connector's account of a fetch that
/// failed: a connection that could not be made or broke, or a fetch that ran
/// out of its budget. A worker that predates the `smg-media-fault` trailer
/// reports these on `UNAVAILABLE`, in the connector's words. A worker that
/// predates the `smg-media-code` trailer names the fault alone, and these
/// words tell its fetch failures from its other refusals of the media.
fn names_a_fetch_failure(message: &str) -> bool {
    message.contains("HTTP error while fetching media")
        || (message.contains("media fetch") && message.contains("timed out after"))
}

/// Whether a status message is the gRPC framing's refusal of a message
/// longer than its four-byte length prefix carries: tonic's encoder, on
/// whichever side encodes the message (this gateway's own client when the
/// request's pixel tensors ride in the Generate request, a worker's when its
/// own hop does), answers `RESOURCE_EXHAUSTED` in these words. The size is
/// the request's own, which no replay changes. The words are tonic 0.14's
/// `codec::encode::finish_encoding`, private wording a tonic bump re-checks
/// here: the tests build the status by hand, since pinning the words means
/// encoding a message over 4 GiB.
fn names_the_frame_cap(message: &str) -> bool {
    message.contains("Cannot return body with more than 4GB")
}

/// Whether a status message is the same encoder's refusal under a configured
/// limit below the framing's (`max_encoding_message_size`; this gateway sets
/// none, an operator's build may), answered `OUT_OF_RANGE` in these words,
/// from the same function as [`names_the_frame_cap`]'s.
fn names_the_encoding_limit(message: &str) -> bool {
    message.contains("encoded message length too large")
}

/// Whether `status` is the encoder's refusal of the message's size, at the
/// framing's cap or at a configured limit under it (see
/// [`names_the_frame_cap`] and [`names_the_encoding_limit`]).
fn is_frame_cap_refusal(status: &tonic::Status) -> bool {
    match status.code() {
        Code::ResourceExhausted => names_the_frame_cap(status.message()),
        Code::OutOfRange => names_the_encoding_limit(status.message()),
        _ => false,
    }
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
        // The gateway's own request timeout (`--grpc-request-timeout-secs`): the
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

    fn is_no_answer(&self) -> bool {
        is_transport_failure(self)
            || self.code() == Code::Cancelled
            || dial_failure_without_peer(self).is_some()
            || is_keep_alive_timeout(self)
    }

    fn client_message(&self) -> &str {
        dial_failure_without_peer(self).unwrap_or_else(|| self.message())
    }

    fn error_reason(&self) -> &'static str {
        use metrics_labels::*;
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
        WorkerMediaFault::from_status(self)
            .or_else(|| {
                (self.code() == Code::Unavailable && names_a_fetch_failure(self.message()))
                    .then_some(WorkerMediaFault::Transient)
            })
            .or_else(|| is_frame_cap_refusal(self).then_some(WorkerMediaFault::Client))
    }

    fn media_code(&self) -> MediaErrorCode {
        smg_grpc_client::media_code(self)
            .and_then(MediaErrorCode::parse)
            .unwrap_or_else(|| {
                if is_frame_cap_refusal(self) {
                    MediaErrorCode::TooLarge
                } else {
                    MediaErrorCode::FetchFailed
                }
            })
    }

    fn media_message(&self, fault: WorkerMediaFault) -> String {
        let names_a_fetch = match smg_grpc_client::media_code(self) {
            Some(code) => MediaErrorCode::parse(code) == Some(MediaErrorCode::FetchFailed),
            // No code trailer: only the connector's own fetch words say that
            // the failure was a fetch.
            None => names_a_fetch_failure(self.message()),
        };
        if !names_a_fetch {
            return self.message().to_string();
        }
        match fault {
            WorkerMediaFault::Client => MEDIA_FETCH_FAILED_CLIENT,
            WorkerMediaFault::Transient => MEDIA_FETCH_FAILED_TRANSIENT,
        }
        .to_string()
    }

    fn breaker_status(&self) -> Option<u16> {
        if is_constraint_failure(self)
            || is_constraint_refusal(self)
            || self.media_fault().is_some()
        {
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

    fn constraint_refusal_response(&self) -> Option<Response> {
        if !is_constraint_refusal(self) {
            return None;
        }
        let mut response =
            self.to_http_error(UNSUPPORTED_CONSTRAINT_CODE, self.message().to_string());
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
    use smg_worker::health::resilience::{
        DEFAULT_CAPACITY_STATUS_CODES, DEFAULT_RETRYABLE_STATUS_CODES,
    };

    use super::*;

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

    /// A worker's refusal of a constraint its engine cannot enforce is the
    /// request's 400 under its own code with the worker's words, terminal for
    /// the retry layer and sampled by no breaker; an `INVALID_ARGUMENT`
    /// without the trailer is no refusal of the kind.
    #[test]
    fn a_constraint_refusal_is_a_terminal_400_the_breaker_does_not_sample() {
        use smg_grpc_client::constraint_refused_status;

        use crate::routers::{common::retry::NonRetryable, error};

        let status = constraint_refused_status("The JSON schema constraint cannot be enforced");
        assert_eq!(status.http_status(), StatusCode::BAD_REQUEST);
        let result: Result<(), tonic::Status> = Err(status.clone());
        assert_eq!(result.breaker_status(), None);
        let response = status
            .constraint_refusal_response()
            .expect("the refusal is answered");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error::extract_error_code_from_response(&response),
            UNSUPPORTED_CONSTRAINT_CODE
        );
        // A 400 is not replayed by status alone; the refusal is marked
        // terminal whatever statuses the operator makes retryable.
        assert!(response.extensions().get::<NonRetryable>().is_some());
        let plain =
            tonic::Status::invalid_argument("The JSON schema constraint cannot be enforced");
        assert!(plain.constraint_refusal_response().is_none());
        assert_eq!(plain.breaker_status(), Some(400));
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

    /// A status that is no answer of the worker's: an HTTP/2 fault under
    /// the call, the worker's side cancelling, a dial that failed, a
    /// keep-alive timeout. A status the worker answered with is one.
    #[tokio::test]
    async fn a_status_says_whether_the_worker_answered() {
        use smg_grpc_client::h2::{Error, Reason};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let closed = format!("grpc://{}", listener.local_addr().expect("its address"));
        drop(listener);
        let Err(refused) = smg_grpc_client::connect_channel(&closed).await else {
            panic!("nothing listens on {closed}");
        };
        for (status, no_answer) in [
            (
                tonic::Status::from_error(Box::new(Error::from(Reason::PROTOCOL_ERROR))),
                true,
            ),
            (tonic::Status::cancelled("operation was canceled"), true),
            (tonic::Status::from_error(refused), true),
            (smg_grpc_client::keep_alive_timeout_status(), true),
            (tonic::Status::internal("the servicer failed"), false),
            (tonic::Status::unavailable("draining"), false),
            (tonic::Status::unimplemented("no"), false),
            (tonic::Status::deadline_exceeded("too slow"), false),
            (tonic::Status::invalid_argument("bad"), false),
            (tonic::Status::resource_exhausted("kv cache full"), false),
        ] {
            assert_eq!(status.is_no_answer(), no_answer, "{status:?}");
        }
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
        // The client reads the gateway's sentence for a fetch that failed,
        // by whose fault it is; the worker's own words stay on the log line.
        assert_eq!(
            client.media_message(WorkerMediaFault::Client),
            MEDIA_FETCH_FAILED_CLIENT
        );

        let transient = WorkerMediaFault::Transient.stamp(tonic::Status::unavailable(
            "media fetch of https://media.example/a.png timed out after 10s",
        ));
        assert_eq!(transient.media_fault(), Some(WorkerMediaFault::Transient));
        assert_eq!(transient.http_status(), StatusCode::BAD_GATEWAY);
        assert_eq!(transient.media_code(), MediaErrorCode::FetchFailed);
        assert_eq!(
            transient.media_message(WorkerMediaFault::Transient),
            MEDIA_FETCH_FAILED_TRANSIENT
        );
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
        // A size refusal and media refused as sent describe the request's
        // payload: the worker's words reach the client.
        assert_eq!(
            too_large.media_message(WorkerMediaFault::Client),
            too_large.message()
        );

        let refused = smg_grpc_client::stamp_media_code(
            WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument(
                "data URL parse error: missing comma in data url",
            )),
            "invalid_multimodal_request",
        );
        assert_eq!(refused.media_fault(), Some(WorkerMediaFault::Client));
        assert_eq!(refused.http_status(), StatusCode::BAD_REQUEST);
        assert_eq!(refused.media_code(), MediaErrorCode::InvalidRequest);
        assert_eq!(refused.breaker_status(), None);
        assert_eq!(
            refused.media_message(WorkerMediaFault::Client),
            "data URL parse error: missing comma in data url"
        );

        // A fetch the worker names by its code reads the sentence, whatever
        // words the worker has for it.
        let named = smg_grpc_client::stamp_media_code(
            WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument(
                "the reference could not be followed",
            )),
            "media_fetch_failed",
        );
        assert_eq!(named.media_code(), MediaErrorCode::FetchFailed);
        assert_eq!(
            named.media_message(WorkerMediaFault::Client),
            MEDIA_FETCH_FAILED_CLIENT
        );

        // The gRPC framing's refusal of a message longer than its four-byte
        // length prefix carries (tonic's words; this gateway's own client
        // when the request's pixel tensors ride in the Generate request): the
        // request's own size, a 400 under `media_too_large` and no breaker
        // sample, where a RESOURCE_EXHAUSTED otherwise reads as the worker's
        // capacity pushback, a 429 the retry layer replays.
        let framing = tonic::Status::resource_exhausted(
            "Cannot return body with more than 4GB of data but got 5065026759 bytes",
        );
        assert_eq!(framing.media_fault(), Some(WorkerMediaFault::Client));
        assert_eq!(framing.http_status(), StatusCode::BAD_REQUEST);
        assert_eq!(framing.media_code(), MediaErrorCode::TooLarge);
        assert_eq!(framing.breaker_status(), None);
        // The framing's refusal names no fetch: its own words reach the client.
        assert_eq!(
            framing.media_message(WorkerMediaFault::Client),
            framing.message()
        );
        let pushback = tonic::Status::resource_exhausted("the engine is at capacity");
        assert_eq!(pushback.media_fault(), None);
        assert_eq!(pushback.http_status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(pushback.breaker_status(), Some(429));
        // The same encoder's refusal under a configured limit below the
        // framing's cap (`max_encoding_message_size`) reads the same way;
        // any other OUT_OF_RANGE stays the plain 400 it was.
        let limited = tonic::Status::out_of_range(
            "Error, encoded message length too large: found 5065026759 bytes, the limit is: 4194304 bytes",
        );
        assert_eq!(limited.media_fault(), Some(WorkerMediaFault::Client));
        assert_eq!(limited.http_status(), StatusCode::BAD_REQUEST);
        assert_eq!(limited.media_code(), MediaErrorCode::TooLarge);
        assert_eq!(limited.breaker_status(), None);
        let plain_range = tonic::Status::out_of_range("position 9 is past the end");
        assert_eq!(plain_range.media_fault(), None);
        assert_eq!(plain_range.http_status(), StatusCode::BAD_REQUEST);

        let unknown = smg_grpc_client::stamp_media_code(
            WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument("new words")),
            "media_from_the_future",
        );
        assert_eq!(unknown.media_code(), MediaErrorCode::FetchFailed);
        // A code this router does not know names no fetch: the worker's words
        // reach the client under the fallback code.
        assert_eq!(unknown.media_message(WorkerMediaFault::Client), "new words");
    }

    /// The gateway's sentence stands only for a status that names a fetch. A
    /// worker that predates the code trailer names the fault alone: where its
    /// words describe the request's payload they reach the client, as the
    /// same words do under a code.
    #[test]
    fn a_fault_without_a_code_keeps_the_words_that_name_no_fetch() {
        for message in [
            "data URL parse error: missing comma in data url",
            "image decode error: The image format could not be determined",
            "video input payload exceeds the maximum size of 536870912 bytes",
        ] {
            let payload = WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument(message));
            assert_eq!(payload.media_fault(), Some(WorkerMediaFault::Client));
            assert_eq!(payload.http_status(), StatusCode::BAD_REQUEST);
            assert_eq!(payload.breaker_status(), None);
            assert_eq!(
                payload.media_message(WorkerMediaFault::Client),
                message,
                "{message}"
            );
        }
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
            assert_eq!(
                legacy.media_message(WorkerMediaFault::Transient),
                MEDIA_FETCH_FAILED_TRANSIENT
            );
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
