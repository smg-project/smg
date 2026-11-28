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
    /// which tonic has no code for. A worker's answer above the receive
    /// bound ([`is_receive_bound_refusal`]) is a bad gateway too: an answer
    /// the gateway could not read, where any other `OUT_OF_RANGE` is the
    /// request's fault.
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
    /// worker's side cancelling), a worker's answer above the receive bound
    /// ([`is_receive_bound_refusal`], a bad response), then the code the
    /// worker answered: its deadline, its capacity pushback, its
    /// unavailability, the request's own fault, the engine's own report
    /// that the request failed, or a failure inside the worker.
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
    /// either, and for a worker's answer above the receive bound, which the
    /// worker gave rightly ([`is_receive_bound_refusal`]).
    fn breaker_status(&self) -> Option<u16>;

    /// What Grand Central is told where this failure ends a request
    /// (`RequestEnd::report_with`): the status the circuit breaker samples
    /// for it ([`TonicStatusExt::breaker_status`]), and whether the worker
    /// itself answered it ([`TonicStatusExt::is_no_answer`]). `None` where
    /// the breaker takes no sample: the failure says nothing about the
    /// worker, and nobody reports it.
    fn end_status(&self) -> Option<(u16, bool)>;

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

/// Whether `status` is a decoder's refusal of a message above its receive
/// bound (`max_decoding_message_size`, tonic's default where none is set):
/// `OUT_OF_RANGE` in the words of tonic 0.14's `codec::decode`, private
/// wording a tonic bump re-checks here. The gateway's own client answers so
/// for a worker's message it cannot read: the request was valid, the worker
/// did nothing wrong, and another worker's answer would be as large.
pub(crate) fn is_receive_bound_refusal(status: &tonic::Status) -> bool {
    status.code() == Code::OutOfRange
        && status
            .message()
            .contains("decoded message length too large")
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
        // The worker's answer was above the receive bound: an answer the
        // gateway could not read, not a fault of the request.
        if is_receive_bound_refusal(self) {
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
        // An answer above the receive bound is as large from any worker:
        // final for the retry layer.
        if is_receive_bound_refusal(self) {
            mark_non_retryable(&mut response);
        }
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
        if is_receive_bound_refusal(self) {
            return REASON_BAD_RESPONSE;
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
            || is_receive_bound_refusal(self)
        {
            return None;
        }
        Some(self.http_status().as_u16())
    }

    fn end_status(&self) -> Option<(u16, bool)> {
        self.breaker_status()
            .map(|status| (status, !self.is_no_answer()))
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
#[path = "../../../../tests/unit/routers/grpc/utils/tonic_ext/tests.rs"]
mod tests;
