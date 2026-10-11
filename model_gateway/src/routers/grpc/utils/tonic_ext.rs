//! Extension traits for tonic gRPC types.

use axum::response::Response;
use http::StatusCode;
use smg_grpc_client::WorkerMediaFault;
use tonic::Code;

use crate::routers::error;

/// Extension methods for `tonic::Status`.
pub(crate) trait TonicStatusExt {
    /// Map gRPC status code to the corresponding HTTP status code. A failure
    /// of the request's media (see [`TonicStatusExt::media_fault`]) is
    /// answered for the media: the request's own fault 400, the media host's
    /// 502.
    fn http_status(&self) -> StatusCode;

    /// Convert this gRPC error into an HTTP error response with the appropriate status code.
    fn to_http_error(&self, code: &str, msg: String) -> Response;

    /// The worker's report that the request's media failed, when the status
    /// carries one: the `smg-media-fault` trailer, or, from a worker that
    /// predates the trailer, the media connector's own words on
    /// `UNAVAILABLE`.
    fn media_fault(&self) -> Option<WorkerMediaFault>;

    /// The HTTP status the circuit breaker samples for this failure; `None`
    /// for a failure of the request's media, which says nothing about the
    /// worker that fetched it.
    fn breaker_status(&self) -> Option<u16>;
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
        match self.media_fault() {
            Some(WorkerMediaFault::Client) => return StatusCode::BAD_REQUEST,
            Some(WorkerMediaFault::Transient) => return StatusCode::BAD_GATEWAY,
            None => {}
        }
        match self.code() {
            Code::Ok => StatusCode::OK,
            Code::InvalidArgument
            | Code::FailedPrecondition
            | Code::OutOfRange
            | Code::Cancelled => StatusCode::BAD_REQUEST,
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
        error::create_error(self.http_status(), code, msg)
    }

    fn media_fault(&self) -> Option<WorkerMediaFault> {
        WorkerMediaFault::from_status(self).or_else(|| {
            (self.code() == Code::Unavailable && names_a_fetch_failure(self.message()))
                .then_some(WorkerMediaFault::Transient)
        })
    }

    fn breaker_status(&self) -> Option<u16> {
        match self.media_fault() {
            Some(_) => None,
            None => Some(self.http_status().as_u16()),
        }
    }
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
    /// media (the request's own fault 400, the media host's 502) and is no
    /// circuit-breaker sample, whichever code carries it.
    #[test]
    fn a_media_fault_answers_for_the_media_and_is_no_breaker_sample() {
        let client = WorkerMediaFault::Client.stamp(tonic::Status::invalid_argument(
            "HTTP error while fetching media: HTTP status client error (404 Not Found) for \
             url (https://media.example/missing.png)",
        ));
        assert_eq!(client.media_fault(), Some(WorkerMediaFault::Client));
        assert_eq!(client.http_status(), StatusCode::BAD_REQUEST);
        assert_eq!(client.breaker_status(), None);

        let transient = WorkerMediaFault::Transient.stamp(tonic::Status::unavailable(
            "media fetch of https://media.example/a.png timed out after 10s",
        ));
        assert_eq!(transient.media_fault(), Some(WorkerMediaFault::Transient));
        assert_eq!(transient.http_status(), StatusCode::BAD_GATEWAY);
        let result: Result<(), tonic::Status> = Err(transient);
        assert_eq!(result.breaker_status(), None);
    }

    /// A worker that predates the trailer reports a fetch failure on
    /// `UNAVAILABLE` in the connector's words; it is read as the media host's
    /// fault. The worker's own trouble, and a caller fault the worker already
    /// answers 4xx, are read as before.
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
