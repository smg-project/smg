//! Extension traits for tonic gRPC types.

use axum::response::Response;
use http::StatusCode;
use tonic::Code;

use crate::routers::error;

/// Extension methods for `tonic::Status`.
pub(crate) trait TonicStatusExt {
    /// Map gRPC status code to the corresponding HTTP status code.
    fn http_status(&self) -> StatusCode;

    /// Convert this gRPC error into an HTTP error response with the appropriate status code.
    fn to_http_error(&self, code: &str, msg: String) -> Response;
}

impl TonicStatusExt for tonic::Status {
    fn http_status(&self) -> StatusCode {
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
}

/// Extension for `Result<T, tonic::Status>` to extract HTTP status for CB recording.
pub(crate) trait TonicResultExt {
    /// Returns the HTTP status code for circuit breaker recording.
    /// `Ok` → 200, `Err(status)` → mapped HTTP status code.
    fn cb_status_code(&self) -> u16;
}

impl<T> TonicResultExt for Result<T, tonic::Status> {
    fn cb_status_code(&self) -> u16 {
        self.as_ref()
            .map_or_else(|e| e.http_status().as_u16(), |_| 200)
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
            assert_eq!(result.cb_status_code(), 400);
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
            assert_eq!(result.cb_status_code(), 500);
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
        assert_eq!(ok.cb_status_code(), 200);
    }
}
