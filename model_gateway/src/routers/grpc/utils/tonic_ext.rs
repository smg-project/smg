//! Extension traits for tonic gRPC types.

use axum::response::Response;
use http::StatusCode;
use tonic::Code;

use crate::routers::{common::overload, error};

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
        let mut response = error::create_error(self.http_status(), code, msg);
        overload::apply_capacity_contract(&mut response);
        response
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
    use http::header::RETRY_AFTER;

    use super::*;
    use crate::routers::common::retry::is_retryable_response;

    #[test]
    fn resource_exhausted_maps_to_terminal_429_with_retry_after() {
        let status = tonic::Status::resource_exhausted("engine overloaded");
        let response = status.to_http_error("engine_overloaded", status.message().to_string());

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(response.headers().contains_key(RETRY_AFTER));
        assert!(!is_retryable_response(&response));
    }

    #[test]
    fn unrelated_grpc_error_keeps_its_existing_mapping() {
        let status = tonic::Status::unavailable("offline");
        let response = status.to_http_error("offline", status.message().to_string());

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(!response.headers().contains_key(RETRY_AFTER));
        assert!(is_retryable_response(&response));
    }
}
