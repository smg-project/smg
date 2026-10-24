//! The gRPC status for a request an engine aborted with an HTTP-style status:
//! one table for every wire whose engine attributes its aborts, the table the
//! engines' Python servicers share, so the two front ends of one engine answer
//! the same engine condition with the same status.

/// The gRPC status for a request the engine aborted with an HTTP-style
/// status. 400 is the caller's fault, 404 a missing resource, 408 and 504 a
/// deadline, 429 a full queue, 503 an unavailable engine; anything else, a
/// missing status included, is internal.
pub(crate) fn engine_abort_status(status: Option<u16>, message: String) -> tonic::Status {
    match status {
        Some(400) => tonic::Status::invalid_argument(message),
        Some(404) => tonic::Status::not_found(message),
        Some(408 | 504) => tonic::Status::deadline_exceeded(message),
        Some(429) => tonic::Status::resource_exhausted(message),
        Some(503) => tonic::Status::unavailable(message),
        _ => tonic::Status::internal(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_engine_abort_carries_the_status_its_http_code_means() {
        let rows = [
            (Some(400), tonic::Code::InvalidArgument),
            (Some(404), tonic::Code::NotFound),
            (Some(408), tonic::Code::DeadlineExceeded),
            (Some(429), tonic::Code::ResourceExhausted),
            (Some(500), tonic::Code::Internal),
            (Some(503), tonic::Code::Unavailable),
            (Some(504), tonic::Code::DeadlineExceeded),
            (Some(418), tonic::Code::Internal),
            (None, tonic::Code::Internal),
        ];
        for (status, code) in rows {
            let got = engine_abort_status(status, "why".to_string());
            assert_eq!(got.code(), code, "{status:?}");
            assert_eq!(got.message(), "why");
        }
    }
}
