//! Shared utilities for Anthropic router
//!
//! This module contains common helper functions used across different
//! Anthropic API handlers (messages, models, etc.)

use smg_http_utils::read_body_capped;
// ============================================================================
// Header Propagation
// ============================================================================

/// Check if header should be propagated to Anthropic backend
///
/// Only propagates authentication and Anthropic-specific headers.
/// This prevents leaking sensitive headers like cookies or internal routing info.
pub fn should_propagate_header(key: &str) -> bool {
    key.eq_ignore_ascii_case("authorization")
        || key.eq_ignore_ascii_case("x-api-key")
        || key.eq_ignore_ascii_case("anthropic-version")
        || key.eq_ignore_ascii_case("anthropic-beta")
}

// ============================================================================
// Response Body Reading
// ============================================================================

/// Result of reading a response body with size limit
pub enum ReadBodyResult {
    /// Successfully read the full body
    Ok(String),
    /// Body exceeded max size
    TooLarge,
    /// Error reading body
    Error(String),
}

/// Read a response body incrementally with a size limit.
///
/// SECURITY: This prevents DoS by avoiding unbounded buffering when
/// content-length is unknown (e.g., chunked transfer encoding).
pub async fn read_response_body_limited(
    response: reqwest::Response,
    max_size: usize,
) -> ReadBodyResult {
    match read_body_capped(response.bytes_stream(), max_size).await {
        Ok((_, true)) => ReadBodyResult::TooLarge,
        Ok((body, false)) => match String::from_utf8(body.to_vec()) {
            Ok(body) => ReadBodyResult::Ok(body),
            Err(e) => ReadBodyResult::Error(format!("invalid UTF-8 in response body: {e}")),
        },
        Err(e) => ReadBodyResult::Error(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    fn response(chunks: Vec<Result<Bytes, std::io::Error>>) -> reqwest::Response {
        let body = reqwest::Body::wrap_stream(futures::stream::iter(chunks));
        http::Response::new(body).into()
    }

    #[tokio::test]
    async fn decodes_utf8_after_all_chunks() {
        let input = response(vec![
            Ok(Bytes::from_static(&[0xe4])),
            Ok(Bytes::from_static(&[0xbd, 0xa0])),
        ]);
        let result = read_response_body_limited(input, 3).await;
        assert!(matches!(result, ReadBodyResult::Ok(ref text) if text == "你"));
    }

    #[tokio::test]
    async fn rejects_invalid_utf8() {
        let input = response(vec![Ok(Bytes::from_static(&[0xff]))]);
        let result = read_response_body_limited(input, 1).await;
        assert!(matches!(result, ReadBodyResult::Error(ref e)
            if e.starts_with("invalid UTF-8 in response body:")));
    }

    #[tokio::test]
    async fn reports_overflow_before_utf8_decode() {
        let input = response(vec![Ok(Bytes::from_static(&[0xff, 0xff]))]);
        assert!(matches!(
            read_response_body_limited(input, 1).await,
            ReadBodyResult::TooLarge
        ));
    }

    #[tokio::test]
    async fn preserves_a_body_read_error() {
        let input = response(vec![Err(std::io::Error::other("body read failed"))]);
        assert!(matches!(
            read_response_body_limited(input, 8).await,
            ReadBodyResult::Error(_)
        ));
    }
}
