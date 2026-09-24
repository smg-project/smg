//! Shared utilities for Anthropic router
//!
//! This module contains common helper functions used across different
//! Anthropic API handlers (messages, models, etc.)

use futures::StreamExt;
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
    let mut stream = response.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut total_size: usize = 0;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                total_size += chunk.len();
                if total_size > max_size {
                    return ReadBodyResult::TooLarge;
                }
                buf.extend_from_slice(&chunk);
            }
            Err(e) => {
                return ReadBodyResult::Error(e.to_string());
            }
        }
    }

    // Decode the entire buffer at once to avoid corrupting multibyte UTF-8
    // sequences that may be split across chunk boundaries.
    match String::from_utf8(buf) {
        Ok(body) => ReadBodyResult::Ok(body),
        Err(e) => ReadBodyResult::Error(format!("invalid UTF-8 in response body: {e}")),
    }
}
