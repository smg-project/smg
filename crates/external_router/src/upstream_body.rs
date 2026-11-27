//! An upstream's error body on its way to the client: read whole within a
//! bound, the provider's account identifiers removed, the provider's words in
//! the log.
//!
//! [`answer_error`] (or [`read_error`] where the answer is built later, then
//! [`error_response`]) is the one answer every relay gives a provider's
//! refusal: the chat relay, the realtime REST relay, the WebRTC setup. The
//! bound keeps a large or endless body from being buffered without limit.

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderValue, StatusCode},
    response::Response,
};
use futures::StreamExt;
use tracing::warn;

use crate::error;

/// The most of an upstream error body that is read before it is answered.
pub const MAX_ERROR_RESPONSE_SIZE: usize = 1024 * 1024;

/// What a caller reads where an upstream error body past the bound would
/// have stood.
pub(crate) const ERROR_BODY_TOO_LARGE: &str = "Upstream error response too large";

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

/// An upstream's error body as the text a relay builds its own message
/// around: the body within [`MAX_ERROR_RESPONSE_SIZE`] with the provider's
/// account identifiers removed, [`ERROR_BODY_TOO_LARGE`] for a body past the
/// bound, and nothing for a body that cannot be read.
#[cfg(feature = "openai")]
pub(crate) async fn error_body_text(resp: reqwest::Response) -> String {
    match read_response_body_limited(resp, MAX_ERROR_RESPONSE_SIZE).await {
        ReadBodyResult::Ok(body) => error::sanitize_error_body(&body),
        ReadBodyResult::TooLarge => ERROR_BODY_TOO_LARGE.to_string(),
        ReadBodyResult::Error(_) => String::new(),
    }
}

/// An upstream's error answer once read: its status, its content type, and
/// its body with the provider's account identifiers removed, or the reason
/// there is none to show.
#[derive(Debug)]
pub struct UpstreamError {
    /// The upstream's status: the provider's verdict, kept on the answer.
    pub status: StatusCode,
    content_type: Option<HeaderValue>,
    body: Result<String, Unread>,
}

/// Why an error body is not shown: past the bound, or the read failed.
#[derive(Debug, Clone, Copy)]
enum Unread {
    TooLarge,
    Failed,
}

/// Read an upstream's error answer: the body whole within
/// [`MAX_ERROR_RESPONSE_SIZE`]. The provider's own words go to the log at
/// warn; the copy kept for the client loses the provider's account
/// identifiers ([`error::sanitize_error_body`]).
pub async fn read_error(resp: reqwest::Response, model: &str) -> UpstreamError {
    let status = resp.status();
    let content_type = resp.headers().get(CONTENT_TYPE).cloned();
    let body = match read_response_body_limited(resp, MAX_ERROR_RESPONSE_SIZE).await {
        ReadBodyResult::Ok(body) => {
            warn!(
                model = %model,
                status = %status,
                body_preview = %body.chars().take(200).collect::<String>(),
                "Upstream error"
            );
            Ok(error::sanitize_error_body(&body))
        }
        ReadBodyResult::TooLarge => {
            warn!(
                model = %model,
                status = %status,
                max_size = MAX_ERROR_RESPONSE_SIZE,
                "Upstream error body too large"
            );
            Err(Unread::TooLarge)
        }
        ReadBodyResult::Error(e) => {
            warn!(
                model = %model,
                status = %status,
                error = %e,
                "Failed to read upstream error body"
            );
            Err(Unread::Failed)
        }
    };
    UpstreamError {
        status,
        content_type,
        body,
    }
}

/// The client's answer for a read upstream error: the sanitized body under
/// the upstream's status and content type; the gateway's own envelope under
/// the upstream's status, and none of the body, when it was past the bound
/// or could not be read. The status is the provider's verdict and the retry
/// layer reads it: a 500 would be retried.
pub fn error_response(error: UpstreamError) -> Response {
    match error.body {
        Ok(body) => {
            let mut response = Response::new(Body::from(body));
            *response.status_mut() = error.status;
            if let Some(ct) = error.content_type {
                response.headers_mut().insert(CONTENT_TYPE, ct);
            }
            response
        }
        Err(Unread::TooLarge) => {
            error::create_error(error.status, "upstream_error", ERROR_BODY_TOO_LARGE)
        }
        Err(Unread::Failed) => error::create_error(
            error.status,
            "upstream_error",
            "Failed to read upstream error response",
        ),
    }
}

/// [`read_error`] and [`error_response`] in one step, for a relay with
/// nothing to do in between.
pub async fn answer_error(resp: reqwest::Response, model: &str) -> Response {
    error_response(read_error(resp, model).await)
}

#[cfg(test)]
#[path = "../tests/unit/upstream_body/tests.rs"]
mod tests;
