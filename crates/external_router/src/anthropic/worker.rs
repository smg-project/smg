//! HTTP transport layer for Anthropic router
//!
//! Contains the core primitives for building requests, sending them,
//! and processing responses. These functions are composed by the
//! streaming and non-streaming processors.

use std::time::{Duration, Instant};

use axum::{
    http::{HeaderMap, HeaderValue},
    response::{IntoResponse, Response},
};
use openai_protocol::messages::{CreateMessageRequest, Message};
use tracing::{debug, error, info, warn};

use super::utils::should_propagate_header;
use crate::{
    error,
    metrics::{bool_to_static_str, error_reason_from_status, metrics_labels, Metrics},
    upstream_body::{read_response_body_limited, ReadBodyResult, MAX_ERROR_RESPONSE_SIZE},
    worker::ExternalWorker,
};

/// Build the target URL and propagated headers for a worker request.
pub(crate) fn build_request(
    worker: &dyn ExternalWorker,
    headers: Option<&HeaderMap>,
) -> (String, HeaderMap) {
    let url = format!("{}/v1/messages", worker.url());
    let propagated = propagated_headers(headers, worker.api_key());
    debug!(url = %url, header_count = %propagated.len(), "Request built");
    (url, propagated)
}

/// The headers a worker request carries: the caller's propagated ones, and,
/// when the caller sent neither `authorization` nor `x-api-key` (a gateway
/// key ends at the gateway's auth middleware), the worker's configured key
/// as `x-api-key` with the API version the provider requires, as the
/// provider's own header rule applies it.
fn propagated_headers(headers: Option<&HeaderMap>, configured_key: Option<&String>) -> HeaderMap {
    let mut propagated = HeaderMap::new();
    if let Some(input_headers) = headers {
        for (key, value) in input_headers {
            if should_propagate_header(key.as_str()) {
                propagated.insert(key.clone(), value.clone());
            }
        }
    }
    if !propagated.contains_key("authorization") && !propagated.contains_key("x-api-key") {
        if let Some(value) = configured_key.and_then(|key| HeaderValue::from_str(key).ok()) {
            propagated.insert("x-api-key", value);
            if !propagated.contains_key("anthropic-version") {
                propagated.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
            }
        }
    }
    propagated
}

/// Send the HTTP request to the worker.
pub(crate) async fn send_request(
    http_client: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
    request: &CreateMessageRequest,
    timeout: Duration,
) -> Result<reqwest::Response, Response> {
    debug!(url = %url, "Sending request to worker");

    let mut builder = http_client.post(url).json(request).timeout(timeout);
    for (key, value) in headers {
        builder = builder.header(key, value);
    }

    match builder.send().await {
        Ok(response) => {
            debug!(url = %url, status = %response.status(), "Received response from worker");
            Ok(response)
        }
        Err(e) => {
            warn!(url = %url, error = %e, "Request to worker failed");

            if e.is_timeout() {
                Err(error::gateway_timeout(
                    "timeout",
                    format!("Request timeout: {e}"),
                ))
            } else if e.is_connect() {
                Err(error::bad_gateway(
                    "connection_failed",
                    format!("Connection failed: {e}"),
                ))
            } else {
                Err(error::bad_gateway(
                    "request_failed",
                    format!("Request failed: {e}"),
                ))
            }
        }
    }
}

/// Parse a successful non-streaming JSON response.
pub(crate) async fn parse_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
) -> Result<Message, Response> {
    match response.json::<Message>().await {
        Ok(message) => {
            debug!(
                model = %model_id,
                message_id = %message.id,
                "Successfully parsed response"
            );
            record_success_metrics(model_id, &message, start_time);
            info!(
                model = %model_id,
                message_id = %message.id,
                input_tokens = %message.usage.input_tokens,
                output_tokens = %message.usage.output_tokens,
                "Completed non-streaming request"
            );
            Ok(message)
        }
        Err(e) => {
            error!(model = %model_id, error = %e, "Failed to parse response");
            record_failure_metrics(
                model_id,
                start_time,
                "parse_error",
                metrics_labels::REASON_BAD_RESPONSE,
            );
            Err(error::bad_gateway(
                "parse_error",
                "Invalid response from backend",
            ))
        }
    }
}

/// Handle a non-success response: read body (limited) and record metrics.
pub(crate) async fn handle_error_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
) -> Response {
    let status = response.status();

    let body = match read_response_body_limited(response, MAX_ERROR_RESPONSE_SIZE).await {
        ReadBodyResult::Ok(b) if b.is_empty() => format!("Backend returned error: {status}"),
        ReadBodyResult::Ok(b) => b,
        ReadBodyResult::TooLarge => {
            warn!(
                model = %model_id,
                max_size = %MAX_ERROR_RESPONSE_SIZE,
                "Error response body too large"
            );
            format!("Backend returned error: {status} (response too large)")
        }
        ReadBodyResult::Error(e) => {
            warn!(model = %model_id, error = %e, "Failed to read error response body");
            format!("Backend returned error: {status}")
        }
    };
    // The provider's own words go to the operator's log; the caller gets the
    // body with the provider's account identifiers (the operator's, not the
    // caller's) removed.
    warn!(
        model = %model_id,
        status = %status,
        body_preview = %body.chars().take(200).collect::<String>(),
        "Backend error"
    );
    let body = error::sanitize_error_body(&body);

    record_failure_metrics(
        model_id,
        start_time,
        metrics_labels::ERROR_BACKEND,
        error_reason_from_status(status),
    );

    (status, body).into_response()
}

// ============================================================================
// Metrics Helpers
// ============================================================================

pub(crate) fn record_router_request(model_id: &str, streaming: bool) {
    Metrics::record_router_request(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        bool_to_static_str(streaming),
    );
}

fn record_success_metrics(model_id: &str, message: &Message, start_time: Instant) {
    Metrics::record_router_duration(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        start_time.elapsed(),
    );
    Metrics::record_router_tokens(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        model_id,
        "messages",
        metrics_labels::TOKEN_INPUT,
        message.usage.input_tokens as u64,
    );
    Metrics::record_router_tokens(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        model_id,
        "messages",
        metrics_labels::TOKEN_OUTPUT,
        message.usage.output_tokens as u64,
    );
}

/// Records a failed Messages request: its duration, then the error under
/// `kind` and `reason`, with the route's fixed labels.
pub(crate) fn record_failure_metrics(
    model_id: &str,
    start_time: Instant,
    kind: &'static str,
    reason: &'static str,
) {
    Metrics::record_router_duration(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        start_time.elapsed(),
    );
    Metrics::record_router_error(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        kind,
        reason,
    );
}

#[cfg(test)]
mod tests {
    use axum::{body::to_bytes, http::StatusCode};

    use super::*;

    const UPSTREAM_REFUSAL: &str = r#"{"error":{"message":"Rate limit reached for model in organization org-abc123 project proj_xyz789","type":"rate_limit","code":"rate_limit_exceeded"}}"#;

    /// A provider's refusal reaches the Messages caller under its status with
    /// the operator's organization and project gone.
    #[tokio::test]
    async fn an_upstream_errors_body_loses_the_operators_identifiers() {
        let upstream = reqwest::Response::from(
            http::Response::builder()
                .status(429)
                .body(reqwest::Body::from(UPSTREAM_REFUSAL))
                .expect("a response"),
        );
        let response = handle_error_response(upstream, "m", Instant::now()).await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("a body");
        let text = std::str::from_utf8(&body).expect("utf-8");
        assert!(!text.contains("org-abc123"), "{text}");
        assert!(!text.contains("proj_xyz789"), "{text}");
        assert!(text.contains("Rate limit reached for model"), "{text}");
    }

    fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
        headers.get(name).and_then(|value| value.to_str().ok())
    }

    /// The configured key goes out as `x-api-key` with the version header when
    /// the caller sent no credential; a caller's own `x-api-key` or
    /// `authorization` wins and the configured key stays home; without either
    /// nothing is added.
    #[test]
    fn the_configured_key_goes_out_when_the_caller_sent_none() {
        let configured = "provider-key".to_string();

        let none = propagated_headers(None, Some(&configured));
        assert_eq!(header(&none, "x-api-key"), Some("provider-key"));
        assert_eq!(header(&none, "anthropic-version"), Some("2023-06-01"));

        let mut callers = HeaderMap::new();
        callers.insert("x-api-key", HeaderValue::from_static("callers-key"));
        callers.insert("anthropic-version", HeaderValue::from_static("2024-01-01"));
        callers.insert("x-request-id", HeaderValue::from_static("r-1"));
        let own = propagated_headers(Some(&callers), Some(&configured));
        assert_eq!(header(&own, "x-api-key"), Some("callers-key"));
        assert_eq!(header(&own, "anthropic-version"), Some("2024-01-01"));
        assert!(own.get("x-request-id").is_none());

        let mut bearer = HeaderMap::new();
        bearer.insert(
            "authorization",
            HeaderValue::from_static("Bearer callers-key"),
        );
        let with_bearer = propagated_headers(Some(&bearer), Some(&configured));
        assert_eq!(
            header(&with_bearer, "authorization"),
            Some("Bearer callers-key")
        );
        assert!(with_bearer.get("x-api-key").is_none());

        assert!(propagated_headers(None, None).is_empty());
    }
}
