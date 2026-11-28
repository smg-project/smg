//! HTTP transport layer for Anthropic router
//!
//! Contains the core primitives for building requests, sending them,
//! and processing responses. These functions are composed by the
//! streaming and non-streaming processors.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue},
    response::{IntoResponse, Response},
};
use openai_protocol::messages::{CreateMessageRequest, Message};
use serde::Deserialize;
use serde_json::value::RawValue;
use tracing::{debug, error, info, warn};

use super::utils::should_propagate_header;
use crate::{
    error,
    metrics::{bool_to_static_str, error_reason_from_status, metrics_labels, Metrics},
    upstream_body::{read_response_body_limited, ReadBodyResult, MAX_ERROR_RESPONSE_SIZE},
    worker::{redacted_url, ExternalWorker},
};

/// Build the target URL and propagated headers for a worker request.
pub(crate) fn build_request(
    worker: &dyn ExternalWorker,
    headers: Option<&HeaderMap>,
) -> (String, HeaderMap) {
    let url = format!("{}/v1/messages", worker.url());
    let propagated = propagated_headers(headers, worker.api_key());
    debug!(url = %redacted_url(&url), header_count = %propagated.len(), "Request built");
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

/// Sends the HTTP request to the worker. A send that fails below HTTP is
/// counted against the model before it becomes the caller's answer: a timeout
/// under `timeout`/`deadline` with the 504, any other transport failure under
/// `backend_error`/`transport_disconnect` with the 502.
pub(crate) async fn send_request(
    http_client: &reqwest::Client,
    url: &str,
    headers: &HeaderMap,
    request: &CreateMessageRequest,
    timeout: Duration,
    model_id: &str,
    start_time: Instant,
) -> Result<reqwest::Response, Response> {
    debug!(url = %redacted_url(url), "Sending request to worker");

    let mut builder = http_client.post(url).json(request).timeout(timeout);
    for (key, value) in headers {
        builder = builder.header(key, value);
    }

    match builder.send().await {
        Ok(response) => {
            debug!(url = %redacted_url(url), status = %response.status(), "Received response from worker");
            Ok(response)
        }
        Err(e) => {
            // The client's text ends with the request URL: the line names the
            // worker masked, and the answer names no address.
            let e = e.without_url();
            warn!(url = %redacted_url(url), error = %e, "Request to worker failed");

            let (kind, reason) = if e.is_timeout() {
                (
                    metrics_labels::ERROR_TIMEOUT,
                    metrics_labels::REASON_DEADLINE,
                )
            } else {
                (
                    metrics_labels::ERROR_BACKEND,
                    metrics_labels::REASON_TRANSPORT_DISCONNECT,
                )
            };
            record_failure_metrics(model_id, start_time, kind, reason);

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
            record_success_metrics(
                model_id,
                u64::from(message.usage.input_tokens),
                u64::from(message.usage.output_tokens),
                start_time,
            );
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
            error!(model = %model_id, error = %e.without_url(), "Failed to parse response");
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

/// Relay a successful non-streaming answer as the provider wrote it, for the
/// plain path, which reads nothing of it. The body must be a JSON object: one
/// that is not is the 502 `parse_error`, as when the answer was read typed. A
/// body that stops coming is sorted as a send that fails (`send_request`): the
/// request's timeout running out mid-body is the 504 under
/// `timeout`/`deadline`, any other read failure the 502 under
/// `backend_error`/`transport_disconnect`. The answer's usage counts go to the
/// token metrics, and its bytes leave under the upstream's status and content
/// type, so a key or a block this build does not model reaches the client as
/// sent.
pub(crate) async fn relay_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
) -> Response {
    let status = response.status();
    let content_type = response.headers().get(CONTENT_TYPE).cloned();
    let body = match response.bytes().await {
        Ok(body) => body,
        Err(e) => {
            // The client's text ends with the request URL; the answers name
            // no address.
            let e = e.without_url();
            warn!(model = %model_id, error = %e, "Reading the response body failed");
            if e.is_timeout() {
                record_failure_metrics(
                    model_id,
                    start_time,
                    metrics_labels::ERROR_TIMEOUT,
                    metrics_labels::REASON_DEADLINE,
                );
                return error::gateway_timeout("timeout", format!("Request timeout: {e}"));
            }
            record_failure_metrics(
                model_id,
                start_time,
                metrics_labels::ERROR_BACKEND,
                metrics_labels::REASON_TRANSPORT_DISCONNECT,
            );
            return error::bad_gateway("request_failed", format!("Request failed: {e}"));
        }
    };
    // The top-level keys alone, each value left as written: a map refuses an
    // answer that is no JSON object, and the rest of the document is scanned
    // past, not built.
    let answer = match serde_json::from_slice::<HashMap<String, &RawValue>>(&body) {
        Ok(answer) => answer,
        Err(e) => {
            error!(model = %model_id, error = %e, "Failed to parse response");
            record_failure_metrics(
                model_id,
                start_time,
                "parse_error",
                metrics_labels::REASON_BAD_RESPONSE,
            );
            return error::bad_gateway("parse_error", "Invalid response from backend");
        }
    };
    #[derive(Default, Deserialize)]
    struct Counts {
        #[serde(default)]
        input_tokens: u64,
        #[serde(default)]
        output_tokens: u64,
    }
    let counts = answer
        .get("usage")
        .and_then(|usage| serde_json::from_str::<Counts>(usage.get()).ok())
        .unwrap_or_default();
    let message_id = answer
        .get("id")
        .and_then(|id| serde_json::from_str::<String>(id.get()).ok())
        .unwrap_or_default();
    record_success_metrics(
        model_id,
        counts.input_tokens,
        counts.output_tokens,
        start_time,
    );
    info!(
        model = %model_id,
        message_id = %message_id,
        input_tokens = %counts.input_tokens,
        output_tokens = %counts.output_tokens,
        "Completed non-streaming request"
    );
    let mut relayed = Response::new(Body::from(body));
    *relayed.status_mut() = status;
    relayed.headers_mut().insert(
        CONTENT_TYPE,
        content_type.unwrap_or_else(|| HeaderValue::from_static("application/json")),
    );
    relayed
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

fn record_success_metrics(
    model_id: &str,
    input_tokens: u64,
    output_tokens: u64,
    start_time: Instant,
) {
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
        input_tokens,
    );
    Metrics::record_router_tokens(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        model_id,
        "messages",
        metrics_labels::TOKEN_OUTPUT,
        output_tokens,
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

/// What a test reads back from a local recorder: the series a failed request
/// was counted under.
#[cfg(test)]
#[path = "../../tests/unit/anthropic/worker/probe.rs"]
pub(crate) mod probe;

#[cfg(test)]
#[path = "../../tests/unit/anthropic/worker/tests.rs"]
mod tests;
