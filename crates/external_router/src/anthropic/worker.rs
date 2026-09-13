//! HTTP transport layer for Anthropic router
//!
//! Contains the core primitives for building requests, sending them,
//! and processing responses. These functions are composed by the
//! streaming and non-streaming processors.

use std::time::{Duration, Instant};

use axum::{
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use openai_protocol::messages::{CreateMessageRequest, Message};
use tracing::{debug, error, info, warn};

use super::utils::{read_response_body_limited, should_propagate_header, ReadBodyResult};
use crate::{
    error,
    metrics::{bool_to_static_str, metrics_labels, Metrics},
    worker::ExternalWorker,
};

/// Maximum error response body size to prevent DoS (1 MB)
const MAX_ERROR_RESPONSE_SIZE: usize = 1024 * 1024;

/// Build the target URL and propagated headers for a worker request.
pub(crate) fn build_request(
    worker: &dyn ExternalWorker,
    headers: Option<&HeaderMap>,
) -> (String, HeaderMap) {
    let url = format!("{}/v1/messages", worker.url());
    let mut propagated = HeaderMap::new();
    if let Some(input_headers) = headers {
        for (key, value) in input_headers {
            if should_propagate_header(key.as_str()) {
                propagated.insert(key.clone(), value.clone());
            }
        }
    }
    debug!(url = %url, header_count = %propagated.len(), "Request built");
    (url, propagated)
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
                "parse_error",
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

    warn!(
        model = %model_id,
        status = %status,
        body_preview = %body.chars().take(200).collect::<String>(),
        "Backend error"
    );

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
        metrics_labels::ERROR_BACKEND,
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
