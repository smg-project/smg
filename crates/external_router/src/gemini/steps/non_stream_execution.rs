//! NonStreamRequestExecution step.
//!
//! Transition: NonStreamRequest → ProcessResponse (no tool calls)
//!                               | NonStreamRequest (MCP tool loop — Phase 2)

use axum::response::{IntoResponse, Response};
use serde_json::Value;

use crate::{
    error,
    gemini::{
        context::RequestContext,
        state::{RequestState, StepResult},
    },
    header_utils::ApiProvider,
    retry::{is_request_timeout, request_timeout_response},
    upstream_body::{
        read_response_body_limited, ReadBodyResult, ERROR_BODY_TOO_LARGE, MAX_ERROR_RESPONSE_SIZE,
    },
    worker::redacted_url,
};

/// POST the payload to the upstream worker and handle the response.
///
/// For Phase 1 (no MCP), this step simply sends the request, records
/// circuit breaker outcomes, and advances to `ProcessResponse`.
///
/// Phase 2 will add MCP tool call detection: if the response contains
/// function calls that map to MCP tools, they are executed inline and
/// the state stays at `NonStreamRequest` (the driver re-enters this step).
pub(crate) async fn non_stream_request_execution(
    ctx: &mut RequestContext,
) -> Result<StepResult, Response> {
    let payload = ctx
        .processing
        .payload
        .take()
        .ok_or_else(|| error::internal_error("internal_error", "Payload not prepared"))?;

    let upstream_url = ctx
        .processing
        .upstream_url
        .as_deref()
        .ok_or_else(|| error::internal_error("internal_error", "Upstream URL not set"))?;

    let worker = ctx
        .processing
        .worker
        .as_ref()
        .ok_or_else(|| error::internal_error("internal_error", "Worker not selected"))?
        .clone();

    // Build the upstream request — always use Gemini provider (not URL-based detection)
    let provider = ApiProvider::Gemini;
    let auth_header = provider.extract_auth_header(ctx.input.headers.as_ref(), worker.api_key());
    let request_builder = provider.apply_headers(
        ctx.components
            .client
            .post(upstream_url)
            .json(&payload)
            .timeout(ctx.components.request_timeout),
        auth_header.as_ref(),
    );

    // Send the request
    let response = match request_builder.send().await {
        Ok(r) => r,
        // The request's own timeout is its end, not a worker fault to route
        // around (see `request_timeout_response`).
        Err(e) if is_request_timeout(&e) => {
            worker.record_outcome(504);
            tracing::warn!(url = %redacted_url(upstream_url), error = %e.without_url(), "Request to worker timed out");
            return Err(request_timeout_response(
                "Upstream worker request timed out",
            ));
        }
        Err(e) => {
            let status = if e.is_timeout() { 504 } else { 502 };
            worker.record_outcome(status);
            let e = e.without_url();
            tracing::warn!(url = %redacted_url(upstream_url), error = %e, "Request to worker failed");

            return if e.is_timeout() {
                Err(error::gateway_timeout(
                    "upstream_timeout",
                    "Upstream worker request timed out",
                ))
            } else {
                Err(error::bad_gateway(
                    "upstream_error",
                    "Failed to reach upstream worker",
                ))
            };
        }
    };

    let status = response.status();
    worker.record_outcome(status.as_u16());

    if !status.is_success() {
        let content_type = response.headers().get(http::header::CONTENT_TYPE).cloned();
        // Within the capped reader's bound, as the crate's other relays read
        // an error body: one past it is not passed on.
        let body = match read_response_body_limited(response, MAX_ERROR_RESPONSE_SIZE).await {
            ReadBodyResult::Ok(body) => error::sanitize_error_body(&body),
            ReadBodyResult::TooLarge => {
                return Err(error::create_error(
                    status,
                    "upstream_error",
                    ERROR_BODY_TOO_LARGE,
                ));
            }
            ReadBodyResult::Error(e) => {
                tracing::warn!(url = %redacted_url(upstream_url), error = %e, "Failed to read upstream error body");
                String::new()
            }
        };
        let mut resp = (status, body).into_response();
        if let Some(content_type) = content_type {
            resp.headers_mut()
                .insert(http::header::CONTENT_TYPE, content_type);
        }
        return Err(resp);
    }

    // Parse response body
    let response_json: Value = match response.json().await.map_err(reqwest::Error::without_url) {
        Ok(r) => r,
        Err(e) if is_request_timeout(&e) => {
            tracing::warn!(url = %redacted_url(upstream_url), error = %e, "Upstream response timed out");
            return Err(request_timeout_response(
                "Upstream worker response timed out",
            ));
        }
        Err(e) => {
            tracing::warn!(url = %redacted_url(upstream_url), error = %e, "Upstream returned invalid JSON");
            return Err(error::bad_gateway(
                "upstream_error",
                "Upstream returned an invalid response",
            ));
        }
    };

    // TODO (Phase 2): Scan outputs for function_call items that map to MCP tools.
    // If found, execute tools, build resume payload, and stay in NonStreamRequest.

    ctx.processing.upstream_response = Some(response_json);
    ctx.state = RequestState::ProcessResponse;
    Ok(StepResult::Continue)
}

#[cfg(test)]
#[path = "../../../tests/unit/gemini/steps/non_stream_execution/tests.rs"]
mod tests;
