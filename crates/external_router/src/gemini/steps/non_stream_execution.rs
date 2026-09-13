//! NonStreamRequestExecution step.
//!
//! Transition: NonStreamRequest → ProcessResponse (no tool calls)
//!                               | NonStreamRequest (MCP tool loop — Phase 2)

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::Value;

use crate::{
    error,
    gemini::{
        context::RequestContext,
        state::{RequestState, StepResult},
    },
    header_utils::ApiProvider,
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
        Err(e) => {
            let status = if e.is_timeout() { 504 } else { 502 };
            worker.record_outcome(status);
            tracing::warn!(url = %upstream_url, error = %e, "Request to worker failed");

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
        let status = StatusCode::from_u16(response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let content_type = response.headers().get(http::header::CONTENT_TYPE).cloned();
        let body = response.text().await.unwrap_or_else(|e| {
            tracing::warn!("Failed to read upstream error body: {e}");
            String::new()
        });
        let body = error::sanitize_error_body(&body);
        let mut resp = (status, body).into_response();
        if let Some(content_type) = content_type {
            resp.headers_mut()
                .insert(http::header::CONTENT_TYPE, content_type);
        }
        return Err(resp);
    }

    // Parse response body
    let response_json: Value = match response.json().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(url = %upstream_url, error = %e, "Upstream returned invalid JSON");
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
