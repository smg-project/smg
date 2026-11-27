//! Shared helpers for Realtime API REST proxy responses.

use std::{sync::Arc, time::Instant};

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use tracing::error;

use super::RealtimeLabels;
use crate::{
    header_utils::extract_auth_header,
    metrics::{
        error_reason_for_refused_selection, error_reason_from_status, metrics_labels, Metrics,
    },
    upstream_body,
    worker::ExternalWorker,
};

/// Forward a realtime REST request to the upstream worker.
///
/// Shared logic for sessions, client_secrets, and transcription_sessions:
/// auth, load tracking, metrics, and proxy.
///
/// The caller is responsible for worker selection; this function receives
/// the pre-selected worker (or an error response).
pub async fn forward_realtime_rest(
    labels: RealtimeLabels,
    worker: Result<Arc<dyn ExternalWorker>, Response>,
    headers: Option<&HeaderMap>,
    body: &(impl serde::Serialize + Sync),
    model: &str,
    endpoint: &str,
    endpoint_label: &'static str,
) -> Response {
    let start = Instant::now();

    Metrics::record_router_request(
        labels.router,
        labels.backend,
        metrics_labels::CONNECTION_HTTP,
        model,
        endpoint_label,
        "false",
    );

    let worker = match worker {
        Ok(w) => w,
        Err(response) => {
            Metrics::record_router_error(
                labels.router,
                labels.backend,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint_label,
                metrics_labels::ERROR_NO_WORKERS,
                error_reason_for_refused_selection(response.status()),
            );
            return response;
        }
    };

    let Some(auth) = extract_auth_header(headers, worker.api_key()) else {
        Metrics::record_router_error(
            labels.router,
            labels.backend,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint_label,
            metrics_labels::ERROR_VALIDATION,
            metrics_labels::REASON_INVALID_REQUEST,
        );
        return StatusCode::UNAUTHORIZED.into_response();
    };

    // Track load for the duration of the upstream request
    let _guard = worker.hold_load(headers);

    let upstream_url = format!("{}{endpoint}", worker.url().trim_end_matches('/'));

    let result = worker
        .http_client()
        .post(&upstream_url)
        .header("Authorization", &auth)
        .json(body)
        .send()
        .await;

    match result {
        Ok(resp) => {
            let status = resp.status();
            worker.record_outcome(status.as_u16());
            let success = status.is_success();
            let response = proxy_response(resp, model).await;
            if success {
                Metrics::record_router_duration(
                    labels.router,
                    labels.backend,
                    metrics_labels::CONNECTION_HTTP,
                    model,
                    endpoint_label,
                    start.elapsed(),
                );
            } else {
                Metrics::record_router_error(
                    labels.router,
                    labels.backend,
                    metrics_labels::CONNECTION_HTTP,
                    model,
                    endpoint_label,
                    metrics_labels::ERROR_BACKEND,
                    error_reason_from_status(status),
                );
            }
            response
        }
        Err(e) => {
            worker.record_outcome(502);
            error!(
                url = %worker.redacted_url(),
                error = %e.without_url(),
                endpoint,
                "Failed to forward realtime REST request"
            );
            Metrics::record_router_error(
                labels.router,
                labels.backend,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint_label,
                metrics_labels::ERROR_BACKEND,
                metrics_labels::REASON_TRANSPORT_DISCONNECT,
            );
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}

/// The upstream's answer for the client: a success as it came, status, body
/// and content type; a refusal through [`upstream_body::answer_error`]
/// (the body bounded, the provider's account identifiers removed, its words
/// in the log at warn).
pub async fn proxy_response(resp: reqwest::Response, model: &str) -> Response {
    let status = resp.status();
    if !status.is_success() {
        return upstream_body::answer_error(resp, model).await;
    }
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    match resp.bytes().await {
        Ok(body) => (status, [(http::header::CONTENT_TYPE, content_type)], body).into_response(),
        Err(e) => {
            error!(model, error = %e.without_url(), "Failed to read upstream response body");
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/realtime/rest/tests.rs"]
mod tests;
