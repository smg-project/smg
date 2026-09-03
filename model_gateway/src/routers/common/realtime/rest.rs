//! Shared helpers for Realtime API REST proxy responses.

use std::{sync::Arc, time::Instant};

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use tracing::error;

use super::RealtimeLabels;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::common::header_utils::extract_auth_header,
    worker::{worker::WorkerLoadGuard, Worker},
};

/// Forward a realtime REST request to the upstream worker.
///
/// Shared logic for sessions, client_secrets, and transcription_sessions:
/// auth, load tracking, metrics, and proxy.
///
/// The caller is responsible for worker selection; this function receives
/// the pre-selected worker (or an error response).
pub(crate) async fn forward_realtime_rest(
    labels: RealtimeLabels,
    worker: Result<Arc<dyn Worker>, Response>,
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
            );
            return response;
        }
    };

    let auth = match extract_auth_header(headers, worker.api_key()) {
        Some(v) => v,
        None => {
            Metrics::record_router_error(
                labels.router,
                labels.backend,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint_label,
                metrics_labels::ERROR_VALIDATION,
            );
            return StatusCode::UNAUTHORIZED.into_response();
        }
    };

    // Track load for the duration of the upstream request
    let _guard = WorkerLoadGuard::new(worker.clone(), headers);

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
            let response = proxy_response(resp).await;
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
                );
            }
            response
        }
        Err(e) => {
            worker.record_outcome(502);
            error!(error = %e, endpoint, "Failed to forward realtime REST request");
            Metrics::record_router_error(
                labels.router,
                labels.backend,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint_label,
                metrics_labels::ERROR_BACKEND,
            );
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}

/// Convert an upstream reqwest Response into an axum Response,
/// preserving status code and body.
pub(crate) async fn proxy_response(resp: reqwest::Response) -> Response {
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    match resp.bytes().await {
        Ok(body) => (status, [(http::header::CONTENT_TYPE, content_type)], body).into_response(),
        Err(e) => {
            error!(error = %e, "Failed to read upstream response body");
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}
