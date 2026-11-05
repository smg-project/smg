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
                metrics_labels::REASON_INVALID_REQUEST,
            );
            return StatusCode::UNAUTHORIZED.into_response();
        }
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
            error!(error = %e, endpoint, "Failed to forward realtime REST request");
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
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
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
            error!(error = %e, "Failed to read upstream response body");
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::{body::to_bytes, http::header::CONTENT_TYPE};

    use super::*;

    /// The provider's error envelope as its realtime REST endpoints answer
    /// it: the message names the operator's organization and project.
    const REALTIME_REFUSAL: &str = r#"{"error":{"message":"Rate limit reached for realtime-model in organization org-abc123 project proj_xyz789 on requests per min (RPM): Limit 3, Used 3, Requested 1.","type":"requests","param":null,"code":"rate_limit_exceeded"}}"#;

    fn upstream(status: u16, body: &'static str) -> reqwest::Response {
        reqwest::Response::from(
            http::Response::builder()
                .status(status)
                .header(CONTENT_TYPE, "application/json")
                .body(reqwest::Body::from(body))
                .expect("a response"),
        )
    }

    async fn text_of(response: Response) -> String {
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("a body");
        String::from_utf8(bytes.to_vec()).expect("utf-8")
    }

    /// A refusal of a realtime REST call reaches the caller under its status
    /// and content type with the operator's organization and project gone
    /// and the rest of the message kept.
    #[tokio::test]
    async fn a_realtime_refusal_reaches_the_caller_without_the_operators_identifiers() {
        let response = proxy_response(upstream(429, REALTIME_REFUSAL), "realtime-model").await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
        let text = text_of(response).await;
        assert!(!text.contains("org-abc123"), "{text}");
        assert!(!text.contains("proj_xyz789"), "{text}");
        assert!(
            text.contains("Rate limit reached for realtime-model"),
            "{text}"
        );
        assert!(text.contains(r#""code":"rate_limit_exceeded""#), "{text}");
    }

    /// A success passes as it came, byte for byte.
    #[tokio::test]
    async fn a_realtime_success_passes_as_it_came() {
        const SESSION: &str =
            r#"{"id":"sess_1","object":"realtime.session","client_secret":{"value":"ek_test"}}"#;
        let response = proxy_response(upstream(200, SESSION), "realtime-model").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
        assert_eq!(text_of(response).await, SESSION);
    }
}
