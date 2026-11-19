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
mod tests {
    use axum::http::header::CONTENT_TYPE;

    use super::*;
    use crate::test_support::{
        answering, http_client, text_of, upstream, RefusedAddress, StubWorker,
    };

    /// The provider's error envelope as its realtime REST endpoints answer
    /// it: the message names the operator's organization and project.
    const REALTIME_REFUSAL: &str = r#"{"error":{"message":"Rate limit reached for realtime-model in organization org-abc123 project proj_xyz789 on requests per min (RPM): Limit 3, Used 3, Requested 1.","type":"requests","param":null,"code":"rate_limit_exceeded"}}"#;

    /// A refusal of a realtime REST call reaches the caller under its status
    /// and content type with the operator's organization and project gone
    /// and the rest of the message kept.
    #[tokio::test]
    async fn a_realtime_refusal_reaches_the_caller_without_the_operators_identifiers() {
        let response = proxy_response(
            upstream(429, Some("application/json"), REALTIME_REFUSAL),
            "realtime-model",
        )
        .await;
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
        let response = proxy_response(
            upstream(200, Some("application/json"), SESSION),
            "realtime-model",
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
        assert_eq!(text_of(response).await, SESSION);
    }

    /// A realtime REST call nothing answers, to an address registered with a
    /// query: the failure line names the worker masked and says why without
    /// the query. The HTTP client's own error text ends with the request URL,
    /// its query kept.
    #[tracing_test::traced_test]
    #[tokio::test]
    async fn a_failed_realtime_send_goes_without_the_clients_url_in_its_line() {
        let refused = RefusedAddress::reserve();
        let worker = StubWorker::at(&format!("http://{}?key=s3cret", refused.addr()));
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer k".parse().expect("a header value"));

        let response = forward_realtime_rest(
            RealtimeLabels {
                router: metrics_labels::ROUTER_OPENAI,
                backend: metrics_labels::BACKEND_EXTERNAL,
            },
            Ok(worker),
            Some(&headers),
            &serde_json::json!({}),
            "realtime-model",
            "/v1/realtime/sessions",
            "realtime_sessions",
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert!(logs_contain("Failed to forward realtime REST request"));
        assert!(!logs_contain("s3cret"));
    }

    /// A realtime answer lost mid-body: the failure line names the model and
    /// says why without the request URL. The HTTP client's text of a failed
    /// body read ends with it, its query kept.
    #[tracing_test::traced_test]
    #[tokio::test]
    async fn a_realtime_body_cut_short_goes_without_the_clients_url_in_its_line() {
        let addr = answering(200, "short", true).await;
        let resp = http_client()
            .get(format!("http://{addr}/v1/realtime/sessions?key=s3cret"))
            .send()
            .await
            .expect("the answer's head");

        let response = proxy_response(resp, "realtime-model").await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert!(logs_contain("Failed to read upstream response body"));
        assert!(logs_contain("realtime-model"));
        assert!(!logs_contain("s3cret"));
    }
}
