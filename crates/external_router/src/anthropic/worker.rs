//! HTTP transport layer for Anthropic router
//!
//! Contains the core primitives for building requests, sending them,
//! and processing responses. These functions are composed by the
//! streaming and non-streaming processors.

use std::time::{Duration, Instant};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue},
    response::{IntoResponse, Response},
};
use openai_protocol::messages::{CreateMessageRequest, Message};
use serde_json::Value;
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

/// Relay a successful non-streaming answer as the provider wrote it, for the
/// plain path, which reads nothing of it. The body must be a JSON object: one
/// that is not is the 502 `parse_error`, as when the answer was read typed.
/// Its usage counts go to the token metrics, and its bytes leave under the
/// upstream's status and content type, so a key or a block this build does
/// not model reaches the client as sent.
pub(crate) async fn relay_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
) -> Response {
    let status = response.status();
    let content_type = response.headers().get(CONTENT_TYPE).cloned();
    let read = match response.bytes().await {
        Ok(body) => serde_json::from_slice::<Value>(&body)
            .map_err(|e| e.to_string())
            .and_then(|answer| {
                if answer.is_object() {
                    Ok((body, answer))
                } else {
                    Err("the answer is not a JSON object".to_string())
                }
            }),
        Err(e) => Err(e.to_string()),
    };
    let (body, answer) = match read {
        Ok(read) => read,
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
    let usage = |key: &str| answer["usage"][key].as_u64().unwrap_or(0);
    let (input_tokens, output_tokens) = (usage("input_tokens"), usage("output_tokens"));
    record_success_metrics(model_id, input_tokens, output_tokens, start_time);
    info!(
        model = %model_id,
        message_id = %answer["id"].as_str().unwrap_or_default(),
        input_tokens = %input_tokens,
        output_tokens = %output_tokens,
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
pub(crate) mod probe {
    use metrics_exporter_prometheus::PrometheusHandle;

    /// The recorder's rendered line of `metric` carrying `model`, if any.
    pub(crate) fn series_line(
        handle: &PrometheusHandle,
        metric: &str,
        model: &str,
    ) -> Option<String> {
        let rendered = handle.render();
        let model = format!("model=\"{model}\"");
        rendered
            .lines()
            .find(|line| line.starts_with(&format!("{metric}{{")) && line.contains(&model))
            .map(str::to_string)
    }

    /// The error series the failed request is counted under for `model`, as
    /// `(error_type, reason, count)`, with the number of duration samples.
    pub(crate) fn counted_failure(
        handle: &PrometheusHandle,
        model: &str,
    ) -> (String, String, u64, u64) {
        let errors = series_line(handle, "smg_router_request_errors_total", model)
            .expect("the failed request is counted");
        let label = |name: &str| {
            let key = format!("{name}=\"");
            let start = errors.find(&key).expect(name) + key.len();
            let end = errors[start..].find('"').expect(name) + start;
            errors[start..end].to_string()
        };
        let count = |line: &str| -> u64 {
            line.rsplit(' ')
                .next()
                .expect("a value")
                .parse()
                .expect("a count")
        };
        let durations = series_line(handle, "smg_router_request_duration_seconds_count", model)
            .expect("the failed request has a duration");
        (
            label("error_type"),
            label("reason"),
            count(&errors),
            count(&durations),
        )
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use metrics_exporter_prometheus::PrometheusBuilder;

    use super::{probe::counted_failure, *};
    use crate::test_support::{
        text_of, upstream, RefusedAddress, StubWorker, CREDENTIALED_WORKER_URL, MASKED_WORKER_URL,
    };

    const UPSTREAM_REFUSAL: &str = r#"{"error":{"message":"Rate limit reached for model in organization org-abc123 project proj_xyz789","type":"rate_limit","code":"rate_limit_exceeded"}}"#;

    /// A provider's answer with what this build does not model: a web fetch
    /// result block, the diagnostics, and usage keys beyond the two counts.
    const UPSTREAM_ANSWER: &str = r#"{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"web_fetch_tool_result","tool_use_id":"srvtoolu_1","content":{"type":"web_fetch_result","url":"https://example.com","content":{"type":"document","source":{"type":"text","media_type":"text/plain","data":"page"}}}},{"type":"text","text":"done"}],"stop_reason":"end_turn","stop_sequence":null,"stop_details":null,"diagnostics":{"cache_miss_reason":null},"usage":{"input_tokens":7,"output_tokens":3,"speed":"fast","server_tool_use":{"web_search_requests":0,"web_fetch_requests":1}},"container":null}"#;

    /// The rendered `smg_router_tokens_total` line of `model` for one token type.
    fn tokens_line(
        handle: &metrics_exporter_prometheus::PrometheusHandle,
        model: &str,
        token_type: &str,
    ) -> Option<String> {
        let model = format!("model=\"{model}\"");
        let token_type = format!("token_type=\"{token_type}\"");
        handle
            .render()
            .lines()
            .find(|line| {
                line.starts_with("smg_router_tokens_total{")
                    && line.contains(&model)
                    && line.contains(&token_type)
            })
            .map(str::to_string)
    }

    /// A plain answer leaves as the provider wrote it, under its status and
    /// content type, what this build does not model included, and its token
    /// counts are counted.
    #[tokio::test]
    async fn a_plain_answer_leaves_as_the_provider_wrote_it() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let answer = upstream(200, Some("application/json"), UPSTREAM_ANSWER);

        let response = relay_response(answer, "relayed", Instant::now()).await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .expect("a content type"),
            "application/json"
        );
        assert_eq!(text_of(response).await, UPSTREAM_ANSWER);
        let input = tokens_line(&handle, "relayed", "input").expect("input tokens counted");
        let output = tokens_line(&handle, "relayed", "output").expect("output tokens counted");
        assert!(input.ends_with(" 7"), "{input}");
        assert!(output.ends_with(" 3"), "{output}");
    }

    /// A plain answer that is no JSON object is still the 502 `parse_error`,
    /// counted as a bad response.
    #[tokio::test]
    async fn a_plain_answer_that_is_no_json_object_is_still_a_parse_error() {
        for body in ["not json", "[1, 2]"] {
            let recorder = PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            let _guard = metrics::set_default_local_recorder(&recorder);
            let answer = upstream(200, Some("application/json"), body);

            let response = relay_response(answer, "broken", Instant::now()).await;

            assert_eq!(response.status(), StatusCode::BAD_GATEWAY, "{body}");
            assert!(text_of(response).await.contains("parse_error"), "{body}");
            let (error_type, _, count, _) = counted_failure(&handle, "broken");
            assert_eq!((error_type.as_str(), count), ("parse_error", 1), "{body}");
        }
    }

    /// An answer without a content type is relayed as JSON, which it was
    /// checked to be.
    #[tokio::test]
    async fn a_plain_answer_without_a_content_type_is_relayed_as_json() {
        let response =
            relay_response(upstream(200, None, UPSTREAM_ANSWER), "m", Instant::now()).await;
        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .expect("a content type"),
            "application/json"
        );
    }

    /// A provider's refusal reaches the Messages caller under its status with
    /// the operator's organization and project gone.
    #[tokio::test]
    async fn an_upstream_errors_body_loses_the_operators_identifiers() {
        let refusal = upstream(429, None, UPSTREAM_REFUSAL);
        let response = handle_error_response(refusal, "m", Instant::now()).await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let text = text_of(response).await;
        assert!(!text.contains("org-abc123"), "{text}");
        assert!(!text.contains("proj_xyz789"), "{text}");
        assert!(text.contains("Rate limit reached for model"), "{text}");
    }

    /// A client whose budget for a call is `timeout`, the process TLS crypto
    /// provider installed first.
    #[expect(
        clippy::disallowed_methods,
        reason = "a test binary's client factory; the process TLS crypto provider is installed first"
    )]
    fn client_with_timeout(timeout: Duration) -> reqwest::Client {
        use rustls::crypto::ring::default_provider;
        let _ = default_provider().install_default();
        reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("a client")
    }

    fn request() -> CreateMessageRequest {
        serde_json::from_value(serde_json::json!({
            "model": "m",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "q"}],
        }))
        .expect("a request")
    }

    /// Every line the transport writes about a worker names its address
    /// masked: a worker registered with credentials at a port nobody listens
    /// on fails its send, and neither the request-built line nor the send
    /// lines carry the credentials. The capture is the tracing subscriber
    /// itself, so the rendering is read as a log reader would.
    #[tracing_test::traced_test]
    #[tokio::test]
    async fn the_transports_lines_name_the_worker_masked() {
        let worker = StubWorker::at(CREDENTIALED_WORKER_URL);
        let (url, headers) = build_request(worker.as_ref(), None);
        assert_eq!(url, format!("{CREDENTIALED_WORKER_URL}/v1/messages"));

        let answer = send_request(
            worker.http_client(),
            &url,
            &headers,
            &request(),
            Duration::from_secs(5),
            "m",
            Instant::now(),
        )
        .await;

        let response = answer.expect_err("a failed send");
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert!(!logs_contain("secret"));
        assert!(!logs_contain("user:"));
        assert!(logs_contain("Request built"));
        assert!(logs_contain("Request to worker failed"));
        assert!(logs_contain(&format!("{MASKED_WORKER_URL}/v1/messages")));
    }

    /// A send the request timeout cuts is the caller's 504 and one error
    /// counted against the model under `timeout`/`deadline`, with a duration
    /// sample; the upstream completes the handshake and never answers.
    #[tokio::test]
    async fn a_send_that_times_out_is_counted_as_a_deadline_against_the_model() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let silent = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a port");
        let url = format!(
            "http://{}/v1/messages",
            silent.local_addr().expect("an address")
        );
        let budget = Duration::from_millis(50);

        let answer = send_request(
            &client_with_timeout(budget),
            &url,
            &HeaderMap::new(),
            &request(),
            budget,
            "silent-upstream",
            Instant::now(),
        )
        .await;

        let response = answer.expect_err("a failed send");
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            counted_failure(&handle, "silent-upstream"),
            ("timeout".to_string(), "deadline".to_string(), 1, 1)
        );
    }

    /// A send nothing answers (the port refuses the connection) is the
    /// caller's 502 and one error counted against the model under
    /// `backend_error`/`transport_disconnect`, with a duration sample.
    #[tokio::test]
    async fn a_refused_connection_is_counted_as_a_transport_disconnect_against_the_model() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let refused = RefusedAddress::reserve();
        let url = format!("http://{}/v1/messages", refused.addr());

        let answer = send_request(
            &client_with_timeout(Duration::from_secs(5)),
            &url,
            &HeaderMap::new(),
            &request(),
            Duration::from_secs(5),
            "closed-port",
            Instant::now(),
        )
        .await;

        let response = answer.expect_err("a failed send");
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            counted_failure(&handle, "closed-port"),
            (
                "backend_error".to_string(),
                "transport_disconnect".to_string(),
                1,
                1
            )
        );
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
