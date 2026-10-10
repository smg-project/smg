//! The Messages API's error envelope.
//!
//! Every error answered on `/v1/messages` and `/v1/messages/count_tokens`
//! leaves as `{"type":"error","error":{"type","message"},"request_id"}`,
//! the shape the Anthropic SDKs decode, whatever produced it inside: the
//! gateway's OpenAI-shaped envelope, a plain-text refusal, an empty body. The
//! status is kept and `error.type` follows from it.

use axum::{
    body::{Body, Bytes},
    extract::Request,
    http::{
        header::{CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE, TRANSFER_ENCODING},
        HeaderMap, HeaderValue, StatusCode,
    },
    middleware::Next,
    response::Response,
};
use openai_protocol::messages::{ErrorEnvelope, ErrorResponse};
use serde_json::Value;

use super::request_id::{generate_request_id, RequestId};

/// Error bodies are small; one that is not (an upstream relay) is cut here
/// and the status alone names the error.
const ERROR_BODY_LIMIT: usize = 1 << 20;

/// The routes answered in the Messages API's shape.
pub fn is_messages_route(path: &str) -> bool {
    path == "/v1/messages" || path.starts_with("/v1/messages/")
}

/// A request that expects the Messages API's envelope: one to a Messages route,
/// or one that carries the `anthropic-version` header the Anthropic SDKs send
/// on every call (a model lookup under `/v1/models`, an unknown URL).
pub fn wants_messages_envelope(headers: &HeaderMap, path: &str) -> bool {
    is_messages_route(path) || headers.contains_key("anthropic-version")
}

/// Anthropic's `error.type` for a status: the documented vocabulary (504 is
/// its `timeout_error`), the other 4xx as `invalid_request_error`, the other
/// 5xx as `api_error`.
pub fn messages_error_type(status: StatusCode) -> &'static str {
    match status {
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
        status if status.as_u16() == 529 => "overloaded_error",
        StatusCode::GATEWAY_TIMEOUT => "timeout_error",
        status if status.is_server_error() => "api_error",
        _ => "invalid_request_error",
    }
}

/// Re-shapes every error answered on a Messages route into the Messages
/// API's envelope. Sits outside the route's other middleware so that an
/// authentication, admission or body-timeout refusal is covered as well, and
/// once more at the edge, outside the body-size limit, so a declared
/// over-limit length (refused before any route layer runs) is covered too;
/// a body already in the shape passes the second pass unchanged.
pub async fn messages_error_envelope_middleware(request: Request, next: Next) -> Response {
    if !is_messages_route(request.uri().path()) {
        return next.run(request).await;
    }
    let request_id = request_id_of(&request);
    let response = next.run(request).await;
    if !(response.status().is_client_error() || response.status().is_server_error()) {
        return response;
    }
    into_messages_envelope(response, request_id).await
}

/// The id the request-id middleware assigned (echoed as `x-request-id`), or
/// a fresh one when the layer is absent.
pub fn request_id_of(request: &Request) -> String {
    request.extensions().get::<RequestId>().map_or_else(
        || generate_request_id(request.uri().path()),
        |id| id.0.clone(),
    )
}

/// The response with its body re-shaped into the Messages envelope. Status,
/// headers and extensions stay (the error-code header and metric label, the
/// `WWW-Authenticate` challenge, `Retry-After`); a body that already has the
/// shape keeps its `error` and gains the `request_id` it lacks.
pub async fn into_messages_envelope(response: Response, request_id: String) -> Response {
    let (mut parts, body) = response.into_parts();
    let status = parts.status;
    let bytes = axum::body::to_bytes(body, ERROR_BODY_LIMIT)
        .await
        .unwrap_or_default();
    let error = match serde_json::from_slice::<Value>(&bytes) {
        Ok(json) => match messages_error_of(&json) {
            Some(error) => error,
            None => ErrorResponse {
                error_type: messages_error_type(status).to_string(),
                message: message_of(&json).unwrap_or_else(|| reason(status)),
            },
        },
        Err(_) => ErrorResponse {
            error_type: messages_error_type(status).to_string(),
            message: text_of(&bytes).unwrap_or_else(|| reason(status)),
        },
    };
    let envelope = ErrorEnvelope::new(error, request_id);
    let json = serde_json::to_vec(&envelope).unwrap_or_default();
    parts
        .headers
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    parts
        .headers
        .insert(CONTENT_LENGTH, HeaderValue::from(json.len()));
    // The body is rebuilt as plain JSON: an encoding or framing the inside
    // declared for its own body no longer applies.
    parts.headers.remove(CONTENT_ENCODING);
    parts.headers.remove(TRANSFER_ENCODING);
    Response::from_parts(parts, Body::from(json))
}

/// The `error` of a body that already is a Messages envelope.
fn messages_error_of(json: &Value) -> Option<ErrorResponse> {
    if json.get("type").and_then(Value::as_str) != Some("error") {
        return None;
    }
    let error = json.get("error")?;
    Some(ErrorResponse {
        error_type: error.get("type")?.as_str()?.to_string(),
        message: error.get("message")?.as_str()?.to_string(),
    })
}

/// The message of a JSON error body of any other shape: the gateway's
/// `error.message`, a bare `error` string, a top-level `message`.
fn message_of(json: &Value) -> Option<String> {
    match json.get("error") {
        Some(Value::Object(error)) => match error.get("message") {
            Some(Value::String(message)) => Some(message.clone()),
            Some(other) => Some(other.to_string()),
            None => Some(Value::Object(error.clone()).to_string()),
        },
        Some(Value::String(message)) => Some(message.clone()),
        _ => match json.get("message") {
            Some(Value::String(message)) => Some(message.clone()),
            _ => None,
        },
    }
}

fn text_of(bytes: &Bytes) -> Option<String> {
    let text = std::str::from_utf8(bytes).ok()?.trim();
    (!text.is_empty()).then(|| text.to_string())
}

fn reason(status: StatusCode) -> String {
    status
        .canonical_reason()
        .unwrap_or("Request failed")
        .to_string()
}

#[cfg(test)]
mod tests {
    use axum::{
        http::header::{RETRY_AFTER, WWW_AUTHENTICATE},
        middleware::from_fn,
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };
    use serde_json::json;
    use tower::ServiceExt;

    use super::*;
    use crate::{middleware::RequestIdLayer, routers::error as route_error};

    fn app() -> Router {
        Router::new()
            .route(
                "/v1/messages",
                post(|| async { route_error::bad_request("process_messages_failed", "Failed to apply chat template") }),
            )
            .route(
                "/v1/messages/count_tokens",
                post(|| async { (StatusCode::NOT_FOUND, "No router available for this request").into_response() }),
            )
            .route("/v1/messages/empty", get(|| async { StatusCode::NOT_FOUND }))
            .route(
                "/v1/messages/limited",
                get(|| async {
                    (
                        StatusCode::TOO_MANY_REQUESTS,
                        [(RETRY_AFTER, "2")],
                        Json(json!({"error": {"message": "queue full", "type": "rate_limit", "code": "admission_queue_full"}})),
                    )
                        .into_response()
                }),
            )
            .route(
                "/v1/messages/unauthorized",
                get(|| async {
                    (
                        StatusCode::UNAUTHORIZED,
                        [(WWW_AUTHENTICATE, "Bearer realm=\"data-plane\"")],
                        Json(json!({"error": {"message": "Missing API key", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}})),
                    )
                        .into_response()
                }),
            )
            .route(
                "/v1/messages/relayed",
                get(|| async {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(json!({"type": "error", "error": {"type": "invalid_request_error", "message": "upstream says no"}})),
                    )
                        .into_response()
                }),
            )
            .route("/v1/messages/ok", get(|| async { Json(json!({"id": "msg_1"})) }))
            .route(
                "/v1/messages/encoded",
                get(|| async {
                    (
                        StatusCode::BAD_GATEWAY,
                        [(CONTENT_ENCODING, "gzip")],
                        Json(json!({"error": {"message": "upstream body", "type": "api_error"}})),
                    )
                        .into_response()
                }),
            )
            .route(
                "/v1/chat/completions",
                post(|| async { route_error::bad_request("json_parse_error", "nope") }),
            )
            .route_layer(from_fn(messages_error_envelope_middleware))
            .layer(RequestIdLayer::new(vec!["x-client-request-id".to_string()]))
    }

    async fn call(
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
    ) -> (StatusCode, HeaderMap, Value) {
        let mut request = Request::builder().method(method).uri(path);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        let response = app()
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap();
        let (parts, body) = response.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (parts.status, parts.headers, json)
    }

    fn assert_envelope(json: &Value, error_type: &str, message: &str) {
        assert_eq!(json["type"], "error", "{json}");
        assert_eq!(json["error"]["type"], error_type, "{json}");
        assert_eq!(json["error"]["message"], message, "{json}");
        assert!(
            json["request_id"].as_str().is_some_and(|id| !id.is_empty()),
            "{json}"
        );
        assert_eq!(json.as_object().unwrap().len(), 3, "{json}");
        assert_eq!(json["error"].as_object().unwrap().len(), 2, "{json}");
    }

    #[tokio::test]
    async fn gateway_envelopes_become_messages_envelopes() {
        let (status, headers, json) = call("POST", "/v1/messages", &[]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_envelope(
            &json,
            "invalid_request_error",
            "Failed to apply chat template",
        );
        assert_eq!(headers["content-type"], "application/json");
        assert_eq!(
            headers[route_error::HEADER_X_SMG_ERROR_CODE],
            "process_messages_failed"
        );
        assert_eq!(
            headers["content-length"]
                .to_str()
                .unwrap()
                .parse::<usize>()
                .unwrap(),
            serde_json::to_vec(&json).unwrap().len()
        );
    }

    #[tokio::test]
    async fn plain_text_and_empty_bodies_become_messages_envelopes() {
        let (status, _, json) = call("POST", "/v1/messages/count_tokens", &[]).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_envelope(
            &json,
            "not_found_error",
            "No router available for this request",
        );

        let (status, _, json) = call("GET", "/v1/messages/empty", &[]).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_envelope(&json, "not_found_error", "Not Found");
    }

    #[tokio::test]
    async fn status_headers_and_extensions_survive() {
        let (status, headers, json) = call("GET", "/v1/messages/limited", &[]).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_envelope(&json, "rate_limit_error", "queue full");
        assert_eq!(headers[RETRY_AFTER], "2");

        let (status, headers, json) = call("GET", "/v1/messages/unauthorized", &[]).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_envelope(&json, "authentication_error", "Missing API key");
        assert_eq!(headers[WWW_AUTHENTICATE], "Bearer realm=\"data-plane\"");
    }

    #[tokio::test]
    async fn a_rebuilt_body_carries_no_stale_encoding_headers() {
        let (status, headers, json) = call("GET", "/v1/messages/encoded", &[]).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert_envelope(&json, "api_error", "upstream body");
        assert!(headers.get(CONTENT_ENCODING).is_none(), "{headers:?}");
        assert!(headers.get(TRANSFER_ENCODING).is_none(), "{headers:?}");
    }

    #[tokio::test]
    async fn request_id_is_the_one_echoed_in_the_header() {
        let (_, headers, json) = call("POST", "/v1/messages", &[]).await;
        assert_eq!(
            json["request_id"],
            headers["x-request-id"].to_str().unwrap()
        );

        let (_, headers, json) = call(
            "POST",
            "/v1/messages",
            &[("x-client-request-id", "req_client_1")],
        )
        .await;
        assert_eq!(json["request_id"], "req_client_1");
        assert_eq!(headers["x-request-id"], "req_client_1");
    }

    #[tokio::test]
    async fn a_body_already_in_the_shape_keeps_its_error() {
        let (status, _, json) = call("GET", "/v1/messages/relayed", &[]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_envelope(&json, "invalid_request_error", "upstream says no");
    }

    #[tokio::test]
    async fn successes_and_other_routes_pass_untouched() {
        let (status, _, json) = call("GET", "/v1/messages/ok", &[]).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json, json!({"id": "msg_1"}));

        let (status, _, json) = call("POST", "/v1/chat/completions", &[]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(json.get("type").is_none(), "{json}");
        assert_eq!(json["error"]["code"], "json_parse_error");
    }

    #[test]
    fn error_types_follow_the_documented_vocabulary() {
        for (status, expected) in [
            (StatusCode::BAD_REQUEST, "invalid_request_error"),
            (StatusCode::METHOD_NOT_ALLOWED, "invalid_request_error"),
            (StatusCode::UNPROCESSABLE_ENTITY, "invalid_request_error"),
            (StatusCode::UNAUTHORIZED, "authentication_error"),
            (StatusCode::FORBIDDEN, "permission_error"),
            (StatusCode::NOT_FOUND, "not_found_error"),
            (StatusCode::PAYLOAD_TOO_LARGE, "request_too_large"),
            (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
            (StatusCode::INTERNAL_SERVER_ERROR, "api_error"),
            (StatusCode::NOT_IMPLEMENTED, "api_error"),
            (StatusCode::BAD_GATEWAY, "api_error"),
            (StatusCode::SERVICE_UNAVAILABLE, "overloaded_error"),
            (StatusCode::GATEWAY_TIMEOUT, "timeout_error"),
            (StatusCode::from_u16(529).unwrap(), "overloaded_error"),
        ] {
            assert_eq!(messages_error_type(status), expected, "{status}");
        }
    }

    #[test]
    fn the_shape_is_wanted_by_path_or_by_the_anthropic_header() {
        let mut headers = HeaderMap::new();
        assert!(wants_messages_envelope(&headers, "/v1/messages"));
        assert!(wants_messages_envelope(
            &headers,
            "/v1/messages/count_tokens"
        ));
        assert!(!wants_messages_envelope(&headers, "/v1/models/claude-x"));
        assert!(!wants_messages_envelope(&headers, "/v1/messagesx"));
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        assert!(wants_messages_envelope(&headers, "/v1/models/claude-x"));
    }
}
