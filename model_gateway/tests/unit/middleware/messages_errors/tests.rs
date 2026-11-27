use axum::{
    http::header::{RETRY_AFTER, WWW_AUTHENTICATE},
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
        .route_layer(MessagesErrorEnvelopeLayer)
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
