//! The body-limit layer's refusal in the gateway's error envelope.
//!
//! `tower_http::limit::RequestBodyLimitLayer` refuses a declared
//! `Content-Length` over `--max-payload-size` before any layer inside it
//! runs, with a plain-text 413 (`length limit exceeded`); a chunked upload
//! that crosses the limit is refused by the JSON extractor with the envelope
//! and the code `request_body_too_large`. Two passes give the declared-length
//! refusal the same envelope, and only that refusal: an inner pass marks every
//! response that came back through the limit layer, and an outer pass
//! re-shapes a 413 that carries no mark. A 413 minted inside (a handler's own
//! limit, an upstream's page relayed by the HTTP router, the extractor's
//! chunked refusal) is marked and left as it is.

use axum::{extract::Request, http::StatusCode, middleware::Next, response::Response};

use super::body_timeout::request_body_too_large_response;

/// Set on every response that came back through the body-limit layer; a 413
/// without it never got past that layer, so it is the layer's own refusal.
#[derive(Clone, Copy, Debug)]
pub struct PassedBodyLimit;

/// The inner pass (applied inside the body-limit layer): marks the response.
pub async fn mark_passed_body_limit_middleware(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    response.extensions_mut().insert(PassedBodyLimit);
    response
}

/// The outer pass (applied outside the body-limit layer): the limit layer's
/// own 413 leaves as the gateway's envelope with the extractor's code and the
/// metric label; every marked response passes unchanged.
pub async fn payload_too_large_envelope_middleware(request: Request, next: Next) -> Response {
    let response = next.run(request).await;
    if response.status() != StatusCode::PAYLOAD_TOO_LARGE
        || response.extensions().get::<PassedBodyLimit>().is_some()
    {
        return response;
    }
    request_body_too_large_response()
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{to_bytes, Body},
        http::{
            header::{CONTENT_LENGTH, CONTENT_TYPE},
            Request, StatusCode,
        },
        middleware::from_fn,
        response::IntoResponse,
        routing::post,
        Json, Router,
    };
    use serde_json::{json, Value};
    use tower::ServiceExt;
    use tower_http::limit::RequestBodyLimitLayer;

    use super::*;
    use crate::{
        middleware::body_timeout::REQUEST_BODY_TOO_LARGE,
        routers::error::extract_error_code_from_response,
    };

    /// The edge stack of the server: the mark inside the limit layer, the
    /// envelope outside it.
    fn app() -> Router {
        Router::new()
            .route("/ok", post(|| async { StatusCode::OK }))
            .route(
                "/text-413",
                post(|| async {
                    // an upstream's page relayed as is, or a handler's own limit
                    (StatusCode::PAYLOAD_TOO_LARGE, "Request Entity Too Large").into_response()
                }),
            )
            .route(
                "/json-413",
                post(|| async {
                    (
                        StatusCode::PAYLOAD_TOO_LARGE,
                        Json(json!({"error": {"code": "custom", "message": "mine"}})),
                    )
                        .into_response()
                }),
            )
            .layer(from_fn(mark_passed_body_limit_middleware))
            .layer(RequestBodyLimitLayer::new(16))
            .layer(from_fn(payload_too_large_envelope_middleware))
    }

    async fn send(uri: &str, declared: usize) -> Response {
        app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header(CONTENT_TYPE, "application/json")
                    .header(CONTENT_LENGTH, declared)
                    .body(Body::from("x".repeat(declared)))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    /// The limit layer's plain-text 413 leaves as the envelope, with the code
    /// on the response for the metric label.
    #[tokio::test]
    async fn declared_over_limit_length_gets_the_envelope() {
        let response = send("/ok", 100).await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            extract_error_code_from_response(&response),
            REQUEST_BODY_TOO_LARGE
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], REQUEST_BODY_TOO_LARGE, "{json}");
        assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
    }

    /// A 413 minted inside the limit layer, plain text (an upstream's page,
    /// a handler's own limit) or JSON, and any other status pass unchanged.
    #[tokio::test]
    async fn responses_from_inside_the_limit_layer_pass_unchanged() {
        let response = send("/text-413", 4).await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(extract_error_code_from_response(&response), "");
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(&body[..], b"Request Entity Too Large");

        let response = send("/json-413", 4).await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "custom", "{json}");

        let response = send("/ok", 4).await;
        assert_eq!(response.status(), StatusCode::OK);
    }
}
