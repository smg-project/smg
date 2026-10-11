//! The body-limit layer's refusal in the gateway's error envelope.
//!
//! `tower_http::limit::RequestBodyLimitLayer` refuses a declared
//! `Content-Length` over `--max-payload-size` before any route layer runs,
//! with a plain-text 413 (`length limit exceeded`); a chunked upload that
//! crosses the limit is refused by the JSON extractor with the envelope and
//! the code `request_body_too_large`. This pass, applied outside the limit
//! layer, gives the declared-length refusal the same envelope, so a client
//! parsing `error.code` and the `error_code` metric label see one class for
//! one cause whatever the framing. The Messages routes re-shape it once more
//! into their own envelope, as they do every error.

use axum::{
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
    middleware::Next,
    response::Response,
};

use crate::routers::error::create_error;

/// The error code of a request body over the payload limit, the extractor's
/// code for the chunked framing.
pub const REQUEST_BODY_TOO_LARGE: &str = "request_body_too_large";

/// A 413 without a JSON body (the limit layer's plain text) leaves as the
/// gateway's envelope with [`REQUEST_BODY_TOO_LARGE`]; every other response
/// passes unchanged.
pub async fn payload_too_large_envelope_middleware(request: Request, next: Next) -> Response {
    let response = next.run(request).await;
    if response.status() != StatusCode::PAYLOAD_TOO_LARGE || has_json_body(&response) {
        return response;
    }
    create_error(
        StatusCode::PAYLOAD_TOO_LARGE,
        REQUEST_BODY_TOO_LARGE,
        "Request body exceeded the payload limit (max_payload_size)",
    )
}

fn has_json_body(response: &Response) -> bool {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|content_type| content_type.starts_with("application/json"))
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
    use crate::routers::error::extract_error_code_from_response;

    fn app() -> Router {
        Router::new()
            .route("/ok", post(|| async { StatusCode::OK }))
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

    /// A 413 that already carries a JSON body (a handler's own) and any other
    /// status pass unchanged.
    #[tokio::test]
    async fn json_413s_and_other_statuses_pass_unchanged() {
        let response = send("/json-413", 4).await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "custom", "{json}");

        let response = send("/ok", 4).await;
        assert_eq!(response.status(), StatusCode::OK);
    }
}
