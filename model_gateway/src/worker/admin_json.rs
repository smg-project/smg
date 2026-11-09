//! The JSON body extractor of the control-plane management endpoints (the
//! worker, WASM module and tokenizer routes): a body the type does not take
//! is refused in the control plane's own error envelope (`{"error": ...,
//! "code": ...}`, the shape its 400s already use), not with the framework's
//! plain-text 422, so a client reads one shape from these endpoints whatever
//! went wrong. [`refusal`] renders the same envelope for a handler's own
//! refusals.

use axum::{
    extract::{
        rejection::{BytesRejection, FailedToBufferBody, JsonRejection},
        FromRequest, Request,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::worker::WorkerErrorResponse;
use serde::de::DeserializeOwned;

/// A JSON body of the control-plane management endpoints.
pub(crate) struct AdminJson<T>(pub T);

impl<S, T> FromRequest<S> for AdminJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(body)) => Ok(Self(body)),
            Err(rejection) => Err(refuse(rejection)),
        }
    }
}

/// The envelope for a body the extractor refused: the status the condition
/// has on the data plane (413 past the payload limit, 400 for everything the
/// client can fix) and the control plane's codes.
fn refuse(rejection: JsonRejection) -> Response {
    let (status, code, error) = match rejection {
        JsonRejection::BytesRejection(BytesRejection::FailedToBufferBody(
            FailedToBufferBody::LengthLimitError(_),
        )) => (
            StatusCode::PAYLOAD_TOO_LARGE,
            "PAYLOAD_TOO_LARGE",
            "Request body exceeded the payload limit (max_payload_size)".to_string(),
        ),
        JsonRejection::JsonDataError(e) => (
            StatusCode::BAD_REQUEST,
            "BAD_REQUEST",
            format!("Invalid JSON data: {e}"),
        ),
        JsonRejection::JsonSyntaxError(e) => (
            StatusCode::BAD_REQUEST,
            "BAD_REQUEST",
            format!("JSON syntax error: {e}"),
        ),
        JsonRejection::MissingJsonContentType(_) => (
            StatusCode::BAD_REQUEST,
            "BAD_REQUEST",
            "Missing Content-Type: application/json header".to_string(),
        ),
        other => (
            StatusCode::BAD_REQUEST,
            "BAD_REQUEST",
            format!("Failed to parse JSON: {other}"),
        ),
    };
    refusal(status, code, error)
}

/// The control plane's refusal envelope: `status` with `{"error", "code"}`,
/// the shape the worker routes answer, for a handler's own refusal.
pub(crate) fn refusal(status: StatusCode, code: &str, error: String) -> Response {
    (
        status,
        Json(WorkerErrorResponse {
            error,
            code: code.to_string(),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{header::CONTENT_TYPE, Request, StatusCode},
        routing::post,
        Router,
    };
    use openai_protocol::worker::{WorkerErrorResponse, WorkerSpec};
    use tower::ServiceExt;

    use super::*;

    fn app() -> Router {
        Router::new().route(
            "/workers",
            post(|AdminJson(_): AdminJson<WorkerSpec>| async { "ok" }),
        )
    }

    async fn send(body: &'static str, content_type: Option<&'static str>) -> (StatusCode, String) {
        let mut request = Request::builder().method("POST").uri("/workers");
        if let Some(content_type) = content_type {
            request = request.header(CONTENT_TYPE, content_type);
        }
        let response = app()
            .oneshot(request.body(Body::from(body)).expect("request"))
            .await
            .expect("response");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The envelope of a refused body: `code` and `error`, the error naming
    /// what was wrong.
    fn envelope(body: &str, text: &str) -> WorkerErrorResponse {
        serde_json::from_str(text)
            .unwrap_or_else(|e| panic!("{body}: not the control plane's envelope ({e}): {text}"))
    }

    /// A body the type does not take (an unknown variant, a wrong type, a
    /// negative or out-of-range integer, an array, broken JSON) is a 400 in
    /// the control plane's envelope with the deserializer's reason; a body
    /// the type takes reaches the handler.
    #[tokio::test]
    async fn a_body_the_type_does_not_take_is_a_400_in_the_control_plane_envelope() {
        for (body, detail) in [
            (
                r#"{"url":"grpc://w:1","worker_type":"bogus"}"#,
                "unknown variant `bogus`",
            ),
            (r#"{"url":"grpc://w:1","cost":"abc"}"#, "cost"),
            (r#"{"url":"grpc://w:1","priority":-1}"#, "priority"),
            (
                r#"{"url":"grpc://w:1","bootstrap_port":70000}"#,
                "bootstrap_port",
            ),
            (r"[1, 2]", "invalid type"),
            (r"{not json", "syntax"),
        ] {
            let (status, text) = send(body, Some("application/json")).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {text}");
            let refusal = envelope(body, &text);
            assert_eq!(refusal.code, "BAD_REQUEST", "{body}");
            assert!(refusal.error.contains(detail), "{body}: {}", refusal.error);
        }
        let (status, text) = send(
            r#"{"url":"grpc://w:1","worker_type":"regular"}"#,
            Some("application/json"),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");
    }

    /// `refusal` renders a handler's own refusal in the same envelope.
    #[tokio::test]
    async fn a_handler_refusal_carries_the_status_the_code_and_the_reason() {
        let response = refusal(
            StatusCode::NOT_FOUND,
            "WORKER_NOT_FOUND",
            "no such worker".to_string(),
        );
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let refusal = envelope("refusal", &String::from_utf8_lossy(&bytes));
        assert_eq!(refusal.code, "WORKER_NOT_FOUND");
        assert_eq!(refusal.error, "no such worker");
    }

    /// A body without the JSON content type is refused in the same envelope.
    #[tokio::test]
    async fn a_body_without_the_json_content_type_is_a_400_in_the_envelope() {
        let body = r#"{"url":"grpc://w:1"}"#;
        let (status, text) = send(body, None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
        let refusal = envelope(body, &text);
        assert_eq!(refusal.code, "BAD_REQUEST");
        assert!(refusal.error.contains("Content-Type"), "{}", refusal.error);
    }
}
