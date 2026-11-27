//! The JSON body extractor of the control-plane management endpoints (the
//! worker, WASM module, tokenizer, parse and profile routes; the profile
//! routes take it through `OptionalFromRequest`, a request without a
//! `Content-Type` header reading as no body): a body the type does not take
//! is refused in the control plane's own error envelope (`{"error": ...,
//! "code": ...}`, the shape its 400s already use), not with the framework's
//! plain-text 422, so an extractor refusal reads in one shape with the
//! handlers' own refusals that [`refusal`] renders (the parse handlers'
//! own errors keep their older `{"error", "success"}` body).

use axum::{
    extract::{
        rejection::{BytesRejection, FailedToBufferBody, JsonRejection},
        FromRequest, OptionalFromRequest, Request,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::{
    validated::{
        content_encoding_message, undecodable_content_encoding, with_accept_encoding_identity,
    },
    worker::WorkerErrorResponse,
};
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
        if let Some(encoding) = undecodable_content_encoding(&req) {
            return Err(unsupported_encoding(&encoding));
        }
        match <Json<T> as FromRequest<S>>::from_request(req, state).await {
            Ok(Json(body)) => Ok(Self(body)),
            Err(rejection) => Err(refuse(rejection)),
        }
    }
}

/// The optional form, for a route whose body may be absent (the profile
/// routes take their defaults then): no `Content-Type` header is no body,
/// as the framework reads it; a body that is present is read as above and
/// refused in the envelope when the type does not take it.
impl<S, T> OptionalFromRequest<S> for AdminJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Option<Self>, Self::Rejection> {
        if let Some(encoding) = undecodable_content_encoding(&req) {
            return Err(unsupported_encoding(&encoding));
        }
        match <Json<T> as OptionalFromRequest<S>>::from_request(req, state).await {
            Ok(Some(Json(body))) => Ok(Some(Self(body))),
            Ok(None) => Ok(None),
            Err(rejection) => Err(refuse(rejection)),
        }
    }
}

/// The control plane's 415 for a body under a `Content-Encoding` the
/// gateway does not decode, answered before the body is read, with
/// `Accept-Encoding: identity`.
fn unsupported_encoding(encoding: &str) -> Response {
    with_accept_encoding_identity(refusal(
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        "UNSUPPORTED_MEDIA_TYPE",
        content_encoding_message(encoding),
    ))
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
#[path = "../../tests/unit/admin/json/tests.rs"]
mod tests;
