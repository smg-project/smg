use std::fmt::Display;

use axum::{
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use smg_data_connector::StorageFailure;

#[derive(Serialize)]
struct ErrorResponse<'a> {
    error: ErrorDetail<'a>,
}

#[derive(Serialize)]
struct ErrorDetail<'a> {
    #[serde(rename = "type")]
    error_type: &'static str,
    code: &'a str,
    message: &'a str,
    param: Option<String>,
}

pub const HEADER_X_SMG_ERROR_CODE: &str = "X-SMG-Error-Code";

/// Gateway-minted error code, carried as a process-local response extension.
///
/// Keep metrics independent of client-visible headers: only [`create_error`]
/// adds this extension.
#[derive(Clone)]
struct GatewayErrorCode(String);

pub fn internal_error(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::INTERNAL_SERVER_ERROR, code, message)
}

pub fn bad_request(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::BAD_REQUEST, code, message)
}

pub fn not_found(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::NOT_FOUND, code, message)
}

pub fn service_unavailable(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::SERVICE_UNAVAILABLE, code, message)
}

pub fn too_many_requests(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::TOO_MANY_REQUESTS, code, message)
}

pub fn failed_dependency(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::FAILED_DEPENDENCY, code, message)
}

pub fn not_implemented(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::NOT_IMPLEMENTED, code, message)
}

pub fn bad_gateway(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::BAD_GATEWAY, code, message)
}

pub fn gateway_timeout(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::GATEWAY_TIMEOUT, code, message)
}

pub fn method_not_allowed(code: impl Into<String>, message: impl Into<String>) -> Response {
    create_error(StatusCode::METHOD_NOT_ALLOWED, code, message)
}

/// The answer to a failed history store call. A store that is not available
/// (a call that exceeded the store's bound, no connection to be had, the
/// connection gone or timed out, the server shutting down or refusing
/// connections) is a `503 storage_unavailable`: the client may retry and the
/// metric label names the dependency. Any other failure (the store's own
/// refusal, a hook's rejection, a record that cannot be decoded, an invalid
/// chain) keeps the endpoint's own 500 `code`. `context` opens the message.
pub fn storage_failure(
    code: &str,
    context: impl Display,
    error: &(impl StorageFailure + Display),
) -> Response {
    if error.is_unavailable() {
        service_unavailable("storage_unavailable", format!("{context}: {error}"))
    } else {
        internal_error(code, format!("{context}: {error}"))
    }
}

pub fn create_error(
    status: StatusCode,
    code: impl Into<String>,
    message: impl Into<String>,
) -> Response {
    let code_str = code.into();
    let message_str = message.into();

    let mut headers = HeaderMap::with_capacity(1);
    if let Ok(val) = HeaderValue::from_str(&code_str) {
        headers.insert(HEADER_X_SMG_ERROR_CODE, val);
    }

    let mut response = (
        status,
        headers,
        Json(ErrorResponse {
            error: ErrorDetail {
                error_type: error_type_for_status(status),
                code: &code_str,
                message: &message_str,
                param: None,
            },
        }),
    )
        .into_response();
    response.extensions_mut().insert(GatewayErrorCode(code_str));
    response
}

/// The `error.type` of the OpenAI-shaped envelope for a status.
///
/// The public API's vocabulary, which clients switch on to decide between
/// fixing the request, re-authenticating, backing off and retrying: a 4xx is
/// an `invalid_request_error` (the public API files its 404s there too, the
/// detail in `code`, e.g. `model_not_found`, and its 401s, with the code
/// `invalid_api_key`), except the two statuses with a word of their own;
/// every 5xx is a `server_error`.
pub fn error_type_for_status(status: StatusCode) -> &'static str {
    match status {
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        status if status.is_server_error() => "server_error",
        _ => "invalid_request_error",
    }
}

pub fn model_not_found(model: &str) -> Response {
    create_error(
        StatusCode::NOT_FOUND,
        "model_not_found",
        format!("No worker available for model '{model}'"),
    )
}

/// Error-code metric label for a response.
///
/// Reads only the `GatewayErrorCode` extension set by [`create_error`],
/// never the client-visible `X-SMG-Error-Code` header.
pub fn extract_error_code_from_response<B>(response: &Response<B>) -> &str {
    response
        .extensions()
        .get::<GatewayErrorCode>()
        .map_or("", |code| code.0.as_str())
}

#[expect(
    clippy::expect_used,
    reason = "static regex patterns are compile-time constants; invalid pattern is a developer bug"
)]
static ORG_ID_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\s*\borganization org-\S+").expect("static regex pattern is valid")
});
#[expect(
    clippy::expect_used,
    reason = "static regex patterns are compile-time constants; invalid pattern is a developer bug"
)]
static PROJ_ID_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\s*\bproject proj_\S+").expect("static regex pattern is valid"));

/// Sanitize upstream error response bodies to prevent leaking internal identifiers.
/// - Strips org-ID patterns (`org-xxx`)
/// - Strips project-ID patterns (`proj_xxx`)
/// - Replaces `invalid_image_url` error messages
/// - Non-JSON bodies pass through unchanged
pub fn sanitize_error_body(body: &str) -> String {
    let mut json: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return body.to_string(),
    };

    let mut modified = false;

    if let Some(error) = json.get_mut("error").and_then(Value::as_object_mut) {
        if error.get("code").and_then(Value::as_str) == Some("invalid_image_url") {
            error.insert("message".into(), Value::String("Invalid Image URL".into()));
            modified = true;
        } else if let Some(Value::String(msg)) = error.get("message") {
            let sanitized = ORG_ID_RE.replace_all(msg, "");
            let sanitized = PROJ_ID_RE.replace_all(&sanitized, "");
            if sanitized.as_ref() != msg.as_str() {
                error.insert("message".into(), Value::String(sanitized.into_owned()));
                modified = true;
            }
        }
    }

    if modified {
        serde_json::to_string(&json).unwrap_or_else(|_| body.to_string())
    } else {
        body.to_string()
    }
}

#[cfg(test)]
#[path = "../tests/unit/error/tests.rs"]
mod tests;
