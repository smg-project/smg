use axum::{
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Serialize;
use serde_json::Value;

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
/// Reads only the [`GatewayErrorCode`] extension set by [`create_error`],
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
mod tests {
    use super::*;

    /// A reason phrase ("Bad Request", "Not Found") is not a `type` the SDKs
    /// know; the envelope's `type` is the public vocabulary, `code` stays the
    /// gateway's string and `param` is present (null when no field is named).
    #[tokio::test]
    async fn envelope_type_is_the_public_vocabulary_for_every_status_class() {
        for (status, expected) in [
            (StatusCode::BAD_REQUEST, "invalid_request_error"),
            (StatusCode::NOT_FOUND, "invalid_request_error"),
            (StatusCode::METHOD_NOT_ALLOWED, "invalid_request_error"),
            (StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error"),
            (StatusCode::FAILED_DEPENDENCY, "invalid_request_error"),
            (StatusCode::UNAUTHORIZED, "invalid_request_error"),
            (StatusCode::FORBIDDEN, "permission_error"),
            (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
            (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
            (StatusCode::NOT_IMPLEMENTED, "server_error"),
            (StatusCode::BAD_GATEWAY, "server_error"),
            (StatusCode::SERVICE_UNAVAILABLE, "server_error"),
            (StatusCode::GATEWAY_TIMEOUT, "server_error"),
        ] {
            let response = create_error(status, "some_code", "what went wrong");
            assert_eq!(response.status(), status);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let json: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json["error"]["type"], expected, "{status}");
            assert_eq!(json["error"]["code"], "some_code", "{status}");
            assert_eq!(json["error"]["message"], "what went wrong", "{status}");
            assert!(
                json["error"].get("param").is_some_and(Value::is_null),
                "{status}: param must be present and null, got {json}"
            );
        }
    }

    #[test]
    fn model_not_found_is_an_invalid_request_error_with_the_code() {
        let response = model_not_found("gpt-nope");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            extract_error_code_from_response(&response),
            "model_not_found"
        );
        assert_eq!(
            error_type_for_status(response.status()),
            "invalid_request_error"
        );
    }

    #[test]
    fn extract_error_code_reads_gateway_minted_code() {
        let response = create_error(StatusCode::NOT_FOUND, "model_not_found", "nope");

        assert_eq!(
            extract_error_code_from_response(&response),
            "model_not_found"
        );
        // The client-facing header is still set.
        assert_eq!(
            response.headers().get(HEADER_X_SMG_ERROR_CODE).unwrap(),
            "model_not_found"
        );
    }

    #[test]
    fn extract_error_code_ignores_header_without_gateway_extension() {
        // A response without a gateway extension must not mint a metric label,
        // even if another response decorator adds the public header.
        let mut response = StatusCode::INTERNAL_SERVER_ERROR.into_response();
        response.headers_mut().insert(
            HEADER_X_SMG_ERROR_CODE,
            HeaderValue::from_static("upstream-minted-code"),
        );

        assert_eq!(extract_error_code_from_response(&response), "");
    }

    #[test]
    fn extract_error_code_survives_header_tampering() {
        let mut response = create_error(StatusCode::BAD_GATEWAY, "upstream_error", "boom");
        response
            .headers_mut()
            .insert(HEADER_X_SMG_ERROR_CODE, HeaderValue::from_static("forged"));

        assert_eq!(
            extract_error_code_from_response(&response),
            "upstream_error"
        );
    }

    #[test]
    fn test_sanitize_org_id() {
        let body = r#"{"error":{"message":"Rate limit reached for model in organization org-abc123","type":"rate_limit","code":"rate_limit_exceeded"}}"#;
        let result = sanitize_error_body(body);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        let msg = parsed["error"]["message"].as_str().unwrap();
        assert!(!msg.contains("org-"));
        assert!(msg.contains("Rate limit reached for model"));
    }

    #[test]
    fn test_sanitize_project_id() {
        let body = r#"{"error":{"message":"Quota exceeded for project proj_xyz789","type":"insufficient_quota","code":"quota_exceeded"}}"#;
        let result = sanitize_error_body(body);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        let msg = parsed["error"]["message"].as_str().unwrap();
        assert!(!msg.contains("proj_"));
        assert!(msg.contains("Quota exceeded"));
    }

    #[test]
    fn test_sanitize_invalid_image_url() {
        let body = r#"{"error":{"message":"Could not process image at URL https://internal.corp/img.png: connection refused from 10.0.1.5","type":"invalid_request_error","code":"invalid_image_url"}}"#;
        let result = sanitize_error_body(body);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(
            parsed["error"]["message"].as_str().unwrap(),
            "Invalid Image URL"
        );
    }

    #[test]
    fn test_sanitize_both_org_and_project() {
        let body = r#"{"error":{"message":"Rate limit for organization org-abc123 project proj_xyz789","type":"rate_limit","code":"rate_limit_exceeded"}}"#;
        let result = sanitize_error_body(body);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        let msg = parsed["error"]["message"].as_str().unwrap();
        assert!(!msg.contains("org-"));
        assert!(!msg.contains("proj_"));
        assert!(msg.contains("Rate limit for"));
    }

    #[test]
    fn test_sanitize_org_id_at_start() {
        let body = r#"{"error":{"message":"organization org-abc123 exceeded quota","type":"rate_limit","code":"rate_limit_exceeded"}}"#;
        let result = sanitize_error_body(body);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        let msg = parsed["error"]["message"].as_str().unwrap();
        assert!(!msg.contains("org-"));
        assert!(msg.contains("exceeded quota"));
    }

    #[test]
    fn test_sanitize_project_id_at_start() {
        let body = r#"{"error":{"message":"project proj_xyz789 quota exceeded","type":"insufficient_quota","code":"quota_exceeded"}}"#;
        let result = sanitize_error_body(body);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        let msg = parsed["error"]["message"].as_str().unwrap();
        assert!(!msg.contains("proj_"));
        assert!(msg.contains("quota exceeded"));
    }

    #[test]
    fn test_sanitize_non_json_passthrough() {
        let body = "Bad Gateway";
        let result = sanitize_error_body(body);
        assert_eq!(result, "Bad Gateway");
    }

    #[test]
    fn test_sanitize_json_without_error_field() {
        let body = r#"{"status":"ok","data":"hello"}"#;
        let result = sanitize_error_body(body);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["status"].as_str().unwrap(), "ok");
    }
}
