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

/// A store call that exceeded its bound is answered 503 `storage_unavailable`
/// (a `server_error`), any other store failure with the endpoint's own 500
/// code; the message opens with the endpoint's context.
#[tokio::test]
async fn a_timed_out_store_call_is_answered_as_unavailable() {
    use smg_data_connector::ResponseStorageError;

    for (error, status, code) in [
        (
            ResponseStorageError::Timeout(
                "the postgres get_response call exceeded its bound of 5 s".to_string(),
            ),
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_unavailable",
        ),
        (
            ResponseStorageError::Unavailable(
                "no connection from the pool: error connecting to server: Connection refused"
                    .to_string(),
            ),
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_unavailable",
        ),
        // The store answered (a constraint, a refused statement, a hook's
        // refusal): the endpoint's own failure, not an outage to retry.
        (
            ResponseStorageError::StorageError("db error".to_string()),
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
        ),
        (
            ResponseStorageError::Rejected("tenant".to_string()),
            StatusCode::INTERNAL_SERVER_ERROR,
            "storage_error",
        ),
    ] {
        let response = storage_failure("storage_error", "Failed to get response", &error);
        assert_eq!(response.status(), status);
        assert_eq!(
            response.headers().get(HEADER_X_SMG_ERROR_CODE).unwrap(),
            code
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "server_error");
        assert_eq!(json["error"]["code"], code);
        assert_eq!(
            json["error"]["message"],
            format!("Failed to get response: {error}")
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
