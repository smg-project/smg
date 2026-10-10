// Validated JSON extractor for automatic request validation
//
// This module provides a ValidatedJson extractor that automatically validates
// requests using the validator crate's Validate trait.

/// Trait for request types that need post-deserialization normalization
pub trait Normalizable {
    /// Normalize the request by applying defaults and transformations
    fn normalize(&mut self) {
        // Default: no-op
    }
}

#[cfg(feature = "axum")]
use axum::{
    extract::{
        rejection::{BytesRejection, FailedToBufferBody, JsonRejection},
        FromRequest, Request,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
#[cfg(feature = "axum")]
use serde::de::DeserializeOwned;
#[cfg(feature = "axum")]
use serde_json::{json, Value};
#[cfg(feature = "axum")]
use validator::{Validate, ValidationError, ValidationErrors, ValidationErrorsKind};

/// A JSON extractor that automatically validates and normalizes the request body
///
/// This extractor deserializes the request body and automatically calls `.validate()`
/// on types that implement the `Validate` trait. If validation fails, it returns
/// a 400 Bad Request with detailed error information.
///
/// # Example
///
/// ```rust,ignore
/// async fn create_chat(
///     ValidatedJson(request): ValidatedJson<ChatCompletionRequest>,
/// ) -> Response {
///     // request is guaranteed to be valid here
///     process_request(request).await
/// }
/// ```
#[cfg(feature = "axum")]
pub struct ValidatedJson<T>(pub T);

/// The gateway's JSON error envelope for a rejected request body: the public
/// API's shape (`message`, `type`, `param`, `code`), `code` a string and
/// `param` the offending field when one is known.
#[cfg(feature = "axum")]
fn rejection(status: StatusCode, code: &str, param: Option<String>, message: String) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": code
            }
        })),
    )
        .into_response()
}

/// The axum prefix of a body that deserialized into the wrong shape; the rest
/// is serde's message, with the path to the offending value in front when the
/// value is nested (`messages[0].content: invalid type ...`).
#[cfg(feature = "axum")]
const DESERIALIZE_PREFIX: &str = "Failed to deserialize the JSON body into the target type: ";

/// The field a deserialization error is about, as the public API's `param`:
/// the path serde recorded, the field a `missing field` / `unknown field`
/// error names, or both (`items[0].role`); None for a syntax error.
#[cfg(feature = "axum")]
fn param_from_deserialize_message(message: &str) -> Option<String> {
    let detail = message
        .rsplit_once(DESERIALIZE_PREFIX)
        .map_or(message, |(_, detail)| detail);
    let is_path_char =
        |c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '[' | ']' | '-');
    let path = detail
        .split_once(": ")
        .map(|(head, _)| head)
        .filter(|head| !head.is_empty() && head.chars().all(is_path_char));
    let field = ["missing field `", "unknown field `"]
        .iter()
        .find_map(|marker| detail.split_once(marker))
        .and_then(|(_, rest)| rest.split_once('`'))
        .map(|(field, _)| field);
    match (path, field) {
        // A refused unknown key is already the last segment of serde's path.
        (Some(path), Some(field)) if path == field || path.ends_with(&format!(".{field}")) => {
            Some(path.to_string())
        }
        (Some(path), Some(field)) => Some(format!("{path}.{field}")),
        (Some(path), None) => Some(path.to_string()),
        (None, Some(field)) => Some(field.to_string()),
        (None, None) => None,
    }
}

/// The first violation of a validation report, as the public API's
/// (`param`, `code`): the path of the field (`items[1].role`; None for a rule
/// over the whole request) and the code of the rule, validator's built-in
/// rules named the way the public API names them. Keys are visited in sorted
/// order, so the same report always yields the same pair.
#[cfg(feature = "axum")]
fn first_violation(errors: &ValidationErrors) -> (Option<String>, String) {
    fn walk(errors: &ValidationErrors, prefix: &str) -> Option<(Option<String>, String)> {
        let mut entries: Vec<_> = errors.errors().iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (key, kind) in entries {
            let path = match (key.as_ref(), prefix.is_empty()) {
                ("__all__", _) => prefix.to_string(),
                (key, true) => key.to_string(),
                (key, false) => format!("{prefix}.{key}"),
            };
            match kind {
                ValidationErrorsKind::Field(field_errors) => {
                    if let Some(error) = field_errors.first() {
                        let param = (!path.is_empty()).then_some(path);
                        return Some((param, public_code(error)));
                    }
                }
                ValidationErrorsKind::Struct(inner) => {
                    if let Some(found) = walk(inner, &path) {
                        return Some(found);
                    }
                }
                ValidationErrorsKind::List(items) => {
                    for (index, inner) in items {
                        if let Some(found) = walk(inner, &format!("{path}[{index}]")) {
                            return Some(found);
                        }
                    }
                }
            }
        }
        None
    }
    walk(errors, "").unwrap_or_else(|| (None, "invalid_value".to_string()))
}

/// The public API's code for a validation error: a custom rule's own code
/// passes through (`tool_choice_requires_tools`), validator's built-in rules
/// get the public API's names for the same conditions.
#[cfg(feature = "axum")]
fn public_code(error: &ValidationError) -> String {
    match error.code.as_ref() {
        "range" => range_code(error),
        "length" => length_code(error),
        "required" => "missing_required_parameter".to_string(),
        "email"
        | "url"
        | "regex"
        | "contains"
        | "does_not_contain"
        | "must_match"
        | "ip"
        | "credit_card"
        | "phone"
        | "non_control_character" => "invalid_value".to_string(),
        // A rule whose "code" is a sentence ("messages cannot be empty") has
        // its text in the message already; the code stays an identifier.
        code if code.contains(char::is_whitespace) => "invalid_value".to_string(),
        code => code.to_string(),
    }
}

#[cfg(feature = "axum")]
fn bound(error: &ValidationError, name: &str) -> Option<f64> {
    error.params.get(name).and_then(Value::as_f64)
}

/// `integer_below_min_value`, `decimal_above_max_value`, ...: the kind of the
/// value and the bound it crossed.
#[cfg(feature = "axum")]
fn range_code(error: &ValidationError) -> String {
    let value = error.params.get("value");
    let kind = match value {
        Some(value) if value.is_i64() || value.is_u64() => "integer",
        _ => "decimal",
    };
    match value.and_then(Value::as_f64) {
        Some(v)
            if bound(error, "min").is_some_and(|min| v < min)
                || bound(error, "exclusive_min").is_some_and(|min| v <= min) =>
        {
            format!("{kind}_below_min_value")
        }
        Some(v)
            if bound(error, "max").is_some_and(|max| v > max)
                || bound(error, "exclusive_max").is_some_and(|max| v >= max) =>
        {
            format!("{kind}_above_max_value")
        }
        _ => "invalid_value".to_string(),
    }
}

/// `string_above_max_length`, `empty_array`, `object_above_max_properties`,
/// ...: the kind of the value and the bound its length crossed.
#[cfg(feature = "axum")]
fn length_code(error: &ValidationError) -> String {
    let (kind, unit, len) = match error.params.get("value") {
        Some(Value::String(s)) => ("string", "length", Some(s.chars().count())),
        Some(Value::Array(a)) => ("array", "length", Some(a.len())),
        Some(Value::Object(o)) => ("object", "properties", Some(o.len())),
        _ => ("value", "length", None),
    };
    let min = bound(error, "min");
    let max = bound(error, "max");
    match len {
        Some(0) if kind == "array" && min.is_some_and(|min| min >= 1.0) => {
            "empty_array".to_string()
        }
        Some(len) if min.is_some_and(|min| (len as f64) < min) => {
            format!("{kind}_below_min_{unit}")
        }
        Some(len) if max.is_some_and(|max| (len as f64) > max) => {
            format!("{kind}_above_max_{unit}")
        }
        _ => "invalid_value".to_string(),
    }
}

#[cfg(feature = "axum")]
impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    T: DeserializeOwned + Validate + Normalizable + Send,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        // First, extract and deserialize the JSON
        let Json(mut data) =
            Json::<T>::from_request(req, state)
                .await
                .map_err(|err: JsonRejection| {
                    // A body that crosses the payload limit is a size error,
                    // not a client JSON error: a declared Content-Length over
                    // the limit is already refused with 413 before the body,
                    // and a chunked upload must get the same answer the moment
                    // it crosses the limit (reading stops there; the frames
                    // after it are never pulled in).
                    if let JsonRejection::BytesRejection(BytesRejection::FailedToBufferBody(
                        FailedToBufferBody::LengthLimitError(_),
                    )) = &err
                    {
                        return rejection(
                            StatusCode::PAYLOAD_TOO_LARGE,
                            "request_body_too_large",
                            None,
                            "Request body exceeded the payload limit (max_payload_size)"
                                .to_string(),
                        );
                    }
                    let error_message = match err {
                        JsonRejection::JsonDataError(e) => {
                            format!("Invalid JSON data: {e}")
                        }
                        JsonRejection::JsonSyntaxError(e) => {
                            format!("JSON syntax error: {e}")
                        }
                        JsonRejection::MissingJsonContentType(_) => {
                            "Missing Content-Type: application/json header".to_string()
                        }
                        _ => format!("Failed to parse JSON: {err}"),
                    };
                    let param = param_from_deserialize_message(&error_message);
                    rejection(
                        StatusCode::BAD_REQUEST,
                        "json_parse_error",
                        param,
                        error_message,
                    )
                })?;

        // Normalize the request (apply defaults based on other fields)
        data.normalize();

        // Then, automatically validate the data
        data.validate().map_err(|validation_errors| {
            let (param, code) = first_violation(&validation_errors);
            rejection(
                StatusCode::BAD_REQUEST,
                &code,
                param,
                validation_errors.to_string(),
            )
        })?;

        Ok(ValidatedJson(data))
    }
}

// Implement Deref to allow transparent access to the inner value
#[cfg(feature = "axum")]
impl<T> std::ops::Deref for ValidatedJson<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(feature = "axum")]
impl<T> std::ops::DerefMut for ValidatedJson<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(all(test, feature = "axum"))]
mod tests {
    use axum::body::Body;
    use serde::{Deserialize, Serialize};
    use validator::Validate;

    use super::*;

    #[derive(Debug, Deserialize, Serialize, Validate)]
    struct TestRequest {
        #[validate(range(min = 0.0, max = 1.0))]
        value: f32,
        #[validate(length(min = 1))]
        name: String,
    }

    impl Normalizable for TestRequest {
        // Use default no-op implementation
    }

    #[tokio::test]
    async fn test_validated_json_valid() {
        // This test is conceptual - actual testing would require Axum test harness
        let request = TestRequest {
            value: 0.5,
            name: "test".to_string(),
        };
        assert!(request.validate().is_ok());
    }

    #[tokio::test]
    async fn test_validated_json_invalid_range() {
        let request = TestRequest {
            value: 1.5, // Out of range
            name: "test".to_string(),
        };
        assert!(request.validate().is_err());
    }

    #[tokio::test]
    async fn test_validated_json_invalid_length() {
        let request = TestRequest {
            value: 0.5,
            name: String::new(), // Empty name
        };
        assert!(request.validate().is_err());
    }

    // The envelope of a rejected body, through the extractor itself.

    #[derive(Debug, Deserialize, Serialize, Validate)]
    struct Item {
        #[validate(length(min = 1))]
        role: String,
    }

    /// A plain field whose type refuses unknown keys: serde's path to such a
    /// key already ends with it, so the param must not repeat it.
    #[derive(Debug, Deserialize, Serialize, Validate)]
    #[serde(deny_unknown_fields)]
    struct Dependency {
        #[validate(length(min = 1))]
        version: String,
    }

    fn not_vip(tier: &str) -> Result<(), ValidationError> {
        if tier == "vip" {
            return Err(ValidationError::new("unsupported_value"));
        }
        if tier == "gold" {
            return Err(ValidationError::new("tier must not be gold"));
        }
        Ok(())
    }

    fn not_both(request: &BodyRequest) -> Result<(), ValidationError> {
        if request.both {
            return Err(ValidationError::new("mutually_exclusive_parameters"));
        }
        Ok(())
    }

    #[derive(Debug, Deserialize, Serialize, Validate)]
    #[validate(schema(function = "not_both"))]
    struct BodyRequest {
        #[validate(range(min = 0.0, max = 1.0))]
        value: f32,
        #[validate(length(min = 1, max = 8))]
        name: String,
        #[serde(default)]
        #[validate(range(min = 1, max = 20))]
        count: Option<u32>,
        #[serde(default)]
        #[validate(length(min = 1))]
        tags: Option<Vec<String>>,
        #[serde(default)]
        #[validate(length(max = 2))]
        metadata: Option<std::collections::HashMap<String, String>>,
        #[serde(default)]
        #[validate(nested)]
        items: Vec<Item>,
        #[serde(default)]
        #[validate(nested)]
        dependency: Option<Dependency>,
        #[serde(default)]
        #[validate(custom(function = "not_vip"))]
        tier: Option<String>,
        #[serde(default)]
        both: bool,
    }

    impl Normalizable for BodyRequest {}

    /// The status and body of a rejected body; an accepted body shows up as
    /// a 200 with a null body (and fails the caller's assertions).
    async fn reject(body: &str) -> (StatusCode, Value) {
        let request = Request::builder()
            .method("POST")
            .uri("/")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap_or_default();
        let response = match ValidatedJson::<BodyRequest>::from_request(request, &()).await {
            Ok(_) => return (StatusCode::OK, Value::Null),
            Err(response) => response,
        };
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap_or_default();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    /// (status, type, code, param) of a rejection; `code` must be a string
    /// and `param` present (a string or null), as in the public API. A
    /// field of the wrong shape is reported in place, so the assertion that
    /// follows names it.
    async fn envelope(body: &str) -> (StatusCode, String, String, Option<String>) {
        let (status, json) = reject(body).await;
        let error = json["error"].as_object().cloned().unwrap_or_default();
        let code = match error.get("code") {
            Some(Value::String(code)) => code.clone(),
            other => format!("<code must be a string, got {other:?}>"),
        };
        let param = match error.get("param") {
            Some(Value::Null) => None,
            Some(Value::String(param)) => Some(param.clone()),
            other => Some(format!(
                "<param must be present as a string or null, got {other:?}>"
            )),
        };
        let kind = error
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("<type must be a string>")
            .to_string();
        assert!(
            error
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|m| !m.is_empty()),
            "{json}"
        );
        (status, kind, code, param)
    }

    #[tokio::test]
    async fn range_violations_name_the_field_and_the_bound_crossed() {
        for (body, code, param) in [
            (
                r#"{"value": 1.5, "name": "x"}"#,
                "decimal_above_max_value",
                "value",
            ),
            (
                r#"{"value": -0.5, "name": "x"}"#,
                "decimal_below_min_value",
                "value",
            ),
            (
                r#"{"value": 0.5, "name": "x", "count": 0}"#,
                "integer_below_min_value",
                "count",
            ),
            (
                r#"{"value": 0.5, "name": "x", "count": 25}"#,
                "integer_above_max_value",
                "count",
            ),
        ] {
            assert_eq!(
                envelope(body).await,
                (
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error".to_string(),
                    code.to_string(),
                    Some(param.to_string())
                ),
                "{body}"
            );
        }
    }

    #[tokio::test]
    async fn length_violations_name_the_kind_of_value() {
        for (body, code, param) in [
            (
                r#"{"value": 0.5, "name": ""}"#,
                "string_below_min_length",
                "name",
            ),
            (
                r#"{"value": 0.5, "name": "far too long"}"#,
                "string_above_max_length",
                "name",
            ),
            (
                r#"{"value": 0.5, "name": "x", "tags": []}"#,
                "empty_array",
                "tags",
            ),
            (
                r#"{"value": 0.5, "name": "x", "metadata": {"a": "1", "b": "2", "c": "3"}}"#,
                "object_above_max_properties",
                "metadata",
            ),
        ] {
            let (_, _, got_code, got_param) = envelope(body).await;
            assert_eq!(
                (got_code.as_str(), got_param.as_deref()),
                (code, Some(param)),
                "{body}"
            );
        }
    }

    #[tokio::test]
    async fn nested_violations_carry_the_full_path() {
        let body = r#"{"value": 0.5, "name": "x", "items": [{"role": "user"}, {"role": ""}]}"#;
        let (_, _, code, param) = envelope(body).await;
        assert_eq!(code, "string_below_min_length");
        assert_eq!(param.as_deref(), Some("items[1].role"));
    }

    #[tokio::test]
    async fn custom_rule_codes_pass_through_unless_they_are_sentences() {
        let body = r#"{"value": 0.5, "name": "x", "tier": "vip"}"#;
        let (_, _, code, param) = envelope(body).await;
        assert_eq!(code, "unsupported_value");
        assert_eq!(param.as_deref(), Some("tier"));

        let body = r#"{"value": 0.5, "name": "x", "tier": "gold"}"#;
        let (_, _, code, param) = envelope(body).await;
        assert_eq!(code, "invalid_value");
        assert_eq!(param.as_deref(), Some("tier"));
    }

    #[tokio::test]
    async fn rules_over_the_whole_request_have_no_param() {
        let body = r#"{"value": 0.5, "name": "x", "both": true}"#;
        let (_, _, code, param) = envelope(body).await;
        assert_eq!(code, "mutually_exclusive_parameters");
        assert_eq!(param, None);
    }

    #[tokio::test]
    async fn deserialization_errors_name_the_offending_field() {
        for (body, param) in [
            (r#"{"value": 0.5}"#, "name"),
            (r#"{"value": "high", "name": "x"}"#, "value"),
            (
                r#"{"value": 0.5, "name": "x", "items": [{"nope": 1}]}"#,
                "items[0].role",
            ),
        ] {
            let (status, kind, code, got_param) = envelope(body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(kind, "invalid_request_error", "{body}");
            assert_eq!(code, "json_parse_error", "{body}");
            assert_eq!(got_param.as_deref(), Some(param), "{body}");
        }
    }

    /// `Dependency` refuses unknown keys, and serde's path to such a key
    /// already ends with it (`dependency.nope`): the param names it once.
    #[tokio::test]
    async fn an_unknown_key_of_a_nested_object_is_named_once() {
        let body = r#"{"value": 0.5, "name": "x", "dependency": {"version": "1", "nope": true}}"#;
        let (status, kind, code, param) = envelope(body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(kind, "invalid_request_error");
        assert_eq!(code, "json_parse_error");
        assert_eq!(param.as_deref(), Some("dependency.nope"));
    }

    #[tokio::test]
    async fn syntax_errors_have_no_param() {
        let (status, json) = reject(r#"{"value": "#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "json_parse_error");
        assert!(json["error"]["param"].is_null(), "{json}");
        assert!(
            json["error"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with("JSON syntax error")),
            "{json}"
        );
    }

    #[test]
    fn param_is_read_from_serde_paths_and_field_names() {
        for (message, param) in [
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: missing field `messages` at line 1 column 15",
                Some("messages"),
            ),
            (
                "Failed to deserialize the JSON body into the target type: messages[0]: unknown field `foo`, expected one of `role`, `content` at line 1 column 40",
                Some("messages[0].foo"),
            ),
            (
                "Failed to deserialize the JSON body into the target type: dependency.nope: unknown field `nope`, expected `version` at line 1 column 58",
                Some("dependency.nope"),
            ),
            (
                "Failed to deserialize the JSON body into the target type: frobnicate: unknown field `frobnicate`, expected one of `model`, `input` at line 1 column 44",
                Some("frobnicate"),
            ),
            (
                "Failed to deserialize the JSON body into the target type: messages[0].content: data did not match any variant of untagged enum Content at line 1 column 60",
                Some("messages[0].content"),
            ),
            (
                "Failed to deserialize the JSON body into the target type: invalid type: integer `5`, expected a string at line 1 column 10",
                None,
            ),
            ("JSON syntax error: expected value at line 1 column 1", None),
            ("Missing Content-Type: application/json header", None),
        ] {
            assert_eq!(param_from_deserialize_message(message).as_deref(), param, "{message}");
        }
    }
}
