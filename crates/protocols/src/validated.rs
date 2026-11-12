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
use std::borrow::Cow;

#[cfg(feature = "axum")]
use axum::{
    body::Bytes,
    extract::{
        path::ErrorKind,
        rejection::{
            BytesRejection, FailedToBufferBody, JsonRejection, PathRejection, QueryRejection,
        },
        FromRequest, FromRequestParts, Path, Query, Request,
    },
    http::{header::CONTENT_TYPE, request::Parts, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
#[cfg(feature = "axum")]
use serde::de::DeserializeOwned;
#[cfg(feature = "axum")]
use serde_json::{json, Value};
#[cfg(feature = "axum")]
use validator::{Validate, ValidationError, ValidationErrors, ValidationErrorsKind};

#[cfg(feature = "axum")]
use crate::common::FIELD_PARAM;

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

/// The `error.code` of a body this extractor refused, carried as a response
/// extension: the gateway's HTTP metrics read it when a response did not come
/// from the gateway's own error builder, so the `error_code` label records the
/// code the client sees (`json_parse_error`, `integer_below_min_value`, ...).
/// The codes are the extractor's own fixed set, never client input.
#[cfg(feature = "axum")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RejectionCode(pub String);

/// The gateway's JSON error envelope for a rejected request body: the public
/// API's shape (`message`, `type`, `param`, `code`), `code` a string and
/// `param` the offending field when one is known; the code also rides on the
/// response as a [`RejectionCode`] extension.
#[cfg(feature = "axum")]
pub(crate) fn rejection(
    status: StatusCode,
    code: &str,
    param: Option<String>,
    message: String,
) -> Response {
    let mut response = (
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
        .into_response();
    response
        .extensions_mut()
        .insert(RejectionCode(code.to_string()));
    response
}

/// The axum prefix of a body that deserialized into the wrong shape; the rest
/// is serde's message, with the path to the offending value in front when the
/// value is nested (`messages[0].content: invalid type ...`).
#[cfg(feature = "axum")]
const DESERIALIZE_PREFIX: &str = "Failed to deserialize the JSON body into the target type: ";

/// The words serde opens a reason with, in the order the public API's `code`
/// is read from them (the longer `invalid value: integer `-` before `invalid
/// value`): `missing_required_parameter` for a field that is not there,
/// `unknown_parameter` for one the type does not know, `invalid_value` for a
/// value outside an enumeration, `integer_below_min_value` for a negative
/// integer where the field cannot be negative, `invalid_type` for a value of
/// the wrong shape (a body that is not an object included; a JSON array
/// where an object was expected is a wrong type too, whether serde reports
/// the type or the length of the sequence).
#[cfg(feature = "axum")]
const SERDE_REASONS: &[(&str, &str)] = &[
    ("missing field `", "missing_required_parameter"),
    ("unknown field `", "unknown_parameter"),
    ("unknown variant `", "invalid_value"),
    ("invalid value: integer `-", "integer_below_min_value"),
    ("invalid value", "invalid_value"),
    ("invalid type", "invalid_type"),
    ("invalid length", "invalid_type"),
    ("data did not match any variant", "invalid_type"),
];

/// Whether `text` opens with one of serde's reasons.
#[cfg(feature = "axum")]
fn opens_a_reason(text: &str) -> bool {
    SERDE_REASONS
        .iter()
        .any(|(words, _)| text.starts_with(words))
}

/// Serde's message without axum's prefix (split at the prefix's first copy:
/// axum's comes before serde's text, a client's value after it), as the path
/// serde puts in front of a nested value (`messages[0].content`) and the
/// reason. The path is the first `: ` split's head when that head reads as a
/// path (field names and indexes); the reason is the rest, serde's own words
/// or a custom error's, and a value a custom error repeats cannot move the
/// split, whatever it says. A key that is no plain path (a space, a colon, a
/// letter outside ASCII) is a path all the same: the reason is then found by
/// serde's own words, at the first split whose remainder opens with one
/// (serde quotes the value it refused after its words, never before). This
/// order reads two cases wrong: a key with a colon and a space in it, whose
/// part before the colon reads as a path, is cut at the colon whatever
/// follows; and a key that is no plain path, in front of a custom error,
/// keeps no path.
#[cfg(feature = "axum")]
fn deserialize_detail(message: &str) -> (Option<&str>, &str) {
    let detail = message
        .split_once(DESERIALIZE_PREFIX)
        .map_or(message, |(_, detail)| detail);
    if opens_a_reason(detail) {
        return (None, detail);
    }
    let mut splits = detail
        .match_indices(": ")
        .map(|(at, _)| (&detail[..at], &detail[at + 2..]));
    let is_path_char =
        |c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '[' | ']' | '-');
    match splits.next() {
        Some((head, reason)) if !head.is_empty() && head.chars().all(is_path_char) => {
            (Some(head), reason)
        }
        Some(first) => std::iter::once(first)
            .chain(splits)
            .find(|(_, reason)| opens_a_reason(reason))
            .map_or((None, detail), |(head, reason)| (Some(head), reason)),
        None => (None, detail),
    }
}

/// The public API's `code` for a body the deserializer refused: the code of
/// the first of [`SERDE_REASONS`] its reason opens with (see
/// [`deserialize_detail`]), `json_parse_error` for anything else.
#[cfg(feature = "axum")]
fn deserialize_code(message: &str) -> &'static str {
    let (_, reason) = deserialize_detail(message);
    SERDE_REASONS
        .iter()
        .find(|(words, _)| reason.starts_with(words))
        .map_or("json_parse_error", |(_, code)| code)
}

/// The field a deserialization error is about, as the public API's `param`:
/// the path serde recorded, the field a `missing field` / `unknown field`
/// reason names, or both (`items[0].role`); None for a syntax error.
#[cfg(feature = "axum")]
fn param_from_deserialize_message(message: &str) -> Option<String> {
    let (path, reason) = deserialize_detail(message);
    let field = ["missing field `", "unknown field `"]
        .iter()
        .find_map(|marker| reason.strip_prefix(marker))
        .and_then(|rest| rest.split_once('`'))
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

/// The key the validator crate files a rule over the whole request under
/// (and a rule over one nested value, under that value's path). It is the
/// crate's own and names no field of the public API: it is never printed
/// and never a `param`.
#[cfg(feature = "axum")]
const WHOLE_REQUEST_KEY: &str = "__all__";

/// `key` under `prefix`: the key alone at the top, else `prefix.key`.
#[cfg(feature = "axum")]
fn join(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

/// The path of a report entry: a field's key under its prefix; the prefix
/// itself for a rule over the whole value.
#[cfg(feature = "axum")]
fn path_of(prefix: &str, key: &str) -> String {
    if key == WHOLE_REQUEST_KEY {
        prefix.to_string()
    } else {
        join(prefix, key)
    }
}

/// The report's entries in sorted key order, so that the same report always
/// reads the same way.
#[cfg(feature = "axum")]
fn sorted_entries(errors: &ValidationErrors) -> Vec<(&Cow<'static, str>, &ValidationErrorsKind)> {
    let mut entries: Vec<_> = errors.errors().iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries
}

/// The first violation of a validation report, as the public API's
/// (`param`, `code`): the path of the field (`items[1].role`; the field a
/// rule checked across fields names for itself, `FIELD_PARAM`; None for a
/// rule over the whole request that names none) and the code of the rule,
/// validator's built-in rules named the way the public API names them. Keys
/// are visited in sorted order, so the same report always yields the same
/// pair.
#[cfg(feature = "axum")]
fn first_violation(errors: &ValidationErrors) -> (Option<String>, String) {
    fn walk(errors: &ValidationErrors, prefix: &str) -> Option<(Option<String>, String)> {
        for (key, kind) in sorted_entries(errors) {
            let path = path_of(prefix, key);
            match kind {
                ValidationErrorsKind::Field(field_errors) => {
                    if let Some(error) = field_errors.first() {
                        let param = match error.params.get(FIELD_PARAM).and_then(Value::as_str) {
                            Some(field) => Some(join(&path, field)),
                            None => (!path.is_empty()).then_some(path),
                        };
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

/// The refusal's message for a validation report: the violations each on a
/// line of their own, in the order [`first_violation`] visits them (the
/// first line is the violation whose `param` and `code` the envelope
/// carries); `field: message` for a field's rule, a nested field by its path
/// (`items[1].role: ...`), and the message alone for a rule over the whole
/// request, whose key in the report is the validator crate's own
/// (`WHOLE_REQUEST_KEY`) and no field of the public API.
#[cfg(feature = "axum")]
fn message_of(errors: &ValidationErrors) -> String {
    fn walk(out: &mut String, errors: &ValidationErrors, prefix: &str) {
        for (key, kind) in sorted_entries(errors) {
            let path = path_of(prefix, key);
            match kind {
                ValidationErrorsKind::Field(field_errors) => {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    if !path.is_empty() {
                        out.push_str(&path);
                        out.push_str(": ");
                    }
                    for (index, error) in field_errors.iter().enumerate() {
                        if index > 0 {
                            out.push_str(", ");
                        }
                        out.push_str(&error.to_string());
                    }
                }
                ValidationErrorsKind::Struct(inner) => walk(out, inner, &path),
                ValidationErrorsKind::List(items) => {
                    for (index, inner) in items {
                        walk(out, inner, &format!("{path}[{index}]"));
                    }
                }
            }
        }
    }
    let mut out = String::new();
    walk(&mut out, errors, "");
    out
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
        // A rule whose "code" is a sentence ("query cannot be empty") has
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

/// Whether the request declares the plain `application/json` content type,
/// bare or with well-formed `name=value` parameters (a token name; a token
/// or a non-empty quoted string as the value; a parameter may follow its `;`
/// after spaces, nothing else): a strict subset of what axum's extractor
/// accepts, parsed directly. A `+json` suffix, a parameter the check doubts,
/// whitespace where the MIME grammar has none, or anything else is left to
/// axum's extractor, which decides as before.
#[cfg(feature = "axum")]
fn has_plain_json_content_type(req: &Request) -> bool {
    let Some(value) = req
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let mut parts = value.split(';');
    let mime = parts.next().unwrap_or("");
    mime.eq_ignore_ascii_case("application/json") && parts.all(is_mime_parameter)
}

/// `name=value` after the spaces that may follow the `;`: a token name and a
/// token or non-empty quoted-string value, with no whitespace of its own.
#[cfg(feature = "axum")]
fn is_mime_parameter(part: &str) -> bool {
    fn is_token(text: &str) -> bool {
        !text.is_empty()
            && text
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte))
    }
    let Some((name, value)) = part.trim_start_matches(' ').split_once('=') else {
        return false;
    };
    // A quoted value: non-empty, printable ASCII and spaces only (no tab, no
    // control character, no inner quote, no backslash escape), the part of
    // the grammar axum's parser and this check read alike.
    let quoted = value.len() >= 3
        && value.starts_with('"')
        && value.ends_with('"')
        && value[1..value.len() - 1]
            .bytes()
            .all(|byte| (b' '..=b'~').contains(&byte) && byte != b'"' && byte != b'\\');
    is_token(name) && (is_token(value) || quoted)
}

/// The gateway's answer to a body axum's extractor refused.
#[cfg(feature = "axum")]
fn json_rejection(err: JsonRejection) -> Response {
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
            "Request body exceeded the payload limit (max_payload_size)".to_string(),
        );
    }
    let (code, error_message) = match err {
        JsonRejection::JsonDataError(e) => {
            let message = format!("Invalid JSON data: {e}");
            (deserialize_code(&message), message)
        }
        JsonRejection::JsonSyntaxError(e) => ("invalid_json", format!("JSON syntax error: {e}")),
        JsonRejection::MissingJsonContentType(_) => (
            "json_parse_error",
            "Missing Content-Type: application/json header".to_string(),
        ),
        _ => ("json_parse_error", format!("Failed to parse JSON: {err}")),
    };
    let param = param_from_deserialize_message(&error_message);
    rejection(StatusCode::BAD_REQUEST, code, param, error_message)
}

#[cfg(feature = "axum")]
impl<S, T> FromRequest<S> for ValidatedJson<T>
where
    T: DeserializeOwned + Validate + Normalizable + Send,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        // A body under the plain `application/json` content type is read and
        // parsed once, directly; axum's extractor parses it a second time only
        // when that fails, so that the rejection names the path of the failing
        // field as before. (axum's extractor always parses with path tracking,
        // a chain of allocations per nested value: 118 of a tool-heavy chat
        // request's 2,652 allocation calls, measured 2026-10-21.) Any other
        // content type goes to axum's extractor as a whole, which decides what
        // it accepts and how it refuses.
        let Json(mut data) = if has_plain_json_content_type(&req) {
            let bytes = Bytes::from_request(req, state)
                .await
                .map_err(|err| json_rejection(JsonRejection::from(err)))?;
            match serde_json::from_slice::<T>(&bytes) {
                Ok(data) => Json(data),
                Err(_) => Json::<T>::from_bytes(&bytes).map_err(json_rejection)?,
            }
        } else {
            Json::<T>::from_request(req, state)
                .await
                .map_err(json_rejection)?
        };

        // Normalize the request (apply defaults based on other fields)
        data.normalize();

        // Then, automatically validate the data
        data.validate().map_err(|validation_errors| {
            let (param, code) = first_violation(&validation_errors);
            rejection(
                StatusCode::BAD_REQUEST,
                &code,
                param,
                message_of(&validation_errors),
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

/// A query string a route reads as `T` (the Conversations list route, the
/// realtime routes): a value the type does not take is refused in the
/// envelope, with the code the body extractor gives the same serde reason
/// (`unknown_parameter`, `invalid_type`, ...) and `invalid_value` for a
/// value that does not parse, where the framework answers text/plain.
#[cfg(feature = "axum")]
pub struct ValidatedQuery<T>(pub T);

#[cfg(feature = "axum")]
impl<S, T> FromRequestParts<S> for ValidatedQuery<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        match Query::<T>::from_request_parts(parts, state).await {
            Ok(Query(value)) => Ok(Self(value)),
            Err(err) => Err(query_rejection(&err)),
        }
    }
}

/// A path segment a route reads as `T` (`/v1/models/{*id}`, the
/// Conversations and Responses ids): a segment that is not UTF-8 or does not
/// parse as the type is refused in the envelope as `invalid_value`, `param`
/// the segment's name, where the framework answers text/plain. A rejection
/// the route itself causes (no such parameter, a shape the extractor cannot
/// fill) keeps the framework's answer: a 500 of the route's making, never a
/// client's.
#[cfg(feature = "axum")]
pub struct ValidatedPath<T>(pub T);

#[cfg(feature = "axum")]
impl<S, T> FromRequestParts<S> for ValidatedPath<T>
where
    T: DeserializeOwned + Send,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        match Path::<T>::from_request_parts(parts, state).await {
            Ok(Path(value)) => Ok(Self(value)),
            Err(err) => Err(path_rejection(err)),
        }
    }
}

/// The framework's prefix on a query string the type did not take.
#[cfg(feature = "axum")]
const QUERY_PREFIX: &str = "Failed to deserialize query string: ";

#[cfg(feature = "axum")]
fn query_rejection(err: &QueryRejection) -> Response {
    let message = err.body_text();
    let reason = message.strip_prefix(QUERY_PREFIX).unwrap_or(&message);
    // The body extractor's code for the same serde reason; its fallback
    // names JSON parsing, which a query string never had: a value that does
    // not parse is `invalid_value`.
    let code = match deserialize_code(reason) {
        "json_parse_error" => "invalid_value",
        code => code,
    };
    let param = param_from_deserialize_message(reason);
    rejection(StatusCode::BAD_REQUEST, code, param, message)
}

#[cfg(feature = "axum")]
fn path_rejection(err: PathRejection) -> Response {
    match err {
        PathRejection::FailedToDeserializePathParams(failed)
            if failed.status() == StatusCode::BAD_REQUEST =>
        {
            let param = match failed.kind() {
                ErrorKind::InvalidUtf8InPathParam { key }
                | ErrorKind::ParseErrorAtKey { key, .. }
                | ErrorKind::DeserializeError { key, .. } => Some(key.clone()),
                _ => None,
            };
            rejection(
                StatusCode::BAD_REQUEST,
                "invalid_value",
                param,
                failed.body_text(),
            )
        }
        other => other.into_response(),
    }
}

#[cfg(all(test, feature = "axum"))]
mod tests {
    use axum::body::Body;
    use serde::{Deserialize, Serialize};
    use validator::Validate;

    use super::*;
    use crate::chat::{ChatCompletionRequest, ChatMessage};

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
        #[serde(default)]
        kind: Option<Kind>,
    }

    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum Kind {
        Plain,
        Fancy,
    }

    impl Normalizable for BodyRequest {}

    /// The status and body of a rejected body; an accepted body shows up as
    /// a 200 with a null body (and fails the caller's assertions).
    async fn reject(body: &str) -> (StatusCode, Value) {
        rejected::<BodyRequest>(body).await
    }

    /// The body of a request under `content_type`, through the extractor.
    async fn extract(content_type: &str, body: &str) -> Result<BodyRequest, (StatusCode, Value)> {
        let request = Request::builder()
            .method("POST")
            .uri("/")
            .header("content-type", content_type)
            .body(Body::from(body.to_string()))
            .unwrap_or_default();
        match ValidatedJson::<BodyRequest>::from_request(request, &()).await {
            Ok(ValidatedJson(body)) => Ok(body),
            Err(response) => {
                let status = response.status();
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap_or_default();
                Err((
                    status,
                    serde_json::from_slice(&bytes).unwrap_or(Value::Null),
                ))
            }
        }
    }

    /// A good body under `application/json` is parsed once, directly; one that
    /// fails is parsed again by axum's extractor, whose rejection names the
    /// failing field as before; a content type that is not the plain one goes
    /// to axum's extractor as a whole.
    #[tokio::test]
    async fn bodies_parse_once_and_a_failure_keeps_its_named_path() {
        let good = r#"{"value":0.5,"name":"x"}"#;
        assert!(extract("application/json", good).await.is_ok());
        assert!(
            extract("application/json; charset=utf-8", good)
                .await
                .is_ok(),
            "parameters do not change the content type"
        );
        for malformed in [
            "application/json; charset",
            "application/json;;",
            "application/json; a=b=c\"",
            "application/json ;charset=utf-8",
            "application/json;\tcharset=utf-8",
            "application/json; charset=utf-8 ; x=y",
            "application/json; foo=\"\"",
            "application/json; foo=\"a\tb\"",
            "application/json; foo=\"a\\\"b\"",
        ] {
            let (status, body) = extract(malformed, good).await.unwrap_err();
            assert_eq!(status, StatusCode::BAD_REQUEST, "{malformed}");
            assert!(
                body.to_string().contains("Missing Content-Type"),
                "a parameter the check doubts goes to axum's extractor, which refuses it as before: {malformed}: {body}"
            );
        }
        assert!(
            extract("application/vnd.api+json", good).await.is_ok(),
            "a +json type is accepted through axum's extractor"
        );
        let (status, body) = extract("text/plain", good).await.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.to_string().contains("Missing Content-Type"), "{body}");
        let (status, body) = extract("application/json", r#"{"value":"half","name":"x"}"#)
            .await
            .unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let text = body.to_string();
        assert!(
            text.contains("value") && text.contains("invalid type"),
            "the second parse names the failing field: {text}"
        );
        let (status, body) = extract("application/json", r#"{"value":0.5,"name":"#)
            .await
            .unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.to_string().contains("syntax"), "{body}");
    }

    /// A rejection carries its code as the `RejectionCode` extension (the
    /// gateway's metrics read it), the same code the envelope shows.
    #[tokio::test]
    async fn rejections_carry_their_code_as_an_extension() {
        for (body, code) in [
            (r#"{"value":0.5,"name":"x","tier":"vi"#, "invalid_json"),
            (r#"{"value":1.5,"name":"x"}"#, "decimal_above_max_value"),
        ] {
            let request = Request::builder()
                .method("POST")
                .uri("/")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap_or_default();
            let response = ValidatedJson::<BodyRequest>::from_request(request, &())
                .await
                .err()
                .expect("rejected");
            assert_eq!(
                response.extensions().get::<RejectionCode>(),
                Some(&RejectionCode(code.to_string())),
                "{body}"
            );
        }
    }

    /// A Messages body whose `thinking.display` is outside the documented
    /// set is refused in place: `code` `invalid_value`, `param` the path
    /// serde recorded, a message that names the field and lists the set.
    /// One with a documented value goes through.
    #[tokio::test]
    async fn a_thinking_display_outside_its_set_is_refused_by_name() {
        use crate::messages::{CreateMessageRequest, ThinkingConfig, ThinkingDisplay};

        let extract = |display: &str| {
            let body = json!({
                "model": "m", "max_tokens": 64,
                "thinking": {"type": "adaptive", "display": display},
                "messages": [{"role": "user", "content": "hi"}]
            })
            .to_string();
            let request = Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap_or_default();
            ValidatedJson::<CreateMessageRequest>::from_request(request, &())
        };
        let Ok(ValidatedJson(accepted)) = extract("updates").await else {
            panic!("`updates` is refused");
        };
        assert!(matches!(
            accepted.thinking,
            Some(ThinkingConfig::Adaptive {
                display: Some(ThinkingDisplay::Updates)
            })
        ));
        let Err(refused) = extract("full").await else {
            panic!("`full` is accepted");
        };
        assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(refused.into_body(), usize::MAX)
            .await
            .unwrap_or_default();
        let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        assert_eq!(json["error"]["code"], "invalid_value", "{json}");
        assert_eq!(json["error"]["param"], "thinking", "{json}");
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains(
                "thinking: unknown variant `full` for `thinking.display`, expected one of `summarized`, `omitted`, `updates`"
            ),
            "{message}"
        );
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
        for (body, code, param) in [
            (r#"{"value": 0.5}"#, "missing_required_parameter", "name"),
            (r#"{"value": "high", "name": "x"}"#, "invalid_type", "value"),
            (
                r#"{"value": 0.5, "name": "x", "items": [{"nope": 1}]}"#,
                "missing_required_parameter",
                "items[0].role",
            ),
        ] {
            let (status, kind, got_code, got_param) = envelope(body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(kind, "invalid_request_error", "{body}");
            assert_eq!(got_code, code, "{body}");
            assert_eq!(got_param.as_deref(), Some(param), "{body}");
        }
    }

    /// The code names the deserializer's reason, as the public API's does:
    /// a wrong type, a value outside an enumeration, a negative integer
    /// where the field cannot be negative, a body that is not an object.
    #[tokio::test]
    async fn deserialization_errors_carry_the_public_api_code() {
        for (body, code, param) in [
            (
                r#"{"value": 0.5, "name": "x", "kind": "odd"}"#,
                "invalid_value",
                Some("kind"),
            ),
            (
                r#"{"value": 0.5, "name": "x", "count": -1}"#,
                "integer_below_min_value",
                Some("count"),
            ),
            (
                r#"{"value": 0.5, "name": "x", "tags": "a"}"#,
                "invalid_type",
                Some("tags"),
            ),
            ("[]", "invalid_type", None),
        ] {
            let (status, kind, got_code, got_param) = envelope(body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(kind, "invalid_request_error", "{body}");
            assert_eq!(got_code, code, "{body}");
            assert_eq!(got_param.as_deref(), param, "{body}");
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
        assert_eq!(code, "unknown_parameter");
        assert_eq!(param.as_deref(), Some("dependency.nope"));
    }

    /// A Messages content block the gateway cannot read is refused by its
    /// position and its own reason, which the envelope's `param` and `code`
    /// follow: the `param` is serde's path, ending in the tag's key or the
    /// missing field, as the module names fields (the unknown `type` is a
    /// value, not a shape).
    #[tokio::test]
    async fn an_unreadable_messages_block_is_named_by_position() {
        use crate::messages::CreateMessageRequest;

        let refusal = |body: String| async move {
            let request = Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap_or_default();
            let Err(refused) =
                ValidatedJson::<CreateMessageRequest>::from_request(request, &()).await
            else {
                panic!("accepted");
            };
            assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
            let bytes = axum::body::to_bytes(refused.into_body(), usize::MAX)
                .await
                .unwrap_or_default();
            serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null)
        };
        let json = refusal(
            json!({
                "model": "m", "max_tokens": 8,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "user", "content": [{"type": "container_upload", "file_id": "file_1"}]}
                ]
            })
            .to_string(),
        )
        .await;
        assert_eq!(json["error"]["code"], "invalid_value", "{json}");
        assert_eq!(
            json["error"]["param"], "messages[1].content[0].type",
            "{json}"
        );
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.starts_with(
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: messages[1].content[0].type: unknown variant `container_upload`, expected one of `text`, `image`"
            ),
            "{message}"
        );
        let json = refusal(
            json!({
                "model": "m", "max_tokens": 8,
                "messages": [{"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}]
            })
            .to_string(),
        )
        .await;
        assert_eq!(
            json["error"]["code"], "missing_required_parameter",
            "{json}"
        );
        assert_eq!(
            json["error"]["param"], "messages[0].content[0].tool_use_id",
            "{json}"
        );
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains("messages[0].content[0]: missing field `tool_use_id`"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn syntax_errors_have_no_param() {
        let (status, json) = reject(r#"{"value": "#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "invalid_json");
        assert!(json["error"]["param"].is_null(), "{json}");
        assert!(
            json["error"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with("JSON syntax error")),
            "{json}"
        );
    }

    /// A request whose rule checked across fields names its field
    /// (`invalid_at`), and one whose rule over the whole request names none.
    #[derive(Debug, Deserialize, Serialize, Validate)]
    #[validate(schema(function = "crossed"))]
    struct CrossRequest {
        #[serde(default)]
        base: bool,
        #[serde(default)]
        top: bool,
        #[serde(default)]
        whole: bool,
        #[serde(default)]
        #[validate(range(min = 1, max = 3))]
        level: Option<u32>,
    }

    impl Normalizable for CrossRequest {}

    fn crossed(request: &CrossRequest) -> Result<(), ValidationError> {
        if request.top && !request.base {
            return Err(crate::common::invalid_at(
                "top",
                "top_requires_base",
                "top is only allowed when base is set",
            ));
        }
        if request.whole {
            return Err(crate::common::invalid(
                "whole_request_refused",
                "the request as a whole is refused",
            ));
        }
        Ok(())
    }

    /// The status and body of a rejected body of type `T`.
    async fn rejected<T>(body: &str) -> (StatusCode, Value)
    where
        T: DeserializeOwned + Validate + Normalizable + Send,
    {
        let request = Request::builder()
            .method("POST")
            .uri("/")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap_or_default();
        let response = match ValidatedJson::<T>::from_request(request, &()).await {
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

    /// A rule checked across fields names its field, which the envelope
    /// reports as `param`; a rule over the whole request that names none
    /// keeps `param` null; neither prints the key the validator crate files
    /// them under. A field's rule prints `field: message` as before, a nested
    /// one by its path, several violations each on a line in key order.
    #[tokio::test]
    async fn request_level_refusals_name_their_field_and_print_no_internal_key() {
        let (status, json) = rejected::<CrossRequest>(r#"{"top": true}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "top_requires_base", "{json}");
        assert_eq!(json["error"]["param"], "top", "{json}");
        assert_eq!(
            json["error"]["message"], "top is only allowed when base is set",
            "{json}"
        );

        let (status, json) = rejected::<CrossRequest>(r#"{"whole": true}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["code"], "whole_request_refused", "{json}");
        assert!(json["error"]["param"].is_null(), "{json}");
        assert_eq!(
            json["error"]["message"], "the request as a whole is refused",
            "{json}"
        );

        let (_, json) = rejected::<CrossRequest>(r#"{"level": 9}"#).await;
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.starts_with("level: Validation error: range"),
            "{json}"
        );
        assert_eq!(json["error"]["param"], "level", "{json}");

        let (_, json) = reject(r#"{"value": 0.5, "name": "x", "items": [{"role": ""}]}"#).await;
        let message = json["error"]["message"].as_str().unwrap_or_default();
        assert!(message.starts_with("items[0].role: "), "{json}");

        let (_, json) = reject(r#"{"value": 1.5, "name": ""}"#).await;
        let message = json["error"]["message"].as_str().unwrap_or_default();
        let lines: Vec<&str> = message.lines().collect();
        assert_eq!(lines.len(), 2, "{json}");
        assert!(lines[0].starts_with("name: "), "{json}");
        assert!(lines[1].starts_with("value: "), "{json}");
        assert_eq!(json["error"]["param"], "name", "{json}");
    }

    /// The Chat Completions rules checked across fields name their field;
    /// a provider profile's rule keeps its code and message with nothing in
    /// front; the Messages rules print the public API's words alone.
    #[tokio::test]
    async fn the_public_requests_refusals_name_the_field_without_the_internal_key() {
        use crate::{chat::ChatCompletionRequest, messages::CreateMessageRequest};

        for (body, code, param, message) in [
            (
                r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"top_logprobs":2}"#,
                "top_logprobs_requires_logprobs",
                "top_logprobs",
                "top_logprobs is only allowed when logprobs is enabled",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream_options":{"include_usage":true}}"#,
                "stream_options_requires_stream",
                "stream_options",
                "The 'stream_options' parameter is only allowed when 'stream' is enabled",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tool_choice":"required"}"#,
                "tool_choice_requires_tools",
                "tool_choice",
                "Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":{"type":"function","function":{"name":"nope"}}}"#,
                "tool_choice_function_not_found",
                "tool_choice",
                "Invalid value for 'tool_choice': function 'nope' not found in 'tools'.",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":[]}}]}"#,
                "invalid_type",
                "tools[0].function.parameters",
                "Invalid 'tools[0].function.parameters': expected an object, but got an array instead",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"regex":"a+","ebnf":"root ::= \"a\""}"#,
                "multiple_constraints",
                "ebnf",
                "only one structured output constraint (regex, ebnf, or json_schema) can be active at a time",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"regex":"a+","response_format":{"type":"structural_tag","structures":[],"triggers":[]}}"#,
                "regex_conflicts_with_json",
                "regex",
                "cannot use regex constraint with a JSON or structural_tag response format",
            ),
            (
                r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"ebnf":"root ::= \"a\"","response_format":{"type":"structural_tag","structures":[],"triggers":[]}}"#,
                "ebnf_conflicts_with_json",
                "ebnf",
                "cannot use EBNF constraint with a JSON or structural_tag response format",
            ),
        ] {
            let (status, json) = rejected::<ChatCompletionRequest>(body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(json["error"]["code"], code, "{json}");
            assert_eq!(json["error"]["param"], param, "{json}");
            assert_eq!(json["error"]["message"], message, "{json}");
        }

        let body = r#"{"model":"kimi-k2","messages":[{"role":"root","content":"hi"}]}"#;
        let (status, json) = rejected::<ChatCompletionRequest>(body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
        assert_eq!(json["error"]["code"], "invalid_role", "{json}");
        assert_eq!(json["error"]["message"], "invalid role: root", "{json}");
        assert!(json["error"]["param"].is_null(), "{json}");

        let body = r#"{"model":"m","max_tokens":8,"temperature":1.5,"messages":[{"role":"user","content":"hi"}]}"#;
        let (status, json) = rejected::<CreateMessageRequest>(body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
        assert_eq!(json["error"]["code"], "value_out_of_range", "{json}");
        assert_eq!(
            json["error"]["message"], "temperature: range: 0..1",
            "{json}"
        );
    }

    #[test]
    fn deserialize_code_follows_the_reason_in_the_message() {
        for (message, code) in [
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: missing field `model` at line 1 column 14",
                "missing_required_parameter",
            ),
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: frobnicate: unknown field `frobnicate`, expected one of `include`, `input` at line 1 column 44",
                "unknown_parameter",
            ),
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: include[0]: unknown variant `bogus.field`, expected one of `code_interpreter_call.outputs` at line 1 column 60",
                "invalid_value",
            ),
            // A custom error that opens with serde's words takes their code:
            // a `thinking.display` outside its set is refused as a value.
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: thinking: unknown variant `full` for `thinking.display`, expected one of `summarized`, `omitted`, `updates` at line 1 column 71",
                "invalid_value",
            ),
            // Both reasoning keys with different text are refused as a value too.
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: messages[1]: invalid value: `reasoning_content` and `reasoning` disagree; send one of them, or the same text in both at line 1 column 312",
                "invalid_value",
            ),
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: max_output_tokens: invalid value: integer `-1`, expected u32 at line 1 column 49",
                "integer_below_min_value",
            ),
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: messages: invalid type: string \"hi\", expected a sequence at line 1 column 29",
                "invalid_type",
            ),
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: input: data did not match any variant of untagged enum ResponseInput at line 1 column 71",
                "invalid_type",
            ),
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: invalid type: sequence, expected struct ChatCompletionRequest at line 1 column 0",
                "invalid_type",
            ),
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: invalid length 0, expected struct BodyRequest with 11 elements at line 1 column 2",
                "invalid_type",
            ),
            ("Missing Content-Type: application/json header", "json_parse_error"),
            // A value that repeats serde's words does not steer the code: the
            // reason is read at its start, after the path.
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: reasoning_effort: unknown variant `missing field `x`, expected one of `low`, `medium`, `high` at line 1 column 60",
                "invalid_value",
            ),
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: messages: invalid type: string \"an invalid value\", expected a sequence at line 1 column 42",
                "invalid_type",
            ),
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: invalid type: string \"unknown field `x`\", expected struct ChatCompletionRequest at line 1 column 20",
                "invalid_type",
            ),
            // Nor does a value that repeats axum's prefix: the prefix is split
            // at its first copy.
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: truncation: unknown variant `Failed to deserialize the JSON body into the target type: missing field `x``, expected `auto` or `disabled` at line 1 column 110",
                "invalid_value",
            ),
            // A key with a space, a colon or a letter outside ASCII is a path
            // like any other: the reason is found by serde's words.
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: max tokens: unknown field `max tokens`, expected one of `model`, `input` at line 1 column 19",
                "unknown_parameter",
            ),
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: températur: unknown field `températur`, expected one of `model`, `input` at line 1 column 15",
                "unknown_parameter",
            ),
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: metadata.user id: invalid type: integer `5`, expected a string at line 1 column 40",
                "invalid_type",
            ),
            // A custom error opens with none of serde's words: no code of
            // the table, as before.
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: reasoning_effort: reasoning_effort must be a string, number, or null at line 1 column 42",
                "json_parse_error",
            ),
            // A custom error that repeats the client's value does not hand
            // the value the code: the reason starts after the path.
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: function_call: unknown function_call value: \"x: invalid type\" at line 1 column 44",
                "json_parse_error",
            ),
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: function_call: unknown function_call value: \"x: missing field `y`\" at line 1 column 48",
                "json_parse_error",
            ),
            // A key this order reads wrong: a colon and a space in it, after a
            // part that reads as a path, is cut at the colon whatever follows;
            // here what follows the cut is no reason of serde's.
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: a: b: unknown field `a: b`, expected one of `model`, `input` at line 1 column 12",
                "json_parse_error",
            ),
        ] {
            assert_eq!(deserialize_code(message), code, "{message}");
        }
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
            // A field inside a tagged enum keeps the enum's path: the message
            // names the field, the param is the path serde recorded.
            (
                "Failed to deserialize the JSON body into the target type: thinking: unknown variant `full` for `thinking.display`, expected one of `summarized`, `omitted`, `updates` at line 1 column 71",
                Some("thinking"),
            ),
            // A custom error inside a message keeps the message's path as the param.
            (
                "Failed to deserialize the JSON body into the target type: messages[1]: invalid value: `reasoning_content` and `reasoning` disagree; send one of them, or the same text in both at line 1 column 312",
                Some("messages[1]"),
            ),
            (
                "Failed to deserialize the JSON body into the target type: invalid type: integer `5`, expected a string at line 1 column 10",
                None,
            ),
            // A value that repeats a field marker does not steer the param.
            (
                "Failed to deserialize the JSON body into the target type: reasoning_effort: unknown variant `missing field `x`, expected one of `low`, `medium`, `high` at line 1 column 60",
                Some("reasoning_effort"),
            ),
            // Nor one that repeats axum's prefix.
            (
                "Invalid JSON data: Failed to deserialize the JSON body into the target type: truncation: unknown variant `Failed to deserialize the JSON body into the target type: missing field `x``, expected `auto` or `disabled` at line 1 column 110",
                Some("truncation"),
            ),
            // Keys with a space, a colon or a letter outside ASCII.
            (
                "Failed to deserialize the JSON body into the target type: max tokens: unknown field `max tokens`, expected one of `model`, `input` at line 1 column 19",
                Some("max tokens"),
            ),
            (
                "Failed to deserialize the JSON body into the target type: températur: unknown field `températur`, expected one of `model`, `input` at line 1 column 15",
                Some("températur"),
            ),
            (
                "Failed to deserialize the JSON body into the target type: metadata.user id: invalid type: integer `5`, expected a string at line 1 column 40",
                Some("metadata.user id"),
            ),
            // A custom error keeps its path, and a value it repeats cannot
            // move the split that reads the path.
            (
                "Failed to deserialize the JSON body into the target type: reasoning_effort: reasoning_effort must be a string, number, or null at line 1 column 42",
                Some("reasoning_effort"),
            ),
            (
                "Failed to deserialize the JSON body into the target type: function_call: unknown function_call value: \"x: invalid type\" at line 1 column 44",
                Some("function_call"),
            ),
            (
                "Failed to deserialize the JSON body into the target type: function_call: unknown function_call value: \"x: missing field `y`\" at line 1 column 48",
                Some("function_call"),
            ),
            // A key read wrong: a colon and a space in it, after a part that
            // reads as a path, is cut at the colon.
            (
                "Failed to deserialize the JSON body into the target type: a: b: unknown field `a: b`, expected one of `model`, `input` at line 1 column 12",
                Some("a"),
            ),
            ("JSON syntax error: expected value at line 1 column 1", None),
            ("Missing Content-Type: application/json header", None),
        ] {
            assert_eq!(param_from_deserialize_message(message).as_deref(), param, "{message}");
        }
    }

    /// The request of issue #5092: an agent client's second request, with the
    /// model's reasoning under both keys.
    fn replayed_turn(reasoning: &str) -> String {
        json!({
            "model": "zai-org/GLM-5.3",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning": reasoning,
                    "reasoning_content": "I should run ls.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{\"command\":\"ls\"}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "file.txt"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "terminal",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}
                }
            }]
        })
        .to_string()
    }

    /// Reads a chat completion body through the extractor, as the gateway does.
    async fn chat_request(body: &str) -> Result<ChatCompletionRequest, (StatusCode, Value)> {
        let request = Request::builder()
            .method("POST")
            .uri("/")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap_or_default();
        match ValidatedJson::<ChatCompletionRequest>::from_request(request, &()).await {
            Ok(ValidatedJson(body)) => Ok(body),
            Err(response) => {
                let status = response.status();
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap_or_default();
                Err((
                    status,
                    serde_json::from_slice(&bytes).unwrap_or(Value::Null),
                ))
            }
        }
    }

    /// Both keys with the same text make a request, as before the alias. Both
    /// keys with different text are refused as a value, and the param names
    /// the message.
    #[tokio::test]
    async fn an_assistant_turn_spelling_its_reasoning_both_ways_is_a_request() {
        let request = chat_request(&replayed_turn("I should run ls."))
            .await
            .unwrap();
        let ChatMessage::Assistant { reasoning, .. } = &request.messages[1] else {
            panic!("{:?}: not an assistant message", request.messages[1]);
        };
        assert_eq!(reasoning.text.as_deref(), Some("I should run ls."));

        let (status, json) = chat_request(&replayed_turn("I should run pwd."))
            .await
            .unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
        assert_eq!(json["error"]["code"], "invalid_value", "{json}");
        assert_eq!(json["error"]["param"], "messages[1]", "{json}");
        assert!(
            json["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("`reasoning_content` and `reasoning` disagree")),
            "{json}"
        );
    }
}

#[cfg(all(test, feature = "axum"))]
mod query_tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Debug, Deserialize)]
    struct Page {
        limit: Option<usize>,
    }

    /// The limit the extractor read from `uri`, or the status and the body of
    /// its refusal.
    async fn extract(uri: &str) -> Result<Option<usize>, (StatusCode, Value)> {
        let request = Request::builder().uri(uri).body(()).unwrap_or_default();
        let (mut parts, ()) = request.into_parts();
        match ValidatedQuery::<Page>::from_request_parts(&mut parts, &()).await {
            Ok(ValidatedQuery(page)) => Ok(page.limit),
            Err(response) => {
                let status = response.status();
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap_or_default();
                Err((status, serde_json::from_slice(&bytes).unwrap_or_default()))
            }
        }
    }

    /// A query string the type takes reaches the handler; a value the type
    /// does not take is a 400 in the envelope, `invalid_value`, with the
    /// framework's message.
    #[tokio::test]
    async fn a_query_string_the_type_does_not_take_is_refused_in_the_envelope() {
        assert_eq!(extract("/list?limit=5").await, Ok(Some(5)));
        assert_eq!(extract("/list").await, Ok(None));
        let (status, body) = extract("/list?limit=-1").await.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        assert_eq!(body["error"]["code"], "invalid_value", "{body}");
        assert_eq!(body["error"]["param"], "limit", "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or("")
                .starts_with(QUERY_PREFIX),
            "{body}"
        );
    }
}

#[cfg(all(test, feature = "axum"))]
mod path_tests {
    use super::*;

    /// A path parameter the route did not declare is a rejection of the
    /// route's own making: the framework's answer (a 500, text/plain)
    /// reaches the client as it is, not as a 400 in the envelope.
    #[tokio::test]
    async fn a_rejection_of_the_routes_own_making_keeps_the_frameworks_answer() {
        let request = Request::builder()
            .uri("/items/7")
            .body(())
            .unwrap_or_default();
        let (mut parts, ()) = request.into_parts();
        let Err(response) = ValidatedPath::<u32>::from_request_parts(&mut parts, &()).await else {
            panic!("a request no route matched carries no path parameters");
        };
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("");
        assert!(content_type.starts_with("text/plain"), "{content_type}");
    }
}
