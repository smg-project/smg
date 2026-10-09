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
use serde_json::json;
#[cfg(feature = "axum")]
use validator::Validate;

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

/// The gateway's JSON error envelope for a rejected request body.
#[cfg(feature = "axum")]
fn rejection(status: StatusCode, code: &str, message: String) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code
            }
        })),
    )
        .into_response()
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
                    rejection(StatusCode::BAD_REQUEST, "json_parse_error", error_message)
                })?;

        // Normalize the request (apply defaults based on other fields)
        data.normalize();

        // Then, automatically validate the data
        data.validate().map_err(|validation_errors| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": validation_errors.to_string(),
                        "type": "invalid_request_error",
                        "code": 400
                    }
                })),
            )
                .into_response()
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
}
