//! Axum extractors for multipart/form-data inference endpoints.
//!
//! JSON endpoints get [`crate::validated::ValidatedJson`]; the
//! `/v1/audio/transcriptions` endpoint uses multipart/form-data and gets
//! [`AudioTranscriptionMultipart`], which parses the form into a typed
//! `(TranscriptionRequest, AudioFile)` pair before the handler runs.

#[cfg(feature = "axum")]
use axum::{
    extract::{multipart::MultipartError, FromRequest, Multipart, Request},
    http::StatusCode,
    response::Response,
};

#[cfg(feature = "axum")]
use crate::transcription::{AudioFile, TranscriptionRequest};

/// Extractor for `/v1/audio/transcriptions` requests.
///
/// Parses `multipart/form-data` into a [`TranscriptionRequest`] (text fields)
/// plus an [`AudioFile`] (the `file` part). Returns `400 Bad Request` on
/// malformed parts, missing/empty `file`, missing/blank `model`, or
/// out-of-range `temperature`.
#[cfg(feature = "axum")]
pub struct AudioTranscriptionMultipart {
    pub request: TranscriptionRequest,
    pub audio: AudioFile,
}

#[cfg(feature = "axum")]
impl<S: Send + Sync> FromRequest<S> for AudioTranscriptionMultipart {
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let mut multipart = Multipart::from_request(req, state)
            .await
            .map_err(|e| bad_request("invalid_multipart", None, e.body_text()))?;

        let mut file_bytes: Option<bytes::Bytes> = None;
        let mut file_name: Option<String> = None;
        let mut file_content_type: Option<String> = None;
        let mut request = TranscriptionRequest::default();
        let mut timestamp_granularities: Vec<String> = Vec::new();

        loop {
            let field = match multipart.next_field().await {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    return Err(bad_request(
                        "invalid_multipart",
                        None,
                        format!("Failed to read multipart field: {e}"),
                    ));
                }
            };

            let name = field.name().unwrap_or("").to_string();
            match name.as_str() {
                "file" => {
                    file_name = field.file_name().map(str::to_string);
                    file_content_type = field.content_type().map(str::to_string);
                    match field.bytes().await {
                        Ok(b) => file_bytes = Some(b),
                        Err(e) => {
                            return Err(bad_request(
                                "invalid_multipart",
                                Some("file"),
                                format!("Failed to read audio file bytes: {e}"),
                            ));
                        }
                    }
                }
                "model" => match field.text().await {
                    Ok(t) => request.model = t,
                    Err(e) => return Err(bad_text_field("model", e)),
                },
                "language" => match field.text().await {
                    Ok(t) => request.language = Some(t),
                    Err(e) => return Err(bad_text_field("language", e)),
                },
                "prompt" => match field.text().await {
                    Ok(t) => request.prompt = Some(t),
                    Err(e) => return Err(bad_text_field("prompt", e)),
                },
                "response_format" => match field.text().await {
                    Ok(t) => request.response_format = Some(t),
                    Err(e) => return Err(bad_text_field("response_format", e)),
                },
                "temperature" => match field.text().await {
                    Ok(t) => match t.trim().parse::<f32>() {
                        Ok(v) if v.is_finite() && (0.0..=1.0).contains(&v) => {
                            request.temperature = Some(v);
                        }
                        Ok(v) => {
                            return Err(bad_request(
                                "invalid_value",
                                Some("temperature"),
                                format!("Invalid 'temperature' value: {v} (must be a finite number in [0.0, 1.0])"),
                            ));
                        }
                        Err(e) => {
                            return Err(bad_request(
                                "invalid_value",
                                Some("temperature"),
                                format!("Invalid 'temperature' value: {e}"),
                            ));
                        }
                    },
                    Err(e) => return Err(bad_text_field("temperature", e)),
                },
                "timestamp_granularities" | "timestamp_granularities[]" => {
                    match field.text().await {
                        Ok(t) => timestamp_granularities.push(t),
                        Err(e) => return Err(bad_text_field("timestamp_granularities", e)),
                    }
                }
                "stream" => {
                    match field.text().await {
                        Ok(t) => match t.as_str() {
                            "true" | "True" | "TRUE" | "1" => request.stream = Some(true),
                            "false" | "False" | "FALSE" | "0" => request.stream = Some(false),
                            other => {
                                return Err(bad_request(
                                "invalid_value",
                                Some("stream"),
                                format!("Invalid 'stream' value: '{other}' (expected true/false/1/0)"),
                            ));
                            }
                        },
                        Err(e) => return Err(bad_text_field("stream", e)),
                    }
                }
                _ => {
                    // Unknown field; drain to free resources but otherwise ignore.
                    let _ = field.bytes().await;
                }
            }
        }

        if request.model.trim().is_empty() {
            return Err(bad_request(
                "missing_required_parameter",
                Some("model"),
                "Missing required 'model' field".to_string(),
            ));
        }
        request.model = request.model.trim().to_string();

        let bytes = match file_bytes {
            Some(b) if !b.is_empty() => b,
            Some(_) => {
                return Err(bad_request(
                    "invalid_value",
                    Some("file"),
                    "Uploaded 'file' part is empty".to_string(),
                ));
            }
            None => {
                return Err(bad_request(
                    "missing_required_parameter",
                    Some("file"),
                    "Missing required 'file' part".to_string(),
                ));
            }
        };

        if !timestamp_granularities.is_empty() {
            request.timestamp_granularities = Some(timestamp_granularities);
        }

        let audio = AudioFile {
            bytes,
            file_name: file_name.unwrap_or_else(|| "audio".to_string()),
            content_type: file_content_type,
        };

        Ok(AudioTranscriptionMultipart { request, audio })
    }
}

/// The refusal of a form the parser cannot take: the gateway's JSON error
/// object, written as a JSON body's rejection is, with `param` the field the
/// refusal is about when there is one.
#[cfg(feature = "axum")]
fn bad_request(code: &str, param: Option<&str>, message: String) -> Response {
    crate::validated::rejection(
        StatusCode::BAD_REQUEST,
        code,
        param.map(str::to_owned),
        message,
    )
}

#[cfg(feature = "axum")]
fn bad_text_field(field: &str, e: MultipartError) -> Response {
    bad_request(
        "invalid_multipart",
        Some(field),
        format!("Failed to read '{field}' field: {e}"),
    )
}

#[cfg(test)]
#[cfg(feature = "axum")]
mod tests {
    use axum::{
        body::Body,
        extract::{FromRequest, Request},
    };

    use super::*;

    /// The parser's refusal of `form` (CRLF line ends, boundary `cut`): its
    /// status and JSON body.
    async fn refusal(form: &str) -> (StatusCode, serde_json::Value) {
        let request = Request::builder()
            .method("POST")
            .header("content-type", "multipart/form-data; boundary=cut")
            .body(Body::from(form.replace('\n', "\r\n")))
            .expect("a request");
        refusal_of(request).await
    }

    /// The parser's refusal of `request`: its status and JSON body.
    async fn refusal_of(request: Request) -> (StatusCode, serde_json::Value) {
        let response = match AudioTranscriptionMultipart::from_request(request, &()).await {
            Ok(_) => panic!("the form was taken"),
            Err(refusal) => refusal,
        };
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("drain body");
        (
            status,
            serde_json::from_slice(&body).expect("json error body"),
        )
    }

    /// A request that is no multipart form at all (a JSON body here) is
    /// refused by the extractor before any part is read, with the same JSON
    /// error object and code as a part the parser cannot read.
    #[tokio::test]
    async fn a_request_that_is_not_a_form_is_refused_with_the_json_error_object() {
        let request = Request::builder()
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model": "whisper"}"#))
            .expect("a request");
        let (status, body) = refusal_of(request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        assert_eq!(body["error"]["code"], "invalid_multipart", "{body}");
        assert!(body["error"]["param"].is_null(), "{body}");
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(message.contains("boundary"), "{body}");
    }

    #[tokio::test]
    async fn a_form_without_its_file_is_refused_with_the_json_error_object() {
        let (status, body) =
            refusal("--cut\nContent-Disposition: form-data; name=\"model\"\n\nwhisper\n--cut--\n")
                .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        assert_eq!(
            body["error"]["code"], "missing_required_parameter",
            "{body}"
        );
        assert_eq!(body["error"]["param"], "file", "{body}");
        assert_eq!(
            body["error"]["message"], "Missing required 'file' part",
            "{body}"
        );
    }

    #[tokio::test]
    async fn a_form_without_a_model_is_refused_with_the_json_error_object() {
        let (status, body) = refusal(
            "--cut\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\nContent-Type: audio/wav\n\nRIFF\n--cut--\n",
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(
            body["error"]["code"], "missing_required_parameter",
            "{body}"
        );
        assert_eq!(body["error"]["param"], "model", "{body}");
        assert_eq!(
            body["error"]["message"], "Missing required 'model' field",
            "{body}"
        );
    }

    #[tokio::test]
    async fn a_temperature_that_is_not_a_number_is_refused_with_the_json_error_object() {
        let (status, body) = refusal(
            "--cut\nContent-Disposition: form-data; name=\"temperature\"\n\nwarm\n--cut--\n",
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(body["error"]["code"], "invalid_value", "{body}");
        assert_eq!(body["error"]["param"], "temperature", "{body}");
        let message = body["error"]["message"].as_str().expect("a message");
        assert!(
            message.starts_with("Invalid 'temperature' value:"),
            "{body}"
        );
    }
}
