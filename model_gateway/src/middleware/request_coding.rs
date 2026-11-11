//! The request's content coding at the HTTP edge.
//!
//! The gateway reads every request body as it arrives: no layer decodes a
//! content coding, so a body sent under `Content-Encoding: gzip` reached the
//! JSON parsers as compressed bytes and was refused as malformed JSON (the
//! typed routes' `invalid_json`, the plain routes' parse error), leaving the
//! client to debug its JSON. HTTP's answer to a content coding the server
//! does not decode is `415 Unsupported Media Type` naming the coding (RFC
//! 9110, section 8.4.1), so a request whose `Content-Encoding` names anything
//! but `identity` is refused here, before any layer inside reads the body, in
//! the gateway's error envelope under the extractor's header-level code (the
//! code a wrong `Content-Type` gets), with the coding named and the remedy
//! said. Decoding the coding instead (a decompression layer, the body limit
//! inside it) is a design the owner may take later; this check is the
//! boundary either way.

use axum::{
    extract::Request,
    http::{header::CONTENT_ENCODING, HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Json, Response},
};
use openai_protocol::validated::RejectionCode;
use serde_json::json;

/// The envelope's code: the JSON extractor's header-level code, the one a
/// request without `Content-Type: application/json` gets.
const CODE: &str = "json_parse_error";

/// The most of a coding's name the message repeats (a header value is the
/// client's text).
const NAMED_AT_MOST: usize = 40;

/// The first content coding of `headers` the gateway does not decode: any
/// token of any `Content-Encoding` value other than `identity` (in any
/// case); an empty value names no coding. A value that is not ASCII names
/// no coding the gateway knows either, and is shown as it was sent.
fn undecoded_coding(headers: &HeaderMap) -> Option<String> {
    for value in headers.get_all(CONTENT_ENCODING) {
        let Ok(value) = value.to_str() else {
            return Some(String::from_utf8_lossy(value.as_bytes()).into_owned());
        };
        if let Some(coding) = value
            .split(',')
            .map(str::trim)
            .find(|coding| !coding.is_empty() && !coding.eq_ignore_ascii_case("identity"))
        {
            return Some(coding.to_owned());
        }
    }
    None
}

/// The refusal: `415` in the gateway's envelope, the coding named and the
/// remedy said; the code rides on the response for the metrics layer's
/// `error_code` label, as the extractor's own refusals do.
fn refusal(coding: &str) -> Response {
    let named: String = coding.chars().take(NAMED_AT_MOST).collect();
    let mut response = (
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        Json(json!({
            "error": {
                "message": format!(
                    "Unsupported Content-Encoding '{named}': this gateway does not decode request bodies; send the body unencoded (identity)."
                ),
                "type": "invalid_request_error",
                "param": null,
                "code": CODE,
            }
        })),
    )
        .into_response();
    response
        .extensions_mut()
        .insert(RejectionCode(CODE.to_string()));
    response
}

/// The edge check: a request whose body arrives under a content coding the
/// gateway does not decode is refused by name before any layer inside reads
/// it; every other request passes untouched.
pub async fn refuse_encoded_request_body(request: Request, next: Next) -> Response {
    match undecoded_coding(request.headers()) {
        Some(coding) => refusal(&coding),
        None => next.run(request).await,
    }
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{to_bytes, Body, Bytes},
        http::{header::CONTENT_TYPE, HeaderValue},
        routing::post,
        Router,
    };
    use tower::ServiceExt;

    use super::*;

    /// `/read` answers the length of the body it read, behind the check.
    fn app() -> Router {
        Router::new()
            .route(
                "/read",
                post(|body: Bytes| async move { body.len().to_string() }),
            )
            .layer(axum::middleware::from_fn(refuse_encoded_request_body))
    }

    fn request(codings: &[HeaderValue]) -> Request {
        let mut builder = Request::builder()
            .method("POST")
            .uri("/read")
            .header(CONTENT_TYPE, "application/json");
        for coding in codings {
            builder = builder.header(CONTENT_ENCODING, coding.clone());
        }
        builder.body(Body::from("{}")).unwrap()
    }

    fn values(codings: &[&str]) -> Vec<HeaderValue> {
        codings
            .iter()
            .map(|coding| HeaderValue::from_str(coding).unwrap())
            .collect()
    }

    async fn envelope(response: Response) -> serde_json::Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn a_body_under_a_coding_the_gateway_does_not_decode_is_refused_by_name() {
        let mut cases = vec![
            (values(&["gzip"]), "gzip"),
            (values(&["GZIP"]), "GZIP"),
            (values(&["identity, br"]), "br"),
            (values(&["identity", "deflate"]), "deflate"),
            (values(&["x-custom"]), "x-custom"),
        ];
        cases.push((
            vec![HeaderValue::from_bytes(b"gz\xC3\xAFp").unwrap()],
            "gz\u{EF}p",
        ));
        for (codings, named) in cases {
            let response = app().oneshot(request(&codings)).await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "{codings:?}"
            );
            assert_eq!(
                response.extensions().get::<RejectionCode>(),
                Some(&RejectionCode(CODE.to_string())),
                "{codings:?}"
            );
            let json = envelope(response).await;
            assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
            assert_eq!(json["error"]["code"], CODE, "{json}");
            assert_eq!(json["error"]["param"], serde_json::Value::Null, "{json}");
            let message = json["error"]["message"].as_str().unwrap_or("");
            assert!(
                message.contains(&format!("'{named}'")) && message.contains("identity"),
                "the message names the coding and the remedy: {message}"
            );
        }
    }

    #[tokio::test]
    async fn no_coding_identity_and_an_empty_value_reach_the_handler() {
        for codings in [
            values(&[]),
            values(&["identity"]),
            values(&["Identity"]),
            values(&[""]),
            values(&["identity, identity"]),
        ] {
            let response = app().oneshot(request(&codings)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{codings:?}");
            assert!(
                response.extensions().get::<RejectionCode>().is_none(),
                "{codings:?}"
            );
            let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
            assert_eq!(&body[..], b"2", "{codings:?}");
        }
    }

    #[tokio::test]
    async fn a_long_coding_name_is_cut_in_the_message() {
        let long = "x".repeat(10 * NAMED_AT_MOST);
        let response = app().oneshot(request(&values(&[&long]))).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
        let json = envelope(response).await;
        let message = json["error"]["message"].as_str().unwrap();
        assert!(
            message.contains(&"x".repeat(NAMED_AT_MOST)) && !message.contains(&"x".repeat(NAMED_AT_MOST + 1)),
            "{message}"
        );
    }
}
