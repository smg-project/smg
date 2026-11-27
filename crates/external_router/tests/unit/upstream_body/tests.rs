use super::*;
use crate::test_support::{text_of, upstream};

const UPSTREAM_REFUSAL: &str = r#"{"error":{"message":"Rate limit reached for model in organization org-abc123 project proj_xyz789","type":"rate_limit","code":"rate_limit_exceeded"}}"#;

/// A provider's refusal reaches the caller under its status and content
/// type with the operator's organization and project gone.
#[tokio::test]
async fn an_upstream_errors_body_loses_the_operators_identifiers() {
    let response = answer_error(
        upstream(429, Some("application/json"), UPSTREAM_REFUSAL),
        "m",
    )
    .await;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
    let text = text_of(response).await;
    assert!(!text.contains("org-abc123"), "{text}");
    assert!(!text.contains("proj_xyz789"), "{text}");
    assert!(text.contains("Rate limit reached for model"), "{text}");
    assert!(text.contains(r#""code":"rate_limit_exceeded""#), "{text}");
}

/// An error body past the bound is not buffered or relayed: the caller
/// gets the upstream's status with the gateway's envelope and none of it.
#[tokio::test]
async fn an_upstream_error_body_past_the_bound_is_answered_without_it() {
    let body = format!(
        r#"{{"error":{{"message":"{}"}}}}"#,
        "x".repeat(MAX_ERROR_RESPONSE_SIZE)
    );
    let response = answer_error(upstream(429, Some("application/json"), body), "m").await;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    let text = text_of(response).await;
    assert!(!text.contains("xxxx"), "{text}");
    assert!(text.contains("too large"), "{text}");
}

/// A refusal whose body cannot be read keeps the upstream's status: the
/// caller gets it with the gateway's envelope and without the transport
/// error's text, not a 500 the retry layer would act on.
#[tokio::test]
async fn an_upstream_error_body_that_cannot_be_read_keeps_the_upstreams_status() {
    let chunks = vec![
        Ok::<Vec<u8>, std::io::Error>(b"{\"error\":".to_vec()),
        Err(std::io::Error::other("connection reset")),
    ];
    let body = reqwest::Body::wrap_stream(futures::stream::iter(chunks));
    let response = answer_error(upstream(403, Some("application/json"), body), "m").await;
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let text = text_of(response).await;
    assert!(
        text.contains("Failed to read upstream error response"),
        "{text}"
    );
    assert!(!text.contains("connection reset"), "{text}");
}
