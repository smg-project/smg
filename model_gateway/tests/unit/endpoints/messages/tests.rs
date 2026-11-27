use axum::{
    body::to_bytes,
    http::{HeaderValue, StatusCode},
};

use super::*;

fn with(name: &'static str, value: &'static str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(name, HeaderValue::from_static(value));
    headers
}

fn message_request(body: serde_json::Value) -> CreateMessageRequest {
    serde_json::from_value(body).expect("a well-formed request")
}

/// A request without `anthropic-version` is served (this gateway's own
/// SDK clients send none); a version that is not a real calendar date of
/// 2023 or later is refused, as the public API refuses it.
#[test]
fn messages_requests_take_a_missing_version_and_refuse_a_malformed_one() {
    assert!(MessagesHeaders::parse(&HeaderMap::new()).is_ok());
    for ok in ["2023-06-01", "2023-01-01", "2024-02-29", "2026-12-31"] {
        assert!(
            MessagesHeaders::parse(&with("anthropic-version", ok)).is_ok(),
            "{ok}"
        );
        assert_eq!(
            parse_version(ok).map(|date| date.to_string()).as_deref(),
            Some(ok)
        );
    }
    for bad in [
        "1999-01-01",
        "latest",
        "2023-6-1",
        "2023-02-30",
        "2023-13-01",
        "2023-00-10",
        "2023-02-29",
    ] {
        let refused = MessagesHeaders::parse(&with("anthropic-version", bad)).expect_err(bad);
        assert_eq!(refused.status(), StatusCode::BAD_REQUEST, "{bad}");
        assert_eq!(
            route_error::extract_error_code_from_response(&refused),
            "invalid_anthropic_version",
            "{bad}"
        );
    }
}

/// The message of the 400 that refuses `value` as a version, read from
/// the refusal's body.
async fn refusal_message(value: HeaderValue) -> String {
    let mut headers = HeaderMap::new();
    headers.insert("anthropic-version", value);
    let refused = MessagesHeaders::parse(&headers).expect_err("a refused version");
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(refused.into_body(), usize::MAX)
        .await
        .expect("the refusal's body");
    let body: serde_json::Value = serde_json::from_slice(&body).expect("a JSON body");
    body["error"]["message"]
        .as_str()
        .expect("a message")
        .to_string()
}

/// The refusal echoes the client's value only up to the header echo
/// bound and marks the cut (the value is the client's text, and only the
/// server's header limit bounds it), and a value that is not ASCII is
/// echoed as text, not as an empty value.
#[tokio::test]
async fn the_version_refusal_echoes_a_bounded_value() {
    let long = "9".repeat(8000);
    let rows = [
        (
            "a short value",
            HeaderValue::from_static("latest"),
            "latest".to_string(),
        ),
        (
            "a value of 8000 characters",
            HeaderValue::from_str(&long).expect("a header value"),
            format!("{}...", "9".repeat(64)),
        ),
        (
            "a value with a byte outside ASCII",
            HeaderValue::from_bytes(b"2023-06-01\xe9").expect("a header value"),
            "2023-06-01\u{fffd}".to_string(),
        ),
    ];
    // Counts can agree while a character is wrong, so a failure line shows each text's end.
    let tail = |text: &str| -> String {
        let skip = text.chars().count().saturating_sub(16);
        text.chars().skip(skip).collect()
    };
    let mut misread = Vec::new();
    for (row, value, echo) in rows {
        let message = refusal_message(value).await;
        let due = format!("anthropic-version: \"{echo}\" is not a valid version");
        if message != due {
            let echoed = message
                .strip_prefix("anthropic-version: \"")
                .and_then(|rest| rest.strip_suffix("\" is not a valid version"))
                .unwrap_or(&message);
            let (shown, wanted) = (echoed.chars().count(), echo.chars().count());
            misread.push(format!(
                "{row}: the message echoes {shown} characters of the value, due {wanted}; \
                     it ends {:?}, due {:?}",
                tail(echoed),
                tail(&echo)
            ));
        }
    }
    assert!(misread.is_empty(), "{}", misread.join("\n"));
}

/// A thinking budget at or above `max_tokens` is refused as the public
/// API refuses it, except under the interleaved-thinking beta, where the
/// public API takes it.
#[test]
fn thinking_budget_must_stay_under_max_tokens_unless_interleaved() {
    let request = |max_tokens: u32, budget: u32| {
        message_request(serde_json::json!({
            "model": "m", "max_tokens": max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": budget},
            "messages": [{"role": "user", "content": "hi"}]
        }))
    };
    let plain = MessagesHeaders::parse(&HeaderMap::new()).expect("no header to refuse");
    assert!(plain.validate(&request(4096, 1024)).is_ok());
    let refused = plain.validate(&request(1024, 1024)).expect_err("refused");
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        route_error::extract_error_code_from_response(&refused),
        "thinking_budget_exceeds_max_tokens"
    );

    let interleaved = MessagesHeaders::parse(&with(
        "anthropic-beta",
        "fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14",
    ))
    .expect("a beta header refuses nothing");
    assert!(interleaved.validate(&request(1024, 1024)).is_ok());
    assert!(interleaved.validate(&request(1024, 8192)).is_ok());
}

/// A `thread` field needs the message-threads beta header, as the public
/// API requires; a request without `thread` needs no header.
#[test]
fn a_thread_field_needs_the_message_threads_beta_header() {
    let request = |thread: bool| {
        let mut body = serde_json::json!({
            "model": "m", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]
        });
        if thread {
            body["thread"] = serde_json::json!({"type": "create"});
        }
        message_request(body)
    };
    let plain = MessagesHeaders::parse(&HeaderMap::new()).expect("no header to refuse");
    assert!(plain.validate(&request(false)).is_ok());
    let refused = plain.validate(&request(true)).expect_err("refused");
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        route_error::extract_error_code_from_response(&refused),
        "thread_beta_required"
    );

    let named = MessagesHeaders::parse(&with(
        "anthropic-beta",
        "interleaved-thinking-2025-05-14, message-threads-2026-08-12",
    ))
    .expect("a beta header refuses nothing");
    assert!(named.validate(&request(true)).is_ok());
}
