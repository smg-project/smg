//! The Messages route's request headers, read once per request as a typed
//! extractor: the public API's dated `anthropic-version` and the betas named
//! by `anthropic-beta`. A malformed version is refused with the public API's
//! own 400 before the body is read; the betas feed the request rules that
//! read a header beside the body ([`validate_with_betas`]). The handlers keep
//! the raw [`HeaderMap`] they forward: nothing here changes what the backend
//! receives. The extractor runs where the handlers run, on the buffered body
//! path; a request the stream layer forwards whole reaches no handler and
//! none of these checks, as before this module.

use std::borrow::Cow;

use axum::{
    extract::FromRequestParts,
    http::{request::Parts, HeaderMap},
    response::Response,
};
use chrono::{Datelike, NaiveDate};
use openai_protocol::messages::{validate_with_betas, CreateMessageRequest, MessagesBetas};

use crate::routers::error as route_error;

/// The first year of the public API's dated versions: a version before it is
/// one the API never served.
const FIRST_VERSION_YEAR: i32 = 2023;

/// The typed headers of a Messages request. The version is read for its
/// refusal and not kept: a request without the header is served as the
/// current version (this gateway's own SDK clients send none), and the header
/// selects nothing here; the backend reads it from the forwarded raw headers.
#[derive(Debug, Clone)]
pub(crate) struct MessagesHeaders {
    /// The betas named by `anthropic-beta`.
    betas: MessagesBetas,
}

impl MessagesHeaders {
    /// The typed headers of `headers`, or the public API's 400 for an
    /// `anthropic-version` that is not a calendar date of 2023 or later.
    pub(crate) fn parse(headers: &HeaderMap) -> Result<Self, Response> {
        if let Some(value) = headers.get("anthropic-version") {
            let version = value.to_str().unwrap_or_default();
            if parse_version(version).is_none() {
                return Err(route_error::bad_request(
                    "invalid_anthropic_version",
                    format!("anthropic-version: \"{version}\" is not a valid version"),
                ));
            }
        }
        let betas = MessagesBetas::from_header_values(
            headers
                .get_all("anthropic-beta")
                .iter()
                .filter_map(|value| value.to_str().ok()),
        );
        Ok(Self { betas })
    }

    /// The request rules that read the betas beside the body, refused with
    /// the public API's own 400 (the code and the message of the rule).
    pub(crate) fn validate(&self, body: &CreateMessageRequest) -> Result<(), Response> {
        validate_with_betas(body, self.betas).map_err(|refusal| {
            route_error::bad_request(
                refusal.code.into_owned(),
                refusal.message.map(Cow::into_owned).unwrap_or_default(),
            )
        })
    }
}

impl<S: Send + Sync> FromRequestParts<S> for MessagesHeaders {
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        Self::parse(&parts.headers)
    }
}

/// `YYYY-MM-DD`, the public API's ten-character form, as a real calendar
/// date of 2023 or later: the shape is checked as the public API writes it
/// (chrono's parser alone would also take an unpadded or a signed field),
/// chrono checks the calendar.
fn parse_version(version: &str) -> Option<NaiveDate> {
    let bytes = version.as_bytes();
    let shaped = bytes.len() == 10
        && bytes.iter().enumerate().all(|(i, b)| {
            if i == 4 || i == 7 {
                *b == b'-'
            } else {
                b.is_ascii_digit()
            }
        });
    if !shaped {
        return None;
    }
    NaiveDate::parse_from_str(version, "%Y-%m-%d")
        .ok()
        .filter(|date| date.year() >= FIRST_VERSION_YEAR)
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderValue, StatusCode};

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
}
