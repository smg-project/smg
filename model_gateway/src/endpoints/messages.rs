//! The Messages route's request headers, read once per request as a typed
//! extractor: the public API's dated `anthropic-version` and the betas named
//! by `anthropic-beta`. A malformed version is refused with the public API's
//! own 400 before the body is read; the betas feed the request rules that
//! read a header beside the body ([`validate_with_betas`]). The handlers keep
//! the raw [`HeaderMap`] they forward: nothing here changes what the backend
//! receives. The extractor runs where the handlers run, on the buffered body
//! path. A request the stream layer forwards whole reaches no handler: the
//! layer makes the version check itself (`MessagesHeaders::check_version`),
//! and the rules that read the body stay with the handler.

use std::borrow::Cow;

use axum::{
    extract::FromRequestParts,
    http::{request::Parts, HeaderMap},
    response::Response,
};
use chrono::{Datelike, NaiveDate};
use openai_protocol::{
    messages::{validate_with_betas, CreateMessageRequest, MessagesBetas},
    validated::header_echo,
};

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
    /// The public API's 400 for an `anthropic-version` that is not a calendar
    /// date of 2023 or later. The stream layer makes this check on a request
    /// it forwards whole, which reaches no handler. The message echoes the
    /// client's value as text (a byte outside ASCII is shown, not dropped),
    /// cut at the header echo bound.
    pub(crate) fn check_version(headers: &HeaderMap) -> Result<(), Response> {
        if let Some(value) = headers.get("anthropic-version") {
            let version = String::from_utf8_lossy(value.as_bytes());
            if parse_version(&version).is_none() {
                let echo = header_echo(&version);
                return Err(route_error::bad_request(
                    "invalid_anthropic_version",
                    format!("anthropic-version: \"{echo}\" is not a valid version"),
                ));
            }
        }
        Ok(())
    }

    /// The typed headers of `headers`, or the refusal of
    /// [`Self::check_version`].
    pub(crate) fn parse(headers: &HeaderMap) -> Result<Self, Response> {
        Self::check_version(headers)?;
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
#[path = "../../tests/unit/endpoints/messages/tests.rs"]
mod tests;
