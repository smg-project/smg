//! The version-policy grammar shared by `--rl-version-policy` and the
//! `x-smg-version-policy` request header.

use std::{fmt, str::FromStr};

use http::HeaderMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::version::Version;

/// Per-request override of the configured policy, same grammar.
pub const VERSION_POLICY_HEADER: &str = "x-smg-version-policy";

/// Which engines a request may be routed to, judged against the model's
/// fleet maximum: `any | latest-only | min-version:<v> | max-staleness:<k>`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum VersionPolicy {
    #[default]
    Any,
    LatestOnly,
    MinVersion(Version),
    MaxStaleness(u64),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("invalid version policy `{input}`: {reason}")]
pub struct VersionPolicyError {
    pub input: String,
    pub reason: &'static str,
}

impl VersionPolicy {
    /// The header override, if the request carries one. `Err` when it is
    /// present but malformed (including non-UTF-8 bytes).
    pub fn from_headers(headers: &HeaderMap) -> Result<Option<Self>, VersionPolicyError> {
        let Some(value) = headers.get(VERSION_POLICY_HEADER) else {
            return Ok(None);
        };
        let text = value.to_str().map_err(|_| VersionPolicyError {
            input: String::from_utf8_lossy(value.as_bytes()).into_owned(),
            reason: "header value is not UTF-8",
        })?;
        text.trim().parse().map(Some)
    }
}

impl FromStr for VersionPolicy {
    type Err = VersionPolicyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let err = |reason| VersionPolicyError {
            input: s.to_string(),
            reason,
        };
        match s {
            "any" => Ok(Self::Any),
            "latest-only" => Ok(Self::LatestOnly),
            _ => {
                if let Some(v) = s.strip_prefix("min-version:") {
                    if v.is_empty() || v.contains(char::is_whitespace) {
                        return Err(err("min-version needs a version with no whitespace"));
                    }
                    return Ok(Self::MinVersion(Version::parse(v)));
                }
                if let Some(k) = s.strip_prefix("max-staleness:") {
                    return k
                        .parse::<u64>()
                        .map(Self::MaxStaleness)
                        .map_err(|_| err("max-staleness needs a decimal integer"));
                }
                Err(err(
                    "expected any | latest-only | min-version:<v> | max-staleness:<k>",
                ))
            }
        }
    }
}

impl fmt::Display for VersionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Any => f.write_str("any"),
            Self::LatestOnly => f.write_str("latest-only"),
            Self::MinVersion(v) => write!(f, "min-version:{v}"),
            Self::MaxStaleness(k) => write!(f, "max-staleness:{k}"),
        }
    }
}

impl Serialize for VersionPolicy {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for VersionPolicy {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use http::HeaderValue;

    use super::*;

    #[test]
    fn grammar_round_trips() {
        for text in [
            "any",
            "latest-only",
            "min-version:42",
            "min-version:step-3",
            "max-staleness:2",
        ] {
            let policy: VersionPolicy = text.parse().unwrap();
            assert_eq!(policy.to_string(), text);
            let json = serde_json::to_string(&policy).unwrap();
            assert_eq!(json, format!("\"{text}\""));
            assert_eq!(
                serde_json::from_str::<VersionPolicy>(&json).unwrap(),
                policy
            );
        }
        assert_eq!(VersionPolicy::default(), VersionPolicy::Any);
        assert_eq!(
            "max-staleness:1".parse::<VersionPolicy>().unwrap(),
            VersionPolicy::MaxStaleness(1)
        );
    }

    #[test]
    fn rejects_bad_inputs() {
        for bad in [
            "",
            "Any",
            "latest",
            "min-version:",
            "min-version:a b",
            "max-staleness:x",
            "max-staleness:-1",
            "max-staleness:",
        ] {
            let err = bad.parse::<VersionPolicy>().unwrap_err();
            assert_eq!(err.input, bad);
        }
        assert!(serde_json::from_str::<VersionPolicy>("\"nope\"").is_err());
    }

    #[test]
    fn header_is_optional_and_validated() {
        let mut headers = HeaderMap::new();
        assert_eq!(VersionPolicy::from_headers(&headers).unwrap(), None);
        headers.insert(
            VERSION_POLICY_HEADER,
            HeaderValue::from_static(" latest-only "),
        );
        assert_eq!(
            VersionPolicy::from_headers(&headers).unwrap(),
            Some(VersionPolicy::LatestOnly)
        );
        headers.insert(VERSION_POLICY_HEADER, HeaderValue::from_static("bogus"));
        assert!(VersionPolicy::from_headers(&headers).is_err());
        headers.insert(
            VERSION_POLICY_HEADER,
            HeaderValue::from_bytes(b"\xff").unwrap(),
        );
        assert_eq!(
            VersionPolicy::from_headers(&headers).unwrap_err().reason,
            "header value is not UTF-8"
        );
    }
}
