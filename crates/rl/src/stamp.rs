//! Response stamping: the header names the gateway sets, and the probe that
//! decides whether a buffered `/generate` body spanned two weight versions.

use serde::{de::IgnoredAny, Deserialize};
use serde_json::Value;

use crate::version::Version;

/// The version SMG believed the routed engine held when it emitted the
/// response head.
pub const WEIGHT_VERSION_HEADER: &str = "x-smg-weight-version";
/// Set to `true` on a buffered `/generate` response whose tokens came from
/// more than one version, or whose engine-reported version differs from the
/// stamped one.
pub const MIXED_VERSION_HEADER: &str = "x-smg-mixed-version";

#[derive(Deserialize, Default)]
struct MetaProbe {
    #[serde(default)]
    weight_version: Option<Value>,
    #[serde(default)]
    weight_versions: Option<Vec<IgnoredAny>>,
}

#[derive(Deserialize)]
struct GenerateProbe {
    #[serde(default)]
    meta_info: Option<MetaProbe>,
}

fn reported_version(value: &Value) -> Option<Version> {
    match value {
        Value::String(s) if !s.trim().is_empty() => Some(Version::parse(s)),
        Value::Number(n) => Some(Version::parse(&n.to_string())),
        _ => None,
    }
}

fn is_mixed(meta: &MetaProbe, expected: Option<&Version>) -> bool {
    if meta
        .weight_versions
        .as_ref()
        .is_some_and(|spans| spans.len() > 1)
    {
        return true;
    }
    match (
        meta.weight_version.as_ref().and_then(reported_version),
        expected,
    ) {
        (Some(reported), Some(expected)) => reported != *expected,
        _ => false,
    }
}

/// Whether a buffered `/generate` body (one object, or an array for `n > 1`)
/// spans more than one weight version or reports a version other than
/// `expected`. Only `meta_info.weight_version` and the span count are read;
/// an unparsable body is not mixed.
pub fn generate_is_mixed(body: &[u8], expected: Option<&Version>) -> bool {
    let first = body.iter().find(|b| !b.is_ascii_whitespace());
    match first {
        Some(b'[') => serde_json::from_slice::<Vec<GenerateProbe>>(body)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|p| p.meta_info.as_ref())
                    .any(|m| is_mixed(m, expected))
            })
            .unwrap_or(false),
        Some(b'{') => serde_json::from_slice::<GenerateProbe>(body)
            .ok()
            .and_then(|p| p.meta_info)
            .is_some_and(|m| is_mixed(&m, expected)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_SPANS: &[u8] = br#"{"text":"x","meta_info":{"weight_version":"42","weight_versions":[{"version":"41","start":0,"end":57},{"version":"42","start":57,"end":128}]}}"#;
    const ONE_SPAN: &[u8] = br#"{"text":"x","meta_info":{"weight_version":"42","weight_versions":[{"version":"42","start":0,"end":128}]}}"#;

    #[test]
    fn two_spans_are_mixed_whatever_the_expectation() {
        assert!(generate_is_mixed(TWO_SPANS, None));
        assert!(generate_is_mixed(TWO_SPANS, Some(&Version::parse("42"))));
    }

    #[test]
    fn one_span_is_mixed_only_when_the_reported_version_differs() {
        assert!(!generate_is_mixed(ONE_SPAN, None));
        assert!(!generate_is_mixed(ONE_SPAN, Some(&Version::parse("42"))));
        assert!(!generate_is_mixed(ONE_SPAN, Some(&Version::parse("042"))));
        assert!(generate_is_mixed(ONE_SPAN, Some(&Version::parse("41"))));
        let no_spans = br#"{"text":"x","meta_info":{"weight_version":"default"}}"#;
        assert!(!generate_is_mixed(no_spans, None));
        assert!(generate_is_mixed(no_spans, Some(&Version::parse("3"))));
    }

    #[test]
    fn batches_and_garbage() {
        let batch = format!(
            "[{}, {}]",
            std::str::from_utf8(ONE_SPAN).unwrap(),
            std::str::from_utf8(TWO_SPANS).unwrap()
        );
        assert!(generate_is_mixed(
            batch.as_bytes(),
            Some(&Version::parse("42"))
        ));
        let clean = format!(" [{}]", std::str::from_utf8(ONE_SPAN).unwrap());
        assert!(!generate_is_mixed(
            clean.as_bytes(),
            Some(&Version::parse("42"))
        ));
        assert!(!generate_is_mixed(b"", None));
        assert!(!generate_is_mixed(b"not json", Some(&Version::parse("1"))));
        assert!(!generate_is_mixed(
            br#"{"text":"no meta"}"#,
            Some(&Version::parse("1"))
        ));
    }
}
