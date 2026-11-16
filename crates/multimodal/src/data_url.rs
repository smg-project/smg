//! The size of an inline `data:` URL's payload, read before anything decodes
//! it: the router's plan check, the worker's item cap and the serving
//! engine's own check bound inline media by this one rule.
//! The module also holds the payload readers (`data_url_payload`,
//! `base64_payload`): the header rule and the empty-payload rule the
//! router-side fetch and the reference placement's door share.

use crate::error::MediaConnectorError;

/// Approximate decoded byte size of a `data:` URL's payload; `None` for any
/// other URL.
///
/// The rule is the servicer's: the scheme is matched without regard to case,
/// a URL with no comma has an empty payload, and a base64 payload is sized as
/// `len * 3 / 4` with its padding counted (at most two bytes over the decoded
/// size). The payload's length is its UTF-8 byte length, the unit of the cap
/// (the servicer counts code points, which is the same figure for the base64
/// and percent-encoded payloads media arrives in and a smaller one for a
/// plain payload with non-ASCII text). An over-estimate at the cap is the
/// safe side; an estimate that differs between the places that enforce the
/// cap is not.
pub fn data_url_payload_bytes(url: &str) -> Option<usize> {
    let rest = url
        .get(..5)
        .filter(|prefix| prefix.eq_ignore_ascii_case("data:"))
        .map(|_| &url[5..])?;
    let (header, payload) = rest.split_once(',').unwrap_or((rest, ""));
    let base64 = header
        .as_bytes()
        .windows(7)
        .any(|window| window.eq_ignore_ascii_case(b";base64"));
    Some(if base64 {
        payload.len() * 3 / 4
    } else {
        payload.len()
    })
}

/// The base64 payload of `data_url`, as every reader of the crate takes it:
/// the header must end in `;base64` (the only encoding the fetch decodes) and
/// the payload must carry something to decode ([`base64_payload`]). One rule
/// for the router-side fetch and the reference placement's door, so a data
/// URL no engine serves is refused in the same words wherever the request's
/// media is processed.
///
/// A header may carry parameters before the marker
/// (`data:image/png;charset=x;base64,`, a form RFC 2397 writes): the rule
/// reads the header's tail alone, so the fetch decodes such a payload and the
/// door forwards such a URL as written. Past the door the answer is the
/// worker's: an engine whose own fetcher reads the encoding after the
/// header's first `;` refuses the form, one that reads the last takes it.
pub fn data_url_payload(data_url: &str) -> Result<&str, MediaConnectorError> {
    let (metadata, data) = data_url
        .split_once(',')
        .ok_or_else(|| MediaConnectorError::DataUrl("missing comma in data url".into()))?;
    if !metadata.ends_with(";base64") {
        return Err(MediaConnectorError::DataUrl(
            "only base64 encoded data URLs are supported".into(),
        ));
    }
    base64_payload(data)
}

/// A base64 payload with something to decode: the text without its
/// surrounding whitespace, or the data URL refusal for an empty one (an empty
/// payload decodes to no bytes, which no decoder takes).
pub fn base64_payload(data: &str) -> Result<&str, MediaConnectorError> {
    let data = data.trim();
    if data.is_empty() {
        return Err(MediaConnectorError::DataUrl("empty payload".into()));
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::{base64_payload, data_url_payload, data_url_payload_bytes};

    /// The payload rules every reader shares: the header must end in
    /// `;base64`, whatever parameters come before it, the payload must carry
    /// something; the text comes back without its surrounding whitespace.
    #[test]
    fn the_payload_rules_are_the_readers() {
        assert_eq!(
            data_url_payload("data:image/png;base64,AAAA").unwrap(),
            "AAAA"
        );
        assert_eq!(
            data_url_payload("data:image/png;base64, AAAA ").unwrap(),
            "AAAA"
        );
        for url in [
            "data:image/png;charset=x;base64,AAAA",
            "data:image/png;charset=x;name=a.png;base64,AAAA",
        ] {
            assert_eq!(data_url_payload(url).unwrap(), "AAAA", "{url}");
        }
        for (url, why) in [
            ("data:image/png;base64", "missing comma in data url"),
            (
                "data:image/png,%89PNG%0D%0A",
                "only base64 encoded data URLs are supported",
            ),
            (
                "data:image/png;base64;charset=x,AAAA",
                "only base64 encoded data URLs are supported",
            ),
            ("data:image/png;base64,", "empty payload"),
            ("data:image/png;base64,   ", "empty payload"),
        ] {
            let error = data_url_payload(url).expect_err(url);
            assert_eq!(
                error.to_string(),
                format!("data URL parse error: {why}"),
                "{url}"
            );
        }
        assert_eq!(base64_payload(" AQ== ").unwrap(), "AQ==");
        assert!(base64_payload(" ").is_err());
    }

    #[test]
    fn sizes_base64_and_plain_payloads() {
        assert_eq!(
            data_url_payload_bytes("data:image/png;base64,AAAAAAAA"),
            Some(6)
        );
        assert_eq!(data_url_payload_bytes("data:text/plain,hello"), Some(5));
        assert_eq!(data_url_payload_bytes("data:,"), Some(0));
        // Bytes, not code points: the unit of the cap.
        assert_eq!(data_url_payload_bytes("data:text/plain,\u{e9}"), Some(2));
        assert_eq!(
            data_url_payload_bytes("data:text/plain;charset=utf-8,%C3%A9"),
            Some(6)
        );
    }

    #[test]
    fn other_urls_are_not_sized() {
        assert_eq!(data_url_payload_bytes("https://example.com/x.png"), None);
        assert_eq!(data_url_payload_bytes("/local/path.png"), None);
        assert_eq!(data_url_payload_bytes("data"), None);
        assert_eq!(data_url_payload_bytes("dat\u{e9}:x,y"), None);
    }

    #[test]
    fn the_scheme_case_does_not_matter() {
        assert_eq!(
            data_url_payload_bytes("DATA:image/png;base64,AAAAAAAA"),
            Some(6)
        );
        assert_eq!(data_url_payload_bytes("Data:text/plain,hello"), Some(5));
        assert_eq!(
            data_url_payload_bytes("data:image/png;BASE64,AAAAAAAA"),
            Some(6)
        );
    }

    #[test]
    fn a_missing_comma_is_an_empty_payload() {
        assert_eq!(
            data_url_payload_bytes("data:image/png;base64AAAAAAAA"),
            Some(0)
        );
        assert_eq!(data_url_payload_bytes("data:"), Some(0));
    }

    #[test]
    fn base64_padding_is_counted() {
        assert_eq!(
            data_url_payload_bytes("data:image/png;base64,AAAA"),
            Some(3)
        );
        assert_eq!(
            data_url_payload_bytes("data:image/png;base64,AAA="),
            Some(3)
        );
        assert_eq!(
            data_url_payload_bytes("data:image/png;base64,AA=="),
            Some(3)
        );
    }
}
