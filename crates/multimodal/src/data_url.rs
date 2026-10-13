//! The size of an inline `data:` URL's payload, read before anything decodes
//! it: the router's plan check, the worker's item cap and the serving
//! engine's own check bound inline media by this one rule.

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

#[cfg(test)]
mod tests {
    use super::data_url_payload_bytes;

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
