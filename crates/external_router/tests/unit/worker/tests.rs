use super::redacted_url;

/// A URL without userinfo, query or fragment is the text as given,
/// borrowed: whatever its scheme, host form, path or trailing slash.
#[test]
fn a_plain_url_reads_byte_for_byte() {
    for plain in [
        "https://api.example.com",
        "http://127.0.0.1:8000/v1",
        "http://[::1]:8000/v1/messages",
        "http://Worker:3000/",
        "wss://host/v1/realtime",
        "127.0.0.1:8000",
        "",
    ] {
        let shown = redacted_url(plain);
        assert!(matches!(shown, std::borrow::Cow::Borrowed(_)), "{plain}");
        assert_eq!(shown, plain);
    }
}

/// A userinfo is masked whole, whatever its shape, and a query or a
/// fragment is cut; nothing of the credential survives.
#[test]
fn a_userinfo_is_masked_whole_and_a_query_is_cut() {
    for (url, masked) in [
        (
            "https://user:secret@127.0.0.1:1/v1/messages",
            "https://***@127.0.0.1:1/v1/messages",
        ),
        ("https://api-token@host:8000", "https://***@host:8000"),
        ("https://tok-123:x-oauth-basic@host", "https://***@host"),
        (
            "wss://user:secret@127.0.0.1:1/v1/realtime?model=m",
            "wss://***@127.0.0.1:1/v1/realtime",
        ),
        ("http://user:secret@host:8000/", "http://***@host:8000"),
        ("http://host:8000/v1?token=secret", "http://host:8000/v1"),
        ("http://host:8000/v1#secret", "http://host:8000/v1"),
        ("http://host:8000/?x=1", "http://host:8000"),
        // Schemeless: an `@` before the first `/` is a userinfo too.
        ("user:secret@host/sse", "***@host/sse"),
        // A `://` inside a path or a query is not a scheme separator.
        ("host/x?next=http://user:secret@y", "host/x"),
    ] {
        let shown = redacted_url(url);
        assert_eq!(shown, masked, "{url}");
        for credential in ["secret", "user", "api-token", "tok-123", "x-oauth-basic"] {
            assert!(!shown.contains(credential), "{url} -> {shown}");
        }
    }
}
