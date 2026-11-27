use rustls::crypto::ring;

use super::*;

/// The beta is found in a repeated header and in a comma-separated list,
/// by its undated name; another beta or no header is not it.
#[test]
fn anthropic_beta_named_reads_repeated_and_listed_values() {
    let mut headers = HeaderMap::new();
    assert!(!anthropic_beta_named(&headers, "message-threads"));
    headers.append(
        "anthropic-beta",
        HeaderValue::from_static("interleaved-thinking-2025-05-14"),
    );
    assert!(!anthropic_beta_named(&headers, "message-threads"));
    assert!(anthropic_beta_named(&headers, "interleaved-thinking"));
    headers.append(
        "anthropic-beta",
        HeaderValue::from_static(
            "fine-grained-tool-streaming-2025-05-14, message-threads-2026-08-12",
        ),
    );
    assert!(anthropic_beta_named(&headers, "message-threads"));
    assert!(anthropic_beta_named(
        &headers,
        "fine-grained-tool-streaming"
    ));
    assert!(!anthropic_beta_named(&headers, "threads"));
}

/// The dated form alone is the beta: a longer beta name sharing the
/// prefix, the undated name, a malformed date and a date of the right
/// shape but the wrong length are not it.
#[test]
fn anthropic_beta_named_requires_the_dated_suffix() {
    let mut headers = HeaderMap::new();
    headers.append(
        "anthropic-beta",
        HeaderValue::from_static("token-counting-2024-11-01, token-efficient-tools-2025-02-19"),
    );
    assert!(anthropic_beta_named(&headers, "token-counting"));
    assert!(anthropic_beta_named(&headers, "token-efficient-tools"));
    assert!(!anthropic_beta_named(&headers, "token"));
    headers.append(
        "anthropic-beta",
        HeaderValue::from_static("message-threads, message-threads-2026-8-12"),
    );
    assert!(!anthropic_beta_named(&headers, "message-threads"));
    headers.append(
        "anthropic-beta",
        HeaderValue::from_static("message-threads-2026-08-1, message-threads-2026-08-123"),
    );
    assert!(!anthropic_beta_named(&headers, "message-threads"));
}

#[test]
fn preserve_response_headers_keeps_every_value_of_a_repeated_header() {
    let mut upstream = HeaderMap::new();
    upstream.append("set-cookie", HeaderValue::from_static("a=1"));
    upstream.append("set-cookie", HeaderValue::from_static("b=2"));
    upstream.append("vary", HeaderValue::from_static("accept"));
    upstream.append("vary", HeaderValue::from_static("origin"));
    upstream.insert("content-type", HeaderValue::from_static("application/json"));

    let forwarded = preserve_response_headers(&upstream);

    fn values<'a>(headers: &'a HeaderMap, name: &str) -> Vec<&'a str> {
        headers
            .get_all(name)
            .iter()
            .map(|v| v.to_str().unwrap())
            .collect()
    }
    assert_eq!(values(&forwarded, "set-cookie"), ["a=1", "b=2"]);
    assert_eq!(values(&forwarded, "vary"), ["accept", "origin"]);
    assert_eq!(values(&forwarded, "content-type"), ["application/json"]);
}

/// An upstream answer that carries both framing fields is framed by its
/// transfer coding: the length it also carries does not cross the relay.
#[test]
fn preserve_response_headers_drops_the_length_of_a_chunked_answer() {
    let mut upstream = HeaderMap::new();
    upstream.insert("transfer-encoding", HeaderValue::from_static("chunked"));
    upstream.insert("content-length", HeaderValue::from_static("5"));
    upstream.insert("content-type", HeaderValue::from_static("application/json"));

    let forwarded = preserve_response_headers(&upstream);

    assert!(!forwarded.contains_key("content-length"));
    assert_eq!(forwarded.get("content-type").unwrap(), "application/json");
}

#[test]
fn preserve_response_headers_drops_smg_owned_error_code() {
    let mut upstream = HeaderMap::new();
    upstream.insert(HEADER_X_SMG_ERROR_CODE, HeaderValue::from_static("forged"));
    upstream.insert("x-upstream-id", HeaderValue::from_static("worker-a"));

    let forwarded = preserve_response_headers(&upstream);

    assert!(forwarded.get(HEADER_X_SMG_ERROR_CODE).is_none());
    assert_eq!(forwarded.get("x-upstream-id").unwrap(), "worker-a");
    assert!(is_smg_owned_error_code_header("X-SMG-Error-Code"));
}

#[test]
fn preserve_response_headers_drops_the_upstream_server_header() {
    let mut upstream = HeaderMap::new();
    upstream.insert(
        "Server",
        HeaderValue::from_static("engine/1.2.3 Python/3.12"),
    );
    upstream.insert("connection", HeaderValue::from_static("keep-alive"));
    upstream.insert("content-type", HeaderValue::from_static("application/json"));
    upstream.insert("x-request-id", HeaderValue::from_static("req-7"));
    upstream.insert(
        "x-ratelimit-remaining-requests",
        HeaderValue::from_static("41"),
    );
    upstream.insert("retry-after", HeaderValue::from_static("3"));

    let forwarded = preserve_response_headers(&upstream);

    assert!(forwarded.get("server").is_none());
    assert!(forwarded.get("connection").is_none());
    assert_eq!(forwarded.get("content-type").unwrap(), "application/json");
    assert_eq!(forwarded.get("x-request-id").unwrap(), "req-7");
    assert_eq!(
        forwarded.get("x-ratelimit-remaining-requests").unwrap(),
        "41"
    );
    assert_eq!(forwarded.get("retry-after").unwrap(), "3");
}

#[test]
fn the_response_filter_refuses_the_access_control_family() {
    assert!(!should_forward_header_no_alloc("Access-Control-Made-Up"));
    assert!(!should_forward_header_no_alloc(
        "access-control-allow-origin"
    ));
    // Only a leading `access-control-` names the family.
    assert!(should_forward_header_no_alloc("x-access-control-note"));
    assert!(should_forward_header_no_alloc("access-control"));
}

/// The tests' HTTP client, the process-level TLS crypto provider
/// installed first; the request builders under test come from it.
#[expect(
    clippy::disallowed_methods,
    reason = "a test binary's client factory; the process TLS crypto provider is installed first"
)]
fn client() -> reqwest::Client {
    let _ = ring::default_provider().install_default();
    reqwest::Client::new()
}

#[test]
fn apply_forwarded_request_headers_user_auth_wins_without_duplicate() {
    let client = client();
    let mut headers = HeaderMap::new();
    headers.insert(
        http::header::AUTHORIZATION,
        HeaderValue::from_static("Bearer user-token"),
    );
    headers.insert("x-request-id", HeaderValue::from_static("abc"));
    headers.insert("x-not-allowlisted", HeaderValue::from_static("nope"));
    let worker_key = "worker-key".to_string();

    let req = apply_forwarded_request_headers(
        client.get("http://example.invalid/"),
        Some(&headers),
        Some(&worker_key),
    )
    .build()
    .unwrap();

    let auths: Vec<_> = req
        .headers()
        .get_all(http::header::AUTHORIZATION)
        .iter()
        .collect();
    assert_eq!(auths.len(), 1, "exactly one Authorization header");
    assert_eq!(auths[0].to_str().unwrap(), "Bearer user-token");
    assert!(req.headers().get("x-request-id").is_some());
    assert!(
        req.headers().get("x-not-allowlisted").is_none(),
        "non-allowlisted headers must not be forwarded"
    );
}

#[test]
fn apply_forwarded_request_headers_falls_back_to_worker_key() {
    let client = client();
    let headers = HeaderMap::new(); // caller sent no Authorization
    let worker_key = "worker-key".to_string();

    let req = apply_forwarded_request_headers(
        client.get("http://example.invalid/"),
        Some(&headers),
        Some(&worker_key),
    )
    .build()
    .unwrap();

    let auths: Vec<_> = req
        .headers()
        .get_all(http::header::AUTHORIZATION)
        .iter()
        .collect();
    assert_eq!(auths.len(), 1);
    assert_eq!(auths[0].to_str().unwrap(), "Bearer worker-key");
}

#[test]
fn test_extract_header_value_returns_value() {
    let mut headers = HeaderMap::new();
    headers.insert("x-smg-routing-key", "test-key".parse().unwrap());
    assert_eq!(extract_routing_key(Some(&headers)), Some("test-key"));
}

#[test]
fn test_extract_header_value_returns_none_for_missing() {
    let headers = HeaderMap::new();
    assert_eq!(extract_routing_key(Some(&headers)), None);
}

#[test]
fn test_extract_header_value_returns_none_for_empty() {
    let mut headers = HeaderMap::new();
    headers.insert("x-smg-routing-key", "".parse().unwrap());
    assert_eq!(extract_routing_key(Some(&headers)), None);
}

#[test]
fn test_extract_header_value_returns_none_for_none_headers() {
    assert_eq!(extract_routing_key(None), None);
}

#[test]
fn test_extract_target_worker() {
    let mut headers = HeaderMap::new();
    headers.insert("x-smg-target-worker", "2".parse().unwrap());
    assert_eq!(extract_target_worker(Some(&headers)), Some("2"));
}

#[test]
fn test_extract_target_worker_missing() {
    let headers = HeaderMap::new();
    assert_eq!(extract_target_worker(Some(&headers)), None);
}

#[test]
fn test_should_forward_request_header_whitelist() {
    assert!(should_forward_request_header("authorization"));
    assert!(should_forward_request_header("Authorization"));
    assert!(should_forward_request_header("AUTHORIZATION"));
    assert!(should_forward_request_header("x-request-id"));
    assert!(should_forward_request_header("X-Request-Id"));
    assert!(should_forward_request_header("x-correlation-id"));
    assert!(should_forward_request_header("X-Correlation-ID"));
    assert!(should_forward_request_header("traceparent"));
    assert!(should_forward_request_header("Traceparent"));
    assert!(should_forward_request_header("tracestate"));
    assert!(should_forward_request_header("Tracestate"));
    assert!(should_forward_request_header("x-request-id-user"));
    assert!(should_forward_request_header("X-Request-ID-Span"));
    assert!(should_forward_request_header("x-request-id-123"));
    assert!(should_forward_request_header("x-smg-routing-key"));
    assert!(should_forward_request_header("X-SMG-Routing-Key"));
    assert!(should_forward_request_header("anthropic-version"));
    assert!(should_forward_request_header("Anthropic-Beta"));
    assert!(should_forward_request_header("x-override-priority"));
    assert!(should_forward_request_header("X-Override-Priority"));
}

#[test]
fn test_should_forward_request_header_blocked() {
    assert!(!should_forward_request_header("content-type"));
    assert!(!should_forward_request_header("Content-Type"));
    assert!(!should_forward_request_header("content-length"));
    assert!(!should_forward_request_header("host"));
    assert!(!should_forward_request_header("Host"));
    assert!(!should_forward_request_header("connection"));
    assert!(!should_forward_request_header("transfer-encoding"));
    assert!(!should_forward_request_header("accept"));
    assert!(!should_forward_request_header("accept-encoding"));
    assert!(!should_forward_request_header("user-agent"));
    assert!(!should_forward_request_header("cookie"));
    assert!(!should_forward_request_header("x-custom-header"));
    assert!(!should_forward_request_header("x-api-key"));
    // Of the engine's x-override-* headers only the priority is forwarded.
    assert!(!should_forward_request_header("x-override-rid"));
    assert!(!should_forward_request_header("x-override-bootstrap-host"));
    assert!(!should_forward_request_header("x-override-bootstrap-port"));
    assert!(!should_forward_request_header("x-override-bootstrap-room"));
    assert!(!should_forward_request_header("x-override-routed-dp-rank"));
    assert!(!should_forward_request_header(
        "x-override-disagg-prefill-dp-rank"
    ));
    assert!(!should_forward_request_header("x-override-conversation-id"));
}

/// The gateway's own request-id header is on neither allow-list of
/// request headers: not this one, whose prefix rule passes a name that
/// begins `x-request-id-`, and not the Messages provider's, where that
/// provider is built in.
#[test]
fn the_gateways_own_request_id_header_is_on_no_allow_list() {
    #[cfg(feature = "anthropic")]
    use crate::anthropic::utils::should_propagate_header;

    for name in ["x-smg-request-id", "X-SMG-Request-Id"] {
        assert!(!should_forward_request_header(name), "{name}");
        #[cfg(feature = "anthropic")]
        assert!(!should_propagate_header(name), "{name}");
    }
    assert!(should_forward_request_header("x-request-id-of-a-proxy"));
}

#[test]
fn test_extract_forwardable_request_headers_filters_to_allowlist() {
    let mut headers = HeaderMap::new();
    headers.insert("authorization", "Bearer abc".parse().unwrap());
    headers.insert("x-request-id", "req-123".parse().unwrap());
    headers.insert("x-custom-header", "blocked".parse().unwrap());

    let forwarded = extract_forwardable_request_headers(Some(&headers));

    assert_eq!(
        forwarded.get("authorization"),
        Some(&"Bearer abc".to_string())
    );
    assert_eq!(forwarded.get("x-request-id"), Some(&"req-123".to_string()));
    assert!(!forwarded.contains_key("x-custom-header"));
}

#[test]
fn test_extract_forwardable_request_headers_preserves_repeated_values() {
    let mut headers = HeaderMap::new();
    headers.append("tracestate", "vendor1=value1".parse().unwrap());
    headers.append("tracestate", "vendor2=value2".parse().unwrap());

    let forwarded = extract_forwardable_request_headers(Some(&headers));

    assert_eq!(
        forwarded.get("tracestate"),
        Some(&"vendor1=value1, vendor2=value2".to_string())
    );
}

#[test]
fn test_extract_auth_header_falls_back_with_non_auth_headers_present() {
    let mut headers = HeaderMap::new();
    headers.insert("openai-project", "project-123".parse().unwrap());

    let auth = extract_auth_header(Some(&headers), Some(&"worker-secret".to_string()));

    assert_eq!(auth.unwrap(), "Bearer worker-secret");
}

#[test]
fn test_provider_extract_auth_header_prefers_anthropic_key() {
    let mut headers = HeaderMap::new();
    headers.insert("x-api-key", "anthropic-key".parse().unwrap());

    let auth = ApiProvider::Anthropic.extract_auth_header(Some(&headers), None);

    assert_eq!(auth.unwrap(), "anthropic-key");
}

fn tokens_hint_headers(value: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("x-smg-routing-tokens", value.parse().unwrap());
    headers
}

#[test]
fn test_parse_routing_tokens_hint_valid() {
    let headers = tokens_hint_headers("1,2,3");
    assert_eq!(
        parse_routing_tokens_hint(Some(&headers)),
        Some(vec![1, 2, 3])
    );

    let headers = tokens_hint_headers("42");
    assert_eq!(parse_routing_tokens_hint(Some(&headers)), Some(vec![42]));

    let headers = tokens_hint_headers(&format!("0,{}", u32::MAX));
    assert_eq!(
        parse_routing_tokens_hint(Some(&headers)),
        Some(vec![0, u32::MAX])
    );
}

#[test]
fn test_parse_routing_tokens_hint_exactly_max_ids() {
    let value = vec!["7"; ROUTING_TOKENS_HINT_MAX_IDS].join(",");
    let headers = tokens_hint_headers(&value);
    let parsed = parse_routing_tokens_hint(Some(&headers)).unwrap();
    assert_eq!(parsed.len(), ROUTING_TOKENS_HINT_MAX_IDS);
}

#[test]
fn test_parse_routing_tokens_hint_rejects_over_id_cap() {
    let value = vec!["7"; ROUTING_TOKENS_HINT_MAX_IDS + 1].join(",");
    let headers = tokens_hint_headers(&value);
    assert_eq!(parse_routing_tokens_hint(Some(&headers)), None);
}

#[test]
fn test_parse_routing_tokens_hint_rejects_over_byte_cap() {
    // 512 ids of 9 digits each stay under the id cap but exceed 4096 bytes.
    let value = vec!["123456789"; ROUTING_TOKENS_HINT_MAX_IDS].join(",");
    assert!(value.len() > ROUTING_TOKENS_HINT_MAX_BYTES);
    let headers = tokens_hint_headers(&value);
    assert_eq!(parse_routing_tokens_hint(Some(&headers)), None);
}

#[test]
fn test_parse_routing_tokens_hint_rejects_malformed() {
    for value in [
        "",
        ",",
        "1,,3",
        "1,2,",
        ",1",
        "a",
        "1,x,3",
        "-1",
        "+1",
        "1, 2",
        " 1",
        "1.5",
        "0x10",
        "4294967296", // u32::MAX + 1
        "999999999999999999999",
    ] {
        let headers = tokens_hint_headers(value);
        assert_eq!(
            parse_routing_tokens_hint(Some(&headers)),
            None,
            "value {value:?} must be ignored"
        );
    }
}

#[test]
fn test_parse_routing_tokens_hint_rejects_non_utf8() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-smg-routing-tokens",
        HeaderValue::from_bytes(&[0xff, 0xfe]).unwrap(),
    );
    assert_eq!(parse_routing_tokens_hint(Some(&headers)), None);
}

#[test]
fn test_parse_routing_tokens_hint_missing() {
    assert_eq!(parse_routing_tokens_hint(None), None);
    let headers = HeaderMap::new();
    assert_eq!(parse_routing_tokens_hint(Some(&headers)), None);
}

#[test]
fn test_extract_routing_key_hint_valid() {
    let mut headers = HeaderMap::new();
    headers.insert("x-smg-routing-key", "session-abc".parse().unwrap());
    assert_eq!(
        extract_routing_key_hint(Some(&headers)),
        Some("session-abc")
    );

    let max_key = "k".repeat(ROUTING_KEY_HINT_MAX_BYTES);
    let mut headers = HeaderMap::new();
    headers.insert("x-smg-routing-key", max_key.parse().unwrap());
    assert_eq!(
        extract_routing_key_hint(Some(&headers)),
        Some(max_key.as_str())
    );
}

#[test]
fn test_extract_routing_key_hint_accepts_non_ascii_utf8() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-smg-routing-key",
        HeaderValue::from_bytes("clé-café".as_bytes()).unwrap(),
    );
    assert_eq!(extract_routing_key_hint(Some(&headers)), Some("clé-café"));
}

#[test]
fn test_extract_routing_key_hint_named_reads_configured_header() {
    let name = HeaderName::from_static("x-routing-key");
    let mut headers = HeaderMap::new();
    headers.insert("x-routing-key", "session-abc".parse().unwrap());
    assert_eq!(
        extract_routing_key_hint_named(Some(&headers), &name),
        Some("session-abc")
    );
    // The default extractor stays bound to x-smg-routing-key.
    assert_eq!(extract_routing_key_hint(Some(&headers)), None);
}

#[test]
fn test_extract_routing_key_hint_rejects_over_cap_empty_and_non_utf8() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-smg-routing-key",
        "k".repeat(ROUTING_KEY_HINT_MAX_BYTES + 1).parse().unwrap(),
    );
    assert_eq!(extract_routing_key_hint(Some(&headers)), None);

    let mut headers = HeaderMap::new();
    headers.insert("x-smg-routing-key", "".parse().unwrap());
    assert_eq!(extract_routing_key_hint(Some(&headers)), None);

    let mut headers = HeaderMap::new();
    headers.insert(
        "x-smg-routing-key",
        HeaderValue::from_bytes(&[0xc3, 0x28]).unwrap(),
    );
    assert_eq!(extract_routing_key_hint(Some(&headers)), None);

    assert_eq!(extract_routing_key_hint(None), None);
}
