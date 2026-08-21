use std::collections::HashMap;

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, HeaderValue},
};
use http::header::HeaderName;

static HEADER_TARGET_WORKER: HeaderName = HeaderName::from_static("x-smg-target-worker");
static HEADER_ROUTING_KEY: HeaderName = HeaderName::from_static("x-smg-routing-key");
static HEADER_ROUTING_TOKENS: HeaderName = HeaderName::from_static("x-smg-routing-tokens");
static HEADER_MCP: HeaderName = HeaderName::from_static("x-smg-mcp");
static HEADER_ROUTED_WORKER_ID: HeaderName = HeaderName::from_static("x-smg-routed-worker-id");

/// `x-smg-routing-tokens` hard caps: token id count and header-value bytes.
pub const ROUTING_TOKENS_HINT_MAX_IDS: usize = 512;
pub const ROUTING_TOKENS_HINT_MAX_BYTES: usize = 4096;
/// `x-smg-routing-key` hard cap: bytes of opaque UTF-8.
pub const ROUTING_KEY_HINT_MAX_BYTES: usize = 128;

/// Parse the `x-smg-routing-tokens` hint: comma-separated decimal u32 ids.
/// Hints affect placement only, never authorization; malformed or over-cap
/// values are ignored entirely (`None`, falling back to body-derived
/// routing), never an error.
pub fn parse_routing_tokens_hint(headers: Option<&HeaderMap>) -> Option<Vec<u32>> {
    let value = headers?.get(&HEADER_ROUTING_TOKENS)?.as_bytes();
    if value.is_empty() || value.len() > ROUTING_TOKENS_HINT_MAX_BYTES {
        return None;
    }
    let text = std::str::from_utf8(value).ok()?;
    // Every id but the last costs at least two bytes (digit + comma), so this
    // capacity always suffices and the loop never reallocates.
    let mut ids = Vec::with_capacity(ROUTING_TOKENS_HINT_MAX_IDS.min(value.len() / 2 + 1));
    for part in text.split(',') {
        if ids.len() == ROUTING_TOKENS_HINT_MAX_IDS {
            return None;
        }
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        ids.push(part.parse::<u32>().ok()?);
    }
    Some(ids)
}

/// Extract a routing-key hint from `name`: opaque non-empty UTF-8. Over-cap
/// or non-UTF-8 values are ignored (`None`), never an error.
pub fn extract_routing_key_hint_named<'a>(
    headers: Option<&'a HeaderMap>,
    name: &HeaderName,
) -> Option<&'a str> {
    let value = headers?.get(name)?.as_bytes();
    if value.is_empty() || value.len() > ROUTING_KEY_HINT_MAX_BYTES {
        return None;
    }
    std::str::from_utf8(value).ok()
}

/// Extract the `x-smg-routing-key` hint (the default routing-key header).
pub fn extract_routing_key_hint(headers: Option<&HeaderMap>) -> Option<&str> {
    extract_routing_key_hint_named(headers, &HEADER_ROUTING_KEY)
}

/// Declared size of the incoming raw body; pre-sizes reserialization buffers.
pub fn content_length(headers: Option<&HeaderMap>) -> Option<usize> {
    headers?
        .get(http::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

fn extract_header_value<'a>(headers: Option<&'a HeaderMap>, name: &HeaderName) -> Option<&'a str> {
    headers
        .and_then(|h| h.get(name))
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
}

pub fn extract_target_worker(headers: Option<&HeaderMap>) -> Option<&str> {
    extract_header_value(headers, &HEADER_TARGET_WORKER)
}

pub fn extract_routing_key(headers: Option<&HeaderMap>) -> Option<&str> {
    extract_header_value(headers, &HEADER_ROUTING_KEY)
}

/// Check if SMG MCP orchestration is enabled via `X-SMG-MCP: enabled` header.
pub fn is_smg_mcp_enabled(headers: Option<&HeaderMap>) -> bool {
    headers
        .and_then(|h| h.get(&HEADER_MCP))
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("enabled"))
}

/// Copy request headers to a Vec of name-value string pairs
/// Used for forwarding headers to backend workers
pub fn copy_request_headers(req: &Request<Body>) -> Vec<(String, String)> {
    req.headers()
        .iter()
        .filter_map(|(name, value)| {
            // Convert header value to string, skipping non-UTF8 headers
            value
                .to_str()
                .ok()
                .map(|v| (name.to_string(), v.to_string()))
        })
        .collect()
}

/// Convert headers from reqwest Response to axum HeaderMap
/// Filters out hop-by-hop headers that shouldn't be forwarded
pub fn preserve_response_headers(reqwest_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();

    for (name, value) in reqwest_headers {
        // Skip hop-by-hop headers that shouldn't be forwarded
        // Use eq_ignore_ascii_case to avoid string allocation
        if should_forward_header_no_alloc(name.as_str()) {
            // The original name and value are already valid, so we can just clone them
            headers.insert(name.clone(), value.clone());
        }
    }

    headers
}

/// Stamp the response with the id of the worker that served the request — the
/// worker URL as the gateway knows it, including the `@<rank>` suffix for
/// dp-aware workers.
///
/// Lets a client see which replica answered: which pod to pull logs from, and
/// whether retries landed somewhere new. Call after
/// [`preserve_response_headers`] so the gateway's value wins over anything an
/// upstream set under the same name. Skipped if the URL isn't representable as
/// a header value.
pub fn insert_routed_worker_id(headers: &mut HeaderMap, worker_url: &str) {
    if let Ok(value) = HeaderValue::from_str(worker_url) {
        headers.insert(HEADER_ROUTED_WORKER_ID.clone(), value);
    }
}

/// Determine if a header should be forwarded without allocating (case-insensitive)
fn should_forward_header_no_alloc(name: &str) -> bool {
    // List of headers that should NOT be forwarded (hop-by-hop headers)
    // Use eq_ignore_ascii_case to avoid to_lowercase() allocation
    !(name.eq_ignore_ascii_case("connection")
        || name.eq_ignore_ascii_case("keep-alive")
        || name.eq_ignore_ascii_case("proxy-authenticate")
        || name.eq_ignore_ascii_case("proxy-authorization")
        || name.eq_ignore_ascii_case("te")
        || name.eq_ignore_ascii_case("trailers")
        || name.eq_ignore_ascii_case("transfer-encoding")
        || name.eq_ignore_ascii_case("upgrade")
        || name.eq_ignore_ascii_case("content-encoding")
        || name.eq_ignore_ascii_case("host"))
}

/// API provider types for provider-specific header handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiProvider {
    Anthropic,
    Xai,
    OpenAi,
    Gemini,
    Generic,
}

impl ApiProvider {
    /// Detect provider type from URL
    pub fn from_url(url: &str) -> Self {
        if url.contains("anthropic") {
            ApiProvider::Anthropic
        } else if url.contains("x.ai") {
            ApiProvider::Xai
        } else if url.contains("openai.com") {
            ApiProvider::OpenAi
        } else if url.contains("googleapis.com") {
            ApiProvider::Gemini
        } else {
            ApiProvider::Generic
        }
    }

    /// Extract auth credential from request headers with provider-specific logic.
    ///
    /// - **Gemini**: prefers `x-goog-api-key`, then `Authorization`, then worker key.
    /// - **Anthropic**: prefers `x-api-key`, then `Authorization`, then worker key.
    /// - **All others**: prefers `Authorization`, then worker key with `Bearer` prefix.
    pub fn extract_auth_header(
        self,
        headers: Option<&HeaderMap>,
        worker_api_key: Option<&String>,
    ) -> Option<HeaderValue> {
        if let Some(h) = headers {
            match self {
                ApiProvider::Anthropic => {
                    // Prefer x-api-key
                    if let Some(v) = h.get("x-api-key").and_then(|v| {
                        v.to_str()
                            .ok()
                            .filter(|s| !s.trim().is_empty())
                            .map(|_| v.clone())
                    }) {
                        return Some(v);
                    }
                }
                ApiProvider::Gemini => {
                    // Prefer x-goog-api-key
                    if let Some(v) = h.get("x-goog-api-key").and_then(|v| {
                        v.to_str()
                            .ok()
                            .filter(|s| !s.trim().is_empty())
                            .map(|_| v.clone())
                    }) {
                        return Some(v);
                    }
                }
                _ => {}
            }
        }

        // Standard: Authorization header first, then worker key with Bearer
        extract_auth_header(headers, worker_api_key)
    }

    /// Apply provider-specific auth headers to a reqwest request builder.
    ///
    /// - **Anthropic**: strips `Bearer` prefix and sets `x-api-key` + `anthropic-version`.
    /// - **Gemini**: strips `Bearer` prefix and sets `x-goog-api-key`.
    /// - **Others**: forwards the `Authorization` header as-is.
    pub fn apply_headers(
        self,
        mut req: reqwest::RequestBuilder,
        auth_header: Option<&HeaderValue>,
    ) -> reqwest::RequestBuilder {
        match self {
            ApiProvider::Anthropic => {
                if let Some(auth) = auth_header {
                    if let Ok(auth_str) = auth.to_str() {
                        // Strip Bearer scheme case-insensitively (RFC 7235)
                        let api_key = auth_str
                            .split_once(' ')
                            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
                            .map(|(_, token)| token)
                            .unwrap_or(auth_str)
                            .trim();
                        if !api_key.is_empty() {
                            req = req
                                .header("x-api-key", api_key)
                                .header("anthropic-version", "2023-06-01");
                        }
                    }
                }
            }
            ApiProvider::Gemini => {
                if let Some(auth) = auth_header {
                    if let Ok(auth_str) = auth.to_str() {
                        let api_key = auth_str
                            .split_once(' ')
                            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
                            .map(|(_, token)| token)
                            .unwrap_or(auth_str)
                            .trim();
                        if !api_key.is_empty() {
                            req = req.header("x-goog-api-key", api_key);
                        }
                    }
                }
            }
            ApiProvider::Xai | ApiProvider::OpenAi | ApiProvider::Generic => {
                if let Some(auth) = auth_header {
                    req = req.header("Authorization", auth);
                }
            }
        }

        req
    }
}

/// Apply provider-specific headers to request
pub fn apply_provider_headers(
    req: reqwest::RequestBuilder,
    url: &str,
    auth_header: Option<&HeaderValue>,
) -> reqwest::RequestBuilder {
    ApiProvider::from_url(url).apply_headers(req, auth_header)
}

/// Extract auth header with passthrough semantics.
///
/// Passthrough mode: User's Authorization header takes priority.
/// Fallback: Worker's API key is used only if user didn't provide auth.
///
/// This enables use cases where:
/// 1. Users send their own API keys (multi-tenant, BYOK)
/// 2. Router has a default key for users who don't provide one
pub fn extract_auth_header(
    headers: Option<&HeaderMap>,
    worker_api_key: Option<&String>,
) -> Option<HeaderValue> {
    // Passthrough: Try user's auth header first
    let user_auth = headers.and_then(|h| {
        h.get("authorization")
            .or_else(|| h.get("Authorization"))
            .cloned()
    });

    // Return user's auth if provided, otherwise use worker's API key
    user_auth
        .or_else(|| worker_api_key.and_then(|k| HeaderValue::from_str(&format!("Bearer {k}")).ok()))
}

/// Apply the effective `Authorization` header plus every other forwardable
/// request header to an outbound reqwest builder, without ever emitting a
/// duplicate `Authorization`.
///
/// `reqwest::RequestBuilder::header` appends rather than replaces, so setting the
/// worker API key and *then* forwarding the caller's `Authorization` separately
/// sends two `Authorization` headers (worker-key-first) and inverts the
/// passthrough precedence documented on [`extract_auth_header`]. Callers that
/// proxy a request to a worker should use this instead of doing both: it resolves
/// the single correct value (user header wins, worker key is the fallback) and
/// forwards every other allow-listed header.
pub fn apply_forwarded_request_headers(
    mut builder: reqwest::RequestBuilder,
    headers: Option<&HeaderMap>,
    worker_api_key: Option<&String>,
) -> reqwest::RequestBuilder {
    if let Some(auth) = extract_auth_header(headers, worker_api_key) {
        builder = builder.header(http::header::AUTHORIZATION, auth);
    }

    if let Some(headers) = headers {
        for (name, value) in headers {
            // Authorization is applied above with the correct precedence; never
            // forward it again or reqwest appends a second header.
            if name.as_str().eq_ignore_ascii_case("authorization") {
                continue;
            }
            if should_forward_request_header(name.as_str()) {
                builder = builder.header(name, value);
            }
        }
    }

    builder
}

/// Extract the subset of request headers that SMG is allowed to preserve for
/// internal execution paths such as MCP tool calls.
pub fn extract_forwardable_request_headers(headers: Option<&HeaderMap>) -> HashMap<String, String> {
    let Some(headers) = headers else {
        return HashMap::new();
    };

    let mut forwarded = HashMap::new();

    for (name, value) in headers {
        if !should_forward_request_header(name.as_str()) {
            continue;
        }

        let Ok(value) = value.to_str() else {
            continue;
        };

        forwarded
            .entry(name.as_str().to_string())
            .and_modify(|existing: &mut String| {
                existing.push_str(", ");
                existing.push_str(value);
            })
            .or_insert_with(|| value.to_string());
    }

    forwarded
}

#[inline]
pub fn should_forward_request_header(name: &str) -> bool {
    const REQUEST_ID_PREFIX: &str = "x-request-id-";

    name.eq_ignore_ascii_case("authorization")
        || name.eq_ignore_ascii_case("x-request-id")
        || name.eq_ignore_ascii_case("x-correlation-id")
        || name.eq_ignore_ascii_case("traceparent")
        || name.eq_ignore_ascii_case("tracestate")
        || name.eq_ignore_ascii_case("x-smg-routing-key")
        || name
            .get(..REQUEST_ID_PREFIX.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(REQUEST_ID_PREFIX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_forwarded_request_headers_user_auth_wins_without_duplicate() {
        let client = reqwest::Client::new();
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
        let client = reqwest::Client::new();
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
}
