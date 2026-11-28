use std::collections::HashMap;

use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, HeaderValue},
};
use http::header::HeaderName;
use openai_protocol::messages::MessagesBetas;

use crate::error::HEADER_X_SMG_ERROR_CODE;

static HEADER_TARGET_WORKER: HeaderName = HeaderName::from_static("x-smg-target-worker");
static HEADER_ROUTING_KEY: HeaderName = HeaderName::from_static("x-smg-routing-key");
static HEADER_ROUTING_TOKENS: HeaderName = HeaderName::from_static("x-smg-routing-tokens");
static HEADER_MCP: HeaderName = HeaderName::from_static("x-smg-mcp");
/// The response header [`insert_routed_worker_id`] stamps.
pub static HEADER_ROUTED_WORKER_ID: HeaderName = HeaderName::from_static("x-smg-routed-worker-id");

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

/// Whether the request's `anthropic-beta` header names the beta `name` in
/// its dated form, `name-YYYY-MM-DD` (the reading of [`MessagesBetas`],
/// without building the set): the header may repeat, and each value may list
/// several betas, comma separated. A longer beta name that begins with
/// `name` is another beta, not this one.
pub fn anthropic_beta_named(headers: &HeaderMap, name: &str) -> bool {
    MessagesBetas::values_name(
        headers
            .get_all("anthropic-beta")
            .iter()
            .filter_map(|value| value.to_str().ok()),
        name,
    )
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

/// Convert headers from reqwest Response to axum HeaderMap.
///
/// Filters out hop-by-hop and SMG-owned headers that must not be forwarded, and
/// the upstream's `server` header: the upstream's software and version are not the
/// client's to read, and the gateway's own answers carry no `server` header. An
/// answer that also carries `transfer-encoding` loses its `content-length`
/// (RFC 9112, 6.3): the gateway's own framing describes the body it sends.
/// The upstream's `access-control-*` fields stay behind as well: only the
/// gateway's CORS layer writes them.
pub fn preserve_response_headers(reqwest_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();
    // A message with both fields is framed by its transfer coding: the length
    // it also carries is removed before the message goes on (RFC 9112, 6.3).
    let framed_by_transfer_coding = reqwest_headers.contains_key(http::header::TRANSFER_ENCODING);

    for (name, value) in reqwest_headers {
        if framed_by_transfer_coding && name == http::header::CONTENT_LENGTH {
            continue;
        }
        // Use case-insensitive checks to avoid string allocation.
        if should_forward_header_no_alloc(name.as_str()) {
            // One pair per value: a header the upstream sent more than once
            // (`set-cookie`, `vary`, `link`, `www-authenticate`) keeps every
            // value, in order. The name and value are already valid: a clone.
            headers.append(name.clone(), value.clone());
        }
    }

    headers
}

/// Stamp the response with the id of the worker that served the request: the
/// rendering of its URL that the caller passes. Callers must pass it with any
/// userinfo masked (keeping a dp-aware worker's `@<rank>` suffix), since the
/// value reaches every client and the access log.
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

/// Whether `name` is SMG's own error-code response header.
///
/// Backend responses must not mint this marker: downstream gateways use it to
/// distinguish a gateway decision from an upstream response.
pub fn is_smg_owned_error_code_header(name: &str) -> bool {
    name.eq_ignore_ascii_case(HEADER_X_SMG_ERROR_CODE)
}

/// Whether an upstream response header crosses the relay (case-insensitive, no allocation).
///
/// Every HTTP relay reads this one list: `preserve_response_headers` for the buffered
/// and the OpenAI-style streamed answers, `anthropic::sse::build_sse_response` for the
/// Messages streamed ones.
pub(crate) fn should_forward_header_no_alloc(name: &str) -> bool {
    const ACCESS_CONTROL_PREFIX: &str = "access-control-";

    // List of headers that should NOT be forwarded: hop-by-hop, the SMG error-code
    // marker, the upstream's `server` header, which names the upstream's software
    // and version to every client while the gateway's own answers carry none, and
    // the `access-control-*` family, which only the gateway's CORS layer writes.
    !(name.eq_ignore_ascii_case("connection")
        || name.eq_ignore_ascii_case("keep-alive")
        || name.eq_ignore_ascii_case("proxy-authenticate")
        || name.eq_ignore_ascii_case("proxy-authorization")
        || name.eq_ignore_ascii_case("te")
        || name.eq_ignore_ascii_case("trailers")
        || name.eq_ignore_ascii_case("transfer-encoding")
        || name.eq_ignore_ascii_case("upgrade")
        || name.eq_ignore_ascii_case("content-encoding")
        || name.eq_ignore_ascii_case("host")
        || name.eq_ignore_ascii_case("server")
        || name
            .get(..ACCESS_CONTROL_PREFIX.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(ACCESS_CONTROL_PREFIX))
        || is_smg_owned_error_code_header(name))
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

/// Whether a worker is sent a request header of this name: the routers' one
/// allow-list of request headers. The gateway's own request-id header,
/// `x-smg-request-id`, never joins it: the gateway states its own id of a
/// request to Grand Central under that name, and no worker is sent it.
#[inline]
pub fn should_forward_request_header(name: &str) -> bool {
    const REQUEST_ID_PREFIX: &str = "x-request-id-";

    name.eq_ignore_ascii_case("authorization")
        // Anthropic Messages protocol headers: engines serving /v1/messages
        // select API version and beta behavior from them.
        || name.eq_ignore_ascii_case("anthropic-version")
        || name.eq_ignore_ascii_case("anthropic-beta")
        || name.eq_ignore_ascii_case("x-request-id")
        || name.eq_ignore_ascii_case("x-correlation-id")
        || name.eq_ignore_ascii_case("traceparent")
        || name.eq_ignore_ascii_case("tracestate")
        || name.eq_ignore_ascii_case("x-smg-routing-key")
        // SGLang's scheduling priority, read by an engine started with
        // SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES. The other x-override-* headers stay
        // blocked: the engine would apply them over the PD, DP-rank and rid fields.
        || name.eq_ignore_ascii_case("x-override-priority")
        || name
            .get(..REQUEST_ID_PREFIX.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(REQUEST_ID_PREFIX))
}

#[cfg(test)]
#[path = "../tests/unit/header_utils/tests.rs"]
mod tests;
