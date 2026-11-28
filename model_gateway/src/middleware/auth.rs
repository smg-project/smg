//! Bearer-token auth, keyed by SHA-256 hash of the presented token rather
//! than the raw value. Hashing first — not a constant-time comparison — is
//! what makes lookup safe: SHA-256's avalanche property means a guess that's
//! "close" to a valid token produces an unrelated hash, so there's no
//! byte-at-a-time timing signal to recover from a plain hash-keyed lookup,
//! unlike comparing raw secrets.
//!
//! A bearer that is one of the gateway's own keys ends here: once the caller
//! is identified ([`DataPlaneCaller`]) the `Authorization` header is removed
//! from the request, and so is the same key presented under a provider's own
//! header name (`x-api-key`, `x-goog-api-key`), so no route behind this
//! middleware reads the gateway's key as the caller's credential for an
//! upstream. Those upstreams (a self-hosted worker, a third-party provider)
//! get their own configured key, as they do for a request that carried no
//! header: the HTTP and PD routers' upstream calls and the external providers apply
//! the caller's header, else the configured key; an upstream configured
//! without a key is reached with none. A bearer that is not a gateway key is
//! refused before that, so a request that reaches a route with its
//! `Authorization` still on it was not authenticated by this middleware (auth
//! disabled). The public GET proxies (model and server info, the health
//! probe) sit outside it: they forward the caller's header as they did and
//! add no credential of their own, since there "no header" means anonymous.

use std::{collections::HashMap, fmt::Write as _, sync::Arc};

use arc_swap::ArcSwap;
use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use smg_auth::bearer_token;

use super::logging::AccessLogRecord;
use crate::{
    config::TenantApiKeyEntry,
    tenant::{authenticated_tenant_key_from_sha256, DataPlaneCaller, TenantIdentity, TenantKey},
};

/// The configured keys by hash, built at start and swapped whole by a
/// reload of the key files. The map is shared: the clone axum takes of the
/// middleware's state on every request is a reference count, whatever the
/// number of keys, and a request takes the map once.
#[derive(Clone)]
pub struct AuthConfig {
    keys: Arc<ArcSwap<HashMap<[u8; 32], TenantKey>>>,
}

impl AuthConfig {
    /// Single shared key; every caller resolves to the same hash-derived
    /// identity. Use [`Self::with_tenant_keys`] for per-tenant separation.
    pub fn new(api_key: Option<String>) -> Self {
        Self::with_tenant_keys(api_key, &[])
    }

    /// `tenant_api_keys` adds per-tenant keys on top of `api_key`, each
    /// resolving to its own `auth:<tenant_id>` identity.
    pub fn with_tenant_keys(
        api_key: Option<String>,
        tenant_api_keys: &[TenantApiKeyEntry],
    ) -> Self {
        Self {
            keys: Arc::new(ArcSwap::from_pointee(Self::key_map(
                api_key,
                tenant_api_keys,
            ))),
        }
    }

    /// The keys by hash: the shared key's hash-derived identity, then each
    /// tenant key's own.
    pub(crate) fn key_map(
        api_key: Option<String>,
        tenant_api_keys: &[TenantApiKeyEntry],
    ) -> HashMap<[u8; 32], TenantKey> {
        let mut keys = HashMap::with_capacity(tenant_api_keys.len() + 1);

        if let Some(key) = api_key {
            let key_hash = hash_key(&key);
            keys.insert(key_hash, authenticated_tenant_key_from_sha256(key_hash));
        }

        for entry in tenant_api_keys {
            keys.insert(
                hash_key(&entry.key),
                TenantIdentity::Authenticated(entry.tenant_id.as_str().into()).into_key(),
            );
        }

        keys
    }

    /// Put `keys` in force and return the map they replace; a request that
    /// took the previous map finishes under it.
    pub(crate) fn replace(
        &self,
        keys: HashMap<[u8; 32], TenantKey>,
    ) -> Arc<HashMap<[u8; 32], TenantKey>> {
        self.keys.swap(Arc::new(keys))
    }

    /// Whether any key is configured. Empty means [`auth_middleware`]
    /// passes every request through.
    pub fn is_enabled(&self) -> bool {
        !self.keys.load().is_empty()
    }

    /// Resolve the bearer against the current key map without consuming it.
    pub(crate) fn require_tenant(&self, headers: &HeaderMap) -> Result<TenantKey, Response> {
        let hash = bearer_hash(headers);
        let keys = self.keys.load();
        hash.as_ref()
            .and_then(|hash| keys.get(hash))
            .cloned()
            .ok_or_else(|| unauthorized(hash.is_some()))
    }

    /// Whether `token` matches any configured key (shared or per-tenant).
    /// For callers outside `auth_middleware` that need to distinguish "this
    /// is one of our own gateway credentials" from "unrecognized token" —
    /// e.g. `/v1/models`' BYOK short-circuit, which must not treat a
    /// tenant-scoped key as a foreign upstream-provider credential and
    /// forward it externally.
    pub fn contains_token(&self, token: &str) -> bool {
        self.keys.load().contains_key(&hash_key(token))
    }
}

fn bearer_hash(headers: &HeaderMap) -> Option<[u8; 32]> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(bearer_token)
        .map(hash_key)
}

fn hash_key(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

/// The first 16 hex digits of the key's SHA-256: how the access log and the
/// configuration view name a key without writing it.
pub fn key_fingerprint(key: &str) -> String {
    fingerprint_of(&hash_key(key))
}

/// [`key_fingerprint`] of a key already hashed.
pub(crate) fn fingerprint_of(key_hash: &[u8; 32]) -> String {
    let mut id = String::with_capacity(16);
    for byte in &key_hash[..8] {
        // Writing into a String cannot fail.
        let _ = write!(id, "{byte:02x}");
    }
    id
}

/// The header names a provider reads its own key from, which an SDK may fill
/// with the gateway's key beside the bearer (`x-api-key` for the Messages
/// API's providers, `x-goog-api-key` for Gemini's).
const PROVIDER_KEY_HEADERS: [&str; 2] = ["x-api-key", "x-goog-api-key"];

/// Middleware to validate Bearer token against configured API key(s).
/// Only active when the router has at least one key configured. A matched
/// key leaves the request with the caller's identity in its place (see the
/// module doc).
pub async fn auth_middleware(
    State(auth_config): State<AuthConfig>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    // The map is read once and let go before the inner service runs: a
    // guard kept across a streaming response would hold one of the thread's
    // few fast load slots for the response's whole life, and every later
    // load on that thread, this map's or any other swap's, would pay the
    // shared count instead.
    let keys = auth_config.keys.load();
    if !keys.is_empty() {
        let token_hash = bearer_hash(request.headers());

        let tenant_key = token_hash.as_ref().and_then(|hash| keys.get(hash));

        let Some(tenant_key) = tenant_key else {
            return unauthorized(token_hash.is_some());
        };

        // The access log, when it carries the key: by its fingerprint, never
        // the key.
        if let (Some(record), Some(hash)) = (
            request.extensions().get::<AccessLogRecord>(),
            token_hash.as_ref(),
        ) {
            record.key_id(hash);
        }

        request
            .extensions_mut()
            .insert(DataPlaneCaller::new(tenant_key.clone()));
        // The gateway's key is spent: it identifies the caller and nothing
        // else, and it must not travel on as an upstream's credential; a
        // gateway key presented a second time under a provider's own header
        // name goes with it (every value of a repeated header is read, since
        // the removal takes the whole name), where a caller's own provider
        // key there stays. The map already in hand answers, not a second
        // load.
        request.headers_mut().remove(header::AUTHORIZATION);
        for name in PROVIDER_KEY_HEADERS {
            let is_a_gateway_key = request
                .headers()
                .get_all(name)
                .iter()
                .filter_map(|value| value.to_str().ok())
                .any(|value| keys.contains_key(&hash_key(value.trim())));
            if is_a_gateway_key {
                request.headers_mut().remove(name);
            }
        }
    }
    drop(keys);

    next.run(request).await
}

/// Unconditionally rejects with 401 — for route groups that must never fall
/// back to open just because their auth config happens to be empty.
pub async fn deny_all_middleware(request: Request<Body>, _next: Next) -> Response {
    let token_presented = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(bearer_token)
        .is_some();
    unauthorized(token_presented)
}

/// Internal provenance for serving-plane refusals. Upstream providers may
/// independently return HTTP 401.
#[derive(Clone)]
pub(crate) struct ServingAuthRejected;

/// The serving plane's 401: the `WWW-Authenticate` challenge RFC 6750
/// requires (`error="invalid_token"` when a token was presented but is not a
/// configured key) and an OpenAI-style error body, so SDK clients surface a
/// message instead of a blank error.
fn unauthorized(token_presented: bool) -> Response {
    let (challenge, message) = if token_presented {
        (
            "Bearer realm=\"data-plane\", error=\"invalid_token\"",
            "Invalid API key",
        )
    } else {
        (
            "Bearer realm=\"data-plane\"",
            "Missing API key: send it as `Authorization: Bearer <key>`",
        )
    };
    let mut response = (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, challenge)],
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": null,
                "code": "invalid_api_key",
            }
        })),
    )
        .into_response();
    response.extensions_mut().insert(ServingAuthRejected);
    response
}

#[cfg(test)]
#[path = "../../tests/unit/middleware/auth/tests.rs"]
mod tests;
