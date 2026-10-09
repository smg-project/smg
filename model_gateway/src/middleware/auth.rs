//! Bearer-token auth, keyed by SHA-256 hash of the presented token rather
//! than the raw value. Hashing first — not a constant-time comparison — is
//! what makes lookup safe: SHA-256's avalanche property means a guess that's
//! "close" to a valid token produces an unrelated hash, so there's no
//! byte-at-a-time timing signal to recover from a plain hash-keyed lookup,
//! unlike comparing raw secrets.

use std::collections::HashMap;

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use smg_auth::bearer_token;

use crate::{
    config::TenantApiKeyEntry,
    tenant::{authenticated_tenant_key_from_sha256, DataPlaneCaller, TenantIdentity, TenantKey},
};

#[derive(Clone)]
pub struct AuthConfig {
    keys: HashMap<[u8; 32], TenantKey>,
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

        Self { keys }
    }

    /// Whether any key is configured. Empty means [`auth_middleware`]
    /// passes every request through.
    pub fn is_enabled(&self) -> bool {
        !self.keys.is_empty()
    }

    /// Whether `token` matches any configured key (shared or per-tenant).
    /// For callers outside `auth_middleware` that need to distinguish "this
    /// is one of our own gateway credentials" from "unrecognized token" —
    /// e.g. `/v1/models`' BYOK short-circuit, which must not treat a
    /// tenant-scoped key as a foreign upstream-provider credential and
    /// forward it externally.
    pub fn contains_token(&self, token: &str) -> bool {
        self.keys.contains_key(&hash_key(token))
    }
}

fn hash_key(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

/// Middleware to validate Bearer token against configured API key(s).
/// Only active when the router has at least one key configured.
pub async fn auth_middleware(
    State(auth_config): State<AuthConfig>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    if !auth_config.keys.is_empty() {
        let token_hash = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|h| h.to_str().ok())
            .and_then(bearer_token)
            .map(hash_key);

        let tenant_key = token_hash
            .as_ref()
            .and_then(|hash| auth_config.keys.get(hash));

        let Some(tenant_key) = tenant_key else {
            return unauthorized(token_hash.is_some());
        };

        request
            .extensions_mut()
            .insert(DataPlaneCaller::new(tenant_key.clone()));
    }

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
    (
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
        .into_response()
}

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{header, Request, StatusCode},
        middleware::from_fn_with_state,
        response::IntoResponse,
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    use super::*;

    async fn handler(request: Request<Body>) -> impl IntoResponse {
        request
            .extensions()
            .get::<DataPlaneCaller>()
            .map(|caller| caller.tenant_key().to_string())
            .unwrap_or_else(|| "missing".to_string())
    }

    fn app(auth_config: AuthConfig) -> Router {
        Router::new()
            .route("/", get(handler))
            .route_layer(from_fn_with_state(auth_config, auth_middleware))
    }

    async fn body_text(response: Response) -> String {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        std::str::from_utf8(&body).unwrap().to_string()
    }

    #[tokio::test]
    async fn no_key_configured_allows_any_request() {
        let response = app(AuthConfig::new(None))
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn shared_key_resolves_to_hash_derived_tenant() {
        let auth_config = AuthConfig::new(Some("shared-secret".to_string()));
        let expected = authenticated_tenant_key_from_sha256(hash_key("shared-secret"));

        let response = app(auth_config)
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(header::AUTHORIZATION, "Bearer shared-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_text(response).await, expected.to_string());
    }

    #[tokio::test]
    async fn tenant_key_resolves_to_explicit_tenant_id() {
        let auth_config = AuthConfig::with_tenant_keys(
            None,
            &[TenantApiKeyEntry {
                tenant_id: "team-red".to_string(),
                key: "team-red-secret".to_string(),
            }],
        );

        let response = app(auth_config)
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(header::AUTHORIZATION, "Bearer team-red-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_text(response).await, "auth:team-red");
    }

    #[tokio::test]
    async fn distinct_tenant_keys_resolve_to_distinct_tenants() {
        let auth_config = AuthConfig::with_tenant_keys(
            None,
            &[
                TenantApiKeyEntry {
                    tenant_id: "team-red".to_string(),
                    key: "red-secret".to_string(),
                },
                TenantApiKeyEntry {
                    tenant_id: "team-blue".to_string(),
                    key: "blue-secret".to_string(),
                },
            ],
        );

        for (key, expected_tenant) in [
            ("red-secret", "auth:team-red"),
            ("blue-secret", "auth:team-blue"),
        ] {
            let response = app(auth_config.clone())
                .oneshot(
                    Request::builder()
                        .uri("/")
                        .header(header::AUTHORIZATION, format!("Bearer {key}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(body_text(response).await, expected_tenant);
        }
    }

    #[tokio::test]
    async fn unknown_token_is_rejected() {
        let auth_config = AuthConfig::with_tenant_keys(
            None,
            &[TenantApiKeyEntry {
                tenant_id: "team-red".to_string(),
                key: "team-red-secret".to_string(),
            }],
        );

        let response = app(auth_config)
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(header::AUTHORIZATION, "Bearer not-a-configured-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn missing_header_is_rejected_when_keys_configured() {
        let auth_config = AuthConfig::new(Some("shared-secret".to_string()));

        let response = app(auth_config)
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn bearer_scheme_is_matched_case_insensitively() {
        let auth_config = AuthConfig::new(Some("shared-secret".to_string()));

        for scheme in ["bearer", "BEARER"] {
            let response = app(auth_config.clone())
                .oneshot(
                    Request::builder()
                        .uri("/")
                        .header(header::AUTHORIZATION, format!("{scheme} shared-secret"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::OK, "scheme {scheme}");
        }
    }

    fn challenge(response: &Response) -> String {
        response
            .headers()
            .get(header::WWW_AUTHENTICATE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string()
    }

    #[tokio::test]
    async fn missing_key_gets_a_bearer_challenge_and_a_message() {
        let response = app(AuthConfig::new(Some("shared-secret".to_string())))
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(challenge(&response), "Bearer realm=\"data-plane\"");
        let body: serde_json::Value = serde_json::from_str(&body_text(response).await).unwrap();
        assert_eq!(body["error"]["code"], "invalid_api_key");
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| !message.is_empty()),
            "{body}"
        );
    }

    #[tokio::test]
    async fn unknown_key_gets_an_invalid_token_challenge() {
        let response = app(AuthConfig::new(Some("shared-secret".to_string())))
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header(header::AUTHORIZATION, "Bearer not-a-configured-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            challenge(&response),
            "Bearer realm=\"data-plane\", error=\"invalid_token\""
        );
    }

    #[tokio::test]
    async fn deny_all_carries_the_challenge() {
        let response = Router::new()
            .route("/", get(handler))
            .route_layer(axum::middleware::from_fn(deny_all_middleware))
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(challenge(&response), "Bearer realm=\"data-plane\"");
    }

    #[test]
    fn contains_token_recognizes_shared_and_tenant_keys() {
        let auth_config = AuthConfig::with_tenant_keys(
            Some("shared-secret".to_string()),
            &[TenantApiKeyEntry {
                tenant_id: "team-red".to_string(),
                key: "team-red-secret".to_string(),
            }],
        );

        assert!(auth_config.contains_token("shared-secret"));
        assert!(auth_config.contains_token("team-red-secret"));
        assert!(!auth_config.contains_token("not-a-configured-key"));
    }
}
