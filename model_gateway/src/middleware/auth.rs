//! Bearer-token auth, keyed by SHA-256 hash of the presented token rather
//! than the raw value. Hashing first — not a constant-time comparison — is
//! what makes lookup safe: SHA-256's avalanche property means a guess that's
//! "close" to a valid token produces an unrelated hash, so there's no
//! byte-at-a-time timing signal to recover from a plain hash-keyed lookup,
//! unlike comparing raw secrets.
//!
//! With client-certificate authentication on the HTTPS listener
//! (`--tls-client-auth`), the connection's verified certificate comes first:
//! the identity it names (a [`ClientCertificate`] extension, attached by the
//! listener's acceptor) is the caller's tenant and no bearer is consulted;
//! under `required` a request without that identity is refused; under
//! `optional` it falls through to the bearer check as before.

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

use super::logging::AccessLogRecord;
use crate::{
    config::{TenantApiKeyEntry, TlsClientAuth},
    server::client_identity::ClientCertificate,
    tenant::{authenticated_tenant_key_from_sha256, DataPlaneCaller, TenantIdentity, TenantKey},
};

#[derive(Clone)]
pub struct AuthConfig {
    keys: HashMap<[u8; 32], TenantKey>,
    client_auth: TlsClientAuth,
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

        Self {
            keys,
            client_auth: TlsClientAuth::Off,
        }
    }

    /// The configuration with client certificates of the HTTPS listener as a
    /// credential: a connection's verified certificate identity is the
    /// caller's tenant; under `Required` a request without one is refused.
    /// For the serving routes only: the admin routes' shared-key fallback
    /// does not take certificate identities (they are tenants).
    pub fn with_client_certificates(mut self, mode: TlsClientAuth) -> Self {
        self.client_auth = mode;
        self
    }

    /// Whether the serving routes require a credential: a key is configured
    /// or client certificates are required. Neither means [`auth_middleware`]
    /// passes every request through (an optional client certificate still
    /// identifies the caller when presented).
    pub fn is_enabled(&self) -> bool {
        !self.keys.is_empty() || self.client_auth == TlsClientAuth::Required
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

/// Middleware to authenticate the caller: by the connection's verified
/// client certificate when the listener requests one, else by the Bearer
/// token against the configured API key(s). Only active when the router has
/// at least one key configured or requests client certificates.
pub async fn auth_middleware(
    State(auth_config): State<AuthConfig>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    if auth_config.client_auth.is_on() {
        let certificate = request.extensions().get::<ClientCertificate>().cloned();
        let required = auth_config.client_auth == TlsClientAuth::Required;
        match certificate {
            Some(ClientCertificate::Identified(identity)) => {
                request.extensions_mut().insert(DataPlaneCaller::new(
                    TenantIdentity::Authenticated(identity).into_key(),
                ));
                return next.run(request).await;
            }
            Some(ClientCertificate::Unidentified) if required => {
                return forbidden_certificate();
            }
            None if required => {
                return unauthorized_certificate();
            }
            // Optional: a connection without an identity goes on to the bearer check.
            _ => {}
        }
    }

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
    }

    next.run(request).await
}

/// A verified client certificate that names no identity in the configured
/// field, where one is required: the caller is authenticated at the TLS
/// layer but cannot be a tenant, so 403, with the field to fix.
fn forbidden_certificate() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(json!({
            "error": {
                "message": "The client certificate names no identity in the configured field (see --tls-client-identity)",
                "type": "invalid_request_error",
                "param": null,
                "code": "client_certificate_unidentified",
            }
        })),
    )
        .into_response()
}

/// No client certificate on a route that requires one: a request that did
/// not come through the listener's client authentication.
fn unauthorized_certificate() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, "Bearer realm=\"data-plane\"")],
        Json(json!({
            "error": {
                "message": "A client certificate is required on this route",
                "type": "invalid_request_error",
                "param": null,
                "code": "client_certificate_required",
            }
        })),
    )
        .into_response()
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
    use std::sync::Arc;

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

    fn with_certificate(certificate: Option<ClientCertificate>) -> Request<Body> {
        let mut request = Request::builder().uri("/");
        if let Some(certificate) = certificate {
            request = request.extension(certificate);
        }
        request.body(Body::empty()).unwrap()
    }

    fn identified(identity: &str) -> ClientCertificate {
        ClientCertificate::Identified(Arc::from(identity))
    }

    /// A verified certificate's identity is the tenant, with or without keys
    /// and whatever bearer the request carries.
    #[tokio::test]
    async fn a_client_certificate_identity_is_the_tenant() {
        for (auth_config, mode) in [
            (AuthConfig::new(None), TlsClientAuth::Required),
            (AuthConfig::new(None), TlsClientAuth::Optional),
            (
                AuthConfig::new(Some("shared-secret".to_string())),
                TlsClientAuth::Required,
            ),
        ] {
            let app = app(auth_config.with_client_certificates(mode));
            let spiffe = "spiffe://cluster.local/ns/default/sa/team-red";
            let mut request = with_certificate(Some(identified(spiffe)));
            request.headers_mut().insert(
                header::AUTHORIZATION,
                "Bearer not-a-configured-key".parse().unwrap(),
            );
            let response = app.oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{mode}");
            assert_eq!(
                body_text(response).await,
                format!("auth:{spiffe}"),
                "{mode}"
            );
        }
    }

    /// Required: a certificate without an identity is refused as forbidden,
    /// a request without a certificate as unauthorized, keys or no keys.
    #[tokio::test]
    async fn required_client_certificates_refuse_requests_without_an_identity() {
        let auth_config = AuthConfig::new(Some("shared-secret".to_string()))
            .with_client_certificates(TlsClientAuth::Required);
        assert!(auth_config.is_enabled());
        assert!(AuthConfig::new(None)
            .with_client_certificates(TlsClientAuth::Required)
            .is_enabled());

        let response = app(auth_config.clone())
            .oneshot(with_certificate(Some(ClientCertificate::Unidentified)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        let body: serde_json::Value = serde_json::from_str(&body_text(response).await).unwrap();
        assert_eq!(body["error"]["code"], "client_certificate_unidentified");

        let mut keyed = with_certificate(None);
        keyed.headers_mut().insert(
            header::AUTHORIZATION,
            "Bearer shared-secret".parse().unwrap(),
        );
        let response = app(auth_config).oneshot(keyed).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "a key does not stand in for a required certificate"
        );
        assert_eq!(challenge(&response), "Bearer realm=\"data-plane\"");
        let body: serde_json::Value = serde_json::from_str(&body_text(response).await).unwrap();
        assert_eq!(body["error"]["code"], "client_certificate_required");
    }

    /// Optional: without an identity the bearer check runs as before; the
    /// mode alone does not make a credential required.
    #[tokio::test]
    async fn optional_client_certificates_fall_through_to_the_bearer_check() {
        let open = AuthConfig::new(None).with_client_certificates(TlsClientAuth::Optional);
        assert!(!open.is_enabled());
        let response = app(open.clone())
            .oneshot(with_certificate(None))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_text(response).await, "missing");
        let response = app(open)
            .oneshot(with_certificate(Some(ClientCertificate::Unidentified)))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "unidentified under optional"
        );

        let keyed = AuthConfig::new(Some("shared-secret".to_string()))
            .with_client_certificates(TlsClientAuth::Optional);
        let response = app(keyed.clone())
            .oneshot(with_certificate(None))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "no certificate, no key"
        );
        let mut with_key = with_certificate(None);
        with_key.headers_mut().insert(
            header::AUTHORIZATION,
            "Bearer shared-secret".parse().unwrap(),
        );
        let response = app(keyed).oneshot(with_key).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "the key still works");
    }

    /// A configuration that does not take client certificates (the admin
    /// routes' fallback) ignores a connection's certificate identity.
    #[tokio::test]
    async fn a_configuration_without_client_certificates_ignores_the_identity() {
        let response = app(AuthConfig::new(Some("shared-secret".to_string())))
            .oneshot(with_certificate(Some(identified(
                "spiffe://cluster.local/ns/default/sa/x",
            ))))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
