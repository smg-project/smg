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
use crate::{
    middleware::test_support::{json_of, text_of},
    routers::common::header_utils::extract_auth_header,
};

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

/// Whether the request still carries an `Authorization` header when it
/// reaches the route.
async fn authorization_seen(request: Request<Body>) -> impl IntoResponse {
    if request.headers().contains_key(header::AUTHORIZATION) {
        "present"
    } else {
        "absent"
    }
}

fn header_reporting_app(auth_config: AuthConfig) -> Router {
    Router::new()
        .route("/", get(authorization_seen))
        .route_layer(from_fn_with_state(auth_config, auth_middleware))
}

/// The provider key headers the route still sees, by name and value.
async fn provider_keys_seen(request: Request<Body>) -> impl IntoResponse {
    PROVIDER_KEY_HEADERS
        .iter()
        .filter_map(|name| {
            request
                .headers()
                .get(*name)
                .and_then(|value| value.to_str().ok())
                .map(|value| format!("{name}={value}"))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A gateway key presented under a provider's own header name beside the
/// bearer ends at the middleware too, wherever it sits among a repeated
/// header's values; a caller's own provider key in those headers (bring
/// your own key) stays.
#[tokio::test]
async fn a_gateway_key_under_a_providers_header_name_ends_at_the_middleware_too() {
    let app = || {
        Router::new()
            .route("/", get(provider_keys_seen))
            .route_layer(from_fn_with_state(
                AuthConfig::new(Some("shared-secret".to_string())),
                auth_middleware,
            ))
    };
    let twice = Request::builder()
        .uri("/")
        .header(header::AUTHORIZATION, "Bearer shared-secret")
        .header("x-api-key", "shared-secret")
        .header("x-goog-api-key", "shared-secret")
        .body(Body::empty())
        .unwrap();
    let response = app().oneshot(twice).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(text_of(response).await, "");

    let own = Request::builder()
        .uri("/")
        .header(header::AUTHORIZATION, "Bearer shared-secret")
        .header("x-api-key", "providers-own-key")
        .body(Body::empty())
        .unwrap();
    let response = app().oneshot(own).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(text_of(response).await, "x-api-key=providers-own-key");

    // The caller's own key first, the gateway's key in a later value of
    // the same header: the whole name goes, since one value is the
    // gateway's and a header is removed by name.
    let repeated = Request::builder()
        .uri("/")
        .header(header::AUTHORIZATION, "Bearer shared-secret")
        .header("x-api-key", "providers-own-key")
        .header("x-api-key", "shared-secret")
        .body(Body::empty())
        .unwrap();
    let response = app().oneshot(repeated).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(text_of(response).await, "");
}

async fn authorization_after(auth_config: AuthConfig, bearer: Option<&str>) -> String {
    let mut request = Request::builder().uri("/");
    if let Some(bearer) = bearer {
        request = request.header(header::AUTHORIZATION, format!("Bearer {bearer}"));
    }
    let response = header_reporting_app(auth_config)
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    text_of(response).await
}

/// A matched gateway key ends at the middleware: the route sees the
/// caller's identity and no `Authorization` header, so nothing downstream
/// forwards the gateway's key to an upstream as the caller's credential.
#[tokio::test]
async fn a_matched_gateway_key_does_not_travel_past_the_middleware() {
    let shared = AuthConfig::new(Some("shared-secret".to_string()));
    assert_eq!(
        authorization_after(shared, Some("shared-secret")).await,
        "absent"
    );

    let tenant = AuthConfig::with_tenant_keys(
        None,
        &[TenantApiKeyEntry {
            tenant_id: "team-red".to_string(),
            key: "team-red-secret".to_string(),
        }],
    );
    assert_eq!(
        authorization_after(tenant, Some("team-red-secret")).await,
        "absent"
    );
}

/// The credential a route sends upstream, as the HTTP router and the
/// external router compute it (`header_utils::extract_auth_header`:
/// the caller's header, else the upstream's configured key).
async fn upstream_credential(request: Request<Body>) -> impl IntoResponse {
    let configured = "provider-key".to_string();
    extract_auth_header(Some(request.headers()), Some(&configured))
        .and_then(|value| value.to_str().ok().map(str::to_string))
        .unwrap_or_else(|| "none".to_string())
}

/// The same credential for an upstream configured without a key.
async fn upstream_credential_without_a_key(request: Request<Body>) -> impl IntoResponse {
    extract_auth_header(Some(request.headers()), None)
        .and_then(|value| value.to_str().ok().map(str::to_string))
        .unwrap_or_else(|| "none".to_string())
}

async fn upstream_credential_after(auth_config: AuthConfig, bearer: Option<&str>) -> String {
    let mut request = Request::builder().uri("/");
    if let Some(bearer) = bearer {
        request = request.header(header::AUTHORIZATION, format!("Bearer {bearer}"));
    }
    let response = Router::new()
        .route("/", get(upstream_credential))
        .route("/without-a-key", get(upstream_credential_without_a_key))
        .route_layer(from_fn_with_state(auth_config, auth_middleware))
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    text_of(response).await
}

/// What leaves for the upstream: under gateway auth the configured
/// provider key, never the gateway key the caller presented; without
/// gateway auth the caller's own key, as before, and the configured key
/// when the caller sent none.
#[tokio::test]
async fn the_upstream_gets_its_configured_key_under_gateway_auth_and_the_callers_without() {
    let gateway = AuthConfig::with_tenant_keys(
        Some("shared-secret".to_string()),
        &[TenantApiKeyEntry {
            tenant_id: "team-red".to_string(),
            key: "team-red-secret".to_string(),
        }],
    );
    assert_eq!(
        upstream_credential_after(gateway.clone(), Some("shared-secret")).await,
        "Bearer provider-key"
    );
    assert_eq!(
        upstream_credential_after(gateway, Some("team-red-secret")).await,
        "Bearer provider-key"
    );
    assert_eq!(
        upstream_credential_after(AuthConfig::new(None), Some("callers-own-provider-key")).await,
        "Bearer callers-own-provider-key"
    );
    assert_eq!(
        upstream_credential_after(AuthConfig::new(None), None).await,
        "Bearer provider-key"
    );
    // An upstream configured without a key is reached with none under
    // gateway auth: a worker that served on the forwarded gateway key is
    // registered with its own key from now.
    assert_eq!(
        upstream_credential_without_a_key_after(
            AuthConfig::new(Some("shared-secret".to_string())),
            Some("shared-secret")
        )
        .await,
        "none"
    );
}

async fn upstream_credential_without_a_key_after(
    auth_config: AuthConfig,
    bearer: Option<&str>,
) -> String {
    let mut request = Request::builder().uri("/without-a-key");
    if let Some(bearer) = bearer {
        request = request.header(header::AUTHORIZATION, format!("Bearer {bearer}"));
    }
    let response = Router::new()
        .route("/without-a-key", get(upstream_credential_without_a_key))
        .route_layer(from_fn_with_state(auth_config, auth_middleware))
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    text_of(response).await
}

/// With no key configured the middleware authenticates nothing and leaves
/// the header where it is: a caller's own upstream key still passes
/// through.
#[tokio::test]
async fn without_a_configured_key_the_header_stays_on_the_request() {
    assert_eq!(
        authorization_after(AuthConfig::new(None), Some("a-providers-own-key")).await,
        "present"
    );
    assert_eq!(
        authorization_after(AuthConfig::new(None), None).await,
        "absent"
    );
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
    assert_eq!(text_of(response).await, expected.to_string());
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
    assert_eq!(text_of(response).await, "auth:team-red");
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
        assert_eq!(text_of(response).await, expected_tenant);
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
    let body = json_of(response).await;
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
