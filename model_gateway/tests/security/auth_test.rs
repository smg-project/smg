//! Authentication and authorization integration tests
//!
//! Tests for API key enforcement and access control.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::TenantApiKeyEntry;
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

const AUTH_HEADER: &str = "Authorization";

#[cfg(test)]
mod auth_tests {
    use super::*;

    /// Test request without API key when auth is not required
    #[tokio::test]
    async fn test_no_auth_required() {
        let config = TestRouterConfig::round_robin(4300);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20300)]).await;

        let app = ctx.create_app();

        // Request without auth header should succeed when no auth required
        let payload = json!({
            "text": "Test without auth",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request without auth should succeed when no auth required"
        );

        ctx.shutdown().await;
    }

    /// Test request with valid API key format
    #[tokio::test]
    async fn test_with_api_key_header() {
        let config = TestRouterConfig::round_robin(4301);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20301)]).await;

        let app = ctx.create_app();

        // Request with Bearer token header
        let payload = json!({
            "text": "Test with auth header",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header(AUTH_HEADER, "Bearer test-api-key-12345")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Without auth enforcement, request should succeed
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request with auth header should succeed"
        );

        ctx.shutdown().await;
    }

    /// Test health endpoint doesn't require authentication
    #[tokio::test]
    async fn test_health_endpoint_no_auth() {
        let config = TestRouterConfig::round_robin(4302);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20302)]).await;

        let app = ctx.create_app();

        // Health endpoint should be accessible without auth
        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Health endpoint should not require auth"
        );

        ctx.shutdown().await;
    }

    /// Test OpenAI-compatible API key header (X-API-Key)
    #[tokio::test]
    async fn test_openai_api_key_header() {
        let config = TestRouterConfig::round_robin(4303);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20303)]).await;

        let app = ctx.create_app();

        // Request with X-API-Key header (OpenAI style)
        let payload = json!({
            "text": "Test with X-API-Key",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("X-API-Key", "sk-test-key-12345")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request with X-API-Key should succeed"
        );

        ctx.shutdown().await;
    }

    /// Test multiple concurrent authenticated requests
    #[tokio::test]
    #[expect(clippy::disallowed_methods)]
    async fn test_concurrent_authenticated_requests() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let config = TestRouterConfig::round_robin(4304);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20304)]).await;

        let app = ctx.create_app();
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for i in 0..20 {
            let app_clone = app.clone();
            let success_clone = Arc::clone(&success_count);

            let handle = tokio::spawn(async move {
                let payload = json!({
                    "text": format!("Concurrent auth test {i}"),
                    "stream": false
                });

                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .header(AUTH_HEADER, format!("Bearer test-key-{i}"))
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();

                let resp = app_clone.oneshot(req).await.unwrap();
                if resp.status() == StatusCode::OK {
                    success_clone.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(
            success_count.load(Ordering::SeqCst),
            20,
            "All concurrent authenticated requests should succeed"
        );

        ctx.shutdown().await;
    }

    /// A tenant key must authenticate `/generate` but not admin routes.
    /// Configures a shared key too, so there's a real admin credential to
    /// test against. Tenant-only deployments deny admin routes outright;
    /// see `test_tenant_only_deployment_denies_admin_routes`.
    #[tokio::test]
    async fn test_tenant_key_cannot_access_admin_routes() {
        let mut config = TestRouterConfig::round_robin(4306);
        config.api_key = Some("shared-secret".to_string());
        config.tenant_api_keys = vec![TenantApiKeyEntry {
            tenant_id: "team-red".to_string(),
            key: "team-red-secret".to_string(),
        }];

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20306)]).await;

        let app = ctx.create_app();

        let generate_req = |bearer: &str| {
            let payload = json!({"text": "hi", "stream": false});
            Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .header(AUTH_HEADER, format!("Bearer {bearer}"))
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap()
        };
        let workers_req = |bearer: &str| {
            Request::builder()
                .method("GET")
                .uri("/workers")
                .header(AUTH_HEADER, format!("Bearer {bearer}"))
                .body(Body::empty())
                .unwrap()
        };

        // Tenant key: serving route ok, admin route rejected.
        assert_eq!(
            app.clone()
                .oneshot(generate_req("team-red-secret"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK,
            "Tenant key should authenticate serving-path requests"
        );
        assert_eq!(
            app.clone()
                .oneshot(workers_req("team-red-secret"))
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED,
            "Tenant key must not authenticate admin/worker-management routes"
        );

        // Shared key: both serving and admin routes still accept it.
        assert_eq!(
            app.clone()
                .oneshot(generate_req("shared-secret"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK,
            "Shared key should still authenticate serving-path requests"
        );
        assert_eq!(
            app.oneshot(workers_req("shared-secret"))
                .await
                .unwrap()
                .status(),
            StatusCode::OK,
            "Shared key should still authenticate admin/worker-management routes"
        );

        ctx.shutdown().await;
    }

    /// Tenant-only deployment (no shared `--api-key`, no control-plane auth):
    /// admin routes must be denied outright, not fall back to open.
    #[tokio::test]
    async fn test_tenant_only_deployment_denies_admin_routes() {
        let mut config = TestRouterConfig::round_robin(4307);
        config.tenant_api_keys = vec![TenantApiKeyEntry {
            tenant_id: "team-red".to_string(),
            key: "team-red-secret".to_string(),
        }];

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20307)]).await;

        let app = ctx.create_app();

        let payload = json!({"text": "hi", "stream": false});
        let generate_req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header(AUTH_HEADER, "Bearer team-red-secret")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        assert_eq!(
            app.clone().oneshot(generate_req).await.unwrap().status(),
            StatusCode::OK,
            "Tenant key should still authenticate serving-path requests"
        );

        let workers_with_tenant_key = Request::builder()
            .method("GET")
            .uri("/workers")
            .header(AUTH_HEADER, "Bearer team-red-secret")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.clone()
                .oneshot(workers_with_tenant_key)
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED,
            "Tenant-only deployment must deny admin routes even with a valid tenant key"
        );

        let workers_no_auth = Request::builder()
            .method("GET")
            .uri("/workers")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.oneshot(workers_no_auth).await.unwrap().status(),
            StatusCode::UNAUTHORIZED,
            "Tenant-only deployment must deny admin routes with no credential"
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod mtls_tests {
    use super::*;

    /// Test that TLS configuration options exist
    /// Note: Actual mTLS testing would require certificate setup
    #[tokio::test]
    async fn test_tls_config_available() {
        // This test verifies the config builder accepts TLS-related options
        // Actual mTLS testing requires certificate infrastructure
        let config = TestRouterConfig::round_robin(4305);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20305)]).await;

        let app = ctx.create_app();

        // Basic request should work (no TLS in test mode)
        let payload = json!({
            "text": "TLS config test",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        ctx.shutdown().await;
    }
}

/// Failed JWT setup must not select build_app's shared-key/open fallback.
#[tokio::test]
async fn failed_jwt_initialization_keeps_admin_routes_protected() {
    use smg_auth::{ApiKeyEntry, ControlPlaneAuthConfig, ControlPlaneAuthState, JwtConfig, Role};

    use crate::common::test_app::create_test_app_with_context_and_auth;

    for (index, shared_key) in [None, Some("gateway-shared-key")].into_iter().enumerate() {
        let mut config = TestRouterConfig::round_robin(4310 + index as u16);
        config.api_key = shared_key.map(str::to_owned);
        let ctx = AppTestContext::new_with_config(config, vec![]).await;
        for with_keys in [false, true] {
            let auth_config = ControlPlaneAuthConfig {
                jwt: Some(
                    JwtConfig::new("https://issuer.example.com", "audience")
                        .with_jwks_uri("https://127.0.0.1/jwks"),
                ),
                api_keys: if with_keys {
                    vec![
                        ApiKeyEntry::new("admin", "Admin", "cp-admin-key", Role::Admin),
                        ApiKeyEntry::new("user", "User", "cp-user-key", Role::User),
                    ]
                } else {
                    vec![]
                },
                audit_enabled: false,
            };
            let auth = ControlPlaneAuthState::try_init(Some(&auth_config)).await;
            let app = create_test_app_with_context_and_auth(
                ctx.router.clone(),
                ctx.app_context.clone(),
                auth,
            );
            for (method, path) in [("GET", "/workers"), ("POST", "/flush_cache")] {
                for token in [
                    None,
                    Some("gateway-shared-key"),
                    Some("unknown-key"),
                    Some("a.b.c"),
                ] {
                    let mut request = Request::builder().method(method).uri(path);
                    if let Some(token) = token {
                        request = request.header(AUTH_HEADER, format!("Bearer {token}"));
                    }
                    let response = app
                        .clone()
                        .oneshot(request.body(Body::empty()).unwrap())
                        .await
                        .unwrap();
                    assert_eq!(
                        response.status(),
                        StatusCode::UNAUTHORIZED,
                        "{path}, shared_key={shared_key:?}, with_keys={with_keys}, token={token:?}"
                    );
                }
            }
            for (token, status) in [
                (
                    "cp-admin-key",
                    if with_keys {
                        StatusCode::OK
                    } else {
                        StatusCode::UNAUTHORIZED
                    },
                ),
                (
                    "cp-user-key",
                    if with_keys {
                        StatusCode::FORBIDDEN
                    } else {
                        StatusCode::UNAUTHORIZED
                    },
                ),
            ] {
                let response = app
                    .clone()
                    .oneshot(
                        Request::builder()
                            .uri("/workers")
                            .header(AUTH_HEADER, format!("Bearer {token}"))
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), status);
            }
            let health = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/health")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(health.status(), StatusCode::OK);
        }
        ctx.shutdown().await;
    }
}
