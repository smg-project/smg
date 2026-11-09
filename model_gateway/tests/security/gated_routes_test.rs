//! The gated control-plane routes on the production route tables: without
//! control-plane authentication they are not mounted, so the shared serving
//! key gets 404 for them on the data listener and on the admin listener
//! alike; with it they sit with the control plane on the admin listener and
//! nowhere on the data listener.

use axum::{
    body::Body,
    extract::Request,
    http::{header::AUTHORIZATION, StatusCode},
};
use smg::{config::RouterConfig, server::GATED_ROUTES};
use smg_auth::{ApiKeyEntry, ControlPlaneAuthConfig, ControlPlaneAuthState, Role};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

const ADMIN_KEY: &str = "gated-admin-key";
const SHARED_KEY: &str = "shared-serving-secret";

fn control_plane_auth() -> ControlPlaneAuthState {
    ControlPlaneAuthState::new(
        ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![ApiKeyEntry::new("admin", "Admin", ADMIN_KEY, Role::Admin)],
            audit_enabled: false,
        },
        None,
    )
}

/// A gateway with the shared serving key: the credential that opens the
/// admin and worker routes when no control-plane authentication is set.
fn shared_key_config(port: u16) -> RouterConfig {
    let mut config = TestRouterConfig::round_robin(port);
    config.api_key = Some(SHARED_KEY.to_string());
    config
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper: a malformed request is a test bug"
)]
async fn status(app: &axum::Router, method: &str, path: &str, key: &str) -> StatusCode {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header(AUTHORIZATION, format!("Bearer {key}"))
        .body(Body::empty())
        .unwrap();
    app.clone().oneshot(request).await.unwrap().status()
}

#[tokio::test]
async fn without_control_plane_authentication_the_gated_routes_are_not_mounted_on_either_listener()
{
    let ctx = AppTestContext::new_with_config(
        shared_key_config(4354),
        vec![TestWorkerConfig::healthy(20354)],
    )
    .await;
    let apps = ctx.create_apps(true);
    let admin = apps.admin.expect("an admin router with the admin listener");
    let data = apps.data;

    // The shared key opens the control plane's other routes here...
    assert_eq!(
        status(&admin, "GET", "/workers", SHARED_KEY).await,
        StatusCode::OK
    );
    // ...and gets 404 for every gated route on both listeners: the path is
    // unknown, not refused.
    for (method, path) in GATED_ROUTES {
        assert_eq!(
            status(&admin, method, path, SHARED_KEY).await,
            StatusCode::NOT_FOUND,
            "{method} {path} on the admin listener with the shared key"
        );
        assert_eq!(
            status(&data, method, path, SHARED_KEY).await,
            StatusCode::NOT_FOUND,
            "{method} {path} on the data listener with the shared key"
        );
    }

    ctx.shutdown().await;
}

#[tokio::test]
async fn with_control_plane_authentication_the_gated_routes_sit_on_the_admin_listener_alone() {
    let ctx = AppTestContext::new_with_config(
        shared_key_config(4355),
        vec![TestWorkerConfig::healthy(20355)],
    )
    .await;
    let apps = ctx.create_apps_with_auth(Some(control_plane_auth()), true);
    let admin = apps.admin.expect("an admin router with the admin listener");
    let data = apps.data;

    for (method, path) in GATED_ROUTES {
        // The gate passed: the route answers for itself (the reload is a
        // 409 here, no key having come from a file).
        let answer = status(&admin, method, path, ADMIN_KEY).await;
        assert!(
            !matches!(
                answer,
                StatusCode::UNAUTHORIZED
                    | StatusCode::FORBIDDEN
                    | StatusCode::NOT_FOUND
                    | StatusCode::METHOD_NOT_ALLOWED
            ),
            "{method} {path} on the admin listener with the admin key: {answer}"
        );
        // The shared serving key is no control-plane credential.
        assert_eq!(
            status(&admin, method, path, SHARED_KEY).await,
            StatusCode::UNAUTHORIZED,
            "{method} {path} on the admin listener with the shared key"
        );
        assert_eq!(
            status(&data, method, path, ADMIN_KEY).await,
            StatusCode::NOT_FOUND,
            "{method} {path} on the data listener"
        );
    }
    assert_eq!(
        status(&admin, "GET", "/config", ADMIN_KEY).await,
        StatusCode::OK
    );
    assert_eq!(
        status(&admin, "POST", "/secrets/reload", ADMIN_KEY).await,
        StatusCode::CONFLICT,
        "no key came from a file in this start"
    );

    ctx.shutdown().await;
}
