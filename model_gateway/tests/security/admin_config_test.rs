//! `GET /config` on the production route table: mounted only with
//! control-plane authentication, read by the `admin` role alone, and its
//! answer names the configured keys by fingerprint, never by value.

use axum::{
    body::Body,
    extract::Request,
    http::{header::AUTHORIZATION, StatusCode},
    response::Response,
    Router,
};
use serde_json::Value;
use smg::config::{RouterConfig, TenantApiKeyEntry};
use smg_auth::{ApiKeyEntry, ControlPlaneAuthConfig, ControlPlaneAuthState, Role};
use tower::ServiceExt;

use crate::common::{
    test_app::create_test_app_with_context_and_auth, AppTestContext, TestRouterConfig,
    TestWorkerConfig,
};

const ADMIN_KEY: &str = "config-admin-key";
const USER_KEY: &str = "config-user-key";
const SHARED_KEY: &str = "shared-serving-secret";
const TENANT_KEY: &str = "team-red-serving-secret";

fn control_plane_auth() -> ControlPlaneAuthState {
    ControlPlaneAuthState::new(
        ControlPlaneAuthConfig {
            jwt: None,
            api_keys: vec![
                ApiKeyEntry::new("admin", "Admin", ADMIN_KEY, Role::Admin),
                ApiKeyEntry::new("reader", "Reader", USER_KEY, Role::User),
            ],
            audit_enabled: false,
        },
        None,
    )
}

/// A gateway with a shared serving key and one tenant key: the secrets the
/// view must not show.
fn keyed_config(port: u16) -> RouterConfig {
    let mut config = TestRouterConfig::round_robin(port);
    config.api_key = Some(SHARED_KEY.to_string());
    config.tenant_api_keys = vec![TenantApiKeyEntry {
        tenant_id: "team-red".to_string(),
        key: TENANT_KEY.to_string(),
    }];
    config
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper: a malformed request is a test bug"
)]
async fn get_config(app: &Router, key: Option<&str>) -> Response {
    let mut request = Request::builder().uri("/config");
    if let Some(key) = key {
        request = request.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    app.clone()
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap()
}

#[expect(clippy::unwrap_used, reason = "test helper: the answer is text")]
async fn text(response: Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[tokio::test]
async fn the_admin_reads_the_redacted_view_and_the_user_or_an_unknown_caller_does_not() {
    let ctx =
        AppTestContext::new_with_config(keyed_config(4352), vec![TestWorkerConfig::healthy(20352)])
            .await;
    let app = create_test_app_with_context_and_auth(
        ctx.router.clone(),
        ctx.app_context.clone(),
        Some(control_plane_auth()),
    );

    assert_eq!(
        get_config(&app, None).await.status(),
        StatusCode::UNAUTHORIZED,
        "without a key"
    );
    assert_eq!(
        get_config(&app, Some(USER_KEY)).await.status(),
        StatusCode::FORBIDDEN,
        "the user role may not read the view"
    );

    let answer = get_config(&app, Some(ADMIN_KEY)).await;
    assert_eq!(answer.status(), StatusCode::OK, "the admin role reads it");
    let body = text(answer).await;
    for secret in [SHARED_KEY, TENANT_KEY] {
        assert!(!body.contains(secret), "{secret} reached the view: {body}");
    }
    let view: Value = serde_json::from_str(&body).unwrap();
    let fingerprint = view["config"]["api_key"]
        .as_str()
        .unwrap_or_else(|| panic!("the shared key is named by its fingerprint: {view}"));
    assert_eq!(fingerprint.len(), 16, "{fingerprint}");
    assert!(
        fingerprint.chars().all(|c| c.is_ascii_hexdigit()),
        "{fingerprint}"
    );
    let tenant = &view["config"]["tenant_api_keys"][0];
    assert_eq!(tenant["tenant_id"], "team-red", "{view}");
    assert!(tenant["fingerprint"].is_string(), "{view}");
    assert!(tenant.get("key").is_none(), "{view}");
    assert!(view["build"]["version"].is_string(), "{view}");
    assert!(view["secret_sources"].is_object(), "{view}");

    ctx.shutdown().await;
}

#[tokio::test]
async fn without_control_plane_authentication_the_view_is_not_mounted() {
    let ctx =
        AppTestContext::new_with_config(keyed_config(4353), vec![TestWorkerConfig::healthy(20353)])
            .await;
    let app = ctx.create_app();

    assert_eq!(
        get_config(&app, Some(SHARED_KEY)).await.status(),
        StatusCode::NOT_FOUND,
        "the shared key reaches the control plane, and the path is unknown"
    );

    ctx.shutdown().await;
}
