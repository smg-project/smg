//! Control-plane role matrix on the production route table: `admin` reaches
//! every route, `user` the read-only ones, and no or an unknown key none;
//! the gated routes, mounted only behind control-plane authentication, are
//! the server's own list and `admin` alone reaches them.

use axum::{
    body::Body,
    extract::Request,
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE},
        StatusCode,
    },
    Router,
};
use smg::server::GATED_ROUTES;
use smg_auth::{ApiKeyEntry, ControlPlaneAuthConfig, ControlPlaneAuthState, Role};
use tower::ServiceExt;

use crate::common::{
    test_app::create_test_app_with_context_and_auth, AppTestContext, TestRouterConfig,
    TestWorkerConfig,
};

pub(super) const ADMIN_KEY: &str = "matrix-admin-key";
const USER_KEY: &str = "matrix-user-key";

/// Read-only control-plane routes: the `user` role reaches them.
const READ_ROUTES: &[(&str, &str)] = &[
    ("GET", "/inflight"),
    ("GET", "/workers"),
    ("GET", "/get_loads"),
    ("GET", "/v1/tokenizers"),
    ("GET", "/wasm"),
    ("GET", "/policy/explain/1"),
];

/// Mutating control-plane routes: `admin` only.
const WRITE_ROUTES: &[(&str, &str)] = &[
    ("POST", "/flush_cache"),
    ("POST", "/workers"),
    ("DELETE", "/workers/no-such-worker"),
    ("POST", "/v1/tokenizers"),
    ("POST", "/stop_profile"),
    ("POST", "/heap_profile"),
];

pub(super) fn control_plane_auth() -> ControlPlaneAuthState {
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

#[expect(
    clippy::unwrap_used,
    reason = "test helper: a malformed request is a test bug"
)]
pub(super) async fn status(
    app: &Router,
    method: &str,
    path: &str,
    key: Option<&str>,
) -> StatusCode {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header(CONTENT_TYPE, "application/json");
    if let Some(key) = key {
        request = request.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    let body = if method == "GET" {
        Body::empty()
    } else {
        Body::from("{}")
    };
    app.clone()
        .oneshot(request.body(body).unwrap())
        .await
        .unwrap()
        .status()
}

/// Past authentication and authorization: whatever the handler answers.
fn reached_the_handler(status: StatusCode) -> bool {
    status != StatusCode::UNAUTHORIZED && status != StatusCode::FORBIDDEN
}

#[tokio::test]
async fn control_plane_role_matrix() {
    let config = TestRouterConfig::round_robin(4350);
    let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20350)]).await;
    let app = create_test_app_with_context_and_auth(
        ctx.router.clone(),
        ctx.app_context.clone(),
        Some(control_plane_auth()),
    );

    for (method, path) in READ_ROUTES.iter().chain(WRITE_ROUTES) {
        assert_eq!(
            status(&app, method, path, None).await,
            StatusCode::UNAUTHORIZED,
            "{method} {path} without a key"
        );
        assert_eq!(
            status(&app, method, path, Some("not-a-configured-key")).await,
            StatusCode::UNAUTHORIZED,
            "{method} {path} with an unknown key"
        );
        let admin = status(&app, method, path, Some(ADMIN_KEY)).await;
        assert!(
            reached_the_handler(admin),
            "{method} {path} with the admin key: {admin}"
        );
    }

    for (method, path) in READ_ROUTES {
        let user = status(&app, method, path, Some(USER_KEY)).await;
        assert!(
            reached_the_handler(user),
            "{method} {path} with the user key: {user}"
        );
    }
    for (method, path) in WRITE_ROUTES {
        assert_eq!(
            status(&app, method, path, Some(USER_KEY)).await,
            StatusCode::FORBIDDEN,
            "{method} {path} with the user key"
        );
    }

    // The gated routes: the server's own list, mounted here because
    // control-plane authentication is configured; `admin` alone reads them.
    assert!(!GATED_ROUTES.is_empty(), "the gated group has a member");
    for (method, path) in GATED_ROUTES {
        assert_eq!(
            status(&app, method, path, None).await,
            StatusCode::UNAUTHORIZED,
            "{method} {path} without a key"
        );
        assert_eq!(
            status(&app, method, path, Some("not-a-configured-key")).await,
            StatusCode::UNAUTHORIZED,
            "{method} {path} with an unknown key"
        );
        assert_eq!(
            status(&app, method, path, Some(USER_KEY)).await,
            StatusCode::FORBIDDEN,
            "{method} {path} with the user key"
        );
        // The gate passed, the route answers for itself (a reload with no
        // file configured is a 409 here; its own test has the 200).
        let admin = status(&app, method, path, Some(ADMIN_KEY)).await;
        assert!(
            !matches!(
                admin,
                StatusCode::UNAUTHORIZED
                    | StatusCode::FORBIDDEN
                    | StatusCode::NOT_FOUND
                    | StatusCode::METHOD_NOT_ALLOWED
            ),
            "{method} {path} with the admin key passes the gate: {admin}"
        );
    }

    ctx.shutdown().await;
}
