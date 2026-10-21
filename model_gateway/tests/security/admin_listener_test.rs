//! The control plane on a listener of its own: with an admin listener, the
//! admin and worker-registry routes answer there and not on the data
//! listener, which answers 404 for them; the serving routes stay off the
//! admin listener; the probe routes answer on both. Without one, the single
//! router carries every route, as before.

use axum::{body::Body, extract::Request, http::StatusCode};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[expect(
    clippy::unwrap_used,
    reason = "test helper: a malformed request or a failed call is a test failure"
)]
async fn status(app: &axum::Router, method: &str, uri: &str) -> StatusCode {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    app.clone().oneshot(request).await.unwrap().status()
}

#[tokio::test]
async fn the_admin_listener_takes_the_control_plane_off_the_data_listener() {
    let ctx = AppTestContext::new_with_config(
        TestRouterConfig::round_robin(3940),
        vec![TestWorkerConfig::healthy(19940)],
    )
    .await;
    let apps = ctx.create_apps(true);
    let admin = apps
        .admin
        .expect("an admin router when the admin listener is on");
    let data = apps.data;

    // The control plane answers on the admin listener alone: the data
    // listener answers 404 for its routes, the edge fallback's answer for a
    // route it does not have.
    for uri in ["/workers", "/get_loads", "/v1/tokenizers"] {
        assert_eq!(
            status(&admin, "GET", uri).await,
            StatusCode::OK,
            "GET {uri} on the admin listener"
        );
        assert_eq!(
            status(&data, "GET", uri).await,
            StatusCode::NOT_FOUND,
            "GET {uri} on the data listener"
        );
    }
    assert_eq!(
        status(&data, "POST", "/flush_cache").await,
        StatusCode::NOT_FOUND
    );
    // A route the admin listener has, asked badly: refused, not unknown.
    assert_ne!(
        status(&admin, "POST", "/workers").await,
        StatusCode::NOT_FOUND
    );

    // The serving routes stay off the admin listener.
    for uri in ["/v1/models", "/get_server_info"] {
        assert_eq!(status(&data, "GET", uri).await, StatusCode::OK, "{uri}");
        assert_eq!(
            status(&admin, "GET", uri).await,
            StatusCode::NOT_FOUND,
            "GET {uri} on the admin listener"
        );
    }

    // The probe routes answer on both.
    for uri in ["/health", "/liveness"] {
        assert_eq!(status(&data, "GET", uri).await, StatusCode::OK, "{uri}");
        assert_eq!(status(&admin, "GET", uri).await, StatusCode::OK, "{uri}");
    }
    assert_ne!(
        status(&admin, "GET", "/readiness").await,
        StatusCode::NOT_FOUND
    );

    ctx.shutdown().await;
}

#[tokio::test]
async fn without_an_admin_listener_the_data_listener_carries_the_control_plane() {
    let ctx = AppTestContext::new_with_config(
        TestRouterConfig::round_robin(3941),
        vec![TestWorkerConfig::healthy(19941)],
    )
    .await;
    let apps = ctx.create_apps(false);
    assert!(apps.admin.is_none(), "no admin router without the listener");
    for uri in ["/workers", "/v1/tokenizers", "/v1/models", "/health"] {
        assert_eq!(
            status(&apps.data, "GET", uri).await,
            StatusCode::OK,
            "{uri}"
        );
    }
    ctx.shutdown().await;
}
