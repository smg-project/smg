#![allow(clippy::unwrap_used)]
//! The Query and Path rejections answer in the envelopes: the data plane's
//! (`error.code`, `error.param`) on the public routes and the control plane's
//! (`{"error", "code"}`) on the management routes, where the framework
//! answered text/plain.

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, Request, StatusCode},
};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};

fn worker(port: u16) -> MockWorkerConfig {
    MockWorkerConfig {
        port,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }
}

/// The status, the content type and the body of a `method` request to
/// `uri`; a body
/// that is not JSON is kept under `not_json`, so an assertion names it.
async fn send(
    app: &axum::Router,
    method: &str,
    uri: &str,
) -> (StatusCode, String, serde_json::Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| serde_json::json!({"not_json": String::from_utf8_lossy(&bytes)}));
    (status, content_type, body)
}

/// A path segment that is not UTF-8 on the models and Conversations routes,
/// and a query string the Conversations list route does not take, are a
/// 400 in the public envelope: `invalid_value`, `param` the segment's name.
#[tokio::test]
async fn the_public_routes_refuse_a_bad_path_segment_or_query_string_in_the_envelope() {
    let ctx = AppTestContext::new(vec![worker(18132)]).await;
    let app = ctx.create_app();
    for (uri, param) in [
        ("/v1/models/%ff", "id"),
        ("/v1/conversations/%ff", "conversation_id"),
    ] {
        let (status, content_type, body) = send(&app, "GET", uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {body}");
        assert!(
            content_type.starts_with("application/json"),
            "{uri}: {content_type} {body}"
        );
        assert_eq!(
            body["error"]["type"], "invalid_request_error",
            "{uri}: {body}"
        );
        assert_eq!(body["error"]["code"], "invalid_value", "{uri}: {body}");
        assert_eq!(body["error"]["param"], param, "{uri}: {body}");
    }
    let uri = "/v1/conversations/conv_missing/items?limit=-1";
    let (status, content_type, body) = send(&app, "GET", uri).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {body}");
    assert!(
        content_type.starts_with("application/json"),
        "{uri}: {content_type} {body}"
    );
    assert_eq!(
        body["error"]["type"], "invalid_request_error",
        "{uri}: {body}"
    );
    assert_eq!(body["error"]["code"], "invalid_value", "{uri}: {body}");
    assert_eq!(body["error"]["param"], "limit", "{uri}: {body}");
    ctx.shutdown().await;
}

/// A query string the worker list does not take and a worker path that is
/// not UTF-8 are a 400 in the control plane's envelope.
#[tokio::test]
async fn the_management_routes_refuse_a_bad_path_segment_or_query_string_in_the_envelope() {
    let ctx = AppTestContext::new(vec![worker(18133)]).await;
    let app = ctx.create_app();
    for uri in [
        "/workers?origin=bogus",
        "/workers?format=bogus",
        "/workers/%ff",
    ] {
        let (status, content_type, body) = send(&app, "GET", uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {body}");
        assert!(
            content_type.starts_with("application/json"),
            "{uri}: {content_type} {body}"
        );
        assert_eq!(body["code"], "BAD_REQUEST", "{uri}: {body}");
        assert!(body["error"].is_string(), "{uri}: {body}");
    }
    // the WASM module route, the one control-plane path site beside the worker routes
    let (status, content_type, body) = send(&app, "DELETE", "/wasm/%ff").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        content_type.starts_with("application/json"),
        "{content_type} {body}"
    );
    assert_eq!(body["code"], "BAD_REQUEST", "{body}");
    assert!(body["error"].is_string(), "{body}");
    ctx.shutdown().await;
}
