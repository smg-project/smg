//! `POST /v1/conversations` with the public `items` field: the created
//! conversation lists them. Storage-backed: no worker is needed.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, Method, StatusCode},
};
use serde_json::{json, Value};
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common::AppTestContext;

async fn app_over_no_workers() -> (AppTestContext, axum::Router) {
    let mut config = RouterConfig::builder()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(3013)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_size(0)
        .queue_timeout_secs(60)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    let ctx = AppTestContext::new_with_config(config, vec![]).await;
    let app = ctx.create_app();
    (ctx, app)
}

/// One call of the app: the status and the JSON body.
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
async fn call(app: &axum::Router, method: Method, uri: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// A conversation created with `items` and `metadata` answers the conversation
/// object and lists the items, in the order given, with the shape the add-items
/// route stores; a create whose items the add-items route would refuse is the
/// same 400 and creates no conversation.
#[tokio::test]
async fn a_conversation_created_with_items_lists_them() {
    let (ctx, app) = app_over_no_workers().await;

    let (status, created) = call(
        &app,
        Method::POST,
        "/v1/conversations",
        json!({
            "items": [
                {"type": "message", "role": "user", "content": "first"},
                {"type": "message", "role": "assistant", "content": "second"}
            ],
            "metadata": {"a": "b"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{created}");
    assert_eq!(created["object"], "conversation", "{created}");
    assert_eq!(created["metadata"]["a"], "b", "{created}");
    let conversation_id = created["id"].as_str().expect("a conversation id");

    let (status, listed) = call(
        &app,
        Method::GET,
        &format!("/v1/conversations/{conversation_id}/items?order=asc"),
        json!({}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{listed}");
    assert_eq!(listed["data"].as_array().map(Vec::len), Some(2), "{listed}");
    assert_eq!(listed["data"][0]["role"], "user", "{listed}");
    assert_eq!(listed["data"][0]["content"][0]["text"], "first", "{listed}");
    assert_eq!(listed["data"][1]["role"], "assistant", "{listed}");
    assert_eq!(
        listed["data"][1]["content"][0]["type"], "output_text",
        "{listed}"
    );
    assert_eq!(
        listed["data"][1]["content"][0]["text"], "second",
        "{listed}"
    );

    for body in [
        json!({"items": [{"type": "bogus"}]}),
        json!({"items": "first"}),
        json!({"items": [{"type": "message", "content": "no role"}]}),
    ] {
        let (status, refused) = call(&app, Method::POST, "/v1/conversations", body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {refused}");
        assert_eq!(
            refused["error"]["type"], "invalid_request_error",
            "{body}: {refused}"
        );
        assert!(
            refused.get("id").is_none(),
            "{body}: no conversation is created: {refused}"
        );
    }

    ctx.shutdown().await;
}
