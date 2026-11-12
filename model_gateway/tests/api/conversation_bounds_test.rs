//! The Conversations routes under the public API's bounds: a conversation's
//! `metadata` (16 pairs, 64-character keys, 512-character string values,
//! strings only), a message item's `role`, `status` and content parts, and
//! the request body's shape. Storage-backed: no worker is needed.

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
        .port(3014)
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

/// One call of the app: the status and the JSON body (a null `body` sends
/// none, for a GET).
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
async fn call(app: &axum::Router, method: Method, uri: &str, body: Value) -> (StatusCode, Value) {
    let request = if body.is_null() {
        Request::builder()
            .method(method)
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    } else {
        Request::builder()
            .method(method)
            .uri(uri)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap()
    };
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

/// The `code` and `param` of a refusal, once the status, the envelope's type
/// and a non-empty message are checked.
fn refusal(status: StatusCode, body: &Value) -> (String, Value) {
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| !message.is_empty()),
        "{body}"
    );
    (
        body["error"]["code"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        body["error"]["param"].clone(),
    )
}

/// `POST /v1/conversations` refuses metadata beyond the public bounds and a
/// body that is not an object, in the error envelope with the public code and
/// the field named, and keeps metadata at the bounds; `POST
/// /v1/conversations/{id}` refuses a patch that crosses them and changes
/// nothing.
#[tokio::test]
async fn conversation_metadata_is_refused_beyond_the_public_bounds() {
    let (ctx, app) = app_over_no_workers().await;
    let create = |body: Value| call(&app, Method::POST, "/v1/conversations", body);

    let (status, body) = create(json!({"metadata": {"k": "v".repeat(600)}})).await;
    assert_eq!(
        refusal(status, &body),
        ("string_above_max_length".to_owned(), json!("metadata"))
    );
    let mut long_key = serde_json::Map::new();
    long_key.insert("k".repeat(70), json!("v"));
    let (status, body) = create(json!({"metadata": Value::Object(long_key)})).await;
    assert_eq!(
        refusal(status, &body),
        ("string_above_max_length".to_owned(), json!("metadata"))
    );
    let (status, body) = create(json!({"metadata": {"k": {"nested": 1}, "n": 5}})).await;
    assert_eq!(
        refusal(status, &body),
        ("invalid_type".to_owned(), json!("metadata"))
    );
    let (status, body) = create(json!([])).await;
    assert_eq!(
        refusal(status, &body),
        ("invalid_type".to_owned(), Value::Null)
    );

    let at_the_bounds: serde_json::Map<String, Value> = (0..16)
        .map(|i| (format!("{i:0>64}"), json!("v".repeat(512))))
        .collect();
    let (status, created) = create(json!({"metadata": Value::Object(at_the_bounds.clone())})).await;
    assert_eq!(status, StatusCode::OK, "{created}");
    assert_eq!(
        created["metadata"],
        Value::Object(at_the_bounds.clone()),
        "{created}"
    );
    let id = created["id"].as_str().unwrap_or_default().to_owned();
    let uri = format!("/v1/conversations/{id}");

    let key0 = format!("{:0>64}", 0);
    let one = |value: Value| {
        let mut map = serde_json::Map::new();
        map.insert(key0.clone(), value);
        json!({"metadata": Value::Object(map)})
    };
    for (patch, code) in [
        (one(json!("v".repeat(513))), "string_above_max_length"),
        (
            json!({"metadata": {"seventeenth": "v"}}),
            "object_above_max_properties",
        ),
        (one(json!(true)), "invalid_type"),
        (json!([]), "invalid_type"),
    ] {
        let (status, body) = call(&app, Method::POST, &uri, patch.clone()).await;
        assert_eq!(refusal(status, &body).0, code, "{patch}");
    }
    let (status, read) = call(&app, Method::GET, &uri, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{read}");
    assert_eq!(
        read["metadata"],
        Value::Object(at_the_bounds),
        "a refused update changes nothing: {read}"
    );

    ctx.shutdown().await;
}

/// `POST /v1/conversations/{id}/items` refuses a message whose `role`,
/// `status` or content part type is outside the public API's set, in the
/// error envelope with the field named, and stores nothing of the batch; the
/// same items on `POST /v1/conversations` create no conversation; items
/// inside the sets are stored and listed as sent.
#[tokio::test]
async fn message_items_outside_the_public_sets_are_refused_and_not_stored() {
    let (ctx, app) = app_over_no_workers().await;
    let (status, created) = call(&app, Method::POST, "/v1/conversations", json!({})).await;
    assert_eq!(status, StatusCode::OK, "{created}");
    let id = created["id"].as_str().unwrap_or_default().to_owned();
    let items_uri = format!("/v1/conversations/{id}/items");

    for (items, code, param) in [
        (
            json!([{"type": "message", "role": "bogus", "content": "x"}]),
            "invalid_value",
            "items[].role",
        ),
        (
            json!([{"type": "message", "role": "user", "content": [{"type": "bogus_part", "x": 1}]}]),
            "invalid_value",
            "items[].content[0].type",
        ),
        (
            json!([{"type": "message", "role": "user", "content": "x", "status": "flying"}]),
            "invalid_value",
            "items[].status",
        ),
    ] {
        let (status, body) = call(&app, Method::POST, &items_uri, json!({"items": items})).await;
        assert_eq!(
            refusal(status, &body),
            (code.to_owned(), json!(param)),
            "{items}"
        );
        let (status, body) = call(
            &app,
            Method::POST,
            "/v1/conversations",
            json!({"items": items}),
        )
        .await;
        assert_eq!(refusal(status, &body).0, code, "on create: {items}");
    }
    let (status, listed) = call(&app, Method::GET, &items_uri, Value::Null).await;
    assert_eq!(status, StatusCode::OK, "{listed}");
    assert_eq!(
        listed["data"].as_array().map(Vec::len),
        Some(0),
        "nothing of a refused batch is stored: {listed}"
    );

    let (status, added) = call(
        &app,
        Method::POST,
        &items_uri,
        json!({"items": [
            {"type": "message", "role": "developer", "content": "rules", "status": "incomplete"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
        ]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{added}");
    let (status, listed) = call(
        &app,
        Method::GET,
        &format!("{items_uri}?order=asc"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{listed}");
    assert_eq!(listed["data"][0]["role"], "developer", "{listed}");
    assert_eq!(listed["data"][0]["status"], "incomplete", "{listed}");
    assert_eq!(
        listed["data"][1]["content"][0]["type"], "input_text",
        "{listed}"
    );

    ctx.shutdown().await;
}
