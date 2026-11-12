//! The stored-Responses read routes under a query: `GET /v1/responses/{id}`
//! reads `include` and `stream`, `GET /v1/responses/{id}/input_items` pages by
//! `limit`, `order` and `after`. Storage-backed: no worker is needed.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::{json, Value};
use smg::config::RouterConfig;
use smg_data_connector::{ResponseId, StoredResponse};
use tower::ServiceExt;

use crate::common::AppTestContext;

/// The app over an empty fleet, with `resp_q` stored: three input messages in
/// the order the request gave them, under the ids the store minted.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
async fn app_with_a_stored_response() -> (AppTestContext, axum::Router) {
    let mut config = RouterConfig::builder()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(3012)
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

    let mut stored = StoredResponse::new(None);
    stored.id = ResponseId::from("resp_q");
    stored.input = Value::Array(
        (0..3)
            .map(|i| {
                json!({"id": format!("msg_{i}"), "type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": format!("q{i}")}]})
            })
            .collect(),
    );
    stored.raw_response =
        json!({"id": "resp_q", "object": "response", "status": "completed", "output": []});
    ctx.app_context
        .response_storage
        .store_response(stored)
        .await
        .expect("the response is stored");
    (ctx, app)
}

/// `resp_legacy` stored beside it: twenty-five input messages without ids,
/// the shape of a record from before ids were minted at store time.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
async fn store_a_record_without_item_ids(ctx: &AppTestContext) {
    let mut legacy = StoredResponse::new(None);
    legacy.id = ResponseId::from("resp_legacy");
    legacy.input = Value::Array(
        (0..25)
            .map(|i| {
                json!({"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": format!("q{i}")}]})
            })
            .collect(),
    );
    legacy.raw_response =
        json!({"id": "resp_legacy", "object": "response", "status": "completed", "output": []});
    ctx.app_context
        .response_storage
        .store_response(legacy)
        .await
        .expect("the response is stored");
}

/// One GET of the app: the status, the content type and the JSON body.
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
async fn get(app: &axum::Router, uri: &str) -> (StatusCode, String, Value) {
    let request = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, content_type, json)
}

#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn texts(list: &Value) -> Vec<String> {
    list["data"]
        .as_array()
        .expect("a data array")
        .iter()
        .map(|item| {
            item["content"][0]["text"]
                .as_str()
                .unwrap_or("?")
                .to_owned()
        })
        .collect()
}

/// `limit` bounds the page and `has_more` says whether items follow it,
/// `order` picks the direction (the newest first by default), `after` starts
/// the page after the item named; a value outside the contract is refused in
/// the error envelope with the parameter named.
#[tokio::test]
async fn an_input_items_page_follows_limit_order_and_after() {
    let (ctx, app) = app_with_a_stored_response().await;

    let (status, _, list) = get(&app, "/v1/responses/resp_q/input_items?limit=1").await;
    assert_eq!(status, StatusCode::OK, "{list}");
    assert_eq!(
        texts(&list),
        ["q2"],
        "the newest first, one to the page: {list}"
    );
    assert_eq!(list["has_more"], true, "{list}");
    assert_eq!(list["first_id"], "msg_2", "{list}");
    assert_eq!(list["last_id"], "msg_2", "{list}");

    let (status, _, next) = get(&app, "/v1/responses/resp_q/input_items?limit=1&after=msg_2").await;
    assert_eq!(status, StatusCode::OK, "{next}");
    assert_eq!(texts(&next), ["q1"], "the page after the cursor: {next}");
    assert_eq!(next["has_more"], true, "{next}");

    let (status, _, ascending) = get(&app, "/v1/responses/resp_q/input_items?order=asc").await;
    assert_eq!(status, StatusCode::OK, "{ascending}");
    assert_eq!(texts(&ascending), ["q0", "q1", "q2"], "{ascending}");
    assert_eq!(ascending["has_more"], false, "{ascending}");

    let (status, _, descending) = get(&app, "/v1/responses/resp_q/input_items?order=desc").await;
    assert_eq!(status, StatusCode::OK, "{descending}");
    assert_eq!(texts(&descending), ["q2", "q1", "q0"], "{descending}");

    for (uri, code, param_prefix) in [
        (
            "/v1/responses/resp_q/input_items?limit=-1",
            "invalid_query_parameter",
            "limit",
        ),
        (
            "/v1/responses/resp_q/input_items?limit=0",
            "invalid_query_parameter",
            "limit",
        ),
        (
            "/v1/responses/resp_q/input_items?limit=1000",
            "invalid_query_parameter",
            "limit",
        ),
        (
            "/v1/responses/resp_q/input_items?order=sideways",
            "invalid_query_parameter",
            "order",
        ),
        (
            "/v1/responses/resp_q/input_items?include[]=bogus",
            "invalid_value",
            "include[",
        ),
    ] {
        let (status, content_type, error) = get(&app, uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}: {error}");
        assert!(
            content_type.starts_with("application/json"),
            "{uri}: {content_type}"
        );
        assert_eq!(error["error"]["code"], code, "{uri}: {error}");
        assert_eq!(
            error["error"]["type"], "invalid_request_error",
            "{uri}: {error}"
        );
        let names_it = error["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .starts_with(param_prefix)
            || error["error"]["param"]
                .as_str()
                .unwrap_or_default()
                .starts_with(param_prefix);
        assert!(names_it, "{uri}: the refusal names the parameter: {error}");
    }

    ctx.shutdown().await;
}

/// A record whose items were stored without ids pages to its end: the ids the
/// route lists for them are the same on every read, so a client that follows
/// `after=<last_id>` sees every item once and reaches `has_more: false`, and a
/// second walk reads the same ids.
#[tokio::test]
async fn a_record_without_item_ids_pages_to_its_end() {
    let (ctx, app) = app_with_a_stored_response().await;
    store_a_record_without_item_ids(&ctx).await;

    /// One walk of the record, ten items to a page: the ids in the order read,
    /// once the pages, the texts and the ids' shape are checked.
    async fn walk(app: &axum::Router) -> Vec<String> {
        let (mut ids, mut seen, mut pages): (Vec<String>, Vec<String>, usize) =
            (Vec::new(), Vec::new(), 0);
        let mut uri = Some("/v1/responses/resp_legacy/input_items?limit=10".to_owned());
        while let Some(uri_now) = uri.take() {
            let (status, _, list) = get(app, &uri_now).await;
            assert_eq!(status, StatusCode::OK, "{list}");
            pages += 1;
            assert!(pages <= 4, "the walk did not end in four pages: {ids:?}");
            let page = texts(&list);
            assert!(page.len() <= 10, "a page of at most ten: {list}");
            seen.extend(page);
            ids.extend(
                list["data"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|item| item["id"].as_str().unwrap_or_default().to_owned()),
            );
            if list["has_more"] == true {
                uri = Some(format!(
                    "/v1/responses/resp_legacy/input_items?limit=10&after={}",
                    list["last_id"].as_str().unwrap_or_default()
                ));
            }
        }
        assert_eq!(pages, 3, "{ids:?}");
        assert_eq!(
            seen,
            (0..25).rev().map(|i| format!("q{i}")).collect::<Vec<_>>()
        );
        assert!(ids.iter().all(|id| id.starts_with("msg_")), "{ids:?}");
        assert_eq!(
            ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
            25,
            "{ids:?}"
        );
        ids
    }
    let ids = walk(&app).await;
    assert_eq!(walk(&app).await, ids, "a second walk reads the same ids");

    ctx.shutdown().await;
}

/// `GET /v1/responses/{id}?stream=true` is refused in the error envelope (the
/// gateway keeps no event stream of a stored response to join), never answered
/// as a JSON object to a client that asked for events; an unknown `include`
/// value is refused as the conversation item routes refuse it; `stream=false`
/// and a valid `include` read the stored object.
#[tokio::test]
async fn a_stored_response_read_as_a_stream_is_refused_in_the_envelope() {
    let (ctx, app) = app_with_a_stored_response().await;

    let (status, content_type, error) = get(&app, "/v1/responses/resp_q?stream=true").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(
        content_type.starts_with("application/json"),
        "{content_type}"
    );
    assert_eq!(error["error"]["code"], "invalid_query_parameter", "{error}");
    assert!(
        error["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .starts_with("stream:"),
        "{error}"
    );

    let (status, _, error) = get(&app, "/v1/responses/resp_q?include[]=bogus").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["error"]["code"], "invalid_value", "{error}");
    assert_eq!(error["error"]["param"], "include[0]", "{error}");

    for uri in [
        "/v1/responses/resp_q",
        "/v1/responses/resp_q?stream=false",
        "/v1/responses/resp_q?include[]=message.output_text.logprobs",
    ] {
        let (status, content_type, object) = get(&app, uri).await;
        assert_eq!(status, StatusCode::OK, "{uri}: {object}");
        assert!(
            content_type.starts_with("application/json"),
            "{uri}: {content_type}"
        );
        assert_eq!(object["id"], "resp_q", "{uri}: {object}");
        assert_eq!(object["status"], "completed", "{uri}: {object}");
    }

    ctx.shutdown().await;
}
