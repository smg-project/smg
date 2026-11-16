//! The control-plane handlers of the production route table name the
//! resource they acted on in their response, for the audit outcome line: a
//! worker's replacement names its worker id; a refused body and a read name
//! none.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
    response::Response,
    Router,
};
use serde_json::{json, Value};
use smg_auth::AuditResource;
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[expect(
    clippy::unwrap_used,
    reason = "test helper: a malformed request is a test bug"
)]
async fn call(app: &Router, method: &str, path: &str, body: Option<Value>) -> Response {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header(CONTENT_TYPE, "application/json");
    let body = match body {
        Some(body) => Body::from(body.to_string()),
        None => Body::empty(),
    };
    app.clone()
        .oneshot(request.body(body).unwrap())
        .await
        .unwrap()
}

#[expect(clippy::unwrap_used, reason = "test helper: the answer is JSON")]
async fn json_body(response: Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn a_worker_replacement_names_its_worker_id_and_a_refusal_or_a_read_names_none() {
    let config = TestRouterConfig::round_robin(4351);
    let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20351)]).await;
    let app = ctx.create_app();

    let listed = call(&app, "GET", "/workers", None).await;
    assert_eq!(listed.status(), StatusCode::OK);
    assert!(
        listed.extensions().get::<AuditResource>().is_none(),
        "a read names no resource"
    );
    let worker = json_body(listed).await["workers"][0].clone();
    let (worker_id, url) = (
        worker["id"].as_str().unwrap(),
        worker["url"].as_str().unwrap(),
    );

    let replaced = call(
        &app,
        "PUT",
        &format!("/workers/{worker_id}"),
        Some(json!({ "url": url, "priority": 7 })),
    )
    .await;
    assert_eq!(replaced.status(), StatusCode::ACCEPTED);
    assert_eq!(
        replaced.extensions().get::<AuditResource>(),
        Some(&AuditResource(worker_id.to_string())),
        "the accepted replacement names the worker it queued"
    );

    let refused = call(&app, "POST", "/workers", Some(json!({}))).await;
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
    assert!(
        refused.extensions().get::<AuditResource>().is_none(),
        "a refused body names no resource"
    );

    ctx.shutdown().await;
}
