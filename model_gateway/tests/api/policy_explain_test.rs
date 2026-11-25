//! `GET /policy/explain/{request_id}` on the production route table: a kept
//! routing decision is read back by its request's id, and an id that finds
//! none is answered 404 with its reason.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, StatusCode},
    Router,
};
use serde_json::{json, Value};
use smg::config::RouterConfig;
use smg_grand_central::document::PolicySpec;
use tower::ServiceExt;

use crate::common::{self, AppTestContext, TestWorkerConfig};

/// A gateway of two workers whose routing settings are the default ones,
/// which a supplied policy spec stands beside. With a `sample` it runs
/// Grand Central under the `round_robin` preset as a policy spec that
/// keeps every `sample`th decision, and none where it is zero. Without one
/// it leaves Grand Central off.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
async fn gateway(sample: Option<u32>) -> AppTestContext {
    let mut config = RouterConfig::builder()
        .regular_mode(vec![])
        .host("127.0.0.1")
        .port(0)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    if let Some(sample) = sample {
        let mut spec = PolicySpec::preset("round_robin").expect("a built-in preset");
        spec.generation = 1;
        spec.explain.sample = sample;
        config.grand_central = true;
        config.policy_spec = Some(spec);
    }
    let workers = vec![TestWorkerConfig::healthy(0), TestWorkerConfig::healthy(0)];
    AppTestContext::new_with_config(config, workers).await
}

/// One request to `app`: its status, its headers and its JSON body, null
/// where the body is no JSON.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
async fn sent(app: &Router, request: Request) -> (StatusCode, HeaderMap, Value) {
    let response = app.clone().oneshot(request).await.expect("an answer");
    let (status, headers) = (response.status(), response.headers().clone());
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("a whole body");
    let body = serde_json::from_slice(&body).unwrap_or(Value::Null);
    (status, headers, body)
}

/// A generate request whose body states `rid` as its id.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn generate(rid: &str) -> Request {
    Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"text": "one request", "stream": false, "rid": rid}).to_string(),
        ))
        .expect("a request")
}

/// The read of the decision kept of the request `id` names.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
fn explain(id: &str) -> Request {
    Request::builder()
        .method("GET")
        .uri(format!("/policy/explain/{id}"))
        .body(Body::empty())
        .expect("a request")
}

/// Under a policy spec that keeps every decision, a request is read back by
/// the id its body states. The answer names one target, by the address of
/// the worker the request was sent to: the one the gateway names in the
/// response's routed-worker header. It holds the names of the plan and the
/// profile that decided, and states no id of the request.
#[tokio::test]
async fn a_kept_decision_is_read_by_its_requests_id_with_the_address_it_was_sent_to() {
    let gateway = gateway(Some(1)).await;
    let app = gateway.create_app();

    for id in ["explain-a", "explain-b", "explain-c"] {
        let (status, headers, _) = sent(&app, generate(id)).await;
        assert_eq!(status, StatusCode::OK, "{id} is served");
        let sent_to = headers["x-smg-routed-worker-id"].to_str().unwrap();
        assert!(gateway.worker_urls.iter().any(|url| url == sent_to));

        let (status, _, kept) = sent(&app, explain(id)).await;
        assert_eq!(status, StatusCode::OK, "{id}: {kept}");
        let targets = kept["decision"]["pick"]["targets"].as_array().unwrap();
        assert_eq!(targets.len(), 1, "{id}: {kept}");
        assert_eq!(targets[0]["url"], json!(sent_to), "{id}: {kept}");
        assert!(targets[0]["id"].is_string(), "{id}: {kept}");
        assert_eq!(kept["profile_name"], json!("round_robin"), "{id}: {kept}");
        assert!(kept["plan_name"].is_string(), "{id}: {kept}");
        assert_eq!(kept.get("request_id"), None, "{id}: {kept}");
    }

    gateway.shutdown().await;
}

/// An id that finds no kept decision is answered 404 in the control plane's
/// error envelope, with one of three reasons: Grand Central is off,
/// the policy spec keeps no sample, or the sample holds no decision of that
/// id. A run that turns Grand Central on for every context reads
/// the second reason in the first one's place.
#[tokio::test]
async fn an_id_that_finds_no_kept_decision_is_a_404_with_its_reason() {
    let off = if common::grand_central_forced() {
        "EXPLAIN_SAMPLE_OFF"
    } else {
        "GRAND_CENTRAL_OFF"
    };
    for (sample, code) in [
        (None, off),
        (Some(0), "EXPLAIN_SAMPLE_OFF"),
        (Some(1), "DECISION_NOT_KEPT"),
    ] {
        let gateway = gateway(sample).await;
        let app = gateway.create_app();
        let (status, _, _) = sent(&app, generate("explain-kept")).await;
        assert_eq!(status, StatusCode::OK, "{sample:?}: the request is served");

        let (status, _, refused) = sent(&app, explain("explain-other")).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{sample:?}: {refused}");
        assert_eq!(refused["code"], json!(code), "{sample:?}: {refused}");
        assert!(refused["error"].is_string(), "{sample:?}: {refused}");

        gateway.shutdown().await;
    }
}
