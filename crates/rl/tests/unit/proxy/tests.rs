use std::sync::Arc;

use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use openai_protocol::worker::{ConnectionMode, RuntimeType};
use serde_json::json;
use tower::ServiceExt;

use super::*;
use crate::{
    calls::{CallFailure, RankReply},
    config::RlConfig,
    router,
    testing::{test_client, worker, zmq_worker, FakeCalls, FakeEngine, FakeView},
};

fn state(workers: Vec<RlWorkerInfo>, timeout_secs: u64) -> Arc<RlState> {
    let cfg = RlConfig {
        enabled: true,
        control_timeout_secs: timeout_secs,
        fanout_concurrency: 4,
    };
    Arc::new(RlState::new(Arc::new(FakeView(workers)), cfg))
}

async fn json_body(resp: Response) -> Value {
    serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap()
}

#[tokio::test]
async fn forwards_method_path_query_body_and_worker_key_only() {
    let engine = FakeEngine::start(StatusCode::OK, json!({"success": true}), 0).await;
    let mut w = worker("w1", &engine.url, RuntimeType::Vllm);
    w.api_key = Some("engine-secret".to_string());
    let app = router::<()>(state(vec![w], 5));

    let req = Request::post("/workers/w1/engine/pause?mode=keep&selector=engine%3Dvllm")
        .header("content-type", "application/json")
        .header("authorization", "Bearer caller-token")
        .header("x-request-id", "req-1")
        .header("x-smg-routing-key", "leak")
        .body(Body::from(r#"{"raw": "bytes"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["worker_id"], "w1");
    assert_eq!(body["status"], 200);
    assert_eq!(body["body"]["success"], true);
    assert!(body["latency_ms"].is_u64());

    let seen = engine.seen();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].method, Method::POST);
    assert_eq!(seen[0].path, "/pause");
    assert_eq!(seen[0].query.as_deref(), Some("mode=keep"));
    assert_eq!(&seen[0].body[..], br#"{"raw": "bytes"}"#);
    assert_eq!(seen[0].headers["authorization"], "Bearer engine-secret");
    assert_eq!(seen[0].headers["x-request-id"], "req-1");
    assert!(seen[0].headers.get("x-smg-routing-key").is_none());
}

#[tokio::test]
async fn defaults_content_type_when_body_has_no_header() {
    let engine = FakeEngine::start(StatusCode::OK, json!({}), 0).await;
    let app = router::<()>(state(
        vec![worker("w1", &engine.url, RuntimeType::Sglang)],
        5,
    ));

    let req = Request::post("/workers/w1/engine/update_weight_version")
        .header(
            "traceparent",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        )
        .body(Body::from(r#"{"new_version":"3"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let seen = engine.seen();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].headers["content-type"], "application/json");
    assert_eq!(
        seen[0].headers["traceparent"],
        "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    );
    assert_eq!(&seen[0].body[..], br#"{"new_version":"3"}"#);
}

#[tokio::test]
async fn get_is_forwarded_and_text_bodies_are_wrapped() {
    let engine = FakeEngine::start(StatusCode::OK, json!("plain"), 0).await;
    let app = router::<()>(state(
        vec![worker("w1", &engine.url, RuntimeType::Sglang)],
        5,
    ));
    let resp = app
        .oneshot(
            Request::get("/workers/w1/engine/server_info")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(json_body(resp).await["body"], "plain");
    assert_eq!(engine.seen()[0].method, Method::GET);
}

#[tokio::test]
async fn upstream_status_is_mirrored_not_translated() {
    let engine = FakeEngine::start(
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({"error": "boom"}),
        0,
    )
    .await;
    let app = router::<()>(state(
        vec![worker("w1", &engine.url, RuntimeType::Sglang)],
        5,
    ));
    let resp = app
        .oneshot(
            Request::post("/workers/w1/engine/flush_cache")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = json_body(resp).await;
    assert_eq!(body["status"], 500);
    assert_eq!(body["body"]["error"], "boom");
}

#[tokio::test]
async fn errors_404_422_400_405() {
    let engine = FakeEngine::start(StatusCode::OK, json!({}), 0).await;
    let mut grpc = worker("g1", &engine.url, RuntimeType::Sglang);
    grpc.connection_mode = ConnectionMode::Grpc;
    grpc.control_url = None;
    let app = router::<()>(state(
        vec![worker("w1", &engine.url, RuntimeType::Sglang), grpc],
        5,
    ));

    let r = app
        .clone()
        .oneshot(
            Request::post("/workers/zz/engine/pause")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::NOT_FOUND);
    assert_eq!(json_body(r).await["error"], "worker_not_found");

    let r = app
        .clone()
        .oneshot(
            Request::post("/workers/g1/engine/pause")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json_body(r).await["error"], "no_control_endpoint");

    let r = app
        .clone()
        .oneshot(
            Request::post("/workers/w1/engine/a/b/c/d/e")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    assert_eq!(json_body(r).await["error"], "invalid_engine_path");

    let r = app
        .oneshot(
            Request::delete("/workers/w1/engine/pause")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert!(engine.seen().is_empty());
}

/// The transport SMG uses for data is irrelevant to control: a gRPC
/// worker with a control endpoint is proxied exactly like an HTTP one.
#[tokio::test]
async fn grpc_worker_with_a_control_endpoint_is_proxied() {
    let control_app = FakeEngine::start(StatusCode::OK, json!({"success": true}), 0).await;
    let mut w = worker("g1", "grpc://engine:30000", RuntimeType::TokenSpeed);
    w.connection_mode = ConnectionMode::Grpc;
    w.control_url = Some(control_app.url.clone());
    w.api_key = Some("ts-secret".to_string());
    let app = router::<()>(state(vec![w], 5));

    let resp = app
        .oneshot(
            Request::post("/workers/g1/engine/pause_generation")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"mode":"wait"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let seen = control_app.seen();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].path, "/pause_generation");
    assert_eq!(&seen[0].body[..], br#"{"mode":"wait"}"#);
    assert_eq!(seen[0].headers["authorization"], "Bearer ts-secret");
}

#[tokio::test]
async fn control_url_without_a_client_is_reported_not_routed() {
    let mut w = worker("g2", "grpc://engine:30000", RuntimeType::TokenSpeed);
    w.connection_mode = ConnectionMode::Grpc;
    w.control_url = Some("http://ctl:1".to_string());
    w.control_client = Err("bad CA bundle".to_string());
    let app = router::<()>(state(vec![w], 5));
    let r = app
        .oneshot(
            Request::post("/workers/g2/engine/pause")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = json_body(r).await;
    assert_eq!(body["error"], "no_control_endpoint");
    let message = body["message"].as_str().unwrap();
    assert!(message.contains("bad CA bundle"), "{message}");
}

/// The control deadline is applied per request, so it holds even when
/// the worker's client carries no client-level timeout of its own.
#[tokio::test]
async fn control_timeout_is_applied_per_request_on_the_worker_client() {
    let slow_engine = FakeEngine::start(StatusCode::OK, json!({}), 1500).await;
    let mut w = worker("slow", &slow_engine.url, RuntimeType::Sglang);
    w.control_client = Ok(test_client());
    let app = router::<()>(state(vec![w], 1));

    let r = app
        .oneshot(
            Request::post("/workers/slow/engine/flush_cache")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::GATEWAY_TIMEOUT);
    let body = json_body(r).await;
    assert_eq!(body["error"], "upstream_timeout");
    assert!(body["message"]
        .as_str()
        .unwrap()
        .ends_with("timed out after 1s"));
}

#[tokio::test]
async fn unreachable_and_timeout_map_to_502_and_504() {
    let dead = worker("d1", "http://127.0.0.1:9", RuntimeType::Sglang);
    let slow_engine = FakeEngine::start(StatusCode::OK, json!({}), 1500).await;
    let slow = worker("s1", &slow_engine.url, RuntimeType::Sglang);
    let app = router::<()>(state(vec![dead, slow], 1));

    let r = app
        .clone()
        .oneshot(
            Request::post("/workers/d1/engine/pause")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(json_body(r).await["error"], "upstream_unreachable");

    let r = app
        .oneshot(
            Request::post("/workers/s1/engine/pause")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(json_body(r).await["error"], "upstream_timeout");
}

#[test]
fn large_text_bodies_are_truncated() {
    let big = vec![b'x'; BODY_CAP + 10];
    let (v, truncated) = parse_body(None, &big, false);
    assert!(truncated);
    assert_eq!(v.as_str().unwrap().len(), BODY_CAP);
}

#[tokio::test]
async fn large_json_bodies_are_truncated_and_flagged() {
    let engine = FakeEngine::start(
        StatusCode::OK,
        json!({"blob": "x".repeat(BODY_CAP + 4096)}),
        0,
    )
    .await;
    let app = router::<()>(state(
        vec![worker("w1", &engine.url, RuntimeType::Sglang)],
        5,
    ));
    let resp = app
        .oneshot(
            Request::get("/workers/w1/engine/server_info")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["body_truncated"], true);
    assert!(body["body"].as_str().unwrap().len() <= BODY_CAP);
}

/// A worker with no control app whose engine connection carries the
/// calls is driven over it: the route is the call's method, its JSON body
/// the argument, the control deadline the wait, and the ranks' replies
/// come back in the route's own shape and status.
#[tokio::test]
async fn a_worker_without_a_control_app_is_driven_over_its_engine_connection() {
    let calls = FakeCalls::succeeding(2);
    let app = router::<()>(state(vec![zmq_worker("z1", calls.clone())], 7));
    let resp = app
        .oneshot(
            Request::post("/workers/z1/engine/update_weights_from_disk")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model_path":"/ckpt","flush_cache":true}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["status"], 200);
    assert_eq!(body["body"], json!({"success": true, "message": ""}));
    let seen = calls.seen();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].0, "update_weights_from_disk");
    assert_eq!(
        seen[0].1,
        Some(json!({"model_path": "/ckpt", "flush_cache": true}))
    );
    assert_eq!(seen[0].2, Duration::from_secs(7));

    // A route without a body is a call without an argument; a rank's
    // refusal is the route's 400 with the engine's words.
    let refused = FakeCalls::answering(Ok(vec![RankReply {
        success: false,
        message: "a pause is already in progress".to_string(),
    }]));
    let app = router::<()>(state(vec![zmq_worker("z2", refused.clone())], 7));
    let resp = app
        .oneshot(
            Request::post("/workers/z2/engine/pause_generation")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = json_body(resp).await;
    assert_eq!(body["body"]["message"], "a pause is already in progress");
    assert_eq!(refused.seen()[0].1, None);

    // `{}`, which callers send to bodyless routes, is no argument either.
    let flushed = FakeCalls::succeeding(1);
    let app = router::<()>(state(vec![zmq_worker("z4", flushed.clone())], 7));
    let resp = app
        .oneshot(
            Request::post("/workers/z4/engine/flush_cache")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        flushed.seen()[0],
        ("flush_cache".to_string(), None, Duration::from_secs(7))
    );
}

/// The engine connection carries POST operations named by one route
/// segment, with an object body; anything else is refused before any call,
/// and a call that times out or cannot be made maps as an HTTP one does.
#[tokio::test]
async fn what_the_engine_connection_cannot_carry_is_refused() {
    let calls = FakeCalls::succeeding(1);
    let app = router::<()>(state(vec![zmq_worker("z1", calls.clone())], 5));
    for (request, status, code) in [
        (
            Request::get("/workers/z1/engine/get_weight_version")
                .body(Body::empty())
                .unwrap(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "no_control_endpoint",
        ),
        (
            Request::post("/workers/z1/engine/v1/loads")
                .body(Body::empty())
                .unwrap(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "no_control_endpoint",
        ),
        (
            Request::post("/workers/z1/engine/pause_generation")
                .body(Body::from("[1, 2]"))
                .unwrap(),
            StatusCode::BAD_REQUEST,
            "invalid_body",
        ),
    ] {
        let resp = app.clone().oneshot(request).await.unwrap();
        assert_eq!(resp.status(), status);
        assert_eq!(json_body(resp).await["error"], code);
    }
    assert!(calls.seen().is_empty());

    for (failure, status) in [
        (CallFailure::Timeout, StatusCode::GATEWAY_TIMEOUT),
        (
            CallFailure::Failed("engine not connected".to_string()),
            StatusCode::BAD_GATEWAY,
        ),
        (
            CallFailure::Unsupported(
                "the engine's handshake declares no control reply".to_string(),
            ),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
    ] {
        let app = router::<()>(state(
            vec![zmq_worker("z3", FakeCalls::answering(Err(failure)))],
            5,
        ));
        let resp = app
            .oneshot(
                Request::post("/workers/z3/engine/continue_generation")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), status);
    }
}

/// A control app the operator names wins over the engine connection.
#[tokio::test]
async fn a_labelled_control_app_wins_over_the_engine_connection() {
    let control_app = FakeEngine::start(StatusCode::OK, json!({"success": true}), 0).await;
    let calls = FakeCalls::succeeding(1);
    let mut w = zmq_worker("z1", calls.clone());
    w.control_url = Some(control_app.url.clone());
    let app = router::<()>(state(vec![w], 5));
    let resp = app
        .oneshot(
            Request::post("/workers/z1/engine/continue_generation")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(control_app.seen().len(), 1);
    assert!(calls.seen().is_empty());
}
