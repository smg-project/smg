use std::sync::Arc;

use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use openai_protocol::worker::{ConnectionMode, RuntimeType};
use serde_json::Value;
use tower::ServiceExt;

use super::*;
use crate::{
    config::RlConfig,
    router,
    state::RlState,
    testing::{worker, zmq_worker, FakeCalls, FakeView},
};

fn state(workers: Vec<RlWorkerInfo>) -> Arc<RlState> {
    Arc::new(RlState::new(
        Arc::new(FakeView(workers)),
        RlConfig::default(),
    ))
}

async fn json_body(resp: Response) -> Value {
    serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap()
}

#[test]
fn collapse_groups_dp_ranks_and_sorts() {
    let ws = vec![
        worker("b1", "http://b:1@1", RuntimeType::Sglang),
        worker("a0", "http://a:1", RuntimeType::Sglang),
        worker("b0", "http://b:1@0", RuntimeType::Sglang),
        worker("b2", "http://b:1@2", RuntimeType::Sglang),
    ];
    let rows = collapse(ws);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0.id, "a0");
    assert_eq!(rows[0].1, 1);
    assert_eq!(rows[1].0.id, "b0", "lowest rank wins");
    assert_eq!(rows[1].1, 3);
}

#[tokio::test]
async fn collapse_keeps_non_dp_worker_sharing_a_dp_base_url() {
    let ws = vec![
        worker("n", "http://b:1", RuntimeType::Sglang),
        worker("r0", "http://b:1@0", RuntimeType::Sglang),
        worker("r1", "http://b:1@1", RuntimeType::Sglang),
    ];
    let rows = collapse(ws.clone());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0.id, "n");
    assert_eq!(rows[0].1, 1);
    assert_eq!(rows[1].0.id, "r0");
    assert_eq!(rows[1].1, 2);

    let app = router::<()>(state(ws));

    let resp = app
        .clone()
        .oneshot(Request::get("/workers/r1").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["dp_ranks"], 2);

    let resp = app
        .oneshot(Request::get("/workers/n").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["dp_ranks"], 1);
}

#[test]
fn merged_labels_add_synthetic_keys_that_shadow() {
    let mut w = worker("w1", "http://a:1", RuntimeType::Vllm);
    w.labels.insert("engine".to_string(), "spoofed".to_string());
    w.labels.insert("role".to_string(), "reward".to_string());
    let m = merged_labels(&w);
    assert_eq!(m["engine"], "vllm");
    assert_eq!(m["id"], "w1");
    assert_eq!(m["url"], "http://a:1");
    assert_eq!(m["model"], "mock-model");
    assert_eq!(m["health"], "ready");
    assert_eq!(m["weight_version"], "default");
    assert_eq!(m["role"], "reward");
    assert_eq!(m["tp_size"], "1");
}

#[tokio::test]
async fn list_reports_the_protocol_version() {
    let app = router::<()>(state(vec![worker("w1", "http://a:1", RuntimeType::Sglang)]));
    let resp = app
        .oneshot(Request::get("/workers").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body: Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["protocol_version"], 1);
}

#[tokio::test]
async fn list_and_get_workers() {
    let mut w = worker("w1", "http://a:1", RuntimeType::Sglang);
    w.labels.insert("tp_size".to_string(), "x".to_string());
    w.labels.insert("version".to_string(), "0.5.15".to_string());
    let app = router::<()>(state(vec![w]));

    let resp = app
        .clone()
        .oneshot(Request::get("/workers").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["total"], 1);
    let e = &body["workers"][0];
    assert_eq!(e["id"], "w1");
    assert_eq!(e["engine"], "sglang");
    assert_eq!(e["engine_version"], "0.5.15");
    assert_eq!(e["tp_size"], Value::Null, "garbage label -> null");
    assert_eq!(e["dp_size"], 1);
    assert_eq!(e["dp_ranks"], 1);
    assert_eq!(e["health"], "ready");
    assert_eq!(e["weight_version"], "default");
    assert_eq!(e["capabilities"]["source"], "static");
    assert_eq!(e["capabilities"]["pause_modes"][0], "abort");

    let resp = app
        .clone()
        .oneshot(Request::get("/workers/w1").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = app
        .oneshot(Request::get("/workers/nope").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn entry_reports_the_control_url() {
    let mut g = worker("g1", "grpc://a:1", RuntimeType::TokenSpeed);
    g.connection_mode = ConnectionMode::Grpc;
    g.control_url = Some("http://a:40100".to_string());
    let mut n = worker("n1", "grpc://b:1", RuntimeType::TokenSpeed);
    n.connection_mode = ConnectionMode::Grpc;
    n.control_url = None;
    let z = zmq_worker("z1", FakeCalls::succeeding(1));
    let app = router::<()>(state(vec![g, n, z]));
    let resp = app
        .oneshot(Request::get("/workers").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body = json_body(resp).await;
    let by_id = |id: &str| {
        body["workers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|w| w["id"] == id)
            .cloned()
            .unwrap()
    };
    assert_eq!(by_id("g1")["control_url"], "http://a:40100");
    assert_eq!(by_id("n1")["control_url"], Value::Null);
    assert_eq!(by_id("g1")["engine"], "tokenspeed");
    // A worker controlled over its engine connection says so; the
    // others leave the flag out.
    assert_eq!(by_id("z1")["control_url"], Value::Null);
    assert_eq!(by_id("z1")["engine_control_calls"], true);
    assert_eq!(by_id("n1").get("engine_control_calls"), None);
}
