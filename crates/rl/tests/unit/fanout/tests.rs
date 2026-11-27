use std::sync::Arc;

use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use openai_protocol::worker::{ConnectionMode, RuntimeType};
use serde_json::{json, Value};
use tower::ServiceExt;

use super::*;
use crate::{
    config::RlConfig,
    router,
    testing::{worker, FakeEngine, FakeView},
};

fn state(workers: Vec<RlWorkerInfo>, concurrency: usize) -> Arc<RlState> {
    let cfg = RlConfig {
        enabled: true,
        control_timeout_secs: 5,
        fanout_concurrency: concurrency,
    };
    Arc::new(RlState::new(Arc::new(FakeView(workers)), cfg))
}

async fn json_body(resp: Response) -> Value {
    serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes()).unwrap()
}

#[tokio::test]
async fn all_success_is_200_and_each_engine_called_once() {
    let e1 = FakeEngine::start(StatusCode::OK, json!({"ok": 1}), 0).await;
    let e2 = FakeEngine::start(StatusCode::OK, json!({"ok": 2}), 0).await;
    let app = router::<()>(state(
        vec![
            worker("w1", &e1.url, RuntimeType::Sglang),
            worker("w2", &e2.url, RuntimeType::Sglang),
            worker("v1", &format!("{}/v", e2.url), RuntimeType::Vllm),
        ],
        8,
    ));
    let req = Request::post("/engine/pause_generation?selector=engine%3Dsglang")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"mode":"abort"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["total"], 2);
    assert_eq!(body["succeeded"], 2);
    assert_eq!(body["failed"].as_array().unwrap().len(), 0);
    assert_eq!(body["results"]["w1"]["body"]["ok"], 1);
    assert_eq!(body["results"]["w2"]["status"], 200);
    assert_eq!(e1.seen().len(), 1);
    assert_eq!(
        e2.seen().len(),
        1,
        "vllm worker sharing e2 was not selected"
    );
    assert_eq!(&e1.seen()[0].body[..], br#"{"mode":"abort"}"#);
    assert_eq!(
        e1.seen()[0].query,
        None,
        "selector stripped from engine query"
    );
}

#[tokio::test]
async fn one_failure_is_207_and_names_the_worker() {
    let good = FakeEngine::start(StatusCode::OK, json!({}), 0).await;
    let bad = FakeEngine::start(StatusCode::INTERNAL_SERVER_ERROR, json!({"e": 1}), 0).await;
    let mut grpc = worker("g1", &format!("{}/grpc", good.url), RuntimeType::Sglang);
    grpc.connection_mode = ConnectionMode::Grpc;
    grpc.control_url = None;
    let app = router::<()>(state(
        vec![
            worker("w1", &good.url, RuntimeType::Sglang),
            worker("w2", &bad.url, RuntimeType::Sglang),
            worker("w3", "http://127.0.0.1:9", RuntimeType::Sglang),
            grpc,
        ],
        8,
    ));
    let req = Request::post("/engine/flush_cache?selector=engine%3Dsglang")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::MULTI_STATUS);
    let body = json_body(resp).await;
    assert_eq!(body["total"], 4);
    assert_eq!(body["succeeded"], 1);
    let failed: Vec<&str> = body["failed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["worker_id"].as_str().unwrap())
        .collect();
    assert_eq!(failed, ["g1", "w2", "w3"]);
    let by_id = |id: &str| {
        body["failed"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["worker_id"] == id)
            .cloned()
            .unwrap()
    };
    assert_eq!(by_id("w2")["error"], "upstream_error");
    assert_eq!(by_id("w2")["status"], 500);
    assert_eq!(by_id("w3")["error"], "upstream_unreachable");
    assert_eq!(by_id("g1")["error"], "no_control_endpoint");
    assert_eq!(by_id("g1")["connection_mode"], "grpc");
    assert_eq!(
        body["results"]["w2"]["status"], 500,
        "failed also in results"
    );
    assert!(
        body["results"].get("w3").is_none(),
        "no outcome for transport failure"
    );
}

#[tokio::test]
async fn selector_errors_are_400() {
    let e = FakeEngine::start(StatusCode::OK, json!({}), 0).await;
    let app = router::<()>(state(vec![worker("w1", &e.url, RuntimeType::Sglang)], 8));

    let r = app
        .clone()
        .oneshot(Request::post("/engine/pause").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    assert_eq!(json_body(r).await["error"], "selector_required");

    let r = app
        .clone()
        .oneshot(
            Request::post("/engine/pause?selector=")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(json_body(r).await["error"], "selector_required");

    let r = app
        .clone()
        .oneshot(
            Request::post("/engine/pause?selector=engine%3D%3D")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    assert_eq!(json_body(r).await["error"], "invalid_selector");

    let r = app
        .oneshot(
            Request::post("/engine/pause?selector=engine%3Dtrtllm")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    let b = json_body(r).await;
    assert_eq!(b["error"], "no_workers_match");
    assert_eq!(b["selector"], "engine=trtllm");
    assert!(e.seen().is_empty());
}

#[tokio::test]
async fn dp_ranks_collapse_to_one_call_and_url_selector_targets_one() {
    let e = FakeEngine::start(StatusCode::OK, json!({}), 0).await;
    let app = router::<()>(state(
        vec![
            worker("r0", &format!("{}@0", e.url), RuntimeType::Sglang),
            worker("r1", &format!("{}@1", e.url), RuntimeType::Sglang),
            worker("r2", &format!("{}@2", e.url), RuntimeType::Sglang),
        ],
        8,
    ));
    let sel = format!("url={}@0", e.url);
    let uri = format!("/engine/pause?selector={}", urlenc(&sel));
    let r = app
        .oneshot(Request::post(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(json_body(r).await["total"], 1);
    assert_eq!(e.seen().len(), 1);
}

#[tokio::test]
async fn concurrency_is_capped() {
    let e = FakeEngine::start(StatusCode::OK, json!({}), 150).await;
    let workers: Vec<RlWorkerInfo> = (0..6)
        .map(|i| {
            worker(
                &format!("w{i}"),
                &format!("{}/{i}", e.url),
                RuntimeType::Sglang,
            )
        })
        .collect();
    // base_url differs per worker (path suffix) so they are 6 targets on one engine.
    let app = router::<()>(state(workers, 2));
    let r = app
        .oneshot(
            Request::post("/engine/pause?selector=engine%3Dsglang")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(e.seen().len(), 6);
    assert!(e.peak_concurrency() <= 2, "peak {}", e.peak_concurrency());
}

fn urlenc(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// A fan-out reaches the workers driven over their engine connection
/// like any other, each once, with the same body.
#[tokio::test]
async fn a_fanout_reaches_workers_over_their_engine_connection() {
    use crate::testing::{zmq_worker, FakeCalls};

    let (a, b) = (FakeCalls::succeeding(2), FakeCalls::succeeding(1));
    let app = router::<()>(state(
        vec![zmq_worker("z1", a.clone()), zmq_worker("z2", b.clone())],
        8,
    ));
    let req = Request::post("/engine/release_memory_occupation?selector=engine%3Dtokenspeed")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"tags":["kv_cache"]}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["succeeded"], 2);
    for calls in [a, b] {
        let seen = calls.seen();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, "release_memory_occupation");
        assert_eq!(seen[0].1, Some(json!({"tags": ["kv_cache"]})));
    }
}
