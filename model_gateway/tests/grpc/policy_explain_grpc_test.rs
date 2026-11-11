//! A decision the request asked to see, read back through the admin plane:
//! the trace header of a request sent with `x-smg-cache-trace: 1` names the
//! decision id, and `GET /policy/explain/{id}` answers the ring's record of
//! that decision, whichever decisions came after it.
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test failures should panic"
)]

use std::{sync::Arc, time::Duration};

use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, Request, StatusCode},
    Router,
};
use openai_protocol::worker::WorkerStatus;
use serde_json::{json, Value};
use smg::{
    config::{PolicyEngineMode, RoutingMode},
    worker::{RuntimeType, Worker, WorkerType},
};
use tower::ServiceExt;

use crate::common::{
    grpc_harness::{
        grpc_router, grpc_router_config, grpc_test_context, grpc_workers, serve_canned_worker,
        Canned,
    },
    test_app::create_test_app_with_context,
};

const MODEL: &str = "explain-grpc-model";

async fn canned_worker() -> u16 {
    serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens: 3,
    })
    .await
}

/// A completion that asks for its decision record: the status, the decoded
/// `x-smg-cache-trace` response header when one came, and the body's text.
async fn traced_request(app: &Router) -> (StatusCode, Option<Value>, String) {
    let payload = json!({ "model": MODEL, "prompt": "hello", "max_tokens": 3, "stream": false });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(CONTENT_TYPE, "application/json")
                .header("x-smg-cache-trace", "1")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let header = response
        .headers()
        .get("x-smg-cache-trace")
        .map(|value| serde_json::from_str(value.to_str().unwrap()).unwrap());
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, header, String::from_utf8_lossy(&body).into_owned())
}

/// A completion that asks for its decision record and is served: the status
/// and the decoded header.
async fn traced_completion(app: &Router) -> (StatusCode, Value) {
    let (status, header, body) = traced_request(app).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let header = header.expect("the request asked for its decision record");
    (status, header)
}

async fn explain(app: &Router, id: &str) -> (StatusCode, Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(format!("/policy/explain/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

#[tokio::test]
async fn a_traced_decision_is_explained_by_the_id_its_header_names() {
    let (first, second) = (canned_worker().await, canned_worker().await);
    let config = grpc_router_config()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .round_robin_policy()
        .policy_engine(PolicyEngineMode::Engine)
        .build_unchecked();
    let app_context = grpc_test_context(config, MODEL).await;
    let workers = grpc_workers(
        &[(first, WorkerType::Regular), (second, WorkerType::Regular)],
        RuntimeType::TokenSpeed,
        MODEL,
    );
    let router = grpc_router(&app_context, workers).await;
    let app = create_test_app_with_context(Arc::from(router), Arc::clone(&app_context));

    let (_, header) = traced_completion(&app).await;
    let selection = &header["selections"][0];
    let worker = selection["worker"]
        .as_str()
        .expect("the record names the worker placed");
    let prediction = &selection["prediction"];
    assert_eq!(prediction["source"], "policy_engine", "{header}");
    let id = prediction["decision_id"]
        .as_u64()
        .expect("the trace header names the decision id");
    assert_eq!(prediction["outcome"], "pick");
    assert_eq!(prediction["stage"], "round-robin");

    // A later traced decision moves the ring on; the first id still answers
    // its own record, not the newest.
    let (_, later) = traced_completion(&app).await;
    let later_id = later["selections"][0]["prediction"]["decision_id"]
        .as_u64()
        .unwrap();
    assert_ne!(id, later_id);

    let (status, record) = explain(&app, &id.to_string()).await;
    assert_eq!(status, StatusCode::OK, "{record}");
    assert_eq!(record["decision_id"], id);
    assert_eq!(record["model"], MODEL);
    assert_eq!(record["profile"], "default");
    assert_eq!(record["phase"], "single");
    assert_eq!(record["outcome"], "pick");
    assert_eq!(record["stage"], "round-robin");
    assert_eq!(record["survivors"], 2);
    assert_eq!(record["pool_size"], 2);
    assert_eq!(record["targets"][0]["worker"], worker, "{record}");
    assert_eq!(record["top"].as_array().map(Vec::len), Some(2));

    let (status, body) = explain(&app, &(later_id + 1_000_000).to_string()).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "decision_not_found", "{body}");
}

/// A request the engine sheds gets its decision named too: with every worker
/// out of the pool the decision is a shed, the refused response carries the
/// trace header with the shed's id and status, and the explain route serves
/// the shed's record.
#[tokio::test]
async fn a_traced_shed_is_explained_by_the_id_its_refusal_names() {
    let (first, second) = (canned_worker().await, canned_worker().await);
    let config = grpc_router_config()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .round_robin_policy()
        .policy_engine(PolicyEngineMode::Engine)
        .build_unchecked();
    let app_context = grpc_test_context(config, MODEL).await;
    let workers = grpc_workers(
        &[(first, WorkerType::Regular), (second, WorkerType::Regular)],
        RuntimeType::TokenSpeed,
        MODEL,
    );
    let handles: Vec<Arc<dyn Worker>> = workers.iter().map(Arc::clone).collect();
    let router = grpc_router(&app_context, workers).await;
    let app = create_test_app_with_context(Arc::from(router), Arc::clone(&app_context));
    for worker in &handles {
        worker.set_status(WorkerStatus::NotReady);
    }

    let (status, header, body) = traced_request(&app).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    let header = header.expect("a refused request that asked for its record gets the trace header");
    assert_eq!(header["status"], 503, "{header}");
    let prediction = &header["selections"][0]["prediction"];
    assert_eq!(prediction["source"], "policy_engine", "{header}");
    // Every worker fails the health filter: the shed names that stage.
    assert_eq!(prediction["outcome"], "no_endpoints", "{header}");
    assert_eq!(prediction["stage"], "healthy", "{header}");
    let id = prediction["decision_id"]
        .as_u64()
        .expect("the refusal names the decision id");

    let (status, record) = explain(&app, &id.to_string()).await;
    assert_eq!(status, StatusCode::OK, "{record}");
    assert_eq!(record["decision_id"], id);
    assert_eq!(record["model"], MODEL);
    assert_eq!(record["outcome"], prediction["outcome"], "{record}");
    assert_eq!(record["stage"], prediction["stage"], "{record}");
    assert_eq!(
        record["targets"].as_array().map(Vec::len),
        Some(0),
        "{record}"
    );
}
