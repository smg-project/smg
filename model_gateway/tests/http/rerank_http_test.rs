//! The rerank routes through the production server, the gateway and the HTTP
//! router over a worker that answers in the Cohere-style `results` object of
//! the OpenAI-compatible rerank servers, not in the gateway's native list.
#![expect(clippy::unwrap_used, reason = "test failures should panic")]

use std::sync::{Arc, Mutex};

use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::{header::CONTENT_TYPE, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use common::test_app::{create_test_app_context, create_test_app_with_context};
use serde_json::{json, Value};
use smg::{
    routers::{factory::router_ids, gateway::Gateway, http::router::Router as HttpRouter},
    worker::{BasicWorkerBuilder, ModelCard},
};
use tokio::{net::TcpListener, task::JoinHandle};
use tower::ServiceExt;

use crate::common;

const MODEL: &str = "rerank-model";

/// What the worker answers and what it was asked: the path and the body of
/// every request.
#[derive(Clone)]
struct UpstreamState {
    requests: Arc<Mutex<Vec<(String, Value)>>>,
    body: Value,
}

struct Upstream {
    url: String,
    state: UpstreamState,
    task: JoinHandle<()>,
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn record_request(State(state): State<UpstreamState>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let body = to_bytes(body, 1024 * 1024).await.unwrap();
    state.requests.lock().unwrap().push((
        parts.uri.path().to_string(),
        serde_json::from_slice(&body).unwrap(),
    ));
    Json(state.body.clone()).into_response()
}

#[expect(
    clippy::disallowed_methods,
    reason = "test server is aborted by the owning Upstream on drop"
)]
async fn upstream(body: Value) -> Upstream {
    let state = UpstreamState {
        requests: Arc::new(Mutex::new(Vec::new())),
        body,
    };
    let app = Router::new()
        .route("/v1/rerank", post(record_request))
        .with_state(state.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Upstream { url, state, task }
}

/// The production app over `upstream` as the one regular HTTP worker of
/// [`MODEL`], dispatched through the gateway as a served deployment is.
async fn app_for(upstream: &Upstream) -> Router {
    let ctx = create_test_app_context().await;
    ctx.worker_registry
        .register(Arc::new(
            BasicWorkerBuilder::new(&upstream.url)
                .model(ModelCard::new(MODEL))
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        ))
        .unwrap();
    let router = Arc::new(HttpRouter::new(&ctx).await.unwrap());
    let gateway = Arc::new(Gateway::new(ctx.worker_registry.clone()));
    gateway.register_router(router_ids::HTTP_REGULAR, router);
    create_test_app_with_context(gateway, ctx)
}

fn request(path: &str, body: Value) -> Request {
    Request::builder()
        .method("POST")
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// The answer of a server that writes the Cohere-style object: `results` of
/// `index` and `relevance_score`, sorted, no document echoed, the model and
/// the usage beside them.
fn results_object() -> Value {
    json!({
        "model": MODEL,
        "object": "list",
        "usage": {"prompt_tokens": 14, "total_tokens": 14},
        "results": [
            {"index": 1, "relevance_score": 0.93},
            {"index": 0, "relevance_score": 0.07}
        ]
    })
}

async fn json_of(response: Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap()).unwrap()
}

#[tokio::test]
async fn a_results_object_from_the_worker_is_answered_in_the_gateway_shape() {
    let upstream = upstream(results_object()).await;
    let app = app_for(&upstream).await;

    let response = app
        .oneshot(request(
            "/rerank",
            json!({
                "model": MODEL,
                "query": "what is rust",
                "documents": ["a kind of oxide", "a systems language"]
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_of(response).await;
    assert_eq!(body["object"], "rerank");
    assert_eq!(body["model"], MODEL);
    assert_eq!(
        body["results"],
        json!([{"score": 0.93, "index": 1}, {"score": 0.07, "index": 0}]),
        "the worker's ranking, in the gateway's spelling, no document echoed"
    );
    let requests = upstream.state.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].0, "/v1/rerank", "the worker's own rerank route");
    assert_eq!(requests[0].1["model"], MODEL);
    assert_eq!(
        requests[0].1["documents"],
        json!(["a kind of oxide", "a systems language"])
    );
}

#[tokio::test]
async fn v1_rerank_routes_by_the_body_model_and_applies_top_n() {
    let upstream = upstream(results_object()).await;
    let app = app_for(&upstream).await;

    let response = app
        .oneshot(request(
            "/v1/rerank",
            json!({
                "model": MODEL,
                "query": "what is rust",
                "documents": ["a kind of oxide", "a systems language"],
                "top_n": 1
            }),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_of(response).await;
    assert_eq!(
        body["results"],
        json!([{"score": 0.93, "index": 1}]),
        "top_n truncates the worker's ranked list"
    );
    let requests = upstream.state.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].1["model"], MODEL,
        "the body's model reaches the worker"
    );
}

#[tokio::test]
async fn v1_rerank_for_a_model_no_worker_serves_is_refused_before_dispatch() {
    let upstream = upstream(results_object()).await;
    let app = app_for(&upstream).await;

    let response = app
        .oneshot(request(
            "/v1/rerank",
            json!({"model": "other-model", "query": "q", "documents": ["d"]}),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert!(upstream.state.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_document_object_from_the_worker_is_the_document_and_dropped_when_not_asked() {
    let upstream = upstream(json!({
        "id": "r1",
        "model": MODEL,
        "usage": {"total_tokens": 9},
        "results": [
            {"index": 0, "document": {"text": "a systems language"}, "relevance_score": 0.8},
            {"index": 1, "document": {"text": "a kind of oxide"}, "relevance_score": 0.2}
        ]
    }))
    .await;
    let app = app_for(&upstream).await;
    let documents = json!(["a systems language", "a kind of oxide"]);

    let response = app
        .clone()
        .oneshot(request(
            "/rerank",
            json!({"model": MODEL, "query": "q", "documents": documents}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        json_of(response).await["results"][0]["document"],
        "a systems language"
    );

    let response = app
        .oneshot(request(
            "/rerank",
            json!({"model": MODEL, "query": "q", "documents": documents, "return_documents": false}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        json_of(response).await["results"][0].get("document"),
        None,
        "return_documents false drops the echoed document"
    );
}

#[tokio::test]
async fn a_model_less_v1_rerank_reaches_a_worker_without_a_model_key() {
    let upstream = upstream(results_object()).await;
    let app = app_for(&upstream).await;

    let response = app
        .oneshot(request(
            "/v1/rerank",
            json!({"query": "what is rust", "documents": ["a kind of oxide", "a systems language"]}),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let requests = upstream.state.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].1.get("model"),
        None,
        "the gateway's wildcard is not the worker's model: {}",
        requests[0].1
    );
}
