//! A TokenSpeed engine SMG speaks gRPC to is driven through `/v1/rl` via its
//! HTTP control endpoint. The workers register through `POST /workers`, so the
//! endpoint and the capabilities come out of the engine's own server info the
//! way they do in production: discovery reports them, the proxy reaches the
//! endpoint with the worker's bearer, a fan-out over a mixed HTTP+gRPC fleet
//! hits each worker once, a gRPC worker whose engine advertises nothing is
//! named in `failed[]`, and the gRPC `/generate` path reports the version the
//! engine stamped on the response.
//!
//! Each test builds its own fleet on its own pair of mock HTTP ports: the HTTP
//! mock binds the port it is configured with, and the tests in one binary run
//! concurrently, so a shared pair would race for the same listener.

#[path = "common/mod.rs"]
mod common;

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use common::{
    mock_worker::{
        set_request_recorder, HealthStatus, MockWorker, MockWorkerConfig, RequestRecorder,
        WorkerType as MockWorkerType,
    },
    test_app::create_test_app_with_context,
};
use http_body_util::BodyExt;
use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::generate::GenerateRequest;
use serde_json::{json, Value};
use smg::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
};
use tokio::net::TcpListener;
use tower::ServiceExt;

const MODEL: &str = "rl-ts-test-model";
/// What the mock engine stamps on every generate response it produces.
const ENGINE_VERSION: &str = "v7";
/// What the workers carry as their registration-time `weight_version` label,
/// so a response that reports `ENGINE_VERSION` can only have come from the
/// engine rather than from the label.
const REGISTERED_VERSION: &str = "registered";
/// The bearer the engine's control app expects, registered as the worker's
/// `api_key`.
const CONTROL_KEY: &str = "ts-secret";

/// What a current TokenSpeed engine puts in its server info for the RL
/// control plane, with its control app at `control_url`.
fn advertisement(control_url: &str) -> BTreeMap<String, String> {
    [
        ("rl.control_url", control_url),
        ("rl.pause_modes", "wait,abort,keep"),
        ("rl.update_from", "distributed,mooncake"),
        ("rl.abort", "true"),
        ("rl.flush_cache", "true"),
        ("rl.sleep_wake", "true"),
        ("rl.reports_weight_version", "true"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// A canned mock TokenSpeed gRPC engine that advertises `server_args` and
/// stamps `ENGINE_VERSION` on every generate response.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_mock_grpc_engine(server_args: BTreeMap<String, String>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock gRPC engine");
    let port = listener
        .local_addr()
        .expect("mock gRPC engine address")
        .port();
    let cfg = Arc::new(mock_worker::config::Config {
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: 3,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
        server_args,
        weight_version: Some(ENGINE_VERSION.to_string()),
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    port
}

/// A gRPC-transport regular router context with the RL control plane mounted
/// and a tokenizer preloaded for `MODEL`: the gRPC router needs one, and the
/// mock engine carries no tokenizer artifacts of its own, so autoload stays
/// off.
#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn grpc_rl_context() -> Arc<AppContext> {
    // Load the tokenizer before building the config so no `RouterConfig`
    // (a large struct) is held across an await point in this future.
    let registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
    registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();

    let mut config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection()
        .round_robin_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    config.disable_tokenizer_autoload = true;
    config.rl.enabled = true;
    config.rl.control_timeout_secs = 5;

    common::create_test_context_with_tokenizer_registry(config, registry).await
}

fn mock_http(port: u16) -> MockWorkerConfig {
    MockWorkerConfig {
        port,
        worker_type: MockWorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn json_of(resp: axum::response::Response) -> Value {
    serde_json::from_slice(&resp.into_body().collect().await.unwrap().to_bytes())
        .unwrap_or(Value::Null)
}

/// Register `spec` through the worker API, as an operator or a launcher does.
#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn register(app: &axum::Router, spec: Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::post("/workers")
                .header("content-type", "application/json")
                .body(Body::from(spec.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED, "{spec}");
}

/// The discovery rows once `n` workers are registered and ready. Registration
/// runs in the background, so poll.
#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn wait_for_workers(app: &axum::Router, n: usize) -> Vec<Value> {
    let mut workers = Vec::new();
    for _ in 0..100 {
        let resp = app
            .clone()
            .oneshot(Request::get("/v1/rl/workers").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = json_of(resp).await;
        workers = body["workers"].as_array().cloned().unwrap_or_default();
        if workers.len() == n && workers.iter().all(|w| w["health"] == "ready") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(workers.len(), n, "{workers:?}");
    workers
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("rl-ts-test-tenant"))
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn generate_request(stream: bool) -> GenerateRequest {
    serde_json::from_value(json!({
        "model": MODEL,
        "input_ids": [1, 2, 3],
        "sampling_params": {"max_new_tokens": 3},
        "stream": stream,
    }))
    .unwrap()
}

/// The gRPC `/generate` path answers with one element per sample; `n` is
/// unset here, so the single element is the whole answer.
fn first_generate_result(body: Value) -> Value {
    match body {
        Value::Array(items) => items.into_iter().next().unwrap_or(Value::Null),
        other => other,
    }
}

/// Two TokenSpeed gRPC workers (one whose engine advertises a control app at
/// an HTTP mock that records what it receives, one whose engine advertises
/// nothing) plus one HTTP SGLang mock, all registered through `POST /workers`.
struct Fleet {
    ctx: Arc<AppContext>,
    app: axum::Router,
    router: Arc<dyn RouterTrait>,
    /// The discovery rows, in the order `GET /v1/rl/workers` listed them.
    workers: Vec<Value>,
    control_recorder: Arc<RequestRecorder>,
    sglang_recorder: Arc<RequestRecorder>,
    control_url: String,
    _control_app: MockWorker,
    _sglang: MockWorker,
}

#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn fleet(control_port: u16, sglang_port: u16) -> Fleet {
    let control_recorder = RequestRecorder::new();
    set_request_recorder(control_port, control_recorder.clone());
    let mut control_app = MockWorker::new(mock_http(control_port));
    let control_url = control_app.start().await.unwrap();
    let sglang_recorder = RequestRecorder::new();
    set_request_recorder(sglang_port, sglang_recorder.clone());
    let mut sglang = MockWorker::new(mock_http(sglang_port));
    let sglang_url = sglang.start().await.unwrap();

    let ctx = grpc_rl_context().await;
    let router: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_router(&ctx)
            .await
            .expect("gRPC RL router should build"),
    );
    let app = create_test_app_with_context(Arc::clone(&router), Arc::clone(&ctx));

    let advertising = start_mock_grpc_engine(advertisement(&control_url)).await;
    let silent = start_mock_grpc_engine(BTreeMap::new()).await;
    register(
        &app,
        json!({
            "url": format!("grpc://127.0.0.1:{advertising}"),
            "runtime_type": "tokenspeed",
            "api_key": CONTROL_KEY,
            "labels": {"weight_version": REGISTERED_VERSION},
        }),
    )
    .await;
    register(
        &app,
        json!({
            "url": format!("grpc://127.0.0.1:{silent}"),
            "runtime_type": "tokenspeed",
            "labels": {"weight_version": REGISTERED_VERSION},
        }),
    )
    .await;
    register(&app, json!({"url": sglang_url})).await;
    let workers = wait_for_workers(&app, 3).await;

    Fleet {
        ctx,
        app,
        router,
        workers,
        control_recorder,
        sglang_recorder,
        control_url,
        _control_app: control_app,
        _sglang: sglang,
    }
}

impl Fleet {
    /// The one discovery row for `engine` whose `control_url` is (or is not)
    /// set.
    #[expect(
        clippy::expect_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    fn by_engine(&self, engine: &str, has_control: bool) -> &Value {
        self.workers
            .iter()
            .find(|w| w["engine"] == engine && w["control_url"].is_string() == has_control)
            .expect("worker present")
    }

    #[expect(
        clippy::expect_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    fn id(&self, engine: &str, has_control: bool) -> String {
        self.by_engine(engine, has_control)["id"]
            .as_str()
            .expect("worker id")
            .to_string()
    }
}

#[tokio::test]
async fn discovery_reports_the_advertised_endpoint_and_capabilities() {
    let f = fleet(18921, 18922).await;

    let ts = f.by_engine("tokenspeed", true);
    assert_eq!(ts["connection_mode"], "grpc");
    assert_eq!(ts["control_url"], f.control_url);
    assert_eq!(ts["capabilities"]["source"], "label");
    assert_eq!(
        ts["capabilities"]["pause_modes"],
        json!(["wait", "abort", "keep"])
    );
    assert_eq!(
        ts["capabilities"]["update_from"],
        json!(["distributed", "mooncake"])
    );
    assert_eq!(
        ts["weight_version"], REGISTERED_VERSION,
        "the registration label wins over the mock's model-info placeholder"
    );

    let silent = f.by_engine("tokenspeed", false);
    assert_eq!(silent["control_url"], Value::Null);
    assert_eq!(
        silent["capabilities"]["source"], "static",
        "an engine that advertises nothing gets the built-in TokenSpeed row"
    );
    assert_eq!(
        silent["capabilities"]["update_from"],
        json!(["distributed"])
    );

    let sglang = f.by_engine("sglang", true);
    assert_eq!(
        sglang["control_url"], sglang["base_url"],
        "HTTP workers control themselves"
    );
}

#[tokio::test]
async fn proxy_reaches_the_control_endpoint_with_the_worker_bearer() {
    let f = fleet(18923, 18924).await;
    let id = f.id("tokenspeed", true);

    let resp = f
        .app
        .clone()
        .oneshot(
            Request::post(format!("/v1/rl/workers/{id}/engine/pause_generation"))
                .header("content-type", "application/json")
                .header("authorization", "Bearer caller-token")
                .body(Body::from(r#"{"mode":"keep"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(f.control_recorder.only_body(), json!({"mode": "keep"}));
    assert_eq!(
        f.control_recorder.authorizations(),
        vec![Some(format!("Bearer {CONTROL_KEY}"))],
        "the worker's own key, not the caller's"
    );
}

#[tokio::test]
async fn fanout_over_workers_with_endpoints_is_200_and_hits_each_once() {
    let f = fleet(18925, 18926).await;
    let ts = f.id("tokenspeed", true);
    let sglang = f.id("sglang", true);

    let resp = f
        .app
        .clone()
        .oneshot(
            Request::post(format!(
                "/v1/rl/engine/pause_generation?selector=id%20in%20({ts},{sglang})"
            ))
            .header("content-type", "application/json")
            .body(Body::from(r#"{"mode":"abort"}"#))
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_of(resp).await;
    assert_eq!(body["total"], 2);
    assert_eq!(body["succeeded"], 2);
    assert_eq!(
        f.control_recorder.only_body(),
        json!({"mode": "abort"}),
        "the gRPC worker's control app, exactly once"
    );
    assert_eq!(
        f.sglang_recorder.only_body(),
        json!({"mode": "abort"}),
        "the HTTP worker itself, exactly once"
    );
}

#[tokio::test]
async fn fanout_names_the_worker_without_an_endpoint() {
    let f = fleet(18927, 18928).await;
    let resp = f
        .app
        .clone()
        .oneshot(
            Request::post("/v1/rl/engine/flush_cache?selector=worker_type%3Dregular")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::MULTI_STATUS);
    let body = json_of(resp).await;
    assert_eq!(body["total"], 3);
    assert_eq!(
        body["succeeded"], 2,
        "the HTTP SGLang mock and the TokenSpeed mock with an endpoint"
    );
    let failed = body["failed"].as_array().unwrap();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0]["error"], "no_control_endpoint");
    assert_eq!(failed[0]["connection_mode"], "grpc");
    assert_eq!(failed[0]["worker_id"], f.id("tokenspeed", false));

    // Control calls never touch breakers or load accounting.
    for worker in f.ctx.worker_registry.get_all() {
        assert!(worker.circuit_breaker_can_execute(), "{}", worker.url());
        assert_eq!(worker.load(), 0, "{}", worker.url());
    }
}

#[tokio::test]
async fn grpc_generate_reports_the_engine_stamped_version() {
    let f = fleet(18929, 18930).await;

    let resp = f
        .router
        .route_generate(None, &tenant(), generate_request(false), MODEL)
        .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let first = first_generate_result(json_of(resp).await);
    assert_eq!(
        first["meta_info"]["weight_version"], ENGINE_VERSION,
        "engine value beats the `{REGISTERED_VERSION}` label"
    );

    let resp = f
        .router
        .route_generate(None, &tenant(), generate_request(true), MODEL)
        .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let text = String::from_utf8(
        resp.into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(
        text.contains(&format!(r#""weight_version":"{ENGINE_VERSION}""#)),
        "{text}"
    );
}
