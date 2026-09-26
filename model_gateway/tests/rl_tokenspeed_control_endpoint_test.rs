//! A TokenSpeed engine SMG speaks gRPC to is driven through `/v1/rl` via its
//! HTTP control endpoint: discovery reports it, the proxy reaches it with the
//! worker's bearer, a mixed HTTP+gRPC fleet fans out, a gRPC worker without an
//! endpoint is named in `failed[]`, and the gRPC `/generate` path reports the
//! version the engine stamped on the response.
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
use openai_protocol::{
    generate::GenerateRequest, model_card::ModelCard, worker::HealthCheckConfig,
};
use serde_json::{json, Value};
use smg::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType, UNKNOWN_MODEL_ID},
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
/// and a tokenizer preloaded for `MODEL` (the gRPC router needs one, and the
/// mock engine carries no tokenizer artifacts of its own).
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
    config.rl.enabled = true;
    config.rl.control_timeout_secs = 5;

    common::create_test_context_with_tokenizer_registry(config, registry).await
}

fn health_off() -> HealthCheckConfig {
    HealthCheckConfig {
        disable_health_check: true,
        ..Default::default()
    }
}

/// Register a TokenSpeed gRPC worker whose RL capabilities come from labels.
/// `control_url` sets the `rl.control_url` label an engine would advertise;
/// `api_key` is the bearer the gateway presents to that control app.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn register_tokenspeed(
    ctx: &Arc<AppContext>,
    grpc_port: u16,
    control_url: Option<&str>,
    api_key: Option<&str>,
) {
    let mut builder = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{grpc_port}"))
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::TokenSpeed)
        .model(ModelCard::new(MODEL))
        .health_config(health_off())
        .label("weight_version", REGISTERED_VERSION)
        .label("rl.pause_modes", "wait,abort,keep")
        .label("rl.update_from", "disk,distributed")
        .label("rl.abort", "true")
        .label("rl.flush_cache", "true")
        .label("rl.sleep_wake", "true")
        .label("rl.reports_weight_version", "true");
    if let Some(url) = control_url {
        builder = builder.label("rl.control_url", url);
    }
    if let Some(key) = api_key {
        builder = builder.api_key(key);
    }
    ctx.worker_registry
        .register(Arc::new(builder.build()))
        .expect("TokenSpeed worker registered");
}

/// Register a TokenSpeed gRPC worker with no model card and no `model_id`
/// label. `Worker::model_id`'s fallback then reports [`UNKNOWN_MODEL_ID`],
/// so the registry's model index carries an extra `"unknown"` entry
/// alongside any real model -- the shape the model-less `/generate` default
/// must see through.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn register_untagged_tokenspeed(ctx: &Arc<AppContext>, grpc_port: u16) {
    let worker = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{grpc_port}"))
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::TokenSpeed)
        .health_config(health_off())
        .build();
    ctx.worker_registry
        .register(Arc::new(worker))
        .expect("untagged TokenSpeed worker registered");
}

/// Register an HTTP SGLang worker, which controls itself over its own URL.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn register_http_sglang(ctx: &Arc<AppContext>, url: &str) {
    let worker = BasicWorkerBuilder::new(url)
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Http)
        .runtime_type(RuntimeType::Sglang)
        .model(ModelCard::new(MODEL))
        .health_config(health_off())
        .build();
    ctx.worker_registry
        .register(Arc::new(worker))
        .expect("SGLang worker registered");
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

/// Two TokenSpeed gRPC workers (one with a control endpoint at an HTTP mock
/// that records what it receives, one without) plus one HTTP SGLang mock.
struct Fleet {
    ctx: Arc<AppContext>,
    app: axum::Router,
    router: Arc<dyn RouterTrait>,
    control_recorder: Arc<RequestRecorder>,
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
    let mut sglang = MockWorker::new(mock_http(sglang_port));
    let sglang_url = sglang.start().await.unwrap();

    let ctx = grpc_rl_context().await;
    let advertised = BTreeMap::from([("rl.control_url".to_string(), control_url.clone())]);
    let with_endpoint = start_mock_grpc_engine(advertised).await;
    let without_endpoint = start_mock_grpc_engine(BTreeMap::new()).await;
    register_tokenspeed(
        &ctx,
        with_endpoint,
        Some(control_url.as_str()),
        Some("ts-secret"),
    );
    register_tokenspeed(&ctx, without_endpoint, None, None);
    register_http_sglang(&ctx, &sglang_url);

    let router: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_router(&ctx)
            .await
            .expect("gRPC RL router should build"),
    );
    let app = create_test_app_with_context(Arc::clone(&router), Arc::clone(&ctx));
    Fleet {
        ctx,
        app,
        router,
        control_recorder,
        control_url,
        _control_app: control_app,
        _sglang: sglang,
    }
}

/// The one discovery row for `engine` whose `control_url` is (or is not) set.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn by_engine<'a>(workers: &'a [Value], engine: &str, has_control: bool) -> &'a Value {
    workers
        .iter()
        .find(|w| w["engine"] == engine && w["control_url"].is_string() == has_control)
        .expect("worker present")
}

#[tokio::test]
async fn discovery_reports_control_endpoint_and_advertised_capabilities() {
    let f = fleet(18921, 18922).await;
    let resp = f
        .app
        .clone()
        .oneshot(Request::get("/v1/rl/workers").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_of(resp).await;
    assert_eq!(body["total"], 3);
    let workers = body["workers"].as_array().unwrap();

    let ts = by_engine(workers, "tokenspeed", true);
    assert_eq!(ts["connection_mode"], "grpc");
    assert_eq!(ts["control_url"], f.control_url);
    assert_eq!(ts["capabilities"]["source"], "label");
    assert_eq!(
        ts["capabilities"]["pause_modes"],
        json!(["wait", "abort", "keep"])
    );

    let bare = by_engine(workers, "tokenspeed", false);
    assert_eq!(bare["control_url"], Value::Null);
    assert_eq!(bare["capabilities"]["source"], "label");

    let sglang = by_engine(workers, "sglang", true);
    assert_eq!(
        sglang["control_url"], sglang["base_url"],
        "HTTP workers control themselves"
    );
}

#[tokio::test]
async fn proxy_reaches_the_control_endpoint_with_the_worker_bearer() {
    let f = fleet(18923, 18924).await;
    let workers = json_of(
        f.app
            .clone()
            .oneshot(Request::get("/v1/rl/workers").body(Body::empty()).unwrap())
            .await
            .unwrap(),
    )
    .await;
    let id = by_engine(workers["workers"].as_array().unwrap(), "tokenspeed", true)["id"]
        .as_str()
        .unwrap()
        .to_string();

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
        vec![Some("Bearer ts-secret".to_string())],
        "the worker's own key, not the caller's"
    );
}

#[tokio::test]
async fn fanout_spans_transports_and_names_the_worker_without_an_endpoint() {
    let f = fleet(18925, 18926).await;
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

    // Control calls never touch breakers or load accounting.
    for worker in f.ctx.worker_registry.get_all() {
        assert!(worker.circuit_breaker_can_execute(), "{}", worker.url());
        assert_eq!(worker.load(), 0, "{}", worker.url());
    }
}

#[tokio::test]
async fn grpc_generate_reports_the_engine_stamped_version() {
    let f = fleet(18927, 18928).await;

    let resp = f
        .router
        .route_generate(None, &tenant(), generate_request(false), MODEL)
        .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_of(resp).await;
    assert!(body.is_object(), "single prompt -> object, got {body}");
    assert_eq!(
        body["meta_info"]["weight_version"], ENGINE_VERSION,
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

/// slime's rollout client sends one prompt per `/generate` call and indexes
/// `meta_info` straight off the response; SGLang answers that shape with a
/// single JSON object rather than a one-element list, and the gRPC path
/// must match.
#[tokio::test]
async fn single_prompt_generate_answers_with_an_object_like_sglang() {
    let f = fleet(18929, 18930).await;
    let resp = f
        .router
        .route_generate(None, &tenant(), generate_request(false), MODEL)
        .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_of(resp).await;
    assert!(body.is_object(), "single prompt -> object, got {body}");
    assert_eq!(body["meta_info"]["weight_version"], ENGINE_VERSION);
}

/// slime never sends `model` on `/generate`; when the fleet serves exactly
/// one model (as this fixture does), the gRPC router defaults the wildcard
/// placeholder to it instead of 404ing.
#[tokio::test]
async fn model_less_generate_defaults_to_the_single_served_model() {
    let f = fleet(18931, 18932).await;
    let request: GenerateRequest = serde_json::from_value(json!({
        "input_ids": [1, 2, 3],
        "sampling_params": {"max_new_tokens": 3},
    }))
    .unwrap();
    assert_eq!(request.model, UNKNOWN_MODEL_ID, "no `model` in the body");

    let resp = f
        .router
        .route_generate(None, &tenant(), request, UNKNOWN_MODEL_ID)
        .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_of(resp).await;
    assert!(body.is_object(), "single prompt -> object, got {body}");
    assert_eq!(body["meta_info"]["weight_version"], ENGINE_VERSION);
}

/// An untagged worker (no model card, no `model_id` label) registers under
/// the wildcard itself and must not defeat the single-model default: the
/// registry then carries one real model plus the wildcard's own entry, and
/// the default has to see through the latter.
#[tokio::test]
async fn model_less_generate_defaults_when_an_untagged_worker_is_also_registered() {
    let ctx = grpc_rl_context().await;
    let tagged_port = start_mock_grpc_engine(BTreeMap::new()).await;
    let untagged_port = start_mock_grpc_engine(BTreeMap::new()).await;
    register_tokenspeed(&ctx, tagged_port, None, None);
    register_untagged_tokenspeed(&ctx, untagged_port);
    let mut served = ctx.worker_registry.get_models();
    served.sort();
    assert_eq!(
        served,
        vec![MODEL.to_string(), UNKNOWN_MODEL_ID.to_string()],
        "the untagged worker registers under the wildcard placeholder itself, \
         which the model-less default must filter out"
    );

    let router: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_router(&ctx)
            .await
            .expect("gRPC RL router should build"),
    );
    let request: GenerateRequest = serde_json::from_value(json!({
        "input_ids": [1, 2, 3],
        "sampling_params": {"max_new_tokens": 3},
    }))
    .unwrap();

    let resp = router
        .route_generate(None, &tenant(), request, UNKNOWN_MODEL_ID)
        .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_of(resp).await;
    assert!(body.is_object(), "single prompt -> object, got {body}");
    assert_eq!(body["meta_info"]["weight_version"], ENGINE_VERSION);
}
