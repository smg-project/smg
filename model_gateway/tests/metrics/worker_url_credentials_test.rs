//! The `worker` label of a worker's metric series is the worker's rendering
//! of its URL (`Worker::redacted_url`): a worker registered with credentials
//! in its URL (`http://user:secret@host`) has them masked whole on every
//! series (`http://***@host`), the string its routed-worker response header
//! shows; a worker without credentials keeps its URL byte for byte. The
//! series are keyed by that label: a removal retires what the registration
//! activated, and two registrations at one address that differ only by
//! their credentials share one label, whose series stay with the live
//! registration until the last one leaves.
//!
//! The recorder is this binary's, installed once for all its tests (see
//! `recorder()` in the group's `mod.rs`).

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test failures should panic; the upstream tasks live for the test process"
)]

use std::{net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
    routing::post,
    Json, Router,
};
use openai_protocol::{completion::CompletionRequest, model_card::ModelCard};
use serde_json::json;
use smg::{
    middleware::TenantRequestMeta,
    observability::{metrics::Metrics, metrics_server::start_metrics_server},
    routers::{
        factory::router_ids, gateway::Gateway, http::router::Router as HttpRouter, RouterTrait,
    },
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerRegistry, WorkerType},
};
use tokio::net::TcpListener;
use tower::ServiceExt;

use crate::common::{
    grpc_harness::{
        build_grpc_router, grpc_router_config, health_checks_off, serve_canned_worker, Canned,
    },
    test_app::{create_test_app_context, create_test_app_with_context},
};

const MODEL: &str = "label-test-model";
const ROUTED_WORKER_ID: &str = "x-smg-routed-worker-id";

async fn scrape(addr: SocketAddr) -> String {
    smg::clients::http_client()
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("metrics endpoint reachable")
        .text()
        .await
        .expect("metrics body")
}

/// The sample lines of `body` whose worker is `label`.
fn worker_lines<'a>(body: &'a str, label: &str) -> Vec<&'a str> {
    let worker = format!("worker=\"{label}\"");
    body.lines().filter(|line| line.contains(&worker)).collect()
}

fn has_family(lines: &[&str], prefix: &str) -> bool {
    lines.iter().any(|line| line.starts_with(prefix))
}

/// The value of the worker's request counter for `MODEL`, if the series exists.
fn requests_total(body: &str, label: &str) -> Option<u64> {
    let worker = format!("worker=\"{label}\"");
    let model = format!("model=\"{MODEL}\"");
    body.lines()
        .find(|line| {
            line.starts_with("smg_worker_requests_total{")
                && line.contains(&worker)
                && line.contains(&model)
        })
        .map(|line| line.rsplit(' ').next().unwrap().parse().unwrap())
}

/// The value of the worker's gauge of `family`, if the series exists.
fn gauge(body: &str, family: &str, label: &str) -> Option<f64> {
    let prefix = format!("{family}{{");
    let worker = format!("worker=\"{label}\"");
    body.lines()
        .find(|line| line.starts_with(&prefix) && line.contains(&worker))
        .map(|line| line.rsplit(' ').next().unwrap().parse().unwrap())
}

/// The worker's `smg_worker_state` series: `(state, reason)` and the value,
/// one entry per series.
fn state_series(body: &str, label: &str) -> Vec<((String, String), f64)> {
    fn label_value(line: &str, name: &str) -> String {
        let key = format!("{name}=\"");
        let start = line.find(&key).unwrap() + key.len();
        line[start..].split('"').next().unwrap().to_owned()
    }
    let worker = format!("worker=\"{label}\"");
    body.lines()
        .filter(|line| line.starts_with("smg_worker_state{") && line.contains(&worker))
        .map(|line| {
            let value = line.rsplit(' ').next().unwrap().parse().unwrap();
            (
                (label_value(line, "state"), label_value(line, "reason")),
                value,
            )
        })
        .collect()
}

/// An HTTP upstream that answers every `/generate` with a completion; the
/// credentials the gateway sends for a URL with userinfo are accepted as any
/// header is.
async fn start_http_upstream() -> String {
    let app = Router::new().route(
        "/generate",
        post(|| async { Json(json!({"text": "ok", "meta_info": {"finish_reason": "stop"}})) }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    url
}

/// The production HTTP app with one regular worker at `upstream`.
async fn http_app(upstream: &str) -> Router {
    let ctx = create_test_app_context().await;
    ctx.worker_registry
        .register(Arc::new(
            BasicWorkerBuilder::new(upstream)
                .model(ModelCard::new(MODEL))
                .health_config(health_checks_off())
                .build(),
        ))
        .unwrap();
    let router = Arc::new(HttpRouter::new(&ctx).await.unwrap());
    let gateway = Arc::new(Gateway::new(ctx.worker_registry.clone()));
    gateway.register_router(router_ids::HTTP_REGULAR, router);
    create_test_app_with_context(gateway, ctx)
}

fn generate_request() -> Request {
    Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"text": "hello", "sampling_params": {"max_new_tokens": 2}}).to_string(),
        ))
        .unwrap()
}

/// An in-process mock gRPC engine that answers any prompt.
async fn start_grpc_upstream() -> String {
    let port = serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens: 2,
    })
    .await;
    format!("grpc://127.0.0.1:{port}")
}

/// The production gRPC router with one regular worker at `upstream`.
async fn grpc_router(upstream: &str) -> Box<dyn RouterTrait> {
    let config = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .build_unchecked();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(upstream)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(health_checks_off())
            .build(),
    );
    build_grpc_router(config, MODEL, vec![worker]).await
}

fn completion() -> CompletionRequest {
    serde_json::from_value(json!({
        "model": MODEL,
        "prompt": "Hello Hello Hello",
        "max_tokens": 2,
        "stream": false,
    }))
    .unwrap()
}

/// `upstream` with `user:secret@` in front of its host.
fn with_credentials(upstream: &str) -> String {
    upstream.replacen("://", "://user:secret@", 1)
}

/// What the gateway renders for `url`: the userinfo masked whole.
fn masked(url: &str) -> String {
    url.replacen("user:secret@", "***@", 1)
}

fn routed_worker_id(response: &axum::response::Response) -> String {
    response.headers()[ROUTED_WORKER_ID]
        .to_str()
        .unwrap()
        .to_owned()
}

/// A regular worker at `url` that starts routable and is never probed.
fn worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new(MODEL))
            .health_config(health_checks_off())
            .build(),
    )
}

#[tokio::test]
async fn a_credentialed_workers_series_carry_the_masked_label() {
    let handle = crate::metrics::recorder();
    let (metrics_addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");

    // HTTP: the counter's label is the header's string, and every series
    // that names the worker's address names it masked.
    let upstream = with_credentials(&start_http_upstream().await);
    let label = masked(&upstream);
    let app = http_app(&upstream).await;
    let response = app.clone().oneshot(generate_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(routed_worker_id(&response), label);
    let body = scrape(metrics_addr).await;
    assert_eq!(
        requests_total(&body, &label),
        Some(1),
        "the counter under the masked label:\n{body}"
    );
    assert_eq!(
        gauge(&body, "smg_worker_health", &label),
        Some(1.0),
        "{body}"
    );
    let address = upstream.rsplit('@').next().unwrap();
    let naming_the_address: Vec<&str> =
        body.lines().filter(|line| line.contains(address)).collect();
    assert!(
        naming_the_address.len() >= 5,
        "the worker's registration series:\n{body}"
    );
    assert!(
        naming_the_address
            .iter()
            .all(|line| line.contains(&format!("worker=\"{label}\""))),
        "a series names the address under another label:\n{body}"
    );

    // gRPC: the same on the request-execution stage.
    let grpc_upstream = with_credentials(&start_grpc_upstream().await);
    let grpc_label = masked(&grpc_upstream);
    let router = grpc_router(&grpc_upstream).await;
    let tenant = TenantRequestMeta::new(TenantKey::new("label-test-tenant"));
    let response = router
        .route_completion(None, &tenant, completion(), MODEL)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(routed_worker_id(&response), grpc_label);
    let body = scrape(metrics_addr).await;
    assert_eq!(
        requests_total(&body, &grpc_label),
        Some(1),
        "the counter under the masked label:\n{body}"
    );

    assert!(
        !body.contains("user:secret@"),
        "a series carries the secret:\n{body}"
    );
}

#[tokio::test]
async fn a_plain_workers_label_is_its_url_byte_for_byte() {
    let handle = crate::metrics::recorder();
    let (metrics_addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");
    let upstream = start_http_upstream().await;
    let app = http_app(&upstream).await;
    let response = app.clone().oneshot(generate_request()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(routed_worker_id(&response), upstream);
    let body = scrape(metrics_addr).await;
    assert_eq!(requests_total(&body, &upstream), Some(1), "{body}");
    assert_eq!(
        gauge(&body, "smg_worker_health", &upstream),
        Some(1.0),
        "{body}"
    );
}

/// The series registry activates and retires by the label: a removal takes
/// the masked label's series out of the scrape, a write that arrives late
/// is dropped, and a re-registration records again from fresh series.
#[tokio::test]
async fn a_removal_retires_the_masked_labels_series_and_a_re_registration_starts_fresh() {
    let handle = crate::metrics::recorder();
    let (metrics_addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");
    let registry = WorkerRegistry::new();
    let url = "http://user:secret@10.255.255.1:1";
    let label = "http://***@10.255.255.1:1";

    let worker_id = registry
        .register(worker(url))
        .expect("a fresh URL registers");
    Metrics::record_worker_cb_transition(label, "closed", "open");
    let body = scrape(metrics_addr).await;
    let lines = worker_lines(&body, label);
    for prefix in [
        "smg_worker_health{",
        "smg_worker_http2{",
        "smg_worker_cb_state{",
        "smg_worker_cb_transitions_total{",
        "smg_worker_state{",
        "smg_worker_requests_total{",
    ] {
        assert!(
            has_family(&lines, prefix),
            "{prefix} missing under the masked label:\n{body}"
        );
    }
    assert!(
        !body.contains("user:secret@"),
        "a series carries the secret:\n{body}"
    );

    assert!(registry.remove(&worker_id).is_some());
    // Writes that arrive after the removal: a health check that raced it,
    // the outcome of a request that was in flight.
    Metrics::set_worker_health(label, true);
    Metrics::record_worker_cb_outcome(label, "failure");
    let body = scrape(metrics_addr).await;
    assert!(
        worker_lines(&body, label).is_empty(),
        "series of the removed worker still in the scrape:\n{body}"
    );

    registry
        .register(worker(url))
        .expect("the retired URL registers again");
    let body = scrape(metrics_addr).await;
    let lines = worker_lines(&body, label);
    assert!(
        has_family(&lines, "smg_worker_health{") && has_family(&lines, "smg_worker_cb_state{"),
        "series missing after the re-registration:\n{body}"
    );
    assert!(
        !has_family(&lines, "smg_worker_cb_transitions_total{"),
        "the removed registration's counter came back:\n{body}"
    );
}

/// Two registrations at one address that differ only by their credentials
/// render to one label. A credential rotation registers the new URL and then
/// removes the old one: the label's series stay for the live registration
/// and read its state (one `smg_worker_state` pair at 1, the leaving
/// registration's demotion lowered again), and are retired with the last.
#[tokio::test]
async fn a_credential_rotation_at_one_address_keeps_the_labels_series() {
    let handle = crate::metrics::recorder();
    let (metrics_addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");
    let registry = WorkerRegistry::new();
    let old = "http://user:old@10.255.255.2:1";
    let new = "http://user:new@10.255.255.2:1";
    let label = "http://***@10.255.255.2:1";
    let old_id = registry
        .register(worker(old))
        .expect("the old URL registers");
    let new_id = registry
        .register(worker(new))
        .expect("the new URL registers");
    let body = scrape(metrics_addr).await;
    assert_eq!(
        gauge(&body, "smg_worker_health", label),
        Some(1.0),
        "{body}"
    );

    assert!(registry.remove(&old_id).is_some());
    let body = scrape(metrics_addr).await;
    assert_eq!(
        gauge(&body, "smg_worker_health", label),
        Some(1.0),
        "the live registration's series survive the other's removal:\n{body}"
    );
    let states = state_series(&body, label);
    let at_one: Vec<&(String, String)> = states
        .iter()
        .filter(|(_, value)| *value == 1.0)
        .map(|(pair, _)| pair)
        .collect();
    assert_eq!(
        at_one,
        vec![&("available".to_string(), "none".to_string())],
        "one state pair at 1 after the rotation, the live registration's:\n{body}"
    );
    assert_eq!(
        gauge(&body, "smg_worker_cb_state", label),
        Some(0.0),
        "{body}"
    );
    assert_eq!(
        gauge(&body, "smg_worker_requests_active", label),
        Some(0.0),
        "{body}"
    );
    assert!(
        !body.contains("user:old@") && !body.contains("user:new@"),
        "a series carries a credential:\n{body}"
    );

    assert!(registry.remove(&new_id).is_some());
    let body = scrape(metrics_addr).await;
    assert_eq!(
        gauge(&body, "smg_worker_health", label),
        None,
        "the last registration's removal retires the label:\n{body}"
    );
    assert!(
        state_series(&body, label).is_empty(),
        "the last registration's removal retires the state family:\n{body}"
    );
}
