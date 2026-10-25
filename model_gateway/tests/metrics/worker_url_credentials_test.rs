//! A worker registered with credentials in its URL (`http://user:secret@host`)
//! never shows the secret on a request's success path: the
//! `x-smg-routed-worker-id` response header, the request-sent event line and
//! the `worker` label of its series carry the URL with the secret masked
//! (`http://user:***@host`), on the HTTP and on the gRPC path. A worker
//! without credentials keeps its URL byte for byte on all three.
//!
//! The recorder is this binary's, installed once for all its tests (see
//! `recorder()` in the group's `mod.rs`), and so is the capture of the
//! request events (`event_lines`).

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test failures should panic; the upstream tasks live for the test process"
)]

use std::{
    io,
    net::SocketAddr,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

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
    observability::metrics_server::start_metrics_server,
    routers::{
        factory::router_ids, gateway::Gateway, http::router::Router as HttpRouter, RouterTrait,
    },
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType},
};
use tokio::net::TcpListener;
use tower::ServiceExt;
use tracing_subscriber::{
    filter::{LevelFilter, Targets},
    fmt::MakeWriter,
    layer::SubscriberExt,
    Layer,
};

use crate::common::{
    grpc_harness::{
        build_grpc_router, grpc_router_config, health_checks_off, serve_canned_worker, Canned,
    },
    test_app::{create_test_app_context, create_test_app_with_context},
};

const MODEL: &str = "credentials-test-model";
const ROUTED_WORKER_ID: &str = "x-smg-routed-worker-id";

/// The lines a subscriber wrote, read back as text.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Capture {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).expect("utf-8 lines")
    }
}

impl io::Write for Capture {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for Capture {
    type Writer = Capture;

    fn make_writer(&'a self) -> Capture {
        self.clone()
    }
}

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

/// The value of the worker's request counter in a scrape, if the series exists.
fn requests_total(body: &str, worker_label: &str) -> Option<u64> {
    let worker = format!("worker=\"{worker_label}\"");
    let model = format!("model=\"{MODEL}\"");
    body.lines()
        .find(|line| {
            line.starts_with("smg_worker_requests_total{")
                && line.contains(&worker)
                && line.contains(&model)
        })
        .map(|line| line.rsplit(' ').next().unwrap().parse().unwrap())
}

/// The value of a worker's gauge of `family` in a scrape, if the series exists.
fn gauge(body: &str, family: &str, worker_label: &str) -> Option<f64> {
    let prefix = format!("{family}{{");
    let worker = format!("worker=\"{worker_label}\"");
    body.lines()
        .find(|line| line.starts_with(&prefix) && line.contains(&worker))
        .map(|line| line.rsplit(' ').next().unwrap().parse().unwrap())
}

/// The value of the worker's health gauge in a scrape, if the series exists.
fn health(body: &str, worker_label: &str) -> Option<f64> {
    gauge(body, "smg_worker_health", worker_label)
}

/// The worker's `smg_worker_state` series in a scrape: `(state, reason)` and
/// the value, one entry per series.
fn state_series(body: &str, worker_label: &str) -> Vec<((String, String), f64)> {
    fn label(line: &str, name: &str) -> String {
        let key = format!("{name}=\"");
        let start = line.find(&key).unwrap() + key.len();
        line[start..].split('"').next().unwrap().to_owned()
    }
    let worker = format!("worker=\"{worker_label}\"");
    body.lines()
        .filter(|line| line.starts_with("smg_worker_state{") && line.contains(&worker))
        .map(|line| {
            let value = line.rsplit(' ').next().unwrap().parse().unwrap();
            ((label(line, "state"), label(line, "reason")), value)
        })
        .collect()
}

/// An HTTP upstream that answers every `/generate` with a completion; the
/// credentials the gateway's client sends for a URL with userinfo are
/// accepted as any header is.
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

/// What the gateway renders for `url`: the secret masked.
fn masked(url: &str) -> String {
    url.replacen("user:secret@", "user:***@", 1)
}

/// The request events of this process, captured once for every test of the
/// binary: a subscriber on the events module at DEBUG (the request-sent
/// event is a DEBUG line with no tracing exporter), installed as the global
/// default on the first call. A subscriber scoped to one request shares the
/// process-wide callsite-interest cache with the other tests of the binary
/// and missed the line in a parallel run.
fn event_lines() -> &'static Capture {
    static CAPTURE: OnceLock<Capture> = OnceLock::new();
    CAPTURE.get_or_init(|| {
        let capture = Capture::default();
        let layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(capture.clone())
            .with_filter(
                Targets::new().with_target("smg::observability::events", LevelFilter::DEBUG),
            );
        tracing::subscriber::set_global_default(tracing_subscriber::registry().with(layer))
            .expect("this binary installs its one tracing subscriber here");
        capture
    })
}

/// One request through the HTTP app, and the request-sent event line that
/// names `worker` (the URL as the gateway renders it), if one was written.
async fn http_request(app: &Router, worker: &str) -> (axum::response::Response, Option<String>) {
    let lines = event_lines();
    let response = app.clone().oneshot(generate_request()).await.unwrap();
    let sent = lines
        .text()
        .lines()
        .find(|line| line.contains("Sending request") && line.contains(worker))
        .map(str::to_owned);
    (response, sent)
}

fn routed_worker_id(response: &axum::response::Response) -> String {
    response.headers()[ROUTED_WORKER_ID]
        .to_str()
        .unwrap()
        .to_owned()
}

#[tokio::test]
async fn a_worker_urls_secret_is_masked_on_the_header_the_event_and_the_label() {
    let handle = crate::metrics::recorder();
    let (metrics_addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");

    // HTTP: the header, the request-sent event line and the counter's label.
    let upstream = with_credentials(&start_http_upstream().await);
    let app = http_app(&upstream).await;
    let (response, sent) = http_request(&app, &masked(&upstream)).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(routed_worker_id(&response), masked(&upstream));
    let sent = sent.unwrap_or_else(|| {
        panic!(
            "no request-sent event line names {}:\n{}",
            masked(&upstream),
            event_lines().text()
        )
    });
    assert!(!sent.contains("secret"), "the event line: {sent}");
    let body = scrape(metrics_addr).await;
    assert_eq!(
        requests_total(&body, &masked(&upstream)),
        Some(1),
        "the counter under the masked label:\n{body}"
    );

    // gRPC: the header on the response and the counter's label.
    let grpc_upstream = with_credentials(&start_grpc_upstream().await);
    let router = grpc_router(&grpc_upstream).await;
    let tenant = TenantRequestMeta::new(TenantKey::new("credentials-test-tenant"));
    let response = router
        .route_completion(None, &tenant, completion(), MODEL)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(routed_worker_id(&response), masked(&grpc_upstream));
    let body = scrape(metrics_addr).await;
    assert_eq!(
        requests_total(&body, &masked(&grpc_upstream)),
        Some(1),
        "the counter under the masked label:\n{body}"
    );

    // No series of either worker carries the secret.
    assert!(
        !body.contains("user:secret@"),
        "a series carries the secret:\n{body}"
    );
}

#[tokio::test]
async fn a_worker_url_without_credentials_is_rendered_byte_for_byte() {
    let handle = crate::metrics::recorder();
    let (metrics_addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");
    let upstream = start_http_upstream().await;
    let app = http_app(&upstream).await;
    let (response, sent) = http_request(&app, &upstream).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(routed_worker_id(&response), upstream);
    let sent = sent.unwrap_or_else(|| {
        panic!(
            "no request-sent event line names {upstream}:\n{}",
            event_lines().text()
        )
    });
    assert!(
        sent.contains(&format!("url={upstream}")),
        "the event line: {sent}"
    );
    let body = scrape(metrics_addr).await;
    assert_eq!(requests_total(&body, &upstream), Some(1), "{body}");
}

/// Two registrations at one address that differ only by their secret render
/// to one label. A credential rotation registers the new URL and then removes
/// the old one: the label's series stay for the live registration, read its
/// state (one `smg_worker_state` pair at 1, the leaving registration's
/// demotion lowered again) and are retired with the last one.
#[tokio::test]
async fn a_credential_rotation_at_one_address_keeps_the_labels_series() {
    let handle = crate::metrics::recorder();
    let (metrics_addr, _server) = start_metrics_server(handle, "127.0.0.1".to_string(), 0)
        .await
        .expect("metrics server binds an ephemeral port");
    let ctx = create_test_app_context().await;
    let worker = |url: &str| -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(MODEL))
                .health_config(health_checks_off())
                .build(),
        )
    };
    let old = "http://user:old@127.0.0.1:1";
    let new = "http://user:new@127.0.0.1:1";
    let label = "http://user:***@127.0.0.1:1";
    ctx.worker_registry.register(worker(old)).unwrap();
    ctx.worker_registry.register(worker(new)).unwrap();
    let body = scrape(metrics_addr).await;
    assert_eq!(health(&body, label), Some(1.0), "{body}");

    ctx.worker_registry.remove_by_url(old).unwrap();
    let body = scrape(metrics_addr).await;
    assert_eq!(
        health(&body, label),
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
        states.iter().map(|(_, value)| value).sum::<f64>(),
        1.0,
        "the state family is one-hot:\n{body}"
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

    ctx.worker_registry.remove_by_url(new).unwrap();
    let body = scrape(metrics_addr).await;
    assert_eq!(
        health(&body, label),
        None,
        "the last registration's removal retires the label:\n{body}"
    );
    assert!(
        state_series(&body, label).is_empty(),
        "the last registration's removal retires the state family:\n{body}"
    );
}
