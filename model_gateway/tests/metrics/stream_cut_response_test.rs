//! A stream the engine cut mid-way is counted as what the client read. The
//! gateway ends such a body with the error event the clients know and then
//! `[DONE]`, so by its frames alone the body went out in full; the response
//! counter used to file it under the clean 200 of a stream that completed.
//! Here an in-process realistic mock engine with its admin API arms the cut
//! (a status after three tokens), the gateway's full HTTP stack serves a
//! stream over it on each route the count covers (chat and text completions,
//! Messages, the native generate route), and this binary's recorder is read
//! before and after: the cut stream lands on the head's 200 with
//! `stream_error`, the stream that completes on the clean row.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, Request},
};
use http_body_util::BodyExt;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use serde_json::{json, Value};
use smg::{
    health::ProbeState,
    middleware::AuthConfig,
    routers::RouterFactory,
    server::{build_app, AppState},
    worker::{RuntimeType, Worker, WorkerType},
};
use smg_test_harness::{
    grpc_harness::{free_port, grpc_router_config, grpc_test_context_with_tokenizer, grpc_worker},
    scripted_tokenizer::ScriptedTokenizer,
};
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle};
use tower::ServiceExt;

const MODEL: &str = "stream-cut-count-test-model";
/// The streamed routes the count covers, each with a request of its own: the
/// chat and text completions, the Messages route and the native generate
/// route all end a cut stream with an error event (the first three then
/// `[DONE]`; Messages ends on its `error` event). The last flag says whether
/// this test alone writes the route's clean `200` row in this binary: the
/// worker request counter test sends non-streamed `/generate` requests
/// through the same recorder, so on that route only the `stream_error` row,
/// which no other test writes, is read.
fn routes() -> Vec<(&'static str, Value, bool, bool)> {
    let prompt = "count to twenty";
    vec![
        (
            "/v1/chat/completions",
            json!({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, "stream": true}),
            true,
            true,
        ),
        (
            "/v1/completions",
            json!({"model": MODEL, "prompt": prompt, "max_tokens": 16, "stream": true}),
            true,
            true,
        ),
        (
            "/v1/messages",
            json!({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, "stream": true}),
            false,
            true,
        ),
        (
            "/generate",
            json!({"model": MODEL, "text": prompt, "sampling_params": {"max_new_tokens": 16}, "stream": true}),
            true,
            false,
        ),
    ]
}

/// An in-process realistic mock engine with its admin API, both stopped
/// when the test ends.
struct MockEngine {
    port: u16,
    admin_port: u16,
    name: String,
    client: reqwest::Client,
    stop: Option<(oneshot::Sender<()>, oneshot::Sender<()>)>,
    served: Option<(JoinHandle<()>, JoinHandle<()>)>,
}

impl MockEngine {
    #[expect(
        clippy::expect_used,
        clippy::disallowed_methods,
        reason = "test helper: the serve tasks end with their stop signals, which the fixture \
                  sends when it is dropped; a fixture that does not come up fails the test"
    )]
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the mock engine");
        let port = listener.local_addr().expect("mock engine address").port();
        let admin_port = free_port().await;
        let cfg = Arc::new(mock_worker::config::Config {
            admin_port: None,
            context_length: 32768,
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
            output_tokens: 16,
            realistic: true,
            step_reports: true,
            engine: mock_worker::engine::EngineParams::default(),
            loads_like: mock_worker::engine::LoadsLike::Vllm,
            ..mock_worker::config::Config::default()
        });
        let (stop, stopped) = oneshot::channel();
        let served = tokio::spawn(mock_worker::grpc::serve_with_listener_until(
            Arc::clone(&cfg),
            listener,
            async move {
                let _ = stopped.await;
            },
        ));
        let (stop_admin, admin_stopped) = oneshot::channel();
        let admin = tokio::spawn(mock_worker::admin::serve_until(
            cfg,
            "127.0.0.1".to_string(),
            admin_port,
            async move {
                let _ = admin_stopped.await;
            },
        ));
        let client = smg::server::http_client_builder()
            .no_proxy()
            .build()
            .expect("test client");
        let engine = Self {
            port,
            admin_port,
            name: format!("grpc:{port}"),
            client,
            stop: Some((stop, stop_admin)),
            served: Some((served, admin)),
        };
        let health = format!("http://127.0.0.1:{admin_port}/admin/health");
        let deadline = Instant::now() + Duration::from_secs(5);
        while engine.client.get(&health).send().await.is_err() {
            assert!(Instant::now() < deadline, "the admin API did not come up");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        engine
    }

    /// Arm a fault hook of the engine through its admin API.
    #[expect(
        clippy::expect_used,
        reason = "test helper: an admin API that does not answer fails the test"
    )]
    async fn fault(&self, hook: &str) {
        let url = format!(
            "http://127.0.0.1:{}/admin/fault/{}/{hook}",
            self.admin_port, self.name
        );
        let response = self
            .client
            .post(&url)
            .send()
            .await
            .expect("admin API reachable");
        assert!(
            response.status().is_success(),
            "{url}: {}",
            response.status()
        );
    }

    async fn stop(mut self) {
        if let Some((stop, stop_admin)) = self.stop.take() {
            let _ = stop.send(());
            let _ = stop_admin.send(());
        }
        if let Some((served, admin)) = self.served.take() {
            let _ = served.await;
            let _ = admin.await;
        }
    }
}

/// The gateway's HTTP app (every layer of the edge, the metrics layer at
/// its outside) over a gRPC regular router with the one worker at `port`.
#[expect(
    clippy::expect_used,
    reason = "test helper: an app that does not build fails the fixture"
)]
async fn gateway_app(port: u16) -> axum::Router {
    let config = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .disable_retries()
        .build_unchecked();
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(ScriptedTokenizer::from_chunks(
        (0..32).map(|i| format!("w{i} ")).collect(),
    ));
    let app_context = grpc_test_context_with_tokenizer(config, MODEL, tokenizer).await;
    let worker: Arc<dyn Worker> = Arc::new(
        grpc_worker(
            port,
            WorkerType::Regular,
            RuntimeType::TokenSpeed,
            ModelCard::new(MODEL),
        )
        .build(),
    );
    worker.set_status(WorkerStatus::Ready);
    app_context
        .worker_registry
        .register(Arc::clone(&worker))
        .expect("register the worker");
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("gRPC router");
    let app_state = Arc::new(AppState {
        router: Arc::from(router),
        probe_state: ProbeState::new(app_context.inflight_tracker.clone()),
        context: app_context,
        admission_queue: None,
        gateway: None,
        mesh_handler: None,
        mesh_adapters: None,
        secret_sources: Default::default(),
        secrets: None,
    });
    build_app(
        app_state,
        AuthConfig::new(None),
        AuthConfig::new(None),
        None,
        256 * 1024 * 1024,
        vec!["x-request-id".to_string()],
        vec![],
    )
    .expect("the gateway app")
}

/// One streamed request of `body` to `path` through `app`, read to its end:
/// the status and the body's text.
#[expect(clippy::expect_used, reason = "test helper")]
async fn streamed(app: &axum::Router, path: &str, body: &Value) -> (u16, String) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let response = app.clone().oneshot(request).await.expect("response");
    let status = response.status().as_u16();
    let bytes = tokio::time::timeout(Duration::from_secs(30), response.into_body().collect())
        .await
        .expect("the stream ends within its budget")
        .expect("the body")
        .to_bytes();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// The value of the label `name` on a rendered sample line.
fn label<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let key = format!("{name}=\"");
    let start = line.find(&key)? + key.len();
    let end = start + line[start..].find('"')?;
    Some(&line[start..end])
}

/// The `smg_http_responses_total` samples of `path` in this binary's
/// recorder, as `(status_code, error_code, count)`, sorted.
fn responses_of_path(path: &str) -> Vec<(String, String, u64)> {
    let rendered = crate::metrics::recorder().render();
    let mut samples: Vec<(String, String, u64)> = rendered
        .lines()
        .filter(|line| {
            line.starts_with("smg_http_responses_total{") && label(line, "path") == Some(path)
        })
        .filter_map(|line| {
            let count = line.rsplit(' ').next()?.parse().ok()?;
            Some((
                label(line, "status_code").unwrap_or_default().to_owned(),
                label(line, "error_code").unwrap_or_default().to_owned(),
                count,
            ))
        })
        .collect();
    samples.sort();
    samples
}

/// The count of the `(status_code, error_code)` row among `samples`.
fn count(samples: &[(String, String, u64)], status_code: &str, error_code: &str) -> u64 {
    samples
        .iter()
        .find(|(status, code, _)| status == status_code && code == error_code)
        .map_or(0, |(_, _, count)| *count)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_the_engine_cut_counts_as_a_body_that_failed() {
    let engine = MockEngine::start().await;
    let app = gateway_app(engine.port).await;

    for (path, body, ends_with_done, clean_row_is_ours) in routes() {
        let before = responses_of_path(path);

        // The engine cuts the next stream after three tokens.
        engine.fault("fail?status=503&after_tokens=3&count=1").await;
        let (status, text) = streamed(&app, path, &body).await;
        assert_eq!(
            status, 200,
            "{path}: the head was out before the cut: {text}"
        );
        assert!(
            text.contains("Stream error"),
            "{path}: the body tells the client of the cut: {text}"
        );
        if ends_with_done {
            assert!(
                text.trim_end().ends_with("data: [DONE]"),
                "{path}: the body ends as the clients know it: {text}"
            );
        }
        let after_cut = responses_of_path(path);
        assert_eq!(
            count(&after_cut, "200", "stream_error") - count(&before, "200", "stream_error"),
            1,
            "{path}: the cut stream is counted as a body that failed: {after_cut:?}"
        );
        if clean_row_is_ours {
            assert_eq!(
                count(&after_cut, "200", "") - count(&before, "200", ""),
                0,
                "{path}: and not as a stream that completed: {after_cut:?}"
            );
        }

        // The fault is spent: the next stream completes and counts on the clean row.
        let (status, text) = streamed(&app, path, &body).await;
        assert_eq!(status, 200, "{path}: {text}");
        assert!(
            !text.contains("Stream error"),
            "{path}: a complete stream: {text}"
        );
        let after_clean = responses_of_path(path);
        if clean_row_is_ours {
            assert_eq!(
                count(&after_clean, "200", "") - count(&after_cut, "200", ""),
                1,
                "{path}: the complete stream is counted as one: {after_clean:?}"
            );
        }
        assert_eq!(
            count(&after_clean, "200", "stream_error") - count(&after_cut, "200", "stream_error"),
            0,
            "{path}: and not as a body that failed: {after_clean:?}"
        );
    }

    engine.stop().await;
}
