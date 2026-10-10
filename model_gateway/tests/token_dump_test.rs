//! End-to-end check of the token dump against in-process mock TokenSpeed
//! workers: what the gateway records for each engine call matches what the
//! mock received and sent, for one worker and for a PD pair, and calls that
//! are dropped or refused are recorded as such.

#[path = "common/mod.rs"]
mod common;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use llm_tokenizer::{
    chat_template::ChatTemplateParams,
    traits::{ChatTemplateOutput, Tokenizer},
    Decoder, Encoder, Encoding, MockTokenizer, SpecialTokens, TokenizerRegistry,
};
use openai_protocol::{
    completion::CompletionRequest, model_card::ModelCard, worker::HealthCheckConfig,
};
use serde_json::{json, Value};
use smg::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    observability::token_dump::StartRequest,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::net::TcpListener;

const MODEL: &str = "token-dump-test-model";
/// Tokens the canned mock emits per request.
const OUTPUT_TOKENS: u32 = 3;

/// The canned backend emits ids from 100, outside MockTokenizer's
/// vocabulary; decode those ids as visible text.
struct CannedTokenizer(MockTokenizer);

impl Encoder for CannedTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
        self.0.encode(input, add_special_tokens)
    }

    fn encode_batch(
        &self,
        inputs: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        self.0.encode_batch(inputs, add_special_tokens)
    }
}

impl Decoder for CannedTokenizer {
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        let ids: Vec<u32> = ids
            .iter()
            .map(|&id| {
                if (100..100 + OUTPUT_TOKENS).contains(&id) {
                    1
                } else {
                    id
                }
            })
            .collect();
        self.0.decode(&ids, skip_special_tokens)
    }
}

impl Tokenizer for CannedTokenizer {
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn get_special_tokens(&self) -> &SpecialTokens {
        self.0.get_special_tokens()
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }
    fn eos_token_ids(&self) -> &[u32] {
        self.0.eos_token_ids()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn apply_chat_template_with_encoding(
        &self,
        messages: &[Value],
        params: ChatTemplateParams,
        assistant_prefix: Option<&str>,
    ) -> anyhow::Result<ChatTemplateOutput> {
        self.0
            .apply_chat_template_with_encoding(messages, params, assistant_prefix)
    }
}

/// Spawn a canned mock gRPC worker in-process, capturing every request it
/// receives into `capture` when set, and holding each answer for `delay`.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_mock_worker(capture: Option<PathBuf>, delay: Duration) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock gRPC worker");
    let port = listener.local_addr().expect("mock worker address").port();
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
        gen_delay: delay,
        output_tokens: OUTPUT_TOKENS,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
        replay: mock_worker::config::ReplayConfig { capture },
        ..mock_worker::config::Config::default()
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    port
}

fn regular() -> RoutingMode {
    RoutingMode::Regular {
        worker_urls: vec![],
    }
}

fn prefill_decode() -> RoutingMode {
    RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    }
}

/// A gRPC router over mock TokenSpeed `workers`, dumping into `dump_dir`
/// from boot when `on_start`.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_router(
    mode: RoutingMode,
    workers: &[(u16, WorkerType)],
    dump_dir: &Path,
    on_start: bool,
) -> (Box<dyn RouterTrait>, Arc<AppContext>) {
    let mut config = RouterConfig::builder()
        .mode(mode)
        .grpc_connection()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    config.token_dump_dir = Some(dump_dir.display().to_string());
    config.token_dump_on_start = on_start;

    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = Arc::new(CannedTokenizer(MockTokenizer::new())) as Arc<dyn Tokenizer>;
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    let app_context =
        common::create_test_context_with_tokenizer_registry(config, tokenizer_registry).await;
    for &(port, worker_type) in workers {
        let worker = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        app_context
            .worker_registry
            .register(Arc::new(worker))
            .unwrap();
    }
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("gRPC router should build");
    (router, app_context)
}

#[expect(clippy::unwrap_used, reason = "test helper")]
fn completion_request(stream: bool) -> CompletionRequest {
    serde_json::from_value(json!({
        "model": MODEL,
        "prompt": "Hello world",
        "max_tokens": 8,
        "stream": stream,
    }))
    .unwrap()
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("token-dump-test-tenant"))
}

#[expect(clippy::expect_used, reason = "test helper")]
async fn read_body(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    String::from_utf8(bytes.to_vec()).expect("utf-8 body")
}

#[expect(clippy::unwrap_used, reason = "test helper")]
fn read_jsonl(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// The one dump file in `dir`.
#[expect(clippy::unwrap_used, reason = "test helper")]
fn dump_lines(dir: &Path) -> Vec<Value> {
    let files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("token-dump-"))
        })
        .collect();
    assert_eq!(files.len(), 1, "{files:?}");
    read_jsonl(&files[0])
}

/// The lines of each call, by call id.
fn group_calls(lines: &[Value]) -> BTreeMap<u64, Vec<Value>> {
    let mut calls: BTreeMap<u64, Vec<Value>> = BTreeMap::new();
    for line in lines {
        if let Some(call) = line["call"].as_u64() {
            calls.entry(call).or_default().push(line.clone());
        }
    }
    for call in calls.values() {
        assert_eq!(call[0]["kind"], "request", "{call:?}");
    }
    calls
}

/// One non-streaming completion that must succeed.
async fn complete_once(router: &dyn RouterTrait) {
    let response = router
        .route_completion(None, &tenant(), completion_request(false), MODEL)
        .await;
    assert_eq!(response.status(), http::StatusCode::OK);
    read_body(response).await;
}

fn kinds(call: &[Value]) -> Vec<&str> {
    call.iter()
        .map(|line| line["kind"].as_str().unwrap_or_default())
        .collect()
}

#[tokio::test]
async fn single_worker_calls_record_what_the_engine_saw_and_sent() {
    let dump_dir = tempfile::tempdir().unwrap();
    let capture_dir = tempfile::tempdir().unwrap();
    let capture = capture_dir.path().join("capture.jsonl");
    let port = start_mock_worker(Some(capture.clone()), Duration::ZERO).await;
    let (router, app) = build_router(
        regular(),
        &[(port, WorkerType::Regular)],
        dump_dir.path(),
        true,
    )
    .await;

    for stream in [false, true] {
        let response = router
            .route_completion(None, &tenant(), completion_request(stream), MODEL)
            .await;
        let status = response.status();
        let body = read_body(response).await;
        assert_eq!(status, http::StatusCode::OK, "{body}");
    }
    app.token_dump.as_ref().unwrap().shutdown().await;

    let lines = dump_lines(dump_dir.path());
    assert_eq!(lines[0]["kind"], "session");
    assert_eq!(lines.last().unwrap()["kind"], "session_end");
    assert_eq!(lines.last().unwrap()["reason"], "shutdown");
    let calls = group_calls(&lines);
    let captured = read_jsonl(&capture);
    assert_eq!(calls.len(), 2, "{lines:?}");
    assert_eq!(captured.len(), 2);
    for (call, captured) in calls.values().zip(&captured) {
        let request = &call[0];
        assert_eq!(request["leg"], "single");
        assert_eq!(request["runtime"], "tokenspeed");
        assert_eq!(request["transport"], "grpc");
        assert_eq!(request["model"], MODEL);
        assert_eq!(request["worker"], format!("grpc://127.0.0.1:{port}"));
        assert_eq!(request["type"], "tokenspeed.grpc.scheduler.GenerateRequest");
        assert_eq!(
            request["input_ids"], captured["input_ids"],
            "the dump records the prompt the engine received"
        );
        assert_eq!(request["request_id"], captured["request_id"]);

        let responses: Vec<&Value> = call
            .iter()
            .filter(|line| line["kind"] == "response")
            .collect();
        assert_eq!(responses.len(), OUTPUT_TOKENS as usize + 1, "{call:?}");
        for (seq, chunk) in responses.iter().take(OUTPUT_TOKENS as usize).enumerate() {
            assert_eq!(chunk["part"], "chunk");
            assert_eq!(chunk["seq"], seq);
            assert_eq!(chunk["token_ids"], json!([100 + seq]));
        }
        let complete = responses.last().unwrap();
        assert_eq!(complete["part"], "complete");
        assert_eq!(complete["token_ids"], json!([100, 101, 102]));
        assert_eq!(complete["finish_reason"], "stop");

        let end = call.last().unwrap();
        assert_eq!(end["kind"], "end");
        assert_eq!(end["status"], "ok", "{call:?}");
        assert_eq!(end["responses"], responses.len());
    }
}

#[tokio::test]
async fn a_pd_request_records_both_legs() {
    let dump_dir = tempfile::tempdir().unwrap();
    let prefill = start_mock_worker(None, Duration::ZERO).await;
    let decode = start_mock_worker(None, Duration::ZERO).await;
    let (router, app) = build_router(
        prefill_decode(),
        &[(prefill, WorkerType::Prefill), (decode, WorkerType::Decode)],
        dump_dir.path(),
        true,
    )
    .await;

    let response = router
        .route_completion(None, &tenant(), completion_request(false), MODEL)
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, http::StatusCode::OK, "{body}");
    app.token_dump.as_ref().unwrap().shutdown().await;

    let lines = dump_lines(dump_dir.path());
    let calls = group_calls(&lines);
    let legs: BTreeMap<String, &Vec<Value>> = calls
        .values()
        .map(|call| (call[0]["leg"].as_str().unwrap().to_string(), call))
        .collect();
    assert_eq!(
        legs.keys().collect::<Vec<_>>(),
        ["decode", "prefill"],
        "{lines:?}"
    );
    assert_eq!(
        legs["prefill"][0]["worker"],
        format!("grpc://127.0.0.1:{prefill}")
    );
    assert_eq!(
        legs["decode"][0]["worker"],
        format!("grpc://127.0.0.1:{decode}")
    );
    for (leg, call) in &legs {
        assert_eq!(call.last().unwrap()["kind"], "end", "{leg}: {call:?}");
        assert_eq!(call.last().unwrap()["status"], "ok", "{leg}: {call:?}");
    }
}

#[tokio::test]
async fn a_call_dropped_before_the_engine_answers_is_recorded_cancelled() {
    let dump_dir = tempfile::tempdir().unwrap();
    // The mock holds its answer far longer than the client waits.
    let port = start_mock_worker(None, Duration::from_secs(30)).await;
    let (router, app) = build_router(
        regular(),
        &[(port, WorkerType::Regular)],
        dump_dir.path(),
        true,
    )
    .await;

    let tenant = tenant();
    let pending = router.route_completion(None, &tenant, completion_request(false), MODEL);
    assert!(
        tokio::time::timeout(Duration::from_millis(500), pending)
            .await
            .is_err(),
        "the mock holds its answer"
    );
    app.token_dump.as_ref().unwrap().shutdown().await;

    let calls = group_calls(&dump_lines(dump_dir.path()));
    assert_eq!(calls.len(), 1, "{calls:?}");
    let call = calls.values().next().unwrap();
    assert_eq!(kinds(call), ["request", "end"]);
    assert_eq!(call[1]["status"], "cancelled");
    assert_eq!(call[1]["responses"], 0);
}

/// Writing the mock's capture to /dev/full fails, so the mock refuses each
/// call with INTERNAL before it starts a stream.
#[cfg(target_os = "linux")]
#[tokio::test]
async fn a_call_the_engine_refuses_is_recorded_start_failed() {
    let dump_dir = tempfile::tempdir().unwrap();
    let port = start_mock_worker(Some(PathBuf::from("/dev/full")), Duration::ZERO).await;
    let (router, app) = build_router(
        regular(),
        &[(port, WorkerType::Regular)],
        dump_dir.path(),
        true,
    )
    .await;

    let response = router
        .route_completion(None, &tenant(), completion_request(false), MODEL)
        .await;
    assert!(response.status().is_server_error(), "{}", response.status());
    app.token_dump.as_ref().unwrap().shutdown().await;

    let calls = group_calls(&dump_lines(dump_dir.path()));
    assert!(!calls.is_empty());
    for call in calls.values() {
        assert_eq!(kinds(call), ["request", "end"], "{call:?}");
        assert_eq!(call[1]["status"], "start_failed");
        assert_eq!(call[1]["error"]["code"], "Internal");
    }
}

#[tokio::test]
async fn a_runtime_session_records_only_while_it_runs() {
    let dump_dir = tempfile::tempdir().unwrap();
    let port = start_mock_worker(None, Duration::ZERO).await;
    let (router, app) = build_router(
        regular(),
        &[(port, WorkerType::Regular)],
        dump_dir.path(),
        false,
    )
    .await;
    let dump = app.token_dump.clone().unwrap();

    complete_once(router.as_ref()).await; // no session: not recorded
    let started = dump
        .start(StartRequest {
            duration_secs: Some(60),
            models: vec![MODEL.to_string()],
        })
        .unwrap();
    complete_once(router.as_ref()).await;
    let stopped = dump.stop().unwrap();
    complete_once(router.as_ref()).await; // stopped: not recorded

    assert_eq!(stopped.session, started.session);
    assert_eq!(stopped.totals.calls, 1);
    let mut lines = Vec::new();
    for _ in 0..100 {
        lines = read_jsonl(&started.file);
        if lines
            .last()
            .is_some_and(|line| line["kind"] == "session_end")
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(group_calls(&lines).len(), 1, "{lines:?}");
    let end = lines.last().unwrap();
    assert_eq!(end["kind"], "session_end");
    assert_eq!(end["reason"], "stopped");
    assert_eq!(end["calls"], 1);
}
