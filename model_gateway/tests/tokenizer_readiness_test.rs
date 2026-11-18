//! Real loopback gateway processes: slow tokenizer preparation must not withdraw
//! a serving model, either at the local REST API or at a peer's mesh import.

use std::{
    collections::HashSet,
    convert::Infallible,
    future::Future,
    io::{Cursor, Write},
    pin::Pin,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use bytes::Bytes;
use http::{HeaderMap, Request, Response};
use http_body::Frame;
use http_body_util::StreamBody;
use prost::Message;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use smg_grpc_client::common_proto::GetTokenizerChunk;
use smg_test_harness::grpc_harness::serve_mock_worker_with;
use tokio::sync::Notify;
use tonic::{body::Body, transport::Channel};
use tower::{Service, ServiceExt};
use zip::{write::SimpleFileOptions, ZipWriter};

#[path = "common/gateway_launch.rs"]
mod gateway_launch;

const WAIT: Duration = Duration::from_secs(60);

struct BundleGate {
    calls: AtomicUsize,
    held: HashSet<usize>,
    released: AtomicBool,
    failing: AtomicBool,
    changed: Notify,
    bundle: Vec<u8>,
}

impl BundleGate {
    fn new(held: &[usize]) -> Arc<Self> {
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        let mut vocab = serde_json::Map::new();
        vocab.insert("[UNK]".into(), json!(0));
        for index in 1..4096 {
            vocab.insert(format!("t{index}"), json!(index));
        }
        let tokenizer = json!({"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
            "post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":vocab,"unk_token":"[UNK]"}});
        for (name, data) in [
            ("tokenizer.json", tokenizer),
            (
                "tokenizer_config.json",
                json!({"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}"}),
            ),
        ] {
            writer
                .start_file(name, SimpleFileOptions::default())
                .unwrap();
            writer.write_all(data.to_string().as_bytes()).unwrap();
        }
        Arc::new(Self {
            calls: AtomicUsize::new(0),
            held: held.iter().copied().collect(),
            released: AtomicBool::new(false),
            failing: AtomicBool::new(false),
            changed: Notify::new(),
            bundle: writer.finish().unwrap().into_inner(),
        })
    }

    fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        self.changed.notify_waiters();
    }

    async fn wait_for_calls(&self, count: usize) {
        let started = Instant::now();
        while self.calls.load(Ordering::SeqCst) < count {
            assert!(started.elapsed() < WAIT, "GetTokenizer was never reached");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

#[derive(Clone)]
struct BundleProxy {
    channel: Channel,
    gate: Arc<BundleGate>,
}

fn grpc_body(message: Option<Vec<u8>>, status: &'static str) -> Response<Body> {
    let mut frames = Vec::new();
    if let Some(message) = message {
        let mut framed = vec![0];
        framed.extend_from_slice(&(message.len() as u32).to_be_bytes());
        framed.extend(message);
        frames.push(Ok::<_, Infallible>(Frame::data(Bytes::from(framed))));
    }
    let mut trailers = HeaderMap::new();
    trailers.insert("grpc-status", status.parse().unwrap());
    frames.push(Ok(Frame::trailers(trailers)));
    Response::builder()
        .header("content-type", "application/grpc")
        .body(Body::new(StreamBody::new(futures::stream::iter(frames))))
        .unwrap()
}

impl Service<Request<Body>> for BundleProxy {
    type Response = Response<Body>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: Request<Body>) -> Self::Future {
        let mut channel = self.channel.clone();
        let gate = Arc::clone(&self.gate);
        Box::pin(async move {
            if !request.uri().path().ends_with("/GetTokenizer") {
                return Ok(match channel.ready().await {
                    Ok(channel) => channel
                        .call(request)
                        .await
                        .unwrap_or_else(|_| grpc_body(None, "14")),
                    Err(_) => grpc_body(None, "14"),
                });
            }
            let call = gate.calls.fetch_add(1, Ordering::SeqCst) + 1;
            while gate.held.contains(&call) {
                let changed = gate.changed.notified();
                if gate.released.load(Ordering::SeqCst) {
                    break;
                }
                changed.await;
            }
            if gate.failing.load(Ordering::SeqCst) {
                return Ok(grpc_body(None, "14"));
            }
            let chunk = GetTokenizerChunk {
                data: gate.bundle.clone(),
                sha256: Sha256::digest(&gate.bundle)
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect(),
            };
            Ok(grpc_body(Some(chunk.encode_to_vec()), "0"))
        })
    }
}

async fn worker(model: &'static str, held: &[usize]) -> (String, Arc<BundleGate>) {
    let port = serve_mock_worker_with(|port| mock_worker::config::Config {
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        grpc_wire: mock_worker::config::SchedulerWire::Sglang,
        model_id: model.to_owned(),
        tokenizer_path: format!("s3://readiness-fixture/{model}"),
        gen_delay: Duration::ZERO,
        output_tokens: 3,
        realistic: false,
        ..Default::default()
    })
    .await;
    let channel = Channel::from_shared(format!("http://127.0.0.1:{port}"))
        .unwrap()
        .connect()
        .await
        .unwrap();
    let gate = BundleGate::new(held);
    let service = BundleProxy {
        channel,
        gate: Arc::clone(&gate),
    };
    let router = axum::Router::new().fallback_service(tower::service_fn(
        move |request: Request<axum::body::Body>| service.clone().oneshot(request.map(Body::new)),
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    #[expect(
        clippy::disallowed_methods,
        reason = "loopback fixture runs for the test runtime lifetime"
    )]
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (format!("grpc://{address}"), gate)
}

struct Gateway {
    child: Child,
    client: Client,
    http: String,
    mesh_port: u16,
    log: std::path::PathBuf,
    hub_calls: Arc<AtomicUsize>,
    hub_task: tokio::task::JoinHandle<()>,
    _directory: tempfile::TempDir,
}

impl Drop for Gateway {
    fn drop(&mut self) {
        self.hub_task.abort();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Gateway {
    async fn start(name: &str, peer: Option<u16>, mesh: bool) -> Self {
        Self::start_mode(name, peer, mesh, true).await
    }

    async fn start_mode(
        name: &str,
        peer: Option<u16>,
        mesh: bool,
        wait_for_tokenizer: bool,
    ) -> Self {
        let client = smg::server::http_client_builder()
            .no_proxy()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let hub_calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&hub_calls);
        let hub = axum::Router::new().fallback(move || {
            let counter = Arc::clone(&counter);
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                StatusCode::NOT_FOUND
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let hub_url = format!("http://{}", listener.local_addr().unwrap());
        #[expect(
            clippy::disallowed_methods,
            reason = "loopback fixture handle is retained and aborted by Gateway::drop"
        )]
        let hub_task = tokio::spawn(async move {
            axum::serve(listener, hub).await.unwrap();
        });
        let [port, metrics_port, mesh_port] = gateway_launch::free_ports();
        let directory = tempfile::tempdir().unwrap();
        let log = directory.path().join("gateway.log");
        let output = std::fs::File::create(&log).unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_smg"));
        command
            .args(gateway_launch::base_args(port, metrics_port))
            .args([
                "--job-queue-concurrency",
                "1",
                "--disable-health-check",
                "--drain-settle-secs",
                "0",
            ])
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("TOKIO_WORKER_THREADS", "2")
            .env("HF_ENDPOINT", hub_url)
            .env_remove("RUST_LOG")
            .stdin(Stdio::null())
            .stdout(output.try_clone().unwrap())
            .stderr(output);
        if wait_for_tokenizer {
            command.arg("--wait-for-tokenizer");
        }
        if mesh {
            command.args([
                "--enable-mesh",
                "--mesh-server-name",
                name,
                "--mesh-host",
                "127.0.0.1",
                "--mesh-port",
                &mesh_port.to_string(),
            ]);
            if let Some(peer) = peer {
                command.args(["--mesh-peer-urls", &format!("127.0.0.1:{peer}")]);
            }
        }
        let mut gateway = Self {
            child: command.spawn().unwrap(),
            client,
            http: format!("http://127.0.0.1:{port}"),
            mesh_port,
            log,
            hub_calls,
            hub_task,
            _directory: directory,
        };
        let started = Instant::now();
        loop {
            if gateway
                .client
                .get(format!("{}/health", gateway.http))
                .send()
                .await
                .is_ok_and(|r| r.status() == StatusCode::OK)
            {
                break;
            }
            assert!(
                gateway.child.try_wait().unwrap().is_none() && started.elapsed() < WAIT,
                "gateway did not start: {}",
                gateway.logs()
            );
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
        gateway
    }

    fn logs(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }

    async fn json(&self, path: &str) -> (StatusCode, Value) {
        let response = self
            .client
            .get(format!("{}{path}", self.http))
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .unwrap();
        let status = response.status();
        (status, response.json().await.unwrap())
    }

    async fn create(&self, url: &str, model: &str) -> String {
        let response = self
            .client
            .post(format!("{}/workers", self.http))
            .json(&spec(url, model))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED, "{}", self.logs());
        response.json::<Value>().await.unwrap()["worker_id"]
            .as_str()
            .unwrap()
            .to_owned()
    }

    async fn ready_worker(&self, id: &str) -> Value {
        let started = Instant::now();
        loop {
            let (_, worker) = self.json(&format!("/workers/{id}")).await;
            if worker["is_healthy"] == true && worker["status"] == "ready" {
                return worker;
            }
            assert!(
                started.elapsed() < WAIT,
                "worker never ready: {worker}; {}",
                self.logs()
            );
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
    }

    async fn infer(&self, model: &str) -> StatusCode {
        self.client.post(format!("{}/v1/chat/completions", self.http))
            .json(&json!({"model":model,"messages":[{"role":"user","content":"hello"}],"max_tokens":3,"temperature":0}))
            .timeout(Duration::from_secs(5)).send().await.unwrap().status()
    }

    async fn wait_ready(&self) {
        let started = Instant::now();
        while self.json("/readiness").await.0 != StatusCode::OK {
            assert!(
                started.elapsed() < WAIT,
                "gateway never ready: {}",
                self.logs()
            );
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
    }

    async fn stays_ready(&self) {
        for _ in 0..20 {
            let (status, body) = self.json("/readiness").await;
            assert_eq!(
                status,
                StatusCode::OK,
                "readiness regressed: {body}; {}",
                self.logs()
            );
            assert_eq!(
                self.infer("model-a").await,
                StatusCode::OK,
                "{}",
                self.logs()
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

fn spec(url: &str, model: &str) -> Value {
    json!({"url":url,"connection_mode":"grpc","runtime_type":"sglang",
        "models":[{"id":model,"tokenizer_path":format!("s3://readiness-fixture/{model}")}],"health":{"drain_settle_secs":0}})
}

async fn seed_a(gateway: &Gateway) {
    let (url, _) = worker("model-a", &[]).await;
    let id = gateway.create(&url, "model-a").await;
    gateway.ready_worker(&id).await;
    gateway.wait_ready().await;
    assert_eq!(
        gateway.infer("model-a").await,
        StatusCode::OK,
        "{}",
        gateway.logs()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn local_preparation_keeps_a_ready_and_b_processing_with_one_queue_slot() {
    let gateway = Gateway::start("local", None, false).await;
    seed_a(&gateway).await;
    let (url, gate) = worker("model-b", &[1]).await;
    let id = gateway.create(&url, "model-b").await;
    gate.wait_for_calls(1).await;
    let (_, pending) = gateway.json(&format!("/workers/{id}")).await;
    assert_eq!(pending["status"], "pending");
    assert_eq!(pending["job_status"]["status"], "processing");
    assert_ne!(gateway.infer("model-b").await, StatusCode::OK);
    gateway.stays_ready().await;
    assert_eq!(gateway.hub_calls.load(Ordering::SeqCst), 0);
    gate.release();
    gateway.ready_worker(&id).await;
    assert_eq!(gateway.infer("model-b").await, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cold_gateway_stays_unready_until_its_first_tokenizer_is_prepared() {
    let gateway = Gateway::start("cold", None, false).await;
    let (url, gate) = worker("model-b", &[1]).await;
    let id = gateway.create(&url, "model-b").await;
    gate.wait_for_calls(1).await;
    assert_eq!(
        gateway.json("/readiness").await.0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        gateway.json(&format!("/workers/{id}")).await.1["status"],
        "pending"
    );
    gate.release();
    gateway.ready_worker(&id).await;
    gateway.wait_ready().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_peer_prepares_its_own_tokenizer_before_importing_a_ready_worker() {
    let owner = Gateway::start("owner", None, true).await;
    let peer = Gateway::start("peer", Some(owner.mesh_port), true).await;
    seed_a(&owner).await;
    let started = Instant::now();
    while peer.json("/readiness").await.0 != StatusCode::OK {
        assert!(
            started.elapsed() < WAIT,
            "mesh did not import A: {}",
            peer.logs()
        );
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
    let (url, gate) = worker("model-b", &[2]).await;
    let id = owner.create(&url, "model-b").await;
    owner.ready_worker(&id).await;
    gate.wait_for_calls(2).await;
    assert_eq!(
        peer.json(&format!("/workers/{id}")).await.0,
        StatusCode::NOT_FOUND
    );
    owner.stays_ready().await;
    peer.stays_ready().await;
    assert_ne!(peer.infer("model-b").await, StatusCode::OK);
    gate.release();
    peer.ready_worker(&id).await;
    assert_eq!(peer.infer("model-b").await, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_mesh_tombstone_cancels_preparation_without_resurrecting_the_import() {
    let owner = Gateway::start("owner", None, true).await;
    let peer = Gateway::start("peer", Some(owner.mesh_port), true).await;
    seed_a(&owner).await;
    let (url, gate) = worker("model-b", &[2]).await;
    let id = owner.create(&url, "model-b").await;
    owner.ready_worker(&id).await;
    gate.wait_for_calls(2).await;
    let response = owner
        .client
        .delete(format!("{}/workers/{id}", owner.http))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let started = Instant::now();
    while owner.json(&format!("/workers/{id}")).await.0 != StatusCode::NOT_FOUND {
        assert!(started.elapsed() < WAIT, "owner did not remove B");
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    gateway_launch::log_with_line(
        &peer.log,
        &format!("Ended mesh tokenizer preparation worker_id=\"{id}\""),
        WAIT,
    )
    .unwrap();
    gate.release();
    peer.stays_ready().await;
    for _ in 0..20 {
        assert_eq!(
            peer.json(&format!("/workers/{id}")).await.0,
            StatusCode::NOT_FOUND,
            "{}",
            peer.logs()
        );
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn default_local_registration_keeps_the_existing_asynchronous_readiness_behavior() {
    let gateway = Gateway::start_mode("legacy", None, false, false).await;
    seed_a(&gateway).await;
    let (url, gate) = worker("model-b", &[1]).await;
    let id = gateway.create(&url, "model-b").await;
    gate.wait_for_calls(1).await;
    gateway.ready_worker(&id).await;
    let started = Instant::now();
    loop {
        if gateway.json("/readiness").await.0 == StatusCode::SERVICE_UNAVAILABLE {
            break;
        }
        assert!(
            started.elapsed() < WAIT,
            "default behavior unexpectedly changed"
        );
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert_eq!(gateway.infer("model-a").await, StatusCode::OK);
    assert!(gateway.hub_calls.load(Ordering::SeqCst) > 0);
    gate.release();
    gateway.wait_ready().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_replicas_deduplicate_and_other_models_do_not_wait_for_their_load() {
    let gateway = Gateway::start("concurrent", None, false).await;
    seed_a(&gateway).await;
    let (url, first) = worker("model-b", &[1]).await;
    let first_id = gateway.create(&url, "model-b").await;
    first.wait_for_calls(1).await;
    let (url, second) = worker("model-b", &[1]).await;
    let second_id = gateway.create(&url, "model-b").await;
    let (url, third) = worker("model-c", &[]).await;
    let third_id = gateway.create(&url, "model-c").await;
    gateway.ready_worker(&third_id).await;
    assert_eq!(third.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        second.calls.load(Ordering::SeqCst),
        0,
        "a replica bypassed the per-model load lock"
    );
    gateway.stays_ready().await;
    first.release();
    gateway.ready_worker(&first_id).await;
    gateway.ready_worker(&second_id).await;
    assert_eq!(first.calls.load(Ordering::SeqCst), 1);
    assert_eq!(second.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn local_cancellation_does_not_publish_a_worker_after_its_bundle_is_released() {
    let gateway = Gateway::start("cancel", None, false).await;
    seed_a(&gateway).await;
    let (url, gate) = worker("model-b", &[1]).await;
    let id = gateway.create(&url, "model-b").await;
    gate.wait_for_calls(1).await;
    let response = gateway
        .client
        .delete(format!("{}/workers/{id}", gateway.http))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    let started = Instant::now();
    while gateway.json(&format!("/workers/{id}")).await.0 != StatusCode::NOT_FOUND {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "cancellation did not interrupt preparation"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    gate.release();
    gateway.stays_ready().await;
    assert_eq!(
        gateway.json(&format!("/workers/{id}")).await.0,
        StatusCode::NOT_FOUND
    );
    // A retry owns a fresh UUID; late cleanup must not remove it.
    let retry = gateway.create(&url, "model-b").await;
    assert_ne!(retry, id);
    gateway.ready_worker(&retry).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_preparation_leaves_a_serving_and_allows_a_clean_same_url_retry() {
    let gateway = Gateway::start("failure", None, false).await;
    seed_a(&gateway).await;
    let (url, gate) = worker("model-b", &[1]).await;
    gate.failing.store(true, Ordering::SeqCst);
    let id = gateway.create(&url, "model-b").await;
    gate.wait_for_calls(1).await;
    gateway.stays_ready().await;
    gate.release();
    let started = Instant::now();
    while gateway.json(&format!("/workers/{id}")).await.0 != StatusCode::NOT_FOUND {
        assert!(
            started.elapsed() < WAIT,
            "failed registration did not end: {}",
            gateway.logs()
        );
        assert_eq!(gateway.json("/readiness").await.0, StatusCode::OK);
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    gateway_launch::log_with_line(&gateway.log, "Failed job: type=AddWorker", WAIT).unwrap();
    gate.failing.store(false, Ordering::SeqCst);
    let retry = gateway.create(&url, "model-b").await;
    gateway.ready_worker(&retry).await;
    assert_eq!(gateway.infer("model-b").await, StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_slow_put_keeps_the_old_worker_and_cannot_overwrite_a_newer_put() {
    let gateway = Gateway::start("put", None, false).await;
    let (url, gate) = worker("model-a", &[2]).await;
    let id = gateway.create(&url, "model-a").await;
    gateway.ready_worker(&id).await;
    gateway.wait_ready().await;
    let response = gateway
        .client
        .put(format!("{}/workers/{id}", gateway.http))
        .json(&spec(&url, "model-b"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    gate.wait_for_calls(2).await;
    gateway.stays_ready().await;
    let response = gateway
        .client
        .put(format!("{}/workers/{id}", gateway.http))
        .json(&spec(&url, "model-c"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let started = Instant::now();
    loop {
        let (_, current) = gateway.json(&format!("/workers/{id}")).await;
        if current["models"][0]["id"] == "model-c" {
            break;
        }
        assert!(
            started.elapsed() < WAIT,
            "newer replacement did not register: {current}; {}",
            gateway.logs()
        );
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    gate.release();
    tokio::time::sleep(Duration::from_millis(500)).await;
    let (_, current) = gateway.json(&format!("/workers/{id}")).await;
    assert_eq!(current["models"][0]["id"], "model-c");
    assert_eq!(current["status"], "ready");
    assert_eq!(gateway.infer("model-c").await, StatusCode::OK);

    // A replacement that ultimately fails must retain the committed worker.
    gate.failing.store(true, Ordering::SeqCst);
    let response = gateway
        .client
        .put(format!("{}/workers/{id}", gateway.http))
        .json(&spec(&url, "model-d"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let started = Instant::now();
    loop {
        let (_, current) = gateway.json(&format!("/workers/{id}")).await;
        assert_eq!(current["id"], id);
        assert_eq!(current["models"][0]["id"], "model-c");
        assert_eq!(current["status"], "ready");
        assert_eq!(gateway.json("/readiness").await.0, StatusCode::OK);
        assert_eq!(gateway.infer("model-c").await, StatusCode::OK);
        if current["job_status"]["status"] == "failed" {
            break;
        }
        assert!(
            started.elapsed() < WAIT,
            "failed PUT did not terminate: {current}; {}",
            gateway.logs()
        );
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn default_mesh_imports_keep_the_existing_no_local_autoload_behavior() {
    let owner = Gateway::start_mode("owner", None, true, false).await;
    let peer = Gateway::start_mode("peer", Some(owner.mesh_port), true, false).await;
    let (url, gate) = worker("model-a", &[]).await;
    let id = owner.create(&url, "model-a").await;
    owner.wait_ready().await;
    peer.ready_worker(&id).await;
    let started = Instant::now();
    while peer.json("/readiness").await.0 != StatusCode::SERVICE_UNAVAILABLE {
        assert!(started.elapsed() < WAIT);
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert_eq!(
        gate.calls.load(Ordering::SeqCst),
        1,
        "default mesh unexpectedly loaded a tokenizer"
    );
    assert_eq!(peer.infer("model-a").await, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_newer_mesh_generation_wins_over_a_held_older_tokenizer_load() {
    let owner = Gateway::start("owner", None, true).await;
    let peer = Gateway::start("peer", Some(owner.mesh_port), true).await;
    seed_a(&owner).await;
    peer.wait_ready().await;
    let (url, gate) = worker("model-b", &[2]).await;
    let id = owner.create(&url, "model-b").await;
    owner.ready_worker(&id).await;
    gate.wait_for_calls(2).await;
    let response = owner
        .client
        .put(format!("{}/workers/{id}", owner.http))
        .json(&spec(&url, "model-c"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let started = Instant::now();
    loop {
        let (_, current) = peer.json(&format!("/workers/{id}")).await;
        if current["models"][0]["id"] == "model-c" && current["status"] == "ready" {
            break;
        }
        assert!(
            started.elapsed() < WAIT,
            "new mesh generation was not imported: {current}; {}",
            peer.logs()
        );
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
    gate.release();
    peer.stays_ready().await;
    assert_eq!(
        peer.json(&format!("/workers/{id}")).await.1["models"][0]["id"],
        "model-c"
    );
    assert_eq!(peer.infer("model-c").await, StatusCode::OK);
}
