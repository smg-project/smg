// Mock worker for testing - these functions are used by integration tests
#![allow(dead_code, clippy::allow_attributes)]

use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{Json, Multipart, Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response, Sse,
    },
    routing::{get, post},
    Router,
};
use futures_util::stream::{self, StreamExt};
use serde_json::json;
use tokio::sync::{oneshot, Notify, RwLock, Semaphore};
use uuid::Uuid;

/// Test-controlled hold gate for the `/generate` endpoint.
///
/// Lets a test pin requests on the mock worker *before* it emits any byte —
/// i.e. holds them in the pre-TTFT window — and release them on demand. This
/// replaces racing `sleep`s in scheduler admission tests with explicit
/// signalling, so capacity occupancy and preemption windows are deterministic.
///
/// Flow:
/// 1. Each `/generate` handler, on entry, bumps `arrived` (one permit) so the
///    test can `wait_for_arrivals(n)` to learn that exactly `n` requests have
///    reached the worker and are now occupying scheduler slots.
/// 2. The handler then parks on `release` until the test calls `release()`.
///    A request whose admission cancel token fires while parked is unwound by
///    the gateway's `PreemptionGuard` (the handler future is dropped), so the
///    gate never needs to observe cancellation itself.
pub struct HoldGate {
    arrived: Semaphore,
    release: Notify,
    released: AtomicBool,
}

impl HoldGate {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            arrived: Semaphore::new(0),
            release: Notify::new(),
            released: AtomicBool::new(false),
        })
    }

    /// Handler side: signal arrival, then park until released.
    async fn arrive_and_wait(&self) {
        self.arrived.add_permits(1);
        // Loop guards against spurious wakeups and the release-before-park
        // race: `notified()` registered before the `released` check would miss
        // a `notify_waiters()` that already fired, so re-check the flag.
        while !self.released.load(Ordering::Acquire) {
            let notified = self.release.notified();
            if self.released.load(Ordering::Acquire) {
                break;
            }
            notified.await;
        }
    }

    /// Test side: block until `n` handlers have reached the worker (and are
    /// therefore occupying `n` scheduler slots).
    #[expect(
        clippy::expect_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    pub async fn wait_for_arrivals(&self, n: u32) {
        let permits = self
            .arrived
            .acquire_many(n)
            .await
            .expect("hold-gate semaphore is never closed");
        permits.forget();
    }

    /// Test side: release every parked handler (and any that arrive later).
    pub fn release(&self) {
        self.released.store(true, Ordering::Release);
        self.release.notify_waiters();
    }
}

/// Per-port scheduler test controls, looked up by the mock worker's bound
/// port. Kept *outside* [`MockWorkerConfig`] so adding scheduler knobs doesn't
/// force every existing `MockWorkerConfig { .. }` literal in the test suite to
/// gain new fields. Tests register controls via [`set_scheduler_controls`]
/// before the worker starts; the relevant handlers consult them by port.
#[derive(Clone, Default)]
pub struct SchedulerControls {
    /// Value reported as `max_running_requests` in `/server_info`. `None`
    /// keeps the historical default (2048). The priority scheduler derives
    /// backend capacity from the sum of this across the healthy fleet, so
    /// admission tests set it to a small, known number (e.g. 1) to pin
    /// capacity deterministically.
    pub max_running_requests: Option<u16>,
    /// Optional pre-TTFT hold gate for `/generate`. When set, each generate
    /// request signals arrival then parks until [`HoldGate::release`] is
    /// called. `None` preserves the normal immediate-response behavior.
    pub hold_gate: Option<Arc<HoldGate>>,
}

static SCHEDULER_CONTROLS: OnceLock<Mutex<HashMap<u16, SchedulerControls>>> = OnceLock::new();

fn scheduler_controls_table() -> &'static Mutex<HashMap<u16, SchedulerControls>> {
    SCHEDULER_CONTROLS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register scheduler test controls for a mock worker that will bind `port`.
/// Call this *before* the worker starts (pick a free port via `portpicker`
/// and pass the same value in [`MockWorkerConfig::port`]).
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
pub fn set_scheduler_controls(port: u16, controls: SchedulerControls) {
    scheduler_controls_table()
        .lock()
        .expect("scheduler controls mutex poisoned")
        .insert(port, controls);
}

#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn scheduler_controls_for(port: u16) -> SchedulerControls {
    scheduler_controls_table()
        .lock()
        .expect("scheduler controls mutex poisoned")
        .get(&port)
        .cloned()
        .unwrap_or_default()
}

/// Remove a port's scheduler controls. Called on [`MockWorker`] teardown so a
/// later worker that reuses the same port (portpicker recycles freed ports)
/// doesn't inherit stale `max_running_requests` / `hold_gate`. Tolerant of a
/// poisoned mutex since it runs from `Drop`.
fn clear_scheduler_controls(port: u16) {
    if let Ok(mut table) = scheduler_controls_table().lock() {
        table.remove(&port);
    }
}

/// Records the JSON body of every chat request a mock worker receives.
///
/// A test can only read the mock's canned answer, which says nothing about
/// what the gateway actually forwarded. This exposes the forwarded body, so a
/// test can assert on rewrites the gateway performs on the way out — for
/// instance that a request addressed to a model alias reaches the worker under
/// the canonical model ID.
///
/// Kept in a port-keyed table rather than in [`MockWorkerConfig`], for the
/// same reason as [`SchedulerControls`]: adding a knob must not force every
/// existing `MockWorkerConfig { .. }` literal in the suite to gain a field.
#[derive(Default)]
pub struct RequestRecorder {
    bodies: Mutex<Vec<serde_json::Value>>,
}

impl RequestRecorder {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Every body received so far, oldest first.
    #[expect(
        clippy::expect_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    pub fn bodies(&self) -> Vec<serde_json::Value> {
        self.bodies
            .lock()
            .expect("request recorder mutex poisoned")
            .clone()
    }

    /// The single body received, panicking unless exactly one arrived.
    #[expect(
        clippy::expect_used,
        reason = "test helper - panicking on failure is intentional"
    )]
    pub fn only_body(&self) -> serde_json::Value {
        let bodies = self.bodies();
        assert_eq!(
            bodies.len(),
            1,
            "expected exactly one recorded request, got {}",
            bodies.len()
        );
        bodies.into_iter().next().expect("length checked above")
    }
}

static REQUEST_RECORDERS: OnceLock<Mutex<HashMap<u16, Arc<RequestRecorder>>>> = OnceLock::new();

fn request_recorders_table() -> &'static Mutex<HashMap<u16, Arc<RequestRecorder>>> {
    REQUEST_RECORDERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Attach a recorder to the mock worker that will bind `port`. Call before the
/// worker starts, like [`set_scheduler_controls`].
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
pub fn set_request_recorder(port: u16, recorder: Arc<RequestRecorder>) {
    request_recorders_table()
        .lock()
        .expect("request recorders mutex poisoned")
        .insert(port, recorder);
}

fn record_request(port: u16, body: &serde_json::Value) {
    let recorder = request_recorders_table()
        .lock()
        .ok()
        .and_then(|table| table.get(&port).cloned());
    if let Some(recorder) = recorder {
        if let Ok(mut bodies) = recorder.bodies.lock() {
            bodies.push(body.clone());
        }
    }
}

/// Remove a port's recorder on teardown so a later worker reusing the port
/// does not append to it. Tolerant of a poisoned mutex since it runs from
/// `Drop`.
fn clear_request_recorder(port: u16) {
    if let Ok(mut table) = request_recorders_table().lock() {
        table.remove(&port);
    }
}

/// Configuration for mock worker behavior
#[derive(Clone)]
pub struct MockWorkerConfig {
    pub port: u16,
    pub worker_type: WorkerType,
    pub health_status: HealthStatus,
    pub response_delay_ms: u64,
    pub fail_rate: f32,
}

#[derive(Clone, Debug)]
pub enum WorkerType {
    Regular,
    Prefill,
    Decode,
}

#[derive(Clone, Debug)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Degraded,
}

/// Mock worker server for testing
pub struct MockWorker {
    config: Arc<RwLock<MockWorkerConfig>>,
    shutdown_handle: Option<tokio::task::JoinHandle<()>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    /// Resolved bind port, cached so sync `Drop` can prune this worker's entry
    /// from the global scheduler-controls table.
    bound_port: Option<u16>,
}

impl MockWorker {
    pub fn new(config: MockWorkerConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            shutdown_handle: None,
            shutdown_tx: None,
            bound_port: None,
        }
    }

    /// Start the mock worker server
    #[expect(
        clippy::disallowed_methods,
        clippy::print_stderr,
        reason = "test infrastructure"
    )]
    pub async fn start(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let config = self.config.clone();
        let port = config.read().await.port;

        // If port is 0, find an available port
        let port = if port == 0 {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            drop(listener);
            config.write().await.port = port;
            port
        } else {
            port
        };
        self.bound_port = Some(port);

        let app = Router::new()
            .route("/health", get(health_handler))
            .route("/health_generate", get(health_generate_handler))
            .route("/server_info", get(server_info_handler))
            .route("/get_server_info", get(server_info_handler))
            .route("/model_info", get(model_info_handler))
            .route("/get_model_info", get(model_info_handler))
            .route("/generate", post(generate_handler))
            .route("/v1/chat/completions", post(chat_completions_handler))
            .route("/v1/messages", post(messages_handler))
            .route("/v1/completions", post(completions_handler))
            .route("/v1/rerank", post(rerank_handler))
            .route(
                "/v1/audio/transcriptions",
                post(audio_transcriptions_handler),
            )
            .route("/v1/responses", post(responses_handler))
            .route("/v1/realtime/sessions", post(realtime_rest_handler))
            .route("/v1/realtime/client_secrets", post(realtime_rest_handler))
            .route(
                "/v1/realtime/transcription_sessions",
                post(realtime_rest_handler),
            )
            .route("/v1/responses/{response_id}", get(responses_get_handler))
            .route(
                "/v1/responses/{response_id}/cancel",
                post(responses_cancel_handler),
            )
            .route("/flush_cache", post(flush_cache_handler))
            .route("/v1/models", get(v1_models_handler))
            .with_state(config);

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        self.shutdown_tx = Some(shutdown_tx);

        // Spawn the server in a separate task
        let handle = tokio::spawn(async move {
            let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Failed to bind to port {port}: {e}");
                    return;
                }
            };

            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            });

            if let Err(e) = server.await {
                eprintln!("Server error: {e}");
            }
        });

        self.shutdown_handle = Some(handle);

        // Wait for the server to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let url = format!("http://127.0.0.1:{port}");
        Ok(url)
    }

    /// Stop the mock worker server
    pub async fn stop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }

        if let Some(handle) = self.shutdown_handle.take() {
            // Wait for the server to shut down
            let _ = tokio::time::timeout(tokio::time::Duration::from_secs(5), handle).await;
        }
    }
}

impl Drop for MockWorker {
    fn drop(&mut self) {
        // Clean shutdown when dropped
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        // Prune our scheduler controls and recorder so a later worker reusing
        // this port doesn't inherit stale state.
        if let Some(port) = self.bound_port {
            clear_scheduler_controls(port);
            clear_request_recorder(port);
        }
    }
}

// Handler implementations

/// Check if request should fail based on configured fail_rate
fn should_fail(config: &MockWorkerConfig) -> bool {
    rand::random::<f32>() < config.fail_rate
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn health_handler(State(config): State<Arc<RwLock<MockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    match config.health_status {
        HealthStatus::Healthy => Json(json!({
            "status": "healthy",
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "worker_type": format!("{:?}", config.worker_type),
        }))
        .into_response(),
        HealthStatus::Unhealthy => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unhealthy",
                "error": "Worker is not responding"
            })),
        )
            .into_response(),
        HealthStatus::Degraded => Json(json!({
            "status": "degraded",
            "warning": "High load detected"
        }))
        .into_response(),
    }
}

async fn health_generate_handler(State(config): State<Arc<RwLock<MockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    if should_fail(&config) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    if matches!(config.health_status, HealthStatus::Healthy) {
        Json(json!({
            "status": "ok",
            "queue_length": 0,
            "processing_time_ms": config.response_delay_ms
        }))
        .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "Generation service unavailable"
            })),
        )
            .into_response()
    }
}

async fn server_info_handler(State(config): State<Arc<RwLock<MockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    // server_info is a metadata endpoint used during worker registration.
    // Must always succeed regardless of fail_rate so workers register properly.

    // The worker-registration metadata pipeline reads `max_running_requests`
    // from here and stores it as a label; `WorkerCapacity` sums it across the
    // healthy fleet to size the priority scheduler. Tests pin it to a small
    // value (looked up by port) to make scheduler capacity deterministic;
    // absent an override we keep the historical 2048 so non-scheduler tests
    // are unaffected.
    let max_running_requests = scheduler_controls_for(config.port)
        .max_running_requests
        .map_or(2048, u32::from);

    Json(json!({
        "model_path": "mock-model",
        "served_model_name": "mock-model",
        "tokenizer_path": "mock-tokenizer-path",
        "port": config.port,
        "host": "127.0.0.1",
        "max_num_batched_tokens": 32768,
        "max_prefill_tokens": 16384,
        "mem_fraction_static": 0.88,
        "tp_size": 1,
        "dp_size": 1,
        "stream_interval": 8,
        "dtype": "float16",
        "device": "cuda",
        "enable_flashinfer": true,
        "enable_p2p_check": true,
        "context_length": 32768,
        "chat_template": null,
        "disable_radix_cache": false,
        "enable_torch_compile": false,
        "trust_remote_code": false,
        "show_time_cost": false,
        "waiting_queue_size": 0,
        "running_queue_size": 0,
        "req_to_token_ratio": 1.2,
        "min_running_requests": 0,
        "max_running_requests": max_running_requests,
        "max_req_num": 8192,
        "max_batch_tokens": 32768,
        "schedule_policy": "lpm",
        "schedule_conservativeness": 1.0,
        "version": "0.3.0",
        "internal_states": [{
            "waiting_queue_size": 0,
            "running_queue_size": 0
        }]
    }))
    .into_response()
}

async fn model_info_handler(State(config): State<Arc<RwLock<MockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    if should_fail(&config) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    Json(json!({
        "model_path": "mock-model",
        "tokenizer_path": "mock-tokenizer-path",
        "is_generation": true,
        "preferred_sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 2048
        }
    }))
    .into_response()
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn generate_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let config = config.read().await;
    let worker_id = format!("worker-{}", config.port);

    if should_fail(&config) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            [("x-worker-id", worker_id)],
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    // Pre-TTFT hold: signal arrival (so the test knows this request now holds a
    // scheduler slot) and park until released. The gate is looked up by port
    // from the test-controlled registry. If the gateway preempts this request
    // while parked, `PreemptionGuard` drops this whole future, so we never
    // reach the response below.
    if let Some(gate) = scheduler_controls_for(config.port).hold_gate {
        gate.arrive_and_wait().await;
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if is_stream {
        let stream_delay = config.response_delay_ms;

        // Check if it's a batch request
        let is_batch = payload.get("text").and_then(|t| t.as_array()).is_some();

        let batch_size = if is_batch {
            payload
                .get("text")
                .and_then(|t| t.as_array())
                .map(|arr| arr.len())
                .unwrap_or(1)
        } else {
            1
        };

        let mut events = Vec::new();

        // Generate events for each item in batch
        for i in 0..batch_size {
            let timestamp_start = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();

            let data = json!({
                "text": format!("Mock response {}", i + 1),
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "completion_tokens_wo_jump_forward": 5,
                    "input_token_logprobs": null,
                    "output_token_logprobs": null,
                    "first_token_latency": stream_delay as f64 / 1000.0,
                    "time_to_first_token": stream_delay as f64 / 1000.0,
                    "time_per_output_token": 0.01,
                    "end_time": timestamp_start + (stream_delay as f64 / 1000.0),
                    "start_time": timestamp_start,
                    "finish_reason": {
                        "type": "stop",
                        "reason": "length"
                    }
                },
                "stage": "mid"
            });

            events.push(Ok::<_, Infallible>(Event::default().data(data.to_string())));
        }

        // Add [DONE] event
        events.push(Ok(Event::default().data("[DONE]")));

        let stream = stream::iter(events);

        (
            [("x-worker-id", worker_id)],
            Sse::new(stream).keep_alive(KeepAlive::default()),
        )
            .into_response()
    } else {
        (
            [("x-worker-id", worker_id)],
            Json(json!({
                "text": "This is a mock response.",
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "completion_tokens_wo_jump_forward": 5,
                    "input_token_logprobs": null,
                    "output_token_logprobs": null,
                    "first_token_latency": config.response_delay_ms as f64 / 1000.0,
                    "time_to_first_token": config.response_delay_ms as f64 / 1000.0,
                    "time_per_output_token": 0.01,
                    "finish_reason": {
                        "type": "stop",
                        "reason": "length"
                    }
                }
            })),
        )
            .into_response()
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn chat_completions_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let config = config.read().await;
    // Before any early return, so a test still sees what arrived even when the
    // mock is configured to fail the request.
    record_request(config.port, &payload);

    if should_fail(&config) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Random failure for testing",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if is_stream {
        let request_id = format!("chatcmpl-{}", Uuid::now_v7());

        let stream = stream::once(async move {
            let chunk = json!({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "This is a mock chat response."
                    },
                    "finish_reason": null
                }]
            });

            Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
        })
        .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(json!({
            "id": format!("chatcmpl-{}", Uuid::now_v7()),
            "object": "chat.completion",
            "created": timestamp,
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock chat response."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .into_response()
    }
}

async fn realtime_rest_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let config = config.read().await;
    record_request(config.port, &payload);

    if should_fail(&config) {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }

    Json(json!({"status": "ok"})).into_response()
}

async fn messages_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let config = config.read().await;

    if should_fail(&config) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Random failure for testing"
                }
            })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("mock-model")
        .to_string();
    let message_id = format!("msg_{}", Uuid::now_v7());

    if is_stream {
        let message_id_for_stream = message_id.clone();
        let model_for_stream = model.clone();
        let events = vec![
            (
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": message_id_for_stream,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model_for_stream,
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": {"input_tokens": 10, "output_tokens": 0}
                    }
                }),
            ),
            (
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                }),
            ),
            (
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Mock streamed response."}
                }),
            ),
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0}),
            ),
            (
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                    "usage": {"output_tokens": 5}
                }),
            ),
            ("message_stop", json!({"type": "message_stop"})),
        ];

        let stream = stream::iter(events.into_iter().map(|(event, data)| {
            Ok::<_, Infallible>(Event::default().event(event).data(data.to_string()))
        }));

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(json!({
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "This is a mock messages response."}
            ],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }))
        .into_response()
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn completions_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let config = config.read().await;

    if should_fail(&config) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Random failure for testing",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if is_stream {
        let request_id = format!("cmpl-{}", Uuid::now_v7());

        let stream = stream::once(async move {
            let chunk = json!({
                "id": request_id,
                "object": "text_completion",
                "created": timestamp,
                "model": "mock-model",
                "choices": [{
                    "text": "This is a mock completion.",
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": null
                }]
            });

            Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
        })
        .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(json!({
            "id": format!("cmpl-{}", Uuid::now_v7()),
            "object": "text_completion",
            "created": timestamp,
            "model": "mock-model",
            "choices": [{
                "text": "This is a mock completion.",
                "index": 0,
                "logprobs": null,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .into_response()
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn responses_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let config = config.read().await;

    if should_fail(&config) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Random failure for testing",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    // Background storage simulation
    let is_background = payload
        .get("background")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let req_id = payload
        .get("request_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    if is_background {
        if let Some(id) = &req_id {
            store_response_for_port(config.port, id);
        }
    }

    if is_stream {
        let request_id = format!("resp-{}", Uuid::now_v7());

        // Check if this is an MCP tool call scenario
        let has_tools = payload
            .get("tools")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter().any(|tool| {
                    tool.get("type")
                        .and_then(|t| t.as_str())
                        .map(|t| t == "function")
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);
        let has_function_output = payload
            .get("input")
            .and_then(|v| v.as_array())
            .map(|items| {
                items.iter().any(|item| {
                    item.get("type")
                        .and_then(|t| t.as_str())
                        .map(|t| t == "function_call_output")
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);
        let has_prior_tool_context = has_function_output;

        if has_tools && !has_prior_tool_context {
            // First turn: emit streaming tool call events using OpenAI-style function_call ids
            let call_id = format!("call_{}", &Uuid::now_v7().simple().to_string()[..24]);
            let item_id = format!("fc_{}", Uuid::now_v7().simple());
            let rid = request_id.clone();

            let events = vec![
                // response.created
                Ok::<_, Infallible>(
                    Event::default().event("response.created").data(
                        json!({
                            "type": "response.created",
                            "response": {
                                "id": rid.clone(),
                                "object": "response",
                                "created_at": timestamp,
                                "model": "mock-model",
                                "status": "in_progress"
                            }
                        })
                        .to_string(),
                    ),
                ),
                // response.in_progress
                Ok(Event::default().event("response.in_progress").data(
                    json!({
                        "type": "response.in_progress",
                        "response": {
                            "id": rid.clone(),
                            "object": "response",
                            "created_at": timestamp,
                            "model": "mock-model",
                            "status": "in_progress"
                        }
                    })
                    .to_string(),
                )),
                // response.output_item.added with function_call
                Ok(Event::default().event("response.output_item.added").data(
                    json!({
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "id": item_id.clone(),
                            "type": "function_call",
                            "call_id": call_id.clone(),
                            "name": "brave_web_search",
                            "arguments": "",
                            "status": "in_progress"
                        }
                    })
                    .to_string(),
                )),
                // response.function_call_arguments.delta events
                Ok(Event::default()
                    .event("response.function_call_arguments.delta")
                    .data(
                        json!({
                            "type": "response.function_call_arguments.delta",
                            "output_index": 0,
                            "item_id": item_id.clone(),
                            "delta": "{\"query\""
                        })
                        .to_string(),
                    )),
                Ok(Event::default()
                    .event("response.function_call_arguments.delta")
                    .data(
                        json!({
                            "type": "response.function_call_arguments.delta",
                            "output_index": 0,
                            "item_id": item_id.clone(),
                            "delta": ":\"SGLang"
                        })
                        .to_string(),
                    )),
                Ok(Event::default()
                    .event("response.function_call_arguments.delta")
                    .data(
                        json!({
                            "type": "response.function_call_arguments.delta",
                            "output_index": 0,
                            "item_id": item_id.clone(),
                            "delta": " router MCP"
                        })
                        .to_string(),
                    )),
                Ok(Event::default()
                    .event("response.function_call_arguments.delta")
                    .data(
                        json!({
                            "type": "response.function_call_arguments.delta",
                            "output_index": 0,
                            "item_id": item_id.clone(),
                            "delta": " integration\"}"
                        })
                        .to_string(),
                    )),
                // response.function_call_arguments.done
                Ok(Event::default()
                    .event("response.function_call_arguments.done")
                    .data(
                        json!({
                            "type": "response.function_call_arguments.done",
                            "output_index": 0,
                            "item_id": item_id.clone()
                        })
                        .to_string(),
                    )),
                // response.output_item.done
                Ok(Event::default().event("response.output_item.done").data(
                    json!({
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "id": item_id.clone(),
                            "type": "function_call",
                            "call_id": call_id.clone(),
                            "name": "brave_web_search",
                            "arguments": "{\"query\":\"SGLang router MCP integration\"}",
                            "status": "completed"
                        }
                    })
                    .to_string(),
                )),
                // response.completed
                Ok(Event::default().event("response.completed").data(
                    json!({
                        "type": "response.completed",
                        "response": {
                            "id": rid,
                            "object": "response",
                            "created_at": timestamp,
                            "model": "mock-model",
                            "status": "completed"
                        }
                    })
                    .to_string(),
                )),
                // [DONE]
                Ok(Event::default().data("[DONE]")),
            ];

            let stream = stream::iter(events);
            Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response()
        } else if has_tools && has_prior_tool_context {
            // Resume turn: emit streaming text response
            let rid = request_id.clone();
            let msg_id = format!("msg_{}", &Uuid::now_v7().simple().to_string()[..24]);

            let events = vec![
                // response.created
                Ok::<_, Infallible>(
                    Event::default().event("response.created").data(
                        json!({
                            "type": "response.created",
                            "response": {
                                "id": rid.clone(),
                                "object": "response",
                                "created_at": timestamp,
                                "model": "mock-model",
                                "status": "in_progress"
                            }
                        })
                        .to_string(),
                    ),
                ),
                // response.in_progress
                Ok(Event::default().event("response.in_progress").data(
                    json!({
                        "type": "response.in_progress",
                        "response": {
                            "id": rid.clone(),
                            "object": "response",
                            "created_at": timestamp,
                            "model": "mock-model",
                            "status": "in_progress"
                        }
                    })
                    .to_string(),
                )),
                // response.output_item.added with message
                Ok(Event::default().event("response.output_item.added").data(
                    json!({
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "id": msg_id.clone(),
                            "type": "message",
                            "role": "assistant",
                            "content": []
                        }
                    })
                    .to_string(),
                )),
                // response.content_part.added
                Ok(Event::default().event("response.content_part.added").data(
                    json!({
                        "type": "response.content_part.added",
                        "output_index": 0,
                        "item_id": msg_id.clone(),
                        "part": {
                            "type": "output_text",
                            "text": ""
                        }
                    })
                    .to_string(),
                )),
                // response.output_text.delta events
                Ok(Event::default().event("response.output_text.delta").data(
                    json!({
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "Tool result"
                    })
                    .to_string(),
                )),
                Ok(Event::default().event("response.output_text.delta").data(
                    json!({
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": " consumed;"
                    })
                    .to_string(),
                )),
                Ok(Event::default().event("response.output_text.delta").data(
                    json!({
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": " here is the final answer."
                    })
                    .to_string(),
                )),
                // response.output_text.done
                Ok(Event::default().event("response.output_text.done").data(
                    json!({
                        "type": "response.output_text.done",
                        "output_index": 0,
                        "content_index": 0,
                        "text": "Tool result consumed; here is the final answer."
                    })
                    .to_string(),
                )),
                // response.output_item.done
                Ok(Event::default().event("response.output_item.done").data(
                    json!({
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "output_text",
                                "text": "Tool result consumed; here is the final answer."
                            }]
                        }
                    })
                    .to_string(),
                )),
                // response.completed
                Ok(Event::default().event("response.completed").data(
                    json!({
                        "type": "response.completed",
                        "response": {
                            "id": rid,
                            "object": "response",
                            "created_at": timestamp,
                            "model": "mock-model",
                            "status": "completed",
                            "usage": {
                                "input_tokens": 12,
                                "output_tokens": 7,
                                "total_tokens": 19
                            }
                        }
                    })
                    .to_string(),
                )),
                // [DONE]
                Ok(Event::default().data("[DONE]")),
            ];

            let stream = stream::iter(events);
            Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response()
        } else {
            // Default streaming response
            let stream = stream::once(async move {
                let chunk = json!({
                    "id": request_id,
                    "object": "response",
                    "created_at": timestamp,
                    "model": "mock-model",
                    "status": "in_progress",
                    "output": [{
                        "type": "message",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": "This is a mock responses streamed output."
                        }]
                    }]
                });
                Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
            })
            .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

            Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response()
        }
    } else if is_background {
        let rid = req_id.unwrap_or_else(|| format!("resp-{}", Uuid::now_v7()));
        Json(json!({
            "id": rid,
            "object": "response",
            "created_at": timestamp,
            "model": "mock-model",
            "output": [],
            "status": "queued",
            "usage": null
        }))
        .into_response()
    } else {
        // If tools are provided and this is the first call (no previous_response_id),
        // emit a single function_call to trigger the router's MCP flow.
        let has_tools = payload
            .get("tools")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter().any(|tool| {
                    tool.get("type")
                        .and_then(|t| t.as_str())
                        .map(|t| t == "function")
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);
        let has_function_output = payload
            .get("input")
            .and_then(|v| v.as_array())
            .map(|items| {
                items.iter().any(|item| {
                    item.get("type")
                        .and_then(|t| t.as_str())
                        .map(|t| t == "function_call_output")
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);
        let has_prior_tool_context = has_function_output;

        if has_tools && !has_prior_tool_context {
            let rid = format!("resp-{}", Uuid::now_v7());
            let call_id = format!("call_{}", &Uuid::now_v7().simple().to_string()[..24]);
            let item_id = format!("fc_{}", Uuid::now_v7().simple());
            Json(json!({
                "id": rid,
                "object": "response",
                "created_at": timestamp,
                "model": "mock-model",
                "output": [{
                    "type": "function_call",
                    "id": item_id,
                    "call_id": call_id,
                    "name": "brave_web_search",
                    "arguments": "{\"query\":\"SGLang router MCP integration\"}",
                    "status": "in_progress"
                }],
                "status": "in_progress",
                "usage": null
            }))
            .into_response()
        } else if has_tools && has_prior_tool_context {
            Json(json!({
                "id": format!("resp-{}", Uuid::now_v7()),
                "object": "response",
                "created_at": timestamp,
                "model": "mock-model",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": "Tool result consumed; here is the final answer."
                    }]
                }],
                "status": "completed",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 7,
                    "total_tokens": 19
                }
            }))
            .into_response()
        } else {
            Json(json!({
                "id": format!("resp-{}", Uuid::now_v7()),
                "object": "response",
                "created_at": timestamp,
                "model": "mock-model",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": "This is a mock responses output."
                    }]
                }],
                "status": "completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15
                }
            }))
            .into_response()
        }
    }
}

async fn flush_cache_handler(State(config): State<Arc<RwLock<MockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    if should_fail(&config) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    Json(json!({
        "message": "Cache flushed successfully"
    }))
    .into_response()
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn v1_models_handler(State(config): State<Arc<RwLock<MockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    if should_fail(&config) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Random failure for testing",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            })),
        )
            .into_response();
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Json(json!({
        "object": "list",
        "data": [{
            "id": "mock-model",
            "object": "model",
            "created": timestamp,
            "owned_by": "organization-owner"
        }]
    }))
    .into_response()
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn responses_get_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    Path(response_id): Path<String>,
) -> Response {
    let config = config.read().await;
    if should_fail(&config) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Random failure for testing" })),
        )
            .into_response();
    }
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    // Only return 200 if this worker "stores" the response id
    if response_exists_for_port(config.port, &response_id) {
        Json(json!({
            "id": response_id,
            "object": "response",
            "created_at": timestamp,
            "model": "mock-model",
            "output": [],
            "status": "completed",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
        }))
        .into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn responses_cancel_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    Path(response_id): Path<String>,
) -> Response {
    let config = config.read().await;
    if should_fail(&config) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": "Random failure for testing" })),
        )
            .into_response();
    }
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    if response_exists_for_port(config.port, &response_id) {
        Json(json!({
            "id": response_id,
            "object": "response",
            "created_at": timestamp,
            "model": "mock-model",
            "output": [],
            "status": "cancelled",
            "usage": null
        }))
        .into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

// --- Simple in-memory response store per worker port (for tests) ---
static RESP_STORE: OnceLock<Mutex<HashMap<u16, HashSet<String>>>> = OnceLock::new();

fn get_store() -> &'static Mutex<HashMap<u16, HashSet<String>>> {
    RESP_STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn store_response_for_port(port: u16, response_id: &str) {
    let mut map = get_store().lock().unwrap();
    map.entry(port).or_default().insert(response_id.to_string());
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn response_exists_for_port(port: u16, response_id: &str) -> bool {
    let map = get_store().lock().unwrap();
    map.get(&port)
        .map(|set| set.contains(response_id))
        .unwrap_or(false)
}

/// Minimal `/v1/audio/transcriptions` handler.
///
/// Collects the multipart text fields into a JSON object and records it, so a
/// test can assert on the model name the gateway put in the form. The audio
/// part itself is drained and discarded.
async fn audio_transcriptions_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    mut multipart: Multipart,
) -> Response {
    let config = config.read().await;

    let mut fields = serde_json::Map::new();
    while let Ok(Some(field)) = multipart.next_field().await {
        let Some(name) = field.name().map(ToOwned::to_owned) else {
            continue;
        };
        if name == "file" {
            let _ = field.bytes().await;
            continue;
        }
        if let Ok(value) = field.text().await {
            fields.insert(name, serde_json::Value::String(value));
        }
    }
    record_request(config.port, &serde_json::Value::Object(fields));

    if should_fail(&config) {
        return (StatusCode::INTERNAL_SERVER_ERROR, "Simulated failure").into_response();
    }

    Json(json!({"text": "mock transcription"})).into_response()
}

// Minimal rerank handler returning mock results; router shapes final response
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn rerank_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let config = config.read().await;
    record_request(config.port, &payload);

    // Simulate response delay
    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    // Simulate failure rate
    if rand::random::<f32>() < config.fail_rate {
        return (StatusCode::INTERNAL_SERVER_ERROR, "Simulated failure").into_response();
    }

    // Extract documents from the request to create mock results
    let empty_vec = vec![];
    let documents = payload
        .get("documents")
        .and_then(|d| d.as_array())
        .unwrap_or(&empty_vec);

    // Create mock rerank results with scores based on document index
    let mut mock_results = Vec::new();
    for (i, doc) in documents.iter().enumerate() {
        let score = 0.95 - (i as f32 * 0.1); // Decreasing scores

        // "PAD:<n>" documents are echoed back as n-byte strings so tests can
        // make the worker return a body far larger than the request.
        let doc = doc.as_str().unwrap_or("");
        let doc = match doc.strip_prefix("PAD:") {
            Some(n) => "x".repeat(n.parse().expect("PAD size must be a usize")),
            None => doc.to_string(),
        };
        let result = serde_json::json!({
            "score": score,
            "document": doc,
            "index": i,
            "meta_info": {
                "confidence": if score > 0.9 { "high" } else { "medium" }
            }
        });
        mock_results.push(result);
    }

    // Sort by score (highest first) to simulate proper ranking
    mock_results.sort_by(|a, b| {
        b["score"]
            .as_f64()
            .unwrap()
            .partial_cmp(&a["score"].as_f64().unwrap())
            .unwrap()
    });

    (StatusCode::OK, Json(mock_results)).into_response()
}

impl Default for MockWorkerConfig {
    fn default() -> Self {
        Self {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }
    }
}
