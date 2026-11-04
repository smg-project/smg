//! Integration tests for tenant rate limiting on the regular HTTP router
//! (`routers/http/router.rs`'s `route_typed_request`, the reservation
//! resolved in `routers/http/tenant_rate_limit.rs`), against a real (mock)
//! HTTP worker whose answer -- its status, its usage body, JSON or SSE --
//! each test sets.
//!
//! Like the gRPC suite (`tenant_rate_limiting_grpc_test.rs`), most cases
//! call `RouterTrait` methods directly with a tenant identity of their own;
//! one goes through the full app, to prove the body-path layer does not
//! stream a request past the typed route that holds it to its budget.
//!
//! This router has no tokenizer: the reserve-time estimate is a request's
//! own token ids, or 1.3 tokens per word of its text, rounded up, so
//! [`PROMPT_TEXT`]'s two words reserve [`ESTIMATED_INPUT_TOKENS`]. The budget cases run on the in-memory
//! backend a policy YAML configures; the cases that check exactly what a
//! request reserved and how its answer resolved it run on a backend that
//! records every call ([`RecordingBackend`]).

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helpers - panicking on failure is intentional"
)]

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use axum::{
    body::{to_bytes, Body, Bytes},
    extract::{Request, State},
    http::{
        header::{CONTENT_TYPE, RETRY_AFTER},
        StatusCode,
    },
    response::{IntoResponse, Response},
    routing::post,
    Json,
};
use common::test_app::create_test_app_with_context;
use llm_tokenizer::registry::TokenizerRegistry;
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    messages::{CountMessageTokensRequest, CreateMessageRequest},
    rerank::RerankRequest,
    responses::ResponsesRequest,
    worker::HealthCheckConfig,
};
use parking_lot::Mutex;
use serde_json::{json, Value};
use smg::{
    app_context::AppContext,
    config::{PolicyConfig, RetryConfig, RouterConfig},
    middleware::TenantRequestMeta,
    rate_limit::{
        RateLimitBackend, RateLimitManager, ReserveOutcome, ReserveRequest, UsageSettlement,
    },
    routers::{http::router::Router as HttpRouter, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ModelCard},
};
use tokio::{net::TcpListener, task::JoinHandle};
use tower::ServiceExt;
use uuid::Uuid;

use crate::common;

/// Two words: the reserve-time estimate is `ceil(2 * 1.3)` tokens.
pub(super) const PROMPT_TEXT: &str = "Hello world";
pub(super) const ESTIMATED_INPUT_TOKENS: u32 = 3;
const TEN_WORDS: &str = "one two three four five six seven eight nine ten";
pub(super) const MODEL: &str = "tenant-rl-http-test-model";

// ---------------------------------------------------------------------------
// The mock worker.
// ---------------------------------------------------------------------------

/// What the mock worker answers and what it was asked.
#[derive(Clone, Default)]
struct WorkerState {
    /// The answer's body: sent as JSON, or, to a request that asks to
    /// stream, as one SSE event followed by `[DONE]`.
    body: Arc<Mutex<Value>>,
    /// How many of the next requests are answered 503 instead.
    failures: Arc<AtomicUsize>,
    /// How many requests arrived.
    requests: Arc<AtomicUsize>,
}

/// A mock HTTP worker serving every route the tests call; stopped on drop.
pub(super) struct MockWorker {
    pub(super) url: String,
    state: WorkerState,
    task: JoinHandle<()>,
}

impl MockWorker {
    pub(super) fn answer(&self, body: Value) {
        *self.state.body.lock() = body;
    }

    pub(super) fn fail_next(&self, requests: usize) {
        self.state.failures.store(requests, Ordering::SeqCst);
    }

    pub(super) fn requests(&self) -> usize {
        self.state.requests.load(Ordering::SeqCst)
    }
}

impl Drop for MockWorker {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn answer(State(state): State<WorkerState>, body: Bytes) -> Response {
    state.requests.fetch_add(1, Ordering::SeqCst);
    if state
        .failures
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
        .is_ok()
    {
        return (StatusCode::SERVICE_UNAVAILABLE, "overloaded").into_response();
    }
    let streamed = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|request| request.get("stream")?.as_bool())
        .unwrap_or(false);
    let body = state.body.lock().clone();
    if streamed {
        (
            [(CONTENT_TYPE, "text/event-stream")],
            format!("data: {body}\n\ndata: [DONE]\n\n"),
        )
            .into_response()
    } else {
        Json(body).into_response()
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "the mock worker's task is aborted by the owning MockWorker on drop"
)]
pub(super) async fn mock_worker() -> MockWorker {
    let state = WorkerState::default();
    let app = [
        "/generate",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/embeddings",
        "/v1/classify",
        "/v1/responses",
        "/v1/rerank",
    ]
    .into_iter()
    .fold(axum::Router::new(), |app, path| {
        app.route(path, post(answer))
    })
    .with_state(state.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    MockWorker { url, state, task }
}

// ---------------------------------------------------------------------------
// The limiter and the router under test.
// ---------------------------------------------------------------------------

/// What a reservation asked for, kept past the call (the request borrows
/// its model id).
struct Reserved {
    estimated_input_tokens: u32,
    model_id: Option<String>,
}

/// A backend that admits every reservation and records every call made to
/// it: what was reserved, settled, closed unsettled and abandoned.
#[derive(Default)]
pub(super) struct RecordingBackend {
    reserves: Mutex<Vec<Reserved>>,
    settlements: Mutex<Vec<UsageSettlement>>,
    closes: AtomicUsize,
    abandons: AtomicUsize,
}

impl RecordingBackend {
    pub(super) fn estimates(&self) -> Vec<u32> {
        self.reserves
            .lock()
            .iter()
            .map(|request| request.estimated_input_tokens)
            .collect()
    }

    /// The model each reservation was keyed by.
    pub(super) fn models(&self) -> Vec<Option<String>> {
        self.reserves
            .lock()
            .iter()
            .map(|request| request.model_id.clone())
            .collect()
    }

    pub(super) fn settlements(&self) -> Vec<UsageSettlement> {
        self.settlements.lock().clone()
    }

    pub(super) fn closes(&self) -> usize {
        self.closes.load(Ordering::SeqCst)
    }

    pub(super) fn abandons(&self) -> usize {
        self.abandons.load(Ordering::SeqCst)
    }

    /// Wait for `n` abandons: a dropped body's abandon runs on a task of
    /// its own.
    pub(super) async fn await_abandons(&self, n: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while self.abandons() < n {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the dropped body abandons its reservation");
    }
}

#[async_trait]
impl RateLimitBackend for RecordingBackend {
    async fn reserve(&self, request: ReserveRequest<'_>) -> ReserveOutcome {
        self.reserves.lock().push(Reserved {
            estimated_input_tokens: request.estimated_input_tokens,
            model_id: request.model_id.map(str::to_owned),
        });
        ReserveOutcome::Admitted
    }

    async fn settle_success(&self, _request_charge_id: Uuid, usage: UsageSettlement) {
        self.settlements.lock().push(usage);
    }

    async fn close_reserved_only(&self, _request_charge_id: Uuid) {
        self.closes.fetch_add(1, Ordering::SeqCst);
    }

    async fn abandon(&self, _request_charge_id: Uuid) {
        self.abandons.fetch_add(1, Ordering::SeqCst);
    }
}

/// How the router under test limits its tenants.
pub(super) enum Limiter {
    Off,
    /// The in-memory backend under a policy of this many tokens per minute,
    /// and more requests per minute than any test makes.
    Tokens(u32),
    /// The in-memory backend under a default policy of 1,000 tokens per
    /// minute and a rule of this many for the one model named exactly.
    ModelRule(&'static str, u32),
    Recording(Arc<RecordingBackend>),
}

/// The context of a router under `limiter`, its retries on or off (on, they
/// back off for a millisecond), routing at random, with no worker yet, and
/// the tempdir of its policy YAML (kept for the test's duration).
pub(super) async fn limited_context(
    limiter: Limiter,
    retries: bool,
) -> (Arc<AppContext>, Option<tempfile::TempDir>) {
    limited_context_with_policy(limiter, retries, PolicyConfig::Random).await
}

/// [`limited_context`] routing by `policy`.
pub(super) async fn limited_context_with_policy(
    limiter: Limiter,
    retries: bool,
    policy: PolicyConfig,
) -> (Arc<AppContext>, Option<tempfile::TempDir>) {
    let mut builder = RouterConfig::builder()
        .regular_mode(vec![])
        .policy(policy)
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .request_timeout_secs(30)
        .retries(retries)
        .retry_config(RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 1,
            max_backoff_ms: 1,
            backoff_multiplier: 1.0,
            jitter_factor: 0.0,
        });
    let yaml = match &limiter {
        Limiter::Tokens(tokens_per_minute) => Some(format!(
            "default_policy:\n  tokens_per_minute: {tokens_per_minute}\n  \
             requests_per_minute: 1000\n"
        )),
        Limiter::ModelRule(model, tokens_per_minute) => Some(format!(
            "default_policy:\n  tokens_per_minute: 1000\n  requests_per_minute: 1000\n  \
             model_rules:\n    - rule_id: the-model\n      \
             matcher: {{type: exact, value: {model}}}\n      \
             tokens_per_minute: {tokens_per_minute}\n      requests_per_minute: 1000\n"
        )),
        Limiter::Off | Limiter::Recording(_) => None,
    };
    let dir = if let Some(yaml) = yaml {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rate_limit.yaml");
        std::fs::write(&path, yaml).unwrap();
        builder = builder
            .tenant_rate_limit_enabled(true)
            .tenant_rate_limit_config(Some(path.to_str().unwrap().to_string()));
        Some(dir)
    } else {
        None
    };
    let mut config = builder.build_unchecked();
    config.health_check.disable_health_check = true;

    let ctx = match limiter {
        Limiter::Recording(backend) => {
            common::create_test_context_with_rate_limiter(
                config,
                Arc::new(TokenizerRegistry::new()),
                Some(Arc::new(RateLimitManager::new(backend))),
            )
            .await
        }
        Limiter::Off | Limiter::Tokens(_) | Limiter::ModelRule(..) => {
            common::create_test_context(config).await
        }
    };
    (ctx, dir)
}

/// [`limited_context`] with `worker` registered as its one regular worker.
pub(super) async fn context(
    limiter: Limiter,
    worker: &MockWorker,
    retries: bool,
) -> (Arc<AppContext>, Option<tempfile::TempDir>) {
    let (ctx, dir) = limited_context(limiter, retries).await;
    ctx.worker_registry
        .register(Arc::new(
            BasicWorkerBuilder::new(&worker.url)
                .model(ModelCard::new(MODEL))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        ))
        .unwrap();
    (ctx, dir)
}

/// The regular HTTP router over `worker` (see [`context`]).
pub(super) async fn router(
    limiter: Limiter,
    worker: &MockWorker,
    retries: bool,
) -> (HttpRouter, Option<tempfile::TempDir>) {
    let (ctx, dir) = context(limiter, worker, retries).await;
    (HttpRouter::new(&ctx).await.unwrap(), dir)
}

pub(super) fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("integration-test-tenant"))
}

pub(super) async fn body_of(response: Response) -> String {
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

// ---------------------------------------------------------------------------
// The requests.
// ---------------------------------------------------------------------------

/// An endpoint the HTTP router reserves on.
#[derive(Clone, Copy, Debug)]
pub(super) enum Endpoint {
    Generate,
    Chat,
    Completion,
    Messages,
    Embeddings,
    Classify,
    Responses,
}

impl Endpoint {
    /// The endpoints that stream when asked.
    pub(super) const STREAMING: [Self; 5] = [
        Self::Generate,
        Self::Chat,
        Self::Completion,
        Self::Messages,
        Self::Responses,
    ];

    /// One request of [`PROMPT_TEXT`] through `router`, streamed or not.
    pub(super) async fn call(self, router: &dyn RouterTrait, stream: bool) -> Response {
        let tenant = tenant();
        match self {
            Self::Generate => {
                let body = request::<GenerateRequest>(json!({
                    "model": MODEL, "text": PROMPT_TEXT, "stream": stream,
                }));
                router.route_generate(None, &tenant, body, MODEL).await
            }
            Self::Chat => {
                let body = request::<ChatCompletionRequest>(json!({
                    "model": MODEL,
                    "messages": [{"role": "user", "content": PROMPT_TEXT}],
                    "stream": stream,
                }));
                router.route_chat(None, &tenant, body, MODEL).await
            }
            Self::Completion => {
                let body = request::<CompletionRequest>(json!({
                    "model": MODEL, "prompt": PROMPT_TEXT, "stream": stream,
                }));
                router.route_completion(None, &tenant, body, MODEL).await
            }
            Self::Messages => {
                let body = request::<CreateMessageRequest>(json!({
                    "model": MODEL,
                    "messages": [{"role": "user", "content": PROMPT_TEXT}],
                    "max_tokens": 32,
                    "stream": stream,
                }));
                router.route_messages(None, &tenant, body, MODEL).await
            }
            Self::Embeddings => {
                let body = request::<EmbeddingRequest>(json!({
                    "model": MODEL, "input": PROMPT_TEXT,
                }));
                router.route_embeddings(None, &tenant, body, MODEL).await
            }
            Self::Classify => {
                let body = request::<ClassifyRequest>(json!({
                    "model": MODEL, "input": PROMPT_TEXT,
                }));
                router.route_classify(None, &tenant, body, MODEL).await
            }
            Self::Responses => {
                let body = request::<ResponsesRequest>(json!({
                    "model": MODEL, "input": PROMPT_TEXT, "stream": stream, "store": false,
                }));
                router.route_responses(None, &tenant, body, MODEL).await
            }
        }
    }

    /// The usage body a worker answers this endpoint with, reporting
    /// `input` and `output` tokens in the endpoint's own shape. Embeddings
    /// and classify generate nothing and report no `output`, as OpenAI's
    /// embeddings usage has no `completion_tokens`.
    pub(super) fn usage_body(self, input: u32, output: u32) -> Value {
        match self {
            Self::Generate => json!({
                "text": "hi",
                "meta_info": {"prompt_tokens": input, "completion_tokens": output},
            }),
            Self::Chat | Self::Completion => json!({
                "usage": {
                    "prompt_tokens": input,
                    "completion_tokens": output,
                    "total_tokens": input + output,
                },
            }),
            Self::Embeddings | Self::Classify => json!({
                "usage": {"prompt_tokens": input, "total_tokens": input},
            }),
            Self::Messages | Self::Responses => json!({
                "usage": {"input_tokens": input, "output_tokens": output},
            }),
        }
    }

    /// What [`Self::usage_body`] settles.
    pub(super) fn settlement(self, input: u32, output: u32) -> UsageSettlement {
        match self {
            Self::Embeddings | Self::Classify => usage(input, 0),
            _ => usage(input, output),
        }
    }
}

pub(super) fn request<T: serde::de::DeserializeOwned>(body: Value) -> T {
    serde_json::from_value(body).unwrap()
}

pub(super) fn usage(input: u32, output: u32) -> UsageSettlement {
    UsageSettlement {
        actual_input_tokens: input,
        completion_tokens: output,
        ..UsageSettlement::default()
    }
}

// ---------------------------------------------------------------------------
// Admission.
// ---------------------------------------------------------------------------

/// A denial the tenant can wait out carries `Retry-After`, and never
/// reaches the worker. A fresh bucket starts full, so the first request
/// spends the budget: it reserves 3 of 6 and settles the worker's 1 + 4,
/// leaving 1, short of the second request's 3.
#[tokio::test]
async fn a_denial_the_tenant_can_wait_out_carries_retry_after() {
    let worker = mock_worker().await;
    worker.answer(Endpoint::Generate.usage_body(1, 4));
    let (router, _dir) = router(Limiter::Tokens(6), &worker, true).await;

    let first = Endpoint::Generate.call(&router, false).await;
    assert_eq!(first.status(), StatusCode::OK);

    let second = Endpoint::Generate.call(&router, false).await;
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(
        second.headers().get(RETRY_AFTER).is_some(),
        "a finite-wait denial carries Retry-After"
    );
    assert_eq!(worker.requests(), 1, "a denied request is not dispatched");
}

/// A request whose estimate exceeds the tenant's whole budget can never be
/// admitted: its denial carries no `Retry-After` (the backend's "never"
/// sentinel must not leak into the header), whichever path dispatch takes.
#[tokio::test]
async fn an_impossible_request_is_denied_without_retry_after() {
    for retries in [false, true] {
        let worker = mock_worker().await;
        let (router, _dir) = router(Limiter::Tokens(1), &worker, retries).await;

        let response = Endpoint::Chat.call(&router, false).await;
        assert_eq!(
            response.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "retries={retries}"
        );
        assert!(
            response.headers().get(RETRY_AFTER).is_none(),
            "retries={retries}: an impossible request's denial carries no Retry-After"
        );
        assert_eq!(worker.requests(), 0, "retries={retries}");
    }
}

/// A request reserves every input it carries, past what routing reads: each
/// sequence of a tokenized batch, the token ids of embeddings and classify,
/// a Responses request's instructions and custom tool calls, a chat
/// history's tool-call arguments, a Messages history's tool results.
#[tokio::test]
async fn every_input_the_request_carries_is_reserved() {
    let worker = mock_worker().await;
    let backend = Arc::new(RecordingBackend::default());
    let (router, _dir) = router(Limiter::Recording(backend.clone()), &worker, true).await;
    let tenant = tenant();

    worker.answer(Endpoint::Generate.usage_body(1, 1));
    let batch = request::<GenerateRequest>(json!({
        "model": MODEL, "input_ids": [[1, 2], [3, 4, 5, 6]],
    }));
    router.route_generate(None, &tenant, batch, MODEL).await;

    worker.answer(Endpoint::Embeddings.usage_body(1, 0));
    let ids = request::<EmbeddingRequest>(json!({"model": MODEL, "input": [1, 2, 3, 4]}));
    router.route_embeddings(None, &tenant, ids, MODEL).await;
    let sequences = request::<ClassifyRequest>(json!({"model": MODEL, "input": [[1, 2], [3]]}));
    router.route_classify(None, &tenant, sequences, MODEL).await;

    worker.answer(Endpoint::Responses.usage_body(1, 1));
    let instructed = request::<ResponsesRequest>(json!({
        "model": MODEL,
        "instructions": TEN_WORDS,
        "input": PROMPT_TEXT,
        "store": false,
    }));
    router
        .route_responses(None, &tenant, instructed, MODEL)
        .await;
    let custom = request::<ResponsesRequest>(json!({
        "model": MODEL,
        "input": [
            {"type": "message", "role": "user", "content": PROMPT_TEXT},
            {"type": "custom_tool_call", "call_id": "c1", "name": "grep", "input": TEN_WORDS},
            {"type": "custom_tool_call_output", "call_id": "c1", "output": TEN_WORDS},
        ],
        "store": false,
    }));
    router.route_responses(None, &tenant, custom, MODEL).await;

    worker.answer(Endpoint::Chat.usage_body(1, 1));
    let tool_call = request::<ChatCompletionRequest>(json!({
        "model": MODEL,
        "messages": [
            {"role": "user", "content": PROMPT_TEXT},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "grep", "arguments": TEN_WORDS},
            }]},
        ],
    }));
    router.route_chat(None, &tenant, tool_call, MODEL).await;

    worker.answer(Endpoint::Messages.usage_body(1, 1));
    let tool_result = request::<CreateMessageRequest>(json!({
        "model": MODEL,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT_TEXT},
            {"type": "tool_result", "tool_use_id": "t1", "content": TEN_WORDS},
        ]}],
    }));
    router
        .route_messages(None, &tenant, tool_result, MODEL)
        .await;

    // Ten words of instructions and two of input: ceil(12 * 1.3). Two words
    // of input and ten of a custom tool's input and ten of its output:
    // ceil(22 * 1.3). Two words and ten of a tool call or a tool result.
    assert_eq!(backend.estimates(), vec![6, 4, 3, 16, 29, 16, 16]);
}

// ---------------------------------------------------------------------------
// Settlement.
// ---------------------------------------------------------------------------

/// A buffered success settles the counts the worker's body reports, in each
/// endpoint's own shape, once, over the request's one reservation of its
/// estimate; the client receives the worker's body unchanged. On both
/// dispatch paths: the one attempt of a request without retries, and the
/// retry loop.
#[tokio::test]
async fn a_buffered_success_settles_the_usage_its_body_reports() {
    for retries in [false, true] {
        for endpoint in [
            Endpoint::Generate,
            Endpoint::Chat,
            Endpoint::Completion,
            Endpoint::Messages,
            Endpoint::Embeddings,
            Endpoint::Classify,
            Endpoint::Responses,
        ] {
            let worker = mock_worker().await;
            let answer = endpoint.usage_body(11, 7);
            worker.answer(answer.clone());
            let backend = Arc::new(RecordingBackend::default());
            let (router, _dir) =
                router(Limiter::Recording(backend.clone()), &worker, retries).await;

            let response = endpoint.call(&router, false).await;
            let case = format!("{endpoint:?} retries={retries}");
            assert_eq!(response.status(), StatusCode::OK, "{case}");
            assert_eq!(
                serde_json::from_str::<Value>(&body_of(response).await).unwrap(),
                answer,
                "{case}: the client gets the worker's body"
            );
            assert_eq!(backend.estimates(), vec![ESTIMATED_INPUT_TOKENS], "{case}");
            assert_eq!(
                backend.settlements(),
                vec![endpoint.settlement(11, 7)],
                "{case}"
            );
            assert_eq!((backend.closes(), backend.abandons()), (0, 0), "{case}");
        }
    }
}

/// A `/generate` answer of several entries settles every completion and
/// each prompt once: the prompts of a batch sum, and the `n` samples of one
/// prompt repeat its count.
#[tokio::test]
async fn a_generate_batch_settles_each_prompt_once_and_every_completion() {
    for (body, answer, settled) in [
        (
            json!({"model": MODEL, "input_ids": [[1, 2], [3, 4, 5, 6]]}),
            json!([
                {"text": "a", "meta_info": {"prompt_tokens": 2, "completion_tokens": 9}},
                {"text": "b", "meta_info": {"prompt_tokens": 4, "completion_tokens": 2}},
            ]),
            usage(6, 11),
        ),
        (
            json!({"model": MODEL, "text": PROMPT_TEXT, "sampling_params": {"n": 2}}),
            json!([
                {"text": "a", "meta_info": {"prompt_tokens": 4, "completion_tokens": 9}},
                {"text": "b", "meta_info": {"prompt_tokens": 4, "completion_tokens": 2}},
            ]),
            usage(4, 11),
        ),
    ] {
        let worker = mock_worker().await;
        worker.answer(answer);
        let backend = Arc::new(RecordingBackend::default());
        let (router, _dir) = router(Limiter::Recording(backend.clone()), &worker, true).await;

        let request = request::<GenerateRequest>(body.clone());
        let response = router.route_generate(None, &tenant(), request, MODEL).await;
        assert_eq!(response.status(), StatusCode::OK, "{body}");
        assert_eq!(backend.settlements(), vec![settled], "{body}");
    }
}

/// A Messages answer's `input_tokens` leave out the cached prefix: it
/// settles them with the tokens read from the cache and written to it.
#[tokio::test]
async fn a_cached_messages_prompt_settles_whole() {
    let worker = mock_worker().await;
    worker.answer(json!({"usage": {
        "input_tokens": 50,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 50,
        "output_tokens": 7,
    }}));
    let backend = Arc::new(RecordingBackend::default());
    let (router, _dir) = router(Limiter::Recording(backend.clone()), &worker, true).await;

    let response = Endpoint::Messages.call(&router, false).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(backend.settlements(), vec![usage(1000, 7)]);
    assert_eq!((backend.closes(), backend.abandons()), (0, 0));
}

/// A success whose body reports no usage this router can read whole closes
/// its reservation at the estimate: no settlement, one close, nothing left
/// open for the cell's drop to abandon. The client still gets the body.
#[tokio::test]
async fn usage_the_answer_does_not_report_whole_closes_at_the_estimate() {
    for (endpoint, answer) in [
        (Endpoint::Chat, json!({"choices": []})),
        (Endpoint::Chat, json!({"usage": {"prompt_tokens": 5}})),
        (Endpoint::Messages, json!({"usage": null})),
        (
            Endpoint::Generate,
            json!({"usage": {"prompt_tokens": 5, "completion_tokens": 1}}),
        ),
        (Endpoint::Completion, json!("not an object")),
    ] {
        let worker = mock_worker().await;
        worker.answer(answer.clone());
        let backend = Arc::new(RecordingBackend::default());
        let (router, _dir) = router(Limiter::Recording(backend.clone()), &worker, true).await;

        let response = endpoint.call(&router, false).await;
        let case = format!("{endpoint:?} answering {answer}");
        assert_eq!(response.status(), StatusCode::OK, "{case}");
        assert_eq!(
            serde_json::from_str::<Value>(&body_of(response).await).unwrap(),
            answer,
            "{case}"
        );
        tokio::task::yield_now().await;
        assert!(backend.settlements().is_empty(), "{case}");
        assert_eq!((backend.closes(), backend.abandons()), (1, 0), "{case}");
    }
}

/// The same on the in-memory backend: an answer with no usage keeps
/// exactly its 3-token estimate. A budget of 6 admits two such requests
/// and refuses the third: had a request kept nothing (an open reservation
/// refunded) the third would fit, had it kept more the second would not.
#[tokio::test]
async fn an_answer_without_usage_keeps_exactly_its_estimate() {
    let worker = mock_worker().await;
    worker.answer(json!({"choices": []}));
    let (router, _dir) = router(Limiter::Tokens(2 * ESTIMATED_INPUT_TOKENS), &worker, true).await;

    for request in 1..=2 {
        let response = Endpoint::Chat.call(&router, false).await;
        assert_eq!(response.status(), StatusCode::OK, "request {request} fits");
    }
    let third = Endpoint::Chat.call(&router, false).await;
    assert_eq!(third.status(), StatusCode::TOO_MANY_REQUESTS);
}

/// A streamed answer is relayed as the worker sent it and never parsed: it
/// settles nothing, even with a usage chunk in the stream, and its
/// reservation resolves when the body drops, keeping its estimate (the
/// abandon of an open reservation keeps the debit).
#[tokio::test]
async fn a_streamed_answer_resolves_when_its_body_drops() {
    for endpoint in Endpoint::STREAMING {
        let worker = mock_worker().await;
        worker.answer(endpoint.usage_body(1000, 1000));
        let backend = Arc::new(RecordingBackend::default());
        let (router, _dir) = router(Limiter::Recording(backend.clone()), &worker, true).await;

        let response = endpoint.call(&router, true).await;
        assert_eq!(response.status(), StatusCode::OK, "{endpoint:?}");
        assert_eq!(
            (backend.closes(), backend.abandons()),
            (0, 0),
            "{endpoint:?}: the body holds the reservation while it lives"
        );

        let body = body_of(response).await;
        assert!(
            body.contains("1000") && body.ends_with("data: [DONE]\n\n"),
            "{endpoint:?}: the stream is relayed whole: {body}"
        );
        backend.await_abandons(1).await;
        assert_eq!(
            backend.estimates(),
            vec![ESTIMATED_INPUT_TOKENS],
            "{endpoint:?}"
        );
        assert!(backend.settlements().is_empty(), "{endpoint:?}");
        assert_eq!(backend.closes(), 0, "{endpoint:?}");
    }
}

/// The same on the in-memory backend: each streamed request keeps exactly
/// its 3-token estimate, whatever usage its stream reports. A budget of 9
/// admits three and refuses the fourth: kept more, the third would not fit;
/// kept less than 3 (2 or under), the fourth would.
#[tokio::test]
async fn a_streamed_answer_keeps_exactly_its_estimate() {
    let worker = mock_worker().await;
    worker.answer(Endpoint::Chat.usage_body(1000, 1000));
    let (router, _dir) = router(Limiter::Tokens(3 * ESTIMATED_INPUT_TOKENS), &worker, true).await;

    for request in 1..=3 {
        let response = Endpoint::Chat.call(&router, true).await;
        assert_eq!(response.status(), StatusCode::OK, "request {request} fits");
        body_of(response).await;
    }
    let fourth = Endpoint::Chat.call(&router, true).await;
    assert_eq!(fourth.status(), StatusCode::TOO_MANY_REQUESTS);
}

// ---------------------------------------------------------------------------
// Retries and failures.
// ---------------------------------------------------------------------------

/// A request the router retries reserves once, before its first attempt,
/// and its final answer settles that one reservation.
#[tokio::test]
async fn a_retried_request_reserves_once_and_settles_its_final_answer() {
    let worker = mock_worker().await;
    worker.answer(Endpoint::Chat.usage_body(11, 7));
    worker.fail_next(1);
    let backend = Arc::new(RecordingBackend::default());
    let (router, _dir) = router(Limiter::Recording(backend.clone()), &worker, true).await;

    let response = Endpoint::Chat.call(&router, false).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(worker.requests(), 2, "the 503 was retried");
    assert_eq!(backend.estimates(), vec![ESTIMATED_INPUT_TOKENS]);
    assert_eq!(backend.settlements(), vec![usage(11, 7)]);
    assert_eq!((backend.closes(), backend.abandons()), (0, 0));
}

/// A request whose every attempt fails closes its one reservation at the
/// estimate, on both dispatch paths.
#[tokio::test]
async fn a_failed_request_closes_its_reservation_at_the_estimate() {
    for retries in [false, true] {
        let worker = mock_worker().await;
        worker.fail_next(usize::MAX);
        let backend = Arc::new(RecordingBackend::default());
        let (router, _dir) = router(Limiter::Recording(backend.clone()), &worker, retries).await;

        let response = Endpoint::Generate.call(&router, false).await;
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "retries={retries}"
        );
        tokio::task::yield_now().await;
        assert_eq!(
            backend.estimates(),
            vec![ESTIMATED_INPUT_TOKENS],
            "retries={retries}"
        );
        assert!(backend.settlements().is_empty(), "retries={retries}");
        assert_eq!(
            (backend.closes(), backend.abandons()),
            (1, 0),
            "retries={retries}"
        );
    }
}

// ---------------------------------------------------------------------------
// Scope.
// ---------------------------------------------------------------------------

/// Without a tenant policy every request is served as before, streamed or
/// not.
#[tokio::test]
async fn without_a_tenant_policy_requests_are_served_as_before() {
    let worker = mock_worker().await;
    worker.answer(Endpoint::Chat.usage_body(1000, 1000));
    let (router, _dir) = router(Limiter::Off, &worker, true).await;

    for stream in [false, true] {
        for _ in 0..3 {
            let response = Endpoint::Chat.call(&router, stream).await;
            assert_eq!(response.status(), StatusCode::OK, "stream={stream}");
            body_of(response).await;
        }
    }
}

/// The endpoints outside the limiter's scope dispatch without reserving.
#[tokio::test]
async fn an_opted_out_endpoint_never_reserves() {
    let worker = mock_worker().await;
    worker.answer(json!([{"score": 0.5, "document": "d", "index": 0}]));
    let backend = Arc::new(RecordingBackend::default());
    let (router, _dir) = router(Limiter::Recording(backend.clone()), &worker, true).await;

    let rerank = request::<RerankRequest>(json!({
        "model": MODEL, "query": PROMPT_TEXT, "documents": ["d"],
    }));
    let response = router.route_rerank(None, &tenant(), rerank, MODEL).await;
    assert_eq!(response.status(), StatusCode::OK);

    worker.answer(json!({"input_tokens": 3}));
    let count = request::<CountMessageTokensRequest>(json!({
        "model": MODEL, "messages": [{"role": "user", "content": PROMPT_TEXT}],
    }));
    let response = router
        .route_messages_count_tokens(None, &tenant(), count, MODEL)
        .await;
    assert_eq!(response.status(), StatusCode::OK);

    assert_eq!(worker.requests(), 2, "both were dispatched");
    assert!(backend.estimates().is_empty(), "neither reserved");
}

/// Through the full app, a request the body-path layer would otherwise
/// stream to the worker unread (retries off, one model, a policy that needs
/// no text, a known length) is buffered for the typed route, which holds it
/// to the tenant's budget: one token a minute refuses it.
#[tokio::test]
async fn the_app_does_not_stream_a_request_past_the_tenant_budget() {
    let worker = mock_worker().await;
    worker.answer(Endpoint::Chat.usage_body(1, 1));
    let (ctx, _dir) = context(Limiter::Tokens(1), &worker, false).await;
    let router = Arc::new(HttpRouter::new(&ctx).await.unwrap());
    let app = create_test_app_with_context(router, ctx);

    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT_TEXT}],
    })
    .to_string();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .header("content-length", body.len())
        .body(Body::from(body))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(worker.requests(), 0);
}
