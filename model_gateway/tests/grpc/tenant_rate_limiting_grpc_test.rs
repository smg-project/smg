//! Integration tests for tenant rate limiting wired into the gRPC router
//! (`routers/grpc/common/stages/rate_limit.rs`), verified end-to-end against
//! a real (mock) gRPC backend rather than by driving `RateLimitReserveStage`
//! in isolation.
//!
//! Calls `RouterTrait` methods directly, mirroring
//! `routers/grpc/router.rs`'s own `pd_tests` module, rather than going
//! through the full HTTP app: `test_harness/src/test_app.rs`'s lightweight test
//! app doesn't wire the tenant-resolution middleware that would otherwise
//! populate `tenant_request_meta`, so a real per-request tenant identity is
//! constructed here directly, same as the real middleware would produce.
//!
//! Determinism relies on `mock-worker`'s canned (non-`realistic`) gRPC mode:
//! it always reports `prompt_tokens: 1` in its final response regardless of
//! the real input length, while `completion_tokens` is exactly the
//! configured `output_tokens`. Reserve-time debits use the gateway's own
//! tokenization (via `MockTokenizer`'s tiny fixed vocab); settle-time trueup
//! uses the backend's reported usage. Choosing a token budget between those
//! two numbers proves settle actually ran with real backend-reported usage,
//! not just the initial reserve estimate.

use std::{sync::Arc, time::Duration};

use common::{
    grpc_harness::{
        grpc_router, grpc_router_config, grpc_test_context_with_tokenizer, grpc_worker,
        serve_canned_worker, serve_engine, Canned,
    },
    mock_mcp_server::MockMCPServer,
    scripted_tokenizer::ScriptedTokenizer,
    scripted_worker::ScriptedWorker,
};
use llm_tokenizer::{
    traits::{MediaHandling, RendererCapabilities, Tokenizer},
    MockTokenizer,
};
use openai_protocol::{
    chat::ChatCompletionRequest,
    classify::ClassifyRequest,
    completion::CompletionRequest,
    embedding::EmbeddingRequest,
    generate::GenerateRequest,
    model_card::ModelCard,
    responses::ResponsesRequest,
    transcription::{AudioFile, TranscriptionRequest},
};
use serde_json::{json, Value};
use smg::{
    config::RouterConfig,
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, Worker, WorkerType},
};
use smg_grpc_client::{
    tokenspeed_scheduler::tokenspeed_proto::token_speed_scheduler_server::TokenSpeedSchedulerServer,
    vllm_proto as proto,
};
use tokio::time::timeout;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::common;

/// "Hello world" through `MockTokenizer`'s fixed vocab ("Hello"=1, "world"=2)
/// tokenizes to exactly 2 ids -- the gateway's reserve-time estimate.
const PROMPT_TEXT: &str = "Hello world";
const ESTIMATED_INPUT_TOKENS: u32 = 2;
const MODEL: &str = "tenant-rl-test-model";

/// Spawn a real (mock) gRPC worker in-process, canned mode: always reports
/// `prompt_tokens: 1` and `completion_tokens: output_tokens`, regardless of
/// what was actually sent.
async fn start_mock_grpc_worker(output_tokens: u32) -> u16 {
    serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens,
    })
    .await
}

/// The tenant's budget of a test, the one the policy's other limit never
/// decides: `Tokens` is the tokens per minute under 1000 requests per
/// minute, `Requests` the requests per minute under a token budget nothing
/// here can spend.
#[derive(Clone, Copy)]
enum Budget {
    Tokens(u32),
    Requests(u32),
}

impl Budget {
    fn policy_yaml(self) -> String {
        let (tokens_per_minute, requests_per_minute) = match self {
            Budget::Tokens(tokens) => (tokens, 1000),
            Budget::Requests(requests) => (1_000_000, requests),
        };
        format!(
            "default_policy:\n  tokens_per_minute: {tokens_per_minute}\n  \
             requests_per_minute: {requests_per_minute}\n"
        )
    }
}

/// The router configuration of these tests: tenant rate limiting against a
/// tiny in-memory [`Budget`] when one is given, none otherwise. Returns the
/// config and its backing tempdir (must outlive the call -- the YAML is read
/// synchronously during router construction, but keeping it bound for the
/// caller's whole test avoids any doubt about timing).
#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn router_config(budget: Option<Budget>) -> (RouterConfig, Option<tempfile::TempDir>) {
    let mut builder = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60);

    let dir = if let Some(budget) = budget {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rate_limit.yaml");
        std::fs::write(&path, budget.policy_yaml()).unwrap();
        builder = builder
            .tenant_rate_limit_enabled(true)
            .tenant_rate_limit_config(Some(path.to_str().unwrap().to_string()));
        Some(dir)
    } else {
        None
    };
    (builder.build_unchecked(), dir)
}

/// A real gRPC router over `worker`, serving `model` with `tokenizer`, under
/// [`router_config`].
async fn build_router_over(
    budget: Option<Budget>,
    worker: Arc<dyn Worker>,
    model: &str,
    tokenizer: Arc<dyn Tokenizer>,
) -> (Box<dyn RouterTrait>, Option<tempfile::TempDir>) {
    let (config, dir) = router_config(budget);
    let app_context = grpc_test_context_with_tokenizer(config, model, tokenizer).await;
    (grpc_router(&app_context, vec![worker]).await, dir)
}

/// Build a real gRPC router with tenant rate limiting configured against a
/// tiny in-memory budget, and a real mock gRPC worker registered.
async fn build_router(
    budget: Option<Budget>,
    grpc_port: u16,
) -> (Box<dyn RouterTrait>, Option<tempfile::TempDir>) {
    let worker = grpc_worker(
        grpc_port,
        WorkerType::Regular,
        RuntimeType::TokenSpeed,
        ModelCard::new(MODEL),
    )
    .build();
    build_router_over(
        budget,
        Arc::new(worker),
        MODEL,
        Arc::new(MockTokenizer::new()),
    )
    .await
}

fn generate_request() -> GenerateRequest {
    generate_request_for(MODEL, PROMPT_TEXT)
}

/// A generate request of `text` to `model`; the reserve estimate is the
/// number of `text`'s words the mock vocabulary knows.
#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn generate_request_for(model: &str, text: &str) -> GenerateRequest {
    serde_json::from_value(json!({
        "model": model,
        "text": text,
        "stream": false,
    }))
    .unwrap()
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn completion_request() -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": PROMPT_TEXT,
        "stream": false,
    }))
    .unwrap()
}

fn fresh_tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("integration-test-tenant"))
}

/// A denial with a finite wait (the request *could* eventually fit, just
/// not right now) must carry `Retry-After`. A fresh bucket always starts
/// full, so a finite-wait denial requires a prior request to have already
/// consumed some of the budget -- unlike the "can never fit, no matter how
/// long you wait" case (see `impossible_request_denies_without_retry_after`
/// below), which can be hit on the very first request.
#[tokio::test]
async fn denial_returns_429_with_retry_after() {
    let grpc_port = start_mock_grpc_worker(4).await;
    let (router, _dir) = build_router(Some(Budget::Tokens(6)), grpc_port).await;
    let request = generate_request();

    let first = router
        .route_generate(None, &fresh_tenant(), request.clone(), MODEL)
        .await;
    assert_eq!(first.status(), http::StatusCode::OK);

    // Same budget arithmetic as the settle test below: after settling the
    // first request's real usage, only 1 token remains -- not enough for a
    // second 2-token reserve, but well within the 6-token capacity, so this
    // is a genuine "wait and retry" denial, not an impossible one.
    let second = router
        .route_generate(None, &fresh_tenant(), request, MODEL)
        .await;
    assert_eq!(second.status(), http::StatusCode::TOO_MANY_REQUESTS);
    assert!(
        second.headers().get(http::header::RETRY_AFTER).is_some(),
        "a finite-wait denial must carry Retry-After"
    );
}

/// A request whose estimated cost exceeds the tenant's *total* budget can
/// never be admitted, no matter how long the client waits -- the backend's
/// `u64::MAX` "never" sentinel must not leak into the HTTP response as a
/// literal ~584-billion-year `Retry-After` value.
#[tokio::test]
async fn impossible_request_denies_without_retry_after() {
    let grpc_port = start_mock_grpc_worker(4).await;
    // 1 token/minute can never admit this request's 2-token reserve, ever.
    let (router, _dir) = build_router(Some(Budget::Tokens(1)), grpc_port).await;

    let request = generate_request();
    let response = router
        .route_generate(None, &fresh_tenant(), request, MODEL)
        .await;

    assert_eq!(response.status(), http::StatusCode::TOO_MANY_REQUESTS);
    assert!(
        response.headers().get(http::header::RETRY_AFTER).is_none(),
        "an impossible request's 'never' sentinel must not leak into Retry-After"
    );
}

/// Proves settle_success runs with the backend's real reported usage, not
/// just the reserve-time estimate. Canned mock backend always reports
/// prompt_tokens=1 and completion_tokens=OUTPUT_TOKENS regardless of what
/// was actually sent, so real settled usage (1 + OUTPUT_TOKENS) is provably
/// larger than the reserve-time estimate alone (ESTIMATED_INPUT_TOKENS).
///
/// Budget=6: request 1 reserves 2 (estimate), settles to a true cost of
/// 1+4=5 (trueup delta +3), leaving 1 token remaining. Request 2 needs to
/// reserve another 2 estimated tokens -- only 1 remains, so it's denied. If
/// settle had never run (a regression back to "just reserve, no trueup"),
/// only 2 tokens would ever have been spent, leaving 4 remaining, and
/// request 2 (needing 2) would have been admitted instead.
#[tokio::test]
async fn non_streaming_settle_uses_real_usage_not_just_the_estimate() {
    const OUTPUT_TOKENS: u32 = 4;
    let grpc_port = start_mock_grpc_worker(OUTPUT_TOKENS).await;
    let (router, _dir) = build_router(Some(Budget::Tokens(6)), grpc_port).await;

    let request = generate_request();

    let first = router
        .route_generate(None, &fresh_tenant(), request.clone(), MODEL)
        .await;
    assert_eq!(
        first.status(),
        http::StatusCode::OK,
        "first request should be admitted and dispatched successfully"
    );

    let second = router
        .route_generate(None, &fresh_tenant(), request, MODEL)
        .await;
    assert_eq!(
        second.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "second request should be denied -- only true if settle used real \
         usage (1 + {OUTPUT_TOKENS} completion) rather than leaving only the \
         {ESTIMATED_INPUT_TOKENS}-token reserve estimate debited"
    );
}

/// With tenant rate limiting disabled entirely, requests must behave exactly
/// as they did before this feature existed -- no interference.
#[tokio::test]
async fn feature_disabled_is_a_no_op() {
    let grpc_port = start_mock_grpc_worker(4).await;
    let (router, _dir) = build_router(None, grpc_port).await;

    let request = generate_request();
    let response = router
        .route_generate(None, &fresh_tenant(), request, MODEL)
        .await;

    assert_eq!(response.status(), http::StatusCode::OK);
}

/// `RateLimitReserveStage` is inserted identically into the Chat, Messages,
/// Completion, and Generate pipelines (`pipeline.rs::build()`), and the
/// tests above already prove the mechanism end-to-end (real reserve, real
/// settle, real denial with/without `Retry-After`) through Generate. This
/// mirrors that same impossible-budget proof through Completion, to confirm
/// the stage is actually live on a second wired pipeline too, not just
/// constructed and never reached.
///
/// Chat and Messages are not covered the same way here. `MockTokenizer`
/// (`crates/tokenizer/src/mock.rs`) renders a minimal `role: content`
/// template, so a `route_chat` mirror of this case would reach the reserve
/// stage; the mechanics it would exercise are the ones Completion already
/// proves. Harmony and PD dispatch are also not covered:
/// Harmony needs a harmony-flavored model registered so `HarmonyDetector`
/// routes into it, and PD needs a distinct prefill+decode worker pair --
/// both larger harness investments than this suite carries for its other
/// cases, and the underlying reserve/settle mechanics they'd exercise are
/// identical to what's already proven above and in `pipeline.rs`'s own
/// `rate_limit_reserve_tests` (which drives `RateLimitReserveStage` directly
/// rather than through a real backend).
#[tokio::test]
async fn completion_denial_reaches_the_client() {
    let grpc_port = start_mock_grpc_worker(4).await;
    let (router, _dir) = build_router(Some(Budget::Tokens(1)), grpc_port).await;

    let request = completion_request();
    let response = router
        .route_completion(None, &fresh_tenant(), request, MODEL)
        .await;

    assert_eq!(
        response.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "RateLimitReserveStage must be reachable through route_completion, not just route_generate"
    );
}

// ---------------------------------------------------------------------------
// Embeddings, classification and transcription take the same reservation.
// ---------------------------------------------------------------------------

/// An engine on the vLLM wire whose `Embed` answers every request with a
/// two-dimensional vector and reports `prompt_tokens` -- or fails it with
/// INTERNAL when `fail` is set. The generate side is unused.
#[derive(Clone)]
struct EmbedEngine {
    prompt_tokens: u32,
    fail: bool,
}

#[tonic::async_trait]
impl proto::vllm_engine_server::VllmEngine for EmbedEngine {
    async fn start_profile(
        &self,
        _request: Request<smg_grpc_client::common_proto::StartProfileRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn stop_profile(
        &self,
        _request: Request<smg_grpc_client::common_proto::StopProfileRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;

    async fn generate(
        &self,
        _request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    async fn embed(
        &self,
        _request: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        if self.fail {
            return Err(Status::internal("embed-test induced engine failure"));
        }
        Ok(Response::new(proto::EmbedResponse {
            embedding: vec![0.25, 0.75],
            prompt_tokens: self.prompt_tokens,
            embedding_dim: 2,
        }))
    }

    async fn abort(
        &self,
        _request: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        Ok(Response::new(proto::AbortResponse::default()))
    }

    async fn flush_cache(
        &self,
        _request: Request<smg_grpc_client::common_proto::FlushCacheRequest>,
    ) -> Result<Response<smg_grpc_client::common_proto::FlushCacheResponse>, Status> {
        Ok(Response::new(
            smg_grpc_client::common_proto::FlushCacheResponse::default(),
        ))
    }

    async fn health_check(
        &self,
        _request: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        Ok(Response::new(proto::HealthCheckResponse {
            healthy: true,
            ..Default::default()
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        Ok(Response::new(proto::GetModelInfoResponse::default()))
    }

    async fn get_server_info(
        &self,
        _request: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        Ok(Response::new(proto::GetServerInfoResponse::default()))
    }

    async fn get_loads(
        &self,
        _request: Request<proto::GetLoadsRequest>,
    ) -> Result<Response<proto::GetLoadsResponse>, Status> {
        Ok(Response::new(proto::GetLoadsResponse::default()))
    }

    type GetTokenizerStream =
        ReceiverStream<Result<smg_grpc_client::common_proto::GetTokenizerChunk, Status>>;

    async fn get_tokenizer(
        &self,
        _request: Request<smg_grpc_client::common_proto::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }

    type SubscribeKvEventsStream =
        ReceiverStream<Result<smg_grpc_client::common_proto::KvEventBatch, Status>>;

    async fn subscribe_kv_events(
        &self,
        _request: Request<smg_grpc_client::common_proto::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("unused in this test"))
    }
}

/// A router over one [`EmbedEngine`] serving [`MODEL`] on the vLLM wire.
async fn build_embed_router(
    budget: Option<Budget>,
    engine: EmbedEngine,
) -> (Box<dyn RouterTrait>, Option<tempfile::TempDir>) {
    let port = serve_engine(proto::vllm_engine_server::VllmEngineServer::new(engine)).await;
    let worker = grpc_worker(
        port,
        WorkerType::Regular,
        RuntimeType::Vllm,
        ModelCard::new(MODEL),
    )
    .build();
    build_router_over(
        budget,
        Arc::new(worker),
        MODEL,
        Arc::new(MockTokenizer::new()),
    )
    .await
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn embeddings_request() -> EmbeddingRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "input": PROMPT_TEXT,
    }))
    .unwrap()
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn classify_request() -> ClassifyRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "input": PROMPT_TEXT,
    }))
    .unwrap()
}

/// The model id of the transcription cases: a name the one transcription
/// family recognises, served by the canned mock worker (a transcription is a
/// chat-shaped generation to the engine).
const ASR_MODEL: &str = "qwen3-asr-tenant-rl-test-model";

/// A router over the canned mock worker serving [`ASR_MODEL`]. Its mock
/// tokenizer renders media as template text, so the audio turn needs no
/// media processing and the rendered prompt's tokens are the words of the
/// caller's prompt: "Hello world" = the 2-token reserve estimate again. The
/// canned answer's ids (100 up) are given words, so the transcript is text.
async fn build_transcription_router(
    budget: Option<Budget>,
    output_tokens: u32,
) -> (Box<dyn RouterTrait>, Option<tempfile::TempDir>) {
    let port = serve_canned_worker(Canned {
        model: ASR_MODEL,
        gen_delay: Duration::ZERO,
        output_tokens,
    })
    .await;
    let worker = grpc_worker(
        port,
        WorkerType::Regular,
        RuntimeType::TokenSpeed,
        ModelCard::new(ASR_MODEL),
    )
    .build();
    let tokenizer = MockTokenizer::new()
        .with_renderer_capabilities(RendererCapabilities {
            media_handling: MediaHandling::TemplateText,
            ..RendererCapabilities::default()
        })
        .with_tokens(&[
            ("spoken", 100),
            ("words", 101),
            ("heard", 102),
            ("here", 103),
        ]);
    build_router_over(budget, Arc::new(worker), ASR_MODEL, Arc::new(tokenizer)).await
}

/// The status of `response` and, for the failure message, its body.
async fn status_and_body(response: axum::response::Response) -> (http::StatusCode, String) {
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap_or_default();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// One transcription of a few bytes of audio, with [`PROMPT_TEXT`] as the
/// caller's prompt.
async fn transcribe(router: &dyn RouterTrait) -> axum::response::Response {
    let request = TranscriptionRequest {
        model: ASR_MODEL.to_string(),
        prompt: Some(PROMPT_TEXT.to_string()),
        ..Default::default()
    };
    let audio = AudioFile {
        bytes: bytes::Bytes::from_static(b"not real audio"),
        file_name: "clip.wav".to_string(),
        content_type: Some("audio/wav".to_string()),
    };
    router
        .route_audio_transcriptions(None, &fresh_tenant(), &request, audio, ASR_MODEL)
        .await
}

/// The impossible-budget proof of `completion_denial_reaches_the_client`
/// through the embeddings endpoint: the reserve stage is live on it.
#[tokio::test]
async fn embeddings_denial_reaches_the_client() {
    let (router, _dir) = build_embed_router(
        Some(Budget::Tokens(1)),
        EmbedEngine {
            prompt_tokens: 1,
            fail: false,
        },
    )
    .await;

    let response = router
        .route_embeddings(None, &fresh_tenant(), embeddings_request(), MODEL)
        .await;
    assert_eq!(
        response.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "RateLimitReserveStage must be reachable through route_embeddings"
    );
}

/// The same through the classification endpoint.
#[tokio::test]
async fn classify_denial_reaches_the_client() {
    let (router, _dir) = build_embed_router(
        Some(Budget::Tokens(1)),
        EmbedEngine {
            prompt_tokens: 1,
            fail: false,
        },
    )
    .await;

    let response = router
        .route_classify(None, &fresh_tenant(), classify_request(), MODEL)
        .await;
    assert_eq!(
        response.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "RateLimitReserveStage must be reachable through route_classify"
    );
}

/// The same through the transcription endpoint.
#[tokio::test]
async fn transcription_denial_reaches_the_client() {
    let (router, _dir) = build_transcription_router(Some(Budget::Tokens(1)), 4).await;

    let response = transcribe(router.as_ref()).await;
    assert_eq!(
        response.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "RateLimitReserveStage must be reachable through route_audio_transcriptions"
    );
}

/// An embeddings answer settles the prompt count the engine reports, as a
/// generate answer settles its usage. Budget 6: the request reserves its
/// 2-token estimate and settles to the engine's 5, leaving 1; the second
/// request's 2-token reserve is denied. Had the settlement not run, 4 would
/// remain and the second request would be admitted.
#[tokio::test]
async fn embeddings_settle_the_engine_prompt_count() {
    let (router, _dir) = build_embed_router(
        Some(Budget::Tokens(6)),
        EmbedEngine {
            prompt_tokens: 5,
            fail: false,
        },
    )
    .await;

    let first = router
        .route_embeddings(None, &fresh_tenant(), embeddings_request(), MODEL)
        .await;
    assert_eq!(first.status(), http::StatusCode::OK);

    let second = router
        .route_embeddings(None, &fresh_tenant(), embeddings_request(), MODEL)
        .await;
    assert_eq!(
        second.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "the first answer's 5 prompt tokens must have been settled, not the 2-token estimate"
    );
}

/// The same settlement through the classification endpoint.
#[tokio::test]
async fn classify_settles_the_engine_prompt_count() {
    let (router, _dir) = build_embed_router(
        Some(Budget::Tokens(6)),
        EmbedEngine {
            prompt_tokens: 5,
            fail: false,
        },
    )
    .await;

    let first = router
        .route_classify(None, &fresh_tenant(), classify_request(), MODEL)
        .await;
    assert_eq!(first.status(), http::StatusCode::OK);

    let second = router
        .route_classify(None, &fresh_tenant(), classify_request(), MODEL)
        .await;
    assert_eq!(
        second.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "the first answer's 5 prompt tokens must have been settled, not the 2-token estimate"
    );
}

/// A transcription settles the engine's counts of its generation, as a chat
/// answer does: the canned worker reports prompt_tokens 1 and 4 completion
/// tokens, so budget 6 holds the first request (reserve 2, settle 5, 1 left)
/// and refuses the second.
#[tokio::test]
async fn transcription_settles_the_engine_counts() {
    let (router, _dir) = build_transcription_router(Some(Budget::Tokens(6)), 4).await;

    let (status, body) = status_and_body(transcribe(router.as_ref()).await).await;
    assert_eq!(status, http::StatusCode::OK, "{body}");

    let second = transcribe(router.as_ref()).await;
    assert_eq!(
        second.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "the transcript's 1 + 4 tokens must have been settled, not the 2-token estimate"
    );
}

/// An embeddings request the engine fails keeps its reservation's debit, as
/// a failed chat request does (the router closes the reservation unsettled).
/// Budget 3: the failed request's 2-token estimate stays debited, so the
/// second request's 2-token reserve is denied; a refund would admit it.
#[tokio::test]
async fn a_failed_embeddings_request_keeps_its_reservation_debit() {
    let (router, _dir) = build_embed_router(
        Some(Budget::Tokens(3)),
        EmbedEngine {
            prompt_tokens: 1,
            fail: true,
        },
    )
    .await;

    let first = router
        .route_embeddings(None, &fresh_tenant(), embeddings_request(), MODEL)
        .await;
    assert!(
        first.status().is_server_error(),
        "the engine's failure reaches the client, got {}",
        first.status()
    );

    let second = router
        .route_embeddings(None, &fresh_tenant(), embeddings_request(), MODEL)
        .await;
    assert_eq!(
        second.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "a failed request's reservation keeps its debit"
    );
}

// ---------------------------------------------------------------------------
// The Responses iterations take the same reservation, one per iteration.
// ---------------------------------------------------------------------------

/// A chat request of [`PROMPT_TEXT`]: through the mock tokenizer's
/// `role: content` template, the 2-token reserve estimate again.
#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn chat_request() -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT_TEXT}],
        "stream": false,
    }))
    .unwrap()
}

/// A Responses request of [`PROMPT_TEXT`] to `model` with `tools` (a mock
/// MCP server's, or none), streamed or not, never stored.
#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn responses_request(model: &str, stream: bool, tools: Value) -> ResponsesRequest {
    serde_json::from_value(json!({
        "model": model,
        "input": PROMPT_TEXT,
        "stream": stream,
        "store": false,
        "max_output_tokens": 16,
        "tools": tools,
    }))
    .unwrap()
}

/// One Responses request through `router`, within the test's patience: its
/// status and body (the final object, or the whole SSE stream).
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn responses(
    router: &dyn RouterTrait,
    request: ResponsesRequest,
    model: &str,
) -> (http::StatusCode, String) {
    let response = timeout(
        Duration::from_secs(30),
        router.route_responses(None, &fresh_tenant(), request, model),
    )
    .await
    .expect("the Responses request answers");
    status_and_body(response).await
}

/// The model of the tool-loop cases, served by the scripted engine.
const TOOL_LOOP_MODEL: &str = "tenant-rl-tool-loop-model";

/// The answer the scripted engine's tokens decode to once a tool result is
/// in the prompt: the loop's second and last iteration.
const FINAL_ANSWER: &str = "Final answer after tools";

/// Twelve words the mock vocabulary knows: a generate request of this text
/// reserves a 12-token estimate.
const PROBE_TEXT: &str = "Hello world Hello world Hello world Hello world Hello world Hello world";

/// A router over a scripted engine whose every generation is `output_tokens`
/// tokens decoded as one `brave_web_search` call (and as [`FINAL_ANSWER`]
/// once a tool result is in the prompt), with a mock MCP server that
/// executes the call: a Responses request with the server's tools runs two
/// iterations, the call and then the answer. The engine reports
/// `prompt_tokens: 1` and `completion_tokens: output_tokens` per generation.
/// The caller stops the MCP server.
#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_tool_loop_router(
    budget: Option<Budget>,
    output_tokens: u32,
) -> (
    Box<dyn RouterTrait>,
    Option<tempfile::TempDir>,
    MockMCPServer,
) {
    let port = serve_engine(TokenSpeedSchedulerServer::new(ScriptedWorker {
        output_tokens,
        finish_reasons: vec!["stop"],
        generation: Default::default(),
        requests: Default::default(),
    }))
    .await;
    let worker = grpc_worker(
        port,
        WorkerType::Regular,
        RuntimeType::TokenSpeed,
        ModelCard::new(TOOL_LOOP_MODEL),
    )
    .build();
    let tokenizer = ScriptedTokenizer::new(
        "<tool_call>\n{\"name\":\"brave_web_search\",\"arguments\":{\"query\":\"Hello world\"}}\n</tool_call>",
    )
    .with_final_answer(1, FINAL_ANSWER);
    let (mut config, dir) = router_config(budget);
    config.tool_call_parser = Some("qwen".into());
    let app_context =
        grpc_test_context_with_tokenizer(config, TOOL_LOOP_MODEL, Arc::new(tokenizer)).await;
    let router = grpc_router(&app_context, vec![Arc::new(worker)]).await;
    let mcp = MockMCPServer::start()
        .await
        .expect("the mock MCP server starts");
    (router, dir, mcp)
}

/// The tools of a Responses request that runs the tool loop over `mcp`.
fn mcp_tools(mcp: &MockMCPServer) -> Value {
    json!([{
        "type": "mcp",
        "server_label": "test-tools",
        "server_url": mcp.url(),
        "require_approval": "never",
    }])
}

/// A Responses request counts against the tenant's request budget as a chat
/// request does: two chat requests spend a budget of two, and the third
/// request is refused whether it is a chat or a Responses request, streamed
/// or not (the denial, a finite wait, carries `Retry-After`).
#[tokio::test]
async fn responses_take_the_tenant_request_budget_as_chat_does() {
    for stream in [false, true] {
        let grpc_port = start_mock_grpc_worker(4).await;
        let (router, _dir) = build_router(Some(Budget::Requests(2)), grpc_port).await;

        for _ in 0..2 {
            let response = router
                .route_chat(None, &fresh_tenant(), chat_request(), MODEL)
                .await;
            assert_eq!(response.status(), http::StatusCode::OK);
        }

        let request = responses_request(MODEL, stream, json!([]));
        let response = timeout(
            Duration::from_secs(30),
            router.route_responses(None, &fresh_tenant(), request, MODEL),
        )
        .await
        .unwrap_or_else(|_| panic!("stream={stream}: the Responses request answers"));
        assert_eq!(
            response.status(),
            http::StatusCode::TOO_MANY_REQUESTS,
            "stream={stream}: the third request of a budget of two is refused on the Responses path too"
        );
        assert!(
            response.headers().get(http::header::RETRY_AFTER).is_some(),
            "stream={stream}: a finite-wait denial carries Retry-After"
        );
    }
}

/// A Responses tool loop takes one reservation per iteration: its two
/// iterations (the call, then the answer) spend a request budget of two, so
/// the tenant's next request is refused; a budget of three has one left.
#[tokio::test]
async fn a_responses_tool_loop_reserves_once_per_iteration() {
    for stream in [false, true] {
        for (budget, probe_status) in [
            (2, http::StatusCode::TOO_MANY_REQUESTS),
            (3, http::StatusCode::OK),
        ] {
            let (router, _dir, mut mcp) =
                build_tool_loop_router(Some(Budget::Requests(budget)), 2).await;
            let (status, body) = responses(
                router.as_ref(),
                responses_request(TOOL_LOOP_MODEL, stream, mcp_tools(&mcp)),
                TOOL_LOOP_MODEL,
            )
            .await;
            mcp.stop().await;
            assert_eq!(
                status,
                http::StatusCode::OK,
                "stream={stream} budget={budget}: {body}"
            );
            assert!(
                body.contains(FINAL_ANSWER),
                "stream={stream} budget={budget}: the loop ran to its answer: {body}"
            );

            let probe = router
                .route_generate(
                    None,
                    &fresh_tenant(),
                    generate_request_for(TOOL_LOOP_MODEL, PROMPT_TEXT),
                    TOOL_LOOP_MODEL,
                )
                .await;
            assert_eq!(
                probe.status(),
                probe_status,
                "stream={stream} budget={budget}: the loop's two iterations took two of the budget's requests"
            );
        }
    }
}

/// Each iteration of a Responses tool loop settles its reservation with the
/// engine's counts, as a chat request does. The engine reports 1 prompt and
/// 20 completion tokens per generation: the loop's two iterations spend 42
/// of a token budget of 44 and the 12-token probe is refused; one generation
/// alone leaves 23 and the probe fits. Had an iteration reserved nothing, or
/// settled its 2-token estimate instead of the engine's 21, the probe would
/// fit after the loop as well.
#[tokio::test]
async fn a_responses_tool_loop_settles_each_iteration_with_the_engine_counts() {
    for stream in [false, true] {
        let (router, _dir, mut mcp) = build_tool_loop_router(Some(Budget::Tokens(44)), 20).await;
        let (status, body) = responses(
            router.as_ref(),
            responses_request(TOOL_LOOP_MODEL, stream, mcp_tools(&mcp)),
            TOOL_LOOP_MODEL,
        )
        .await;
        mcp.stop().await;
        assert_eq!(status, http::StatusCode::OK, "stream={stream}: {body}");
        assert!(
            body.contains(FINAL_ANSWER),
            "stream={stream}: the loop ran to its answer: {body}"
        );

        let probe = router
            .route_generate(
                None,
                &fresh_tenant(),
                generate_request_for(TOOL_LOOP_MODEL, PROBE_TEXT),
                TOOL_LOOP_MODEL,
            )
            .await;
        assert_eq!(
            probe.status(),
            http::StatusCode::TOO_MANY_REQUESTS,
            "stream={stream}: 2 of 44 tokens are left once both iterations settled 1 + 20"
        );
    }

    // The control: one settled generation leaves 23 of 44, the probe fits.
    let (router, _dir, mut mcp) = build_tool_loop_router(Some(Budget::Tokens(44)), 20).await;
    mcp.stop().await;
    let first = router
        .route_generate(
            None,
            &fresh_tenant(),
            generate_request_for(TOOL_LOOP_MODEL, PROMPT_TEXT),
            TOOL_LOOP_MODEL,
        )
        .await;
    assert_eq!(first.status(), http::StatusCode::OK);
    let probe = router
        .route_generate(
            None,
            &fresh_tenant(),
            generate_request_for(TOOL_LOOP_MODEL, PROBE_TEXT),
            TOOL_LOOP_MODEL,
        )
        .await;
    assert_eq!(
        probe.status(),
        http::StatusCode::OK,
        "23 of 44 tokens are left after one settled generation"
    );
}

/// Without a tenant policy a Responses request is served as before, the tool
/// loop included.
#[tokio::test]
async fn responses_without_a_tenant_policy_are_unchanged() {
    for stream in [false, true] {
        let grpc_port = start_mock_grpc_worker(4).await;
        let (router, _dir) = build_router(None, grpc_port).await;
        let (status, body) = responses(
            router.as_ref(),
            responses_request(MODEL, stream, json!([])),
            MODEL,
        )
        .await;
        assert_eq!(status, http::StatusCode::OK, "stream={stream}: {body}");

        let (router, _dir, mut mcp) = build_tool_loop_router(None, 2).await;
        let (status, body) = responses(
            router.as_ref(),
            responses_request(TOOL_LOOP_MODEL, stream, mcp_tools(&mcp)),
            TOOL_LOOP_MODEL,
        )
        .await;
        mcp.stop().await;
        assert_eq!(status, http::StatusCode::OK, "stream={stream}: {body}");
        assert!(
            body.contains(FINAL_ANSWER),
            "stream={stream}: the loop ran to its answer: {body}"
        );
    }
}
