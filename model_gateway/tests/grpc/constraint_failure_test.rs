//! A worker's report that the engine could not continue the request under
//! its constraint (the status every wire ends such a request with) is one
//! explicit answer at the gateway: a 422 under the code `constraint_failed`,
//! the type a client reads for its own request, the retry layer replaying
//! nothing (one call per client request on the only worker, with a budget of
//! five), the worker's breaker untouched and the worker in rotation. The same
//! whether the report comes before the call opens, before the first token or
//! after some tokens; a streamed response the report cuts after its first
//! tokens ends with the error event under the same code, then `[DONE]`.

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{body::to_bytes, http::StatusCode};
use futures::{stream, Stream};
use openai_protocol::{
    chat::ChatCompletionRequest, model_card::ModelCard, worker::HealthCheckConfig,
};
use serde_json::{json, Value};
use smg::{
    config::{RetryConfig, RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::RouterFactory,
    tenant::TenantKey,
    worker::{
        circuit_breaker::CircuitState, BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker,
        WorkerType,
    },
};
use smg_grpc_client::{
    common_proto, constraint_failed_status, tokenspeed_scheduler::tokenspeed_proto as ts,
};
use tokio::{net::TcpListener, time::timeout};
use tonic::{transport::Server, Request, Response, Status};
use ts::{
    generate_response::Response as GenResp,
    token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
};

use crate::{common, scripted_worker};

const MODEL: &str = "constraint-failure-model";
/// The engine's own words, as the wires keep them.
const ENGINE_WORDS: &str = "the scheduler aborted the request: Aborted";

/// When the worker reports the failure.
#[derive(Clone, Copy, Debug)]
enum ReportAt {
    /// The call itself fails: the report in the trailers of a stream that
    /// never opened (a servicer that aborts before its first message).
    Call,
    /// The call opens and the stream's first item is the report.
    FirstItem,
    /// The stream delivers this many tokens, then the report.
    AfterTokens(u32),
}

/// A worker whose engine could not continue every request under its
/// constraint, reporting it at `report_at`; every call counted.
struct ConstraintFailingWorker {
    report_at: ReportAt,
    calls: AtomicUsize,
    served: scripted_worker::ScriptedWorker,
}

type Served = scripted_worker::ScriptedWorker;
type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;

#[tonic::async_trait]
impl TokenSpeedScheduler for ConstraintFailingWorker {
    type GenerateStream = GenStream;
    type SubscribeKvEventsStream = <Served as TokenSpeedScheduler>::SubscribeKvEventsStream;
    type GetTokenizerStream = <Served as TokenSpeedScheduler>::GetTokenizerStream;

    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let request_id = request.into_inner().request_id;
        let report = constraint_failed_status(ENGINE_WORDS);
        let tokens = match self.report_at {
            ReportAt::Call => return Err(report),
            ReportAt::FirstItem => 0,
            ReportAt::AfterTokens(tokens) => tokens,
        };
        let mut items: Vec<Result<ts::GenerateResponse, Status>> = (0..tokens)
            .map(|i| {
                Ok(ts::GenerateResponse {
                    request_id: request_id.clone(),
                    response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                        token_ids: vec![100 + i],
                        prompt_tokens: 1,
                        completion_tokens: i + 1,
                        cached_tokens: 0,
                        output_logprobs: None,
                        weight_version: None,
                        index: 0,
                    })),
                })
            })
            .collect();
        items.push(Err(report));
        Ok(Response::new(Box::pin(stream::iter(items))))
    }

    async fn health_check(
        &self,
        request: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        self.served.health_check(request).await
    }

    async fn abort(
        &self,
        request: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        self.served.abort(request).await
    }

    async fn get_model_info(
        &self,
        request: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        self.served.get_model_info(request).await
    }

    async fn get_server_info(
        &self,
        request: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        self.served.get_server_info(request).await
    }

    async fn get_loads(
        &self,
        request: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        self.served.get_loads(request).await
    }

    async fn subscribe_kv_events(
        &self,
        request: Request<common_proto::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        self.served.subscribe_kv_events(request).await
    }

    async fn flush_cache(
        &self,
        request: Request<common_proto::FlushCacheRequest>,
    ) -> Result<Response<common_proto::FlushCacheResponse>, Status> {
        self.served.flush_cache(request).await
    }

    async fn start_profile(
        &self,
        request: Request<common_proto::StartProfileRequest>,
    ) -> Result<Response<common_proto::ProfileResponse>, Status> {
        self.served.start_profile(request).await
    }

    async fn stop_profile(
        &self,
        request: Request<common_proto::StopProfileRequest>,
    ) -> Result<Response<common_proto::ProfileResponse>, Status> {
        self.served.stop_profile(request).await
    }

    async fn get_tokenizer(
        &self,
        request: Request<common_proto::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        self.served.get_tokenizer(request).await
    }
}

/// One answer of the router: the status, the content type, the
/// `x-smg-error-code` header and the body (JSON, or the SSE text).
struct Answer {
    status: StatusCode,
    content_type: String,
    code_header: Option<String>,
    body: String,
}

impl Answer {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or(Value::Null)
    }

    /// The `data:` payloads of an SSE body, in order.
    fn sse_payloads(&self) -> Vec<String> {
        self.body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(str::to_string)
            .collect()
    }
}

/// A router with one gRPC worker reporting the constraint failure at
/// `report_at` and a retry budget of five; `requests` chat requests through
/// it (streamed or not), in order; the calls the worker saw; the worker as
/// the router sees it.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test harness: failures should panic; the in-process worker is aborted after the requests"
)]
async fn answers_of(
    report_at: ReportAt,
    streamed: bool,
    requests: usize,
) -> (Vec<Answer>, usize, Arc<dyn Worker>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let worker_impl = Arc::new(ConstraintFailingWorker {
        report_at,
        calls: AtomicUsize::new(0),
        served: scripted_worker::ScriptedWorker {
            output_tokens: 2,
            finish_reasons: vec!["stop"],
            generation: Default::default(),
            requests: Default::default(),
        },
    });
    let calls = Arc::clone(&worker_impl);
    let worker_task = tokio::spawn(async move {
        Server::builder()
            .add_service(TokenSpeedSchedulerServer::from_arc(worker_impl))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .expect("constraint-failing gRPC worker failed");
    });
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .retry_config(RetryConfig {
            max_retries: 5,
            initial_backoff_ms: 1,
            max_backoff_ms: 2,
            ..Default::default()
        })
        .build_unchecked();
    config.health_check.disable_health_check = true;
    let context = common::create_test_context_with_mock_tokenizer(config, MODEL).await;
    context
        .worker_registry
        .register(Arc::clone(&worker))
        .unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Name a city."}],
        "max_tokens": 8,
        "stream": streamed,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "city",
                "schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        },
    }))
    .unwrap();
    let mut answers = Vec::with_capacity(requests);
    for _ in 0..requests {
        let answer = timeout(Duration::from_secs(30), async {
            let response = router
                .route_chat(None, &tenant, request.clone(), MODEL)
                .await;
            let status = response.status();
            let content_type = response
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok())
                .unwrap_or_default()
                .to_string();
            let code_header = response
                .headers()
                .get("x-smg-error-code")
                .map(|value| value.to_str().unwrap().to_string());
            let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            Answer {
                status,
                content_type,
                code_header,
                body: String::from_utf8_lossy(&bytes).into_owned(),
            }
        })
        .await
        .expect("the request terminates");
        answers.push(answer);
    }
    worker_task.abort();
    (answers, calls.calls.load(Ordering::SeqCst), worker)
}

fn assert_constraint_failed_body(answer: &Answer, name: &str) {
    assert_eq!(
        answer.status,
        StatusCode::UNPROCESSABLE_ENTITY,
        "{name}: {}",
        answer.body
    );
    assert!(
        answer.content_type.starts_with("application/json"),
        "{name}: {}",
        answer.content_type
    );
    let body = answer.json();
    assert_eq!(body["error"]["code"], "constraint_failed", "{name}: {body}");
    assert_eq!(
        body["error"]["type"], "invalid_request_error",
        "{name}: {body}"
    );
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.starts_with("the engine could not continue the request under its constraint: ")
            && message.ends_with(ENGINE_WORDS),
        "{name}: {message}"
    );
    assert_eq!(
        answer.code_header.as_deref(),
        Some("constraint_failed"),
        "{name}"
    );
}

/// Five requests the engine could not continue under their constraint, the
/// report arriving at each point of the stream, are each answered a 422
/// `constraint_failed` with the engine's words, cost the worker one call each
/// under a retry budget of five, and leave its breaker closed and the worker
/// in rotation.
#[tokio::test]
async fn a_constraint_failure_is_one_unprocessable_answer_and_one_call() {
    for report_at in [
        ReportAt::Call,
        ReportAt::FirstItem,
        ReportAt::AfterTokens(2),
    ] {
        let name = format!("{report_at:?}");
        let (answers, calls, worker) = answers_of(report_at, false, 5).await;
        for answer in &answers {
            assert_constraint_failed_body(answer, &name);
        }
        assert_eq!(calls, 5, "{name}: one call per request, no replay");
        assert_eq!(
            worker.circuit_breaker_state(),
            CircuitState::Closed,
            "{name}"
        );
        assert!(worker.is_available(), "{name}");
    }
}

/// A streamed request the report cuts after its first tokens ends with the
/// error event under the same code (the type a client reads for its own
/// request), then `[DONE]`; a streamed request refused before the call
/// opened is the JSON answer, as any worker rejection of a stream is.
#[tokio::test]
async fn a_streamed_constraint_failure_ends_with_the_error_event_under_its_code() {
    let (answers, calls, worker) = answers_of(ReportAt::AfterTokens(2), true, 3).await;
    for answer in &answers {
        assert_eq!(answer.status, StatusCode::OK, "{}", answer.body);
        assert!(
            answer.content_type.starts_with("text/event-stream"),
            "{}",
            answer.content_type
        );
        let payloads = answer.sse_payloads();
        assert!(
            payloads.len() >= 3,
            "chunks, the error, [DONE]: {payloads:?}"
        );
        assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
        let event: Value = serde_json::from_str(&payloads[payloads.len() - 2]).unwrap();
        assert_eq!(event["error"]["code"], "constraint_failed", "{event}");
        assert_eq!(event["error"]["type"], "invalid_request_error", "{event}");
        let message = event["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.starts_with(
                "Stream error: the engine could not continue the request under its constraint: "
            ) && message.ends_with(ENGINE_WORDS),
            "{message}"
        );
        let first: Value = serde_json::from_str(&payloads[0]).unwrap();
        assert!(first["choices"].is_array(), "a chunk came first: {first}");
    }
    assert_eq!(calls, 3, "one call per request, no replay");
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);

    let (answers, calls, _) = answers_of(ReportAt::Call, true, 2).await;
    for answer in &answers {
        assert_constraint_failed_body(answer, "streamed, refused before the call opened");
    }
    assert_eq!(calls, 2);
}
