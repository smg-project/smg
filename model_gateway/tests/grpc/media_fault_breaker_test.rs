//! A worker's report that the request's media failed is answered for the
//! media (a 400 under the code the worker names) and sampled by no circuit
//! breaker: twenty requests whose image cannot be fetched leave the only
//! worker in rotation, and the next request is served. A worker's own
//! failure still opens the breaker.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{body::to_bytes, http::StatusCode};
use common::engine_answers::{TokenSpeedAnswers, TokenSpeedGenerateStream};
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
    stamp_media_code, tokenspeed_scheduler::tokenspeed_proto as ts, WorkerMediaFault,
};
use tokio::{net::TcpListener, time::timeout};
use tonic::{transport::Server, Request, Response, Status};

use crate::{common, scripted_worker};

const MODEL: &str = "media-fault-breaker-model";

/// A worker that refuses the first `refusals` generations with `refusal` and
/// serves every one after.
struct RefusingWorker {
    refusals: AtomicUsize,
    refusal: Status,
    served: scripted_worker::ScriptedWorker,
}

#[tonic::async_trait]
impl TokenSpeedAnswers for RefusingWorker {
    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<TokenSpeedGenerateStream>, Status> {
        let refused = self
            .refusals
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(1)
            })
            .is_ok();
        if refused {
            return Err(self.refusal.clone());
        }
        self.served.generate(request).await
    }
}

/// One answer of the router: the status, the `x-smg-error-code` header and
/// the JSON body.
struct Answer {
    status: StatusCode,
    code_header: Option<String>,
    body: Value,
}

/// A router with one gRPC worker that refuses the first `refusals` requests
/// with `refusal`; `requests` chat requests through it, in order, and the
/// worker as the router sees it.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test harness: failures should panic; the in-process worker is aborted after the requests"
)]
async fn answers_of(
    refusal: Status,
    refusals: usize,
    requests: usize,
) -> (Vec<Answer>, Arc<dyn Worker>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let worker_task = tokio::spawn(async move {
        Server::builder()
            .add_service(
                RefusingWorker {
                    refusals: AtomicUsize::new(refusals),
                    refusal,
                    served: scripted_worker::ScriptedWorker {
                        output_tokens: 2,
                        finish_reasons: vec!["stop"],
                        generation: Default::default(),
                        requests: Default::default(),
                    },
                }
                .into_server(),
            )
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .expect("refusing gRPC worker failed");
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
            max_retries: 1,
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
        "messages": [{"role": "user", "content": "Describe the picture."}],
        "max_tokens": 4,
    }))
    .unwrap();
    let mut answers = Vec::with_capacity(requests);
    for _ in 0..requests {
        let answer = timeout(Duration::from_secs(30), async {
            let response = router
                .route_chat(None, &tenant, request.clone(), MODEL)
                .await;
            let status = response.status();
            let code_header = response
                .headers()
                .get("x-smg-error-code")
                .map(|value| value.to_str().unwrap().to_string());
            let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
            Answer {
                status,
                code_header,
                body,
            }
        })
        .await
        .expect("the request terminates");
        answers.push(answer);
    }
    worker_task.abort();
    (answers, worker)
}

/// Twenty requests whose media the worker could not fetch, in each of the
/// forms the worker reports it, are each answered for the media (the
/// request's own reference a 400 `invalid_request_error`, an origin that
/// does not deliver right now a 502 `server_error`, under the code the
/// worker names, and `media_fetch_failed` from a worker that names none) and
/// leave the worker's breaker closed and the worker in rotation; the request
/// after them is served.
#[tokio::test]
async fn media_faults_are_answered_for_the_media_and_leave_the_worker_in_rotation() {
    let cases = [
        (
            "the request's own fault, reported in the trailer",
            WorkerMediaFault::Client.stamp(Status::invalid_argument(
                "Failed to finalize multimodal tracker: HTTP error while fetching media: HTTP \
                 status client error (404 Not Found) for url (https://media.example/missing.png)",
            )),
            "media_fetch_failed",
            StatusCode::BAD_REQUEST,
        ),
        (
            "the media host's fault, reported in the trailer",
            WorkerMediaFault::Transient.stamp(Status::unavailable(
                "Failed to finalize multimodal tracker: media fetch of \
                 https://media.example/picture.png timed out after 10s",
            )),
            "media_fetch_failed",
            StatusCode::BAD_GATEWAY,
        ),
        (
            "a size refusal, with its code in the trailer",
            stamp_media_code(
                WorkerMediaFault::Client.stamp(Status::invalid_argument(
                    "Failed to finalize multimodal tracker: decoded video RGB payload would be \
                     2208384000 bytes, exceeding SMG_VIDEO_MAX_DECODED_BYTES=1073741824",
                )),
                "media_too_large",
            ),
            "media_too_large",
            StatusCode::BAD_REQUEST,
        ),
        (
            "media refused as the request sent it, with its code in the trailer",
            stamp_media_code(
                WorkerMediaFault::Client.stamp(Status::invalid_argument(
                    "Failed to finalize multimodal tracker: data URL parse error: missing comma \
                     in data url",
                )),
                "invalid_multimodal_request",
            ),
            "invalid_multimodal_request",
            StatusCode::BAD_REQUEST,
        ),
        (
            "a worker that predates the trailer",
            Status::unavailable(
                "Failed to finalize multimodal tracker: HTTP error while fetching media: error \
                 sending request for url (https://media.example/picture.png)",
            ),
            "media_fetch_failed",
            StatusCode::BAD_GATEWAY,
        ),
    ];
    for (name, refusal, code, status) in cases {
        let kind = if status.is_server_error() {
            "server_error"
        } else {
            "invalid_request_error"
        };
        let (answers, worker) = answers_of(refusal, 20, 21).await;
        for (index, answer) in answers[..20].iter().enumerate() {
            assert_eq!(
                answer.status, status,
                "{name}: request {index}: {}",
                answer.body
            );
            assert_eq!(
                answer.body["error"]["code"], code,
                "{name}: {}",
                answer.body
            );
            assert_eq!(
                answer.body["error"]["type"], kind,
                "{name}: {}",
                answer.body
            );
            assert_eq!(answer.code_header.as_deref(), Some(code), "{name}");
        }
        assert_eq!(
            worker.circuit_breaker_state(),
            CircuitState::Closed,
            "{name}"
        );
        assert!(
            worker.is_available(),
            "{name}: the worker stays in rotation"
        );
        let served = &answers[20];
        assert_eq!(served.status, StatusCode::OK, "{name}: {}", served.body);
    }
}

/// The worker's own failure to start a generation is still a worker fault,
/// whether its engine refuses (`UNAVAILABLE`) or its own media handling
/// failed on its side after the bytes arrived (`INTERNAL`: the scratch file
/// of a video decode in a temporary directory that is full or not writable):
/// the breaker opens at its threshold.
#[tokio::test]
async fn a_workers_own_failure_still_opens_the_breaker() {
    for (name, refusal, status) in [
        (
            "the engine refuses",
            Status::unavailable("engine is restarting"),
            StatusCode::SERVICE_UNAVAILABLE,
        ),
        (
            "the worker's scratch file fails",
            Status::internal(
                "media fetch failed: I/O error while decoding media: No space left on device (os \
                 error 28)",
            ),
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
    ] {
        let (answers, worker) = answers_of(refusal, 20, 5).await;
        for answer in &answers {
            assert_eq!(answer.status, status, "{name}: {}", answer.body);
            assert_eq!(
                answer.body["error"]["code"], "start_generation_failed",
                "{name}: {}",
                answer.body
            );
        }
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Open, "{name}");
        assert!(!worker.is_available(), "{name}");
    }
}
