use std::{
    fs,
    sync::{
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
        Arc, OnceLock,
    },
    time::{Duration, Instant},
};

use http::header::RETRY_AFTER;
use llm_tokenizer::{factory, registry::TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use portpicker::pick_unused_port;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use serde_json::json;
use smg_data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
use smg_grpc_client::{common_proto as common, tokenspeed_proto as ts};
use smg_mcp::{McpConfig, McpOrchestrator};
use tonic::{transport::Server, Request as TonicRequest, Response as TonicResponse, Status};
use tool_parser::ParserFactory as ToolParserFactory;
use ts::token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer};
use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

use super::*;
use crate::{
    config::{ParserChoice, PolicyConfig, RouterConfig, RoutingMode},
    policies::PolicyRegistry,
    rate_limit::{
        RateLimitBackend, ReserveOutcome, ReserveRequest, SharedReservationHandle, UsageSettlement,
    },
    tenant::TenantKey,
    worker::{BasicWorker, BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerRegistry},
    workflow::{
        create_worker_workflow_data, steps::local::SubmitTokenizerJobStep, JobQueue,
        JobQueueConfig, TokenizerLoadFailure, WorkerKind, WorkerRegistrationMode, WorkerSpec,
        WorkerWorkflowData, WorkflowEngines,
    },
};

fn regular_routing_mode() -> RoutingMode {
    RoutingMode::Regular {
        worker_urls: vec![],
    }
}

fn pd_routing_mode() -> RoutingMode {
    RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    }
}

fn epd_routing_mode() -> RoutingMode {
    RoutingMode::EncodePrefillDecode {
        encode_urls: vec![],
        prefill_urls: vec![],
        decode_urls: vec![],
        encode_policy: None,
        prefill_policy: None,
        decode_policy: None,
    }
}

/// A mode that does not serve an endpoint refuses it with the JSON error
/// object under the endpoint's code, as the router trait's default does.
#[tokio::test]
async fn a_mode_without_the_endpoint_refuses_it_with_the_json_error_object() {
    let ctx = grpc_ctx(pd_routing_mode()).await;
    let router = GrpcRouter::new(&ctx, Mode::PrefillDecode).expect("pd router");
    let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let audio = AudioFile {
        bytes: bytes::Bytes::new(),
        file_name: "a.wav".to_owned(),
        content_type: None,
    };
    fn body<T: serde::de::DeserializeOwned>(json: serde_json::Value) -> T {
        serde_json::from_value(json).expect("a request body")
    }
    let answers = [
        (
            router
                .route_embeddings(
                    None,
                    &tenant_meta,
                    body(serde_json::json!({"model": "m", "input": "hi"})),
                    "m",
                )
                .await,
            "embeddings_not_implemented",
        ),
        (
            router
                .route_classify(
                    None,
                    &tenant_meta,
                    body(serde_json::json!({"model": "m", "input": "hi"})),
                    "m",
                )
                .await,
            "classify_not_implemented",
        ),
        (
            router
                .route_audio_transcriptions(
                    None,
                    &tenant_meta,
                    &TranscriptionRequest::default(),
                    audio,
                    "m",
                )
                .await,
            "audio_transcriptions_not_implemented",
        ),
    ];
    for (answer, code) in answers {
        let status = answer.status();
        let body = axum::body::to_bytes(answer.into_body(), usize::MAX)
            .await
            .expect("drain body");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("json error body");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
        assert_eq!(body["error"]["code"], code, "{body}");
    }
}

/// Minimal `AppContext` for constructing a disaggregated gRPC router. PD
/// serves /v1/responses, so the MCP orchestrator must be initialized.
async fn grpc_ctx(mode: RoutingMode) -> Arc<AppContext> {
    grpc_ctx_with_parsers(mode, ParserChoice::default(), None).await
}

/// [`grpc_ctx`] with the gateway's `--parser` choice and its
/// `--tool-call-parser` name, which under Symphony pins the table that
/// answers to it for every model the registry does not know.
async fn grpc_ctx_with_parsers(
    mode: RoutingMode,
    parser: ParserChoice,
    tool_parser: Option<&str>,
) -> Arc<AppContext> {
    let mut builder = RouterConfig::builder()
        .mode(mode)
        .parser(parser)
        .grpc_connection()
        .policy(PolicyConfig::Random)
        .host("127.0.0.1")
        .port(3001)
        .max_payload_size(1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(10)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60);
    if let Some(name) = tool_parser {
        builder = builder.tool_call_parser(name);
    }
    let config = builder.build_unchecked();

    let mcp_orchestrator = Arc::new(OnceLock::new());
    mcp_orchestrator
        .set(Arc::new(
            McpOrchestrator::new(McpConfig::default())
                .await
                .expect("mcp orchestrator"),
        ))
        .ok();

    Arc::new(
        AppContext::builder()
            .router_config(config.clone())
            .client(crate::clients::http_client())
            .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
            .reasoning_parser_factory(Some(ReasoningParserFactory::new()))
            .tool_parser_factory(Some(ToolParserFactory::new()))
            .worker_registry(Arc::new(WorkerRegistry::new()))
            .policy_registry(Arc::new(PolicyRegistry::new(config.policy.clone())))
            .response_storage(Arc::new(MemoryResponseStorage::new()))
            .conversation_storage(Arc::new(MemoryConversationStorage::new()))
            .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
            .worker_job_queue(Arc::new(OnceLock::new()))
            .workflow_engines(Arc::new(OnceLock::new()))
            .mcp_orchestrator(mcp_orchestrator)
            .build()
            .expect("app context"),
    )
}

fn responses_request(model: &str) -> ResponsesRequest {
    serde_json::from_value(json!({"model": model, "input": "hi"})).expect("responses request")
}

fn count_request(model: &str) -> CountMessageTokensRequest {
    serde_json::from_value(json!({
        "model": model, "messages": [{"role": "user", "content": "hi"}]
    }))
    .expect("count request")
}

/// count_tokens is the router's own answer, from the registry's
/// tokenizer: the rendered prompt's token count for a served model (here
/// a renderer that encodes itself hands back three ids), the 404
/// model_not_found for a model nobody serves. No worker takes part.
#[tokio::test]
async fn count_tokens_is_answered_from_the_tokenizer() {
    let ctx = grpc_ctx(regular_routing_mode()).await;
    let tokenizer: Arc<dyn llm_tokenizer::traits::Tokenizer> = Arc::new(
        llm_tokenizer::mock::MockTokenizer::new().with_deferred_chat_ids(vec![11, 12, 13]),
    );
    ctx.tokenizer_registry
        .load("m", "m", "mock", || async move { Ok(tokenizer) })
        .await
        .expect("register the tokenizer");
    let worker = BasicWorkerBuilder::new("grpc://worker:30000")
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .model(ModelCard::new("m"))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    ctx.worker_registry
        .register(Arc::new(worker))
        .expect("register worker");
    let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
    let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

    let response = router
        .route_messages_count_tokens(None, &tenant_meta, count_request("m"), "m")
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes).expect("json"),
        json!({"input_tokens": 3})
    );

    let response = router
        .route_messages_count_tokens(None, &tenant_meta, count_request("nope"), "nope")
        .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "model_not_found"
    );
}

/// count_tokens shares the Messages preparation refusals: an image block
/// to a model whose only worker runs an engine without image input in the
/// capability matrix is refused with the sentence `/v1/messages` gives,
/// instead of counted, before any media is fetched; to an engine that
/// takes images the count goes on into media resolution (here, to the
/// config lookup the mock tokenizer cannot satisfy).
#[tokio::test]
async fn count_tokens_refuses_media_no_worker_of_the_model_takes() {
    for (runtime, code, message) in [
        (
            RuntimeType::Mlx,
            "multimodal_not_supported",
            Some("backend mlx does not support image inputs"),
        ),
        (RuntimeType::Sglang, "invalid_multimodal_request", None),
    ] {
        let ctx = grpc_ctx(regular_routing_mode()).await;
        let tokenizer: Arc<dyn llm_tokenizer::traits::Tokenizer> = Arc::new(
            llm_tokenizer::mock::MockTokenizer::new().with_deferred_chat_ids(vec![11, 12, 13]),
        );
        ctx.tokenizer_registry
            .load("m", "m", "mock", || async move { Ok(tokenizer) })
            .await
            .expect("register the tokenizer");
        let worker = BasicWorkerBuilder::new("grpc://worker:30000")
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(runtime)
            .model(ModelCard::new("m"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        ctx.worker_registry
            .register(Arc::new(worker))
            .expect("register worker");
        let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
        let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));
        let request: CountMessageTokensRequest = serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What colour is this pixel?"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="
                }}
            ]}]
        }))
        .expect("count request");

        let response = router
            .route_messages_count_tokens(None, &tenant_meta, request, "m")
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{runtime}");
        assert_eq!(
            error::extract_error_code_from_response(&response),
            code,
            "{runtime}"
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json error body");
        assert_eq!(body["error"]["code"], code, "{runtime}: {body}");
        if let Some(message) = message {
            assert_eq!(body["error"]["message"], message, "{runtime}: {body}");
        }
    }
}

/// A tokenizer a removed worker left in the registry does not make its
/// model served: the count is refused as an unknown model, like every
/// other route refuses it.
#[tokio::test]
async fn count_tokens_refuses_a_model_no_worker_serves_whatever_the_tokenizer_registry_holds() {
    let ctx = grpc_ctx(regular_routing_mode()).await;
    let tokenizer: Arc<dyn llm_tokenizer::traits::Tokenizer> = Arc::new(
        llm_tokenizer::mock::MockTokenizer::new().with_deferred_chat_ids(vec![11, 12, 13]),
    );
    ctx.tokenizer_registry
        .load("gone", "gone", "mock", || async move { Ok(tokenizer) })
        .await
        .expect("register the tokenizer");
    let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
    let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

    let response = router
        .route_messages_count_tokens(None, &tenant_meta, count_request("gone"), "gone")
        .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "model_not_found"
    );
}

/// A served model whose tokenizer the gateway has not registered yet
/// (the window after its worker turned healthy) is not ready: a
/// retryable 503 with a `Retry-After`, not a 500.
#[tokio::test]
async fn count_tokens_answers_not_ready_while_a_served_models_tokenizer_loads() {
    let ctx = grpc_ctx(regular_routing_mode()).await;
    let worker = BasicWorkerBuilder::new("grpc://worker:30000")
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .model(ModelCard::new("m"))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    ctx.worker_registry
        .register(Arc::new(worker))
        .expect("register worker");
    let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
    let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

    let response = router
        .route_messages_count_tokens(None, &tenant_meta, count_request("m"), "m")
        .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "tokenizer_not_ready"
    );
    assert_eq!(
        response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
}

/// A local gRPC worker of model `m`, as the pipeline tests register it.
fn grpc_worker_for_m() -> Arc<BasicWorker> {
    Arc::new(
        BasicWorkerBuilder::new("grpc://worker:30000")
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .model(ModelCard::new("m"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    )
}

/// The tokenizer-registration step's input for a local gRPC worker with
/// these labels, as the worker registration workflow hands it over.
fn tokenizer_step_input(
    ctx: &Arc<AppContext>,
    worker: &Arc<BasicWorker>,
    labels: &[(&str, &str)],
) -> WorkflowContext<WorkerWorkflowData> {
    let mut data = create_worker_workflow_data(
        WorkerSpec::new("grpc://worker:30000"),
        WorkerRegistrationMode::Upsert,
        ctx.clone(),
    );
    data.worker_kind = Some(WorkerKind::Local);
    data.connection_mode = Some(ConnectionMode::Grpc);
    data.actual_workers = Some(vec![worker.clone()]);
    data.final_labels = labels
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    WorkflowContext::new(WorkflowInstanceId::new(), data)
}

/// A served model whose worker registered without a tokenizer path (and
/// none in the router configuration) gets no tokenizer from the autoload,
/// ever: its requests are answered a 503 of its own code without a
/// `Retry-After`, not the not-ready 503 of a load in flight.
#[tokio::test]
async fn count_tokens_answers_load_failed_without_retry_after_for_a_worker_without_a_tokenizer_path(
) {
    let ctx = grpc_ctx(regular_routing_mode()).await;
    ctx.worker_job_queue
        .set(JobQueue::new(
            JobQueueConfig::default(),
            Arc::downgrade(&ctx),
        ))
        .ok();
    let worker = grpc_worker_for_m();
    ctx.worker_registry
        .register(worker.clone())
        .expect("register worker");
    let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
    let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

    let mut step = tokenizer_step_input(&ctx, &worker, &[]);
    SubmitTokenizerJobStep
        .execute(&mut step)
        .await
        .expect("the step runs");

    let response = router
        .route_messages_count_tokens(None, &tenant_meta, count_request("m"), "m")
        .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "tokenizer_load_failed"
    );
    assert!(
        response.headers().get(RETRY_AFTER).is_none(),
        "a retry changes nothing until a tokenizer registers"
    );
}

/// A one-token tokenizer (`hello`), enough for a count.
const MINIMAL_TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"BPE","vocab":{"hello":0},"merges":[]}}"#;

/// A tokenizer registered after a failed load clears the failure: here a
/// worker re-registering with a usable path (an operator's
/// `POST /v1/tokenizers` takes the same job), after which the model
/// answers again.
#[tokio::test]
async fn a_tokenizer_registered_after_a_failed_load_clears_the_failure() {
    let ctx = grpc_ctx(regular_routing_mode()).await;
    ctx.worker_job_queue
        .set(JobQueue::new(
            JobQueueConfig::default(),
            Arc::downgrade(&ctx),
        ))
        .ok();
    ctx.workflow_engines
        .set(WorkflowEngines::new(&ctx.router_config))
        .ok();
    let worker = grpc_worker_for_m();
    ctx.worker_registry
        .register(worker.clone())
        .expect("register worker");
    let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
    let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    assert!(ctx.tokenizer_load_failures.record(
        "m",
        TokenizerLoadFailure::Failed("Workflow failed at step load_tokenizer: boom".to_string())
    ));

    let response = router
        .route_messages_count_tokens(None, &tenant_meta, count_request("m"), "m")
        .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "tokenizer_load_failed"
    );
    assert!(response.headers().get(RETRY_AFTER).is_none());

    // The worker registers again, this time with a tokenizer to load.
    let tokenizer_dir = tempfile::tempdir().expect("tempdir");
    fs::write(
        tokenizer_dir.path().join("tokenizer.json"),
        MINIMAL_TOKENIZER_JSON,
    )
    .expect("tokenizer.json");
    fs::write(
        tokenizer_dir.path().join("tokenizer_config.json"),
        r#"{"chat_template": "hello"}"#,
    )
    .expect("tokenizer_config.json");
    let path = tokenizer_dir.path().to_str().expect("utf-8 path");
    let mut step = tokenizer_step_input(&ctx, &worker, &[("tokenizer_path", path)]);
    SubmitTokenizerJobStep
        .execute(&mut step)
        .await
        .expect("the step runs");
    // Submitted: the model is pending again, not failed.
    assert_eq!(ctx.tokenizer_load_failures.get("m"), None);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ctx.tokenizer_registry.contains("m") {
        assert!(Instant::now() < deadline, "the tokenizer did not register");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    let response = router
        .route_messages_count_tokens(None, &tenant_meta, count_request("m"), "m")
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(ctx.tokenizer_load_failures.get("m"), None);
}

/// Two registrations of one model: while worker A's tokenizer load is in
/// flight (held here on the registry's per-name lock), worker B
/// registering without a path does not turn the model's answer into a
/// final failure; the model stays not ready with its retry hint, and
/// answers once A's load ends.
#[tokio::test]
async fn a_registration_without_a_path_defers_to_the_models_load_in_flight() {
    let ctx = grpc_ctx(regular_routing_mode()).await;
    ctx.worker_job_queue
        .set(JobQueue::new(
            JobQueueConfig::default(),
            Arc::downgrade(&ctx),
        ))
        .ok();
    ctx.workflow_engines
        .set(WorkflowEngines::new(&ctx.router_config))
        .ok();
    let worker = grpc_worker_for_m();
    ctx.worker_registry
        .register(worker.clone())
        .expect("register worker");
    let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
    let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let tokenizer_dir = tempfile::tempdir().expect("tempdir");
    fs::write(
        tokenizer_dir.path().join("tokenizer.json"),
        MINIMAL_TOKENIZER_JSON,
    )
    .expect("tokenizer.json");
    fs::write(
        tokenizer_dir.path().join("tokenizer_config.json"),
        r#"{"chat_template": "hello"}"#,
    )
    .expect("tokenizer_config.json");
    let path = tokenizer_dir
        .path()
        .to_str()
        .expect("utf-8 path")
        .to_string();

    // A load of `m` held on the registry's per-name lock for 1.5 s: worker
    // A's job, submitted next, waits behind it; worker B registers without
    // a path meanwhile.
    let held_path = path.clone();
    let held = ctx
        .tokenizer_registry
        .load("held", "m", &path, move || async move {
            tokio::time::sleep(Duration::from_millis(1500)).await;
            factory::create_tokenizer_with_chat_template(&held_path, None)
                .map_err(|e| e.to_string())
        });
    let registrations = async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let mut step = tokenizer_step_input(&ctx, &worker, &[("tokenizer_path", &path)]);
        SubmitTokenizerJobStep
            .execute(&mut step)
            .await
            .expect("the step runs");
        tokio::time::sleep(Duration::from_millis(200)).await;
        let mut step = tokenizer_step_input(&ctx, &worker, &[]);
        SubmitTokenizerJobStep
            .execute(&mut step)
            .await
            .expect("the step runs");
        router
            .route_messages_count_tokens(None, &tenant_meta, count_request("m"), "m")
            .await
    };
    let (held, response) = tokio::join!(held, registrations);
    held.expect("the held load succeeds");
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "tokenizer_not_ready",
        "a load in flight keeps the retry hint"
    );
    assert_eq!(
        response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let response = router
            .route_messages_count_tokens(None, &tenant_meta, count_request("m"), "m")
            .await;
        if response.status() == StatusCode::OK {
            break;
        }
        assert!(Instant::now() < deadline, "the model did not answer");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

fn message_request(body: serde_json::Value) -> CreateMessageRequest {
    serde_json::from_value(body).expect("message request")
}

fn sse_event(frame: &str, name: &str) -> serde_json::Value {
    let prefix = format!("event: {name}\ndata: ");
    let data = frame
        .strip_prefix(prefix.as_str())
        .unwrap_or_else(|| panic!("expected a {name} frame, got {frame:?}"));
    serde_json::from_str(data).expect("event json")
}

/// `max_tokens: 0` is the router's own answer, as the public API's: a
/// message with no content, `stop_reason: max_tokens` and the prompt's
/// token count in `usage` (the renderer hands back three ids), under the
/// id the pipeline would have given it (the client's `rid` when it sends
/// one); streamed, that message as `message_start`, then `message_delta`
/// with the stop reason and `message_stop`. No worker takes part; a
/// model nobody serves is a 404, and a prompt longer than the smallest
/// context window the model's workers advertise is the pipeline's 400.
#[tokio::test]
async fn max_tokens_zero_is_answered_with_an_empty_message() {
    let ctx = grpc_ctx(regular_routing_mode()).await;
    let tokenizer: Arc<dyn llm_tokenizer::traits::Tokenizer> = Arc::new(
        llm_tokenizer::mock::MockTokenizer::new().with_deferred_chat_ids(vec![11, 12, 13]),
    );
    ctx.tokenizer_registry
        .load("m", "m", "mock", || async move { Ok(tokenizer) })
        .await
        .expect("register the tokenizer");
    let worker = BasicWorkerBuilder::new("grpc://worker:30000")
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .model(ModelCard::new("m"))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    ctx.worker_registry
        .register(Arc::new(worker))
        .expect("register worker");
    let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
    let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let body = json!({
        "model": "m", "max_tokens": 0,
        "messages": [{"role": "user", "content": "hi"}]
    });

    let response = router
        .route_messages(None, &tenant_meta, message_request(body.clone()), "m")
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let message: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(message["type"], "message");
    assert_eq!(message["role"], "assistant");
    assert_eq!(message["content"], json!([]));
    assert_eq!(message["stop_reason"], "max_tokens");
    assert_eq!(message["usage"]["input_tokens"], 3);
    assert_eq!(message["usage"]["output_tokens"], 0);
    assert_eq!(message["usage"]["cache_read_input_tokens"], 0);
    assert!(message["id"]
        .as_str()
        .is_some_and(|id| id.starts_with("msg_")));

    let mut with_rid = body.clone();
    with_rid["rid"] = json!("req-7");
    let response = router
        .route_messages(None, &tenant_meta, message_request(with_rid), "m")
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let message: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(message["id"], "req-7");

    let mut streamed = body.clone();
    streamed["stream"] = json!(true);
    let response = router
        .route_messages(None, &tenant_meta, message_request(streamed), "m")
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream; charset=utf-8")
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let text = String::from_utf8(bytes.to_vec()).expect("utf-8");
    let frames: Vec<&str> = text.split("\n\n").filter(|f| !f.is_empty()).collect();
    assert_eq!(frames.len(), 3, "{text}");
    let start = sse_event(frames[0], "message_start");
    assert_eq!(start["message"]["content"], json!([]));
    assert_eq!(start["message"]["stop_reason"], serde_json::Value::Null);
    assert_eq!(start["message"]["usage"]["input_tokens"], 3);
    let delta = sse_event(frames[1], "message_delta");
    assert_eq!(delta["delta"]["stop_reason"], "max_tokens");
    assert_eq!(delta["usage"]["output_tokens"], 0);
    assert_eq!(delta["usage"]["input_tokens"], 3);
    assert_eq!(
        sse_event(frames[2], "message_stop"),
        json!({"type": "message_stop"})
    );

    let response = router
        .route_messages(
            None,
            &tenant_meta,
            message_request(json!({
                "model": "nope", "max_tokens": 0,
                "messages": [{"role": "user", "content": "hi"}]
            })),
            "nope",
        )
        .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "model_not_found"
    );

    // The window check the pipeline makes after worker selection, here on
    // the smallest window the model's workers advertise: a worker whose
    // card holds the three-token prompt exactly changes nothing; one with
    // a two-token window refuses it as the pipeline would.
    let window_worker = |url: &str, context_length: u32| {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .model(ModelCard::new("m").with_context_length(context_length))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    };
    ctx.worker_registry
        .register(window_worker("grpc://worker-window-3:30000", 3))
        .expect("register worker");
    let response = router
        .route_messages(None, &tenant_meta, message_request(body.clone()), "m")
        .await;
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "a three-token prompt fits a three-token window"
    );
    ctx.worker_registry
        .register(window_worker("grpc://worker-window-2:30000", 2))
        .expect("register worker");
    let response = router
        .route_messages(None, &tenant_meta, message_request(body), "m")
        .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        error::extract_error_code_from_response(&response),
        "context_length_exceeded"
    );
}

/// count_tokens and `max_tokens: 0` render `output_config.effort` as
/// `/v1/messages` renders it, as the `reasoning_effort` template kwarg:
/// the count is that of the prompt with the effort, and an effort of
/// another type is a 400 `invalid_type`.
#[tokio::test]
async fn counts_render_the_output_config_effort() {
    use llm_tokenizer::traits::Tokenizer as _;

    let ctx = grpc_ctx(regular_routing_mode()).await;
    let mut tokenizer =
        llm_tokenizer::TiktokenTokenizer::new(llm_tokenizer::TiktokenModel::Cl100kBase)
            .expect("tokenizer");
    tokenizer
        .set_chat_template(
            "{%- for m in messages -%}{{ m.role }}: {{ m.content }} {% endfor -%}\
             effort={{ reasoning_effort | default('') }}"
                .to_string(),
        )
        .expect("template");
    let tokenizer: Arc<dyn llm_tokenizer::traits::Tokenizer> = Arc::new(tokenizer);
    let tokens = |text: &str| {
        tokenizer
            .encode(text, false)
            .expect("encode")
            .token_ids()
            .len()
    };
    let (unset, high) = (tokens("user: hi effort="), tokens("user: hi effort=high"));
    assert_ne!(unset, high);
    let registered = Arc::clone(&tokenizer);
    ctx.tokenizer_registry
        .load("m", "m", "mock", || async move { Ok(registered) })
        .await
        .expect("register the tokenizer");
    let worker = BasicWorkerBuilder::new("grpc://worker:30000")
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .model(ModelCard::new("m"))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    ctx.worker_registry
        .register(Arc::new(worker))
        .expect("register worker");
    let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
    let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

    for (output_config, input_tokens) in [
        (json!({"effort": "high"}), high),
        (json!({"effort": null}), unset),
    ] {
        let body = json!({
            "model": "m", "output_config": output_config,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let count = router
            .route_messages_count_tokens(
                None,
                &tenant_meta,
                serde_json::from_value(body.clone()).expect("count request"),
                "m",
            )
            .await;
        let mut message = body;
        message["max_tokens"] = json!(0);
        let empty = router
            .route_messages(None, &tenant_meta, message_request(message), "m")
            .await;
        for (response, field) in [(count, "/input_tokens"), (empty, "/usage/input_tokens")] {
            assert_eq!(response.status(), StatusCode::OK, "{output_config}");
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("body");
            let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
            assert_eq!(
                json.pointer(field),
                Some(&json!(input_tokens)),
                "{output_config}"
            );
        }
    }
}

/// A rate-limit backend that counts the reservations closed unsettled.
#[derive(Default)]
struct ClosingBackend {
    closed: AtomicUsize,
}

#[async_trait]
impl RateLimitBackend for ClosingBackend {
    async fn reserve(&self, _request: ReserveRequest) -> ReserveOutcome {
        ReserveOutcome::Admitted
    }
    async fn settle_success(&self, _request_charge_id: Uuid, _usage: UsageSettlement) {}
    async fn close_reserved_only(&self, _request_charge_id: Uuid) {
        self.closed.fetch_add(1, AtomicOrdering::SeqCst);
    }
    async fn abandon(&self, _request_charge_id: Uuid) {}
}

/// The reserved execution hands the run the canonical id, a fresh cell
/// and the model's retry config; the reservation the run made is closed
/// when the answer is not a 2xx and left to its own settlement when it is.
#[tokio::test]
async fn execute_reserved_closes_the_reservation_of_a_failed_answer() {
    let ctx = grpc_ctx(regular_routing_mode()).await;
    let router = GrpcRouter::new(&ctx, Mode::Regular).expect("router");
    let override_retries = router.retry_config.max_retries + 3;
    ctx.worker_registry.set_model_retry_config(
        "canonical-model",
        RetryConfig {
            max_retries: override_retries,
            ..RetryConfig::default()
        },
        true,
    );

    for (status, closed) in [(StatusCode::INTERNAL_SERVER_ERROR, 1), (StatusCode::OK, 0)] {
        let backend = Arc::new(ClosingBackend::default());
        let handle = Arc::new(SharedReservationHandle::new(
            backend.clone() as Arc<dyn RateLimitBackend>,
            Uuid::now_v7(),
        ));
        let response = router
            .execute_reserved(
                "canonical-model".to_string(),
                |model_id, cell, retry_config| async move {
                    assert_eq!(model_id, "canonical-model");
                    assert_eq!(retry_config.max_retries, override_retries);
                    assert!(cell.peek().is_none(), "the cell starts empty");
                    cell.set_for_test(RateLimitOutcome::Admitted(handle));
                    status.into_response()
                },
            )
            .await;
        assert_eq!(response.status(), status);
        assert_eq!(
            backend.closed.load(AtomicOrdering::SeqCst),
            closed,
            "reservation closed after a {status} answer"
        );
    }
}

/// PD-mode completion must honor a per-model retry override, not the router
/// default.
#[tokio::test]
async fn pd_completion_honors_per_model_retry_override() {
    let ctx = grpc_ctx(pd_routing_mode()).await;
    let router = GrpcRouter::new(&ctx, Mode::PrefillDecode).expect("pd router");

    // Router default differs from the override so the assertion is meaningful.
    let default_retries = router.retry_config.max_retries;
    let override_retries = default_retries + 7;
    let override_config = RetryConfig {
        max_retries: override_retries,
        ..RetryConfig::default()
    };
    ctx.worker_registry
        .set_model_retry_config("model-a", override_config, true);

    let resolved = router.resolve_retry_config_for_canonical("model-a");
    assert_eq!(
        resolved.max_retries, override_retries,
        "PD completion must use the per-model override, not the router default"
    );

    let fallback = router.resolve_retry_config_for_canonical("model-without-override");
    assert_eq!(fallback.max_retries, default_retries);
}

/// PD serves /v1/responses: an unknown model is rejected per-request (404),
/// not gated behind a blanket 501, and cancel reaches storage.
#[tokio::test]
async fn pd_debug_counts_include_zmq_workers() {
    let ctx = grpc_ctx(pd_routing_mode()).await;
    for (url, worker_type, mode) in [
        (
            "grpc://prefill:30000",
            WorkerType::Prefill,
            ConnectionMode::Grpc,
        ),
        (
            "ipc:///tmp/smg-test-prefill",
            WorkerType::Prefill,
            ConnectionMode::Zmq,
        ),
        (
            "ipc:///tmp/smg-test-decode",
            WorkerType::Decode,
            ConnectionMode::Zmq,
        ),
    ] {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .connection_mode(mode)
            .model(ModelCard::new("m"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        ctx.worker_registry
            .register(Arc::new(worker))
            .expect("register worker");
    }

    let router = GrpcRouter::new(&ctx, Mode::PrefillDecode).expect("pd router");
    let debug = format!("{router:?}");
    assert!(
        debug.contains("prefill_workers_count: 2"),
        "ZMQ prefill worker missing from debug counts: {debug}"
    );
    assert!(
        debug.contains("decode_workers_count: 1"),
        "ZMQ decode worker missing from debug counts: {debug}"
    );
}

#[tokio::test]
async fn pd_router_serves_responses_and_cancel() {
    let ctx = grpc_ctx(pd_routing_mode()).await;
    let router = GrpcRouter::new(&ctx, Mode::PrefillDecode).expect("pd router");
    let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

    let request = responses_request("missing-model");
    let response = router
        .route_responses(None, &tenant_meta, request.clone(), "missing-model")
        .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let cancel = router
        .cancel_response(None, &tenant_meta, "resp_missing")
        .await;
    assert_eq!(cancel.status(), StatusCode::NOT_FOUND);
}

/// EPD still 501s /v1/responses and cancel.
#[tokio::test]
async fn epd_router_501s_responses_and_cancel() {
    let ctx = grpc_ctx(epd_routing_mode()).await;
    let router = GrpcRouter::new(&ctx, Mode::EncodePrefillDecode).expect("epd router");
    let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

    let request = responses_request("missing-model");
    let response = router
        .route_responses(None, &tenant_meta, request.clone(), "missing-model")
        .await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);

    let cancel = router
        .cancel_response(None, &tenant_meta, "resp_missing")
        .await;
    assert_eq!(cancel.status(), StatusCode::NOT_IMPLEMENTED);
}

/// A Responses request addressed to an alias must reach the Responses
/// layer under the canonical model ID. That layer builds its SSE events
/// and tool-call responses from this request rather than from the pipeline
/// context, so leaving the alias here is what made the streaming and
/// tool-call paths disagree with the plain-answer path about which model
/// ran.
#[tokio::test]
async fn responses_request_is_canonical_before_the_responses_layer_sees_it() {
    let ctx = grpc_ctx(pd_routing_mode()).await;
    for (url, worker_type) in [
        ("grpc://prefill:30000", WorkerType::Prefill),
        ("grpc://decode:30000", WorkerType::Decode),
    ] {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .model(ModelCard::new("canonical-model").with_alias("model-alias"))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        ctx.worker_registry
            .register(Arc::new(worker))
            .expect("register worker");
    }

    let request = responses_request("model-alias");
    let (body, model_id) =
        canonicalize_responses_request(&ctx.worker_registry, &request, "model-alias");
    assert_eq!(
        body.model, "canonical-model",
        "the Responses layer must see the canonical model ID"
    );
    assert_eq!(
        model_id, "canonical-model",
        "worker selection must use the canonical model ID"
    );

    // The canonical ID is left alone, and costs no copy.
    let already_canonical = responses_request("canonical-model");
    let (body, model_id) =
        canonicalize_responses_request(&ctx.worker_registry, &already_canonical, "canonical-model");
    assert!(matches!(body, Cow::Borrowed(_)));
    assert!(matches!(model_id, Cow::Borrowed(_)));
}

/// A per-model retry override must survive an alias. The override is keyed
/// by canonical model ID and is read before the pipeline canonicalizes, so
/// without its own resolution an alias silently falls back to the router
/// default.
#[tokio::test]
async fn retry_override_survives_a_model_alias() {
    let ctx = grpc_ctx(pd_routing_mode()).await;
    let worker = BasicWorkerBuilder::new("grpc://decode:30000")
        .worker_type(WorkerType::Decode)
        .connection_mode(ConnectionMode::Grpc)
        .model(ModelCard::new("canonical-model").with_alias("model-alias"))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    ctx.worker_registry
        .register(Arc::new(worker))
        .expect("register worker");

    let router = GrpcRouter::new(&ctx, Mode::PrefillDecode).expect("pd router");
    let default_retries = router.retry_config.max_retries;
    let override_retries = default_retries + 7;
    ctx.worker_registry.set_model_retry_config(
        "canonical-model",
        RetryConfig {
            max_retries: override_retries,
            ..RetryConfig::default()
        },
        true,
    );

    // Mirrors the real call sequence every route_*_impl uses: resolve the
    // canonical model once, then look up the retry config for it.
    let retry_config_for = |model_id: &str| {
        router.resolve_retry_config_for_canonical(&router.resolve_canonical_model_id(model_id))
    };

    assert_eq!(
        retry_config_for("canonical-model").max_retries,
        override_retries
    );
    assert_eq!(
        retry_config_for("model-alias").max_retries,
        override_retries,
        "an aliased request must get the same retry override as the canonical ID"
    );
    assert_eq!(
        retry_config_for("unrelated-model").max_retries,
        default_retries
    );
}

/// A gRPC engine on loopback that answers every generate call with
/// `output_ids`: one chunk carrying them, then the complete frame.
#[derive(Clone)]
struct ScriptedEngine {
    output_ids: Vec<u32>,
}

type ScriptedStream<T> = std::pin::Pin<Box<dyn futures::Stream<Item = Result<T, Status>> + Send>>;

#[tonic::async_trait]
impl TokenSpeedScheduler for ScriptedEngine {
    type GenerateStream = ScriptedStream<ts::GenerateResponse>;
    type SubscribeKvEventsStream = ScriptedStream<common::KvEventBatch>;
    type GetTokenizerStream = ScriptedStream<common::GetTokenizerChunk>;

    async fn generate(
        &self,
        request: TonicRequest<ts::GenerateRequest>,
    ) -> Result<TonicResponse<Self::GenerateStream>, Status> {
        use ts::generate_response::Response as GenResp;
        let request_id = request.into_inner().request_id;
        let completion_tokens = self.output_ids.len() as u32;
        let frames = vec![
            Ok(ts::GenerateResponse {
                request_id: request_id.clone(),
                response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                    token_ids: self.output_ids.clone(),
                    prompt_tokens: 3,
                    completion_tokens,
                    index: 0,
                    ..Default::default()
                })),
            }),
            Ok(ts::GenerateResponse {
                request_id,
                response: Some(GenResp::Complete(ts::GenerateComplete {
                    output_ids: self.output_ids.clone(),
                    finish_reason: "stop".to_string(),
                    prompt_tokens: 3,
                    completion_tokens,
                    index: 0,
                    ..Default::default()
                })),
            }),
        ];
        Ok(TonicResponse::new(Box::pin(futures::stream::iter(frames))))
    }

    async fn health_check(
        &self,
        _request: TonicRequest<ts::HealthCheckRequest>,
    ) -> Result<TonicResponse<ts::HealthCheckResponse>, Status> {
        Ok(TonicResponse::new(ts::HealthCheckResponse {
            healthy: true,
            message: "ok".to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: TonicRequest<ts::AbortRequest>,
    ) -> Result<TonicResponse<ts::AbortResponse>, Status> {
        Ok(TonicResponse::new(ts::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: TonicRequest<ts::GetModelInfoRequest>,
    ) -> Result<TonicResponse<ts::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("scripted engine"))
    }

    async fn get_server_info(
        &self,
        _request: TonicRequest<ts::GetServerInfoRequest>,
    ) -> Result<TonicResponse<ts::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("scripted engine"))
    }

    async fn get_loads(
        &self,
        _request: TonicRequest<ts::GetLoadsRequest>,
    ) -> Result<TonicResponse<ts::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("scripted engine"))
    }

    async fn subscribe_kv_events(
        &self,
        _request: TonicRequest<common::SubscribeKvEventsRequest>,
    ) -> Result<TonicResponse<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("scripted engine"))
    }

    async fn flush_cache(
        &self,
        _request: TonicRequest<common::FlushCacheRequest>,
    ) -> Result<TonicResponse<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("scripted engine"))
    }

    async fn start_profile(
        &self,
        _request: TonicRequest<common::StartProfileRequest>,
    ) -> Result<TonicResponse<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("scripted engine"))
    }

    async fn stop_profile(
        &self,
        _request: TonicRequest<common::StopProfileRequest>,
    ) -> Result<TonicResponse<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("scripted engine"))
    }

    async fn get_tokenizer(
        &self,
        _request: TonicRequest<common::GetTokenizerRequest>,
    ) -> Result<TonicResponse<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("scripted engine"))
    }
}

/// Serves a [`ScriptedEngine`] on a free loopback port and returns the port
/// once it accepts connections.
#[expect(
    clippy::disallowed_methods,
    reason = "test helper; the engine's server lives for the test process"
)]
async fn spawn_engine(output_ids: Vec<u32>) -> u16 {
    let port = pick_unused_port().expect("a free port for the scripted engine");
    let addr = format!("127.0.0.1:{port}").parse().expect("engine address");
    tokio::spawn(async move {
        Server::builder()
            .add_service(TokenSpeedSchedulerServer::new(ScriptedEngine {
                output_ids,
            }))
            .serve(addr)
            .await
            .expect("scripted engine server");
    });
    for _ in 0..100 {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return port;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the scripted engine on port {port} never came up");
}

/// A gRPC worker of `worker_type` serving `card`'s model at `url`.
fn register_worker(ctx: &AppContext, url: &str, worker_type: WorkerType, card: ModelCard) {
    let worker = BasicWorkerBuilder::new(url)
        .worker_type(worker_type)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::TokenSpeed)
        .model(card)
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    ctx.worker_registry
        .register(Arc::new(worker))
        .expect("register worker");
}

/// The worker types of a mode's roles.
fn roles(mode: Mode) -> &'static [WorkerType] {
    match mode {
        Mode::Regular => &[WorkerType::Regular],
        Mode::PrefillDecode => &[WorkerType::Prefill, WorkerType::Decode],
        Mode::EncodePrefillDecode => &[WorkerType::Encode, WorkerType::Prefill, WorkerType::Decode],
    }
}

fn routing_mode(mode: Mode) -> RoutingMode {
    match mode {
        Mode::Regular => regular_routing_mode(),
        Mode::PrefillDecode => pd_routing_mode(),
        Mode::EncodePrefillDecode => epd_routing_mode(),
    }
}

/// The output the `qwen3` table reads: a thought, content and
/// one call of the declared `lookup` tool, which the engine answers as the
/// one token the tokenizer decodes into it.
const TABLE_OUTPUT: &str = "<think>plan</think>Hello <tool_call>\n\
                            {\"name\": \"lookup\", \"arguments\": {\"q\": 1}}\n</tool_call>";
const TABLE_OUTPUT_ID: u32 = 100;
/// A chat template whose generation prompt ends at the turn opener the
/// `qwen3` table replays from.
const CHAT_TEMPLATE: &str = "{% for message in messages %}<|im_start|>{{ message.role }}\n\
                             {{ message.content }}<|im_end|>\n{% endfor %}\
                             {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// A tokenizer for `model`: renders `chat_template`, encodes the prompt as
/// three ids, and decodes the one id of `output` into its text.
async fn register_tokenizer_for(
    ctx: &AppContext,
    model: &str,
    chat_template: &str,
    output: (&str, u32),
) {
    let tokenizer: Arc<dyn llm_tokenizer::traits::Tokenizer> = Arc::new(
        llm_tokenizer::mock::MockTokenizer::new()
            .with_chat_template(chat_template)
            .expect("chat template")
            .with_deferred_chat_ids(vec![11, 12, 13])
            .with_tokens(&[output]),
    );
    ctx.tokenizer_registry
        .load(model, model, "mock", || async move { Ok(tokenizer) })
        .await
        .expect("register the tokenizer");
}

/// One chat completion for `model` through `router` under `headers`, with
/// the thought asked apart and one `lookup` tool declared: the status, the
/// `x-smg-parser` response header, and the body.
async fn chat(
    router: &GrpcRouter,
    model: &str,
    headers: &HeaderMap,
) -> (StatusCode, Option<String>, serde_json::Value) {
    let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let body: ChatCompletionRequest = serde_json::from_value(json!({
        "model": model, "separate_reasoning": true,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {"q": {"type": "integer"}}}
        }}]
    }))
    .expect("chat request");
    let response = router
        .route_chat(Some(headers), &tenant_meta, body, model)
        .await;
    let status = response.status();
    let parser = response
        .headers()
        .get("x-smg-parser")
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned());
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| json!({"raw": String::from_utf8_lossy(&bytes)}));
    (status, parser, body)
}

/// The served models of the mixed-fleet test: one the registry knows, whose
/// workers label no parser, and one it does not know, of the family whose
/// table the gateway's tool-call parser name selects.
const KNOWN_MODEL: &str = "Qwen/Qwen3-8B";
const UNKNOWN_MODEL: &str = "acme/ministral-3-14b-instruct-w4a16";
/// The unknown model's output in its family's syntax: the thought, content
/// and call of [`TABLE_OUTPUT`], as the `mistral_v13` table writes them.
const V13_OUTPUT: &str = "[THINK]plan[/THINK]Hello [TOOL_CALLS]lookup[ARGS]{\"q\": 1}";
const V13_OUTPUT_ID: u32 = 101;
/// A chat template whose generation prompt ends at the `mistral_v13`
/// table's turn opener.
const V13_CHAT_TEMPLATE: &str =
    "{% for message in messages %}[INST]{{ message.content }}[/INST]{% endfor %}";

/// One gateway serving two families under `--parser symphony
/// --tool-call-parser mistral_v13`, in both modes: the model the registry
/// knows, whose workers label no parser, is read by the registry's table
/// (its own syntax read apart, not the fleet-wide name's), and the model
/// the registry does not know is read by the fleet-wide name's table; both
/// answers name Symphony.
#[tokio::test]
async fn an_unlabeled_model_the_registry_knows_is_read_by_its_table_beside_the_fleet_wide_names() {
    assert_eq!(
        ::symphony::formats::table_for(KNOWN_MODEL).map(|table| table.name()),
        Some("qwen3")
    );
    assert_eq!(::symphony::formats::table_for(UNKNOWN_MODEL), None);
    for mode in [Mode::Regular, Mode::PrefillDecode] {
        let ctx = grpc_ctx_with_parsers(
            routing_mode(mode),
            ParserChoice::Symphony,
            Some("mistral_v13"),
        )
        .await;
        register_tokenizer_for(
            &ctx,
            KNOWN_MODEL,
            CHAT_TEMPLATE,
            (TABLE_OUTPUT, TABLE_OUTPUT_ID),
        )
        .await;
        register_tokenizer_for(
            &ctx,
            UNKNOWN_MODEL,
            V13_CHAT_TEMPLATE,
            (V13_OUTPUT, V13_OUTPUT_ID),
        )
        .await;
        for worker_type in roles(mode) {
            for (model, output_id) in [
                (KNOWN_MODEL, TABLE_OUTPUT_ID),
                (UNKNOWN_MODEL, V13_OUTPUT_ID),
            ] {
                let port = spawn_engine(vec![output_id]).await;
                register_worker(
                    &ctx,
                    &format!("grpc://127.0.0.1:{port}"),
                    *worker_type,
                    ModelCard::new(model),
                );
            }
        }
        let router = GrpcRouter::new(&ctx, mode).expect("router");
        for model in [KNOWN_MODEL, UNKNOWN_MODEL] {
            let (status, parser, answer) = chat(&router, model, &HeaderMap::new()).await;
            assert_eq!(status, StatusCode::OK, "{mode:?} {model}: {answer}");
            assert_eq!(
                parser.as_deref(),
                Some("symphony"),
                "{mode:?} {model}: {answer}"
            );
            let message = &answer["choices"][0]["message"];
            assert_eq!(
                message["reasoning_content"], "plan",
                "{mode:?} {model}: {answer}"
            );
            assert_eq!(
                message["content"].as_str().map(str::trim),
                Some("Hello"),
                "{mode:?} {model}: {answer}"
            );
            let call = &message["tool_calls"][0]["function"];
            assert_eq!(call["name"], "lookup", "{mode:?} {model}: {answer}");
            let arguments: serde_json::Value =
                serde_json::from_str(call["arguments"].as_str().unwrap_or_default())
                    .unwrap_or_else(|_| json!({}));
            assert_eq!(arguments["q"], 1, "{mode:?} {model}: {answer}");
            assert_eq!(
                answer["choices"][0]["finish_reason"], "tool_calls",
                "{mode:?} {model}: {answer}"
            );
        }
    }
}

/// One gateway under `--parser symphony` with no parser names: the model the registry does
/// not know is read by the legacy parsers, its answer says so, and the request is counted
/// once, under `no_table` and no table; a second request counts a second; the model the
/// registry knows is read by its table and counts nothing. The count is the pipeline's one
/// per request, where request building decides the reader.
#[tokio::test]
async fn a_request_the_legacy_parsers_read_under_symphony_is_counted_once_through_the_pipeline() {
    let ctx =
        grpc_ctx_with_parsers(routing_mode(Mode::Regular), ParserChoice::Symphony, None).await;
    register_tokenizer_for(
        &ctx,
        KNOWN_MODEL,
        CHAT_TEMPLATE,
        (TABLE_OUTPUT, TABLE_OUTPUT_ID),
    )
    .await;
    register_tokenizer_for(
        &ctx,
        UNKNOWN_MODEL,
        V13_CHAT_TEMPLATE,
        (V13_OUTPUT, V13_OUTPUT_ID),
    )
    .await;
    for (model, output_id) in [
        (KNOWN_MODEL, TABLE_OUTPUT_ID),
        (UNKNOWN_MODEL, V13_OUTPUT_ID),
    ] {
        let port = spawn_engine(vec![output_id]).await;
        register_worker(
            &ctx,
            &format!("grpc://127.0.0.1:{port}"),
            WorkerType::Regular,
            ModelCard::new(model),
        );
    }
    let router = GrpcRouter::new(&ctx, Mode::Regular).expect("router");
    // A recorder of this test's own: the runtime is single-threaded, so it sees every
    // record of the requests below.
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let guard = metrics::set_default_local_recorder(&recorder);
    let fallback_lines = |scrape: &str| -> Vec<String> {
        scrape
            .lines()
            .filter(|line| line.starts_with("smg_symphony_fallback_total"))
            .map(str::to_string)
            .collect()
    };
    for count in [" 1", " 2"] {
        let (status, parser, answer) = chat(&router, UNKNOWN_MODEL, &HeaderMap::new()).await;
        assert_eq!(status, StatusCode::OK, "{answer}");
        assert_eq!(parser.as_deref(), Some("legacy"), "{answer}");
        let lines = fallback_lines(&handle.render());
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(
            lines[0].contains("reason=\"no_table\"")
                && lines[0].contains("table=\"none\"")
                && lines[0].ends_with(count),
            "{}",
            lines[0]
        );
    }
    let (status, parser, answer) = chat(&router, KNOWN_MODEL, &HeaderMap::new()).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(parser.as_deref(), Some("symphony"), "{answer}");
    let lines = fallback_lines(&handle.render());
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].ends_with(" 2"), "{}", lines[0]);
    drop(guard);
}

/// The per-request media series count a `max_tokens: 0` request as the
/// route counts it, as a Messages request; a count_tokens call of the same
/// prompt is not a request and leaves the series as they were.
#[test]
fn max_tokens_zero_records_the_requests_media_and_count_tokens_does_not() {
    fn items_total(body: &str) -> f64 {
        body.lines()
            .find(|line| {
                line.starts_with("smg_mm_media_items_total")
                    && line.contains("model=\"m\"")
                    && line.contains("modality=\"image\"")
                    && line.contains("source=\"data_url\"")
            })
            .and_then(|line| line.rsplit(' ').next()?.parse().ok())
            .unwrap_or(0.0)
    }

    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let messages = json!([{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}
    ]}]);
    ::metrics::with_local_recorder(&recorder, || {
        runtime.block_on(async {
            let ctx = grpc_ctx(regular_routing_mode()).await;
            let tokenizer: Arc<dyn llm_tokenizer::traits::Tokenizer> = Arc::new(
                llm_tokenizer::mock::MockTokenizer::new().with_deferred_chat_ids(vec![11, 12, 13]),
            );
            ctx.tokenizer_registry
                .load("m", "m", "mock", || async move { Ok(tokenizer) })
                .await
                .expect("register the tokenizer");
            let worker = BasicWorkerBuilder::new("grpc://worker:30000")
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .model(ModelCard::new("m"))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build();
            ctx.worker_registry
                .register(Arc::new(worker))
                .expect("register worker");
            let router = GrpcRouter::new(&ctx, Mode::Regular).expect("regular router");
            let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

            let count: CountMessageTokensRequest =
                serde_json::from_value(json!({"model": "m", "messages": messages.clone()}))
                    .expect("count request");
            let _ = router
                .route_messages_count_tokens(None, &tenant_meta, count, "m")
                .await;
            assert_eq!(
                items_total(&handle.render()),
                0.0,
                "a token count is not a request"
            );

            let body = json!({"model": "m", "max_tokens": 0, "messages": messages});
            let _ = router
                .route_messages(None, &tenant_meta, message_request(body), "m")
                .await;
            assert_eq!(
                items_total(&handle.render()),
                1.0,
                "the max_tokens: 0 answer is a Messages request"
            );
        });
    });
}
