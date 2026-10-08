//! OpenAI Decisions through the production HTTP app and a real SGLang gRPC connection.
#![expect(clippy::unwrap_used, reason = "test failures should panic")]

mod common;

use std::sync::{Arc, Mutex};

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    Router,
};
use futures::stream;
use llm_tokenizer::{
    chat_template::ChatTemplateParams, traits::Tokenizer, Decoder, Encoder, Encoding,
    MockTokenizer, SpecialTokens, TokenizerRegistry,
};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    routers::{factory::router_ids, gateway::Gateway, RouterFactory},
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker},
};
use smg_grpc_client::{
    common_proto,
    sglang_proto::{
        self as sg,
        sglang_scheduler_server::{SglangScheduler, SglangSchedulerServer},
    },
};
use tokio::{net::TcpListener, task::JoinHandle};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request as GrpcRequest, Response as GrpcResponse, Status};
use tower::ServiceExt;

const MODEL: &str = "decision-grpc-model";
const ALIAS: &str = "decision-grpc-alias";
const LABELS: [(&str, u32); 6] = [
    ("yes", 1),
    ("no", 2),
    ("A", 3),
    ("B", 4),
    ("0", 5),
    ("1", 6),
];

/// A deterministic tokenizer: answer labels each occupy one token, with all
/// other characters encoded separately. This exercises the real preparation
/// stage's rendering and checks at the answer position without a model download.
struct DecisionTokenizer(MockTokenizer);

impl Encoder for DecisionTokenizer {
    fn encode(&self, mut text: &str, _add_special: bool) -> anyhow::Result<Encoding> {
        let mut ids = Vec::new();
        while !text.is_empty() {
            if let Some((label, id)) = LABELS.iter().find(|(label, _)| text.starts_with(label)) {
                ids.push(*id);
                text = &text[label.len()..];
            } else {
                let ch = text.chars().next().unwrap();
                ids.push(u32::from(ch) + 100);
                text = &text[ch.len_utf8()..];
            }
        }
        Ok(Encoding::Plain(ids))
    }

    fn encode_batch(&self, texts: &[&str], add_special: bool) -> anyhow::Result<Vec<Encoding>> {
        texts
            .iter()
            .map(|text| self.encode(text, add_special))
            .collect()
    }
}

impl Decoder for DecisionTokenizer {
    fn decode(&self, ids: &[u32], _skip_special: bool) -> anyhow::Result<String> {
        Ok(ids
            .iter()
            .map(|id| {
                LABELS
                    .iter()
                    .find(|(_, candidate)| candidate == id)
                    .map(|(label, _)| (*label).to_string())
                    .unwrap_or_else(|| char::from_u32(id.saturating_sub(100)).unwrap().to_string())
            })
            .collect())
    }
}

impl Tokenizer for DecisionTokenizer {
    fn vocab_size(&self) -> usize {
        0x110000 + 100
    }
    fn get_special_tokens(&self) -> &SpecialTokens {
        self.0.get_special_tokens()
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        LABELS
            .iter()
            .find(|(label, _)| *label == token)
            .map(|(_, id)| *id)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        LABELS
            .iter()
            .find(|(_, candidate)| *candidate == id)
            .map(|(label, _)| (*label).to_string())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn apply_chat_template(
        &self,
        messages: &[Value],
        params: ChatTemplateParams,
    ) -> anyhow::Result<String> {
        self.0.apply_chat_template(messages, params)
    }
}

type RpcStream<T> = std::pin::Pin<Box<dyn futures::Stream<Item = Result<T, Status>> + Send>>;

#[derive(Clone, Copy, Debug, Default)]
enum Reply {
    #[default]
    Scores,
    ReorderedCandidates,
    MissingLogprobs,
    MissingSelectedRow,
    ExtraSelectedRow,
    MissingCandidate,
    DuplicateCandidate,
    UnknownCandidate,
    UnequalLengths,
    NonFinite,
    GeneratedToken,
    MismatchedPromptCount,
    NoCompletion,
    InvalidArgument,
    Unavailable,
    UnavailableOnce,
}

#[derive(Clone, Default)]
struct ScoringWorker {
    requests: Arc<Mutex<Vec<sg::GenerateRequest>>>,
    reply: Reply,
}

#[tonic::async_trait]
impl SglangScheduler for ScoringWorker {
    type GenerateStream = RpcStream<sg::GenerateResponse>;
    type GetTokenizerStream = RpcStream<common_proto::GetTokenizerChunk>;
    type SubscribeKvEventsStream = RpcStream<common_proto::KvEventBatch>;

    async fn generate(
        &self,
        request: GrpcRequest<sg::GenerateRequest>,
    ) -> Result<GrpcResponse<Self::GenerateStream>, Status> {
        let request = request.into_inner();
        let request_number = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request.clone());
            requests.len()
        };
        match self.reply {
            Reply::InvalidArgument => {
                return Err(Status::invalid_argument("invalid scoring request"))
            }
            Reply::Unavailable => return Err(Status::unavailable("scoring worker unavailable")),
            Reply::UnavailableOnce if request_number == 1 => {
                return Err(Status::unavailable("retry this scoring request"));
            }
            Reply::NoCompletion => return Ok(GrpcResponse::new(Box::pin(stream::empty()))),
            _ => {}
        }
        let probabilities: [f32; 2] = match request.token_ids_logprob.as_slice() {
            [1, 2] => [0.8, 0.2],
            [3, 4] => [0.25, 0.75],
            [5, 6] => [0.1, 0.9],
            labels => {
                return Err(Status::invalid_argument(format!(
                    "unexpected labels {labels:?}"
                )))
            }
        };
        let mut row = sg::TopLogProbs {
            values: probabilities.into_iter().map(f32::ln).collect(),
            token_ids: request.token_ids_logprob,
        };
        match self.reply {
            Reply::ReorderedCandidates => {
                row.values.reverse();
                row.token_ids.reverse();
            }
            Reply::MissingCandidate => {
                row.values.pop();
                row.token_ids.pop();
            }
            Reply::DuplicateCandidate => row.token_ids[1] = row.token_ids[0],
            Reply::UnknownCandidate => row.token_ids[1] = 98765,
            Reply::UnequalLengths => {
                row.values.pop();
            }
            Reply::NonFinite => row.values[0] = f32::NAN,
            _ => {}
        }
        let rows = match self.reply {
            Reply::MissingSelectedRow => vec![],
            Reply::ExtraSelectedRow => vec![row.clone(), row],
            _ => vec![row],
        };
        let mut complete = sg::GenerateComplete {
            output_ids: vec![],
            prompt_tokens: u32::try_from(request.tokenized.as_ref().unwrap().input_ids.len())
                .unwrap(),
            completion_tokens: 0,
            cached_tokens: 2,
            finish_reason: "length".to_string(),
            output_logprobs: Some(sg::OutputLogProbs {
                token_ids_logprobs: rows,
                ..Default::default()
            }),
            ..Default::default()
        };
        if matches!(self.reply, Reply::MissingLogprobs) {
            complete.output_logprobs = None;
        }
        if matches!(self.reply, Reply::GeneratedToken) {
            complete.output_ids = vec![999];
            complete.completion_tokens = 1;
        }
        if matches!(self.reply, Reply::MismatchedPromptCount) {
            complete.prompt_tokens -= 1;
        }
        let response = sg::GenerateResponse {
            request_id: request.request_id,
            response: Some(sg::generate_response::Response::Complete(complete)),
        };
        Ok(GrpcResponse::new(Box::pin(stream::iter([Ok(response)]))))
    }
    async fn embed(
        &self,
        _: GrpcRequest<sg::EmbedRequest>,
    ) -> Result<GrpcResponse<sg::EmbedResponse>, Status> {
        Err(Status::unimplemented("test scoring worker"))
    }
    async fn health_check(
        &self,
        _: GrpcRequest<sg::HealthCheckRequest>,
    ) -> Result<GrpcResponse<sg::HealthCheckResponse>, Status> {
        Ok(GrpcResponse::new(sg::HealthCheckResponse::default()))
    }
    async fn abort(
        &self,
        _: GrpcRequest<sg::AbortRequest>,
    ) -> Result<GrpcResponse<sg::AbortResponse>, Status> {
        Ok(GrpcResponse::new(sg::AbortResponse {
            success: true,
            ..Default::default()
        }))
    }
    async fn get_model_info(
        &self,
        _: GrpcRequest<sg::GetModelInfoRequest>,
    ) -> Result<GrpcResponse<sg::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("test scoring worker"))
    }
    async fn get_server_info(
        &self,
        _: GrpcRequest<sg::GetServerInfoRequest>,
    ) -> Result<GrpcResponse<sg::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("test scoring worker"))
    }
    async fn get_loads(
        &self,
        _: GrpcRequest<sg::GetLoadsRequest>,
    ) -> Result<GrpcResponse<sg::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("test scoring worker"))
    }
    async fn flush_cache(
        &self,
        _: GrpcRequest<common_proto::FlushCacheRequest>,
    ) -> Result<GrpcResponse<common_proto::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("test scoring worker"))
    }
    async fn start_profile(
        &self,
        _: GrpcRequest<common_proto::StartProfileRequest>,
    ) -> Result<GrpcResponse<common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("test scoring worker"))
    }
    async fn stop_profile(
        &self,
        _: GrpcRequest<common_proto::StopProfileRequest>,
    ) -> Result<GrpcResponse<common_proto::ProfileResponse>, Status> {
        Err(Status::unimplemented("test scoring worker"))
    }
    async fn get_tokenizer(
        &self,
        _: GrpcRequest<common_proto::GetTokenizerRequest>,
    ) -> Result<GrpcResponse<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("test scoring worker"))
    }
    async fn subscribe_kv_events(
        &self,
        _: GrpcRequest<common_proto::SubscribeKvEventsRequest>,
    ) -> Result<GrpcResponse<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("test scoring worker"))
    }
    async fn load_lo_ra_adapter(
        &self,
        _: GrpcRequest<sg::LoadLoRaAdapterRequest>,
    ) -> Result<GrpcResponse<sg::LoadLoRaAdapterResponse>, Status> {
        Err(Status::unimplemented("test scoring worker"))
    }
    async fn unload_lo_ra_adapter(
        &self,
        _: GrpcRequest<sg::UnloadLoRaAdapterRequest>,
    ) -> Result<GrpcResponse<sg::UnloadLoRaAdapterResponse>, Status> {
        Err(Status::unimplemented("test scoring worker"))
    }
    async fn list_loaded_lo_ra_adapters(
        &self,
        _: GrpcRequest<sg::ListLoadedLoRaAdaptersRequest>,
    ) -> Result<GrpcResponse<sg::ListLoadedLoRaAdaptersResponse>, Status> {
        Err(Status::unimplemented("test scoring worker"))
    }
}

struct Fixture {
    app: Router,
    worker: ScoringWorker,
    server: JoinHandle<()>,
    context: Arc<AppContext>,
    registered_worker: Arc<dyn Worker>,
    _rate_limit_config: Option<tempfile::TempDir>,
}

#[derive(Default)]
struct FixtureOptions {
    context_length: Option<u32>,
    token_budget: Option<u32>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn fixture() -> Fixture {
    fixture_with(Reply::Scores, RuntimeType::Sglang, false).await
}

async fn fixture_with(reply: Reply, runtime: RuntimeType, pd: bool) -> Fixture {
    fixture_options(reply, runtime, pd, FixtureOptions::default()).await
}

#[expect(
    clippy::disallowed_methods,
    reason = "test server is aborted when its fixture drops"
)]
async fn fixture_options(
    reply: Reply,
    runtime: RuntimeType,
    pd: bool,
    options: FixtureOptions,
) -> Fixture {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let worker = ScoringWorker {
        reply,
        ..Default::default()
    };
    let service = SglangSchedulerServer::new(worker.clone());
    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(service)
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let mut config = RouterConfig::builder()
        .grpc_connection()
        .regular_mode(vec![])
        .round_robin_policy()
        .build_unchecked();
    config.health_check.disable_health_check = true;
    config.retry.max_retries = 2;
    config.retry.initial_backoff_ms = 1;
    config.retry.max_backoff_ms = 1;
    config.retry.jitter_factor = 0.0;
    let rate_limit_config = options.token_budget.map(|budget| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rate-limit.yaml");
        std::fs::write(
            &path,
            format!(
                "default_policy:\n  tokens_per_minute: {budget}\n  requests_per_minute: 1000\n"
            ),
        )
        .unwrap();
        config.tenant_rate_limit_enabled = true;
        config.tenant_rate_limit_config = Some(path.to_str().unwrap().to_string());
        dir
    });
    if pd {
        config.mode = RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        };
    }
    let tokenizers = Arc::new(TokenizerRegistry::new());
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(DecisionTokenizer(MockTokenizer::new()));
    tokenizers
        .load("decision-tokenizer", MODEL, "test", || async move {
            Ok(tokenizer)
        })
        .await
        .unwrap();
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    let mut card = ModelCard::new(MODEL).with_alias(ALIAS);
    if let Some(limit) = options.context_length {
        card = card.with_context_length(limit);
    }
    let registered_worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://{address}"))
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(runtime)
            .model(card)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    context
        .worker_registry
        .register(registered_worker.clone())
        .unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    let gateway = Arc::new(Gateway::new(context.worker_registry.clone()));
    let router_id = if pd {
        router_ids::GRPC_PD
    } else {
        router_ids::GRPC_REGULAR
    };
    gateway.register_router(router_id, Arc::from(router));
    let app = common::test_app::create_test_app_with_context(gateway, context.clone());
    Fixture {
        app,
        worker,
        server,
        context,
        registered_worker,
        _rate_limit_config: rate_limit_config,
    }
}

fn request(body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn predicate_request() -> Value {
    json!({
        "model": MODEL,
        "input": "The sky is blue.",
        "questions": [{"type": "predicate", "instructions": "Is the sky blue?"}]
    })
}

async fn call(fixture: &Fixture, body: Value) -> (StatusCode, Value) {
    let response = fixture.app.clone().oneshot(request(body)).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn assert_near(value: &Value, expected: f64) {
    assert!(
        (value.as_f64().unwrap() - expected).abs() < 1e-6,
        "{value} != {expected}"
    );
}

#[tokio::test]
async fn decisions_http_endpoint_reaches_regular_sglang_grpc_scoring() {
    let fixture = fixture().await;
    let (status, result) = call(&fixture, predicate_request()).await;
    assert_eq!(status, StatusCode::OK, "{result}");
    assert_near(&result["answers"][0]["probability"], 0.8);
    assert_eq!(fixture.worker.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn grpc_scoring_preserves_order_duplicate_names_typed_values_and_usage() {
    let fixture = fixture().await;
    let (status, result) = call(
        &fixture,
        json!({
            "model": ALIAS,
            "input": [{"role": "user", "content": "first evidence"},
                {"role": "user", "content": [{"type": "input_text", "text": "second evidence"}]}],
            "questions": [
                {"type": "predicate", "instructions": "Is this positive?"},
                {"type": "choice", "name": "same", "instructions": "Choose", "choices": [
                    {"value": true}, {"value": "true", "description": "string"}]},
                {"type": "score", "name": "same", "instructions": "Score", "levels": [
                    {"label": "low"}, {"label": "high", "description": "excellent"}]}
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{result}");
    assert_eq!(result["model"], MODEL);
    let answers = result["answers"].as_array().unwrap();
    assert_eq!(answers.len(), 3);
    assert_eq!(answers[0]["type"], "predicate");
    assert_eq!(answers[0]["name"], Value::Null);
    assert_near(&answers[0]["probability"], 0.8);
    assert_eq!(answers[1]["type"], "choice");
    assert_eq!(answers[1]["name"], "same");
    assert_eq!(answers[1]["choice"], "true");
    assert_eq!(answers[1]["probabilities"][0]["value"], true);
    assert_eq!(answers[1]["probabilities"][1]["value"], "true");
    assert_near(&answers[1]["probabilities"][0]["probability"], 0.25);
    assert_near(&answers[1]["probabilities"][1]["probability"], 0.75);
    assert_eq!(answers[2]["type"], "score");
    assert_eq!(answers[2]["name"], "same");
    assert_eq!(answers[2]["probabilities"][0]["label"], "low");
    assert_eq!(answers[2]["probabilities"][1]["label"], "high");
    assert_eq!(answers[2]["probabilities"][0]["value"], 0);
    assert_eq!(answers[2]["probabilities"][1]["value"], 1);
    assert_near(&answers[2]["score"], 0.9);
    let requests = fixture.worker.requests.lock().unwrap();
    let input_tokens = requests
        .iter()
        .map(|request| request.tokenized.as_ref().unwrap().input_ids.len())
        .sum::<usize>();
    assert_eq!(result["usage"]["input_tokens"], input_tokens);
    assert_eq!(result["usage"]["output_tokens"], 0);
    assert_eq!(result["usage"]["total_tokens"], input_tokens);
    assert_eq!(result["usage"]["input_tokens_details"]["cached_tokens"], 6);
    assert_eq!(
        result["usage"]["output_tokens_details"]["reasoning_tokens"],
        0
    );
    assert_eq!(requests.len(), 3);
    let mut candidates = requests
        .iter()
        .map(|request| request.token_ids_logprob.clone())
        .collect::<Vec<_>>();
    candidates.sort();
    assert_eq!(candidates, [vec![1, 2], vec![3, 4], vec![5, 6]]);
    let tokenizer = DecisionTokenizer(MockTokenizer::new());
    for request in requests.iter() {
        assert!(!request.stream);
        assert!(request.return_logprob);
        assert_eq!(request.logprob_start_len, -1);
        assert_eq!(request.top_logprobs_num, 0);
        assert_eq!(
            request.sampling_params.as_ref().unwrap().max_new_tokens,
            Some(0)
        );
        assert_eq!(request.sampling_params.as_ref().unwrap().n, 1);
        assert!(!request.require_reasoning);
        assert!(request.mm_inputs.is_none());
        assert!(request.disaggregated_params.is_none());
        let ids = &request.tokenized.as_ref().unwrap().input_ids;
        assert!(!ids.is_empty());
        let rendered = tokenizer.decode(ids, false).unwrap();
        assert!(rendered.contains("first evidence"), "{rendered}");
        assert!(rendered.contains("second evidence"), "{rendered}");
    }
}

#[tokio::test]
async fn selected_logprobs_are_correlated_by_candidate_id() {
    let fixture = fixture_with(Reply::ReorderedCandidates, RuntimeType::Sglang, false).await;
    let (status, result) = call(&fixture, predicate_request()).await;
    assert_eq!(status, StatusCode::OK, "{result}");
    assert_near(&result["answers"][0]["probability"], 0.8);
}

#[tokio::test]
async fn malformed_scoring_responses_are_gateway_errors() {
    for reply in [
        Reply::MissingLogprobs,
        Reply::MissingSelectedRow,
        Reply::ExtraSelectedRow,
        Reply::MissingCandidate,
        Reply::DuplicateCandidate,
        Reply::UnknownCandidate,
        Reply::UnequalLengths,
        Reply::NonFinite,
        Reply::GeneratedToken,
        Reply::MismatchedPromptCount,
    ] {
        let fixture = fixture_with(reply, RuntimeType::Sglang, false).await;
        let (status, result) = call(&fixture, predicate_request()).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY, "{reply:?}: {result}");
        assert_eq!(
            result["error"]["code"], "invalid_decisions_response",
            "{reply:?}: {result}"
        );
        if matches!(reply, Reply::MissingLogprobs | Reply::MissingSelectedRow) {
            assert!(
                result["error"]["message"]
                    .as_str()
                    .unwrap()
                    .to_lowercase()
                    .contains("upgrade"),
                "{result}"
            );
        }
    }
}

#[tokio::test]
async fn incomplete_grpc_stream_is_an_error_instead_of_an_empty_answer() {
    let fixture = fixture_with(Reply::NoCompletion, RuntimeType::Sglang, false).await;
    let (status, result) = call(&fixture, predicate_request()).await;
    assert!(status.is_server_error(), "{result}");
    assert!(result["error"]["message"].is_string(), "{result}");
}

#[tokio::test]
async fn grpc_backend_errors_preserve_client_and_unavailable_statuses() {
    for (reply, expected) in [
        (Reply::InvalidArgument, StatusCode::BAD_REQUEST),
        (Reply::Unavailable, StatusCode::SERVICE_UNAVAILABLE),
    ] {
        let fixture = fixture_with(reply, RuntimeType::Sglang, false).await;
        let (status, result) = call(&fixture, predicate_request()).await;
        assert_eq!(status, expected, "{result}");
        assert!(result["error"]["message"].is_string());
        assert!(!fixture.worker.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn unsupported_grpc_engines_and_pd_do_not_dispatch_scoring() {
    for (runtime, pd) in [
        (RuntimeType::Vllm, false),
        (RuntimeType::TokenSpeed, false),
        (RuntimeType::Trtllm, false),
        (RuntimeType::Sglang, true),
    ] {
        let fixture = fixture_with(Reply::Scores, runtime, pd).await;
        let (status, result) = call(&fixture, predicate_request()).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{result}");
        assert!(fixture.worker.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn explicit_unknown_model_is_rejected_before_grpc_dispatch() {
    let fixture = fixture().await;
    let mut body = predicate_request();
    body["model"] = json!("unknown");
    let (status, result) = call(&fixture, body).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{result}");
    assert!(fixture.worker.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn invalid_or_unsupported_decisions_do_not_dispatch_scoring() {
    let fixture = fixture().await;
    let mut image = predicate_request();
    image["input"] = json!([{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}]);
    let mut extension = predicate_request();
    extension["unsupported_extension"] = json!(false);
    let mut missing_model = predicate_request();
    missing_model.as_object_mut().unwrap().remove("model");
    for body in [image, extension, missing_model] {
        let (status, result) = call(&fixture, body).await;
        assert!(status.is_client_error(), "{result}");
    }
    assert!(fixture.worker.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn mixed_model_pool_routes_only_to_supported_sglang_workers() {
    let supported = fixture().await;
    let unsupported = fixture_with(Reply::InvalidArgument, RuntimeType::Vllm, false).await;
    supported
        .context
        .worker_registry
        .register(unsupported.registered_worker.clone())
        .unwrap();
    // Round-robin must complete a full rotation of the eligible set. If the
    // unsupported worker enters it, one of these requests reaches that worker.
    for _ in 0..4 {
        let (status, result) = call(&supported, predicate_request()).await;
        assert_eq!(status, StatusCode::OK, "{result}");
        assert_near(&result["answers"][0]["probability"], 0.8);
    }
    assert_eq!(supported.worker.requests.lock().unwrap().len(), 4);
    assert!(unsupported.worker.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn scoring_retry_stays_on_sglang_in_a_mixed_model_pool() {
    let supported = fixture_with(Reply::UnavailableOnce, RuntimeType::Sglang, false).await;
    let unsupported = fixture_with(Reply::InvalidArgument, RuntimeType::Vllm, false).await;
    supported
        .context
        .worker_registry
        .register(unsupported.registered_worker.clone())
        .unwrap();
    let (status, result) = call(&supported, predicate_request()).await;
    assert_eq!(status, StatusCode::OK, "{result}");
    assert_near(&result["answers"][0]["probability"], 0.8);
    let requests = supported.worker.requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        2,
        "one failed attempt followed by one success"
    );
    assert_eq!(requests[0].tokenized, requests[1].tokenized);
    assert_eq!(requests[0].token_ids_logprob, requests[1].token_ids_logprob);
    assert_ne!(requests[0].request_id, requests[1].request_id);
    assert!(unsupported.worker.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn scoring_context_boundary_accepts_below_and_rejects_at_limit() {
    // The literal predicate prompt has 103 characters. The fixture tokenizer
    // encodes "yes" and "no" as one token each, giving 100 input tokens.
    for (context_length, expected) in [(101, StatusCode::OK), (100, StatusCode::BAD_REQUEST)] {
        let fixture = fixture_options(
            Reply::Scores,
            RuntimeType::Sglang,
            false,
            FixtureOptions {
                context_length: Some(context_length),
                ..Default::default()
            },
        )
        .await;
        let (status, result) = call(&fixture, predicate_request()).await;
        assert_eq!(status, expected, "context={context_length}: {result}");
        let requests = fixture.worker.requests.lock().unwrap();
        if expected == StatusCode::OK {
            assert_eq!(requests.len(), 1);
            assert_eq!(requests[0].tokenized.as_ref().unwrap().input_ids.len(), 100);
            assert_eq!(
                requests[0].sampling_params.as_ref().unwrap().max_new_tokens,
                Some(0)
            );
            assert_eq!(result["usage"]["output_tokens"], 0);
        } else {
            assert_eq!(result["error"]["code"], "context_length_exceeded");
            assert!(requests.is_empty());
        }
    }
}

#[tokio::test]
async fn each_question_gets_its_own_context_window() {
    let fixture = fixture_options(
        Reply::Scores,
        RuntimeType::Sglang,
        false,
        FixtureOptions {
            context_length: Some(101),
            ..Default::default()
        },
    )
    .await;
    let mut body = predicate_request();
    let question = body["questions"][0].clone();
    body["questions"].as_array_mut().unwrap().push(question);
    let (status, result) = call(&fixture, body).await;
    assert_eq!(status, StatusCode::OK, "{result}");
    assert_eq!(result["usage"]["input_tokens"], 200);
    assert_eq!(result["usage"]["output_tokens"], 0);
    assert_eq!(fixture.worker.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn rate_limit_accounts_for_all_question_prompts_without_output_tokens() {
    let fixture = fixture_options(
        Reply::Scores,
        RuntimeType::Sglang,
        false,
        FixtureOptions {
            token_budget: Some(150),
            ..Default::default()
        },
    )
    .await;
    let mut body = predicate_request();
    let question = body["questions"][0].clone();
    body["questions"].as_array_mut().unwrap().push(question);
    let (status, result) = call(&fixture, body).await;
    assert_eq!(
        status,
        StatusCode::TOO_MANY_REQUESTS,
        "two 100-token prompts exceed 150: {result}"
    );
    assert!(fixture.worker.requests.lock().unwrap().is_empty());
    let (status, result) = call(&fixture, predicate_request()).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "one 100-token prompt fits 150: {result}"
    );
    assert_eq!(result["usage"]["input_tokens"], 100);
    assert_eq!(result["usage"]["output_tokens"], 0);
    let (status, result) = call(&fixture, predicate_request()).await;
    assert_eq!(
        status,
        StatusCode::TOO_MANY_REQUESTS,
        "only 50 input tokens remain: {result}"
    );
    assert_eq!(fixture.worker.requests.lock().unwrap().len(), 1);
}
