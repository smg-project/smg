//! Shared driver for Responses tests that send scripted Harmony turns through
//! the server-tool loop in plain and streamed forms.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{body::to_bytes, http::StatusCode};
use futures::stream;
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{grpc::backend_client::BasicWorkerBuilder, RouterFactory},
    tenant::TenantKey,
};
use smg_grpc_client::tokenspeed_scheduler::tokenspeed_proto as ts;
use smg_test_harness::{
    engine_answers::{TokenSpeedAnswers, TokenSpeedGenerateStream},
    grpc_harness::serve_engine,
};
use smg_worker::identity::basic::{ConnectionMode, RuntimeType, WorkerType};
use tokio::time::timeout;
use tonic::{Request, Response, Status};
use ts::generate_response::Response as GenResp;

use crate::common;

/// The name selects the Harmony pipeline.
const MODEL: &str = "gpt-oss-loop-usage-model";

pub(super) const ANSWER: &str = "<|channel|>final<|message|>The answer after the search<|return|>";

/// How a scripted generation ends.
#[derive(Clone, Copy)]
pub(super) enum End {
    /// The terminal frame, with this finish reason.
    Finish(&'static str),
    /// The stream ends before its terminal frame.
    Eof,
}

/// One scripted generation.
#[derive(Clone)]
pub(super) struct Turn {
    pub(super) output_ids: Vec<u32>,
    pub(super) prompt_tokens: u32,
    pub(super) end: End,
}

/// The token ids of each Harmony text, its special tokens included.
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
pub(super) fn token_ids(texts: &[&str]) -> Vec<Vec<u32>> {
    let texts: Vec<String> = texts.iter().map(|text| (*text).to_string()).collect();
    // The encoding's loader may block, which a runtime thread must not.
    std::thread::spawn(move || {
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
        texts
            .iter()
            .map(|text| encoding.tokenizer().encode_with_special_tokens(text))
            .collect()
    })
    .join()
    .unwrap()
}

/// Answers its generations in order and counts them.
struct HarmonyTurns {
    turns: Vec<Turn>,
    generations: Arc<AtomicUsize>,
}

#[tonic::async_trait]
impl TokenSpeedAnswers for HarmonyTurns {
    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<TokenSpeedGenerateStream>, Status> {
        let request_id = request.into_inner().request_id;
        let generation = self.generations.fetch_add(1, Ordering::SeqCst);
        let Some(turn) = self.turns.get(generation) else {
            return Err(Status::unavailable("no generation is scripted"));
        };
        // Chunk counters are cumulative.
        let mut generated = 0;
        let mut items: Vec<Result<ts::GenerateResponse, Status>> = turn
            .output_ids
            .chunks(3)
            .map(|ids| {
                generated += ids.len() as u32;
                Ok(ts::GenerateResponse {
                    request_id: request_id.clone(),
                    response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                        token_ids: ids.to_vec(),
                        prompt_tokens: turn.prompt_tokens,
                        completion_tokens: generated,
                        cached_tokens: 0,
                        output_logprobs: None,
                        weight_version: None,
                        index: 0,
                    })),
                })
            })
            .collect();
        if let End::Finish(finish_reason) = turn.end {
            items.push(Ok(ts::GenerateResponse {
                request_id,
                response: Some(GenResp::Complete(ts::GenerateComplete {
                    output_ids: turn.output_ids.clone(),
                    finish_reason: finish_reason.to_string(),
                    prompt_tokens: turn.prompt_tokens,
                    completion_tokens: generated,
                    ..Default::default()
                })),
            }));
        }
        Ok(Response::new(Box::pin(stream::iter(items))))
    }
}

/// What one request through the loop left behind.
pub(super) struct Served {
    /// The response object: the plain answer, or the one the stream's
    /// terminal event carries.
    pub(super) response: Value,
    /// The stream's events in order; none in the plain form.
    pub(super) events: Vec<Value>,
    /// The stored object.
    pub(super) stored: Value,
    /// The generations the worker answered.
    pub(super) generations: usize,
    /// The tool calls the tool server received.
    pub(super) tool_calls: usize,
}

/// One Responses request, plain or streamed, through a router whose one
/// worker answers `turns` in order, with the mock tool server as the
/// request's MCP tool. Protocol, setup, or storage failures panic and stop
/// the calling table; successful responses are returned for collective checks.
pub(super) async fn serve(turns: &[Turn], stream: bool) -> Served {
    serve_with_input(turns, stream, json!("Hello")).await
}

#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper: failures should panic"
)]
pub(super) async fn serve_with_input(turns: &[Turn], stream: bool, input: Value) -> Served {
    let mut mcp = common::mock_mcp_server::MockMCPServer::start()
        .await
        .unwrap();
    let generations = Arc::new(AtomicUsize::new(0));
    let port = serve_engine(
        HarmonyTurns {
            turns: turns.to_vec(),
            generations: Arc::clone(&generations),
        }
        .into_server(),
    )
    .await;
    let worker = Arc::new(
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
        .build_unchecked();
    config.health_check.disable_health_check = true;
    let context = common::create_test_context_with_mock_tokenizer(config, MODEL).await;
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let request = json!({
        "model": MODEL, "input": input, "stream": stream, "store": true,
        "tools": [{
            "type": "mcp", "server_label": "test-tools", "server_url": mcp.url(),
            "require_approval": "never",
        }],
    });
    let request = serde_json::from_value(request).unwrap();
    let (status, bytes) = timeout(Duration::from_secs(30), async {
        let response = router.route_responses(None, &tenant, request, MODEL).await;
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (status, bytes)
    })
    .await
    .unwrap();
    mcp.stop().await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let events: Vec<Value> = if stream {
        std::str::from_utf8(&bytes)
            .unwrap()
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str::<Value>(data).ok())
            .collect()
    } else {
        Vec::new()
    };
    let response: Value = if stream {
        events
            .iter()
            .find(|event| {
                matches!(
                    event["type"].as_str(),
                    Some("response.completed" | "response.incomplete" | "response.failed")
                )
            })
            .expect("the stream's terminal event")["response"]
            .clone()
    } else {
        serde_json::from_slice(&bytes).unwrap()
    };
    let stored = context
        .response_storage
        .get_response(&smg_data_connector::ResponseId::from(
            response["id"].as_str().unwrap(),
        ))
        .await
        .unwrap()
        .unwrap();
    Served {
        response,
        events,
        stored: stored.raw_response,
        generations: generations.load(Ordering::SeqCst),
        tool_calls: mcp.call_count(),
    }
}
