//! A Responses request the Harmony tool loop serves with two model calls
//! reports the usage of both, in the plain and the streamed form, in the
//! answer and in the stored object.

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
    routers::RouterFactory,
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use smg_grpc_client::tokenspeed_scheduler::tokenspeed_proto as ts;
use smg_test_harness::{
    engine_answers::{TokenSpeedAnswers, TokenSpeedGenerateStream},
    grpc_harness::serve_engine,
};
use tokio::time::timeout;
use tonic::{Request, Response, Status};
use ts::generate_response::Response as GenResp;

use crate::common;

/// The name selects the Harmony pipeline.
const MODEL: &str = "gpt-oss-loop-usage-model";

/// The first generation calls the mock tool server's search tool, the second
/// answers.
const SERVER_CALL: &str = "<|channel|>commentary to=functions.brave_web_search <|constrain|>json<|message|>{\"query\": \"usage\"}<|call|>";
const ANSWER: &str = "<|channel|>final<|message|>The answer after the search<|return|>";

/// Distinct per generation, so one generation's usage alone cannot read as
/// the sum.
const PROMPT_TOKENS: [u32; 2] = [11, 23];

#[derive(Clone)]
struct Turn {
    output_ids: Vec<u32>,
    prompt_tokens: u32,
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
        items.push(Ok(ts::GenerateResponse {
            request_id,
            response: Some(GenResp::Complete(ts::GenerateComplete {
                output_ids: turn.output_ids.clone(),
                finish_reason: "stop".to_string(),
                prompt_tokens: turn.prompt_tokens,
                completion_tokens: generated,
                ..Default::default()
            })),
        }));
        Ok(Response::new(Box::pin(stream::iter(items))))
    }
}

#[tokio::test]
async fn harmony_mcp_usage_includes_every_model_call_in_both_modes() {
    // The encoding's loader may block, which a runtime thread must not.
    let encoding = std::thread::spawn(|| load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss))
        .join()
        .unwrap()
        .unwrap();
    let turns: Vec<Turn> = [SERVER_CALL, ANSWER]
        .into_iter()
        .zip(PROMPT_TOKENS)
        .map(|(turn, prompt_tokens)| Turn {
            output_ids: encoding.tokenizer().encode_with_special_tokens(turn),
            prompt_tokens,
        })
        .collect();
    let output_tokens: u64 = turns.iter().map(|turn| turn.output_ids.len() as u64).sum();
    let input_tokens: u64 = PROMPT_TOKENS.iter().map(|&tokens| u64::from(tokens)).sum();

    // Each form's usage, in the answer and then in the stored object.
    let mut reported = Vec::new();
    for stream in [false, true] {
        let mut mcp = common::mock_mcp_server::MockMCPServer::start()
            .await
            .unwrap();
        let generations = Arc::new(AtomicUsize::new(0));
        let port = serve_engine(
            HarmonyTurns {
                turns: turns.clone(),
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
        let request = serde_json::from_value(json!({
            "model": MODEL, "input": "Hello", "stream": stream, "store": true,
            "tools": [{
                "type": "mcp", "server_label": "test-tools", "server_url": mcp.url(),
                "require_approval": "never",
            }],
        }))
        .unwrap();
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
        let response: Value = if stream {
            std::str::from_utf8(&bytes)
                .unwrap()
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                .find(|event| event["type"] == "response.completed")
                .unwrap()["response"]
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

        // Two model calls around one executed tool call, answered with the
        // second's text.
        assert_eq!(
            generations.load(Ordering::SeqCst),
            2,
            "stream={stream}: {response}"
        );
        assert_eq!(mcp.call_count(), 1, "stream={stream}: {response}");
        assert_eq!(
            response["status"], "completed",
            "stream={stream}: {response}"
        );
        assert!(
            response["output"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["type"] == "message"
                    && item["content"][0]["text"] == "The answer after the search"),
            "stream={stream}: {response}"
        );
        for usage in [&response["usage"], &stored.raw_response["usage"]] {
            reported.push((
                usage["input_tokens"].as_u64(),
                usage["output_tokens"].as_u64(),
                usage["total_tokens"].as_u64(),
            ));
        }
    }
    let both_calls = (
        Some(input_tokens),
        Some(output_tokens),
        Some(input_tokens + output_tokens),
    );
    assert_eq!(reported, vec![both_calls; 4]);
}
