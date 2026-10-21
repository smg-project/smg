//! Chat Completions logprobs over a real gRPC worker connection: a unary
//! response's `logprobs.content` covers the same tokens as the logprobs on a
//! stream's content deltas.

#[path = "common/mod.rs"]
mod common;

#[path = "common/scripted_tokenizer.rs"]
#[expect(dead_code, reason = "this test decodes chunks only")]
mod scripted_tokenizer;

#[path = "common/scripted_worker.rs"]
mod scripted_worker;

use std::{sync::Arc, time::Duration};

use axum::body::to_bytes;
use llm_tokenizer::{traits::Tokenizer, TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::RouterFactory,
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::{net::TcpListener, time::timeout};

const MODEL: &str = "chat-logprobs-model";

/// The model's output, one chunk per token; the worker sends one more token
/// that decodes to nothing, like an EOS token.
const CHUNKS: &[&str] = &[
    "<think>", "check", " the", " map", "</think>", "Hello", " my", " friend", ".",
];

/// The `(token, logprob)` pairs of a chat response: a unary response's
/// `logprobs.content`, or the logprobs on a stream's content deltas.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process worker is aborted after the request"
)]
async fn chat_logprobs(stream: bool) -> Vec<(String, f64)> {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: u32::try_from(CHUNKS.len() + 1).unwrap(),
            finish_reasons: vec!["stop"],
            generation: Default::default(),
            requests: Default::default(),
        }
        .serve(listener),
    );
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
    config.reasoning_parser = Some("qwen3".into());
    let tokenizers = Arc::new(TokenizerRegistry::new());
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(scripted_tokenizer::ScriptedTokenizer::from_chunks(
            CHUNKS.iter().map(|chunk| chunk.to_string()).collect(),
        ));
    tokenizers
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let request = serde_json::from_value(json!({
        "model": MODEL, "max_tokens": 16, "stream": stream, "logprobs": true,
        "messages": [{"role": "user", "content": "Hello"}],
    }))
    .unwrap();
    let bytes = timeout(Duration::from_secs(30), async {
        let response = router.route_chat(None, &tenant, request, MODEL).await;
        to_bytes(response.into_body(), 1024 * 1024).await.unwrap()
    })
    .await
    .expect("chat response should finish");
    server.abort();
    let body = std::str::from_utf8(&bytes).unwrap();
    let items = |logprobs: &Value| -> Vec<(String, f64)> {
        logprobs["content"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|item| {
                let token = item["token"].as_str().unwrap().to_string();
                (token, item["logprob"].as_f64().unwrap())
            })
            .collect()
    };
    if !stream {
        let response: Value = serde_json::from_str(body).unwrap();
        let message = &response["choices"][0]["message"];
        assert_eq!(message["content"], "Hello my friend.", "{response}");
        assert_eq!(message["reasoning_content"], "check the map", "{response}");
        return items(&response["choices"][0]["logprobs"]);
    }
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str::<Value>(data).unwrap())
        .filter_map(|event| event["choices"].get(0).cloned())
        .filter(|choice| {
            choice["delta"]["content"]
                .as_str()
                .is_some_and(|text| !text.is_empty())
        })
        .flat_map(|choice| items(&choice["logprobs"]))
        .collect()
}

/// A unary response's logprobs cover its content's tokens, as a stream's
/// content deltas do: not the reasoning, its delimiters or the EOS token.
#[tokio::test]
async fn unary_logprobs_cover_the_content_as_a_stream_does() {
    // Token `100 + i` (chunk `i`) reports logprob `-(i + 1)`.
    let content: Vec<(String, f64)> = ["Hello", " my", " friend", "."]
        .iter()
        .map(|token| token.to_string())
        .zip([-6.0, -7.0, -8.0, -9.0])
        .collect();
    assert_eq!(chat_logprobs(false).await, content);
    assert_eq!(chat_logprobs(true).await, content);
}
