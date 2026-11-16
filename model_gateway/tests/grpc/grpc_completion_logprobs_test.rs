//! Completions logprobs over a real gRPC worker connection: a unary
//! response's `choices[].logprobs` lists every generated token in the legacy
//! shape (`tokens`, `token_logprobs`, `top_logprobs`, `text_offset`), as the
//! engine's own server returns them, and the stream carries the same lists
//! chunk by chunk.

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

use crate::{common, scripted_tokenizer, scripted_worker};

const MODEL: &str = "completion-logprobs-model";

/// The model's output, one chunk per token.
const CHUNKS: &[&str] = &["Red", ",", " green", ",", " blue"];

/// The `logprobs` object of a `logprobs: 2` completion: a unary response's
/// `choices[0].logprobs`, or the stream's chunks' objects joined list by
/// list, in order. The choice's text is checked on the way.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process worker is aborted after the request"
)]
async fn completion_logprobs(stream: bool) -> Value {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: u32::try_from(CHUNKS.len()).unwrap(),
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
        "model": MODEL, "prompt": "Name three primary colours", "max_tokens": 16,
        "temperature": 0, "stream": stream, "logprobs": 2,
    }))
    .unwrap();
    let bytes = timeout(Duration::from_secs(30), async {
        let response = router.route_completion(None, &tenant, request, MODEL).await;
        to_bytes(response.into_body(), 1024 * 1024).await.unwrap()
    })
    .await
    .expect("completion response should finish");
    server.abort();
    let body = std::str::from_utf8(&bytes).unwrap();
    if !stream {
        let response: Value = serde_json::from_str(body).unwrap();
        let choice = &response["choices"][0];
        assert_eq!(choice["text"], CHUNKS.concat(), "{response}");
        assert_eq!(choice["finish_reason"], "stop", "{response}");
        return choice["logprobs"].clone();
    }
    let choices: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str::<Value>(data).unwrap())
        .filter_map(|event| event["choices"].get(0).cloned())
        .collect();
    let text: String = choices
        .iter()
        .filter_map(|choice| choice["text"].as_str())
        .collect();
    assert_eq!(text, CHUNKS.concat(), "{body}");
    let mut joined = json!({
        "tokens": [], "token_logprobs": [], "top_logprobs": [], "text_offset": [],
    });
    for logprobs in choices.iter().map(|choice| &choice["logprobs"]) {
        if logprobs.is_null() {
            continue;
        }
        for list in ["tokens", "token_logprobs", "top_logprobs", "text_offset"] {
            let items = logprobs[list].as_array().expect(list).clone();
            joined[list].as_array_mut().unwrap().extend(items);
        }
    }
    joined
}

/// What a `logprobs: 2` completion of [`CHUNKS`] lists: every generated
/// token with its decode, its logprob (token `i` reports `-(i + 1)`), its
/// two candidates (itself and the next token, half a nat lower) and its
/// character offset in the text.
fn expected_logprobs() -> Value {
    let count = CHUNKS.len();
    let mut offset = 0;
    let text_offset: Vec<usize> = CHUNKS
        .iter()
        .map(|chunk| {
            let start = offset;
            offset += chunk.chars().count();
            start
        })
        .collect();
    let top_logprobs: Vec<Value> = (0..count)
        .map(|i| {
            let logprob = -((i + 1) as f64);
            let mut candidates = serde_json::Map::new();
            candidates.insert(CHUNKS[i].to_string(), json!(logprob));
            candidates.insert(CHUNKS[(i + 1) % count].to_string(), json!(logprob - 0.5));
            Value::Object(candidates)
        })
        .collect();
    json!({
        "tokens": CHUNKS,
        "token_logprobs": (1..=count).map(|i| -(i as f64)).collect::<Vec<f64>>(),
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    })
}

/// A unary response's `logprobs` lists every generated token in the legacy
/// shape, as the engine's own server returns them, and the stream's chunks
/// carry the same lists piece by piece.
#[tokio::test]
async fn unary_and_streamed_completion_logprobs_list_every_generated_token() {
    let expected = expected_logprobs();
    assert_eq!(completion_logprobs(false).await, expected);
    assert_eq!(completion_logprobs(true).await, expected);
}
