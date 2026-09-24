//! End-to-end check of the gRPC PD n>1 fan-out against a mock TokenSpeed
//! prefill/decode pair.
//!
//! The mock worker in canned mode answers every generate request with one
//! index-0 completion whatever `n` says, which is what a rendezvous-room
//! engine effectively does with a sample count it cannot serve from a single
//! room. So `n` choices can only come back if the gateway fans the request
//! out into `n` single-sample pairs and restamps each pair's index, which is
//! what these tests pin, non-streaming and streaming.

#[path = "common/mod.rs"]
mod common;

use std::{collections::BTreeSet, sync::Arc, time::Duration};

use llm_tokenizer::{
    chat_template::ChatTemplateParams,
    traits::{ChatTemplateOutput, Tokenizer},
    Decoder, Encoder, Encoding, MockTokenizer, SpecialTokens, TokenizerRegistry,
};
use openai_protocol::{
    completion::CompletionRequest, model_card::ModelCard, worker::HealthCheckConfig,
};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::net::TcpListener;

const MODEL: &str = "pd-fanout-test-model";
/// Tokens the canned mock emits per request; every sample reports exactly this.
const OUTPUT_TOKENS: u32 = 3;

/// The canned backend emits IDs starting at 100, outside MockTokenizer's
/// vocabulary. Decode those IDs as visible text so chat tests exercise deltas.
struct CannedTokenizer(MockTokenizer);

impl Encoder for CannedTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
        self.0.encode(input, add_special_tokens)
    }

    fn encode_batch(
        &self,
        inputs: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        self.0.encode_batch(inputs, add_special_tokens)
    }
}

impl Decoder for CannedTokenizer {
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        let ids: Vec<u32> = ids
            .iter()
            .map(|&id| {
                if (100..100 + OUTPUT_TOKENS).contains(&id) {
                    1
                } else {
                    id
                }
            })
            .collect();
        self.0.decode(&ids, skip_special_tokens)
    }
}

impl Tokenizer for CannedTokenizer {
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn get_special_tokens(&self) -> &SpecialTokens {
        self.0.get_special_tokens()
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }
    fn eos_token_ids(&self) -> &[u32] {
        self.0.eos_token_ids()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn apply_chat_template_with_encoding(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
        assistant_prefix: Option<&str>,
    ) -> anyhow::Result<ChatTemplateOutput> {
        self.0
            .apply_chat_template_with_encoding(messages, params, assistant_prefix)
    }
}

/// Spawn a canned mock gRPC worker in-process: one index-0 completion of
/// `OUTPUT_TOKENS` tokens per request, `n` ignored.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_mock_grpc_worker() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock gRPC worker");
    let port = listener
        .local_addr()
        .expect("mock gRPC worker address")
        .port();
    let cfg = Arc::new(mock_worker::config::Config {
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: OUTPUT_TOKENS,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    port
}

/// A PD gRPC router over one mock prefill and one mock decode worker, both
/// registered as TokenSpeed so the dispatch takes the rendezvous-room path.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_pd_router(prefill_port: u16, decode_port: u16) -> Box<dyn RouterTrait> {
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        })
        .grpc_connection()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;

    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = Arc::new(CannedTokenizer(MockTokenizer::new())) as Arc<dyn Tokenizer>;
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    let app_context =
        common::create_test_context_with_tokenizer_registry(config, tokenizer_registry).await;
    for (port, worker_type) in [
        (prefill_port, WorkerType::Prefill),
        (decode_port, WorkerType::Decode),
    ] {
        let worker = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        app_context
            .worker_registry
            .register(Arc::new(worker))
            .unwrap();
    }
    RouterFactory::create_router(&app_context)
        .await
        .expect("PD gRPC router should build")
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn completion_request(n: u32, stream: bool) -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": "Hello world",
        "n": n,
        "max_tokens": 8,
        "stream": stream,
    }))
    .unwrap()
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-fanout-test-tenant"))
}

#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn read_body(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    String::from_utf8(bytes.to_vec()).expect("utf-8 body")
}

/// Three samples through one rendezvous-room PD pair: the fan-out runs three
/// single-sample pairs and the merged stream carries choices 0, 1 and 2.
#[tokio::test]
async fn n_samples_come_back_as_n_choices_through_a_room_based_pd_pair() {
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    let router = build_pd_router(prefill, decode).await;

    let response = router
        .route_completion(None, &tenant(), completion_request(3, false), MODEL)
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, http::StatusCode::OK, "{body}");

    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let choices = json["choices"].as_array().unwrap();
    let indices: BTreeSet<u64> = choices
        .iter()
        .map(|choice| choice["index"].as_u64().unwrap())
        .collect();
    assert_eq!(indices, BTreeSet::from([0, 1, 2]), "{body}");
    assert_eq!(choices.len(), 3, "{body}");
    // Every sample reports the full prompt once and its own completion:
    // prompt tokens are counted once, completion tokens add up.
    let usage = &json["usage"];
    assert_eq!(usage["prompt_tokens"].as_u64().unwrap(), 1, "{body}");
    assert_eq!(
        usage["completion_tokens"].as_u64().unwrap(),
        u64::from(OUTPUT_TOKENS) * 3,
        "{body}"
    );
}

/// The same request streamed: every choice index shows up in the SSE events
/// and the stream still terminates with `[DONE]`.
#[tokio::test]
async fn streamed_samples_carry_every_choice_index() {
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    let router = build_pd_router(prefill, decode).await;

    let response = router
        .route_completion(None, &tenant(), completion_request(2, true), MODEL)
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, http::StatusCode::OK, "{body}");

    let mut indices = BTreeSet::new();
    let mut done = false;
    for line in body.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload.trim() == "[DONE]" {
            done = true;
            continue;
        }
        let event: serde_json::Value = serde_json::from_str(payload).unwrap();
        for choice in event["choices"].as_array().into_iter().flatten() {
            indices.insert(choice["index"].as_u64().unwrap());
        }
    }
    assert_eq!(indices, BTreeSet::from([0, 1]), "{body}");
    assert!(done, "stream did not terminate with [DONE]: {body}");
}

/// A single sample takes the plain single-pair path unchanged.
#[tokio::test]
async fn a_single_sample_is_one_choice() {
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    let router = build_pd_router(prefill, decode).await;

    let response = router
        .route_completion(None, &tenant(), completion_request(1, false), MODEL)
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, http::StatusCode::OK, "{body}");
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["choices"].as_array().unwrap().len(), 1, "{body}");
    assert_eq!(
        json["usage"]["completion_tokens"].as_u64().unwrap(),
        u64::from(OUTPUT_TOKENS),
        "{body}"
    );
}

/// Both the single-pair and n>1 merged chat paths must attach a cumulative
/// snapshot to every JSON chunk, including role and finish events.
#[tokio::test]
async fn chat_continuous_usage_is_opt_in_for_single_and_multiple_choices() {
    let prefill = start_mock_grpc_worker().await;
    let decode = start_mock_grpc_worker().await;
    let router = build_pd_router(prefill, decode).await;

    for n in [1, 2] {
        for options in [
            serde_json::Value::Null,
            serde_json::json!({"include_usage": true}),
            serde_json::json!({"include_usage": true, "continuous_usage_stats": false}),
            serde_json::json!({"include_usage": false, "continuous_usage_stats": true}),
            serde_json::json!({"include_usage": true, "continuous_usage_stats": true}),
        ] {
            let include = options["include_usage"].as_bool().unwrap_or(false);
            let continuous =
                include && options["continuous_usage_stats"].as_bool().unwrap_or(false);
            let request = serde_json::from_value(serde_json::json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello world"}],
                "n": n,
                "max_tokens": 8,
                "stream": true,
                "stream_options": options,
            }))
            .unwrap();
            let response = router.route_chat(None, &tenant(), request, MODEL).await;
            let status = response.status();
            let body = read_body(response).await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            assert!(body.contains("data: [DONE]"), "{body}");
            let events: Vec<serde_json::Value> = body
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|payload| *payload != "[DONE]")
                .map(|payload| serde_json::from_str(payload).unwrap())
                .collect();
            let mut previous_completion = 0;
            let mut saw_completion_progress = false;
            let mut roles = BTreeSet::new();
            let mut finishes = BTreeSet::new();
            let mut contents = BTreeSet::new();
            for event in &events {
                assert!(event.get("error").is_none(), "{event}");
                let choices = event["choices"].as_array().unwrap();
                assert_eq!(
                    !event["usage"].is_null(),
                    continuous || (include && choices.is_empty()),
                    "{event}"
                );
                if let Some(usage) = event["usage"].as_object() {
                    let completion = usage["completion_tokens"].as_u64().unwrap();
                    assert!(completion >= previous_completion, "{event}");
                    previous_completion = completion;
                    saw_completion_progress |= !choices.is_empty() && completion > 0;
                    assert_eq!(usage["prompt_tokens"], 1, "shared prompt counted once");
                    assert_eq!(usage["total_tokens"], 1 + completion);
                }
                for choice in choices {
                    let index = choice["index"].as_u64().unwrap();
                    if choice["delta"]["role"] == "assistant" {
                        roles.insert(index);
                    }
                    if choice["delta"]["content"]
                        .as_str()
                        .is_some_and(|s| !s.is_empty())
                    {
                        contents.insert(index);
                    }
                    if !choice["finish_reason"].is_null() {
                        finishes.insert(index);
                    }
                }
            }
            let expected: BTreeSet<u64> = (0..n).collect();
            assert_eq!(roles, expected, "{body}");
            assert_eq!(contents, expected, "{body}");
            assert_eq!(finishes, expected, "{body}");
            if continuous {
                assert!(
                    saw_completion_progress,
                    "no progress before final usage: {body}"
                );
            }
            if include {
                assert_eq!(previous_completion, u64::from(OUTPUT_TOKENS) * n);
                assert!(events.last().unwrap()["choices"]
                    .as_array()
                    .unwrap()
                    .is_empty());
            }
        }
    }
}
