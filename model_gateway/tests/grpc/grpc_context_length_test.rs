//! An input longer than the model's context window is a client error, and
//! the gateway says so before any engine sees it.
//!
//! Before this check, an over-long prompt on the disaggregated gRPC path was
//! forwarded to the prefill engine, whose rejection came back as
//! `500 prefill_worker_failed_to_start` -- a server error for a client
//! mistake, blamed on a healthy worker (#2380). The mock engines here accept
//! anything, so a 400 can only have come from the gateway, and a 200 for the
//! same request means the gateway forwarded it.
//!
//! Workers advertise the window on their model card; the mock tokenizer
//! yields one token per known word, so `"Hello ".repeat(n)` is exactly `n`
//! input tokens.

use std::{sync::Arc, time::Duration};

use axum::http::StatusCode;
use common::{
    grpc_harness::{
        build_grpc_router, grpc_router_config, grpc_worker, serve_canned_worker, Canned,
    },
    test_app::read_body,
};
use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, model_card::ModelCard,
};
use smg::{
    config::RoutingMode,
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};

use crate::common;

const MODEL: &str = "context-length-test-model";
/// Context window every worker in these tests advertises.
const CONTEXT_LENGTH: u32 = 16;
const CONTEXT_LENGTH_EXCEEDED: &str = "context_length_exceeded";
const PREFILL_FAILED: &str = "prefill_worker_failed_to_start";

/// Spawn an in-process mock gRPC worker that accepts any prompt length.
async fn start_mock_grpc_worker() -> u16 {
    serve_canned_worker(Canned {
        model: MODEL,
        gen_delay: Duration::ZERO,
        output_tokens: 2,
    })
    .await
}

fn worker(
    port: u16,
    worker_type: WorkerType,
    context_length: Option<u32>,
) -> Arc<dyn smg::worker::Worker> {
    let mut card = ModelCard::new(MODEL);
    if let Some(len) = context_length {
        card = card.with_context_length(len);
    }
    Arc::new(grpc_worker(port, worker_type, RuntimeType::TokenSpeed, card).build())
}

async fn build_router(
    mode: RoutingMode,
    workers: Vec<Arc<dyn smg::worker::Worker>>,
) -> Box<dyn RouterTrait> {
    let config = grpc_router_config()
        .mode(mode)
        .random_policy()
        .build_unchecked();
    build_grpc_router(config, MODEL, workers).await
}

async fn pd_router(prefill_ctx: Option<u32>, decode_ctx: Option<u32>) -> Box<dyn RouterTrait> {
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    build_router(
        RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        },
        vec![
            worker(prefill, WorkerType::Prefill, prefill_ctx),
            worker(decode, WorkerType::Decode, decode_ctx),
        ],
    )
    .await
}

async fn regular_router(context_length: Option<u32>) -> Box<dyn RouterTrait> {
    let port = start_mock_grpc_worker().await;
    build_router(
        RoutingMode::Regular {
            worker_urls: vec![],
        },
        vec![worker(port, WorkerType::Regular, context_length)],
    )
    .await
}

/// `tokens` words the mock tokenizer knows: exactly `tokens` input tokens.
fn prompt_of(tokens: usize) -> String {
    "Hello ".repeat(tokens).trim_end().to_string()
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn completion(tokens: usize, stream: bool) -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": prompt_of(tokens),
        "max_tokens": 2,
        "stream": stream,
    }))
    .unwrap()
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn batched_completion(token_counts: &[usize]) -> CompletionRequest {
    let prompts: Vec<String> = token_counts.iter().map(|&n| prompt_of(n)).collect();
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": prompts,
        "max_tokens": 2,
    }))
    .unwrap()
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("context-length-test-tenant"))
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn assert_context_length_rejection(response: axum::response::Response) {
    let status = response.status();
    let content_type = response
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = read_body(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        content_type.starts_with("application/json"),
        "rejection must be a JSON error, got {content_type:?}: {body}"
    );
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["error"]["code"], CONTEXT_LENGTH_EXCEEDED, "{body}");
    assert!(
        !body.contains(PREFILL_FAILED),
        "a client mistake must not be reported as a worker failure: {body}"
    );
    let message = json["error"]["message"].as_str().unwrap();
    assert!(
        message.contains(&format!("{CONTEXT_LENGTH} tokens")),
        "message should name the window: {message}"
    );
}

/// The headline case from #2380: over-long prompt, disaggregated gRPC.
#[tokio::test]
async fn pd_over_long_prompt_is_a_400_not_a_prefill_failure() {
    let router = pd_router(Some(CONTEXT_LENGTH), Some(CONTEXT_LENGTH)).await;

    let response = router
        .route_completion(
            None,
            &tenant(),
            completion(CONTEXT_LENGTH as usize + 1, false),
            MODEL,
        )
        .await;

    assert_context_length_rejection(response).await;
}

/// A streaming request is rejected the same way, as a JSON error rather
/// than an event stream (the client has to be able to read the error).
#[tokio::test]
async fn pd_over_long_streaming_prompt_is_a_json_400() {
    let router = pd_router(Some(CONTEXT_LENGTH), Some(CONTEXT_LENGTH)).await;

    let response = router
        .route_completion(
            None,
            &tenant(),
            completion(CONTEXT_LENGTH as usize * 4, true),
            MODEL,
        )
        .await;

    assert_context_length_rejection(response).await;
}

/// The window is a capacity: a prompt that fits (even exactly) goes through
/// to the engines untouched.
#[tokio::test]
async fn pd_prompt_within_the_window_is_forwarded() {
    let router = pd_router(Some(CONTEXT_LENGTH), Some(CONTEXT_LENGTH)).await;

    for tokens in [1usize, CONTEXT_LENGTH as usize / 2, CONTEXT_LENGTH as usize] {
        let response = router
            .route_completion(None, &tenant(), completion(tokens, false), MODEL)
            .await;
        let status = response.status();
        let body = read_body(response).await;
        assert_eq!(status, StatusCode::OK, "{tokens} tokens: {body}");
    }
}

/// A disaggregated prompt must fit both engines; the decode side's tighter
/// window governs even when the prefill side would have taken it.
#[tokio::test]
async fn pd_is_bounded_by_the_tighter_side() {
    let router = pd_router(Some(CONTEXT_LENGTH * 8), Some(CONTEXT_LENGTH)).await;

    let response = router
        .route_completion(
            None,
            &tenant(),
            completion(CONTEXT_LENGTH as usize + 1, false),
            MODEL,
        )
        .await;

    assert_context_length_rejection(response).await;
}

/// Workers that never advertised a window keep today's behavior: the gateway
/// has no opinion and the engine remains the arbiter.
#[tokio::test]
async fn pd_without_an_advertised_window_forwards_as_before() {
    let router = pd_router(None, None).await;

    let response = router
        .route_completion(
            None,
            &tenant(),
            completion(CONTEXT_LENGTH as usize * 8, false),
            MODEL,
        )
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// The window bounds each prompt of a batch, not the batch total: three
/// prompts that together exceed it are fine, one that alone exceeds it is not.
#[tokio::test]
async fn pd_batched_completion_is_bounded_per_prompt() {
    let router = pd_router(Some(CONTEXT_LENGTH), Some(CONTEXT_LENGTH)).await;
    let half = CONTEXT_LENGTH as usize / 2;

    let response = router
        .route_completion(
            None,
            &tenant(),
            batched_completion(&[half, half, half]),
            MODEL,
        )
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let response = router
        .route_completion(
            None,
            &tenant(),
            batched_completion(&[1, CONTEXT_LENGTH as usize + 1]),
            MODEL,
        )
        .await;
    assert_context_length_rejection(response).await;
}

/// The same check guards the single-worker gRPC path.
#[tokio::test]
async fn regular_over_long_prompt_is_a_400() {
    let router = regular_router(Some(CONTEXT_LENGTH)).await;

    let response = router
        .route_completion(
            None,
            &tenant(),
            completion(CONTEXT_LENGTH as usize + 1, false),
            MODEL,
        )
        .await;
    assert_context_length_rejection(response).await;

    let response = router
        .route_completion(
            None,
            &tenant(),
            completion(CONTEXT_LENGTH as usize, false),
            MODEL,
        )
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// Chat requests are bounded on their rendered, tokenized prompt.
#[tokio::test]
async fn regular_over_long_chat_prompt_is_a_400() {
    let router = regular_router(Some(CONTEXT_LENGTH)).await;
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_of(CONTEXT_LENGTH as usize * 4)}],
        "max_tokens": 2,
    }))
    .unwrap();

    let response = router.route_chat(None, &tenant(), request, MODEL).await;

    assert_context_length_rejection(response).await;
}

/// A disaggregated pair's budget is the decode side's to hold: the prefill
/// side runs one token. With a prefill window narrower than the decode
/// window, a budget within the decode window is forwarded and one beyond it
/// is refused naming the decode window; the prompt stays bound by the
/// narrower side.
#[tokio::test]
async fn pd_budget_is_bound_by_the_decode_window_not_the_prefill_window() {
    let (prefill_window, decode_window) = (CONTEXT_LENGTH, CONTEXT_LENGTH * 8);
    let router = pd_router(Some(prefill_window), Some(decode_window)).await;
    let tenant = tenant();
    for (budget, expected) in [
        (prefill_window * 2, StatusCode::OK),
        (decode_window + 1, StatusCode::BAD_REQUEST),
    ] {
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "prompt": prompt_of(prefill_window as usize / 4),
            "max_tokens": budget,
        }))
        .expect("a completion request");
        let response = router.route_completion(None, &tenant, request, MODEL).await;
        let status = response.status();
        let body = read_body(response).await;
        assert_eq!(status, expected, "budget {budget}: {body}");
        if expected == StatusCode::BAD_REQUEST {
            assert!(
                body.contains(&format!("{decode_window} tokens")) && body.contains("max_tokens"),
                "the refusal names the decode window and the field: {body}"
            );
            assert!(!body.contains(PREFILL_FAILED), "{body}");
        }
    }
    let over_long = router
        .route_completion(
            None,
            &tenant,
            completion(prefill_window as usize + 1, false),
            MODEL,
        )
        .await;
    assert_context_length_rejection(over_long).await;
}

/// A completion budget the window could never hold is a client mistake on
/// every route that names one, whatever the model's provider profile: the
/// gateway refuses it before any engine sees it (the mock engines accept
/// anything), naming the window and the budget, and forwards a budget that
/// fits. The input's share of the window stays with the engine, as before.
#[tokio::test]
async fn regular_budget_beyond_the_window_is_a_400_on_every_route() {
    use openai_protocol::{
        generate::GenerateRequest, messages::CreateMessageRequest, responses::ResponsesRequest,
    };

    let router = regular_router(Some(CONTEXT_LENGTH)).await;
    let tenant = tenant();
    let prompt = prompt_of(2);
    let user = prompt.as_str();
    for beyond in [true, false] {
        let budget = if beyond {
            CONTEXT_LENGTH + 1
        } else {
            CONTEXT_LENGTH
        };
        let chat: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": budget,
        }))
        .expect("a chat request");
        let completion: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "prompt": user,
            "max_tokens": budget,
        }))
        .expect("a completion request");
        let messages: CreateMessageRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "max_tokens": budget,
            "messages": [{"role": "user", "content": user}],
        }))
        .expect("a messages request");
        let responses: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "input": user,
            "store": false,
            "max_output_tokens": budget,
        }))
        .expect("a responses request");
        let generate: GenerateRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL,
            "text": user,
            "sampling_params": {"max_new_tokens": budget},
        }))
        .expect("a generate request");
        let answers = [
            ("chat", router.route_chat(None, &tenant, chat, MODEL).await),
            (
                "completions",
                router
                    .route_completion(None, &tenant, completion, MODEL)
                    .await,
            ),
            (
                "messages",
                router.route_messages(None, &tenant, messages, MODEL).await,
            ),
            (
                "responses",
                router
                    .route_responses(None, &tenant, responses, MODEL)
                    .await,
            ),
            (
                "generate",
                router.route_generate(None, &tenant, generate, MODEL).await,
            ),
        ];
        for (route, response) in answers {
            let status = response.status();
            let body = read_body(response).await;
            if !beyond {
                assert_eq!(status, StatusCode::OK, "{route}: {body}");
                continue;
            }
            assert_eq!(status, StatusCode::BAD_REQUEST, "{route}: {body}");
            let json: serde_json::Value = serde_json::from_str(&body)
                .unwrap_or_else(|error| panic!("{route}: {error}: {body}"));
            assert_eq!(
                json["error"]["code"], CONTEXT_LENGTH_EXCEEDED,
                "{route}: {body}"
            );
            let message = json["error"]["message"]
                .as_str()
                .unwrap_or_else(|| panic!("{route}: no message in {body}"));
            assert!(
                message.contains(&format!("{CONTEXT_LENGTH} tokens"))
                    && message.contains(&format!("{budget} tokens for the completion"))
                    && message.contains("max_tokens"),
                "{route}: the refusal names the window, the budget and the field: {message}"
            );
            assert!(
                !body.contains(PREFILL_FAILED),
                "{route}: a client mistake must not be reported as a worker failure: {body}"
            );
        }
    }
}
