//! A Messages request that replays an MCP exchange (`mcp_tool_use`,
//! `mcp_tool_result`) is refused on an engine route by name, before any
//! engine call (`routers/grpc/utils/message_utils.rs`): the engine runs no
//! MCP loop, and a silent drop would answer without the call.

use std::sync::Arc;

use axum::{body::to_bytes, http::StatusCode};
use llm_tokenizer::MockTokenizer;
use openai_protocol::{messages::CreateMessageRequest, model_card::ModelCard};
use serde_json::json;
use smg::{
    middleware::TenantRequestMeta,
    routers::RouterTrait,
    tenant::TenantKey,
    worker::{RuntimeType, WorkerType},
};

use crate::common::{
    engine_answers::VllmAnswers,
    grpc_harness::{
        grpc_router, grpc_router_config, grpc_test_context_with_tokenizer, grpc_worker,
        serve_engine,
    },
};

const MODEL: &str = "mcp-replay-test-model";

/// An engine on the vLLM wire that answers nothing but health: a request that
/// reached it would fail as unscripted, which the refusal must prevent.
#[derive(Clone)]
struct IdleEngine;

#[tonic::async_trait]
impl VllmAnswers for IdleEngine {}

/// A gRPC router over one [`IdleEngine`] serving [`MODEL`] with the mock
/// tokenizer.
async fn router() -> Box<dyn RouterTrait> {
    let port = serve_engine(IdleEngine.into_server()).await;
    let worker = grpc_worker(
        port,
        WorkerType::Regular,
        RuntimeType::Vllm,
        ModelCard::new(MODEL),
    )
    .build();
    let config = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked();
    let app_context =
        grpc_test_context_with_tokenizer(config, MODEL, Arc::new(MockTokenizer::new())).await;
    grpc_router(&app_context, vec![Arc::new(worker)]).await
}

/// The exchange in the assistant turn, as the public API answers one under
/// the MCP beta: refused as a 400 naming the block and its place.
#[tokio::test]
async fn a_replayed_mcp_exchange_is_refused_before_the_engine() {
    let router = router().await;
    let request: CreateMessageRequest = serde_json::from_value(json!({
        "model": MODEL,
        "max_tokens": 16,
        "messages": [
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "content": [
                {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "lookup", "server_name": "s", "input": {"q": "x"}},
                {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "42"}]},
                {"type": "text", "text": "it is 42"}
            ]},
            {"role": "user", "content": "and again?"}
        ]
    }))
    .expect("a Messages request");
    let tenant = TenantRequestMeta::new(TenantKey::new("mcp-replay-test-tenant"));
    let response = router.route_messages(None, &tenant, request, MODEL).await;
    let status = response.status();
    let body = to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("the body");
    let body = String::from_utf8_lossy(&body);
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.contains("process_messages_failed"), "{body}");
    assert!(
        body.contains(
            "messages.1.content.0: block type 'mcp_tool_use' is not supported on an engine route"
        ),
        "{body}"
    );
}
