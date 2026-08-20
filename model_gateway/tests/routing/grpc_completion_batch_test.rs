//! gRPC /v1/completions prompt-array acceptance (issue #1903).
//!
//! Prompt arrays previously short-circuited in `CompletionPreparationStage`
//! with `batch_prompts_not_supported` (400). They now tokenize per prompt and
//! flow into the shared pipeline in every gRPC mode, so with no workers
//! registered the request must die at worker selection instead.

use std::sync::{Arc, OnceLock};

use axum::http::StatusCode;
use llm_tokenizer::registry::TokenizerRegistry;
use openai_protocol::completion::CompletionRequest;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use smg::{
    app_context::AppContext,
    config::{PolicyConfig, RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    policies::PolicyRegistry,
    routers::{error::extract_error_code_from_response, RouterFactory},
    tenant::TenantKey,
    worker::WorkerRegistry,
};
use smg_data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
use smg_mcp::{McpConfig, McpOrchestrator};
use tool_parser::ParserFactory as ToolParserFactory;

use crate::common::ensure_tokenizer_cached;

const MODEL: &str = "test-model";

fn grpc_modes() -> Vec<RoutingMode> {
    vec![
        RoutingMode::Regular {
            worker_urls: vec![],
        },
        RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        },
        RoutingMode::EncodePrefillDecode {
            encode_urls: vec![],
            prefill_urls: vec![],
            decode_urls: vec![],
            encode_policy: None,
            prefill_policy: None,
            decode_policy: None,
        },
    ]
}

#[expect(
    clippy::expect_used,
    reason = "test setup helper; failures should panic"
)]
async fn grpc_ctx(mode: RoutingMode) -> Arc<AppContext> {
    let config = RouterConfig::builder()
        .mode(mode)
        .grpc_connection()
        .policy(PolicyConfig::Random)
        .host("127.0.0.1")
        .port(3001)
        .max_payload_size(1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(10)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked();

    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    // Blocking download helper; must run off the async runtime.
    let tokenizer_path = tokio::task::spawn_blocking(ensure_tokenizer_cached)
        .await
        .expect("tokenizer download");
    let tokenizer_source = tokenizer_path.to_string_lossy().to_string();
    tokenizer_registry
        .load("test-tokenizer", MODEL, &tokenizer_source, || async {
            llm_tokenizer::factory::create_tokenizer_from_file(&tokenizer_source)
                .map_err(|e| e.to_string())
        })
        .await
        .expect("load tokenizer");

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
            .client(reqwest::Client::new())
            .tokenizer_registry(tokenizer_registry)
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

#[expect(
    clippy::expect_used,
    reason = "test setup helper; failures should panic"
)]
fn completion_request(prompt: serde_json::Value) -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 8,
    }))
    .expect("completion request")
}

/// Every gRPC mode must carry prompt arrays past preparation: no workers are
/// registered, so both scalar and array requests hit the same worker-selection
/// wall instead of the removed array gate.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn grpc_completion_accepts_prompt_arrays_in_every_mode() {
    for mode in grpc_modes() {
        let mode_label = format!("{mode:?}");
        let ctx = grpc_ctx(mode).await;
        let router = RouterFactory::create_router(&ctx).await.expect("router");
        let tenant_meta = TenantRequestMeta::new(TenantKey::new("test-tenant"));

        let scalar = completion_request(serde_json::json!("Hello world"));
        let scalar_response = router
            .route_completion(None, &tenant_meta, scalar, MODEL)
            .await;
        let scalar_status = scalar_response.status();
        let scalar_code = extract_error_code_from_response(&scalar_response).to_string();

        let array = completion_request(serde_json::json!(["Hello world", "Hello test"]));
        let array_response = router
            .route_completion(None, &tenant_meta, array, MODEL)
            .await;

        assert_ne!(
            array_response.status(),
            StatusCode::BAD_REQUEST,
            "array prompt must not be rejected at preparation ({mode_label})"
        );
        let array_code = extract_error_code_from_response(&array_response).to_string();
        assert_ne!(
            array_code, "batch_prompts_not_supported",
            "array gate must be gone ({mode_label})"
        );
        assert_eq!(
            (array_response.status(), array_code),
            (scalar_status, scalar_code),
            "array and scalar prompts must hit the same worker-selection wall ({mode_label})"
        );
    }
}
