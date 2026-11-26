use std::sync::Arc;

use axum::http::StatusCode;
use openai_protocol::responses::ResponsesRequest;
use serde_json::json;
use smg_data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
use smg_mcp::{McpConfig, McpOrchestrator};

use super::{super::streaming::handle_streaming_response, handle_non_streaming_response};
use crate::{
    openai::{
        context::{
            ComponentRefs, PayloadState, RequestContext, ResponsesComponents, SharedComponents,
            WorkerSelection,
        },
        provider::OpenAIProvider,
    },
    test_support::{
        answering, http_client, text_of, RefusedAddress, StubWorker, CREDENTIALED_WORKER_URL,
        MASKED_WORKER_URL,
    },
};

/// A request context past worker selection with its payload ready, the
/// worker at `worker_url`; the request names no tool, so the plain
/// forward is taken.
fn context_for(worker_url: &str) -> RequestContext {
    let request: ResponsesRequest =
        serde_json::from_value(json!({"model": "m", "input": "hi"})).expect("a request");
    let mut ctx = RequestContext::for_responses(
        Arc::new(request),
        None,
        Some("m".to_string()),
        ComponentRefs::Shared(Arc::new(SharedComponents {
            client: http_client(),
        })),
    );
    ctx.state.worker = Some(WorkerSelection {
        worker: StubWorker::at(worker_url),
        provider: Arc::new(OpenAIProvider),
    });
    ctx.state.payload = Some(PayloadState {
        json: json!({}),
        url: format!("{worker_url}/v1/responses"),
    });
    ctx
}

/// A request context past worker selection, the worker at `worker_url`,
/// over a router with MCP wiring and no configured server; the request
/// names one `mcp` tool with no `server_url` under a label no server is
/// held under.
async fn mcp_context_for(worker_url: &str) -> RequestContext {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "m",
        "input": "hi",
        "tools": [{"type": "mcp", "server_label": "conf"}]
    }))
    .expect("a request");
    let components = ResponsesComponents {
        shared: Arc::new(SharedComponents {
            client: http_client(),
        }),
        mcp_orchestrator: Arc::new(
            McpOrchestrator::new(McpConfig::default())
                .await
                .expect("an orchestrator without servers"),
        ),
        mcp_format_registry: Default::default(),
        response_storage: Arc::new(MemoryResponseStorage::new()),
        conversation_storage: Arc::new(MemoryConversationStorage::new()),
        conversation_item_storage: Arc::new(MemoryConversationItemStorage::new()),
    };
    let mut ctx = RequestContext::for_responses(
        Arc::new(request),
        None,
        Some("m".to_string()),
        ComponentRefs::Responses(Arc::new(components)),
    );
    ctx.state.worker = Some(WorkerSelection {
        worker: StubWorker::at(worker_url),
        provider: Arc::new(OpenAIProvider),
    });
    ctx.state.payload = Some(PayloadState {
        json: json!({}),
        url: format!("{worker_url}/v1/responses"),
    });
    ctx
}

/// An `mcp` tool that binds no server is refused on the missing
/// connection before the forward, on both routes: the worker nobody
/// listens at is never reached (a forward there fails with 502).
#[tokio::test]
async fn an_mcp_tool_that_binds_nothing_is_refused_on_both_routes_before_the_forward() {
    let response =
        handle_non_streaming_response(mcp_context_for(CREDENTIALED_WORKER_URL).await).await;
    assert_eq!(response.status(), StatusCode::FAILED_DEPENDENCY);

    let response = handle_streaming_response(mcp_context_for(CREDENTIALED_WORKER_URL).await).await;
    assert_eq!(response.status(), StatusCode::FAILED_DEPENDENCY);
}

/// The forward's failure line names the worker masked: a worker
/// registered with credentials at a port nobody listens on fails the
/// send, and the line carries the masked address, never the
/// credentials. The capture is the tracing subscriber itself, so the
/// rendering is read as a log reader would.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_failed_forward_names_the_worker_masked_in_its_line() {
    let response = handle_non_streaming_response(context_for(CREDENTIALED_WORKER_URL)).await;

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(!logs_contain("secret"));
    assert!(!logs_contain("user:"));
    assert!(logs_contain("Failed to forward request to OpenAI"));
    assert!(logs_contain(&format!("{MASKED_WORKER_URL}/v1/responses")));
}

/// A forward nothing answers: the caller's 502 says why without the
/// worker's address. The HTTP client's own error text ends with the
/// request URL, which the line names masked.
#[tokio::test]
async fn a_failed_forward_answers_without_the_workers_address() {
    let refused = RefusedAddress::reserve();
    let worker_url = format!("http://{}", refused.addr());

    let response = handle_non_streaming_response(context_for(&worker_url)).await;

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = text_of(response).await;
    assert!(!body.contains(&refused.addr().to_string()), "{body}");
}

/// An answer that is no JSON: the caller's 500 says why without the
/// worker's address. The HTTP client's text of a failed body read ends
/// with the request URL.
#[tokio::test]
async fn an_answer_that_is_no_json_is_answered_without_the_workers_address() {
    let addr = answering(200, "not json", false).await;

    let response = handle_non_streaming_response(context_for(&format!("http://{addr}"))).await;

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = text_of(response).await;
    assert!(body.contains("Failed to parse upstream response"), "{body}");
    assert!(!body.contains(&addr.to_string()), "{body}");
}
