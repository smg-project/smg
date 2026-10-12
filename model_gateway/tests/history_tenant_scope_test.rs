//! Stored responses and conversations are scoped to the tenant that created
//! them: with tenant identity on, every route that addresses stored state by
//! id answers another tenant's id as it answers an unknown one, and the owner
//! is served as before. Through the real app (its middleware resolves the
//! tenant from the trusted header) over an in-process scripted engine, with
//! the in-memory history backend. Without tenant identity nothing changes.

#[path = "common/mod.rs"]
mod common;

#[path = "common/scripted_worker.rs"]
mod scripted_worker;

use std::sync::Arc;

use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, Method, Request, StatusCode},
    Router,
};
use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    routers::RouterFactory,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::net::TcpListener;
use tower::ServiceExt;

const MODEL: &str = "tenant-scope-model";
const TENANT_HEADER: &str = "x-smg-tenant-id";

/// The app over one scripted engine; `trust_tenant_header` switches tenant
/// identity on. The engine's task is aborted when the handle drops.
#[expect(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process engine is aborted with its handle"
)]
async fn app(trust_tenant_header: bool) -> (Router, tokio::task::AbortHandle) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let engine = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: 2,
            finish_reasons: vec!["stop"],
            generation: Default::default(),
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
    config.tenant_resolution.trust_tenant_header = trust_tenant_header;
    let tokenizers = Arc::new(TokenizerRegistry::new());
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
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
    let app = common::test_app::create_test_app_with_context(Arc::from(router), context);
    (app, engine.abort_handle())
}

/// One call as `tenant` (the trusted header; `None` sends no header).
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
async fn call(
    app: &Router,
    method: Method,
    path: &str,
    tenant: Option<&str>,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut request = Request::builder().method(method).uri(path);
    if let Some(tenant) = tenant {
        request = request.header(TENANT_HEADER, tenant);
    }
    let request = match body {
        Some(body) => request
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap(),
        None => request.body(Body::empty()).unwrap(),
    };
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

/// A stored response, a conversation and one item, all created by `tenant`.
#[expect(clippy::expect_used, reason = "test helper: failures should panic")]
async fn objects_of(app: &Router, tenant: Option<&str>) -> (String, String, String) {
    let (status, response) = call(
        app,
        Method::POST,
        "/v1/responses",
        tenant,
        Some(json!({"model": MODEL, "input": "Hello", "max_output_tokens": 16})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "completed", "{response}");
    let response_id = response["id"].as_str().expect("a response id").to_string();

    let (status, conversation) = call(
        app,
        Method::POST,
        "/v1/conversations",
        tenant,
        Some(json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{conversation}");
    let conversation_id = conversation["id"]
        .as_str()
        .expect("a conversation id")
        .to_string();
    let (status, items) = call(
        app,
        Method::POST,
        &format!("/v1/conversations/{conversation_id}/items"),
        tenant,
        Some(json!({"items": [{"type": "message", "role": "user", "content": "a secret"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{items}");
    let item_id = items["data"][0]["id"]
        .as_str()
        .expect("an item id")
        .to_string();
    (response_id, conversation_id, item_id)
}

/// Every route that addresses tenant a's objects by id, with the answer
/// another caller must get: the one an unknown id gets on that route.
fn cross_tenant_calls(
    response_id: &str,
    conversation_id: &str,
    item_id: &str,
) -> Vec<(Method, String, Option<Value>, StatusCode, &'static str)> {
    let conversation = format!("/v1/conversations/{conversation_id}");
    vec![
        (
            Method::GET,
            format!("/v1/responses/{response_id}"),
            None,
            StatusCode::NOT_FOUND,
            "not_found",
        ),
        (
            Method::GET,
            format!("/v1/responses/{response_id}/input_items"),
            None,
            StatusCode::NOT_FOUND,
            "not_found",
        ),
        (
            Method::POST,
            format!("/v1/responses/{response_id}/cancel"),
            None,
            StatusCode::NOT_FOUND,
            "response_not_found",
        ),
        (
            Method::GET,
            conversation.clone(),
            None,
            StatusCode::NOT_FOUND,
            "not_found",
        ),
        (
            Method::POST,
            conversation.clone(),
            Some(json!({"metadata": {"owner": "b"}})),
            StatusCode::NOT_FOUND,
            "not_found",
        ),
        (
            Method::GET,
            format!("{conversation}/items"),
            None,
            StatusCode::NOT_FOUND,
            "not_found",
        ),
        (
            Method::POST,
            format!("{conversation}/items"),
            Some(json!({"items": [{"type": "message", "role": "user", "content": "intruder"}]})),
            StatusCode::NOT_FOUND,
            "not_found",
        ),
        (
            Method::GET,
            format!("{conversation}/items/{item_id}"),
            None,
            StatusCode::NOT_FOUND,
            "not_found",
        ),
        (
            Method::POST,
            "/v1/responses".to_string(),
            Some(json!({"model": MODEL, "input": "Hello", "previous_response_id": response_id})),
            StatusCode::BAD_REQUEST,
            "previous_response_not_found",
        ),
        (
            Method::POST,
            "/v1/responses".to_string(),
            Some(json!({"model": MODEL, "input": "Hello", "conversation": conversation_id})),
            StatusCode::NOT_FOUND,
            "conversation_not_found",
        ),
        (
            Method::DELETE,
            format!("{conversation}/items/{item_id}"),
            None,
            StatusCode::NOT_FOUND,
            "not_found",
        ),
        (
            Method::DELETE,
            format!("/v1/responses/{response_id}"),
            None,
            StatusCode::NOT_FOUND,
            "not_found",
        ),
        (
            Method::DELETE,
            conversation,
            None,
            StatusCode::NOT_FOUND,
            "not_found",
        ),
    ]
}

#[tokio::test]
async fn another_tenant_cannot_read_continue_write_into_or_delete_stored_state() {
    let (app, engine) = app(true).await;
    let (response_id, conversation_id, item_id) = objects_of(&app, Some("tenant-a")).await;

    // Tenant b, and a caller without a tenant, get the unknown-id answer on
    // every route, so an id tells them nothing and changes nothing.
    for caller in [Some("tenant-b"), None] {
        for (method, path, body, expected, code) in
            cross_tenant_calls(&response_id, &conversation_id, &item_id)
        {
            let (status, answer) = call(&app, method.clone(), &path, caller, body).await;
            assert_eq!(status, expected, "{caller:?} {method} {path}: {answer}");
            assert_eq!(
                answer["error"]["code"], code,
                "{caller:?} {method} {path}: {answer}"
            );
        }
    }
    // Tenant b's own conversation cannot take a's item either: an
    // `item_reference` to it is as unknown as a missing id, and an item
    // arriving under a's item id is refused rather than served or overwritten.
    let (status, own) = call(
        &app,
        Method::POST,
        "/v1/conversations",
        Some("tenant-b"),
        Some(json!({})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{own}");
    let conv_b = own["id"].as_str().expect("b's conversation").to_string();
    let (status, answer) = call(
        &app,
        Method::POST,
        &format!("/v1/conversations/{conv_b}/items"),
        Some("tenant-b"),
        Some(json!({"items": [{"type": "item_reference", "id": item_id}]})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{answer}");
    assert_eq!(answer["error"]["code"], "not_found", "{answer}");
    let (status, answer) = call(
        &app,
        Method::POST,
        &format!("/v1/conversations/{conv_b}/items"),
        Some("tenant-b"),
        Some(json!({"items": [{"type": "message", "id": item_id, "role": "user", "content": "mine now"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{answer}");
    let (status, listed) = call(
        &app,
        Method::GET,
        &format!("/v1/conversations/{conv_b}/items"),
        Some("tenant-b"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{listed}");
    assert_eq!(
        listed["data"].as_array().map(Vec::len),
        Some(0),
        "nothing of a's reached b's conversation: {listed}"
    );

    // The refusal is the answer an unknown id gets.
    let (status, unknown) = call(
        &app,
        Method::GET,
        "/v1/responses/resp_unknown",
        Some("tenant-b"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (_, refused) = call(
        &app,
        Method::GET,
        &format!("/v1/responses/{response_id}"),
        Some("tenant-b"),
        None,
    )
    .await;
    assert_eq!(refused["error"]["code"], unknown["error"]["code"]);

    // Nothing changed for tenant a, whose own calls work as before.
    let a = Some("tenant-a");
    let (status, stored) = call(
        &app,
        Method::GET,
        &format!("/v1/responses/{response_id}"),
        a,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{stored}");
    assert_eq!(stored["id"], response_id);
    let (status, items) = call(
        &app,
        Method::GET,
        &format!("/v1/responses/{response_id}/input_items"),
        a,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{items}");
    let (status, listed) = call(
        &app,
        Method::GET,
        &format!("/v1/conversations/{conversation_id}/items"),
        a,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{listed}");
    assert_eq!(listed["data"].as_array().map(Vec::len), Some(1), "{listed}");
    let (status, continued) = call(
        &app,
        Method::POST,
        "/v1/responses",
        a,
        Some(json!({"model": MODEL, "input": "Again", "previous_response_id": response_id})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{continued}");
    let (status, in_conversation) = call(
        &app,
        Method::POST,
        "/v1/responses",
        a,
        Some(json!({"model": MODEL, "input": "Again", "conversation": conversation_id})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{in_conversation}");
    let (status, listed) = call(
        &app,
        Method::GET,
        &format!("/v1/conversations/{conversation_id}/items"),
        a,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{listed}");
    assert!(
        listed["data"].as_array().map(Vec::len) > Some(1),
        "a's turn was appended to a's conversation: {listed}"
    );
    for (method, path) in [
        (
            Method::DELETE,
            format!("/v1/conversations/{conversation_id}/items/{item_id}"),
        ),
        (Method::DELETE, format!("/v1/responses/{response_id}")),
        (
            Method::DELETE,
            format!("/v1/conversations/{conversation_id}"),
        ),
    ] {
        let (status, answer) = call(&app, method.clone(), &path, a, None).await;
        assert_eq!(status, StatusCode::OK, "{method} {path}: {answer}");
    }
    engine.abort();
}

#[tokio::test]
async fn without_tenant_identity_stored_state_is_shared_as_before() {
    let (app, engine) = app(false).await;
    // The header is not trusted here: it is no identity, and nothing is scoped.
    let (response_id, conversation_id, item_id) = objects_of(&app, Some("tenant-a")).await;
    for caller in [Some("tenant-b"), None] {
        for (method, path) in [
            (Method::GET, format!("/v1/responses/{response_id}")),
            (
                Method::GET,
                format!("/v1/responses/{response_id}/input_items"),
            ),
            (Method::GET, format!("/v1/conversations/{conversation_id}")),
            (
                Method::GET,
                format!("/v1/conversations/{conversation_id}/items"),
            ),
            (
                Method::GET,
                format!("/v1/conversations/{conversation_id}/items/{item_id}"),
            ),
        ] {
            let (status, answer) = call(&app, method.clone(), &path, caller, None).await;
            assert_eq!(
                status,
                StatusCode::OK,
                "{caller:?} {method} {path}: {answer}"
            );
        }
    }
    engine.abort();
}
