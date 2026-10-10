//! A Messages request that asks for a structured output format, through the
//! gRPC router and a scripted worker: the gateway cannot constrain a Messages
//! answer to a format, so it refuses the request instead of answering with
//! unconstrained text.

#[path = "common/mod.rs"]
mod common;

#[path = "common/scripted_worker.rs"]
mod scripted_worker;

use std::{sync::Arc, time::Duration};

use axum::{body::to_bytes, http::StatusCode};
use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::{net::TcpListener, task::JoinHandle, time::timeout};

const MODEL: &str = "messages-output-format-model";

struct Gateway {
    router: Box<dyn RouterTrait>,
    worker: JoinHandle<()>,
}

impl Drop for Gateway {
    fn drop(&mut self) {
        self.worker.abort();
    }
}

#[expect(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "test helper; the scripted worker is aborted when the gateway is dropped"
)]
async fn serve() -> Gateway {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let worker = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: 2,
            finish_reasons: vec!["stop"],
            generation: Default::default(),
        }
        .serve(listener),
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
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    let tokenizers = Arc::new(TokenizerRegistry::new());
    tokenizers
        .load(MODEL, MODEL, "test", || async move { Ok(tokenizer) })
        .await
        .unwrap();
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    let worker_entry = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::TokenSpeed)
        .model(ModelCard::new(MODEL))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    context
        .worker_registry
        .register(Arc::new(worker_entry))
        .unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    Gateway { router, worker }
}

/// Sends a Messages request with `extra` fields; returns the status and, for
/// a refusal, the `error` object.
#[expect(clippy::unwrap_used, clippy::expect_used, reason = "test helper")]
async fn send(gateway: &Gateway, stream: bool, extra: &Value) -> (StatusCode, Value) {
    let mut request = json!({
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Answer in JSON."}],
        "stream": stream,
    });
    request
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    let request = serde_json::from_value(request).unwrap();
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let response = gateway
        .router
        .route_messages(None, &tenant, request, MODEL)
        .await;
    let status = response.status();
    let bytes = timeout(
        Duration::from_secs(30),
        to_bytes(response.into_body(), 1 << 20),
    )
    .await
    .expect("request should finish")
    .unwrap();
    if status == StatusCode::OK {
        return (status, Value::Null);
    }
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    (status, body["error"].clone())
}

/// `output_format` and `output_config.format` are refused with 400
/// `unsupported_response_format`, naming the field, unary and streamed.
#[tokio::test]
async fn a_messages_output_format_is_refused() {
    let gateway = serve().await;
    let format = json!({"type": "json_schema", "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }});
    for (extra, field) in [
        (json!({"output_format": format}), "output_format"),
        (
            json!({"output_config": {"format": format}}),
            "output_config.format",
        ),
    ] {
        for stream in [false, true] {
            let (status, error) = send(&gateway, stream, &extra).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{extra}: {error}");
            assert_eq!(error["code"], "unsupported_response_format", "{extra}");
            let message = error["message"].as_str().unwrap_or_default();
            assert!(message.starts_with(field), "{message}");
        }
    }
}

/// A null format is no format, and `output_config` without a format is
/// served as before.
#[tokio::test]
async fn a_messages_request_without_an_output_format_is_served() {
    let gateway = serve().await;
    let extras = [
        json!({}),
        json!({"output_format": null}),
        json!({"output_config": {"format": null}}),
        json!({"output_config": {"effort": "high"}}),
    ];
    for extra in &extras {
        for stream in [false, true] {
            let (status, error) = send(&gateway, stream, extra).await;
            assert_eq!(status, StatusCode::OK, "{extra}: {error}");
        }
    }
}
