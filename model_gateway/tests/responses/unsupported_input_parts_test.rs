//! A Responses request whose message carries a part the chat pipeline has no
//! form for (an `input_file`; an `input_image` named by a `file_id` alone) is
//! refused at the door under the chat route's `unsupported_content_part`,
//! naming the part, with nothing dispatched. Before, the part was dropped on
//! the way to the chat pipeline and the request answered 200 with the same
//! usage as without it, as if the document had never been sent.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{body::to_bytes, http::StatusCode};
use llm_tokenizer::MockTokenizer;
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

use crate::{common, scripted_worker};

const MODEL: &str = "responses-unsupported-parts-model";

/// The router's answer to one Responses request over one scripted gRPC
/// worker: the status, the JSON body (`Null` for a stream) and the number
/// of generations the worker saw.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test harness: failures should panic; the in-process worker is aborted after the request"
)]
async fn answer(request: Value) -> (StatusCode, Value, usize) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let server = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: 2,
            finish_reasons: vec!["stop"],
            generation: Default::default(),
            requests: Arc::clone(&seen),
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
    let tokenizers = common::tokenizer_registry_with(MODEL, Arc::new(MockTokenizer::new())).await;
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let request = serde_json::from_value(request).unwrap();
    let (status, bytes) = timeout(Duration::from_secs(30), async {
        let response = router.route_responses(None, &tenant, request, MODEL).await;
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (status, bytes)
    })
    .await
    .expect("the request terminates");
    server.abort();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let generations = seen.lock().unwrap().len();
    (status, body, generations)
}

fn responses_request(parts: Value, stream: bool) -> Value {
    json!({
        "model": MODEL,
        "input": [{"role": "user", "content": parts}],
        "max_output_tokens": 2,
        "stream": stream,
    })
}

/// An `input_file` beside the text, and an `input_image` that names only a
/// `file_id`, are refused as `400 unsupported_content_part` naming the part,
/// on the plain and the streamed path alike, and the worker sees no
/// generation; the same message with its text alone is served.
#[tokio::test]
async fn a_part_the_chat_pipeline_has_no_form_for_is_refused_before_dispatch() {
    let cases = [
        (
            "an input_file beside the text",
            json!([
                {"type": "input_text", "text": "Summarize the attached document."},
                {"type": "input_file", "file_data": "JVBERi0xLjQK", "filename": "a.pdf"}
            ]),
            false,
            "\"input_file\"",
        ),
        (
            "an input_image named by a file_id alone, streamed",
            json!([
                {"type": "input_text", "text": "What is in the picture?"},
                {"type": "input_image", "file_id": "file_123"}
            ]),
            true,
            "\"input_image\"",
        ),
    ];
    for (name, parts, stream, part) in cases {
        let (status, body, generations) = answer(responses_request(parts, stream)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{name}: {body}");
        assert_eq!(
            body["error"]["code"], "unsupported_content_part",
            "{name}: {body}"
        );
        assert_eq!(
            body["error"]["type"], "invalid_request_error",
            "{name}: {body}"
        );
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains(part), "{name}: {body}");
        assert_eq!(generations, 0, "{name}: nothing reaches the worker");
    }

    let (status, body, generations) = answer(responses_request(
        json!([{"type": "input_text", "text": "Summarize the attached document."}]),
        false,
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "completed", "{body}");
    assert_eq!(generations, 1, "the text alone is served");
}
