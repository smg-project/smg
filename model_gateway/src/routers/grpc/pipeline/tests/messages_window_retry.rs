//! A retried Messages request judges a `length` finish against the window
//! of the worker that answered: the retry's re-selection refreshes the
//! response spec's window along with the workers.

use std::{
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use openai_protocol::{
    messages::CreateMessageRequest, model_card::ModelCard, worker::HealthCheckConfig,
};
use serde_json::{json, Value};

use super::{components, fast_retry_config, spawn_stub, GatedScheduler, MODEL};
use crate::{
    config::PolicyConfig,
    policies::PolicyRegistry,
    routers::grpc::pipeline::{Endpoint, Mode, PipelineDeps, RequestPipeline},
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerRegistry, WorkerType},
};

/// The request's budget, used whole by the answering engine.
const BUDGET: u32 = 150;
/// The refusing worker's window, the budget itself: admitted by the
/// selection (the budget is not over it) and filled by the budget whatever
/// the prompt's count.
const SMALL_WINDOW: u32 = BUDGET;
/// The answering worker's window: the prompt and the budget stay well under it.
const LARGE_WINDOW: u32 = 4096;

/// [`super::register_worker`] with an advertised context window.
fn register_worker_with_window(
    registry: &WorkerRegistry,
    port: u16,
    window: u32,
) -> Arc<dyn Worker> {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL).with_context_length(window))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    registry
        .register(Arc::clone(&worker))
        .expect("register release-test worker");
    worker
}

/// A Messages request with the budget.
fn messages_request() -> Arc<CreateMessageRequest> {
    let request = json!({
        "model": MODEL,
        "max_tokens": BUDGET,
        "messages": [{"role": "user", "content": "hi"}]
    });
    Arc::new(serde_json::from_value(request).expect("a well-formed request"))
}

/// The first attempt lands on an engine whose window the prompt and the
/// budget fill, and which refuses; the retry lands on an engine with a wide
/// window and answers `length` with the budget used whole. The stop reason
/// is `max_tokens`, judged against the answering engine's window; the
/// window selected before the retry would have read it as
/// `model_context_window_exceeded`.
#[tokio::test]
async fn a_retried_length_finish_is_judged_against_the_answering_workers_window() {
    let seen_refusing = Arc::new(Mutex::new(Vec::new()));
    let seen_serving = Arc::new(Mutex::new(Vec::new()));
    let refusing_port = spawn_stub(GatedScheduler {
        fail_always: true,
        seen_request_ids: Arc::clone(&seen_refusing),
        ..Default::default()
    })
    .await;
    let serving_port = spawn_stub(GatedScheduler {
        finish_length: Some(BUDGET),
        seen_request_ids: Arc::clone(&seen_serving),
        ..Default::default()
    })
    .await;
    let calls =
        |seen: &Mutex<Vec<String>>| seen.lock().unwrap_or_else(PoisonError::into_inner).len();

    let worker_registry = Arc::new(WorkerRegistry::new());
    register_worker_with_window(&worker_registry, refusing_port, SMALL_WINDOW);
    register_worker_with_window(&worker_registry, serving_port, LARGE_WINDOW);
    let deps = PipelineDeps::pair(
        Arc::clone(&worker_registry),
        Arc::new(PolicyRegistry::new(PolicyConfig::ConsistentHashing)),
        None,
        None,
        Duration::from_secs(60),
    );
    let pipeline = RequestPipeline::build(Endpoint::Messages, Mode::Regular, &deps)
        .expect("messages pipeline");
    let components = components(worker_registry).await;
    let retry = fast_retry_config(3);

    // Keys hash to either engine; the first one owned by the refusing
    // engine is the case under test.
    for key in (0..64).map(|i| format!("session-{i}")) {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", key.parse().unwrap());
        let refused_before = calls(&seen_refusing);
        let served_before = calls(&seen_serving);
        let response = pipeline
            .execute_messages(
                messages_request(),
                Some(headers),
                MODEL.to_string(),
                Arc::clone(&components),
                None,
                None,
                Some(&retry),
            )
            .await;
        assert_eq!(response.status(), http::StatusCode::OK, "key {key}");
        if calls(&seen_refusing) == refused_before {
            continue;
        }
        assert_eq!(
            calls(&seen_serving),
            served_before + 1,
            "the retry landed on the serving engine"
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("a body");
        let message: Value = serde_json::from_slice(&body).expect("a Messages answer");
        assert_eq!(
            message["stop_reason"], "max_tokens",
            "judged against the answering engine's window of {LARGE_WINDOW}: {message}"
        );
        assert_eq!(message["usage"]["output_tokens"], BUDGET);
        return;
    }
    panic!("no key hashed to the refusing engine");
}
