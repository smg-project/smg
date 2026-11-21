//! The HTTP pair router over the policy engine of the gateway's own process
//! and a registry of loopback workers.

use std::{error::Error, time::Duration};

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use serde_json::{json, Value};
use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec},
    engine::PolicyEngine as OwnEngine,
};
use smg_worker::health::circuit_breaker::CircuitBreakerConfig;
use tokio::sync::Semaphore;

use super::{
    tests::{create_test_pd_router, spawn_recording_stub, spawn_stub},
    *,
};
use crate::{routers::grpc::backend_client::BasicWorkerBuilder, tenant::TenantKey};

const MODEL: &str = "model-a";

/// The stock pair document, round robin on both sides.
fn pair_document() -> PolicySpec {
    PolicySpec::pair_presets("round_robin", "round_robin").expect("the pair presets")
}

/// A pair router over `registry` whose attempts ask the engine of this
/// process, round robin on both sides, with no real back-off between them.
fn own_router(registry: Arc<WorkerRegistry>) -> PDRouter {
    own_router_under(registry, pair_document())
}

/// [`own_router`] under `document`.
fn own_router_under(registry: Arc<WorkerRegistry>, mut document: PolicySpec) -> PDRouter {
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the presets validate");
    let engine = PolicyEngine::own(Arc::new(OwnEngine::new(spec)), Arc::clone(&registry));
    PDRouter {
        worker_registry: registry,
        policy_engine: Some(Arc::new(engine)),
        retry_config: RetryConfig {
            initial_backoff_ms: 1,
            max_backoff_ms: 1,
            ..Default::default()
        },
        ..create_test_pd_router()
    }
}

/// A ready HTTP worker of [`MODEL`] at `url` on `side`, in `registry`.
fn listed(registry: &WorkerRegistry, side: WorkerType, url: &str) {
    let worker = BasicWorkerBuilder::new(url)
        .model(ModelCard::new(MODEL))
        .worker_type(side)
        .status(WorkerStatus::Ready);
    assert!(registry.register(Arc::new(worker.build())).is_some());
}

/// The status one unary generate request of [`MODEL`] reads through `router`.
async fn generate(router: &PDRouter) -> StatusCode {
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let request = serde_json::from_value(json!({"text": "hi"})).expect("a generate request");
    let answer = router.route_generate(None, &meta, request, MODEL).await;
    answer.status()
}

/// A decode server that holds each request until its case opens `gate`, then
/// answers it 503, and the count of the requests that reached it. Once open
/// the gate stays open.
async fn spawn_held_refusal(gate: Arc<Semaphore>) -> (String, Arc<AtomicUsize>) {
    let reached = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&reached);
    let app = axum::Router::new().fallback(axum::routing::any(move || {
        let (count, gate) = (Arc::clone(&count), Arc::clone(&gate));
        async move {
            count.fetch_add(1, Ordering::Release);
            let _open = gate.acquire().await;
            let refusal = json!({"error": {"message": "not served now"}});
            (StatusCode::SERVICE_UNAVAILABLE, axum::Json(refusal))
        }
    }));
    (spawn_stub(app).await, reached)
}

/// Where no other pair stands, a decode worker's refusal is replayed for
/// every attempt the retry settings allow, and the request charges that
/// worker's circuit breaker once. An attempt's outcome is recorded through
/// the request's ledger behind the engine's placement, as it is behind the
/// pair selection without the policy controller.
#[tokio::test]
async fn retried_decode_refusals_charge_one_breaker_failure() {
    let (prefill, _) = spawn_recording_stub("{}").await;
    // The gate stands open: each request is refused as it arrives.
    let (refusing, reached) = spawn_held_refusal(Arc::new(Semaphore::new(1))).await;
    let registry = Arc::new(WorkerRegistry::new());
    listed(&registry, WorkerType::Prefill, &prefill);
    let decode = BasicWorkerBuilder::new(refusing)
        .model(ModelCard::new(MODEL))
        .worker_type(WorkerType::Decode)
        .status(WorkerStatus::Ready)
        .circuit_breaker_config(CircuitBreakerConfig {
            failure_threshold: 2,
            ..Default::default()
        });
    let decode: Arc<dyn Worker> = Arc::new(decode.build());
    assert!(registry.register(Arc::clone(&decode)).is_some());
    let router = own_router(registry);

    assert_eq!(generate(&router).await, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        reached.load(Ordering::Acquire),
        router.retry_config.max_retries as usize,
        "the attempts on the refusing decode worker"
    );
    assert!(
        decode.circuit_breaker_can_execute(),
        "one request's retried decode refusals opened the breaker"
    );
}

/// A retry is not sent to the decode worker that just refused its request,
/// even where the rotation's next turn is that worker's again. One request
/// is held at the refusing decode while another takes the free decode's
/// turn, so the turn behind it is the refusing decode's. The held request's
/// retry names the worker it tried, the engine decides over the other one,
/// and the refusing decode is reached once.
#[tokio::test]
#[expect(
    clippy::disallowed_methods,
    reason = "a request's task ends with the test's runtime"
)]
async fn a_retry_leaves_the_decode_that_refused_though_its_turn_comes_again(
) -> Result<(), Box<dyn Error>> {
    let (prefill, _) = spawn_recording_stub("{}").await;
    let gate = Arc::new(Semaphore::new(0));
    let (refusing, reached) = spawn_held_refusal(Arc::clone(&gate)).await;
    let (free, _) = spawn_recording_stub(r#"{"text":"done"}"#).await;
    let registry = Arc::new(WorkerRegistry::new());
    listed(&registry, WorkerType::Prefill, &prefill);
    listed(&registry, WorkerType::Decode, &refusing);
    listed(&registry, WorkerType::Decode, &free);
    let router = Arc::new(own_router(registry));
    let sent = || {
        let router = Arc::clone(&router);
        tokio::spawn(async move { generate(&router).await })
    };

    // The request whose first attempt takes the refusing decode's turn is
    // held there. One that takes the free decode's turn is served at once.
    let mut held = sent();
    for _ in 0..2_000 {
        if reached.load(Ordering::Acquire) != 0 {
            break;
        }
        if held.is_finished() {
            assert_eq!(held.await?, StatusCode::OK);
            held = sent();
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(reached.load(Ordering::Acquire), 1, "no request was held");

    // Another request takes the next turn, the free decode's.
    let other = tokio::time::timeout(Duration::from_secs(30), generate(&router)).await;
    assert_eq!(
        other?,
        StatusCode::OK,
        "the next turn was the free decode's"
    );

    gate.add_permits(1);
    assert_eq!(held.await?, StatusCode::OK, "the retry was served");
    let reached = reached.load(Ordering::Acquire);
    assert_eq!(reached, 1, "the retry went back to the decode that refused");
    Ok(())
}

/// A pair of a model that no pair binding of the document names is declined
/// by its binding. The router answers 503 `policy_declined`, as every route
/// answers a request its policy declined, and neither worker is sent it.
#[tokio::test]
async fn a_pair_of_a_model_no_pair_binding_names_is_declined() -> Result<(), Box<dyn Error>> {
    let (prefill, to_prefill) = spawn_recording_stub("{}").await;
    let (decode, to_decode) = spawn_recording_stub(r#"{"text":"done"}"#).await;
    let registry = Arc::new(WorkerRegistry::new());
    listed(&registry, WorkerType::Prefill, &prefill);
    listed(&registry, WorkerType::Decode, &decode);
    let mut document = pair_document();
    document.bindings[0].matcher.model = "another-model".to_owned();
    let router = own_router_under(registry, document);

    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let request = serde_json::from_value(json!({"text": "hi"}))?;
    let answer = router.route_generate(None, &meta, request, MODEL).await;
    assert_eq!(answer.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(answer.into_body(), 4096).await?;
    let body: Value = serde_json::from_slice(&body)?;
    assert_eq!(body["error"]["code"], "policy_declined", "{body}");
    for sent in [to_prefill, to_decode] {
        assert!(sent.lock().expect("the stub's log").is_empty());
    }
    Ok(())
}
