//! The HTTP pair router over the policy engine of the gateway's own process,
//! for a pair whose dispatch sends its prefill first: under a document that
//! picks the decode after the prefill, the router asks the engine for the
//! decode again behind the prefill's answer.

use std::time::Duration;

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use serde_json::json;
use smg_policy_controller::{
    document::{AppliedSpec, DecodePick, PolicySpec},
    engine::PolicyEngine as OwnEngine,
};
use tokio::sync::Semaphore;

use super::{
    tests::{create_test_pd_router, spawn_recording_stub, spawn_stub},
    *,
};
use crate::{routers::grpc::backend_client::BasicWorkerBuilder, tenant::TenantKey};

const MODEL: &str = "model-a";

/// A pair router over `registry` whose attempts ask the engine of this
/// process, round robin on both sides, under a document that picks a pair's
/// decode as `picked` says.
fn router_picking(registry: Arc<WorkerRegistry>, picked: DecodePick) -> PDRouter {
    let document = PolicySpec::pair_presets("round_robin", "round_robin");
    let mut document = document.expect("the pair presets");
    document.generation = 1;
    document.plans.pd_pair.decode = picked;
    let spec = AppliedSpec::validate(&document).expect("the document validates");
    let engine = PolicyEngine::own(Arc::new(OwnEngine::new(spec)), Arc::clone(&registry));
    PDRouter {
        worker_registry: registry,
        policy_engine: Some(Arc::new(engine)),
        ..create_test_pd_router()
    }
}

/// A ready HTTP worker of [`MODEL`] at `url` on `side`, in `registry`, of a
/// runtime whose pair sends its prefill first.
fn listed(registry: &WorkerRegistry, side: WorkerType, url: &str) -> Arc<dyn Worker> {
    let worker = BasicWorkerBuilder::new(url)
        .model(ModelCard::new(MODEL))
        .worker_type(side)
        .runtime_type(RuntimeType::Vllm)
        .status(WorkerStatus::Ready);
    let worker: Arc<dyn Worker> = Arc::new(worker.build());
    assert!(registry.register(Arc::clone(&worker)).is_some());
    worker
}

/// A prefill server that holds each request until its case opens `gate`,
/// then answers it with no handoff params, and the count of the requests
/// that reached it.
async fn spawn_held_prefill(gate: Arc<Semaphore>) -> (String, Arc<AtomicUsize>) {
    let reached = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&reached);
    let app = axum::Router::new().fallback(axum::routing::any(move || {
        let (count, gate) = (Arc::clone(&count), Arc::clone(&gate));
        async move {
            count.fetch_add(1, Ordering::Release);
            let _open = gate.acquire().await;
            axum::Json(json!({}))
        }
    }));
    (spawn_stub(app).await, reached)
}

/// One chat request through a pair of one held prefill and two decodes,
/// under a document that picks the decode as `picked` says. The second
/// decode is not ready, so the first decision names the first. While the
/// prefill holds its answer the first decode stops being ready and the
/// second comes back. The arrivals at the two decodes come back.
async fn decode_arrivals_behind_a_decode_that_left(picked: DecodePick) -> [usize; 2] {
    let gate = Arc::new(Semaphore::new(0));
    let (prefill, reached) = spawn_held_prefill(Arc::clone(&gate)).await;
    let (first, first_seen) = spawn_recording_stub(r#"{"object":"chat.completion"}"#).await;
    let (second, second_seen) = spawn_recording_stub(r#"{"object":"chat.completion"}"#).await;
    let registry = Arc::new(WorkerRegistry::new());
    listed(&registry, WorkerType::Prefill, &prefill);
    let first = listed(&registry, WorkerType::Decode, &first);
    let second = listed(&registry, WorkerType::Decode, &second);
    second.set_status(WorkerStatus::NotReady);
    let router = router_picking(registry, picked);

    let leaves = async {
        // The prefill holds its answer: its request is in, the decode's is not.
        for _ in 0..2_000 {
            if reached.load(Ordering::Acquire) != 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        first.set_status(WorkerStatus::NotReady);
        second.set_status(WorkerStatus::Ready);
        gate.add_permits(1);
    };
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .expect("a chat request");
    let chat = router.route_chat(None, &meta, Arc::new(request), MODEL);
    let (response, ()) = tokio::join!(chat, leaves);
    assert_eq!(response.status(), StatusCode::OK);
    [&first_seen, &second_seen].map(|seen| seen.lock().map_or(0, |seen| seen.len()))
}

/// Under a document that picks the decode after the prefill, the engine of
/// this process decides the decode again when the prefill has answered. The
/// decode its first decision named left meanwhile, so the request goes to
/// the decode that is ready at that instant.
#[tokio::test]
async fn a_decode_that_left_behind_its_prefill_is_asked_again_and_the_pair_moves() {
    let arrivals = decode_arrivals_behind_a_decode_that_left(DecodePick::AfterPrefill).await;
    assert_eq!(
        arrivals,
        [0, 1],
        "the decode stayed on the worker that left"
    );
}

/// Under the stock document, which picks a pair's decode with its prefill,
/// no decode is asked again: the request goes to the decode its one
/// decision named.
#[tokio::test]
async fn a_decode_picked_with_its_prefill_is_not_asked_again() {
    let arrivals = decode_arrivals_behind_a_decode_that_left(DecodePick::WithPrefill).await;
    assert_eq!(arrivals, [1, 0], "the decode was asked again");
}
