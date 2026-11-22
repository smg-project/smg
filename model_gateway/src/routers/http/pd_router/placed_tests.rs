//! MoRI choices over the live HTTP PD placement path.
//!
//! Upstream #7746 removed the experimental gateway controller integration.
//! The two historical `hosted_moriio_` case IDs remain, while their routing,
//! admission, affinity and cleanup assertions now exercise PolicyRegistry.

use std::{any::Any, collections::HashSet, error::Error, sync::Mutex, time::Duration};

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use serde_json::json;

use super::{
    tests::{
        create_test_pd_router, spawn_choice_decode_stub_with_model, spawn_moriio_prefill_stub,
        StubHandoff,
    },
    *,
};
use crate::{
    config::types::{PolicyConfig, RoutingKeyOverrideConfig},
    policies::{LoadBalancingPolicy, ManualPolicy, PinState, SelectWorkerInfo},
    routers::grpc::backend_client::BasicWorkerBuilder,
    tenant::TenantKey,
};

type Case = Result<(), Box<dyn Error>>;
const MODEL: &str = "m";

#[derive(Clone, Debug)]
struct ChoicePlacement {
    role: WorkerRole,
    rid_key: Option<String>,
    worker_load: usize,
    keyed_load: usize,
}

/// Observe the real selection input, then keep the parent's actual manual
/// affinity pin. One worker per role removes random selection as a witness.
#[derive(Debug)]
struct ChoiceRoutingKey {
    seen: Arc<Mutex<Vec<ChoicePlacement>>>,
    manual: Arc<ManualPolicy>,
}

impl LoadBalancingPolicy for ChoiceRoutingKey {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let at = self.manual.select_worker(workers, info)?;
        self.seen.lock().unwrap().push(ChoicePlacement {
            role: info.role,
            rid_key: info.rid_key.map(str::to_owned),
            worker_load: workers[at].load(),
            keyed_load: workers[at].routing_key_inflight("conversation"),
        });
        Some(at)
    }

    fn name(&self) -> &'static str {
        "test-moriio-choice-routing-key"
    }

    fn honors_routing_key(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

async fn hosted_moriio_choices_keep_parent_key(fault: Option<&'static str>) -> Case {
    const RID: &str = "conversation_t3";
    const KEY: &str = "conversation";
    let (prefill, prefill_seen) = spawn_moriio_prefill_stub(StubHandoff::Minted).await;
    let (decode, decode_seen) = spawn_choice_decode_stub_with_model(fault, MODEL).await;
    let mut router = create_test_pd_router();
    router.retry_config.max_retries = 0;
    // A parent admission and its child cannot coexist at capacity one.
    // Retaining the parent's guard would hang before the first child sends.
    router.prefill_admission = Some(Arc::new(PrefillAdmission::new(
        1,
        1,
        Duration::from_secs(30),
    )));
    let observed: Arc<Mutex<Vec<ChoicePlacement>>> = Arc::default();
    let manual = Arc::new(ManualPolicy::new());
    let policy: Arc<dyn LoadBalancingPolicy> = Arc::new(ChoiceRoutingKey {
        seen: Arc::clone(&observed),
        manual: Arc::clone(&manual),
    });
    let registry = Arc::new(PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        RoutingKeyOverrideConfig {
            enabled: true,
            ..RoutingKeyOverrideConfig::default()
        },
    ));
    registry.set_prefill_policy(Arc::clone(&policy));
    registry.set_decode_policy(policy);
    router.policy_registry = registry;
    for (url, role, host) in [
        (&prefill, WorkerType::Prefill, "10.0.0.1"),
        (&decode, WorkerType::Decode, "10.0.0.2"),
    ] {
        let mut worker = BasicWorkerBuilder::new(url.clone())
            .model(ModelCard::new(MODEL))
            .worker_type(role)
            .runtime_type(RuntimeType::Vllm)
            .kv_connector("MoRIIOConnector")
            .label("moriio_mode", "read")
            .label("moriio_host", host)
            .zmq_request_timeout(Duration::from_secs(3))
            .status(WorkerStatus::Ready);
        if role == WorkerType::Decode {
            worker = worker
                .label("moriio_notify_port", "62005")
                .label("tp_size", "4");
        }
        router
            .worker_registry
            .register_or_replace(Arc::new(worker.build()));
    }
    for (role, url) in [
        (WorkerRole::Prefill, &prefill),
        (WorkerRole::Decode, &decode),
    ] {
        manual.pin_front(&format!("{}{}", role.routing_id_prefix(), KEY), url);
    }
    let meta = TenantRequestMeta::new(TenantKey::from("test-tenant"));
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": MODEL,
        "rid": RID,
        "request_id": RID,
        "messages": [{"role": "user", "content": "hi"}],
        "n": 3,
        "max_tokens": 16,
        "temperature": 0.8,
    }))?;
    let response = router
        .route_chat(None, &meta, Arc::new(request), MODEL)
        .await;
    let expected = if fault.is_some() {
        StatusCode::BAD_GATEWAY
    } else {
        StatusCode::OK
    };
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
    assert_eq!(status, expected, "{}", String::from_utf8_lossy(&bytes));
    let count = if fault.is_some() { 2 } else { 3 };
    if fault.is_none() {
        let reply: Value = serde_json::from_slice(&bytes)?;
        assert_eq!(reply["choices"].as_array().ok_or("choices")?.len(), 3);
    }
    let prefills = prefill_seen.lock().map_err(|e| e.to_string())?.clone();
    let decodes = decode_seen.lock().map_err(|e| e.to_string())?.clone();
    assert_eq!(prefills.len(), count);
    assert_eq!(
        decodes.len(),
        count,
        "no child after a failed second choice"
    );
    let mut child_ids = HashSet::new();
    let mut transfers = HashSet::new();
    for ((_, prefill), (_, decode)) in prefills.iter().zip(&decodes) {
        assert_eq!(prefill["n"], 1);
        assert_eq!(decode["n"], 1);
        assert_eq!(prefill["request_id"], decode["request_id"]);
        assert_eq!(decode["test_header_id"], decode["request_id"]);
        let id = decode["request_id"].as_str().ok_or("child request id")?;
        assert_ne!(id, RID);
        assert!(
            child_ids.insert(id.to_owned()),
            "siblings share a transport id"
        );
        let transfer = decode
            .pointer("/kv_transfer_params/transfer_id")
            .and_then(Value::as_str)
            .ok_or("child transfer id")?;
        assert_eq!(prefill["kv_transfer_params"]["transfer_id"], transfer);
        assert!(transfers.insert(transfer), "siblings share a KV handoff");
    }
    let keys = observed.lock().map_err(|e| e.to_string())?;
    for role in [WorkerRole::Prefill, WorkerRole::Decode] {
        assert!(
            keys.iter().filter(|row| row.role == role).count() > count,
            "the parent and every child must ask for placement of {role:?}: {keys:?}"
        );
    }
    assert!(
        keys.iter().all(|row| row.rid_key.as_deref() == Some(KEY)),
        "every child must retain the parent's routing key: {keys:?}"
    );
    assert!(
        keys.iter()
            .all(|row| row.worker_load == 0 && row.keyed_load == 0),
        "the unsent parent or a completed child retained placement load: {keys:?}"
    );
    assert_eq!(
        manual.entry_count(),
        2,
        "only the parent's two role pins may persist"
    );
    for (role, workers) in [
        (
            WorkerRole::Prefill,
            router.worker_registry.get_prefill_workers(),
        ),
        (
            WorkerRole::Decode,
            router.worker_registry.get_decode_workers(),
        ),
    ] {
        assert!(
            matches!(
                manual.peek_pin(
                    &workers,
                    &format!("{}{}", role.routing_id_prefix(), KEY),
                    |_| true
                ),
                PinState::Pinned(0)
            ),
            "the parent's affinity pin changed"
        );
        for id in &child_ids {
            assert!(
                matches!(
                    manual.peek_pin(
                        &workers,
                        &format!("{}{}", role.routing_id_prefix(), id),
                        |_| true
                    ),
                    PinState::Vacant
                ),
                "a private child UUID became a persistent routing pin"
            );
        }
        for worker in workers.iter() {
            assert_eq!(worker.load(), 0, "a choice left its worker load guard live");
            assert_eq!(
                worker.routing_key_load(),
                0,
                "a choice left a key load live"
            );
            assert_eq!(
                worker.routing_key_inflight(KEY),
                0,
                "a choice left the parent's key lease live"
            );
            for id in &child_ids {
                assert_eq!(
                    worker.routing_key_inflight(id),
                    0,
                    "a choice left its private key lease live"
                );
            }
        }
    }
    Ok(())
}

#[tokio::test]
async fn hosted_moriio_choices_preserve_parent_routing_key_and_release_parent_placement() -> Case {
    tokio::time::timeout(
        Duration::from_secs(10),
        hosted_moriio_choices_keep_parent_key(None),
    )
    .await?
}

#[tokio::test]
async fn hosted_moriio_second_child_failure_preserves_parent_routing_key_and_releases_leases(
) -> Case {
    tokio::time::timeout(
        Duration::from_secs(10),
        hosted_moriio_choices_keep_parent_key(Some("http")),
    )
    .await?
}
