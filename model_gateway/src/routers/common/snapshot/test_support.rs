//! What the gateway's tests build their workers with. What stands here
//! names the gateway's own builder and registry: a worker of a given role, a
//! labelled gRPC worker, the registry holding a test's workers, and the KV
//! cache event that clears a worker's cache.

use std::sync::Arc;

use openai_protocol::{
    model_card::ModelCard,
    worker::{RuntimeType, WorkerType},
};
use smg_grpc_client::common_proto::{kv_cache_event, KvCacheCleared, KvCacheEvent};
use smg_worker::{
    identity::basic::{ConnectionMode, Worker},
    test_support::no_health_check,
};

use crate::routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BasicWorkerBuilder};

/// A worker of `worker_type` at `url` with no health checker behind it: what
/// a test fills a pool with when no health checker may run.
pub fn worker_of(url: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .health_config(no_health_check())
            .build(),
    )
}

/// A regular gRPC worker of `runtime` at `url` serving `model`, with
/// `labels` and no health checker behind it: what the multimodal router
/// tests put in their pools.
pub fn grpc_worker(
    url: &str,
    model: &str,
    runtime: RuntimeType,
    labels: &[(&str, &str)],
) -> Arc<dyn Worker> {
    let mut builder = BasicWorkerBuilder::new(url)
        .model(ModelCard::new(model))
        .worker_type(WorkerType::Regular)
        .runtime_type(runtime)
        .connection_mode(ConnectionMode::Grpc)
        .health_config(no_health_check());
    for (key, value) in labels {
        builder = builder.label(*key, *value);
    }
    Arc::new(builder.build())
}

/// A registry holding `workers`, each registered as the test built it.
#[expect(
    clippy::expect_used,
    reason = "a test fixture: a worker the registry refuses is a test failure"
)]
pub fn registry_with(workers: Vec<Arc<dyn Worker>>) -> WorkerRegistry {
    let registry = WorkerRegistry::new();
    for worker in workers {
        registry.register(worker).expect("register");
    }
    registry
}

/// The KV cache event that clears a worker's whole cache: the state the KV
/// event monitor's and the index backends' tests start over from.
pub fn kv_cache_cleared() -> KvCacheEvent {
    KvCacheEvent {
        event_id: 0,
        data: Some(kv_cache_event::Data::Cleared(KvCacheCleared::default())),
    }
}
