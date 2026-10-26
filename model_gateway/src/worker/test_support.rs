//! What the crate's tests build their workers with: a health-check
//! configuration that leaves the worker as the test built it, a ready
//! worker built with it, a labelled gRPC worker and the registry holding
//! a test's workers, and the admission attempt that selects one worker.

use std::sync::Arc;

use openai_protocol::{
    model_card::ModelCard,
    worker::{HealthCheckConfig, RuntimeType, WorkerStatus, WorkerType},
};

use crate::worker::{
    prefill_admission::{PrefillAdmissionAttempt, PrefillSelectionContext},
    BasicWorkerBuilder, ConnectionMode, Worker, WorkerRegistry,
};

/// The health checker disabled, everything else default: a test's worker
/// keeps the state it was built with, with no probe behind it.
pub(crate) fn no_health_check() -> HealthCheckConfig {
    HealthCheckConfig {
        disable_health_check: true,
        ..Default::default()
    }
}

/// A ready HTTP worker at `url` serving `model`, with no health checker
/// behind it.
pub(crate) fn ready_worker(url: &str, model: &str) -> Arc<dyn Worker> {
    let worker = BasicWorkerBuilder::new(url)
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Http)
        .model(ModelCard::new(model))
        .health_config(no_health_check())
        .build();
    worker.set_status(WorkerStatus::Ready);
    Arc::new(worker)
}

/// The attempt a test makes when it wants exactly `worker`: it is selected,
/// with a unit payload, whenever it has capacity.
pub(crate) fn select_worker(
    worker: &Arc<dyn Worker>,
) -> impl for<'a> FnMut(&PrefillSelectionContext<'a>) -> PrefillAdmissionAttempt<()> + Send {
    let worker = Arc::clone(worker);
    move |capacity| capacity.select(Arc::clone(&worker), ())
}

/// A regular gRPC worker of `runtime` at `url` serving `model`, with
/// `labels` and no health checker behind it: what the multimodal router
/// tests put in their pools.
pub(crate) fn grpc_worker(
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
pub(crate) fn registry_with(workers: Vec<Arc<dyn Worker>>) -> WorkerRegistry {
    let registry = WorkerRegistry::new();
    for worker in workers {
        registry.register(worker).expect("register");
    }
    registry
}
