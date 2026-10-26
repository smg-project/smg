//! What the crate's tests build their workers with: a health-check
//! configuration that leaves the worker as the test built it, and a ready
//! worker built with it.

use std::sync::Arc;

use openai_protocol::{
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerStatus, WorkerType},
};

use crate::worker::{BasicWorkerBuilder, ConnectionMode, Worker};

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
