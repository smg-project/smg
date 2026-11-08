//! Service discovery integration tests, and the two fixtures both providers'
//! tests start from: a context with a drain-settle time and a mock engine.

use std::sync::Arc;

use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use smg::{app_context::AppContext, config::RouterConfig};

use crate::common;

mod file_discovery_test;
mod k8s_discovery_test;

/// A context whose workers start within two seconds and drain for `drain_settle_secs`.
async fn test_context_with_drain(drain_settle_secs: u64) -> Arc<AppContext> {
    let mut config = RouterConfig::builder()
        .worker_startup_timeout_secs(2)
        .build_unchecked();
    config.health_check.drain_settle_secs = drain_settle_secs;
    common::create_test_context(config).await
}

/// A healthy mock engine on an ephemeral port, with the port it took.
#[expect(clippy::unwrap_used, reason = "test helper")]
async fn start_mock_engine() -> (MockWorker, u16) {
    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let url = worker.start().await.unwrap();
    let port = url.rsplit(':').next().unwrap().parse().unwrap();
    (worker, port)
}
