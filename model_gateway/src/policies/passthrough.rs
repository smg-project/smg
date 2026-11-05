//! Passthrough routing policy for single-backend serving.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use tracing::warn;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

/// Passthrough selection policy for single-backend deployments.
///
/// Forwards every request to the single healthy worker, selecting the first
/// healthy index. It is intended for one-worker gateways where load balancing
/// carries no benefit.
///
/// Being neither load-aware nor cache-aware, it deliberately skips the
/// machinery those policies require: the `WorkerMonitor` never polls it for
/// load (it is excluded from the load-aware set in `policies/registry.rs`), and
/// no KV-event monitor is started for it (gated on cache-aware in
/// `app_context.rs`). For single-backend gateways this also eliminates the
/// `SubscribeKvEvents` subscription overhead.
#[derive(Debug, Default)]
pub struct PassthroughPolicy {
    /// Latches once we have warned about a multi-worker pool, so the warning is
    /// emitted at most once instead of on every request (hot path).
    warned_multi_worker: AtomicBool,
}

impl PassthroughPolicy {
    pub fn new() -> Self {
        Self::default()
    }
}

impl LoadBalancingPolicy for PassthroughPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        if workers.len() > 1 && !self.warned_multi_worker.swap(true, Ordering::Relaxed) {
            warn!(
                worker_count = workers.len(),
                "passthrough policy is intended for single-backend serving but multiple workers \
                 are registered; only the first healthy worker will receive traffic"
            );
        }

        get_healthy_worker_indices(workers, info).first().copied()
    }

    fn name(&self) -> &'static str {
        "passthrough"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::WorkerStatus;

    use super::*;
    use crate::worker::{test_support::no_health_check, BasicWorkerBuilder, WorkerType};

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        )
    }

    #[test]
    fn test_selects_only_worker() {
        let policy = PassthroughPolicy::new();
        let workers = vec![worker("http://w1:8000")];
        for _ in 0..10 {
            assert_eq!(
                policy.select_worker(&workers, &SelectWorkerInfo::default()),
                Some(0)
            );
        }
    }

    #[test]
    fn test_selects_first_healthy_with_two_workers() {
        let policy = PassthroughPolicy::new();
        let workers = vec![worker("http://w1:8000"), worker("http://w2:8000")];
        // Deterministic: always the first healthy index.
        for _ in 0..10 {
            assert_eq!(
                policy.select_worker(&workers, &SelectWorkerInfo::default()),
                Some(0)
            );
        }
    }

    #[test]
    fn test_skips_unhealthy_first_worker() {
        let policy = PassthroughPolicy::new();
        let workers = vec![worker("http://w1:8000"), worker("http://w2:8000")];
        workers[0].set_status(WorkerStatus::NotReady);
        for _ in 0..10 {
            assert_eq!(
                policy.select_worker(&workers, &SelectWorkerInfo::default()),
                Some(1)
            );
        }
    }

    #[test]
    fn test_none_when_empty() {
        let policy = PassthroughPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            None
        );
    }

    #[test]
    fn test_none_when_all_unhealthy() {
        let policy = PassthroughPolicy::new();
        let workers = vec![worker("http://w1:8000"), worker("http://w2:8000")];
        workers[0].set_status(WorkerStatus::NotReady);
        workers[1].set_status(WorkerStatus::NotReady);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            None
        );
    }

    #[test]
    fn test_name() {
        assert_eq!(PassthroughPolicy::new().name(), "passthrough");
    }
}
