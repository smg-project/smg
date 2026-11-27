//! Passthrough routing policy for single-backend serving.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use smg_worker::identity::basic::Worker;
use tracing::warn;

use super::{LoadBalancingPolicy, SelectWorkerInfo};

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

        workers
            .iter()
            .position(|worker| info.admits(worker.routing_state()))
    }

    fn name(&self) -> &'static str {
        "passthrough"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
#[path = "../../tests/unit/policies/passthrough/tests.rs"]
mod tests;
