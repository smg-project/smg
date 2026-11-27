//! Random load balancing policy

use std::sync::Arc;

use rand::RngExt;
use smg_worker::identity::basic::Worker;

use super::{draw_eligible, get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};

/// Random selection policy
///
/// Selects workers randomly with uniform distribution among healthy workers:
/// a pick draws by rejection over the pool (`draw_eligible`), reading the
/// eligibility of the drawn workers alone and allocating nothing; a pool
/// mostly vetoed is walked once instead.
#[derive(Debug, Default)]
pub struct RandomPolicy;

impl RandomPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl LoadBalancingPolicy for RandomPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        draw_eligible(workers, info, None).or_else(|| {
            // The draw gave up: few eligible workers, or none. Walk the pool once.
            let healthy_indices = get_healthy_worker_indices(workers, info);
            (!healthy_indices.is_empty())
                .then(|| healthy_indices[rand::rng().random_range(0..healthy_indices.len())])
        })
    }

    fn name(&self) -> &'static str {
        "random"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
#[path = "../../tests/unit/policies/random/tests.rs"]
mod tests;
