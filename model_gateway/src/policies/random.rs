//! Random load balancing policy

use std::sync::Arc;

use rand::RngExt;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

/// Random selection policy
///
/// Selects workers randomly with uniform distribution among healthy workers.
#[derive(Debug, Default)]
pub struct RandomPolicy;

impl RandomPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl LoadBalancingPolicy for RandomPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers, info);

        if healthy_indices.is_empty() {
            return None;
        }

        let mut rng = rand::rng();
        let random_idx = rng.random_range(0..healthy_indices.len());

        Some(healthy_indices[random_idx])
    }

    fn name(&self) -> &'static str {
        "random"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use openai_protocol::worker::WorkerStatus;

    use super::*;
    use crate::policies::test_support::make_workers;

    #[test]
    fn test_random_selection() {
        let policy = RandomPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        let mut counts = HashMap::new();
        for _ in 0..100 {
            if let Some(idx) = policy.select_worker(&workers, &SelectWorkerInfo::default()) {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }

        // All workers should be selected at least once
        assert_eq!(counts.len(), 3);
        assert!(counts.values().all(|&count| count > 0));
    }

    #[test]
    fn test_random_with_unhealthy_workers() {
        let policy = RandomPolicy::new();
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        // Mark first worker as unhealthy
        workers[0].set_status(WorkerStatus::NotReady);

        // Should always select the healthy worker (index 1)
        for _ in 0..10 {
            assert_eq!(
                policy.select_worker(&workers, &SelectWorkerInfo::default()),
                Some(1)
            );
        }
    }

    #[test]
    fn test_random_no_healthy_workers() {
        let policy = RandomPolicy::new();
        let workers = make_workers(&["http://w1:8000"]);

        workers[0].set_status(WorkerStatus::NotReady);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            None
        );
    }
}
