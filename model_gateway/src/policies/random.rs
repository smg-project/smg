//! Random load balancing policy

use std::sync::Arc;

use rand::RngExt;

use super::{draw_eligible, get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

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

    /// Eight workers, two of them not ready: each eligible worker is picked
    /// one time in six and a vetoed worker never. The draw is uniform over
    /// the eligible workers, as the listed pick was.
    #[test]
    fn a_pick_is_uniform_over_the_eligible_workers_and_never_a_vetoed_one() {
        let policy = RandomPolicy::new();
        let urls: Vec<String> = (0..8).map(|i| format!("http://w{i}:8000")).collect();
        let workers = make_workers(&urls.iter().map(String::as_str).collect::<Vec<_>>());
        for vetoed in [2, 5] {
            workers[vetoed].set_status(WorkerStatus::NotReady);
        }
        let mut picks = [0usize; 8];
        for _ in 0..6_000 {
            let idx = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .expect("six workers are eligible");
            picks[idx] += 1;
        }
        assert_eq!(
            picks[2] + picks[5],
            0,
            "a not-ready worker was picked: {picks:?}"
        );
        for (idx, &count) in picks.iter().enumerate() {
            if idx != 2 && idx != 5 {
                assert!(
                    (800..=1_200).contains(&count),
                    "worker {idx} took {count} of 6,000 picks, one in six expected: {picks:?}"
                );
            }
        }
    }
}
