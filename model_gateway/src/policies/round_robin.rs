//! Round-robin load balancing policy

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

/// Round-robin selection policy
///
/// Selects workers in sequential order, cycling through all healthy workers.
#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

impl LoadBalancingPolicy for RoundRobinPolicy {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Get and increment counter atomically
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let selected_idx = count % healthy_indices.len();

        Some(healthy_indices[selected_idx])
    }

    fn name(&self) -> &'static str {
        "round_robin"
    }

    fn filters_unavailable_workers(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_round_robin_selection() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // Should select workers in order: 0, 1, 2, 0, 1, 2, ...
        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
    }

    #[test]
    fn test_round_robin_with_unhealthy_workers() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // Mark middle worker as unhealthy
        workers[1].set_status(WorkerStatus::NotReady);

        // Should skip unhealthy worker: 0, 2, 0, 2, ...
        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
    }

    #[test]
    fn test_round_robin_reset() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // Advance the counter
        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(1));

        // Reset should start from beginning
        policy.reset();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
    }

    fn assert_even_two_pool_coverage(prefill: [usize; 4], decode: [usize; 4]) {
        assert_eq!(
            prefill,
            [10, 10, 10, 10],
            "even two-pool round-robin coverage: prefill"
        );
        assert_eq!(
            decode,
            [10, 10, 10, 10],
            "even two-pool round-robin coverage: decode"
        );
    }

    fn make_regular_workers(prefix: &str, n: usize) -> Vec<Arc<dyn Worker>> {
        (0..n)
            .map(|i| {
                Arc::new(
                    BasicWorkerBuilder::new(format!("http://{prefix}{i}:8000"))
                        .worker_type(WorkerType::Regular)
                        .health_config(no_health_check())
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    #[test]
    #[should_panic(expected = "even two-pool round-robin coverage")]
    fn test_shared_counter_fails_even_two_pool_coverage() {
        // Same even-coverage bar as the independent test; a shared counter must fail it.
        let prefill_workers = make_regular_workers("p", 4);
        let decode_workers = make_regular_workers("d", 4);
        let info = SelectWorkerInfo::default();

        let shared = RoundRobinPolicy::new();
        let mut shared_prefill = [0usize; 4];
        let mut shared_decode = [0usize; 4];
        for _ in 0..40 {
            let p = shared.select_worker(&prefill_workers, &info).unwrap();
            let d = shared.select_worker(&decode_workers, &info).unwrap();
            shared_prefill[p] += 1;
            shared_decode[d] += 1;
        }
        assert_even_two_pool_coverage(shared_prefill, shared_decode);
    }

    #[test]
    fn test_independent_counters_pass_even_two_pool_coverage() {
        let prefill_workers = make_regular_workers("p", 4);
        let decode_workers = make_regular_workers("d", 4);
        let info = SelectWorkerInfo::default();

        let prefill_policy = RoundRobinPolicy::new();
        let decode_policy = RoundRobinPolicy::new();
        let mut indep_prefill = [0usize; 4];
        let mut indep_decode = [0usize; 4];
        for _ in 0..40 {
            let p = prefill_policy
                .select_worker(&prefill_workers, &info)
                .unwrap();
            let d = decode_policy.select_worker(&decode_workers, &info).unwrap();
            indep_prefill[p] += 1;
            indep_decode[d] += 1;
        }
        assert_even_two_pool_coverage(indep_prefill, indep_decode);
    }
}
