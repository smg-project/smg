//! Round-robin load balancing policy

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

use dashmap::DashMap;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

/// Distinct candidate sets one instance keeps a rotation for; past this a
/// new set evicts the least recently used one. Sets come and go with worker
/// health, so this is a bound on memory, not a limit on fleets.
const MAX_TRACKED_SETS: usize = 4096;

/// One candidate set's rotation.
#[derive(Debug)]
struct Rotation {
    next: AtomicUsize,
    /// The policy's tick when the set was last selected from, for eviction.
    last_used: AtomicU64,
}

/// Round-robin selection policy
///
/// Selects workers in sequential order, cycling through all healthy workers.
/// The rotation is kept per candidate set: one instance serves several sets
/// (the decode partners of each PD cohort, say), and each set walks its own
/// workers in turn. A single counter would advance on every call whatever
/// the set, so its position modulo one set's size would follow how the sets
/// happen to interleave, pinning each set to a subset of its workers.
///
/// A set is identified by its healthy workers in the order the caller lists
/// them; callers hand over a stable order (a pool snapshot, a pair index's
/// partners), which the rotation relies on either way. A set seen for the
/// first time starts from a shared, advancing position rather than its
/// first worker, so a fleet whose shape keeps changing does not lean on
/// index 0.
#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counters: DashMap<u64, Rotation>,
    next_start: AtomicUsize,
    tick: AtomicU64,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            counters: DashMap::new(),
            next_start: AtomicUsize::new(0),
            tick: AtomicU64::new(0),
        }
    }

    /// The identity of a candidate set: its healthy workers' URLs, in
    /// order. A worker that re-registers under the same URL resumes its
    /// sets' rotations, which is harmless; the hashing is a short string
    /// per healthy candidate, on every selection.
    fn set_key(workers: &[Arc<dyn Worker>], healthy: &[usize]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for &i in healthy {
            workers[i].url().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// This set's next position, creating its rotation on first sight.
    ///
    /// Age is measured in misses, since only misses evict: the steady state
    /// is a read guard, a relaxed load of the miss count and two per-set
    /// atomics, with no shared write. A miss is a new shape: at the cap it
    /// evicts the least recently used set first, a scan of the tracked sets
    /// that only a churning fleet at the cap pays. `last_used` only moves
    /// forward, so selections completing out of order cannot age a set.
    fn advance(&self, key: u64) -> usize {
        if let Some(rotation) = self.counters.get(&key) {
            rotation
                .last_used
                .fetch_max(self.tick.load(Ordering::Relaxed), Ordering::Relaxed);
            return rotation.next.fetch_add(1, Ordering::Relaxed);
        }
        let now = self.tick.fetch_add(1, Ordering::Relaxed) + 1;
        if self.counters.len() >= MAX_TRACKED_SETS {
            // The iterator's shard guards are dropped with the statement,
            // before the removal takes its shard's write lock.
            let stale = self
                .counters
                .iter()
                .min_by_key(|entry| entry.last_used.load(Ordering::Relaxed))
                .map(|entry| *entry.key());
            if let Some(stale) = stale {
                self.counters.remove(&stale);
            }
        }
        // The shared position advances only for the request that inserts,
        // so concurrent first sightings of one set do not skip starts; a
        // set another request inserted meanwhile is refreshed as used now.
        let rotation = self.counters.entry(key).or_insert_with(|| Rotation {
            next: AtomicUsize::new(self.next_start.fetch_add(1, Ordering::Relaxed)),
            last_used: AtomicU64::new(now),
        });
        rotation.last_used.fetch_max(now, Ordering::Relaxed);
        rotation.next.fetch_add(1, Ordering::Relaxed)
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

        let count = self.advance(Self::set_key(workers, &healthy_indices));
        let selected_idx = count % healthy_indices.len();

        Some(healthy_indices[selected_idx])
    }

    fn name(&self) -> &'static str {
        "round_robin"
    }

    fn reset(&self) {
        self.counters.clear();
        self.next_start.store(0, Ordering::Relaxed);
        self.tick.store(0, Ordering::Relaxed);
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
    fn test_one_instance_keeps_each_candidate_set_fair() {
        // One instance serving two interleaved sets: each set gets its own
        // rotation, so the interleaving cannot pin a set to some of its
        // workers (a single counter would give each set only two of four).
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
    fn test_a_set_that_shrinks_and_grows_keeps_rotating() {
        // Health changes the set's identity. The shrunken set is new, so it
        // starts from the shared position (1 by then), not from worker 0
        // again; the full set resumes where it left off when it returns.
        let workers = make_regular_workers("w", 3);
        let info = SelectWorkerInfo::default();
        let policy = RoundRobinPolicy::new();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        workers[1].set_status(WorkerStatus::NotReady);
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        workers[1].set_status(WorkerStatus::Ready);
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
        assert_eq!(policy.select_worker(&workers, &info), Some(2));
    }

    #[test]
    fn test_past_the_cap_a_new_set_evicts_the_least_recently_used() {
        let info = SelectWorkerInfo::default();
        let policy = RoundRobinPolicy::new();
        let hot = make_regular_workers("hot", 2);
        assert_eq!(policy.select_worker(&hot, &info), Some(0));
        // Distinct one-worker sets fill the map to just under the cap.
        let filler: Vec<Vec<Arc<dyn Worker>>> = (0..MAX_TRACKED_SETS + 3)
            .map(|i| make_regular_workers(&format!("f{i}-"), 1))
            .collect();
        for set in &filler[..MAX_TRACKED_SETS - 2] {
            policy.select_worker(set, &info);
        }
        // The hot set is used again, so it is recent when the cap is hit.
        assert_eq!(policy.select_worker(&hot, &info), Some(1));
        for set in &filler[MAX_TRACKED_SETS - 2..] {
            policy.select_worker(set, &info);
        }
        // Evictions took the oldest fillers, one per miss: the map stays at
        // the cap and the hot set kept its rotation, which continues.
        assert_eq!(policy.counters.len(), MAX_TRACKED_SETS);
        assert!(policy
            .counters
            .contains_key(&RoundRobinPolicy::set_key(&hot, &[0, 1])));
        assert!(!policy
            .counters
            .contains_key(&RoundRobinPolicy::set_key(&filler[0], &[0])));
        assert_eq!(policy.select_worker(&hot, &info), Some(0));
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
