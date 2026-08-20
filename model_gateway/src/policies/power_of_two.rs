//! Power-of-two choices load balancing policy

use std::{collections::HashMap, sync::Arc};

use openai_protocol::worker::WorkerLoadResponse;
use rand::RngExt;

use super::{get_healthy_worker_indices, LeastLoadPolicy, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

/// Power-of-two choices policy: sample two distinct healthy workers uniformly
/// and route to the one with the lower expected wait, scored exactly like
/// [`LeastLoadPolicy`] — `(queued_tokens + in-flight credit) / throughput`
/// plus the convex KV-pressure barrier. Least-load restricted to a random
/// pair: near-least-load quality at O(1) load reads per pick, without the
/// full-fleet scan.
#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    /// Expected-wait scorer shared with least-load: load cache, since-poll
    /// in-flight credit, and the scoring tunables.
    scorer: LeastLoadPolicy,
}

impl PowerOfTwoPolicy {
    pub fn new() -> Self {
        Self {
            scorer: LeastLoadPolicy::new(),
        }
    }
}

impl LoadBalancingPolicy for PowerOfTwoPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);

        if healthy.is_empty() {
            return None;
        }
        if healthy.len() == 1 {
            return Some(healthy[0]);
        }

        // Select two distinct workers - use offset to guarantee different
        // selection in O(1).
        let mut rng = rand::rng();
        let idx1 = rng.random_range(0..healthy.len());
        let idx2 = (idx1 + 1 + rng.random_range(0..healthy.len() - 1)) % healthy.len();
        let pair = [healthy[idx1], healthy[idx2]];

        self.scorer
            .select_min_expected_wait(workers, &pair, info, self.name())
    }

    fn name(&self) -> &'static str {
        "power_of_two"
    }

    fn update_loads(&self, loads: &HashMap<String, WorkerLoadResponse>) {
        self.scorer.update_loads(loads);
    }

    fn needs_backend_loads(&self) -> bool {
        true
    }

    fn remove_worker(&self, url: &str) {
        self.scorer.remove_worker(url);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for PowerOfTwoPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::{HealthCheckConfig, SchedulerLoadSnapshot};

    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    /// One DP rank with the given queued tokens, KV utilization, and throughput.
    fn make_load(
        num_waiting_uncached_tokens: i32,
        token_usage: f64,
        gen_throughput: f64,
    ) -> WorkerLoadResponse {
        WorkerLoadResponse {
            timestamp: String::new(),
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                dp_rank: 0,
                num_waiting_uncached_tokens,
                token_usage,
                gen_throughput,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// One DP rank reporting a queue depth with no token count for it — the
    /// shape produced by any backend scored from Prometheus gauges.
    fn make_load_reqs_only(
        num_waiting_reqs: i32,
        token_usage: f64,
        gen_throughput: f64,
    ) -> WorkerLoadResponse {
        let mut load = make_load(0, token_usage, gen_throughput);
        load.loads[0].num_waiting_reqs = num_waiting_reqs;
        load
    }

    fn mk(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        )
    }

    #[test]
    fn single_worker_always_selected() {
        let policy = PowerOfTwoPolicy::new();
        let workers = vec![mk("http://w1:8000")];
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn requires_backend_loads() {
        // The monitor only polls for policies that ask; a regression here
        // would leave the policy permanently on the dark-pair fallback.
        assert!(PowerOfTwoPolicy::new().needs_backend_loads());
    }

    #[test]
    fn remove_worker_forgets_stale_snapshot() {
        // a's heavy snapshot steers picks to b; after remove_worker(a), a is
        // scored on the missing-snapshot drain-time path (idle -> 0) instead
        // of the stale queue.
        let policy = PowerOfTwoPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(8000, 0.2, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(1000, 0.2, 100.0));
        policy.update_loads(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );

        policy.remove_worker("http://a:8000");
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn update_loads_resets_inflight_credit() {
        // pick1 -> a; its credit tips pick2 -> b; a fresh poll resets the
        // credits so pick3 returns to a.
        let policy = PowerOfTwoPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(1000, 0.2, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(2000, 0.2, 100.0));
        policy.update_loads(&loads);

        let info = SelectWorkerInfo::default();
        // a: 10.04s vs b: 20.11s.
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        // a: (1000 + 1024)/100 = 20.28s vs b: 20.11s.
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
        policy.update_loads(&loads);
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
    }

    #[test]
    fn queue_length_steers_the_pair() {
        // Equal KV/throughput; the worker with the shorter token queue wins.
        let policy = PowerOfTwoPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(8000, 0.2, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(1000, 0.2, 100.0));
        policy.update_loads(&loads);
        // a: 8000/100 = 80s ; b: 1000/100 = 10s -> pick b.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn waiting_reqs_estimate_when_backend_reports_no_queued_tokens() {
        // A backend exposing queue depth but not queued tokens must not be
        // read as idle: its queue is estimated at waiting_reqs · mean prefill.
        let policy = PowerOfTwoPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            make_load_reqs_only(8, 0.2, 100.0),
        );
        loads.insert(
            "http://b:8000".to_string(),
            make_load_reqs_only(0, 0.2, 100.0),
        );
        policy.update_loads(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn kv_pressure_steers_when_queues_empty() {
        // No queued work anywhere: the KV barrier decides.
        let policy = PowerOfTwoPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.8, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.1, 100.0));
        policy.update_loads(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn missing_snapshot_scored_by_drain_time_not_raw_count() {
        // a reports (idle queue, hot KV); b has no snapshot but 5 live
        // requests. b is scored as drain time (5 · p̄ / throughput ≈ 2.56s),
        // comparable to a's KV barrier (0.15 · 0.9/0.1 = 1.35s) -> pick a.
        let policy = PowerOfTwoPolicy::new();
        let a = mk("http://a:8000");
        let b = mk("http://b:8000");
        for _ in 0..5 {
            b.increment_load();
        }
        let workers = vec![a, b];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.9, 0.0));
        policy.update_loads(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn dark_pair_joins_shortest_queue() {
        // Neither sampled worker reports loads: fall back to live in-flight.
        let policy = PowerOfTwoPolicy::new();
        let a = mk("http://a:8000");
        let b = mk("http://b:8000");
        for _ in 0..5 {
            a.increment_load();
        }
        for _ in 0..3 {
            b.increment_load();
        }
        let workers = vec![a, b];
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn cold_start_distribution_prefers_lower_in_flight() {
        // Three dark workers with in-flight 10/5/0: every pair containing w3
        // picks w3, the {w1,w2} pair picks w2, and w1 never wins.
        let policy = PowerOfTwoPolicy::new();
        let counts = {
            let w1 = mk("http://w1:8000");
            let w2 = mk("http://w2:8000");
            let w3 = mk("http://w3:8000");
            for _ in 0..10 {
                w1.increment_load();
            }
            for _ in 0..5 {
                w2.increment_load();
            }
            let workers = vec![w1, w2, w3];
            let mut counts = [0usize; 3];
            for _ in 0..300 {
                let idx = policy
                    .select_worker(&workers, &SelectWorkerInfo::default())
                    .unwrap();
                counts[idx] += 1;
            }
            counts
        };
        assert_eq!(counts[0], 0, "highest-loaded worker must never win a pair");
        assert!(
            counts[2] > counts[1],
            "idle worker wins most pairs: {counts:?}"
        );
    }

    #[test]
    fn inflight_credit_water_fills_identical_workers() {
        // Two identically-loaded workers: the first pick's in-flight credit
        // tips the second pick to the other worker instead of herding.
        let policy = PowerOfTwoPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        for url in ["http://a:8000", "http://b:8000"] {
            loads.insert(url.to_string(), make_load(1000, 0.2, 100.0));
        }
        policy.update_loads(&loads);

        let mut seen = [false; 2];
        for _ in 0..2 {
            let idx = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .unwrap();
            seen[idx] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "consecutive picks must spread across equal workers, saw {seen:?}"
        );
    }
}
