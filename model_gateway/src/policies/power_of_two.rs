//! Power-of-two choices load balancing policy

use std::{cmp::Ordering, sync::Arc};

use rand::RngExt;

use super::{
    draw_eligible, get_healthy_worker_indices, LeastLoadPolicy, LoadBalancingPolicy,
    SelectWorkerInfo,
};
use crate::{
    config::PowerOfTwoLoadMetric,
    worker::{load_state::LoadReceiver, Worker},
};

/// Power-of-two choices policy: sample two distinct healthy workers uniformly
/// and route to the one with the lower expected wait, scored exactly like
/// [`LeastLoadPolicy`] — `(queued_tokens + in-flight credit) / throughput`
/// plus the convex KV-pressure barrier. Least-load restricted to a random
/// pair: near-least-load quality at a cost that does not grow with the pool.
///
/// A pick draws its pair by rejection over the pool (`draw_eligible`): the
/// eligibility of the drawn workers alone is read, the pair's two loads are
/// scored, and nothing is allocated. A pool too thin for the draw (fewer
/// than two eligible workers, or most of them vetoed) is walked once instead,
/// which lists the eligible workers. The `LeastRequests` metric reads every
/// eligible worker's in-flight count by definition, so it walks every pick.
#[derive(Debug)]
pub struct PowerOfTwoPolicy {
    /// Expected-wait scorer shared with least-load: the load feed, the
    /// since-report dispatch credit and the scoring tunables.
    scorer: LeastLoadPolicy,
    load_metric: PowerOfTwoLoadMetric,
}

impl PowerOfTwoPolicy {
    pub fn new() -> Self {
        Self::with_load_metric(PowerOfTwoLoadMetric::default())
    }

    pub fn with_load_metric(load_metric: PowerOfTwoLoadMetric) -> Self {
        Self {
            scorer: LeastLoadPolicy::new(),
            load_metric,
        }
    }
}

/// The pair members with the fewest router in-flight requests: one of them,
/// or both when they tie. Each count is read once.
fn fewest_of_pair<'a>(workers: &[Arc<dyn Worker>], pair: &'a [usize; 2]) -> &'a [usize] {
    match workers[pair[0]].load().cmp(&workers[pair[1]].load()) {
        Ordering::Less => &pair[..1],
        Ordering::Greater => &pair[1..],
        Ordering::Equal => &pair[..],
    }
}

/// The candidates with the fewest router in-flight requests. Each count is read
/// once, so concurrent dispatches cannot empty the result.
fn fewest_in_flight(workers: &[Arc<dyn Worker>], candidates: &[usize]) -> Vec<usize> {
    let loads: Vec<usize> = candidates.iter().map(|&i| workers[i].load()).collect();
    let fewest = loads.iter().copied().min().unwrap_or(0);
    candidates
        .iter()
        .zip(loads)
        .filter(|&(_, load)| load == fewest)
        .map(|(&i, _)| i)
        .collect()
}

impl LoadBalancingPolicy for PowerOfTwoPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        if self.load_metric == PowerOfTwoLoadMetric::LeastRequests {
            let healthy = get_healthy_worker_indices(workers, info);
            return match healthy.as_slice() {
                [] => None,
                [only] => Some(*only),
                _ => {
                    let fewest = fewest_in_flight(workers, &healthy);
                    self.scorer
                        .select_min_expected_wait(workers, &fewest, info, self.name())
                }
            };
        }

        let drawn = draw_eligible(workers, info, None).and_then(|first| {
            draw_eligible(workers, info, Some(first)).map(|second| [first, second])
        });
        let pair = match drawn {
            Some(pair) => pair,
            // The draw gave up: fewer than two eligible workers, or most of
            // the pool vetoed. Walk it once; a lone eligible worker is the
            // pick, and the pair comes from the list with an offset that
            // keeps the two apart.
            None => {
                let healthy = get_healthy_worker_indices(workers, info);
                match healthy.as_slice() {
                    [] => return None,
                    [only] => return Some(*only),
                    _ => {
                        let mut rng = rand::rng();
                        let idx1 = rng.random_range(0..healthy.len());
                        let idx2 =
                            (idx1 + 1 + rng.random_range(0..healthy.len() - 1)) % healthy.len();
                        [healthy[idx1], healthy[idx2]]
                    }
                }
            }
        };

        let candidates: &[usize] = if self.load_metric == PowerOfTwoLoadMetric::Requests {
            fewest_of_pair(workers, &pair)
        } else {
            &pair
        };
        self.scorer
            .select_min_expected_wait(workers, candidates, info, self.name())
    }

    fn name(&self) -> &'static str {
        "power_of_two"
    }

    fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        self.scorer.set_load_receiver(rx);
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
    use std::collections::HashMap;

    use super::*;
    use crate::{
        policies::test_support::{fed_pair, make_worker},
        worker::load_state::{one_rank_load, one_rank_load_reqs_only, TestFeed},
    };

    #[test]
    fn single_worker_always_selected() {
        let policy = PowerOfTwoPolicy::new();
        let workers = vec![make_worker("http://w1:8000")];
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
    fn a_worker_the_snapshot_dropped_is_scored_like_its_peers() {
        // a's heavy report (far more than 50 picks of credit can overtake)
        // steers picks to b; once the monitor drops a's entry and the policy
        // forgets a's tally, a is scored like its reporting peer instead of
        // by the stale queue, so both are picked.
        let (policy, feed, workers) = fed_pair(PowerOfTwoPolicy::new());
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            one_rank_load(100_000, 0.2, 100.0),
        );
        loads.insert("http://b:8000".to_string(), one_rank_load(1000, 0.2, 100.0));
        feed.publish(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );

        loads.remove("http://a:8000");
        feed.publish(&loads);
        policy.remove_worker("http://a:8000");
        let mut seen = [false; 2];
        for _ in 0..50 {
            let idx = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .unwrap();
            seen[idx] = true;
        }
        assert!(
            seen == [true, true],
            "both workers must be picked once the stale snapshot is dropped, saw {seen:?}"
        );
    }

    #[test]
    fn a_new_report_releases_the_in_flight_credit() {
        // pick1 -> a; its credit tips pick2 -> b; a fresh report releases
        // the credit so pick3 returns to a.
        let (policy, feed, workers) = fed_pair(PowerOfTwoPolicy::new());
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(1000, 0.2, 100.0));
        loads.insert("http://b:8000".to_string(), one_rank_load(2000, 0.2, 100.0));
        feed.publish(&loads);

        let info = SelectWorkerInfo::default();
        // a: 10.04s vs b: 20.11s.
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        // a: (1000 + 1024)/100 = 20.28s vs b: 20.11s.
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
        feed.publish(&loads);
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
    }

    #[test]
    fn queue_length_steers_the_pair() {
        // Equal KV/throughput; the worker with the shorter token queue wins.
        let (policy, feed, workers) = fed_pair(PowerOfTwoPolicy::new());
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(8000, 0.2, 100.0));
        loads.insert("http://b:8000".to_string(), one_rank_load(1000, 0.2, 100.0));
        feed.publish(&loads);
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
        let (policy, feed, workers) = fed_pair(PowerOfTwoPolicy::new());
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            one_rank_load_reqs_only(8, 0.2, 100.0),
        );
        loads.insert(
            "http://b:8000".to_string(),
            one_rank_load_reqs_only(0, 0.2, 100.0),
        );
        feed.publish(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn kv_pressure_steers_when_queues_empty() {
        // No queued work anywhere: the KV barrier decides.
        let (policy, feed, workers) = fed_pair(PowerOfTwoPolicy::new());
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(0, 0.8, 100.0));
        loads.insert("http://b:8000".to_string(), one_rank_load(0, 0.1, 100.0));
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
        let a = make_worker("http://a:8000");
        let b = make_worker("http://b:8000");
        for _ in 0..5 {
            b.increment_load();
        }
        let workers = vec![a, b];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(0, 0.9, 0.0));
        feed.publish(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn dark_pair_joins_shortest_queue() {
        // Neither sampled worker reports loads: fall back to live in-flight.
        let policy = PowerOfTwoPolicy::new();
        let a = make_worker("http://a:8000");
        let b = make_worker("http://b:8000");
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
            let w1 = make_worker("http://w1:8000");
            let w2 = make_worker("http://w2:8000");
            let w3 = make_worker("http://w3:8000");
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
        let (policy, feed, workers) = fed_pair(PowerOfTwoPolicy::new());
        let mut loads = HashMap::new();
        for url in ["http://a:8000", "http://b:8000"] {
            loads.insert(url.to_string(), one_rank_load(1000, 0.2, 100.0));
        }
        feed.publish(&loads);

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

    fn holding(in_flight: &[usize]) -> Vec<Arc<dyn Worker>> {
        in_flight
            .iter()
            .enumerate()
            .map(|(i, &requests)| {
                let worker = make_worker(&format!("http://w{i}:8000"));
                (0..requests).for_each(|_| worker.increment_load());
                worker
            })
            .collect()
    }

    #[test]
    fn requests_metric_prefers_fewer_in_flight_over_a_lighter_report() {
        let workers = holding(&[5, 2]);
        let mut loads = HashMap::new();
        loads.insert("http://w0:8000".to_string(), one_rank_load(0, 0.1, 100.0));
        loads.insert(
            "http://w1:8000".to_string(),
            one_rank_load(50_000, 0.1, 100.0),
        );
        let info = SelectWorkerInfo::default();

        let expected_wait = PowerOfTwoPolicy::new();
        TestFeed::attached_to(&expected_wait).publish(&loads);
        assert_eq!(expected_wait.select_worker(&workers, &info), Some(0));

        let requests = PowerOfTwoPolicy::with_load_metric(PowerOfTwoLoadMetric::Requests);
        TestFeed::attached_to(&requests).publish(&loads);
        for _ in 0..20 {
            assert_eq!(requests.select_worker(&workers, &info), Some(1));
        }
    }

    #[test]
    fn least_requests_takes_the_fewest_in_flight_over_every_worker() {
        let policy = PowerOfTwoPolicy::with_load_metric(PowerOfTwoLoadMetric::LeastRequests);
        let feed = TestFeed::attached_to(&policy);
        let workers = holding(&[4, 5, 1, 3]);
        let mut loads = HashMap::new();
        loads.insert("http://w0:8000".to_string(), one_rank_load(0, 0.1, 100.0));
        loads.insert(
            "http://w2:8000".to_string(),
            one_rank_load(50_000, 0.1, 100.0),
        );
        feed.publish(&loads);
        for _ in 0..20 {
            assert_eq!(
                policy.select_worker(&workers, &SelectWorkerInfo::default()),
                Some(2)
            );
        }
    }

    /// Eight dark workers with nothing in flight, two of them not ready:
    /// every pair ties on the scorer, so each eligible worker wins one pick
    /// in six and a vetoed worker never. The draw is uniform over the
    /// eligible workers, as the listed pair was.
    #[test]
    fn a_pick_draws_its_pair_uniformly_over_the_eligible_workers_and_never_a_vetoed_one() {
        let policy = PowerOfTwoPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = (0..8)
            .map(|i| make_worker(&format!("http://w{i}:8000")))
            .collect();
        for vetoed in [2, 5] {
            workers[vetoed].set_status(openai_protocol::worker::WorkerStatus::NotReady);
        }
        let mut wins = [0usize; 8];
        for _ in 0..6_000 {
            let idx = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .expect("six workers are eligible");
            wins[idx] += 1;
        }
        assert_eq!(
            wins[2] + wins[5],
            0,
            "a not-ready worker won a pick: {wins:?}"
        );
        for (idx, &count) in wins.iter().enumerate() {
            if idx != 2 && idx != 5 {
                assert!(
                    (800..=1_200).contains(&count),
                    "worker {idx} won {count} of 6,000 picks, one in six expected: {wins:?}"
                );
            }
        }
    }

    /// A pool with one eligible worker picks it every time (the draw cannot
    /// find a second and the walk finds the one); a pool with none picks
    /// nothing.
    #[test]
    fn a_pool_with_one_eligible_worker_picks_it_and_a_vetoed_pool_picks_nothing() {
        let policy = PowerOfTwoPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = (0..4)
            .map(|i| make_worker(&format!("http://w{i}:8000")))
            .collect();
        for vetoed in [0, 1, 3] {
            workers[vetoed].set_status(openai_protocol::worker::WorkerStatus::NotReady);
        }
        for _ in 0..20 {
            assert_eq!(
                policy.select_worker(&workers, &SelectWorkerInfo::default()),
                Some(2)
            );
        }
        workers[2].set_status(openai_protocol::worker::WorkerStatus::NotReady);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            None
        );
    }
}
