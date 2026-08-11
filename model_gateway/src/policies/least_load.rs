use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use openai_protocol::worker::WorkerLoadResponse;
use rand::RngExt;
use tracing::debug;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::Worker;

/// Default KV-pressure weight `λ_t` (seconds): the time-cost of KV contention,
/// chosen commensurate with the expected-queue-wait term so the two add cleanly.
pub const DEFAULT_KV_PRESSURE_WEIGHT: f64 = 0.15;

/// Default mean prefill length (tokens), used to estimate in-flight token-work
/// for a dispatched request whose token count is unknown at routing time.
pub const DEFAULT_MEAN_PREFILL_TOKENS: u32 = 1024;

/// Default fallback throughput (tokens/s) for the `/throughput` term when a
/// backend reports KV usage but no live `gen_throughput`. On a homogeneous
/// fleet its absolute value mainly sets the work-vs-barrier balance, so it
/// co-tunes with `kv_pressure_weight`.
pub const DEFAULT_THROUGHPUT: f64 = 2000.0;

/// Least-(token-)work routing — route to the worker with the lowest estimated
/// time-to-drain plus a convex KV-pressure barrier (argmin, lower is better):
///
/// ```text
///   score_i = (queued_tokens_i + inflight_tokens_i) / throughput_i
///             + kv_pressure_weight · k_i / (1 − k_i)
/// ```
///
/// - `queued_tokens` — the backend's waiting-queue token-work
///   (`num_waiting_uncached_tokens`). Token-work, not request count, is what
///   sets the wait under size-skewed traffic: a long prompt is far more work
///   than a short one, regardless of how many requests are queued.
/// - `inflight_tokens` — token-work this router has dispatched to the worker
///   since its last load poll. Polls are stale between intervals; without this
///   correction, plain argmin sends a whole interval's arrivals to one worker
///   (incast). Crediting each dispatch water-fills load across workers instead.
/// - `/ throughput` — normalizes work to *time*, comparing heterogeneous
///   workers by drain time rather than raw token count.
/// - `k / (1 − k)` — the M/M/1 expected-occupancy barrier on KV utilization
///   `k`; convex and divergent at the KV cliff, so routing avoids the
///   preemption/recompute that begins as KV fills.
///
/// Both terms are in seconds, so they add directly. Missing signals degrade
/// gracefully and stay in time units:
/// - no queued-token report (backend doesn't expose waiting-queue tokens):
///   `queued_tokens = 0`, leaving in-flight-corrected drain time plus the barrier;
/// - zero/absent throughput (backend reports no generation rate): falls back to
///   the configured `default_throughput`, so the work term stays in seconds and
///   the KV barrier stays relevant;
/// - a worker with no fresh snapshot while peers report: its live in-flight is
///   converted to a drain-time estimate (`load · p̄ / fleet_nominal_throughput`)
///   so it is comparable to reporting workers, not scored on a raw count;
/// - the whole fleet dark (true cold start, or a backend that never reports
///   loads): join-shortest-queue on the live in-flight count.
///
/// In-flight token-work is exact on the gRPC routing path (the request's token
/// count is known at selection); the HTTP path has no token count and falls
/// back to `p̄ · count`, which is weaker on size-skewed traffic. This policy is
/// therefore intended for gRPC workers.
///
/// # Tuning knobs
///
/// All are fields of `PolicyConfig::LeastLoad` with the defaults below:
/// - `kv_pressure_weight` (λ_t, default `0.15` s) — weight of the KV-pressure
///   barrier. Raise it to steer harder away from near-full KV; lower it to
///   weight raw drain time more.
/// - `default_throughput` (default `2000` tok/s) — drain rate used when a
///   backend reports no live `gen_throughput`. Set it to the fleet's measured
///   per-replica generation rate; it co-tunes with `kv_pressure_weight`.
/// - `mean_prefill_tokens` (p̄, default `1024`) — per-request token estimate for
///   the in-flight term when the request's token count is unknown at routing
///   (the HTTP path; ignored when tokens are known, i.e. gRPC).
/// - `load_check_interval_secs` (default `10`) — worker-load poll period; the
///   in-flight correction absorbs staleness between polls.
#[derive(Debug)]
pub struct LeastLoadPolicy {
    /// Cached load reports from the worker monitor (keyed by worker URL).
    cached_loads: RwLock<HashMap<String, WorkerLoadResponse>>,
    /// In-flight token-work dispatched per worker since its last load poll
    /// (keyed by worker URL); reset when a fresh report arrives.
    inflight_tokens: RwLock<HashMap<String, u64>>,
    /// KV-pressure weight `λ_t` (seconds).
    kv_pressure_weight: f64,
    /// Mean prefill length (tokens) for estimating in-flight token-work when a
    /// request's token count is unknown at routing time.
    mean_prefill_tokens: u32,
    /// Fallback throughput (tokens/s) for the `/throughput` term when a backend
    /// reports no live `gen_throughput`.
    default_throughput: f64,
}

impl LeastLoadPolicy {
    pub fn new() -> Self {
        Self::with_params(
            DEFAULT_KV_PRESSURE_WEIGHT,
            DEFAULT_MEAN_PREFILL_TOKENS,
            DEFAULT_THROUGHPUT,
        )
    }

    pub fn with_kv_pressure_weight(kv_pressure_weight: f64) -> Self {
        Self::with_params(
            kv_pressure_weight,
            DEFAULT_MEAN_PREFILL_TOKENS,
            DEFAULT_THROUGHPUT,
        )
    }

    pub fn with_params(
        kv_pressure_weight: f64,
        mean_prefill_tokens: u32,
        default_throughput: f64,
    ) -> Self {
        Self {
            cached_loads: RwLock::new(HashMap::new()),
            inflight_tokens: RwLock::new(HashMap::new()),
            kv_pressure_weight: if kv_pressure_weight.is_finite() && kv_pressure_weight >= 0.0 {
                kv_pressure_weight
            } else {
                DEFAULT_KV_PRESSURE_WEIGHT
            },
            mean_prefill_tokens: mean_prefill_tokens.max(1),
            default_throughput: if default_throughput.is_finite() && default_throughput > 0.0 {
                default_throughput
            } else {
                DEFAULT_THROUGHPUT
            },
        }
    }

    /// Expected-wait score for a worker (lower is better).
    ///
    /// `inflight` maps worker URL -> token-work dispatched since its last poll.
    /// `nominal_throughput` (a peer-derived mean) estimates drain rate for a
    /// worker missing a fresh snapshot; `fleet_has_loads` is false only when no
    /// worker reports at all, in which case we fall back to join-shortest-queue
    /// on the live in-flight count (which, unlike the since-poll estimate,
    /// reflects completions and so suits backends that never report loads).
    fn score(
        &self,
        worker: &Arc<dyn Worker>,
        loads: Option<&HashMap<String, WorkerLoadResponse>>,
        inflight: &HashMap<String, u64>,
        nominal_throughput: f64,
        fleet_has_loads: bool,
    ) -> f64 {
        let url = worker.url();
        match loads.and_then(|m| m.get(url)) {
            Some(load) => {
                let inflight_tokens = inflight.get(url).copied().unwrap_or(0) as f64;
                let queued_tokens = load.total_waiting_uncached_tokens() as f64;
                let live_throughput = load.total_gen_throughput();
                let throughput = if live_throughput > 0.0 {
                    live_throughput
                } else {
                    self.default_throughput
                };
                let k = load.effective_token_usage().clamp(0.0, 0.999);
                (queued_tokens + inflight_tokens) / throughput
                    + self.kv_pressure_weight * k / (1.0 - k)
            }
            // No fresh snapshot, but peers report: estimate this worker's drain
            // time from its live in-flight (count × mean prefill) at the fleet's
            // nominal throughput, keeping the same units as reporting workers.
            None if fleet_has_loads => {
                worker.load() as f64 * self.mean_prefill_tokens as f64 / nominal_throughput
            }
            // Whole fleet dark (cold start, or a backend that never reports
            // loads): join-shortest-queue on live in-flight.
            None => worker.load() as f64,
        }
    }

    /// Token-work the request being routed adds to the chosen worker's
    /// in-flight estimate: its token count if known, else the mean prefill.
    fn request_tokens(&self, info: &SelectWorkerInfo) -> u64 {
        info.tokens
            .map(|t| t.len() as u64)
            .unwrap_or(self.mean_prefill_tokens as u64)
    }
}

impl LoadBalancingPolicy for LeastLoadPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        if healthy.len() == 1 {
            return Some(healthy[0]);
        }

        let loads_guard = self.cached_loads.read().ok();
        let loads = loads_guard.as_deref();

        // Fleet-nominal throughput (mean of positive reports) stands in for a
        // worker missing a fresh snapshot; `fleet_has_loads` distinguishes a
        // partial gap (estimate that worker's drain time at the nominal rate)
        // from a fully dark fleet (fall back to join-shortest-queue).
        let (tp_sum, tp_count) = healthy
            .iter()
            .filter_map(|&i| loads.and_then(|m| m.get(workers[i].url())))
            .map(|l| l.total_gen_throughput())
            .filter(|t| *t > 0.0)
            .fold((0.0, 0u32), |(s, n), t| (s + t, n + 1));
        let nominal_throughput = if tp_count > 0 {
            tp_sum / tp_count as f64
        } else {
            self.default_throughput
        };
        let fleet_has_loads = loads
            .map(|m| healthy.iter().any(|&i| m.contains_key(workers[i].url())))
            .unwrap_or(false);

        // Held across selection so the in-flight estimate stays consistent and
        // the chosen worker can be credited before the guard is released.
        let mut inflight = self
            .inflight_tokens
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Argmin with reservoir tie-breaking: equal-score workers (the common
        // idle/homogeneous case scores exactly equal) are sampled uniformly
        // instead of first-index-wins, which herded ties onto one worker.
        let mut rng = rand::rng();
        let mut best = healthy[0];
        let mut best_score = self.score(
            &workers[best],
            loads,
            &inflight,
            nominal_throughput,
            fleet_has_loads,
        );
        let mut tied = 1u32;
        for &idx in &healthy[1..] {
            let s = self.score(
                &workers[idx],
                loads,
                &inflight,
                nominal_throughput,
                fleet_has_loads,
            );
            if s < best_score {
                best = idx;
                best_score = s;
                tied = 1;
            } else if s == best_score {
                // Keep each tying candidate with probability 1/k so the final
                // pick is uniform over all ties without collecting them.
                tied += 1;
                if rng.random_range(0..tied) == 0 {
                    best = idx;
                }
            }
        }

        // In-flight correction: credit the chosen worker with this request's
        // token-work until its next poll refreshes the snapshot.
        let req_tokens = self.request_tokens(info);
        *inflight.entry(workers[best].url().to_string()).or_insert(0) += req_tokens;
        drop(inflight);

        debug!(
            "least_load selected {} (score {:.4}, in_flight {})",
            workers[best].url(),
            best_score,
            workers[best].load()
        );
        workers[best].increment_processed();
        Some(best)
    }

    fn name(&self) -> &'static str {
        "least_load"
    }

    fn update_loads(&self, loads: &HashMap<String, WorkerLoadResponse>) {
        if let Ok(mut cached) = self.cached_loads.write() {
            cached.extend(loads.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        // A fresh snapshot already reflects work up to the poll, so reset the
        // since-poll in-flight estimate for the workers it covers.
        if let Ok(mut inflight) = self.inflight_tokens.write() {
            for url in loads.keys() {
                inflight.insert(url.clone(), 0);
            }
        }
    }

    fn remove_worker(&self, url: &str) {
        if let Ok(mut cached) = self.cached_loads.write() {
            cached.remove(url);
        }
        if let Ok(mut inflight) = self.inflight_tokens.write() {
            inflight.remove(url);
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for LeastLoadPolicy {
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
                num_running_reqs: 0,
                num_waiting_reqs: 0,
                num_waiting_uncached_tokens,
                num_total_reqs: 0,
                num_used_tokens: 0,
                max_total_num_tokens: 0,
                token_usage,
                gen_throughput,
                cache_hit_rate: 0.0,
                utilization: 0.0,
                max_running_requests: 0,
                ..Default::default()
            }],
        }
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
    fn equal_score_ties_spread_across_workers() {
        // Three identically-loaded workers score exactly equal; the argmin
        // must sample ties uniformly rather than herd on the first index.
        // Fresh policy per draw: selection credits in-flight tokens to the
        // winner, which breaks the tie for subsequent draws on one instance.
        let urls = ["http://a:8000", "http://b:8000", "http://c:8000"];
        let mut seen = [false; 3];
        for _ in 0..150 {
            let policy = LeastLoadPolicy::new();
            let workers: Vec<Arc<dyn Worker>> = urls.iter().map(|u| mk(u)).collect();
            let mut loads = HashMap::new();
            for url in urls {
                loads.insert(url.to_string(), make_load(1000, 0.2, 100.0));
            }
            policy.update_loads(&loads);
            let idx = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .unwrap();
            seen[idx] = true;
            if seen.iter().all(|&s| s) {
                break;
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "equal-score ties must spread across all workers, saw {seen:?}"
        );
    }

    #[test]
    fn cold_start_picks_lowest_in_flight() {
        // No load reports yet -> join-shortest-queue on live in-flight count.
        let policy = LeastLoadPolicy::new();
        let a = mk("http://a:8000");
        let b = mk("http://b:8000");
        for _ in 0..5 {
            a.increment_load();
        }
        let workers = vec![a, b];
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn routes_to_lower_queued_token_work() {
        // Equal KV/throughput; the worker with fewer queued tokens wins.
        let policy = LeastLoadPolicy::new();
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
    fn throughput_normalization_prefers_faster_worker() {
        // Same queued tokens; the faster worker (higher throughput) drains sooner.
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(5000, 0.2, 50.0));
        loads.insert("http://b:8000".to_string(), make_load(5000, 0.2, 500.0));
        policy.update_loads(&loads);
        // a: 5000/50 = 100s ; b: 5000/500 = 10s -> pick b.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn zero_throughput_falls_back_to_default() {
        // A backend that reports no gen_throughput (0); the score must still
        // discriminate via the configured default_throughput, not collapse.
        let policy = LeastLoadPolicy::new(); // default_throughput = 2000
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(10000, 0.2, 0.0));
        loads.insert("http://b:8000".to_string(), make_load(1000, 0.2, 0.0));
        policy.update_loads(&loads);
        // gen_throughput=0 -> default 2000: a 10000/2000=5s ; b 1000/2000=0.5s -> pick b.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn missing_snapshot_estimated_in_time_units() {
        // Worker a reports ~40s of queued work; worker b has no snapshot but 5
        // live in-flight. Scoring b on raw count (5) would wrongly beat a's 40s;
        // scoring it as drain time (5 * p̄ / nominal ≈ 51s) keeps the lighter a.
        let policy = LeastLoadPolicy::new(); // p̄ = 1024
        let a = mk("http://a:8000");
        let b = mk("http://b:8000");
        for _ in 0..5 {
            b.increment_load();
        }
        let workers = vec![a, b];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(4000, 0.0, 100.0));
        policy.update_loads(&loads);
        // a: 4000/100 = 40s ; b: 5 * 1024 / 100 ≈ 51.2s -> pick a.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn kv_barrier_avoids_full_worker() {
        // No queued work; the convex KV barrier steers off the near-full worker.
        let policy = LeastLoadPolicy::with_kv_pressure_weight(2.0);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.98, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.0, 100.0));
        policy.update_loads(&loads);
        // a: 0 + 2*0.98/0.02 = 98 ; b: 0 -> pick b.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn inflight_correction_spreads_within_poll_interval() {
        // Two identical workers, no fresh poll between dispatches: the in-flight
        // token credit must push the second request to the other worker rather
        // than herding both onto the first.
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.1, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.1, 100.0));
        policy.update_loads(&loads);

        let info = SelectWorkerInfo::default(); // tokens unknown -> mean prefill
        let first = policy.select_worker(&workers, &info).unwrap();
        let second = policy.select_worker(&workers, &info).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn update_loads_resets_inflight() {
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.1, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.1, 100.0));
        policy.update_loads(&loads);

        let info = SelectWorkerInfo::default();
        for _ in 0..4 {
            policy.select_worker(&workers, &info);
        }
        assert!(policy
            .inflight_tokens
            .read()
            .unwrap()
            .values()
            .any(|&v| v > 0));

        // A fresh poll clears the since-poll estimate.
        policy.update_loads(&loads);
        assert!(policy
            .inflight_tokens
            .read()
            .unwrap()
            .values()
            .all(|&v| v == 0));
    }

    #[test]
    fn single_worker_always_selected() {
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000")];
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn remove_worker_prunes_state() {
        let policy = LeastLoadPolicy::new();
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.5, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.3, 100.0));
        policy.update_loads(&loads);
        assert_eq!(policy.cached_loads.read().unwrap().len(), 2);

        // Removing a worker drops only its entry (no unbounded growth on churn).
        policy.remove_worker("http://a:8000");
        let cached = policy.cached_loads.read().unwrap();
        assert_eq!(cached.len(), 1);
        assert!(!cached.contains_key("http://a:8000"));
        assert!(cached.contains_key("http://b:8000"));
    }
}
