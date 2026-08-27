use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use openai_protocol::worker::WorkerLoadResponse;
use rand::RngExt;
use tracing::debug;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::worker::{load_state::LoadSnapshot, Worker};

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

/// Since-poll dispatch tally for one worker.
#[derive(Clone, Copy, Debug, Default)]
struct SincePollDispatch {
    tokens: u64,
    requests: u64,
}

/// Least-(token-)work routing — route to the worker with the lowest estimated
/// time-to-drain plus a convex KV-pressure barrier (argmin, lower is better):
///
/// ```text
///   score_i = (queued_tokens_i + inflight_tokens_i) / throughput_i
///             + kv_pressure_weight · k_i / (1 − k_i)
/// ```
///
/// - `queued_tokens` — the backend's waiting-queue token-work
///   (`num_waiting_uncached_tokens`, or `waiting_reqs · p̄` when the backend
///   reports a queue depth but no token count for it). Token-work, not request
///   count, is what sets the wait under size-skewed traffic: a long prompt is
///   far more work than a short one, regardless of how many requests are queued.
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
/// - no queued-token report (backend exposes a queue depth but not its token
///   count, as Prometheus-gauge backends do): the queue is estimated at
///   `waiting_reqs · p̄`, keeping the queue visible in time units rather than
///   scoring a backlogged worker as idle. Only a backend reporting neither
///   scores `queued_tokens = 0`;
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
/// - `max_waiting_requests` (default `0` = disabled) — per-worker waiting-queue
///   cap: a worker whose reported waiting requests, plus requests dispatched to
///   it since its last poll, have reached the cap is skipped. When every
///   candidate is at the cap the selection returns none, so the request falls
///   to the router's admission queue instead of deepening a backlog. Set it
///   below the engine's max batch size.
#[derive(Debug)]
pub struct LeastLoadPolicy {
    /// Cached load reports from the worker monitor (keyed by worker URL).
    cached_loads: RwLock<HashMap<String, WorkerLoadResponse>>,
    /// Per-worker dispatch tally since the last load poll (keyed by worker
    /// URL); reset when a fresh report arrives. Token-work feeds the score's
    /// in-flight term; the request count feeds the waiting-queue veto.
    inflight_tokens: RwLock<HashMap<String, SincePollDispatch>>,
    /// KV-pressure weight `λ_t` (seconds).
    kv_pressure_weight: f64,
    /// Mean prefill length (tokens) for estimating in-flight token-work when a
    /// request's token count is unknown at routing time.
    mean_prefill_tokens: u32,
    /// Fallback throughput (tokens/s) for the `/throughput` term when a backend
    /// reports no live `gen_throughput`.
    default_throughput: f64,
    /// Per-worker waiting-queue cap; `0` disables the veto.
    max_waiting_requests: u32,
}

impl LeastLoadPolicy {
    pub fn new() -> Self {
        Self::with_params(
            DEFAULT_KV_PRESSURE_WEIGHT,
            DEFAULT_MEAN_PREFILL_TOKENS,
            DEFAULT_THROUGHPUT,
            0,
        )
    }

    pub fn with_kv_pressure_weight(kv_pressure_weight: f64) -> Self {
        Self::with_params(
            kv_pressure_weight,
            DEFAULT_MEAN_PREFILL_TOKENS,
            DEFAULT_THROUGHPUT,
            0,
        )
    }

    pub fn with_params(
        kv_pressure_weight: f64,
        mean_prefill_tokens: u32,
        default_throughput: f64,
        max_waiting_requests: u32,
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
            max_waiting_requests,
        }
    }

    /// Test-only view of the tunables so registry tests can assert
    /// operator values propagated.
    #[cfg(test)]
    pub(crate) fn params_for_test(&self) -> (f64, u32, f64, u32) {
        (
            self.kv_pressure_weight,
            self.mean_prefill_tokens,
            self.default_throughput,
            self.max_waiting_requests,
        )
    }

    /// Test-only view of one worker's backend-snapshot presence and atomic
    /// since-poll dispatch credit: `(has_load, tokens, requests)`.
    #[cfg(test)]
    pub(super) fn load_state_for_test(&self, url: &str) -> (bool, u64, u64) {
        let has_load = self
            .cached_loads
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .contains_key(url);
        let dispatch = self
            .inflight_tokens
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(url)
            .copied()
            .unwrap_or_default();
        (has_load, dispatch.tokens, dispatch.requests)
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
        complete_snapshot: Option<&LoadSnapshot>,
        inflight: &HashMap<String, SincePollDispatch>,
        nominal_throughput: f64,
        fleet_has_loads: bool,
    ) -> f64 {
        let url = worker.url();
        match Self::fresh_load(loads, complete_snapshot, url) {
            Some(load) => {
                let inflight_tokens = inflight.get(url).copied().unwrap_or_default().tokens as f64;
                let queued_tokens = self.queued_tokens(load);
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

    /// Look up a poll-fed load only while the complete WorkerMonitor snapshot
    /// still contains that URL. The published values are deliberately not used
    /// for scoring: `update_loads` is the boundary that pairs each successful
    /// worker's new load with its since-poll credit reset.
    fn fresh_load<'a>(
        loads: Option<&'a HashMap<String, WorkerLoadResponse>>,
        complete_snapshot: Option<&LoadSnapshot>,
        url: &str,
    ) -> Option<&'a WorkerLoadResponse> {
        if complete_snapshot.is_some_and(|snapshot| snapshot.get(url).is_none()) {
            return None;
        }
        loads.and_then(|map| map.get(url))
    }

    /// Waiting-queue token-work for a worker.
    ///
    /// Prefers the backend's own `num_waiting_uncached_tokens`. Backends that
    /// report a queue depth but no token count for it — anything scored from
    /// Prometheus gauges, which have no waiting-token equivalent — would
    /// otherwise be read as having an empty queue, and the policy would go
    /// blind to the very imbalance it exists to correct. Estimate their queue
    /// from the same mean prefill the in-flight term uses, so a queued request
    /// and a just-dispatched one weigh the same.
    fn queued_tokens(&self, load: &WorkerLoadResponse) -> f64 {
        let reported = load.total_waiting_uncached_tokens();
        if reported > 0 {
            return reported as f64;
        }
        load.total_waiting_reqs().max(0) as f64 * self.mean_prefill_tokens as f64
    }

    /// Token-work the request being routed adds to the chosen worker's
    /// in-flight estimate: its token count if known, else the mean prefill.
    fn request_tokens(&self, info: &SelectWorkerInfo) -> u64 {
        info.tokens
            .map(|t| t.len() as u64)
            .unwrap_or(self.mean_prefill_tokens as u64)
    }

    /// Argmin of the expected-wait score over `candidates` (indices into
    /// `workers`), crediting the winner's in-flight estimate. The nominal
    /// throughput and dark-fleet fallback are scoped to `candidates`, so a
    /// caller scoring a sampled subset (power-of-two) gets a self-consistent
    /// comparison. `policy` labels the selection log line.
    pub(super) fn select_min_expected_wait(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        info: &SelectWorkerInfo,
        policy: &'static str,
    ) -> Option<usize> {
        self.select_min_expected_wait_with_freshness(workers, candidates, info, policy, None)
    }

    /// CacheAware supplies the complete WorkerMonitor snapshot so an
    /// absent report cannot survive in this scorer's incremental poll cache.
    /// Only candidate URLs are checked; this does not scan the snapshot or the
    /// fleet, and selection plus winner credit remains one atomic operation.
    pub(super) fn select_min_expected_wait_with_freshness(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        info: &SelectWorkerInfo,
        policy: &'static str,
        complete_snapshot: Option<&LoadSnapshot>,
    ) -> Option<usize> {
        let loads_guard = self.cached_loads.read().ok();
        let loads = loads_guard.as_deref();

        // Waiting-queue veto: drop candidates whose reported queue, plus
        // requests dispatched since their last poll, has reached the cap.
        // Workers without a snapshot stay eligible — there is no queue
        // evidence to veto on, and a dark fleet must keep routing.
        let capped: Vec<usize>;
        let candidates = if self.max_waiting_requests == 0 {
            candidates
        } else {
            let cap = self.max_waiting_requests as u64;
            let inflight_guard = self
                .inflight_tokens
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            capped = candidates
                .iter()
                .copied()
                .filter(|&i| {
                    let url = workers[i].url();
                    match Self::fresh_load(loads, complete_snapshot, url) {
                        Some(load) => {
                            let since_poll = inflight_guard
                                .get(url)
                                .copied()
                                .unwrap_or_default()
                                .requests;
                            (load.total_waiting_reqs().max(0) as u64) + since_poll < cap
                        }
                        None => true,
                    }
                })
                .collect();
            &capped
        };
        let (&first, rest) = candidates.split_first()?;

        // Nominal throughput (mean of positive reports) stands in for a
        // worker missing a fresh snapshot; `fleet_has_loads` distinguishes a
        // partial gap (estimate that worker's drain time at the nominal rate)
        // from a fully dark fleet (fall back to join-shortest-queue).
        let (tp_sum, tp_count) = candidates
            .iter()
            .filter_map(|&i| Self::fresh_load(loads, complete_snapshot, workers[i].url()))
            .map(|l| l.total_gen_throughput())
            .filter(|t| *t > 0.0)
            .fold((0.0, 0u32), |(s, n), t| (s + t, n + 1));
        let nominal_throughput = if tp_count > 0 {
            tp_sum / tp_count as f64
        } else {
            self.default_throughput
        };
        let fleet_has_loads = candidates
            .iter()
            .any(|&i| Self::fresh_load(loads, complete_snapshot, workers[i].url()).is_some());

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
        let mut best = first;
        let mut best_score = self.score(
            &workers[best],
            loads,
            complete_snapshot,
            &inflight,
            nominal_throughput,
            fleet_has_loads,
        );
        let mut tied = 1u32;
        for &idx in rest {
            let s = self.score(
                &workers[idx],
                loads,
                complete_snapshot,
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
        let tally = inflight.entry(workers[best].url().to_string()).or_default();
        tally.tokens += req_tokens;
        tally.requests += 1;
        drop(inflight);

        debug!(
            "{policy} selected {} (score {:.4}, in_flight {})",
            workers[best].url(),
            best_score,
            workers[best].load()
        );
        workers[best].increment_processed();
        Some(best)
    }

    fn update_loads_inner<F>(&self, loads: &HashMap<String, WorkerLoadResponse>, after_publish: F)
    where
        F: FnOnce(),
    {
        // Selectors acquire these in the same order and retain the snapshot
        // guard through winner credit. Holding both before either mutation
        // makes snapshot publication and since-poll reset one critical section.
        let Ok(mut cached) = self.cached_loads.write() else {
            return;
        };
        let Ok(mut inflight) = self.inflight_tokens.write() else {
            return;
        };
        cached.extend(loads.iter().map(|(k, v)| (k.clone(), v.clone())));
        after_publish();
        // A fresh snapshot already reflects work up to the poll, so reset the
        // since-poll in-flight estimate for the workers it covers.
        for url in loads.keys() {
            inflight.insert(url.clone(), SincePollDispatch::default());
        }
    }
}

impl LoadBalancingPolicy for LeastLoadPolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        // The single-worker shortcut must not bypass the waiting-queue veto.
        if healthy.len() == 1 && self.max_waiting_requests == 0 {
            return Some(healthy[0]);
        }
        self.select_min_expected_wait(workers, &healthy, info, self.name())
    }

    fn name(&self) -> &'static str {
        "least_load"
    }

    fn update_loads(&self, loads: &HashMap<String, WorkerLoadResponse>) {
        self.update_loads_inner(loads, || {});
    }

    fn needs_backend_loads(&self) -> bool {
        true
    }

    fn remove_worker(&self, url: &str) {
        let Ok(mut cached) = self.cached_loads.write() else {
            return;
        };
        let Ok(mut inflight) = self.inflight_tokens.write() else {
            return;
        };
        cached.remove(url);
        inflight.remove(url);
    }

    fn reset(&self) {
        let Ok(mut cached) = self.cached_loads.write() else {
            return;
        };
        let Ok(mut inflight) = self.inflight_tokens.write() else {
            return;
        };
        cached.clear();
        inflight.clear();
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
            ..Default::default()
        }
    }

    /// One DP rank reporting a queue depth with no token count for it — the
    /// shape produced by any backend scored from Prometheus gauges, which have
    /// no waiting-token equivalent.
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
    fn waiting_queue_veto_skips_capped_worker() {
        // a would win the argmin (655s vs 10,000s) but reports 64 waiting
        // (>= cap 48); the veto must exclude it before scoring.
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 100.0, 48);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            make_load_reqs_only(64, 0.0, 100.0),
        );
        loads.insert(
            "http://b:8000".to_string(),
            make_load(1_000_000, 0.0, 100.0),
        );
        policy.update_loads(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn waiting_queue_veto_all_capped_returns_none() {
        // Every candidate at the cap: selection must fail so the request
        // falls to the router's admission queue instead of piling on.
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 100.0, 48);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            make_load_reqs_only(48, 0.0, 100.0),
        );
        loads.insert(
            "http://b:8000".to_string(),
            make_load_reqs_only(48, 0.0, 100.0),
        );
        policy.update_loads(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            None
        );
    }

    #[test]
    fn waiting_queue_veto_counts_since_poll_dispatches() {
        // Cap 2: a reports 1 waiting and wins the first pick; the dispatch
        // counts one since-poll request, lifting a to the cap, so the second
        // pick must go to b even though b scores far worse.
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 100.0, 2);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            make_load_reqs_only(1, 0.0, 100.0),
        );
        loads.insert("http://b:8000".to_string(), make_load(400_000, 0.0, 100.0));
        policy.update_loads(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn waiting_queue_veto_ignores_workers_without_snapshots() {
        // A dark fleet with a cap configured has no queue evidence to veto
        // on; routing must continue on join-shortest-queue.
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 100.0, 1);
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
    fn waiting_queue_veto_applies_to_single_worker() {
        // The one-healthy-worker shortcut must not bypass the veto.
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 100.0, 48);
        let workers = vec![mk("http://a:8000")];
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            make_load_reqs_only(64, 0.0, 100.0),
        );
        policy.update_loads(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            None
        );
    }

    #[test]
    fn waiting_queue_cap_zero_disables_veto() {
        // Cap 0 keeps the historical behavior: arbitrarily deep queues stay
        // routable.
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 100.0, 0);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            make_load_reqs_only(1000, 0.0, 100.0),
        );
        loads.insert(
            "http://b:8000".to_string(),
            make_load_reqs_only(1000, 0.0, 100.0),
        );
        policy.update_loads(&loads);
        assert!(policy
            .select_worker(&workers, &SelectWorkerInfo::default())
            .is_some());
    }

    #[test]
    fn nominal_throughput_is_scoped_to_the_candidates() {
        // Dark a (2 in-flight) vs reporting b, with an outsider c reporting an
        // extreme throughput. a's drain-time estimate must use the nominal
        // rate of the CANDIDATES (b's 100 tok/s -> 20.48s > b's 10.04s), not
        // a fleet-wide mean that c's 100k tok/s would dominate (0.04s < b).
        let policy = LeastLoadPolicy::new();
        let a = mk("http://a:8000");
        for _ in 0..2 {
            a.increment_load();
        }
        let workers = vec![a, mk("http://b:8000"), mk("http://c:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://b:8000".to_string(), make_load(1000, 0.2, 100.0));
        loads.insert("http://c:8000".to_string(), make_load(0, 0.2, 100_000.0));
        policy.update_loads(&loads);
        assert_eq!(
            policy.select_min_expected_wait(
                &workers,
                &[0, 1],
                &SelectWorkerInfo::default(),
                "test"
            ),
            Some(1)
        );
    }

    #[test]
    fn known_token_count_credits_exact_inflight_work() {
        // pick1 routes a 2100-token request to idle a, crediting exactly its
        // token count: a becomes 21.04s vs b's 20.52s, so pick2 goes to b.
        // The p̄ = 1024 fallback would leave a at 10.28s and herd onto a.
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.2, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(2048, 0.2, 100.0));
        policy.update_loads(&loads);

        let tokens = vec![7u32; 2100];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
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
    fn queue_depth_without_token_count_still_ranks_workers() {
        // Both backends report a queue but no token count for it. Reading the
        // absent count as an empty queue scores them identically, so the pick
        // is a coin flip and a badly backlogged worker keeps drawing traffic.
        let policy = LeastLoadPolicy::new(); // p̄ = 1024
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            make_load_reqs_only(64, 0.2, 100.0),
        );
        loads.insert(
            "http://b:8000".to_string(),
            make_load_reqs_only(6, 0.2, 100.0),
        );
        policy.update_loads(&loads);
        // a: 64*1024/100 ≈ 655s ; b: 6*1024/100 ≈ 61s -> b, on every draw
        // (20 is well short of the in-flight credit crossover).
        for _ in 0..20 {
            assert_eq!(
                policy.select_worker(&workers, &SelectWorkerInfo::default()),
                Some(1)
            );
        }
    }

    #[test]
    fn reported_queue_tokens_win_over_the_estimate() {
        // A backend reporting both must be scored on the real token count:
        // 200 queued short requests are less work than one long one.
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut a = make_load(500, 0.2, 100.0);
        a.loads[0].num_waiting_reqs = 200;
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), a);
        loads.insert("http://b:8000".to_string(), make_load(8000, 0.2, 100.0));
        policy.update_loads(&loads);
        // a: its reported 500/100 = 5s, not 200*1024 ; b: 8000/100 = 80s -> a.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn empty_queue_is_not_penalized() {
        // Neither backend has a queue; the estimate must not manufacture one,
        // leaving the KV barrier to decide.
        let policy = LeastLoadPolicy::with_kv_pressure_weight(2.0);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            make_load_reqs_only(0, 0.9, 100.0),
        );
        loads.insert(
            "http://b:8000".to_string(),
            make_load_reqs_only(0, 0.1, 100.0),
        );
        policy.update_loads(&loads);
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
            .any(|v| v.tokens > 0));

        // A fresh poll clears the since-poll estimate.
        policy.update_loads(&loads);
        assert!(policy
            .inflight_tokens
            .read()
            .unwrap()
            .values()
            .all(|v| v.tokens == 0));
    }

    #[test]
    fn update_publishes_snapshot_and_resets_credit_as_one_critical_section() {
        use std::sync::mpsc::sync_channel;

        let policy = Arc::new(LeastLoadPolicy::new());
        let workers = vec![mk("http://a:8000")];
        assert_eq!(
            policy.select_min_expected_wait(&workers, &[0], &SelectWorkerInfo::default(), "test"),
            Some(0)
        );
        assert_eq!(
            policy.load_state_for_test("http://a:8000"),
            (false, 1024, 1)
        );

        let loads = HashMap::from([("http://a:8000".to_string(), make_load(0, 0.1, 100.0))]);
        let (published_tx, published_rx) = sync_channel::<()>(0);
        let (resume_tx, resume_rx) = sync_channel::<()>(0);
        let updater_policy = Arc::clone(&policy);
        let updater = std::thread::spawn(move || {
            updater_policy.update_loads_inner(&loads, || {
                published_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
            });
        });

        published_rx.recv().unwrap();
        let inflight_was_locked = policy.inflight_tokens.try_write().is_err();
        resume_tx.send(()).unwrap();
        updater.join().unwrap();

        assert!(
            inflight_was_locked,
            "a selector could credit against the new snapshot before its reset"
        );
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 0, 0));
        assert_eq!(
            policy.select_min_expected_wait(&workers, &[0], &SelectWorkerInfo::default(), "test"),
            Some(0)
        );
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 1024, 1));
    }

    #[test]
    fn reset_discards_backend_loads_and_inflight_credit() {
        // Backend snapshots say a is badly queued and b is idle, even though
        // live request counts say the opposite. Before reset expected wait
        // must choose b; after reset the dark-fleet fallback must see only the
        // live counts and choose a. Keeping either cached map makes the second
        // assertion fail.
        let policy = LeastLoadPolicy::new();
        let a = mk("http://a:8000");
        let b = mk("http://b:8000");
        for _ in 0..5 {
            b.increment_load();
        }
        let workers = vec![a, b];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(100_000, 0.1, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.1, 100.0));
        policy.update_loads(&loads);

        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
        policy.reset();
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
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
