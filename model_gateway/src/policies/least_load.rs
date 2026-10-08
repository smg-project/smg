use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex, RwLock},
    time::Instant,
};

use openai_protocol::worker::WorkerLoadResponse;
use rand::{rngs::StdRng, RngExt, SeedableRng};
use tracing::debug;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
// Retain the public paths used by existing callers and configuration defaults.
pub use crate::worker::expected_wait::{
    DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT,
};
use crate::{
    observability::cache_trace,
    worker::{expected_wait::ExpectedWait, load_state::LoadSnapshot, Worker},
};

/// Dispatches kept per worker between reports. A worker that never reports
/// (a dark fleet is scored by its live in-flight count instead) would
/// otherwise accumulate them without end; past this the oldest are dropped.
const SINCE_POLL_DISPATCHES_KEPT: usize = 8_192;

/// Expected waits within this of the minimum are one tie and are drawn from
/// uniformly. Idle workers with identical reports score exactly equal, but a
/// report's throughput or a KV digit sets two otherwise identical workers a
/// few nanoseconds apart, and an exact-equality tie then hands every request
/// to the lower index: a cold fleet never spreads that way.
const TIE_EPSILON_SECS: f64 = 1e-6;

/// Since-poll dispatch tally for one worker: the token-work and request count
/// of the dispatches no report has reflected yet, with each dispatch's
/// instant, so a report that says when the engine was sampled releases
/// exactly the dispatches it saw and keeps the later ones as credit.
#[derive(Clone, Debug, Default)]
struct SincePollDispatch {
    tokens: u64,
    requests: u64,
    /// `(dispatched_at, tokens)` per dispatch, oldest first.
    dispatches: VecDeque<(Instant, u64)>,
}

impl SincePollDispatch {
    fn record(&mut self, at: Instant, tokens: u64) {
        self.tokens += tokens;
        self.requests += 1;
        self.dispatches.push_back((at, tokens));
        if self.dispatches.len() > SINCE_POLL_DISPATCHES_KEPT {
            self.pop_oldest();
        }
    }

    /// Release the dispatches a report sampled at `sampled_at` already
    /// reflects: those made at or before it.
    fn release_through(&mut self, sampled_at: Instant) {
        while self
            .dispatches
            .front()
            .is_some_and(|&(at, _)| at <= sampled_at)
        {
            self.pop_oldest();
        }
    }

    fn pop_oldest(&mut self) {
        if let Some((_, tokens)) = self.dispatches.pop_front() {
            self.tokens -= tokens;
            self.requests -= 1;
        }
    }
}

/// Least-(token-)work routing — route to the worker with the lowest estimated
/// time-to-drain plus a convex KV-pressure barrier (argmin, lower is better):
///
/// ```text
///   score_i = (queued_tokens_i + running_estimated_tokens_i + inflight_tokens_i) / throughput_i
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
/// - running requests are estimated
///   at `running_reqs · p̄`: a poll releases dispatch credit but does not mean
///   the request finished. This is a fallback estimate, not remaining prefill;
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
    /// Per-worker dispatch tally since the last load report (keyed by worker
    /// URL): a report that carries `sampled_at` releases the dispatches made
    /// up to that instant, one without it resets the tally. Token-work feeds
    /// the score's in-flight term; the request count feeds the waiting-queue
    /// veto.
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
    /// Seeded source for the tie draw when set (tests reproduce a selection
    /// sequence with it); the thread's generator otherwise.
    tie_rng: Option<Mutex<StdRng>>,
}

/// Everything one expected-wait score reads besides the worker itself.
#[derive(Clone, Copy)]
struct ScoreInputs<'a> {
    loads: Option<&'a HashMap<String, WorkerLoadResponse>>,
    complete_snapshot: Option<&'a LoadSnapshot>,
    inflight: &'a HashMap<String, SincePollDispatch>,
    nominal_throughput: f64,
    fleet_has_loads: bool,
    /// The best-known reporting peer's score: what a worker without a
    /// fresh report scores before its own in-flight.
    peer_baseline: f64,
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
            tie_rng: None,
        }
    }

    /// Draw ties from a seeded generator instead of the thread's.
    pub fn with_tie_break_seed(mut self, seed: u64) -> Self {
        self.tie_rng = Some(Mutex::new(StdRng::seed_from_u64(seed)));
        self
    }

    /// Uniform draw in `0..n`: the reservoir step of the argmin's tie-break.
    fn tie_draw(&self, n: u32) -> u32 {
        match &self.tie_rng {
            Some(rng) => rng
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .random_range(0..n),
            None => rand::rng().random_range(0..n),
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
        let inflight = self
            .inflight_tokens
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let dispatch = inflight.get(url);
        (
            has_load,
            dispatch.map_or(0, |dispatch| dispatch.tokens),
            dispatch.map_or(0, |dispatch| dispatch.requests),
        )
    }

    /// Expected-wait score for a worker (lower is better).
    ///
    /// `inflight` maps worker URL -> token-work dispatched since its last poll.
    /// `nominal_throughput` (a peer-derived mean) estimates drain rate for a
    /// worker missing a fresh snapshot; `fleet_has_loads` is false only when no
    /// worker reports at all, in which case we fall back to join-shortest-queue
    /// on the live in-flight count (which, unlike the since-poll estimate,
    /// reflects completions and so suits backends that never report loads).
    fn score(&self, worker: &Arc<dyn Worker>, inputs: &ScoreInputs<'_>) -> f64 {
        let ScoreInputs {
            loads,
            complete_snapshot,
            inflight,
            nominal_throughput,
            fleet_has_loads,
            peer_baseline,
        } = *inputs;
        let url = worker.url();
        match Self::fresh_load(loads, complete_snapshot, url) {
            Some(load) => {
                let inflight_tokens = inflight.get(url).map_or(0, |dispatch| dispatch.tokens);
                let queued_tokens = self.queued_tokens(load) + self.running_estimated_tokens(load);
                ExpectedWait::new(
                    queued_tokens,
                    self.drain_rate(load),
                    load.effective_kv_pressure(),
                    self.kv_pressure_weight,
                )
                .seconds(inflight_tokens)
            }
            // No fresh snapshot, but peers report: score it as the best-known
            // reporting peer plus its own live in-flight (count × mean prefill)
            // at the fleet's nominal throughput. Unknown load must not read as
            // idle, or every request would herd onto the one worker whose
            // report is missing; nor must it be starved.
            None if fleet_has_loads => {
                peer_baseline
                    + worker.load() as f64 * self.mean_prefill_tokens as f64 / nominal_throughput
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

    /// The drain rate a reporting worker's wait is priced at: its live
    /// generation rate, else the configured default.
    fn drain_rate(&self, load: &WorkerLoadResponse) -> f64 {
        let live = load.total_gen_throughput();
        if live > 0.0 {
            live
        } else {
            self.default_throughput
        }
    }

    /// The scoring inputs for one pass over `candidates`: the nominal drain
    /// rate (mean of the positive reports) that stands in for a worker
    /// missing a fresh snapshot; whether anyone reports at all, which
    /// separates a partial gap (estimate the missing worker at the nominal
    /// rate) from a dark fleet (join-shortest-queue on live in-flight); and
    /// the best-known reporting peer's score, which a worker without a report
    /// starts from: never better than a worker whose load is known, never
    /// starved by one. The baseline is computed only when some candidate
    /// lacks a report, so the common all-reporting case pays nothing for it.
    fn score_inputs<'a>(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        loads: Option<&'a HashMap<String, WorkerLoadResponse>>,
        complete_snapshot: Option<&'a LoadSnapshot>,
        inflight: &'a HashMap<String, SincePollDispatch>,
    ) -> ScoreInputs<'a> {
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
        let reporting = candidates
            .iter()
            .filter(|&&i| Self::fresh_load(loads, complete_snapshot, workers[i].url()).is_some())
            .count();
        let mut inputs = ScoreInputs {
            loads,
            complete_snapshot,
            inflight,
            nominal_throughput,
            fleet_has_loads: reporting > 0,
            peer_baseline: 0.0,
        };
        if reporting < candidates.len() {
            let best_known = candidates
                .iter()
                .filter(|&&i| {
                    Self::fresh_load(loads, complete_snapshot, workers[i].url()).is_some()
                })
                .map(|&i| self.score(&workers[i], &inputs))
                .fold(f64::INFINITY, f64::min);
            // Nobody reports: the dark-fleet arm scores by live in-flight and
            // never reads the baseline; keep it neutral rather than infinite.
            if best_known.is_finite() {
                inputs.peer_baseline = best_known;
            }
        }
        inputs
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

    /// Load snapshots do not report remaining running token-work. Keep running
    /// requests visible after the poll resets dispatch credit, including during
    /// a mixed-version rollout: optional KV fields must not change this cost.
    fn running_estimated_tokens(&self, load: &WorkerLoadResponse) -> f64 {
        load.loads
            .iter()
            .map(|rank| rank.num_running_reqs.max(0) as f64 * self.mean_prefill_tokens as f64)
            .sum()
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
                                .map_or(0, |dispatch| dispatch.requests);
                            let waiting = load.total_waiting_reqs().max(0) as u64;
                            let eligible = waiting + since_poll < cap;
                            if cache_trace::enabled() {
                                cache_trace::gate(serde_json::json!({
                                    "source": "waiting_queue_cap", "worker": url,
                                    "waiting_requests": waiting, "since_poll_requests": since_poll,
                                    "cap": cap, "eligible": eligible,
                                }));
                            }
                            eligible
                        }
                        None => true,
                    }
                })
                .collect();
            &capped
        };
        let (&first, rest) = candidates.split_first()?;

        // Held across selection so the in-flight estimate stays consistent and
        // the chosen worker can be credited before the guard is released.
        let mut inflight = self
            .inflight_tokens
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let inputs = self.score_inputs(workers, candidates, loads, complete_snapshot, &inflight);

        // Argmin with reservoir tie-breaking: workers within
        // `TIE_EPSILON_SECS` of the minimum (the common idle/homogeneous case
        // scores equal to the digit) are sampled uniformly instead of
        // first-index-wins, which herded ties onto one worker.
        let observe_score = |idx: usize, score: f64| {
            if cache_trace::enabled() {
                let url = workers[idx].url();
                let load = Self::fresh_load(loads, complete_snapshot, url);
                cache_trace::score(serde_json::json!({
                    "source": "expected_wait", "worker": url, "score": score,
                    "score_unit": if inputs.fleet_has_loads { "seconds" } else { "requests" },
                    "queued_tokens": load.map(|load| self.queued_tokens(load)),
                    "running_estimated_tokens": load.map(|load| self.running_estimated_tokens(load)),
                    "kv_pressure": load.map(WorkerLoadResponse::effective_kv_pressure),
                    "resident_token_usage": load.map(WorkerLoadResponse::effective_token_usage),
                    "drain_rate": load.map(|load| self.drain_rate(load)),
                    "since_poll_tokens": inflight.get(url).map_or(0, |dispatch| dispatch.tokens),
                    "fresh_load": load.is_some(),
                }));
            }
        };
        let mut best = first;
        let mut best_score = self.score(&workers[best], &inputs);
        observe_score(best, best_score);
        let mut tied = 1u32;
        for &idx in rest {
            let s = self.score(&workers[idx], &inputs);
            observe_score(idx, s);
            if s < best_score - TIE_EPSILON_SECS {
                best = idx;
                best_score = s;
                tied = 1;
            } else if (s - best_score).abs() <= TIE_EPSILON_SECS {
                // Keep each tying candidate with probability 1/k so the final
                // pick is uniform over all ties without collecting them.
                tied += 1;
                best_score = best_score.min(s);
                if self.tie_draw(tied) == 0 {
                    best = idx;
                }
            }
        }

        // In-flight correction: credit the chosen worker with this request's
        // token-work until its next poll refreshes the snapshot.
        let req_tokens = self.request_tokens(info);
        inflight
            .entry(workers[best].url().to_string())
            .or_default()
            .record(Instant::now(), req_tokens);
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
        // A report reflects the work dispatched up to the instant the engine
        // was sampled: release those dispatches and keep the later ones as
        // credit, so a record republished unchanged (the monitor republishes
        // the shared snapshot when a pushed record arrives) releases nothing
        // twice and a dispatch made after the sample is not lost. A report
        // with no sample instant resets the tally as before.
        for (url, load) in loads {
            match load.sampled_at {
                Some(sampled_at) => {
                    if let Some(dispatch) = inflight.get_mut(url) {
                        dispatch.release_through(sampled_at);
                    }
                }
                None => {
                    inflight.insert(url.clone(), SincePollDispatch::default());
                }
            }
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
    fn running_work_survives_the_poll_credit_reset() {
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut busy = make_load(0, 0.0, 0.0);
        busy.loads[0].num_running_reqs = 1;
        let loads = HashMap::from([
            (workers[0].url().to_string(), busy),
            (workers[1].url().to_string(), make_load(0, 0.1, 0.0)),
        ]);
        // Dispatch credit is gone once the scheduler reports the request.
        policy.update_loads(&loads);
        assert_eq!(policy.load_state_for_test(workers[0].url()), (true, 0, 0));
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn reclaimable_prefix_cache_does_not_make_an_idle_worker_busy() {
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut idle = make_load(0, 1.0, 0.0);
        idle.loads[0].active_token_usage = Some(0.0);
        let mut busy = make_load(0, 0.5, 0.0);
        busy.loads[0].active_token_usage = Some(0.5);
        busy.loads[0].num_running_reqs = 1;
        policy.update_loads(&HashMap::from([
            (workers[0].url().to_string(), idle),
            (workers[1].url().to_string(), busy),
        ]));
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn a_cold_fleet_of_128_spreads_a_thousand_misses() {
        // No report anywhere: every worker scores its live in-flight count,
        // zero for all, so the whole fleet ties on every request.
        let policy = LeastLoadPolicy::new().with_tie_break_seed(7);
        let workers: Vec<Arc<dyn Worker>> =
            (0..128).map(|i| mk(&format!("http://w{i}:8000"))).collect();
        let info = SelectWorkerInfo::default();
        let mut hits = vec![0usize; workers.len()];
        for _ in 0..1000 {
            hits[policy.select_worker(&workers, &info).unwrap()] += 1;
        }
        let mean = 1000.0 / workers.len() as f64;
        assert!(
            hits.iter().all(|&h| h >= 1),
            "a worker never chosen: {hits:?}"
        );
        let max = *hits.iter().max().unwrap();
        assert!(
            max as f64 <= 3.0 * mean,
            "max {max} over a mean of {mean:.1}: {hits:?}"
        );
    }

    #[test]
    fn a_strictly_cheaper_worker_wins_every_time() {
        let policy = LeastLoadPolicy::new().with_tie_break_seed(7);
        let workers: Vec<Arc<dyn Worker>> =
            (0..128).map(|i| mk(&format!("http://w{i}:8000"))).collect();
        for (i, worker) in workers.iter().enumerate() {
            if i != 77 {
                worker.increment_load();
            }
        }
        let info = SelectWorkerInfo::default();
        for _ in 0..1000 {
            assert_eq!(policy.select_worker(&workers, &info), Some(77));
        }
    }

    #[test]
    fn scores_within_epsilon_of_the_minimum_tie() {
        // Four idle workers whose reports differ by a KV digit far below the
        // epsilon; an exact-equality tie handed everything to the first.
        let policy = LeastLoadPolicy::new().with_tie_break_seed(7);
        let workers: Vec<Arc<dyn Worker>> =
            (0..4).map(|i| mk(&format!("http://w{i}:8000"))).collect();
        let mut loads = HashMap::new();
        for (i, worker) in workers.iter().enumerate() {
            loads.insert(
                worker.url().to_string(),
                make_load(0, i as f64 * 1e-9, 100.0),
            );
        }
        policy.update_loads(&loads);
        let info = SelectWorkerInfo::default();
        let mut hits = [0usize; 4];
        for _ in 0..400 {
            hits[policy.select_worker(&workers, &info).unwrap()] += 1;
            // Release the winner's credit so every pick sees the same four scores.
            policy.update_loads(&loads);
        }
        assert!(hits.iter().all(|&h| h >= 50), "{hits:?}");
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
    fn a_worker_without_a_report_is_scored_like_its_peers() {
        // Two workers report queues of different depth; the third never
        // answered the load poll. Unknown load must not read as idle, or the
        // whole fleet herds onto the one worker nobody has heard from; nor
        // may it read as the peers' average, which would let the deeper
        // queue beat it. It ties with the best-known peer: both the lightly
        // loaded peer and the unreported worker are picked, the heavily
        // loaded peer never.
        let urls = ["http://a:8000", "http://b:8000", "http://c:8000"];
        let mut seen = [0usize; 3];
        for _ in 0..150 {
            let policy = LeastLoadPolicy::new();
            let workers: Vec<Arc<dyn Worker>> = urls.iter().map(|u| mk(u)).collect();
            let mut loads = HashMap::new();
            loads.insert(urls[0].to_string(), make_load_reqs_only(1, 0.125, 100.0));
            loads.insert(urls[1].to_string(), make_load_reqs_only(3, 0.125, 100.0));
            policy.update_loads(&loads);
            let idx = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .unwrap();
            seen[idx] += 1;
        }
        assert_eq!(seen[1], 0, "the deeper queue must never win: {seen:?}");
        assert!(
            seen[0] > 0 && seen[2] > 0,
            "the unreported worker ties with the best-known peer: {seen:?}"
        );
        // Two-way tie sampled uniformly: Binomial(150, 1/2), mean 75.
        assert!(
            seen[2] < 120,
            "the unreported worker must share, not take, the traffic: {seen:?}"
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
    fn a_report_releases_the_dispatches_made_up_to_its_sample_and_keeps_the_rest() {
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000")];
        let info = SelectWorkerInfo::default();
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.1, 100.0));
        policy.update_loads(&loads);

        // Two dispatches, then the engine is sampled, then one more.
        policy.select_min_expected_wait(&workers, &[0], &info, "test");
        policy.select_min_expected_wait(&workers, &[0], &info, "test");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let sampled_at = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(2));
        policy.select_min_expected_wait(&workers, &[0], &info, "test");
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 3072, 3));

        let mut sampled = make_load(0, 0.1, 100.0);
        sampled.sampled_at = Some(sampled_at);
        loads.insert("http://a:8000".to_string(), sampled.clone());
        policy.update_loads(&loads);
        assert_eq!(
            policy.load_state_for_test("http://a:8000"),
            (true, 1024, 1),
            "the dispatch after the sample stays as credit"
        );

        // The same record republished (the monitor's shared snapshot on a
        // pushed record from another worker) releases nothing more.
        policy.update_loads(&loads);
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 1024, 1));

        // A report without a sample instant resets, as every poll did before.
        loads.insert("http://a:8000".to_string(), make_load(0, 0.1, 100.0));
        policy.update_loads(&loads);
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 0, 0));
    }

    #[test]
    fn the_tally_keeps_a_bounded_history_for_a_worker_that_never_reports() {
        let mut dispatch = SincePollDispatch::default();
        let at = Instant::now();
        for _ in 0..(SINCE_POLL_DISPATCHES_KEPT + 100) {
            dispatch.record(at, 7);
        }
        assert_eq!(dispatch.dispatches.len(), SINCE_POLL_DISPATCHES_KEPT);
        assert_eq!(dispatch.requests as usize, SINCE_POLL_DISPATCHES_KEPT);
        assert_eq!(dispatch.tokens as usize, 7 * SINCE_POLL_DISPATCHES_KEPT);
        dispatch.release_through(at);
        assert_eq!((dispatch.tokens, dispatch.requests), (0, 0));
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
