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
    worker::{
        expected_wait::ExpectedWait,
        load_state::{LoadReceiver, LoadSnapshot},
        Worker,
    },
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

/// Since-report dispatch tally for one worker: the token-work and request
/// count of the dispatches no report has reflected yet, with each dispatch's
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
    /// reflects: those made at or before it. Idempotent: the dispatches are
    /// kept oldest first, so the same report read again, or an older sample
    /// read after a newer one, releases nothing more.
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
///   score_i = (queued_tokens_i + running_estimated_tokens_i + unaccounted_i · p̄
///              + inflight_tokens_i) / throughput_i
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
/// - `unaccounted` — requests this router holds on the worker that neither the
///   report nor the since-poll credit accounts for: dispatches still on their
///   way to an engine that has not seen them. A report releases the credit of
///   the dispatches made before its sample; while those sit in transit the
///   report reads emptier than the worker is, and the router would keep
///   piling on (see [`Self::unaccounted_requests`]). Each is priced at `p̄`,
///   like a dispatch whose size is unknown.
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
///   it since its last poll and the requests the router holds on it beyond
///   its report, have reached the cap is skipped. When every
///   candidate is at the cap the selection returns none, so the request falls
///   to the router's admission queue instead of deepening a backlog. Set it
///   below the engine's max batch size.
#[derive(Debug)]
pub struct LeastLoadPolicy {
    /// The worker monitor's load feed: the shared snapshot of every worker's
    /// latest report, read once per decision. `None` until the registry
    /// wires it; the fleet is dark until then (join-shortest-queue on the
    /// live in-flight counts).
    loads: RwLock<Option<LoadReceiver>>,
    /// Per-worker dispatch tally (keyed by worker URL): the dispatches no
    /// report reflects yet. A decision releases the dispatches made at or
    /// before the sample instant of the worker's current report before it
    /// reads the tally. Token-work feeds the score's in-flight term; the
    /// request count feeds the waiting-queue veto.
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
    /// The load feed's current snapshot; `None` until the registry wires the
    /// feed.
    snapshot: Option<&'a LoadSnapshot>,
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
            loads: RwLock::new(None),
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

    /// The feed's current snapshot: one `Arc` clone, no guard held afterwards.
    fn snapshot(&self) -> Option<Arc<LoadSnapshot>> {
        let feed = self
            .loads
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        feed.as_ref().map(|rx| rx.borrow().clone())
    }

    /// Bring a worker's tally up to date with its current report: the
    /// dispatches made at or before the report's sample are the engine's by
    /// now and leave the credit; the later ones stay.
    fn release_reported(
        snapshot: Option<&LoadSnapshot>,
        inflight: &mut HashMap<String, SincePollDispatch>,
        url: &str,
    ) {
        if let (Some((_, sampled_at)), Some(dispatch)) = (
            snapshot.and_then(|snapshot| snapshot.report(url)),
            inflight.get_mut(url),
        ) {
            dispatch.release_through(sampled_at);
        }
    }

    /// Test-only view of one worker's report presence and dispatch credit as
    /// the next decision reads them: `(has_load, tokens, requests)`.
    #[cfg(test)]
    pub(super) fn load_state_for_test(&self, url: &str) -> (bool, u64, u64) {
        let snapshot = self.snapshot();
        let snapshot = snapshot.as_deref();
        let mut inflight = self
            .inflight_tokens
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Self::release_reported(snapshot, &mut inflight, url);
        let dispatch = inflight.get(url);
        (
            Self::fresh_load(snapshot, url).is_some(),
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
            snapshot,
            inflight,
            nominal_throughput,
            fleet_has_loads,
            peer_baseline,
        } = *inputs;
        let url = worker.url();
        match Self::fresh_load(snapshot, url) {
            Some(load) => {
                let dispatch = inflight.get(url);
                let inflight_tokens = dispatch.map_or(0, |dispatch| dispatch.tokens);
                let since_poll = dispatch.map_or(0, |dispatch| dispatch.requests);
                let queued_tokens = self.queued_tokens(load)
                    + self.running_estimated_tokens(load)
                    + Self::unaccounted_requests(worker, load, since_poll) as f64
                        * f64::from(self.mean_prefill_tokens);
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

    /// The worker's current report, while the feed has one: a report the
    /// monitor pruned (a failed fetch, a replaced worker) is gone here too.
    fn fresh_load<'a>(
        snapshot: Option<&'a LoadSnapshot>,
        url: &str,
    ) -> Option<&'a WorkerLoadResponse> {
        snapshot.and_then(|snapshot| snapshot.get(url))
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
        snapshot: Option<&'a LoadSnapshot>,
        inflight: &'a HashMap<String, SincePollDispatch>,
    ) -> ScoreInputs<'a> {
        let (tp_sum, tp_count) = candidates
            .iter()
            .filter_map(|&i| Self::fresh_load(snapshot, workers[i].url()))
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
            .filter(|&&i| Self::fresh_load(snapshot, workers[i].url()).is_some())
            .count();
        let mut inputs = ScoreInputs {
            snapshot,
            inflight,
            nominal_throughput,
            fleet_has_loads: reporting > 0,
            peer_baseline: 0.0,
        };
        if reporting < candidates.len() {
            let best_known = candidates
                .iter()
                .filter(|&&i| Self::fresh_load(snapshot, workers[i].url()).is_some())
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

    /// Requests the router holds on `worker` beyond what its report and the
    /// since-poll credit account for: work in transit that the engine has not
    /// seen yet.
    ///
    /// A report is sampled at the engine, and the credit for the dispatches
    /// made before its sample is released on the assumption that the engine
    /// had them by then. Under a backlog between the router and the engine's
    /// queue (a saturated router, a slow servicer) that assumption fails: the
    /// report keeps saying "nearly idle" while the router piles requests on
    /// the worker, every fresh sample releases the credit for dispatches the
    /// engine never saw, and the worker reads as the emptiest of the fleet
    /// until it holds most of it (a fleet sent every new request to three of
    /// its workers for over a minute this way while the rest sat idle). The
    /// router's own in-flight count is the floor: a report cannot claim fewer
    /// requests than the router has outstanding on the worker.
    fn unaccounted_requests(
        worker: &Arc<dyn Worker>,
        load: &WorkerLoadResponse,
        since_poll_requests: u64,
    ) -> u64 {
        let reported =
            u64::try_from(load.total_running_reqs() + load.total_waiting_reqs()).unwrap_or(0);
        (worker.load() as u64).saturating_sub(reported.saturating_add(since_poll_requests))
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
    ///
    /// One decision is one critical section over the dispatch tally: each
    /// candidate's credit is brought up to date with its current report, the
    /// veto and the scores read the result, and the winner is credited, all
    /// under the one guard, so no decision pairs a report with a credit the
    /// report already reflects or a new dispatch with an old report.
    pub(super) fn select_min_expected_wait(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        info: &SelectWorkerInfo,
        policy: &'static str,
    ) -> Option<usize> {
        self.select_min_wait(workers, candidates, None, info, policy)
    }

    /// Argmin over `candidates` of the expected wait plus the work the
    /// request itself adds at each worker: `cached_share[i]` is the share of
    /// the prompt `candidates[i]` already holds in its cache (0 for a cold
    /// worker, 1 for a prompt cached whole), so the worker still has
    /// `request_tokens × (1 − share)` to prefill before the request's first
    /// token. That work is priced like the queued work ahead of it, at the
    /// fleet's nominal drain rate (a dark fleet, scored in requests, prices
    /// it in mean-prefill units). A holder is therefore left for a colder
    /// worker exactly when the holder's wait exceeds the other's by more
    /// than the time the held prefix saves. Equal totals go to the warmer
    /// worker, then to the draw. The winner is credited with the work it
    /// must do, not the whole prompt. The waiting-queue veto, the critical
    /// section and the tie draw are those of
    /// [`Self::select_min_expected_wait`].
    pub(super) fn select_min_total_wait(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        cached_share: &[f64],
        info: &SelectWorkerInfo,
        policy: &'static str,
    ) -> Option<usize> {
        debug_assert_eq!(candidates.len(), cached_share.len());
        self.select_min_wait(workers, candidates, Some(cached_share), info, policy)
    }

    /// The shared body of the two selections above: `cached_share`, when
    /// given, adds each candidate's own uncached work to its score and sets
    /// the winner's credit to that work, never under the mean prefill (the
    /// price a report puts on a running request), so a run of cached hits
    /// within one poll interval still moves the home's score.
    fn select_min_wait(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        cached_share: Option<&[f64]>,
        info: &SelectWorkerInfo,
        policy: &'static str,
    ) -> Option<usize> {
        let snapshot = self.snapshot();
        let snapshot = snapshot.as_deref();

        // Held across the decision so the in-flight estimate stays consistent
        // and the chosen worker can be credited before the guard is released.
        let mut inflight = self
            .inflight_tokens
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for &i in candidates {
            Self::release_reported(snapshot, &mut inflight, workers[i].url());
        }

        // Waiting-queue veto: drop candidates whose reported queue, plus
        // requests dispatched since their last report and the requests the
        // router holds on them beyond the report, has reached the cap.
        // Workers without a report stay eligible — there is no queue
        // evidence to veto on, and a dark fleet must keep routing. The
        // positions kept are into `candidates`, so each keeps its share.
        let capped: Vec<usize>;
        let positions: &[usize] = if self.max_waiting_requests == 0 {
            &[]
        } else {
            let cap = self.max_waiting_requests as u64;
            capped = (0..candidates.len())
                .filter(|&pos| {
                    let i = candidates[pos];
                    let url = workers[i].url();
                    match Self::fresh_load(snapshot, url) {
                        Some(load) => {
                            let since_poll =
                                inflight.get(url).map_or(0, |dispatch| dispatch.requests);
                            let unaccounted =
                                Self::unaccounted_requests(&workers[i], load, since_poll);
                            let waiting = load.total_waiting_reqs().max(0) as u64;
                            let eligible = waiting + since_poll + unaccounted < cap;
                            if cache_trace::enabled() {
                                cache_trace::gate(serde_json::json!({
                                    "source": "waiting_queue_cap", "worker": url,
                                    "waiting_requests": waiting, "since_poll_requests": since_poll,
                                    "unaccounted_requests": unaccounted,
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
        let scored: Vec<usize>;
        let (candidates, shares): (&[usize], Option<Vec<f64>>) = if self.max_waiting_requests == 0 {
            (candidates, cached_share.map(<[f64]>::to_vec))
        } else {
            scored = positions.iter().map(|&pos| candidates[pos]).collect();
            (
                &scored,
                cached_share.map(|shares| positions.iter().map(|&pos| shares[pos]).collect()),
            )
        };
        let (&first, rest) = candidates.split_first()?;

        let inputs = self.score_inputs(workers, candidates, snapshot, &inflight);

        // The request's own work at the candidate in position `pos`: the
        // whole prompt, or the uncached share of it when shares are given;
        // and that work in the score's unit (seconds at the nominal drain
        // rate; requests, in mean-prefill units, over a dark fleet).
        let request_work = self.request_tokens(info);
        let own_tokens = |pos: usize| -> u64 {
            match &shares {
                Some(shares) => {
                    (request_work as f64 * (1.0 - shares[pos].clamp(0.0, 1.0))).round() as u64
                }
                None => request_work,
            }
        };
        let own_wait = |pos: usize| -> f64 {
            if shares.is_none() {
                return 0.0;
            }
            let tokens = own_tokens(pos) as f64;
            if inputs.fleet_has_loads {
                tokens / inputs.nominal_throughput
            } else {
                tokens / f64::from(self.mean_prefill_tokens)
            }
        };
        let share_of = |pos: usize| shares.as_ref().map_or(0.0, |shares| shares[pos]);

        // Argmin with reservoir tie-breaking: workers within
        // `TIE_EPSILON_SECS` of the minimum (the common idle/homogeneous case
        // scores equal to the digit) are sampled uniformly instead of
        // first-index-wins, which herded ties onto one worker.
        let observe_score = |pos: usize, score: f64| {
            if cache_trace::enabled() {
                let idx = candidates[pos];
                let url = workers[idx].url();
                let load = Self::fresh_load(snapshot, url);
                cache_trace::score(serde_json::json!({
                    "source": "expected_wait", "worker": url, "score": score,
                    "score_unit": if inputs.fleet_has_loads { "seconds" } else { "requests" },
                    "queued_tokens": load.map(|load| self.queued_tokens(load)),
                    "running_estimated_tokens": load.map(|load| self.running_estimated_tokens(load)),
                    "kv_pressure": load.map(WorkerLoadResponse::effective_kv_pressure),
                    "resident_token_usage": load.map(WorkerLoadResponse::effective_token_usage),
                    "drain_rate": load.map(|load| self.drain_rate(load)),
                    "since_poll_tokens": inflight.get(url).map_or(0, |dispatch| dispatch.tokens),
                    "unaccounted_requests": load.map(|load| {
                        let since_poll = inflight.get(url).map_or(0, |dispatch| dispatch.requests);
                        Self::unaccounted_requests(&workers[idx], load, since_poll)
                    }),
                    "fresh_load": load.is_some(),
                    "cached_share": shares.as_ref().map(|shares| shares[pos]),
                    "own_work_tokens": shares.as_ref().map(|_| own_tokens(pos)),
                    "own_wait": shares.as_ref().map(|_| own_wait(pos)),
                }));
            }
        };
        let mut best_pos = 0usize;
        let mut best = first;
        let mut best_score = self.score(&workers[best], &inputs) + own_wait(0);
        observe_score(0, best_score);
        let mut tied = 1u32;
        for (offset, &idx) in rest.iter().enumerate() {
            let pos = offset + 1;
            let s = self.score(&workers[idx], &inputs) + own_wait(pos);
            observe_score(pos, s);
            if s < best_score - TIE_EPSILON_SECS {
                best_pos = pos;
                best = idx;
                best_score = s;
                tied = 1;
            } else if (s - best_score).abs() <= TIE_EPSILON_SECS {
                // Equal totals go to the warmer worker (the one with more of
                // the prompt cached: its affinity is kept for the turns to
                // come); between equally warm ones, keep each tying
                // candidate with probability 1/k so the final pick is
                // uniform over all ties without collecting them.
                let warmer = share_of(pos) - share_of(best_pos);
                if warmer > 0.0 {
                    best_pos = pos;
                    best = idx;
                    best_score = best_score.min(s);
                    tied = 1;
                } else if warmer == 0.0 {
                    tied += 1;
                    best_score = best_score.min(s);
                    if self.tie_draw(tied) == 0 {
                        best_pos = pos;
                        best = idx;
                    }
                }
            }
        }

        // In-flight correction: credit the chosen worker with this request's
        // token-work until its next poll refreshes the snapshot: the part it
        // must still prefill, when the shares are known, and never less than
        // the mean prefill, the price a report puts on a running request. A
        // hit cached whole would otherwise credit nothing, and every arrival
        // of one poll interval would land on the home before its next
        // report, the incast this correction exists to prevent.
        let credited_tokens = match &shares {
            Some(_) => own_tokens(best_pos).max(u64::from(self.mean_prefill_tokens)),
            None => own_tokens(best_pos),
        };
        inflight
            .entry(workers[best].url().to_string())
            .or_default()
            .record(Instant::now(), credited_tokens);
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

    fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        *self
            .loads
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = rx;
    }

    fn needs_backend_loads(&self) -> bool {
        true
    }

    fn remove_worker(&self, url: &str) {
        self.inflight_tokens
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(url);
    }

    fn reset(&self) {
        self.inflight_tokens
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
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
    use crate::worker::{load_state::TestFeed, BasicWorkerBuilder, WorkerType};

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
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut busy = make_load(0, 0.0, 0.0);
        busy.loads[0].num_running_reqs = 1;
        let loads = HashMap::from([
            (workers[0].url().to_string(), busy),
            (workers[1].url().to_string(), make_load(0, 0.1, 0.0)),
        ]);
        // Dispatch credit is gone once the scheduler reports the request.
        feed.publish(&loads);
        assert_eq!(policy.load_state_for_test(workers[0].url()), (true, 0, 0));
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn reclaimable_prefix_cache_does_not_make_an_idle_worker_busy() {
        let policy = LeastLoadPolicy::new();
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut idle = make_load(0, 1.0, 0.0);
        idle.loads[0].active_token_usage = Some(0.0);
        let mut busy = make_load(0, 0.5, 0.0);
        busy.loads[0].active_token_usage = Some(0.5);
        busy.loads[0].num_running_reqs = 1;
        feed.publish(&HashMap::from([
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
        let feed = TestFeed::attached_to(&policy);
        let workers: Vec<Arc<dyn Worker>> =
            (0..4).map(|i| mk(&format!("http://w{i}:8000"))).collect();
        let mut loads = HashMap::new();
        for (i, worker) in workers.iter().enumerate() {
            loads.insert(
                worker.url().to_string(),
                make_load(0, i as f64 * 1e-9, 100.0),
            );
        }
        feed.publish(&loads);
        let info = SelectWorkerInfo::default();
        let mut hits = [0usize; 4];
        for _ in 0..400 {
            hits[policy.select_worker(&workers, &info).unwrap()] += 1;
            // Release the winner's credit so every pick sees the same four scores.
            feed.publish(&loads);
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
            let feed = TestFeed::attached_to(&policy);
            let workers: Vec<Arc<dyn Worker>> = urls.iter().map(|u| mk(u)).collect();
            let mut loads = HashMap::new();
            for url in urls {
                loads.insert(url.to_string(), make_load(1000, 0.2, 100.0));
            }
            feed.publish(&loads);
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
            let feed = TestFeed::attached_to(&policy);
            let workers: Vec<Arc<dyn Worker>> = urls.iter().map(|u| mk(u)).collect();
            let mut loads = HashMap::new();
            loads.insert(urls[0].to_string(), make_load_reqs_only(1, 0.125, 100.0));
            loads.insert(urls[1].to_string(), make_load_reqs_only(3, 0.125, 100.0));
            feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
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
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
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
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            make_load_reqs_only(1, 0.0, 100.0),
        );
        loads.insert("http://b:8000".to_string(), make_load(400_000, 0.0, 100.0));
        feed.publish(&loads);
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
    fn unaccounted_requests_are_the_in_flight_beyond_the_report_and_the_credit() {
        let w = mk("http://a:8000");
        for _ in 0..12 {
            w.increment_load();
        }
        let mut load = make_load(0, 0.0, 100.0);
        load.loads[0].num_running_reqs = 5;
        load.loads[0].num_waiting_reqs = 2;
        assert_eq!(
            LeastLoadPolicy::unaccounted_requests(&w, &load, 3),
            2,
            "12 held, 7 reported, 3 credited since the sample"
        );
        assert_eq!(
            LeastLoadPolicy::unaccounted_requests(&w, &load, 10),
            0,
            "a report that covers the in-flight leaves nothing"
        );
    }

    #[test]
    fn a_report_that_omits_the_requests_the_router_holds_is_floored_by_them() {
        // Both workers report an empty engine, sampled now; the router holds
        // 40 requests on `a` that the report does not show (dispatched, not
        // yet seen by the engine, their credit released by the fresh sample).
        // `a` must read as 40 requests of work, not as an idle peer of `b`:
        // every pick goes to `b` until its own credit has grown to the same.
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 2000.0, 0).with_tie_break_seed(1);
        let feed = TestFeed::attached_to(&policy);
        let a = mk("http://a:8000");
        let b = mk("http://b:8000");
        for _ in 0..40 {
            a.increment_load();
        }
        let workers = vec![a, b];
        let mut idle = make_load(0, 0.0, 2000.0);
        idle.sampled_at = Some(Instant::now());
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), idle.clone());
        loads.insert("http://b:8000".to_string(), idle);
        feed.publish(&loads);
        let info = SelectWorkerInfo::default();
        for pick in 0..40 {
            assert_eq!(
                policy.select_worker(&workers, &info),
                Some(1),
                "pick {pick} went to the worker whose report hides its load"
            );
        }
    }

    #[test]
    fn a_report_that_accounts_for_the_routers_requests_adds_nothing() {
        // `a` holds 10 requests and reports them running; `b` holds 10 and
        // reports them waiting, at the mean prefill each. The floor adds
        // nothing to either, so the reports alone decide: `a`, whose work is
        // running rather than queued, ties with `b` on the estimate, and a
        // third worker that holds nothing wins outright.
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 2000.0, 0);
        let feed = TestFeed::attached_to(&policy);
        let a = mk("http://a:8000");
        let b = mk("http://b:8000");
        let c = mk("http://c:8000");
        for _ in 0..10 {
            a.increment_load();
            b.increment_load();
        }
        let workers = vec![a, b, c];
        let mut running = make_load(0, 0.0, 2000.0);
        running.loads[0].num_running_reqs = 10;
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), running);
        loads.insert(
            "http://b:8000".to_string(),
            make_load_reqs_only(10, 0.0, 2000.0),
        );
        loads.insert("http://c:8000".to_string(), make_load(0, 0.0, 2000.0));
        feed.publish(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(2)
        );
        let snapshot = LoadSnapshot::from_loads_for_test(loads.into_iter().collect());
        let inputs = ScoreInputs {
            snapshot: Some(&snapshot),
            inflight: &HashMap::new(),
            nominal_throughput: 2000.0,
            fleet_has_loads: true,
            peer_baseline: 0.0,
        };
        let score_a = policy.score(&workers[0], &inputs);
        let score_b = policy.score(&workers[1], &inputs);
        assert!(
            (score_a - score_b).abs() < TIE_EPSILON_SECS,
            "{score_a} vs {score_b}"
        );
        assert!((score_a - 10.0 * 1024.0 / 2000.0).abs() < 1e-9, "{score_a}");
    }

    #[test]
    fn waiting_queue_veto_counts_the_requests_the_router_holds_beyond_the_report() {
        // Cap 5: `a` reports 1 waiting and nothing running while the router
        // holds 6 on it; the 5 the report does not show count toward the
        // cap, so `a` is skipped and the pick goes to `b` although `b`
        // scores far worse.
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 100.0, 5);
        let feed = TestFeed::attached_to(&policy);
        let a = mk("http://a:8000");
        for _ in 0..6 {
            a.increment_load();
        }
        let workers = vec![a, mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            make_load_reqs_only(1, 0.0, 100.0),
        );
        loads.insert("http://b:8000".to_string(), make_load(400_000, 0.0, 100.0));
        feed.publish(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn the_floor_excludes_hidden_backlogs_without_tilting_the_healthy_fleet() {
        // Two of eight workers hold 16 requests each that their fresh reports
        // do not show (admission throttled: the requests wait in front of the
        // engine); the six healthy ones hold 2 each, reported as running.
        // Over 60 picks, each dispatched (the load counter moves as the
        // router's would) with no report in between, the two take nothing
        // and the six water-fill evenly: the floor steers off the hidden
        // backlogs, not onto a favourite.
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 2000.0, 0);
        let feed = TestFeed::attached_to(&policy);
        let workers: Vec<Arc<dyn Worker>> =
            (0..8).map(|i| mk(&format!("http://w{i}:8000"))).collect();
        let mut loads = HashMap::new();
        for (i, worker) in workers.iter().enumerate() {
            let hidden = i < 2;
            for _ in 0..if hidden { 16 } else { 2 } {
                worker.increment_load();
            }
            let mut load = make_load(0, 0.0, 2000.0);
            load.loads[0].num_running_reqs = if hidden { 0 } else { 2 };
            loads.insert(worker.url().to_string(), load);
        }
        feed.publish(&loads);
        let mut picks = [0usize; 8];
        for _ in 0..60 {
            let idx = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .unwrap();
            workers[idx].increment_load();
            picks[idx] += 1;
        }
        assert_eq!(&picks[..2], &[0, 0], "{picks:?}");
        let healthy = &picks[2..];
        let spread = healthy.iter().max().unwrap() - healthy.iter().min().unwrap();
        assert!(spread <= 1, "{picks:?}");
    }

    #[test]
    fn a_report_releases_the_dispatches_made_up_to_its_sample_and_keeps_the_rest() {
        let policy = LeastLoadPolicy::new();
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000")];
        let info = SelectWorkerInfo::default();
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.1, 100.0));
        feed.publish(&loads);

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
        feed.publish(&loads);
        assert_eq!(
            policy.load_state_for_test("http://a:8000"),
            (true, 1024, 1),
            "the dispatch after the sample stays as credit"
        );

        // The same record republished (the monitor's shared snapshot on a
        // pushed record from another worker) releases nothing more.
        feed.publish(&loads);
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 1024, 1));

        // A report without a sample instant is taken as sampled when it is
        // published: it releases everything dispatched before then.
        loads.insert("http://a:8000".to_string(), make_load(0, 0.1, 100.0));
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000")];
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            make_load_reqs_only(64, 0.0, 100.0),
        );
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
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
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
        let a = mk("http://a:8000");
        for _ in 0..2 {
            a.increment_load();
        }
        let workers = vec![a, mk("http://b:8000"), mk("http://c:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://b:8000".to_string(), make_load(1000, 0.2, 100.0));
        loads.insert("http://c:8000".to_string(), make_load(0, 0.2, 100_000.0));
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.2, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(2048, 0.2, 100.0));
        feed.publish(&loads);

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
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(8000, 0.2, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(1000, 0.2, 100.0));
        feed.publish(&loads);
        // a: 8000/100 = 80s ; b: 1000/100 = 10s -> pick b.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn total_wait_ranks_the_queue_plus_the_work_each_worker_still_has_to_do() {
        // Three workers at 100 tokens/s and equal KV; the prompt is 64
        // tokens: a holds none of it, b half, c all. Totals in seconds:
        // queue / 100 plus the uncached tokens / 100. The winner is
        // credited its uncached work, never under the mean prefill (4 here).
        let policy =
            LeastLoadPolicy::with_params(DEFAULT_KV_PRESSURE_WEIGHT, 4, DEFAULT_THROUGHPUT, 0);
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![
            mk("http://a:8000"),
            mk("http://b:8000"),
            mk("http://c:8000"),
        ];
        let tokens: Vec<u32> = (0..64).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        let shares = [0.0, 0.5, 1.0];
        // (queued tokens per worker) -> the pick and the tokens it is credited.
        for (queued, pick, credited) in [
            // a 0.64, b 0.30 + 0.32, c 0.50: c, with nothing to prefill,
            // credited the mean-prefill floor.
            ([0, 30, 50], 2, 4),
            // c at 0.70: b.
            ([0, 30, 70], 1, 32),
            // b at 0.60 + 0.32: a, with the whole prompt to do.
            ([0, 60, 70], 0, 64),
        ] {
            let loads = workers
                .iter()
                .zip(queued)
                .map(|(worker, queued)| (worker.url().to_string(), make_load(queued, 0.2, 100.0)))
                .collect();
            feed.publish(&loads);
            let selected =
                policy.select_min_total_wait(&workers, &[0, 1, 2], &shares, &info, "test");
            assert_eq!(selected, Some(pick), "queued {queued:?}");
            assert_eq!(
                policy.load_state_for_test(workers[pick].url()),
                (true, credited, 1),
                "credit, queued {queued:?}"
            );
        }
    }

    #[test]
    fn equal_total_waits_go_to_the_warmer_worker() {
        // a: a 32-token queue and the 32-token prompt cached whole; b: an
        // empty queue and none of it. Both total 0.32 s: a keeps the
        // request, no draw.
        let policy = LeastLoadPolicy::new();
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        feed.publish(&HashMap::from([
            (workers[0].url().to_string(), make_load(32, 0.2, 100.0)),
            (workers[1].url().to_string(), make_load(0, 0.2, 100.0)),
        ]));
        let tokens: Vec<u32> = (0..32).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        assert_eq!(
            policy.select_min_total_wait(&workers, &[0, 1], &[1.0, 0.0], &info, "test"),
            Some(0)
        );
        // The plain selection, which gives no credit for the cache, would
        // have found a's queue the deeper.
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
    }

    #[test]
    fn a_run_of_full_hits_leaves_the_home_before_the_next_report() {
        // a holds the 32-token prompt whole, b none of it, both queues
        // empty, no report between the picks. Each hit costs a nothing to
        // prefill, but is credited the mean prefill (16 here) like any
        // dispatch, so a's score climbs and the gate opens within the poll
        // interval: b is left with the prompt to prefill once a's lead
        // passes the 32 tokens the prefix saves, then a is preferred again.
        let policy =
            LeastLoadPolicy::with_params(DEFAULT_KV_PRESSURE_WEIGHT, 16, DEFAULT_THROUGHPUT, 0);
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        feed.publish(&HashMap::from([
            (workers[0].url().to_string(), make_load(0, 0.2, 100.0)),
            (workers[1].url().to_string(), make_load(0, 0.2, 100.0)),
        ]));
        let tokens: Vec<u32> = (0..32).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        let picks: Vec<usize> = (0..20)
            .map(|_| {
                let pick = policy
                    .select_min_total_wait(&workers, &[0, 1], &[1.0, 0.0], &info, "test")
                    .expect("two candidates");
                workers[pick].increment_load();
                pick
            })
            .collect();
        assert_eq!(picks[0], 0, "the home takes the first hit");
        assert_eq!(
            picks[..3],
            [0, 0, 0],
            "a credit of 16 per hit opens the gate only past the 32 tokens saved"
        );
        assert!(
            picks.contains(&1),
            "the home is left before the next report"
        );
        let to_home = picks.iter().filter(|&&pick| pick == 0).count();
        assert!(
            to_home > picks.len() - to_home,
            "and still takes most of the run"
        );
    }

    #[test]
    fn a_dark_fleet_prices_the_work_in_mean_prefill_units() {
        // No report anywhere: the score is the in-flight count, and the
        // uncached work counts tokens / mean_prefill_tokens (1024). a holds
        // the 512-token prompt whole, b none: idle, a wins (0 against 0.5);
        // with one request on a, b wins (1 against 0.5).
        let policy = LeastLoadPolicy::new();
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let tokens: Vec<u32> = (0..512).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        assert_eq!(
            policy.select_min_total_wait(&workers, &[0, 1], &[1.0, 0.0], &info, "test"),
            Some(0)
        );
        workers[0].increment_load();
        assert_eq!(
            policy.select_min_total_wait(&workers, &[0, 1], &[1.0, 0.0], &info, "test"),
            Some(1)
        );
    }

    #[test]
    fn queue_depth_without_token_count_still_ranks_workers() {
        // Both backends report a queue but no token count for it. Reading the
        // absent count as an empty queue scores them identically, so the pick
        // is a coin flip and a badly backlogged worker keeps drawing traffic.
        let policy = LeastLoadPolicy::new(); // p̄ = 1024
        let feed = TestFeed::attached_to(&policy);
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
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut a = make_load(500, 0.2, 100.0);
        a.loads[0].num_waiting_reqs = 200;
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), a);
        loads.insert("http://b:8000".to_string(), make_load(8000, 0.2, 100.0));
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
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
        feed.publish(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn throughput_normalization_prefers_faster_worker() {
        // Same queued tokens; the faster worker (higher throughput) drains sooner.
        let policy = LeastLoadPolicy::new();
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(5000, 0.2, 50.0));
        loads.insert("http://b:8000".to_string(), make_load(5000, 0.2, 500.0));
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(10000, 0.2, 0.0));
        loads.insert("http://b:8000".to_string(), make_load(1000, 0.2, 0.0));
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
        let a = mk("http://a:8000");
        let b = mk("http://b:8000");
        for _ in 0..5 {
            b.increment_load();
        }
        let workers = vec![a, b];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(4000, 0.0, 100.0));
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.98, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.0, 100.0));
        feed.publish(&loads);
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
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.1, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.1, 100.0));
        feed.publish(&loads);

        let info = SelectWorkerInfo::default(); // tokens unknown -> mean prefill
        let first = policy.select_worker(&workers, &info).unwrap();
        let second = policy.select_worker(&workers, &info).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn a_new_report_releases_the_credit_the_next_decision_reads() {
        let policy = LeastLoadPolicy::new();
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.1, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.1, 100.0));
        feed.publish(&loads);

        let info = SelectWorkerInfo::default();
        for _ in 0..4 {
            policy.select_worker(&workers, &info);
        }
        let credited: u64 = workers
            .iter()
            .map(|worker| policy.load_state_for_test(worker.url()).1)
            .sum();
        assert_eq!(credited, 4 * 1024);

        // A fresh report, sampled after the dispatches, releases them all.
        feed.publish(&loads);
        for worker in &workers {
            assert_eq!(policy.load_state_for_test(worker.url()), (true, 0, 0));
        }
    }

    #[test]
    fn reset_discards_the_in_flight_credit() {
        // pick1 -> a; its credit tips pick2 -> b; after reset the credit is
        // gone and pick3 returns to a. The reports are the feed's, not the
        // policy's to discard.
        let policy = LeastLoadPolicy::new();
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(1000, 0.2, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(2000, 0.2, 100.0));
        feed.publish(&loads);

        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
        assert_eq!(policy.select_worker(&workers, &info), Some(1));
        policy.reset();
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 0, 0));
        assert_eq!(policy.select_worker(&workers, &info), Some(0));
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
    fn remove_worker_prunes_its_tally() {
        let policy = LeastLoadPolicy::new();
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![mk("http://a:8000"), mk("http://b:8000")];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), make_load(0, 0.1, 100.0));
        loads.insert("http://b:8000".to_string(), make_load(0, 0.1, 100.0));
        feed.publish(&loads);
        let info = SelectWorkerInfo::default();
        let first = policy.select_worker(&workers, &info).unwrap();
        let second = policy.select_worker(&workers, &info).unwrap();
        assert_ne!(first, second);
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 1024, 1));

        // Removing a worker drops only its tally (no unbounded growth on
        // churn); its report is the monitor's to evict.
        policy.remove_worker("http://a:8000");
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 0, 0));
        assert_eq!(policy.load_state_for_test("http://b:8000"), (true, 1024, 1));
    }
}
