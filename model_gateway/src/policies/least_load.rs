use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex, RwLock},
    time::Instant,
};

use openai_protocol::worker::WorkerLoadResponse;
#[cfg(test)]
use rand::SeedableRng;
use rand::{rngs::StdRng, RngExt};
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
        Worker, WorkerType,
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

/// The candidate count up to which one write guard is held across the
/// decision. Above it the credit is read under the read guard, the scores
/// run with no guard held and the write guard covers the release and the
/// credit alone. Measured on 64 concurrent decisions over eight cores in
/// three sessions: at two and four candidates the split costs 18 to 23 %
/// of the decisions per second and one held guard, with the candidates'
/// scratch thread-local instead of shared under it, reads at main or
/// above; at eight the held guard ties main while the split reads ahead
/// in two sessions of three (by 23 and 52 %, alike in the first); from 16
/// candidates up the split returns 1.3 to 4.6 times. Five to seven are
/// unmeasured and sit on the shape that ties main at both measured
/// neighbours; the quiet-box measurement of candidates four to eight one
/// by one is tracked as item #350 and may move this down.
const HELD_GUARD_CANDIDATES: usize = 7;

/// Since-report dispatch tally for one worker: the token-work and request
/// count of the dispatches no report has reflected yet, with each dispatch's
/// instant, so a report that says when the engine was sampled releases
/// exactly the dispatches it saw and keeps the later ones as credit.
#[derive(Clone, Debug, Default)]
struct SincePollDispatch {
    tokens: u64,
    requests: u64,
    /// `(dispatched_at, tokens)` per dispatch, oldest first: the instant is
    /// read under the tally's write guard, so the pushes keep the clock's order.
    dispatches: VecDeque<(Instant, u64)>,
}

impl SincePollDispatch {
    fn record(&mut self, at: Instant, tokens: u64) {
        debug_assert!(
            self.dispatches.back().is_none_or(|&(last, _)| last <= at),
            "a dispatch recorded out of clock order"
        );
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

    /// The credit a decision reads, `(tokens, requests, release)`: the tally
    /// less the dispatches made at or before `sampled_at`, which a report
    /// sampled then already reflects, read without releasing them.
    /// `release` says whether any such dispatch is held, so the decision's
    /// write phase can pop them ([`Self::release_through`]) once it holds
    /// the guard. No sample instant (no report) reads the whole tally.
    fn credit_after(&self, sampled_at: Option<Instant>) -> (u64, u64, bool) {
        let (mut tokens, mut requests) = (self.tokens, self.requests);
        let mut release = false;
        if let Some(sampled_at) = sampled_at {
            for &(at, dispatched) in &self.dispatches {
                if at > sampled_at {
                    break;
                }
                tokens -= dispatched;
                requests -= 1;
                release = true;
            }
        }
        (tokens, requests, release)
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
///   piling on (see `Self::unaccounted_requests`). Each is priced at `p̄`,
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
/// - prefill workers: the sum of valid computed input rates across all DP
///   ranks, or the candidate fleet's nominal input rate when unavailable;
///   without any measured candidate, use configured `default_throughput`;
/// - zero/absent generation throughput (the backend reports no rate, or
///   reports one for a rank with nothing running, which is a stale gauge
///   rather than a rate; see `live_throughput`): the fleet's nominal rate,
///   the mean live rate among the candidates, or `default_throughput` when
///   none has one;
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
/// - `default_throughput` (default `2000` tok/s) — drain rate used when no
///   candidate reports a live `gen_throughput` or measured prefill capacity.
///   Set it to per-replica input tokens/s for prefill, output tokens/s otherwise;
///   it co-tunes with `kv_pressure_weight`.
/// - `mean_prefill_tokens` (p̄, default `1024`) — per-request token estimate for
///   the in-flight term when the request's token count is unknown at routing
///   (the HTTP path; ignored when tokens are known, i.e. gRPC).
#[derive(Debug)]
pub struct LeastLoadPolicy {
    /// The worker monitor's load feed: the shared snapshot of every worker's
    /// latest report, read once per decision. `None` until the registry
    /// wires it; the fleet is dark until then (join-shortest-queue on the
    /// live in-flight counts).
    loads: RwLock<Option<LoadReceiver>>,
    /// The per-worker dispatch tally, by worker URL: the dispatches no
    /// report reflects yet. A decision reads each candidate's credit as it
    /// stands after its current report's sample under the read guard,
    /// scores from what it resolved with no guard held, and takes the write
    /// guard for the release and the winner's credit alone, so decisions on
    /// other threads read meanwhile. Token-work feeds the score's in-flight
    /// term; the request count its floor on the router's own in-flight
    /// count.
    tally: RwLock<HashMap<String, SincePollDispatch>>,
    /// KV-pressure weight `λ_t` (seconds).
    kv_pressure_weight: f64,
    /// Mean prefill length (tokens) for estimating in-flight token-work when a
    /// request's token count is unknown at routing time.
    mean_prefill_tokens: u32,
    /// Fallback throughput (tokens/s) for the `/throughput` term when no
    /// candidate reports a live rate; input tokens/s for prefill.
    default_throughput: f64,
    /// Seeded source for the tie draw when set (tests reproduce a selection
    /// sequence with it); the thread's generator otherwise.
    tie_rng: Option<Mutex<StdRng>>,
}

thread_local! {
    /// The candidates of the decision in progress on this thread, each
    /// resolved once; kept across decisions, so a decision allocates
    /// nothing. Per thread rather than under the tally's guard, so a
    /// decision scores with no guard held.
    static CANDIDATES: RefCell<Vec<Candidate>> = const { RefCell::new(Vec::new()) };
}

/// One candidate as a decision reads it, resolved once: the facts of its
/// current report, if any, and its since-report credit less the dispatches
/// that report reflects. Plain numbers, so the decision scores from a
/// scratch slice instead of looking the worker up in the snapshot and the
/// tally at every step.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    /// The worker's position in the decision's slice.
    idx: usize,
    /// `None` without a fresh report: the feed is not wired, or it holds
    /// none for this worker (a report the monitor pruned after a failed
    /// fetch or a replacement is gone from the snapshot too).
    report: Option<ReportFacts>,
    /// Token-work and requests dispatched since the report's sample.
    since_poll_tokens: u64,
    since_poll_requests: u64,
    /// The report's sample instant, and whether the tally still holds
    /// dispatches made at or before it for the decision's write phase to
    /// release (a read under the held guard released them as it read).
    sampled_at: Option<Instant>,
    release: bool,
}

/// The credit of `url` as a read under the tally's read guard answers it:
/// the dispatches the report reflects leave the credit and stay in the
/// tally for the decision's write phase to release
/// ([`SincePollDispatch::credit_after`]).
fn credit_read(
    tally: &HashMap<String, SincePollDispatch>,
    url: &str,
    sampled_at: Option<Instant>,
) -> (u64, u64, bool) {
    tally
        .get(url)
        .map_or((0, 0, false), |dispatch| dispatch.credit_after(sampled_at))
}

/// The credit of `url` as the guard held across a small decision answers
/// it: the dispatches the report reflects are released as the credit is
/// read ([`SincePollDispatch::release_through`]), one lookup per candidate,
/// and nothing is left for the write phase.
fn credit_released(
    tally: &mut HashMap<String, SincePollDispatch>,
    url: &str,
    sampled_at: Option<Instant>,
) -> (u64, u64, bool) {
    tally.get_mut(url).map_or((0, 0, false), |dispatch| {
        if let Some(sampled_at) = sampled_at {
            dispatch.release_through(sampled_at);
        }
        (dispatch.tokens, dispatch.requests, false)
    })
}

/// What the expected-wait score reads of one report.
#[derive(Clone, Copy, Debug)]
struct ReportFacts {
    /// Waiting-queue token-work (see `LeastLoadPolicy::queued_tokens`).
    queued_tokens: f64,
    /// Running requests priced at the mean prefill.
    running_estimated_tokens: f64,
    /// Running plus waiting requests: what the report says the worker holds.
    reported_requests: u64,
    kv_pressure: f64,
    /// Resident token usage, for the trace record.
    token_usage: f64,
    /// Context tokens of the requests the worker holds, when every rank
    /// reports them (see `WorkerLoadResponse::total_resident_tokens`).
    resident_tokens: Option<f64>,
    /// The worker's live drain rate when it has one (see
    /// `LeastLoadPolicy::live_drain_rate`): the live output rate of a decode
    /// or regular worker, the measured input capacity of a prefill worker.
    live_rate: Option<f64>,
}

/// What one expected-wait score reads besides the candidate and its worker.
#[derive(Clone, Copy)]
struct ScoreInputs {
    nominal_throughput: f64,
    fleet_has_loads: bool,
    /// The best-known reporting peer's score: what a worker without a
    /// fresh report scores before its own in-flight.
    peer_baseline: f64,
}

/// What a decision minimizes.
#[derive(Clone, Copy, Debug)]
enum Objective {
    /// The expected wait (see [`LeastLoadPolicy::score`]).
    ExpectedWait,
    /// The resident work, each request priced at `request_tokens` (see
    /// [`LeastLoadPolicy::resident_score`]).
    ResidentWork { request_tokens: u64 },
}

/// What one resident-work score reads besides the candidate and its worker.
#[derive(Clone, Copy)]
struct ResidentInputs {
    request_tokens: f64,
    /// Whether any candidate reports resident tokens.
    fleet_has_resident: bool,
    /// The best-known reporting peer's score: what a worker without resident
    /// tokens scores before its own in-flight.
    peer_baseline: f64,
}

/// Generation rate a report can be scored against: the sum over ranks with
/// requests running.
///
/// An engine never clears `gen_throughput`: it is the rate of the last decode
/// interval, so an idle rank reports a residual (SGLang: the ~0.3 tok/s of
/// its last one-token health generation) or nothing at all. A dispatch
/// divided by a residual reads as thousands of seconds, so a burst followed
/// whichever worker happened to report the larger stale number until the
/// next poll. A rank with nothing running contributes no rate; a worker with
/// none is scored at the fleet's nominal rate, like one that never reports a
/// rate.
fn live_throughput(load: &WorkerLoadResponse) -> f64 {
    load.loads
        .iter()
        .filter(|rank| rank.num_running_reqs > 0)
        .map(|rank| rank.gen_throughput)
        .sum()
}

impl LeastLoadPolicy {
    pub fn new() -> Self {
        Self::with_params(
            DEFAULT_KV_PRESSURE_WEIGHT,
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
            loads: RwLock::new(None),
            tally: RwLock::new(HashMap::new()),
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
            tie_rng: None,
        }
    }

    /// Draw ties from a seeded generator instead of the thread's: the tests'
    /// determinism. Production draws from the thread's generator.
    #[cfg(test)]
    pub(crate) fn with_tie_break_seed(mut self, seed: u64) -> Self {
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
    pub(crate) fn params_for_test(&self) -> (f64, u32, f64) {
        (
            self.kv_pressure_weight,
            self.mean_prefill_tokens,
            self.default_throughput,
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

    /// One candidate as the decision reads it: one read of the snapshot and
    /// one of the tally through `read_credit`, which answers `(tokens, requests,
    /// release)` for the worker's url and its report's sample instant: the
    /// dispatches made at or before the sample are the engine's by now and
    /// leave the credit, the later ones stay. The two guard shapes answer it
    /// differently ([`credit_read`], [`credit_released`]).
    fn resolve(
        &self,
        idx: usize,
        worker: &Arc<dyn Worker>,
        snapshot: Option<&LoadSnapshot>,
        read_credit: &mut impl FnMut(&str, Option<Instant>) -> (u64, u64, bool),
    ) -> Candidate {
        let url = worker.url();
        let report = snapshot.and_then(|snapshot| snapshot.report(url));
        let sampled_at = report.map(|(_, sampled_at)| sampled_at);
        let (since_poll_tokens, since_poll_requests, release) = read_credit(url, sampled_at);
        Candidate {
            idx,
            report: report.map(|(load, _)| self.report_facts(worker, load)),
            since_poll_tokens,
            since_poll_requests,
            sampled_at,
            release,
        }
    }

    /// What the score reads of `worker`'s report.
    fn report_facts(&self, worker: &Arc<dyn Worker>, load: &WorkerLoadResponse) -> ReportFacts {
        ReportFacts {
            queued_tokens: self.queued_tokens(load),
            running_estimated_tokens: self.running_estimated_tokens(load),
            reported_requests: u64::try_from(load.total_running_reqs() + load.total_waiting_reqs())
                .unwrap_or(0),
            kv_pressure: load.effective_kv_pressure(),
            token_usage: load.effective_token_usage(),
            resident_tokens: load
                .total_resident_tokens()
                .map(|tokens| tokens.max(0) as f64),
            live_rate: Self::live_drain_rate(worker, load),
        }
    }

    /// Test-only: the candidates of a decision over `indices`, as the scorer
    /// resolves them against the feed's current snapshot.
    #[cfg(test)]
    fn resolve_for_test(&self, workers: &[Arc<dyn Worker>], indices: &[usize]) -> Vec<Candidate> {
        let snapshot = self.snapshot();
        let snapshot = snapshot.as_deref();
        let tally = self
            .tally
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut read_credit = |url: &str, sampled_at| credit_read(&tally, url, sampled_at);
        indices
            .iter()
            .map(|&idx| self.resolve(idx, &workers[idx], snapshot, &mut read_credit))
            .collect()
    }

    /// Test-only view of one worker's report presence and dispatch credit as
    /// the next decision reads them: `(has_load, tokens, requests)`.
    #[cfg(test)]
    pub(super) fn load_state_for_test(&self, url: &str) -> (bool, u64, u64) {
        let snapshot = self.snapshot();
        let report = snapshot
            .as_deref()
            .and_then(|snapshot| snapshot.report(url));
        let tally = self
            .tally
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (tokens, requests, _) =
            credit_read(&tally, url, report.map(|(_, sampled_at)| sampled_at));
        (report.is_some(), tokens, requests)
    }

    /// Test-only: the dispatches the tally still holds for `url`, released
    /// by a report or not.
    #[cfg(test)]
    fn held_dispatches_for_test(&self, url: &str) -> usize {
        self.tally
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(url)
            .map_or(0, |dispatch| dispatch.dispatches.len())
    }

    /// Expected-wait score for a resolved candidate (lower is better).
    ///
    /// `nominal_throughput` (a peer-derived mean) estimates drain rate for a
    /// worker missing a fresh snapshot or reporting no live rate;
    /// `fleet_has_loads` is false only when no
    /// worker reports at all, in which case we fall back to join-shortest-queue
    /// on the live in-flight count (which, unlike the since-poll estimate,
    /// reflects completions and so suits backends that never report loads).
    fn score(&self, worker: &Arc<dyn Worker>, candidate: &Candidate, inputs: &ScoreInputs) -> f64 {
        let ScoreInputs {
            nominal_throughput,
            fleet_has_loads,
            peer_baseline,
        } = *inputs;
        match candidate.report {
            Some(report) => {
                let queued_tokens = report.queued_tokens
                    + report.running_estimated_tokens
                    + Self::unaccounted_requests(
                        worker,
                        report.reported_requests,
                        candidate.since_poll_requests,
                    ) as f64
                        * f64::from(self.mean_prefill_tokens);
                ExpectedWait::new(
                    queued_tokens,
                    Self::drain_rate(report.live_rate, nominal_throughput),
                    report.kv_pressure,
                    self.kv_pressure_weight,
                )
                .seconds(candidate.since_poll_tokens)
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

    /// Generation throughput is a usable drain rate only for decode and regular
    /// workers, and only from ranks with requests running (see
    /// [`live_throughput`]). Prefill consumes input tokens, which the output
    /// rate cannot price; use complete, valid computed input capacity instead.
    fn live_drain_rate(worker: &Arc<dyn Worker>, load: &WorkerLoadResponse) -> Option<f64> {
        if worker.worker_type() == &WorkerType::Prefill {
            // Partial DP reports cannot represent aggregate engine capacity.
            if load.loads.is_empty() || load.loads.len() < load.dp_rank_count.max(0) as usize {
                return None;
            }
            let rate = load.loads.iter().try_fold(0.0, |sum, rank| {
                rank.prefill_throughput
                    .filter(|rate| rate.is_finite() && *rate > 0.0)
                    .map(|rate| sum + rate)
            })?;
            return (rate.is_finite() && rate > 0.0).then_some(rate);
        }
        let live = live_throughput(load);
        (live > 0.0).then_some(live)
    }

    /// The rate a worker's queued work drains at: its live rate when it has
    /// one (`live_rate`, the worker's [`Self::live_drain_rate`]: measured
    /// input capacity for prefill, the live output rate for the other roles),
    /// else `nominal_throughput`. No live rate (none reported, or only idle
    /// ranks) means the fleet's nominal rate, so an idle worker beside busy
    /// peers reads as fast as they are, not slower; with no live rate
    /// anywhere that is the configured default.
    fn drain_rate(live_rate: Option<f64>, nominal_throughput: f64) -> f64 {
        live_rate.unwrap_or(nominal_throughput)
    }

    /// The scoring inputs for one pass over the resolved `candidates`: the
    /// nominal drain rate (mean of the live rates, see [`live_throughput`])
    /// that stands in for a worker missing a fresh snapshot or a live rate;
    /// whether anyone reports at all, which
    /// separates a partial gap (estimate the missing worker at the nominal
    /// rate) from a dark fleet (join-shortest-queue on live in-flight); and
    /// the best-known reporting peer's score, which a worker without a report
    /// starts from: never better than a worker whose load is known, never
    /// starved by one. The baseline is computed only when some candidate
    /// lacks a report, so the common all-reporting case pays nothing for it.
    fn score_inputs(&self, workers: &[Arc<dyn Worker>], candidates: &[Candidate]) -> ScoreInputs {
        let (tp_sum, tp_count) = candidates
            .iter()
            .filter_map(|candidate| candidate.report.and_then(|report| report.live_rate))
            .fold((0.0, 0u32), |(s, n), t| (s + t, n + 1));
        let nominal_throughput = if tp_count > 0 && tp_sum.is_finite() {
            tp_sum / tp_count as f64
        } else {
            self.default_throughput
        };
        let reporting = candidates
            .iter()
            .filter(|candidate| candidate.report.is_some())
            .count();
        let mut inputs = ScoreInputs {
            nominal_throughput,
            fleet_has_loads: reporting > 0,
            peer_baseline: 0.0,
        };
        if reporting < candidates.len() {
            let best_known = candidates
                .iter()
                .filter(|candidate| candidate.report.is_some())
                .map(|candidate| self.score(&workers[candidate.idx], candidate, &inputs))
                .fold(f64::INFINITY, f64::min);
            // Nobody reports: the dark-fleet arm scores by live in-flight and
            // never reads the baseline; keep it neutral rather than infinite.
            if best_known.is_finite() {
                inputs.peer_baseline = best_known;
            }
        }
        inputs
    }

    /// Resident-work score for a resolved candidate (lower is better), in
    /// context tokens: the tokens the worker's report says it holds, the
    /// prompts dispatched to it since that report, and `request_tokens` for
    /// every request it holds. Requests the router holds beyond the report
    /// and the credit ([`Self::unaccounted_requests`]) are priced at the
    /// worker's mean context.
    ///
    /// A worker without resident tokens (no fresh report, or an engine that
    /// does not count them) scores as the best-known reporting peer plus its
    /// live in-flight, each request at the mean prefill and `request_tokens`;
    /// with no candidate reporting them, the live in-flight count decides.
    fn resident_score(
        &self,
        worker: &Arc<dyn Worker>,
        candidate: &Candidate,
        inputs: &ResidentInputs,
    ) -> f64 {
        let mean_prefill = f64::from(self.mean_prefill_tokens);
        let reported = candidate.report.and_then(|report| {
            report
                .resident_tokens
                .map(|resident| (report.reported_requests, resident))
        });
        match reported {
            Some((reported_requests, resident)) => {
                let unaccounted = Self::unaccounted_requests(
                    worker,
                    reported_requests,
                    candidate.since_poll_requests,
                );
                let mean_context = if reported_requests > 0 {
                    resident / reported_requests as f64
                } else {
                    mean_prefill
                };
                let requests = reported_requests + candidate.since_poll_requests + unaccounted;
                resident
                    + candidate.since_poll_tokens as f64
                    + unaccounted as f64 * mean_context
                    + inputs.request_tokens * requests as f64
            }
            None if inputs.fleet_has_resident => {
                inputs.peer_baseline + worker.load() as f64 * (mean_prefill + inputs.request_tokens)
            }
            None => worker.load() as f64,
        }
    }

    /// The resident-work inputs for one pass over the resolved `candidates`:
    /// whether any reports resident tokens, and the best such score.
    fn resident_inputs(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[Candidate],
        request_tokens: u64,
    ) -> ResidentInputs {
        let mut inputs = ResidentInputs {
            request_tokens: request_tokens as f64,
            fleet_has_resident: false,
            peer_baseline: 0.0,
        };
        let best_known = candidates
            .iter()
            .filter(|candidate| {
                candidate
                    .report
                    .is_some_and(|report| report.resident_tokens.is_some())
            })
            .map(|candidate| self.resident_score(&workers[candidate.idx], candidate, &inputs))
            .fold(f64::INFINITY, f64::min);
        if best_known.is_finite() {
            inputs.fleet_has_resident = true;
            inputs.peer_baseline = best_known;
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
    /// `reported_requests` is the report's running plus waiting count.
    fn unaccounted_requests(
        worker: &Arc<dyn Worker>,
        reported_requests: u64,
        since_poll_requests: u64,
    ) -> u64 {
        (worker.load() as u64).saturating_sub(reported_requests.saturating_add(since_poll_requests))
    }

    /// Token-work the request being routed adds to the chosen worker's
    /// in-flight estimate: its token count if known, else the mean prefill.
    fn request_tokens(&self, info: &SelectWorkerInfo) -> u64 {
        info.tokens
            .map(|t| t.len() as u64)
            .unwrap_or(self.mean_prefill_tokens as u64)
    }

    /// The mean prefill (tokens): the work a request of unknown size, or a
    /// reported running request, is priced at.
    pub(super) fn mean_prefill_tokens(&self) -> u32 {
        self.mean_prefill_tokens
    }

    /// Argmin of the expected-wait score over `candidates` (indices into
    /// `workers`), crediting the winner's in-flight estimate. The nominal
    /// throughput and dark-fleet fallback are scoped to `candidates`, so a
    /// caller scoring a sampled subset (power-of-two) gets a self-consistent
    /// comparison. `policy` labels the selection log line.
    ///
    /// Up to [`HELD_GUARD_CANDIDATES`] candidates one write guard is held
    /// across the decision. Above it the decision reads under the tally's
    /// read guard, scores with no guard held and writes under the write
    /// guard: each candidate's credit is
    /// read as it stands after its current report's sample, the veto and
    /// the scores read the result, then the dispatches those reports
    /// reflect are released and the winner is credited. No decision pairs a
    /// report with a credit the report already reflects, or a new dispatch
    /// with an old report. Decisions on other threads read between one
    /// decision's read and its write, so every decision in flight at once
    /// can pick the least-loaded worker before any of their credits lands:
    /// a herd of at most one decision per runtime thread, inside one
    /// scoring window; every credit lands.
    pub(super) fn select_min_expected_wait(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        info: &SelectWorkerInfo,
        policy: &'static str,
    ) -> Option<usize> {
        self.select_min_expected_wait_credited(workers, candidates, info, policy, None)
    }

    /// [`Self::select_min_expected_wait`], crediting the winner with
    /// `credit(winner)` tokens of work instead of the request's whole prompt
    /// ([`Self::request_tokens`]) when `credit` is given: a caller that knows
    /// how much of the prompt the winner already holds in its cache credits
    /// the work left to prefill.
    pub(super) fn select_min_expected_wait_credited(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        info: &SelectWorkerInfo,
        policy: &'static str,
        credit: Option<&dyn Fn(usize) -> u64>,
    ) -> Option<usize> {
        CANDIDATES.with(|scratch| {
            self.decide(
                workers,
                candidates,
                info,
                policy,
                Objective::ExpectedWait,
                credit,
                &mut scratch.borrow_mut(),
            )
        })
    }

    /// Argmin of the resident-work score ([`Self::resident_score`]) over
    /// `candidates`, with each request priced at `request_tokens` context
    /// tokens, crediting the winner with the request's prompt. The decision,
    /// its tie-break and its credit are [`Self::select_min_expected_wait`]'s.
    pub(super) fn select_min_resident_work(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        info: &SelectWorkerInfo,
        policy: &'static str,
        request_tokens: u64,
    ) -> Option<usize> {
        CANDIDATES.with(|scratch| {
            self.decide(
                workers,
                candidates,
                info,
                policy,
                Objective::ResidentWork { request_tokens },
                None,
                &mut scratch.borrow_mut(),
            )
        })
    }

    /// The argmin of `objective` over `candidates`, crediting the winner (see
    /// [`Self::select_min_expected_wait_credited`]), with `resolved` this
    /// thread's scratch for the candidates it reads.
    #[expect(clippy::too_many_arguments)]
    fn decide(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        info: &SelectWorkerInfo,
        policy: &'static str,
        objective: Objective,
        credit: Option<&dyn Fn(usize) -> u64>,
        resolved: &mut Vec<Candidate>,
    ) -> Option<usize> {
        let snapshot = self.snapshot();
        let snapshot = snapshot.as_deref();

        // Up to `HELD_GUARD_CANDIDATES` candidates one write guard is held
        // across the decision: each candidate is read under it with the
        // dispatches its report reflects released as they are read (one
        // lookup per candidate), the scoring is short, and a second guard
        // handoff costs more than it frees. Above it each candidate is read
        // once under the read guard, its report and its credit after the
        // report's sample, every pass below scores from this with the guard
        // released, and the write guard covers the release and the credit
        // alone.
        let mut held = (candidates.len() <= HELD_GUARD_CANDIDATES).then(|| {
            self.tally
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        });
        resolved.clear();
        match held.as_deref_mut() {
            Some(tally) => {
                let mut read_credit =
                    |url: &str, sampled_at| credit_released(tally, url, sampled_at);
                resolved.extend(
                    candidates
                        .iter()
                        .map(|&idx| self.resolve(idx, &workers[idx], snapshot, &mut read_credit)),
                );
            }
            None => {
                let tally = self
                    .tally
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let mut read_credit = |url: &str, sampled_at| credit_read(&tally, url, sampled_at);
                resolved.extend(
                    candidates
                        .iter()
                        .map(|&idx| self.resolve(idx, &workers[idx], snapshot, &mut read_credit)),
                );
            }
        }

        let (first, rest) = resolved.split_first()?;

        let inputs = self.score_inputs(workers, resolved);
        let resident = match objective {
            Objective::ExpectedWait => None,
            Objective::ResidentWork { request_tokens } => {
                Some(self.resident_inputs(workers, resolved, request_tokens))
            }
        };
        let score_of = |candidate: &Candidate| match &resident {
            Some(resident) => self.resident_score(&workers[candidate.idx], candidate, resident),
            None => self.score(&workers[candidate.idx], candidate, &inputs),
        };

        // Argmin with reservoir tie-breaking: workers within
        // `TIE_EPSILON_SECS` of the minimum (the common idle/homogeneous case
        // scores equal to the digit) are sampled uniformly instead of
        // first-index-wins, which herded ties onto one worker.
        let observe_score = |candidate: &Candidate, score: f64| {
            if cache_trace::enabled() {
                let worker = &workers[candidate.idx];
                let report = candidate.report;
                if let Some(resident) = &resident {
                    cache_trace::score(serde_json::json!({
                        "source": "resident_work", "worker": worker.url(), "score": score,
                        "score_unit": if resident.fleet_has_resident { "tokens" } else { "requests" },
                        "resident_tokens": report.and_then(|report| report.resident_tokens),
                        "reported_requests": report.map(|report| report.reported_requests),
                        "since_poll_tokens": candidate.since_poll_tokens,
                        "since_poll_requests": candidate.since_poll_requests,
                        "unaccounted_requests": report.map(|report| {
                            Self::unaccounted_requests(
                                worker,
                                report.reported_requests,
                                candidate.since_poll_requests,
                            )
                        }),
                        "fresh_load": report.is_some(),
                    }));
                    return;
                }
                cache_trace::score(serde_json::json!({
                    "source": "expected_wait", "worker": worker.url(), "score": score,
                    "score_unit": if inputs.fleet_has_loads { "seconds" } else { "requests" },
                    "queued_tokens": report.map(|report| report.queued_tokens),
                    "running_estimated_tokens": report.map(|report| report.running_estimated_tokens),
                    "kv_pressure": report.map(|report| report.kv_pressure),
                    "resident_token_usage": report.map(|report| report.token_usage),
                    "drain_rate": report.map(|report| {
                        Self::drain_rate(report.live_rate, inputs.nominal_throughput)
                    }),
                    "since_poll_tokens": candidate.since_poll_tokens,
                    "unaccounted_requests": report.map(|report| {
                        Self::unaccounted_requests(
                            worker,
                            report.reported_requests,
                            candidate.since_poll_requests,
                        )
                    }),
                    "fresh_load": report.is_some(),
                }));
            }
        };
        let mut best = first;
        let mut best_score = score_of(best);
        observe_score(best, best_score);
        let mut tied = 1u32;
        for candidate in rest {
            let s = score_of(candidate);
            observe_score(candidate, s);
            if s < best_score - TIE_EPSILON_SECS {
                best = candidate;
                best_score = s;
                tied = 1;
            } else if (s - best_score).abs() <= TIE_EPSILON_SECS {
                // Keep each tying candidate with probability 1/k so the final
                // pick is uniform over all ties without collecting them.
                tied += 1;
                best_score = best_score.min(s);
                if self.tie_draw(tied) == 0 {
                    best = candidate;
                }
            }
        }
        let best = best.idx;

        // In-flight correction: credit the chosen worker with this request's
        // token-work until its next poll refreshes the snapshot. Above
        // `HELD_GUARD_CANDIDATES` the write guard taken here covers the
        // release of the dispatches the reports read above reflect, and the
        // credit; the guard held across a smaller decision released as it
        // read, so only the credit is left under it. The key is allocated
        // the first time a worker is dispatched to, never after.
        let req_tokens = credit.map_or_else(|| self.request_tokens(info), |credit| credit(best));
        let mut tally = match held.take() {
            Some(guard) => guard,
            None => self
                .tally
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        };
        // Read under the guard: the deque's order is the clock's order only when
        // no decision records an earlier instant after a later one landed.
        let dispatched_at = Instant::now();
        for candidate in resolved.iter().filter(|candidate| candidate.release) {
            if let (Some(dispatch), Some(sampled_at)) = (
                tally.get_mut(workers[candidate.idx].url()),
                candidate.sampled_at,
            ) {
                dispatch.release_through(sampled_at);
            }
        }
        match tally.get_mut(workers[best].url()) {
            Some(dispatch) => dispatch.record(dispatched_at, req_tokens),
            None => {
                let mut dispatch = SincePollDispatch::default();
                dispatch.record(dispatched_at, req_tokens);
                tally.insert(workers[best].url().to_string(), dispatch);
            }
        }
        drop(tally);

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
        let healthy = get_healthy_worker_indices(workers, info);
        if healthy.is_empty() {
            return None;
        }
        if healthy.len() == 1 {
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
        self.tally
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(url);
    }

    fn reset(&self) {
        self.tally
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
    use openai_protocol::worker::SchedulerLoadSnapshot;

    use super::*;
    use crate::{
        policies::test_support::{fed_pair, make_worker},
        worker::{
            load_state::{one_rank_load, one_rank_load_reqs_only, TestFeed},
            test_support::{no_health_check, worker_of},
            BasicWorkerBuilder, WorkerType,
        },
    };

    /// Invalid or incomplete rank capacity falls back to the fleet input rate.
    #[test]
    fn prefill_uses_only_valid_complete_input_capacity() {
        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://prefill:1")
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        let drain = |load: &WorkerLoadResponse| {
            LeastLoadPolicy::drain_rate(LeastLoadPolicy::live_drain_rate(&prefill, load), 9_000.0)
        };
        let mut load = one_rank_load(10_000, 0.0, 0.5);
        // Input capacity remains usable between batches, with no running requests.
        load.loads[0].num_running_reqs = 0;
        for rate in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            load.loads[0].prefill_throughput = rate;
            assert_eq!(drain(&load), 9_000.0);
        }
        load.loads[0].prefill_throughput = Some(5_000.0);
        assert_eq!(drain(&load), 5_000.0);
        load.dp_rank_count = 2;
        assert_eq!(drain(&load), 9_000.0);
        let mut second = load.loads[0].clone();
        second.dp_rank = 1;
        second.prefill_throughput = Some(7_000.0);
        load.loads.push(second);
        assert_eq!(drain(&load), 12_000.0);
        load.loads[1].prefill_throughput = None;
        assert_eq!(drain(&load), 9_000.0);
        for rank in &mut load.loads {
            rank.prefill_throughput = Some(f64::MAX);
        }
        assert_eq!(drain(&load), 9_000.0);
        load.loads.clear();
        assert_eq!(drain(&load), 9_000.0);
    }

    /// Finite per-worker rates can overflow when summed; unmeasured peers must
    /// still price their queued work at a finite, positive fallback rate.
    #[test]
    fn prefill_fleet_rate_overflow_uses_configured_capacity() {
        let policy = LeastLoadPolicy::with_params(0.15, 1024, 4_000.0);
        let feed = TestFeed::attached_to(&policy);
        let workers: Vec<Arc<dyn Worker>> = ["http://p1:1", "http://p2:2", "http://p3:3"]
            .into_iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(url)
                        .worker_type(WorkerType::Prefill)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        for (measured, expected) in [(5_000.0, 5_000.0), (f64::MAX, 4_000.0)] {
            let loads = workers
                .iter()
                .enumerate()
                .map(|(index, worker)| {
                    let mut load = one_rank_load(1_000, 0.0, 0.0);
                    load.loads[0].prefill_throughput = (index < 2).then_some(measured);
                    (worker.url().to_string(), load)
                })
                .collect();
            feed.publish(&loads);
            let candidates = policy.resolve_for_test(&workers, &[0, 1, 2]);
            let inputs = policy.score_inputs(&workers, &candidates);
            assert_eq!(inputs.nominal_throughput, expected);
            assert_eq!(
                policy.score(&workers[2], &candidates[2], &inputs),
                2_024.0 / expected
            );
        }
    }

    /// A mixed prefill pool, each worker's load read through its own
    /// engine's `GetLoads` conversion: a measured worker is priced on its
    /// own rate whichever engine reported it, so the faster one wins with
    /// four times the queued work; a worker whose conversion carries no
    /// measurement is priced at the fleet's measured mean, not at the
    /// configured default (which would hand the pick to its slower peer).
    #[test]
    fn a_mixed_fleet_prices_measured_workers_alike_and_unmeasured_ones_at_the_mean() {
        use smg_grpc_client::{sglang_proto, tokenspeed_proto};
        let report = |snapshot: SchedulerLoadSnapshot| WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![snapshot],
            ..Default::default()
        };
        let measured_fast = report(SchedulerLoadSnapshot::from(sglang_proto::SchedulerLoad {
            num_running_reqs: 1,
            num_waiting_uncached_tokens: 8_000,
            prefill_throughput: Some(40_000.0),
            ..Default::default()
        }));
        let measured_slow = report(SchedulerLoadSnapshot::from(
            tokenspeed_proto::SchedulerLoad {
                num_running_reqs: 1,
                num_waiting_uncached_tokens: 2_000,
                prefill_throughput: Some(4_000.0),
                ..Default::default()
            },
        ));
        let unmeasured = report(SchedulerLoadSnapshot::from(
            tokenspeed_proto::SchedulerLoad {
                num_running_reqs: 1,
                num_waiting_uncached_tokens: 2_000,
                ..Default::default()
            },
        ));
        for (peer, peer_score, expected) in [
            (measured_slow, 3_024.0 / 4_000.0, 0),
            (unmeasured, 3_024.0 / 40_000.0, 1),
        ] {
            let policy = LeastLoadPolicy::with_params(0.15, 1024, 1_000.0);
            let feed = TestFeed::attached_to(&policy);
            let workers: Vec<Arc<dyn Worker>> = ["http://p1:1", "http://p2:2"]
                .into_iter()
                .map(|url| worker_of(url, WorkerType::Prefill))
                .collect();
            feed.publish(&HashMap::from([
                (workers[0].url().to_string(), measured_fast.clone()),
                (workers[1].url().to_string(), peer),
            ]));
            let candidates = policy.resolve_for_test(&workers, &[0, 1]);
            let inputs = policy.score_inputs(&workers, &candidates);
            assert_eq!(
                policy.score(&workers[0], &candidates[0], &inputs),
                9_024.0 / 40_000.0
            );
            assert_eq!(
                policy.score(&workers[1], &candidates[1], &inputs),
                peer_score
            );
            assert_eq!(
                policy.select_worker(&workers, &SelectWorkerInfo::default()),
                Some(expected)
            );
        }
    }

    /// Prefill scores and missing-peer estimates must not track output-token rates.
    #[test]
    fn prefill_ignores_generation_rate() {
        let policy = LeastLoadPolicy::with_params(0.15, 1024, 4_000.0);
        let feed = TestFeed::attached_to(&policy);
        let workers: Vec<Arc<dyn Worker>> = ["http://prefill:1", "http://missing:2"]
            .into_iter()
            .map(|url| worker_of(url, WorkerType::Prefill))
            .collect();
        workers[1].increment_load();
        for generated in [0.0, 0.5, 10_000.0] {
            feed.publish(&HashMap::from([(
                workers[0].url().to_string(),
                one_rank_load(10_000, 0.0, generated),
            )]));
            let candidates = policy.resolve_for_test(&workers, &[0, 1]);
            let inputs = policy.score_inputs(&workers, &candidates);
            assert_eq!(inputs.nominal_throughput, 4_000.0);
            // 10,000 queued plus the fixture's one running request at p̄.
            let prefill = (10_000.0 + 1024.0) / 4_000.0;
            assert_eq!(policy.score(&workers[0], &candidates[0], &inputs), prefill);
            assert_eq!(
                policy.score(&workers[1], &candidates[1], &inputs),
                prefill + 1024.0 / 4_000.0
            );
        }
    }

    /// Decode and regular workers retain live generation rates and zero-rate fallback.
    #[test]
    fn generation_rate_still_prices_decode_and_regular() {
        let policy = LeastLoadPolicy::new();
        let feed = TestFeed::attached_to(&policy);
        for role in [WorkerType::Decode, WorkerType::Regular] {
            let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
                BasicWorkerBuilder::new("http://worker:1")
                    .worker_type(role)
                    .build(),
            )];
            for generated in [0.0, 0.5, 10_000.0] {
                let mut load = one_rank_load(10_000, 0.0, generated);
                load.loads[0].prefill_throughput = Some(99_000.0);
                let expected = if generated > 0.0 {
                    generated
                } else {
                    DEFAULT_THROUGHPUT
                };
                assert_eq!(
                    LeastLoadPolicy::drain_rate(
                        LeastLoadPolicy::live_drain_rate(&workers[0], &load),
                        DEFAULT_THROUGHPUT
                    ),
                    expected
                );
                feed.publish(&HashMap::from([(workers[0].url().to_string(), load)]));
                let candidates = policy.resolve_for_test(&workers, &[0]);
                let inputs = policy.score_inputs(&workers, &candidates);
                assert_eq!(inputs.nominal_throughput, expected);
                // 10,000 queued plus the fixture's one running request at p̄.
                assert_eq!(
                    policy.score(&workers[0], &candidates[0], &inputs),
                    (10_000.0 + 1024.0) / expected
                );
            }
        }
    }

    #[test]
    fn running_work_survives_the_poll_credit_reset() {
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
        let mut busy = one_rank_load(0, 0.0, 0.0);
        busy.loads[0].num_running_reqs = 1;
        let mut idle = one_rank_load(0, 0.1, 0.0);
        idle.loads[0].num_running_reqs = 0;
        let loads = HashMap::from([
            (workers[0].url().to_string(), busy),
            (workers[1].url().to_string(), idle),
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
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
        let mut idle = one_rank_load(0, 1.0, 0.0);
        idle.loads[0].active_token_usage = Some(0.0);
        let mut busy = one_rank_load(0, 0.5, 0.0);
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
        let workers: Vec<Arc<dyn Worker>> = (0..128)
            .map(|i| make_worker(&format!("http://w{i}:8000")))
            .collect();
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
        let workers: Vec<Arc<dyn Worker>> = (0..128)
            .map(|i| make_worker(&format!("http://w{i}:8000")))
            .collect();
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
        let workers: Vec<Arc<dyn Worker>> = (0..4)
            .map(|i| make_worker(&format!("http://w{i}:8000")))
            .collect();
        let mut loads = HashMap::new();
        for (i, worker) in workers.iter().enumerate() {
            loads.insert(
                worker.url().to_string(),
                one_rank_load(0, i as f64 * 1e-9, 100.0),
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
            let workers: Vec<Arc<dyn Worker>> = urls.iter().map(|u| make_worker(u)).collect();
            let mut loads = HashMap::new();
            for url in urls {
                loads.insert(url.to_string(), one_rank_load(1000, 0.2, 100.0));
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
            let workers: Vec<Arc<dyn Worker>> = urls.iter().map(|u| make_worker(u)).collect();
            let mut loads = HashMap::new();
            loads.insert(
                urls[0].to_string(),
                one_rank_load_reqs_only(1, 0.125, 100.0),
            );
            loads.insert(
                urls[1].to_string(),
                one_rank_load_reqs_only(3, 0.125, 100.0),
            );
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
        let a = make_worker("http://a:8000");
        let b = make_worker("http://b:8000");
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
    fn unaccounted_requests_are_the_in_flight_beyond_the_report_and_the_credit() {
        let w = make_worker("http://a:8000");
        for _ in 0..12 {
            w.increment_load();
        }
        let mut load = one_rank_load(0, 0.0, 100.0);
        load.loads[0].num_running_reqs = 5;
        load.loads[0].num_waiting_reqs = 2;
        let reported = LeastLoadPolicy::new()
            .report_facts(&w, &load)
            .reported_requests;
        assert_eq!(reported, 7, "running plus waiting");
        assert_eq!(
            LeastLoadPolicy::unaccounted_requests(&w, reported, 3),
            2,
            "12 held, 7 reported, 3 credited since the sample"
        );
        assert_eq!(
            LeastLoadPolicy::unaccounted_requests(&w, reported, 10),
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
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 2000.0).with_tie_break_seed(1);
        let feed = TestFeed::attached_to(&policy);
        let a = make_worker("http://a:8000");
        let b = make_worker("http://b:8000");
        for _ in 0..40 {
            a.increment_load();
        }
        let workers = vec![a, b];
        let mut idle = one_rank_load(0, 0.0, 2000.0);
        idle.loads[0].num_running_reqs = 0;
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
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 2000.0);
        let feed = TestFeed::attached_to(&policy);
        let a = make_worker("http://a:8000");
        let b = make_worker("http://b:8000");
        let c = make_worker("http://c:8000");
        for _ in 0..10 {
            a.increment_load();
            b.increment_load();
        }
        let workers = vec![a, b, c];
        let mut running = one_rank_load(0, 0.0, 2000.0);
        running.loads[0].num_running_reqs = 10;
        let mut waiting = one_rank_load_reqs_only(10, 0.0, 2000.0);
        waiting.loads[0].num_running_reqs = 0;
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), running);
        loads.insert("http://b:8000".to_string(), waiting);
        loads.insert("http://c:8000".to_string(), one_rank_load(0, 0.0, 2000.0));
        feed.publish(&loads);
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(2)
        );
        let inputs = ScoreInputs {
            nominal_throughput: 2000.0,
            fleet_has_loads: true,
            peer_baseline: 0.0,
        };
        let candidates = policy.resolve_for_test(&workers, &[0, 1]);
        let score_a = policy.score(&workers[0], &candidates[0], &inputs);
        let score_b = policy.score(&workers[1], &candidates[1], &inputs);
        assert!(
            (score_a - score_b).abs() < TIE_EPSILON_SECS,
            "{score_a} vs {score_b}"
        );
        assert!((score_a - 10.0 * 1024.0 / 2000.0).abs() < 1e-9, "{score_a}");
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
        let policy = LeastLoadPolicy::with_params(0.0, 1024, 2000.0);
        let feed = TestFeed::attached_to(&policy);
        let workers: Vec<Arc<dyn Worker>> = (0..8)
            .map(|i| make_worker(&format!("http://w{i}:8000")))
            .collect();
        let mut loads = HashMap::new();
        for (i, worker) in workers.iter().enumerate() {
            let hidden = i < 2;
            for _ in 0..if hidden { 16 } else { 2 } {
                worker.increment_load();
            }
            let mut load = one_rank_load(0, 0.0, 2000.0);
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
        let workers = vec![make_worker("http://a:8000")];
        let info = SelectWorkerInfo::default();
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(0, 0.1, 100.0));
        feed.publish(&loads);

        // Two dispatches, then the engine is sampled, then one more.
        policy.select_min_expected_wait(&workers, &[0], &info, "test");
        policy.select_min_expected_wait(&workers, &[0], &info, "test");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let sampled_at = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(2));
        policy.select_min_expected_wait(&workers, &[0], &info, "test");
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 3072, 3));

        let mut sampled = one_rank_load(0, 0.1, 100.0);
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
        loads.insert("http://a:8000".to_string(), one_rank_load(0, 0.1, 100.0));
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
    fn waiting_queue_cap_zero_disables_veto() {
        // Cap 0 keeps the historical behavior: arbitrarily deep queues stay
        // routable.
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::with_params(0.0, 1024, 100.0));
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            one_rank_load_reqs_only(1000, 0.0, 100.0),
        );
        loads.insert(
            "http://b:8000".to_string(),
            one_rank_load_reqs_only(1000, 0.0, 100.0),
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
        let a = make_worker("http://a:8000");
        for _ in 0..2 {
            a.increment_load();
        }
        let workers = vec![
            a,
            make_worker("http://b:8000"),
            make_worker("http://c:8000"),
        ];
        let mut loads = HashMap::new();
        loads.insert("http://b:8000".to_string(), one_rank_load(1000, 0.2, 100.0));
        loads.insert(
            "http://c:8000".to_string(),
            one_rank_load(0, 0.2, 100_000.0),
        );
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
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(0, 0.2, 100.0));
        loads.insert("http://b:8000".to_string(), one_rank_load(2048, 0.2, 100.0));
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
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
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
    fn queue_depth_without_token_count_still_ranks_workers() {
        // Both backends report a queue but no token count for it. Reading the
        // absent count as an empty queue scores them identically, so the pick
        // is a coin flip and a badly backlogged worker keeps drawing traffic.
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new()); // p̄ = 1024
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            one_rank_load_reqs_only(64, 0.2, 100.0),
        );
        loads.insert(
            "http://b:8000".to_string(),
            one_rank_load_reqs_only(6, 0.2, 100.0),
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
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
        let mut a = one_rank_load(500, 0.2, 100.0);
        a.loads[0].num_waiting_reqs = 200;
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), a);
        loads.insert("http://b:8000".to_string(), one_rank_load(8000, 0.2, 100.0));
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
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::with_params(
            2.0,
            DEFAULT_MEAN_PREFILL_TOKENS,
            DEFAULT_THROUGHPUT,
        ));
        let mut loads = HashMap::new();
        loads.insert(
            "http://a:8000".to_string(),
            one_rank_load_reqs_only(0, 0.9, 100.0),
        );
        loads.insert(
            "http://b:8000".to_string(),
            one_rank_load_reqs_only(0, 0.1, 100.0),
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
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(5000, 0.2, 50.0));
        loads.insert("http://b:8000".to_string(), one_rank_load(5000, 0.2, 500.0));
        feed.publish(&loads);
        // a: 5000/50 = 100s ; b: 5000/500 = 10s -> pick b.
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(1)
        );
    }

    #[test]
    fn idle_rank_rate_is_not_a_live_throughput() {
        // Only a rank with requests running has a live rate; an idle rank's
        // gauge is whatever its last decode interval left behind.
        let mut load = one_rank_load(0, 0.1, 0.36);
        load.loads[0].num_running_reqs = 0;
        assert_eq!(live_throughput(&load), 0.0);

        load.loads.push(SchedulerLoadSnapshot {
            dp_rank: 1,
            num_running_reqs: 4,
            gen_throughput: 800.0,
            ..Default::default()
        });
        assert_eq!(live_throughput(&load), 800.0);
    }

    #[test]
    fn stale_idle_rate_does_not_herd_a_burst() {
        // Two idle workers: a reports the ~0.36 tok/s residual of its last
        // one-token health generation, b reports nothing. Scored against the
        // residual, a's first dispatch would read as 1024 / 0.36 ≈ 2800 s and
        // the rest of the burst would follow b until the next poll. Neither
        // rate is live, so both score at the default throughput and the
        // burst water-fills.
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
        let mut a = one_rank_load(0, 0.1, 0.36);
        a.loads[0].num_running_reqs = 0;
        let mut b = one_rank_load(0, 0.1, 0.0);
        b.loads[0].num_running_reqs = 0;
        feed.publish(&HashMap::from([
            ("http://a:8000".to_string(), a),
            ("http://b:8000".to_string(), b),
        ]));

        let mut picks = [0usize; 2];
        for _ in 0..8 {
            let idx = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .unwrap();
            picks[idx] += 1;
        }
        assert_eq!(picks, [4, 4]);
    }

    #[test]
    fn idle_worker_is_scored_at_its_busy_peers_rate() {
        // a is busy at 8000 tok/s; b is idle, its gauge a stale residual. At a
        // flat 2000 tok/s a dispatch to b would read as four times one to a
        // and a would take four of every five; at its peers' nominal rate the
        // two score alike (a's running request apart) and the burst
        // water-fills, the ties drawn at random.
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
        let mut b = one_rank_load(0, 0.1, 0.36);
        b.loads[0].num_running_reqs = 0;
        feed.publish(&HashMap::from([
            ("http://a:8000".to_string(), one_rank_load(0, 0.1, 8000.0)),
            ("http://b:8000".to_string(), b),
        ]));

        let mut picks = [0usize; 2];
        for _ in 0..8 {
            let idx = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .unwrap();
            picks[idx] += 1;
        }
        assert!(picks.iter().all(|&n| n >= 3), "{picks:?}");
    }

    #[test]
    fn zero_throughput_falls_back_to_default() {
        // A backend that reports no gen_throughput (0); the score must still
        // discriminate via the configured default_throughput, not collapse.
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new()); // default_throughput = 2000
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(10000, 0.2, 0.0));
        loads.insert("http://b:8000".to_string(), one_rank_load(1000, 0.2, 0.0));
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
        let a = make_worker("http://a:8000");
        let b = make_worker("http://b:8000");
        for _ in 0..5 {
            b.increment_load();
        }
        let workers = vec![a, b];
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(4000, 0.0, 100.0));
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
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::with_params(
            2.0,
            DEFAULT_MEAN_PREFILL_TOKENS,
            DEFAULT_THROUGHPUT,
        ));
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(0, 0.98, 100.0));
        loads.insert("http://b:8000".to_string(), one_rank_load(0, 0.0, 100.0));
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
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(0, 0.1, 100.0));
        loads.insert("http://b:8000".to_string(), one_rank_load(0, 0.1, 100.0));
        feed.publish(&loads);

        let info = SelectWorkerInfo::default(); // tokens unknown -> mean prefill
        let first = policy.select_worker(&workers, &info).unwrap();
        let second = policy.select_worker(&workers, &info).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn a_new_report_releases_the_credit_the_next_decision_reads() {
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(0, 0.1, 100.0));
        loads.insert("http://b:8000".to_string(), one_rank_load(0, 0.1, 100.0));
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
    fn a_credited_selection_books_the_work_the_caller_gives() {
        // a's report is the lighter: it wins both picks. Uncredited, the
        // 64-token prompt is booked whole; credited, the caller's count is.
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(0, 0.1, 100.0));
        loads.insert(
            "http://b:8000".to_string(),
            one_rank_load(10_000, 0.1, 100.0),
        );
        feed.publish(&loads);
        let tokens: Vec<u32> = (0..64).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };

        let picked = policy.select_min_expected_wait(&workers, &[0, 1], &info, "test");
        assert_eq!(picked, Some(0));
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 64, 1));

        let credit = |idx: usize| if idx == 0 { 7 } else { 1_000 };
        let picked = policy.select_min_expected_wait_credited(
            &workers,
            &[0, 1],
            &info,
            "test",
            Some(&credit),
        );
        assert_eq!(picked, Some(0));
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 71, 2));
        assert_eq!(policy.load_state_for_test("http://b:8000"), (true, 0, 0));
    }

    #[test]
    fn reset_discards_the_in_flight_credit() {
        // pick1 -> a; its credit tips pick2 -> b; after reset the credit is
        // gone and pick3 returns to a. The reports are the feed's, not the
        // policy's to discard.
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(1000, 0.2, 100.0));
        loads.insert("http://b:8000".to_string(), one_rank_load(2000, 0.2, 100.0));
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
        let workers = vec![make_worker("http://a:8000")];
        assert_eq!(
            policy.select_worker(&workers, &SelectWorkerInfo::default()),
            Some(0)
        );
    }

    #[test]
    fn remove_worker_prunes_its_tally() {
        let (policy, feed, workers) = fed_pair(LeastLoadPolicy::new());
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(0, 0.1, 100.0));
        loads.insert("http://b:8000".to_string(), one_rank_load(0, 0.1, 100.0));
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

    /// SplitMix64: the replay's fleet state, the same run to run.
    fn mix(mut value: u64) -> u64 {
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Worker `worker`'s report for round `round` of the replay: queued
    /// token-work, running and waiting requests, KV pressure, a live rate
    /// that is sometimes missing, a second rank on every third worker, and
    /// the instant it was sampled.
    fn replay_report(round: u64, worker: u64, sampled_at: Instant) -> WorkerLoadResponse {
        let draw = |salt: u64| mix(round * 1_000 + worker * 10 + salt);
        let mut load = one_rank_load(
            (draw(1) % 8_192) as i32,
            (draw(2) % 95) as f64 / 100.0,
            if draw(3) % 4 == 0 {
                0.0
            } else {
                500.0 + (draw(3) % 3_500) as f64
            },
        );
        load.loads[0].num_running_reqs = (draw(4) % 12) as i32;
        load.loads[0].num_waiting_reqs = (draw(5) % 6) as i32;
        if worker.is_multiple_of(3) {
            let mut rank = load.loads[0].clone();
            rank.dp_rank = 1;
            rank.num_running_reqs = (draw(6) % 12) as i32;
            rank.num_waiting_uncached_tokens = (draw(7) % 8_192) as i32;
            rank.gen_throughput = (draw(8) % 3_000) as f64;
            load.loads.push(rank);
            load.dp_rank_count = 2;
        }
        load.sampled_at = Some(sampled_at);
        load
    }

    /// An instant every later dispatch is strictly after, so a report
    /// sampled then releases exactly the dispatches made before it.
    fn instant_before_the_next_dispatch() -> Instant {
        let at = Instant::now();
        while Instant::now() <= at {}
        at
    }

    /// The pick sequence over a replayed fleet: twelve workers, one of them
    /// prefill and one that never reports; reports published with explicit
    /// sample instants between rounds of picks; the router's in-flight
    /// counts moving with the dispatches and with completions; token counts
    /// known and unknown; every third pick over every other worker with a
    /// holder's credit, as cache-aware picks; ties drawn from a seeded
    /// generator. The sequence was recorded from the scorer that looked each
    /// candidate up at every step of a decision; the scorer that resolves a
    /// candidate once must pick the same.
    #[test]
    fn the_replayed_fleet_picks_the_recorded_sequence() {
        const WORKERS: usize = 12;
        const PREFILL: usize = 3;
        const DARK: usize = 11;
        const ROUNDS: u64 = 8;
        const PICKS_PER_ROUND: usize = 12;
        const RECORDED: [usize; 96] = [
            11, 10, 10, 9, 11, 10, 9, 9, 10, 9, 9, 10, 1, 1, 8, 1, 11, 0, 1, 4, 0, 11, 0, 8, 5, 5,
            2, 5, 2, 2, 5, 10, 2, 5, 2, 8, 0, 4, 8, 9, 7, 4, 5, 9, 2, 8, 5, 2, 4, 10, 4, 10, 4, 4,
            10, 4, 10, 2, 10, 4, 7, 1, 2, 7, 1, 2, 1, 5, 8, 7, 1, 2, 7, 7, 10, 7, 7, 4, 7, 3, 10,
            3, 7, 4, 0, 0, 0, 0, 8, 2, 6, 8, 2, 6, 0, 2,
        ];
        let policy = LeastLoadPolicy::with_params(0.15, 1024, 2_000.0).with_tie_break_seed(11);
        let feed = TestFeed::attached_to(&policy);
        let workers: Vec<Arc<dyn Worker>> = (0..WORKERS)
            .map(|i| {
                let role = if i == PREFILL {
                    WorkerType::Prefill
                } else {
                    WorkerType::Regular
                };
                Arc::new(
                    BasicWorkerBuilder::new(format!("http://w{i:02}:8000"))
                        .worker_type(role)
                        .health_config(no_health_check())
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        let prompt: Vec<u32> = (0..4_096).collect();
        let every_other: Vec<usize> = (0..WORKERS).step_by(2).collect();
        let held = |idx: usize| 256 + (idx as u64 * 97) % 2_048;
        let mut picks = Vec::with_capacity(RECORDED.len());
        for round in 0..ROUNDS {
            let sampled_at = instant_before_the_next_dispatch();
            let reports = (0..WORKERS)
                .filter(|&i| i != DARK)
                .map(|i| {
                    (
                        workers[i].url().to_string(),
                        replay_report(round, i as u64, sampled_at),
                    )
                })
                .collect();
            feed.publish(&reports);
            for pick in 0..PICKS_PER_ROUND {
                let draw = mix(round * 100 + pick as u64);
                let tokens =
                    (!draw.is_multiple_of(3)).then(|| &prompt[..64 + (draw % 4_000) as usize]);
                let info = SelectWorkerInfo {
                    tokens,
                    ..Default::default()
                };
                let chosen = if pick % 3 == 2 {
                    policy.select_min_expected_wait_credited(
                        &workers,
                        &every_other,
                        &info,
                        "replay",
                        Some(&held),
                    )
                } else {
                    policy.select_worker(&workers, &info)
                }
                .expect("a healthy pool picks");
                workers[chosen].increment_load();
                picks.push(chosen);
                if pick % 4 == 3 {
                    // A completion: the request dispatched three picks ago ends.
                    workers[picks[picks.len() - 4]].decrement_load();
                }
            }
        }
        assert_eq!(
            picks, RECORDED,
            "the replay picked {picks:?}; the recorded sequence is the decision"
        );
    }

    /// A read under the read guard (a split decision's resolve, the test
    /// view) subtracts the dispatches a report reflects without releasing
    /// them; a decision under the held guard releases them as it resolves;
    /// a split decision releases them for its candidates in its write
    /// phase, once it holds the guard. Either decision credits its winner.
    #[test]
    fn a_decision_releases_what_its_reports_reflect_and_a_read_does_not() {
        let policy = LeastLoadPolicy::new();
        let feed = TestFeed::attached_to(&policy);
        let workers = vec![make_worker("http://a:8000"), make_worker("http://b:8000")];
        let info = SelectWorkerInfo::default();
        let mut loads = HashMap::new();
        loads.insert("http://a:8000".to_string(), one_rank_load(0, 0.1, 100.0));
        loads.insert("http://b:8000".to_string(), one_rank_load(0, 0.1, 100.0));
        feed.publish(&loads);
        for _ in 0..3 {
            policy.select_min_expected_wait(&workers, &[0], &info, "test");
        }
        assert_eq!(policy.held_dispatches_for_test("http://a:8000"), 3);
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 3072, 3));

        // A report sampled after the three: the credit reads released, the
        // tally still holds them.
        feed.publish(&loads);
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 0, 0));
        assert_eq!(
            policy.held_dispatches_for_test("http://a:8000"),
            3,
            "a read releases nothing"
        );
        // A decision whose candidates do not include a leaves its tally.
        assert_eq!(
            policy.select_min_expected_wait(&workers, &[1], &info, "test"),
            Some(1)
        );
        assert_eq!(policy.held_dispatches_for_test("http://a:8000"), 3);
        assert_eq!(policy.held_dispatches_for_test("http://b:8000"), 1);
        // A decision over both releases a's three (a reads idle, b holds
        // its credit) and records the dispatch on a.
        assert_eq!(
            policy.select_min_expected_wait(&workers, &[0, 1], &info, "test"),
            Some(0)
        );
        assert_eq!(policy.held_dispatches_for_test("http://a:8000"), 1);
        assert_eq!(policy.held_dispatches_for_test("http://b:8000"), 1);
        assert_eq!(policy.load_state_for_test("http://a:8000"), (true, 1024, 1));
    }

    /// Decisions on several threads at once each land their credit: the
    /// tally counts every dispatch, whichever worker each decision picked.
    #[test]
    fn decisions_on_several_threads_all_land_their_credit() {
        // Four workers hold one guard across the decision, sixteen split it.
        threads_land_their_credit(4);
        threads_land_their_credit(16);
    }

    fn threads_land_their_credit(fleet: usize) {
        let policy = Arc::new(LeastLoadPolicy::new());
        let feed = TestFeed::attached_to(policy.as_ref());
        let workers: Arc<Vec<Arc<dyn Worker>>> = Arc::new(
            (0..fleet)
                .map(|i| make_worker(&format!("http://w{i}:8000")))
                .collect(),
        );
        let mut loads = HashMap::new();
        for worker in workers.iter() {
            loads.insert(worker.url().to_string(), one_rank_load(0, 0.1, 100.0));
        }
        feed.publish(&loads);
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (policy, workers) = (policy.clone(), workers.clone());
                std::thread::spawn(move || {
                    let info = SelectWorkerInfo::default();
                    for _ in 0..250 {
                        policy
                            .select_worker(&workers, &info)
                            .expect("a healthy pool picks");
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("a decision thread panicked");
        }
        let requests: u64 = workers
            .iter()
            .map(|worker| policy.load_state_for_test(worker.url()).2)
            .sum();
        assert_eq!(requests, 8 * 250);
    }

    /// The two guard shapes meet at [`HELD_GUARD_CANDIDATES`]: seven
    /// candidates hold one write guard, eight split the read, the scoring and
    /// the write. On the same inputs both pick the same worker, twice, and
    /// leave the same tally.
    #[test]
    fn seven_and_eight_candidates_take_the_two_guard_shapes_and_agree() {
        assert_eq!(HELD_GUARD_CANDIDATES, 7);
        let decide_twice = |candidates: usize| {
            let policy = LeastLoadPolicy::new();
            let feed = TestFeed::attached_to(&policy);
            let workers: Vec<Arc<dyn Worker>> = (0..9)
                .map(|i| make_worker(&format!("http://w{i}:8000")))
                .collect();
            let mut loads = HashMap::new();
            for (i, worker) in workers.iter().enumerate() {
                // Worker 5 is idle, every other one queues work.
                let waiting = if i == 5 { 0 } else { 4096 };
                loads.insert(worker.url().to_string(), one_rank_load(waiting, 0.1, 100.0));
            }
            feed.publish(&loads);
            let info = SelectWorkerInfo::default();
            let indices: Vec<usize> = (0..candidates).collect();
            let first = policy.select_min_expected_wait(&workers, &indices, &info, "test");
            let second = policy.select_min_expected_wait(&workers, &indices, &info, "test");
            let tally: Vec<(u64, u64, usize)> = workers
                .iter()
                .map(|worker| {
                    let (_, tokens, requests) = policy.load_state_for_test(worker.url());
                    (
                        tokens,
                        requests,
                        policy.held_dispatches_for_test(worker.url()),
                    )
                })
                .collect();
            (first, second, tally)
        };
        let seven = decide_twice(7);
        let eight = decide_twice(8);
        assert_eq!(seven.0, Some(5));
        assert_eq!(eight.0, Some(5));
        assert_eq!(
            seven.1, eight.1,
            "the second pick, after one credit, agrees"
        );
        assert_eq!(seven.2, eight.2, "the tallies agree");
    }
}
