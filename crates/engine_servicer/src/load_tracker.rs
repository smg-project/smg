//! What a servicer can tell about its engine's load from the requests it
//! forwards: queued token-work, prefill/generation throughput and the prefix-cache
//! hit rate, per data-parallel rank. The gateway's expected-wait score reads
//! them from `GetLoads`; an engine whose stats do not carry them (vLLM
//! reports running, waiting and KV usage) would otherwise be scored on
//! defaults and routed unlike its peers.
//!
//! The servicer sees every request it submits and the rank it goes to, the
//! engine's first output for it (with the prompt and cached token counts) and
//! every token streamed, so, for each rank:
//!
//! - queued token-work is the uncached prompt tokens of the requests the
//!   rank has not started: of the submitted requests without a first
//!   output, the youngest `num_waiting_reqs` (the engine admits FCFS, so the
//!   older ones are the ones in prefill), each prompt discounted by the
//!   rank's recent hit rate since its own cached count is not known until it
//!   starts;
//! - prefill capacity uses actual computed tokens over the union of their
//!   engine scheduled-to-first-output intervals (not summed request latencies).
//!   It excludes initial queueing and idle gaps, but is still an observed
//!   service-rate estimate, not GPU kernel time. At least two valid samples
//!   are required; up to 64 are retained for 60 seconds. Missing engine timing,
//!   preemptions, zero-work hits and intervals shorter than 1 ms do not calibrate
//!   capacity. vLLM must enable `log_stats` to emit the scheduling events. A
//!   wire that carries no engine timing measures on the servicer's clock
//!   instead, from the requests that met no queue on their rank
//!   ([`PrefillProbe`]: the submit to the first output, once per request);
//! - generation throughput is the tokens the rank streamed over the last
//!   [`THROUGHPUT_WINDOW`];
//! - the hit rate is cached over prompt tokens across the rank's last
//!   [`HIT_RATE_SAMPLES`] first outputs.
//!
//! A rank keeps its own figures: a report that put one rank's figures on
//! another, or the whole servicer's on its first rank, would read a serving
//! rank as idle or misses as hits. These are the field semantics the SGLang
//! servicer reports from its scheduler and the mock engine from its queue.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    time::{Duration, Instant},
};

use engine_zmq_client::protocol::EngineLoad;

use crate::servicer::WireState;

/// How far back streamed tokens count toward the throughput.
pub(crate) const THROUGHPUT_WINDOW: Duration = Duration::from_secs(2);
/// How many first outputs the hit rate averages over.
pub(crate) const HIT_RATE_SAMPLES: usize = 64;
/// Bound the input-capacity estimator independently of cache-hit history.
const PREFILL_SAMPLES: usize = 64;
const PREFILL_SAMPLE_TTL: Duration = Duration::from_secs(60);

/// The estimated fields, as `GetLoads` reports them for one rank.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LoadEstimate {
    /// Uncached prompt tokens of the requests the rank has not started.
    pub queued_token_work: i32,
    /// Tokens the rank streamed per second over the window.
    pub gen_throughput: f64,
    /// Actual computed input tokens per active prefill second.
    pub prefill_throughput: Option<f64>,
    /// Cached over prompt tokens of the rank's recent first outputs, in
    /// `[0, 1]`.
    pub cache_hit_rate: f64,
}

struct Pending {
    request_id: String,
    prompt_tokens: u32,
    /// The rank the request went to.
    rank: i32,
}

/// One rank's recent first outputs and streamed tokens.
#[derive(Default)]
struct Rank {
    /// Recent first outputs: (prompt tokens, cached tokens).
    hits: VecDeque<(u64, u64)>,
    prompt_sum: u64,
    cached_sum: u64,
    /// Tokens streamed, by time, within the window.
    generated: VecDeque<(Instant, u64)>,
    generated_sum: u64,
    /// (observed locally, computed tokens, engine start, engine end).
    prefills: VecDeque<(Instant, u32, f64, f64)>,
}

impl Rank {
    fn trim(&mut self, now: Instant) {
        while let Some(&(at, tokens)) = self.generated.front() {
            if now.duration_since(at) <= THROUGHPUT_WINDOW {
                break;
            }
            self.generated.pop_front();
            self.generated_sum -= tokens;
        }
    }

    /// Estimate active input capacity from a bounded window of completed prefills.
    fn prefill_throughput(&mut self, now: Instant) -> Option<f64> {
        // Retain capacity across short idle periods, but never indefinitely.
        self.prefills
            .retain(|(at, ..)| now.saturating_duration_since(*at) <= PREFILL_SAMPLE_TTL);
        if self.prefills.len() < 2 {
            return None;
        }
        let tokens: f64 = self
            .prefills
            .iter()
            .map(|(_, tokens, ..)| f64::from(*tokens))
            .sum();
        let mut intervals: Vec<_> = self
            .prefills
            .iter()
            .map(|(_, _, start, end)| (*start, *end))
            .collect();
        intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut seconds = 0.0;
        let mut until = f64::NEG_INFINITY;
        for (start, end) in intervals {
            // Union concurrent intervals: each engine second counts once.
            seconds += (end - start.max(until)).max(0.0);
            until = until.max(end);
        }
        let rate = tokens / seconds;
        (rate.is_finite() && rate > 0.0).then_some(rate)
    }

    fn hit_rate(&self) -> f64 {
        if self.prompt_sum == 0 {
            0.0
        } else {
            (self.cached_sum as f64 / self.prompt_sum as f64).clamp(0.0, 1.0)
        }
    }
}

#[derive(Default)]
struct Inner {
    /// Submitted requests without a first output yet, oldest first, each
    /// under the rank it went to.
    pending: VecDeque<Pending>,
    /// The ranks that have started a request or streamed a token.
    ranks: BTreeMap<i32, Rank>,
}

/// Per-servicer load bookkeeping, kept per rank; every method is cheap and
/// lock-scoped.
pub(crate) struct LoadTracker {
    inner: Mutex<Inner>,
    /// Where the servicer-clock prefill samples count their seconds from
    /// ([`Self::prefill_observed`]): one axis per tracker, so a rank's
    /// intervals union on it.
    epoch: Instant,
}

impl Default for LoadTracker {
    fn default() -> Self {
        Self {
            inner: Mutex::default(),
            epoch: Instant::now(),
        }
    }
}

impl LoadTracker {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A request with `prompt_tokens` was handed to `rank`. Whether the rank
    /// had no request of this servicer in flight ahead of it (none submitted
    /// and not started), which a servicer-clock prefill sample needs
    /// ([`PrefillProbe`]).
    pub(crate) fn submitted(&self, rank: i32, request_id: &str, prompt_tokens: u32) -> bool {
        let mut inner = self.lock();
        let idle = !inner.pending.iter().any(|pending| pending.rank == rank);
        inner.pending.push_back(Pending {
            request_id: request_id.to_string(),
            prompt_tokens,
            rank,
        });
        idle
    }

    /// `rank`'s first output for a request: it has started, and its prompt
    /// had `cached_tokens` of `prompt_tokens` in that rank's prefix cache.
    pub(crate) fn first_output(
        &self,
        rank: i32,
        request_id: &str,
        prompt_tokens: u32,
        cached_tokens: u32,
    ) {
        let mut inner = self.lock();
        drop_pending(&mut inner, request_id);
        if prompt_tokens == 0 {
            return;
        }
        let sample = (
            u64::from(prompt_tokens),
            u64::from(cached_tokens.min(prompt_tokens)),
        );
        let rank = inner.ranks.entry(rank).or_default();
        rank.hits.push_back(sample);
        rank.prompt_sum += sample.0;
        rank.cached_sum += sample.1;
        if rank.hits.len() > HIT_RATE_SAMPLES {
            if let Some((prompt, cached)) = rank.hits.pop_front() {
                rank.prompt_sum -= prompt;
                rank.cached_sum -= cached;
            }
        }
    }

    /// Record an actual prefill computation, using one engine clock for both
    /// ends. Zero-work cache hits and invalid timings cannot calibrate capacity.
    pub(crate) fn prefill_computed(&self, rank: i32, tokens: u32, start: f64, end: f64) {
        self.prefill_computed_at(rank, tokens, start, end, Instant::now());
    }

    /// A prefill observed on the servicer's clock, on a wire that carries no
    /// engine timing: `tokens` computed between the submit (`started`, the
    /// engine had accepted the request) and its first output (`ended`). The
    /// caller vouches that the request met no queue on the rank (see
    /// [`PrefillProbe`]); the sample goes through the same validity rule and
    /// estimator as an engine-timed one.
    pub(crate) fn prefill_observed(
        &self,
        rank: i32,
        tokens: u32,
        started: Instant,
        ended: Instant,
    ) {
        let seconds = |at: Instant| at.saturating_duration_since(self.epoch).as_secs_f64();
        self.prefill_computed_at(rank, tokens, seconds(started), seconds(ended), ended);
    }

    /// Validate and retain a sample with a local expiry timestamp.
    fn prefill_computed_at(&self, rank: i32, tokens: u32, start: f64, end: f64, now: Instant) {
        if tokens == 0
            || !start.is_finite()
            || !end.is_finite()
            || start < 0.0
            || end - start < 0.001
        {
            return;
        }
        let mut inner = self.lock();
        let figures = inner.ranks.entry(rank).or_default();
        figures.prefills.push_back((now, tokens, start, end));
        if figures.prefills.len() > PREFILL_SAMPLES {
            figures.prefills.pop_front();
        }
    }

    /// `rank` streamed `tokens` to the client at `at` (the instant the
    /// caller already read for its own bookkeeping).
    pub(crate) fn generated_at(&self, rank: i32, tokens: u32, at: Instant) {
        if tokens == 0 {
            return;
        }
        let mut inner = self.lock();
        let rank = inner.ranks.entry(rank).or_default();
        rank.generated.push_back((at, u64::from(tokens)));
        rank.generated_sum += u64::from(tokens);
        rank.trim(at);
    }

    /// A request ended (or was aborted) without the engine starting it.
    pub(crate) fn finished(&self, request_id: &str) {
        drop_pending(&mut self.lock(), request_id);
    }

    /// `rank`'s estimate now, given the engine's own count of that rank's
    /// waiting requests.
    pub(crate) fn estimate(&self, rank: i32, num_waiting_reqs: i32) -> LoadEstimate {
        self.estimate_at(rank, num_waiting_reqs, Instant::now())
    }

    /// Estimate rank load at a supplied observation time.
    pub(crate) fn estimate_at(
        &self,
        rank: i32,
        num_waiting_reqs: i32,
        now: Instant,
    ) -> LoadEstimate {
        let mut inner = self.lock();
        let (hit_rate, gen_throughput, prefill_throughput) = match inner.ranks.get_mut(&rank) {
            Some(figures) => {
                figures.trim(now);
                (
                    figures.hit_rate(),
                    figures.generated_sum as f64 / THROUGHPUT_WINDOW.as_secs_f64(),
                    figures.prefill_throughput(now),
                )
            }
            None => (0.0, 0.0, None),
        };
        let waiting = usize::try_from(num_waiting_reqs).unwrap_or(0);
        let queued: f64 = inner
            .pending
            .iter()
            .rev()
            .filter(|pending| pending.rank == rank)
            .take(waiting)
            .map(|pending| f64::from(pending.prompt_tokens) * (1.0 - hit_rate))
            .sum();
        LoadEstimate {
            queued_token_work: queued.round().clamp(0.0, f64::from(i32::MAX)) as i32,
            gen_throughput,
            prefill_throughput,
            cache_hit_rate: hit_rate,
        }
    }

    /// Submitted requests the engine has not started.
    #[cfg(test)]
    pub(crate) fn pending(&self) -> usize {
        self.lock().pending.len()
    }

    /// The prefill samples `rank` holds (expired ones included until the
    /// next estimate trims them).
    #[cfg(test)]
    pub(crate) fn prefill_samples(&self, rank: i32) -> usize {
        self.lock()
            .ranks
            .get(&rank)
            .map_or(0, |figures| figures.prefills.len())
    }
}

/// The data-parallel rank a choice's engine request went to, as the tracker
/// and `GetLoads` name it (an engine whose identity names no index counts as
/// the first).
pub(crate) fn rank_of(engine_index: Option<u32>) -> i32 {
    engine_index
        .and_then(|index| i32::try_from(index).ok())
        .unwrap_or(0)
}

/// Whether a rank's last load tail leaves a newly handed request no queue:
/// nothing waiting and room in the admission window (`window` 0 or less:
/// unknown, counts as room). A rank without a tail yet has reported no
/// batch since boot and has room.
pub(crate) fn rank_has_room(tail: Option<EngineLoad>, window: i32) -> bool {
    let Some(tail) = tail else {
        return true;
    };
    let window = u64::try_from(window).unwrap_or(0);
    tail.num_waiting == 0 && (window == 0 || tail.num_running < window)
}

/// One request's servicer-clock prefill sample on a wire without engine
/// timing: armed at the submit when the request met no queue on its rank
/// (nothing of this servicer in flight ahead of it, the rank's last load
/// tail with nothing waiting and room in its window, a single choice), taken
/// once at the request's first output with its prompt figures. The interval
/// holds more than the prefill: the two wire hops and the wait for the
/// scheduler's current step (near-constant), and on the workers the gateway
/// reads this field for, the prefill role of a disaggregated pair, the
/// bootstrap wait for the decode peer and the KV transfer, which grow with
/// the prompt (an encoder's embeddings awaited on a language-side worker
/// likewise). The reading is therefore low, never high: by a fixed share on
/// a co-located worker, by a proportional one on a prefill-role worker. A
/// saturated rank yields none and its estimate expires into the gateway's
/// fallback.
#[derive(Debug, Default)]
pub(crate) struct PrefillProbe {
    started: Option<Instant>,
}

impl PrefillProbe {
    /// Armed at `started` when `admitted` (the request met no queue), unarmed
    /// otherwise.
    pub(crate) fn new(admitted: bool, started: Instant) -> Self {
        Self {
            started: admitted.then_some(started),
        }
    }

    /// The request's first output with its prompt figures, read at `now`:
    /// the computed tokens (the prompt less what was cached) over the
    /// interval since the submit, once; nothing when unarmed or taken.
    pub(crate) fn first_output(
        &mut self,
        loads: &LoadTracker,
        rank: i32,
        prompt_tokens: u32,
        cached_tokens: u32,
        now: Instant,
    ) {
        if let Some(started) = self.started.take() {
            loads.prefill_observed(
                rank,
                prompt_tokens.saturating_sub(cached_tokens),
                started,
                now,
            );
        }
    }
}

/// What a wire knows when it hands a request to the engine, for
/// [`LoadTracker::submit`].
pub(crate) struct Submission<'a> {
    pub(crate) request_id: &'a str,
    /// The rank the request's first choice went to: the choice that reports
    /// the prompt, so the queued work and the hit rate land there.
    pub(crate) rank: i32,
    pub(crate) prompt_tokens: u32,
    /// The request's `n`; 0 is unset, one choice.
    pub(crate) choices: u32,
    /// The rank's last load tail, `None` before its first batch.
    pub(crate) tail: Option<EngineLoad>,
    /// The engine's admission window; 0 or less when unknown.
    pub(crate) window: i32,
    /// The servicer's instant after the engine accepted the request.
    pub(crate) at: Instant,
}

impl LoadTracker {
    /// A request handed to the engine: queued as by [`Self::submitted`], and
    /// the servicer-clock probe for its first choice, armed when the request
    /// met no queue on its rank (nothing of ours ahead of it, the rank's
    /// last load tail with nothing waiting and room in its window, a single
    /// choice).
    pub(crate) fn submit(&self, submission: Submission<'_>) -> PrefillProbe {
        let idle = self.submitted(
            submission.rank,
            submission.request_id,
            submission.prompt_tokens,
        );
        PrefillProbe::new(
            idle && submission.choices <= 1 && rank_has_room(submission.tail, submission.window),
            submission.at,
        )
    }
}

/// One choice's share of the tracker's bookkeeping, held by the wire's
/// choice stream: the rank its own engine request went to (an unpinned
/// fan-out is placed per choice, so each credits its tokens where they were
/// made), the servicer-clock probe (the first choice's, armed when the
/// request met no queue; unarmed on the rest) and whether the request's
/// prompt was counted.
#[derive(Debug)]
pub(crate) struct ChoiceLoad {
    rank: i32,
    /// Whether this is the request's first choice, the one that reports its
    /// prompt (every choice of an `n > 1` fan-out reports the same one).
    first: bool,
    probe: PrefillProbe,
    prompt_counted: bool,
}

impl ChoiceLoad {
    pub(crate) fn new(rank: i32, index: u32, probe: PrefillProbe) -> Self {
        Self {
            rank,
            first: index == 0,
            probe,
            prompt_counted: false,
        }
    }

    /// The rank the choice's engine request went to.
    pub(crate) fn rank(&self) -> i32 {
        self.rank
    }

    /// The choice's output with the request's prompt figures, read at
    /// `now`: once per request, the tracker learns that the engine started
    /// it and how much of the prompt was cached, and the probe reads the
    /// prefill it timed.
    pub(crate) fn count_prompt(
        &mut self,
        loads: &LoadTracker,
        request_id: &str,
        prompt_tokens: u32,
        cached_tokens: u32,
        now: Instant,
    ) {
        if self.prompt_counted || prompt_tokens == 0 {
            return;
        }
        if self.first {
            loads.first_output(self.rank, request_id, prompt_tokens, cached_tokens);
            self.probe
                .first_output(loads, self.rank, prompt_tokens, cached_tokens, now);
        }
        self.prompt_counted = true;
    }

    /// The choice streamed `tokens` to the client at `at`.
    pub(crate) fn generated(&self, loads: &LoadTracker, tokens: u32, at: Instant) {
        loads.generated_at(self.rank, tokens, at);
    }
}

/// Takes a request out of the tracker's queue when its stream ends, in case
/// the engine never started it; the request's stream holds one.
pub(crate) struct PendingGuard<S: WireState> {
    state: Arc<S>,
    request_id: String,
}

impl<S: WireState> PendingGuard<S> {
    pub(crate) fn new(state: Arc<S>, request_id: &str) -> Self {
        Self {
            state,
            request_id: request_id.to_string(),
        }
    }
}

impl<S: WireState> Drop for PendingGuard<S> {
    fn drop(&mut self) {
        self.state.shared().loads.finished(&self.request_id);
    }
}

fn drop_pending(inner: &mut Inner, request_id: &str) {
    if let Some(index) = inner
        .pending
        .iter()
        .position(|pending| pending.request_id == request_id)
    {
        inner.pending.remove(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Concurrent intervals count once and capacity expires independently per rank.
    #[test]
    fn prefill_capacity_unions_concurrency_preserves_idle_and_expires_per_rank() {
        let tracker = LoadTracker::default();
        let now = Instant::now();
        tracker.prefill_computed_at(0, 1_000, 10.0, 12.0, now);
        assert_eq!(tracker.estimate_at(0, 0, now).prefill_throughput, None);
        tracker.prefill_computed_at(0, 2_000, 11.0, 13.0, now);
        // 3000 computed tokens / 3 active seconds, not 4 summed request seconds.
        assert_eq!(
            tracker.estimate_at(0, 0, now).prefill_throughput,
            Some(1_000.0)
        );
        tracker.generated_at(0, 1, now);
        assert_eq!(tracker.estimate_at(0, 0, now).gen_throughput, 0.5);
        assert_eq!(
            tracker
                .estimate_at(0, 0, now + Duration::from_secs(30))
                .prefill_throughput,
            Some(1_000.0)
        );
        assert_eq!(tracker.estimate_at(1, 0, now).prefill_throughput, None);
        assert_eq!(
            tracker
                .estimate_at(0, 0, now + Duration::from_secs(61))
                .prefill_throughput,
            None
        );
    }

    /// Invalid timings and cache-only work cannot train the capacity estimate.
    #[test]
    fn prefill_capacity_rejects_cache_hits_and_invalid_samples() {
        let tracker = LoadTracker::default();
        let now = Instant::now();
        for (tokens, start, end) in [
            (0, 1.0, 2.0),
            (100, 2.0, 1.0),
            (100, 1.0, 1.0),
            (100, f64::NAN, 2.0),
            (100, 1.0, f64::INFINITY),
        ] {
            tracker.prefill_computed_at(0, tokens, start, end, now);
        }
        assert_eq!(tracker.estimate_at(0, 0, now).prefill_throughput, None);
        tracker.prefill_computed_at(0, 1_000, 1.0, 2.0, now);
        tracker.prefill_computed_at(0, 1_000, 100.0, 101.0, now);
        // Idle engine time between requests is excluded.
        assert_eq!(
            tracker.estimate_at(0, 0, now).prefill_throughput,
            Some(1_000.0)
        );
    }

    #[test]
    fn queued_token_work_is_the_youngest_waiting_prompts_discounted_by_the_hit_rate() {
        let tracker = LoadTracker::default();
        let now = Instant::now();
        assert_eq!(tracker.estimate_at(0, 5, now), LoadEstimate::default());

        // Three in flight, none started: the engine says two are waiting, so
        // the oldest is in prefill and the two youngest are queued work.
        tracker.submitted(0, "a", 1_000);
        tracker.submitted(0, "b", 300);
        tracker.submitted(0, "c", 200);
        assert_eq!(tracker.estimate_at(0, 2, now).queued_token_work, 500);
        assert_eq!(tracker.estimate_at(0, 0, now).queued_token_work, 0);
        assert_eq!(
            tracker.estimate_at(0, 10, now).queued_token_work,
            1_500,
            "never more than we hold"
        );

        // The first output of `a`: half its prompt was cached. Queued prompts
        // are discounted by that rate.
        tracker.first_output(0, "a", 1_000, 500);
        assert_eq!(tracker.pending(), 2);
        let estimate = tracker.estimate_at(0, 2, now);
        assert_eq!(estimate.cache_hit_rate, 0.5);
        assert_eq!(estimate.queued_token_work, 250);

        // A request that ends before starting leaves the queue.
        tracker.finished("b");
        assert_eq!(tracker.estimate_at(0, 2, now).queued_token_work, 100);
        tracker.first_output(0, "c", 200, 200);
        assert_eq!(tracker.pending(), 0);
        assert_eq!(tracker.estimate_at(0, 3, now).queued_token_work, 0);
        // 700 cached of 1_200 prompt tokens.
        assert!((tracker.estimate_at(0, 0, now).cache_hit_rate - 700.0 / 1_200.0).abs() < 1e-9);
    }

    #[test]
    fn throughput_counts_tokens_inside_the_window_only() {
        let tracker = LoadTracker::default();
        let start = Instant::now();
        tracker.generated_at(0, 1_000, start);
        tracker.generated_at(0, 3_000, start + Duration::from_secs(1));
        let per_second = |estimate: LoadEstimate| estimate.gen_throughput;
        assert_eq!(
            per_second(tracker.estimate_at(0, 0, start + Duration::from_secs(1))),
            2_000.0
        );
        // Two seconds after the first sample it falls out of the window.
        assert_eq!(
            per_second(tracker.estimate_at(0, 0, start + Duration::from_millis(2_500))),
            1_500.0
        );
        assert_eq!(
            per_second(tracker.estimate_at(0, 0, start + Duration::from_secs(4))),
            0.0
        );
        tracker.generated_at(0, 0, start + Duration::from_secs(4));
        assert_eq!(
            per_second(tracker.estimate_at(0, 0, start + Duration::from_secs(4))),
            0.0
        );
    }

    #[test]
    fn the_hit_rate_averages_recent_first_outputs_and_ignores_empty_prompts() {
        let tracker = LoadTracker::default();
        let now = Instant::now();
        tracker.submitted(0, "x", 0);
        tracker.first_output(0, "x", 0, 0);
        assert_eq!(tracker.estimate_at(0, 0, now).cache_hit_rate, 0.0);
        for index in 0..HIT_RATE_SAMPLES {
            tracker.first_output(0, &format!("old-{index}"), 100, 0);
        }
        assert_eq!(tracker.estimate_at(0, 0, now).cache_hit_rate, 0.0);
        for index in 0..HIT_RATE_SAMPLES {
            tracker.first_output(0, &format!("new-{index}"), 100, 100);
        }
        assert_eq!(
            tracker.estimate_at(0, 0, now).cache_hit_rate,
            1.0,
            "the old samples aged out"
        );
        // Cached can never exceed the prompt.
        tracker.first_output(0, "odd", 10, 50);
        assert!(tracker.estimate_at(0, 0, now).cache_hit_rate <= 1.0);
    }

    /// One recorded sequence of prefills reads the same capacity whether its
    /// ends come from an engine's clock or from the servicer's instants (the
    /// wires without engine timing): the two paths share the estimator.
    #[test]
    fn servicer_clock_samples_read_the_same_capacity_as_engine_clock_ones() {
        let recorded = [
            (1_000_u32, 10.0_f64, 12.0_f64),
            (2_000, 11.0, 13.0),
            (600, 20.0, 20.5),
        ];
        let engine_timed = LoadTracker::default();
        let servicer_timed = LoadTracker::default();
        let now = Instant::now();
        let origin = servicer_timed.epoch;
        for (tokens, start, end) in recorded {
            engine_timed.prefill_computed_at(0, tokens, start, end, now);
            servicer_timed.prefill_observed(
                0,
                tokens,
                origin + Duration::from_secs_f64(start),
                origin + Duration::from_secs_f64(end),
            );
        }
        // 3600 computed tokens over 3 + 0.5 active seconds, the overlap of
        // the first two prefills counted once.
        let expected = 3_600.0 / 3.5;
        assert_eq!(
            engine_timed.estimate_at(0, 0, now).prefill_throughput,
            Some(expected)
        );
        assert_eq!(
            servicer_timed
                .estimate_at(0, 0, origin + Duration::from_secs(21))
                .prefill_throughput,
            Some(expected)
        );
        assert_eq!(servicer_timed.prefill_samples(0), 3);
    }

    /// A servicer-clock sample is taken once per request, from an armed
    /// probe only, and its tokens are the prompt less what was cached.
    #[test]
    fn a_probe_samples_the_computed_tokens_once_when_armed() {
        let tracker = LoadTracker::default();
        let start = Instant::now();
        let second = Duration::from_secs(1);
        let mut unarmed = PrefillProbe::new(false, start);
        unarmed.first_output(&tracker, 0, 1_000, 0, start + second);
        let mut default = PrefillProbe::default();
        default.first_output(&tracker, 0, 1_000, 0, start + second);
        assert_eq!(tracker.prefill_samples(0), 0);
        let mut armed = PrefillProbe::new(true, start);
        armed.first_output(&tracker, 0, 1_000, 400, start + second);
        // A later output of the same request adds nothing.
        armed.first_output(&tracker, 0, 1_000, 400, start + 2 * second);
        assert_eq!(tracker.prefill_samples(0), 1);
        let mut later = PrefillProbe::new(true, start + 10 * second);
        later.first_output(&tracker, 0, 300, 0, start + 11 * second);
        // 600 + 300 computed tokens over two one-second prefills.
        assert_eq!(
            tracker
                .estimate_at(0, 0, start + 11 * second)
                .prefill_throughput,
            Some(450.0)
        );
        // A prompt served whole from the cache computes nothing.
        let mut hit = PrefillProbe::new(true, start + 20 * second);
        hit.first_output(&tracker, 0, 500, 500, start + 21 * second);
        assert_eq!(tracker.prefill_samples(0), 2);
    }

    /// The submit says whether the rank had nothing of ours in flight ahead
    /// of the request; the rank's last load tail must show no queue and room
    /// in its window for the request to have met no queue.
    #[test]
    fn a_request_meets_no_queue_only_on_an_idle_rank_with_room() {
        let tracker = LoadTracker::default();
        assert!(tracker.submitted(0, "a", 100));
        assert!(!tracker.submitted(0, "b", 100), "a is ahead of b");
        assert!(tracker.submitted(1, "c", 100), "another rank");
        tracker.first_output(0, "a", 100, 0);
        assert!(!tracker.submitted(0, "d", 100), "b has not started");
        tracker.finished("b");
        tracker.first_output(0, "d", 100, 0);
        assert!(tracker.submitted(0, "e", 100));

        let tail = |num_running, num_waiting| {
            Some(EngineLoad {
                num_running,
                num_waiting,
                kv_cache_usage: 0.0,
            })
        };
        assert!(rank_has_room(None, 8), "no batch since boot");
        assert!(rank_has_room(tail(3, 0), 8));
        assert!(!rank_has_room(tail(3, 1), 8), "one waits");
        assert!(!rank_has_room(tail(8, 0), 8), "the window is full");
        assert!(rank_has_room(tail(8, 0), 0), "an unknown window");
        assert!(rank_has_room(tail(8, 0), -1), "an unknown window");
        assert_eq!(
            (rank_of(None), rank_of(Some(3)), rank_of(Some(u32::MAX))),
            (0, 3, 0)
        );
    }

    /// Each choice of a request credits its own rank: a fan-out placed per
    /// choice never lands choice `i`'s tokens on choice 0's rank, the first
    /// choice alone counts the prompt (once) and drops the pending entry, a
    /// request of more than one choice takes no prefill sample, and the
    /// single-choice form of the same submit arms the probe.
    #[test]
    fn choice_loads_credit_their_own_rank() {
        let loads = LoadTracker::default();
        let now = Instant::now();
        let probe = loads.submit(Submission {
            request_id: "r",
            rank: 0,
            prompt_tokens: 100,
            choices: 2,
            tail: None,
            window: 0,
            at: now,
        });
        assert_eq!(loads.pending(), 1);
        let mut first = ChoiceLoad::new(0, 0, probe);
        let mut second = ChoiceLoad::new(1, 1, PrefillProbe::default());
        assert_eq!((first.rank(), second.rank()), (0, 1));
        let later = now + Duration::from_millis(50);
        // Not the first choice: no prompt sample, the request stays pending.
        second.count_prompt(&loads, "r", 100, 0, later);
        assert_eq!(loads.pending(), 1);
        second.generated(&loads, 7, later);
        first.count_prompt(&loads, "r", 100, 40, later);
        // Once per request: a later output of the first choice adds nothing.
        first.count_prompt(&loads, "r", 100, 40, later);
        first.generated(&loads, 3, later);
        assert_eq!(loads.pending(), 0);
        assert_eq!(loads.prefill_samples(0), 0, "two choices never sample");
        let rank0 = loads.estimate_at(0, 0, later);
        let rank1 = loads.estimate_at(1, 0, later);
        // 3 tokens on rank 0 and 7 on rank 1 over the window, 40 of 100
        // prompt tokens cached on rank 0 and nothing counted on rank 1.
        assert_eq!((rank0.gen_throughput, rank1.gen_throughput), (1.5, 3.5));
        assert_eq!((rank0.cache_hit_rate, rank1.cache_hit_rate), (0.4, 0.0));
        let armed = loads.submit(Submission {
            request_id: "s",
            rank: 1,
            prompt_tokens: 100,
            choices: 1,
            tail: None,
            window: 0,
            at: later,
        });
        let mut only = ChoiceLoad::new(1, 0, armed);
        only.count_prompt(&loads, "s", 100, 0, later + Duration::from_millis(10));
        assert_eq!((loads.pending(), loads.prefill_samples(1)), (0, 1));
    }

    /// Two ranks serve: each rank's hit rate, throughput and queued work are
    /// its own, and a rank that served nothing reads as idle, not as its
    /// peer.
    #[test]
    fn each_rank_keeps_its_own_hit_rate_throughput_and_queue() {
        let tracker = LoadTracker::default();
        let now = Instant::now();
        tracker.submitted(0, "a", 1_000);
        tracker.submitted(1, "b", 400);
        tracker.submitted(1, "c", 600);
        tracker.first_output(0, "a", 1_000, 0);
        tracker.first_output(1, "b", 400, 400);
        tracker.generated_at(0, 100, now);
        tracker.generated_at(1, 900, now);

        let rank0 = tracker.estimate_at(0, 1, now);
        let rank1 = tracker.estimate_at(1, 1, now);
        assert_eq!((rank0.cache_hit_rate, rank1.cache_hit_rate), (0.0, 1.0));
        assert_eq!((rank0.gen_throughput, rank1.gen_throughput), (50.0, 450.0));
        // Rank 0 has nothing waiting; rank 1's `c` waits behind `b`, with
        // rank 1's own hit rate discounting it to nothing.
        assert_eq!((rank0.queued_token_work, rank1.queued_token_work), (0, 0));
        tracker.first_output(1, "b", 400, 0);
        assert_eq!(tracker.estimate_at(1, 1, now).queued_token_work, 300);
        assert_eq!(tracker.estimate_at(2, 1, now), LoadEstimate::default());
    }
}
