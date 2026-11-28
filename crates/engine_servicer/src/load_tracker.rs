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
//!   service-rate estimate, not GPU kernel time. Samples whose intervals
//!   overlap merge as they arrive (tokens summed, interval unioned, the count
//!   kept) into an entry that takes samples for one second after its first
//!   and is retained for 60 seconds after its newest; any other sample opens
//!   the next entry with its own interval, the union at read time counts a
//!   second once however many entries span it, a sample's seconds leave with
//!   its tokens and never before them, and the reading is the rank's rate
//!   over the last window of first outputs however long it has been busy;
//!   the ring of 64 entries outlasts the window; at least two samples are
//!   required. Missing engine timing, preemptions, zero-work hits and
//!   intervals shorter than 1 ms do not calibrate capacity. vLLM must enable
//!   `log_stats` to emit the scheduling events. A wire that carries no engine
//!   timing measures on the servicer's clock instead ([`PrefillProbe`]: every
//!   engine request's submit to its first output, once); the seconds a queued
//!   request waits, whatever holds the queue, are counted once by the union;
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

use crate::servicer::WireState;

/// How far back streamed tokens count toward the throughput.
pub(crate) const THROUGHPUT_WINDOW: Duration = Duration::from_secs(2);
/// How many first outputs the hit rate averages over.
pub(crate) const HIT_RATE_SAMPLES: usize = 64;
/// Bound the input-capacity estimator's ring of busy-period slices
/// independently of cache-hit history.
const PREFILL_ENTRIES: usize = 64;
/// How long an entry takes samples after its first; sealed after that.
const PREFILL_ENTRY_SPAN: Duration = Duration::from_secs(1);
/// How long an entry outlives its newest sample.
const PREFILL_SAMPLE_TTL: Duration = Duration::from_secs(60);
// The ring outlasts the window, so the window alone decides what the reading
// remembers of a rank that never idles.
const _: () =
    assert!(PREFILL_ENTRIES as u64 * PREFILL_ENTRY_SPAN.as_secs() >= PREFILL_SAMPLE_TTL.as_secs());

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
    /// The wire's id, shared with its guard.
    request_id: Arc<str>,
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
    /// Busy periods of computed prefill work, oldest first.
    prefills: VecDeque<PrefillEntry>,
}

/// Computed prefill work on a rank over one slice of a busy period: the
/// samples whose intervals overlapped, merged as they arrived for
/// [`PREFILL_ENTRY_SPAN`] after the first and sealed after that, so the
/// estimate remembers the window, not the busy period.
struct PrefillEntry {
    /// When the first sample arrived: the entry takes samples until
    /// [`PREFILL_ENTRY_SPAN`] after it.
    opened: Instant,
    /// When the newest sample arrived; the entry expires
    /// [`PREFILL_SAMPLE_TTL`] after it, and a sealed entry's stamp never
    /// moves.
    observed: Instant,
    samples: u32,
    tokens: u64,
    /// The union of the samples' intervals, on the samples' clock: one
    /// interval, since every merged sample overlapped it. Entries may
    /// overlap one another; the estimator unions them, and these seconds
    /// stay with these tokens until the entry expires.
    start: f64,
    end: f64,
}

impl PrefillEntry {
    /// Whether `start..end` overlaps this entry's interval (touching counts).
    fn overlaps(&self, start: f64, end: f64) -> bool {
        start <= self.end && end >= self.start
    }
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
            .retain(|entry| now.saturating_duration_since(entry.observed) <= PREFILL_SAMPLE_TTL);
        if self
            .prefills
            .iter()
            .map(|entry| u64::from(entry.samples))
            .sum::<u64>()
            < 2
        {
            return None;
        }
        let tokens: f64 = self.prefills.iter().map(|entry| entry.tokens as f64).sum();
        let mut intervals: Vec<_> = self
            .prefills
            .iter()
            .map(|entry| (entry.start, entry.end))
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

    /// A request with `prompt_tokens` was handed to `rank`: queued work
    /// there until its first output.
    pub(crate) fn submitted(&self, rank: i32, request_id: &Arc<str>, prompt_tokens: u32) {
        self.lock().pending.push_back(Pending {
            request_id: Arc::clone(request_id),
            prompt_tokens,
            rank,
        });
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
    /// engine had accepted the request) and its first output (`ended`); see
    /// [`PrefillProbe`] for what the interval holds. The sample goes through
    /// the same validity rule and estimator as an engine-timed one.
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

    /// Validate and retain a sample. It merges into the rank's newest entry
    /// while that entry takes samples and their intervals overlap (tokens
    /// over union is unchanged by the merge, since the union of two
    /// overlapping intervals is one interval); any other sample opens the
    /// next entry with its own interval, whether it reaches past the newest
    /// entry, lies wholly before it (an engine stream polled late) or spans
    /// idle seconds between entries that none of them owns. A start clipped
    /// to the newest entry's end would drop those seconds and keep the
    /// tokens, and read high.
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
        if let Some(entry) = figures.prefills.back_mut() {
            let open = now.saturating_duration_since(entry.opened) < PREFILL_ENTRY_SPAN;
            if open && entry.overlaps(start, end) {
                entry.observed = now;
                entry.samples = entry.samples.saturating_add(1);
                entry.tokens += u64::from(tokens);
                entry.start = entry.start.min(start);
                entry.end = entry.end.max(end);
                return;
            }
        }
        figures.prefills.push_back(PrefillEntry {
            opened: now,
            observed: now,
            samples: 1,
            tokens: u64::from(tokens),
            start,
            end,
        });
        if figures.prefills.len() > PREFILL_ENTRIES {
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

    /// The prefill samples `rank` holds, merged or not (expired entries
    /// included until the next estimate trims them).
    #[cfg(test)]
    pub(crate) fn prefill_samples(&self, rank: i32) -> usize {
        self.lock().ranks.get(&rank).map_or(0, |figures| {
            figures
                .prefills
                .iter()
                .map(|entry| entry.samples as usize)
                .sum()
        })
    }

    /// The busy periods (merged entries) `rank` holds.
    #[cfg(test)]
    pub(crate) fn prefill_entries(&self, rank: i32) -> usize {
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

/// One engine request's servicer-clock prefill sample on a wire without
/// engine timing: armed at the submit, taken once at the request's first
/// output with its prompt figures. The interval holds more than the prefill:
/// the two wire hops and the wait for the scheduler's current step
/// (near-constant); behind other requests, the queue, whose seconds the
/// union of a rank's samples counts once whatever holds it: the prefills of
/// the requests ahead (a burst then reads the capacity of the whole batch),
/// but also a decode peer's bootstrap or KV backpressure on a prefill-role
/// worker, or a full running window on a co-located worker, where the
/// figure follows that bottleneck; and on the workers the gateway reads this
/// field for, the prefill role of a disaggregated pair, the bootstrap wait
/// for the decode peer and the KV transfer, which grow with the prompt (an
/// encoder's embeddings awaited on a language-side worker likewise). The
/// reading is therefore low, never high: every second in the union is one
/// the rank spent on the counted work or on what held it, and nothing is
/// taken out of it (a sample's seconds stay with its tokens until its entry
/// expires), so tokens over seconds is at or below the rate in a steady
/// regime (by a fixed share on a co-located worker, a proportional one on a
/// prefill-role worker, the bottleneck's share under one, the queue's wait
/// under a deep one), and a shift in the mix leaves the reading within the
/// window, since an entry takes samples for a second and is forgotten a
/// window after its newest.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PrefillProbe {
    started: Option<Instant>,
}

impl PrefillProbe {
    /// Armed at `started`, the submit.
    pub(crate) fn armed(started: Instant) -> Self {
        Self {
            started: Some(started),
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
    pub(crate) request_id: &'a Arc<str>,
    /// The rank the request's first choice went to: the choice that reports
    /// the prompt, so the queued work and the hit rate land there.
    pub(crate) rank: i32,
    pub(crate) prompt_tokens: u32,
    /// The servicer's instant after the engine accepted the request.
    pub(crate) at: Instant,
}

impl LoadTracker {
    /// A request handed to the engine: queued as by [`Self::submitted`], and
    /// the servicer-clock probe each of its choices holds, armed at the
    /// submit (every choice's engine request prefills its own prompt and is
    /// its own sample, as on an engine-timed wire).
    pub(crate) fn submit(&self, submission: Submission<'_>) -> PrefillProbe {
        self.submitted(
            submission.rank,
            submission.request_id,
            submission.prompt_tokens,
        );
        PrefillProbe::armed(submission.at)
    }
}

/// One choice's share of the tracker's bookkeeping, held by the wire's
/// choice stream: the rank its own engine request went to (an unpinned
/// fan-out is placed per choice, so each credits its tokens where they were
/// made), the servicer-clock probe of its own prefill (unarmed on a wire
/// with engine timing) and whether the request's prompt was counted.
#[derive(Debug)]
pub(crate) struct ChoiceLoad {
    rank: i32,
    /// Whether this is the request's first choice, the one that reports its
    /// prompt (every choice of an `n > 1` fan-out reports the same one).
    first: bool,
    probe: PrefillProbe,
    prompt_counted: bool,
    /// Whether the choice is a client's, counted; the servicer's own health
    /// probe is not ([`Self::uncounted`]).
    counted: bool,
}

impl ChoiceLoad {
    pub(crate) fn new(rank: i32, index: u32, probe: PrefillProbe) -> Self {
        Self {
            rank,
            first: index == 0,
            probe,
            prompt_counted: false,
            counted: true,
        }
    }

    /// The share of a request the tracker never sees: the servicer's own
    /// health probe of the scheduler, whose one token on an idle rank is no
    /// measure of the rank's prefill capacity, its prompt no hit-rate sample
    /// and its token no generation of the rank's.
    pub(crate) fn uncounted() -> Self {
        Self {
            rank: 0,
            first: false,
            probe: PrefillProbe::default(),
            prompt_counted: true,
            counted: false,
        }
    }

    /// The rank the choice's engine request went to.
    pub(crate) fn rank(&self) -> i32 {
        self.rank
    }

    /// The choice's output with the request's prompt figures, read at
    /// `now`: once per request (from its first choice), the tracker learns
    /// that the engine started it and how much of the prompt was cached;
    /// once per choice, the probe reads the prefill it timed.
    pub(crate) fn count_prompt(
        &mut self,
        loads: &LoadTracker,
        request_id: &str,
        prompt_tokens: u32,
        cached_tokens: u32,
        now: Instant,
    ) {
        if !self.counted || self.prompt_counted || prompt_tokens == 0 {
            return;
        }
        if self.first {
            loads.first_output(self.rank, request_id, prompt_tokens, cached_tokens);
        }
        self.probe
            .first_output(loads, self.rank, prompt_tokens, cached_tokens, now);
        self.prompt_counted = true;
    }

    /// The choice streamed `tokens` to the client at `at`.
    pub(crate) fn generated(&self, loads: &LoadTracker, tokens: u32, at: Instant) {
        if self.counted {
            loads.generated_at(self.rank, tokens, at);
        }
    }
}

/// Takes a request out of the tracker's queue when its stream ends, in case
/// the engine never started it; the request's stream holds one.
pub(crate) struct PendingGuard<S: WireState> {
    state: Arc<S>,
    request_id: Arc<str>,
}

impl<S: WireState> PendingGuard<S> {
    pub(crate) fn new(state: Arc<S>, request_id: &Arc<str>) -> Self {
        Self {
            state,
            request_id: Arc::clone(request_id),
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
        .position(|pending| &*pending.request_id == request_id)
    {
        inner.pending.remove(index);
    }
}

#[cfg(test)]
#[path = "../tests/unit/load_tracker/tests.rs"]
mod tests;
