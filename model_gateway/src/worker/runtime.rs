//! A worker's shared runtime state: what a same-URL replacement inherits.
//!
//! [`WorkerRuntime`] keeps the lifecycle status, the load and processed
//! counters, the health-check counters, the routing-key load, the overload
//! and liveness vetoes with their clocks, the PD admission rooms with their
//! queue, and the prefill book the wedged rule's bound reads. The `Worker`
//! trait and `BasicWorker` in `basic.rs` read and write it.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicU8, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use openai_protocol::worker::WorkerStatus;
use parking_lot::Mutex;
use tokio::sync::Notify;

// The trait is named only by the doc links of this module.
#[cfg(doc)]
use super::basic::Worker;
use super::basic::WorkerRoutingKeyLoad;

/// Why the liveness tracker vetoes a worker (see the `liveness` module).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum StallReason {
    /// Its transport fails and nothing has been heard from it for the stall
    /// threshold.
    Unreachable = 1,
    /// It answers polls but holds requests that make no progress while its
    /// queue grows.
    Wedged = 2,
}

impl StallReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unreachable => "unreachable",
            Self::Wedged => "wedged",
        }
    }

    const fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Unreachable),
            2 => Some(Self::Wedged),
            _ => None,
        }
    }
}

/// One-shot routing snapshot — see [`Worker::routing_state`].
#[derive(Clone, Copy, Debug)]
pub struct RoutingState {
    /// `status == Ready`.
    pub healthy: bool,
    /// Circuit breaker permits execution (closed or half-open).
    pub can_execute: bool,
    /// Active in-flight request count.
    pub load: usize,
    /// Lifetime processed-request count (min-load tie-break).
    pub processed: usize,
    /// Absolute overload veto, set by the load monitor at ingestion time.
    pub overloaded: bool,
    /// Liveness veto: unreachable or wedged (see the `liveness` module).
    pub stalled: bool,
}

impl RoutingState {
    /// Every routing veto but the absolute overload one: ready, circuit
    /// breaker closed or half-open, not stalled. What a selection that lifted
    /// the overload veto routes on (`SelectWorkerInfo::overload_lifted`).
    pub const fn eligible_ignoring_overload(self) -> bool {
        self.healthy && self.can_execute && !self.stalled
    }

    /// The full routing eligibility test. Costs nothing beyond the reads the
    /// gather pass already performed: every field rides the one guard
    /// [`Worker::routing_state`] took.
    pub const fn eligible(self) -> bool {
        self.eligible_ignoring_overload() && !self.overloaded
    }
}

/// Shared mutable worker state preserved across same-URL replacements.
#[derive(Debug)]
pub struct WorkerRuntime {
    status: AtomicU8,
    cache_trace_epochs_valid: AtomicBool,
    consecutive_failures: AtomicUsize,
    consecutive_successes: AtomicUsize,
    total_pending_probes: AtomicUsize,
    load_counter: AtomicUsize,
    /// Bootstrap rooms the PD admission gate has claimed on this worker and
    /// not yet released. Separate from `load_counter` because admission must
    /// *claim* against the engine's running window rather than read it: two
    /// dispatches that both saw the last free slot would both send, which is
    /// the over-window burst the gate exists to prevent. Lives here so a
    /// same-URL replacement inherits the rooms the engine still holds.
    pd_admitted: AtomicUsize,
    /// The dispatches waiting at the PD admission gate for this worker's
    /// rooms, oldest first, so a room that frees goes to the one that has
    /// waited longest for it (see [`Self::pd_enqueue`]), each with the
    /// notifier its wait sleeps on. Kept with the rooms rather than in a
    /// registry of its own: the order lives and dies with the worker, and a
    /// same-URL replacement inherits it with them.
    pd_queue: Mutex<PdQueue>,
    processed_counter: AtomicUsize,
    worker_routing_key_load: WorkerRoutingKeyLoad,
    revision: AtomicU64,
    /// Absolute overload veto. Lives here rather than in the load-snapshot map
    /// so selection reads it under the guard it already holds, and so a
    /// same-URL replacement inherits it with the rest of the shared runtime.
    overloaded: AtomicBool,
    /// Liveness veto, a [`StallReason`] as `u8` (0 = none).
    stall: AtomicU8,
    /// The `smg_worker_state` series last published for the worker (the
    /// series table's index plus one; 0 = none yet), so the series the
    /// worker leaves is set to zero when the next is set to one. Shared
    /// with a same-URL replacement like the facts the series summarises.
    published_state: AtomicU8,
    /// Serializes the publishes of the `smg_worker_state` series, from the
    /// read of the facts to the last gauge write: two publishes racing on
    /// different codes would otherwise interleave their writes and leave two
    /// series at 1, or a stale code over a fresh one.
    publish_lock: Mutex<()>,
    /// Last successful contact and last token or completion, in
    /// [`super::liveness::now_ms`] milliseconds.
    last_contact_ms: AtomicU64,
    last_token_ms: AtomicU64,
    /// Waiting-queue depth from the previous load report.
    last_waiting_reqs: AtomicI64,
    /// A transport failure happened since the last contact.
    transport_failed: AtomicBool,
    /// The worker's KV-event stream has pushed a load record: silence is
    /// then judged by the stream's cadence (see [`super::liveness`]).
    pushed_loads: AtomicBool,
    /// The latest pushed load record reports the engine silent on its
    /// in-flight requests past the wedge bound (see [`super::liveness`]).
    engine_stalled: AtomicBool,
    /// In-flight count at the previous liveness sweep.
    last_load_sample: AtomicUsize,
    /// Woken on every contact, so a loop backing off from this worker (the
    /// KV event subscriber) retries the moment the worker is heard from.
    contact_wake: Arc<Notify>,
    /// When the worker last became routable (promoted to Ready, or a liveness
    /// veto cleared), in [`super::liveness::now_ms`] milliseconds.
    admitted_at_ms: AtomicU64,
    /// The warm-up slice's baseline: the admission it was taken for and the
    /// blocks the index held for the worker then (see [`Self::warmup_growth`]).
    warmup_base_admitted_ms: AtomicU64,
    warmup_base_blocks: AtomicUsize,
    /// Clock before which this thin worker, with requests in flight, takes
    /// no further diverted hit.
    divert_until_ms: AtomicU64,
    /// When the current run of in-flight requests began (the load counter
    /// left zero), in [`super::liveness::now_ms`] milliseconds; zero while
    /// idle. The no-progress clock of the wedged rule starts here, not at
    /// registration or the last token before an idle spell.
    busy_since_ms: AtomicU64,
    /// Requests in flight whose responses the gateway sees one by one (see
    /// [`Worker::tracked_load`]).
    tracked_in_flight: AtomicUsize,
    /// Open streams that have produced a token or completion (see
    /// [`Worker::output_silence`]).
    producing_streams: AtomicUsize,
    /// The prefill book: pending prompt tokens, the busy interval and the
    /// rate window, moved together under one lock so a request's end and
    /// another's start cannot race the interval away (see [`PrefillBook`]).
    prefill: Mutex<PrefillBook>,
}

/// Prefill rate assumed for a worker that has not shown one yet: slow enough
/// that a cold engine handed a burst of long prompts is not called wedged
/// before it can have answered (10k tokens/s is a small model on a modest
/// GPU; a current datacenter GPU prefills 8B weights at ~90k).
pub(crate) const COLD_PREFILL_TOKENS_PER_SEC: u64 = 10_000;

/// First tokens are summed over windows at least this long before they make
/// a rate sample, so the sample is the engine's aggregate throughput and not
/// one request's time to first token (which includes its wait in the batch).
const PREFILL_WINDOW_MS: u64 = 1_000;

/// The busy time the busy-time average behind the prefill rate holds (see
/// [`PrefillBook`]): past it, the average's tokens and busy time are scaled
/// back to it together, so older windows weigh less as the engine works on.
/// Long enough to hold the prefill of prompts answered together (long
/// prompts prefilled side by side for many seconds; the longer their
/// prefill against the span, the further their answers lift the average
/// above the engine's throughput), short enough that a slow spell (a kernel
/// the engine built) fades within a minute or two of busy time.
///
/// That fade is the price of the cap, on every worker whose prefill is
/// timed and in every rule that reads its rate (the pile rule's bound, the
/// engine-reported allowance, the PD sides' bounds): after a slow spell the
/// rate climbs back at the average's pace, not the halved sample's, in
/// about 50 s of busy time from half the engine's throughput to within a
/// tenth of it, where the halved sample took three windows. Counted in busy
/// time, the climb is longer on the clock on a lightly loaded worker (over
/// three minutes, busy a quarter of the time). Meanwhile those bounds give
/// the engine more time than it needs, never past their 120 s cap: patience,
/// where an overestimate fails healthy requests.
const PREFILL_AVERAGE_SPAN_MS: u64 = 30_000;

/// The prefill books of one worker: what is pending, how long the engine
/// has been prefilling, and the rate window. One struct under one lock, so
/// that a request's end racing another's start cannot lose the busy
/// interval (a lost interval would leave the window open for good on a
/// worker that never drains, and its rate stale). A handoff a decode side
/// showed is evidence for the prefill side rule only.
///
/// The rate is measured over busy time only: time with timed prompt tokens
/// pending on the worker (overlapping prompts share one interval), banked
/// when the last of them ends. First tokens are summed over windows of at
/// least [`PREFILL_WINDOW_MS`] of busy time, and a window closes at the
/// first token that completes it, whether or not the pending count ever
/// touches zero: a worker that is never idle still updates its rate at
/// every first token past a second of prefill.
///
/// Each window's sample halves into the estimate, which never exceeds the
/// busy-time average: the tokens of the windows that made samples over
/// their busy time, scaled back together to [`PREFILL_AVERAGE_SPAN_MS`] of
/// it once they hold more. A window counts the first tokens that came back
/// in it, and their prompts may have been prefilled in the windows before:
/// prompts answered together (a batch the engine finishes at once, long
/// prompts prefilled side by side) leave the windows that timed their
/// prefill short of their tokens and put them in the window they come back
/// in, a second or so long. Halved into the estimate, that window's sample
/// put it at several times the engine's throughput (over four times, for
/// two 90k prompts answered together after 10 s beside a 2k prompt a
/// second), and every bound read from it short of the prefill it allows
/// for: a prefill side's request failed while the engine was still
/// prefilling. The average divides those tokens by the busy time that held
/// their prefill, as far as it still holds that time: until the book has
/// held the span, it never exceeds the engine's throughput. Once older
/// windows are scaled back, a prefill's busy time weighs less than the
/// tokens it comes back with, and the average reads above the throughput
/// by a margin that grows with the prefill's length against the span. On a
/// book that has held the span: 4% for the pair above, a quarter for two
/// 245k prompts prefilled side by side for 27 s, over four fifths for four
/// of them (55 s), where the halved sample read the same shapes at 4.6 to
/// 24.6 times. An engine that slows down still pulls the estimate down at
/// its next sample.
///
/// Only the prompts whose first token the gateway will see (streaming
/// generations) are timed. A non-streaming generation answers once, at its
/// end, so its prompt stays pending through its whole decode: it counts in
/// the backlog (the engine prefills it before anything dispatched behind it
/// streams) but opens no busy interval and makes no sample, or the idle
/// gaps around it would count as prefill again and its decode would be
/// timed as the prefill of its prompt.
///
/// A first token teaches the tokens the engine computed for its prompt:
/// the prompt's length less the prefix-cache hit the engine reports with
/// it (the servicers forward the engine's per-request cached-token count
/// on every chunk; one that reports none teaches the full length). Counted
/// at full length, a worker fed mostly cache hits (a multi-turn chat)
/// learned a rate many times its cold one, and a later burst of cold
/// prompts got a bound at the bare threshold: the false positive the bound
/// exists to prevent. Two things still err toward patience: the backlog
/// counts a pending prompt at full length, since its hit is known only
/// when the engine answers; and the time to first token of a mostly cached
/// prompt is the engine's step latency more than prefill, so a worker fed
/// mostly hits learns a rate below its cold one. An underestimate gives
/// way as the engine prefills cold prompts; an overestimate vetoes a
/// healthy worker, and nothing corrects it.
#[derive(Debug, Default)]
struct PrefillBook {
    /// Prompt tokens dispatched to the worker and not yet answered with a
    /// first token or, for a non-streaming generation, with its one answer
    /// (or released by a dropped stream): the backlog.
    pending: u64,
    /// The part of `pending` whose prefill is timed: the prompts whose
    /// first token the gateway will see.
    timed: u64,
    /// Observed prefill rate in tokens per second; zero until a window of
    /// first tokens has been seen. Never above the busy-time average.
    rate_tps: u64,
    /// The busy-time average: the tokens of the windows that made samples
    /// and their busy time, scaled back together to
    /// [`PREFILL_AVERAGE_SPAN_MS`] of busy time once they hold more.
    average_tokens: u64,
    average_busy_ms: u64,
    /// Gateway clock at which the timed tokens left zero; `None` while
    /// none is pending.
    busy_since_ms: Option<u64>,
    /// Busy time banked into the open window from closed intervals.
    window_busy_ms: u64,
    /// First tokens' prompt tokens in the open window.
    window_tokens: u64,
    /// Gateway clock of the latest first response (or one answer) to a
    /// prompt dispatched here; zero until one.
    last_answer_ms: u64,
    /// Gateway clock of the latest handoff of a prompt dispatched here that
    /// a decode side showed (see [`Worker::prefill_handed_off_ago`]); zero
    /// until one. No answer: it releases no prompt, times nothing, answers
    /// no run and shows no shape.
    last_handoff_ms: u64,
    /// The prompts pending, prompt tokens to how many of that length, for
    /// the longest of them: the largest shape the engine is on.
    pending_prompts: BTreeMap<u64, u32>,
    /// The longest prompt answered with a first token (or one answer): the
    /// largest shape the engine has shown it runs; zero until one.
    longest_answered: u64,
    /// Gateway clock at which `pending` last left zero: the start of the
    /// current run of prompts waiting on the worker; zero while none is.
    run_since_ms: u64,
    /// Whether a prompt was answered since the current run began.
    run_answered: bool,
    /// How long the runs since the latest answer held prompts and ended
    /// without answering any (given up on, or dropped): silence that the
    /// idle spells between them do not erase. Zero after an answer.
    unanswered_ms: u64,
}

impl PrefillBook {
    fn started(&mut self, tokens: u64, timed: bool, now_ms: u64) {
        if tokens == 0 {
            return;
        }
        if self.pending == 0 {
            self.run_since_ms = now_ms.max(1);
            self.run_answered = false;
        }
        self.pending = self.pending.saturating_add(tokens);
        *self.pending_prompts.entry(tokens).or_insert(0) += 1;
        if !timed {
            return;
        }
        if self.timed == 0 {
            self.busy_since_ms = Some(now_ms);
        }
        self.timed = self.timed.saturating_add(tokens);
    }

    fn ended(&mut self, tokens: u64, timed: bool, cached: Option<u64>, now_ms: u64) {
        self.pending = self.pending.saturating_sub(tokens);
        self.forget_prompt(tokens);
        if cached.is_some() {
            self.last_answer_ms = now_ms;
            self.longest_answered = self.longest_answered.max(tokens);
            self.run_answered = true;
            self.unanswered_ms = 0;
        }
        if self.pending == 0 && self.run_since_ms > 0 {
            // A run that ends on a release with no answer leaves the time it
            // held prompts unanswered behind, for the next run to start from.
            if cached.is_none() {
                let since = self.last_answer_ms.max(self.run_since_ms);
                self.unanswered_ms = self
                    .unanswered_ms
                    .saturating_add(now_ms.saturating_sub(since));
            }
            self.run_since_ms = 0;
        }
        if !timed {
            return;
        }
        self.timed = self.timed.saturating_sub(tokens);
        if tokens > 0 {
            if let Some(cached) = cached {
                // What the engine computed: the prompt less its cache hit.
                self.observe(tokens.saturating_sub(cached), now_ms);
            }
        }
        if self.timed == 0 {
            if let Some(since) = self.busy_since_ms.take() {
                self.window_busy_ms = self
                    .window_busy_ms
                    .saturating_add(now_ms.saturating_sub(since));
            }
        }
    }

    /// Busy time in the open window as of `now_ms`: what is banked plus the
    /// interval still running.
    fn window_span_ms(&self, now_ms: u64) -> u64 {
        let open = self
            .busy_since_ms
            .map_or(0, |since| now_ms.saturating_sub(since));
        self.window_busy_ms.saturating_add(open)
    }

    /// Fold `tokens` computed for a prompt whose first token came back at
    /// `now_ms` into the observed rate: once the window holds
    /// [`PREFILL_WINDOW_MS`] of busy time its tokens over that time are a
    /// sample, and each sample halves into the running estimate, held at
    /// the busy-time average (see [`Self::average_with`]). A window in
    /// which the engine computed nothing (every prompt a whole cache hit)
    /// makes no sample.
    fn observe(&mut self, tokens: u64, now_ms: u64) {
        self.window_tokens = self.window_tokens.saturating_add(tokens);
        let span_ms = self.window_span_ms(now_ms);
        if span_ms < PREFILL_WINDOW_MS {
            return;
        }
        if self.window_tokens > 0 {
            let sample = self.window_tokens.saturating_mul(1000) / span_ms;
            let estimate = if self.rate_tps == 0 {
                sample
            } else {
                self.rate_tps.midpoint(sample)
            };
            let average = self.average_with(self.window_tokens, span_ms);
            self.rate_tps = estimate.min(average).max(1);
        }
        self.window_busy_ms = 0;
        self.window_tokens = 0;
        if self.busy_since_ms.is_some() {
            // Still prefilling: the next window's busy time starts here.
            self.busy_since_ms = Some(now_ms);
        }
    }

    /// Add a window of `tokens` over `busy_ms` (at least
    /// [`PREFILL_WINDOW_MS`]) to the busy-time average and return the
    /// average in tokens per second. Past [`PREFILL_AVERAGE_SPAN_MS`] of
    /// busy time both sums are then scaled back to it together: the
    /// average stands, and the windows to come weigh more against it.
    fn average_with(&mut self, tokens: u64, busy_ms: u64) -> u64 {
        self.average_tokens = self.average_tokens.saturating_add(tokens);
        self.average_busy_ms = self.average_busy_ms.saturating_add(busy_ms);
        let average = self.average_tokens.saturating_mul(1000) / self.average_busy_ms;
        if self.average_busy_ms > PREFILL_AVERAGE_SPAN_MS {
            self.average_tokens =
                self.average_tokens.saturating_mul(PREFILL_AVERAGE_SPAN_MS) / self.average_busy_ms;
            self.average_busy_ms = PREFILL_AVERAGE_SPAN_MS;
        }
        average
    }

    /// The time to prefill what is pending plus `extra` tokens at the
    /// observed rate (the cold prior until one is observed).
    fn eta(&self, extra: u64) -> Duration {
        let pending = self.pending.saturating_add(extra);
        if pending == 0 {
            return Duration::ZERO;
        }
        let rate = match self.rate_tps {
            0 => COLD_PREFILL_TOKENS_PER_SEC,
            rate => rate,
        };
        Duration::from_millis(pending.saturating_mul(1000) / rate)
    }

    /// How long before `now_ms` the latest prompt was answered; `None`
    /// until one has been.
    fn answered_ago(&self, now_ms: u64) -> Option<Duration> {
        (self.last_answer_ms > 0)
            .then(|| Duration::from_millis(now_ms.saturating_sub(self.last_answer_ms)))
    }

    /// A decode side showed a handoff at `now_ms`; an older stamp never
    /// moves the latest back.
    fn handed_off(&mut self, now_ms: u64) {
        self.last_handoff_ms = self.last_handoff_ms.max(now_ms);
    }

    /// How long before `now_ms` a decode side last showed a handoff; `None`
    /// until one has.
    fn handed_off_ago(&self, now_ms: u64) -> Option<Duration> {
        (self.last_handoff_ms > 0)
            .then(|| Duration::from_millis(now_ms.saturating_sub(self.last_handoff_ms)))
    }

    /// How long, as of `now_ms`, the current run of pending prompts has
    /// waited for its first answer, plus what earlier runs since the latest
    /// answer held prompts unanswered: the worker's silence with the spells
    /// it held nothing left out (an idle worker owes no answer). `None`
    /// while nothing is pending or once the run has had an answer.
    fn unanswered_run(&self, now_ms: u64) -> Option<Duration> {
        (self.run_since_ms > 0 && !self.run_answered).then(|| {
            Duration::from_millis(
                self.unanswered_ms
                    .saturating_add(now_ms.saturating_sub(self.run_since_ms)),
            )
        })
    }

    /// Whether the engine is on a shape it has not shown it runs: a prompt
    /// pending here, or one of `extra` tokens about to be dispatched, longer
    /// than any answered, on a worker that has answered one. A worker that
    /// has answered nothing has shown no shape to go by.
    fn cold_shape(&self, extra: u64) -> bool {
        let longest_pending = self
            .pending_prompts
            .keys()
            .next_back()
            .copied()
            .unwrap_or(0);
        self.longest_answered > 0 && longest_pending.max(extra) > self.longest_answered
    }

    /// A prompt of `tokens` is no longer pending; one that was never booked
    /// (a release with tokens no start had) changes nothing.
    fn forget_prompt(&mut self, tokens: u64) {
        if let Some(count) = self.pending_prompts.get_mut(&tokens) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.pending_prompts.remove(&tokens);
            }
        }
    }
}

/// The PD admission gate's queue on one worker (see [`Worker::pd_enqueue`]):
/// the places outstanding, oldest first, each with the rooms it needs and
/// the notifier its wait sleeps on, and the next place number to hand out. A
/// place is handed out and queued under the one lock, so the queue is always
/// in arrival order.
#[derive(Debug, Default)]
struct PdQueue {
    next: u64,
    waiting: VecDeque<PdPlace>,
}

#[derive(Debug)]
struct PdPlace {
    place: u64,
    rooms: usize,
    wake: Arc<Notify>,
}

impl PdQueue {
    /// Whether `place` may claim: no older place needs as few rooms as it
    /// does, or fewer. Requests of a size are served in arrival order, and a
    /// request that needs more rooms than the ones behind it never idles the
    /// window waiting for them to free all at once: they pass it, and it
    /// claims when its rooms are free together, as it did before the queue
    /// existed. A place that is not queued may not claim.
    fn may_claim(&self, place: u64) -> bool {
        let mut fewest_ahead = usize::MAX;
        for waiting in &self.waiting {
            if waiting.place == place {
                return waiting.rooms < fewest_ahead;
            }
            fewest_ahead = fewest_ahead.min(waiting.rooms);
        }
        false
    }

    /// The notifiers of the places that may claim now (see
    /// [`Self::may_claim`]): the front, and behind it each place that needs
    /// fewer rooms than every place ahead of it. Woken when rooms are
    /// released and when one of them leaves; the others have nothing to try.
    fn claimable_wakes(&self) -> Vec<Arc<Notify>> {
        let mut fewest_ahead = usize::MAX;
        let mut wakes = Vec::new();
        for waiting in &self.waiting {
            if waiting.rooms < fewest_ahead {
                wakes.push(Arc::clone(&waiting.wake));
                fewest_ahead = waiting.rooms;
            }
        }
        wakes
    }
}

impl WorkerRuntime {
    pub fn new(url: &str, initial_status: WorkerStatus) -> Self {
        Self {
            status: AtomicU8::new(initial_status as u8),
            cache_trace_epochs_valid: AtomicBool::new(!matches!(
                initial_status,
                WorkerStatus::NotReady | WorkerStatus::Failed
            )),
            consecutive_failures: AtomicUsize::new(0),
            consecutive_successes: AtomicUsize::new(0),
            total_pending_probes: AtomicUsize::new(0),
            load_counter: AtomicUsize::new(0),
            pd_admitted: AtomicUsize::new(0),
            pd_queue: Mutex::new(PdQueue::default()),
            processed_counter: AtomicUsize::new(0),
            worker_routing_key_load: WorkerRoutingKeyLoad::new(url),
            revision: AtomicU64::new(0),
            overloaded: AtomicBool::new(false),
            stall: AtomicU8::new(0),
            published_state: AtomicU8::new(0),
            publish_lock: Mutex::new(()),
            last_contact_ms: AtomicU64::new(super::liveness::now_ms()),
            last_token_ms: AtomicU64::new(super::liveness::now_ms()),
            last_waiting_reqs: AtomicI64::new(0),
            transport_failed: AtomicBool::new(false),
            pushed_loads: AtomicBool::new(false),
            engine_stalled: AtomicBool::new(false),
            last_load_sample: AtomicUsize::new(0),
            contact_wake: Arc::new(Notify::new()),
            admitted_at_ms: AtomicU64::new(super::liveness::now_ms()),
            warmup_base_admitted_ms: AtomicU64::new(u64::MAX),
            warmup_base_blocks: AtomicUsize::new(0),
            divert_until_ms: AtomicU64::new(0),
            busy_since_ms: AtomicU64::new(0),
            tracked_in_flight: AtomicUsize::new(0),
            producing_streams: AtomicUsize::new(0),
            prefill: Mutex::new(PrefillBook::default()),
        }
    }

    pub fn note_admitted(&self) {
        self.admitted_at_ms
            .store(super::liveness::now_ms(), Ordering::Relaxed);
    }

    pub fn admitted_age(&self) -> Duration {
        Self::age_of(self.admitted_at_ms.load(Ordering::Relaxed))
    }

    pub fn divert_until_ms(&self) -> u64 {
        self.divert_until_ms.load(Ordering::Relaxed)
    }

    pub fn note_diverted(&self, until_ms: u64) {
        self.divert_until_ms.store(until_ms, Ordering::Relaxed);
    }

    /// Blocks the index has gained for the worker since its current admission.
    /// The index size itself is no measure of a returned worker's cache: a
    /// restarted engine's stale blocks stay indexed until its first batch
    /// reveals the restart, and an engine that gets no request sends no batch,
    /// so the stale count would end the warm-up meant to break that circle.
    /// The baseline is the count first seen for the current admission, and
    /// drops to zero when the index was cleared underneath (the count went
    /// down). Two racing callers may both take the same baseline; nothing
    /// worse.
    pub fn warmup_growth(&self, indexed: usize) -> usize {
        let admitted = self.admitted_at_ms.load(Ordering::Relaxed);
        if self.warmup_base_admitted_ms.load(Ordering::Relaxed) != admitted {
            self.warmup_base_admitted_ms
                .store(admitted, Ordering::Relaxed);
            self.warmup_base_blocks.store(indexed, Ordering::Relaxed);
            return 0;
        }
        let base = self.warmup_base_blocks.load(Ordering::Relaxed);
        if indexed < base {
            self.warmup_base_blocks.store(0, Ordering::Relaxed);
            return indexed;
        }
        indexed - base
    }

    // ── Liveness ────────────────────────────────────────────────────

    pub fn note_transport_failure(&self) {
        self.transport_failed.store(true, Ordering::Relaxed);
    }

    pub fn transport_failure_pending(&self) -> bool {
        self.transport_failed.load(Ordering::Relaxed)
    }

    pub fn stall_reason(&self) -> Option<StallReason> {
        StallReason::from_u8(self.stall.load(Ordering::Acquire))
    }

    /// Set or clear the veto; `true` when it changed.
    pub fn set_stall(&self, reason: Option<StallReason>) -> bool {
        let next = reason.map_or(0, |reason| reason as u8);
        self.stall.swap(next, Ordering::AcqRel) != next
    }

    /// The code of the `smg_worker_state` series published last (0 when
    /// none was).
    pub fn published_state(&self) -> u8 {
        self.published_state.load(Ordering::Acquire)
    }

    /// Remember `code` as the published `smg_worker_state` series; the code
    /// published before it comes back (0 when none was). Called under
    /// [`Self::publish_lock`].
    pub fn swap_published_state(&self, code: u8) -> u8 {
        self.published_state.swap(code, Ordering::AcqRel)
    }

    /// The lock a publish of the `smg_worker_state` series holds from the
    /// read of the facts to its last gauge write.
    pub fn publish_lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.publish_lock.lock()
    }

    pub fn note_contact(&self) {
        self.last_contact_ms
            .store(super::liveness::now_ms(), Ordering::Relaxed);
        self.transport_failed.store(false, Ordering::Relaxed);
        self.contact_wake.notify_one();
    }

    pub fn note_token_progress(&self) {
        let now = super::liveness::now_ms();
        // The stamps are millisecond stamps and the wake keeps one permit, so
        // a token within the millisecond of the previous one has nothing to
        // record: it leaves with one plain load. Every stream of the worker
        // reports here for every token, from whichever core runs it; the
        // three stores and the notifier's compare-exchange would otherwise
        // bounce the worker's cache lines between cores once per token.
        if self.last_token_ms.load(Ordering::Relaxed) == now
            && !self.transport_failed.load(Ordering::Relaxed)
        {
            return;
        }
        self.last_token_ms.store(now, Ordering::Relaxed);
        self.last_contact_ms.store(now, Ordering::Relaxed);
        self.transport_failed.store(false, Ordering::Relaxed);
        self.contact_wake.notify_one();
    }

    /// The notifier [`Self::note_contact`] fires; `notify_one` semantics, so
    /// a contact that happens before anyone waits is not lost.
    pub fn contact_wake(&self) -> Arc<Notify> {
        Arc::clone(&self.contact_wake)
    }

    pub fn contact_age(&self) -> Duration {
        Self::age_of(self.last_contact_ms.load(Ordering::Relaxed))
    }

    pub fn note_load_record(&self) {
        self.pushed_loads.store(true, Ordering::Relaxed);
    }

    pub fn pushes_load_records(&self) -> bool {
        self.pushed_loads.load(Ordering::Relaxed)
    }

    pub fn note_engine_stall(&self, stalled: bool) {
        self.engine_stalled.store(stalled, Ordering::Relaxed);
    }

    pub fn engine_stalled(&self) -> bool {
        self.engine_stalled.load(Ordering::Relaxed)
    }

    /// Time without a token or completion, counted from the later of the last
    /// one and the start of the current run of in-flight requests: a worker
    /// that was idle (or just registered) has nothing to show progress on
    /// until something is dispatched to it.
    pub fn token_progress_age(&self) -> Duration {
        Self::age_of(self.progress_reference_ms())
    }

    /// The stamp [`Self::token_progress_age`] counts from.
    pub fn progress_reference_ms(&self) -> u64 {
        self.last_token_ms
            .load(Ordering::Relaxed)
            .max(self.busy_since_ms.load(Ordering::Relaxed))
    }

    // ── Prefill backlog (the wedged rule's bound) ───────────────────

    pub fn note_prefill_started(&self, tokens: u64, timed: bool) {
        self.note_prefill_started_at(tokens, timed, super::liveness::now_ms().max(1));
    }

    /// [`Self::note_prefill_started`] at `now_ms` on the gateway clock.
    pub fn note_prefill_started_at(&self, tokens: u64, timed: bool, now_ms: u64) {
        self.prefill.lock().started(tokens, timed, now_ms);
    }

    pub fn note_prefill_ended(&self, tokens: u64, timed: bool, cached: Option<u64>) {
        self.note_prefill_ended_at(tokens, timed, cached, super::liveness::now_ms().max(1));
    }

    /// [`Self::note_prefill_ended`] at `now_ms` on the gateway clock.
    pub fn note_prefill_ended_at(
        &self,
        tokens: u64,
        timed: bool,
        cached: Option<u64>,
        now_ms: u64,
    ) {
        self.prefill.lock().ended(tokens, timed, cached, now_ms);
    }

    pub fn prefill_rate_tps(&self) -> u64 {
        self.prefill.lock().rate_tps
    }

    pub fn prefill_backlog(&self) -> Duration {
        self.prefill.lock().eta(0)
    }

    pub fn prefill_eta(&self, tokens: u64) -> Duration {
        self.prefill.lock().eta(tokens)
    }

    pub fn prefill_answered_ago(&self) -> Option<Duration> {
        self.prefill_answered_ago_at(super::liveness::now_ms().max(1))
    }

    /// [`Self::prefill_answered_ago`] as of `now_ms` on the gateway clock.
    pub fn prefill_answered_ago_at(&self, now_ms: u64) -> Option<Duration> {
        self.prefill.lock().answered_ago(now_ms)
    }

    /// A decode side showed a handoff of this worker's (see
    /// [`Worker::note_prefill_handoff`]): stamped apart from the answers.
    pub fn note_prefill_handoff(&self) {
        self.note_prefill_handoff_at(super::liveness::now_ms().max(1));
    }

    /// [`Self::note_prefill_handoff`] at `now_ms` on the gateway clock.
    pub fn note_prefill_handoff_at(&self, now_ms: u64) {
        self.prefill.lock().handed_off(now_ms);
    }

    pub fn prefill_handed_off_ago(&self) -> Option<Duration> {
        self.prefill_handed_off_ago_at(super::liveness::now_ms().max(1))
    }

    /// [`Self::prefill_handed_off_ago`] as of `now_ms` on the gateway clock.
    pub fn prefill_handed_off_ago_at(&self, now_ms: u64) -> Option<Duration> {
        self.prefill.lock().handed_off_ago(now_ms)
    }

    pub fn prefill_cold_shape(&self, tokens: u64) -> bool {
        self.prefill.lock().cold_shape(tokens)
    }

    pub fn prefill_unanswered_run(&self) -> Option<Duration> {
        self.prefill_unanswered_run_at(super::liveness::now_ms().max(1))
    }

    /// [`Self::prefill_unanswered_run`] as of `now_ms` on the gateway clock.
    pub fn prefill_unanswered_run_at(&self, now_ms: u64) -> Option<Duration> {
        self.prefill.lock().unanswered_run(now_ms)
    }

    pub fn swap_waiting_reqs(&self, waiting: i64) -> i64 {
        self.last_waiting_reqs.swap(waiting, Ordering::Relaxed)
    }

    pub fn swap_load_sample(&self, load: usize) -> usize {
        self.last_load_sample.swap(load, Ordering::Relaxed)
    }

    fn age_of(stamp_ms: u64) -> Duration {
        Duration::from_millis(super::liveness::now_ms().saturating_sub(stamp_ms))
    }

    // ── Lifecycle status ────────────────────────────────────────────

    pub fn status(&self) -> WorkerStatus {
        WorkerStatus::from_u8(self.status.load(Ordering::Acquire))
    }

    /// Whether the cache-trace epochs the worker's labels advertise are
    /// current: cleared when it leaves the routable states (see
    /// [`Self::set_status`]).
    pub(super) fn cache_trace_epochs_valid(&self) -> bool {
        self.cache_trace_epochs_valid.load(Ordering::Acquire)
    }

    pub fn set_status(&self, status: WorkerStatus) {
        if matches!(status, WorkerStatus::NotReady | WorkerStatus::Failed) {
            self.cache_trace_epochs_valid
                .store(false, Ordering::Release);
        }
        self.status.store(status as u8, Ordering::Release);
    }

    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    pub fn bump_revision(&self) -> u64 {
        self.revision.fetch_add(1, Ordering::AcqRel) + 1
    }

    // ── Health-check counters ───────────────────────────────────────

    pub fn consecutive_failures_increment(&self) -> usize {
        self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn consecutive_failures_reset(&self) {
        self.consecutive_failures.store(0, Ordering::Release);
    }

    pub fn consecutive_successes_increment(&self) -> usize {
        self.consecutive_successes.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn consecutive_successes_reset(&self) {
        self.consecutive_successes.store(0, Ordering::Release);
    }

    pub fn total_pending_probes(&self) -> usize {
        self.total_pending_probes.load(Ordering::Relaxed)
    }

    pub fn total_pending_probes_increment(&self) -> usize {
        self.total_pending_probes.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn total_pending_probes_reset(&self) {
        self.total_pending_probes.store(0, Ordering::Relaxed);
    }

    // ── Load counter ────────────────────────────────────────────────

    pub fn load(&self) -> usize {
        self.load_counter.load(Ordering::Relaxed)
    }

    pub fn increment_load(&self) {
        if self.load_counter.fetch_add(1, Ordering::Relaxed) == 0 {
            self.note_busy_started();
        }
    }

    pub fn try_increment_load(&self, max: usize) -> bool {
        match self
            .load_counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1).filter(|next| *next <= max)
            }) {
            Ok(0) => {
                self.note_busy_started();
                true
            }
            Ok(_) => true,
            Err(_) => false,
        }
    }

    /// A tracked request started. The wedged rule's clock starts with the
    /// first of a run, so an untracked request dispatched earlier (an HTTP or
    /// a non-streaming one) cannot age the run before it begins.
    pub fn note_tracked_started(&self) {
        if self.tracked_in_flight.fetch_add(1, Ordering::Relaxed) == 0 {
            self.note_busy_started();
        }
    }
    pub fn note_tracked_ended(&self) {
        let _ =
            self.tracked_in_flight
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |tracked| {
                    Some(tracked.saturating_sub(1))
                });
    }
    pub fn tracked_load(&self) -> usize {
        self.tracked_in_flight.load(Ordering::Relaxed)
    }
    /// A stream from this worker produced its first token or completion.
    pub fn note_output_started(&self) {
        self.producing_streams.fetch_add(1, Ordering::Relaxed);
    }
    /// A stream that had produced output ended.
    pub fn note_output_ended(&self) {
        let _ = self
            .producing_streams
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                Some(n.saturating_sub(1))
            });
    }
    /// See [`Worker::output_silence`]: the age of the last token or
    /// completion while a stream that produced output is still open.
    pub fn output_silence(&self) -> Option<Duration> {
        (self.producing_streams.load(Ordering::Relaxed) > 0)
            .then(|| Self::age_of(self.last_token_ms.load(Ordering::Relaxed)))
    }
    /// The load counter left zero: the wedged rule's clock starts now.
    fn note_busy_started(&self) {
        self.busy_since_ms
            .store(super::liveness::now_ms().max(1), Ordering::Relaxed);
    }

    /// Saturating decrement. Returns `true` if the counter was decremented,
    /// `false` if it was already zero — callers can log when that happens.
    pub fn try_decrement_load(&self) -> bool {
        match self
            .load_counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(1)
            }) {
            Ok(1) => {
                // Idle again: nothing in flight to wait for.
                self.busy_since_ms.store(0, Ordering::Relaxed);
                true
            }
            Ok(_) => true,
            Err(_) => false,
        }
    }

    // ── PD admission claims ─────────────────────────────────────────

    pub fn pd_admitted(&self) -> usize {
        self.pd_admitted.load(Ordering::Relaxed)
    }

    /// Claim `count` rooms, but only while the claimed total stays within
    /// `window`. All-or-nothing: a refusal claims nothing.
    pub fn try_admit_pd(&self, count: usize, window: usize) -> bool {
        self.pd_admitted
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |admitted| {
                admitted.checked_add(count).filter(|total| *total <= window)
            })
            .is_ok()
    }

    /// Release `count` claimed rooms, saturating at zero, and wake the waits
    /// that may claim now.
    pub fn release_pd(&self, count: usize) {
        let _ = self
            .pd_admitted
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |admitted| {
                Some(admitted.saturating_sub(count))
            });
        self.wake_claimable();
    }

    /// Wake the places that may claim now; the others would find nothing to
    /// do. One wakeup per such place, not one per waiter: `notify_one` keeps
    /// a permit for a wait that is between tries, so none is lost.
    fn wake_claimable(&self) {
        let wakes = self.pd_queue.lock().claimable_wakes();
        for wake in wakes {
            wake.notify_one();
        }
    }

    /// Join the admission queue at the back, needing `rooms`; the place comes
    /// back through [`Self::pd_dequeue`] on every way out of the wait.
    pub fn pd_enqueue(&self, rooms: usize) -> u64 {
        let mut queue = self.pd_queue.lock();
        let place = queue.next;
        queue.next += 1;
        queue.waiting.push_back(PdPlace {
            place,
            rooms,
            wake: Arc::new(Notify::new()),
        });
        place
    }

    /// Whether `place` may claim: no older place still waiting needs as few
    /// rooms as it does, or fewer (see `PdQueue::may_claim`).
    pub fn pd_may_claim(&self, place: u64) -> bool {
        self.pd_queue.lock().may_claim(place)
    }

    /// Leave the queue from wherever the place is: after a claim or the shed,
    /// or when a dispatch stops waiting early. The places behind move up; a
    /// place already gone is a no-op. Only a place that could claim held any
    /// other back, so only then are the places that may claim now woken. The
    /// scans are linear in the queue, which holds only the dispatches waiting
    /// on this worker now.
    pub fn pd_dequeue(&self, place: u64) {
        let wakes = {
            let mut queue = self.pd_queue.lock();
            let Some(at) = queue
                .waiting
                .iter()
                .position(|waiting| waiting.place == place)
            else {
                return;
            };
            let could_claim = queue.may_claim(place);
            queue.waiting.remove(at);
            if could_claim {
                queue.claimable_wakes()
            } else {
                Vec::new()
            }
        };
        for wake in wakes {
            wake.notify_one();
        }
    }

    /// The notifier of `place` (see [`Worker::pd_wake`]); `None` once the
    /// place is given back.
    pub fn pd_wake(&self, place: u64) -> Option<Arc<Notify>> {
        self.pd_queue
            .lock()
            .waiting
            .iter()
            .find(|waiting| waiting.place == place)
            .map(|waiting| Arc::clone(&waiting.wake))
    }

    // ── Routing-key load ────────────────────────────────────────────

    pub fn routing_key_load(&self) -> usize {
        self.worker_routing_key_load.value()
    }

    pub fn routing_key_inflight(&self, routing_key: &str) -> usize {
        self.worker_routing_key_load.key_inflight(routing_key)
    }

    pub fn increment_routing_key_load(&self, routing_key: &str) {
        self.worker_routing_key_load.increment(routing_key);
    }

    pub fn decrement_routing_key_load(&self, routing_key: &str) {
        self.worker_routing_key_load.decrement(routing_key);
    }

    // ── Processed-request counter ───────────────────────────────────

    pub fn processed_requests(&self) -> usize {
        self.processed_counter.load(Ordering::Relaxed)
    }

    pub fn increment_processed(&self) {
        self.processed_counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn is_overloaded(&self) -> bool {
        self.overloaded.load(Ordering::Relaxed)
    }

    /// Returns `true` when the flag transitioned, so the caller can move the
    /// per-model counters and gauge exactly once per edge.
    pub fn set_overloaded(&self, overloaded: bool) -> bool {
        self.overloaded.swap(overloaded, Ordering::Relaxed) != overloaded
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use tokio::time;

    use super::*;
    use crate::worker::liveness::now_ms;

    #[test]
    fn a_token_within_the_stamped_millisecond_records_nothing_new_and_keeps_the_semantics() {
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        let wake = runtime.contact_wake();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        // Whether the notifier holds a permit: `notify_one` leaves one for a
        // later waiter, so a waiter that returns within the bound found it.
        let permit = |bound_ms: u64| {
            rt.block_on(async {
                time::timeout(Duration::from_millis(bound_ms), wake.notified())
                    .await
                    .is_ok()
            })
        };

        // Two tokens inside one millisecond: the first stamps and leaves the
        // permit, the second has nothing new to record and leaves none.
        // Retried until both calls land in the same millisecond.
        let mut aligned = false;
        for _ in 0..1_000 {
            // A fresh millisecond: the runtime's own construction stamp (or the
            // previous attempt's) must not be the one the first token repeats.
            thread::sleep(Duration::from_millis(2));
            let ms = now_ms();
            runtime.note_token_progress();
            assert!(permit(50), "the first token leaves the notifier's permit");
            runtime.note_token_progress();
            if now_ms() == ms {
                aligned = true;
                break;
            }
            // The millisecond turned between the two: the second one stamped
            // and left a permit; drain it and try again.
            let _ = permit(50);
        }
        assert!(aligned, "two tokens never landed in one millisecond");
        assert!(
            !permit(10),
            "a token within the stamped millisecond leaves no new permit"
        );

        // A transport failure flagged between two tokens of one millisecond is
        // cleared by the next token all the same, and that token stamps.
        runtime.note_transport_failure();
        assert!(runtime.transport_failure_pending());
        runtime.note_token_progress();
        assert!(
            !runtime.transport_failure_pending(),
            "progress clears a transport failure, same millisecond or not"
        );
        assert!(permit(50), "the clearing token wakes the notifier");

        // Once the millisecond turns, the next token stamps again.
        let stamped = runtime.progress_reference_ms();
        thread::sleep(Duration::from_millis(3));
        runtime.note_token_progress();
        assert!(runtime.progress_reference_ms() > stamped);
        assert!(
            permit(50),
            "a token in a new millisecond wakes the notifier"
        );
    }

    #[test]
    fn progress_is_counted_from_the_first_dispatch_not_registration() {
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        let registered = runtime.progress_reference_ms();
        thread::sleep(Duration::from_millis(15));
        runtime.increment_load();
        assert!(
            runtime.progress_reference_ms() > registered,
            "the clock starts at the dispatch, not at registration"
        );
        assert!(runtime.token_progress_age() < Duration::from_millis(10));
        assert!(runtime.try_decrement_load());
        assert_eq!(
            runtime.progress_reference_ms(),
            registered,
            "idle again: back to the last token"
        );
        assert!(runtime.try_increment_load(4));
        assert!(runtime.progress_reference_ms() > registered);
    }

    #[test]
    fn prefill_backlog_follows_pending_tokens_and_the_observed_rate() {
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        assert_eq!(runtime.prefill_backlog(), Duration::ZERO);
        // 128 prompts of 1,152 tokens on a worker that has shown no rate yet:
        // the cold prior, 10k tokens/s, gives the engine 14.7 s.
        runtime.note_prefill_started_at(128 * 1_152, true, 1_000);
        assert_eq!(runtime.prefill_backlog(), Duration::from_millis(14_745));
        // Their first tokens come back within 1.5 s of busy time (the window
        // closes at the first one past a second): an observed rate of ~98k
        // tokens/s replaces the prior, and nothing is pending.
        runtime.note_prefill_ended_at(64 * 1_152, true, Some(0), 1_600);
        runtime.note_prefill_ended_at(64 * 1_152, true, Some(0), 2_500);
        assert_eq!(runtime.prefill_backlog(), Duration::ZERO);
        assert_eq!(runtime.prefill_rate_tps(), 98_304);
        runtime.note_prefill_started_at(1_000_000, true, 3_000);
        assert_eq!(runtime.prefill_backlog(), Duration::from_millis(10_172));
        // A dropped stream releases its tokens without a rate sample.
        runtime.note_prefill_ended_at(2_000_000, true, None, 3_500);
        assert_eq!(runtime.prefill_backlog(), Duration::ZERO);
    }

    #[test]
    fn prefill_eta_adds_a_prompt_about_to_be_dispatched_to_the_backlog() {
        // The disaggregated prefill side's bound is read before its prompt is
        // booked: the ETA is the backlog plus that prompt, at the rate the
        // worker has shown (the cold prior until it has).
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        assert_eq!(runtime.prefill_eta(0), Duration::ZERO);
        assert_eq!(runtime.prefill_eta(30_000), Duration::from_secs(3));
        runtime.note_prefill_started_at(20_000, true, 1_000);
        assert_eq!(runtime.prefill_eta(30_000), Duration::from_secs(5));
        assert_eq!(runtime.prefill_eta(0), runtime.prefill_backlog());
        // A measured rate (20k tokens over 1.5 s of busy time: 13,333
        // tokens/s) replaces the prior.
        runtime.note_prefill_ended_at(10_000, true, Some(0), 1_600);
        runtime.note_prefill_ended_at(10_000, true, Some(0), 2_500);
        assert_eq!(runtime.prefill_rate_tps(), 13_333);
        assert_eq!(runtime.prefill_eta(40_000), Duration::from_millis(3_000));
    }

    #[test]
    fn a_prompt_longer_than_any_answered_is_a_cold_shape_once_the_worker_has_answered_one() {
        // The wedge rule's grace for an engine building a kernel at a new
        // shape reads these books: the longest prompt answered against the
        // longest pending (or about to be dispatched).
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        // Nothing answered yet: no shape to go by, whatever is pending.
        runtime.note_prefill_started_at(8_192, true, 1_000);
        assert!(!runtime.prefill_cold_shape(0));
        assert!(!runtime.prefill_cold_shape(100_000));
        runtime.note_prefill_ended_at(8_192, true, Some(0), 2_000);
        assert!(!runtime.prefill_cold_shape(0), "nothing pending");
        // 8k answered: a 4k prompt is a shape shown, a 16k one is not,
        // pending or about to be dispatched.
        runtime.note_prefill_started_at(4_096, true, 2_000);
        assert!(!runtime.prefill_cold_shape(0));
        assert!(
            !runtime.prefill_cold_shape(8_192),
            "as long as one answered"
        );
        assert!(runtime.prefill_cold_shape(16_384), "about to be dispatched");
        runtime.note_prefill_started_at(16_384, true, 2_100);
        runtime.note_prefill_started_at(16_384, true, 2_200);
        assert!(runtime.prefill_cold_shape(0), "pending");
        // The 4k prompt's answer shows nothing new; a 16k stream dropped
        // before its first token shows nothing either, and its twin keeps
        // the shape cold; the twin's answer shows it, and the next 16k
        // prompt is a shape run.
        runtime.note_prefill_ended_at(4_096, true, Some(0), 2_500);
        assert!(runtime.prefill_cold_shape(0));
        runtime.note_prefill_ended_at(16_384, true, None, 3_000);
        assert!(runtime.prefill_cold_shape(0), "the twin is still pending");
        runtime.note_prefill_ended_at(16_384, true, Some(0), 4_000);
        assert!(!runtime.prefill_cold_shape(0), "nothing pending");
        assert!(!runtime.prefill_cold_shape(16_384), "shown");
        assert!(runtime.prefill_cold_shape(16_385));
        // A non-streaming generation's one answer shows its shape too.
        runtime.note_prefill_started_at(32_768, false, 5_000);
        assert!(runtime.prefill_cold_shape(0));
        runtime.note_prefill_ended_at(32_768, false, Some(0), 9_000);
        assert!(!runtime.prefill_cold_shape(32_768));
        // A release of tokens no start booked changes nothing.
        runtime.note_prefill_ended_at(1_000_000, true, None, 9_500);
        assert!(!runtime.prefill_cold_shape(0));
    }

    #[test]
    fn the_latest_first_response_is_stamped_on_the_worker() {
        // A prefill side past its own bound asks whether its worker answers
        // anything at all: the stamp is the latest first response to any
        // prompt dispatched there, never the drop of a stream that got none.
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        assert_eq!(
            runtime.prefill_answered_ago_at(5_000),
            None,
            "never answered"
        );
        runtime.note_prefill_started_at(256, true, 1_000);
        runtime.note_prefill_started_at(256, true, 1_000);
        runtime.note_prefill_ended_at(256, true, None, 1_500);
        assert_eq!(
            runtime.prefill_answered_ago_at(5_000),
            None,
            "a dropped stream is no answer"
        );
        runtime.note_prefill_ended_at(256, true, Some(0), 2_000);
        assert_eq!(
            runtime.prefill_answered_ago_at(5_000),
            Some(Duration::from_secs(3))
        );
        runtime.note_prefill_started_at(64, false, 6_000);
        runtime.note_prefill_ended_at(64, false, Some(0), 6_500);
        assert_eq!(
            runtime.prefill_answered_ago_at(7_000),
            Some(Duration::from_millis(500)),
            "a non-streaming generation's one answer counts too"
        );
    }

    #[test]
    fn a_handoff_is_evidence_for_the_silence_rule_and_nothing_else() {
        // A decode side's first output shows the prefill worker handed a
        // prompt off. Stamped apart from the answers, it releases no prompt,
        // times nothing, answers no run and shows no shape.
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        runtime.note_prefill_started_at(4_096, true, 1_000);
        runtime.note_prefill_handoff_at(3_000);
        let second = Some(Duration::from_secs(1));
        assert_eq!(runtime.prefill_handed_off_ago_at(4_000), second);
        assert_eq!(runtime.prefill_answered_ago_at(4_000), None, "no answer");
        assert_eq!(runtime.prefill_backlog(), Duration::from_millis(409));
        assert_eq!(runtime.prefill_rate_tps(), 0, "nothing timed");
        assert_eq!(
            runtime.prefill_unanswered_run_at(4_000),
            Some(Duration::from_secs(3)),
            "the run still awaits its first answer"
        );
        assert!(!runtime.prefill_cold_shape(0), "no shape shown");
        runtime.note_prefill_handoff_at(2_000);
        assert_eq!(runtime.prefill_handed_off_ago_at(4_000), second, "not back");
        runtime.note_prefill_ended_at(4_096, true, Some(0), 5_000);
        assert_eq!(runtime.prefill_answered_ago_at(5_000), Some(Duration::ZERO));
        assert_eq!(runtime.prefill_unanswered_run_at(5_000), None);
        assert_eq!(
            runtime.prefill_handed_off_ago_at(5_000),
            Some(Duration::from_secs(2)),
            "the answer leaves the handoff alone"
        );
    }

    #[test]
    fn a_run_of_prompts_awaits_its_first_answer_from_its_start() {
        // A prefill side past its own bound on a worker that answered
        // nothing within it asks whether the worker's current run of
        // pending prompts still awaits its first answer, and for how long:
        // an idle spell is no silence, an answer ends the wait.
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        assert_eq!(
            runtime.prefill_unanswered_run_at(5_000),
            None,
            "nothing at all"
        );
        // Never answered: waiting since its first prompt reached it.
        runtime.note_prefill_started_at(1_000, true, 1_000);
        runtime.note_prefill_started_at(1_000, true, 1_200);
        assert_eq!(
            runtime.prefill_unanswered_run_at(4_000),
            Some(Duration::from_secs(3))
        );
        // A dropped stream is no answer, and the run goes on.
        runtime.note_prefill_ended_at(1_000, true, None, 4_500);
        assert_eq!(
            runtime.prefill_unanswered_run_at(5_000),
            Some(Duration::from_secs(4))
        );
        // An answer that ends the run, then an idle minute: the next run
        // waits from its own start, not from the answer before it.
        runtime.note_prefill_ended_at(1_000, true, Some(0), 6_000);
        assert_eq!(runtime.prefill_unanswered_run_at(7_000), None, "idle");
        runtime.note_prefill_started_at(1_000, true, 66_000);
        assert_eq!(
            runtime.prefill_unanswered_run_at(67_000),
            Some(Duration::from_secs(1)),
            "waiting since the burst after the idle minute"
        );
        assert_eq!(
            runtime.prefill_answered_ago_at(67_000),
            Some(Duration::from_secs(61)),
            "the latest answer itself is still a minute back"
        );
        // An answer inside the run ends the wait for good: the run has
        // shown the worker answers.
        runtime.note_prefill_started_at(1_000, true, 67_500);
        runtime.note_prefill_ended_at(1_000, true, Some(0), 68_000);
        assert_eq!(runtime.prefill_unanswered_run_at(69_000), None);
        assert_eq!(
            runtime.prefill_answered_ago_at(69_000),
            Some(Duration::from_secs(1))
        );
    }

    #[test]
    fn a_run_that_ends_unanswered_leaves_its_silence_to_the_next() {
        // The worker answered at 1.1 s, then held a prompt from 2 s until it
        // was given up on at 60 s (a release with no answer), sat idle, and
        // took its next prompt at 61 s. The run given up on is silence the
        // next run starts from; the idle spells around it are not.
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        runtime.note_prefill_started_at(1_000, true, 1_000);
        runtime.note_prefill_ended_at(1_000, true, Some(0), 1_100);
        runtime.note_prefill_started_at(1_000, true, 2_000);
        runtime.note_prefill_ended_at(1_000, true, None, 60_000);
        assert_eq!(runtime.prefill_unanswered_run_at(60_500), None, "idle");
        runtime.note_prefill_started_at(1_000, true, 61_000);
        assert_eq!(
            runtime.prefill_unanswered_run_at(62_000),
            Some(Duration::from_secs(59)),
            "58 s held unanswered before, one second since"
        );
        // Another run given up on, after one that was answered then left
        // unanswered: only the time since the answer is carried.
        runtime.note_prefill_ended_at(1_000, true, None, 63_000);
        runtime.note_prefill_started_at(1_000, true, 64_000);
        runtime.note_prefill_started_at(1_000, true, 64_000);
        runtime.note_prefill_ended_at(1_000, true, Some(0), 65_000);
        runtime.note_prefill_ended_at(1_000, true, None, 70_000);
        runtime.note_prefill_started_at(1_000, true, 80_000);
        assert_eq!(
            runtime.prefill_unanswered_run_at(81_000),
            Some(Duration::from_secs(6)),
            "five seconds held since the answer at 65 s, one since the new run"
        );
        // An answer clears what was carried.
        runtime.note_prefill_ended_at(1_000, true, Some(0), 82_000);
        runtime.note_prefill_started_at(1_000, true, 90_000);
        assert_eq!(
            runtime.prefill_unanswered_run_at(91_000),
            Some(Duration::from_secs(1))
        );
    }

    #[test]
    fn an_idle_gap_between_sparse_prompts_is_not_slow_prefill() {
        // A lightly used worker: a 47-token prompt, answered in 10 ms, then a
        // minute of nothing, then another. Measured over wall time the two
        // made a rate of about one token per second, and one pending prompt
        // then stretched the wedge bound to the better part of a minute: the
        // pile rule went blind exactly where a frozen engine holds one
        // request and no pile. Only busy time counts now.
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        runtime.note_prefill_started_at(47, true, 1_000);
        runtime.note_prefill_ended_at(47, true, Some(0), 1_010);
        runtime.note_prefill_started_at(47, true, 61_000);
        runtime.note_prefill_ended_at(47, true, Some(0), 61_010);
        assert_eq!(
            runtime.prefill_rate_tps(),
            0,
            "20 ms of busy time is no window yet: the prior stands"
        );
        runtime.note_prefill_started_at(47, true, 62_000);
        assert_eq!(
            runtime.prefill_backlog(),
            Duration::from_millis(4),
            "one short pending prompt at the cold prior, not 47 s"
        );
        runtime.note_prefill_ended_at(47, true, Some(0), 62_010);
        // Short prompts do make a rate once their busy time adds up to the
        // window: 100 of them at 10 ms each are a second of prefill.
        let mut now = 100_000;
        for _ in 0..97 {
            runtime.note_prefill_started_at(47, true, now);
            runtime.note_prefill_ended_at(47, true, Some(0), now + 10);
            now += 5_000;
        }
        assert_eq!(
            runtime.prefill_rate_tps(),
            4_700,
            "4,700 tokens over 1,000 ms of busy time"
        );
        // Overlapping prompts share one busy interval: their tokens count
        // over the union of their prefill time, not the sum of their waits.
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        runtime.note_prefill_started_at(10_000, true, 1_000);
        runtime.note_prefill_started_at(10_000, true, 1_200);
        runtime.note_prefill_ended_at(10_000, true, Some(0), 1_900);
        runtime.note_prefill_ended_at(10_000, true, Some(0), 2_400);
        assert_eq!(runtime.prefill_rate_tps(), 20_000 * 1000 / 1_400);
    }

    #[test]
    fn a_worker_that_is_never_idle_keeps_updating_its_prefill_rate() {
        // Continuous arrivals: a new 500-token prompt every 200 ms, each
        // answered 300 ms after its dispatch, so the pending count never
        // touches zero. The window still closes at every first token past a
        // second of busy time, the rate follows the engine (500 tokens per
        // 200 ms = 2,500 tokens/s), and the bound stays near the threshold.
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        let mut now = 1_000;
        let mut samples = 0;
        let mut last_rate = 0;
        for i in 0..60u64 {
            runtime.note_prefill_started_at(500, true, now);
            if i >= 2 {
                // the prompt dispatched 400 ms ago gets its first token
                runtime.note_prefill_ended_at(500, true, Some(0), now - 100);
            }
            if runtime.prefill_rate_tps() != last_rate {
                samples += 1;
                last_rate = runtime.prefill_rate_tps();
            }
            now += 200;
        }
        assert!(samples >= 8, "the rate kept updating: {samples} samples");
        let rate = runtime.prefill_rate_tps();
        assert!(
            (2_000..=3_000).contains(&rate),
            "about 2,500 tokens/s, got {rate}"
        );
        // Two prompts pending (1,000 tokens): a bound well under a second.
        assert!(runtime.prefill_backlog() < Duration::from_secs(1));
    }

    #[test]
    fn first_tokens_that_come_back_together_never_raise_the_rate_past_the_busy_time_average() {
        // An engine that prefills 20k tokens per second of busy time, whose
        // first tokens come back bunched: the windows that timed a bunch's
        // prefill hold few of its tokens, and the window it comes back in
        // holds the rest over a second or so. Halved into the estimate,
        // that window's sample put it at over four times the engine's
        // throughput (92k and 81k tokens/s in the two shapes below), and a
        // 90k prompt then had an ETA of a second for the 4.5 s it takes.
        // However they bunch, the answers are never more than the engine
        // prefilled in the busy time behind them, and on a book that has not
        // held the span yet, as in both shapes here, neither is the estimate.
        const RATE: u64 = 20_000;
        let answer = |runtime: &WorkerRuntime, tokens: u64, at: u64| {
            runtime.note_prefill_ended_at(tokens, true, Some(0), at);
            let rate = runtime.prefill_rate_tps();
            assert!(rate <= RATE, "{rate} tokens/s at {at} ms, over {RATE}");
        };
        // Two 90k prompts prefilled side by side for 10 s beside a 2k prompt
        // dispatched every second and answered 100 ms later (18k of the long
        // ones and the short one per second), the long ones answered
        // together at 11 s; then the short ones alone.
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        runtime.note_prefill_started_at(90_000, true, 1_000);
        runtime.note_prefill_started_at(90_000, true, 1_000);
        for second in 0..30 {
            let at = 1_000 + second * 1_000;
            if at == 11_000 {
                answer(&runtime, 90_000, at);
                answer(&runtime, 90_000, at);
            }
            runtime.note_prefill_started_at(2_000, true, at);
            answer(&runtime, 2_000, at + 100);
        }
        assert_eq!(runtime.prefill_rate_tps(), RATE);
        assert_eq!(
            runtime.prefill_eta(90_000),
            Duration::from_millis(4_500),
            "the prefill a 90k prompt takes"
        );
        // A batch of eight 20k prompts the engine finishes at once, 8 s
        // after their dispatch, then a 2k prompt every second.
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        for _ in 0..8 {
            runtime.note_prefill_started_at(20_000, true, 1_000);
        }
        for _ in 0..8 {
            answer(&runtime, 20_000, 9_000);
        }
        for second in 0..20 {
            let at = 9_000 + second * 1_000;
            runtime.note_prefill_started_at(2_000, true, at);
            answer(&runtime, 2_000, at + 100);
        }
        assert_eq!(runtime.prefill_rate_tps(), RATE);
    }

    /// A worker whose engine has prefilled a 2k prompt a second, each
    /// answered 100 ms after its dispatch (20k tokens per second of busy
    /// time), for 40 s of busy time: a book that has held the span. Returned
    /// with the clock of its next second.
    fn warmed_at_20k_tokens_per_sec() -> (WorkerRuntime, u64) {
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        let mut at = 1_000;
        for _ in 0..400 {
            runtime.note_prefill_started_at(2_000, true, at);
            runtime.note_prefill_ended_at(2_000, true, Some(0), at + 100);
            at += 1_000;
        }
        assert_eq!(runtime.prefill_rate_tps(), 20_000);
        (runtime, at)
    }

    #[test]
    fn a_pair_answered_together_after_a_warm_spell_lifts_the_rate_less_than_five_percent() {
        // The pair of 90k prompts above, prefilled side by side for 10 s
        // beside a 2k prompt a second, on a worker that has prefilled at the
        // same 20k tokens per second for longer than the span: each window of
        // the pair's prefill scales back the windows before it, the pair's
        // earlier ones among them, while the pair's tokens come back at full
        // weight. The estimate reads above the engine's throughput (20,775
        // tokens/s here, 3.9% over), within 5% of it for a prefill a third of
        // the span long; longer ones read further above (see `PrefillBook`),
        // all far below the 92k the halved sample read.
        const RATE: u64 = 20_000;
        let (runtime, start) = warmed_at_20k_tokens_per_sec();
        runtime.note_prefill_started_at(90_000, true, start);
        runtime.note_prefill_started_at(90_000, true, start);
        for second in 0..30 {
            let at = start + second * 1_000;
            if second == 10 {
                runtime.note_prefill_ended_at(90_000, true, Some(0), at);
                runtime.note_prefill_ended_at(90_000, true, Some(0), at);
            }
            runtime.note_prefill_started_at(2_000, true, at);
            runtime.note_prefill_ended_at(2_000, true, Some(0), at + 100);
            let rate = runtime.prefill_rate_tps();
            assert!(
                rate <= RATE * 105 / 100,
                "{rate} tokens/s {second} s into the pair: over 5% above {RATE}"
            );
        }
    }

    #[test]
    fn an_engine_that_slows_down_pulls_the_rate_down_at_its_next_sample() {
        // The average caps the estimate and never holds it up: a worker at
        // 20k tokens per second whose engine slows to half that (its 2k
        // prompts answered 200 ms after their dispatch) has each sample
        // halved into the estimate at once, as before, while the average,
        // still holding the span at the old pace, stands above it.
        let (runtime, mut at) = warmed_at_20k_tokens_per_sec();
        for rate in [15_000, 12_500, 11_250] {
            // Five such prompts are a window: 10k tokens over 1 s of busy time.
            for _ in 0..5 {
                runtime.note_prefill_started_at(2_000, true, at);
                runtime.note_prefill_ended_at(2_000, true, Some(0), at + 200);
                at += 1_000;
            }
            assert_eq!(
                runtime.prefill_rate_tps(),
                rate,
                "halfway to the 10k sample at each window"
            );
        }
    }

    #[test]
    fn a_request_ending_as_another_starts_never_loses_the_busy_interval() {
        // The race the review described: one request's end (pending to
        // zero) and another's start happen at the same instant. Whichever
        // order the two are applied in, the book ends up with the new
        // prompt pending and a busy interval running from that instant, so
        // the window keeps measuring; a lost interval would have left this
        // worker's rate stale for as long as it stayed busy.
        for order in ["end-then-start", "start-then-end"] {
            let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
            runtime.note_prefill_started_at(100, true, 1_000);
            match order {
                "end-then-start" => {
                    runtime.note_prefill_ended_at(100, true, Some(0), 1_500);
                    runtime.note_prefill_started_at(50, true, 1_500);
                }
                _ => {
                    runtime.note_prefill_started_at(50, true, 1_500);
                    runtime.note_prefill_ended_at(100, true, Some(0), 1_500);
                }
            }
            // The next first token, 600 ms later, completes a window of
            // 1.1 s of busy time either way (500 ms banked or still open,
            // plus 600 ms) and makes a rate: 150 tokens over 1.1 s.
            runtime.note_prefill_ended_at(50, true, Some(0), 2_100);
            assert_eq!(
                runtime.prefill_rate_tps(),
                150 * 1000 / 1_100,
                "{order}: the window closed and the rate is measured"
            );
            assert_eq!(runtime.prefill_backlog(), Duration::ZERO, "{order}");
        }
    }

    #[test]
    fn a_non_streaming_generation_in_flight_times_no_prefill() {
        // A non-streaming generation answers once, at its end: the gateway
        // never sees its first token, so its prompt stays pending through
        // its whole decode. Counted as busy time, that decode turned the
        // idle gaps between the streaming prompts around it back into
        // prefill time (five short prompts 5 s apart learned 9 tokens/s,
        // the rate this change takes out), and its one answer timed the
        // 30-s decode as the prefill of 4,000 tokens (a 405 tokens/s rate
        // and a 116-ms bound for the next short prompt). Its prompt is in
        // the backlog (the engine prefills it before anything dispatched
        // behind it streams) but opens no busy interval and makes no sample.
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        runtime.note_prefill_started_at(4_000, false, 1_000);
        assert_eq!(
            runtime.prefill_backlog(),
            Duration::from_millis(400),
            "its prompt is in the backlog at the cold prior"
        );
        // Streaming 47-token prompts every 5 s, answered in 10 ms each,
        // while it decodes.
        let mut now = 6_000;
        for _ in 0..5 {
            runtime.note_prefill_started_at(47, true, now);
            runtime.note_prefill_ended_at(47, true, Some(0), now + 10);
            now += 5_000;
        }
        assert_eq!(
            runtime.prefill_rate_tps(),
            0,
            "50 ms of busy time is no window: the gaps stayed idle"
        );
        // Its answer, 30 s after dispatch: no sample, and the backlog is gone.
        runtime.note_prefill_ended_at(4_000, false, Some(0), 31_000);
        assert_eq!(
            runtime.prefill_rate_tps(),
            0,
            "its one answer times no prefill"
        );
        assert_eq!(runtime.prefill_backlog(), Duration::ZERO);
        // The next short prompt gets the prior's bound, not the gaps'.
        runtime.note_prefill_started_at(47, true, 31_500);
        assert_eq!(runtime.prefill_backlog(), Duration::from_millis(4));
    }

    #[test]
    fn a_prefix_cache_hit_teaches_the_tokens_the_engine_computed() {
        // A worker serving multi-turn conversations: 2,000-token prompts of
        // which the engine's prefix cache holds 1,900, so it computes 100
        // tokens per turn, 10 ms each at its 10k tokens/s. Counted at their
        // full length, a hundred such turns taught 200k tokens/s, and a
        // burst of four cold 16,000-token prompts then got a backlog of
        // 320 ms, the bare threshold, for a prefill the engine needs 6.4 s
        // for, during which nothing streams: the false positive the bound
        // exists to prevent. A first token teaches the tokens the engine
        // computed, the prompt less the hit it reports.
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        let mut now = 1_000;
        for _ in 0..100 {
            runtime.note_prefill_started_at(2_000, true, now);
            runtime.note_prefill_ended_at(2_000, true, Some(1_900), now + 10);
            now += 500;
        }
        assert_eq!(
            runtime.prefill_rate_tps(),
            10_000,
            "a hundred turns of 100 computed tokens over a second of busy time"
        );
        runtime.note_prefill_started_at(4 * 16_000, true, now);
        assert_eq!(
            runtime.prefill_backlog(),
            Duration::from_millis(6_400),
            "the cold burst's prefill time at the rate learned, not 320 ms"
        );
        // The cold prompts' first tokens, with no hit, teach their full
        // length: 64,000 tokens over 6.4 s, the same rate.
        runtime.note_prefill_ended_at(4 * 16_000, true, Some(0), now + 6_400);
        assert_eq!(runtime.prefill_rate_tps(), 10_000);
        assert_eq!(runtime.prefill_backlog(), Duration::ZERO);
        // An engine that serves whole prompts from its cache computes
        // nothing the window can time: no sample, the prior stands.
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        for i in 0..100u64 {
            let now = 1_000 + i * 500;
            runtime.note_prefill_started_at(2_000, true, now);
            runtime.note_prefill_ended_at(2_000, true, Some(2_000), now + 10);
        }
        assert_eq!(
            runtime.prefill_rate_tps(),
            0,
            "nothing computed, nothing learned"
        );
        runtime.note_prefill_started_at(4 * 16_000, true, 60_000);
        assert_eq!(
            runtime.prefill_backlog(),
            Duration::from_millis(6_400),
            "the cold prior, 10k tokens/s"
        );
    }

    #[test]
    fn warm_up_growth_is_measured_since_admission_and_survives_a_clear() {
        let runtime = WorkerRuntime::new("http://w1:8000", WorkerStatus::Ready);
        // A returned worker is first seen with its stale blocks still indexed:
        // those are the baseline, not growth.
        assert_eq!(runtime.warmup_growth(2000), 0);
        assert_eq!(runtime.warmup_growth(2100), 100);
        // The restart is detected and the index cleared: growth restarts from
        // zero, not from a baseline the index no longer holds.
        assert_eq!(runtime.warmup_growth(50), 50);
        assert_eq!(runtime.warmup_growth(700), 700);
        // Admitted again: a fresh baseline.
        thread::sleep(Duration::from_millis(2));
        runtime.note_admitted();
        assert_eq!(runtime.warmup_growth(900), 0);
        assert_eq!(runtime.warmup_growth(1300), 400);
    }
}
