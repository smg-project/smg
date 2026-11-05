//! Progress-based liveness beside the health check.
//!
//! The health checker needs `failure_threshold * check_interval` to exclude a
//! worker, tens of seconds with the defaults. The transport often knows
//! sooner: a worker that dies or restarts resets its connection and fails its
//! KV event stream, its load poll and its in-flight streams at once; one that
//! falls silent fails them at the keepalive timeout or the poll deadline.
//! This module turns those failures into a routing veto and clears it on the
//! first successful contact.
//!
//! Two vetoes, both read by routing through [`Worker::stall_reason`]
//! (the second reached by two rules):
//!
//! - **unreachable**: a connection failure (load poll, KV event stream) while
//!   nothing has been heard from the worker for the stall threshold. Any
//!   successful contact clears it. Health probes count as contact when they
//!   pass; a failed probe is left to the health state machine, because probe
//!   timeouts on a slow but streaming worker are not unreachability.
//! - **wedged**: the worker still answers polls but has produced no token or
//!   completion for the wedge bound while it holds in-flight requests and
//!   its waiting queue grows. Progress clears it; so does an empty pile (the
//!   sweep lifts the veto once nothing is in flight, and a stuck engine
//!   re-arms it within the bound on the next requests: steering, never a
//!   refusal); and a transport failure followed by silence turns it into an
//!   unreachable veto, which contact clears, so a wedged worker whose
//!   connection the keepalive tears down is not left waiting for progress
//!   from streams that no longer exist. A paused engine that keeps
//!   answering health looks exactly like this. The servicer's pushed load
//!   record carries the engine's own silence as well
//!   (`EngineLoad.engine_silence_ms`: how long the engine has produced
//!   nothing while the servicer holds requests on it that the engine
//!   reports every step; a generation the engine reports at intervals, one
//!   that does not stream or one streamed every so many tokens, is quiet
//!   while healthy and counts none); past the bound that
//!   is the same veto without a pile, so a frozen engine with one request on
//!   it, or with requests from another gateway, leaves routing within the
//!   bound instead of once a pile forms. The bound is the threshold or, if
//!   longer, the allowance for the prefill the servicer reports in hand
//!   (`prefill_pending_tokens`: the prompt tokens of every client's requests
//!   the engine has produced nothing for yet, so a gateway with nothing of
//!   its own on the worker allows for the same prefill): twice that prefill
//!   at the prefill rate this gateway has measured on the worker (twice,
//!   the rate being an estimate), under the wedge bound's cap. Which rate,
//!   and whether a verdict comes without the figure, depends on the wire.
//!   A wire that carries token batches only is silent through a whole
//!   prefill, so it is judged only once the worker has shown a rate (the
//!   cold prior behind the bound is a guess, not evidence against an engine
//!   that prefills slower than it) and only with the figure. An engine
//!   whose wire reports every scheduler step (`engine_reports_steps`: it
//!   has shown a stats-only step batch, or its launcher told the servicer
//!   that its statistics are on) is silent for one step at most while it
//!   serves, and the step that starts a long multimodal prompt is long (an
//!   encoder runs inside it over a clip's frames, with nothing on the wire
//!   until it ends; a kernel the engine builds for a new shape within a
//!   length it has run is silent in the same way), so its silence is judged
//!   against the allowance too, from the cold prior until the worker shows
//!   a rate (one step rides on the guess, and the bare threshold is what
//!   judged it before), and against the bare threshold while the servicer
//!   reports nothing pending (an engine frozen in decode) or no figure at
//!   all (an older servicer). Like the grace, the allowance arms no veto
//!   and lifts none: a vetoed worker is judged at the bare threshold on
//!   either wire until its engine produces, however long a prompt reaches
//!   it meanwhile. The figure is a sum over every request pending
//!   at the engine, so a frozen engine with a deep queue of short prompts
//!   is caught at twice that queue's prefill time rather than the
//!   threshold: the bound the gateway's own pile rule already applies to
//!   its in-flight prompts; the record carries no per-request figure to be
//!   sharper with. That veto ends with
//!   progress, or with a record that shows the engine producing again while
//!   this gateway has no silent pile of its own. The requests counted are the
//!   tracked ones, streaming generations to the worker over gRPC, whose
//!   responses the gateway sees one by one ([`Worker::tracked_load`]): an
//!   HTTP worker, a PD side or a non-streaming generation gives no signal
//!   between dispatch and completion, so it never forms a pile and is never
//!   judged by one; a non-streaming generation's one answer is progress all
//!   the same, and its prompt counts in the prefill backlog (but times no
//!   prefill: the gateway never sees its first token). The clock starts at
//!   the first dispatch of a run of
//!   tracked requests, never at registration, and the bound is the
//!   configured threshold or, if longer, the time the engine may still need
//!   to prefill what is in flight ([`Worker::prefill_backlog`]): a batch that
//!   is all in prefill streams nothing and is not wedged. The bound is the
//!   cap outright ([`COLD_SHAPE_GRACE`]) while the worker is on a prompt
//!   longer than any it has answered, once it has answered one
//!   ([`Worker::prefill_cold_shape`]): an engine meeting a shape it has not
//!   run may build a kernel for it first, a minute of silence on the engines
//!   we know, with nothing on any wire to tell it from a freeze (the
//!   servicer's record flows through both, its health answers through
//!   both), and the gateway's own books are the one evidence there is. The
//!   grace arms no veto and lifts none: a worker vetoed before the longer
//!   prompt reached it is judged as before until it produces; a real wedge
//!   met on such a prompt is caught at the cap instead of the threshold,
//!   once per new longest prompt; a worker that has answered nothing is
//!   judged as before. A zero threshold
//!   turns the rule off. The same bound, read per request with twice the
//!   prefill time in it (the engine-reported rule's margin on an estimated
//!   rate), caps how long a disaggregated prefill side may stay silent
//!   ([`prefill_side_bound`]): a side is not part of any pile, so its
//!   request is what fails, over to another prefill worker when one is
//!   available. A side past its bound is
//!   given up on only while its worker has answered nothing within the
//!   bound ([`prefill_side_reprieve`]): a worker that keeps answering other
//!   sides is backed up (on the parallel dispatch shape a side's first
//!   response also waits for the decode side), not frozen, and its side
//!   waits on; on that shape a decode side's first output counts as an
//!   answer of its prefill worker's too (the decode engine produces it from
//!   the KV the worker handed off, and the prefill engine may answer only an
//!   engine step later: a TokenSpeed prefill worker answers once it has
//!   polled the transfer's success after its next forward), for at most a
//!   bound past where its answers leave the side, and not for one that has
//!   never answered; a side in flight when the longest prompt yet lands behind
//!   it is given what is left of the cold-shape grace since its dispatch;
//!   and a worker whose current run of pending prompts has not answered
//!   yet (a burst after an idle spell, or before its first answer ever) has
//!   shown no silence before the run began, so its side waits until the
//!   run has gone unanswered for the time the backlog the worker held then
//!   takes, as the pile rule gives a worker its backlog.
//!
//! - **stale records** (an unreachable veto by another rule): a worker whose
//!   KV-event stream has pushed load records is heard from every few
//!   seconds while it is reachable (a record rides every event batch and a
//!   heartbeat fills every silence), so no contact at all for the stale
//!   threshold is a dead or one-way link that the transport has not
//!   reported yet: a one-way cut leaves the connection open until the
//!   keepalive verdict, tens of seconds later, and a failed poll deadline
//!   is deliberately not a transport failure. The veto is `unreachable`,
//!   cleared by the first contact like any other. Workers that never pushed
//!   a record (HTTP workers, older servicers, a stream still connecting)
//!   are not judged by their silence: a poll every ten seconds and a probe
//!   every minute is all that is expected of them. Tokens are contact too,
//!   so a worker that still streams is never stale. A zero threshold turns
//!   the rule off.
//!
//! No veto touches the worker's health status: the health checker keeps
//! its own state machine, and the veto is simply gone once the worker talks.

use std::{
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use openai_protocol::worker::WorkerStatus;
use tracing::{info, warn};

use super::{
    basic::{StallReason, COLD_PREFILL_TOKENS_PER_SEC},
    Worker,
};
use crate::{config::RouterConfig, observability::metrics::Metrics};

const DEFAULT_STALL: Duration = Duration::from_secs(2);
const DEFAULT_WEDGE: Duration = Duration::from_secs(3);
/// Three idle heartbeats of a servicer's stream (one every five seconds once
/// its record stops changing), and past the ten-second load-poll interval, so
/// a stream that stopped while the poll still answers is never judged by its
/// silence alone.
const DEFAULT_STALE: Duration = Duration::from_secs(15);

/// How often the sweep runs.
pub(crate) const SWEEP_INTERVAL: Duration = Duration::from_millis(250);

/// In-flight requests this many deep with no token for the wedge threshold
/// are a wedged engine even when nothing new arrives (a saturated client
/// stops adding to the pile); fewer could be a long prefill.
const WEDGE_PILE: usize = 4;

/// The prefill-aware wedge bound never exceeds this (unless the configured
/// threshold itself does): a leak in the prefill books must not blind the
/// rule for good.
const WEDGE_BOUND_CAP: Duration = Duration::from_secs(120);

/// The wedge bound while a worker is on a prompt longer than any it has
/// answered ([`Worker::prefill_cold_shape`]): the cap, outright. An engine
/// meeting a shape it has not run may build a kernel for it first, a minute
/// of silence with nothing on the wire to tell it from a freeze, so the
/// gateway gives the shape the most the prefill stretch could, and a real
/// wedge met there is caught at the cap, once per new longest prompt.
const COLD_SHAPE_GRACE: Duration = WEDGE_BOUND_CAP;

/// The wedge bound for `worker` now: the configured threshold, or the time
/// the engine may still need before the first token of its in-flight prompts
/// is due, or the cold-shape grace while one of them is longer than any the
/// worker has answered, whichever is longest.
fn wedge_bound(worker: &Arc<dyn Worker>, wedge: Duration) -> Duration {
    wedge_bound_given(worker, wedge, cold_shape_grace(worker, 0))
}

/// [`wedge_bound`] at the given cold-shape `grace`.
fn wedge_bound_given(worker: &Arc<dyn Worker>, wedge: Duration, grace: Duration) -> Duration {
    bound_over(wedge, worker.prefill_backlog().max(grace))
}

/// The cold-shape grace for `worker` with `extra` more prompt tokens about
/// to be dispatched: [`COLD_SHAPE_GRACE`] while the worker is on a prompt
/// longer than any it has answered, zero otherwise.
fn cold_shape_grace(worker: &Arc<dyn Worker>, extra: u64) -> Duration {
    if worker.prefill_cold_shape(extra) {
        COLD_SHAPE_GRACE
    } else {
        Duration::ZERO
    }
}

/// The wedge rule's bound over an `eta` the engine may still need: the
/// configured threshold or the ETA, whichever is longer, capped.
fn bound_over(wedge: Duration, eta: Duration) -> Duration {
    wedge.max(eta).min(wedge.max(WEDGE_BOUND_CAP))
}

/// How long a disaggregated prefill side of `prompt_tokens` dispatched to
/// `worker` now may produce nothing before its request gives up on it: the
/// wedge rule's bound, with the side's own prompt behind what the worker is
/// already prefilling ([`Worker::prefill_eta`]) at twice the time the
/// worker's rate gives, as the engine-reported rule allows for the prefill
/// in hand ([`silence_allowance`]): the rate is an estimate, the answer can
/// come an engine step after the prefill (a TokenSpeed prefill worker
/// answers once it has polled the KV transfer, after its next forward), and
/// a side's verdict fails its request where that rule's veto only steers.
/// So a long prompt on a busy worker gets twice the time its measured rate
/// says it needs, a short one the configured threshold, a prompt longer
/// than any the worker has answered the cold-shape grace, and nothing more
/// than the cap; a decode side counts that time once
/// ([`decode_side_bound`]). A PD side joins no pile and is never vetoed by
/// one; the request is what the bound ends. `None` when the rule is off (a
/// zero threshold).
pub(crate) fn prefill_side_bound(worker: &Arc<dyn Worker>, prompt_tokens: u64) -> Option<Duration> {
    let (_, wedge, _) = thresholds();
    prefill_side_bound_with(worker, prompt_tokens, wedge)
}

/// [`prefill_side_bound`] at the given wedge threshold.
fn prefill_side_bound_with(
    worker: &Arc<dyn Worker>,
    prompt_tokens: u64,
    wedge: Duration,
) -> Option<Duration> {
    side_bound_with(worker, prompt_tokens, wedge, 2)
}

/// A side's bound at the given wedge threshold: the threshold, `times` the
/// time a prompt of `prompt_tokens` takes behind what `worker` is already
/// prefilling, or the cold-shape grace, whichever is longest, capped.
/// `None` when the rule is off (a zero threshold).
fn side_bound_with(
    worker: &Arc<dyn Worker>,
    prompt_tokens: u64,
    wedge: Duration,
    times: u32,
) -> Option<Duration> {
    (!wedge.is_zero()).then(|| {
        let eta = worker.prefill_eta(prompt_tokens).saturating_mul(times);
        bound_over(wedge, eta.max(cold_shape_grace(worker, prompt_tokens)))
    })
}

/// How long a sequential disaggregated decode side of `prompt_tokens`
/// dispatched now may produce nothing before its request gives up on it:
/// the prefill side's bound for the same prompt on `prefills`, the worker
/// whose prefill of the prompt stands before the side's first token
/// ([`prefill_side_bound`]): the prefill worker when the side was handed its
/// KV (the handoff moves what that prefill produced), the decode worker when
/// it recomputes the prompt itself (no handoff: a fan-out, or a prefill that
/// returned no transfer parameters), with that prefill's time once, not
/// twice: the side's first token waits on a prefill that has answered, or
/// on the decode worker's own at the cold rate, not on one still to come
/// behind a busy worker's backlog. The configured threshold for a short
/// prompt, the time the worker's measured prefill rate says a long prompt
/// takes, and nothing more than the cap. The side was admitted to a room on
/// its decode worker before its dispatch. `None` when the rule is off (a
/// zero threshold).
pub(crate) fn decode_side_bound(
    prefills: &Arc<dyn Worker>,
    prompt_tokens: u64,
) -> Option<Duration> {
    let (_, wedge, _) = thresholds();
    decode_side_bound_with(prefills, prompt_tokens, wedge)
}

/// [`decode_side_bound`] at the given wedge threshold.
fn decode_side_bound_with(
    prefills: &Arc<dyn Worker>,
    prompt_tokens: u64,
    wedge: Duration,
) -> Option<Duration> {
    side_bound_with(prefills, prompt_tokens, wedge, 1)
}

/// Where a prefill side past its own bound stands (see
/// [`prefill_side_reprieve`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SideReprieve {
    /// The worker answered another prompt, or a decode side showed one of
    /// its handoffs (see [`answered_or_handed_off`]), within the side's
    /// bound, or a cold-shape grace is left: the side is judged again after
    /// this.
    WithinBound(Duration),
    /// The worker answered nothing within the side's bound, but its current
    /// run of prompts has not answered yet and has not gone unanswered for
    /// the time its backlog takes: the side is judged again after this, a
    /// bound at most.
    FirstAnswerDue(Duration),
    /// The verdict: the worker was `silent` (`None`: it never answered) for
    /// the `window` it was judged against, the side's bound or the time its
    /// backlog takes.
    Silent {
        silent: Option<Duration>,
        window: Duration,
    },
}

impl SideReprieve {
    /// Whether the worker answered nothing within the side's bound and no
    /// cold-shape grace is left: where the side was given up on before a
    /// run's first answer was allowed for, and where a stalled decode
    /// partner still takes the verdict.
    pub(crate) fn past_bound(self) -> bool {
        !matches!(self, Self::WithinBound(_))
    }
}

/// Where a prefill side past its own bound stands, re-read at each check.
/// A side is given up on only when its worker is silent as a whole, as the
/// pile rule judges a worker by its pile and not by one request: a worker
/// that answered another prompt within the side's bound is backed up, not
/// frozen (on the parallel dispatch shape a side's first response also
/// waits for the decode side to admit it), so the side waits until a bound
/// has passed since that answer. The bound was read at the side's dispatch,
/// `dispatched_ago` back: a prompt longer than any the worker has answered
/// that landed behind the side since blocks the engine for the side too
/// while it builds, so while the worker is on that shape the side is given
/// what is left of the cold-shape grace since its dispatch, as a side
/// dispatched now would be ([`prefill_side_bound`]), a bound at a time so
/// the shape's first token returns it to the rule above.
///
/// On the parallel dispatch shape a decode side's first output is the
/// decode engine producing from the KV the worker handed off
/// ([`Worker::prefill_handed_off_ago`]), and it may come an engine step
/// before the prefill side's own response (a TokenSpeed prefill worker
/// answers once it has polled the transfer's success, after its next
/// forward). The rule above takes it for the worker's answer, until two
/// bounds after the later of the side's dispatch and the worker's latest
/// answer, so an engine that keeps handing off while its answers never
/// reach the gateway does not hold a side for good; on a worker that has
/// never answered it counts for nothing ([`answered_or_handed_off`]). A
/// handoff is good for one bound, as an answer is, so an engine step longer
/// than the bound still gets the verdict, after the handoff as before it:
/// an engine whose steps can take longer (a cold node building kernels)
/// needs a longer wedge threshold either way.
///
/// A worker whose current run of pending prompts has not answered yet (a
/// burst that reached it after an idle spell, or before it ever answered)
/// has shown no silence before the run began, and the side dispatched first
/// to its empty book has a short bound while the engine may well serve the
/// burst booked behind it first. Past the rule above, its side waits until
/// the run has gone unanswered ([`Worker::prefill_unanswered_run`]) for the
/// time the worker's backlog takes at its prefill rate, capped
/// ([`backlog_bound`]). That backlog is read once, the first time the rule
/// above gives up on the side, and kept in `backlog`, so prompts that go on
/// arriving at a frozen worker do not extend the wait; the side is judged
/// again a bound at a time, so a first answer returns it to the rule above
/// and a stalled decode partner is seen within a bound. A run that has
/// answered is judged by the rule above alone.
pub(crate) fn prefill_side_reprieve(
    worker: &Arc<dyn Worker>,
    bound: Duration,
    dispatched_ago: Duration,
    backlog: &OnceLock<Duration>,
) -> SideReprieve {
    side_reprieve(
        cold_shape_grace(worker, 0),
        answered_or_handed_off(
            worker.prefill_answered_ago(),
            worker.prefill_handed_off_ago(),
            bound,
            dispatched_ago,
        ),
        worker.prefill_unanswered_run(),
        bound,
        || *backlog.get_or_init(|| backlog_bound(worker)),
        dispatched_ago,
    )
}

/// The silence the rule of a side's `bound` judges its worker by: its latest
/// first response `answered` ago (`None`: never, whatever was handed off),
/// or the latest handoff a decode side showed, `handed_off` ago, while that
/// is newer than the answer and less than a bound old, but never less than
/// what has passed of a second bound since the later of the side's dispatch
/// (`dispatched_ago`) and that answer. A handoff so stands in for an answer
/// for at most a bound past where the answers alone leave the side; in that
/// credit the silence is below the bound (a reprieve, never a verdict), and
/// out of it the answer's age, as a verdict reports it.
fn answered_or_handed_off(
    answered: Option<Duration>,
    handed_off: Option<Duration>,
    bound: Duration,
    dispatched_ago: Duration,
) -> Option<Duration> {
    let answered = answered?;
    let past = answered.min(dispatched_ago).saturating_sub(bound);
    Some(match handed_off {
        Some(handed_off) if handed_off < answered && handed_off < bound && past < bound => {
            handed_off.max(past)
        }
        _ => answered,
    })
}

/// How long a worker whose run of prompts has not answered yet may take to
/// answer one: the time its whole pending backlog takes at its measured
/// prefill rate (the cold prior until one is measured), capped. A side's
/// own bound covers its prompt behind what was pending at its dispatch; the
/// prompts of the run booked since, which the engine need not serve after
/// it, are in the backlog by the time that bound passes. The backlog's time
/// counts once, as the pile rule counts it ([`wedge_bound`]), not twice as
/// a side's bound counts its prompt's: it only ever lengthens a side's
/// window past that bound, and a run that never answers is how a frozen
/// worker looks, whose sides twice the backlog would hold twice as long.
fn backlog_bound(worker: &Arc<dyn Worker>) -> Duration {
    worker.prefill_backlog().min(WEDGE_BOUND_CAP)
}

/// [`prefill_side_reprieve`] for a side of `bound` on a worker under a
/// cold-shape `grace` (zero: on no such shape), whose latest first response
/// (or the handoff standing in for it) was `answered_ago` (`None`: never),
/// and whose current run of prompts has gone `unanswered_run` without an
/// answer (`None`: nothing pending, or the run has answered). `backlog` is
/// read only once the rule of the bound gives up on the side. A grace the
/// side's bound already covers (a longer configured threshold) adds nothing.
fn side_reprieve(
    grace: Duration,
    answered_ago: Option<Duration>,
    unanswered_run: Option<Duration>,
    bound: Duration,
    backlog: impl FnOnce() -> Duration,
    dispatched_ago: Duration,
) -> SideReprieve {
    if grace > bound {
        if let Some(left) = grace
            .checked_sub(dispatched_ago)
            .filter(|left| !left.is_zero())
        {
            return SideReprieve::WithinBound(left.min(bound));
        }
    }
    if let Some(left) = reprieve(answered_ago, bound) {
        return SideReprieve::WithinBound(left);
    }
    let Some(silent) = unanswered_run else {
        return SideReprieve::Silent {
            silent: answered_ago,
            window: bound,
        };
    };
    let window = bound.max(backlog());
    match reprieve(Some(silent), window) {
        Some(left) => SideReprieve::FirstAnswerDue(left.min(bound)),
        None => SideReprieve::Silent {
            silent: Some(silent),
            window,
        },
    }
}

/// How much of `bound` is left to a worker silent for `silent` (`None`:
/// never answered): `None` once it has all passed.
fn reprieve(silent: Option<Duration>, bound: Duration) -> Option<Duration> {
    bound.checked_sub(silent?).filter(|left| !left.is_zero())
}

static THRESHOLDS: OnceLock<(Duration, Duration, Duration)> = OnceLock::new();
static WARMUP: OnceLock<Warmup> = OnceLock::new();
static EPOCH: OnceLock<Instant> = OnceLock::new();

/// The warm-up slice: one cache miss in `1 / share` is routed to a warming
/// worker (the least-loaded one) so it builds a cache instead of idling behind
/// the fleet's affinity. A worker is warming for `secs` after it became
/// routable, until its index has gained `blocks` blocks since (a new worker's
/// first cache), or, whatever its age, while its index is thin: holding less
/// than `thin_ratio` of the fleet's level (the median over healthy workers) or
/// nothing at all. A resync after a publisher restart, an `OUT_OF_RANGE` or
/// `DATA_LOSS`, or an engine that came back empty leaves a worker whose every
/// prompt has an overlap elsewhere, so no miss would ever reach it otherwise;
/// it stays warming until it crosses the ratio, however many blocks it regains
/// on the way (the `blocks` cap bounds the age rule only: on the churn run a
/// worker that regrew to 1,765 of a 32,767 level was dropped at the cap after
/// one diversion and idled for 25 minutes). `share == 0` disables;
/// `thin_ratio == 0` keeps the age rule alone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Warmup {
    pub secs: Duration,
    pub share: f32,
    pub blocks: usize,
    pub thin_ratio: f32,
    /// One hit in this many goes to a thin worker although another worker
    /// holds its prefix (0 disables): on a replay where every request has a
    /// holder the miss path never runs, and the slice alone would leave an
    /// emptied worker idle for good (see `CacheAwarePolicy::warmup_divert`).
    pub divert_every: u64,
}

impl Warmup {
    /// Whether a worker admitted `age` ago, holding `indexed` blocks against a
    /// fleet level of `fleet_level`, whose index has gained `growth` blocks
    /// since its admission (`None` when the index has never seen it; the
    /// baseline restarts when the count drops), is warming up: thin against
    /// the fleet, or young and short of its first `blocks`.
    pub(crate) fn applies(
        &self,
        age: Duration,
        growth: Option<usize>,
        indexed: usize,
        fleet_level: usize,
    ) -> bool {
        self.share > 0.0
            && (self.is_thin(indexed, fleet_level)
                || (age < self.secs && growth.is_none_or(|blocks| blocks < self.blocks)))
    }

    /// Whether an index of `indexed` blocks is thin against the fleet's
    /// `fleet_level`: empty, or below `thin_ratio` of the level, once the
    /// fleet holds a cache worth catching up to (a level of at least the
    /// warm-up `blocks`). A fleet below that (young, or tiny) makes nobody
    /// thin; the age rule decides there.
    pub(crate) fn is_thin(&self, indexed: usize, fleet_level: usize) -> bool {
        self.thin_ratio > 0.0
            && fleet_level >= self.blocks
            && (indexed == 0 || (indexed as f64) < fleet_level as f64 * f64::from(self.thin_ratio))
    }

    /// Every how many misses one goes to a warming worker.
    pub(crate) fn period(&self) -> u64 {
        if self.share <= 0.0 {
            return u64::MAX;
        }
        ((1.0 / f64::from(self.share)).round() as u64).max(1)
    }
}

const DEFAULT_WARMUP: Warmup = Warmup {
    secs: Duration::from_secs(60),
    share: 0.25,
    blocks: 1024,
    thin_ratio: 0.5,
    divert_every: 8,
};

/// Set the warm-up slice from the gateway configuration; the first call wins.
pub(crate) fn configure_warmup(warmup: Warmup) {
    let _ = WARMUP.set(warmup);
}

pub(crate) fn warmup() -> Warmup {
    WARMUP.get().copied().unwrap_or(DEFAULT_WARMUP)
}

/// Set the stall, wedge and stale thresholds from the gateway configuration.
/// The first call wins; the defaults are two, three and fifteen seconds.
pub(crate) fn configure(stall: Duration, wedge: Duration, stale: Duration) {
    let _ = THRESHOLDS.set((stall, wedge, stale));
}

/// Apply every liveness budget the configuration carries: the stall, wedge
/// and stale thresholds and the warm-up slice. The one place the budgets
/// are read from a configuration, for the server's start-up and for a
/// context built by hand (the test harness); the first call in a process
/// wins, as for [`configure`].
pub(crate) fn configure_from(config: &RouterConfig) {
    configure(
        Duration::from_secs(config.worker_stall_secs),
        Duration::from_secs(config.worker_wedge_secs),
        Duration::from_secs(config.worker_stale_secs),
    );
    configure_warmup(Warmup {
        secs: Duration::from_secs(config.worker_warmup_secs),
        share: config.worker_warmup_share,
        blocks: config.worker_warmup_blocks,
        thin_ratio: config.worker_warmup_thin_ratio,
        divert_every: config.worker_warmup_divert_every,
    });
}

fn thresholds() -> (Duration, Duration, Duration) {
    THRESHOLDS
        .get()
        .copied()
        .unwrap_or((DEFAULT_STALL, DEFAULT_WEDGE, DEFAULT_STALE))
}

/// The liveness budgets in force in this process: what a configuration set
/// through `configure_from`, or the defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budgets {
    /// A worker that has not answered a contact for this long is stalled.
    pub stall: Duration,
    /// A worker whose engine is silent on its requests this long is vetoed.
    pub wedge: Duration,
    /// A load report older than this is stale.
    pub stale: Duration,
    /// The warm-up slice's age bound.
    pub warmup_secs: Duration,
    /// The share of cache misses the warm-up slice diverts.
    pub warmup_share: f32,
    /// The index size that ends a young worker's warm-up.
    pub warmup_blocks: usize,
    /// The index ratio below which a worker is thin against the fleet.
    pub warmup_thin_ratio: f32,
    /// One hit in this many goes to a thin worker.
    pub warmup_divert_every: u64,
}

/// The liveness budgets in force in this process.
pub fn budgets() -> Budgets {
    let (stall, wedge, stale) = thresholds();
    let warmup = warmup();
    Budgets {
        stall,
        wedge,
        stale,
        warmup_secs: warmup.secs,
        warmup_share: warmup.share,
        warmup_blocks: warmup.blocks,
        warmup_thin_ratio: warmup.thin_ratio,
        warmup_divert_every: warmup.divert_every,
    }
}

/// Milliseconds since the gateway started: the clock behind the workers'
/// contact and progress stamps.
pub(crate) fn now_ms() -> u64 {
    u64::try_from(EPOCH.get_or_init(Instant::now).elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Whether a gRPC status code is one the connection produces rather than
/// the request: what a dead peer, a reset, a failed keepalive or an engine
/// that says it is unavailable end a stream with. A deadline or an
/// engine-side error says the worker is slow or wrong, not gone, and a slow
/// worker that still streams tokens must not flap in and out of routing.
///
/// The code alone decides, because the question is whether the peer is
/// reachable for work, and every way of being unreachable ends the
/// monitors' polls and streams with one of these codes; which layer
/// produced the status does not matter here. The gRPC client's
/// [`smg_grpc_client::is_transport_failure`] asks a different question of a
/// status under a call, whether it is the HTTP/2 layer's doing rather than
/// the worker's answer, and reads the status's source chain because the
/// code cannot tell those apart. The two are not interchangeable: an
/// `Unavailable` the worker answered with is a connection code here and no
/// transport failure there; an `Internal` made of a protocol fault is a
/// transport failure there and no connection code here.
pub(crate) fn is_connection_code(code: tonic::Code) -> bool {
    matches!(
        code,
        tonic::Code::Unavailable
            | tonic::Code::Unknown
            | tonic::Code::Cancelled
            | tonic::Code::Aborted
    )
}

/// A successful interaction over the transport (a poll answered, an event
/// batch, a response): the worker is reachable, so an unreachable veto ends
/// here. A worker the health checker demoted while it was gone (its probes
/// failed with its transport) is signalled to the manager, which probes it
/// now rather than at its next scheduled probe and promotes it on what the
/// probe reads (a ZMQ worker outright: its handshake is its health read);
/// any other demotion is the health checker's to lift, at its success
/// threshold. A reachable transport is not health: an engine whose
/// `/health` answers 503 while it drains, or a servicer whose health is held
/// through a warm-up, still relays KV batches and answers `GetLoads`, and a
/// worker promoted on one of those takes requests its engine is not ready
/// for.
pub(crate) fn on_contact(worker: &Arc<dyn Worker>) {
    worker.note_contact();
    if worker.stall_reason() != Some(StallReason::Unreachable) {
        return;
    }
    set(worker, None, "contact");
    if matches!(
        worker.status(),
        WorkerStatus::NotReady | WorkerStatus::Failed
    ) {
        worker.signal_connected();
    }
}

/// A health probe passed: contact, and the end of an unreachable veto, but
/// no promotion. The health state machine promotes on its success threshold
/// (a pending worker's reads an interval apart, a demoted worker's a probe
/// timeout apart, or an interval if that is shorter, once one passes); a
/// probe that short-circuited it would make one passing probe out of a
/// flapping engine's run a return to service.
pub(crate) fn on_probe_passed(worker: &Arc<dyn Worker>) {
    worker.note_contact();
    if worker.stall_reason() == Some(StallReason::Unreachable) {
        set(worker, None, "probe");
    }
}

/// A transport failure from `what`. Vetoes the worker when nothing has been
/// heard from it for the stall threshold; a failure right after a contact is
/// remembered, and [`sweep`] vetoes the worker once the threshold passes
/// without a contact. A worker vetoed as wedged is no exception: the failure
/// took the connection its pile was on, so the streams that were to show
/// progress are gone and only a contact can say the worker is back; its veto
/// becomes `Unreachable`, which the first contact clears. On the one-second
/// keepalive this could not arise, the transport failed before any pile
/// wedged; on the thirty-second profile a partitioned or frozen worker is
/// wedged at 3 s and loses its connection at 40 s, and a wedged veto that
/// waited for progress from streams the keepalive had already failed held
/// the worker out of routing for good.
pub(crate) fn on_contact_failed(worker: &Arc<dyn Worker>, what: &'static str) {
    on_contact_failed_with(worker, what, thresholds().0);
}

/// [`on_contact_failed`] at the given stall threshold.
fn on_contact_failed_with(worker: &Arc<dyn Worker>, what: &'static str, stall: Duration) {
    worker.note_transport_failure();
    if worker.stall_reason() == Some(StallReason::Unreachable) {
        return;
    }
    if stalled(worker.contact_age(), stall) {
        set(worker, Some(StallReason::Unreachable), what);
    }
}

/// Periodic check: a worker whose transport failed and that has then stayed
/// silent for the stall threshold is vetoed now, not at its next failed poll,
/// so the exclusion lands at the threshold itself.
pub(crate) fn sweep(worker: &Arc<dyn Worker>) {
    let (stall, wedge, stale) = thresholds();
    sweep_with(worker, stall, wedge, stale);
}

/// [`sweep`] at the given thresholds.
fn sweep_with(worker: &Arc<dyn Worker>, stall: Duration, wedge: Duration, stale: Duration) {
    let load = worker.tracked_load();
    let previous_load = worker.swap_load_sample(load);
    let contact_age = worker.contact_age();
    let silent_after_failure = worker.transport_failure_pending() && stalled(contact_age, stall);
    // The stream that pushed this worker's load records has gone quiet, and
    // so has everything else: unreachability the transport has not reported.
    let records_stopped = records_stopped(worker.pushes_load_records(), contact_age, stale);
    match worker.stall_reason() {
        Some(StallReason::Unreachable) => return,
        Some(StallReason::Wedged) => {
            // A wedged veto ends with its connection or with its pile. The
            // connection failed and the worker has been silent since: the
            // streams that were to show progress are gone, so the veto is
            // unreachable, which the first contact clears. Nothing in flight:
            // the pile that was the evidence is gone, and if the engine is
            // still stuck the next requests re-arm the veto within the wedge
            // bound; that is steering.
            if silent_after_failure {
                set(
                    worker,
                    Some(StallReason::Unreachable),
                    "silent since a transport failure",
                );
            } else if records_stopped {
                set(
                    worker,
                    Some(StallReason::Unreachable),
                    "load records stopped",
                );
            } else if load == 0 && !worker.engine_stalled() {
                // (Unless the engine's own record still says it is stuck:
                // then the evidence is the engine's, not this pile's.)
                set(worker, None, "drained");
            }
            return;
        }
        None => {}
    }
    if silent_after_failure {
        set(
            worker,
            Some(StallReason::Unreachable),
            "silent since a transport failure",
        );
        return;
    }
    if records_stopped {
        set(
            worker,
            Some(StallReason::Unreachable),
            "load records stopped",
        );
        return;
    }
    // The gateway's own view of a wedged engine: tracked requests pile up on
    // it and none has produced a response for the wedge bound. Needs no poll;
    // off at a zero threshold.
    if !wedge.is_zero()
        && wedged_by_pile(
            load,
            previous_load,
            worker.token_progress_age(),
            wedge_bound(worker, wedge),
        )
    {
        set(
            worker,
            Some(StallReason::Wedged),
            "in-flight requests pile up without progress",
        );
    }
}

/// A token or a completion from the worker: progress clears any veto.
pub(crate) fn on_token_progress(worker: &Arc<dyn Worker>) {
    worker.note_token_progress();
    if worker.stall_reason().is_some() {
        set(worker, None, "progress");
    }
}

/// A load report: the engine answers, but does it move? Wedged when tracked
/// requests are in flight, the engine reports a waiting queue (or one that
/// grew since the previous report) and no token or completion arrived within
/// the wedge threshold; wedged as well when the report itself says the engine
/// has produced nothing while it holds requests (`engine_silence`, from a
/// pushed record; a poll carries none), whatever this gateway has in flight
/// on it: past the allowance for the prefill the servicer reports in hand
/// (`engine_prefill_pending`, at the worker's rate, never under the wedge
/// threshold; see [`silence_allowance`]), which a wire that reports every
/// scheduler step (`engine_reports_steps`) gets from the cold prior while
/// the worker has shown no rate and a token-batch wire only once it has.
/// A standing veto is judged at the threshold on either wire, whatever
/// the servicer reports pending: the allowance lifts no veto, as the
/// cold-shape grace lifts none.
pub(crate) fn on_load_report(
    worker: &Arc<dyn Worker>,
    waiting: i64,
    engine_silence: Option<Duration>,
    engine_reports_steps: bool,
    engine_prefill_pending: Option<u64>,
) {
    let (_, wedge, _) = thresholds();
    on_load_report_with(
        worker,
        waiting,
        engine_silence,
        engine_reports_steps,
        engine_prefill_pending,
        wedge,
    );
}

/// [`on_load_report`] at the given wedge threshold.
fn on_load_report_with(
    worker: &Arc<dyn Worker>,
    waiting: i64,
    engine_silence: Option<Duration>,
    engine_reports_steps: bool,
    engine_prefill_pending: Option<u64>,
    wedge: Duration,
) {
    let previous = worker.swap_waiting_reqs(waiting);
    // The grace of a cold shape arms no veto and lifts none: a worker vetoed
    // before the longer prompt reached it is judged as before until it
    // produces, so the grace never reads as the engine producing again.
    let standing = worker.stall_reason().is_some();
    let grace = if standing {
        Duration::ZERO
    } else {
        cold_shape_grace(worker, 0)
    };
    let bound = wedge_bound_given(worker, wedge, grace);
    // Nor does the prefill allowance lift one, on either wire: a long prompt
    // that reaches a vetoed worker (a veto steers, it never refuses)
    // stretches the servicer's pending figure, and must not read as the
    // engine producing; a standing veto is judged at the bare threshold
    // until the engine does produce (its silence restarts), or a token
    // arrives.
    let allowance = if standing {
        Some(wedge)
    } else {
        silence_allowance(worker, wedge, engine_prefill_pending, engine_reports_steps)
    };
    let engine_stalled = engine_stalled(
        engine_silence,
        engine_reports_steps,
        wedge,
        allowance,
        grace,
    );
    worker.note_engine_stall(engine_stalled);
    match worker.stall_reason() {
        Some(StallReason::Wedged) => {
            // A clock that restarted with this gateway's first dispatch is
            // not progress while the engine still reports silence: a token
            // clears that veto through `on_token_progress`.
            if !engine_stalled && worker.token_progress_age() < wedge {
                set(worker, None, "progress");
            } else if !engine_stalled
                && (worker.tracked_load() == 0 || worker.token_progress_age() < bound)
            {
                // The engine produces again and this gateway holds no silent
                // pile of its own on it: whatever armed the veto is over.
                set(worker, None, "engine output resumed");
            }
        }
        Some(StallReason::Unreachable) => {}
        None => {
            if engine_stalled {
                set(
                    worker,
                    Some(StallReason::Wedged),
                    "the engine reports no output for its in-flight requests",
                );
            } else if !wedge.is_zero()
                && wedged_by_queue(
                    worker.tracked_load(),
                    waiting,
                    previous,
                    worker.token_progress_age(),
                    bound,
                )
            {
                set(
                    worker,
                    Some(StallReason::Wedged),
                    "no progress with a waiting queue",
                );
            }
        }
    }
}

/// The engine-reported wedge rule on its inputs: the servicer says its
/// engine has produced nothing for `silence` while holding requests, and
/// that is the `allowance` or longer (the prefill the servicer reports in
/// hand at the worker's rate, never under the wedge threshold; see
/// [`silence_allowance`]), or the wedge threshold or longer when there is no
/// allowance and the wire reports every scheduler step (`reports_steps`:
/// the servicer reports no pending figure, and the silence is the
/// scheduler's own), or the cold-shape `grace` or longer when that is more.
/// A token-batch wire without an allowance gets no verdict: a whole prefill
/// is silence there, and nothing sizes it. Off at a zero threshold; a
/// record without the figure (an idle engine, an older servicer) says
/// nothing.
fn engine_stalled(
    silence: Option<Duration>,
    reports_steps: bool,
    wedge: Duration,
    allowance: Option<Duration>,
    grace: Duration,
) -> bool {
    let allowed = match allowance {
        Some(allowance) => allowance,
        None if reports_steps => wedge,
        None => return false,
    };
    !wedge.is_zero() && silence.is_some_and(|silence| silence >= allowed.max(grace))
}

/// What the engine may be silent for before its silence counts: the
/// configured threshold or, if longer, twice the time the engine may need to
/// prefill the prompt tokens its servicer reports in hand
/// (`EngineLoad.prefill_pending_tokens`: every client's requests, not just
/// this gateway's, so a gateway with nothing of its own on the worker allows
/// for the same prefill) at the prefill rate this gateway has measured on
/// the worker (twice, the rate being an estimate), under the same cap as the
/// wedge bound. `None` without the pending figure (an older servicer). A
/// token-batch wire is silent through a whole prefill, so its allowance is
/// `None` as well, and the rule gives no verdict, while the worker has
/// shown no rate: the cold prior behind the wedge bound is a guess, not
/// evidence against an engine that prefills slower than it. A wire that
/// reports every scheduler step (`reports_steps`) puts one step on the
/// allowance at most (the step that starts a long multimodal prompt shows
/// nothing until the encoder has run over the clip's frames), and the bare
/// threshold is what judged that step before, so the cold prior sizes its
/// allowance until the worker shows a rate. The pending figure is a sum
/// over every request the engine has produced nothing for, with no
/// per-request figure beside it, so a frozen engine behind a deep queue of
/// short prompts is caught at twice the queue's prefill time rather than
/// the threshold (the gateway's own pile rule allows its in-flight prompts
/// the same); a standing veto is judged without the allowance (see
/// [`on_load_report`]).
fn silence_allowance(
    worker: &Arc<dyn Worker>,
    wedge: Duration,
    engine_prefill_pending: Option<u64>,
    reports_steps: bool,
) -> Option<Duration> {
    let pending = engine_prefill_pending?;
    let rate = match worker.prefill_rate_tps() {
        0 if reports_steps => COLD_PREFILL_TOKENS_PER_SEC,
        0 => return None,
        rate => rate,
    };
    let prefill = Duration::from_millis(pending.saturating_mul(2_000) / rate);
    Some(wedge.max(prefill).min(wedge.max(WEDGE_BOUND_CAP)))
}

/// The unreachable rule on its inputs.
fn stalled(contact_age: Duration, stall: Duration) -> bool {
    contact_age >= stall
}

/// The stale-records rule on its inputs: a worker whose stream has pushed
/// load records, heard from on no channel for the stale threshold. Off at a
/// zero threshold.
fn records_stopped(pushes_load_records: bool, contact_age: Duration, stale: Duration) -> bool {
    pushes_load_records && !stale.is_zero() && contact_age >= stale
}

/// The wedged rule from an engine's load report: work in flight, a waiting
/// queue (or one that grew), and silence for the wedge threshold.
fn wedged_by_queue(
    in_flight: usize,
    waiting: i64,
    previous: i64,
    token_age: Duration,
    wedge: Duration,
) -> bool {
    in_flight > 0 && (waiting > 0 || waiting > previous) && token_age >= wedge
}

/// The wedged rule from the gateway's own counters: the pile of in-flight
/// requests grew, or is already deep, and nothing moved for the threshold.
fn wedged_by_pile(
    in_flight: usize,
    previous_in_flight: usize,
    token_age: Duration,
    wedge: Duration,
) -> bool {
    token_age >= wedge
        && in_flight > 0
        && (in_flight > previous_in_flight || in_flight >= WEDGE_PILE)
}

fn set(worker: &Arc<dyn Worker>, reason: Option<StallReason>, cause: &'static str) {
    let previous = worker.stall_reason();
    if !worker.set_stall(reason) {
        return;
    }
    match reason {
        Some(reason) => {
            warn!(
                worker_url = %worker.redacted_url(),
                reason = reason.as_str(),
                cause,
                contact_age_ms = u64::try_from(worker.contact_age().as_millis()).unwrap_or(u64::MAX),
                "Worker vetoed by liveness"
            );
            // A veto that changes reason (wedged, then unreachable once its
            // connection failed) leaves one gauge up, not two.
            if let Some(previous) = previous.filter(|previous| *previous != reason) {
                Metrics::set_worker_stalled(worker.url(), previous.as_str(), false);
            }
            // The channel the engine's silence came on is gone with the
            // transport; the next record speaks for the engine again.
            if reason == StallReason::Unreachable {
                worker.note_engine_stall(false);
            }
            Metrics::set_worker_stalled(worker.url(), reason.as_str(), true);
        }
        None => {
            info!(worker_url = %worker.redacted_url(), cause, "Worker re-admitted by liveness");
            worker.note_admitted();
            if let Some(previous) = previous {
                // The outage's connection failures opened the circuit breaker
                // as well, and it would hold the worker out of routing for
                // its timeout (30 s by default) and reopen on one error from
                // a stale channel. The contact that clears the veto says the
                // worker is back; fresh failures reopen the breaker as usual.
                if previous == StallReason::Unreachable {
                    worker.reset_circuit_breaker();
                }
                Metrics::set_worker_stalled(worker.url(), previous.as_str(), false);
            }
        }
    }
}

#[cfg(test)]
mod tests;
