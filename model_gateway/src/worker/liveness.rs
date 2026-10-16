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
//!   bound instead of once a pile forms. Which bound depends on the wire:
//!   an engine whose wire reports every scheduler step
//!   (`engine_reports_steps`: it has shown a stats-only step batch, which
//!   vLLM sends every step with stats logging on, or its launcher told the
//!   servicer that logging is on) is silent only when its scheduler is, so
//!   the bare threshold applies; a wire that carries token batches only
//!   (SGLang, TokenSpeed, vLLM with stats logging off) is silent through a
//!   long prefill as well, so its silence is judged against the prefill its
//!   servicer reports in hand (`prefill_pending_tokens`: every client's
//!   requests, so a gateway with nothing of its own on the worker allows for
//!   the same prefill) at the prefill rate this gateway has measured on that
//!   worker (twice over, the rate being an estimate, under the wedge bound's
//!   cap), and only once the worker has shown a rate: the cold prior behind
//!   the bound is a guess, not evidence against an engine that prefills
//!   slower than it. That veto ends with
//!   progress, or with a record that shows the engine producing again while
//!   this gateway has no silent pile of its own. The requests counted are the
//!   tracked ones, streaming generations to the worker over gRPC, whose
//!   responses the gateway sees one by one ([`Worker::tracked_load`]): an
//!   HTTP worker, a PD leg or a non-streaming generation gives no signal
//!   between dispatch and completion, so it never forms a pile and is never
//!   judged by one; a non-streaming generation's one answer is progress all
//!   the same, and its prompt counts in the prefill backlog (but times no
//!   prefill: the gateway never sees its first token). The clock starts at
//!   the first dispatch of a run of
//!   tracked requests, never at registration, and the bound is the
//!   configured threshold or, if longer, the time the engine may still need
//!   to prefill what is in flight ([`Worker::prefill_backlog`]): a batch that
//!   is all in prefill streams nothing and is not wedged. A zero threshold
//!   turns the rule off. The same bound, read per request, caps how long a
//!   disaggregated prefill leg may stay silent ([`prefill_leg_bound`]): a
//!   leg is not part of any pile, so its request is what fails, over to
//!   another prefill worker when one is available. A leg past its bound is
//!   given up on only while its worker has answered nothing within the
//!   bound ([`prefill_leg_reprieve`]): a worker that keeps answering other
//!   legs is backed up (on the parallel dispatch shape a leg's first
//!   response also waits for the decode side), not frozen, and its leg
//!   waits on.
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

use super::{worker::StallReason, Worker};
use crate::observability::metrics::Metrics;

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

/// The wedge bound for `worker` now: the configured threshold, or the time
/// the engine may still need before the first token of its in-flight prompts
/// is due, whichever is longer.
fn wedge_bound(worker: &Arc<dyn Worker>, wedge: Duration) -> Duration {
    bound_over(wedge, worker.prefill_backlog())
}

/// The wedge rule's bound over an `eta` the engine may still need: the
/// configured threshold or the ETA, whichever is longer, capped.
fn bound_over(wedge: Duration, eta: Duration) -> Duration {
    wedge.max(eta).min(wedge.max(WEDGE_BOUND_CAP))
}

/// How long a disaggregated prefill leg of `prompt_tokens` dispatched to
/// `worker` now may produce nothing before its request gives up on it: the
/// wedge rule's bound, with the leg's own prompt behind what the worker is
/// already prefilling ([`Worker::prefill_eta`]), so a long prompt on a busy
/// worker gets the time its measured rate says it needs, a short one the
/// configured threshold, and nothing more than the cap. A PD leg joins no
/// pile and is never vetoed by one; the request is what the bound ends.
/// `None` when the rule is off (a zero threshold).
pub(crate) fn prefill_leg_bound(worker: &Arc<dyn Worker>, prompt_tokens: u64) -> Option<Duration> {
    let (_, wedge, _) = thresholds();
    prefill_leg_bound_with(worker, prompt_tokens, wedge)
}

/// [`prefill_leg_bound`] at the given wedge threshold.
fn prefill_leg_bound_with(
    worker: &Arc<dyn Worker>,
    prompt_tokens: u64,
    wedge: Duration,
) -> Option<Duration> {
    (!wedge.is_zero()).then(|| bound_over(wedge, worker.prefill_eta(prompt_tokens)))
}

/// How long a sequential disaggregated decode leg of `prompt_tokens`
/// dispatched now may produce nothing before its request gives up on it:
/// the prefill leg's bound for the same prompt on `prefills`, the worker
/// whose prefill of the prompt stands before the leg's first token
/// ([`prefill_leg_bound`]): the prefill worker when the leg was handed its
/// KV (the handoff moves what that prefill produced), the decode worker when
/// it recomputes the prompt itself (no handoff: a fan-out, or a prefill that
/// returned no transfer parameters). The configured threshold for a short
/// prompt, the time the worker's measured prefill rate says a long prompt
/// takes, and nothing more than the cap. The leg was admitted to a room on
/// its decode worker before its dispatch. `None` when the rule is off (a
/// zero threshold).
pub(crate) fn decode_leg_bound(prefills: &Arc<dyn Worker>, prompt_tokens: u64) -> Option<Duration> {
    prefill_leg_bound(prefills, prompt_tokens)
}

/// Whether a prefill leg past its own bound waits on, and how much longer.
/// A leg is given up on only when its worker is silent as a whole, with no
/// first response on any prompt within the bound, as the pile rule judges
/// a worker by its pile and not by one request. A worker that answered
/// another leg a moment ago is backed up, not frozen (on the parallel
/// dispatch shape a leg's first response also waits for the decode side to
/// admit it), so the leg waits until a bound has passed since that answer.
/// `None` when the worker has answered nothing within the bound, or ever:
/// the verdict.
pub(crate) fn prefill_leg_reprieve(worker: &Arc<dyn Worker>, bound: Duration) -> Option<Duration> {
    reprieve(worker.prefill_answered_ago(), bound)
}

/// [`prefill_leg_reprieve`] for a worker whose latest first response was
/// `answered_ago` (`None`: never).
fn reprieve(answered_ago: Option<Duration>, bound: Duration) -> Option<Duration> {
    bound
        .checked_sub(answered_ago?)
        .filter(|left| !left.is_zero())
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

fn thresholds() -> (Duration, Duration, Duration) {
    THRESHOLDS
        .get()
        .copied()
        .unwrap_or((DEFAULT_STALL, DEFAULT_WEDGE, DEFAULT_STALE))
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
/// failed with its transport) is promoted now rather than at its next
/// scheduled probe; any other demotion is the health checker's to lift, at
/// its success threshold. A reachable transport is not health: an engine
/// whose `/health` answers 503 while it drains still relays KV batches and
/// answers `GetLoads`, and promoting it on each of those would flap it back
/// to Ready within milliseconds of every demotion.
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
/// no promotion. The health state machine promotes on its success threshold;
/// a probe that short-circuited it would make one passing probe out of a
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
/// on it: for the wedge threshold when the engine's wire reports every
/// scheduler step (`engine_reports_steps`), for the prefill-stretched bound
/// when it carries token batches only and a long prefill is silence too.
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
    let bound = wedge_bound(worker, wedge);
    let engine_stalled = engine_stalled(
        engine_silence,
        engine_reports_steps,
        wedge,
        silence_allowance(worker, wedge, engine_prefill_pending),
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
/// that is the wedge threshold or longer when the wire reports every
/// scheduler step (`reports_steps`: the silence is the scheduler's own), or
/// the `allowance` or longer when it carries token batches only (a long
/// prefill is silence there; the allowance covers the prefill the servicer
/// reports in hand, and is `None` when this gateway cannot size it). Off at
/// a zero threshold; a record without the figure (an idle engine, an older
/// servicer) says nothing.
fn engine_stalled(
    silence: Option<Duration>,
    reports_steps: bool,
    wedge: Duration,
    allowance: Option<Duration>,
) -> bool {
    let allowed = if reports_steps {
        wedge
    } else if let Some(allowance) = allowance {
        allowance
    } else {
        return false;
    };
    !wedge.is_zero() && silence.is_some_and(|silence| silence >= allowed)
}

/// What a token-batch wire may be silent for before its silence counts: the
/// configured threshold or, if longer, twice the time the engine may need to
/// prefill the prompt tokens its servicer reports in hand
/// (`EngineLoad.prefill_pending_tokens`: every client's requests, not just
/// this gateway's, so a gateway with nothing of its own on the worker allows
/// for the same prefill) at the prefill rate this gateway has measured on
/// the worker (twice, the rate being an estimate), under the same cap as the
/// wedge bound. `None`, and so no verdict, while the servicer reports no
/// pending figure or the worker has shown no rate: the cold prior behind the
/// wedge bound is a guess, not evidence against an engine that prefills
/// slower than it.
fn silence_allowance(
    worker: &Arc<dyn Worker>,
    wedge: Duration,
    engine_prefill_pending: Option<u64>,
) -> Option<Duration> {
    let pending = engine_prefill_pending?;
    let rate = worker.prefill_rate_tps();
    if rate == 0 {
        return None;
    }
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
                worker_url = %worker.url(),
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
            info!(worker_url = %worker.url(), cause, "Worker re-admitted by liveness");
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
mod tests {
    use std::thread;

    use smg_grpc_client::{h2, is_transport_failure};
    use tokio::sync::mpsc;

    use super::*;
    use crate::worker::{
        circuit_breaker::CircuitBreakerConfig, event::WorkerConnected, BasicWorkerBuilder,
    };

    fn worker() -> Arc<dyn Worker> {
        Arc::new(BasicWorkerBuilder::new("http://w1:8000").build())
    }

    /// The expectation the three callers share, for every gRPC code: the
    /// load poll, the KV subscription and the KV stream all mark a worker
    /// as contact failed when their call or stream ends with a connection
    /// code, and leave it alone on any other code. Four codes are the
    /// connection's; the rest describe the request. A `match` with no
    /// wildcard arm: a code added to tonic is a build error here until it
    /// has its row.
    fn expectation(code: tonic::Code) -> (bool, &'static str) {
        use tonic::Code;
        match code {
            Code::Ok => (false, "a clean end is not a failure"),
            Code::Cancelled => (true, "a reset stream: the peer or the link went away"),
            Code::Unknown => (
                true,
                "an error with no gRPC code: a closed or broken connection",
            ),
            Code::InvalidArgument => (false, "the request was wrong, the worker is there"),
            Code::DeadlineExceeded => (false, "our own deadline: slow, not gone"),
            Code::NotFound => (false, "the request's"),
            Code::AlreadyExists => (false, "the request's"),
            Code::PermissionDenied => (false, "the request's"),
            Code::ResourceExhausted => (false, "the worker's capacity pushback: busy, not gone"),
            Code::FailedPrecondition => (false, "the request's"),
            Code::Aborted => (true, "the worker aborted the stream under us"),
            Code::OutOfRange => (
                false,
                "the KV subscription resyncs on it before asking; never a contact failure",
            ),
            Code::Unimplemented => (false, "the request's"),
            Code::Internal => (
                false,
                "the worker's own failure on a request: wrong, not gone",
            ),
            Code::Unavailable => (
                true,
                "a dead peer, a refused stream or a worker that says it is unavailable",
            ),
            Code::DataLoss => (
                false,
                "the KV stream resyncs on it before asking; never a contact failure",
            ),
            Code::Unauthenticated => (false, "the request's"),
        }
    }

    /// Every gRPC code against the callers' expectation. The codes come
    /// from tonic's own numbering: `from_i32` answers `Unknown` for a
    /// number tonic has no code for, so the numbers must map back to
    /// themselves (each a real code, once) and the 17th must not.
    #[test]
    fn a_connection_code_is_one_of_four_and_every_other_code_is_the_requests() {
        use tonic::Code;
        let codes: Vec<Code> = (0..=16).map(Code::from_i32).collect();
        assert_eq!(
            codes
                .iter()
                .map(|&code| i32::from(code))
                .collect::<Vec<_>>(),
            (0..=16).collect::<Vec<_>>()
        );
        assert_eq!(
            Code::from_i32(17),
            Code::Unknown,
            "tonic numbers a 17th code: give it a row"
        );
        for &code in &codes {
            let (expected, why) = expectation(code);
            assert_eq!(is_connection_code(code), expected, "{code:?}: {why}");
        }
        assert_eq!(codes.iter().filter(|&&code| expectation(code).0).count(), 4);
    }

    /// The gRPC client's `is_transport_failure` answers another question
    /// (is this status the HTTP/2 layer's doing rather than the worker's
    /// answer?) from the status's source chain, and the two disagree by
    /// design in both directions.
    #[test]
    fn a_connection_code_and_a_transport_failure_are_different_questions() {
        // An Unavailable the worker answered with: unreachable for work, yet
        // not the transport's doing.
        let answered = tonic::Status::unavailable("shutting down");
        assert!(is_connection_code(answered.code()));
        assert!(!is_transport_failure(&answered));

        // An Internal tonic made of a protocol fault: the transport's doing,
        // yet its code is the one a worker's own failure carries.
        let fault =
            tonic::Status::from_error(Box::new(h2::Error::from(h2::Reason::PROTOCOL_ERROR)));
        assert_eq!(fault.code(), tonic::Code::Internal);
        assert!(is_transport_failure(&fault));
        assert!(!is_connection_code(fault.code()));

        // A refused stream is both: an Unavailable with an HTTP/2 cause.
        let refused =
            tonic::Status::from_error(Box::new(h2::Error::from(h2::Reason::REFUSED_STREAM)));
        assert_eq!(refused.code(), tonic::Code::Unavailable);
        assert!(is_transport_failure(&refused));
        assert!(is_connection_code(refused.code()));
    }

    #[test]
    fn unreachable_needs_a_stall_not_just_a_failure() {
        assert!(!stalled(Duration::from_millis(500), DEFAULT_STALL));
        assert!(stalled(Duration::from_secs(2), DEFAULT_STALL));
    }

    #[test]
    fn wedged_by_queue_needs_in_flight_work_a_queue_and_silence() {
        let wedge = DEFAULT_WEDGE;
        assert!(wedged_by_queue(3, 5, 2, Duration::from_secs(4), wedge));
        assert!(
            wedged_by_queue(3, 5, 5, Duration::from_secs(4), wedge),
            "a standing queue counts once the client stops adding to it"
        );
        assert!(
            !wedged_by_queue(0, 5, 2, Duration::from_secs(4), wedge),
            "nothing in flight"
        );
        assert!(
            !wedged_by_queue(3, 0, 0, Duration::from_secs(4), wedge),
            "no queue: a long prefill, not a wedge"
        );
        assert!(
            !wedged_by_queue(3, 5, 2, Duration::from_secs(1), wedge),
            "tokens still flowing"
        );
    }

    #[test]
    fn wedged_by_pile_needs_growth_or_depth_and_silence() {
        let wedge = DEFAULT_WEDGE;
        assert!(
            wedged_by_pile(2, 1, Duration::from_secs(4), wedge),
            "growing"
        );
        assert!(wedged_by_pile(4, 4, Duration::from_secs(4), wedge), "deep");
        assert!(
            !wedged_by_pile(1, 1, Duration::from_secs(4), wedge),
            "one quiet request could be prefilling"
        );
        assert!(
            !wedged_by_pile(8, 4, Duration::from_secs(1), wedge),
            "tokens still flowing"
        );
    }

    #[test]
    fn the_first_requests_after_registration_are_not_a_wedge() {
        // The policy lane saw every mock worker vetoed 3.5 s after
        // registration when the first requests arrived: the no-progress clock
        // counted from registration. It counts from the dispatch now.
        let w = worker();
        thread::sleep(Duration::from_millis(15));
        for _ in 0..3 {
            w.increment_load();
            w.note_tracked_started();
        }
        let age = w.token_progress_age();
        assert!(
            age < Duration::from_millis(10),
            "clock started at the dispatch"
        );
        sweep(&w);
        assert!(w.stall_reason().is_none());
        on_load_report(&w, 3, None, false, None);
        assert!(w.stall_reason().is_none());
        // The same pile with a clock that had run from registration would be
        // the false positive.
        assert!(wedged_by_pile(
            3,
            0,
            Duration::from_millis(3_500),
            DEFAULT_WEDGE
        ));
        assert!(!wedged_by_pile(3, 0, age, DEFAULT_WEDGE));
    }

    #[test]
    fn a_batch_still_in_prefill_is_not_a_wedge() {
        // The GPU lane saw the veto fire with no token for 3 to 5 s while a
        // running batch was all in prefill: 128 prompts of 1,152 tokens.
        let w = worker();
        for _ in 0..128 {
            w.increment_load();
        }
        w.note_prefill_started(128 * 1_152, true);
        let bound = wedge_bound(&w, DEFAULT_WEDGE);
        assert_eq!(
            bound,
            Duration::from_millis(14_745),
            "cold prior: 10k tokens/s"
        );
        let silent = Duration::from_millis(5_300);
        assert!(!wedged_by_pile(128, 0, silent, bound));
        assert!(!wedged_by_queue(128, 16, 0, silent, bound));
        assert!(
            wedged_by_pile(128, 0, silent, DEFAULT_WEDGE),
            "the bare threshold would have fired"
        );
        // The first tokens arrive: the backlog is gone and the bound is the
        // configured threshold again.
        w.note_prefill_ended(128 * 1_152, true, Some(0));
        assert_eq!(wedge_bound(&w, DEFAULT_WEDGE), DEFAULT_WEDGE);
        // The bound is capped against leaking books.
        w.note_prefill_started(100_000_000, true);
        assert_eq!(wedge_bound(&w, DEFAULT_WEDGE), WEDGE_BOUND_CAP);
        assert_eq!(
            wedge_bound(&w, Duration::from_secs(600)),
            Duration::from_secs(600),
            "a longer configured threshold stands"
        );
    }

    #[test]
    fn a_paused_engine_with_short_prompts_is_still_a_wedge() {
        // The pause drill: eight chat prompts of ~150 tokens in flight, the
        // engine frozen, health and load polls still answering.
        let w = worker();
        for _ in 0..8 {
            w.increment_load();
        }
        w.note_prefill_started(8 * 150, true);
        let bound = wedge_bound(&w, DEFAULT_WEDGE);
        assert_eq!(
            bound, DEFAULT_WEDGE,
            "0.12 s of prefill is inside the threshold"
        );
        let silent = Duration::from_millis(3_200);
        assert!(wedged_by_queue(8, 8, 8, silent, bound), "standing queue");
        assert!(wedged_by_pile(8, 8, silent, bound), "deep pile");
        assert!(!wedged_by_pile(8, 8, Duration::from_millis(2_900), bound));
    }

    #[test]
    fn a_prefill_legs_bound_follows_its_prompt_the_workers_rate_and_the_cap() {
        // A disaggregated prefill leg is bounded like the pile rule bounds a
        // worker: the threshold for a short prompt on an idle worker, the
        // prefill time its prompt needs behind what is already pending at
        // the worker's rate (the cold prior of 10k tokens/s until one is
        // measured), never more than the cap.
        let w = worker();
        assert_eq!(
            prefill_leg_bound_with(&w, 256, DEFAULT_WEDGE),
            Some(DEFAULT_WEDGE),
            "26 ms of prefill is inside the threshold"
        );
        assert_eq!(
            prefill_leg_bound_with(&w, 32_768, DEFAULT_WEDGE),
            Some(Duration::from_millis(3_276)),
            "a 32k prompt on a cold worker"
        );
        w.note_prefill_started(8 * 32_768, true);
        assert_eq!(
            prefill_leg_bound_with(&w, 32_768, DEFAULT_WEDGE),
            Some(Duration::from_millis(29_491)),
            "behind eight pending 32k prompts"
        );
        w.note_prefill_ended(8 * 32_768, true, Some(0));
        assert_eq!(
            prefill_leg_bound_with(&w, 100_000_000, DEFAULT_WEDGE),
            Some(WEDGE_BOUND_CAP),
            "capped"
        );
        assert_eq!(
            prefill_leg_bound_with(&w, 100_000_000, Duration::from_secs(600)),
            Some(Duration::from_secs(600)),
            "a longer configured threshold stands"
        );
        assert_eq!(
            prefill_leg_bound_with(&w, 32_768, Duration::ZERO),
            None,
            "a zero threshold turns the bound off"
        );
    }

    #[test]
    fn a_prefill_leg_past_its_bound_waits_on_while_its_worker_answers_other_legs() {
        // The verdict is on the worker, not the leg: a worker that has
        // answered nothing (ever, or within the bound) is silent and the leg
        // is given up on; one that answered another leg a moment ago is
        // backed up, and the leg waits until a bound has passed since that
        // answer.
        let bound = DEFAULT_WEDGE;
        assert_eq!(reprieve(None, bound), None, "never answered: silent");
        assert_eq!(
            reprieve(Some(Duration::from_millis(600)), bound),
            Some(Duration::from_millis(2_400)),
            "answered 0.6 s ago: the leg waits the rest of a bound"
        );
        assert_eq!(
            reprieve(Some(bound), bound),
            None,
            "a bound has passed since the latest answer"
        );
        assert_eq!(reprieve(Some(Duration::from_secs(10)), bound), None);
        let w = worker();
        assert_eq!(
            prefill_leg_reprieve(&w, bound),
            None,
            "a fresh worker has answered nothing"
        );
        w.note_prefill_started(256, true);
        w.note_prefill_ended(256, true, Some(0));
        assert!(
            prefill_leg_reprieve(&w, bound).is_some_and(|left| left > Duration::from_secs(2)),
            "answered just now: a bound's reprieve, less the moment since"
        );
    }

    #[test]
    fn records_stopped_needs_a_pushing_worker_silent_for_the_threshold() {
        let stale = DEFAULT_STALE;
        assert!(records_stopped(true, Duration::from_secs(15), stale));
        assert!(
            !records_stopped(true, Duration::from_secs(14), stale),
            "inside the threshold"
        );
        assert!(
            !records_stopped(false, Duration::from_secs(60), stale),
            "a worker that never pushed a record is polled and probed, not judged by silence"
        );
        assert!(
            !records_stopped(true, Duration::from_secs(60), Duration::ZERO),
            "zero turns the rule off"
        );
    }

    #[test]
    fn a_pushing_worker_whose_records_stop_is_vetoed_and_the_next_contact_restores_it() {
        let stale = Duration::from_millis(20);
        let w = worker();
        // Nothing pushed yet: the silence of a polled worker is not a verdict.
        thread::sleep(Duration::from_millis(30));
        sweep_with(&w, DEFAULT_STALL, DEFAULT_WEDGE, stale);
        assert!(w.stall_reason().is_none(), "never pushed");
        // The stream delivers a record: from here on silence counts.
        w.note_load_record();
        on_contact(&w);
        sweep_with(&w, DEFAULT_STALL, DEFAULT_WEDGE, stale);
        assert!(w.stall_reason().is_none(), "just heard from");
        thread::sleep(Duration::from_millis(30));
        sweep_with(&w, DEFAULT_STALL, DEFAULT_WEDGE, stale);
        assert_eq!(w.stall_reason(), Some(StallReason::Unreachable));
        assert!(!w.routing_state().eligible());
        // The next record, poll answer or probe is a contact: re-admitted.
        on_contact(&w);
        assert!(w.stall_reason().is_none());
        assert!(!w.routing_state().stalled);
        // A worker that still streams tokens is heard from: never stale.
        thread::sleep(Duration::from_millis(30));
        on_token_progress(&w);
        sweep_with(&w, DEFAULT_STALL, DEFAULT_WEDGE, stale);
        assert!(w.stall_reason().is_none(), "tokens are contact");
        // Off at zero.
        thread::sleep(Duration::from_millis(30));
        sweep_with(&w, DEFAULT_STALL, DEFAULT_WEDGE, Duration::ZERO);
        assert!(w.stall_reason().is_none(), "rule off");
    }

    #[test]
    fn a_wedged_worker_whose_records_stop_becomes_unreachable() {
        let stale = Duration::from_millis(20);
        let w = worker();
        w.note_load_record();
        w.increment_load();
        w.note_tracked_started();
        set(&w, Some(StallReason::Wedged), "test");
        sweep_with(&w, DEFAULT_STALL, DEFAULT_WEDGE, stale);
        assert_eq!(
            w.stall_reason(),
            Some(StallReason::Wedged),
            "still heard from"
        );
        thread::sleep(Duration::from_millis(30));
        sweep_with(&w, DEFAULT_STALL, DEFAULT_WEDGE, stale);
        assert_eq!(w.stall_reason(), Some(StallReason::Unreachable));
        on_contact(&w);
        assert!(w.stall_reason().is_none());
    }

    #[test]
    fn a_veto_removes_the_worker_from_routing_and_contact_restores_it() {
        let w = worker();
        assert!(w.stall_reason().is_none());
        set(&w, Some(StallReason::Unreachable), "test");
        assert_eq!(w.stall_reason(), Some(StallReason::Unreachable));
        assert!(w.routing_state().stalled);
        assert!(!w.routing_state().eligible());
        assert!(!w.is_healthy_and_eligible());
        on_contact(&w);
        assert!(w.stall_reason().is_none());
        assert!(!w.routing_state().stalled);
    }

    #[test]
    fn a_contact_that_clears_the_unreachable_veto_closes_the_breaker() {
        let w: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );
        for _ in 0..8 {
            w.record_circuit_breaker_outcome(false);
        }
        assert!(
            !w.circuit_breaker_can_execute(),
            "the outage's failures opened the breaker"
        );
        set(&w, Some(StallReason::Unreachable), "test");
        on_contact(&w);
        assert!(w.stall_reason().is_none());
        assert!(
            w.circuit_breaker_can_execute(),
            "back in routing at once, breaker closed"
        );
    }

    /// A pile of `n` tracked requests dispatched now.
    fn tracked_pile(w: &Arc<dyn Worker>, n: usize) {
        for _ in 0..n {
            w.increment_load();
            w.note_tracked_started();
        }
    }

    #[test]
    fn an_untracked_pile_is_never_a_wedge() {
        // An HTTP worker under steady traffic (review finding): four in
        // flight for longer than the wedge bound, no token signal because
        // HTTP responses are not tracked, and before the fix the sweep vetoed
        // it as wedged 3 s into its busy run with nothing to clear it. The
        // same for a PD leg and a non-streaming gRPC generation.
        let w = worker();
        for _ in 0..4 {
            w.increment_load();
        }
        thread::sleep(Duration::from_millis(5));
        sweep_with(&w, DEFAULT_STALL, Duration::from_millis(1), DEFAULT_STALE);
        assert!(w.stall_reason().is_none(), "no signal, no pile, no veto");
        on_load_report_with(&w, 8, None, false, None, Duration::from_millis(1));
        assert!(
            w.stall_reason().is_none(),
            "a waiting queue without tracked requests is not a wedge either"
        );
        assert_eq!(w.tracked_load(), 0);
    }

    #[test]
    fn a_tracked_pile_without_progress_is_a_wedge_until_a_response() {
        let w = worker();
        tracked_pile(&w, 4);
        thread::sleep(Duration::from_millis(5));
        sweep_with(&w, DEFAULT_STALL, Duration::from_millis(1), DEFAULT_STALE);
        assert_eq!(
            w.stall_reason(),
            Some(StallReason::Wedged),
            "four streams and no response past the bound"
        );
        on_token_progress(&w);
        assert!(w.stall_reason().is_none(), "a response clears it");
        for _ in 0..4 {
            w.note_tracked_ended();
            w.decrement_load();
        }
        assert_eq!(w.tracked_load(), 0);
    }

    #[test]
    fn a_completion_inside_the_bound_keeps_a_waiting_pile_routable() {
        // Review finding on the pushed head: a saturated engine steadily
        // finishing non-streaming requests while four streaming ones wait
        // for their first token. The completions are progress (the tracked
        // stream records its one answer like a token), so the pile is not a
        // wedge; without them the veto still lands.
        let w = worker();
        tracked_pile(&w, 4);
        thread::sleep(Duration::from_millis(5));
        on_token_progress(&w);
        sweep_with(&w, DEFAULT_STALL, Duration::from_millis(50), DEFAULT_STALE);
        assert!(
            w.stall_reason().is_none(),
            "finishing work is progress, whatever the request's shape"
        );
        thread::sleep(Duration::from_millis(55));
        sweep_with(&w, DEFAULT_STALL, Duration::from_millis(50), DEFAULT_STALE);
        assert_eq!(
            w.stall_reason(),
            Some(StallReason::Wedged),
            "and without a completion inside the bound the pile is a wedge"
        );
    }

    #[test]
    fn the_wedged_clock_starts_with_the_first_tracked_request() {
        // An untracked request has been in flight for a while when the first
        // stream is dispatched: the stream's run starts now.
        //
        // The clock's first millisecond is `busy_since`'s idle marker (a zero
        // stamp reads as "never busy"), so a worker built while the clock
        // still reads zero starts its run a millisecond late and the first
        // assertion fails whenever this test is the first to read the clock:
        // let it tick first.
        let _ = now_ms();
        thread::sleep(Duration::from_millis(2));
        let w = worker();
        w.increment_load();
        thread::sleep(Duration::from_millis(15));
        assert!(w.token_progress_age() >= Duration::from_millis(15));
        w.increment_load();
        w.note_tracked_started();
        assert!(
            w.token_progress_age() < Duration::from_millis(10),
            "the clock restarted with the tracked run"
        );
    }

    #[test]
    fn a_zero_wedge_threshold_turns_the_rule_off() {
        let w = worker();
        tracked_pile(&w, 4);
        thread::sleep(Duration::from_millis(5));
        sweep_with(&w, DEFAULT_STALL, Duration::ZERO, DEFAULT_STALE);
        assert!(w.stall_reason().is_none(), "--worker-wedge-secs 0");
        on_load_report_with(&w, 8, None, false, None, Duration::ZERO);
        assert!(w.stall_reason().is_none());
        assert!(
            wedged_by_pile(4, 4, Duration::from_millis(5), Duration::ZERO),
            "the bare predicate would fire at once: the guard is in the callers"
        );
    }

    #[test]
    fn a_transport_failure_turns_a_silent_wedged_worker_unreachable_and_contact_clears_it() {
        // The 45 s partition on the 30 s keepalive: wedged at 3 s, the
        // connection torn down at 40 s with the streams on it, the link back
        // at 48 s, and before this the worker never returned.
        let w = worker();
        tracked_pile(&w, 4);
        set(&w, Some(StallReason::Wedged), "test");
        on_contact_failed_with(&w, "kv stream", Duration::from_secs(600));
        assert_eq!(
            w.stall_reason(),
            Some(StallReason::Wedged),
            "heard from within the threshold: remembered, the wedge stands"
        );
        assert!(w.transport_failure_pending());
        thread::sleep(Duration::from_millis(5));
        on_contact_failed_with(&w, "kv stream", Duration::from_millis(1));
        assert_eq!(
            w.stall_reason(),
            Some(StallReason::Unreachable),
            "silent past the threshold: the pile's connection is gone"
        );
        on_contact(&w);
        assert!(
            w.stall_reason().is_none(),
            "the first contact is the return"
        );
        assert!(w.circuit_breaker_can_execute(), "with the breaker closed");
    }

    #[test]
    fn the_sweep_turns_a_wedged_worker_unreachable_once_silent_after_a_failure() {
        let w = worker();
        tracked_pile(&w, 4);
        set(&w, Some(StallReason::Wedged), "test");
        on_contact_failed_with(&w, "load poll", Duration::from_secs(600));
        sweep_with(&w, Duration::from_secs(600), DEFAULT_WEDGE, DEFAULT_STALE);
        assert_eq!(
            w.stall_reason(),
            Some(StallReason::Wedged),
            "not silent for the threshold yet"
        );
        thread::sleep(Duration::from_millis(5));
        sweep_with(&w, Duration::from_millis(1), DEFAULT_WEDGE, DEFAULT_STALE);
        assert_eq!(w.stall_reason(), Some(StallReason::Unreachable));
        on_contact(&w);
        assert!(w.stall_reason().is_none());
    }

    #[test]
    fn the_sweep_clears_a_wedged_veto_once_the_pile_is_gone_and_a_new_pile_re_arms_it() {
        // Load returns to zero without any contact or progress: the streams
        // ended in error, or the clients gave up.
        let w = worker();
        tracked_pile(&w, 4);
        set(&w, Some(StallReason::Wedged), "test");
        sweep(&w);
        assert_eq!(w.stall_reason(), Some(StallReason::Wedged), "still piled");
        for _ in 0..4 {
            w.note_tracked_ended();
            w.decrement_load();
        }
        assert!(!w.transport_failure_pending());
        sweep(&w);
        assert!(
            w.stall_reason().is_none(),
            "nothing in flight: the evidence is gone"
        );
        // The engine is still stuck: the next requests pile up and re-arm
        // the veto within the bound.
        tracked_pile(&w, 4);
        thread::sleep(Duration::from_millis(5));
        sweep_with(&w, DEFAULT_STALL, Duration::from_millis(1), DEFAULT_STALE);
        assert_eq!(w.stall_reason(), Some(StallReason::Wedged), "re-armed");
    }

    #[test]
    fn a_wedged_veto_survives_polls_and_ends_with_progress() {
        let w = worker();
        set(&w, Some(StallReason::Wedged), "test");
        on_contact(&w);
        assert_eq!(
            w.stall_reason(),
            Some(StallReason::Wedged),
            "answering a poll is not progress"
        );
        on_token_progress(&w);
        assert!(w.stall_reason().is_none());
    }

    #[test]
    fn warm_up_ends_with_time_or_blocks_and_slices_by_share() {
        let warmup = DEFAULT_WARMUP;
        let s = Duration::from_secs;
        // A young fleet: every index empty, nobody thin by comparison, the
        // age rule decides.
        assert!(warmup.applies(s(10), None, 0, 0), "never indexed");
        assert!(warmup.applies(s(10), Some(100), 100, 100));
        assert!(!warmup.applies(s(61), Some(100), 100, 100), "too old");
        assert!(
            !warmup.applies(s(10), Some(2048), 2048, 2048),
            "warm already"
        );
        assert_eq!(warmup.period(), 4);
        let off = Warmup {
            share: 0.0,
            ..warmup
        };
        assert!(!off.applies(Duration::ZERO, None, 0, 0));
        assert_eq!(off.period(), u64::MAX);
    }

    #[test]
    fn a_worker_emptied_by_a_resync_is_thin_until_it_regrows() {
        // The soaks' case: hours after admission a publisher restart cleared
        // the worker's index (32,739 -> 180 blocks) while the fleet held
        // ~30,000 per worker; every prompt had an overlap elsewhere, so no
        // miss ever reached it and it idled for good.
        let warmup = DEFAULT_WARMUP;
        let old = Duration::from_secs(3_600);
        assert!(warmup.is_thin(180, 30_000));
        assert!(
            warmup.applies(old, Some(180), 180, 30_000),
            "emptied: thin, 180 blocks regrown"
        );
        assert!(warmup.applies(old, Some(0), 0, 30_000), "empty outright");
        // Churn c3: the index regrew to 1,765 of a 32,767 level within a
        // minute of the clear (the decode blocks of the requests in flight)
        // and the cap meant for a new worker's first cache ended the warm-up
        // there, thin or not; one diversion, then idle for 25 minutes.
        assert!(
            warmup.applies(old, Some(1_765), 1_765, 32_767),
            "regrown past the warm-up blocks but still thin: served on"
        );
        assert!(
            warmup.applies(old, Some(16_383), 16_383, 32_767),
            "thin until the ratio"
        );
        assert!(
            !warmup.applies(old, Some(16_384), 16_384, 32_767),
            "at the ratio it is back to affinity"
        );
        assert!(
            !warmup.applies(old, Some(0), 20_000, 30_000),
            "two thirds of the fleet's level is not thin at a half"
        );
        assert!(
            warmup.applies(Duration::from_secs(10), Some(2_000), 2_000, 30_000),
            "a young worker in an old fleet is served past its first blocks while thin"
        );
        assert!(
            !warmup.applies(Duration::from_secs(10), Some(2_000), 2_000, 2_000),
            "the cap bounds the age rule: warm at the fleet's level"
        );
        assert!(
            !warmup.applies(old, None, 0, 0),
            "an empty fleet has no level: the age rule alone, and this worker is old"
        );
        assert!(
            !warmup.applies(old, Some(0), 0, 100),
            "a fleet holding less than the warm-up blocks is not worth catching up to"
        );
        assert!(
            warmup.applies(old, Some(0), 0, 1_024),
            "at the warm-up blocks it is"
        );
        let age_only = Warmup {
            thin_ratio: 0.0,
            ..warmup
        };
        assert!(
            !age_only.applies(old, Some(0), 0, 30_000),
            "thin_ratio 0 keeps the age rule alone"
        );
    }

    #[test]
    fn re_admission_restarts_the_warm_up_clock() {
        let w = worker();
        thread::sleep(Duration::from_millis(20));
        let before = w.admitted_age();
        set(&w, Some(StallReason::Unreachable), "test");
        on_contact(&w);
        assert!(
            w.admitted_age() < before,
            "cleared veto counts as an admission"
        );
    }

    fn signalling_worker() -> (Arc<dyn Worker>, mpsc::UnboundedReceiver<WorkerConnected>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let w: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .connect_signal_tx(tx)
                .build(),
        );
        (w, rx)
    }

    #[test]
    fn contact_asks_for_promotion_only_of_a_worker_that_was_unreachable() {
        let (w, mut rx) = signalling_worker();
        on_contact(&w);
        assert!(rx.try_recv().is_err(), "a Ready worker needs no promotion");
        // Demoted by its probes while its transport was gone: the first
        // contact is its return, promoted now.
        set(&w, Some(StallReason::Unreachable), "test");
        w.set_status(WorkerStatus::NotReady);
        on_contact(&w);
        assert!(w.stall_reason().is_none());
        let signal = rx
            .try_recv()
            .expect("a demoted worker back from a transport outage is signalled");
        assert_eq!(signal.url, "http://w1:8000");
        // Demoted by its probes alone (a draining engine answering 503 on
        // /health while it still relays KV batches and answers GetLoads):
        // every batch is a contact and none of them is health.
        w.set_status(WorkerStatus::NotReady);
        on_contact(&w);
        on_contact(&w);
        assert!(
            rx.try_recv().is_err(),
            "a reachable transport does not lift a health demotion"
        );
        w.set_status(WorkerStatus::Failed);
        on_contact(&w);
        assert!(rx.try_recv().is_err(), "nor a Failed one");
    }

    #[test]
    fn a_passing_probe_is_contact_but_never_a_promotion() {
        let (w, mut rx) = signalling_worker();
        set(&w, Some(StallReason::Unreachable), "test");
        w.set_status(WorkerStatus::NotReady);
        thread::sleep(Duration::from_millis(15));
        let before = w.contact_age();
        on_probe_passed(&w);
        assert!(w.contact_age() < before, "a passing probe is a contact");
        assert!(w.stall_reason().is_none(), "and ends an unreachable veto");
        assert!(
            rx.try_recv().is_err(),
            "the health state machine promotes at its success threshold, not the probe"
        );
    }

    #[test]
    fn a_failure_right_after_contact_is_a_blip() {
        let w = worker();
        w.note_contact();
        on_contact_failed(&w, "test");
        assert!(w.stall_reason().is_none());
        assert!(
            w.transport_failure_pending(),
            "but it is remembered for the sweep"
        );
        sweep(&w);
        assert!(
            w.stall_reason().is_none(),
            "the sweep waits for the threshold"
        );
        w.note_contact();
        assert!(!w.transport_failure_pending(), "a contact forgets it");
    }

    #[test]
    fn an_engine_that_reports_its_own_silence_is_a_wedge_without_a_pile() {
        // The freeze drills: one request in flight on a frozen engine whose
        // servicer keeps heartbeating. No pile grows, the record shows no
        // queue, and before this rule nothing fired until the engine thawed.
        let w = worker();
        tracked_pile(&w, 1);
        thread::sleep(Duration::from_millis(5));
        let wedge = Duration::from_millis(1);
        on_load_report_with(&w, 0, None, true, None, wedge);
        assert!(
            w.stall_reason().is_none(),
            "one quiet request alone is not a wedge"
        );
        on_load_report_with(&w, 0, Some(Duration::from_millis(1)), true, None, wedge);
        assert_eq!(w.stall_reason(), Some(StallReason::Wedged));
        assert!(w.engine_stalled());
        // This gateway's progress clock restarted with a dispatch after the
        // veto: not progress while the engine still reports its silence.
        on_load_report_with(&w, 0, Some(Duration::from_millis(2)), true, None, wedge);
        assert_eq!(w.stall_reason(), Some(StallReason::Wedged), "no flap");
        on_token_progress(&w);
        assert!(w.stall_reason().is_none(), "progress clears it");
    }

    /// A worker that has shown no prefill rate yet (its engine's first
    /// request after the servicer connected) is judged at the bare threshold
    /// when its wire reports every step, and gets no verdict on a token-batch
    /// wire, whose allowance needs a rate.
    #[test]
    fn a_cold_worker_on_a_step_reporting_wire_is_vetoed_at_the_bare_threshold() {
        let w = worker();
        tracked_pile(&w, 1);
        assert_eq!(w.prefill_rate_tps(), 0, "no rate shown yet");
        on_load_report_with(
            &w,
            0,
            Some(Duration::from_secs(3)),
            true,
            Some(2_048),
            DEFAULT_WEDGE,
        );
        assert_eq!(w.stall_reason(), Some(StallReason::Wedged));
        let batch_only = worker();
        tracked_pile(&batch_only, 1);
        on_load_report_with(
            &batch_only,
            0,
            Some(Duration::from_secs(3)),
            false,
            Some(2_048),
            DEFAULT_WEDGE,
        );
        assert!(
            batch_only.stall_reason().is_none(),
            "a token-batch wire without a rate gets no verdict"
        );
    }

    #[test]
    fn a_step_reporting_engines_silence_is_judged_against_the_bare_threshold() {
        // A batch in prefill stretches this gateway's bound, not the
        // engine's: a wire that reports every scheduler step (vLLM's
        // stats-only step batches) is silent only when its scheduler is.
        let w = worker();
        tracked_pile(&w, 1);
        w.note_prefill_started(128 * 1_152, true);
        assert!(wedge_bound(&w, DEFAULT_WEDGE) > Duration::from_secs(10));
        on_load_report_with(
            &w,
            0,
            Some(Duration::from_millis(2_900)),
            true,
            None,
            DEFAULT_WEDGE,
        );
        assert!(w.stall_reason().is_none(), "under the threshold");
        on_load_report_with(
            &w,
            0,
            Some(Duration::from_secs(3)),
            true,
            None,
            DEFAULT_WEDGE,
        );
        assert_eq!(w.stall_reason(), Some(StallReason::Wedged));
        // A zero threshold turns the rule off with the rest of the wedge rule.
        let off = worker();
        tracked_pile(&off, 1);
        on_load_report_with(
            &off,
            0,
            Some(Duration::from_secs(600)),
            true,
            None,
            Duration::ZERO,
        );
        assert!(off.stall_reason().is_none());
        assert!(!engine_stalled(
            Some(Duration::from_secs(600)),
            true,
            Duration::ZERO,
            Some(Duration::ZERO)
        ));
        assert!(!engine_stalled(
            None,
            true,
            DEFAULT_WEDGE,
            Some(DEFAULT_WEDGE)
        ));
    }

    #[test]
    fn a_token_batch_wires_silence_is_judged_against_the_servicers_pending_prefill() {
        // The review's case: a lightly loaded single worker on a wire that
        // sends nothing for a request still in chunked prefill (SGLang,
        // TokenSpeed) takes one long prompt. Its servicer reports a growing
        // silence from the dispatch on, and beside it the prompt tokens it
        // holds with no output yet; the silence is judged against that
        // prefill at the rate this gateway has measured on the worker (twice
        // over, the rate being an estimate), not the bare threshold, and not
        // this gateway's own backlog: a second gateway with nothing in flight
        // on the worker sees the same record and reaches the same verdict.
        // The rate is the one the gateway learns from the first tokens of
        // its streaming requests over the time their prompts were pending
        // (a non-streaming generation's one answer times no prefill), so the
        // allowance follows the engine's measured prefill speed, not the
        // idle gaps between requests.
        let with_backlog = BasicWorkerBuilder::new("http://w1:8000").build();
        let idle = BasicWorkerBuilder::new("http://w1:8000").build();
        for basic in [&with_backlog, &idle] {
            // Both gateways have measured the worker: two 1,152-token
            // streaming prompts back to back, 750 ms to the first token
            // each, 2,304 tokens in 1.5 s of busy time, 1,536 tokens/s.
            let runtime = basic.runtime.load_full();
            runtime.note_prefill_started_at(1_152, true, 1_000);
            runtime.note_prefill_ended_at(1_152, true, Some(0), 1_750);
            runtime.note_prefill_started_at(1_152, true, 1_750);
            runtime.note_prefill_ended_at(1_152, true, Some(0), 2_500);
            assert_eq!(runtime.prefill_rate_tps(), 1_536);
        }
        let with_backlog: Arc<dyn Worker> = Arc::new(with_backlog);
        let idle: Arc<dyn Worker> = Arc::new(idle);
        tracked_pile(&with_backlog, 1);
        with_backlog.note_prefill_started(16 * 1_152, true);
        let pending = Some(16 * 1_152);
        // 18,432 tokens at 1,536 tokens/s: 12 s of prefill, 24 s allowed.
        assert_eq!(
            silence_allowance(&idle, DEFAULT_WEDGE, pending),
            Some(Duration::from_secs(24))
        );
        for w in [&with_backlog, &idle] {
            on_load_report_with(
                w,
                0,
                Some(Duration::from_secs(3)),
                false,
                pending,
                DEFAULT_WEDGE,
            );
            assert!(w.stall_reason().is_none(), "3 s into a 12 s prefill");
            on_load_report_with(
                w,
                0,
                Some(Duration::from_secs(20)),
                false,
                pending,
                DEFAULT_WEDGE,
            );
            assert!(w.stall_reason().is_none(), "a slow prefill is not a stall");
            on_load_report_with(
                w,
                0,
                Some(Duration::from_secs(24)),
                false,
                pending,
                DEFAULT_WEDGE,
            );
            assert_eq!(
                w.stall_reason(),
                Some(StallReason::Wedged),
                "silent past twice what the prefill could need"
            );
        }
        // The first token ends the prefill (the servicer reports nothing
        // pending) and the veto; with nothing pending the allowance is the
        // threshold again, so a frozen engine on such a wire is vetoed at the
        // threshold like any other, on both gateways.
        with_backlog.note_prefill_ended(16 * 1_152, true, Some(0));
        for w in [&with_backlog, &idle] {
            on_token_progress(w);
            assert!(w.stall_reason().is_none());
            on_load_report_with(
                w,
                0,
                Some(Duration::from_millis(2_900)),
                false,
                Some(0),
                DEFAULT_WEDGE,
            );
            assert!(w.stall_reason().is_none());
            on_load_report_with(
                w,
                0,
                Some(Duration::from_secs(3)),
                false,
                Some(0),
                DEFAULT_WEDGE,
            );
            assert_eq!(w.stall_reason(), Some(StallReason::Wedged));
        }
        // No verdict without a measured rate (the cold prior is a guess) or
        // without the servicer's pending figure (an older servicer), and a
        // record that does not say which wire it is counts as token batches.
        let cold = worker();
        assert_eq!(silence_allowance(&cold, DEFAULT_WEDGE, pending), None);
        on_load_report_with(
            &cold,
            0,
            Some(Duration::from_secs(60)),
            false,
            pending,
            DEFAULT_WEDGE,
        );
        assert!(
            cold.stall_reason().is_none(),
            "no rate shown yet: no verdict"
        );
        // A gateway whose own requests to the worker never stream is in the
        // same place: a non-streaming generation's one answer comes after
        // its decode and times no prefill, so no rate is learned from it
        // and the silence on a token-batch wire gets no verdict there (its
        // prompt still counts in the servicer's pending figure, which the
        // gateways with a rate judge against).
        let unstreamed = BasicWorkerBuilder::new("http://w1:8000").build();
        {
            let runtime = unstreamed.runtime.load_full();
            runtime.note_prefill_started_at(4_000, false, 1_000);
            runtime.note_prefill_ended_at(4_000, false, Some(0), 31_000);
            assert_eq!(runtime.prefill_rate_tps(), 0);
        }
        let unstreamed: Arc<dyn Worker> = Arc::new(unstreamed);
        assert_eq!(silence_allowance(&unstreamed, DEFAULT_WEDGE, pending), None);
        assert_eq!(silence_allowance(&idle, DEFAULT_WEDGE, None), None);
        assert!(!engine_stalled(
            Some(Duration::from_secs(60)),
            false,
            DEFAULT_WEDGE,
            None
        ));
        // The allowance never exceeds the wedge bound's cap.
        assert_eq!(
            silence_allowance(&idle, DEFAULT_WEDGE, Some(100_000_000)),
            Some(WEDGE_BOUND_CAP)
        );
    }

    #[test]
    fn an_engine_reported_wedge_outlives_an_empty_pile_and_ends_with_the_engines_output() {
        // Nothing of this gateway's in flight (the frozen requests are another
        // gateway's): the sweep's "drained" rule must not lift the veto while
        // the record still says the engine is stuck; the record that shows
        // the engine producing again does.
        let w = worker();
        thread::sleep(Duration::from_millis(5));
        let wedge = Duration::from_millis(1);
        on_load_report_with(&w, 0, Some(Duration::from_secs(4)), true, None, wedge);
        assert_eq!(w.stall_reason(), Some(StallReason::Wedged));
        sweep_with(&w, DEFAULT_STALL, wedge, DEFAULT_STALE);
        assert_eq!(
            w.stall_reason(),
            Some(StallReason::Wedged),
            "not drained while the engine says it is stuck"
        );
        on_load_report_with(&w, 0, Some(Duration::ZERO), true, None, wedge);
        assert!(w.stall_reason().is_none(), "the engine produces again");
        assert!(!w.engine_stalled());
        // A pile of this gateway's own that is still silent keeps the veto
        // when the engine reports output (its progress is what clears it).
        let busy = worker();
        tracked_pile(&busy, 2);
        thread::sleep(Duration::from_millis(5));
        on_load_report_with(&busy, 0, Some(Duration::from_secs(4)), true, None, wedge);
        assert_eq!(busy.stall_reason(), Some(StallReason::Wedged));
        on_load_report_with(&busy, 0, None, true, None, wedge);
        assert_eq!(
            busy.stall_reason(),
            Some(StallReason::Wedged),
            "this gateway's pile is still silent"
        );
        on_token_progress(&busy);
        assert!(busy.stall_reason().is_none());
        // The flag goes with the record channel: an unreachable veto forgets
        // it, and the next record speaks for the engine again.
        on_load_report_with(&w, 0, Some(Duration::from_secs(4)), true, None, wedge);
        assert!(w.engine_stalled());
        set(&w, Some(StallReason::Unreachable), "test");
        assert!(!w.engine_stalled());
    }
}
