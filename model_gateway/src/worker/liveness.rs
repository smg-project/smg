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
//! Two vetoes, both read by routing through [`Worker::stall_reason`]:
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
//!   answering health looks exactly like this. The requests counted are the
//!   tracked ones, streaming generations to the worker over gRPC, whose
//!   responses the gateway sees one by one ([`Worker::tracked_load`]): an
//!   HTTP worker, a PD leg or a non-streaming generation gives no signal
//!   between dispatch and completion, so it never forms a pile and is never
//!   judged by one; a non-streaming generation's one answer is progress all
//!   the same, and its prompt counts in the prefill backlog. The clock
//!   starts at the first dispatch of a run of
//!   tracked requests, never at registration, and the bound is the
//!   configured threshold or, if longer, the time the engine may still need
//!   to prefill what is in flight ([`Worker::prefill_backlog`]): a batch that
//!   is all in prefill streams nothing and is not wedged. A zero threshold
//!   turns the rule off.
//!
//! Neither veto touches the worker's health status: the health checker keeps
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
    wedge
        .max(worker.prefill_backlog())
        .min(wedge.max(WEDGE_BOUND_CAP))
}

static THRESHOLDS: OnceLock<(Duration, Duration)> = OnceLock::new();
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

/// Set the stall and wedge thresholds from the gateway configuration. The
/// first call wins; the defaults are two and three seconds.
pub(crate) fn configure(stall: Duration, wedge: Duration) {
    let _ = THRESHOLDS.set((stall, wedge));
}

fn thresholds() -> (Duration, Duration) {
    THRESHOLDS
        .get()
        .copied()
        .unwrap_or((DEFAULT_STALL, DEFAULT_WEDGE))
}

/// Milliseconds since the gateway started: the clock behind the workers'
/// contact and progress stamps.
pub(crate) fn now_ms() -> u64 {
    u64::try_from(EPOCH.get_or_init(Instant::now).elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Whether a gRPC status describes the connection rather than the request:
/// what a dead peer, a reset or a failed keepalive produce. A deadline or an
/// engine-side error says the worker is slow or wrong, not gone, and a slow
/// worker that still streams tokens must not flap in and out of routing.
pub(crate) fn is_transport_failure(code: tonic::Code) -> bool {
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
    let (stall, wedge) = thresholds();
    sweep_with(worker, stall, wedge);
}

/// [`sweep`] at the given thresholds.
fn sweep_with(worker: &Arc<dyn Worker>, stall: Duration, wedge: Duration) {
    let load = worker.tracked_load();
    let previous_load = worker.swap_load_sample(load);
    let silent_after_failure =
        worker.transport_failure_pending() && stalled(worker.contact_age(), stall);
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
            } else if load == 0 {
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
/// the wedge threshold.
pub(crate) fn on_load_report(worker: &Arc<dyn Worker>, waiting: i64) {
    let (_, wedge) = thresholds();
    on_load_report_with(worker, waiting, wedge);
}

/// [`on_load_report`] at the given wedge threshold.
fn on_load_report_with(worker: &Arc<dyn Worker>, waiting: i64, wedge: Duration) {
    let previous = worker.swap_waiting_reqs(waiting);
    match worker.stall_reason() {
        Some(StallReason::Wedged) => {
            if worker.token_progress_age() < wedge {
                set(worker, None, "progress");
            }
        }
        Some(StallReason::Unreachable) => {}
        None => {
            if !wedge.is_zero()
                && wedged_by_queue(
                    worker.tracked_load(),
                    waiting,
                    previous,
                    worker.token_progress_age(),
                    wedge_bound(worker, wedge),
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

/// The unreachable rule on its inputs.
fn stalled(contact_age: Duration, stall: Duration) -> bool {
    contact_age >= stall
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

    use tokio::sync::mpsc;

    use super::*;
    use crate::worker::{
        circuit_breaker::CircuitBreakerConfig, event::WorkerConnected, BasicWorkerBuilder,
    };

    fn worker() -> Arc<dyn Worker> {
        Arc::new(BasicWorkerBuilder::new("http://w1:8000").build())
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
        on_load_report(&w, 3);
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
        w.note_prefill_started(128 * 1_152);
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
        w.note_prefill_ended(128 * 1_152, true);
        assert_eq!(wedge_bound(&w, DEFAULT_WEDGE), DEFAULT_WEDGE);
        // The bound is capped against leaking books.
        w.note_prefill_started(100_000_000);
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
        w.note_prefill_started(8 * 150);
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
        sweep_with(&w, DEFAULT_STALL, Duration::from_millis(1));
        assert!(w.stall_reason().is_none(), "no signal, no pile, no veto");
        on_load_report_with(&w, 8, Duration::from_millis(1));
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
        sweep_with(&w, DEFAULT_STALL, Duration::from_millis(1));
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
        sweep_with(&w, DEFAULT_STALL, Duration::from_millis(50));
        assert!(
            w.stall_reason().is_none(),
            "finishing work is progress, whatever the request's shape"
        );
        thread::sleep(Duration::from_millis(55));
        sweep_with(&w, DEFAULT_STALL, Duration::from_millis(50));
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
        sweep_with(&w, DEFAULT_STALL, Duration::ZERO);
        assert!(w.stall_reason().is_none(), "--worker-wedge-secs 0");
        on_load_report_with(&w, 8, Duration::ZERO);
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
        sweep_with(&w, Duration::from_secs(600), DEFAULT_WEDGE);
        assert_eq!(
            w.stall_reason(),
            Some(StallReason::Wedged),
            "not silent for the threshold yet"
        );
        thread::sleep(Duration::from_millis(5));
        sweep_with(&w, Duration::from_millis(1), DEFAULT_WEDGE);
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
        sweep_with(&w, DEFAULT_STALL, Duration::from_millis(1));
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
}
