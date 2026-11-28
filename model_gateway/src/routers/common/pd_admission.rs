//! Admission control for disaggregated (PD) dispatch, bounded by the decode
//! engine's running window.
//!
//! SGLang-lineage engines start a bootstrap deadline on the prefill side the
//! moment a request lands, and it clears only once the decode scheduler has
//! *admitted* that request and answered with its KV manifest. Decode admission
//! is bounded by the engine's running window (`--max-num-seqs` /
//! `--max-running-requests`), so a burst wider than that window leaves the
//! prefill deadline racing a queue the gateway itself created: prefill times
//! out rooms the decode has not reached yet, and the decode then pre-allocates
//! those same rooms and waits out its own transfer deadline for a peer that is
//! already gone.
//!
//! The gate below is the gateway's half of the fix — never post more rooms to
//! a pair than its decode can take. A request that arrives with the window
//! full waits for a room rather than joining the engine's queue, and sheds if
//! none frees in time, with the same 503 selection already answers when every
//! worker is vetoed.
//!
//! Admission is a *claim*, not a look: it reserves its rooms on the worker
//! atomically ([`Worker::try_admit_pd`]) and hands back a guard that releases
//! them. Reading the in-flight count and then sending would let two dispatches
//! both take the last free room, which is the burst this exists to stop.
//!
//! The wait is first come, first served. A dispatch takes a place in the
//! decode worker's queue the moment it arrives ([`Worker::pd_enqueue`]) and
//! claims only while no older place needs as few rooms as it does, so a room
//! that frees goes to the request that has waited longest for one — not to
//! whichever attempt happens to run first, which under a steady arrival rate
//! is a newcomer's first try, made before any sleeping waiter wakes; a waiter
//! that lost that race once kept losing it until the budget shed it. A
//! request that needs more rooms than the ones behind it (a fan-out, a
//! batch) does not hold them back while its rooms free one at a time: they
//! pass it, and it claims when its rooms are free together, as before. The
//! place is given back on every way out of the wait: a claim, the shed, a
//! dispatch dropped mid-wait. The worker wakes the waits that may claim when
//! rooms are released or a place ahead of them leaves, so the right one
//! claims the moment it can; the queue and its notifiers live on the worker
//! with its rooms, not in a registry of their own, and a request that finds
//! nobody waiting still claims at once.
//!
//! Nothing here runs when the engine does not report a window: admission is
//! not the gateway's to decide then, and dispatch behaves exactly as it did
//! before this module existed. That is said once, loudly, when such a decode
//! worker registers (`announce_decode_window`), and the window itself is
//! published per worker, so a gate that is off is a value an operator can
//! read rather than an absence of series.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::response::Response;
use smg_worker::identity::basic::{Worker, WorkerType};
use tokio::{sync::Notify, time::Instant};
use tracing::{debug, warn};

use crate::{observability::metrics::Metrics, routers::common::overload};

/// Default seconds a PD dispatch may wait for decode rooms. Well under the
/// engines' bootstrap deadline (120 s on one engine), so a request that does
/// wait still dispatches with the whole deadline ahead of it.
pub const DEFAULT_PD_ADMISSION_WAIT_SECS: u64 = 30;

/// How often the wait retries its claim when nothing wakes it.
///
/// The worker wakes the waits that may claim whenever rooms are released or
/// a place ahead of them leaves ([`Worker::pd_wake`]), so the right waiter
/// normally claims the moment it can and a freed room does not idle for a
/// retry interval under a full window. The retry stays as the backstop — a
/// worker without notifiers, and the deadline check: 50 ms is far finer than
/// both the engine step that actually frees a room and the wait deadline
/// below, and the sleep is asynchronous — a waiting request occupies no
/// thread. The notifiers live on the worker with its rooms; there is no
/// registry of them outside it, with the eviction problem that would bring.
const CLAIM_RETRY_INTERVAL: Duration = Duration::from_millis(50);

/// Seconds a PD dispatch waits for decode rooms before shedding.
///
/// Process-wide for the same reason as the overload shed's `Retry-After`: the
/// gate is a free function on a dispatch path every router reaches, and the
/// value is one operator knob rather than a per-request input. Latched once at
/// startup from `--pd-admission-wait-secs`.
static PD_ADMISSION_WAIT_SECS: AtomicU64 = AtomicU64::new(DEFAULT_PD_ADMISSION_WAIT_SECS);

/// Latch the admission wait. Called once at startup from the router config.
pub fn set_pd_admission_wait_secs(secs: u64) {
    PD_ADMISSION_WAIT_SECS.store(secs, Ordering::Relaxed);
}

fn admission_wait() -> Duration {
    Duration::from_secs(PD_ADMISSION_WAIT_SECS.load(Ordering::Relaxed))
}

/// A claim on `rooms` slots in the decode engine's running window.
///
/// Rides in the dispatch's `LoadGuards`, so the rooms are released on every
/// path the dispatch can end on: an error before the sides go out, a failed
/// side, a client disconnect mid-stream, a retry that re-selects, or normal
/// completion.
pub(crate) struct PdAdmissionGuard {
    worker: Arc<dyn Worker>,
    rooms: usize,
}

impl Drop for PdAdmissionGuard {
    fn drop(&mut self) {
        self.worker.release_pd(self.rooms);
    }
}

impl std::fmt::Debug for PdAdmissionGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdAdmissionGuard")
            .field("worker", &self.worker.url())
            .field("rooms", &self.rooms)
            .finish()
    }
}

/// A place in the decode worker's admission queue ([`Worker::pd_enqueue`]),
/// taken when the dispatch arrives and given back however its wait ends: a
/// claim, the shed at the deadline, or the dispatch dropped mid-wait (its
/// client gone). A place may claim only while no older one needs as few
/// rooms, which is what makes the wait first come, first served; giving the
/// place back promptly is what lets the next one through.
struct Place<'a> {
    worker: &'a dyn Worker,
    place: u64,
}

impl<'a> Place<'a> {
    fn take(worker: &'a dyn Worker, rooms: usize) -> Self {
        Self {
            worker,
            place: worker.pd_enqueue(rooms),
        }
    }

    /// Whether no older place still waiting needs as few rooms as this one:
    /// the gate lets it claim only then.
    fn may_claim(&self) -> bool {
        self.worker.pd_may_claim(self.place)
    }

    /// The notifier the worker fires when this place is worth a new try.
    fn wake(&self) -> Option<Arc<Notify>> {
        self.worker.pd_wake(self.place)
    }
}

impl Drop for Place<'_> {
    fn drop(&mut self) {
        self.worker.pd_dequeue(self.place);
    }
}

/// How a claim was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Claim {
    /// The rooms were free on the first try.
    Immediate,
    /// The rooms freed during the wait.
    Waited,
    /// The wait deadline passed with the window still full.
    Refused,
}

/// Retry `try_claim` until it succeeds or `wait` elapses: again as soon as
/// `wake` fires, and at least every [`CLAIM_RETRY_INTERVAL`].
///
/// The claim itself is what makes admission safe under concurrency, so this
/// never reads a count: every attempt is the same all-or-nothing reservation,
/// and the first one to succeed owns the rooms.
async fn claim_within(
    wait: Duration,
    wake: Option<&Notify>,
    try_claim: impl Fn() -> bool,
) -> Claim {
    let deadline = Instant::now() + wait;
    let mut first = true;
    loop {
        // Registered before the try, so a room released between this try and
        // the wait below still wakes it.
        let woken = wake.map(Notify::notified);
        if try_claim() {
            return if first {
                Claim::Immediate
            } else {
                Claim::Waited
            };
        }
        first = false;
        let now = Instant::now();
        if now >= deadline {
            return Claim::Refused;
        }
        let retry = tokio::time::sleep(CLAIM_RETRY_INTERVAL.min(deadline - now));
        match woken {
            Some(woken) => tokio::select! {
                () = woken => {}
                () = retry => {}
            },
            None => retry.await,
        }
    }
}

/// Gate one disaggregated dispatch on the decode side's admission window,
/// claiming the `rooms` bootstrap rooms it is about to post.
///
/// `Ok(Some(guard))` holds the claim; `Ok(None)` means the engine reports no
/// window and admission is not the gateway's to decide; `Err` is the shed the
/// caller must return instead of dispatching.
pub(crate) async fn admit_decode(
    decode: &Arc<dyn Worker>,
    model_id: &str,
    rooms: usize,
) -> Result<Option<PdAdmissionGuard>, Response> {
    // `None` (unreported, or a nonsense zero) leaves admission to the engine.
    let Some(window) = decode.max_running_requests().map(usize::from) else {
        return Ok(None);
    };
    // A plan always posts at least one room, whatever its sub-request count
    // claims.
    let rooms = rooms.max(1);

    // A batched plan can demand more rooms than the pair will ever run at
    // once. No wait can free more than the window holds, so answer now
    // instead of sleeping out the budget to reach the same shed.
    if rooms > window {
        Metrics::record_pd_admission_shed();
        debug!(
            worker = decode.redacted_url(),
            model_id, window, rooms, "PD admission shed: more rooms than the decode window holds"
        );
        return Err(overload::shed_pd_admission(
            decode.redacted_url(),
            model_id,
            window,
            rooms,
        ));
    }

    let wait = admission_wait();
    // The place is taken before the first try, so a request that arrives
    // while others of its size (or narrower) wait joins behind them even
    // when a room happens to be free, and one that finds nobody it must
    // yield to claims without a wait, as before. The guard gives the place
    // back on every way out of the wait, the dispatch dropped mid-wait
    // included.
    let place = Place::take(decode.as_ref(), rooms);
    let wake = place.wake();
    let claim = claim_within(wait, wake.as_deref(), || {
        place.may_claim() && decode.try_admit_pd(rooms, window)
    })
    .await;
    drop(place);
    match claim {
        Claim::Immediate => Ok(Some(PdAdmissionGuard {
            worker: Arc::clone(decode),
            rooms,
        })),
        Claim::Waited => {
            Metrics::record_pd_admission_wait();
            debug!(
                worker = decode.redacted_url(),
                model_id, window, rooms, "PD admission waited for decode rooms"
            );
            Ok(Some(PdAdmissionGuard {
                worker: Arc::clone(decode),
                rooms,
            }))
        }
        Claim::Refused => {
            Metrics::record_pd_admission_shed();
            debug!(
                worker = decode.redacted_url(),
                model_id,
                window,
                rooms,
                admitted = decode.pd_admitted(),
                waited_secs = wait.as_secs(),
                "PD admission shed: no decode rooms freed"
            );
            Err(overload::shed_pd_admission(
                decode.redacted_url(),
                model_id,
                window,
                rooms,
            ))
        }
    }
}

/// What one try at a decode's gate came to ([`try_admit_decode`]).
pub(crate) enum Tried {
    /// The engine reports no window: the gate abstains, as [`admit_decode`]
    /// does, and the dispatch holds no claim there.
    NoWindow,
    /// The rooms are claimed.
    Claimed(PdAdmissionGuard),
    /// The window is full, or an older place is in line for the rooms.
    /// Nothing is held.
    Full,
}

/// One try at `decode`'s gate for `rooms`, with no wait: for a dispatch
/// that holds its claim at another decode's gate and would move to this
/// one. The try runs under the gate's own order. It takes a place, claims
/// while no older place needs as few rooms and the rooms are free, and gives
/// the place back whatever it came to. So a move passes no request that
/// waits at `decode`, and a try that fails holds nothing.
pub(crate) fn try_admit_decode(decode: &Arc<dyn Worker>, rooms: usize) -> Tried {
    let Some(window) = decode.max_running_requests().map(usize::from) else {
        return Tried::NoWindow;
    };
    let rooms = rooms.max(1);
    let place = Place::take(decode.as_ref(), rooms);
    let claimed = place.may_claim() && decode.try_admit_pd(rooms, window);
    drop(place);
    if claimed {
        Tried::Claimed(PdAdmissionGuard {
            worker: Arc::clone(decode),
            rooms,
        })
    } else {
        Tried::Full
    }
}

/// Whether `decode`'s gate would seat one more room now: its running window
/// has a room that none of the places already waiting at the gate is in line
/// for. A worker that reports no window always would, as the gate abstains
/// there. Two atomic reads, so a selection can ask it of every partner of a
/// pick.
///
/// A wider place ahead (a batched plan) counts every room it needs, though
/// the gate itself lets a narrower newcomer pass it: the answer is "no" more
/// often than the gate waits, never "yes" where it would.
pub(crate) fn has_free_room(decode: &dyn Worker) -> bool {
    decode
        .max_running_requests()
        .is_none_or(|window| decode.pd_admitted() + decode.pd_waiting_rooms() < usize::from(window))
}

/// Announce a decode worker's admission bound as it registers.
///
/// `admit_decode` abstains for a worker that reports no window, which used to
/// be invisible: the admission families only appear once the gate acts, so an
/// operator who turned on PD in front of such engines saw no series and
/// believed the bound existed (the vLLM servicer reported no window at all
/// until it advertised `max_num_seqs`). Say it once where the worker joins,
/// and publish the window (`smg_pd_admission_window`, 0 for none) so "off" is
/// a number on the scrape. Workers of any other type are not the gate's.
pub(crate) fn announce_decode_window(worker: &dyn Worker) {
    if !matches!(worker.worker_type(), WorkerType::Decode) {
        return;
    }
    let window = worker.max_running_requests();
    Metrics::set_pd_admission_window(worker.url(), window);
    match window {
        Some(window) => debug!(
            worker = worker.redacted_url(),
            window, "PD admission bounds dispatch to this decode worker by its running window"
        ),
        None => warn!(
            worker = worker.redacted_url(),
            "PD admission is off for this decode worker: it reports no running window (no \
             `max_num_seqs` or `max_running_requests` label), so a burst wider than its \
             scheduler runs at once queues at the engine instead of waiting at the gateway"
        ),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/routers/common/pd_admission/tests.rs"]
mod tests;
