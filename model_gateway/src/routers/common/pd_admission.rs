//! Admission control for disaggregated (PD) dispatch, bounded by the decode
//! engine's running window.
//!
//! SGLang-lineage engines start a bootstrap deadline on the prefill leg the
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
//! worker registers ([`announce_decode_window`]), and the window itself is
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
use tokio::{sync::Notify, time::Instant};
use tracing::{debug, warn};

use crate::{
    observability::metrics::Metrics,
    routers::common::overload,
    worker::{Worker, WorkerType},
};

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
/// path the dispatch can end on: an error before the legs go out, a failed
/// leg, a client disconnect mid-stream, a retry that re-selects, or normal
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

/// Gate one disaggregated dispatch on the decode leg's admission window,
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
            worker = decode.url(),
            model_id, window, rooms, "PD admission shed: more rooms than the decode window holds"
        );
        return Err(overload::shed_pd_admission(
            decode.url(),
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
                worker = decode.url(),
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
                worker = decode.url(),
                model_id,
                window,
                rooms,
                admitted = decode.pd_admitted(),
                waited_secs = wait.as_secs(),
                "PD admission shed: no decode rooms freed"
            );
            Err(overload::shed_pd_admission(
                decode.url(),
                model_id,
                window,
                rooms,
            ))
        }
    }
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
            worker = worker.url(),
            window, "PD admission bounds dispatch to this decode worker by its running window"
        ),
        None => warn!(
            worker = worker.url(),
            "PD admission is off for this decode worker: it reports no running window (no \
             `max_num_seqs` or `max_running_requests` label), so a burst wider than its \
             scheduler runs at once queues at the engine instead of waiting at the gateway"
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
        Mutex,
    };

    use axum::http::{header::RETRY_AFTER, StatusCode};
    use futures::future::join_all;
    use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};

    use super::*;
    use crate::{
        routers::{common::retry::is_retryable_response, error::extract_error_code_from_response},
        worker::{BasicWorkerBuilder, ConnectionMode, WorkerType},
    };

    fn decode_worker(url: &str, window: Option<u16>) -> Arc<dyn Worker> {
        let mut labels = std::collections::HashMap::new();
        if let Some(window) = window {
            labels.insert("max_running_requests".to_string(), window.to_string());
        }
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("m"))
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Grpc)
                .labels(labels)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    /// An engine that reports no window keeps the pre-gate behavior: dispatch,
    /// however deep the gateway's in-flight count already is.
    #[tokio::test]
    async fn unknown_window_never_waits_or_sheds() {
        let decode = decode_worker("grpc://127.0.0.1:9901", None);
        for _ in 0..1_000 {
            decode.increment_load();
        }
        let admission = admit_decode(&decode, "m", 1).await.expect("no shed");
        assert!(admission.is_none(), "an unreported window claims nothing");
        assert_eq!(decode.pd_admitted(), 0);
    }

    /// Below the window there is nothing to wait for — but the room is still
    /// claimed, and released with the guard.
    #[tokio::test(start_paused = true)]
    async fn a_free_window_admits_without_waiting_and_releases_on_drop() {
        let decode = decode_worker("grpc://127.0.0.1:9902", Some(4));
        let started = Instant::now();
        let admission = admit_decode(&decode, "m", 1)
            .await
            .expect("no shed")
            .expect("a claim");
        assert_eq!(
            started.elapsed(),
            Duration::ZERO,
            "no wait below the window"
        );
        assert_eq!(decode.pd_admitted(), 1, "admission claims its room");

        drop(admission);
        assert_eq!(decode.pd_admitted(), 0, "the claim releases with the guard");
    }

    /// The claim, not a read, is what bounds the window: N requests racing a
    /// window of one must produce exactly one admission, and the loser sheds.
    #[tokio::test(start_paused = true)]
    async fn concurrent_requests_never_exceed_a_window_of_one() {
        set_pd_admission_wait_secs(1);
        let decode = decode_worker("grpc://127.0.0.1:9903", Some(1));

        let live = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let attempts = (0..8).map(|_| async {
            let outcome = admit_decode(&decode, "m", 1).await;
            if outcome.as_ref().is_ok_and(Option::is_some) {
                let now = live.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                peak.fetch_max(now, AtomicOrdering::SeqCst);
                // Hold the claim past every other attempt's deadline.
                tokio::time::sleep(Duration::from_secs(5)).await;
                live.fetch_sub(1, AtomicOrdering::SeqCst);
            }
            outcome
        });
        let outcomes = join_all(attempts).await;

        let admitted = outcomes
            .iter()
            .filter(|outcome| outcome.as_ref().is_ok_and(Option::is_some))
            .count();
        assert_eq!(admitted, 1, "a window of one admits exactly one request");
        assert_eq!(peak.load(AtomicOrdering::SeqCst), 1, "never two at once");
        assert_eq!(
            outcomes.iter().filter(|o| o.is_err()).count(),
            7,
            "every request that could not claim a room is shed"
        );
        set_pd_admission_wait_secs(DEFAULT_PD_ADMISSION_WAIT_SECS);
    }

    /// A batched plan claims one room per sub-request, so its siblings cannot
    /// slip past a gate that only ever checked for one.
    #[tokio::test(start_paused = true)]
    async fn a_batched_plan_claims_one_room_per_sub_request() {
        set_pd_admission_wait_secs(1);
        let decode = decode_worker("grpc://127.0.0.1:9904", Some(4));

        let batch = admit_decode(&decode, "m", 4)
            .await
            .expect("no shed")
            .expect("a claim");
        assert_eq!(decode.pd_admitted(), 4, "one room per sub-request");

        // The window is now full: a single-room request must shed, not slip in.
        assert!(
            admit_decode(&decode, "m", 1).await.is_err(),
            "a full window admits nothing more"
        );

        drop(batch);
        assert!(admit_decode(&decode, "m", 1).await.is_ok());
        set_pd_admission_wait_secs(DEFAULT_PD_ADMISSION_WAIT_SECS);
    }

    /// A plan wider than the window can never be admitted, so it sheds at
    /// once rather than sleeping out the budget to reach the same answer.
    #[tokio::test(start_paused = true)]
    async fn a_plan_wider_than_the_window_sheds_without_waiting() {
        let decode = decode_worker("grpc://127.0.0.1:9905", Some(4));
        let started = Instant::now();

        let response = admit_decode(&decode, "m", 5).await.expect_err("shed");

        assert_eq!(started.elapsed(), Duration::ZERO);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(decode.pd_admitted(), 0, "a shed claims nothing");
    }

    /// A room freeing mid-wait releases the request instead of shedding it.
    #[tokio::test(start_paused = true)]
    async fn a_freed_room_admits_the_waiting_request() {
        let claimed = AtomicUsize::new(1);
        let free_a_room = async {
            tokio::time::sleep(Duration::from_millis(500)).await;
            claimed.store(0, AtomicOrdering::SeqCst);
        };
        let gate = claim_within(Duration::from_secs(30), None, || {
            claimed
                .compare_exchange(0, 1, AtomicOrdering::SeqCst, AtomicOrdering::SeqCst)
                .is_ok()
        });

        let ((), claim) = tokio::join!(free_a_room, gate);

        assert_eq!(claim, Claim::Waited);
    }

    /// A window that never frees sheds at the deadline, not before it.
    #[tokio::test(start_paused = true)]
    async fn a_full_window_sheds_at_the_deadline() {
        let started = Instant::now();
        let claim = claim_within(Duration::from_secs(30), None, || false).await;
        assert_eq!(claim, Claim::Refused);
        assert!(
            started.elapsed() >= Duration::from_secs(30),
            "the shed must wait out the whole admission window, waited {:?}",
            started.elapsed()
        );
    }

    /// A zero wait is the "shed immediately" setting: no sleep, no admission.
    #[tokio::test(start_paused = true)]
    async fn a_zero_wait_sheds_without_sleeping() {
        let started = Instant::now();
        let claim = claim_within(Duration::ZERO, None, || false).await;
        assert_eq!(claim, Claim::Refused);
        assert_eq!(started.elapsed(), Duration::ZERO);
    }

    /// A decode worker that reports no window is announced once, as a WARN,
    /// with the gauge at zero; one that reports a window publishes it.
    #[tracing_test::traced_test]
    #[test]
    fn an_unreported_window_is_announced_and_published_as_zero() {
        use crate::observability::metrics::test_support::render_with_recorder;

        let rendered = render_with_recorder(|| {
            announce_decode_window(decode_worker("grpc://127.0.0.1:9907", None).as_ref());
            announce_decode_window(decode_worker("grpc://127.0.0.1:9908", Some(64)).as_ref());
        });
        logs_assert(|lines: &[&str]| {
            let warned: Vec<&&str> = lines
                .iter()
                .filter(|line| line.contains("PD admission is off for this decode worker"))
                .collect();
            match warned.as_slice() {
                [line] if line.contains("grpc://127.0.0.1:9907") => Ok(()),
                _ => Err(format!(
                    "one warning, for the worker without a window, expected: {warned:?}"
                )),
            }
        });
        for (worker, value) in [("9907", " 0"), ("9908", " 64")] {
            let line = rendered
                .lines()
                .find(|line| {
                    line.starts_with("smg_pd_admission_window{")
                        && line.contains(&format!("worker=\"grpc://127.0.0.1:{worker}\""))
                })
                .unwrap_or_else(|| panic!("no window gauge for {worker}:\n{rendered}"));
            assert!(line.ends_with(value), "{line}");
        }
    }

    /// Only decode workers carry the gate's window; a prefill worker without
    /// one is not announced.
    #[tracing_test::traced_test]
    #[test]
    fn only_decode_workers_are_announced() {
        use crate::observability::metrics::test_support::render_with_recorder;

        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:9909")
                .model(ModelCard::new("m"))
                .worker_type(WorkerType::Prefill)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        let rendered = render_with_recorder(|| announce_decode_window(prefill.as_ref()));
        assert!(!logs_contain("PD admission is off"));
        assert!(!rendered.contains("smg_pd_admission_window{"), "{rendered}");
    }

    /// A room that frees goes to the request that has waited longest. With
    /// the window full and two requests waiting, a newcomer that tries in the
    /// instant the room frees — before either sleeping waiter wakes — claims
    /// nothing: the first waiter is admitted, then the second, then the
    /// newcomer.
    #[tokio::test(start_paused = true)]
    async fn a_freed_room_goes_to_the_oldest_waiter_not_to_a_newcomer() {
        let decode = decode_worker("grpc://127.0.0.1:9910", Some(1));
        let order = Mutex::new(Vec::new());
        let holder = admit_decode(&decode, "m", 1)
            .await
            .expect("no shed")
            .expect("a claim");

        // Arrive after `at`; once admitted, hold the room a moment so who
        // comes next is observable.
        async fn arrive(
            decode: &Arc<dyn Worker>,
            order: &Mutex<Vec<&'static str>>,
            name: &'static str,
            at: Duration,
        ) {
            tokio::time::sleep(at).await;
            let admission = admit_decode(decode, "m", 1)
                .await
                .expect("no shed")
                .expect("a claim");
            order.lock().unwrap().push(name);
            tokio::time::sleep(Duration::from_millis(10)).await;
            drop(admission);
        }
        // The room frees at 20 ms, and the newcomer tries in that same
        // instant: well inside both waiters' retry interval.
        let free_the_room = async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(holder);
        };
        tokio::time::timeout(
            Duration::from_secs(5),
            futures::future::join4(
                arrive(&decode, &order, "first waiter", Duration::ZERO),
                arrive(&decode, &order, "second waiter", Duration::from_millis(10)),
                free_the_room,
                arrive(&decode, &order, "newcomer", Duration::from_millis(20)),
            ),
        )
        .await
        .expect("everyone is admitted well within the budget");

        assert_eq!(
            *order.lock().unwrap(),
            ["first waiter", "second waiter", "newcomer"],
            "rooms are handed out in arrival order"
        );
        assert_eq!(decode.pd_admitted(), 0);
    }

    /// A dispatch that stops waiting (its client went away) gives its place
    /// back: the waiter behind it is served when the room frees, not left
    /// behind a place nobody holds until the budget sheds it.
    #[tokio::test(start_paused = true)]
    async fn a_waiter_that_stops_waiting_gives_its_place_to_the_next() {
        let decode = decode_worker("grpc://127.0.0.1:9911", Some(1));
        let holder = admit_decode(&decode, "m", 1)
            .await
            .expect("no shed")
            .expect("a claim");

        // Both take their places with the window full; then the first is
        // dropped mid-wait.
        let mut abandoned = Box::pin(admit_decode(&decode, "m", 1));
        assert!(futures::poll!(abandoned.as_mut()).is_pending());
        let mut behind = Box::pin(admit_decode(&decode, "m", 1));
        assert!(futures::poll!(behind.as_mut()).is_pending());
        drop(abandoned);

        drop(holder);
        let started = Instant::now();
        let admission = tokio::time::timeout(Duration::from_secs(5), behind)
            .await
            .expect("served, not stuck behind a place nobody holds")
            .expect("no shed")
            .expect("a claim");
        assert_eq!(
            started.elapsed(),
            Duration::ZERO,
            "woken by the release, not served at a later retry"
        );
        drop(admission);
    }

    /// A room released wakes the oldest waiter at once: it claims without
    /// waiting for its next retry, so a freed room does not idle under a full
    /// window.
    #[tokio::test(start_paused = true)]
    async fn a_released_room_wakes_the_oldest_waiter_at_once() {
        let decode = decode_worker("grpc://127.0.0.1:9913", Some(1));
        let holder = admit_decode(&decode, "m", 1)
            .await
            .expect("no shed")
            .expect("a claim");
        let mut waiter = Box::pin(admit_decode(&decode, "m", 1));
        assert!(
            futures::poll!(waiter.as_mut()).is_pending(),
            "the window is full"
        );

        drop(holder);
        let started = Instant::now();
        let admission = waiter.await.expect("no shed").expect("a claim");
        assert_eq!(
            started.elapsed(),
            Duration::ZERO,
            "woken by the release, not by the retry interval"
        );
        drop(admission);
    }

    /// A request that needs more rooms than are free does not hold back the
    /// narrower requests behind it: they take the rooms that free one at a
    /// time, and the wide request claims when its rooms are free together.
    #[tokio::test(start_paused = true)]
    async fn a_wide_request_at_the_front_does_not_hold_back_the_narrow_ones_behind_it() {
        let decode = decode_worker("grpc://127.0.0.1:9914", Some(2));
        let first_room = admit_decode(&decode, "m", 1)
            .await
            .expect("no shed")
            .expect("a claim");
        let second_room = admit_decode(&decode, "m", 1)
            .await
            .expect("no shed")
            .expect("a claim");

        // The wide request is the oldest waiter, a narrow one behind it.
        let mut wide = Box::pin(admit_decode(&decode, "m", 2));
        assert!(futures::poll!(wide.as_mut()).is_pending());
        let mut narrow = Box::pin(admit_decode(&decode, "m", 1));
        assert!(futures::poll!(narrow.as_mut()).is_pending());

        // One room frees: the narrow request takes it, the wide one waits on.
        drop(first_room);
        let narrow = tokio::time::timeout(Duration::from_secs(5), narrow)
            .await
            .expect("the narrow request is not held back by the wide one")
            .expect("no shed")
            .expect("a claim");
        assert!(
            futures::poll!(wide.as_mut()).is_pending(),
            "two rooms are not free together yet"
        );
        assert_eq!(decode.pd_admitted(), 2);

        // The other held room frees: still one room free, the wide one waits.
        drop(second_room);
        assert!(futures::poll!(wide.as_mut()).is_pending());

        // The narrow request ends: its rooms are free together, it claims.
        drop(narrow);
        let wide = tokio::time::timeout(Duration::from_secs(5), wide)
            .await
            .expect("the wide request claims once its rooms are free together")
            .expect("no shed")
            .expect("a claim");
        assert_eq!(decode.pd_admitted(), 2);
        drop(wide);
    }

    /// The budget is still the bound: a waiter whose room never frees sheds
    /// at the deadline, and its place goes with it, so the next request that
    /// finds the room free claims it at once.
    #[tokio::test(start_paused = true)]
    async fn the_shed_at_the_deadline_gives_the_place_back() {
        set_pd_admission_wait_secs(1);
        let decode = decode_worker("grpc://127.0.0.1:9912", Some(1));
        let holder = admit_decode(&decode, "m", 1)
            .await
            .expect("no shed")
            .expect("a claim");

        let started = Instant::now();
        let response = admit_decode(&decode, "m", 1)
            .await
            .expect_err("a room that never frees sheds");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            started.elapsed() >= Duration::from_secs(1),
            "the shed waits out the budget"
        );

        drop(holder);
        let started = Instant::now();
        let admission = admit_decode(&decode, "m", 1)
            .await
            .expect("no shed")
            .expect("a claim");
        assert_eq!(
            started.elapsed(),
            Duration::ZERO,
            "a shed waiter holds no place"
        );
        drop(admission);
        set_pd_admission_wait_secs(DEFAULT_PD_ADMISSION_WAIT_SECS);
    }

    /// The shed is the overload guard's 503, terminal for the retry layer and
    /// carrying the client's pacing hint.
    #[tokio::test(start_paused = true)]
    async fn the_shed_is_the_overload_503_with_retry_after() {
        set_pd_admission_wait_secs(1);
        let decode = decode_worker("grpc://127.0.0.1:9906", Some(2));
        assert!(decode.try_admit_pd(2, 2), "fill the window");

        let response = admit_decode(&decode, "m", 1)
            .await
            .expect_err("a full window sheds");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            extract_error_code_from_response(&response),
            "worker_overload_protection_shed"
        );
        assert!(
            response
                .headers()
                .get(RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .is_some_and(|secs| secs >= 1),
            "Retry-After must carry whole seconds"
        );
        assert!(
            !is_retryable_response(&response),
            "the wait already outlived any backoff a retry would add"
        );
        set_pd_admission_wait_secs(DEFAULT_PD_ADMISSION_WAIT_SECS);
    }
}
