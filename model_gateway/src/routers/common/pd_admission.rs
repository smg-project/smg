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
//! Nothing here runs when the engine does not report a window: admission is
//! not the gateway's to decide then, and dispatch behaves exactly as it did
//! before this module existed.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::response::Response;
use tokio::time::Instant;
use tracing::debug;

use crate::{observability::metrics::Metrics, routers::common::overload, worker::Worker};

/// Default seconds a PD dispatch may wait for decode rooms. Well under the
/// engines' bootstrap deadline (120 s on TokenSpeed), so a request that does
/// wait still dispatches with the whole deadline ahead of it.
pub const DEFAULT_PD_ADMISSION_WAIT_SECS: u64 = 30;

/// How often the wait retries its claim.
///
/// Polling, not a notifier: the event we would signal is a PD admission guard
/// dropping, and a per-worker registry of notifiers would be process-wide
/// mutable state with its own eviction problem. 50 ms is far finer than both
/// the engine step that actually frees a room and the wait deadline below,
/// and the sleep is asynchronous — a waiting request occupies no thread.
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

/// Retry `try_claim` until it succeeds or `wait` elapses.
///
/// The claim itself is what makes admission safe under concurrency, so this
/// never reads a count: every attempt is the same all-or-nothing reservation,
/// and the first one to succeed owns the rooms.
async fn claim_within(wait: Duration, try_claim: impl Fn() -> bool) -> Claim {
    if try_claim() {
        return Claim::Immediate;
    }
    let deadline = Instant::now() + wait;
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Claim::Refused;
        }
        tokio::time::sleep(CLAIM_RETRY_INTERVAL.min(deadline - now)).await;
        if try_claim() {
            return Claim::Waited;
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
    let claim = claim_within(wait, || decode.try_admit_pd(rooms, window)).await;
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

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
        let gate = claim_within(Duration::from_secs(30), || {
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
        let claim = claim_within(Duration::from_secs(30), || false).await;
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
        let claim = claim_within(Duration::ZERO, || false).await;
        assert_eq!(claim, Claim::Refused);
        assert_eq!(started.elapsed(), Duration::ZERO);
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
