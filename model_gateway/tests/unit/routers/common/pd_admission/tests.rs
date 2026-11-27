use std::sync::{
    atomic::{AtomicUsize, Ordering as AtomicOrdering},
    Mutex,
};

use axum::http::{header::RETRY_AFTER, StatusCode};
use futures::future::join_all;
use openai_protocol::model_card::ModelCard;
use smg_worker::{
    identity::basic::{ConnectionMode, WorkerType},
    test_support::no_health_check,
};

use super::*;
use crate::routers::{
    common::retry::is_retryable_response, error::extract_error_code_from_response,
    grpc::backend_client::BasicWorkerBuilder,
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
            .health_config(no_health_check())
            .build(),
    )
}

/// One try with no wait claims a free room and holds nothing where the
/// window is full. It yields to an older place that is in line for the
/// room, so a dispatch that would move here passes no waiting request.
/// Where no window is reported the gate abstains.
#[test]
fn one_try_claims_now_or_holds_nothing() {
    let decode = decode_worker("grpc://one-try:9000", Some(1));
    let Tried::Claimed(claim) = try_admit_decode(&decode, 1) else {
        panic!("a free room is claimed");
    };
    assert_eq!(decode.pd_admitted(), 1);
    assert!(matches!(try_admit_decode(&decode, 1), Tried::Full));
    assert_eq!(decode.pd_waiting_rooms(), 0, "the try left its place");

    // A request waits for the room, and the room frees: the try yields.
    let waiting = decode.pd_enqueue(1);
    drop(claim);
    assert_eq!(decode.pd_admitted(), 0);
    assert!(matches!(try_admit_decode(&decode, 1), Tried::Full));
    assert_eq!(decode.pd_admitted(), 0);
    decode.pd_dequeue(waiting);
    assert!(matches!(try_admit_decode(&decode, 1), Tried::Claimed(_)));
    assert_eq!(
        decode.pd_admitted(),
        0,
        "a claim dropped gives its room back"
    );

    let open = decode_worker("grpc://no-window:9000", None);
    assert!(matches!(try_admit_decode(&open, 1), Tried::NoWindow));
}

/// The selection's read of the gate: a worker without a window always
/// has a room (the gate abstains there); with one, the rooms claimed and
/// the rooms the waiting places need both take from it.
#[tokio::test]
async fn a_free_room_is_one_no_claim_and_no_waiting_place_has_taken() {
    assert!(has_free_room(
        decode_worker("grpc://127.0.0.1:9910", None).as_ref()
    ));
    let decode = decode_worker("grpc://127.0.0.1:9911", Some(2));
    assert!(has_free_room(decode.as_ref()));
    let first = admit_decode(&decode, "m", 1)
        .await
        .expect("no shed")
        .expect("a claim");
    assert!(has_free_room(decode.as_ref()), "one of two rooms is free");
    let place = decode.pd_enqueue(1);
    assert!(
        !has_free_room(decode.as_ref()),
        "the free room is spoken for by the waiting place"
    );
    decode.pd_dequeue(place);
    assert!(has_free_room(decode.as_ref()), "the place gave it back");
    let second = admit_decode(&decode, "m", 1)
        .await
        .expect("no shed")
        .expect("a claim");
    assert!(!has_free_room(decode.as_ref()), "the window is full");
    drop(first);
    assert!(
        has_free_room(decode.as_ref()),
        "a released room is free again"
    );
    drop(second);
}

/// A wide place ahead counts every room it needs: the gate itself would
/// let a narrower newcomer pass it, the selection answers no, and the
/// request goes to a sibling with a room when there is one.
#[test]
fn a_waiting_batch_counts_every_room_it_needs() {
    let decode = decode_worker("grpc://127.0.0.1:9912", Some(4));
    let batch = decode.pd_enqueue(3);
    assert_eq!(decode.pd_waiting_rooms(), 3);
    assert!(
        has_free_room(decode.as_ref()),
        "one room beyond the batch's three"
    );
    let behind = decode.pd_enqueue(2);
    assert!(!has_free_room(decode.as_ref()));
    decode.pd_dequeue(behind);
    decode.pd_dequeue(batch);
    decode.pd_dequeue(batch);
    assert_eq!(
        decode.pd_waiting_rooms(),
        0,
        "a place given back twice counts once"
    );
    assert!(has_free_room(decode.as_ref()));
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
            .health_config(no_health_check())
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
