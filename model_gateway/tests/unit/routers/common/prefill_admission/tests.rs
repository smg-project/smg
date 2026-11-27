use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

use openai_protocol::worker::WorkerStatus;
use smg_worker::{health::circuit_breaker::CircuitBreakerConfig, identity::basic::WorkerType};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

use super::{test_support::select_worker, *};
use crate::{
    config::types::PolicyConfig,
    policies::PolicyRegistry,
    routers::{
        common::{
            pair_index::PdPairIndex,
            placement::{
                select_pair,
                tests::{
                    hearing_pair_sides_on, hearing_pairs_on, keyed_pairs_on, one_pair, pairs_of,
                    MODEL,
                },
                PairFailure, PlacedPair, PlacementInputs, Stated,
            },
            snapshot::WorkerRegistry,
        },
        grpc::backend_client::BasicWorkerBuilder,
    },
};

fn worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Prefill)
            .build(),
    )
}

/// A task that attempts `worker` through `admission` with the plain
/// select-this-worker attempt; the test joins it, and aborts it first where
/// it must never be admitted.
#[expect(
    clippy::disallowed_methods,
    reason = "the test joins (and where it must, aborts) the task before it ends"
)]
fn plain_attempt(
    admission: &PrefillAdmission,
    worker: &Arc<dyn Worker>,
) -> JoinHandle<Result<AdmittedPrefill<()>, PrefillAdmissionRejection>> {
    let admission = admission.clone();
    let worker = Arc::clone(worker);
    tokio::spawn(async move { admission.admit(None, select_worker(&worker)).await })
}

async fn wait_for_queued(admission: &PrefillAdmission, expected: usize) {
    for _ in 0..1_000 {
        if admission.queued_requests() == expected {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!(
        "queue depth did not reach {expected}; actual={}",
        admission.queued_requests()
    );
}

#[tokio::test]
async fn one_client_request_uses_one_worker_slot() {
    let admission = PrefillAdmission::new(1, 0, Duration::from_secs(1));
    let worker = worker("http://prefill-one");

    let admitted = admission
        .admit(None, {
            let worker = Arc::clone(&worker);
            move |capacity| capacity.select(Arc::clone(&worker), "selected")
        })
        .await
        .unwrap();

    assert_eq!(admitted.selected, "selected");
    assert_eq!(worker.load(), 1);
    drop(admitted.reservation);
    assert_eq!(worker.load(), 0);
}

#[tokio::test]
async fn zero_size_queue_rejects_at_capacity() {
    let admission = PrefillAdmission::new(1, 0, Duration::from_secs(1));
    let worker = worker("http://prefill-no-queue");
    let first = admission.admit(None, select_worker(&worker)).await.unwrap();

    let result = admission.admit(None, select_worker(&worker)).await;
    assert!(matches!(result, Err(PrefillAdmissionRejection::QueueFull)));
    drop(first);
}

#[tokio::test]
async fn full_queue_rejects_without_bypassing_waiter() {
    let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
    let worker = worker("http://prefill-full-queue");
    let first = admission.admit(None, select_worker(&worker)).await.unwrap();

    let waiter = plain_attempt(&admission, &worker);
    wait_for_queued(&admission, 1).await;

    let policy_called = Arc::new(AtomicBool::new(false));
    let result = admission
        .admit(None, {
            let policy_called = Arc::clone(&policy_called);
            let worker = Arc::clone(&worker);
            move |capacity| {
                if !capacity.has_capacity(&worker) {
                    return PrefillAdmissionAttempt::AtCapacity;
                }
                policy_called.store(true, Ordering::Relaxed);
                capacity.select(Arc::clone(&worker), ())
            }
        })
        .await;
    assert!(matches!(result, Err(PrefillAdmissionRejection::QueueFull)));
    assert!(!policy_called.load(Ordering::Relaxed));

    waiter.abort();
    assert!(matches!(
        waiter.await,
        Err(error) if error.is_cancelled()
    ));
    drop(first);
}

#[tokio::test(start_paused = true)]
async fn queued_request_times_out_and_removes_its_ticket() {
    let admission = PrefillAdmission::new(1, 1, Duration::from_secs(5));
    let worker = worker("http://prefill-timeout");
    let first = admission.admit(None, select_worker(&worker)).await.unwrap();

    let waiter = plain_attempt(&admission, &worker);
    wait_for_queued(&admission, 1).await;

    tokio::time::advance(Duration::from_secs(5)).await;
    assert!(matches!(
        waiter.await.unwrap(),
        Err(PrefillAdmissionRejection::QueueTimeout)
    ));
    assert_eq!(admission.queued_requests(), 0);
    drop(first);
}

#[tokio::test]
async fn released_worker_load_is_visible_to_the_next_selection() {
    let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
    let worker = worker("http://prefill-release-order");
    let first = admission.admit(None, select_worker(&worker)).await.unwrap();

    #[expect(
        clippy::disallowed_methods,
        reason = "test waiter task is joined before the test ends"
    )]
    let waiter = tokio::spawn({
        let admission = admission.clone();
        let worker = Arc::clone(&worker);
        async move {
            admission
                .admit(None, move |capacity| {
                    if !capacity.has_capacity(&worker) {
                        return PrefillAdmissionAttempt::AtCapacity;
                    }
                    assert_eq!(worker.load(), 0);
                    capacity.select(Arc::clone(&worker), ())
                })
                .await
        }
    });
    wait_for_queued(&admission, 1).await;

    drop(first);
    let admitted = waiter.await.unwrap().unwrap();
    drop(admitted);
}

#[tokio::test]
async fn queued_requests_are_admitted_in_strict_fifo_order() {
    let admission = PrefillAdmission::new(1, 2, Duration::from_secs(1));
    let worker = worker("http://prefill-fifo");
    let initial = admission.admit(None, select_worker(&worker)).await.unwrap();
    let (acquired_tx, mut acquired_rx) = mpsc::unbounded_channel();
    let (release_first_tx, release_first_rx) = oneshot::channel();

    #[expect(
        clippy::disallowed_methods,
        reason = "test waiter task is joined before the test ends"
    )]
    let first = tokio::spawn({
        let admission = admission.clone();
        let worker = Arc::clone(&worker);
        let acquired_tx = acquired_tx.clone();
        async move {
            let admitted = admission.admit(None, select_worker(&worker)).await.unwrap();
            acquired_tx.send(1).unwrap();
            let _ = release_first_rx.await;
            drop(admitted);
        }
    });
    wait_for_queued(&admission, 1).await;

    #[expect(
        clippy::disallowed_methods,
        reason = "test waiter task is joined before the test ends"
    )]
    let second = tokio::spawn({
        let admission = admission.clone();
        let worker = Arc::clone(&worker);
        async move {
            let admitted = admission.admit(None, select_worker(&worker)).await.unwrap();
            acquired_tx.send(2).unwrap();
            drop(admitted);
        }
    });
    wait_for_queued(&admission, 2).await;

    drop(initial);
    assert_eq!(acquired_rx.recv().await, Some(1));
    assert!(acquired_rx.try_recv().is_err());
    release_first_tx.send(()).unwrap();
    assert_eq!(acquired_rx.recv().await, Some(2));

    first.await.unwrap();
    second.await.unwrap();
    assert_eq!(worker.load(), 0);
}

#[tokio::test]
async fn admitted_head_wakes_next_head_while_capacity_remains() {
    let admission = PrefillAdmission::new(2, 2, Duration::from_secs(1));
    let worker = worker("http://prefill-drain");
    let available = Arc::new(AtomicBool::new(false));
    let (acquired_tx, mut acquired_rx) = mpsc::unbounded_channel();
    let (release_first_tx, release_first_rx) = oneshot::channel();

    #[expect(
        clippy::disallowed_methods,
        reason = "test waiter task is joined before the test ends"
    )]
    let first = tokio::spawn({
        let admission = admission.clone();
        let worker = Arc::clone(&worker);
        let available = Arc::clone(&available);
        let acquired_tx = acquired_tx.clone();
        async move {
            let admitted = admission
                .admit(None, move |capacity| {
                    if available.load(Ordering::Relaxed) {
                        capacity.select(Arc::clone(&worker), ())
                    } else {
                        PrefillAdmissionAttempt::AtCapacity
                    }
                })
                .await
                .unwrap();
            acquired_tx.send(1).unwrap();
            let _ = release_first_rx.await;
            drop(admitted);
        }
    });
    wait_for_queued(&admission, 1).await;

    #[expect(
        clippy::disallowed_methods,
        reason = "test waiter task is joined before the test ends"
    )]
    let second = tokio::spawn({
        let admission = admission.clone();
        let worker = Arc::clone(&worker);
        let available = Arc::clone(&available);
        async move {
            let admitted = admission
                .admit(None, move |capacity| {
                    if available.load(Ordering::Relaxed) {
                        capacity.select(Arc::clone(&worker), ())
                    } else {
                        PrefillAdmissionAttempt::AtCapacity
                    }
                })
                .await
                .unwrap();
            acquired_tx.send(2).unwrap();
            drop(admitted);
        }
    });
    wait_for_queued(&admission, 2).await;

    available.store(true, Ordering::Relaxed);
    admission.notify_capacity_changed();
    assert_eq!(acquired_rx.recv().await, Some(1));
    assert_eq!(acquired_rx.recv().await, Some(2));
    release_first_tx.send(()).unwrap();

    first.await.unwrap();
    second.await.unwrap();
    assert_eq!(worker.load(), 0);
}

#[tokio::test]
async fn queue_head_rechecks_circuit_breaker_recovery_without_event() {
    let admission = PrefillAdmission::new(1, 1, Duration::from_secs(5));
    let full = worker("http://prefill-full");
    let occupied = admission.admit(None, select_worker(&full)).await.unwrap();
    let recovering: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://prefill-recovering")
            .worker_type(WorkerType::Prefill)
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 1,
                timeout_duration: Duration::from_millis(250),
                ..Default::default()
            })
            .build(),
    );
    recovering.set_status(WorkerStatus::Ready);
    recovering.record_circuit_breaker_outcome(false);
    assert!(!recovering.is_available());

    let admitted = tokio::time::timeout(
        Duration::from_secs(3),
        admission.admit(None, {
            let full = Arc::clone(&full);
            let recovering = Arc::clone(&recovering);
            move |capacity| {
                if recovering.is_available() {
                    capacity.select(Arc::clone(&recovering), ())
                } else {
                    capacity.select(Arc::clone(&full), ())
                }
            }
        }),
    )
    .await
    .expect("queue head should recheck a silently recovered worker")
    .unwrap();

    assert_eq!(full.load(), 1);
    assert_eq!(recovering.load(), 1);
    drop(admitted);
    drop(occupied);
    assert_eq!(full.load(), 0);
    assert_eq!(recovering.load(), 0);
}

#[tokio::test]
async fn cancelled_waiters_are_removed_without_blocking_fifo() {
    let admission = PrefillAdmission::new(1, 2, Duration::from_secs(1));
    let worker = worker("http://prefill-cancel-head");
    let initial = admission.admit(None, select_worker(&worker)).await.unwrap();

    let cancelled = plain_attempt(&admission, &worker);
    wait_for_queued(&admission, 1).await;

    let middle = plain_attempt(&admission, &worker);
    wait_for_queued(&admission, 2).await;
    middle.abort();
    assert!(matches!(
        middle.await,
        Err(error) if error.is_cancelled()
    ));
    wait_for_queued(&admission, 1).await;

    let attempts = Arc::new(AtomicUsize::new(0));
    #[expect(
        clippy::disallowed_methods,
        reason = "test waiter task is joined before the test ends"
    )]
    let next = tokio::spawn({
        let admission = admission.clone();
        let worker = Arc::clone(&worker);
        let attempts = Arc::clone(&attempts);
        async move {
            admission
                .admit(None, move |capacity| {
                    attempts.fetch_add(1, Ordering::Relaxed);
                    capacity.select(Arc::clone(&worker), ())
                })
                .await
        }
    });
    wait_for_queued(&admission, 2).await;

    cancelled.abort();
    assert!(matches!(
        cancelled.await,
        Err(error) if error.is_cancelled()
    ));
    wait_for_queued(&admission, 1).await;
    for _ in 0..1_000 {
        if attempts.load(Ordering::Relaxed) > 0 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(attempts.load(Ordering::Relaxed) > 0);

    drop(initial);
    let admitted = next.await.unwrap().unwrap();
    drop(admitted);
}

#[tokio::test]
async fn queued_request_selects_again_without_worker_binding() {
    let admission = PrefillAdmission::new(1, 1, Duration::from_secs(1));
    let first_worker = worker("http://prefill-a");
    let second_worker = worker("http://prefill-b");
    let first = admission
        .admit(None, select_worker(&first_worker))
        .await
        .unwrap();
    let second = admission
        .admit(None, select_worker(&second_worker))
        .await
        .unwrap();
    let policy_calls = Arc::new(AtomicUsize::new(0));

    #[expect(
        clippy::disallowed_methods,
        reason = "test waiter task is joined before the test ends"
    )]
    let waiter = tokio::spawn({
        let admission = admission.clone();
        let first_worker = Arc::clone(&first_worker);
        let second_worker = Arc::clone(&second_worker);
        let policy_calls = Arc::clone(&policy_calls);
        async move {
            admission
                .admit(None, move |capacity| {
                    let selected = if capacity.has_capacity(&first_worker) {
                        Some(Arc::clone(&first_worker))
                    } else if capacity.has_capacity(&second_worker) {
                        Some(Arc::clone(&second_worker))
                    } else {
                        None
                    };
                    let Some(selected) = selected else {
                        return PrefillAdmissionAttempt::AtCapacity;
                    };
                    policy_calls.fetch_add(1, Ordering::Relaxed);
                    let url = selected.url().to_owned();
                    capacity.select(selected, url)
                })
                .await
        }
    });
    wait_for_queued(&admission, 1).await;
    assert_eq!(policy_calls.load(Ordering::Relaxed), 0);

    drop(second);
    let admitted = waiter.await.unwrap().unwrap();
    assert_eq!(admitted.selected, "http://prefill-b");
    assert_eq!(policy_calls.load(Ordering::Relaxed), 1);
    drop(admitted);
    drop(first);
}

#[tokio::test]
async fn unavailable_head_exposes_the_next_waiter() {
    let admission = PrefillAdmission::new(1, 2, Duration::from_secs(1));
    let worker = worker("http://prefill-unavailable-head");
    let unavailable = Arc::new(AtomicBool::new(false));

    #[expect(
        clippy::disallowed_methods,
        reason = "test waiter task is joined before the test ends"
    )]
    let first = tokio::spawn({
        let admission = admission.clone();
        let unavailable = Arc::clone(&unavailable);
        async move {
            admission
                .admit(None, move |_| {
                    if unavailable.load(Ordering::Relaxed) {
                        PrefillAdmissionAttempt::<()>::Unavailable
                    } else {
                        PrefillAdmissionAttempt::AtCapacity
                    }
                })
                .await
        }
    });
    wait_for_queued(&admission, 1).await;

    let second = plain_attempt(&admission, &worker);
    wait_for_queued(&admission, 2).await;

    unavailable.store(true, Ordering::Relaxed);
    admission.notify_capacity_changed();
    assert!(matches!(
        first.await.unwrap(),
        Err(PrefillAdmissionRejection::Unavailable)
    ));
    let admitted = second.await.unwrap().unwrap();
    drop(admitted);
}

#[tokio::test]
async fn unavailable_is_not_queued() {
    let admission = PrefillAdmission::new(1, 2, Duration::from_secs(1));
    let worker = worker("http://prefill-unavailable");
    let occupied = admission.admit(None, select_worker(&worker)).await.unwrap();

    let queued = plain_attempt(&admission, &worker);
    wait_for_queued(&admission, 1).await;

    let result = admission
        .admit(None, |_| PrefillAdmissionAttempt::<()>::Unavailable)
        .await;

    assert!(matches!(
        result,
        Err(PrefillAdmissionRejection::Unavailable)
    ));
    assert_eq!(admission.queued_requests(), 1);

    queued.abort();
    assert!(matches!(
        queued.await,
        Err(error) if error.is_cancelled()
    ));
    drop(occupied);
}

#[tokio::test]
async fn an_admitted_prefill_reports_completion_when_its_reservation_drops() {
    use smg_worker::identity::basic::{RequestCompletionSink, Worker, WorkerType};

    use crate::routers::grpc::backend_client::BasicWorkerBuilder;
    #[derive(Debug, Default)]
    struct CompletionSpy(std::sync::Mutex<Vec<String>>);
    impl RequestCompletionSink for CompletionSpy {
        fn request_completed(&self, worker: &dyn Worker) {
            self.0.lock().unwrap().push(worker.url().to_string());
        }
    }

    let spy = Arc::new(CompletionSpy::default());
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://prefill-admitted")
            .worker_type(WorkerType::Prefill)
            .build(),
    );
    worker.set_completion_sink(Some(spy.clone() as Arc<dyn RequestCompletionSink>));
    let admission = PrefillAdmission::new(1, 0, Duration::from_secs(1));
    let admitted = admission
        .admit(None, select_worker(&worker))
        .await
        .expect("admitted");
    assert_eq!(worker.load(), 1);
    assert!(spy.0.lock().unwrap().is_empty());
    drop(admitted);
    assert_eq!(worker.load(), 0);
    assert_eq!(
        spy.0.lock().unwrap().as_slice(),
        ["grpc://prefill-admitted"]
    );
}

/// A pair's selection as a pair route hands it to [`acquire_prefill`]:
/// the pair placement over the fleet's pair index, under `capacity`.
fn pair_selection<'a>(
    registry: &'a WorkerRegistry,
    policies: &'a PolicyRegistry,
    pairs: &'a PdPairIndex,
) -> impl for<'c> FnMut(Option<&PrefillSelectionContext<'c>>) -> Result<PlacedPair, Box<PairFailure>>
       + Send
       + 'a {
    move |capacity| {
        let inputs = PlacementInputs::default();
        select_pair(
            registry,
            policies,
            MODEL,
            pairs,
            None,
            capacity,
            inputs,
            Stated::nothing,
        )
    }
}

/// A pair's prefill handed on behind its load, as a pair route hands it
/// to [`acquire_prefill`]: the prefill's worker and the decode's answer
/// as the route keeps them, and what the prefill's answer owes its
/// send.
fn handed_on(pair: PlacedPair) -> ((Arc<dyn Worker>, Placed), Option<Unsent>) {
    let (prefill, unsent) = pair.prefill.unsent();
    ((prefill, pair.decode), unsent)
}

/// A pair acquired as a pair route acquires it, under the route's key
/// `"key"`, with admission on where `admission` is one.
async fn acquired(
    registry: &WorkerRegistry,
    policies: &PolicyRegistry,
    admission: Option<&PrefillAdmission>,
    inputs: PlacementInputs<'_>,
) -> ((Arc<dyn Worker>, Placed), PrefillAhead) {
    let pairs = pairs_of(registry, policies);
    let select = |capacity: Option<&PrefillSelectionContext<'_>>| {
        select_pair(
            registry,
            policies,
            MODEL,
            &pairs,
            None,
            capacity,
            inputs,
            Stated::nothing,
        )
    };
    let acquired = acquire_prefill(
        admission,
        Some("key"),
        |pair: &PlacedPair| &pair.prefill,
        select,
        handed_on,
    )
    .await;
    let Ok(acquired) = acquired else {
        panic!("the pair is placed");
    };
    acquired
}

/// A served pair of Grand Central through [`acquire_prefill`], with
/// admission off and on. The acquisition counts the prefill worker,
/// under the route's key, and tells the engine nothing. The engine
/// hears the prefill's dispatch where the route says its send, and the
/// load's release is the prefill side's end: two dispatches and two
/// ends in all, and no release. With admission on the reservation's own
/// load guard is the prefill's, so the prefill is counted once.
#[tokio::test]
async fn a_served_pair_of_the_engine_is_heard_at_its_prefills_send() {
    for admitted in [false, true] {
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_pairs_on(&policies, false);
        let said = || heard.lock().expect("the stage heard").clone();
        let admission = admitted.then(|| PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let inputs = PlacementInputs::default();
        let ((prefill, decode), ahead) =
            acquired(&registry, &policies, admission.as_ref(), inputs).await;
        assert_eq!(
            (prefill.load(), prefill.routing_key_inflight("key")),
            (1, 1),
            "counted from the selection on: admission {admitted}"
        );
        assert_eq!(
            said(),
            ["issued", "issued"],
            "nothing is said ahead of the send: admission {admitted}"
        );

        let load = ahead.sent();
        assert_eq!(said()[2..], ["dispatched Prefill"], "admission {admitted}");
        assert_eq!(prefill.load(), 1, "counted once: admission {admitted}");
        let decode_guard = decode.sent(decode.ahead(Some("key")));
        drop(decode);
        drop(load);
        assert_eq!(prefill.load(), 0, "admission {admitted}");
        assert_eq!(said()[3..], ["dispatched Decode", "completed Cancelled"]);
        drop(decode_guard);
        assert_eq!(
            said()[3..],
            [
                "dispatched Decode",
                "completed Cancelled",
                "completed Cancelled"
            ],
            "two dispatches, two ends, no release: admission {admitted}"
        );
    }
}

/// A pair of Grand Central whose prefill is never sent, with
/// admission off and on: its load is dropped as it was acquired. The
/// worker is counted no more, the admission's slot takes the next
/// request, and the engine hears the prefill's side released: one
/// issue, one release, no dispatch and no end.
#[tokio::test]
async fn a_prefill_never_sent_is_heard_as_a_release() {
    for admitted in [false, true] {
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_pairs_on(&policies, false);
        let said = || heard.lock().expect("the stage heard").clone();
        let admission = admitted.then(|| PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let inputs = PlacementInputs::default();
        let ((prefill, decode), ahead) =
            acquired(&registry, &policies, admission.as_ref(), inputs).await;
        assert_eq!(prefill.load(), 1, "admission {admitted}");
        assert_eq!(said(), ["issued", "issued"], "admission {admitted}");

        drop(ahead);
        assert_eq!(
            (prefill.load(), prefill.routing_key_load()),
            (0, 0),
            "admission {admitted}"
        );
        assert_eq!(
            said()[2..],
            ["released Abandoned"],
            "the prefill's side, with no dispatch: admission {admitted}"
        );
        drop(decode);
        assert_eq!(
            said()[2..],
            ["released Abandoned", "released Abandoned"],
            "admission {admitted}"
        );

        // The slot is free: the next request's pair is admitted.
        let (_, next) = acquired(&registry, &policies, admission.as_ref(), inputs).await;
        assert_eq!(prefill.load(), 1, "admission {admitted}");
        drop(next);
    }
}

/// What a load asks ahead of its send is asked for an answer of Grand
/// Central alone, with admission off and on. Refused, the load is
/// dropped as it was acquired and the engine hears the prefill's side
/// released; not refused, the engine hears the dispatch at the send. A
/// classic answer's send says nothing, and nothing is asked ahead of
/// it.
#[tokio::test]
async fn a_load_asks_ahead_of_its_send_for_the_engines_answer_alone() {
    for admitted in [false, true] {
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_pairs_on(&policies, false);
        let said = || heard.lock().expect("the stage heard").clone();
        let admission = admitted.then(|| PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let inputs = PlacementInputs::default();

        let ((prefill, decode), ahead) =
            acquired(&registry, &policies, admission.as_ref(), inputs).await;
        assert_eq!(
            ahead.ahead_of_its_send(|| Err("refused")),
            Err("refused"),
            "admission {admitted}"
        );
        drop(ahead);
        assert_eq!(prefill.load(), 0, "admission {admitted}");
        assert_eq!(
            said()[2..],
            ["released Abandoned"],
            "refused, with no dispatch and no end: admission {admitted}"
        );
        drop(decode);

        let heard_so_far = said().len();
        let ((_, decode), ahead) = acquired(&registry, &policies, admission.as_ref(), inputs).await;
        assert_eq!(
            ahead.ahead_of_its_send(|| Ok::<(), &str>(())),
            Ok(()),
            "admission {admitted}"
        );
        let load = ahead.sent();
        assert_eq!(
            said()[heard_so_far..],
            ["issued", "issued", "dispatched Prefill"],
            "admission {admitted}"
        );
        drop((load, decode));
    }

    let classic = PrefillAhead::classic(PrefillLoadGuard::Unbounded {
        _guard: WorkerLoadGuard::with_key(worker("grpc://classic:1"), None),
    });
    let asked = classic.ahead_of_its_send(|| -> Result<(), &str> {
        panic!("a classic answer's send asks nothing")
    });
    assert_eq!(asked, Ok(()));
}

/// Where Grand Central names a routing value for a pair, the
/// prefill worker counts the request under the route's own key from
/// the selection to the send, and under the engine's value from the
/// send on, with admission off and on. Its load stands at one from the
/// selection to the load's release. The document here reads the value
/// from the body's own id, which the engine keeps in the clear.
#[tokio::test]
async fn a_prefill_counts_under_the_engines_routing_value_from_its_send_on() {
    for admitted in [false, true] {
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        keyed_pairs_on(&policies);
        let admission = admitted.then(|| PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let inputs = PlacementInputs {
            rid: Some("conversation-1"),
            ..Default::default()
        };
        let ((prefill, _decode), ahead) =
            acquired(&registry, &policies, admission.as_ref(), inputs).await;
        let counts = || {
            (
                prefill.load(),
                prefill.routing_key_inflight("key"),
                prefill.routing_key_inflight("conversation-1"),
            )
        };
        assert_eq!(
            counts(),
            (1, 1, 0),
            "ahead of the send: admission {admitted}"
        );

        let load = ahead.sent();
        assert_eq!(
            counts(),
            (1, 0, 1),
            "from the send on: admission {admitted}"
        );
        let replica = load.replicate();
        drop(load);
        drop(replica);
        assert_eq!(counts(), (0, 0, 0), "admission {admitted}");
        assert_eq!(prefill.routing_key_load(), 0, "admission {admitted}");
    }
}

/// Under Grand Central a pair whose prefills are all at the
/// admission limit is answered as the wait, as a classic placement
/// answers it: the request is not refused, and with no queue it is
/// told the queue is full.
#[tokio::test]
async fn the_engines_pair_answers_prefills_at_their_limit_as_the_wait() {
    let registry = one_pair();
    let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
    let _heard = hearing_pairs_on(&policies, false);
    let pairs = pairs_of(&registry, &policies);
    let admission = PrefillAdmission::new(1, 0, Duration::from_secs(1));
    let acquire = || {
        acquire_prefill(
            Some(&admission),
            None,
            |pair: &PlacedPair| &pair.prefill,
            pair_selection(&registry, &policies, &pairs),
            handed_on,
        )
    };
    let Ok(held) = acquire().await else {
        panic!("the first pair is admitted");
    };

    let mut verdicts = Vec::new();
    let mut select = pair_selection(&registry, &policies, &pairs);
    let second = admission
        .admit(None, |capacity| match select(Some(capacity)) {
            Ok(pair) => capacity.select(pair.prefill.into_worker(), ()),
            Err(failure) => {
                verdicts.push(failure.is_at_capacity());
                PrefillAdmissionAttempt::AtCapacity
            }
        })
        .await;
    assert!(matches!(second, Err(PrefillAdmissionRejection::QueueFull)));
    assert_eq!(verdicts, [true], "the engine's own answer is the wait");
    assert!(matches!(
        acquire().await,
        Err(PrefillAcquireError::Rejected(
            PrefillAdmissionRejection::QueueFull
        ))
    ));

    drop(held);
    assert!(acquire().await.is_ok(), "a released prefill takes the next");
}

/// Each child of a fan-out or a batch holds a replica of its request's
/// one prefill load, and every replica asks the one prefill side. It
/// may share the admission's reservation or be a guard of its own, and
/// it may be made of a replica. A child's failure stands against a
/// sibling's good answer, told ahead of it or behind it, and a sibling
/// that never answered tells nothing. The side ends once, behind the
/// last child's load.
#[tokio::test]
async fn every_replica_of_a_prefills_load_tells_the_one_side() {
    const NEVER_SENT: &str = "released Decode Abandoned";
    // Where the sibling's good answer is told: ahead of the failure,
    // behind it, or never.
    let told = [Some(true), Some(false), None];
    let orders = [false, true].map(|admitted| told.map(|good_first| (admitted, good_first)));
    for (admitted, good_first) in orders.into_iter().flatten() {
        let order = format!("admission {admitted}, a good answer first {good_first:?}");
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let heard = hearing_pair_sides_on(&policies);
        let ends = || {
            let heard = heard.lock().expect("the stage heard");
            let ends = heard
                .iter()
                .filter(|word| word.starts_with("completed") || word.starts_with("released"));
            let mut ends: Vec<String> = ends.cloned().collect();
            ends.sort();
            ends
        };
        let admission = admitted.then(|| PrefillAdmission::new(1, 0, Duration::from_secs(1)));
        let inputs = PlacementInputs::default();
        let ((prefill, decode), ahead) =
            acquired(&registry, &policies, admission.as_ref(), inputs).await;
        drop(decode);

        // The loads are made as a fan-out makes its children's.
        let mut loads = ahead.sent().replicate_to(2);
        let load = loads.pop().expect("the load itself");
        let mut replicas = loads.pop().expect("its replica").replicate_to(2);
        let replica = replicas.pop().expect("the replica itself");
        let of_a_replica = replicas.pop().expect("a replica of the replica");
        // A reservation is counted once, a guard of its own each time.
        let counted = if admitted { 1 } else { 3 };
        assert_eq!(prefill.load(), counted, "{order}");
        if good_first == Some(true) {
            load.end().report(200, || true);
        }
        of_a_replica.end().report(503, || true);
        if good_first == Some(false) {
            load.end().report(200, || true);
        }
        drop(load);
        drop(of_a_replica);
        assert_eq!(
            ends(),
            [NEVER_SENT],
            "no end while a replica stands: {order}"
        );

        drop(replica);
        assert_eq!(
            ends(),
            ["completed Prefill ServerError 503", NEVER_SENT],
            "{order}"
        );
        assert_eq!(prefill.load(), 0, "{order}");
    }
}

/// A replica of a load the classic policies placed keeps nothing, as the
/// load it is made of. Nothing is allocated for it, and the handle asked of
/// either is empty. A replica's handle lies in the slot a worker guard
/// keeps its maker's value in, so the load's type is no wider for it.
#[test]
#[cfg(target_pointer_width = "64")]
fn a_classic_replica_keeps_nothing_and_the_load_is_no_wider() {
    let prefill = worker("grpc://p:1");
    let load = PrefillLoadGuard::Unbounded {
        _guard: WorkerLoadGuard::with_key(Arc::clone(&prefill), None),
    };
    let mut loads = load.replicate_to(2);
    let load = loads.pop().expect("the load itself");
    let replica = loads.pop().expect("its replica");
    assert!(load.end().is_empty() && replica.end().is_empty());
    let PrefillLoadGuard::Unbounded { _guard: guard } = &replica else {
        panic!("a guard of its own");
    };
    assert!(guard.kept().is_none());
    assert_eq!(prefill.load(), 2);

    assert_eq!(size_of::<WorkerLoadGuard>(), 56);
    assert_eq!(size_of::<PrefillLoadGuard>(), 56);
}
