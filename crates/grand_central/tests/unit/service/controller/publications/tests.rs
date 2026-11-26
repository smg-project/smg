use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::task::JoinSet;

use super::*;
use crate::{
    engine::view::{Role, WireModes, WorkerId},
    store::{WorkerFacts, WorkersStore},
};

/// A store with the loop running over it, and the loop's stop.
fn served() -> (Arc<MemoryStore>, watch::Sender<bool>, JoinSet<()>) {
    let store = Arc::new(MemoryStore::new());
    let (stop, stopped) = watch::channel(false);
    let mut tasks = JoinSet::new();
    let published = Arc::clone(&store);
    tasks.spawn(async move { run(&published, stopped).await });
    (store, stop, tasks)
}

/// Registers a worker under `key`.
fn register(store: &MemoryStore, key: &str) -> WorkerId {
    let facts = WorkerFacts {
        key: key.to_owned(),
        engine: key.to_owned(),
        incarnation: 1,
        role: Role::Single,
        wire_modes: WireModes::HTTP,
        models: vec!["m".to_owned()],
        dp_size: 1,
        ..WorkerFacts::default()
    };
    store.upsert(facts).id()
}

/// Waits for the loop's task to end, inside a bound of the paused
/// clock: a loop its stop did not end fails the case and hangs nothing.
async fn ended(mut tasks: JoinSet<()>) {
    let joined = tokio::time::timeout(TICK * 5, tasks.join_next()).await;
    let ended = joined.expect("the stop ends the loop");
    ended
        .expect("the loop's task")
        .expect("it ended without a panic");
}

/// A millisecond behind `instant`, on the paused clock.
async fn after(instant: Instant) {
    let behind = tokio::time::Instant::from_std(instant) + Duration::from_millis(1);
    tokio::time::sleep_until(behind).await;
}

/// A registration of another task leaves its build owed, and the store
/// wakes the loop: the build is paid at the instant the store names,
/// with no later write and well ahead of the loop's next tick. Two
/// registrations inside one window are one build at its end.
#[tokio::test(start_paused = true)]
async fn a_registration_of_another_task_is_built_at_the_instant_the_store_names() {
    let (store, stop, tasks) = served();
    // The loop is up: its first tick is behind it and it reads no
    // instant.
    tokio::time::sleep(Duration::from_millis(1)).await;
    let started = now();

    register(&store, "http://a:1");
    assert_eq!(store.membership_builds(), 0);
    after(store.publish_due().expect("a build is owed")).await;
    assert_eq!((store.membership_builds(), store.publish_due()), (1, None));

    register(&store, "http://b:1");
    register(&store, "http://c:1");
    after(store.publish_due().expect("a build is owed")).await;
    assert_eq!((store.membership_builds(), store.publish_due()), (2, None));
    // The store stamps a write by the system clock, so the time this
    // thread loses ahead of a registration moves the instants the case
    // sleeps to: a bound of milliseconds would count that time.
    assert!(now() < started + TICK / 2, "a tick paid the build");

    stop.send_replace(true);
    ended(tasks).await;
}

/// A write that wakes nobody is shown at the loop's next tick, a second
/// behind the loop's start: a registration leaves its build owed with
/// the store's wake put out, the loop sleeps through the instant the
/// store names, and the tick behind it pays the build. The last wake
/// handed stands: the write calls that one, once, and not the loop's.
#[tokio::test(start_paused = true)]
async fn a_write_that_wakes_nobody_is_shown_at_the_next_tick() {
    let (store, stop, tasks) = served();
    tokio::time::sleep(Duration::from_millis(1)).await;
    let started = now();
    // The loop's wake is replaced by one that wakes nobody and counts
    // its calls.
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    store.wake_with(Box::new(move || {
        counted.fetch_add(1, Ordering::SeqCst);
    }));
    register(&store, "http://a:1");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "the write's one call");
    let due = store.publish_due().expect("a build is owed");

    // No wake came: the instant passes and the build is still owed.
    after(due).await;
    assert_eq!(store.membership_builds(), 0, "the loop was woken");
    after(started.max(due) + TICK).await;
    assert_eq!((store.membership_builds(), store.publish_due()), (1, None));

    stop.send_replace(true);
    ended(tasks).await;
}

/// The stop ends the loop, and nothing is published behind it: a build
/// owed at the stop stays owed.
#[tokio::test(start_paused = true)]
async fn the_stop_ends_the_loop_and_nothing_is_paid_behind_it() {
    let (store, stop, tasks) = served();
    tokio::time::sleep(Duration::from_millis(1)).await;
    stop.send_replace(true);
    ended(tasks).await;

    register(&store, "http://a:1");
    tokio::time::sleep(TICK * 3).await;
    assert_eq!(store.membership_builds(), 0);
    assert!(store.publish_due().is_some(), "the build stays owed");
}
