//! The store's publications, paid by one loop of the serve.
//!
//! The store shows its readers what a write left waiting at the end of the
//! window the write fell in, and names that instant
//! ([`LoadSink::publish_due`]). No write pays for a publication itself:
//! this loop keeps the instant and publishes at it ([`LoadSink::publish`]).
//! It runs on every replica for the serve's length, whoever wrote: a
//! followed row reaches the published world as a leader's own does.
//!
//! A membership write of another task can leave the store an instant the
//! loop read none of, a registration's build: the store wakes the loop
//! ([`LoadSink::wake_with`]), which reads the instant again. A write that
//! wakes nobody is read at the loop's next tick.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::{
    sync::{watch, Notify},
    time::MissedTickBehavior,
};

use crate::store::LoadSink;

/// How often the loop reads the store's instant again with no wake: how
/// late a write that wakes nobody is shown, at the most, behind its window.
const TICK: Duration = Duration::from_secs(1);

/// The host's clock through tokio's, so a test under paused time moves
/// every instant the store compares.
fn now() -> Instant {
    tokio::time::Instant::now().into_std()
}

/// Waits for the instant; for ever where there is none.
async fn at(instant: Option<Instant>) {
    match instant {
        Some(instant) => tokio::time::sleep_until(tokio::time::Instant::from_std(instant)).await,
        None => std::future::pending().await,
    }
}

/// Pays `store`'s publications until `stop` reads `true`.
pub(super) async fn run<S: LoadSink + ?Sized>(store: &S, mut stop: watch::Receiver<bool>) {
    let mut ticker = tokio::time::interval(TICK);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let woken = Arc::new(Notify::new());
    store.wake_with(Box::new({
        let woken = Arc::clone(&woken);
        move || woken.notify_one()
    }));
    loop {
        // The instant is read again behind every turn: a publication, a
        // wake, a tick.
        let due = store.publish_due();
        tokio::select! {
            biased;
            _ = stop.wait_for(|stop| *stop) => return,
            () = woken.notified() => {}
            () = at(due) => store.publish(now()),
            _ = ticker.tick() => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::task::JoinSet;

    use super::*;
    use crate::{
        engine::view::{LoadRow, Role, WireModes, WorkerId},
        store::{LoadSource, MemoryStore, WorkerFacts, WorkersStore, PUBLISH_WINDOW},
    };

    /// A store with the loop running over it, and the loop's stop.
    fn served() -> (Arc<MemoryStore>, watch::Sender<bool>, JoinSet<()>) {
        let store = Arc::new(MemoryStore::new());
        let (stop, stopped) = watch::channel(false);
        let mut tasks = JoinSet::new();
        let published = Arc::clone(&store);
        tasks.spawn(async move { run(&*published, stopped).await });
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
    /// behind the loop's start: a load record that lands inside a window
    /// waits for the window's end, the loop sleeps through it, and the tick
    /// behind it publishes.
    #[tokio::test(start_paused = true)]
    async fn a_write_that_wakes_nobody_is_shown_at_the_next_tick() {
        let (store, stop, tasks) = served();
        tokio::time::sleep(Duration::from_millis(1)).await;
        let id = register(&store, "http://a:1");
        store.publish_now();
        let started = now();
        let record = |running: u32| {
            let row = LoadRow {
                running,
                ..LoadRow::default()
            };
            store.record(id, 0, row, now(), LoadSource::Stream);
        };

        // The first record lands behind a quiet window and is shown by its
        // own write. The second lands inside the window it opened.
        record(1);
        tokio::time::sleep(Duration::from_millis(1)).await;
        record(2);
        assert!(store.publish_due().is_some(), "the second record waits");

        // No wake came: the window's end passes and the record still waits.
        tokio::time::sleep(PUBLISH_WINDOW * 2).await;
        assert!(store.publish_due().is_some(), "the loop was woken");
        after(started + Duration::from_secs(1)).await;
        assert_eq!(store.publish_due(), None, "the tick published it");

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
}
