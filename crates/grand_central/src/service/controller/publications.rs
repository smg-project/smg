//! The store's publications, paid by one loop of the serve.
//!
//! The store shows its readers what a write left waiting at the end of the
//! window the write fell in, and names that instant
//! ([`MemoryStore::publish_due`]). No write pays for a publication itself:
//! this loop keeps the instant and publishes at it ([`MemoryStore::publish`]).
//! It runs on every replica for the serve's length, whoever wrote: a
//! followed member reaches the published world as a leader's own does.
//!
//! A membership write of another task can leave the store an instant the
//! loop read none of, a registration's build: the store calls the wake the
//! loop handed it ([`MemoryStore::wake_with`], a
//! [`Wake`](crate::store::Wake)), and the loop reads the instant again. A
//! write that wakes nobody is read at the loop's next tick.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::{
    sync::{watch, Notify},
    time::MissedTickBehavior,
};

use crate::store::MemoryStore;

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
pub(super) async fn run(store: &MemoryStore, mut stop: watch::Receiver<bool>) {
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
#[path = "../../../tests/unit/service/controller/publications/tests.rs"]
mod tests;
