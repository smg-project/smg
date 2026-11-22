//! The load monitor's snapshot, as an ask decided in this process reads it.
//!
//! The monitor publishes its snapshot whole, on a channel
//! ([`LoadReceiver`]). A classic policy reads the channel at every request:
//! a read lock on its own slot, the channel's read lock and a clone of the
//! snapshot's `Arc`, three words every request in flight writes. An ask of
//! the policy engine reads it here with none of them. Each calling thread
//! keeps a receiver of its own and the `Arc` of the snapshot it last saw
//! there, and an ask reads one word: the channel's version, which the
//! monitor alone writes. Where the version moved, the thread reads the
//! channel once and keeps the new snapshot.
//!
//! **What a thread keeps** is a pointer to the monitor's own published
//! value, never a copy of a report, and no ask reads it without that look
//! at the version: an ask is lent the snapshot a read of the channel at
//! that instant would give. A feed whose monitor is gone keeps its last
//! snapshot, and an ask is lent it.
//!
//! **What it costs.** A thread that asked holds one snapshot until its next
//! ask: at most one old snapshot for each thread of the runtime that ever
//! asked. A snapshot is 672 bytes a worker that reports one rank, and 320
//! more for each further rank, by the allocator's count of what one
//! publication leaves allocated: a third of a megabyte for 512 workers of
//! one rank each, a megabyte and a half where each reports eight. The
//! monitor publishes a fresh `Arc` at each change and never needs to be a
//! snapshot's only holder, so a thread's keep slows no publication.
//!
//! **A loan stays on its thread** ([`Lent`]): it is taken out of the
//! thread's keep and put back at its drop, so no reference count moves.

use std::{
    cell::Cell,
    marker::PhantomData,
    sync::{Arc, OnceLock},
};

use smg_worker::load::state::{LoadReceiver, LoadSnapshot};

/// The monitor's feed for the asks the engine of this process decides,
/// where its document reads load reports. Empty under any other document,
/// and until the gateway's start has built its monitor: an ask is then
/// lent no snapshot, and its records carry no report.
#[derive(Debug, Default)]
pub(super) struct Loads {
    feed: OnceLock<LoadReceiver>,
}

/// A calling thread's receiver of a feed and the snapshot it last saw there.
struct Seen {
    receiver: LoadReceiver,
    snapshot: Arc<LoadSnapshot>,
}

thread_local! {
    /// What this thread last saw of the feed it last read. An ask takes it
    /// out for as long as it decides ([`Lent`]) and puts it back.
    static SEEN: Cell<Option<Seen>> = const { Cell::new(None) };
}

impl Loads {
    /// Reads `feed` from here on: the monitor's channel, handed once, at
    /// the gateway's start. A second hand-over is ignored.
    pub(super) fn read(&self, feed: LoadReceiver) {
        let _ = self.feed.set(feed);
    }

    /// Whether a feed was handed.
    #[cfg(test)]
    pub(super) fn reads_a_feed(&self) -> bool {
        self.feed.get().is_some()
    }

    /// The feed's current snapshot, lent to one ask of the calling thread.
    ///
    /// The steady state takes no lock and writes nothing another thread
    /// reads: one load of the channel's version. Where the monitor has
    /// published since this thread's last ask, the thread reads the channel
    /// once and keeps the new snapshot.
    ///
    /// An ask made while another loan of this thread stands finds the
    /// thread's keep empty and reads the channel itself: it is lent the
    /// current snapshot all the same, and nothing panics (the case
    /// `an_ask_made_while_a_loan_stands_reads_the_channel_itself`).
    pub(super) fn lend(&self) -> Lent {
        let Some(feed) = self.feed.get() else {
            return Lent(None, PhantomData);
        };
        // The thread's own receiver of this feed, or its first read of it.
        let mine = SEEN.take().filter(|seen| seen.receiver.same_channel(feed));
        let mut seen = mine.unwrap_or_else(|| {
            let mut receiver = feed.clone();
            let snapshot = Arc::clone(&receiver.borrow_and_update());
            Seen { receiver, snapshot }
        });
        // A closed channel holds its last snapshot: it stands.
        if seen.receiver.has_changed().unwrap_or(false) {
            seen.snapshot = Arc::clone(&seen.receiver.borrow_and_update());
        }
        Lent(Some(seen), PhantomData)
    }
}

/// A snapshot lent to one ask: held while the ask decides, and put back
/// into the thread's keep at its drop.
///
/// It cannot leave its thread. A loan that crossed to another thread would
/// be put back into that thread's keep, whose receiver would then answer
/// "seen" for a snapshot it never read.
pub(super) struct Lent(Option<Seen>, PhantomData<*const ()>);

// A loan is not `Send`, held at compile time: were it `Send`, both of the
// two blanket answers below would fit and the line would not compile.
const _: fn() = || {
    trait StaysOnItsThread<Answer> {
        fn held() {}
    }
    impl<T: ?Sized> StaysOnItsThread<()> for T {}
    impl<T: ?Sized + Send> StaysOnItsThread<u8> for T {}
    let _ = <Lent as StaysOnItsThread<_>>::held;
};

impl Lent {
    /// The snapshot, as the monitor published it. `None` where the engine
    /// reads no feed.
    pub(super) fn snapshot(&self) -> Option<&Arc<LoadSnapshot>> {
        self.0.as_ref().map(|seen| &seen.snapshot)
    }
}

impl Drop for Lent {
    fn drop(&mut self) {
        if let Some(seen) = self.0.take() {
            SEEN.set(Some(seen));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use smg_worker::load::state::{one_rank_load, TestFeed};

    use super::*;

    /// A feed `loads` reads from here on, where it reads none yet.
    fn fed(loads: &Loads) -> TestFeed {
        TestFeed::attached_to(|receiver| loads.read(receiver.expect("the feed's receiver")))
    }

    /// Publishes a snapshot that holds one report, of the worker at `url`.
    fn publish(feed: &TestFeed, url: &str) {
        let report = one_rank_load(1, 0.1, 1_000.0);
        feed.publish(&HashMap::from([(url.to_owned(), report)]));
    }

    /// An ask is lent the snapshot the monitor last published: the same
    /// `Arc` at every ask until the next publication, and the new one at
    /// the first ask behind it. A feed handed later is not read in the
    /// first one's place.
    #[test]
    fn an_ask_is_lent_the_snapshot_last_published() {
        let loads = Loads::default();
        assert!(loads.lend().snapshot().is_none(), "no feed is read yet");
        let feed = fed(&loads);
        publish(&feed, "http://worker-0:8000");
        let first = Arc::clone(loads.lend().snapshot().expect("a snapshot"));
        assert!(first.get("http://worker-0:8000").is_some());
        for _ in 0..3 {
            let lent = loads.lend();
            assert!(Arc::ptr_eq(lent.snapshot().expect("a snapshot"), &first));
        }
        publish(&feed, "http://worker-1:8000");
        let second = Arc::clone(loads.lend().snapshot().expect("a snapshot"));
        assert!(!Arc::ptr_eq(&second, &first));
        assert!(second.get("http://worker-1:8000").is_some());

        let later = fed(&loads);
        publish(&later, "http://worker-2:8000");
        let lent = loads.lend();
        assert!(Arc::ptr_eq(lent.snapshot().expect("a snapshot"), &second));
    }

    /// Two engines' feeds read on one thread do not answer for each other:
    /// each ask is lent its own feed's snapshot.
    #[test]
    fn a_thread_that_reads_two_feeds_is_lent_each_ones_own() {
        let (one, other) = (Loads::default(), Loads::default());
        let (feed_one, feed_other) = (fed(&one), fed(&other));
        publish(&feed_one, "http://one:8000");
        publish(&feed_other, "http://other:8000");
        for _ in 0..2 {
            let lent = one.lend();
            let snapshot = lent.snapshot().expect("a snapshot");
            assert!(snapshot.get("http://one:8000").is_some());
            drop(lent);
            let lent = other.lend();
            let snapshot = lent.snapshot().expect("a snapshot");
            assert!(snapshot.get("http://other:8000").is_some());
        }
    }

    /// A feed whose monitor is gone holds its last snapshot, and an ask is
    /// lent it.
    #[test]
    fn a_feed_whose_monitor_is_gone_lends_its_last_snapshot() {
        let loads = Loads::default();
        let feed = fed(&loads);
        publish(&feed, "http://worker-0:8000");
        let first = Arc::clone(loads.lend().snapshot().expect("a snapshot"));
        drop(feed);
        let lent = loads.lend();
        assert!(Arc::ptr_eq(lent.snapshot().expect("a snapshot"), &first));
    }

    /// An ask made while another loan of its thread stands finds the
    /// thread's keep empty: it reads the channel itself and is lent the
    /// current snapshot. Both loans go back in turn, and the thread's next
    /// ask is lent the snapshot again.
    #[test]
    fn an_ask_made_while_a_loan_stands_reads_the_channel_itself() {
        let loads = Loads::default();
        let feed = fed(&loads);
        publish(&feed, "http://worker-0:8000");
        let outer = loads.lend();
        let inner = loads.lend();
        let (held, again) = (
            outer.snapshot().expect("a snapshot"),
            inner.snapshot().expect("a snapshot"),
        );
        assert!(Arc::ptr_eq(held, again));
        drop(inner);
        drop(outer);
        let next = loads.lend();
        assert!(next
            .snapshot()
            .expect("a snapshot")
            .get("http://worker-0:8000")
            .is_some());
    }

    /// A snapshot published while an ask holds its loan is the next ask's:
    /// the ask that holds one keeps the snapshot it was lent to its end.
    #[test]
    fn a_loan_stands_while_its_ask_decides() {
        let loads = Loads::default();
        let feed = fed(&loads);
        publish(&feed, "http://worker-0:8000");
        let lent = loads.lend();
        let held = Arc::clone(lent.snapshot().expect("a snapshot"));
        publish(&feed, "http://worker-1:8000");
        assert!(Arc::ptr_eq(lent.snapshot().expect("a snapshot"), &held));
        drop(lent);
        let next = loads.lend();
        assert!(next
            .snapshot()
            .expect("a snapshot")
            .get("http://worker-1:8000")
            .is_some());
    }
}
