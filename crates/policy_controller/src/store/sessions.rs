//! The sessions store: a session key's pin to its target (design 5.1, the
//! sessions row), and the count of the key's requests in flight.

use std::{collections::hash_map::Entry, time::Instant};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::{
    sessions::{shard_of, SHARDS},
    view::{Session, WorkerId},
};

/// A session's pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pin {
    pub worker: WorkerId,
    /// The last decision that used it, on the host's clock: a merge reads
    /// it.
    pub last_used: Instant,
    /// The instant the pin ends, by the TTL of the rule that placed its
    /// key: at or past it the sweep drops the pin and a decision's read
    /// takes none. It stands on the decision clock, in microseconds since
    /// the store's origin, as the engine's table holds the instant it
    /// forgets the key: a dispatch writes it as read, or as its lease's
    /// own rule counts it from the lease's issue, never earlier than the
    /// end that stood, and converts no clock. `None` for a key its rule
    /// keeps with no TTL: such a pin stays until a dispatch under a table
    /// moves it and its end, or its worker's row goes.
    pub ends: Option<u64>,
}

/// Where session pins are written: at the commit of a keyed decision, never
/// at selection. Keys are fingerprints (never `0`). A worker that leaves
/// keeps its pins while its key may be registered again, and takes them with
/// it where it is not. By the design the gateways keep a bounded table of the pins
/// they received and replay it in `Hello`, merged here by last use: no
/// gateway replays its pins in this build, and the merge door has no caller.
pub trait SessionsStore: Send + Sync {
    /// Pin `key` to `worker` at `at`, to end at `ends` on the decision
    /// clock (`None`: at no instant); a re-pin moves it and its end.
    fn pin(&self, key: u64, worker: WorkerId, at: Instant, ends: Option<u64>);

    /// Pin `key` as [`pin`](Self::pin) does, for a writer that knows the
    /// key's rule by its lease alone: the end that stands stays where it is
    /// the later one, and no end is later than any instant. A lease
    /// dispatched behind a newer one of its key, or behind a dispatch whose
    /// table ended the pin later, then never ends the pin earlier.
    fn pin_no_earlier(&self, key: u64, worker: WorkerId, at: Instant, ends: Option<u64>);

    fn lookup(&self, key: u64) -> Option<Pin>;

    fn unpin(&self, key: u64);

    /// Workers left for good: their pins go with them, in one walk.
    /// Returns how many.
    fn drop_workers(&self, workers: &[WorkerId]) -> usize;

    /// Drop the pins whose own end is at or before `now`, an instant of
    /// the decision clock. Returns how many.
    fn expire(&self, now: u64) -> usize;

    /// Merge a pin a gateway replayed: a key already pinned keeps the later
    /// use. The pin's end is a number of the clock of the store that wrote
    /// it: a caller turns it into this store's before it merges.
    fn merge(&self, key: u64, pin: Pin);

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A session key's requests in flight: the workers that carry them, and
/// how many each carries. The store keeps one for a key while a lease of
/// the key stands on file, and a keyed decision under a rule with an
/// in-flight cap reads it. The first worker stands in the entry itself: a
/// key whose requests all stand on one worker, as a pinned key's do,
/// allocates nothing. A worker whose count falls to zero leaves.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct InFlight {
    /// The first worker and its count. A count of zero is no worker, and
    /// then none stands behind it.
    first: (WorkerId, u32),
    /// Every other worker that carries a request of the key, each with a
    /// count above zero.
    rest: Vec<(WorkerId, u32)>,
}

impl InFlight {
    /// One request more of the key on `worker`.
    fn book(&mut self, worker: WorkerId) {
        if self.first.1 == 0 {
            self.first = (worker, 1);
        } else if self.first.0 == worker {
            self.first.1 = self.first.1.saturating_add(1);
        } else if let Some((_, count)) = self.rest.iter_mut().find(|(id, _)| *id == worker) {
            *count = count.saturating_add(1);
        } else {
            self.rest.push((worker, 1));
        }
    }

    /// One request of the key on `worker` ended. `true` once no worker
    /// carries one: the key's entry then leaves its table. A worker that
    /// carries none changes nothing.
    fn free(&mut self, worker: WorkerId) -> bool {
        if self.first.0 == worker {
            self.first.1 = self.first.1.saturating_sub(1);
            if self.first.1 == 0 {
                // The last of the others takes the first place.
                self.first = self.rest.pop().unwrap_or_default();
            }
        } else if let Some(at) = self.rest.iter().position(|(id, _)| *id == worker) {
            self.rest[at].1 -= 1;
            if self.rest[at].1 == 0 {
                self.rest.swap_remove(at);
            }
        }
        self.first.1 == 0
    }

    /// The workers that carry a request of the key, each with its count,
    /// in no order.
    fn workers(&self) -> impl Iterator<Item = (WorkerId, u32)> + '_ {
        let first = (self.first.1 != 0).then_some(self.first);
        first.into_iter().chain(self.rest.iter().copied())
    }
}

/// One shard of a store's sessions' tables.
#[derive(Default)]
pub(super) struct Shard {
    /// The pins of record.
    pub(super) pins: FxHashMap<u64, Pin>,
    /// The requests in flight of each key that has a lease on file.
    in_flight: FxHashMap<u64, InFlight>,
}

/// A store's sessions' tables, in the shards of the engine's own table
/// ([`shard_of`]): each key's pin of record and, beside it, the key's
/// requests in flight. One lock a shard holds both, so a keyed decision
/// reads its key's pin and its count under one hold
/// ([`SessionTables::read`]).
///
/// A lease's file writes under its own turn: the turn first, then the
/// key's shard. A registration that takes its key's pins along renames
/// them under the workers lock: that lock first, then each shard. Every
/// other caller takes a shard with no lock of the store's held, and
/// nothing is taken under one, so no path takes two of them the other
/// way. The engine's own table is told each leave and join under the
/// workers lock too, and takes its absent list's lock, then a shard of
/// its own ([`crate::sessions::SessionTable`]).
#[derive(Default)]
pub(super) struct SessionTables {
    shards: [Mutex<Shard>; SHARDS],
}

impl SessionTables {
    /// The shard that holds `key`'s pin of record and its count.
    pub(super) fn shard(&self, key: u64) -> &Mutex<Shard> {
        &self.shards[shard_of(key)]
    }

    /// Every shard, for a pass that walks them one hold after another.
    pub(super) fn shards(&self) -> &[Mutex<Shard>] {
        &self.shards
    }

    /// One request more of `session` stands on `worker`: a lease's side
    /// was laid on file. No session books nothing.
    pub(super) fn book(&self, session: Session, worker: WorkerId) {
        let key = session.get();
        if key == 0 {
            return;
        }
        let mut shard = self.shard(key).lock();
        shard.in_flight.entry(key).or_default().book(worker);
    }

    /// One request of `session` on `worker` left the file. The key's entry
    /// leaves with its last request.
    pub(super) fn free(&self, session: Session, worker: WorkerId) {
        let key = session.get();
        if key == 0 {
            return;
        }
        let mut shard = self.shard(key).lock();
        if let Entry::Occupied(mut held) = shard.in_flight.entry(key) {
            if held.get_mut().free(worker) {
                held.remove();
            }
        }
    }

    /// A request of `session` moved from `from` to `to`, booked before it
    /// is freed under one hold: no reader finds the key short of it.
    pub(super) fn moved(&self, session: Session, from: WorkerId, to: WorkerId) {
        let key = session.get();
        if key == 0 {
            return;
        }
        let mut shard = self.shard(key).lock();
        let held = shard.in_flight.entry(key).or_default();
        held.book(to);
        let emptied = held.free(from);
        debug_assert!(!emptied, "the request stands on the worker it moved to");
    }

    /// What a keyed decision reads of `session` under one hold of its
    /// shard: the key's pin of record, and onto `carried` the workers that
    /// carry its requests in flight, each with its count.
    pub(super) fn read(&self, session: Session, carried: &mut Vec<(WorkerId, u32)>) -> Option<Pin> {
        let key = session.get();
        let shard = self.shard(key).lock();
        if let Some(held) = shard.in_flight.get(&key) {
            carried.extend(held.workers());
        }
        shard.pins.get(&key).copied()
    }

    /// How many keys have a request in flight: no door tells it.
    #[cfg(test)]
    pub(super) fn keys_in_flight(&self) -> usize {
        let shards = self.shards.iter();
        shards.map(|shard| shard.lock().in_flight.len()).sum()
    }
}
