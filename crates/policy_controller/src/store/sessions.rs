//! The sessions store: the count of a session key's requests in flight.

use std::collections::hash_map::Entry;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::engine::{
    sticky_sessions::{shard_of, SHARDS},
    view::{Session, WorkerId},
};

/// A session key's requests in flight: the workers that carry them, and
/// how many each carries. The store keeps one for a key while a lease of
/// the key stands on file. No decision reads it: a controller serves no
/// in-flight cap yet. The first worker stands in the entry itself: a
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
}

/// One shard of a store's sessions' table.
#[derive(Default)]
struct Shard {
    /// The requests in flight of each key that has a lease on file.
    in_flight: FxHashMap<u64, InFlight>,
}

/// A store's count of each session key's requests in flight, in the shards
/// of the engine's own table ([`shard_of`]).
///
/// A lease's file writes under its own turn: the turn first, then the
/// key's shard. Nothing is taken under a shard.
#[derive(Default)]
pub(super) struct SessionTables {
    shards: [Mutex<Shard>; SHARDS],
}

impl SessionTables {
    /// The shard that holds `key`'s count.
    fn shard(&self, key: u64) -> &Mutex<Shard> {
        &self.shards[shard_of(key)]
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
}
