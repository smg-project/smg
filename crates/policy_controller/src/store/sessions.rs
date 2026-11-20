//! The sessions store: a session key's pin to its target (design 5.1, the
//! sessions row), and the count of the key's requests in flight.

use std::{collections::hash_map::Entry, sync::Arc, time::Instant};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::{
    counts::{Counts, Slot},
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
/// the key stands on file. A keyed decision reads it under a rule that
/// has an in-flight cap or reads the keys each worker carries. The first
/// place stands in the entry itself: a key whose requests all stand on
/// one worker, as a pinned key's do, allocates nothing. A place whose
/// count falls to zero leaves.
///
/// A place is a worker under one slot of the counts. A key with a request
/// at a place is one of that slot's keys ([`Counts::book_key`]), from its
/// first request there to its last: each lease gives back the slot it
/// booked. A worker back under its id stands under the table's next slot
/// and is another place: the leases of the registration that left count
/// on the slot they booked, as their requests do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InFlight {
    /// The first place. A count of zero is no place, and then none stands
    /// behind it.
    first: Carried,
    /// Every other place that carries a request of the key, each with a
    /// count above zero.
    rest: Vec<Carried>,
}

/// One place that carries requests of a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Carried {
    worker: WorkerId,
    /// The slot the place's leases book: [`Slot::NONE`] for leases that
    /// book none.
    slot: Slot,
    /// The key's requests at the place.
    count: u32,
}

impl Carried {
    /// No place.
    const NONE: Self = Self {
        worker: WorkerId(0),
        slot: Slot::NONE,
        count: 0,
    };
}

impl Default for InFlight {
    fn default() -> Self {
        Self {
            first: Carried::NONE,
            rest: Vec::new(),
        }
    }
}

impl InFlight {
    /// Where `worker` stands in the entry, under `slot` where one is
    /// named and under any slot otherwise: `0` is the first place, and
    /// `n` the place before the `n`th of the rest.
    fn place(&self, worker: WorkerId, slot: Option<Slot>) -> Option<usize> {
        let stands = |held: &Carried| {
            held.count != 0 && held.worker == worker && slot.is_none_or(|slot| held.slot == slot)
        };
        if stands(&self.first) {
            return Some(0);
        }
        self.rest.iter().position(stands).map(|at| at + 1)
    }

    /// One request more of the key on `worker`, by a lease that books
    /// `slot`. Whether it is the key's first request at that place: the
    /// key is one more of the slot's keys then.
    fn book(&mut self, worker: WorkerId, slot: Slot) -> bool {
        let held = match self.place(worker, Some(slot)) {
            Some(0) => &mut self.first,
            Some(at) => &mut self.rest[at - 1],
            None => {
                let new = Carried {
                    worker,
                    slot,
                    count: 1,
                };
                if self.first.count == 0 {
                    self.first = new;
                } else {
                    self.rest.push(new);
                }
                return true;
            }
        };
        held.count = held.count.saturating_add(1);
        false
    }

    /// One request of the key on `worker` ended, of a lease that booked
    /// `slot`. The slot whose keys the key leaves, where that was its last
    /// request at the place. A lease whose place is not found gives its
    /// worker's first place back, and a worker that carries none changes
    /// nothing.
    fn free(&mut self, worker: WorkerId, slot: Slot) -> Option<Slot> {
        let exact = self.place(worker, Some(slot));
        debug_assert!(
            exact.is_some() || self.place(worker, None).is_none(),
            "a lease frees the place it booked"
        );
        let at = exact.or_else(|| self.place(worker, None))?;
        let held = match at {
            0 => &mut self.first,
            at => &mut self.rest[at - 1],
        };
        held.count -= 1;
        if held.count != 0 {
            return None;
        }
        let left = held.slot;
        if at == 0 {
            // The last of the others takes the first place.
            self.first = self.rest.pop().unwrap_or(Carried::NONE);
        } else {
            self.rest.swap_remove(at - 1);
        }
        Some(left)
    }

    /// Whether no place carries a request of the key: its entry then
    /// leaves its table.
    fn is_empty(&self) -> bool {
        self.first.count == 0
    }

    /// Adds onto `carried` the workers that carry a request of the key,
    /// each once with its count over its places, in no order.
    fn onto(&self, carried: &mut Vec<(WorkerId, u32)>) {
        let from = carried.len();
        let first = (self.first.count != 0).then_some(&self.first);
        for held in first.into_iter().chain(&self.rest) {
            let mut told = carried[from..].iter_mut();
            match told.find(|(worker, _)| *worker == held.worker) {
                Some((_, count)) => *count = count.saturating_add(held.count),
                None => carried.push((held.worker, held.count)),
            }
        }
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
///
/// A worker's keys are counted in the store's counts, a cell a slot beside
/// its requests ([`Counts::book_key`]): a key is one of a worker's from
/// its first request there to its last, under the hold of the key's shard.
/// A decision reads that count with no lock.
pub(super) struct SessionTables {
    shards: [Mutex<Shard>; SHARDS],
    counts: Arc<Counts>,
}

impl SessionTables {
    /// The tables of a store whose leases book `counts`.
    pub(super) fn over(counts: Arc<Counts>) -> Self {
        Self {
            shards: Default::default(),
            counts,
        }
    }

    /// The shard that holds `key`'s pin of record and its count.
    pub(super) fn shard(&self, key: u64) -> &Mutex<Shard> {
        &self.shards[shard_of(key)]
    }

    /// Every shard, for a pass that walks them one hold after another.
    pub(super) fn shards(&self) -> &[Mutex<Shard>] {
        &self.shards
    }

    /// One request more of `session` stands on `worker`: a lease's side
    /// was laid on file. `slot` is the lease's place in the counts, where
    /// the worker's keys are counted too. No session books nothing.
    pub(super) fn book(&self, session: Session, worker: WorkerId, slot: Option<Slot>) {
        let key = session.get();
        if key == 0 {
            return;
        }
        let slot = slot.unwrap_or(Slot::NONE);
        let mut shard = self.shard(key).lock();
        if shard.in_flight.entry(key).or_default().book(worker, slot) {
            self.counts.book_key(slot);
        }
    }

    /// One request of `session` on `worker` left the file, of a lease
    /// that booked `slot`. The key's entry leaves with its last request.
    pub(super) fn free(&self, session: Session, worker: WorkerId, slot: Option<Slot>) {
        let key = session.get();
        if key == 0 {
            return;
        }
        let slot = slot.unwrap_or(Slot::NONE);
        let mut shard = self.shard(key).lock();
        if let Entry::Occupied(mut held) = shard.in_flight.entry(key) {
            if let Some(left) = held.get_mut().free(worker, slot) {
                self.counts.free_key(left);
            }
            if held.get().is_empty() {
                held.remove();
            }
        }
    }

    /// A request of `session` moved from the worker and slot of `from` to
    /// those of `to`, booked before it is freed under one hold: no reader
    /// finds the key short of it.
    pub(super) fn moved(
        &self,
        session: Session,
        from: (WorkerId, Option<Slot>),
        to: (WorkerId, Option<Slot>),
    ) {
        let key = session.get();
        if key == 0 {
            return;
        }
        let slot = |slot: Option<Slot>| slot.unwrap_or(Slot::NONE);
        let mut shard = self.shard(key).lock();
        let held = shard.in_flight.entry(key).or_default();
        if held.book(to.0, slot(to.1)) {
            self.counts.book_key(slot(to.1));
        }
        if let Some(left) = held.free(from.0, slot(from.1)) {
            self.counts.free_key(left);
        }
        debug_assert!(
            !held.is_empty(),
            "the request stands on the worker it moved to"
        );
    }

    /// What a keyed decision reads of `session` under one hold of its
    /// shard: the key's pin of record, and onto `carried` the workers that
    /// carry its requests in flight, each with its count.
    pub(super) fn read(&self, session: Session, carried: &mut Vec<(WorkerId, u32)>) -> Option<Pin> {
        let key = session.get();
        let shard = self.shard(key).lock();
        if let Some(held) = shard.in_flight.get(&key) {
            held.onto(carried);
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
