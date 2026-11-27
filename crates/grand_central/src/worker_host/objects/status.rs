//! The holder's status side: a worker object's status leads its row's.
//!
//! On a replica that leads, a status is written on the worker object and
//! the store is told it. A probe's verdict moves the object by the
//! registry's own state machine, and the health adapter then tells the
//! store the status that stands ([`Objects::tell_status`]). A connect
//! signal and a drain are decided here, over the object
//! ([`Objects::connect`], [`Objects::drain_row`]): the store is told
//! first, and the registry moves the object where the store took the
//! status.
//!
//! **The lock order.** Every entry here runs under the object's mutation
//! lock, the registry's lock of one worker, and takes the store's
//! membership lock inside it: the object's lock, then the store's. No
//! holder of the store's lock takes an object's. So the read of an
//! object's status and its telling are one critical section. Of two tells
//! the later read stands, and the table ends where the object does. No
//! decision takes an object's mutation lock. Nor does a first copy's cut
//! ([`mod@super::read`]): it holds the holder's turn and, inside it, the
//! store's lock. A holder's write of a row holds that turn, then an
//! object's lock, then the store's: one order for the three.
//!
//! A tell hands a store that tells a journal the models its object serves,
//! read in that same critical section and ahead of the store's lock: the
//! log's entry of a status names them and no row. Of the table's row a
//! leader's status entry holds one thing alone, the status the row left,
//! which is the log's own word. A store with no journal is handed none,
//! and no list is built for it.
//!
//! A tell the store refuses writes nothing, and the call does not try it
//! again: the row left the revision the object was built from, no row
//! stands, or the writer's term ended. A writer that names its term hands
//! it on with its word ([`LeadWord`]): the store's journal compares it
//! where the status's event takes its place, so a term that ended by then
//! writes nothing either. The row's next lay or the object's next move
//! settles it. A status the object took ahead of a tell the
//! store takes no word of is taken back in the tell's own turn: the
//! object takes the status its row holds, and where no row stands for it
//! the status it left.
//!
//! From the table to an object there are two ways, both of a told status.
//! A standby's applier and a gateway's watcher lay the status their
//! leader's log tells ([`mod@super::standby`]). And a tell the store
//! refuses gives the object the status its row holds, as said above.

use std::sync::Arc;

use smg_worker::{
    identity::basic::Worker,
    registry::{Generation, WorkerId as ObjectId},
    worker::WorkerStatus as ObjectStatus,
};

use super::{super::object::row_status, Objects};
use crate::{
    engine::view::WorkerId,
    store::{HealthWrite, LeadWord, MemoryStore},
};

/// An object's move: the status it left, then the one that stands.
type Moved = Option<(ObjectStatus, ObjectStatus)>;

/// The ids of the models `object` serves, in its cards' order, as the one
/// shared list a status entry holds. Each id is copied, and no card. A
/// status write of this module calls it under the object's mutation
/// lock. A standby's applier calls it in the holder's turn, with no lock
/// of the object.
pub(super) fn served(object: &dyn Worker) -> Arc<[String]> {
    let mut ids = Vec::new();
    object.any_model(&mut |card| {
        ids.push(card.id.clone());
        false
    });
    ids.into()
}

/// What a status write does to its object, read under the object's lock.
#[derive(Clone, Copy)]
enum Write {
    /// Nothing: the status that stands is told. With it, the move the
    /// telling follows where a driver made one ahead of it: the status
    /// the object left, then the one it took.
    Tell(Moved),
    /// Ready at once, with its probes counted anew. A ready or a draining
    /// object stands, and its counters with it.
    Connect,
    /// Draining. A draining object stands.
    Drain,
}

impl Write {
    /// The status an object that stands in `from` moves to, and none where
    /// it stands.
    fn next(self, from: ObjectStatus) -> Option<ObjectStatus> {
        let to = match self {
            Self::Tell(_) => return None,
            Self::Connect if from == ObjectStatus::Draining => return None,
            Self::Connect => ObjectStatus::Ready,
            Self::Drain => ObjectStatus::Draining,
        };
        (to != from).then_some(to)
    }
}

impl<G: Generation> Objects<G> {
    /// Tell `store` the status that stands on the object under `id`, where
    /// the object stands at `revision`, its own. A health driver calls it
    /// behind a move the registry's state machine made, `moved`: the
    /// status the object left, then the one it took. `leads` is the
    /// driver's last word, asked under the object's lock and again under
    /// the store's. A driver's word names no term to the store's journal.
    /// Where the store takes no word, the move is taken back in this turn
    /// ([`Self::status_under`]). What the store made of the status.
    pub(crate) fn tell_status(
        &self,
        store: &MemoryStore,
        id: &ObjectId,
        revision: u64,
        moved: (ObjectStatus, ObjectStatus),
        leads: &dyn Fn() -> bool,
    ) -> HealthWrite {
        let write = Write::Tell(Some(moved));
        let word = LeadWord::of(leads);
        self.status_under(store, id, revision, None, word, write).0
    }

    /// The connect signal of the object under `id`, as
    /// [`Self::tell_status`] names one: ready at once, with its probes
    /// counted anew. A ready or a draining object stands, and its counters
    /// with it. The object's move, where it made one.
    pub(crate) fn connect(
        &self,
        store: &MemoryStore,
        id: &ObjectId,
        revision: u64,
        leads: &dyn Fn() -> bool,
    ) -> Moved {
        let word = LeadWord::of(leads);
        self.status_under(store, id, revision, None, word, Write::Connect)
            .1
    }

    /// The connect signal of the row `row`, read at `revision`: its object
    /// is ready at once, with its probes counted anew, and `store` is told.
    /// A ready or a draining object stands, and its counters with it. A row
    /// that left the revision, or left, moves nothing. One of a host's
    /// write calls ([`Objects`]): its caller asks its term itself, so the
    /// call hands the store no last word and no term.
    pub fn connect_row(&self, store: &MemoryStore, row: WorkerId, revision: u64) {
        self.connect_row_under(store, row, revision, LeadWord::FREE);
    }

    /// [`Self::connect_row`] under its writer's word of its lead, `word`,
    /// for a caller whose term can end while the write waits. Its read is
    /// asked under the object's lock and again under the store's, as
    /// [`Self::tell_status`] asks a driver's, and its term goes to the
    /// store's journal with the status. Where any of the three says no,
    /// no status is written and the object stands.
    pub(crate) fn connect_row_under(
        &self,
        store: &MemoryStore,
        row: WorkerId,
        revision: u64,
        word: LeadWord<'_>,
    ) {
        self.row_status(store, row, revision, word, Write::Connect);
    }

    /// The drain of the row `row`, read at `revision`, as
    /// [`Self::connect_row`] writes a status: its object drains, and
    /// `store` is told. Whether the row's object moved: a row that left
    /// the revision, or left, moves nothing, and a draining object stands.
    pub fn drain_row(&self, store: &MemoryStore, row: WorkerId, revision: u64) -> bool {
        self.drain_row_under(store, row, revision, LeadWord::FREE)
    }

    /// [`Self::drain_row`] under its writer's word of its lead, as
    /// [`Self::connect_row_under`] takes one. A drain whose word says no
    /// moves nothing, as one whose row left the revision: its caller asks
    /// its term to tell the two apart.
    pub(crate) fn drain_row_under(
        &self,
        store: &MemoryStore,
        row: WorkerId,
        revision: u64,
        word: LeadWord<'_>,
    ) -> bool {
        self.row_status(store, row, revision, word, Write::Drain)
            .is_some()
    }

    /// Tell `store` the status of the object that stands for `row`, behind
    /// a write that laid the object in another's place: it took the
    /// standing object's status there, and the table holds its own.
    pub(super) fn tell_row(&self, store: &MemoryStore, row: WorkerId, revision: u64) {
        self.row_status(store, row, revision, LeadWord::FREE, Write::Tell(None));
    }

    /// Settle the status of the object that stands for `row` with the
    /// table's, behind a lay outside a write call that put the object in
    /// another's place: it took the standing object's status there, and a
    /// move a health driver made on that one ahead of its telling tells
    /// nothing ([`Self::status_under`]). `leads` is the caller's own word
    /// of its lead, and it names no term. Where it holds, the table is
    /// told the object's status, as [`Self::tell_row`] tells it. Where it
    /// does not, the table is not written, and the object takes the status
    /// its row holds.
    pub(super) fn settle_row(
        &self,
        store: &MemoryStore,
        row: WorkerId,
        revision: u64,
        leads: &dyn Fn() -> bool,
    ) {
        let word = LeadWord::of(leads);
        self.row_status(store, row, revision, word, Write::Tell(None));
    }

    /// A status write that names its row: on the object that stands for
    /// `row`, where it was built from `revision`, under its writer's word
    /// of its lead.
    fn row_status(
        &self,
        store: &MemoryStore,
        row: WorkerId,
        revision: u64,
        word: LeadWord<'_>,
        write: Write,
    ) -> Moved {
        let id = self.object(row)?;
        let object = self.registry.get(&id)?;
        self.status_under(store, &id, object.revision(), Some(revision), word, write)
            .1
    }

    /// One status write, under the object's mutation lock: the writer's
    /// word, the row the holder built this object from, the store's
    /// answer, and the object's move where the store took the status or
    /// holds it already. `seen` is the row's revision where the caller
    /// read one. What the store made of the status, and the object's move.
    ///
    /// A tell the store takes no word of takes the object's move back, in
    /// this same turn, whether the term ended at the turn's start or at
    /// the store's lock. The registry's state machine moved the object
    /// ahead of the telling. Where a row stands at the revision the object
    /// was built from, the object takes the status that row holds, by the
    /// read a told status is laid by ([`Self::table_status`]): a leader's
    /// told status can have landed since the move, so the table's word
    /// stands, never the status the driver saw the object leave. Where no
    /// such row stands (the row left the revision, the row left, or the
    /// holder laid no such object), the object takes back the status its
    /// driver saw it leave, where it stands in the moved one still. A told
    /// status is laid only on an object whose row stands at that revision,
    /// so none is written over there. And a move left on the object would
    /// outlive it: an object laid in its place takes the status it stands
    /// in, with no word to the table.
    ///
    /// A tell that finds the object replaced or gone takes nothing back:
    /// the registry answers none under the revision the driver names, and
    /// the object laid in its place took the status the standing one
    /// stood in, the move with it. What laid it settles the two. A door
    /// tells the store its object's status behind the put
    /// ([`Self::tell_row`]). A told row's put and a cut are followed by
    /// the table's status ([`mod@super::standby`]).
    fn status_under(
        &self,
        store: &MemoryStore,
        id: &ObjectId,
        revision: u64,
        seen: Option<u64>,
        word: LeadWord<'_>,
        write: Write,
    ) -> (HealthWrite, Moved) {
        let under = |object: &Arc<dyn Worker>| {
            let from = object.status();
            // What a driver's move goes back to where no row stands for
            // the object: the status it left, where the object stands in
            // the one it took still.
            let left = match write {
                Write::Tell(Some((left, took))) if took == from => Some(left),
                _ => None,
            };
            let standing = self.standing_of(id, object);
            let read = |revision: u64| seen.is_none_or(|seen| seen == revision);
            let Some(standing) = standing.filter(|standing| read(standing.revision)) else {
                return (HealthWrite::Stale, left);
            };
            let next = write.next(from);
            let status = row_status(next.unwrap_or(from));
            // A call can wait here for the object's lock through the end of
            // its term: it tells nothing.
            let told = if (word.leads)() {
                // The models the object serves at this instant, read under
                // its lock and handed ahead of the store's: the log's entry
                // of a status names them, and no row of the table. A store
                // that tells no entry is handed none, and no list is built.
                let models = store.journals().then(|| served(object.as_ref()));
                store.status_told(standing.row, status, standing.revision, word, models)
            } else {
                HealthWrite::Stale
            };
            if told.stands().is_none() {
                // The store took no word. A status the object took ahead
                // of its telling goes back in this turn: to the row's,
                // and where no row stands to the one it left.
                let back = match write {
                    Write::Tell(_) => Self::table_status(store, standing).or(left),
                    Write::Connect | Write::Drain => None,
                };
                return (told, back);
            }
            if next.is_some() && matches!(write, Write::Connect) {
                if let Some(state) = object.runtime_state() {
                    state.consecutive_failures_reset();
                    state.consecutive_successes_reset();
                    state.total_pending_probes_reset();
                }
            }
            (told, next)
        };
        self.registry
            .apply_if_revision(id, revision, under)
            .unwrap_or((HealthWrite::Stale, None))
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/worker_host/objects/status/tests.rs"]
pub(crate) mod tests;
