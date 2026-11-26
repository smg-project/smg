//! The holder's write side: a leader's write of a row is decided over the
//! worker objects, and the store follows it.
//!
//! A door hands the holder the write it would have asked of the store's
//! table: a registration's facts, a patch, or a removal. The holder takes
//! its turn and decides over the row read back
//! from the object that stands ([`Objects::row_of`]), by the rule every
//! writer of a row decides by. It builds the object of the row the change
//! leaves, tells the store the decided change
//! ([`MemoryStore::follow_objects`]), and puts the object in the registry.
//! So the object stands when the write returns.
//! No row of the store's table is read at the head of a write: every
//! writer of a row holds the turn, so the object states the row.
//!
//! The store is told between the build and the registry's write: its write
//! files the member's count slot, which a lease books the moment a decision
//! picks the worker off the registry. A removal goes the other way round:
//! the object leaves the registry, then the store is told. A build cannot
//! fail, so the store is never told a change whose object was not built.
//!
//! A write the store does not take answers `None`: another writer moved
//! the row under the holder's turn. The store tests that under its own
//! lock, by the revision the holder read: a door and a standby's applier
//! write in the holder's turn, and a caller of the store's own writes
//! takes none.
//! The holder then reads the row that stands under the write's key in the
//! store, and that row alone, before it answers: the registry holds what
//! that writer left under the key. A removal has taken its object out by
//! then, and the row that writer left gets an object anew.
//! A change the store took and the registry did not get, because the
//! writer's thread ended between the two, is laid by that read where the
//! store refuses the key's next registration.
//!
//! A patch tells the store the whole row read back from the object, where
//! the store's own patch wrote the patched facts alone. The two are one
//! row while an object states what its row stated: what an object does not
//! hold as stated is kept beside it, and nothing in a controller sets a
//! fact of an object behind its build, as a gateway's refresh of a
//! worker's models does in a gateway. A change that sets one writes the
//! table at the row's next patch.
//!
//! A status is the object's word ([`mod@super::status`]). A row changed in
//! place keeps the status its table holds, and the holder tells the store
//! the status of the object laid in the standing one's place. A key's next
//! life starts in the status the rule hands on from the object that stood,
//! and its object is built in it.

use std::time::Instant;

use smg_worker::registry::Generation;

use super::Objects;
use crate::{
    engine::view::WorkerId,
    store::{
        MembershipChange, MemoryStore, Registrant, WorkerFacts, WorkerPatch, WorkerRow,
        WorkersStore,
    },
};

impl<G: Generation> Objects<G> {
    /// Lay a registration's `facts`, by `maker`, over the object that
    /// stands under their key, by the one rule every writer of a row
    /// decides by. The change it made, or `None` where `store` did not take
    /// it: another writer moved the row under the call.
    ///
    /// A row changed in place gets a new object in the standing one's
    /// place, under its id in the registry and with its running state. A
    /// key's next life registers its object under a new id, and the old
    /// life's object is dropped first.
    ///
    /// One of a host's write calls ([`Objects`]).
    pub fn upsert(
        &self,
        store: &MemoryStore,
        facts: WorkerFacts,
        maker: Registrant,
    ) -> Option<MembershipChange> {
        let _turn = self.turn.lock();
        let standing = self.row_under(&facts.key);
        let seen = standing.as_ref().map(|row| row.revision);
        let (change, row) = facts.laid_over(standing.as_ref(), maker, Instant::now());
        self.lay_decided(store, change, &row, seen)
    }

    /// Write `patch` over the row `id` where its object was built at
    /// `revision`. The change it made, or `None` where the row moved or
    /// left, or `store` did not take the write.
    ///
    /// One of a host's write calls ([`Objects`]).
    pub fn patch(
        &self,
        store: &MemoryStore,
        id: WorkerId,
        patch: WorkerPatch,
        revision: u64,
    ) -> Option<MembershipChange> {
        let _turn = self.turn.lock();
        let mut row = self.guarded(id, revision)?;
        patch.laid_on(&mut row.facts);
        row.revision += 1;
        let change = MembershipChange::Relabelled { id };
        self.lay_decided(store, change, &row, Some(revision))
    }

    /// Remove the row `id` where its object was built at `revision`. The
    /// object leaves the registry, then `store` is told. The change it
    /// made, or `None` where the row moved or left, or `store` did not take
    /// the write.
    ///
    /// One of a host's write calls ([`Objects`]).
    pub fn remove(
        &self,
        store: &MemoryStore,
        id: WorkerId,
        revision: u64,
    ) -> Option<MembershipChange> {
        let _turn = self.turn.lock();
        let row = self.guarded(id, revision)?;
        self.take_decided(store, &row)
    }

    /// The row `id` as its object and what stands beside it state it,
    /// where the object was built at `revision`.
    fn guarded(&self, id: WorkerId, revision: u64) -> Option<WorkerRow> {
        let row = self.row(id)?;
        (row.revision == revision).then_some(row)
    }

    /// The decided `change` and the `row` it leaves, over a row read back
    /// at the revision `seen` in the holder's turn: the row's object is
    /// built, the store is told, and the object is put in the registry. A
    /// change the store does not take builds an object nobody sees
    /// ([`Self::moved`]).
    fn lay_decided(
        &self,
        store: &MemoryStore,
        change: MembershipChange,
        row: &WorkerRow,
        seen: Option<u64>,
    ) -> Option<MembershipChange> {
        let built = self.build(row);
        if !store.follow_objects(change, Some(row.clone()), seen) {
            return self.moved(store, &row.facts.key, change);
        }
        self.put(store, row, built);
        // The old life left with this write: the keeper forgets its row.
        if let MembershipChange::Replaced { old, .. } = change {
            self.forget_one(self.keeper_clock(), old);
        }
        // The object's status leads. One laid in another's place took the
        // standing object's there, and the table is told it. A new object
        // is built in the status its row starts in.
        if matches!(
            change,
            MembershipChange::Relabelled { .. } | MembershipChange::Resized { .. }
        ) {
            self.tell_row(store, row.id, row.revision);
        }
        Some(change)
    }

    /// The decided removal of `row`, read back in the holder's turn: its
    /// object leaves the registry, then the store is told. Where the store
    /// does not take it, the row that stands gets an object anew
    /// ([`Self::moved`]): a new object under a new id in the registry. The
    /// leases the store counts on the row are stated on it, and what the
    /// old object alone held is not.
    fn take_decided(&self, store: &MemoryStore, row: &WorkerRow) -> Option<MembershipChange> {
        self.end(row.id);
        let change = MembershipChange::Removed { id: row.id };
        if !store.follow_objects(change, None, Some(row.revision)) {
            return self.moved(store, &row.facts.key, change);
        }
        self.forget_one(self.keeper_clock(), row.id);
        Some(change)
    }

    /// The store did not take a write of the row under `key`: a writer that
    /// took no turn moved the row, or the registry never got a change the
    /// store took. The write's own record is read, and no other: the row
    /// that stands under the key in the store. A patch and a removal name
    /// their row by its id, and a key holds one row: the read finds the
    /// row under that id where it stands still, and the key's next life
    /// where it left. The row is laid unless its object was built from it,
    /// and an object of another life of the key gives way to it
    /// ([`Self::put`]). A row that left and came back under its id is
    /// another life too, made by another or admitted at another time: its
    /// old object is ended first, so
    /// the new one takes none of its running state. Where no row stands,
    /// the key's object goes. So the
    /// registry holds what stands under the key when the write answers. No
    /// other read of the store's rows is left in a write.
    ///
    /// `change` is the change the store refused. The keeper forgets the
    /// row it was decided over, where the read found that row gone, and
    /// asks about no other: the read saw one record.
    fn moved(
        &self,
        store: &MemoryStore,
        key: &str,
        change: MembershipChange,
    ) -> Option<MembershipChange> {
        let read = self.keeper_clock();
        match store.by_key(key) {
            Some(row) => {
                // Another life under the id the object stands for.
                if self.remade(&row) {
                    self.end(row.id);
                }
                if !self.stands(&row) {
                    self.lay(store, &row);
                }
            }
            None => {
                if let Some(stood) = self.row_under(key) {
                    self.end(stood.id);
                }
            }
        }
        let over = match change {
            MembershipChange::Replaced { old, .. } => Some(old),
            MembershipChange::Relabelled { id }
            | MembershipChange::Resized { id }
            | MembershipChange::Removed { id } => Some(id),
            MembershipChange::Registered { .. } | MembershipChange::StatusChanged { .. } => None,
        };
        if let Some(over) = over {
            self.forget_one(read, over);
        }
        None
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/worker_host/objects/write/tests.rs"]
pub(super) mod tests;
