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
//! The store is handed the row a write takes out too, as it was read back:
//! a removal's row, and the old life a key's next life replaces. The log
//! is told from it, in the status the table holds.
//!
//! The store is told between the build and the registry's write: its write
//! files the member's count slot, which a lease books the moment a decision
//! picks the worker off the registry. A removal goes the other way round:
//! the object leaves the registry, then the store is told. A build cannot
//! fail, so the store is never told a change whose object was not built.
//!
//! A write the store does not take answers `None`. A registration's and
//! a patch's leave the objects as they stood: the object built for it is
//! dropped. A removal's object left the registry ahead of the store's
//! word: a new one is laid from the row read off it, under a new id, and
//! a lane the row holds in the host's KV tables is made anew, empty.
//! Where the store refused the removal for its term, that row comes back
//! in the status the table holds, and the object is laid in it. The
//! holder reads no row of the store behind a refusal: that status alone
//! comes back, on the row the holder handed.
//!
//! The store takes a change that fits its table, over a row that stands
//! at the revision the holder read. It tests both under its own lock.
//! Every product writer of a row holds the holder's turn: a door, a
//! standby's applier and a watcher's. So the store refuses a change where
//! its table was written around the holder, through an entry of the
//! store's own ([`MemoryStore::follow`]). The row and the object then
//! stand apart until a cut is laid whole ([`mod@super::standby`]). A
//! standby's applier and a watcher also lay the row at the next event
//! that tells it and that the store takes.
//!
//! A door's write names the term its call was admitted under
//! ([`Objects::upsert_under`]). The store's journal takes the row's event
//! only while the store leads that term, asked ahead of every move of the
//! store's tables. A write the store refuses for its term is one it does
//! not take: nothing moved there. A host's own write names no term.
//!
//! A writer's thread that ends between its two writes, the store's and
//! the registry's, leaves a row and its object apart too. No line of the
//! holder between the two returns early or unwraps a result.
//!
//! A patch tells the store the whole row read back from the object, and
//! the store writes it whole. It is the table's row with the patch laid
//! on it while an object states what its row stated: what an object does not
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
    store::{MembershipChange, MemoryStore, Registrant, WorkerFacts, WorkerPatch, WorkerRow},
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
    /// One of a host's write calls ([`Objects`]): its caller asks its term
    /// itself, so the call hands the store no term.
    pub fn upsert(
        &self,
        store: &MemoryStore,
        facts: WorkerFacts,
        maker: Registrant,
    ) -> Option<MembershipChange> {
        self.upsert_under(store, facts, maker, None)
    }

    /// [`Self::upsert`] under `term`, the term its caller's write is made
    /// under, for a caller whose term can end while the write waits. The
    /// term goes to the store's journal with the row's event, ahead of
    /// every move of the store's tables. Where the store has left it,
    /// nothing is written and the objects stand as they stood.
    pub(crate) fn upsert_under(
        &self,
        store: &MemoryStore,
        facts: WorkerFacts,
        maker: Registrant,
        term: Option<u64>,
    ) -> Option<MembershipChange> {
        let _turn = self.turn.lock();
        let mut standing = self.row_under(&facts.key);
        let seen = standing.as_ref().map(|row| row.revision);
        let (change, row) = WorkerRow::laid_over(facts, standing.as_ref(), maker, Instant::now());
        // A key's next life takes the standing row out: the store is
        // handed that row, as it was read back here.
        let replaced = matches!(change, MembershipChange::Replaced { .. });
        let left = standing.as_mut().filter(|_| replaced);
        self.lay_decided(store, change, &row, seen, left, term)
    }

    /// Write `patch` over the row `id` where its object was built at
    /// `revision`. The change it made, or `None` where the row moved or
    /// left, or `store` did not take the write.
    ///
    /// One of a host's write calls ([`Objects`]), under no term.
    pub fn patch(
        &self,
        store: &MemoryStore,
        id: WorkerId,
        patch: WorkerPatch,
        revision: u64,
    ) -> Option<MembershipChange> {
        self.patch_under(store, id, patch, revision, None)
    }

    /// [`Self::patch`] under `term`, as [`Self::upsert_under`] takes one.
    pub(crate) fn patch_under(
        &self,
        store: &MemoryStore,
        id: WorkerId,
        patch: WorkerPatch,
        revision: u64,
        term: Option<u64>,
    ) -> Option<MembershipChange> {
        let _turn = self.turn.lock();
        let mut row = self.guarded(id, revision)?;
        patch.laid_on(&mut row.facts);
        row.revision += 1;
        let change = MembershipChange::Relabelled { id };
        self.lay_decided(store, change, &row, Some(revision), None, term)
    }

    /// Remove the row `id` where its object was built at `revision`. The
    /// object leaves the registry, then `store` is told. The change it
    /// made, or `None` where the row moved or left, or `store` did not take
    /// the write.
    ///
    /// One of a host's write calls ([`Objects`]), under no term.
    pub fn remove(
        &self,
        store: &MemoryStore,
        id: WorkerId,
        revision: u64,
    ) -> Option<MembershipChange> {
        self.remove_under(store, id, revision, None)
    }

    /// [`Self::remove`] under `term`, as [`Self::upsert_under`] takes one.
    /// A removal the store refuses for its term lays the row again, in
    /// the status the table holds ([`Self::take_decided`]).
    pub(crate) fn remove_under(
        &self,
        store: &MemoryStore,
        id: WorkerId,
        revision: u64,
        term: Option<u64>,
    ) -> Option<MembershipChange> {
        let _turn = self.turn.lock();
        let mut row = self.guarded(id, revision)?;
        self.take_decided(store, &mut row, term)
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
    /// change the store does not take builds an object nobody sees, and
    /// nothing is laid. `left` is the row a key's next life takes out, as
    /// it was read back: the store tells the log from it. `term` is the
    /// term the write is made under, where its caller names one: the
    /// store's journal takes the write, and the status told behind a
    /// change in place, only while the store leads it.
    fn lay_decided(
        &self,
        store: &MemoryStore,
        change: MembershipChange,
        row: &WorkerRow,
        seen: Option<u64>,
        left: Option<&mut WorkerRow>,
        term: Option<u64>,
    ) -> Option<MembershipChange> {
        let built = self.build(row);
        if !store.follow_objects(change, Some(row.clone()), seen, left, term) {
            return None;
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
            self.tell_row(store, row.id, row.revision, term);
        }
        Some(change)
    }

    /// The decided removal of `row`, read back in the holder's turn: its
    /// object leaves the registry, then the store is told. Where the store
    /// does not take it, the row is laid again: a new object under a new
    /// id in the registry. Where the host handed the holder KV tables,
    /// a lane the row holds there is new too, and holds nothing until its
    /// first batch. The leases the store counts on the row are stated on the
    /// new object. What the old object alone held is not, and what the
    /// old lane held is not. The store is handed `row`, and tells the log
    /// from it. `term` is the term the removal is made under, where its
    /// caller names one. A removal the store refuses for that term hands
    /// `row` back in the status the table holds, and the object is laid
    /// in it: a status a health driver moved on the old object ahead of
    /// its telling leaves with that object, and the table and the new
    /// object read alike.
    fn take_decided(
        &self,
        store: &MemoryStore,
        row: &mut WorkerRow,
        term: Option<u64>,
    ) -> Option<MembershipChange> {
        self.end(row.id);
        let change = MembershipChange::Removed { id: row.id };
        if !store.follow_objects(change, None, Some(row.revision), Some(&mut *row), term) {
            self.lay(store, row);
            return None;
        }
        self.forget_one(self.keeper_clock(), row.id);
        Some(change)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/worker_host/objects/write/tests.rs"]
pub(super) mod tests;
