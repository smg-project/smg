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
//! So the object a write leaves stands when the write returns.
//! No row of the store's table is read at the head of a write: every
//! writer of a row holds the turn, so the object states the row.
//! The store is handed the row a write takes out too, as it was read back:
//! a removal's row, and the old life a key's next life replaces. The log
//! is told from it, in the status the table holds.
//!
//! The store is told first, and the registry is written behind its yes.
//! A row's object is built ahead of the store's word and put behind it:
//! the store's write files the member's count slot, which a lease books
//! the moment a decision picks the worker off the registry. A removal's
//! object leaves the registry behind the store's word too, as the old
//! object of a key's next life does. A build cannot fail, so the store is
//! never told a change whose object was not built.
//!
//! So an object that leaves outlives its row for a moment. The moment
//! opens inside the store's write, where the row leaves the table. The
//! same write then has the row's count give up the leases it held, and
//! pays the removal's own build: the world it publishes holds the worker
//! in no pool, and no veto of a gateway on it. The moment lasts until
//! the registry's removal makes a ready worker not ready, which it does
//! under the worker's own lock and ahead of taking the object out. A
//! status call can hold that lock. In the moment the worker stands in the
//! lists a decision is lent, in the status its object states, and a
//! decision can pick it. Behind the count's step the object counts no
//! lease that stands on it. What a lease files on the worker from there
//! on is on no host's count, until a row stands under the id again. A
//! list lent ahead of the object's end holds the worker.
//!
//! A write the store does not take answers `None`. The object built for
//! a registration or a patch is dropped, and the objects stand as they
//! stood. A removal's object was not touched: the same object stands
//! under the same id in the registry, with its running state, and the
//! lane its row holds in the host's KV tables keeps what it holds. The
//! registry tells no event. The holder reads no row of the store behind
//! a refusal.
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
//! the registry's, leaves a row and its object apart too, where the
//! write lays an object: a registration's row then has no object, and no
//! list lends it. A removal's two steps run under one catch
//! ([`leave_whole`]). Where either unwinds and the store holds the row no
//! more, the second step runs, a second time where it had begun. A fault
//! of that run stops there, and the unwinding goes on. No line between
//! the store's yes and the object's end returns early or unwraps a
//! result.
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

use super::{leave_whole, Objects};
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

    /// Remove the row `id` where its object was built at `revision`.
    /// `store` is told, then the object leaves the registry. The change it
    /// made, or `None` where the row moved or left, or `store` did not take
    /// the write: the object then stands as it stood.
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
    /// A removal the store refuses for its term leaves its object as it
    /// stood ([`Self::take_decided`]).
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

    /// The decided removal of `row`, read back in the holder's turn: the
    /// store is told, and the object leaves the registry behind its yes.
    /// Where the store does not take the removal, the object is not
    /// touched: the same object stands under the same id in the registry,
    /// with its running state and its status, and the lane its row holds
    /// in the host's KV tables keeps what it holds. Behind the store's
    /// yes the removal's second step runs: the object's end, then the
    /// keeper's forgetting of the row. Where a step unwinds and the store
    /// holds the row no more, the second step runs on the way out
    /// ([`leave_whole`]). The store is handed `row`, and tells
    /// the log from it, in the status the table holds. `term` is the term
    /// the removal is made under, where its caller names one: a removal
    /// the store refuses for that term is one it does not take.
    fn take_decided(
        &self,
        store: &MemoryStore,
        row: &mut WorkerRow,
        term: Option<u64>,
    ) -> Option<MembershipChange> {
        let (id, seen) = (row.id, Some(row.revision));
        let change = MembershipChange::Removed { id };
        let told = || store.follow_objects(change, None, seen, Some(&mut *row), term);
        let behind = || {
            self.end(id);
            self.forget_one(self.keeper_clock(), id);
        };
        leave_whole(store, id, told, behind).then_some(change)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/worker_host/objects/write/tests.rs"]
pub(super) mod tests;
