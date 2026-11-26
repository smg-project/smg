//! The holder's side of a leader's stream: a standby lays the object of
//! each row its leader tells, and the store follows.
//!
//! A standby's applier hands the holder what it would have handed the
//! store: a membership event at its place in the leader's log, or the
//! leader's snapshot whole. The holder takes its turn,
//! as at a write of its own ([`mod@super::write`]). The leader
//! decided the change, so nothing is decided here. The holder builds the
//! object of the row the event tells, hands the store the event
//! ([`MemoryStore::follow_landed`]), and puts the object in the registry
//! where the store took it. So the object stands when the applier goes on
//! to its next event. With a change that takes a row out, a removal or a
//! key's next life, the holder hands the store that row as read back from
//! its object, as at a leader's write: the standby's own log is told from
//! it. Where no object stands under the row's id, the store tells its own.
//!
//! The order is the one of a leader's write. The store is told between the
//! build and the registry's write, and a removal's object leaves the
//! registry before the store is told. An event the store does not take
//! lays nothing: it does not fit the tables, and the applier stops its
//! batch there. Where the store does not take a removal, the row is
//! laid again as it was read back.
//!
//! The wire tells a row's age, not the instant it was admitted at. A row
//! an event changes in place keeps the instant its entry holds, as the
//! store keeps its own row's. A row of a watch's snapshot whose object
//! stands keeps it too. The holder states it on the snapshot's row before
//! the store takes the snapshot, and the store keeps none there.
//!
//! A told status is the leader's word. A status event is handed to the
//! store, and the holder lays the status on the object in the same turn.
//! So are the statuses of a cut laid whole, each on its row's object. A
//! told row lays its status too: an object laid in another's place takes
//! the standing object's, and a probe of a term that ended can have moved
//! that one ahead of the row's event. This is one of the two ways from a
//! table's status to an object's where the objects lead. The other is a
//! tell the store refuses ([`mod@super::status`]).
//!
//! A restore is the store's restore and the lay of the cut's own rows in
//! one turn of the holder. No row is read back. Nothing that takes the
//! turn finds the objects of one history beside the rows of another.
//!
//! A gateway's watcher hands the holder the same two: each change of its
//! watch's tail ([`Objects::follow_at`]), and the watch's snapshot at its
//! last page ([`Objects::watch_from`]). The snapshot is laid as a restore
//! is, by the store's own door for it.
//!
//! The locks, in their order: the applier's hold of its event's place in
//! the log, which is a mark and no lock, then the holder's turn, then the
//! store's membership lock inside the store's write. No lock of the store
//! is held over a write of the registry: the store's write has returned
//! when the object is put. A restore that carries a generation takes the
//! turn under the lock of the document's door, which its applier holds
//! from the generation's pass: nothing under the turn takes that door.

use smg_worker::registry::Generation;

use super::Objects;
use crate::store::{MembershipChange, MemoryStore, Position, StoreCheckpoint, WorkerRow};

#[cfg_attr(
    not(any(test, feature = "controller")),
    expect(
        dead_code,
        reason = "a controller's standby and a gateway's watcher follow through the holder"
    )
)]
impl<G: Generation> Objects<G> {
    /// One membership event of the leader's stream, at its place `at` in
    /// the leader's log: the write of [`MemoryStore::follow`], laid here.
    /// `epoch`, `change` and `row` are the event's, as that write takes
    /// them. Whether the store took the event: a gateway's watcher reads
    /// it here, and a standby's applier off the log's guard of the
    /// event's place.
    pub(crate) fn follow_at(
        &self,
        store: &MemoryStore,
        at: Position,
        epoch: u64,
        change: MembershipChange,
        row: Option<WorkerRow>,
    ) -> bool {
        let _turn = self.turn.lock();
        match (change, row) {
            // The row is read back, the object leaves the registry, then
            // the store is told and handed the row. A removal the store
            // refuses lays the row again, as it was read back.
            (MembershipChange::Removed { id }, None) => {
                let mut left = self.row(id);
                self.end(id);
                let landed = store.follow_landed(at, epoch, change, None, left.as_mut());
                if let (false, Some(row)) = (landed, &left) {
                    self.lay(store, row);
                }
                landed
            }
            (change, Some(mut row)) => {
                if matches!(
                    change,
                    MembershipChange::Relabelled { .. } | MembershipChange::Resized { .. }
                ) {
                    self.keep_admitted(&mut row);
                }
                // A key's next life takes the old life's row out: the store
                // is handed that row, read back from its object.
                let mut left = match change {
                    MembershipChange::Replaced { old, .. } => self.row(old),
                    _ => None,
                };
                let built = self.build(&row);
                let told = Some(row.clone());
                let landed = store.follow_landed(at, epoch, change, told, left.as_mut());
                if landed {
                    self.put(store, &row, built);
                    // The row's status is the leader's word too.
                    self.copy_status(store, row.id);
                }
                landed
            }
            // A told status: the table takes it, and the object behind it.
            (change, None) => {
                let landed = store.follow_landed(at, epoch, change, None, None);
                if let (true, MembershipChange::StatusChanged { id, .. }) = (landed, change) {
                    self.copy_status(store, id);
                }
                landed
            }
        }
    }

    /// `row` takes the instant its object's entry holds, where an object
    /// stands under its id. The wire tells a row's age, so a row told again
    /// comes with an instant of its own.
    fn keep_admitted(&self, row: &mut WorkerRow) {
        let stood = self.object(row.id);
        if let Some(stood) = stood.and_then(|object| self.remainder(&object)) {
            row.admitted_at = stood.admitted_at;
        }
    }

    /// The leader's snapshot whole: the write of
    /// [`MemoryStore::restore_from`], laid here. The store is restored and
    /// the cut's rows are laid in one turn, so their objects stand when
    /// this returns. `false`, with the store and the objects as they were,
    /// where the store refuses the cut's place.
    pub(crate) fn restore_from(&self, store: &MemoryStore, cut: StoreCheckpoint) -> bool {
        let _turn = self.turn.lock();
        self.laid_whole(store, cut, MemoryStore::restore_from)
    }

    /// A watch's snapshot whole, at its last page: the write of
    /// [`MemoryStore::watch_from`], laid here as a restore is. The objects
    /// of the snapshot's rows stand when this returns, and a row the
    /// snapshot drops has none. A row whose object stands keeps the instant
    /// it was admitted at ([`Self::keep_admitted`]): a restore keeps none.
    /// `false`, with the store and the objects as they were, where the
    /// store refuses the snapshot's place.
    #[cfg_attr(
        not(any(test, feature = "client")),
        expect(dead_code, reason = "a gateway's watcher lays its snapshot here")
    )]
    pub(crate) fn watch_from(&self, store: &MemoryStore, mut cut: StoreCheckpoint) -> bool {
        let _turn = self.turn.lock();
        for row in &mut cut.rows {
            self.keep_admitted(row);
        }
        self.laid_whole(store, cut, MemoryStore::watch_from)
    }

    /// One body for a cut the store takes whole, under the holder's turn
    /// its caller holds: `door` is the store's own entry for it. The
    /// holder keeps the cut's own rows and reads none back. Each is held
    /// against the object that stands for it ([`Self::reconcile`]), on its
    /// own: the rows' order moves no object. The keeper forgets the rows
    /// that left behind the store's write.
    fn laid_whole(
        &self,
        store: &MemoryStore,
        cut: StoreCheckpoint,
        door: impl FnOnce(&MemoryStore, StoreCheckpoint) -> bool,
    ) -> bool {
        let rows = cut.rows.clone();
        if !door(store, cut) {
            return false;
        }
        let read = self.keeper_clock();
        self.reconcile(store, &rows);
        self.forget_left(read);
        // A cut's statuses are told ones, as a status event's is.
        for row in rows {
            self.copy_status(store, row.id);
        }
        true
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/worker_host/objects/standby/tests.rs"]
mod tests;
