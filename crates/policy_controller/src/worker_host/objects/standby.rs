//! The holder's side of a leader's stream: a standby lays the object of
//! each row its leader tells, and the store follows.
//!
//! A standby's applier hands the holder what it would have handed the
//! store: a membership event at its place in the leader's log, or the
//! leader's snapshot whole. The holder takes its turn and reads the store
//! to its end, as at a write of its own ([`mod@super::write`]). The leader
//! decided the change, so nothing is decided here. The holder builds the
//! object of the row the event tells, hands the store the event
//! ([`MemoryStore::follow_landed`]), and puts the object in the registry
//! where the store took it. So the object stands when the applier goes on
//! to its next event, with no follow.
//!
//! The order is the one of a leader's write. The store is told between the
//! build and the registry's write, and a removal's object leaves the
//! registry before the store is told. An event the store does not take
//! lays nothing: it does not fit the tables, and the applier stops its
//! batch there. Where the store does not take a removal, the row that
//! stands is settled again and gets an object anew.
//!
//! The wire tells a row's age, not the instant it was admitted at. A row
//! an event changes in place keeps the instant its entry holds, as the
//! store keeps its own row's.
//!
//! A status is the table's word still. A status event is handed to the
//! store, and the holder copies the status onto the object in the same
//! turn, where it follows statuses ([`Objects::with_status_followed`]).
//!
//! A restore is the store's restore and the read of its rows in one turn
//! of the holder: nothing that takes the turn finds the objects of one
//! history beside the place of another.
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
    /// them. The applier reads whether the event landed off the log's
    /// guard of its place.
    pub(crate) fn follow_at(
        &self,
        store: &MemoryStore,
        at: Position,
        epoch: u64,
        change: MembershipChange,
        row: Option<WorkerRow>,
    ) {
        let mut place = self.read_to.lock();
        self.follow_from(store, &mut place);
        match (change, row) {
            // The object leaves the registry, then the store is told.
            (MembershipChange::Removed { id }, None) => {
                self.end(id);
                if !store.follow_landed(at, epoch, change, None) {
                    self.settle(store, id);
                }
            }
            (change, Some(mut row)) => {
                if matches!(
                    change,
                    MembershipChange::Relabelled { .. } | MembershipChange::Resized { .. }
                ) {
                    let stood = self.object(row.id);
                    if let Some(stood) = stood.and_then(|object| self.remainder(&object)) {
                        row.admitted_at = stood.admitted_at;
                    }
                }
                let built = self.build(&row);
                if store.follow_landed(at, epoch, change, Some(row.clone())) {
                    self.put(store, &row, built);
                }
            }
            // A status, which the table keeps: the object follows it.
            (change, None) => {
                if store.follow_landed(at, epoch, change, None) {
                    if let MembershipChange::StatusChanged { id, .. } = change {
                        self.follow_status(store, id);
                    }
                }
            }
        }
    }

    /// The leader's snapshot whole: the write of
    /// [`MemoryStore::restore_from`], laid here. The store is restored and
    /// its rows are read in one turn, so the objects of the cut's rows
    /// stand when this returns. `false`, with the store and the objects as
    /// they were, where the store refuses the cut's place.
    pub(crate) fn restore_from(&self, store: &MemoryStore, cut: StoreCheckpoint) -> bool {
        self.laid_whole(store, cut, MemoryStore::restore_from)
    }

    /// A watch's snapshot whole, at its last page: the write of
    /// [`MemoryStore::watch_from`], laid here as a restore is. The objects
    /// of the snapshot's rows stand when this returns, and a row the
    /// snapshot drops has none. `false`, with the store and the objects as
    /// they were, where the store refuses the snapshot's place.
    #[cfg_attr(
        not(any(test, feature = "client")),
        expect(dead_code, reason = "a gateway's watcher lays its snapshot here")
    )]
    pub(crate) fn watch_from(&self, store: &MemoryStore, cut: StoreCheckpoint) -> bool {
        self.laid_whole(store, cut, MemoryStore::watch_from)
    }

    /// One body for a cut the store takes whole: `door` is the store's own
    /// entry for it. The store's write and the read of its rows are one
    /// turn of the holder.
    fn laid_whole(
        &self,
        store: &MemoryStore,
        cut: StoreCheckpoint,
        door: impl FnOnce(&MemoryStore, StoreCheckpoint) -> bool,
    ) -> bool {
        let mut place = self.read_to.lock();
        if !door(store, cut) {
            return false;
        }
        self.follow_from(store, &mut place);
        true
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc, sync::Arc, time::Duration};

    use super::{
        super::write::tests::{facts, Replica, A},
        *,
    };
    use crate::store::{memory::seam, WorkerFacts, WorkerStatus, WorkersStore};

    const B: &str = "http://b:8000";

    /// The place of an event in a leader's log. The journal of these cases
    /// keeps no place: the store's epoch alone says where an event fits.
    fn at(seq: u64) -> Position {
        Position { seq, term: 1 }
    }

    /// The leader's last write as its stream tells it: the change and the
    /// row it left.
    fn told(leader: &Replica) -> (MembershipChange, Option<WorkerRow>) {
        let (change, row, _) = leader.last();
        (change, row)
    }

    /// Hand `standby`'s holder one event, at the epoch `ahead` of the
    /// standby's own: the next one is one ahead.
    fn hand(standby: &Replica, ahead: u64, change: MembershipChange, row: Option<WorkerRow>) {
        let epoch = standby.store.epoch() + ahead;
        let (store, objects) = (&standby.store, &standby.objects);
        objects.follow_at(store, at(epoch), epoch, change, row);
    }

    /// Hand `standby`'s holder the leader's last write as its next event.
    fn follow(standby: &Replica, leader: &Replica) {
        let (change, row) = told(leader);
        hand(standby, 1, change, row);
    }

    /// Each event a leader tells is laid as it comes: the object stands
    /// when the event's write returns, with no follow, and the standby's
    /// journal, its table and the object state one row, the leader's. A row
    /// changed in place keeps its object's id in the registry, a key's next
    /// life takes another id, and a removal takes the object out.
    #[test]
    fn a_told_change_leaves_its_object_standing_and_one_row_in_three_places() {
        let (leader, standby) = (Replica::new(), Replica::new());
        // A new key.
        leader.upsert(facts(1, "gold"));
        follow(&standby, &leader);
        assert_eq!(standby.one_row(A), leader.one_row(A));
        let object = standby.object(A);
        // The row's own incarnation: in place, under the object's id.
        leader.upsert(facts(1, "iron"));
        follow(&standby, &leader);
        assert_eq!(standby.one_row(A), leader.one_row(A));
        assert_eq!(standby.object(A), object, "in place");
        // The key's next life: another id, and nothing of the old life.
        leader.upsert(facts(2, "iron"));
        follow(&standby, &leader);
        assert_eq!(standby.one_row(A), leader.one_row(A));
        assert_ne!(standby.object(A), object, "a next life");
        assert_eq!(standby.objects.len(), 1);
        // A removal: the object leaves with its row.
        let row = leader.one_row(A);
        let removed = leader.objects.remove(&leader.store, row.id, row.revision);
        assert_eq!(removed, Some(MembershipChange::Removed { id: row.id }));
        follow(&standby, &leader);
        assert_eq!(standby.store.by_key(A), None);
        assert_eq!(standby.object(A), None);
        assert!(standby.objects.is_empty());
    }

    /// The locks in their order. Behind the store's write of an event and
    /// ahead of the put, the holder's turn is held, the store's membership
    /// lock is free and its table holds the row, and the registry does not
    /// list the row's object yet. Behind the event the turn is free.
    #[test]
    fn the_holders_turn_is_held_over_the_stores_write_and_no_lock_of_the_store_over_the_put() {
        let (leader, standby) = (Replica::new(), Rc::new(Replica::new()));
        leader.upsert(facts(1, "gold"));
        let seen = Rc::new(Cell::new(None));
        let (within, read) = (Rc::clone(&standby), Rc::clone(&seen));
        seam::on_release(move || {
            let turn = within.objects.read_to.is_locked();
            let table = within.store.by_key(A).is_some();
            read.set(Some((turn, table, within.object(A).is_some())));
        });
        follow(&standby, &leader);
        assert_eq!(seen.get(), Some((true, true, false)));
        assert!(standby.object(A).is_some(), "put behind the store's write");
        assert!(!standby.objects.read_to.is_locked(), "the turn is let go");
    }

    /// The wire tells a row's age, so each event's row comes with an
    /// instant of its own. A row told in place keeps the instant it was
    /// admitted at, by a relabel and by a resize: the table's row and the
    /// row read back from the object state the first one.
    #[test]
    fn a_row_told_in_place_keeps_the_instant_it_was_admitted_at() {
        let (leader, standby) = (Replica::new(), Replica::new());
        leader.upsert(facts(1, "gold"));
        follow(&standby, &leader);
        let admitted = standby.one_row(A).admitted_at;
        let wider = WorkerFacts {
            dp_size: 2,
            ..facts(1, "iron")
        };
        for (revision, stated, resizes) in [(2, facts(1, "iron"), false), (3, wider, true)] {
            leader.upsert(stated);
            let (change, row) = told(&leader);
            let resized = matches!(change, MembershipChange::Resized { .. });
            assert_eq!(resized, resizes, "{change:?}");
            let mut row = row.expect("a row changed in place is told");
            row.admitted_at += Duration::from_secs(5);
            hand(&standby, 1, change, Some(row));
            let row = standby.one_row(A);
            assert_eq!((row.revision, row.admitted_at), (revision, admitted));
        }
    }

    /// An event the store does not take lays nothing: here one at an epoch
    /// that is not the next. The object that stands stays in the registry,
    /// and it states the row the table holds.
    #[test]
    fn an_event_the_store_does_not_take_lays_nothing() {
        let (leader, standby) = (Replica::new(), Replica::new());
        leader.upsert(facts(1, "gold"));
        follow(&standby, &leader);
        let stands = || {
            let id = standby.object(A).expect("the row's object");
            standby.objects.registry().get(&id).expect("it stands")
        };
        let stood = stands();
        leader.upsert(facts(1, "iron"));
        let (change, row) = told(&leader);
        hand(&standby, 2, change, row);
        assert_eq!(standby.store.epoch(), 1, "the store took nothing");
        assert!(Arc::ptr_eq(&stood, &stands()), "the object that stood");
        let object = standby.object(A).expect("the row's object");
        let table = standby.store.by_key(A).expect("the row stands");
        assert_eq!(table.facts, facts(1, "gold"));
        assert_eq!(standby.objects.row_of(&object), Some(table));
    }

    /// A removal the store does not take: the object left the registry
    /// ahead of the store's word, so the row that stands is settled again
    /// and has an object when the event's write returns.
    #[test]
    fn a_removal_the_store_does_not_take_leaves_the_row_an_object() {
        let (leader, standby) = (Replica::new(), Replica::new());
        leader.upsert(facts(1, "gold"));
        follow(&standby, &leader);
        let row = standby.one_row(A);
        hand(&standby, 2, MembershipChange::Removed { id: row.id }, None);
        let object = standby
            .object(A)
            .expect("the row that stands has an object");
        assert_eq!(standby.objects.row_of(&object), Some(row));
    }

    /// A status the leader tells is the table's word, and the object
    /// follows it in the event's own turn: no follow reads it.
    #[test]
    fn a_told_status_is_on_the_object_when_the_event_returns() {
        let (leader, standby) = (Replica::new(), Replica::new());
        leader.upsert(facts(1, "gold"));
        follow(&standby, &leader);
        let id = standby.one_row(A).id;
        let (from, to) = (WorkerStatus::Pending, WorkerStatus::Ready);
        hand(
            &standby,
            1,
            MembershipChange::StatusChanged { id, from, to },
            None,
        );
        let table = standby.store.by_key(A).expect("the row stands");
        assert_eq!(table.status, WorkerStatus::Ready);
        let object = standby.object(A).expect("the row's object");
        assert_eq!(standby.objects.row_of(&object), Some(table));
        assert_eq!(standby.objects.read_to(), Some(1), "no follow read it");
    }

    /// A restore lays the cut's rows in its own turn: each has its object
    /// when the restore returns, a row the cut drops has none, and the
    /// holder stands at the cut's epoch.
    #[test]
    fn a_restores_rows_have_their_objects_when_it_returns() {
        let (leader, standby) = (Replica::new(), Replica::new());
        standby.upsert(WorkerFacts {
            key: B.to_owned(),
            engine: B.to_owned(),
            ..facts(1, "gold")
        });
        assert!(standby.object(B).is_some());
        leader.upsert(facts(1, "gold"));
        leader.upsert(facts(1, "iron"));
        let cut = leader.store.checkpoint();
        assert!(standby.objects.restore_from(&standby.store, cut));
        let object = standby.object(A).expect("the cut's row has its object");
        assert_eq!(standby.objects.row_of(&object), leader.store.by_key(A));
        assert_eq!(standby.object(B), None, "a row the cut drops");
        assert_eq!(standby.objects.len(), 1);
        assert_eq!(standby.objects.read_to(), Some(leader.store.epoch()));
    }

    /// A watch's snapshot is laid as a restore is, by the store's door for
    /// a watch: its rows have their objects when it returns, and a row it
    /// drops has none. A row that stands through a second snapshot keeps
    /// the object it had.
    #[test]
    fn a_watch_snapshots_rows_have_their_objects_when_it_returns() {
        let (told, watching) = (Replica::new(), Replica::new());
        let other = || WorkerFacts {
            key: B.to_owned(),
            engine: B.to_owned(),
            ..facts(1, "gold")
        };
        watching.upsert(other());
        told.upsert(facts(1, "gold"));
        let (store, objects) = (&watching.store, &watching.objects);
        assert!(objects.watch_from(store, told.store.checkpoint()));
        let stands = || {
            let id = watching
                .object(A)
                .expect("the snapshot's row has its object");
            (
                objects.row_of(&id),
                objects.registry().get(&id).expect("it stands"),
            )
        };
        let (row, stood) = stands();
        assert_eq!(row, told.store.by_key(A));
        assert_eq!(watching.object(B), None, "a row the snapshot drops");
        assert_eq!(objects.len(), 1);
        // A second snapshot, with one row more.
        told.upsert(other());
        assert!(objects.watch_from(store, told.store.checkpoint()));
        assert!(Arc::ptr_eq(&stood, &stands().1), "the object that stood");
        assert!(watching.object(B).is_some(), "the row the snapshot brings");
        assert_eq!(objects.len(), 2);
        assert_eq!(objects.read_to(), Some(told.store.epoch()));
    }
}
