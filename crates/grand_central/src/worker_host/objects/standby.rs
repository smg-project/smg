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
//! to its next event.
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
//! A told status is the leader's word. A status event is handed to the
//! store, and the holder lays the status on the object in the same turn.
//! So are the statuses of a cut laid whole, each on its row's object. A
//! told row lays its status too: an object laid in another's place takes
//! the standing object's, and a probe of a term that ended can have moved
//! that one ahead of the row's event. This is one of the two ways from a
//! table's status to an object's where the objects lead. The other is a
//! tell the store refuses ([`mod@super::status`]).
//!
//! A restore is the store's restore and the read of its rows in one turn
//! of the holder: nothing that takes the turn finds the objects of one
//! history beside the rows of another.
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
use crate::store::{
    MembershipChange, MemoryStore, Position, StoreCheckpoint, WorkerRow, WorkersStore,
};

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
            // The object leaves the registry, then the store is told.
            (MembershipChange::Removed { id }, None) => {
                self.end(id);
                let landed = store.follow_landed(at, epoch, change, None);
                if !landed {
                    self.settle(store, id);
                }
                landed
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
                let landed = store.follow_landed(at, epoch, change, Some(row.clone()));
                if landed {
                    self.put(store, &row, built);
                    // The row's status is the leader's word too.
                    self.copy_status(store, row.id);
                }
                landed
            }
            // A told status: the table takes it, and the object behind it.
            (change, None) => {
                let landed = store.follow_landed(at, epoch, change, None);
                if let (true, MembershipChange::StatusChanged { id, .. }) = (landed, change) {
                    self.copy_status(store, id);
                }
                landed
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
    /// turn of the holder. The rows are read once, whole, and each is held
    /// against the object that stands for it ([`Self::reconcile`]). The
    /// keeper forgets the rows that left behind the read.
    fn laid_whole(
        &self,
        store: &MemoryStore,
        cut: StoreCheckpoint,
        door: impl FnOnce(&MemoryStore, StoreCheckpoint) -> bool,
    ) -> bool {
        let _turn = self.turn.lock();
        if !door(store, cut) {
            return false;
        }
        let read = self.keeper_clock();
        let (_, rows) = store.all_rows();
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
mod tests {
    use std::{
        cell::Cell,
        rc::Rc,
        sync::{mpsc, Arc},
        thread,
        time::Duration,
    };

    use smg_worker::{
        identity::basic::Worker, kv_events::port::BatchOutcome, registry::WorkerRegistry,
        worker::WorkerStatus as ObjectStatus,
    };

    use super::{
        super::{
            tests::stored_chain,
            write::tests::{facts, Replica, A},
        },
        *,
    };
    use crate::{
        store::{memory::seam, Signals, WorkerFacts, WorkerStatus, WorkersStore},
        worker_host::{KvTables, Members},
    };

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
    /// standby's own: the next one is one ahead. The holder's answer.
    fn hand(
        standby: &Replica,
        ahead: u64,
        change: MembershipChange,
        row: Option<WorkerRow>,
    ) -> bool {
        let epoch = standby.store.epoch() + ahead;
        let (store, objects) = (&standby.store, &standby.objects);
        objects.follow_at(store, at(epoch), epoch, change, row)
    }

    /// Hand `standby`'s holder the leader's last write as its next event.
    fn follow(standby: &Replica, leader: &Replica) {
        let (change, row) = told(leader);
        hand(standby, 1, change, row);
    }

    /// Each event a leader tells is laid as it comes: the object stands
    /// when the event's write returns, and the standby's
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

    /// A write call asks no term, so a host that does not lead must not
    /// call one. Where it does, the write lands in its own store alone: its
    /// leader's next event no longer fits that store and lays nothing, and
    /// the leader's snapshot takes the write back.
    #[test]
    fn a_standbys_own_write_is_lost_at_its_leaders_next_snapshot() {
        let (leader, standby) = (Replica::new(), Replica::new());
        leader.upsert(facts(1, "gold"));
        follow(&standby, &leader);
        // The standby's own write, which no leader told.
        let own = WorkerFacts {
            key: B.to_owned(),
            engine: B.to_owned(),
            ..facts(1, "iron")
        };
        assert!(standby.upsert(own).is_some());
        assert!(standby.object(B).is_some());
        // The leader's next event comes at the leader's epoch.
        leader.upsert(facts(1, "iron"));
        let (change, row) = told(&leader);
        let epoch = leader.store.epoch();
        let (store, objects) = (&standby.store, &standby.objects);
        assert!(!objects.follow_at(store, at(epoch), epoch, change, row));
        let stale = store.by_key(A).expect("the row stands");
        assert_eq!(stale.revision, 1, "nothing of the event is laid");
        // The applier starts again from the leader's snapshot.
        assert!(objects.restore_from(store, leader.store.checkpoint()));
        assert_eq!(standby.object(B), None, "the standby's own write is gone");
        assert_eq!(store.by_key(B), None);
        let object = standby.object(A).expect("the leader's row has its object");
        assert_eq!(objects.row_of(&object), leader.store.by_key(A));
        assert_eq!(objects.len(), 1);
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
            let turn = within.objects.turn.is_locked();
            let table = within.store.by_key(A).is_some();
            read.set(Some((turn, table, within.object(A).is_some())));
        });
        follow(&standby, &leader);
        assert_eq!(seen.get(), Some((true, true, false)));
        assert!(standby.object(A).is_some(), "put behind the store's write");
        assert!(!standby.objects.turn.is_locked(), "the turn is let go");
    }

    /// A cut laid whole takes the holder's turn ahead of the store's write.
    /// While another holds the turn the restore waits there, and the store
    /// stands as it was. Behind the turn the cut is laid, and its row has
    /// its object.
    #[test]
    fn a_cut_laid_whole_waits_for_the_holders_turn() {
        let (leader, standby) = (Replica::new(), Replica::new());
        leader.upsert(facts(1, "gold"));
        let cut = leader.store.checkpoint();
        let (store, objects) = (&standby.store, &standby.objects);
        let (laid, lay) = mpsc::channel();
        let turn = objects.turn.lock();
        thread::scope(|scope| {
            let call = scope.spawn(move || {
                let took = objects.restore_from(store, cut);
                let _ = laid.send(());
                took
            });
            assert_eq!(
                lay.recv_timeout(Duration::from_millis(200)),
                Err(mpsc::RecvTimeoutError::Timeout)
            );
            assert_eq!(store.by_key(A), None, "the cut waits");
            drop(turn);
            assert!(call.join().expect("the call ends"), "the cut is laid");
        });
        assert!(standby.object(A).is_some(), "its row has its object");
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

    /// The holder answers whether the store took an event, by each kind: a
    /// row, a status and a removal answer yes at the next epoch, and no at
    /// the epoch the store holds already, where nothing fits.
    #[test]
    fn a_told_change_answers_whether_the_store_took_it() {
        let (leader, standby) = (Replica::new(), Replica::new());
        leader.upsert(facts(1, "gold"));
        let (change, row) = told(&leader);
        assert!(hand(&standby, 1, change, row.clone()), "a row");
        assert!(!hand(&standby, 0, change, row), "a row told again");
        let id = standby.store.by_key(A).expect("the row stands").id;
        let (from, to) = (WorkerStatus::Pending, WorkerStatus::Ready);
        let status = MembershipChange::StatusChanged { id, from, to };
        assert!(hand(&standby, 1, status, None), "a status");
        assert!(!hand(&standby, 0, status, None), "a status told again");
        let removal = MembershipChange::Removed { id };
        assert!(!hand(&standby, 0, removal, None), "a removal out of place");
        assert!(standby.object(A).is_some(), "the row stands, laid anew");
        assert!(hand(&standby, 1, removal, None), "a removal");
        assert!(standby.objects.is_empty());
    }

    /// A status the leader tells is laid on the object in the event's own
    /// turn.
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
    }

    /// A tell the store takes no word of leaves the object at the status
    /// its row holds. A probe's move ahead of a tell whose term ended is
    /// taken back to the row's status. Where the standby's applier laid a
    /// status its new leader told since the move, the object keeps the
    /// leader's word: the status the probe saw it leave is not put back.
    #[test]
    fn a_refused_tell_leaves_the_object_at_the_status_its_row_holds() {
        let replica = Replica::new();
        replica.upsert(facts(1, "gold"));
        let row = replica.one_row(A).id;
        let id = replica.object(A).expect("the row's object");
        let registry = replica.objects.registry();
        let revision = registry.get(&id).expect("it stands").revision();
        let statuses = || {
            let object = registry.get(&id).expect("it stands").status();
            let table = replica.store.by_key(A).expect("the row stands");
            (object, table.status)
        };
        // The registry's state machine moves the object ahead of a tell,
        // and the tell's term ended: the store takes no word.
        let moved = (ObjectStatus::Pending, ObjectStatus::Ready);
        let moves = || {
            let ready = |_: &Arc<dyn Worker>| ((), Some(moved.1));
            registry.apply_if_revision(&id, revision, ready);
        };
        let refused = || {
            let (store, objects) = (&replica.store, &replica.objects);
            objects
                .tell_status(store, &id, revision, moved, &|| false)
                .stands()
        };
        moves();
        assert_eq!(refused(), None);
        assert_eq!(statuses(), (ObjectStatus::Pending, WorkerStatus::Pending));
        // The same, with the new leader's word laid between the two.
        moves();
        let (from, to) = (WorkerStatus::Pending, WorkerStatus::Ready);
        let told = MembershipChange::StatusChanged { id: row, from, to };
        assert!(hand(&replica, 1, told, None));
        assert_eq!(refused(), None);
        assert_eq!(statuses(), (ObjectStatus::Ready, WorkerStatus::Ready));
    }

    /// A told row lays its status too. Its object is laid in the standing
    /// one's place and takes that one's status, which a probe of a term
    /// that ended can have moved ahead of the row's event: the leader's
    /// word stands on the object the event leaves.
    #[test]
    fn a_told_row_lays_the_status_its_leader_holds() {
        let (leader, standby) = (Replica::new(), Replica::new());
        leader.upsert(facts(1, "gold"));
        follow(&standby, &leader);
        let id = standby.object(A).expect("the row's object");
        let registry = standby.objects.registry();
        let revision = registry.get(&id).expect("it stands").revision();
        // The ended term's probe moves the object, and its tell has not run.
        let ready = |_: &Arc<dyn Worker>| ((), Some(ObjectStatus::Ready));
        registry.apply_if_revision(&id, revision, ready);
        // The leader's next word of the row, in place.
        leader.upsert(facts(1, "iron"));
        follow(&standby, &leader);
        let laid = standby.object(A).expect("the row's object");
        let object = registry.get(&laid).expect("it stands");
        assert_eq!(object.status(), ObjectStatus::Pending);
        assert_eq!(standby.one_row(A), leader.one_row(A));
    }

    /// A restore lays the cut's rows in its own turn: each has its object
    /// when the restore returns, and a row the cut drops has none.
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
    }

    /// A cut's statuses are told ones. Each object states the status its
    /// row holds in the cut when the cut is laid, by a restore and by a
    /// watch's snapshot: here the object of a row that stood through it,
    /// which no lay builds anew.
    #[test]
    fn a_cuts_statuses_are_on_the_objects_when_it_is_laid() {
        type Lay = fn(&Objects<Members>, &MemoryStore, StoreCheckpoint) -> bool;
        let lays: [(&str, Lay); 2] = [
            ("a restore", |objects, store, cut| {
                objects.restore_from(store, cut)
            }),
            ("a watch's snapshot", |objects, store, cut| {
                objects.watch_from(store, cut)
            }),
        ];
        for (name, lay) in lays {
            let (leader, standby) = (Replica::new(), Replica::new());
            leader.upsert(facts(1, "gold"));
            follow(&standby, &leader);
            let stood = standby.object(A).expect("the row's object");
            // The leader readies the row, and the standby is not told.
            let id = leader.one_row(A).id;
            leader.objects.connect_row(&leader.store, id, 1);
            let cut = leader.store.checkpoint();
            assert!(lay(&standby.objects, &standby.store, cut), "{name}");
            assert_eq!(standby.object(A).as_ref(), Some(&stood), "{name}");
            let row = standby.objects.row_of(&stood).expect("its row");
            assert_eq!(row.status, WorkerStatus::Ready, "{name}");
            assert_eq!(Some(row), leader.store.by_key(A), "{name}");
        }
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
    }

    /// A leader, and a standby whose holder was handed the KV tables that
    /// come back beside them.
    fn tabled() -> (Replica, Replica, Arc<KvTables>) {
        let tables = Arc::new(KvTables::default());
        let mut standby = Replica::new();
        standby.objects =
            Objects::new(Arc::new(WorkerRegistry::new())).with_kv_tables(Arc::clone(&tables));
        (Replica::new(), standby, tables)
    }

    /// [`facts`] of the row at `key`, whose signals ask for cache events
    /// where `events` says so.
    fn asking(key: &str, events: bool) -> WorkerFacts {
        WorkerFacts {
            key: key.to_owned(),
            engine: key.to_owned(),
            signals: Signals {
                events,
                ..Signals::default()
            },
            ..facts(1, "gold")
        }
    }

    /// A standby's holder that was handed KV tables keeps a lane for each
    /// row that asks for cache events, where its leader's event is told. A
    /// row told quiet in place gives its lane up, one told asking again
    /// takes one, and a row told gone holds none.
    #[test]
    fn a_standbys_holder_keeps_a_lane_for_each_told_row_that_asks_for_events() {
        let (leader, standby, tables) = tabled();
        let lane = |id| tables.holds_lane(id, 0);
        leader.upsert(asking(A, true));
        follow(&standby, &leader);
        let row = leader.store.by_key(A).expect("the leader's row").id;
        assert!(lane(row), "a told row that asks");
        leader.upsert(asking(A, false));
        follow(&standby, &leader);
        assert!(!lane(row), "told quiet in place");
        leader.upsert(asking(A, true));
        follow(&standby, &leader);
        assert!(lane(row), "told asking again");
        let stands = leader.store.by_key(A).expect("the leader's row");
        assert_eq!(stands.id, row, "in place");
        let removed = leader
            .objects
            .remove(&leader.store, stands.id, stands.revision);
        assert_eq!(removed, Some(MembershipChange::Removed { id: row }));
        follow(&standby, &leader);
        assert!(!lane(row), "told gone");
        assert_eq!(tables.lanes(), 0);
    }

    /// A cut laid whole leaves a lane for each of its rows that asks for
    /// cache events, by a restore and by a watch's snapshot, and a row the
    /// cut drops holds none. A row that stays through the next cut keeps the
    /// blocks this replica heard for it: they are the replica's own, in no
    /// cut, so a lay that made its lane again would lose them.
    #[test]
    fn a_cut_laid_whole_leaves_a_lane_for_each_row_that_asks_for_events() {
        type Lay = fn(&Objects<Members>, &MemoryStore, StoreCheckpoint) -> bool;
        let lays: [(&str, Lay); 2] = [
            ("a restore", |objects, store, cut| {
                objects.restore_from(store, cut)
            }),
            ("a watch's snapshot", |objects, store, cut| {
                objects.watch_from(store, cut)
            }),
        ];
        for (name, lay) in lays {
            let (leader, standby, tables) = tabled();
            let lane = |id| tables.holds_lane(id, 0);
            // A row of the standby's own, which the cut drops.
            standby.upsert(asking(B, true));
            let dropped = standby.store.by_key(B).expect("its own row").id;
            assert!(lane(dropped), "{name}");
            leader.upsert(asking(A, true));
            leader.upsert(asking("http://c:8000", false));
            let cut = leader.store.checkpoint();
            assert!(lay(&standby.objects, &standby.store, cut), "{name}");
            let asks = leader.store.by_key(A).expect("the cut's row").id;
            assert!(lane(asks), "{name}: a row of the cut that asks");
            assert!(!lane(dropped), "{name}: a row the cut drops");
            assert_eq!(tables.lanes(), 1, "{name}: the quiet row holds none");
            // The replica hears one block of the row, and the leader's next
            // cut brings one row more: the row that stays keeps its block.
            let heard = stored_chain(&tables, asks, 1, &[1]);
            assert_eq!(heard, BatchOutcome::Applied, "{name}");
            assert_eq!(tables.indexed_blocks(asks, 0), 1, "{name}");
            leader.upsert(asking("http://d:8000", true));
            let next = leader.store.checkpoint();
            assert!(lay(&standby.objects, &standby.store, next), "{name}");
            assert_eq!(tables.lanes(), 2, "{name}: the row the next cut brings");
            let kept = tables.indexed_blocks(asks, 0);
            assert_eq!(
                kept, 1,
                "{name}: the lane of a row that stays was made again"
            );
        }
    }
}
