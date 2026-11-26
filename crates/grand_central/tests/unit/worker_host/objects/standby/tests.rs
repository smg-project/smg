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
        write::tests::{facts, refresh, Replica, A},
    },
    *,
};
use crate::{
    store::{memory::seam, Signals, WorkerFacts, WorkerStatus, WorkersStore},
    worker_host::{KvTables, Members},
};

const B: &str = "http://b:8000";

/// One of the two entries that lay a cut whole.
type Lay = fn(&Objects<Members>, &MemoryStore, StoreCheckpoint) -> bool;

/// The two entries that lay a cut whole, each with its name.
const LAYS: [(&str, Lay); 2] = [
    ("a restore", |objects, store, cut| {
        objects.restore_from(store, cut)
    }),
    ("a watch's snapshot", |objects, store, cut| {
        objects.watch_from(store, cut)
    }),
];

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
fn hand(standby: &Replica, ahead: u64, change: MembershipChange, row: Option<WorkerRow>) -> bool {
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

/// A cut laid whole takes the holder's turn ahead of the store's write,
/// by a restore and by a watch's snapshot. While another holds the turn
/// the call waits there, and the store stands as it was. Behind the
/// turn the cut is laid, and its row has its object.
#[test]
fn a_cut_laid_whole_waits_for_the_holders_turn() {
    for (name, lay) in LAYS {
        let (leader, standby) = (Replica::new(), Replica::new());
        leader.upsert(facts(1, "gold"));
        let cut = leader.store.checkpoint();
        let (store, objects) = (&standby.store, &standby.objects);
        let (laid, done) = mpsc::channel();
        let turn = objects.turn.lock();
        thread::scope(|scope| {
            let call = scope.spawn(move || {
                let took = lay(objects, store, cut);
                let _ = laid.send(());
                took
            });
            assert_eq!(
                done.recv_timeout(Duration::from_millis(200)),
                Err(mpsc::RecvTimeoutError::Timeout),
                "{name}"
            );
            assert_eq!(store.by_key(A), None, "{name}: the cut waits");
            drop(turn);
            assert!(call.join().expect("the call ends"), "{name}");
        });
        assert!(standby.object(A).is_some(), "{name}: its row's object");
    }
}

/// A watch's snapshot tells a row's age too. A row whose object stands
/// keeps the instant it was admitted at, as it stood and with facts
/// changed under its id. The table's row and the row read back from
/// the object state the first instant. A restore keeps none: its row
/// takes the cut's.
#[test]
fn a_watch_snapshots_row_keeps_the_instant_its_object_holds() {
    let (told, watching) = (Replica::new(), Replica::new());
    told.upsert(facts(1, "gold"));
    let (store, objects) = (&watching.store, &watching.objects);
    assert!(objects.watch_from(store, told.store.checkpoint()));
    let admitted = store.by_key(A).expect("the snapshot's row").admitted_at;
    let later = || {
        let mut cut = told.store.checkpoint();
        cut.rows[0].admitted_at += Duration::from_secs(5);
        cut
    };
    let instants = || {
        let object = watching.object(A).expect("the row's object");
        let read = objects.row_of(&object).expect("its row");
        (read.admitted_at, store.by_key(A).map(|row| row.admitted_at))
    };
    // The row as it stood: its object stays, and its instant.
    assert!(objects.watch_from(store, later()));
    assert_eq!(instants(), (admitted, Some(admitted)));
    // Facts changed under the row's id: a new object in its place.
    told.upsert(facts(1, "iron"));
    assert!(objects.watch_from(store, later()));
    assert_eq!(instants(), (admitted, Some(admitted)));
    let cut = later();
    let stated = cut.rows[0].admitted_at;
    assert!(objects.restore_from(store, cut));
    assert_eq!(instants(), (stated, Some(stated)));
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
/// ahead of the store's word, so the row is laid again as it was read
/// off the object, and has an object when the event's write returns.
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

/// A removal the store does not take lays the row read off its object,
/// and not the one the store holds under the id. The standby's table is
/// handed a patch of the row around the holder, as a decided change, so
/// the two rows differ. A removal out of place takes the object out and
/// lays it again from the row it stated.
#[test]
fn a_removal_the_store_does_not_take_lays_the_row_read_off_its_object() {
    let (leader, standby) = (Replica::new(), Replica::new());
    leader.upsert(facts(1, "gold"));
    follow(&standby, &leader);
    let row = standby.one_row(A);
    let mut around = row.clone();
    around.facts = facts(1, "iron");
    around.revision += 1;
    let epoch = standby.store.epoch() + 1;
    let patch = MembershipChange::Relabelled { id: row.id };
    standby.store.follow(at(epoch), epoch, patch, Some(around));
    let table = standby.store.get(row.id).expect("the row stands");
    assert_eq!(table.facts, facts(1, "iron"), "the table's row moved");
    let removal = MembershipChange::Removed { id: row.id };
    assert!(!hand(&standby, 2, removal, None), "out of place");
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

/// A told change that takes a row out tells the standby's own log that
/// row in the status the standby's table holds, by a removal and by a
/// key's next life. A probe of a term that ended moved the object ahead
/// of the table: the row read back states the object's status, and the
/// log is told the table's.
#[test]
fn a_row_a_told_change_takes_out_is_told_in_the_status_the_table_holds() {
    for next_life in [false, true] {
        let (leader, standby) = (Replica::new(), Replica::new());
        leader.upsert(facts(1, "gold"));
        follow(&standby, &leader);
        let id = standby.object(A).expect("the row's object");
        let registry = standby.objects.registry();
        let revision = registry.get(&id).expect("it stands").revision();
        let ready = |_: &Arc<dyn Worker>| ((), Some(ObjectStatus::Ready));
        registry.apply_if_revision(&id, revision, ready);
        let stood = standby.store.by_key(A).expect("the row stands");
        assert_eq!(stood.status, WorkerStatus::Pending, "the table's word");
        let read = standby.objects.row_of(&id).expect("the object's row");
        assert_eq!(read.status, WorkerStatus::Ready, "the object's word");
        if next_life {
            leader.upsert(facts(2, "gold"));
        } else {
            let row = leader.one_row(A);
            leader.objects.remove(&leader.store, row.id, row.revision);
        }
        follow(&standby, &leader);
        let (_, _, left) = standby.last();
        assert_eq!(left, Some(stood), "a next life: {next_life}");
    }
}

/// While the table stands, a debug build holds the row a standby's
/// holder hands to the row its table held, as on a leader. Here another
/// writer of the registry set the object's models behind its build: the
/// told removal that hands the row read back stops.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "a row its holder read back is the row the table held")]
fn a_told_removal_that_hands_a_row_the_table_does_not_hold_stops_a_debug_build() {
    let (leader, standby) = (Replica::new(), Replica::new());
    leader.upsert(facts(1, "gold"));
    follow(&standby, &leader);
    refresh(&standby);
    let row = leader.one_row(A);
    leader.objects.remove(&leader.store, row.id, row.revision);
    follow(&standby, &leader);
}

/// The same for a told next life of a key, which hands the old life's
/// row.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "a row its holder read back is the row the table held")]
fn a_told_next_life_that_hands_a_row_the_table_does_not_hold_stops_a_debug_build() {
    let (leader, standby) = (Replica::new(), Replica::new());
    leader.upsert(facts(1, "gold"));
    follow(&standby, &leader);
    refresh(&standby);
    leader.upsert(facts(2, "gold"));
    follow(&standby, &leader);
}

/// A gateway's watcher follows through the same entry, and its store
/// tells no entry: the row its holder hands is held to nothing there.
/// Its refresh of a worker's models sets a fact of the object behind
/// the build, and a told next life and a told removal are taken behind
/// it, in a debug build too.
#[test]
fn a_host_that_tells_no_entry_takes_a_told_change_behind_a_refresh() {
    let (leader, watcher) = (Replica::new(), Replica::untold());
    leader.upsert(facts(1, "gold"));
    follow(&watcher, &leader);
    refresh(&watcher);
    leader.upsert(facts(2, "gold"));
    let (change, row) = told(&leader);
    assert!(hand(&watcher, 1, change, row), "a next life");
    refresh(&watcher);
    let row = leader.one_row(A);
    leader.objects.remove(&leader.store, row.id, row.revision);
    let (change, row) = told(&leader);
    assert!(hand(&watcher, 1, change, row), "a removal");
    assert!(watcher.objects.is_empty());
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
    for (name, lay) in LAYS {
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
    for (name, lay) in LAYS {
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
