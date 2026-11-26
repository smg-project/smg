use std::sync::Arc;

use smg_worker::{registry::WorkerRegistry, worker::WorkerStatus as ObjectStatus};

use super::{
    super::write::tests::{facts, lay_standing, Replica, A},
    *,
};
use crate::{
    store::{identity, Registrant, WorkerFacts, WorkerPatch, WorkerStatus, WorkersStore},
    worker_host::Members,
};

/// The row `id` as a door reads it, held against the table's by id and
/// by key.
fn read(replica: &Replica, id: WorkerId) -> Option<WorkerRow> {
    let (store, objects) = (&replica.store, &replica.objects);
    let by_id = objects.record(id);
    assert_eq!(by_id, store.get(id), "by id against the table");
    let by_key = objects.record_under(A);
    assert_eq!(by_key, store.by_key(A), "by key against the table");
    by_id
}

/// A door reads off the object the row each write left in the table: a
/// registration, a change in place, a patch, a key's next life and a
/// removal, by id and by key.
#[test]
fn a_door_reads_the_row_each_write_left() {
    let replica = Replica::new();
    let (store, objects) = (&replica.store, &replica.objects);
    let (first, second) = (identity(A, 1), identity(A, 2));
    replica.upsert(facts(1, "gold"));
    assert_eq!(read(&replica, first).map(|row| row.revision), Some(1));
    replica.upsert(facts(1, "iron"));
    assert_eq!(read(&replica, first).map(|row| row.revision), Some(2));
    let patch = WorkerPatch {
        priority: Some(7),
        ..WorkerPatch::default()
    };
    objects.patch(store, first, patch, 2).expect("a patch");
    assert_eq!(read(&replica, first).map(|row| row.facts.priority), Some(7));
    replica.upsert(facts(2, "iron"));
    assert_eq!(read(&replica, first), None, "the old life left");
    assert_eq!(read(&replica, second).map(|row| row.revision), Some(1));
    objects.remove(store, second, 1).expect("a removal");
    assert_eq!(read(&replica, second), None);
}

/// No read starts with a read of the store. A row a writer outside the
/// holder left in the table is in no read until the holder lays it,
/// here behind a refused write of its key.
/// It is in each read behind that: by id, by key, among
/// its engine's rows, in a first copy and in a lone read.
#[test]
fn a_row_written_around_the_holder_is_read_once_the_holder_lays_it() {
    type Read = fn(&Replica) -> Option<WorkerRow>;
    let reads: [(&str, Read); 5] = [
        ("a first copy", |replica| {
            replica.objects.checkpoint(&replica.store).rows.pop()
        }),
        ("a lone read", |replica| {
            let (store, objects) = (&replica.store, &replica.objects);
            objects.rows_serving(store, &[], 1).0.pop()
        }),
        ("by id", |replica| replica.objects.record(identity(A, 1))),
        ("by key", |replica| replica.objects.record_under(A)),
        ("by engine", |replica| replica.objects.records_of(A).pop()),
    ];
    for (name, read) in reads {
        let replica = Replica::new();
        // The store's own write, which takes no turn of the holder.
        replica
            .store
            .upsert_by(facts(1, "gold"), Registrant::Control);
        assert_eq!(read(&replica), None, "{name}: no object, so no row");
        // The key's next write through the holder is refused, and its
        // read lays the row.
        lay_standing(&replica.objects, &replica.store);
        let row = read(&replica);
        assert!(row.is_some(), "{name}");
        assert_eq!(row, replica.store.by_key(A), "{name}");
    }
}

/// An engine's rows by the holder's index of ranks are the rows a pass
/// over the table gives: each row whose facts name the engine, a plain
/// row beside its ranks among them. A row that dials the engine's URL
/// and names no engine is not one of them.
#[test]
fn an_engines_rows_are_the_rows_that_name_it() {
    const E: &str = "http://e:8000";
    let replica = Replica::new();
    let (store, objects) = (&replica.store, &replica.objects);
    let stated = |key: String, engine: &str, rank: u16, width: u16| WorkerFacts {
        key,
        engine: engine.to_owned(),
        dp_rank: rank,
        dp_size: width,
        ..facts(1, "gold")
    };
    // The engine whole beside two of its ranks, and a row under
    // another URL that names no engine.
    replica.upsert(stated(E.to_owned(), E, 0, 1));
    replica.upsert(stated(format!("{E}@0"), E, 0, 2));
    replica.upsert(stated(format!("{E}@1"), E, 1, 2));
    replica.upsert(stated(A.to_owned(), "", 0, 1));
    let by_key = |mut rows: Vec<WorkerRow>| {
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        rows
    };
    let pass = |engine: &str| {
        let (_, mut rows) = store.all_rows();
        rows.retain(|row| row.facts.engine == engine);
        by_key(rows)
    };
    let read = |engine: &str| by_key(objects.records_of(engine));
    assert_eq!(read(E), pass(E));
    assert_eq!(read(E).len(), 3, "the plain row and two ranks");
    assert_eq!(read(A), pass(A));
    assert!(read(A).is_empty(), "the row under it names no engine");
}

/// A record a door reads carries the status its object states, by id,
/// by key and among its engine's records: here an object the holder
/// readied, and then a status written into the table around the
/// object, which no read carries.
#[test]
fn a_record_carries_the_status_its_object_states() {
    let store = MemoryStore::default();
    let objects: Objects<Members> = Objects::new(Arc::new(WorkerRegistry::new()));
    let change = objects.upsert(&store, facts(1, "gold"), Registrant::Control);
    let id = change.expect("registered").id();
    objects.connect_row(&store, id, 1);
    let read = || {
        let by_id = objects.record(id);
        let by_key = objects.record_under(A);
        let by_engine = objects.records_of(A).pop();
        [by_id, by_key, by_engine].map(|row| row.map(|row| row.status))
    };
    assert_eq!(read(), [Some(WorkerStatus::Ready); 3]);
    let told = store.get(id).map(|row| row.status);
    assert_eq!(told, Some(WorkerStatus::Ready), "the table is told");
    // A status written into the table around the object.
    store
        .set_status(id, WorkerStatus::NotReady, 1)
        .expect("a status");
    assert_eq!(read(), [Some(WorkerStatus::Ready); 3]);
}

/// A first copy cut from the objects states what the store's own cut
/// of its table states: the rows by key, the epoch and the place.
/// Behind a registration, a change in place, a patch, a told status, a
/// key's next life, a second key and a removal.
#[test]
fn a_first_copy_states_the_rows_the_table_holds() {
    const B: &str = "http://b:8000";
    let replica = Replica::new();
    let (store, objects) = (&replica.store, &replica.objects);
    let same = |what: &str| {
        let (cut, table) = (objects.checkpoint(store), store.checkpoint());
        assert_eq!((cut.at, cut.epoch), (table.at, table.epoch), "{what}");
        assert_eq!(cut.rows, table.rows, "{what}");
        cut.rows.len()
    };
    let (first, second) = (identity(A, 1), identity(A, 2));
    assert_eq!(same("an empty store"), 0);
    replica.upsert(facts(1, "gold"));
    assert_eq!(same("a registration"), 1);
    replica.upsert(facts(1, "iron"));
    same("a change in place");
    let patch = WorkerPatch {
        priority: Some(7),
        ..WorkerPatch::default()
    };
    objects.patch(store, first, patch, 2).expect("a patch");
    same("a patch");
    objects.connect_row(store, first, 3);
    same("a told status");
    replica.upsert(facts(2, "iron"));
    same("a key's next life");
    replica.upsert(WorkerFacts {
        key: B.to_owned(),
        engine: B.to_owned(),
        ..facts(1, "gold")
    });
    assert_eq!(same("a second key"), 2);
    objects.remove(store, second, 1).expect("a removal");
    assert_eq!(same("a removal"), 1);
}

/// A first copy carries the status the journal told of a row, where a
/// door's read carries its object's. An object stands ahead of the
/// journal between a driver's move and the tell: here a move with no
/// tell behind it. Every row's copy and a lone read's alike.
#[test]
fn a_first_copy_carries_the_status_the_journal_told() {
    let replica = Replica::new();
    let (store, objects) = (&replica.store, &replica.objects);
    replica.upsert(facts(1, "gold"));
    let id = identity(A, 1);
    objects.connect_row(store, id, 1);
    let copies = || {
        let cut = objects.checkpoint(store).rows;
        let (lone, serving) = objects.rows_serving(store, &["m".to_owned()], 1);
        assert_eq!(serving, 1);
        [cut, lone].map(|rows| rows.first().map(|row| row.status))
    };
    assert_eq!(copies(), [Some(WorkerStatus::Ready); 2]);
    // A driver's move, not told.
    let object = replica.object(A).expect("the row's object");
    let object = objects.registry().get(&object).expect("it stands");
    object.set_status(ObjectStatus::NotReady);
    let door = objects.record(id).map(|row| row.status);
    assert_eq!(door, Some(WorkerStatus::NotReady), "the object's");
    assert_eq!(copies(), [Some(WorkerStatus::Ready); 2], "the journal's");
}

/// The two holds of a first copy, measured over 512 rows. The store's
/// membership lock: held by the store's own cut for a copy of every
/// row (the call bar its sort), and by the told read for a place and
/// one status a row. The holder's turn: held by the holder's cut for
/// the call bar its sort, the told read inside it.
#[test]
#[ignore = "a measurement for the Numbers table, run by name"]
fn the_two_holds_of_a_first_copy_measured() {
    use std::{hint::black_box, io::Write as _, time::Instant};

    const ROWS: usize = 512;
    const RUNS: u32 = 2_000;
    let replica = Replica::new();
    let (store, objects) = (&replica.store, &replica.objects);
    for n in 0..ROWS {
        let key = format!("http://w{n}:8000");
        replica.upsert(WorkerFacts {
            key: key.clone(),
            engine: key,
            ..facts(1, "gold")
        });
    }
    assert_eq!(objects.checkpoint(store).rows.len(), ROWS);
    let mut out = std::io::stdout();
    let mut timed = |what: &str, run: &dyn Fn()| {
        let since = Instant::now();
        for _ in 0..RUNS {
            run();
        }
        let micros = since.elapsed().as_micros() / u128::from(RUNS);
        writeln!(out, "{ROWS} rows, {what}: {micros} us a call").expect("stdout");
    };
    timed("the store's own cut of its table", &|| {
        black_box(store.checkpoint());
    });
    timed("the told read", &|| {
        black_box(store.told());
    });
    timed("the holder's cut", &|| {
        black_box(objects.checkpoint(store));
    });
}
