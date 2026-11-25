//! The holder's read side: a door reads the record that stands off the
//! worker objects.
//!
//! A door reads a record before it writes one: the row under a URL, the
//! row under an id, an engine's rows. The holder takes its turn
//! and reads the record back from the object that stands
//! ([`Objects::row_of`]). That is the record the holder's write decides
//! over ([`mod@super::write`]), so a door's test and the write behind it
//! read one record. The two lookups stand here once: the write side asks
//! them inside the turn it holds.
//!
//! The turn makes a door's read exact. A
//! writer of a row holds the turn from its read to its put, so a door that
//! waited behind it reads the object that writer left, and never the
//! registry one write behind the table.
//!
//! A worker object's status leads its row's ([`mod@super::status`]): a
//! record a door reads carries the status its object states, and no read
//! here asks the store's table for a row.
//!
//! An engine's rows are found by the holder's index of ranks. It lists a
//! row under the address it dials: every row that names the engine, and
//! the row under the engine's own URL that names none. The read keeps the
//! rows that name the engine, which is what a door asks.
//!
//! A first copy is cut here too: every row at one place of the journal,
//! for a watch's start, a standby's restore and an operator's read
//! ([`Objects::checkpoint`]), and the few rows of an operator's lone read
//! ([`Objects::rows_serving`]). A row's facts are read off its object in
//! the holder's turn. Its status is the one the journal told
//! ([`MemoryStore::told`]), not its object's: an object stands in a
//! status its log has not told yet between a driver's move and the tell,
//! and a copy holds the log's word at the copy's place. A cut takes no
//! object's mutation lock. A row a writer left with no turn of the holder
//! is in no cut until its host has the holder follow the store
//! ([`Objects::follow`]).

use parking_lot::MutexGuard;
use smg_worker::registry::Generation;

use super::{Laid, Objects, Place};
use crate::{
    engine::view::WorkerId,
    store::{memory::Told, MemoryStore, StoreCheckpoint, WorkerRow},
};

/// The row `laid` stands for, with the status the journal told of it.
/// `None` where the journal told none: the row left the store's table.
fn told_row(laid: &Laid, told: &Told) -> Option<WorkerRow> {
    let mut row = laid.remainder.row(laid.object.as_ref());
    row.status = told.status(row.id)?;
    Some(row)
}

#[cfg_attr(
    not(any(test, feature = "controller")),
    expect(dead_code, reason = "a controller's doors read through the holder")
)]
impl<G: Generation> Objects<G> {
    /// The record under the row id `id`, as a door reads it: in the
    /// holder's turn, off the object. `None` where no object stands for
    /// the row.
    pub(crate) fn record(&self, id: WorkerId) -> Option<WorkerRow> {
        let _turn = self.read_to.lock();
        self.row(id)
    }

    /// The record under the registration key `key`, as [`Self::record`]
    /// reads one.
    pub(crate) fn record_under(&self, key: &str) -> Option<WorkerRow> {
        let _turn = self.read_to.lock();
        self.row_under(key)
    }

    /// Every record whose facts name `engine`, as [`Self::record`] reads
    /// one: its plain row, or each rank that stands. In no order. Asked
    /// once a registration at a door: no decision reads it.
    pub(crate) fn records_of(&self, engine: &str) -> Vec<WorkerRow> {
        let _turn = self.read_to.lock();
        let dialing: Vec<WorkerId> = {
            let held = self.held.read();
            let ranked = held.ranks.get(engine).into_iter().flatten();
            ranked.map(|&(_, row)| row).collect()
        };
        dialing
            .into_iter()
            .filter_map(|row| self.row(row))
            .filter(|row| row.facts.engine == engine)
            .collect()
    }

    /// Every row at one place of the journal, cut from the worker objects:
    /// what a watch starts from, what a standby is restored from and what
    /// an operator reads whole. The rows come by registration key.
    ///
    /// In the holder's turn. Every
    /// writer of a row's facts holds that turn from its decision to the
    /// journal's telling, so the objects state each row the journal told
    /// and none it did not. A row's status is the one the journal told,
    /// read with the place and the epoch in one hold of the store's
    /// membership lock ([`MemoryStore::told`]). A status tell takes no
    /// turn of the holder: it changes no fact, and the copy holds its
    /// status where the place holds its event.
    ///
    /// The store's lock is held for the place and the statuses, and the
    /// holder's turn for the rows: a door's write and a standby's applier
    /// wait for the copy there, and no decision does.
    pub(crate) fn checkpoint(&self, store: &MemoryStore) -> StoreCheckpoint {
        let (turn, told) = self.told_turn(store);
        let mut rows: Vec<WorkerRow> = {
            let held = self.held.read();
            let laid = held.rows.values();
            laid.filter_map(|laid| told_row(laid, &told)).collect()
        };
        // Sorted after the turn: a writer waits for the copy alone.
        drop(turn);
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        StoreCheckpoint {
            at: told.at,
            epoch: told.epoch,
            rows,
        }
    }

    /// The first `limit` rows in key order of the ones that serve one of
    /// `models`, with the count that serve one; no model asked is every
    /// row. What an operator's read that tells a few rows takes, where
    /// [`Self::checkpoint`] cuts every row. Read as that cut is: in the
    /// holder's turn, each row with its told status. Every object's
    /// models are asked, and only the rows told are built. A model matches
    /// by the rule a watch and a shared snapshot filter by, an equal id
    /// (`serves`, in the operators' snapshot rules): the two change
    /// together. The rows name no write of the journal: no reader starts
    /// a tail from them.
    pub(crate) fn rows_serving(
        &self,
        store: &MemoryStore,
        models: &[String],
        limit: usize,
    ) -> (Vec<WorkerRow>, usize) {
        let mut asked: Vec<&str> = models.iter().map(String::as_str).collect();
        asked.sort_unstable();
        let serves = |laid: &&Laid| {
            let serves = |id: &str| asked.binary_search(&id).is_ok();
            asked.is_empty() || laid.object.any_model(&mut |card| serves(&card.id))
        };
        let (turn, told) = self.told_turn(store);
        let held = self.held.read();
        let mut serving: Vec<&Laid> = held.rows.values().filter(serves).collect();
        let count = serving.len();
        if limit < count {
            if let Some(last) = limit.checked_sub(1) {
                serving.select_nth_unstable_by(last, |a, b| a.object.url().cmp(b.object.url()));
            }
            serving.truncate(limit);
        }
        let mut rows: Vec<WorkerRow> = serving
            .into_iter()
            .filter_map(|laid| told_row(laid, &told))
            .collect();
        // Sorted after the turn, as the cut's rows are.
        drop(held);
        drop(turn);
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        (rows, count)
    }

    /// The holder's turn for a first copy, with the statuses the journal
    /// told: one read of `store`, inside the turn. Every writer of a row's
    /// facts holds the turn, so the objects state each row the journal
    /// told at the place this read takes, and no row of the store's table
    /// is read. A status tell, a generation's apply and a term's start
    /// take no turn: none changes a row's facts, and the told read takes
    /// the place in the hold that takes the statuses. A caller of the
    /// store's own entries for a row takes none either: no product code
    /// calls them, and a row written so is in no copy until its host has
    /// the holder follow the store ([`Self::follow`]).
    fn told_turn(&self, store: &MemoryStore) -> (MutexGuard<'_, Place>, Told) {
        let turn = self.read_to.lock();
        let told = store.told();
        (turn, told)
    }

    /// The row `id` as its object and what stands beside it state it. It
    /// takes no turn: a door's read and the holder's write ask it inside
    /// theirs.
    pub(super) fn row(&self, id: WorkerId) -> Option<WorkerRow> {
        self.row_of(&self.object(id)?)
    }

    /// The row under `key`, as [`Self::row`] reads one. Under the exact
    /// key, as the store's table finds a row: the registry's other lookup
    /// also finds a bare address under the form with its scheme, and that
    /// is another key.
    pub(super) fn row_under(&self, key: &str) -> Option<WorkerRow> {
        let object = self.registry.get_id_by_url_or_reservation(key)?;
        self.row_of(&object)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smg_worker::{registry::WorkerRegistry, worker::WorkerStatus as ObjectStatus};

    use super::{
        super::write::tests::{facts, Replica, A},
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
    /// holder left in the table is in no read until its host has the holder
    /// follow the store, and in each read behind that: by id, by key, among
    /// its engine's rows, in a first copy and in a lone read.
    #[test]
    fn a_row_written_around_the_holder_is_read_once_its_host_follows() {
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
            // The host's own follow, as a host that writes its store calls it.
            replica.objects.follow(&replica.store);
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
}
