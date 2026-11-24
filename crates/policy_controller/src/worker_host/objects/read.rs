//! The holder's read side: a door reads the record that stands off the
//! worker objects.
//!
//! A door reads a record before it writes one: the row under a URL, the
//! row under an id, an engine's rows. The holder takes its turn, reads the
//! store to its end, and reads the record back from the object that stands
//! ([`Objects::row_of`]). That is the record the holder's write decides
//! over ([`mod@super::write`]), so a door's test and the write behind it
//! read one record. The two lookups stand here once: the write side asks
//! them inside the turn it holds.
//!
//! The turn and the read to the store's end make a door's read exact. A
//! writer of a row holds the turn from its read to its put, so a door that
//! waited behind it reads the object that writer left, and never the
//! registry one write behind the table.
//!
//! A status is the table's word until a worker object's leads: a drain and
//! a connect signal write the table, and an object trails it by the status
//! writes its holder has not followed. So a record a door reads carries
//! the status its table holds ([`Objects::with_table_status`]), read by
//! the store's own read of a row. That read goes when an object's status
//! leads: the helper is then the record as the object states it.
//!
//! An engine's rows are found by the holder's index of ranks. It lists a
//! row under the address it dials: every row that names the engine, and
//! the row under the engine's own URL that names none. The read keeps the
//! rows that name the engine, which is what a door asks.

use parking_lot::MutexGuard;
use smg_worker::registry::Generation;

use super::{Objects, Place};
use crate::{
    engine::view::WorkerId,
    store::{MemoryStore, WorkerRow, WorkersStore},
};

#[cfg_attr(
    not(any(test, feature = "controller")),
    expect(dead_code, reason = "a controller's doors read through the holder")
)]
impl<G: Generation> Objects<G> {
    /// The record under the row id `id`, as a door reads it: in the
    /// holder's turn, behind a read of `store` to its end. `None` where no
    /// object stands for the row.
    pub(crate) fn record(&self, store: &MemoryStore, id: WorkerId) -> Option<WorkerRow> {
        let _turn = self.turn(store);
        Self::with_table_status(store, self.row(id)?)
    }

    /// The record under the registration key `key`, as [`Self::record`]
    /// reads one.
    pub(crate) fn record_under(&self, store: &MemoryStore, key: &str) -> Option<WorkerRow> {
        let _turn = self.turn(store);
        Self::with_table_status(store, self.row_under(key)?)
    }

    /// Every record whose facts name `engine`, as [`Self::record`] reads
    /// one: its plain row, or each rank that stands. In no order. Asked
    /// once a registration at a door: no decision reads it.
    pub(crate) fn records_of(&self, store: &MemoryStore, engine: &str) -> Vec<WorkerRow> {
        let _turn = self.turn(store);
        let dialing: Vec<WorkerId> = {
            let held = self.held.read();
            let ranked = held.ranks.get(engine).into_iter().flatten();
            ranked.map(|&(_, row)| row).collect()
        };
        dialing
            .into_iter()
            .filter_map(|row| self.row(row))
            .filter(|row| row.facts.engine == engine)
            .filter_map(|row| Self::with_table_status(store, row))
            .collect()
    }

    /// The holder's turn, with `store` read to its end: where a door's
    /// read starts, as the holder's write does.
    fn turn(&self, store: &MemoryStore) -> MutexGuard<'_, Place> {
        let mut place = self.read_to.lock();
        self.follow_from(store, &mut place);
        place
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

    /// `row` with the status the table holds, and `None` where the table
    /// holds no row under its id. A status is the table's word until an
    /// object's leads. Then this is `Some(row)`: the status the object
    /// states.
    fn with_table_status(store: &MemoryStore, mut row: WorkerRow) -> Option<WorkerRow> {
        row.status = store.get(row.id)?.status;
        Some(row)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smg_worker::registry::WorkerRegistry;

    use super::{
        super::write::tests::{facts, Replica, A},
        *,
    };
    use crate::{
        store::{identity, Registrant, WorkerFacts, WorkerPatch, WorkerStatus},
        worker_host::Members,
    };

    /// The row `id` as a door reads it, held against the table's by id and
    /// by key.
    fn read(replica: &Replica, id: WorkerId) -> Option<WorkerRow> {
        let (store, objects) = (&replica.store, &replica.objects);
        let by_id = objects.record(store, id);
        assert_eq!(by_id, store.get(id), "by id against the table");
        let by_key = objects.record_under(store, A);
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

    /// Each read takes the holder's turn and reads the store to its end
    /// first: a row a writer outside the holder left in the table is read
    /// by id, by key and among its engine's rows, with no follow of the
    /// case's own.
    #[test]
    fn each_read_follows_a_writer_outside_the_holder() {
        type Read = fn(&Replica) -> Option<WorkerRow>;
        let reads: [(&str, Read); 3] = [
            ("by id", |replica| {
                let id = identity(A, 1);
                replica.objects.record(&replica.store, id)
            }),
            ("by key", |replica| {
                replica.objects.record_under(&replica.store, A)
            }),
            ("by engine", |replica| {
                replica.objects.records_of(&replica.store, A).pop()
            }),
        ];
        for (name, read) in reads {
            let replica = Replica::new();
            // The store's own write, which takes no turn of the holder.
            replica
                .store
                .upsert_by(facts(1, "gold"), Registrant::Control);
            assert!(replica.objects.is_empty(), "{name}: no object yet");
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
        let read = |engine: &str| by_key(objects.records_of(store, engine));
        assert_eq!(read(E), pass(E));
        assert_eq!(read(E).len(), 3, "the plain row and two ranks");
        assert_eq!(read(A), pass(A));
        assert!(read(A).is_empty(), "the row under it names no engine");
    }

    /// A record a door reads carries the status the table holds, whatever
    /// the object states: here a holder that follows no status, whose
    /// object stays pending behind a status the table took.
    #[test]
    fn a_record_carries_the_status_the_table_holds() {
        let store = MemoryStore::default();
        let objects: Objects<Members> = Objects::new(Arc::new(WorkerRegistry::new()));
        let change = objects.upsert(&store, facts(1, "gold"), Registrant::Control);
        let id = change.expect("registered").id();
        store
            .set_status(id, WorkerStatus::Ready, 1)
            .expect("a status");
        let object = objects.object(id).expect("the object");
        let stated = objects.row_of(&object).expect("the object's row");
        assert_eq!(
            stated.status,
            WorkerStatus::Pending,
            "the object is not told"
        );
        let record = objects.record(&store, id).expect("the record");
        assert_eq!(record.status, WorkerStatus::Ready);
        assert_eq!(
            Some(record),
            store.get(id),
            "the table's row, status and all"
        );
    }
}
