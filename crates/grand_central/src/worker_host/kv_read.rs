//! What the engine over worker lists reads of a host's cache events: the
//! index the host's event streams write into its KV tables
//! ([`KvTables`]), lent through the engine's own traits. No block is copied,
//! and no stream is read twice.
//!
//! The engine names a worker by its address, and the tables name a row by
//! its id, which holds the registration's incarnation. So a worker's number
//! is found from its address through the registry and the holder's record
//! beside the object: the worker that stands under the address now, which
//! is the one a request to the address reaches. The registry is asked as
//! the holder asks it where it lends a list ([`Objects::lent`]).
//!
//! A row's number is its lane's place in the model's index, held from the
//! lane's making: a row that asks for cache events has its number ahead of
//! its stream, as a gateway's index numbers a worker where its stream's
//! task opens its place. The index holds no block of such a row until its
//! stream writes, and the engine reads a worker with no block as one the
//! record of sent prefixes speaks for. An index that holds no block is none
//! to the engine.
//!
//! # Locks
//!
//! A read here takes, one after the other and never one under another: the
//! registry's own read for the address's id, the holder's entry under its
//! read lock for the row beside the object, then the tables' table lock for
//! the row's place. No lane's lock is taken, so a read waits for no batch
//! a worker's stream writes. The holder's write turn takes its entry and
//! then the tables (the table's lock, then a lane's), never the other way,
//! and it asks no list and calls nothing of the engine. A decision holds no lock of the store while it reads here:
//! it decides over a loaded pointer. A served decision already makes the
//! first two reads, once a generation, where it asks the holder for its
//! lists ([`Objects::lent`]).
//!
//! # Limits
//!
//! - An index id is free for the next row once its row leaves, and the
//!   engine maps ids to the workers of a list once a table, which stands up
//!   to a second. An id taken again inside that second reads as the worker
//!   that held it when the table was read. A gateway's index has the same
//!   window: its own table stands a second, by age alone.
//! - The engine asks one index a list, by its first worker's model
//!   ([`CacheIndexes::index`]), and a row is indexed under its first model.

use std::sync::{Arc, Weak};

use smg_worker::registry::Generation;

use super::{kv_tables::LentIndex, KvTables, Objects};
use crate::engine::{overlap, CacheIndex, CacheIndexes};

/// A host's indexes of cache events as the engine over worker lists asks
/// for them: the tables' own, a model each.
pub(crate) struct EventIndexes<G> {
    tables: Arc<KvTables>,
    /// Held weakly: an engine keeps these for its life, and the holder
    /// holds the registry whose lists hold that engine's tables.
    objects: Weak<Objects<G>>,
}

impl<G: Generation> EventIndexes<G> {
    /// The indexes of `tables`, read by the addresses of the worker objects
    /// `objects` holds. `tables` are the ones that holder was handed
    /// ([`Objects::with_kv_tables`]): a host that keeps both hands both, so
    /// no reader is left with tables nothing writes.
    pub(crate) fn new(tables: Arc<KvTables>, objects: &Arc<Objects<G>>) -> Self {
        debug_assert!(
            objects
                .kv_tables()
                .is_some_and(|held| Arc::ptr_eq(&held, &tables)),
            "the tables lent are the holder's own"
        );
        Self {
            tables,
            objects: Arc::downgrade(objects),
        }
    }
}

impl<G: Generation> CacheIndexes for EventIndexes<G> {
    /// A lending of the model's index, made anew at each call: the engine
    /// asks once a table it reads. `None` while the index holds no block.
    fn index(&self, model: &str) -> Option<Arc<dyn CacheIndex>> {
        let lent = self.tables.lend(model)?;
        if lent.is_empty() {
            return None;
        }
        let objects = Weak::clone(&self.objects);
        Some(Arc::new(EventIndex { lent, objects }))
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.tables.block_size(model)
    }
}

/// One model's index, lent for one table of the engine's.
struct EventIndex<G> {
    lent: LentIndex,
    objects: Weak<Objects<G>>,
}

impl<G: Generation> CacheIndex for EventIndex<G> {
    fn worker_number(&self, address: &str) -> Option<u32> {
        let objects = self.objects.upgrade()?;
        // The registry first: no lock of the holder is held across it.
        let object = objects.registry().get_id_by_url(address)?;
        let row = objects.standing(&object)?.row;
        self.lent.number_of(row)
    }

    fn worker_blocks(&self, number: u32) -> usize {
        self.lent.blocks(number)
    }

    fn deepest_into(
        &self,
        block_hashes: &[u64],
        block_size: u32,
        within: &[u64],
        cap: usize,
        below: &mut dyn FnMut(u32) -> u32,
        kept: &mut Vec<(u32, u32)>,
    ) {
        let (chain, groups) = self.lent.chain();
        overlap::deepest_within_groups(
            chain,
            groups,
            block_hashes,
            block_size,
            within,
            cap,
            below,
            kept,
        );
    }
}

#[cfg(test)]
mod tests {
    use smg_worker::{kv_events::port::BatchOutcome, registry::WorkerRegistry};

    use super::*;
    use crate::{
        engine::view::WorkerId,
        store::{MemoryStore, Signals, WorkerFacts, WorkersStore},
        worker_host::{objects::stored_chain, Members},
    };

    const MODEL: &str = "m";
    const OTHER: &str = "another";
    const A: &str = "http://a:8000";
    const B: &str = "http://b:8000";
    const C: &str = "http://c:8000";

    /// A store, the holder of its worker objects with the tables it was
    /// handed, and the indexes lent of both.
    struct Bed {
        store: MemoryStore,
        objects: Arc<Objects<Members>>,
        tables: Arc<KvTables>,
        indexes: EventIndexes<Members>,
    }

    impl Bed {
        fn new() -> Self {
            let tables = Arc::new(KvTables::default());
            let registry = Arc::new(WorkerRegistry::<Members>::new());
            let objects = Arc::new(Objects::new(registry).with_kv_tables(Arc::clone(&tables)));
            let indexes = EventIndexes::new(Arc::clone(&tables), &objects);
            Self {
                store: MemoryStore::new(),
                objects,
                tables,
                indexes,
            }
        }

        /// A row at `key` that serves `model` and asks for cache events,
        /// stated to the holder.
        fn row(&self, key: &str, model: &str) -> WorkerId {
            let facts = WorkerFacts {
                key: key.to_owned(),
                engine: key.to_owned(),
                incarnation: 1,
                models: vec![model.to_owned()],
                dp_size: 1,
                signals: Signals {
                    events: true,
                    ..Signals::default()
                },
                ..WorkerFacts::default()
            };
            let id = self.store.upsert(facts).id();
            self.objects.follow(&self.store);
            id
        }

        /// The row `id` leaves, and the holder is told.
        fn leaves(&self, id: WorkerId) {
            let row = self.store.get(id).expect("the row stands");
            assert!(self.store.remove(id, row.revision).is_some());
            self.objects.follow(&self.store);
        }

        /// One batch that stores a chain of `contents` for the row `id`.
        fn stores(&self, id: WorkerId, contents: &[u64]) {
            let outcome = stored_chain(&self.tables, id, 1, contents);
            assert_eq!(outcome, BatchOutcome::Applied);
        }

        /// The model's index as one table of the engine's is lent it.
        fn index(&self) -> Arc<dyn CacheIndex> {
            self.indexes.index(MODEL).expect("the model's index")
        }
    }

    /// The deepest holders of a request of `hashes` among the numbers
    /// `within` names, `cap` at most, as `index` answers a table's lookup.
    fn deepest(index: &dyn CacheIndex, hashes: &[u64], within: u64, cap: usize) -> Vec<(u32, u32)> {
        let mut kept = Vec::new();
        index.deepest_into(hashes, 1, &[within], cap, &mut |_| 0, &mut kept);
        kept.sort_unstable();
        kept
    }

    /// What a worker's stream stored is named under its number, as deep as
    /// its chain holds the request, within the numbers and the cap a table
    /// asks. A gateway's index answers a table by the same lookup
    /// (`model_gateway/src/policies/kv_index/index_backend.rs:471`).
    #[test]
    fn a_stored_chain_is_named_under_its_workers_number_within_the_cap() {
        let bed = Bed::new();
        let (a, b) = (bed.row(A, MODEL), bed.row(B, MODEL));
        bed.stores(a, &[1, 2, 3]);
        bed.stores(b, &[1, 2]);
        let index = bed.index();
        let first = index.worker_number(A).expect("a's number");
        let second = index.worker_number(B).expect("b's number");
        assert_ne!(first, second);
        assert_eq!(index.worker_blocks(first), 3);
        assert_eq!(index.worker_blocks(second), 2);
        let request = [1, 2, 3, 4];
        let both = (1 << first) | (1 << second);
        let mut whole = vec![(first, 3), (second, 2)];
        whole.sort_unstable();
        assert_eq!(deepest(&*index, &request, both, 8), whole);
        // The numbers a table names bound the lookup, and so does its cap.
        assert_eq!(deepest(&*index, &request, 1 << second, 8), [(second, 2)]);
        assert_eq!(deepest(&*index, &request, both, 1), [(first, 3)]);
    }

    /// An index that holds no block is none to the engine, as where a
    /// gateway lends its own
    /// (`model_gateway/src/policies/kv_index/event_monitor.rs:349-352`). In
    /// an index that holds a block, a row whose lane heard no batch has its
    /// number and no block: its place is made where the holder states it,
    /// as a gateway's index numbers a worker where its stream's task opens
    /// its place
    /// (`model_gateway/src/policies/kv_index/event_monitor/index_sink.rs:186-196`).
    /// The engine reads a numbered worker with no block as the record's of
    /// sent prefixes: its own cases hold that
    /// (`the_index_speaks_for_its_workers_and_the_record_for_the_others`).
    #[test]
    fn a_row_with_a_place_and_no_batch_has_its_number_and_no_block() {
        let bed = Bed::new();
        let (a, b) = (bed.row(A, MODEL), bed.row(B, MODEL));
        assert!(bed.indexes.index(MODEL).is_none(), "no block yet");
        bed.stores(a, &[1]);
        let index = bed.index();
        let quiet = index.worker_number(B).expect("its place stands");
        assert_eq!(index.worker_blocks(quiet), 0);
        assert_eq!(bed.tables.cursor(b, 0), None, "no batch ran through it");
        assert_eq!(deepest(&*index, &[1], 1 << quiet, 8), []);
    }

    /// The limit the module names. A number is free for the next row once
    /// its row leaves, and a lending keeps no count of takes: a lending
    /// made before a worker left reads, under that worker's number, the
    /// blocks of the row that took the number since. A gateway's own table
    /// has the same window: it stands a second by age alone
    /// (`model_gateway/src/policies/cache_aware.rs:165`, `:1146-1147`).
    #[test]
    fn a_number_taken_again_reads_as_the_row_that_took_it_through_an_older_lending() {
        let bed = Bed::new();
        let (a, b) = (bed.row(A, MODEL), bed.row(B, MODEL));
        bed.stores(a, &[1, 2]);
        bed.stores(b, &[9]);
        let older = bed.index();
        let number = older.worker_number(A).expect("a's number");
        bed.leaves(a);
        assert_eq!(older.worker_blocks(number), 0, "its blocks left with it");
        let c = bed.row(C, MODEL);
        bed.stores(c, &[5, 6, 7]);
        let newer = bed.index();
        assert_eq!(newer.worker_number(C), Some(number), "the freed number");
        assert_eq!(newer.worker_number(A), None);
        // The older lending reads the new row's blocks under the number it
        // gave the worker that left.
        assert_eq!(older.worker_blocks(number), 3);
        assert_eq!(deepest(&*older, &[5, 6, 7], 1 << number, 8), [(number, 3)]);
    }

    /// A row has a number in the index its lane writes, and in no other: a
    /// row of another model has none in this model's, an address nobody
    /// registered has none, and a row that left has none. A model whose
    /// last indexed row left has no index.
    #[test]
    fn a_row_of_another_model_and_a_row_that_left_have_no_number() {
        let bed = Bed::new();
        let (a, b) = (bed.row(A, MODEL), bed.row(B, OTHER));
        bed.stores(a, &[1]);
        bed.stores(b, &[1]);
        let index = bed.index();
        assert!(index.worker_number(A).is_some());
        assert_eq!(index.worker_number(B), None, "another model's lane");
        assert_eq!(index.worker_number("http://nobody:8000"), None);
        bed.leaves(a);
        assert_eq!(index.worker_number(A), None);
        assert!(bed.indexes.index(MODEL).is_none(), "its last row left");
        assert!(bed.indexes.index(OTHER).is_some());
    }

    /// A lent index keeps the holder of the worker objects alive no longer
    /// than its host does: an engine's table holds the index, and the
    /// holder's registry holds the list that table stands in.
    #[test]
    fn a_lent_index_holds_the_worker_objects_weakly() {
        let bed = Bed::new();
        let a = bed.row(A, MODEL);
        bed.stores(a, &[1]);
        let index = bed.index();
        let number = index.worker_number(A).expect("a's number");
        let Bed {
            objects, indexes, ..
        } = bed;
        let held = Arc::downgrade(&objects);
        drop(objects);
        assert!(held.upgrade().is_none(), "nothing lent holds the holder");
        assert_eq!(index.worker_number(A), None, "no holder is left to ask");
        assert_eq!(index.worker_blocks(number), 1, "the tables' own count");
        drop(indexes);
    }

    /// The block size is the tables' of record, and none for a model they
    /// hold none of.
    #[test]
    fn the_block_size_is_the_tables_of_record() {
        let bed = Bed::new();
        assert_eq!(bed.indexes.block_size(MODEL), None);
        bed.tables.set_block_size(MODEL, 16);
        assert_eq!(bed.indexes.block_size(MODEL), Some(16));
        assert_eq!(bed.indexes.block_size(OTHER), None);
    }
}
