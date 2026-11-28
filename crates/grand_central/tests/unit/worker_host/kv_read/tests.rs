use smg_worker::{
    identity::basic::Worker, kv_events::port::BatchOutcome, registry::WorkerRegistry,
    test_support::ready_worker,
};

use super::*;
use crate::{
    document::spec::{AppliedSpec, PolicySpec},
    engine::{
        decision::{Decision, Shed, ShedReason, Trace},
        view::{AffinityHolder, RequestFacts, WorkerId},
        Affinity, PoolAffinity,
    },
    store::{MemoryStore, Registrant, Signals, WorkerFacts},
    worker_host::{objects::tests::stored_chain, Members},
};

const MODEL: &str = "m";
const OTHER: &str = "another";
const A: &str = "http://a:8000";
const B: &str = "http://b:8000";
const C: &str = "http://c:8000";
/// The instant every request of a case is decided at, in microseconds.
const NOW: u64 = 10_000_000;

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
        let written = self.objects.upsert(&self.store, facts, Registrant::Control);
        written.expect("the store takes the write").id()
    }

    /// The row `id` leaves, and the holder is told.
    fn leaves(&self, id: WorkerId) {
        let row = self.store.get(id).expect("the row stands");
        let removed = self.objects.remove(&self.store, id, row.revision);
        assert!(removed.is_some());
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

    /// The engine's list of the workers at `A`, `B` and `C`, in that
    /// order, under the cache-aware preset: lent the tables' indexes as
    /// a controller lends them to its engines.
    fn list(&self) -> List {
        let affinity = Affinity::default();
        let indexes = EventIndexes::new(Arc::clone(&self.tables), &self.objects);
        affinity.set_indexes(Arc::new(indexes));
        let mut document = PolicySpec::preset("cache_aware").expect("the preset");
        document.generation = 1;
        List {
            affinity,
            pool: PoolAffinity::default(),
            spec: AppliedSpec::validate(&document).expect("the preset validates"),
            workers: [A, B, C].map(|address| ready_worker(address, MODEL)).into(),
            ids: (1..=3).map(WorkerId).collect(),
        }
    }
}

/// One list of the engine's and what the engine keeps of it.
struct List {
    affinity: Affinity,
    pool: PoolAffinity,
    spec: AppliedSpec,
    workers: Vec<Arc<dyn Worker>>,
    ids: Vec<WorkerId>,
}

impl List {
    /// The holders the engine names a request of `hashes`: each one's
    /// place in the list and its blocks in the index. Every request is
    /// decided at one instant, so the table one request read stands for
    /// the next by its age.
    fn holders(&self, hashes: &[u64]) -> Vec<(u32, u32)> {
        let mut facts = RequestFacts {
            model: MODEL,
            block_hashes: hashes,
            now_micros: NOW,
            ..RequestFacts::default()
        };
        let held = |holder: &AffinityHolder| (holder.row, holder.overlap_blocks);
        let mut named = Vec::new();
        overlap::decide(
            &self.affinity,
            &self.pool,
            &self.spec,
            &self.workers,
            &self.ids,
            &mut facts,
            None,
            |_| None,
            || None,
            |_| true,
            |request| {
                let rows = request.named.expect("the preset is named its rows");
                named = rows.holders.iter().map(held).collect();
                Decision::Shed(Shed {
                    reason: ShedReason::PolicyDeclined,
                    emptied_by: "a case",
                    retry_after_secs: 0,
                    wait_allowed: false,
                    side: None,
                    trace: Trace::default(),
                })
            },
        );
        named
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

/// An index that holds no block is lent and says so itself: the engine
/// reads it as none, as where a gateway lends its own
/// (`KvEventMonitor::index`,
/// `model_gateway/src/policies/kv_index/event_monitor.rs`). The word is
/// the chain's own at each ask, so the lending made while the index was
/// empty says its first block. In
/// an index that holds a block, a row whose lane heard no batch has its
/// number and no block: its place is made where the holder states it,
/// as a gateway's index numbers a worker where its stream's task opens
/// its place
/// (`model_gateway/src/policies/kv_index/event_monitor/index_sink.rs:186-196`).
/// On a pair's side the engine reads a numbered worker with no block
/// as the record's of sent prefixes, and a single request of token ids
/// names it by nothing: its own cases hold both
/// (`an_empty_index_write_keeps_the_predicted_worker_named`,
/// `a_list_under_an_index_that_holds_a_block_reads_no_record`).
#[test]
fn a_row_with_a_place_and_no_batch_has_its_number_and_no_block() {
    let bed = Bed::new();
    let (a, b) = (bed.row(A, MODEL), bed.row(B, MODEL));
    let empty = bed.index();
    assert!(!empty.holds_a_block(), "no block yet");
    bed.stores(a, &[1]);
    assert!(
        empty.holds_a_block(),
        "the same lending says the first block"
    );
    let index = bed.index();
    let quiet = index.worker_number(B).expect("its place stands");
    assert_eq!(index.worker_blocks(quiet), 0);
    assert_eq!(bed.tables.cursor(b, 0), None, "no batch ran through it");
    assert_eq!(deepest(&*index, &[1], 1 << quiet, 8), []);
    // The one holder leaves: the model keeps its index, by the quiet
    // row's lane, and both lendings say it holds no block.
    bed.leaves(a);
    assert!(!index.holds_a_block(), "its last block left");
    assert!(!empty.holds_a_block());
}

/// A number is free for the next row once its row leaves, and the next
/// row may take it while an engine's table stands. The lending says so:
/// its count of numbers handed out again moves at the take, and at no
/// removal. So the engine's next lookup, at the table's own instant,
/// reads the table anew and names the worker that took the number,
/// never the one that left. The engine's own cases hold the same for a
/// chain index (`a_number_that_changed_hands_names_its_new_worker`).
///
/// The count alone tells the engine. A lending made before the worker
/// left reads, under that worker's number, the blocks of the row that
/// took the number since.
#[test]
fn a_number_taken_again_names_the_worker_that_took_it() {
    let bed = Bed::new();
    let (a, b) = (bed.row(A, MODEL), bed.row(B, MODEL));
    bed.stores(a, &[1, 2]);
    bed.stores(b, &[9]);
    // The engine reads its table: nobody holds the request.
    let list = bed.list();
    assert_eq!(list.holders(&[5, 6, 7]), []);
    let older = bed.index();
    let number = older.worker_number(A).expect("a's number");
    let reused = older.numbers_reused();
    bed.leaves(a);
    assert_eq!(older.worker_blocks(number), 0, "its blocks left with it");
    assert_eq!(older.numbers_reused(), reused, "a removal alone");
    let c = bed.row(C, MODEL);
    bed.stores(c, &[5, 6, 7]);
    let newer = bed.index();
    assert_eq!(newer.worker_number(C), Some(number), "the freed number");
    assert_eq!(newer.worker_number(A), None);
    // The table that stands gave the number to the list's first worker.
    // The engine names its third, which holds the number now.
    assert_eq!(list.holders(&[5, 6, 7]), [(2, 3)]);
    // The older lending reads the new row's blocks under the number it
    // gave the worker that left, and says a number was handed on.
    assert_eq!(older.worker_blocks(number), 3);
    assert_eq!(deepest(&*older, &[5, 6, 7], 1 << number, 8), [(number, 3)]);
    assert_eq!(older.numbers_reused(), reused + 1);
    assert_eq!(newer.numbers_reused(), reused + 1);
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
