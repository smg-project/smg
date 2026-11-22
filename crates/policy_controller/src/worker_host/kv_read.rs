//! What the engine over worker lists reads of a host's cache events: the
//! index its workers' event streams write ([`KvFeed`](super::KvFeed)), lent
//! through the engine's own traits. No block is copied, and no stream is
//! read twice.
//!
//! The engine names a worker by its address, and the store's index names a
//! row by its id, which holds the registration's incarnation. So a worker's
//! number is found from its address through the registry and the holder's
//! row: the worker that stands under the address now, which is the one a
//! request to the address reaches. The address is read as the worker
//! registered it, and as no other form of it. A lending made before a
//! worker left drops that worker's number at its lookups, and counts no
//! block under it ([`LentIndex`]).
//!
//! An index that holds no block is none to the engine. In an index that
//! holds one, a row that asks for cache events has its number from its
//! registration, ahead of its stream, where the index had a number left.
//! The index holds no block of such a row until its stream writes, and the
//! engine reads a worker with no block as one that publishes none.
//!
//! # Limits
//!
//! - The depth is the chain's own. A worker that keeps other cache groups
//!   beside main attention is read as deep as its main blocks reach.
//! - The engine asks one index a list, by its first worker's model
//!   ([`CacheIndexes::index`]), and a row is indexed under its first model.

use std::sync::{Arc, Weak};

use smg_worker::registry::Generation;

use super::Objects;
use crate::{
    engine::{CacheIndex, CacheIndexes},
    store::{index::LentIndex, KvEventSink as _, MemoryStore},
};

/// A host's indexes of cache events as the engine over worker lists asks
/// for them: the store's own, a model each.
pub(super) struct EventIndexes<G> {
    store: Arc<MemoryStore>,
    objects: Arc<Objects<G>>,
}

impl<G> EventIndexes<G> {
    /// The indexes `store` holds, read by the addresses of the worker
    /// objects `objects` holds for its rows.
    pub(super) fn new(store: Arc<MemoryStore>, objects: Arc<Objects<G>>) -> Self {
        Self { store, objects }
    }
}

impl<G: Generation> CacheIndexes for EventIndexes<G> {
    /// A lending of the model's index, made anew at each call: the engine
    /// asks once a table it reads. `None` while the index holds no block,
    /// which the store keeps from a model's first row that asks for events:
    /// no worker of the model streamed one yet.
    fn index(&self, model: &str) -> Option<Arc<dyn CacheIndex>> {
        let lent = self.store.lend_index(model)?;
        if lent.is_empty() {
            return None;
        }
        let objects = Arc::downgrade(&self.objects);
        Some(Arc::new(EventIndex { lent, objects }))
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.store.block_size(model)
    }
}

/// One model's index, lent for one table of the engine's. The table stands
/// in the engine's value of a worker list, which the registry's lists hold,
/// and the worker objects' holder holds the registry: so the holder is
/// held weakly here, and a table keeps no registry alive.
struct EventIndex<G> {
    lent: LentIndex,
    objects: Weak<Objects<G>>,
}

impl<G: Generation> CacheIndex for EventIndex<G> {
    fn worker_number(&self, address: &str) -> Option<u32> {
        let objects = self.objects.upgrade()?;
        // The address as its worker registered it, and no other form of
        // it: a bare `host:port` whose worker left is not read as the
        // worker that stands under its canonical form.
        let object = objects.registry().get_id_by_url_or_reservation(address)?;
        let row = objects.standing(&object)?.row;
        self.lent.number_of(row)
    }

    fn worker_blocks(&self, number: u32) -> usize {
        self.lent.blocks(number)
    }

    fn deepest_into(
        &self,
        block_hashes: &[u64],
        _: u32,
        within: &[u64],
        cap: usize,
        below: &mut dyn FnMut(u32) -> u32,
        kept: &mut Vec<(u32, u32)>,
    ) {
        self.lent
            .deepest_into(block_hashes, within, cap, below, kept);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use kv_index::salt::{content_hash_with_seed, namespace_seed};
    use smg_worker::{
        common_proto::{kv_cache_event::Data, KvBlock, KvBlocksStored, KvCacheEvent, KvEventBatch},
        identity::basic::Worker,
        kv_events::{
            batch_rank,
            port::{BatchFacts, BatchOutcome, KvEventSink as _},
            KvEventHost as _,
        },
        registry::WorkerRegistry,
    };

    use super::*;
    use crate::{
        document::spec::{AppliedSpec, PolicySpec},
        engine::{
            overlap,
            view::{RequestFacts, Role, WorkerId},
            Affinity, Decision, PoolAffinity, Shed, ShedReason, Trace,
        },
        store::{
            self, AffinityQuery, AffinityRowsBuffers, GatewayId, KvEvent, Signals, ViewBuffers,
            ViewQuery, WorkerFacts, WorkersStore,
        },
        worker_host::{KvAdapter, KvFeed, KvHost, Members},
    };

    const MODEL: &str = "m";
    const FIRST: &str = "http://a:8000";
    const SECOND: &str = "http://b:8000";
    const THIRD: &str = "http://c:8000";
    /// Rank 1 of the engine at [`FIRST`].
    const UPPER: &str = "http://a:8000#1";

    type Store = dyn store::KvEventSink<Batch = [KvEvent]>;

    /// A store, the holder of its worker objects, an adapter its streams
    /// write through and the indexes the feed lends of it.
    struct Bed {
        store: Arc<MemoryStore>,
        objects: Arc<Objects<Members>>,
        sink: Arc<KvAdapter<Store>>,
        indexes: Arc<dyn CacheIndexes>,
    }

    /// The facts of a row at `key` that asks for events: rank `dp_rank` of
    /// the engine at `engine`.
    fn facts_of(key: &str, engine: &str, dp_rank: u16, incarnation: u64) -> WorkerFacts {
        WorkerFacts {
            key: key.into(),
            engine: engine.into(),
            incarnation,
            models: vec![MODEL.into()],
            dp_rank,
            dp_size: 1,
            signals: Signals {
                events: true,
                ..Signals::default()
            },
            ..WorkerFacts::default()
        }
    }

    impl Bed {
        fn new() -> Self {
            let store = Arc::new(MemoryStore::new());
            let objects = Arc::new(Objects::new(Arc::new(WorkerRegistry::new())));
            let sink = Arc::new(KvAdapter::new(Arc::clone(&store) as Arc<Store>));
            let indexes = KvFeed::new(Arc::clone(&store), Arc::clone(&objects)).indexes();
            Self {
                store,
                objects,
                sink,
                indexes,
            }
        }

        /// The row of `facts`, written, with its object laid.
        fn row(&self, facts: WorkerFacts) -> WorkerId {
            let row = self.store.upsert(facts).id();
            self.objects.follow(self.store.as_ref());
            row
        }

        /// A row of one rank at `key`, at `incarnation`.
        fn single(&self, key: &str, incarnation: u64) -> WorkerId {
            self.row(facts_of(key, key, 0, incarnation))
        }

        /// One batch on `row`'s own stream, which the store takes.
        fn stores(&self, row: WorkerId, sequence: u64, batch: &KvEventBatch) {
            let facts = BatchFacts {
                sequence,
                ..BatchFacts::default()
            };
            assert_eq!(self.sink.admit(row, 0, facts, batch), BatchOutcome::Applied);
        }

        /// The worker object that stands under `address`.
        fn worker(&self, address: &str) -> Arc<dyn Worker> {
            let registry = self.objects.registry();
            registry.get_by_url(address).expect("the worker's object")
        }

        /// The model's index as one table of the engine's is lent it.
        fn index(&self) -> Arc<dyn CacheIndex> {
            self.indexes.index(MODEL).expect("the model's index")
        }

        /// The holders the engine over worker lists is named for a request
        /// of `hashes` over the workers at `addresses`, each with its depth.
        fn named(&self, addresses: &[&str], hashes: &[u64]) -> Vec<(String, u32)> {
            let mut document = PolicySpec::preset("cache_aware").expect("the preset");
            document.generation = 1;
            let spec = AppliedSpec::validate(&document).expect("the preset validates");
            let affinity = Affinity::default();
            affinity.set_indexes(Arc::clone(&self.indexes));
            let workers: Vec<_> = addresses
                .iter()
                .map(|address| self.worker(address))
                .collect();
            let ids: Vec<_> = (1..=workers.len() as u64).map(WorkerId).collect();
            let mut facts = RequestFacts {
                model: MODEL,
                block_hashes: hashes,
                seed: 7,
                now_micros: 10_000_000,
                ..RequestFacts::default()
            };
            let mut named = Vec::new();
            let pool = PoolAffinity::default();
            // The whole decision is the case's own: it keeps the rows it is
            // named and sheds, so the answer says nothing of the naming.
            let _shed = overlap::decide(
                &affinity,
                &pool,
                &spec,
                &workers,
                &ids,
                &mut facts,
                |facts| {
                    let rows = facts.named.expect("rows named");
                    let holder = |row: u32| addresses[row as usize].to_owned();
                    named.extend(
                        rows.holders
                            .iter()
                            .map(|held| (holder(held.row), held.overlap_blocks)),
                    );
                    Decision::Shed(Shed {
                        reason: ShedReason::PolicyDeclined,
                        emptied_by: "a case",
                        retry_after_secs: 0,
                        wait_allowed: false,
                        trace: Trace::default(),
                    })
                },
            );
            named
        }

        /// The holders the store's own reader names for the same request,
        /// each row by its id.
        fn named_by_the_store(&self, hashes: &[u64]) -> Vec<(WorkerId, u32)> {
            self.store.publish_now();
            let gateway = GatewayId(0);
            let ask = ViewQuery {
                gateway,
                now: Instant::now(),
                pinned: None,
                affinity: false,
                signal_cells: false,
            };
            self.store.deciding(|world| {
                let mut view = ViewBuffers::default();
                let built = world
                    .build(MODEL, Role::Single, &ask, &mut view)
                    .expect("the pool");
                let naming = AffinityQuery {
                    gateway,
                    block_hashes: hashes,
                    namespace_seed: namespace_seed(None, None),
                    seed: 7,
                    cap: 0,
                    wires: None,
                    now_micros: built.now_micros,
                };
                let mut named = AffinityRowsBuffers::default();
                world.fill_affinity_rows(
                    MODEL,
                    Role::Single,
                    &naming,
                    &view.candidates,
                    &mut named,
                );
                let holders = named.holders.iter();
                holders
                    .map(|held| (view.candidates[held.row as usize].id, held.overlap_blocks))
                    .collect()
            })
        }
    }

    /// The blocks of a case's prompt, by the engine's hash of each and its
    /// tokens: a chain of three.
    const PROMPT: [(i64, [u32; 2]); 3] = [(-1, [1, 2]), (2, [3, 4]), (3, [5, 6])];

    /// A block of no case's prompt: what a worker stores to hold a block.
    const OTHER: [(i64, [u32; 2]); 1] = [(9, [7, 8])];

    /// The request's hashes of [`PROMPT`]'s blocks, as a stream's store
    /// hashes them.
    fn hashes() -> Vec<u64> {
        let seed = namespace_seed(None, None);
        let hash = |(_, tokens): &(i64, [u32; 2])| content_hash_with_seed(tokens, seed).0;
        PROMPT.iter().map(hash).collect()
    }

    /// A main attention store of `blocks`, a chain behind `parent`.
    fn stored(parent: Option<i64>, blocks: &[(i64, [u32; 2])]) -> KvEventBatch {
        let block = |&(block_hash, tokens): &(i64, [u32; 2])| KvBlock {
            block_hash,
            token_ids: tokens.to_vec(),
            ..KvBlock::default()
        };
        let event = KvBlocksStored {
            blocks: blocks.iter().map(block).collect(),
            parent_block_hash: parent,
            group_idx: Some(0),
            kv_cache_spec_kind: Some("full_attention".into()),
            ..KvBlocksStored::default()
        };
        KvEventBatch {
            events: vec![KvCacheEvent {
                data: Some(Data::Stored(event)),
                ..KvCacheEvent::default()
            }],
            ..KvEventBatch::default()
        }
    }

    /// The deepest holders of the prompt among the numbers `within` names,
    /// `cap` at most, as `index` answers a table's lookup.
    fn deepest(index: &dyn CacheIndex, within: u64, cap: usize) -> Vec<(u32, u32)> {
        let mut kept = Vec::new();
        index.deepest_into(&hashes(), 2, &[within], cap, &mut |_| 0, &mut kept);
        kept.sort_unstable();
        kept
    }

    /// What a worker's stream stored is named by the engine over worker
    /// lists as the store's own reader names it: the same holders, each as
    /// deep. A worker that stored nothing is no holder.
    #[test]
    fn a_stored_prefix_is_named_as_the_stores_reader_names_it() {
        let bed = Bed::new();
        let whole = bed.single(FIRST, 1);
        let head = bed.single(SECOND, 1);
        bed.single(THIRD, 1);
        bed.stores(whole, 1, &stored(None, &PROMPT[..1]));
        bed.stores(whole, 2, &stored(Some(PROMPT[0].0), &PROMPT[1..]));
        bed.stores(head, 1, &stored(None, &PROMPT[..1]));

        let mut by_the_store = bed.named_by_the_store(&hashes());
        by_the_store.sort_unstable();
        let mut want = vec![(whole, 3), (head, 1)];
        want.sort_unstable();
        assert_eq!(by_the_store, want);

        let named = bed.named(&[THIRD, FIRST, SECOND], &hashes());
        assert_eq!(named, [(FIRST.to_owned(), 3), (SECOND.to_owned(), 1)]);
    }

    /// A table's lookup keeps to its cap, the deepest first, and to the
    /// numbers the table names.
    #[test]
    fn a_lookup_keeps_to_its_cap_and_to_the_numbers_named() {
        let bed = Bed::new();
        let whole = bed.single(FIRST, 1);
        let head = bed.single(SECOND, 1);
        bed.stores(whole, 1, &stored(None, &PROMPT));
        bed.stores(head, 1, &stored(None, &PROMPT[..1]));
        let index = bed.index();
        let number = |address| index.worker_number(address).expect("a number");
        let (deep, shallow) = (number(FIRST), number(SECOND));
        let both = (1 << deep) | (1 << shallow);
        let mut every = [(deep, 3), (shallow, 1)];
        every.sort_unstable();
        assert_eq!(deepest(&*index, both, 8), every);
        assert_eq!(deepest(&*index, both, 1), [(deep, 3)]);
        assert_eq!(deepest(&*index, 1 << shallow, 8), [(shallow, 1)]);
    }

    /// A worker replaced under its address is read under its new row. A
    /// lending made before the replacement drops the number and reads no
    /// block under it: the index handed it to the new row, whose blocks are
    /// not the old row's.
    #[test]
    fn a_replaced_worker_is_read_under_its_new_row_and_an_older_lending_drops_its_number() {
        let bed = Bed::new();
        let old = bed.single(FIRST, 1);
        bed.single(SECOND, 1);
        bed.stores(old, 1, &stored(None, &PROMPT[..1]));
        let before = bed.index();
        let number = before.worker_number(FIRST).expect("the row's number");
        assert_eq!(before.worker_blocks(number), 1);
        assert_eq!(deepest(&*before, 1 << number, 8), [(number, 1)]);

        let new = bed.single(FIRST, 2);
        assert_ne!(new, old);
        bed.stores(new, 1, &stored(None, &PROMPT[..2]));
        let after = bed.index();
        // The index hands the number the old row gave back to the new one.
        assert_eq!(after.worker_number(FIRST), Some(number));
        assert_eq!(after.worker_blocks(number), 2);
        assert_eq!(deepest(&*after, 1 << number, 8), [(number, 2)]);
        // The older lending: the number was taken behind it.
        assert_eq!(before.worker_number(FIRST), None);
        assert_eq!(before.worker_blocks(number), 0);
        assert!(deepest(&*before, 1 << number, 8).is_empty());
    }

    /// A bare address whose worker left is not read as the worker that
    /// stands under its canonical form: a list that still holds both gives
    /// the number to the one that stands.
    #[test]
    fn a_bare_address_of_a_worker_that_left_has_no_number() {
        const BARE: &str = "a:8000";
        let bed = Bed::new();
        let canonical = bed.single(FIRST, 1);
        let bare = bed.single(BARE, 1);
        bed.stores(canonical, 1, &stored(None, &OTHER));
        let index = bed.index();
        let (stands, left) = (index.worker_number(FIRST), index.worker_number(BARE));
        assert!(stands.is_some() && left.is_some() && stands != left);

        let revision = bed.store.get(bare).expect("the row").revision;
        bed.store.remove(bare, revision).expect("the row leaves");
        bed.objects.follow(bed.store.as_ref());
        // The registry itself reads the bare address as the canonical one.
        let found = bed.objects.registry().get_by_url(BARE);
        assert_eq!(
            found.map(|worker| worker.url().to_owned()),
            Some(FIRST.to_owned())
        );
        let index = bed.index();
        assert_eq!(index.worker_number(FIRST), stands);
        assert_eq!(index.worker_number(BARE), None);
    }

    /// A lent index keeps the worker objects' holder alive no longer than
    /// its host does: the engine's table of a list holds the index, and the
    /// holder's registry holds the list.
    #[test]
    fn a_lent_index_holds_the_worker_objects_weakly() {
        let bed = Bed::new();
        let row = bed.single(FIRST, 1);
        bed.stores(row, 1, &stored(None, &OTHER));
        let index = bed.index();
        assert!(index.worker_number(FIRST).is_some());
        let Bed {
            objects, indexes, ..
        } = bed;
        let holder = Arc::downgrade(&objects);
        drop((objects, indexes));
        assert!(holder.upgrade().is_none(), "the lent index kept the holder");
        assert_eq!(index.worker_number(FIRST), None);
    }

    /// An index that holds no block is none to the engine, as an empty
    /// index is none to the classic router
    /// (`model_gateway/src/policies/cache_aware.rs:1818`): the store keeps
    /// one from a model's first row that asks for events. In an index that
    /// holds a block, a row that stored nothing has its number and no block.
    #[test]
    fn an_index_that_holds_no_block_is_none_to_the_engine() {
        let bed = Bed::new();
        let row = bed.single(FIRST, 1);
        bed.single(SECOND, 1);
        assert!(bed.store.lend_index(MODEL).is_some());
        assert!(bed.indexes.index(MODEL).is_none());

        bed.stores(row, 1, &stored(None, &OTHER));
        let index = bed.index();
        let silent = index.worker_number(SECOND).expect("its number");
        assert_eq!(index.worker_blocks(silent), 0);
    }

    /// A row that left is in no index: its address answers no number, and a
    /// model whose last indexed row left has no index.
    #[test]
    fn a_row_that_left_is_in_no_index() {
        let bed = Bed::new();
        let row = bed.single(FIRST, 1);
        let other = bed.single(SECOND, 1);
        bed.stores(row, 1, &stored(None, &PROMPT[..1]));
        bed.stores(other, 1, &stored(None, &OTHER));
        let revision = |row| bed.store.get(row).expect("the row").revision;
        bed.store
            .remove(row, revision(row))
            .expect("the row leaves");
        bed.objects.follow(bed.store.as_ref());
        let index = bed.index();
        assert_eq!(index.worker_number(FIRST), None);
        assert!(index.worker_number(SECOND).is_some());
        assert!(deepest(&*index, u64::MAX, 8).is_empty());

        bed.store
            .remove(other, revision(other))
            .expect("the row leaves");
        bed.objects.follow(bed.store.as_ref());
        assert!(bed.store.lend_index(MODEL).is_none());
        assert!(bed.indexes.index(MODEL).is_none());
    }

    /// A model's index is made again where a row comes after its last
    /// indexed row left. A lending of the index before names no row of the
    /// one after: their numbers are another index's, one of them a number
    /// the old index never gave.
    #[test]
    fn a_lending_of_an_index_made_again_names_no_row() {
        let bed = Bed::new();
        let row = bed.single(FIRST, 1);
        bed.stores(row, 1, &stored(None, &OTHER));
        let before = bed.index();
        assert!(before.worker_number(FIRST).is_some());
        let revision = bed.store.get(row).expect("the row").revision;
        bed.store.remove(row, revision).expect("the row leaves");
        bed.objects.follow(bed.store.as_ref());
        assert!(bed.store.lend_index(MODEL).is_none());

        let holds = bed.single(SECOND, 1);
        bed.single(THIRD, 1);
        bed.stores(holds, 1, &stored(None, &OTHER));
        let after = bed.index();
        let (second, third) = (after.worker_number(SECOND), after.worker_number(THIRD));
        assert!(second.is_some() && third.is_some() && second != third);
        assert_eq!(
            (before.worker_number(SECOND), before.worker_number(THIRD)),
            (None, None)
        );
    }

    /// A row of another model has no number in this model's index: its lane
    /// writes the other model's, where its number is its own.
    #[test]
    fn a_row_of_another_model_has_no_number() {
        let bed = Bed::new();
        let first = bed.single(FIRST, 1);
        let second = bed.row(WorkerFacts {
            models: vec!["n".into()],
            ..facts_of(SECOND, SECOND, 0, 1)
        });
        bed.stores(first, 1, &stored(None, &OTHER));
        bed.stores(second, 1, &stored(None, &OTHER));
        let index = bed.index();
        assert!(index.worker_number(FIRST).is_some());
        assert_eq!(index.worker_number(SECOND), None);
        let others = bed.indexes.index("n").expect("the other model's index");
        assert!(others.worker_number(SECOND).is_some());
        assert_eq!(others.worker_number(FIRST), None);
    }

    /// A rank above 0 opens no feed: its engine's one stream writes its
    /// row. The rank's own address answers the rank's own number.
    #[test]
    fn a_rank_above_0_is_numbered_by_its_own_row() {
        let bed = Bed::new();
        let ranked = |key, dp_rank| WorkerFacts {
            dp_size: 3,
            ..facts_of(key, FIRST, dp_rank, 1)
        };
        let lower = bed.row(ranked(FIRST, 0));
        bed.row(ranked(UPPER, 1));
        let host = KvHost::new(
            Arc::clone(&bed.objects),
            Arc::clone(&bed.store) as Arc<Store>,
        );
        let worker = bed.worker(FIRST);
        let ticket = host.enroll(&worker, MODEL);
        let (sink, opened) = host.open(ticket, &worker, MODEL).expect("the feed");
        assert_eq!(opened, lower);
        let facts = BatchFacts {
            sequence: 1,
            ..BatchFacts::default()
        };
        // Rank 1's batch on rank 0's stream, with the rank the collector
        // reads of it.
        let batch = KvEventBatch {
            dp_rank: Some(1),
            ..stored(None, &PROMPT[..1])
        };
        let rank = batch_rank(&batch).expect("the batch's rank");
        assert_eq!(rank, 1);
        assert_eq!(
            sink.admit(lower, rank, facts, &batch),
            BatchOutcome::Applied
        );

        let index = bed.index();
        let upper = index.worker_number(UPPER).expect("rank 1's number");
        let own = index.worker_number(FIRST).expect("rank 0's number");
        assert_ne!(upper, own);
        assert_eq!(
            (index.worker_blocks(own), index.worker_blocks(upper)),
            (0, 1)
        );
    }

    /// The block size is the store's of record, and none for a model it
    /// holds none of.
    #[test]
    fn the_block_size_is_the_stores_of_record() {
        let bed = Bed::new();
        assert_eq!(bed.indexes.block_size(MODEL), None);
        bed.store.set_block_size(MODEL, 16);
        assert_eq!(bed.indexes.block_size(MODEL), Some(16));
    }
}
