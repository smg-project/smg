//! A batch of cache events through the worker host's adapter
//! (`worker_host::KvAdapter`), as a worker's event stream hands it over. A
//! turn's blocks are stored after a prefix every worker holds and then
//! evicted: two batches, the steady state of a cache at its capacity. Each
//! shape runs through the adapter and, beside it, on a chain index with
//! nothing around it (the floor: the same two writes by the index's own
//! entries, the blocks hashed ahead of the clock). The adapter's share of a
//! batch is the difference of two rows of one binary: it finds the row's
//! state and its lane, takes the lane's lock, runs the header through the
//! cursor and the events through the worker crate's apply rules, which hash
//! each block's tokens.
//!
//! Ids: `policy_kv_adapter/<floor|adapter>/<blocks>/<workers>`: 4 and 256
//! blocks a turn, of 16 tokens each, beside 64 and 512 workers that publish
//! events and hold the prefix.
//!
//! A bench target of its own, under the `worker-host` feature. No gate
//! reads it.
#![expect(
    clippy::expect_used,
    reason = "a bench's setup has no caller to answer"
)]

use std::{hint::black_box, mem, sync::Arc};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kv_index::{compute_content_hash, ChainBlockMap, ChainIndex, SequenceHash, StoredBlock};
use smg_grand_central::{
    engine::WorkerId,
    store::{MemoryStore, Registrant, Signals, WorkerFacts},
    worker_host::{KvAdapter, KvHost, KvTables, Members, Objects},
};
use smg_worker::{
    common_proto::{
        kv_cache_event::Data, KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheEvent, KvEventBatch,
    },
    kv_events::{
        port::{BatchFacts, BatchOutcome, KvEventSink},
        KvEventHost,
    },
    registry::WorkerRegistry,
};

const MODEL: &str = "m";
/// Tokens of a block.
const BLOCK: u32 = 16;
/// Blocks of the prefix every worker holds: a 512-token system prompt.
const PREFIX: usize = 32;
const TURNS: [usize; 2] = [4, 256];
const WORKERS: [usize; 2] = [64, 512];
/// The most workers a model's index holds in the worker host's KV tables:
/// the floor's index is sized alike.
const INDEX_WORKERS: usize = 1024;

/// SplitMix64's output step: distinct words from distinct words.
fn mix(word: u64) -> u64 {
    let mut z = word.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The block `word` names as a worker's event states it: the worker's own
/// hash of the block, and the block's tokens.
fn block(word: u64) -> KvBlock {
    let own = mix(word);
    KvBlock {
        block_hash: own as i64,
        token_ids: (0..u64::from(BLOCK))
            .map(|at| mix(own ^ at) as u32)
            .collect(),
        block_size: BLOCK as i32,
        ..KvBlock::default()
    }
}

/// The prefix every worker of a cell holds.
fn prefix() -> Vec<KvBlock> {
    (0..PREFIX as u64).map(block).collect()
}

/// The turn of writer `writer`: `blocks` blocks of its own.
fn turn_of(writer: usize, blocks: usize) -> Vec<KvBlock> {
    let own = (writer as u64 + 1) << 40;
    (0..blocks as u64).map(|at| block(own | at)).collect()
}

/// One batch of the one event `data`.
fn batch(data: Data) -> KvEventBatch {
    KvEventBatch {
        events: vec![KvCacheEvent {
            data: Some(data),
            ..KvCacheEvent::default()
        }],
        ..KvEventBatch::default()
    }
}

/// The batch that stores `blocks` after `parent`, from the root for none.
fn stored(blocks: &[KvBlock], parent: Option<i64>) -> KvEventBatch {
    batch(Data::Stored(KvBlocksStored {
        blocks: blocks.to_vec(),
        parent_block_hash: parent,
        ..KvBlocksStored::default()
    }))
}

/// The batch that evicts `blocks`.
fn evicted(blocks: &[KvBlock]) -> KvEventBatch {
    batch(Data::Removed(KvBlocksRemoved {
        block_hashes: blocks.iter().map(|block| block.block_hash).collect(),
        ..KvBlocksRemoved::default()
    }))
}

fn facts(sequence: u64) -> BatchFacts {
    BatchFacts {
        sequence,
        ..BatchFacts::default()
    }
}

/// `blocks` as the index's own entries: what the apply rules make of a
/// store that names no adapter and no salt.
fn entries(blocks: &[KvBlock]) -> Vec<StoredBlock> {
    blocks
        .iter()
        .map(|block| StoredBlock {
            seq_hash: SequenceHash::from(block.block_hash),
            content_hash: compute_content_hash(&block.token_ids),
        })
        .collect()
}

/// A chain index of `workers` workers that each hold `prefix`, sized as
/// the tables size a model's: the floor's index. Each worker's map, by its
/// number.
fn bare(workers: usize, prefix: &[StoredBlock]) -> (ChainIndex, Vec<ChainBlockMap>) {
    let index = ChainIndex::with_max_workers(INDEX_WORKERS);
    let maps = (0..workers)
        .map(|worker| {
            let number = index
                .intern_worker(&format!("{worker:016x}"))
                .expect("room");
            assert_eq!(number as usize, worker, "numbers in the order of interning");
            let mut map = ChainBlockMap::new();
            index
                .apply_stored(number, prefix, None, &mut map)
                .expect("the prefix");
            map
        })
        .collect();
    (index, maps)
}

/// One worker's turn on the floor: its blocks stored after the prefix,
/// then evicted, by the index's own entries.
struct FloorTurn {
    worker: u32,
    map: ChainBlockMap,
    parent: SequenceHash,
    blocks: Vec<StoredBlock>,
    hashes: Vec<SequenceHash>,
}

impl FloorTurn {
    fn new(writer: usize, maps: &mut [ChainBlockMap], parent: SequenceHash, blocks: usize) -> Self {
        let blocks = entries(&turn_of(writer, blocks));
        Self {
            worker: writer as u32,
            map: mem::take(&mut maps[writer]),
            parent,
            hashes: blocks.iter().map(|block| block.seq_hash).collect(),
            blocks,
        }
    }

    fn run(&mut self, index: &ChainIndex) {
        index
            .apply_stored(
                self.worker,
                black_box(&self.blocks),
                Some(self.parent),
                &mut self.map,
            )
            .expect("stored");
        index.apply_removed(self.worker, black_box(&self.hashes), &mut self.map);
    }
}

/// A cell's workers behind the adapter: a host over a holder of worker
/// objects that was handed the tables, each worker's feed open and the
/// prefix stored by its first batch.
struct Hosted {
    tables: Arc<KvTables>,
    sink: Arc<KvAdapter>,
    rows: Vec<WorkerId>,
    /// The host the feeds were opened by, kept for as long as they write.
    _host: KvHost<Members>,
}

impl Hosted {
    fn new(workers: usize, prefix: &[KvBlock]) -> Self {
        let url = |worker: usize| format!("grpc://w{worker}:8000");
        let store = MemoryStore::new();
        let tables = Arc::new(KvTables::default());
        let registry = Arc::new(WorkerRegistry::<Members>::new());
        let objects = Arc::new(Objects::new(registry).with_kv_tables(Arc::clone(&tables)));
        for worker in 0..workers {
            let facts = WorkerFacts {
                key: url(worker),
                engine: url(worker),
                incarnation: 1,
                models: vec![MODEL.to_owned()],
                dp_size: 1,
                kv_block_size: Some(BLOCK),
                signals: Signals {
                    events: true,
                    ..Signals::default()
                },
                ..WorkerFacts::default()
            };
            let written = objects.upsert(&store, facts, Registrant::Control);
            written.expect("the store takes the write");
        }
        let host = KvHost::new(Arc::clone(&objects));
        let held = stored(prefix, None);
        let feeds: Vec<_> = (0..workers)
            .map(|worker| {
                let object = objects.registry().get_by_url(&url(worker));
                let object = object.expect("the row's object");
                let ticket = host.enroll(&object, MODEL);
                let (sink, row) = host.open(ticket, &object, MODEL).expect("the row's feed");
                assert_eq!(sink.admit(row, 0, facts(1), &held), BatchOutcome::Applied);
                assert_eq!(tables.indexed_blocks(row, 0), PREFIX);
                (sink, row)
            })
            .collect();
        let sink = Arc::clone(&feeds.first().expect("a worker").0);
        Self {
            tables,
            sink,
            rows: feeds.into_iter().map(|(_, row)| row).collect(),
            _host: host,
        }
    }
}

/// The same turn as the adapter takes it: the batch that stores the blocks
/// and the batch that evicts them, at the worker's next two sequence
/// numbers.
struct AdapterTurn {
    row: WorkerId,
    stored: KvEventBatch,
    evicted: KvEventBatch,
    sequence: u64,
}

impl AdapterTurn {
    /// The turn of `writer`, proved once: the tables hold the turn's blocks
    /// after the first batch and the prefix alone after the second.
    fn new(hosted: &Hosted, writer: usize, parent: i64, blocks: usize) -> Self {
        let turn = turn_of(writer, blocks);
        let row = hosted.rows[writer];
        let (stored, evicted) = (stored(&turn, Some(parent)), evicted(&turn));
        let (sink, tables) = (&hosted.sink, &hosted.tables);
        assert_eq!(sink.admit(row, 0, facts(2), &stored), BatchOutcome::Applied);
        assert_eq!(tables.indexed_blocks(row, 0), PREFIX + blocks);
        assert_eq!(
            sink.admit(row, 0, facts(3), &evicted),
            BatchOutcome::Applied
        );
        assert_eq!(tables.indexed_blocks(row, 0), PREFIX);
        Self {
            row,
            stored,
            evicted,
            sequence: 3,
        }
    }

    fn run(&mut self, sink: &KvAdapter) -> (BatchOutcome, BatchOutcome) {
        self.sequence += 2;
        (
            sink.admit(
                self.row,
                0,
                facts(self.sequence - 1),
                black_box(&self.stored),
            ),
            sink.admit(self.row, 0, facts(self.sequence), black_box(&self.evicted)),
        )
    }

    /// Every timed batch was applied: the cursor stands at the last one
    /// with no gap behind it, and the prefix alone is held.
    fn check(&self, tables: &KvTables) {
        let cursor = tables.cursor(self.row, 0).expect("heard");
        assert_eq!((cursor.applied, cursor.degraded), (self.sequence, false));
        assert_eq!(tables.indexed_blocks(self.row, 0), PREFIX);
    }
}

fn turns(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_kv_adapter");
    let prefix = prefix();
    let parent = prefix[PREFIX - 1].block_hash;
    for &workers in &WORKERS {
        let (index, mut maps) = bare(workers, &entries(&prefix));
        let hosted = Hosted::new(workers, &prefix);
        for (at, &blocks) in TURNS.iter().enumerate() {
            // A writer of its own a turn size, from the middle of the workers.
            let writer = workers / 2 + at;
            group.throughput(Throughput::Elements(2 * blocks as u64));
            let mut floor = FloorTurn::new(writer, &mut maps, SequenceHash::from(parent), blocks);
            group.bench_function(BenchmarkId::new(format!("floor/{blocks}"), workers), |b| {
                b.iter(|| floor.run(&index));
            });
            let mut written = AdapterTurn::new(&hosted, writer, parent, blocks);
            group.bench_function(
                BenchmarkId::new(format!("adapter/{blocks}"), workers),
                |b| {
                    b.iter(|| written.run(&hosted.sink));
                },
            );
            written.check(&hosted.tables);
        }
    }
    group.finish();
}

criterion_group!(benches, turns);
criterion_main!(benches);
