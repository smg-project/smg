//! The KV sink's write path. A turn's blocks are stored after a prefix every
//! row holds and then evicted: two batches, the steady state of a cache at
//! its capacity. Each shape runs through the store's sink and, beside it, on
//! a chain index with no store around it (the floor: the same two writes by
//! the index's own entries), so the sink's share of a batch (it finds its
//! lane, takes the lane's lock and runs through the cursor) is the
//! difference of two rows of one binary.
//!
//! Ids: `policy_store_kv/<floor|sink>/<blocks>/<rows>`, 4, 256 and 4,096
//! blocks a turn beside 64 and 512 publishing rows; and
//! `policy_store_kv_threads/<floor|sink>/<threads>/<rows>`, one and eight
//! threads writing turns of 4 blocks, each to a row of its own. A threads
//! row's time is one turn on one thread (the slowest thread's wall time over
//! its turns): eight against one reads what the lock every batch shares, the
//! table it finds its lane in, costs beside other writers.
//! `policy_store_kv_leave/<floor|sink>/<blocks>`: a row that holds 8,192
//! blocks leaves beside 64 others, by the index's own removal (the floor)
//! and by the store's (`sink/0` is the same removal of a row that holds
//! nothing): what a membership write pays for a leaving row's blocks.
//!
//! `STORE_KV_FILL=<rows>` runs no cell: it fills an index of that many rows
//! through the sink, each to 8,192 blocks of conversation-shaped chains
//! (sessions of 8 to 512 blocks, six in ten after one of eight shared
//! prefixes, each held by one to four rows; stores of at most 16 blocks
//! under their parent), and prints the blocks held, the time the sink took
//! and the resident memory the fill added.
//!
//! A bench target of its own: the store's gated rows are the `store_view`
//! binary's.
#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "a bench's setup has no caller to answer, and the fill's report is its output"
)]

#[path = "common/filled.rs"]
mod filled;

use std::{
    fs,
    hint::black_box,
    mem, thread,
    time::{Duration, Instant},
};

use criterion::{criterion_group, BatchSize, BenchmarkId, Criterion, Throughput};
use kv_index::{chain_prefix_hash, ChainBlockMap, ChainIndex};
use smg_policy_controller::{
    engine::{Role, WorkerId},
    store::{
        BatchFacts, BatchOutcome, ContentHash, KvEvent, KvEventSink, LaneCounts, MemoryStore,
        SequenceHash, Signals, StoredBlock, WorkerFacts, WorkersStore, INDEX_WORKERS,
    },
};

const MODEL: &str = "m";
/// Blocks of the prefix every row holds: a 512-token system prompt at 16
/// tokens a block.
const PREFIX: usize = 32;
const TURNS: [usize; 3] = [4, 256, 4_096];
const ROWS: [usize; 2] = [64, 512];
const THREADS: [usize; 2] = [1, 8];
/// Blocks of a turn in the threads rows.
const THREAD_TURN: usize = 4;
/// Blocks a row is filled to under `STORE_KV_FILL`.
const FILL_BLOCKS: usize = 8_192;
/// What a leaving row holds in the leave cells: this many chains of this
/// many blocks after the prefix, 8,192 blocks in all.
const LEAVE_CHAINS: usize = 64;
const LEAVE_CHAIN: usize = 128;

type Sink<'a> = &'a dyn KvEventSink<Batch = [KvEvent]>;

/// SplitMix64's output step: distinct contents from distinct words.
fn mix(word: u64) -> u64 {
    let mut z = word.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The blocks of `contents` after `parent` as an engine names a chain: a
/// block's hash is a hash of its parent's hash and its own content, and a
/// chain's first block is named by its content.
fn chain(parent: Option<SequenceHash>, contents: impl Iterator<Item = u64>) -> Vec<StoredBlock> {
    let mut last = parent;
    contents
        .map(|content| {
            let content_hash = ContentHash(content);
            let seq_hash = last.map_or(SequenceHash(content), |parent| {
                chain_prefix_hash(parent, content_hash)
            });
            last = Some(seq_hash);
            StoredBlock {
                seq_hash,
                content_hash,
            }
        })
        .collect()
}

/// The prefix every row of a cell holds.
fn prefix() -> Vec<StoredBlock> {
    chain(None, (0..PREFIX as u64).map(mix))
}

/// The turn of writer `writer`: `blocks` blocks of its own after the prefix.
fn turn_of(writer: usize, blocks: usize, parent: SequenceHash) -> Vec<StoredBlock> {
    let own = (writer as u64 + 1) << 40;
    chain(Some(parent), (0..blocks as u64).map(|i| mix(own | i)))
}

fn batch(sequence: u64) -> BatchFacts {
    BatchFacts {
        sequence,
        ..BatchFacts::default()
    }
}

/// The facts of row `row`: it publishes events.
fn publisher(row: usize) -> WorkerFacts {
    let key = format!("http://w{row}:8000");
    WorkerFacts {
        engine: key.clone(),
        key,
        incarnation: 1,
        models: vec![MODEL.to_string()],
        dp_size: 1,
        signals: Signals {
            events: true,
            ..Signals::default()
        },
        ..WorkerFacts::default()
    }
}

/// A store of `rows` rows that publish events; their ids in order.
fn publishers(rows: usize) -> (MemoryStore, Vec<WorkerId>) {
    let store = MemoryStore::new();
    let ids = (0..rows)
        .map(|row| store.upsert(publisher(row)).id())
        .collect();
    store.publish_now();
    filled::filled(&store, MODEL, Role::Single, rows);
    (store, ids)
}

/// A store of `rows` publishing rows that each hold `prefix`, stored by
/// their first batch.
fn holders(rows: usize, prefix: &[StoredBlock]) -> (MemoryStore, Vec<WorkerId>) {
    let (store, ids) = publishers(rows);
    let held = [KvEvent::Stored {
        parent: None,
        blocks: prefix.to_vec(),
    }];
    for id in &ids {
        assert_eq!(store.admit(*id, 0, batch(1), &held), BatchOutcome::Applied);
    }
    (store, ids)
}

/// A chain index of `rows` workers that each hold `prefix`, sized as the
/// store sizes a model's: the floor's index. Each worker's map, by its id.
fn bare(rows: usize, prefix: &[StoredBlock]) -> (ChainIndex, Vec<ChainBlockMap>) {
    let index = ChainIndex::with_max_workers(INDEX_WORKERS);
    let maps = (0..rows)
        .map(|row| {
            let worker = index.intern_worker(&format!("{row:016x}")).expect("room");
            assert_eq!(worker as usize, row, "ids in the order of interning");
            let mut map = ChainBlockMap::new();
            index
                .apply_stored(worker, prefix, None, &mut map)
                .expect("the prefix");
            map
        })
        .collect();
    (index, maps)
}

/// One row's turn on the floor: its blocks stored after the prefix, then
/// evicted, by the index's own entries.
struct FloorTurn {
    worker: u32,
    map: ChainBlockMap,
    parent: SequenceHash,
    blocks: Vec<StoredBlock>,
    hashes: Vec<SequenceHash>,
}

impl FloorTurn {
    fn new(row: usize, maps: &mut [ChainBlockMap], parent: SequenceHash, blocks: usize) -> Self {
        let blocks = turn_of(row, blocks, parent);
        Self {
            worker: row as u32,
            map: mem::take(&mut maps[row]),
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

/// The same turn as the sink takes it: the batch that stores the blocks and
/// the batch that evicts them, at the row's next two sequence numbers.
struct SinkTurn {
    id: WorkerId,
    stored: [KvEvent; 1],
    evicted: [KvEvent; 1],
    sequence: u64,
}

impl SinkTurn {
    /// The turn of `row`, proved once: the store holds the turn's blocks
    /// after the first batch and the prefix alone after the second.
    fn new(
        store: &MemoryStore,
        row: usize,
        ids: &[WorkerId],
        parent: SequenceHash,
        blocks: usize,
    ) -> Self {
        let turn = turn_of(row, blocks, parent);
        let id = ids[row];
        let evicted = [KvEvent::Removed(
            turn.iter().map(|block| block.seq_hash).collect(),
        )];
        let stored = [KvEvent::Stored {
            parent: Some(parent),
            blocks: turn,
        }];
        assert_eq!(store.admit(id, 0, batch(2), &stored), BatchOutcome::Applied);
        assert_eq!(store.indexed_blocks(id, 0), PREFIX + blocks);
        assert_eq!(
            store.admit(id, 0, batch(3), &evicted),
            BatchOutcome::Applied
        );
        assert_eq!(store.indexed_blocks(id, 0), PREFIX);
        Self {
            id,
            stored,
            evicted,
            sequence: 3,
        }
    }

    fn run(&mut self, sink: Sink<'_>) -> (BatchOutcome, BatchOutcome) {
        self.sequence += 2;
        (
            sink.admit(
                self.id,
                0,
                batch(self.sequence - 1),
                black_box(&self.stored),
            ),
            sink.admit(self.id, 0, batch(self.sequence), black_box(&self.evicted)),
        )
    }

    /// Every timed batch was applied: the cursor stands at the last one with
    /// no gap behind it, the prefix alone is held and nothing was declined.
    fn check(&self, store: &MemoryStore) {
        let cursor = store.cursor(self.id, 0).expect("heard");
        assert_eq!((cursor.applied, cursor.degraded), (self.sequence, false));
        assert_eq!(store.indexed_blocks(self.id, 0), PREFIX);
        assert_eq!(store.lane_counts(self.id, 0), Some(LaneCounts::default()));
    }
}

fn turns(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_kv");
    let prefix = prefix();
    let parent = prefix[PREFIX - 1].seq_hash;
    for &rows in &ROWS {
        let (index, mut maps) = bare(rows, &prefix);
        let (store, ids) = holders(rows, &prefix);
        for (at, &blocks) in TURNS.iter().enumerate() {
            // A writer of its own a turn size, from the middle of the ids.
            let row = rows / 2 + at;
            group.throughput(Throughput::Elements(2 * blocks as u64));
            let mut floor = FloorTurn::new(row, &mut maps, parent, blocks);
            group.bench_function(BenchmarkId::new(format!("floor/{blocks}"), rows), |b| {
                b.iter(|| floor.run(&index));
            });
            let mut written = SinkTurn::new(&store, row, &ids, parent, blocks);
            group.bench_function(BenchmarkId::new(format!("sink/{blocks}"), rows), |b| {
                b.iter(|| written.run(&store));
            });
            written.check(&store);
        }
    }
    group.finish();
}

/// Each writer's `iters` turns on a thread of its own: the slowest thread's
/// wall time.
fn slowest<W: Send>(writers: &mut [W], iters: u64, turn: impl Fn(&mut W) + Sync) -> Duration {
    thread::scope(|s| {
        let writing: Vec<_> = writers
            .iter_mut()
            .map(|writer| {
                let turn = &turn;
                s.spawn(move || {
                    let start = Instant::now();
                    for _ in 0..iters {
                        turn(writer);
                    }
                    start.elapsed()
                })
            })
            .collect();
        writing
            .into_iter()
            .map(|thread| thread.join().expect("a writing thread"))
            .max()
            .unwrap_or_default()
    })
}

fn threads(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_kv_threads");
    let prefix = prefix();
    let parent = prefix[PREFIX - 1].seq_hash;
    let most = THREADS.into_iter().max().unwrap_or(1);
    for &rows in &ROWS {
        let (index, mut maps) = bare(rows, &prefix);
        let (store, ids) = holders(rows, &prefix);
        // The writers' rows are spread over the ids: at 512 rows each sits
        // in a coverage word of its own.
        let spread = |writer: usize| writer * rows / most;
        let mut floors: Vec<FloorTurn> = (0..most)
            .map(|writer| FloorTurn::new(spread(writer), &mut maps, parent, THREAD_TURN))
            .collect();
        let mut written: Vec<SinkTurn> = (0..most)
            .map(|writer| SinkTurn::new(&store, spread(writer), &ids, parent, THREAD_TURN))
            .collect();
        for &threads in &THREADS {
            group.bench_function(BenchmarkId::new(format!("floor/{threads}"), rows), |b| {
                b.iter_custom(|iters| {
                    slowest(&mut floors[..threads], iters, |turn| turn.run(&index))
                });
            });
            group.bench_function(BenchmarkId::new(format!("sink/{threads}"), rows), |b| {
                b.iter_custom(|iters| {
                    slowest(&mut written[..threads], iters, |turn| {
                        black_box(turn.run(&store));
                    })
                });
            });
        }
        for turn in &written {
            turn.check(&store);
        }
    }
    group.finish();
}

fn leaves(c: &mut Criterion) {
    let mut group = c.benchmark_group("policy_store_kv_leave");
    let rows = 64;
    let prefix = prefix();
    let parent = prefix[PREFIX - 1].seq_hash;
    let chains: Vec<Vec<StoredBlock>> = (0..LEAVE_CHAINS)
        .map(|chain| turn_of(rows + chain, LEAVE_CHAIN, parent))
        .collect();
    let blocks = LEAVE_CHAINS * LEAVE_CHAIN;

    let (index, _maps) = bare(rows, &prefix);
    group.bench_function(BenchmarkId::new("floor", blocks), |b| {
        b.iter_batched(
            || {
                let worker = index.intern_worker("leaving").expect("room");
                let mut map = ChainBlockMap::new();
                index
                    .apply_stored(worker, &prefix, None, &mut map)
                    .expect("the prefix");
                for chain in &chains {
                    index
                        .apply_stored(worker, chain, Some(parent), &mut map)
                        .expect("stored");
                }
                (worker, map)
            },
            |(worker, map)| index.remove_worker(worker, map),
            BatchSize::PerIteration,
        );
    });

    let (store, _ids) = holders(rows, &prefix);
    let held = [KvEvent::Stored {
        parent: None,
        blocks: prefix.clone(),
    }];
    let stored: Vec<[KvEvent; 1]> = chains
        .iter()
        .map(|chain| {
            [KvEvent::Stored {
                parent: Some(parent),
                blocks: chain.clone(),
            }]
        })
        .collect();
    for filled in [false, true] {
        let blocks = if filled { blocks } else { 0 };
        group.bench_function(BenchmarkId::new("sink", blocks), |b| {
            b.iter_batched(
                || {
                    let change = store.upsert(publisher(rows));
                    store.publish_now();
                    let id = change.id();
                    if filled {
                        assert_eq!(store.admit(id, 0, batch(1), &held), BatchOutcome::Applied);
                        for (at, chain) in stored.iter().enumerate() {
                            store.admit(id, 0, batch(at as u64 + 2), chain);
                        }
                        assert_eq!(store.indexed_blocks(id, 0), PREFIX + blocks);
                    }
                    let revision = WorkersStore::get(&store, id).expect("the row").revision;
                    (id, revision)
                },
                |(id, revision)| {
                    let removed = store.remove(id, revision).expect("removed");
                    store.publish_now();
                    removed
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

/// SplitMix64: a deterministic fleet.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let out = mix(self.0);
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        out
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Log-uniform in `[lo, hi]`.
    fn log_uniform(&mut self, lo: usize, hi: usize) -> usize {
        let (lo, hi) = ((lo as f64).ln(), (hi as f64).ln());
        let unit = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (lo + (hi - lo) * unit).exp().round() as usize
    }
}

/// The process's resident set in KiB (`VmRSS` of `/proc/self/status`); `0`
/// where there is none to read.
fn resident_kib() -> u64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            let line = status.lines().find_map(|l| l.strip_prefix("VmRSS:"))?;
            line.split_whitespace().next()?.parse().ok()
        })
        .unwrap_or(0)
}

/// Fill an index of `rows` rows through the sink and print what it cost
/// (see the module docs).
fn fill(rows: usize) {
    let mut rng = Rng(1);
    let prefixes: Vec<Vec<u64>> = (0..8u64)
        .map(|shared| {
            let len = rng.log_uniform(16, 64) as u64;
            (0..len).map(|i| mix(shared << 32 | i)).collect()
        })
        .collect();
    let sessions: Vec<Vec<StoredBlock>> = (1..=rows as u64 * 48)
        .map(|session| {
            let mut contents = if rng.below(10) < 6 {
                prefixes[rng.below(prefixes.len())].clone()
            } else {
                Vec::new()
            };
            let body = rng.log_uniform(8, 512) as u64;
            contents.extend((0..body).map(|i| mix(session << 32 | 1 << 31 | i)));
            chain(None, contents.into_iter())
        })
        .collect();
    let mut assigned = vec![Vec::new(); rows];
    for session in 0..sessions.len() {
        for _ in 0..=rng.below(4) {
            assigned[rng.below(rows)].push(session);
        }
    }
    let (store, ids) = publishers(rows);
    let before = resident_kib();
    let mut spent = Duration::ZERO;
    let (mut batches, mut stored) = (0u64, 0usize);
    for (row, id) in ids.iter().enumerate() {
        let mut budget = FILL_BLOCKS;
        let mut own = 0u64;
        while budget > 0 {
            let alone;
            let blocks = if let Some(session) = assigned[row].pop() {
                &sessions[session]
            } else {
                // A chain this row alone holds.
                own += 1;
                let len = rng.log_uniform(8, 256).min(budget.max(8)) as u64;
                let word = (row as u64 + 1) << 48 | own << 16;
                alone = chain(None, (0..len).map(|i| mix(word | i)));
                &alone
            };
            let events: Vec<KvEvent> = blocks
                .chunks(16)
                .enumerate()
                .map(|(at, chunk)| KvEvent::Stored {
                    parent: (at > 0).then(|| blocks[at * 16 - 1].seq_hash),
                    blocks: chunk.to_vec(),
                })
                .collect();
            batches += 1;
            let start = Instant::now();
            let outcome = store.admit(*id, 0, batch(batches), &events);
            spent += start.elapsed();
            assert_eq!(outcome, BatchOutcome::Applied);
            stored += blocks.len();
            budget = budget.saturating_sub(blocks.len());
        }
    }
    let added = resident_kib().saturating_sub(before);
    let held: usize = ids.iter().map(|id| store.indexed_blocks(*id, 0)).sum();
    let refused: u64 = ids
        .iter()
        .filter_map(|id| store.lane_counts(*id, 0))
        .map(|counts| counts.refused_blocks)
        .sum();
    println!(
        "policy_store_kv fill: {rows} rows, {held} blocks held ({refused} declined); \
         {batches} batches of {stored} stored blocks in {:.3} s ({:.0} ns a block); \
         resident memory +{:.1} MiB ({:.0} bytes a held block)",
        spent.as_secs_f64(),
        spent.as_nanos() as f64 / stored as f64,
        added as f64 / 1024.0,
        added as f64 * 1024.0 / held as f64,
    );
}

criterion_group!(benches, turns, threads, leaves);

fn main() {
    let rows = std::env::var("STORE_KV_FILL")
        .ok()
        .and_then(|rows| rows.parse().ok());
    if let Some(rows) = rows {
        fill(rows);
        return;
    }
    benches();
    Criterion::default().configure_from_args().final_summary();
}
