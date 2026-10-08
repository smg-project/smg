//! Open-loop replay of an exported Mooncake indexer corpus against this crate's indexers, with
//! drain-inclusive accounting.
//!
//! The corpus is a prepared schedule of the public Mooncake trace (after trace duplication,
//! deadline sort and id assignment) in the export format described in `README.md` next to this
//! file. Replaying it here measures the indexer alone, with the harness's own threads and queues:
//!
//! - queries go to `worker_id % query_lanes` lanes, each an OS thread that services lookups
//!   inline; events go to `(worker, dp_rank)`-pinned event lanes assigned round-robin on first
//!   sight, each an OS thread draining an unbounded queue;
//! - one query issuer and `--issuer-threads` event issuers (events sharded by contiguous worker
//!   ranges) issue at absolute deadlines (`clock_nanosleep` + a spin) and never wait for earlier
//!   operations; at equal deadlines queries are published before events;
//! - timing ends at the last completion (drain included); a trial is `generator_valid` when every
//!   operation was issued within 1.01× the window and `kept_up` when replay plus drain fit in
//!   1.10×; a block op is a requested, stored or removed block hash;
//! - lookup `query_service` is the time inside the indexer, `query_scheduled_to_finished` includes
//!   queueing; percentiles use the nearest-rank method.
//!
//! What the harness charges every backend, stated so numbers can be read correctly: lanes are OS
//! threads parked on `std::thread::park`; event queues are `std::sync::mpsc` (unbounded, one
//! consumer). With `--owned-payloads` (default) the lanes pay for owning their events: each event
//! arrives as an owned payload in the engine's wire layout (40 bytes per block, allocated before
//! the trial) that the lane converts into this crate's 16-byte blocks and frees after the apply,
//! and each lookup copies its hashes into this crate's hash type, as an adapter in front of the
//! index would. The binary links mimalloc, so every backend is measured under one allocator.
#![expect(clippy::expect_used, clippy::print_stdout, clippy::print_stderr)]
// The harness pins threads and sleeps to absolute monotonic deadlines through libc, which the
// standard library does not expose; the five calls are wrapped in small checked helpers below.
#![expect(unsafe_code)]
#![recursion_limit = "256"]

use std::{
    collections::BTreeMap,
    hint::black_box,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        mpsc, Arc, Barrier, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use kv_index::{
    ChainBlockMap, Claimed, ContentHash, Control, LaneHooks, LanePool, LanePoolConfig,
    PositionalIndexer, QueueFull, ReferenceIndexer, SequenceHash, ShardedChainIndex, StoredBlock,
    WorkerBlockMap,
};
use rustc_hash::FxHashMap;
use serde_json::json;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MAGIC: &[u8; 8] = b"SMGMCK01";
const EMPTY_OPERATION_ID: u32 = u32::MAX;
const WARMUP_QUERIES: usize = 128;

// ---------------------------------------------------------------------------------------------
// Corpus
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct Totals {
    requests: u64,
    stored_events: u64,
    removed_events: u64,
    cleared_events: u64,
    request_blocks: u64,
    stored_blocks: u64,
    removed_blocks: u64,
}

impl Totals {
    fn events(&self) -> u64 {
        self.stored_events + self.removed_events + self.cleared_events
    }
    fn block_ops(&self) -> u64 {
        self.request_blocks + self.stored_blocks + self.removed_blocks
    }
    /// The block ops the dispatch schedules: all of them, or the events' alone under
    /// `--queries off`.
    fn scheduled_block_ops(&self, queries: Queries) -> u64 {
        match queries {
            Queries::On => self.block_ops(),
            Queries::Off => self.stored_blocks + self.removed_blocks,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum OpKind {
    Query {
        start: u64,
        len: u32,
    },
    Stored {
        dp_rank: u32,
        parent: Option<u64>,
        start: u64,
        len: u32,
    },
    Removed {
        dp_rank: u32,
        start: u64,
        len: u32,
    },
    Cleared {
        dp_rank: u32,
    },
}

#[derive(Clone, Copy, Debug)]
struct Op {
    id: u32,
    deadline_ns: u64,
    worker: u64,
    kind: OpKind,
}

impl Op {
    fn is_query(&self) -> bool {
        matches!(self.kind, OpKind::Query { .. })
    }
    fn dp_rank(&self) -> u32 {
        match self.kind {
            OpKind::Query { .. } => 0,
            OpKind::Stored { dp_rank, .. }
            | OpKind::Removed { dp_rank, .. }
            | OpKind::Cleared { dp_rank } => dp_rank,
        }
    }
}

struct Corpus {
    block_size: u32,
    reference_window_ns: u64,
    trace_duplication_factor: u64,
    trace_length_factor: u64,
    inference_worker_duplication_factor: u64,
    logical_workers: u64,
    totals: Totals,
    trace_path: String,
    /// blake3 of the corpus file, recorded in the result's provenance.
    file_blake3: String,
    hashes: Box<[ContentHash]>,
    blocks: Box<[StoredBlock]>,
    removed: Box<[SequenceHash]>,
    ops: Vec<Op>,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| anyhow::anyhow!("corpus truncated at byte {}", self.at))?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }
    fn u8(&mut self) -> anyhow::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> anyhow::Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> anyhow::Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
}

fn load_corpus(path: &str) -> anyhow::Result<Corpus> {
    let bytes = std::fs::read(path)?;
    let file_blake3 = blake3::hash(&bytes).to_hex().to_string();
    let mut c = Cursor {
        bytes: &bytes,
        at: 0,
    };
    anyhow::ensure!(c.take(8)? == MAGIC, "not an SMG Mooncake corpus: {path}");
    let version = c.u32()?;
    anyhow::ensure!(version == 1, "unsupported corpus format version {version}");
    let block_size = c.u32()?;
    let reference_window_ns = c.u64()?;
    let trace_duplication_factor = c.u64()?;
    let trace_length_factor = c.u64()?;
    let inference_worker_duplication_factor = c.u64()?;
    let logical_workers = c.u64()?;
    let totals = Totals {
        requests: c.u64()?,
        stored_events: c.u64()?,
        removed_events: c.u64()?,
        cleared_events: c.u64()?,
        request_blocks: c.u64()?,
        stored_blocks: c.u64()?,
        removed_blocks: c.u64()?,
    };
    let path_len = c.u64()? as usize;
    let trace_path = String::from_utf8_lossy(c.take(path_len)?).into_owned();
    let n = c.u64()? as usize;
    let mut hashes = Vec::with_capacity(n);
    for _ in 0..n {
        hashes.push(ContentHash(c.u64()?));
    }
    let n = c.u64()? as usize;
    let mut blocks = Vec::with_capacity(n);
    for _ in 0..n {
        let seq = c.u64()?;
        let content = c.u64()?;
        blocks.push(StoredBlock {
            seq_hash: SequenceHash(seq),
            content_hash: ContentHash(content),
        });
    }
    let n = c.u64()? as usize;
    let mut removed = Vec::with_capacity(n);
    for _ in 0..n {
        removed.push(SequenceHash(c.u64()?));
    }
    let n = c.u64()? as usize;
    let mut ops = Vec::with_capacity(n);
    for _ in 0..n {
        let id = c.u32()?;
        let deadline_ns = c.u64()?;
        let worker = c.u64()?;
        let kind = match c.u8()? {
            0 => OpKind::Query {
                start: c.u64()?,
                len: c.u32()?,
            },
            1 => {
                let dp_rank = c.u32()?;
                let _event_id = c.u64()?;
                let has_parent = c.u8()? != 0;
                let parent = c.u64()?;
                let _has_start = c.u8()?;
                let _start_position = c.u32()?;
                OpKind::Stored {
                    dp_rank,
                    parent: has_parent.then_some(parent),
                    start: c.u64()?,
                    len: c.u32()?,
                }
            }
            2 => {
                let dp_rank = c.u32()?;
                let _event_id = c.u64()?;
                OpKind::Removed {
                    dp_rank,
                    start: c.u64()?,
                    len: c.u32()?,
                }
            }
            3 => {
                let dp_rank = c.u32()?;
                let _event_id = c.u64()?;
                OpKind::Cleared { dp_rank }
            }
            other => anyhow::bail!("unknown operation kind {other}"),
        };
        ops.push(Op {
            id,
            deadline_ns,
            worker,
            kind,
        });
    }
    anyhow::ensure!(c.at == bytes.len(), "trailing bytes in corpus");
    // The ids are the sorted positions; verify the invariants the replay relies on.
    for (expected, op) in ops.iter().enumerate() {
        anyhow::ensure!(op.id as usize == expected, "operation ids are not dense");
    }
    anyhow::ensure!(
        ops.windows(2).all(|w| w[0].deadline_ns <= w[1].deadline_ns),
        "operations are not deadline-sorted"
    );
    let mut recount = Totals::default();
    for op in &ops {
        match op.kind {
            OpKind::Query { len, .. } => {
                recount.requests += 1;
                recount.request_blocks += u64::from(len);
            }
            OpKind::Stored { len, .. } => {
                recount.stored_events += 1;
                recount.stored_blocks += u64::from(len);
            }
            OpKind::Removed { len, .. } => {
                recount.removed_events += 1;
                recount.removed_blocks += u64::from(len);
            }
            OpKind::Cleared { .. } => recount.cleared_events += 1,
        }
    }
    anyhow::ensure!(
        recount.block_ops() == totals.block_ops() && recount.events() == totals.events(),
        "corpus totals disagree with its operations"
    );
    Ok(Corpus {
        block_size,
        reference_window_ns,
        trace_duplication_factor,
        trace_length_factor,
        inference_worker_duplication_factor,
        logical_workers,
        totals,
        trace_path,
        file_blake3,
        hashes: hashes.into_boxed_slice(),
        blocks: blocks.into_boxed_slice(),
        removed: removed.into_boxed_slice(),
        ops,
    })
}

// ---------------------------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------------------------

/// What a backend must offer the replay. `Lane` is the per-event-lane state (SMG keeps a block
/// map per worker in the lane that owns the worker, like the gateway's event monitor). Workers
/// are `(worker_id, dp_rank)` pairs as the engine streams key them; a backend interns them as it
/// likes.
trait ReplayBackend: Send + Sync + 'static {
    type Lane: Send;
    fn name(&self) -> &'static str;
    fn new_lane(&self) -> Self::Lane;
    /// A lane with its index and the CPUs it is pinned to (one CPU under `--pin-event-lanes`);
    /// backends that place state by lane override it.
    fn new_lane_for(&self, _lane: usize, _cpus: &[usize]) -> Self::Lane {
        self.new_lane()
    }
    /// Figures to print and record at the end of a run, if the backend has any.
    fn report(&self) -> Option<String> {
        None
    }
    /// Apply one stored event; `false` when the backend rejected it (counted).
    fn apply_stored(
        &self,
        lane: &mut Self::Lane,
        worker: (u64, u32),
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
    ) -> bool;
    fn apply_removed(
        &self,
        lane: &mut Self::Lane,
        worker: (u64, u32),
        hashes: &[SequenceHash],
    ) -> bool;
    fn apply_cleared(&self, lane: &mut Self::Lane, worker: (u64, u32)) -> bool;
    /// The shard an event lane's workers live in (its socket under `--shards` with pinned
    /// lanes); the lane pools of `--lane-scheduling stealing` are one per shard, so a stolen
    /// worker is always applied by a lane of its own shard.
    fn lane_shard(&self, _lane: usize, _cpus: &[usize]) -> usize {
        0
    }
    /// Answer one lookup; the return value is only consumed by `black_box`.
    fn lookup(&self, hashes: &[ContentHash]) -> usize;
    /// Shards a lookup can be fanned out over under `--lookups per-shard`; one when the backend
    /// has no shards (the fan-out then is `all-shards`).
    fn lookup_shards(&self) -> usize {
        1
    }
    /// The CPUs of the event lanes placed on `shard`, when the backend places lanes by shard;
    /// the query lanes of that shard float over them.
    fn shard_cpus(&self, _shard: usize) -> Option<Vec<usize>> {
        None
    }
    /// Answer one lookup over `shard` alone, appending its `(worker, score)` pairs to `out`:
    /// the per-shard half of a fanned-out lookup. A backend with one shard never sees it.
    fn lookup_shard(&self, _shard: usize, hashes: &[ContentHash], _out: &mut Vec<(u32, u32)>) {
        black_box(self.lookup(hashes));
    }
}

/// One stored block in the engine's wire layout: two hashes and an always-empty multimodal slot.
struct WireBlock {
    block_hash: u64,
    tokens_hash: u64,
    /// Never set by the Mooncake trace; present so the record has the wire layout's size.
    #[expect(dead_code)]
    mm_extra_info: Option<Vec<u64>>,
}
const _: () = assert!(size_of::<WireBlock>() == 40);

/// The owned event payload an issuer moves to a lane under `--owned-payloads`; `None` when
/// the lane reads the event from the corpus slabs instead.
enum Payload {
    None,
    Stored(Vec<WireBlock>),
    Removed(Vec<u64>),
}

struct EventMsg {
    id: u32,
    /// The worker's slot (its rank of first appearance in the corpus), the lane pool's key.
    slot: u32,
    payload: Payload,
}

struct Positional {
    inner: PositionalIndexer,
}

struct PositionalLane {
    workers: FxHashMap<(u64, u32), (u32, WorkerBlockMap)>,
}

impl ReplayBackend for Positional {
    type Lane = PositionalLane;

    fn name(&self) -> &'static str {
        "smg-positional"
    }

    fn new_lane(&self) -> Self::Lane {
        PositionalLane {
            workers: FxHashMap::default(),
        }
    }

    fn apply_stored(
        &self,
        lane: &mut Self::Lane,
        worker: (u64, u32),
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
    ) -> bool {
        let (smg_id, blocks_map) = lane.workers.entry(worker).or_insert_with(|| {
            let id = self
                .inner
                .intern_worker(&format!("{}:{}", worker.0, worker.1))
                .expect("worker id space");
            (id, WorkerBlockMap::default())
        });
        self.inner
            .apply_stored(*smg_id, blocks, parent, blocks_map)
            .is_ok()
    }

    fn apply_removed(
        &self,
        lane: &mut Self::Lane,
        worker: (u64, u32),
        hashes: &[SequenceHash],
    ) -> bool {
        let Some((smg_id, blocks_map)) = lane.workers.get_mut(&worker) else {
            return false;
        };
        self.inner.apply_removed(*smg_id, hashes, blocks_map);
        true
    }

    fn apply_cleared(&self, lane: &mut Self::Lane, worker: (u64, u32)) -> bool {
        if let Some((smg_id, blocks_map)) = lane.workers.get_mut(&worker) {
            self.inner.apply_cleared(*smg_id, blocks_map);
        }
        true
    }

    fn lookup(&self, hashes: &[ContentHash]) -> usize {
        self.inner.find_matches(hashes, false).scores.len()
    }
}

struct Chain {
    inner: ShardedChainIndex,
    /// The shard of every backend CPU (by position in the backend CPU list): the NUMA node's
    /// index when the list spans exactly `shards` nodes, else contiguous groups of the list.
    shard_of_cpu: FxHashMap<usize, usize>,
    event_lanes: usize,
    /// Under `--count-shard-heads`: lookups by how many shards held the request's first block
    /// (index = that count); a shared counter the query lanes write, so a diagnostic only.
    heads_held: Option<Vec<AtomicUsize>>,
}

impl Chain {
    fn shard_for(&self, lane: usize, cpus: &[usize]) -> usize {
        let shards = self.inner.shards();
        if shards == 1 {
            return 0;
        }
        match cpus {
            [cpu] => self.shard_of_cpu.get(cpu).copied().unwrap_or(0),
            // A floating lane has no socket: shards by lane index, exact but without the
            // placement the split exists for.
            _ => lane * shards / self.event_lanes.max(1),
        }
    }
}

struct ChainLane {
    shard: usize,
    workers: FxHashMap<(u64, u32), (u32, ChainBlockMap)>,
}

impl ReplayBackend for Chain {
    type Lane = ChainLane;

    fn name(&self) -> &'static str {
        "smg-chain"
    }

    fn new_lane(&self) -> Self::Lane {
        ChainLane {
            shard: 0,
            workers: FxHashMap::default(),
        }
    }

    fn new_lane_for(&self, lane: usize, cpus: &[usize]) -> Self::Lane {
        ChainLane {
            shard: self.shard_for(lane, cpus),
            workers: FxHashMap::default(),
        }
    }

    fn lane_shard(&self, lane: usize, cpus: &[usize]) -> usize {
        self.shard_for(lane, cpus)
    }

    fn report(&self) -> Option<String> {
        let mut out = format!("shards = {}", self.inner.shards());
        for (shard, stats) in self.inner.shard_stats().iter().enumerate() {
            out.push_str(&format!(
                "\n  shard {shard}: distinct blocks {} runs live {} blocks in live runs {} arena bytes {} (chunks {}, free-listed {}) slab bytes {} partial entries {} (max {} per run) child entries {} (tombstones {})",
                self.inner.shard(shard).entry_count(),
                stats.runs_live,
                stats.blocks_live,
                stats.arena_bytes,
                stats.arena_chunk_bytes,
                stats.arena_free_bytes,
                stats.slab_bytes,
                stats.partial_entries,
                stats.max_partials,
                stats.child_entries,
                stats.child_tombstones
            ));
        }
        let total = self.inner.stats();
        out.push_str(&format!(
            "\n  total: memberships {} distinct blocks summed over shards {} arena bytes {} chunks {} slab bytes {}",
            self.inner.current_size(),
            self.inner.entry_count(),
            total.arena_bytes,
            total.arena_chunk_bytes,
            total.slab_bytes
        ));
        if let Some(counts) = &self.heads_held {
            let counts: Vec<usize> = counts.iter().map(|c| c.load(Ordering::Relaxed)).collect();
            out.push_str(&format!(
                "\n  lookups by shards holding the first block (0, 1, 2, ..): {counts:?}"
            ));
        }
        Some(out)
    }

    fn apply_stored(
        &self,
        lane: &mut Self::Lane,
        worker: (u64, u32),
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
    ) -> bool {
        let (smg_id, blocks_map) = lane.workers.entry(worker).or_insert_with(|| {
            let id = self
                .inner
                .intern_worker_in(lane.shard, &format!("{}:{}", worker.0, worker.1))
                .expect("worker slots; raise --max-workers");
            (id, ChainBlockMap::default())
        });
        self.inner
            .apply_stored(*smg_id, blocks, parent, blocks_map)
            .is_ok()
    }

    fn apply_removed(
        &self,
        lane: &mut Self::Lane,
        worker: (u64, u32),
        hashes: &[SequenceHash],
    ) -> bool {
        let Some((smg_id, blocks_map)) = lane.workers.get_mut(&worker) else {
            return false;
        };
        self.inner.apply_removed(*smg_id, hashes, blocks_map);
        true
    }

    fn apply_cleared(&self, lane: &mut Self::Lane, worker: (u64, u32)) -> bool {
        if let Some((smg_id, blocks_map)) = lane.workers.get_mut(&worker) {
            self.inner.apply_cleared(*smg_id, blocks_map);
        }
        true
    }

    fn lookup(&self, hashes: &[ContentHash]) -> usize {
        if let (Some(counts), Some(first)) = (&self.heads_held, hashes.first()) {
            let held = self.inner.shards_holding_head(first.0);
            counts[held.min(counts.len() - 1)].fetch_add(1, Ordering::Relaxed);
        }
        let mut scored = 0usize;
        self.inner
            .score_into(hashes, |content| content.0, false, |_, _| scored += 1);
        scored
    }

    fn lookup_shards(&self) -> usize {
        self.inner.shards()
    }

    fn shard_cpus(&self, shard: usize) -> Option<Vec<usize>> {
        let mut cpus: Vec<usize> = self
            .shard_of_cpu
            .iter()
            .filter(|(_, &s)| s == shard)
            .map(|(&cpu, _)| cpu)
            .collect();
        cpus.sort_unstable();
        (!cpus.is_empty()).then_some(cpus)
    }

    fn lookup_shard(&self, shard: usize, hashes: &[ContentHash], out: &mut Vec<(u32, u32)>) {
        self.inner.score_shard_into(
            shard,
            hashes,
            |content| content.0,
            false,
            |worker, score| out.push((worker, score)),
        );
    }
}

struct Reference {
    inner: Mutex<ReferenceIndexer>,
    ids: Mutex<FxHashMap<(u64, u32), u32>>,
}

impl Reference {
    fn id(&self, key: (u64, u32)) -> u32 {
        let mut ids = self.ids.lock().unwrap_or_else(|e| e.into_inner());
        let next = ids.len() as u32;
        *ids.entry(key).or_insert(next)
    }
}

impl ReplayBackend for Reference {
    type Lane = ();

    fn name(&self) -> &'static str {
        "smg-reference"
    }

    fn new_lane(&self) -> Self::Lane {}

    fn apply_stored(
        &self,
        _lane: &mut Self::Lane,
        worker: (u64, u32),
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
    ) -> bool {
        let worker = self.id(worker);
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .apply_stored(worker, blocks, parent)
            .is_ok()
    }

    fn apply_removed(
        &self,
        _lane: &mut Self::Lane,
        worker: (u64, u32),
        hashes: &[SequenceHash],
    ) -> bool {
        let worker = self.id(worker);
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .apply_removed(worker, hashes);
        true
    }

    fn apply_cleared(&self, _lane: &mut Self::Lane, worker: (u64, u32)) -> bool {
        let worker = self.id(worker);
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .apply_cleared(worker);
        true
    }

    fn lookup(&self, hashes: &[ContentHash]) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .find_matches(hashes)
            .len()
    }
}

/// Discards events and answers lookups with the sequence length: measures the harness alone
/// (issuers, queues, lanes, timing) with no indexer behind it, which bounds what any backend can
/// be measured at on a given layout.
struct Null;

impl ReplayBackend for Null {
    type Lane = ();

    fn name(&self) -> &'static str {
        "null"
    }

    fn new_lane(&self) -> Self::Lane {}

    fn apply_stored(
        &self,
        _lane: &mut Self::Lane,
        _worker: (u64, u32),
        _blocks: &[StoredBlock],
        _parent: Option<SequenceHash>,
    ) -> bool {
        true
    }

    fn apply_removed(
        &self,
        _lane: &mut Self::Lane,
        _worker: (u64, u32),
        _hashes: &[SequenceHash],
    ) -> bool {
        true
    }

    fn apply_cleared(&self, _lane: &mut Self::Lane, _worker: (u64, u32)) -> bool {
        true
    }

    fn lookup(&self, hashes: &[ContentHash]) -> usize {
        hashes.len()
    }
}

// ---------------------------------------------------------------------------------------------
// Lanes
// ---------------------------------------------------------------------------------------------

struct QueryLane {
    slots: Box<[AtomicU32]>,
    published: AtomicUsize,
    closed: AtomicBool,
    consumer: Mutex<Option<thread::Thread>>,
}

impl QueryLane {
    fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity)
                .map(|_| AtomicU32::new(EMPTY_OPERATION_ID))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            published: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            consumer: Mutex::new(None),
        }
    }
    fn wake(&self) {
        if let Some(consumer) = self
            .consumer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
        {
            consumer.unpark();
        }
    }
    fn publish(&self, count: usize) {
        self.published.store(count, Ordering::Release);
        self.wake();
    }
    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.wake();
    }
}

#[derive(Clone, Copy, Default)]
struct QueryCompletion {
    id: u32,
    started_ns: u64,
    finished_ns: u64,
}

/// The hand-over point of one fanned-out lookup (`--lookups per-shard`): the member lanes of a
/// lookup group share one slot per published position, since every member receives the same
/// ids in the same order. A lane that finishes before the others leaves its partial answer and
/// its start time here; the last one merges and records the completion.
#[derive(Default)]
struct FanSlot {
    parked: Mutex<Option<FanPartial>>,
}

struct FanPartial {
    /// The earliest start among the lanes finished so far.
    started_ns: u64,
    /// Their partial answers, concatenated (worker sets are disjoint across shards).
    scores: Vec<(u32, u32)>,
    /// How many lanes have finished.
    finished: usize,
}

/// What a query lane walks: every shard, or its own shard of a fanned-out lookup.
enum LaneLookup {
    AllShards,
    Shard {
        shard: usize,
        fan: usize,
        slots: Arc<[FanSlot]>,
    },
}

/// Per-lane counters of the fan-out, summed per shard at the end (nothing shared on the lookup
/// path).
#[derive(Clone, Copy, Debug, Default)]
struct FanStats {
    /// Lookups this lane walked its shard for.
    lookups: u64,
    /// Of those, lookups whose shard held something of the request.
    with_holders: u64,
    /// Holders this lane's shard reported, summed over its lookups.
    holders: u64,
    /// Lookups this lane completed: it finished last and merged.
    merged: u64,
}

impl FanStats {
    fn add(&mut self, other: &FanStats) {
        self.lookups += other.lookups;
        self.with_holders += other.with_holders;
        self.holders += other.holders;
        self.merged += other.merged;
    }
}

struct QueryLaneOutput {
    completions: Vec<QueryCompletion>,
    failure: Option<&'static str>,
    cpu_ns: u64,
    fan: FanStats,
}

fn query_lane_worker<B: ReplayBackend>(
    backend: Arc<B>,
    lane: Arc<QueryLane>,
    corpus: Arc<Corpus>,
    epoch: Instant,
    cpus: Arc<[usize]>,
    mirror: bool,
    lookup: LaneLookup,
) -> QueryLaneOutput {
    let _ = pin_current_thread(&cpus);
    let cpu_started = thread_cpu_time_ns();
    *lane.consumer.lock().unwrap_or_else(|e| e.into_inner()) = Some(thread::current());
    let mut completions = Vec::with_capacity(lane.slots.len());
    let mut consumed = 0usize;
    let mut failure = None;
    let mut partial: Vec<(u32, u32)> = Vec::new();
    let mut stats = FanStats::default();
    loop {
        let published = lane.published.load(Ordering::Acquire);
        while consumed < published {
            let id = lane.slots[consumed].load(Ordering::Relaxed);
            if id == EMPTY_OPERATION_ID {
                failure = Some("query_lane_missing_published_id");
                break;
            }
            let OpKind::Query { start, len } = corpus.ops[id as usize].kind else {
                failure = Some("query_lane_non_query_id");
                break;
            };
            let hashes = &corpus.hashes[start as usize..start as usize + len as usize];
            let started_ns = elapsed_ns(epoch);
            match &lookup {
                LaneLookup::AllShards => {
                    if mirror {
                        // The adapter's copy into this crate's hash type, inside the timed lookup.
                        let owned: Vec<ContentHash> = hashes.to_vec();
                        black_box(backend.lookup(&owned));
                    } else {
                        black_box(backend.lookup(hashes));
                    }
                    let finished_ns = elapsed_ns(epoch);
                    completions.push(QueryCompletion {
                        id,
                        started_ns,
                        finished_ns,
                    });
                }
                LaneLookup::Shard { shard, fan, slots } => {
                    partial.clear();
                    if mirror {
                        let owned: Vec<ContentHash> = hashes.to_vec();
                        backend.lookup_shard(*shard, &owned, &mut partial);
                    } else {
                        backend.lookup_shard(*shard, hashes, &mut partial);
                    }
                    stats.lookups += 1;
                    stats.with_holders += u64::from(!partial.is_empty());
                    stats.holders += partial.len() as u64;
                    let Some(slot) = slots.get(consumed) else {
                        failure = Some("query_lane_fan_slot_overflow");
                        break;
                    };
                    let mut parked = slot.parked.lock().unwrap_or_else(|e| e.into_inner());
                    match parked.as_mut() {
                        Some(earlier) if earlier.finished + 1 == *fan => {
                            // Last to finish: merge the answers and complete the lookup.
                            let mut merged = std::mem::take(&mut earlier.scores);
                            let started_ns = started_ns.min(earlier.started_ns);
                            *parked = None;
                            drop(parked);
                            merged.extend_from_slice(&partial);
                            black_box(&merged);
                            let finished_ns = elapsed_ns(epoch);
                            completions.push(QueryCompletion {
                                id,
                                started_ns,
                                finished_ns,
                            });
                            stats.merged += 1;
                        }
                        Some(earlier) => {
                            earlier.finished += 1;
                            earlier.started_ns = earlier.started_ns.min(started_ns);
                            earlier.scores.extend_from_slice(&partial);
                        }
                        None => {
                            // First to finish: hand the partial answer over (one small vector
                            // allocated on this lane's socket, freed by the merging lane).
                            *parked = Some(FanPartial {
                                started_ns,
                                scores: partial.clone(),
                                finished: 1,
                            });
                        }
                    }
                }
            }
            consumed += 1;
        }
        if failure.is_some() {
            break;
        }
        if lane.closed.load(Ordering::Acquire) && consumed == lane.published.load(Ordering::Acquire)
        {
            break;
        }
        if consumed == lane.published.load(Ordering::Acquire)
            && !lane.closed.load(Ordering::Acquire)
        {
            thread::park();
        }
    }
    QueryLaneOutput {
        completions,
        failure,
        cpu_ns: thread_cpu_time_ns().saturating_sub(cpu_started),
        fan: stats,
    }
}

#[derive(Clone, Copy, Default)]
struct EventCompletion {
    id: u32,
    finished_ns: u64,
    ok: bool,
}

/// Where an event lane runs: its index, the CPUs it is pinned to (one under `--pin-event-lanes`)
/// and whether it prefers its own NUMA node for memory.
struct LanePlacement {
    index: usize,
    cpus: Arc<[usize]>,
    local_memory: bool,
}

/// What an event lane thread returns: its completions, its thread CPU time, the part of it
/// spent inside the apply calls (the rest is the lane's own loop, or the pool's overhead under
/// `--lane-scheduling stealing`), and the minor page faults it took during the run.
struct LaneOutcome {
    completions: Vec<EventCompletion>,
    cpu_ns: u64,
    apply_ns: u64,
    minor_faults: u64,
}

fn event_lane_worker<B: ReplayBackend>(
    backend: Arc<B>,
    receiver: mpsc::Receiver<EventMsg>,
    corpus: Arc<Corpus>,
    epoch: Instant,
    placement: LanePlacement,
    expected: usize,
) -> LaneOutcome {
    place_lane(&placement);
    let LanePlacement {
        index: lane_index,
        cpus,
        ..
    } = placement;
    let cpu_started = thread_cpu_time_ns();
    let faults_started = thread_minor_faults();
    let mut lane = backend.new_lane_for(lane_index, &cpus);
    let mut completions = Vec::with_capacity(expected);
    let mut apply_ns = 0u64;
    while let Ok(EventMsg { id, payload, .. }) = receiver.recv() {
        let started = Instant::now();
        let ok = apply_event(&*backend, &mut lane, &corpus, id, payload);
        apply_ns += started.elapsed().as_nanos() as u64;
        completions.push(EventCompletion {
            id,
            finished_ns: elapsed_ns(epoch),
            ok,
        });
    }
    LaneOutcome {
        completions,
        cpu_ns: thread_cpu_time_ns().saturating_sub(cpu_started),
        apply_ns,
        minor_faults: thread_minor_faults().saturating_sub(faults_started),
    }
}

/// Pin an event lane's thread and set its memory policy as its placement says.
fn place_lane(placement: &LanePlacement) {
    let LanePlacement {
        index: lane_index,
        cpus,
        local_memory,
    } = placement;
    let _ = pin_current_thread(cpus);
    if *local_memory {
        match cpus.first().and_then(|&cpu| node_of_cpu(cpu)) {
            Some(node) => {
                if let Err(err) = prefer_node(node) {
                    println!(
                        "event lane {lane_index}: set_mempolicy for node {node} failed: {err}"
                    );
                }
            }
            None => println!(
                "event lane {lane_index}: no NUMA node for CPUs {cpus:?}, memory policy inherited"
            ),
        }
    }
}

/// Apply one event to the backend through a lane's (or a worker's) state; `false` when the
/// backend rejected it.
fn apply_event<B: ReplayBackend>(
    backend: &B,
    lane: &mut B::Lane,
    corpus: &Corpus,
    id: u32,
    payload: Payload,
) -> bool {
    {
        let op = &corpus.ops[id as usize];
        let worker = (op.worker, op.dp_rank());
        match (&op.kind, payload) {
            (&OpKind::Stored { parent, .. }, Payload::Stored(wire)) => {
                // The adapter's copy, wire records into this crate's blocks; the payload is
                // freed after the apply, as a lane frees the event it was handed.
                let owned: Vec<StoredBlock> = wire
                    .iter()
                    .map(|block| StoredBlock {
                        seq_hash: SequenceHash(block.block_hash),
                        content_hash: ContentHash(block.tokens_hash),
                    })
                    .collect();
                let ok = backend.apply_stored(lane, worker, &owned, parent.map(SequenceHash));
                drop(wire);
                ok
            }
            (
                &OpKind::Stored {
                    parent, start, len, ..
                },
                Payload::None,
            ) => backend.apply_stored(
                lane,
                worker,
                &corpus.blocks[start as usize..start as usize + len as usize],
                parent.map(SequenceHash),
            ),
            (&OpKind::Removed { .. }, Payload::Removed(wire)) => {
                let owned: Vec<SequenceHash> =
                    wire.iter().map(|&hash| SequenceHash(hash)).collect();
                let ok = backend.apply_removed(lane, worker, &owned);
                drop(wire);
                ok
            }
            (&OpKind::Removed { start, len, .. }, Payload::None) => backend.apply_removed(
                lane,
                worker,
                &corpus.removed[start as usize..start as usize + len as usize],
            ),
            (&OpKind::Cleared { .. }, _) => backend.apply_cleared(lane, worker),
            _ => false,
        }
    }
}

/// How long a pooled lane blocks on its channel when nothing is ready anywhere in its pool.
const POOL_WAIT_STEALABLE: Duration = Duration::from_micros(100);
const POOL_WAIT_QUIET: Duration = Duration::from_millis(1);

/// An event lane under `--lane-scheduling stealing`: a lane of its shard's `LanePool`. It drains
/// its own channel into the pool's per-worker queues (its ingress is unchanged: the issuer still
/// sends a worker's events to the worker's home lane) and serves ready workers from any lane of
/// the pool, whole workers at a time, so a worker's events keep their order while a quiet lane
/// works off a stalled lane's backlog. The per-worker state is a backend lane state holding that
/// one worker, created by whichever lane of the pool first serves it, which under pinned lanes is
/// on the worker's own shard.
struct PoolLane<'a, B: ReplayBackend> {
    backend: &'a B,
    corpus: &'a Corpus,
    receiver: mpsc::Receiver<EventMsg>,
    pool: &'a LanePool<B::Lane, EventMsg>,
    /// This lane's index within its pool.
    lane: usize,
    /// This lane's index among all event lanes (what `new_lane_for` places by).
    lane_index: usize,
    cpus: Arc<[usize]>,
    epoch: Instant,
    /// Lanes of this pool whose channel is still open; the pool is done when it is zero and
    /// nothing is queued or held.
    open_channels: &'a AtomicUsize,
    held_total: &'a AtomicUsize,
    closed: bool,
    /// An event the pool refused (the worker's queue at its cap), re-offered before reading on.
    held: Option<EventMsg>,
    completions: Vec<EventCompletion>,
    enqueue_ns: u64,
    stealable_idle_turns: u64,
    apply_ns: u64,
}

impl<B: ReplayBackend> PoolLane<'_, B> {
    fn offer(&mut self, msg: EventMsg) {
        let started = Instant::now();
        let refused = self.pool.enqueue(self.lane, msg.slot, msg);
        self.enqueue_ns += started.elapsed().as_nanos() as u64;
        if let Err(QueueFull(msg)) = refused {
            self.held = Some(msg);
            self.held_total.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn close(&mut self) {
        if !self.closed {
            self.closed = true;
            self.open_channels.fetch_sub(1, Ordering::AcqRel);
        }
    }

    fn control(&self) -> Control {
        if self.open_channels.load(Ordering::Acquire) == 0
            && self.held_total.load(Ordering::Acquire) == 0
            && self.pool.queued() == 0
        {
            Control::Stop
        } else {
            Control::Continue
        }
    }
}

impl<B: ReplayBackend> LaneHooks<B::Lane, EventMsg> for PoolLane<'_, B> {
    fn apply(&mut self, claimed: Claimed<'_, B::Lane>, msg: EventMsg) {
        let (backend, lane_index, cpus) = (self.backend, self.lane_index, &self.cpus);
        let state = claimed
            .state
            .get_or_insert_with(|| backend.new_lane_for(lane_index, cpus));
        let started = Instant::now();
        let ok = apply_event(backend, state, self.corpus, msg.id, msg.payload);
        self.apply_ns += started.elapsed().as_nanos() as u64;
        self.completions.push(EventCompletion {
            id: msg.id,
            finished_ns: elapsed_ns(self.epoch),
            ok,
        });
    }

    fn pump(&mut self) -> Control {
        if let Some(msg) = self.held.take() {
            self.held_total.fetch_sub(1, Ordering::AcqRel);
            self.offer(msg);
            if self.held.is_some() {
                return Control::Continue;
            }
        }
        if self.closed {
            return self.control();
        }
        loop {
            match self.receiver.try_recv() {
                Ok(msg) => {
                    self.offer(msg);
                    if self.held.is_some() {
                        return Control::Continue;
                    }
                }
                Err(mpsc::TryRecvError::Empty) => return Control::Continue,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.close();
                    return self.control();
                }
            }
        }
    }

    fn wait(&mut self) -> Control {
        let stealable = self.pool.has_stealable();
        if stealable {
            // Nothing was ready or stealable for this lane this turn, yet some lane is flagged.
            self.stealable_idle_turns += 1;
        }
        let timeout = if stealable {
            POOL_WAIT_STEALABLE
        } else {
            POOL_WAIT_QUIET
        };
        if self.closed || self.held.is_some() {
            self.pool.park_lane(self.lane, timeout);
            return self.control();
        }
        match self.receiver.recv_timeout(timeout) {
            Ok(msg) => {
                self.offer(msg);
                Control::Continue
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Control::Continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.close();
                self.control()
            }
        }
    }
}

/// The pool of one shard's event lanes and the counters its lanes share.
struct ShardPool<B: ReplayBackend> {
    pool: LanePool<B::Lane, EventMsg>,
    open_channels: AtomicUsize,
    held_total: AtomicUsize,
    /// Time the lanes spent inside `enqueue` (the worker queue's lock and the ready list's),
    /// summed: a producer-side convoy on a stolen worker's queue shows up here.
    enqueue_ns: AtomicU64,
    /// Turns on which a lane found nothing to serve or steal while the pool still reported a
    /// stealable lane: a stealable flag left set on a lane that is not running shows up here.
    stealable_idle_turns: AtomicU64,
}

#[expect(clippy::too_many_arguments)]
fn event_lane_worker_pooled<B: ReplayBackend>(
    backend: Arc<B>,
    receiver: mpsc::Receiver<EventMsg>,
    corpus: Arc<Corpus>,
    epoch: Instant,
    placement: LanePlacement,
    expected: usize,
    shard_pool: Arc<ShardPool<B>>,
    pool_lane: usize,
) -> LaneOutcome {
    place_lane(&placement);
    let LanePlacement {
        index: lane_index,
        cpus,
        ..
    } = placement;
    let cpu_started = thread_cpu_time_ns();
    let faults_started = thread_minor_faults();
    let mut hooks = PoolLane {
        backend: &*backend,
        corpus: &corpus,
        receiver,
        pool: &shard_pool.pool,
        lane: pool_lane,
        lane_index,
        cpus,
        epoch,
        open_channels: &shard_pool.open_channels,
        held_total: &shard_pool.held_total,
        closed: false,
        held: None,
        completions: Vec::with_capacity(expected),
        enqueue_ns: 0,
        stealable_idle_turns: 0,
        apply_ns: 0,
    };
    shard_pool.pool.run_lane(pool_lane, &mut hooks);
    shard_pool
        .enqueue_ns
        .fetch_add(hooks.enqueue_ns, Ordering::Relaxed);
    shard_pool
        .stealable_idle_turns
        .fetch_add(hooks.stealable_idle_turns, Ordering::Relaxed);
    LaneOutcome {
        completions: hooks.completions,
        cpu_ns: thread_cpu_time_ns().saturating_sub(cpu_started),
        apply_ns: hooks.apply_ns,
        minor_faults: thread_minor_faults().saturating_sub(faults_started),
    }
}

// ---------------------------------------------------------------------------------------------
// Issuers
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct IssueRecord {
    scheduled_ns: u64,
    accepted_ns: u64,
    is_query: bool,
    accepted: bool,
}

struct Clock {
    epoch: Instant,
    monotonic_epoch_ns: u64,
    spin_ns: u64,
}

impl Clock {
    fn new(spin_ns: u64) -> anyhow::Result<Self> {
        Ok(Self {
            epoch: Instant::now(),
            monotonic_epoch_ns: monotonic_now_ns()?,
            spin_ns,
        })
    }
    fn now_ns(&self) -> u64 {
        elapsed_ns(self.epoch)
    }
    fn wait_until(&self, target_ns: u64) {
        let sleep_target = target_ns.saturating_sub(self.spin_ns);
        if sleep_target > self.now_ns() {
            sleep_until_monotonic(self.monotonic_epoch_ns.saturating_add(sleep_target));
        }
        while self.now_ns() < target_ns {
            std::hint::spin_loop();
        }
    }
}

fn elapsed_ns(epoch: Instant) -> u64 {
    epoch.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn monotonic_now_ns() -> anyhow::Result<u64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime writes a timespec it is given a valid pointer to.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    anyhow::ensure!(rc == 0, "clock_gettime failed");
    Ok((ts.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(ts.tv_nsec as u64))
}

fn sleep_until_monotonic(target_ns: u64) {
    let request = libc::timespec {
        tv_sec: (target_ns / 1_000_000_000) as libc::time_t,
        tv_nsec: (target_ns % 1_000_000_000) as libc::c_long,
    };
    loop {
        // SAFETY: an absolute sleep on CLOCK_MONOTONIC with a valid timespec and no remainder.
        let rc = unsafe {
            libc::clock_nanosleep(
                libc::CLOCK_MONOTONIC,
                libc::TIMER_ABSTIME,
                &request,
                std::ptr::null_mut(),
            )
        };
        if rc != libc::EINTR {
            return;
        }
    }
}

fn pin_current_thread(cpus: &[usize]) -> std::io::Result<()> {
    if cpus.is_empty() {
        return Ok(());
    }
    // SAFETY: a zeroed cpu_set_t is a valid empty set; CPU_SET/CPU_ZERO only touch that set;
    // sched_setaffinity(0) applies it to the calling thread.
    unsafe {
        let mut set = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_ZERO(&mut set);
        for &cpu in cpus {
            libc::CPU_SET(cpu, &mut set);
        }
        let rc = libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

/// The NUMA node a CPU belongs to, from `/sys/devices/system/node/node*/cpulist`.
fn node_of_cpu(cpu: usize) -> Option<usize> {
    let nodes = std::fs::read_dir("/sys/devices/system/node").ok()?;
    for entry in nodes.flatten() {
        let name = entry.file_name();
        let Some(number) = name.to_str().and_then(|n| n.strip_prefix("node")) else {
            continue;
        };
        let Ok(node) = number.parse::<usize>() else {
            continue;
        };
        let Ok(list) = std::fs::read_to_string(entry.path().join("cpulist")) else {
            continue;
        };
        if parse_cpu_list(list.trim()).is_ok_and(|cpus| cpus.contains(&cpu)) {
            return Some(node);
        }
    }
    None
}

/// Prefer `node` for this thread's page allocations from now on (`set_mempolicy`,
/// `MPOL_PREFERRED`): the arena chunks, run slab chunks and lane maps a lane first touches land
/// on its own socket whatever the process policy (an interleaving policy set by the launcher,
/// for instance).
fn prefer_node(node: usize) -> std::io::Result<()> {
    let mut mask = [0u64; 16];
    mask[node / 64] |= 1u64 << (node % 64);
    // SAFETY: set_mempolicy reads `maxnode` bits from `mask`, which holds 16 * 64 of them, and
    // changes only the calling thread's allocation policy.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_set_mempolicy,
            libc::MPOL_PREFERRED as libc::c_long,
            mask.as_ptr(),
            (mask.len() * 64) as libc::c_ulong,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

/// Minor page faults taken by the calling thread so far (`getrusage(RUSAGE_THREAD)`), so a lane's
/// record can say whether its time went into faulting in memory; 0 when the call fails.
fn thread_minor_faults() -> u64 {
    // SAFETY: rusage is plain data and getrusage writes the whole struct on success.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut usage) };
    if rc != 0 {
        return 0;
    }
    u64::try_from(usage.ru_minflt).unwrap_or(0)
}

fn thread_cpu_time_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: as in monotonic_now_ns.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if rc != 0 {
        return 0;
    }
    (ts.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(ts.tv_nsec as u64)
}

fn parse_cpu_list(value: &str) -> anyhow::Result<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in value.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let (a, b): (usize, usize) = (a.parse()?, b.parse()?);
            anyhow::ensure!(a <= b, "descending CPU range {part}");
            cpus.extend(a..=b);
        } else {
            cpus.push(part.parse()?);
        }
    }
    Ok(cpus)
}

/// Contiguous worker ranges per event issuer.
fn event_issuer_for(worker: u64, logical_workers: u64, issuers: usize) -> usize {
    let per = (logical_workers as usize).div_ceil(issuers).max(1);
    ((worker as usize) / per).min(issuers - 1)
}

struct Shared {
    clock: Clock,
    start_ns: AtomicUsize,
    /// Queries of each deadline group not yet published; its events wait for zero.
    deadline_pending: Box<[AtomicU32]>,
    peer_failed: AtomicBool,
}

fn issue_queries(
    shared: &Shared,
    corpus: &Corpus,
    dispatch: &[(u32, u32, u16)], // (id, deadline_group, lookup group)
    lanes: &[Arc<QueryLane>],
    fan: usize, // lanes per lookup group: the lookup's shards under `--lookups per-shard`
    records: &mut Vec<(u32, IssueRecord)>,
) -> (u64, Option<&'static str>) {
    let cpu_started = thread_cpu_time_ns();
    let start_ns = shared.start_ns.load(Ordering::Acquire) as u64;
    let mut cursors = vec![0usize; lanes.len()];
    let mut touched = Vec::with_capacity(lanes.len());
    let mut touched_flags = vec![false; lanes.len()];
    let mut i = 0usize;
    let mut failure = None;
    while i < dispatch.len() && failure.is_none() {
        let deadline_ns = corpus.ops[dispatch[i].0 as usize].deadline_ns;
        let group = dispatch[i].1 as usize;
        shared
            .clock
            .wait_until(start_ns.saturating_add(deadline_ns));
        touched.clear();
        let group_start = i;
        while i < dispatch.len() && corpus.ops[dispatch[i].0 as usize].deadline_ns == deadline_ns {
            let (id, _, target) = dispatch[i];
            let first_lane = target as usize * fan;
            for lane in first_lane..first_lane + fan {
                let Some(slot) = lanes[lane].slots.get(cursors[lane]) else {
                    failure = Some("issuer_query_lane_overflow");
                    break;
                };
                slot.store(id, Ordering::Relaxed);
                cursors[lane] += 1;
                if !touched_flags[lane] {
                    touched_flags[lane] = true;
                    touched.push(lane);
                }
            }
            if failure.is_some() {
                break;
            }
            records.push((
                id,
                IssueRecord {
                    scheduled_ns: start_ns.saturating_add(deadline_ns),
                    accepted_ns: shared.clock.now_ns(),
                    is_query: true,
                    accepted: true,
                },
            ));
            i += 1;
        }
        for &lane in &touched {
            lanes[lane].publish(cursors[lane]);
            touched_flags[lane] = false;
        }
        if failure.is_some() {
            break;
        }
        shared.deadline_pending[group].fetch_sub((i - group_start) as u32, Ordering::Release);
    }
    if failure.is_some() {
        shared.peer_failed.store(true, Ordering::Release);
    }
    (thread_cpu_time_ns().saturating_sub(cpu_started), failure)
}

/// The owned payload of one event in the engine's wire layout, as its lane receives it under
/// `--owned-payloads`.
fn payload_for(corpus: &Corpus, op: &Op) -> Payload {
    match op.kind {
        OpKind::Stored { start, len, .. } => Payload::Stored(
            corpus.blocks[start as usize..start as usize + len as usize]
                .iter()
                .map(|block| WireBlock {
                    block_hash: block.seq_hash.0,
                    tokens_hash: block.content_hash.0,
                    mm_extra_info: None,
                })
                .collect(),
        ),
        OpKind::Removed { start, len, .. } => Payload::Removed(
            corpus.removed[start as usize..start as usize + len as usize]
                .iter()
                .map(|hash| hash.0)
                .collect(),
        ),
        _ => Payload::None,
    }
}

/// One event in an issuer's dispatch: its operation, deadline group, lane and (under
/// `--owned-payloads`) the owned payload the lane receives.
struct EventDispatch {
    id: u32,
    group: u32,
    lane: u16,
    slot: u32,
    payload: Payload,
}

fn issue_events(
    shared: &Shared,
    corpus: &Corpus,
    dispatch: Vec<EventDispatch>,
    senders: &[mpsc::Sender<EventMsg>],
    records: &mut Vec<(u32, IssueRecord)>,
) -> (u64, Option<&'static str>) {
    let cpu_started = thread_cpu_time_ns();
    let start_ns = shared.start_ns.load(Ordering::Acquire) as u64;
    let mut entries = dispatch.into_iter().peekable();
    let mut failure = None;
    while let Some(head) = entries.peek() {
        let deadline_ns = corpus.ops[head.id as usize].deadline_ns;
        let group = head.group as usize;
        shared
            .clock
            .wait_until(start_ns.saturating_add(deadline_ns));
        // Queries of this deadline are published before its events.
        while shared.deadline_pending[group].load(Ordering::Acquire) != 0 {
            if shared.peer_failed.load(Ordering::Acquire) {
                failure = Some("issuer_peer_failed");
                break;
            }
            std::hint::spin_loop();
        }
        if failure.is_some() {
            break;
        }
        while entries
            .peek()
            .is_some_and(|entry| corpus.ops[entry.id as usize].deadline_ns == deadline_ns)
        {
            let Some(entry) = entries.next() else {
                break;
            };
            let message = EventMsg {
                id: entry.id,
                slot: entry.slot,
                payload: entry.payload,
            };
            if senders[entry.lane as usize].send(message).is_err() {
                failure = Some("issuer_event_lane_offline");
                break;
            }
            records.push((
                entry.id,
                IssueRecord {
                    scheduled_ns: start_ns.saturating_add(deadline_ns),
                    accepted_ns: shared.clock.now_ns(),
                    is_query: false,
                    accepted: true,
                },
            ));
        }
        if failure.is_some() {
            break;
        }
    }
    if failure.is_some() {
        shared.peer_failed.store(true, Ordering::Release);
    }
    (thread_cpu_time_ns().saturating_sub(cpu_started), failure)
}

// ---------------------------------------------------------------------------------------------
// Chain
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum LaneMemory {
    /// The process policy, whatever the launcher set.
    Inherit,
    /// Each pinned event lane prefers its own NUMA node (`set_mempolicy(MPOL_PREFERRED)`).
    Local,
}

/// Whether the corpus's lookups are issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Queries {
    /// Queries and events, the corpus as recorded.
    On,
    /// Events only: queries are dropped at dispatch, the deadline schedule is kept and the query
    /// lanes stay idle; offered, issued and achieved rates and the kept-up verdict count event
    /// block ops alone, so the row is a lane-cost discriminator and not comparable with mixed
    /// rows (the JSON says `"queries": "off"` and `total_requests` is 0).
    Off,
}

/// Which thread builds the owned event payloads of `--owned-payloads`, and so which
/// allocator arena and NUMA node they live on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum LaneScheduling {
    /// Every event lane applies its own workers' events, in order, and nothing else (today).
    Owned,
    /// Each shard's event lanes form one lane pool: an idle lane serves any ready worker of
    /// its shard, and takes one from a lane that is not running once `--steal-after` events
    /// are queued behind it.
    Stealing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum PayloadHome {
    /// The main thread, before the trial, as a preparation step does: one arena for every
    /// payload, which every lane frees into. On two sockets that one arena's lock and free lists
    /// bounce between the sockets on every event, and the lanes' CPU per block doubles.
    Main,
    /// Each event issuer, in its pinned thread, for its own dispatch: payloads live on the
    /// issuer's socket and the lanes free into the issuer's arena (under `--issuer-by-lane` the
    /// same socket), so a two-socket row measures the index and not the allocator.
    Issuer,
}

/// How a lookup is spread over the index's shards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Lookups {
    /// One query lane walks every shard (the lane floats over all lane cores).
    AllShards,
    /// A lookup fans out to one query lane per shard, each floating over its shard's cores and
    /// walking only its shard; the lane finishing last merges the partial answers. Worker sets
    /// are disjoint across shards, so the merge is a concatenation and the answer is
    /// `all-shards`'s. The remote half of a lookup becomes one hand-over of a small vector
    /// instead of remote reads of the lines the local event lanes write. One shard: as
    /// `all-shards`.
    PerShard,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendKind {
    /// This crate's event-driven PositionalIndexer.
    Positional,
    /// The single-threaded reference indexer (small corpora only).
    Reference,
    /// This crate's run-compressed ChainIndex (worker slots from `--max-workers`); `run` is the
    /// name it had and stays accepted until the positional indexer's removal.
    #[value(alias = "run")]
    Chain,
    /// No indexer: the harness's own ceiling on this layout.
    Null,
}

#[derive(Parser, Debug)]
#[command(
    about = "Open-loop replay of an exported Mooncake indexer corpus with drain-inclusive accounting"
)]
struct Args {
    /// Corpus file in the export format described in benches/README.md: a prepared schedule of
    /// the Mooncake trace.
    corpus: String,
    #[arg(long, value_enum, default_value = "positional")]
    backend: BackendKind,
    /// Jump size of the positional indexer's lookup.
    #[arg(long, default_value = "8")]
    jump_size: usize,
    /// Worker slots of the chain index (one coverage bit per slot per run, at most 1024).
    #[arg(long, default_value = "256")]
    max_workers: usize,
    /// Chain index shards, one per socket of the lane set: each shard is a complete chain index
    /// written only by the lanes placed on it (a pinned lane goes to its CPU's NUMA node's
    /// shard when the lane set spans exactly this many nodes, else to its contiguous group of
    /// the backend CPU list), and a lookup walks every shard. One shard is the chain index itself.
    #[arg(long, default_value = "1")]
    shards: usize,
    /// Memory policy of the event lanes: `inherit` the process policy, or `local`, which makes
    /// each pinned lane prefer its own NUMA node for what it allocates (set_mempolicy).
    #[arg(long, value_enum, default_value = "inherit")]
    lane_memory: LaneMemory,
    /// Diagnostic: count lookups by how many shards hold the request's first block (a shared
    /// counter on the lookup path; not for timing rows).
    #[arg(long)]
    count_shard_heads: bool,
    /// Issue the corpus's lookups (`on`) or events only (`off`, a lane-cost discriminator whose
    /// rates count event block ops alone).
    #[arg(long, value_enum, default_value = "on")]
    queries: Queries,
    /// Spread each lookup over the shards: `all-shards` (one lane walks every shard) or
    /// `per-shard` (one lane per shard on the shard's cores, merged by the last to finish).
    #[arg(long, value_enum, default_value = "all-shards")]
    lookups: Lookups,
    /// Who builds the owned event payloads: the `main` thread before the trial, or each `issuer`
    /// in its own pinned thread (payloads on the issuer's socket).
    #[arg(long, value_enum, default_value = "main")]
    payload_home: PayloadHome,
    /// How event lanes share work: `owned` (each lane applies its own workers only, as a
    /// channel per lane) or `stealing` (the event lanes of each shard are the lanes of one
    /// `LanePool`: any lane serves any ready worker of its shard, whole workers at a time, and a
    /// lane that is not running loses a worker once its backlog is `--steal-after` events deep).
    /// Recorded in the result with the pools' counters.
    #[arg(long, value_enum, default_value = "owned")]
    lane_scheduling: LaneScheduling,
    /// Under `--lane-scheduling stealing`: events a ready worker may have queued on a lane that
    /// is not serving before another lane takes it (the pool's `steal_after`); 0 keeps the
    /// pool's serving-lane rule alone.
    #[arg(long, default_value = "0")]
    steal_after: usize,
    /// Replay window in milliseconds; deadlines are rescaled linearly from the corpus's
    /// reference window when they differ.
    #[arg(long, conflicts_with = "offered_block_ops_per_sec")]
    benchmark_duration_ms: Option<u64>,
    /// Offered rate in block ops per second: sets the window from the corpus's block-op total
    /// (window = total / rate), the knob a sustained-throughput threshold search moves.
    #[arg(long)]
    offered_block_ops_per_sec: Option<f64>,
    #[arg(long, default_value = "128")]
    query_lanes: usize,
    /// Event lanes (OS threads applying events).
    #[arg(long, default_value = "64")]
    event_lanes: usize,
    /// Event issuer threads; events are sharded by contiguous worker ranges.
    #[arg(long, default_value = "4")]
    issuer_threads: usize,
    /// CPUs for the event issuers (one per issuer thread when given).
    #[arg(long)]
    issuer_cpus: Option<String>,
    /// Query issuer threads; query lanes are sharded over them in contiguous ranges (one issuer
    /// caps the generator near 1.5B block ops/s here).
    #[arg(long, default_value = "1")]
    query_issuer_threads: usize,
    /// CPUs for the query issuers (one per thread when given); `--query-issuer-cpu` is an alias.
    #[arg(long, alias = "query-issuer-cpu")]
    query_issuer_cpus: Option<String>,
    /// CPUs for query and event lanes.
    #[arg(long)]
    backend_cpus: Option<String>,
    #[arg(long, default_value = "100")]
    issuer_spin_us: u64,
    #[arg(long, default_value = "250")]
    issue_lag_diagnostic_threshold_us: u64,
    #[arg(long, default_value = "5000")]
    pre_run_quiescence_ms: u64,
    /// Pin each event lane to one backend CPU (round robin) instead of letting it float over
    /// the set; a diagnostic for scheduler effects.
    #[arg(long)]
    pin_event_lanes: bool,
    /// Give each worker's events to the issuer whose lane range holds the worker's lane (issuer k
    /// feeds lanes [k * lanes / issuers, (k + 1) * lanes / issuers)) instead of contiguous
    /// worker-id ranges; with `--pin-event-lanes` and the issuer CPUs listed in lane order every
    /// issuer then sits on the socket of the lanes it feeds. A two-socket diagnostic.
    #[arg(long)]
    issuer_by_lane: bool,
    /// Charge every backend for owning its events: each event arrives as an owned payload in the
    /// engine's wire layout (40 bytes per block, allocated before the trial) that the lane converts
    /// into this crate's blocks and frees after the apply, and each lookup copies its hashes into
    /// this crate's hash type. Off: lanes read the corpus slabs and copy nothing.
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    owned_payloads: bool,
    #[arg(long, default_value = "mooncake_replay_result.json")]
    result_json_output: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let corpus = load_corpus(&args.corpus)?;
    let window_ns = match (args.benchmark_duration_ms, args.offered_block_ops_per_sec) {
        (Some(ms), _) => ms * 1_000_000,
        (None, Some(rate)) => {
            anyhow::ensure!(rate > 0.0, "--offered-block-ops-per-sec must be positive");
            (corpus.totals.scheduled_block_ops(args.queries) as f64 / rate * 1e9).round() as u64
        }
        (None, None) => corpus.reference_window_ns,
    };
    match args.backend {
        BackendKind::Positional => run(
            &args,
            corpus,
            window_ns,
            Arc::new(Positional {
                inner: PositionalIndexer::new(args.jump_size),
            }),
        ),
        BackendKind::Reference => {
            if corpus.ops.len() > 2_000_000 {
                eprintln!(
                    "warning: the reference backend is single-threaded; this corpus is large"
                );
            }
            run(
                &args,
                corpus,
                window_ns,
                Arc::new(Reference {
                    inner: Mutex::new(ReferenceIndexer::new()),
                    ids: Mutex::new(FxHashMap::default()),
                }),
            )
        }
        BackendKind::Chain => {
            let backend_cpus = args
                .backend_cpus
                .as_deref()
                .map(parse_cpu_list)
                .transpose()?
                .unwrap_or_default();
            let shard_of_cpu = shard_map(&backend_cpus, args.shards.max(1));
            run(
                &args,
                corpus,
                window_ns,
                Arc::new(Chain {
                    inner: ShardedChainIndex::new(args.shards.max(1), args.max_workers),
                    shard_of_cpu,
                    event_lanes: args.event_lanes,
                    heads_held: args.count_shard_heads.then(|| {
                        (0..=args.shards.max(1))
                            .map(|_| AtomicUsize::new(0))
                            .collect()
                    }),
                }),
            )
        }
        BackendKind::Null => run(&args, corpus, window_ns, Arc::new(Null)),
    }
}

/// The shard of each backend CPU: the index of its NUMA node among the nodes the list spans when
/// there are exactly `shards` of them, else the CPU's contiguous group of the list.
fn shard_map(backend_cpus: &[usize], shards: usize) -> FxHashMap<usize, usize> {
    let mut map = FxHashMap::default();
    if backend_cpus.is_empty() || shards <= 1 {
        return map;
    }
    let mut nodes: Vec<usize> = Vec::new();
    let mut node_of: Vec<Option<usize>> = Vec::with_capacity(backend_cpus.len());
    for &cpu in backend_cpus {
        let node = node_of_cpu(cpu);
        if let Some(node) = node {
            if !nodes.contains(&node) {
                nodes.push(node);
            }
        }
        node_of.push(node);
    }
    if nodes.len() == shards && node_of.iter().all(Option::is_some) {
        for (&cpu, node) in backend_cpus.iter().zip(&node_of) {
            let node = node.expect("checked");
            map.insert(cpu, nodes.iter().position(|&n| n == node).expect("listed"));
        }
    } else {
        for (position, &cpu) in backend_cpus.iter().enumerate() {
            map.insert(cpu, position * shards / backend_cpus.len());
        }
    }
    map
}

fn run<B: ReplayBackend>(
    args: &Args,
    mut corpus: Corpus,
    window_ns: u64,
    backend: Arc<B>,
) -> anyhow::Result<()> {
    anyhow::ensure!(args.query_lanes > 0 && args.event_lanes > 0 && args.issuer_threads > 0);
    let backend_cpus: Arc<[usize]> = args
        .backend_cpus
        .as_deref()
        .map(parse_cpu_list)
        .transpose()?
        .unwrap_or_default()
        .into();
    let issuer_cpus = args
        .issuer_cpus
        .as_deref()
        .map(parse_cpu_list)
        .transpose()?
        .unwrap_or_default();
    let issuer_threads = if issuer_cpus.is_empty() {
        args.issuer_threads
    } else {
        issuer_cpus.len()
    };
    let query_cpus = args
        .query_issuer_cpus
        .as_deref()
        .map(parse_cpu_list)
        .transpose()?
        .unwrap_or_default();
    let query_issuers = if query_cpus.is_empty() {
        args.query_issuer_threads.max(1)
    } else {
        query_cpus.len()
    };
    // Issuers and lanes must not share cores: a lane on an issuer's core eats the issue schedule
    // and the trial measures the layout mistake, not the indexer.
    if !backend_cpus.is_empty() {
        let overlap: Vec<usize> = issuer_cpus
            .iter()
            .chain(query_cpus.iter())
            .copied()
            .filter(|cpu| backend_cpus.contains(cpu))
            .collect();
        anyhow::ensure!(
            overlap.is_empty(),
            "issuer CPUs {overlap:?} overlap the backend CPU set; give lanes their own cores"
        );
    }
    // The layout, first line of every run log, so the provenance shows it at a glance.
    println!(
        "layout: event issuers {} on {:?}, query issuers {} on {:?}, lanes {} event + {} query on {:?} ({} cores), shards {}, lane memory {:?}, event lanes {}, queries {:?}, lookups {:?}, payload home {:?}, lane scheduling {:?} (steal after {})",
        issuer_threads,
        issuer_cpus,
        query_issuers,
        query_cpus,
        args.event_lanes,
        args.query_lanes,
        backend_cpus,
        backend_cpus.len(),
        args.shards.max(1),
        args.lane_memory,
        if args.pin_event_lanes { "pinned one per core" } else { "floating" },
        args.queries,
        args.lookups,
        args.payload_home,
        args.lane_scheduling,
        args.steal_after,
    );
    if window_ns != corpus.reference_window_ns {
        let reference = corpus.reference_window_ns.max(1) as u128;
        for op in &mut corpus.ops {
            op.deadline_ns = ((op.deadline_ns as u128 * window_ns as u128) / reference) as u64;
        }
    }
    // Lane assignment and deadline groups.
    let mirror = args.owned_payloads;
    // Lanes per lookup: one, or one per shard under `--lookups per-shard` (query lane
    // `group * fan + shard` walks `shard` for lookup group `group`).
    let fan = match args.lookups {
        Lookups::AllShards => 1,
        Lookups::PerShard => backend.lookup_shards().max(1),
    };
    anyhow::ensure!(
        args.query_lanes.is_multiple_of(fan),
        "--query-lanes {} is not a multiple of the {fan} shards a per-shard lookup fans out to",
        args.query_lanes
    );
    let lookup_groups = args.query_lanes / fan;
    let mut lane_capacities = vec![0usize; args.query_lanes];
    let mut deadline_query_counts: Vec<u32> = Vec::new();
    let mut previous_deadline = None;
    // A worker's slot is its rank of first appearance; its lane is the slot modulo the lane count.
    let mut event_lane_of: FxHashMap<(u64, u32), u32> = FxHashMap::default();
    let mut event_lane_expected = vec![0usize; args.event_lanes];
    let mut query_dispatch: Vec<Vec<(u32, u32, u16)>> = vec![Vec::new(); query_issuers];
    let mut event_dispatch: Vec<Vec<EventDispatch>> =
        (0..issuer_threads).map(|_| Vec::new()).collect();
    for op in &corpus.ops {
        if previous_deadline != Some(op.deadline_ns) {
            previous_deadline = Some(op.deadline_ns);
            deadline_query_counts.push(0);
        }
        let group = (deadline_query_counts.len() - 1) as u32;
        if op.is_query() {
            if args.queries == Queries::Off {
                continue;
            }
            let target = (op.worker as usize) % lookup_groups;
            for member in 0..fan {
                lane_capacities[target * fan + member] += 1;
            }
            deadline_query_counts[group as usize] += 1;
            query_dispatch[target * query_issuers / lookup_groups].push((
                op.id,
                group,
                target as u16,
            ));
        } else {
            let next = event_lane_of.len() as u32;
            let slot = *event_lane_of
                .entry((op.worker, op.dp_rank()))
                .or_insert(next);
            let lane = (slot as usize % args.event_lanes) as u16;
            event_lane_expected[lane as usize] += 1;
            let shard = if args.issuer_by_lane {
                (lane as usize) * issuer_threads / args.event_lanes
            } else {
                event_issuer_for(op.worker, corpus.logical_workers, issuer_threads)
            };
            // Under --owned-payloads the payload is built here, before the trial, or by the
            // issuer itself under --payload-home issuer.
            let payload = if mirror && args.payload_home == PayloadHome::Main {
                payload_for(&corpus, op)
            } else {
                Payload::None
            };
            event_dispatch[shard].push(EventDispatch {
                id: op.id,
                group,
                lane,
                slot,
                payload,
            });
        }
    }
    let deadline_pending: Box<[AtomicU32]> = deadline_query_counts
        .iter()
        .map(|&count| AtomicU32::new(count))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    // Quiescence: return preparation pages and let the allocator settle.
    // SAFETY: malloc_trim takes an integer pad and has no other preconditions.
    unsafe {
        libc::malloc_trim(0);
    }
    if args.pre_run_quiescence_ms > 0 {
        thread::sleep(Duration::from_millis(args.pre_run_quiescence_ms));
    }
    let corpus = Arc::new(corpus);
    // Page-touch the corpus once.
    let mut checksum = 0u64;
    for hash in &*corpus.hashes {
        checksum ^= hash.0;
    }
    for block in &*corpus.blocks {
        checksum ^= block.seq_hash.0 ^ block.content_hash.0;
    }
    for op in &corpus.ops {
        checksum ^= op.deadline_ns ^ op.worker ^ u64::from(op.id);
    }
    black_box(checksum);

    pin_current_thread(&backend_cpus)?;
    let clock = Clock::new(args.issuer_spin_us.saturating_mul(1_000))?;
    let epoch = clock.epoch;
    // Fixed lookup warm-up.
    for op in corpus
        .ops
        .iter()
        .filter(|op| op.is_query())
        .take(if args.queries == Queries::On {
            WARMUP_QUERIES
        } else {
            0
        })
    {
        if let OpKind::Query { start, len } = op.kind {
            black_box(
                backend.lookup(&corpus.hashes[start as usize..start as usize + len as usize]),
            );
        }
    }

    // Lanes.
    let lanes: Vec<Arc<QueryLane>> = lane_capacities
        .iter()
        .map(|&capacity| Arc::new(QueryLane::new(capacity)))
        .collect();
    // Under `--lookups per-shard`, the hand-over slots of every lookup group, one per published
    // position (every member lane of a group receives the same ids in the same order).
    let fan_slots: Vec<Arc<[FanSlot]>> = (0..lookup_groups)
        .map(|group| {
            let slots = if fan > 1 {
                lane_capacities[group * fan]
            } else {
                0
            };
            (0..slots)
                .map(|_| FanSlot::default())
                .collect::<Vec<_>>()
                .into()
        })
        .collect();
    let mut query_threads = Vec::with_capacity(lanes.len());
    for (index, lane) in lanes.iter().enumerate() {
        let (backend, lane, corpus) = (Arc::clone(&backend), Arc::clone(lane), Arc::clone(&corpus));
        let (lookup, cpus) = if fan > 1 {
            let shard = index % fan;
            let cpus: Arc<[usize]> = backend
                .shard_cpus(shard)
                .map_or_else(|| Arc::clone(&backend_cpus), Arc::from);
            let lookup = LaneLookup::Shard {
                shard,
                fan,
                slots: Arc::clone(&fan_slots[index / fan]),
            };
            (lookup, cpus)
        } else {
            (LaneLookup::AllShards, Arc::clone(&backend_cpus))
        };
        query_threads.push(thread::spawn(move || {
            query_lane_worker(backend, lane, corpus, epoch, cpus, mirror, lookup)
        }));
    }
    // Wait until every query lane has registered its parker.
    for lane in &lanes {
        while lane
            .consumer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_none()
        {
            thread::yield_now();
        }
    }
    let lane_cpus: Vec<Arc<[usize]>> = (0..args.event_lanes)
        .map(|idx| {
            if args.pin_event_lanes && !backend_cpus.is_empty() {
                Arc::from(vec![backend_cpus[idx % backend_cpus.len()]])
            } else {
                Arc::clone(&backend_cpus)
            }
        })
        .collect();
    // Under `--lane-scheduling stealing` the event lanes of each shard are the lanes of one
    // pool, so a worker is only ever applied by a lane of its own shard; the pool index of a lane
    // is its rank among its shard's lanes.
    let stealing = args.lane_scheduling == LaneScheduling::Stealing;
    let lane_shard: Vec<usize> = (0..args.event_lanes)
        .map(|idx| backend.lane_shard(idx, &lane_cpus[idx]))
        .collect();
    let shard_count = lane_shard.iter().copied().max().map_or(1, |s| s + 1);
    let mut pool_lane_index = vec![0usize; args.event_lanes];
    let mut lanes_per_shard = vec![0usize; shard_count];
    for (idx, &shard) in lane_shard.iter().enumerate() {
        pool_lane_index[idx] = lanes_per_shard[shard];
        lanes_per_shard[shard] += 1;
    }
    let shard_pools: Vec<Option<Arc<ShardPool<B>>>> = lanes_per_shard
        .iter()
        .map(|&lanes| {
            (stealing && lanes > 0).then(|| {
                Arc::new(ShardPool {
                    pool: LanePool::new(LanePoolConfig {
                        lanes,
                        max_workers: event_lane_of.len().max(1),
                        // Never refuse: the harness's own queues are unbounded, and a refusal
                        // here would only move an event into the lane's held slot.
                        depth_cap: corpus.totals.events().try_into().unwrap_or(usize::MAX),
                        batch: 32,
                        steal_after: args.steal_after,
                    }),
                    open_channels: AtomicUsize::new(lanes),
                    held_total: AtomicUsize::new(0),
                    enqueue_ns: AtomicU64::new(0),
                    stealable_idle_turns: AtomicU64::new(0),
                })
            })
        })
        .collect();
    let mut senders = Vec::with_capacity(args.event_lanes);
    let mut event_threads = Vec::with_capacity(args.event_lanes);
    for (idx, &expected) in event_lane_expected.iter().enumerate() {
        let (tx, rx) = mpsc::channel::<EventMsg>();
        senders.push(tx);
        let (backend, corpus) = (Arc::clone(&backend), Arc::clone(&corpus));
        let placement = LanePlacement {
            index: idx,
            cpus: Arc::clone(&lane_cpus[idx]),
            local_memory: matches!(args.lane_memory, LaneMemory::Local),
        };
        match shard_pools[lane_shard[idx]].as_ref().map(Arc::clone) {
            Some(shard_pool) => {
                let pool_lane = pool_lane_index[idx];
                event_threads.push(thread::spawn(move || {
                    event_lane_worker_pooled(
                        backend, rx, corpus, epoch, placement, expected, shard_pool, pool_lane,
                    )
                }));
            }
            None => event_threads.push(thread::spawn(move || {
                event_lane_worker(backend, rx, corpus, epoch, placement, expected)
            })),
        }
    }

    let shared = Shared {
        clock,
        start_ns: AtomicUsize::new(0),
        deadline_pending,
        peer_failed: AtomicBool::new(false),
    };
    let ready = Barrier::new(issuer_threads + query_issuers + 1);
    let start = Barrier::new(issuer_threads + query_issuers + 1);
    let (start_ns, outputs) = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(issuer_threads + query_issuers);
        let (shared_ref, corpus_ref, lanes_ref, ready_ref, start_ref) =
            (&shared, &corpus, &lanes, &ready, &start);
        for (idx, dispatch) in query_dispatch.iter().enumerate() {
            let cpu = query_cpus.get(idx).copied();
            handles.push(scope.spawn(move || {
                let pin = cpu.map_or(Ok(()), |cpu| pin_current_thread(&[cpu]));
                let mut failure = pin.err().map(|_| "issuer_affinity");
                if failure.is_some() {
                    shared_ref.peer_failed.store(true, Ordering::Release);
                }
                ready_ref.wait();
                start_ref.wait();
                let mut records = Vec::with_capacity(dispatch.len());
                let (cpu_ns, f) = if failure.is_none() {
                    issue_queries(
                        shared_ref,
                        corpus_ref,
                        dispatch,
                        lanes_ref,
                        fan,
                        &mut records,
                    )
                } else {
                    (0, None)
                };
                failure = failure.or(f);
                (records, cpu_ns, failure)
            }));
        }
        let issuer_payloads = mirror && args.payload_home == PayloadHome::Issuer;
        for (idx, dispatch) in event_dispatch.into_iter().enumerate() {
            let cpu = issuer_cpus.get(idx).copied();
            let senders = &senders;
            handles.push(scope.spawn(move || {
                let mut dispatch = dispatch;
                let pin = cpu.map_or(Ok(()), |cpu| pin_current_thread(&[cpu]));
                let mut failure = pin.err().map(|_| "issuer_affinity");
                if failure.is_some() {
                    shared_ref.peer_failed.store(true, Ordering::Release);
                }
                if issuer_payloads {
                    // Built here, pinned: the payloads take this thread's arena and NUMA node.
                    for entry in &mut dispatch {
                        entry.payload = payload_for(corpus_ref, &corpus_ref.ops[entry.id as usize]);
                    }
                }
                ready_ref.wait();
                start_ref.wait();
                let mut records = Vec::with_capacity(dispatch.len());
                let (cpu_ns, f) = if failure.is_none() {
                    issue_events(shared_ref, corpus_ref, dispatch, senders, &mut records)
                } else {
                    (0, None)
                };
                failure = failure.or(f);
                (records, cpu_ns, failure)
            }));
        }
        ready.wait();
        let start_ns = shared.clock.now_ns().saturating_add(20_000_000);
        shared.start_ns.store(start_ns as usize, Ordering::Release);
        start.wait();
        let outputs: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("issuer thread panicked"))
            .collect();
        (start_ns, outputs)
    });
    let producer_stop_ns = shared.clock.now_ns();
    drop(senders);
    for lane in &lanes {
        lane.close();
    }
    let mut failure_reasons: Vec<String> = Vec::new();
    let mut query_results = Vec::with_capacity(lanes.len());
    let mut query_lane_cpu_ns = 0u64;
    let mut fan_stats = vec![FanStats::default(); fan];
    for (index, handle) in query_threads.into_iter().enumerate() {
        let output = handle.join().expect("query lane panicked");
        if let Some(failure) = output.failure {
            failure_reasons.push(failure.to_string());
        }
        query_lane_cpu_ns += output.cpu_ns;
        fan_stats[index % fan].add(&output.fan);
        query_results.push(output.completions);
    }
    let outcomes: Vec<LaneOutcome> = event_threads
        .into_iter()
        .map(|h| h.join().expect("event lane panicked"))
        .collect();
    let event_lane_cpu_ns: Vec<u64> = outcomes.iter().map(|o| o.cpu_ns).collect();
    let event_lane_apply_ms: Vec<f64> = outcomes.iter().map(|o| o.apply_ns as f64 / 1e6).collect();
    let event_lane_minor_faults: Vec<u64> = outcomes.iter().map(|o| o.minor_faults).collect();
    let event_results: Vec<Vec<EventCompletion>> =
        outcomes.into_iter().map(|o| o.completions).collect();

    // Merge issue records.
    let n = corpus.ops.len();
    let mut records = vec![IssueRecord::default(); n];
    let mut issuer_cpu_ns = 0u64;
    for (local, cpu_ns, failure) in outputs {
        issuer_cpu_ns += cpu_ns;
        if let Some(failure) = failure {
            failure_reasons.push(failure.to_string());
        }
        for (id, record) in local {
            if records[id as usize].accepted {
                failure_reasons.push("duplicate_issue_record".to_string());
            }
            records[id as usize] = record;
        }
    }
    // Completions by id, with order checks.
    let mut query_done: Vec<Option<QueryCompletion>> = vec![None; n];
    for (lane_idx, completions) in query_results.iter().enumerate() {
        let group = lane_idx / fan;
        let expected: Vec<u32> = query_dispatch
            .iter()
            .flatten()
            .filter(|(_, _, target)| *target as usize == group)
            .map(|(id, _, _)| *id)
            .collect();
        let actual: Vec<u32> = completions.iter().map(|c| c.id).collect();
        let in_order = if fan == 1 {
            expected == actual
        } else {
            // A member lane completes the lookups it merged, in publish order: a subsequence.
            let mut remaining = expected.iter();
            actual.iter().all(|id| remaining.any(|e| e == id))
        };
        if !in_order {
            failure_reasons.push(format!("query_lane_order_{lane_idx}"));
        }
        for c in completions {
            if query_done[c.id as usize].replace(*c).is_some() {
                failure_reasons.push("duplicate_query_completion".to_string());
            }
        }
    }
    let mut event_done: Vec<Option<EventCompletion>> = vec![None; n];
    // A worker's events in the order they were applied: by finish time, since under
    // `--lane-scheduling stealing` a worker's events are applied by more than one lane (one at
    // a time, so finish order is apply order) and its completions sit in several lanes' lists.
    let mut actual_by_worker: BTreeMap<(u64, u32), Vec<(u64, u32)>> = BTreeMap::new();
    let mut failed_events = 0usize;
    for completions in &event_results {
        for c in completions {
            let op = &corpus.ops[c.id as usize];
            actual_by_worker
                .entry((op.worker, op.dp_rank()))
                .or_default()
                .push((c.finished_ns, c.id));
            if !c.ok {
                failed_events += 1;
            }
            if event_done[c.id as usize].replace(*c).is_some() {
                failure_reasons.push("duplicate_event_completion".to_string());
            }
        }
    }
    let actual_by_worker: BTreeMap<(u64, u32), Vec<u32>> = actual_by_worker
        .into_iter()
        .map(|(worker, mut done)| {
            done.sort_unstable();
            (worker, done.into_iter().map(|(_, id)| id).collect())
        })
        .collect();
    let mut expected_by_worker: BTreeMap<(u64, u32), Vec<u32>> = BTreeMap::new();
    for op in corpus.ops.iter().filter(|op| !op.is_query()) {
        expected_by_worker
            .entry((op.worker, op.dp_rank()))
            .or_default()
            .push(op.id);
    }
    let mut fifo_violations = 0usize;
    for (worker, expected) in &expected_by_worker {
        if actual_by_worker.get(worker) != Some(expected) {
            fifo_violations += 1;
        }
    }
    if fifo_violations > 0 {
        failure_reasons.push(format!("event_worker_fifo_{fifo_violations}"));
    }

    let tolerance_ns = args.issue_lag_diagnostic_threshold_us.saturating_mul(1_000);
    let (mut read_lag, mut update_lag, mut queue_wait, mut service, mut query_e2e) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut update_acc_fin, mut update_e2e) = (Vec::new(), Vec::new());
    let (mut delayed_reads, mut delayed_updates, mut races) = (0usize, 0usize, 0usize);
    let mut query_edges = Vec::new();
    let mut update_edges = Vec::new();
    let mut last_completion = 0u64;
    let mut unissued = 0usize;
    let (mut queued_queries_at_stop, mut outstanding_updates_at_stop) = (0usize, 0usize);
    for (id, record) in records.iter().enumerate() {
        if !record.accepted {
            // Under `--queries off` the lookups are left out of the dispatch on purpose.
            if args.queries == Queries::On || !corpus.ops[id].is_query() {
                unissued += 1;
            }
            continue;
        }
        let lag = record.accepted_ns.saturating_sub(record.scheduled_ns);
        if record.is_query {
            read_lag.push(lag);
            delayed_reads += usize::from(lag > tolerance_ns);
            let Some(c) = query_done[id] else {
                failure_reasons.push(format!("missing_query_completion_{id}"));
                continue;
            };
            last_completion = last_completion.max(c.finished_ns);
            queued_queries_at_stop += usize::from(
                record.accepted_ns <= producer_stop_ns && c.started_ns > producer_stop_ns,
            );
            queue_wait.push(c.started_ns.saturating_sub(record.accepted_ns));
            service.push(c.finished_ns.saturating_sub(c.started_ns));
            query_e2e.push(c.finished_ns.saturating_sub(record.scheduled_ns));
            query_edges.push((record.accepted_ns, 1i8));
            query_edges.push((c.started_ns.max(record.accepted_ns), -1i8));
        } else {
            update_lag.push(lag);
            delayed_updates += usize::from(lag > tolerance_ns);
            let Some(c) = event_done[id] else {
                failure_reasons.push(format!("missing_event_completion_{id}"));
                continue;
            };
            last_completion = last_completion.max(c.finished_ns);
            outstanding_updates_at_stop += usize::from(c.finished_ns > producer_stop_ns);
            if c.finished_ns < record.accepted_ns {
                races += 1;
            }
            update_acc_fin.push(c.finished_ns.saturating_sub(record.accepted_ns));
            update_e2e.push(c.finished_ns.saturating_sub(record.scheduled_ns));
            update_edges.push((record.accepted_ns, 1i8));
            update_edges.push((c.finished_ns.max(record.accepted_ns), -1i8));
        }
    }
    if unissued > 0 {
        failure_reasons.push(format!("unissued_operations_{unissued}"));
    }
    if failure_reasons.len() > 32 {
        failure_reasons.truncate(32);
        failure_reasons.push("...".to_string());
    }
    let end_ns = if last_completion > 0 {
        last_completion
    } else {
        producer_stop_ns
    };
    let issue_span_ns = records
        .iter()
        .filter(|r| r.accepted)
        .map(|r| r.accepted_ns)
        .max()
        .unwrap_or(start_ns)
        .saturating_sub(start_ns);
    let drain_ns = end_ns.saturating_sub(producer_stop_ns);
    let elapsed = end_ns.saturating_sub(start_ns).max(1);
    let totals = corpus.totals;
    let (issued_requests, issued_request_blocks) = match args.queries {
        Queries::On => (totals.requests, totals.request_blocks),
        Queries::Off => (0, 0),
    };
    let total_logical_ops = issued_requests + totals.events();
    let total_block_ops = totals.scheduled_block_ops(args.queries);
    let offered_s = window_ns.max(1) as f64 / 1e9;
    let achieved_s = elapsed as f64 / 1e9;
    let issue_s = issue_span_ns.max(1) as f64 / 1e9;
    let issue_span_valid = issue_span_ns <= window_ns.saturating_mul(101) / 100;
    let generator_valid = failure_reasons.is_empty() && issue_span_valid;
    let kept_up = generator_valid && elapsed <= window_ns.saturating_mul(110) / 100;

    // Per-lane diagnostics: CPU time, event count and the time of the
    // last completion of each event lane, which tell scheduler starvation from slower work.
    let event_lane_cpu_ms: Vec<f64> = event_lane_cpu_ns
        .iter()
        .map(|&ns| ns as f64 / 1e6)
        .collect();
    let event_lane_events: Vec<usize> = event_results.iter().map(Vec::len).collect();
    let event_lane_last_finished_ms: Vec<f64> = event_results
        .iter()
        .map(|c| {
            c.last()
                .map_or(0.0, |c| c.finished_ns.saturating_sub(start_ns) as f64 / 1e6)
        })
        .collect();
    let result = json!({
        "schema_version": 3,
        "harness": "smg-mooncake-replay",
        "backend": backend.name(),
        "corpus": args.corpus,
        "owned_payloads": args.owned_payloads,
        "shards": args.shards.max(1),
        "lane_memory": format!("{:?}", args.lane_memory).to_lowercase(),
        "queries": format!("{:?}", args.queries).to_lowercase(),
        "lookups": match args.lookups {
            Lookups::AllShards => "all-shards",
            Lookups::PerShard => "per-shard",
        },
        "payload_home": format!("{:?}", args.payload_home).to_lowercase(),
        "lane_scheduling": format!("{:?}", args.lane_scheduling).to_lowercase(),
        "steal_after": args.steal_after,
        "lane_pool": stealing.then(|| {
            shard_pools
                .iter()
                .enumerate()
                .filter_map(|(shard, pool)| pool.as_ref().map(|pool| (shard, pool)))
                .map(|(shard, shard_pool)| {
                    let m = shard_pool.pool.metrics();
                    json!({
                        "shard": shard,
                        "lanes": shard_pool.pool.config().lanes,
                        "steal_after": shard_pool.pool.config().steal_after,
                        "enqueued": m.enqueued,
                        "applied": m.applied,
                        "rejected": m.rejected,
                        "steals": m.steals,
                        "max_depth": m.max_depth,
                        "max_queued": m.max_queued,
                        "busy_ms": m.busy_ns as f64 / 1e6,
                        "idle_ms": m.idle_ns as f64 / 1e6,
                        "enqueue_wait_ms": shard_pool.enqueue_ns.load(Ordering::Relaxed) as f64 / 1e6,
                        "stealable_idle_turns": shard_pool.stealable_idle_turns.load(Ordering::Relaxed),
                    })
                })
                .collect::<Vec<_>>()
        }),
        "lookup_fanout": (fan > 1).then(|| json!({
            "fan": fan,
            "groups": lookup_groups,
            "shards": fan_stats.iter().enumerate().map(|(shard, s)| json!({
                "shard": shard,
                "cpus": backend.shard_cpus(shard),
                "lookups": s.lookups,
                "with_holders": s.with_holders,
                "holders": s.holders,
                "merged": s.merged,
            })).collect::<Vec<_>>(),
        })),
        "backend_report": backend.report(),
        "provenance": {
            "argv": std::env::args().collect::<Vec<_>>(),
            "binary": std::env::current_exe().ok().map(|p| p.display().to_string()),
            "binary_blake3": std::env::current_exe()
                .ok()
                .and_then(|p| std::fs::read(p).ok())
                .map(|b| blake3::hash(&b).to_hex().to_string()),
            "corpus_blake3": corpus.file_blake3,
            "corpus_reference_window_ns": corpus.reference_window_ns,
            "trace_path": corpus.trace_path,
            "trace_block_size": corpus.block_size,
            "trace_duplication_factor": corpus.trace_duplication_factor,
            "trace_length_factor": corpus.trace_length_factor,
            "inference_worker_duplication_factor": corpus.inference_worker_duplication_factor,
            "num_unique_inference_workers": corpus.logical_workers,
            "jump_size": args.jump_size,
            "issuer_spin_us": args.issuer_spin_us,
            "issue_lag_diagnostic_threshold_us": args.issue_lag_diagnostic_threshold_us,
        },
        "timer": "clock_nanosleep_monotonic_absolute",
        "benchmark_duration_ms": window_ns / 1_000_000,
        "block_size": corpus.block_size,
        "pre_run_quiescence_ms": args.pre_run_quiescence_ms,
        "query_lanes": args.query_lanes,
        "issuer_threads": issuer_threads,
        "event_workers": args.event_lanes,
        "issuer_cpus": issuer_cpus,
        "query_issuer_threads": query_issuers,
        "query_issuer_cpus": query_cpus,
        "backend_cpus": backend_cpus.to_vec(),
        "total_requests": issued_requests,
        "total_events": totals.events(),
        "total_stored_events": totals.stored_events,
        "total_removed_events": totals.removed_events,
        "total_cleared_events": totals.cleared_events,
        "total_request_blocks": issued_request_blocks,
        "total_stored_blocks": totals.stored_blocks,
        "total_removed_blocks": totals.removed_blocks,
        "total_logical_ops": total_logical_ops,
        "total_block_ops": total_block_ops,
        "offered_logical_ops_per_sec": total_logical_ops as f64 / offered_s,
        "actual_issue_logical_ops_per_sec": total_logical_ops as f64 / issue_s,
        "achieved_logical_ops_per_sec": total_logical_ops as f64 / achieved_s,
        "offered_block_ops_per_sec": total_block_ops as f64 / offered_s,
        "actual_issue_block_ops_per_sec": total_block_ops as f64 / issue_s,
        "achieved_block_ops_per_sec": total_block_ops as f64 / achieved_s,
        "read_issue_lag": distribution(read_lag),
        "update_issue_lag": distribution(update_lag),
        "generator_gate": "issue_span_exact_completion",
        "query_queue_wait": distribution(queue_wait),
        "query_service": distribution(service),
        "query_scheduled_to_finished": distribution(query_e2e),
        "update_accepted_to_finished": distribution(update_acc_fin),
        "update_scheduled_to_finished": distribution(update_e2e),
        "delayed_reads": delayed_reads,
        "delayed_updates": delayed_updates,
        "maximum_query_queue_depth": maximum_depth(&mut query_edges),
        "maximum_outstanding_updates": maximum_depth(&mut update_edges),
        "queued_queries_at_stop": queued_queries_at_stop,
        "outstanding_updates_at_stop": outstanding_updates_at_stop,
        "post_acceptance_completion_races": races,
        "rejected_events": failed_events,
        "issuer_cpu_ns": issuer_cpu_ns,
        "pin_event_lanes": args.pin_event_lanes,
        "issuer_by_lane": args.issuer_by_lane,
        "event_lane_cpu_ms": event_lane_cpu_ms,
        "event_lane_apply_ms": event_lane_apply_ms,
        "event_lane_minor_faults": event_lane_minor_faults,
        "event_lane_events": event_lane_events,
        "event_lane_last_finished_ms": event_lane_last_finished_ms,
        "query_lane_cpu_ms_total": query_lane_cpu_ns as f64 / 1e6,
        "issue_span_ns": issue_span_ns,
        "drain_ns": drain_ns,
        "generator_valid": generator_valid,
        "kept_up": kept_up,
        "failure_reasons": failure_reasons,
    });
    println!(
        "{} window {} ms: offered {:.1}M achieved {:.1}M block ops/s, kept_up {}, valid {}, \
         lookup service p50 {:.2} us p99 {:.2} us, scheduled->finished p99 {:.1} us, drain {:.1} ms",
        backend.name(),
        window_ns / 1_000_000,
        result["offered_block_ops_per_sec"].as_f64().unwrap_or(0.0) / 1e6,
        result["achieved_block_ops_per_sec"].as_f64().unwrap_or(0.0) / 1e6,
        kept_up,
        generator_valid,
        result["query_service"]["p50_ns"].as_u64().unwrap_or(0) as f64 / 1e3,
        result["query_service"]["p99_ns"].as_u64().unwrap_or(0) as f64 / 1e3,
        result["query_scheduled_to_finished"]["p99_ns"].as_u64().unwrap_or(0) as f64 / 1e3,
        drain_ns as f64 / 1e6,
    );
    if fan > 1 {
        let shards: Vec<String> = fan_stats
            .iter()
            .enumerate()
            .map(|(shard, s)| {
                format!(
                    "shard {shard}: lookups {} with holders {} holders {} merged {}",
                    s.lookups, s.with_holders, s.holders, s.merged
                )
            })
            .collect();
        println!("lookups per shard (fan-out {fan}): {}", shards.join("; "));
    }
    if !result["failure_reasons"]
        .as_array()
        .is_some_and(Vec::is_empty)
    {
        println!("failure reasons: {}", result["failure_reasons"]);
    }
    std::fs::write(
        &args.result_json_output,
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}

fn maximum_depth(edges: &mut [(u64, i8)]) -> usize {
    edges.sort_unstable_by(|l, r| l.0.cmp(&r.0).then_with(|| r.1.cmp(&l.1)));
    let (mut depth, mut maximum) = (0isize, 0isize);
    for &(_, delta) in edges.iter() {
        depth += delta as isize;
        maximum = maximum.max(depth);
    }
    maximum.max(0) as usize
}

fn distribution(mut values: Vec<u64>) -> serde_json::Value {
    if values.is_empty() {
        return json!({"p50_ns": 0, "p99_ns": 0, "p999_ns": 0, "max_ns": 0});
    }
    values.sort_unstable();
    let rank = |num: usize, den: usize| {
        let r = values.len().saturating_mul(num).div_ceil(den).max(1);
        values[r.saturating_sub(1).min(values.len() - 1)]
    };
    json!({
        "p50_ns": rank(50, 100),
        "p99_ns": rank(99, 100),
        "p999_ns": rank(999, 1000),
        "max_ns": values[values.len() - 1],
    })
}
