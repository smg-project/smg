//! A run-compressed KV index: the event-driven prefix index as a tree of runs with per-run
//! worker coverage bitsets, lock-free and allocation-free for readers, bounded in memory.
//!
//! The index answers the same question as [`PositionalIndexer`](crate::PositionalIndexer): for a
//! request given as its per-block content hashes, how many leading blocks does each worker hold,
//! where "holds" means the worker stored, at every position up to there, the block that sits on
//! the request's chain. It is fed by the same engine events (stored / removed / cleared, keyed by
//! the engine's block hashes) and keeps the same per-worker block map the gateway's event monitor
//! owns, so it drops into the same call sites. Its results are checked against
//! [`ReferenceIndexer`](crate::ReferenceIndexer) by `tests/exactness_chain.rs` (including evictions
//! that leave holes in a chain) and under concurrent lanes by `tests/concurrency_chain.rs`.
//!
//! Shape:
//! - A **run** is a maximal stretch of consecutive positions on one chain whose set of holding
//!   workers is the same at every position. It stores one content hash per position and one
//!   coverage bit per worker. Runs form a tree: a run's children continue it with different next
//!   blocks. A store appends to a run or adds a child; a divergence or a coverage change splits a
//!   run into a prefix and a suffix; positions never move except by a split, which forwards them.
//!   Because coverage is uniform within a run, a worker that evicts a block in the middle of a
//!   chain simply stops covering the piece that holds it, and a lookup stops there for that
//!   worker and nowhere else: holes are exact.
//! - A **lookup** walks from the root, comparing the request's content hashes against each run on
//!   the path (one compare loop per run, not one probe per block) and ANDing the alive set with
//!   the run's coverage; a worker that drops out scores the position where it dropped. Readers
//!   take no locks, allocate nothing but the result map and write to no memory at all: every run
//!   header, hash array and child table lives in an arena addressed by integer ids, a run's
//!   window `(hash array, base, length, children)` is read under a seqlock version that is
//!   checked again after the run's hashes, coverage and child entry have been read, so a split,
//!   a growth, an unlink or a reuse of the run is atomic to a reader.
//! - **Writers** (the event lanes, one per engine worker) lock one run at a time, plus its parent
//!   for the moment it takes to unlink an empty run. A decode extension of the worker's own leaf
//!   appends in place: one lock, no allocation. A split shares the hash array between prefix and
//!   suffix (no copy) and leaves a forwarding record so map entries written before it still
//!   resolve, which keeps other lanes' maps untouched. The lane's own map takes an engine hash to
//!   `(run, offset)`.
//! - **Memory is recycled.** Run headers, hash arrays (reference-counted across the runs a split
//!   leaves sharing one) and child tables return to free lists when they die; a run id carries a
//!   generation so a stale child entry or forwarding record to a reused id is recognised.
//!   Children are an open-addressing table (linear probing, tombstones, rebuilt at 3/4 load), so
//!   a node with many children, the root above all, inserts in constant time.
//!
//! Engine hashes: the index trusts the engine's parent pointers and block identities, as the
//! positional indexer does. Nothing is shared between workers through the maps, so one engine
//! reusing a hash cannot corrupt another worker's view. The index carries one engine hash per
//! distinct block, the one its first holder stored, in an array parallel to the content hashes.
//! The engine hash is a chain hash (a hash of the parent's hash and the block's content), which
//! is what lets a store walk match a run by one engine hash at the end of its window instead of
//! a content hash per block (`match_run`); the content hash at the landing is still checked, and
//! a store that fails it (`landing_mismatches`) is placed by its content, block by block. A
//! worker whose engine names the same content by other hashes (`engine_conflicts` in the stats)
//! is matched by content too; its own hashes key its lane map, so nothing else changes for it.
//! The index therefore assumes one engine hash per worker per (parent, content) position. A
//! second name for a position the worker holds (a twin) is filed onto that position and
//! counted, not stored twice: the worker still scores the position, but once the first name is
//! removed the position goes with it while the second name is still in the lane map, and a
//! store under that name cuts the run at the parent and goes on after it (`store_in_run`). The
//! reference indexer keeps a position until its last name goes, so under twins the index holds
//! at most what the reference holds and never scores a worker above it (the exactness suite's
//! twins test); the normalizer guarantees one name per position for vLLM, whose hash is the
//! chain hash, so the count stays zero in the gateway, and a reading of sliding-window events
//! against the wrong tokens is what produced thousands of them in an offline capture.
//! A lane-map slot carries its key: a probe is settled in the map itself, with no read into the
//! index per block (key-less 8-byte slots checked through the index cost 30-40% of lane CPU).
//!
//! Memory: 16 bytes per distinct block on a chain (its content hash and its engine hash, shared
//! by every worker that holds it, with about 12% slack for growth) plus a 64-byte run header, the
//! coverage words and a child table per branching run, against the 16-byte lane-map slot each
//! lane keeps per held block for removals.

use std::{
    collections::BTreeSet,
    sync::{
        atomic::{fence, AtomicIsize, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};

use crossbeam_queue::SegQueue;
use crossbeam_utils::CachePadded;
use dashmap::{mapref::entry::Entry, DashMap};
use parking_lot::{Mutex, MutexGuard};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::event_tree::{
    chain_prefix_hash, ApplyError, ContentHash, OverlapScores, SequenceHash, StoredBlock,
    WorkerIdExhausted,
};

mod arena;
mod slab;
#[cfg(test)]
mod tests;
mod walk;

use arena::*;
use slab::*;

/// The virtual root: position 0's parent, holds no blocks, never dies.
const ROOT: u32 = 0;
/// "No table" / "no array": word 0 of the arena is never handed out.
const NONE: u32 = 0;
/// Forwarding target of blocks whose last holder evicted them: nowhere.
const GONE: u32 = u32::MAX;
/// A table slot whose child was unlinked.
const TOMB: u64 = u64::MAX;
/// Runs per slab chunk (1024) and chunks in the directory (64 Mi runs in all).
const RUN_CHUNK_BITS: u32 = 10;
const RUN_CHUNK: usize = 1 << RUN_CHUNK_BITS;
const RUN_DIR: usize = 1 << 16;
/// Words per arena chunk (1 Mi, 8 MiB) and chunks in the directory (4 Gi words in all).
const WORD_CHUNK_BITS: u32 = 20;
const WORD_CHUNK: usize = 1 << WORD_CHUNK_BITS;
const WORD_DIR: usize = 1 << 12;
/// Hash array capacities: multiples of 8 up to 128, then powers of two.
const SMALL_ARRAY_CLASSES: usize = 16;
const ARRAY_CLASSES: usize = SMALL_ARRAY_CLASSES + 4 * 13;
/// Classes above the wanted one an allocation may take a freed array from (one octave).
const ARRAY_FIT_SPAN: usize = 4;
/// Child table slot counts: powers of two from 2.
const MIN_TABLE_SLOTS: usize = 2;
const TABLE_CLASSES: usize = 27;
/// Coverage words per run at most: 1024 workers.
const MAX_WORDS: usize = 16;
/// Partial holders a lookup can buffer per run: every worker at most.
const MAX_PARTIAL: usize = MAX_WORDS * 64;
/// Prefix holders a run carries before it is split at their median cutoff: a lookup reads every
/// entry of a run it walks, so a hot chain held to a hundred different depths costs a hundred
/// entries per lookup as one run and a few bitset words as a handful; a split here turns the
/// holders at or past the median into whole holders of the prefix. Sixteen left the A2 pairs
/// unchanged at one and two shards where thirty-two still cost 0.1 us at two. Splits for this reason are
/// bounded by holders, not by requests, so they do not accumulate the way splits at every
/// divergence did.
const PARTIAL_CAP: usize = 16;

thread_local! {
    /// The lookup's buffer for one run's partial-holder entries, per thread: filled and read
    /// within one walk, never cleared, so a lookup does not zero 8 KB of stack first.
    static PARTIAL_BUFFER: std::cell::RefCell<[u64; MAX_PARTIAL]> =
        const { std::cell::RefCell::new([0; MAX_PARTIAL]) };
}

/// One step of [`ChainIndex::hop`] along a block's forwarding chain.
enum Hop {
    /// The place forwards on: the next place and the generation its run must have.
    Next(BlockRef, u32),
    /// The place stands; the run's generation.
    Here(u32),
    /// Nobody holds the block any more.
    Gone,
}

/// Where one of a worker's blocks lives: the run and the offset of the block within it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockRef {
    pub run: u32,
    pub offset: u32,
}

pub use crate::lane_map::ChainBlockMap;

/// Memory and shape counters, as the replay harness reports them and the gateway exports them.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChainIndexStats {
    /// Run headers ever created (resident).
    pub runs_allocated: usize,
    /// Dead headers waiting for reuse.
    pub runs_free: usize,
    /// Runs linked in the tree.
    pub runs_live: usize,
    /// Content hashes held by live runs.
    pub blocks_live: usize,
    /// Bytes of the word arena handed out so far (hash arrays, child tables, free lists
    /// included).
    pub arena_bytes: usize,
    /// Bytes of the word arena sitting in free lists.
    pub arena_free_bytes: usize,
    /// Bytes the word arena holds from the process allocator: `arena_bytes` rounded up to whole
    /// chunks (8 MiB each).
    pub arena_chunk_bytes: usize,
    /// Bytes of run headers, coverage words included (all allocated runs).
    pub header_bytes: usize,
    /// Bytes the run slab holds from the process allocator: whole chunks of headers and coverage
    /// words, used or not.
    pub slab_bytes: usize,
    /// Stored blocks whose engine hash differed from the one the index carries for the block
    /// (an engine fleet that does not agree on hashes); such a block is matched by content and
    /// keyed by the worker's own hash in its lane map.
    pub engine_conflicts: usize,
    /// Stores whose blocks carried the engine hashes the index holds for other content: what
    /// the relay's hash check refuses at the engine, seen from the index. Such a store is
    /// placed by its content, so the index stays exact; the count names a broken engine.
    pub landing_mismatches: usize,
    /// Blocks a store moved to another place for their worker (an engine hash stored again at
    /// a different position, as after a store without its parent); the old membership is
    /// released so a hash is held at one place per worker.
    pub moved_hashes: usize,
    /// Runs split because a stored chain diverged inside them: always zero, since a chain that
    /// diverges inside a run hangs a child off that offset and leaves the run whole; the churn
    /// gate and the split-counter test pin it.
    pub splits_by_branch: usize,
    /// Runs split because a removal left a worker holding blocks on both sides of a hole.
    pub splits_by_hole: usize,
    /// Runs split because a store entered them under a parent the worker did not hold up to
    /// (a stale parent, or a worker re-entering a chain it had lost the head of).
    pub splits_by_mid_run_store: usize,
    /// Runs split at the median cutoff of their prefix holders once more than `PARTIAL_CAP`
    /// held a prefix of them (bounded by holders, never by requests).
    pub splits_by_prefix_holders: usize,
    /// Runs unlinked from the tree since the index was created (their headers go back to the
    /// slab's free list).
    pub runs_died: usize,
    /// Prefix-holder entries over the live runs (a lookup reads a run's entries when a holder it
    /// follows is not a whole holder of the run), and the most one run carries.
    pub partial_entries: usize,
    pub max_partials: usize,
    /// Child-table entries over the live runs, live and tombstoned.
    pub child_entries: usize,
    pub child_tombstones: usize,
}

/// Why a run was split: a hole a worker opens in a run it holds, or the cut at a parent entry
/// that points past what the worker holds.
#[derive(Clone, Copy)]
enum SplitCause {
    Hole,
    MidRunStore,
}

/// Worker slots: a slot is in use from `intern_worker` until `remove_worker`, after which it is
/// handed out again (every coverage bit of a removed worker is clear by then).
#[derive(Default)]
struct WorkerRegistry {
    names: Vec<Option<Arc<str>>>,
    free: Vec<u32>,
}

/// The run-compressed index. Worker ids are interned `u32`s, as in the positional indexer.
pub struct ChainIndex {
    slab: RunSlab,
    arena: WordArena,
    words: usize,
    max_workers: usize,
    worker_to_id: DashMap<Arc<str>, u32, FxBuildHasher>,
    registry: Mutex<WorkerRegistry>,
    /// Blocks held per worker, one cache line each: the lanes write these on every event.
    worker_blocks: Box<[CachePadded<AtomicUsize>]>,
    /// Per-worker signed contributions to the distinct-block count, summed on demand.
    distinct_blocks: Box<[CachePadded<AtomicIsize>]>,
    /// Stored blocks whose engine hash was not the one the index carries.
    engine_conflicts: AtomicUsize,
    /// Stores whose blocks carried the engine hashes the index holds for other content.
    landing_mismatches: AtomicUsize,
    /// Blocks a store moved to another place for their worker (the old membership released).
    /// Coverage words a lookup has to read: enough for the highest worker id interned so far
    /// (ids are handed out densely and reused), at most `words`. An index sized for a thousand
    /// workers that serves eight reads one word per run instead of sixteen.
    live_words: AtomicUsize,
    moved_hashes: AtomicUsize,
    /// Splits by cause and runs unlinked, always counted (one relaxed increment per split or
    /// death): the fragmentation figures a churn harness and the gateway's gauges read.
    splits_branch: AtomicUsize,
    splits_hole: AtomicUsize,
    splits_mid_run: AtomicUsize,
    splits_prefix_holders: AtomicUsize,
    runs_died: AtomicUsize,
}

impl Default for ChainIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// How a worker holds a run.
enum Holding {
    Full,
    /// The first `cutoff` blocks only.
    Partial(usize),
    None,
}

/// Outcome of one attempt to walk a store into the tree.
enum Walk {
    Done,
    /// The walk met a run another lane unlinked meanwhile; start over from the parent block.
    Restart,
    /// The parent block is not held after all (its run died since the map was written).
    NoParent,
}

/// What the lock-free look at a run during a store walk decided.
enum Plan {
    /// Nothing changes here: `matched` blocks are already held; carry on after them. With them,
    /// how many of the matched blocks carry an engine hash that is not the one the index holds.
    Skip(usize, usize),
    /// Continue in the child `(id, generation)` from its first block.
    Descend(u32, u32),
    /// Open a new child for the blocks in hand at this offset of the run (its end, or a
    /// divergence inside it), without the run's lock if the table has room.
    InsertAt(usize),
    /// The run must change: take its lock and re-read it.
    Lock,
}

/// Outcome of claiming a child-table slot.
enum Claim {
    Inserted,
    /// Another writer already linked a child with this head.
    Exists(u32, u32),
    /// No free slot (or no table): grow under the lock.
    Full,
    /// The run changed since the plan was made (a split moved its end): plan again.
    Changed,
}

/// What [`ChainIndex::store_in_run`] found.
enum InRun<'b> {
    /// Every block is placed.
    Done,
    /// The blocks still to place start right after the run's last block.
    Continue(&'b [StoredBlock]),
    /// The blocks still to place diverge from the run at this offset: they continue it from
    /// there as a child.
    ContinueAt(&'b [StoredBlock], usize),
    /// Carry on in this run (id, generation) from its first block.
    MoveTo(u32, u32),
}

/// Blocks a store walk placed: `count` blocks from `start` in the event sit in `run` from
/// `offset`. Recorded under the run lock, written into the lane map after it.
struct Placed {
    run: u32,
    offset: u32,
    start: usize,
    count: usize,
    /// Blocks of the range whose engine hash differs from the one the index carries for the
    /// block (counted in the stats; the worker's own hash keys its lane map either way).
    conflicts: usize,
}

/// How many of the stored blocks carry an engine hash that is not the one the index holds
/// (`engines` runs parallel to `stored`); zero in a fleet whose engines agree.
fn engine_conflicts(stored: &[StoredBlock], engines: &[AtomicU64]) -> usize {
    stored
        .iter()
        .zip(engines)
        .filter(|(block, slot)| block.seq_hash.0 != slot.load(Ordering::Relaxed))
        .count()
}

/// A batch of a worker's offsets in one run, with the generation the run must still have.
struct Removal {
    run: u32,
    generation: Option<u32>,
    offsets: Vec<u32>,
}

impl ChainIndex {
    /// An index for up to 256 workers.
    pub fn new() -> Self {
        Self::with_max_workers(256)
    }

    /// An index for up to `max_workers` interned workers (at most 1024); coverage costs one bit
    /// per worker per run, rounded up to 64.
    pub fn with_max_workers(max_workers: usize) -> Self {
        let max_workers = max_workers.clamp(1, MAX_WORDS * 64);
        let words = max_workers.div_ceil(64);
        let slab = RunSlab::new(words);
        // The root is run 0: position 0's parent, no blocks, no coverage.
        let root = slab.alloc(
            0,
            ROOT,
            Window {
                block: NONE,
                base: 0,
                engine: NONE,
                len: 0,
                children: NONE,
                partials: NONE,
                forwards: NONE,
            },
        );
        debug_assert_eq!(root, ROOT);
        Self {
            slab,
            arena: WordArena::new(),
            words,
            max_workers,
            worker_to_id: DashMap::with_hasher(FxBuildHasher),
            registry: Mutex::new(WorkerRegistry::default()),
            worker_blocks: (0..max_workers)
                .map(|_| CachePadded::new(AtomicUsize::new(0)))
                .collect(),
            distinct_blocks: (0..max_workers)
                .map(|_| CachePadded::new(AtomicIsize::new(0)))
                .collect(),
            engine_conflicts: AtomicUsize::new(0),
            landing_mismatches: AtomicUsize::new(0),
            live_words: AtomicUsize::new(0),
            moved_hashes: AtomicUsize::new(0),
            splits_branch: AtomicUsize::new(0),
            splits_hole: AtomicUsize::new(0),
            splits_mid_run: AtomicUsize::new(0),
            splits_prefix_holders: AtomicUsize::new(0),
            runs_died: AtomicUsize::new(0),
        }
    }

    /// Lock a run's writer state.
    #[inline]
    fn lock_run(&self, run_id: u32) -> MutexGuard<'_, RunMeta> {
        self.slab.run(run_id).meta.lock()
    }

    /// Count a split by its cause: a hole a removal left (`Own`), or a store under a parent the
    /// worker did not hold up to (`Shared`).
    #[inline]
    fn count_split(&self, cause: SplitCause) {
        let counter = match cause {
            SplitCause::Hole => &self.splits_hole,
            SplitCause::MidRunStore => &self.splits_mid_run,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Intern a worker name; the same name maps to the same id until the worker is removed.
    ///
    /// One writer per id: an id's holdings are the lane map of whoever interned it. A second
    /// subscription that interns a name before the first has called
    /// [`remove_worker`](Self::remove_worker) receives the first one's id, and the first one's
    /// removal then takes the second one's blocks out of the runs they share and frees the slot
    /// under it, where a third name may be interned. A caller replacing a worker under the same
    /// name finishes the removal before interning again, or keeps the id and empties it with
    /// [`apply_cleared`](Self::apply_cleared) instead; the index cannot tell two writers apart.
    pub fn intern_worker(&self, worker: &str) -> Result<u32, WorkerIdExhausted> {
        if let Some(entry) = self.worker_to_id.get(worker) {
            return Ok(*entry.value());
        }
        let name: Arc<str> = Arc::from(worker);
        match self.worker_to_id.entry(name.clone()) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                let mut registry = self.registry.lock();
                let id = match registry.free.pop() {
                    Some(id) => id,
                    None if registry.names.len() < self.max_workers => {
                        registry.names.push(None);
                        let id = (registry.names.len() - 1) as u32;
                        // Published before the id can hold anything: the worker's first store
                        // comes after this returns.
                        self.live_words
                            .fetch_max(id as usize / 64 + 1, Ordering::Release);
                        id
                    }
                    None => return Err(WorkerIdExhausted),
                };
                registry.names[id as usize] = Some(name);
                entry.insert(id);
                Ok(id)
            }
        }
    }

    /// Hand a removed worker's slot back once nothing refers to it any more.
    fn release_worker(&self, worker: u32) {
        // Lock order is name map shard, then registry (as in `intern_worker`): never hold the
        // registry while touching the name map, or an intern and a release deadlock each other.
        let name = {
            let mut registry = self.registry.lock();
            registry
                .names
                .get_mut(worker as usize)
                .and_then(Option::take)
        };
        let Some(name) = name else {
            return;
        };
        self.worker_to_id.remove(&*name);
        self.registry.lock().free.push(worker);
    }

    pub fn worker_id(&self, worker: &str) -> Option<u32> {
        self.worker_to_id.get(worker).map(|entry| *entry.value())
    }

    /// Whether no worker holds any block: the root has no live child, read under the root's
    /// version. O(1), for a caller that asks before every lookup (`current_size` reads a line
    /// per worker slot).
    pub fn is_empty(&self) -> bool {
        let root = self.slab.run(ROOT);
        loop {
            let (window, version) = root.snapshot();
            let empty = window.children == NONE || self.arena.table_live(window.children) == 0;
            if root.confirm(version) {
                return empty;
            }
        }
    }

    /// Blocks held across all workers (a block two workers hold counts twice).
    pub fn current_size(&self) -> usize {
        self.worker_blocks
            .iter()
            .map(|count| count.load(Ordering::Relaxed))
            .sum()
    }

    /// Distinct blocks held by at least one worker.
    pub fn entry_count(&self) -> usize {
        let total: isize = self
            .distinct_blocks
            .iter()
            .map(|count| count.load(Ordering::Relaxed))
            .sum();
        total.max(0) as usize
    }

    pub fn worker_block_count(&self, worker: u32) -> usize {
        self.worker_blocks
            .get(worker as usize)
            .map_or(0, |count| count.load(Ordering::Relaxed))
    }

    fn credit(&self, worker: u32, blocks: usize) {
        if blocks == 0 {
            return;
        }
        self.worker_blocks[worker as usize].fetch_add(blocks, Ordering::Relaxed);
    }

    fn debit(&self, worker: u32, blocks: usize) {
        if blocks == 0 {
            return;
        }
        self.worker_blocks[worker as usize].fetch_sub(blocks, Ordering::Relaxed);
    }

    /// `worker` became the first holder of `blocks` distinct blocks.
    fn distinct_add(&self, worker: u32, blocks: usize) {
        if blocks != 0 {
            self.distinct_blocks[worker as usize].fetch_add(blocks as isize, Ordering::Relaxed);
        }
    }

    /// `worker` was the last holder of `blocks` distinct blocks.
    fn distinct_sub(&self, worker: u32, blocks: usize) {
        if blocks != 0 {
            self.distinct_blocks[worker as usize].fetch_sub(blocks as isize, Ordering::Relaxed);
        }
    }

    /// The hash at `offset` of a run, from the writer's side (the run is locked).
    #[inline]
    fn hash_at(&self, run: &Run, offset: usize) -> u64 {
        let data = run.block.load(Ordering::Relaxed) + run.base.load(Ordering::Relaxed);
        self.arena
            .word(data + offset as u32)
            .load(Ordering::Relaxed)
    }

    /// The child-table key of a child continuing a run from `offset` with the content hash
    /// `head`: a child may hang off any offset of its parent (a divergence inside a run does not
    /// split the run), so the key carries the offset beside the hash. Never zero, which a table
    /// slot reads as empty.
    #[inline]
    fn child_key(offset: usize, head: u64) -> u64 {
        let mixed = head
            ^ (offset as u64 + 1)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .rotate_left(23);
        if mixed == 0 {
            1
        } else {
            mixed
        }
    }

    /// The content hash of a run's first block, read under its version (the run is not locked).
    fn child_head(&self, run_id: u32) -> u64 {
        let run = self.slab.run(run_id);
        loop {
            let (window, version) = run.snapshot();
            let head = self
                .arena
                .word(window.block + window.base)
                .load(Ordering::Relaxed);
            if run.confirm(version) {
                return head;
            }
        }
    }

    /// One step along a block's forwarding chain, read under the run's version: the next place
    /// and the generation it must have, the place standing (with the run's generation), or the
    /// block gone (a dead or reused run, a `GONE` suffix). A split publishes its shorter length
    /// and its forwarding record under one lock but in two steps, so a reader that meets an
    /// offset at or past the length without a record waits for the lock once and reads again;
    /// after that the record is there, or the reference was stale.
    fn hop(&self, at: BlockRef, expected: Option<u32>) -> Hop {
        let mut waited = false;
        loop {
            if at.run == GONE {
                return Hop::Gone;
            }
            let run = self.slab.run(at.run);
            let (window, version) = run.snapshot();
            let generation = (version >> 32) as u32;
            if expected.is_some_and(|wanted| wanted != generation)
                || (at.run != ROOT && window.len == 0)
            {
                return Hop::Gone;
            }
            let hop = self.arena.forwards_find(window.forwards, at.offset);
            if !run.confirm(version) {
                continue;
            }
            match hop {
                Some((next, next_generation)) => return Hop::Next(next, next_generation),
                None if at.run != ROOT && at.offset >= window.len => {
                    if waited {
                        return Hop::Gone;
                    }
                    drop(run.meta.lock());
                    waited = true;
                }
                None => return Hop::Here(generation),
            }
        }
    }

    /// Where a block recorded at `at` lives now, with the generation of the run: the end of its
    /// forwarding chain. `None` when nobody holds it any more.
    fn resolve(&self, mut at: BlockRef) -> Option<(BlockRef, u32)> {
        let mut expected: Option<u32> = None;
        loop {
            match self.hop(at, expected) {
                Hop::Next(next, generation) => {
                    at = next;
                    expected = Some(generation);
                }
                Hop::Here(generation) => return Some((at, generation)),
                Hop::Gone => return None,
            }
        }
    }

    /// Whether `to` is `from` or a place `from` has forwarded to since: the places one block has
    /// had. Forwarding records only accumulate while their runs live (a prefix outlives its
    /// suffix, being its parent), so the answer does not depend on when it is read, unlike the
    /// ends of two chains resolved one after the other.
    fn forwards_to(&self, from: BlockRef, to: BlockRef) -> bool {
        let mut at = from;
        let mut expected: Option<u32> = None;
        loop {
            if at == to {
                return true;
            }
            match self.hop(at, expected) {
                Hop::Next(next, generation) => {
                    at = next;
                    expected = Some(generation);
                }
                Hop::Here(_) | Hop::Gone => return false,
            }
        }
    }

    /// Whether `worker`'s lane map holds the block with engine hash `key`.
    pub fn is_held(&self, map: &ChainBlockMap, key: SequenceHash) -> bool {
        let _ = self;
        map.contains_key(key)
    }

    /// Add `child` to the run's table (the run is locked): claims a slot like a lock-free
    /// inserter, growing the table under a version step that first drains inserters in flight.
    /// `Some` when another writer linked a child with this head meanwhile: the caller descends
    /// into that one instead.
    fn link_child(&self, run: &Run, offset: usize, head: u64, child: u32) -> Option<(u32, u32)> {
        let generation = self.slab.run(child).generation();
        let key = Self::child_key(offset, head);
        loop {
            let table = run.children.load(Ordering::Acquire);
            match self.arena.table_claim(table, key, child, generation) {
                Claim::Inserted => return None,
                Claim::Exists(other, other_generation) => return Some((other, other_generation)),
                Claim::Full | Claim::Changed => {}
            }
            run.begin_update();
            run.wait_inflight();
            let table = run.children.load(Ordering::Relaxed);
            let grown = if table == NONE {
                self.arena.alloc_table(MIN_TABLE_SLOTS)
            } else {
                self.arena.table_grown(table)
            };
            run.children.store(grown, Ordering::Relaxed);
            run.end_update();
            self.arena.free_table(table);
        }
    }

    /// Take `child` out of the run's table (the run is locked); an emptied table goes away once
    /// no insert is in flight on it.
    fn unlink_child(&self, run: &Run, child: u32) {
        let table = run.children.load(Ordering::Relaxed);
        if table == NONE {
            return;
        }
        let live = self.arena.table_take(table, child);
        if live == 0 {
            run.begin_update();
            run.wait_inflight();
            if self.arena.word(table + 1).load(Ordering::Relaxed) == 0 {
                run.children.store(NONE, Ordering::Relaxed);
                run.end_update();
                self.arena.free_table(table);
            } else {
                run.end_update();
            }
        } else if self.arena.table_dead(table) > self.arena.table_slots(table) / 2 {
            // Children come and go at every offset of a long-lived run (a decode tail per prompt
            // end): a table more than half tombstones is rebuilt from its live entries, so a
            // probe never walks the dead keys of a thousand finished requests.
            run.begin_update();
            run.wait_inflight();
            let fresh = self.arena.table_from(&self.arena.table_entries(table));
            run.children.store(fresh, Ordering::Relaxed);
            run.end_update();
            self.arena.free_table(table);
        }
    }

    /// A lock-free attempt to link `child` (prepared, unpublished) under `run` for the blocks
    /// after its last one, as seen in the snapshot with `planned` version: holds `inflight`
    /// across the claim so a split or table growth cannot move the table under the insert, and
    /// gives up with `Claim::Changed` if the run moved on since the plan (its end is elsewhere
    /// now). `Claim::Full` means the locked path must do it.
    fn insert_child(
        &self,
        run_id: u32,
        child: u32,
        offset: usize,
        head: u64,
        planned: u64,
    ) -> Claim {
        let run = self.slab.run(run_id);
        loop {
            run.inflight.fetch_add(1, Ordering::SeqCst);
            let version = run.version.load(Ordering::SeqCst);
            if version & 1 == 1 {
                run.inflight.fetch_sub(1, Ordering::SeqCst);
                std::hint::spin_loop();
                continue;
            }
            if version != planned {
                run.inflight.fetch_sub(1, Ordering::SeqCst);
                return Claim::Changed;
            }
            let table = run.children.load(Ordering::Acquire);
            let len = run.len();
            if run.version.load(Ordering::SeqCst) != version {
                run.inflight.fetch_sub(1, Ordering::SeqCst);
                continue;
            }
            if offset > len {
                run.inflight.fetch_sub(1, Ordering::SeqCst);
                return Claim::Changed;
            }
            // The child continues this run from `offset`: its end, or a divergence inside it.
            let new_run = self.slab.run(child);
            new_run
                .start
                .store((run.start() + offset) as u32, Ordering::Relaxed);
            new_run.parent.store(run_id, Ordering::Release);
            let claim = self.arena.table_claim(
                table,
                Self::child_key(offset, head),
                child,
                new_run.generation(),
            );
            run.inflight.fetch_sub(1, Ordering::SeqCst);
            return claim;
        }
    }

    /// A run prepared for a store that another lane beat to the slot: back to the slab.
    fn discard_run(&self, run_id: u32, worker: u32, freed: &mut Vec<u32>) {
        let mut meta = self.slab.run(run_id).meta.lock();
        clear(self.slab.coverage(run_id), worker);
        self.kill(run_id, &mut meta, freed);
    }

    /// Retire a run that is unlinked (locked by the caller): its array reference goes, its
    /// window empties, and its id is queued for reuse once the caller has dropped the lock.
    fn kill(&self, run_id: u32, meta: &mut RunMeta, freed: &mut Vec<u32>) {
        let run = self.slab.run(run_id);
        let block = run.block.load(Ordering::Relaxed);
        let engine = run.engine.load(Ordering::Relaxed);
        let partials = run.partials.load(Ordering::Relaxed);
        let forwards = run.forwards.load(Ordering::Relaxed);
        run.begin_update();
        run.block.store(NONE, Ordering::Relaxed);
        run.base.store(0, Ordering::Relaxed);
        run.engine.store(NONE, Ordering::Relaxed);
        run.len.store(0, Ordering::Relaxed);
        run.children.store(NONE, Ordering::Relaxed);
        run.partials.store(NONE, Ordering::Relaxed);
        run.forwards.store(NONE, Ordering::Relaxed);
        run.end_update();
        self.arena.array_release(block);
        self.arena.array_release(engine);
        self.arena.free_partials(partials);
        self.arena.free_table(forwards);
        meta.dead = true;
        self.runs_died.fetch_add(1, Ordering::Relaxed);
        freed.push(run_id);
    }

    /// Dead ids go back to the slab only after their locks are released, so a thread holding a
    /// live run's lock and reviving a dead id never waits on a thread that holds the dead id's
    /// lock and wants the live run.
    fn recycle(&self, freed: &mut Vec<u32>) {
        for id in freed.drain(..) {
            self.slab.free.push(id);
        }
    }

    /// How `worker` holds a run: all of it, a prefix of it, or nothing.
    fn holding(&self, run_id: u32, worker: u32) -> Holding {
        if has(self.slab.coverage(run_id), worker) {
            return Holding::Full;
        }
        let table = self.slab.run(run_id).partials.load(Ordering::Relaxed);
        match self.arena.partial_find(table, worker) {
            Some((_, cutoff)) => Holding::Partial(cutoff as usize),
            None => Holding::None,
        }
    }

    /// Blocks of a run that `worker` holds.
    fn held_by(&self, run_id: u32, worker: u32) -> usize {
        match self.holding(run_id, worker) {
            Holding::Full => self.slab.run(run_id).len(),
            Holding::Partial(cutoff) => cutoff,
            Holding::None => 0,
        }
    }

    /// Blocks of a run held by at least one worker: all of them while anybody holds the whole
    /// run, otherwise the longest partial prefix.
    fn held_len(&self, run_id: u32) -> usize {
        let run = self.slab.run(run_id);
        if coverage_is_empty(self.slab.coverage(run_id)) {
            self.arena.partial_max(run.partials.load(Ordering::Relaxed))
        } else {
            run.len()
        }
    }

    fn has_holders(&self, run_id: u32) -> bool {
        !coverage_is_empty(self.slab.coverage(run_id))
            || self
                .arena
                .partials_live(self.slab.run(run_id).partials.load(Ordering::Relaxed))
                > 0
    }

    /// Distinct-block accounting around a change to `run_id` (locked by the caller): the delta
    /// of blocks held by anybody, attributed to `worker`. A split changes nothing in total (the
    /// prefix and the suffix hold between them what the run held), so callers take `before`
    /// after any split; the suffix is published by then and other lanes account for their own
    /// changes to it.
    fn settle_distinct(&self, worker: u32, before: usize, run_id: u32) {
        let after = self.held_len(run_id);
        if after > before {
            self.distinct_add(worker, after - before);
        } else {
            self.distinct_sub(worker, before - after);
        }
    }

    /// Make `worker` hold exactly `[0, cutoff)` of the run (the run is locked): the whole run when
    /// `cutoff` reaches its length, nothing when 0. Readers treat the coverage bit as the truth
    /// when both forms are visible, so a worker gains its bit before its partial entry goes and
    /// gains a partial entry before its bit goes.
    fn set_holding(&self, run_id: u32, worker: u32, cutoff: usize) {
        let run = self.slab.run(run_id);
        let coverage = self.slab.coverage(run_id);
        let len = run.len();
        let was_full = has(coverage, worker);
        let table = run.partials.load(Ordering::Relaxed);
        let entry = if was_full {
            None
        } else {
            self.arena.partial_find(table, worker)
        };
        if cutoff >= len {
            if !was_full {
                set(coverage, worker);
            }
            if let Some((slot, _)) = entry {
                self.drop_partial(run, table, slot);
            }
        } else if cutoff == 0 {
            if was_full {
                clear(coverage, worker);
            }
            if let Some((slot, _)) = entry {
                self.drop_partial(run, table, slot);
            }
        } else {
            match entry {
                Some((slot, old)) => {
                    if old as usize != cutoff {
                        self.arena.partial_set(table, slot, worker, cutoff as u32);
                    }
                }
                None if was_full => {
                    // A reader that saw neither the entry nor the bit would score nothing for a
                    // worker that holds a prefix: keep the two writes inside one version step.
                    run.begin_update();
                    self.add_partial(run, table, worker, cutoff as u32);
                    clear(coverage, worker);
                    run.end_update();
                }
                None => self.add_partial(run, table, worker, cutoff as u32),
            }
        }
    }

    /// Append a partial entry, growing (and republishing) the table when it is full.
    fn add_partial(&self, run: &Run, table: u32, worker: u32, cutoff: u32) {
        if self.arena.partial_put(table, worker, cutoff) {
            return;
        }
        let grown = if table == NONE {
            self.arena.alloc_partials(MIN_TABLE_SLOTS)
        } else {
            self.arena.partials_grown(table, 1)
        };
        let placed = self.arena.partial_put(grown, worker, cutoff);
        debug_assert!(placed);
        let nested = run.version.load(Ordering::Relaxed) & 1 == 1;
        if !nested {
            run.begin_update();
        }
        run.partials.store(grown, Ordering::Relaxed);
        if !nested {
            run.end_update();
        }
        self.arena.free_partials(table);
    }

    /// Tombstone a partial entry; an emptied table goes away.
    fn drop_partial(&self, run: &Run, table: u32, slot: usize) {
        if self.arena.partial_remove(table, slot) == 0 {
            run.begin_update();
            run.partials.store(NONE, Ordering::Relaxed);
            run.end_update();
            self.arena.free_partials(table);
        }
    }

    /// Split `run` (locked by the caller, whose guard is `_meta`) at `at`: the run keeps `[0, at)`; a new suffix run takes
    /// `[at, len)` on the same hash array, with the run's children, its full holders and the
    /// partial holders reaching past `at`; partial holders reaching `at` become full holders of
    /// the prefix. A suffix nobody would hold is not created when the run has no children: the
    /// forwarding record says those blocks are gone. No worker's holdings change in total.
    /// Split the run (locked by the caller) at the median cutoff of its prefix holders when more
    /// than `PARTIAL_CAP` of them hold a prefix of it: the holders at or past the median become
    /// whole holders of the prefix, the rest keep their entries on one side or the other.
    fn cap_prefix_holders(&self, run_id: u32, meta: &mut RunMeta) {
        let run = self.slab.run(run_id);
        let table = run.partials.load(Ordering::Relaxed);
        if table == NONE || self.arena.partials_live(table) <= PARTIAL_CAP {
            return;
        }
        let mut cutoffs: Vec<u32> = self
            .arena
            .partial_entries(table)
            .iter()
            .map(|&(_, cutoff)| cutoff)
            .collect();
        if cutoffs.len() <= PARTIAL_CAP {
            return;
        }
        cutoffs.sort_unstable();
        let at = cutoffs[cutoffs.len() / 2] as usize;
        if at == 0 || at >= run.len() {
            return;
        }
        self.splits_prefix_holders.fetch_add(1, Ordering::Relaxed);
        self.split_locked(run_id, meta, at);
    }

    fn split_locked(&self, run_id: u32, _meta: &mut RunMeta, at: usize) -> u32 {
        let run = self.slab.run(run_id);
        let coverage = self.slab.coverage(run_id);
        let len = run.len();
        debug_assert!(at > 0 && at < len, "split inside the run: 0 < {at} < {len}");
        let block = run.block.load(Ordering::Relaxed);
        let engine = run.engine.load(Ordering::Relaxed);
        let base = run.base.load(Ordering::Relaxed);
        let children = run.children.load(Ordering::Relaxed);
        let partials = self
            .arena
            .partial_entries(run.partials.load(Ordering::Relaxed));
        let beyond: Vec<(u32, u32)> = partials
            .iter()
            .filter(|(_, cutoff)| *cutoff as usize > at)
            .map(|&(worker, cutoff)| (worker, cutoff - at as u32))
            .collect();
        let suffix_held = !coverage_is_empty(coverage) || !beyond.is_empty();
        // Children hang off any offset of the run: those past `at` move to the suffix, keyed by
        // their offset within it; the rest stay. Decided under the version step, after
        // the inserts in flight have landed, so none is missed.
        run.begin_update();
        run.wait_inflight();
        let start = run.start();
        let mut kept: Vec<(u64, u32, u32)> = Vec::new();
        let mut moved: Vec<(u64, u32, u32)> = Vec::new();
        for (key, child, generation) in self.arena.table_entries(children) {
            let child_offset = self.slab.run(child).start().saturating_sub(start);
            // A child at `at` itself stays an end child of the prefix (a walk never descends
            // from offset zero of a run, so the suffix must not carry one there).
            if child_offset <= at {
                kept.push((key, child, generation));
            } else {
                moved.push((
                    Self::child_key(child_offset - at, self.child_head(child)),
                    child,
                    generation,
                ));
            }
        }
        let suffix_id = if !suffix_held && moved.is_empty() {
            run.len.store(at as u32, Ordering::Relaxed);
            run.end_update();
            GONE
        } else {
            self.arena.array_retain(block);
            self.arena.array_retain(engine);
            let suffix_partials = self.arena.partials_from(&beyond);
            let suffix_children = self.arena.table_from(&moved);
            let suffix_id = self.slab.alloc(
                start + at,
                run_id,
                Window {
                    block,
                    base: base + at as u32,
                    engine,
                    len: (len - at) as u32,
                    children: suffix_children,
                    partials: suffix_partials,
                    forwards: NONE,
                },
            );
            let suffix = self.slab.run(suffix_id);
            for (slot, word) in self.slab.coverage(suffix_id).iter().zip(coverage) {
                slot.store(word.load(Ordering::Relaxed), Ordering::Relaxed);
            }
            for &(_, child, _) in &moved {
                self.slab
                    .run(child)
                    .parent
                    .store(suffix_id, Ordering::Release);
            }
            kept.push((
                Self::child_key(at, self.hash_at(run, at)),
                suffix_id,
                suffix.generation(),
            ));
            let table = self.arena.table_from(&kept);
            run.len.store(at as u32, Ordering::Relaxed);
            run.children.store(table, Ordering::Relaxed);
            run.end_update();
            self.arena.free_table(children);
            suffix_id
        };
        // The prefix is `[0, at)` now: a partial holder that reached it holds all of it. Done
        // after the truncation so no reader sees a bit for the whole old run.
        for (worker, cutoff) in partials {
            if cutoff as usize >= at {
                self.set_holding(run_id, worker, at);
            }
        }
        let generation = if suffix_id == GONE {
            0
        } else {
            self.slab.run(suffix_id).generation()
        };
        self.add_forward(run, at as u32, suffix_id, generation);
        suffix_id
    }

    /// Record a split on the run (locked by the caller); a full table is replaced under a version
    /// step so a lock-free reader never follows a recycled one.
    fn add_forward(&self, run: &Run, at: u32, suffix: u32, generation: u32) {
        let table = run.forwards.load(Ordering::Relaxed);
        if self.arena.forwards_push(table, at, suffix, generation) {
            return;
        }
        let grown = if table == NONE {
            self.arena.alloc_forwards(MIN_TABLE_SLOTS)
        } else {
            self.arena.forwards_grown(table)
        };
        let placed = self.arena.forwards_push(grown, at, suffix, generation);
        debug_assert!(placed);
        run.begin_update();
        run.forwards.store(grown, Ordering::Relaxed);
        run.end_update();
        self.arena.free_table(table);
    }

    /// Unlink `run` (locked by the caller, known to be an uncovered leaf) from its parent and
    /// retire it, then the parent if that leaves it an uncovered leaf too.
    fn unlink_locked(&self, run_id: u32, meta: &mut RunMeta, freed: &mut Vec<u32>) {
        if run_id == ROOT || meta.dead {
            return;
        }
        let run = self.slab.run(run_id);
        loop {
            let parent_id = run.parent.load(Ordering::Acquire);
            let parent = self.slab.run(parent_id);
            // Child-then-parent is the only order in which two run locks are ever held. A split
            // of the parent may have re-parented this run while we waited; check and retry.
            let mut parent_meta = parent.meta.lock();
            if run.parent.load(Ordering::Acquire) != parent_id {
                continue;
            }
            self.unlink_child(parent, run_id);
            self.kill(run_id, meta, freed);
            if parent_id != ROOT
                && parent.children.load(Ordering::Relaxed) == NONE
                && !self.has_holders(parent_id)
            {
                self.unlink_locked(parent_id, &mut parent_meta, freed);
            }
            return;
        }
    }

    /// Store `blocks` for `worker` after `parent` (position 0 when `None`).
    pub fn apply_stored(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
        map: &mut ChainBlockMap,
    ) -> Result<(), ApplyError> {
        if blocks.is_empty() {
            return Ok(());
        }
        let origin = match parent {
            None => None,
            Some(hash) => {
                if map.is_empty() {
                    return Err(ApplyError::WorkerNotTracked);
                }
                match map.get(hash) {
                    Some(at) => Some((hash, at)),
                    None => return Err(ApplyError::ParentBlockNotFound),
                }
            }
        };
        loop {
            match self.store_walk(worker, blocks, origin, map) {
                Walk::Done => return Ok(()),
                Walk::Restart => {}
                Walk::NoParent => return Err(ApplyError::ParentBlockNotFound),
            }
        }
    }

    /// One attempt to place a store, from the parent block (or the root) down the tree. Map
    /// entries for the placed blocks are written after the run locks are released: the lane map
    /// is private, and a split meanwhile is covered by the forwarding records.
    fn store_walk(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        origin: Option<(SequenceHash, BlockRef)>,
        map: &mut ChainBlockMap,
    ) -> Walk {
        let mut pending: Vec<Placed> = Vec::new();
        let outcome = self.store_walk_locked(worker, blocks, origin, map, &mut pending);
        self.write_placements(worker, blocks, pending, map);
        outcome
    }

    /// The lane map writes of one store, after the run locks are released. A block the engine
    /// names by a hash this worker already holds elsewhere (a store that moves the hash, as a
    /// store without its parent followed by the whole chain does) keeps one place per hash: the
    /// old membership is released. A re-store of the same block at the same place is not a
    /// move, and "the same place" is read through the forwarding records: the placement was
    /// recorded under the run's lock, and splits by other lanes land freely between that and
    /// this write, so the recorded place and the map's old entry may both be behind the block's
    /// current place by any number of splits. A block that did not move has its new place on
    /// the forwarding chain from its old one, and that stays true however many splits follow
    /// (`forwards_to`); comparing the two places resolved to their ends instead raced with a
    /// split between the two reads, took the block for a moved hash, and released the worker's
    /// holding at its real place while the map kept the entry (lookups then scored the worker
    /// short, and a store under the block as parent found no parent).
    fn write_placements(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        pending: Vec<Placed>,
        map: &mut ChainBlockMap,
    ) {
        let mut moved: Vec<BlockRef> = Vec::new();
        for placed in pending {
            let first = BlockRef {
                run: placed.run,
                offset: placed.offset,
            };
            let range = &blocks[placed.start..placed.start + placed.count];
            if placed.conflicts > 0 {
                self.engine_conflicts
                    .fetch_add(placed.conflicts, Ordering::Relaxed);
            }
            map.insert_run(
                range.iter().map(|stored| stored.seq_hash),
                first,
                |old, new| {
                    if !self.forwards_to(old, new) {
                        moved.push(old);
                    }
                },
            );
        }
        if !moved.is_empty() {
            self.moved_hashes.fetch_add(moved.len(), Ordering::Relaxed);
            let work = group_by_run(moved);
            self.apply_removals(worker, work);
        }
    }

    fn store_walk_locked(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        origin: Option<(SequenceHash, BlockRef)>,
        map: &mut ChainBlockMap,
        pending: &mut Vec<Placed>,
    ) -> Walk {
        let (mut run_id, mut offset, mut expected) = match origin {
            None => (ROOT, 0usize, self.slab.run(ROOT).generation()),
            Some((hash, at)) => {
                let Some((at, generation)) = self.resolve(at) else {
                    map.remove(hash);
                    return Walk::NoParent;
                };
                map.insert(hash, at);
                (at.run, at.offset as usize + 1, generation)
            }
        };
        let mut remaining = blocks;
        // Set when a lock-free insert found the child table full: the next look at the same run
        // takes its lock, whose path grows the table.
        let mut force_lock = false;
        loop {
            // Look at the run under its version first: a run this worker already holds up to
            // the blocks in hand, or a run whose child continues them, is passed without its
            // lock. Only a run that must change is locked, and re-read under the lock.
            let run = self.slab.run(run_id);
            let (window, version) = run.snapshot();
            if (version >> 32) as u32 != expected || (run_id != ROOT && window.len == 0) {
                return Walk::Restart;
            }
            let len = window.len as usize;
            if offset > len {
                // A split moved the blocks after the parent into a suffix: resolve again.
                return Walk::Restart;
            }
            let block_start = blocks.len() - remaining.len();
            let plan = if force_lock {
                force_lock = false;
                Plan::Lock
            } else if offset < len {
                let held = self.held_in(run_id, &window, worker);
                if held < offset {
                    Plan::Lock
                } else {
                    let data = window.block + window.base + offset as u32;
                    let hashes = self.arena.words(data, len - offset);
                    let engines = self
                        .arena
                        .words(window.engine + window.base + offset as u32, len - offset);
                    let (matched, conflicts) = self.match_run(hashes, engines, remaining, false);
                    if matched > 0 && offset + matched <= held {
                        // Blocks already held: step past them. A divergence after them is the
                        // next round's business, at the offset where it starts.
                        Plan::Skip(matched, conflicts)
                    } else if matched == 0 && offset > 0 {
                        // The blocks in hand leave the run's content right here: they continue
                        // it as a child hanging off this offset, found or opened without the
                        // lock. The run itself does not change.
                        self.plan_child(&window, offset, remaining[0].content_hash.0)
                    } else {
                        Plan::Lock
                    }
                }
            } else {
                self.plan_child(&window, len, remaining[0].content_hash.0)
            };
            if !run.confirm(version) {
                continue;
            }
            match plan {
                Plan::InsertAt(at) => {
                    let head = remaining[0].content_hash.0;
                    let contents: Vec<u64> = remaining
                        .iter()
                        .map(|stored| stored.content_hash.0)
                        .collect();
                    let engines: Vec<u64> =
                        remaining.iter().map(|stored| stored.seq_hash.0).collect();
                    let block = self
                        .arena
                        .alloc_array(&contents, capacity_for(contents.len()));
                    let engine = self
                        .arena
                        .alloc_array(&engines, self.arena.array_capacity(block));
                    let new_id = self.slab.alloc(
                        0,
                        run_id,
                        Window {
                            block,
                            base: 0,
                            engine,
                            len: contents.len() as u32,
                            children: NONE,
                            partials: NONE,
                            forwards: NONE,
                        },
                    );
                    set(self.slab.coverage(new_id), worker);
                    match self.insert_child(run_id, new_id, at, head, version) {
                        Claim::Inserted => {
                            pending.push(Placed {
                                run: new_id,
                                offset: 0,
                                start: block_start,
                                count: remaining.len(),
                                conflicts: 0,
                            });
                            self.credit(worker, contents.len());
                            self.distinct_add(worker, contents.len());
                            return Walk::Done;
                        }
                        Claim::Exists(child, generation) => {
                            let mut freed = Vec::new();
                            self.discard_run(new_id, worker, &mut freed);
                            self.recycle(&mut freed);
                            run_id = child;
                            expected = generation;
                            offset = 0;
                        }
                        Claim::Full => {
                            let mut freed = Vec::new();
                            self.discard_run(new_id, worker, &mut freed);
                            self.recycle(&mut freed);
                            force_lock = true;
                        }
                        Claim::Changed => {
                            let mut freed = Vec::new();
                            self.discard_run(new_id, worker, &mut freed);
                            self.recycle(&mut freed);
                            return Walk::Restart;
                        }
                    }
                }
                Plan::Skip(matched, conflicts) => {
                    pending.push(Placed {
                        run: run_id,
                        offset: offset as u32,
                        start: block_start,
                        count: matched,
                        conflicts,
                    });
                    if matched == remaining.len() {
                        return Walk::Done;
                    }
                    remaining = &remaining[matched..];
                    offset += matched;
                }
                Plan::Descend(child, generation) => {
                    run_id = child;
                    expected = generation;
                    offset = 0;
                }
                Plan::Lock => {
                    let mut meta = self.lock_run(run_id);
                    if meta.dead || run.generation() != expected || offset > run.len() {
                        return Walk::Restart;
                    }
                    let next = match self.store_in_run(
                        worker,
                        run_id,
                        &mut meta,
                        offset,
                        remaining,
                        block_start,
                        pending,
                    ) {
                        InRun::Done => return Walk::Done,
                        InRun::Continue(rest) => {
                            remaining = rest;
                            let end = run.len();
                            self.store_at(
                                worker,
                                run_id,
                                &meta,
                                end,
                                remaining,
                                blocks.len() - remaining.len(),
                                pending,
                            )
                        }
                        InRun::ContinueAt(rest, at) => {
                            remaining = rest;
                            self.store_at(
                                worker,
                                run_id,
                                &meta,
                                at,
                                remaining,
                                blocks.len() - remaining.len(),
                                pending,
                            )
                        }
                        InRun::MoveTo(child, generation) => Some((child, generation)),
                    };
                    drop(meta);
                    match next {
                        Some((child, generation)) => {
                            run_id = child;
                            expected = generation;
                            offset = 0;
                        }
                        None => return Walk::Done,
                    }
                }
            }
        }
    }

    /// How far `remaining` continues a run whose hashes from the position in hand are `contents`
    /// and `engines`: the matched count and how many of the matched blocks carry an engine hash
    /// that is not the one the index holds.
    ///
    /// An engine hash is a chain hash: an engine names a block by a hash of its parent's hash and
    /// the block's own content, so two chains that agree at a position agree at every position
    /// before it. The match is therefore decided by the engine hash at the end of the window,
    /// and on a mismatch by bisection to the first differing position, instead of a content
    /// compare per block. Two checks keep it exact for a worker whose engine breaks the
    /// assumption, both by falling back to the compare per block: the content hash at the
    /// landing (a block named by the hash the index carries but holding other content, which
    /// is what the relay's hash check exists to catch; `landing_mismatches` counts it), and the
    /// content hash right after the match (an engine that hashes the same content differently
    /// matches by content where it cannot by engine hash, and is counted).
    ///
    /// `count` says whether a landing mismatch is counted: the lock-free look at a run finds it
    /// first and then takes the lock, where the locked look finds it again.
    fn match_run(
        &self,
        contents: &[AtomicU64],
        engines: &[AtomicU64],
        remaining: &[StoredBlock],
        count: bool,
    ) -> (usize, usize) {
        let window = remaining.len().min(contents.len()).min(engines.len());
        if window == 0 {
            return (0, 0);
        }
        let engine_eq = |at: usize| remaining[at].seq_hash.0 == engines[at].load(Ordering::Relaxed);
        let content_eq =
            |at: usize| remaining[at].content_hash.0 == contents[at].load(Ordering::Relaxed);
        let matched = if engine_eq(window - 1) {
            window
        } else {
            // The positions that agree are a prefix: find the first that does not.
            let (mut low, mut high) = (0usize, window - 1);
            while low < high {
                let mid = low + (high - low) / 2;
                if engine_eq(mid) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low
        };
        let landing_holds = matched == 0 || content_eq(matched - 1);
        let same_content_after = matched < window && content_eq(matched);
        if landing_holds && !same_content_after {
            return (matched, 0);
        }
        if !landing_holds && count {
            self.landing_mismatches.fetch_add(1, Ordering::Relaxed);
        }
        let matched = remaining
            .iter()
            .zip(contents)
            .take_while(|(stored, slot)| stored.content_hash.0 == slot.load(Ordering::Relaxed))
            .count();
        (
            matched,
            engine_conflicts(&remaining[..matched], &engines[..matched]),
        )
    }

    /// Blocks of a run `worker` holds, read from a snapshot (no lock).
    #[inline]
    fn held_in(&self, run_id: u32, window: &Window, worker: u32) -> usize {
        if has(self.slab.coverage(run_id), worker) {
            window.len as usize
        } else {
            self.arena
                .partial_find(window.partials, worker)
                .map_or(0, |(_, cutoff)| cutoff as usize)
        }
    }

    /// The plan where the blocks in hand leave the run's content, at its end or at a divergence
    /// at `offset`: descend into the child that continues the run there, open one without the
    /// lock when the run has a table, or lock a leaf (the worker's own to extend, or a shared one
    /// without a table yet).
    fn plan_child(&self, window: &Window, offset: usize, head: u64) -> Plan {
        match self
            .arena
            .table_find(window.children, Self::child_key(offset, head))
        {
            Some((child, generation)) => {
                // The child's header is the next line the walk reads: ask for it while the
                // version is confirmed.
                crate::prefetch::prefetch_read(std::ptr::from_ref(self.slab.run(child)));
                Plan::Descend(child, generation)
            }
            None if window.children != NONE => Plan::InsertAt(offset),
            None => Plan::Lock,
        }
    }

    /// Match `remaining` against the run from `offset`: join the run as far as it matches, record
    /// the matched blocks, and say how to go on (a divergence inside the run continues as a
    /// child hanging off it; the run is never split for one).
    #[expect(clippy::too_many_arguments)]
    fn store_in_run<'b>(
        &self,
        worker: u32,
        run_id: u32,
        meta: &mut RunMeta,
        offset: usize,
        remaining: &'b [StoredBlock],
        block_start: usize,
        pending: &mut Vec<Placed>,
    ) -> InRun<'b> {
        let run = self.slab.run(run_id);
        let len = run.len();
        if offset >= len {
            return InRun::Continue(remaining);
        }
        let held = self.held_by(run_id, worker);
        if held < offset {
            // The parent entry pointed past what this worker holds: the engine re-stored under
            // a stale parent, or named one position by two engine hashes (the same content under
            // the same parent) and removed the one the index filed the position under. Cut here
            // and join the suffix; when nobody holds anything past the cut and no child hangs
            // there, the split leaves nothing to join (`GONE`) and the run simply ends at the
            // cut: the blocks go after it, like any store past a run's end.
            self.count_split(SplitCause::MidRunStore);
            let suffix = self.split_locked(run_id, meta, offset);
            if suffix == GONE {
                return InRun::Continue(remaining);
            }
            return InRun::MoveTo(suffix, self.slab.run(suffix).generation());
        }
        let base = run.base.load(Ordering::Relaxed) + offset as u32;
        let data = run.block.load(Ordering::Relaxed) + base;
        let hashes = self.arena.words(data, len - offset);
        let engines = self
            .arena
            .words(run.engine.load(Ordering::Relaxed) + base, len - offset);
        let (matched, conflicts) = self.match_run(hashes, engines, remaining, true);
        let available = len - offset;
        let reach = offset + matched;
        // A divergence inside the run leaves the run whole: the blocks from the divergence on
        // continue it as a child hanging off `reach` (see `store_at`).
        let diverges = matched < available && matched < remaining.len();
        let before = self.held_len(run_id);
        if reach > held {
            self.set_holding(run_id, worker, reach);
            self.credit(worker, reach - held);
        }
        self.settle_distinct(worker, before, run_id);
        pending.push(Placed {
            run: run_id,
            offset: offset as u32,
            start: block_start,
            count: matched,
            conflicts,
        });
        if matched == remaining.len() {
            // The store ends in this run: a safe point to cap its prefix holders (a split here
            // moves nothing this store still has to place).
            self.cap_prefix_holders(run_id, meta);
            return InRun::Done;
        }
        if diverges {
            return InRun::ContinueAt(&remaining[matched..], reach);
        }
        InRun::Continue(&remaining[matched..])
    }

    /// Place `remaining` after `offset` blocks of the run (locked by the caller): in the child
    /// that already continues the run there, in place when the run is this worker's own leaf and
    /// `offset` is its end, or in a new child linked at `offset`. `Some` names a child that
    /// already existed, for the caller to carry on in.
    #[expect(clippy::too_many_arguments)]
    fn store_at(
        &self,
        worker: u32,
        run_id: u32,
        _meta: &RunMeta,
        offset: usize,
        remaining: &[StoredBlock],
        block_start: usize,
        pending: &mut Vec<Placed>,
    ) -> Option<(u32, u32)> {
        let run = self.slab.run(run_id);
        let coverage = self.slab.coverage(run_id);
        let head = remaining[0].content_hash.0;
        let children = run.children.load(Ordering::Relaxed);
        if let Some(found) = self
            .arena
            .table_find(children, Self::child_key(offset, head))
        {
            return Some(found);
        }
        let len = run.len();
        debug_assert!(
            offset <= len,
            "a child hangs off the run: {offset} <= {len}"
        );
        let contents: Vec<u64> = remaining
            .iter()
            .map(|stored| stored.content_hash.0)
            .collect();
        let engines: Vec<u64> = remaining.iter().map(|stored| stored.seq_hash.0).collect();
        let own_leaf = offset == len
            && run_id != ROOT
            && children == NONE
            && run.forwards.load(Ordering::Relaxed) == NONE
            && covered_only_by(coverage, worker);
        let (target, first) = if own_leaf {
            self.append(run, &contents, &engines);
            (run_id, len)
        } else {
            let block = self
                .arena
                .alloc_array(&contents, capacity_for(contents.len()));
            let engine = self
                .arena
                .alloc_array(&engines, self.arena.array_capacity(block));
            let new_id = self.slab.alloc(
                run.start() + offset,
                run_id,
                Window {
                    block,
                    base: 0,
                    engine,
                    len: contents.len() as u32,
                    children: NONE,
                    partials: NONE,
                    forwards: NONE,
                },
            );
            set(self.slab.coverage(new_id), worker);
            if let Some(existing) = self.link_child(run, offset, head, new_id) {
                let mut freed = Vec::new();
                self.discard_run(new_id, worker, &mut freed);
                self.recycle(&mut freed);
                return Some(existing);
            }
            (new_id, 0)
        };
        pending.push(Placed {
            run: target,
            offset: first as u32,
            start: block_start,
            count: remaining.len(),
            conflicts: 0,
        });
        self.credit(worker, contents.len());
        self.distinct_add(worker, contents.len());
        None
    }

    /// Extend a leaf in place when its window ends the hash array and the array has room;
    /// otherwise move it to a larger array.
    fn append(&self, run: &Run, contents: &[u64], engines: &[u64]) {
        let block = run.block.load(Ordering::Relaxed);
        let engine = run.engine.load(Ordering::Relaxed);
        let base = run.base.load(Ordering::Relaxed) as usize;
        let len = run.len();
        let header = self.arena.array_header(block);
        let (used, capacity) = unpack(header.load(Ordering::Acquire));
        let end = base + len;
        let claimed = end == used as usize
            && end + contents.len() <= capacity as usize
            && header
                .compare_exchange(
                    pack(used, capacity),
                    pack((end + contents.len()) as u32, capacity),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok();
        if claimed {
            let slots = self.arena.words(block + end as u32, contents.len());
            for (slot, &hash) in slots.iter().zip(contents) {
                slot.store(hash, Ordering::Relaxed);
            }
            let slots = self.arena.words(engine + end as u32, engines.len());
            for (slot, &hash) in slots.iter().zip(engines) {
                slot.store(hash, Ordering::Relaxed);
            }
            run.len
                .store((len + contents.len()) as u32, Ordering::Release);
            return;
        }
        let mut grown: Vec<u64> = self
            .arena
            .words(block + base as u32, len)
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed))
            .collect();
        grown.extend_from_slice(contents);
        let mut grown_engines: Vec<u64> = self
            .arena
            .words(engine + base as u32, len)
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed))
            .collect();
        grown_engines.extend_from_slice(engines);
        let new_block = self.arena.alloc_array(&grown, capacity_for(grown.len()));
        // The engine array must hold at least what the content array can: an in-place append
        // claims room from the content array's header and writes both.
        let new_engine = self
            .arena
            .alloc_array(&grown_engines, self.arena.array_capacity(new_block));
        run.begin_update();
        run.block.store(new_block, Ordering::Relaxed);
        run.base.store(0, Ordering::Relaxed);
        run.engine.store(new_engine, Ordering::Relaxed);
        run.len.store(grown.len() as u32, Ordering::Release);
        run.end_update();
        self.arena.array_release(block);
        self.arena.array_release(engine);
    }

    /// Forget the named blocks of `worker`; unknown hashes are ignored.
    pub fn apply_removed(&self, worker: u32, hashes: &[SequenceHash], map: &mut ChainBlockMap) {
        let mut refs: Vec<BlockRef> = Vec::with_capacity(hashes.len());
        map.remove_all(hashes, |at| refs.push(at));
        let work = group_by_run(refs);
        self.apply_removals(worker, work);
    }

    /// Drop `worker` from the grouped places, one run at a time, forwarding what a split moved.
    fn apply_removals(&self, worker: u32, mut work: Vec<Removal>) {
        // The runs' headers are the lines the locked work reads first: ask for all of them now
        // so their misses overlap instead of serialising one run after another.
        for removal in &work {
            if removal.run != GONE {
                crate::prefetch::prefetch_read(std::ptr::from_ref(self.slab.run(removal.run)));
            }
        }
        let mut freed = Vec::new();
        while let Some(removal) = work.pop() {
            self.remove_from_run(worker, removal, &mut work, &mut freed);
            self.recycle(&mut freed);
        }
    }

    /// Drop `worker` from the offsets of one run, forwarding offsets a split moved on.
    fn remove_from_run(
        &self,
        worker: u32,
        removal: Removal,
        work: &mut Vec<Removal>,
        freed: &mut Vec<u32>,
    ) {
        if removal.run == GONE {
            return;
        }
        let run = self.slab.run(removal.run);
        let mut meta = self.lock_run(removal.run);
        if meta.dead
            || removal
                .generation
                .is_some_and(|generation| generation != run.generation())
        {
            return;
        }
        let offsets = reforward(
            &self.arena,
            run.forwards.load(Ordering::Relaxed),
            removal.offsets,
            work,
        );
        if offsets.is_empty() || self.held_by(removal.run, worker) == 0 {
            return;
        }
        self.remove_ranges(worker, removal.run, &mut meta, offsets);
        if !self.has_holders(removal.run) && run.children.load(Ordering::Relaxed) == NONE {
            self.unlink_locked(removal.run, &mut meta, freed);
        }
    }

    /// Drop `worker` from the given offsets of a run it covers, splitting the run so that the
    /// pieces it still covers keep their offsets.
    fn remove_ranges(&self, worker: u32, run_id: u32, meta: &mut RunMeta, mut offsets: Vec<u32>) {
        offsets.sort_unstable();
        offsets.dedup();
        let mut held = self.held_by(run_id, worker);
        offsets.retain(|&offset| (offset as usize) < held);
        // Contiguous ranges, highest first, so earlier ranges keep their offsets after the splits
        // a later range causes.
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for &offset in &offsets {
            let offset = offset as usize;
            match ranges.last_mut() {
                Some((_, high)) if *high + 1 == offset => *high = offset,
                _ => ranges.push((offset, offset)),
            }
        }
        for (low, high) in ranges.into_iter().rev() {
            // Blocks after the range stay held by this worker too: a hole, so the tail becomes
            // its own run. A range reaching the worker's end only lowers its cutoff.
            if high + 1 < held {
                self.count_split(SplitCause::Hole);
                self.split_locked(run_id, meta, high + 1);
            }
            let before = self.held_len(run_id);
            self.set_holding(run_id, worker, low);
            self.debit(worker, high + 1 - low);
            self.settle_distinct(worker, before, run_id);
            held = low;
        }
        // Every range is applied: cap the run's prefix holders (a tail eviction adds one).
        self.cap_prefix_holders(run_id, meta);
    }

    /// Forget every block of `worker` (the engine cleared its cache); the map is emptied.
    pub fn apply_cleared(&self, worker: u32, map: &mut ChainBlockMap) {
        let drained = std::mem::take(map);
        self.drop_worker(worker, drained);
    }

    /// Forget every block of `worker` (the worker left) and free its slot; the same name interns
    /// afresh afterwards. Nothing may write under `worker` from here on: the slot goes to the
    /// next name interned.
    pub fn remove_worker(&self, worker: u32, map: ChainBlockMap) {
        self.drop_worker(worker, map);
        self.release_worker(worker);
    }

    fn drop_worker(&self, worker: u32, map: ChainBlockMap) {
        // Keyed by (run, generation): a forwarding record to a dead generation of an id must not
        // shadow the live run that reused the id.
        let mut seen: FxHashSet<(u32, Option<u32>)> = FxHashSet::default();
        let mut work: Vec<(u32, Option<u32>)> = Vec::new();
        for (_, at) in map {
            if seen.insert((at.run, None)) {
                work.push((at.run, None));
            }
        }
        let mut freed = Vec::new();
        while let Some((run_id, generation)) = work.pop() {
            if run_id == GONE {
                continue;
            }
            let run = self.slab.run(run_id);
            let mut meta = self.lock_run(run_id);
            if meta.dead || generation.is_some_and(|generation| generation != run.generation()) {
                continue;
            }
            // Blocks of this worker may have moved into suffixes since the map was written.
            for (_, suffix, suffix_generation) in self
                .arena
                .forward_records(run.forwards.load(Ordering::Relaxed))
            {
                if suffix != GONE && seen.insert((suffix, Some(suffix_generation))) {
                    work.push((suffix, Some(suffix_generation)));
                }
            }
            let held = self.held_by(run_id, worker);
            if held > 0 {
                let before = self.held_len(run_id);
                self.set_holding(run_id, worker, 0);
                self.debit(worker, held);
                self.settle_distinct(worker, before, run_id);
                if !self.has_holders(run_id) && run.children.load(Ordering::Relaxed) == NONE {
                    self.unlink_locked(run_id, &mut meta, &mut freed);
                }
            }
            drop(meta);
            self.recycle(&mut freed);
        }
    }

    /// Every block every worker holds, as `(worker, position, content hash, prefix hash)`;
    /// for tests and for comparing against the reference indexer. Not consistent under
    /// concurrent writes.
    #[doc(hidden)]
    pub fn debug_blocks(&self) -> BTreeSet<(u32, usize, ContentHash, SequenceHash)> {
        let mut out = BTreeSet::new();
        let mut stack: Vec<(u32, Option<SequenceHash>)> = Vec::new();
        let (root, _) = self.slab.run(ROOT).snapshot();
        for (_, child, _) in self.arena.table_entries(root.children) {
            stack.push((child, None));
        }
        while let Some((run_id, mut prefix)) = stack.pop() {
            let run = self.slab.run(run_id);
            let (window, _) = run.snapshot();
            let start = run.start();
            let holders = workers(self.slab.coverage(run_id));
            let partials = self.arena.partial_entries(window.partials);
            let hashes = self
                .arena
                .words(window.block + window.base, window.len as usize);
            // The prefix hash after each count of the run's blocks: a child hangs off any offset
            // and continues from the hash at its own.
            let mut prefixes: Vec<Option<SequenceHash>> = Vec::with_capacity(hashes.len() + 1);
            prefixes.push(prefix);
            for (offset, slot) in hashes.iter().enumerate() {
                let content = ContentHash(slot.load(Ordering::Relaxed));
                let next = match prefix {
                    Some(previous) => chain_prefix_hash(previous, content),
                    None => SequenceHash(content.0),
                };
                for &worker in &holders {
                    out.insert((worker, start + offset, content, next));
                }
                for &(worker, cutoff) in &partials {
                    if offset < cutoff as usize {
                        out.insert((worker, start + offset, content, next));
                    }
                }
                prefix = Some(next);
                prefixes.push(prefix);
            }
            for (_, child, _) in self.arena.table_entries(window.children) {
                let child_offset = self.slab.run(child).start().saturating_sub(start);
                stack.push((child, prefixes[child_offset.min(prefixes.len() - 1)]));
            }
        }
        out
    }

    /// Adjacent runs a compaction could fold: a run with exactly one child whose coverage equals
    /// its own and that has no prefix holders of its own (a prefix holder of the child that is
    /// not a whole holder of the parent holds a suffix, which one run could not express).
    /// Returns the pairs and the blocks the children hold. A diagnostic walk, not consistent
    /// under concurrent writes.
    #[doc(hidden)]
    pub fn debug_mergeable(&self) -> (usize, usize) {
        let (strict, _, blocks) = self.debug_mergeable_by_rule();
        (strict, blocks)
    }

    /// As [`debug_mergeable`](Self::debug_mergeable), but also counting the pairs a generalised
    /// merge could fold: the child's whole holders a subset of the parent's and every prefix
    /// holder of the child a whole holder of the parent (the parent's extra holders would become
    /// prefix holders of the merged run). Returns `(strict pairs, generalised pairs, blocks of the
    /// generalised pairs' children)`.
    #[doc(hidden)]
    fn debug_mergeable_by_rule(&self) -> (usize, usize, usize) {
        let (mut strict, mut general, mut blocks) = (0usize, 0usize, 0usize);
        let mut stack: Vec<u32> = Vec::new();
        let (root, _) = self.slab.run(ROOT).snapshot();
        for (_, child, _) in self.arena.table_entries(root.children) {
            stack.push(child);
        }
        while let Some(run_id) = stack.pop() {
            let (window, _) = self.slab.run(run_id).snapshot();
            let children: Vec<u32> = self
                .arena
                .table_entries(window.children)
                .into_iter()
                .map(|(_, child, _)| child)
                .collect();
            if let [only] = children[..] {
                let (child_window, _) = self.slab.run(only).snapshot();
                let parent_words: Vec<u64> = self
                    .slab
                    .coverage(run_id)
                    .iter()
                    .map(|w| w.load(Ordering::Relaxed))
                    .collect();
                let child_words: Vec<u64> = self
                    .slab
                    .coverage(only)
                    .iter()
                    .map(|w| w.load(Ordering::Relaxed))
                    .collect();
                let same_coverage = parent_words == child_words;
                let subset = parent_words
                    .iter()
                    .zip(&child_words)
                    .all(|(p, c)| c & !p == 0);
                let child_partials = self.arena.partial_entries(child_window.partials);
                let partials_held = child_partials
                    .iter()
                    .all(|&(worker, _)| has(self.slab.coverage(run_id), worker));
                if same_coverage && child_partials.is_empty() {
                    strict += 1;
                }
                if subset && partials_held {
                    general += 1;
                    blocks += child_window.len as usize;
                }
            }
            stack.extend(children);
        }
        (strict, general, blocks)
    }

    /// Shape and memory counters.
    pub fn stats(&self) -> ChainIndexStats {
        let allocated = self.slab.allocated();
        let mut stats = ChainIndexStats {
            runs_allocated: allocated,
            runs_free: self.slab.free.len(),
            arena_bytes: self.arena.used() as usize * size_of::<AtomicU64>(),
            arena_free_bytes: self.arena.free_words() * size_of::<AtomicU64>(),
            arena_chunk_bytes: self.arena.chunk_bytes(),
            header_bytes: allocated * (size_of::<Run>() + self.words * size_of::<AtomicU64>()),
            slab_bytes: self.slab.chunk_bytes(),
            engine_conflicts: self.engine_conflicts.load(Ordering::Relaxed),
            landing_mismatches: self.landing_mismatches.load(Ordering::Relaxed),
            moved_hashes: self.moved_hashes.load(Ordering::Relaxed),
            splits_by_branch: self.splits_branch.load(Ordering::Relaxed),
            splits_by_hole: self.splits_hole.load(Ordering::Relaxed),
            splits_by_mid_run_store: self.splits_mid_run.load(Ordering::Relaxed),
            splits_by_prefix_holders: self.splits_prefix_holders.load(Ordering::Relaxed),
            runs_died: self.runs_died.load(Ordering::Relaxed),
            ..ChainIndexStats::default()
        };
        for id in 1..allocated as u32 {
            let run = self.slab.run(id);
            if run.meta.lock().dead {
                continue;
            }
            stats.runs_live += 1;
            stats.blocks_live += run.len();
            let (window, _) = run.snapshot();
            let partials = self.arena.partial_entries(window.partials).len();
            stats.partial_entries += partials;
            stats.max_partials = stats.max_partials.max(partials);
            if window.children != NONE {
                let live = self.arena.table_live(window.children);
                stats.child_entries += live;
                stats.child_tombstones += self.arena.table_dead(window.children);
            }
        }
        stats
    }
}

/// Offsets taken from the map may have moved into suffixes since they were written: send those
/// on, tagged with the generation the suffix had at the split.
/// The places of one worker's blocks grouped by run (stored coordinates; the per-run work
/// forwards what a split moved).
fn group_by_run(mut refs: Vec<BlockRef>) -> Vec<Removal> {
    refs.sort_unstable_by_key(|at| at.run);
    let mut work: Vec<Removal> = Vec::new();
    let mut index = 0;
    while index < refs.len() {
        let run = refs[index].run;
        let end = refs[index..]
            .iter()
            .position(|at| at.run != run)
            .map_or(refs.len(), |count| index + count);
        work.push(Removal {
            run,
            generation: None,
            offsets: refs[index..end].iter().map(|at| at.offset).collect(),
        });
        index = end;
    }
    work
}

fn reforward(
    arena: &WordArena,
    forwards: u32,
    mut offsets: Vec<u32>,
    work: &mut Vec<Removal>,
) -> Vec<u32> {
    if forwards == NONE {
        return offsets;
    }
    let mut forwarded: FxHashMap<(u32, u32), Vec<u32>> = FxHashMap::default();
    offsets.retain(|&offset| match arena.forwards_find(forwards, offset) {
        Some((next, generation)) => {
            if next.run != GONE {
                forwarded
                    .entry((next.run, generation))
                    .or_default()
                    .push(next.offset);
            }
            false
        }
        None => true,
    });
    work.extend(
        forwarded
            .into_iter()
            .map(|((run, generation), offsets)| Removal {
                run,
                generation: Some(generation),
                offsets,
            }),
    );
    offsets
}
