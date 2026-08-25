//! Positional indexer for cache-aware routing.
//!
//! Uses a single `DashMap<(usize, ContentHash), IndexEntry>` keyed by (position, content_hash).
//! Unbounded by default; [`PositionalIndexer::prune`] optionally bounds it with a
//! last-touch TTL and/or a capacity ceiling (oldest-first eviction).
//! Jump search skips positions in strides, yielding amortized O(D/J + W) complexity.
//!
//! **Dual-hash scheme**: backends send a position-aware `block_hash` (SequenceHash)
//! and raw `token_ids` per block. The router computes a position-independent
//! ContentHash (XXH3) from token_ids, then a rolling prefix hash (also XXH3) from
//! the ContentHash sequence. SeqEntry is keyed by the router's prefix hash for
//! precise disambiguation at query time. The backend's SequenceHash is stored in
//! worker_blocks only, used for `apply_removed` reverse lookup.
//!
//! **Performance**: Internal u32 worker IDs eliminate Arc<str> hashing and atomic
//! refcount bouncing in the hot query loop. Caller-owned `WorkerBlockMap` gives
//! direct HashMap access (~5ns) on the write path — no DashMap hash+shard locking.
//! Atomic tree_sizes provide O(1) size queries.
//!
//! Thread safety: the shared `index` DashMap is internally synchronized via sharding.
//! `WorkerBlockMap` is caller-owned (one per tokio task), so no cross-thread
//! synchronization is needed on the write path.

use std::{
    fmt,
    sync::{
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    time::Instant,
};

use dashmap::{mapref::entry::Entry, DashMap};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

/// Seed for XXH3 hashing.
pub const XXH3_SEED: u64 = 1337;

/// Shard count for the main index DashMap.
/// Tuned iteratively — higher values reduce per-shard contention under concurrent
/// reads+writes at the cost of more memory for shard locks.
const INDEX_SHARD_COUNT: usize = 1024;

/// Shard count for worker-keyed DashMaps (worker_blocks, worker_to_id).
/// These maps hold at most ~500 entries (one per worker), so 8 shards is sufficient.
const WORKER_SHARD_COUNT: usize = 8;

/// Capacity-pass eviction skips entries touched within this many seconds.
///
/// `apply_stored` inserts a batch's index entries before its single deferred
/// `tree_sizes` increment; evicting one of those entries in that gap would
/// decrement a counter that was never incremented (clamping at zero) and
/// leave the later increment counting an already-evicted entry. Entries
/// stamped within the grace are never capacity-eviction candidates, so a
/// prune cannot race a store batch that is still being accounted (the TTL
/// pass is inherently safe: a just-stored stamp is always newer than any
/// cutoff). Side effect: a burst of all-fresh entries can transiently exceed
/// the ceiling until they age past the grace.
const CAPACITY_EVICTION_GRACE_SECS: u32 = 2;

/// Length of the first `TreeSizes` segment (covers worker ids 0..2048).
/// Segment `s` doubles to `FIRST_SEGMENT_LEN << s` entries, so worker count is
/// unbounded while reads stay a lock-free array index on the query hot path.
const FIRST_SEGMENT_LEN: usize = 2048;

/// log2 of [`FIRST_SEGMENT_LEN`].
const FIRST_SEGMENT_BITS: u32 = FIRST_SEGMENT_LEN.trailing_zeros();

/// Number of doubling segments needed to cover the entire u32 worker-id space.
const SEGMENT_COUNT: usize = (u32::BITS - FIRST_SEGMENT_BITS + 1) as usize;

/// Position-independent content hash of tokens within a single block.
/// Computed via XXH3-64 from token IDs. Same tokens always produce the same hash
/// regardless of their position in the sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ContentHash(pub u64);

/// Position-aware block hash from backend (sequence hash).
/// Matches the `block_hash` field in KvBlock proto (i64, bitwise reinterpreted as u64).
/// Different from ContentHash because it encodes the full prefix history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SequenceHash(pub u64);

impl From<i64> for SequenceHash {
    fn from(value: i64) -> Self {
        Self(value as u64)
    }
}

impl From<u64> for SequenceHash {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

/// Internal worker identifier used in [`OverlapScores`].
///
/// Consumers map worker URLs to this type via [`PositionalIndexer::worker_id`].
pub type WorkerId = u32;

/// A block from a store event, carrying both hash representations.
#[derive(Debug, Clone, Copy)]
pub struct StoredBlock {
    /// Position-aware hash from the backend proto (`block_hash` field).
    pub seq_hash: SequenceHash,
    /// Position-independent hash computed from token IDs via XXH3.
    pub content_hash: ContentHash,
}

/// Error returned by [`PositionalIndexer::apply_stored`] when the event cannot be applied.
#[derive(Debug)]
pub enum ApplyError {
    /// Worker has no entries in the index — cannot resolve parent block.
    WorkerNotTracked,
    /// The specified `parent_seq_hash` was not found in this worker's reverse lookup.
    ParentBlockNotFound,
}

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkerNotTracked => write!(f, "worker not tracked in index"),
            Self::ParentBlockNotFound => write!(f, "parent block hash not found for worker"),
        }
    }
}

impl std::error::Error for ApplyError {}

/// Error returned by [`PositionalIndexer::intern_worker`] when the u32 worker-id
/// space is exhausted. Ids are assigned monotonically and never recycled, so this
/// can only be reached after `u32::MAX + 1` distinct worker URLs have been interned
/// over the indexer's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerIdExhausted;

impl fmt::Display for WorkerIdExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "worker id space exhausted (u32::MAX workers interned)")
    }
}

impl std::error::Error for WorkerIdExhausted {}

/// Overlap scores: how many consecutive blocks each worker has cached.
///
/// Keys are internal `u32` worker IDs. Use [`PositionalIndexer::worker_id`] to
/// map a worker URL to its internal ID for lookups.
#[derive(Debug, Default)]
pub struct OverlapScores {
    /// internal_worker_id → number of matching prefix blocks (depth in indexer)
    pub scores: FxHashMap<u32, u32>,
    /// internal_worker_id → total blocks cached by this worker
    pub tree_sizes: FxHashMap<u32, usize>,
}

/// Compute content hash from token IDs (position-independent).
/// Uses XXH3-64 streaming hasher with standard seed — avoids intermediate allocation.
pub fn compute_content_hash(token_ids: &[u32]) -> ContentHash {
    use std::hash::Hasher;
    let mut hasher = xxhash_rust::xxh3::Xxh3::with_seed(XXH3_SEED);
    for &t in token_ids {
        hasher.write(&t.to_le_bytes());
    }
    ContentHash(hasher.finish())
}

/// Chunk request tokens by block size and compute a [`ContentHash`] per full block.
///
/// This is the entry point for the **query path**: given a request's token IDs and
/// the backend's block size, produce the content-hash sequence that
/// [`PositionalIndexer::find_matches`] expects.
///
/// Partial trailing chunks (fewer tokens than `block_size`) are discarded because
/// backends only cache full blocks.
///
/// Returns an empty `Vec` if `block_size` is 0.
pub fn compute_request_content_hashes(tokens: &[u32], block_size: usize) -> Vec<ContentHash> {
    if block_size == 0 {
        tracing::warn!("compute_request_content_hashes called with block_size=0, returning empty");
        return Vec::new();
    }
    tokens
        .chunks(block_size)
        .filter(|chunk| chunk.len() == block_size)
        .map(compute_content_hash)
        .collect()
}

// ---------------------------------------------------------------------------
// SeqEntry: optimizes for the common case (one seq_hash per position+content)
// ---------------------------------------------------------------------------

/// Entry for the innermost level of the index.
///
/// Optimizes for the common case where there's only one sequence hash
/// at a given (position, content_hash) pair, avoiding HashMap allocation.
#[derive(Debug, Clone)]
enum SeqEntry {
    /// Single seq_hash → workers mapping (common case, no HashMap allocation).
    Single(SequenceHash, FxHashSet<u32>),
    /// Multiple seq_hash → workers mappings (rare: different prefixes with same content).
    Multi(FxHashMap<SequenceHash, FxHashSet<u32>>),
}

impl SeqEntry {
    fn new(seq_hash: SequenceHash, worker_id: u32) -> Self {
        let mut workers = FxHashSet::default();
        workers.insert(worker_id);
        Self::Single(seq_hash, workers)
    }

    /// Insert a worker for a given seq_hash, upgrading to Multi if needed.
    /// Returns whether the membership was newly added (false for a duplicate
    /// store of an existing membership) so callers can keep `tree_sizes`
    /// consistent with what is actually in the index.
    fn insert(&mut self, seq_hash: SequenceHash, worker_id: u32) -> bool {
        match self {
            Self::Single(existing_hash, workers) if *existing_hash == seq_hash => {
                workers.insert(worker_id)
            }
            Self::Single(existing_hash, existing_workers) => {
                let mut map = FxHashMap::with_capacity_and_hasher(2, FxBuildHasher);
                map.insert(*existing_hash, std::mem::take(existing_workers));
                let added = map.entry(seq_hash).or_default().insert(worker_id);
                *self = Self::Multi(map);
                added
            }
            Self::Multi(map) => map.entry(seq_hash).or_default().insert(worker_id),
        }
    }

    /// Remove a worker from a given seq_hash.
    /// Returns `(removed, now_empty)`: whether the worker was actually present
    /// under that seq_hash (so callers can keep `tree_sizes` consistent — a
    /// membership already gone, e.g. pruned, must not be decremented twice),
    /// and whether the entry is now completely empty and should be dropped.
    fn remove(&mut self, seq_hash: SequenceHash, worker_id: u32) -> (bool, bool) {
        match self {
            Self::Single(existing_hash, workers) if *existing_hash == seq_hash => {
                let removed = workers.remove(&worker_id);
                (removed, workers.is_empty())
            }
            Self::Single(_, _) => (false, false),
            Self::Multi(map) => {
                let mut removed = false;
                if let Some(workers) = map.get_mut(&seq_hash) {
                    removed = workers.remove(&worker_id);
                    if workers.is_empty() {
                        map.remove(&seq_hash);
                    }
                }
                (removed, map.is_empty())
            }
        }
    }

    /// Count per-worker memberships in this entry, accumulating into `acc`.
    /// A worker appearing under multiple prefix hashes counts once per prefix
    /// hash — mirroring how `apply_stored` incremented `tree_sizes`. Used by
    /// [`PositionalIndexer::prune`] to decrement counters on eviction.
    fn accumulate_worker_counts(&self, acc: &mut FxHashMap<u32, usize>) {
        match self {
            Self::Single(_, workers) => {
                for &w in workers {
                    *acc.entry(w).or_default() += 1;
                }
            }
            Self::Multi(map) => {
                for workers in map.values() {
                    for &w in workers {
                        *acc.entry(w).or_default() += 1;
                    }
                }
            }
        }
    }

    /// Get workers for a specific prefix hash (used in query path and event processing).
    fn get(&self, seq_hash: SequenceHash) -> Option<&FxHashSet<u32>> {
        match self {
            Self::Single(existing_hash, workers) if *existing_hash == seq_hash => Some(workers),
            Self::Single(_, _) => None,
            Self::Multi(map) => map.get(&seq_hash),
        }
    }

    /// For Single entries, return the worker set directly without prefix hash check.
    /// Content hash collisions at 64-bit XXH3 are practically impossible (~2^-64),
    /// so a matching content_hash at the same position is unambiguous — the rolling
    /// hash computation can be skipped entirely.
    /// Returns None for Multi entries — caller must compute prefix hash to disambiguate.
    #[inline]
    fn workers_if_single(&self) -> Option<&FxHashSet<u32>> {
        match self {
            Self::Single(_, workers) => Some(workers),
            Self::Multi(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// TreeSizes: growable, lock-free per-worker block counters
// ---------------------------------------------------------------------------

/// Per-worker block counters indexed by worker id, with no worker-count cap.
///
/// Storage is segmented with doubling sizes: segment `s` covers ids
/// `[FIRST_SEGMENT_LEN * (2^s - 1), FIRST_SEGMENT_LEN * (2^(s+1) - 1))`, so
/// [`SEGMENT_COUNT`] segments cover every possible u32 id. Segments are
/// allocated on first write and never moved, which keeps reads a lock-free
/// array index (~1ns) on the query hot path — the property the previous fixed
/// 2048-slot Vec was built for — while supporting unbounded worker counts.
///
/// Memory: 8 bytes per worker id, at most ~2x the high-water id due to
/// doubling; segments are never freed. Ids are never recycled, so cost grows
/// with workers ever interned — same lifecycle as the `worker_to_id` map,
/// whose per-URL entries are an order of magnitude larger.
struct TreeSizes {
    segments: [OnceLock<Box<[AtomicUsize]>>; SEGMENT_COUNT],
}

impl TreeSizes {
    fn new() -> Self {
        Self {
            segments: std::array::from_fn(|_| OnceLock::new()),
        }
    }

    /// Map a worker id to (segment index, offset within segment).
    ///
    /// Computed in u64: `id + FIRST_SEGMENT_LEN` overflows u32 for ids near
    /// `u32::MAX`, and the last segment's length (`2^32`) overflows 32-bit usize.
    #[inline]
    fn locate(id: u32) -> (usize, usize) {
        let virtual_idx = id as u64 + FIRST_SEGMENT_LEN as u64;
        let msb = 63 - virtual_idx.leading_zeros();
        let segment = (msb - FIRST_SEGMENT_BITS) as usize;
        let offset = (virtual_idx - (1u64 << msb)) as usize;
        (segment, offset)
    }

    #[inline]
    fn segment_len(segment: usize) -> usize {
        FIRST_SEGMENT_LEN << segment
    }

    /// Counter slot for a worker id, allocating its segment on first use.
    fn slot(&self, id: u32) -> &AtomicUsize {
        let (segment, offset) = Self::locate(id);
        let entries = self.segments[segment].get_or_init(|| {
            (0..Self::segment_len(segment))
                .map(|_| AtomicUsize::new(0))
                .collect()
        });
        &entries[offset]
    }

    /// Lock-free read. Ids whose segment was never written read as 0.
    #[inline]
    fn load(&self, id: u32) -> usize {
        let (segment, offset) = Self::locate(id);
        match self.segments[segment].get() {
            Some(entries) => entries[offset].load(Ordering::Relaxed),
            None => 0,
        }
    }

    /// Reset a worker's count to 0 without allocating its segment if absent.
    fn reset(&self, id: u32) {
        let (segment, offset) = Self::locate(id);
        if let Some(entries) = self.segments[segment].get() {
            entries[offset].store(0, Ordering::Relaxed);
        }
    }

    /// Sum of all counters across allocated segments.
    fn total(&self) -> usize {
        self.segments
            .iter()
            .filter_map(OnceLock::get)
            .flat_map(|entries| entries.iter())
            .map(|size| size.load(Ordering::Relaxed))
            .sum()
    }

    /// Subtract `n` from a worker's count, saturating at 0. Prune-side
    /// decrements are approximate (see `apply_stored`'s duplicate-store
    /// counting), so saturation — not wrapping — is the safe failure mode.
    fn sub_saturating(&self, id: u32, n: usize) {
        let slot = self.slot(id);
        let mut current = slot.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_sub(n);
            match slot.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PositionalIndexer
// ---------------------------------------------------------------------------

/// Per-worker reverse lookup: backend_seq_hash → (position, content_hash, prefix_hash).
/// The `prefix_hash` is the router-computed rolling hash used as the SeqEntry key.
///
/// Callers own one `WorkerBlockMap` per worker and pass it to write-path methods.
pub type WorkerBlockMap = FxHashMap<SequenceHash, (usize, ContentHash, SequenceHash)>;

/// Value stored in the positional index: the seq-hash/worker entry plus a
/// coarse last-access stamp consumed by [`PositionalIndexer::prune`].
#[derive(Debug)]
struct IndexEntry {
    seq: SeqEntry,
    /// Seconds since the indexer's epoch at the last store or successful query
    /// read. Relaxed atomics throughout: a stamp stale by one prune cycle only
    /// delays eviction by one interval.
    last_touch: AtomicU32,
}

impl IndexEntry {
    fn new(seq: SeqEntry, now: u32) -> Self {
        Self {
            seq,
            last_touch: AtomicU32::new(now),
        }
    }

    #[inline]
    fn touch(&self, now: u32) {
        self.last_touch.store(now, Ordering::Relaxed);
    }
}

/// The positional index map type (see [`IndexEntry`]).
type PosIndex = DashMap<(usize, ContentHash), IndexEntry, FxBuildHasher>;

/// Outcome of a [`PositionalIndexer::prune`] pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PruneStats {
    /// Entries inspected across both passes.
    pub scanned: usize,
    /// Entries evicted because their last touch exceeded the TTL.
    pub evicted_ttl: usize,
    /// Entries evicted to enforce the capacity ceiling (oldest first).
    pub evicted_capacity: usize,
    /// Entries remaining after the pass.
    pub remaining: usize,
}

/// Positional indexer for cache-aware routing.
///
/// Uses a single `DashMap<(usize, ContentHash), IndexEntry>` — keyed by
/// (position, content_hash). Unbounded by default; [`prune`](Self::prune)
/// optionally bounds it with a last-touch TTL and a capacity ceiling.
/// Jump search gives amortized O(D/J + W) matching complexity.
///
/// Write-path methods take a caller-owned `&mut WorkerBlockMap` (one per worker).
/// This gives direct HashMap access (~5ns) instead of DashMap hash+shard locking
/// (~25ns), achieving zero-contention for single-writer-per-worker.
pub struct PositionalIndexer {
    /// Single flat index: (position, content_hash) → IndexEntry.
    /// Bounded only when the owner runs [`prune`](Self::prune).
    index: PosIndex,
    /// Per-worker block counts, tracked atomically for O(1) reads during queries.
    /// Segmented array indexed by worker_id — lock-free reads on the query hot
    /// path (array index ~1ns vs DashMap hash+lock+probe ~25ns per access),
    /// growable so the worker count is unbounded.
    tree_sizes: TreeSizes,
    /// Worker URL → internal u32 ID (fast path: DashMap shard read).
    worker_to_id: DashMap<Arc<str>, u32, FxBuildHasher>,
    /// Monotonic counter for assigning new worker IDs. Never recycled; u64 so
    /// exhaustion of the u32 id space is detected instead of wrapping.
    next_worker_id: AtomicU64,
    /// Jump size for search optimization (default 64).
    jump_size: usize,
    /// Origin of the coarse `last_touch` clock (whole seconds since creation).
    epoch: Instant,
    /// Test override for the coarse clock; `u32::MAX` = disabled.
    #[cfg(test)]
    test_now: AtomicU32,
}

impl PositionalIndexer {
    /// Create a new PositionalIndexer with the given jump size.
    ///
    /// `jump_size` controls how many positions the search algorithm skips at a time.
    /// Larger values reduce lookups on long matching prefixes but increase scan range
    /// when workers drain. Default: 64.
    pub fn new(jump_size: usize) -> Self {
        assert!(jump_size > 0, "jump_size must be greater than 0");
        Self {
            index: DashMap::with_hasher_and_shard_amount(FxBuildHasher, INDEX_SHARD_COUNT),
            tree_sizes: TreeSizes::new(),
            worker_to_id: DashMap::with_hasher_and_shard_amount(FxBuildHasher, WORKER_SHARD_COUNT),
            next_worker_id: AtomicU64::new(0),
            jump_size,
            epoch: Instant::now(),
            #[cfg(test)]
            test_now: AtomicU32::new(u32::MAX),
        }
    }

    /// Coarse clock: whole seconds since this indexer was created. Saturates
    /// at `u32::MAX - 1` (~136 years) so `u32::MAX` stays free as the test
    /// override sentinel.
    fn now_secs(&self) -> u32 {
        #[cfg(test)]
        {
            let t = self.test_now.load(Ordering::Relaxed);
            if t != u32::MAX {
                return t;
            }
        }
        self.epoch.elapsed().as_secs().min(u32::MAX as u64 - 1) as u32
    }

    /// Override the coarse clock in tests (u32::MAX restores the real clock).
    #[cfg(test)]
    fn set_test_now(&self, now: u32) {
        self.test_now.store(now, Ordering::Relaxed);
    }

    /// Get the internal u32 ID for a worker URL, if it has been interned.
    ///
    /// Used by consumers to look up scores in [`OverlapScores`] by worker URL.
    /// Returns `None` if the worker has never been seen by this indexer.
    pub fn worker_id(&self, worker: &str) -> Option<u32> {
        self.worker_to_id.get(worker).map(|entry| *entry.value())
    }

    /// Apply a "blocks stored" event for a worker.
    ///
    /// `worker_id`: internal ID from [`intern_worker`].
    /// `blocks`: ordered sequence of stored blocks (each with seq_hash + content_hash).
    /// `parent_seq_hash`: if Some, the sequence extends from the parent's position + 1.
    ///   If None, the sequence starts from position 0.
    /// `worker_blocks`: caller-owned reverse lookup for this worker.
    pub fn apply_stored(
        &self,
        worker_id: u32,
        blocks: &[StoredBlock],
        parent_seq_hash: Option<SequenceHash>,
        worker_blocks: &mut WorkerBlockMap,
    ) -> Result<(), ApplyError> {
        if blocks.is_empty() {
            return Ok(());
        }

        // Determine starting position and parent's router prefix hash.
        let (start_pos, parent_prefix) = match parent_seq_hash {
            Some(parent_hash) => {
                if worker_blocks.is_empty() {
                    return Err(ApplyError::WorkerNotTracked);
                }
                let Some(&(parent_pos, _, parent_pfx)) = worker_blocks.get(&parent_hash) else {
                    return Err(ApplyError::ParentBlockNotFound);
                };
                (parent_pos + 1, Some(parent_pfx))
            }
            None => (0, None),
        };

        let mut prev_prefix = parent_prefix;
        let mut num_new_blocks = 0usize;
        // One coarse stamp per batch — cheaper than per-block clock reads and
        // precise enough for prune's second-granularity TTL.
        let now = self.now_secs();
        for (i, block) in blocks.iter().enumerate() {
            let position = start_pos + i;
            let content_hash = block.content_hash;

            // Compute router prefix hash (rolling XXH3 of content hashes).
            // This is the SeqEntry key — consistent between store and query paths.
            let prefix_hash = match prev_prefix {
                Some(prev) => SequenceHash(Self::compute_next_seq_hash(prev.0, content_hash.0)),
                // Position 0: prefix_hash == content_hash (no parent to chain from).
                None => SequenceHash(content_hash.0),
            };

            let membership_added = match self.index.entry((position, content_hash)) {
                Entry::Occupied(mut occupied) => {
                    let entry = occupied.get_mut();
                    let added = entry.seq.insert(prefix_hash, worker_id);
                    entry.touch(now);
                    added
                }
                Entry::Vacant(vacant) => {
                    vacant.insert(IndexEntry::new(SeqEntry::new(prefix_hash, worker_id), now));
                    true
                }
            };

            // Keep the reverse map current regardless; count only memberships
            // actually added to the index. Duplicate store events still don't
            // inflate tree_sizes, and re-storing a block whose entry was
            // pruned (its stale reverse mapping kept) restores its count —
            // mirroring apply_removed, which decrements only memberships
            // actually removed.
            worker_blocks.insert(block.seq_hash, (position, content_hash, prefix_hash));
            if membership_added {
                num_new_blocks += 1;
            }
            prev_prefix = Some(prefix_hash);
        }

        // Atomically update tree_sizes — lock-free array index.
        if num_new_blocks > 0 {
            self.tree_sizes
                .slot(worker_id)
                .fetch_add(num_new_blocks, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Apply a "blocks removed" event for a worker.
    ///
    /// `worker_id`: internal ID from [`intern_worker`].
    /// `seq_hashes`: position-aware block hashes to remove (from proto).
    /// `worker_blocks`: caller-owned reverse lookup for this worker.
    ///
    /// **Note on orphaned entries**: Removing a block at position N does not cascade to
    /// blocks at positions > N. Those entries become orphaned — they remain in the index
    /// but won't match queries because the rolling prefix hash chain is broken at the gap.
    /// This is harmless: orphaned entries waste a small amount of memory and are cleaned up
    /// when the worker is cleared or removed. Backends typically evict from the tail (LRU),
    /// so mid-sequence gaps are rare in practice.
    pub fn apply_removed(
        &self,
        worker_id: u32,
        seq_hashes: &[SequenceHash],
        worker_blocks: &mut WorkerBlockMap,
    ) {
        let mut num_removed = 0usize;
        for &seq_hash in seq_hashes {
            let Some((position, content_hash, prefix_hash)) = worker_blocks.remove(&seq_hash)
            else {
                continue;
            };

            // Count only memberships actually removed from the index. A stale
            // reverse-map entry (its index entry was pruned earlier) must not
            // decrement tree_sizes a second time.
            if let Entry::Occupied(mut occupied) = self.index.entry((position, content_hash)) {
                let (removed, now_empty) = occupied.get_mut().seq.remove(prefix_hash, worker_id);
                if now_empty {
                    occupied.remove();
                }
                if removed {
                    num_removed += 1;
                }
            }
        }

        if num_removed > 0 {
            self.tree_sizes
                .slot(worker_id)
                .fetch_sub(num_removed, Ordering::Relaxed);
        }
    }

    /// Apply a "cache cleared" event — drain blocks, clean index, caller keeps the empty map.
    ///
    /// `worker_id`: internal ID from [`intern_worker`].
    /// `worker_blocks`: caller-owned reverse lookup — drained and left empty.
    pub fn apply_cleared(&self, worker_id: u32, worker_blocks: &mut WorkerBlockMap) {
        let drained = std::mem::take(worker_blocks);
        for &(position, content_hash, prefix_hash) in drained.values() {
            if let Entry::Occupied(mut occ) = self.index.entry((position, content_hash)) {
                let (_, now_empty) = occ.get_mut().seq.remove(prefix_hash, worker_id);
                if now_empty {
                    occ.remove();
                }
            }
        }
        self.tree_sizes.reset(worker_id);
    }

    /// Remove a worker entirely — takes ownership of blocks, cleans index, worker is gone.
    ///
    /// `worker_id`: internal ID from [`intern_worker`].
    /// `worker_blocks`: caller-owned reverse lookup — consumed. Removal is
    /// proportional to this worker's blocks, not to the total index size.
    pub fn remove_worker(&self, worker_id: u32, worker_blocks: WorkerBlockMap) {
        for (position, content_hash, prefix_hash) in worker_blocks.into_values() {
            if let Entry::Occupied(mut occ) = self.index.entry((position, content_hash)) {
                let (_, now_empty) = occ.get_mut().seq.remove(prefix_hash, worker_id);
                if now_empty {
                    occ.remove();
                }
            }
        }
        self.tree_sizes.reset(worker_id);
    }

    /// Get total number of blocks across all workers.
    pub fn current_size(&self) -> usize {
        self.tree_sizes.total()
    }

    /// Number of `(position, content_hash)` entries currently in the index.
    /// O(shards); intended for prune decisions and observability, not hot paths.
    pub fn entry_count(&self) -> usize {
        self.index.len()
    }

    /// Evict stale and/or excess index entries. Intended to be driven
    /// periodically by the owner (e.g. the routing policy's eviction cycle).
    ///
    /// * `ttl_secs`: entries neither stored to nor read by a query within this
    ///   many seconds are evicted. `None` or `Some(0)` disables the TTL pass.
    /// * `max_entries`: when the index holds more entries than this ceiling,
    ///   the oldest-touched entries are evicted down to a low-water mark of
    ///   90% of the ceiling (avoids re-evicting every cycle at the boundary).
    ///   `None` or `Some(0)` disables the capacity pass.
    ///
    /// Semantics and caveats:
    /// * Queries touch exactly the entries they read (position 0, jump
    ///   landings, and linear-scan ranges), so entries a hot request stream
    ///   actually needs stay resident; interior positions the jump shortcut
    ///   skips may age out — harmless, the shortcut never reads them, and a
    ///   later drain across an evicted position under-counts that one score
    ///   (same tolerance as the documented `apply_removed` gap behavior).
    /// * Eviction leaves the per-worker reverse maps (`WorkerBlockMap`)
    ///   untouched; a later `apply_removed` for a pruned block is a safe
    ///   no-op (membership check), and stale reverse entries are dropped by
    ///   the backend's own removal/clear events or worker removal.
    /// * `tree_sizes` is decremented per evicted membership (saturating), so
    ///   worker totals stay approximately consistent under eviction.
    ///
    /// Runs off the hot path; both passes collect candidate keys first and
    /// then re-check under the entry lock, sparing entries touched since the
    /// scan.
    pub fn prune(&self, ttl_secs: Option<u32>, max_entries: Option<usize>) -> PruneStats {
        self.prune_with_now(self.now_secs(), ttl_secs, max_entries)
    }

    fn prune_with_now(
        &self,
        now: u32,
        ttl_secs: Option<u32>,
        max_entries: Option<usize>,
    ) -> PruneStats {
        let mut stats = PruneStats::default();
        let mut decrements: FxHashMap<u32, usize> = FxHashMap::default();

        // Pass 1: TTL. `cutoff == 0` (indexer younger than the TTL) means
        // nothing can be stale yet.
        if let Some(ttl) = ttl_secs.filter(|&t| t > 0) {
            let cutoff = now.saturating_sub(ttl);
            if cutoff > 0 {
                let mut expired: Vec<(usize, ContentHash)> = Vec::new();
                for item in &self.index {
                    stats.scanned += 1;
                    if item.value().last_touch.load(Ordering::Relaxed) < cutoff {
                        expired.push(*item.key());
                    }
                }
                for key in expired {
                    if let Entry::Occupied(occ) = self.index.entry(key) {
                        // Re-check under the entry lock: touched since the scan → spare.
                        if occ.get().last_touch.load(Ordering::Relaxed) < cutoff {
                            occ.get().seq.accumulate_worker_counts(&mut decrements);
                            occ.remove();
                            stats.evicted_ttl += 1;
                        }
                    }
                }
            }
        }

        // Pass 2: capacity ceiling, oldest-touched first. Entries inside the
        // freshness grace are not candidates (see CAPACITY_EVICTION_GRACE_SECS),
        // so eviction can fall short of the low-water mark under an all-fresh
        // burst; the next cycle catches up once entries age.
        if let Some(max) = max_entries.filter(|&m| m > 0) {
            let len = self.index.len();
            if len > max {
                let low_water = max - max / 10;
                let fresh_cutoff = now.saturating_sub(CAPACITY_EVICTION_GRACE_SECS);
                let mut aged: Vec<(u32, (usize, ContentHash))> = Vec::new();
                for item in &self.index {
                    stats.scanned += 1;
                    let stamp = item.value().last_touch.load(Ordering::Relaxed);
                    if stamp < fresh_cutoff {
                        aged.push((stamp, *item.key()));
                    }
                }
                let evict_n = len.saturating_sub(low_water).min(aged.len());
                if evict_n > 0 {
                    aged.select_nth_unstable_by_key(evict_n - 1, |&(stamp, _)| stamp);
                    for &(stamp, key) in &aged[..evict_n] {
                        if let Entry::Occupied(occ) = self.index.entry(key) {
                            // Touched since the scan → spare it this cycle.
                            if occ.get().last_touch.load(Ordering::Relaxed) == stamp {
                                occ.get().seq.accumulate_worker_counts(&mut decrements);
                                occ.remove();
                                stats.evicted_capacity += 1;
                            }
                        }
                    }
                }
            }
        }

        for (worker, n) in decrements {
            self.tree_sizes.sub_saturating(worker, n);
        }
        stats.remaining = self.index.len();
        stats
    }

    /// Find overlap scores for a request's content hash sequence.
    ///
    /// Uses jump search: strides by `jump_size` positions, only scanning
    /// intermediate positions when workers drain (stop matching).
    /// Complexity: amortized O(D/J + W) where D=depth, J=jump_size, W=workers.
    ///
    /// When `early_exit` is true, returns immediately after finding any match
    /// at position 0 (score = 1 for all matching workers). Useful when the caller
    /// only needs to know whether any worker has cached data for this sequence.
    ///
    /// **Assumption**: Block sequences are prefix-closed — if a worker has a block at
    /// position N, it has blocks at all positions 0..N. This holds when backends evict
    /// from the tail (LRU). If `apply_removed` creates a mid-sequence gap, the rolling
    /// prefix hash detects it (the chain breaks at the gap), but the jump heuristic may
    /// over-count if it lands past the gap. In practice, backends only evict tail blocks.
    pub fn find_matches(&self, content_hashes: &[ContentHash], early_exit: bool) -> OverlapScores {
        self.jump_search_matches(content_hashes, early_exit)
    }

    // -----------------------------------------------------------------------
    // Internal: router prefix hash + jump search
    //
    // The router computes its own rolling hash from ContentHashes (XXH3).
    // This hash is stored in SeqEntry during apply_stored and recomputed
    // at query time for precise filtering.
    // The backend's SequenceHash (from proto block_hash) stays in
    // worker_blocks only, used for apply_removed reverse lookup.
    // -----------------------------------------------------------------------

    /// Compute rolling prefix hash: XXH3(prev || current).
    #[inline]
    fn compute_next_seq_hash(prev_seq_hash: u64, current_content_hash: u64) -> u64 {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&prev_seq_hash.to_le_bytes());
        bytes[8..].copy_from_slice(&current_content_hash.to_le_bytes());
        xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, XXH3_SEED)
    }

    /// Lazily compute prefix hashes up to `target_pos`.
    #[inline]
    fn ensure_seq_hash_computed(
        seq_hashes: &mut Vec<SequenceHash>,
        target_pos: usize,
        sequence: &[ContentHash],
    ) {
        while seq_hashes.len() <= target_pos {
            let pos = seq_hashes.len();
            if pos == 0 {
                seq_hashes.push(SequenceHash(sequence[0].0));
            } else {
                let prev = seq_hashes[pos - 1].0;
                let current = sequence[pos].0;
                seq_hashes.push(SequenceHash(Self::compute_next_seq_hash(prev, current)));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal: worker interning (u32 IDs)
    // -----------------------------------------------------------------------

    /// Intern a worker URL to an internal u32 ID.
    /// Fast path: DashMap shard read (no lock). Slow path: DashMap entry API (once per worker).
    ///
    /// Ids are assigned monotonically and never recycled: a URL keeps its id for
    /// the indexer's lifetime (including across [`remove_worker`](Self::remove_worker)),
    /// and new URLs always get a fresh id. Must never panic — it runs inside
    /// per-worker subscription tasks where a panic would silently stop KV event
    /// indexing for that worker. The only error is u32 id-space exhaustion.
    pub fn intern_worker(&self, worker: &str) -> Result<WorkerId, WorkerIdExhausted> {
        // Fast path: already interned
        if let Some(entry) = self.worker_to_id.get(worker) {
            return Ok(*entry.value());
        }
        // Slow path: the entry API holds the shard lock, so the vacant arm runs
        // at most once per URL. Nothing is inserted on the error path.
        match self.worker_to_id.entry(Arc::from(worker)) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                let id = self.next_worker_id.fetch_add(1, Ordering::Relaxed);
                let id = u32::try_from(id).map_err(|_| WorkerIdExhausted)?;
                entry.insert(id);
                Ok(id)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal: query helpers
    // -----------------------------------------------------------------------

    /// Get workers at a position matching content_hash (and prefix_hash for Multi).
    /// Copies worker IDs into a Vec — used only once at position 0 to initialize `active`.
    /// Skips rolling hash computation for Single entries (unambiguous match).
    fn get_workers_lazy(
        index: &PosIndex,
        position: usize,
        content_hash: ContentHash,
        seq_hashes: &mut Vec<SequenceHash>,
        sequence: &[ContentHash],
        now: u32,
    ) -> Option<Vec<u32>> {
        let entry = index.get(&(position, content_hash))?;
        entry.value().touch(now);
        if let Some(workers) = entry.value().seq.workers_if_single() {
            return Some(workers.iter().copied().collect());
        }
        // Multi: need rolling hash to disambiguate
        Self::ensure_seq_hash_computed(seq_hashes, position, sequence);
        entry
            .value()
            .seq
            .get(seq_hashes[position])
            .map(|workers| workers.iter().copied().collect())
    }

    /// Count workers at a position matching the prefix_hash (no set materialization).
    /// Skips rolling hash computation for Single entries (unambiguous match).
    fn count_workers_at(
        index: &PosIndex,
        position: usize,
        content_hash: ContentHash,
        seq_hashes: &mut Vec<SequenceHash>,
        sequence: &[ContentHash],
        now: u32,
    ) -> usize {
        let Some(entry) = index.get(&(position, content_hash)) else {
            return 0;
        };
        entry.value().touch(now);
        if let Some(workers) = entry.value().seq.workers_if_single() {
            return workers.len();
        }
        // Multi: need rolling hash to disambiguate
        Self::ensure_seq_hash_computed(seq_hashes, position, sequence);
        entry
            .value()
            .seq
            .get(seq_hashes[position])
            .map(|workers| workers.len())
            .unwrap_or(0)
    }

    /// Scan positions sequentially, draining workers that stop matching.
    /// Accesses DashMap entries directly — no set cloning.
    /// Skips rolling hash computation for Single entries (unambiguous match).
    /// Uses retain guard: skips retain when workers.len() >= active.len()
    /// (all active workers are still present, no work to do).
    #[expect(clippy::too_many_arguments)]
    fn linear_scan_drain(
        index: &PosIndex,
        sequence: &[ContentHash],
        seq_hashes: &mut Vec<SequenceHash>,
        active: &mut Vec<u32>,
        internal_scores: &mut FxHashMap<u32, u32>,
        lo: usize,
        hi: usize,
        early_exit: bool,
        now: u32,
    ) {
        for (offset, &content_hash) in sequence[lo..hi].iter().enumerate() {
            if active.is_empty() {
                break;
            }
            let pos = lo + offset;

            let Some(entry) = index.get(&(pos, content_hash)) else {
                for &w in active.iter() {
                    internal_scores.insert(w, pos as u32);
                }
                active.clear();
                break;
            };
            entry.value().touch(now);

            // Fast path: Single entry — skip rolling hash, use workers directly.
            if let Some(workers) = entry.value().seq.workers_if_single() {
                // Retain guard: only retain when some workers
                // have dropped off. When workers.len() >= active.len(), all active
                // workers are still present — skip the O(active) iteration.
                if workers.len() < active.len() {
                    let mut i = 0;
                    while i < active.len() {
                        if workers.contains(&active[i]) {
                            i += 1;
                        } else {
                            internal_scores.insert(active[i], pos as u32);
                            active.swap_remove(i);
                        }
                    }
                }
                if early_exit && !active.is_empty() {
                    break;
                }
                continue;
            }

            // Multi: need rolling hash to disambiguate.
            Self::ensure_seq_hash_computed(seq_hashes, pos, sequence);
            let seq_hash = seq_hashes[pos];

            let Some(workers) = entry.value().seq.get(seq_hash) else {
                for &w in active.iter() {
                    internal_scores.insert(w, pos as u32);
                }
                active.clear();
                break;
            };

            // Retain guard: only iterate when some workers dropped off.
            if workers.len() < active.len() {
                let mut i = 0;
                while i < active.len() {
                    if workers.contains(&active[i]) {
                        i += 1;
                    } else {
                        internal_scores.insert(active[i], pos as u32);
                        active.swap_remove(i);
                    }
                }
            }

            if early_exit && !active.is_empty() {
                break;
            }
        }
    }

    fn jump_search_matches(
        &self,
        content_hashes: &[ContentHash],
        early_exit: bool,
    ) -> OverlapScores {
        let mut scores = OverlapScores::default();

        if content_hashes.is_empty() {
            return scores;
        }

        let mut seq_hashes = Vec::with_capacity(content_hashes.len());
        let now = self.now_secs();

        let Some(initial_workers) = Self::get_workers_lazy(
            &self.index,
            0,
            content_hashes[0],
            &mut seq_hashes,
            content_hashes,
            now,
        ) else {
            return scores;
        };

        let mut active = initial_workers;
        if active.is_empty() {
            return scores;
        }

        let len = content_hashes.len();
        let mut internal_scores: FxHashMap<u32, u32> = FxHashMap::default();

        // Early exit: just record that workers matched at position 0.
        if early_exit {
            for &w in &active {
                internal_scores.insert(w, 1);
            }
            scores.scores = internal_scores;
            for &int_id in scores.scores.keys() {
                scores
                    .tree_sizes
                    .insert(int_id, self.tree_sizes.load(int_id));
            }
            return scores;
        }

        let mut current_pos = 0;

        while current_pos < len - 1 && !active.is_empty() {
            let next_pos = (current_pos + self.jump_size).min(len - 1);

            let count = Self::count_workers_at(
                &self.index,
                next_pos,
                content_hashes[next_pos],
                &mut seq_hashes,
                content_hashes,
                now,
            );

            // If the worker count at the jump destination matches the active set size,
            // all active workers are still present — safe to skip intermediate positions.
            if count == active.len() {
                current_pos = next_pos;
            } else {
                Self::linear_scan_drain(
                    &self.index,
                    content_hashes,
                    &mut seq_hashes,
                    &mut active,
                    &mut internal_scores,
                    current_pos + 1,
                    next_pos + 1,
                    false,
                    now,
                );
                current_pos = next_pos;
            }
        }

        let final_score = len as u32;
        for &w in &active {
            internal_scores.insert(w, final_score);
        }

        scores.scores = internal_scores;

        // Populate tree_sizes from atomic counters — lock-free array index.
        for &int_id in scores.scores.keys() {
            scores
                .tree_sizes
                .insert(int_id, self.tree_sizes.load(int_id));
        }

        scores
    }
}

impl Default for PositionalIndexer {
    fn default() -> Self {
        Self::new(32)
    }
}

impl fmt::Debug for PositionalIndexer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PositionalIndexer")
            .field("entries", &self.index.len())
            .field("jump_size", &self.jump_size)
            .field("workers", &self.next_worker_id.load(Ordering::Relaxed))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a sequence of StoredBlocks with distinct seq_hashes and content_hashes.
    fn make_blocks(content_hashes: &[u64]) -> Vec<StoredBlock> {
        // Generate seq_hashes as rolling hashes of content
        let mut blocks = Vec::new();
        let mut prev_seq: u64 = 0;
        for (i, &ch) in content_hashes.iter().enumerate() {
            let seq = if i == 0 {
                ch
            } else {
                PositionalIndexer::compute_next_seq_hash(prev_seq, ch)
            };
            prev_seq = seq;
            blocks.push(StoredBlock {
                seq_hash: SequenceHash(seq),
                content_hash: ContentHash(ch),
            });
        }
        blocks
    }

    /// Helper: create ContentHash sequence for find_matches.
    fn hashes(values: &[u64]) -> Vec<ContentHash> {
        values.iter().map(|&v| ContentHash(v)).collect()
    }

    #[test]
    fn test_new_indexer_is_empty() {
        let indexer = PositionalIndexer::default();
        let scores = indexer.find_matches(&hashes(&[1, 2, 3]), false);
        assert!(scores.scores.is_empty());
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_store_and_find_single_worker() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(scores.scores.get(&w1), Some(&3));
        assert_eq!(scores.tree_sizes.get(&w1), Some(&3));
    }

    #[test]
    fn test_store_partial_prefix_match() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        // Request has longer sequence — only first 3 match
        let scores = indexer.find_matches(&hashes(&[10, 20, 30, 40, 50]), false);
        assert_eq!(scores.scores.get(&w1), Some(&3));
    }

    #[test]
    fn test_store_no_match() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        let scores = indexer.find_matches(&hashes(&[99, 88, 77]), false);
        assert!(scores.scores.is_empty());
    }

    #[test]
    fn test_two_workers_different_depths() {
        let indexer = PositionalIndexer::new(64);
        let blocks_w1 = make_blocks(&[10, 20, 30]);
        let blocks_w2 = make_blocks(&[10, 20]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &blocks_w1, None, &mut wb1)
            .unwrap();
        indexer
            .apply_stored(w2, &blocks_w2, None, &mut wb2)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[10, 20, 30, 40]), false);
        assert_eq!(scores.scores.get(&w1), Some(&3));
        assert_eq!(scores.scores.get(&w2), Some(&2));
    }

    #[test]
    fn test_remove_blocks() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        let seq_hash_of_30 = blocks[2].seq_hash;
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        indexer.apply_removed(w1, &[seq_hash_of_30], &mut wb1);

        // After removing block at position 2, w1 should only match 2 blocks
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(scores.scores.get(&w1), Some(&2));
        assert_eq!(scores.tree_sizes.get(&w1), Some(&2));
    }

    #[test]
    fn test_clear_worker() {
        let indexer = PositionalIndexer::new(64);
        let blocks_w1 = make_blocks(&[10, 20, 30]);
        let blocks_w2 = make_blocks(&[10, 20]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &blocks_w1, None, &mut wb1)
            .unwrap();
        indexer
            .apply_stored(w2, &blocks_w2, None, &mut wb2)
            .unwrap();

        indexer.apply_cleared(w1, &mut wb1);

        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert!(!scores.scores.contains_key(&w1));
        assert_eq!(scores.scores.get(&w2), Some(&2));
    }

    #[test]
    fn test_tree_sizes() {
        let indexer = PositionalIndexer::new(64);
        let blocks_w1 = make_blocks(&[10, 20, 30]);
        let blocks_w2 = make_blocks(&[10, 20]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &blocks_w1, None, &mut wb1)
            .unwrap();
        indexer
            .apply_stored(w2, &blocks_w2, None, &mut wb2)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[10]), false);
        assert_eq!(scores.tree_sizes.get(&w1), Some(&3));
        assert_eq!(scores.tree_sizes.get(&w2), Some(&2));
    }

    #[test]
    fn test_store_with_parent_hash() {
        let indexer = PositionalIndexer::new(64);
        // First store: blocks at positions 0, 1
        let blocks1 = make_blocks(&[10, 20]);
        let parent_seq_hash = blocks1[1].seq_hash;
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks1, None, &mut wb1).unwrap();

        // Second store: blocks at positions 2, 3 (extending from parent)
        let blocks2 = vec![
            StoredBlock {
                seq_hash: SequenceHash(300),
                content_hash: ContentHash(30),
            },
            StoredBlock {
                seq_hash: SequenceHash(400),
                content_hash: ContentHash(40),
            },
        ];
        indexer
            .apply_stored(w1, &blocks2, Some(parent_seq_hash), &mut wb1)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[10, 20, 30, 40]), false);
        assert_eq!(scores.scores.get(&w1), Some(&4));
        assert_eq!(scores.tree_sizes.get(&w1), Some(&4));
    }

    #[test]
    fn test_store_with_parent_error_worker_not_tracked() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let result = indexer.apply_stored(w1, &blocks, Some(SequenceHash(999)), &mut wb1);
        assert!(matches!(result, Err(ApplyError::WorkerNotTracked)));
    }

    #[test]
    fn test_store_with_parent_error_parent_not_found() {
        let indexer = PositionalIndexer::new(64);
        let blocks1 = make_blocks(&[10, 20]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks1, None, &mut wb1).unwrap();

        let blocks2 = make_blocks(&[30]);
        let result = indexer.apply_stored(w1, &blocks2, Some(SequenceHash(999_999)), &mut wb1);
        assert!(matches!(result, Err(ApplyError::ParentBlockNotFound)));
    }

    #[test]
    fn test_remove_missing_block_is_noop() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        indexer.apply_removed(w1, &[SequenceHash(999)], &mut wb1);
        assert_eq!(indexer.current_size(), 3);
    }

    #[test]
    fn test_remove_unknown_worker_is_noop() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://unknown:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_removed(w1, &[SequenceHash(1)], &mut wb1);
    }

    #[test]
    fn test_remove_worker() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        indexer.remove_worker(w1, wb1);

        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert!(scores.scores.is_empty());
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_remove_worker_uses_reverse_map_and_preserves_unrelated_entries() {
        let indexer = PositionalIndexer::new(64);
        let removed = indexer.intern_worker("http://removed:8000").unwrap();
        let survivor = indexer.intern_worker("http://survivor:8000").unwrap();
        let mut removed_blocks = WorkerBlockMap::default();
        let mut survivor_blocks = WorkerBlockMap::default();

        indexer
            .apply_stored(removed, &make_blocks(&[1]), None, &mut removed_blocks)
            .unwrap();
        for value in 10_000..11_000 {
            indexer
                .apply_stored(survivor, &make_blocks(&[value]), None, &mut survivor_blocks)
                .unwrap();
        }

        indexer.remove_worker(removed, removed_blocks);

        assert_eq!(indexer.current_size(), survivor_blocks.len());
        let scores = indexer.find_matches(&hashes(&[10_999]), false);
        assert_eq!(scores.scores.get(&survivor), Some(&1));
        assert!(!scores.scores.contains_key(&removed));
    }

    #[test]
    fn test_multiple_workers_same_position() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let w3 = indexer.intern_worker("http://w3:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        let mut wb3 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &make_blocks(&[10]), None, &mut wb1)
            .unwrap();
        indexer
            .apply_stored(w2, &make_blocks(&[10]), None, &mut wb2)
            .unwrap();
        indexer
            .apply_stored(w3, &make_blocks(&[10]), None, &mut wb3)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[10]), false);
        assert_eq!(scores.scores.get(&w1), Some(&1));
        assert_eq!(scores.scores.get(&w2), Some(&1));
        assert_eq!(scores.scores.get(&w3), Some(&1));
    }

    #[test]
    fn test_empty_blocks_is_noop() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &[], None, &mut wb1).unwrap();
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_single_block_sequence() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[42]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        let scores = indexer.find_matches(&hashes(&[42]), false);
        assert_eq!(scores.scores.get(&w1), Some(&1));
    }

    #[test]
    fn test_request_content_hash_chunking() {
        let hashes = compute_request_content_hashes(&[1, 2, 3, 4, 5, 6, 7, 8], 4);
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0], compute_content_hash(&[1, 2, 3, 4]));
        assert_eq!(hashes[1], compute_content_hash(&[5, 6, 7, 8]));
    }

    #[test]
    fn test_request_content_hash_zero_block_size() {
        let hashes = compute_request_content_hashes(&[1, 2, 3], 0);
        assert!(hashes.is_empty());
    }

    // -----------------------------------------------------------------------
    // Jump search edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_jump_search_long_prefix() {
        let indexer = PositionalIndexer::new(4); // small jump_size to exercise jump logic
        let values: Vec<u64> = (1..=20).collect();
        let blocks = make_blocks(&values);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        let scores = indexer.find_matches(&hashes(&values), false);
        assert_eq!(scores.scores.get(&w1), Some(&20));
    }

    #[test]
    fn test_jump_search_worker_drains_mid_jump() {
        let indexer = PositionalIndexer::new(4);
        // w1 has 10 blocks, w2 has 6
        let values_w1: Vec<u64> = (1..=10).collect();
        let values_w2: Vec<u64> = (1..=6).collect();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &make_blocks(&values_w1), None, &mut wb1)
            .unwrap();
        indexer
            .apply_stored(w2, &make_blocks(&values_w2), None, &mut wb2)
            .unwrap();

        let query: Vec<u64> = (1..=10).collect();
        let scores = indexer.find_matches(&hashes(&query), false);
        assert_eq!(scores.scores.get(&w1), Some(&10));
        assert_eq!(scores.scores.get(&w2), Some(&6));
    }

    #[test]
    fn test_jump_search_multiple_drains() {
        let indexer = PositionalIndexer::new(3);
        // w1: 12, w2: 7, w3: 4
        let v1: Vec<u64> = (1..=12).collect();
        let v2: Vec<u64> = (1..=7).collect();
        let v3: Vec<u64> = (1..=4).collect();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let w3 = indexer.intern_worker("http://w3:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        let mut wb3 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &make_blocks(&v1), None, &mut wb1)
            .unwrap();
        indexer
            .apply_stored(w2, &make_blocks(&v2), None, &mut wb2)
            .unwrap();
        indexer
            .apply_stored(w3, &make_blocks(&v3), None, &mut wb3)
            .unwrap();

        let query: Vec<u64> = (1..=12).collect();
        let scores = indexer.find_matches(&hashes(&query), false);
        assert_eq!(scores.scores.get(&w1), Some(&12));
        assert_eq!(scores.scores.get(&w2), Some(&7));
        assert_eq!(scores.scores.get(&w3), Some(&4));
    }

    #[test]
    fn test_concurrent_store_and_match() {
        use std::{sync::Arc, thread};

        let indexer = Arc::new(PositionalIndexer::new(64));
        let indexer_writer = Arc::clone(&indexer);

        let writer = thread::spawn(move || {
            for i in 0..100u64 {
                let blocks = make_blocks(&[i * 10, i * 10 + 1, i * 10 + 2]);
                let wid = indexer_writer
                    .intern_worker(&format!("http://w{i}:8000"))
                    .unwrap();
                let mut wb = WorkerBlockMap::default();
                let _ = indexer_writer.apply_stored(wid, &blocks, None, &mut wb);
            }
        });

        let reader = thread::spawn({
            let indexer = Arc::clone(&indexer);
            move || {
                for _ in 0..1000 {
                    let _ = indexer.find_matches(&hashes(&[0, 1, 2, 3, 4]), false);
                }
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();
    }

    #[test]
    fn test_seq_entry_single_to_multi_upgrade() {
        let indexer = PositionalIndexer::new(64);

        // Two workers with same content hashes but different rolling prefixes
        // Worker 1: blocks at position 0 with content_hash=10
        let blocks_w1 = vec![StoredBlock {
            seq_hash: SequenceHash(100),
            content_hash: ContentHash(10),
        }];
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &blocks_w1, None, &mut wb1)
            .unwrap();

        // Worker 2: same content_hash but different seq_hash
        // Both start at position 0, so prefix_hash == content_hash.0 for both
        // This means they share the same prefix_hash → Single entry, both workers in set
        let blocks_w2 = vec![StoredBlock {
            seq_hash: SequenceHash(200),
            content_hash: ContentHash(10),
        }];
        indexer
            .apply_stored(w2, &blocks_w2, None, &mut wb2)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[10]), false);
        assert_eq!(scores.scores.get(&w1), Some(&1));
        assert_eq!(scores.scores.get(&w2), Some(&1));
    }

    #[test]
    fn test_seq_entry_distinct_prefix_same_content() {
        let indexer = PositionalIndexer::new(64);

        // Worker 1: position 0 = content 10, position 1 = content 99
        // Prefix at pos 1 = XXH3(10 || 99)
        let blocks_w1 = make_blocks(&[10, 99]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &blocks_w1, None, &mut wb1)
            .unwrap();

        // Worker 2: position 0 = content 20, position 1 = content 99
        // Prefix at pos 1 = XXH3(20 || 99) ← different because position 0 differs
        let blocks_w2 = make_blocks(&[20, 99]);
        indexer
            .apply_stored(w2, &blocks_w2, None, &mut wb2)
            .unwrap();

        // Query [10, 99] should only match w1
        let scores = indexer.find_matches(&hashes(&[10, 99]), false);
        assert_eq!(scores.scores.get(&w1), Some(&2));
        // w2 has a different prefix at position 0, so it won't be in initial active set

        // Query [20, 99] should only match w2
        let scores = indexer.find_matches(&hashes(&[20, 99]), false);
        assert_eq!(scores.scores.get(&w2), Some(&2));
    }

    // -----------------------------------------------------------------------
    // early_exit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_early_exit_returns_score_one() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), true);
        // early_exit: score is 1 (matched at position 0), not full depth
        assert_eq!(scores.scores.get(&w1), Some(&1));
        // tree_sizes still populated
        assert_eq!(scores.tree_sizes.get(&w1), Some(&3));
    }

    #[test]
    fn test_early_exit_no_match() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        let scores = indexer.find_matches(&hashes(&[99, 88]), true);
        assert!(scores.scores.is_empty());
    }

    // -----------------------------------------------------------------------
    // worker_id tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_worker_id_unknown() {
        let indexer = PositionalIndexer::default();
        assert!(indexer.worker_id("http://unknown:8000").is_none());
    }

    #[test]
    fn test_worker_id_after_store() {
        let indexer = PositionalIndexer::default();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &make_blocks(&[10]), None, &mut wb1)
            .unwrap();
        assert!(indexer.worker_id("http://w1:8000").is_some());
    }

    // -----------------------------------------------------------------------
    // Atomic tree_sizes consistency
    // -----------------------------------------------------------------------

    #[test]
    fn test_tree_sizes_after_store_and_remove() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30, 40, 50]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        assert_eq!(indexer.current_size(), 5);

        // Remove 2 blocks
        indexer.apply_removed(w1, &[blocks[3].seq_hash, blocks[4].seq_hash], &mut wb1);
        assert_eq!(indexer.current_size(), 3);

        // Verify tree_sizes in query results
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(scores.tree_sizes.get(&w1), Some(&3));
    }

    #[test]
    fn test_duplicate_store_does_not_inflate_tree_size() {
        let indexer = PositionalIndexer::new(64);
        let blocks = make_blocks(&[10, 20, 30]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();

        // First store: 3 new blocks
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(scores.tree_sizes.get(&w1), Some(&3));

        // Replay the same store event — tree_size must not change
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(
            scores.tree_sizes.get(&w1),
            Some(&3),
            "Duplicate store event must not inflate tree_size"
        );

        // Overlap scores should also be unchanged
        assert_eq!(scores.scores.get(&w1), Some(&3));
    }

    #[test]
    fn test_remove_worker_nonexistent_is_noop() {
        let indexer = PositionalIndexer::default();
        let w = indexer.intern_worker("http://ghost:8000").unwrap();
        indexer.remove_worker(w, WorkerBlockMap::default()); // no-op, no panic
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_concurrent_read_write() {
        let indexer = Arc::new(PositionalIndexer::new(4));
        let content: Vec<u64> = (1..=20).collect();
        let blocks = make_blocks(&content);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        let mut handles = Vec::new();

        // Spawn readers
        for _ in 0..4 {
            let idx = Arc::clone(&indexer);
            let ch = hashes(&content);
            handles.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    let scores = idx.find_matches(&ch, false);
                    let w1 = idx.worker_id("http://w1:8000").unwrap();
                    assert!(scores.scores.contains_key(&w1));
                }
            }));
        }

        // Spawn writers (add new workers concurrently)
        for i in 0..4 {
            let idx = Arc::clone(&indexer);
            let worker_content: Vec<u64> = (1..=5).collect();
            handles.push(std::thread::spawn(move || {
                let worker = format!("http://writer{i}:8000");
                let wid = idx.intern_worker(&worker).unwrap();
                let mut wb = WorkerBlockMap::default();
                let blks = make_blocks(&worker_content);
                for _ in 0..50 {
                    idx.apply_stored(wid, &blks, None, &mut wb).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // w1 should still be matchable
        let scores = indexer.find_matches(&hashes(&content), false);
        assert_eq!(scores.scores.get(&w1), Some(&20));
    }

    #[test]
    fn test_dashmap_cleanup_no_memory_leak() {
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[10, 20, 30]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        indexer.apply_stored(w2, &blocks, None, &mut wb2).unwrap();

        assert!(!indexer.index.is_empty());

        indexer.remove_worker(w1, wb1);
        assert!(!indexer.index.is_empty());

        indexer.remove_worker(w2, wb2);
        assert_eq!(indexer.index.len(), 0);
    }

    #[test]
    fn test_compute_content_hash_empty_tokens() {
        let hash = compute_content_hash(&[]);
        let hash2 = compute_content_hash(&[]);
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_compute_content_hash_single_token() {
        let hash = compute_content_hash(&[42]);
        assert_ne!(hash, compute_content_hash(&[43]));
    }

    #[test]
    fn test_seq_hash_rolling_correctness() {
        let content = vec![10u64, 20, 30, 40, 50];
        let blocks = make_blocks(&content);
        let content_hashes = hashes(&content);

        let mut seq_hashes: Vec<SequenceHash> = Vec::new();
        PositionalIndexer::ensure_seq_hash_computed(&mut seq_hashes, 4, &content_hashes);

        for (i, block) in blocks.iter().enumerate() {
            assert_eq!(
                seq_hashes[i], block.seq_hash,
                "seq_hash mismatch at position {i}"
            );
        }
    }

    #[test]
    fn test_query_prefix_of_stored() {
        let indexer = PositionalIndexer::default();
        let blocks = make_blocks(&[10, 20, 30, 40, 50]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        let scores = indexer.find_matches(&hashes(&[10, 20]), false);
        assert_eq!(scores.scores.get(&w1), Some(&2));
        assert_eq!(scores.tree_sizes.get(&w1), Some(&5));
    }

    #[test]
    fn test_disjoint_workers_no_shared_prefix() {
        let indexer = PositionalIndexer::default();
        let blocks_w1 = make_blocks(&[10, 20, 30]);
        let blocks_w2 = make_blocks(&[99, 88, 77]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &blocks_w1, None, &mut wb1)
            .unwrap();
        indexer
            .apply_stored(w2, &blocks_w2, None, &mut wb2)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(scores.scores.get(&w1), Some(&3));
        assert!(!scores.scores.contains_key(&w2));

        let scores = indexer.find_matches(&hashes(&[99, 88, 77]), false);
        assert!(!scores.scores.contains_key(&w1));
        assert_eq!(scores.scores.get(&w2), Some(&3));
    }

    #[test]
    #[should_panic(expected = "jump_size must be greater than 0")]
    fn test_zero_jump_size_panics() {
        let _ = PositionalIndexer::new(0);
    }

    #[test]
    fn test_current_size_across_operations() {
        let indexer = PositionalIndexer::default();
        assert_eq!(indexer.current_size(), 0);

        let blocks = make_blocks(&[10, 20, 30]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        assert_eq!(indexer.current_size(), 3);

        indexer.apply_stored(w2, &blocks, None, &mut wb2).unwrap();
        assert_eq!(indexer.current_size(), 6);

        indexer.apply_removed(w1, &[blocks[2].seq_hash], &mut wb1);
        assert_eq!(indexer.current_size(), 5);

        indexer.apply_cleared(w2, &mut wb2);
        assert_eq!(indexer.current_size(), 2);

        indexer.remove_worker(w1, wb1);
        assert_eq!(indexer.current_size(), 0);
    }

    // -----------------------------------------------------------------------
    // compute_request_content_hashes tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_hashes_basic() {
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_request_content_hashes(&tokens, 4);
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0], compute_content_hash(&[1, 2, 3, 4]));
        assert_eq!(hashes[1], compute_content_hash(&[5, 6, 7, 8]));
    }

    #[test]
    fn test_request_hashes_partial_trailing_chunk_discarded() {
        let tokens: Vec<u32> = (1..=10).collect();
        let hashes = compute_request_content_hashes(&tokens, 4);
        assert_eq!(hashes.len(), 2);
    }

    #[test]
    fn test_request_hashes_fewer_than_block_size() {
        let hashes = compute_request_content_hashes(&[1, 2, 3], 4);
        assert!(hashes.is_empty());
    }

    #[test]
    fn test_request_hashes_empty_tokens() {
        let hashes = compute_request_content_hashes(&[], 16);
        assert!(hashes.is_empty());
    }

    #[test]
    fn test_request_hashes_exact_multiple() {
        let tokens: Vec<u32> = (1..=6).collect();
        let hashes = compute_request_content_hashes(&tokens, 2);
        assert_eq!(hashes.len(), 3);
    }

    #[test]
    fn test_request_hashes_zero_block_size_returns_empty() {
        let hashes = compute_request_content_hashes(&[1, 2, 3], 0);
        assert!(hashes.is_empty());
    }

    #[test]
    fn test_request_hashes_block_size_1() {
        let tokens = vec![10u32, 20, 30];
        let hashes = compute_request_content_hashes(&tokens, 1);
        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes[0], compute_content_hash(&[10]));
        assert_eq!(hashes[1], compute_content_hash(&[20]));
        assert_eq!(hashes[2], compute_content_hash(&[30]));
    }

    // -----------------------------------------------------------------------
    // End-to-end: store events → query with compute_request_content_hashes
    // -----------------------------------------------------------------------

    #[test]
    fn test_end_to_end_store_and_query() {
        let indexer = PositionalIndexer::default();
        let block_size = 4;
        let tokens: Vec<u32> = (1..=16).collect();

        let content_hashes: Vec<ContentHash> = tokens
            .chunks(block_size)
            .map(compute_content_hash)
            .collect();

        let blocks: Vec<StoredBlock> = content_hashes
            .iter()
            .enumerate()
            .map(|(i, &ch)| StoredBlock {
                seq_hash: SequenceHash(0xBEEF_0000 + i as u64),
                content_hash: ch,
            })
            .collect();

        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        let query_hashes = compute_request_content_hashes(&tokens, block_size);
        let scores = indexer.find_matches(&query_hashes, false);
        assert_eq!(scores.scores.get(&w1), Some(&4));
    }

    #[test]
    fn test_end_to_end_partial_overlap() {
        let indexer = PositionalIndexer::default();
        let block_size = 4;

        let cached_tokens: Vec<u32> = (1..=8).collect();
        let blocks: Vec<StoredBlock> = cached_tokens
            .chunks(block_size)
            .enumerate()
            .map(|(i, chunk)| StoredBlock {
                seq_hash: SequenceHash(i as u64 + 1),
                content_hash: compute_content_hash(chunk),
            })
            .collect();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        let query_tokens: Vec<u32> = (1..=16).collect();
        let query_hashes = compute_request_content_hashes(&query_tokens, block_size);
        let scores = indexer.find_matches(&query_hashes, false);
        assert_eq!(scores.scores.get(&w1), Some(&2));
        assert_eq!(scores.tree_sizes.get(&w1), Some(&2));
    }

    #[test]
    fn test_end_to_end_different_backends_same_content() {
        let indexer = PositionalIndexer::new(4);
        let block_size = 4;
        let tokens: Vec<u32> = (1..=8).collect();
        let content_hashes: Vec<ContentHash> = tokens
            .chunks(block_size)
            .map(compute_content_hash)
            .collect();

        let blocks_w1: Vec<StoredBlock> = content_hashes
            .iter()
            .enumerate()
            .map(|(i, &ch)| StoredBlock {
                seq_hash: SequenceHash(0xAAAA_0000 + i as u64),
                content_hash: ch,
            })
            .collect();

        let blocks_w2: Vec<StoredBlock> = content_hashes
            .iter()
            .enumerate()
            .map(|(i, &ch)| StoredBlock {
                seq_hash: SequenceHash(0xBBBB_0000 + i as u64),
                content_hash: ch,
            })
            .collect();

        let sglang = indexer.intern_worker("http://sglang:8000").unwrap();
        let vllm = indexer.intern_worker("http://vllm:8000").unwrap();
        let mut wb_sg = WorkerBlockMap::default();
        let mut wb_vl = WorkerBlockMap::default();
        indexer
            .apply_stored(sglang, &blocks_w1, None, &mut wb_sg)
            .unwrap();
        indexer
            .apply_stored(vllm, &blocks_w2, None, &mut wb_vl)
            .unwrap();

        let query_hashes = compute_request_content_hashes(&tokens, block_size);
        let scores = indexer.find_matches(&query_hashes, false);
        assert_eq!(scores.scores.get(&sglang), Some(&2));
        assert_eq!(scores.scores.get(&vllm), Some(&2));
    }

    // -----------------------------------------------------------------------
    // Jump boundary tests
    // -----------------------------------------------------------------------

    /// Helper: store a sequence for a worker via chained continuations of `chunk_size` blocks.
    fn store_via_continuations(
        indexer: &PositionalIndexer,
        worker: &str,
        content: &[u64],
        chunk_size: usize,
        worker_blocks: &mut WorkerBlockMap,
    ) {
        let worker_id = indexer.intern_worker(worker).unwrap();
        let all_blocks = make_blocks(content);
        let mut offset = 0;
        let mut parent: Option<SequenceHash> = None;
        while offset < all_blocks.len() {
            let end = (offset + chunk_size).min(all_blocks.len());
            let chunk = &all_blocks[offset..end];
            indexer
                .apply_stored(worker_id, chunk, parent, worker_blocks)
                .unwrap();
            parent = Some(chunk.last().unwrap().seq_hash);
            offset = end;
        }
    }

    #[test]
    fn test_divergence_at_jump_boundaries() {
        let indexer = PositionalIndexer::new(32);
        let full: Vec<u64> = (1..=128).collect();
        let full_blocks = make_blocks(&full);
        let full_id = indexer.intern_worker("http://full:8000").unwrap();
        let mut wb_full = WorkerBlockMap::default();
        indexer
            .apply_stored(full_id, &full_blocks, None, &mut wb_full)
            .unwrap();

        for &depth in &[31, 32, 33] {
            let partial_blocks = make_blocks(&full[..depth]);
            let worker = format!("http://depth{depth}:8000");
            let wid = indexer.intern_worker(&worker).unwrap();
            let mut wb = WorkerBlockMap::default();
            indexer
                .apply_stored(wid, &partial_blocks, None, &mut wb)
                .unwrap();
        }

        for &depth in &[63, 64, 65] {
            let partial_blocks = make_blocks(&full[..depth]);
            let worker = format!("http://depth{depth}:8000");
            let wid = indexer.intern_worker(&worker).unwrap();
            let mut wb = WorkerBlockMap::default();
            indexer
                .apply_stored(wid, &partial_blocks, None, &mut wb)
                .unwrap();
        }

        let scores = indexer.find_matches(&hashes(&full), false);
        assert_eq!(scores.scores.get(&full_id), Some(&128));
        for &depth in &[31u64, 32, 33, 63, 64, 65] {
            let worker = format!("http://depth{depth}:8000");
            let wid = indexer.worker_id(&worker).unwrap();
            assert_eq!(scores.scores.get(&wid), Some(&(depth as u32)));
        }
    }

    #[test]
    fn test_exact_jump_size_sequences() {
        let indexer = PositionalIndexer::new(32);

        for &len in &[32, 64, 96] {
            let content: Vec<u64> = (1..=len as u64).collect();
            let blocks = make_blocks(&content);
            let worker = format!("http://len{len}:8000");
            let wid = indexer.intern_worker(&worker).unwrap();
            let mut wb = WorkerBlockMap::default();
            indexer.apply_stored(wid, &blocks, None, &mut wb).unwrap();

            let scores = indexer.find_matches(&hashes(&content), false);
            assert_eq!(
                scores.scores.get(&wid),
                Some(&(len as u32)),
                "exact match failed for sequence length {len}"
            );
        }
    }

    #[test]
    fn test_off_by_one_jump_boundaries() {
        let indexer = PositionalIndexer::new(32);
        let full: Vec<u64> = (1..=128).collect();

        for &len in &[31, 33, 63, 65, 95, 97] {
            let content = &full[..len];
            let blocks = make_blocks(content);
            let worker = format!("http://len{len}:8000");
            let wid = indexer.intern_worker(&worker).unwrap();
            let mut wb = WorkerBlockMap::default();
            indexer.apply_stored(wid, &blocks, None, &mut wb).unwrap();

            let scores = indexer.find_matches(&hashes(content), false);
            assert_eq!(
                scores.scores.get(&wid),
                Some(&(len as u32)),
                "exact match failed for sequence length {len}"
            );
        }
    }

    #[test]
    fn test_staggered_workers_across_jump_boundaries() {
        let indexer = PositionalIndexer::new(32);
        let full: Vec<u64> = (1..=100).collect();

        let depths = [10, 20, 35, 64, 100];
        for &depth in &depths {
            let blocks = make_blocks(&full[..depth]);
            let worker = format!("http://w{depth}:8000");
            let wid = indexer.intern_worker(&worker).unwrap();
            let mut wb = WorkerBlockMap::default();
            indexer.apply_stored(wid, &blocks, None, &mut wb).unwrap();
        }

        let scores = indexer.find_matches(&hashes(&full), false);
        for &depth in &depths {
            let worker = format!("http://w{depth}:8000");
            let wid = indexer.worker_id(&worker).unwrap();
            assert_eq!(
                scores.scores.get(&wid),
                Some(&(depth as u32)),
                "worker at depth {depth} has wrong score"
            );
        }
    }

    #[test]
    fn test_shared_prefix_diverge_at_jump_boundary() {
        let indexer = PositionalIndexer::new(32);
        let shared: Vec<u64> = (1..=40).collect();

        let mut content_w1 = shared.clone();
        content_w1.extend(1001..=1060);
        let blocks_w1 = make_blocks(&content_w1);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let w3 = indexer.intern_worker("http://w3:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        let mut wb3 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &blocks_w1, None, &mut wb1)
            .unwrap();

        let mut content_w2 = shared.clone();
        content_w2.extend(2001..=2020);
        let blocks_w2 = make_blocks(&content_w2);
        indexer
            .apply_stored(w2, &blocks_w2, None, &mut wb2)
            .unwrap();

        let blocks_w3 = make_blocks(&shared);
        indexer
            .apply_stored(w3, &blocks_w3, None, &mut wb3)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&content_w1), false);
        assert_eq!(scores.scores.get(&w1), Some(&100));
        assert_eq!(scores.scores.get(&w2), Some(&40));
        assert_eq!(scores.scores.get(&w3), Some(&40));
    }

    #[test]
    fn test_very_long_sequence() {
        let indexer = PositionalIndexer::new(64);
        let content: Vec<u64> = (1..=1000).collect();
        let blocks = make_blocks(&content);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        let scores = indexer.find_matches(&hashes(&content), false);
        assert_eq!(scores.scores.get(&w1), Some(&1000));

        let scores = indexer.find_matches(&hashes(&content[..500]), false);
        assert_eq!(scores.scores.get(&w1), Some(&500));

        let mut divergent = content[..499].to_vec();
        divergent.push(999999);
        let scores = indexer.find_matches(&hashes(&divergent), false);
        assert_eq!(scores.scores.get(&w1), Some(&499));
    }

    // -----------------------------------------------------------------------
    // Deep continuation chain tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_deep_continuation_chain() {
        let indexer = PositionalIndexer::new(64);
        let content: Vec<u64> = (1..=200).collect();
        let mut wb1 = WorkerBlockMap::default();
        store_via_continuations(&indexer, "http://w1:8000", &content, 10, &mut wb1);

        assert_eq!(indexer.current_size(), 200);

        let w1 = indexer.worker_id("http://w1:8000").unwrap();
        let scores = indexer.find_matches(&hashes(&content), false);
        assert_eq!(scores.scores.get(&w1), Some(&200));

        let scores = indexer.find_matches(&hashes(&content[..150]), false);
        assert_eq!(scores.scores.get(&w1), Some(&150));
    }

    #[test]
    fn test_continuation_chain_with_multiple_workers() {
        let indexer = PositionalIndexer::new(32);
        let content: Vec<u64> = (1..=100).collect();

        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        store_via_continuations(&indexer, "http://w1:8000", &content, 10, &mut wb1);
        store_via_continuations(&indexer, "http://w2:8000", &content[..50], 10, &mut wb2);

        let w1 = indexer.worker_id("http://w1:8000").unwrap();
        let w2 = indexer.worker_id("http://w2:8000").unwrap();
        let scores = indexer.find_matches(&hashes(&content), false);
        assert_eq!(scores.scores.get(&w1), Some(&100));
        assert_eq!(scores.scores.get(&w2), Some(&50));
    }

    #[test]
    fn test_multiple_disjoint_sequences_per_worker() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();

        let blocks1 = make_blocks(&[10, 20, 30]);
        indexer.apply_stored(w1, &blocks1, None, &mut wb1).unwrap();

        let blocks2 = make_blocks(&[100, 200, 300, 400]);
        indexer.apply_stored(w1, &blocks2, None, &mut wb1).unwrap();

        let scores = indexer.find_matches(&hashes(&[100, 200, 300, 400]), false);
        assert_eq!(scores.scores.get(&w1), Some(&4));

        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(scores.scores.get(&w1), Some(&3));
    }

    // -----------------------------------------------------------------------
    // Long sequence partial removal and stale entry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_long_sequence_partial_removal() {
        let indexer = PositionalIndexer::new(32);
        let content: Vec<u64> = (1..=100).collect();
        let blocks = make_blocks(&content);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        let to_remove: Vec<SequenceHash> = blocks[80..].iter().map(|b| b.seq_hash).collect();
        indexer.apply_removed(w1, &to_remove, &mut wb1);

        assert_eq!(indexer.current_size(), 80);

        let scores = indexer.find_matches(&hashes(&content), false);
        assert_eq!(scores.scores.get(&w1), Some(&80));

        let scores = indexer.find_matches(&hashes(&content[..80]), false);
        assert_eq!(scores.scores.get(&w1), Some(&80));
    }

    #[test]
    fn test_remove_parent_does_not_cascade() {
        let indexer = PositionalIndexer::new(1);
        let blocks = make_blocks(&[10, 20, 30, 40, 50]);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        indexer.apply_removed(w1, &[blocks[1].seq_hash], &mut wb1);

        assert_eq!(indexer.current_size(), 4);

        let scores = indexer.find_matches(&hashes(&[10, 20, 30, 40, 50]), false);
        assert_eq!(scores.scores.get(&w1), Some(&1));
    }

    #[test]
    fn test_long_sequence_clear_and_rebuild() {
        let indexer = PositionalIndexer::new(32);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();

        let original: Vec<u64> = (1..=100).collect();
        let blocks = make_blocks(&original);
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();

        indexer.apply_cleared(w1, &mut wb1);
        assert_eq!(indexer.current_size(), 0);

        let replacement: Vec<u64> = (1001..=1100).collect();
        let new_blocks = make_blocks(&replacement);
        indexer
            .apply_stored(w1, &new_blocks, None, &mut wb1)
            .unwrap();

        let scores = indexer.find_matches(&hashes(&original), false);
        assert!(!scores.scores.contains_key(&w1));

        let scores = indexer.find_matches(&hashes(&replacement), false);
        assert_eq!(scores.scores.get(&w1), Some(&100));
    }

    #[test]
    fn test_interleaved_long_sequences() {
        let indexer = PositionalIndexer::new(32);
        let content: Vec<u64> = (1..=100).collect();

        let depths = [25, 50, 75, 100];
        for &depth in &depths {
            let blocks = make_blocks(&content[..depth]);
            let worker = format!("http://w{depth}:8000");
            let wid = indexer.intern_worker(&worker).unwrap();
            let mut wb = WorkerBlockMap::default();
            indexer.apply_stored(wid, &blocks, None, &mut wb).unwrap();
        }

        let scores = indexer.find_matches(&hashes(&content), false);
        for &depth in &depths {
            let worker = format!("http://w{depth}:8000");
            let wid = indexer.worker_id(&worker).unwrap();
            assert_eq!(
                scores.scores.get(&wid),
                Some(&(depth as u32)),
                "worker at depth {depth} has wrong score"
            );
            assert_eq!(
                scores.tree_sizes.get(&wid),
                Some(&depth),
                "worker at depth {depth} has wrong tree_size"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Worker interning: growth past 2048, id lifecycle, exhaustion
    // -----------------------------------------------------------------------

    #[test]
    fn test_tree_sizes_locate_covers_u32_id_space() {
        // First segment: ids 0..2048.
        assert_eq!(TreeSizes::locate(0), (0, 0));
        assert_eq!(TreeSizes::locate(2047), (0, 2047));
        // Doubling segments: 2048..6144, 6144..14336, ...
        assert_eq!(TreeSizes::locate(2048), (1, 0));
        assert_eq!(TreeSizes::locate(6143), (1, 4095));
        assert_eq!(TreeSizes::locate(6144), (2, 0));
        // The largest possible id maps inside the last segment.
        let (segment, offset) = TreeSizes::locate(u32::MAX);
        assert_eq!(segment, SEGMENT_COUNT - 1);
        assert!(offset < TreeSizes::segment_len(segment));
        // Segment starts are contiguous: each boundary id maps to offset 0.
        let mut start = 0u64;
        for segment in 0..SEGMENT_COUNT {
            if start > u32::MAX as u64 {
                break;
            }
            assert_eq!(TreeSizes::locate(start as u32), (segment, 0));
            start += TreeSizes::segment_len(segment) as u64;
        }
    }

    #[test]
    fn test_intern_worker_past_2048_workers() {
        let indexer = PositionalIndexer::new(64);
        // Ids must be dense and uncapped — 2049+ used to panic.
        let ids: Vec<u32> = (0..5000u32)
            .map(|w| indexer.intern_worker(&format!("http://w{w}:8000")).unwrap())
            .collect();
        for (expected, &id) in ids.iter().enumerate() {
            assert_eq!(id, expected as u32);
        }

        // Shared 3-block prefix plus a distinct tail per worker, for workers on
        // both sides of the old 2048 cap (2048 is also a segment boundary).
        let shared: Vec<u64> = vec![10, 20, 30];
        let probe_ids = [0u32, 1, 2047, 2048, 2049, 4999];
        for &wid in &probe_ids {
            let mut content = shared.clone();
            content.push(1_000_000 + wid as u64);
            let blocks = make_blocks(&content);
            let mut wb = WorkerBlockMap::default();
            indexer.apply_stored(wid, &blocks, None, &mut wb).unwrap();
        }

        // All probed workers share the 3-block prefix.
        let scores = indexer.find_matches(&hashes(&shared), false);
        for &wid in &probe_ids {
            assert_eq!(scores.scores.get(&wid), Some(&3), "worker {wid} score");
            assert_eq!(
                scores.tree_sizes.get(&wid),
                Some(&4),
                "worker {wid} tree_size"
            );
        }

        // Only worker 2049 has the tail block — the rest drain at depth 3.
        let mut full = shared.clone();
        full.push(1_000_000 + 2049);
        let scores = indexer.find_matches(&hashes(&full), false);
        assert_eq!(scores.scores.get(&2049), Some(&4));
        assert_eq!(scores.scores.get(&2048), Some(&3));

        assert_eq!(indexer.current_size(), 4 * probe_ids.len());
    }

    #[test]
    fn test_worker_ids_not_recycled_after_removal() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        assert_eq!((w1, w2), (0, 1));

        let blocks = make_blocks(&[10, 20, 30]);
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        indexer.remove_worker(w1, wb1);

        // The removed worker's URL keeps its id; it is not freed for reuse.
        assert_eq!(indexer.intern_worker("http://w1:8000").unwrap(), w1);
        // New URLs continue monotonically — removal never recycles ids.
        assert_eq!(indexer.intern_worker("http://w3:8000").unwrap(), 2);

        // The removed worker no longer matches; re-storing under its id works.
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert!(scores.scores.is_empty());
        let mut wb1 = WorkerBlockMap::default();
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(scores.scores.get(&w1), Some(&3));
        assert_eq!(indexer.current_size(), 3);
    }

    #[test]
    fn test_remove_worker_in_unallocated_segment_is_noop() {
        let indexer = PositionalIndexer::new(64);
        // High ids whose tree_sizes segments were never written: removal and
        // clear must not panic (and must not allocate the segment).
        indexer.remove_worker(123_456, WorkerBlockMap::default());
        indexer.apply_cleared(654_321, &mut WorkerBlockMap::default());
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn test_intern_worker_exhaustion_returns_error() {
        let indexer = PositionalIndexer::new(64);
        // Jump the counter to the last valid id. Interning never touches
        // tree_sizes, so no segment is allocated for this id.
        indexer
            .next_worker_id
            .store(u32::MAX as u64, Ordering::Relaxed);
        assert_eq!(indexer.intern_worker("http://last:8000").unwrap(), u32::MAX);
        // The id space is now exhausted: new URLs error, known URLs still resolve.
        assert_eq!(
            indexer.intern_worker("http://one-too-many:8000"),
            Err(WorkerIdExhausted)
        );
        assert_eq!(indexer.intern_worker("http://last:8000").unwrap(), u32::MAX);
        assert_eq!(indexer.worker_id("http://last:8000"), Some(u32::MAX));
    }

    #[test]
    fn test_concurrent_interning_across_segment_boundary() {
        let indexer = Arc::new(PositionalIndexer::new(64));
        // Pre-assign ids up to just below the 2048 segment boundary, so the
        // concurrent writers race across it.
        for w in 0..2040 {
            indexer
                .intern_worker(&format!("http://pre{w}:8000"))
                .unwrap();
        }

        let mut handles = Vec::new();
        for t in 0..4u32 {
            let idx = Arc::clone(&indexer);
            handles.push(std::thread::spawn(move || {
                for i in 0..8u32 {
                    let wid = idx.intern_worker(&format!("http://t{t}-{i}:8000")).unwrap();
                    let mut wb = WorkerBlockMap::default();
                    let blocks = make_blocks(&[wid as u64 * 100 + 1, wid as u64 * 100 + 2]);
                    idx.apply_stored(wid, &blocks, None, &mut wb).unwrap();
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        // 4 threads x 8 workers x 2 blocks each, ids 2040..2072 straddling the
        // segment boundary; every worker's blocks must be matchable.
        assert_eq!(indexer.current_size(), 64);
        for t in 0..4u32 {
            for i in 0..8u32 {
                let wid = indexer.worker_id(&format!("http://t{t}-{i}:8000")).unwrap();
                let scores = indexer.find_matches(
                    &hashes(&[wid as u64 * 100 + 1, wid as u64 * 100 + 2]),
                    false,
                );
                assert_eq!(scores.scores.get(&wid), Some(&2));
            }
        }
    }

    // -----------------------------------------------------------------------
    // prune: TTL + capacity bounding
    // -----------------------------------------------------------------------

    #[test]
    fn test_prune_disabled_is_noop() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &make_blocks(&[10, 20, 30]), None, &mut wb1)
            .unwrap();
        indexer.set_test_now(1_000_000);

        for (ttl, max) in [(None, None), (Some(0), Some(0)), (None, Some(0))] {
            let stats = indexer.prune(ttl, max);
            assert_eq!(stats.evicted_ttl, 0);
            assert_eq!(stats.evicted_capacity, 0);
            assert_eq!(stats.remaining, 3);
        }
        assert_eq!(indexer.current_size(), 3);
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(scores.scores.get(&w1), Some(&3));
    }

    #[test]
    fn test_prune_ttl_evicts_stale_keeps_recent() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();

        indexer.set_test_now(0);
        indexer
            .apply_stored(w1, &make_blocks(&[10, 20, 30]), None, &mut wb1)
            .unwrap();
        indexer.set_test_now(500);
        indexer
            .apply_stored(w2, &make_blocks(&[40, 50]), None, &mut wb2)
            .unwrap();

        indexer.set_test_now(600);
        let stats = indexer.prune(Some(200), None);
        assert_eq!(stats.evicted_ttl, 3);
        assert_eq!(stats.evicted_capacity, 0);
        assert_eq!(stats.remaining, 2);

        assert!(indexer
            .find_matches(&hashes(&[10, 20, 30]), false)
            .scores
            .is_empty());
        let scores = indexer.find_matches(&hashes(&[40, 50]), false);
        assert_eq!(scores.scores.get(&w2), Some(&2));
        assert_eq!(indexer.current_size(), 2);
    }

    #[test]
    fn test_prune_ttl_query_touch_keeps_hot_entries() {
        // jump_size 1 so a full query touches every position of the chain.
        let indexer = PositionalIndexer::new(1);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();

        indexer.set_test_now(0);
        indexer
            .apply_stored(w1, &make_blocks(&[10, 20, 30]), None, &mut wb1)
            .unwrap();
        indexer
            .apply_stored(w2, &make_blocks(&[40, 50, 60]), None, &mut wb2)
            .unwrap();

        // Only w1's chain is queried (touched) after storing.
        indexer.set_test_now(500);
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(scores.scores.get(&w1), Some(&3));

        indexer.set_test_now(600);
        let stats = indexer.prune(Some(200), None);
        assert_eq!(stats.evicted_ttl, 3); // w2's untouched chain

        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(scores.scores.get(&w1), Some(&3));
        assert!(indexer
            .find_matches(&hashes(&[40, 50, 60]), false)
            .scores
            .is_empty());
    }

    #[test]
    fn test_prune_capacity_evicts_oldest_first() {
        let indexer = PositionalIndexer::new(1);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();

        // Ten independent single-block chains stored at increasing times.
        for i in 0u64..10 {
            indexer.set_test_now(i as u32);
            indexer
                .apply_stored(w1, &make_blocks(&[100 + i]), None, &mut wb1)
                .unwrap();
        }
        assert_eq!(indexer.entry_count(), 10);

        indexer.set_test_now(100);
        // Ceiling 5 → low-water 5 (5/10 == 0) → evict the 5 oldest.
        let stats = indexer.prune(None, Some(5));
        assert_eq!(stats.evicted_ttl, 0);
        assert_eq!(stats.evicted_capacity, 5);
        assert_eq!(stats.remaining, 5);
        assert_eq!(indexer.current_size(), 5);

        for i in 0u64..10 {
            let found = !indexer
                .find_matches(&hashes(&[100 + i]), false)
                .scores
                .is_empty();
            assert_eq!(found, i >= 5, "chain {i} presence");
        }
    }

    #[test]
    fn test_prune_capacity_spares_fresh_entries() {
        // Entries stamped within the freshness grace are never capacity
        // candidates — a prune must not race a store batch whose deferred
        // tree_sizes increment hasn't landed yet.
        let indexer = PositionalIndexer::new(1);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();

        indexer.set_test_now(100);
        for i in 0u64..10 {
            indexer
                .apply_stored(w1, &make_blocks(&[100 + i]), None, &mut wb1)
                .unwrap();
        }

        // All entries are at now == stamp → inside the grace → spared.
        let stats = indexer.prune(None, Some(5));
        assert_eq!(stats.evicted_capacity, 0);
        assert_eq!(indexer.entry_count(), 10);

        // Once they age past the grace, the ceiling is enforced.
        indexer.set_test_now(100 + CAPACITY_EVICTION_GRACE_SECS + 1);
        let stats = indexer.prune(None, Some(5));
        assert_eq!(stats.evicted_capacity, 5);
        assert_eq!(indexer.entry_count(), 5);
    }

    #[test]
    fn test_restore_after_prune_restores_counts() {
        // Re-storing blocks whose entries were pruned (with the stale reverse
        // mapping still present) must restore both the index entries and the
        // per-worker counts — counting is keyed on index memberships, not on
        // reverse-map novelty.
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let blocks = make_blocks(&[10, 20, 30]);

        indexer.set_test_now(0);
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        indexer.set_test_now(1000);
        indexer.prune(Some(100), None);
        assert_eq!(indexer.current_size(), 0);
        assert_eq!(wb1.len(), 3); // reverse map untouched by prune

        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        assert_eq!(indexer.current_size(), 3);
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(scores.scores.get(&w1), Some(&3));

        // And duplicate stores of live memberships still don't inflate.
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        assert_eq!(indexer.current_size(), 3);
    }

    #[test]
    fn test_prune_capacity_noop_under_ceiling() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        indexer
            .apply_stored(w1, &make_blocks(&[10, 20, 30]), None, &mut wb1)
            .unwrap();

        let stats = indexer.prune(None, Some(10));
        assert_eq!(stats.evicted_capacity, 0);
        assert_eq!(stats.remaining, 3);
    }

    #[test]
    fn test_prune_then_stale_removal_no_double_decrement() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let blocks = make_blocks(&[10, 20, 30]);

        indexer.set_test_now(0);
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        indexer.set_test_now(1000);
        let stats = indexer.prune(Some(100), None);
        assert_eq!(stats.evicted_ttl, 3);
        assert_eq!(indexer.current_size(), 0);

        // The reverse map still holds the pruned blocks; their removal events
        // must be a counting no-op (a wrap would make current_size huge).
        let seq_hashes: Vec<SequenceHash> = blocks.iter().map(|b| b.seq_hash).collect();
        indexer.apply_removed(w1, &seq_hashes, &mut wb1);
        assert!(wb1.is_empty());
        assert_eq!(indexer.current_size(), 0);

        // The worker can start a fresh chain afterwards.
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        assert_eq!(indexer.current_size(), 3);
        let scores = indexer.find_matches(&hashes(&[10, 20, 30]), false);
        assert_eq!(scores.scores.get(&w1), Some(&3));
    }

    #[test]
    fn test_prune_multi_worker_entry_decrements_each() {
        let indexer = PositionalIndexer::new(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        let blocks = make_blocks(&[10, 20, 30]);

        indexer.set_test_now(0);
        indexer.apply_stored(w1, &blocks, None, &mut wb1).unwrap();
        indexer.apply_stored(w2, &blocks, None, &mut wb2).unwrap();
        assert_eq!(indexer.current_size(), 6);
        assert_eq!(indexer.entry_count(), 3); // shared entries

        indexer.set_test_now(1000);
        let stats = indexer.prune(Some(100), None);
        assert_eq!(stats.evicted_ttl, 3);
        assert_eq!(indexer.current_size(), 0);
        assert_eq!(indexer.entry_count(), 0);
    }
}
