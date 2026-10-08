//! The chain index split into shards: each shard is a complete [`ChainIndex`] holding the workers
//! assigned to it, a lookup walks every shard and unions the scores.
//!
//! Why: with lanes on both sockets of a host, a single index has every run's version, length,
//! coverage words and child table, the arena's bump pointer and free lists, and the run slab
//! written from both sockets; the replay's duplicated corpus turns that into true sharing, and
//! the lane CPU per event doubles to triples against one socket. One shard per socket, each
//! written only by the lanes of that socket, removes every such line: no run state is shared
//! between shards, and the lookup, which writes nothing, reads them all.
//!
//! Exactness does not depend on the assignment: a shard is exact for its own workers whatever
//! the caller's affinity, the worker sets of the shards are disjoint, so the union of their
//! scores is the single index's answer. A caller with no control over which thread applies
//! which worker (the gateway's runtime workers on both sockets) is exact and merely pays the
//! sharing it did not avoid. `shards = 1` is the chain index itself behind one indirection:
//! the same ids, the same lookups, the same memory.
//!
//! Worker ids carry the shard: `shard << SHARD_SHIFT | local id`, so an id names its shard
//! without a table, and with one shard the ids are the chain index's own. Content held by workers
//! on two shards is stored once per shard (hash arrays and run headers; the lane maps are per
//! worker either way), which is the price of the split; `entry_count` counts it once per shard.

use std::{
    collections::BTreeSet,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    chain_index::{ChainIndex, ChainIndexStats},
    event_tree::{
        ApplyError, ContentHash, OverlapScores, SequenceHash, StoredBlock, WorkerIdExhausted,
    },
    lane_map::ChainBlockMap,
};

/// Bits of a worker id below the shard number; a shard holds at most `1 << SHARD_SHIFT` worker
/// slots (the chain index allows 1,024).
pub const SHARD_SHIFT: u32 = 16;

const LOCAL_MASK: u32 = (1 << SHARD_SHIFT) - 1;

/// A chain index per shard; see the module documentation.
pub struct ShardedChainIndex {
    shards: Box<[ChainIndex]>,
    /// Where the next worker interned without a shard goes (round robin).
    next_shard: AtomicUsize,
}

impl ShardedChainIndex {
    /// `shards` chain indexes (at least one) with `max_workers` worker slots each.
    pub fn new(shards: usize, max_workers: usize) -> Self {
        let shards = shards.max(1);
        assert!(
            shards <= 1 << (32 - SHARD_SHIFT),
            "at most {} shards",
            1 << (32 - SHARD_SHIFT)
        );
        Self {
            shards: (0..shards)
                .map(|_| ChainIndex::with_max_workers(max_workers))
                .collect(),
            next_shard: AtomicUsize::new(0),
        }
    }

    /// `shards` chain indexes with the chain index's default worker slots.
    pub fn with_shards(shards: usize) -> Self {
        let shards = shards.max(1);
        Self {
            shards: (0..shards).map(|_| ChainIndex::new()).collect(),
            next_shard: AtomicUsize::new(0),
        }
    }

    pub fn shards(&self) -> usize {
        self.shards.len()
    }

    /// One shard's index, for per-shard figures.
    pub fn shard(&self, shard: usize) -> &ChainIndex {
        &self.shards[shard]
    }

    /// The shard a worker id names.
    #[inline]
    pub fn shard_of(worker: u32) -> usize {
        (worker >> SHARD_SHIFT) as usize
    }

    /// A worker's id within its shard.
    #[inline]
    pub fn local_id(worker: u32) -> u32 {
        worker & LOCAL_MASK
    }

    #[inline]
    fn global(shard: usize, local: u32) -> u32 {
        ((shard as u32) << SHARD_SHIFT) | local
    }

    #[inline]
    fn index_of(&self, worker: u32) -> &ChainIndex {
        &self.shards[Self::shard_of(worker)]
    }

    /// Intern `name` in `shard`; the id it already has if it is interned anywhere.
    pub fn intern_worker_in(&self, shard: usize, name: &str) -> Result<u32, WorkerIdExhausted> {
        assert!(
            shard < self.shards.len(),
            "shard {shard} of {}",
            self.shards.len()
        );
        if let Some(id) = self.worker_id(name) {
            return Ok(id);
        }
        self.shards[shard]
            .intern_worker(name)
            .map(|local| Self::global(shard, local))
    }

    /// Intern `name`, in the shard it already has or else the next one round robin: for a
    /// caller without a placement of its own (exact whatever the shard, see the module notes).
    pub fn intern_worker(&self, name: &str) -> Result<u32, WorkerIdExhausted> {
        if let Some(id) = self.worker_id(name) {
            return Ok(id);
        }
        let shard = if self.shards.len() == 1 {
            0
        } else {
            self.next_shard.fetch_add(1, Ordering::Relaxed) % self.shards.len()
        };
        self.intern_worker_in(shard, name)
    }

    /// The id of an interned worker.
    pub fn worker_id(&self, name: &str) -> Option<u32> {
        self.shards.iter().enumerate().find_map(|(shard, index)| {
            index
                .worker_id(name)
                .map(|local| Self::global(shard, local))
        })
    }

    /// Store `blocks` for `worker` after `parent` (position 0 when `None`).
    pub fn apply_stored(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
        map: &mut ChainBlockMap,
    ) -> Result<(), ApplyError> {
        self.index_of(worker)
            .apply_stored(Self::local_id(worker), blocks, parent, map)
    }

    /// Forget the named blocks of `worker`; unknown hashes are ignored.
    pub fn apply_removed(&self, worker: u32, hashes: &[SequenceHash], map: &mut ChainBlockMap) {
        self.index_of(worker)
            .apply_removed(Self::local_id(worker), hashes, map);
    }

    /// Forget every block of `worker` (the engine cleared its cache); the map is emptied.
    pub fn apply_cleared(&self, worker: u32, map: &mut ChainBlockMap) {
        self.index_of(worker)
            .apply_cleared(Self::local_id(worker), map);
    }

    /// Forget every block of `worker` (the worker left) and free its slot in its shard.
    pub fn remove_worker(&self, worker: u32, map: ChainBlockMap) {
        self.index_of(worker)
            .remove_worker(Self::local_id(worker), map);
    }

    /// Whether `worker`'s lane map holds the block with engine hash `key`.
    pub fn is_held(&self, worker: u32, map: &ChainBlockMap, key: SequenceHash) -> bool {
        self.index_of(worker).is_held(map, key)
    }

    /// Blocks `worker` holds.
    pub fn worker_block_count(&self, worker: u32) -> usize {
        self.index_of(worker)
            .worker_block_count(Self::local_id(worker))
    }

    /// Whether no worker of any shard holds a block (O(shards), each under its root's version).
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(ChainIndex::is_empty)
    }

    /// Blocks held across all workers (a block two workers hold counts twice).
    pub fn current_size(&self) -> usize {
        self.shards.iter().map(ChainIndex::current_size).sum()
    }

    /// Distinct blocks held, per shard and summed: content held on two shards counts twice,
    /// which is exactly the memory the split costs.
    pub fn entry_count(&self) -> usize {
        self.shards.iter().map(ChainIndex::entry_count).sum()
    }

    /// Score every worker by how many leading blocks of the request it holds; the union over
    /// the shards. With `early_exit`, the workers holding the first block, each scored 1.
    pub fn find_matches(&self, content_hashes: &[ContentHash], early_exit: bool) -> OverlapScores {
        if let [only] = &*self.shards {
            return only.find_matches(content_hashes, early_exit);
        }
        let mut out = OverlapScores::default();
        self.score_into(
            content_hashes,
            |content| content.0,
            early_exit,
            |worker, score| {
                out.scores.insert(worker, score);
            },
        );
        out
    }

    /// The lookup behind [`find_matches`](Self::find_matches) for callers with their own hash
    /// type and result shape, over every shard: `report` receives `(worker, score)` once per
    /// worker with a non-empty prefix. Returns the runs walked, summed over the shards.
    pub fn score_into<T>(
        &self,
        content_hashes: &[T],
        hash_of: impl Fn(&T) -> u64,
        early_exit: bool,
        mut report: impl FnMut(u32, u32),
    ) -> usize {
        if let [only] = &*self.shards {
            return only.score_into(content_hashes, &hash_of, early_exit, report);
        }
        let Some(first) = content_hashes.first() else {
            return 0;
        };
        let first = hash_of(first);
        let mut walked = 0;
        for (shard, index) in self.shards.iter().enumerate() {
            // A shard whose root has no run starting with the request's first block holds
            // nothing of it: one probe of its root table and the shard is passed over.
            let Some(entry) = index.head_entry(first) else {
                continue;
            };
            walked += index.score_from(
                entry,
                content_hashes,
                &hash_of,
                early_exit,
                |worker, score| {
                    report(Self::global(shard, worker), score);
                },
            );
        }
        walked
    }

    /// The lookup over one shard alone, for a caller that fans a lookup out to one thread per
    /// shard (one per socket) and merges: the call walks only that shard's memory, and `report`
    /// receives `(worker, score)` with the worker's global id, exactly the pairs
    /// [`score_into`](Self::score_into) would report for the shard. Worker sets are disjoint
    /// across shards, so the concatenation of every shard's reports is `score_into`'s answer.
    /// Returns the runs walked on the shard.
    pub fn score_shard_into<T>(
        &self,
        shard: usize,
        content_hashes: &[T],
        hash_of: impl Fn(&T) -> u64,
        early_exit: bool,
        mut report: impl FnMut(u32, u32),
    ) -> usize {
        self.shards[shard].score_into(content_hashes, hash_of, early_exit, |worker, score| {
            report(Self::global(shard, worker), score);
        })
    }

    /// How many shards hold a chain starting with `first` (a diagnostic for the lookup's shard
    /// filter: a shard without the head costs one root probe, a shard with it a walk).
    pub fn shards_holding_head(&self, first: u64) -> usize {
        self.shards
            .iter()
            .filter(|index| index.head_entry(first).is_some())
            .count()
    }

    /// Mergeable adjacent pairs and the blocks their children hold, summed over the shards (see
    /// [`ChainIndex::debug_mergeable`]).
    #[doc(hidden)]
    pub fn debug_mergeable(&self) -> (usize, usize) {
        self.shards
            .iter()
            .map(ChainIndex::debug_mergeable)
            .fold((0, 0), |(p, b), (q, c)| (p + q, b + c))
    }

    /// Shape and memory counters summed over the shards.
    pub fn stats(&self) -> ChainIndexStats {
        let mut total = ChainIndexStats::default();
        for stats in self.shards.iter().map(ChainIndex::stats) {
            total.runs_allocated += stats.runs_allocated;
            total.runs_free += stats.runs_free;
            total.runs_live += stats.runs_live;
            total.blocks_live += stats.blocks_live;
            total.arena_bytes += stats.arena_bytes;
            total.arena_free_bytes += stats.arena_free_bytes;
            total.arena_chunk_bytes += stats.arena_chunk_bytes;
            total.header_bytes += stats.header_bytes;
            total.slab_bytes += stats.slab_bytes;
            total.engine_conflicts += stats.engine_conflicts;
            total.landing_mismatches += stats.landing_mismatches;
            total.moved_hashes += stats.moved_hashes;
            total.splits_by_branch += stats.splits_by_branch;
            total.splits_by_hole += stats.splits_by_hole;
            total.splits_by_mid_run_store += stats.splits_by_mid_run_store;
            total.splits_by_prefix_holders += stats.splits_by_prefix_holders;
            total.runs_died += stats.runs_died;
            total.partial_entries += stats.partial_entries;
            total.max_partials = total.max_partials.max(stats.max_partials);
            total.child_entries += stats.child_entries;
            total.child_tombstones += stats.child_tombstones;
        }
        total
    }

    /// Shape and memory counters of each shard.
    pub fn shard_stats(&self) -> Vec<ChainIndexStats> {
        self.shards.iter().map(ChainIndex::stats).collect()
    }

    /// Every block every worker holds, as `(worker, position, content hash, prefix hash)`,
    /// with the workers' ids as the caller knows them.
    pub fn debug_blocks(&self) -> BTreeSet<(u32, usize, ContentHash, SequenceHash)> {
        self.shards
            .iter()
            .enumerate()
            .flat_map(|(shard, index)| {
                index
                    .debug_blocks()
                    .into_iter()
                    .map(move |(worker, position, content, prefix)| {
                        (Self::global(shard, worker), position, content, prefix)
                    })
            })
            .collect()
    }
}

impl Default for ShardedChainIndex {
    fn default() -> Self {
        Self::with_shards(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::{request_prefix_hashes, ReferenceIndexer};

    fn content(stream: u64, position: usize) -> ContentHash {
        crate::compute_content_hash(&[stream as u32, (stream >> 32) as u32, position as u32])
    }

    fn blocks_of(contents: &[ContentHash]) -> Vec<StoredBlock> {
        contents
            .iter()
            .zip(request_prefix_hashes(contents))
            .map(|(&content_hash, seq_hash)| StoredBlock {
                seq_hash,
                content_hash,
            })
            .collect()
    }

    fn sorted(scores: OverlapScores) -> Vec<(u32, u32)> {
        let mut v: Vec<(u32, u32)> = scores.scores.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// One shard is the chain index: the same ids, scores, blocks and counters for the same
    /// events.
    #[test]
    fn one_shard_is_the_chain_index() {
        let single = ChainIndex::with_max_workers(8);
        let sharded = ShardedChainIndex::new(1, 8);
        let chain: Vec<ContentHash> = (0..40).map(|p| content(1, p)).collect();
        let mut fork = chain[..10].to_vec();
        fork.extend((10..30).map(|p| content(2, p)));
        for (name, contents) in [("a", &chain), ("b", &fork), ("c", &chain)] {
            let (w1, w2) = (
                single.intern_worker(name).expect("id"),
                sharded.intern_worker(name).expect("id"),
            );
            assert_eq!(w1, w2);
            let (mut m1, mut m2) = (ChainBlockMap::default(), ChainBlockMap::default());
            single
                .apply_stored(w1, &blocks_of(contents), None, &mut m1)
                .expect("store");
            sharded
                .apply_stored(w2, &blocks_of(contents), None, &mut m2)
                .expect("store");
            let tail: Vec<SequenceHash> = blocks_of(contents)[25..]
                .iter()
                .map(|block| block.seq_hash)
                .collect();
            single.apply_removed(w1, &tail, &mut m1);
            sharded.apply_removed(w2, &tail, &mut m2);
            assert_eq!(m1.len(), m2.len());
        }
        for query in [&chain, &fork] {
            assert_eq!(
                sorted(single.find_matches(query, false)),
                sorted(sharded.find_matches(query, false))
            );
            assert_eq!(
                sorted(single.find_matches(query, true)),
                sorted(sharded.find_matches(query, true))
            );
        }
        assert_eq!(single.debug_blocks(), sharded.debug_blocks());
        assert_eq!(single.entry_count(), sharded.entry_count());
        assert_eq!(single.current_size(), sharded.current_size());
        assert_eq!(single.stats().arena_bytes, sharded.stats().arena_bytes);
        assert_eq!(sharded.shards(), 1);
        assert_eq!(
            ShardedChainIndex::shard_of(sharded.worker_id("b").expect("b")),
            0
        );
        assert_eq!(single.is_empty(), sharded.is_empty());
        assert!(ShardedChainIndex::new(2, 4).is_empty());
    }

    /// Two shards, workers placed round robin and by hand: every lookup, with and without the
    /// early exit, and the final block set agree with the reference, through stores,
    /// extensions, a divergence, a hole, a clear and a worker's removal.
    #[test]
    fn two_shards_are_exact_against_the_reference() {
        let index = ShardedChainIndex::new(2, 8);
        let mut reference = ReferenceIndexer::new();
        let chain: Vec<ContentHash> = (0..60).map(|p| content(3, p)).collect();
        let mut fork = chain[..20].to_vec();
        fork.extend((20..50).map(|p| content(4, p)));
        let names = ["a", "b", "c", "d"];
        let ids: Vec<u32> = names
            .iter()
            .map(|name| index.intern_worker(name).expect("id"))
            .collect();
        let e = index.intern_worker_in(1, "e").expect("id");
        assert_eq!(
            ids.iter()
                .map(|&id| ShardedChainIndex::shard_of(id))
                .collect::<Vec<_>>(),
            vec![0, 1, 0, 1]
        );
        assert_eq!(ShardedChainIndex::shard_of(e), 1);
        assert_eq!(index.intern_worker("e").expect("again"), e);
        let mut maps: Vec<ChainBlockMap> = (0..5).map(|_| ChainBlockMap::default()).collect();
        let workers = [ids[0], ids[1], ids[2], ids[3], e];
        let stores = [
            (0usize, blocks_of(&chain)),
            (1, blocks_of(&chain)),
            (2, blocks_of(&fork)),
            (3, blocks_of(&fork)),
            (4, blocks_of(&chain[..30])),
        ];
        for (slot, blocks) in &stores {
            index
                .apply_stored(workers[*slot], blocks, None, &mut maps[*slot])
                .expect("store");
            reference
                .apply_stored(workers[*slot], blocks, None)
                .expect("ref store");
        }
        // A hole in b's chain, e extends after its last block, c is cleared, d leaves.
        let hole: Vec<SequenceHash> = blocks_of(&chain)[10..15]
            .iter()
            .map(|block| block.seq_hash)
            .collect();
        index.apply_removed(workers[1], &hole, &mut maps[1]);
        reference.apply_removed(workers[1], &hole);
        let more = blocks_of(&chain);
        index
            .apply_stored(
                workers[4],
                &more[30..45],
                Some(more[29].seq_hash),
                &mut maps[4],
            )
            .expect("extend");
        reference
            .apply_stored(workers[4], &more[30..45], Some(more[29].seq_hash))
            .expect("ref extend");
        index.apply_cleared(workers[2], &mut maps[2]);
        reference.apply_cleared(workers[2]);
        index.remove_worker(workers[3], std::mem::take(&mut maps[3]));
        reference.remove_worker(workers[3]);
        let mut extended = chain[..45].to_vec();
        extended.extend((45..70).map(|p| content(5, p)));
        for query in [&chain, &fork, &extended, &chain[..12].to_vec()] {
            let expected: Vec<(u32, u32)> = reference.find_matches(query).into_iter().collect();
            assert_eq!(
                sorted(index.find_matches(query, false)),
                expected,
                "{query:?}"
            );
            let first: Vec<(u32, u32)> = expected.iter().map(|&(w, _)| (w, 1)).collect();
            assert_eq!(sorted(index.find_matches(query, true)), first);
        }
        assert_eq!(index.debug_blocks(), reference.blocks());
        assert_eq!(
            index.current_size(),
            reference.blocks().len(),
            "memberships summed over the shards"
        );
        assert!(index.is_held(workers[0], &maps[0], more[0].seq_hash));
        assert!(!index.is_held(workers[1], &maps[1], more[12].seq_hash));
        assert_eq!(index.worker_block_count(workers[1]), 55);
        assert_eq!(index.worker_block_count(workers[2]), 0);
        assert!(!index.is_empty());
        // The chain is held on both shards: once per shard in the distinct count and the arena.
        let per_shard = index.shard_stats();
        assert_eq!(per_shard.len(), 2);
        assert!(per_shard.iter().all(|stats| stats.blocks_live > 0));
        assert_eq!(
            index.entry_count(),
            index.shard(0).entry_count() + index.shard(1).entry_count()
        );
        assert!(index.entry_count() > 60 + 30);
    }

    /// A caller that interns into shards of its own choosing, including the same name from
    /// two places, gets one id per name and the shard it asked for first.
    /// A lookup fanned out one shard at a time reports, over the shards, exactly what the union
    /// lookup reports, and a shard reports only the workers it holds.
    #[test]
    fn a_lookup_fanned_out_per_shard_is_the_union_lookup() {
        let index = ShardedChainIndex::new(2, 8);
        let contents: Vec<ContentHash> = (0..40).map(|p| content(7, p)).collect();
        let mut workers = Vec::new();
        for (shard, name, len) in [(0, "a", 40), (1, "b", 25), (0, "c", 10), (1, "d", 40)] {
            let worker = index.intern_worker_in(shard, name).expect("id");
            let mut map = ChainBlockMap::default();
            index
                .apply_stored(worker, &blocks_of(&contents[..len]), None, &mut map)
                .expect("store");
            workers.push(worker);
        }
        let query = &contents[..32];
        let mut fanned: Vec<(u32, u32)> = Vec::new();
        for shard in 0..2 {
            index.score_shard_into(
                shard,
                query,
                |c| c.0,
                false,
                |worker, score| {
                    fanned.push((worker, score));
                },
            );
        }
        fanned.sort_unstable();
        assert_eq!(fanned, sorted(index.find_matches(query, false)));
        let mut expected = vec![
            (workers[0], 32),
            (workers[1], 25),
            (workers[2], 10),
            (workers[3], 32),
        ];
        expected.sort_unstable();
        assert_eq!(fanned, expected);
        let mut from_shard_1 = Vec::new();
        index.score_shard_into(
            1,
            query,
            |c| c.0,
            false,
            |worker, _| from_shard_1.push(worker),
        );
        from_shard_1.sort_unstable();
        assert_eq!(from_shard_1, vec![workers[1], workers[3]]);
    }

    #[test]
    fn interning_is_idempotent_across_shards() {
        let index = ShardedChainIndex::new(3, 4);
        let a = index.intern_worker_in(2, "a").expect("a");
        assert_eq!(ShardedChainIndex::shard_of(a), 2);
        assert_eq!(index.intern_worker_in(0, "a").expect("a again"), a);
        assert_eq!(index.intern_worker("a").expect("a once more"), a);
        assert_eq!(index.worker_id("a"), Some(a));
        assert_eq!(index.worker_id("zz"), None);
        // Round robin fills the shards evenly for names without a placement.
        let placed: Vec<usize> = ["p", "q", "r", "s", "t", "u"]
            .iter()
            .map(|name| ShardedChainIndex::shard_of(index.intern_worker(name).expect("id")))
            .collect();
        assert_eq!(placed.iter().filter(|&&s| s == 0).count(), 2);
        assert_eq!(placed.iter().filter(|&&s| s == 1).count(), 2);
        assert_eq!(placed.iter().filter(|&&s| s == 2).count(), 2);
        // A shard's slots run out on their own.
        for name in ["v", "w", "x", "y"] {
            let _ = index.intern_worker_in(0, name);
        }
        assert!(index.intern_worker_in(0, "overflow").is_err());
    }
}
