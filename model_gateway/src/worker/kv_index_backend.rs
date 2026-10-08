//! The event-driven KV index behind cache-aware routing, as one type over two
//! implementations: the [`PositionalIndexer`] the gateway has routed with so
//! far and the chain index ([`ShardedChainIndex`], one shard here: chains stored as runs),
//! selected at startup by `--kv-index {positional,chain}` ([`KvIndexKind`]).
//!
//! Both indexers are fed the same engine events and answer the same question
//! (how many leading blocks of a request each worker holds), and their
//! write-path methods already share a shape: every call takes a caller-owned
//! per-worker reverse map. What differs is the map's type, so [`WorkerBlocks`]
//! carries whichever map the index needs, created on first use. The monitor and
//! the policy see only [`KvIndex`] and [`WorkerBlocks`].
//!
//! An enum rather than a trait object: the lookup, the one call on the request
//! path, dispatches on a discriminant the branch predictor learns at startup,
//! and each variant's method is called directly, so the positional path costs
//! what it did before this type existed.
//!
//! Under `cfg(test)` a third variant wraps the crate's single-threaded
//! [`ReferenceIndexer`](kv_index::ReferenceIndexer), so the exactness tests
//! feed the production apply path to all three and compare answers.

use std::{collections::BTreeSet, fmt};

use kv_index::{
    ApplyError, ChainBlockMap, ChainIndexStats, ContentHash, OverlapScores, PositionalIndexer,
    PruneStats, SequenceHash, ShardedChainIndex, StoredBlock, WorkerBlockMap, WorkerIdExhausted,
};

pub use crate::config::KvIndexKind;

/// Workers one chain index interns at once: the index's own ceiling per shard.
/// Ids are handed back when a worker is removed, so this bounds the workers
/// of one model that stream events at the same time, not the churn over a
/// lifetime. Coverage costs one bit per worker per run, rounded up to 64,
/// and the lookup ANDs that many words per run on the matched path.
const CHAIN_INDEX_MAX_WORKERS: usize = 1024;

/// Shards of the chain index: one. The sharded type keeps one chain index per
/// shard and merges lookups across them, so that a gateway whose runtime
/// spans both sockets can later place each worker's index on the socket its
/// event lane runs on; until that placement exists, one shard is the plain
/// chain index with the same ids.
const CHAIN_INDEX_SHARDS: usize = 1;

/// The gateway's KV index: one per model, shared by the model's workers'
/// event subscriptions (writers) and the cache-aware policy (readers).
#[expect(
    clippy::large_enum_variant,
    reason = "one value per model behind an Arc: boxing the positional indexer would put a \
              second pointer hop on the lookup path to save bytes nobody pays for"
)]
pub enum KvIndex {
    /// One index entry per `(position, content hash)`, probed per block.
    Positional(PositionalIndexer),
    /// The chain index: chains as runs with per-run worker coverage;
    /// lock-free, store-free lookups.
    Chain(ChainBackend),
    /// The single-threaded reference the other two are checked against.
    #[cfg(test)]
    Reference(reference::ReferenceBackend),
}

/// The chain index with what the gateway keeps beside it. Every call the
/// gateway makes into the chain index goes through this type: a plain
/// `ChainIndex` and the sharded one share their call surface, so the payload
/// is the sharded type at one shard, and a per-socket placement later is a
/// matter of which shard a worker is interned into.
pub struct ChainBackend {
    index: ShardedChainIndex,
}

impl ChainBackend {
    fn new() -> Self {
        Self {
            index: ShardedChainIndex::new(CHAIN_INDEX_SHARDS, CHAIN_INDEX_MAX_WORKERS),
        }
    }

    fn intern_worker(&self, worker: &str) -> Result<u32, WorkerIdExhausted> {
        self.index.intern_worker(worker)
    }

    fn worker_id(&self, worker: &str) -> Option<u32> {
        self.index.worker_id(worker)
    }

    fn apply_stored(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
        held: &mut WorkerBlocks,
    ) -> Result<(), ApplyError> {
        self.index
            .apply_stored(worker, blocks, parent, held.chain())
    }

    fn apply_removed(&self, worker: u32, hashes: &[SequenceHash], held: &mut WorkerBlocks) {
        self.index.apply_removed(worker, hashes, held.chain());
    }

    fn apply_cleared(&self, worker: u32, held: &mut WorkerBlocks) {
        self.index.apply_cleared(worker, held.chain());
    }

    fn remove_worker(&self, worker: u32, held: WorkerBlocks) {
        self.index
            .remove_worker(worker, held.chain.unwrap_or_default());
    }

    #[inline]
    fn find_matches(&self, content_hashes: &[ContentHash], early_exit: bool) -> OverlapScores {
        self.index.find_matches(content_hashes, early_exit)
    }

    fn worker_block_count(&self, worker: u32) -> usize {
        self.index.worker_block_count(worker)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn current_size(&self) -> usize {
        self.index.current_size()
    }

    fn entry_count(&self) -> usize {
        self.index.entry_count()
    }

    fn stats(&self) -> ChainIndexStats {
        self.index.stats()
    }

    fn debug_blocks(&self) -> BTreeSet<(u32, usize, ContentHash, SequenceHash)> {
        self.index.debug_blocks()
    }
}

/// One worker's share of a [`KvIndex`]: the per-worker reverse map the index
/// needs, in the shape the index needs it, created the first time the worker
/// stores a block. A worker's state only ever meets the one index its
/// subscription writes to, so at most one map is live.
#[derive(Default)]
pub struct WorkerBlocks {
    positional: Option<WorkerBlockMap>,
    chain: Option<ChainBlockMap>,
    #[cfg(test)]
    reference: Option<reference::ReferenceBlocks>,
}

impl WorkerBlocks {
    /// Whether the worker holds the block the engine named `seq_hash`.
    pub fn contains_key(&self, seq_hash: SequenceHash) -> bool {
        if let Some(map) = &self.positional {
            return map.contains_key(&seq_hash);
        }
        if let Some(map) = &self.chain {
            return map.contains_key(seq_hash);
        }
        #[cfg(test)]
        if let Some(held) = &self.reference {
            return held.contains(seq_hash);
        }
        false
    }

    /// Blocks the worker holds.
    pub fn len(&self) -> usize {
        if let Some(map) = &self.positional {
            return map.len();
        }
        if let Some(map) = &self.chain {
            return map.len();
        }
        #[cfg(test)]
        if let Some(held) = &self.reference {
            return held.len();
        }
        0
    }

    /// Whether the worker holds no block.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn positional(&mut self) -> &mut WorkerBlockMap {
        self.positional.get_or_insert_with(WorkerBlockMap::default)
    }

    fn chain(&mut self) -> &mut ChainBlockMap {
        self.chain.get_or_insert_with(ChainBlockMap::default)
    }
}

impl KvIndex {
    /// An index of `kind`. `jump_size` is the positional indexer's historical
    /// tuning knob and is ignored by the chain index.
    pub fn new(kind: KvIndexKind, jump_size: usize) -> Self {
        match kind {
            KvIndexKind::Positional => Self::positional(jump_size),
            KvIndexKind::Chain => Self::chain(),
        }
    }

    pub fn positional(jump_size: usize) -> Self {
        Self::Positional(PositionalIndexer::new(jump_size))
    }

    pub fn chain() -> Self {
        Self::Chain(ChainBackend::new())
    }

    #[cfg(test)]
    pub fn reference() -> Self {
        Self::Reference(reference::ReferenceBackend::default())
    }

    /// The variant's name, for logs.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Positional(_) => "positional",
            Self::Chain(_) => "chain",
            #[cfg(test)]
            Self::Reference(_) => "reference",
        }
    }

    /// Intern a worker name; the same name maps to the same id until the
    /// worker is removed.
    pub fn intern_worker(&self, worker: &str) -> Result<u32, WorkerIdExhausted> {
        match self {
            Self::Positional(index) => index.intern_worker(worker),
            Self::Chain(chain) => chain.intern_worker(worker),
            #[cfg(test)]
            Self::Reference(reference) => Ok(reference.intern_worker(worker)),
        }
    }

    /// The id a worker name was interned to, if it is interned now.
    pub fn worker_id(&self, worker: &str) -> Option<u32> {
        match self {
            Self::Positional(index) => index.worker_id(worker),
            Self::Chain(chain) => chain.worker_id(worker),
            #[cfg(test)]
            Self::Reference(reference) => reference.worker_id(worker),
        }
    }

    /// Store `blocks` for `worker` after `parent` (position 0 when `None`).
    pub fn apply_stored(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
        held: &mut WorkerBlocks,
    ) -> Result<(), ApplyError> {
        match self {
            Self::Positional(index) => {
                index.apply_stored(worker, blocks, parent, held.positional())
            }
            Self::Chain(chain) => chain.apply_stored(worker, blocks, parent, held),
            #[cfg(test)]
            Self::Reference(reference) => reference.apply_stored(worker, blocks, parent, held),
        }
    }

    /// Forget the named blocks of `worker`; unknown hashes are ignored.
    pub fn apply_removed(&self, worker: u32, hashes: &[SequenceHash], held: &mut WorkerBlocks) {
        match self {
            Self::Positional(index) => index.apply_removed(worker, hashes, held.positional()),
            Self::Chain(chain) => chain.apply_removed(worker, hashes, held),
            #[cfg(test)]
            Self::Reference(reference) => reference.apply_removed(worker, hashes, held),
        }
    }

    /// Forget every block of `worker`; the caller keeps the emptied state.
    pub fn apply_cleared(&self, worker: u32, held: &mut WorkerBlocks) {
        match self {
            Self::Positional(index) => index.apply_cleared(worker, held.positional()),
            Self::Chain(chain) => chain.apply_cleared(worker, held),
            #[cfg(test)]
            Self::Reference(reference) => reference.apply_cleared(worker, held),
        }
    }

    /// Forget every block of `worker` and the worker itself; proportional to
    /// the worker's blocks, not to the index.
    pub fn remove_worker(&self, worker: u32, held: WorkerBlocks) {
        match self {
            Self::Positional(index) => {
                index.remove_worker(worker, held.positional.unwrap_or_default());
            }
            Self::Chain(chain) => chain.remove_worker(worker, held),
            #[cfg(test)]
            Self::Reference(reference) => reference.remove_worker(worker),
        }
    }

    /// Score every worker by how many leading blocks of the request it holds.
    /// With `early_exit`, report the workers holding the first block, each
    /// scored 1. The request path's one call into the index.
    #[inline]
    pub fn find_matches(&self, content_hashes: &[ContentHash], early_exit: bool) -> OverlapScores {
        match self {
            Self::Positional(index) => index.find_matches(content_hashes, early_exit),
            Self::Chain(chain) => chain.find_matches(content_hashes, early_exit),
            #[cfg(test)]
            Self::Reference(reference) => reference.find_matches(content_hashes, early_exit),
        }
    }

    /// Blocks the index holds for `worker`: one counter read.
    pub fn worker_block_count(&self, worker: u32) -> usize {
        match self {
            Self::Positional(index) => index.worker_block_count(worker),
            Self::Chain(chain) => chain.worker_block_count(worker),
            #[cfg(test)]
            Self::Reference(reference) => reference.worker_block_count(worker),
        }
    }

    /// Whether no worker holds a block. Read once per request before the
    /// lookup, so it must stay cheap: the positional indexer keeps a running
    /// total; the chain index reads its root's child table under one seqlock.
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Positional(index) => index.current_size() == 0,
            Self::Chain(chain) => chain.is_empty(),
            #[cfg(test)]
            Self::Reference(reference) => reference.is_empty(),
        }
    }

    /// Blocks held across all workers (a block two workers hold counts
    /// twice). Not for the request path: see [`is_empty`](Self::is_empty).
    pub fn current_size(&self) -> usize {
        match self {
            Self::Positional(index) => index.current_size(),
            Self::Chain(chain) => chain.current_size(),
            #[cfg(test)]
            Self::Reference(reference) => reference.current_size(),
        }
    }

    /// Distinct index entries: `(position, content hash)` pairs in the
    /// positional indexer, distinct blocks on a chain in the chain index.
    pub fn entry_count(&self) -> usize {
        match self {
            Self::Positional(index) => index.entry_count(),
            Self::Chain(chain) => chain.entry_count(),
            #[cfg(test)]
            Self::Reference(reference) => reference.entry_count(),
        }
    }

    /// Evict stale and excess entries. Only the positional indexer has a prune
    /// (its entries carry a last-touch stamp); the chain index holds exactly
    /// what the engines report and shrinks with their removals, so for it
    /// this is `None` and the bounds do not apply.
    pub fn prune(&self, ttl_secs: Option<u32>, max_entries: Option<usize>) -> Option<PruneStats> {
        match self {
            Self::Positional(index) => Some(index.prune(ttl_secs, max_entries)),
            Self::Chain(_) => None,
            #[cfg(test)]
            Self::Reference(_) => None,
        }
    }

    /// The chain index's shape and memory counters; `None` for the others.
    pub fn chain_stats(&self) -> Option<ChainIndexStats> {
        match self {
            Self::Chain(chain) => Some(chain.stats()),
            Self::Positional(_) => None,
            #[cfg(test)]
            Self::Reference(_) => None,
        }
    }

    /// Every membership as `(worker, position, content hash, prefix hash)`:
    /// a full walk, for the exactness tests only.
    #[doc(hidden)]
    pub fn debug_blocks(&self) -> BTreeSet<(u32, usize, ContentHash, SequenceHash)> {
        match self {
            Self::Positional(index) => index.debug_blocks().into_iter().collect(),
            Self::Chain(chain) => chain.debug_blocks(),
            #[cfg(test)]
            Self::Reference(reference) => reference.blocks(),
        }
    }
}

impl fmt::Debug for KvIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvIndex")
            .field("kind", &self.name())
            .field("blocks", &self.current_size())
            .finish()
    }
}

#[cfg(test)]
mod exactness;

#[cfg(test)]
mod mock_streams;

#[cfg(test)]
mod reference {
    //! The reference indexer behind the [`KvIndex`](super::KvIndex) surface:
    //! interning and the per-worker membership set the other variants keep in
    //! their maps, over `kv_index`'s single-threaded model.

    use std::collections::{BTreeSet, HashMap, HashSet};

    use kv_index::{
        ApplyError, ContentHash, OverlapScores, ReferenceIndexer, SequenceHash, StoredBlock,
    };
    use parking_lot::Mutex;

    use super::WorkerBlocks;

    /// The engine hashes a worker holds, mirrored from the reference so the
    /// monitor's copy counting sees the same `contains_key` answers.
    #[derive(Default)]
    pub struct ReferenceBlocks(HashSet<SequenceHash>);

    impl ReferenceBlocks {
        pub fn contains(&self, seq_hash: SequenceHash) -> bool {
            self.0.contains(&seq_hash)
        }

        pub fn len(&self) -> usize {
            self.0.len()
        }
    }

    #[derive(Default)]
    struct Inner {
        index: ReferenceIndexer,
        names: HashMap<String, u32>,
        next: u32,
    }

    #[derive(Default)]
    pub struct ReferenceBackend {
        inner: Mutex<Inner>,
    }

    impl ReferenceBackend {
        pub fn intern_worker(&self, worker: &str) -> u32 {
            let mut inner = self.inner.lock();
            if let Some(&id) = inner.names.get(worker) {
                return id;
            }
            let id = inner.next;
            inner.next += 1;
            inner.names.insert(worker.to_string(), id);
            id
        }

        pub fn worker_id(&self, worker: &str) -> Option<u32> {
            self.inner.lock().names.get(worker).copied()
        }

        pub fn apply_stored(
            &self,
            worker: u32,
            blocks: &[StoredBlock],
            parent: Option<SequenceHash>,
            held: &mut WorkerBlocks,
        ) -> Result<(), ApplyError> {
            self.inner
                .lock()
                .index
                .apply_stored(worker, blocks, parent)?;
            let mirror = held.reference.get_or_insert_with(ReferenceBlocks::default);
            mirror.0.extend(blocks.iter().map(|block| block.seq_hash));
            Ok(())
        }

        pub fn apply_removed(&self, worker: u32, hashes: &[SequenceHash], held: &mut WorkerBlocks) {
            self.inner.lock().index.apply_removed(worker, hashes);
            if let Some(mirror) = held.reference.as_mut() {
                for hash in hashes {
                    mirror.0.remove(hash);
                }
            }
        }

        pub fn apply_cleared(&self, worker: u32, held: &mut WorkerBlocks) {
            self.inner.lock().index.apply_cleared(worker);
            held.reference = None;
        }

        pub fn remove_worker(&self, worker: u32) {
            let mut inner = self.inner.lock();
            inner.index.remove_worker(worker);
            inner.names.retain(|_, id| *id != worker);
        }

        pub fn find_matches(
            &self,
            content_hashes: &[ContentHash],
            early_exit: bool,
        ) -> OverlapScores {
            let inner = self.inner.lock();
            let scored = if early_exit {
                inner
                    .index
                    .find_matches(&content_hashes[..content_hashes.len().min(1)])
            } else {
                inner.index.find_matches(content_hashes)
            };
            let mut out = OverlapScores::default();
            for (worker, score) in scored {
                out.scores.insert(worker, score);
            }
            out
        }

        pub fn worker_block_count(&self, worker: u32) -> usize {
            self.inner.lock().index.worker_block_count(worker)
        }

        pub fn is_empty(&self) -> bool {
            self.inner.lock().index.blocks().is_empty()
        }

        pub fn current_size(&self) -> usize {
            self.inner.lock().index.blocks().len()
        }

        pub fn entry_count(&self) -> usize {
            let inner = self.inner.lock();
            inner
                .index
                .blocks()
                .into_iter()
                .map(|(_, position, content, prefix)| (position, content, prefix))
                .collect::<BTreeSet<_>>()
                .len()
        }

        pub fn blocks(&self) -> BTreeSet<(u32, usize, ContentHash, SequenceHash)> {
            self.inner.lock().index.blocks()
        }
    }
}

#[cfg(test)]
mod tests {
    use kv_index::compute_content_hash;

    use super::*;

    fn chain(contents: &[&[u32]]) -> Vec<StoredBlock> {
        let hashes: Vec<ContentHash> = contents.iter().map(|c| compute_content_hash(c)).collect();
        hashes
            .iter()
            .zip(kv_index::request_prefix_hashes(&hashes))
            .map(|(&content_hash, seq_hash)| StoredBlock {
                seq_hash,
                content_hash,
            })
            .collect()
    }

    fn backends() -> Vec<KvIndex> {
        vec![
            KvIndex::positional(64),
            KvIndex::chain(),
            KvIndex::reference(),
        ]
    }

    #[test]
    fn every_backend_scores_stores_removals_and_clears_the_same_way() {
        let blocks = chain(&[&[1, 2, 3, 4], &[5, 6, 7, 8], &[9, 10, 11, 12]]);
        let request: Vec<ContentHash> = blocks.iter().map(|b| b.content_hash).collect();
        for index in backends() {
            let name = index.name();
            assert!(index.is_empty(), "{name}: empty at start");
            let w1 = index.intern_worker("grpc://w1").unwrap();
            let w2 = index.intern_worker("grpc://w2").unwrap();
            assert_eq!(index.worker_id("grpc://w2"), Some(w2), "{name}");
            let mut held1 = WorkerBlocks::default();
            let mut held2 = WorkerBlocks::default();
            index.apply_stored(w1, &blocks, None, &mut held1).unwrap();
            index
                .apply_stored(w2, &blocks[..2], None, &mut held2)
                .unwrap();
            assert!(!index.is_empty(), "{name}: not empty after stores");
            assert!(held1.contains_key(blocks[2].seq_hash), "{name}");
            assert!(!held2.contains_key(blocks[2].seq_hash), "{name}");
            assert_eq!(held1.len(), 3, "{name}");
            assert_eq!(index.worker_block_count(w1), 3, "{name}");
            assert_eq!(index.worker_block_count(w2), 2, "{name}");
            assert_eq!(index.current_size(), 5, "{name}");

            let scores = index.find_matches(&request, false).scores;
            assert_eq!(scores.get(&w1), Some(&3), "{name}: w1 holds the chain");
            assert_eq!(scores.get(&w2), Some(&2), "{name}: w2 holds two blocks");
            let early = index.find_matches(&request, true).scores;
            assert_eq!(early.get(&w1), Some(&1), "{name}: early exit scores 1");
            assert_eq!(early.get(&w2), Some(&1), "{name}: early exit scores 1");

            // A middle removal ends w1's match at the hole; w2 is untouched.
            index.apply_removed(w1, &[blocks[1].seq_hash], &mut held1);
            let scores = index.find_matches(&request, false).scores;
            assert_eq!(scores.get(&w1), Some(&1), "{name}: hole ends the match");
            assert_eq!(scores.get(&w2), Some(&2), "{name}");
            assert_eq!(index.worker_block_count(w1), 2, "{name}");

            // A store after the parent heals the hole.
            index
                .apply_stored(w1, &blocks[1..2], Some(blocks[0].seq_hash), &mut held1)
                .unwrap();
            let scores = index.find_matches(&request, false).scores;
            assert_eq!(scores.get(&w1), Some(&3), "{name}: healed");

            // An unknown parent is reported, as the monitor's fallback expects.
            assert!(
                matches!(
                    index.apply_stored(w2, &blocks[2..], Some(SequenceHash(0xdead)), &mut held2),
                    Err(ApplyError::ParentBlockNotFound)
                ),
                "{name}"
            );
            let mut fresh = WorkerBlocks::default();
            let w3 = index.intern_worker("grpc://w3").unwrap();
            assert!(
                matches!(
                    index.apply_stored(w3, &blocks[1..], Some(blocks[0].seq_hash), &mut fresh),
                    Err(ApplyError::WorkerNotTracked)
                ),
                "{name}"
            );

            index.apply_cleared(w1, &mut held1);
            assert!(held1.is_empty(), "{name}: cleared state is empty");
            assert_eq!(index.worker_block_count(w1), 0, "{name}");
            assert!(
                !index.find_matches(&request, false).scores.contains_key(&w1),
                "{name}: cleared worker scores nothing"
            );
            index.remove_worker(w2, held2);
            assert_eq!(index.worker_block_count(w2), 0, "{name}");
            assert!(index.is_empty(), "{name}: empty again");
            assert_eq!(index.current_size(), 0, "{name}");
            assert!(index.debug_blocks().is_empty(), "{name}");
        }
    }

    #[test]
    fn the_chain_index_hands_back_removed_worker_ids_and_knows_when_it_is_empty() {
        let index = KvIndex::chain();
        let first = index.intern_worker("grpc://a").unwrap();
        let mut held = WorkerBlocks::default();
        let blocks = chain(&[&[1, 2, 3, 4]]);
        index.apply_stored(first, &blocks, None, &mut held).unwrap();
        index.remove_worker(first, held);
        assert_eq!(index.worker_id("grpc://a"), None);
        assert!(index.is_empty());
        // The freed id is reused; the emptiness check still covers it.
        let again = index.intern_worker("grpc://b").unwrap();
        assert_eq!(again, first);
        let mut held = WorkerBlocks::default();
        index.apply_stored(again, &blocks, None, &mut held).unwrap();
        assert!(!index.is_empty());
        assert_eq!(index.chain_stats().map(|s| s.blocks_live), Some(1));
    }

    #[test]
    fn only_the_positional_index_prunes() {
        let positional = KvIndex::positional(64);
        assert!(positional.prune(Some(1), None).is_some());
        assert!(KvIndex::chain().prune(Some(1), Some(1)).is_none());
        assert!(KvIndex::chain().chain_stats().is_some());
        assert!(positional.chain_stats().is_none());
    }

    #[test]
    fn kind_selects_the_backend() {
        assert_eq!(
            KvIndex::new(KvIndexKind::Positional, 8).name(),
            "positional"
        );
        assert_eq!(KvIndexKind::Chain.as_str(), "chain");
        assert_eq!(KvIndex::new(KvIndexKind::Chain, 8).name(), "chain");
        assert_eq!(
            format!("{:?}", KvIndex::chain()),
            "KvIndex { kind: \"chain\", blocks: 0 }"
        );
    }
}
