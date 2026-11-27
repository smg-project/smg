//! The event-driven KV index behind cache-aware routing: the chain index
//! ([`ShardedChainIndex`], one shard here: chains stored as runs), behind the
//! one surface the monitor and the policy use.
//!
//! The index is fed the engines' events and answers one question: how many
//! leading blocks of a request each worker holds. Every write-path method
//! takes a caller-owned per-worker reverse map ([`WorkerBlocks`]), created
//! the first time the worker stores a block. The monitor and the policy see
//! only [`KvIndex`] and [`WorkerBlocks`].
//!
//! Under `cfg(test)` a second variant wraps the crate's single-threaded
//! [`ReferenceIndexer`](kv_index::ReferenceIndexer), so the exactness tests
//! feed the production apply path to both and compare answers. Nothing
//! selects it at runtime: the gateway builds chain indexes only.

use std::{collections::BTreeSet, fmt, sync::Arc};

use kv_index::{
    request_prefix_hashes, ApplyError, ChainBlockMap, ChainIndexStats, ContentHash, GroupCap,
    GroupViews, OverlapScores, SequenceHash, ShardedChainIndex, StoredBlock, WorkerGroups,
    WorkerIdExhausted,
};
use smg_grand_central::engine::{overlap, CacheIndex};
use smg_worker::kv_events::apply::{BlockIndex, HeldBlocks};

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
pub enum KvIndex {
    /// The chain index: chains as runs with per-run worker coverage;
    /// lock-free, store-free lookups.
    Chain(ChainBackend),
    /// The single-threaded reference the chain index is checked against.
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
    /// The cache groups of the model's hybrid workers beside the index
    /// ([`kv_index::groups`]): empty, and never touched, on a plain fleet.
    groups: GroupViews,
}

impl ChainBackend {
    fn new() -> Self {
        Self {
            index: ShardedChainIndex::new(CHAIN_INDEX_SHARDS, CHAIN_INDEX_MAX_WORKERS),
            groups: GroupViews::new(),
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
/// needs, created the first time the worker stores a block. A worker's state
/// only ever meets the one index its subscription writes to.
#[derive(Default)]
pub struct WorkerBlocks {
    chain: Option<ChainBlockMap>,
    #[cfg(test)]
    reference: Option<reference::ReferenceBlocks>,
}

impl WorkerBlocks {
    /// Whether the worker holds the block the engine named `seq_hash`.
    pub fn contains_key(&self, seq_hash: SequenceHash) -> bool {
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

    /// The engine's hashes of the blocks the worker holds, in no order.
    pub fn hashes(&self) -> Vec<SequenceHash> {
        if let Some(map) = &self.chain {
            return map.iter().map(|(seq_hash, _)| seq_hash).collect();
        }
        #[cfg(test)]
        if let Some(held) = &self.reference {
            return held.hashes();
        }
        Vec::new()
    }

    fn chain(&mut self) -> &mut ChainBlockMap {
        self.chain.get_or_insert_with(ChainBlockMap::default)
    }
}

impl KvIndex {
    /// A chain index: what every model gets.
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
            Self::Chain(_) => "chain",
            #[cfg(test)]
            Self::Reference(_) => "reference",
        }
    }

    /// Intern a worker name; the same name maps to the same id until the
    /// worker is removed.
    pub fn intern_worker(&self, worker: &str) -> Result<u32, WorkerIdExhausted> {
        match self {
            Self::Chain(chain) => chain.intern_worker(worker),
            #[cfg(test)]
            Self::Reference(reference) => Ok(reference.intern_worker(worker)),
        }
    }

    /// The id a worker name was interned to, if it is interned now.
    pub fn worker_id(&self, worker: &str) -> Option<u32> {
        match self {
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
            Self::Chain(chain) => chain.apply_stored(worker, blocks, parent, held),
            #[cfg(test)]
            Self::Reference(reference) => reference.apply_stored(worker, blocks, parent, held),
        }
    }

    /// Forget the named blocks of `worker`; unknown hashes are ignored.
    pub fn apply_removed(&self, worker: u32, hashes: &[SequenceHash], held: &mut WorkerBlocks) {
        match self {
            Self::Chain(chain) => chain.apply_removed(worker, hashes, held),
            #[cfg(test)]
            Self::Reference(reference) => reference.apply_removed(worker, hashes, held),
        }
    }

    /// Forget every block of `worker`; the caller keeps the emptied state.
    pub fn apply_cleared(&self, worker: u32, held: &mut WorkerBlocks) {
        match self {
            Self::Chain(chain) => chain.apply_cleared(worker, held),
            #[cfg(test)]
            Self::Reference(reference) => reference.apply_cleared(worker, held),
        }
    }

    /// Forget every block of `worker` and the worker itself, with its group
    /// view; proportional to the worker's blocks, not to the index.
    pub fn remove_worker(&self, worker: u32, held: WorkerBlocks) {
        self.groups().remove(worker);
        match self {
            Self::Chain(chain) => chain.remove_worker(worker, held),
            #[cfg(test)]
            Self::Reference(reference) => reference.remove_worker(worker),
        }
    }

    /// The non-main cache groups of the model's workers, by index id: written
    /// by the workers' event streams, read by the decision through
    /// [`Self::cap_by_groups`].
    #[inline]
    pub fn groups(&self) -> &GroupViews {
        match self {
            Self::Chain(chain) => &chain.groups,
            #[cfg(test)]
            Self::Reference(reference) => &reference.groups,
        }
    }

    /// The engine's reuse rule over a hybrid worker's cache groups, applied
    /// to the overlap the index scored: every holder with a group view has
    /// its depth shrunk to what its sliding-window and state groups can serve
    /// ([`kv_index::WorkerGroups::cap`]), and a holder shrunk to nothing
    /// leaves the scores. Returns the holders it shrank, for the decision
    /// trace. One atomic load on a fleet without such workers, one read of
    /// the views table for every holder on a fleet with them.
    pub fn cap_by_groups(
        &self,
        content_hashes: &[ContentHash],
        block_size: usize,
        overlap: &mut OverlapScores,
    ) -> Vec<GroupCap> {
        let views = self.groups();
        if views.is_empty() {
            return Vec::new();
        }
        let table = views.read();
        let mut prefixes = Vec::new();
        let mut caps = Vec::new();
        overlap.scores.retain(|&worker, depth| {
            let Some(view) = table.view(worker) else {
                return true;
            };
            if prefixes.is_empty() {
                prefixes = request_prefix_hashes(content_hashes);
            }
            let cap = view.cap(&prefixes, *depth, block_size);
            if cap.depth < *depth {
                caps.push(GroupCap {
                    worker,
                    uncapped: *depth,
                    capped: cap.depth,
                    group: cap.by.unwrap_or(0),
                });
                *depth = cap.depth;
            }
            cap.depth > 0
        });
        caps
    }

    /// Score every worker by how many leading blocks of the request it holds.
    /// With `early_exit`, report the workers holding the first block, each
    /// scored 1. The request path's one call into the index.
    #[inline]
    pub fn find_matches(&self, content_hashes: &[ContentHash], early_exit: bool) -> OverlapScores {
        match self {
            Self::Chain(chain) => chain.find_matches(content_hashes, early_exit),
            #[cfg(test)]
            Self::Reference(reference) => reference.find_matches(content_hashes, early_exit),
        }
    }

    /// Blocks the index holds for `worker`: one counter read.
    pub fn worker_block_count(&self, worker: u32) -> usize {
        match self {
            Self::Chain(chain) => chain.worker_block_count(worker),
            #[cfg(test)]
            Self::Reference(reference) => reference.worker_block_count(worker),
        }
    }

    /// Whether no worker holds a block. Read once per request before the
    /// lookup, so it must stay cheap: the chain index reads its root's child
    /// table under one seqlock.
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Chain(chain) => chain.is_empty(),
            #[cfg(test)]
            Self::Reference(reference) => reference.is_empty(),
        }
    }

    /// Blocks held across all workers (a block two workers hold counts
    /// twice). Not for the request path: see [`is_empty`](Self::is_empty).
    pub fn current_size(&self) -> usize {
        match self {
            Self::Chain(chain) => chain.current_size(),
            #[cfg(test)]
            Self::Reference(reference) => reference.current_size(),
        }
    }

    /// Distinct index entries: distinct blocks on a chain.
    pub fn entry_count(&self) -> usize {
        match self {
            Self::Chain(chain) => chain.entry_count(),
            #[cfg(test)]
            Self::Reference(reference) => reference.entry_count(),
        }
    }

    /// The chain index's shape and memory counters; `None` for the reference.
    #[cfg_attr(
        not(test),
        expect(
            clippy::unnecessary_wraps,
            reason = "the test-only reference variant has no stats"
        )
    )]
    pub fn chain_stats(&self) -> Option<ChainIndexStats> {
        match self {
            Self::Chain(chain) => Some(chain.stats()),
            #[cfg(test)]
            Self::Reference(_) => None,
        }
    }

    /// Every membership as `(worker, position, content hash, prefix hash)`:
    /// a full walk, for the exactness tests only.
    #[doc(hidden)]
    pub fn debug_blocks(&self) -> BTreeSet<(u32, usize, ContentHash, SequenceHash)> {
        match self {
            Self::Chain(chain) => chain.debug_blocks(),
            #[cfg(test)]
            Self::Reference(reference) => reference.blocks(),
        }
    }
}

impl BlockIndex for KvIndex {
    type Held = WorkerBlocks;

    #[inline]
    fn apply_stored(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
        held: &mut WorkerBlocks,
    ) -> Result<(), ApplyError> {
        KvIndex::apply_stored(self, worker, blocks, parent, held)
    }

    #[inline]
    fn apply_removed(&self, worker: u32, hashes: &[SequenceHash], held: &mut WorkerBlocks) {
        KvIndex::apply_removed(self, worker, hashes, held);
    }

    #[inline]
    fn apply_cleared(&self, worker: u32, held: &mut WorkerBlocks) {
        KvIndex::apply_cleared(self, worker, held);
    }

    #[inline]
    fn groups(&self, worker: u32) -> Arc<WorkerGroups> {
        KvIndex::groups(self).get_or_create(worker)
    }
}

impl HeldBlocks for WorkerBlocks {
    #[inline]
    fn contains_key(&self, hash: SequenceHash) -> bool {
        WorkerBlocks::contains_key(self, hash)
    }

    #[inline]
    fn len(&self) -> usize {
        WorkerBlocks::len(self)
    }

    #[inline]
    fn hashes(&self) -> Vec<SequenceHash> {
        WorkerBlocks::hashes(self)
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

// Grand Central's capped lookup reads the chain index's one shard, whose
// ids are the index's own.
const _: () = assert!(CHAIN_INDEX_SHARDS == 1);

/// Grand Central's reads of the index, where it decides in the
/// gateway's own process: a worker's id and its block count once a list of
/// workers is read, and one lookup a request. The count of ids handed out
/// again is read beside both: ahead of a list's ids, and by a request
/// where it takes the list's table and behind its lookup.
impl CacheIndex for KvIndex {
    fn worker_number(&self, address: &str) -> Option<u32> {
        self.worker_id(address)
    }

    /// The chain index's one shard counts them; the reference index of the
    /// tests hands no id out twice.
    fn numbers_reused(&self) -> u64 {
        match self {
            Self::Chain(chain) => chain.index.shard(0).ids_reused(),
            #[cfg(test)]
            Self::Reference(_) => 0,
        }
    }

    /// The index's own test, the one the classic policy asks before every
    /// request ([`Self::is_empty`]): the engine asks it where a decision
    /// takes its list's table.
    fn holds_a_block(&self) -> bool {
        !self.is_empty()
    }

    fn worker_blocks(&self, number: u32) -> usize {
        self.worker_block_count(number)
    }

    /// On a fleet with no hybrid worker the chain index keeps the deepest
    /// holders itself, and nothing is allocated. Where a worker keeps cache
    /// groups, every holder's depth is first lowered to what its groups can
    /// serve ([`Self::cap_by_groups`]'s rule), as the classic decision
    /// lowers it, and the deepest are kept after. The lookup is the Grand
    /// Central crate's ([`overlap::deepest_within_groups`]).
    fn deepest_into(
        &self,
        block_hashes: &[u64],
        block_size: u32,
        within: &[u64],
        cap: usize,
        below: &mut dyn FnMut(u32) -> u32,
        kept: &mut Vec<(u32, u32)>,
    ) {
        match self {
            Self::Chain(chain) => {
                let (index, groups) = (chain.index.shard(0), &chain.groups);
                overlap::deepest_within_groups(
                    index,
                    groups,
                    block_hashes,
                    block_size,
                    within,
                    cap,
                    below,
                    kept,
                );
            }
            #[cfg(test)]
            Self::Reference(reference) => {
                let contents: Vec<ContentHash> =
                    block_hashes.iter().map(|&hash| ContentHash(hash)).collect();
                let scores = self.find_matches(&contents, false).scores;
                overlap::deepest_of_within_groups(
                    scores,
                    &reference.groups,
                    block_hashes,
                    block_size,
                    within,
                    cap,
                    below,
                    kept,
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/policies/kv_index/index_backend/apply_tests.rs"]
mod apply_tests;

#[cfg(test)]
#[path = "../../../tests/unit/policies/kv_index/index_backend/exactness.rs"]
mod exactness;

#[cfg(test)]
#[path = "../../../tests/unit/policies/kv_index/index_backend/mock_streams.rs"]
mod mock_streams;

#[cfg(test)]
#[path = "../../../tests/unit/policies/kv_index/index_backend/reference.rs"]
mod reference;

#[cfg(test)]
#[path = "../../../tests/unit/policies/kv_index/index_backend/tests.rs"]
mod tests;
