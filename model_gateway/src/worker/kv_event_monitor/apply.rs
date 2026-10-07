//! One event into the index: which tiers, cache groups, localities and
//! owners are indexed, content hashing under the event's namespace, and the
//! physical copies of a block counted per worker so a removal evicts the
//! block only when its last copy goes.

use std::collections::{hash_map::Entry, HashMap};

use kv_index::{
    salt::{content_hash_with_seed, namespace_seed},
    ApplyError, SequenceHash, StoredBlock,
};
use smg_grpc_client::common_proto::{
    kv_cache_event, KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheEvent, KvCacheLocality,
    KvCacheTier,
};
use tracing::warn;

use super::KvEventMonitor;
use crate::worker::kv_index_backend::{KvIndex, WorkerBlocks};

impl KvEventMonitor {
    /// Apply a single KV cache event to the indexer.
    pub(crate) fn apply_event(
        event: &KvCacheEvent,
        worker_id: u32,
        indexer: &KvIndex,
        worker_blocks: &mut WorkerIndexState,
    ) {
        let Some(ref data) = event.data else {
            return;
        };

        match data {
            kv_cache_event::Data::Stored(stored) => {
                Self::apply_stored(stored, worker_id, indexer, worker_blocks);
            }
            kv_cache_event::Data::Removed(removed) => {
                Self::apply_removed(removed, worker_id, indexer, worker_blocks);
            }
            kv_cache_event::Data::Cleared(cleared) => {
                if worker_blocks.admits(None, None, None, cleared.ownership.as_deref()) {
                    Self::apply_cleared(worker_id, indexer, worker_blocks);
                }
            }
        }
    }

    /// Convert proto `KvBlocksStored` and apply to the indexer.
    ///
    /// Blocks on the disk and external tiers, in cache groups other than main
    /// attention, not local to the worker, or owned by a residency agent are
    /// counted and skipped. Content hashes are computed under the event's
    /// LoRA name and cache salt, so a salted block matches only a request
    /// hashed under the same namespace.
    fn apply_stored(
        stored: &KvBlocksStored,
        worker_id: u32,
        indexer: &KvIndex,
        worker_blocks: &mut WorkerIndexState,
    ) {
        if !worker_blocks.admits(
            stored.kv_cache_spec_kind.as_deref(),
            stored.group_idx,
            stored.locality,
            stored.ownership.as_deref(),
        ) {
            return;
        }
        let first_level = stored.blocks.first().and_then(|block| block.cache_level);
        let Some(tier) = indexed_tier(stored.tier, first_level) else {
            worker_blocks.counters.untracked_tier += 1;
            return;
        };

        let seed = namespace_seed(stored.lora_name.as_deref(), stored.cache_salt.as_deref());
        let blocks: Vec<StoredBlock> = stored
            .blocks
            .iter()
            .map(|block| convert_kv_block(block, seed))
            .collect();
        worker_blocks.note_stored(&blocks, tier);

        let parent_seq_hash = stored.parent_block_hash.map(SequenceHash::from);

        match indexer.apply_stored(
            worker_id,
            &blocks,
            parent_seq_hash,
            &mut worker_blocks.blocks,
        ) {
            Ok(()) => {}
            Err(ApplyError::WorkerNotTracked | ApplyError::ParentBlockNotFound) => {
                // Cold start or parent evicted — retry without parent to start a new chain.
                worker_blocks.counters.parentless_stores += 1;
                worker_blocks.counters.parentless_blocks += blocks.len() as u64;
                if let Err(e) =
                    indexer.apply_stored(worker_id, &blocks, None, &mut worker_blocks.blocks)
                {
                    warn!(
                        worker_id = worker_id,
                        error = %e,
                        "Failed to apply stored event after fallback"
                    );
                }
            }
        }
    }

    /// Convert proto `KvBlocksRemoved` and apply to the indexer.
    ///
    /// A removal names one tier; a block leaves the index only when no
    /// indexed copy remains on another tier.
    fn apply_removed(
        removed: &KvBlocksRemoved,
        worker_id: u32,
        indexer: &KvIndex,
        worker_blocks: &mut WorkerIndexState,
    ) {
        if !worker_blocks.admits(
            None,
            removed.group_idx,
            removed.locality,
            removed.ownership.as_deref(),
        ) {
            return;
        }
        let Some(tier) = indexed_tier(removed.tier, removed.cache_level) else {
            worker_blocks.counters.untracked_tier += 1;
            return;
        };

        let hashes = removed.block_hashes.iter().map(|&h| SequenceHash::from(h));
        let seq_hashes: Vec<SequenceHash> =
            if tier == IndexedTier::Device && worker_blocks.copies.is_empty() {
                hashes.collect()
            } else {
                hashes
                    .filter(|&seq_hash| worker_blocks.release(seq_hash, tier))
                    .collect()
            };

        indexer.apply_removed(worker_id, &seq_hashes, &mut worker_blocks.blocks);
    }

    /// Drop every block of a worker from the indexer and forget its copies.
    pub(super) fn apply_cleared(
        worker_id: u32,
        indexer: &KvIndex,
        worker_blocks: &mut WorkerIndexState,
    ) {
        indexer.apply_cleared(worker_id, &mut worker_blocks.blocks);
        worker_blocks.copies.clear();
    }
}

/// Convert a proto `KvBlock` to a kv-index `StoredBlock`, hashing its tokens
/// under `seed` (see [`namespace_seed`]).
pub(super) fn convert_kv_block(block: &KvBlock, seed: u64) -> StoredBlock {
    StoredBlock {
        seq_hash: SequenceHash::from(block.block_hash),
        content_hash: content_hash_with_seed(&block.token_ids, seed),
    }
}

/// The residency tiers the index tracks: the device, and the host cache the
/// engine restores from without recompute. Disk and external copies are not
/// indexed; a hit there costs an engine-side fetch the router cannot price.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IndexedTier {
    Device,
    Host,
}

/// The tier an event names: its `tier` when set, else a block's
/// `cache_level` (absent means the device). `None` when the index does not
/// track that tier.
fn indexed_tier(tier: Option<i32>, cache_level: Option<i32>) -> Option<IndexedTier> {
    let tier = match tier.and_then(|tier| KvCacheTier::try_from(tier).ok()) {
        Some(KvCacheTier::Unspecified) | None => match cache_level.unwrap_or(0) {
            0 => KvCacheTier::Device,
            1 => KvCacheTier::Host,
            2 => KvCacheTier::Disk,
            _ => KvCacheTier::External,
        },
        Some(tier) => tier,
    };
    match tier {
        KvCacheTier::Unspecified | KvCacheTier::Device => Some(IndexedTier::Device),
        KvCacheTier::Host => Some(IndexedTier::Host),
        KvCacheTier::Disk | KvCacheTier::External => None,
    }
}

/// Cache-group kinds whose blocks hold the main attention KV, the ones
/// prefix matching is about. Sliding-window and Mamba groups are skipped.
const MAIN_ATTENTION_KINDS: [&str; 3] = ["full_attention", "mla_attention", "sink_full_attention"];

/// The most physical copies of one block counted per tier. vLLM's opt-in
/// `kv_cache_report_mode: full` re-announces whole hit chains without
/// removals, which would grow a count without bound; the cap turns that into
/// at most this many extra removals before the block leaves the index.
const COPIES_CAP: u8 = 8;

/// What the positional index did not take at face value, by reason; logged
/// when the worker's subscription ends.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerIndexCounters {
    /// Stores and removals on tiers the index does not track.
    pub(crate) untracked_tier: u64,
    /// Events for cache groups other than main attention.
    pub(crate) non_main_group: u64,
    /// Events for blocks not local to the worker.
    pub(crate) remote: u64,
    /// Events owned by a residency agent rather than the engine.
    pub(crate) foreign_owner: u64,
    /// Removals on a tier that held no counted copy of the block.
    pub(crate) unknown_copy: u64,
    /// Stores of a block already indexed on that tier (a second physical copy).
    pub(crate) duplicate_copies: u64,
    /// Copy counts that hit [`COPIES_CAP`].
    pub(crate) capped_copies: u64,
    /// Stores whose parent the index did not hold, placed as a new chain from
    /// the root instead, and the blocks they carried. Every one duplicates
    /// content the chain may already hold further down, so a rising count is
    /// where fragmentation of the chain index is looked for first.
    pub(crate) parentless_stores: u64,
    pub(crate) parentless_blocks: u64,
}

impl WorkerIndexCounters {
    /// Parent-less stores and blocks this state saw since `before`.
    pub(super) fn parentless_since(&self, before: &Self) -> (u64, u64) {
        (
            self.parentless_stores - before.parentless_stores,
            self.parentless_blocks - before.parentless_blocks,
        )
    }
}

/// Physical copies of one block per tier, all of the worker's ranks pooled:
/// the worker URL is the routing target and its copies are interchangeable
/// for a prefix hit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Copies {
    device: u8,
    host: u8,
}

impl Copies {
    fn on(&mut self, tier: IndexedTier) -> &mut u8 {
        match tier {
            IndexedTier::Device => &mut self.device,
            IndexedTier::Host => &mut self.host,
        }
    }

    fn none(self) -> bool {
        self.device == 0 && self.host == 0
    }
}

/// A worker's share of the positional index: the indexer's reverse map plus
/// the physical copies of each block per tier, once the worker reports more
/// than one copy or a tier other than the device.
///
/// The engines do not deduplicate physical blocks: vLLM recomputes the last
/// block of an exact resend into a second copy with the same hash and emits
/// `BlockRemoved` per copy, and SGLang's HiCache keeps a host copy next to
/// the device one. The relay forwards every store and removal, so this state
/// counts copies and lets a removal evict the block only when none remains.
///
/// Counting is sparse. A block gets an entry in `copies` only on a host
/// store or on a second store of an indexed hash; an indexed block without
/// an entry is a single device copy. A worker that never duplicates and
/// never offloads pays one lookup per stored block and no memory.
#[derive(Default)]
pub(crate) struct WorkerIndexState {
    /// The indexer's caller-owned reverse map for this worker.
    pub(crate) blocks: WorkerBlocks,
    /// Copies per tier of blocks with a host copy or more than one copy.
    copies: HashMap<SequenceHash, Copies>,
    /// Cache groups whose kind is not main attention.
    non_main_groups: Vec<u32>,
    pub(crate) counters: WorkerIndexCounters,
}

impl WorkerIndexState {
    /// Whether an event with these attributes belongs in the index. A store
    /// names its group's kind; later events for that group may omit it, so
    /// non-main groups are remembered.
    fn admits(
        &mut self,
        kind: Option<&str>,
        group_idx: Option<u32>,
        locality: Option<i32>,
        ownership: Option<&str>,
    ) -> bool {
        if ownership.is_some_and(|owner| owner.eq_ignore_ascii_case("kvcr")) {
            self.counters.foreign_owner += 1;
            return false;
        }
        if locality.is_some_and(|locality| locality == KvCacheLocality::Remote as i32) {
            self.counters.remote += 1;
            return false;
        }
        let main = match kind {
            Some(kind) => {
                let main = MAIN_ATTENTION_KINDS.contains(&kind);
                if let Some(group) = group_idx {
                    if main {
                        self.non_main_groups.retain(|&known| known != group);
                    } else if !self.non_main_groups.contains(&group) {
                        self.non_main_groups.push(group);
                    }
                }
                main
            }
            None => !group_idx.is_some_and(|group| self.non_main_groups.contains(&group)),
        };
        if !main {
            self.counters.non_main_group += 1;
        }
        main
    }

    /// Count a store on `tier`, before the indexer applies it. A second copy
    /// of an indexed block, or any host copy, opens the block's entry; the
    /// implicit single device copy is credited when it does.
    fn note_stored(&mut self, blocks: &[StoredBlock], tier: IndexedTier) {
        for block in blocks {
            let indexed = self.blocks.contains_key(block.seq_hash);
            let entry = match self.copies.entry(block.seq_hash) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(vacant) => match tier {
                    IndexedTier::Device if !indexed => continue,
                    _ => vacant.insert(Copies {
                        device: u8::from(indexed),
                        host: 0,
                    }),
                },
            };
            let count = entry.on(tier);
            if *count > 0 {
                self.counters.duplicate_copies += 1;
            }
            if *count < COPIES_CAP {
                *count += 1;
            } else {
                self.counters.capped_copies += 1;
            }
        }
    }

    /// Drop one copy of a block on `tier`; `true` when no counted copy
    /// remains and the block should leave the index.
    fn release(&mut self, seq_hash: SequenceHash, tier: IndexedTier) -> bool {
        match self.copies.entry(seq_hash) {
            Entry::Occupied(mut entry) => {
                let count = entry.get_mut().on(tier);
                if *count == 0 {
                    self.counters.unknown_copy += 1;
                    return false;
                }
                *count -= 1;
                if entry.get().none() {
                    entry.remove();
                    true
                } else {
                    false
                }
            }
            Entry::Vacant(_) => match tier {
                IndexedTier::Device => true,
                IndexedTier::Host => {
                    self.counters.unknown_copy += 1;
                    false
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use kv_index::{
        compute_content_hash, compute_request_content_hashes,
        salt::namespaced_request_content_hashes, ContentHash, XXH3_SEED,
    };
    use smg_grpc_client::common_proto::KvCacheCleared;

    use super::*;

    #[test]
    fn a_proto_block_becomes_its_engine_and_content_hashes() {
        let block = KvBlock {
            block_hash: 42,
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: None,
            cache_level: None,
            ..Default::default()
        };
        let stored = convert_kv_block(&block, XXH3_SEED);
        assert_eq!(stored.seq_hash, SequenceHash::from(42i64));
        assert_eq!(stored.content_hash, compute_content_hash(&[1, 2, 3, 4]));
    }

    #[test]
    fn a_negative_engine_hash_keeps_its_bits() {
        let block = KvBlock {
            block_hash: -1,
            token_ids: vec![10, 20],
            block_size: 2,
            lora_id: None,
            cache_level: None,
            ..Default::default()
        };
        let stored = convert_kv_block(&block, XXH3_SEED);
        assert_eq!(stored.seq_hash, SequenceHash(u64::MAX));
    }

    #[test]
    fn a_block_without_tokens_hashes_the_empty_content() {
        let block = KvBlock {
            block_hash: 100,
            token_ids: vec![],
            block_size: 0,
            lora_id: None,
            cache_level: None,
            ..Default::default()
        };
        let stored = convert_kv_block(&block, XXH3_SEED);
        assert_eq!(stored.seq_hash, SequenceHash::from(100i64));
        assert_eq!(stored.content_hash, compute_content_hash(&[]));
    }

    #[test]
    fn a_root_store_indexes_its_blocks() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let stored = KvBlocksStored {
            blocks: vec![
                KvBlock {
                    block_hash: 1,
                    token_ids: vec![10, 20, 30, 40],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                    ..Default::default()
                },
                KvBlock {
                    block_hash: 2,
                    token_ids: vec![50, 60, 70, 80],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                    ..Default::default()
                },
            ],
            parent_block_hash: None,
            ..Default::default()
        };

        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2);
    }

    #[test]
    fn a_chained_store_extends_its_parent() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        let stored1 = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 1,
                token_ids: vec![10, 20, 30, 40],
                block_size: 4,
                lora_id: None,
                cache_level: None,
                ..Default::default()
            }],
            parent_block_hash: None,
            ..Default::default()
        };
        KvEventMonitor::apply_stored(&stored1, w1, &indexer, &mut wb);

        let stored2 = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 2,
                token_ids: vec![50, 60, 70, 80],
                block_size: 4,
                lora_id: None,
                cache_level: None,
                ..Default::default()
            }],
            parent_block_hash: Some(1),
            ..Default::default()
        };
        KvEventMonitor::apply_stored(&stored2, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2);
    }

    #[test]
    fn a_store_with_an_unknown_parent_starts_a_new_chain() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://new-worker:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        // Pass parent_block_hash for an untracked worker — should fallback to no parent.
        let stored = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 1,
                token_ids: vec![10, 20, 30, 40],
                block_size: 4,
                lora_id: None,
                cache_level: None,
                ..Default::default()
            }],
            parent_block_hash: Some(999),
            ..Default::default()
        };
        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
    }

    #[test]
    fn a_removal_takes_the_block_out() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        let stored = KvBlocksStored {
            blocks: vec![
                KvBlock {
                    block_hash: 1,
                    token_ids: vec![10, 20, 30, 40],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                    ..Default::default()
                },
                KvBlock {
                    block_hash: 2,
                    token_ids: vec![50, 60, 70, 80],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                    ..Default::default()
                },
            ],
            parent_block_hash: None,
            ..Default::default()
        };
        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);

        let removed = KvBlocksRemoved {
            block_hashes: vec![2],
            cache_level: None,
            ..Default::default()
        };
        KvEventMonitor::apply_removed(&removed, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
    }

    #[test]
    fn a_clear_empties_the_worker() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        let stored = KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: 1,
                token_ids: vec![10, 20, 30, 40],
                block_size: 4,
                lora_id: None,
                cache_level: None,
                ..Default::default()
            }],
            parent_block_hash: None,
            ..Default::default()
        };
        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);

        KvEventMonitor::apply_cleared(w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn apply_event_dispatches_a_store() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let event = KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: vec![KvBlock {
                    block_hash: 42,
                    token_ids: vec![1, 2, 3, 4],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                    ..Default::default()
                }],
                parent_block_hash: None,
                ..Default::default()
            })),
        };

        KvEventMonitor::apply_event(&event, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
    }

    #[test]
    fn apply_event_dispatches_a_removal() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        let stored_event = KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: vec![KvBlock {
                    block_hash: 1,
                    token_ids: vec![1, 2, 3, 4],
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                    ..Default::default()
                }],
                parent_block_hash: None,
                ..Default::default()
            })),
        };
        KvEventMonitor::apply_event(&stored_event, w1, &indexer, &mut wb);

        let removed_event = KvCacheEvent {
            event_id: 2,
            data: Some(kv_cache_event::Data::Removed(KvBlocksRemoved {
                block_hashes: vec![1],
                cache_level: None,
                ..Default::default()
            })),
        };
        KvEventMonitor::apply_event(&removed_event, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn apply_event_dispatches_a_clear() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        KvEventMonitor::apply_event(
            &KvCacheEvent {
                event_id: 1,
                data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                    blocks: vec![KvBlock {
                        block_hash: 1,
                        token_ids: vec![1, 2, 3, 4],
                        block_size: 4,
                        lora_id: None,
                        cache_level: None,
                        ..Default::default()
                    }],
                    parent_block_hash: None,
                    ..Default::default()
                })),
            },
            w1,
            &indexer,
            &mut wb,
        );

        // Clear
        KvEventMonitor::apply_event(
            &KvCacheEvent {
                event_id: 2,
                data: Some(kv_cache_event::Data::Cleared(KvCacheCleared::default())),
            },
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn an_event_without_data_is_ignored() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let event = KvCacheEvent {
            event_id: 1,
            data: None,
        };
        KvEventMonitor::apply_event(&event, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
    }

    const TOKENS: [u32; 4] = [10, 20, 30, 40];

    fn stored_event(hash: i64, tokens: &[u32]) -> KvBlocksStored {
        KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: hash,
                token_ids: tokens.to_vec(),
                block_size: tokens.len() as i32,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn removed_event(hash: i64) -> KvBlocksRemoved {
        KvBlocksRemoved {
            block_hashes: vec![hash],
            ..Default::default()
        }
    }

    fn routable(indexer: &KvIndex, worker: u32, hashes: &[ContentHash]) -> bool {
        indexer
            .find_matches(hashes, false)
            .scores
            .get(&worker)
            .is_some_and(|&depth| depth > 0)
    }

    #[test]
    fn salted_stores_match_only_their_namespace() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let mut stored = stored_event(1, &TOKENS);
        stored.lora_name = Some("adapter".to_string());
        stored.cache_salt = Some("tenant-a".to_string());
        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);

        let same = namespaced_request_content_hashes(&TOKENS, 4, Some("adapter"), Some("tenant-a"));
        assert!(routable(&indexer, w1, &same));
        let plain = compute_request_content_hashes(&TOKENS, 4);
        assert!(!routable(&indexer, w1, &plain));
        let lora_only = namespaced_request_content_hashes(&TOKENS, 4, Some("adapter"), None);
        assert!(!routable(&indexer, w1, &lora_only));
    }

    #[test]
    fn device_removal_keeps_a_block_still_on_the_host() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let hashes = compute_request_content_hashes(&TOKENS, 4);

        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        let mut on_host = stored_event(1, &TOKENS);
        on_host.tier = Some(KvCacheTier::Host as i32);
        KvEventMonitor::apply_stored(&on_host, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);

        KvEventMonitor::apply_removed(&removed_event(1), w1, &indexer, &mut wb);
        assert!(
            routable(&indexer, w1, &hashes),
            "the host copy keeps the block routable"
        );

        let mut from_host = removed_event(1);
        from_host.tier = Some(KvCacheTier::Host as i32);
        KvEventMonitor::apply_removed(&from_host, w1, &indexer, &mut wb);
        assert!(!routable(&indexer, w1, &hashes));
        assert_eq!(indexer.current_size(), 0);
        assert!(wb.copies.is_empty());
    }

    #[test]
    fn cache_level_stands_in_for_the_tier() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let hashes = compute_request_content_hashes(&TOKENS, 4);

        let mut on_host = stored_event(1, &TOKENS);
        on_host.blocks[0].cache_level = Some(1);
        KvEventMonitor::apply_stored(&on_host, w1, &indexer, &mut wb);
        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);

        let mut from_host = removed_event(1);
        from_host.cache_level = Some(1);
        KvEventMonitor::apply_removed(&from_host, w1, &indexer, &mut wb);
        assert!(
            routable(&indexer, w1, &hashes),
            "the device copy keeps the block routable"
        );

        KvEventMonitor::apply_removed(&removed_event(1), w1, &indexer, &mut wb);
        assert!(!routable(&indexer, w1, &hashes));
    }

    #[test]
    fn host_removal_without_a_host_copy_evicts_nothing() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let hashes = compute_request_content_hashes(&TOKENS, 4);

        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        let mut from_host = removed_event(1);
        from_host.tier = Some(KvCacheTier::Host as i32);
        KvEventMonitor::apply_removed(&from_host, w1, &indexer, &mut wb);
        assert!(routable(&indexer, w1, &hashes));
        assert_eq!(wb.counters.unknown_copy, 1);
    }

    #[test]
    fn disk_and_external_tiers_are_counted_not_indexed() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        let mut on_disk = stored_event(1, &TOKENS);
        on_disk.tier = Some(KvCacheTier::Disk as i32);
        KvEventMonitor::apply_stored(&on_disk, w1, &indexer, &mut wb);
        let mut external = stored_event(2, &TOKENS);
        external.blocks[0].cache_level = Some(3);
        KvEventMonitor::apply_stored(&external, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);

        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        let mut from_disk = removed_event(1);
        from_disk.tier = Some(KvCacheTier::Disk as i32);
        KvEventMonitor::apply_removed(&from_disk, w1, &indexer, &mut wb);
        assert_eq!(
            indexer.current_size(),
            1,
            "a disk removal does not touch the device copy"
        );
        assert_eq!(wb.counters.untracked_tier, 3);
    }

    #[test]
    fn non_main_attention_groups_are_skipped_once_their_kind_is_known() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        let mut sliding = stored_event(1, &TOKENS);
        sliding.group_idx = Some(1);
        sliding.kv_cache_spec_kind = Some("sliding_window".to_string());
        KvEventMonitor::apply_stored(&sliding, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);

        let mut full = stored_event(2, &TOKENS);
        full.group_idx = Some(0);
        full.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_stored(&full, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);

        // Later events for group 1 omit the kind; the group is remembered.
        let mut later = stored_event(3, &[50, 60, 70, 80]);
        later.group_idx = Some(1);
        KvEventMonitor::apply_stored(&later, w1, &indexer, &mut wb);
        let mut removal = removed_event(2);
        removal.group_idx = Some(1);
        KvEventMonitor::apply_removed(&removal, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
        assert_eq!(wb.counters.non_main_group, 3);

        removal.group_idx = Some(0);
        KvEventMonitor::apply_removed(&removal, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
    }

    #[test]
    fn remote_and_residency_agent_events_are_skipped() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        let mut remote = stored_event(1, &TOKENS);
        remote.locality = Some(KvCacheLocality::Remote as i32);
        KvEventMonitor::apply_stored(&remote, w1, &indexer, &mut wb);
        let mut agent = stored_event(1, &TOKENS);
        agent.ownership = Some("kvcr".to_string());
        KvEventMonitor::apply_stored(&agent, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);

        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        let cleared = KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Cleared(KvCacheCleared {
                ownership: Some("kvcr".to_string()),
            })),
        };
        KvEventMonitor::apply_event(&cleared, w1, &indexer, &mut wb);
        assert_eq!(
            indexer.current_size(),
            1,
            "an agent's clear leaves the engine's blocks"
        );
        assert_eq!(wb.counters.remote, 1);
        assert_eq!(wb.counters.foreign_owner, 2);
    }

    #[test]
    fn clearing_forgets_copies_and_residency() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let hashes = compute_request_content_hashes(&TOKENS, 4);

        let mut on_host = stored_event(1, &TOKENS);
        on_host.tier = Some(KvCacheTier::Host as i32);
        KvEventMonitor::apply_stored(&on_host, w1, &indexer, &mut wb);
        assert!(!wb.copies.is_empty());

        KvEventMonitor::apply_cleared(w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
        assert!(wb.copies.is_empty());

        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        KvEventMonitor::apply_removed(&removed_event(1), w1, &indexer, &mut wb);
        assert!(
            !routable(&indexer, w1, &hashes),
            "no stale host bit survives a clear"
        );
    }

    #[test]
    fn two_copies_need_two_removals() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let hashes = compute_request_content_hashes(&TOKENS, 4);

        // vLLM recomputes the last block of an exact resend into a second
        // physical copy with the same hash and removes the copies one by one.
        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        assert!(wb.copies.is_empty(), "a single device copy costs no entry");
        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        assert_eq!(wb.counters.duplicate_copies, 1);

        KvEventMonitor::apply_removed(&removed_event(1), w1, &indexer, &mut wb);
        assert!(
            routable(&indexer, w1, &hashes),
            "the other copy is still cached"
        );
        assert_eq!(indexer.current_size(), 1);

        KvEventMonitor::apply_removed(&removed_event(1), w1, &indexer, &mut wb);
        assert!(!routable(&indexer, w1, &hashes));
        assert_eq!(indexer.current_size(), 0);
        assert!(wb.copies.is_empty());
    }

    #[test]
    fn device_and_host_copies_are_counted_per_tier() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let hashes = compute_request_content_hashes(&TOKENS, 4);
        let mut on_host = stored_event(1, &TOKENS);
        on_host.tier = Some(KvCacheTier::Host as i32);
        let mut from_host = removed_event(1);
        from_host.tier = Some(KvCacheTier::Host as i32);

        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        KvEventMonitor::apply_stored(&on_host, w1, &indexer, &mut wb);
        assert_eq!(
            wb.copies[&SequenceHash::from(1i64)],
            Copies { device: 2, host: 1 }
        );

        // A host removal consumes the host copy only.
        KvEventMonitor::apply_removed(&from_host, w1, &indexer, &mut wb);
        assert!(routable(&indexer, w1, &hashes));
        // A second host removal has nothing to take and evicts nothing.
        KvEventMonitor::apply_removed(&from_host, w1, &indexer, &mut wb);
        assert!(routable(&indexer, w1, &hashes));
        assert_eq!(wb.counters.unknown_copy, 1);

        KvEventMonitor::apply_removed(&removed_event(1), w1, &indexer, &mut wb);
        assert!(routable(&indexer, w1, &hashes), "one device copy left");
        KvEventMonitor::apply_removed(&removed_event(1), w1, &indexer, &mut wb);
        assert!(!routable(&indexer, w1, &hashes));
    }

    #[test]
    fn clearing_forgets_copy_counts() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let hashes = compute_request_content_hashes(&TOKENS, 4);

        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        KvEventMonitor::apply_cleared(w1, &indexer, &mut wb);
        assert!(wb.copies.is_empty());

        KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        KvEventMonitor::apply_removed(&removed_event(1), w1, &indexer, &mut wb);
        assert!(
            !routable(&indexer, w1, &hashes),
            "no count survives a clear"
        );
    }

    #[test]
    fn copy_counts_are_capped() {
        let indexer = KvIndex::positional(64);
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let hashes = compute_request_content_hashes(&TOKENS, 4);

        // vLLM's `kv_cache_report_mode: full` re-announces a hit chain on
        // every lookup without removals; the count stops at the cap.
        for _ in 0..20 {
            KvEventMonitor::apply_stored(&stored_event(1, &TOKENS), w1, &indexer, &mut wb);
        }
        assert_eq!(wb.copies[&SequenceHash::from(1i64)].device, COPIES_CAP);
        assert_eq!(wb.counters.capped_copies, 20 - u64::from(COPIES_CAP));

        for _ in 1..COPIES_CAP {
            KvEventMonitor::apply_removed(&removed_event(1), w1, &indexer, &mut wb);
        }
        assert!(routable(&indexer, w1, &hashes));
        KvEventMonitor::apply_removed(&removed_event(1), w1, &indexer, &mut wb);
        assert!(
            !routable(&indexer, w1, &hashes),
            "the cap bounds the extra removals"
        );
    }
}
