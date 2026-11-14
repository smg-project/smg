//! One event into the index: which tiers, localities and owners are indexed,
//! which cache groups go to the index and which to the worker's group view
//! beside it, content hashing under the event's namespace, and the physical
//! copies of a block counted per worker so a removal evicts the block only
//! when its last copy goes.

use std::{
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
};

use kv_index::{
    salt::{content_hash_with_seed, namespace_seed},
    ApplyError, ContentHash, GroupKind, SequenceHash, StoredBlock, WorkerGroups,
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
                match worker_blocks.admit(
                    stored.kv_cache_spec_kind.as_deref(),
                    stored.group_idx,
                    tokenful(stored),
                    stored.locality,
                    stored.ownership.as_deref(),
                ) {
                    // A kind-less store of a group that is the main signal
                    // is the index's, as its tokenful stores are.
                    Admission::Main | Admission::MainAndGroup(_) => {
                        Self::apply_stored(stored, worker_id, indexer, worker_blocks);
                    }
                    Admission::Group(mask) => {
                        Self::apply_group_stored(stored, mask, worker_id, indexer, worker_blocks);
                    }
                    Admission::Skip => {}
                }
            }
            kv_cache_event::Data::Removed(removed) => {
                match worker_blocks.admit(
                    None,
                    removed.group_idx,
                    false,
                    removed.locality,
                    removed.ownership.as_deref(),
                ) {
                    Admission::Main => {
                        if removed
                            .group_idx
                            .is_some_and(|group| worker_blocks.is_main_signal(group))
                        {
                            worker_blocks.counters.window_only_removals += 1;
                        }
                        Self::apply_removed(removed, worker_id, indexer, worker_blocks);
                    }
                    Admission::MainAndGroup(mask) => {
                        worker_blocks.counters.window_only_removals += 1;
                        Self::apply_removed(removed, worker_id, indexer, worker_blocks);
                        Self::apply_group_removed(removed, mask, worker_blocks);
                    }
                    Admission::Group(mask) => {
                        Self::apply_group_removed(removed, mask, worker_blocks);
                    }
                    Admission::Skip => {}
                }
            }
            kv_cache_event::Data::Cleared(cleared) => {
                if worker_blocks.admit(None, None, false, None, cleared.ownership.as_deref())
                    == Admission::Main
                {
                    Self::apply_cleared(worker_id, indexer, worker_blocks);
                }
            }
        }
    }

    /// Apply a batch's events in their order, except that the stores of
    /// cache groups other than main attention come last: the engines
    /// publish them ahead of the main group's store of the same blocks, and
    /// the worker's view places them through the main blocks the index holds.
    /// A batch carrying one gives the worker its view before anything is
    /// applied, so the batch's main stores are named in it whichever came
    /// first on the wire, and shows the batch's groups first, so a kind-less
    /// event of a group earlier in the batch than the group's first store is
    /// the group's.
    pub(crate) fn apply_events(
        events: &[KvCacheEvent],
        worker_id: u32,
        indexer: &KvIndex,
        worker_blocks: &mut WorkerIndexState,
    ) {
        // The main group first, so a window group's store ahead of it in
        // the batch reads as a group's, not as the worker's main signal. A
        // store the index skips (a residency agent's, a remote tier's) shows
        // no group here either.
        let mut hybrid = false;
        for event in events {
            if let Some(kv_cache_event::Data::Stored(stored)) = &event.data {
                if foreign(stored) {
                    continue;
                }
                let main = match (stored.kv_cache_spec_kind.as_deref(), stored.group_idx) {
                    (Some(kind), _) => MAIN_ATTENTION_KINDS.contains(&kind),
                    (None, None) => true,
                    (None, Some(_)) => false,
                };
                if main && Self::main_group_shown(worker_id, indexer, worker_blocks) {
                    hybrid = true;
                }
            }
        }
        for event in events {
            if let Some(kv_cache_event::Data::Stored(stored)) = &event.data {
                if foreign(stored) {
                    continue;
                }
                if let (Some(kind), Some(group)) =
                    (stored.kv_cache_spec_kind.as_deref(), stored.group_idx)
                {
                    if MAIN_ATTENTION_KINDS.contains(&kind) {
                        continue;
                    }
                    if worker_blocks.main_shown || !tokenful(stored) {
                        worker_blocks.show_group(group);
                        hybrid = true;
                    } else {
                        worker_blocks.main_signal(group);
                    }
                }
            }
        }
        if hybrid && worker_blocks.groups.is_none() {
            let indexed = worker_blocks.blocks.len();
            if indexed > 0 {
                worker_blocks.counters.group_view_late += 1;
                warn!(
                    worker_id,
                    indexed,
                    "KV cache group view created after the worker's main blocks were indexed; its rule stays off until the worker's next clear or a restart leaves the view complete"
                );
            }
            worker_blocks.view(indexer, worker_id).created_over(indexed);
        }
        let mut deferred = Vec::new();
        for event in events {
            if let Some(kv_cache_event::Data::Stored(stored)) = &event.data {
                if worker_blocks.is_group_store(stored) {
                    deferred.push(event);
                    continue;
                }
            }
            Self::apply_event(event, worker_id, indexer, worker_blocks);
        }
        for event in deferred {
            Self::apply_event(event, worker_id, indexer, worker_blocks);
        }
    }

    /// Convert proto `KvBlocksStored` and apply to the indexer.
    ///
    /// Blocks on the disk and external tiers are counted and skipped, as is a
    /// store of blocks narrower than the worker's block of record
    /// ([`WorkerIndexState::sub_block`]). Content hashes are computed under
    /// the event's LoRA name and cache salt, so a salted block matches only a
    /// request hashed under the same namespace. A worker with a group view
    /// learns the blocks' names there too, as the index placed them.
    fn apply_stored(
        stored: &KvBlocksStored,
        worker_id: u32,
        indexer: &KvIndex,
        worker_blocks: &mut WorkerIndexState,
    ) {
        let first_level = stored.blocks.first().and_then(|block| block.cache_level);
        let Some(tier) = indexed_tier(stored.tier, first_level) else {
            worker_blocks.counters.untracked_tier += 1;
            return;
        };
        if worker_blocks.sub_block(worker_id, &stored.blocks) {
            return;
        }
        if stored.blocks.iter().any(|block| block.token_ids.is_empty()) {
            // A main block without tokens has no content hash to index under;
            // a group's hashes-only store that lost its group reads so too.
            worker_blocks.counters.tokenless_stores += 1;
            return;
        }

        let seed = namespace_seed(stored.lora_name.as_deref(), stored.cache_salt.as_deref());
        let blocks: Vec<StoredBlock> = stored
            .blocks
            .iter()
            .map(|block| convert_kv_block(block, seed))
            .collect();
        worker_blocks.note_stored(&blocks, tier);

        let parent_seq_hash = stored.parent_block_hash.map(SequenceHash::from);

        let placed_after = match indexer.apply_stored(
            worker_id,
            &blocks,
            parent_seq_hash,
            &mut worker_blocks.blocks,
        ) {
            Ok(()) => Some((parent_seq_hash, blocks.len())),
            Err(ApplyError::WorkerNotTracked | ApplyError::ParentBlockNotFound) => {
                // Cold start or parent evicted — retry without parent to start a new chain.
                worker_blocks.counters.parentless_stores += 1;
                worker_blocks.counters.parentless_blocks += blocks.len() as u64;
                match indexer.apply_stored(worker_id, &blocks, None, &mut worker_blocks.blocks) {
                    Ok(()) => Some((None, blocks.len())),
                    Err(ApplyError::ArenaCapReached { placed }) => {
                        Self::note_cap_refusal(worker_blocks, worker_id, blocks.len() - placed);
                        Some((None, placed))
                    }
                    Err(e) => {
                        warn!(
                            worker_id = worker_id,
                            error = %e,
                            "Failed to apply stored event after fallback"
                        );
                        None
                    }
                }
            }
            Err(ApplyError::ArenaCapReached { placed }) => {
                Self::note_cap_refusal(worker_blocks, worker_id, blocks.len() - placed);
                Some((parent_seq_hash, placed))
            }
        };
        // A store the index declined at an arena cap placed its leading
        // blocks alone: the group view records those, as the index holds them.
        if let (Some((parent, placed)), Some(view)) = (placed_after, &worker_blocks.groups) {
            let named: Vec<(SequenceHash, ContentHash)> = blocks
                .iter()
                .take(placed)
                .map(|block| (block.seq_hash, block.content_hash))
                .collect();
            worker_blocks.counters.group_blocks += view.main_stored(parent, &named) as u64;
            worker_blocks.counters.group_unnamed = view.unnamed_blocks();
        }
    }

    /// A store the index declined at an arena cap (a run's fan-out or length) from its
    /// `refused` last blocks on: the engine keeps them, the index does not route by them.
    /// Counted per worker (published beside the parent-less counters), said once per worker.
    fn note_cap_refusal(worker_blocks: &mut WorkerIndexState, worker_id: u32, refused: usize) {
        worker_blocks.counters.cap_refused_stores += 1;
        worker_blocks.counters.cap_refused_blocks += refused as u64;
        if worker_blocks.counters.cap_refused_stores == 1 {
            warn!(
                worker_id = worker_id,
                blocks = refused,
                "KV index declined a store at an arena cap (a run's fan-out or length); \
                 further refusals for this worker are counted, not logged"
            );
        }
    }

    /// A store of cache groups other than main attention (`mask`, one bit
    /// per group index: one bit for an engine that publishes per group, the
    /// groups keeping a boundary's hashes for one that publishes them once):
    /// its blocks are placed in the worker's view beside the index, through
    /// the main blocks the index holds ([`WorkerGroups::stored`]). A store
    /// narrower than the worker's block of record is a prefix entry at the
    /// engine's hash granularity, counted with the main group's.
    fn apply_group_stored(
        stored: &KvBlocksStored,
        mask: u16,
        worker_id: u32,
        indexer: &KvIndex,
        worker_blocks: &mut WorkerIndexState,
    ) {
        let first_level = stored.blocks.first().and_then(|block| block.cache_level);
        if indexed_tier(stored.tier, first_level).is_none() {
            worker_blocks.counters.untracked_tier += 1;
            return;
        }
        let width = stored
            .blocks
            .first()
            .map_or(0, |block| block.block_size)
            .max(0);
        if width > 0 && worker_blocks.block_of_record > 0 && width < worker_blocks.block_of_record {
            worker_blocks.counters.sub_block_stores += 1;
            worker_blocks.counters.sub_block_blocks += stored.blocks.len() as u64;
            return;
        }
        let view = worker_blocks.view(indexer, worker_id);
        // A store of one group names its kind; a masked store names the
        // groups alone (their kinds come with their own stores).
        let kind = (mask.count_ones() == 1)
            .then(|| {
                group_kind(
                    stored.kv_cache_spec_kind.as_deref(),
                    stored.kv_cache_spec_sliding_window,
                )
            })
            .flatten();
        for group in (0..u16::BITS).filter(|group| mask & (1 << group) != 0) {
            view.learn(group, kind, width as u32);
        }
        let hashes: Vec<SequenceHash> = stored
            .blocks
            .iter()
            .map(|block| SequenceHash::from(block.block_hash))
            .collect();
        let (placed, unplaced) = view.stored(mask, &hashes);
        worker_blocks.counters.group_stores += 1;
        worker_blocks.counters.group_blocks += placed as u64;
        worker_blocks.counters.group_unplaced += unplaced as u64;
        worker_blocks.counters.group_dropped = view.dropped_blocks();
    }

    /// A removal from cache groups other than main attention (`mask` as in
    /// [`Self::apply_group_stored`]): the groups let go of the blocks in the
    /// worker's view.
    fn apply_group_removed(
        removed: &KvBlocksRemoved,
        mask: u16,
        worker_blocks: &mut WorkerIndexState,
    ) {
        if indexed_tier(removed.tier, removed.cache_level).is_none() {
            worker_blocks.counters.untracked_tier += 1;
            return;
        }
        worker_blocks.counters.group_removals += 1;
        if let Some(view) = &worker_blocks.groups {
            let hashes: Vec<SequenceHash> = removed
                .block_hashes
                .iter()
                .map(|&hash| SequenceHash::from(hash))
                .collect();
            view.removed(mask, &hashes);
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
        if let Some(view) = &worker_blocks.groups {
            view.main_removed(&seq_hashes);
        }
    }

    /// A main-attention store, or a plain worker's, in the batch: the worker
    /// has a main group. The first on a worker whose main signal was a
    /// window group restarts it: the index lets the signal's blocks go
    /// (counted; the main group's stores rebuild it, in this very batch for
    /// an engine whose group lane merely ran ahead of its main lane), the
    /// worker's view restarts with it (made now when the worker has none;
    /// a hashes-only store of the group gave it one before, created late and
    /// incomplete): its names go, its memberships and the signal's own
    /// blocks, every one of them the signal group's, wait under the worker's
    /// hashes for the main group's names ([`WorkerGroups::restarted`],
    /// [`WorkerGroups::waiting`]), and it is complete from here; two signal
    /// groups leave whose block is whose unknown, and the view incomplete,
    /// the overlap standing until a clear, counted and logged as a late
    /// view is. The signal's groups move to the view, whose events they
    /// publish from here on. `true` when the worker needs its view.
    fn main_group_shown(
        worker_id: u32,
        indexer: &KvIndex,
        worker_blocks: &mut WorkerIndexState,
    ) -> bool {
        if worker_blocks.main_shown {
            return false;
        }
        worker_blocks.main_shown = true;
        if worker_blocks.main_signal_groups.is_empty() {
            return false;
        }
        let signal = worker_blocks.blocks.hashes();
        indexer.apply_cleared(worker_id, &mut worker_blocks.blocks);
        worker_blocks.copies.clear();
        worker_blocks.counters.main_signal_dropped += signal.len() as u64;
        let groups = std::mem::take(&mut worker_blocks.main_signal_groups);
        let view = worker_blocks.view(indexer, worker_id);
        view.restarted();
        match groups.as_slice() {
            [group] if group_mask(*group) != 0 => {
                view.waiting(group_mask(*group), &signal);
                worker_blocks.counters.group_dropped = view.dropped_blocks();
            }
            // Whose block is whose is unknown (or the one group is past the
            // rule's reach): the rule stays off, on the record as a late
            // view is.
            _ => {
                if !signal.is_empty() {
                    worker_blocks.counters.group_view_late += 1;
                    warn!(
                        worker_id,
                        main_signal_dropped = signal.len(),
                        groups = groups.len(),
                        "KV cache group view restarted over the blocks of more than one main-signal group, or of one the rule cannot read; its rule stays off until the worker's next clear"
                    );
                }
                view.created_over(signal.len());
            }
        }
        for group in groups {
            worker_blocks.show_group(group);
        }
        true
    }

    /// Drop every block of a worker from the indexer and forget its copies
    /// and its block of record: a clear is the worker's new cache incarnation
    /// (its publisher restarted or resynced), and a worker that comes back
    /// with a narrower block than it stored before must index its stores as
    /// they come, not count every one as a sub-block store. The dropped
    /// record is kept until that first store, which is logged when it is
    /// narrower ([`WorkerIndexState::sub_block`]).
    pub(super) fn apply_cleared(
        worker_id: u32,
        indexer: &KvIndex,
        worker_blocks: &mut WorkerIndexState,
    ) {
        indexer.apply_cleared(worker_id, &mut worker_blocks.blocks);
        if let Some(view) = &worker_blocks.groups {
            view.clear();
        }
        worker_blocks.copies.clear();
        worker_blocks.record_before_clear = worker_blocks.block_of_record;
        worker_blocks.block_of_record = 0;
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

/// Cache-group kinds whose blocks hold the main attention KV, the ones the
/// prefix index is built from. Every other kind goes to the worker's group
/// view, where the sliding-window and state kinds bound a hit the way the
/// engine does ([`WorkerGroups::cap`]).
const MAIN_ATTENTION_KINDS: [&str; 3] = ["full_attention", "mla_attention", "sink_full_attention"];

/// A store or removal a residency agent (`kvcr`) owns: skipped and counted
/// by [`WorkerIndexState::admit`], read by neither batch pre-scan.
fn foreign_owner(ownership: Option<&str>) -> bool {
    ownership.is_some_and(|owner| owner.eq_ignore_ascii_case("kvcr"))
}

/// An event about blocks on a remote tier: skipped like a foreign owner's.
fn remote(locality: Option<i32>) -> bool {
    locality.is_some_and(|locality| locality == KvCacheLocality::Remote as i32)
}

/// Whether [`WorkerIndexState::admit`] skips a store before it reads the
/// store's group: the batch pre-scans skip it too, so a foreign store shows
/// neither a main group nor a window group.
fn foreign(stored: &KvBlocksStored) -> bool {
    foreign_owner(stored.ownership.as_deref()) || remote(stored.locality)
}

/// Whether every block of a store carries its tokens: a main-lane store
/// does, a group lane's store (hashes only) does not.
fn tokenful(stored: &KvBlocksStored) -> bool {
    !stored.blocks.is_empty()
        && stored
            .blocks
            .iter()
            .all(|block| !block.token_ids.is_empty())
}

/// Cache-group kinds that keep a sliding window of the sequence.
const WINDOW_KINDS: [&str; 2] = ["sliding_window", "sliding_window_mla"];

/// The cache-group kind that keeps a recurrent state per block.
const STATE_KIND: &str = "mamba";

/// What a store's group kind means to the reuse rule; `None` when the store
/// names no kind. A sliding-window group's window comes with its kind.
fn group_kind(kind: Option<&str>, window: Option<u32>) -> Option<GroupKind> {
    let kind = kind?;
    Some(if MAIN_ATTENTION_KINDS.contains(&kind) {
        GroupKind::Main
    } else if WINDOW_KINDS.contains(&kind) {
        GroupKind::SlidingWindow {
            window: window.unwrap_or(0),
        }
    } else if kind == STATE_KIND {
        GroupKind::State
    } else {
        GroupKind::Unknown
    })
}

/// Where an event goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Admission {
    /// The prefix index's: a main-attention store, a plain worker's event,
    /// or the main signal of a worker without a main group.
    Main,
    /// The worker's view's, under the groups' mask.
    Group(u16),
    /// A removal of a group that is the worker's main signal while the view
    /// knows it too (a hashes-only store of it came before any main group):
    /// both hold its blocks, both let go.
    MainAndGroup(u16),
    /// Counted, not applied.
    Skip,
}

/// The mask bit of a group index; a group past the view's last bit is
/// counted as unplaced there.
fn group_mask(group: u32) -> u16 {
    1u16.checked_shl(group).unwrap_or(0)
}

/// The most physical copies of one block counted per tier. vLLM's opt-in
/// `kv_cache_report_mode: full` re-announces whole hit chains without
/// removals, which would grow a count without bound; the cap turns that into
/// at most this many extra removals before the block leaves the index.
const COPIES_CAP: u8 = 8;

/// What the index did not take at face value, by reason; logged
/// when the worker's subscription ends.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerIndexCounters {
    /// Stores and removals on tiers the index does not track.
    pub(crate) untracked_tier: u64,
    /// Stores and removals of cache groups other than main attention, kept
    /// in the worker's group view beside the index; the blocks of those
    /// stores the view placed through the index, and the ones it could not
    /// when the store came (their main block was not indexed yet: they wait
    /// in the view and count as placed when it comes).
    pub(crate) group_stores: u64,
    pub(crate) group_blocks: u64,
    pub(crate) group_unplaced: u64,
    pub(crate) group_removals: u64,
    /// Stores of a kind other than main attention that name no group.
    pub(crate) group_nameless: u64,
    /// Group blocks the view's wait refused, and main blocks it could not
    /// name under a parent it did not know (the view's own counts, copied).
    pub(crate) group_dropped: u64,
    pub(crate) group_unnamed: u64,
    /// Views whose rule went off when they were made or restarted: created
    /// after the worker's main blocks were indexed (off until the worker's
    /// next clear, or until a restart leaves the view complete), or
    /// restarted over the blocks of more than one main-signal group (or of
    /// one the rule cannot read), off until the worker's next clear.
    pub(crate) group_view_late: u64,
    /// Main stores with a block that carried no tokens: nothing to index under.
    pub(crate) tokenless_stores: u64,
    /// Stores of a group other than main attention that carried tokens on a
    /// worker that has shown no main group: the worker's main signal, the
    /// prefix index's.
    pub(crate) window_only_stores: u64,
    /// Kind-less removals of such a group: they leave the prefix index.
    pub(crate) window_only_removals: u64,
    /// Blocks the prefix index let go when such a worker showed a main
    /// group: the main group's stores rebuild it.
    pub(crate) main_signal_dropped: u64,
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
    /// Stores the index declined because a run's child table or hash array would outgrow
    /// an arena chunk, and the blocks they left out (the leading blocks a declined store
    /// placed are not among them); `smg_kv_index_cap_refusals` counts the stores over the
    /// index, these are per worker.
    pub(crate) cap_refused_stores: u64,
    pub(crate) cap_refused_blocks: u64,
    /// Stores of blocks narrower than the worker's block of record, and the
    /// blocks they carried: a hybrid engine's prefix entries at its hash
    /// granularity (the prompt's tail inside its last block), or an engine's
    /// last short page. Not indexed: a request is chunked at the block of
    /// record and never matches one, and their parents are intermediate
    /// hashes the engine does not publish, so indexed they would read as
    /// parent-less stores and seed root chains that only collide.
    pub(crate) sub_block_stores: u64,
    pub(crate) sub_block_blocks: u64,
}

impl WorkerIndexCounters {
    /// Parent-less stores and blocks this state saw since `before`.
    pub(super) fn parentless_since(&self, before: &Self) -> (u64, u64) {
        (
            self.parentless_stores - before.parentless_stores,
            self.parentless_blocks - before.parentless_blocks,
        )
    }

    /// Stores declined at an arena cap and the blocks they left out since `before`.
    pub(super) fn cap_refused_since(&self, before: &Self) -> (u64, u64) {
        (
            self.cap_refused_stores - before.cap_refused_stores,
            self.cap_refused_blocks - before.cap_refused_blocks,
        )
    }

    /// Sub-block stores and blocks this state saw since `before`.
    pub(super) fn sub_block_since(&self, before: &Self) -> (u64, u64) {
        (
            self.sub_block_stores - before.sub_block_stores,
            self.sub_block_blocks - before.sub_block_blocks,
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

/// A worker's share of the index: its reverse map plus
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
    /// The worker's cache groups other than main attention, in the index's
    /// registry: created at the worker's first batch carrying a store of
    /// such a group.
    groups: Option<Arc<WorkerGroups>>,
    /// The cache groups other than main attention the worker has shown, by
    /// index, independent of the view and of its bit limit: a kind-less
    /// event of one of them (a removal never names its kind) is the group's,
    /// never the main group's.
    non_main_groups: Vec<u32>,
    /// Whether the worker has shown a main-attention group: a store with a
    /// main-attention kind, or one naming no kind and no group (an engine
    /// with one group names nothing). Until it has, a store of another group
    /// that carries tokens is the worker's main signal, the shape the relay
    /// gives a rank whose attention is all sliding window (a group lane's
    /// store carries no tokens): the prefix index takes it as a plain
    /// worker's. Learned per batch ahead of its events and kept across
    /// clears, like the groups, so a snapshot replay reads as the stream did.
    main_shown: bool,
    /// The groups whose tokenful stores are the worker's main signal: a
    /// kind-less removal of one leaves the prefix index (and the view, when
    /// a hashes-only store of the group gave the worker a view before any
    /// main group). Drained into the view's groups when a main group shows
    /// ([`KvEventMonitor::main_group_shown`]).
    main_signal_groups: Vec<u32>,
    /// Tokens per block of the widest block the worker's stores carried so
    /// far, 0 until a store names a width: the worker's block of record.
    block_of_record: i32,
    /// The block of record a clear dropped, until the worker's first store
    /// after it names a width: a narrower one is logged, since the model-level
    /// size that chunks requests keeps the old width.
    record_before_clear: i32,
    pub(crate) counters: WorkerIndexCounters,
}

impl WorkerIndexState {
    /// Where an event with these attributes goes. A store names its group's
    /// kind; later events for that group may omit it, so the groups the view
    /// learned decide for them, and an unlearned group without a kind is the
    /// main group (an engine with one group names nothing). A store of a
    /// group other than main attention that carries tokens, on a worker that
    /// has shown no main group, is the worker's main signal
    /// ([`Self::main_shown`]): the index's, and its group's kind-less
    /// removals follow it there.
    fn admit(
        &mut self,
        kind: Option<&str>,
        group_idx: Option<u32>,
        tokenful: bool,
        locality: Option<i32>,
        ownership: Option<&str>,
    ) -> Admission {
        if foreign_owner(ownership) {
            self.counters.foreign_owner += 1;
            return Admission::Skip;
        }
        if remote(locality) {
            self.counters.remote += 1;
            return Admission::Skip;
        }
        match (kind, group_idx) {
            (Some(kind), group) if MAIN_ATTENTION_KINDS.contains(&kind) => {
                self.main_shown = true;
                if let Some(group) = group {
                    self.non_main_groups.retain(|&known| known != group);
                }
                Admission::Main
            }
            (Some(_), Some(group)) if tokenful && !self.main_shown => {
                self.main_signal(group);
                self.counters.window_only_stores += 1;
                Admission::Main
            }
            (Some(_), Some(group)) => {
                self.show_group(group);
                Admission::Group(group_mask(group))
            }
            (Some(_), None) => {
                self.counters.group_nameless += 1;
                Admission::Skip
            }
            (None, Some(group)) => match (self.is_main_signal(group), self.knows_group(group)) {
                (true, true) => Admission::MainAndGroup(group_mask(group)),
                (false, true) => Admission::Group(group_mask(group)),
                _ => Admission::Main,
            },
            (None, None) => Admission::Main,
        }
    }

    /// Whether the worker has shown `group` as other than main attention.
    fn knows_group(&self, group: u32) -> bool {
        self.non_main_groups.contains(&group)
    }

    /// A store named `group` with a kind other than main attention.
    fn show_group(&mut self, group: u32) {
        if !self.non_main_groups.contains(&group) {
            self.non_main_groups.push(group);
        }
    }

    /// Whether tokenful stores of `group` were the worker's main signal.
    fn is_main_signal(&self, group: u32) -> bool {
        self.main_signal_groups.contains(&group)
    }

    /// A tokenful store of `group` on a worker that has shown no main group.
    fn main_signal(&mut self, group: u32) {
        if !self.main_signal_groups.contains(&group) {
            self.main_signal_groups.push(group);
        }
    }

    /// Whether a store belongs to a cache group other than main attention
    /// ([`Self::admit`] without the counting): the stores a batch applies
    /// last.
    fn is_group_store(&self, stored: &KvBlocksStored) -> bool {
        match (stored.kv_cache_spec_kind.as_deref(), stored.group_idx) {
            (Some(kind), Some(_)) => {
                !MAIN_ATTENTION_KINDS.contains(&kind) && (self.main_shown || !tokenful(stored))
            }
            (None, Some(group)) => self.knows_group(group) && !self.is_main_signal(group),
            _ => false,
        }
    }

    /// The worker's group view, created in the index's registry at its first
    /// batch carrying a store of a cache group other than main attention.
    fn view(&mut self, indexer: &KvIndex, worker_id: u32) -> Arc<WorkerGroups> {
        Arc::clone(
            self.groups
                .get_or_insert_with(|| indexer.groups().get_or_create(worker_id)),
        )
    }

    /// Whether a store names blocks narrower than the worker's block of
    /// record, the widest block its stores carried so far; a wider store
    /// raises the record first. Hybrid engines publish, beside the blocks of
    /// their cache group, prefix entries at their hash granularity (a
    /// fraction of the block, for a prefix boundary a later request can
    /// resume at) under parents they never publish; the index can neither
    /// match them (requests are chunked at the block of record) nor hang
    /// them on the chain, so they are counted and left out. The engines
    /// write one width per store, so the first block's width is the store's;
    /// a width of zero is unknown and neither raises nor trips the record.
    fn sub_block(&mut self, worker_id: u32, blocks: &[KvBlock]) -> bool {
        let Some(width) = blocks
            .first()
            .map(|block| block.block_size)
            .filter(|&width| width > 0)
        else {
            return false;
        };
        if self.record_before_clear > 0 {
            if width < self.record_before_clear {
                warn!(
                    worker_id,
                    before = self.record_before_clear,
                    after = width,
                    "the first KV event store after a clear names a narrower block than the \
                     worker's previous record: its stores are indexed, but requests are chunked \
                     at the model's learned block size until the model's last worker leaves"
                );
            }
            self.record_before_clear = 0;
        }
        if width > self.block_of_record {
            self.block_of_record = width;
        } else if width < self.block_of_record {
            self.counters.sub_block_stores += 1;
            self.counters.sub_block_blocks += blocks.len() as u64;
            return true;
        }
        false
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
        salt::namespaced_request_content_hashes, ContentHash, GroupCap, XXH3_SEED,
    };
    use smg_grpc_client::common_proto::KvCacheCleared;

    use super::*;

    #[test]
    fn a_proto_block_becomes_its_engine_and_content_hashes() {
        let proto = block(42, &[1, 2, 3, 4]);
        let stored = convert_kv_block(&proto, XXH3_SEED);
        assert_eq!(stored.seq_hash, SequenceHash::from(42i64));
        assert_eq!(stored.content_hash, compute_content_hash(&[1, 2, 3, 4]));
    }

    #[test]
    fn a_negative_engine_hash_keeps_its_bits() {
        let proto = block(-1, &[10, 20]);
        let stored = convert_kv_block(&proto, XXH3_SEED);
        assert_eq!(stored.seq_hash, SequenceHash(u64::MAX));
    }

    #[test]
    fn a_block_without_tokens_hashes_the_empty_content() {
        let proto = block(100, &[]);
        let stored = convert_kv_block(&proto, XXH3_SEED);
        assert_eq!(stored.seq_hash, SequenceHash::from(100i64));
        assert_eq!(stored.content_hash, compute_content_hash(&[]));
    }

    #[test]
    fn a_root_store_indexes_its_blocks() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let stored = KvBlocksStored {
            blocks: vec![block(1, &[10, 20, 30, 40]), block(2, &[50, 60, 70, 80])],
            parent_block_hash: None,
            ..Default::default()
        };

        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2);
    }

    #[test]
    fn a_chained_store_extends_its_parent() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        let stored1 = KvBlocksStored {
            blocks: vec![block(1, &[10, 20, 30, 40])],
            parent_block_hash: None,
            ..Default::default()
        };
        KvEventMonitor::apply_stored(&stored1, w1, &indexer, &mut wb);

        let stored2 = KvBlocksStored {
            blocks: vec![block(2, &[50, 60, 70, 80])],
            parent_block_hash: Some(1),
            ..Default::default()
        };
        KvEventMonitor::apply_stored(&stored2, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2);
    }

    #[test]
    fn a_store_with_an_unknown_parent_starts_a_new_chain() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://new-worker:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        // Pass parent_block_hash for an untracked worker — should fallback to no parent.
        let stored = KvBlocksStored {
            blocks: vec![block(1, &[10, 20, 30, 40])],
            parent_block_hash: Some(999),
            ..Default::default()
        };
        KvEventMonitor::apply_stored(&stored, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
    }

    #[test]
    fn a_removal_takes_the_block_out() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        let stored = KvBlocksStored {
            blocks: vec![block(1, &[10, 20, 30, 40]), block(2, &[50, 60, 70, 80])],
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
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        let stored = KvBlocksStored {
            blocks: vec![block(1, &[10, 20, 30, 40])],
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
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let event = KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: vec![block(42, &[1, 2, 3, 4])],
                parent_block_hash: None,
                ..Default::default()
            })),
        };

        KvEventMonitor::apply_event(&event, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
    }

    #[test]
    fn apply_event_dispatches_a_removal() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        let stored_event = KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: vec![block(1, &[1, 2, 3, 4])],
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
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        KvEventMonitor::apply_event(
            &KvCacheEvent {
                event_id: 1,
                data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                    blocks: vec![block(1, &[1, 2, 3, 4])],
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
        let indexer = KvIndex::chain();
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

    /// A block of these tokens at this hash, as wide as its tokens, the rest default.
    fn block(hash: i64, tokens: &[u32]) -> KvBlock {
        KvBlock {
            block_hash: hash,
            token_ids: tokens.to_vec(),
            block_size: tokens.len() as i32,
            ..Default::default()
        }
    }

    fn stored_event(hash: i64, tokens: &[u32]) -> KvBlocksStored {
        KvBlocksStored {
            blocks: vec![block(hash, tokens)],
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
        let indexer = KvIndex::chain();
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
        let indexer = KvIndex::chain();
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
        let indexer = KvIndex::chain();
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
        let indexer = KvIndex::chain();
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
        let indexer = KvIndex::chain();
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

    fn store_event(stored: KvBlocksStored) -> KvCacheEvent {
        KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Stored(stored)),
        }
    }

    fn remove_event(removed: KvBlocksRemoved) -> KvCacheEvent {
        KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Removed(removed)),
        }
    }

    /// A store of `group` of `kind` naming `hashes` without tokens, the way
    /// the relay forwards a group other than main attention.
    fn group_store(
        group: u32,
        kind: Option<&str>,
        window: Option<u32>,
        hashes: &[i64],
    ) -> KvCacheEvent {
        store_event(KvBlocksStored {
            blocks: hashes
                .iter()
                .map(|&hash| KvBlock {
                    block_hash: hash,
                    token_ids: Vec::new(),
                    block_size: 4,
                    lora_id: None,
                    cache_level: None,
                    ..Default::default()
                })
                .collect(),
            parent_block_hash: None,
            group_idx: Some(group),
            kv_cache_spec_kind: kind.map(str::to_string),
            kv_cache_spec_sliding_window: window,
            ..Default::default()
        })
    }

    /// The index's overlap for `hashes` on `worker`, after the cache groups'
    /// rule, with the holders the rule shrank.
    fn capped(
        indexer: &KvIndex,
        worker: u32,
        hashes: &[ContentHash],
        block_size: usize,
    ) -> (Option<u32>, Vec<GroupCap>) {
        let mut overlap = indexer.find_matches(hashes, false);
        let caps = indexer.cap_by_groups(hashes, block_size, &mut overlap);
        (overlap.scores.get(&worker).copied(), caps)
    }

    /// A hybrid worker: the window group's store precedes the main store of
    /// the same blocks in a batch and is placed through it; a kind-less event
    /// on the learned group follows the group; the main group's chain is the
    /// index's, the window group's blocks the view's.
    #[test]
    fn non_main_attention_groups_are_kept_in_the_worker_view() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let mut full = stored_event(1, &TOKENS);
        full.group_idx = Some(0);
        full.kv_cache_spec_kind = Some("full_attention".to_string());
        let mut second = stored_event(2, &[50, 60, 70, 80]);
        second.parent_block_hash = Some(1);
        let batch = [
            group_store(1, Some("sliding_window"), Some(5), &[1, 2]),
            store_event(full),
            store_event(second),
        ];
        KvEventMonitor::apply_events(&batch, w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2, "the main chain alone is indexed");
        assert_eq!(indexer.groups().len(), 1);
        let view = indexer.groups().get(w1).expect("the view");
        assert_eq!(view.retained_blocks(), 2, "both window blocks placed");
        assert_eq!(
            (
                wb.counters.group_stores,
                wb.counters.group_blocks,
                wb.counters.group_unplaced
            ),
            (1, 2, 0)
        );

        let hashes = compute_request_content_hashes(&[10, 20, 30, 40, 50, 60, 70, 80], 4);
        assert_eq!(capped(&indexer, w1, &hashes, 4), (Some(2), Vec::new()));

        // A kind-less removal on the learned group takes the block out of the
        // view, not the index: a window of 5 tokens over 4-token blocks needs
        // one block ending at the hit; block 2 gone, the hit ends at block 1.
        let mut gone = removed_event(2);
        gone.group_idx = Some(1);
        KvEventMonitor::apply_event(&remove_event(gone), w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2);
        assert_eq!(view.retained_blocks(), 1);
        assert_eq!(wb.counters.group_removals, 1);
        let (depth, caps) = capped(&indexer, w1, &hashes, 4);
        assert_eq!(depth, Some(1));
        assert_eq!(
            caps,
            vec![GroupCap {
                worker: w1,
                uncapped: 2,
                capped: 1,
                group: 1
            }]
        );

        // A kind-less store on the learned group follows the group.
        let mut later = stored_event(2, &[]);
        later.group_idx = Some(1);
        later.blocks[0].token_ids.clear();
        KvEventMonitor::apply_event(&store_event(later), w1, &indexer, &mut wb);
        assert_eq!(view.retained_blocks(), 2);
        assert_eq!(capped(&indexer, w1, &hashes, 4).0, Some(2));

        // A group store naming blocks the index does not hold is counted, not placed.
        KvEventMonitor::apply_event(&group_store(1, None, None, &[77]), w1, &indexer, &mut wb);
        assert_eq!(
            (wb.counters.group_blocks, wb.counters.group_unplaced),
            (3, 1)
        );

        // The main group's removal keeps the window block's name while the
        // window retains it; a clear empties the view with the index.
        let mut main_gone = removed_event(2);
        main_gone.group_idx = Some(0);
        KvEventMonitor::apply_event(&remove_event(main_gone), w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
        assert_eq!(view.named_blocks(), 2);
        KvEventMonitor::apply_cleared(w1, &indexer, &mut wb);
        assert_eq!((view.retained_blocks(), view.named_blocks()), (0, 0));
        assert!(view.knows(1));

        // The worker's removal drops its view.
        indexer.remove_worker(w1, wb.blocks);
        assert!(indexer.groups().is_empty());
    }

    /// The main store first in the batch (an engine that publishes the main
    /// group first), or in a batch of its own after the window group's store
    /// (a relay snapshot replays the group lanes first): the window's blocks
    /// are placed either way, in the batch that names them.
    #[test]
    fn a_group_store_is_placed_whichever_side_of_the_main_store_it_comes() {
        let hashes = compute_request_content_hashes(&[10, 20, 30, 40, 50, 60, 70, 80], 4);
        let mut second = stored_event(2, &[50, 60, 70, 80]);
        second.parent_block_hash = Some(1);

        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        KvEventMonitor::apply_events(
            &[
                store_event(stored_event(1, &TOKENS)),
                store_event(second.clone()),
                group_store(1, Some("sliding_window"), Some(5), &[1, 2]),
            ],
            w1,
            &indexer,
            &mut wb,
        );
        let view = indexer.groups().get(w1).expect("the view");
        assert_eq!((view.retained_blocks(), view.pending_blocks()), (2, 0));
        assert_eq!(
            (wb.counters.group_blocks, wb.counters.group_unplaced),
            (2, 0)
        );
        assert_eq!(capped(&indexer, w1, &hashes, 4), (Some(2), Vec::new()));

        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        KvEventMonitor::apply_events(
            &[group_store(1, Some("sliding_window"), Some(5), &[1, 2])],
            w1,
            &indexer,
            &mut wb,
        );
        let view = indexer.groups().get(w1).expect("the view");
        assert_eq!((view.retained_blocks(), view.pending_blocks()), (0, 2));
        assert_eq!(
            (wb.counters.group_blocks, wb.counters.group_unplaced),
            (0, 2),
            "counted as they wait"
        );
        KvEventMonitor::apply_events(
            &[store_event(stored_event(1, &TOKENS))],
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(
            capped(&indexer, w1, &hashes, 4),
            (Some(1), Vec::new()),
            "one block named: the leading run"
        );
        KvEventMonitor::apply_events(&[store_event(second)], w1, &indexer, &mut wb);
        assert_eq!((view.retained_blocks(), view.pending_blocks()), (2, 0));
        assert_eq!(
            (wb.counters.group_blocks, wb.counters.group_unplaced),
            (2, 2),
            "placed as the main stores came"
        );
        assert_eq!(capped(&indexer, w1, &hashes, 4), (Some(2), Vec::new()));
    }

    /// A group a window store names before any main store of its blocks
    /// places nothing when the main store never comes (another rank's cache,
    /// a sub-block entry): the rule then leaves the worker's overlap standing
    /// rather than take it away on a group it cannot read.
    #[test]
    fn a_group_that_never_placed_a_block_leaves_the_overlap_standing() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        KvEventMonitor::apply_events(
            &[
                group_store(1, Some("sliding_window"), Some(5), &[9]),
                store_event(stored_event(1, &TOKENS)),
            ],
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(
            (wb.counters.group_blocks, wb.counters.group_unplaced),
            (0, 1)
        );
        let hashes = compute_request_content_hashes(&TOKENS, 4);
        assert_eq!(capped(&indexer, w1, &hashes, 4), (Some(1), Vec::new()));

        // An unknown kind places its blocks and is left out of the rule too.
        KvEventMonitor::apply_event(
            &group_store(2, Some("chunked_local"), None, &[1]),
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(capped(&indexer, w1, &hashes, 4), (Some(1), Vec::new()));
        // A state group that retains nothing on the path shrinks it to nothing: the holder leaves.
        KvEventMonitor::apply_event(
            &group_store(3, Some("mamba"), None, &[1]),
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(capped(&indexer, w1, &hashes, 4).0, Some(1));
        let mut gone = removed_event(1);
        gone.group_idx = Some(3);
        KvEventMonitor::apply_event(&remove_event(gone), w1, &indexer, &mut wb);
        let (depth, caps) = capped(&indexer, w1, &hashes, 4);
        assert_eq!(depth, None, "capped to nothing: no longer a holder");
        assert_eq!(caps[0].capped, 0);
        assert_eq!(
            indexer.current_size(),
            1,
            "the index still holds the main block"
        );
    }

    /// A kind-less event of a group (a removal never names its kind) is the
    /// group's even when it comes earlier in the batch than the group's first
    /// store, and even for a group past the view's bit limit: the main block
    /// of the same hash stays indexed. A main store without tokens indexes
    /// nothing.
    #[test]
    fn a_kind_less_group_event_never_reaches_the_main_index() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let mut gone = removed_event(1);
        gone.group_idx = Some(1);
        KvEventMonitor::apply_events(
            &[
                store_event(stored_event(1, &TOKENS)),
                remove_event(gone),
                group_store(1, Some("sliding_window"), Some(5), &[1]),
            ],
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(indexer.current_size(), 1, "the main block stays");
        assert_eq!(wb.counters.group_removals, 1);
        let view = indexer.groups().get(w1).expect("the view");
        assert_eq!(view.retained_blocks(), 1);

        // A group past the sixteenth bit: shown, never placed, and its
        // kind-less removal is still not the main group's.
        let mut second = stored_event(2, &[50, 60, 70, 80]);
        second.parent_block_hash = Some(1);
        KvEventMonitor::apply_events(
            &[
                group_store(17, Some("mamba"), None, &[2]),
                store_event(second),
            ],
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(indexer.current_size(), 2);
        let mut gone = removed_event(2);
        gone.group_idx = Some(17);
        KvEventMonitor::apply_event(&remove_event(gone), w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2, "the main block stays");
        assert_eq!(wb.counters.group_removals, 2);
        assert!(
            view.groups().iter().all(|group| group.index != 17),
            "a group without a bit is shown, never placed, never in the rule"
        );

        // A main store without tokens is counted, not indexed.
        let mut bare = stored_event(3, &[]);
        bare.blocks[0].token_ids.clear();
        KvEventMonitor::apply_event(&store_event(bare), w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2);
        assert_eq!(wb.counters.tokenless_stores, 1);
    }

    /// A worker whose first batch carried main stores alone gets its view
    /// later, over blocks the view never named: the rule leaves its overlap
    /// standing (never a wrong name, never a shrunk hit on a path it cannot
    /// read) until a clear, after which the view names every block as it is
    /// indexed and reads again.
    #[test]
    fn a_view_created_after_the_first_main_stores_caps_nothing_until_a_clear() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        KvEventMonitor::apply_events(
            &[store_event(stored_event(1, &TOKENS))],
            w1,
            &indexer,
            &mut wb,
        );
        assert!(indexer.groups().is_empty());
        let mut second = stored_event(2, &[50, 60, 70, 80]);
        second.parent_block_hash = Some(1);
        KvEventMonitor::apply_events(
            &[
                store_event(second),
                group_store(1, Some("sliding_window"), Some(5), &[2]),
            ],
            w1,
            &indexer,
            &mut wb,
        );
        let view = indexer.groups().get(w1).expect("the view");
        assert!(!view.is_complete());
        assert_eq!(
            view.named_blocks(),
            0,
            "block 2 hangs on a block the view never named"
        );
        assert_eq!(
            (wb.counters.group_unnamed, wb.counters.group_view_late),
            (1, 1)
        );
        let hashes = compute_request_content_hashes(&[10, 20, 30, 40, 50, 60, 70, 80], 4);
        assert_eq!(
            capped(&indexer, w1, &hashes, 4),
            (Some(2), Vec::new()),
            "the index's overlap stands"
        );

        KvEventMonitor::apply_cleared(w1, &indexer, &mut wb);
        assert!(view.is_complete());
        let mut second = stored_event(2, &[50, 60, 70, 80]);
        second.parent_block_hash = Some(1);
        KvEventMonitor::apply_events(
            &[
                store_event(stored_event(1, &TOKENS)),
                store_event(second),
                group_store(1, Some("sliding_window"), Some(5), &[1]),
            ],
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(view.named_blocks(), 2);
        let (depth, caps) = capped(&indexer, w1, &hashes, 4);
        assert_eq!(
            depth,
            Some(1),
            "the window's one block ends the hit at block 1"
        );
        assert_eq!(caps.len(), 1);
    }

    /// A store of a kind other than main attention without a group index
    /// names nothing the view can keep; a plain worker never gets a view.
    #[test]
    fn a_nameless_group_store_is_counted_and_a_plain_worker_has_no_view() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let mut nameless = stored_event(1, &TOKENS);
        nameless.kv_cache_spec_kind = Some("sliding_window".to_string());
        KvEventMonitor::apply_event(&store_event(nameless), w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
        assert_eq!(wb.counters.group_nameless, 1);
        KvEventMonitor::apply_event(
            &store_event(stored_event(1, &TOKENS)),
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(indexer.current_size(), 1);
        assert!(indexer.groups().is_empty(), "no view without a group");
        let hashes = compute_request_content_hashes(&TOKENS, 4);
        assert_eq!(capped(&indexer, w1, &hashes, 4), (Some(1), Vec::new()));
    }

    #[test]
    fn remote_and_residency_agent_events_are_skipped() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();

        let mut remote = stored_event(1, &TOKENS);
        remote.locality = Some(KvCacheLocality::Remote as i32);
        KvEventMonitor::apply_event(&store_event(remote), w1, &indexer, &mut wb);
        let mut agent = stored_event(1, &TOKENS);
        agent.ownership = Some("kvcr".to_string());
        KvEventMonitor::apply_event(&store_event(agent), w1, &indexer, &mut wb);
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
        let indexer = KvIndex::chain();
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
        let indexer = KvIndex::chain();
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
        let indexer = KvIndex::chain();
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
        let indexer = KvIndex::chain();
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
        let indexer = KvIndex::chain();
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

    fn stored_width(hash: i64, width: i32, tokens: &[u32], parent: Option<i64>) -> KvBlocksStored {
        KvBlocksStored {
            blocks: vec![KvBlock {
                block_hash: hash,
                token_ids: tokens.to_vec(),
                block_size: width,
                ..Default::default()
            }],
            parent_block_hash: parent,
            ..Default::default()
        }
    }

    /// A hybrid engine registers the prompt's tail inside its last block as a
    /// prefix entry at its hash granularity: a one-block store narrower than
    /// the block of record whose parent is an intermediate hash it never
    /// publishes. It is counted and left out of the index, never a
    /// parent-less store or a root chain of its own.
    #[test]
    fn a_sub_block_store_is_counted_and_not_indexed() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let first: Vec<u32> = (0..32).collect();
        KvEventMonitor::apply_stored(&stored_width(1, 32, &first, None), w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);

        // The tail entry: four tokens under a parent the chain never named.
        KvEventMonitor::apply_stored(
            &stored_width(77, 4, &[1, 2, 3, 4], Some(555)),
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(
            indexer.current_size(),
            1,
            "a sub-block entry is not indexed"
        );
        assert_eq!(wb.counters.sub_block_stores, 1);
        assert_eq!(wb.counters.sub_block_blocks, 1);
        assert_eq!(wb.counters.parentless_stores, 0, "not a parent-less store");
        assert_eq!(wb.counters.parentless_blocks, 0);

        // The chain goes on from the full block as if the entry never came.
        let second: Vec<u32> = (32..64).collect();
        KvEventMonitor::apply_stored(
            &stored_width(2, 32, &second, Some(1)),
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(indexer.current_size(), 2);
        assert_eq!(wb.counters.parentless_stores, 0);

        // The engine's eventual removal of the entry finds nothing to take out.
        KvEventMonitor::apply_removed(
            &KvBlocksRemoved {
                block_hashes: vec![77],
                ..Default::default()
            },
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(indexer.current_size(), 2);
        assert_eq!(wb.counters.sub_block_stores, 1);
    }

    /// The block of record is the widest block the worker stored so far: a
    /// narrower store before any wider one is indexed as it comes, a wider
    /// store raises the record, and a width of zero says nothing either way.
    #[test]
    fn the_block_of_record_is_the_widest_block_stored() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        KvEventMonitor::apply_stored(
            &stored_width(1, 4, &[1, 2, 3, 4], None),
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(indexer.current_size(), 1, "nothing to compare against yet");
        KvEventMonitor::apply_stored(&stored_width(2, 0, &[5, 6], None), w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 2, "an unknown width is indexed");
        let full: Vec<u32> = (10..42).collect();
        KvEventMonitor::apply_stored(&stored_width(3, 32, &full, None), w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 3, "a wider block raises the record");
        KvEventMonitor::apply_stored(
            &stored_width(4, 4, &[7, 8, 9, 10], Some(3)),
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(
            indexer.current_size(),
            3,
            "narrower than the record: left out"
        );
        KvEventMonitor::apply_stored(&stored_width(5, 0, &[11], Some(3)), w1, &indexer, &mut wb);
        assert_eq!(
            indexer.current_size(),
            4,
            "an unknown width is still indexed"
        );
        assert_eq!(wb.counters.sub_block_stores, 1);
        assert_eq!(wb.counters.sub_block_blocks, 1);
        assert_eq!(wb.counters.parentless_stores, 0);
    }

    /// A clear is the worker's new cache incarnation (its publisher restarted
    /// or resynced): the block of record goes with the blocks, so a worker
    /// that comes back with a narrower block indexes its stores as they come
    /// instead of counting every one as a sub-block store.
    #[test]
    fn a_clear_resets_the_block_of_record() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let full: Vec<u32> = (10..42).collect();
        KvEventMonitor::apply_stored(&stored_width(1, 32, &full, None), w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
        KvEventMonitor::apply_cleared(w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
        assert_eq!(
            wb.record_before_clear, 32,
            "kept for the first store's log line"
        );
        KvEventMonitor::apply_stored(
            &stored_width(2, 4, &[1, 2, 3, 4], None),
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(
            indexer.current_size(),
            1,
            "the restarted worker's narrower block is indexed, not left out"
        );
        assert_eq!(wb.counters.sub_block_stores, 0);
        assert_eq!(wb.counters.sub_block_blocks, 0);
        assert_eq!(wb.block_of_record, 4);
        assert_eq!(
            wb.record_before_clear, 0,
            "the first store after the clear settles it"
        );
    }

    /// A store of a window group that carries tokens, the way the relay
    /// forwards a rank without a main-attention group.
    fn window_only_store(
        group: u32,
        hash: i64,
        parent: Option<i64>,
        tokens: &[u32],
    ) -> KvCacheEvent {
        store_event(KvBlocksStored {
            blocks: vec![block(hash, tokens)],
            parent_block_hash: parent,
            group_idx: Some(group),
            kv_cache_spec_kind: Some("sliding_window".to_string()),
            kv_cache_spec_sliding_window: Some(64),
            ..Default::default()
        })
    }

    /// A kind-less removal naming its group, as the engines publish them.
    fn removal_in(group: u32, hash: i64) -> KvCacheEvent {
        let mut removed = removed_event(hash);
        removed.group_idx = Some(group);
        remove_event(removed)
    }

    fn overlap(indexer: &KvIndex, worker: u32, hashes: &[ContentHash]) -> u32 {
        indexer
            .find_matches(hashes, false)
            .scores
            .get(&worker)
            .copied()
            .unwrap_or(0)
    }

    const SECOND: [u32; 4] = [50, 60, 70, 80];
    const THIRD: [u32; 4] = [90, 91, 92, 93];

    /// A worker whose attention is all sliding window has no main group:
    /// its window stores carry tokens and are the prefix index's, as a plain
    /// worker's; no view is made for it; a kind-less removal of the group
    /// leaves the index.
    #[test]
    fn a_window_only_workers_stores_and_removals_are_the_prefix_indexs() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let both: Vec<u32> = TOKENS.iter().chain(SECOND.iter()).copied().collect();
        let hashes = compute_request_content_hashes(&both, 4);
        let one = compute_request_content_hashes(&TOKENS, 4);

        KvEventMonitor::apply_events(
            &[
                window_only_store(0, 1, None, &TOKENS),
                window_only_store(0, 2, Some(1), &SECOND),
            ],
            w1,
            &indexer,
            &mut wb,
        );
        assert!(
            wb.groups.is_none(),
            "no view for a worker without a main group"
        );
        assert_eq!(indexer.current_size(), 2);
        let full = overlap(&indexer, w1, &hashes);
        assert!(full > 0 && full > overlap(&indexer, w1, &one));
        assert_eq!(
            (wb.counters.window_only_stores, wb.counters.group_stores),
            (2, 0)
        );

        KvEventMonitor::apply_events(&[removal_in(0, 2)], w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 1);
        assert_eq!(overlap(&indexer, w1, &hashes), overlap(&indexer, w1, &one));
        assert_eq!(
            (wb.counters.window_only_removals, wb.counters.group_removals),
            (1, 0)
        );
    }

    /// A store the index skips (a residency agent's, a remote tier's) shows
    /// no group in a batch: a foreign main store ahead of a window-only
    /// worker's store leaves that store in the prefix index, and a foreign
    /// window store shows no group.
    #[test]
    fn a_foreign_store_in_a_batch_shows_no_group() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let mut agent_main = stored_event(7, &THIRD);
        agent_main.kv_cache_spec_kind = Some("full_attention".to_string());
        agent_main.group_idx = Some(1);
        agent_main.ownership = Some("kvcr".to_string());
        let mut remote_window = window_only_store(2, 8, None, &SECOND);
        if let Some(kv_cache_event::Data::Stored(stored)) = &mut remote_window.data {
            stored.locality = Some(KvCacheLocality::Remote as i32);
        }

        KvEventMonitor::apply_events(
            &[
                store_event(agent_main),
                window_only_store(0, 1, None, &TOKENS),
                remote_window,
            ],
            w1,
            &indexer,
            &mut wb,
        );
        assert!(!wb.main_shown, "a foreign main store shows no main group");
        assert!(wb.groups.is_none(), "a foreign window store shows no group");
        assert_eq!(
            indexer.current_size(),
            1,
            "the window store is the prefix index's"
        );
        assert_eq!(
            (
                wb.counters.window_only_stores,
                wb.counters.group_stores,
                wb.counters.foreign_owner,
                wb.counters.remote,
            ),
            (1, 0, 1, 1)
        );

        // The group pass's own case: a worker whose main group has shown
        // gets a foreign window store, which would show that group and give
        // the worker a view; skipped, the worker keeps none.
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let mut remote_window = window_only_store(2, 9, None, &THIRD);
        if let Some(kv_cache_event::Data::Stored(stored)) = &mut remote_window.data {
            stored.locality = Some(KvCacheLocality::Remote as i32);
        }
        KvEventMonitor::apply_events(
            &[store_event(stored_event(1, &TOKENS)), remote_window],
            w2,
            &indexer,
            &mut wb,
        );
        assert!(wb.main_shown, "the main store shows the main group");
        assert!(
            wb.groups.is_none(),
            "a foreign window store after the main group shows no group"
        );
        assert_eq!((wb.counters.group_stores, wb.counters.remote), (0, 1));
    }

    /// A relay snapshot replays the stores after a clear in one batch: the
    /// batch reads as the stream did, and a removal after it still leaves
    /// the index.
    #[test]
    fn a_window_only_workers_snapshot_replay_reads_as_its_stream() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let both: Vec<u32> = TOKENS.iter().chain(SECOND.iter()).copied().collect();
        let hashes = compute_request_content_hashes(&both, 4);
        KvEventMonitor::apply_events(
            &[window_only_store(0, 1, None, &TOKENS)],
            w1,
            &indexer,
            &mut wb,
        );
        let cleared = KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Cleared(KvCacheCleared::default())),
        };
        KvEventMonitor::apply_events(
            &[
                cleared,
                window_only_store(0, 1, None, &TOKENS),
                window_only_store(0, 2, Some(1), &SECOND),
            ],
            w1,
            &indexer,
            &mut wb,
        );
        assert!(wb.groups.is_none());
        assert_eq!(indexer.current_size(), 2);
        assert!(routable(&indexer, w1, &hashes));
        assert_eq!(wb.counters.window_only_stores, 3);

        KvEventMonitor::apply_events(&[removal_in(0, 1), removal_in(0, 2)], w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0);
        assert!(!routable(&indexer, w1, &hashes));
        assert_eq!(wb.counters.window_only_removals, 2);
    }

    /// A worker whose first batches carried only a window group's stores
    /// (its main signal) and then shows a main group: the index lets the
    /// signal's blocks go and the main group's stores rebuild it, the window
    /// group moves to the view, and its removals are the view's from then
    /// on, never the main group's blocks'.
    #[test]
    fn a_main_group_shown_later_moves_the_window_group_to_the_view() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let one = compute_request_content_hashes(&TOKENS, 4);
        KvEventMonitor::apply_events(
            &[window_only_store(0, 1, None, &TOKENS)],
            w1,
            &indexer,
            &mut wb,
        );
        assert!(routable(&indexer, w1, &one));

        // The main lane's batch of the same block: the signal's copy goes,
        // the main group's comes, one copy counted, the window group known
        // to the view.
        let mut main = stored_event(1, &TOKENS);
        main.group_idx = Some(1);
        main.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_events(&[store_event(main)], w1, &indexer, &mut wb);
        assert!(
            wb.groups.is_some(),
            "the main group makes the worker hybrid"
        );
        assert!(routable(&indexer, w1, &one));
        assert_eq!(indexer.current_size(), 1);
        assert_eq!(wb.counters.main_signal_dropped, 1);
        assert!(
            wb.copies.is_empty(),
            "the main group's copy is the only one"
        );

        // The window group's later store is the view's, and its removals
        // are the view's: the main group's blocks stand until the main
        // group lets them go.
        KvEventMonitor::apply_events(
            &[group_store(0, Some("sliding_window"), Some(64), &[1])],
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(wb.counters.group_stores, 1);
        let mut later = stored_event(4, &[90, 91, 92, 93]);
        later.group_idx = Some(1);
        later.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_events(&[store_event(later)], w1, &indexer, &mut wb);
        KvEventMonitor::apply_events(&[removal_in(0, 1), removal_in(0, 4)], w1, &indexer, &mut wb);
        assert_eq!(
            indexer.current_size(),
            2,
            "the window group's removals leave the index alone"
        );
        assert_eq!(
            (wb.counters.window_only_removals, wb.counters.group_removals),
            (0, 2)
        );
        KvEventMonitor::apply_events(&[removal_in(1, 1)], w1, &indexer, &mut wb);
        assert!(!routable(&indexer, w1, &one), "the main group let it go");
        assert_eq!(indexer.current_size(), 1);
    }

    /// A hashes-only store of the window group before any main group gives
    /// the worker a view late, over the signal's blocks, incomplete; the main
    /// group's store restarts the worker: the view names its blocks from an
    /// empty index, complete, and the window group's memberships stay for
    /// the names to come, the retained ones and the signal's own, so a path
    /// the engine serves keeps its credit.
    #[test]
    fn a_view_created_late_is_complete_after_the_restart() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let one = compute_request_content_hashes(&TOKENS, 4);
        let two = compute_request_content_hashes(&SECOND, 4);
        let three = compute_request_content_hashes(&THIRD, 4);
        KvEventMonitor::apply_events(
            &[window_only_store(0, 1, None, &TOKENS)],
            w1,
            &indexer,
            &mut wb,
        );
        KvEventMonitor::apply_events(
            &[group_store(0, Some("sliding_window"), Some(64), &[1])],
            w1,
            &indexer,
            &mut wb,
        );
        let view = wb
            .groups
            .clone()
            .expect("the hashes-only store gives a view");
        assert!(!view.is_complete(), "created over the signal's block");
        assert_eq!(wb.counters.group_view_late, 1);

        // A second signal block, named by the view and retained by the
        // window group through it; a third with no hashes-only store of its
        // own, named only.
        KvEventMonitor::apply_events(
            &[window_only_store(0, 2, None, &SECOND)],
            w1,
            &indexer,
            &mut wb,
        );
        KvEventMonitor::apply_events(
            &[group_store(0, Some("sliding_window"), Some(64), &[2])],
            w1,
            &indexer,
            &mut wb,
        );
        KvEventMonitor::apply_events(
            &[window_only_store(0, 3, None, &THIRD)],
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!((view.named_blocks(), view.retained_blocks()), (2, 1));

        let mut main = stored_event(2, &SECOND);
        main.group_idx = Some(1);
        main.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_events(&[store_event(main)], w1, &indexer, &mut wb);
        assert!(view.is_complete(), "the restart cleared the names");
        assert_eq!(wb.counters.main_signal_dropped, 3);
        assert!(!routable(&indexer, w1, &one), "the signal's block went");
        assert!(routable(&indexer, w1, &two));
        assert_eq!(indexer.current_size(), 1);
        assert_eq!(view.named_blocks(), 1, "the main group's block is named");
        assert_eq!(
            view.retained_blocks(),
            1,
            "the window group's membership came back with the name"
        );
        assert_eq!(
            capped(&indexer, w1, &two, 4).0,
            Some(1),
            "the path the engine serves keeps its credit"
        );

        // The third block, the signal's own: the main group's store names it
        // and the window group's membership is there.
        let mut third = stored_event(3, &THIRD);
        third.group_idx = Some(1);
        third.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_events(&[store_event(third)], w1, &indexer, &mut wb);
        assert_eq!((view.named_blocks(), view.retained_blocks()), (2, 2));
        assert_eq!(capped(&indexer, w1, &three, 4).0, Some(1));
    }

    /// The signal's own blocks are the window group's: a worker whose window
    /// stores were its main signal shows a main group, and after the restart
    /// the view holds the window group's membership for every block the
    /// index let go, so a path the engine serves keeps its credit once the
    /// group can be read.
    #[test]
    fn the_signals_blocks_keep_the_window_groups_membership_across_the_restart() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let one = compute_request_content_hashes(&TOKENS, 4);
        KvEventMonitor::apply_events(
            &[window_only_store(0, 1, None, &TOKENS)],
            w1,
            &indexer,
            &mut wb,
        );
        assert!(wb.groups.is_none());

        let mut main = stored_event(1, &TOKENS);
        main.group_idx = Some(1);
        main.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_events(&[store_event(main)], w1, &indexer, &mut wb);
        let view = wb.groups.clone().expect("the restart makes the view");
        assert!(view.is_complete());
        assert_eq!(
            view.retained_blocks(),
            1,
            "the signal's block waited and was placed as the window group's"
        );

        // The window group can be read once a store of its own names its
        // kind.
        let mut second = stored_event(2, &SECOND);
        second.group_idx = Some(1);
        second.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_events(
            &[
                group_store(0, Some("sliding_window"), Some(64), &[2]),
                store_event(second),
            ],
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(view.retained_blocks(), 2);
        assert_eq!(
            capped(&indexer, w1, &one, 4).0,
            Some(1),
            "the path the engine serves keeps its credit"
        );
    }

    /// The restart keeps every membership the view holds, retained or waiting,
    /// not only the signal group's. A state group's hashes-only store of the
    /// signal's first block (the view made late over it) waits for a name; its
    /// store of a second signal block, one the view names, is retained. The
    /// restart moves the retained membership back to waiting beside the
    /// waiting one, and the main group's names place both, each beside the
    /// signal group's seeded bit. A clear in place of the move drops both; the
    /// move dropped alone loses the retained one.
    #[test]
    fn another_groups_memberships_survive_the_restart() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let one = compute_request_content_hashes(&TOKENS, 4);
        let two = compute_request_content_hashes(&SECOND, 4);
        let placed_by = |view: &WorkerGroups, index: u32| {
            view.groups()
                .iter()
                .find(|group| group.index == index)
                .map_or(0, |group| group.placed)
        };
        KvEventMonitor::apply_events(
            &[window_only_store(0, 1, None, &TOKENS)],
            w1,
            &indexer,
            &mut wb,
        );
        // The state group's store of the signal's first block: the view is
        // made late over the block, which it cannot name, so the membership
        // waits under the state group's bit.
        KvEventMonitor::apply_events(
            &[group_store(2, Some("mamba"), None, &[1])],
            w1,
            &indexer,
            &mut wb,
        );
        let view = wb
            .groups
            .clone()
            .expect("the hashes-only store gives a view");
        assert!(!view.is_complete(), "created over the signal's block");
        // A second signal block, named by the view, and the state group's
        // store of it: retained.
        KvEventMonitor::apply_events(
            &[window_only_store(0, 2, None, &SECOND)],
            w1,
            &indexer,
            &mut wb,
        );
        KvEventMonitor::apply_events(
            &[group_store(2, Some("mamba"), None, &[2])],
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(
            (
                view.named_blocks(),
                view.retained_blocks(),
                view.pending_blocks()
            ),
            (1, 1, 1),
            "block 2 named and retained, block 1 waiting"
        );
        assert_eq!(placed_by(&view, 2), 1);

        // The restart: the main group names block 2 anew. The retained
        // membership moved back to waiting and is placed with the name, the
        // waiting one keeps waiting; the signal group's bit is seeded beside.
        let mut main = stored_event(2, &SECOND);
        main.group_idx = Some(1);
        main.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_events(&[store_event(main)], w1, &indexer, &mut wb);
        assert!(view.is_complete(), "the restart cleared the names");
        assert_eq!(wb.counters.main_signal_dropped, 2);
        assert_eq!(
            (
                view.named_blocks(),
                view.retained_blocks(),
                view.pending_blocks()
            ),
            (1, 1, 1),
            "block 2 placed again, block 1 still waiting"
        );
        assert_eq!(
            placed_by(&view, 2),
            2,
            "the retained membership came back with the name (placed twice: before and after)"
        );
        assert_eq!(
            placed_by(&view, 0),
            1,
            "the signal group's seeded bit placed"
        );
        assert_eq!(
            capped(&indexer, w1, &two, 4).0,
            Some(1),
            "the state group reads block 2: the path keeps its credit"
        );

        // The main group names block 1: the waiting membership is placed too.
        let mut first = stored_event(1, &TOKENS);
        first.group_idx = Some(1);
        first.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_events(&[store_event(first)], w1, &indexer, &mut wb);
        assert_eq!((view.retained_blocks(), view.pending_blocks()), (2, 0));
        assert_eq!((placed_by(&view, 2), placed_by(&view, 0)), (3, 2));
        assert_eq!(capped(&indexer, w1, &one, 4).0, Some(1));
    }

    /// Two groups gave a worker its main signal: whose block is whose is
    /// unknown at the restart, so the view stays incomplete and the overlap
    /// stands until a clear, counted as a late view is.
    #[test]
    fn two_signal_groups_leave_the_restarted_view_incomplete() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        KvEventMonitor::apply_events(
            &[
                window_only_store(0, 1, None, &TOKENS),
                window_only_store(2, 2, None, &SECOND),
            ],
            w1,
            &indexer,
            &mut wb,
        );
        let mut main = stored_event(1, &TOKENS);
        main.group_idx = Some(1);
        main.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_events(&[store_event(main)], w1, &indexer, &mut wb);
        let view = wb.groups.clone().expect("the restart makes the view");
        assert!(!view.is_complete(), "whose block is whose is unknown");
        assert_eq!(wb.counters.main_signal_dropped, 2);
        assert_eq!(
            wb.counters.group_view_late, 1,
            "on the record, as a late view"
        );
        assert_eq!(view.retained_blocks(), 0);
        assert!(wb.knows_group(0) && wb.knows_group(2));
    }

    /// The restart's wait is bounded. Hashes-only stores of the window group
    /// before any main group fill the worker's view with blocks waiting for
    /// a name; when the main group shows, the signal's own block finds the
    /// wait full and is counted dropped, and the worker's counter reads the
    /// view's count after the restart.
    #[test]
    fn a_signal_block_the_restarts_full_wait_refuses_counts_as_dropped() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        KvEventMonitor::apply_events(
            &[window_only_store(0, 1, None, &TOKENS)],
            w1,
            &indexer,
            &mut wb,
        );

        // Blocks the view has no name for wait in it, up to its bound: fill
        // the wait in batches until a batch adds nothing.
        const BATCH: i64 = 1 << 16;
        let waiting_store = |from: i64| {
            let hashes: Vec<i64> = (from..from + BATCH).collect();
            group_store(0, Some("sliding_window"), Some(64), &hashes)
        };
        KvEventMonitor::apply_events(&[waiting_store(1_000)], w1, &indexer, &mut wb);
        let view = wb
            .groups
            .clone()
            .expect("the hashes-only store gives a view");
        let mut from = 1_000 + BATCH;
        let mut filled = false;
        for _ in 0..64 {
            let waiting = view.pending_blocks();
            KvEventMonitor::apply_events(&[waiting_store(from)], w1, &indexer, &mut wb);
            from += BATCH;
            if view.pending_blocks() == waiting {
                filled = true;
                break;
            }
        }
        assert!(filled, "the wait has a bound");
        let full = view.pending_blocks();
        let dropped = view.dropped_blocks();
        assert!(dropped > 0, "the last batch found the wait full");
        assert_eq!(wb.counters.group_dropped, dropped);

        // The main group shows: the signal's block leaves the index and
        // asks the full wait for a place; refused, it is counted dropped,
        // and the counter reads the view.
        let mut main = stored_event(2, &SECOND);
        main.group_idx = Some(1);
        main.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_events(&[store_event(main)], w1, &indexer, &mut wb);
        assert_eq!(wb.counters.main_signal_dropped, 1);
        assert!(view.is_complete(), "the restart made the view complete");
        assert_eq!(view.pending_blocks(), full, "the wait is still full");
        assert_eq!(
            view.dropped_blocks(),
            dropped + 1,
            "the signal's block found the wait full"
        );
        assert_eq!(
            wb.counters.group_dropped,
            view.dropped_blocks(),
            "the counter reads the view after the restart"
        );
    }

    /// Two groups gave a worker its main signal and their blocks left the
    /// index before the main group showed: the restart finds no block under
    /// the view, so the view is complete and no late view is counted.
    #[test]
    fn a_restart_over_no_blocks_counts_no_late_view() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        KvEventMonitor::apply_events(
            &[
                window_only_store(0, 1, None, &TOKENS),
                window_only_store(2, 2, None, &SECOND),
            ],
            w1,
            &indexer,
            &mut wb,
        );
        KvEventMonitor::apply_events(&[removal_in(0, 1), removal_in(2, 2)], w1, &indexer, &mut wb);
        assert_eq!(indexer.current_size(), 0, "the signals' blocks left");
        assert_eq!(wb.counters.window_only_removals, 2);

        let mut main = stored_event(1, &TOKENS);
        main.group_idx = Some(1);
        main.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_events(&[store_event(main)], w1, &indexer, &mut wb);
        let view = wb.groups.clone().expect("the restart makes the view");
        assert!(view.is_complete(), "nothing stood under the view");
        assert_eq!(wb.counters.main_signal_dropped, 0);
        assert_eq!(
            wb.counters.group_view_late, 0,
            "no late view over no blocks"
        );
        assert!(wb.knows_group(0) && wb.knows_group(2));
        assert_eq!(indexer.current_size(), 1, "the main group's block is in");
    }

    /// A window-only worker's second copy of a block (an engine's resend)
    /// needs a second removal, as a main group's does.
    #[test]
    fn a_window_only_workers_second_copy_needs_a_second_removal() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let one = compute_request_content_hashes(&TOKENS, 4);
        KvEventMonitor::apply_events(
            &[
                window_only_store(0, 1, None, &TOKENS),
                window_only_store(0, 1, None, &TOKENS),
            ],
            w1,
            &indexer,
            &mut wb,
        );
        assert_eq!(wb.counters.duplicate_copies, 1);
        KvEventMonitor::apply_events(&[removal_in(0, 1)], w1, &indexer, &mut wb);
        assert!(routable(&indexer, w1, &one), "one copy remains");
        KvEventMonitor::apply_events(&[removal_in(0, 1)], w1, &indexer, &mut wb);
        assert!(!routable(&indexer, w1, &one));
        assert_eq!(indexer.current_size(), 0);
        assert!(wb.copies.is_empty());
        assert_eq!(wb.counters.window_only_removals, 2);
    }

    /// A hybrid worker whose window stores carry tokens (an engine that
    /// publishes every group's blocks with their tokens): the main group's
    /// store in the same batch keeps them the view's.
    #[test]
    fn a_hybrid_workers_tokenful_window_store_stays_in_the_view() {
        let indexer = KvIndex::chain();
        let w1 = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerIndexState::default();
        let mut main = stored_event(1, &TOKENS);
        main.group_idx = Some(0);
        main.kv_cache_spec_kind = Some("full_attention".to_string());
        KvEventMonitor::apply_events(
            &[window_only_store(1, 1, None, &TOKENS), store_event(main)],
            w1,
            &indexer,
            &mut wb,
        );
        assert!(wb.groups.is_some());
        assert_eq!(indexer.current_size(), 1);
        assert_eq!(
            (wb.counters.window_only_stores, wb.counters.group_stores),
            (0, 1)
        );
    }
}
