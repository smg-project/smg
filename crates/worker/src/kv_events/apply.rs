//! One event into the index: which tiers, localities and owners are indexed,
//! which cache groups go to the index and which to the worker's group view
//! beside it, content hashing under the event's namespace, and the physical
//! copies of a block counted per worker so a removal evicts the block only
//! when its last copy goes.

use std::{
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use kv_index::{
    salt::{content_hash_with_seed, namespace_seed},
    ApplyError, ContentHash, GroupKind, SequenceHash, StoredBlock, WorkerGroups,
};
use tracing::warn;

use crate::{
    common_proto::{
        kv_cache_event, KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheEvent, KvCacheLocality,
        KvCacheTier, KvEventBatch,
    },
    metrics::writers::Metrics,
};

/// One model's index, as the cache events write it.
pub trait BlockIndex {
    /// The reverse map a writer keeps for each worker.
    type Held: HeldBlocks;

    /// Store blocks after `parent`, or from the root when it is absent.
    fn apply_stored(
        &self,
        worker: u32,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
        held: &mut Self::Held,
    ) -> Result<(), ApplyError>;

    /// Forget the named blocks of the worker.
    fn apply_removed(&self, worker: u32, hashes: &[SequenceHash], held: &mut Self::Held);

    /// Forget every block of the worker.
    fn apply_cleared(&self, worker: u32, held: &mut Self::Held);

    /// The worker's cache group view, created on the first call.
    fn groups(&self, worker: u32) -> Arc<WorkerGroups>;
}

/// The blocks the index holds for one worker, by sequence hash.
#[expect(
    clippy::len_without_is_empty,
    reason = "The apply rules need the number of held blocks"
)]
pub trait HeldBlocks: Default {
    /// Whether the worker holds this block.
    fn contains_key(&self, hash: SequenceHash) -> bool;

    /// The number of blocks the worker holds.
    fn len(&self) -> usize;

    /// The sequence hashes the worker holds, in any order.
    fn hashes(&self) -> Vec<SequenceHash>;
}

/// Apply a single KV cache event to the indexer.
pub fn apply_event<I: BlockIndex>(
    event: &KvCacheEvent,
    worker_id: u32,
    indexer: &I,
    worker_blocks: &mut WorkerIndexState<I::Held>,
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
                Placement::Main | Placement::MainAndGroup(_) => {
                    apply_stored(stored, worker_id, indexer, worker_blocks);
                }
                Placement::Group(mask) => {
                    apply_group_stored(stored, mask, worker_id, indexer, worker_blocks);
                }
                Placement::Skip => {}
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
                Placement::Main => {
                    if removed
                        .group_idx
                        .is_some_and(|group| worker_blocks.shown.is_main_signal(group))
                    {
                        worker_blocks.counters.window_only_removals += 1;
                    }
                    apply_removed(removed, worker_id, indexer, worker_blocks);
                }
                Placement::MainAndGroup(mask) => {
                    worker_blocks.counters.window_only_removals += 1;
                    apply_removed(removed, worker_id, indexer, worker_blocks);
                    apply_group_removed(removed, mask, worker_blocks);
                }
                Placement::Group(mask) => {
                    apply_group_removed(removed, mask, worker_blocks);
                }
                Placement::Skip => {}
            }
        }
        kv_cache_event::Data::Cleared(cleared) => {
            if worker_blocks.admit(None, None, false, None, cleared.ownership.as_deref())
                == Placement::Main
            {
                apply_cleared(worker_id, indexer, worker_blocks);
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
pub fn apply_events<I: BlockIndex>(
    events: &[KvCacheEvent],
    worker_id: u32,
    indexer: &I,
    worker_blocks: &mut WorkerIndexState<I::Held>,
) {
    let (restarted, hybrid) = worker_blocks.shown.show_batch(events);
    if let Some((group, group_count)) = restarted {
        main_group_shown(worker_id, indexer, worker_blocks, group, group_count);
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
            if worker_blocks.shown.is_group_store(stored) {
                deferred.push(event);
                continue;
            }
        }
        apply_event(event, worker_id, indexer, worker_blocks);
    }
    for event in deferred {
        apply_event(event, worker_id, indexer, worker_blocks);
    }
}
/// Apply a batch a worker's index takes and account for it: the blocks it
/// stored (under `stored_op`) and removed, the stores the index did not
/// take as is, the batch's age, and the batch itself under `disposition`.
/// The series are the worker's, under `worker_url`. Returns how long the
/// apply took: a live stream publishes it as its pace, a snapshot chunk
/// does not. The worker's index-blocks gauge is its host's to set, since
/// each host reads that count where its index keeps it.
pub fn apply_counted<I: BlockIndex>(
    batch: &KvEventBatch,
    worker_url: &str,
    worker_id: u32,
    indexer: &I,
    worker_blocks: &mut WorkerIndexState<I::Held>,
    stored_op: &'static str,
    disposition: &'static str,
) -> Duration {
    let started = Instant::now();
    let counters_before = worker_blocks.counters;
    let (mut stored_blocks, mut removed_blocks) = (0usize, 0usize);
    for event in &batch.events {
        match &event.data {
            Some(kv_cache_event::Data::Stored(stored)) => stored_blocks += stored.blocks.len(),
            Some(kv_cache_event::Data::Removed(removed)) => {
                removed_blocks += removed.block_hashes.len();
            }
            _ => {}
        }
    }
    apply_events(&batch.events, worker_id, indexer, worker_blocks);
    let took = started.elapsed();
    record_not_indexed(worker_url, &worker_blocks.counters, &counters_before);
    if stored_blocks > 0 {
        Metrics::record_kv_event_blocks(worker_url, stored_op, stored_blocks);
    }
    if removed_blocks > 0 {
        Metrics::record_kv_event_blocks(worker_url, "removed", removed_blocks);
    }
    record_lag(worker_url, batch.timestamp);
    Metrics::record_kv_event_batch(worker_url, disposition);
    took
}

/// Publish the stores a batch placed without their parent, the sub-block
/// stores it left out of the index and the stores the index declined at an
/// arena cap, if any.
fn record_not_indexed(worker_url: &str, after: &WorkerIndexCounters, before: &WorkerIndexCounters) {
    let (stores, blocks) = after.parentless_since(before);
    if stores > 0 {
        Metrics::record_kv_event_parentless(worker_url, stores, blocks);
    }
    let (stores, blocks) = after.sub_block_since(before);
    if stores > 0 {
        Metrics::record_kv_event_sub_block(worker_url, stores, blocks);
    }
    let (stores, blocks) = after.cap_refused_since(before);
    if stores > 0 {
        Metrics::record_kv_event_cap_refused(worker_url, stores, blocks);
    }
}

/// Age of a batch when applied, from the publisher's wall-clock stamp.
fn record_lag(worker_url: &str, published_at: f64) {
    if published_at <= 0.0 {
        return;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let lag = now - published_at;
    if lag.is_finite() && lag >= 0.0 {
        Metrics::record_kv_event_lag(worker_url, lag);
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
fn apply_stored<I: BlockIndex>(
    stored: &KvBlocksStored,
    worker_id: u32,
    indexer: &I,
    worker_blocks: &mut WorkerIndexState<I::Held>,
) {
    let Some(tier) = stored_tier(stored) else {
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
                    note_cap_refusal(worker_blocks, worker_id, blocks.len() - placed);
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
            note_cap_refusal(worker_blocks, worker_id, blocks.len() - placed);
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
fn note_cap_refusal<H: HeldBlocks>(
    worker_blocks: &mut WorkerIndexState<H>,
    worker_id: u32,
    refused: usize,
) {
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
fn apply_group_stored<I: BlockIndex>(
    stored: &KvBlocksStored,
    mask: u16,
    worker_id: u32,
    indexer: &I,
    worker_blocks: &mut WorkerIndexState<I::Held>,
) {
    if stored_tier(stored).is_none() {
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
/// [`apply_group_stored`]): the groups let go of the blocks in the
/// worker's view.
fn apply_group_removed<H: HeldBlocks>(
    removed: &KvBlocksRemoved,
    mask: u16,
    worker_blocks: &mut WorkerIndexState<H>,
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
fn apply_removed<I: BlockIndex>(
    removed: &KvBlocksRemoved,
    worker_id: u32,
    indexer: &I,
    worker_blocks: &mut WorkerIndexState<I::Held>,
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
/// publish from here on.
fn main_group_shown<I: BlockIndex>(
    worker_id: u32,
    indexer: &I,
    worker_blocks: &mut WorkerIndexState<I::Held>,
    group: u32,
    group_count: usize,
) {
    let signal = worker_blocks.blocks.hashes();
    indexer.apply_cleared(worker_id, &mut worker_blocks.blocks);
    worker_blocks.copies.clear();
    worker_blocks.counters.main_signal_dropped += signal.len() as u64;
    let view = worker_blocks.view(indexer, worker_id);
    view.restarted();
    match group_count {
        1 if group_mask(group) != 0 => {
            view.waiting(group_mask(group), &signal);
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
                    groups = group_count,
                    "KV cache group view restarted over the blocks of more than one main-signal group, or of one the rule cannot read; its rule stays off until the worker's next clear"
                );
            }
            view.created_over(signal.len());
        }
    }
}

/// Drop every block of a worker from the indexer and forget its copies
/// and its block of record: a clear is the worker's new cache incarnation
/// (its publisher restarted or resynced), and a worker that comes back
/// with a narrower block than it stored before must index its stores as
/// they come, not count every one as a sub-block store. The dropped
/// record is kept until that first store, which is logged when it is
/// narrower (`WorkerIndexState::sub_block`).
pub fn apply_cleared<I: BlockIndex>(
    worker_id: u32,
    indexer: &I,
    worker_blocks: &mut WorkerIndexState<I::Held>,
) {
    indexer.apply_cleared(worker_id, &mut worker_blocks.blocks);
    worker_blocks.index_emptied();
}

/// Convert a proto `KvBlock` to a kv-index `StoredBlock`, hashing its tokens
/// under `seed` (see [`namespace_seed`]).
#[inline]
pub fn convert_kv_block(block: &KvBlock, seed: u64) -> StoredBlock {
    StoredBlock {
        seq_hash: SequenceHash::from(block.block_hash),
        content_hash: content_hash_with_seed(&block.token_ids, seed),
    }
}

/// The residency tiers the index tracks: the device, and the host cache the
/// engine restores from without recompute. Disk and external copies are not
/// indexed; a hit there costs an engine-side fetch the router cannot price.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexedTier {
    /// Blocks in device memory.
    Device,
    /// Blocks in the host cache.
    Host,
}

/// The tier an event names: its `tier` when set, else a block's
/// `cache_level` (absent means the device). `None` when the index does not
/// track that tier.
#[inline]
pub fn indexed_tier(tier: Option<i32>, cache_level: Option<i32>) -> Option<IndexedTier> {
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

/// The tier a store's blocks are on, when the index tracks it: the store's
/// `tier`, else its first block's `cache_level`.
#[inline]
pub fn stored_tier(stored: &KvBlocksStored) -> Option<IndexedTier> {
    let first_level = stored.blocks.first().and_then(|block| block.cache_level);
    indexed_tier(stored.tier, first_level)
}

/// The width a store names for its blocks, in tokens: its first block's,
/// since an engine writes one width per store. A width of zero or less is
/// unknown, and names none.
#[inline]
pub fn store_width(stored: &KvBlocksStored) -> Option<u32> {
    stored
        .blocks
        .first()
        .and_then(|block| u32::try_from(block.block_size).ok())
        .filter(|&width| width > 0)
}

/// Cache-group kinds whose blocks hold the main attention KV, the ones the
/// prefix index is built from. Every other kind goes to the worker's group
/// view, where the sliding-window and state kinds bound a hit the way the
/// engine does ([`WorkerGroups::cap`]).
const MAIN_ATTENTION_KINDS: [&str; 3] = ["full_attention", "mla_attention", "sink_full_attention"];

/// A store or removal a residency agent (`kvcr`) owns: skipped by
/// [`GroupsShown::admit`], counted where its verdict is read, and read by
/// neither batch pre-scan.
#[inline]
fn foreign_owner(ownership: Option<&str>) -> bool {
    ownership.is_some_and(|owner| owner.eq_ignore_ascii_case("kvcr"))
}

/// An event about blocks on a remote tier: skipped like a foreign owner's.
#[inline]
fn remote(locality: Option<i32>) -> bool {
    locality.is_some_and(|locality| locality == KvCacheLocality::Remote as i32)
}

/// Whether [`GroupsShown::admit`] skips a store before it reads the
/// store's group: the batch pre-scans skip it too, so a foreign store shows
/// neither a main group nor a window group.
#[inline]
fn foreign(stored: &KvBlocksStored) -> bool {
    foreign_owner(stored.ownership.as_deref()) || remote(stored.locality)
}

/// Whether every block of a store carries its tokens: a main-lane store
/// does, a group lane's store (hashes only) does not.
#[inline]
pub fn tokenful(stored: &KvBlocksStored) -> bool {
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

/// Where a cache event goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    /// An event for the main index.
    Main,
    /// A tokenful store that feeds the main index until a main group shows.
    MainSignal,
    /// An event for the named group in the worker's view.
    Group(u32),
    /// A removal for both the main index and the named group in the view.
    MainAndGroup(u32),
    /// An event owned by a residency agent.
    ForeignOwner,
    /// An event about blocks on a remote tier.
    Remote,
    /// A store of a group kind that supplies no group index.
    GroupNameless,
}

/// Where the index applies an event, under the view's group mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
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

/// The most physical copies of one block counted per tier. An engine's
/// full report mode re-announces whole hit chains without removals, which
/// would grow a count without bound; the cap turns that into at most this
/// many extra removals before the block leaves the index.
const COPIES_CAP: u8 = 8;

/// What the index did not take at face value, by reason; logged
/// when the worker's subscription ends.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerIndexCounters {
    /// Stores and removals on tiers the index does not track.
    untracked_tier: u64,
    /// Stores and removals of cache groups other than main attention, kept
    /// in the worker's group view beside the index; the blocks of those
    /// stores the view placed through the index, and the ones it could not
    /// when the store came (their main block was not indexed yet: they wait
    /// in the view and count as placed when it comes).
    pub group_stores: u64,
    /// Group blocks placed in the view, including ones placed after waiting.
    pub group_blocks: u64,
    /// Group blocks whose main block was missing when the store came.
    pub group_unplaced: u64,
    /// Removals from the worker's group view.
    pub group_removals: u64,
    /// Stores of a kind other than main attention that name no group.
    pub group_nameless: u64,
    /// Group blocks the view's wait refused, and main blocks it could not
    /// name under a parent it did not know (the view's own counts, copied).
    group_dropped: u64,
    /// Main blocks the view could not name under an unknown parent.
    pub group_unnamed: u64,
    /// Views whose rule went off when they were made or restarted: created
    /// after the worker's main blocks were indexed (off until the worker's
    /// next clear, or until a restart leaves the view complete), or
    /// restarted over the blocks of more than one main-signal group (or of
    /// one the rule cannot read), off until the worker's next clear.
    pub group_view_late: u64,
    /// Main stores with a block that carried no tokens: nothing to index under.
    tokenless_stores: u64,
    /// Stores of a group other than main attention that carried tokens on a
    /// worker that has shown no main group: the worker's main signal, the
    /// prefix index's.
    window_only_stores: u64,
    /// Kind-less removals of such a group: they leave the prefix index.
    window_only_removals: u64,
    /// Blocks the prefix index let go when such a worker showed a main
    /// group: the main group's stores rebuild it.
    pub main_signal_dropped: u64,
    /// Events for blocks not local to the worker.
    remote: u64,
    /// Events owned by a residency agent rather than the engine.
    foreign_owner: u64,
    /// Removals on a tier that held no counted copy of the block.
    unknown_copy: u64,
    /// Stores of a block already indexed on that tier (a second physical copy).
    duplicate_copies: u64,
    /// Copy counts that hit [`COPIES_CAP`].
    capped_copies: u64,
    /// Stores whose parent the index did not hold, placed as a new chain from
    /// the root instead, and the blocks they carried. Every one duplicates
    /// content the chain may already hold further down, so a rising count is
    /// where fragmentation of the chain index is looked for first.
    pub parentless_stores: u64,
    /// Blocks placed in chains started over without their parent.
    pub parentless_blocks: u64,
    /// Stores the index declined because a run's child table or hash array would outgrow
    /// an arena chunk, and the blocks they left out (the leading blocks a declined store
    /// placed are not among them); `smg_kv_index_cap_refusals` counts the stores over the
    /// index, these are per worker.
    pub cap_refused_stores: u64,
    /// Blocks left out of a store that reached an arena cap.
    pub cap_refused_blocks: u64,
    /// Stores of blocks narrower than the worker's block of record, and the
    /// blocks they carried: a hybrid engine's prefix entries at its hash
    /// granularity (the prompt's tail inside its last block), or an engine's
    /// last short page. Not indexed: a request is chunked at the block of
    /// record and never matches one, and their parents are intermediate
    /// hashes the engine does not publish, so indexed they would read as
    /// parent-less stores and seed root chains that only collide.
    pub sub_block_stores: u64,
    /// Blocks left out because the store was narrower than the block of record.
    pub sub_block_blocks: u64,
}

impl WorkerIndexCounters {
    /// Parent-less stores and blocks this state saw since `before`.
    pub fn parentless_since(&self, before: &Self) -> (u64, u64) {
        (
            self.parentless_stores - before.parentless_stores,
            self.parentless_blocks - before.parentless_blocks,
        )
    }

    /// Stores declined at an arena cap and the blocks they left out since `before`.
    pub fn cap_refused_since(&self, before: &Self) -> (u64, u64) {
        (
            self.cap_refused_stores - before.cap_refused_stores,
            self.cap_refused_blocks - before.cap_refused_blocks,
        )
    }

    /// Sub-block stores and blocks this state saw since `before`.
    pub fn sub_block_since(&self, before: &Self) -> (u64, u64) {
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

/// What a worker's stream has shown about its cache groups.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GroupsShown {
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
    /// ([`Self::main_group_shown`]).
    main_signal_groups: Vec<u32>,
}

impl GroupsShown {
    /// Where an event with these attributes goes. A store names its group's
    /// kind; later events for that group may omit it, so the groups learned
    /// from the stream decide for them, and an unlearned group without a kind is the
    /// main group (an engine with one group names nothing). A store of a
    /// group other than main attention that carries tokens, on a worker that
    /// has shown no main group, is the worker's main signal for the index,
    /// and its group's kind-less removals follow it there.
    #[inline]
    pub fn admit(
        &mut self,
        kind: Option<&str>,
        group_idx: Option<u32>,
        tokenful: bool,
        locality: Option<i32>,
        ownership: Option<&str>,
    ) -> Admission {
        if foreign_owner(ownership) {
            return Admission::ForeignOwner;
        }
        if remote(locality) {
            return Admission::Remote;
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
                Admission::MainSignal
            }
            (Some(_), Some(group)) => {
                self.show_group(group);
                Admission::Group(group)
            }
            (Some(_), None) => Admission::GroupNameless,
            (None, Some(group)) => match (self.is_main_signal(group), self.knows_group(group)) {
                (true, true) => Admission::MainAndGroup(group),
                (false, true) => Admission::Group(group),
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
    /// and should be applied last.
    #[inline]
    pub fn is_group_store(&self, stored: &KvBlocksStored) -> bool {
        match (stored.kv_cache_spec_kind.as_deref(), stored.group_idx) {
            (Some(kind), Some(_)) => {
                !MAIN_ATTENTION_KINDS.contains(&kind) && (self.main_shown || !tokenful(stored))
            }
            (None, Some(group)) => self.knows_group(group) && !self.is_main_signal(group),
            _ => false,
        }
    }

    /// A main group shows. The first on a worker whose main signal was a
    /// window group restarts it: `true`, and the index lets the signal's
    /// blocks go, the main group's stores rebuild it, and the signal's
    /// groups are groups of another kind from here.
    fn main_group_shown(&mut self) -> bool {
        if self.main_shown {
            return false;
        }
        self.main_shown = true;
        if self.main_signal_groups.is_empty() {
            return false;
        }
        for group in std::mem::take(&mut self.main_signal_groups) {
            self.show_group(group);
        }
        true
    }

    /// Read the groups a batch shows before applying any event. On restart,
    /// return the first group and group count of the old main signal. The
    /// flag says whether the worker needs a group view.
    #[inline]
    pub fn show_batch(&mut self, events: &[KvCacheEvent]) -> (Option<(u32, usize)>, bool) {
        // The main group first, so a window group's store ahead of it in
        // the batch reads as a group's, not as the worker's main signal. A
        // store the index skips (a residency agent's, a remote tier's) shows
        // no group here either.
        let mut restarted = None;
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
                if main && !self.main_shown {
                    let signal_groups = &self.main_signal_groups;
                    let signal = signal_groups
                        .first()
                        .map(|&group| (group, signal_groups.len()));
                    if self.main_group_shown() {
                        restarted = signal;
                        hybrid = true;
                    }
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
                    if self.main_shown || !tokenful(stored) {
                        self.show_group(group);
                        hybrid = true;
                    } else {
                        self.main_signal(group);
                    }
                }
            }
        }
        (restarted, hybrid)
    }
}

/// A worker's share of the index: its reverse map plus
/// the physical copies of each block per tier, once the worker reports more
/// than one copy or a tier other than the device.
///
/// The engines do not deduplicate physical blocks: one recomputes the last
/// block of an exact resend into a second copy with the same hash and emits
/// a removal per copy, and another's host cache keeps a host copy next to
/// the device one. The relay forwards every store and removal, so this state
/// counts copies and lets a removal evict the block only when none remains.
///
/// Counting is sparse. A block gets an entry in `copies` only on a host
/// store or on a second store of an indexed hash; an indexed block without
/// an entry is a single device copy. A worker that never duplicates and
/// never offloads pays one lookup per stored block and no memory.
#[derive(Default)]
pub struct WorkerIndexState<H: HeldBlocks> {
    /// The indexer's caller-owned reverse map for this worker.
    pub blocks: H,
    /// Copies per tier of blocks with a host copy or more than one copy.
    copies: HashMap<SequenceHash, Copies>,
    /// The worker's cache groups other than main attention, in the index's
    /// registry: created at the worker's first batch carrying a store of
    /// such a group.
    pub groups: Option<Arc<WorkerGroups>>,
    shown: GroupsShown,
    /// Tokens per block of the widest block the worker's stores carried so
    /// far, 0 until a store names a width: the worker's block of record.
    block_of_record: i32,
    /// The block of record a clear dropped, until the worker's first store
    /// after it names a width: a narrower one is logged, since the model-level
    /// size that chunks requests keeps the old width.
    record_before_clear: i32,
    /// What happened to the worker's events as the index applied them.
    pub counters: WorkerIndexCounters,
}

impl<H: HeldBlocks> WorkerIndexState<H> {
    /// The index dropped every block of the worker: at a clear of its
    /// stream ([`apply_cleared`]), or by itself, where an index empties a
    /// worker's blocks in its own turn and says so here before the worker's
    /// next event. The worker's group view is emptied, and its copies and
    /// its block of record are forgotten, as a clear leaves them.
    pub fn index_emptied(&mut self) {
        if let Some(view) = &self.groups {
            view.clear();
        }
        self.copies.clear();
        self.record_before_clear = self.block_of_record;
        self.block_of_record = 0;
    }

    /// The worker's block of record: tokens per block of the widest block
    /// its stores carried since its last clear, and none until a store
    /// names a width.
    pub fn block_of_record(&self) -> Option<u32> {
        u32::try_from(self.block_of_record)
            .ok()
            .filter(|&width| width > 0)
    }

    /// Count the verdict and place any group under its view mask.
    fn admit(
        &mut self,
        kind: Option<&str>,
        group_idx: Option<u32>,
        tokenful: bool,
        locality: Option<i32>,
        ownership: Option<&str>,
    ) -> Placement {
        match self
            .shown
            .admit(kind, group_idx, tokenful, locality, ownership)
        {
            Admission::Main => Placement::Main,
            Admission::MainSignal => {
                self.counters.window_only_stores += 1;
                Placement::Main
            }
            Admission::Group(group) => Placement::Group(group_mask(group)),
            Admission::MainAndGroup(group) => Placement::MainAndGroup(group_mask(group)),
            Admission::ForeignOwner => {
                self.counters.foreign_owner += 1;
                Placement::Skip
            }
            Admission::Remote => {
                self.counters.remote += 1;
                Placement::Skip
            }
            Admission::GroupNameless => {
                self.counters.group_nameless += 1;
                Placement::Skip
            }
        }
    }

    /// The worker's group view, created in the index's registry at its first
    /// batch carrying a store of a cache group other than main attention.
    fn view<I: BlockIndex<Held = H>>(&mut self, indexer: &I, worker_id: u32) -> Arc<WorkerGroups> {
        Arc::clone(self.groups.get_or_insert_with(|| indexer.groups(worker_id)))
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
mod tests;
