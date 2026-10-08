//! The relay's record of the engine's live blocks, and the state snapshot it
//! serves a subscriber from that record.
//!
//! [`LiveState`] is a function of the normalized stream the relay forwards,
//! kept per data-parallel rank, tier and engine hash: a `Stored` adds one
//! physical copy of each of its blocks together with what emitting the block
//! again takes (parent, tokens, LoRA id, cache level, extra keys, and the
//! event's tier, medium, group, locality, ownership, session and namespace);
//! a `Removed` takes one copy away and drops the block at zero; an
//! `AllBlocksCleared` empties the rank. Copies are capped per block the way
//! the gateway caps them ([`COPIES_CAP`]), so a snapshot rebuilds the copy
//! counts the gateway would hold after the same stream. Memory is one entry
//! per live `(rank, tier, hash)`, about 230 bytes at block size 16; nothing
//! is kept per relayed batch.
//!
//! A snapshot ([`LiveState::snapshot`]) is one pass over the entries that
//! clones the per-block `Arc`s. The relay takes it under its lock together
//! with the sequence the state is current through, so the cut is atomic:
//! every batch up to that sequence is in it and every later one reaches the
//! subscriber through the live channel afterwards. Everything else, the
//! parent-first order, the run merging and the batch protos, happens outside
//! the lock in [`SnapshotChunks`] as the subscriber's stream is polled, so a
//! slow snapshot reader never holds up the publisher task or any other
//! subscriber. Each rank keeps its entries in a dense slot vector in store
//! order under a hash index, so the pass is a sequential scan whose `Arc`
//! clones follow the records' allocation order: tens of milliseconds for a
//! pool of several hundred thousand blocks in a release build, which is the
//! worst live latency another subscriber sees while a snapshot is taken;
//! the ordering pass and the chunk encoding that follow run off the lock.
//!
//! The chunks of one snapshot carry consecutive sequence numbers ending at
//! the sequence the state was taken at, so the gateway's per-rank cursor
//! continues into the live stream with no gap and no duplicate; chunk 0
//! begins with an `AllBlocksCleared`, and every chunk is marked with
//! [`KvSnapshotChunk`](common::KvSnapshotChunk). The chunk size grows from
//! [`CHUNK_BLOCKS`] only when the stamping needs fewer chunks than the
//! default would make (a state far larger than the number of sequences
//! behind it), and then never beyond twice the largest store any single
//! relayed batch could have carried.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use smg_grpc_client::common_proto::{self as common, kv_cache_event, KvCacheTier};

/// The most physical copies of one block counted per tier: the gateway's
/// `COPIES_CAP` in `kv_event_monitor.rs`, mirrored so a snapshot rebuilds
/// the counts the gateway would hold.
pub const COPIES_CAP: u32 = 8;

/// Blocks per snapshot chunk unless the stamping rule needs larger chunks.
pub const CHUNK_BLOCKS: usize = 2_048;

/// What a `Stored` event says about all of its blocks: shared by the blocks
/// of one event, and by consecutive events that repeat it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct StoredTail {
    pub tier: Option<i32>,
    pub medium: Option<String>,
    pub group_idx: Option<u32>,
    pub kv_cache_spec_kind: Option<String>,
    pub kv_cache_spec_sliding_window: Option<u32>,
    pub locality: Option<i32>,
    pub ownership: Option<String>,
    pub session_id: Option<String>,
    pub lora_name: Option<String>,
    pub cache_salt: Option<String>,
}

impl StoredTail {
    fn of(stored: &common::KvBlocksStored) -> Self {
        Self {
            tier: stored.tier,
            medium: stored.medium.clone(),
            group_idx: stored.group_idx,
            kv_cache_spec_kind: stored.kv_cache_spec_kind.clone(),
            kv_cache_spec_sliding_window: stored.kv_cache_spec_sliding_window,
            locality: stored.locality,
            ownership: stored.ownership.clone(),
            session_id: stored.session_id.clone(),
            lora_name: stored.lora_name.clone(),
            cache_salt: stored.cache_salt.clone(),
        }
    }

    /// Whether `stored` carries exactly this tail (without allocating one).
    fn matches(&self, stored: &common::KvBlocksStored) -> bool {
        self.tier == stored.tier
            && self.medium == stored.medium
            && self.group_idx == stored.group_idx
            && self.kv_cache_spec_kind == stored.kv_cache_spec_kind
            && self.kv_cache_spec_sliding_window == stored.kv_cache_spec_sliding_window
            && self.locality == stored.locality
            && self.ownership == stored.ownership
            && self.session_id == stored.session_id
            && self.lora_name == stored.lora_name
            && self.cache_salt == stored.cache_salt
    }
}

/// One live block, as the relay can emit it again.
#[derive(Debug)]
pub(crate) struct LiveBlock {
    pub hash: i64,
    /// The hash the block chains from: the previous block of its store, or
    /// the store's `parent_block_hash` for the first.
    pub parent: Option<i64>,
    /// The tier it is live on (a `KvCacheTier` value).
    pub tier: i32,
    pub tokens: Box<[u32]>,
    pub block_size: i32,
    pub lora_id: Option<i64>,
    pub cache_level: Option<i32>,
    pub extra_keys: Box<[common::KvBlockExtraKey]>,
    pub tail: Arc<StoredTail>,
    /// Store order across the relay's lifetime; a snapshot keeps it.
    pub order: u64,
}

struct Entry {
    block: Arc<LiveBlock>,
    copies: u32,
}

/// One rank's live blocks: a dense slot vector in store order (freed slots
/// are reused) under a `(tier, hash)` index, so a snapshot is one
/// sequential scan whose `Arc` clones follow the records' allocation order
/// instead of a hash map's.
#[derive(Default)]
struct RankBlocks {
    slots: Vec<Option<Entry>>,
    index: HashMap<(i32, i64), u32>,
    free: Vec<u32>,
    live: usize,
}

impl RankBlocks {
    fn len(&self) -> usize {
        self.live
    }

    fn is_empty(&self) -> bool {
        self.live == 0
    }

    fn get(&self, key: (i32, i64)) -> Option<&Entry> {
        let slot = *self.index.get(&key)?;
        self.slots[slot as usize].as_ref()
    }

    fn get_mut(&mut self, key: (i32, i64)) -> Option<&mut Entry> {
        let slot = *self.index.get(&key)?;
        self.slots[slot as usize].as_mut()
    }

    /// Store a block the rank does not hold on that tier.
    fn insert(&mut self, key: (i32, i64), entry: Entry) {
        let slot = match self.free.pop() {
            Some(slot) => {
                self.slots[slot as usize] = Some(entry);
                slot
            }
            None => {
                self.slots.push(Some(entry));
                u32::try_from(self.slots.len() - 1).unwrap_or(u32::MAX)
            }
        };
        self.index.insert(key, slot);
        self.live += 1;
    }

    fn remove(&mut self, key: (i32, i64)) -> Option<Entry> {
        let slot = self.index.remove(&key)?;
        let entry = self.slots[slot as usize].take();
        if entry.is_some() {
            self.free.push(slot);
            self.live -= 1;
        }
        entry
    }

    fn iter(&self) -> impl Iterator<Item = &Entry> {
        self.slots.iter().flatten()
    }
}

/// What the state has done since the relay started.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct StateCounts {
    /// Physical copies added by stores.
    pub stored: u64,
    /// Copies taken away by removals.
    pub removed: u64,
    /// Removals of a block the state did not hold on that tier.
    pub removed_unknown: u64,
    /// Stores of a block already at [`COPIES_CAP`] copies on its tier.
    pub capped: u64,
    /// Ranks emptied by `AllBlocksCleared`.
    pub cleared: u64,
}

/// The engine's live blocks per rank, as the relayed stream describes them.
#[derive(Default)]
pub struct LiveState {
    ranks: BTreeMap<Option<i32>, RankBlocks>,
    /// The last store's tail, shared with the next store that repeats it.
    last_tail: Option<Arc<StoredTail>>,
    order: u64,
    /// Live physical copies over all ranks and tiers.
    blocks: u64,
    counts: StateCounts,
}

/// The tier a store or removal names, as the gateway reads it: `tier` when
/// set, else the blocks' `cache_level` (absent means the device).
fn tier_key(tier: Option<i32>, cache_level: Option<i32>) -> i32 {
    match tier {
        Some(tier) if tier != KvCacheTier::Unspecified as i32 => tier,
        _ => match cache_level.unwrap_or(0) {
            0 => KvCacheTier::Device as i32,
            1 => KvCacheTier::Host as i32,
            2 => KvCacheTier::Disk as i32,
            _ => KvCacheTier::External as i32,
        },
    }
}

impl LiveState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one normalized batch into the state.
    pub fn apply(&mut self, batch: &common::KvEventBatch) {
        for event in &batch.events {
            match &event.data {
                Some(kv_cache_event::Data::Stored(stored)) => self.store(batch.dp_rank, stored),
                Some(kv_cache_event::Data::Removed(removed)) => {
                    self.remove(batch.dp_rank, removed);
                }
                Some(kv_cache_event::Data::Cleared(_)) => self.clear_rank(batch.dp_rank),
                None => {}
            }
        }
    }

    fn tail_for(&mut self, stored: &common::KvBlocksStored) -> Arc<StoredTail> {
        if let Some(tail) = &self.last_tail {
            if tail.matches(stored) {
                return Arc::clone(tail);
            }
        }
        let tail = Arc::new(StoredTail::of(stored));
        self.last_tail = Some(Arc::clone(&tail));
        tail
    }

    fn store(&mut self, rank: Option<i32>, stored: &common::KvBlocksStored) {
        if stored.blocks.is_empty() {
            return;
        }
        let tier = tier_key(
            stored.tier,
            stored.blocks.first().and_then(|block| block.cache_level),
        );
        let tail = self.tail_for(stored);
        let blocks = self.ranks.entry(rank).or_default();
        let mut parent = stored.parent_block_hash;
        for block in &stored.blocks {
            let key = (tier, block.block_hash);
            if let Some(entry) = blocks.get_mut(key) {
                if entry.copies < COPIES_CAP {
                    entry.copies += 1;
                    self.blocks += 1;
                } else {
                    self.counts.capped += 1;
                }
            } else {
                self.order += 1;
                blocks.insert(
                    key,
                    Entry {
                        block: Arc::new(LiveBlock {
                            hash: block.block_hash,
                            parent,
                            tier,
                            tokens: block.token_ids.clone().into_boxed_slice(),
                            block_size: block.block_size,
                            lora_id: block.lora_id,
                            cache_level: block.cache_level,
                            extra_keys: block.extra_keys.clone().into_boxed_slice(),
                            tail: Arc::clone(&tail),
                            order: self.order,
                        }),
                        copies: 1,
                    },
                );
                self.blocks += 1;
            }
            self.counts.stored += 1;
            parent = Some(block.block_hash);
        }
    }

    fn remove(&mut self, rank: Option<i32>, removed: &common::KvBlocksRemoved) {
        let tier = tier_key(removed.tier, removed.cache_level);
        let Some(blocks) = self.ranks.get_mut(&rank) else {
            self.counts.removed_unknown += removed.block_hashes.len() as u64;
            return;
        };
        for &hash in &removed.block_hashes {
            let key = (tier, hash);
            let Some(entry) = blocks.get_mut(key) else {
                self.counts.removed_unknown += 1;
                continue;
            };
            entry.copies -= 1;
            self.blocks -= 1;
            self.counts.removed += 1;
            if entry.copies == 0 {
                blocks.remove(key);
            }
        }
    }

    fn clear_rank(&mut self, rank: Option<i32>) {
        if let Some(blocks) = self.ranks.remove(&rank) {
            let copies: u64 = blocks.iter().map(|entry| u64::from(entry.copies)).sum();
            self.blocks -= copies;
        }
        self.counts.cleared += 1;
    }

    /// Forget everything: the publisher started over.
    pub fn clear(&mut self) {
        self.ranks.clear();
        self.last_tail = None;
        self.blocks = 0;
    }

    /// Live `(rank, tier, hash)` entries.
    pub fn entries(&self) -> usize {
        self.ranks.values().map(RankBlocks::len).sum()
    }

    /// Live physical copies.
    pub fn blocks(&self) -> u64 {
        self.blocks
    }

    /// Ranks holding at least one block.
    pub fn ranks(&self) -> usize {
        self.ranks.values().filter(|rank| !rank.is_empty()).count()
    }

    pub(crate) fn counts(&self) -> &StateCounts {
        &self.counts
    }

    /// Live copies of `hash` on `tier` for `rank`.
    pub fn copies(&self, rank: Option<i32>, tier: KvCacheTier, hash: i64) -> u32 {
        self.ranks
            .get(&rank)
            .and_then(|blocks| blocks.get((tier as i32, hash)))
            .map_or(0, |entry| entry.copies)
    }

    /// The live set as it stands: the per-block records shared, nothing
    /// copied. One pass over the entries; the caller orders and encodes it
    /// outside the lock with [`SnapshotChunks::new`].
    pub fn snapshot(&self) -> Snapshot {
        let ranks = self
            .ranks
            .iter()
            .filter(|(_, blocks)| !blocks.is_empty())
            .map(|(rank, blocks)| {
                let mut entries = Vec::with_capacity(blocks.len());
                entries.extend(blocks.iter().map(|entry| SnapshotEntry {
                    block: Arc::clone(&entry.block),
                    copies: entry.copies,
                }));
                (*rank, entries)
            })
            .collect();
        Snapshot {
            ranks,
            blocks: self.blocks,
        }
    }
}

/// One live block of a snapshot and how many copies the engine holds.
pub(crate) struct SnapshotEntry {
    pub(crate) block: Arc<LiveBlock>,
    pub(crate) copies: u32,
}

/// The live set at one point of the stream, unordered.
pub struct Snapshot {
    /// Per rank, in rank order; ranks without blocks are left out.
    pub(crate) ranks: Vec<(Option<i32>, Vec<SnapshotEntry>)>,
    /// Physical copies over all ranks.
    pub(crate) blocks: u64,
}

impl Snapshot {
    pub fn entries(&self) -> usize {
        self.ranks.iter().map(|(_, entries)| entries.len()).sum()
    }
}

/// A snapshot ordered and cut into the batches a subscriber receives, built
/// lazily chunk by chunk.
pub struct SnapshotChunks {
    /// Per rank, one unit per physical copy in emission order: parents
    /// before children, store order otherwise, copies adjacent.
    ranks: Vec<(Option<i32>, Vec<Arc<LiveBlock>>)>,
    chunk_blocks: usize,
    count: u32,
    next: u32,
    /// `(rank, offset)` of the next chunk's first unit.
    position: (usize, usize),
    through: u64,
    timestamp: f64,
    blocks: u64,
    unknown_before: u64,
}

impl SnapshotChunks {
    /// Order `snapshot`, taken at sequence `through`, and size its chunks so
    /// that they can be stamped `through - count + 1 ..= through`; `timestamp`
    /// is what the chunks carry (the gateway reads it as the publish time),
    /// `unknown_before` how many publisher sequences before the relay's
    /// record the snapshot cannot cover (0 for a record from the start).
    pub fn new(snapshot: Snapshot, through: u64, timestamp: f64, unknown_before: u64) -> Self {
        let blocks = snapshot.blocks;
        let ranks: Vec<(Option<i32>, Vec<Arc<LiveBlock>>)> = snapshot
            .ranks
            .into_iter()
            .map(|(rank, entries)| (rank, order_rank(entries)))
            .collect();
        let largest = ranks
            .iter()
            .map(|(_, units)| units.len())
            .max()
            .unwrap_or(0);
        let chunks_at = |chunk_blocks: usize| -> usize {
            ranks
                .iter()
                .map(|(_, units)| units.len().div_ceil(chunk_blocks))
                .sum::<usize>()
                .max(1)
        };
        // Every rank in the state came from at least one relayed batch, so
        // one chunk per rank always fits under `through + 1`; the loop only
        // runs when the default chunk would need more sequences than exist.
        let mut chunk_blocks = CHUNK_BLOCKS;
        while chunks_at(chunk_blocks) as u64 > through + 1 && chunk_blocks < largest {
            chunk_blocks *= 2;
        }
        let count = u32::try_from(chunks_at(chunk_blocks)).unwrap_or(u32::MAX);
        Self {
            ranks,
            chunk_blocks,
            count,
            next: 0,
            position: (0, 0),
            through,
            timestamp,
            blocks,
            unknown_before,
        }
    }

    /// Chunks in the snapshot (at least one: the clear).
    pub fn chunk_count(&self) -> u32 {
        self.count
    }

    /// Physical copies in the snapshot.
    pub fn blocks(&self) -> u64 {
        self.blocks
    }

    /// The sequence the state was taken at: the last chunk's stamp.
    pub fn through(&self) -> u64 {
        self.through
    }

    /// The next chunk, or `None` once all `count` have been produced.
    pub(crate) fn next_chunk(&mut self) -> Option<common::KvEventBatch> {
        if self.next >= self.count {
            return None;
        }
        let index = self.next;
        self.next += 1;
        let sequence_number = self
            .through
            .saturating_sub(u64::from(self.count) - 1)
            .saturating_add(u64::from(index));
        let mut events = Vec::new();
        if index == 0 {
            events.push(common::KvCacheEvent {
                event_id: 0,
                data: Some(kv_cache_event::Data::Cleared(
                    common::KvCacheCleared::default(),
                )),
            });
        }
        let mut dp_rank = None;
        while self.position.0 < self.ranks.len() {
            let (rank_index, offset) = self.position;
            let (rank, units) = &self.ranks[rank_index];
            if offset >= units.len() {
                self.position = (rank_index + 1, 0);
                continue;
            }
            let end = (offset + self.chunk_blocks).min(units.len());
            self.position = if end == units.len() {
                (rank_index + 1, 0)
            } else {
                (rank_index, end)
            };
            dp_rank = *rank;
            stored_events(&units[offset..end], &mut events);
            break;
        }
        Some(common::KvEventBatch {
            sequence_number,
            timestamp: self.timestamp,
            events,
            dp_rank,
            snapshot: Some(common::KvSnapshotChunk {
                index,
                count: self.count,
                blocks: self.blocks,
                unknown_before: self.unknown_before,
            }),
            load: None,
        })
    }
}

impl Iterator for SnapshotChunks {
    type Item = common::KvEventBatch;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_chunk()
    }
}

/// One rank's entries in emission order, one unit per physical copy: store
/// order, with the live ancestors of a block hoisted ahead of it when they
/// were stored later (a parent evicted and stored again under a surviving
/// child), so the gateway's index derives every position from a parent it
/// already holds.
fn order_rank(mut entries: Vec<SnapshotEntry>) -> Vec<Arc<LiveBlock>> {
    entries.sort_unstable_by_key(|entry| entry.block.order);
    let mut by_hash: HashMap<i64, Vec<usize>> = HashMap::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        by_hash.entry(entry.block.hash).or_default().push(index);
    }
    let units: usize = entries.iter().map(|entry| entry.copies as usize).sum();
    let mut out = Vec::with_capacity(units);
    let mut done = vec![false; entries.len()];
    let mut path: Vec<i64> = Vec::new();
    let push_units = |index: usize, out: &mut Vec<Arc<LiveBlock>>| {
        let entry = &entries[index];
        for _ in 0..entry.copies {
            out.push(Arc::clone(&entry.block));
        }
    };
    for index in 0..entries.len() {
        if done[index] {
            continue;
        }
        path.clear();
        let mut cursor = entries[index].block.parent;
        while let Some(hash) = cursor {
            let Some(indices) = by_hash.get(&hash) else {
                // Not live: the gateway starts a chain there, as it did
                // when the parent left the engine before its child.
                break;
            };
            if done[indices[0]] {
                break;
            }
            // Every tier's entry of a hash is marked and emitted together.
            for &sibling in indices {
                done[sibling] = true;
            }
            path.push(hash);
            cursor = entries[indices[0]].block.parent;
        }
        for hash in path.iter().rev() {
            for &ancestor in &by_hash[hash] {
                push_units(ancestor, &mut out);
            }
        }
        for &sibling in &by_hash[&entries[index].block.hash] {
            if !done[sibling] {
                done[sibling] = true;
                push_units(sibling, &mut out);
            }
        }
    }
    out
}

/// Whether unit `at` is one of several copies of the same block (copies are
/// adjacent in the emission order).
fn is_copy(units: &[Arc<LiveBlock>], at: usize) -> bool {
    (at > 0 && Arc::ptr_eq(&units[at - 1], &units[at]))
        || (at + 1 < units.len() && Arc::ptr_eq(&units[at], &units[at + 1]))
}

/// `Stored` events for `units`: a run of blocks chaining one from the other
/// under one tail and tier becomes one event, as the engine stored them;
/// each extra copy of a block is its own single-block event so the gateway
/// counts it.
fn stored_events(units: &[Arc<LiveBlock>], events: &mut Vec<common::KvCacheEvent>) {
    let mut start = 0;
    while start < units.len() {
        let mut end = start + 1;
        if !is_copy(units, start) {
            while end < units.len()
                && !is_copy(units, end)
                && units[end].parent == Some(units[end - 1].hash)
                && units[end].tier == units[end - 1].tier
                && Arc::ptr_eq(&units[end].tail, &units[end - 1].tail)
            {
                end += 1;
            }
        }
        events.push(stored_event(&units[start..end]));
        start = end;
    }
}

fn stored_event(run: &[Arc<LiveBlock>]) -> common::KvCacheEvent {
    let tail = &run[0].tail;
    common::KvCacheEvent {
        event_id: 0,
        data: Some(kv_cache_event::Data::Stored(common::KvBlocksStored {
            blocks: run
                .iter()
                .map(|block| common::KvBlock {
                    block_hash: block.hash,
                    token_ids: block.tokens.to_vec(),
                    block_size: block.block_size,
                    lora_id: block.lora_id,
                    cache_level: block.cache_level,
                    extra_keys: block.extra_keys.to_vec(),
                })
                .collect(),
            parent_block_hash: run[0].parent,
            tier: tail.tier,
            medium: tail.medium.clone(),
            group_idx: tail.group_idx,
            kv_cache_spec_kind: tail.kv_cache_spec_kind.clone(),
            kv_cache_spec_sliding_window: tail.kv_cache_spec_sliding_window,
            locality: tail.locality,
            ownership: tail.ownership.clone(),
            session_id: tail.session_id.clone(),
            lora_name: tail.lora_name.clone(),
            cache_salt: tail.cache_salt.clone(),
        })),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use smg_grpc_client::common_proto::{KvBlock, KvBlocksRemoved, KvBlocksStored, KvCacheEvent};

    use super::*;

    fn block(hash: i64) -> KvBlock {
        KvBlock {
            block_hash: hash,
            token_ids: (0..4u32)
                .map(|i| (hash as u32).wrapping_mul(16).wrapping_add(i))
                .collect(),
            block_size: 4,
            ..Default::default()
        }
    }

    fn stored(parent: Option<i64>, hashes: &[i64], tier: KvCacheTier) -> KvCacheEvent {
        KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                blocks: hashes.iter().map(|&hash| block(hash)).collect(),
                parent_block_hash: parent,
                tier: Some(tier as i32),
                medium: Some("GPU".to_string()),
                group_idx: Some(0),
                kv_cache_spec_kind: Some("full_attention".to_string()),
                ..Default::default()
            })),
        }
    }

    fn removed(hashes: &[i64], tier: KvCacheTier) -> KvCacheEvent {
        KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Removed(KvBlocksRemoved {
                block_hashes: hashes.to_vec(),
                tier: Some(tier as i32),
                ..Default::default()
            })),
        }
    }

    fn cleared() -> KvCacheEvent {
        KvCacheEvent {
            event_id: 1,
            data: Some(kv_cache_event::Data::Cleared(
                common::KvCacheCleared::default(),
            )),
        }
    }

    fn batch(seq: u64, rank: Option<i32>, events: Vec<KvCacheEvent>) -> common::KvEventBatch {
        common::KvEventBatch {
            sequence_number: seq,
            timestamp: 1.0,
            events,
            dp_rank: rank,
            snapshot: None,
            load: None,
        }
    }

    /// `(rank, tier, hash, parent)` per block of every stored event, in
    /// emission order.
    fn emitted(chunks: &[common::KvEventBatch]) -> Vec<(Option<i32>, i32, i64, Option<i64>)> {
        let mut out = Vec::new();
        for chunk in chunks {
            for event in &chunk.events {
                if let Some(kv_cache_event::Data::Stored(stored)) = &event.data {
                    let mut parent = stored.parent_block_hash;
                    for block in &stored.blocks {
                        out.push((
                            chunk.dp_rank,
                            stored.tier.unwrap(),
                            block.block_hash,
                            parent,
                        ));
                        parent = Some(block.block_hash);
                    }
                }
            }
        }
        out
    }

    fn chunks_of(state: &LiveState, through: u64) -> Vec<common::KvEventBatch> {
        SnapshotChunks::new(state.snapshot(), through, 2.0, 0).collect()
    }

    #[test]
    fn stores_add_copies_removals_take_them_and_a_clear_empties_the_rank() {
        let device = KvCacheTier::Device;
        let mut state = LiveState::new();
        state.apply(&batch(1, Some(0), vec![stored(None, &[1, 2], device)]));
        state.apply(&batch(2, Some(0), vec![stored(Some(2), &[3], device)]));
        state.apply(&batch(3, Some(1), vec![stored(None, &[1], device)]));
        assert_eq!((state.entries(), state.blocks(), state.ranks()), (4, 4, 2));
        // A second physical copy of 2 (vLLM's resend) counts without a new entry.
        state.apply(&batch(4, Some(0), vec![stored(Some(1), &[2], device)]));
        assert_eq!((state.entries(), state.blocks()), (4, 5));
        assert_eq!(state.copies(Some(0), device, 2), 2);
        // One removal per copy; the block stays until the last one.
        state.apply(&batch(5, Some(0), vec![removed(&[2], device)]));
        assert_eq!(state.copies(Some(0), device, 2), 1);
        state.apply(&batch(6, Some(0), vec![removed(&[2, 99], device)]));
        assert_eq!(state.copies(Some(0), device, 2), 0);
        assert_eq!(state.counts().removed_unknown, 1);
        assert_eq!((state.entries(), state.blocks()), (3, 3));
        // The host tier is a copy of its own.
        state.apply(&batch(
            7,
            Some(0),
            vec![stored(None, &[1], KvCacheTier::Host)],
        ));
        assert_eq!(state.copies(Some(0), KvCacheTier::Host, 1), 1);
        assert_eq!(state.copies(Some(0), device, 1), 1);
        state.apply(&batch(8, Some(0), vec![removed(&[1], device)]));
        assert_eq!(state.copies(Some(0), device, 1), 0);
        assert_eq!(state.copies(Some(0), KvCacheTier::Host, 1), 1);
        // Clearing rank 0 leaves rank 1 alone.
        state.apply(&batch(9, Some(0), vec![cleared()]));
        assert_eq!((state.entries(), state.blocks(), state.ranks()), (1, 1, 1));
        assert_eq!(state.copies(Some(1), device, 1), 1);
        assert_eq!(state.counts().cleared, 1);
        state.clear();
        assert_eq!((state.entries(), state.blocks(), state.ranks()), (0, 0, 0));
    }

    #[test]
    fn copies_are_capped_like_the_gateway_counts_them() {
        let mut state = LiveState::new();
        for seq in 0..20 {
            state.apply(&batch(
                seq,
                None,
                vec![stored(None, &[7], KvCacheTier::Device)],
            ));
        }
        assert_eq!(state.copies(None, KvCacheTier::Device, 7), COPIES_CAP);
        assert_eq!(state.blocks(), u64::from(COPIES_CAP));
        assert_eq!(state.counts().capped, 20 - u64::from(COPIES_CAP));
        let chunks = chunks_of(&state, 19);
        assert_eq!(emitted(&chunks).len(), COPIES_CAP as usize);
    }

    #[test]
    fn a_snapshot_lists_the_live_set_parents_first_with_the_store_fields() {
        let device = KvCacheTier::Device;
        let mut state = LiveState::new();
        // A child stored before its (re-stored) parent must still follow it.
        state.apply(&batch(1, Some(0), vec![stored(None, &[1, 2, 3], device)]));
        state.apply(&batch(2, Some(0), vec![removed(&[1], device)]));
        state.apply(&batch(3, Some(0), vec![stored(Some(3), &[4], device)]));
        state.apply(&batch(4, Some(0), vec![stored(None, &[1], device)]));
        state.apply(&batch(5, Some(0), vec![stored(Some(4), &[5], device)]));
        let chunks = chunks_of(&state, 5);
        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert_eq!(chunk.sequence_number, 5);
        assert_eq!(chunk.dp_rank, Some(0));
        assert_eq!(
            chunk.snapshot,
            Some(common::KvSnapshotChunk {
                index: 0,
                count: 1,
                blocks: 5,
                unknown_before: 0,
            })
        );
        assert!(
            matches!(chunk.events[0].data, Some(kv_cache_event::Data::Cleared(_))),
            "the clear comes first"
        );
        let order = emitted(&chunks);
        assert_eq!(order.len(), 5);
        let position = |hash: i64| order.iter().position(|entry| entry.2 == hash).unwrap();
        for (_, _, hash, parent) in &order {
            if let Some(parent) = parent {
                assert!(
                    position(*parent) < position(*hash),
                    "{parent} before {hash}: {order:?}"
                );
            }
        }
        // Block 1's chain was broken by the removal: 2 keeps naming 1 as its
        // parent, 1 is live again, so 1 comes first and the run 2, 3 follows.
        assert_eq!(order[0].2, 1);
        assert_eq!(
            &order[1..3],
            &[(Some(0), 1, 2, Some(1)), (Some(0), 1, 3, Some(2))]
        );
        let stores: Vec<&KvBlocksStored> = chunk
            .events
            .iter()
            .filter_map(|event| match &event.data {
                Some(kv_cache_event::Data::Stored(stored)) => Some(stored),
                _ => None,
            })
            .collect();
        // Every block chains from the one emitted before it under one tail:
        // the five come as one run, the way an engine stores a prompt.
        assert_eq!(stores.len(), 1, "{stores:#?}");
        let first = &stores[0];
        assert_eq!(first.parent_block_hash, None);
        assert_eq!(first.blocks.len(), 5);
        assert_eq!(first.medium.as_deref(), Some("GPU"));
        assert_eq!(first.kv_cache_spec_kind.as_deref(), Some("full_attention"));
        assert_eq!(first.group_idx, Some(0));
        assert_eq!(first.blocks[0].token_ids, block(1).token_ids);
        assert_eq!(first.blocks[0].block_size, 4);
    }

    #[test]
    fn chunks_are_cut_per_rank_and_stamped_up_to_the_cut() {
        let device = KvCacheTier::Device;
        let mut state = LiveState::new();
        let hashes: Vec<i64> = (1..=CHUNK_BLOCKS as i64 + 10).collect();
        state.apply(&batch(1, Some(0), vec![stored(None, &hashes, device)]));
        state.apply(&batch(2, Some(1), vec![stored(None, &[1, 2], device)]));
        let chunks = chunks_of(&state, 40);
        assert_eq!(chunks.len(), 3, "rank 0 takes two chunks, rank 1 one");
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| (chunk.sequence_number, chunk.dp_rank))
                .collect::<Vec<_>>(),
            vec![(38, Some(0)), (39, Some(0)), (40, Some(1))]
        );
        for (index, chunk) in chunks.iter().enumerate() {
            let marker = chunk.snapshot.as_ref().unwrap();
            assert_eq!((marker.index as usize, marker.count), (index, 3));
            assert_eq!(marker.blocks, CHUNK_BLOCKS as u64 + 12);
            assert_eq!(
                matches!(chunk.events[0].data, Some(kv_cache_event::Data::Cleared(_))),
                index == 0
            );
        }
        // The split run continues from the first chunk's last block.
        let second = emitted(&chunks[1..2]);
        assert_eq!(
            second[0],
            (
                Some(0),
                1,
                CHUNK_BLOCKS as i64 + 1,
                Some(CHUNK_BLOCKS as i64)
            )
        );
        assert_eq!(emitted(&chunks).len(), CHUNK_BLOCKS + 12);
    }

    #[test]
    fn few_sequences_behind_a_large_state_mean_larger_chunks() {
        let mut state = LiveState::new();
        let hashes: Vec<i64> = (1..=4 * CHUNK_BLOCKS as i64).collect();
        state.apply(&batch(
            0,
            None,
            vec![stored(None, &hashes, KvCacheTier::Device)],
        ));
        state.apply(&batch(
            1,
            None,
            vec![stored(None, &[-1], KvCacheTier::Device)],
        ));
        // Two sequences exist (0 and 1): at most two chunks, stamped 0 and 1.
        let chunks = chunks_of(&state, 1);
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.sequence_number)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(emitted(&chunks).len(), 4 * CHUNK_BLOCKS + 1);
    }

    #[test]
    fn an_empty_state_snapshots_as_one_clear() {
        let state = LiveState::new();
        let chunks = chunks_of(&state, 12);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].sequence_number, 12);
        assert_eq!(chunks[0].events.len(), 1);
        assert_eq!(
            chunks[0].snapshot,
            Some(common::KvSnapshotChunk {
                index: 0,
                count: 1,
                blocks: 0,
                unknown_before: 0,
            })
        );
        assert_eq!(chunks[0].dp_rank, None);
        // A record that starts late says so on every chunk.
        let late: Vec<common::KvEventBatch> =
            SnapshotChunks::new(state.snapshot(), 12, 2.0, 7).collect();
        assert_eq!(
            late[0].snapshot.as_ref().map(|chunk| chunk.unknown_before),
            Some(7)
        );
    }

    #[test]
    #[expect(
        clippy::print_stderr,
        reason = "the measured pass is this test's report; read it with --nocapture"
    )]
    fn a_snapshot_of_a_large_state_is_one_brief_pass() {
        // A 676k-block pool (a large worker's) in chains of 64.
        let mut state = LiveState::new();
        let mut hash = 1i64;
        for seq in 0..(676_144 / 64) {
            let hashes: Vec<i64> = (hash..hash + 64).collect();
            hash += 64;
            state.apply(&batch(
                seq,
                Some(0),
                vec![stored(None, &hashes, KvCacheTier::Device)],
            ));
        }
        assert_eq!(state.blocks(), 676_096);
        let started = Instant::now();
        let snapshot = state.snapshot();
        let collected = started.elapsed();
        let started = Instant::now();
        let mut chunks = SnapshotChunks::new(snapshot, 20_000, 3.0, 0);
        let ordered = started.elapsed();
        let started = Instant::now();
        let mut blocks = 0;
        let count = chunks.chunk_count();
        while let Some(chunk) = chunks.next_chunk() {
            blocks += emitted(std::slice::from_ref(&chunk)).len();
        }
        let encoded = started.elapsed();
        eprintln!(
            "676k snapshot: collect {collected:?}, order {ordered:?}, {count} chunks built in {encoded:?}"
        );
        assert_eq!(blocks, 676_096);
        assert_eq!(count as usize, 676_096_usize.div_ceil(CHUNK_BLOCKS));
        assert!(
            collected.as_millis() < 1_000,
            "collecting under the lock took {collected:?}"
        );
    }
}
