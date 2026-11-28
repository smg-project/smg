//! The relay's record of the engine's live blocks, and the state snapshot it
//! serves a subscriber from that record.
//!
//! [`LiveState`] is a function of the normalized stream the relay forwards,
//! kept per data-parallel rank, tier, cache-group lane and engine hash: a
//! `Stored` adds one physical copy of each of its blocks together with what
//! emitting the block again takes (parent, tokens, LoRA id, cache level,
//! extra keys, and the event's tier, medium, group, locality, ownership,
//! session and namespace); a `Removed` takes one copy away and drops the
//! block at zero; an `AllBlocksCleared` empties the rank. Copies are capped
//! per block the way the gateway caps them ([`COPIES_CAP`]), so a snapshot
//! rebuilds the copy counts the gateway would hold after the same stream.
//! The lane keeps a hybrid model's groups apart: the main-attention group's
//! blocks (and every block of a publisher without groups) are lane 0, and a
//! sliding-window or state group's block, published under the same hash, is
//! its own entry on its group's lane, not a second copy of the main one, so
//! a snapshot replays the group's memberships as the stream carried them.
//! Memory is one entry per live `(rank, tier, lane, hash)`, about 230 bytes
//! at block size 16 on the main lane and about 100 bytes on a group lane
//! (no tokens); nothing is kept per relayed batch.
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
//! Each rank's chunks carry consecutive sequence numbers ending at the
//! sequence that rank's record was taken at ([`RankCut`]; every publisher
//! numbers its own batches), so the gateway's per-rank cursor continues into
//! the live stream with no gap and no duplicate; chunk 0 begins with an
//! `AllBlocksCleared`, and every chunk is marked with
//! [`KvSnapshotChunk`](common::KvSnapshotChunk). A chunk holds at most
//! [`CHUNK_BLOCKS`] blocks and at most [`CHUNK_BYTES`] of encoded block
//! entries, whichever is reached first: with large token blocks the bytes
//! bind, so a chunk stays under what a gRPC client decodes by default. Both
//! bounds grow only when the stamping needs fewer chunks than they would make
//! (a state far larger than the number of sequences behind it), and then
//! never beyond twice the largest store any single relayed batch could have
//! carried.

use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use prost::Message;
use smg_grpc_client::common_proto::{self as common, kv_cache_event, KvCacheTier};

use crate::kv_wire::MAIN_ATTENTION_KINDS;

/// The most physical copies of one block counted per tier: the gateway's
/// `COPIES_CAP` in `kv_event_monitor.rs`, mirrored so a snapshot rebuilds
/// the counts the gateway would hold.
pub const COPIES_CAP: u32 = 8;

/// Blocks per snapshot chunk unless the stamping rule needs larger chunks.
pub const CHUNK_BLOCKS: usize = 2_048;

/// Encoded bytes of block entries per snapshot chunk, the bound that binds
/// when blocks are large: [`CHUNK_BLOCKS`] blocks with their token ids are
/// 60-300 KB at 16-128-token blocks but about 4.8 MB at 1,152-token blocks,
/// over the 4 MiB a tonic client decodes by default. The cap leaves that
/// limit room for the events' and the batch's framing at any block size.
/// Grows with [`CHUNK_BLOCKS`] under the stamping rule.
pub const CHUNK_BYTES: usize = 1024 * 1024;

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
    pub group_mask: Option<u64>,
}

impl StoredTail {
    fn of(stored: &common::KvBlocksStored) -> Self {
        Self {
            tier: stored.tier,
            medium: stored.medium.clone(),
            group_idx: stored.group_idx,
            group_mask: stored.group_mask,
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
            && self.group_mask == stored.group_mask
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
    /// The cache-group lane it is live on: 0 for the main-attention group
    /// and for a publisher without groups, `group_idx + 1` for a group of
    /// another kind (see [`LiveState`]).
    pub lane: u32,
    pub tokens: Box<[u32]>,
    pub block_size: i32,
    pub lora_id: Option<i64>,
    pub cache_level: Option<i32>,
    pub extra_keys: Box<[common::KvBlockExtraKey]>,
    pub tail: Arc<StoredTail>,
    /// Store order across the relay's lifetime; a snapshot keeps it.
    pub order: u64,
    /// The block's encoded size on the wire, the same when the relay emits
    /// it again.
    pub wire_len: u32,
}

impl LiveBlock {
    /// The bytes the block takes in a `KvBlocksStored`: its message behind
    /// the field's tag and length.
    fn entry_len(&self) -> usize {
        let len = self.wire_len as usize;
        1 + varint_len(len as u64) + len
    }
}

/// Bytes of `value` as a protobuf varint.
fn varint_len(value: u64) -> usize {
    ((70 - u64::leading_zeros(value | 1)) / 7) as usize
}

struct Entry {
    block: Arc<LiveBlock>,
    copies: u32,
}

/// What names one live entry of a rank: the tier, the cache-group lane and
/// the engine hash.
type BlockKey = (i32, u32, i64);

/// The main-attention lane: every block of a publisher without groups, and
/// the main group's blocks of one with them.
const MAIN_LANE: u32 = 0;

/// One rank's live blocks: a dense slot vector in store order (freed slots
/// are reused) under a `(tier, lane, hash)` index, so a snapshot is one
/// sequential scan whose `Arc` clones follow the records' allocation order
/// instead of a hash map's.
#[derive(Default)]
struct RankBlocks {
    slots: Vec<Option<Entry>>,
    index: HashMap<BlockKey, u32>,
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

    fn get(&self, key: BlockKey) -> Option<&Entry> {
        let slot = *self.index.get(&key)?;
        self.slots[slot as usize].as_ref()
    }

    fn get_mut(&mut self, key: BlockKey) -> Option<&mut Entry> {
        let slot = *self.index.get(&key)?;
        self.slots[slot as usize].as_mut()
    }

    /// Store a block the rank does not hold on that tier and lane.
    fn insert(&mut self, key: BlockKey, entry: Entry) {
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

    fn remove(&mut self, key: BlockKey) -> Option<Entry> {
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
    /// Per rank and cache group, whether the group's kind is main attention,
    /// learned from the stores (which name the kind) for the removals (which
    /// do not). Kept across a clear: the groups describe the model.
    groups: HashMap<(Option<i32>, u32), bool>,
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

    /// The lane an event's cache group puts it on: a store names its kind
    /// and teaches the group; a removal, or a store without a kind, follows
    /// what the group taught; a group never seen, or no group, is the main
    /// lane (a publisher with a single group names none).
    fn lane_of(&mut self, rank: Option<i32>, group_idx: Option<u32>, kind: Option<&str>) -> u32 {
        let Some(group) = group_idx else {
            return MAIN_LANE;
        };
        let main = match kind {
            Some(kind) => {
                let main = MAIN_ATTENTION_KINDS.contains(&kind);
                self.groups.insert((rank, group), main);
                main
            }
            None => self.groups.get(&(rank, group)).copied().unwrap_or(true),
        };
        if main {
            MAIN_LANE
        } else {
            group + 1
        }
    }

    fn store(&mut self, rank: Option<i32>, stored: &common::KvBlocksStored) {
        if stored.blocks.is_empty() {
            return;
        }
        let tier = tier_key(
            stored.tier,
            stored.blocks.first().and_then(|block| block.cache_level),
        );
        let lane = self.lane_of(rank, stored.group_idx, stored.kv_cache_spec_kind.as_deref());
        let tail = self.tail_for(stored);
        let blocks = self.ranks.entry(rank).or_default();
        let mut parent = stored.parent_block_hash;
        for block in &stored.blocks {
            let key = (tier, lane, block.block_hash);
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
                            lane,
                            tokens: block.token_ids.clone().into_boxed_slice(),
                            block_size: block.block_size,
                            lora_id: block.lora_id,
                            cache_level: block.cache_level,
                            extra_keys: block.extra_keys.clone().into_boxed_slice(),
                            tail: Arc::clone(&tail),
                            order: self.order,
                            wire_len: u32::try_from(block.encoded_len()).unwrap_or(u32::MAX),
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
        let lane = self.lane_of(rank, removed.group_idx, None);
        let Some(blocks) = self.ranks.get_mut(&rank) else {
            self.counts.removed_unknown += removed.block_hashes.len() as u64;
            return;
        };
        for &hash in &removed.block_hashes {
            let key = (tier, lane, hash);
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

    pub(crate) fn clear_rank(&mut self, rank: Option<i32>) {
        if let Some(blocks) = self.ranks.remove(&rank) {
            let copies: u64 = blocks.iter().map(|entry| u64::from(entry.copies)).sum();
            self.blocks -= copies;
        }
        self.counts.cleared += 1;
    }

    /// Forget everything: the publisher started over.
    pub fn clear(&mut self) {
        self.ranks.clear();
        self.groups.clear();
        self.last_tail = None;
        self.blocks = 0;
    }

    /// Live `(rank, tier, lane, hash)` entries.
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

    /// Live copies of `hash` on `tier` for `rank`, on the main lane.
    pub fn copies(&self, rank: Option<i32>, tier: KvCacheTier, hash: i64) -> u32 {
        self.ranks
            .get(&rank)
            .and_then(|blocks| blocks.get((tier as i32, MAIN_LANE, hash)))
            .map_or(0, |entry| entry.copies)
    }

    /// Live copies of `hash` on `tier` for `rank` on cache group `group`'s
    /// own lane: 0 for a main-attention group, whose blocks are on the main
    /// lane.
    pub fn group_copies(&self, rank: Option<i32>, tier: KvCacheTier, group: u32, hash: i64) -> u32 {
        if self.groups.get(&(rank, group)).copied().unwrap_or(true) {
            return 0;
        }
        self.ranks
            .get(&rank)
            .and_then(|blocks| blocks.get((tier as i32, group + 1, hash)))
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

/// Where one rank's record stood when a snapshot was taken: the sequence the
/// rank's state is current through, which its last chunk is stamped with so
/// that the rank's live events continue right after it, and how many of the
/// rank's publisher sequences before the relay's record the snapshot cannot
/// cover (0 for a record from the rank's first batch).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RankCut {
    pub through: u64,
    pub unknown_before: u64,
}

/// A snapshot ordered and cut into the batches a subscriber receives, built
/// lazily chunk by chunk.
pub struct SnapshotChunks {
    /// Per rank, one unit per physical copy in emission order: parents
    /// before children, store order otherwise, copies adjacent.
    ranks: Vec<(Option<i32>, Vec<Arc<LiveBlock>>)>,
    /// The units of each chunk, `(rank index, start, end)`, in order; empty
    /// for an empty state, whose one chunk is the clear.
    cuts: Vec<(usize, usize, usize)>,
    /// Each chunk's rank, sequence stamp and `unknown_before`, in chunk order.
    stamps: Vec<(Option<i32>, u64, u64)>,
    count: u32,
    next: u32,
    timestamp: f64,
    blocks: u64,
}

impl SnapshotChunks {
    /// Order `snapshot` and size its chunks so that each rank's can be
    /// stamped `through - chunks + 1 ..= through` of that rank's cut in
    /// `cuts` (a rank without one is cut at 0); `timestamp` is what the
    /// chunks carry (the gateway reads it as the publish time). An empty
    /// state is one chunk, the clear, stamped with the first cut and carrying
    /// its rank (the stamp moves that rank's cursor at the gateway).
    pub fn new(snapshot: Snapshot, cuts: &BTreeMap<Option<i32>, RankCut>, timestamp: f64) -> Self {
        let blocks = snapshot.blocks;
        let mut ranks: Vec<(Option<i32>, Vec<Arc<LiveBlock>>)> = snapshot
            .ranks
            .into_iter()
            .map(|(rank, entries)| (rank, order_rank(entries)))
            .collect();
        // A rank with a group-lane entry goes first: the ranks of one engine
        // share one view at a gateway that pools them, and the view must
        // exist before any rank's main lane is applied (its leading entry,
        // see `order_rank`). Each rank's chunks are stamped from its own cut,
        // so the rank order moves no cursor.
        if let Some(first) = ranks
            .iter()
            .position(|(_, units)| units.first().is_some_and(|unit| unit.lane != MAIN_LANE))
        {
            ranks.swap(0, first);
        }
        let cut_of = |rank: &Option<i32>| cuts.get(rank).copied().unwrap_or_default();
        let largest_units = ranks
            .iter()
            .map(|(_, units)| units.len())
            .max()
            .unwrap_or(0);
        let largest_bytes = ranks
            .iter()
            .map(|(_, units)| units.iter().map(|unit| unit.entry_len()).sum::<usize>())
            .max()
            .unwrap_or(0);
        // Every rank in the state came from at least one relayed batch, so
        // one chunk per rank always fits under its `through + 1`; the loop
        // only runs when the default bounds would need more sequences than a
        // rank has, and stops at the latest once every rank is one chunk.
        let (mut chunk_blocks, mut chunk_bytes) = (CHUNK_BLOCKS, CHUNK_BYTES);
        let mut chunks = cut_chunks(&ranks, chunk_blocks, chunk_bytes);
        let too_many = |chunks: &[(usize, usize, usize)]| {
            chunks_per_rank(ranks.len(), chunks)
                .iter()
                .enumerate()
                .any(|(index, &count)| {
                    count as u64 > cut_of(&ranks[index].0).through.saturating_add(1)
                })
        };
        while too_many(&chunks) && (chunk_blocks < largest_units || chunk_bytes < largest_bytes) {
            chunk_blocks = chunk_blocks.saturating_mul(2);
            chunk_bytes = chunk_bytes.saturating_mul(2);
            chunks = cut_chunks(&ranks, chunk_blocks, chunk_bytes);
        }
        let per_rank = chunks_per_rank(ranks.len(), &chunks);
        let mut seen = vec![0u64; ranks.len()];
        let mut stamps: Vec<(Option<i32>, u64, u64)> = chunks
            .iter()
            .map(|&(rank_index, _, _)| {
                let rank = ranks[rank_index].0;
                let cut = cut_of(&rank);
                let stamp = cut
                    .through
                    .saturating_sub(per_rank[rank_index] as u64 - 1)
                    .saturating_add(seen[rank_index]);
                seen[rank_index] += 1;
                (rank, stamp, cut.unknown_before)
            })
            .collect();
        if stamps.is_empty() {
            let (rank, first) = match cuts.iter().next() {
                Some((rank, cut)) => (*rank, *cut),
                None => (None, RankCut::default()),
            };
            stamps.push((rank, first.through, first.unknown_before));
        }
        let count = u32::try_from(stamps.len()).unwrap_or(u32::MAX);
        Self {
            ranks,
            cuts: chunks,
            stamps,
            count,
            next: 0,
            timestamp,
            blocks,
        }
    }

    /// [`Self::new`] with one cut for every rank: a state whose ranks all
    /// stand at `through` with `unknown_before` sequences before the record.
    pub fn at(snapshot: Snapshot, through: u64, timestamp: f64, unknown_before: u64) -> Self {
        let cut = RankCut {
            through,
            unknown_before,
        };
        let mut cuts: BTreeMap<Option<i32>, RankCut> = snapshot
            .ranks
            .iter()
            .map(|(rank, _)| (*rank, cut))
            .collect();
        cuts.entry(None).or_insert(cut);
        Self::new(snapshot, &cuts, timestamp)
    }

    /// Chunks in the snapshot (at least one: the clear).
    pub fn chunk_count(&self) -> u32 {
        self.count
    }

    /// Physical copies in the snapshot.
    pub fn blocks(&self) -> u64 {
        self.blocks
    }

    /// The next chunk, or `None` once all `count` have been produced.
    pub(crate) fn next_chunk(&mut self) -> Option<common::KvEventBatch> {
        if self.next >= self.count {
            return None;
        }
        let index = self.next;
        self.next += 1;
        let (dp_rank, sequence_number, unknown_before) = self.stamps[index as usize];
        let mut events = Vec::new();
        if index == 0 {
            events.push(common::KvCacheEvent {
                event_id: 0,
                data: Some(kv_cache_event::Data::Cleared(
                    common::KvCacheCleared::default(),
                )),
            });
        }
        if let Some(&(rank_index, start, end)) = self.cuts.get(index as usize) {
            stored_events(&self.ranks[rank_index].1[start..end], &mut events);
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
                unknown_before,
            }),
            load: None,
        })
    }
}

/// How many of `chunks` each of `ranks` rank indices got.
fn chunks_per_rank(ranks: usize, chunks: &[(usize, usize, usize)]) -> Vec<usize> {
    let mut counts = vec![0usize; ranks];
    for &(rank_index, _, _) in chunks {
        counts[rank_index] += 1;
    }
    counts
}

impl Iterator for SnapshotChunks {
    type Item = common::KvEventBatch;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_chunk()
    }
}

/// The chunks of `ranks` under both bounds: a chunk ends where the next unit
/// would take it past `chunk_blocks` units or `chunk_bytes` of block entries
/// (a unit larger than the byte bound is a chunk of its own), and at the end
/// of its rank.
fn cut_chunks(
    ranks: &[(Option<i32>, Vec<Arc<LiveBlock>>)],
    chunk_blocks: usize,
    chunk_bytes: usize,
) -> Vec<(usize, usize, usize)> {
    let mut cuts = Vec::new();
    for (rank_index, (_, units)) in ranks.iter().enumerate() {
        let (mut start, mut bytes) = (0, 0usize);
        for (at, unit) in units.iter().enumerate() {
            let entry = unit.entry_len();
            if at > start && (at - start >= chunk_blocks || bytes + entry > chunk_bytes) {
                cuts.push((rank_index, start, at));
                start = at;
                bytes = 0;
            }
            bytes += entry;
        }
        if start < units.len() {
            cuts.push((rank_index, start, units.len()));
        }
    }
    cuts
}

/// One rank's entries in emission order, one unit per physical copy: one
/// group-lane entry first when the rank has any, then the main lane in store
/// order with the live ancestors of a block hoisted ahead of it when they
/// were stored later (a parent evicted and stored again under a surviving
/// child), so the gateway's index derives every position from a parent it
/// already holds, then the remaining group lanes in store order. The gateway
/// names a batch's main blocks in a worker's group view only once a group's
/// store has given it one, so the leading entry puts the view in the rank's
/// first chunk ([`SnapshotChunks::new`] puts a rank that has one first) and
/// every main block is named in chain order; the group entries that follow
/// the main lane then land on named blocks, and the leading one waits for
/// its own. Left among the main lane they would hoist a main block ahead of
/// its own parent; all ahead of it, a whole lane would wait.
fn order_rank(entries: Vec<SnapshotEntry>) -> Vec<Arc<LiveBlock>> {
    let (mut entries, mut grouped): (Vec<SnapshotEntry>, Vec<SnapshotEntry>) = entries
        .into_iter()
        .partition(|entry| entry.block.lane == MAIN_LANE);
    entries.sort_unstable_by_key(|entry| entry.block.order);
    grouped.sort_unstable_by_key(|entry| entry.block.order);
    let mut by_hash: HashMap<i64, Vec<usize>> = HashMap::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        by_hash.entry(entry.block.hash).or_default().push(index);
    }
    let units: usize = entries
        .iter()
        .chain(&grouped)
        .map(|entry| entry.copies as usize)
        .sum();
    let mut out = Vec::with_capacity(units);
    let mut grouped = grouped.into_iter();
    let leading = grouped.next();
    if let Some(entry) = &leading {
        out.push(Arc::clone(&entry.block));
    }
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
    if let Some(entry) = &leading {
        for _ in 1..entry.copies {
            out.push(Arc::clone(&entry.block));
        }
    }
    for entry in grouped {
        for _ in 0..entry.copies {
            out.push(Arc::clone(&entry.block));
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
            group_mask: tail.group_mask,
        })),
    }
}

#[cfg(test)]
#[path = "../tests/unit/kv_state/tests.rs"]
mod tests;
