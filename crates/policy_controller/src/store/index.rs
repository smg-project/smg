//! The KV index in the store (design 5.1, the KV index row): one event-fed
//! index a model, each an instance of the `kv-index` crate's chain index.
//! The store writes no index code of its own.
//!
//! One invariant: **an index is written only through a lane, and a lane
//! lives exactly as long as its row.** A lane is one member row's writer:
//! the admission cursor of its rank, the reverse map the chain index asks
//! each writer to keep, and the row's worker id in its model's index. A
//! membership write makes the lane with the row and takes it out with the
//! row, under the workers lock: the ids are handed out in the order of the
//! membership writes, and a batch for a worker the store holds no row for
//! writes nothing. A batch runs through the cursor first; what the cursor
//! admits is applied by the index's own entries, in the batch's order.
//!
//! A lane is a rank's and a snapshot is an engine's. The collector, which
//! alone sees the engine's stream whole, says a snapshot's start to every
//! row of the engine before its first block, the rows the snapshot holds
//! nothing for included: a resync, which empties the lane and starts its
//! cursor over. A chunk only adds.
//!
//! What a lane indexes is what its collector decoded ([`KvEvent`]): which
//! blocks reach an index (their tier, their cache group, the namespace
//! their content is hashed under) is the collector's decision. A row is
//! indexed under its first model name, and only when its signals ask for
//! events.
//!
//! No decision reads an index yet: the change that fills the affinity
//! column from it brings the reader.
//!
//! Locks: the table's, taken under the workers lock by a membership write
//! and alone by a batch, each for as long as it takes to find or to change
//! an entry; then the lane's own, so two rows apply at once. No product
//! path holds one while it asks for the other: a leaving row's blocks go
//! out under its lane's lock alone, and no batch of another row waits
//! behind them. The cases' reader of overlaps takes each lane's lock under
//! the table's, an order no path inverts.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use kv_index::{ApplyError, ChainBlockMap, ChainIndex};
pub use kv_index::{ContentHash, SequenceHash, StoredBlock};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::{
    store::{
        kv_events::{Admission, BatchFacts, BatchOutcome, KvCursor, RankCursor, ResyncReason},
        workers::WorkerRow,
    },
    view::WorkerId,
};

#[cfg(test)]
mod tests;

/// Workers one model's index holds at once: the chain index's own ceiling.
/// A row written past it holds no lane until one of its own later writes
/// finds a place; the store counts the rows that stand so.
pub const INDEX_WORKERS: usize = 1_024;

/// One event of a batch as an index applies it. A collector decodes the
/// engine's message into these.
#[derive(Clone, Debug)]
pub enum KvEvent {
    /// Blocks stored in order after `parent`, the engine's hash of the block
    /// before the first; `None` starts a chain.
    Stored {
        parent: Option<SequenceHash>,
        blocks: Vec<StoredBlock>,
    },
    /// Blocks evicted, by the engine's hashes.
    Removed(Vec<SequenceHash>),
    /// The engine emptied its cache.
    Cleared,
}

/// What a lane counted since it was made, beside its blocks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaneCounts {
    /// Stores whose parent the worker did not hold (evicted, or lost with a
    /// cleared rank): placed as a chain of their own.
    pub parentless_stores: u64,
    /// Blocks the index declined at one of its size caps: the engine keeps
    /// them, the index does not route by them.
    pub refused_blocks: u64,
}

/// One worker's blocks in an index, as a lane writes them.
trait Blocks {
    fn stored(
        &mut self,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
    ) -> Result<(), ApplyError>;
    fn removed(&mut self, hashes: &[SequenceHash]);
    fn cleared(&mut self);
}

/// A lane's place in its model's index.
struct Held {
    index: Arc<ChainIndex>,
    /// The row's id in `index`.
    worker: u32,
    /// The worker's blocks by the engine's hash: the index asks each writer
    /// to keep its own.
    map: ChainBlockMap,
}

impl Blocks for Held {
    fn stored(
        &mut self,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
    ) -> Result<(), ApplyError> {
        self.index
            .apply_stored(self.worker, blocks, parent, &mut self.map)
    }

    fn removed(&mut self, hashes: &[SequenceHash]) {
        self.index.apply_removed(self.worker, hashes, &mut self.map);
    }

    fn cleared(&mut self) {
        self.index.apply_cleared(self.worker, &mut self.map);
    }
}

/// Run one batch through `cursor` and apply what it admits to `blocks`, in
/// the batch's order. The reason comes back when the batch emptied the rank
/// before it was applied.
fn admit<B: Blocks>(
    cursor: &mut RankCursor,
    blocks: &mut B,
    facts: BatchFacts,
    events: &[KvEvent],
    counts: &mut LaneCounts,
) -> (BatchOutcome, Option<ResyncReason>) {
    let mut emptied = None;
    if let Some(chunk) = facts.snapshot {
        // A chunk only adds: the snapshot's start was said to the row, by
        // a resync, before its first block.
        cursor.resync_to(facts.sequence, chunk.unknown_before);
    } else {
        // A clear among the events is the header's fact too, wherever it
        // stands: what a batch that clears leaves owes nothing to what
        // came before it, so the cursor may start over there.
        let clears = facts.clears || events.iter().any(|e| matches!(e, KvEvent::Cleared));
        match cursor.admit(facts.sequence, clears) {
            Admission::Stale => return (BatchOutcome::Skipped, None),
            Admission::Replay { expected } => {
                let gap = BatchOutcome::Gap {
                    expected,
                    received: facts.sequence,
                };
                return (gap, None);
            }
            Admission::Restart => {
                blocks.cleared();
                emptied = Some(ResyncReason::PublisherRestart);
            }
            Admission::Unrecovered { cleared: true, .. } => {
                blocks.cleared();
                emptied = Some(ResyncReason::GapCleared);
            }
            Admission::Apply | Admission::Unrecovered { cleared: false, .. } => {}
        }
    }
    for event in events {
        match event {
            KvEvent::Stored {
                parent,
                blocks: new,
            } => store(blocks, new, *parent, counts),
            KvEvent::Removed(hashes) => blocks.removed(hashes),
            KvEvent::Cleared => blocks.cleared(),
        }
    }
    (BatchOutcome::Applied, emptied)
}

/// One store. A parent the worker does not hold starts a chain of its own,
/// as the engines' first stores after a clear do; what the index declines at
/// a size cap is counted.
fn store<B: Blocks>(
    held: &mut B,
    blocks: &[StoredBlock],
    parent: Option<SequenceHash>,
    counts: &mut LaneCounts,
) {
    let mut placed = held.stored(blocks, parent);
    if matches!(
        placed,
        Err(ApplyError::WorkerNotTracked | ApplyError::ParentBlockNotFound)
    ) && parent.is_some()
    {
        counts.parentless_stores += 1;
        placed = held.stored(blocks, None);
    }
    if let Err(ApplyError::ArenaCapReached { placed }) = placed {
        counts.refused_blocks += blocks.len().saturating_sub(placed) as u64;
    }
}

/// One row's writer.
struct Lane {
    cursor: RankCursor,
    /// A batch ran through the cursor since the lane was made or its worker
    /// was reported gone.
    heard: bool,
    /// `None` once the row left: a batch that found the lane before the
    /// removal writes nothing.
    held: Option<Held>,
    counts: LaneCounts,
}

/// A row's lane as the table keeps it.
struct Entry {
    rank: u16,
    model: Box<str>,
    lane: Arc<Mutex<Lane>>,
}

/// One model's index, and how many lanes write it.
struct ModelIndex {
    index: Arc<ChainIndex>,
    lanes: usize,
}

#[derive(Default)]
struct Table {
    lanes: FxHashMap<WorkerId, Entry>,
    indexes: FxHashMap<Box<str>, ModelIndex>,
    block_sizes: HashMap<String, u32>,
}

/// The resync reasons, counted apart.
const REASONS: usize = 9;

const fn slot(reason: ResyncReason) -> usize {
    match reason {
        ResyncReason::OutOfRange => 0,
        ResyncReason::DataLoss => 1,
        ResyncReason::StreamEnded => 2,
        ResyncReason::SnapshotAbandoned => 3,
        ResyncReason::RelayRestart => 4,
        ResyncReason::EngineReplaced => 5,
        ResyncReason::PublisherRestart => 6,
        ResyncReason::GapCleared => 7,
        ResyncReason::Snapshot => 8,
    }
}

/// The name a row is interned under in its model's index: its id, so a
/// replaced incarnation never meets its predecessor's.
fn name_of(id: WorkerId) -> String {
    format!("{:016x}", id.0)
}

/// The model a row is indexed under: its first, when its signals ask for
/// events.
pub(super) fn indexed_under(row: &WorkerRow) -> Option<&String> {
    let facts = &row.facts;
    facts.models.first().filter(|_| facts.signals.events)
}

/// The store's KV tables: the lanes, the indexes they write and the block
/// sizes of record.
#[derive(Default)]
pub(super) struct KvTables {
    table: Mutex<Table>,
    resyncs: [AtomicU64; REASONS],
}

impl KvTables {
    /// Make, keep or retire `row`'s lane as its facts ask: a row is indexed
    /// under its first model name when its signals ask for events. Called
    /// under the workers lock by every membership write that makes a row or
    /// changes its facts. The row's id in its model's index comes back when
    /// this call interned it: what a replica must repeat to hold the same
    /// ids. A lane made anew holds nothing and is unheard: its cursor reads
    /// `None` until its first batch, which is what tells a collector to
    /// feed it from a snapshot.
    pub(super) fn member(&self, row: &WorkerRow) -> Option<u32> {
        let facts = &row.facts;
        let model = indexed_under(row);
        let stands = self.table.lock().lanes.get(&row.id).map(|entry| {
            entry.rank == facts.dp_rank && model.is_some_and(|model| **model == *entry.model)
        });
        match stands {
            Some(true) => return None,
            Some(false) => self.forget(row.id),
            None => {}
        }
        let model = model?;
        let mut table = self.table.lock();
        let held = table
            .indexes
            .entry(model.as_str().into())
            .or_insert_with(|| ModelIndex {
                index: Arc::new(ChainIndex::with_max_workers(INDEX_WORKERS)),
                lanes: 0,
            });
        // An index at its ceiling: the row holds no lane, and stands in the
        // store's count, until one of its own later writes finds a place.
        let worker = held.index.intern_worker(&name_of(row.id)).ok()?;
        held.lanes += 1;
        let index = Arc::clone(&held.index);
        let lane = Lane {
            cursor: RankCursor::default(),
            heard: false,
            held: Some(Held {
                index,
                worker,
                map: ChainBlockMap::new(),
            }),
            counts: LaneCounts::default(),
        };
        table.lanes.insert(
            row.id,
            Entry {
                rank: facts.dp_rank,
                model: model.as_str().into(),
                lane: Arc::new(Mutex::new(lane)),
            },
        );
        Some(worker)
    }

    /// Take `id`'s lane out, when it has one: its blocks leave the index,
    /// its id there is free for the next row, and a model's index goes with
    /// its last lane. The table's lock is held for the table's own edits
    /// alone: the blocks go out under the lane's lock. Called under the
    /// workers lock, which keeps every other membership write out between
    /// the steps.
    pub(super) fn forget(&self, id: WorkerId) {
        let Some(entry) = self.table.lock().lanes.remove(&id) else {
            return;
        };
        if let Some(Held { index, worker, map }) = entry.lane.lock().held.take() {
            index.remove_worker(worker, map);
        }
        let mut table = self.table.lock();
        let last = table.indexes.get_mut(&entry.model).is_some_and(|model| {
            model.lanes -= 1;
            model.lanes == 0
        });
        if last {
            table.indexes.remove(&entry.model);
        }
    }

    /// The lane of `id`, when the store holds one for `rank` (any rank
    /// under `None`).
    fn lane(&self, id: WorkerId, rank: Option<u16>) -> Option<Arc<Mutex<Lane>>> {
        let table = self.table.lock();
        let entry = table.lanes.get(&id)?;
        rank.is_none_or(|rank| rank == entry.rank)
            .then(|| Arc::clone(&entry.lane))
    }

    pub(super) fn admit(
        &self,
        id: WorkerId,
        rank: u16,
        facts: BatchFacts,
        events: &[KvEvent],
    ) -> BatchOutcome {
        let Some(lane) = self.lane(id, Some(rank)) else {
            return BatchOutcome::Skipped;
        };
        let mut lane = lane.lock();
        let Lane {
            cursor,
            heard,
            held,
            counts,
        } = &mut *lane;
        let Some(held) = held else {
            return BatchOutcome::Skipped;
        };
        *heard = true;
        let (outcome, emptied) = admit(cursor, held, facts, events, counts);
        if let Some(reason) = emptied {
            self.note(reason);
        }
        outcome
    }

    pub(super) fn reconnected(&self, id: WorkerId) {
        if let Some(lane) = self.lane(id, None) {
            lane.lock().cursor.reconnected();
        }
    }

    pub(super) fn resync(&self, id: WorkerId, rank: Option<u16>, reason: ResyncReason) {
        if let Some(lane) = self.lane(id, rank) {
            let mut lane = lane.lock();
            lane.cursor.reset();
            if let Some(held) = &mut lane.held {
                held.cleared();
            }
        }
        self.note(reason);
    }

    pub(super) fn cursor(&self, id: WorkerId, rank: u16) -> Option<KvCursor> {
        let lane = self.lane(id, Some(rank))?;
        let lane = lane.lock();
        lane.heard.then(|| lane.cursor.view())
    }

    /// The worker was reported gone: its blocks leave the index and its
    /// cursor starts over. The lane itself leaves with the row.
    pub(super) fn removed(&self, id: WorkerId) {
        if let Some(lane) = self.lane(id, None) {
            let mut lane = lane.lock();
            lane.cursor.reset();
            lane.heard = false;
            if let Some(held) = &mut lane.held {
                held.cleared();
            }
        }
    }

    pub(super) fn block_size(&self, model: &str) -> Option<u32> {
        self.table.lock().block_sizes.get(model).copied()
    }

    pub(super) fn set_block_size(&self, model: &str, size: u32) {
        let mut table = self.table.lock();
        let of_record = table.block_sizes.entry(model.to_string()).or_insert(size);
        if size > *of_record {
            *of_record = size;
        }
    }

    fn note(&self, reason: ResyncReason) {
        self.resyncs[slot(reason)].fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn resyncs(&self, reason: ResyncReason) -> u64 {
        self.resyncs[slot(reason)].load(Ordering::Relaxed)
    }

    /// How many rows hold a lane.
    pub(super) fn lanes(&self) -> usize {
        self.table.lock().lanes.len()
    }

    /// Blocks the index holds for the row of `id` at `rank`.
    pub(super) fn indexed_blocks(&self, id: WorkerId, rank: u16) -> usize {
        let Some(lane) = self.lane(id, Some(rank)) else {
            return 0;
        };
        let lane = lane.lock();
        lane.held
            .as_ref()
            .map_or(0, |held| held.index.worker_block_count(held.worker))
    }

    pub(super) fn counts(&self, id: WorkerId, rank: u16) -> Option<LaneCounts> {
        Some(self.lane(id, Some(rank))?.lock().counts)
    }

    /// The rows holding a prefix of the request's blocks in `model`'s
    /// index, each with the leading blocks it holds, by id: the read no
    /// decision makes yet, for the cases.
    #[cfg(test)]
    pub(super) fn overlaps(&self, model: &str, hashes: &[u64]) -> Vec<(WorkerId, u32)> {
        let (index, rows) = {
            let table = self.table.lock();
            let Some(ModelIndex { index, .. }) = table.indexes.get(model) else {
                return Vec::new();
            };
            let rows: FxHashMap<u32, WorkerId> = table
                .lanes
                .iter()
                .filter(|(_, entry)| &*entry.model == model)
                .filter_map(|(id, entry)| Some((entry.lane.lock().held.as_ref()?.worker, *id)))
                .collect();
            (Arc::clone(index), rows)
        };
        let mut held = Vec::new();
        index.score_into(
            hashes,
            |hash| *hash,
            false,
            |worker, blocks| {
                if let Some(id) = rows.get(&worker) {
                    held.push((*id, blocks));
                }
            },
        );
        held.sort_unstable();
        held
    }

    /// How many models hold an index.
    #[cfg(test)]
    pub(super) fn indexes(&self) -> usize {
        self.table.lock().indexes.len()
    }

    /// Blocks `model`'s index holds across its workers (a block two hold
    /// counts twice).
    #[cfg(test)]
    pub(super) fn held_blocks(&self, model: &str) -> usize {
        let table = self.table.lock();
        table
            .indexes
            .get(model)
            .map_or(0, |held| held.index.current_size())
    }
}
