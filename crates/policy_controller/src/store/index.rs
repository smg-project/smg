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
//! cursor over. A chunk only adds. A snapshot its stream did not finish
//! starts the cursor over and keeps what its chunks showed.
//!
//! What a lane indexes is what the body of an admitted batch writes: which
//! blocks reach an index (their tier, their cache group, the namespace
//! their content is hashed under) is the collector's decision. A row is
//! indexed under its first model name, and only when its signals ask for
//! events.
//!
//! An index id is free for the next row once its row leaves. A lane says by
//! one bit of its model's index (`Index`) that it applied an event, since a
//! worker that asks for cache events may never publish one.
//!
//! Locks: the table's, taken under the workers lock by a membership write
//! and alone by a batch, each for as long as it takes to find or to change
//! an entry; then the lane's own, so two rows apply at once. No product
//! path holds one while it asks for the other: a leaving row's blocks go
//! out under its lane's lock alone, and no batch of another row waits
//! behind them. The cases' reader of overlaps takes each lane's lock under
//! the table's, an order no path inverts.

use std::{
    cell::Cell,
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

pub use kv_index::{
    ApplyError, ChainBlockMap, ContentHash, SequenceHash, StoredBlock, WorkerGroups,
};
use kv_index::{ChainIndex, GroupViews};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::{
    engine::view::WorkerId,
    store::{
        kv_events::{BatchAdmission, BatchFacts, BatchOutcome, KvCursor, RankCursor, ResyncReason},
        workers::WorkerRow,
    },
};

#[cfg(test)]
mod tests;

/// Workers one model's index holds at once: the chain index's own ceiling.
/// A row written past it holds no lane until one of its own later writes
/// finds a place; the store counts the rows that stand so.
pub const INDEX_WORKERS: usize = 1_024;

/// One worker's blocks in an index, as a lane's cursor empties them.
trait Blocks {
    fn cleared(&mut self);
}

/// A lane's place in its model's index.
struct Held {
    index: Arc<Index>,
    /// The row's id in `index`.
    worker: u32,
    /// The worker's blocks by the engine's hash: the index asks each writer
    /// to keep its own.
    map: ChainBlockMap,
    /// The value the place took where it was made, and takes at each
    /// emptying ([`LaneWriter::emptied`]).
    emptied: u64,
}

impl Blocks for Held {
    fn cleared(&mut self) {
        self.index.chain.apply_cleared(self.worker, &mut self.map);
        // The row's view of its other cache groups goes with its blocks:
        // what a reader can see is emptied in the store's own turn.
        if let Some(view) = self.index.groups.get(self.worker) {
            view.clear();
        }
        // The one routine that empties a rank says so, whoever asked: a
        // resync, the cursor, the worker reported gone, a lent body that
        // unwound.
        self.emptied = emptying();
    }
}

/// The count a lane's makings and emptyings are drawn from. It is one for
/// the process and only rises, so no two of them read alike: a lane made
/// again under a row's id reads another value than the lane before it,
/// whatever each one counted. A reader tests inequality alone.
static EMPTYINGS: AtomicU64 = AtomicU64::new(0);

/// The next value of [`EMPTYINGS`].
fn emptying() -> u64 {
    EMPTYINGS.fetch_add(1, Ordering::Relaxed) + 1
}

/// A row's lane, lent to the body of one admitted batch
/// ([`KvEventSink::admit_through`](super::KvEventSink::admit_through)): the
/// chain's calls under the row's id in its model's index, the row's view
/// in the model's group views, and the row's reverse map.
///
/// The map is the lane's between batches, since the store empties a rank
/// by it. A body whose apply rules hold the map as their own state takes
/// it for the batch and puts it back before it returns. That call stays
/// inside the crate: the store empties a rank by the map with no caller's
/// help, and that rule is no host's contract. The writes stay inside the
/// crate with it, since each writes through that map: a write through any
/// other would leave blocks that no emptying finds.
///
/// The body runs under the lane's lock. It calls nothing of the sink for
/// its own row. A body that unwinds leaves its rank half written, so the
/// store has the lane start over before the unwind goes on.
pub struct LaneWriter<'a> {
    held: &'a mut Held,
    /// Whether the body wrote: the lane says its row heard behind a body
    /// that did.
    wrote: Cell<bool>,
}

impl LaneWriter<'_> {
    /// The row's id in its model's index.
    pub fn worker(&self) -> u32 {
        self.held.worker
    }

    /// A value that moves each time the store empties the rank, and where
    /// the lane is made: a resync, a restart or a cleared gap its cursor
    /// read, the worker reported gone, a lent body that
    /// unwound. The emptying this batch's admission made has moved it. It
    /// never repeats under a row,
    /// across the lanes the store makes for it. A body that keeps state
    /// beside the map remembers the value it read, and resets that state
    /// where the next batch reads another.
    pub fn emptied(&self) -> u64 {
        self.held.emptied
    }

    /// The row's blocks by the engine's hash: the reverse map the chain
    /// asks a writer to keep. The crate's worker host alone takes it.
    #[cfg(any(test, feature = "worker-host"))]
    pub(crate) fn map(&mut self) -> &mut ChainBlockMap {
        &mut self.held.map
    }

    /// Store `blocks` after `parent`, or from the root when it is absent,
    /// with `map` the row's reverse map.
    #[cfg(any(test, feature = "worker-host"))]
    pub(crate) fn stored(
        &self,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
        map: &mut ChainBlockMap,
    ) -> Result<(), ApplyError> {
        self.wrote.set(true);
        self.held
            .index
            .chain
            .apply_stored(self.held.worker, blocks, parent, map)
    }

    /// Forget the named blocks of the row.
    #[cfg(any(test, feature = "worker-host"))]
    pub(crate) fn removed(&self, hashes: &[SequenceHash], map: &mut ChainBlockMap) {
        self.wrote.set(true);
        self.held
            .index
            .chain
            .apply_removed(self.held.worker, hashes, map);
    }

    /// Forget every block of the row.
    #[cfg(any(test, feature = "worker-host"))]
    pub(crate) fn cleared(&self, map: &mut ChainBlockMap) {
        self.wrote.set(true);
        self.held.index.chain.apply_cleared(self.held.worker, map);
    }

    /// The row's view of its cache groups other than main attention, in
    /// the model's group views, created at the first call.
    pub fn groups(&self) -> Arc<WorkerGroups> {
        self.held.index.groups.get_or_create(self.held.worker)
    }
}

/// A lane lent to a body. A body that unwinds leaves the rank half
/// written, and the state its host keeps beside the map half applied. So,
/// dropped behind such a body, this has the lane start over
/// ([`Lane::start_over`]), which empties the rank through the one routine
/// that does: the next body reads another value of
/// [`LaneWriter::emptied`] and resets what it keeps.
struct Lent<'a> {
    lane: &'a mut Lane,
    /// The body returned.
    whole: bool,
}

impl Lent<'_> {
    /// Lend the lane's place to `body`, once, and say the row heard where
    /// the body wrote.
    fn lend(mut self, body: &mut dyn FnMut(&mut LaneWriter<'_>)) {
        if let Some(held) = &mut self.lane.held {
            let mut writer = LaneWriter {
                held: &mut *held,
                wrote: Cell::new(false),
            };
            body(&mut writer);
            let wrote = writer.wrote.get();
            hear_written(held, wrote);
        }
        self.whole = true;
    }
}

impl Drop for Lent<'_> {
    fn drop(&mut self) {
        if !self.whole {
            self.lane.start_over();
        }
    }
}

/// The index speaks for a row from the first admitted batch whose lent
/// body wrote. A batch the cursor refuses writes no block.
fn hear_written(held: &Held, wrote: bool) {
    if wrote && !held.index.heard(held.worker) {
        held.index.hear(held.worker, true);
    }
}

/// Run one batch's header through `cursor`. `Ok` where the batch is
/// admitted, with the reason when the cursor emptied the rank ahead of it.
/// `Err` with the batch's outcome where it is not.
fn admitted<B: Blocks>(
    cursor: &mut RankCursor,
    blocks: &mut B,
    facts: BatchFacts,
) -> Result<Option<ResyncReason>, BatchOutcome> {
    if let Some(chunk) = facts.snapshot {
        // A chunk only adds: the snapshot's start was said to the row, by
        // a resync, before its first block.
        cursor.resync_to(facts.sequence, chunk.unknown_before);
        return Ok(None);
    }
    match cursor.admit(facts.sequence, facts.clears) {
        BatchAdmission::Stale => Err(BatchOutcome::Skipped),
        BatchAdmission::Replay { expected } => Err(BatchOutcome::Gap {
            expected,
            received: facts.sequence,
        }),
        BatchAdmission::Restart => {
            blocks.cleared();
            Ok(Some(ResyncReason::PublisherRestart))
        }
        BatchAdmission::Unrecovered { cleared: true, .. } => {
            blocks.cleared();
            Ok(Some(ResyncReason::GapCleared))
        }
        BatchAdmission::Apply | BatchAdmission::Unrecovered { cleared: false, .. } => Ok(None),
    }
}

/// One row's writer.
struct Lane {
    cursor: RankCursor,
    /// A batch ran through the cursor since the lane was made or its worker
    /// was reported gone. The index keeps a bit beside it
    /// ([`Index::hear`]), set at the first such batch whose body writes,
    /// and cleared with this word.
    heard: bool,
    /// `None` once the row left: a batch that found the lane before the
    /// removal writes nothing.
    held: Option<Held>,
}

impl Lane {
    /// The lane starts over: its cursor reads nothing, its row is unheard
    /// and its rank is empty. Where its worker was reported gone, and
    /// where a lent body unwound.
    fn start_over(&mut self) {
        self.cursor.reset();
        self.heard = false;
        if let Some(held) = &mut self.held {
            held.index.hear(held.worker, false);
            held.cleared();
        }
    }
}

/// A row's lane as the table keeps it.
struct Entry {
    rank: u16,
    model: Box<str>,
    lane: Arc<Mutex<Lane>>,
}

/// The words of a read's mask at most: a bit an index id.
const MASK_WORDS: usize = INDEX_WORKERS / 64;

/// One model's index: the chain index, each row's view of its cache groups,
/// and whether each id's lane was heard.
struct Index {
    chain: ChainIndex,
    /// Each row's view of its cache groups other than main attention,
    /// under the row's id: written by a lent body ([`LaneWriter::groups`]).
    groups: GroupViews,
    /// A bit an index id: the body of an admitted batch wrote through the
    /// lane of the row that holds the id. A body that writes nothing sets
    /// nothing, and neither does a batch the lane's cursor refuses:
    /// the index holds nothing of the row yet. The lane's own word says
    /// less, that any batch ran through it
    /// ([`Lane::heard`]). The lane writes its own bit under its lock.
    /// The bit is a hint of the instant it was read at and orders nothing.
    heard: [AtomicU64; MASK_WORDS],
}

impl Index {
    fn new() -> Self {
        Self {
            chain: ChainIndex::with_max_workers(INDEX_WORKERS),
            groups: GroupViews::new(),
            heard: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// The lane that holds `id` says whether an admitted batch's body
    /// wrote through it.
    fn hear(&self, id: u32, heard: bool) {
        let Some(word) = self.heard.get((id / 64) as usize) else {
            return;
        };
        let bit = 1u64 << (id % 64);
        if heard {
            word.fetch_or(bit, Ordering::Relaxed);
        } else {
            word.fetch_and(!bit, Ordering::Relaxed);
        }
    }

    /// Whether an admitted batch's body wrote through the lane
    /// that holds `id`.
    fn heard(&self, id: u32) -> bool {
        let word = self.heard.get((id / 64) as usize);
        word.is_some_and(|word| word.load(Ordering::Relaxed) & (1 << (id % 64)) != 0)
    }

    /// A row takes `id`, which the chain just handed out. Said under the
    /// table's lock, before the row's lane exists.
    fn take(&self, id: u32) {
        // The row that held the id before may have been heard: the row
        // that takes it is not.
        self.hear(id, false);
        // Nor does it find the view of the row before it, whatever path
        // that row left by.
        self.groups.remove(id);
    }
}

/// One model's index, and how many lanes write it.
struct ModelIndex {
    index: Arc<Index>,
    lanes: usize,
}

#[derive(Default)]
struct Table {
    lanes: FxHashMap<WorkerId, Entry>,
    indexes: FxHashMap<Box<str>, ModelIndex>,
    block_sizes: HashMap<String, u32>,
}

/// The resync reasons, counted apart.
const REASONS: usize = 6;

const fn slot(reason: ResyncReason) -> usize {
    match reason {
        ResyncReason::OutOfRange => 0,
        ResyncReason::DataLoss => 1,
        ResyncReason::SnapshotAbandoned => 2,
        ResyncReason::PublisherRestart => 3,
        ResyncReason::GapCleared => 4,
        ResyncReason::Snapshot => 5,
    }
}

/// The name a row is interned under in its model's index: its id, so a
/// replaced incarnation never meets its predecessor's.
fn name_of(id: WorkerId) -> String {
    format!("{:016x}", id.0)
}

/// The model a row is indexed under: its first, when its signals ask for
/// events.
pub(crate) fn indexed_under(row: &WorkerRow) -> Option<&String> {
    let facts = &row.facts;
    facts.models.first().filter(|_| facts.signals.events)
}

/// The store's KV tables: the lanes, the indexes they write and the block
/// sizes of record.
#[derive(Default)]
pub(crate) struct KvTables {
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
                index: Arc::new(Index::new()),
                lanes: 0,
            });
        // An index at its ceiling: the row holds no lane, and stands in the
        // store's count, until one of its own later writes finds a place.
        let worker = held.index.chain.intern_worker(&name_of(row.id)).ok()?;
        held.index.take(worker);
        held.lanes += 1;
        let index = Arc::clone(&held.index);
        let lane = Lane {
            cursor: RankCursor::default(),
            heard: false,
            held: Some(Held {
                index,
                worker,
                map: ChainBlockMap::new(),
                emptied: emptying(),
            }),
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
        if let Some(Held {
            index, worker, map, ..
        }) = entry.lane.lock().held.take()
        {
            index.chain.remove_worker(worker, map);
            // The index hands the id to a later row: the view goes too.
            index.groups.remove(worker);
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

    /// Run one batch through its row's lane, for its host's own rules to
    /// apply: the cursor reads the header alone, and an admitted batch
    /// lends the lane to `body`, once, inside the lane's turn. The lane
    /// says its row heard behind a body that wrote. A body that unwinds
    /// has the lane start over ([`Lent`]).
    pub(super) fn admit_through(
        &self,
        id: WorkerId,
        rank: u16,
        facts: BatchFacts,
        body: &mut dyn FnMut(&mut LaneWriter<'_>),
    ) -> BatchOutcome {
        let Some(lane) = self.lane(id, Some(rank)) else {
            return BatchOutcome::Skipped;
        };
        let mut lane = lane.lock();
        let Lane {
            cursor,
            heard,
            held,
            ..
        } = &mut *lane;
        let Some(held) = held else {
            return BatchOutcome::Skipped;
        };
        *heard = true;
        let emptied = match admitted(cursor, held, facts) {
            Ok(emptied) => emptied,
            Err(outcome) => return outcome,
        };
        if let Some(reason) = emptied {
            self.note(reason);
        }
        Lent {
            lane: &mut lane,
            whole: false,
        }
        .lend(body);
        BatchOutcome::Applied
    }

    pub(super) fn reconnected(&self, id: WorkerId) {
        if let Some(lane) = self.lane(id, None) {
            lane.lock().cursor.reconnected();
        }
    }

    /// Starts the lane's cursor over for `reason` and empties the lane. A
    /// snapshot its stream did not finish is the one reason that keeps the
    /// blocks: the stream cannot be trusted, so the next subscription asks
    /// from zero, and what the chunks showed stays until the next
    /// snapshot's start empties the lane.
    pub(super) fn resync(&self, id: WorkerId, rank: Option<u16>, reason: ResyncReason) {
        if let Some(lane) = self.lane(id, rank) {
            let mut lane = lane.lock();
            lane.cursor.reset();
            if reason != ResyncReason::SnapshotAbandoned {
                if let Some(held) = &mut lane.held {
                    held.cleared();
                }
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
            lane.lock().start_over();
        }
    }

    pub(super) fn block_size(&self, model: &str) -> Option<u32> {
        self.table.lock().block_sizes.get(model).copied()
    }

    pub(super) fn set_block_size(&self, model: &str, size: u32) {
        let mut table = self.table.lock();
        let of_record = table.block_sizes.get(model).copied();
        if of_record.is_none_or(|of_record| size > of_record) {
            table.block_sizes.insert(model.to_string(), size);
        }
    }

    /// Drop the size of record so a returning model can set it anew.
    pub(super) fn forget_block_size(&self, model: &str) {
        self.table.lock().block_sizes.remove(model);
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
            .map_or(0, |held| held.index.chain.worker_block_count(held.worker))
    }

    /// Whether the row of `id` holds a lane at `rank`, for the cases.
    #[cfg(test)]
    pub(crate) fn holds_lane(&self, id: WorkerId, rank: u16) -> bool {
        self.lane(id, Some(rank)).is_some()
    }

    /// The rows holding a prefix of the request's blocks in `model`'s
    /// index, each with the leading blocks it holds, by id: every holder
    /// listed, for the cases.
    #[cfg(test)]
    pub(crate) fn overlaps(&self, model: &str, hashes: &[u64]) -> Vec<(WorkerId, u32)> {
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
        index.chain.score_into(
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
            .map_or(0, |held| held.index.chain.current_size())
    }
}
