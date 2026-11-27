//! The KV index beside the worker objects (design 5.1, the KV index row):
//! one event-fed index a model, each an instance of the `kv-index` crate's
//! chain index. The worker host writes no index code of its own.
//!
//! One invariant: **an index is written only through a lane, and a lane
//! lives exactly as long as its row.** A lane is one member row's writer:
//! the admission cursor of its rank, the reverse map the chain index asks
//! each writer to keep, and the row's worker id in its model's index. The
//! holder of the worker objects makes the lane where it states a row and
//! takes it out where the row leaves, under its own lock: the ids are
//! handed out in the order the holder states its rows, and a batch for a
//! row the holder has not stated writes nothing. A batch runs through the
//! cursor first; what the cursor admits is applied by the index's own
//! entries, in the batch's order.
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
//! An index id is free for the next row once its row leaves. A row holds
//! its id from its lane's making, ahead of its first event: a worker that
//! asks for cache events may never publish one, and the index then holds no
//! block under its id.
//!
//! Locks, in one order: the holder's, then the table's, then a lane's. The
//! table's is taken under the holder's lock where a row is stated or
//! leaves, and alone by a batch, each for as long as it takes to find or to
//! change an entry; then the lane's own, so two rows apply at once. Nothing
//! here calls the holder, and what runs in a lane's turn reads neither the
//! holder nor the adapter's list of rows: a holder that waits for a lane's
//! turn waits for one batch and no more. No product path holds the table's
//! lock while it asks for a lane's: a leaving row's blocks go out under its
//! lane's lock alone, and no batch of another row waits behind them. The
//! cases' reader of overlaps takes each lane's lock under the table's, an
//! order no path inverts.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

pub(super) use kv_index::{ApplyError, ChainBlockMap, SequenceHash, StoredBlock, WorkerGroups};
use kv_index::{ChainIndex, GroupViews};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use smg_worker::kv_events::port::{BatchFacts, BatchOutcome, KvCursor, ResyncReason};

use super::kv_cursor::{BatchAdmission, RankCursor};
use crate::{engine::view::WorkerId, store::workers::WorkerRow};

#[cfg(test)]
#[path = "../../tests/unit/worker_host/kv_tables/tests.rs"]
mod tests;

/// Workers one model's index holds at once: the chain index's own ceiling.
/// A row stated past it holds no lane until one of its own later
/// statements finds a place.
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
        // what a reader can see is emptied in the lane's own turn.
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
/// ([`KvTables::admit_through`]): the chain's calls under the row's id in
/// its model's index, the row's view in the model's group views, and the
/// row's reverse map.
///
/// The map is the lane's between batches, since the tables empty a rank
/// by it. A body whose apply rules hold the map as their own state takes
/// it for the batch and puts it back before it returns. That call stays
/// inside the crate: the tables empty a rank by the map with no caller's
/// help, and that rule is no host's contract. The writes stay inside the
/// crate with it, since each writes through that map: a write through any
/// other would leave blocks that no emptying finds.
///
/// The body runs under the lane's lock. It calls nothing of the tables for
/// its own row. A body that unwinds leaves its rank half written, so the
/// lane starts over before the unwind goes on.
pub struct LaneWriter<'a> {
    held: &'a mut Held,
}

impl LaneWriter<'_> {
    /// The row's id in its model's index.
    pub fn worker(&self) -> u32 {
        self.held.worker
    }

    /// A value that moves each time the tables empty the rank, and where
    /// the lane is made: a resync, a restart or a cleared gap its cursor
    /// read, the worker reported gone, a lent body that unwound. The
    /// emptying this batch's admission made has moved it. It never repeats
    /// under a row, across the lanes the holder makes for it. A body that
    /// keeps state beside the map remembers the value it read, and resets
    /// that state where the next batch reads another.
    pub fn emptied(&self) -> u64 {
        self.held.emptied
    }

    /// The row's blocks by the engine's hash: the reverse map the chain
    /// asks a writer to keep. The cache-event adapter alone takes it.
    pub(crate) fn map(&mut self) -> &mut ChainBlockMap {
        &mut self.held.map
    }

    /// Store `blocks` after `parent`, or from the root when it is absent,
    /// with `map` the row's reverse map.
    pub(crate) fn stored(
        &self,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
        map: &mut ChainBlockMap,
    ) -> Result<(), ApplyError> {
        self.held
            .index
            .chain
            .apply_stored(self.held.worker, blocks, parent, map)
    }

    /// Forget the named blocks of the row.
    pub(crate) fn removed(&self, hashes: &[SequenceHash], map: &mut ChainBlockMap) {
        self.held
            .index
            .chain
            .apply_removed(self.held.worker, hashes, map);
    }

    /// Forget every block of the row.
    pub(crate) fn cleared(&self, map: &mut ChainBlockMap) {
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
    /// Lend the lane's place to `body`, once.
    fn lend(mut self, body: &mut dyn FnMut(&mut LaneWriter<'_>)) {
        if let Some(held) = &mut self.lane.held {
            body(&mut LaneWriter { held });
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
    /// was reported gone: what a subscriber's read of the cursor answers by
    /// ([`KvTables::cursor`]). The row's id in its index does not wait for
    /// it.
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
            held.cleared();
        }
    }
}

/// A row's lane as the table keeps it.
struct Entry {
    rank: u16,
    model: Box<str>,
    /// The index the lane writes and the row's id in it, as the lane holds
    /// them ([`Held`]): set once at the lane's making and never changed, so
    /// a reader of the id asks the table alone and waits for no batch. The
    /// entry leaves the table before the lane lets them go.
    #[cfg(feature = "controller")]
    index: Arc<Index>,
    #[cfg(feature = "controller")]
    worker: u32,
    lane: Arc<Mutex<Lane>>,
}

/// One model's index: the chain index and each row's view of its cache
/// groups.
struct Index {
    chain: ChainIndex,
    /// Each row's view of its cache groups other than main attention,
    /// under the row's id: written by a lent body ([`LaneWriter::groups`]).
    groups: GroupViews,
}

impl Index {
    fn new() -> Self {
        Self {
            chain: ChainIndex::with_max_workers(INDEX_WORKERS),
            groups: GroupViews::new(),
        }
    }

    /// A row takes `id`, which the chain just handed out. Said under the
    /// table's lock, before the row's lane exists.
    fn take(&self, id: u32) {
        // The row does not find the view of the row before it, whatever
        // path that row left by.
        self.groups.remove(id);
    }
}

/// One model's index, lent to the engine over worker lists through the
/// worker host's reader ([`super::kv_read`]): the chain the lanes write and
/// its group views, with no copy of a block.
///
/// A row's id here is its lane's place: held from the lane's making, ahead
/// of its first event, and free for the next row once its row leaves. The
/// chain counts each freed id it hands to the next row, and the lending
/// answers that count ([`Self::ids_reused`]). A reader that maps ids to
/// rows once a table reads the count ahead of the ids and again behind a
/// lookup: once it moved, an id of the table may be another row's.
#[cfg(feature = "controller")]
pub(super) struct LentIndex {
    tables: Arc<KvTables>,
    index: Arc<Index>,
}

#[cfg(feature = "controller")]
impl LentIndex {
    /// Whether no row holds a block in the index.
    pub(super) fn is_empty(&self) -> bool {
        self.index.chain.is_empty()
    }

    /// The id of `row` in this index: its lane's place, whether or not a
    /// batch ran through the lane. `None` for a row with no lane, and for
    /// one whose lane writes another index: another model's, or this
    /// model's made again behind its last lane. Read from the table's entry
    /// under the table's lock alone: no lane's lock is taken, so the read
    /// waits for no batch.
    pub(super) fn number_of(&self, row: WorkerId) -> Option<u32> {
        let table = self.tables.table.lock();
        let entry = table.lanes.get(&row)?;
        Arc::ptr_eq(&entry.index, &self.index).then_some(entry.worker)
    }

    /// How many times the index handed an id one row freed to the next row
    /// that asked for one: the chain's own count, which a removal alone
    /// does not move. A row takes its id before its lane is made
    /// ([`KvTables::member`]), so the count moves ahead of the row's first
    /// block. One load, and no lock.
    pub(super) fn ids_reused(&self) -> u64 {
        self.index.chain.ids_reused()
    }

    /// How many blocks the index holds under `id`.
    pub(super) fn blocks(&self, id: u32) -> usize {
        self.index.chain.worker_block_count(id)
    }

    /// The chain index and the group views beside it, for the engine's one
    /// lookup.
    pub(super) fn chain(&self) -> (&ChainIndex, &GroupViews) {
        (&self.index.chain, &self.index.groups)
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

/// The worker host's KV tables: the lanes, the indexes they write and the
/// block sizes of record.
///
/// A host builds one ([`KvTables::default`]) and hands it to the holder of
/// its worker objects
/// ([`Objects::with_kv_tables`](super::Objects::with_kv_tables)). The holder
/// alone makes and retires lanes, and the host's cache-event adapter alone
/// writes blocks, cursors and block sizes through them: nothing outside the
/// crate writes. The public face is four readers: a rank's cursor, a row's
/// indexed blocks, the resyncs counted under a reason and a model's block
/// size of record. Each answers a plain value behind one short hold of a
/// lock of the tables' own (the table's, then one lane's where it reads a
/// lane), so no caller holds a lock of the tables.
#[derive(Default)]
pub struct KvTables {
    table: Mutex<Table>,
    resyncs: [AtomicU64; REASONS],
}

impl KvTables {
    /// Make, keep or retire `row`'s lane as its facts ask: a row is indexed
    /// under its first model name when its signals ask for events. Called
    /// by the holder of the worker objects, under its lock, wherever it
    /// states a row's facts. The row's id in its model's index comes back when
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
        // An index at its ceiling: the row holds no lane until one of its
        // own later statements finds a place.
        let worker = held.index.chain.intern_worker(&name_of(row.id)).ok()?;
        held.index.take(worker);
        held.lanes += 1;
        let index = Arc::clone(&held.index);
        #[cfg(feature = "controller")]
        let lent = Arc::clone(&index);
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
                #[cfg(feature = "controller")]
                index: lent,
                #[cfg(feature = "controller")]
                worker,
                lane: Arc::new(Mutex::new(lane)),
            },
        );
        Some(worker)
    }

    /// Take `id`'s lane out, when it has one: its blocks leave the index,
    /// its id there is free for the next row, and a model's index goes with
    /// its last lane. The table's lock is held for the table's own edits
    /// alone: the blocks go out under the lane's lock. Called under the
    /// holder's lock, which keeps every other statement of a row out
    /// between the steps.
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

    /// The lane of `id`, when the tables hold one for `rank` (any rank
    /// under `None`).
    fn lane(&self, id: WorkerId, rank: Option<u16>) -> Option<Arc<Mutex<Lane>>> {
        let table = self.table.lock();
        let entry = table.lanes.get(&id)?;
        rank.is_none_or(|rank| rank == entry.rank)
            .then(|| Arc::clone(&entry.lane))
    }

    /// The index of `model`, lent to the engine over worker lists
    /// ([`LentIndex`]). `None` where the tables hold no index of the model:
    /// one is made with a model's first lane and goes with its last.
    #[cfg(feature = "controller")]
    pub(super) fn lend(self: &Arc<Self>, model: &str) -> Option<LentIndex> {
        let table = self.table.lock();
        let index = Arc::clone(&table.indexes.get(model)?.index);
        Some(LentIndex {
            tables: Arc::clone(self),
            index,
        })
    }

    /// Run one batch through its row's lane, for its host's own rules to
    /// apply: the cursor reads the header alone, and an admitted batch
    /// lends the lane to `body`, once, inside the lane's turn. A body that
    /// unwinds has the lane start over ([`Lent`]).
    ///
    /// A batch the cursor skips or answers with a gap runs no body. The
    /// cursor reads `facts` alone, since it holds no batch to look into: a
    /// batch that carries the engine's own clear says so in
    /// [`BatchFacts::clears`]. The body runs under the lane's lock and
    /// calls nothing of the tables for its own row.
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

    /// A new stream connection was made for the worker: the first batch on
    /// it shows where the publisher stands.
    pub(super) fn reconnected(&self, id: WorkerId) {
        if let Some(lane) = self.lane(id, None) {
            lane.lock().cursor.reconnected();
        }
    }

    /// Starts the lane's cursor over for `reason` and empties the lane. A
    /// snapshot its stream did not finish is the one reason that keeps the
    /// blocks: the stream cannot be trusted, so the next subscription asks
    /// from zero, and what the chunks showed stays until the next
    /// snapshot's start empties the lane. A `rank` that is not the row's
    /// resets nothing; `None` asks no check.
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

    /// The cursor of the row `id` at `rank`, once the rank has been heard
    /// from; `None` for a row that holds no lane at that rank.
    pub fn cursor(&self, id: WorkerId, rank: u16) -> Option<KvCursor> {
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

    /// The block size of record for `model`, in tokens: the size a request
    /// of the model is hashed by. A registration names it, provisionally
    /// and only upward. A stored block confirms it: from then it is the
    /// engines' own size, the widest block the model's indexed stores
    /// named, which the next stored block puts back where a registration
    /// raised it. It is dropped with the model's last events subscription.
    /// `None` where none was named.
    pub fn block_size(&self, model: &str) -> Option<u32> {
        self.table.lock().block_sizes.get(model).copied()
    }

    /// Name a model's block size: provisional until a stored block confirms
    /// it, and only ever raised.
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

    /// How many resyncs were recorded for `reason`.
    pub fn resyncs(&self, reason: ResyncReason) -> u64 {
        self.resyncs[slot(reason)].load(Ordering::Relaxed)
    }

    /// Blocks the index holds for the row of `id` at `rank`; `0` for a row
    /// that holds no lane there.
    pub fn indexed_blocks(&self, id: WorkerId, rank: u16) -> usize {
        let Some(lane) = self.lane(id, Some(rank)) else {
            return 0;
        };
        let lane = lane.lock();
        lane.held
            .as_ref()
            .map_or(0, |held| held.index.chain.worker_block_count(held.worker))
    }
}
