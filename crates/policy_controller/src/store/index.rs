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
//! What a lane indexes is what its collector decoded ([`KvEvent`]): which
//! blocks reach an index (their tier, their cache group, the namespace
//! their content is hashed under) is the collector's decision. A row is
//! indexed under its first model name, and only when its signals ask for
//! events.
//!
//! The store publishes each pool's standing in its model's index
//! (`PoolIndex`) with the pool's rows: where each index id stands in the
//! pool, and the pool's warming rows as a write last read them. An index
//! id is free for the next row once its row leaves: the index counts its
//! takes and keeps the count each id was last taken at (`Index`).
//!
//! A pool whose profile reads sent prefixes holds its model's record of
//! sends. The record speaks for a row until the row's lane is heard: a row
//! with no lane, and a row whose lane applied no event yet, since a
//! worker that asks for cache events may never publish one. A lane says so
//! by one bit of its model's index (`Index`): no publication follows its
//! first applied event. Issuing a lease writes the blocks sent to its
//! first target while the record speaks for it: token blocks, or windows
//! of a request's text, which the index of cache events is never asked
//! for. Once a lane is heard the index alone speaks for its row, and what
//! the record holds of the row ages out under the record's bound.
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
        atomic::{fence, AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Weak,
    },
};

pub use kv_index::{
    ApplyError, ChainBlockMap, ContentHash, SequenceHash, StoredBlock, WorkerGroups,
};
use kv_index::{ChainIndex, GroupViews};
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashMap;

use crate::{
    engine::{
        view::{Candidate, Role, WorkerId},
        warm_up::{WarmUpRule, WarmUpState},
    },
    store::{
        kv_events::{BatchAdmission, BatchFacts, BatchOutcome, KvCursor, RankCursor, ResyncReason},
        workers::WorkerRow,
    },
    tracking::prefix_predictor::PrefixPredictor,
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

/// What a lane counted since it was made, beside its blocks, over its
/// typed batches. A batch lent to a body
/// ([`KvEventSink::admit_through`](super::KvEventSink::admit_through))
/// moves none of them: the body's rules count for themselves.
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
    fn stored(
        &mut self,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
    ) -> Result<(), ApplyError> {
        // Before the batch's blocks reach the chain: a decision that reads
        // one of them then counts this row's take of its id ([`Index`]).
        fence(Ordering::Release);
        self.index
            .chain
            .apply_stored(self.worker, blocks, parent, &mut self.map)
    }

    fn removed(&mut self, hashes: &[SequenceHash]) {
        self.index
            .chain
            .apply_removed(self.worker, hashes, &mut self.map);
    }

    fn cleared(&mut self) {
        self.index.chain.apply_cleared(self.worker, &mut self.map);
        // The row's view of its other cache groups goes with its blocks:
        // what a reader can see is emptied in the store's own turn.
        if let Some(view) = self.index.groups.get(self.worker) {
            view.clear();
        }
        // The one routine that empties a rank says so, whoever asked: a
        // resync, the cursor, the worker reported gone, a typed clear.
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
    /// read, the worker reported gone, a typed clear, a lent body that
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
        // Before the batch's blocks reach the chain, as in the lane's own
        // write ([`Held`]).
        fence(Ordering::Release);
        self.held
            .index
            .chain
            .apply_stored(self.held.worker, blocks, parent, map)
    }

    /// Forget the named blocks of the row.
    #[cfg(feature = "worker-host")]
    pub(crate) fn removed(&self, hashes: &[SequenceHash], map: &mut ChainBlockMap) {
        self.wrote.set(true);
        self.held
            .index
            .chain
            .apply_removed(self.held.worker, hashes, map);
    }

    /// Forget every block of the row.
    #[cfg(feature = "worker-host")]
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

/// The index speaks for a row from the first admitted batch that wrote:
/// one that carried an event, or one whose lent body wrote. A batch the
/// cursor refuses writes no block, and the record of sent prefixes keeps
/// its cover until one is applied.
fn hear_written(held: &Held, wrote: bool) {
    if wrote && !held.index.heard(held.worker) {
        held.index.hear(held.worker, true);
    }
}

/// Run one batch's header through `cursor`, with `clears` the fact that
/// the batch carries the engine's own clear. `Ok` where the batch is
/// admitted, with the reason when the cursor emptied the rank ahead of it.
/// `Err` with the batch's outcome where it is not.
fn admitted<B: Blocks>(
    cursor: &mut RankCursor,
    blocks: &mut B,
    facts: BatchFacts,
    clears: bool,
) -> Result<Option<ResyncReason>, BatchOutcome> {
    if let Some(chunk) = facts.snapshot {
        // A chunk only adds: the snapshot's start was said to the row, by
        // a resync, before its first block.
        cursor.resync_to(facts.sequence, chunk.unknown_before);
        return Ok(None);
    }
    match cursor.admit(facts.sequence, clears) {
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
    // A clear among the events is the header's fact too, wherever it
    // stands: what a batch that clears leaves owes nothing to what came
    // before it, so the cursor may start over there.
    let clears = facts.clears || events.iter().any(|e| matches!(e, KvEvent::Cleared));
    let emptied = match admitted(cursor, blocks, facts, clears) {
        Ok(emptied) => emptied,
        Err(outcome) => return (outcome, None),
    };
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
    /// was reported gone. The index keeps a bit beside it
    /// ([`Index::hear`]), set at the first such batch that carries an event
    /// and is applied, and cleared with this word, where a decision reads
    /// it with no lock.
    heard: bool,
    /// `None` once the row left: a batch that found the lane before the
    /// removal writes nothing.
    held: Option<Held>,
    counts: LaneCounts,
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
    /// The row's id in its model's index.
    worker: u32,
    lane: Arc<Mutex<Lane>>,
}

/// No row: an index id no row of the pool holds, a row its model's index
/// does not hold.
const NONE: u32 = u32::MAX;

/// The words of a read's mask at most: a bit an index id.
const MASK_WORDS: usize = INDEX_WORKERS / 64;

/// The bits of the wire type: a pool keeps one mask a bit.
const WIRE_BITS: usize = u8::BITS as usize;

/// One model's record of the prefixes sent to its workers whose cache
/// events the store has not heard, shared by every deciding thread.
///
/// A membership write takes the workers lock, then the table's lock, then
/// a record's guard when it needs all three. A decision takes a record's
/// guard alone.
pub(super) struct Predicted {
    record: RwLock<PrefixPredictor>,
    /// The record's bound as the last pool set it: a pool's build takes
    /// the record's guard, which the deciders hold, only where it moved.
    bound: AtomicUsize,
    /// Work waits for the sweep: a worker left, or the bound came down.
    sweeping: AtomicBool,
}

/// What a pool's generation asks of its model's record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Predict {
    /// The record's bound, in placements.
    pub(super) bound: usize,
    /// Whether the record's holders are named ahead of the index's: where
    /// the pool's profile reads sent prefixes and no prefix stage of it
    /// reads cache events, so the index's holders must not take the room.
    pub(super) ahead: bool,
}

/// Where a pool stands in its model's index, as the membership write that
/// made the pool derived it.
#[expect(
    dead_code,
    reason = "the decision views were the one reader of four of these fields: the tables leave in a following change"
)]
struct Tables {
    /// The pool the tables are of: its model and its rows' role.
    model: Box<str>,
    role: Option<Role>,
    /// The model's index; `None` while no row of the model is indexed.
    index: Option<Arc<Index>>,
    /// The model's record of sent prefixes, where the pool's generation
    /// asks for one. It speaks for a row until the row's lane is heard.
    predicted: Option<Arc<Predicted>>,
    /// Whether a row of the pool holds no lane in the index: the record
    /// speaks for that row whatever a lane hears.
    lacking: bool,
    /// Whether an issue over this pool may still record for each row.
    /// The store clears the word when the worker leaves, or stays under
    /// another offer or rank. Empty where the pool has no record.
    allowed: Box<[Arc<AtomicBool>]>,
    /// Index id -> row of the pool; [`NONE`] for an id no row of it holds.
    row_of: Box<[u32]>,
    /// Row -> index id; [`NONE`] for a row the index does not hold.
    id_of: Box<[u32]>,
    /// The pool's rows as the index names them, a bit an id: what a read is
    /// kept within.
    members: Box<[u64]>,
    /// The members that state each bit of the wire type, by the bit's
    /// number, as `members` holds them; empty for a bit no member states.
    by_wire: [Box<[u64]>; WIRE_BITS],
    /// Each row's admission, in the rows' clock.
    admitted: Box<[u64]>,
    /// The count of the index's takes the tables are known whole at, each
    /// id its row's own ([`Index::taken`]): the count they were made at,
    /// then the count at each membership write that kept them
    /// ([`KvTables::stands`]), both under the table's lock. A naming reads
    /// it before its lookup: a count told later covers no earlier read.
    taken: AtomicU64,
}

/// What a publication carries of one pool's index: where the pool stands in
/// it, and the pool's warming rows by its profile's warm-up ruling as a
/// write last read them. The tables are of the membership write that made
/// the pool. The warming rows are read again at a window's publication and
/// replaced when they moved, so a decision reads them with the rows it
/// loaded and never derives them. A young row's naming ends at an instant
/// of its own, which the list carries: a decision past it names the row no
/// more, whether or not a window was published between.
pub(super) struct PoolIndex {
    tables: Arc<Tables>,
    /// The ruling the warming rows were read under.
    rule: Option<WarmUpRule>,
    /// The generation's settings for the sent-prefix record.
    predict: Option<Predict>,
    /// The pool's warming rows, ascending, each with the instant its naming
    /// ends on the store's clock, in microseconds.
    warming: Box<[(u32, u64)]>,
    /// Whether the rows are the thin, and from when they are named.
    ruled: WarmUpState,
    /// The earliest end among the rows: a decision before it names them as
    /// they stand.
    #[expect(
        dead_code,
        reason = "the decision views were this instant's one reader: it leaves in a following change"
    )]
    first_end: u64,
}

/// A writer's buffers for the reading of a pool, kept across readings.
#[derive(Default)]
pub(super) struct Reading {
    sizes: Vec<u32>,
    level: Vec<u32>,
    rows: Vec<(u32, u64)>,
}

impl PoolIndex {
    /// The pool's warming rows at `at` under `rule`, into `reading`, and
    /// how they are read at a later instant: the ruling's own entry over
    /// each row's blocks in the index and its admission.
    fn warming(
        tables: &Tables,
        rule: Option<WarmUpRule>,
        at: u64,
        reading: &mut Reading,
    ) -> WarmUpState {
        let Some(rule) = rule else {
            reading.rows.clear();
            return WarmUpState::default();
        };
        // The index read once, ahead of the walk: the tables hold a cell (the
        // count they are whole at), so a shared reference to them does not tell
        // the compiler this word stands still across the walk's calls.
        let index = tables.index.as_deref();
        let size = |&id: &u32| match index {
            Some(index) if id != NONE => {
                u32::try_from(index.chain.worker_block_count(id)).unwrap_or(u32::MAX)
            }
            _ => 0,
        };
        reading.sizes.clear();
        reading.sizes.extend(tables.id_of.iter().map(size));
        rule.list_warming_rows(
            &reading.sizes,
            &tables.admitted,
            at,
            &mut reading.level,
            &mut reading.rows,
        )
    }

    /// The ruling the warming rows were read under.
    pub(super) fn rule(&self) -> Option<WarmUpRule> {
        self.rule
    }

    pub(super) fn predict(&self) -> Option<Predict> {
        self.predict
    }

    /// Whether the standing is of the pool of `rows` serving `model`: a
    /// pool is its model's in one role.
    pub(super) fn is_of(&self, model: &str, rows: &[Candidate]) -> bool {
        let tables = &*self.tables;
        *tables.model == *model && tables.role == rows.first().map(|row| row.role)
    }

    /// Whether the tables are of `rows` as they stand, each admitted at its
    /// instant in `admitted`, where no lane of theirs moved since the
    /// tables were made: the rows are as many, each was admitted when it
    /// was, and each lane's row states the wires it stated. A row with no
    /// lane is in no table but by its admission.
    pub(super) fn holds(&self, rows: &[Candidate], admitted: &[u64]) -> bool {
        let tables = &*self.tables;
        rows.len() == tables.id_of.len()
            && *admitted == *tables.admitted
            && tables
                .allowed
                .iter()
                .all(|allowed| allowed.load(Ordering::Relaxed))
            && rows.iter().zip(tables.id_of.iter()).all(|(row, &id)| {
                let (word, bit) = ((id / 64) as usize, 1u64 << (id % 64));
                id == NONE
                    || tables.by_wire.iter().enumerate().all(|(wire, stated)| {
                        stated.get(word).is_some_and(|held| held & bit != 0)
                            == (row.wire_modes.bits() & (1 << wire) != 0)
                    })
            })
    }

    /// The pool read again at `at` under `rule`: `None` while the warming
    /// rows stand as they were read, so a publication that moved none
    /// allocates nothing.
    pub(super) fn again(
        &self,
        rule: Option<WarmUpRule>,
        at: u64,
        reading: &mut Reading,
    ) -> Option<Self> {
        let ruled = Self::warming(&self.tables, rule, at, reading);
        let stands = rule == self.rule && ruled == self.ruled && *reading.rows == *self.warming;
        (!stands).then(|| {
            Self::read(
                Arc::clone(&self.tables),
                rule,
                self.predict,
                ruled,
                &reading.rows,
            )
        })
    }

    /// A pool's standing from a reading of its warming rows.
    fn read(
        tables: Arc<Tables>,
        rule: Option<WarmUpRule>,
        predict: Option<Predict>,
        ruled: WarmUpState,
        rows: &[(u32, u64)],
    ) -> Self {
        let ends = rows.iter().map(|&(_, until)| until);
        Self {
            tables,
            rule,
            predict,
            warming: rows.into(),
            ruled,
            first_end: ends.min().unwrap_or(u64::MAX),
        }
    }
}

/// One model's index: the chain index, how many times an id was taken, the
/// count each id was last taken at, and whether each id's lane was heard.
///
/// The chain frees an id for the next row once its row leaves, and a pool's
/// tables of before still map the id to the row that left
/// ([`Tables::row_of`]). So a reader of the tables has to ask after its
/// lookup whether the count of takes stands where its tables were known
/// whole before it ([`Self::taken`]), and the answer must be no older than
/// the blocks the lookup read. Where it moved, each row the lookup kept is
/// asked whether its id was taken since, by the count its last take made.
/// The chain reads its coverage relaxed and writes it by release, and its
/// reader runs an acquire fence after its reads, for its own snapshot: this
/// file rests on neither. What orders the answer after a block, by who
/// wrote it:
///
/// - the row's own lane: the take is said under the table's lock before
///   the row's lane exists ([`KvTables::member`]), a batch finds its lane
///   under that lock, and the lane passes a release fence before a batch's
///   blocks reach the chain ([`Held`]);
/// - another row's lane, at a run's split, which writes the holdings of
///   every row on the run, or where a partial table grows, which copies
///   every row's entry: the chain passes a release fence of its own where
///   a version step begins, under the run's lock, and every write of a
///   split and the store that publishes a grown table stand behind one. A
///   holding stands on a run by a write under that lock, or by the write
///   of the lane that made the run, before another lane could find it;
/// - the reader: an acquire fence between its lookup and its question.
///
/// A release fence and the reader's synchronize where the lookup read what
/// was written behind the first, so a reader that read a row's block
/// counts that row's take, and reads the count that take left on its id.
/// A take tells its id a later count than any before it, so an id taken
/// since a count answers so for good, whoever took it. The orderings of
/// an id's own count carry nothing.
struct Index {
    chain: ChainIndex,
    /// Each row's view of its cache groups other than main attention,
    /// under the row's id: written by a lent body ([`LaneWriter::groups`]).
    groups: GroupViews,
    /// Index id -> the count its last take made; 0 for an id never taken.
    took: [AtomicU64; INDEX_WORKERS],
    /// How many times a row took an id ([`Self::take`]).
    taken: AtomicU64,
    /// A bit an index id: a batch that carried an event was applied
    /// through the lane of the row that holds the id. A batch with no
    /// event sets nothing, and neither does one the lane's cursor refuses:
    /// the index holds nothing of the row yet, so the record keeps its
    /// cover. The lane's own word says less, that any batch ran through it
    /// ([`Lane::heard`]). The lane writes its own bit under its lock.
    /// The bit is a hint of the instant it was read at and orders nothing.
    heard: [AtomicU64; MASK_WORDS],
}

impl Index {
    fn new() -> Self {
        Self {
            chain: ChainIndex::with_max_workers(INDEX_WORKERS),
            groups: GroupViews::new(),
            took: std::array::from_fn(|_| AtomicU64::new(0)),
            taken: AtomicU64::new(0),
            heard: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// The lane that holds `id` says whether a batch that carried an event
    /// was applied through it.
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

    /// Whether a batch that carried an event was applied through the lane
    /// that holds `id`.
    fn heard(&self, id: u32) -> bool {
        let word = self.heard.get((id / 64) as usize);
        word.is_some_and(|word| word.load(Ordering::Relaxed) & (1 << (id % 64)) != 0)
    }

    /// A row takes `id`, which the chain just handed out: the take is
    /// counted, and the id keeps the count it made. Said under the table's
    /// lock, before the row's lane exists: every block the row writes
    /// reaches the chain after it, behind the lane's fence.
    fn take(&self, id: u32) {
        // The row that held the id before may have been heard: the row
        // that takes it is not.
        self.hear(id, false);
        // Nor does it find the view of the row before it, whatever path
        // that row left by.
        self.groups.remove(id);
        let count = self.taken.fetch_add(1, Ordering::Release) + 1;
        if let Some(took) = self.took.get(id as usize) {
            took.store(count, Ordering::Relaxed);
        }
    }

    /// How many times a row took an id of this index. A pool's tables keep
    /// the count they are known whole at, said under the table's lock, and
    /// a reader asks again after its lookup and its fence: while the count
    /// stands, every block it read under an id is the row's that the tables
    /// name.
    fn taken(&self) -> u64 {
        self.taken.load(Ordering::Acquire)
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
    /// A record stays alive only while a pool holds it.
    predicted: FxHashMap<Box<str>, Weak<Predicted>>,
    /// The same word in every pool that can record for this worker.
    allowed: FxHashMap<WorkerId, Weak<AtomicBool>>,
    block_sizes: HashMap<String, u32>,
    /// How many times a lane was made or retired, a model's block size of
    /// record moved, or a worker's word to record was cleared: a pool's
    /// place in an index moves with one of the first two, its record's
    /// bound with the third, a row's word with the fourth, and none of
    /// them with any other write.
    moves: u64,
}

impl Table {
    fn predicted(&mut self, model: &str, bound: usize) -> Arc<Predicted> {
        if let Some(predicted) = self.predicted.get(model).and_then(Weak::upgrade) {
            // Every pool's build passes here under the table's lock. The
            // record's guard is taken only where the bound moved: a
            // generation's apply, or a block size of record.
            let was = predicted.bound.swap(bound, Ordering::Relaxed);
            if was != bound {
                let mut record = predicted.record.write();
                record.set_bound(bound);
                // Under the guard, behind the bound: a sweep that reads
                // this word reads the lower bound with it.
                if bound < was {
                    predicted.sweeping.store(true, Ordering::Relaxed);
                }
            }
            return predicted;
        }
        // Drop names of models whose last pool let its record go.
        self.predicted
            .retain(|_, predicted| predicted.strong_count() != 0);
        let predicted = Arc::new(Predicted {
            record: RwLock::new(PrefixPredictor::new(bound)),
            bound: AtomicUsize::new(bound),
            sweeping: AtomicBool::new(false),
        });
        self.predicted
            .insert(model.into(), Arc::downgrade(&predicted));
        predicted
    }

    fn allows(&mut self, rows: &[Candidate]) -> Box<[Arc<AtomicBool>]> {
        rows.iter()
            .map(|row| {
                let held = self.allowed.entry(row.id).or_default();
                held.upgrade().unwrap_or_else(|| {
                    let allowed = Arc::new(AtomicBool::new(true));
                    *held = Arc::downgrade(&allowed);
                    allowed
                })
            })
            .collect()
    }
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
pub(crate) fn indexed_under(row: &WorkerRow) -> Option<&String> {
    let facts = &row.facts;
    facts.models.first().filter(|_| facts.signals.events)
}

/// The store's KV tables: the lanes, the indexes they write, the records
/// of sent prefixes and the block sizes of record.
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
        table.moves += 1;
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
            counts: LaneCounts::default(),
        };
        table.lanes.insert(
            row.id,
            Entry {
                rank: facts.dp_rank,
                model: model.as_str().into(),
                worker,
                lane: Arc::new(Mutex::new(lane)),
            },
        );
        Some(worker)
    }

    /// Where the pool of `rows` serving `model` stands in the model's index
    /// as this membership write leaves it, with its warming rows at `at`
    /// under `rule`. A row that lists the model after its first name is in
    /// the pool and not in the index, and a model with no indexed row has
    /// no index: its rows warm by their admission alone. `admitted` holds
    /// each row's admission and `at` the write's instant, both in the rows'
    /// clock. Called under the workers lock, once a pool a membership
    /// write.
    #[expect(
        clippy::too_many_arguments,
        reason = "publishes the pool's reading and both of its settings"
    )]
    pub(super) fn pool(
        &self,
        model: &str,
        rows: &[Candidate],
        admitted: &[u64],
        rule: Option<WarmUpRule>,
        predict: Option<Predict>,
        at: u64,
        reading: &mut Reading,
    ) -> PoolIndex {
        let (index, predicted, allowed, id_of, taken) = {
            let mut table = self.table.lock();
            let index = table.indexes.get(model).map(|held| Arc::clone(&held.index));
            let taken = index.as_ref().map_or(0, |index| index.taken());
            let id_of: Box<[u32]> = rows
                .iter()
                .map(|row| match table.lanes.get(&row.id) {
                    Some(entry) if *entry.model == *model => entry.worker,
                    _ => NONE,
                })
                .collect();
            let predicted = predict.map(|predict| table.predicted(model, predict.bound));
            let allowed = if predicted.is_some() {
                table.allows(rows)
            } else {
                Box::default()
            };
            (index, predicted, allowed, id_of, taken)
        };
        let ids = id_of
            .iter()
            .filter(|&&id| id != NONE)
            .map(|&id| id as usize + 1)
            .max()
            .unwrap_or(0);
        let mut row_of = vec![NONE; ids];
        let words = ids.div_ceil(64);
        let mut members = vec![0u64; words];
        let mut by_wire: [Vec<u64>; WIRE_BITS] = Default::default();
        for (row, (&id, candidate)) in id_of.iter().zip(rows).enumerate() {
            let (word, bit) = ((id / 64) as usize, 1u64 << (id % 64));
            let (Some(place), Some(member)) = (row_of.get_mut(id as usize), members.get_mut(word))
            else {
                continue;
            };
            *place = row as u32;
            *member |= bit;
            for (wire, stated) in by_wire.iter_mut().enumerate() {
                if candidate.wire_modes.bits() & (1 << wire) != 0 {
                    if stated.is_empty() {
                        stated.resize(words, 0);
                    }
                    if let Some(stated) = stated.get_mut(word) {
                        *stated |= bit;
                    }
                }
            }
        }
        let tables = Arc::new(Tables {
            model: model.into(),
            role: rows.first().map(|row| row.role),
            index,
            predicted,
            lacking: id_of.contains(&NONE),
            allowed,
            row_of: row_of.into(),
            id_of,
            members: members.into(),
            by_wire: by_wire.map(Vec::into_boxed_slice),
            admitted: admitted.into(),
            taken: AtomicU64::new(taken),
        });
        let ruled = PoolIndex::warming(&tables, rule, at, reading);
        PoolIndex::read(tables, rule, predict, ruled, &reading.rows)
    }

    /// How many times a lane was made or retired ([`Self::member`],
    /// [`Self::forget`]), a model's block size of record moved
    /// ([`Self::set_block_size`], [`Self::forget_block_size`]), or a
    /// worker's word to record was cleared ([`Self::forget_sent`]): while
    /// the count stands, every pool stands in its model's index where its
    /// tables hold it, its record of sent prefixes under its bound, and
    /// each of its rows may record as it could.
    pub(super) fn moves(&self) -> u64 {
        self.table.lock().moves
    }

    /// Whether `standing`, which an earlier membership write made of the
    /// pool of `rows` serving `model`, stands in the model's index where
    /// [`Self::pool`] would stand it now: the index is the one it reads,
    /// and each row's id in it is the one its tables hold. Asked where a
    /// lane moved since the last write ([`Self::moves`]), of a standing
    /// whose tables are of these rows ([`PoolIndex::holds`]). Called under
    /// the workers lock.
    pub(super) fn stands(&self, model: &str, rows: &[Candidate], standing: &PoolIndex) -> bool {
        let tables = &*standing.tables;
        let table = self.table.lock();
        let index = table.indexes.get(model).map(|held| &held.index);
        let same = match (index, &tables.index) {
            (Some(now), Some(then)) => Arc::ptr_eq(now, then),
            (None, None) => true,
            _ => false,
        };
        let stands = same
            && rows.iter().zip(tables.id_of.iter()).all(|(row, &then)| {
                let now = match table.lanes.get(&row.id) {
                    Some(entry) if *entry.model == *model => entry.worker,
                    _ => NONE,
                };
                now == then
            });
        // Each row holds the id its tables hold, under the table's lock:
        // the tables are whole at the count of this instant. By release:
        // a naming that reads this count before its lookup reads what
        // this write found.
        if let (true, Some(index)) = (stands, index) {
            tables.taken.store(index.taken(), Ordering::Release);
        }
        stands
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
        table.moves += 1;
        let last = table.indexes.get_mut(&entry.model).is_some_and(|model| {
            model.lanes -= 1;
            model.lanes == 0
        });
        if last {
            table.indexes.remove(&entry.model);
        }
    }

    /// A worker left, or stays under another offer or rank.
    /// Skip its predictions at once and take its placements out over later
    /// sweeps. A record that holds no place for it is read under its
    /// shared guard and left: a model's deciders wait on no removal from
    /// another model. Called under the workers lock.
    pub(super) fn forget_sent(&self, id: WorkerId) {
        let records: Vec<_> = {
            let mut table = self.table.lock();
            if let Some(allowed) = table.allowed.remove(&id).and_then(|held| held.upgrade()) {
                allowed.store(false, Ordering::Relaxed);
                // The pools that hold the word are made anew at this
                // write, though it may move no member and no offer.
                table.moves += 1;
            }
            table.predicted.values().filter_map(Weak::upgrade).collect()
        };
        for predicted in records {
            // The word above is cleared: no write gives the worker a place
            // behind this read.
            if !predicted.record.read().holds(id) {
                continue;
            }
            let mut record = predicted.record.write();
            record.forget(id);
            predicted.sweeping.store(true, Ordering::Relaxed);
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
        // The index speaks for the row from the first batch that carried
        // an event and was applied. A batch the cursor refuses writes no
        // block, and the record keeps its cover until one is applied.
        hear_written(held, outcome == BatchOutcome::Applied && !events.is_empty());
        if let Some(reason) = emptied {
            self.note(reason);
        }
        outcome
    }

    /// [`Self::admit`] for a batch its host's own rules apply: the cursor
    /// reads the header alone, and an admitted batch lends the lane to
    /// `body`, once, inside the lane's turn. The lane says its row heard
    /// behind a body that wrote, as behind a typed batch with an event. A
    /// body that unwinds has the lane start over ([`Lent`]).
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
        let emptied = match admitted(cursor, held, facts, facts.clears) {
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
            table.moves += 1;
        }
    }

    /// Drop the size of record so a returning model can set it anew.
    pub(super) fn forget_block_size(&self, model: &str) {
        let mut table = self.table.lock();
        if table.block_sizes.remove(model).is_some() {
            table.moves += 1;
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
            .map_or(0, |held| held.index.chain.worker_block_count(held.worker))
    }

    pub(super) fn counts(&self, id: WorkerId, rank: u16) -> Option<LaneCounts> {
        Some(self.lane(id, Some(rank))?.lock().counts)
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
