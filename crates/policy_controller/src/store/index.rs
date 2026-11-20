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
//! A decision reads an index through its pool's standing in it
//! (`PoolIndex`), which the store publishes with the pool's rows: where
//! each index id stands in the pool, and the pool's warming rows as a
//! write last read them. The lookup itself is live, as the index is: the
//! request's blocks are looked up once a decision, under the cap of its
//! profile (its sample, or no cap where it takes none), and the holders
//! are named by their place in the decision's rows. An index id is free
//! for the next row once its row leaves, and a decision may read a world
//! published before that: the index counts its takes and keeps the count
//! each id was last taken at (`Index`), and a row is named a holder only
//! where its id was not taken since the tables the decision holds were
//! known whole.
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
        atomic::{fence, AtomicU64, Ordering},
        Arc,
    },
};

use kv_index::{ApplyError, ChainBlockMap, ChainIndex};
pub use kv_index::{ContentHash, SequenceHash, StoredBlock};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::{
    affinity::{WarmRule, Warming},
    store::{
        kv_events::{Admission, BatchFacts, BatchOutcome, KvCursor, RankCursor, ResyncReason},
        workers::WorkerRow,
        world::Naming,
        GatewayId,
    },
    view::{Candidate, NamedAffinity, NamedHolder, Role, WorkerId},
    Draw,
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
    index: Arc<Index>,
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

/// The streams of a decision's seed the cut of its holders and the cut of
/// its warming rows are drawn from: none another site reads ([`Draw`]).
const HOLDERS_STREAM: u64 = 0x3c6e_f372_fe94_f82b;
const WARMING_STREAM: u64 = 0xa54f_f53a_5f1d_36f1;

/// A value under `under` from `draw`, by a multiply: no division.
fn below(draw: &mut Draw, under: u32) -> u32 {
    ((u128::from(draw.next_u64()) * u128::from(under)) >> 64) as u32
}

/// The rows a naming wrote in the affinity column's place
/// ([`World::name`](crate::store::World::name)), with the naming's own
/// buffers: a host keeps it across decisions beside its view.
#[derive(Debug, Default)]
pub struct Named {
    /// The rows that hold a leading part of the request, ascending by row.
    pub holders: Vec<NamedHolder>,
    /// The pool's warming rows the decision reads, ascending.
    pub warming: Vec<u32>,
    /// The warming rows are the thin.
    pub thin: bool,
    kept: Vec<(u32, u32)>,
    picked: Vec<u32>,
    live: Vec<(u32, u64)>,
}

impl Named {
    /// The lists as a request carries them
    /// ([`RequestFacts::named`](crate::RequestFacts::named)).
    pub fn affinity(&self) -> NamedAffinity<'_> {
        NamedAffinity {
            holders: &self.holders,
            warming: &self.warming,
            thin: self.thin,
            mean_in_flight: None,
        }
    }

    /// No row named.
    pub(super) fn clear(&mut self) {
        self.holders.clear();
        self.warming.clear();
        self.thin = false;
    }
}

/// Where a pool stands in its model's index, as the membership write that
/// made the pool derived it.
struct Tables {
    /// The pool the tables are of: its model and its rows' role.
    model: Box<str>,
    role: Option<Role>,
    /// The model's index; `None` while no row of the model is indexed.
    index: Option<Arc<Index>>,
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

impl Tables {
    /// The workers a read is kept within for the view a decision holds: the
    /// pool's members, those that state one of its gateway's wires where it
    /// states them, less another gateway's host-local rows. `mask` is the
    /// caller's, written where the pool's own does not serve as it stands.
    fn within<'a>(
        &'a self,
        naming: &Naming<'_>,
        locals: &[(u32, GatewayId)],
        mask: &'a mut [u64; MASK_WORDS],
    ) -> &'a [u64] {
        let foreign = locals.iter().any(|&(_, owner)| owner != naming.gateway);
        if naming.wires.is_none() && !foreign {
            return &self.members;
        }
        let words = self.members.len().min(MASK_WORDS);
        let mask = &mut mask[..words];
        match naming.wires {
            None => mask.copy_from_slice(&self.members[..words]),
            Some(wires) => {
                mask.fill(0);
                // The narrowing's own test, a bit of the wire type at a time:
                // a member is kept where it states one of the gateway's.
                for (bit, stated) in self.by_wire.iter().enumerate() {
                    if wires.bits() & (1 << bit) != 0 {
                        for (word, stated) in mask.iter_mut().zip(stated.iter()) {
                            *word |= stated;
                        }
                    }
                }
            }
        }
        for &(row, owner) in locals {
            let id = self.id_of.get(row as usize).copied().unwrap_or(NONE);
            if owner != naming.gateway && id != NONE {
                if let Some(word) = mask.get_mut((id / 64) as usize) {
                    *word &= !(1u64 << (id % 64));
                }
            }
        }
        mask
    }

    /// The workers a read is kept within for a view its caller cut further
    /// than [`Self::within`] reads it: the view's own rows, each found
    /// among the pool's by one walk of the two lists, which stand in one
    /// order. A row the index does not hold is no bit.
    fn within_view<'a>(
        &'a self,
        pool: &[Candidate],
        view: &[Candidate],
        mask: &'a mut [u64; MASK_WORDS],
    ) -> &'a [u64] {
        let words = self.members.len().min(MASK_WORDS);
        let mask = &mut mask[..words];
        mask.fill(0);
        let mut rows = pool.iter().zip(self.id_of.iter());
        for kept in view {
            let Some((_, &id)) = rows.find(|(row, _)| row.id == kept.id) else {
                break;
            };
            if id != NONE {
                if let Some(word) = mask.get_mut((id / 64) as usize) {
                    *word |= 1u64 << (id % 64);
                }
            }
        }
        mask
    }

    /// How many times a naming makes its lookup again before it names no
    /// holder: each round more needs a take under a row it kept.
    const ROUNDS: usize = 4;

    /// What stands of the read in `kept`, kept within `within`, for a
    /// decision whose tables were known whole at `whole_at` and whose index
    /// counts a take since. Each row kept is asked whether its id was taken
    /// since ([`Index::taken_since`]): a take tells a later count for good,
    /// the same row back on its id included, so the ask is right at any
    /// instant behind the lookup's fence. Where no id kept was taken since,
    /// the read stands. Else a block may be another row's, and the lookup
    /// is made again ([`Self::again`]). The asks stand in a frame of their
    /// own: a decision that holds tables from before a take pays them, and
    /// nothing of a lookup it does not make. Off a decision's usual path.
    #[cold]
    #[inline(never)]
    fn look_again(
        &self,
        naming: &Naming<'_>,
        within: &[u64],
        cap: usize,
        whole_at: u64,
        kept: &mut Vec<(u32, u32)>,
    ) {
        let Some(index) = self.index.as_deref() else {
            return;
        };
        if kept.iter().any(|&(id, _)| index.taken_since(id, whole_at)) {
            Self::again(index, naming, within, cap, whole_at, kept);
        }
    }

    /// The lookup made again for a read that kept an id taken since
    /// `whole_at`: within the ids not taken since, by a draw of that round's
    /// own ([`Index::deepest_again`]), and its rows are asked again. Past
    /// [`Self::ROUNDS`] lookups no read of the naming stands, and it names
    /// no holder. Out of line: the rounds' frame and their seeds are no
    /// cost of the asks.
    #[cold]
    #[inline(never)]
    fn again(
        index: &Index,
        naming: &Naming<'_>,
        within: &[u64],
        cap: usize,
        whole_at: u64,
        kept: &mut Vec<(u32, u32)>,
    ) {
        let words = within.len().min(MASK_WORDS);
        // A round's stream is seeded by the first value of the one before
        // it, the first round's by the holders' stream.
        let mut seed = naming.seed ^ HOLDERS_STREAM;
        for _ in 0..Self::ROUNDS {
            seed = Draw::new(seed).next_u64();
            let own = index.own(&within[..words], whole_at);
            #[cfg(test)]
            seam::at(seam::Point::Again);
            index.deepest_again(naming, &own[..words], cap, seed, kept);
            // As in the naming: the read before the fence, the asks after.
            fence(Ordering::Acquire);
            if !kept.iter().any(|&(id, _)| index.taken_since(id, whole_at)) {
                return;
            }
        }
        kept.clear();
    }
}

/// A seam for the cases: a hook the naming thread runs once at a point
/// inside one naming, where a take or a keep lands between the naming's
/// reads. No case reaches there from outside. In no build but the cases'.
#[cfg(test)]
mod seam {
    use std::cell::RefCell;

    /// Where inside a naming a hook runs.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Point {
        /// Behind the naming's lookup, ahead of its fence and its question.
        Looked,
        /// Ahead of a lookup made again, behind the ids it is kept within.
        Again,
    }

    type Hook = Box<dyn FnOnce()>;

    thread_local! {
        static HOOK: RefCell<Option<(Point, Hook)>> = const { RefCell::new(None) };
    }

    /// `hook` runs once, in the next naming of this thread that reaches
    /// `point`.
    pub(super) fn set(point: Point, hook: impl FnOnce() + 'static) {
        HOOK.with(|held| *held.borrow_mut() = Some((point, Box::new(hook))));
    }

    /// Whether a hook is still set. A case holds that none is behind its
    /// naming: every step it set ran.
    pub(super) fn pending() -> bool {
        HOOK.with(|held| held.borrow().is_some())
    }

    /// The hook set for `point` runs. It is taken out first: a naming the
    /// hook itself asks for meets none.
    pub(super) fn at(point: Point) {
        let hook = HOOK.with(|held| {
            let mut held = held.borrow_mut();
            match &*held {
                Some((set, _)) if *set == point => held.take(),
                _ => None,
            }
        });
        if let Some((_, hook)) = hook {
            hook();
        }
    }
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
    rule: Option<WarmRule>,
    /// The pool's warming rows, ascending, each with the instant its naming
    /// ends on the store's clock, in microseconds.
    warming: Box<[(u32, u64)]>,
    /// Whether the rows are the thin, and from when they are named.
    ruled: Warming,
    /// The earliest end among the rows: a decision before it names them as
    /// they stand.
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
    fn warming(tables: &Tables, rule: Option<WarmRule>, at: u64, reading: &mut Reading) -> Warming {
        let Some(rule) = rule else {
            reading.rows.clear();
            return Warming::default();
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
        rule.name(
            &reading.sizes,
            &tables.admitted,
            at,
            &mut reading.level,
            &mut reading.rows,
        )
    }

    /// The ruling the warming rows were read under.
    pub(super) fn rule(&self) -> Option<WarmRule> {
        self.rule
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
        rule: Option<WarmRule>,
        at: u64,
        reading: &mut Reading,
    ) -> Option<Self> {
        let ruled = Self::warming(&self.tables, rule, at, reading);
        let stands = rule == self.rule && ruled == self.ruled && *reading.rows == *self.warming;
        (!stands).then(|| Self::read(Arc::clone(&self.tables), rule, ruled, &reading.rows))
    }

    /// A pool's standing from a reading of its warming rows.
    fn read(
        tables: Arc<Tables>,
        rule: Option<WarmRule>,
        ruled: Warming,
        rows: &[(u32, u64)],
    ) -> Self {
        let ends = rows.iter().map(|&(_, until)| until);
        Self {
            tables,
            rule,
            warming: rows.into(),
            ruled,
            first_end: ends.min().unwrap_or(u64::MAX),
        }
    }

    /// The rows of the pool a read for `naming` is kept within, by their
    /// index in the pool, ascending: for the cases.
    #[cfg(test)]
    pub(super) fn reached(&self, naming: &Naming<'_>, locals: &[(u32, GatewayId)]) -> Vec<u32> {
        let mut mask = [0u64; MASK_WORDS];
        let within = self.tables.within(naming, locals, &mut mask);
        let named = |id: &usize| within[id / 64] & (1 << (id % 64)) != 0;
        let ids = (0..within.len() * 64).filter(named);
        let mut rows: Vec<u32> = ids
            .filter_map(|id| self.tables.row_of.get(id).copied())
            .collect();
        rows.sort_unstable();
        rows
    }

    /// Name the rows of a decision's view for the affinity stages
    /// ([`NamedAffinity`]): the request's deepest holders and the pool's
    /// warming rows, each list at most `naming.cap` rows, ascending, by
    /// position in `view`. `pool` is the pool's rows as the publication
    /// holds them, `locals` its host-local rows, and `view` the rows the
    /// decision is made over: the pool's, less another gateway's host-local
    /// rows and the rows its gateway's wires took.
    ///
    /// The holders are one lookup under the cap: whole depths from the
    /// deepest while they fit, the depth the cap cuts by a draw from the
    /// decision's seed on a stream of its own; a head the whole pool holds
    /// is never listed. The lookup is kept within the rows the view can
    /// hold, so the cap is filled from them. Where an id of the index was
    /// taken since the pool's tables were known whole, each row the lookup
    /// kept is asked whether its own id was. Where one was, the lookup runs
    /// again, within the ids not taken since: a row that left is named by
    /// no block of the row that took its id, and takes no place of the cap.
    /// The warming rows are the
    /// publication's that the view holds, less the rows whose naming ended
    /// by the decision's instant: whole within the cap, a draw of the cap
    /// past it, on another stream.
    #[inline]
    pub(super) fn name(
        &self,
        naming: &Naming<'_>,
        pool: &[Candidate],
        locals: &[(u32, GatewayId)],
        view: &[Candidate],
        named: &mut Named,
    ) {
        self.name_in::<false>(naming, pool, locals, view, named);
    }

    /// [`Self::name`] for a view its caller cut by more than its gateway's
    /// wires since the build, as a request's hard demands cut it: the
    /// holders' lookup is kept within the view's own rows, so a row the
    /// cut took is named by nobody and takes no place of the cap.
    pub(super) fn name_cut(
        &self,
        naming: &Naming<'_>,
        pool: &[Candidate],
        locals: &[(u32, GatewayId)],
        view: &[Candidate],
        named: &mut Named,
    ) {
        self.name_in::<true>(naming, pool, locals, view, named);
    }

    /// The one naming, built once for each of its two callers. `CUT` says
    /// which rows the holders' lookup is kept within, and a build a
    /// caller keeps each caller's lookup in a frame of its own.
    fn name_in<const CUT: bool>(
        &self,
        naming: &Naming<'_>,
        pool: &[Candidate],
        locals: &[(u32, GatewayId)],
        view: &[Candidate],
        named: &mut Named,
    ) {
        named.clear();
        named.thin = self.ruled.thin;
        let tables = &*self.tables;
        let whole = view.len() == pool.len();
        let cap = match naming.cap {
            0 => usize::MAX,
            cap => cap as usize,
        };
        // Where a row of the pool stands in the view: at its own index where
        // the view is the pool whole, else by its id, the sorted slice its
        // own index.
        let place = |row: u32| -> Option<u32> {
            if whole {
                return Some(row);
            }
            let id = pool.get(row as usize)?.id;
            let at = view.binary_search_by_key(&id, |candidate| candidate.id);
            at.ok().map(|at| at as u32)
        };
        let index = tables.index.as_ref();
        if let Some(index) = index.filter(|_| !naming.block_hashes.is_empty()) {
            let mut mask = [0u64; MASK_WORDS];
            let within = if CUT {
                tables.within_view(pool, view, &mut mask)
            } else {
                tables.within(naming, locals, &mut mask)
            };
            // The count the tables are whole at, read before the lookup: a
            // membership write that keeps them tells a later count, and
            // that count covers only what is read after it.
            let whole_at = tables.taken.load(Ordering::Acquire);
            // The naming's own lookup, in its own frame: one instantiation
            // with a second caller goes out of line for every decision
            // ([`Index::deepest_again`]).
            let mut draw = Draw::new(naming.seed ^ HOLDERS_STREAM);
            index.chain.deepest_into(
                naming.block_hashes,
                |hash| *hash,
                within,
                cap,
                |under| below(&mut draw, under),
                &mut named.kept,
            );
            #[cfg(test)]
            seam::at(seam::Point::Looked);
            // Every read of the lookup stands before this fence, and the
            // question of what was taken since after it ([`Index`]).
            fence(Ordering::Acquire);
            if index.taken() != whole_at {
                tables.look_again(naming, within, cap, whole_at, &mut named.kept);
            }
            for &(id, blocks) in &named.kept {
                let row = tables.row_of.get(id as usize).copied().unwrap_or(NONE);
                debug_assert!(row != NONE, "a read is kept within the pool's rows");
                let at = place(row);
                debug_assert!(
                    at.is_some(),
                    "a holder the read was kept within is in the view"
                );
                if let Some(at) = at {
                    named.holders.push(NamedHolder {
                        row: at,
                        overlap_blocks: blocks,
                        predicted_blocks: 0,
                    });
                }
            }
            named.holders.sort_unstable_by_key(|holder| holder.row);
        }
        // The warming rows of the decision's instant: nobody before the
        // ruling names them, and each row until its own end.
        let now = naming.now_micros;
        if now < self.ruled.from {
            return;
        }
        // The warming rows the view holds at the decision's instant, each
        // by its place there: the cap and the draw are over these, so a
        // view its gateway narrowed is named the cap of its own rows. The
        // pool's list serves as it stands where the view is the pool whole
        // and no naming has ended.
        let mut warming: &[(u32, u64)] = &self.warming;
        if !whole || now >= self.first_end {
            named.live.clear();
            let live = warming.iter().filter(|&&(_, until)| now < until);
            let placed = live.filter_map(|&(row, until)| Some((place(row)?, until)));
            named.live.extend(placed);
            warming = &named.live;
        }
        if warming.len() <= cap {
            named.warming.extend(warming.iter().map(|&(row, _)| row));
            return;
        }
        // Past the cap, a draw of the cap by position: each row as likely as
        // another, none twice.
        let mut draw = Draw::new(naming.seed ^ WARMING_STREAM);
        let count = warming.len() as u32;
        named.picked.clear();
        for last in count - cap as u32..count {
            let at = below(&mut draw, last + 1);
            let taken = named.picked.contains(&at);
            named.picked.push(if taken { last } else { at });
        }
        let rows = named
            .picked
            .iter()
            .filter_map(|&at| warming.get(at as usize));
        named.warming.extend(rows.map(|&(row, _)| row));
        named.warming.sort_unstable();
    }
}

/// One model's index: the chain index, how many times an id was taken, and
/// the count each id was last taken at.
///
/// The chain frees an id for the next row once its row leaves, and a pool's
/// tables of before still map the id to the row that left
/// ([`Tables::row_of`]). So a decision asks after its lookup whether the
/// count of takes stands where its tables were known whole before it
/// ([`Self::taken`]), and the answer must be no older than the blocks the
/// lookup read. Where it moved, each row the lookup kept is asked whether
/// its id was taken since ([`Self::taken_since`]). The chain reads its
/// coverage relaxed and writes it by release, and its reader runs an
/// acquire fence after its reads, for its own snapshot: this file rests on
/// neither. What orders the answer after a block, by who wrote it:
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
/// - the reader: an acquire fence in the naming, between its lookup and
///   its question ([`PoolIndex::name`]).
///
/// A release fence and the reader's synchronize where the lookup read what
/// was written behind the first, so a decision that read a row's block
/// counts that row's take, and reads the count that take left on its id.
/// A take tells its id a later count than any before it, so an id taken
/// since a count answers so for good, whoever took it. The orderings of
/// an id's own count carry nothing.
struct Index {
    chain: ChainIndex,
    /// Index id -> the count its last take made; 0 for an id never taken.
    took: [AtomicU64; INDEX_WORKERS],
    /// How many times a row took an id ([`Self::take`]).
    taken: AtomicU64,
}

impl Index {
    fn new() -> Self {
        Self {
            chain: ChainIndex::with_max_workers(INDEX_WORKERS),
            took: std::array::from_fn(|_| AtomicU64::new(0)),
            taken: AtomicU64::new(0),
        }
    }

    /// A row takes `id`, which the chain just handed out: the take is
    /// counted, and the id keeps the count it made. Said under the table's
    /// lock, before the row's lane exists: every block the row writes
    /// reaches the chain after it, behind the lane's fence.
    fn take(&self, id: u32) {
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

    /// The naming's lookup made again ([`Tables::again`]): the request's
    /// deepest holders within `within` under `cap`, into `kept` by index
    /// id, the depth the cap cuts by a draw from `seed`, the round's own. A
    /// round runs only where the lookup before it kept an id taken since,
    /// so the stream that lookup cut by would cut by the same values again,
    /// and favour the rows it reaches first. A lookup of its own beside the
    /// naming's: with one instantiation for both, every decision would pay
    /// a call for the lookup it makes once.
    fn deepest_again(
        &self,
        naming: &Naming<'_>,
        within: &[u64],
        cap: usize,
        seed: u64,
        kept: &mut Vec<(u32, u32)>,
    ) {
        let mut draw = Draw::new(seed);
        self.chain.deepest_into(
            naming.block_hashes,
            |hash| *hash,
            within,
            cap,
            |under| below(&mut draw, under),
            kept,
        );
    }

    /// Whether `id` was taken since the count `whole_at`. A reader asks
    /// behind its lookup's fence, for an id it read blocks under: a take
    /// counts for good, the same row back on its id included, so the blocks
    /// are the row's that held the id at that count only where this says
    /// no. An id nobody took since says no for a row that left, and the
    /// chain holds none of that row's blocks.
    fn taken_since(&self, id: u32, whole_at: u64) -> bool {
        let took = self.took.get(id as usize);
        took.is_none_or(|took| took.load(Ordering::Relaxed) > whole_at)
    }

    /// `within` less the ids taken since `whole_at`: what a lookup made
    /// again is kept within ([`Tables::again`]).
    fn own(&self, within: &[u64], whole_at: u64) -> [u64; MASK_WORDS] {
        let mut own = [0u64; MASK_WORDS];
        for (word, (own, &ids)) in own.iter_mut().zip(within).enumerate() {
            let mut left = ids;
            while left != 0 {
                let bit = left.trailing_zeros();
                left &= left - 1;
                if !self.taken_since(word as u32 * 64 + bit, whole_at) {
                    *own |= 1 << bit;
                }
            }
        }
        own
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
    /// How many times a lane was made or retired: a pool's place in an
    /// index moves with one of the two and with no other write.
    moves: u64,
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
    pub(super) fn pool(
        &self,
        model: &str,
        rows: &[Candidate],
        admitted: &[u64],
        rule: Option<WarmRule>,
        at: u64,
        reading: &mut Reading,
    ) -> PoolIndex {
        let (index, id_of, taken) = {
            let table = self.table.lock();
            let index = table.indexes.get(model).map(|held| Arc::clone(&held.index));
            let taken = index.as_ref().map_or(0, |index| index.taken());
            let id_of: Box<[u32]> = rows
                .iter()
                .map(|row| match table.lanes.get(&row.id) {
                    Some(entry) if *entry.model == *model => entry.worker,
                    _ => NONE,
                })
                .collect();
            (index, id_of, taken)
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
            row_of: row_of.into(),
            id_of,
            members: members.into(),
            by_wire: by_wire.map(Vec::into_boxed_slice),
            admitted: admitted.into(),
            taken: AtomicU64::new(taken),
        });
        let ruled = PoolIndex::warming(&tables, rule, at, reading);
        PoolIndex::read(tables, rule, ruled, &reading.rows)
    }

    /// How many times a lane was made or retired ([`Self::member`],
    /// [`Self::forget`]): while the count stands, every pool stands in its
    /// model's index where its tables hold it.
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
        if let Some(Held { index, worker, map }) = entry.lane.lock().held.take() {
            index.chain.remove_worker(worker, map);
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

    pub(super) fn counts(&self, id: WorkerId, rank: u16) -> Option<LaneCounts> {
        Some(self.lane(id, Some(rank))?.lock().counts)
    }

    /// The rows holding a prefix of the request's blocks in `model`'s
    /// index, each with the leading blocks it holds, by id: every holder
    /// listed, for the cases.
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
