//! What a decision reads of the prompts its workers hold, from the
//! gateway's own data.
//!
//! The affinity stages of a profile ask two things of a request's pool:
//! which workers hold a leading part of the prompt, and which workers are
//! warming ([`AffinityRows`]). In a gateway's own process the answers live
//! in the gateway's index of its workers' cache events ([`CacheIndex`]) and
//! on the worker objects. [`decide`] names a request's rows from them, hands
//! them to the decision and, where the decision picked, writes the engine's
//! record of the prefixes it sent. A pair decision reads a column a side in
//! the rows' place ([`AffinityFacts`]): [`column`] fills one from the same
//! data, and [`sent`] writes the record behind the pair's pick.
//!
//! What is kept, and what guards it:
//!
//! - **Once a list of workers** ([`PoolAffinity`]): each worker's number in
//!   the model's index, what the index held for it and when it was
//!   admitted, and the list's warming rows by the profile's own ruling
//!   ([`WarmUpRule`]). It is a ruling read from the index and the worker
//!   objects, never a copy of a worker's health or load. One decision reads
//!   it again at most once a second, behind one atomic flag, and swaps it in
//!   by pointer. A decision loads the pointer and writes no shared word.
//! - **Once a model** ([`Affinity`]): the record of the prefixes sent to the
//!   workers the index does not speak for ([`PrefixPredictor`]), behind one
//!   reader-writer lock. A naming takes it as a reader, and only where the
//!   list holds such a worker. A pick takes it by a try for its first
//!   sixteen blocks, so a pick that meets a held lock writes nothing and
//!   waits for nobody; a longer prompt's later blocks wait their turn. The
//!   map of the records is behind a mutex that only the read of a list's
//!   table, a removed worker and the timer's sweep take.
//! - **Once a thread**: the buffers a naming writes into. A decision
//!   allocates nothing once they have grown.
//!
//! No decision holds a lock across its stages: the naming ends before the
//! decision runs, and the record is written after it answered.

use std::{
    cell::RefCell,
    fmt, mem,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};

use arc_swap::ArcSwapOption;
use kv_index::ChainIndex;
use parking_lot::{Mutex, RwLock, RwLockWriteGuard};
use rustc_hash::FxHashMap;
use smg_worker::identity::basic::Worker;

use crate::{
    document::spec::{AppliedSpec, IndexKind},
    engine::{
        decision::Decision,
        view::{
            AffinityFacts, AffinityHolder, AffinityRows, HashKind, Phase, RequestFacts, WorkerId,
        },
        warm_up::WarmUpState,
    },
    tracking::prefix_predictor::PrefixPredictor,
    Draw,
};

/// No row, and no number in the index.
const NONE: u32 = u32::MAX;

/// How long a list's table stands before one decision reads it again, in
/// microseconds.
const TABLE_STANDS_MICROS: u64 = 1_000_000;

/// The index numbers a table maps. An index hands its numbers out from zero
/// and takes them back when a worker leaves, so a number at or past this
/// one is a worker the table reads as not in the index.
const NUMBERS: usize = 1 << 16;

/// The stream of a decision's seed the cut of the holders is drawn from.
const HOLDERS_STREAM: u64 = 0x3c6e_f372_fe94_f82b;

/// The stream the cut of the warming rows is drawn from.
const WARMING_STREAM: u64 = 0xa54f_f53a_5f1d_36f1;

/// A value under `under` from the draw's stream.
fn below(draw: &mut Draw, under: u32) -> u32 {
    ((u128::from(draw.next_u64()) * u128::from(under)) >> 64) as u32
}

/// One model's index of cache events as its host keeps it: what a naming
/// reads of it. A gateway implements it for the index its workers' event
/// streams write.
pub trait CacheIndex: Send + Sync {
    /// The index's own number of the worker at `address`. `None` while no
    /// stream of the worker's cache events was opened into the index.
    fn worker_number(&self, address: &str) -> Option<u32>;

    /// How many blocks the index holds for the worker of `number`.
    fn worker_blocks(&self, number: u32) -> usize;

    /// The deepest holders of a request. `block_hashes` are the request's
    /// token blocks from its first, each `block_size` tokens. Of the workers
    /// `within` names, a bit a number, that hold a leading part of the
    /// request, at most `cap` go to `kept`, which is emptied first: each
    /// with the count of leading blocks it holds, in no order. Every worker
    /// of a depth that fits whole is kept, from the deepest down, and of the
    /// depth the cap cuts as many as fit, drawn by `below`, which is asked
    /// for a value under its argument.
    fn deepest_into(
        &self,
        block_hashes: &[u64],
        block_size: u32,
        within: &[u64],
        cap: usize,
        below: &mut dyn FnMut(u32) -> u32,
        kept: &mut Vec<(u32, u32)>,
    );
}

/// The chain index itself. A host that keeps more beside its index (the
/// cache groups of a hybrid engine) implements the trait for its own type.
impl CacheIndex for ChainIndex {
    fn worker_number(&self, address: &str) -> Option<u32> {
        self.worker_id(address)
    }

    fn worker_blocks(&self, number: u32) -> usize {
        self.worker_block_count(number)
    }

    fn deepest_into(
        &self,
        block_hashes: &[u64],
        _: u32,
        within: &[u64],
        cap: usize,
        below: &mut dyn FnMut(u32) -> u32,
        kept: &mut Vec<(u32, u32)>,
    ) {
        Self::deepest_into(self, block_hashes, |hash| *hash, within, cap, below, kept);
    }
}

/// Where a host keeps its models' indexes of cache events.
pub trait CacheIndexes: Send + Sync {
    /// The index of `model`, the model id its workers registered under.
    /// `None` where no worker of the model streams cache events.
    fn index(&self, model: &str) -> Option<Arc<dyn CacheIndex>>;

    /// The size of one cache block of `model` in tokens, as its workers'
    /// events taught it. `None` where the host knows none.
    fn block_size(&self, model: &str) -> Option<u32>;
}

/// What the engine keeps for the affinity stages across decisions: the
/// host's indexes of cache events, and its own record of the prefixes it
/// sent, one a model.
#[derive(Default)]
pub struct Affinity {
    indexes: OnceLock<Arc<dyn CacheIndexes>>,
    /// The record of sent prefixes of each model, by the model id its
    /// workers registered under. Locked where a list's table is read, where
    /// a worker left and at the timer's sweep: never by a decision's naming
    /// or its write, which hold their record by the list's table.
    sent: Mutex<FxHashMap<Box<str>, Arc<Sent>>>,
}

impl fmt::Debug for Affinity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Affinity")
            .field("indexes", &self.indexes.get().is_some())
            .field("records", &self.sent.lock().len())
            .finish()
    }
}

impl Affinity {
    /// Hands the engine its host's indexes of cache events, once, at the
    /// host's start. A later call is ignored. An engine handed none reads
    /// every worker as publishing no cache events.
    pub fn set_indexes(&self, indexes: Arc<dyn CacheIndexes>) {
        // The first hand-over stands.
        drop(self.indexes.set(indexes));
    }

    /// The size of one cache block of `model` in tokens, as the host's
    /// indexes know it: what a prompt of the model is hashed by. `None`
    /// where they know none, and the document's size serves.
    pub fn block_size(&self, model: &str) -> Option<u32> {
        self.indexes.get()?.block_size(model)
    }

    /// A worker left the pool or was replaced: every record of sent
    /// prefixes forgets it. Takes the records' map and, for a record that
    /// holds the worker, that record's lock as a writer.
    pub fn forget(&self, worker: WorkerId) {
        for sent in self.sent.lock().values() {
            sent.forget(worker);
        }
    }

    /// The work the records left waiting, a bounded step of each: the
    /// placements of workers that left, and the blocks past a bound that
    /// came down. For the engine's timer. A record a decision holds is
    /// passed over.
    pub fn sweep(&self) {
        for sent in self.sent.lock().values() {
            if sent.sweeping.load(Ordering::Relaxed) {
                if let Some(mut record) = sent.record.try_write() {
                    sent.sweep(&mut record);
                }
            }
        }
    }

    /// How many picks' prefixes the records left out, every model's
    /// together: a write that met a held record at its first turn waits for
    /// nobody and records nothing. For a host's own count beside its count
    /// of decisions, never for a decision.
    pub fn left_out(&self) -> u64 {
        let records = self.sent.lock();
        let left_out = |sent: &Arc<Sent>| sent.left_out.load(Ordering::Relaxed);
        records.values().map(left_out).sum()
    }

    /// The record of `model`, made at its first asking and held to `bound`
    /// placements.
    fn sent_of(&self, model: &str, bound: usize) -> Arc<Sent> {
        let sent = {
            let mut records = self.sent.lock();
            if let Some(sent) = records.get(model) {
                Arc::clone(sent)
            } else {
                let sent = Arc::new(Sent::new(bound));
                records.insert(model.into(), Arc::clone(&sent));
                sent
            }
        };
        // Outside the map's mutex: a bound that moved takes the record.
        sent.bound_to(bound);
        sent
    }
}

/// One model's record of the prefixes sent to its workers the index does
/// not speak for, shared by every deciding thread.
struct Sent {
    record: RwLock<PrefixPredictor>,
    /// The record's bound as the last table set it: a table's read takes
    /// the record, which the deciders hold, only where the bound moved.
    bound: AtomicUsize,
    /// Work waits for a sweep: a worker left, or the bound came down.
    sweeping: AtomicBool,
    /// How many workers were told gone. Written under the record's write
    /// guard: a write in parts stops where it moved since its naming.
    left: AtomicU64,
    /// How many writes were left out whole, their first turn having met a
    /// held lock. Written by such a write alone, so a pick that records
    /// writes no shared word for it.
    left_out: AtomicU64,
}

/// What a naming read of the record its pick may write.
#[derive(Clone, Copy, Debug)]
struct RecordRead {
    tick: u16,
    /// The record holds the whole request for a worker of the list.
    whole: bool,
    /// The request's last block was used at this tick.
    fresh: bool,
    /// [`Sent::left`] at the read.
    left: u64,
}

/// What the fill of a column read of the record of sent prefixes
/// ([`column`]). The write behind the pick made over that column stands on
/// it ([`sent`]).
#[derive(Clone, Copy, Debug)]
pub struct SentRead {
    read: RecordRead,
    /// The rows sent the record's deepest block of the request, the first
    /// of them, as many as fit: a pick of one of them may be a repeat that
    /// writes nothing.
    deepest: [u32; Self::KEPT],
}

impl SentRead {
    /// The deepest rows a read keeps. A pick of a row past them is written
    /// again, which costs a write and moves nothing a reader can see.
    const KEPT: usize = 8;
}

/// The record's tick at an instant of the decision clock: about a second.
const fn tick_of(now_micros: u64) -> u16 {
    (now_micros >> 20) as u16
}

impl Sent {
    /// A write gives the lock back after this many blocks.
    const SLICE: usize = 16;
    /// A sweep's step.
    const SWEEP: usize = 64;

    fn new(bound: usize) -> Self {
        Self {
            record: RwLock::new(PrefixPredictor::new(bound)),
            bound: AtomicUsize::new(bound),
            sweeping: AtomicBool::new(false),
            left: AtomicU64::new(0),
            left_out: AtomicU64::new(0),
        }
    }

    /// Holds the record to `bound` placements. A lower bound forgets
    /// nothing here: the next writes and sweeps take the oldest blocks out.
    fn bound_to(&self, bound: usize) {
        if self.bound.swap(bound, Ordering::Relaxed) != bound {
            self.record.write().set_bound(bound);
            self.sweeping.store(true, Ordering::Relaxed);
        }
    }

    fn forget(&self, worker: WorkerId) {
        if !self.record.read().holds(worker) {
            return;
        }
        let mut record = self.record.write();
        self.left.fetch_add(1, Ordering::Relaxed);
        record.forget(worker);
        self.sweeping.store(true, Ordering::Relaxed);
    }

    /// One step of the work that waits, under the writer's guard.
    fn sweep(&self, record: &mut PrefixPredictor) {
        if self.sweeping.load(Ordering::Relaxed) && record.sweep(Self::SWEEP) < Self::SWEEP {
            self.sweeping.store(false, Ordering::Relaxed);
        }
    }

    /// Whether a pick behind `read` is a repeat that writes nothing: the
    /// record already holds the whole request for the picked worker, its
    /// end used at this tick, so a write would move nothing a reader can
    /// see. `held` says the naming found the picked worker among those
    /// sent the request's deepest block.
    fn repeats(&self, read: RecordRead, held: bool) -> bool {
        if !(read.whole && read.fresh && held) {
            return false;
        }
        // A fresh repeat need not wait for a writer to sweep.
        if self.sweeping.load(Ordering::Relaxed) {
            if let Some(mut record) = self.record.try_write() {
                self.sweep(&mut record);
            }
        }
        true
    }

    /// Records the request of `keys` as sent to `target`, behind the naming
    /// that read `read`. Not for a repeat ([`Self::repeats`]).
    ///
    /// The request is written sixteen blocks a turn of the lock. An entry
    /// is a prediction, so a write that meets a held lock at its first turn
    /// is left out whole and waits for no thread; its later turns wait, so
    /// a prompt begun is recorded to its end. A worker that left since the
    /// naming stops the write: the record gives no place to a worker told
    /// gone.
    fn write(&self, keys: &[u64], read: RecordRead, target: WorkerId) {
        if keys.is_empty() {
            return;
        }
        let ends = (Self::SLICE..keys.len())
            .step_by(Self::SLICE)
            .chain(std::iter::once(keys.len()));
        for end in ends {
            let mut record = if end > Self::SLICE {
                self.record.write()
            } else if let Some(record) = self.record.try_write() {
                record
            } else {
                self.left_out.fetch_add(1, Ordering::Relaxed);
                return;
            };
            if self.left.load(Ordering::Relaxed) != read.left {
                return;
            }
            record.record(&keys[..end], target, read.tick);
            if end == keys.len() {
                self.sweep(&mut record);
            }
            RwLockWriteGuard::unlock_fair(record);
        }
    }
}

/// What is kept of one list of workers for the affinity stages, beside the
/// list: its table, and the flag of the one decision that reads the table
/// again.
#[derive(Default)]
pub struct PoolAffinity {
    table: ArcSwapOption<Table>,
    /// A decision is reading the table again. The others decide by the
    /// table that stands.
    reading: AtomicBool,
}

impl fmt::Debug for PoolAffinity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolAffinity")
            .field("read", &self.table.load().is_some())
            .finish()
    }
}

/// Gives the flag of a table's read back, whatever becomes of the read.
struct Reading<'a>(&'a AtomicBool);

impl Drop for Reading<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// One list of workers as the affinity stages' rows are named from it, for
/// the decisions of one model and phase: read in one pass over the list and
/// standing for a second. It holds what the index and the worker objects
/// said at its read, as a ruling: who the index speaks for, and who warms.
struct Table {
    /// The place the table answers for.
    model: Box<str>,
    phase: Phase,
    /// The decision clock at the read.
    read_at: u64,
    /// The most rows of each list a decision is named: the profile's
    /// sample. `0` where the profile is named no row.
    cap: usize,
    /// The place's stages read affinity. Where they do and the profile is
    /// named no row, a single decision is handed the column whole.
    reads: bool,
    /// One entry a worker of the list, in the list's order: its age at the
    /// read, the blocks the index held for it and what they gained since
    /// its admission, and nothing held. A decision that takes a column
    /// copies it, brings the ages to its own instant and writes its
    /// request's holders over it. A decision reads these here, and not from
    /// the index and the worker objects, because that costs a lookup a
    /// worker where this costs one copy of the list. Empty where the stages
    /// read no affinity.
    column: Box<[AffinityFacts]>,
    /// The model's index of cache events, where it has one.
    index: Option<Arc<dyn CacheIndex>>,
    /// Index number -> row of the list.
    row_of: Box<[u32]>,
    /// A bit an index number: a row of the list the index speaks for. The
    /// index's lookup is kept within these.
    within: Box<[u64]>,
    /// A bit a row: the index speaks for the row. It holds a number in the
    /// index and a block or more there. The record of sent prefixes speaks
    /// for every other row.
    heard: Box<[u64]>,
    /// Whether the index speaks for any row, and whether the record does.
    heard_any: bool,
    unheard_any: bool,
    /// The model's record of sent prefixes, where the profile reads one.
    sent: Option<Arc<Sent>>,
    /// The record's holders are named ahead of the index's: no prefix stage
    /// of the profile reads cache events.
    ahead: bool,
    /// The list's workers by id, each with its row: where a worker of the
    /// record stands. Empty where the list keeps no record.
    by_id: Box<[(WorkerId, u32)]>,
    /// The list's warming rows, ascending, each with the instant its naming
    /// ends on the decision clock.
    warming: Box<[(u32, u64)]>,
    /// Whether the rows are the thin, and from when they are named.
    ruled: WarmUpState,
    /// The earliest end among the rows.
    first_end: u64,
}

/// A duration in microseconds, saturating.
fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

/// Microseconds as a column's whole milliseconds, saturating.
fn millis(micros: u64) -> u32 {
    u32::try_from(micros / 1_000).unwrap_or(u32::MAX)
}

/// A count as a row's 32 bits, saturating.
fn count(blocks: usize) -> u32 {
    u32::try_from(blocks).unwrap_or(u32::MAX)
}

/// Sets the bit of `at`.
fn set(mask: &mut [u64], at: usize) {
    if let Some(word) = mask.get_mut(at / 64) {
        *word |= 1 << (at % 64);
    }
}

impl Table {
    /// The table of the list asked, for the decisions of its place, read at
    /// the decision's instant: one pass over the list. For each worker, its
    /// number in the model's index (one lookup by address), the blocks the
    /// index holds for it, what its index gained since its admission and
    /// its admission instant. Then the profile's ruling of who warms, over
    /// those columns.
    fn read(asked: &Asked<'_>) -> Self {
        let Asked {
            affinity,
            spec,
            workers,
            ids,
            model,
            phase,
            now,
        } = *asked;
        let engine = spec.engine();
        let cap = match engine.affinity_rows_cap(model, phase) {
            u32::MAX => usize::MAX,
            cap => cap as usize,
        };
        let reads = cap != 0 || engine.needs_for(model, phase).affinity;
        let mut table = Self {
            model: model.into(),
            phase,
            read_at: now,
            cap,
            reads,
            column: Box::default(),
            index: None,
            row_of: Box::default(),
            within: Box::default(),
            heard: Box::default(),
            heard_any: false,
            unheard_any: false,
            sent: None,
            ahead: false,
            by_id: Box::default(),
            warming: Box::default(),
            ruled: WarmUpState::default(),
            first_end: u64::MAX,
        };
        if !reads {
            return table;
        }
        // The index is its workers' model's, by the id they registered
        // under: a request may name the model by another.
        let registered = workers.first().map_or(model, |worker| worker.model_id());
        let index = affinity
            .indexes
            .get()
            .and_then(|indexes| indexes.index(registered));
        let rows = workers.len();
        let mut numbers = vec![NONE; rows];
        let mut sizes = vec![0u32; rows];
        let mut growth = vec![0u32; rows];
        let mut admitted = vec![0u64; rows];
        let mut column = vec![AffinityFacts::default(); rows];
        let mut numbered = 0usize;
        for (row, worker) in workers.iter().enumerate() {
            let age = micros(worker.admitted_age());
            admitted[row] = now.saturating_sub(age);
            column[row].admitted_age_ms = millis(age);
            let Some(index) = &index else {
                continue;
            };
            let number = index.worker_number(worker.url());
            let Some(number) = number.filter(|&number| (number as usize) < NUMBERS) else {
                continue;
            };
            let blocks = index.worker_blocks(number);
            numbers[row] = number;
            sizes[row] = count(blocks);
            // The worker keeps its own baseline: what its index held when
            // this admission was first read.
            growth[row] = count(worker.warmup_growth(blocks));
            column[row].indexed_blocks = sizes[row];
            column[row].indexed_growth = growth[row];
            numbered = numbered.max(number as usize + 1);
        }
        table.column = column.into();
        let mut row_of = vec![NONE; numbered];
        let mut within = vec![0u64; numbered.div_ceil(64)];
        let mut heard = vec![0u64; rows.div_ceil(64)];
        for (row, (&number, &size)) in numbers.iter().zip(&sizes).enumerate() {
            if number == NONE {
                continue;
            }
            row_of[number as usize] = row as u32;
            if size > 0 {
                set(&mut within, number as usize);
                set(&mut heard, row);
                table.heard_any = true;
            }
        }
        table.unheard_any = (0..rows).any(|row| !is_set(&heard, row));
        table.index = index;
        table.row_of = row_of.into();
        table.within = within.into();
        table.heard = heard.into();
        if let Some(rule) = engine.pool_rule_for(model, phase) {
            let (mut level, mut warming) = (Vec::new(), Vec::new());
            table.ruled = rule.list_warming_rows_grown(
                &sizes,
                &growth,
                &admitted,
                now,
                &mut level,
                &mut warming,
            );
            let ends = warming.iter().map(|&(_, until)| until);
            table.first_end = ends.min().unwrap_or(u64::MAX);
            table.warming = warming.into();
        }
        let record = engine.pool_record_for(model, phase);
        let settings = spec.index_for(model);
        if let Some(ahead) = record.filter(|_| settings.kind == IndexKind::Tree) {
            let size = affinity.block_size(registered);
            let size = size.unwrap_or_else(|| spec.block_size()).max(1);
            let bound = (settings.max_size / u64::from(size)).max(1);
            let bound = usize::try_from(bound).unwrap_or(usize::MAX);
            table.sent = Some(affinity.sent_of(registered, bound));
            table.ahead = ahead;
            let mut by_id: Vec<(WorkerId, u32)> = ids.iter().copied().zip(0u32..).collect();
            by_id.sort_unstable_by_key(|&(id, _)| id);
            table.by_id = by_id.into();
        }
        table
    }

    /// Whether the table answers for the decisions of `model` at `phase`.
    fn is_of(&self, model: &str, phase: Phase) -> bool {
        self.phase == phase && *self.model == *model
    }

    /// Whether the index speaks for `row`.
    fn heard(&self, row: u32) -> bool {
        is_set(&self.heard, row as usize)
    }

    /// The row of the worker `id` where the record of sent prefixes speaks
    /// for it: a worker of the list the index does not speak for.
    fn spoken_row(&self, id: WorkerId) -> Option<u32> {
        let at = self.by_id.binary_search_by_key(&id, |&(id, _)| id).ok()?;
        let row = self.by_id[at].1;
        (!self.heard(row)).then_some(row)
    }

    /// Names a request's rows into `buffers`: its deepest holders and the
    /// list's warming rows, each list at most the cap, ascending by row.
    ///
    /// The index names the holders among the rows it speaks for, in one
    /// lookup under the cap. The record of sent prefixes names them among
    /// the others: after the index's, in the room they leave, or ahead of
    /// them where no prefix stage of the profile reads cache events. So no
    /// row is named twice. The index is asked for a request hashed by token
    /// blocks alone, the record for a request of either kind.
    ///
    /// The warming rows are the table's whose naming has not ended at the
    /// request's instant: whole within the cap, a draw of the cap past it.
    ///
    /// Returns what was read of the record, where it was read: the pick's
    /// write stands on it.
    fn name(&self, facts: &RequestFacts<'_>, buffers: &mut Buffers) -> Option<RecordRead> {
        let Buffers {
            holders,
            warming,
            kept,
            keys,
            deepest,
            picked,
            live,
            ..
        } = buffers;
        holders.clear();
        warming.clear();
        let cap = self.cap;
        let mut draw = Draw::new(facts.seed ^ HOLDERS_STREAM);
        let hashed = !facts.block_hashes.is_empty();
        // The record is read where it speaks for a row of the list.
        let record = self.sent.as_deref().filter(|_| hashed && self.unheard_any);
        let mut read = None;
        if self.ahead {
            read =
                record.map(|sent| self.name_sent(sent, facts, keys, deepest, holders, &mut draw));
        }
        let room = cap.saturating_sub(holders.len());
        // An engine's events are token blocks: a request hashed by windows
        // of its text asks the record alone.
        let tokens = hashed && facts.hash_kind == HashKind::Token;
        let index = self.index.as_deref();
        if let Some(index) = index.filter(|_| tokens && room > 0 && self.heard_any) {
            index.deepest_into(
                facts.block_hashes,
                facts.block_size,
                &self.within,
                room,
                &mut |under| below(&mut draw, under),
                kept,
            );
            for &(number, blocks) in kept.iter() {
                let row = self.row_of.get(number as usize).copied().unwrap_or(NONE);
                if row != NONE {
                    holders.push(AffinityHolder {
                        row,
                        overlap_blocks: blocks,
                        predicted_blocks: 0,
                    });
                }
            }
        }
        if !self.ahead {
            read =
                record.map(|sent| self.name_sent(sent, facts, keys, deepest, holders, &mut draw));
        }
        holders.sort_unstable_by_key(|holder| holder.row);
        self.name_warming(facts, warming, picked, live);
        read
    }

    /// Reads the record of sent prefixes for the request: the deepest block
    /// of it that a worker the record speaks for was sent, as a count of
    /// blocks, and the workers sent it, into `deepest`. The request's keys
    /// are folded into `keys`. The record's lock is held as a reader for
    /// the one read and given back before a row is chosen.
    fn read_sent(
        &self,
        sent: &Sent,
        facts: &RequestFacts<'_>,
        keys: &mut Vec<u64>,
        deepest: &mut Vec<WorkerId>,
    ) -> (u32, RecordRead) {
        PrefixPredictor::fold_into(keys, facts.namespace_seed, facts.block_hashes);
        let tick = tick_of(facts.now_micros);
        let usable = |id: WorkerId| self.spoken_row(id).is_some();
        let record = sent.record.read();
        let found = record.deepest_among(keys, &usable);
        deepest.clear();
        deepest.extend(found.holders());
        let read = RecordRead {
            tick,
            whole: found.blocks() as usize == keys.len(),
            fresh: found.fresh(tick),
            left: sent.left.load(Ordering::Relaxed),
        };
        (found.blocks(), read)
    }

    /// The record's holders of the request, behind the holders already
    /// named and in the room the cap leaves: the workers sent the deepest
    /// block of it that a worker the record speaks for was sent, a draw of
    /// them where they would fill more than the room.
    fn name_sent(
        &self,
        sent: &Sent,
        facts: &RequestFacts<'_>,
        keys: &mut Vec<u64>,
        deepest: &mut Vec<WorkerId>,
        holders: &mut Vec<AffinityHolder>,
        draw: &mut Draw,
    ) -> RecordRead {
        let (blocks, read) = self.read_sent(sent, facts, keys, deepest);
        let base = holders.len();
        let room = self.cap.saturating_sub(base);
        if room == 0 {
            return read;
        }
        let mut seen = 0usize;
        for &id in deepest.iter() {
            let Some(row) = self.spoken_row(id) else {
                continue;
            };
            let holder = AffinityHolder {
                row,
                overlap_blocks: 0,
                predicted_blocks: blocks,
            };
            seen += 1;
            if seen <= room {
                holders.push(holder);
            } else {
                // Past the room each worker is as likely to stand as another.
                let drawn = below(draw, count(seen)) as usize;
                if drawn < room {
                    holders[base + drawn] = holder;
                }
            }
        }
        read
    }

    /// The list's warming rows at the request's instant, into `warming`.
    fn name_warming(
        &self,
        facts: &RequestFacts<'_>,
        warming: &mut Vec<u32>,
        picked: &mut Vec<u32>,
        live: &mut Vec<(u32, u64)>,
    ) {
        let now = facts.now_micros;
        // Nobody before the ruling names them, and each row until its end.
        if self.warming.is_empty() || now < self.ruled.from {
            return;
        }
        let mut rows: &[(u32, u64)] = &self.warming;
        if now >= self.first_end {
            live.clear();
            live.extend(rows.iter().filter(|&&(_, until)| now < until).copied());
            rows = live;
        }
        if rows.len() <= self.cap {
            warming.extend(rows.iter().map(|&(row, _)| row));
            return;
        }
        // Past the cap, a draw of the cap by position: each row as likely
        // as another, none twice.
        let mut draw = Draw::new(facts.seed ^ WARMING_STREAM);
        let (all, cap) = (count(rows.len()), count(self.cap));
        picked.clear();
        for last in all - cap..all {
            let at = below(&mut draw, last + 1);
            let taken = picked.contains(&at);
            picked.push(if taken { last } else { at });
        }
        let drawn = picked.iter().filter_map(|&at| rows.get(at as usize));
        warming.extend(drawn.map(|&(row, _)| row));
        warming.sort_unstable();
    }

    /// Fills `column`, one entry a worker of the list in the list's order:
    /// what the table read of each worker, its age brought to the request's
    /// instant, and over it what the request's holders hold.
    ///
    /// The index says what the rows it speaks for hold, every holder with
    /// its own depth, in one lookup. The record of sent prefixes says it
    /// for the others: the workers sent the deepest block of the request
    /// that such a worker was sent. A column has an entry a row, so nothing
    /// is capped and nothing drawn. As for the rows a decision is named,
    /// the index is asked for a request hashed by token blocks alone and
    /// the record for a request of either kind.
    ///
    /// Returns what was read of the record, where it was read: the pick's
    /// write stands on it.
    fn fill(
        &self,
        facts: &RequestFacts<'_>,
        buffers: &mut Buffers,
        column: &mut Vec<AffinityFacts>,
    ) -> Option<SentRead> {
        let Buffers {
            kept,
            keys,
            deepest,
            ..
        } = buffers;
        column.clear();
        column.extend_from_slice(&self.column);
        let later = millis(facts.now_micros.saturating_sub(self.read_at));
        if later > 0 {
            for entry in column.iter_mut() {
                entry.admitted_age_ms = entry.admitted_age_ms.saturating_add(later);
            }
        }
        if facts.block_hashes.is_empty() {
            return None;
        }
        let tokens = facts.hash_kind == HashKind::Token;
        let index = self.index.as_deref();
        if let Some(index) = index.filter(|_| tokens && self.heard_any) {
            // Every holder fits: the lookup cuts nothing and draws nothing.
            index.deepest_into(
                facts.block_hashes,
                facts.block_size,
                &self.within,
                column.len(),
                &mut |_| 0,
                kept,
            );
            for &(number, blocks) in kept.iter() {
                let row = self.row_of.get(number as usize).copied().unwrap_or(NONE);
                if let Some(entry) = column.get_mut(row as usize) {
                    entry.overlap_blocks = blocks;
                }
            }
        }
        // The record is read where it speaks for a row of the list.
        let sent = self.sent.as_deref().filter(|_| self.unheard_any)?;
        let (blocks, record) = self.read_sent(sent, facts, keys, deepest);
        let mut read = SentRead {
            read: record,
            deepest: [NONE; SentRead::KEPT],
        };
        let mut named = 0usize;
        for &id in deepest.iter() {
            let Some(row) = self.spoken_row(id) else {
                continue;
            };
            if let Some(entry) = column.get_mut(row as usize) {
                entry.predicted_blocks = blocks;
            }
            if let Some(slot) = read.deepest.get_mut(named) {
                *slot = row;
            }
            named += 1;
        }
        Some(read)
    }
}

/// Whether the bit of `at` is set. A place past the mask's end is not.
fn is_set(mask: &[u64], at: usize) -> bool {
    mask.get(at / 64)
        .is_some_and(|word| word & (1 << (at % 64)) != 0)
}

/// A naming's buffers, the calling thread's own across decisions.
#[derive(Default)]
struct Buffers {
    /// The rows that hold a leading part of the request, ascending by row.
    holders: Vec<AffinityHolder>,
    /// The warming rows the decision reads, ascending.
    warming: Vec<u32>,
    /// The index's answer: a number and its depth.
    kept: Vec<(u32, u32)>,
    /// One key a block of the request, for the record.
    keys: Vec<u64>,
    /// The workers sent the record's deepest block of the request.
    deepest: Vec<WorkerId>,
    picked: Vec<u32>,
    live: Vec<(u32, u64)>,
    /// The column a single decision is handed where its profile is named
    /// no row.
    column: Vec<AffinityFacts>,
}

impl Buffers {
    const fn new() -> Self {
        Self {
            holders: Vec::new(),
            warming: Vec::new(),
            kept: Vec::new(),
            keys: Vec::new(),
            deepest: Vec::new(),
            picked: Vec::new(),
            live: Vec::new(),
            column: Vec::new(),
        }
    }
}

/// Runs `with` over the calling thread's buffers.
fn with_buffers<T>(with: impl FnOnce(&mut Buffers) -> T) -> T {
    BUFFERS.with(|buffers| match buffers.try_borrow_mut() {
        Ok(mut buffers) => with(&mut buffers),
        // A stage asked for a decision from inside this one, on this
        // thread: that decision takes buffers of its own.
        Err(_) => with(&mut Buffers::new()),
    })
}

thread_local! {
    /// The calling thread's buffers: a naming allocates nothing once they
    /// have grown to a request's blocks and the profile's cap.
    static BUFFERS: RefCell<Buffers> = const { RefCell::new(Buffers::new()) };
}

/// The list one decision is made over and the place it is decided for.
#[derive(Clone, Copy)]
struct Asked<'a> {
    affinity: &'a Affinity,
    spec: &'a AppliedSpec,
    workers: &'a [Arc<dyn Worker>],
    ids: &'a [WorkerId],
    /// The model as the request names it, and the phase the list is decided
    /// at: a pair's request carries one phase for its two lists.
    model: &'a str,
    phase: Phase,
    /// The decision's instant.
    now: u64,
}

/// One single decision of a profile whose stages read affinity.
///
/// It names the request's rows from the host's own data, hands them to
/// `decide` in the request's facts ([`RequestFacts::named`]) and returns
/// what `decide` answers. Where that is a pick and the record of sent
/// prefixes speaks for the first target, the request's prefix is written
/// for it: the record says where a prefix was decided to go.
///
/// `workers` is the list the decision is made over and `pool` what the
/// engine keeps beside that list. `ids` holds the workers' ids by their
/// place in the list, and a pick's first target names its worker by that
/// place and that id. `facts` carries the request's block hashes and the
/// decision's instant ([`RequestFacts::now_micros`]) on the one clock of the
/// engine's decisions.
///
/// A profile whose stages read no affinity, and a request whose host filled
/// a column or named rows itself, pass straight to `decide`. A profile that
/// reads affinity and is named no row (it shows its rows to a plugin, whose
/// stage reads the column whole) is handed the column [`column`] fills, in
/// [`RequestFacts::affinity`].
///
/// The rows live in the calling thread's buffers until this returns:
/// `decide` reads them and keeps nothing of them. Nothing is allocated once
/// the buffers have grown, but where a table is read: once a list, and then
/// by one decision a second.
pub fn decide(
    affinity: &Affinity,
    pool: &PoolAffinity,
    spec: &AppliedSpec,
    workers: &[Arc<dyn Worker>],
    ids: &[WorkerId],
    facts: &RequestFacts<'_>,
    decide: impl FnOnce(&RequestFacts<'_>) -> Decision,
) -> Decision {
    if !facts.affinity.is_empty() || facts.named.is_some() {
        return decide(facts);
    }
    debug_assert_eq!(workers.len(), ids.len(), "one id a worker of the list");
    let asked = Asked {
        affinity,
        spec,
        workers,
        ids,
        model: facts.model,
        phase: facts.phase,
        now: facts.now_micros,
    };
    with_buffers(|buffers| pool.with_table(&asked, |table| table.decide(facts, buffers, decide)))
}

/// One list's column for a decision that reads a column in the rows' place:
/// each side of a pair decision, by the side's own list and `phase`.
///
/// `column` is emptied and filled with one entry a worker of `workers`, in
/// the list's order: the worker's age, what the model's index holds for it
/// and what that gained since its admission, and what the worker holds of
/// the request, by the index where the index speaks for the worker and by
/// the record of sent prefixes where it does not. The caller hands the
/// column to the side it was filled for and to no other.
///
/// The parameters are [`decide`]'s, and `phase` is the phase the list is
/// decided at, which a pair's request does not carry for both its lists.
/// Returns what was read of the record of sent prefixes, where the record
/// speaks for a worker of the list: the caller hands it to [`sent`] behind
/// the pick. Nothing is allocated once `column` and the calling thread's
/// buffers have grown, but where a table is read.
#[expect(
    clippy::too_many_arguments,
    reason = "the list as `decide` takes it, the phase a pair's request does not carry, and the column"
)]
pub fn column(
    affinity: &Affinity,
    pool: &PoolAffinity,
    spec: &AppliedSpec,
    workers: &[Arc<dyn Worker>],
    ids: &[WorkerId],
    phase: Phase,
    facts: &RequestFacts<'_>,
    column: &mut Vec<AffinityFacts>,
) -> Option<SentRead> {
    debug_assert_eq!(workers.len(), ids.len(), "one id a worker of the list");
    let asked = Asked {
        affinity,
        spec,
        workers,
        ids,
        model: facts.model,
        phase,
        now: facts.now_micros,
    };
    let read =
        with_buffers(|buffers| pool.with_table(&asked, |table| table.fill(facts, buffers, column)));
    // A place whose stages read no affinity keeps no table: every worker
    // reads as holding nothing, just admitted.
    column.resize(workers.len(), AffinityFacts::default());
    read
}

/// Behind a pick made over a column ([`column`]): records the request's
/// prefix as sent to the picked worker `id`, where the record of sent
/// prefixes speaks for that worker. The record says where a prefix was
/// decided to go.
///
/// `pool`, `phase` and `facts` are the ones the column was filled with, and
/// `read` is what that fill returned. A worker the index speaks for, a
/// worker that is not of the list, and a list whose table was since read
/// for another place write nothing.
pub fn sent(
    pool: &PoolAffinity,
    phase: Phase,
    facts: &RequestFacts<'_>,
    read: SentRead,
    id: WorkerId,
) {
    let standing = pool.table.load();
    let table = standing.as_deref();
    let Some(table) = table.filter(|table| table.is_of(facts.model, phase)) else {
        return;
    };
    let (Some(sent), Some(row)) = (table.sent.as_deref(), table.spoken_row(id)) else {
        return;
    };
    if sent.repeats(read.read, read.deepest.contains(&row)) {
        return;
    }
    with_buffers(|buffers| {
        // The keys are the request's own, whichever list was filled last.
        PrefixPredictor::fold_into(&mut buffers.keys, facts.namespace_seed, facts.block_hashes);
        sent.write(&buffers.keys, read.read, id);
    });
}

impl PoolAffinity {
    /// Runs `with` over the list's table for the place asked: the one that
    /// stands, or the one this decision reads.
    fn with_table<T>(&self, asked: &Asked<'_>, with: impl FnOnce(&Table) -> T) -> T {
        // The pointer is loaded with no shared write, and the table it
        // names stands while this decision reads it.
        let standing = self.table.load();
        let table = standing.as_deref();
        if let Some(table) = table.filter(|table| table.is_of(asked.model, asked.phase)) {
            let stands = asked.now.saturating_sub(table.read_at) < TABLE_STANDS_MICROS;
            // One decision reads a table again; the others decide by the
            // one that stands.
            if stands || self.reading.swap(true, Ordering::AcqRel) {
                return with(table);
            }
            let table = {
                let _reading = Reading(&self.reading);
                self.read(asked)
            };
            return with(&table);
        }
        // A list with no table of this place: each decision that meets it
        // reads one, and the last one swapped in stands.
        with(&self.read(asked))
    }

    /// Reads the list's table at the decision's instant and swaps it in.
    fn read(&self, asked: &Asked<'_>) -> Arc<Table> {
        let table = Arc::new(Table::read(asked));
        self.table.store(Some(Arc::clone(&table)));
        table
    }
}

impl Table {
    /// One decision by this table: the naming, the decision, and the
    /// record's write behind a pick.
    fn decide(
        &self,
        facts: &RequestFacts<'_>,
        buffers: &mut Buffers,
        decide: impl FnOnce(&RequestFacts<'_>) -> Decision,
    ) -> Decision {
        if !self.reads {
            return decide(facts);
        }
        if self.cap == 0 {
            return self.decide_by_column(facts, buffers, decide);
        }
        let read = self.name(facts, buffers);
        let decision = decide(&RequestFacts {
            named: Some(AffinityRows {
                holders: &buffers.holders,
                warming: &buffers.warming,
                thin: self.ruled.thin,
                mean_in_flight: None,
            }),
            ..*facts
        });
        let (Some(read), Some(sent)) = (read, self.sent.as_deref()) else {
            return decision;
        };
        let target = decision.pick().and_then(|pick| pick.targets.first());
        // The index speaks for a worker whose cache events are heard: the
        // record takes no entry for it.
        let Some(target) = target.filter(|target| !self.heard(target.position)) else {
            return decision;
        };
        if !sent.repeats(read, buffers.deepest.contains(&target.id)) {
            sent.write(&buffers.keys, read, target.id);
        }
        decision
    }

    /// One decision of a profile that reads the column whole: the fill, the
    /// decision over the column, and the record's write behind a pick.
    fn decide_by_column(
        &self,
        facts: &RequestFacts<'_>,
        buffers: &mut Buffers,
        decide: impl FnOnce(&RequestFacts<'_>) -> Decision,
    ) -> Decision {
        // The column leaves the buffers while the fill writes the others.
        let mut column = mem::take(&mut buffers.column);
        let read = self.fill(facts, buffers, &mut column);
        let decision = decide(&RequestFacts {
            affinity: &column,
            ..*facts
        });
        buffers.column = column;
        let (Some(read), Some(sent)) = (read, self.sent.as_deref()) else {
            return decision;
        };
        let target = decision.pick().and_then(|pick| pick.targets.first());
        let Some(target) = target.filter(|target| !self.heard(target.position)) else {
            return decision;
        };
        if !sent.repeats(read.read, read.deepest.contains(&target.position)) {
            sent.write(&buffers.keys, read.read, target.id);
        }
        decision
    }
}

#[cfg(test)]
mod tests;
