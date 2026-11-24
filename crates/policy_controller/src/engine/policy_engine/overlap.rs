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
//! the rows' place ([`AffinityFacts`]): [`column()`] fills one from the same
//! data, and [`record_prefix`] writes the record behind the pair's pick.
//!
//! Behind a single decision the pick states what its first target holds of
//! the request ([`Overlap`]): the worker's entry of the rows the decision was
//! given, whoever named them. No stage is asked for it. It is stated where
//! the request's own profile reads affinity: rows named for a binding's
//! shadow alone state nothing of the live pick.
//!
//! A request whose host names its entry in a token tree ([`TokenKey`]) is
//! decided another way where the host lends the tree of the request's place
//! ([`CacheIndexes::token_tree`]) and the model has no index of cache
//! events: inside the tree's match ([`TokenTree`]). The match names the
//! workers sent the deepest prefix of the request, and the same descent
//! writes the request for the worker picked. The tree is the host's own,
//! the one its policy reads and writes: the engine keeps none.
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
//!   list holds such a worker. A pick writes behind the blocks its worker
//!   already holds, sixteen blocks a turn, and takes each turn by a try: a
//!   turn that meets a held lock ends the write there, so a pick waits for
//!   nobody. The map of the records is behind a mutex that only the read of
//!   a list's table, a change of membership and the timer's sweep take.
//! - **Once a thread**: the buffers a naming writes into. A decision
//!   allocates nothing once they have grown, but what a tree's own match
//!   and write allocate.
//!
//! No decision holds a lock across its stages: the naming ends before the
//! decision runs, and the record is written after it answered. A decision
//! inside a tree's match runs between the tree's read and its write, which
//! hold no lock across it.
//!
//! [`WarmUpRule`]: crate::engine::WarmUpRule

use std::{
    cell::{Cell, RefCell},
    fmt, mem,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};

use arc_swap::ArcSwapOption;
use kv_index::{
    chain_prefix_hash, ChainIndex, ContentHash, GroupViews, SequenceHash, TenantId, TokenTree,
};
use parking_lot::{Mutex, RwLock, RwLockWriteGuard};
use rustc_hash::{FxHashMap, FxHashSet};
use smg_worker::identity::basic::Worker;

use crate::{
    document::spec::{AppliedSpec, IndexKind},
    engine::{
        decision::{Decision, Overlap, Target},
        decision_engine::TokenKey,
        lease_events,
        view::{
            blocks_within, fingerprint, AffinityFacts, AffinityHolder, AffinityRows, Feed,
            HashKind, Phase, RequestFacts, WorkerId,
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

/// [`CacheIndex::deepest_into`] for a host that keeps its workers' cache
/// groups beside its chain index.
///
/// On a fleet with no worker that keeps groups the chain index keeps the
/// deepest holders itself, and nothing is allocated. Where a worker keeps
/// cache groups, every holder's depth is first lowered to what its groups
/// can serve ([`kv_index::WorkerGroups::cap`]), a holder lowered to nothing
/// leaves, and the deepest are kept after.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "the lookup's own arguments, beside the index and its groups"
)]
pub fn deepest_within_groups(
    index: &ChainIndex,
    groups: &GroupViews,
    block_hashes: &[u64],
    block_size: u32,
    within: &[u64],
    cap: usize,
    below: &mut dyn FnMut(u32) -> u32,
    kept: &mut Vec<(u32, u32)>,
) {
    if groups.is_empty() {
        index.deepest_into(block_hashes, |hash| *hash, within, cap, below, kept);
        return;
    }
    kept.clear();
    index.score_into(
        block_hashes,
        |hash| *hash,
        false,
        |worker, depth| {
            if is_set(within, worker as usize) {
                kept.push((worker, depth));
            }
        },
    );
    keep_within_groups(Some(groups), block_hashes, block_size, cap, below, kept);
}

/// [`deepest_within_groups`] over holders a host scored by an index of its
/// own: each of `holders` is a worker's number and the count of leading
/// blocks it holds. Those `within` names go to `kept`, which is emptied
/// first. Each is lowered to what its groups can serve, and the deepest are
/// kept.
#[expect(
    clippy::too_many_arguments,
    reason = "the lookup's own arguments, beside the holders and their groups"
)]
pub fn deepest_of_within_groups(
    holders: impl IntoIterator<Item = (u32, u32)>,
    groups: &GroupViews,
    block_hashes: &[u64],
    block_size: u32,
    within: &[u64],
    cap: usize,
    below: &mut dyn FnMut(u32) -> u32,
    kept: &mut Vec<(u32, u32)>,
) {
    kept.clear();
    let holders = holders.into_iter();
    kept.extend(holders.filter(|&(worker, _)| is_set(within, worker as usize)));
    let groups = (!groups.is_empty()).then_some(groups);
    keep_within_groups(groups, block_hashes, block_size, cap, below, kept);
}

thread_local! {
    /// The calling thread's own list for a request's chain of prefix
    /// hashes, taken for one lookup and put back behind it: the chain
    /// allocates nothing once the list has grown to the depth of the
    /// thread's deepest holder.
    static PREFIXES: Cell<Vec<SequenceHash>> = const { Cell::new(Vec::new()) };
}

/// Lowers each holder of `kept` to what its cache groups can serve, where
/// the host's workers keep `groups`, and keeps the deepest, at most `cap`.
///
/// The request's chain of prefix hashes is made from its block hashes in
/// the calling thread's own list ([`PREFIXES`]), no further than the depth
/// of the deepest holder with a group view: no view reads the chain past
/// its holder's depth.
fn keep_within_groups(
    groups: Option<&GroupViews>,
    block_hashes: &[u64],
    block_size: u32,
    cap: usize,
    below: &mut dyn FnMut(u32) -> u32,
    kept: &mut Vec<(u32, u32)>,
) {
    if let Some(groups) = groups {
        let table = groups.read();
        // A lookup made while its thread ends finds the thread's list
        // gone and takes a list of its own.
        let mut prefixes = PREFIXES.try_with(Cell::take).unwrap_or_default();
        prefixes.clear();
        kept.retain_mut(|(worker, depth)| {
            let Some(view) = table.view(*worker) else {
                return true;
            };
            // A holder's groups read the chain no further than its depth.
            let until = (*depth as usize).min(block_hashes.len());
            for &hash in block_hashes.get(prefixes.len()..until).unwrap_or_default() {
                // The chain's first hash is the first block's own.
                let prefix = prefixes.last().map_or(SequenceHash(hash), |&last| {
                    chain_prefix_hash(last, ContentHash(hash))
                });
                prefixes.push(prefix);
            }
            *depth = (*depth).min(view.cap(&prefixes, *depth, block_size as usize).depth);
            *depth > 0
        });
        let _ = PREFIXES.try_with(|list| list.set(prefixes));
    }
    keep_deepest(kept, cap, below);
}

/// Keeps the deepest of `kept`, at most `cap`, as the chain index's capped
/// lookup does: every holder of a depth that fits whole, from the deepest
/// down, and of the depth the cap cuts as many as fit, each as likely as
/// another by `below`, which is asked for a value under its argument.
fn keep_deepest(kept: &mut Vec<(u32, u32)>, cap: usize, below: &mut dyn FnMut(u32) -> u32) {
    if kept.len() <= cap {
        return;
    }
    kept.sort_unstable_by_key(|&(_, depth)| std::cmp::Reverse(depth));
    let Some(&(_, cut)) = kept.get(cap) else {
        return;
    };
    // The holders of the depth the cap cuts, and those of them that fit.
    let from = kept.partition_point(|&(_, depth)| depth > cut);
    let until = kept.partition_point(|&(_, depth)| depth >= cut);
    let room = cap.saturating_sub(from);
    for taken in 0..room {
        let left = until - from - taken;
        if left < 2 {
            break;
        }
        let left = u32::try_from(left).unwrap_or(u32::MAX);
        let drawn = below(left).min(left - 1) as usize;
        kept.swap(from + taken, from + taken + drawn);
    }
    kept.truncate(cap);
}

/// Where a host keeps its models' indexes of cache events, and its trees
/// of the prompts it sent.
pub trait CacheIndexes: Send + Sync {
    /// The index of `model`, the model id its workers registered under.
    /// `None` where no worker of the model streams cache events, and while
    /// the index holds no block: a request is then decided by the host's
    /// token tree, where it lends one ([`Self::token_tree`]).
    fn index(&self, model: &str) -> Option<Arc<dyn CacheIndex>>;

    /// The size of one cache block of `model` in tokens, as its workers'
    /// events taught it. `None` where the host knows none.
    fn block_size(&self, model: &str) -> Option<u32>;

    /// The host's own token tree of the prompts sent to the workers that
    /// serve `model` at `phase`: the tree its policy reads and writes for
    /// such a request, with a worker a tenant by its address. A request
    /// whose host names its entry ([`TokenKey`]) is decided inside this
    /// tree's match, which writes the request for the worker picked. The
    /// engine calls the match alone: the host bounds the tree and takes a
    /// worker that left out of it. A host never makes a tree for the engine
    /// alone where it has one. `None`, the default, where the host keeps
    /// none: the request is named its rows from the record of hashed
    /// prefixes. This question lends the one tree a tokenized request
    /// reads. A host's other records of the prompts it sent are not lent.
    fn token_tree(&self, model: &str, phase: Phase) -> Option<Arc<TokenTree>> {
        let _ = (model, phase);
        None
    }
}

/// What the engine keeps for the affinity stages across decisions: the
/// host's indexes of cache events, and its own record of the prefixes it
/// sent, one a model.
#[derive(Default)]
pub struct Affinity {
    indexes: OnceLock<Arc<dyn CacheIndexes>>,
    /// The record of sent prefixes of each model, by the model id its
    /// workers registered under. Locked where a list's table is read, at a
    /// change of membership and at the timer's sweep: never by a decision's naming
    /// or its write, which hold their record by the list's table.
    records: Mutex<FxHashMap<Box<str>, Arc<PrefixRecord>>>,
    /// What the records that left the map had left out
    /// ([`Self::left_out`]). Written under the map's mutex.
    left_out: AtomicU64,
}

impl fmt::Debug for Affinity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Affinity")
            .field("indexes", &self.indexes.get().is_some())
            .field("records", &self.records.lock().len())
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

    /// A worker left the pool, was replaced, or joins with no leave told
    /// ahead: every record of sent prefixes forgets it. Takes the records'
    /// map and, for a record that holds the worker, that record's lock as
    /// a writer.
    pub fn forget(&self, worker: WorkerId) {
        for record in self.records.lock().values() {
            record.forget(worker);
        }
    }

    /// The workers that stand, said whole behind a read of the host's
    /// registry: every record forgets each worker it holds a place for
    /// that is not among them, so a worker whose leave was never told
    /// keeps no place. Takes what [`Self::forget`] takes.
    pub(super) fn keep(&self, standing: &[WorkerId]) {
        let standing: FxHashSet<WorkerId> = standing.iter().copied().collect();
        for record in self.records.lock().values() {
            record.keep(&standing);
        }
    }

    /// The work the records left waiting, a bounded step of each: the
    /// placements of workers that left, and the blocks past a bound that
    /// came down. For the engine's timer. A record a decision holds is
    /// passed over.
    ///
    /// A record leaves the map here once the map alone holds it and it
    /// holds no worker's place: no list's table reads it, and every worker
    /// it was written for was forgotten. So a model's record goes behind
    /// its last worker, and a record with no placement yet can go between
    /// two tables of its list. The model's next table read makes a fresh
    /// record. A pick named from a record that left writes to the fresh
    /// one or writes nothing: an entry is a prediction, and the next
    /// request of the prefix writes it.
    pub fn sweep(&self) {
        let mut records = self.records.lock();
        // Nobody takes a record from the map but by a clone under this
        // mutex, so one the map alone holds is read by no table and written
        // by no pick.
        let left: Vec<_> = records
            .extract_if(|_, record| {
                let lone = Arc::get_mut(record)
                    .is_some_and(|record| record.predictor.get_mut().held().next().is_none());
                if !lone && record.sweeping.load(Ordering::Relaxed) {
                    if let Some(mut predictor) = record.predictor.try_write() {
                        record.sweep(&mut predictor);
                    }
                }
                lone
            })
            .collect();
        let left_out = left
            .iter()
            .map(|(_, record)| record.left_out.load(Ordering::Relaxed))
            .sum();
        self.left_out.fetch_add(left_out, Ordering::Relaxed);
        // What left is freed once the mutex is given back, so the read of a
        // list's table waits for no free.
        drop(records);
    }

    /// How many picks' prefixes the records left out in whole or in part,
    /// every model's together, those of the records that left the map among
    /// them: a write that meets a held record ends there and waits for
    /// nobody. For a host's own count beside its count of decisions, never
    /// for a decision.
    pub fn left_out(&self) -> u64 {
        let records = self.records.lock();
        let left_out = |record: &Arc<PrefixRecord>| record.left_out.load(Ordering::Relaxed);
        self.left_out.load(Ordering::Relaxed) + records.values().map(left_out).sum::<u64>()
    }

    /// The record of `model`, made at its first asking and held to `bound`
    /// placements.
    fn record_of(&self, model: &str, bound: usize) -> Arc<PrefixRecord> {
        let record = {
            let mut records = self.records.lock();
            if let Some(record) = records.get(model) {
                Arc::clone(record)
            } else {
                let record = Arc::new(PrefixRecord::new(bound));
                records.insert(model.into(), Arc::clone(&record));
                record
            }
        };
        // Outside the map's mutex: a bound that moved takes the record.
        record.bound_to(bound);
        record
    }
}

/// One model's record of the prefixes sent to its workers the index does
/// not speak for, shared by every deciding thread.
struct PrefixRecord {
    predictor: RwLock<PrefixPredictor>,
    /// The record's bound as the last table set it: a table's read takes
    /// the record, which the deciders hold, only where the bound moved.
    bound: AtomicUsize,
    /// Work waits for a sweep: a worker left, or the bound came down.
    sweeping: AtomicBool,
    /// How many leaves were told. Written under the record's guard, a
    /// reader's where the leave takes no place out: a write in parts stops
    /// where it moved since its naming.
    left: AtomicU64,
    /// How many writes ended before the request's last block, a turn of
    /// theirs having met a held lock. Written by such a write alone, so a
    /// pick that records whole writes no shared word for it.
    left_out: AtomicU64,
}

/// What a naming read of the record its pick may write.
#[derive(Clone, Copy, Debug)]
struct RecordRead {
    tick: u16,
    /// The leading blocks of the request the record holds for a worker of
    /// the list: the depth its deepest holders were sent.
    blocks: u32,
    /// The record holds the whole request for a worker of the list.
    whole: bool,
    /// The request's last block was used at this tick.
    fresh: bool,
    /// [`PrefixRecord::left`] at the read.
    left: u64,
}

/// What the fill of a column read of the record of sent prefixes
/// ([`column()`]). The write behind the pick made over that column stands on
/// it ([`record_prefix`]).
#[derive(Clone, Copy, Debug)]
pub struct PrefixRead {
    read: RecordRead,
    /// The rows sent the record's deepest block of the request, the first
    /// of them, as many as fit: a pick of one of them may be a repeat that
    /// writes nothing.
    deepest: [u32; Self::KEPT],
}

impl PrefixRead {
    /// The deepest rows a read keeps. A pick of a row past them is written
    /// again, which costs a write and moves nothing a reader can see.
    const KEPT: usize = 8;
}

/// The record's tick at an instant of the decision clock: about a second.
const fn tick_of(now_micros: u64) -> u16 {
    (now_micros >> 20) as u16
}

impl PrefixRecord {
    /// A write gives the lock back after this many blocks.
    const SLICE: usize = 16;
    /// A sweep's step.
    const SWEEP: usize = 64;

    fn new(bound: usize) -> Self {
        Self {
            predictor: RwLock::new(PrefixPredictor::new(bound)),
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
            self.predictor.write().set_bound(bound);
            self.sweeping.store(true, Ordering::Relaxed);
        }
    }

    /// Counts the leave also where the record holds no place for `worker`:
    /// a pick named ahead of the leave would give the worker its first
    /// place, and its write stops where the count moved. The count moves
    /// under the guard that reads or clears the place, so no read pairs
    /// the count behind a leave with the place ahead of it.
    fn forget(&self, worker: WorkerId) {
        let held = self.predictor.read();
        if !held.holds(worker) {
            self.left.fetch_add(1, Ordering::Relaxed);
            return;
        }
        drop(held);
        let mut predictor = self.predictor.write();
        self.left.fetch_add(1, Ordering::Relaxed);
        predictor.forget(worker);
        self.sweeping.store(true, Ordering::Relaxed);
    }

    /// Forgets every worker the record holds a place for that is not in
    /// `standing` ([`Self::forget`]), and counts one leave ahead of them: a
    /// worker `standing` names no more may be under its first pick.
    fn keep(&self, standing: &FxHashSet<WorkerId>) {
        let held = self.predictor.read();
        self.left.fetch_add(1, Ordering::Relaxed);
        let gone: Vec<WorkerId> = held
            .held()
            .filter(|worker| !standing.contains(worker))
            .collect();
        drop(held);
        for worker in gone {
            self.forget(worker);
        }
    }

    /// One step of the work that waits, under the writer's guard.
    fn sweep(&self, predictor: &mut PrefixPredictor) {
        if self.sweeping.load(Ordering::Relaxed) && predictor.sweep(Self::SWEEP) < Self::SWEEP {
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
            if let Some(mut predictor) = self.predictor.try_write() {
                self.sweep(&mut predictor);
            }
        }
        true
    }

    /// Records the request of `keys` as sent to `target`, behind the naming
    /// that read `read`. Not for a repeat ([`Self::repeats`]). `held` says
    /// the naming found the target among the workers sent the request's
    /// deepest recorded block.
    ///
    /// The write begins behind the blocks the target already holds: a turn
    /// of a conversation writes the blocks the turn added. It takes the
    /// lock sixteen blocks a turn, each turn by a try. An entry is a
    /// prediction and a leading part of a request is a true one, so a turn
    /// that meets a held lock ends the write there and waits for no thread:
    /// the next request of the prefix writes on from where this one
    /// stopped. A worker that left since the naming stops the write too:
    /// the record gives no place to a worker told gone.
    fn write(&self, keys: &[u64], read: RecordRead, held: bool, target: WorkerId) {
        if keys.is_empty() {
            return;
        }
        // The target holds the naming's depth where the naming found it
        // there, and nothing the naming knows of otherwise.
        let known = if held { read.blocks as usize } else { 0 };
        let mut end = known.min(keys.len());
        // A request the target holds whole has its last use to move and
        // no block to add: left undone, it leaves nothing out.
        let adds = end < keys.len();
        loop {
            end = (end + Self::SLICE).min(keys.len());
            let Some(mut predictor) = self.predictor.try_write() else {
                if adds {
                    self.left_out.fetch_add(1, Ordering::Relaxed);
                }
                return;
            };
            if self.left.load(Ordering::Relaxed) != read.left {
                return;
            }
            predictor.record(&keys[..end], target, read.tick);
            if end == keys.len() {
                self.sweep(&mut predictor);
                return;
            }
            RwLockWriteGuard::unlock_fair(predictor);
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
    /// The place's own profile reads affinity
    /// ([`Engine::profile_reads_affinity`]): a pick states its worker's
    /// entry. Where the binding's shadow alone reads it, the rows are the
    /// shadow's and the live pick states nothing.
    ///
    /// [`Engine::profile_reads_affinity`]: crate::engine::Engine::profile_reads_affinity
    states: bool,
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
    record: Option<Arc<PrefixRecord>>,
    /// The record's holders are named ahead of the index's: no prefix stage
    /// of the profile reads cache events.
    ahead: bool,
    /// The host's token tree for the place, where it lends one, the
    /// profile reads a record and the model has no index of cache events: a
    /// request whose host names its entry is decided inside the tree's
    /// match. The handle is the host's own tree, kept while the table
    /// stands.
    sent: Option<Arc<TokenTree>>,
    /// The document's block size, in tokens: what a tree's match is
    /// counted by for a request its host counted with none
    /// ([`Self::blocks_of`]).
    block: u32,
    /// The list's workers by the fingerprint of their address, each with
    /// its row: where a worker a tree names stands. Empty where the host
    /// lends the list no tree.
    by_address: Box<[(u64, u32)]>,
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
            states: engine.profile_reads_affinity(model, phase),
            column: Box::default(),
            index: None,
            row_of: Box::default(),
            within: Box::default(),
            heard: Box::default(),
            heard_any: false,
            unheard_any: false,
            record: None,
            ahead: false,
            sent: None,
            block: spec.block_size(),
            by_address: Box::default(),
            by_id: Box::default(),
            warming: Box::default(),
            ruled: WarmUpState::default(),
            first_end: u64::MAX,
        };
        // A list of no worker names no holder and writes no prefix, and its
        // model would be the request's own word: nothing is read for it.
        let Some(first) = workers.first().filter(|_| reads) else {
            return table;
        };
        // The index is its workers' model's, by the id they registered
        // under: a request may name the model by another.
        let registered = first.model_id();
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
            table.record = Some(affinity.record_of(registered, bound));
            table.ahead = ahead;
            if table.index.is_none() && cap != 0 {
                let lent = affinity.indexes.get();
                table.sent = lent.and_then(|indexes| indexes.token_tree(registered, phase));
            }
            if table.sent.is_some() {
                let print = |worker: &Arc<dyn Worker>| fingerprint(worker.url().as_bytes());
                let mut by_address: Vec<(u64, u32)> =
                    workers.iter().map(print).zip(0u32..).collect();
                by_address.sort_unstable();
                table.by_address = by_address.into();
            }
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

    /// Calls `each` with the row of every worker of the list at `address`,
    /// the address a tree names a worker by: found by the address's
    /// fingerprint and held to the address itself.
    fn rows_at(&self, workers: &[Arc<dyn Worker>], address: &str, mut each: impl FnMut(u32)) {
        let print = fingerprint(address.as_bytes());
        let from = self.by_address.partition_point(|&(other, _)| other < print);
        for &(other, row) in self.by_address.iter().skip(from) {
            if other != print {
                break;
            }
            let worker = workers.get(row as usize);
            if worker.is_some_and(|worker| worker.url() == address) {
                each(row);
            }
        }
    }

    /// Names a request's rows into `buffers` from a tree's match: the
    /// workers of the list among `sent`, the tenants of the deepest node the
    /// request's prefix reaches, each holding its first `blocks` blocks, and
    /// the list's warming rows. Each list is at most the cap, ascending by
    /// row: past the cap the holders are a draw. A match under one block
    /// names no holder.
    fn name_sent(
        &self,
        sent: &[TenantId],
        blocks: u32,
        workers: &[Arc<dyn Worker>],
        facts: &RequestFacts<'_>,
        buffers: &mut Buffers,
    ) {
        let Buffers {
            holders,
            warming,
            picked,
            live,
            ..
        } = buffers;
        holders.clear();
        warming.clear();
        let mut draw = Draw::new(facts.seed ^ HOLDERS_STREAM);
        let mut seen = 0usize;
        for address in sent.iter().filter(|_| blocks > 0) {
            self.rows_at(workers, address, |row| {
                let holder = AffinityHolder {
                    row,
                    overlap_blocks: 0,
                    predicted_blocks: blocks,
                };
                seen += 1;
                if seen <= self.cap {
                    holders.push(holder);
                } else if let Some(slot) = holders.get_mut(below(&mut draw, count(seen)) as usize) {
                    // Past the cap each worker is as likely to stand as
                    // another.
                    *slot = holder;
                }
            });
        }
        holders.sort_unstable_by_key(|holder| holder.row);
        self.name_warming(facts, warming, picked, live);
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
        let record = self
            .record
            .as_deref()
            .filter(|_| hashed && self.unheard_any);
        let mut read = None;
        if self.ahead {
            read = record
                .map(|record| self.name_recorded(record, facts, keys, deepest, holders, &mut draw));
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
            read = record
                .map(|record| self.name_recorded(record, facts, keys, deepest, holders, &mut draw));
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
    fn read_record(
        &self,
        record: &PrefixRecord,
        facts: &RequestFacts<'_>,
        keys: &mut Vec<u64>,
        deepest: &mut Vec<WorkerId>,
    ) -> (u32, RecordRead) {
        PrefixPredictor::fold_into(keys, facts.namespace_seed, facts.block_hashes);
        let tick = tick_of(facts.now_micros);
        let usable = |id: WorkerId| self.spoken_row(id).is_some();
        let predictor = record.predictor.read();
        let found = predictor.deepest_among(keys, &usable);
        deepest.clear();
        deepest.extend(found.holders());
        let read = RecordRead {
            tick,
            blocks: found.blocks(),
            whole: found.blocks() as usize == keys.len(),
            fresh: found.fresh(tick),
            left: record.left.load(Ordering::Relaxed),
        };
        (found.blocks(), read)
    }

    /// The record's holders of the request, behind the holders already
    /// named and in the room the cap leaves: the workers sent the deepest
    /// block of it that a worker the record speaks for was sent, a draw of
    /// them where they would fill more than the room.
    fn name_recorded(
        &self,
        record: &PrefixRecord,
        facts: &RequestFacts<'_>,
        keys: &mut Vec<u64>,
        deepest: &mut Vec<WorkerId>,
        holders: &mut Vec<AffinityHolder>,
        draw: &mut Draw,
    ) -> RecordRead {
        let (blocks, read) = self.read_record(record, facts, keys, deepest);
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
    /// In place in both namings: a list with none costs a test, not a call.
    #[inline(always)]
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
    ) -> Option<PrefixRead> {
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
        let record = self.record.as_deref().filter(|_| self.unheard_any)?;
        let (blocks, found) = self.read_record(record, facts, keys, deepest);
        let mut read = PrefixRead {
            read: found,
            deepest: [NONE; PrefixRead::KEPT],
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
    // The key's own call hands out the pointer alone: what a decision
    // answers through that call is copied on its way out of it.
    let buffers = BUFFERS.with(Rc::clone);
    let Ok(mut buffers) = buffers.try_borrow_mut() else {
        // A stage asked for a decision from inside this one, on this
        // thread: that decision takes buffers of its own.
        return with(&mut Buffers::new());
    };
    with(&mut buffers)
}

thread_local! {
    /// The calling thread's buffers: a naming allocates nothing once they
    /// have grown to a request's blocks and the profile's cap.
    static BUFFERS: Rc<RefCell<Buffers>> = Rc::new(RefCell::new(Buffers::new()));
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
/// prefixes speaks for the target the request goes to, the first one its
/// caller sends to (`sends_to`), the request's prefix is written for it:
/// the record says where a prefix was sent.
///
/// `workers` is the list the decision is made over and `pool` what the
/// engine keeps beside that list. `ids` holds the workers' ids by their
/// place in the list, and a pick's first target names its worker by that
/// place and that id. `facts` carries the request's block hashes and the
/// decision's instant ([`RequestFacts::now_micros`]) on the one clock of the
/// engine's decisions.
///
/// A request whose host filled a column or named rows itself passes straight
/// to `decide`. So does every request of a phase none of whose bindings
/// reads affinity ([`Engine::needs_for_phase`]): that is one test, and the
/// list keeps no table for it. Under a document where another binding of
/// the phase reads affinity, a profile that reads none passes on behind its
/// list's table, which says so. A profile that reads affinity and is named
/// no row (it shows its rows to a plugin, whose stage reads the column
/// whole) is handed the column [`column()`] fills, in
/// [`RequestFacts::affinity`].
///
/// `token_key` names the request's entry in the host's token tree of sent
/// prefixes ([`TokenKey`]), given the tree's page in tokens. It is called
/// at most once, and only where the host lends the place such a tree
/// ([`CacheIndexes::token_tree`]): its profile reads a record of sent
/// prefixes and is named rows, and the model has no index of cache events.
/// A request it names an entry for is decided inside the tree's match.
/// `decide` is called once either way.
///
/// `sends_to` says whether the request's caller sends it to a target of the
/// pick ([`RequestMaker::sends_to`]). A decision made inside the tree's
/// match, and one the record is written behind, asks it of the pick's
/// targets, best first: the tree stores the request's pages, and the record
/// its prefix, for the first one it takes. No other decision asks it.
///
/// A pick states what its first target holds of the request ([`Overlap`]),
/// from the rows its decision was given: the rows named here, the column
/// filled here, a column its host filled or the rows its host named. A
/// request decided inside a tree's match states nothing: a match is a count
/// of tokens and no entry of blocks. Neither does a decision whose own
/// profile reads no affinity, whether it is given no rows or the rows of
/// its binding's shadow.
///
/// The rows live in the calling thread's buffers until this returns:
/// `decide` reads them and keeps nothing of them. Nothing is allocated once
/// the buffers have grown, but where a table is read (once a list, and then
/// by one decision a second) and by a tree's own match and write.
///
/// [`Engine::needs_for_phase`]: crate::engine::Engine::needs_for_phase
/// [`RequestMaker::sends_to`]: crate::engine::RequestMaker::sends_to
#[expect(
    clippy::too_many_arguments,
    reason = "the list, what is kept beside it, the request's facts, its entry in a tree and the target its caller takes, and the decision"
)]
pub fn decide<'p>(
    affinity: &Affinity,
    pool: &PoolAffinity,
    spec: &AppliedSpec,
    workers: &[Arc<dyn Worker>],
    ids: &[WorkerId],
    facts: &mut RequestFacts<'_>,
    token_key: impl FnOnce(usize) -> Option<TokenKey<'p>>,
    sends_to: impl Fn(&Target) -> bool,
    mut decide: impl FnMut(&mut RequestFacts<'_>) -> Decision,
) -> Decision {
    // No binding of the phase reads affinity: one test, and no table.
    let unread = !spec.engine().needs_for_phase(facts.phase).affinity;
    debug_assert!(
        !unread || !spec.engine().needs_for(facts.model, facts.phase).affinity,
        "a phase's needs hold the needs of each of its decisions"
    );
    if unread {
        return decide(facts);
    }
    if !facts.affinity.is_empty() || facts.named.is_some() {
        // Its host fed the decision itself. The pick states what that feed
        // says, where the request's own profile reads it.
        let mut decision = decide(facts);
        if spec
            .engine()
            .profile_reads_affinity(facts.model, facts.phase)
        {
            state(&mut decision, facts.feed(), facts, spec.block_size());
        }
        return decision;
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
    with_buffers(|buffers| {
        pool.with_table(&asked, |table| {
            table.decide(workers, facts, buffers, token_key, sends_to, decide)
        })
    })
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
/// speaks for a worker of the list: the caller hands it to [`record_prefix`] behind
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
) -> Option<PrefixRead> {
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

/// States in a decision's pick what its first target holds of the request
/// ([`Overlap`]): that worker's entry of `feed`, the rows the decision was
/// given. They are the rows named or the column filled here, a column the
/// request's host filled, or the rows that host named
/// ([`RequestFacts::feed`]): whoever named them, the entry is read the same
/// way, and no stage is asked. A worker the rows do not name holds nothing,
/// as it did to the stages: the pick then states a miss of the request's
/// blocks. `facts` are the request's own, which say its size, and `block`
/// is the document's block size, by which the request is counted where its
/// host counted with none.
///
/// The feed is handed by itself, and not the facts the decision was made
/// over: a decision that reads no fact, as a pin's answer handed through
/// here, then makes no copy of them for this statement.
///
/// Nothing is stated where the decision picked nobody, and for a pick the
/// binding's sessions rule placed alone and keeps from the profile
/// ([`told_target`]): such a request is none of the profile's, as a
/// gateway's sticky map hands a request it places to no policy. The one
/// place a pick's overlap is stated. Its callers ask it where the
/// request's own profile reads affinity, and not for a shadow's rows.
fn state(decision: &mut Decision, feed: Feed<'_>, facts: &RequestFacts<'_>, block: u32) {
    let Some(row) = told_target(decision).map(|target| target.position) else {
        return;
    };
    let held = feed.entry_or_default(row);
    if let Decision::Pick(pick) = decision {
        pick.targets.state(Overlap::held(held, facts.blocks(block)));
    }
}

/// Whether the profile is told of a decision's pick: the pick's first
/// target where it is, and none for a decision that picked nobody and for
/// a pick the binding's sessions rule placed alone and keeps from the
/// profile ([`lease_events::stages_hear`]). The one test in front of every
/// write that keeps where a prompt went. A single's writes name the worker
/// the request goes to ([`sent_target`]), which may be a later target.
///
/// The rule's word is read through that one function at five places: the
/// words of a single's lease, the record's two writes here, the store of a
/// prompt in a lent token tree ([`sent_address`]), the hand-over of a pin
/// the keyed entry followed to this lookup, and a pair's two writes behind
/// its pick, a side at a time. Under the rule's default word a request it
/// places is the profile's own, and its prompt is kept as any pick's. Under
/// the other the rule keeps the request from the profile, as a gateway's
/// sticky map hands a request it places to no policy: the policy's record
/// never sees that request's prompt.
pub(super) fn told_target(decision: &Decision) -> Option<&Target> {
    if !lease_events::stages_hear(decision) {
        return None;
    }
    decision.pick()?.targets.first()
}

/// Behind a pick made over a column ([`column()`]): records the request's
/// prefix as sent to the picked worker `id`, where the record of sent
/// prefixes speaks for that worker. The record says where a prefix was
/// sent, and this write has no `sends_to` to ask: `id` is the worker its
/// caller says the request goes to.
///
/// `pool`, `phase` and `facts` are the ones the column was filled with, and
/// `read` is what that fill returned. A worker the index speaks for, a
/// worker that is not of the list, and a list whose table was since read
/// for another place write nothing.
pub fn record_prefix(
    pool: &PoolAffinity,
    phase: Phase,
    facts: &RequestFacts<'_>,
    read: PrefixRead,
    id: WorkerId,
) {
    let standing = pool.table.load();
    let table = standing.as_deref();
    let Some(table) = table.filter(|table| table.is_of(facts.model, phase)) else {
        return;
    };
    let (Some(record), Some(row)) = (table.record.as_deref(), table.spoken_row(id)) else {
        return;
    };
    let held = read.deepest.contains(&row);
    if record.repeats(read.read, held) {
        return;
    }
    with_buffers(|buffers| {
        // The keys are the request's own, whichever list was filled last.
        PrefixPredictor::fold_into(&mut buffers.keys, facts.namespace_seed, facts.block_hashes);
        record.write(&buffers.keys, read.read, held, id);
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

/// The target a decision's request goes to: the first target of a pick the
/// profile is told of ([`told_target`]) that the request's caller sends to
/// (`sends_to`), asked best first. It is the worker a tree's match, the
/// record's two writes and a pair's two writes behind its first pick keep
/// the request's prefix for. `None` where the decision picked nobody, the
/// binding's sessions rule keeps the request from the profile, or the
/// caller sends to none of the pick's targets: nothing is stored.
pub(super) fn sent_target(
    decision: &Decision,
    sends_to: impl Fn(&Target) -> bool,
) -> Option<&Target> {
    told_target(decision)?;
    let mut targets = decision.pick()?.targets.iter();
    targets.find(|target| sends_to(target))
}

/// The address of the worker a decision's request goes to
/// ([`sent_target`]), in the list it was made over.
fn sent_address<'w>(
    workers: &'w [Arc<dyn Worker>],
    decision: &Decision,
    sends_to: impl Fn(&Target) -> bool,
) -> Option<&'w str> {
    let target = sent_target(decision, sends_to)?;
    Some(workers.get(target.position as usize)?.url())
}

impl Table {
    /// One decision by this table: the naming, the decision, and the
    /// record's write behind a pick. A request whose host names its entry
    /// in the token tree it lends is decided inside the tree's match
    /// ([`Self::decide_in_tree`]).
    fn decide<'p>(
        &self,
        workers: &[Arc<dyn Worker>],
        facts: &mut RequestFacts<'_>,
        buffers: &mut Buffers,
        token_key: impl FnOnce(usize) -> Option<TokenKey<'p>>,
        sends_to: impl Fn(&Target) -> bool,
        mut decide: impl FnMut(&mut RequestFacts<'_>) -> Decision,
    ) -> Decision {
        if !self.reads {
            return decide(facts);
        }
        if self.cap == 0 {
            return self.decide_by_column(facts, buffers, sends_to, decide);
        }
        // The entry is asked for where the host lends the place a tree,
        // once, by the tree's own page.
        let sent = self.sent.as_deref();
        let keyed = sent.and_then(|sent| Some((sent, token_key(sent.page_size())?)));
        if let Some((sent, key)) = keyed {
            return self.decide_in_tree((sent, &key), sends_to, workers, facts, buffers, decide);
        }
        let read = self.name(facts, buffers);
        let named = AffinityRows {
            holders: &buffers.holders,
            warming: &buffers.warming,
            thin: self.ruled.thin,
            mean_in_flight: None,
        };
        let mut decision = decide(&mut RequestFacts {
            named: Some(named),
            ..*facts
        });
        if self.states {
            state(&mut decision, Feed::Named(named), facts, self.block);
        }
        let (Some(read), Some(record)) = (read, self.record.as_deref()) else {
            return decision;
        };
        // The index speaks for a worker whose cache events are heard: the
        // record takes no entry for it.
        let sent = sent_target(&decision, sends_to);
        let Some(target) = sent.filter(|to| !self.heard(to.position)) else {
            return decision;
        };
        let held = buffers.deepest.contains(&target.id);
        if !record.repeats(read, held) {
            record.write(&buffers.keys, read, held, target.id);
        }
        decision
    }

    /// How many of the request's blocks a token tree's match of `tokens`
    /// tokens holds, counted as the stages count the request
    /// ([`blocks_within`]): by the size its host counted it with, and by the
    /// document's where the host counted with none. The tree's page is no
    /// unit of the request: a match is a count of tokens, whatever the page
    /// the tree walked them by.
    fn blocks_of(&self, tokens: usize, facts: &RequestFacts<'_>) -> u32 {
        let block = facts.block_size_or(self.block);
        blocks_within(count(tokens), facts.hash_kind, block)
    }

    /// One decision of a request inside the match of the host's token tree
    /// for its place, by the tree and the request's entry in it. The match
    /// names the workers sent the deepest prefix of the request, the
    /// decision is made over them, and the same descent writes the
    /// request's whole pages for the worker the request goes to: the first
    /// target of the pick its caller sends to (`sends_to`). A decision that
    /// picks nobody, or none of whose targets its caller sends to, stores
    /// nothing: the match has marked the path it walked as used, as every
    /// match of the tree does. The record of hashed prefixes is neither
    /// read nor written.
    fn decide_in_tree(
        &self,
        (sent, key): (&TokenTree, &TokenKey<'_>),
        sends_to: impl Fn(&Target) -> bool,
        workers: &[Arc<dyn Worker>],
        facts: &RequestFacts<'_>,
        buffers: &mut Buffers,
        mut decide: impl FnMut(&mut RequestFacts<'_>) -> Decision,
    ) -> Decision {
        let mut decide_over = |holders: &[TenantId], blocks: u32| {
            self.name_sent(holders, blocks, workers, facts, buffers);
            decide(&mut RequestFacts {
                named: Some(AffinityRows {
                    holders: &buffers.holders,
                    warming: &buffers.warming,
                    thin: self.ruled.thin,
                    mean_in_flight: None,
                }),
                ..*facts
            })
        };
        let mut decided = None;
        sent.match_and_insert_with(&key.tokens, |found| {
            // The marker is no part of the prompt.
            let matched = found.matched_token_count.saturating_sub(key.marker);
            let decision = decide_over(&found.matched_tenants, self.blocks_of(matched, facts));
            let to = sent_address(workers, &decision, &sends_to);
            decided = Some(decision);
            to
        });
        // The tree asks for the worker once, so the decision was made
        // there. Were it not asked, the request is decided here, with no
        // holder named and nothing written.
        decided.unwrap_or_else(|| decide_over(&[], 0))
    }

    /// One decision of a profile that reads the column whole: the fill, the
    /// decision over the column, and the record's write behind a pick.
    fn decide_by_column(
        &self,
        facts: &RequestFacts<'_>,
        buffers: &mut Buffers,
        sends_to: impl Fn(&Target) -> bool,
        decide: impl FnOnce(&mut RequestFacts<'_>) -> Decision,
    ) -> Decision {
        // The column leaves the buffers while the fill writes the others.
        let mut column = mem::take(&mut buffers.column);
        let read = self.fill(facts, buffers, &mut column);
        let mut decision = decide(&mut RequestFacts {
            affinity: &column,
            ..*facts
        });
        if self.states {
            state(&mut decision, Feed::Column(&column), facts, self.block);
        }
        buffers.column = column;
        let (Some(read), Some(record)) = (read, self.record.as_deref()) else {
            return decision;
        };
        let sent = sent_target(&decision, sends_to);
        let Some(target) = sent.filter(|to| !self.heard(to.position)) else {
            return decision;
        };
        let held = read.deepest.contains(&target.position);
        if !record.repeats(read.read, held) {
            record.write(&buffers.keys, read.read, held, target.id);
        }
        decision
    }
}

#[cfg(test)]
mod tests;
