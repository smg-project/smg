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
//! the rows' place ([`AffinityFacts`]): the engine's own pair decision fills
//! one from the same data and writes the record behind the pair's pick
//! (`column`, `record_by_column`).
//!
//! Behind a single decision the pick states what its first target holds of
//! the request ([`Overlap`]): the worker's entry of the rows the decision was
//! given, whoever named them. No stage is asked for it. It is stated where
//! the request's own profile reads affinity: rows named for a binding's
//! shadow alone state nothing of the live pick. A pair's first decision
//! states the same of its prefill, the first target's worker, from the
//! prefills' column, and nothing of its decode (`state`).
//!
//! A request whose host names its entry in a token tree ([`TokenKey`]) is
//! decided another way where the host lends the tree of the request's place
//! ([`CacheIndexes::token_tree`]) and the model has no index of cache
//! events that holds a block: inside the tree's match ([`TokenTree`]). The
//! match names the workers sent the deepest prefix of the request, and the
//! same descent writes the request for the worker picked. The pick states
//! its first target's entry of the rows the match named, as every other
//! pick states its rows'. The tree is the host's own, the one its policy
//! reads and writes: the engine keeps none. A request its host holds as
//! text is decided the same way inside the host's string tree
//! ([`CacheIndexes::string_tree`], [`TextKey`]), whether or not the model
//! has an index: an index speaks of token blocks.
//!
//! What is kept, and what guards it:
//!
//! - **Once a list of workers** ([`PoolAffinity`]): each worker's number in
//!   the model's index, what the index held for it and when it was
//!   admitted, and the list's warming rows by the profile's own ruling
//!   ([`WarmUpRule`]), none while the model's index holds no block. It is
//!   a ruling read from the index and the worker
//!   objects, never a copy of a worker's health or load. One decision reads
//!   it again once its second is up, once the index handed one of its
//!   numbers on to another worker, or once the index's word of holding a
//!   block moved, behind one atomic flag, and swaps it in by pointer. A
//!   decision loads the pointer and writes no shared word.
//! - **Once a model** ([`Affinity`]): the record of the prefixes sent to the
//!   workers the index does not speak for, and to every worker by a request
//!   hashed by its text or decided by a profile that reads sent prefixes
//!   alone ([`PrefixPredictor`]), behind one reader-writer
//!   lock. A naming takes it as a reader, and only where the record speaks
//!   for a worker of the list on its request. A pick writes behind the
//!   blocks its naming found its worker to hold, sixteen blocks a turn, each
//!   turn by a try: a turn that meets a held lock ends the write there, so a
//!   pick waits for nobody. Its thread keeps the rest, reads it at its own
//!   namings and writes it on behind its next picks. The map of the records
//!   is behind a mutex that only the read of a list's table, a change of
//!   membership and the timer's sweep take.
//! - **Once a thread**: the buffers a naming writes into, and the writes
//!   of a record that met a held lock, eight at most. A decision allocates
//!   nothing once they have grown, but what a tree's own match and write
//!   allocate, and the list of a prompt that is hashed on a road that reads
//!   it.
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
    Tree as StringTree,
};
use parking_lot::{Mutex, RwLock, RwLockWriteGuard};
use rustc_hash::{FxHashMap, FxHashSet};
use smg_worker::identity::basic::Worker;

use crate::{
    document::spec::{AppliedSpec, IndexKind},
    engine::{
        decision::{Decision, Overlap, Target},
        decision_engine::{TextKey, TokenKey},
        lease_events,
        prompt_hashes::PromptHashes,
        view::{
            blocks_within, fingerprint, AffinityFacts, AffinityHolder, AffinityRows, Feed,
            HashKind, Phase, RequestFacts, TreeMatch, WorkerId,
        },
        warm_up::WarmUpState,
    },
    tracking::{lease::Side, prefix_predictor::PrefixPredictor},
    Draw,
};

/// No row, and no number in the index.
const NONE: u32 = u32::MAX;

/// How long a list's table stands before one decision reads it again, in
/// microseconds. One whose numbers the index handed on, or whose index's
/// word of holding a block moved, is read again before that
/// ([`Table::stands_by_index`]).
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
    /// The index's own number of the worker at `address`. `None` while the
    /// index holds no place for the worker. A host makes the place where
    /// the worker's stream of cache events is opened, or where the worker's
    /// record first asks for those events: in either case ahead of the
    /// worker's first block, so a number says nothing of what was heard.
    fn worker_number(&self, address: &str) -> Option<u32>;

    /// How many times the index gave a number one worker handed back to the
    /// next worker that asked for one. A list's table reads it ahead of the
    /// numbers and is read again once it moved, and a lookup it moved under
    /// names nobody: a number that changed hands never names its former
    /// worker. The default is an index that hands no number out twice.
    fn numbers_reused(&self) -> u64 {
        0
    }

    /// Whether the index holds a block for any worker. A list's table reads
    /// it ahead of the numbers, reads an index that holds none as no index,
    /// and is read again once the word moved: a decision asks it where it
    /// takes the list's table, so the answer must cost as little as a
    /// count's read. The default is an index that holds one, which is the
    /// word of a host that hands no index while its index is empty
    /// ([`CacheIndexes::index`]).
    fn holds_a_block(&self) -> bool {
        true
    }

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

    fn numbers_reused(&self) -> u64 {
        self.ids_reused()
    }

    /// The root's own test, in constant time.
    fn holds_a_block(&self) -> bool {
        !self.is_empty()
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
///
/// Built under `test-util` alone: the package's own cases turn the feature
/// on, and so does a host for its cases over a reference index of its own.
#[cfg(feature = "test-util")]
#[doc(hidden)]
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
    /// `None` where no worker of the model streams cache events. An index
    /// that holds no block is read as none: a request is then decided by
    /// the host's token tree, where it lends one ([`Self::token_tree`]).
    /// A host says so in one of two ways. It hands the index whenever the
    /// model has one, and the index answers whether it holds a block
    /// ([`CacheIndex::holds_a_block`]): a list's table is read again by
    /// the first decision that finds the word moved, so a model's first
    /// block and its last block's leaving decide from that table on. Or
    /// it hands none while the index holds no block and keeps the word's
    /// default: the engine then learns of either with the list's next
    /// table, up to a second later. An index that holds a block decides a
    /// single request of token ids alone where a prefix stage of its
    /// profile reads cache events: no record of sent prefixes is read or
    /// written for it.
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
    /// reads, and [`Self::string_tree`] the one a request of text reads.
    fn token_tree(&self, model: &str, phase: Phase) -> Option<Arc<TokenTree>> {
        let _ = (model, phase);
        None
    }

    /// The host's own string tree of the prompts sent as text to the
    /// workers that serve `model` at `phase`: the tree its policy reads and
    /// writes for a request that holds no token ids, with a worker a tenant
    /// by its address. A request hashed by its text whose host names its
    /// entry ([`TextKey`]) is decided inside this tree's match, as a
    /// tokenized one is inside the token tree's, and the host keeps the
    /// tree the same way. It is asked for whatever [`Self::index`] answers:
    /// an index of cache events speaks of token blocks, and says nothing of
    /// a request of text. `None`, the default, where the host keeps none:
    /// the request is named its rows from the record of hashed windows of
    /// its text, which speaks for every worker of its list.
    fn string_tree(&self, model: &str, phase: Phase) -> Option<Arc<StringTree>> {
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
    /// The record of sent prefixes of each model, by the id a list's first
    /// worker lists the model under. Locked where a list's table is read, at a
    /// change of membership and at the timer's sweep: never by a decision's naming
    /// or its write, which hold their record by the list's table. A write
    /// a thread keeps holds its record itself ([`Waiting`]).
    records: Mutex<FxHashMap<Box<str>, Arc<PrefixRecord>>>,
    /// How many times a table took a record of the map
    /// ([`Self::record_of`]). Written under the map's mutex.
    taken: AtomicU64,
    /// What the records that left the map had left out
    /// ([`Self::left_out`]). Written under the map's mutex.
    left_out: AtomicU64,
    /// Where the next record made begins its count of leaves: past the
    /// count of every record that left the map, so a write reads the count
    /// its naming read in no record made since. Written under the map's
    /// mutex.
    left_past: AtomicU64,
    /// How many leaves were told ([`Self::leaves`]), each counted ahead of
    /// its records' own counts.
    leaves: AtomicU64,
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

    /// How many leaves the records were told ([`Self::forget`],
    /// [`Self::keep`]): read ahead of an answer that [`decide_behind`] is
    /// handed.
    pub(super) fn leaves(&self) -> u64 {
        self.leaves.load(Ordering::Relaxed)
    }

    /// A worker left the pool, was replaced, or joins with no leave told
    /// ahead: every record of sent prefixes forgets it. Takes the records'
    /// map and, for a record that holds the worker, that record's lock as
    /// a writer.
    pub fn forget(&self, worker: WorkerId) {
        self.leaves.fetch_add(1, Ordering::Relaxed);
        for record in self.records.lock().values() {
            record.forget(worker);
        }
    }

    /// The workers that stand, said whole behind a read of the host's
    /// registry: every record forgets each worker it holds a place for
    /// that is not among them, so a worker whose leave was never told
    /// keeps no place. Takes what [`Self::forget`] takes. A store that
    /// took its membership whole says them too.
    pub(super) fn keep(&self, standing: &[WorkerId]) {
        self.leaves.fetch_add(1, Ordering::Relaxed);
        let standing: FxHashSet<WorkerId> = standing.iter().copied().collect();
        for record in self.records.lock().values() {
            record.keep(&standing);
        }
    }

    /// The records no list may read again, for the follow of a host's
    /// registry: each one the map alone holds that holds a worker's place,
    /// under a model `stated` says none of the follow's workers states as
    /// its own, with the number of its last taking. A list keeps its
    /// record under a model its first worker lists (the id of the card
    /// that answers to the request's name, else its first model), so no
    /// list reads a record whose model no worker that stands lists. The
    /// follow asks its registry of each ahead of [`Self::let_go`].
    pub(super) fn unstated(&self, stated: impl Fn(&str) -> bool) -> Vec<(Box<str>, u64)> {
        let mut records = self.records.lock();
        let unstated = records.iter_mut().filter_map(|(model, record)| {
            let record = Arc::get_mut(record)?;
            record.predictor.get_mut().held().next()?;
            (!stated(model)).then(|| (model.clone(), *record.taken.get_mut()))
        });
        unstated.collect()
    }

    /// The records named by [`Self::unstated`] leave the map with their
    /// places: each one the map alone still holds that no table took since
    /// it was named. The model's next table read makes a fresh record, as
    /// behind a sweep ([`Self::sweep`]), and what a record left out stays
    /// in the total ([`Self::left_out`]).
    pub(super) fn let_go(&self, unstated: &[(Box<str>, u64)]) {
        let mut records = self.records.lock();
        let left: Vec<_> = unstated
            .iter()
            .filter_map(|(model, taken)| {
                let record = Arc::get_mut(records.get_mut(model)?)?;
                (*record.taken.get_mut() == *taken).then(|| records.remove(model))?
            })
            .collect();
        self.count_gone(left.iter());
        // Freed once the mutex is given back, as at a sweep.
        drop(records);
    }

    /// Counts the records that left the map, under its mutex: what they
    /// had left out stays in the total ([`Self::left_out`]), and the next
    /// record made begins its count of leaves past theirs
    /// ([`Self::left_past`]).
    fn count_gone<'a>(&self, gone: impl Iterator<Item = &'a Arc<PrefixRecord>>) {
        let (mut left_out, mut past) = (0, None);
        for record in gone {
            left_out += record.left_out.load(Ordering::Relaxed);
            past = past.max(Some(record.left.load(Ordering::Relaxed)));
        }
        self.left_out.fetch_add(left_out, Ordering::Relaxed);
        if let Some(past) = past {
            self.left_past
                .fetch_max(past.saturating_add(1), Ordering::Relaxed);
        }
    }

    /// The work the records left waiting, a bounded step of each: the
    /// placements of workers that left, and the blocks past a bound that
    /// came down. For the engine's timer. A record a decision holds is
    /// passed over.
    ///
    /// A record leaves the map here once the map alone holds it and it
    /// holds no worker's place: no list's table reads it, no thread keeps
    /// a write for it, and every worker it was written for was forgotten.
    /// So a model's record goes behind its last worker, and a record with
    /// no placement yet can go between two tables of its list. The model's
    /// next table read makes a fresh record, whose count of leaves begins
    /// past this one's: a pick named from a record that left writes
    /// nothing, as behind a leave. An entry is a prediction, and the next
    /// request of the prefix writes it.
    ///
    /// A write a thread keeps holds its record until the thread writes it
    /// on behind a pick, or until the thread ends. The record stays in the
    /// map that long and is stepped here like any other, so its placements
    /// of workers that left still go.
    pub fn sweep(&self) {
        let mut records = self.records.lock();
        // Nobody takes a record from the map but by a clone under this
        // mutex, so one the map alone holds is read by no table, written
        // by no pick and kept by no thread.
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
        self.count_gone(left.iter().map(|(_, record)| record));
        // What left is freed once the mutex is given back, so the read of a
        // list's table waits for no free.
        drop(records);
    }

    /// How many picks' prefixes the records left out in whole or in part,
    /// every model's together, those of the records that left the map among
    /// them: a write that meets a held record waits for nobody and is kept
    /// by its thread, and one the thread no longer keeps is left out. For a
    /// host's own count beside its count of decisions, never for a decision.
    pub fn left_out(&self) -> u64 {
        let records = self.records.lock();
        let left_out = |record: &Arc<PrefixRecord>| record.left_out.load(Ordering::Relaxed);
        self.left_out.load(Ordering::Relaxed) + records.values().map(left_out).sum::<u64>()
    }

    /// The workers each record holds a place for, by its model, for the
    /// cases of a host's membership.
    #[cfg(test)]
    pub(crate) fn held(&self) -> Vec<(Box<str>, Vec<WorkerId>)> {
        let records = self.records.lock();
        let held = |record: &Arc<PrefixRecord>| record.predictor.read().held().collect();
        records
            .iter()
            .map(|(model, record)| (model.clone(), held(record)))
            .collect()
    }

    /// The record of `model`, made at its first asking with its keys at
    /// `boundaries` ([`PrefixRecord::boundaries`]) and held to the
    /// placements `max_size` tokens make at `size` tokens a block, the size
    /// the list's first worker is hashed by. A record a list reads from
    /// `behind` its first worker's first model is read by lists of other
    /// first models and other sizes: it is held by the smallest size asked
    /// ([`PrefixRecord::hold`]). A record of the model that is keyed
    /// another way leaves the map as one let go ([`Self::let_go`]): keys
    /// at boundaries are no run of blocks, so neither kind reads what the
    /// other wrote. The tables and the writes that hold it keep it.
    fn record_of(
        &self,
        model: &str,
        max_size: u64,
        size: u32,
        behind: bool,
        boundaries: Option<&[u32]>,
    ) -> Arc<PrefixRecord> {
        let bound = |size: u32| {
            let bound = (max_size / u64::from(size.max(1))).max(1);
            usize::try_from(bound).unwrap_or(usize::MAX)
        };
        let (record, gone) = {
            let mut records = self.records.lock();
            let kept = records
                .get(model)
                .filter(|record| record.boundaries.as_deref() == boundaries)
                .map(Arc::clone);
            let (record, gone) = if let Some(record) = kept {
                (record, None)
            } else {
                let gone = records.remove(model);
                self.count_gone(gone.iter());
                let left = self.left_past.load(Ordering::Relaxed);
                let record = Arc::new(PrefixRecord::new(bound(size), left, boundaries));
                records.insert(model.into(), Arc::clone(&record));
                (record, gone)
            };
            // The number of this taking, which no other taking of the map
            // has: a record made anew under its model never reads as the
            // one that left.
            let taken = self.taken.fetch_add(1, Ordering::Relaxed) + 1;
            record.taken.store(taken, Ordering::Relaxed);
            (record, gone)
        };
        // Outside the map's mutex: a record that left is freed, as at a
        // sweep, and a bound that moved takes the record.
        drop(gone);
        record.hold(size, behind, bound);
        record
    }
}

/// One model's record of the prefixes sent to its workers the index does
/// not speak for, and to every worker by a request hashed by its text or
/// decided by a profile that reads sent prefixes alone
/// ([`Table::recorded`]), shared by every deciding thread.
struct PrefixRecord {
    predictor: RwLock<PrefixPredictor>,
    /// The boundaries the record's keys lie at, a hash index's
    /// ([`Table::fold_keys`]); none where they are one a block. A table
    /// keyed another way takes a record of its own
    /// ([`Affinity::record_of`]).
    boundaries: Option<Box<[u32]>>,
    /// The record's bound as the last table set it: a table's read takes
    /// the record, which the deciders hold, only where the bound moved.
    bound: AtomicUsize,
    /// The block sizes the record's lists asked it with, which hold its
    /// bound once a list read it from behind its first model. Such a
    /// record is read by lists hashed at different sizes: held by the
    /// size of each, its bound would follow which worker heads a list.
    /// One table's read at a time takes them, to size the record
    /// ([`Self::hold`]).
    sizes: Mutex<Sizes>,
    /// Work waits for a sweep: a worker left, or the bound came down.
    sweeping: AtomicBool,
    /// How many leaves were told, counted on from where the map's records
    /// before this one stopped ([`Affinity::left_past`]). Written under the
    /// record's guard, a reader's where the leave takes no place out: a
    /// write in parts stops where it moved since its naming.
    left: AtomicU64,
    /// How many writes ended before the request's last block: a turn of
    /// theirs met a held lock, and their thread kept them no longer
    /// ([`Waiting`]). Written for such a write alone, so a pick that
    /// records whole writes no shared word for it.
    left_out: AtomicU64,
    /// The number of the last taking of the record by a table, among
    /// every taking of the map ([`Affinity::record_of`]): written under
    /// the map's mutex and by a table's read alone. A record named as one
    /// no list reads again stays where a table took it since, and a
    /// record made anew under its model is not the one named
    /// ([`Affinity::let_go`]).
    taken: AtomicU64,
}

/// What a naming read of the record its pick may write.
#[derive(Clone, Copy, Debug)]
struct RecordRead {
    tick: u16,
    /// The leading keys of the request the record holds for a worker of
    /// the list: the depth its deepest holders were sent. A key is a block
    /// of the request, and under a hash index a boundary of the view the
    /// request reaches ([`Table::fold_keys`]).
    blocks: u32,
    /// Those keys in blocks of the request, as a stage counts a sent
    /// prefix: their count, and under a hash index the depth of the
    /// deepest boundary among them ([`Table::depth_of`]).
    depth: u32,
    /// The record holds every key of the request for a worker of the list.
    whole: bool,
    /// The request's last block was used at this tick.
    fresh: bool,
    /// [`PrefixRecord::left`] at the read.
    left: u64,
    /// The depth is a write's the reading thread keeps ([`Waiting`]), past
    /// what the record holds.
    waits: bool,
    /// The reader's word the record was read by ([`Table::recorded`]): it
    /// reads sent prefixes alone. The pick's write asks the rule by the
    /// same word, so a write follows its read.
    alone: bool,
}

/// What the fill of a column read of the record of sent prefixes
/// ([`column()`]). The write behind the pick made over that column stands on
/// it ([`record_by_column`]).
#[derive(Clone, Copy, Debug)]
pub(super) struct PrefixRead {
    read: RecordRead,
}

impl PrefixRead {
    /// Whether the fill found the worker of `entry`, an entry of the column
    /// it wrote, among those sent the request's deepest block: it wrote
    /// that depth into the entry of each of them, however many they are.
    fn found(self, entry: &AffinityFacts) -> bool {
        self.read.depth != 0 && entry.predicted_blocks == self.read.depth
    }
}

/// The record's tick at an instant of the decision clock: about a second.
const fn tick_of(now_micros: u64) -> u16 {
    (now_micros >> 20) as u16
}

/// The block sizes a record's lists asked it with ([`PrefixRecord::hold`]).
struct Sizes {
    /// The smallest size a list asked with since the record was made.
    smallest: u32,
    /// A list read the record from behind its first model.
    shared: bool,
}

impl PrefixRecord {
    /// A write gives the lock back after this many blocks.
    const SLICE: usize = 16;
    /// A sweep's step.
    const SWEEP: usize = 64;

    fn new(bound: usize, left: u64, boundaries: Option<&[u32]>) -> Self {
        Self {
            predictor: RwLock::new(PrefixPredictor::new(bound)),
            boundaries: boundaries.map(Into::into),
            bound: AtomicUsize::new(bound),
            sizes: Mutex::new(Sizes {
                smallest: u32::MAX,
                shared: false,
            }),
            sweeping: AtomicBool::new(false),
            left: AtomicU64::new(left),
            left_out: AtomicU64::new(0),
            taken: AtomicU64::new(0),
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

    /// Holds the record to the placements `bound` makes of the block size
    /// a list asks with: `size` while every list reads the record under
    /// its first model, as its bound always was; the smallest size asked
    /// since the record was made, by the lists ahead too, once a list read
    /// it from `behind` its first model. That size does not rise while the
    /// record lives. The size is picked and the bound set under one lock:
    /// a list that picked ahead of a smaller size sets no bound behind it.
    fn hold(&self, size: u32, behind: bool, bound: impl Fn(u32) -> usize) {
        let mut sizes = self.sizes.lock();
        sizes.smallest = sizes.smallest.min(size);
        sizes.shared |= behind;
        let size = if sizes.shared { sizes.smallest } else { size };
        self.bound_to(bound(size));
    }

    /// Counts the leave also where the record holds no place for `worker`:
    /// a pick named ahead of the leave would give the worker its first
    /// place, and its write stops where the count moved. The count moves
    /// under the guard that reads or clears the place, so no read pairs
    /// the count behind a leave with the place ahead of it.
    fn forget(&self, worker: WorkerId) {
        let held = self.predictor.read();
        if !held.holds(worker) {
            self.left.fetch_add(1, Ordering::Release);
            return;
        }
        drop(held);
        let mut predictor = self.predictor.write();
        self.left.fetch_add(1, Ordering::Release);
        predictor.forget(worker);
        self.sweeping.store(true, Ordering::Relaxed);
    }

    /// Forgets every worker the record holds a place for that is not in
    /// `standing` ([`Self::forget`]), and counts one leave ahead of them: a
    /// worker `standing` names no more may be under its first pick.
    fn keep(&self, standing: &FxHashSet<WorkerId>) {
        let held = self.predictor.read();
        self.left.fetch_add(1, Ordering::Release);
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

    /// Whether a pick behind `read` is a repeat, which records nothing of
    /// its request: by `read` the record holds the whole request for the
    /// picked worker, its end used at this tick. While that still stands
    /// a write would add no block and move no tick: it would make that
    /// end the newest of the record's list again, which can change the
    /// block that leaves next at the record's bound. A repeat gives that
    /// up. `held` says the naming found the picked worker among those
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

    /// Records the request of `keys` as sent to `target`, on from its first
    /// `done` blocks, behind a naming that read the record at `tick` with
    /// `left` leaves told. Not for a repeat ([`Self::repeats`]).
    ///
    /// It takes the lock sixteen blocks a turn, each turn by a try, and
    /// waits for no thread. A turn that meets a held lock ends the write
    /// there and returns how many leading blocks stand: the thread keeps
    /// the rest and writes on behind its next picks ([`Waiting`]). `None`
    /// where nothing is left to write: the request stands whole, or a
    /// worker left since the naming, and the record gives no place to a
    /// worker told gone. A turn that meets the lock held asks for such a
    /// leave too: a thread keeps no write that can write nothing.
    fn write(
        &self,
        keys: &[u64],
        done: usize,
        target: WorkerId,
        tick: u16,
        left: u64,
    ) -> Option<usize> {
        if keys.is_empty() {
            return None;
        }
        let mut done = done.min(keys.len());
        loop {
            // What the target holds as the turn begins is `done`: the
            // naming's depth or where a kept write stood, then the turns
            // before this one.
            let end = (done + Self::SLICE).min(keys.len());
            let Some(mut predictor) = self.predictor.try_write() else {
                // Asked where the lock is met held, and at no other turn: a
                // stale write handed back from here would be kept by its
                // thread, in a place of a write that can stand.
                return (self.left.load(Ordering::Relaxed) == left).then_some(done);
            };
            // Under the lock the count is the one the record stands by.
            if self.left.load(Ordering::Relaxed) != left {
                return None;
            }
            predictor.record_behind(&keys[..end], done, target, tick);
            if end == keys.len() {
                self.sweep(&mut predictor);
                return None;
            }
            done = end;
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
/// standing for a second, or until the index hands one of its numbers on
/// or its word of holding a block moves ([`Self::stands_by_index`]). It
/// holds what the index and the worker objects said at its read, as a
/// ruling: who the index speaks for, and who warms.
struct Table {
    /// The place the table answers for.
    model: Box<str>,
    phase: Phase,
    /// The decision clock at the read.
    read_at: u64,
    /// The instant the table stands until: its second, or sooner where its
    /// warming rows are thin ones that all hold nothing, at the end of the
    /// last of them. Behind it a ruling names the young rows this one left
    /// out for the thin, so the table is read again there.
    stands_until: u64,
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
    /// The model's index of cache events, where its host handed one. It is
    /// kept while it holds no block too: its word is asked again where a
    /// decision takes the table ([`Self::stands_by_index`]).
    index: Option<Arc<dyn CacheIndex>>,
    /// The index's word of holding a block ([`CacheIndex::holds_a_block`])
    /// as the table was built under it, read ahead of the count and the
    /// numbers below. An index that holds none is read as no index: no
    /// number, no worker heard, and the host's token tree where it lends
    /// one.
    holds: bool,
    /// Whether the index decides a single request of token ids by itself
    /// ([`Self::index_alone`]): said once, at the table's read.
    decides: bool,
    /// The index's count of numbers handed out a second time
    /// ([`CacheIndex::numbers_reused`]), read ahead of the numbers below:
    /// they are the index's while it reads the same ([`Self::numbered`]).
    reused: u64,
    /// Index number -> row of the list.
    row_of: Box<[u32]>,
    /// A bit an index number: a row of the list the index speaks for. The
    /// index's lookup is kept within these.
    within: Box<[u64]>,
    /// Row of the list -> its number in the index, `NONE` where it has
    /// none: the bits a request held to places is kept within
    /// ([`Self::mask`]).
    numbers: Box<[u32]>,
    /// A bit a row: the index speaks for the row on a request hashed by
    /// token blocks. The row holds a number in the index and a block or more
    /// there. The record of sent prefixes speaks for every other row, and
    /// for every row on a request hashed by its text or read by a profile
    /// that reads sent prefixes alone ([`Self::recorded`]). That is the
    /// answer for a pair's side, and for a single request where the table
    /// holds no index that holds a block: where the index decides a single
    /// request alone, the record speaks for no row ([`Self::index_alone`]).
    heard: Box<[u64]>,
    /// Whether the index speaks for any row, and whether the record does on
    /// a request hashed by token blocks that a profile reading cache events
    /// decides. The second is asked for a pair's side, and for a single
    /// request where the table holds no index that holds a block
    /// ([`Self::index_alone`]).
    heard_any: bool,
    unheard_any: bool,
    /// The model's record of sent prefixes, where the profile reads one.
    record: Option<Arc<PrefixRecord>>,
    /// The reader of a single decision over the place reads sent prefixes
    /// alone: no prefix stage of it reads cache events
    /// ([`Engine::pool_record_for`]). The record speaks to it for every row
    /// ([`Self::recorded`]), and its holders are named ahead of the
    /// index's. A single decision reads by this word. A pair's side reads
    /// by its own profile's ([`Self::side_alone`]), whatever this word is.
    /// For a single decision's reader that reads cache events too, the index
    /// decides a request of token ids alone ([`Self::index_alone`]).
    ///
    /// [`Engine::pool_record_for`]: crate::engine::Engine::pool_record_for
    alone: bool,
    /// The profile that decides the place's side of the model's pair reads
    /// sent prefixes alone ([`Engine::side_reads_sent_alone`]): the word a
    /// side's column is filled by ([`column()`]). It can differ from
    /// [`Self::alone`], which is the word of the profile a request of the
    /// phase asked alone resolves to. Read here once a table, as that word
    /// is: both are the document's the list's value was made under.
    ///
    /// [`Engine::side_reads_sent_alone`]: crate::engine::Engine::side_reads_sent_alone
    side_alone: bool,
    /// The view's boundaries, in tokens and ascending, where the model's
    /// index is of the kind hash: the record of sent prefixes then holds a
    /// request of token ids by one key a boundary it reaches and a request
    /// hashed by its text by none ([`Self::fold_keys`]), as the classic
    /// policy keys its placement map. `None` under a tree index, whose
    /// record holds a request by every block.
    boundaries: Option<Box<[u32]>>,
    /// The host's token tree for the place, where it lends one, the
    /// profile reads a record and the model has no index of cache events
    /// that holds a block ([`Self::holds`]): a request whose host names its
    /// entry is decided inside the tree's match. The handle is the host's
    /// own tree, kept while the table stands.
    sent: Option<Arc<TokenTree>>,
    /// The host's string tree for the place, where it lends one and the
    /// profile reads a record: a request hashed by its text whose host
    /// names its entry is decided inside the tree's match, under a model
    /// with an index of cache events too. The host's own tree, as
    /// [`Self::sent`] is.
    sent_text: Option<Arc<StringTree>>,
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
    /// those columns, asked while the index holds a block: under an index
    /// that holds none the table names no warming row.
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
            stands_until: now.saturating_add(TABLE_STANDS_MICROS),
            cap,
            reads,
            states: engine.profile_reads_affinity(model, phase),
            column: Box::default(),
            index: None,
            holds: false,
            decides: false,
            reused: 0,
            row_of: Box::default(),
            within: Box::default(),
            numbers: Box::default(),
            heard: Box::default(),
            heard_any: false,
            unheard_any: false,
            record: None,
            alone: false,
            side_alone: false,
            boundaries: None,
            sent: None,
            sent_text: None,
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
        // The word first, ahead of the count and the numbers: a first block
        // or a last one's leaving under this pass leaves the table a word
        // the index has moved past, and the next decision reads it again.
        table.holds = index.as_ref().is_some_and(|index| index.holds_a_block());
        table.decides = table.holds && phase == Phase::Single;
        // Ahead of the numbers: one that changes hands under this pass
        // leaves the table a count the index has moved past.
        table.reused = index.as_ref().map_or(0, |index| index.numbers_reused());
        // An index that holds no block is read as none: no worker is asked
        // for its number, as under a host that hands no such index.
        let heard_by = index.as_ref().filter(|_| table.holds);
        let rows = workers.len();
        let mut numbers = vec![NONE; rows];
        let mut sizes = vec![0u32; rows];
        let mut growth = vec![0u32; rows];
        let mut admitted = vec![0u64; rows];
        let mut emptied = vec![0u64; rows];
        let mut column = vec![AffinityFacts::default(); rows];
        let mut numbered = 0usize;
        for (row, worker) in workers.iter().enumerate() {
            let age = micros(worker.admitted_age());
            admitted[row] = now.saturating_sub(age);
            emptied[row] = admitted[row];
            column[row].admitted_age_ms = millis(age);
            let Some(index) = heard_by else {
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
            // A row that holds nothing is thin by the warm-up clock, which
            // starts again where the read of the growth sees an index
            // emptied underneath: read behind it, the clock shows an
            // emptying in this table. A young row is young by its admission,
            // and the column keeps the admitted age, which is all its record
            // holds.
            emptied[row] = now.saturating_sub(micros(worker.warmup_age()));
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
        table.numbers = numbers.into();
        table.heard = heard.into();
        // A warming row catches up to what the index holds: under an index
        // that holds no block a single request is named none, as the classic
        // policy calls its warm-up slice on its road of cache events alone.
        // A pair's side reads the list's column, which its stage rules by
        // age and size.
        if let Some(rule) = engine.pool_rule_for(model, phase).filter(|_| table.holds) {
            let (mut level, mut warming) = (Vec::new(), Vec::new());
            table.ruled = rule.list_warming_rows_emptied(
                &sizes,
                &growth,
                &admitted,
                &emptied,
                now,
                &mut level,
                &mut warming,
            );
            let ends = warming.iter().map(|&(_, until)| until);
            table.first_end = ends.clone().min().unwrap_or(u64::MAX);
            // A thin row that holds nothing ends with its window; one that
            // holds a block has no end.
            if table.ruled.thin {
                let last_end = ends.max().unwrap_or(u64::MAX);
                table.stands_until = table.stands_until.min(last_end);
            }
            table.warming = warming.into();
        }
        let record = engine.pool_record_for(model, phase);
        // The record is the model's the request names, and so are its
        // settings and the trees lent for it, by the id of the first
        // worker's card that answers to the name, an id ahead of an alias
        // as the registry reads a name: a list of a model holds workers of
        // different first models. A name no card answers to is the
        // caller's own word, and reads the first model's record.
        let mut named = None;
        first.any_model(&mut |card| {
            let id = card.id == model;
            if id || (named.is_none() && card.matches(model)) {
                named = Some(card.id.clone());
            }
            id
        });
        let named = named.as_deref().unwrap_or(registered);
        let settings = spec.index_for(named);
        // A hash index keeps what it placed and reads no tree: the record
        // is keyed by the request's head at the view's boundaries alone,
        // and its host is asked for no tree. Whom the record speaks for is
        // the reader's own word, as under a tree: where the model's index
        // of cache events holds a block it decides a single request of a
        // reader that reads cache events alone (`Table::index_alone`).
        let hash = settings.kind == IndexKind::Hash;
        if let Some(alone) = record {
            let size = affinity.block_size(registered);
            let size = size.unwrap_or_else(|| spec.block_size()).max(1);
            let behind = named != registered;
            let boundaries: Option<Box<[u32]>> = hash.then(|| spec.boundaries().into());
            let record = affinity.record_of(
                named,
                settings.max_size,
                size,
                behind,
                boundaries.as_deref(),
            );
            table.record = Some(record);
            table.alone = alone;
            table.side_alone = engine.side_reads_sent_alone(model, phase);
            table.boundaries = boundaries;
            if cap != 0 && !hash {
                let lent = affinity.indexes.get();
                if !table.holds {
                    table.sent = lent.and_then(|indexes| indexes.token_tree(named, phase));
                }
                // A request of text reads no index: its tree is asked for
                // beside one too.
                table.sent_text = lent.and_then(|indexes| indexes.string_tree(named, phase));
            }
            if table.sent.is_some() || table.sent_text.is_some() {
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

    /// Whether the numbers the table read are still the index's: it handed
    /// none out a second time since ([`CacheIndex::numbers_reused`]). The
    /// count does not say which number it handed on: once it moved, any
    /// number of the table may name a former worker at its new worker's
    /// blocks.
    fn numbered(&self) -> bool {
        let index = self.index.as_ref();
        index.is_none_or(|index| index.numbers_reused() == self.reused)
    }

    /// Whether the table still stands by its index, asked where a decision
    /// takes the table: the index's word of holding a block is the one the
    /// table was built under ([`CacheIndex::holds_a_block`]), and under an
    /// index that held one its numbers are still the index's
    /// ([`Self::numbered`]). A table built under an index that held none
    /// read no number, and is asked the word alone. So a model's first
    /// block, and its last block's leaving, read the table again at the
    /// first decision that finds the word moved. A table that was handed no
    /// index answers at one test.
    fn stands_by_index(&self) -> bool {
        let Some(index) = self.index.as_ref() else {
            return true;
        };
        index.holds_a_block() == self.holds
            && (!self.holds || index.numbers_reused() == self.reused)
    }

    /// Whom the record of sent prefixes speaks for on a request hashed by
    /// `kind`, to a reader that reads sent prefixes alone (`alone`) or to
    /// one that reads cache events too: the one rule every read and every
    /// write of the record asks. `row` asks it of one worker of the list,
    /// and `None` asks whether the record speaks for any.
    ///
    /// An index of cache events speaks of token blocks. On a request hashed
    /// by token blocks it speaks for each worker it holds a block for, and
    /// the record for the others. A request hashed by windows of its text
    /// asks no index, so the record speaks for every worker of the list: a
    /// worker the index holds a block for is read and written as any other.
    /// So it is, on a request of either kind, for a reader that reads sent
    /// prefixes alone: no prefix stage of it reads what the index holds.
    ///
    /// To a reader that reads cache events too, the record speaks for no
    /// worker of the list where the model's index decides a request hashed
    /// by token blocks alone ([`Self::index_alone`]).
    fn recorded(&self, kind: HashKind, alone: bool, row: Option<u32>) -> bool {
        match (kind, row) {
            (HashKind::Text, _) => true,
            (HashKind::Token, _) if alone => true,
            (HashKind::Token, _) if self.index_alone() => false,
            (HashKind::Token, Some(row)) => !is_set(&self.heard, row as usize),
            (HashKind::Token, None) => self.unheard_any,
        }
    }

    /// Whether the model's index of cache events decides a request hashed
    /// by token blocks by itself, as the classic policy's event path does:
    /// a single request, under a model whose index holds a block, for a
    /// reader one of whose prefix stages reads cache events. The rule asks
    /// it behind its answer to a reader that reads sent prefixes alone
    /// ([`Self::recorded`]). The record of sent prefixes is then neither
    /// read nor written, for a row the index speaks for or for any other:
    /// a worker the index holds nothing for is named nothing. Each side of
    /// a pair is decided at its own phase and keeps both.
    ///
    /// The answer is the table's own, said once at its read from the
    /// index's word of holding a block ([`CacheIndex::holds_a_block`]): the
    /// rule asks this at every question and reads one field. The word is
    /// as old as the table. A decision that finds it moved reads the table
    /// again ([`Self::stands_by_index`]), and while it does, the others
    /// decide by the table that stands: the old word decides for the
    /// length of one read. Under a host that hands no index while its
    /// index is empty the word keeps its default, and a model's first
    /// block is seen with the list's next table, up to a second later.
    fn index_alone(&self) -> bool {
        self.decides
    }

    /// The row of the worker `id` where the record of sent prefixes speaks
    /// for it on a request hashed by `kind`, to a reader of the word
    /// `alone` ([`Self::recorded`]).
    fn spoken_row(&self, id: WorkerId, kind: HashKind, alone: bool) -> Option<u32> {
        let at = self.by_id.binary_search_by_key(&id, |&(id, _)| id).ok()?;
        let row = self.by_id[at].1;
        self.recorded(kind, alone, Some(row)).then_some(row)
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
    /// names its workers with no block held: its caller hands the workers
    /// of any match that holds something, and the decision reads how much
    /// from the match's own two numbers ([`AffinityRows::tree`]). A request
    /// held to `places` is named the rows among them, and the cap counts no
    /// other.
    fn name_sent(
        &self,
        sent: &[TenantId],
        blocks: u32,
        workers: &[Arc<dyn Worker>],
        facts: &RequestFacts<'_>,
        places: Option<&[u32]>,
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
        for address in sent {
            self.rows_at(workers, address, |row| {
                if !among(places, row) {
                    return;
                }
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
        self.name_warming(facts, places, warming, picked, live);
    }

    /// The mask the index's lookup is kept within for a request: the
    /// table's, and for a request held to `places` the table's bits of the
    /// rows among them, written into `own`. A request that states no places
    /// is handed the table's by one test and no write.
    fn mask<'m>(&'m self, places: Option<&[u32]>, own: &'m mut Vec<u64>) -> &'m [u64] {
        let Some(places) = places else {
            return &self.within;
        };
        own.clear();
        own.resize(self.within.len(), 0);
        for &row in places {
            let number = self.numbers.get(row as usize).copied().unwrap_or(NONE);
            if number != NONE && is_set(&self.within, number as usize) {
                set(own, number as usize);
            }
        }
        own
    }

    /// Names a request's rows into `buffers`: its deepest holders and the
    /// list's warming rows, each list at most the cap, ascending by row.
    ///
    /// The index names the holders among the rows it speaks for, in one
    /// lookup under the cap. The record of sent prefixes names them among
    /// the others, after the index's and in the room they leave. Where no
    /// prefix stage of the profile reads cache events, the record names
    /// its holders first, among every row, and the index is asked in the
    /// room they leave. A row both name there stands once, with both
    /// counts. It took two of the cap's places before the two were made
    /// one, so under a cap that binds the naming can stand short, by
    /// entries of the index's own. So no row is named twice. The index is
    /// asked for a request hashed by token blocks alone, the record for a
    /// request of either kind: a request hashed by its text is named its
    /// holders by the record among every row ([`Self::recorded`]). A single
    /// request the model's index decides alone is named the index's
    /// holders and no other ([`Self::index_alone`]).
    ///
    /// The warming rows are the table's whose naming has not ended at the
    /// request's instant: whole within the cap, a draw of the cap past it.
    ///
    /// A request held to `places` is named the rows among them: the index
    /// is asked within them ([`Self::mask`]), and the record's holders and
    /// the warming rows outside them are passed over before a cap counts.
    /// So a row outside the places takes no room from one among them.
    ///
    /// Returns what was read of the record, where it was read: the pick's
    /// write stands on it.
    fn name(
        &self,
        facts: &RequestFacts<'_>,
        places: Option<&[u32]>,
        buffers: &mut Buffers,
    ) -> Option<RecordRead> {
        let Buffers {
            holders,
            warming,
            kept,
            keys,
            deepest,
            picked,
            live,
            within,
            waiting,
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
            .filter(|_| hashed && self.recorded(facts.hash_kind, self.alone, None));
        let mut read = None;
        if self.alone {
            read = record.map(|record| {
                self.name_recorded(
                    record, facts, places, keys, deepest, waiting, holders, &mut draw,
                )
            });
        }
        // The record's holders named ahead of the index's.
        let first = holders.len();
        let room = cap.saturating_sub(first);
        // An engine's events are token blocks: a request hashed by windows
        // of its text asks the record alone.
        let tokens = hashed && facts.hash_kind == HashKind::Token;
        let index = self.index.as_deref();
        if let Some(index) = index.filter(|_| tokens && room > 0 && self.heard_any) {
            index.deepest_into(
                facts.block_hashes,
                facts.block_size,
                self.mask(places, within),
                room,
                &mut |under| below(&mut draw, under),
                kept,
            );
            // A number handed on since the table read it holds its new
            // worker's blocks: such a lookup names nobody.
            if !self.numbered() {
                kept.clear();
            }
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
        if !self.alone {
            read = record.map(|record| {
                self.name_recorded(
                    record, facts, places, keys, deepest, waiting, holders, &mut draw,
                )
            });
        }
        holders.sort_unstable_by_key(|holder| holder.row);
        // A reader that reads sent prefixes alone is named a worker by the
        // record whatever the index holds for it, and the index may name
        // that worker behind it. The two stand as one holder with both
        // counts, as one entry of a column does: a read of the rows takes
        // a row's first entry.
        if first != 0 && first != holders.len() {
            holders.dedup_by(|next, named| {
                let same = next.row == named.row;
                if same {
                    named.overlap_blocks = named.overlap_blocks.max(next.overlap_blocks);
                    named.predicted_blocks = named.predicted_blocks.max(next.predicted_blocks);
                }
                same
            });
        }
        self.name_warming(facts, places, warming, picked, live);
        read
    }

    /// The depths, in blocks of the request, that a hash index keys it at:
    /// the whole blocks under each of `boundaries`, the view's
    /// ([`Self::blocks_of`]), ascending. Two boundaries that end in one
    /// block are one depth, and a boundary under one block is none.
    fn depths<'a>(
        &'a self,
        boundaries: &'a [u32],
        facts: &'a RequestFacts<'_>,
    ) -> impl Iterator<Item = usize> + 'a {
        let mut last = 0;
        boundaries.iter().filter_map(move |&boundary| {
            let depth = self.blocks_of(boundary as usize, facts) as usize;
            (depth != last).then(|| {
                last = depth;
                depth
            })
        })
    }

    /// The request's keys of the record of sent prefixes into `keys`: one
    /// a block ([`PrefixPredictor::fold_into`]). Under a hash index they
    /// are one a boundary of the view the request reaches, the fold through
    /// the boundary's last whole block ([`Self::depths`]), as a
    /// consistent-hash stage keys a prefix: the heads the classic policy
    /// keys its placement map by. A boundary counts in whole blocks, where
    /// the classic policy asks for its every token. A request of token ids
    /// reaches a boundary that holds a whole block where it carries a hash
    /// for each block under it. A request that carries fewer hashes than
    /// the first depth counts has no key, and neither has one hashed by
    /// windows of its text, since such an index keys token ids: the record
    /// names it nobody and takes no placement for it.
    fn fold_keys(&self, keys: &mut Vec<u64>, facts: &RequestFacts<'_>) {
        let hashes = facts.block_hashes;
        let Some(boundaries) = self.boundaries.as_deref() else {
            return PrefixPredictor::fold_into(keys, facts.namespace_seed, hashes);
        };
        // No key lies past the deepest boundary.
        let deepest = match facts.hash_kind {
            HashKind::Token => self.depths(boundaries, facts).last().unwrap_or(0),
            HashKind::Text => 0,
        };
        let within = hashes.len().min(deepest);
        PrefixPredictor::fold_into(keys, facts.namespace_seed, &hashes[..within]);
        let mut kept = 0;
        for depth in self.depths(boundaries, facts) {
            if depth > keys.len() {
                break;
            }
            keys[kept] = keys[depth - 1];
            kept += 1;
        }
        keys.truncate(kept);
    }

    /// The request's first `held` keys ([`Self::fold_keys`]) in blocks of
    /// it: their count, and under a hash index the depth the last of them
    /// was folded at ([`Self::depths`]).
    fn depth_of(&self, held: u32, facts: &RequestFacts<'_>) -> u32 {
        let Some(boundaries) = self.boundaries.as_deref() else {
            return held;
        };
        let last = (held as usize).checked_sub(1);
        let depth = last.and_then(|last| self.depths(boundaries, facts).nth(last));
        depth.map_or(0, count)
    }

    /// Reads the record of sent prefixes for the request: the deepest key
    /// of it that a worker the record speaks for was sent, as a count of
    /// keys, and the workers sent it, into `deepest`. `alone` is the
    /// reader's word the record speaks by ([`Self::recorded`]), which the
    /// read keeps for its pick's write. The request's keys
    /// are folded into `keys` ([`Self::fold_keys`]). The record's lock is
    /// held as a reader for the one read and given back before a row is
    /// chosen. A request with no key reads no block: no lock is taken.
    fn read_record(
        &self,
        record: &PrefixRecord,
        facts: &RequestFacts<'_>,
        alone: bool,
        keys: &mut Vec<u64>,
        deepest: &mut Vec<WorkerId>,
    ) -> (u32, RecordRead) {
        self.fold_keys(keys, facts);
        let tick = tick_of(facts.now_micros);
        deepest.clear();
        if keys.is_empty() {
            let read = RecordRead {
                tick,
                blocks: 0,
                depth: 0,
                whole: true,
                fresh: false,
                left: record.left.load(Ordering::Acquire),
                waits: false,
                alone,
            };
            return (0, read);
        }
        let usable = |id: WorkerId| self.spoken_row(id, facts.hash_kind, alone).is_some();
        let predictor = record.predictor.read();
        let found = predictor.deepest_among(keys, &usable);
        deepest.extend(found.holders());
        let read = RecordRead {
            tick,
            blocks: found.blocks(),
            depth: self.depth_of(found.blocks(), facts),
            whole: found.blocks() as usize == keys.len(),
            fresh: found.fresh(tick),
            // Acquire against the leaves' Release: a raise read here was
            // made behind the holder's own, which a hand-on's compare reads.
            left: record.left.load(Ordering::Acquire),
            waits: false,
            alone,
        };
        (found.blocks(), read)
    }

    /// Reads the record for the request by the reader's word `alone`
    /// ([`Self::read_record`]) and, beside
    /// it, the writes the calling thread keeps for the record (`waiting`),
    /// each as the record will hold it: a thread reads what it wrote.
    fn read_written(
        &self,
        record: &PrefixRecord,
        facts: &RequestFacts<'_>,
        alone: bool,
        (keys, deepest): (&mut Vec<u64>, &mut Vec<WorkerId>),
        waiting: &Waiting,
    ) -> (u32, RecordRead) {
        let (_, mut read) = self.read_record(record, facts, alone, keys, deepest);
        for write in &waiting.writes {
            // A leave told since stops the write, so it names nobody.
            let stands = std::ptr::eq(&*write.record, record) && write.left == read.left;
            if !stands
                || self
                    .spoken_row(write.target, facts.hash_kind, alone)
                    .is_none()
            {
                continue;
            }
            let blocks = shared(&write.keys, keys) as u32;
            // A waiting write speaks where it is deeper than the record. At
            // the record's own depth the record's holders stand alone, so a
            // pick's write starts from what the record holds for its worker.
            if blocks < read.blocks || (blocks == read.blocks && !read.waits) {
                continue;
            }
            if blocks > read.blocks {
                deepest.clear();
                read = RecordRead {
                    blocks,
                    depth: self.depth_of(blocks, facts),
                    whole: blocks as usize == keys.len(),
                    fresh: false,
                    waits: true,
                    ..read
                };
            }
            if !deepest.contains(&write.target) {
                deepest.push(write.target);
            }
        }
        (read.depth, read)
    }

    /// The record's holders of the request, behind the holders already
    /// named and in the room the cap leaves: the workers sent the deepest
    /// block of it that a worker the record speaks for was sent, a draw of
    /// them where they would fill more than the room. The deepest block is
    /// the list's, as a tree's walk finds it: a request held to `places` is
    /// named the workers sent it among them, and none sent a shorter one.
    #[expect(
        clippy::too_many_arguments,
        reason = "the record, the request and its places, the read's two buffers and the thread's kept writes, and the naming's list and draw"
    )]
    fn name_recorded(
        &self,
        record: &PrefixRecord,
        facts: &RequestFacts<'_>,
        places: Option<&[u32]>,
        keys: &mut Vec<u64>,
        deepest: &mut Vec<WorkerId>,
        waiting: &Waiting,
        holders: &mut Vec<AffinityHolder>,
        draw: &mut Draw,
    ) -> RecordRead {
        // A single decision reads by the word its table read.
        let alone = self.alone;
        let (blocks, read) = self.read_written(record, facts, alone, (keys, deepest), waiting);
        let base = holders.len();
        let room = self.cap.saturating_sub(base);
        if room == 0 {
            return read;
        }
        let mut seen = 0usize;
        for &id in deepest.iter() {
            let spoken = self.spoken_row(id, facts.hash_kind, alone);
            let Some(row) = spoken.filter(|&row| among(places, row)) else {
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
        places: Option<&[u32]>,
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
        if now >= self.first_end || places.is_some() {
            // A request held to places is named the rows among them, before
            // the cap cuts.
            live.clear();
            let live_among = |&&(row, until): &&(u32, u64)| now < until && among(places, row);
            live.extend(rows.iter().filter(live_among).copied());
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
    /// the record for a request of either kind: for the rows the index does
    /// not speak for, and for every row where the request is hashed by its
    /// text or the column's reader reads sent prefixes alone (`alone`,
    /// [`Self::recorded`]). An entry then holds both counts. The record is
    /// asked for no row of a single request the model's index decides
    /// alone ([`Self::index_alone`]).
    ///
    /// Returns what was read of the record, where it was read: the pick's
    /// write stands on it.
    fn fill(
        &self,
        facts: &RequestFacts<'_>,
        alone: bool,
        buffers: &mut Buffers,
        column: &mut Vec<AffinityFacts>,
    ) -> Option<PrefixRead> {
        let Buffers {
            kept,
            keys,
            deepest,
            waiting,
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
            // As where a decision is named its rows: no holder of a number
            // handed on since the table read it.
            if !self.numbered() {
                kept.clear();
            }
            for &(number, blocks) in kept.iter() {
                let row = self.row_of.get(number as usize).copied().unwrap_or(NONE);
                if let Some(entry) = column.get_mut(row as usize) {
                    entry.overlap_blocks = blocks;
                }
            }
        }
        // The record is read where it speaks for a row of the list.
        let speaks = self.recorded(facts.hash_kind, alone, None);
        let record = self.record.as_deref().filter(|_| speaks)?;
        let (blocks, read) = self.read_written(record, facts, alone, (keys, deepest), waiting);
        for &id in deepest.iter() {
            let Some(row) = self.spoken_row(id, facts.hash_kind, alone) else {
                continue;
            };
            if let Some(entry) = column.get_mut(row as usize) {
                entry.predicted_blocks = blocks;
            }
        }
        Some(PrefixRead { read })
    }
}

/// Whether the bit of `at` is set. A place past the mask's end is not.
fn is_set(mask: &[u64], at: usize) -> bool {
    mask.get(at / 64)
        .is_some_and(|word| word & (1 << (at % 64)) != 0)
}

/// Whether `row` is among the places a request is held to, which ascend:
/// every row is, for a request that states none.
fn among(places: Option<&[u32]>, row: u32) -> bool {
    places.is_none_or(|places| places.binary_search(&row).is_ok())
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
    /// The mask of a request held to places, for the index's lookup.
    within: Vec<u64>,
    /// The column a single decision is handed where its profile is named
    /// no row.
    column: Vec<AffinityFacts>,
    waiting: Waiting,
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
            within: Vec::new(),
            column: Vec::new(),
            waiting: Waiting::new(),
        }
    }
}

/// A write of a record that met a held lock, kept by its thread.
struct WaitingWrite {
    record: Arc<PrefixRecord>,
    /// The request's keys, and how many of its leading blocks stand in the
    /// record.
    keys: Vec<u64>,
    done: usize,
    target: WorkerId,
    left: u64,
}

impl WaitingWrite {
    /// The write ends unwritten and is counted as left out. One that a
    /// leave told since its naming had stopped is not: it would have
    /// written nothing ([`PrefixRecord::write`]).
    fn leave_out(&self) {
        if self.record.left.load(Ordering::Relaxed) == self.left {
            self.record.left_out.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The writes of the calling thread that met a held lock
/// ([`PrefixRecord::write`]). No decision waits for a record's lock: a write
/// that met it held is kept here and written on behind the thread's next
/// picks, the oldest first. Until it stands the thread's own namings read it
/// beside the record ([`Table::read_written`]), so the next turn of a
/// conversation the thread decides is named the worker of the turn before.
/// Another thread reads it once it stands.
///
/// A thread keeps the newest eight, each as long as its request's blocks. A
/// write that eight newer ones push out, or that is kept as its thread's
/// buffers go, is left out and counted ([`Affinity::left_out`]), unless a
/// leave had stopped it.
#[derive(Default)]
struct Waiting {
    writes: Vec<WaitingWrite>,
    /// The key buffers of the writes that ended, for the next that wait: a
    /// kept write allocates nothing once they have grown.
    spare: Vec<Vec<u64>>,
}

impl Waiting {
    /// The most writes a thread keeps.
    const MOST: usize = 8;

    const fn new() -> Self {
        Self {
            writes: Vec::new(),
            spare: Vec::new(),
        }
    }

    /// Takes out the write that waits for `target` with a leading part of
    /// the request of `keys`, which that request's own write continues, and
    /// returns how many of its blocks stand. A write named before a leave
    /// told since (`left`) hands on no progress: the worker may be a new one.
    ///
    /// Where the two counts differ and the kept write's is its record's
    /// still, the request's own naming is the older one. Its write stops
    /// at the record ([`PrefixRecord::write`]), so the kept write is not
    /// taken out: it stays for its own turn ([`Self::write_on`]).
    fn take_leading(
        &mut self,
        record: &PrefixRecord,
        keys: &[u64],
        target: WorkerId,
        left: u64,
    ) -> Option<usize> {
        let found = self.writes.iter().enumerate().find(|(_, write)| {
            // A key folds every block up to its own: the last one says all.
            let last = write.keys.last();
            let leads = last.is_some_and(|key| keys.get(write.keys.len() - 1) == Some(key));
            leads && write.target == target && std::ptr::eq(&*write.record, record)
        });
        let (at, kept) = found?;
        if kept.left != left && kept.left == record.left.load(Ordering::Relaxed) {
            return None;
        }
        let write = self.writes.remove(at);
        self.spare.push(write.keys);
        (write.left == left).then_some(write.done)
    }

    /// Keeps the write of `keys` for `target`, `done` of its leading blocks
    /// standing. The buffer goes with the write and `keys` takes a spare
    /// one. Where the thread keeps the most, the oldest leaves for it.
    fn keep(
        &mut self,
        record: &Arc<PrefixRecord>,
        keys: &mut Vec<u64>,
        done: usize,
        target: WorkerId,
        read: RecordRead,
    ) {
        if self.writes.len() >= Self::MOST {
            let oldest = self.writes.remove(0);
            oldest.leave_out();
            self.spare.push(oldest.keys);
        }
        let mut kept = self.spare.pop().unwrap_or_default();
        mem::swap(keys, &mut kept);
        self.writes.push(WaitingWrite {
            record: Arc::clone(record),
            keys: kept,
            done,
            target,
            left: read.left,
        });
    }

    /// Writes on, the oldest first, until a write that can stand meets a
    /// held lock. A stale one ends at its turn, at a held lock too, and the
    /// next is written on. A write that waited is a use at `tick`, the tick
    /// of the pick that writes it on, and not at its own older one: it sets
    /// no block's last use back. A pick named ahead of the write may hold
    /// the older tick: a pair's, where a stage asked for the write's
    /// decision inside the pair's own.
    fn write_on(&mut self, tick: u16) {
        while let Some(write) = self.writes.first_mut() {
            let (record, keys) = (&write.record, &write.keys);
            let done = record.write(keys, write.done, write.target, tick, write.left);
            if let Some(done) = done {
                write.done = done;
                return;
            }
            let write = self.writes.remove(0);
            self.spare.push(write.keys);
        }
    }
}

impl Drop for Waiting {
    fn drop(&mut self) {
        for write in &self.writes {
            write.leave_out();
        }
    }
}

/// How many leading blocks two requests share, by their keys. A key folds
/// every block up to its own, so the keys part once.
fn shared(one: &[u64], other: &[u64]) -> usize {
    let (mut low, mut high) = (0, one.len().min(other.len()));
    while low < high {
        let middle = low + (high - low) / 2;
        if one.get(middle) == other.get(middle) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

/// The record's write behind a pick of `target` that `read` named: the
/// request of the thread's keys, on from the blocks the target holds, then
/// the writes the thread keeps ([`Waiting`]). `held` says the naming found
/// the target among the workers sent the request's deepest block. A request
/// that meets the lock held is kept, with whatever waits behind it. A stale
/// one is not: it ends there, and what waits behind it is written on.
fn write_behind(
    record: &Arc<PrefixRecord>,
    buffers: &mut Buffers,
    read: RecordRead,
    held: bool,
    target: WorkerId,
) {
    let Buffers { keys, waiting, .. } = buffers;
    // The target holds the naming's depth where the naming found it there
    // in the record, and what a waiting write of the request's leading part
    // had written of it.
    let waited = waiting.take_leading(record, keys, target, read.left);
    let named = if held && !read.waits {
        read.blocks as usize
    } else {
        0
    };
    let known = named.max(waited.unwrap_or(0));
    let done = record.write(keys, known, target, read.tick, read.left);
    // A request the target holds whole has its last use to move and no
    // block to add: left undone, it leaves nothing out.
    if let Some(done) = done.filter(|&done| done < keys.len()) {
        waiting.keep(record, keys, done, target, read);
    } else {
        waiting.write_on(read.tick);
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
/// the record says where a prefix was sent. The record speaks for a worker
/// the index holds no block for, and for every worker of the list where
/// the request is hashed by its text: an index of cache events speaks of
/// token blocks. So it does, for every worker, where the place's profile
/// reads sent prefixes alone.
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
/// whole) is handed its list's column, in [`RequestFacts::affinity`]:
/// one entry a worker of the list, as a side of the engine's pair decision
/// is handed, with the record of sent prefixes read by that profile's own
/// word. Where it reads sent prefixes alone, the column holds what the
/// record holds for every worker.
///
/// `token_key` names the request's entry in the host's token tree of sent
/// prefixes ([`TokenKey`]), given the tree's page in tokens. It is called
/// at most once, and only where the host lends the place such a tree
/// ([`CacheIndexes::token_tree`]): its profile reads a record of sent
/// prefixes and is named rows, and the model has no index of cache events.
/// A request it names an entry for is decided inside the tree's match.
/// `text_key` names the entry of a request hashed by its text in the host's
/// string tree ([`TextKey`]). It is called at most once, of such a request
/// that was named no token entry, and only where the host lends the place
/// a string tree ([`CacheIndexes::string_tree`]), whatever the model's
/// index holds. A request it names an entry for is decided inside that
/// tree's match. `decide` is called once either way.
///
/// `sends_to` says whether the request's caller sends it to a target of the
/// pick ([`RequestMaker::sends_to`]). A decision made inside the tree's
/// match, and one the record is written behind, asks it of the pick's
/// targets, best first: the tree stores the request's pages, and the record
/// its prefix, for the first one it takes. No other decision asks it.
///
/// A pick states what its first target holds of the request ([`Overlap`]),
/// from the rows its decision was given: the rows named here, from the
/// host's data or from a tree's match, the column filled here, a column its
/// host filled or the rows its host named. A decision whose own profile
/// reads no affinity states nothing, whether it is given no rows or the
/// rows of its binding's shadow.
///
/// A request decided inside a tree's match states the match's own two
/// numbers for a worker named among its holders: the match's size and the
/// request's, in the tree's unit, tokens or chars
/// ([`OverlapSource::Tree`](crate::engine::OverlapSource::Tree)). Any
/// other worker is a miss of the request's blocks. A gateway's classic
/// cache-aware policy observes the same two numbers of such a request,
/// whoever it chose, so the two differ in two places. A pick that is none
/// of the match's holders states a miss, where the classic policy observes
/// the match's share. Past the cap the holders named are a draw, so a
/// holder the draw left out states a miss too.
///
/// The rows live in the calling thread's buffers until this returns:
/// `decide` reads them and keeps nothing of them. Nothing is allocated once
/// the buffers have grown, but where a table is read (once a list, then by
/// one decision a second and by one behind a number the index handed on)
/// and by a tree's own match and write.
///
/// [`Engine::needs_for_phase`]: crate::engine::Engine::needs_for_phase
/// [`RequestMaker::sends_to`]: crate::engine::RequestMaker::sends_to
#[expect(
    clippy::too_many_arguments,
    reason = "the list, what is kept beside it, the request's facts, its entry in each tree and the target its caller takes, and the decision"
)]
pub fn decide<'p>(
    affinity: &Affinity,
    pool: &PoolAffinity,
    spec: &AppliedSpec,
    workers: &[Arc<dyn Worker>],
    ids: &[WorkerId],
    facts: &mut RequestFacts<'_>,
    places: Option<&[u32]>,
    token_key: impl FnOnce(usize) -> Option<TokenKey<'p>>,
    text_key: impl FnOnce() -> Option<TextKey<'p>>,
    sends_to: impl Fn(&Target) -> bool,
    decide: impl FnMut(&mut RequestFacts<'_>) -> Decision,
) -> Decision {
    let hashed = || None;
    decide_hashed(
        affinity, pool, spec, workers, ids, facts, places, token_key, text_key, sends_to, hashed,
        decide,
    )
}

/// [`decide`] for a request whose making may have left its prompt's block
/// hashes out of `facts`. `hashed` answers the whole list where it did
/// ([`Made::hashed_whole`]) and `None` where the facts hold it. It is asked
/// on the roads that read the list, by an index of cache events and by the
/// record of sent prefixes, once, and those roads decide by a copy of the
/// facts that holds it. A request decided inside a lent tree's match reads
/// none of the list, and asks nothing.
///
/// [`Made::hashed_whole`]: super::place::Made::hashed_whole
#[expect(
    clippy::too_many_arguments,
    reason = "what `decide` takes, and the prompt's hashes where the request's making left them out"
)]
pub(super) fn decide_hashed<'p, 'h>(
    affinity: &Affinity,
    pool: &PoolAffinity,
    spec: &AppliedSpec,
    workers: &[Arc<dyn Worker>],
    ids: &[WorkerId],
    facts: &mut RequestFacts<'_>,
    places: Option<&[u32]>,
    token_key: impl FnOnce(usize) -> Option<TokenKey<'p>>,
    text_key: impl FnOnce() -> Option<TextKey<'p>>,
    sends_to: impl Fn(&Target) -> bool,
    hashed: impl FnOnce() -> Option<&'h PromptHashes>,
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
            let keys = (token_key, text_key);
            table.decide(
                workers, facts, places, buffers, keys, sends_to, hashed, decide,
            )
        })
    })
}

/// [`decide`] behind an answer made ahead of the lookup, a pin the keyed
/// entry followed. `leaves` is [`Affinity::leaves`] as read ahead of that
/// answer: where a leave was told since, the answer's worker may be the
/// one that left, and its prompt is stored for no target.
#[expect(
    clippy::too_many_arguments,
    reason = "what `decide_hashed` takes but a request's places, the count of leaves read ahead of the answer, and the answer"
)]
pub(super) fn decide_behind<'p, 'h>(
    affinity: &Affinity,
    pool: &PoolAffinity,
    spec: &AppliedSpec,
    workers: &[Arc<dyn Worker>],
    ids: &[WorkerId],
    facts: &mut RequestFacts<'_>,
    token_key: impl FnOnce(usize) -> Option<TokenKey<'p>>,
    text_key: impl FnOnce() -> Option<TextKey<'p>>,
    sends_to: impl Fn(&Target) -> bool,
    hashed: impl FnOnce() -> Option<&'h PromptHashes>,
    leaves: u64,
    made: Decision,
) -> Decision {
    decide_hashed(
        affinity,
        pool,
        spec,
        workers,
        ids,
        facts,
        // No places: a request held to places never comes here. The keyed
        // entry answers nothing for it (`Engine::decide_keyed_among`), so
        // its decision hands the naming its places itself.
        None,
        token_key,
        text_key,
        |target| affinity.leaves() == leaves && sends_to(target),
        hashed,
        |_| made,
    )
}

/// One list's column for a decision that reads a column in the rows' place:
/// each side of the engine's pair decision, by the side's own list and
/// `phase`.
///
/// `column` is emptied and filled with one entry a worker of `workers`, in
/// the list's order: the worker's age, what the model's index holds for it
/// and what that gained since its admission, and what the worker holds of
/// the request, by the index where the index speaks for the worker and by
/// the record of sent prefixes where it does not: for every worker where
/// the request is hashed by its text. The record is read by the word of the
/// side's own profile, which the list's table read with its other words of
/// the document ([`Table::side_alone`]): under a profile that reads sent
/// prefixes alone the record speaks for every worker of the list, whatever
/// the index holds for it ([`Table::recorded`]). The table's word of a
/// single decision's reader is not asked: it is another profile's where a
/// binding of the phase is listed ahead of the pair's. The caller hands the
/// column to the side it was filled for and to no other.
///
/// The parameters are [`decide`]'s, and `phase` is the phase the list is
/// decided at, which a pair's request does not carry for both its lists.
/// Returns what was read of the record of sent prefixes, where the record
/// speaks for a worker of the list: the read keeps its word, and the write
/// behind the pick asks the rule by it ([`record_by_column`]). Nothing is
/// allocated once `column` and the calling thread's buffers have grown, but
/// where a table is read.
#[expect(
    clippy::too_many_arguments,
    reason = "the list as `decide` takes it, the phase a pair's request does not carry, and the column"
)]
pub(super) fn column(
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
    let read = with_buffers(|buffers| {
        pool.with_table(&asked, |table| {
            table.fill(facts, table.side_alone, buffers, column)
        })
    });
    // The table of a place whose stages read no affinity holds an empty
    // column: every worker reads as holding nothing, just admitted.
    column.resize(workers.len(), AffinityFacts::default());
    read
}

/// States in a decision's pick what its first target holds of the request
/// ([`Overlap`]): that worker's entry of `feed`, the rows the decision was
/// given. They are the rows named or the column filled here, a column the
/// request's host filled, the rows that host named
/// ([`RequestFacts::feed`]), or the prefills' column of a pair's first
/// decision: whoever named them, the entry is read the same way, and no
/// stage is asked. A worker the rows do not name holds nothing,
/// as it did to the stages: the pick then states a miss of the request's
/// blocks. `facts` are the request's own, which say its size, and `block`
/// is the document's block size, by which the request is counted where its
/// host counted with none.
///
/// Rows a tree's match named carry the match ([`AffinityRows::tree`]), and
/// a worker named among them is stated the match's own two numbers
/// ([`Overlap::in_tree`]): what the classic policy's series divides.
///
/// The feed is handed by itself, and not the facts the decision was made
/// over: a decision that reads no fact, as a pin's answer handed through
/// here, then makes no copy of them for this statement.
///
/// A pair's first target is its prefill, with its decode beside it: handed
/// the prefills' column, a pair's pick states what its prefill holds. A
/// pick has room for one statement, so nothing is stated of the decode.
///
/// Nothing is stated where the decision picked nobody, and for a pick the
/// binding's sessions rule placed alone and keeps from the profile
/// ([`told_target`]): such a request is none of the profile's, as a
/// gateway's sticky map hands a request it places to no policy. The one
/// place a pick's overlap is stated. Its callers ask it where the
/// request's own profile reads affinity, and not for a shadow's rows.
pub(super) fn state(decision: &mut Decision, feed: Feed<'_>, facts: &RequestFacts<'_>, block: u32) {
    let Some(row) = told_target(decision).map(|target| target.position) else {
        return;
    };
    // The tree is asked first: rows no tree named cost no search here.
    let tree = match feed {
        Feed::Named(named) => named.tree.filter(|_| named.names(row)),
        Feed::Column(_) => None,
    };
    let stated = match tree {
        Some(tree) => Overlap::in_tree(tree.matched, tree.of, false),
        None => Overlap::held(feed.entry_or_default(row), facts.blocks(block)),
    };
    if let Decision::Pick(pick) = decision {
        pick.targets.state(stated);
    }
}

/// Whether the profile is told of a decision's pick: the pick's first
/// target where it is, and none for a decision that picked nobody and for
/// a pick the binding's sessions rule placed alone and keeps from the
/// profile ([`lease_events::stages_hear`]). The one test in front of every
/// write that keeps where a prompt went. A single's writes name the worker
/// the request goes to ([`sent_target`]), which may be a later target.
///
/// The rule's word is read through that one function by the words of a
/// lease, the record's two writes here, the store of a prompt in a lent
/// tree ([`sent_address`]), the hand-over of a pin the keyed entry followed
/// to this lookup, a pair's two writes behind its pick, and the statement
/// of what a pick's first target holds ([`state`]). A pair's two writes ask
/// it a side at a time, each of its own side (`pair::record_sides`), and
/// then name the target the request goes to ([`sent_to`]). A pair's first
/// pick states what its prefill holds by the pick's own word, which is the
/// prefill side's, whatever placed its decode: its first target is its
/// prefill. Under the rule's default word a request it
/// places is the profile's own, and its prompt is kept as any pick's. Under
/// the other the rule keeps the request from the profile, as a gateway's
/// sticky map hands a request it places to no policy: the policy's record
/// never sees that request's prompt.
pub(super) fn told_target(decision: &Decision) -> Option<&Target> {
    if !lease_events::stages_hear(decision, Side::Single) {
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
/// `pool`, `phase` and `facts` are the ones the column was filled with,
/// `read` is what that fill returned and `column` the column it filled. A
/// worker the index speaks for, a worker that is not of the list, and a
/// list whose table was since read for another place write nothing. The
/// index speaks for no worker on a request hashed by its text, and for none
/// where the fill read the record by the word of a profile that reads sent
/// prefixes alone: such a request is written for any worker of the list.
///
/// The picked worker's entry of the column says whether the fill found
/// that worker among those sent the request's deepest block
/// ([`PrefixRead::found`]): a pick of one of them may be a repeat that
/// writes nothing. A worker the column holds no entry for is written as no
/// repeat. The fill gives every worker of its list an entry, so no decision
/// of the engine hands such a column.
pub(super) fn record_by_column(
    pool: &PoolAffinity,
    phase: Phase,
    facts: &RequestFacts<'_>,
    read: PrefixRead,
    column: &[AffinityFacts],
    id: WorkerId,
) {
    let standing = pool.table.load();
    let table = standing.as_deref();
    let Some(table) = table.filter(|table| table.is_of(facts.model, phase)) else {
        return;
    };
    // The write follows its read: by the word the fill read the record by.
    let spoken = table.spoken_row(id, facts.hash_kind, read.read.alone);
    let (Some(record), Some(row)) = (table.record.as_ref(), spoken) else {
        return;
    };
    let held = column
        .get(row as usize)
        .is_some_and(|sent| read.found(sent));
    if record.repeats(read.read, held) {
        return with_buffers(|buffers| buffers.waiting.write_on(read.read.tick));
    }
    with_buffers(|buffers| {
        // The keys are the request's own, whichever list was filled last.
        table.fold_keys(&mut buffers.keys, facts);
        write_behind(record, buffers, read.read, held, id);
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
            // It stands for its second, or until its last thin row that
            // holds nothing ends, while the index hands none of its numbers
            // on and keeps its word of holding a block.
            let stands = asked.now < table.stands_until && table.stands_by_index();
            // One decision reads a table again; the others decide by the
            // one that stands, whose lookup names nobody once a number of
            // it was handed on, and whose word of the index is the old one
            // for the length of that read.
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

/// The target a single decision's request goes to: the first target of a
/// pick the profile is told of ([`told_target`]) that the request's caller
/// sends to (`sends_to`), asked best first. It is the worker a tree's
/// match and the record's two writes keep the request's prefix for. `None`
/// where the decision picked nobody, the binding's sessions rule keeps the
/// request from the profile, or the caller sends to none of the pick's
/// targets: nothing is stored.
pub(super) fn sent_target(
    decision: &Decision,
    sends_to: impl Fn(&Target) -> bool,
) -> Option<&Target> {
    told_target(decision)?;
    sent_to(decision, sends_to)
}

/// The first target of a decision's pick that the request's caller sends
/// to (`sends_to`), asked best first, whatever the sessions rule says of
/// the pick: a pair's two writes ask the rule's word of each side first
/// (`pair::record_sides`). `None` where the decision picked nobody or the
/// caller sends to none of the pick's targets.
pub(super) fn sent_to(decision: &Decision, sends_to: impl Fn(&Target) -> bool) -> Option<&Target> {
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

/// What a tree's walk hands its match to ([`Table::decide_in_tree`]): the
/// workers the match found, the match's own two numbers and the request's
/// whole blocks it holds, answered with the address the walk writes the
/// request for, where it writes it.
type Over<'a, 'w> = dyn FnMut(&[TenantId], TreeMatch, u32) -> Option<&'w str> + 'a;

/// The bytes of the leading `matched` chars of `text` that lie behind its
/// first `marker` chars: what a string tree's match holds of a prompt whose
/// key begins with a partition's marker. A string tree counts chars, and a
/// request of text is sized in bytes. One pass over the matched chars,
/// which its caller makes only for a key of several bytes a char. A
/// holder's entry alone reads it, for the request's whole windows: the
/// decision divides the match's chars by the request's.
fn bytes_behind(text: &str, marker: usize, matched: usize) -> usize {
    let matched = text.chars().take(matched);
    matched.skip(marker).map(char::len_utf8).sum()
}

impl Table {
    /// One decision by this table: the naming, the decision, and the
    /// record's write behind a pick. A request whose host names its entry
    /// in a tree it lends is decided inside that tree's match
    /// ([`Self::decide_in_tree`]): by its token ids in the token tree, and
    /// a request hashed by its text in the string tree.
    #[expect(
        clippy::too_many_arguments,
        reason = "the list, the request's facts and its places, the buffers, its entry in each tree and the target its caller takes, its hashes where its making left them out, and the decision"
    )]
    fn decide<'p, 'h>(
        &self,
        workers: &[Arc<dyn Worker>],
        facts: &mut RequestFacts<'_>,
        places: Option<&[u32]>,
        buffers: &mut Buffers,
        (token_key, text_key): (
            impl FnOnce(usize) -> Option<TokenKey<'p>>,
            impl FnOnce() -> Option<TextKey<'p>>,
        ),
        sends_to: impl Fn(&Target) -> bool,
        hashed: impl FnOnce() -> Option<&'h PromptHashes>,
        mut decide: impl FnMut(&mut RequestFacts<'_>) -> Decision,
    ) -> Decision {
        if !self.reads {
            return decide(facts);
        }
        if self.cap == 0 {
            // The fill reads the prompt's hashes: the list is asked for.
            let whole = hashed().map(|hashes| hashes.over(facts));
            let facts = whole.as_ref().unwrap_or(facts);
            return self.decide_by_column(facts, buffers, sends_to, decide);
        }
        // The entry is asked for where the host lends the place a tree,
        // once, by the tree's own page.
        let sent = self.sent.as_deref();
        let keyed = sent.and_then(|sent| Some((sent, token_key(sent.page_size())?)));
        if let Some((sent, key)) = keyed {
            let facts = &*facts;
            let walk = |over: &mut Over<'_, '_>| {
                sent.match_and_insert_with(&key.tokens, |found| {
                    // The marker is no part of the prompt, nor of its size.
                    let matched = found.matched_token_count.saturating_sub(key.marker);
                    let of = found.input_token_count.saturating_sub(key.marker);
                    let tree = TreeMatch::new(count(matched), count(of));
                    over(&found.matched_tenants, tree, self.blocks_of(matched, facts))
                });
            };
            return self.decide_in_tree(walk, sends_to, workers, facts, places, buffers, decide);
        }
        // A request hashed by its text is asked for its entry where the
        // host lends the place a string tree, once.
        let text = facts.hash_kind == HashKind::Text;
        let sent = self.sent_text.as_deref().filter(|_| text);
        if let Some((sent, key)) = sent.and_then(|sent| Some((sent, text_key()?))) {
            let facts = &*facts;
            let walk = |over: &mut Over<'_, '_>| {
                sent.match_and_insert_with(&key.text, |found| {
                    // The match and the key count the marker's chars too.
                    let chars = found.matched_char_count.saturating_sub(key.marker);
                    let of = found.input_char_count.saturating_sub(key.marker);
                    let tree = TreeMatch::new(count(chars), count(of));
                    // A key of as many bytes as chars holds one byte a
                    // char: its chars are its bytes, and no pass is made
                    // over it.
                    let bytes = if found.input_char_count == key.text.len() {
                        chars
                    } else {
                        bytes_behind(&key.text, key.marker, found.matched_char_count)
                    };
                    over(
                        &found.matched_tenants,
                        tree,
                        self.blocks_of_text(bytes, facts),
                    )
                });
            };
            return self.decide_in_tree(walk, sends_to, workers, facts, places, buffers, decide);
        }
        // The naming reads the prompt's hashes: the list is asked for.
        let whole = hashed().map(|hashes| hashes.over(facts));
        let facts = whole.as_ref().unwrap_or(facts);
        let read = self.name(facts, places, buffers);
        let named = self.named_rows(buffers);
        let mut decision = decide(&mut RequestFacts {
            named: Some(named),
            ..*facts
        });
        self.state_named(&mut decision, named, facts);
        let (Some(read), Some(record)) = (read, self.record.as_ref()) else {
            return decision;
        };
        // The record takes an entry for the worker it speaks for on this
        // request, by the word it was read by ([`Self::recorded`]). On one
        // hashed by token blocks that is no worker where the index decides
        // alone, and else no worker the index holds a block for, unless
        // the profile reads sent prefixes alone.
        let (kind, alone) = (facts.hash_kind, read.alone);
        let sent = sent_target(&decision, sends_to);
        let Some(target) = sent.filter(|to| self.recorded(kind, alone, Some(to.position))) else {
            return decision;
        };
        let held = buffers.deepest.contains(&target.id);
        if record.repeats(read, held) {
            buffers.waiting.write_on(read.tick);
        } else {
            write_behind(record, buffers, read, held, target.id);
        }
        decision
    }

    /// The rows named into `buffers`, as a decision is handed them in the
    /// affinity column's place.
    fn named_rows<'b>(&self, buffers: &'b Buffers) -> AffinityRows<'b> {
        AffinityRows {
            holders: &buffers.holders,
            warming: &buffers.warming,
            thin: self.ruled.thin,
            mean_in_flight: None,
            tree: None,
        }
    }

    /// States the pick of a decision made over `named` rows: its first
    /// target's entry of them ([`state`]), where the request's own profile
    /// reads affinity. One function for the two paths that name rows, the
    /// table's own naming ([`Self::name`]) and a tree's match
    /// ([`Self::name_sent`]), so both state by one rule.
    // Each path makes its decision in its own place and states it there:
    // handed back by one function that also decides, a decision over the
    // table's own rows costs a copy more, 30 instructions, counted with
    // Callgrind.
    fn state_named(
        &self,
        decision: &mut Decision,
        named: AffinityRows<'_>,
        facts: &RequestFacts<'_>,
    ) {
        if self.states {
            state(decision, Feed::Named(named), facts, self.block);
        }
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

    /// How many of the request's blocks a string tree's match of `bytes`
    /// bytes of its text holds. A request hashed by its text is sized in
    /// bytes, and its blocks are windows of the size its host counted it
    /// with: the divisor the stages count such a request's blocks by
    /// ([`RequestFacts::blocks`]).
    fn blocks_of_text(&self, bytes: usize, facts: &RequestFacts<'_>) -> u32 {
        count(bytes) / facts.block_size_or(self.block).max(1)
    }

    /// One decision of a request inside the match of a tree its host lends
    /// for its place: the token tree, or the string tree for a request of
    /// text. `walk` runs the tree's match by the request's entry in it and
    /// hands [`Over`] what the match found: the workers sent the deepest
    /// prefix of the request, the match's size and the request's in the
    /// tree's own unit, and how many of the request's whole blocks that
    /// prefix holds. The decision is made over them, and the same descent
    /// writes the request for the worker it goes to, whose address `Over`
    /// answers: the first target of the pick its caller sends to
    /// (`sends_to`). A token tree stores the request's whole pages. A
    /// decision that picks nobody, or none of whose targets its caller
    /// sends to, stores nothing: the match has marked the path it walked as
    /// used, as every match of a tree does. The record of hashed prefixes
    /// is neither read nor written.
    ///
    /// The rows carry the match's two numbers ([`AffinityRows::tree`]): the
    /// profile's prefix stage divides them as the classic policy divides a
    /// tree's match, with no count rounded to a block. The pick states them
    /// for a holder named ([`state`]), where the request's own profile
    /// reads affinity, and a miss for any other worker.
    #[expect(
        clippy::too_many_arguments,
        reason = "the tree's walk, the target the caller takes, the list, the request's facts and its places, the buffers, and the decision"
    )]
    fn decide_in_tree<'w>(
        &self,
        walk: impl FnOnce(&mut Over<'_, 'w>),
        sends_to: impl Fn(&Target) -> bool,
        workers: &'w [Arc<dyn Worker>],
        facts: &RequestFacts<'_>,
        places: Option<&[u32]>,
        buffers: &mut Buffers,
        mut decide: impl FnMut(&mut RequestFacts<'_>) -> Decision,
    ) -> Decision {
        let mut decide_over = |holders: &[TenantId], tree: Option<TreeMatch>, blocks: u32| {
            // A match of anything behind the marker names its workers, as
            // the classic policy makes every tenant of a match a candidate.
            // A match of nothing names nobody.
            let found = tree.is_some_and(|tree| tree.matched > 0);
            let holders = if found { holders } else { &[] };
            self.name_sent(holders, blocks, workers, facts, places, buffers);
            let named = AffinityRows {
                tree,
                ..self.named_rows(buffers)
            };
            let mut decision = decide(&mut RequestFacts {
                named: Some(named),
                ..*facts
            });
            self.state_named(&mut decision, named, facts);
            decision
        };
        let mut decided = None;
        walk(&mut |holders, tree, blocks| {
            let decision = decide_over(holders, Some(tree), blocks);
            let to = sent_address(workers, &decision, &sends_to);
            decided = Some(decision);
            to
        });
        // The tree asks for the worker once, so the decision was made
        // there. Were it not asked, the request is decided here, with no
        // holder named and nothing written: its pick states a miss.
        decided.unwrap_or_else(|| decide_over(&[], None, 0))
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
        // A single decision reads by the word its table read.
        let mut column = mem::take(&mut buffers.column);
        let read = self.fill(facts, self.alone, buffers, &mut column);
        let mut decision = decide(&mut RequestFacts {
            affinity: &column,
            ..*facts
        });
        if self.states {
            state(&mut decision, Feed::Column(&column), facts, self.block);
        }
        buffers.column = column;
        let (Some(read), Some(record)) = (read, self.record.as_ref()) else {
            return decision;
        };
        let (kind, alone) = (facts.hash_kind, read.read.alone);
        let sent = sent_target(&decision, sends_to);
        let Some(target) = sent.filter(|to| self.recorded(kind, alone, Some(to.position))) else {
            return decision;
        };
        let sent = buffers.column.get(target.position as usize);
        let held = sent.is_some_and(|sent| read.found(sent));
        if record.repeats(read.read, held) {
            buffers.waiting.write_on(read.read.tick);
        } else {
            write_behind(record, buffers, read.read, held, target.id);
        }
        decision
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/in_process/overlap/tests.rs"]
mod tests;
