//! What a decision reads of the prompts its workers hold, from the
//! gateway's own data.
//!
//! The affinity stages of a profile ask two things of a request's pool:
//! which workers hold a leading part of the prompt, and which workers are
//! warming ([`AffinityRows`]). In a gateway's own process the answers live
//! in the gateway's index of its workers' cache events ([`CacheIndex`]), on
//! the worker objects, and in the engine's record of the prefixes it sent
//! ([`Affinity`]). [`decide`] names a request's rows from them, hands them
//! to the decision and, where the decision picked, records the request's
//! prefix as sent to the picked worker. A pair decision reads a column a
//! side in the rows' place ([`AffinityFacts`]): [`column()`] fills one from
//! the index and the worker objects.
//!
//! **One request reads one record.** The choice is the classic cache-aware
//! policy's own, stated once for both ([`PrefixSource::of`]). A tokenized
//! request of a model whose workers publish cache events is named by the
//! index alone. A tokenized request of a model with no index is named by
//! the model's token tree. A request its host did not tokenize is named by
//! the model's string tree, over its text. So where a model has an index,
//! a worker the index does not speak for is named no holder.
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
//!   model's workers, in the two trees a classic cache-aware policy keeps
//!   for a model ([`TokenTree`], [`StringTree`]): the same types, never
//!   that policy's own trees. A tree is made at the model's first request
//!   of its kind. One walk reads and writes it, and the whole decision
//!   runs inside that walk, between its match and its insert, with no lock
//!   of the tree held: a tree locks one node at a time, so no pick waits
//!   for another pick's walk. The map of the records is behind a mutex that
//!   only the read of a list's table, a removed worker and the timer's turn
//!   take.
//! - **Once a thread**: the buffers a naming writes into.
//!
//! A decision the index names allocates nothing once the thread's buffers
//! have grown. A decision a tree names allocates what the tree's walk
//! allocates, as the classic policy's decision does.
//!
//! The request's tokens and text are lent to the walk by the request's
//! maker ([`PromptLent`]) and are read nowhere else: the engine keeps
//! neither, and no fact of the request holds them.
//!
//! [`WarmUpRule`]: crate::engine::WarmUpRule

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
use kv_index::{CacheNamespace, ChainIndex, PrefixSource, StringTree, TenantId, TokenTree};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use smg_worker::identity::basic::Worker;

use crate::{
    document::spec::{AppliedSpec, IndexKind},
    engine::{
        decision::Decision,
        decision_engine::PromptLent,
        view::{
            fingerprint, AffinityFacts, AffinityHolder, AffinityRows, HashKind, Phase,
            RequestFacts, WorkerId, TEXT_BYTES_PER_TOKEN,
        },
        warm_up::WarmUpState,
    },
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
    /// a worker left and at the timer's turn. A decision made by a table
    /// that stands locks nothing here: it holds its record by the table.
    records: Mutex<FxHashMap<Box<str>, Arc<SentPrefixes>>>,
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

    /// The worker at `address` left the pool or was replaced: every model's
    /// trees drop the prefixes it was sent, so no later request is named a
    /// worker that is gone. The records' map is held for a copy of its
    /// handles alone: the trees are walked behind it, a node at a time, and
    /// no decision waits on the removal.
    pub fn forget(&self, address: &str) {
        let records: Vec<Arc<SentPrefixes>> = self.records.lock().values().cloned().collect();
        if records.is_empty() {
            return;
        }
        let worker = TenantId::from(address);
        for record in records {
            record.forget(&worker);
        }
    }

    /// The timer's turn at `now` on the decision clock: the trees of each
    /// model whose interval has passed are evicted down to their bound, the
    /// least recently used prefixes first, as the classic policy's eviction
    /// task does. Under its bound a tree is not walked. Over it the pass
    /// walks the whole tree, so a host runs its timer where that may take
    /// its time. The records' map is held to find the records that are due
    /// and no longer.
    pub fn sweep(&self, now: u64) {
        let records = self.records.lock();
        let due = records.values().filter(|record| record.due(now));
        let due: Vec<Arc<SentPrefixes>> = due.cloned().collect();
        drop(records);
        for record in due {
            record.evict(now);
        }
    }

    /// The record of `model`, made at its first asking with a token tree
    /// of `page` tokens a page, and held to `max_size` tokens or chars a
    /// tree, evicted every `every` microseconds. A record that stands keeps
    /// the page it was made with.
    fn record_of(
        &self,
        model: &str,
        page: usize,
        max_size: usize,
        every: u64,
        now: u64,
    ) -> Arc<SentPrefixes> {
        let record = {
            let mut records = self.records.lock();
            if let Some(record) = records.get(model) {
                Arc::clone(record)
            } else {
                let record = Arc::new(SentPrefixes::new(page, now));
                records.insert(model.into(), Arc::clone(&record));
                record
            }
        };
        record.max_size.store(max_size, Ordering::Relaxed);
        record.every.store(every, Ordering::Relaxed);
        record
    }
}

/// One model's record of the prefixes sent to its workers: the two trees a
/// classic cache-aware policy keeps for a model, shared by every deciding
/// thread. A worker is a tenant of a tree, by its address.
///
/// A tree is made at the model's first request of its kind, so a model
/// whose requests all come tokenized keeps no string tree.
struct SentPrefixes {
    /// The prefixes of the tokenized requests that no index named.
    tokens: OnceLock<TokenTree>,
    /// The prefixes of the requests their host did not tokenize, by text.
    text: OnceLock<StringTree>,
    /// The token tree's page, in tokens: the model's block size when the
    /// record was made. A prefix under one page is one no engine's cache
    /// can serve, and the tree keeps none.
    page: usize,
    /// The bound of each tree, in tokens or chars, as the last table read
    /// set it from the document's index settings.
    max_size: AtomicUsize,
    /// The time between two evictions in microseconds, set beside it.
    every: AtomicU64,
    /// The decision clock at the last eviction, and at the record's making
    /// before the first.
    evicted_at: AtomicU64,
}

impl SentPrefixes {
    fn new(page: usize, now: u64) -> Self {
        Self {
            tokens: OnceLock::new(),
            text: OnceLock::new(),
            page: page.max(1),
            max_size: AtomicUsize::new(usize::MAX),
            every: AtomicU64::new(u64::MAX),
            evicted_at: AtomicU64::new(now),
        }
    }

    /// The token tree, sized to the model's block as the classic policy
    /// sizes its own.
    fn tokens(&self) -> &TokenTree {
        self.tokens
            .get_or_init(|| TokenTree::with_config(self.page, Default::default()))
    }

    /// The string tree.
    fn text(&self) -> &StringTree {
        self.text.get_or_init(StringTree::new)
    }

    /// Whether an eviction is due at `now`.
    fn due(&self, now: u64) -> bool {
        let since = now.saturating_sub(self.evicted_at.load(Ordering::Relaxed));
        since >= self.every.load(Ordering::Relaxed)
    }

    /// One eviction of the trees that stand, each down to the bound, at
    /// `now`: the trees' own pass.
    fn evict(&self, now: u64) {
        self.evicted_at.store(now, Ordering::Relaxed);
        let max_size = self.max_size.load(Ordering::Relaxed);
        if let Some(tree) = self.tokens.get() {
            tree.evict_tenant_by_size(max_size);
        }
        if let Some(tree) = self.text.get() {
            tree.evict_tenant_by_size(max_size);
        }
    }

    /// Drops every prefix sent to `worker` from the trees that stand.
    fn forget(&self, worker: &TenantId) {
        if let Some(tree) = self.tokens.get() {
            tree.remove_tenant_all(worker);
        }
        if let Some(tree) = self.text.get() {
            tree.remove_tenant_all(worker);
        }
    }
}

/// What one request's holders are named from.
#[derive(Clone, Copy)]
enum Holders<'a> {
    /// The model's index of cache events, asked where the request is
    /// hashed by token blocks.
    Index,
    /// A tree's walk: the workers at the deepest node the request's prefix
    /// reaches, each holding its first `blocks` blocks.
    Sent {
        workers: &'a [TenantId],
        blocks: u32,
    },
    /// Nothing: no holder is named.
    Nobody,
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
/// said at its read, as a ruling: whom the index holds blocks for, and who
/// warms.
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
    /// A bit an index number: a row of the list the index holds a block
    /// for. The index's lookup is kept within these.
    within: Box<[u64]>,
    /// Whether the index holds a block for any row of the list.
    indexed_any: bool,
    /// The model's record of sent prefixes, where the profile reads one.
    record: Option<Arc<SentPrefixes>>,
    /// No prefix stage of the profile reads cache events: the index names
    /// no request's holders, and a tree names every request a prompt is
    /// lent for.
    record_alone: bool,
    /// The list's workers by id, each with its row: where a worker a tree
    /// names stands. Empty where the list keeps no record.
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
            indexed_any: false,
            record: None,
            record_alone: false,
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
        for (row, (&number, &size)) in numbers.iter().zip(&sizes).enumerate() {
            if number == NONE {
                continue;
            }
            row_of[number as usize] = row as u32;
            if size > 0 {
                set(&mut within, number as usize);
                table.indexed_any = true;
            }
        }
        table.index = index;
        table.row_of = row_of.into();
        table.within = within.into();
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
        if let Some(alone) = record.filter(|_| settings.kind == IndexKind::Tree) {
            // The token tree's page is the model's block: the size its
            // cache events taught the host's index, and the document's
            // where the index knows none.
            let size = affinity.block_size(registered);
            let page = size.unwrap_or_else(|| spec.block_size()).max(1) as usize;
            let max_size = usize::try_from(settings.max_size).unwrap_or(usize::MAX);
            let every = u64::from(settings.sweep_s).saturating_mul(1_000_000);
            table.record = Some(affinity.record_of(registered, page, max_size, every, now));
            table.record_alone = alone;
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

    /// Calls `each` with the row of every worker of the list at `address`,
    /// the address a tree names a worker by: found by the id every decision
    /// knows a worker by, the fingerprint of its address, and held to the
    /// address itself.
    fn rows_at(&self, workers: &[Arc<dyn Worker>], address: &str, mut each: impl FnMut(u32)) {
        let id = WorkerId(fingerprint(address.as_bytes()));
        let from = self.by_id.partition_point(|&(other, _)| other < id);
        for &(other, row) in self.by_id.iter().skip(from) {
            if other != id {
                break;
            }
            let worker = workers.get(row as usize);
            if worker.is_some_and(|worker| worker.url() == address) {
                each(row);
            }
        }
    }

    /// Names a request's rows into `buffers`: its deepest holders and the
    /// list's warming rows, each list at most the cap, ascending by row.
    ///
    /// The holders come from one source ([`Holders`]). The index names
    /// them in one lookup under the cap, among the rows it holds blocks
    /// for, where the request is hashed by token blocks. A tree's walk
    /// names the workers at the deepest node the request's prefix reaches:
    /// those of them in this list, a draw of them where they would fill
    /// more than the cap.
    ///
    /// The warming rows are the table's whose naming has not ended at the
    /// request's instant: whole within the cap, a draw of the cap past it.
    fn name(
        &self,
        from: Holders<'_>,
        workers: &[Arc<dyn Worker>],
        facts: &RequestFacts<'_>,
        buffers: &mut Buffers,
    ) {
        let Buffers {
            holders,
            warming,
            kept,
            picked,
            live,
            ..
        } = buffers;
        holders.clear();
        warming.clear();
        let cap = self.cap;
        let mut draw = Draw::new(facts.seed ^ HOLDERS_STREAM);
        match from {
            Holders::Index => {
                // An engine's events are token blocks: a request hashed by
                // windows of its text has no holder in the index.
                let tokens = !facts.block_hashes.is_empty() && facts.hash_kind == HashKind::Token;
                let index = self.index.as_deref();
                if let Some(index) = index.filter(|_| tokens && cap > 0 && self.indexed_any) {
                    index.deepest_into(
                        facts.block_hashes,
                        facts.block_size,
                        &self.within,
                        cap,
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
            }
            Holders::Sent {
                workers: sent,
                blocks,
            } if blocks > 0 => {
                let mut seen = 0usize;
                for address in sent {
                    self.rows_at(workers, address, |row| {
                        let holder = AffinityHolder {
                            row,
                            overlap_blocks: 0,
                            predicted_blocks: blocks,
                        };
                        seen += 1;
                        if seen <= cap {
                            holders.push(holder);
                        } else if let Some(slot) =
                            holders.get_mut(below(&mut draw, count(seen)) as usize)
                        {
                            // Past the cap each worker is as likely to
                            // stand as another.
                            *slot = holder;
                        }
                    });
                }
            }
            Holders::Sent { .. } | Holders::Nobody => {}
        }
        holders.sort_unstable_by_key(|holder| holder.row);
        self.name_warming(facts, warming, picked, live);
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
    /// instant, and over it what the request's holders hold by the index.
    ///
    /// With `indexed`, the index says what the rows it holds blocks for
    /// hold of the request, every holder with its own depth, in one lookup,
    /// where the request is hashed by token blocks. A column has an entry a
    /// row, so nothing is capped and nothing drawn.
    ///
    /// No tree is read for a column. A profile that is named no row keeps
    /// no record of sent prefixes, so its column names no worker by a prefix
    /// it was sent.
    fn fill(
        &self,
        indexed: bool,
        facts: &RequestFacts<'_>,
        buffers: &mut Buffers,
        column: &mut Vec<AffinityFacts>,
    ) {
        let Buffers { kept, .. } = buffers;
        column.clear();
        column.extend_from_slice(&self.column);
        let later = millis(facts.now_micros.saturating_sub(self.read_at));
        if later > 0 {
            for entry in column.iter_mut() {
                entry.admitted_age_ms = entry.admitted_age_ms.saturating_add(later);
            }
        }
        let tokens = !facts.block_hashes.is_empty() && facts.hash_kind == HashKind::Token;
        let index = self.index.as_deref();
        let Some(index) = index.filter(|_| indexed && tokens && self.indexed_any) else {
            return;
        };
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
    /// The calling thread's buffers: a naming by the index allocates
    /// nothing once they have grown to the profile's cap.
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
/// what `decide` answers. `decide` is called once.
///
/// One record names the request's holders: the module's head says which.
/// Where it is a tree of the record of sent prefixes, `prompt` is asked for
/// the request's tokens or text, once, and `decide` runs inside the tree's
/// walk of them. The walk finds the workers the prefix was sent to, the
/// decision is made over them, and where it picked, the walk records the
/// prefix as sent to the first target before it ends: the record says
/// where a prefix was decided to go. A request whose host lends no prompt
/// reads and writes no tree. A place whose stages read no record of sent
/// prefixes never asks `prompt`.
///
/// `workers` is the list the decision is made over and `pool` what the
/// engine keeps beside that list. `ids` holds the workers' ids by their
/// place in the list, and a pick's first target names its worker by that
/// place. `facts` carries the request's block hashes and the decision's
/// instant ([`RequestFacts::now_micros`]) on the one clock of the engine's
/// decisions.
///
/// A profile whose stages read no affinity, and a request whose host filled
/// a column or named rows itself, pass straight to `decide`. A profile that
/// reads affinity and is named no row (it shows its rows to a plugin, whose
/// stage reads the column whole) is handed the column in
/// [`RequestFacts::affinity`], filled from the index: such a profile keeps
/// no record of sent prefixes.
///
/// The rows live in the calling thread's buffers until this returns:
/// `decide` reads them and keeps nothing of them.
#[expect(
    clippy::too_many_arguments,
    reason = "the list, what is kept beside it, the request's facts and prompt, and the decision"
)]
pub fn decide<'p>(
    affinity: &Affinity,
    pool: &PoolAffinity,
    spec: &AppliedSpec,
    workers: &[Arc<dyn Worker>],
    ids: &[WorkerId],
    facts: &mut RequestFacts<'_>,
    prompt: impl FnOnce() -> Option<PromptLent<'p>>,
    decide: impl Fn(&mut RequestFacts<'_>) -> Decision,
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
    with_buffers(|buffers| {
        pool.with_table(&asked, |table| {
            table.decide(workers, facts, buffers, prompt, &decide)
        })
    })
}

/// One list's column for a decision that reads a column in the rows' place:
/// each side of a pair decision, by the side's own list and `phase`.
///
/// `column` is emptied and filled with one entry a worker of `workers`, in
/// the list's order: the worker's age, what the model's index holds for it
/// and what that gained since its admission, and what the worker holds of
/// the request by the index. The caller hands the column to the side it
/// was filled for and to no other.
///
/// The record of sent prefixes is not read here. A tree is read and
/// written in one walk around a decision ([`decide`]), and a column is
/// filled ahead of one: a side's column names no worker by a prefix it was
/// sent.
///
/// The parameters are [`decide`]'s, and `phase` is the phase the list is
/// decided at, which a pair's request does not carry for both its lists.
/// Nothing is allocated once `column` and the calling thread's buffers have
/// grown, but where a table is read.
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
) {
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
    with_buffers(|buffers| {
        pool.with_table(&asked, |table| {
            table.fill(true, facts, buffers, column);
        });
    });
    // A place whose stages read no affinity keeps no table: every worker
    // reads as holding nothing, just admitted.
    column.resize(workers.len(), AffinityFacts::default());
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

/// The address of the worker a decision picked first, in the list it was
/// made over: the worker a tree's walk records the request's prefix for.
/// `None` where the decision picked nobody, and the walk records nothing.
fn first_address<'w>(workers: &'w [Arc<dyn Worker>], decision: &Decision) -> Option<&'w str> {
    let target = decision.pick()?.targets.first()?;
    Some(workers.get(target.position as usize)?.url())
}

/// The size in tokens of one of the request's blocks, for a prefix a token
/// tree matched: the size the request was counted with, and the tree's page
/// where it was counted with none.
fn token_block(facts: &RequestFacts<'_>, page: usize) -> usize {
    let counted = facts.hash_kind == HashKind::Token && facts.block_size > 0;
    if counted {
        facts.block_size as usize
    } else {
        page
    }
}

/// The whole blocks of the request within the `matched` chars a string tree
/// found of its `input` chars, where its text is `bytes` long. A block of a
/// text is a window of bytes: the size the request was counted with, and the
/// tree's page in tokens times the bytes of a token where it was counted
/// with none. The chars are read as bytes by the text's own mean, which is
/// exact for a text of one-byte chars.
fn text_blocks(
    facts: &RequestFacts<'_>,
    page: usize,
    matched: usize,
    input: usize,
    bytes: usize,
) -> u32 {
    if input == 0 {
        return 0;
    }
    let counted = facts.hash_kind == HashKind::Text && facts.block_size > 0;
    let window = if counted {
        facts.block_size as usize
    } else {
        page.saturating_mul(TEXT_BYTES_PER_TOKEN as usize)
    };
    let matched_bytes = (matched as u128 * bytes as u128 / input as u128) as usize;
    count(matched_bytes / window.max(1))
}

impl Table {
    /// One decision by this table: the choice of the record that names the
    /// request's holders, and the decision over them. Where the record is a
    /// tree, the decision is made inside the tree's walk of the request,
    /// and the walk records the request for the worker picked.
    fn decide<'p>(
        &self,
        workers: &[Arc<dyn Worker>],
        facts: &mut RequestFacts<'_>,
        buffers: &mut Buffers,
        prompt: impl FnOnce() -> Option<PromptLent<'p>>,
        decide: &impl Fn(&mut RequestFacts<'_>) -> Decision,
    ) -> Decision {
        if !self.reads {
            return decide(facts);
        }
        // With no tree read, the index names the holders where a prefix
        // stage of the profile reads cache events, and nobody does where
        // none reads them.
        let unsent = if self.record_alone {
            Holders::Nobody
        } else {
            Holders::Index
        };
        // The prompt is asked for where the place keeps a record of sent
        // prefixes, once. A host that lends none, or a prompt of neither
        // tokens nor text, has no tree read.
        let lent = self.record.as_deref().and_then(|record| {
            let lent = prompt()?;
            (lent.tokens.is_some() || lent.text.is_some()).then_some((record, lent))
        });
        let Some((record, lent)) = lent else {
            return self.decide_by(unsent, workers, facts, buffers, decide);
        };
        // The classic policy's choice between the three records. The index
        // is one of them where a prefix stage of the profile reads it.
        let index = || self.index.as_deref().filter(|_| !self.record_alone);
        match PrefixSource::of(lent.tokens, lent.text, index) {
            PrefixSource::Events { .. } => {
                self.decide_by(Holders::Index, workers, facts, buffers, decide)
            }
            PrefixSource::TokenTree(tokens) => {
                // The request's key under its cache partition. A match
                // count read from the tree has the marker's length taken
                // off.
                let (key, marker) =
                    CacheNamespace::keyed_tokens(lent.namespace, tokens, || record.page);
                let block = token_block(facts, record.page);
                let mut decided = None;
                record.tokens().match_and_insert_with(&key, |found| {
                    let matched = found.matched_token_count.saturating_sub(marker);
                    let sent = Holders::Sent {
                        workers: &found.matched_tenants,
                        blocks: count(matched / block.max(1)),
                    };
                    let decision = self.decide_by(sent, workers, facts, buffers, decide);
                    let to = first_address(workers, &decision);
                    decided = Some(decision);
                    to
                });
                // The walk makes its choice once, so the decision was made
                // there. A walk that made none leaves the request decided
                // here, with no holder named and nothing recorded.
                decided.unwrap_or_else(|| {
                    self.decide_by(Holders::Nobody, workers, facts, buffers, decide)
                })
            }
            PrefixSource::StringTree(text) => {
                // The same rule of the partition, in chars.
                let (key, marker) = CacheNamespace::keyed_text(lent.namespace, text);
                // The marker is one byte a char.
                let bytes = key.len().saturating_sub(marker);
                let mut decided = None;
                record.text().match_and_insert_with(&key, |found| {
                    let matched = found.matched_char_count.saturating_sub(marker);
                    let input = found.input_char_count.saturating_sub(marker);
                    let sent = Holders::Sent {
                        workers: &found.matched_tenants,
                        blocks: text_blocks(facts, record.page, matched, input, bytes),
                    };
                    let decision = self.decide_by(sent, workers, facts, buffers, decide);
                    let to = first_address(workers, &decision);
                    decided = Some(decision);
                    to
                });
                decided.unwrap_or_else(|| {
                    self.decide_by(Holders::Nobody, workers, facts, buffers, decide)
                })
            }
        }
    }

    /// One decision over the holders `from` names: by the rows named, or,
    /// for a profile that is named no row, by the column whole.
    fn decide_by(
        &self,
        from: Holders<'_>,
        workers: &[Arc<dyn Worker>],
        facts: &RequestFacts<'_>,
        buffers: &mut Buffers,
        decide: &impl Fn(&mut RequestFacts<'_>) -> Decision,
    ) -> Decision {
        if self.cap == 0 {
            // A profile named no row keeps no record of sent prefixes, so
            // its column holds the index's word alone. The column leaves
            // the buffers while the fill writes the others.
            let mut column = mem::take(&mut buffers.column);
            self.fill(matches!(from, Holders::Index), facts, buffers, &mut column);
            let decision = decide(&mut RequestFacts {
                affinity: &column,
                ..*facts
            });
            buffers.column = column;
            return decision;
        }
        self.name(from, workers, facts, buffers);
        decide(&mut RequestFacts {
            named: Some(AffinityRows {
                holders: &buffers.holders,
                warming: &buffers.warming,
                thin: self.ruled.thin,
                mean_in_flight: None,
            }),
            ..*facts
        })
    }
}

#[cfg(test)]
mod tests;
