//! The worker's KV port over the worker host's KV tables. A lane's cursor
//! reads a batch's header, and an admitted batch is applied inside the
//! row's lane by the worker crate's apply rules, the ones every index of
//! cache events is written by.
//!
//! # What the adapter keeps
//!
//! The rules keep state a row beside the index: the copies of a block on
//! each tier, so a removal takes a block out with its last copy; the block
//! of record, so a store narrower than it is left out; the row's view of
//! its cache groups other than main attention; and what the row's stream
//! showed of its groups. The adapter keeps that state a row until the
//! worker leaves. The row's reverse map is the lane's: the state holds it
//! for a batch alone, and nothing between batches.
//!
//! The tables empty a rank in their own turn: a resync, a restart or a gap
//! a cursor reads, the worker reported gone. The lane says a value that
//! moves with each emptying, and a batch that reads another value than the
//! row's last one resets the row's state first. The tables' own lock is let
//! go before a row's state is locked, and a row's state is locked inside
//! its lane's turn alone.
//!
//! A row opened without that state loses any cached blocks and cursor
//! before its feed starts.
//!
//! # The ranks of an engine
//!
//! A data-parallel engine is one row a rank, and it publishes one events
//! stream for all of them. The stream is its rank 0 row's: a row above
//! rank 0 opens no feed. A batch names its rank, and the adapter writes it
//! to that rank's row, which it asks the holder for. What a row's batches
//! showed of its groups is kept for the row a batch is written to. A
//! stream's reset, its new connection and its removal reach every rank's
//! row the stream writes.
//!
//! One stream writes a row. A row whose facts move it under another
//! engine is that engine's stream's from its next batch: it begins from
//! nothing there, and the stream that wrote it reaches it no longer. A
//! feed's own row that becomes a rank of another engine is taken the same
//! way, and its own stream, which is ending, leaves it alone. That stream
//! writes no row of any rank from then on, and the holder answers it no
//! rank once its row stands above rank 0. A ticket from before a row's
//! object was laid again opens no feed.
//!
//! A row's stream and a stream's rows are kept under one lock: a row
//! moves between two streams in one step, and its lane is cleared in
//! that step, so the clear takes nothing a later stream wrote. A stream
//! that leaves reaches only the rows it still writes. It clears them in
//! the tables outside that lock, which every row's batch takes, and each
//! stays its row until its clear ends. A stream that would take one in
//! between passes its batch over, so a removal clears nothing a new
//! stream wrote.
//!
//! An open feed's own row that the holder stands at rank 0 is taken by no
//! other stream: only a route read before the row was laid again asks for
//! it, and that batch is passed over. A feed's open asks the holder
//! whether its object still stands for the row, and whether the row
//! still stands at rank 0, in the hold that writes the row's entry. An
//! object the holder kept through a move stands for its row above rank 0
//! too, so the first answer does not give the second. A row laid again
//! behind those answers is taken from the stale feed by the stream the
//! holder then routes to it, and never by the stale feed from that
//! stream.
//!
//! # The block size
//!
//! A batch the tables take that carries a store names the model's block
//! size: the row's block of record, the widest block its stores carried
//! since its last clear. The engine's word only rises: an engine also
//! publishes prefix entries narrower than its block, and a request hashed
//! at their width would match no whole block. The size of record is put to
//! the engine's word wherever the two differ. So the first stored block
//! replaces a size a registration named, and the next one puts back a size
//! a registration raised behind it. A worker that states no size needs no
//! other word. The adapter keeps the model each open row streams for and
//! the engine's word for each model. The tables keep the size of record.
//!
//! # The series
//!
//! A batch the tables take is applied and accounted for by the worker
//! crate's one routine, under the URL of the stream's worker: its blocks
//! by operation, the stores the index did not take as they came, its age
//! and the batch itself. The adapter adds what its own answers say: a
//! live batch's pace, the worker's index blocks, a batch behind its
//! lane's cursor, a gap that asks a replay, and each reset by its reason.
//! The series are the worker crate's, written to the recorder in scope as
//! the collector's are.
//!
//! # Limits
//!
//! - A store on an indexed tier raises the row's block of record whether or
//!   not the index takes its blocks.
//! - A batch of a rank whose row the holder has not laid is passed over.
//!   That row's index begins with the blocks stored once it is laid.
//! - A feed's own row laid again in place as rank 0 of another engine is
//!   written by its old stream until that stream ends. Its end clears
//!   every row it wrote.
//! - What a row's old stream had in flight as the row moved, a batch or a
//!   reset, can land behind the new stream's clear.
//! - What a lane's cursor decides by itself is in no series: a gap kept
//!   or cleared, the batches missed, the degraded ranks, and the reason it
//!   emptied a rank ahead of a batch.
//! - The index-blocks gauge moves at a batch: a rank the tables emptied
//!   shows at its row's next batch.
//! - A batch the adapter passes over by itself is counted nowhere: one of
//!   a rank with no row laid, or one on a stream that is ending.

use std::{
    collections::HashMap,
    mem,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use parking_lot::Mutex;
use smg_worker::{
    common_proto::{kv_cache_event::Data, KvEventBatch},
    identity::basic::Worker,
    kv_events::{
        apply::{apply_counted, apply_events, BlockIndex, HeldBlocks, WorkerIndexState},
        port::{BatchFacts, BatchOutcome, KvCursor, KvEventSink, ResyncReason},
        KvEventFeed, KvEventHost,
    },
    metrics::writers::Metrics,
    registry::Generation,
};

use super::{
    kv_tables::{
        ApplyError, ChainBlockMap, KvTables, LaneWriter, SequenceHash, StoredBlock, WorkerGroups,
    },
    Objects,
};
use crate::engine::view::WorkerId;

/// The worker crate's KV event port over the worker host's KV tables. A
/// host builds [`KvHost`] and never the adapter: its subscriptions get the
/// host's one adapter as their sink.
pub struct KvAdapter {
    /// The rows the streams write: a row's stream and a stream's rows,
    /// which move together under this one lock.
    rows: Mutex<Rows>,
    /// What the streams said of their models' block sizes.
    sizes: Mutex<Sizes>,
    /// The URL each open feed's worker is named by in the cache-event
    /// series, by the feed's own row.
    urls: Mutex<HashMap<WorkerId, Arc<str>>>,
    /// Where the ranks of an engine stand, where a host handed its holder.
    ranks: Option<Arc<dyn Ranks>>,
    tables: Arc<KvTables>,
}

/// The rows the adapter's streams write, both ways.
#[derive(Default)]
struct Rows {
    /// What the adapter keeps of each row a stream writes, by the row.
    written: HashMap<WorkerId, Written>,
    /// The rows of other ranks each stream writes, by the stream's row:
    /// what its reset, its new connection and its removal reach beside its
    /// own. A row stands in the list of the stream its entry names, and in
    /// no other.
    fed: HashMap<WorkerId, Vec<WorkerId>>,
}

impl Rows {
    /// Write `stream` as `row`'s stream, from nothing shown: the row leaves
    /// the list of the stream that wrote it. The row's new state comes
    /// back. `None`, with nothing done, where that stream is leaving and
    /// still clears the row.
    fn claim(&mut self, stream: WorkerId, row: WorkerId) -> Option<Arc<Mutex<Applied>>> {
        if self.written.get(&row).is_some_and(|before| before.leaving) {
            return None;
        }
        let begun = Written::begun(stream);
        let applied = Arc::clone(&begun.applied);
        let before = self.written.insert(row, begun);
        if let Some(list) = before.and_then(|before| self.fed.get_mut(&before.stream)) {
            list.retain(|&other| other != row);
        }
        if row != stream {
            self.fed.entry(stream).or_default().push(row);
        }
        Some(applied)
    }
}

/// What the adapter keeps of a row a stream writes.
struct Written {
    /// What the row's batches left under the apply rules, behind its own
    /// lock: a batch holds it inside the row's lane, with the table's lock
    /// let go.
    applied: Arc<Mutex<Applied>>,
    /// The stream that writes the row, named by its own row: the row
    /// itself for a feed's row, its engine's rank 0 row for the row of
    /// another rank.
    stream: WorkerId,
    /// The stream is leaving, and its removal has not cleared the row in
    /// the tables yet: no stream takes the row until it has.
    leaving: bool,
}

impl Written {
    /// What is kept of a row `stream` begins from nothing.
    fn begun(stream: WorkerId) -> Self {
        Self {
            applied: Arc::default(),
            stream,
            leaving: false,
        }
    }
}

/// A row's state under the worker crate's apply rules.
#[derive(Default)]
struct Applied {
    /// The rules' own state of the row. Its map is the lane's, here for a
    /// batch alone ([`Lending`]).
    state: WorkerIndexState<Lent>,
    /// The value the row's lane said when the state was last applied
    /// ([`LaneWriter::emptied`]), and none until the row's first batch. A
    /// batch that reads another resets the state first: the tables emptied
    /// the rank in between.
    emptied: Option<u64>,
    /// The blocks the row's map held behind its last batch.
    blocks: usize,
}

/// The row's reverse map inside the rules' state. The map is the lane's,
/// since the tables empty a rank by it. So this holds it for a batch
/// alone and nothing between batches, where no read answers for the row.
#[derive(Default)]
struct Lent(Option<ChainBlockMap>);

impl Lent {
    /// The map, inside a batch. The rules read it there alone: a read at
    /// rest is the adapter's fault, and a debug build says so.
    fn lent(&self) -> Option<&ChainBlockMap> {
        debug_assert!(
            self.0.is_some(),
            "the row's map is the lane's between batches"
        );
        self.0.as_ref()
    }
}

impl HeldBlocks for Lent {
    fn contains_key(&self, hash: SequenceHash) -> bool {
        self.lent().is_some_and(|map| map.contains_key(hash))
    }

    fn len(&self) -> usize {
        self.lent().map_or(0, ChainBlockMap::len)
    }

    fn hashes(&self) -> Vec<SequenceHash> {
        self.lent()
            .map(|map| map.iter().map(|(hash, _)| hash).collect())
            .unwrap_or_default()
    }
}

/// A lent lane as the apply rules' index. The rules name the row's id in
/// the index, and the lane writes under that id by itself.
struct Lane<'a, 'w>(&'a LaneWriter<'w>);

impl BlockIndex for Lane<'_, '_> {
    type Held = Lent;

    fn apply_stored(
        &self,
        _: u32,
        blocks: &[StoredBlock],
        parent: Option<SequenceHash>,
        held: &mut Lent,
    ) -> Result<(), ApplyError> {
        match &mut held.0 {
            Some(map) => self.0.stored(blocks, parent, map),
            None => Err(ApplyError::WorkerNotTracked),
        }
    }

    fn apply_removed(&self, _: u32, hashes: &[SequenceHash], held: &mut Lent) {
        if let Some(map) = &mut held.0 {
            self.0.removed(hashes, map);
        }
    }

    fn apply_cleared(&self, _: u32, held: &mut Lent) {
        if let Some(map) = &mut held.0 {
            self.0.cleared(map);
        }
    }

    fn groups(&self, _: u32) -> Arc<WorkerGroups> {
        self.0.groups()
    }
}

/// The lane's map inside a row's state for one batch. Dropped, it puts the
/// map back, also behind rules that unwound: the tables then empty the
/// rank by a map that names every block the row wrote.
struct Lending<'a, 'w> {
    lane: &'a mut LaneWriter<'w>,
    state: &'a mut WorkerIndexState<Lent>,
}

impl<'a, 'w> Lending<'a, 'w> {
    fn new(lane: &'a mut LaneWriter<'w>, state: &'a mut WorkerIndexState<Lent>) -> Self {
        state.blocks = Lent(Some(mem::take(lane.map())));
        Self { lane, state }
    }
}

impl Drop for Lending<'_, '_> {
    fn drop(&mut self) {
        if let Some(map) = self.state.blocks.0.take() {
            *self.lane.map() = map;
        }
    }
}

/// Where the ranks of an engine stand: the holder's answer, as the adapter
/// reads it to route a batch.
trait Ranks: Send + Sync {
    /// The row of rank `rank` of the engine `row` is a rank of.
    fn row_of(&self, row: WorkerId, rank: u16) -> Option<WorkerId>;
}

impl<G: Generation> Ranks for Objects<G> {
    fn row_of(&self, row: WorkerId, rank: u16) -> Option<WorkerId> {
        self.rank_row(row, rank)
    }
}

/// What the adapter keeps to name a model's block size from its streams.
#[derive(Default)]
struct Sizes {
    /// The model each open row streams for.
    models: HashMap<WorkerId, Arc<str>>,
    /// The widest block a model's stored blocks named: the engine's word,
    /// which only rises.
    named: HashMap<Arc<str>, u32>,
}

impl KvAdapter {
    /// An adapter that writes `tables`, every batch to its stream's row. It
    /// has no holder to state its rows: the cases that build one state
    /// theirs to the tables themselves.
    #[cfg(test)]
    fn new(tables: Arc<KvTables>) -> Self {
        Self::over(tables, None)
    }

    fn over(tables: Arc<KvTables>, ranks: Option<Arc<dyn Ranks>>) -> Self {
        Self {
            rows: Mutex::new(Rows::default()),
            sizes: Mutex::new(Sizes::default()),
            urls: Mutex::new(HashMap::new()),
            ranks,
            tables,
        }
    }

    /// The row a batch of `rank` on `id`'s stream is written to. Rank 0 is
    /// the stream's own row, which alone opens a feed. Another rank is the
    /// holder's answer: none while it has laid no row of that rank.
    fn routed(&self, id: WorkerId, rank: u16) -> Option<WorkerId> {
        match &self.ranks {
            Some(ranks) if rank != 0 => ranks.row_of(id, rank),
            _ => Some(id),
        }
    }

    /// The rows `id`'s stream writes now, read in one hold of the lock:
    /// its own first, unless another stream took it, then the rows of
    /// other ranks.
    fn written_by(&self, id: WorkerId) -> Vec<WorkerId> {
        let rows = self.rows.lock();
        let own = rows.written.get(&id).is_none_or(|own| own.stream == id);
        let fed = rows.fed.get(&id).into_iter().flatten().copied();
        own.then_some(id).into_iter().chain(fed).collect()
    }

    /// Whether the holder stands `row` at rank 0: the row of a feed of its
    /// own, which no other stream's batch is routed to. An adapter with no
    /// holder to ask routes every batch to its stream's own row, so every
    /// row stands there.
    fn stands_lowest(&self, row: WorkerId) -> bool {
        self.ranks
            .as_ref()
            .is_none_or(|ranks| ranks.row_of(row, 0).is_some())
    }

    /// The blocks the rows `id`'s stream writes held behind their last
    /// batches: what its worker's index-blocks gauge says.
    fn held(&self, id: WorkerId) -> usize {
        let states: Vec<_> = {
            let rows = self.rows.lock();
            let own = rows.written.get(&id).filter(|own| own.stream == id);
            let fed = rows.fed.get(&id).into_iter().flatten();
            let fed = fed.filter_map(|row| rows.written.get(row));
            own.into_iter()
                .chain(fed)
                .map(|written| Arc::clone(&written.applied))
                .collect()
        };
        // Each row's state by itself, outside the table's lock.
        states.iter().map(|state| state.lock().blocks).sum()
    }

    /// The state of `row` for a batch on `id`'s stream, and whether the
    /// table keeps it: the stream's own row that no entry is kept of gets a
    /// state the table takes once a batch was admitted ([`Self::keep`]).
    /// `None` where the stream writes no row any longer: another stream
    /// took its own row, so it is ending, and it writes no row of any
    /// rank. A row of another rank that the stream writes for the first
    /// time, or that another stream wrote, is taken: `None` where it
    /// cannot be yet.
    fn applied(&self, id: WorkerId, row: WorkerId) -> Option<(Arc<Mutex<Applied>>, bool)> {
        let known = {
            let rows = self.rows.lock();
            if rows.written.get(&id).is_some_and(|own| own.stream != id) {
                return None;
            }
            rows.written
                .get(&row)
                .map(|written| (written.stream == id).then(|| Arc::clone(&written.applied)))
        };
        match known {
            Some(Some(applied)) => Some((applied, true)),
            // The stream's own row that no entry is kept of.
            None if row == id => Some((Arc::default(), false)),
            _ => self.take(id, row).map(|applied| (applied, true)),
        }
    }

    /// `stream` takes `row`, the row of another rank, for a batch. The row
    /// begins from nothing, as a feed's own row does at its open: what the
    /// tables kept of it was decided by facts that are gone, or by another
    /// stream's batches. The stream that wrote it reaches it no longer:
    /// the row changes lists, and its lane is cleared, in one hold of the
    /// lock. The clear is of one row, and of nothing where the tables never
    /// held it. The row's new state comes back. `None`, with nothing done,
    /// where the row is not this stream's to take, and the batch is passed
    /// over. Its stream is leaving and still clears it: a later batch
    /// takes the row. Or it is an open feed's own row that the holder
    /// still stands at rank 0: only a route read before the row was laid
    /// again asks for it.
    fn take(&self, stream: WorkerId, row: WorkerId) -> Option<Arc<Mutex<Applied>>> {
        let mut rows = self.rows.lock();
        let own_feed = rows
            .written
            .get(&row)
            .is_some_and(|before| before.stream == row);
        // The holder is asked in the hold, and for such a row alone.
        if own_feed && self.stands_lowest(row) {
            return None;
        }
        let applied = rows.claim(stream, row)?;
        // In the hold that wrote the entry: no stream claims the row, and
        // writes it, ahead of this clear.
        self.tables.removed(row);
        Some(applied)
    }

    /// A feed's own row begins. An error, with nothing done, says why it
    /// does not. [`SkipReason::NoRow`]: `stands` says the feed's object stands
    /// for the row no longer. [`SkipReason::UpperRank`]: the holder stands the
    /// row above rank 0. An object can stand for such a row and state no
    /// rank: it states one where its row's width is above one. Both are asked in
    /// the hold that writes the row's entry: a row laid again behind the
    /// answers is taken from this feed by the stream the holder then
    /// routes to it, and never by this feed from that stream.
    ///
    /// A row the adapter holds no facts for begins from nothing: what the
    /// tables kept of its stream was decided by facts that are gone. The
    /// row's lane starts over at whatever rank it stands, and a row with
    /// none loses nothing. A row another stream wrote, as a rank of
    /// its engine, begins from nothing too. One a leaving stream still
    /// clears is left to it, and this feed's batches are passed over until
    /// that ends.
    fn begin(&self, row: WorkerId, stands: impl FnOnce() -> bool) -> Result<(), SkipReason> {
        let mut rows = self.rows.lock();
        if !stands() {
            return Err(SkipReason::NoRow);
        }
        if !self.stands_lowest(row) {
            return Err(SkipReason::UpperRank);
        }
        let own = rows
            .written
            .get(&row)
            .is_some_and(|written| written.stream == row);
        if !own && rows.claim(row, row).is_some() {
            // In the hold that wrote the entry, as a take's clear is, and
            // on the caller's task: it ends before the feed is handed back.
            self.tables.removed(row);
        }
        Ok(())
    }

    /// Keep the state `id`'s own row was applied under, where the table
    /// kept none of the row as its batch began. An entry made in between is
    /// left: another stream took the row behind the batch's read.
    fn keep(&self, id: WorkerId, applied: Arc<Mutex<Applied>>) {
        let mut rows = self.rows.lock();
        rows.written.entry(id).or_insert(Written {
            applied,
            stream: id,
            leaving: false,
        });
    }

    /// A row `id`'s stream writes holds a block of record of `size` tokens:
    /// what it says of the block size of the model the row streams for, by
    /// the module's rule. A row no feed was opened for names no model.
    fn learn(&self, id: WorkerId, size: u32) {
        // Held through the tables' calls: two rows of one model learn in turn.
        let mut sizes = self.sizes.lock();
        let Sizes { models, named } = &mut *sizes;
        let Some(model) = models.get(&id) else {
            return;
        };
        let last = named.get(model).copied();
        let word = last.map_or(size, |last| last.max(size));
        let of_record = self.tables.block_size(model);
        if of_record != Some(word) {
            // The tables' setter only raises, so a wider size that no stored
            // block named goes first. A request hashed between the two calls
            // reads no size and states no hash.
            if of_record.is_some_and(|of_record| of_record > word) {
                self.tables.forget_block_size(model);
            }
            self.tables.set_block_size(model, word);
            tracing::info!(
                model = %model,
                block_size = word,
                previous = ?of_record,
                "learned a model's block size from a stored block"
            );
        }
        if last != Some(word) {
            named.insert(Arc::clone(model), word);
        }
    }
}

impl KvEventSink for KvAdapter {
    type Batch = KvEventBatch;
    type Id = WorkerId;

    fn admit(
        &self,
        id: WorkerId,
        rank: u16,
        facts: BatchFacts,
        batch: &KvEventBatch,
    ) -> BatchOutcome {
        // A rank whose row the holder has not laid has no place yet.
        let Some(row) = self.routed(id, rank) else {
            return BatchOutcome::Skipped;
        };
        // Another stream took this stream's own row: it writes none now.
        let Some((applied, kept)) = self.applied(id, row) else {
            return BatchOutcome::Skipped;
        };
        // The stream's worker, as its series name it.
        let url = self.urls.lock().get(&id).cloned();
        let snapshot = facts.snapshot.is_some();
        // A batch that carries a store names the model's block size.
        let stores = batch
            .events
            .iter()
            .any(|event| matches!(event.data, Some(Data::Stored(_))));
        let mut record = None;
        let mut took = None;
        // The cursor reads the header alone. The body runs where it admits
        // the batch: inside the lane's turn, with the table's lock let go.
        let outcome = self.tables.admit_through(row, rank, facts, &mut |lane| {
            let mut applied = applied.lock();
            let Applied {
                state,
                emptied,
                blocks,
            } = &mut *applied;
            if *emptied != Some(lane.emptied()) {
                // The tables emptied the rank since the row's last batch, or
                // the row begins: what the rules kept beside the map goes
                // with the blocks.
                state.index_emptied();
                // The view is the index's, under the row's id there. A
                // lane made again holds another: the rules write that one.
                if state.groups.is_some() {
                    state.groups = Some(lane.groups());
                }
            }
            let worker = lane.worker();
            {
                let lending = Lending::new(lane, state);
                took = match &url {
                    // Applied and accounted for under the worker's name.
                    Some(url) => {
                        let (op, disposition) = if snapshot {
                            ("snapshot", "snapshot")
                        } else {
                            ("stored", "applied")
                        };
                        Some(apply_counted(
                            batch,
                            url,
                            worker,
                            &Lane(lending.lane),
                            lending.state,
                            op,
                            disposition,
                        ))
                    }
                    // A row no feed was opened for names no series.
                    None => {
                        apply_events(&batch.events, worker, &Lane(lending.lane), lending.state);
                        None
                    }
                };
                *blocks = lending.state.blocks.len();
            }
            *emptied = Some(lane.emptied());
            record = state.block_of_record();
        });
        match outcome {
            BatchOutcome::Applied => {
                if !kept {
                    self.keep(id, applied);
                }
                if let Some(width) = record.filter(|_| stores) {
                    self.learn(id, width);
                }
                if let Some(url) = &url {
                    // A live batch's duration is the stream's pace. A
                    // snapshot chunk publishes none.
                    if let Some(took) = took.filter(|_| !snapshot) {
                        Metrics::record_kv_event_apply(url, took.as_secs_f64());
                    }
                    Metrics::set_kv_index_blocks(url, self.held(id));
                }
            }
            BatchOutcome::Skipped => {
                // A batch behind its lane's cursor. A row with no lane
                // skips every batch and counts none.
                let behind = || self.tables.cursor(row, rank).is_some();
                if let Some(url) = url.as_ref().filter(|_| behind()) {
                    Metrics::record_kv_event_batch(url, "stale");
                }
            }
            BatchOutcome::Gap { .. } => {
                // The collector asks the publisher to replay from here.
                if let Some(url) = &url {
                    Metrics::record_kv_event_gap(url, "replay_requested", 0);
                }
            }
        }
        outcome
    }

    fn reconnected(&self, id: WorkerId) {
        for row in self.written_by(id) {
            self.tables.reconnected(row);
        }
    }

    fn resync(&self, id: WorkerId, rank: Option<u16>, reason: ResyncReason) {
        if let Some(url) = self.urls.lock().get(&id) {
            Metrics::record_kv_event_resync(url, reason.as_str());
        }
        match rank {
            // One rank's reset is its own row's, and a rank with no row
            // holds nothing to reset.
            Some(rank) => {
                if let Some(row) = self.routed(id, rank) {
                    self.tables.resync(row, Some(rank), reason);
                }
            }
            // The stream's reset reaches every row it writes, its own
            // first.
            None => {
                for row in self.written_by(id) {
                    self.tables.resync(row, None, reason);
                }
            }
        }
    }

    fn cursor(&self, id: WorkerId, rank: u16) -> Option<KvCursor> {
        let row = self.routed(id, rank)?;
        self.tables.cursor(row, rank)
    }

    fn removed(&self, id: WorkerId) {
        self.sizes.lock().models.remove(&id);
        self.urls.lock().remove(&id);
        // What the stream still writes goes with it, since no stream feeds
        // it now: the rows of other ranks, and its own unless another
        // stream took it. Each is marked in the one hold that reads the
        // list, cleared in the tables outside the lock, and forgotten. A
        // stream that would take one in between passes its batch over, so
        // nothing a new stream wrote is cleared here. These clears take
        // every block the stream stored: no batch waits for them.
        let leaving = {
            let mut rows = self.rows.lock();
            let Rows { written, fed } = &mut *rows;
            let own = written.get(&id).is_none_or(|own| own.stream == id);
            let mut leaving = fed.remove(&id).unwrap_or_default();
            if own {
                leaving.push(id);
            }
            for &row in &leaving {
                let mark = Written {
                    leaving: true,
                    ..Written::begun(id)
                };
                written.insert(row, mark);
            }
            leaving
        };
        for &row in &leaving {
            self.tables.removed(row);
        }
        let mut rows = self.rows.lock();
        for row in &leaving {
            rows.written.remove(row);
        }
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.tables.block_size(model)
    }

    fn set_block_size(&self, model: &str, size: u32) {
        self.tables.set_block_size(model, size);
    }
}

/// Why a subscription got no feed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SkipReason {
    /// The holder keeps no row for the worker's object: it laid none, or it
    /// has laid another object in its place.
    NoRow,
    /// The worker's row is a rank above 0 of a data-parallel worker.
    UpperRank,
}

/// How many subscriptions a host gave no feed, by the reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SkippedSubscriptions {
    /// The holder kept no row for the worker's object.
    pub no_row: u64,
    /// The worker's row was a rank above 0.
    pub upper_rank: u64,
}

/// The KV collector's host over the tables an events subscription feeds.
pub struct KvHost<G> {
    objects: Arc<Objects<G>>,
    sink: Arc<KvAdapter>,
    no_row: AtomicU64,
    upper_rank: AtomicU64,
}

impl<G: Generation> KvHost<G> {
    /// A host whose subscriptions feed the tables `objects` was handed
    /// ([`Objects::with_kv_tables`]), each under the row `objects` holds
    /// for its worker. A batch of another rank of the worker's engine is
    /// written to the row `objects` holds for that rank.
    ///
    /// The holder states the tables once, so the lanes it makes and the
    /// events written here meet in one set. Over a holder that was handed
    /// none, no row has a lane: every batch is skipped and nothing is
    /// indexed.
    pub fn new(objects: Arc<Objects<G>>) -> Self {
        let ranks: Arc<dyn Ranks> = Arc::clone(&objects) as _;
        let tables = objects.kv_tables().unwrap_or_default();
        Self {
            objects,
            sink: Arc::new(KvAdapter::over(tables, Some(ranks))),
            no_row: AtomicU64::new(0),
            upper_rank: AtomicU64::new(0),
        }
    }

    /// The row `worker`'s object stands for: none where the holder keeps no
    /// row for it, or has laid another object in its place.
    fn row_of(&self, worker: &Arc<dyn Worker>) -> Option<WorkerId> {
        let id = self.objects.registry().get_id_by_url(worker.url())?;
        self.objects
            .standing_of(&id, worker)
            .map(|standing| standing.row)
    }
}

impl<G> KvHost<G> {
    /// How many subscriptions got no feed so far, by the reason.
    pub fn skipped_subscriptions(&self) -> SkippedSubscriptions {
        SkippedSubscriptions {
            no_row: self.no_row.load(Ordering::Relaxed),
            upper_rank: self.upper_rank.load(Ordering::Relaxed),
        }
    }
}

impl<G> KvEventHost for KvHost<G>
where
    G: Generation,
{
    type Sink = KvAdapter;
    type Ticket = Result<WorkerId, SkipReason>;

    fn enroll(&self, worker: &Arc<dyn Worker>, model: &str) -> Self::Ticket {
        // A size the tables' width cannot hold is not a size of record.
        if let Some(size) = worker
            .metadata()
            .spec
            .kv_block_size
            .and_then(|size| u32::try_from(size).ok())
            .filter(|&size| size > 0)
        {
            self.sink.set_block_size(model, size);
        }
        // A stream is its engine's and resumes by rank 0: a row above it
        // opens no feed, and its engine's rank 0 stream feeds it.
        if worker.dp_rank().is_some_and(|rank| rank > 0) {
            return Err(SkipReason::UpperRank);
        }
        // The row this very object stands for: one the holder has laid
        // another object in the place of has none.
        self.row_of(worker).ok_or(SkipReason::NoRow)
    }

    /// The feed names the worker's row. A subscription with no feed is
    /// counted and said here, by its reason.
    fn open(
        &self,
        ticket: Self::Ticket,
        worker: &Arc<dyn Worker>,
        model: &str,
    ) -> Option<KvEventFeed<Self::Sink>> {
        // The ticket was written at the enrollment, and the feed opens
        // later, on the subscription's task. A row laid again since is
        // another object's, and may be another stream's by now: its ticket
        // opens nothing. Nor does a ticket for a row the holder stands
        // above rank 0. The holder is asked both as the row begins, and
        // the answer names the reason.
        let ticket = ticket.and_then(|row| {
            self.sink
                .begin(row, || self.row_of(worker) == Some(row))
                .map(|()| row)
        });
        match ticket {
            Ok(row) => {
                // The model the row's stored blocks name a size for.
                self.sink.sizes.lock().models.insert(row, Arc::from(model));
                // The name the stream's series go under.
                let url = Arc::from(worker.url());
                self.sink.urls.lock().insert(row, url);
                Some((Arc::clone(&self.sink), row))
            }
            Err(SkipReason::NoRow) => {
                self.no_row.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    worker_url = %worker.redacted_url(),
                    "no KV event feed: the holder keeps no row for the worker's object"
                );
                None
            }
            Err(SkipReason::UpperRank) => {
                self.upper_rank.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(
                    worker_url = %worker.redacted_url(),
                    "no KV event feed for a rank above 0: its engine's rank 0 stream feeds it"
                );
                None
            }
        }
    }

    fn model_left(&self, model: &str) {
        self.sink.sizes.lock().named.remove(model);
        self.sink.tables.forget_block_size(model);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/worker_host/kv/tests.rs"]
mod tests;
